"""Regression tests for the review of the gap fixes (sim): stimulus kept out of the profile text (review #1), empty /
failed batches do not abort a survey (#3), shared refusal + empty handling in the other runners (#11), numeric
parsing (#9), free-text line only with free-text fields (#12). Offline, FakeLLM-based."""
import json
import re
import unittest
import warnings

import pandas as pd

from simengine.llm import EmptyResponseError, FakeLLM, LLMError
from simengine.sim import (REFRAME_NOTE, Usage, ab_many, ask_many, build_prompts, as_personas, persona_distribution,
                           predict_people, schema_responder, segment_distribution, survey)
from simengine.sim._stats import to_number
from simengine.sim.panel import REFRAME_NOTE_GENERIC
from simengine.types import LLMResponse
from tests.test_sim_survey_gaps import REFUSAL, answer_all, panel


class SlotLLM(FakeLLM):
    """FakeLLM whose responder may raise; map() honours return_exceptions like the real backends and records the
    keyword arguments and max_tokens of every call."""

    def __init__(self, responder):
        super().__init__(responder)
        self.kws, self.max_tokens = [], []

    def complete(self, system, user, *, max_tokens=256, **kw):
        self.calls.append((system, user))
        self.max_tokens.append(max_tokens)
        text = self.responder(system, user, max_tokens)
        return LLMResponse(text=text, model=self.model, input_tokens=1, output_tokens=1)

    def map(self, prompts, *, max_tokens=256, parallel=4, return_exceptions=False, **kw):
        self.kws.append(dict(kw, return_exceptions=return_exceptions))
        out = []
        for s, u in prompts:
            try:
                out.append(self.complete(s, u, max_tokens=max_tokens, **kw))
            except Exception as e:  # noqa: BLE001
                if not return_exceptions:
                    raise
                out.append(e)
        return out


def empty_error():
    return EmptyResponseError("claude-haiku-5-5: empty reply (stop_reason=max_tokens, 300 output tokens billed)")


# ---------------------------------------------------------------- review #1
class TestStimulusNotInProfile(unittest.TestCase):
    def _personas(self):
        from simengine.persona import from_table
        df = pd.DataFrame({"uid": [f"u{i}" for i in range(6)], "age": [25, 35, 45, 55, 65, 30],
                           "msg": ["MSG_A", "MSG_B"] * 3, "y": [1, 2, 3, 4, 5, 1]})
        return from_table(df, id_col="uid", forbidden=["y"])

    def test_each_prompt_shows_its_stimulus_once_in_every_arm(self):
        llm = FakeLLM(schema_responder())
        q = [{"id": "q1", "text": "Support?", "kind": "numeric", "min": 0, "max": 100, "free_text": False}]
        res = survey(self._personas(), q, llm=llm, stimulus_col="msg", batch=3)
        self.assertEqual(res["n_answered"], 6)
        self.assertEqual(len(llm.calls), 6)          # persona, no_persona, persona_swap x 2 stimuli
        for _, user in llm.calls:
            shown = re.search(r"Context shown to everyone:\n(MSG_[AB])", user).group(1)
            other = "MSG_B" if shown == "MSG_A" else "MSG_A"
            self.assertEqual(user.count(shown), 1, user)
            self.assertNotIn(other, user)            # the swap arm no longer carries the next persona's stimulus
            self.assertNotIn("msg=", user)
        self.assertEqual(res["stimuli"]["u1"], "MSG_B")

    def test_talp_dict_facet_and_flat_dict(self):
        talp = [{"identity": {"persona_id": f"t{i}", "age_band": "30-39"}, "segment": "s",
                 "evidence": {"measured_behavior": {"facets": {"income": "mid", "ad": f"AD{i % 2}"}}}}
                for i in range(4)]
        flat = [{"id": f"f{i}", "city": "X", "ad": f"AD{i % 2}"} for i in range(4)]
        for ps in (talp, flat):
            llm = FakeLLM(lambda s, u: answer_all(u))
            survey(ps, "Q?", llm=llm, stimulus_col="ad", controls=[])
            for _, user in llm.calls:
                self.assertEqual(len(re.findall(r"AD\d", user)), 1, user)
            self.assertTrue(any("income=mid" in u or "city: X" in u for _, u in llm.calls))


# ---------------------------------------------------------------- review #3
class TestEmptyBatchDoesNotAbort(unittest.TestCase):
    def test_one_empty_batch_counts_as_unparsed(self):
        def responder(system, user, max_tokens):
            if "### p0" in user:
                raise empty_error()                  # also on the larger-max_tokens re-ask
            return answer_all(user)
        llm = SlotLLM(responder)
        res = survey(panel(4), "Q?", llm=llm, batch=2, controls=[])
        self.assertEqual((res["n_answered"], res["unparsed"], res["refused"]), (2, 2, 0))
        self.assertEqual(res["usage"]["empty_calls"], 2)       # first ask + re-ask
        self.assertIn("EmptyResponseError", res["usage"]["last_error"])
        self.assertTrue(any("empty replies" in w and "thinking='off'" in w for w in res["warnings"]))
        self.assertEqual(sorted(set(llm.max_tokens))[-1], 4000)  # re-ask with a larger budget

    def test_empty_reply_recovered_with_larger_budget(self):
        def responder(system, user, max_tokens):
            if max_tokens < 4000:
                raise empty_error()
            return answer_all(user)
        res = survey(panel(4), "Q?", llm=SlotLLM(responder), batch=2, controls=[])
        self.assertEqual(res["n_answered"], 4)
        self.assertEqual((res["usage"]["empty_calls"], res["usage"]["empty_recovered"]), (2, 2))

    def test_empty_retry_kw_and_off(self):
        seen = []

        def responder(system, user, max_tokens):
            raise empty_error()
        llm = SlotLLM(responder)
        with self.assertRaises(EmptyResponseError):      # nobody answered: a real failure is still loud
            survey(panel(2), "Q?", llm=llm, controls=[], empty_retry_kw={"thinking": "off"})
        seen = [kw.get("thinking") for kw in llm.kws]
        self.assertEqual(seen, [None, "off"])
        llm2 = SlotLLM(responder)
        with self.assertRaises(EmptyResponseError):
            survey(panel(2), "Q?", llm=llm2, controls=[], empty_retry=False)
        self.assertEqual(len(llm2.calls), 1)

    def test_other_llm_error_in_one_batch(self):
        def responder(system, user, max_tokens):
            if "### p2" in user:
                raise LLMError("HTTP 500 after retries")
            return answer_all(user)
        res = survey(panel(4), "Q?", llm=SlotLLM(responder), batch=2, controls=[])
        self.assertEqual((res["n_answered"], res["unparsed"]), (2, 2))
        self.assertEqual(res["usage"]["error_calls"], 1)

    def test_programming_errors_still_raise(self):
        def responder(system, user, max_tokens):
            raise KeyError("bug")
        with self.assertRaises(KeyError):
            survey(panel(2), "Q?", llm=SlotLLM(responder), controls=[])

    def test_ask_many_statuses(self):
        def responder(system, user, max_tokens):
            if user.endswith("refuse"):
                return "Recovered" if REFRAME_NOTE_GENERIC in user else REFUSAL
            if user.endswith("empty"):
                raise empty_error()
            if user.endswith("err"):
                raise LLMError("boom")
            return '{"a": 1}'
        st, errs, u = [], [], Usage()
        rs = ask_many(SlotLLM(responder), [("", "ok"), ("", "refuse"), ("", "empty"), ("", "err")], max_tokens=100,
                      usage=u, reframe=REFRAME_NOTE_GENERIC, statuses=st, errors=errs)
        self.assertEqual(st, ["ok", "recovered", "empty", "error"])
        self.assertEqual([r is None for r in rs], [False, False, True, True])
        self.assertEqual((u.refused_calls, u.recovered_calls, u.empty_calls, u.error_calls, len(errs)), (1, 1, 2, 1, 3))


# ---------------------------------------------------------------- review #11
class TestOtherRunnersShareHandling(unittest.TestCase):
    SEGS = {"young": "adults 18-29", "old": "adults 65+"}
    Q = [{"id": "q1", "text": "Pick one", "options": ["yes", "no"]}]

    def test_segment_distribution_refusal_reasked_and_not_cached(self):
        def responder(system, user, max_tokens):
            if REFRAME_NOTE_GENERIC in user:
                return '{"1": 70, "2": 30}'
            return "I'm not comfortable generating fake survey responses for these groups."
        llm = SlotLLM(responder)
        res = segment_distribution(llm, self.SEGS, self.Q, framings={"v": "{group}"})
        self.assertEqual(res["invalid"], 0)
        self.assertAlmostEqual(res["dist"]["young"]["q1"][0], 0.7)
        self.assertEqual(res["usage"]["refused_calls"], 2)
        self.assertIn("cache_if", llm.kws[0])
        self.assertFalse(llm.kws[0]["cache_if"]("I can't produce this as a survey dataset."))

    def test_segment_distribution_empty_cell_is_invalid_not_abort(self):
        def responder(system, user, max_tokens):
            if "65+" in user:
                raise empty_error()
            return '{"1": 50, "2": 50}'
        res = segment_distribution(SlotLLM(responder), self.SEGS, self.Q, framings={"v": "{group}"})
        self.assertEqual(res["invalid"], 1)
        self.assertEqual(res["dist"]["old"]["q1"], [0.5, 0.5])    # uniform fallback
        self.assertEqual(res["usage"]["empty_calls"], 2)

    def test_persona_distribution_predict_and_ab_survive_one_empty_slot(self):
        fake = schema_responder()

        def responder(system, user, max_tokens):
            if "p000:" in user:
                raise empty_error()
            return fake(system, user)
        ps = [{"id": f"p00{i}", "text": f"person {i}", "segment": "A"} for i in range(4)]
        res = persona_distribution(SlotLLM(responder), ps, self.Q, batch=2)
        self.assertEqual(res["invalid"], 2)
        self.assertEqual(res["usage"]["empty_calls"], 2)

        def pred(system, user, max_tokens):
            if "person 0" in user:
                raise empty_error()
            return json.dumps({"buy": 40})
        r = predict_people(SlotLLM(pred), {f"x{i}": f"person {i}" for i in range(3)}, ["buy"], retries=0,
                           sample_kw=None)
        self.assertEqual(sorted(r["p"]), ["x1", "x2"])
        self.assertEqual(r["failed"], ["x0"])
        self.assertEqual(r["usage"]["empty_calls"], 2)

        def votes(system, user, max_tokens):
            if "t1:" in user:
                raise empty_error()
            return json.dumps([{"id": "r1|t0", "a": 5, "b": 3, "choice": "A"}])
        tests = [{"test_id": "t0", "a": "x", "b": "y"}, {"test_id": "t1", "a": "x", "b": "y"}]
        u = Usage()
        df = ab_many([{"id": "r1", "text": "reader"}], tests, llm=SlotLLM(votes), tests_per_call=1, usage=u)
        self.assertEqual(df.set_index("test_id")["n_votes"].to_dict(), {"t0": 1, "t1": 0})
        self.assertEqual(u.empty_calls, 2)


# ---------------------------------------------------------------- review #9 / #12
class TestNumericParsingAndPrompt(unittest.TestCase):
    def test_exponent(self):
        self.assertEqual(to_number("1e2"), 100.0)
        self.assertEqual(to_number("2.5E-1"), 0.25)
        self.assertEqual(to_number("7/10"), 7.0)
        self.assertEqual(to_number("70%"), 70.0)

    def test_fraction_answers_warned(self):
        def responder(system, user):
            ids = re.findall(r"^### (\S+)\s*$", user, re.M)
            return json.dumps([{"id": p, "q1_value": 0.75} for p in ids])
        q = [{"id": "q1", "text": "Support 0-100?", "kind": "numeric", "min": 0, "max": 100, "free_text": False}]
        res = survey(panel(4), q, llm=FakeLLM(responder), controls=[])
        self.assertTrue(any("fractions" in w for w in res["warnings"]))
        q01 = [{"id": "q1", "text": "Share?", "kind": "numeric", "min": 0, "max": 1, "free_text": False}]
        res = survey(panel(4), q01, llm=FakeLLM(responder), controls=[])
        self.assertFalse(any("fractions" in w for w in res["warnings"]))

    def test_free_text_line_only_with_text_fields(self):
        ps = as_personas(panel(2))
        (_, u1), _ = build_prompts(ps, "Task", {"q1_value": "number 0-100"})[0]
        (_, u2), _ = build_prompts(ps, "Task", {"q1_reason": "str: one sentence", "q1_value": "number 0-100"})[0]
        self.assertNotIn("Free-text fields", u1)
        self.assertIn("Answer independently for each person. JSON array only.", u1)
        self.assertIn("Free-text fields in English.", u2)


if __name__ == "__main__":
    unittest.main()
