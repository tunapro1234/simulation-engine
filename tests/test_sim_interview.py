"""sim.interview: open-ended single question panel with control arms (G6 rehearsal #3)."""
import json
import re
import unittest

import pandas as pd

from simengine.llm import EmptyResponseError, FakeLLM, LLMError
from simengine.sim import REFRAME_NOTE, schema_responder
from simengine.sim.interview import interview, interview_frame

REFUSAL = ("I can't produce this as a survey dataset. I'd be inventing answers for people who don't exist, "
           "which could be mistaken for real data.")


def panel(n=6, **extra):
    return [{"id": f"p{i}", "text": f"person {i}, age {20 + i}", "segment": "A" if i % 2 else "B",
             **{k: (v[i] if isinstance(v, list) else v) for k, v in extra.items()}} for i in range(n)]


def ids_in(user):
    return re.findall(r"^### (\S+)\s*$", user, re.M)


def answer(user, text=lambda pid: f"answer of {pid}"):
    return json.dumps([{"id": pid, "answer": text(pid)} for pid in ids_in(user)])


class SlotLLM(FakeLLM):
    """FakeLLM whose responder may raise; map() returns exceptions in their slot (return_exceptions)."""

    def __init__(self, responder):
        super().__init__(None)
        self.fn = responder
        self.tags = []

    def map(self, prompts, *, max_tokens=256, parallel=4, return_exceptions=False, **kw):
        self.tags.append(kw.get("tag"))
        out = []
        for s, u in prompts:
            self.calls.append((s, u))
            try:
                out.append(self._resp(self.fn(s, u)))
            except Exception as e:  # noqa: BLE001
                if not return_exceptions:
                    raise
                out.append(e)
        return out

    def _resp(self, text):
        from simengine.types import LLMResponse
        return LLMResponse(text=text, model="fake")


class TestInterview(unittest.TestCase):
    def test_default_arms_and_layout(self):
        llm = FakeLLM(schema_responder())
        res = interview(panel(6), "Why did you stop ordering?", llm=llm, batch=3)
        self.assertEqual(res["runner"], "interview")
        self.assertEqual(res["main_arm"], "persona")
        self.assertEqual(res["controls"]["arms"], ["no_persona", "persona_swap"])
        self.assertEqual(set(res["answers"]), {f"p{i}" for i in range(6)})
        self.assertTrue(all(t.startswith("(fake) answer") for t in res["answers"].values()))
        for arm in ("no_persona", "persona_swap"):
            self.assertEqual(len(res["controls"]["answers"][arm]), res["controls"]["plan"][arm]["n"])
        self.assertEqual((res["refused"], res["empty"], res["error"]), (0, 0, 0))
        self.assertEqual(set(res["arm_stats"]), {"persona", "no_persona", "persona_swap"})
        json.dumps(res)
        _, user = llm.calls[0]
        self.assertIn("Question:\nWhy did you stop ordering?", user)
        self.assertIn("at most 600 characters", user)

    def test_control_prompts_follow_control_personas(self):
        llm = SlotLLM(lambda s, u: answer(u))
        interview(panel(4), "Why?", llm=llm, batch=4)
        users = [u for _, u in llm.calls]
        self.assertEqual(len(users), 3)
        self.assertIn("person 0, age 20", users[0])
        self.assertIn("A typical adult consumer.", users[1])
        self.assertNotIn("person 0", users[1])
        # swap: p0 keeps its id and takes p1's content
        self.assertIn("### p0\np0\nperson 1, age 21", users[2])

    def test_ledger_tags(self):
        llm = SlotLLM(lambda s, u: answer(u))
        interview(panel(4), "Why?", llm=llm)
        self.assertEqual(llm.tags, ["interview:persona", "interview:no_persona", "interview:persona_swap"])
        llm = SlotLLM(lambda s, u: answer(u))
        interview(panel(4), "Why?", llm=llm, tag="dev", arms=["persona"])     # a label, as in survey
        self.assertEqual(llm.tags, ["dev:interview:persona"])

    def test_arm_subsets_and_aliases(self):
        llm = SlotLLM(lambda s, u: answer(u))
        res = interview(panel(4), "Why?", llm=llm, arms=("swap", "persona"))
        self.assertEqual(list(res["arm_stats"]), ["persona", "persona_swap"])
        res = interview(panel(4), "Why?", llm=SlotLLM(lambda s, u: answer(u)), arms=["personasiz"])
        self.assertEqual(res["main_arm"], "no_persona")
        self.assertEqual(len(res["answers"]), 4)
        with self.assertRaises(ValueError):
            interview(panel(2), "Why?", llm=llm, arms=["bogus"])
        with self.assertRaises(ValueError):
            interview(panel(2), "Why?", llm=llm, arms=[])
        with self.assertRaises(ValueError):
            interview(panel(2), "  ", llm=llm)

    def test_single_persona_skips_swap(self):
        res = interview(panel(1), "Why?", llm=SlotLLM(lambda s, u: answer(u)))
        self.assertIn("skipped", res["arm_stats"]["persona_swap"])
        self.assertEqual(res["controls"]["answers"]["persona_swap"], {})
        self.assertTrue(any("persona_swap skipped" in w for w in res["warnings"]))

    def test_refused_empty_error_buckets(self):
        def fn(s, u):
            ids = ids_in(u)
            if "p0" in ids:
                return REFUSAL                                    # persists after the reframe
            if "p2" in ids:
                raise LLMError("HTTP 500 after retries")
            if "p4" in ids:   # p4 blank, p5 missing from the reply
                return json.dumps([{"id": "p4", "answer": "  "}])
            return answer(u)
        res = interview(panel(6), "Why?", llm=SlotLLM(fn), batch=2, arms=["persona"])
        self.assertEqual((res["refused"], res["error"], res["empty"]), (2, 2, 2))
        self.assertEqual(res["refused_ids"], ["p0", "p1"])
        self.assertEqual(res["error_ids"], ["p2", "p3"])
        self.assertEqual(res["empty_ids"], ["p4", "p5"])
        self.assertEqual(res["n_answered"], 0)
        st = res["arm_stats"]["persona"]
        self.assertEqual((st["calls"], st["refused_calls"], st["recovered_calls"]), (3, 1, 0))
        self.assertTrue(any("without an answer" in w for w in res["warnings"]))

    def test_model_refusal_inside_answer_is_refused(self):
        def fn(s, u):
            return answer(u, lambda pid: "As an AI, I can't speak for this person." if pid == "p0"
                          else "Bunu söylemek istemiyorum." if pid == "p1" else f"answer of {pid}")
        res = interview(panel(3), "Why?", llm=SlotLLM(fn), arms=["persona"])
        self.assertEqual(set(res["answers"]), {"p1", "p2"})       # the persona's own "rather not say" stays
        self.assertEqual((res["refused_ids"], res["refused_in_answer"]), (["p0"], 1))
        self.assertIn("p0", res["arm_stats"]["persona"]["refused_answers"])
        self.assertTrue(any("inside the answer" in w for w in res["warnings"]))
        self.assertEqual(interview_frame(res).set_index("unit").loc["p0|persona", "status"], "refused")
        res = interview(panel(3), "Why?", llm=SlotLLM(fn), arms=["persona"], answer_refusal=False)
        self.assertEqual((res["n_answered"], res["refused"]), (3, 0))

    def test_refusal_recovered_by_reframe(self):
        def fn(s, u):
            return answer(u) if REFRAME_NOTE in u else REFUSAL
        res = interview(panel(3), "Why?", llm=SlotLLM(fn), arms=["persona"])
        self.assertEqual((res["n_answered"], res["refused"]), (3, 0))
        self.assertEqual(res["arm_stats"]["persona"]["recovered_calls"], 1)

    def test_all_errors_raise(self):
        def fn(s, u):
            raise EmptyResponseError("empty reply")
        with self.assertRaises(EmptyResponseError):
            interview(panel(2), "Why?", llm=SlotLLM(fn), empty_retry=False)

    def test_truncation_and_non_text(self):
        def fn(s, u):
            return json.dumps([{"id": "p0", "answer": "x" * 50}, {"id": "p1", "answer": ["a", "b"]},
                               {"id": "p2", "answer": {"k": 1}}])
        res = interview(panel(3), "Why?", llm=SlotLLM(fn), max_chars=10, arms=["persona"])
        self.assertEqual(res["answers"], {"p0": "x" * 10, "p1": "a b"})
        self.assertEqual((res["truncated"], res["empty_ids"]), (1, ["p2"]))
        self.assertTrue(any("max_chars=10" in w for w in res["warnings"]))

    def test_stimulus_col_shown_once_and_dropped_from_profile(self):
        llm = SlotLLM(lambda s, u: answer(u))
        res = interview(panel(4, msg=["MSG_A", "MSG_B"] * 2), "Why?", llm=llm, stimulus_col="msg", batch=4)
        self.assertEqual(res["stimuli"], {"p0": "MSG_A", "p1": "MSG_B", "p2": "MSG_A", "p3": "MSG_B"})
        for _, u in llm.calls:
            shown = re.search(r"Context shown to everyone:\n(MSG_[AB])", u).group(1)
            self.assertEqual(u.count(shown), 1)
            self.assertNotIn("MSG_B" if shown == "MSG_A" else "MSG_A", u)
        self.assertEqual(len(llm.calls), 6)                     # 3 arms x 2 stimulus groups

    def test_stimuli_dict_inline(self):
        llm = SlotLLM(lambda s, u: answer(u))
        res = interview(panel(2), "Why?", llm=llm, context="Shared ad.", stimuli={"p0": "S0", "p1": "S1"},
                        stimulus_batching="inline", arms=["persona"])
        _, u = llm.calls[0]
        self.assertIn("Shared ad.", u)
        self.assertIn("Shown to this person: S0", u)
        self.assertEqual(res["n_answered"], 2)
        with self.assertRaises(ValueError):
            interview(panel(2), "Why?", llm=llm, stimuli={"p0": "S0"}, stimulus_col="msg")

    def test_on_event(self):
        ev = []
        interview(panel(2), "Why?", llm=SlotLLM(lambda s, u: answer(u)), on_event=ev.append)
        kinds = [e["type"] for e in ev]
        self.assertEqual(kinds.count("stage"), 3)
        self.assertEqual(kinds[-1], "done")
        self.assertEqual({e["arm"] for e in ev if e["type"] == "row"}, {"persona", "no_persona", "persona_swap"})


class TestInterviewFrame(unittest.TestCase):
    def test_one_row_per_person_and_arm(self):
        def fn(s, u):
            return REFUSAL if "### p0\np0\nperson 0," in u else answer(u)
        res = interview(panel(3), "Why?", llm=SlotLLM(fn), batch=1)
        f = interview_frame(res, truth={"p1": "price"})
        self.assertEqual(len(f), 3 * 3)
        self.assertEqual(f["unit"].iloc[0], "p0|persona")
        r = f.set_index("unit")
        self.assertEqual(r.loc["p0|persona", "status"], "refused")
        self.assertIsNone(r.loc["p0|persona", "pred_text"])
        self.assertEqual(r.loc["p0|no_persona", "status"], "answered")
        self.assertEqual(r.loc["p1|persona", "truth"], "price")
        self.assertEqual(r.loc["p1|persona", "chars"], len("answer of p1"))
        self.assertEqual(set(f["item"]), {"persona", "no_persona", "persona_swap"})
        self.assertNotIn("stimulus", f.columns)

    def test_missing_answer_is_none_under_string_inference(self):
        # pandas 3 (and 2.x with future.infer_string) gives text columns a str dtype that turns None into NaN
        res = interview(panel(1), "Why?", llm=SlotLLM(lambda s, u: answer(u)))
        with pd.option_context("future.infer_string", True):
            f = interview_frame(res).set_index("unit")
        self.assertIsNone(f.loc["p0|persona_swap", "pred_text"])
        self.assertEqual(f.loc["p0|persona", "pred_text"], "answer of p0")
        self.assertEqual(f.loc["p0|persona_swap", "chars"], 0)

    def test_missing_truth_and_stimulus_are_none_under_string_inference(self):
        res = interview(panel(3), "Why?", llm=SlotLLM(lambda s, u: answer(u)))
        res["stimuli"] = {"p0": "ad A", "p1": None, "p2": "ad B"}
        with pd.option_context("future.infer_string", True):
            f = interview_frame(res, truth={"p0": "price"}).set_index("unit")
            g = interview_frame(res, truth={"p0": 1.0}).set_index("unit")
        self.assertIsNone(f.loc["p1|persona", "truth"])
        self.assertEqual(f.loc["p0|persona", "truth"], "price")
        self.assertIsNone(f.loc["p1|persona", "stimulus"])
        self.assertEqual(f.loc["p2|persona", "stimulus"], "ad B")
        # numeric truth keeps the float column (NaN for missing) it has without string inference
        self.assertEqual(g["truth"].dtype, float)
        self.assertTrue(pd.isna(g.loc["p1|persona", "truth"]))

    def test_not_asked_in_subsampled_swap(self):
        res = interview(panel(1), "Why?", llm=SlotLLM(lambda s, u: answer(u)))
        f = interview_frame(res).set_index("unit")
        self.assertEqual(f.loc["p0|persona_swap", "status"], "not_asked")


if __name__ == "__main__":
    unittest.main()
