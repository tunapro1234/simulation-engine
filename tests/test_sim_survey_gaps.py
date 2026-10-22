"""survey / ask_panel: soft refusals (#6), numeric + free_text=False (#7), per-person stimulus (#8), ledger tag (#9)."""
import json
import re
import unittest

from simengine.llm import FakeLLM, LLMError, RefusalError
from simengine.sim import REFRAME_NOTE, Usage, ask_panel, as_personas, looks_like_refusal, schema_responder
from simengine.sim import survey, survey_frame
from simengine.types import LLMResponse

REFUSAL = ("I can't produce this as a survey dataset. I'd be inventing answers for people who don't exist, "
           "which could be mistaken for real data.")


def panel(n=6, **extra):
    return [{"id": f"p{i}", "text": f"person {i}, age {20 + i}", "segment": "A" if i % 2 else "B",
             **{k: (v[i] if isinstance(v, list) else v) for k, v in extra.items()}} for i in range(n)]


def answer_all(user, value=3):
    """Answer every person and every field listed in a panel prompt."""
    ids = re.findall(r"^### (\S+)\s*$", user, re.M)
    block = user.split("with fields:", 1)[1].split("Free-text fields", 1)[0]
    names = [k for k in re.findall(r'^\s+"([^"]+)":', block, re.M) if k != "id"]
    out = []
    for pid in ids:
        d = {"id": pid}
        for k in names:
            d[k] = "b" if k.endswith("_choice") else (value if not k.endswith(("_answer", "_reason", "_en"))
                                                       else f"text of {pid}")
        out.append(d)
    return json.dumps(out)


class RecordingFake(FakeLLM):
    """FakeLLM that records the keyword arguments of every map() call."""

    def __init__(self, responder, tag=""):
        super().__init__(responder)
        self.kws = []
        if tag:
            self.tag = tag

    def map(self, prompts, *, max_tokens=256, parallel=4, **kw):
        self.kws.append(dict(kw))
        return super().map(prompts, max_tokens=max_tokens, parallel=parallel, **kw)


class CacheIfFake(FakeLLM):
    """Mimics the backend's cache_if contract: the reply is always returned, written to `cache` only when
    cache_if(text) is true; a cached prompt is served from `cache`."""

    def __init__(self, responder):
        super().__init__(responder)
        self.cache = {}

    def complete(self, system, user, *, max_tokens=256, **kw):
        key = (system, user)
        if key in self.cache:
            return LLMResponse(text=self.cache[key], model=self.model, cached=True)
        r = super().complete(system, user, max_tokens=max_tokens)
        cache_if = kw.get("cache_if")
        if cache_if is None or cache_if(r.text):
            self.cache[key] = r.text
        return r


class TestRefusals(unittest.TestCase):
    def test_detector(self):
        self.assertTrue(looks_like_refusal(REFUSAL))
        self.assertTrue(looks_like_refusal("I’m not able to simulate these people."))
        self.assertFalse(looks_like_refusal("Sorry, I can't."))
        self.assertFalse(looks_like_refusal('[{"id": "p1", "q1_score": 4}]'))
        self.assertFalse(looks_like_refusal(""))

    def test_local_fallback_when_llm_detector_missing(self):
        import simengine.llm as llm_mod
        saved = llm_mod.__dict__.pop("looks_like_refusal", None)
        try:
            self.assertTrue(looks_like_refusal(REFUSAL))
            self.assertTrue(looks_like_refusal("I\u2019m not able to simulate these people."))
            self.assertTrue(looks_like_refusal("Maalesef bunu yapamam."))
            self.assertFalse(looks_like_refusal("Sorry, I can't."))
            self.assertFalse(looks_like_refusal("I can't afford this at that price."))
            res = survey(panel(2), "Q?", llm=FakeLLM(lambda s, u: answer_all(u) if REFRAME_NOTE in u else REFUSAL),
                         controls=[])
            self.assertEqual((res["n_answered"], res["refusal_rate"]), (2, 1.0))
        finally:
            if saved is not None:
                llm_mod.looks_like_refusal = saved

    def test_free_text_cant_is_not_a_refusal(self):
        llm = FakeLLM(lambda s, u: json.dumps([{"id": "u0", "reason": "I can't afford it", "score": 2}]))
        u = Usage()
        out = ask_panel(llm, as_personas([{"id": "u0", "text": "x"}]), "t", {"score": "int 1-10", "reason": "str"},
                        usage=u)
        self.assertEqual(out["u0"]["score"], 2)
        self.assertEqual((u.refused_calls, u.reframed_calls, len(llm.calls)), (0, 0, 1))

    def test_reframe_recovers_refused_batches(self):
        def responder(system, user):
            return answer_all(user) if REFRAME_NOTE in user else REFUSAL
        llm = RecordingFake(responder)
        res = survey(panel(6), "Do you support it?", llm=llm, batch=3, controls=[])
        self.assertEqual(res["n_answered"], 6)
        self.assertEqual((res["refused"], res["unparsed"]), (0, 0))
        self.assertEqual(res["refusal_rate"], 1.0)
        self.assertEqual(res["refusals"]["persona"]["recovered_calls"], 2)
        self.assertTrue(any("refused at the first ask" in w for w in res["warnings"]))
        self.assertEqual(res["usage"]["reframed_calls"], 2)
        reframed = [s for s, u in llm.calls if REFRAME_NOTE in u]
        self.assertTrue(reframed and all(REFRAME_NOTE in s for s in reframed))

    def test_refusals_counted_apart_from_parse_failures(self):
        def responder(system, user):
            if "### p0" in user:
                return REFUSAL                         # refuses even after the reframe
            if "### p2" in user:
                return "garbage without json"          # parse failure, not a refusal
            return answer_all(user)
        res = survey(panel(6), "Do you support it?", llm=FakeLLM(responder), batch=2, controls=[])
        self.assertEqual(res["n_answered"], 2)
        self.assertEqual((res["refused"], res["unparsed"]), (2, 2))
        self.assertAlmostEqual(res["refusal_rate"], 1 / 3, places=3)
        self.assertEqual(res["refusals"]["persona"]["refused_personas"], 2)
        w = " | ".join(res["warnings"])
        self.assertIn("4 personas unanswered (2 refused", w)
        self.assertIn("2 unparsed", w)

    def test_refusal_retry_off(self):
        llm = FakeLLM(lambda s, u: REFUSAL)
        res = survey(panel(4), "Q?", llm=llm, batch=2, controls=[], refusal_retry=False)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(res["refused"], 4)

    def test_refusals_never_cached(self):
        def responder(system, user):
            return answer_all(user) if REFRAME_NOTE in user else REFUSAL
        llm = CacheIfFake(responder)
        res = survey(panel(4), "Q?", llm=llm, batch=2, controls=[])
        self.assertEqual(res["n_answered"], 4)
        self.assertEqual(len(llm.calls), 4)                       # 2 refused + 2 reframed, no extra re-asks
        self.assertEqual(len(llm.cache), 2)
        self.assertFalse(any(looks_like_refusal(t) for t in llm.cache.values()))
        self.assertTrue(all(REFRAME_NOTE in u for _, u in llm.cache))
        survey(panel(4), "Q?", llm=llm, batch=2, controls=[])     # rerun: refusals asked again, answers cached
        self.assertEqual(len(llm.calls), 6)
        kept = CacheIfFake(responder)
        survey(panel(4), "Q?", llm=kept, batch=2, controls=[], cache_refusals=True)
        self.assertEqual(len(kept.cache), 4)                      # refusals cached too on request

    def test_cache_if_hook_passed(self):
        llm = RecordingFake(lambda s, u: answer_all(u))
        survey(panel(2), "Q?", llm=llm, controls=[], cache_refusals=True)
        self.assertNotIn("cache_if", llm.kws[0])
        llm2 = RecordingFake(lambda s, u: answer_all(u))
        survey(panel(2), "Q?", llm=llm2, controls=[])
        ok = llm2.kws[0]["cache_if"]
        self.assertTrue(ok('[{"id": "p0", "q1_score": 3}]'))
        self.assertFalse(ok(REFUSAL))
        self.assertTrue(llm2.kws[0]["return_exceptions"])

    def test_refusal_error_slot_is_reasked(self):
        class Raising(FakeLLM):
            def map(self, prompts, *, max_tokens=256, parallel=4, return_exceptions=False, **kw):
                out = []
                for s, u in prompts:
                    if REFRAME_NOTE in u:
                        out.append(self.complete(s, u))
                    else:
                        out.append(RefusalError("refusal (stop_details=None)"))
                return out
        res = survey(panel(3), "Q?", llm=Raising(lambda s, u: answer_all(u)), controls=[])
        self.assertEqual((res["n_answered"], res["refused"]), (3, 0))

    def test_other_errors_still_raise(self):
        class Broken(FakeLLM):
            def map(self, prompts, **kw):
                return [LLMError("HTTP 500 boom") for _ in prompts]
        with self.assertRaises(LLMError):
            survey(panel(2), "Q?", llm=Broken(), controls=[])


class TestNumericAndFreeText(unittest.TestCase):
    def test_numeric_slider_without_free_text(self):
        seen = []

        def responder(system, user):
            seen.append(user)
            ids = re.findall(r"^### (\S+)\s*$", user, re.M)
            vals = {"p0": 150, "p1": "70%", "p2": 0, "p3": 100, "p4": "about 42", "p5": -3}
            return json.dumps([{"id": i, "pol_value": vals[i]} for i in ids])
        q = {"id": "pol", "text": "Support for the policy, 0-100", "kind": "numeric", "min": 0, "max": 100,
             "labels": ["fully oppose", "fully support"], "free_text": False}
        res = survey(panel(6), [q], llm=FakeLLM(responder), controls=[])
        self.assertNotIn("_reason", seen[0])
        self.assertIn("number 0-100 (0 = fully oppose, 100 = fully support)", seen[0])
        self.assertNotIn("free-text answer before", seen[0])
        got = {p: a["pol"]["value"] for p, a in res["answers"].items()}
        self.assertEqual(got, {"p1": 70.0, "p2": 0.0, "p3": 100.0, "p4": 42.0})
        r = res["questions"][0]
        self.assertEqual((r["unparsed"], res["unparsed_items"]), (2, 2))       # 150 and -3 are out of range
        self.assertAlmostEqual(r["overall"]["mean"]["value"], 53.0)
        self.assertEqual(r["overall"]["n"], 4)
        self.assertTrue(any("out-of-range" in w for w in res["warnings"]))
        f = survey_frame(res, truth={"pol": {"p1": 60}})
        self.assertEqual(list(f["item"].unique()), ["pol"])
        self.assertEqual(f.set_index("group").loc["p1", "pred_value"], 70.0)
        fi = survey_frame(res, level="item")
        self.assertAlmostEqual(fi.set_index("segment").loc["all", "pred_value"], 53.0)

    def test_numeric_with_reason_and_dry_run(self):
        q = {"id": "wtp", "text": "Price you would pay", "kind": "numeric", "min": 10, "max": 60}
        res = survey(panel(5), [q], llm=FakeLLM(schema_responder(0)), batch=2)
        self.assertEqual(res["n_answered"], 5)
        self.assertTrue(all(10 <= a["wtp"]["value"] <= 60 for a in res["answers"].values()))
        self.assertTrue(all(a["wtp"]["text"] for a in res["answers"].values()))
        self.assertIn("no_persona", res["questions"][0]["arms"])
        f = survey_frame(res, kind="numeric")
        self.assertIn("pred_no_persona", f.columns)

    def test_numeric_bad_range(self):
        with self.assertRaises(ValueError):
            survey(panel(2), [{"text": "x", "kind": "numeric", "min": 5, "max": 5}], llm=FakeLLM(), controls=[])

    def test_likert_free_text_off(self):
        seen = []

        def responder(system, user):
            seen.append(user)
            return answer_all(user, value=4)

        class Dummy:
            def distribution(self, texts, anchors):
                raise AssertionError("SSR must not run without text")
        res = survey(panel(3), ["Q one?", {"id": "q2", "text": "Q two?", "free_text": True}],
                     llm=FakeLLM(responder), controls=[], free_text=False, ssr=Dummy())
        self.assertNotIn('"q1_answer"', seen[0])
        self.assertIn('"q2_answer"', seen[0])
        a = res["answers"]["p0"]
        self.assertEqual((a["q1"]["score"], a["q1"]["text"]), (4.0, None))
        self.assertEqual(a["q2"]["text"], "text of p0")
        self.assertTrue(any("SSR skipped" in w for w in res["warnings"]))

    def test_old_questions_unchanged(self):
        res = survey(panel(4), "Would you buy it?", llm=FakeLLM(schema_responder(1)), controls=[])
        self.assertEqual(res["questions"][0]["kind"], "likert5")
        self.assertTrue(all(a["q1"]["text"] for a in res["answers"].values()))
        self.assertEqual((res["refused"], res["unparsed"], res["refusal_rate"]), (0, 0, 0.0))


class TestStimulus(unittest.TestCase):
    MSGS = ["Message A", "Message B", "Message A", "Message C", "Message B", "Message A"]

    def test_stimulus_col_groups_batches(self):
        seen = []

        def responder(system, user):
            seen.append(user)
            return answer_all(user)
        res = survey(panel(6, msg=self.MSGS), "Would you share it?", llm=FakeLLM(responder), batch=8,
                     controls=["no_persona"], stimulus_col="msg", context="A social media feed.")
        self.assertEqual(res["n_answered"], 6)
        main = [u for u in seen if "person 0" in u or "person 1" in u or "person 3" in u]
        for u in seen:
            shown = set(re.findall(r"Message [ABC]", u))
            self.assertEqual(len(shown), 1, u)                     # one stimulus per batch
            self.assertIn("A social media feed.", u)
        for u in main:
            ids = re.findall(r"^### (\S+)\s*$", u, re.M)
            msg = re.search(r"Message [ABC]", u).group(0)
            self.assertTrue(all(self.MSGS[int(i[1:])] == msg for i in ids))
        self.assertEqual(res["refusals"]["persona"]["calls"], 3)       # 3 distinct messages
        self.assertEqual(res["stimuli"]["p3"], "Message C")
        # the no_persona control keeps each id's own stimulus
        self.assertEqual(sum(1 for u in seen if "typical adult" in u.lower()), 3)
        f = survey_frame(res)
        self.assertEqual(f.set_index("group").loc["p1", "stimulus"], "Message B")

    def test_context_dict_inline(self):
        seen = []

        def responder(system, user):
            seen.append(user)
            return answer_all(user)
        ctx = {f"p{i}": m for i, m in enumerate(self.MSGS)}
        res = survey(panel(6), "Q?", llm=FakeLLM(responder), batch=6, controls=[], context=ctx,
                     stimulus_batching="inline")
        self.assertEqual(len(seen), 1)
        self.assertIn("### p3\np3\nperson 3, age 23\nShown to this person: Message C", seen[0])
        self.assertIn("saw their own material", seen[0])
        self.assertEqual(res["n_answered"], 6)

    def test_stimulus_from_persona_facets_and_errors(self):
        import pandas as pd
        from simengine.persona import from_table
        df = pd.DataFrame({"rid": ["a", "b", "c"], "age": [30, 40, 50], "cond": ["X", "Y", None]})
        ps = from_table(df, id_col="rid")
        seen = []
        res = survey(ps, "Q?", llm=FakeLLM(lambda s, u: (seen.append(u), answer_all(u))[1]), controls=[],
                     stimulus_col="cond")
        self.assertEqual(res["stimuli"], {"a": "X", "b": "Y", "c": None})
        self.assertTrue(any("1 personas have no stimulus" in w for w in res["warnings"]))
        self.assertEqual(len(seen), 3)
        with self.assertRaises(ValueError):
            survey(ps, "Q?", llm=FakeLLM(), context={"a": "x"}, stimulus_col="cond")
        with self.assertRaises(ValueError):
            survey(ps, "Q?", llm=FakeLLM(), stimulus_col="cond", stimulus_batching="mixed")


class TestLedgerTag(unittest.TestCase):
    def test_tag_per_arm(self):
        llm = RecordingFake(schema_responder(0))
        survey(panel(4), "Q?", llm=llm)
        self.assertEqual({k["tag"] for k in llm.kws}, {"survey:persona", "survey:no_persona", "survey:persona_swap"})

    def test_backend_tag_is_kept_as_prefix(self):
        llm = RecordingFake(schema_responder(0), tag="agent7")
        survey(panel(2), "Q?", llm=llm, controls=["no_persona"])
        self.assertEqual({k["tag"] for k in llm.kws}, {"agent7:survey:persona", "agent7:survey:no_persona"})

    def test_ask_panel_llm_tag(self):
        llm = RecordingFake(schema_responder(0))
        ask_panel(llm, as_personas(["a", "b"]), "t", {"score": "int 1-5"}, tag="ab:A")
        ask_panel(llm, as_personas(["a", "b"]), "t", {"score": "int 1-5"}, tag="x", llm_tag="custom")
        self.assertEqual([k["tag"] for k in llm.kws], ["ab:A", "custom"])


if __name__ == "__main__":
    unittest.main()
