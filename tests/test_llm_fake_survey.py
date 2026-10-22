"""FakeLLM.survey / survey_responder: schema-true offline answers for sim.survey prompts (contract test: if the
survey prompt layout changes, these fail instead of a dry run silently going all 'unparsed')."""
import json
import unittest

from simengine.llm import FakeLLM
from simengine.llm.fake import survey_responder
from simengine.sim import schema_responder, survey, survey_frame

QS = [{"id": "q_num", "text": "How much would you pay?", "kind": "numeric", "min": 0.25, "max": 0.75},
      {"id": "q_wide", "text": "Support?", "kind": "slider", "min": -10, "max": 10, "labels": ["no", "yes"]},
      {"id": "q_pick", "text": "Pick one", "options": ["red", "blue", "green"]},
      {"id": "q_lik", "text": "I would buy it."}]


def panel(n=6):
    return [{"id": f"p{i}", "text": f"person {i}"} for i in range(n)]


def run(llm, free_text=True, controls=None, n=6):
    return survey(panel(n), QS, llm=llm, controls=controls, B=20, free_text=free_text)


class TestContract(unittest.TestCase):
    def test_numeric_and_choice_all_parsed_and_in_range(self):
        for free_text in (True, False):
            res = run(FakeLLM.survey(), free_text=free_text)
            self.assertEqual(res["unparsed"], 0, free_text)
            self.assertEqual(res["refused"], 0)
            num = survey_frame(res, kind="numeric")
            for qid, lo, hi in (("q_num", 0.25, 0.75), ("q_wide", -10, 10)):
                v = num.loc[num["item"] == qid, "pred_value"]
                self.assertEqual(len(v), 6)
                self.assertTrue(v.between(lo, hi).all(), (qid, list(v)))
            ch = survey_frame(res, kind="choice")
            self.assertEqual(len(ch), 6)
            self.assertTrue(ch["pred_choice"].isin(["red", "blue", "green"]).all())
            lik = survey_frame(res, kind="likert5")
            self.assertTrue(lik["pred_score"].between(1, 5).all())

    def test_control_arms_parse_too(self):
        res = run(FakeLLM.survey(seed=3), controls=["no_persona", "persona_swap"])
        self.assertEqual(res["unparsed"], 0)
        num = survey_frame(res, kind="numeric")
        cols = [c for c in num.columns if c.startswith("pred_")]
        self.assertEqual(set(cols), {"pred_value", "pred_no_persona", "pred_persona_swap"})
        for col in cols:
            self.assertEqual(num[col].isna().sum(), 0, col)
            self.assertTrue(num[col].between(-10, 10).all(), col)

    def test_bare_or_empty_responder_still_unparsed(self):
        res = run(FakeLLM(lambda s, u: "{}"), controls=[])
        self.assertEqual(res["unparsed"], 6)       # the failure mode FakeLLM.survey removes


class TestValueFn(unittest.TestCase):
    def test_values_specs_and_clamping(self):
        seen = {}

        def fn(pid, qid, spec):
            seen[qid] = spec
            i = int(pid[1:])
            if qid == "q_num":
                return 0.3 + 0.1 * i                # p5 -> 0.8, clamped to 0.75
            if qid == "q_pick":
                return ["red", "b", 2][i % 3]       # option text, letter, index
            return None                             # hashed default

        res = run(FakeLLM.survey(fn), controls=[])
        self.assertEqual(res["unparsed"], 0)
        self.assertEqual(seen["q_num"]["kind"], "numeric")
        self.assertEqual((seen["q_num"]["min"], seen["q_num"]["max"]), (0.25, 0.75))
        self.assertEqual(seen["q_pick"]["options"], ["red", "blue", "green"])
        self.assertEqual(seen["q_pick"]["text"], "Pick one")
        self.assertEqual(seen["q_lik"]["kind"], "likert5")
        self.assertTrue(seen["q_lik"]["integer"])
        num = survey_frame(res, kind="numeric")
        v = num[num["item"] == "q_num"].set_index("group")["pred_value"]
        self.assertAlmostEqual(v["p0"], 0.3)
        self.assertAlmostEqual(v["p2"], 0.5)
        self.assertAlmostEqual(v["p5"], 0.75)
        ch = survey_frame(res, kind="choice").set_index("group")["pred_choice"]
        self.assertEqual([ch[f"p{i}"] for i in range(3)], ["red", "blue", "green"])

    def test_bad_value_raises(self):
        with self.assertRaises(ValueError):
            run(FakeLLM.survey(lambda pid, qid, spec: "purple" if spec["kind"] == "choice" else None), controls=[])
        with self.assertRaises(ValueError):
            run(FakeLLM.survey(lambda pid, qid, spec: "lots" if spec["kind"] == "numeric" else None), controls=[])


class TestResponder(unittest.TestCase):
    def test_deterministic_and_seeded(self):
        a = run(FakeLLM.survey(seed=1), controls=[])
        b = run(FakeLLM.survey(seed=1), controls=[])
        fa, fb = survey_frame(a, kind="numeric"), survey_frame(b, kind="numeric")
        self.assertEqual(list(fa["pred_value"]), list(fb["pred_value"]))

    def test_classmethod_returns_fakellm(self):
        llm = FakeLLM.survey(model="fake-survey")
        self.assertIsInstance(llm, FakeLLM)
        self.assertEqual(llm.model, "fake-survey")

    def test_non_panel_prompt_falls_through_to_schema_responder(self):
        user = "Answer options:\n1. yes\n2. no\n\nGive the percentage for each option."
        self.assertEqual(survey_responder(seed=2)("sys", user), schema_responder(2)("sys", user))
        self.assertEqual(json.loads(survey_responder()("sys", "plain prompt")), {})


if __name__ == "__main__":
    unittest.main()
