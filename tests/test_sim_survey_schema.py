"""survey question schema: kind inferred from options, unknown kind raises, bare-FakeLLM hint, frame keys."""
import unittest

from simengine.llm import FakeLLM
from simengine.sim import schema_responder, survey, survey_frame
from simengine.sim.survey import _questions


def panel(n=4):
    return [{"id": f"p{i}", "text": f"person {i}", "segment": "A" if i % 2 else "B"} for i in range(n)]


class TestQuestionKind(unittest.TestCase):
    def test_options_without_kind_is_choice(self):
        q = _questions([{"id": "c", "text": "Pick one", "options": ["red", "blue"]}])[0]
        self.assertEqual(q["kind"], "choice")
        self.assertEqual(q["options"], ["red", "blue"])

    def test_no_options_no_kind_stays_likert(self):
        self.assertEqual(_questions([{"id": "l", "text": "Agree?"}])[0]["kind"], "likert5")
        self.assertEqual(_questions(["Agree?"])[0]["kind"], "likert5")

    def test_explicit_kind_wins_and_aliases_still_map(self):
        qs = _questions([{"id": "a", "text": "x", "kind": "likert5"},
                         {"id": "b", "text": "x", "kind": "mc", "options": ["y", "n"]},
                         {"id": "c", "text": "x", "kind": "slider", "min": 0, "max": 10},
                         {"id": "d", "text": "x", "tur": "coktan", "secenekler": ["y", "n"]},
                         {"id": "e", "text": "x", "kind": "Numeric"},
                         {"id": "f", "text": "x", "kind": "likert"}])
        self.assertEqual([q["kind"] for q in qs], ["likert5", "choice", "numeric", "choice", "numeric", "likert5"])

    def test_unknown_kind_raises_with_accepted_list(self):
        for bad in ("percent", "multiple_choice"):
            with self.assertRaises(ValueError) as cm:
                _questions([{"id": "q", "text": "x", "kind": bad, "options": ["a", "b"]}])
            msg = str(cm.exception)
            self.assertIn(repr(bad), msg)
            for k in ("likert5", "numeric", "number", "slider", "choice"):
                self.assertIn(k, msg)


class TestSurveySchema(unittest.TestCase):
    def test_options_only_dict_gives_choice_frame(self):
        qs = [{"id": "c", "text": "Pick one", "options": ["red", "blue", "green"]}]
        res = survey(panel(), qs, llm=FakeLLM(schema_responder()), controls=[], B=20)
        self.assertEqual(res["questions"][0]["kind"], "choice")
        fr = survey_frame(res, kind="choice")
        self.assertEqual(len(fr), 4)
        self.assertTrue(fr["pred_choice"].isin(["red", "blue", "green"]).all())
        self.assertEqual(set(fr["group"]), {f"p{i}" for i in range(4)})
        self.assertTrue(all(u == f"{g}|c" for u, g in zip(fr["unit"], fr["group"])))

    def test_bare_fakellm_hint(self):
        res = survey(panel(3), ["Agree?"], llm=FakeLLM(), controls=[], B=20)
        self.assertEqual(res["n_answered"], 0)
        self.assertTrue(any("FakeLLM(simengine.sim.schema_responder())" in w for w in res["warnings"]))

    def test_no_hint_with_responder(self):
        res = survey(panel(3), ["Agree?"], llm=FakeLLM(schema_responder()), controls=[], B=20)
        self.assertFalse(any("no responder" in w for w in res["warnings"]))


if __name__ == "__main__":
    unittest.main()
