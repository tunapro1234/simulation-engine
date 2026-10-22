"""segment_distribution / persona_distribution / ask_constrained accept the llm positionally (legacy) or as llm=."""
import json
import unittest

from simengine.llm import FakeLLM
from simengine.sim import FieldSpec, ask_constrained, persona_distribution, schema_responder, segment_distribution

QS = [{"id": "q1", "text": "Do you approve?", "options": ["Approve", "Neutral", "Disapprove"]},
      {"id": "q2", "text": "Own a car?", "options": ["Yes", "No"]}]
SEGS = {"young": "adults aged 18 to 34", "old": "adults aged 65 or older"}
PS = [{"id": f"u{i}", "text": f"person {i}", "segment": "A" if i < 2 else "B"} for i in range(4)]
FIELDS = [FieldSpec("contribution", "int", min=0, max=20), FieldSpec("vote", "choice", choices=[("1", "Yes"), ("0", "No")])]


def answer(system, user):
    return json.dumps({"contribution": 7, "vote": "1"})


class TestSegmentDistributionLLMKeyword(unittest.TestCase):
    def test_forms_identical(self):
        a, b = FakeLLM(schema_responder(0)), FakeLLM(schema_responder(0))
        r1 = segment_distribution(a, SEGS, QS, population="US adults")
        r2 = segment_distribution(SEGS, QS, llm=b, population="US adults")
        r3 = segment_distribution(llm=FakeLLM(schema_responder(0)), segments=SEGS, questions=QS,
                                  population="US adults")
        self.assertEqual(r1, r2)
        self.assertEqual(r1, r3)
        self.assertEqual(a.calls, b.calls)

    def test_both_or_neither(self):
        llm = FakeLLM(schema_responder(0))
        with self.assertRaisesRegex(TypeError, "both"):
            segment_distribution(llm, SEGS, QS, llm=llm)
        with self.assertRaisesRegex(TypeError, "both"):
            segment_distribution(llm, SEGS, llm=llm)
        with self.assertRaisesRegex(TypeError, "needs an llm"):
            segment_distribution()
        with self.assertRaisesRegex(TypeError, "multiple values"):
            segment_distribution(SEGS, QS, llm=llm, segments=SEGS)
        with self.assertRaisesRegex(TypeError, "missing required argument: 'questions'"):
            segment_distribution(SEGS, llm=llm)
        self.assertEqual(llm.calls, [])


class TestPersonaDistributionLLMKeyword(unittest.TestCase):
    def test_forms_identical(self):
        r1 = persona_distribution(FakeLLM(schema_responder(0)), PS, QS, batch=2)
        r2 = persona_distribution(PS, QS, llm=FakeLLM(schema_responder(0)), batch=2)
        self.assertEqual(r1, r2)

    def test_both_given(self):
        llm = FakeLLM(schema_responder(0))
        with self.assertRaisesRegex(TypeError, "both"):
            persona_distribution(llm, PS, QS, llm=llm)


class TestAskConstrainedLLMKeyword(unittest.TestCase):
    def test_forms_identical(self):
        a, b = FakeLLM(answer), FakeLLM(answer)
        r1 = ask_constrained(a, "persona", "Public goods game, round 1.", FIELDS, meta={"agent_id": "x"})
        r2 = ask_constrained("persona", "Public goods game, round 1.", FIELDS, llm=b, meta={"agent_id": "x"})
        self.assertTrue(r1.completed)
        self.assertEqual(r1.values, {"contribution": 7, "vote": "1"})
        self.assertEqual((r1.values, r1.attempts, r1.mode, r1.meta, r1.trace),
                         (r2.values, r2.attempts, r2.mode, r2.meta, r2.trace))
        self.assertEqual(a.calls, b.calls)

    def test_keyword_kw_still_reaches_complete(self):
        r = ask_constrained("", "page", FIELDS, llm=FakeLLM(answer), max_tokens=64, effort="low")
        self.assertEqual(r.max_tokens, 64)
        self.assertTrue(r.completed)

    def test_both_or_neither(self):
        llm = FakeLLM(answer)
        with self.assertRaisesRegex(TypeError, "both"):
            ask_constrained(llm, "persona", "page", FIELDS, llm=llm)
        with self.assertRaisesRegex(TypeError, "needs an llm"):
            ask_constrained(system="persona", prompt="page", fields=FIELDS)
        with self.assertRaisesRegex(TypeError, "positional arguments"):
            ask_constrained("persona", "page", FIELDS, "extra", llm=llm)
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
