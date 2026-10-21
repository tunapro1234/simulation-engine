"""simengine.sim segment-distribution prompting and per-persona probability vectors."""
import json
import re
import unittest

import numpy as np

from simengine.llm import FakeLLM
from simengine.sim import (FRAMINGS_HIDDEN, FRAMINGS_NAMED, distribution_frame, distribution_prompt,
                           persona_distribution, schema_responder, segment_distribution, w1_ordinal)

QS = [{"id": "q1", "text": "Do you approve?", "options": ["Approve", "Neutral", "Disapprove"]},
      {"id": "q2", "text": "Own a car?", "options": ["Yes", "No"]}]
SEGS = {"young": "adults aged 18 to 34", "old": "adults aged 65 or older"}


class TestSegmentDistribution(unittest.TestCase):
    def test_prompt_shape(self):
        p = distribution_prompt(FRAMINGS_HIDDEN["h1"], "adults aged 18 to 34", QS[0], population="US adults")
        self.assertTrue(p.startswith("In a large national survey of US adults, consider respondents who are adults aged 18"))
        self.assertIn("Answer options:\n1. Approve\n2. Neutral\n3. Disapprove\n", p)
        self.assertIn("Percentages must sum to 100.", p)
        self.assertTrue(p.endswith('e.g. {"1": 33, "2": 33, "3": 33}'))
        n = distribution_prompt(FRAMINGS_NAMED["v2"], "g", QS[1], population="adults", when="2020")
        self.assertIn("conducted in 2020", n)

    def test_calls_average_and_frame(self):
        llm = FakeLLM(schema_responder(0))
        r = segment_distribution(llm, SEGS, QS, population="US adults")
        json.dumps(r)
        self.assertEqual(r["calls"], 3 * 2 * 2)          # framings x segments x questions
        self.assertEqual(len(llm.calls), 12)
        self.assertEqual(r["invalid"], 0)
        for s in SEGS:
            self.assertEqual(len(r["dist"][s]["q1"]), 3)
            self.assertAlmostEqual(sum(r["dist"][s]["q1"]), 1.0)
        cells = [c["p"] for c in r["cells"] if c["segment"] == "young" and c["item"] == "q2"]
        np.testing.assert_allclose(np.mean(cells, axis=0), r["dist"]["young"]["q2"])
        f = distribution_frame(r, truth={"young": {"q1": [0.5, 0.2, 0.3]}})
        self.assertEqual(len(f), 4)
        self.assertEqual(f.loc[f.unit == "young|q1", "truth"].item(), [0.5, 0.2, 0.3])
        self.assertIn("pred_llm_dist", f.columns)

    def test_named_framings_when_source_given(self):
        llm = FakeLLM(schema_responder(0))
        r = segment_distribution(llm, ["Democrats"], QS[:1], population="US adults", source="a 2020 election study",
                                 when="October 2020")
        self.assertEqual(r["framings"], ["v1", "v2", "v3"])
        self.assertIn("In a 2020 election study, a large national survey of US adults fielded in October 2020", llm.calls[0][1])
        self.assertIn("including minority views", llm.calls[2][1])

    def test_invalid_becomes_uniform_or_skipped(self):
        def responder(system, user):
            if "Consider adults aged 18" in user:      # h2 framing for one segment: unusable reply
                return "I cannot estimate that."
            return json.dumps({"1": 60, "2": 30, "3": 10}) if "3. " in user else json.dumps({"1": 80, "2": 20})
        r = segment_distribution(FakeLLM(responder), SEGS, QS)
        self.assertEqual(r["invalid"], 2)
        np.testing.assert_allclose(r["dist"]["young"]["q1"], (2 * np.array([0.6, 0.3, 0.1]) + 1 / 3) / 3)
        np.testing.assert_allclose(r["dist"]["old"]["q1"], [0.6, 0.3, 0.1])
        r = segment_distribution(FakeLLM(responder), SEGS, QS, fallback="skip")
        np.testing.assert_allclose(r["dist"]["young"]["q1"], [0.6, 0.3, 0.1])

    def test_w1(self):
        self.assertAlmostEqual(w1_ordinal([1, 0, 0], [0, 0, 1]), 1.0)
        self.assertAlmostEqual(w1_ordinal([0.5, 0.5], [0.5, 0.5]), 0.0)
        self.assertAlmostEqual(w1_ordinal([1, 0, 0], [0, 1, 0]), 0.5)


class TestPersonaDistribution(unittest.TestCase):
    PS = [{"id": f"u{i}", "text": f"person {i}", "segment": "A" if i < 3 else "B"} for i in range(5)]

    def test_fake_vectors(self):
        llm = FakeLLM(schema_responder(0))
        r = persona_distribution(llm, self.PS, QS, batch=2)
        json.dumps(r)
        self.assertEqual(r["calls"], 3)
        self.assertEqual(r["invalid"], 0)
        self.assertEqual(len(r["person"]["u0"]["q1"]), 3)
        self.assertAlmostEqual(sum(r["dist"]["B"]["q2"]), 1.0)

    def test_partial_and_invalid_rows(self):
        def responder(system, user):
            ids = re.search(r"respondent IDs as keys \(([^)]*)\)", user).group(1).split(", ")
            out = {i: {"p": [[50, 30, 20], [100, 0]]} for i in ids if i != "u1"}
            if "u2" in out:
                out["u2"] = {"p": [[50, 50], [100, 0]]}   # wrong length for q1
            return "Here you go:\n" + json.dumps(out)
        r = persona_distribution(FakeLLM(responder), self.PS, QS)
        self.assertEqual(r["invalid"], 2)
        self.assertIsNone(r["person"]["u1"]["q1"])
        np.testing.assert_allclose(r["dist"]["A"]["q1"], (np.array([0.5, 0.3, 0.2]) + 2 / 3) / 3)
        np.testing.assert_allclose(r["dist"]["B"]["q2"], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
