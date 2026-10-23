"""sim.segment_distribution include_unsure= (D6 #5): the default prompt keeps 'ignore "not sure" answers'
byte-identical (prompt hashes / caches), include_unsure=True drops it, and a not-sure-like option warns."""
import unittest
import warnings

from simengine.llm import FakeLLM
from simengine.llm.fake_distribution import DistributionFake, parse_distribution_prompt
from simengine.sim import FRAMINGS_HIDDEN, distribution_prompt, schema_responder, segment_distribution

Q = {"id": "q1", "text": "Do you approve?", "options": ["Approve", "Neutral", "Disapprove"]}
QU = {"id": "q2", "text": "Should the city ban cars downtown?", "options": ["Yes", "No", "Not sure / neither"]}
SEGS = {"young": "adults aged 18 to 34"}

# The prompt as rendered before include_unsure existed; it must not change (cache keys, pre-registered hashes).
LEGACY = ("In a large national survey of US adults, consider respondents who are adults aged 18 to 34.\n"
          "Question: Do you approve?\n"
          "Answer options:\n1. Approve\n2. Neutral\n3. Disapprove\n"
          'Estimate the percentage of these respondents who would choose each option (ignore "not sure" answers). '
          "Percentages must sum to 100.\n"
          'Return ONLY a JSON object mapping option number to percentage, e.g. {"1": 33, "2": 33, "3": 33}')


class TestIncludeUnsure(unittest.TestCase):
    def test_default_prompt_byte_identical(self):
        p = distribution_prompt(FRAMINGS_HIDDEN["h1"], "adults aged 18 to 34", Q, population="US adults")
        self.assertEqual(p, LEGACY)
        self.assertEqual(p, distribution_prompt(FRAMINGS_HIDDEN["h1"], "adults aged 18 to 34", Q,
                                                population="US adults", include_unsure=None))

    def test_true_drops_parenthetical(self):
        p = distribution_prompt(FRAMINGS_HIDDEN["h1"], "adults aged 18 to 34", Q, population="US adults",
                                include_unsure=True)
        self.assertNotIn("not sure", p)
        self.assertEqual(p, LEGACY.replace(' (ignore "not sure" answers)', ""))
        self.assertEqual(distribution_prompt(FRAMINGS_HIDDEN["h1"], "g", Q, include_unsure=False),
                         distribution_prompt(FRAMINGS_HIDDEN["h1"], "g", Q))

    def test_segment_distribution_passes_flag(self):
        for flag, has in ((None, True), (False, True), (True, False)):
            llm = FakeLLM(schema_responder(0))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                segment_distribution(SEGS, [Q, QU], llm=llm, include_unsure=flag)
            prompts = [user for _, user in llm.calls]
            self.assertEqual(len(prompts), 3 * 2)
            self.assertTrue(all(('(ignore "not sure" answers)' in p) == has for p in prompts), flag)

    def test_warns_once_on_unsure_option(self):
        qs = [Q, QU, {"id": "q3", "text": "Trust?", "options": ["A lot", "Some", "Don\u2019t know"]}]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            segment_distribution(SEGS, qs, llm=FakeLLM(schema_responder(0)))
        msgs = [str(x.message) for x in w if "include_unsure" in str(x.message)]
        self.assertEqual(len(msgs), 1)
        self.assertIn("Not sure / neither", msgs[0])
        self.assertIn("Don\u2019t know", msgs[0])
        for flag in (True, False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                segment_distribution(SEGS, qs, llm=FakeLLM(schema_responder(0)), include_unsure=flag)
            self.assertFalse([x for x in w if "include_unsure" in str(x.message)])
        with warnings.catch_warnings(record=True) as w:     # ordinary options: silent
            warnings.simplefilter("always")
            segment_distribution(SEGS, [Q], llm=FakeLLM(schema_responder(0)))
        self.assertFalse([x for x in w if "include_unsure" in str(x.message)])

    def test_fake_parses_both_variants(self):
        for flag in (None, True):
            p = distribution_prompt(FRAMINGS_HIDDEN["h1"], "adults aged 18 to 34", QU, population="US adults",
                                    include_unsure=flag)
            d = parse_distribution_prompt(p)
            self.assertEqual(d["options"], QU["options"])
            self.assertEqual(d["group"], "adults aged 18 to 34")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = segment_distribution(SEGS, [QU], llm=DistributionFake(), include_unsure=flag)
            self.assertEqual(r["invalid"], 0)
            self.assertAlmostEqual(sum(r["dist"]["young"]["q2"]), 1.0)


if __name__ == "__main__":
    unittest.main()
