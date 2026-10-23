"""compare.estimate_cost prices every prompt as its own call (no long-context price from summed tokens).

    nice -n 19 python -m unittest tests.test_persona_inject_compare_cost
"""
import unittest

import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM
from simengine.llm.pricing import cost_usd, price_for
from simengine.persona.inject._base import Prompt

HAIKU = "claude-haiku-5-5"


def short_prompts(n: int, chars: int = 2000) -> list[Prompt]:
    return [Prompt("s" * 100, "u" * (chars - 100)) for _ in range(n)]


class TestEstimateCostPerCall(unittest.TestCase):
    def test_many_short_prompts_stay_under_long_context_price(self):
        n, mt = 1000, 24                      # 1000 x 500 tokens = 500k summed, far above the 100k threshold
        llm = FakeLLM(model=HAIKU)
        est = pi.estimate_cost(llm, short_prompts(n), mt)
        one = cost_usd(HAIKU, 2000 // 4, mt)
        self.assertEqual(est["calls"], n)
        self.assertEqual(est["input_tokens_est"], n * 500)
        self.assertEqual(est["output_tokens_max"], n * mt)
        self.assertAlmostEqual(est["usd_per_call_max"], round(one, 6), places=6)
        self.assertAlmostEqual(est["usd_max"], round(n * one, 4), places=4)
        p = price_for(HAIKU)
        short = (n * 500 * p.input_per_mtok + n * mt * p.output_per_mtok) / 1e6
        self.assertAlmostEqual(est["usd_max"], round(short, 4), places=4)
        self.assertGreater(est["input_tokens_est"], p.long_context_threshold)
        self.assertEqual(est["model"], HAIKU)

    def test_one_long_prompt_still_gets_long_price(self):
        mt = 10
        est = pi.estimate_cost(FakeLLM(model=HAIKU), [Prompt("", "x" * 4 * 150_000)], mt)
        self.assertAlmostEqual(est["usd_max"], round(cost_usd(HAIKU, 150_000, mt), 4))
        p = price_for(HAIKU)
        self.assertAlmostEqual(est["usd_max"], (150_000 * p.long_input_per_mtok + mt * p.long_output_per_mtok) / 1e6,
                               places=4)

    def test_per_call_max_is_the_dearest_call(self):
        ps = [Prompt("", "a" * 400), Prompt("", "b" * 4000)]
        est = pi.estimate_cost(FakeLLM(model=HAIKU), ps, 8)
        self.assertAlmostEqual(est["usd_per_call_max"], round(cost_usd(HAIKU, 1000, 8), 6))
        self.assertAlmostEqual(est["usd_max"], round(cost_usd(HAIKU, 100, 8) + cost_usd(HAIKU, 1000, 8), 4))

    def test_unpriced_model_gives_none(self):
        est = pi.estimate_cost(FakeLLM(), short_prompts(5), 16)     # model "fake"
        self.assertIsNone(est["usd_max"])
        self.assertIsNone(est["usd_per_call_max"])
        self.assertEqual(est["calls"], 5)
        self.assertEqual(est["input_tokens_est"], 5 * 500)
        self.assertEqual(est["output_tokens_max"], 5 * 16)
        self.assertEqual(est["model"], "fake")

    def test_empty(self):
        est = pi.estimate_cost(FakeLLM(model=HAIKU), [], 16)
        self.assertEqual((est["calls"], est["input_tokens_est"], est["output_tokens_max"], est["usd_max"]),
                         (0, 0, 0, 0.0))

    def test_max_usd_guard_lets_a_fitting_run_through(self):
        # 40 people x 2 questions x 3 arms = 240 calls with ~600-token personas: summed input > 100k, so the old
        # one-call pricing hit the long price (~5x) and refused a run that fits max_usd.
        people = [{"id": f"u{i}", "fields": {"bio": "z" * 2400}, "answers": {"q0": "yes", "q1": "no"}}
                  for i in range(40)]
        qs = [{"id": "q0", "text": "A?", "options": ["yes", "no"]}, {"id": "q1", "text": "B?", "options": ["yes", "no"]}]
        llm = FakeLLM(model=HAIKU)
        plan = pi.compare_injectors(llm, people, qs, "answers", ["plain_fields"], dry_run=True, max_tokens=24)
        est = plan.estimate
        self.assertGreater(est["input_tokens_est"], 100_000)
        old = cost_usd(HAIKU, est["input_tokens_est"], est["output_tokens_max"])
        self.assertGreater(old, 4 * est["usd_max"])
        res = pi.compare_injectors(llm, people, qs, "answers", ["plain_fields"], max_tokens=24, B=50,
                                   max_usd=2 * est["usd_max"])
        self.assertEqual(len(llm.calls), est["calls"])
        self.assertIsNotNone(res.table)


if __name__ == "__main__":
    unittest.main()
