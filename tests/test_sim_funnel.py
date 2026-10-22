"""sim.funnel: per-step continue rates, cumulative reach, drop-off, paired variant comparison, long frame."""
import re
import unittest

import numpy as np

from simengine.llm.fake import FakeLLM, survey_responder
from simengine.sim.funnel import QUESTION, funnel, funnel_frame, step_context

STEP = re.compile(r"Now you are on step (\d+) of (\d+):\n(.*)")


def panel(n=10):
    return [{"id": f"p{i}", "text": f"person {i}", "segment": "A" if i % 2 else "B"} for i in range(n)]


def responder():
    """Continue 0-100 read off the current step: 'BAD' steps ~20, others ~80, with a per-persona offset (mean 0 on 10 personas)."""
    def respond(system: str, user: str) -> str:
        m = STEP.search(user)
        base = 20 if (m and "BAD" in m.group(3)) else 80
        return survey_responder(lambda pid, qid, spec: base + 2 * int(pid[1:]) - 9)(system, user)
    return respond


VARIANTS = {"good": ["Landing: a budgeting app", "Pricing: free, then $3/month", "Checkout: card"],
            "bad": ["Landing: a budgeting app", "Pricing: BAD $30/month, no trial", "Checkout: card"]}


class TestFunnel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = FakeLLM(responder())
        cls.res = funnel(panel(), VARIANTS, llm=cls.llm, B=300)

    def test_bad_step_drops(self):
        good, bad = self.res["variants"]["good"], self.res["variants"]["bad"]
        self.assertAlmostEqual(bad["steps"][1]["continue"]["value"], 0.20, places=6)
        self.assertAlmostEqual(good["steps"][1]["continue"]["value"], 0.80, places=6)
        self.assertAlmostEqual(bad["steps"][1]["drop_rate"], 0.80, places=6)
        self.assertEqual(bad["biggest_drop"], "s2")
        # drop at step k = reach(k-1) - reach(k)
        self.assertAlmostEqual(bad["steps"][1]["drop"]["value"],
                               bad["steps"][0]["reach"]["value"] - bad["steps"][1]["reach"]["value"], places=9)
        self.assertLess(bad["conversion"]["value"], good["conversion"]["value"])

    def test_reach_is_persona_product_and_non_increasing(self):
        for vid, v in self.res["variants"].items():
            reach = [s["reach"]["value"] for s in v["steps"]]
            self.assertTrue(all(b <= a + 1e-12 for a, b in zip(reach, reach[1:])), vid)
            self.assertLessEqual(reach[0], 1.0)
            ans = self.res["answers"][vid]
            prod = np.mean([np.prod(a["p"]) for a in ans.values()])
            self.assertAlmostEqual(v["conversion"]["value"], prod, places=9)

    def test_cis_ordered(self):
        for v in self.res["variants"].values():
            for s in v["steps"]:
                for key in ("continue", "reach", "drop"):
                    c = s[key]
                    self.assertLessEqual(c["low"], c["value"], key)
                    self.assertLessEqual(c["value"], c["high"], key)
                    self.assertLess(c["low"], c["high"], key)       # per-persona offsets: non-degenerate CIs
                    self.assertEqual(c["n"], 10)
        pr = self.res["comparison"]["pairs"][0]
        self.assertEqual((pr["variant"], pr["baseline"]), ("bad", "good"))
        self.assertLessEqual(pr["diff"]["low"], pr["diff"]["value"])
        self.assertLessEqual(pr["diff"]["value"], pr["diff"]["high"])
        self.assertLess(pr["diff"]["high"], 0)
        self.assertLess(pr["p_better"], 0.05)
        self.assertEqual(self.res["comparison"]["best"], "good")

    def test_earlier_steps_in_later_prompts(self):
        users = [u for _, u in self.llm.calls]
        third = [u for u in users if "step 3 of 3" in u and "BAD" in u]
        self.assertTrue(third)
        for u in third:
            seen = u.split("You have already seen", 1)[1].split("Now you are on step", 1)[0]
            self.assertIn("Landing: a budgeting app", seen)
            self.assertIn("Pricing: BAD $30/month", seen)
        first = [u for u in users if "step 1 of 3" in u]
        self.assertTrue(first)
        self.assertTrue(all("You have already seen" not in u for u in first))
        self.assertTrue(all(QUESTION in u for u in users))

    def test_no_controls_by_default_and_usage(self):
        self.assertEqual(self.res["controls"], [])
        self.assertTrue(all("controls" not in s for v in self.res["variants"].values() for s in v["steps"]))
        self.assertEqual(self.res["usage"]["surveys"], 6)
        self.assertEqual(self.res["usage"]["calls"], len(self.llm.calls))
        self.assertEqual(self.res["warnings"], [])

    def test_frame(self):
        f = funnel_frame(self.res)
        self.assertEqual(len(f), 2 * 3 * 10)
        self.assertEqual(list(f.columns), ["variant", "step", "step_index", "persona", "group", "segment",
                                           "p_continue", "reach"])
        g = f[f.variant == "bad"].groupby("step_index")["reach"].mean()
        self.assertAlmostEqual(g.iloc[-1], self.res["variants"]["bad"]["conversion"]["value"], places=9)
        self.assertTrue((f["group"] == f["persona"]).all())


class TestFunnelOptions(unittest.TestCase):
    def test_controls_arm_and_step_dicts(self):
        res = funnel(panel(4), {"A": [{"id": "land", "text": "Landing"}, {"id": "pay", "text": "Pay BAD"}]},
                     llm=FakeLLM(responder()), controls=["no_persona"], B=50)
        steps = res["variants"]["A"]["steps"]
        self.assertEqual([s["id"] for s in steps], ["land", "pay"])
        self.assertIn("no_persona", steps[0]["controls"])
        c = steps[0]["controls"]["no_persona"]
        self.assertTrue(0 <= c["value"] <= 1)
        self.assertEqual(res["comparison"]["pairs"], [])
        self.assertEqual(res["comparison"]["best"], "A")

    def test_missing_answers_leave_reach_and_warn(self):
        def respond(system, user):
            out = responder()(system, user)
            if "step 2 of 2" in user:          # p0 / p1 unanswered at step 2 (also on re-asks)
                out = re.sub(r'\{"id": "p[01]"[^}]*\}, ?', "", out)
            return out
        res = funnel(panel(6), {"A": ["Landing", "Pay"]}, llm=FakeLLM(respond), B=50, batch=10)
        v = res["variants"]["A"]
        self.assertEqual(v["n_complete"], 4)
        self.assertEqual(v["steps"][1]["reach"]["n"], 4)
        self.assertTrue(any("missing an answer" in w for w in res["warnings"]))

    def test_step_context_and_validation(self):
        steps = [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}]
        self.assertNotIn("already seen", step_context(steps, 0, "shop"))
        ctx = step_context(steps, 1, "shop")
        self.assertTrue(ctx.startswith("shop"))
        self.assertIn("Step 1: one", ctx)
        self.assertIn("step 2 of 2:\ntwo", ctx)
        llm = FakeLLM(responder())
        for bad in ({}, {"A": []}, {"A": ["ok", "  "]}, {"A": [{"id": "x", "text": "1"}, {"id": "x", "text": "2"}]}):
            with self.assertRaises(ValueError):
                funnel(panel(2), bad, llm=llm)


if __name__ == "__main__":
    unittest.main()
