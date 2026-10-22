"""simengine.sim.effect: direct per-condition effect forecasts, named/hidden arms, memorisation probe."""
import json
import math
import re
import unittest

from simengine.llm import FakeLLM
from simengine.sim.effect import (FRAMINGS, EffectForecast, effect_forecast, effect_frame, effect_prompt,
                                  memorisation_check)

CONDS = {"Control": "No message.", "norms": "Most people support the policy.", "fear": "Disasters are coming.",
         "hope": "We can still fix this.", "cost": "It is cheap.", "trust": "Scientists agree."}
TRUTH = {"norms": 2.0, "fear": -1.0, "hope": 1.0, "cost": 0.5, "trust": 3.0}


def responder(hidden: dict, named: dict | None = None):
    """Replies with `named` when the prompt names a study (a '(' right after the framing's experiment noun)."""
    def r(system, user):
        vals = named if (named is not None and "(Study X)" in user) else hidden
        return json.dumps({"effects": vals})
    return r


class Recorder(FakeLLM):
    """FakeLLM that also records the keyword arguments of every call (to see sample=k)."""
    def __init__(self, responder):
        super().__init__(responder)
        self.kws = []

    def complete(self, system, user, *, max_tokens=256, **kw):
        self.kws.append(kw)
        return super().complete(system, user, max_tokens=max_tokens, **kw)


class TestEffectForecast(unittest.TestCase):
    def test_prompt(self):
        p = effect_prompt(CONDS, "policy support (0-100)", scale=(0, 100), named="Study X", context="Online panel.")
        self.assertTrue(p.startswith("A randomized experiment (Study X) tested several interventions"))
        self.assertIn(" Online panel. Each participant", p)
        self.assertIn("Control condition (Control): No message.", p)
        self.assertIn("- fear: Disasters are coming.", p)
        self.assertNotIn("- Control:", p)
        self.assertIn("an effect can range from -100 to 100", p)
        self.assertIn('("norms", "fear", "hope", "cost", "trust")', p)
        h = effect_prompt({"a": "x", "b": "y"}, "y", control="none", framing=FRAMINGS[1])
        self.assertIn("randomized experiment.", h)
        self.assertIn("Control condition (none): no intervention", h)

    def test_hidden_only_calls_and_table(self):
        llm = Recorder(responder(TRUTH))
        res = effect_forecast(llm, CONDS, "policy support", scale=(0, 100))
        self.assertIsInstance(res, EffectForecast)
        self.assertEqual(res.planned, 3 * 2)
        self.assertEqual(len(llm.calls), 6)
        self.assertEqual(sorted(k.get("sample", 0) for k in llm.kws), [0, 0, 0, 1, 1, 1])
        self.assertEqual(list(res.table.condition), list(TRUTH))
        self.assertNotIn("named", res.table)
        self.assertEqual(res.table.set_index("condition").loc["trust", "hidden"], 3.0)
        self.assertTrue((res.table.n_hidden == 6).all())
        self.assertEqual(res.probe["status"], "not_run")
        self.assertEqual((res.refused, res.unparsed, res.failed, res.invalid), (0, 0, 0, 0))
        json.dumps(res.to_dict())
        self.assertEqual(memorisation_check(res, TRUTH)["status"], "not_run")

    def test_named_arm_probe_and_frame(self):
        named_vals = {k: v + 1.0 for k, v in TRUTH.items()}
        llm = FakeLLM(responder(TRUTH, named_vals))
        res = effect_forecast(llm, CONDS, "policy support", framings=2, samples=3, named="Study X")
        self.assertEqual(res.planned, 2 * 2 * 3)
        self.assertEqual(len(llm.calls), 12)
        self.assertEqual(len(res.prompts["named"]), 2)
        self.assertTrue(all("(Study X)" in p for p in res.prompts["named"]))
        self.assertFalse(any("Study X" in p for p in res.prompts["hidden"]))
        self.assertTrue((res.table.named_minus_hidden == 1.0).all())
        self.assertEqual(res.probe["status"], "ok")
        self.assertAlmostEqual(res.probe["mean_abs_diff"], 1.0)
        self.assertAlmostEqual(res.probe["spearman_arms"], 1.0)
        f = effect_frame(res, TRUTH)
        self.assertEqual(list(f.columns), ["unit", "item", "group", "truth", "pred_effect_hidden",
                                           "pred_effect_named"])
        self.assertEqual(f.truth.tolist(), list(TRUTH.values()))
        mc = memorisation_check(res, TRUTH, B=200)
        self.assertEqual(mc["status"], "ok")                  # shifted, same ranks -> no accuracy gap
        self.assertAlmostEqual(mc["named_minus_hidden"], 0.0)

    def test_memorisation_warning(self):
        hidden = {"norms": 3.0, "fear": 2.0, "hope": 0.5, "cost": 1.0, "trust": -1.0}   # wrong order
        llm = FakeLLM(responder(hidden, TRUTH))                                        # named arm "remembers"
        res = effect_forecast(llm, CONDS, "policy support", framings=1, samples=1, named="Study X")
        mc = memorisation_check(res, TRUTH, B=200)
        self.assertEqual(mc["status"], "warning")
        self.assertAlmostEqual(mc["spearman_named"], 1.0)
        self.assertGreaterEqual(mc["named_minus_hidden"], 0.2)

    def test_refused_unparsed_invalid_counted(self):
        def r(system, user):
            if "research forecast" in user:
                return "I cannot provide forecasts for fabricated experiment results."   # still refused
            if "Researchers compared" in user:
                return "I cannot provide forecasts for fabricated experiment results."
            if "You are forecasting" in user:
                return "no json here" if "Return only valid JSON" not in user else "still none"
            return json.dumps({"effects": {"norms": "2,5", "fear": 900, "hope": "n/a", "Cost": 1, "trust": 3}})
        res = effect_forecast(FakeLLM(r), CONDS, "y", scale=(0, 100), samples=1)
        self.assertEqual(res.refused, 1)
        self.assertEqual(res.unparsed, 1)
        t = res.table.set_index("condition")
        self.assertEqual(t.loc["norms", "hidden"], 2.5)
        self.assertTrue(math.isnan(t.loc["fear", "hidden"]))     # |900| > 100: out of range
        self.assertEqual(t.loc["cost", "hidden"], 1.0)            # key matched case-insensitively
        self.assertEqual(res.invalid, 2)                          # fear out of range, hope not a number

    def test_json_retry_recovers(self):
        def r(system, user):
            if "Return only valid JSON" in user:
                return json.dumps({"effects": TRUTH})
            return "Sure! Here are my thoughts."
        res = effect_forecast(FakeLLM(r), CONDS, "y", framings=1, samples=1)
        self.assertEqual((res.unparsed, res.calls), (0, 2))
        self.assertEqual(res.table.hidden.tolist(), list(TRUTH.values()))

    def test_sample_kw_none_and_custom_framings(self):
        llm = Recorder(responder(TRUTH))
        res = effect_forecast(llm, CONDS, "y", framings=["Experiment{study}.{context}"], samples=2, sample_kw=None)
        self.assertEqual([k.get("sample") for k in llm.kws], [None, None])
        self.assertEqual([re.search(r"\(Forecast (\d) of 2\.\)", u).group(1) for _, u in llm.calls], ["1", "2"])
        self.assertTrue(llm.calls[0][1].startswith("Experiment."))
        self.assertEqual(res.planned, 2)

    def test_list_reply_and_validation(self):
        rows = [{"condition": c, "effect": v} for c, v in TRUTH.items()]
        res = effect_forecast(FakeLLM(lambda s, u: json.dumps(rows)), CONDS, "y", framings=1, samples=1)
        self.assertEqual(res.table.hidden.tolist(), list(TRUTH.values()))
        llm = FakeLLM(responder(TRUTH))
        for bad in ({"conditions": {}}, {"conditions": {"Control": "x"}}, {"framings": 0}, {"framings": 4},
                    {"samples": 0}, {"scale": (5, 1)}):
            kw = {"conditions": CONDS, **bad}
            with self.assertRaises(ValueError, msg=str(bad)):
                effect_forecast(llm, kw.pop("conditions"), "y", **kw)


if __name__ == "__main__":
    unittest.main()
