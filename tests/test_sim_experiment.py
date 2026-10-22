"""simengine.sim.experiment: between/within experiments on a panel, exaggeration ratio, LOO correction,
human-equivalent N, card section."""
import hashlib
import json
import math
import re
import unittest

import numpy as np

from simengine.evidence import section_markdown
from simengine.llm import FakeLLM
from simengine.llm.fake import survey_responder
from simengine.sim import (correct_effects, effect_section, experiment, experiment_frame, exaggeration_ratio,
                           hedges_g, human_equivalent_n)
from simengine.sim.experiment import _pct, _stretch, assign_conditions, paired_g

SD = 10.0      # per-person spread of the fake answers (0-100 scale)


def _noise(pid: str) -> float:
    """Deterministic ~N(0, 1) per persona (Box-Muller on two hashes)."""
    h = hashlib.md5(pid.encode()).digest()
    u1 = (int.from_bytes(h[:4], "big") + 1) / 2 ** 32
    u2 = int.from_bytes(h[4:8], "big") / 2 ** 32
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def responder(true_g: dict, mult: float, neutral_mult: float = 0.0):
    """Panel answers = 50 + SD * (noise + mult * true_g[condition]); the condition is read from the
    '[cond=<name>]' tag in the stimulus. The no_persona arm ('typical adult') gets neutral_mult instead (set
    here: survey_responder keeps value_fn out of that arm and gives it the hashed default)."""
    def respond(system, user):
        m = re.search(r"\[cond=([^\]]+)\]", user)
        g = true_g.get(m.group(1), 0.0) if m else 0.0
        neutral = "typical adult" in user.lower()
        val = lambda pid: 50 + SD * (_noise(pid) + (neutral_mult if neutral else mult) * g)   # noqa: E731
        rows = json.loads(survey_responder(lambda pid, qid, spec: val(pid))(system, user))
        if neutral:
            for r in rows:
                for f, v in r.items():
                    if f != "id" and isinstance(v, (int, float)) and not isinstance(v, bool):
                        r[f] = min(100.0, max(0.0, val(str(r["id"]))))
        return json.dumps(rows)
    return respond


def conds(names):
    return {c: f"You read message {c}. [cond={c}]" for c in names}


PERSONAS = [{"id": f"u{i:03d}", "name": f"Person {i}", "age": 20 + i % 50} for i in range(240)]


class TestEffectSizes(unittest.TestCase):
    def test_hedges_g(self):
        a, b = [1, 2, 3, 4, 5], [0, 1, 2, 3, 4]
        r = hedges_g(a, b)
        d = 1 / np.std(a, ddof=1)
        self.assertAlmostEqual(r["d"], d)
        self.assertAlmostEqual(r["g"], d * (1 - 3 / (4 * 8 - 1)))
        self.assertLess(r["ci"][0], r["g"])
        self.assertIsNone(hedges_g([1], [2, 3])["g"])
        self.assertIsNone(hedges_g([1, 1], [1, 1])["g"])

    def test_paired_and_assignment(self):
        r = paired_g([2, 3, 4, 5, 7], [1, 2, 3, 4, 5])
        self.assertGreater(r["g"], 0)
        self.assertEqual(r["n_treat"], 5)
        a = assign_conditions([f"p{i}" for i in range(11)], ["x", "y", "z"], seed=3)
        counts = sorted(list(a.values()).count(c) for c in "xyz")
        self.assertEqual(counts, [3, 4, 4])
        self.assertEqual(a, assign_conditions([f"p{i}" for i in range(11)], ["x", "y", "z"], seed=3))


class TestExperiment(unittest.TestCase):
    def test_between_recovers_multiple(self):
        truth = {"a": 0.0, "b": 0.4}
        llm = FakeLLM(responder(truth, mult=2.0, neutral_mult=0.5))
        res = experiment(llm, PERSONAS, conds(["a", "b"]), "How much do you support it?", scale=(0, 100), seed=1,
                         ask_kw={"batch": 20})
        self.assertEqual(res["design"], "between")
        self.assertEqual(res["control"], "a")
        self.assertEqual(sorted(res["conditions"]), ["a", "b"])
        self.assertEqual(res["conditions"]["a"]["n"] + res["conditions"]["b"]["n"], 240)
        self.assertEqual(res["conditions"]["a"]["assigned"], 120)
        e = res["effects"]["b"]
        self.assertAlmostEqual(e["g"], 0.8, delta=0.3)          # 2 x 0.4
        self.assertAlmostEqual(e["no_persona_g"], 0.2, delta=0.2)      # neutral arm: 0.5 x 0.4
        self.assertLess(e["ci"][0], e["g"])
        self.assertAlmostEqual(e["diff"], 8.0, delta=3.0)
        self.assertEqual(res["refused"], 0)
        self.assertEqual(len(res["assignment"]), 240)
        # every persona prompt shows one condition only
        for _, user in llm.calls:
            self.assertLessEqual(len(set(re.findall(r"\[cond=([^\]]+)\]", user))), 1)

    def test_within_and_validation(self):
        llm = FakeLLM(responder({"a": 0.0, "b": 0.5}, mult=1.0))
        res = experiment(llm, PERSONAS[:40], conds(["a", "b"]), "Support?", scale=(0, 100), between=False,
                         no_persona=False)
        self.assertEqual(res["design"], "within")
        self.assertEqual(res["effects"]["b"]["n_treat"], 40)
        self.assertGreater(res["effects"]["b"]["g"], 0.5)        # paired: noise cancels, d_av ~ 0.5 / 1
        self.assertNotIn("no_persona_g", res["effects"]["b"])
        with self.assertRaises(ValueError):
            experiment(llm, PERSONAS[:4], {"a": "x"}, "q")
        with self.assertRaises(ValueError):
            experiment(llm, PERSONAS[:4], conds(["a", "b"]), "q", control="zzz")
        with self.assertRaises(ValueError):
            experiment(llm, PERSONAS[:4], conds(["a", "b"]), {"text": "q", "kind": "choice", "options": ["x", "y"]})

    def test_unparsed_counted(self):
        def bad(system, user):
            return "no json here"
        res = experiment(FakeLLM(bad), PERSONAS[:10], conds(["a", "b"]), "Support?", scale=(0, 100),
                         no_persona=False, ask_kw={"refusal_retry": False, "empty_retry": False})
        self.assertEqual(res["unparsed"] + res["refused"], 10)
        self.assertIsNone(res["effects"]["b"]["g"])


class TestCalibration(unittest.TestCase):
    """A FakeLLM panel whose effects are 2x the truth: the ratio is recovered and the correction lowers MAE."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.studies = {}
        cls.human = {}
        truth_all = {}
        for s in range(4):
            names = ["ctl"] + [f"s{s}c{j}" for j in range(3)]
            truth = {"ctl": 0.0, **{n: float(rng.uniform(0.15, 0.6) * rng.choice([-1, 1])) for n in names[1:]}}
            truth_all.update(truth)
            llm = FakeLLM(responder(truth, mult=2.0))
            cls.studies[f"study{s}"] = experiment(llm, PERSONAS, conds(names), "Support?", scale=(0, 100), seed=s,
                                                  no_persona=False, ask_kw={"batch": 30})
            cls.human[f"study{s}"] = {n: {"g": truth[n], "se": 0.05} for n in names[1:]}
        cls.frame = experiment_frame(cls.studies, human=cls.human)

    def test_frame(self):
        f = self.frame
        self.assertEqual(len(f), 12)
        self.assertEqual(list(f.columns[:4]), ["unit", "item", "group", "truth"])
        self.assertFalse(f.truth.isna().any())
        self.assertTrue((f.truth_se == 0.05).all())
        one = experiment_frame(self.studies["study0"], human={"s0c0": 0.3})
        self.assertEqual(one.truth.notna().sum(), 1)

    def test_ratio_recovered_and_correction(self):
        f = self.frame
        fit = exaggeration_ratio(f.pred_g, f.truth, family=f.group, B=400)
        self.assertAlmostEqual(fit["ratio"], 2.0, delta=0.5)
        self.assertAlmostEqual(fit["slope"], 2.0, delta=0.5)
        self.assertEqual(fit["verdict"], "exaggerates")
        self.assertEqual(fit["loo"]["by"], "family")
        self.assertLess(fit["loo"]["mae_ratio"], fit["loo"]["mae_raw"] / 2)
        self.assertLess(fit["loo"]["mae_affine"], fit["loo"]["mae_raw"])
        corr = correct_effects(f.pred_g, human_g=f.truth, family=f.group)
        self.assertLess(np.mean(np.abs(corr - f.truth)), np.mean(np.abs(f.pred_g - f.truth)))
        # applying a measured fit to new effects
        np.testing.assert_allclose(correct_effects(np.array([1.0, -2.0]), 2.0), [0.5, -1.0])
        np.testing.assert_allclose(correct_effects([1.0], fit, method="affine"),
                                   [fit["affine"]["a"] + fit["affine"]["b"]])

    def test_loo_never_uses_own_value(self):
        sim = {"x": 2.0, "y": 4.0, "z": 6.0, "w": 100.0}
        h1 = {"x": 1.0, "y": 2.0, "z": 3.0, "w": 1.0}
        h2 = dict(h1, w=50.0)       # only w's own human value changes
        c1, c2 = correct_effects(sim, human_g=h1), correct_effects(sim, human_g=h2)
        self.assertAlmostEqual(c1["w"], c2["w"])
        self.assertAlmostEqual(c1["w"], 50.0)

    def test_human_equivalent_n(self):
        rng = np.random.default_rng(0)
        h = rng.normal(0, 0.4, 400)
        sim = h + rng.normal(0, 0.1, 400)                  # error var 0.01
        r = human_equivalent_n(sim, h, 0.0, B=300)
        expect = (4 + np.mean(h ** 2) / 2) / np.mean((sim - h) ** 2)
        self.assertAlmostEqual(r["n"], expect, delta=1e-6)
        self.assertLess(r["ci"][0], r["n"])
        self.assertGreater(r["ci"][1], r["n"])
        none = human_equivalent_n(sim, h, 0.2, B=100)       # human noise 0.04 > error 0.01
        self.assertIsNone(none["n"])
        self.assertIn("noise", none["note"])
        clustered = human_equivalent_n(sim, h, 0.05, family=np.arange(400) // 10, B=200)
        self.assertGreater(clustered["n"], r["n"])          # noise subtracted -> fewer errors -> more humans

    def test_effect_section(self):
        f = self.frame
        sec = effect_section(f.pred_g, f.truth, f.truth_se, family=f.group, B=300)
        self.assertEqual(sec["kind"], "effect_calibration")
        self.assertEqual(sec["columns"], ["measure", "value", "95% CI", "note"])
        labels = [r[0] for r in sec["rows"]]
        self.assertTrue(any(lab.startswith("worth ~N humans (ratio-corrected") for lab in labels))
        self.assertTrue(any(lab.startswith("baseline") for lab in labels))
        self.assertGreater(sec["human_n"]["corrected"]["n"], sec["human_n"]["raw"]["n"])
        md = section_markdown(sec)
        self.assertIn("exaggeration ratio", md)
        no_se = effect_section(f.pred_g, f.truth, family=f.group, B=100)
        self.assertTrue(any("human_se" in n for n in no_se["notes"]))

    def test_few_family_ci_and_corrected_slope(self):
        rng = np.random.default_rng(3)
        th = rng.choice([-1, 1], 60) * rng.uniform(0.1, 0.6, 60)
        fam = np.repeat(np.arange(20), 3)
        sim, h = 2 * th + rng.normal(0, 0.1, 60), th + rng.normal(0, 0.15, 60)    # noisy human effects
        fit = exaggeration_ratio(sim, h, fam, human_se=0.15, B=300)
        self.assertLess(fit["slope"], fit["slope_corrected"] - 0.15)           # OLS attenuated by the human noise
        self.assertAlmostEqual(fit["slope_corrected"], 2.0, delta=0.25)
        self.assertLess(fit["slope_corrected_ci"][0], fit["slope_corrected"])
        self.assertAlmostEqual(fit["reliability"], 1 - 0.15 ** 2 / np.var(h, ddof=1))
        # few clusters: the percentile CI is pushed away from the estimate by t(G-1) / z * sqrt(G / (G-1))
        self.assertAlmostEqual(_stretch(3, 0.05), 4.302653 / 1.959964 * math.sqrt(1.5), places=4)
        self.assertLess(_stretch(1000, 0.05), 1.01)
        d = list(np.linspace(0, 1, 101))
        plain, wide = _pct(d, 0.05), _pct(d, 0.05, 0.5, 3)
        self.assertAlmostEqual(wide[1] - 0.5, _stretch(3, 0.05) * (plain[1] - 0.5))
        few = exaggeration_ratio(sim[:9], h[:9], fam[:9], B=300)
        self.assertLess(few["ratio_ci"][0], few["ratio"])
        self.assertGreater(few["ratio_ci"][1], few["ratio"])
        self.assertIsNone(exaggeration_ratio(sim, h, fam, human_se=1.0, B=50)["slope_corrected"])

    def test_section_within_noise_is_not_worse_than_baseline(self):
        rng = np.random.default_rng(5)
        h = rng.normal(0, 0.4, 30)
        fam = np.repeat(np.arange(10), 3)
        sec = effect_section(h + rng.normal(0, 0.02, 30), h, 0.1, family=fam, B=200)   # error far below se^2
        self.assertIsNone(sec["human_n"]["raw"]["n"])
        self.assertIsNotNone(sec["human_n"]["baseline"]["n"])
        self.assertFalse(any("not worth more humans" in n for n in sec["notes"]))

    def test_too_few(self):
        self.assertEqual(exaggeration_ratio([1, 2], [1, 1])["verdict"], "too few")
        with self.assertRaises(ValueError):
            exaggeration_ratio([1, 2, 3], [1, 2])


if __name__ == "__main__":
    unittest.main()
