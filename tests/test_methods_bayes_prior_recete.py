"""bayes_prior recipe gaps: per-pair results, paired prior comparison, weak-prior guard + tiebreak mode, censored
multiplier, card section."""
import unittest
import warnings

import numpy as np

from simengine.evidence import section_markdown
from simengine.methods.bayes_prior import compare_priors, saving_section, simulate_saving_calibrated

NS = (25, 50, 100, 200, 400)
KW = dict(ns=NS, reps=20, draws=2, n_boot=60)


def _pairs(n=150, noise=0.15, seed=3):
    rs = np.random.default_rng(seed)
    base = rs.uniform(0.01, 0.03, n)
    lor = rs.normal(0, 0.25, n)
    ctr_b = 1 / (1 + np.exp(-(np.log(base / (1 - base)) - lor)))
    prior = 1 / (1 + np.exp(-(lor + rs.normal(0, noise, n))))
    return base, ctr_b, prior


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


class TestReturnPairs(unittest.TestCase):
    def test_pair_means_equal_aggregate(self):
        a, b, p = _pairs()
        r = _quiet(simulate_saving_calibrated, a, b, p, return_pairs=True, **KW)
        for n in NS:
            self.assertEqual(r["hybrid_pairs"][str(n)].shape, (150,))
            self.assertAlmostEqual(float(r["hybrid_pairs"][str(n)].mean()), r["hybrid"][str(n)], delta=1e-9)
            self.assertAlmostEqual(float(r["human_pairs"][str(n)].mean()), r["human"][str(n)], delta=1e-9)
        d = r["decisions_n0"]
        self.assertEqual(d.shape, (2, 20, 150))
        self.assertEqual(d.dtype, bool)
        self.assertAlmostEqual(float(np.mean(d == r["a_better"])), r["hybrid"]["100"], delta=1e-9)

    def test_default_has_no_pair_arrays(self):
        a, b, p = _pairs()
        r = _quiet(simulate_saving_calibrated, a, b, p, **KW)
        self.assertNotIn("hybrid_pairs", r)
        self.assertIsInstance(r["warnings"], list)


class TestComparePriors(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.good = _pairs(noise=0.15)
        self.weak = _pairs(noise=2.0, seed=4)[2]

    def test_matches_single_calls_and_pairs(self):
        r = _quiet(compare_priors, {"good": self.good, "weak": self.weak}, self.a, self.b, **KW)
        self.assertEqual(r["reference"], "good")
        for name, p in (("good", self.good), ("weak", self.weak)):
            s = _quiet(simulate_saving_calibrated, self.a, self.b, p, **KW)
            self.assertEqual(r["priors"][name]["hybrid"], s["hybrid"])
            self.assertEqual(r["priors"][name]["multiplier"], s["multiplier"])
            self.assertEqual(r["priors"][name]["ci"], s["ci"])
        d = r["diffs"]["weak"]
        self.assertEqual(d["vs"], "good")
        self.assertNotIn("good", r["diffs"])
        self.assertAlmostEqual(d["accuracy"], r["priors"]["weak"]["hybrid"]["100"] - r["priors"]["good"]["hybrid"]["100"])
        self.assertLess(d["accuracy_ci"][1], 0)          # the weak prior is clearly worse on the same replays
        self.assertLessEqual(d["accuracy_ci"][0], d["accuracy"])
        self.assertGreater(r["vs_human"]["good"]["accuracy_ci"][0], 0)

    def test_clustered_bootstrap(self):
        test_id = np.arange(150) // 3
        r = _quiet(compare_priors, {"good": self.good, "weak": self.weak}, self.a, self.b, cluster=test_id, **KW)
        self.assertIn("50 clusters", r["bootstrap"])
        with self.assertRaises(ValueError):
            compare_priors({"good": self.good}, self.a, self.b, cluster=test_id[:10], **KW)
        with self.assertRaises(ValueError):
            compare_priors({}, self.a, self.b, **KW)


class TestWeakPriorGuard(unittest.TestCase):
    def test_warns_on_weak_prior(self):
        a, b, _ = _pairs()
        weak = _pairs(noise=3.0, seed=7)[2]
        with self.assertWarnsRegex(RuntimeWarning, "weak prior"):
            r = simulate_saving_calibrated(a, b, weak, **KW)
        self.assertTrue(r["warnings"])
        self.assertIn("k_calib", r)

    def test_no_warning_on_strong_prior(self):
        a, b, p = _pairs()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = simulate_saving_calibrated(a, b, p, **KW)
        self.assertEqual(r["warnings"], [])
        self.assertGreater(r["k_calib"]["lo"], 0)

    def test_tiebreak_never_below_human(self):
        # weakly informative prior (about 0.6 right): the Bayes hybrid may lose, tiebreak must not
        for seed in range(4):
            rs = np.random.default_rng(seed)
            a, b, _ = _pairs(seed=seed)
            y = a > b
            weak = np.where(rs.random(150) < 0.6, np.where(y, 0.6, 0.4), np.where(y, 0.4, 0.6))
            r = _quiet(simulate_saving_calibrated, a, b, weak, mode="tiebreak", **KW)
            for n in NS:
                self.assertGreaterEqual(r["hybrid"][str(n)], r["human"][str(n)], (seed, n))
            self.assertAlmostEqual(r["hybrid"]["100"], r["human_decides_prior_breaks_ties"], delta=1e-12)
        with self.assertRaises(ValueError):
            simulate_saving_calibrated(a, b, weak, mode="nope", **KW)

    def test_tiebreak_warns_when_wrong_on_ties(self):
        a, b, _ = _pairs()
        wrong = np.where(a > b, 0.4, 0.6)
        with self.assertWarnsRegex(RuntimeWarning, "tied at n0"):
            r = simulate_saving_calibrated(a, b, wrong, mode="tiebreak", **KW)
        self.assertLess(r["hybrid"]["100"], r["human"]["100"])


class TestBootCalibration(unittest.TestCase):
    def test_points_unchanged_ci_recalibrated(self):
        a, b, good = _pairs()
        weak = _pairs(noise=2.0, seed=4)[2]
        on = _quiet(compare_priors, {"good": good, "weak": weak}, a, b, **KW)
        off = _quiet(compare_priors, {"good": good, "weak": weak}, a, b, boot_calibration=False, **KW)
        self.assertIn("recalibrated per replicate", on["bootstrap"])
        self.assertIn("calibration held fixed", off["bootstrap"])
        for k in ("good", "weak"):
            self.assertEqual(on["priors"][k]["hybrid"], off["priors"][k]["hybrid"])
            self.assertEqual(on["priors"][k]["multiplier"], off["priors"][k]["multiplier"])
        self.assertEqual(on["diffs"]["weak"]["accuracy"], off["diffs"]["weak"]["accuracy"])
        self.assertNotEqual(on["diffs"]["weak"]["accuracy_ci"], off["diffs"]["weak"]["accuracy_ci"])
        # held fixed: replicate accuracy is the resampled mean of the per-pair accuracy (the research script's CI)
        s = _quiet(simulate_saving_calibrated, a, b, weak, boot_calibration=False, return_pairs=True, **KW)
        self.assertEqual(s["ci"], off["priors"]["weak"]["ci"])
        tb = _quiet(simulate_saving_calibrated, a, b, weak, mode="tiebreak", **KW)
        self.assertNotIn("calibration", tb["bootstrap"])


class TestCensoring(unittest.TestCase):
    def test_above_curve(self):
        a, b, p = _pairs(noise=0.02)
        r = _quiet(simulate_saving_calibrated, a, b, p, **KW)
        self.assertIsNone(r["multiplier"])
        self.assertTrue(r["multiplier_censored"])
        self.assertEqual(r["multiplier_lower_bound"], 4.0)          # max(ns) / n0 = 400 / 100
        self.assertTrue(r["ci_censored"][1])
        self.assertIsNone(r["ci"][1])

    def test_not_censored(self):
        a, b, p = _pairs(n=150, noise=0.4, seed=6)
        r = _quiet(simulate_saving_calibrated, a, b, p, ns=(25, 50, 100, 200, 400, 800, 1600), reps=20, draws=2,
                   n_boot=60)
        self.assertFalse(r["multiplier_censored"])
        self.assertIsNone(r["multiplier_lower_bound"])
        self.assertIsNotNone(r["multiplier"])


class TestSavingSection(unittest.TestCase):
    def test_single(self):
        a, b, p = _pairs(noise=0.02)
        r = _quiet(simulate_saving_calibrated, a, b, p, **KW)
        s = saving_section(r)
        self.assertEqual(set(s) >= {"kind", "title", "columns", "rows", "notes"}, True)
        self.assertEqual(s["kind"], "ab_saving")
        md = section_markdown(s)
        self.assertIn("impression multiplier", md)
        self.assertIn("> 4x", md)
        self.assertIn("calibration tests (n_calib) | 20", md)
        self.assertIn("CI: 95% bootstrap (pair, recalibrated per replicate)", md)
        self.assertTrue(s["multiplier_censored"])

    def test_compare(self):
        a, b, p = _pairs()
        weak = _pairs(noise=2.0, seed=4)[2]
        r = _quiet(compare_priors, {"good": p, "weak": weak}, a, b, **KW)
        s = saving_section(r, title="Priors")
        self.assertEqual(s["title"], "Priors")
        self.assertEqual([row[0] for row in s["rows"]], ["human only", "good", "weak"])
        md = section_markdown(s)
        self.assertIn("vs good", md)


if __name__ == "__main__":
    unittest.main()
