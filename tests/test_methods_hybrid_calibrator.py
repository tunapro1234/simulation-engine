"""BinaryCalibrator ratio method / monotone Platt guard and the ShrunkDeviation distribution recipe."""
import unittest
import warnings

import numpy as np

from simengine.methods import BinaryCalibrator, ShrunkDeviation


def _auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


class TestBinaryCalibrator(unittest.TestCase):
    def setUp(self):
        rs = np.random.default_rng(0)
        self.p = rs.uniform(0.05, 0.6, 400)
        self.y = (rs.random(400) < 0.5 * self.p).astype(int)        # true rate is half the raw level

    def test_ratio_matches_base_rate_and_keeps_order(self):
        cal = BinaryCalibrator("ratio").fit(self.p, self.y)
        q = cal.predict(self.p)
        self.assertAlmostEqual(float(q.mean()), float(self.y.mean()), places=9)
        np.testing.assert_array_equal(np.argsort(q, kind="stable"), np.argsort(self.p, kind="stable"))
        self.assertIsNone(cal.fallback_)

    def test_ratio_clipped(self):
        cal = BinaryCalibrator("ratio").fit([0.1, 0.1], [1, 0])          # scale 5
        q = cal.predict(np.array([0.05, 0.5]))
        np.testing.assert_allclose(q, [0.25, 1.0])

    def test_slope_sign_guard(self):
        # dev labels anti-correlated with p: an unconstrained Platt slope is negative
        p = np.linspace(0.1, 0.9, 300)
        y = (p < 0.5).astype(int)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cal = BinaryCalibrator("platt").fit(p, y)
        self.assertLess(cal.slope_, 0)
        self.assertEqual(cal.method_, "ratio")
        self.assertIn("<= 0", cal.fallback_)
        self.assertTrue(any("invert the ranking" in str(x.message) for x in w))
        rs = np.random.default_rng(1)
        pt = rs.uniform(0.05, 0.95, 200)
        yt = (rs.random(200) < pt).astype(int)
        self.assertAlmostEqual(_auc(yt, cal.predict(pt)), _auc(yt, pt), places=9)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = BinaryCalibrator("platt", monotone=False).fit(p, y)
        self.assertEqual(raw.method_, "platt")
        self.assertLess(_auc(yt, raw.predict(pt)), 0.5)                  # ranking inverted without the guard

    def test_small_dev_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cal = BinaryCalibrator().fit(self.p[:50], self.y[:50])
        self.assertTrue(any("n_dev=50" in str(x.message) for x in w))
        self.assertIn(cal.method_, ("platt", "ratio"))

    def test_auto_unchanged_and_no_warning_on_big_dev(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cal = BinaryCalibrator().fit(self.p, self.y)
        self.assertEqual(cal.method_, "platt")
        self.assertGreater(cal.slope_, 0)
        self.assertFalse(w)
        q = cal.predict(self.p)
        self.assertLess(abs(float(q.mean()) - float(self.y.mean())), 0.01)

    def test_one_class_dev_falls_back_to_ratio(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cal = BinaryCalibrator().fit([0.1, 0.2, 0.3], [0, 0, 0])
        self.assertEqual(cal.method_, "ratio")
        self.assertIn("one class", cal.fallback_)
        self.assertTrue(any("one class" in str(x.message) for x in w))
        np.testing.assert_allclose(cal.predict(np.array([0.2, 0.5])), [0.0, 0.0])

    def test_bad_method(self):
        with self.assertRaises(ValueError):
            BinaryCalibrator("sigmoid")


class TestShrunkDeviationDistribution(unittest.TestCase):
    def test_population_plus_beta_segment_deviation(self):
        rs = np.random.default_rng(2)
        K, n = 4, 60
        pop = np.array([0.4, 0.3, 0.2, 0.1])
        true_dev = rs.normal(0, 0.04, (n, K))
        true_dev -= true_dev.mean(1, keepdims=True)
        y = np.clip(pop + true_dev, 0.01, None)
        y /= y.sum(1, keepdims=True)
        llm_pop = np.array([0.25, 0.25, 0.25, 0.25])                     # LLM level is wrong ...
        llm_seg = llm_pop + 2.0 * true_dev + rs.normal(0, 0.01, (n, K))  # ... but its deviation is 2x the truth
        tr, te = slice(0, 40), slice(40, None)
        sd = ShrunkDeviation(distribution=True, alphas=np.linspace(0, 1.5, 16))
        sd.fit(base=pop, llm=llm_seg[tr], y=y[tr], ref=llm_pop)
        self.assertAlmostEqual(sd.alpha_, 0.5, delta=0.11)
        pred = sd.predict(base=pop, llm=llm_seg[te], ref=llm_pop)
        self.assertEqual(pred.shape, (20, K))
        np.testing.assert_allclose(pred.sum(1), 1.0, atol=1e-9)
        self.assertLess(np.mean((pred - y[te]) ** 2), np.mean((pop - y[te]) ** 2))

    def test_row_aligned_base_and_ref(self):
        rs = np.random.default_rng(3)
        n, K = 30, 3
        base, ref = rs.dirichlet([2] * K, n), rs.dirichlet([2] * K, n)   # one item per row
        llm = ref + 0.5 * (rs.dirichlet([2] * K, n) - 1 / K)
        y = np.clip(base + 0.3 * (llm - ref), 0, None)
        y /= y.sum(1, keepdims=True)
        sd = ShrunkDeviation(distribution=True).fit(base=base, llm=llm, y=y, ref=ref)
        self.assertAlmostEqual(sd.alpha_, 0.3)


if __name__ == "__main__":
    unittest.main()
