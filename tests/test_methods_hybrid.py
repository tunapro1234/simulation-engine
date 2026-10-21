"""Hybrid forms: shrunk deviation, ridge to prior, dev stack, calibration, LLM-as-feature GBDT."""
import unittest

import numpy as np
import pandas as pd

from simengine.methods import (BinaryCalibrator, DevStack, LLMFeatureGBDT, ShrunkDeviation, lightgbm_available,
                               ridge_to_prior, select_lambda)


class TestShrunkDeviation(unittest.TestCase):
    def setUp(self):
        rs = np.random.default_rng(0)
        self.g = np.repeat(np.arange(10), 20)
        level = rs.normal(0, 1, 10)[self.g]
        self.signal = rs.normal(0, 1, 200)
        self.y = level + self.signal + rs.normal(0, 0.2, 200)
        self.base = level                                   # data knows the group level
        self.llm_good = 3.0 + self.signal                   # LLM: wrong level, right deviation
        self.llm_noise = 3.0 + rs.normal(0, 1, 200)

    def test_alpha_follows_signal(self):
        good = ShrunkDeviation().fit(self.base, self.llm_good, self.y, groups=self.g)
        self.assertGreaterEqual(good.alpha_, 0.8)
        bad = ShrunkDeviation().fit(self.base, self.llm_noise, self.y, groups=self.g)
        self.assertLessEqual(bad.alpha_, 0.1)
        p = good.predict(self.base, self.llm_good, groups=self.g)
        self.assertLess(np.mean((p - self.y) ** 2), np.mean((self.base - self.y) ** 2))

    def test_distribution_mode(self):
        base = np.tile([0.5, 0.3, 0.2], (4, 1))
        llm = np.array([[0.9, 0.05, 0.05]] * 4)
        ref = np.array([0.6, 0.2, 0.2])
        y = np.tile([0.7, 0.2, 0.1], (4, 1))
        sd = ShrunkDeviation(distribution=True).fit(base, llm, y, ref=ref)
        P = sd.predict(base, llm, ref=ref)
        np.testing.assert_allclose(P.sum(1), 1.0)
        self.assertTrue((P >= 0).all())


class TestRidgePrior(unittest.TestCase):
    def test_limits_and_selection(self):
        rs = np.random.default_rng(1)
        Z = rs.normal(size=(50, 3))
        y = Z @ np.array([1.0, -2.0, 0.5]) + rs.normal(0, 0.1, 50)
        t0 = np.array([5.0, 5.0, 5.0])
        np.testing.assert_allclose(ridge_to_prior(Z, y, t0, 1e9), t0, atol=1e-4)
        np.testing.assert_allclose(ridge_to_prior(Z, y, t0, 1e-9), [1.0, -2.0, 0.5], atol=0.1)
        Zv = rs.normal(size=(50, 3))
        yv = Zv @ np.array([1.0, -2.0, 0.5])
        lam, scores = select_lambda(lambda l: Zv @ ridge_to_prior(Z, y, t0, l), yv)
        self.assertEqual(lam, min(scores, key=scores.get))
        self.assertLessEqual(lam, 3)


class TestStackCalibration(unittest.TestCase):
    def test_dev_stack_binary_and_continuous(self):
        rs = np.random.default_rng(2)
        n = 400
        z = rs.normal(size=n)
        y = (rs.random(n) < 1 / (1 + np.exp(-2 * z))).astype(int)
        p_good = 1 / (1 + np.exp(-2 * z))
        p_bad = rs.uniform(0.05, 0.95, n)
        st = DevStack("binary").fit([p_good, p_bad], y)
        self.assertEqual(st.dev_pred_.shape, (n,))
        pt = st.predict([p_good[:10], p_bad[:10]])
        self.assertTrue(((pt > 0) & (pt < 1)).all())
        yc = 2 * z + rs.normal(0, 0.1, n)
        sc = DevStack("continuous").fit([z, rs.normal(size=n)], yc)
        self.assertGreater(np.corrcoef(sc.dev_pred_, yc)[0, 1], 0.95)
        ym = np.where(z > 0.5, 2, np.where(z < -0.5, 0, 1))
        P = np.column_stack([(ym == k) * 0.8 + 0.1 for k in range(3)])
        P = P / P.sum(1, keepdims=True)
        sm = DevStack("multiclass").fit([P], ym)
        self.assertEqual(sm.dev_pred_.shape, (n, 3))

    def test_calibrator(self):
        rs = np.random.default_rng(3)
        p = rs.uniform(0.01, 0.99, 3000)
        y = (rs.random(3000) < p ** 2).astype(int)           # miscalibrated
        iso = BinaryCalibrator().fit(p, y)
        self.assertEqual(iso.method_, "isotonic")
        q = iso.predict(np.array([0.5]))
        self.assertLess(abs(q[0] - 0.25), 0.08)
        pl = BinaryCalibrator().fit(p[:500], y[:500])
        self.assertEqual(pl.method_, "platt")
        self.assertLess(pl.predict(np.array([0.5]))[0], 0.45)


@unittest.skipUnless(lightgbm_available(), "lightgbm not installed")
class TestLLMFeature(unittest.TestCase):
    def test_fit_predict(self):
        rs = np.random.default_rng(4)
        X = pd.DataFrame({"a": rs.normal(size=600)})
        llm = rs.normal(size=600)
        y = (X.a + 2 * llm > 0).astype(int)
        m = LLMFeatureGBDT(n_estimators=80).fit(X[:400], y[:400], llm[:400],
                                                 eval_set=(X[400:500], y[400:500], llm[400:500]))
        p = m.predict_proba(X[500:], llm[500:])
        self.assertGreater(np.mean((p > 0.5) == y[500:]), 0.85)
        self.assertIn("llm_pred", m.feature_importance(2))


if __name__ == "__main__":
    unittest.main()
