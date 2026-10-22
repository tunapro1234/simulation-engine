"""EarlyTestBayes API: posterior() summary with credible interval, no-information warning, return conventions."""
import unittest
import warnings

import numpy as np

from simengine.methods import EarlyTestBayes, early_lor, true_lor


def _finished(n=200, seed=0):
    rs = np.random.default_rng(seed)
    lor = rs.normal(0, 0.3, n)
    prior = 1 / (1 + np.exp(-(lor + rs.normal(0, 0.2, n))))
    return prior, lor


class TestPosterior(unittest.TestCase):
    def setUp(self):
        prior, lor = _finished()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.b = EarlyTestBayes().fit(prior, lor)

    def test_scalar(self):
        r = self.b.posterior(0.7, 3, 100, 1, 100)
        self.assertEqual(set(r), {"lor", "lo", "hi", "p_better", "var"})
        self.assertIsInstance(r["lor"], float)
        self.assertLess(r["lo"], r["lor"])
        self.assertLess(r["lor"], r["hi"])
        self.assertTrue(0 <= r["p_better"] <= 1)
        self.assertAlmostEqual(r["p_better"], float(self.b.predict(0.7, 3, 100, 1, 100)))

    def test_array_and_alpha(self):
        pp = np.array([0.2, 0.5, 0.8])
        ca, cb = np.array([1, 2, 5]), np.array([4, 2, 1])
        r = self.b.posterior(pp, ca, 100, cb, 100)
        self.assertEqual(r["lor"].shape, (3,))
        self.assertTrue(np.all(r["lo"] < r["lor"]) and np.all(r["lor"] < r["hi"]))
        self.assertTrue(np.all((r["p_better"] >= 0) & (r["p_better"] <= 1)))
        np.testing.assert_allclose(r["p_better"], self.b.predict(pp, ca, 100, cb, 100))
        np.testing.assert_array_equal(r["lor"] > 0, self.b.decide(pp, ca, 100, cb, 100))
        r90 = self.b.posterior(pp, ca, 100, cb, 100, alpha=0.10)
        self.assertTrue(np.all(r90["hi"] - r90["lo"] < r["hi"] - r["lo"]))
        with self.assertRaises(ValueError):
            self.b.posterior(pp, ca, 100, cb, 100, alpha=1.5)

    def test_moments_kept(self):
        m, v = self.b.posterior_moments(0.7, 3, 100, 1, 100)
        r = self.b.posterior(0.7, 3, 100, 1, 100)
        self.assertAlmostEqual(float(m), r["lor"])
        self.assertAlmostEqual(float(v), r["var"])


class TestNoInformationWarning(unittest.TestCase):
    def test_constant_half(self):
        _, lor = _finished()
        with self.assertWarnsRegex(RuntimeWarning, "prior carries no information; k_=0"):
            b = EarlyTestBayes().fit(np.full(len(lor), 0.5), lor)
        self.assertEqual(b.k_, 0.0)
        # k_=0: prior mean 0, so the posterior only shrinks the early estimate toward 0 (same sign, smaller size)
        lor_e, _ = early_lor(3, 100, 1, 100)
        r = b.posterior(0.5, 3, 100, 1, 100)
        self.assertEqual(np.sign(r["lor"]), np.sign(float(lor_e)))
        self.assertLessEqual(abs(r["lor"]), abs(float(lor_e)))

    def test_constant_other(self):
        _, lor = _finished()
        with self.assertWarnsRegex(RuntimeWarning, "prior carries no information"):
            EarlyTestBayes().fit(np.full(len(lor), 0.7), lor)

    def test_informative_no_warning(self):
        prior, lor = _finished()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            EarlyTestBayes().fit(prior, lor)


class TestReturnConventions(unittest.TestCase):
    def test_early_lor_var_and_se(self):
        lor, var = early_lor(3, 100, 1, 100)
        lor2, se = early_lor(3, 100, 1, 100, return_se=True)
        self.assertAlmostEqual(float(lor), float(lor2))
        self.assertAlmostEqual(float(se) ** 2, float(var))

    def test_true_lor_se_and_var(self):
        lor, se = true_lor(0.03, 0.02, 300, 200)
        _, var = true_lor(0.03, 0.02, 300, 200, return_var=True)
        self.assertAlmostEqual(float(se) ** 2, float(var))
        self.assertAlmostEqual(float(se), np.sqrt(1 / 300.5 + 1 / 200.5))


if __name__ == "__main__":
    unittest.main()
