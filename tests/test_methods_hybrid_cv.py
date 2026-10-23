"""ShrunkDeviation cv_groups: alpha chosen out of fold (D5 #7), and the finer low end of the default grid."""
import unittest
import warnings

import numpy as np

from simengine.methods import ShrunkDeviation
from simengine.methods.hybrid import ALPHAS


def noisy(n_groups=12, per=8, seed=0):
    """y = base + noise; the LLM deviation is pure noise, so any alpha > 0 fitted in-sample is over-fitting."""
    rs = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    base = rs.normal(0, 1, len(g))
    y = base + rs.normal(0, 1, len(g))
    llm = rs.normal(0, 1, len(g))
    return base, llm, y, g


class TestCrossFittedAlpha(unittest.TestCase):
    def test_cv_loss_not_below_in_sample(self):
        for seed in range(5):
            base, llm, y, g = noisy(seed=seed)
            sd = ShrunkDeviation().fit(base, llm, y, cv_groups=g)
            self.assertGreaterEqual(sd.cv_loss_, sd.losses_[sd.alpha_] - 1e-12, seed)
            self.assertEqual(set(sd.cv_alphas_), set(np.unique(g)))
            self.assertEqual(sd.cv_pred_.shape, y.shape)
            for a in sd.cv_alphas_.values():
                self.assertIn(a, sd.alphas)

    def test_held_out_rows_use_alpha_from_other_groups(self):
        base, llm, y, g = noisy(seed=3)
        sd = ShrunkDeviation().fit(base, llm, y, cv_groups=g)
        dev = ShrunkDeviation.deviation(llm)
        for k in np.unique(g):
            m = g == k
            a = min(ALPHAS, key=lambda a: np.mean((y[~m] - base[~m] - a * dev[~m]) ** 2))  # dev: all rows' mean
            self.assertEqual(sd.cv_alphas_[k], a)
            np.testing.assert_allclose(sd.cv_pred_[m], base[m] + a * dev[m])

    def test_final_alpha_unchanged_by_cv(self):
        base, llm, y, g = noisy(seed=1)
        a0 = ShrunkDeviation().fit(base, llm, y).alpha_
        self.assertEqual(ShrunkDeviation().fit(base, llm, y, cv_groups=g).alpha_, a0)

    def test_one_fold_same_alpha(self):
        base, llm, y, _ = noisy(seed=2)
        plain = ShrunkDeviation().fit(base, llm, y)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sd = ShrunkDeviation().fit(base, llm, y, cv_groups=np.zeros(len(y)))
        self.assertTrue(any("one group" in str(x.message) for x in w))
        self.assertEqual(sd.alpha_, plain.alpha_)
        self.assertEqual(list(sd.cv_alphas_.values()), [plain.alpha_])
        self.assertAlmostEqual(sd.cv_loss_, plain.losses_[plain.alpha_])

    def test_distribution_with_groups_and_ref(self):
        rs = np.random.default_rng(4)
        K, n = 3, 40
        pop = np.array([0.5, 0.3, 0.2])
        llm = rs.dirichlet(np.ones(K) * 5, n)
        y = np.clip(pop + 0.4 * (llm - llm.mean(0)) + rs.normal(0, 0.01, (n, K)), 0, None)
        y /= y.sum(1, keepdims=True)
        g = np.arange(n) % 5
        sd = ShrunkDeviation(distribution=True).fit(base=pop, llm=llm, y=y, ref=llm.mean(0), cv_groups=g)
        self.assertEqual(sd.cv_pred_.shape, (n, K))
        np.testing.assert_allclose(sd.cv_pred_.sum(1), 1.0)
        self.assertTrue(all(0.2 <= a <= 0.6 for a in sd.cv_alphas_.values()), sd.cv_alphas_)

    def test_cv_groups_length_checked(self):
        base, llm, y, g = noisy()
        with self.assertRaises(ValueError):
            ShrunkDeviation().fit(base, llm, y, cv_groups=g[:-1])


class TestAlphaGrid(unittest.TestCase):
    def test_low_end_points(self):
        for a in (0.01, 0.02, 0.05):
            self.assertIn(a, ALPHAS)
        self.assertEqual(list(ALPHAS), sorted(ALPHAS))
        self.assertEqual(ALPHAS[0], 0.0)
        self.assertEqual(ALPHAS[-1], 1.0)
        for a in np.round(np.linspace(0, 1, 11), 2):     # the old grid is kept
            self.assertIn(a, ALPHAS)

    def test_small_true_alpha_found(self):
        rs = np.random.default_rng(0)
        n = 4000
        base, llm = rs.normal(0, 1, n), rs.normal(0, 1, n)
        y = base + 0.03 * llm + rs.normal(0, 0.02, n)
        a = ShrunkDeviation().fit(base, llm, y).alpha_
        self.assertTrue(0.01 <= a <= 0.05, a)                   # the old grid could only say 0 or 0.1

    def test_explicit_alphas_kept(self):
        sd = ShrunkDeviation(alphas=(0, 0.5, 1))
        self.assertEqual(sd.alphas, (0, 0.5, 1))


if __name__ == "__main__":
    unittest.main()
