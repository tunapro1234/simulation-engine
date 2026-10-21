import unittest

import numpy as np
import pandas as pd

import simengine.abm as A


def synthetic_panel(n=600, n_items=5, k=4, seed=0):
    """4 groups; wave B = wave A shifted by a group-specific amount that is shared across items (+ noise)."""
    rng = np.random.default_rng(seed)
    groups = np.array(["D-young", "D-old", "R-young", "R-old"])
    g = groups[rng.integers(0, 4, n)]
    shift = {"D-young": 1, "D-old": 0, "R-young": -1, "R-old": 0}
    df = pd.DataFrame(dict(segment=g, party=[s[0] for s in g], w=rng.uniform(0.5, 1.5, n)))
    items = {}
    for j in range(n_items):
        base = np.where(df.party == "D", 1, 3) + rng.integers(0, 2, n)
        a = np.clip(base, 1, k)
        move = np.array([shift[s] for s in g]) * (rng.random(n) < 0.6)
        b = np.clip(a + move, 1, k).astype(float)
        a = a.astype(float)
        a[rng.random(n) < 0.05] = np.nan
        df[f"q{j}_a"], df[f"q{j}_b"] = a, b
        items[f"q{j}"] = (f"q{j}_a", f"q{j}_b", k)
    return A.OpinionPanel(df, items, group="segment", weight="w", homophily="party")


class TestHelpers(unittest.TestCase):
    def test_shift_distribution(self):
        p = np.array([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(A.shift_distribution(p, 0.0), p)
        q = A.shift_distribution(p, 0.2)
        self.assertAlmostEqual(q.sum(), 1.0)
        self.assertGreater(A.mean_position(q), A.mean_position(p))

    def test_w1(self):
        self.assertAlmostEqual(A.w1([1, 0, 0], [0, 0, 1]), 1.0)
        self.assertAlmostEqual(A.w1([0.5, 0.5], [0.5, 0.5]), 0.0)


class TestOpinion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = synthetic_panel()
        cls.base = A.opinion_baselines(cls.panel)
        grid = [(0, 0.2, 0.5, 0, 0), (0, 0.2, 0.5, 0, 1.0), (0, 0.2, 0.5, 0.05, 1.0), (0.3, 0.2, 0.9, 0, 1.0)]
        cls.abm = A.BoundedConfidenceABM(grid=grid, seed=1)
        cls.pred = cls.abm.fit_predict(cls.panel)

    def test_distributions_sum_to_one(self):
        for d in (self.base, self.pred):
            s = d.groupby(["item", "group", "method"]).p.sum()
            np.testing.assert_allclose(s.to_numpy(), 1.0, atol=1e-9)

    def test_mean_shift_beats_persistence_when_shift_is_shared(self):
        summ, cells = A.score_opinion(self.base, self.panel)
        w = dict(zip(summ.method, summ.W1))
        self.assertLess(w["mean_shift"], w["persistence"])
        self.assertEqual(set(cells.method), {"persistence", "mean_shift", "transition"})

    def test_abm_methods_and_calibration_log(self):
        self.assertEqual(set(self.pred.method), {"abm_bounded_confidence", "abm_shock_noise"})
        self.assertEqual(len(self.abm.calibration), len(self.panel.items))
        # the shock channel is the right model here, so calibration should pick kappa > 0 somewhere
        self.assertTrue(any(r["kappa"] > 0 for r in self.abm.calibration))

    def test_eval_frame(self):
        ev = A.to_eval_frame(pd.concat([self.base, self.pred]), self.panel)
        self.assertEqual(len(ev), len(self.panel.items) * len(self.panel.groups))
        self.assertIn("pred_mean_shift", ev.columns)
        self.assertEqual(len(ev.truth.iloc[0]), 4)


if __name__ == "__main__":
    unittest.main()
