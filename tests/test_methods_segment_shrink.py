"""Shrunk segment empirical distributions (LLM-free ladder T2b) and the m selector, on synthetic data."""
import unittest

import numpy as np
import pandas as pd

from simengine.methods import segment_distributions
from simengine.methods.baselines import ShrinkSelection, segment_empirical, select_m


def _tvd(p, q):
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum()


class SegmentShrinkTest(unittest.TestCase):
    def setUp(self):
        # segment x: 3 rows all "a"; segment y: 1 row "b" -> overall [0.75, 0.25]
        self.df = pd.DataFrame({"v": ["a", "a", "a", "b"], "s": ["x", "x", "x", "y"],
                                "t": ["p", "q", "p", "q"]})

    def test_formula(self):
        d = segment_distributions(self.df, "v", "s", ["a", "b"], mode="shrunk", m=2)
        np.testing.assert_allclose(d["x"], (3 * np.array([1, 0]) + 2 * np.array([0.75, 0.25])) / 5)
        np.testing.assert_allclose(d["y"], (1 * np.array([0, 1]) + 2 * np.array([0.75, 0.25])) / 3)
        for v in d.values():
            self.assertAlmostEqual(v.sum(), 1.0)

    def test_limits_match_own_and_overall(self):
        own = segment_distributions(self.df, "v", "s", ["a", "b"], mode="own")
        ov = segment_distributions(self.df, "v", "s", ["a", "b"], mode="overall")
        m0 = segment_distributions(self.df, "v", "s", ["a", "b"], mode="shrunk", m=0)
        big = segment_distributions(self.df, "v", "s", ["a", "b"], mode="shrunk", m=1e9)
        for s in ("x", "y"):
            np.testing.assert_allclose(m0[s], own[s])
            np.testing.assert_allclose(big[s], ov[s], atol=1e-8)

    def test_alias_and_backwards_compat(self):
        a = segment_empirical(self.df, "v", "s", ["a", "b"], shrink=2)
        b = segment_distributions(self.df, "v", "s", ["a", "b"], mode="shrunk", m=2)
        np.testing.assert_allclose(a["y"], b["y"])
        np.testing.assert_allclose(segment_empirical(self.df, "v", "s", ["a", "b"])["x"], [1, 0])
        # old positional call still works
        self.assertEqual(segment_distributions(self.df, "v", "s", ["a", "b"], None, "own")["y"].tolist(), [0, 1])
        with self.assertRaises(ValueError):
            segment_distributions(self.df, "v", "s", ["a", "b"], mode="nope")

    def test_cross_segments_prior_and_missing(self):
        d = segment_distributions(self.df, "v", ["s", "t"], ["a", "b"], mode="shrunk", m=1,
                                  segments=[("x", "p"), ("y", "q"), ("y", "p")])
        np.testing.assert_allclose(d[("x", "p")], (2 * np.array([1, 0]) + np.array([0.75, 0.25])) / 3)
        np.testing.assert_allclose(d[("y", "p")], [0.75, 0.25])           # no rows -> the target itself
        own = segment_distributions(self.df, "v", ["s", "t"], ["a", "b"], mode="own")
        self.assertEqual(set(own), {("x", "p"), ("x", "q"), ("y", "q")})
        oth = segment_distributions(self.df, "v", ["s", "t"], ["a", "b"], mode="others")
        np.testing.assert_allclose(oth[("y", "q")], [1, 0])
        # hierarchical: cross cell shrunk toward its one-column shrunk parent
        parent = segment_distributions(self.df, "v", "s", ["a", "b"], mode="shrunk", m=1)
        h = segment_distributions(self.df, "v", ["s", "t"], ["a", "b"], mode="shrunk", m=1,
                                  prior={k: parent[k[0]] for k in own})
        np.testing.assert_allclose(h[("y", "q")], (np.array([0, 1]) + parent["y"]) / 2)
        fixed = segment_distributions(self.df, "v", "s", ["a", "b"], mode="shrunk", m=1, prior=[0.5, 0.5])
        np.testing.assert_allclose(fixed["y"], [0.25, 0.75])

    def test_weights_use_effective_n(self):
        df = self.df.assign(w=[1.0, 1.0, 2.0, 1.0])
        d = segment_distributions(df, "v", "s", ["a", "b"], weight_col="w", mode="shrunk", m=1)
        overall = np.array([4, 1]) / 5
        n_eff = 4 ** 2 / 6
        np.testing.assert_allclose(d["x"], (n_eff * np.array([1, 0]) + overall) / (n_eff + 1))

    def _synthetic(self, n_seg=8, per_seg=15, spread=0.15, seed=0):
        """Segments whose true answer shares sit close to a common base: shrinkage should help."""
        rng = np.random.default_rng(seed)
        base = np.array([0.5, 0.3, 0.2])
        rows, truth = [], {}
        for s in range(n_seg):
            p = np.clip(base + rng.normal(0, spread, 3) * base, 0.01, None)
            p /= p.sum()
            truth[s] = p
            for i in range(per_seg):
                rows.append({"seg": s, "hh": f"{s}-{i // 3}", "q1": rng.choice(list("abc"), p=p),
                             "q2": rng.choice(list("abc"), p=p[::-1])})
        return pd.DataFrame(rows), truth

    def test_select_m_prefers_shrinkage_and_beats_own(self):
        df, truth = self._synthetic()
        sel = select_m(df, ["q1", "q2"], "seg", list("abc"), unit_col="hh", folds=5)
        self.assertIsInstance(sel, ShrinkSelection)
        self.assertGreater(sel.m, 0)
        self.assertGreater(sel.scores[sel.m], sel.scores[0])
        self.assertGreater(sel.n_cells, 0)
        # the chosen m is closer to the truth than raw segment empirical (q1)
        own = segment_distributions(df, "q1", "seg", list("abc"), mode="own")
        shr = segment_distributions(df, "q1", "seg", list("abc"), mode="shrunk", m=sel.m)
        self.assertLess(np.mean([_tvd(shr[s], truth[s]) for s in truth]),
                        np.mean([_tvd(own[s], truth[s]) for s in truth]))
        # deterministic for a fixed seed; per-item categories dict accepted
        again = select_m(df, ["q1", "q2"], "seg", {"q1": list("abc"), "q2": list("abc")}, unit_col="hh")
        self.assertEqual(again.m, sel.m)

    def test_select_m_loo_and_errors(self):
        df, _ = self._synthetic(n_seg=3, per_seg=6)
        sel = select_m(df, "q1", "seg", list("abc"), unit_col="hh", folds=None, grid=(0, 5, 50))
        self.assertIn(sel.m, (0.0, 5.0, 50.0))
        self.assertEqual(set(sel.scores), {0, 5, 50})
        with self.assertRaises(ValueError):
            select_m(df, "q1", "seg", list("abc"), folds=1)
        with self.assertRaises(ValueError):
            select_m(df, "q1", "seg", list("abc"), unit_col="hh", folds=2, min_eval=10 ** 6)


if __name__ == "__main__":
    unittest.main()
