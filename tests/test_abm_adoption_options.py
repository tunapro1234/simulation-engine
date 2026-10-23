"""AdoptionABC options: scale= (series above 100), group_col= (analogy donors of one kind), scenario delta / anchor."""
import unittest

import numpy as np
import pandas as pd

import simengine.abm as A
from tests.test_abm_adoption import SMALL, synthetic_adoption


def above_100(n=6, seed=0):
    """Subscriptions per 100 people: Bass curves saturating at 150-190."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        t = np.arange(1995, 2021, dtype=float)
        y = A.bass_curve(t, 150 + 8 * i, 0.01, 0.45, 1996 + i % 3) + rng.normal(0, 0.5, len(t))
        rows += [dict(series=f"m{i}", t=tt, y=float(max(yy, 0))) for tt, yy in zip(t, y)]
    return pd.DataFrame(rows)


def two_kinds(seed=0):
    """Kind 'fast': high ceiling, fast imitation. Kind 'slow': ceiling ~25, slow."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(10):
        fast = i % 2 == 0
        m, q = (85.0, 0.6) if fast else (25.0 + i, 0.25)
        t = np.arange(1990, 2021, dtype=float)
        y = A.bass_curve(t, m, 0.01, q, 1990 + i % 3) + rng.normal(0, 0.3, len(t))
        rows += [dict(series=f"k{i}", kind="fast" if fast else "slow", t=tt, y=float(np.clip(yy, 0, 100)))
                 for tt, yy in zip(t, y)]
    return pd.DataFrame(rows)


class TestScale(unittest.TestCase):
    def test_default_is_capped_at_100_scale_is_not(self):
        df = above_100()
        hist, c = df[df.t <= 2006], 2006
        truth = df[(df.series == "m0") & (df.t > c) & (df.t <= c + 8)]["y"].to_numpy()
        h = np.arange(1, 9)
        p0 = A.AdoptionABC(**SMALL).predict(A.AdoptionABC(**SMALL).fit(hist, "m0", c), h)
        m = A.AdoptionABC(scale=2.0, **SMALL)
        p2 = m.predict(m.fit(hist, "m0", c), h)
        self.assertLessEqual(p0["y_pred"].max(), 100.0 + 1e-6)          # m <= 1: the defect
        self.assertGreater(p2["hi80"].max(), 100.0)
        self.assertLess(np.abs(p2["y_pred"] - truth).mean(), np.abs(p0["y_pred"] - truth).mean())

    def test_scale_one_is_default(self):
        df = synthetic_adoption(6, seed=2)
        hist = df[df.t <= 2003]
        for cls in (A.AdoptionABC, A.AdoptionABCAnalogy):
            a, b = cls(**SMALL), cls(scale=1.0, **SMALL)
            pd.testing.assert_frame_equal(a.predict(a.fit(hist, "s1", 2003), [1, 3]),
                                          b.predict(b.fit(hist, "s1", 2003), [1, 3]))

    def test_scale_dict_per_series(self):
        df = above_100()
        hist = df[df.t <= 2006]
        m = A.AdoptionABCAnalogy(scale={s: 2.0 for s in df.series.unique()}, **SMALL)
        p = m.predict(m.fit(hist, "m1", 2006), [1, 5])
        m2 = A.AdoptionABCAnalogy(scale=2.0, **SMALL)
        p2 = m2.predict(m2.fit(hist, "m1", 2006), [1, 5])
        pd.testing.assert_frame_equal(p, p2)
        with self.assertRaises(ValueError):
            A.AdoptionABC(scale=0)

    def test_scale_scenario_in_data_units(self):
        df = above_100()
        m = A.AdoptionABC(scale=2.0, **SMALL)
        mod = m.fit(df[df.t <= 2006], "m0", 2006)
        sc = m.scenario(mod, {"imitation": 1.2}, n_draws=30)
        self.assertGreater(sc["y_pred"].max(), 100.0)


class TestGroupCol(unittest.TestCase):
    def test_donors_only_from_same_kind(self):
        df = two_kinds()
        hist, c = df[df.t <= 2000], 2000
        m = A.AdoptionABCAnalogy(group_col="kind", **SMALL)
        mod = m.fit(hist, "k1", c)
        per = m._build_prior(hist, c)
        slow = [v for s, v in per.items() if s != "k1" and s in set(df[df.kind == "slow"].series)]
        fast = [v for s, v in per.items() if s in set(df[df.kind == "fast"].series)]
        self.assertGreaterEqual(len(slow), 3)
        allowed = np.sum(slow, axis=0) > 0
        self.assertTrue(allowed[mod["b"]].all())
        only_fast = (np.sum(fast, axis=0) > 0) & ~allowed
        self.assertFalse(only_fast[mod["b"]].any())

    def test_group_col_none_is_default_and_missing_column_raises(self):
        df = two_kinds()
        hist = df[df.t <= 2000]
        a, b = A.AdoptionABCAnalogy(**SMALL), A.AdoptionABCAnalogy(group_col=None, **SMALL)
        pd.testing.assert_frame_equal(a.predict(a.fit(hist, "k1", 2000), [1, 4]),
                                      b.predict(b.fit(hist, "k1", 2000), [1, 4]))
        with self.assertRaises(ValueError):
            A.AdoptionABCAnalogy(group_col="nope", **SMALL).fit(hist, "k1", 2000)


class TestScenarioAnchor(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_adoption(6, seed=1)
        self.m = A.AdoptionABC(**SMALL)
        self.mod = self.m.fit(self.df[self.df.t <= 2004], "s2", 2004)

    def test_default_columns_unchanged(self):
        sc = self.m.scenario(self.mod, {"imitation": 1.3}, n_draws=40)
        self.assertEqual(list(sc.columns), ["scenario", "t", "y_pred", "lo80", "hi80"])

    def test_delta_vs_baseline(self):
        sc = self.m.scenario(self.mod, {"imitation": 1.3}, n_draws=40, delta=True)
        base = sc[sc.scenario == "baseline"]
        self.assertTrue((base["delta"] == 0).all())
        alt = sc[sc.scenario != "baseline"]
        self.assertGreater(alt["delta"].iloc[-1], 0)            # more imitation -> higher curve
        pd.testing.assert_frame_equal(sc[["scenario", "t", "y_pred", "lo80", "hi80"]],
                                      self.m.scenario(self.mod, {"imitation": 1.3}, n_draws=40))

    def test_anchor_adds_delta_to_best_forecast(self):
        anchor = pd.DataFrame(dict(t=np.arange(2005, 2015, dtype=float), y_pred=np.linspace(40, 60, 10)))
        sc = self.m.scenario(self.mod, {"imitation": 1.3}, n_draws=40, anchor=anchor)
        alt = sc[sc.scenario != "baseline"].set_index("t")
        a = anchor.set_index("t")["y_pred"]
        np.testing.assert_allclose(alt.loc[a.index, "y_anchor"], a + alt.loc[a.index, "delta"])
        base = sc[sc.scenario == "baseline"].set_index("t")
        np.testing.assert_allclose(base.loc[a.index, "y_anchor"], a)
        self.assertTrue(base.loc[2000.0:2004.0, "y_anchor"].isna().all())
        err = sc.attrs["origin_fit_error"]
        obs = float(self.df[(self.df.series == "s2") & (self.df.t == 2004)]["y"].iloc[0])
        b = sc[sc.scenario == "baseline"]
        self.assertAlmostEqual(err, float(np.interp(2004.0, b["t"], b["y_pred"])) - obs, places=6)
        # a panel forecast frame (several series) is cut to the model's series, never averaged across them
        other = anchor.assign(series="s9", y_pred=anchor["y_pred"] + 30)
        panel = pd.concat([anchor.assign(series="s2"), other], ignore_index=True)
        sc3 = self.m.scenario(self.mod, {"imitation": 1.3}, n_draws=40, anchor=panel)
        np.testing.assert_allclose(sc3["y_anchor"].to_numpy(), sc["y_anchor"].to_numpy())
        with self.assertRaises(ValueError):
            self.m.scenario(self.mod, {"imitation": 1.3}, n_draws=40,
                            anchor=pd.concat([other, other.assign(series="s8")], ignore_index=True))
        sc2 = self.m.scenario(self.mod, {"imitation": 1.3}, n_draws=40, anchor={2005.0: 41.0})
        self.assertAlmostEqual(sc2[(sc2.scenario == "baseline") & (sc2.t == 2005)]["y_anchor"].iloc[0], 41.0)


if __name__ == "__main__":
    unittest.main()
