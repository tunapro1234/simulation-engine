"""methods.tabular (TabularRidge, TabularGBDT, RandomEffectsMean, GroupStat) and oof_predict(weight=)."""
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from simengine.methods import oof_predict
from simengine.methods.tabular import (GroupStat, RandomEffectsMean, TabularGBDT, TabularRidge,
                                       dersimonian_laird)


def table(n=400, seed=0):
    rs = np.random.default_rng(seed)
    out = rs.choice(["vote", "fav", "turnout"], n)
    year = rs.choice([2018, 2020], n)
    x = rs.normal(0, 1, n)
    eff = pd.Series(out).map({"vote": 1.0, "fav": 3.0, "turnout": -1.0}).to_numpy() + 0.5 * (year == 2020)
    y = eff + 2.0 * x + rs.normal(0, 0.3, n)
    return pd.DataFrame({"outcome": out, "year": year, "x": x, "y": y, "study": rs.integers(0, 40, n)})


def two_level(n=200):
    """Half the rows y=0, half y=10, nothing else: the fitted level is the (weighted) mean of y."""
    df = pd.DataFrame({"c": ["a"] * n, "x": np.zeros(n), "y": np.r_[np.zeros(n // 2), np.full(n // 2, 10.0)]})
    df["w"] = np.r_[np.full(n // 2, 9.0), np.ones(n // 2)]
    return df


class TabularRidgeTest(unittest.TestCase):
    def test_recovers_effects_and_handles_unseen(self):
        df = table()
        m = TabularRidge(cat=["outcome", "year"], num=["x"]).fit(df[["outcome", "year", "x"]], df.y)
        self.assertAlmostEqual(m.coef_["x"] / df.x.std(ddof=0), 2.0, delta=0.1)
        d = m.coef_["outcome=fav"] - m.coef_["outcome=vote"]
        self.assertAlmostEqual(d, 2.0, delta=0.15)
        new = pd.DataFrame({"outcome": ["unknown", "fav"], "year": [2018, 2018], "x": [np.nan, 0.0]})
        p = m.predict(new)
        self.assertTrue(np.isfinite(p).all())

    def test_infers_columns(self):
        df = table()
        m = TabularRidge().fit(df[["outcome", "x"]], df.y)
        self.assertEqual((m.cat_, m.num_), (["outcome"], ["x"]))

    def test_weighted_fit_moves_toward_heavy_rows(self):
        df = two_level()
        X = df[["c", "x"]]
        pu = TabularRidge().fit(X, df.y).predict(X[:1])[0]
        pw = TabularRidge().fit(X, df.y, sample_weight=df.w).predict(X[:1])[0]
        self.assertAlmostEqual(pu, 5.0, places=6)
        self.assertAlmostEqual(pw, 1.0, places=6)          # (9*0 + 1*10) / 10

    def test_in_oof_predict_with_weight(self):
        df = table()
        df["w"] = np.where(df.outcome == "fav", 5.0, 1.0)
        feats = ["outcome", "year", "x"]
        a = oof_predict(TabularRidge(cat=["outcome", "year"], num=["x"]), df, features=feats, target="y",
                        group="study")
        b = oof_predict(TabularRidge(cat=["outcome", "year"], num=["x"]), df, features=feats, target="y",
                        group="study", weight="w")
        self.assertLess(np.mean(np.abs(a - df.y)), 0.4)
        self.assertFalse(np.allclose(a, b))


class TabularGBDTTest(unittest.TestCase):
    def _check(self, backend):
        df = table(n=600)
        X = df[["outcome", "year", "x"]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = TabularGBDT(cat=["outcome", "year"], num=["x"], backend=backend).fit(X, df.y)
        self.assertEqual(m.backend_, backend)
        self.assertLess(np.mean(np.abs(m.predict(X) - df.y)), 0.5)
        new = pd.DataFrame({"outcome": ["unknown"], "year": [2030], "x": [0.0]})
        self.assertTrue(np.isfinite(m.predict(new)).all())
        dfl = two_level()
        Xl = dfl[["c", "x"]]
        pu = TabularGBDT(backend=backend, min_child_samples=1).fit(Xl, dfl.y) if backend == "lightgbm" else \
            TabularGBDT(backend=backend, min_samples_leaf=1).fit(Xl, dfl.y)
        pw = TabularGBDT(backend=backend, min_child_samples=1).fit(Xl, dfl.y, sample_weight=dfl.w) \
            if backend == "lightgbm" else TabularGBDT(backend=backend, min_samples_leaf=1).fit(Xl, dfl.y,
                                                                                               sample_weight=dfl.w)
        self.assertAlmostEqual(pu.predict(Xl[:1])[0], 5.0, delta=0.05)
        self.assertAlmostEqual(pw.predict(Xl[:1])[0], 1.0, delta=0.05)

    def test_sklearn_backend(self):
        self._check("sklearn")

    def test_lightgbm_backend(self):
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("lightgbm not installed")
        self._check("lightgbm")

    def test_lightgbm_splits_categories_on_small_tables(self):
        # LightGBM's own min_data_per_group=100 never split a 6-level category in 200 rows
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("lightgbm not installed")
        rs = np.random.default_rng(0)
        c = rs.integers(0, 6, 200)
        df = pd.DataFrame({"c": [f"k{i}" for i in c], "x": rs.normal(0, 1, 200)})
        y = rs.normal(0, 1.5, 6)[c] + np.sin(2 * df.x) + rs.normal(0, 1, 200)
        m = TabularGBDT(cat=["c"], num=["x"], backend="lightgbm").fit(df, y)
        self.assertIn("c", set(m.model_.booster_.trees_to_dataframe().split_feature))


class RandomEffectsTest(unittest.TestCase):
    def test_dl_matches_hand_calculation(self):
        y, se = np.array([1.0, 2.0, 4.0]), np.array([1.0, 1.0, 2.0])
        # w = 1, 1, 0.25; ybar_w = (1 + 2 + 1) / 2.25 = 16/9
        # Q = (1-16/9)^2 + (2-16/9)^2 + 0.25 (4-16/9)^2 = 49/81 + 4/81 + 100/81 = 153/81 = 17/9
        # C = 2.25 - (1 + 1 + 0.0625) / 2.25 = 2.25 - 0.916667 = 1.333333;  tau2 = (17/9 - 2) / C < 0 -> 0
        mu, t2, s = dersimonian_laird(y, se)
        self.assertEqual(t2, 0.0)
        self.assertAlmostEqual(mu, 16 / 9, places=12)
        self.assertAlmostEqual(s, 2.25 ** -0.5, places=12)
        # heterogeneous: y = 0, 10, 20 with se = 1 -> w = 1, ybar = 10, Q = 200, C = 3 - 1 = 2, tau2 = (200-2)/2 = 99
        mu, t2, s = dersimonian_laird([0.0, 10.0, 20.0], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(t2, 99.0, places=12)
        self.assertAlmostEqual(mu, 10.0, places=12)
        self.assertAlmostEqual(s, (3 / 100) ** -0.5, places=12)
        # unequal se with tau2 > 0: y = 0, 10, se = 1, 2 -> w = 1, .25; ybar = 2.5/1.25 = 2; Q = 4 + .25*64 = 20
        # C = 1.25 - 1.0625/1.25 = 0.4; tau2 = 19/0.4 = 47.5; w* = 1/48.5, 1/51.5
        mu, t2, _ = dersimonian_laird([0.0, 10.0], [1.0, 2.0])
        self.assertAlmostEqual(t2, 47.5, places=10)
        self.assertAlmostEqual(mu, (10 / 51.5) / (1 / 48.5 + 1 / 51.5), places=10)

    def test_cells_and_fallback(self):
        X = pd.DataFrame({"o": ["a", "a", "a", "a", "b"], "yr": [1, 1, 1, 2, 1],
                          "se": [1.0, 1.0, 2.0, 1.0, 1.0]})
        y = np.array([1.0, 2.0, 4.0, 7.0, -3.0])
        m = RandomEffectsMean(by=["o", "yr"], se_col="se", min_n=2).fit(X, y)
        new = pd.DataFrame({"o": ["a", "a", "b", "z"], "yr": [1, 2, 1, 1]})
        p, lev = m.predict(new), m.level(new)
        self.assertAlmostEqual(p[0], 16 / 9, places=10)                 # cell (a, 1)
        self.assertEqual(list(lev), [2, 1, 0, 0])                       # (a, 2) has 1 row -> "a"; b has 1 -> grand
        self.assertAlmostEqual(p[1], dersimonian_laird(y[:4], X.se[:4])[0], places=10)
        self.assertAlmostEqual(p[2], dersimonian_laird(y, X.se)[0], places=10)
        self.assertEqual(p[2], p[3])

    def test_fixed_tau2_and_bad_se(self):
        X = pd.DataFrame({"o": ["a", "a"], "se": [1.0, 2.0]})
        m = RandomEffectsMean(by=["o"], tau2=0.0).fit(X, [0.0, 10.0])
        self.assertAlmostEqual(m.predict(X[:1])[0], 2.0)               # (0*1 + 10*.25) / 1.25
        with self.assertRaises(ValueError):
            RandomEffectsMean(by=["o"]).fit(pd.DataFrame({"o": ["a"], "se": [0.0]}), [1.0])
        with self.assertWarns(RuntimeWarning):
            RandomEffectsMean(by=["o"]).fit(pd.DataFrame({"o": ["a", "a"], "se": [1.0, np.nan]}), [1.0, 2.0])

    def test_in_oof_predict_and_no_weight(self):
        df = table()
        df["se"] = np.random.default_rng(1).uniform(0.2, 2.0, len(df))
        oof = oof_predict(RandomEffectsMean(by=["outcome", "year"]), df, features=["outcome", "year", "se"],
                          target="y", group="study")
        self.assertTrue(np.isfinite(oof).all())
        with self.assertRaisesRegex(TypeError, "sample_weight"):
            oof_predict(RandomEffectsMean(by=["outcome"]), df, features=["outcome", "se"], target="y",
                        group="study", weight="se")


class GroupStatTest(unittest.TestCase):
    def test_stats(self):
        X = pd.DataFrame({"o": ["a"] * 4 + ["b"], "w": [3.0, 1.0, 1.0, 1.0, 1.0]})
        y = [0.0, 1.0, 2.0, 10.0, 5.0]
        q = pd.DataFrame({"o": ["a", "zz"]})
        self.assertEqual(list(GroupStat(["o"], "median").fit(X, y).predict(q)), [1.5, 2.0])
        self.assertEqual(GroupStat(["o"], "mean").fit(X, y).predict(q)[0], 13 / 4)
        self.assertEqual(GroupStat(["o"], "wmean", weight="w").fit(X, y).predict(q)[0], 13 / 6)
        self.assertEqual(GroupStat(["o"], "median", weight="w").fit(X, y).predict(q)[0], 0.5)  # half weight on 0
        self.assertEqual(GroupStat(["o"], "mean").fit(X, y, sample_weight=X.w).predict(q)[0], 13 / 6)
        with self.assertRaises(ValueError):
            GroupStat(["o"], "wmean").fit(X, y)
        with self.assertRaises(ValueError):
            GroupStat(["o"], "mode")

    def test_weighted_median_skips_zero_weight_rows(self):
        X = pd.DataFrame({"o": ["a"] * 3, "w": [1.0, 0.0, 1.0]})
        self.assertEqual(GroupStat(["o"], "median", weight="w").fit(X, [1.0, 2.0, 3.0]).predict(X[:1])[0], 2.0)

    def test_float_and_int_codes_are_one_category(self):
        # a year column read as float (it had a NaN) at fit and as int at predict is still the same cell
        tr = pd.DataFrame({"yr": [2018.0, 2020.0, np.nan, 2020.0], "x": [0.0, 1.0, 2.0, 3.0]})
        te = pd.DataFrame({"yr": [2018, 2020], "x": [0.0, 1.0]})
        y = [0.0, 10.0, 5.0, 20.0]
        g = GroupStat(["yr"], "mean").fit(tr, y)
        self.assertEqual(list(g.level(te)), [1, 1])
        self.assertEqual(list(g.predict(te)), [0.0, 15.0])
        r = TabularRidge(cat=["yr"], num=(), alphas=[1e-6]).fit(tr, y)
        np.testing.assert_allclose(r.predict(te), r.predict(tr.iloc[[0, 1]]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = TabularGBDT(cat=["yr"], num=(), backend="sklearn").fit(tr, y)
        self.assertEqual(b.categories_["yr"], ["2018", "2020"])

    def test_two_level_fallback(self):
        X = pd.DataFrame({"o": ["a", "a", "a", "b"], "yr": [1, 1, 2, 1]})
        m = GroupStat(["o", "yr"], "mean", min_n=2).fit(X, [1.0, 3.0, 8.0, 0.0])
        q = pd.DataFrame({"o": ["a", "a", "b"], "yr": [1, 2, 1]})
        self.assertEqual(list(m.predict(q)), [2.0, 4.0, 3.0])
        self.assertEqual(list(m.level(q)), [2, 1, 0])

    def test_weighted_in_oof_predict(self):
        df = two_level()
        df["fold"] = np.arange(len(df)) % 4
        a = oof_predict(GroupStat(["c"], "mean"), df, features=["c"], target="y", folds="fold")
        b = oof_predict(GroupStat(["c"], "mean"), df, features=["c"], target="y", folds="fold", weight="w")
        self.assertAlmostEqual(a.mean(), 5.0, places=6)
        self.assertAlmostEqual(b.mean(), 1.0, places=6)


class OOFWeightTest(unittest.TestCase):
    def test_pipeline_routes_to_last_step(self):
        df = table()
        df["w"] = np.random.default_rng(2).uniform(0.1, 5.0, len(df))
        feats = ["x", "year"]
        oof = oof_predict(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), df, features=feats, target="y",
                          n_splits=4, weight="w")
        # same folds, by hand
        k = oof.attrs["fold"]
        for f in range(4):
            tr, te = k != f, k == f
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            m.fit(df.loc[tr, feats], df.y[tr], ridge__sample_weight=df.w[tr])
            np.testing.assert_allclose(oof[te], m.predict(df.loc[te, feats]))
        unw = oof_predict(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), df, features=feats, target="y",
                          n_splits=4)
        self.assertFalse(np.allclose(oof, unw))

    def test_weight_as_array_and_errors(self):
        df = two_level()
        w = df.w.to_numpy()
        a = oof_predict(Ridge(alpha=1e-9), df, features=["x"], target="y", n_splits=4, weight=w)
        self.assertAlmostEqual(a.mean(), 1.0, delta=0.05)

        class NoWeight:
            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X))

        with self.assertRaisesRegex(TypeError, "NoWeight.fit takes no sample_weight"):
            oof_predict(NoWeight(), df, features=["x"], target="y", weight="w")
        bad = df.assign(w=np.r_[np.nan, df.w[1:]])
        with self.assertRaisesRegex(ValueError, "weight"):
            oof_predict(Ridge(), bad, features=["x"], target="y", weight="w")
        with self.assertRaises(KeyError):
            oof_predict(Ridge(), df, features=["x"], target="y", weight="nope")
        # missing target rows may have a missing weight (they are not trained on)
        ok = df.assign(y=np.r_[np.nan, df.y[1:]], w=np.r_[np.nan, df.w[1:]])
        self.assertTrue(np.isnan(oof_predict(Ridge(), ok, features=["x"], target="y", weight="w").iloc[0]))


if __name__ == "__main__":
    unittest.main()
