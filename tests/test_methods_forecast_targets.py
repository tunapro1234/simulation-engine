"""DirectForecaster growth / logit targets, origin-time features, extrapolation check, panel_from_long (D7 #4, #5, #14)."""
import time
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.methods import forecast as F

try:
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401
    from sklearn.linear_model import LinearRegression
    HAVE_SK = True
except ImportError:  # pragma: no cover
    HAVE_SK = False

KW = dict(lags=(1, 2, 3), windows=(3,), n_train_origins=15)


def adoption_panel(n=120, T=31, seed=0):
    """Logistic adoption (ceiling 100): at the origin (t=30) some series are high, but every training base y[T]
    for h=5 (T <= 25) is still low - the D7 situation."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    rows = []
    for i in range(n):
        r, m = rng.uniform(0.4, 0.7), rng.uniform(28, 40)
        rows.append(pd.DataFrame({"id": f"s{i}", "time": t, "y": 100 / (1 + np.exp(-r * (t - m)))}))
    return pd.concat(rows, ignore_index=True)


@unittest.skipUnless(HAVE_SK, "sklearn missing")
class TestTargets(unittest.TestCase):
    def test_logit_stays_below_upper_where_level_and_log_ratio_overshoot(self):
        p = adoption_panel()
        o = 30
        self.assertLess(p.loc[p.time <= o - 5, "y"].max(), 25)          # training bases are low
        self.assertGreater(p.loc[p.time == o, "y"].max(), 50)          # test-origin levels are not
        lr = F.DirectForecaster("hgb", horizons=[5], target="log_ratio", **KW).fit(p, o).predict(p, o)
        lg = F.DirectForecaster("hgb", horizons=[5], target="logit", upper=100, **KW).fit(p, o).predict(p, o)
        lin = F.DirectForecaster(LinearRegression(), horizons=[5], **KW).fit(p, o).predict(p, o)
        self.assertGreater(lr["pred"].max(), 100)
        self.assertGreater(lin["pred"].max(), 100)
        self.assertLess(lg["pred"].max(), 100)
        self.assertGreater(lg["pred"].min(), 0)

    def test_log_ratio_maps_back_exact_growth(self):
        t = np.arange(30)
        p = pd.concat([pd.DataFrame({"id": f"s{i}", "time": t, "y": (1 + i) * 1.1 ** t}) for i in range(30)])
        o = 24
        fr = F.DirectForecaster("hgb", horizons=[1, 3], target="log_ratio", **KW).fit(p, o).predict(p, o)
        lvl = F.DirectForecaster("hgb", horizons=[1, 3], **KW).fit(p, o).predict(p, o)
        base = p[p.time == o].set_index("id")["y"]
        want = base.reindex(fr["id"]).to_numpy() * 1.1 ** fr["h"].to_numpy()
        np.testing.assert_allclose(fr["pred"].to_numpy(), want, rtol=1e-6)
        self.assertLessEqual(lvl["pred"].max(), p.loc[p.time <= o, "y"].max() + 1e-6)   # a level tree cannot

    def test_log_offset_handles_zeros(self):
        t = np.arange(20)
        p = pd.concat([pd.DataFrame({"id": f"s{i}", "time": t, "y": np.r_[np.zeros(5), np.arange(15) * (i + 1.0)]})
                       for i in range(25)])
        fr = F.DirectForecaster("hgb", horizons=[2], target="log_ratio", log_offset=1.0, **KW).fit(p, 15).predict(p, 15)
        self.assertTrue(np.isfinite(fr["pred"]).all())
        fr0 = F.DirectForecaster("hgb", horizons=[2], target="log_ratio", **KW).fit(p, 15).predict(p, 15)
        self.assertTrue(np.isfinite(fr0["pred"]).all())

    def test_validation_and_upper_warning(self):
        with self.assertRaises(ValueError):
            F.DirectForecaster(target="growth")
        with self.assertRaises(ValueError):
            F.DirectForecaster(target="logit")
        p = adoption_panel(n=40)
        p.loc[p.time == 30, "y"] *= 3          # passes the ceiling
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fr = F.DirectForecaster("hgb", horizons=[2], target="logit", upper=100, **KW).fit(p, 30).predict(p, 30)
        self.assertTrue(any("upper" in str(x.message) for x in w))
        self.assertLess(fr["pred"].max(), 100)

    def test_default_is_level(self):
        p = adoption_panel(n=40)
        a = F.DirectForecaster("hgb", horizons=[2], **KW).fit(p, 28).predict(p, 28)
        b = F.DirectForecaster("hgb", horizons=[2], target="level", **KW).fit(p, 28).predict(p, 28)
        pd.testing.assert_frame_equal(a, b)


@unittest.skipUnless(HAVE_SK, "sklearn missing")
class TestOriginFeatures(unittest.TestCase):
    def panel(self):
        rng = np.random.default_rng(3)
        t = np.arange(40)
        rows = []
        for i in range(40):
            other = np.cumsum(rng.uniform(0, 2, len(t)))            # e.g. an older technology's level
            y = 0.5 * other + rng.normal(0, 0.1, len(t))
            rows.append(pd.DataFrame({"id": f"s{i}", "time": t, "y": y, "other": other}))
        return pd.concat(rows, ignore_index=True)

    def test_read_at_origin(self):
        p = self.panel()
        X = F.make_features(p, 30, horizons=[1, 4], origin_features=["other"], **{"lags": (1,), "windows": (2,)})
        at = p[p.time == 30].set_index("id")["other"]
        np.testing.assert_allclose(X["other_at_origin"].to_numpy(), at.reindex(X["id"]).to_numpy())
        fc = F.DirectForecaster("hgb", horizons=[4], origin_features=["other"], **KW).fit(p, 30)
        self.assertIn("other_at_origin", fc.feature_names_)

    def test_no_future_leak(self):
        p = self.panel()
        o = 30
        q = p.copy()
        q.loc[q.time > o, "other"] = np.random.default_rng(9).normal(0, 1e6, int((q.time > o).sum()))
        kw = dict(horizons=[1, 4], origin_features=["other"], **KW)
        a = F.DirectForecaster("hgb", **kw)(p, o)
        b = F.DirectForecaster("hgb", **kw)(q, o)
        pd.testing.assert_frame_equal(a, b)
        # the same column as exog (read at the target time) does see the change: the test is sensitive
        ea = F.DirectForecaster("hgb", horizons=[4], exog=["other"], **KW)(p, o)
        eb = F.DirectForecaster("hgb", horizons=[4], exog=["other"], **KW)(q, o)
        self.assertFalse(np.allclose(ea["pred"], eb["pred"]))

    def test_missing_column(self):
        with self.assertRaises(KeyError):
            F.DirectForecaster("hgb", horizons=[1], origin_features=["nope"], **KW).fit(self.panel(), 30)


@unittest.skipUnless(HAVE_SK, "sklearn missing")
class TestExtrapolationCheck(unittest.TestCase):
    def test_function(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = F.extrapolation_check([1, 5, 9], (0, 10))
        self.assertTrue(r["ok"])
        with self.assertWarns(UserWarning):
            r = F.extrapolation_check(pd.DataFrame({"pred": [1, 5, 150, -20]}), np.array([0, 3, 10.0]), margin=0.25)
        self.assertEqual((r["n_above"], r["n_below"], r["n"]), (1, 1, 4))
        self.assertAlmostEqual(r["share_out"], 0.5)
        r = F.extrapolation_check([150], (0, 10), warn=False)
        self.assertFalse(r["ok"])

    def test_forecaster_warns_on_growth_trap(self):
        p = adoption_panel()
        with self.assertWarns(UserWarning):
            F.DirectForecaster("hgb", horizons=[5], target="log_ratio", check_range=0.25, **KW).fit(p, 30).predict(p, 30)
        fc = F.DirectForecaster("hgb", horizons=[5], target="log_ratio", **KW).fit(p, 30)
        r = F.extrapolation_check(fc.predict(p, 30), fc, warn=False)
        self.assertGreater(r["n_above"], 0)
        self.assertLess(fc.target_range_[1], 100)


class TestPanelFromLong(unittest.TestCase):
    def long(self):
        return pd.DataFrame({
            "country": ["A"] * 6 + ["B"] * 5,
            "year": [1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2],
            "variable": ["cell", "net", "tv", "cell", "net", "tv", "cell", "tv", "cell", "net", "tv"],
            "value": [1.0, 0.1, 50, 2.0, 0.3, 55, 3.0, 40, 4.0, 0.5, 41]})

    def test_layout_and_values(self):
        pn = F.panel_from_long(self.long(), id="country", time="year", var="variable", value="value",
                               targets=["cell", "net"])
        self.assertEqual(list(pn.columns), ["series", "country", "variable", "year", "y", "cell", "net", "tv"])
        self.assertEqual(sorted(pn["series"].unique()), ["A:cell", "A:net", "B:cell", "B:net"])
        r = pn[(pn.series == "B:net") & (pn.year == 1)].iloc[0]
        self.assertTrue(np.isnan(r["y"]))
        self.assertEqual((r["cell"], r["tv"]), (3.0, 40))
        r = pn[(pn.series == "A:cell") & (pn.year == 2)].iloc[0]
        self.assertEqual((r["y"], r["net"], r["variable"]), (2.0, 0.3, "cell"))
        pn2 = F.panel_from_long(self.long(), id="country", time="year", targets=["net"], features=["tv"])
        self.assertEqual(list(pn2.columns), ["series", "country", "variable", "year", "y", "tv"])

    def test_errors(self):
        d = self.long()
        with self.assertRaises(KeyError):
            F.panel_from_long(d, id="country", time="year", targets=["radio"])
        with self.assertRaises(ValueError):
            F.panel_from_long(pd.concat([d, d.iloc[:1]]), id="country", time="year", targets=["cell"])
        with self.assertRaises(ValueError):
            F.panel_from_long(d.replace({"tv": "y"}), id="country", time="year", targets=["cell"])

    def test_dropna_series(self):
        d = self.long()
        d = d[~((d.country == "B") & (d.variable == "net"))]
        pn = F.panel_from_long(d, id="country", time="year", targets=["net"])
        self.assertEqual(list(pn["series"].unique()), ["A:net"])
        pn = F.panel_from_long(d, id="country", time="year", targets=["net"], dropna_series=False)
        self.assertEqual(sorted(pn["series"].unique()), ["A:net", "B:net"])

    def test_vectorised_speed(self):
        rng = np.random.default_rng(0)
        ids, years, vars_ = np.arange(1500), np.arange(40), [f"v{k}" for k in range(6)]
        idx = pd.MultiIndex.from_product([ids, years, vars_], names=["c", "t", "var"]).to_frame(index=False)
        idx["val"] = rng.uniform(0, 1, len(idx))                       # 360k rows
        t0 = time.perf_counter()
        pn = F.panel_from_long(idx, id="c", time="t", var="var", value="val", targets=["v0", "v1", "v2"])
        self.assertLess(time.perf_counter() - t0, 15.0)
        self.assertEqual(len(pn), 3 * 1500 * 40)

    @unittest.skipUnless(HAVE_SK, "sklearn missing")
    def test_end_to_end(self):
        rng = np.random.default_rng(1)
        rows = []
        for c in range(30):
            m = rng.uniform(20, 35)
            for v, shift in (("cell", 0), ("net", 4), ("pc", -3)):
                y = 100 / (1 + np.exp(-0.5 * (np.arange(30) - m - shift)))
                rows.append(pd.DataFrame({"country": f"c{c}", "year": 1970 + np.arange(30), "variable": v, "value": y}))
        d = pd.concat(rows, ignore_index=True)
        pn = F.panel_from_long(d, id="country", time="year", targets=["cell", "net"])
        fc = F.DirectForecaster("hgb", horizons=[1, 5], id_col="series", time_col="year", static=["variable"],
                                target="logit", upper=100, origin_features=["cell", "net", "pc"], **KW)
        fr = fc.fit(pn, 1995).predict(pn, 1995)
        self.assertEqual(len(fr), 60 * 2)
        self.assertTrue(((fr["pred"] > 0) & (fr["pred"] < 100)).all())
        self.assertIn("pc_at_origin", fc.feature_names_)


if __name__ == "__main__":
    unittest.main()
