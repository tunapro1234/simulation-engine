import json
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine.methods as M
from simengine.llm import FakeLLM
from simengine.methods import forecast as F


def _sklearn_hgb():
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401
        return True
    except ImportError:  # pragma: no cover
        return False


def seasonal_panel(n=20, T=120, season=12, trend=0.0, seed=0, noise=True):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    rows = []
    for i in range(n):
        lev = rng.uniform(5, 30)
        phase = rng.uniform(0, 2 * np.pi)
        mu = lev * (1 + trend * t) * (1 + 0.6 * np.sin(2 * np.pi * t / season + phase))
        y = rng.poisson(np.maximum(mu, 0.1)) if noise else mu
        rows.append(pd.DataFrame({"id": f"s{i}", "time": t, "y": y.astype(float)}))
    return pd.concat(rows, ignore_index=True)


class TestBaselines(unittest.TestCase):
    def test_shapes_and_values(self):
        p = pd.DataFrame({"id": ["a"] * 6 + ["b"] * 6, "time": list(range(6)) * 2,
                          "y": [1, 2, 3, 4, 5, 6, 10, 0, 10, 0, 10, 0]})
        fr = M.persistence_forecast(p, 5, 3)
        self.assertEqual(list(fr.columns), ["id", "origin", "h", "time", "pred"])
        self.assertEqual(len(fr), 6)
        self.assertEqual(fr[fr.id == "a"].pred.tolist(), [6, 6, 6])
        self.assertEqual(fr[fr.h == 3].time.tolist(), [8, 8])          # extrapolated target time
        mm = M.moving_mean(p, 5, 2, window=2)
        self.assertEqual(mm[mm.id == "b"].pred.tolist(), [5, 5])
        sn = M.seasonal_naive(p, 5, 2, season=2)
        self.assertEqual(sn[sn.id == "a"].pred.tolist(), [5, 6])
        self.assertTrue(np.isnan(M.seasonal_naive(p, 5, 1, season=10).pred).all())

    def test_seasonal_naive_beats_persistence(self):
        p = seasonal_panel(season=12)
        origins = list(range(60, 110, 5))
        e_sn = F.wmape(*M.backtest(M.seasonal_naive, p, origins=origins, horizon=4, season=12)[["truth", "pred"]].T.values)
        e_p = F.wmape(*M.backtest(M.persistence_forecast, p, origins=origins, horizon=4)[["truth", "pred"]].T.values)
        self.assertLess(e_sn, e_p)

    def test_ses_alpha_tuned_without_future(self):
        p = seasonal_panel(season=12, seed=3)
        fr, a, sc = M.ses(p, 80, 4, return_alpha=True)
        self.assertIn(a, F.SES_ALPHAS)
        self.assertEqual(a, min(sc, key=sc.get))
        p2 = p.copy()
        p2.loc[p2.time > 80, "y"] = 1e6                       # future change must not move alpha or level
        fr2, a2, _ = M.ses(p2, 80, 4, return_alpha=True)
        self.assertEqual(a, a2)
        np.testing.assert_allclose(fr.pred, fr2.pred)
        fixed = M.ses(p, 80, 2, alpha=1.0)
        np.testing.assert_allclose(fixed.pred, M.persistence_forecast(p, 80, 2).pred)


class TestFeatures(unittest.TestCase):
    def test_no_leakage(self):
        p = seasonal_panel(n=5, T=60)
        p["price"] = 1.0 + 0.01 * p["time"]
        p["store"] = p["id"].str[-1]
        kw = dict(lags=(1, 2, 3), windows=(4, 8), horizons=[1, 2, 3], season=12, exog=["price"], static=["store"])
        f1 = M.make_features(p, 40, **kw)
        p2 = p.copy()
        p2.loc[p2.time > 40, "y"] = 999.0
        f2 = M.make_features(p2, 40, **kw)
        pd.testing.assert_frame_equal(f1, f2)
        self.assertEqual(len(f1), 15)
        row = f1[(f1.id == "s0") & (f1.h == 2)].iloc[0]
        y0 = p[p.id == "s0"].set_index("time").y
        self.assertEqual(row.lag1, y0[40])
        self.assertEqual(row.lag3, y0[38])
        self.assertAlmostEqual(row.mean4, y0[37:41].mean())
        self.assertEqual(row.season_lag, y0[42 - 12])
        self.assertAlmostEqual(row.price, 1.42)              # exog read at the target time
        # changing the value AT the origin does change lag1 (sanity: the cut is at the origin, inclusive)
        p3 = p.copy()
        p3.loc[(p3.time == 40) & (p3.id == "s0"), "y"] = 999.0
        self.assertNotEqual(M.make_features(p3, 40, **kw).loc[0, "lag1"], f1.loc[0, "lag1"])


class TestBacktest(unittest.TestCase):
    def test_shape_and_masking(self):
        p = seasonal_panel(n=4, T=40)
        seen = []

        def fn(panel, origin, horizon, **kw):
            seen.append(panel.loc[panel.time > origin, "y"].notna().sum())
            return M.moving_mean(panel, origin, horizon, window=3, **kw)
        bt = M.backtest(fn, p, origins=[20, 24, 37], horizon=4)
        self.assertEqual(seen, [0, 0, 0])                       # callee never sees y after the origin
        self.assertEqual(list(bt.columns), ["id", "origin", "h", "time", "truth", "pred"])
        self.assertEqual(len(bt), 4 * 4 + 4 * 4 + 4 * 2)        # origin 37: only h=1,2 have truth
        r = bt[(bt.id == "s1") & (bt.origin == 24) & (bt.h == 3)].iloc[0]
        self.assertEqual(r.truth, p[(p.id == "s1") & (p.time == 27)].y.iloc[0])
        s = M.summarize_backtest(bt)
        self.assertEqual(list(s.origin), [20, 24, 37])
        self.assertIn("bias", s.columns)
        self.assertEqual(len(M.backtest(fn, p, origins=[37], horizon=4, dropna=False)), 16)


@unittest.skipUnless(_sklearn_hgb(), "sklearn HistGradientBoostingRegressor not available")
class TestBacktestReviewFixes(unittest.TestCase):
    def test_sparse_panel_scores_like_dense(self):
        # "missing cell = 0": dropping the y=0 rows must not drop those weeks from the truth
        dense = seasonal_panel(n=8, T=40, seed=3)
        dense.loc[dense.index % 5 == 0, "y"] = 0.0
        sparse = dense[dense.y != 0]
        a = F.backtest(F.moving_mean, dense, origins=[30, 33], horizon=4)
        b = F.backtest(F.moving_mean, sparse, origins=[30, 33], horizon=4)
        self.assertEqual(len(a), len(b))
        self.assertAlmostEqual(F.wmape(a.truth, a.pred), F.wmape(b.truth, b.pred), places=12)
        c = F.backtest(F.moving_mean, sparse, origins=[30, 33], horizon=4, fill_value=np.nan)
        self.assertLess(len(c), len(a))                        # old behaviour on request
        tail = F.backtest(F.moving_mean, dense, origins=[38], horizon=4)
        self.assertEqual(sorted(tail.h.unique()), [1])         # beyond the panel stays NaN and is dropped

    def test_time_with_no_rows_keeps_horizon_alignment(self):
        p = seasonal_panel(n=4, T=30, seed=1)
        gap = p[p.time != 25]                                  # no row at all at t=25
        fr = F.persistence_forecast(gap, 24, 2)
        self.assertEqual(sorted(fr.time.unique()), [25, 26])
        bt = F.backtest(F.persistence_forecast, gap, origins=[24], horizon=2)
        h2 = bt[bt.h == 2].sort_values("id")
        want = p[p.time == 26].sort_values("id").y.to_numpy()
        np.testing.assert_allclose(h2.truth.to_numpy(), want)
        self.assertTrue((bt[bt.h == 1].truth == 0).all())      # the empty week is a zero-sales week
        d = pd.DataFrame({"id": "a", "time": pd.to_datetime(["2016-01-04", "2016-01-11", "2016-01-25"]),
                          "y": [1.0, 2.0, 3.0]})
        self.assertEqual(len(F._Wide(d, "id", "time", "y").times), 4)

    @unittest.skipUnless(_sklearn_hgb(), "sklearn missing")
    def test_direct_forecaster_horizon_mismatch_warns(self):
        p = seasonal_panel(n=6, T=60, seed=2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bt = F.backtest(F.DirectForecaster("hgb", horizons=2), p, origins=[50], horizon=4)
        self.assertTrue(any("not covered" in str(x.message) for x in w))
        self.assertEqual(sorted(bt.h.unique()), [1, 2])


class TestDirect(unittest.TestCase):
    def test_beats_moving_mean_on_trend_season(self):
        p = seasonal_panel(n=25, T=110, season=12, trend=0.01, seed=1)
        origins = [86, 94, 102]
        model = M.DirectForecaster("hgb", horizons=4, lags=(1, 2, 3, 4), windows=(4, 12), season=12,
                                   n_train_origins=40, clip_min=0.0)
        bt = M.backtest(model, p, origins=origins, horizon=4)
        mm = M.backtest(M.moving_mean, p, origins=origins, horizon=4, window=4)
        self.assertEqual(len(bt), len(mm))
        self.assertLess(F.wmape(bt.truth, bt.pred), F.wmape(mm.truth, mm.pred))
        self.assertTrue((bt.pred >= 0).all())

    def test_fit_ignores_future_and_sklearn_estimator(self):
        from sklearn.linear_model import Ridge
        p = seasonal_panel(n=6, T=60, season=12)
        m = M.DirectForecaster(Ridge(), horizons=[1, 3], windows=(4,), season=12)
        a = m.fit(p, 45).predict(p, 45)
        p2 = p.copy()
        p2.loc[p2.time > 45, "y"] = 1e5
        b = M.DirectForecaster(Ridge(), horizons=[1, 3], windows=(4,), season=12).fit(p2, 45).predict(p2, 45)
        np.testing.assert_allclose(a.pred, b.pred)
        self.assertEqual(sorted(a.h.unique()), [1, 3])

    def test_lightgbm_or_fallback(self):
        p = seasonal_panel(n=6, T=50, season=12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fr = M.DirectForecaster("lightgbm", horizons=2, windows=(4,), n_train_origins=10).fit(p, 45).predict(p)
        self.assertEqual(len(fr), 12)
        self.assertTrue(np.isfinite(fr.pred).all())


class TestPointForMetric(unittest.TestCase):
    def test_poisson_median_lowers_mae(self):
        rng = np.random.default_rng(0)
        lam = rng.uniform(0.1, 2.5, 20000)
        y = rng.poisson(lam)
        med = M.point_for_metric(lam, metric="wmape")
        self.assertLess(np.abs(y - med).mean(), np.abs(y - lam).mean())
        self.assertLess(F.bias(y, med), F.bias(y, lam))          # median biases totals down: report it
        np.testing.assert_array_equal(M.point_for_metric(lam, metric="mse"), lam)
        t = M.point_for_metric(np.array([0.3, 2.6, 5.4]), metric="mae", threshold=3)
        np.testing.assert_array_equal(t, [0.0, 2.0, 5.4])
        with self.assertRaises(ValueError):
            M.point_for_metric(lam, metric="auc")


class TestLLMPrior(unittest.TestCase):
    def test_named_blind_shuffled(self):
        vals = {"FOODS": 1.3, "HOBBIES": 0.9, "HOUSEHOLD": 1.1}

        def responder(system, user):
            for k, v in vals.items():
                if k in user:
                    return json.dumps({"value": v})
            return "about 1.0"                                   # blind arm: not JSON, plain number fallback
        llm = FakeLLM(responder)
        out = M.llm_prior_feature(llm, list(vals), "Sales multiplier for {key} in a holiday week?",
                                  blind_question="Sales multiplier for {key} in a holiday week?", seed=1)
        self.assertEqual(list(out.columns), ["key", "value", "value_blind", "value_shuffled"])
        self.assertEqual(out.value.tolist(), [1.3, 0.9, 1.1])
        self.assertEqual(out.value_blind.tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(sorted(out.value_shuffled), sorted(out.value))
        self.assertTrue((out.value_shuffled != out.value).all())  # derangement
        self.assertEqual(len(llm.calls), 6)
        self.assertNotIn("FOODS", " ".join(u for _, u in llm.calls[3:]))
        self.assertIn("group A", llm.calls[3][1])

    def test_unparsable_is_nan_with_warning(self):
        llm = FakeLLM(lambda s, u: "no idea")
        with self.assertWarns(RuntimeWarning):
            out = M.llm_prior_feature(llm, [("x", 1), ("y", 2)], lambda k: f"q {k}", shuffle_placebo=False)
        self.assertTrue(out.value.isna().all())
        self.assertNotIn("value_shuffled", out.columns)


if __name__ == "__main__":
    unittest.main()
