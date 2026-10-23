"""DirectForecaster target='diff' and per-series ``upper`` ceilings for target='logit' (G10 #1)."""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.methods import forecast as F

try:
    from sklearn.linear_model import LinearRegression
    HAVE_SK = True
except ImportError:  # pragma: no cover
    HAVE_SK = False

KW = dict(lags=(1, 2, 3), windows=(3,), n_train_origins=10)


def trend_panel(n=30, T=25, seed=0):
    """y = a_i + b_i * t: intercepts and slopes differ per series."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    return pd.concat([pd.DataFrame({"id": f"s{i:02d}", "time": t, "y": rng.uniform(0, 50) + rng.uniform(-1, 3) * t})
                      for i in range(n)], ignore_index=True)


def ceiling_panel(n=60, T=30, seed=1):
    """Logistic adoption with a different ceiling per series (30, 60 or 100), carried as a ``cap`` column."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    rows = []
    for i in range(n):
        cap = (30.0, 60.0, 100.0)[i % 3]
        r, m = rng.uniform(0.3, 0.6), rng.uniform(15, 32)
        rows.append(pd.DataFrame({"id": f"s{i:02d}", "time": t, "y": cap / (1 + np.exp(-r * (t - m))), "cap": cap}))
    return pd.concat(rows, ignore_index=True)


def golden_panel():
    rng = np.random.default_rng(3)
    t = np.arange(25)
    rows = []
    for i in range(40):
        r, m = rng.uniform(0.3, 0.6), rng.uniform(15, 30)
        rows.append(pd.DataFrame({"id": f"s{i:02d}", "time": t, "y": 80 / (1 + np.exp(-r * (t - m)))}))
    return pd.concat(rows, ignore_index=True)


@unittest.skipUnless(HAVE_SK, "sklearn missing")
class TestDiffTarget(unittest.TestCase):
    def test_in_targets(self):
        self.assertIn("diff", F.TARGETS)

    def test_recovers_linear_trend(self):
        p = trend_panel()
        o = 20
        fr = F.DirectForecaster(LinearRegression(), horizons=[1, 3], target="diff", **KW).fit(p, o).predict(p, o)
        truth = p.set_index(["id", "time"])["y"]
        want = truth.reindex(list(zip(fr["id"], fr["time"]))).to_numpy()
        np.testing.assert_allclose(fr["pred"].to_numpy(), want, rtol=1e-8, atol=1e-8)

    def test_tree_extrapolates_past_training_levels(self):
        # common slope 2, origin levels above every training target: a level tree is stuck, a diff tree is not
        t = np.arange(40)
        p = pd.concat([pd.DataFrame({"id": f"s{i}", "time": t, "y": 5.0 * i + 2.0 * t}) for i in range(20)])
        o = 39
        kw = dict(lags=(1, 2), windows=(2,), n_train_origins=5)
        d = F.DirectForecaster("hgb", horizons=[2], target="diff", **kw).fit(p, o - 10).predict(p, o)
        lv = F.DirectForecaster("hgb", horizons=[2], **kw).fit(p, o - 10).predict(p, o)
        want = p[p.time == o].set_index("id")["y"].reindex(d["id"]).to_numpy() + 4.0
        np.testing.assert_allclose(d["pred"].to_numpy(), want, atol=1e-3)
        self.assertLess(lv["pred"].max(), want.max() - 10)

    def test_gap_at_origin_uses_last_observed_level(self):
        p = trend_panel()
        o = 20
        gap = p[~((p.id == "s00") & (p.time == o))]
        fr = F.DirectForecaster(LinearRegression(), horizons=[1], target="diff", **KW).fit(gap, o).predict(gap, o)
        last = p[(p.id == "s00") & (p.time == o - 1)]["y"].iloc[0]
        self.assertGreater(fr.loc[fr.id == "s00", "pred"].iloc[0], 0.5 * last)


@unittest.skipUnless(HAVE_SK, "sklearn missing")
class TestPerSeriesUpper(unittest.TestCase):
    def test_scalar_upper_unchanged(self):
        # golden values recorded from the scalar-only implementation before per-series ceilings were added
        p = golden_panel()
        want = {"lin": ([54.9044163091, 0.880797074, 6.2007104995, 50.3951287983, 12.2992510946], 2640.2350021499838),
                "hgb": ([57.1200912351, 0.9765382058, 6.0405369781, 72.9194562844, 11.4078505346], 3149.5373597301086)}
        for name, m in (("lin", LinearRegression()), ("hgb", "hgb")):
            fr = F.DirectForecaster(m, horizons=[1, 3], target="logit", upper=100, **KW).fit(p, 20).predict(p, 20)
            np.testing.assert_allclose(fr["pred"].to_numpy()[[0, 7, 39, 40, 79]], want[name][0], rtol=1e-9)
            self.assertAlmostEqual(float(fr["pred"].sum()), want[name][1], places=6)

    def test_dict_and_column_equal_scalar_when_constant(self):
        p = golden_panel()
        ref = F.DirectForecaster("hgb", horizons=[2], target="logit", upper=100, **KW).fit(p, 20).predict(p, 20)
        d = {k: 100.0 for k in p["id"].unique()}
        fd = F.DirectForecaster("hgb", horizons=[2], target="logit", upper=d, **KW).fit(p, 20).predict(p, 20)
        fc = F.DirectForecaster("hgb", horizons=[2], target="logit", upper="cap", **KW).fit(
            p.assign(cap=100.0), 20).predict(p.assign(cap=100.0), 20)
        np.testing.assert_array_equal(fd["pred"].to_numpy(), ref["pred"].to_numpy())
        np.testing.assert_array_equal(fc["pred"].to_numpy(), ref["pred"].to_numpy())

    def test_per_series_ceiling_never_exceeded(self):
        p = ceiling_panel()
        o = 22
        caps = p.groupby("id")["cap"].first()
        for upper in ("cap", caps.to_dict()):
            fc = F.DirectForecaster("hgb", horizons=[1, 4, 8], target="logit", upper=upper, **KW)
            fr = fc.fit(p, o).predict(p, o)
            c = caps.reindex(fr["id"]).to_numpy()
            self.assertTrue((fr["pred"].to_numpy() < c).all(), upper if isinstance(upper, str) else "dict")
            self.assertTrue((fr["pred"].to_numpy() > 0).all())
        # a single global ceiling lets the low-ceiling series pass their own level
        g = F.DirectForecaster("hgb", horizons=[1, 4, 8], target="logit", upper=100, **KW).fit(p, o).predict(p, o)
        self.assertTrue((g["pred"].to_numpy() > caps.reindex(g["id"]).to_numpy()).any())

    def test_errors(self):
        p = ceiling_panel(n=12)
        caps = p.groupby("id")["cap"].first().to_dict()
        with self.assertRaises(ValueError):
            F.DirectForecaster(target="logit", upper={"a": 0.0})
        with self.assertRaises(ValueError):
            F.DirectForecaster(target="logit", upper=-1)
        miss = dict(caps)
        miss.pop("s03")
        with self.assertRaisesRegex(KeyError, "s03"):
            F.DirectForecaster("hgb", horizons=[1], target="logit", upper=miss, **KW).fit(p, 20)
        with self.assertRaisesRegex(KeyError, "nocol"):
            F.DirectForecaster("hgb", horizons=[1], target="logit", upper="nocol", **KW).fit(p, 20)
        q = p.copy()
        q.loc[q.id == "s05", "cap"] = np.nan
        with self.assertRaisesRegex(ValueError, "s05"):
            F.DirectForecaster("hgb", horizons=[1], target="logit", upper="cap", **KW).fit(q, 20)
        q = p.copy()
        q.loc[(q.id == "s02") & (q.time == 3), "cap"] = 31.0
        with self.assertRaisesRegex(ValueError, "constant"):
            F.DirectForecaster("hgb", horizons=[1], target="logit", upper="cap", **KW).fit(q, 20)

    def test_warns_per_series(self):
        p = ceiling_panel(n=12)
        caps = p.groupby("id")["cap"].first().to_dict()
        top = p[p.time <= 25].groupby("id")["y"].max()
        caps[top.idxmax()] = top.max() / 2            # that series' observed level passes its ceiling
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            F.DirectForecaster("hgb", horizons=[1], target="logit", upper=caps, **KW).fit(p, 25)
        msg = [str(x.message) for x in w if "upper" in str(x.message)]
        self.assertTrue(msg and top.idxmax() in msg[0], msg)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            F.DirectForecaster("hgb", horizons=[1], target="logit", upper="cap", **KW).fit(p, 25)
        self.assertFalse([x for x in w if "upper" in str(x.message)])


if __name__ == "__main__":
    unittest.main()
