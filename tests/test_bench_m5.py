"""m5 bench task: synthetic M5-shaped files (no real rows); the real-data check runs only when $SIMENGINE_DATA is set."""
import os
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from simengine.bench.base import DataNotFound
from simengine.bench.tasks import m5
from simengine.methods import forecast as fc
from simengine.types import Task

STORES = {"CA_1": "CA", "CA_2": "CA", "TX_1": "TX"}
DEPTS = {"FOODS_1": "FOODS", "FOODS_2": "FOODS", "HOBBIES_1": "HOBBIES"}
N_ITEMS, N_DAYS = 4, 1969
DOW = np.array([1.3, 1.4, 0.8, 0.7, 0.8, 0.9, 1.1])        # by wday 1 (Saturday) .. 7


def make_fixture(seed: int = 0) -> dict:
    """Series = item x store with a level, a start day, a weekly profile, SNAP / event bumps and a price effect."""
    rng = np.random.default_rng(seed)
    keys = []
    for dept, cat in DEPTS.items():
        for i in range(1, N_ITEMS + 1):
            for st, state in STORES.items():
                keys.append({"item_id": f"{dept}_{i:03d}", "dept_id": dept, "cat_id": cat, "store_id": st,
                             "state_id": state})
    keys = pd.DataFrame(keys)
    n = len(keys)
    dates = pd.date_range("2011-01-29", periods=N_DAYS, freq="D")
    wday = (np.arange(N_DAYS) % 7) + 1
    week = 11101 + np.arange(N_DAYS) // 7
    event = np.where(rng.random(N_DAYS) < 0.03, "Holiday", None)
    snap = {s: (np.arange(N_DAYS) % 30 < 10).astype(int) for s in ("CA", "TX", "WI")}
    start = rng.integers(0, 600, n)
    start[:3] = 0
    level = rng.gamma(2.0, 1.5, n)
    base_price = rng.uniform(1, 10, n)
    weeks = np.unique(week)
    W = base_price[:, None] * np.where(rng.random((n, len(weeks))) < 0.1, 0.8, 1.0)
    P = W[:, np.searchsorted(weeks, week)]
    lam = (level[:, None] * DOW[wday - 1][None, :] * (1 + 0.3 * np.array([snap[s] for s in keys["state_id"]]))
           * (1 + 0.5 * (event != None)[None, :]) * (base_price[:, None] / P) ** 2)  # noqa: E711
    Y = rng.poisson(lam).astype(float)
    on = np.arange(N_DAYS)[None, :] >= start[:, None]
    Y[~on] = 0.0
    P = np.where(np.arange(N_DAYS)[None, :] >= (start[:, None] // 7) * 7, P, np.nan)
    cal = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "wm_yr_wk": week, "weekday": dates.day_name(),
                        "wday": wday, "month": dates.month, "year": dates.year, "event_name_1": event,
                        "event_type_1": np.where(event != None, "National", None),  # noqa: E711
                        "event_name_2": None, "event_type_2": None,
                        **{f"snap_{s}": v for s, v in snap.items()}})
    return {"keys": keys, "Y": Y, "P": P, "cal": cal, "week": week}


def write_fixture(d: str, fx: dict, as_zip: bool = False) -> None:
    os.makedirs(d, exist_ok=True)
    keys, Y = fx["keys"], fx["Y"]
    files = {}
    tr = pd.concat([keys, pd.DataFrame(Y[:, :1941].astype(int), columns=[f"d_{i}" for i in range(1, 1942)])], axis=1)
    te = pd.concat([keys, pd.DataFrame(Y[:, 1941:].astype(int), columns=[f"d_{i}" for i in range(1942, 1970)])], axis=1)
    files["sales_train_evaluation.csv"] = tr.to_csv(index=False)
    files["sales_test_evaluation.csv"] = te.to_csv(index=False)
    files["calendar.csv"] = fx["cal"].to_csv(index=False)
    weeks = np.unique(fx["week"])
    first = np.searchsorted(fx["week"], weeks)
    rows = []
    for i, k in keys.iterrows():
        p = fx["P"][i, first]
        ok = np.isfinite(p)
        rows.append(pd.DataFrame({"store_id": k["store_id"], "item_id": k["item_id"], "wm_yr_wk": weeks[ok],
                                  "sell_price": np.round(p[ok], 2)}))
    fx["P"] = np.where(np.isfinite(fx["P"]), np.round(fx["P"], 2), np.nan)
    files["sell_prices.csv"] = pd.concat(rows, ignore_index=True).to_csv(index=False)
    if as_zip:
        with zipfile.ZipFile(os.path.join(d, "m5.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            for name, text in files.items():
                z.writestr(name, text)
    else:
        for name, text in files.items():
            with open(os.path.join(d, name), "w") as f:
                f.write(text)


def brute_wrmsse(fx: dict, origin: int, pred: pd.DataFrame) -> tuple[float, list[float]]:
    """Reference WRMSSE straight from the paper's definition with pandas groupby; pred: [item_id, store_id, d, pred]."""
    keys, Y, P = fx["keys"], fx["Y"], fx["P"]
    long = keys.loc[keys.index.repeat(N_DAYS)].reset_index(drop=True)
    long["d"] = np.tile(np.arange(1, N_DAYS + 1), len(keys))
    long["y"], long["price"] = Y.ravel(), P.ravel()
    long = long.merge(pred, on=["item_id", "store_id", "d"], how="left")
    per_level = []
    for level in m5.LEVELS:
        by = list(level) + ["d"]
        tmp = long.assign(dollar=np.nan_to_num(long["y"] * long["price"]))
        agg = tmp.groupby(by)[["y", "pred", "dollar"]].sum(min_count=1).reset_index()
        gk = list(level) or None
        num, wsum = 0.0, 0.0
        groups = agg.groupby(gk) if gk else [((), agg)]
        for _, g in groups:
            g = g.sort_values("d")
            h = g[g["d"] <= origin]["y"].to_numpy()
            f = int(np.argmax(h > 0))
            scale = np.mean(np.diff(h[f:]) ** 2)
            t = g[(g["d"] > origin) & (g["d"] <= origin + 28)]
            r = np.sqrt(np.mean((t["y"] - t["pred"]) ** 2) / scale)
            w = g[(g["d"] > origin - 28) & (g["d"] <= origin)]["dollar"].sum()
            num, wsum = num + w * r, wsum + w
        per_level.append(num / wsum)
    return float(np.mean(per_level)), per_level


class _Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        cls.fx = make_fixture()
        write_fixture(os.path.join(cls.root, "m5-walmart"), cls.fx)
        cls.task = m5.M5(baselines="naive+snaive+ses+ets_lite")
        cls.data = cls.task.load(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestLoad(_Fixture):
    def test_split_layout(self):
        d, n = self.data, len(self.fx["keys"])
        self.assertIsInstance(self.task, Task)
        self.assertEqual(d.meta["origin"], 1941)
        self.assertEqual(len(d.test), n * 28)
        self.assertEqual((d.test["d"].min(), d.test["d"].max()), (1942, 1969))
        self.assertEqual(sorted(d.test["h"].unique()), list(range(1, 29)))
        self.assertEqual(d.train["d"].max(), 1941)
        self.assertEqual(d.train["d"].nunique(), 730)
        for c in ("id", *m5.KEYS, *m5.COVARIATES):
            self.assertIn(c, d.train.columns)
        # truth and covariates match the generator
        i = int(np.flatnonzero(self.fx["keys"]["item_id"] + "_" + self.fx["keys"]["store_id"] == "FOODS_2_003_TX_1")[0])
        r = d.test[(d.test["id"] == "FOODS_2_003_TX_1") & (d.test["d"] == 1950)].iloc[0]
        self.assertEqual(r["sales"], self.fx["Y"][i, 1949])
        self.assertAlmostEqual(float(r["sell_price"]), self.fx["P"][i, 1949], places=4)
        self.assertEqual(int(r["wday"]), int(self.fx["cal"]["wday"][1949]))
        self.assertEqual(int(r["snap"]), int(self.fx["cal"]["snap_TX"][1949]))

    def test_other_splits_and_history(self):
        for sp, o in (("validation", 1913), ("dev", 1885)):
            d = m5.M5(split=sp, history=100).load(self.root)
            self.assertEqual(d.meta["origin"], o)
            self.assertEqual((d.train["d"].min(), d.train["d"].max()), (o - 99, o))
            self.assertEqual((d.test["d"].min(), d.test["d"].max()), (o + 1, o + 28))
        d = m5.M5(origin=1500, history=50).load(self.root)
        self.assertEqual(d.test["d"].max(), 1528)

    def test_options(self):
        with self.assertRaises(ValueError):
            m5.M5(split="private")
        with self.assertRaises(ValueError):
            m5.M5(horizon=7)
        with self.assertRaises(ValueError):
            m5.M5(baselines="snaive+prophet")
        with self.assertRaises(ValueError):
            m5.M5(origin=1960).load(self.root)
        with self.assertRaises(DataNotFound):
            m5.M5().load(os.path.join(self.root, "nothing-here"))

    def test_subsample(self):
        d = m5.M5(stores="CA_1+TX_1", history=60).load(self.root)
        self.assertEqual(set(d.test["store_id"].astype(str)), {"CA_1", "TX_1"})
        a = m5.M5(n_series=10, seed=3, history=60).load(self.root)
        b = m5.M5(n_series=10, seed=3, history=60).load(self.root)
        self.assertEqual(a.meta["n_series"], 10)
        self.assertTrue(a.meta["subsample"])
        self.assertEqual(list(a.meta["keys"]["id"]), list(b.meta["keys"]["id"]))
        lv = a.meta["levels"]
        self.assertAlmostEqual(lv[lv["level"] == 12]["weight"].sum(), 1.0)
        self.assertEqual(int(lv[lv["level"] == 12].shape[0]), 10)

    def test_zip_layout(self):
        with tempfile.TemporaryDirectory() as t:
            write_fixture(os.path.join(t, "m5-walmart"), self.fx, as_zip=True)
            d = m5.M5(history=60, split="validation").load(t)
            e = m5.M5(history=60, split="validation").load(self.root)
            pd.testing.assert_frame_equal(d.test, e.test)
            pd.testing.assert_frame_equal(d.meta["levels"], e.meta["levels"])


class TestHierarchy(_Fixture):
    def test_levels(self):
        lv = self.data.meta["levels"]
        sizes = lv.groupby("level").size().to_dict()
        n_items = len(DEPTS) * N_ITEMS
        self.assertEqual(sizes, {1: 1, 2: 2, 3: 3, 4: 2, 5: 3, 6: 4, 7: 6, 8: 6, 9: 9, 10: n_items,
                                 11: 2 * n_items, 12: 3 * n_items})
        for _, s in lv.groupby("level")["weight"].sum().items():
            self.assertAlmostEqual(s, 1.0)
        self.assertTrue((lv["n_bottom"].groupby(lv["level"]).sum() == 3 * n_items).all())

    def test_weight_and_scale_by_hand(self):
        fx, o = self.fx, 1941
        dollars = np.nansum(fx["Y"][:, o - 28:o] * fx["P"][:, o - 28:o], axis=1)
        ser = self.data.meta["series"]
        ids = fx["keys"]["item_id"] + "_" + fx["keys"]["store_id"]
        np.testing.assert_allclose(ser.loc[ids, "w_L12"].to_numpy(), dollars / dollars.sum(), rtol=1e-5)
        # CA_1 store aggregate (level 3), first-non-zero rule
        A = fx["Y"][(fx["keys"]["store_id"] == "CA_1").to_numpy(), :o].sum(0)
        f = int(np.argmax(A > 0))
        ref = np.mean(np.diff(A[f:]) ** 2)
        got = ser.loc[ids[fx["keys"]["store_id"] == "CA_1"], "sc_L3"].unique()
        np.testing.assert_allclose(got, [ref], rtol=1e-6)

    def test_rmsse_scale(self):
        A = np.array([[0, 0, 2, 4, 4, 1], [1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 3.0]])
        s = m5.rmsse_scale(A)
        np.testing.assert_allclose(s[:2], [np.mean([4, 0, 9]), 1.0])
        self.assertTrue(np.isnan(s[2]) and np.isnan(s[3]))


class TestMetrics(_Fixture):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.frame = cls.task.run(data=cls.data)

    def test_perfect(self):
        f = self.task.frame(self.data, {"oracle": self.data.test["sales"].to_numpy()})
        self.assertAlmostEqual(self.task.score(f, "oracle")["value"], 0.0)
        self.assertAlmostEqual(self.task.score(f, "oracle", "wmape")["value"], 0.0)

    def test_wrmsse_matches_definition(self):
        t = self.data.test
        pred = pd.DataFrame({"item_id": t["item_id"].astype(str), "store_id": t["store_id"].astype(str),
                             "d": t["d"].to_numpy(), "pred": self.frame["pred_snaive"].to_numpy()})
        ref, per = brute_wrmsse(self.fx, 1941, pred)
        s = self.task.score(self.frame, "snaive", B=200)
        self.assertAlmostEqual(s["value"], ref, places=6)
        self.assertLessEqual(s["lo"], s["value"])
        self.assertGreaterEqual(s["hi"], s["value"])
        by = m5.wrmsse_by_level(self.frame, "snaive")
        np.testing.assert_allclose(by["wrmsse"].to_numpy()[:12], per, rtol=1e-6)
        self.assertAlmostEqual(float(by["wrmsse"].iloc[-1]), ref, places=6)
        self.assertAlmostEqual(m5.wrmsse(self.frame, "snaive"), ref, places=6)

    def test_wmape(self):
        s = self.task.score(self.frame, "ses", "wmape", B=200)
        self.assertAlmostEqual(s["value"], fc.wmape(self.frame["truth"], self.frame["pred_ses"]))
        self.assertEqual(s["n"], len(self.fx["keys"]))
        a = self.task.score(self.frame, "ses", B=200, seed=1)
        b = self.task.score(self.frame, "ses", B=200, seed=1)
        self.assertEqual((a["lo"], a["hi"]), (b["lo"], b["hi"]))

    def test_score_all(self):
        tab = self.task.score_all(self.frame, B=50)
        self.assertEqual(set(tab["metric"]), {"wrmsse", "wmape"})
        self.assertEqual(set(tab["method"]), {"naive", "snaive", "ses", "ets_lite"})


class TestBaselines(_Fixture):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.preds = cls.task.baselines(cls.data)

    def test_rule_baselines(self):
        t, Y = self.data.test, self.fx["Y"]
        ids = list(self.fx["keys"]["item_id"] + "_" + self.fx["keys"]["store_id"])
        row = np.array([ids.index(x) for x in t["id"].astype(str)])
        h = t["h"].to_numpy()
        np.testing.assert_array_equal(self.preds["naive"], Y[row, 1940])
        src = 1941 + h - 7 * np.ceil(h / 7).astype(int)              # day index of the same weekday last week
        np.testing.assert_array_equal(self.preds["snaive"], Y[row, src - 1])
        for k in ("ses", "ets_lite"):
            self.assertTrue(np.isfinite(self.preds[k]).all() and (self.preds[k] >= 0).all())

    def test_ets_lite_uses_weekday(self):
        f = self.task.frame(self.data, self.preds)
        ses, ets = (self.task.score(f, k, B=20)["value"] for k in ("ses", "ets_lite"))
        self.assertLess(ets, ses)                                     # the fixture has a strong weekly profile
        s = pd.Series(self.preds["ets_lite"] / np.maximum(self.preds["ses"], 1e-12))
        prof = s.groupby(self.data.test["wday"].to_numpy()).median()
        self.assertGreater(prof.loc[2], prof.loc[4])                  # Sunday above Tuesday, as DOW

    def test_lightgbm(self):
        task = m5.M5(baselines="naive+snaive+lightgbm", lgb_origins=14, history=300)
        d = task.load(self.root)
        f = task.run(data=d)
        self.assertTrue(np.isfinite(f["pred_lightgbm"]).all() and (f["pred_lightgbm"] >= 0).all())
        lgb = task.score(f, "lightgbm", B=20)["value"]
        self.assertLess(lgb, task.score(f, "naive", B=20)["value"])
        self.assertLess(lgb, task.score(f, "snaive", B=20)["value"])     # needs weekday-aligned training origins

    def test_user_method_via_align(self):
        d = self.data

        def ma28(data):
            return m5.align(data, fc.moving_mean(m5.future_panel(data), data.meta["origin"], 28, window=28, **m5.COLS))
        f = self.task.run(methods={"ma28": ma28}, data=d, baselines=False)
        hist = d.train[d.train["d"] > 1941 - 28].groupby("id", observed=True)["sales"].mean()
        np.testing.assert_allclose(f["pred_ma28"], d.test["id"].map(hist).to_numpy(float), rtol=1e-6)
        fp = m5.future_panel(d)
        self.assertEqual(len(fp), len(d.train) + len(d.test))
        self.assertTrue(fp[fp["d"] > 1941]["sales"].isna().all())
        self.assertTrue(fp[fp["d"] > 1941]["sell_price"].notna().any())
        with self.assertRaises(ValueError):
            m5.align(d, pd.concat([pd.DataFrame({"id": ["x", "x"], "d": [1, 1], "pred": [0, 1]})]))


@unittest.skipUnless(os.environ.get("SIMENGINE_DATA"), "set $SIMENGINE_DATA to run the real-data check")
class TestRealData(unittest.TestCase):
    def test_store_subsample(self):
        task = m5.M5(stores="CA_1", n_series=300, baselines="snaive", history=60)
        try:
            data = task.load()
        except DataNotFound as e:
            self.skipTest(str(e))
        f = task.run(data=data)
        s = task.score(f, "snaive", B=100)
        self.assertEqual(len(f), 300 * 28)
        self.assertTrue(0 < s["value"] < 5)
        lv = data.meta["levels"]
        self.assertEqual(sorted(lv["level"].unique()), list(range(1, 13)))


if __name__ == "__main__":
    unittest.main()
