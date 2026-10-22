"""D4 follow-up: block bootstrap (block_col), single-origin warning, mean/median card row, seasonal_ratio (T4/T4b)."""
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine as se
import simengine.methods as M
from simengine.evidence import api


def forecast_frame(n_items=30, origins=(10, 14, 18), shock=0.6, seed=0):
    """Long forecast frame: per-origin multiplicative shock shared by all series (origin-to-origin variation)."""
    rng = np.random.default_rng(seed)
    rows = []
    for o in origins:
        s = np.exp(rng.normal(0, shock))
        for i in range(n_items):
            lev = 5 + i
            y = rng.poisson(lev * s, 4).astype(float)
            for h in range(4):
                rows.append({"item": f"s{i}", "origin": o, "time": o + h + 1, "truth": y[h],
                             "pred_ours": lev * 1.0, "pred_ours_median": np.floor(lev * 0.97),
                             "pred_naive": lev * 0.8})
    return pd.DataFrame(rows)


def _ev(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return se.evaluate(df, "t", ours="ours", task="forecast", unlabeled_baselines=["naive"], B=300,
                           B_aggregate=100, **kw)


def _ci_width(rep, col="tahmin_ours", met="wmape"):
    m = rep["sonuclar"][col]["metrikler"][met]
    return m["ust"] - m["alt"]


class TestBlockBootstrap(unittest.TestCase):
    def test_block_resampling_widens_ci_with_origin_shock(self):
        df = forecast_frame()
        base = _ev(df)
        blk = _ev(df, block_col="origin")
        self.assertNotIn("blok_bootstrap", base)
        self.assertEqual(blk["blok_bootstrap"], {"sutun": "origin", "n_blok": 3})
        self.assertEqual(blk["kart"]["kume_adi"], "madde")      # two-way: clusters stay series, blocks reported apart
        self.assertEqual(blk["kart"]["n_kume"], 30)
        self.assertEqual(base["kart"]["kume_adi"], "madde")
        # point estimates identical, CI wider (shared origin shock is only visible to the block bootstrap)
        self.assertAlmostEqual(base["sonuclar"]["tahmin_ours"]["metrikler"]["wmape"]["deger"],
                               blk["sonuclar"]["tahmin_ours"]["metrikler"]["wmape"]["deger"])
        self.assertGreater(_ci_width(blk), _ci_width(base))
        s = se.summary(blk)
        self.assertEqual(s["block_bootstrap"], {"column": "origin", "n_blocks": 3})
        self.assertIn("blocks of `origin` (3 blocks) × clusters", se.card(blk, lang="en"))
        self.assertTrue(any("block_col=origin" in x for x in blk["sinirlar"]))
        self.assertNotIn("single_origin", s)

    def test_time_alias_and_paired_compare_use_blocks(self):
        df = forecast_frame()
        rep = _ev(df, block_col="time", compare=[("ours", "naive")])
        self.assertEqual(rep["blok_bootstrap"]["sutun"], "zaman")
        self.assertEqual(rep["kayitli_karsilastirmalar"][0]["n"], rep["kart"]["n_kume"])

    def test_missing_block_col_raises(self):
        with self.assertRaises(ValueError):
            _ev(forecast_frame(), block_col="nope")

    def test_block_col_must_be_constant_within_aggregate_groups(self):
        df = forecast_frame()
        df["grp"] = df["item"]
        with self.assertRaises(ValueError):
            _ev(df, aggregate_by="grp", aggregate="sum", block_col="origin")
        df["so"] = df["item"] + "_" + df["origin"].astype(str)
        rep = _ev(df, aggregate_by="so", aggregate="sum", block_col="origin")
        self.assertEqual(rep["blok_bootstrap"]["n_blok"], 3)

    def test_non_forecast_block(self):
        df = forecast_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = se.evaluate(df, "t", ours="ours", task="continuous", B=200, B_aggregate=50, multiplier=False,
                              block_col="origin")
        self.assertEqual(rep["blok_bootstrap"]["sutun"], "origin")


class TestSingleOrigin(unittest.TestCase):
    def test_single_origin_warning_from_origin_column(self):
        df = forecast_frame(origins=(10,))
        rep = _ev(df)
        self.assertEqual(rep["tek_koken"]["sutun"], "origin")
        row = [a for a in rep["denetim"] if a["kontrol"].startswith("tek tahmin kökeni")]
        self.assertEqual(len(row), 1)
        self.assertEqual(row[0]["durum"], "uyari")
        card = se.card(rep, lang="en")
        self.assertIn("single forecast origin: CIs cover series sampling only, not origin-to-origin variation; "
                      "use methods.forecast.backtest(origins=[...])", card)

    def test_single_block_falls_back_and_warns(self):
        df = forecast_frame(origins=(10,))
        rep = _ev(df, block_col="origin")
        self.assertNotIn("blok_bootstrap", rep)
        self.assertEqual(rep["kart"]["kume_adi"], "madde")
        self.assertIn("single_origin", se.summary(rep))

    def test_no_warning_without_time_columns_or_outside_forecast(self):
        df = forecast_frame(origins=(10,)).drop(columns=["origin", "time"])
        self.assertNotIn("tek_koken", _ev(df))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = se.evaluate(forecast_frame(origins=(10,)), "t", ours="ours", task="continuous", B=100,
                              B_aggregate=50, multiplier=False)
        self.assertNotIn("tek_koken", rep)


class TestMeanMedianRow(unittest.TestCase):
    def test_auto_pair_ours_median(self):
        rep = _ev(forecast_frame(), point_pairs="auto")
        mm = se.summary(rep)["mean_median"]
        self.assertEqual(len(mm), 1)
        self.assertEqual((mm[0]["mean"], mm[0]["median"], mm[0]["metric"]), ("ours", "ours_median", "wmape"))
        self.assertLess(mm[0]["median_bias"], mm[0]["mean_bias"])
        card = se.card(rep, lang="en")
        self.assertIn("mean vs median (wmape)", card)
        self.assertIn("biases totals down", card)

    def test_explicit_pairs_and_non_l1(self):
        df = forecast_frame().rename(columns={"pred_ours_median": "pred_med"})
        self.assertNotIn("ortalama_medyan", _ev(df))
        rep = _ev(df, point_pairs=[("pred_ours", "med")], primary="mae")
        self.assertEqual(rep["ortalama_medyan"][0]["metrik"], "mae")
        self.assertNotIn("ortalama_medyan", _ev(df, point_pairs=[("ours", "med")], primary="rmse"))
        with self.assertRaises(ValueError):
            _ev(df, point_pairs=[("ours", "missing")])


class TestSeasonalRatio(unittest.TestCase):
    def test_t4_values_and_format(self):
        p = pd.DataFrame({"id": ["a"] * 12, "time": range(12), "y": np.arange(12, dtype=float)})
        fr = M.seasonal_ratio(p, 9, 2, window=2, season=4, cap=None)
        self.assertEqual(list(fr.columns), list(M.moving_mean(p, 9, 2, window=2).columns))
        # m = mean(8, 9) = 8.5; base = mean(4, 5) = 4.5; h=1: ly = mean(5, 6, 7) = 6; h=2: ly = mean(6, 7, 8) = 7
        np.testing.assert_allclose(fr.pred.to_numpy(), [8.5 * 7 / 5.5, 8.5 * 8 / 5.5])
        capped = M.seasonal_ratio(p, 9, 1, window=2, season=4, cap=(0.5, 1.1))
        self.assertAlmostEqual(capped.pred.iloc[0], 8.5 * 1.1)
        self.assertTrue(np.isnan(M.seasonal_ratio(p, 3, 1, window=2, season=4).pred).all())

    def test_t4b_pooled_ratio(self):
        t = np.arange(10)
        p = pd.concat([pd.DataFrame({"id": "a", "g": "x", "time": t, "y": np.where(t % 4 == 2, 4.0, 1.0)}),
                       pd.DataFrame({"id": "b", "g": "x", "time": t, "y": np.full(10, 2.0)})], ignore_index=True)
        fr = M.seasonal_ratio(p, 9, 1, window=2, season=4, around=0, smooth=0, group_col="g", cap=None)
        # origin p=9, k = 6: ly = (4 + 2) = 6; base (positions 4, 5): a 1, b 2 -> 3; ratio 2
        got = fr.set_index("id").pred
        self.assertAlmostEqual(got["a"], 1.0 * 2)   # a: mean(y8, y9) = 1
        self.assertAlmostEqual(got["b"], 2.0 * 2)
        self.assertIs(se.seasonal_ratio, M.seasonal_ratio)


if __name__ == "__main__":
    unittest.main()


def bias_frame(n_items=40, n_origins=12, seed=1):
    """Fixed per-series bias N(0, 3), no origin shock: dependence lives inside series, across blocks."""
    rng = np.random.default_rng(seed)
    bias = rng.normal(0, 3, n_items)
    rows = []
    for o in range(n_origins):
        for i in range(n_items):
            lev = 10 + i % 7
            for h in range(4):
                rows.append({"item": f"s{i}", "origin": o, "time": o + h + 1, "truth": float(rng.poisson(lev)),
                             "pred_ours": lev + bias[i], "pred_naive": lev * 0.8})
    return pd.DataFrame(rows)


class TestReviewFollowUp(unittest.TestCase):
    def test_few_blocks_warn_and_card_note(self):           # finding 1
        rep = _ev(forecast_frame(origins=(10, 14)), block_col="origin")
        row = [a for a in rep["denetim"] if a["kontrol"] == "blok bootstrap blok sayısı"]
        self.assertEqual(len(row), 1)
        self.assertEqual(row[0]["durum"], "uyari")
        self.assertTrue(any("CI not reliable" in x for x in rep["sinirlar"]))
        self.assertIn("CI not reliable across blocks", se.card(rep, lang="en"))
        many = _ev(forecast_frame(origins=tuple(range(10, 22))), block_col="origin")
        self.assertFalse([a for a in many["denetim"] if a["kontrol"] == "blok bootstrap blok sayısı"])

    def test_two_way_keeps_series_clustering(self):         # finding 2
        df = bias_frame()
        cmp_ = [("ours", "naive")]
        base, blk = _ev(df, compare=cmp_), _ev(df, block_col="origin", compare=cmp_)
        self.assertGreaterEqual(_ci_width(blk), _ci_width(base))       # old one-way block: 0.036 vs 0.049
        hb, hk = base["kayitli_karsilastirmalar"][0], blk["kayitli_karsilastirmalar"][0]
        self.assertGreaterEqual(hk["ust"] - hk["alt"], 0.95 * (hb["ust"] - hb["alt"]))

    def test_two_way_sampler_weights_are_products(self):
        from simengine.evidence.bootstrap import Orneklem
        hb, hc = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
        o = Orneklem(np.arange(4), B=5, iki_yon=(hb, hc))
        for b in range(5):
            rng = np.random.default_rng([0, b])
            cb, cc = np.bincount(rng.integers(0, 2, 2), minlength=2), np.bincount(rng.integers(0, 2, 2), minlength=2)
            np.testing.assert_array_equal(o.sayim(b), (cb[hb] * cc[hc]).astype(float))

    def test_single_origin_independent_of_block_col(self):  # finding 3
        rep = _ev(forecast_frame(origins=(10,)), block_col="time")
        self.assertEqual(se.summary(rep)["single_origin"]["column"], "origin")
        df = forecast_frame()
        df["one"] = 1
        rep = _ev(df, block_col="one")
        self.assertNotIn("tek_koken", rep)
        self.assertNotIn("single forecast origin", se.card(rep, lang="en"))

    def test_nan_block_col_raises_clearly(self):             # finding 4
        df = forecast_frame()
        df.loc[:39, "origin"] = np.nan
        with self.assertRaisesRegex(ValueError, r"block_col 'origin' has 40 missing"):
            _ev(df, block_col="origin")

    def test_limits_text_consistent_under_blocks(self):      # finding 5
        rep = _ev(forecast_frame(), block_col="origin")
        first = rep["sinirlar"][0]
        self.assertNotIn("Seriler (madde) sabit", first)
        self.assertIn("30 küme × 3 'origin' bloğu", first)
        self.assertFalse(any("blok içi kümeler ayrıca örneklenmez" in x for x in rep["sinirlar"]))

    def test_point_pairs_default_off(self):                  # finding 6
        rep = _ev(forecast_frame())
        self.assertNotIn("ortalama_medyan", rep)
        self.assertNotIn("mean_median", se.summary(rep))

    def test_mean_median_sentence_reports_observed_direction(self):   # finding 7
        card = se.card(_ev(forecast_frame(), point_pairs="auto"), lang="en")
        self.assertNotIn("helps L1 metrics (MAE / WMAPE) but biases", card)
        self.assertIn("wmape higher with the median", card)
        self.assertIn("|bias| smaller", card)

    def test_explicit_point_pairs_validated_and_warned(self):          # finding 8
        df = forecast_frame()
        kw = dict(B=100, B_aggregate=50, multiplier=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(ValueError):
                se.evaluate(df, "t", ours="ours", task="continuous", point_pairs=[("ours", "nope")], **kw)
            with self.assertRaises(ValueError):
                _ev(df, point_pairs=[("ours", "nope")], primary="rmse")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = se.evaluate(df, "t", ours="ours", task="forecast", unlabeled_baselines=["naive"], B=100,
                              B_aggregate=50, point_pairs=[("ours", "ours_median")], primary="rmse")
        self.assertNotIn("ortalama_medyan", rep)
        self.assertTrue(any("point_pairs ignored" in str(x.message) for x in w))

    def test_seasonal_ratio_wraps_long_horizon(self):        # finding 9
        rng = np.random.default_rng(0)
        T = 30
        panel = pd.DataFrame([{"id": "a", "time": t, "y": float(rng.poisson(10))} for t in range(T)])
        y = panel["y"].to_numpy()
        p = 20
        out = M.forecast.seasonal_ratio(panel, origin=p, horizon=[5, 6], window=2, season=4, around=1,
                                        smooth=0, cap=None)
        m = y[p - 1:p + 1].mean()
        for h, v in zip((5, 6), out["pred"].to_numpy()):
            k = p + h - 8
            want = m * y[k - 1:min(k + 1, p) + 1].mean() / y[p - 1 - 8:p - 8 + 1].mean()
            self.assertAlmostEqual(float(v), float(want), places=9)
