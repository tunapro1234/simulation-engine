"""evaluate(cluster=) overrides the bootstrap cluster (G3 #13: an item repeated across 18 countries), aggregate=
"unit_mean" (mean within each unit, then across units; D9 #3b) and the count of units aggregate_by drops (D9 #4d),
paired_diff(block_col=) with the same two-way bootstrap as evaluate (D7 #9), English names of the newer audit rows
and numbers kept by the "Turkish (English)" fallback (W8), evaluate(min_effect=) as the badge threshold (D12 #11).
Small synthetic frames, offline."""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.evidence import api, kart

FAST = dict(B=300, B_aggregate=40, multiplier=False)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _repeated_items(n_items=30, n_blocks=18, seed=0):
    """The same item (question x amount) in every block (country); ours carries a per-item bias shared by all blocks,
    so its errors are dependent within an item. group = the (item, block) cell."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n_items)
    bias = rng.normal(0, 0.4, n_items)
    rows = []
    for i in range(n_items):
        for c in range(n_blocks):
            y = e[i] + rng.normal(0, 0.2)
            rows.append({"item": f"q{i}", "country": f"c{c}", "group": f"q{i}-c{c}", "truth": y,
                         "pred_ours": y + bias[i] + rng.normal(0, 0.05), "pred_base": y + rng.normal(0, 0.3)})
    return pd.DataFrame(rows)


def _persons(n=40, seed=1, train_only=0):
    """Binary per-person accuracy that grows with the person's row count (pooled accuracy != mean of persons).
    The last `train_only` persons have only train rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        m = int(2 + 38 * k / (n - 1))
        acc = 0.5 + 0.45 * k / (n - 1)
        for j in range(m):
            y = int(rng.random() < 0.5)
            hit = rng.random() < acc
            rows.append({"person": f"p{k}", "item": f"i{j}", "truth": y, "pred_a": y if hit else 1 - y,
                         "pred_b": int(rng.random() < 0.5),
                         "split": "train" if (k >= n - train_only or j == 0) else "test"})
    return pd.DataFrame(rows)


def _origins(n_series=40, n_origins=6, seed=2):
    """Forecast origins with an origin-level shock in a's error: origin dependence the row bootstrap misses."""
    rng = np.random.default_rng(seed)
    shock = rng.normal(0, 0.6, n_origins)
    rows = []
    for o in range(n_origins):
        for s in range(n_series):
            y = rng.normal(5, 1)
            rows.append({"origin": o, "group": f"s{s}", "truth": y,
                         "pred_a": y + shock[o] + rng.normal(0, 0.1), "pred_b": y + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


class ClusterTest(unittest.TestCase):
    def test_item_cluster_widens_ci(self):
        df = _repeated_items()
        base = _quiet(api.evaluate, df, "t", ours="ours", primary="mae", **FAST)
        item = _quiet(api.evaluate, df, "t", ours="ours", primary="mae", cluster="item", **FAST)
        w = lambda r: api.summary(r)["ours"]["hi"] - api.summary(r)["ours"]["lo"]
        self.assertEqual(api.summary(base)["ours"]["value"], api.summary(item)["ours"]["value"])
        self.assertGreater(w(item), 2 * w(base))
        self.assertEqual(api.summary(item)["cluster"], "item")
        self.assertEqual(api.summary(base)["cluster"], "group")
        self.assertEqual(api.summary(item)["n_clusters"], 30)
        self.assertIn("cluster=`item`", api.card_markdown(item, lang="en"))

    def test_cluster_any_column_and_errors(self):
        df = _repeated_items(n_items=12, n_blocks=12)
        rep = _quiet(api.evaluate, df, "t", ours="ours", primary="mae", cluster="country", **FAST)
        self.assertEqual(api.summary(rep)["cluster"], "country")
        self.assertEqual(api.summary(rep)["n_clusters"], 12)
        with self.assertRaises(ValueError):
            api.evaluate(df, "t", ours="ours", cluster="nope", **FAST)

    def test_cluster_with_aggregate_by_must_be_constant(self):
        df = _repeated_items(n_items=12, n_blocks=12)
        rep = _quiet(api.evaluate, df, "t", ours="ours", aggregate_by="group", cluster="item", **FAST)
        self.assertEqual(api.summary(rep)["n_clusters"], 12)
        with self.assertRaises(ValueError):
            api.evaluate(df, "t", ours="ours", aggregate_by="item", cluster="country", **FAST)


class UnitMeanTest(unittest.TestCase):
    def test_mean_within_then_across_units(self):
        df = _persons()
        rep = _quiet(api.evaluate, df, "t", ours="a", primary="accuracy", aggregate_by="person",
                     aggregate="unit_mean", **FAST)
        t = df[df["split"] == "test"]
        per = (t["truth"] == t["pred_a"]).groupby(t["person"]).mean()
        pooled = float((t["truth"] == t["pred_a"]).mean())
        s = api.summary(rep)
        self.assertAlmostEqual(s["ours"]["value"], float(per.mean()), places=10)
        self.assertGreater(abs(pooled - float(per.mean())), 0.02)
        self.assertEqual(s["n_clusters"], 40)
        self.assertEqual(s["aggregate"]["aggregate"], "unit_mean")
        self.assertEqual(s["task"], "binary")     # rows are not averaged: the task stays binary

    def test_unknown_aggregate_still_rejected(self):
        with self.assertRaises(ValueError):
            api.evaluate(_persons(), "t", ours="a", aggregate_by="person", aggregate="median", **FAST)


class DroppedUnitsTest(unittest.TestCase):
    def _check(self, how):
        df = _persons(n=30, train_only=4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = api.evaluate(df, "t", ours="a", primary="accuracy" if how == "unit_mean" else None,
                               aggregate_by="person", aggregate=how, **FAST)
        a = rep["toplama"]
        self.assertEqual(a["n_units"], 30)
        self.assertEqual(a["n_dropped"], 4)
        self.assertEqual(a["n_rows_dropped"], int(df["person"].isin([f"p{k}" for k in range(26, 30)]).sum()))
        self.assertTrue(any("4 of 30" in str(x.message) for x in w), [str(x.message) for x in w])
        self.assertIn("4 of 30 units dropped", api.card_markdown(rep, lang="en"))
        self.assertEqual(api.summary(rep)["aggregate"]["n_dropped"], 4)

    def test_mean(self):
        self._check("mean")

    def test_unit_mean(self):
        self._check("unit_mean")

    def test_nothing_dropped_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = api.evaluate(_persons(n=20), "t", ours="a", aggregate_by="person", **FAST)
        self.assertEqual(rep["toplama"]["n_dropped"], 0)
        self.assertFalse(any("units dropped" in str(x.message) for x in w))


class PairedDiffBlockTest(unittest.TestCase):
    def test_block_matches_evaluate(self):
        df = _origins()
        rep = _quiet(api.evaluate, df, "t", ours="a", primary="mae", compare=[("a", "b")], block_col="origin",
                     B=400, B_aggregate=40, multiplier=False)
        r = next(c for c in rep["kayitli_karsilastirmalar"] if c.get("metrik") == "mae")
        pd_ = api.paired_diff(df, "a", "b", metric="mae", block_col="origin", B=400)
        self.assertAlmostEqual(pd_["lo"], r["alt"], places=9)
        self.assertAlmostEqual(pd_["hi"], r["ust"], places=9)
        self.assertEqual(pd_["block_col"], "origin")
        plain = api.paired_diff(df, "a", "b", metric="mae", B=400)
        self.assertGreater(pd_["hi"] - pd_["lo"], 1.5 * (plain["hi"] - plain["lo"]))
        self.assertIsNone(plain["block_col"])

    def test_report_reuses_block_col(self):
        df = _origins()
        rep = _quiet(api.evaluate, df, "t", ours="a", primary="mae", compare=[("a", "b")], block_col="origin",
                     B=200, B_aggregate=40, multiplier=False)
        self.assertEqual(api.paired_diff(rep, "a", "b")["block_col"], "origin")
        self.assertEqual(api.paired_diff(rep, "a", "b", block_col="origin")["block_col"], "origin")
        with self.assertRaises(ValueError):
            api.paired_diff(rep, "a", "b", block_col="group")

    def test_block_col_with_aggregate_by(self):
        df = _origins()
        df["cell"] = df["group"] + "-" + df["origin"].astype(str)
        r = api.paired_diff(df, "a", "b", metric="mae", aggregate_by="cell", block_col="origin", B=200)
        self.assertEqual(r["block_col"], "origin")
        with self.assertRaises(ValueError):
            api.paired_diff(df, "a", "b", block_col="nope", B=50)


class EnglishAuditNamesTest(unittest.TestCase):
    NAMES = ["sabit tahmin (sd 0)", "sabit tahmin: sıra metrikleri tanımsız (constant prediction)",
             "etkin örneklem çarpanı (PPI++ kaç insana bedel)",
             "ezber uyarısı ve bizim kol (memorization warning and ours)", "çerçeve ↔ mühürlü dosya",
             "zaman sütunu çözümü (time column parsing)"]

    def test_names_translated(self):
        tr = __import__("re").compile(r"[çğıöşüÇĞİÖŞÜ]")
        for n in self.NAMES:
            en = api._check_en(n)
            self.assertFalse(tr.search(en), (n, en))
            self.assertIn(n, api.CHECK_TO_ENGLISH)

    def test_generic_tail_fallback(self):
        self.assertEqual(api._check_en("yeni denetim adı (new audit name)"), "new audit name")

    def test_en_text_keeps_turkish_only_numbers(self):
        out = api._en_text("Yalnız 12 küme ve 3 blok kaldı, 0,25 pay (few clusters left)")
        for x in ("12", "3", "0.25", "few clusters left"):
            self.assertIn(x, out)
        self.assertEqual(api._en_text("ön kayıt belgesi değişmiş (pre-registration document changed)"),
                         "pre-registration document changed")
        self.assertEqual(api._en_text("Yalnız 12 küme (only 12 clusters left)"), "only 12 clusters left")

    def test_card_en_audit_names(self):
        df = _repeated_items(n_items=12, n_blocks=12)
        df["pred_flat"] = 0.0
        rep = _quiet(api.evaluate, df, "t", ours="ours", **FAST)
        names = [a["name"] for a in api.summary(rep)["audits"]]
        md = api.card_markdown(rep, lang="en")
        for n in names:
            self.assertNotRegex(n, r"[çğıöşüÇĞİÖŞÜ]")
        self.assertNotIn("sabit tahmin", md)


class MinEffectTest(unittest.TestCase):
    def _small_gain(self):
        rng = np.random.default_rng(5)
        n = 3000
        y = rng.normal(0, 1, n)
        return pd.DataFrame({"group": np.arange(n), "truth": y, "pred_base": y + rng.normal(0, 0.3, n),
                             "pred_ours": y + rng.normal(0, 0.28, n)})

    def test_threshold_changes_badge(self):
        df = self._small_gain()
        kw = dict(ours="ours", primary="mae", unlabeled_baselines=["base"], baselines=False, **FAST)
        d0 = _quiet(api.evaluate, df, "t", **kw)
        f = d0["kanit_kart"]["fark"]
        self.assertTrue(f["alt"] > 0 and f["fark"] < kart.ANLAMLI_FARK, f)
        self.assertEqual(d0["kanit_kart"]["rozet"], "kararsiz")
        d1 = _quiet(api.evaluate, df, "t", min_effect=f["fark"] / 2, **kw)
        self.assertEqual(d1["kanit_kart"]["rozet"], "yuksek")
        self.assertEqual(d1["min_effect"], f["fark"] / 2)
        d2 = _quiet(api.evaluate, df, "t", min_effect=0.009 if f["fark"] < 0.009 else f["fark"] * 2, **kw)
        self.assertEqual(d2["kanit_kart"]["rozet"], "kararsiz")
        md = api.card_markdown(d2, lang="en")
        thr = d2["min_effect"]
        self.assertIn(f"at least {thr:g} (min_effect=", md)
        self.assertIn(f"min_effect={thr:g}", md)
        self.assertIn(f"{thr:g}", api.summary(d2)["badge_reason_en"])

    def test_default_unchanged(self):
        self.assertEqual(kart.ANLAMLI_FARK, 0.02)
        self.assertEqual(kart.rozet([{"tur": "kalibre", "deger": 1.0}], {"fark": 0.01, "alt": 0.001, "ust": 0.02},
                                    taban_var=True)[0], "kararsiz")
        self.assertEqual(kart.rozet([{"tur": "kalibre", "deger": 1.0}], {"fark": 0.01, "alt": 0.001, "ust": 0.02},
                                    taban_var=True, anlamli_fark=0.005)[0], "yuksek")
        with self.assertRaises(ValueError):
            api.evaluate(self._small_gain(), "t", ours="ours", min_effect=-1, **FAST)


class ReviewW9Test(unittest.TestCase):
    """Adversarial review of the five fixes above."""

    def _uneven(self, seed=0):
        """Continuous rows; persons with many rows predict well, persons with few predict badly."""
        rng = np.random.default_rng(seed)
        rows = []
        for k in range(30):
            for _ in range(2 + 3 * k):
                y = rng.normal()
                rows.append({"person": f"p{k}", "truth": y, "pred_a": y + rng.normal(0, 0.2 if k > 20 else 2.0),
                             "pred_b": rng.normal()})
        return pd.DataFrame(rows)

    def test_unit_mean_pooled_primary_is_flagged(self):
        # continuous default primary is Pearson (pooled, unweighted): unit_mean used to return the plain row value
        # with "level: unit mean" on the card
        df = self._uneven()
        plain = _quiet(api.evaluate, df, "t", ours="a", **FAST)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            um = api.evaluate(df, "t", ours="a", aggregate_by="person", aggregate="unit_mean", **FAST)
        self.assertEqual(api.summary(um)["ours"]["value"], api.summary(plain)["ours"]["value"])
        self.assertEqual(um["toplama"]["pooled_primary"], "pearson")
        self.assertTrue(any("does not apply to the primary metric pearson" in str(x.message) for x in w))
        md = api.card_markdown(um, lang="en")
        self.assertIn("pearson is pooled, NOT a unit mean", md)
        self.assertNotIn("level: unit mean", md)
        self.assertNotRegex(md.split("pooled metrics take no row weights")[0].rsplit("\n", 1)[-1], r"[çğıöşü]")
        mae = _quiet(api.evaluate, df, "t", ours="a", primary="mae", aggregate_by="person", aggregate="unit_mean",
                     **FAST)
        self.assertNotIn("pooled_primary", mae["toplama"])
        self.assertIn("level: unit mean", api.card_markdown(mae, lang="en"))

    def test_unit_mean_imbalanced_binary_default_flagged(self):
        rng = np.random.default_rng(3)
        rows = []
        for k in range(30):
            for _ in range(2 + 2 * k):
                y = int(rng.random() < 0.85)
                rows.append({"person": f"p{k}", "truth": y, "pred_a": y if rng.random() < 0.5 + k / 60 else 1 - y,
                             "pred_b": 1})
        rep = _quiet(api.evaluate, pd.DataFrame(rows), "t", ours="a", aggregate_by="person", aggregate="unit_mean",
                     **FAST)
        self.assertEqual(rep["toplama"].get("pooled_primary"), "macro_f1")

    def test_paired_diff_unit_mean_and_unknown_aggregate(self):
        df = _persons()
        with self.assertRaises(ValueError):
            api.paired_diff(df, "a", "b", aggregate_by="person", aggregate="median", B=50)
        with self.assertRaises(ValueError):
            api.paired_diff(df, "a", "b", aggregate="unit_mean", B=50)
        rep = _quiet(api.evaluate, df, "t", ours="a", primary="accuracy", aggregate_by="person",
                     aggregate="unit_mean", compare=[("a", "b")], **FAST)
        r = next(c for c in rep["kayitli_karsilastirmalar"] if c.get("metrik") == "dogruluk")
        um = api.paired_diff(df, "a", "b", metric="accuracy", aggregate_by="person", aggregate="unit_mean", B=300)
        mean = api.paired_diff(df, "a", "b", aggregate_by="person", B=300)
        self.assertAlmostEqual(um["diff"], r["fark"], places=10)
        self.assertAlmostEqual(um["lo"], r["alt"], places=9)
        self.assertEqual(um["metric"], "accuracy")
        self.assertNotEqual(mean["metric"], "accuracy")      # the group mean turns the task continuous

    def test_dropped_units_split_by_cause(self):
        df = _persons(n=30, train_only=4)
        df.loc[(df["person"] == "p3") & (df["split"] == "test"), "pred_a"] = np.nan
        rep = _quiet(api.evaluate, df, "t", ours="a", aggregate_by="person", **FAST)
        a = rep["toplama"]
        self.assertEqual((a["n_dropped"], a["n_dropped_no_test_row"], a["n_dropped_missing"]), (5, 4, 1))

    def test_en_text_keeps_numbers_without_turkish_words(self):
        out = api._en_text("pearson: ours − b = -0.127 [-0.257, 0.022], p=0.200 · berabere: eşleşik GA 0'ı "
                           "içeriyor (tied: paired CI includes 0)")
        self.assertNotRegex(out, r"[çğıöşüÇĞİÖŞÜ]")
        for x in ("-0.127", "-0.257", "0.022", "p=0.200", "tied: paired CI includes 0"):
            self.assertIn(x, out)

    def test_constant_prediction_detail_english(self):
        df = _repeated_items(n_items=12, n_blocks=12)
        df["pred_flat"] = 0.0
        df.loc[df["item"] == "q0", "pred_ours"] = 1.0
        rep = _quiet(api.evaluate, df, "t", ours="ours", **FAST)
        md = api.card_markdown(rep, lang="en")
        rows = [ln for ln in md.splitlines() if ln.startswith("| constant prediction")]
        self.assertTrue(rows, md)
        for ln in rows:
            self.assertNotRegex(ln, r"[çğıöşüÇĞİÖŞÜ]")


if __name__ == "__main__":
    unittest.main()
