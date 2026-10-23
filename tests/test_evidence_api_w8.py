"""evaluate(): average precision as a binary metric (primary="ap" / "average_precision" / "pr_auc" with the bootstrap
CI; D13 #5), total and segment rate error rows (rate_error_by=, paired cluster bootstrap; D13 #10), the English
headline of an awaiting-truth card naming ours' seal (D11 #7, D12 #18), the pending card printing the per-item and
label summaries it computes (D9 #9, D12 #16a) and English names of the persona / dropped-field card rows.
Small synthetic frames, offline."""
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import simengine.evidence as ev
from simengine.evidence import api, kart, metrik

FAST = dict(B=200, B_aggregate=40, multiplier=False)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _binary(n=600, seed=0):
    """Binary truth with a segment-dependent rate; `seg` knows the segment rate, `flat` only the overall rate, `good`
    sees the latent score. Scores rounded to 2 decimals so AP has tied thresholds."""
    rng = np.random.default_rng(seed)
    seg = rng.integers(0, 4, n)
    rate = np.array([0.05, 0.15, 0.35, 0.6])[seg]
    y = (rng.random(n) < rate).astype(int)
    return pd.DataFrame({"unit": np.arange(n), "segment": [f"s{k}" for k in seg], "truth": y,
                         "pred_good": np.round(np.clip(rate + rng.normal(0, 0.1, n) + 0.2 * (y - rate), 0.01, 0.99), 2),
                         "pred_seg": rate, "pred_flat": np.full(n, y.mean())})


class AveragePrecisionTest(unittest.TestCase):
    def test_matches_sklearn_with_ties(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            n = int(rng.integers(5, 300))
            y = (rng.random(n) < rng.uniform(0.05, 0.6)).astype(float)
            if y.sum() == 0:
                y[0] = 1
            p = np.round(rng.random(n), int(rng.integers(0, 3)))     # 0 decimals: two thresholds, heavy ties
            self.assertAlmostEqual(metrik._ap(y, p), average_precision_score(y, p), places=12)

    def test_no_positive_is_nan(self):
        self.assertTrue(np.isnan(metrik._ap(np.zeros(5), np.linspace(0, 1, 5))))

    def test_empty_or_nonfinite_score_is_nan(self):
        # sklearn raises on NaN scores (-> nan via _guvenli); _ap used to return a number (0.75 here)
        self.assertTrue(np.isnan(metrik._ap(np.array([0, 1, 0, 1.]), np.array([.1, np.nan, .3, .4]))))
        self.assertTrue(np.isnan(metrik._ap(np.array([]), np.array([]))))

    def test_fast_bootstrap_path_matches_expanded_sample(self):
        # bootstrap row multiplicities (> HIZLI_SIRA_ESIK rows) must give the AP of the expanded resample
        from simengine.evidence import bootstrap
        rng = np.random.default_rng(4)
        m = metrik.Metrik("ap", "toplu", metrik._ap, 1)
        for t in range(80):
            n = int(rng.integers(3, 60))
            y = (rng.random(n) < rng.random()).astype(float)
            p = np.round(rng.random(n), int(t % 3)) if t % 7 else np.full(n, 0.4)
            k = bootstrap._sira_kodu(y, p, None)
            w = rng.integers(0, 3, n).astype(float)
            w[0] += 1
            self.assertIsNotNone(bootstrap._hizli_metrik(m, k))
            ii = np.repeat(np.arange(n), w.astype(int))
            a, b = bootstrap._hizli_metrik(m, k)(w), metrik._ap(y[ii], p[ii])
            if np.isnan(b):
                self.assertTrue(np.isnan(a))
            else:
                self.assertAlmostEqual(a, b, places=12)

    def test_in_binary_family(self):
        fam = {m.ad: m for m in metrik.aile("ikili")}
        self.assertIn("ap", fam)
        self.assertEqual(fam["ap"].yon, +1)
        self.assertEqual(fam["ap"].tur, "toplu")

    def test_primary_aliases_with_ci(self):
        df = _binary()
        for name in ("ap", "average_precision", "pr_auc"):
            rep = _quiet(ev.evaluate, df, ours="good", primary=name, **FAST)
            self.assertEqual(rep["birincil"], "ap")
            r = rep["sonuclar"]["tahmin_good"]["metrikler"]["ap"]
            self.assertAlmostEqual(r["deger"], average_precision_score(df["truth"], df["pred_good"]), places=10)
            self.assertLess(r["alt"], r["deger"])
            self.assertGreater(r["ust"], r["deger"])
        s = ev.summary(rep)
        self.assertEqual(s["primary_metric"], "average_precision")
        self.assertIsNotNone(s["ours"]["lo"])


class RateErrorTest(unittest.TestCase):
    def _rep(self, **kw):
        return _quiet(ev.evaluate, _binary(), ours="good", unlabeled_baselines=["flat"], **FAST, **kw)

    def test_values_match_the_definition(self):
        df = _binary()
        rep = self._rep(rate_error_by="segment")
        oh = rep["oran_hatasi"]
        self.assertEqual(oh["segmentler"], ["segment"])
        rows = {r["sutun"]: r for r in oh["satirlar"]}
        for m in ("good", "seg", "flat"):
            e = df[f"pred_{m}"] - df["truth"]
            r = rows[f"tahmin_{m}"]
            self.assertAlmostEqual(r["toplam"]["deger"], abs(e.sum()) / len(df), places=12)
            seg = e.groupby(df["segment"]).sum().abs().sum() / len(df)
            self.assertAlmostEqual(r["segment"]["segment"]["deger"], seg, places=12)
            # |.| is >= 0 in every replicate: near a zero gap the percentile CI lies above the point (pairs decide)
            self.assertGreaterEqual(r["toplam"]["alt"], 0)
            self.assertLessEqual(r["toplam"]["alt"], r["toplam"]["ust"])
        # flat predicts the overall rate: total error 0, but it misses every segment rate
        self.assertAlmostEqual(rows["tahmin_flat"]["toplam"]["deger"], 0.0, places=12)
        self.assertGreater(rows["tahmin_flat"]["segment"]["segment"]["deger"],
                           rows["tahmin_seg"]["segment"]["segment"]["deger"] + 0.05)
        self.assertIn("taban_marjinal", rows)      # automatic baselines are scored too
        sb = _quiet(ev.segment_bias, df, "segment", ["good", "flat"], B=50)    # frames.segment_bias, same definition
        for m in ("good", "flat"):
            self.assertAlmostEqual(rows[f"tahmin_{m}"]["segment"]["segment"]["deger"], sb["arms"][m]["value"], places=12)

    def test_paired_differences(self):
        rep = self._rep(rate_error_by="segment")
        d = {(f["b"], f["olcu"]): f for f in rep["oran_hatasi"]["farklar"]}
        f = d[("tahmin_flat", "segment")]
        rows = {r["sutun"]: r for r in rep["oran_hatasi"]["satirlar"]}
        self.assertAlmostEqual(f["fark"], rows["tahmin_good"]["segment"]["segment"]["deger"]
                               - rows["tahmin_flat"]["segment"]["segment"]["deger"], places=12)
        self.assertLess(f["ust"], 0)               # ours has the lower (better) segment rate error, CI below 0
        self.assertLess(f["p"], 0.05)
        self.assertTrue(f["a_iyi"])
        self.assertIn(("tahmin_flat", "toplam"), d)

    def test_named_column_summary_and_card(self):
        df = _binary()
        df["region"] = np.where(df["unit"] % 2 == 0, "north", "south")
        rep = _quiet(ev.evaluate, df, ours="good", rate_error_by="region", **FAST)
        self.assertEqual(rep["oran_hatasi"]["segmentler"], ["region"])
        s = ev.summary(rep)
        self.assertEqual(s["rate_error"]["segment_columns"], ["region"])
        self.assertTrue(any(r["column"] == "pred_good" for r in s["rate_error"]["rows"]))
        for lang in ("en", "mixed"):
            md = ev.card(rep, lang=lang)
            self.assertIn("Rate error", md)
            self.assertIn("lower is better", md)
            self.assertIn("region", md)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            self._rep(rate_error_by="nope")
        df = _binary().drop(columns="segment")
        with self.assertRaises(ValueError):
            _quiet(ev.evaluate, df, ours="good", rate_error_by="segment", **FAST)
        dist = pd.DataFrame({"item": ["a", "b", "c"] * 4, "segment": ["x", "y"] * 6,
                             "truth": [[0.2, 0.8]] * 12, "pred_m": [[0.5, 0.5]] * 12})
        with self.assertRaises(ValueError):
            _quiet(ev.evaluate, dist, ours="m", rate_error_by="segment", **FAST)

    def test_off_by_default(self):
        self.assertNotIn("oran_hatasi", self._rep())


def _pending(seed=0):
    rng = np.random.default_rng(seed)
    n = 30
    return pd.DataFrame({"item": np.repeat(["a", "b", "c"], n // 3), "group": [f"p{i}" for i in range(n)],
                         "pred_lgbm": rng.normal(0, 1, n), "pred_ours": rng.normal(0, 1, n)})


class PendingCardTest(unittest.TestCase):
    def test_english_headline_names_ours_seal(self):
        df = _pending()
        with tempfile.TemporaryDirectory() as d:
            z0 = ev.seal(df, "ours: ours, final submission", registry=d, store_copy=False)
            z1 = ev.seal(df[["item", "group", "pred_lgbm"]], "lgbm baseline", registry=d, store_copy=False)
            rep = _quiet(ev.evaluate, df, ours="ours", task="continuous", prereg=d)
        s = ev.summary(rep)
        self.assertIn(z0["sha256"][:12], s["headline_en"])
        self.assertNotIn(z1["sha256"][:12], s["headline_en"])
        self.assertEqual(s["ours_seal"]["sha256"], z0["sha256"])
        self.assertIn(z0["sha256"][:12], ev.card(rep, lang="en").split("## ")[0])

    def test_unmatched_ours_says_last(self):
        df = _pending()
        with tempfile.TemporaryDirectory() as d:
            ev.seal(df[["item", "group", "pred_lgbm"]], "first", registry=d, store_copy=False)
            z = ev.seal(df[["item", "group", "pred_lgbm"]].iloc[:20], "second", registry=d, store_copy=False)
            rep = _quiet(ev.evaluate, df, ours="ours", task="continuous", prereg=d)
        h = ev.summary(rep)["headline_en"]
        self.assertIn(z["sha256"][:12], h)
        self.assertIn("ours not matched", h)

    def test_per_item_summary_printed(self):
        df = _pending()
        df.loc[df["item"] == "b", "pred_ours"] = 0.5
        rep = _quiet(ev.evaluate, df, ours="ours", task="continuous")
        for lang in ("en", "mixed"):
            md = ev.card(rep, lang=lang)
            self.assertIn("Predictions per item", md)
            self.assertIn("1/3 items constant", md)
            sec = md.split("Predictions per item")[1].split("\n## ")[0]
            for it in ("| a |", "| b |", "| c |"):
                self.assertIn(it, sec)
            self.assertIn("0.500 ± 0.000", sec)

    def test_label_summary_printed(self):
        df = _pending()
        rng = np.random.default_rng(2)
        for c in ("pred_lgbm", "pred_ours"):
            df[c] = rng.integers(1, 6, len(df))
        rep = _quiet(ev.evaluate, df, ours="ours")
        oz = rep["tahmin_ozeti"]["tahmin_ours"]
        top = next(iter(oz["en_sik"]))
        for lang in ("en", "mixed"):
            md = ev.card(rep, lang=lang)
            self.assertIn("| distinct |", md)
            row = next(r for r in md.splitlines() if "pred_ours" in r or "tahmin_ours" in r)
            self.assertIn("| 5 |", row)
            self.assertIn(f"{top} ", row)
            sec = md.split("Predictions per item")[1].split("\n## ")[0]
            self.assertRegex(sec, r"\| a \| .*%")


class PersonaRowsTest(unittest.TestCase):
    def test_row_names_in_english(self):
        rep = _quiet(ev.evaluate, _binary(), ours="good", **FAST)
        kart.persona_ekle(rep["kanit_kart"], {"n_fields": 6, "dropped": ["income"], "n_people": 30,
                                              "per_field_missing": {"age": 2}})
        md = ev.card(rep, lang="en")
        self.assertIn("| Persona fields |", md)
        self.assertIn("| Dropped fields |", md)
        self.assertNotIn("| persona |", md)
        self.assertNotIn("| dusen |", md)
        self.assertIn("persona", api.ROW_TO_ENGLISH)
        self.assertIn("dusen", api.ROW_TO_ENGLISH)


if __name__ == "__main__":
    unittest.main()
