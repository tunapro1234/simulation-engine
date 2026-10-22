"""D4 rehearsal gaps (demand forecasting): wmape/bias, aggregate="sum", ci="basic", task="forecast",
selection-bias evidence, seal file hash, no-LLM memorization flag. Small synthetic data, offline."""
import hashlib
import os
import time
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from simengine import evidence as ev
from simengine.evidence import bootstrap as boot
from simengine.evidence import metrik


def _frame(n_items=40, weeks=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        lam = rng.gamma(2.0, 3.0)
        for w in range(weeks):
            y = rng.poisson(lam)
            rows.append({"item": f"i{i}", "store": f"s{i % 4}", "time": f"2016-0{w + 1}-01", "truth": float(y),
                         "pred_good": lam, "pred_naive": max(lam + rng.normal(0, 3), 0.0)})
    return pd.DataFrame(rows)


def _wmape(y, p):
    return np.abs(y - p).sum() / np.abs(y).sum()


class TestRatioMetrics(unittest.TestCase):
    def test_wmape_and_bias_point_and_bootstrap(self):
        df = _frame()
        r = ev.paired_diff(df, "good", "naive", metric="wmape", group="item", B=300)
        self.assertAlmostEqual(r["value_a"], _wmape(df.truth, df.pred_good), places=10)
        self.assertAlmostEqual(r["value_b"], _wmape(df.truth, df.pred_naive), places=10)
        self.assertEqual(r["metric"], "wmape")
        self.assertLess(r["lo"], r["hi"])
        b = ev.paired_diff(df, "good", None, metric="bias", group="item", B=200)
        self.assertAlmostEqual(b["value_a"], df.pred_good.sum() / df.truth.sum() - 1, places=10)

    def test_ratio_of_sums_per_replicate(self):
        # clusters resampled together: replicate value equals ratio of resampled sums
        y = np.array([1.0, 2.0, 10.0, 0.0])
        p = np.array([2.0, 2.0, 5.0, 1.0])
        kume = np.array([0, 0, 1, 2])
        orn = boot.Orneklem(kume, B=5, B_toplu=5)
        nokta, tekrar = boot.hesapla(orn, {"x": (y, p, None)}, [metrik.WMAPE])
        self.assertAlmostEqual(nokta["x"]["wmape"], 7.0 / 13.0)
        c = orn.sayim(0)
        num = np.bincount(kume, np.abs(y - p), 3) @ c
        den = np.bincount(kume, np.abs(y), 3) @ c
        self.assertAlmostEqual(tekrar["x"]["wmape"][0], num / den if den else np.nan)

    def test_evaluate_primary_wmape_and_bias_not_primary(self):
        df = _frame()
        rep = ev.evaluate(df, "t", ours="good", primary="wmape", multiplier=False, B=200, B_aggregate=50)
        self.assertEqual(rep["birincil"], "wmape")
        self.assertIn("yanlilik", rep["sonuclar"]["tahmin_good"]["metrikler"])
        with self.assertRaises(ValueError):
            ev.evaluate(df, "t", ours="good", primary="bias", multiplier=False, B=100, B_aggregate=20)

    def test_aggregate_sum(self):
        df = _frame()
        rep = ev.evaluate(df, "t", ours="good", primary="wmape", aggregate_by="store", aggregate="sum",
                          multiplier=False, B=200, B_aggregate=50)
        g = df.groupby(["store"]).sum(numeric_only=True)   # item column present: still the store total
        self.assertEqual(rep["toplama"]["aggregate"], "sum")
        self.assertAlmostEqual(rep["sonuclar"]["tahmin_good"]["metrikler"]["wmape"]["deger"],
                               _wmape(g.truth, g.pred_good), places=10)
        with self.assertRaises(ValueError):
            ev.evaluate(df, "t", ours="good", aggregate_by="store", aggregate="median", multiplier=False)
        d2 = df.assign(sdw=df.store + "_" + df.time)
        g2 = d2.groupby("sdw").sum(numeric_only=True)
        for fr in (d2, d2.drop(columns=["item"])):        # with or without the item column: same total
            r = ev.paired_diff(fr, "good", "naive", metric="wmape", aggregate_by="sdw", aggregate="sum", B=200)
            self.assertAlmostEqual(r["value_a"], _wmape(g2.truth, g2.pred_good), places=10)
            self.assertEqual(r["n"], d2.sdw.nunique())

    def test_aggregate_sum_missing_prediction_drops_group(self):
        # a NaN prediction must not count as a zero forecast while truth is summed over all rows
        d2 = _frame().assign(sdw=lambda x: x.store + "_" + x.time)
        d2.loc[d2.index[:5], "pred_naive"] = np.nan
        bad = d2.loc[d2.pred_naive.isna(), "sdw"].unique()
        rep = ev.evaluate(d2, "t", ours="good", task="forecast", aggregate_by="sdw", aggregate="sum",
                          B=100, B_aggregate=20)
        g = d2[~d2.sdw.isin(bad)].groupby("sdw").sum(numeric_only=True)
        m = rep["sonuclar"]["tahmin_naive"]["metrikler"]
        self.assertAlmostEqual(m["wmape"]["deger"], _wmape(g.truth, g.pred_naive), places=10)
        self.assertAlmostEqual(m["yanlilik"]["deger"], g.pred_naive.sum() / g.truth.sum() - 1, places=10)
        self.assertTrue(any("eksik sayılır" in x for x in rep["sinirlar"]))

    def test_aggregate_sum_refuses_noise_ceiling(self):
        with self.assertRaises(ValueError):
            ev.evaluate(_frame(), "t", ours="good", task="forecast", aggregate_by="store", aggregate="sum",
                        noise_ceiling="variance_components", B=50, B_aggregate=10)


class TestReviewFixes(unittest.TestCase):
    def test_bias_better_means_closer_to_zero(self):
        df = _frame()
        df = df.assign(pred_under=df.pred_good * 0.7, pred_near=df.pred_good * 0.97)
        r = ev.paired_diff(df, "under", "near", metric="bias", task="forecast", B=100)
        self.assertLess(r["value_a"], r["value_b"])            # under has the lower (more negative) bias
        self.assertFalse(r["a_better"])
        self.assertTrue(ev.paired_diff(df, "near", "under", metric="bias", task="forecast", B=100)["a_better"])
        self.assertFalse(ev.paired_ci(df, "under", "near", metric="bias", task="forecast", B=100)["a_better"])
        rep = ev.evaluate(df, "t", ours="under", task="forecast", B=100, B_aggregate=20)
        yon = {m["ad"]: m["yon"] for m in rep["metrikler"]}
        self.assertEqual(yon["yanlilik"], 0)
        for c in rep["karsilastirmalar"]:
            if c["metrik"] == "yanlilik":
                va = rep["sonuclar"][c["a"]]["metrikler"]["yanlilik"]["deger"]
                vb = rep["sonuclar"][c["b"]]["metrikler"]["yanlilik"]["deger"]
                self.assertEqual(c["a_iyi"], abs(va) < abs(vb))

    def test_forecast_has_no_self_comparison_or_person_rows(self):
        rep = ev.evaluate(_frame(), "t", ours="good", task="forecast", B=100, B_aggregate=20)
        names = [a["kontrol"] for a in rep["denetim"]]
        self.assertFalse(any("madde_ici_kor" in n for n in names))
        self.assertFalse(any("kişiye özgü" in n for n in names))
        self.assertTrue(any(n.startswith("seri içi sinyal") for n in names))
        self.assertFalse(any("yeni kişilere" in x for x in rep["sinirlar"]))
        # the automatic baselines are look-ahead in forecast mode and say so
        tab = [r for r in rep["sonuclar"].values() if r["tur"] == "baseline"]
        self.assertTrue(tab and all("look-ahead" in r["etiket"] for r in tab))
        self.assertTrue(any("ileriye bakar" in x for x in rep["sinirlar"]))
        # non-forecast continuous task keeps the person rows
        rep2 = ev.evaluate(_frame(), "t", ours="good", B=100, B_aggregate=20, multiplier=False)
        self.assertTrue(any("kişiye özgü" in a["kontrol"] for a in rep2["denetim"]))


class TestBasicCI(unittest.TestCase):
    def test_basic_ci_reflects_quantiles(self):
        v = np.linspace(0, 1, 101)
        a, u = boot.ga(v)
        ba, bu = boot.ga(v, nokta=0.2, yontem="basic")
        self.assertAlmostEqual(ba, 0.4 - u)
        self.assertAlmostEqual(bu, 0.4 - a)
        with self.assertRaises(ValueError):
            boot.ga(v, yontem="bca")

    def test_evaluate_ci_basic_and_outside_warning(self):
        df = _frame()
        rep = ev.evaluate(df, "t", ours="good", task="forecast", ci="basic", B=200, B_aggregate=50)
        self.assertEqual(rep.get("ga_yontemi"), "basic")
        m = rep["sonuclar"]["tahmin_good"]["metrikler"]["wmape"]
        self.assertLessEqual(m["alt"], m["deger"])
        # force the percentile CI away from the point (as seen on the rehearsal's store x dept x week WMAPE)
        from unittest import mock
        from simengine.evidence import cekirdek
        real = cekirdek.ga

        def shifted(r, alfa=0.05, nokta=None, yontem="percentile"):
            a, u = real(r, alfa, nokta, yontem)
            return (a + 100, u + 100) if yontem == "percentile" else (a, u)
        with mock.patch.object(cekirdek, "ga", shifted), warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r2 = ev.evaluate(df, "t", ours="good", task="forecast", B=100, B_aggregate=20)
        self.assertEqual(r2["ga_uyarisi"]["metrik"], "wmape")
        self.assertTrue(any("percentile bootstrap CI excludes" in str(x.message) for x in w))
        self.assertTrue(any("yüzdelik GA" in e for e in r2["kanit_kart"]["etiketler"]))
        self.assertTrue(any(a["kontrol"].startswith("bootstrap GA") for a in r2["denetim"]))
        with self.assertRaises(ValueError):
            ev.evaluate(df, "t", ours="good", ci="bca", multiplier=False)


class TestForecastTask(unittest.TestCase):
    def test_forecast_skips_human_rows(self):
        df = _frame()
        df["retest"] = df["truth"]
        rep = ev.evaluate(df, "t", ours="good", task="forecast", unlabeled_baselines=["naive"], B=200, B_aggregate=50)
        self.assertEqual(rep["birincil"], "wmape")
        self.assertTrue(rep.get("tahmin_gorevi"))
        self.assertIsNone(rep["orneklem"])
        self.assertFalse(any(c.startswith("kontrol_karistir") for c in rep["sonuclar"]))
        self.assertFalse(any("persona karıştırma" in a["kontrol"] for a in rep["denetim"]))
        self.assertIsNone(rep["manset"]["tavan"])
        rows = {r["tur"]: r for r in rep["kanit_kart"]["satirlar"]}
        self.assertIn("uygulanamaz", rows["bedel"]["not"])
        self.assertIn("uygulanamaz", rows["tavan"]["not"])
        self.assertEqual(ev.summary(rep)["task"], "forecast")
        self.assertIn("not applicable", ev.card_markdown(rep))

    def test_forecast_awaiting_truth(self):
        df = _frame().drop(columns=["truth"])
        rep = ev.evaluate(df, "t", ours="good", task="forecast")
        rows = {r["tur"]: r for r in rep["kanit_kart"]["satirlar"]}
        self.assertEqual(rep["kanit_kart"]["rozet"], "bekliyor")
        self.assertIn("uygulanamaz", rows["bedel"]["not"])

    def test_time_column_reaches_contamination_audit(self):
        df = _frame(n_items=40, weeks=4)
        rep = ev.evaluate(df, "t", ours="good", task="forecast", B=100, B_aggregate=20)
        k = next(a for a in rep["denetim"] if a["kontrol"].startswith("kontaminasyon"))
        self.assertIn("cutoff=", k["ayrinti"])   # time present, cutoff missing: says so
        rep = ev.evaluate(df, "t", ours="good", task="forecast", cutoff="2016-03-01", B=100, B_aggregate=20)
        k = next(a for a in rep["denetim"] if a["kontrol"].startswith("kontaminasyon"))
        self.assertNotEqual(k["durum"], "atlandi")


class TestNoLLM(unittest.TestCase):
    def test_no_llm_arm_no_memorization_flag(self):
        df = _frame()
        rep = ev.evaluate(df, "t", ours="good", task="forecast", B=100, B_aggregate=20)
        self.assertIn("kanıtlanmadı: sızıntı/ezber", rep["kanit_kart"]["etiketler"])
        rep = ev.evaluate(df, "t", ours="good", task="forecast", ours_uses_llm=False, B=100, B_aggregate=20)
        et = rep["kanit_kart"]["etiketler"]
        self.assertNotIn("kanıtlanmadı: sızıntı/ezber", et)
        self.assertFalse(any("ezber" in e and e.startswith("kanıtlanmadı") for e in et))
        self.assertTrue(any("no LLM arm" in e for e in et))
        ez = next(a for a in rep["denetim"] if a["kontrol"].startswith("ezber testi"))
        self.assertIn("not applicable", ez["ayrinti"])


class TestSelectionBiasAndSeal(unittest.TestCase):
    def _sel(self, rep):
        return next(a for a in rep["denetim"] if a["kontrol"] == "seçim yanlılığı")

    def test_selection_bias_needs_prereg_evidence(self):
        df = _frame()
        rep = ev.evaluate(df, "t", ours="good", task="forecast", n_configs=7, B=100, B_aggregate=20)
        s = self._sel(rep)
        self.assertIn("not verified", s["ayrinti"])
        self.assertNotIn("önceden adlandırıldı", s["ayrinti"])
        self.assertIn("7 yapılandırma", s["ayrinti"])
        with tempfile.TemporaryDirectory() as t:
            reg = Path(t) / "reg"
            ev.seal(df[["item", "time", "pred_naive"]], "naive only", registry=reg, store_copy=False)
            rep = ev.evaluate(df, "t", ours="good", task="forecast", prereg=reg, B=100, B_aggregate=20)
            s = self._sel(rep)
            self.assertEqual(s["durum"], "uyari")
            self.assertIn("ours not found", s["ayrinti"])
            f = reg / "PREREG.md"
            f.write_text("Primary: pred_good, WMAPE.\n")
            old = time.time() - 3600
            os.utime(f, (old, old))                              # written before the seal
            rep = ev.evaluate(df, "t", ours="good", task="forecast", prereg=reg, B=100, B_aggregate=20)
            self.assertIn("önceden adlandırıldı", self._sel(rep)["ayrinti"])
            new = time.time() + 3600
            os.utime(f, (new, new))                              # written after the seal: does not count
            s = self._sel(rep := ev.evaluate(df, "t", ours="good", task="forecast", prereg=reg, B=100, B_aggregate=20))
            self.assertEqual(s["durum"], "uyari")
            self.assertIn("newer than the seal", s["ayrinti"])

    def test_sealed_column_names_are_not_preregistration(self):
        # D4: ours picked after dev results, then every candidate column sealed -> must not read as "named"
        df = _frame()
        with tempfile.TemporaryDirectory() as t:
            reg = Path(t) / "reg"
            ev.seal(df[["item", "time", "pred_good", "pred_naive"]], "test predictions", registry=reg,
                    store_copy=False)
            for ours in ("good", "naive"):
                s = self._sel(ev.evaluate(df, "t", ours=ours, task="forecast", prereg=reg, B=50, B_aggregate=10))
                self.assertNotIn("önceden adlandırıldı", s["ayrinti"])
                self.assertIn("not verified", s["ayrinti"])
            reg2 = Path(t) / "reg2"   # the seal description names ours
            ev.seal(df[["item", "time", "pred_good", "pred_naive"]], "ours: pred_good", registry=reg2,
                    store_copy=False)
            s = self._sel(ev.evaluate(df, "t", ours="good", task="forecast", prereg=reg2, B=50, B_aggregate=10))
            self.assertIn("önceden adlandırıldı", s["ayrinti"])
            reg3 = Path(t) / "reg3"   # two method names, ours not marked -> ambiguous
            ev.seal(df[["item", "time", "pred_good", "pred_naive"]], "pred_good and pred_naive", registry=reg3,
                    store_copy=False)
            s = self._sel(ev.evaluate(df, "t", ours="good", task="forecast", prereg=reg3, B=50, B_aggregate=10))
            self.assertEqual(s["durum"], "uyari")
            self.assertIn("several method names", s["ayrinti"])

    def test_seal_file_sha256(self):
        df = _frame()[["item", "time", "pred_good"]]
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "pred.csv"
            df.to_csv(f, index=False)
            reg = Path(t) / "reg"
            rec = ev.seal(str(f), "file", registry=reg, store_copy=False)
            self.assertEqual(rec["dosya_sha256"], hashlib.sha256(f.read_bytes()).hexdigest())
            self.assertNotEqual(rec["dosya_sha256"], rec["sha256"])
            self.assertIsNone(ev.seal(df, "frame", registry=reg, store_copy=False)["dosya_sha256"])
            self.assertTrue(ev.verify_chain(reg)["gecerli"])
            full = _frame().assign(truth=np.nan)
            rep = ev.evaluate(full, "t", ours="good", task="forecast", prereg=reg)
            md = ev.card_markdown(rep)
            self.assertIn("SHA-256 (file bytes)", md)
            self.assertIn(rec["dosya_sha256"], md)


if __name__ == "__main__":
    unittest.main()
