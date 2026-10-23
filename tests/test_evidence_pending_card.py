"""Awaiting-truth (pending) report (D5 #8e, D7 #13, D8 #9, D9 #9, D11 #7, D12 #16a #18 #19): declared baselines
are kept, the header seal is ours', the frame is checked against the sealed files, and the prediction summary is
per item with a constant-prediction warning and label counts for integer-coded classes. Offline, temp registry."""
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine import evidence as ev


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    n = 30
    return pd.DataFrame({"item": np.repeat(["a", "b", "c"], n // 3), "group": [f"p{i}" for i in range(n)],
                         "pred_persist": rng.normal(0, 1, n), "pred_lgbm": rng.normal(0, 1, n),
                         "pred_ours": rng.normal(0, 1, n)})


def _ev(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ev.evaluate(df, ours="ours", **kw)


def _den(rep):
    return {d["kontrol"]: d for d in rep["denetim"]}


class DeclaredBaselinesTest(unittest.TestCase):
    def test_declared_baselines_reach_the_pending_report(self):
        rep = _ev(_frame(), task="continuous", unlabeled_baselines=["persist"], baselines_oof=["lgbm"],
                  labels={"persist": "Persistence", "lgbm": "LightGBM OOF"})
        self.assertTrue(rep["gercek_bekleniyor"])
        self.assertEqual(rep["bildirilen_tabanlar"], ["Persistence", "LightGBM OOF"])
        self.assertEqual(rep["tabanlar"]["bildirilen"], ["tahmin_persist", "tahmin_lgbm"])
        self.assertEqual(rep["tabanlar"]["oof"], ["tahmin_lgbm"])

    def test_none_declared(self):
        rep = _ev(_frame(), task="continuous")
        self.assertEqual(rep["bildirilen_tabanlar"], [])


class OursSealTest(unittest.TestCase):
    def _chain(self, d, df, ours_desc):
        z = [ev.seal(df, ours_desc, registry=d, store_copy=False),
             ev.seal(df[["item", "group", "pred_lgbm"]], "T3 lgbm baseline", registry=d, store_copy=False)]
        return z

    def test_header_seal_is_ours_not_the_last(self):
        df = _frame()
        with tempfile.TemporaryDirectory() as d:
            z = self._chain(d, df, "ours: ours, final submission")
            rep = _ev(df, task="continuous", prereg=d)
        self.assertEqual(rep["bizim_muhur"]["sha256"], z[0]["sha256"])
        self.assertEqual(rep["kanit_kart"]["muhur"]["sha256"], z[0]["sha256"])
        r = _den(rep)["mühür"]
        self.assertIn(z[0]["sha256"][:12], r["ayrinti"])
        self.assertNotIn(z[1]["sha256"][:12], r["ayrinti"])

    def test_baseline_seal_naming_ours_is_not_ours(self):
        # a later baseline seal whose description mentions ours must not take ours' place in the header
        df = _frame()
        with tempfile.TemporaryDirectory() as d:
            z0 = ev.seal(df, "final submission", registry=d, store_copy=False)
            ev.seal(df[["item", "group", "pred_lgbm"]], "lgbm baseline to compare with ours", registry=d,
                    store_copy=False)
            rep = _ev(df.sample(frac=1, random_state=1), task="continuous", prereg=d)
        self.assertEqual(rep["bizim_muhur"]["sha256"], z0["sha256"])
        self.assertEqual(rep["bizim_muhur"]["eslesme"], "cerceve")
        self.assertEqual(_den(rep)["çerçeve ↔ mühürlü dosya"]["durum"], "tamam")

    def test_fallback_to_last_with_note(self):
        df = _frame()
        with tempfile.TemporaryDirectory() as d:
            ev.seal(df[["item", "group", "pred_persist"]], "first", registry=d, store_copy=False)
            z = ev.seal(df[["item", "group", "pred_lgbm"]], "second", registry=d, store_copy=False)
            rep = _ev(df, task="continuous", prereg=d)
        self.assertEqual(rep["bizim_muhur"]["sha256"], z["sha256"])
        self.assertIn("ours not matched", rep["bizim_muhur"]["not"])

    def test_frame_is_checked_against_sealed_files(self):
        df = _frame()
        with tempfile.TemporaryDirectory() as d:
            self._chain(d, df, "ours: ours")
            ok = _den(_ev(df, task="continuous", prereg=d))["çerçeve ↔ mühürlü dosya"]
            df2 = df.copy()
            df2.loc[0, "pred_ours"] += 1
            bad = _den(_ev(df2, task="continuous", prereg=d))["çerçeve ↔ mühürlü dosya"]
        self.assertEqual(ok["durum"], "tamam")
        self.assertEqual(bad["durum"], "uyari")
        self.assertIn("does not match", bad["ayrinti"])


class PredictionSummaryTest(unittest.TestCase):
    def test_per_item_summary_and_constant_warning(self):
        df = _frame()
        df.loc[df["item"] == "b", "pred_ours"] = 0.5
        rep = _ev(df, task="continuous")
        oz = rep["tahmin_ozeti"]["tahmin_ours"]
        self.assertEqual(set(oz["maddeler"]), {"a", "b", "c"})
        self.assertEqual(oz["maddeler"]["b"]["sd"], 0.0)
        self.assertEqual(oz["n_sabit_madde"], 1)
        r = _den(rep)["sabit tahmin (sd 0)"]
        self.assertEqual(r["durum"], "uyari")
        self.assertIn("1/3", r["ayrinti"])

    def test_integer_codes_get_label_counts(self):
        df = _frame()
        rng = np.random.default_rng(2)
        for c in ("pred_persist", "pred_lgbm", "pred_ours"):
            df[c] = rng.integers(1, 6, len(df))
        rep = _ev(df)
        oz = rep["tahmin_ozeti"]["tahmin_ours"]
        self.assertEqual(oz["tur"], "etiket")
        self.assertNotIn("ortalama", oz)
        self.assertNotIn("sd", oz)
        self.assertEqual(oz["n_farkli"], 5)
        self.assertTrue(oz["en_sik"])

    def test_continuous_integer_values_keep_mean(self):
        df = _frame()
        df["pred_ours"] = np.arange(len(df)) * 7
        oz = _ev(df, task="continuous")["tahmin_ozeti"]["tahmin_ours"]
        self.assertEqual(oz["tur"], "sayi")
        self.assertIn("ortalama", oz)


if __name__ == "__main__":
    unittest.main()
