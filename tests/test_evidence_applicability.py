"""Audit applicability (D5 #8a-b, D6 #3c, D11 #5b, D12 #12 #13 #14 #16): rows that do not apply to the frame's
unit are skipped with a reason instead of being read as claims. Small synthetic frames, offline, no LLM."""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine import evidence as ev

PERSONA = "persona karıştırma (birey bilgisi kullanılıyor mu)"


def _person_frame(n_item=25, n_per=12, seed=0, shrink=1.0):
    """Person-level continuous frame: several rows per item, ours carries person signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_item):
        mu = rng.normal(0, 1)
        t = mu + rng.normal(0, 1, n_per)
        rows.append(pd.DataFrame({"item": f"q{i}", "group": [f"p{j}" for j in range(n_per)], "truth": t,
                                  "pred_other": mu + shrink * (t - mu) * .6 + rng.normal(0, .6, n_per),
                                  "pred_ours": mu + shrink * (t - mu) * .6 + rng.normal(0, .6, n_per)}))
    return pd.concat(rows, ignore_index=True)


def _segment_frame(n_item=25, seed=1):
    """Effect/segment-level frame: one row per (item, segment)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_item):
        mu = rng.normal(0, 1)
        for s in ("18-29", "30-44", "45-64", "65+"):
            t = mu + rng.normal(0, .5)
            rows.append({"item": f"q{i}", "segment": s, "truth": t, "pred_ours": t + rng.normal(0, .4),
                         "pred_other": t + rng.normal(0, .8)})
    return pd.DataFrame(rows)


def _ev(df, **kw):
    kw.setdefault("B", 200)
    kw.setdefault("B_aggregate", 60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ev.evaluate(df, task="continuous", ours="ours", **kw)


def _den(rep):
    return {d["kontrol"]: d for d in rep["denetim"]}


class LevelApplicabilityTest(unittest.TestCase):
    def _assert_not_applicable(self, rep, word):
        self.assertFalse(any(c.startswith("kontrol_karistir_") for c in rep["sonuclar"]))
        p = _den(rep)[PERSONA]
        self.assertEqual(p["durum"], "atlandi")
        self.assertTrue(p.get("uygulanamaz"))
        self.assertIn(word, p["ayrinti"])
        self.assertIsNone(rep["orneklem"])
        self.assertIn(word, rep["carpan_uygulanamaz"])
        bedel = next(r for r in rep["kanit_kart"]["satirlar"] if r["tur"] == "bedel")
        self.assertIsNone(bedel["deger"])
        self.assertIn("uygulanamaz", bedel["not"])
        self.assertIn("not applicable", bedel["not"])

    def test_level_effect_skips_shuffle_and_ppi(self):
        for level in ("effect", "study", "segment"):
            self._assert_not_applicable(_ev(_person_frame(), level=level, multiplier=True), level)

    def test_one_row_per_segment_is_detected_without_level(self):
        rep = _ev(_segment_frame(), multiplier=True)
        self._assert_not_applicable(rep, "segment")

    def test_person_level_frame_keeps_shuffle(self):
        rep = _ev(_person_frame(), multiplier=False)
        self.assertIn("kontrol_karistir_ours", rep["sonuclar"])
        self.assertNotEqual(_den(rep)[PERSONA]["durum"], "atlandi")
        self.assertNotIn("carpan_uygulanamaz", rep)


class StrongestOtherTest(unittest.TestCase):
    def test_tie_is_not_a_warning(self):
        # same generator: the other method's point estimate may be better, the paired CI includes 0
        rep = _ev(_person_frame(seed=3), primary="mae", multiplier=False)
        r = _den(rep)["en güçlü diğer yöntem"]
        k = next(k for k in rep["karsilastirmalar"] if k["b"] == "tahmin_other" and k["metrik"] == "mae")
        self.assertLess(k["alt"], 0)
        self.assertGreater(k["ust"], 0)
        self.assertEqual(r["durum"], "tamam")
        self.assertIn("berabere", r["ayrinti"])
        self.assertNotIn("bizden iyi", r["ayrinti"])

    def test_clearly_better_other_warns(self):
        df = _person_frame(seed=4)
        df["pred_other"] = df["truth"] + np.random.default_rng(9).normal(0, .1, len(df))
        r = _den(_ev(df, primary="mae", multiplier=False))["en güçlü diğer yöntem"]
        self.assertEqual(r["durum"], "uyari")
        self.assertIn("bizden iyi", r["ayrinti"])


class CollapseUnderMaeTest(unittest.TestCase):
    def test_shrinkage_is_a_note_under_mae(self):
        df = _person_frame(seed=5, shrink=0.15)
        r = _den(_ev(df, primary="mae", multiplier=False))["dağılım çöküşü (sd oranı)"]
        self.assertLess(r["deger"], 0.8)
        self.assertEqual(r["durum"], "tamam")
        self.assertTrue(r.get("bilgi_notu"))
        self.assertIn("medyan", r["ayrinti"])

    def test_shrinkage_still_warns_under_rmse(self):
        df = _person_frame(seed=5, shrink=0.15)
        r = _den(_ev(df, primary="rmse", multiplier=False))["dağılım çöküşü (sd oranı)"]
        self.assertEqual(r["durum"], "uyari")


class ConstantPredictionTest(unittest.TestCase):
    def test_constant_items_are_reported(self):
        df = _person_frame(seed=6)
        for q in ("q0", "q1"):
            df.loc[df["item"] == q, "pred_ours"] = 0.25
        rep = _ev(df, primary="mae", multiplier=False)
        r = _den(rep)["sabit tahmin: sıra metrikleri tanımsız (constant prediction)"]
        self.assertEqual(r["durum"], "uyari")
        self.assertEqual(r["deger"], 2)
        self.assertIn("2 madde", r["ayrinti"])
        s = rep["sira_tanimsiz"]["tahmin_ours"]
        self.assertEqual(s["sabit"], 2)
        self.assertGreaterEqual(s["nan"], 2)
        self.assertEqual(s["n_madde"], 25)

    def test_no_row_without_constant_items(self):
        rep = _ev(_person_frame(seed=6), primary="mae", multiplier=False)
        self.assertNotIn("sabit tahmin: sıra metrikleri tanımsız (constant prediction)", _den(rep))


class RawLlmArmTest(unittest.TestCase):
    def test_raw_arm_keeps_memorization_audit(self):
        rep = _ev(_person_frame(), raw="other", ours_uses_llm=False, multiplier=False)
        r = _den(rep)["ezber testi (LLM veriyi biliyor mu)"]
        self.assertFalse(r.get("uygulanamaz"))
        self.assertNotIn("no LLM arm", r["ayrinti"])
        k = _den(rep)["kontaminasyon (kesim öncesi/sonrası)"]
        self.assertNotIn("no LLM arm", k["ayrinti"])

    def test_no_llm_at_all_still_not_applicable(self):
        r = _den(_ev(_person_frame(), ours_uses_llm=False, multiplier=False))["ezber testi (LLM veriyi biliyor mu)"]
        self.assertTrue(r.get("uygulanamaz"))


if __name__ == "__main__":
    unittest.main()
