"""English view of an evidence report: headline_en / card(lang="en") use the user's column names and English labels,
summary() gains badge_reason_en, English audits and a gap_closed {value, lo, hi} that is n/a when the ceiling is about
the baseline. Small synthetic frames, offline."""
import re
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine.evidence as ev
from simengine.evidence import api

FAST = dict(B=120, B_aggregate=30, multiplier=False)
INTERNAL = ("tahmin_", "taban_", "kanıtlanmadı", "yontem")
TR_LETTERS = re.compile(r"[çğıöşüÇĞİÖŞÜ]")


def _binary(n=400, seed=0, retest=None):
    """Binary EvalFrame with a train/test split; retest: None, "good" (a second noisy measurement) or "noise"."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    d = pd.DataFrame({"unit": np.arange(n), "group": [f"p{i}" for i in range(n)],
                      "split": np.where(np.arange(n) < 100, "train", "test"), "segment": rng.choice(["A", "B"], n),
                      "truth": (x + rng.normal(0, 0.8, n) > 0).astype(int),
                      "pred_llm": 1 / (1 + np.exp(-0.4 * x)), "pred_panel_bayes": 1 / (1 + np.exp(-2 * x))})
    if retest == "good":
        d["retest"] = (x + rng.normal(0, 0.8, n) > 0).astype(int)
    elif retest == "noise":
        d["retest"] = rng.integers(0, 2, n)
    return d


def _evaluate(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ev.evaluate(df, title="english view", ours="panel_bayes", raw="llm", **FAST, **kw)


class TestEnglishCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = _evaluate(_binary(retest="good"))
        cls.s = ev.summary(cls.rep)
        cls.md = ev.card(cls.rep, lang="en")

    def test_no_internal_names(self):
        for text in (self.md, self.s["headline_en"], ev.card_markdown(self.rep, lang="en")):
            for bad in INTERNAL:
                self.assertNotIn(bad, text)
        self.assertIn("pred_panel_bayes", self.s["headline_en"])
        self.assertIn("(`pred_llm`)", self.md)
        self.assertIn("`baseline marginal`", self.md)
        self.assertIn("`control shuffled pred_panel_bayes`", self.md)

    def test_english_labels_kinds_flags_limits(self):
        self.assertIn("| method |", self.md)
        self.assertIn("| control |", self.md)
        self.assertIn("Overall rate (`baseline marginal`)", self.md)
        self.assertNotIn("En güçlü basit taban", self.md)
        self.assertIn("Flags: not proven: leakage/memorization", self.md)
        self.assertIn("| group overlap (train ∩ test) | ok | 0 shared group(s) |", self.md)
        limits = self.md.split("## Limits", 1)[1]
        self.assertIsNone(TR_LETTERS.search(limits), limits)
        self.assertIn("Items are taken as fixed", limits)

    def test_mixed_card_keeps_research_wording(self):
        md = ev.card(self.rep)
        self.assertIn("En güçlü basit taban", md)
        self.assertIn("(`tahmin_llm`) | yontem |", md)
        self.assertIn("grup çakışması (eğitim ∩ test)", md)

    def test_badge_reason_en(self):
        self.assertTrue(self.s["badge_reason"])                    # Turkish reason kept
        r = self.s["badge_reason_en"]
        self.assertTrue(r)
        self.assertIsNone(TR_LETTERS.search(r), r)
        self.assertNotRegex(r, r"\d,\d")                           # dot decimals
        self.assertIn(r.split(" (Badge")[0], self.md)

    def test_audits_english(self):
        aud = self.s["audits"]
        self.assertTrue(aud)
        for a in aud:
            self.assertTrue({"name", "status", "detail", "check", "detail_tr"} <= set(a))
            self.assertIn(a["status"], ("ok", "warning", "skipped", "error"))
        by = {a["name"]: a for a in aud}
        self.assertEqual(by["group overlap (train ∩ test)"]["check"], "grup çakışması (eğitim ∩ test)")
        self.assertEqual(by["selection bias"]["status"], "skipped")
        self.assertIsNone(TR_LETTERS.search(by["class imbalance"]["detail"]))

    def test_gap_closed_keys(self):
        g = self.s["gap_closed"]
        self.assertTrue({"value", "lo", "hi"} <= set(g))
        self.assertIsNotNone(g["value"])
        self.assertLessEqual(g["lo"], g["hi"])
        raw = self.rep["manset"]["bosluk_kapanan"]
        self.assertEqual((g["deger"], g["alt"], g["ust"]), (raw["deger"], raw["alt"], raw["ust"]))   # deprecated aliases


class TestGapClosedNotApplicable(unittest.TestCase):
    def test_ceiling_about_baseline(self):
        rep = _evaluate(_binary(seed=3, retest="noise"))
        m = rep["manset"]
        self.assertIsNotNone(m["bosluk_kapanan"])                  # the research number exists ...
        g = ev.summary(rep)["gap_closed"]
        self.assertIsNone(g["value"])                              # ... but divides by noise
        self.assertEqual(g["note"], api.GAP_NA_NOTE)
        for lang in ("en", "mixed"):
            row = next(x for x in ev.card(rep, lang=lang).splitlines() if x.startswith("| Gap closed"))
            self.assertIn("n/a, ceiling ≈ baseline", row)

    def test_no_ceiling(self):
        self.assertIsNone(ev.summary(_evaluate(_binary()))["gap_closed"])


class TestTranslationHelpers(unittest.TestCase):
    def test_column_en(self):
        self.assertEqual(api.column_en("tahmin_x"), "pred_x")
        self.assertEqual(api.column_en("taban_rastgele"), "baseline random")
        self.assertEqual(api.column_en("taban_kalicilik"), "baseline kalicilik")
        self.assertEqual(api.column_en("kontrol_karistir_x"), "control shuffled pred_x")
        self.assertEqual(api.column_en("kalib_x+izotonik"), "calibrated pred_x + isotonic")
        self.assertEqual(api.column_en("tavan"), "ceiling")
        self.assertEqual(api.column_en("segment"), "segment")

    def test_untranslated_falls_back(self):
        self.assertEqual(api._en_text("tamamen yeni bir not, 0,25"), "tamamen yeni bir not, 0.25")
        self.assertEqual(api._en_text(None), "")
        self.assertEqual(api._en_text("yeni denetim ölçüsü (an English gloss here)"), "an English gloss here")
        self.assertEqual(api._check_en("yeni bir denetim"), "yeni bir denetim")
        self.assertEqual(api._label_en("tahmin_a · karıştırılmış"), "pred_a · shuffled")


if __name__ == "__main__":
    unittest.main()
