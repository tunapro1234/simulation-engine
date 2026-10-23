"""evaluate(): sampling-noise ceiling from a standard-error column (mae / rmse / wmape), the declared-vs-scored primary
metric check against the pre-registration document, split="oof" / "cv" / "time" declarations, one sign rule on the
card headline (raw ours − baseline next to the metric-direction difference) and the English reason of a
significant-but-small badge. Small synthetic frames, offline."""
import math
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import simengine.evidence as ev
from simengine.evidence import api
from simengine.evidence.kart import ANLAMLI_FARK
from simengine.research import Prereg

FAST = dict(B=150, B_aggregate=30, multiplier=False)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _se_frame(n=800, seed=0):
    """Observed truth = latent value + N(0, se_i); ours sees the latent value with some error."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n)
    se = rng.uniform(0.2, 1.0, n)
    y = theta + rng.normal(0, 1, n) * se
    return pd.DataFrame({"unit": np.arange(n), "group": np.arange(n), "truth": y, "se": se,
                         "pred_good": theta + rng.normal(0, 0.4, n), "pred_flat": np.zeros(n)})


def _small_gain(n=500, seed=3):
    """Ours beats a declared baseline by a significant but small MAE margin (< ANLAMLI_FARK)."""
    rng = np.random.default_rng(seed)
    y, e = rng.normal(size=n), rng.normal(0, 0.3, n)
    return pd.DataFrame({"unit": np.arange(n), "group": np.arange(n), "truth": y,
                         "pred_base": y + e, "pred_ours": y + 0.96 * e})


class TestSeCeiling(unittest.TestCase):
    def test_mae_ceiling_matches_analytic(self):
        df = _se_frame()
        rep = _quiet(ev.evaluate, df, ours="good", primary="mae", se_col="se", split="test", **FAST)
        c = rep["manset"]["tavan"]
        self.assertIsNotNone(c)
        self.assertIn("sampling noise (SE column)", c["etiket"])
        info = rep["se_tavani"]
        expect = float(np.mean(math.sqrt(2 / math.pi) * df["se"]))
        self.assertAlmostEqual(info["analitik"], expect, places=9)
        self.assertLess(abs(c["deger"] - expect) / expect, 0.03)     # R draws averaged in the metric
        s = ev.summary(rep)
        self.assertAlmostEqual(s["se_ceiling"]["analytic"], expect, places=9)
        self.assertEqual(s["se_ceiling"]["metric"], "mae")
        self.assertNotIn("se", [r["sutun"] for r in rep["merdiven"]])
        self.assertIn("sampling noise (SE column)", ev.card(rep, lang="en"))

    def test_rmse_and_weights(self):
        df = _se_frame(seed=1)
        df["weight"] = np.where(np.arange(len(df)) % 2, 1.0, 3.0)
        rep = _quiet(ev.evaluate, df, ours="good", primary="rmse", se_col="se", split="test", **FAST)
        w = df["weight"].to_numpy()
        self.assertAlmostEqual(rep["se_tavani"]["analitik"], float(np.sqrt((w * df["se"] ** 2).sum() / w.sum())),
                               places=9)

    def test_seeded_and_does_not_need_aggregate_by(self):
        df = _se_frame(seed=2)
        a = _quiet(ev.evaluate, df, ours="good", primary="mae", se_col="se", split="test", seed=4, **FAST)
        b = _quiet(ev.evaluate, df, ours="good", primary="mae", se_col="se", split="test", seed=4, **FAST)
        self.assertEqual(a["manset"]["tavan"]["deger"], b["manset"]["tavan"]["deger"])

    def test_bad_se_and_bad_primary_raise(self):
        df = _se_frame(n=200)
        for bad in (df.assign(se=df["se"].where(df.index != 5)), df.assign(se=-df["se"])):
            with self.assertRaises(ValueError):
                _quiet(ev.evaluate, bad, ours="good", primary="mae", se_col="se", **FAST)
        with self.assertRaises(ValueError):
            _quiet(ev.evaluate, df, ours="good", primary="pearson", se_col="se", **FAST)
        with self.assertRaises(ValueError):
            _quiet(ev.evaluate, df, ours="good", primary="mae", se_col="nope", **FAST)
        with self.assertRaises(ValueError):
            _quiet(ev.evaluate, df.assign(pred_c=0.0), ours="good", primary="mae", se_col="se", ceiling_col="pred_c", **FAST)


class TestDeclaredPrimary(unittest.TestCase):
    def _doc(self, d, **kw):
        return Prereg(title="t", question="q", **kw).write(Path(d) / "ONKAYIT.md")

    def test_research_prereg_metric_differs(self):
        df = _se_frame(n=300)
        with tempfile.TemporaryDirectory() as d:
            p = self._doc(d, metrics=["MAE (primary, lower is better)", "RMSE"])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                rep = ev.evaluate(df, ours="good", primary="rmse", prereg_doc=p, split="test", **FAST)
            self.assertTrue(any("declared" in str(x.message) and "mae" in str(x.message) for x in w))
            row = next(a for a in rep["denetim"] if a["kontrol"] == api.DECLARED_PRIMARY_CHECK)
            self.assertEqual(row["durum"], "uyari")
            self.assertTrue(any("primary metric differs from the declared one" in x
                                for x in rep["kanit_kart"]["yanilgi"]))
            self.assertIn("primary metric differs from the declared one", ev.card(rep, lang="en"))

    def test_same_metric_ok_and_hand_written_line(self):
        df = _se_frame(n=300)
        with tempfile.TemporaryDirectory() as d:
            p = self._doc(d, metrics={"primary": "mae", "secondary": "rmse"})
            rep = _quiet(ev.evaluate, df, ours="good", primary="mae", prereg_doc=p, split="test", **FAST)
            row = next(a for a in rep["denetim"] if a["kontrol"] == api.DECLARED_PRIMARY_CHECK)
            self.assertEqual(row["durum"], "tamam")
            h = Path(d) / "PLAN.md"
            h.write_text("# Plan\n\nPrimary metric: Spearman (higher is better)\nAlso report MAE.\n", encoding="utf-8")
            self.assertEqual(api.declared_primary(h.read_text(encoding="utf-8")), "spearman")
            self.assertEqual(api.declared_primary(p.read_text(encoding="utf-8")), "mae")
            self.assertIsNone(api.declared_primary("# Notes\n\nno metric here\n"))

    def test_no_document_no_row(self):
        rep = _quiet(ev.evaluate, _se_frame(n=200), ours="good", primary="mae", split="test", **FAST)
        self.assertFalse(any(a["kontrol"] == api.DECLARED_PRIMARY_CHECK for a in rep["denetim"]))


class TestSplitDeclared(unittest.TestCase):
    def test_oof_cv_time(self):
        df = _se_frame(n=300).drop(columns="se")
        for sp in ("oof", "cv", "time"):
            rep = _quiet(ev.evaluate, df, ours="good", primary="mae", split=sp, **FAST)
            row = next(a for a in rep["denetim"] if a["kontrol"] == "bölme beyanı (split=)")
            self.assertEqual(row["deger"], sp)
            self.assertIn("(declared)", row["ayrinti"])
            self.assertFalse(any(a["kontrol"] == "bölme bilgisi (eğitim/test)" and a["durum"] == "atlandi"
                                 for a in rep["denetim"]))
            self.assertFalse(any("bölme bilgisi" in e for e in rep["kanit_kart"]["etiketler"]))
        self.assertIn("cross-fitted (declared)", ev.card(_quiet(ev.evaluate, df, ours="good", primary="mae",
                                                                split="oof", **FAST), lang="en"))

    def test_other_labels_still_raise(self):
        with self.assertRaises(ValueError):
            _quiet(ev.evaluate, _se_frame(n=100), ours="good", split="train", **FAST)


class TestCardSigns(unittest.TestCase):
    def test_raw_difference_on_lower_is_better(self):
        rep = _quiet(ev.evaluate, _small_gain(), ours="ours", primary="mae", unlabeled_baselines=["base"],
                     split="test", **FAST)
        f = rep["kanit_kart"]["fark"]
        md = ev.card(rep, lang="en")
        line = next(x for x in md.splitlines() if x.startswith("Ours − strongest baseline"))
        self.assertIn(f"raw ours − baseline = {-f['fark']:+.3f}", line)
        self.assertIn("lower is better", line)

    def test_higher_is_better_says_so(self):
        rep = _quiet(ev.evaluate, _small_gain(), ours="ours", primary="pearson", unlabeled_baselines=["base"],
                     split="test", **FAST)
        line = next(x for x in ev.card(rep).splitlines() if x.startswith("Ours − strongest baseline"))
        self.assertIn("higher is better", line)

    def test_english_badge_reason_significant_but_small(self):
        rep = _quiet(ev.evaluate, _small_gain(), ours="ours", primary="mae", unlabeled_baselines=["base"],
                     split="test", **FAST)
        f = rep["kanit_kart"]["fark"]
        self.assertTrue(f["alt"] > 0 and f["fark"] < ANLAMLI_FARK)
        badge = next(x for x in ev.card(rep, lang="en").splitlines() if x.startswith("**Badge"))
        reason = badge.split("**", 2)[2].strip()
        self.assertTrue(reason)
        self.assertIn(f"{ANLAMLI_FARK:g}", reason)


if __name__ == "__main__":
    unittest.main()
