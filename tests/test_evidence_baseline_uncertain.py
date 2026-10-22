"""D5 #10: 'strongest baseline uncertain' audit. When the best baselines are within paired bootstrap noise of each
other, the card says so, lists the tied set and reports ours − baseline in the worst case over it. Synthetic, offline."""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine import evidence as ev
from simengine.evidence import denetim, kart

KONTROL = "en güçlü taban belirsiz mi"


def _frame(n=300, sds=(1.0, 1.0, 1.0, 1.0), ours_sd=0.6, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 2, n)
    d = {"truth": y, "group": [f"g{i}" for i in range(n)]}
    for k, sd in enumerate(sds, 1):
        d[f"pred_B{k}"] = y + rng.normal(0, sd, n)
    d["pred_ours"] = y + rng.normal(0, ours_sd, n)
    return pd.DataFrame(d)


def _audit(rep):
    rows = [a for a in rep["denetim"] if a["kontrol"].startswith(KONTROL)]
    return rows[0] if rows else None


def _eval(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ev.evaluate(df, "t", ours="ours", task="continuous", primary="mae", B=300, multiplier=False,
                           unlabeled_baselines=[c[5:] for c in df if c.startswith("pred_B")])


class TestFunction(unittest.TestCase):
    def test_tied_replicates_warn_and_report_worst_case(self):
        rng = np.random.default_rng(1)
        common = rng.normal(1.0, 0.05, 400)   # shared resample noise: paired differences are small
        tekrar = {f"b{k}": {"mae": common + rng.normal(0, 0.01, 400) + 0.002 * k} for k in range(4)}
        tekrar["ours"] = {"mae": common - 0.03 + rng.normal(0, 0.01, 400)}
        nokta = {c: {"mae": float(np.mean(r["mae"]))} for c, r in tekrar.items()}
        lab = {f"b{k}": f"B{k}" for k in range(4)}
        r = denetim.taban_belirsizligi(tekrar, list(lab), "mae", -1, lab, bizim="ours", nokta=nokta)
        self.assertEqual(r["durum"], "uyari")
        self.assertEqual(r["esitler"][0], "B0")            # lowest MAE first
        self.assertGreaterEqual(len(r["esitler"]), 2)
        self.assertIn("strongest baseline uncertain", r["ayrinti"])
        self.assertIn("4 taban", r["ayrinti"])
        w = r["en_kotu"]
        # worst case = min over the tied set of (direction-aligned) ours − baseline
        expect = min(-(nokta["ours"]["mae"] - nokta[f"b{k}"]["mae"]) for k in range(4) if f"B{k}" in r["esitler"])
        self.assertAlmostEqual(w["fark"], expect, places=12)
        self.assertLessEqual(w["alt"], w["fark"])

    def test_separated_and_degenerate(self):
        rng = np.random.default_rng(2)
        common = rng.normal(1.0, 0.05, 400)
        tekrar = {"b0": {"mae": common - 0.2}, "b1": {"mae": common + rng.normal(0, 0.01, 400)}}
        r = denetim.taban_belirsizligi(tekrar, ["b0", "b1"], "mae", -1)
        self.assertEqual(r["durum"], "tamam")
        self.assertIsNone(denetim.taban_belirsizligi(tekrar, ["b0"], "mae", -1))
        self.assertIsNone(denetim.taban_belirsizligi(tekrar, ["b0", "missing"], "mae", -1))


class TestCard(unittest.TestCase):
    def test_four_baselines_within_noise_warn(self):
        rep = _eval(_frame())
        a = _audit(rep)
        self.assertIsNotNone(a)
        self.assertEqual(a["durum"], "uyari")
        self.assertGreaterEqual(len(a["esitler"]), 2)
        self.assertTrue(all(e.startswith("B") for e in a["esitler"]))
        self.assertIn("en_kotu", a)
        self.assertTrue(np.isfinite(a["en_kotu"]["alt"]))
        # the worst case is no better than ours against the chosen strongest baseline
        best = rep["manset"]["en_iyi_baseline"]
        self.assertLessEqual(a["en_kotu"]["fark"], -(rep["manset"]["bizim"]["deger"] - best["deger"]) + 1e-12)

    def test_badge_uses_worst_tied_baseline(self):
        rep = _eval(_frame(sds=(1.0, 1.0, 1.0, 1.0), ours_sd=0.6))
        k = kart.kart(rep)
        self.assertEqual(k["rozet"], "yuksek")
        self.assertIn("belirsiz", k["satirlar"][0]["not"])
        self.assertTrue(any("en güçlü taban belirsiz" in e for e in k["etiketler"]))
        self.assertIn("worst tied baseline", k["rozet_gerekce"])
        # the single strongest baseline clears 0, the worst tied one does not: no high-confidence badge
        a = _audit(rep)
        a["en_kotu"]["alt"] = -0.01
        self.assertGreater(k["fark"]["alt"], 0)
        k2 = kart.kart(rep)
        self.assertEqual(k2["rozet"], "kararsiz")
        self.assertEqual(k2["fark"], k["fark"])            # fark stays ours − the single strongest baseline
        self.assertIn("-0.010", k2["rozet_gerekce"])

    def test_worst_tied_bound_never_looser_than_single(self):
        # ci="basic" reflects the per-replicate min around the point: its lower bound can sit above the single
        # strongest baseline's; the badge must not get easier than against that one baseline
        rep = _eval(_frame(sds=(1.0, 1.0, 1.0, 1.0), ours_sd=0.6))
        single = kart.kart(rep)["fark"]["alt"]
        _audit(rep)["en_kotu"]["alt"] = single + 0.5
        self.assertIn(f"{single:+.3f}", kart.kart(rep)["rozet_gerekce"])

    def test_clearly_best_baseline_no_warning(self):
        rep = _eval(_frame(sds=(0.3, 2.0, 2.0, 2.0), ours_sd=0.2))
        a = _audit(rep)
        self.assertIsNotNone(a)
        self.assertEqual(a["durum"], "tamam")
        self.assertNotIn("worst tied baseline", rep["kanit_kart"]["rozet_gerekce"])


if __name__ == "__main__":
    unittest.main()
