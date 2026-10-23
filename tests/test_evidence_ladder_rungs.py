"""Honest automatic ladder rungs (merdiven.baselineler): G4 #13 (binary marginal anti-correlated with the held-out
fold, AUC ~0.38), D9 #4a (constant truth makes random/marginal copy it and win the badge), D11 §8 (no split column:
the labelled rungs must say they are cross-fitted, one-row items never see their own truth), D11 #5b / D5 #8b
(segment = unit: the demographic rung equals the marginal). Small synthetic frames, offline."""
import types
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine.evidence as ev
from simengine.evidence import merdiven

FAST = dict(B=60, B_aggregate=20, multiplier=False)


def _evaluate(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ev.evaluate(df, **FAST, **kw)


def _binary(n=60, seed=0, split=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    d = pd.DataFrame({"group": [f"p{i}" for i in range(n)], "truth": (x + rng.normal(0, 1, n) > 0).astype(int),
                      "pred_ours": 1 / (1 + np.exp(-x))})
    if split:
        d["split"] = np.where(np.arange(n) < n // 3, "train", "test")
    return d


def _continuous_items(n_items=12, per=6, seed=0, segment="repeat"):
    rng = np.random.default_rng(seed)
    item = np.repeat([f"q{i}" for i in range(n_items)], per)
    person = np.tile([f"p{j}" for j in range(per)], n_items)
    t = rng.normal(0, 1, len(item)) + np.repeat(rng.normal(0, 2, n_items), per)
    d = pd.DataFrame({"item": item, "group": person, "truth": t, "pred_ours": t + rng.normal(0, .7, len(t))})
    if segment == "repeat":
        d["segment"] = np.tile(["A", "B"], len(d) // 2)
    elif segment == "unit":
        d["segment"] = person     # one row per (item, segment): segment identifies the row
    return d


class BinaryMarginalTest(unittest.TestCase):
    def test_marginal_is_a_constant_without_items(self):
        # G4 #13: per-fold training means anti-correlate with the held-out fold's positives (was AUC 0.383 here)
        rep = _evaluate(_binary(), task="binary", ours="ours")
        r = rep["sonuclar"]["taban_marjinal"]["metrikler"]
        self.assertAlmostEqual(r["auc"]["deger"], 0.5, places=9)
        for seed in range(1, 6):
            rep = _evaluate(_binary(seed=seed), task="binary", ours="ours")
            self.assertAlmostEqual(rep["sonuclar"]["taban_marjinal"]["metrikler"]["auc"]["deger"], 0.5, places=9)

    def test_merdiven_values_constant_and_out_of_sample_rate(self):
        df = _binary(n=50)
        v = types.SimpleNamespace(df=df.rename(columns={"truth": "gercek", "group": "grup"}), gorev="ikili",
                                  segmentler=[], egitim=None, notlar=[])
        _, deg = merdiven.baselineler(v)["marjinal"]
        self.assertEqual(np.ptp(deg), 0.0)
        self.assertAlmostEqual(float(deg[0]), float(df["truth"].mean()), delta=0.05)

    def test_train_split_marginal_is_the_train_rate(self):
        df = _binary(n=90, split=True)
        rep = _evaluate(df, task="binary", ours="ours")
        r = rep["sonuclar"]["taban_marjinal"]
        self.assertAlmostEqual(r["metrikler"]["auc"]["deger"], 0.5, places=9)
        self.assertNotIn("cross-fitted", r["etiket"])


class ConstantTruthTest(unittest.TestCase):
    def test_continuous_constant_truth_drops_auto_rungs(self):
        # D9 #4a: truth 1.0 everywhere (hit-rate detour): random/median copied it, MAE 0, "below baseline" badge
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"item": np.repeat([f"q{i}" for i in range(10)], 5), "truth": 1.0})
        df["pred_ours"] = 1 + rng.normal(0, .1, len(df))
        rep = _evaluate(df, task="continuous", primary="mae", ours="ours")
        for c in ("taban_rastgele", "taban_marjinal", "taban_demografi"):
            self.assertNotIn(c, rep["sonuclar"])
        self.assertIsNone(rep["manset"]["en_iyi_baseline"])
        notlar = [d for d in rep["denetim"] if d["kontrol"] == "girdi notu" and "constant truth" in d["ayrinti"]]
        self.assertEqual(len(notlar), 1)
        self.assertEqual(notlar[0]["durum"], "uyari")

    def test_binary_constant_truth_keeps_only_the_truth_free_rung(self):
        df = _binary(n=40)
        df["truth"] = 1
        rep = _evaluate(df, task="binary", primary="brier", ours="ours")
        self.assertNotIn("taban_marjinal", rep["sonuclar"])
        self.assertIn("taban_rastgele", rep["sonuclar"])     # 0.5 never reads the truth

    def test_varying_truth_keeps_rungs(self):
        rep = _evaluate(_continuous_items(segment=None), task="continuous", ours="ours")
        self.assertIn("taban_rastgele", rep["sonuclar"])
        self.assertIn("taban_marjinal", rep["sonuclar"])


class CrossFitLabelTest(unittest.TestCase):
    def test_no_split_labels_say_cross_fitted_by_group(self):
        rep = _evaluate(_continuous_items(), task="continuous", ours="ours")
        for c in ("taban_marjinal", "taban_demografi"):
            self.assertIn("cross-fitted, 5-fold by group", rep["sonuclar"][c]["etiket"])
        self.assertNotIn("cross-fitted", rep["sonuclar"]["taban_rastgele"]["etiket"])

    def test_no_group_column_says_by_row(self):
        rep = _evaluate(_continuous_items(segment=None).drop(columns="group"), task="continuous", ours="ours")
        self.assertIn("cross-fitted, 5-fold by row", rep["sonuclar"]["taban_marjinal"]["etiket"])

    def test_split_column_no_cross_fit_label(self):
        df = _continuous_items()
        df["split"] = np.where(df["group"].isin(["p0", "p1"]), "train", "test")
        rep = _evaluate(df, task="continuous", ours="ours")
        self.assertNotIn("cross-fitted", rep["sonuclar"]["taban_marjinal"]["etiket"])

    def test_one_row_items_never_see_their_own_truth(self):
        n = 26
        df = pd.DataFrame({"madde": [f"s{i}" for i in range(n)], "gercek": 1000.0 * np.arange(1, n + 1),
                           "segment": [f"seg{i % 3}" for i in range(n)]})
        for ozet in ("mean", "median"):
            v = types.SimpleNamespace(df=df, gorev="surekli", segmentler=["segment"], egitim=None, notlar=[])
            out = merdiven.baselineler(v, ozet)
            for ad, (_, deg) in out.items():
                self.assertFalse(np.any(np.isclose(deg, df["gercek"].to_numpy())), (ozet, ad))
        b = df.assign(gercek=(np.arange(n) % 2).astype(float))
        v = types.SimpleNamespace(df=b, gorev="ikili", segmentler=[], egitim=None, notlar=[])
        _, deg = merdiven.baselineler(v)["marjinal"]
        self.assertTrue(np.all((deg > 0) & (deg < 1)))


class DemographicUnitTest(unittest.TestCase):
    def test_segment_identifies_the_row_skips_demographic(self):
        rep = _evaluate(_continuous_items(segment="unit"), task="continuous", ours="ours")
        self.assertNotIn("taban_demografi", rep["sonuclar"])
        self.assertIn("taban_marjinal", rep["sonuclar"])

    def test_one_row_per_item_skips_demographic(self):
        rng = np.random.default_rng(1)
        t = rng.normal(0, 1, 26)
        df = pd.DataFrame({"item": [f"s{i}" for i in range(26)], "segment": [f"seg{i % 3}" for i in range(26)],
                           "truth": t, "pred_ours": t + rng.normal(0, .5, 26)})
        rep = _evaluate(df, task="continuous", primary="mae", ours="ours")
        self.assertNotIn("taban_demografi", rep["sonuclar"])

    def test_repeated_segments_keep_demographic(self):
        rep = _evaluate(_continuous_items(segment="repeat"), task="continuous", ours="ours")
        self.assertIn("taban_demografi", rep["sonuclar"])


if __name__ == "__main__":
    unittest.main()
