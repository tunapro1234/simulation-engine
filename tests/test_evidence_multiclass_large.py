"""Many-class multiclass (D9 #3): hard-label kappa / balanced accuracy / macro-F1 from bincounts over the labels
present (no K x K confusion matrix), and label rungs instead of n x K probability vectors above
merdiven.ETIKET_ESIK classes. Small synthetic frames, offline."""
import types
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

import simengine.evidence as ev
from simengine.evidence import merdiven, metrik


def _sk(fn, y, t, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(y, t, **kw)


def _frame(n, K, seed=0, item=False):
    rng = np.random.default_rng(seed)
    cls = np.array([f"c{i}" for i in range(K)])
    t = rng.zipf(1.5, n) % K
    ours = np.where(rng.random(n) < .4, t, rng.integers(0, K, n))
    d = pd.DataFrame({"truth": cls[t], "pred_ours": cls[ours], "group": [f"g{i % max(n // 3, 1)}" for i in range(n)]})
    if item:
        d["item"] = [f"q{i % 5}" for i in range(n)]
    return d


class HardLabelMetricsTest(unittest.TestCase):
    def test_match_sklearn(self):
        rng = np.random.default_rng(0)
        for trial in range(60):
            n, K = int(rng.integers(2, 300)), int(rng.integers(1, 40))
            y = rng.integers(-1, K, n)
            t = np.where(rng.random(n) < rng.random(), y, rng.integers(-1, K + 3, n))
            self.assertAlmostEqual(metrik._dengeli_cs(y, t), _sk(balanced_accuracy_score, y, t), places=12)
            self.assertAlmostEqual(metrik._f1_cs(y, t), _sk(f1_score, y, t, average="macro", zero_division=0),
                                   places=12)
            k, ks = metrik._kappa_cs(y, t), _sk(cohen_kappa_score, y, t)
            if np.isnan(ks):
                self.assertTrue(np.isnan(k))
            else:
                self.assertAlmostEqual(k, ks, places=12)

    def test_probability_input_uses_argmax(self):
        rng = np.random.default_rng(1)
        p = rng.dirichlet(np.ones(6), 80)
        y = rng.integers(0, 6, 80)
        self.assertAlmostEqual(metrik._kappa_cs(y, p), cohen_kappa_score(y, p.argmax(1)), places=12)
        self.assertAlmostEqual(metrik._f1_cs(y, p), f1_score(y, p.argmax(1), average="macro", zero_division=0),
                               places=12)

    def test_degenerate_cases(self):
        y = np.array([2, 2, 2])
        self.assertTrue(np.isnan(metrik._kappa_cs(y, y)))       # p_e = 1: undefined (sklearn: nan)
        self.assertEqual(metrik._dengeli_cs(y, y), 1.0)
        self.assertEqual(metrik._f1_cs(y, np.array([1, 1, 1])), 0.0)

    def test_no_confusion_matrix_for_many_classes(self):
        rng = np.random.default_rng(2)
        y, t = rng.integers(0, 5000, 20000), rng.integers(0, 5000, 20000)
        boom = mock.Mock(side_effect=AssertionError("K x K confusion matrix"))
        with mock.patch("sklearn.metrics._classification.confusion_matrix", boom), \
                mock.patch("sklearn.metrics._classification.multilabel_confusion_matrix", boom):
            for fn in (metrik._kappa_cs, metrik._dengeli_cs, metrik._f1_cs):
                self.assertTrue(np.isfinite(fn(y, t)))


class LabelRungsTest(unittest.TestCase):
    def _veri(self, df, K, egitim=None):
        return types.SimpleNamespace(df=df.rename(columns={"truth": "gercek", "group": "grup", "item": "madde"}),
                                     gorev="cok_sinif", siniflar=[f"c{i}" for i in range(K)], segmentler=[],
                                     egitim=egitim, notlar=[], yontemler=["ours"])

    def test_many_classes_give_label_rungs(self):
        K = merdiven.ETIKET_ESIK + 50
        df = _frame(3000, K, item=True)
        v = self._veri(df.rename(columns={"pred_ours": "tahmin_ours"}), K)
        out = merdiven.baselineler(v)
        for ad in ("rastgele", "marjinal"):
            deg = out[ad][1]
            self.assertIsInstance(deg, np.ndarray)
            self.assertEqual(deg.shape, (len(df),))
            self.assertTrue(set(deg) <= set(v.siniflar))
        # marginal = per-item mode of the other folds (the zipf head class here)
        self.assertGreater(np.mean(out["marjinal"][1] == v.df["gercek"].to_numpy()), 0.3)
        self.assertLess(np.mean(out["rastgele"][1] == v.df["gercek"].to_numpy()), 0.02)

    def test_label_mode_from_training_folds_only(self):
        # every item's rows split across folds: the mode never comes from the row's own fold
        K = merdiven.ETIKET_ESIK + 1
        df = pd.DataFrame({"truth": ["c1"] * 4 + ["c2"], "pred_ours": ["c1"] * 5, "group": list("abcde")})
        v = self._veri(df.rename(columns={"pred_ours": "tahmin_ours"}), K)
        k = merdiven.katlar(v.df)
        _, deg = merdiven.baselineler(v)["marjinal"]
        for i in range(len(df)):
            tr = v.df["gercek"][k != k[i]]
            self.assertEqual(deg[i], tr.value_counts().sort_index().idxmax())

    def test_small_k_keeps_probability_vectors(self):
        df = _frame(200, 5)
        v = self._veri(df.rename(columns={"pred_ours": "tahmin_ours"}), 5)
        out = merdiven.baselineler(v)
        self.assertEqual(len(out["rastgele"][1][0]), 5)
        self.assertEqual(len(out["marjinal"][1][0]), 5)

    def test_evaluate_many_classes(self):
        K = merdiven.ETIKET_ESIK * 3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = ev.evaluate(_frame(2000, K), task="multiclass", ours="ours", B=40, B_aggregate=20,
                              multiplier=False)
        r = rep["sonuclar"]
        self.assertLess(r["taban_rastgele"]["metrikler"]["dogruluk"]["deger"], 0.02)
        self.assertGreater(r["tahmin_ours"]["metrikler"]["kappa"]["deger"], 0.2)
        self.assertIn("mode", r["taban_marjinal"]["etiket"])


if __name__ == "__main__":
    unittest.main()
