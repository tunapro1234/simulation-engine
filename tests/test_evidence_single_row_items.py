"""G9 #3: continuous task with one row per item. The within-item random baseline used to copy the truth
(MAE 0, strongest baseline) and the ceiling ratio divided by zero. Small synthetic data, offline."""
import types
import unittest

import numpy as np
import pandas as pd

from simengine import evidence as ev
from simengine.evidence import merdiven


def _one_row_frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.normal(0, 1, n)
    return pd.DataFrame({"item": [f"e{i}" for i in range(n)], "truth": t,
                         "pred_ours": t + rng.normal(0, .5, n), "pred_raw": t + rng.normal(0, 1, n),
                         "retest": t + rng.normal(0, .3, n)})


class SingleRowItemsTest(unittest.TestCase):
    def test_evaluate_runs_and_random_baseline_is_not_the_truth(self):
        rep = ev.evaluate(_one_row_frame(), task="continuous", primary="mae", ours="ours", raw="raw",
                          ceiling_col="retest")
        r = rep["sonuclar"]["taban_rastgele"]
        self.assertGreater(r["metrikler"]["mae"]["deger"], 0.1)
        self.assertIn("tek satır", r["etiket"])
        self.assertNotEqual(rep["manset"]["en_iyi_baseline"]["sutun"], "taban_rastgele")
        self.assertIsNotNone(rep["manset"]["taban_tavana_oran"])

    def test_within_item_audits_not_applicable(self):
        # one row per item: shuffle control, within-item metrics and spread ratio would be 0/0 or copy ours
        rep = ev.evaluate(_one_row_frame(), task="continuous", primary="mae", ours="ours", raw="raw")
        self.assertNotIn("kontrol_karistir_ours", rep["sonuclar"])
        den = {d["kontrol"]: d for d in rep["denetim"]}
        self.assertEqual(den["persona karıştırma (birey bilgisi kullanılıyor mu)"]["durum"], "atlandi")
        self.assertIn("tek satır", den["persona karıştırma (birey bilgisi kullanılıyor mu)"]["ayrinti"])
        self.assertEqual(den["dağılım çöküşü (sd oranı)"]["durum"], "atlandi")
        self.assertFalse(any("demeaned_r2" in k or "madde_ici_kor" in k for k in den))
        for r in rep["sonuclar"].values():
            self.assertNotIn("demeaned_r2", r["metrikler"])
        self.assertEqual(rep["sonuclar"]["taban_marjinal"]["etiket"], "Genel medyan (madde başına tek satır) (cross-fitted, 5-fold by row)")  # mae → median; no split column

    def test_two_row_items_keep_within_item_audits(self):
        df = _one_row_frame()
        df = pd.concat([df, df.assign(truth=df.truth + .2, pred_ours=df.pred_ours + .1)], ignore_index=True)
        rep = ev.evaluate(df, task="continuous", primary="mae", ours="ours", raw="raw")
        self.assertIn("kontrol_karistir_ours", rep["sonuclar"])
        self.assertIn("demeaned_r2", rep["sonuclar"]["tahmin_ours"]["metrikler"])
        self.assertEqual(rep["sonuclar"]["taban_marjinal"]["etiket"], "Madde medyanı (cross-fitted, 5-fold by row)")

    def test_mixed_frame_single_rows_draw_from_other_items(self):
        df = pd.DataFrame({"madde": ["a", "a", "a", "b", "c", "d"],
                           "gercek": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]})
        v = types.SimpleNamespace(df=df, gorev="surekli", segmentler=[], egitim=None)
        for tohum in range(20):
            out = merdiven._rastgele_surekli(v, tohum)
            self.assertEqual(sorted(out[:3]), [1.0, 2.0, 3.0])  # multi-row item: within-item permutation
            for i in (3, 4, 5):
                self.assertNotEqual(out[i], df["gercek"].iloc[i])
                self.assertIn(out[i], set(df["gercek"].drop(index=i)))
        self.assertEqual(merdiven.baselineler(v)["rastgele"][0], "Rastgele (madde içi gerçek değer)")

    def test_multi_row_items_unchanged(self):
        df = pd.DataFrame({"madde": np.repeat(["a", "b", "c"], 4), "gercek": np.arange(12, dtype=float)})
        v = types.SimpleNamespace(df=df, gorev="surekli", segmentler=[], egitim=None)
        rng = np.random.default_rng(3)
        beklenen = np.empty(12)
        for _, idx in df.groupby("madde").indices.items():
            beklenen[idx] = rng.permutation(df["gercek"].to_numpy(float)[idx])
        np.testing.assert_array_equal(merdiven._rastgele_surekli(v, 3), beklenen)

    def test_single_row_total_keeps_value(self):
        v = types.SimpleNamespace(df=pd.DataFrame({"madde": ["a"], "gercek": [5.0]}), gorev="surekli",
                                  segmentler=[], egitim=None)
        self.assertEqual(list(merdiven._rastgele_surekli(v, 0)), [5.0])


if __name__ == "__main__":
    unittest.main()
