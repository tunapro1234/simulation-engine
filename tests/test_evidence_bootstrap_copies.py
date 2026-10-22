"""Copy-aware item codes in the cluster bootstrap of aggregate metrics (G3 #3). Synthetic data.

A cluster drawn k times must not pair its rows across copies (A1-B2, B1-A2): item-grouped aggregate metrics
(cift_isaret, kazanan_top1, ...) would weight it k^2 and inflate the CI. Holds for cluster = item, cluster = a
group column equal to the item, and the two-way block bootstrap.
"""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.evidence import bootstrap as bt
from simengine.evidence import metrik as M

CIFT = next(m for m in M.aile("ab") if m.ad == "cift_isaret")
# per-row pair correctness (both rows of a test carry the test's value): same point, unit-metric CI
DOGRU = M.Metrik("cift_dogru", "birim", lambda y, p: y, +1)


def _ab(n=400, tohum=1):
    rng = np.random.default_rng(tohum)
    ya = rng.random(n) < 0.5
    dogru = rng.random(n) < 0.7
    pa = np.where(ya == dogru, 0.8, 0.2)
    madde = np.repeat(np.arange(n), 2)
    y = np.column_stack([ya, ~ya]).astype(float).reshape(-1)
    p = np.column_stack([pa, 1 - pa]).reshape(-1)
    return madde, y, p, dogru.astype(float)


def _genislik(orn, madde, y, p, dogru):
    nokta, tekrar = bt.hesapla(orn, {"ours": (y, p, None)}, [CIFT], madde)
    a, u = bt.ga(tekrar["ours"]["cift_isaret"])
    _, tb = bt.hesapla(orn, {"d": (np.repeat(dogru, 2), p, None)}, [DOGRU])
    a2, u2 = bt.ga(tb["d"]["cift_dogru"])
    return nokta["ours"]["cift_isaret"], u - a, u2 - a2


class TestKopyaMadde(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.madde, cls.y, cls.p, cls.dogru = _ab()
        n = len(cls.dogru)
        cls.oran = float(cls.dogru.mean())
        cls.binom = 2 * 1.96 * np.sqrt(cls.oran * (1 - cls.oran) / n)

    def test_madde_kume_and_group_equal_binomial(self):
        for madde_kume in (True, False):      # False: cluster is a group column that equals the item
            orn = bt.Orneklem(self.madde, B=300, B_toplu=300, madde_kume=madde_kume)
            nokta, gen, _ = _genislik(orn, self.madde, self.y, self.p, self.dogru)
            self.assertAlmostEqual(nokta, self.oran, places=12)
            self.assertLess(abs(gen / self.binom - 1), 0.15, (madde_kume, gen, self.binom))

    def test_block_matches_unit_metric(self):
        # two-way (block x cluster) bootstrap is wider than binomial by design (product weights); the aggregate
        # metric must match the same quantity computed as a unit metric, not exceed it (was ~2x before the fix)
        blok = self.madde % 20
        hucre = self.madde
        hb, hc = np.zeros(len(self.dogru), int), np.zeros(len(self.dogru), int)
        hb[hucre], hc[hucre] = blok, self.madde
        orn = bt.Orneklem(hucre, B=300, B_toplu=300, iki_yon=(hb, hc))
        _, gen, gen_birim = _genislik(orn, self.madde, self.y, self.p, self.dogru)
        self.assertLess(abs(gen / gen_birim - 1), 0.15, (gen, gen_birim))

    def test_no_repeat_keeps_item_grouping(self):
        class Kimlik(bt.Orneklem):
            def secim(self, b):
                return np.arange(self.G)
        kume = np.repeat(np.arange(6), 4)
        madde = np.tile(np.arange(4), 6)          # items cross clusters
        orn = Kimlik(kume, B=10, B_toplu=10)
        idx, kod = orn.satirlar(0, madde)
        self.assertTrue((pd.factorize(kod)[0] == pd.factorize(madde[idx])[0]).all())

    def test_repeat_splits_nested_keeps_crossing(self):
        class Iki(bt.Orneklem):
            def secim(self, b):
                return np.array([0, 0, 1])          # cluster 0 drawn twice
        kume = np.array([0, 0, 0, 1, 1, 2])
        madde = np.array([0, 0, 2, 1, 2, 3])        # 0 nested in cluster 0; 2 crosses clusters 0 and 1
        orn = Iki(kume, B=10, B_toplu=10)
        self.assertEqual(bt.madde_yuvali(kume, madde).tolist(), [True, True, False, True])
        idx, kod = orn.satirlar(0, madde)
        m = madde[idx]
        nested = kod[m == 0]
        self.assertEqual(len(set(nested[:2])) + len(set(nested[2:])), 2)
        self.assertNotEqual(nested[0], nested[2])               # copies of item 0 are separate items
        self.assertEqual(len(set(kod[m == 2])), 1)              # crossing item stays one item
        self.assertEqual(len(set(kod)), 4)                    # 0 (x2), 2, 1

    def test_legacy_position_codes_without_items(self):
        orn = bt.Orneklem(np.repeat(np.arange(5), 2), B=10, B_toplu=10, madde_kume=True)
        idx, kod = orn.satirlar(3)
        self.assertEqual(len(set(kod)), 5)
        self.assertIsNone(bt.Orneklem(np.arange(5), B=10, B_toplu=10).satirlar(0)[1])

    def test_evaluate_group_equals_item(self):
        import simengine.evidence as ev
        madde, y, p, dogru = _ab(n=200, tohum=3)
        df = pd.DataFrame({"item": madde, "truth": y, "pred_ours": p})
        d2 = df.assign(group=df["item"])
        gen = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for d in (df, d2):
                rep = ev.evaluate(d, ours="ours", task="ab", B=200, multiplier=False)
                k = next(k for k in rep["sonuclar"] if "ours" in k)
                m = rep["sonuclar"][k]["metrikler"]["cift_isaret"]
                gen.append(m["ust"] - m["alt"])
        self.assertLess(abs(gen[1] / gen[0] - 1), 0.15, gen)

    def test_segment_rows_copy_aware(self):
        # per-segment CI (analiz.segment_analizi) resamples the segment's clusters itself; it was ~1.5-1.7x
        # binomial before the copy-aware codes reached it
        import simengine.evidence as ev
        madde, y, p, dogru = _ab(n=400, tohum=5)
        seg = np.where(madde % 2, "x", "y")
        df = pd.DataFrame({"item": madde, "truth": y, "pred_ours": p, "segment": seg})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = ev.evaluate(df, ours="ours", task="ab", B=500, multiplier=False)
        satir = rep["segment"]["segment"]["satirlar"]
        self.assertEqual(len(satir), 2)
        for r in satir:
            m = r["yontemler"][next(k for k in r["yontemler"] if "ours" in k)]
            d = dogru[np.flatnonzero(seg[::2] == r["segment"])]
            binom = 2 * 1.96 * np.sqrt(d.mean() * (1 - d.mean()) / len(d))
            self.assertAlmostEqual(m["deger"], d.mean(), places=12)
            self.assertLess(abs((m["ust"] - m["alt"]) / binom - 1), 0.3, (r["segment"], m, binom))


if __name__ == "__main__":
    unittest.main()
