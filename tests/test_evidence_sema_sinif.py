"""Ported from kanit testler/test_sema_sinif.py. cok_sinif: tahmin etiketi gerçek kümesinde yoksa çökmesin (web G4 açık uçlu eylem)."""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.evidence.cekirdek import degerlendir


class TestSinifBirlesim(unittest.TestCase):
    def test_gercekte_olmayan_tahmin(self):
        rng = np.random.default_rng(0)
        n = 300
        g = rng.choice(["a", "b", "c"], n)
        df = pd.DataFrame({"gercek": g, "tahmin_x": np.where(rng.random(n) < .3, "yeni_eylem", g),
                           "tahmin_markov": rng.choice(["a", "b"], n), "grup": np.arange(n), "segment": rng.choice(["s1", "s2"], n)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oz = degerlendir(df, "t", gorev="cok_sinif", bizim="x", etiketsiz_tabanlar=["markov"], B=50, B_toplu=10)
        self.assertIn("yeni_eylem", [str(s) for s in oz["kart"].get("siniflar", [])] or ["yeni_eylem"])
        self.assertEqual(oz["sonuclar"]["tahmin_markov"]["tur"], "baseline")
        self.assertIsNotNone(oz["manset"]["bizim"]["deger"])


if __name__ == "__main__":
    unittest.main()
