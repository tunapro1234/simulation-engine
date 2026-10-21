"""Ported from kanit testler/test_kart.py: evidence card schema and badge rule."""
import unittest

import numpy as np
import pandas as pd

from simengine.evidence import kart
from simengine.evidence.cekirdek import degerlendir

ANAHTAR = {"baslik", "alt_baslik", "metrik", "n", "kaynak", "satirlar", "fark", "rejimler", "yanilgi", "muhur",
           "rozet", "rozet_gerekce", "etiketler", "dil"}
SATIR = {"tur", "etiket", "deger", "alt", "ust", "birim", "not", "kaynak"}


class TestKart(unittest.TestCase):
    def test_sema_ve_rozet(self):
        rng = np.random.default_rng(0)
        n = 600
        y = rng.normal(0, 1, n)
        df = pd.DataFrame({"gercek": y, "tahmin_iyi": y + rng.normal(0, 0.5, n), "tahmin_taban": np.zeros(n),
                           "grup": np.arange(n)})
        oz = degerlendir(df, "deneme", gorev="surekli", bizim="iyi", etiketsiz_tabanlar=["taban"], B=100, B_toplu=20)
        k = oz["kanit_kart"]
        self.assertEqual(set(k), ANAHTAR)
        self.assertEqual([r["tur"] for r in k["satirlar"]], list(kart.SIRA))
        for r in k["satirlar"]:
            self.assertEqual(set(r), SATIR)
        self.assertIn(k["rozet"], ("yuksek", "kararsiz", "bekliyor"))
        bedel = k["satirlar"][-1]
        self.assertEqual(bedel["birim"], "×")
        self.assertGreater(bedel["deger"], 1.5)  # ρ ≈ 0,89 → kuram ~4,9
        self.assertIn("kanıtlanmadı: sızıntı/ezber", k["etiketler"])  # ezber/kontaminasyon atlandı

    def test_taban_altinda_yuksek_olmaz(self):
        S = [kart.satir("taban", "t", 0.8), kart.satir("kalibre", "k", 0.6), kart.satir("bedel", "b", 1.7, 1.5, 2.0, "×")]
        r, _ = kart.rozet(S, {"fark": -0.2, "alt": -0.3, "ust": -0.1})
        self.assertEqual(r, "kararsiz")

    def test_rozet_bizim_bedel(self):  # KARŞIT-2 K2-28: rozet yalnız bizim yöntemin bedeline ve tabana farkına bakar
        S = lambda alt: [kart.satir("taban", "t", 0.8), kart.satir("kalibre", "k", 0.8),
                         kart.satir("bedel", "b", 1.8, alt, 2.0, "×")]
        sifir = {"fark": 0.001, "alt": -0.022, "ust": 0.023}
        self.assertEqual(kart.rozet(S(1.5), sifir, bedel_kaynak="ham")[0], "kararsiz")      # oe_sinif örneği
        self.assertEqual(kart.rozet(S(1.5), sifir, bedel_kaynak="bizim")[0], "kararsiz")    # fark sıfırı içeriyor
        self.assertEqual(kart.rozet(S(1.5), None, bedel_kaynak="ham")[0], "kararsiz")       # başka kolun çarpanı dayanak değil
        self.assertEqual(kart.rozet(S(1.12), None, bedel_kaynak="bizim")[0], "yuksek")
        self.assertEqual(kart.rozet(S(1.06), None, bedel_kaynak="bizim")[0], "kararsiz")    # < ESIK_BEDEL
        self.assertEqual(kart.rozet(S(1.3), None, bedel_kaynak="bizim", taban_bedel_alt=1.4)[0], "kararsiz")
        iyi = {"fark": 0.05, "alt": 0.02, "ust": 0.08}
        self.assertEqual(kart.rozet(S(1.3), iyi, bedel_kaynak="bizim")[0], "yuksek")
        self.assertEqual(kart.rozet(S(1.05), iyi, bedel_kaynak="bizim")[0], "kararsiz")
        self.assertEqual(kart.rozet(S(1.05), iyi, bedel_kaynak="ham")[0], "yuksek")         # ham kolun bedeli fark kuralını bozmaz
        # karşıt doğrulayıcı: taban varken fark yoksa ya da GA'sız fark bedelle "yüksek" olmaz; eksik alanda çökmez
        self.assertEqual(kart.rozet(S(1.5), None, bedel_kaynak="bizim", taban_var=True)[0], "kararsiz")
        self.assertEqual(kart.rozet(S(1.5), {"fark": 0.05, "alt": None, "ust": None}, bedel_kaynak="bizim")[0], "kararsiz")
        self.assertEqual(kart.rozet(S(1.5), {"fark": None, "alt": None, "ust": -0.1}, bedel_kaynak="bizim")[0], "kararsiz")

    def test_gercek_yoksa_bekliyor(self):
        r, _ = kart.rozet([kart.satir("taban", "t"), kart.satir("ham", "h"), kart.satir("kalibre", "k")])
        self.assertEqual(r, "bekliyor")


if __name__ == "__main__":
    unittest.main()
