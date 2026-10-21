"""Ported from kanit testler/test_ezber.py (memorization audit)."""
import unittest

from simengine.evidence import ezber


class EzberTest(unittest.TestCase):
    def test_isabet(self):
        self.assertTrue(ezber.isabet("The answer is 42%.", 0.42))
        self.assertTrue(ezber.isabet("0.43", 0.42))
        self.assertFalse(ezber.isabet("about 60", 0.42))
        self.assertTrue(ezber.isabet("Winner: Meet The Man", "meet the man"))

    def test_ezber_var(self):
        # gerçek öğeleri "bilen", kontrolde şans düzeyinde kalan sahte model
        og = [{"istem": f"g{i}", "beklenen": "evet", "kontrol": False} for i in range(60)] + \
             [{"istem": f"k{i}", "beklenen": "evet", "kontrol": True} for i in range(60)]
        cagir = lambda s: "evet" if s.startswith("g") or int(s[1:]) % 2 else "hayır"
        r = ezber.olc(og, cagir, B=300)
        self.assertEqual(r["durum"], "uyari")
        self.assertAlmostEqual(r["deger"], 0.5, places=6)

    def test_ezber_yok(self):
        og = [{"istem": f"g{i}", "beklenen": "evet", "kontrol": False} for i in range(60)] + \
             [{"istem": f"k{i}", "beklenen": "evet", "kontrol": True} for i in range(60)]
        cagir = lambda s: "evet" if int(s[1:]) % 2 else "hayır"
        self.assertEqual(ezber.olc(og, cagir, B=300)["durum"], "tamam")

    def test_sinir(self):
        with self.assertRaises(ValueError):
            ezber.olc([{"istem": "x", "beklenen": 1}] * 3, lambda s: "", en_fazla=2)


if __name__ == "__main__":
    unittest.main()
