"""Ported from kanit testler/test_onkayit.py: pre-registration seal chain."""
import contextlib
import gzip
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from simengine.evidence import onkayit as ok


def ornek():
    return pd.DataFrame({
        "kisi": ["a", "b", "c", "d"],
        "soru": [1, 1, 2, 2],
        "tahmin_llm": [0.1, 0.25, np.nan, 1 / 3],
        "dagilim": [[0.2, 0.8], [0.5, 0.5], [1.0, 0.0], [0.3, 0.7]],
    })


class KanonikTest(unittest.TestCase):
    def test_sira_bagimsiz(self):
        df = ornek()
        karisik = df[["dagilim", "tahmin_llm", "kisi", "soru"]].iloc[[2, 0, 3, 1]].reset_index(drop=True)
        self.assertEqual(ok.kanonik_bayt(df), ok.kanonik_bayt(karisik))

    def test_tek_hucre_degisir(self):
        df = ornek()
        df2 = df.copy()
        df2.loc[1, "tahmin_llm"] = 0.26
        self.assertNotEqual(ok.kanonik_bayt(df), ok.kanonik_bayt(df2))

    def test_float_hassasiyeti_ve_nan(self):
        a = pd.DataFrame({"x": [0.1 + 0.2, np.nan], "y": [1.0, 2.0]})
        b = pd.DataFrame({"x": [0.3, None], "y": [1, 2]})
        self.assertEqual(ok.kanonik_bayt(a), ok.kanonik_bayt(b))
        self.assertIn(b"nan", ok.kanonik_bayt(a))

    def test_dosyadan_okuma(self):
        df = pd.DataFrame({"k": ["a", "b"], "p": [0.5, 0.125]})
        with tempfile.TemporaryDirectory() as d:
            csv_yol = Path(d) / "t.csv"
            df.to_csv(csv_yol, index=False)
            js_yol = Path(d) / "t.json"
            df.to_json(js_yol, orient="records")
            self.assertEqual(ok.kanonik_bayt(df), ok.kanonik_bayt(csv_yol))
            self.assertEqual(ok.kanonik_bayt(df), ok.kanonik_bayt(js_yol))


class MuhurTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.dizin = Path(self._td.name) / ".onkayit"

    def tearDown(self):
        self._td.cleanup()

    def test_muhurle_ve_arsiv(self):
        k = ok.muhurle(ornek(), "ilk", self.dizin)
        self.assertEqual(k["sira"], 0)
        self.assertIsNone(k["onceki"])
        self.assertEqual((k["satir"], k["sutun"]), (4, 4))
        arsiv = gzip.decompress((self.dizin / k["arsiv"]).read_bytes())
        self.assertEqual(hashlib.sha256(arsiv).hexdigest(), k["sha256"])
        self.assertEqual(len((self.dizin / "zincir.jsonl").read_text().splitlines()), 1)

    def test_zincir_bagi(self):
        k1 = ok.muhurle(ornek(), "bir", self.dizin)
        df2 = ornek()
        df2.loc[0, "soru"] = 9
        k2 = ok.muhurle(df2, "iki", self.dizin)
        self.assertEqual(k2["onceki"], k1["sha256"])
        self.assertEqual(k2["onceki_kayit"], k1["kayit_hash"])
        self.assertEqual(ok.zincir_dogrula(self.dizin), {"gecerli": True, "n": 2, "kirik": []})

    def test_zincir_kurcalama(self):
        ok.muhurle(ornek(), "bir", self.dizin)
        ok.muhurle(ornek().iloc[:2], "iki", self.dizin)
        ok.muhurle(ornek().iloc[:3], "üç", self.dizin)
        yol = self.dizin / "zincir.jsonl"
        satirlar = [json.loads(s) for s in yol.read_text().splitlines()]
        satirlar[1]["aciklama"] = "sonradan değiştirildi"
        yol.write_text("".join(json.dumps(s) + "\n" for s in satirlar))
        z = ok.zincir_dogrula(self.dizin)
        self.assertFalse(z["gecerli"])
        self.assertEqual(z["kirik"], [1])

    def test_dogrula(self):
        k = ok.muhurle(ornek(), "ilk", self.dizin)
        karisik = ornek().iloc[::-1]
        s = ok.dogrula(karisik, kayit_dizini=self.dizin)
        self.assertTrue(s["eslesti"])
        self.assertEqual(s["kayit"]["sha256"], k["sha256"])
        self.assertTrue(ok.dogrula(karisik, sha256=k["sha256"], kayit_dizini=self.dizin)["eslesti"])
        degisik = ornek()
        degisik.loc[2, "kisi"] = "z"
        s2 = ok.dogrula(degisik, kayit_dizini=self.dizin)
        self.assertFalse(s2["eslesti"])
        self.assertIsNone(s2["kayit"])
        self.assertFalse(ok.dogrula(degisik, sha256=k["sha256"], kayit_dizini=self.dizin)["eslesti"])

    def test_notla(self):
        ok.muhurle(ornek(), "ilk", self.dizin)
        gercek = pd.DataFrame({"kisi": ["a", "b", "c"], "soru": [1, 1, 2], "gercek": [0, 1, 1]})
        s = ok.notla(ornek(), gercek, ["kisi", "soru"], self.dizin)
        self.assertTrue(s["dogrulama"]["eslesti"])
        self.assertEqual(len(s["tablo"]), 3)
        self.assertIn("gercek", s["tablo"].columns)
        degisik = ornek()
        degisik.loc[0, "tahmin_llm"] = 0.9
        with self.assertRaises(ValueError):
            ok.notla(degisik, gercek, ["kisi", "soru"], self.dizin)

    def test_git_yoksa_uyari(self):
        with self.assertWarns(UserWarning):
            ok.muhurle(ornek(), "git yok", self.dizin, git_commit=True)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            yol = Path(d) / "t.csv"
            ornek().drop(columns="dagilim").to_csv(yol, index=False)
            cikti = io.StringIO()
            with contextlib.redirect_stdout(cikti):
                self.assertEqual(ok.main(["--dizin", str(self.dizin), "muhurle", str(yol), "--aciklama", "cli"]), 0)
                self.assertEqual(ok.main(["--dizin", str(self.dizin), "--json", "dogrula", str(yol)]), 0)
                self.assertEqual(ok.main(["--dizin", str(self.dizin), "zincir"]), 0)
            self.assertIn("Mühürlendi", cikti.getvalue())
            self.assertIn("GEÇERLİ", cikti.getvalue())


if __name__ == "__main__":
    unittest.main()
