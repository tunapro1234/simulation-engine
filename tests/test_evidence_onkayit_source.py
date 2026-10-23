"""Seal record 'kaynak' (D6 #9): only the file name by default, never an absolute machine path; the full path only
with kaynak_yolu="full" (or "tam"); chains written with full paths (older records) still verify unchanged."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from simengine.evidence import onkayit


class TestSealSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.pred = self.dir / "sub" / "preds.csv"
        self.pred.parent.mkdir()
        pd.DataFrame({"id": [1, 2, 3], "pred_x": [0.1, 0.5, 0.9]}).to_csv(self.pred, index=False)
        self.reg = self.dir / "reg"

    def tearDown(self):
        self.tmp.cleanup()

    def _chain_text(self):
        return (self.reg / onkayit.ZINCIR_ADI).read_text(encoding="utf-8")

    def test_default_stores_file_name_only(self):
        k = onkayit.muhurle(str(self.pred.resolve()), "test", self.reg, kopya=False)
        self.assertEqual(k["kaynak"], "preds.csv")
        self.assertNotIn(str(self.dir), self._chain_text())
        self.assertTrue(onkayit.zincir_dogrula(self.reg)["gecerli"])

    def test_path_object_and_dataframe(self):
        self.assertEqual(onkayit.muhurle(self.pred, "p", self.reg, kopya=False)["kaynak"], "preds.csv")
        self.assertIsNone(onkayit.muhurle(pd.read_csv(self.pred), "df", self.reg, kopya=False)["kaynak"])

    def test_full_option_keeps_path(self):
        k = onkayit.muhurle(str(self.pred), "full", self.reg, kopya=False, kaynak_yolu="full")
        self.assertEqual(k["kaynak"], str(self.pred))
        with self.assertRaises(ValueError):
            onkayit.muhurle(str(self.pred), "bad", self.reg, kopya=False, kaynak_yolu="abs")

    def test_cli_default_name_only(self):
        with contextlib.redirect_stdout(io.StringIO()):
            onkayit.main(["--dizin", str(self.reg), "--json", "muhurle", str(self.pred), "--aciklama", "cli"])
        rec = json.loads(self._chain_text().splitlines()[-1])
        self.assertEqual(rec["kaynak"], "preds.csv")

    def test_old_chain_with_full_paths_still_verifies(self):
        # an older chain: records hashed with the absolute path in 'kaynak'
        onkayit.muhurle(str(self.pred), "old 1", self.reg, kopya=False, kaynak_yolu="full")
        onkayit.muhurle(str(self.pred), "old 2", self.reg, kopya=False, kaynak_yolu="full")
        before = self._chain_text()
        self.assertIn(str(self.pred), before)
        self.assertEqual(onkayit.zincir_dogrula(self.reg), {"gecerli": True, "n": 2, "kirik": []})
        # a new default record appends to it; the old records are untouched and the chain stays valid
        k = onkayit.muhurle(str(self.pred), "new", self.reg, kopya=False)
        self.assertEqual(k["kaynak"], "preds.csv")
        self.assertTrue(self._chain_text().startswith(before))
        self.assertEqual(onkayit.zincir_dogrula(self.reg), {"gecerli": True, "n": 3, "kirik": []})
        self.assertTrue(onkayit.dogrula(str(self.pred), kayit_dizini=self.reg)["eslesti"])


if __name__ == "__main__":
    unittest.main()
