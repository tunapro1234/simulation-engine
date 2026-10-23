"""runs/latent/kos.py end to end on a fake prepared-data folder with tiny random models (no download, no seal)."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests._latent_util import HAVE_TORCH, fake_root

YOL = Path(__file__).resolve().parents[1] / "runs" / "latent" / "kos.py"


def _kos():
    spec = importlib.util.spec_from_file_location("latent_kos", YOL)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m                     # dataclasses look their module up
    spec.loader.exec_module(m)
    return m


class TestArmParse(unittest.TestCase):
    def test_parse(self):
        k = _kos()
        a = k.parse_arm("rsd-angc-swap", "q38")
        self.assertEqual((a.name, a.kind, a.model, a.u, a.variant, a.task, a.base),
                         ("rsd-q38-angc-swap", "rsd", "q38", "angc", "swap", "g1", "rsd-q38-angc"))
        self.assertEqual(k.parse_arm("prb-tiny_gemma4-L2").model, "tiny-gemma4")
        self.assertEqual(k.parse_arm("prm-tf0", "g31").task, "g3")
        self.assertEqual(k.parse_arm("enc-tw-e5").task, "g2")
        self.assertEqual(k.parse_arm("caa", "q3").name, "caa-q3")
        for bad in ("prm-L1-swap", "prb-L0", "foo-an"):
            with self.assertRaises(ValueError):
                k.parse_arm(bad, "q38")
        with self.assertRaises(ValueError):
            k.parse_arm("prb-L2")                                  # no model


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestRunnerSmoke(unittest.TestCase):
    def run_kos(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            rc = self.k.main(["--base", self.tmp.name, "--root", str(self.root), "--device", "cpu", "--precision",
                              "fp32", "--kucuk", "8:4:4", "--threads", "4", *args])
        return rc, [json.loads(l) for l in out.getvalue().splitlines() if l.startswith("{")]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = fake_root(Path(self.tmp.name) / "veri")
        self.k = _kos()

    def tearDown(self):
        self.tmp.cleanup()

    def test_g1_arms_resume_and_controls(self):
        arms = ["enc-an", "prm-L1", "prb-L2", "caa", "rsd-angc", "rsd-angc-swap", "sft-angc"]
        rc, res = self.run_kos("--model", "tiny-qwen3_5", "--arm", *arms)
        self.assertEqual(rc, 0, res)
        self.assertEqual([r["status"] for r in res], ["ok"] * len(arms))
        d = Path(self.tmp.name) / "kucuk" / "talp73" / "tahmin"
        p = pd.read_parquet(d / "prb-tiny_qwen3_5-L2_s0.parquet")
        self.assertEqual(list(p.columns), ["uid", "kategori", "p"])
        self.assertEqual(len(p), 4 * 4)                             # 4 test persons x 4 categories
        meta = json.loads((d / "rsd-tiny_qwen3_5-angc_s0.meta.json").read_text())
        for key in ("model", "precision", "layer", "config", "seconds", "card", "val"):
            self.assertIn(key, meta)
        a = pd.read_parquet(d / "rsd-tiny_qwen3_5-angc_s0.parquet").p
        b = pd.read_parquet(d / "rsd-tiny_qwen3_5-angc-swap_s0.parquet").p
        self.assertGreater(float((a - b).abs().max()), 0)           # swap moves the test predictions
        log = [json.loads(l) for l in (Path(self.tmp.name) / "kucuk" / "kosular.jsonl").read_text().splitlines()]
        self.assertEqual(len(log), len(arms))
        self.assertTrue(all("precision" in r and "card" in r for r in log))
        rc, res = self.run_kos("--model", "tiny-qwen3_5", "--arm", "prb-L2", "rsd-angc")
        self.assertEqual([r["why"] for r in res], ["done", "done"])  # resume: finished runs are skipped

    def test_g3_g2_arms_and_score(self):
        rc, res = self.run_kos("--model", "tiny-gemma4", "--arm", "enc-tf", "prm-tf", "rsd-tf", "enc-tw", "prm-tw")
        self.assertEqual(rc, 0, res)
        d = pd.read_parquet(Path(self.tmp.name) / "kucuk" / "twin_fiyat" / "tahmin" / "prm-tiny_gemma4-tf_s0.parquet")
        self.assertEqual(int((d.tur == "sweep").sum()), 4 * 3 * 11)  # test persons x target products x levels
        rc, res = self.run_kos("--score", "g3", "--model", "tiny-gemma4", "--arm", "enc-tf", "rsd-tf", "--B", "20")
        self.assertEqual({r["arm"] for r in res}, {"enc-tf", "rsd-tiny_gemma4-tf"})
        self.assertIn("s_hi", res[0])

    def test_plan_runs_items_in_order(self):
        self.k.PLANS["test"] = [{"arms": ["enc-an"]}, {"model": "tiny-qwen3_5", "precision": "fp32",
                                                      "arms": ["prm-L1", "prb-L2"]}]
        try:
            rc, res = self.run_kos("--plan", "test", "--until", "23:59" if __import__("time").strftime("%H:%M")
                                   < "23:58" else "00:01")
        finally:
            del self.k.PLANS["test"]
        self.assertEqual(rc, 0, res)
        self.assertEqual([r["arm"] for r in res], ["enc-an", "prm-tiny_qwen3_5-L1", "prb-tiny_qwen3_5-L2"])
        self.assertGreater(self.k._deadline("07:30"), __import__("datetime").datetime.now())

    def test_error_is_logged_and_run_goes_on(self):
        rc, res = self.run_kos("--model", "tiny-nosuchfamily", "--arm", "prm-L1", "enc-an")
        self.assertEqual(rc, 1)
        self.assertEqual([r["status"] for r in res], ["error", "ok"])


if __name__ == "__main__":
    unittest.main()
