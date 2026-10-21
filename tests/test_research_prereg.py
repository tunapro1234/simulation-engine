"""Pre-registration lock/verify and sealed predictions (offline)."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from simengine.research import (Loop, Prereg, Space, canonical_bytes, grade_sealed, lock_prereg, seal, verify_chain,
                                verify_prereg, verify_seal, write_prereg)


def _prereg():
    return Prereg(title="synthetic", question="Does the loop beat the start config?", data="synthetic objective",
                  split="dev/test by unit id hash", metrics=["err (lower is better)"], baselines={"start": "x=0.2"},
                  expected="winner err below start on test", falsification="winner does not beat start on test "
                                                                           "(Bonferroni CI)", stop_rule="3 not kept")


class TestPrereg(unittest.TestCase):
    def test_write_lock_verify_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("prereg.md", "prereg.json"):
                p = write_prereg(_prereg(), Path(d) / name)
                self.assertEqual(Prereg.read(p).falsification, _prereg().falsification)
                rec = lock_prereg(p)
                self.assertEqual(len(rec["sha256"]), 64)
                self.assertTrue(verify_prereg(p)["ok"])
                self.assertEqual(lock_prereg(p)["sha256"], rec["sha256"])  # idempotent
                p.write_text(p.read_text() + "\nedited after lock\n")
                v = verify_prereg(p)
                self.assertFalse(v["ok"])
                self.assertIn("changed", v["reason"])
                with self.assertRaises(ValueError):
                    lock_prereg(p)
                self.assertTrue(lock_prereg(p, overwrite=True)["sha256"] != rec["sha256"])
            md = (Path(d) / "prereg.md").read_text()
            self.assertIn("## Falsification", md)

    def test_unlocked(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_prereg(_prereg().to_dict(), Path(d) / "p.md")
            v = verify_prereg(p)
            self.assertFalse(v["ok"])
            self.assertIn("not locked", v["reason"])

    @unittest.skipIf(shutil.which("git") is None, "git not installed")
    def test_git_commit(self):
        with tempfile.TemporaryDirectory() as d:
            g = lambda *a: subprocess.run(["git", *a], cwd=d, capture_output=True, text=True, check=True)
            g("init", "-q")
            g("config", "user.email", "test@example.invalid")
            g("config", "user.name", "simengine-test")
            p = write_prereg(_prereg(), Path(d) / "prereg.md")
            rec = lock_prereg(p, git_commit=True)
            self.assertTrue(rec.get("git_commit"))
            v = verify_prereg(p, check_git=True)
            self.assertTrue(v["ok"], v)
            self.assertTrue(v["git"]["committed"])
            self.assertTrue(v["git"]["first_commit_time"])
            p.write_text(p.read_text() + "x")
            self.assertFalse(verify_prereg(p, check_git=True)["ok"])

    def test_loop_requires_locked_prereg(self):
        def runner(cfg):
            return {"per_unit": pd.DataFrame({"unit": range(50), "s": np.linspace(0, 1, 50) + cfg["a"]})}

        sp = Space({"a": [0, 1]}, defaults={"a": 0})
        with tempfile.TemporaryDirectory() as d:
            p = write_prereg(_prereg(), Path(d) / "prereg.md")
            with self.assertRaises(RuntimeError):
                Loop(runner, sp, "grid", primary="s", prereg=p).run()
            lock_prereg(p)
            out = Loop(runner, sp, "grid", primary="s", prereg=p, workdir=Path(d) / "run").run()
            self.assertEqual(out["final"]["winner_cfg"], {"a": 1})
            self.assertIn("Pre-registration `prereg.md`", (Path(d) / "run" / "JOURNAL.md").read_text())


class TestSealed(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"person": [1, 2, 3], "item": ["q1", "q1", "q2"], "p": [0.1, 0.25, 0.9],
                                "dist": [[0.2, 0.8], [0.5, 0.5], [1.0, 0.0]]})

    def test_canonical_is_order_free(self):
        shuffled = self.df.iloc[[2, 0, 1]][["p", "dist", "item", "person"]]
        self.assertEqual(canonical_bytes(self.df), canonical_bytes(shuffled))
        changed = self.df.copy()
        changed.loc[0, "p"] = 0.11
        self.assertNotEqual(canonical_bytes(self.df), canonical_bytes(changed))

    def test_seal_verify_chain_grade(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Path(d) / "reg"
            csv = Path(d) / "pred.csv"
            self.df.drop(columns="dist").to_csv(csv, index=False)
            r1 = seal(csv, "first prediction", reg)
            r2 = seal({"expected": {"lo": 1.15, "hi": 1.6}}, "expected interval", reg)
            self.assertEqual(r2["prev"], r1["sha256"])
            self.assertEqual(seal(csv, "again", reg)["seq"], 0)  # same content -> oldest record
            self.assertTrue(verify_seal(csv, registry_dir=reg)["matched"])
            self.assertTrue(verify_seal(csv, sha256=r1["sha256"], registry_dir=reg)["matched"])
            self.assertTrue(verify_chain(reg)["valid"])
            self.assertEqual(verify_chain(reg)["n"], 2)
            truth = pd.DataFrame({"person": [1, 2, 3], "item": ["q1", "q1", "q2"], "y": [0, 0, 1]})
            g = grade_sealed(csv, truth, ["person", "item"], reg)
            self.assertEqual(g["n_joined"], 3)
            # edited predictions cannot be graded
            bad = Path(d) / "bad.csv"
            self.df.drop(columns="dist").assign(p=[0.9, 0.9, 0.9]).to_csv(bad, index=False)
            self.assertFalse(verify_seal(bad, registry_dir=reg)["matched"])
            with self.assertRaises(ValueError):
                grade_sealed(bad, truth, ["person", "item"], reg)
            # tampering with the chain is detected
            lines = (reg / "chain.jsonl").read_text().splitlines()
            rec = json.loads(lines[0])
            rec["description"] = "rewritten"
            lines[0] = json.dumps(rec)
            (reg / "chain.jsonl").write_text("\n".join(lines) + "\n")
            self.assertFalse(verify_chain(reg)["valid"])
            self.assertEqual(verify_chain(reg)["broken"], [0])

    def test_raw_file_seal(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "forecast.md"
            f.write_text("winner err 0.95 [0.90, 1.00]\n")
            r = seal(f, "forecast text", Path(d) / "reg")
            self.assertTrue(verify_seal(f, r["sha256"], Path(d) / "reg")["matched"])
            f.write_text("winner err 0.80\n")
            self.assertFalse(verify_seal(f, r["sha256"], Path(d) / "reg")["matched"])


if __name__ == "__main__":
    unittest.main()
