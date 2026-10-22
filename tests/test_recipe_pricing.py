"""examples/recipes/pricing.py end to end in a subprocess: offline default run, a user CSV without a design column,
and the --live guard (no LLM call is made in any of these)."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "recipes" / "pricing.py"


def run(*args, timeout=120):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout, env={"PATH": "/usr/bin:/bin", "ANTHROPIC_API_KEY": ""})


def small_cbc(path, n=120, seed=3):
    """A tiny CBC table in the documented --data format (no design column: design defaults to task)."""
    rs, rows = np.random.default_rng(seed), []
    for i in range(n):
        seg = ["student", "family"][i % 2]
        for t in range(4):
            alts = {a: dict(price=float(rs.choice([229, 299, 379])), meals=float(rs.choice([3, 5])),
                            organic=float(rs.integers(2)), free_cancel=float(rs.integers(2))) for a in "AB"}
            alts["none"] = dict(price=0.0, meals=0.0, organic=0.0, free_cancel=0.0)
            u = {a: (-1.5 if a == "none" else -o["price"] / 100 + 0.3 * o["meals"] + 0.5 * o["organic"])
                 + rs.gumbel() for a, o in alts.items()}
            rows += [{"person": f"r{i}", "segment": seg, "persona": f"a {seg}", "task": f"r{i}_t{t}",
                      "alternative": a, **o, "chosen": int(a == max(u, key=u.get))} for a, o in alts.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


class TestRecipePricing(unittest.TestCase):
    def test_offline_default(self):
        with tempfile.TemporaryDirectory() as d:
            r = run("--out", d, "--B", "60")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            for s in ("LLM calls: 80", "WTP in TL per unit", "price curve for A", "revenue-max price", "badge:"):
                self.assertIn(s, r.stdout)
            for f in ("card.md", "dashboard.html", "price_curve.csv"):
                self.assertTrue((Path(d) / f).exists(), f)
            self.assertTrue(list((Path(d) / ".onkayit").glob("*.jsonl")))       # predictions sealed
            curve = pd.read_csv(Path(d) / "price_curve.csv")
            self.assertTrue((curve.human.diff().dropna() < 0).all())            # demand falls with price
            self.assertTrue(((curve.lo <= curve.human) & (curve.human <= curve.hi)).all())
            card = (Path(d) / "card.md").read_text()
            self.assertIn("stack", card)

    def test_user_csv_without_design(self):
        with tempfile.TemporaryDirectory() as d:
            small_cbc(Path(d) / "cbc.csv")
            r = run("--data", str(Path(d) / "cbc.csv"), "--out", str(Path(d) / "out"), "--B", "30")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn("WTP in TL per unit", r.stdout)

    def test_missing_columns_and_live_guard(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({"person": ["a"], "task": ["t"]}).to_csv(Path(d) / "bad.csv", index=False)
            r = run("--data", str(Path(d) / "bad.csv"), "--out", d)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("missing columns", r.stderr)
            r = run("--live", "--out", d)                                      # no --max-usd: stops before any call
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("--max-usd", r.stderr)


if __name__ == "__main__":
    unittest.main()
