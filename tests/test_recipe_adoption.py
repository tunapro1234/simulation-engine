"""examples/recipes/adoption.py end to end in a subprocess: offline run (FakeLLM + synthetic markets) finishes with
rc=0 in time and prints the backtest, the spend what-ifs and the evidence card; a --data CSV with only series, t, y
works; --live refuses to start without a budget or a key and stops at --max-usd before any call, never printing
the key."""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "recipes" / "adoption.py"


def run(*args, timeout=120):
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "SIMENGINE_KEY_FILE")}
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout,
                          cwd=ROOT, env=env)


class TestAdoptionRecipe(unittest.TestCase):
    def test_offline_run_writes_card(self):
        with tempfile.TemporaryDirectory() as d:
            t0 = time.monotonic()
            p = run("--out", d, "--B", "100")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            self.assertLess(time.monotonic() - t0, 60)
            for s in ("LLM calls: 11", "unusable replies (default prior used): 0", "MAE llm_generic", "MAE bass_abc", "what-ifs for m09", "spend x2",
                      "adopter types", "blend - blend_generic", "badge:"):
                self.assertIn(s, p.stdout)
            card = (Path(d) / "card.md").read_text()
            self.assertIn("Sealed predictions", card)
            self.assertTrue((Path(d) / "dashboard.html").exists())

    def test_user_csv_series_t_y_only(self):
        with tempfile.TemporaryDirectory() as d:
            rows, rs = ["series,t,y"], np.random.default_rng(1)
            for i in range(6):                                     # six Bass-shaped yearly curves, staggered starts
                for t in range(2000, 2022):
                    s = max(t - 1998 - i, 0)
                    e = np.exp(-(0.01 + 0.4) * s)
                    y = (55 + 4 * i) * (1 - e) / (1 + 40 * e) + rs.normal(0, 0.3)
                    rows.append(f"k{i},{t},{min(max(y, 0), 100):.3f}")
            csv = Path(d) / "own.csv"
            csv.write_text("\n".join(rows) + "\n")
            p = run("--data", str(csv), "--out", str(Path(d) / "run"), "--B", "50")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            self.assertIn("LLM calls: 7", p.stdout)                # 6 markets + 1 generic control
            self.assertNotIn("cross-market spend elasticity", p.stdout)   # no spend column: check skipped

    def test_missing_columns_exit(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "bad.csv"
            csv.write_text("series,year\na,2001\n")
            p = run("--data", str(csv), "--out", d)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("missing columns", p.stderr)

    def test_live_needs_budget_and_key(self):
        with tempfile.TemporaryDirectory() as d:
            p = run("--live", "--out", d, timeout=60)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("--max-usd", p.stderr)
            p = run("--live", "--max-usd", "0.1", "--out", d, timeout=60)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("no API key", p.stderr)

    def test_live_budget_stops_before_any_call(self):
        with tempfile.TemporaryDirectory() as d:
            key = Path(d) / "k.txt"
            key.write_text("sk-test-not-a-real-key-123\n")
            p = run("--live", "--key-file", str(key), "--max-usd", "0", "--out", str(Path(d) / "run"), timeout=60)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("no new call made", p.stderr)
            self.assertNotIn("sk-test-not-a-real-key", p.stdout + p.stderr)
            p = run("--live", "--key-file", str(key), "--max-usd", "0.1", "--model", "no-such-model",
                    "--out", str(Path(d) / "run"), timeout=60)            # unpriced model: message, no traceback
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("--live setup failed", p.stderr)
            self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
