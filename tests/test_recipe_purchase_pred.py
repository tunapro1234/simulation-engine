"""examples/recipes/purchase_pred.py end to end in a subprocess: offline run (FakeLLM + synthetic shoppers) finishes
with rc=0 and writes the card with the ladder, controls and gap closed; a --data CSV with --persons runs; missing
columns exit; --live refuses to start without a budget and stops at --max-usd before any network call, never
printing the key."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "recipes" / "purchase_pred.py"


def run(*args, timeout=120):
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout,
                          cwd=ROOT, env=env)


class TestPurchasePredRecipe(unittest.TestCase):
    def test_offline_run_writes_card(self):
        with tempfile.TemporaryDirectory() as d:
            p = run("--out", d, "--B", "100")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            for s in ("LLM calls: 481 (1 no-persona + 240 L1 + 240 L2)", "unparsed category answers: 0",
                      "always-no", "llm_l0", "llm_l2_cal", "history", "stack", "oracle", "gap closed:", "badge:"):
                self.assertIn(s, p.stdout)
            card = (Path(d) / "card.md").read_text()
            self.assertIn("Sealed predictions", card)
            self.assertTrue((Path(d) / "dashboard.html").exists())

    def test_user_csv_with_persons_cap(self):
        with tempfile.TemporaryDirectory() as d:
            rows = ["person_id,category,demo_age,demo_income,n_before,n_recent,bought"]
            for i in range(80):                                   # 3 categories; history predicts the outcome
                for j, cat in enumerate(("food", "toys", "books")):
                    nb = (i + j) % 4
                    rows.append(f"u{i},{cat},{('young', 'old')[i % 2]},{('low', 'high')[(i // 2) % 2]},{nb},"
                                f"{nb // 2},{int(nb >= 2 or (i * 7 + j) % 5 == 0)}")
            csv = Path(d) / "cells.csv"
            csv.write_text("\n".join(rows) + "\n")
            p = run("--data", str(csv), "--persons", "60", "--out", str(Path(d) / "run"), "--B", "50")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            self.assertIn("LLM calls: 121 (1 no-persona + 60 L1 + 60 L2)", p.stdout)
            self.assertTrue((Path(d) / "run" / "card.md").exists())

    def test_missing_columns_exit(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "bad.csv"
            csv.write_text("person_id,category,bought\nu1,food,1\n")
            p = run("--data", str(csv), "--out", d)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("missing columns", p.stderr)

    def test_live_needs_budget(self):
        with tempfile.TemporaryDirectory() as d:
            p = run("--live", "--out", d, timeout=60)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("--max-usd", p.stderr)

    def test_live_budget_stops_before_any_call(self):
        with tempfile.TemporaryDirectory() as d:
            key = Path(d) / "k.txt"
            key.write_text("sk-test-not-a-real-key-123\n")
            p = run("--live", "--key-file", str(key), "--max-usd", "0", "--out", str(Path(d) / "run"), timeout=60)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("stopped by the budget", p.stderr)
            self.assertNotIn("sk-test-not-a-real-key", p.stdout + p.stderr)


if __name__ == "__main__":
    unittest.main()
