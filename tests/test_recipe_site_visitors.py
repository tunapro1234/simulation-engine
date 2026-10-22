"""examples/recipes/site_visitors.py end to end in a subprocess: offline run (FakeLLM + synthetic funnel) finishes
with rc=0 and writes the card; a --data CSV without a date column uses the visitor split; --live refuses to start
without a budget or key and stops at --max-usd before any network call, never printing the key; a run where most
calls fail (here: a model that cannot turn thinking off) stops instead of writing a card of 0.5s."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "recipes" / "site_visitors.py"


def run(*args, timeout=120, env=None):
    e = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"} if env is None else env
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout,
                          cwd=ROOT, env=e)


class TestSiteVisitorsRecipe(unittest.TestCase):
    def test_offline_run_writes_card(self):
        with tempfile.TemporaryDirectory() as d:
            p = run("--out", d, "--B", "100")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            for s in ("LLM calls: 40", "failed 0", "lift B-A: observed", "train visitors (LLM-free)", "no-persona",
                      "shuffled_stack", "baseline_marginal", "stack", "badge:"):
                self.assertIn(s, p.stdout)
            card = (Path(d) / "card.md").read_text()
            self.assertIn("Sealed predictions", card)
            self.assertTrue((Path(d) / "dashboard.html").exists())

    def test_user_csv_without_date(self):
        with tempfile.TemporaryDirectory() as d:
            rows = ["visitor_id,segment,persona,variant,step,step_order,page_text,continued"]
            for i in range(240):                                  # two-step funnel, B converts more
                seg, var = ("new", "returning")[i % 2], "AB"[(i // 2) % 2]
                go1 = int(i % 3 != 0)
                rows.append(f"v{i},{seg},a {seg} visitor,{var},landing,1,landing page {var},{go1}")
                if go1:
                    rows.append(f"v{i},{seg},a {seg} visitor,{var},checkout,2,checkout page {var},"
                                f"{int(var == 'B' or i % 5 == 0)}")
            csv = Path(d) / "funnel.csv"
            csv.write_text("\n".join(rows) + "\n")
            p = run("--data", str(csv), "--out", str(Path(d) / "run"), "--B", "50")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            self.assertIn("LLM calls: 12", p.stdout)               # 2 segments x 2 variants x 2 steps + 4 controls

    def test_missing_columns_exit(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "bad.csv"
            csv.write_text("visitor_id,variant\nv1,A\n")
            p = run("--data", str(csv), "--out", d)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("missing columns", p.stderr)

    def test_live_needs_budget_and_key(self):
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

    def test_live_all_calls_failing_exits(self):
        # Opus cannot turn thinking off -> ValueError in every slot before any network call; the run must stop,
        # not fill 40 slots with 0.5 and write a card
        with tempfile.TemporaryDirectory() as d:
            key = Path(d) / "k.txt"
            key.write_text("sk-test-not-a-real-key-456\n")
            p = run("--live", "--key-file", str(key), "--max-usd", "1", "--model", "claude-opus-5-5",
                    "--out", str(Path(d) / "run"), timeout=60)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("40 of 40 calls failed", p.stderr)
            self.assertFalse((Path(d) / "run" / "card.md").exists())
            self.assertNotIn("sk-test-not-a-real-key", p.stdout + p.stderr)

    def test_unreadable_key_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = run("--live", "--key-file", str(Path(d) / "missing.txt"), "--max-usd", "1", "--out", d, timeout=60)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("--key-file cannot be read", p.stderr)
            self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
