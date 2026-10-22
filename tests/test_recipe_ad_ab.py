"""examples/recipes/ad_ab.py end to end in a subprocess: the offline run (FakeLLM + synthetic campaign) finishes with
rc=0 in time and prints the lift table, the ranking and the evidence card; a --data CSV without dates, a renamed
control arm and an --audience subset work; missing columns are refused; --live refuses to start without a budget
or a key and stops at --max-usd before any call, never printing the key."""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "recipes" / "ad_ab.py"


def run(*args, timeout=120):
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "SIMENGINE_KEY_FILE")}
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout,
                          cwd=ROOT, env=env)


class TestAdABRecipe(unittest.TestCase):
    def test_offline_run_writes_card(self):
        with tempfile.TemporaryDirectory() as d:
            t0 = time.monotonic()
            p = run("--out", d, "--B", "100")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            self.assertLess(time.monotonic() - t0, 60)
            for s in ("LLM calls: 25 (5 no-persona, 0 unparsed", "'yes' bias", "observed lift [95% CI]",
                      "stack lift [95% CI]", "ranking: stack", "cell_rate alone", "auc (person level, test)",
                      "llm_nopersona", "shuffled_stack", "baseline_random", "baseline_segment_mean", "badge:"):
                self.assertIn(s, p.stdout)
            self.assertNotIn("selection bias", p.stdout.split("badge:")[1])   # "ours" is named in prereg.md
            for ad in ("family", "green", "price", "speed"):
                self.assertRegex(p.stdout, rf"\n{ad} +[+-]\d\.\d{{3}} \[[+-]\d\.\d{{3}}, [+-]\d\.\d{{3}}\]")
            self.assertNotRegex(p.stdout, r"\ncontrol +[+-]")          # the control is the reference, not a row
            card = (Path(d) / "card.md").read_text()
            self.assertIn("Sealed predictions", card)
            self.assertRegex(card, r"\| selection bias \| ok \|")
            self.assertTrue((Path(d) / "dashboard.html").exists())

    def test_user_csv_without_dates_renamed_control_and_audience(self):
        with tempfile.TemporaryDirectory() as d:
            rs, rows = np.random.default_rng(2), ["person_id,segment,persona,arm,message,converted"]
            segs = {"a": "student on a budget", "b": "busy parent", "c": "retired and careful with money"}
            arms = {"base": "Shop now", "x1": "30% off today", "x2": "Delivered in 15 minutes", "x3": "Free trial",
                    "x4": "Local and fresh"}
            for i in range(900):
                s, m = rs.choice(list(segs)), rs.choice(list(arms))
                p = 0.08 + 0.06 * (m == "x1") * (s == "a") + 0.05 * (m == "x2") * (s == "b")
                rows.append(f"u{i},{s},{segs[s]},{m},{arms[m]},{int(rs.random() < p)}")
            csv = Path(d) / "own.csv"
            csv.write_text("\n".join(rows) + "\n")
            p = run("--data", str(csv), "--control", "base", "--audience", "a,b", "--out", str(Path(d) / "run"),
                    "--B", "50")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            self.assertIn("audience: a, b", p.stdout)
            self.assertIn("LLM calls: 20 (5 no-persona, 0 unparsed", p.stdout)
            for ad in ("x1", "x2", "x3", "x4"):
                self.assertRegex(p.stdout, rf"\n{ad} +[+-]\d\.\d{{3}} \[")
            self.assertTrue((Path(d) / "run" / "card.md").exists())
            bad = Path(d) / "bad.csv"
            bad.write_text("person_id,segment,arm,converted\nu1,a,base,0\n")
            p = run("--data", str(bad), "--out", str(Path(d) / "bad"), timeout=60)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("--data needs columns", p.stderr)

    def test_live_needs_budget_and_key(self):
        with tempfile.TemporaryDirectory() as d:
            p = run("--live", "--out", d, timeout=60)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("--max-usd", p.stderr)
            p = run("--live", "--max-usd", "0.1", "--out", d, timeout=60)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("no API key", p.stderr)
            key = Path(d) / "k.txt"
            key.write_text("sk-test-not-a-real-key-123\n")
            p = run("--live", "--key-file", str(key), "--max-usd", "0", "--out", str(Path(d) / "run"), timeout=60)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("BudgetExceeded", p.stderr)
            self.assertNotIn("sk-test-not-a-real-key", p.stdout + p.stderr)


if __name__ == "__main__":
    unittest.main()
