"""examples/recipes/spread.py end to end in a subprocess: the offline run (FakeLLM + synthetic grid) finishes with
rc=0, prints the cascade-size table, the ladder with the shuffle and no-persona controls and the seeding table, and
writes the card; a --data CSV runs without the synthetic true world; missing columns and --live without a budget or
key stop with a message; --max-usd 0 stops before any network call and never prints the key."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "recipes" / "spread.py"


def run(*args, timeout=120):
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout,
                          cwd=ROOT, env=env)


def cascades_csv(path, side=6, n_casc=20, p=0.35, seed=0):
    """Small user-style CSV: side x side grid, two segments, 2 seeds per cascade, IC spread with hop numbers."""
    rs, n = np.random.default_rng(seed), side * side
    nb = [[j for j in (i - 1 if i % side else -1, i + 1 if (i + 1) % side else -1, i - side, i + side) if 0 <= j < n]
          for i in range(n)]
    rows = ["cascade_id,message,node,segment,persona,neighbours,seed,shared,step"]
    for c in range(n_casc):
        hop = {int(s): 0 for s in rs.choice(n, 2, replace=False)}
        front, h = list(hop), 0
        while front:
            h, new = h + 1, []
            for i in front:
                for j in nb[i]:
                    if j not in hop and rs.random() < p:
                        hop[j] = h
                        new.append(j)
            front = new
        for i in range(n):
            seg = ("young", "old")[i % 2]
            rows.append(f"c{c},notice number {c} about the park,u{i},{seg},a {seg} neighbour,"
                        f"{' '.join(f'u{j}' for j in nb[i])},{int(hop.get(i) == 0)},{int(i in hop)},"
                        f"{hop.get(i, '')}")
    path.write_text("\n".join(rows) + "\n")


class TestSpreadRecipe(unittest.TestCase):
    def test_offline_run_writes_card(self):
        with tempfile.TemporaryDirectory() as d:
            p = run("--out", d, "--B", "100")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            for s in ("LLM calls: 305; failed 0, unparsed 0", "test cascade size", "llm_calibrated",
                      "kontrol_karistir_llm_calibrated", "tahmin_llm_nopersona", "tahmin_ic_segment", "badge:",
                      "seeding k=5", "greedy_model", "true world"):
                self.assertIn(s, p.stdout)
            self.assertIn("Sealed predictions", (Path(d) / "card.md").read_text())
            self.assertTrue((Path(d) / "dashboard.html").exists())

    def test_user_csv(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "cascades.csv"
            cascades_csv(csv)
            p = run("--data", str(csv), "--out", str(Path(d) / "run"), "--B", "50", "--k", "2",
                    "--message", "notice number 99 about the park")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            self.assertIn("LLM calls: 63;", p.stdout)                # (20 messages + campaign) x (2 segments + control)
            self.assertIn("seeding k=2", p.stdout)
            self.assertNotIn("true world", p.stdout)                 # no simulator of the truth for user data

    def test_missing_columns_exit(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "bad.csv"
            csv.write_text("cascade_id,node\nc1,u1\n")
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
            key.write_text("sk-test-not-a-real-key-789\n")
            p = run("--live", "--key-file", str(key), "--max-usd", "0", "--out", str(Path(d) / "run"), timeout=60)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("BudgetExceeded", p.stderr)
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
