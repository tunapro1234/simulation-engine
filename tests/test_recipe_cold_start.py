"""examples/recipes/cold_start.py end to end in a subprocess: the offline run (FakeLLM + synthetic catalogue) finishes
with rc=0 in time and writes the card; a --data CSV without launch_date uses the item hash split and predicts the
empty-target rows; missing columns exit; --live refuses to start without a budget and stops at --max-usd before
any network call, never printing the key; the --live path against a stub Anthropic client (no network): quoted or
cut-off replies still parse, a refusal is counted as unparsed, mostly unparseable replies stop the run."""
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "recipes" / "cold_start.py"
# Runs the script with se.llm.Claude given a stub client (argv[1] = reply template, %s = a rating; ~1 in 40
# prompts refuses). Exercises the real Claude backend: request building, cache, ledger, budget, map(return_exceptions).
STUB = """
import runpy, sys, zlib
sys.path.insert(0, sys.argv[2])
import simengine as se
tpl = sys.argv[1]
class Messages:
    def create(self, **p):
        assert p["thinking"] == {"type": "disabled"} and p["max_tokens"] == 120
        h = zlib.crc32(p["messages"][0]["content"].encode())
        if h % 40 == 0:
            return {"content": [], "usage": {"input_tokens": 90, "output_tokens": 1}, "stop_reason": "refusal"}
        r = 3 + h % 200 / 100
        return {"content": [{"type": "text", "text": tpl % ("%.2f" % r)}], "usage": {"input_tokens": 90,
                "output_tokens": 20}, "stop_reason": "end_turn", "model": p["model"]}
class Stub:
    messages = Messages()
Claude = se.llm.Claude
se.llm.Claude = lambda *a, **k: Claude(*a, client=Stub(), **k)
sys.argv = sys.argv[3:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def run(*args, timeout=120):
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout,
                          cwd=ROOT, env=env)


class TestColdStartRecipe(unittest.TestCase):
    def test_offline_run_writes_card(self):
        with tempfile.TemporaryDirectory() as d:
            t0 = time.time()
            p = run("--out", d, "--B", "100", timeout=60)
            self.assertLess(time.time() - t0, 60)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            for s in ("LLM calls: 484", "optimism: raw LLM mean", "Orbit Buds", "QuickPress", "knn", "cat_mean",
                      "stack_free", "shuffled_stack", "stack - stack_blind", "badge:"):
                self.assertIn(s, p.stdout)
            card = (Path(d) / "card.md").read_text()
            self.assertIn("Sealed predictions", card)
            self.assertTrue((Path(d) / "dashboard.html").exists())

    def test_user_csv_without_date(self):
        with tempfile.TemporaryDirectory() as d:
            rows = ["item_id,title,category,description,target"]
            words = ["steel", "plastic", "warranty", "app", "battery", "compact"]
            for i in range(60):                                   # steel helps, plastic hurts
                w = f"{words[i % 6]} {words[(i * 5 + 1) % 6]}"
                rows.append(f"x{i},Thing {i},{'abc'[i % 3]},{w},{3.5 + 0.3 * ('steel' in w) - 0.4 * ('plastic' in w)}")
            rows.append("n1,Brand new,a,steel app,")              # empty target = the launch item
            csv = Path(d) / "items.csv"
            csv.write_text("\n".join(rows) + "\n")
            p = run("--data", str(csv), "--out", str(Path(d) / "run"), "--B", "50")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            self.assertIn("LLM calls: 122", p.stdout)               # 61 items x named + blind
            self.assertIn("Brand new", p.stdout)

    def test_missing_columns_exit(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "bad.csv"
            csv.write_text("item_id,target\nx1,3.5\n")
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

    def stub_live(self, template, d):
        key = Path(d) / "k.txt"
        key.write_text("sk-test-not-a-real-key-456\n")
        cmd = [sys.executable, "-c", STUB, template, str(ROOT), str(SCRIPT), "--live", "--key-file", str(key),
               "--max-usd", "1", "--out", str(Path(d) / "run"), "--B", "50"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=ROOT)
        self.assertNotIn("sk-test-not-a-real-key", p.stdout + p.stderr)
        return p

    def test_live_path_with_stub_client(self):
        with tempfile.TemporaryDirectory() as d:          # quoted number + reply cut mid-reason: still parsed
            p = self.stub_live('{"rating": "%s", "reason": "solid build but the hinge looks we', d)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            bad = int(re.search(r"unparsed or failed: (\d+)", p.stdout).group(1))
            self.assertTrue(0 < bad < 40, bad)                    # the refusals only (~1 in 40 of 484)
            self.assertIn("ledger $", p.stdout)
            self.assertIn("stack_free", p.stdout)

    def test_live_unparseable_replies_stop(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.stub_live("I would say around %s stars.", d)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("fix SYSTEM / KEY", p.stderr)
            self.assertFalse((Path(d) / "run" / "card.md").exists())


if __name__ == "__main__":
    unittest.main()
