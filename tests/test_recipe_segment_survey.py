"""examples/recipes/segment_survey.py end to end in a subprocess: offline default run (FakeLLM + synthetic survey)
writes the card and prints the ladder, W1 and PPI++ blocks; --data with the documented columns; a split column that
puts one respondent on both sides is refused; --live without --max-usd refuses; --live with a zero budget stops
before any call and never prints the key; --live without any key exits with a message, not a traceback."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "recipes" / "segment_survey.py"


def run(*args, env=None):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=120, env=env)


class TestSegmentSurveyRecipe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_offline_default(self):
        r = run("--out", str(self.d / "run"), "--B", "100")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        for s in ("LLM calls: 60 + 15 control", "llm_prior", "train_segment", "W1 by arm", "llm_dist - llm_nosegment",
                  "llm_prior - llm_prior_nosegment",
                  "badge:", "written:"):
            self.assertIn(s, r.stdout)
        self.assertIn("PPI++", r.stdout)                 # either the estimate block or the "skipped" line
        self.assertTrue((self.d / "run" / "card.md").exists())
        self.assertTrue((self.d / "run" / ".onkayit").exists())     # predictions sealed before the truth

    def test_user_csv(self):
        rows = ["respondent_id,segment,segment_desc,question_id,question_text,options,answer"]
        for i in range(60):
            seg = "a" if i % 2 else "b"
            for q in ("q1", "q2"):
                ans = ["Yes", "No", "Maybe"][(i * (1 + (q == "q2")) + (seg == "a")) % 3]
                rows.append(f"r{i},{seg},people in group {seg},{q},Do you agree with {q}?,Yes|No|Maybe,{ans}")
        (self.d / "s.csv").write_text("\n".join(rows) + "\n")
        r = run("--data", str(self.d / "s.csv"), "--out", str(self.d / "run"), "--B", "50")
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("written:", r.stdout)

    def test_user_csv_missing_column(self):
        (self.d / "bad.csv").write_text("respondent_id,segment,answer\nr1,a,Yes\n")
        r = run("--data", str(self.d / "bad.csv"), "--out", str(self.d / "run"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("missing columns", r.stderr)

    def test_user_csv_split_by_row_refused(self):
        rows = ["respondent_id,segment,segment_desc,question_id,question_text,options,answer,split",
                "r1,a,group a,q1,Agree?,Yes|No,Yes,train", "r1,a,group a,q2,Agree?,Yes|No,No,holdout",
                "r2,b,group b,q1,Agree?,Yes|No,No,holdout"]
        (self.d / "rows.csv").write_text("\n".join(rows) + "\n")
        r = run("--data", str(self.d / "rows.csv"), "--out", str(self.d / "run"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("both splits", r.stderr)

    def test_live_needs_budget(self):
        r = run("--live", "--out", str(self.d / "run"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--max-usd", r.stderr)

    def test_live_zero_budget_stops_without_call(self):
        key = self.d / "k.txt"
        key.write_text("test-key-value-123\n")
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        r = run("--live", "--max-usd", "0", "--key-file", str(key), "--out", str(self.d / "run"), env=env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("stopped by the budget", r.stderr)
        self.assertNotIn("test-key-value-123", r.stdout + r.stderr)

    def test_live_without_key_exits_cleanly(self):
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "SIMENGINE_KEY_FILE")}
        r = run("--live", "--max-usd", "1", "--out", str(self.d / "run"), env=env)   # key lookup fails before any call
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("AuthError", r.stderr)
        self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
