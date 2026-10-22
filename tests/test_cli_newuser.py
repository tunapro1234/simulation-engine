"""CLI from a new user's seat: one-line errors with exit 2 (missing file, unknown task, unknown --ours / --raw /
--methods column), SIMENGINE_DEBUG=1 traceback, help strings and examples, demo / evaluate --lang en headline."""
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from simengine import cli

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def help_text(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), swallow_exit():
        cli.main([*argv, "--help"])
    return out.getvalue()


@contextlib.contextmanager
def swallow_exit():
    try:
        yield
    except SystemExit:
        pass


def frame_csv(d: Path, n=60, seed=0) -> Path:
    r = np.random.default_rng(seed)
    t = r.normal(size=n)
    df = pd.DataFrame({"group": [f"g{i % 6}" for i in range(n)], "item": "q1", "truth": t,
                       "pred_a": t + r.normal(0, 0.5, n), "pred_b": r.normal(size=n)})
    p = d / "frame.csv"
    df.to_csv(p, index=False)
    return p


class TestCleanErrors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def one_line(self, err):
        lines = [ln for ln in err.splitlines() if ln.startswith("simengine: error:")]
        self.assertEqual(len(lines), 1, err)
        self.assertNotIn("Traceback", err)
        return lines[0]

    def test_missing_files_exit_2(self):
        for argv in (["evaluate", str(self.d / "nope.csv"), "--ours", "x"],
                     ["card", str(self.d / "nope.json")]):
            code, _, err = run_cli(*argv)
            self.assertEqual(code, 2, argv)
            self.assertIn("nope", self.one_line(err))

    def test_unknown_bench_task_exit_2(self):
        code, _, err = run_cli("bench", "describe", "no_such_task")
        self.assertEqual(code, 2)
        self.assertIn("no_such_task", self.one_line(err))

    def test_unknown_ours_lists_columns(self):
        p = frame_csv(self.d)
        code, out, err = run_cli("evaluate", str(p), "--ours", "nonexistent", "--out", str(self.d / "ev"))
        self.assertEqual(code, 2)
        line = self.one_line(err)
        self.assertIn("no column pred_nonexistent; available: pred_a, pred_b", line)
        self.assertFalse((self.d / "ev").exists())

    def test_unknown_raw_and_methods(self):
        p = frame_csv(self.d)
        code, _, err = run_cli("evaluate", str(p), "--ours", "a", "--raw", "zz")
        self.assertEqual(code, 2)
        self.assertIn("pred_zz", self.one_line(err))
        code, _, err = run_cli("evaluate", str(p), "--ours", "a", "--methods", "a,yy")
        self.assertEqual(code, 2)
        self.assertIn("pred_yy", self.one_line(err))

    def test_debug_env_reraises(self):
        with mock.patch.dict(os.environ, {"SIMENGINE_DEBUG": "1"}), self.assertRaises(FileNotFoundError):
            run_cli("card", str(self.d / "nope.json"))

    def test_subprocess_one_line_no_traceback(self):
        env = {k: v for k, v in os.environ.items() if k != "SIMENGINE_DEBUG"}
        r = subprocess.run([sys.executable, "-m", "simengine.cli", "evaluate", str(self.d / "nope.csv")],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 2)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1, r.stderr)
        self.assertTrue(r.stderr.startswith("simengine: error: no such file"), r.stderr)


class TestHelp(unittest.TestCase):
    def test_epilog_examples(self):
        h = help_text()
        for ex in ("simengine bench list", "simengine bench describe <task>",
                   "simengine bench run <task> --out x.csv", "simengine evaluate frame.csv --ours x --lang en"):
            self.assertIn(ex, h)

    def test_flags_have_help(self):
        p = cli.build_parser()
        subs = {}

        def walk(parser, path):
            for act in parser._actions:
                if hasattr(act, "choices") and isinstance(act.choices, dict):
                    for name, sp in act.choices.items():
                        subs[path + (name,)] = sp
                        walk(sp, path + (name,))
        walk(p, ())
        want = {("evaluate",): ["table", "title", "dataset", "B", "B_aggregate", "seed", "lang"],
                ("card",): ["report", "output", "lang"], ("seal",): ["description"],
                ("demo",): ["out", "n", "seed", "B", "lang"], ("loop",): ["workdir"],
                ("bench", "run"): ["B", "seed", "primary_only"]}
        for path, dests in want.items():
            acts = {a.dest: a for a in subs[path]._actions}
            for dst in dests:
                self.assertTrue(acts[dst].help, f"{' '.join(path)} {dst} has no help")
        for name in ("list", "describe", "run"):
            self.assertTrue(subs[("bench", name)].description, name)
        lang = {a.dest: a for a in subs[("evaluate",)]._actions}["lang"].help
        self.assertIn("mixed", lang)
        prim = {a.dest: a for a in subs[("evaluate",)]._actions}["primary"].help
        for m in ("auc", "pearson", "wmape", "wasserstein"):
            self.assertIn(m, prim)

    def test_demo_lang_default_mixed(self):
        self.assertEqual(cli.build_parser().parse_args(["demo"]).lang, "mixed")


class TestEnglishHeadline(unittest.TestCase):
    def test_headline_translates_internal_columns(self):
        s = {"headline": "x 0,5", "headline_en": "pearson: ours (tahmin_score) 0.1 vs (taban_rastgele) 0.2"}
        self.assertEqual(cli._headline(s, "mixed"), "x 0,5")
        en = cli._headline(s, "en")
        self.assertIn("pred_score", en)
        self.assertIn("baseline_random", en)
        self.assertNotIn("taban", en)

    def test_demo_lang_en(self):
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = run_cli("demo", "--lang", "en", "--out", str(Path(d) / "demo"), "--n", "40",
                                   "--B", "30")
            self.assertEqual(code, 0)
            headline = out.splitlines()[1]
            self.assertTrue(headline.strip(), out)
            self.assertIsNone(re.search(r"\d,\d", out), out)
            self.assertNotIn("taban", out)
            self.assertNotIn("tahmin_", headline)

    def test_evaluate_lang_en_prints_english_headline(self):
        with tempfile.TemporaryDirectory() as d:
            p = frame_csv(Path(d))
            code, out, _ = run_cli("evaluate", str(p), "--ours", "a", "--task", "continuous", "--primary",
                                   "pearson", "--B", "40", "--B-aggregate", "20", "--no-multiplier", "--lang", "en",
                                   "--out", str(Path(d) / "ev"))
            self.assertEqual(code, 0)
            headline = out.splitlines()[0]
            self.assertTrue(headline.startswith("pearson: ours (pred_a)"), headline)
            self.assertIsNone(re.search(r"\d,\d", headline), headline)
            self.assertNotIn("taban", headline)


if __name__ == "__main__":
    unittest.main()
