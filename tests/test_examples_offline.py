"""Top-level examples, started the way a reader starts them: `sys.executable examples/<name>.py` from a temporary
working directory with PYTHONPATH and SIMENGINE_DATA unset. Each script must find the package on its own (repo-root
sys.path fallback) and run its offline path (FakeLLM / synthetic data). Always sys.executable, never `python3`: a
bare `python3` on PATH may be a different interpreter without numpy.

    python -m unittest tests.test_examples_offline

Heavy scripts run with their smallest arguments (sugarscape_measure --quick, tree_search_bank --synthetic, a few
seeds for the benches); every case finishes in well under a minute.
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"


def run_example(name, *args, timeout=240):
    """Run examples/<name> from a fresh temp cwd; returns (CompletedProcess, cwd). The cwd is left for inspection
    and removed with the TemporaryDirectory of the caller."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "SIMENGINE_DATA", "ANTHROPIC_API_KEY")}
    env.setdefault("OMP_NUM_THREADS", "1")
    env["MPLBACKEND"] = "Agg"
    tmp = tempfile.TemporaryDirectory(prefix="simengine-ex-")
    args = [a.replace("{tmp}", tmp.name) for a in args]
    out = subprocess.run([sys.executable, str(EX / name), *args], capture_output=True, text=True, timeout=timeout,
                         cwd=tmp.name, env=env)
    return out, tmp


class TestExamplesOffline(unittest.TestCase):
    # (script, args, strings that must be in stdout)
    CASES = [
        ("ab_test.py", (), ["panel winner:"]),
        ("abm_adoption.py", (), ["MAE abm_analogy", "MAE bass_control"]),
        ("abm_new_demo.py", (), ["schelling threshold", "sugarscape gini", "done"]),
        ("auto_social_science_measure.py", (), ['"recovery"', '"conversation_fake"']),
        ("bayesopt_bench.py", ("--seeds", "2", "--loop-seeds", "1", "--calls", "20", "--turns", "4", "--jobs", "1",
                               "--out", "{tmp}/bo.json"), ["full/branin/", "loop/branin/"]),
        ("bench_new_demo.py", (), ["### hillstrom_uplift", "### movielens"]),
        ("conjoint_pricing.py", (), ["part-worths", "WTP"]),
        ("econ_games.py", (), ["persona methods in the zoo", "invalid: 0"]),
        ("funnel.py", (), ["variant A: conversion", "variant B: conversion"]),
        ("hyperband_vs_halving.py", ("--seeds", "3", "--out", "{tmp}/hb.json"),
         ["## sample fidelity", "| hyperband |"]),
        ("quickstart.py", ("{tmp}/qs",), ["# Evidence card: quickstart"]),
        ("research_loop.py", ("{tmp}/loop",), ["verdict:", "journal:"]),
        ("research_new_demo.py", (), ["[bandits]", "[hypotheses]", "done in"]),
        ("segment_distribution.py", (), ["calls: 48"]),
        ("sugarscape_measure.py", ("--quick",), ["smoke run", "[A]", "[B]", "[C]", "[D]", "wall"]),
        ("tree_search_bank.py", ("--synthetic", "--budget", "4", "--seeds", "0", "--out", "{tmp}/tsb"),
         ["synthetic", "loop-random", "baseline constant"]),
    ]

    def test_every_listed_example_is_covered(self):
        listed = {c[0] for c in self.CASES}
        # persona_inject_compare.py belongs to a pre-registered ablation and is run by its own protocol
        have = {p.name for p in EX.glob("*.py")} - {"persona_inject_compare.py"}
        self.assertEqual(have - listed, set(), "new top-level example without an offline smoke case")

    def test_examples_run_offline_without_pythonpath(self):
        with ThreadPoolExecutor(4) as ex:                    # subprocesses: 4 at a time keeps the wall time down
            runs = list(ex.map(lambda c: run_example(c[0], *c[1]), self.CASES))
        for (name, args, needles), (out, tmp) in zip(self.CASES, runs):
            with self.subTest(example=name):
                with tmp:
                    self.assertEqual(out.returncode, 0, f"{name}\n{out.stderr[-3000:]}")
                    for s in needles:
                        self.assertIn(s, out.stdout, name)

    def test_sugarscape_quick_is_labelled_and_fast(self):
        out, tmp = run_example("sugarscape_measure.py", "--quick", timeout=120)
        with tmp:
            self.assertEqual(out.returncode, 0, out.stderr[-3000:])
            self.assertIn("smoke run", out.stdout)
            self.assertIn("not the docs numbers", out.stdout)
            wall = float(out.stdout.rsplit("wall", 1)[1].split()[0])
            self.assertLess(wall, 30)

    def test_tree_search_bank_writes_only_under_out(self):
        out, tmp = run_example("tree_search_bank.py", "--synthetic", "--budget", "3", "--seeds", "1", "--out",
                               "{tmp}/tsb")
        with tmp:
            self.assertEqual(out.returncode, 0, out.stderr[-3000:])
            self.assertTrue((Path(tmp.name) / "tsb" / "summary.tsv").exists())
            self.assertEqual(sorted(p.name for p in Path(tmp.name).iterdir()), ["tsb"])
            for m in ("aide", "mcts", "random", "loop-random"):
                self.assertIn(m, out.stdout)

    def test_fallback_line_precedes_first_simengine_import(self):
        # the subprocess test above passes for the wrong reason where simengine is pip-installed; check the source too
        line = "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))"
        for name, _, _ in self.CASES:
            src = (EX / name).read_text(encoding="utf-8").splitlines()
            first = next(i for i, l in enumerate(src) if l.lstrip().startswith(("import simengine", "from simengine")))
            with self.subTest(example=name):
                self.assertTrue(any(line in l for l in src[:first]), name)

    def test_tree_search_bank_keeps_real_and_synthetic_runs_apart(self):
        # fallback without --out: its own default dir, a marker and a data column, never runs/tree_search_bank
        out, tmp = run_example("tree_search_bank.py", "--budget", "2", "--seeds", "0")
        with tmp:
            self.assertEqual(out.returncode, 0, out.stderr[-3000:])
            runs = Path(tmp.name) / "runs"
            self.assertEqual(sorted(p.name for p in runs.iterdir()), ["tree_search_bank-synthetic"])
            self.assertTrue((runs / "tree_search_bank-synthetic" / "SYNTHETIC").exists())
            summary = pd.read_csv(runs / "tree_search_bank-synthetic" / "summary.tsv", sep="\t")
            self.assertEqual(set(summary["data"]), {"synthetic"})
            # a "real" run (a bank-layout file given as --data-root) refuses the synthetic --out and the reverse
            spec = importlib.util.spec_from_file_location("tsb_example", EX / "tree_search_bank.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            root = Path(tmp.name) / "data"
            mod.write_synthetic_bank(root / "bank-marketing")
            real, _t = run_example("tree_search_bank.py", "--data-root", str(root), "--budget", "2", "--seeds", "0",
                                   "--out", str(runs / "tree_search_bank-synthetic"))
            _t.cleanup()
            self.assertNotEqual(real.returncode, 0)
            self.assertIn("synthetic smoke run", real.stderr)
            real, _t = run_example("tree_search_bank.py", "--data-root", str(root), "--budget", "2", "--seeds", "0",
                                   "--out", str(Path(tmp.name) / "real"))
            _t.cleanup()
            self.assertEqual(real.returncode, 0, real.stderr[-3000:])
            self.assertNotIn("data", pd.read_csv(Path(tmp.name) / "real" / "summary.tsv", sep="\t").columns)
            syn, _t = run_example("tree_search_bank.py", "--synthetic", "--budget", "2", "--seeds", "0",
                                  "--out", str(Path(tmp.name) / "real"))
            _t.cleanup()
            self.assertNotEqual(syn.returncode, 0)
            self.assertIn("real data", syn.stderr)


if __name__ == "__main__":
    unittest.main()
