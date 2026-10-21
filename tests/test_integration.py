"""Integration: package re-exports, CLI subcommands, examples and cross-module paths (offline, FakeLLM)."""
import contextlib
import io
import json
import os
import re
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import simengine as se
from simengine import cli

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def run_cli(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        code = cli.main(list(argv))
    return code, buf.getvalue()


class TestPackage(unittest.TestCase):
    def test_version_and_subpackages(self):
        self.assertRegex(se.__version__, r"^\d+\.\d+\.\d+")
        for name in ("llm", "persona", "sim", "methods", "abm", "evidence", "research", "bench"):
            self.assertTrue(hasattr(getattr(se, name), "__all__"), name)
        self.assertIn("llm", dir(se))

    def test_entry_points(self):
        self.assertIs(se.FakeLLM, se.llm.FakeLLM)
        self.assertIs(se.evaluate, se.evidence.evaluate)
        self.assertIs(se.Loop, se.research.Loop)
        self.assertIs(se.from_table, se.persona.from_table)
        self.assertIs(se.survey, se.sim.survey)
        self.assertIsInstance(se.FakeLLM(), se.LLM)
        with self.assertRaises(AttributeError):
            se.no_such_thing  # noqa: B018

    def test_backends_follow_protocol(self):
        self.assertIsInstance(se.llm.ClaudeCode(runner=lambda *a, **k: None), se.LLM)


class TestCrossModule(unittest.TestCase):
    def test_persona_to_sim_to_evidence(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"uid": [f"u{i}" for i in range(30)], "age": rng.integers(18, 70, 30),
                           "gender": rng.choice(["female", "male"], 30), "income_tier": rng.choice(["low", "high"], 30),
                           "visits": rng.poisson(4, 30)})
        df["y"] = rng.integers(1, 6, 30).astype(float)
        panel = se.persona.from_table(df, id_col="uid", segment="income_tier", forbidden=["y"])
        res = se.sim.survey(panel, "I like this", llm=se.FakeLLM(se.sim.schema_responder()), B=50)
        frame = se.sim.survey_frame(res, {"q1": dict(zip(df.uid, df.y))}).dropna(axis=1, how="all")
        self.assertEqual(set(frame["group"]), set(df.uid))
        self.assertEqual(set(frame["segment"]), {"low", "high"})
        rep = se.evidence.evaluate(frame, ours="score", task="continuous", B=50, B_aggregate=20, multiplier=False)
        self.assertIn("manset", rep)
        self.assertIn("Evidence card", se.evidence.card(rep))

    def test_persona_source_stops_on_llm_stop_error(self):
        from simengine.llm import AuthError

        class Broken:
            model = "broken"
            calls = 0

            def complete(self, system, user, **kw):
                Broken.calls += 1
                raise AuthError("no key")      # message has none of the text markers

            def map(self, prompts, **kw):
                raise AuthError("no key")

        src = se.persona.ClaudeSource(Broken(), retries=3)
        ps = se.persona.FakeSource().generate(se.persona.AudienceSpec(target_description="x"), n=4)
        src.ask(ps, "Rate it", {"score": "int 1-10"}, batch=1)
        self.assertIsNotNone(src.stopped)
        self.assertEqual(Broken.calls, 1)      # first failure stops the run, no retries, no further calls

    def test_research_claude_p_proposer_uses_llm_claude_code(self):
        from simengine.research.proposers import claude_code_llm
        self.assertIsInstance(claude_code_llm("claude-haiku-5-5"), se.llm.ClaudeCode)


class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.demo = cli.run_demo(cls.dir / "demo", n=40, B=60)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_demo_outputs(self):
        for k in ("report", "card", "card_json", "dashboard", "frame"):
            self.assertTrue(Path(self.demo["paths"][k]).exists(), k)
        # FakeLLM is noise: the evidence pipeline must not award it a high-confidence badge
        self.assertNotEqual(self.demo["summary"]["badge"], "high confidence")

    def test_evaluate_and_card(self):
        out = self.dir / "ev"
        code, txt = run_cli("evaluate", self.demo["paths"]["frame"], "--ours", "score", "--task", "continuous",
                            "--unlabeled-baselines", "persistence", "--B", "50", "--no-multiplier", "--out", str(out))
        self.assertEqual(code, 0)
        self.assertIn("badge:", txt)
        for f in ("report.json", "card.md", "card.json", "dashboard.html"):
            self.assertTrue((out / f).exists(), f)
        code, txt = run_cli("card", str(out / "report.json"))
        self.assertTrue(txt.startswith("# Evidence card"))
        code, _ = run_cli("card", str(out / "report.json"), "-o", str(self.dir / "c.json"))
        self.assertIsInstance(json.loads((self.dir / "c.json").read_text()), dict)

    def test_bench_list_and_missing_data(self):
        code, txt = run_cli("bench", "list")
        for name in se.bench.list():
            self.assertIn(name, txt)
        code, txt = run_cli("bench", "run", "gss", "upworthy:split=holdout", "--data-root", str(self.dir / "none"),
                            "--B", "10")
        self.assertEqual(code, 0)
        self.assertIn("data not found", txt)
        self.assertEqual(cli._task_spec("opinionqa_groups:split=dev,setting=B"),
                         ("opinionqa_groups", {"split": "dev", "setting": "B"}))

    def test_loop_from_json(self):
        code, txt = run_cli("loop", str(EXAMPLES / "loop.json"), "--workdir", str(self.dir / "loop"),
                            "--max-turns", "4")
        self.assertEqual(code, 0)
        self.assertIn('"stopped"', txt)
        self.assertTrue((self.dir / "loop" / "state.json").exists())
        v = (json.loads((self.dir / "loop" / "state.json").read_text()).get("final") or {}).get("verdict") or {}
        self.assertIs(v.get("higher_is_better"), False)
        if v.get("dev_gain") and v.get("dev_improvement"):          # lower-is-better: improvement = -gain
            self.assertAlmostEqual(v["dev_improvement"]["diff"], -v["dev_gain"]["diff"])
            self.assertIn("positive = better", txt)

    def test_evaluate_prereg_registry_and_seal_in_card(self):
        frame = pd.read_csv(self.demo["paths"]["frame"])
        code, _ = run_cli("evaluate", self.demo["paths"]["frame"], "--ours", "score", "--task", "continuous",
                          "--prereg", str(self.dir / "no-such-registry"), "--out", str(self.dir / "x"))
        self.assertEqual(code, 2)                      # a wrong registry path fails loudly instead of attaching nothing
        reg = self.dir / "reg"
        rec = se.evidence.seal(frame[["group", "pred_score"]], "demo seal", registry=reg)
        out = self.dir / "ev_sealed"
        code, txt = run_cli("evaluate", self.demo["paths"]["frame"], "--ours", "score", "--task", "continuous",
                            "--primary", "pearson", "--unlabeled-baselines", "persistence", "--B", "40",
                            "--no-multiplier", "--prereg", str(reg), "--out", str(out), "--lang", "en")
        self.assertEqual(code, 0)
        self.assertIn("primary metric: pearson", txt)
        card = (out / "card.md").read_text(encoding="utf-8")
        self.assertIn(rec["sha256"], card)
        self.assertIn("Sealed predictions", card)
        self.assertNotIn("Headline (original wording)", card)
        with self.assertWarns(UserWarning):
            se.evidence.evaluate(frame, ours="score", task="continuous", B=20, B_aggregate=10, multiplier=False,
                                 prereg=self.dir / "no-such-registry")

    def test_below_baseline_badge(self):
        from simengine.evidence.api import _badge_english
        self.assertEqual(_badge_english({"rozet": "kararsiz", "fark": {"fark": -0.7, "alt": -0.9, "ust": -0.4}}),
                         "below baseline")
        self.assertEqual(_badge_english({"rozet": "kararsiz", "fark": {"fark": -0.1, "alt": -0.3, "ust": 0.1}}),
                         "inconclusive")
        self.assertEqual(_badge_english({"rozet": "yuksek", "fark": {"fark": 0.1, "alt": 0.05, "ust": 0.2}}),
                         "high confidence")
        s = self.demo["summary"]
        self.assertEqual(s["below_baseline"], s["badge"] == "below baseline")
        self.assertIn("headline_en", s)

    def test_bench_unknown_option_is_an_error(self):
        code, _ = run_cli("bench", "run", "upworthy:spilt=holdout", "--data-root", str(self.dir / "none"))
        self.assertEqual(code, 2)
        with self.assertRaises(ValueError):
            se.bench.get("upworthy", spilt="holdout")
        self.assertIn("split", se.bench.get("upworthy").describe()["accepted_options"])

    def test_python_dash_m_entry_point(self):
        import subprocess
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run([sys.executable, "-m", "simengine", "--version"], cwd=str(ROOT), env=env,
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("simengine", r.stdout + r.stderr)

    def test_space_from_json_ranges(self):
        sp = cli._space_from_json({"params": {"x": {"range": [0.0, 1.0]}, "m": ["a", "b"]},
                                   "defaults": {"x": 0.5, "m": "a"}})
        full, err = sp.validate({"x": 0.3})
        self.assertEqual(err, [])
        self.assertEqual(full["m"], "a")
        self.assertTrue(sp.validate({"x": 2.0})[1])


class TestExamples(unittest.TestCase):
    """The fast examples run end to end (offline)."""

    def _run(self, name, *args):
        old_argv, old_path = sys.argv, list(sys.path)
        sys.argv = [str(EXAMPLES / name), *args]
        sys.path.insert(0, str(EXAMPLES))
        try:
            with contextlib.redirect_stdout(io.StringIO()) as buf, contextlib.redirect_stderr(io.StringIO()):
                runpy.run_path(str(EXAMPLES / name), run_name="__main__")
        finally:
            sys.argv, sys.path[:] = old_argv, old_path
        return buf.getvalue()

    def test_quickstart(self):
        with tempfile.TemporaryDirectory() as d:
            txt = self._run("quickstart.py", d)
            self.assertTrue((Path(d) / "card.md").exists())
        self.assertIn("Evidence card", txt)

    def test_segment_distribution(self):
        self.assertIn("sealed:", self._run("segment_distribution.py"))

    def test_research_loop(self):
        with tempfile.TemporaryDirectory() as d:
            txt = self._run("research_loop.py", d)
            self.assertTrue((Path(d) / "prereg.md.lock.json").exists() or any(Path(d).glob("*.lock.json")))
        self.assertIn("verdict", txt)

    def test_abm_adoption(self):
        self.assertIn("MAE", self._run("abm_adoption.py"))


class TestHygiene(unittest.TestCase):
    SKIP_DIRS = {".git", "__pycache__", "runs", "data", "build", "dist", ".cache", "secrets", ".venv", "venv"}
    SUFFIXES = {".py", ".md", ".json", ".toml", ".txt", ".cfg", ".ini", ".yaml", ".yml", ".html", ".sh"}
    # the one allowed machine path: the skip-guarded data-catalogue default of the data-present tests
    ALLOWED = {("tests/test_bench_data.py", 'os.environ.get("SIMENGINE_DATA") or "' + '/' + 'data/talp/veri"')}

    def _files(self):
        for p in ROOT.rglob("*"):
            rel = p.relative_to(ROOT)
            if p.is_file() and not (set(rel.parts) & self.SKIP_DIRS) and not rel.parts[0].endswith(".egg-info") \
                    and (p.suffix in self.SUFFIXES or p.name in ("LICENSE", ".gitignore")):
                yield rel, p

    def test_no_machine_paths_or_keys_anywhere_in_the_repo(self):
        bad = re.compile(r"((?<![\w.-])/(srv|data|home|root|tmp)/|sk-ant-[A-Za-z0-9])")
        seen = 0
        for rel, p in self._files():
            seen += 1
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if bad.search(line) and not any(rel.as_posix() == f and s in line for f, s in self.ALLOWED):
                    self.fail(f"{rel}:{i}: machine path or key-like string: {line.strip()[:120]}")
        self.assertGreater(seen, 50)


if __name__ == "__main__":
    unittest.main()
