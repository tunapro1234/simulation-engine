"""docs/CONVENTIONS.md stays true to the signatures: llm placement, batch defaults, the strategy re-export, and
the doc notes on FakeLLM (llm.md) and WTP (games.md)."""
import importlib
import inspect
import re
import sys
import types
import unittest
from pathlib import Path

import simengine.sim as sim
from simengine.llm import FakeLLM
from simengine.methods import LLMDirectAB
from simengine.sim.funnel import funnel
from simengine.sim.interview import interview
from simengine.sim.roundtable import roundtable_strategy

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "CONVENTIONS.md").read_text(encoding="utf-8")

KEYWORD_LLM = [sim.survey, sim.ab_test, sim.ab_many, sim.strategy, sim.roundtable, roundtable_strategy, interview,
               funnel]
FIRST_LLM = [sim.segment_distribution, sim.persona_distribution, sim.predict_people, sim.ask_constrained,
             sim.experiment, sim.ask_panel, sim.ask_many]
# (doc row label, function, parameter, default)
BATCHES = [("sim.survey", sim.survey, "batch", 8), ("sim.survey", sim.survey, "question_group", 6),
           ("sim.ab_test", sim.ab_test, "batch", 10), ("sim.strategy", sim.strategy, "batch", 8),
           ("sim.roundtable", sim.roundtable, "batch", 10),
           ("sim.persona_distribution", sim.persona_distribution, "batch", 12),
           ("sim.ab_many", sim.ab_many, "tests_per_call", 5),
           ("simengine.sim.interview.interview", interview, "batch", 8),
           ("simengine.sim.funnel.funnel", funnel, "batch", 8),
           ("sim.ask_panel", sim.ask_panel, "batch", 10), ("methods.LLMDirectAB", LLMDirectAB, "batch", 15)]


def _doc_row(label: str) -> str:
    m = re.search(r"^\| `" + re.escape(label) + r"` \|.*$", DOC, re.M)
    return m.group(0) if m else ""


class TestLLMPlacement(unittest.TestCase):
    def test_keyword_only(self):
        for f in KEYWORD_LLM:
            p = inspect.signature(f).parameters["llm"]
            self.assertEqual(p.kind, p.KEYWORD_ONLY, f.__name__)
            self.assertIn(f"{f.__name__}`", DOC.split("| first positional")[0])

    def test_first_positional_accepted(self):
        for f in FIRST_LLM:
            first = next(iter(inspect.signature(f).parameters.values()))
            ok = first.name == "llm" or first.kind == first.VAR_POSITIONAL
            self.assertTrue(ok, f"{f.__name__}: first parameter {first}")
            self.assertIn(f"`sim.{f.__name__}`", DOC)

    def test_llm_first_call_runs(self):
        llm = FakeLLM(sim.schema_responder(0))
        qs = [{"id": "q1", "text": "Own a car?", "options": ["Yes", "No"]}]
        r = sim.segment_distribution(llm, {"young": "adults aged 18 to 34"}, qs)
        self.assertIn("young", r["dist"])


class TestBatchDefaults(unittest.TestCase):
    def test_table_matches_signatures(self):
        for label, f, name, default in BATCHES:
            self.assertEqual(inspect.signature(f).parameters[name].default, default, f"{label}.{name}")
            row = _doc_row(label)
            self.assertIn(f"`{name}`", row, label)
            self.assertIn(str(default), row.rsplit("|", 2)[-2], label)


class TestStrategyReexport(unittest.TestCase):
    def test_attribute_is_function_module_via_importlib(self):
        self.assertTrue(callable(sim.strategy) and not isinstance(sim.strategy, types.ModuleType))
        mod = importlib.import_module("simengine.sim.strategy")
        self.assertIsInstance(mod, types.ModuleType)
        self.assertIs(sys.modules["simengine.sim.strategy"], mod)
        self.assertTrue(mod.__doc__)

    def test_variants_alias(self):
        p = inspect.signature(sim.strategy).parameters
        self.assertEqual(p["variants"].kind, p["variants"].KEYWORD_ONLY)


class TestDocNotes(unittest.TestCase):
    def test_no_machine_paths(self):
        for bad in ("/srv/", "/data/", "/root/", "/home/", "/tmp/"):
            self.assertNotIn(bad, DOC)

    def test_unit_formats(self):
        self.assertIn("`<pid>|<qid>`", DOC)
        self.assertIn("`ab|<variant>`", DOC)

    def test_fakellm_row_points_to_responders(self):
        llm_md = (ROOT / "docs" / "modules" / "llm.md").read_text(encoding="utf-8")
        row = next(line for line in llm_md.splitlines() if line.startswith("| `FakeLLM` |"))
        self.assertIn("schema_responder()", row)
        self.assertIn("responder(system, prompt)", row)
        self.assertEqual(FakeLLM().complete("s", "u").text, FakeLLM().complete("s", "u").text)

    def test_games_wtp_disambiguation(self):
        games_md = (ROOT / "docs" / "modules" / "games.md").read_text(encoding="utf-8")
        self.assertIn("willingness-to-pay", games_md)
        self.assertIn("methods.conjoint", games_md)


if __name__ == "__main__":
    unittest.main()
