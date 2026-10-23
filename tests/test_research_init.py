"""simengine.research package exports: the search and method modules are re-exported lazily, every name in
__all__ resolves to the object of its home module, a bare import does not load scipy / scikit-learn, and
examples/research_new_demo.py runs."""
import importlib
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path

import simengine.research as research

ROOT = Path(__file__).resolve().parents[1]


class TestResearchExports(unittest.TestCase):
    def test_all_names_resolve_and_are_unique(self):
        self.assertEqual(len(research.__all__), len(set(research.__all__)))
        for name in research.__all__:
            self.assertTrue(hasattr(research, name), name)

    def test_lazy_names_are_the_module_objects(self):
        for mod, names in research._LAZY_BY_MODULE.items():
            m = importlib.import_module(f"simengine.research.{mod}")
            for name in names:
                self.assertIs(getattr(research, name), getattr(m, name), f"{mod}.{name}")

    def test_submodules_stay_modules(self):
        for mod in research.SUBMODULES:
            self.assertIsInstance(getattr(research, mod), types.ModuleType, mod)
        import simengine.research.hyperband as hb
        self.assertIsInstance(hb, types.ModuleType)
        self.assertTrue(callable(hb.hyperband))

    def test_unknown_name_raises(self):
        with self.assertRaises(AttributeError):
            research.no_such_thing  # noqa: B018

    def test_star_import(self):
        ns: dict = {}
        exec("from simengine.research import *", ns)
        self.assertIs(ns["GPProposer"], research.bayesopt.GPProposer)
        self.assertIs(ns["asha"], research.hyperband.asha)

    def test_bare_import_is_light(self):
        code = ("import sys, simengine.research as r; "
                "print(int('scipy' in sys.modules), int('sklearn' in sys.modules), int('simengine.research.bayesopt' "
                "in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.split(), ["0", "0", "0"])


class TestNewDemo(unittest.TestCase):
    def test_example_runs(self):
        out = subprocess.run([sys.executable, str(ROOT / "examples" / "research_new_demo.py")], cwd=ROOT,
                             capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        for tag in ("bandits", "bayesopt", "hyperband", "tree_search", "prompt_evolution", "hypotheses"):
            self.assertIn(f"[{tag}]", out.stdout)


if __name__ == "__main__":
    unittest.main()
