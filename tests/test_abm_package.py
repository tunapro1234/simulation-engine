"""simengine.abm package surface: the grid / spatial units are re-exported as the same objects, clashing names stay
in their submodules, and examples/abm_new_demo.py runs offline."""
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path

import simengine.abm as abm
import simengine.abm.adoption
import simengine.abm.grid as G
import simengine.abm.mobility as M
import simengine.abm.placement as pl
import simengine.abm.schelling as S
import simengine.abm.spatial_sir as ep
import simengine.abm.sugarscape as SG

ROOT = Path(__file__).resolve().parents[1]

UNITS = {
    G: ["Grid", "Agents", "RunLog", "EMPTY", "NEIGHBOURHOODS", "random_lattice", "random_order_step",
        "synchronous_step", "make_rng", "spawn_rngs", "morans_i", "moran_permutation", "dissimilarity",
        "same_type_share", "gini"],
    S: ["Schelling", "SchellingRun", "tipping_curve", "unlike_pairs", "segregation_coefficient", "ZhangSwap",
        "ZhangRun", "zhang_payoff", "LLMMover", "parse_decision", "threshold_responder"],
    SG: ["Sugarscape", "SugarscapeConfig", "SugarscapeRun", "LLMPolicy", "sugar_capacity", "spice_capacity",
         "load_capacity", "lorenz", "gini_trajectory"],
    M: ["to_cell", "from_cell", "geobleu", "dtw", "user_scores", "slot_mode", "global_slot_mode", "global_slot_table",
        "MarkovMobility", "EPRMobility", "mobility_baselines"],
    pl: ["traj_columns", "CellGrid", "Exposure", "exposures", "home_cells", "coverage", "reach_frequency",
         "Placement", "greedy_placement", "online_bound", "top_visited", "random_cells", "moore_share",
         "toy_trajectories"],
    ep: ["EpiParams", "EpiRun", "SpatialSEIR", "seir_ode", "final_size", "r0_final_size", "r0_growth",
         "r0_index_case", "ode_agreement", "cells_from_xy", "trajectories_from_records", "load_yjmob100k",
         "EpiAgent", "epi_personas", "isolation_prompt", "parse_isolation", "GenerativeEpidemic", "GenerativeRun"],
}


class TestExports(unittest.TestCase):
    def test_same_objects_and_listed(self):
        for mod, names in UNITS.items():
            for n in names:
                self.assertIn(n, abm.__all__, n)
                self.assertIs(getattr(abm, n), getattr(mod, n), f"{mod.__name__}.{n}")

    def test_all_unique_and_resolvable(self):
        self.assertEqual(len(abm.__all__), len(set(abm.__all__)))
        for n in abm.__all__:
            self.assertTrue(hasattr(abm, n), n)

    def test_submodules_not_shadowed(self):
        # placement.adoption must not replace the adoption-ABM submodule; grid.run / sugarscape.gini stay local
        self.assertIsInstance(abm.adoption, types.ModuleType)
        self.assertIs(abm.adoption, simengine.abm.adoption)
        for name in ("grid", "schelling", "sugarscape", "mobility", "placement", "spatial_sir"):
            self.assertIsInstance(getattr(abm, name), types.ModuleType, name)
        for n in ("run", "holdout", "AdoptionRun", "clean", "rule_responder"):
            self.assertNotIn(n, abm.__all__)
        self.assertIs(abm.gini, G.gini)

    def test_docstring_points_to_doc(self):
        self.assertIn("docs/modules/abm.md", abm.__doc__)
        doc = (ROOT / "docs" / "modules" / "abm.md").read_text(encoding="utf-8")
        for unit in ("grid_core", "schelling", "sugarscape", "mobility", "placement", "epidemic"):
            self.assertIn(f"abm/{unit}.md", doc, unit)


class TestDemo(unittest.TestCase):
    def test_abm_new_demo_runs(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        out = subprocess.run([sys.executable, str(ROOT / "examples" / "abm_new_demo.py")], capture_output=True,
                             text=True, env=env, cwd=str(ROOT), timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        for key in ("schelling threshold", "sugarscape gini", "mobility slot_mode", "holdout lazy_greedy",
                    "seir radius None", "generative base", "invalid 0", "done"):
            self.assertIn(key, out.stdout)


if __name__ == "__main__":
    unittest.main()
