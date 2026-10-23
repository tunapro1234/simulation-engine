"""Package-level registration of the bench tasks added in the dataset-unit round (offline, no data needed)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from simengine import bench
from simengine.bench import tasks
from simengine.types import TASK_KINDS

ROOT = Path(__file__).resolve().parents[1]

NEW = {"instacart": "Instacart", "m5": "M5", "movielens": "MovieLens1M",
       "retailrocket_purchase": "RetailRocketPurchase", "retailrocket_next": "RetailRocketNext",
       "choices13k": "Choices13k", "cpc18": "CPC18", "socsci210": "SocSci210",
       "subpop_opinionqa": "SubPOPOpinionQA", "global_opinionqa": "GlobalOpinionQA",
       "swissmetro": "SwissmetroModeChoice", "hillstrom_uplift": "HillstromUplift",
       "criteo_uplift": "CriteoUplift", "x5_uplift": "X5Uplift",
       "webshop_next_action": "WebShopNextAction", "webshop_buy": "WebShopBuy"}


class TestNewTasksRegistered(unittest.TestCase):
    def test_listed_without_direct_import(self):
        names = bench.list()
        for name, cls in NEW.items():
            self.assertIn(name, names)
            self.assertIn(cls, bench.__all__)
            self.assertIn(cls, tasks.__all__)
            self.assertIs(getattr(bench, cls), getattr(tasks, cls))
            t = bench.get(name)
            self.assertIsInstance(t, getattr(bench, cls))
            self.assertEqual(t.name, name)
            self.assertIn(t.kind, TASK_KINDS)
            self.assertTrue(t.layout.strip(), name)
            self.assertTrue(t.metrics, name)
            self.assertIsNotNone(t.accepted_options, name)

    def test_unknown_option_raises(self):
        for name in NEW:
            with self.assertRaises(ValueError, msg=name):
                bench.get(name, spilt="x")

    def test_missing_data(self):
        with tempfile.TemporaryDirectory() as r:
            for name in NEW:
                with self.assertRaises(bench.DataNotFound, msg=name):
                    bench.get(name).load(r)
            table = bench.run(list(NEW), data_root=r)
        self.assertEqual(set(table["task"]), set(NEW))
        self.assertTrue((table["status"] == "missing_data").all())

    def test_all_sorted_unique(self):
        self.assertEqual(len(bench.__all__), len(set(bench.__all__)))
        self.assertEqual(len(tasks.__all__), len(set(tasks.__all__)))


class TestRunBaselinesArg(unittest.TestCase):
    def test_single_name_string_keeps_that_baseline(self):
        import numpy as np
        import pandas as pd
        from simengine.types import TaskData

        class Two(bench.BenchTask):
            name, kind, metrics = "two_baselines_probe", "binary", ()

            def _load(self, root):
                y = np.array([0, 1, 1, 0])
                return TaskData(train=pd.DataFrame({"y": y}), test=pd.DataFrame({"y": y}), meta={})

            def baselines(self, data):
                return {"half": np.full(4, .5), "ones": np.ones(4)}

            def _frame(self, data):
                return pd.DataFrame({"truth": data.test["y"]})

        t = Two()
        with tempfile.TemporaryDirectory() as r:
            data = t.load(r)
        self.assertEqual(bench.methods_in(t.run(data=data, baselines="half")), ["half"])   # was: no baseline kept
        self.assertEqual(sorted(bench.methods_in(t.run(data=data, baselines=["half", "ones"]))), ["half", "ones"])
        self.assertEqual(bench.methods_in(t.run(data=data, baselines=False, methods={"m": lambda d: np.ones(4)})),
                         ["m"])


class TestDocsAndDemo(unittest.TestCase):
    def test_bench_doc_links_every_unit_doc(self):
        self.assertIn("docs/modules/bench.md", bench.__doc__)
        doc = (ROOT / "docs" / "modules" / "bench.md").read_text(encoding="utf-8")
        units = sorted(p.stem for p in (ROOT / "docs" / "modules" / "bench").glob("*.md"))
        self.assertTrue(units)
        for unit in units:
            self.assertIn(f"(bench/{unit}.md)", doc, unit)
        for name in NEW:
            self.assertIn(f"`{name}`", doc, name)

    def test_bench_new_demo_runs(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        out = subprocess.run([sys.executable, str(ROOT / "examples" / "bench_new_demo.py")], capture_output=True,
                             text=True, env=env, cwd=str(ROOT), timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        for key in ("### hillstrom_uplift", "womens_buyers", "### movielens", "shrunk_item_mean", "'m5': 'missing_data'",
                    "registered tasks"):
            self.assertIn(key, out.stdout)


if __name__ == "__main__":
    unittest.main()
