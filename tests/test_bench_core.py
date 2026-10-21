"""Registry, base task contract, leaderboard and metric helpers of simengine.bench (offline, synthetic)."""
import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine import bench
from simengine.bench import stats
from simengine.types import TASK_KINDS, TaskData

from tests.test_bench_adapters import _bank, _gss, _upworthy


@bench.register("toy_binary")
class Toy(bench.BenchTask):
    name = "toy_binary"
    kind = "binary"
    data_subdir = None
    metrics = (bench.Metric("acc", True, lambda t, f, c, B, s: stats.bootstrap_mean(
        ((f[c] >= .5) == (f["truth"] == 1)).astype(float), B=B, seed=s)),)

    def _load(self, root):
        y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        return TaskData(train=pd.DataFrame({"y": y}), test=pd.DataFrame({"y": y, "x": y * .8 + .1}), meta={"split": "toy"})

    def baselines(self, data):
        return {"half": np.full(len(data.test), .5)}

    def _frame(self, data):
        return pd.DataFrame({"truth": data.test["y"]})


class TestRegistry(unittest.TestCase):
    def test_list_and_get(self):
        names = bench.list()
        for n in ["opinionqa_groups", "twin2k500", "open_ecommerce", "upworthy", "simbench", "gss", "bank_marketing"]:
            self.assertIn(n, names)
            t = bench.get(n)
            self.assertIn(t.kind, TASK_KINDS)
            self.assertTrue(t.layout.strip())
            self.assertTrue(t.metrics)
        self.assertEqual(bench.get("gss", holdout_panel="panel2006").options["holdout_panel"], "panel2006")
        with self.assertRaises(KeyError):
            bench.get("nope")

    def test_bad_kind_rejected(self):
        with self.assertRaises(ValueError):
            @bench.register("bad")
            class Bad(bench.BenchTask):
                name, kind = "bad", "regression"

    def test_missing_data(self):
        with tempfile.TemporaryDirectory() as r:
            with self.assertRaises(bench.DataNotFound):
                bench.get("gss").load(os.path.join(r, "absent"))
            with self.assertRaises(bench.DataNotFound):
                bench.get("upworthy").load(r)
        old = os.environ.pop("SIMENGINE_DATA", None)
        try:
            with self.assertRaises(bench.DataNotFound):
                bench.get("gss").load()
        finally:
            if old is not None:
                os.environ["SIMENGINE_DATA"] = old


class TestTaskContract(unittest.TestCase):
    def test_method_forms_and_validation(self):
        t = bench.get("toy_binary")
        data = t.load(".")

        class Obj:
            name = "obj"

            def predict(self, data):
                return data.test["x"].to_numpy()

        fr = t.run({"one": lambda d: d.test["x"], "two": lambda task, d: d.test["x"]}, data=data)
        self.assertTrue({"pred_half", "pred_one", "pred_two"} <= set(fr.columns))
        fr2 = t.run([Obj(), "half"], data=data, baselines=False)
        self.assertEqual(set(bench.methods_in(fr2)), {"obj", "half"})
        self.assertEqual(t.score(fr, "one")["value"], 1.0)
        self.assertEqual(t.score(fr, "pred_half")["value"], 0.625)  # p >= .5 -> "yes" on 5 of 8
        with self.assertRaises(ValueError):
            t.frame(data, {"short": [0.1, 0.2]})
        with self.assertRaises(KeyError):
            t.run(["no_such_baseline"], data=data)
        sc = t.score_all(fr)
        self.assertEqual(len(sc), 3)


class TestLeaderboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = cls._tmp.name
        _gss(cls.root)
        _upworthy(cls.root)
        _bank(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_run_and_markdown(self):
        def own(data):
            return data.test.groupby("item")["truth"].transform("mean").to_numpy() * 0 + 2.5

        table, frames = bench.run(["gss", ("upworthy", {"split": "holdout"}), "simbench"],
                                  methods={"gss": {"flat": own}, "upworthy": ["coin"]}, data_root=self.root, B=30,
                                  return_frames=True)
        self.assertEqual(set(frames), {"gss", "upworthy"})
        self.assertEqual(table.loc[table.task == "simbench", "status"].tolist(), ["missing_data"])
        g = table[(table.task == "gss") & (table.method == "flat")]
        self.assertFalse(g["is_baseline"].any())
        self.assertTrue(table[(table.task == "gss") & (table.method == "pca_ridge")]["is_baseline"].all())
        self.assertTrue((table[table.status == "ok"]["lo"] <= table[table.status == "ok"]["hi"] + 1e-12).all())
        md = bench.to_markdown(table)
        self.assertIn("### gss", md)
        self.assertIn("mean_z (higher is better)", md)
        self.assertIn("simbench: data not found", md)
        first = [ln for ln in md.splitlines() if ln.startswith("| pca_ridge") or ln.startswith("| median")]
        self.assertTrue(first)

    def test_primary_only(self):
        t = bench.run("bank_marketing", data_root=self.root, B=20, metrics="primary")
        self.assertTrue((t["metric"] == "auc").all())


class TestStats(unittest.TestCase):
    def test_helpers(self):
        self.assertAlmostEqual(stats.tvd([.5, .5], [1, 0]), .5)
        self.assertAlmostEqual(stats.wasserstein_ordinal(np.array([1, 0, 0.]), np.array([0, 0, 1.]),
                                                         np.array([0, .5, 1])), 1.0)
        self.assertAlmostEqual(stats.auc([0, 0, 1, 1], [.1, .4, .35, .8]), .75)
        self.assertEqual(stats.pair_accuracy([1, 0, 1], [.9, .5, .2]), 0.5)
        self.assertTrue(np.isnan(stats.pearson([1, 1, 1], [1, 2, 3])))
        r = stats.bootstrap_mean([1, 2, 3, 4], B=200)
        self.assertEqual(r["n"], 4)
        self.assertLessEqual(r["lo"], 2.5)
        c = stats.cluster_bootstrap(lambda i: float(np.mean(np.array([1, 1, 0, 0])[i])), ["a", "a", "b", "b"], B=200)
        self.assertEqual(c["n"], 2)
        self.assertEqual(c["value"], .5)


if __name__ == "__main__":
    unittest.main()
