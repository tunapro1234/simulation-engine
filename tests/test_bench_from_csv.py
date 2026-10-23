"""bench.from_csv: a competition-style CSV becomes a bench task (split, encoded target, baselines, evidence metrics
with a group bootstrap CI). Tiny synthetic CSVs written to a temp directory; no real data."""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from simengine import bench
from simengine.bench import CsvTask, from_csv
from simengine.types import Task

B = 50


def _binary(n_groups=40, per=5, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    item = np.tile(np.arange(per), n_groups)
    x = rng.normal(size=len(g))
    p = 1 / (1 + np.exp(-(1.5 * x + (item - 2) * 0.6)))
    y = np.where(rng.random(len(g)) < p, "yes", "no")
    return pd.DataFrame({"row_id": np.arange(len(g)), "person": g, "q": item, "x": x,
                         "c": np.where(x > 0, "hi", "lo"), "label": y})


class FromCsv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "comp.csv")
        _binary().to_csv(self.path, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_protocol_split_and_encoding(self):
        t = from_csv(self.path, target="label", id_col="row_id", group_col="person", item_col="q", test_frac=0.25)
        self.assertIsInstance(t, CsvTask)
        self.assertIsInstance(t, Task)
        self.assertEqual((t.name, t.kind), ("comp", "binary"))
        d = t.load()
        # group split: no person on both sides; 10 of 40 groups in test
        self.assertFalse(set(d.train["person"]) & set(d.test["person"]))
        self.assertEqual(d.test["person"].nunique(), 10)
        self.assertEqual(set(d.train["label"].unique()), {0, 1})
        self.assertEqual(d.meta["positive"], "yes")
        self.assertEqual(d.meta["features"], ["x", "c"])
        self.assertEqual(t.describe()["metrics"][0], "accuracy")
        self.assertEqual(from_csv(_binary(), target="label").describe()["name"], "csv")
        # same seed -> same split; other seed -> different
        self.assertEqual(list(t.load().test["row_id"]), list(d.test["row_id"]))
        d2 = from_csv(self.path, target="label", group_col="person", test_frac=0.25, seed=1).load()
        self.assertNotEqual(sorted(d2.test["person"].unique()), sorted(d.test["person"].unique()))

    def test_run_score_baselines(self):
        t = from_csv(self.path, target="label", id_col="row_id", group_col="person", item_col="q")
        frame = t.run({"oracle": lambda d: 1 / (1 + np.exp(-1.5 * d.test["x"].to_numpy()))})
        for c in ("truth", "group", "item", "unit", "pred_constant", "pred_majority", "pred_item_marginal",
                  "pred_oracle"):
            self.assertIn(c, frame.columns)
        self.assertEqual(t.primary.name, "accuracy")
        names = [m.name for m in t.metrics]
        for n in ("auc", "brier", "logloss", "within_item_corr"):
            self.assertIn(n, names)
        s = t.score(frame, "oracle", "auc", B=B)
        self.assertGreater(s["value"], 0.7)
        self.assertLessEqual(s["lo"], s["value"])
        self.assertGreaterEqual(s["hi"], s["value"])
        self.assertEqual(s["n"], frame["group"].nunique())       # bootstrap unit = group
        self.assertTrue(np.isnan(t.score(frame, "constant", "auc", B=B)["value"]) or
                        t.score(frame, "constant", "auc", B=B)["value"] == 0.5)
        # item marginal knows the item effect, the constant does not
        bm = t.score(frame, "item_marginal", "brier", B=B)["value"]
        bc = t.score(frame, "constant", "brier", B=B)["value"]
        self.assertLess(bm, bc)
        table = bench.run([t], methods={"oracle": lambda d: d.test["x"].gt(0).astype(float)}, B=B,
                          metrics="primary")
        self.assertTrue((table["status"] == "ok").all())
        self.assertEqual(set(table["method"]), {"constant", "majority", "item_marginal", "oracle"})

    def test_metric_choice_english_or_internal(self):
        t = from_csv(self.path, target="label", metric="roc_auc")
        self.assertEqual(t.primary.name, "auc")
        self.assertEqual(from_csv(self.path, target="label", metric="dogruluk").primary.name, "accuracy")
        with self.assertRaises(ValueError):
            from_csv(self.path, target="label", metric="rmse")

    def test_split_col_and_unlabeled_rows(self):
        df = _binary()
        df["fold"] = np.where(df["person"] < 30, "train", "test")
        df.loc[[0, 1], "label"] = np.nan
        t = from_csv(df, target="label", split_col="fold", features=["x"])
        d = t.load()
        self.assertEqual(len(d.test), 50)
        self.assertEqual(len(d.train), 148)
        self.assertEqual(d.meta["n_unlabeled_dropped"], 2)
        self.assertNotIn("c", d.train.columns)
        self.assertNotIn("fold", d.meta["features"])
        df["fold"] = "train"
        with self.assertRaises(ValueError):
            from_csv(df, target="label", split_col="fold").load()

    def test_multiclass(self):
        rng = np.random.default_rng(3)
        n = 300
        x = rng.normal(size=n)
        lab = np.where(x < -0.5, "a", np.where(x < 0.5, "b", "c"))
        df = pd.DataFrame({"x": x, "y": lab})
        t = from_csv(df, target="y", task="multiclass", test_frac=0.3)
        d = t.load()
        self.assertEqual(d.meta["classes"], ["a", "b", "c"])
        frame = t.run({"codes": lambda d: np.digitize(d.test["x"], [-0.5, 0.5]),
                       "labels": lambda d: np.array(["a", "b", "c"])[np.digitize(d.test["x"], [-0.5, 0.5])]})
        self.assertEqual(len(frame["pred_constant"].iloc[0]), 3)
        self.assertEqual(t.score(frame, "codes", B=B)["value"], 1.0)
        self.assertEqual(t.score(frame, "labels", B=B)["value"], 1.0)
        self.assertLess(t.score(frame, "majority", B=B)["value"], 0.6)
        self.assertTrue(np.isfinite(t.score(frame, "constant", "logloss", B=B)["value"]))
        self.assertEqual(t.score(frame, "codes", B=B)["n"], len(d.test))  # no group_col: row bootstrap

    def test_continuous(self):
        rng = np.random.default_rng(4)
        n = 200
        item = rng.integers(0, 5, n)
        y = item * 2.0 + rng.normal(size=n)
        df = pd.DataFrame({"item": item, "y": y})
        t = from_csv(df, target="y", task="continuous", item_col="item", metric="mae")
        frame = t.run()
        self.assertEqual(set(c for c in frame.columns if c.startswith("pred_")),
                         {"pred_constant", "pred_median", "pred_item_marginal"})
        mi = t.score(frame, "item_marginal", B=B)["value"]
        mc = t.score(frame, "constant", B=B)["value"]
        self.assertLess(mi, mc)
        self.assertFalse(t.primary.higher_is_better)

    def test_errors(self):
        with self.assertRaises(ValueError):
            from_csv(self.path, target="label", task="distribution")
        with self.assertRaises(ValueError):
            from_csv(self.path, target="nope").load()
        with self.assertRaises(bench.DataNotFound):
            from_csv(os.path.join(self.tmp.name, "missing.csv"), target="label").load()
        with self.assertRaises(ValueError):
            from_csv(self.path, target="label", test_frac=1.0)
        df = _binary()
        df["label"] = np.random.default_rng(0).choice(["a", "b", "c"], len(df))
        with self.assertRaises(ValueError):
            from_csv(df, target="label").load()  # three classes for a binary task
        t = from_csv(self.path, target="label")
        with self.assertRaises(ValueError):
            t.run({"short": lambda d: np.zeros(3)})

    def test_silent_mislabel_guards(self):
        # numeric classes 1..5: labels that never hit 5 used to be read as codes (shifted by one, accuracy 0.2)
        rng = np.random.default_rng(5)
        x = rng.normal(size=200)
        df = pd.DataFrame({"x": x, "y": np.digitize(x, [-1, -0.3, 0.3, 1]) + 1})
        t = from_csv(df, target="y", task="multiclass")
        lab = lambda d: np.digitize(d.test["x"], [-1, -0.3, 0.3, 1]) + 1   # noqa: E731
        with self.assertRaises(ValueError):
            t.run({"amb": lambda d: np.clip(lab(d), 1, 4)})
        frame = t.run({"labels": lab, "codes": lambda d: lab(d) - 1,
                       "str_labels": lambda d: lab(d).astype(str),
                       "series": lambda d: pd.Series(lab(d) - 1, index=np.arange(len(d.test)) + 1000)})
        for m in ("labels", "codes", "str_labels", "series"):
            self.assertEqual(t.score(frame, m, B=B)["value"], 1.0, m)
        with self.assertRaises(ValueError):  # sklearn predict_proba without the class unseen in train
            t.run({"short": lambda d: np.full((len(d.test), 4), 0.25)})
        with self.assertRaises(ValueError):
            from_csv(df.assign(y="a"), target="y", task="multiclass").load()
        # binary: one value, or a positive that is not a value, would make every row one class
        b = _binary()
        with self.assertRaises(ValueError):
            from_csv(b.assign(label="no"), target="label").load()
        with self.assertRaises(ValueError):
            from_csv(b, target="label", positive="Yes").load()
        self.assertEqual(from_csv(b.assign(label="no"), target="label", positive="no").load().meta["positive"], "no")
        # split_col: a fold number or an inverted train flag used to pass silently
        with self.assertRaises(ValueError):
            from_csv(b.assign(fold=b["person"] % 5), target="label", split_col="fold").load()
        with self.assertRaises(ValueError):
            from_csv(b.assign(is_train=(b["person"] < 30).astype(int)), target="label", split_col="is_train").load()
        d = from_csv(b.assign(is_test=(b["person"] >= 30).astype(float)), target="label", split_col="is_test").load()
        self.assertEqual(len(d.test), 50)
        with self.assertRaises(ValueError):  # signed metric: no direction to rank by
            from_csv(b, target="label", metric="mean_bias")

    def test_nan_predictions_and_single_cluster(self):
        t = from_csv(self.path, target="label", group_col="person")
        frame = t.run({"few": lambda d: np.where(np.arange(len(d.test)) < 5, 1.0, np.nan)})
        s = t.score(frame, "few", B=B)
        self.assertEqual(s["n_rows"], 5)
        self.assertAlmostEqual(s["coverage"], 5 / len(frame))
        self.assertEqual(t.score(frame, "constant", B=B)["coverage"], 1.0)
        one = from_csv(pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0, 1, 0]}), target="y")
        s1 = one.score(one.run(), "constant", B=B)
        self.assertEqual(s1["n"], 1)
        self.assertTrue(np.isnan(s1["lo"]) and np.isnan(s1["hi"]))

    def test_relative_path_under_data_root(self):
        t = from_csv("comp.csv", target="label")
        self.assertEqual(len(t.load(self.tmp.name).test), 40)
        frame = t.run(data_root=self.tmp.name)
        self.assertIn("pred_majority", frame.columns)


if __name__ == "__main__":
    unittest.main()
