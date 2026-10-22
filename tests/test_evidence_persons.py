"""simengine.evidence.evaluate_distribution_persons: segment x item distributions rebuilt from person rows, person-
cluster paired bootstrap, bias-corrected intervals, several segment schemes. Synthetic people only."""
import unittest

import numpy as np
import pandas as pd

import simengine.evidence as ev
from simengine.evidence.persons import DIST_METRICS, evaluate_distribution_persons

OPTS = ["a", "b", "c"]
TRUE = {"y": [0.6, 0.3, 0.1], "m": [0.3, 0.4, 0.3], "o": [0.1, 0.3, 0.6]}


def _people(P=300, seed=1, p_age=(0.5, 0.4, 0.1)):
    rng = np.random.default_rng(seed)
    age = rng.choice(["y", "m", "o"], P, p=list(p_age))
    df = pd.DataFrame({"unit": [f"u{i}" for i in range(P)], "age": age, "inc": rng.choice(["lo", "hi"], P)})
    df["q1"] = [rng.choice(OPTS, p=TRUE[a]) for a in age]
    df["q2"] = [rng.choice(OPTS, p=TRUE[a][::-1]) for a in age]
    df["good_q1"] = [TRUE[a] for a in age]
    df["good_q2"] = [TRUE[a][::-1] for a in age]
    df["flat_q1"] = df["flat_q2"] = [[1 / 3] * 3] * P
    return df


GOOD = {"q1": "good_q1", "q2": "good_q2"}
FLAT = {"q1": "flat_q1", "q2": "flat_q2"}


class TestDistributionPersons(unittest.TestCase):
    def test_point_matches_direct_computation(self):
        df = _people()
        r = evaluate_distribution_persons(df, ["age"], ["q1"], {"good": GOOD}, B=50)
        vals = []
        for g, sub in df.groupby("age"):
            emp = sub["q1"].value_counts(normalize=True).reindex(OPTS, fill_value=0).to_numpy()
            vals.append(1 - 0.5 * np.abs(emp - np.array(TRUE[g])).sum())
        self.assertAlmostEqual(r["arms"]["good"]["point"], float(np.mean(vals)))
        self.assertEqual(r["n_cells"], 3)
        self.assertEqual(r["n_persons"], 300)
        self.assertEqual(r["metric"], "1-tvd")

    def test_paired_diff_and_schemes(self):
        df = _people()
        r = evaluate_distribution_persons(df, ["age", "inc"], ["q1", "q2"], {"good": GOOD, "flat": FLAT}, B=300)
        self.assertEqual(r["schemes"], ["age", "inc"])
        self.assertEqual(r["n_cells"], (3 + 2) * 2)
        self.assertEqual(set(r["arms"]["good"]["by_scheme"]), {"age", "inc"})
        d = r["diffs"][0]
        self.assertEqual((d["a"], d["b"]), ("good", "flat"))
        self.assertGreater(d["ci"][0], 0)
        self.assertLess(d["p_two_sided"], 0.05)
        # an arm against itself: paired, so the difference has zero width
        r2 = evaluate_distribution_persons(df, ["age"], ["q1"], {"x": GOOD, "y": GOOD}, B=100)
        self.assertEqual(r2["diffs"][0]["ci"], [0.0, 0.0])

    def test_fixed_cells_equal_person_level_constant(self):
        df = _people()
        fixed = {("age", g, i): [1 / 3] * 3 for g in "ymo" for i in ("q1", "q2")}
        r = evaluate_distribution_persons(df, ["age"], ["q1", "q2"], {"flat": FLAT, "fixed": fixed}, B=100)
        self.assertAlmostEqual(r["arms"]["flat"]["point"], r["arms"]["fixed"]["point"])
        self.assertTrue(np.allclose(r["arms"]["flat"]["ci"], r["arms"]["fixed"]["ci"]))
        tab = pd.DataFrame([{"segment": g, "item": i, "pred_m": [1 / 3] * 3} for g in "ymo" for i in ("q1", "q2")])
        r3 = evaluate_distribution_persons(df, ["age"], ["q1", "q2"], {"fixed": tab}, B=100)
        self.assertAlmostEqual(r3["arms"]["fixed"]["point"], r["arms"]["fixed"]["point"])

    def test_bias_correction_small_segments(self):
        """The true generating distribution scores 1 in the population; small segments pull the sample 1-TVD down
        and the bias-corrected point and interval move back toward 1."""
        df = _people(P=90, seed=5, p_age=(0.4, 0.4, 0.2))
        r = evaluate_distribution_persons(df, ["age"], ["q1", "q2"], {"good": GOOD}, B=500, seed=1)
        a = r["arms"]["good"]
        self.assertLess(a["bias"], 0)
        self.assertGreater(a["point_bc"], a["point"])
        self.assertLess(abs(1 - a["point_bc"]), abs(1 - a["point"]))
        self.assertLess(a["ci"][1], a["point"] + 0.02)                # percentile interval sits low
        self.assertGreater(a["ci_bc"][1], a["ci"][1])
        self.assertTrue(np.allclose(a["ci_bc"], np.array(a["ci"]) - a["bias"]))

    def test_cluster_by_person(self):
        df = _people(P=150, seed=2)
        dup = pd.concat([df, df], ignore_index=True)                  # each person twice
        one = evaluate_distribution_persons(df, ["age"], ["q1"], {"good": GOOD}, B=400, seed=3)
        clu = evaluate_distribution_persons(dup, ["age"], ["q1"], {"good": GOOD}, B=400, seed=3)
        flat = evaluate_distribution_persons(dup, ["age"], ["q1"], {"good": GOOD}, B=400, seed=3, id_col=None)
        width = lambda r: r["arms"]["good"]["ci"][1] - r["arms"]["good"]["ci"][0]
        self.assertEqual(clu["n_persons"], 150)
        self.assertAlmostEqual(one["arms"]["good"]["point"], clu["arms"]["good"]["point"])
        self.assertLess(width(flat), 0.85 * width(clu))
        self.assertAlmostEqual(width(one), width(clu), delta=0.35 * width(one))

    def test_hold_truth(self):
        df = _people()
        fixed = {("age", g, "q1"): [1 / 3] * 3 for g in "ymo"}
        r = evaluate_distribution_persons(df, ["age"], ["q1"], {"good": GOOD, "fixed": fixed}, B=50,
                                          hold_truth=True)
        self.assertEqual(r["arms"]["fixed"]["ci"][0], r["arms"]["fixed"]["ci"][1])
        self.assertIn("fixed", r["note"])

    def test_metrics_missing_and_min_n(self):
        df = _people()
        df.loc[:9, "q1"] = np.nan
        r = evaluate_distribution_persons(df, {"age group": "age"}, ["q1"], {"good": GOOD}, B=50, metric="tvd",
                                          min_n=40)
        self.assertFalse(r["higher_is_better"])
        self.assertTrue(all(c["n"] >= 40 for c in r["cells"]))
        self.assertEqual(r["schemes"], ["age group"])
        self.assertIn("1-w1", DIST_METRICS)
        r2 = evaluate_distribution_persons(df, ["age"], ["q1"], {"good": GOOD}, B=20,
                                           metric=lambda t, p: float(np.dot(t, p)))
        self.assertTrue(np.isfinite(r2["arms"]["good"]["point"]))

    def test_errors(self):
        df = _people(P=50)
        with self.assertRaises(ValueError):
            evaluate_distribution_persons(df, ["nope"], ["q1"], {"good": GOOD})
        with self.assertRaises(ValueError):
            evaluate_distribution_persons(df, ["age"], ["q1"], {"good": GOOD}, metric="kl")
        with self.assertRaises(ValueError):
            evaluate_distribution_persons(df, ["age"], ["q1"], {"fixed": {("age", "y", "q1"): [1, 0, 0]}})
        with self.assertRaises(ValueError):
            evaluate_distribution_persons(df, ["age"], ["q1"], {})

    def test_exported(self):
        self.assertIs(ev.evaluate_distribution_persons, evaluate_distribution_persons)


if __name__ == "__main__":
    unittest.main()
