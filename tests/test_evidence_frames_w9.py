"""simengine.evidence.frames: n-weighted national distribution (TALP D6 #3a) and the long (task, alternative)
choice table -> multiclass frame (D8 #5). Synthetic frames only."""
import json
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine.evidence as ev


def _cells(seed=0, items=6, segs=4):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(items):
        k = 3 + i % 2
        for s in range(segs):
            t = rng.dirichlet(np.ones(k) * 3)
            rows.append({"item": f"q{i}", "segment": f"s{s}", "n": int(rng.integers(20, 400)), "truth": t.tolist(),
                         "pred_llm": (0.6 * t + 0.4 / k).tolist(), "pred_flat": np.full(k, 1 / k)})
    return pd.DataFrame(rows)


def _value(rep, arm):
    r = rep["sonuclar"]
    key = next(k for k in r if k in ("pred_" + arm, arm, "tahmin_" + arm))
    return r[key]["metrikler"][rep["birincil"]]["deger"]


class PooledDistributionTest(unittest.TestCase):
    def test_n_weighted_not_averaged(self):
        f = _cells()
        nat = ev.pooled_distribution(f, weight_col="n", by="item")
        self.assertEqual(len(nat), 6)
        g = f[f.item == "q0"]
        w = g.n.to_numpy(float)
        want = (w[:, None] * np.vstack(g.truth)).sum(0) / w.sum()
        np.testing.assert_allclose(nat.set_index("item").loc["q0", "truth"], want)
        mean = np.vstack(g.truth).mean(0)                                    # equal weights differ
        self.assertGreater(np.abs(mean - want).max(), 1e-3)
        self.assertEqual(nat.set_index("item").loc["q0", "n"], w.sum())
        self.assertEqual(set(nat.n_cells), {4})
        for c in ("truth", "pred_llm", "pred_flat"):
            self.assertTrue(all(np.isclose(sum(v), 1) for v in nat[c]))
        self.assertNotIn("segment", nat.columns)
        rep = _quiet(ev.evaluate, nat, ours="llm", task="distribution", primary="one_minus_tvd", B=30,
                     multiplier=False)
        self.assertEqual(rep["birincil"], "bir_eksi_tvd")
        tvd = np.mean([0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum() for a, b in zip(nat.truth, nat.pred_llm)])
        self.assertAlmostEqual(_value(rep, "llm"), 1 - tvd, places=6)

    def test_json_counts_split_and_aliases(self):
        f = _cells(items=2, segs=3)
        f["truth"] = [json.dumps((np.asarray(v) * 100).tolist()) for v in f.truth]   # counts as JSON: normalised
        f["split"] = "test"
        f = f.rename(columns={"item": "madde"})
        nat = ev.pooled_distribution(f, weight_col="n", by="item")
        self.assertEqual(list(nat.columns[:2]), ["madde", "split"])
        self.assertTrue(all(np.isclose(sum(v), 1) for v in nat.truth))

    def test_validation(self):
        f = _cells(items=2, segs=3)
        bad = f.copy()
        bad.loc[0, "n"] = -1
        with self.assertRaisesRegex(ValueError, ">= 0"):
            ev.pooled_distribution(bad)
        bad = f.copy()
        bad.at[1, "pred_llm"] = [0.5, 0.5]                                    # item q0 has 3 options
        with self.assertRaisesRegex(ValueError, "do not align"):
            ev.pooled_distribution(bad)
        bad = f.copy()
        bad.loc[bad.item == "q1", "n"] = 0
        with self.assertRaisesRegex(ValueError, "sum to 0"):
            ev.pooled_distribution(bad)
        bad = f.copy()
        bad.at[2, "truth"] = None
        with self.assertRaisesRegex(ValueError, "not a distribution"):
            ev.pooled_distribution(bad)
        with self.assertRaisesRegex(ValueError, "pred_"):
            ev.pooled_distribution(f[["item", "n", "truth"]])
        with self.assertRaisesRegex(ValueError, "not in the frame"):
            ev.pooled_distribution(f, weight_col="w")


def _long(seed=0, tasks=60):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(tasks):
        alts = ["car", "bus", "train", "bike"] if t % 3 else ["car", "bus", "walk"]   # 'walk' never chosen
        u = rng.normal(size=len(alts)) + np.array([1.0 if a == "car" else 0 for a in alts])
        p = np.exp(u) / np.exp(u).sum()
        pick = rng.choice(len(alts), p=p)
        if alts[pick] == "walk":
            pick = 0
        for j, a in enumerate(alts):
            rows.append({"task": t, "person": t // 3, "alt": a, "chosen": int(j == pick), "pred_mnl": p[j],
                         "u_llm": 0.5 * u[j], "cost": rng.uniform(1, 9)})
    return pd.DataFrame(rows)


class ChoiceFrameTest(unittest.TestCase):
    def test_one_row_per_task_and_logloss(self):
        lg = _long()
        cf = ev.choice_frame(lg, "task", "alt", "chosen", {"mnl": "pred_mnl"})
        self.assertEqual(len(cf), 60)
        self.assertEqual(cf.attrs["classes"], ["bike", "bus", "car", "train", "walk"])
        self.assertIn("person", cf.columns)                                    # constant within a task: carried
        self.assertNotIn("cost", cf.columns)
        v = cf.set_index("task").loc[0, "pred_mnl"]
        self.assertEqual(v[0], 0.0)                                            # bike not offered in task 0
        self.assertAlmostEqual(v.sum(), 1.0)
        want = -np.mean(np.log(lg.loc[lg.chosen == 1, "pred_mnl"]))
        rep = _quiet(ev.evaluate, cf, ours="mnl", task="multiclass", primary="logloss", classes=cf.attrs["classes"],
                     B=30, multiplier=False)
        self.assertEqual(rep["birincil"], "logloss")
        self.assertAlmostEqual(_value(rep, "mnl"), want, places=6)

    def test_utilities_softmax_and_names(self):
        lg = _long()
        cf = ev.choice_frame(lg, "task", "alt", "chosen", ["pred_mnl", "u_llm"], scores="utility")
        self.assertEqual([c for c in cf.columns if c.startswith("pred_")], ["pred_mnl", "pred_u_llm"])
        g = lg[lg.task == 1]
        e = np.exp(g.u_llm - g.u_llm.max())
        got = cf.set_index("task").loc[1, "pred_u_llm"]
        idx = [cf.attrs["classes"].index(a) for a in g.alt]
        np.testing.assert_allclose(got[idx], e / e.sum())
        self.assertEqual(cf.set_index("task").loc[1, "truth"], g.loc[g.chosen == 1, "alt"].item())

    def test_validation(self):
        lg = _long(tasks=6)
        bad = lg.copy()
        bad.loc[bad.task == 2, "chosen"] = 0
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ev.choice_frame(bad, "task", "alt", "chosen", "pred_mnl")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ev.choice_frame(pd.concat([lg, lg.iloc[:1]]), "task", "alt", "chosen", "pred_mnl")
        bad = lg.copy()
        bad.loc[0, "pred_mnl"] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ev.choice_frame(bad, "task", "alt", "chosen", "pred_mnl")
        bad = lg.copy()
        bad.loc[0, "pred_mnl"] = -0.1
        with self.assertRaisesRegex(ValueError, ">= 0"):
            ev.choice_frame(bad, "task", "alt", "chosen", "pred_mnl")
        with self.assertRaisesRegex(ValueError, "missing"):
            ev.choice_frame(lg, "task", "alt", "chosen", "pred_x")
        with self.assertRaisesRegex(ValueError, "0/1"):
            ev.choice_frame(lg.assign(chosen=2), "task", "alt", "chosen", "pred_mnl")
        with self.assertRaises(ValueError):
            ev.choice_frame(lg, "task", "alt", "chosen", "pred_mnl", scores="logit")
        cf = ev.choice_frame(lg.assign(chosen=lg.chosen.astype(bool)), "task", "alt", "chosen", "pred_mnl")
        self.assertEqual(len(cf), 6)

    def test_exported(self):
        for name in ("pooled_distribution", "choice_frame"):
            self.assertIn(name, ev.__all__)


def _quiet(f, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **kw)


if __name__ == "__main__":
    unittest.main()
