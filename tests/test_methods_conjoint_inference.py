"""MNL pricing inference: Hessian standard errors, WTP with delta / bootstrap CIs, share simulation, elasticities,
and the long-format validation errors."""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.methods.conjoint import MNL, elasticity, evaluate_mnl, null_scores, simulate_shares, wtp

BETA = {"price": -0.03, "quality": 0.9, "warranty": 0.4}       # per raw unit (price in currency units)
ASC = np.array([0.0, 0.3, -0.2])


def choices(persons=300, tasks=8, seed=1):
    rs = np.random.default_rng(seed)
    rows, tid = [], 0
    for p in range(persons):
        for _ in range(tasks):
            price, qual, war = rs.uniform(50, 150, 3), rs.uniform(0, 3, 3), rs.integers(0, 3, 3)
            v = BETA["price"] * price + BETA["quality"] * qual + BETA["warranty"] * war + ASC
            pr = np.exp(v - v.max())
            pick = rs.choice(3, p=pr / pr.sum())
            for j in range(3):
                rows.append({"person": p, "task": tid, "alternative": j + 1, "price": price[j],
                             "quality": qual[j], "warranty": war[j], "chosen": int(j == pick)})
            tid += 1
    return pd.DataFrame(rows)


ATTRS = ["price", "quality", "warranty"]
MARKET = pd.DataFrame({"alternative": [1, 2, 3], "price": [90.0, 110.0, 100.0], "quality": [1.0, 2.0, 1.5],
                       "warranty": [0, 2, 1]})


class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = choices()
        cls.m = MNL(ATTRS, scale={"price": 100}).fit(cls.df)

    def test_se_covers_truth(self):
        m = self.m
        self.assertTrue(m.converged_)
        self.assertEqual(list(m.cov_.index), list(m.coef_.index))
        truth = {"price": BETA["price"] * 100, "quality": BETA["quality"], "warranty": BETA["warranty"],
                 "ASC_2": 0.3, "ASC_3": -0.2}
        for k, b in truth.items():
            self.assertTrue(np.isfinite(m.se_[k]) and m.se_[k] > 0, k)
            self.assertLess(abs(m.coef_[k] - b), 3.5 * m.se_[k], k)

    def test_hessian_matches_numeric(self):
        # numeric second derivative of the total log-likelihood around the optimum
        m, b0 = self.m, self.m.coef_.to_numpy()
        X = m._design(m._fit_data)
        g = pd.factorize(m._fit_data["task"])[0]
        y, av = m._fit_data["chosen"].to_numpy(float), m._fit_data["available"].to_numpy(float)

        def grad(b):
            p = np.exp(MNL._logp(b, X, g, av))
            return X.T @ (y - p)

        h = 1e-5
        H = np.column_stack([(grad(b0 + h * e) - grad(b0 - h * e)) / (2 * h) for e in np.eye(len(b0))])
        np.testing.assert_allclose(np.linalg.inv(-H), m.cov_.to_numpy(), rtol=1e-4, atol=1e-8)

    def test_wtp_delta(self):
        w = wtp(self.m, price="price")
        self.assertEqual(list(w.index), ["quality", "warranty"])
        self.assertTrue(np.isfinite(w[["se", "lo", "hi"]].to_numpy()).all())
        for a in ("quality", "warranty"):
            true = -BETA[a] / BETA["price"]
            self.assertAlmostEqual(w.loc[a, "wtp"], -self.m.coef_[a] / (self.m.coef_["price"] / 100))
            self.assertTrue(w.loc[a, "lo"] < true < w.loc[a, "hi"], (a, true, w.loc[a].to_dict()))
        asc = wtp(self.m, attrs=["ASC_2"])
        self.assertAlmostEqual(asc.loc["ASC_2", "wtp"], -self.m.coef_["ASC_2"] * 100 / self.m.coef_["price"])

    def test_wtp_bootstrap(self):
        small = self.df[self.df["person"] < 80]
        m = MNL(ATTRS, scale={"price": 100}).fit(small)
        d = wtp(m)
        b = wtp(m, ci="bootstrap", cluster="person", B=40, seed=3)
        self.assertTrue(np.isfinite(b.to_numpy()).all())
        self.assertTrue((b["lo"] < b["wtp"]).all() and (b["wtp"] < b["hi"]).all())
        # same order of magnitude as the delta-method se
        ratio = (b["se"] / d["se"]).to_numpy()
        self.assertTrue(((ratio > 0.4) & (ratio < 2.5)).all(), ratio)
        np.testing.assert_allclose(wtp(m, ci="bootstrap", B=5, seed=3)["lo"],
                                   wtp(m, ci="bootstrap", B=5, seed=3)["lo"])
        with self.assertRaises(ValueError):
            wtp(m, ci="jackknife")
        with self.assertRaises(ValueError):
            wtp(m, price="cost")

    def test_shares(self):
        s = simulate_shares(self.m, MARKET)
        self.assertAlmostEqual(s.sum(), 1.0)
        v = self.m.coef_
        u = (v["price"] * MARKET["price"] / 100 + v["quality"] * MARKET["quality"] + v["warranty"] * MARKET["warranty"]
             + np.array([0, v["ASC_2"], v["ASC_3"]]))
        np.testing.assert_allclose(s.to_numpy(), np.exp(u) / np.exp(u).sum())
        with self.assertRaises(ValueError):
            simulate_shares(self.m, MARKET.drop(columns="alternative"))
        with self.assertRaises(ValueError):
            simulate_shares(self.m, MARKET.assign(alternative=[1, 2, 9]))

    def test_elasticity_analytic_matches_numeric(self):
        e = elasticity(self.m, MARKET, price_grid=[0.999, 1.001])
        bp = self.m.coef_["price"] / 100
        s, p = e["share"].to_numpy(), MARKET["price"].to_numpy()
        np.testing.assert_allclose(e["own"].to_numpy(), bp * p * (1 - s))
        arc = e["arc"].groupby("alternative")["arc"].mean()
        np.testing.assert_allclose(arc.to_numpy(), e["own"].to_numpy(), rtol=1e-3)
        np.testing.assert_allclose(np.diag(e["cross"].to_numpy()), e["own"].to_numpy())
        # cross: numeric d log s_i / d log p_j for j = 3
        h = 1e-4
        up, dn = MARKET.copy(), MARKET.copy()
        up.loc[2, "price"] *= 1 + h
        dn.loc[2, "price"] *= 1 - h
        num = (np.log(simulate_shares(self.m, up)) - np.log(simulate_shares(self.m, dn))) / (2 * h)
        np.testing.assert_allclose(e["cross"].iloc[:, 2].to_numpy(), num.to_numpy(), rtol=1e-4)
        self.assertEqual(len(elasticity(self.m, MARKET)["arc"]), 12)  # default grid: 4 factors x 3 alternatives
        ints = elasticity(self.m, MARKET.astype({"price": int}))
        np.testing.assert_allclose(ints["own"].to_numpy(), elasticity(self.m, MARKET)["own"].to_numpy())


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.df = choices(persons=20, tasks=4)

    def test_no_attribute_columns(self):
        with self.assertRaisesRegex(ValueError, "no attribute columns"):
            MNL().fit(self.df)
        MNL(asc_only=True).fit(self.df)                       # the explicit baseline still works

    def test_missing_attribute_column(self):
        with self.assertRaisesRegex(ValueError, r"missing attribute column\(s\) \['colour'\]"):
            MNL(["price", "colour"]).fit(self.df)

    def test_missing_required_column(self):
        with self.assertRaisesRegex(ValueError, "missing required column"):
            MNL(ATTRS).fit(self.df.drop(columns="chosen"))

    def test_two_chosen_in_a_task(self):
        d = self.df.copy()
        d.loc[(d.task == 0), "chosen"] = 1
        with self.assertRaisesRegex(ValueError, "more than one chosen"):
            MNL(ATTRS).fit(d)

    def test_task_ids_reused_across_persons(self):
        d = self.df.assign(task=self.df["task"] % 4)
        with self.assertRaisesRegex(ValueError, "unique across persons"):
            MNL(ATTRS).fit(d)

    def test_no_chosen_task_warns_and_drops(self):
        d = self.df.copy()
        d.loc[d.task == 0, "chosen"] = 0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = MNL(ATTRS).fit(d)
        self.assertTrue(any("no chosen" in str(x.message) for x in w))
        self.assertEqual(m.n_tasks_, d.task.nunique() - 1)

    def test_non_binary_chosen(self):
        with self.assertRaisesRegex(ValueError, "0/1"):
            MNL(ATTRS).fit(self.df.assign(chosen=self.df["chosen"] * 2))


class TestEvaluateKeys(unittest.TestCase):
    def test_aliases_and_precision(self):
        df = choices(persons=60, tasks=4)
        out = evaluate_mnl(df, ATTRS, scale={"price": 100})
        for arm in ("mnl", "asc_only", "null"):
            self.assertEqual(out[arm]["acc"], out[arm]["accuracy"])
            self.assertEqual(out[arm]["ll_per_task"], out[arm]["log_likelihood"])
        q = out["mnl"]["coef"]["quality"]
        self.assertNotEqual(q, round(q, 3))                   # full precision
        self.assertIn("se", out["mnl"])
        self.assertEqual(set(null_scores(df)), {"acc", "ll_per_task", "accuracy", "log_likelihood"})


if __name__ == "__main__":
    unittest.main()
