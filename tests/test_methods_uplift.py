"""methods.uplift: segment effects + Neyman SE, Fay-Herriot shrinkage, the split-half dev rig (no unit in both
halves), the heterogeneity test (rejects a heterogeneous effect, keeps a constant one) and T-/S-learners that beat
random targeting on synthetic data with a known effect. All synthetic, no LLM."""
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine.methods as M
from simengine.methods import uplift as UP


def _customers(n=8000, het=True, seed=0, share=0.5):
    """x1 drives the effect when het (+ for x1 > 0, - below); x2 and the segment only the base rate. Additive
    effect 0.06 everywhere when not het (so the risk difference is constant)."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    seg = rng.choice(["a", "b", "c", "d"], n)
    t = (rng.random(n) < share).astype(int)
    base = 0.25 + 0.05 * np.tanh(x2) + 0.03 * (seg == "a")
    eff = 0.15 * np.tanh(2 * x1) if het else np.full(n, 0.06)
    y = (rng.random(n) < np.clip(base + t * eff, 0, 1)).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "seg": seg, "t": t, "y": y, "eff": eff})


class TestSegmentUplift(unittest.TestCase):
    def test_raw_diff_and_neyman_se(self):
        rng = np.random.default_rng(1)
        n = 3000
        df = pd.DataFrame({"g": rng.choice(["p", "q"], n), "t": rng.integers(0, 2, n), "y": rng.normal(size=n)})
        r = M.segment_uplift(df, "t", "y", "g")
        for g in ("p", "q"):
            d = df[df.g == g]
            a, b = d.y[d.t == 1], d.y[d.t == 0]
            self.assertAlmostEqual(r.loc[g, "uplift"], a.mean() - b.mean())
            self.assertAlmostEqual(r.loc[g, "se"], np.sqrt(a.var() / len(a) + b.var() / len(b)))
            self.assertEqual(r.loc[g, "n_treat"], len(a))
        self.assertLess(r.loc["p", "ci_low"], r.loc["p", "uplift"])
        self.assertAlmostEqual(r.attrs["ate"], df.y[df.t == 1].mean() - df.y[df.t == 0].mean())

    def test_binary_small_cell_se_not_zero(self):
        df = pd.DataFrame({"g": ["s"] * 6, "t": [1, 1, 1, 0, 0, 0], "y": [1, 1, 1, 0, 0, 0]})
        r = M.segment_uplift(df, "t", "y", "g")
        self.assertAlmostEqual(r.loc["s", "uplift"], 1.0)
        self.assertGreater(r.loc["s", "se"], 0)   # (k + 1) / (n + 2) proportions in the variance

    def test_missing_arm_and_bad_treatment(self):
        df = pd.DataFrame({"g": ["a", "a", "b", "b"], "t": [1, 0, 1, 1], "y": [1, 0, 1, 0]})
        r = M.segment_uplift(df, "t", "y", "g")
        self.assertTrue(np.isnan(r.loc["b", "uplift"]))
        with self.assertRaises(ValueError):
            M.segment_uplift(df.assign(t=[2, 0, 1, 1]), "t", "y", "g")
        with self.assertRaises(KeyError):
            M.segment_uplift(df, "t", "y", "nope")


class TestShrink(unittest.TestCase):
    def test_tau2_zero_gives_precision_weighted_mean(self):
        rng = np.random.default_rng(2)
        se = rng.uniform(0.5, 2, 30)
        est = pd.Series(1.0 + rng.normal(size=30) * se, index=[f"s{i}" for i in range(30)])
        r = M.shrink_effects(est, se)
        w = 1 / se ** 2
        if r.attrs["tau2"] < 1e-6:   # the usual outcome when there is no between-segment spread
            np.testing.assert_allclose(r.shrunk, np.sum(w * est) / w.sum(), atol=1e-4)
        self.assertEqual(list(r.index), list(est.index))
        self.assertTrue((r.B >= 0).all() and (r.B <= 1).all())

    def test_recovers_tau2_and_beats_raw(self):
        rng = np.random.default_rng(3)
        k, tau = 400, 1.0
        theta = rng.normal(0, tau, k)
        se = rng.uniform(0.5, 1.5, k)
        est = theta + rng.normal(size=k) * se
        r = M.shrink_effects(est, se)
        self.assertAlmostEqual(r.attrs["tau2"], tau ** 2, delta=0.3)
        self.assertLess(np.mean((r.shrunk - theta) ** 2), 0.8 * np.mean((est - theta) ** 2))

    def test_covariate_prior_and_missing_estimate(self):
        rng = np.random.default_rng(4)
        x = np.linspace(-1, 1, 40)
        est = 2 * x + rng.normal(size=40) * 0.3
        est[5] = np.nan
        r = M.shrink_effects(est, np.full(40, 0.3), X=np.c_[np.ones(40), x])
        self.assertAlmostEqual(r.attrs["beta"][1], 2.0, delta=0.3)
        self.assertAlmostEqual(r.shrunk.iloc[5], r.prior.iloc[5])   # no estimate: the prior
        self.assertEqual(r.B.iloc[5], 1.0)


class TestSplitHalf(unittest.TestCase):
    def test_halves_disjoint_and_stratified(self):
        """D11 #3: the first hand version bootstrapped before splitting, so one customer sat in both halves."""
        df = _customers(n=1001, share=0.3).assign(cust=lambda d: np.arange(len(d)) // 1)
        rng = np.random.default_rng(0)
        for boot in (False, True):
            a, b = UP._halves(df["t"].to_numpy(), None, rng, boot)
            self.assertFalse(set(df.cust.to_numpy()[a]) & set(df.cust.to_numpy()[b]), boot)
            ta, tb = df.t.to_numpy()[a], df.t.to_numpy()[b]
            self.assertLessEqual(abs(int(ta.sum()) - int(tb.sum())), 1)
            self.assertLessEqual(abs(int((ta == 0).sum()) - int((tb == 0).sum())), 1)

    def test_units_with_several_rows_stay_together(self):
        df = _customers(n=600)
        df = pd.concat([df.assign(cust=np.arange(600)), df.assign(cust=np.arange(600))], ignore_index=True)
        rng = np.random.default_rng(1)
        a, b = UP._halves(df["t"].to_numpy(), df["cust"].to_numpy(), rng, True)
        self.assertFalse(set(df.cust.to_numpy()[a]) & set(df.cust.to_numpy()[b]))
        with self.assertRaises(ValueError):   # treatment must be constant within a unit
            UP._halves(np.r_[df.t.to_numpy()[:-1], 1 - df.t.to_numpy()[-1]], df.cust.to_numpy(), rng, False)

    def test_frame_and_summary(self):
        df = _customers(n=4000, het=False)
        fr = M.split_half_effects(df, "t", "y", "seg", R=8, seed=0)
        self.assertEqual(set(fr.method), {"raw", "pooled", "shrunk"})
        self.assertEqual(len(fr), 8 * 3 * 4)
        self.assertTrue({"rep", "seg", "pred", "truth", "truth_var", "n_treat", "n_control"} <= set(fr.columns))
        self.assertTrue((fr.truth_var > 0).all())
        np.testing.assert_array_equal(fr, M.split_half_effects(df, "t", "y", "seg", R=8, seed=0))
        s = M.summarize_split_half(fr)
        self.assertEqual(set(s.index), {"raw", "pooled", "shrunk"})
        # constant effect: pooling beats the raw per-segment difference on noise-corrected MSE
        self.assertLess(s.loc["pooled", "mse_d"], s.loc["raw", "mse_d"])

    def test_custom_method(self):
        df = _customers(n=2000)
        fr = M.split_half_effects(df, "t", "y", "seg", R=3, seed=1, boot=False,
                                  methods={"zero": lambda a: pd.Series(0.0, index=["a", "b", "c", "d"])})
        self.assertEqual(set(fr.method), {"zero"})
        self.assertTrue((fr.pred == 0).all())


class TestHeterogeneity(unittest.TestCase):
    def test_rejects_heterogeneous_keeps_constant(self):
        het = M.heterogeneity_test(_customers(het=True), "t", "y", ["x1", "x2", "seg"])
        self.assertLess(het.p, 0.01)
        self.assertIn("x1", set(het.q.by))
        self.assertLess(het.q.set_index("by").loc["x1", "p"], 0.01)
        rej = [M.heterogeneity_test(_customers(het=False, seed=s), "t", "y", ["x1", "x2", "seg"],
                                    scale="additive").p < 0.05 for s in range(10)]
        self.assertLessEqual(sum(rej), 3)
        self.assertIn("p > 0.2", M.heterogeneity_test.__doc__)

    def test_advice_and_df(self):
        r = M.heterogeneity_test(_customers(het=False, seed=1), "t", "y", ["x2", "seg"], seg="seg")
        self.assertEqual(r.df, 1 + 3)            # x2 + three seg dummies
        self.assertIn("seg", set(r.q.by))
        if r.p > 0.2:
            self.assertIn("do not spend time", r.advice)
        r2 = M.heterogeneity_test(_customers(het=True), "t", "y", ["x1"])
        self.assertIn("heterogeneous", r2.advice)
        with self.assertRaises(ValueError):
            M.heterogeneity_test(_customers().assign(y=lambda d: d.y * 2.5), "t", "y", ["x1"], scale="logit")


class TestLearners(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        d = _customers(n=12000)
        self.tr, self.te = d.iloc[:8000], d.iloc[8000:].reset_index(drop=True)

    def _random_uak(self):
        te = self.te
        return np.mean([M.uplift_at_k(te.y, te.t, np.random.default_rng(i).random(len(te)), 0.3) for i in range(50)])

    def test_tlearner_beats_random(self):
        te = self.te
        for model in ("logistic", "gbdt"):
            s = M.TLearner(model=model).fit(self.tr, "t", "y", ["x1", "x2", "seg"]).predict(te)
            self.assertEqual(len(s), len(te))
            self.assertGreater(np.corrcoef(s, te.eff)[0, 1], 0.5, model)
            self.assertGreater(M.uplift_at_k(te.y, te.t, s, 0.3), self._random_uak() + 0.05, model)
            self.assertGreater(M.qini(te.y, te.t, s), 0.1, model)

    def test_slearner_and_estimator_instance(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        te = self.te
        s = M.SLearner().fit(self.tr, "t", "y", ["x1", "x2", "seg"]).predict(te)
        self.assertGreater(np.corrcoef(s, te.eff)[0, 1], 0.5)
        tl = M.TLearner(model=HistGradientBoostingClassifier(max_iter=50, random_state=0))
        p1, p0 = tl.fit(self.tr, "t", "y", ["x1", "x2", "seg"]).predict_arms(te)
        np.testing.assert_allclose(tl.predict(te), p1 - p0)

    def test_default_features_and_unfitted(self):
        with self.assertRaises(RuntimeError):
            M.TLearner().predict(self.te)
        tl = M.TLearner().fit(self.tr.drop(columns="eff"), "t", "y")
        self.assertEqual(sorted(tl.features_), ["seg", "x1", "x2"])
        self.assertEqual(tl.categorical_, ["seg"])


class TestMetricsExported(unittest.TestCase):
    def test_same_objects_as_bench(self):
        from simengine.bench.tasks import uplift as B
        for name in ("qini_coefficient", "auuc", "uplift_at_k", "qini_curve", "uplift_curve", "two_model",
                     "s_learner", "class_transformation"):
            self.assertIs(getattr(B, name), getattr(UP, name), name)
        self.assertIs(M.qini, UP.qini_coefficient)


if __name__ == "__main__":
    unittest.main()
