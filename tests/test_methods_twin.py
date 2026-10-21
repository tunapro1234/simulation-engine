"""TwinStack (person-level twin) and pooled logistic on a synthetic latent-factor panel."""
import unittest

import numpy as np
import pandas as pd

from simengine.methods import TwinStack, base_predict, binned_median, column_correlation, pooled_logistic_oof


def panel(n=260, n_feat=16, n_tgt=4, seed=0):
    rs = np.random.default_rng(seed)
    F = rs.normal(size=(n, 2))
    ids = pd.Index([f"p{i}" for i in range(n)])
    feats = pd.DataFrame(F @ rs.normal(size=(2, n_feat)) + rs.normal(0, 0.7, (n, n_feat)), index=ids,
                         columns=[f"f{j}" for j in range(n_feat)])
    feats = feats.mask(rs.random(feats.shape) < 0.1)        # missing answers
    lat = F @ rs.normal(size=(2, n_tgt)) + rs.normal(0, 0.6, (n, n_tgt))
    T = pd.DataFrame(np.clip(np.round(3 + lat), 1, 5), index=ids, columns=[f"t{j}" for j in range(n_tgt)])
    T["nom"] = np.where(F[:, 0] > 0, 2.0, 7.0)
    return feats, T, lat


class TestTwin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.T, cls.lat = panel()
        cls.test = cls.T.index[-60:]
        cls.num = [c for c in cls.T.columns if c != "nom"]

    def test_stack_ml_donor(self):
        m = TwinStack(seed=0, min_stack=0).fit(self.X, self.T, nominal=["nom"])
        P, O = m.predict(self.test)
        truth = self.T.loc[self.test]
        self.assertGreater(column_correlation(truth[self.num], P[self.num]), 0.4)
        self.assertEqual(m.stacked_, self.num)
        for c in self.num:                                    # donor sample = real answers
            self.assertTrue(set(O[c].unique()) <= set(self.T[c].unique()))
            self.assertTrue(((P[c] >= 1) & (P[c] <= 5)).all())
        self.assertTrue(set(O["nom"].unique()) <= {2.0, 7.0})
        self.assertEqual(P["nom"].nunique(), 1)               # nominal point = mode

    def test_input_order_does_not_change_predictions(self):
        P1, _ = TwinStack(seed=0, min_stack=0).fit(self.X, self.T, nominal=["nom"]).predict(self.test)
        rs = np.random.default_rng(7)
        X2 = self.X.iloc[rs.permutation(len(self.X)), rs.permutation(self.X.shape[1])]
        T2 = self.T.iloc[rs.permutation(len(self.T))]
        P2, _ = TwinStack(seed=0, min_stack=0).fit(X2, T2, nominal=["nom"]).predict(self.test)
        np.testing.assert_allclose(P1[self.num].to_numpy(float), P2.loc[P1.index, self.num].to_numpy(float),
                                   atol=1e-9)

    def test_residual_binned_and_llm(self):
        rs = np.random.default_rng(1)
        llm = pd.DataFrame(np.clip(3 + self.lat + rs.normal(0, 0.3, self.lat.shape), 1, 5), index=self.T.index,
                           columns=self.num)
        ml = TwinStack(min_stack=0).fit(self.X, self.T[self.num])
        hy = TwinStack(min_stack=0, sample="residual", binned=True).fit(self.X, self.T[self.num], llm=llm)
        P0, _ = ml.predict(self.test)
        P1, O1 = hy.predict(self.test)
        truth = self.T.loc[self.test, self.num]
        self.assertGreater(column_correlation(truth, P1), column_correlation(truth, P0))
        for c in self.num:
            self.assertTrue(set(O1[c].unique()) <= set(self.T[c].unique()))
        # fewer labels than min_stack with an LLM present -> calibrated LLM fallback
        fb = TwinStack(min_stack=250).fit(self.X, self.T[self.num], llm=llm)
        P2, _ = fb.predict(self.test)
        self.assertEqual(sorted(fb.llm_fallback_), sorted(self.num))
        self.assertGreater(column_correlation(truth, P2), 0.5)

    def test_unlabelled_column_fallback(self):
        T = self.T[self.num].copy()
        T["new"] = np.nan
        T.loc[T.index[:5], "new"] = 3.0
        m = TwinStack(min_stack=0).fit(self.X, T, bounds={"new": (1, 5)})
        P, O = m.predict(self.test)
        self.assertEqual(P["new"].nunique(), 1)
        self.assertTrue(((O["new"] >= 1) & (O["new"] <= 5)).all())

    def test_helpers(self):
        Xz = (self.X - self.X.mean()) / self.X.std()
        tr, te = self.T.index[:200], self.T.index[200:]
        Yz = (self.T[self.num] - self.T[self.num].mean()) / self.T[self.num].std()
        for b in ("svdridge_r5", "softimpute_r3", "knn20"):
            out = base_predict(b, Xz, Yz, tr, te)
            self.assertEqual(out.shape, (60, len(self.num)))
            self.assertFalse(out.isna().any().any())
        with self.assertRaises(ValueError):
            base_predict("nope", Xz, Yz, tr, te)
        p = np.linspace(0, 1, 200)
        y = (p > 0.5).astype(float)
        np.testing.assert_allclose(binned_median(p, y, np.array([0.1, 0.9])), [0.0, 1.0])
        c = column_correlation(pd.DataFrame({"a": [1, 2, 3] * 5, "b": [1] * 15}),
                               pd.DataFrame({"a": [1, 2, 3] * 5, "b": [2, 3, 4] * 5}))
        self.assertAlmostEqual(c, np.tanh(np.arctanh(0.999999) / 2), places=4)

    def test_pooled_logistic(self):
        rs = np.random.default_rng(2)
        n = 1500
        df = pd.DataFrame({"person": rs.integers(0, 150, n), "product": rs.choice(["x", "y", "z"], n),
                           "ratio": rs.uniform(0, 2, n)})
        z = 1.5 - 2 * df.ratio + (df["product"] == "x") * 1.0
        df["y"] = (rs.random(n) < 1 / (1 + np.exp(-z))).astype(int)
        p = pooled_logistic_oof(df, "y", "person", categorical=["product"], numeric=["ratio"])
        self.assertFalse(np.isnan(p).any())
        self.assertGreater(np.corrcoef(p, df.y)[0, 1], 0.3)


if __name__ == "__main__":
    unittest.main()
