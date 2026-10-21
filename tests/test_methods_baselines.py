"""simengine.methods baselines on synthetic data (offline, fast)."""
import unittest

import numpy as np
import pandas as pd

from simengine.methods import (ConstantBaseline, ItemMarginal, KNNBaseline, LightGBMBaseline, LogisticBaseline,
                               MajorityBaseline, PersistenceBaseline, SegmentRate, TextKNNRegressor, basket_tensor,
                               decayed_frequency, infer_kind, lightgbm_available, overall_copy, persistence,
                               personal_frequency, popularity, segment_distributions, tifu_knn,
                               uniform_distribution, weighted_distribution)


def tabular(n=600, seed=0):
    rs = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rs.normal(size=n), "x2": rs.normal(size=n),
                      "seg": rs.choice(["a", "b", "c"], n)})
    z = 1.5 * X.x1 - X.x2 + (X.seg == "a") * 1.0
    y = (rs.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return X, y, z


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


class TestSimple(unittest.TestCase):
    def test_infer_kind(self):
        self.assertEqual(infer_kind([0, 1, 1]), "binary")
        self.assertEqual(infer_kind([0.2, 1.5]), "continuous")
        self.assertEqual(infer_kind(["x", "y"]), "multiclass")

    def test_constant_and_majority(self):
        c = ConstantBaseline().fit([0, 1, 1, 1])
        self.assertAlmostEqual(c.predict(3)[0], 0.75)
        c = ConstantBaseline().fit([1.0, 2.0, 10.0])
        self.assertEqual(c.predict([0, 0])[0], 2.0)
        c = ConstantBaseline().fit(["a", "b", "b", "b"])
        np.testing.assert_allclose(c.predict(2)[0], [0.25, 0.75])
        m = MajorityBaseline().fit(["x", "y", "y"])
        self.assertEqual(list(m.predict(2)), ["y", "y"])
        self.assertEqual(m.predict_proba(1).sum(), 1.0)

    def test_item_marginal(self):
        df = pd.DataFrame({"item": ["q1"] * 4 + ["q2"] * 4, "truth": [1, 2, 3, 4, 5, 5, 5, 5]})
        m = ItemMarginal().fit(df)
        np.testing.assert_allclose(m.predict(pd.DataFrame({"item": ["q1", "q2", "q9"]})), [2.5, 5.0, 3.75])
        dm = pd.DataFrame({"item": ["q1", "q1", "q2"], "truth": ["a", "b", "a"]})
        P = ItemMarginal().fit(dm).predict(pd.DataFrame({"item": ["q1", "q2", "q3"]}))
        np.testing.assert_allclose(P.sum(1), 1.0)

    def test_segment_rate_shrinkage(self):
        seg = ["a"] * 2 + ["b"] * 100
        y = [1, 1] + [0] * 100
        p = SegmentRate(a=10).fit(seg, y).predict(["a", "b", "zzz"])
        g = 2 / 102
        self.assertAlmostEqual(p[0], (2 + 10 * g) / 12)
        self.assertAlmostEqual(p[2], g)
        r = SegmentRate().fit(seg, y, valid=(["a", "b"], [1, 0]))
        self.assertIn(r.a_, SegmentRate.GRID)

    def test_distributions(self):
        np.testing.assert_allclose(uniform_distribution(4), 0.25)
        d = weighted_distribution(["a", "b", "b", "z"], ["a", "b"], [1, 1, 2, 5])
        np.testing.assert_allclose(d, [0.25, 0.75])
        oc = overall_copy([0.2, 0.8], ["g1", "g2"])
        np.testing.assert_allclose(oc["g2"], [0.2, 0.8])
        df = pd.DataFrame({"v": ["a", "a", "b", "b"], "s": ["x", "x", "y", "y"]})
        self.assertEqual(segment_distributions(df, "v", "s", ["a", "b"], mode="others")["x"].tolist(), [0, 1])
        self.assertEqual(segment_distributions(df, "v", "s", ["a", "b"], mode="own")["x"].tolist(), [1, 0])
        pr = segment_distributions(df, "v", "s", ["a", "b"], mode="pair", pairs={"x": "y", "y": "x"})
        self.assertEqual(pr["x"].tolist(), [0, 1])

    def test_persistence(self):
        h = pd.DataFrame({"unit": ["u1", "u1", "u2", "u3"], "time": [1, 2, 2, 1], "value": [5, 7, 3, 9]})
        p = PersistenceBaseline().fit(h).predict(["u1", "u2", "u3", "new"])
        np.testing.assert_allclose(p, [7, 3, 5, 5])   # u3 has no row in the last period -> median fill
        p = persistence(h, ["u3"], last_n_periods=2, fill="zero")
        np.testing.assert_allclose(p, [9])


class TestBasket(unittest.TestCase):
    def setUp(self):
        rs = np.random.default_rng(1)
        rows = []
        for u in range(40):
            fav = u % 5
            for t in range(10):
                rows.append((u, t, fav))
                if rs.random() < 0.3:
                    rows.append((u, t, int(rs.integers(5))))
        self.df = pd.DataFrame(rows, columns=["user", "period", "item"])
        self.X, self.users, self.items = basket_tensor(self.df, "user", "period", "item", n_periods=10)

    def test_shapes_and_personal(self):
        self.assertEqual(self.X.shape, (40, 10, 5))
        self.assertEqual(popularity(self.X, 8, 4).shape, (5,))
        for S in (personal_frequency(self.X, 8, 4), decayed_frequency(self.X, 8, 4),
                  tifu_knn(self.X, 8, 4, k=10)):
            self.assertEqual(S.shape, (40, 5))
            fav = np.array([u % 5 for u in self.users])
            self.assertGreater(np.mean(S.argmax(1) == fav), 0.95)


class TestSupervised(unittest.TestCase):
    def test_logistic_and_knn(self):
        X, y, _ = tabular()
        tr, te = slice(0, 400), slice(400, None)
        p = LogisticBaseline().fit(X[tr], y[tr]).predict_proba(X[te])
        self.assertGreater(auc(y[te], p), 0.75)
        p = KNNBaseline(k=30).fit(X[tr], y[tr]).predict_proba(X[te])
        self.assertGreater(auc(y[te], p), 0.65)
        yc = X.x1 * 2 + 0.1
        r = LogisticBaseline(kind="continuous").fit(X[tr], yc[tr]).predict(X[te])
        self.assertGreater(np.corrcoef(r, yc[te])[0, 1], 0.95)
        ym = np.where(X.x1 > 0.5, "hi", np.where(X.x1 < -0.5, "lo", "mid"))
        P = LogisticBaseline().fit(X[tr], ym[tr]).predict_proba(X[te])
        self.assertEqual(P.shape[1], 3)

    @unittest.skipUnless(lightgbm_available(), "lightgbm not installed")
    def test_lightgbm(self):
        X, y, _ = tabular(1500)
        m = LightGBMBaseline(n_estimators=100).fit(X[:1000], y[:1000])
        self.assertGreater(auc(y[1000:], m.predict_proba(X[1000:])), 0.75)
        self.assertTrue(m.feature_importance(3))
        m = LightGBMBaseline().fit(X[:800], y[:800], eval_set=(X[800:1000], y[800:1000]))
        self.assertGreater(auc(y[1000:], m.predict(X[1000:])), 0.75)

    def test_text_knn(self):
        texts = [f"topic{i % 3} words about thing {i % 3}" for i in range(30)]
        y = np.array([float(i % 3) for i in range(30)])
        m = TextKNNRegressor(k=3).fit(texts, y, groups=np.arange(30))
        p = m.predict(["topic2 words about thing 2"], groups=[-1])
        self.assertAlmostEqual(p[0], 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
