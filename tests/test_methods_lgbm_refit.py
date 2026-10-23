"""LightGBMBaseline(refit_full=True): early stopping on eval_set picks the tree count, then one refit on train + eval
with round(best_iteration_ * refit_scale) trees (D13-W8 gap #5; the rehearsal did it by hand)."""
import unittest

import numpy as np
import pandas as pd

from simengine.methods import LightGBMBaseline, lightgbm_available

FAST = dict(learning_rate=0.2, min_child_samples=20)


def data(n=3000, seed=0):
    rs = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rs.normal(size=n), "x2": rs.normal(size=n),
                      "c": rs.choice(["a", "b", "c"], size=n)})
    z = -1.0 + 1.5 * X.x1 - 0.5 * X.x2 + (X.c == "b") * 0.7
    y = (rs.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return X, y


def trees(m) -> int:
    return int(m.model_.booster_.num_trees())


@unittest.skipUnless(lightgbm_available(), "lightgbm not installed")
class TestLightGBMRefitFull(unittest.TestCase):

    def setUp(self):
        X, y = data()
        self.tr, self.ytr, self.dv, self.ydv = X.iloc[:2000], y[:2000], X.iloc[2000:], y[2000:]

    def test_default_off_unchanged(self):
        m = LightGBMBaseline(**FAST).fit(self.tr, self.ytr, eval_set=(self.dv, self.ydv))
        self.assertFalse(hasattr(m, "n_estimators_"))
        self.assertEqual(trees(m), m.model_.best_iteration_)          # the booster is cut at the best iteration

    def test_refit_tree_count_and_rows(self):
        es = LightGBMBaseline(**FAST).fit(self.tr, self.ytr, eval_set=(self.dv, self.ydv))
        best = es.model_.best_iteration_
        m = LightGBMBaseline(refit_full=True, **FAST).fit(self.tr, self.ytr, eval_set=(self.dv, self.ydv))
        self.assertEqual(m.best_iteration_, best)
        self.assertEqual(m.n_estimators_, round(best * 1.1))
        self.assertEqual(trees(m), m.n_estimators_)
        # same as the by-hand recipe: a fixed tree count on train + eval
        hand = LightGBMBaseline(n_estimators=round(best * 1.1), **FAST).fit(
            pd.concat([self.tr, self.dv], ignore_index=True), np.concatenate([self.ytr, self.ydv]))
        np.testing.assert_allclose(m.predict_proba(self.dv), hand.predict_proba(self.dv), atol=1e-9)

    def test_refit_scale(self):
        m = LightGBMBaseline(refit_full=True, refit_scale=1.0, **FAST).fit(self.tr, self.ytr,
                                                                          eval_set=(self.dv, self.ydv))
        self.assertEqual(trees(m), m.best_iteration_)

    def test_eval_only_category_kept(self):
        dv = self.dv.copy()
        dv.loc[dv.index[:300], "c"] = "z"           # a level only the eval rows have
        m = LightGBMBaseline(refit_full=True, **FAST).fit(self.tr, self.ytr, eval_set=(dv, self.ydv))
        self.assertIn("z", m.categories_["c"])
        self.assertEqual(len(m.predict_proba(dv)), len(dv))

    def test_numpy_multiclass_continuous(self):
        rs = np.random.default_rng(1)
        X = rs.normal(size=(1500, 3))
        yk = np.array(["p", "q", "r"])[np.digitize(X[:, 0] + 0.3 * rs.normal(size=1500), [-0.5, 0.5])]
        m = LightGBMBaseline(refit_full=True, **FAST).fit(X[:1000], yk[:1000], eval_set=(X[1000:], yk[1000:]))
        self.assertEqual(m.predict_proba(X[:5]).shape, (5, 3))
        self.assertEqual(trees(m), 3 * m.n_estimators_)
        yc = X[:, 0] * 2 + rs.normal(size=1500)
        m = LightGBMBaseline(refit_full=True, **FAST).fit(X[:1000], yc[:1000], eval_set=(X[1000:], yc[1000:]))
        self.assertEqual(trees(m), m.n_estimators_)
        self.assertGreater(np.corrcoef(m.predict(X[1000:]), yc[1000:])[0, 1], 0.7)

    def test_multiclass_string_eval_set(self):
        # LightGBM 4.7 label-encodes eval_set labels but not eval_y: string classes with eval_set raised
        # "could not convert string to float" before the refit option existed
        rs = np.random.default_rng(3)
        X = rs.normal(size=(1200, 2))
        yk = np.array(["lo", "mid", "hi"])[np.digitize(X[:, 0], [-0.5, 0.5])]
        m = LightGBMBaseline(**FAST).fit(X[:800], yk[:800], eval_set=(X[800:], yk[800:]))
        self.assertEqual(m.classes_, ["hi", "lo", "mid"])
        P = m.predict_proba(X[800:])
        acc = np.mean(np.array(m.classes_)[P.argmax(1)] == yk[800:])
        self.assertGreater(acc, 0.8)                 # the eval labels matched the right classes
        with self.assertRaises(ValueError):          # a class the training rows lack
            LightGBMBaseline(**FAST).fit(X[:800], yk[:800], eval_set=(X[800:], np.full(400, "new")))

    def test_errors(self):
        with self.assertRaises(ValueError):          # nothing to stop on
            LightGBMBaseline(refit_full=True).fit(self.tr, self.ytr)
        with self.assertRaises(ValueError):          # Platt on eval rows the refit trained on
            LightGBMBaseline(refit_full=True, calibrate="platt").fit(self.tr, self.ytr, eval_set=(self.dv, self.ydv))
        with self.assertRaises(ValueError):          # eval rows have no weights
            LightGBMBaseline(refit_full=True).fit(self.tr, self.ytr, eval_set=(self.dv, self.ydv),
                                                  sample_weight=np.ones(len(self.tr)))
        with self.assertRaises(ValueError):
            LightGBMBaseline(refit_full=True, refit_scale=0)

    def test_calibrate_set_ok(self):
        X, y = data(seed=2)
        m = LightGBMBaseline(refit_full=True, calibrate="platt", **FAST).fit(
            self.tr, self.ytr, eval_set=(self.dv, self.ydv), calibrate_set=(X.iloc[:800], y[:800]))
        p = m.predict_proba(X.iloc[800:])
        self.assertTrue(np.all((p > 0) & (p < 1)))


if __name__ == "__main__":
    unittest.main()
