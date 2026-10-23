"""Baselines safety: fit argument order, LightGBM rare-positive / eval-overlap / class-weight warnings and Platt
calibration, sample_weight, and pandas copy-on-write (read-only arrays)."""
import contextlib
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.methods import (ConstantBaseline, ItemMarginal, LightGBMBaseline, LogisticBaseline, MajorityBaseline,
                               PersistenceBaseline, SegmentRate, lightgbm_available)
from simengine.methods.baselines import MedianConstant
from simengine.methods.transfer import oof_predict


def frame(n=200, seed=0):
    rs = np.random.default_rng(seed)
    return pd.DataFrame({"x1": rs.normal(size=n), "x2": rs.normal(size=n), "y": rs.lognormal(1, 1, n),
                         "b": (rs.random(n) < 0.3).astype(int), "w": rs.integers(1, 4, n).astype(float)})


def per_fold(s: pd.Series, fit_predict) -> np.ndarray:
    """What a direct fit(y_train) gives, fold by fold, for the folds oof_predict used."""
    fold = s.attrs["fold"]
    out = np.full(len(fold), np.nan)
    for k in np.unique(fold[fold >= 0]):
        out[fold == k] = fit_predict(fold != k, int((fold == k).sum()))
    return out


class TestFitOrder(unittest.TestCase):
    """oof_predict and sklearn callers use fit(X, y); these baselines were written as fit(y, X=None)."""

    def setUp(self):
        self.df = frame()

    def test_oof_predict_swapped_equals_direct(self):
        d = self.df
        cases = [(ConstantBaseline, "y", lambda m, tr, n: ConstantBaseline().fit(d.y[tr]).predict(n)),
                 (ConstantBaseline, "b", lambda m, tr, n: ConstantBaseline().fit(d.b[tr]).predict(n)),
                 (MedianConstant, "y", lambda m, tr, n: MedianConstant().fit(d.y[tr]).predict(n)),
                 (MajorityBaseline, "b",
                  lambda m, tr, n: MajorityBaseline().fit(d.b[tr]).predict(n).astype(float))]
        for M, t, direct in cases:
            for feats in (["x1"], ["x1", "x2"]):
                with self.subTest(model=M.__name__, target=t, features=feats):
                    s = oof_predict(M(), d, features=feats, target=t, n_splits=4)
                    exp = per_fold(s, lambda tr, n: direct(M, tr, n))
                    np.testing.assert_allclose(s.to_numpy(), exp)

    def test_oof_predict_weight_goes_to_median_constant(self):
        d = self.df
        s = oof_predict(MedianConstant(), d, features=["x1", "x2"], target="y", n_splits=4, weight="w")
        exp = per_fold(s, lambda tr, n: MedianConstant().fit(d.y[tr], weight=d.w[tr]).predict(n))
        np.testing.assert_allclose(s.to_numpy(), exp)
        self.assertFalse(np.allclose(exp, per_fold(s, lambda tr, n: MedianConstant().fit(d.y[tr]).predict(n))))

    def test_old_and_keyword_calls_unchanged(self):
        d, X = self.df, self.df[["x1", "x2"]]
        ref = ConstantBaseline().fit(d.y).value_
        for call in (lambda m: m.fit(d.y), lambda m: m.fit(d.y, X), lambda m: m.fit(d.y.to_numpy(), X.to_numpy()),
                     lambda m: m.fit(y=d.y, X=X), lambda m: m.fit(X=X, y=d.y), lambda m: m.fit(X, d.y),
                     lambda m: m.fit(X.to_numpy(), d.y.to_numpy()), lambda m: m.fit(d[["x1"]], d.y),
                     lambda m: m.fit(X, d.y.to_numpy()[:, None])):
            self.assertAlmostEqual(call(ConstantBaseline()).value_, ref)
        self.assertEqual(MajorityBaseline().fit(X, d.b).label_, MajorityBaseline().fit(d.b).label_)
        w = d.w.to_numpy()
        self.assertAlmostEqual(MedianConstant().fit(X, d.y, w).value_, MedianConstant().fit(d.y, weight=w).value_)
        self.assertAlmostEqual(MedianConstant().fit(X, d.y, sample_weight=w).value_,
                               MedianConstant().fit(d.y, weight=w).value_)

    def test_ambiguous_both_1d_keeps_old_order(self):
        a, b = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])
        self.assertEqual(ConstantBaseline().fit(a, b).value_, 2.0)
        self.assertEqual(MedianConstant().fit(a, b).value_, 2.0)

    def test_2d_without_y_raises(self):
        with self.assertRaises(ValueError):
            ConstantBaseline().fit(self.df[["x1", "x2"]])
        with self.assertRaises(ValueError):
            MedianConstant().fit(self.df[["x1", "x2"]])
        with self.assertRaises(ValueError):
            MedianConstant().fit(self.df.y, weight=self.df.w, sample_weight=self.df.w)


@contextlib.contextmanager
def pandas3_mode():
    """copy-on-write (default in pandas 3) and, where the option exists, the pandas 3 string dtype."""
    with contextlib.ExitStack() as st:
        for opt in ("mode.copy_on_write", "future.infer_string"):
            try:
                st.enter_context(pd.option_context(opt, True))
            except (KeyError, ValueError, pd.errors.OptionError):
                pass
        yield


class TestCopyOnWrite(unittest.TestCase):
    """Under copy-on-write ``to_numpy()`` hands back read-only views; filling unseen rows in place failed."""

    def test_unseen_rows_filled(self):
        with pandas3_mode():
            dm = pd.DataFrame({"item": ["q1", "q1", "q2"], "truth": ["a", "b", "a"]})
            P = ItemMarginal().fit(dm).predict(pd.DataFrame({"item": ["q1", "q9"]}))
            np.testing.assert_allclose(P[1], [2 / 3, 1 / 3])
            S = SegmentRate(a=0).fit(["s", "s", "t"], ["x", "y", "x"]).predict(["s", "zz"])
            np.testing.assert_allclose(S[1], [2 / 3, 1 / 3])
            h = pd.DataFrame({"unit": ["u1", "u2"], "time": [1, 1], "value": [1.0, 3.0]})
            np.testing.assert_allclose(PersistenceBaseline().fit(h).predict(["u1", "new"]), [1.0, 2.0])
            X = frame()[["x1", "x2"]]
            self.assertTrue(np.isfinite(LogisticBaseline().fit(X, frame().b).predict(X)).all())


class TestLogisticSampleWeight(unittest.TestCase):
    def setUp(self):
        d = frame(400)
        self.X, self.y, self.w = d[["x1", "x2"]], d.b.to_numpy(), d.w.to_numpy()

    def test_weight_reaches_final_step(self):
        from sklearn.linear_model import LogisticRegression
        m = LogisticBaseline().fit(self.X, self.y, sample_weight=self.w)
        Xt = m.model_[:-1].transform(self.X)
        ref = LogisticRegression(max_iter=2000, C=1.0).fit(Xt, self.y, sample_weight=self.w)
        np.testing.assert_allclose(m.model_[-1].coef_, ref.coef_, rtol=1e-4, atol=1e-6)
        ones = LogisticBaseline().fit(self.X, self.y, sample_weight=np.ones(len(self.y))).predict(self.X)
        np.testing.assert_allclose(ones, LogisticBaseline().fit(self.X, self.y).predict(self.X))

    def test_subsample_keeps_weights_aligned(self):
        from sklearn.linear_model import LogisticRegression
        m = LogisticBaseline(max_rows=150).fit(self.X, self.y, sample_weight=self.w)
        i = np.random.default_rng(0).choice(len(self.X), 150, replace=False)
        Xt = m.model_[:-1].transform(self.X.iloc[i])
        ref = LogisticRegression(max_iter=2000, C=1.0).fit(Xt, self.y[i], sample_weight=self.w[i])
        np.testing.assert_allclose(m.model_[-1].coef_, ref.coef_, rtol=1e-4, atol=1e-6)

    def test_oof_predict_weight(self):
        d = frame(400)
        s = oof_predict(LogisticBaseline(), d, features=["x1", "x2"], target="b", n_splits=4, weight="w")
        self.assertTrue(np.isfinite(s.to_numpy()).all())

    def test_bad_weight_length(self):
        with self.assertRaises(ValueError):
            LogisticBaseline().fit(self.X, self.y, sample_weight=self.w[:10])


def rare(n=3000, rate=0.01, seed=0):
    rs = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rs.normal(size=n), "x2": rs.normal(size=n)})
    z = np.log(rate / (1 - rate)) + 1.0 * X.x1
    return X, (rs.random(n) < 1 / (1 + np.exp(-z))).astype(int)


def caught(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn()
    return out, [str(x.message) for x in w if issubclass(x.category, UserWarning)]


@unittest.skipUnless(lightgbm_available(), "lightgbm not installed")
class TestLightGBMSafety(unittest.TestCase):
    FAST = dict(n_estimators=40)

    def test_sparse_positive_warning(self):
        X, y = rare(rate=0.008)
        _, msgs = caught(lambda: LightGBMBaseline(**self.FAST).fit(X, y))
        self.assertTrue(any("positive" in m and "min_child_samples" in m for m in msgs), msgs)
        _, msgs = caught(lambda: LightGBMBaseline(min_child_samples=300, **self.FAST).fit(X, y))
        self.assertFalse(any("min_child_samples" in m for m in msgs), msgs)
        X, y = rare(rate=0.2)
        _, msgs = caught(lambda: LightGBMBaseline(**self.FAST).fit(X, y))
        self.assertFalse(any("min_child_samples" in m for m in msgs), msgs)

    def test_eval_overlap_warning(self):
        X, y = rare(rate=0.2)
        tr, va = X.iloc[:2000], X.iloc[2000:]
        _, msgs = caught(lambda: LightGBMBaseline(**self.FAST).fit(tr, y[:2000], eval_set=(va, y[2000:])))
        self.assertFalse(any("eval_set" in m for m in msgs), msgs)
        # shared row labels (a resampled frame)
        Xs = X.sample(frac=1.0, random_state=1)
        _, msgs = caught(lambda: LightGBMBaseline(**self.FAST).fit(Xs.iloc[:2000], y[Xs.index[:2000]],
                                                                   eval_set=(X.iloc[1500:], y[1500:])))
        self.assertTrue(any("eval_set" in m and "index" in m for m in msgs), msgs)
        # fresh 0..n indexes, same rows
        Xr = X.iloc[:500].reset_index(drop=True)
        _, msgs = caught(lambda: LightGBMBaseline(**self.FAST).fit(X, y, eval_set=(Xr, y[:500])))
        self.assertTrue(any("eval_set" in m and "identical" in m for m in msgs), msgs)
        # a numpy eval_set has no row labels: a filtered training frame's index is not compared with 0..n-1
        keep = X.x2.to_numpy() > 0
        Xf, Xo = X[keep], X[~keep].reset_index(drop=True)
        _, msgs = caught(lambda: LightGBMBaseline(**self.FAST).fit(Xf.to_numpy(), y[keep],
                                                                   eval_set=(Xo.to_numpy(), y[~keep])))
        self.assertFalse(any("eval_set" in m for m in msgs), msgs)
        _, msgs = caught(lambda: LightGBMBaseline(**self.FAST).fit(X.to_numpy(), y,
                                                                   eval_set=(X.to_numpy()[:300], y[:300])))
        self.assertTrue(any("eval_set" in m and "identical" in m for m in msgs), msgs)

    def test_class_weight_warns_and_platt_calibrates(self):
        X, y = rare(n=4000, rate=0.05)
        Xv, yv = rare(n=2000, rate=0.05, seed=1)
        m, msgs = caught(lambda: LightGBMBaseline(scale_pos_weight=19, **self.FAST).fit(X, y))
        self.assertTrue(any("not calibrated" in m_ and "calibrate='platt'" in m_ for m_ in msgs), msgs)
        self.assertGreater(m.predict_proba(Xv).mean(), 3 * yv.mean())
        for kw in (dict(is_unbalance=True), dict(class_weight="balanced")):
            _, msgs = caught(lambda: LightGBMBaseline(**kw, **self.FAST).fit(X, y))
            self.assertTrue(any("not calibrated" in m_ for m_ in msgs), (kw, msgs))
        _, msgs = caught(lambda: LightGBMBaseline(**self.FAST).fit(X, y))
        self.assertFalse(any("not calibrated" in m_ for m_ in msgs), msgs)
        # measured here: raw test mean 0.35 vs rate 0.083 (40 trees), Platt on 2000 held-out rows 0.068
        Xt, yt = rare(n=4000, rate=0.05, seed=2)
        c, msgs = caught(lambda: LightGBMBaseline(scale_pos_weight=19, calibrate="platt", **self.FAST)
                         .fit(X, y, calibrate_set=(Xv, yv)))
        self.assertFalse(any("not calibrated" in m_ for m_ in msgs), msgs)
        self.assertGreater(c.predict_proba(Xt, raw=True).mean(), 3 * yt.mean())
        self.assertLess(abs(c.predict_proba(Xt).mean() - yt.mean()), 0.03)
        self.assertLess(abs(c.predict(Xt).mean() - yt.mean()), 0.03)
        # eval_set doubles as the calibration rows when calibrate_set is not given
        e = LightGBMBaseline(scale_pos_weight=19, calibrate="platt", **self.FAST).fit(X, y, eval_set=(Xv, yv))
        self.assertLess(abs(e.predict_proba(Xt).mean() - yt.mean()), 0.03)

    def test_calibrate_needs_rows_and_binary(self):
        X, y = rare(rate=0.2)
        with self.assertRaises(ValueError):
            LightGBMBaseline(calibrate="platt", **self.FAST).fit(X, y)
        with self.assertRaises(ValueError):
            LightGBMBaseline(calibrate="isotonic-ish", **self.FAST)
        with self.assertRaises(ValueError):
            LightGBMBaseline(calibrate="platt", **self.FAST).fit(X, X.x1.to_numpy(), eval_set=(X, X.x1))

    def test_sample_weight(self):
        X, y = rare(rate=0.2)
        a = LightGBMBaseline(**self.FAST).fit(X, y).predict(X)
        b = LightGBMBaseline(**self.FAST).fit(X, y, sample_weight=np.ones(len(y))).predict(X)
        np.testing.assert_allclose(a, b)
        w = np.where(y == 1, 5.0, 1.0)
        c = LightGBMBaseline(**self.FAST).fit(X, y, sample_weight=w).predict(X)
        self.assertGreater(c.mean(), a.mean() + 0.1)
        d = frame(400)
        s = oof_predict(LightGBMBaseline(**self.FAST), d, features=["x1", "x2"], target="b", n_splits=4, weight="w")
        self.assertTrue(np.isfinite(s.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
