"""Group transfer (two-stage group level + person deviation) and the out-of-fold helper, on synthetic groups."""
import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

from simengine.methods import GroupTransfer, LogisticBaseline, lightgbm_available, oof_predict


def synth(n_groups=40, n_per=60, seed=0, binary=False, mu_sd=1.0, u_sd=0.8):
    """Real group effect: y depends on the group's MEAN x1 (contextual effect) and a hidden group shift u that a
    noisy 'LLM prior' column partly knows. Within-group spread of x1 is larger than between-group spread, so a
    person-only model on x1 recovers only part of the group level."""
    rs = np.random.default_rng(seed)
    mu = rs.normal(0, mu_sd, n_groups)
    u = rs.normal(0, u_sd, n_groups)
    g = np.repeat(np.arange(n_groups), n_per)
    x1 = mu[g] + rs.normal(0, 1.5, len(g))
    x2 = rs.normal(0, 1, len(g))
    lang = np.where(rs.random(len(g)) < 0.3 + 0.4 * (mu[g] > 0), "a", "b")
    lin = 2.0 * mu[g] + u[g] + 0.8 * x2 + 0.2 * x1
    if binary:
        y = (rs.random(len(g)) < 1 / (1 + np.exp(-(lin - lin.mean()) / 1.5))).astype(float)
    else:
        y = lin + rs.normal(0, 1, len(g))
    df = pd.DataFrame({"country": [f"c{k:02d}" for k in g], "x1": x1, "x2": x2, "lang": lang, "y": y},
                      index=[f"r{i}" for i in range(len(g))])
    prior = pd.DataFrame({"llm_prior": u + rs.normal(0, 0.3, n_groups)},
                         index=[f"c{k:02d}" for k in range(n_groups)])
    return df, prior


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


class SpyModel:
    """Records which groups it was trained on and predicts the training mean."""
    seen = []

    def fit(self, X, y, groups=None):
        SpyModel.seen.append(set(groups))
        self.m_ = float(np.mean(y))
        return self

    def predict(self, X, groups=None):
        self.pred_groups_ = set(groups)
        SpyModel.seen[-1] = (SpyModel.seen[-1], self.pred_groups_)
        return np.full(len(X), self.m_)


class TestOOFPredict(unittest.TestCase):
    def setUp(self):
        self.df, self.prior = synth(n_groups=12, n_per=20)

    def test_group_held_out_no_leak(self):
        SpyModel.seen = []
        s = oof_predict(SpyModel(), self.df, features=["x1", "x2"], target="y", group="country", n_splits=4)
        self.assertEqual(len(SpyModel.seen), 4)
        for train_g, test_g in SpyModel.seen:
            self.assertFalse(train_g & test_g)
        self.assertTrue(s.notna().all())
        self.assertEqual(sorted(set(s.attrs["fold"])), [0, 1, 2, 3])
        # every group lands in exactly one fold
        f = pd.Series(s.attrs["fold"], index=self.df.index).groupby(self.df.country).nunique()
        self.assertTrue((f == 1).all())

    def test_kfold_sklearn_aligned_and_seeded(self):
        df = self.df.sample(frac=1.0, random_state=1)               # non-sorted string index
        a = oof_predict(Ridge(), df, features=["x1", "x2"], target="y", seed=3)
        b = oof_predict(Ridge(), df, features=["x1", "x2"], target="y", seed=3)
        c = oof_predict(Ridge(), df, features=["x1", "x2"], target="y", seed=4)
        self.assertTrue(a.index.equals(df.index))
        np.testing.assert_allclose(a, b)
        self.assertFalse(np.allclose(a, c))
        insample = Ridge().fit(df[["x1", "x2"]], df.y).predict(df[["x1", "x2"]])
        self.assertGreater(mse(df.y, a), mse(df.y, insample))      # OOF is honest, slightly worse than in-sample

    def test_missing_target_not_predicted(self):
        df = self.df.copy()
        df.iloc[:7, df.columns.get_loc("y")] = np.nan
        s = oof_predict(Ridge(), df, features=["x1", "x2"], target="y", group="country", n_splits=3)
        self.assertTrue(s.iloc[:7].isna().all())
        self.assertTrue(s.iloc[7:].notna().all())
        self.assertTrue((s.attrs["fold"][:7] == -1).all())

    def test_predict_proba_factory_and_simengine_baseline(self):
        df = self.df.assign(yb=(self.df.y > self.df.y.median()).astype(int))
        p = oof_predict(lambda: LogisticRegression(max_iter=500), df, features=["x1", "x2"], target="yb",
                        group="country", n_splits=3, method="predict_proba")
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertGreater(roc_auc_score(df.yb, p), 0.6)
        q = oof_predict(LogisticBaseline(kind="binary"), df, features=["x1", "x2", "lang"], target="yb",
                        group="country", n_splits=3)
        self.assertTrue(((q >= 0) & (q <= 1)).all())

    def test_errors(self):
        with self.assertRaises(ValueError):
            oof_predict(Ridge(), self.df, features=["x1"], target="y", group="country", n_splits=20)
        with self.assertRaises(KeyError):
            oof_predict(Ridge(), self.df, features=["nope"], target="y")


class TestGroupTransfer(unittest.TestCase):
    def test_beats_global_mean_and_person_only_continuous(self):
        df, prior = synth()
        feats = ["x1", "x2", "lang"]
        kw = dict(features=feats, target="y", group="country", n_splits=5)
        mean = oof_predict(DummyRegressor(), df, **kw)
        person = oof_predict(LogisticBaseline(kind="continuous"), df, **kw)
        gt = oof_predict(GroupTransfer(person_model="ridge"), df, **kw)
        gt_llm = oof_predict(GroupTransfer(person_model="ridge", group_features=prior), df, **kw)
        m = {k: mse(df.y, v) for k, v in dict(mean=mean, person=person, gt=gt, gt_llm=gt_llm).items()}
        self.assertLess(m["person"], m["mean"])
        self.assertLess(m["gt"], m["person"])
        self.assertLess(m["gt_llm"], m["gt"])                      # the prior knows part of the hidden shift
        # group level: per-group mean of the OOF predictions tracks the true group means
        t = df.groupby("country").y.mean()
        self.assertGreater(np.corrcoef(t, gt_llm.groupby(df.country).mean())[0, 1],
                           np.corrcoef(t, person.groupby(df.country).mean())[0, 1])

    def test_beats_person_only_binary(self):
        df, prior = synth(binary=True, seed=1)
        kw = dict(features=["x1", "x2"], target="y", group="country", n_splits=5)
        person = oof_predict(LogisticBaseline(kind="binary"), df, **kw)
        gt = oof_predict(GroupTransfer(person_model="ridge", group_features=prior), df, **kw)
        self.assertTrue(((gt >= 0) & (gt <= 1)).all())
        self.assertGreater(roc_auc_score(df.y, gt), roc_auc_score(df.y, person))
        self.assertGreater(roc_auc_score(df.y, gt), 0.5)

    def test_new_groups_predict_groups_and_group_col(self):
        df, prior = synth(seed=2)
        tr = df[df.country < "c30"]
        te = df[df.country >= "c30"]
        m = GroupTransfer("country", person_model="ridge", composition=["x1", "lang"]).fit(tr.drop(columns="y"),
                                                                                       tr.y)
        lv = m.predict_groups(te)
        self.assertEqual(set(lv.index), set(te.country))
        self.assertGreater(np.corrcoef(lv.sort_index(), te.groupby("country").y.mean().sort_index())[0, 1], 0.6)
        self.assertIn("c_lang=a", m.group_feature_names_)
        self.assertNotIn("c_x2", m.group_feature_names_)
        p = m.predict(te.drop(columns="y"))
        self.assertEqual(len(p), len(te))
        self.assertTrue(np.isfinite(p).all())

    def test_group_level_only_and_prior_only(self):
        df, prior = synth(seed=3, mu_sd=0.3, u_sd=1.5)           # hidden shift dominates, prior knows it
        kw = dict(features=["x1", "x2"], target="y", group="country", n_splits=5)
        only_prior = oof_predict(GroupTransfer(composition=(), group_features=prior, person_model=None), df, **kw)
        # no person model: one value per group
        self.assertTrue((only_prior.groupby(df.country).nunique() == 1).all())
        mean = oof_predict(DummyRegressor(), df, **kw)
        self.assertLess(mse(df.y, only_prior), mse(df.y, mean))
        # fit-time group_features override, missing groups are filled with the training mean
        m = GroupTransfer(composition=(), person_model=None).fit(df[["x1"]], df.y, groups=df.country,
                                                                 group_features=prior.iloc[:20])
        self.assertTrue(np.isfinite(m.predict(df[["x1"]], groups=df.country, group_features=prior.iloc[:20])).all())

    def test_errors(self):
        df, _ = synth(n_groups=5, n_per=10)
        with self.assertRaises(ValueError):
            GroupTransfer().fit(df[["x1"]], df.y)                      # no groups
        with self.assertRaises(ValueError):
            GroupTransfer().fit(df[["x1"]], df.lang, groups=df.country)  # multiclass target
        with self.assertRaises(KeyError):
            GroupTransfer(composition=["zz"]).fit(df[["x1"]], df.y, groups=df.country)

    @unittest.skipUnless(lightgbm_available(), "lightgbm not installed")
    def test_lightgbm_person_model(self):
        df, prior = synth(n_groups=20, n_per=40, seed=4)
        s = oof_predict(GroupTransfer(group_features=prior), df, features=["x1", "x2", "lang"], target="y",
                        group="country", n_splits=4)
        mean = oof_predict(DummyRegressor(), df, features=["x1"], target="y", group="country", n_splits=4)
        self.assertLess(mse(df.y, s), mse(df.y, mean))


if __name__ == "__main__":
    unittest.main()
