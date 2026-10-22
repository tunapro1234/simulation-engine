"""oof_predict(folds=...): precomputed folds are used exactly; groups still reach GroupTransfer."""
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from simengine.methods import GroupTransfer, oof_predict


def synth(n_groups=12, n_per=15, seed=0):
    rs = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), n_per)
    x1 = rs.normal(0, 1, len(g)) + 0.5 * g / n_groups
    y = 1.5 * x1 + rs.normal(0, 1, n_groups)[g] + rs.normal(0, 0.5, len(g))
    df = pd.DataFrame({"grp": [f"g{k:02d}" for k in g], "x1": x1, "y": y}, index=[f"r{i}" for i in range(len(g))])
    df["fold"] = df.grp.map({f"g{k:02d}": k % 4 for k in range(n_groups)})
    return df


class Spy:
    log = []

    def fit(self, X, y, groups=None):
        Spy.log.append({"n_train": len(X), "train_idx": set(X.index),
                        "train_g": None if groups is None else set(groups)})
        self.m_ = float(np.mean(y))
        return self

    def predict(self, X, groups=None):
        Spy.log[-1]["test_g"] = None if groups is None else set(groups)
        return np.full(len(X), self.m_)


class TestOOFFolds(unittest.TestCase):
    def setUp(self):
        self.df = synth()

    def test_fold_series_equals_manual_loop(self):
        df = self.df
        s = oof_predict(Ridge(), df, features=["x1"], target="y", folds=df["fold"])
        manual = pd.Series(np.nan, index=df.index)
        for k in range(4):
            tr, ho = df.fold != k, df.fold == k
            manual[ho] = Ridge().fit(df.loc[tr, ["x1"]], df.y[tr]).predict(df.loc[ho, ["x1"]])
        np.testing.assert_allclose(s.to_numpy(), manual.to_numpy())
        self.assertEqual(s.attrs["fold_ids"], [0, 1, 2, 3])
        np.testing.assert_array_equal(s.attrs["fold"], df.fold.to_numpy())

    def test_column_name_array_and_shuffled_series(self):
        df = self.df
        a = oof_predict(Ridge(), df, features=["x1"], target="y", folds="fold")
        b = oof_predict(Ridge(), df, features=["x1"], target="y", folds=df.fold.to_numpy())
        c = oof_predict(Ridge(), df, features=["x1"], target="y", folds=df.fold.sample(frac=1.0, random_state=2))
        np.testing.assert_allclose(a, b)
        np.testing.assert_allclose(a, c)                                  # Series aligned by index, not position
        with self.assertRaises(ValueError):
            oof_predict(Ridge(), df, features=["x1"], target="y", folds=df.fold.iloc[:-3])
        with self.assertRaises(KeyError):
            oof_predict(Ridge(), df, features=["x1"], target="y", folds="nope")

    def test_index_pairs_match_kfold(self):
        df = self.df
        pairs = list(KFold(5, shuffle=True, random_state=3).split(df))
        a = oof_predict(Ridge(), df, features=["x1"], target="y", folds=pairs)
        b = oof_predict(Ridge(), df, features=["x1"], target="y", seed=3)
        np.testing.assert_allclose(a, b)
        bad = [(np.arange(10), np.arange(5, 15))]
        with self.assertRaises(ValueError):
            oof_predict(Ridge(), df, features=["x1"], target="y", folds=bad)

    def test_n_splits_conflict(self):
        with self.assertRaises(ValueError):
            oof_predict(Ridge(), self.df, features=["x1"], target="y", folds="fold", n_splits=5)
        s = oof_predict(Ridge(), self.df, features=["x1"], target="y", folds="fold", n_splits=4)
        self.assertTrue(s.notna().all())

    def test_groups_still_reach_model_and_split_is_exact(self):
        Spy.log = []
        s = oof_predict(Spy(), self.df, features=["x1"], target="y", group="grp", folds="fold")
        self.assertEqual(len(Spy.log), 4)
        for k, rec in enumerate(Spy.log):
            self.assertEqual(rec["train_idx"], set(self.df.index[self.df.fold != k]))
            self.assertFalse(rec["train_g"] & rec["test_g"])
        self.assertTrue(s.notna().all())
        # GroupTransfer gets the groups from oof_predict, not from a column of X
        gt = oof_predict(GroupTransfer(person_model="ridge"), self.df, features=["x1"], target="y", group="grp",
                         folds="fold")
        self.assertTrue(np.isfinite(gt).all())

    def test_train_only_and_left_out_rows(self):
        df = self.df.copy()
        df.loc[df.index[:5], "fold"] = -1          # always trained on, never predicted
        df["fold"] = df.fold.astype(float)
        df.loc[df.index[5:8], "fold"] = np.nan     # left out entirely
        df.loc[df.index[8], "y"] = np.nan          # missing target: left out
        Spy.log = []
        s = oof_predict(Spy(), df, features=["x1"], target="y", folds="fold")
        self.assertTrue(s.iloc[:9].isna().all())
        self.assertTrue(s.iloc[9:].notna().all())
        for rec in Spy.log:
            self.assertTrue(set(df.index[:5]) <= rec["train_idx"])
            self.assertFalse(set(df.index[5:9]) & rec["train_idx"])

    def test_warns_when_group_spans_train_and_test(self):
        df = self.df.copy()
        df["fold"] = np.arange(len(df)) % 3                 # row-level folds cut through groups
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            oof_predict(Ridge(), df, features=["x1"], target="y", group="grp", folds="fold")
        self.assertTrue(any("both train and test" in str(x.message) for x in w))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            oof_predict(Ridge(), self.df, features=["x1"], target="y", group="grp", folds="fold")
        self.assertFalse(any("both train and test" in str(x.message) for x in w))

    def test_default_unchanged(self):
        a = oof_predict(Ridge(), self.df, features=["x1"], target="y", group="grp")
        b = oof_predict(Ridge(), self.df, features=["x1"], target="y", group="grp", n_splits=5)
        np.testing.assert_allclose(a, b)
        self.assertEqual(sorted(set(a.attrs["fold"])), [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
