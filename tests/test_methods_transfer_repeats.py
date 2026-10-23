"""oof_predict: n_repeats (seeded group folds, mean of repeats), target=[...] (one column per target), and inputs
that pandas hands back as read-only arrays (copy-on-write)."""
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from simengine.methods import GroupTransfer, oof_predict


def synth(n_groups=14, n_per=12, seed=0):
    rs = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), n_per)
    x1 = rs.normal(0, 1, len(g)) + 0.5 * g / n_groups
    x2 = rs.normal(0, 1, len(g))
    lvl = rs.normal(0, 1, n_groups)[g]
    df = pd.DataFrame({"grp": [f"g{k:02d}" for k in g], "x1": x1, "x2": x2,
                       "y": 1.5 * x1 + lvl + rs.normal(0, 0.5, len(g)),
                       "z": -x2 + 0.5 * lvl + rs.normal(0, 0.5, len(g))},
                      index=[f"r{i}" for i in range(len(g))])
    df.loc[df.index[::7], "z"] = np.nan                       # z has its own missing rows
    return df


class TestRepeats(unittest.TestCase):
    def setUp(self):
        self.df = synth()
        self.kw = dict(features=["x1", "x2"], group="grp", n_splits=4)

    def test_default_is_one_repeat_and_unchanged(self):
        a = oof_predict(Ridge(), self.df, target="y", **self.kw)
        b = oof_predict(Ridge(), self.df, target="y", n_repeats=1, **self.kw)
        np.testing.assert_allclose(a, b)
        self.assertEqual(b.attrs["repeats"], 1)
        np.testing.assert_array_equal(a.attrs["fold"], b.attrs["fold"])

    def test_mean_of_repeats_equals_manual_loop(self):
        df = self.df
        s = oof_predict(Ridge(), df, target="y", n_repeats=3, seed=5, **self.kw)
        self.assertEqual(s.attrs["repeats"], 3)
        P, F = s.attrs["repeat_pred"], s.attrs["repeat_fold"]
        self.assertEqual(P.shape, (len(df), 3))
        self.assertEqual(F.shape, (len(df), 3))
        manual = []
        for r in range(3):
            m = oof_predict(Ridge(), df, features=["x1", "x2"], target="y", group="grp", folds=F[:, r])
            manual.append(m.to_numpy())
            np.testing.assert_allclose(P[:, r], m.to_numpy())
        np.testing.assert_allclose(s.to_numpy(), np.mean(manual, axis=0))
        np.testing.assert_allclose(s.attrs["repeat_sd"], np.std(manual, axis=0, ddof=1))
        np.testing.assert_array_equal(s.attrs["fold"], F[:, 0])

    def test_repeat_folds_are_group_disjoint_seeded_and_differ(self):
        df = self.df
        s = oof_predict(Ridge(), df, target="y", n_repeats=4, seed=1, **self.kw)
        F = s.attrs["repeat_fold"]
        for r in range(4):
            per_group = pd.Series(F[:, r]).groupby(df.grp.to_numpy()).nunique()
            self.assertTrue((per_group == 1).all())                 # a group sits in exactly one fold
            self.assertEqual(sorted(set(F[:, r])), [0, 1, 2, 3])
        self.assertGreater(len({tuple(F[:, r]) for r in range(4)}), 1)
        again = oof_predict(Ridge(), df, target="y", n_repeats=4, seed=1, **self.kw)
        np.testing.assert_array_equal(F, again.attrs["repeat_fold"])
        other = oof_predict(Ridge(), df, target="y", n_repeats=4, seed=2, **self.kw)
        self.assertFalse(np.array_equal(F, other.attrs["repeat_fold"]))

    def test_no_group_repeat_zero_is_the_single_run(self):
        df = self.df
        one = oof_predict(Ridge(), df, features=["x1"], target="y", seed=3)
        rep = oof_predict(Ridge(), df, features=["x1"], target="y", seed=3, n_repeats=2)
        np.testing.assert_allclose(rep.attrs["repeat_pred"][:, 0], one.to_numpy())
        two = oof_predict(Ridge(), df, features=["x1"], target="y", seed=4)
        np.testing.assert_allclose(rep.attrs["repeat_pred"][:, 1], two.to_numpy())

    def test_groups_reach_group_transfer_in_every_repeat(self):
        s = oof_predict(GroupTransfer(person_model="ridge"), self.df, target="y", n_repeats=2, **self.kw)
        self.assertTrue(np.isfinite(s.to_numpy()).all())

    def test_missing_target_rows_stay_nan_in_every_repeat(self):
        s = oof_predict(Ridge(), self.df, target="z", n_repeats=2, **self.kw)
        miss = self.df.z.isna().to_numpy()
        self.assertTrue(np.isnan(s.to_numpy()[miss]).all())
        self.assertTrue(np.isnan(s.attrs["repeat_pred"][miss]).all())
        self.assertTrue((s.attrs["repeat_fold"][miss] == -1).all())
        self.assertTrue(np.isfinite(s.to_numpy()[~miss]).all())

    def test_bad_n_repeats_and_folds_conflict(self):
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                oof_predict(Ridge(), self.df, target="y", n_repeats=bad, **self.kw)
        with self.assertRaises(ValueError):
            oof_predict(Ridge(), self.df, features=["x1"], target="y", folds=np.arange(len(self.df)) % 3,
                        n_repeats=2)


class TestMultiTarget(unittest.TestCase):
    def setUp(self):
        self.df = synth()
        self.kw = dict(features=["x1", "x2"], group="grp", n_splits=4)

    def test_frame_equals_one_call_per_target(self):
        df = self.df
        fr = oof_predict(Ridge(), df, target=["y", "z"], **self.kw)
        self.assertIsInstance(fr, pd.DataFrame)
        self.assertEqual(list(fr.columns), ["oof_y", "oof_z"])
        self.assertTrue(fr.index.equals(df.index))
        for t in ("y", "z"):
            one = oof_predict(Ridge(), df, target=t, **self.kw)
            np.testing.assert_allclose(fr[f"oof_{t}"].to_numpy(), one.to_numpy())
            np.testing.assert_array_equal(fr.attrs["fold"][t], one.attrs["fold"])
        self.assertEqual(fr.attrs["targets"], ["y", "z"])
        self.assertTrue(fr["oof_z"].isna().to_numpy()[df.z.isna().to_numpy()].all())

    def test_multi_target_with_repeats(self):
        fr = oof_predict(Ridge(), self.df, target=("y", "z"), n_repeats=2, seed=7, **self.kw)
        one = oof_predict(Ridge(), self.df, target="y", n_repeats=2, seed=7, **self.kw)
        np.testing.assert_allclose(fr["oof_y"].to_numpy(), one.to_numpy())
        self.assertEqual(fr.attrs["repeats"], 2)
        self.assertEqual(fr.attrs["repeat_pred"]["y"].shape, (len(self.df), 2))

    def test_bad_target_lists(self):
        with self.assertRaises(ValueError):
            oof_predict(Ridge(), self.df, target=[], **self.kw)
        with self.assertRaises(ValueError):
            oof_predict(Ridge(), self.df, target=["y", "y"], **self.kw)
        with self.assertRaises(KeyError):
            oof_predict(Ridge(), self.df, target=["y", "nope"], **self.kw)


class TestReadOnlyInputs(unittest.TestCase):
    """Under pandas copy-on-write (default in pandas 3) ``to_numpy()`` returns read-only views."""

    def test_copy_on_write(self):
        df = synth()
        df.loc[df.index[:5], "y"] = np.nan
        ref = oof_predict(Ridge(), df, features=["x1", "x2"], target="y", group="grp", n_splits=4)
        gt_ref = oof_predict(GroupTransfer(person_model="ridge"), df, features=["x1", "x2"], target="y",
                             group="grp", n_splits=4)
        with pd.option_context("mode.copy_on_write", True):
            df2 = df.copy()
            s = oof_predict(Ridge(), df2, features=["x1", "x2"], target="y", group="grp", n_splits=4)
            gt = oof_predict(GroupTransfer(person_model="ridge"), df2, features=["x1", "x2"], target="y",
                             group="grp", n_splits=4)
            w = oof_predict(Ridge(), df2, features=["x1", "x2"], target="y", group="grp", n_splits=4,
                            weight=np.ones(len(df2)), folds=df2.grp.str[1:].astype(int) % 4)
        np.testing.assert_allclose(s, ref)
        np.testing.assert_allclose(gt, gt_ref)
        self.assertTrue(np.isfinite(w.to_numpy()[5:]).all())


if __name__ == "__main__":
    unittest.main()
