"""Item cold start: item-held-out member features (no held-out information), Procrustes alignment, MFBaseline,
and GroupTransfer(group=item) on synthetic ratings with a real taste structure."""
import unittest

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from simengine.methods import (GroupTransfer, ItemHoldout, MFBaseline, item_holdout_features, oof_predict,
                               procrustes_align)


def synth(n_users=300, n_items=150, k=3, seed=0):
    """Members and items have latent tastes U, V. Members rate items they are drawn to (selection on U.V), the
    rating = level + member bias + item quality + 0.5 U.V + noise, and item quality is tied to V[:, 0]. So a new
    item's audience composition (who rated it) carries its taste position; the genres are sign(V)."""
    rs = np.random.default_rng(seed)
    U = rs.normal(0, 1, (n_users, k))
    V = rs.normal(0, 1, (n_items, k))
    bu = rs.normal(0, 0.5, n_users)
    bi = 0.6 * V[:, 0] + rs.normal(0, 0.15, n_items)
    A = U @ V.T
    mask = rs.random((n_users, n_items)) < 1 / (1 + np.exp(-(-1.0 + 0.8 * A)))
    uu, ii = np.nonzero(mask)
    rating = 3.0 + bu[uu] + bi[ii] + 0.5 * A[uu, ii] + rs.normal(0, 0.7, len(uu))
    ratings = pd.DataFrame({"user_id": uu, "item_id": ii, "rating": rating})
    items = pd.DataFrame((V > 0).astype(float), columns=[f"g{j}" for j in range(k)])
    users = pd.DataFrame({"age": rs.integers(18, 60, n_users).astype(float)})
    return ratings, items, users


FEATURE_COLS = ["user_mean", "user_n", "user_sd", "user_n_pct", "affinity"]


class TestItemHoldoutFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ratings, cls.items, cls.users = synth()
        cls.res = item_holdout_features(cls.ratings, items=cls.items, users=cls.users, svd_rank=4, seed=0)

    def test_shape_and_folds(self):
        res = self.res
        self.assertIsInstance(res, ItemHoldout)
        self.assertEqual(sorted(res.folds.unique()), [0, 1, 2, 3, 4])
        self.assertTrue(res.folds.index.equals(res.dev.index))
        # every dev row belongs to the fold of its item
        np.testing.assert_array_equal(res.dev.item_id.map(res.item_folds).to_numpy(), res.dev.fold.to_numpy())
        self.assertEqual(res.svd_cols, ["svd_0", "svd_1", "svd_2", "svd_3"])
        for c in ["user_mean", "user_n_pct", "user_sd", "affinity", "age"] + res.svd_cols:
            self.assertIn(c, res.features)
        self.assertNotIn("user_n", res.features)                  # absolute count shifts between folds and test
        np.testing.assert_allclose(res.dev.residual, res.dev.rating - res.dev.user_mean)

    def test_fold_statistics_exclude_held_out_items(self):
        res, e = self.res, self.ratings
        for k in range(5):
            held = res.item_folds.index[res.item_folds == k]
            tr = e[~e.item_id.isin(held)]
            d = res.dev[res.dev.fold == k]
            expect = tr.groupby("user_id").rating.mean()
            np.testing.assert_allclose(d.user_mean.to_numpy(), d.user_id.map(expect).to_numpy())

    def _changed(self, item, **kw):
        e = self.ratings.copy()
        m = e.item_id == item
        rs = np.random.default_rng(9)
        e.loc[m, "rating"] = rs.normal(10, 3, m.sum())            # wildly different ratings, same raters
        a = item_holdout_features(self.ratings, items=self.items, svd_rank=4, seed=0, **kw)
        b = item_holdout_features(e, items=self.items, svd_rank=4, seed=0, **kw)
        k = a.item_folds[item]
        return a.dev[a.dev.fold == k], b.dev[b.dev.fold == k], a, b

    def test_held_out_ratings_do_not_change_their_folds_features(self):
        item = int(self.res.item_folds.index[3])
        for kw in ({"align": None}, {"align": None, "cap": 20}):
            da, db, a, b = self._changed(item, **kw)
            self.assertTrue(da[["user_id", "item_id"]].equals(db[["user_id", "item_id"]]))
            cols = FEATURE_COLS + a.svd_cols
            np.testing.assert_allclose(da[cols].to_numpy(), db[cols].to_numpy(), atol=1e-10)
            self.assertFalse(np.allclose(da.rating, db.rating))     # only the target moved
            # negative control: folds that train on the item do see the change
            ka = a.item_folds[item]
            oa, ob = a.dev[a.dev.fold != ka], b.dev[b.dev.fold != ka]
            self.assertFalse(np.allclose(oa.user_mean.to_numpy(), ob.user_mean.to_numpy()))
        # aligned to the full basis: the only held-out information is the k x k rotation (and, with the default
        # align_scale=True, one scalar), so non-SVD features are identical and each member's taste vector keeps its
        # norm and inner products up to that scalar
        for kw in ({"align_scale": False}, {}):
            da, db, a, b = self._changed(item, **kw)
            np.testing.assert_allclose(da[FEATURE_COLS].to_numpy(), db[FEATURE_COLS].to_numpy(), atol=1e-10)
            k = a.item_folds[item]
            sa, sb = a.scales.get(k, 1.0), b.scales.get(k, 1.0)
            self.assertEqual(bool(a.scales), not kw)
            Va, Vb = da[a.svd_cols].to_numpy() / sa, db[a.svd_cols].to_numpy() / sb
            np.testing.assert_allclose(Va @ Va[:50].T, Vb @ Vb[:50].T, atol=1e-8)

    def test_procrustes_makes_fold_vectors_comparable(self):
        raw = item_holdout_features(self.ratings, items=self.items, svd_rank=4, seed=0, align=None)
        al = item_holdout_features(self.ratings, items=self.items, svd_rank=4, seed=0)
        cols = al.svd_cols
        ref = al.members.loc[al.dev.user_id, cols].to_numpy()
        err_al = np.linalg.norm(al.dev[cols].to_numpy() - ref) / np.linalg.norm(ref)
        err_raw = np.linalg.norm(raw.dev[cols].to_numpy() - ref) / np.linalg.norm(ref)
        self.assertLess(err_al, err_raw)
        self.assertLess(err_al, 0.5)
        for R in al.rotations.values():
            np.testing.assert_allclose(R @ R.T, np.eye(4), atol=1e-8)
        # a deliberately flipped / rotated basis is put back
        ref_df = al.members[cols]
        rs = np.random.default_rng(1)
        Q, _ = np.linalg.qr(rs.normal(size=(4, 4)))
        rot = pd.DataFrame(ref_df.to_numpy() @ Q, index=ref_df.index, columns=cols)
        back = procrustes_align(rot, ref_df, cols)
        np.testing.assert_allclose(back[cols].to_numpy(), ref_df.to_numpy(), atol=1e-8)
        with self.assertRaises(ValueError):
            procrustes_align(rot.iloc[:2], ref_df, cols)

    def test_align_scale_matches_test_time_scale(self):
        """Regression: a fold's SVD sees ~80% of the ratings, so with rotation only its taste vectors were ~sqrt(0.8)
        smaller than the transform() vectors (sd ratio ~0.89); the scaled alignment brings the ratio near 1."""
        rot = item_holdout_features(self.ratings, svd_rank=4, seed=0, align_scale=False)
        sc = item_holdout_features(self.ratings, svd_rank=4, seed=0)
        cols = sc.svd_cols
        ratio = lambda r: r.dev[cols].std().to_numpy() / r.transform(r.dev[["user_id", "item_id"]])[cols].std().to_numpy()
        self.assertTrue((ratio(rot)[:3] < 0.95).all(), ratio(rot))
        self.assertTrue((np.abs(ratio(sc)[:3] - 1) < 0.05).all(), ratio(sc))
        self.assertEqual(sorted(sc.scales), sorted(sc.rotations))
        self.assertTrue(all(s > 1 for s in sc.scales.values()))
        self.assertEqual(rot.scales, {})
        # procrustes_align(scale=True) recovers a shrunk, rotated copy exactly
        ref = sc.members[cols]
        Q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(4, 4)))
        small = pd.DataFrame(0.7 * ref.to_numpy() @ Q, index=ref.index, columns=cols)
        back = procrustes_align(small, ref, cols, scale=True)
        np.testing.assert_allclose(back[cols].to_numpy(), ref.to_numpy(), atol=1e-8)
        self.assertAlmostEqual(back.attrs["scale"], 1 / 0.7)
        self.assertNotIn("scale", procrustes_align(small, ref, cols).attrs)

    def test_cap_fold_series_and_errors(self):
        capped = item_holdout_features(self.ratings, items=self.items, svd_rank=2, cap=15, seed=0)
        self.assertLessEqual(capped.dev.groupby("item_id").size().max(), 15)
        again = item_holdout_features(self.ratings, items=self.items, svd_rank=2, cap=15, seed=0)
        self.assertTrue(capped.dev.equals(again.dev))
        f = pd.Series(np.arange(150) % 3, index=np.arange(150))
        r3 = item_holdout_features(self.ratings, svd_rank=2, folds=f)
        self.assertEqual(sorted(r3.folds.unique()), [0, 1, 2])
        self.assertNotIn("affinity", r3.dev.columns)
        few = item_holdout_features(self.ratings, svd_rank=0, min_item_ratings=40)
        self.assertTrue((self.ratings.item_id.value_counts()[few.item_folds.index] >= 40).all())
        self.assertEqual(few.svd_cols, [])
        with self.assertRaises(ValueError):
            item_holdout_features(self.ratings, folds=f, n_splits=3)
        with self.assertRaises(KeyError):
            item_holdout_features(self.ratings, user_col="member")
        with self.assertRaises(ValueError):
            item_holdout_features(self.ratings, svd_rank=500)

    def test_transform_uses_full_basis(self):
        res = self.res
        pairs = pd.DataFrame({"user_id": [0, 1, 2, 999], "item_id": [5, 5, 7, 5]})
        x = res.transform(pairs)
        np.testing.assert_allclose(x.loc[:2, res.svd_cols].to_numpy(),
                                   res.members.loc[[0, 1, 2], res.svd_cols].to_numpy())
        self.assertTrue(np.isnan(x.loc[3, "user_mean"]))
        self.assertTrue(set(res.features) <= set(x.columns))

    def test_group_transfer_on_items_beats_member_mean(self):
        res = self.res
        dev = res.dev
        kw = dict(features=res.features + ["item_id"], target="residual", group="item_id", folds=res.folds)
        gt = oof_predict(GroupTransfer("item_id", group_features=self.items, person_model="ridge"), dev, **kw)
        # member-mean baseline predicts rating = user_mean, i.e. residual 0
        rmse_gt = float(np.sqrt(np.mean((dev.residual - gt) ** 2)))
        rmse_um = float(np.sqrt(np.mean(dev.residual ** 2)))
        self.assertLess(rmse_gt, rmse_um)
        fr = pd.DataFrame({"t": dev.residual, "p": gt, "it": dev.item_id}).groupby("it").mean()
        self.assertGreater(spearmanr(fr.t, fr.p)[0], 0.5)          # item-level residual ranking


class TestMFBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ratings, _, _ = synth(seed=1)

    def test_warm_beats_member_mean(self):
        e = self.ratings.sample(frac=1.0, random_state=0)
        tr, te = e.iloc[: int(0.8 * len(e))], e.iloc[int(0.8 * len(e)):]
        um = tr.groupby("user_id").rating.mean()
        m = MFBaseline(rank=3, reg=5.0).fit(tr, tr.rating)
        rmse = lambda p: float(np.sqrt(np.mean((te.rating - p) ** 2)))   # noqa: E731
        self.assertLess(rmse(m.predict(te)), rmse(te.user_id.map(um).to_numpy()))
        b = MFBaseline(rank=0).fit(tr, tr.rating)
        self.assertLess(rmse(m.predict(te)), rmse(b.predict(te)))      # the factors add over biases only

    def test_cold_start_fallbacks(self):
        e = self.ratings
        tr = e[e.item_id >= 10]
        m = MFBaseline(rank=3).fit(tr, tr.rating)
        cold = pd.DataFrame({"user_id": [0, 0, 1, 10_000], "item_id": [1, 2, 1, 1]})
        p = m.predict(cold)
        u0 = m.users_.get_loc(0)
        self.assertAlmostEqual(p[0], m.mu_ + m.bu_[u0] + m.item_prior_)
        self.assertAlmostEqual(p[0], p[1])                             # same member, two new items
        self.assertAlmostEqual(p[3], m.mu_ + m.item_prior_)
        self.assertAlmostEqual(m.item_prior_, float(np.mean(m.bi_)))
        p2 = MFBaseline(rank=3, item_prior=0.0).fit(tr, tr.rating).predict(cold)
        self.assertNotAlmostEqual(p2[0], p[0])

    def test_in_oof_predict_item_held_out(self):
        e = self.ratings
        s = oof_predict(MFBaseline(rank=2, n_iter=3), e, features=["user_id", "item_id"], target="rating",
                        group="item_id", n_splits=4)
        self.assertTrue(np.isfinite(s).all())
        # item-held-out: every prediction is the member level plus one constant per fold
        d = pd.DataFrame({"u": e.user_id, "f": s.attrs["fold"], "p": s})
        self.assertTrue((d.groupby(["u", "f"]).p.nunique() == 1).all())


if __name__ == "__main__":
    unittest.main()
