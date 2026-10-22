"""Regression tests for the review of GroupTransfer (review #7: group_features index that does not match the group
labels; review #8: documented negative group-level bias of pooled group-held-out predictions under the null)."""
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from simengine.methods import GroupTransfer, oof_predict


def _frame(seed=0, G=30, m=30, signal=True):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(G), m)
    eff = rng.normal(0, 1, G)
    x1 = rng.normal(size=G * m)
    y = eff[g] + 0.5 * x1 + rng.normal(size=G * m)
    df = pd.DataFrame({"country": g, "x1": x1, "y": y})
    prior = pd.DataFrame({"prior": pd.Series(y).groupby(g).mean()})     # oracle group prior, int index
    return df, prior


def _group_rho(df, s):
    gm = df.assign(p=s).groupby("country")[["y", "p"]].mean()
    return spearmanr(gm.y, gm.p)[0]


class TestGroupFeatureIndex(unittest.TestCase):
    def test_str_index_on_int_labels_is_matched_and_warned(self):
        df, prior = _frame()
        ok = oof_predict(GroupTransfer("country", group_features=prior, person_model="ridge"), df,
                         features=["country", "x1"], target="y", group="country")
        p2 = prior.copy()
        p2.index = p2.index.map(str)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = oof_predict(GroupTransfer("country", group_features=p2, person_model="ridge"), df,
                            features=["country", "x1"], target="y", group="country")
        self.assertTrue(any("as strings" in str(x.message) for x in w))
        np.testing.assert_allclose(s.to_numpy(), ok.to_numpy())
        self.assertGreater(_group_rho(df, s), 0.9)

    def test_unmatched_index_warns(self):
        df, prior = _frame()
        bad = prior.copy()
        bad.index = [f"zz{i}" for i in range(len(bad))]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GroupTransfer("country", group_features=bad, person_model=None).fit(df[["country", "x1"]], df.y)
        self.assertTrue(any("have a row in group_features" in str(x.message) for x in w))

    def test_matching_index_is_silent(self):
        df, prior = _frame()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GroupTransfer("country", group_features=prior, person_model=None).fit(df[["country", "x1"]], df.y)
        self.assertFalse([x for x in w if "group_features" in str(x.message)])


class TestNullBiasDocumented(unittest.TestCase):
    def test_pooled_group_oof_is_negative_under_the_null(self):
        rs = []
        for seed in range(6):
            df, _ = _frame(seed, G=40, m=20)
            s = oof_predict(GroupTransfer("country", person_model=None), df, features=["country", "x1"],
                            target="y", group="country")
            rs.append(_group_rho(df, s))
        self.assertLess(np.mean(rs), 0.0)           # documented in oof_predict: compare with a baseline, not with 0
        self.assertIn("NEGATIVE under the null", oof_predict.__doc__)


if __name__ == "__main__":
    unittest.main()
