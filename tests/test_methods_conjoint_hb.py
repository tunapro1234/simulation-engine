"""Hierarchical Bayes MNL and the xlogit mixed-logit wrapper on a synthetic heterogeneous-taste panel."""
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from simengine.methods.conjoint import MNL
from simengine.methods.conjoint_hb import HBMNL, MixedLogit

try:
    import xlogit  # noqa: F401
    HAVE_XLOGIT = True
except ImportError:
    HAVE_XLOGIT = False

MU, SD = np.array([-1.0, 0.5]), np.array([0.6, 2.0])   # quality taste flips sign across persons


def panel(persons=120, tasks=16, seed=0):
    rs = np.random.default_rng(seed)
    rows, tid = [], 0
    for p in range(persons):
        b = MU + rs.standard_normal(2) * SD
        for t in range(tasks):
            X = rs.uniform(0, 2, (3, 2))
            v = X @ b
            pr = np.exp(v - v.max())
            k = rs.choice(3, p=pr / pr.sum())
            for j in range(3):
                rows.append({"birey_id": p, "gorev_id": tid, "alternatif_id": j + 1, "price": X[j, 0] * 100,
                             "quality": X[j, 1], "secildi": int(j == k), "t": t})
            tid += 1
    return pd.DataFrame(rows)


class TestHBMNL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = panel()
        cls.tr = df[df["t"] < 12].drop(columns="t")
        cls.te = df[df["t"] >= 12].drop(columns="t")       # same persons, held-out tasks
        cls.hb = HBMNL(["price", "quality"], scale={"price": 100}, draws=200, burn=300, seed=0).fit(cls.tr)
        cls.mnl = MNL(["price", "quality"], scale={"price": 100}).fit(cls.tr)

    def test_person_holdout_beats_pooled_mnl(self):
        ll_hb, ll_mnl = self.hb.log_likelihood(self.te), self.mnl.log_likelihood(self.te)
        self.assertGreater(ll_hb, ll_mnl + 0.1)
        self.assertGreater(self.hb.accuracy(self.te), self.mnl.accuracy(self.te))

    def test_population_recovered_and_new_person_close_to_mnl(self):
        self.assertAlmostEqual(self.hb.mu_["price"], MU[0], delta=0.3)
        self.assertAlmostEqual(self.hb.mu_["quality"], MU[1], delta=0.5)
        self.assertGreater(self.hb.sigma_.loc["quality", "quality"], 1.5)
        new = panel(persons=40, tasks=4, seed=7).drop(columns="t")
        new["birey_id"] += 1000
        new["gorev_id"] += 10 ** 6
        self.assertGreater(self.hb.log_likelihood(new, new=True), self.mnl.log_likelihood(new) - 0.05)
        with self.assertRaises(ValueError):
            self.hb.predict_person(new)
        p = self.hb.predict_person(new, unknown="population")
        np.testing.assert_allclose(p, self.hb.predict_new(new))

    def test_shapes_and_probabilities(self):
        hb = self.hb
        self.assertEqual(hb.draws_.shape, (200, 120, 4))
        self.assertEqual(list(hb.names_), ["price", "quality", "ASC_2", "ASC_3"])
        self.assertEqual(hb.beta_.shape, (120, 4))
        self.assertTrue(0.1 < hb.acceptance_ < 0.6)
        p = hb.predict_person(self.te)
        sums = pd.Series(p).groupby(self.te["gorev_id"].to_numpy()).sum()
        np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-9)

    def test_deterministic_under_seed(self):
        small = self.tr[self.tr["birey_id"] < 30]
        a = HBMNL(["price", "quality"], scale={"price": 100}, draws=30, burn=50, seed=3).fit(small)
        b = HBMNL(["price", "quality"], scale={"price": 100}, draws=30, burn=50, seed=3).fit(small)
        c = HBMNL(["price", "quality"], scale={"price": 100}, draws=30, burn=50, seed=4).fit(small)
        np.testing.assert_array_equal(a.draws_, b.draws_)
        np.testing.assert_array_equal(a.predict_new(small), b.predict_new(small))
        self.assertFalse(np.array_equal(a.draws_, c.draws_))

    def test_unavailable_rows_and_validation(self):
        small = self.tr[self.tr["birey_id"] < 20].copy()
        small["mevcut"] = 1
        drop = (small["alternatif_id"] == 3) & (small["secildi"] == 0) & (small["gorev_id"] % 2 == 0)
        small.loc[drop, "mevcut"] = 0
        hb = HBMNL(["price", "quality"], scale={"price": 100}, draws=20, burn=40).fit(small)
        p = hb.predict_person(small)
        self.assertTrue((p[small["mevcut"].to_numpy() == 0] < 1e-12).all())
        bad = small.copy()
        bad.loc[bad.index[0], "secildi"] = 1 - bad.loc[bad.index[0], "secildi"]
        with self.assertRaises(ValueError):
            HBMNL(["price", "quality"]).fit(bad)
        with self.assertRaises(ValueError):
            HBMNL(["price", "quality"]).fit(small.drop(columns="birey_id"))


class TestMixedLogit(unittest.TestCase):
    def test_import_error_is_clear(self):
        with mock.patch.dict(sys.modules, {"xlogit": None}):
            with self.assertRaisesRegex(ImportError, "xlogit"):
                MixedLogit(["price"]).fit(panel(persons=5, tasks=2).drop(columns="t"))

    @unittest.skipUnless(HAVE_XLOGIT, "xlogit not installed")
    def test_unsorted_panel_fits_and_scores(self):
        df = panel(persons=80, tasks=12, seed=1)
        tr, te = df[df["birey_id"] < 60].drop(columns="t"), df[df["birey_id"] >= 60].drop(columns="t")
        shuffled = tr.sample(frac=1.0, random_state=5)               # rows not contiguous per person
        kw = dict(randvars={"price": "n", "quality": "n"}, scale={"price": 100}, n_draws=100)
        a = MixedLogit(["price", "quality"], **kw).fit(shuffled)
        b = MixedLogit(["price", "quality"], **kw).fit(tr)
        np.testing.assert_allclose(a.coef_.to_numpy(), b.coef_.to_numpy(), rtol=1e-6, atol=1e-8)
        self.assertIn("sd.quality", a.coef_.index)
        self.assertGreater(abs(a.coef_["sd.quality"]), 1.0)
        p = a.predict_proba(te.sample(frac=1.0, random_state=2))
        self.assertEqual(len(p), len(te))
        self.assertGreater(a.log_likelihood(te), -np.log(3))


if __name__ == "__main__":
    unittest.main()
