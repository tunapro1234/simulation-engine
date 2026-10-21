"""Conditional MNL on simulated conjoint choices."""
import unittest

import numpy as np
import pandas as pd

from simengine.methods import MNL, evaluate_mnl, null_scores, person_holdout

BETA = {"price": -1.5, "quality": 1.0}
ASC = {1: 0.0, 2: 0.5, 3: -0.5}


def choices(persons=250, tasks=8, seed=0):
    rs = np.random.default_rng(seed)
    rows, tid = [], 0
    for p in range(persons):
        for _ in range(tasks):
            price, qual = rs.uniform(0, 2, 3), rs.uniform(0, 2, 3)
            v = BETA["price"] * price + BETA["quality"] * qual + np.array(list(ASC.values()))
            pr = np.exp(v - v.max())
            pick = rs.choice(3, p=pr / pr.sum())
            for j in range(3):
                rows.append({"birey_id": p, "gorev_id": tid, "alternatif_id": j + 1, "price": price[j] * 100,
                             "quality": qual[j], "secildi": int(j == pick), "mevcut": 1})
            tid += 1
    return pd.DataFrame(rows)


class TestMNL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = choices()

    def test_recovers_coefficients(self):
        m = MNL(["price", "quality"], scale={"price": 100}).fit(self.df)
        self.assertTrue(m.converged_)
        self.assertAlmostEqual(m.coef_["price"], BETA["price"], delta=0.25)
        self.assertAlmostEqual(m.coef_["quality"], BETA["quality"], delta=0.25)
        self.assertAlmostEqual(m.coef_["ASC_2"], 0.5, delta=0.25)
        P = m.predict_proba(self.df.head(30))
        np.testing.assert_allclose(P.reshape(-1, 3).sum(1), 1.0)
        self.assertEqual(len(m.predict(self.df.head(30))), 10)

    def test_holdout_beats_null_and_asc(self):
        out = evaluate_mnl(self.df, ["price", "quality"], scale={"price": 100})
        self.assertGreater(out["mnl"]["acc"], out["asc_only"]["acc"])
        self.assertGreater(out["mnl"]["acc"], out["null"]["acc"])
        self.assertGreater(out["mnl"]["ll_per_task"], out["null"]["ll_per_task"])
        self.assertAlmostEqual(out["null"]["acc"], 1 / 3)
        self.assertEqual(out["n"]["test_persons"], 50)

    def test_availability(self):
        d = self.df.rename(columns={"mevcut": "available"}).copy()
        d.loc[d.alternatif_id == 3, "available"] = 0
        tr, te = person_holdout(d)
        m = MNL(["price", "quality"], scale={"price": 100}).fit(tr)
        self.assertNotIn("ASC_3", m.coef_.index)
        p = m.predict_proba(te).reshape(-1, 3)
        self.assertLess(p[:, 2].max(), 1e-6)
        self.assertAlmostEqual(null_scores(te)["acc"], 0.5)


if __name__ == "__main__":
    unittest.main()
