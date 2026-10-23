"""NextActionRanker on a generic query x candidate table (person x time query against candidate cells), plus the
query_col / cand_col options of top1 / hit_at_k. The session schema stays the default (test_methods_next_action)."""
import unittest

import numpy as np
import pandas as pd

from simengine.methods.next_action import NextActionRanker, hit_at_k, top1
from simengine.methods._util import lightgbm_available

N_CELL = 12


def cell_table(n_person=40, n_time=40, seed=0, offset=0):
    """One row per (person x time query, candidate cell): ``hist`` = the person's share of past visits to the cell,
    ``dist`` = distance from the previous cell, ``zone`` = categorical cell type; ``y`` = 1 on the visited cell,
    drawn from 8 * hist - 1.5 * dist + 1.5 * 1[home cell on an evening query] + Gumbel noise."""
    rs = np.random.default_rng(seed)
    rows = []
    zone = np.array(["home", "work", "shop"])[np.arange(N_CELL) % 3]
    for p in range(n_person):
        pref = rs.dirichlet(np.full(N_CELL, 0.4))
        prev = int(rs.integers(N_CELL))
        for t in range(n_time):
            q = f"q{offset}_{p}_{t}"
            hist = pref
            dist = np.abs(np.arange(N_CELL) - prev) / 3.0
            evening = t % 4 == 3
            u = 8 * hist - 1.5 * dist + 1.5 * ((zone == "home") & evening) + rs.gumbel(size=N_CELL)
            pick = int(np.argmax(u))
            for c in range(N_CELL):
                rows.append((q, f"p{p}", t, c, float(hist[c]), float(dist[c]), zone[c], int(evening),
                             int(c == pick)))
            prev = pick
    return pd.DataFrame(rows, columns=["qid", "person", "t", "cell", "hist", "dist", "zone", "evening", "y"])


class TestGenericHelpers(unittest.TestCase):
    def test_top1_and_hit_at_k_take_column_names(self):
        c = pd.DataFrame({"qid": ["a", "a", "a", "b", "b"], "cell": [1, 2, 3, 1, 2], "y": [0, 1, 0, 0, 1]})
        s = [0.1, 0.9, 0.5, 0.8, 0.2]
        t = top1(c, s, query_col="qid", cand_col="cell")
        self.assertEqual(t.columns.tolist(), ["qid", "cell"])
        self.assertEqual(t.cell.tolist(), [2, 1])
        self.assertEqual(hit_at_k(c, s, ks=(1, 2), query_col="qid"), {"hit@1": 0.5, "hit@2": 1.0})

    def test_session_names_stay_default(self):
        c = pd.DataFrame({"ex": [0, 0], "key": ["x", "y"], "y": [0, 1]})
        self.assertEqual(top1(c, [0.0, 1.0]).key.tolist(), ["y"])
        self.assertEqual(hit_at_k(c, [0.0, 1.0], ks=(1,)), {"hit@1": 1.0})

    def test_generic_input_checks(self):
        tab = cell_table(3, 3)
        with self.assertRaises(KeyError):
            NextActionRanker(query_col="qid", cand_col="cell", features=["hist", "nope"]).fit(tab)
        with self.assertRaises(ValueError):
            NextActionRanker(query_col="qid", cand_col="cell", features=["hist"]).fit(pd.concat([tab, tab.head(1)]))
        with self.assertRaises(ValueError):
            NextActionRanker(query_col="qid", cand_col="cell").fit(tab)        # no features
        with self.assertRaises(KeyError):
            NextActionRanker(query_col="qid", features=["hist"]).fit(tab.drop(columns="y"))


@unittest.skipUnless(lightgbm_available(), "lightgbm not installed")
class TestGenericRanker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr = cell_table(40, 40, seed=0)
        cls.te = cell_table(20, 40, seed=1, offset=1)

    def test_ranks_person_time_queries(self):
        r = NextActionRanker(n_estimators=120, query_col="qid", cand_col="cell",
                             features=["hist", "dist", "zone", "evening"]).fit(self.tr)
        te, s = r.score(self.te)
        self.assertEqual(len(s), len(self.te))
        self.assertEqual(te["qid"].tolist(), self.te["qid"].tolist())      # rows stay in input order
        self.assertEqual(r.categorical, ["zone"])                           # object column -> categorical
        h = hit_at_k(te, s, ks=(1, 3), query_col="qid")
        h_hist = hit_at_k(self.te, self.te["hist"].to_numpy(), ks=(1,), query_col="qid")["hit@1"]
        h_dist = hit_at_k(self.te, -self.te["dist"].to_numpy(), ks=(1,), query_col="qid")["hit@1"]
        self.assertGreater(h["hit@1"], max(h_hist, h_dist))
        self.assertGreaterEqual(h["hit@3"], h["hit@1"])
        t = top1(te, s, query_col="qid", cand_col="cell")
        self.assertEqual(len(t), self.te.qid.nunique())

    def test_negative_sampling_and_seeds(self):
        r = NextActionRanker(n_estimators=40, query_col="qid", cand_col="cell", features=["hist", "dist"],
                             sample_negatives=True, top=2, random=2, prior_col="hist", n_seeds=2).fit(self.tr)
        self.assertEqual(len(r.models_), 2)
        self.assertLessEqual(r.n_train_rows_, self.tr.qid.nunique() * 5)
        te, s = r.score(self.te)
        self.assertGreater(hit_at_k(te, s, ks=(1,), query_col="qid")["hit@1"], 0.3)
        full = NextActionRanker(n_estimators=10, query_col="qid", features=["hist"]).fit(self.tr)
        self.assertEqual(full.n_train_rows_, len(self.tr))                 # generic default keeps every row

    def test_graded_relevance_and_unseen_category(self):
        tr = self.tr.assign(y=self.tr.y * 2 + (self.tr.dist == 0).astype(int))
        te = self.te.assign(zone=self.te.zone.where(self.te.cell != 0, "never_seen"))
        r = NextActionRanker(n_estimators=30, query_col="qid", features=["hist", "dist", "zone"]).fit(tr)
        _, s = r.score(te)
        self.assertTrue(np.isfinite(s).all())


if __name__ == "__main__":
    unittest.main()
