import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import simengine.abm.mobility as M
from simengine.bench.tasks.yjmob import YJMob

# HuMob 2023 README example of github.com/yahoojapan/geobleu (commit cdaa364): (d, t, x, y) and the printed values
GEN = [(60, 12, 84, 88), (60, 15, 114, 78), (60, 21, 121, 96), (61, 12, 78, 86), (61, 13, 89, 67), (61, 17, 97, 70),
       (61, 20, 96, 70), (61, 24, 111, 80), (61, 25, 114, 78), (61, 26, 99, 70), (61, 38, 77, 86), (62, 12, 77, 86),
       (62, 14, 102, 129), (62, 15, 104, 131), (62, 17, 106, 131), (62, 18, 104, 110)]
REF = [(60, 12, 82, 93), (60, 15, 114, 78), (60, 21, 116, 96), (61, 12, 82, 84), (61, 13, 89, 67), (61, 17, 97, 70),
       (61, 20, 91, 67), (61, 24, 109, 82), (61, 25, 110, 78), (61, 26, 99, 70), (61, 38, 77, 86), (62, 12, 77, 86),
       (62, 14, 97, 125), (62, 15, 104, 131), (62, 17, 106, 131), (62, 18, 103, 111)]


def _ref_pn(sys_, ans, n, beta):
    """Literal greedy matching as described in the GEO-BLEU reference (sort all pairs, take best, drop row/col)."""
    sg = [sys_[i:i + n] for i in range(len(sys_) - n + 1)]
    ag = [ans[i:i + n] for i in range(len(ans) - n + 1)]
    edges = []
    for i, a in enumerate(sg):
        for j, b in enumerate(ag):
            p = 1.0
            for (x1, y1), (x2, y2) in zip(a, b):
                p *= np.exp(-beta * np.sqrt((x2 - x1) ** 2. + (y2 - y1) ** 2.))
            edges.append(((i, j), p))
    edges.sort(key=lambda e: e[1], reverse=True)
    tot, ur, uc = 0.0, set(), set()
    for (i, j), p in edges:
        if i in ur or j in uc:
            continue
        tot += p
        ur.add(i)
        uc.add(j)
    return tot / len(sg)


def synthetic(n_users=30, days=20, seed=0):
    """Each person: a home and a work cell; home at night, work 9-17 on weekdays, sometimes a random errand."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        home, work = rng.integers(20, 180, 2), rng.integers(20, 180, 2)
        for d in range(days):
            for t in sorted(rng.choice(48, 12, replace=False)):
                xy = work if (d % 7 < 5 and 18 <= t < 34) else home
                if rng.random() < 0.1:
                    xy = xy + rng.integers(-5, 6, 2)
                rows.append((u + 100, d, int(t), int(xy[0]), int(xy[1])))
    return pd.DataFrame(rows, columns=["uid", "d", "t", "x", "y"])


class TestMetrics(unittest.TestCase):
    def test_reference_example_humob23(self):
        g = [r[2:] for r in GEN]
        r = [r[2:] for r in REF]
        s = M.user_scores([1] * len(GEN), [x[0] for x in GEN], [x[1] for x in GEN], g, r, "humob23")
        self.assertAlmostEqual(s.geobleu.iloc[0], 0.21733678721880598, places=12)
        self.assertAlmostEqual(s.dtw.iloc[0], 5.889002930255253, places=12)

    def test_reference_example_humob25(self):
        g = [r[2:] for r in GEN]
        r = [r[2:] for r in REF]
        s = M.user_scores([1] * len(GEN), [x[0] for x in GEN], [x[1] for x in GEN], g, r, "humob25")
        self.assertAlmostEqual(s.geobleu.iloc[0], 0.07556369896234784, places=12)

    def test_identical_and_order(self):
        a = np.array([[1, 1], [2, 2], [5, 9], [5, 9]])
        self.assertAlmostEqual(M.geobleu(a, a), 1.0)
        self.assertEqual(M.dtw(a, a), 0.0)
        far = a + 100
        self.assertLess(M.geobleu(far, a), 1e-10)
        # shuffled rows of a user give the same score (sorted by d, t inside)
        s1 = M.user_scores([7] * 4, [0, 0, 1, 1], [3, 1, 2, 5], a, a[::-1])
        o = [2, 0, 3, 1]
        s2 = M.user_scores([7] * 4, np.array([0, 0, 1, 1])[o], np.array([3, 1, 2, 5])[o], a[o], a[::-1][o])
        np.testing.assert_allclose(s1[["geobleu", "dtw"]], s2[["geobleu", "dtw"]])

    def test_batch_matches_literal_greedy(self):
        rng = np.random.default_rng(3)
        for _ in range(40):
            L = int(rng.integers(1, 9))
            a = rng.integers(0, 4, (L, 2))       # small grid -> many ties
            b = rng.integers(0, 4, (L, 2))
            pn = [_ref_pn([tuple(x) for x in a], [tuple(x) for x in b], n, 0.5) for n in range(1, min(3, L) + 1)]
            want = 0.0 if min(pn) <= 0 else float(np.exp(np.mean(np.log(pn))))
            got = M._geobleu_batch(a[None].astype(float), b[None].astype(float), 3, 0.5)[0]
            self.assertAlmostEqual(got, want, places=12)
            self.assertAlmostEqual(M.geobleu(a, b), want, places=12)

    def test_dtw_variants(self):
        a = np.array([[0, 0], [0, 2]])
        b = np.array([[0, 2], [0, 2], [0, 4]])
        # standard DTW: |0-2| + 0 + 0 + |2-4|, scale 1
        self.assertAlmostEqual(M.dtw(a, b, 1.0, free_start=False), 4.0)
        # 2023 rule: first DP row is 0, the prediction may start at any truth step
        self.assertLessEqual(M.dtw(a, b, 1.0, free_start=True), M.dtw(a, b, 1.0, free_start=False))
        self.assertAlmostEqual(M.dtw(a, b, 2.0, free_start=False), 2.0)
        # a prediction that only matches the end of the truth: free start skips the first truth step at no cost
        self.assertAlmostEqual(M.dtw([[0, 4]], [[0, 0], [0, 4]], 1.0, free_start=True), 0.0)
        self.assertAlmostEqual(M.dtw([[0, 4]], [[0, 0], [0, 4]], 1.0, free_start=False), 4.0)

    def test_unequal_lengths_literal(self):
        # 2025 normalisation (matched sum / predicted n-grams) and the brevity penalty for a short prediction
        rng = np.random.default_rng(11)
        for _ in range(40):
            a = rng.integers(0, 5, (int(rng.integers(1, 8)), 2))
            b = rng.integers(0, 5, (int(rng.integers(1, 8)), 2))
            nmax = min(3, len(a), len(b))
            pn = [_ref_pn([tuple(x) for x in a], [tuple(x) for x in b], n, 0.5) for n in range(1, nmax + 1)]
            bp = 1.0 if len(a) > len(b) else np.exp(1.0 - len(b) / len(a))
            want = 0.0 if min(pn) <= 0 else bp * float(np.exp(np.mean(np.log(pn))))
            self.assertAlmostEqual(M.geobleu(a, b), want, places=12)
        with self.assertRaises(ValueError):
            M.geobleu(np.zeros((0, 2)), [[1, 1]])

    def test_user_scores_refuses_unknown_metric(self):
        args = ([1, 1], [0, 0], [0, 1], [[1, 1], [2, 2]], [[1, 1], [2, 2]])
        for bad in (("geobleu", "dtwx"), (), ("accuracy",)):
            with self.assertRaises(ValueError):
                M.user_scores(*args, metrics=bad)
        s = M.user_scores(*args, metrics="dtw")                  # a bare string is one metric
        self.assertEqual(list(s.columns), ["uid", "n_days", "dtw"])

    def test_cells_roundtrip(self):
        xy = np.array([[1, 1], [200, 200], [37, 150]])
        c = M.to_cell(xy[:, 0], xy[:, 1])
        self.assertEqual(c.tolist(), [0, 39999, 36 * 200 + 149])
        np.testing.assert_array_equal(M.from_cell(c), xy)


class TestBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = synthetic()
        cls.train, cls.test = df[df.d < 14].reset_index(drop=True), df[df.d >= 14].reset_index(drop=True)
        cls.q = cls.test[["uid", "d", "t"]]
        cls.truth = M.to_cell(cls.test.x, cls.test.y)

    def score(self, c):
        s = M.user_scores(self.q.uid, self.q.d, self.q.t, M.from_cell(c), M.from_cell(self.truth))
        return s.geobleu.mean(), s.dtw.mean()

    def test_shapes_and_range(self):
        preds = M.mobility_baselines(self.train, self.q, seed=1)
        self.assertEqual(set(preds), {"slot_mode", "global_slot_mode", "markov_argmax", "markov_sample", "epr"})
        for k, p in preds.items():
            self.assertEqual(len(p), len(self.q), k)
            self.assertTrue(((p >= 0) & (p < M.NC)).all(), k)

    def test_slot_mode_beats_global(self):
        g_user, d_user = self.score(M.slot_mode(self.train, self.q, window=1))
        g_glob, d_glob = self.score(M.global_slot_mode(self.train, self.q))
        self.assertGreater(g_user, g_glob)
        self.assertLess(d_user, d_glob)

    def test_query_order_preserved(self):
        perm = np.random.default_rng(0).permutation(len(self.q))
        for make in (lambda: M.MarkovMobility("argmax"), lambda: M.EPRMobility(seed=4)):
            a = make().fit(self.train).predict(self.q)
            b = make().fit(self.train).predict(self.q.iloc[perm])
            np.testing.assert_array_equal(a[perm], b)

    def test_seeded(self):
        a = M.EPRMobility(seed=2, waiting="powerlaw").fit(self.train).predict(self.q)
        b = M.EPRMobility(seed=2, waiting="powerlaw").fit(self.train).predict(self.q)
        np.testing.assert_array_equal(a, b)
        c = M.MarkovMobility("sample", seed=2).fit(self.train).predict(self.q)
        self.assertEqual(len(c), len(self.q))

    def test_epr_explores_less_with_small_rho(self):
        new = []
        seen = set(zip(self.train.uid, M.to_cell(self.train.x, self.train.y)))
        for rho in (0.0, 1.0):
            p = M.EPRMobility(rho=rho, gamma=0.0, seed=0).fit(self.train).predict(self.q)
            new.append(np.mean([(u, c) not in seen for u, c in zip(self.q.uid, p)]))
        self.assertEqual(new[0], 0.0)              # rho = 0: only returns / first-of-day cells
        self.assertGreater(new[1], 0.0)

    def test_unknown_user_falls_back(self):
        q = pd.DataFrame({"uid": [999999, 999999], "d": [14, 14], "t": [3, 20]})
        tab = M.global_slot_table(self.train)
        for p in (M.slot_mode(self.train, q), M.EPRMobility().fit(self.train).predict(q),
                  M.MarkovMobility().fit(self.train).predict(q)):
            np.testing.assert_array_equal(p, tab[[3, 20]])

    def test_markov_follows_chain(self):
        # every day 1 -> 2 -> 3 at slots 10, 20, 30; a query at slots 5, 25, 40 starts at the slot mode and follows
        tr = pd.DataFrame({"uid": 1, "d": np.repeat(np.arange(5), 3), "t": np.tile([10, 20, 30], 5),
                           "x": np.tile([1, 1, 1], 5), "y": np.tile([2, 3, 4], 5)})
        q = pd.DataFrame({"uid": 1, "d": [6, 6, 6, 7], "t": [10, 25, 40, 30]})
        p = M.MarkovMobility().fit(tr).predict(q)
        self.assertEqual(M.from_cell(p)[:, 1].tolist(), [2, 3, 4, 4])     # day 7 restarts at slot 30's mode
        self.assertEqual(M.MarkovMobility("sample", seed=3).fit(tr).predict(q).tolist(), p.tolist())

    def test_empty_query(self):
        q = self.q.iloc[:0]
        for p in (M.slot_mode(self.train, q), M.global_slot_mode(self.train, q),
                  M.MarkovMobility().fit(self.train).predict(q), M.EPRMobility().fit(self.train).predict(q)):
            self.assertEqual(len(p), 0)

    def test_epr_day_starts_at_first_of_day_cell(self):
        tr = self.train.sort_values(["uid", "d", "t"])
        firsts = set(zip(tr.groupby(["uid", "d"]).uid.first(), M.to_cell(tr.groupby(["uid", "d"]).x.first(),
                                                                          tr.groupby(["uid", "d"]).y.first())))
        p = M.EPRMobility(seed=5).fit(self.train).predict(self.q)
        first_row = ~self.q.duplicated(["uid", "d"]).to_numpy()            # the test rows are (uid, d, t)-sorted
        self.assertTrue(all((u, c) in firsts for u, c in zip(self.q.uid[first_row], p[first_row])))

    def test_epr_distinct_cells_grow_as_paper(self):
        # Song et al. 2010: P_new = rho S^-gamma gives S ~ n^(1/(1+gamma)). One person moving at every record, no day
        # reset, a buffer long enough to remember every explored cell; slope of log S vs log n over 3 seeds.
        tr = pd.DataFrame({"uid": 1, "d": 0, "t": np.arange(4), "x": [100, 101, 100, 101], "y": 100})
        n = 3000
        q = pd.DataFrame({"uid": 1, "d": np.arange(n) // 48, "t": np.arange(n) % 48})
        k = np.unique(np.geomspace(50, n - 1, 25).astype(int))
        slope = {}
        for g in (0.21, 0.6):
            sl = []
            for seed in range(3):
                c = M.EPRMobility(gamma=g, day_reset=False, buffer=4096, seed=seed).fit(tr).predict(q)
                first = np.sort(np.unique(c, return_index=True)[1])
                sl.append(np.polyfit(np.log(k), np.log(np.searchsorted(first, k, side="right")), 1)[0])
            slope[g] = np.mean(sl)
        self.assertTrue(0.7 < slope[0.21] < 0.9, slope)           # paper 1 / 1.21 = 0.83
        self.assertTrue(0.5 < slope[0.6] < 0.7, slope)            # 1 / 1.6 = 0.63
        # rho = 1, gamma = 0: every move explores, so (almost) every row is a new cell
        c = M.EPRMobility(rho=1.0, gamma=0.0, day_reset=False, seed=0).fit(tr).predict(q.iloc[:500])
        self.assertGreater(len(np.unique(c)), 450)

    def test_clean(self):
        df = pd.DataFrame({"uid": [1, 1, 1, 2], "d": [0, 0, 0, 0], "t": [1, 1, 2, 1], "x": [5, 6, 7, 999],
                           "y": [5, 6, 7, 999]})
        c = M.clean(df)
        self.assertEqual(c.uid.tolist(), [1, 1])
        self.assertEqual(c.x.tolist(), [5, 7])


class TestBenchTask(unittest.TestCase):
    def test_synthetic_file(self):
        df = synthetic(n_users=12, days=75, seed=5)
        df.loc[df.uid == 100, ["x", "y"]] = 999                  # a masked user is dropped
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "yjmob100k"))
            df.to_csv(os.path.join(tmp, "yjmob100k", "yjmob100k-dataset1.csv.gz"), index=False)
            task = YJMob(users=8, seed=0, epr=True)
            data = task.load(tmp)
            self.assertEqual(data.meta["users"], 8)
            self.assertEqual(data.meta["dropped_masked_users"], 1)
            self.assertTrue((data.train.d < 60).all() and (data.test.d >= 60).all())
            self.assertNotIn(100, set(data.train.uid))
            frame = task.run(data=data)
            sc = task.score_all(frame, B=50)
            self.assertEqual(set(sc.method), {"slot_mode", "global_slot_mode", "markov_argmax", "markov_sample",
                                              "epr"})
            g = sc[(sc.metric == "geobleu")].set_index("method").value
            self.assertGreater(g["slot_mode"], g["global_slot_mode"])
            bad = np.full(len(data.test), -1)                               # not a cell code
            with self.assertRaises(ValueError):
                task.score(task.frame(data, {"bad": bad}), "bad")
            dev = YJMob(users="all", split="dev", markov=False, epr=False).load(tmp)
            self.assertEqual(int(dev.test.d.max()), 59)
            with self.assertRaises(ValueError):
                YJMob(colour="red")

    @unittest.skipUnless(os.environ.get("SIMENGINE_DATA"), "needs $SIMENGINE_DATA with yjmob100k/")
    def test_real_data_small(self):
        task = YJMob(users=50, seed=0, epr=False, markov=False)
        data = task.load()
        frame = task.run(data=data)
        s = task.score(frame, "slot_mode", B=50)
        self.assertTrue(0 < s["value"] < 1)


if __name__ == "__main__":
    unittest.main()
