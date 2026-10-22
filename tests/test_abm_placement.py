import itertools
import os
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

import simengine.abm.placement as pl
from simengine.abm.diffusion import power_kernel


def random_instance(n_users=60, n_cells=12, density=0.15, seed=0, counts=False):
    rng = np.random.default_rng(seed)
    A = (rng.random((n_users, n_cells)) < density).astype(float)
    if counts:
        A *= rng.integers(1, 4, A.shape)
    return sparse.csc_matrix(A)


def brute_force(M, k=None, costs=None, budget=None, r=1):
    n = M.shape[1]
    best = 0.0
    for size in range(1, (k or n) + 1):
        for S in itertools.combinations(range(n), size):
            if costs is not None and sum(costs[j] for j in S) > budget + 1e-12:
                continue
            best = max(best, pl.coverage(M, S, r=r))
    return best


class TestProtocol(unittest.TestCase):
    def test_aliases_and_array(self):
        a = np.array([[0, 0, 1, 3, 4], [1, 0, 2, 5, 6]])
        df = pd.DataFrame(a, columns=["uid", "d", "t", "x", "y"])
        for t in (a, df, {"uid": a[:, 0], "d": a[:, 1], "t": a[:, 2], "x": a[:, 3], "y": a[:, 4]}):
            c = pl.traj_columns(t)
            self.assertEqual(sorted(c), sorted(pl.COLS))
            np.testing.assert_array_equal(c["x"], [3, 5])
        with self.assertRaises(ValueError):
            pl.traj_columns(np.zeros((3, 4)))

    def test_grid_ignores_mask_and_roundtrips(self):
        t = pd.DataFrame(dict(user=[0, 0, 1, 1], day=0, slot=[0, 1, 0, 1], x=[1, 5, 999, 3], y=[1, 4, 999, 2]))
        g = pl.CellGrid.fit(t, size=2)
        self.assertEqual((g.x0, g.y0, g.nx, g.ny), (1, 1, 3, 2))
        cells = g.cell(np.array([1, 2, 5, 999]), np.array([1, 2, 4, 999]))
        np.testing.assert_array_equal(cells, [0, 0, 5, -1])
        np.testing.assert_array_equal(g.cell(*g.xy([0, 4, 5]).T), [0, 4, 5])


class TestExposures(unittest.TestCase):
    def setUp(self):
        # user 0: cell A three pings on day 0, one on day 1; user 1: cell B once; user 2 only outside the grid
        self.t = pd.DataFrame(dict(user=[0, 0, 0, 0, 1, 2], day=[0, 0, 0, 1, 0, 0], slot=[0, 1, 2, 0, 0, 0],
                                   x=[1, 1, 1, 1, 2, 999], y=[1, 1, 1, 1, 1, 999]))
        self.g = pl.CellGrid(1, 1, 2, 1)

    def test_units(self):
        users = np.array([0, 1, 2])
        for unit, a in (("slot", 4), ("day", 2), ("user", 1)):
            ex = pl.exposures(self.t, self.g, unit=unit, users=users)
            self.assertEqual(ex.M.shape, (3, 2))
            self.assertEqual(ex.M[0, 0], a)
            self.assertEqual(ex.M[1, 1], 1)
            self.assertEqual(ex.M[2].sum(), 0)
        np.testing.assert_array_equal(pl.exposures(self.t, self.g, users=users).active, [True, True, False])
        self.assertEqual(pl.exposures(self.t, self.g, days=[1], users=users).M.sum(), 1)
        with self.assertRaises(ValueError):
            pl.exposures(self.t, self.g, unit="week")
        self.assertEqual(pl.exposures(self.t, self.g, users=np.array([], np.int64)).M.shape, (0, 2))

    def test_reach_frequency(self):
        ex = pl.exposures(self.t, self.g, users=np.array([0, 1, 2]))
        rf = pl.reach_frequency(ex.M, [0, 1], min_freq=2)
        self.assertEqual((rf["reach"], rf["effective_reach"], rf["audience"]), (2, 1, 3))
        self.assertAlmostEqual(rf["frequency"], 1.5)
        self.assertAlmostEqual(rf["grp"], 100.0)
        self.assertEqual(rf["hist"], [1, 1, 1])
        self.assertEqual(pl.reach_frequency(ex.M, [0, 1], users=ex.active)["audience"], 2)

    def test_home_cells(self):
        h = pl.home_cells(self.t, self.g, np.array([0, 1, 2, 7]), night=[0])
        np.testing.assert_array_equal(h, [0, 1, -1, -1])


class TestGreedy(unittest.TestCase):
    def test_lazy_equals_plain(self):
        for seed in range(6):
            M = random_instance(seed=seed, counts=seed % 2 == 1, n_cells=30, n_users=200)
            for r in (1, 2):
                w = None if seed < 3 else np.random.default_rng(seed).integers(1, 5, M.shape[0])
                a = pl.greedy_placement(M, 6, r=r, weights=w, lazy=True)
                b = pl.greedy_placement(M, 6, r=r, weights=w, lazy=False)
                self.assertEqual(a.cells, b.cells)
                self.assertAlmostEqual(a.value, b.value)
                self.assertLessEqual(a.evaluations, b.evaluations)

    def test_guarantee_and_bound(self):
        for seed in range(5):
            M = random_instance(seed=seed, n_cells=10, n_users=40, density=0.2)
            for r in (1, 2):
                p = pl.greedy_placement(M, 3, r=r)
                opt = brute_force(M, k=3, r=r)
                self.assertGreaterEqual(p.value + 1e-9, (1 - 1 / np.e) * opt)
                self.assertLessEqual(opt, p.bound + 1e-9)
                self.assertAlmostEqual(p.value, pl.coverage(M, p.cells, r=r))
                # Theorem 4 holds for any placement
                other = pl.top_visited(M, 3)
                self.assertLessEqual(opt, pl.online_bound(M, other, k=3, r=r) + 1e-9)

    def test_greedy_value_monotone_gains(self):
        M = random_instance(seed=3, n_cells=40, n_users=300)
        p = pl.greedy_placement(M, 10)
        self.assertTrue(all(a >= b for a, b in zip(p.gains, p.gains[1:])))
        self.assertAlmostEqual(sum(p.gains), p.value)

    def test_stops_at_zero_gain_and_fill(self):
        M = sparse.csc_matrix(np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], float))
        p = pl.greedy_placement(M, 3)
        self.assertEqual(p.cells, [0])
        f = pl.greedy_placement(M, 3, fill=True)
        self.assertEqual(f.cells[0], 0)
        self.assertEqual(len(f.cells), 3)
        self.assertEqual(f.extra["filled"], f.cells[1:])

    def test_candidates(self):
        M = random_instance(seed=1, n_cells=20)
        p = pl.greedy_placement(M, 4, candidates=[1, 3, 5, 7, 9])
        self.assertTrue(set(p.cells) <= {1, 3, 5, 7, 9})

    def test_lazy_equals_plain_on_rounding_boundary(self):
        # a gain whose 9th-decimal rounding differs between Python round() and np.round: re-evaluated lazily, it
        # must still tie with (and lose to the lower index against) the plain greedy's key
        a = 9.2301163515
        w = np.array([a, float(np.round(a, 9)), 100.0])
        M = sparse.csc_matrix(np.eye(3))
        self.assertEqual(pl.greedy_placement(M, 2, weights=w).cells,
                         pl.greedy_placement(M, 2, weights=w, lazy=False).cells)

    def test_duplicates_and_stored_zeros(self):
        # column 0 stores user 0 three times (true reach 1), column 1 reaches users 1 and 2
        M = sparse.csc_matrix((np.ones(5), np.array([0, 0, 0, 1, 2]), np.array([0, 3, 5])), shape=(3, 2))
        for lazy in (True, False):
            p = pl.greedy_placement(M, 1, lazy=lazy)
            self.assertEqual((p.cells, p.value), ([1], 2.0))
        self.assertEqual(M.nnz, 5)                                               # caller's matrix untouched
        Z = sparse.csc_matrix((np.array([0., 0., 0., 1.]), np.array([0, 1, 2, 3]), np.array([0, 3, 4])), shape=(4, 2))
        self.assertEqual(pl.top_visited(Z, 1, by="visitors"), [1])
        self.assertEqual(pl.random_cells(Z, 2), [1])

    def test_bound_over_candidates(self):
        M = random_instance(seed=4, n_cells=12, n_users=50, density=0.2)
        cand = [0, 2, 4, 6, 8]
        p = pl.greedy_placement(M, 2, candidates=cand)
        opt = brute_force(M[:, cand], k=2)
        self.assertLessEqual(opt, p.bound + 1e-9)
        self.assertLessEqual(p.bound, pl.online_bound(M, p.cells, k=2) + 1e-9)

    def test_budget_celf(self):
        # the paper's counterexample (Sec. 3.1, eps = 1, B = 10): ratios 2 and 1; benefit/cost greedy alone picks the
        # cheap cell and cannot afford the big one
        B = 10.0
        M = sparse.csc_matrix(np.r_[np.c_[np.ones(2), np.zeros(2)], np.c_[np.zeros(10), np.ones(10)]])
        costs = np.array([1.0, B])
        p = pl.greedy_placement(M, costs=costs, budget=B)
        self.assertEqual(p.cells, [1])
        self.assertEqual(p.extra["rule"], "unit_cost")
        self.assertEqual(p.extra["runs"]["benefit_cost"], 2.0)
        for seed in range(4):
            M = random_instance(seed=seed, n_cells=9, n_users=40, density=0.25)
            c = np.random.default_rng(seed).uniform(0.5, 3.0, 9)
            for lazy in (True, False):
                q = pl.greedy_placement(M, costs=c, budget=4.0, lazy=lazy)
                self.assertLessEqual(c[q.cells].sum(), 4.0 + 1e-9)
                opt = brute_force(M, costs=c, budget=4.0)
                self.assertGreaterEqual(q.value + 1e-9, 0.5 * (1 - 1 / np.e) * opt)
                self.assertLessEqual(opt, q.bound + 1e-9)
            self.assertEqual(pl.greedy_placement(M, costs=c, budget=4.0).cells,
                             pl.greedy_placement(M, costs=c, budget=4.0, lazy=False).cells)
        with self.assertRaises(ValueError):
            pl.greedy_placement(M)
        with self.assertRaises(ValueError):
            pl.greedy_placement(M, costs=np.zeros(9), budget=1.0)


class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.t = pl.toy_trajectories(n_users=300, n_days=10, seed=1)
        self.g = pl.CellGrid.fit(self.t, size=2)
        self.ex = pl.exposures(self.t, self.g)

    def test_top_and_random(self):
        M = self.ex.M
        top = pl.top_visited(M, 5)
        vis = np.asarray(M.sum(0)).ravel()
        self.assertEqual(sorted(vis[top]), sorted(np.sort(vis)[-5:]))
        self.assertEqual(len(set(pl.top_visited(M, 5, by="visitors"))), 5)
        r = pl.random_cells(M, 5, seed=0)
        self.assertEqual(r, pl.random_cells(M, 5, seed=0))
        self.assertTrue((np.diff(M.indptr)[r] > 0).all())
        c = np.full(M.shape[1], 2.0)
        self.assertEqual(len(pl.top_visited(M, costs=c, budget=7.0)), 3)
        with self.assertRaises(ValueError):
            pl.top_visited(M, 3, by="pings")

    def test_greedy_beats_top_visits_on_fixture(self):
        k = 4
        g = pl.greedy_placement(self.ex.M, k)
        self.assertGreater(g.value, pl.coverage(self.ex.M, pl.top_visited(self.ex.M, k)))
        self.assertGreater(g.value, pl.coverage(self.ex.M, pl.random_cells(self.ex.M, k)))


class TestHoldout(unittest.TestCase):
    def test_frame(self):
        t = pl.toy_trajectories(n_users=250, n_days=12, seed=2)
        g = pl.CellGrid.fit(t, size=2)
        df = pl.holdout(t, g, train_days=range(8), test_days=range(8, 12), ks=(3, 6), random_seeds=3, n_boot=200)
        self.assertEqual(len(df), 2 * 5)
        self.assertEqual(set(df.method), {"lazy_greedy", "top_visits", "top_visitors", "random", "oracle_test"})
        for k, sub in df.groupby("k"):
            s = sub.set_index("method")
            self.assertGreater(s.loc["lazy_greedy", "test_reach_share"], s.loc["random", "test_reach_share"])
            self.assertTrue((sub.diff_lo <= sub.diff_hi).all())
            self.assertTrue(((sub.test_reach_share >= 0) & (sub.test_reach_share <= 1)).all())
        self.assertFalse("oracle_test" in set(pl.holdout(t, g, train_days=range(8), test_days=range(8, 12), ks=(3,),
                                                         random_seeds=1, n_boot=50, oracle=False).method))


class TestAdoption(unittest.TestCase):
    def setUp(self):
        self.t = pl.toy_trajectories(n_users=300, n_days=7, seed=3)
        self.g = pl.CellGrid.fit(self.t, size=2)
        ex = pl.exposures(self.t, self.g)
        self.home = pl.home_cells(self.t, self.g, ex.users)
        self.freq = np.asarray(ex.M[:, pl.greedy_placement(ex.M, 3).cells].sum(1)).ravel()

    def test_no_word_of_mouth_matches_exposure_model(self):
        run = pl.adoption(self.freq, self.home, self.g, p_exposure=0.05, q=0.0, steps=3, runs=400, seed=0)
        self.assertTrue((run.from_wom == 0).all())
        self.assertTrue((run.curve == run.curve[:, :1]).all())
        expected = (1 - 0.95 ** self.freq).sum()
        self.assertAlmostEqual(run.from_ad.mean() / expected, 1.0, delta=0.05)

    def test_word_of_mouth_and_kernel(self):
        a = pl.adoption(self.freq, self.home, self.g, q=0.5, steps=6, runs=30, seed=1)
        self.assertTrue((np.diff(a.curve, axis=1) >= 0).all())
        self.assertGreater(a.from_wom.mean(), 0)
        decayed = pl.adoption(self.freq, self.home, self.g, q=0.5, steps=6, runs=30, seed=1,
                              kernel=power_kernel(1, 0.5, horizon=6))
        np.testing.assert_array_equal(decayed.curve[:, :2], a.curve[:, :2])      # same draws until ages differ
        self.assertLess(decayed.mean_curve[-1], a.mean_curve[-1])
        zero = pl.adoption(np.zeros_like(self.freq), self.home, self.g, steps=4)
        self.assertEqual(int(zero.curve.sum()), 0)
        same = pl.adoption(self.freq, self.home, self.g, q=0.5, steps=6, runs=30, seed=1)
        np.testing.assert_array_equal(same.adopt_step, a.adopt_step)

    def test_wom_hook(self):
        calls = []

        def everyone(adopt_w, pop):
            calls.append(adopt_w.shape)
            return np.full(adopt_w.shape, 50.0)
        run = pl.adoption(self.freq, self.home, self.g, q=1.0, steps=2, runs=2, wom=everyone)
        self.assertEqual(calls[0], (2, self.g.ny, self.g.nx))
        housed = (self.home >= 0).sum()
        self.assertTrue((run.curve[:, 1] >= housed).all())

    def test_moore_share(self):
        pop = np.zeros((1, 3, 3))
        pop[0, 1, 1] = 4
        aw = np.zeros((1, 3, 3))
        aw[0, 1, 1] = 1
        s = pl.moore_share(aw, pop)
        self.assertTrue(np.allclose(s, 0.25))


@unittest.skipUnless(os.environ.get("SIMENGINE_DATA"), "SIMENGINE_DATA not set")
class TestYJMob(unittest.TestCase):
    def test_greedy_beats_top_on_train(self):
        from simengine.bench.base import DataNotFound, resolve_root
        try:
            root = resolve_root(None, "yjmob100k")
        except DataNotFound:
            self.skipTest("no yjmob100k")
        path = os.path.join(root, "yjmob100k-dataset1.csv.gz")
        if not os.path.exists(path):
            self.skipTest("no yjmob100k-dataset1.csv.gz")
        t = pd.read_csv(path, nrows=300_000)
        g = pl.CellGrid(1, 1, 40, 40, size=5)
        ex = pl.exposures(t, g, days=range(60))
        p = pl.greedy_placement(ex.M, 10)
        self.assertGreaterEqual(p.value, pl.coverage(ex.M, pl.top_visited(ex.M, 10)))
        self.assertLessEqual(p.value, p.bound)


if __name__ == "__main__":
    unittest.main()
