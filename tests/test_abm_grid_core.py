import unittest

import numpy as np

import simengine.abm.grid as G


def brute_neighbour_sum(a, grid):
    h, w = grid.shape
    out = np.zeros(a.shape, float)
    for r in range(h):
        for c in range(w):
            for dr, dc in grid.offsets:
                rr, cc = r + dr, c + dc
                if grid.torus:
                    out[r, c] += a[rr % h, cc % w]
                elif 0 <= rr < h and 0 <= cc < w:
                    out[r, c] += a[rr, cc]
    return out


def brute_moran(x, grid, mask):
    """Dense weight matrix form of Moran's I."""
    h, w = grid.shape
    idx = [(r, c) for r in range(h) for c in range(w) if mask[r, c]]
    pos = {p: i for i, p in enumerate(idx)}
    n = len(idx)
    W = np.zeros((n, n))
    for (r, c), i in pos.items():
        for dr, dc in grid.offsets:
            rr, cc = r + dr, c + dc
            if grid.torus:
                rr, cc = rr % h, cc % w
            if (rr, cc) in pos:
                W[i, pos[(rr, cc)]] += 1
    v = np.array([x[p] for p in idx])
    z = v - v.mean()
    return n / W.sum() * (z @ W @ z) / (z @ z)


class TestGrid(unittest.TestCase):
    def test_offsets_sizes(self):
        self.assertEqual(G.Grid((9, 9), neighbourhood="moore").k, 8)
        self.assertEqual(G.Grid((9, 9), neighbourhood="von_neumann").k, 4)
        self.assertEqual(G.Grid((9, 9), neighbourhood="moore", radius=2).k, 24)
        self.assertEqual(G.Grid((9, 9), neighbourhood="von_neumann", radius=2).k, 12)
        self.assertEqual(G.Grid((5, 5)).kernel().sum(), 8)
        self.assertEqual(G.Grid((5, 5)).kernel()[1, 1], 0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            G.Grid((2, 5), torus=True)
        with self.assertRaises(ValueError):
            G.Grid((5, 5), neighbourhood="hex")
        G.Grid((2, 5), torus=False)                       # bounded small grid is fine
        with self.assertRaises(ValueError):
            G.Grid((5, 5)).neighbour_sum(np.zeros((4, 5)))

    def test_neighbour_sum_matches_brute_force(self):
        rng = np.random.default_rng(1)
        for torus in (True, False):
            for nb in G.NEIGHBOURHOODS:
                for r in (1, 2):
                    g = G.Grid((7, 9), torus=torus, neighbourhood=nb, radius=r)
                    a = rng.random(g.shape)
                    np.testing.assert_allclose(g.neighbour_sum(a), brute_neighbour_sum(a, g))

    def test_matches_scipy_ndimage(self):
        try:
            from scipy import ndimage
        except ImportError:
            self.skipTest("scipy missing")
        rng = np.random.default_rng(2)
        a = rng.random((11, 13))
        for torus, mode in ((True, "grid-wrap"), (False, "constant")):
            g = G.Grid(a.shape, torus=torus, neighbourhood="moore", radius=2)
            ref = ndimage.correlate(a, g.kernel().astype(float), mode=mode, cval=0.0)
            np.testing.assert_allclose(g.neighbour_sum(a), ref)

    def test_stack_and_bool(self):
        g = G.Grid((6, 6), torus=False)
        m = np.zeros((2, 6, 6), bool)
        m[0, 0, 0] = m[1, 3, 3] = True
        c = g.neighbour_count(m)
        self.assertEqual(c.dtype, np.int32)
        self.assertEqual(c[0].sum(), 3)                   # corner has 3 neighbours
        self.assertEqual(c[1].sum(), 8)

    def test_degree(self):
        self.assertTrue((G.Grid((5, 5)).degree() == 8).all())
        d = G.Grid((5, 5), torus=False).degree()
        self.assertEqual((d[0, 0], d[0, 2], d[2, 2]), (3, 5, 8))
        d = G.Grid((5, 5), torus=False, neighbourhood="von_neumann").degree()
        self.assertEqual((d[0, 0], d[0, 2], d[2, 2]), (2, 3, 4))

    def test_neighbour_table(self):
        for torus in (True, False):
            g = G.Grid((5, 6), torus=torus, neighbourhood="von_neumann")
            t = g.neighbour_table()
            self.assertEqual(t.shape, (30, 4))
            a = np.arange(30, dtype=float).reshape(5, 6) + 1
            flat = np.concatenate([a.ravel(), [0.0]])     # -1 -> the appended 0
            np.testing.assert_allclose(flat[t].sum(1).reshape(g.shape), g.neighbour_sum(a))
        self.assertTrue((G.Grid((5, 5), torus=False).neighbour_table()[0] == -1).sum() == 5)

    def test_type_counts(self):
        g = G.Grid((4, 4))
        lat = np.full((4, 4), G.EMPTY)
        lat[0, 0], lat[0, 1] = 0, 1
        tc = g.type_counts(lat)
        self.assertEqual(tc.shape, (2, 4, 4))
        self.assertEqual(tc[0, 0, 1], 1)
        self.assertEqual(tc[1, 0, 0], 1)
        self.assertEqual(tc[:, 2, 2].sum(), 0)


class TestAgents(unittest.TestCase):
    def test_random_lattice_counts_and_seed(self):
        g = G.Grid((10, 10))
        lat = G.random_lattice(g, [0.3, 0.5], seed=3)
        self.assertEqual(((lat == 0).sum(), (lat == 1).sum(), (lat == G.EMPTY).sum()), (30, 50, 20))
        np.testing.assert_array_equal(lat, G.random_lattice(g, [0.3, 0.5], seed=3))
        self.assertFalse((lat == G.random_lattice(g, [0.3, 0.5], seed=4)).all())
        with self.assertRaises(ValueError):
            G.random_lattice(g, [0.7, 0.5])

    def test_roundtrip(self):
        g = G.Grid((6, 7))
        lat = G.random_lattice(g, [0.4, 0.4], seed=0)
        ag = G.Agents.from_lattice(lat, wealth=np.ones((lat != G.EMPTY).sum()))
        np.testing.assert_array_equal(ag.lattice(g), lat)
        rc = ag.coords(g)
        np.testing.assert_array_equal(lat[rc[:, 0], rc[:, 1]], ag.kind)
        sub = ag.subset(ag.kind == 1)
        self.assertEqual(sub.n, (lat == 1).sum())
        self.assertEqual(len(sub.data["wealth"]), sub.n)

    def test_collision_and_length(self):
        g = G.Grid((3, 3))
        with self.assertRaises(ValueError):
            G.Agents([0, 0], [0, 1]).lattice(g)
        with self.assertRaises(ValueError):
            G.Agents([0, 1], [0, 1], {"x": np.zeros(3)})


class TestSchedulers(unittest.TestCase):
    def test_rng(self):
        r = G.make_rng(5)
        self.assertIs(G.make_rng(r), r)
        a = [x.random() for x in G.spawn_rngs(7, 3)]
        b = [x.random() for x in G.spawn_rngs(7, 3)]
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 3)
        self.assertEqual(len(G.spawn_rngs(np.random.default_rng(1), 2)), 2)

    def test_random_order_sees_latest_state(self):
        state = [0]
        seen = []

        def act(i):
            seen.append(state[0])
            state[0] += 1

        order = G.random_order_step(5, act, np.random.default_rng(0))
        self.assertEqual(sorted(order.tolist()), list(range(5)))
        self.assertEqual(seen, [0, 1, 2, 3, 4])
        o2 = G.random_order_step(5, lambda i: None, np.random.default_rng(1))
        self.assertFalse((order == o2).all())

    def test_synchronous_uses_old_state(self):
        # 1-D shift rule: synchronous gives an exact roll; in-place writes would smear values
        s = np.arange(5)
        new = G.synchronous_step(s, lambda old, rng: np.roll(old, 1), np.random.default_rng(0))
        np.testing.assert_array_equal(new, [4, 0, 1, 2, 3])
        np.testing.assert_array_equal(s, np.arange(5))

        def bad(old, rng):
            old[0] = 9
            return old

        with self.assertRaises(ValueError):
            G.synchronous_step(s, bad, np.random.default_rng(0))

    def test_run_record_stop_and_seed(self):
        x = {"v": 0.0}

        def step(t, rng):
            x["v"] += rng.random()

        log = G.run(step, 10, seed=3, record=lambda t: x["v"], stop=lambda t: t == 4)
        self.assertEqual((log.steps, log.stopped, len(log.records)), (4, True, 5))
        self.assertEqual(log.records[0], 0.0)
        x["v"] = 0.0
        log2 = G.run(step, 4, seed=3, record=lambda t: x["v"])
        self.assertEqual(log.records, log2.records)
        self.assertFalse(log2.stopped)


class TestMetrics(unittest.TestCase):
    def test_moran_reference_patterns(self):
        cb = (np.indices((8, 8)).sum(0) % 2).astype(float)
        vn = G.Grid((8, 8), neighbourhood="von_neumann")
        self.assertAlmostEqual(G.morans_i(cb, vn), -1.0)
        stripes = (np.arange(8)[:, None] // 4 * np.ones((1, 8))).astype(float)   # two halves
        self.assertGreater(G.morans_i(stripes, vn), 0.7)
        self.assertTrue(np.isnan(G.morans_i(np.ones((8, 8)), vn)))

    def test_moran_matches_dense_weights(self):
        rng = np.random.default_rng(4)
        x = rng.random((6, 7))
        mask = rng.random((6, 7)) < 0.8
        for torus in (True, False):
            for nb in G.NEIGHBOURHOODS:
                g = G.Grid((6, 7), torus=torus, neighbourhood=nb)
                self.assertAlmostEqual(G.morans_i(x, g, mask), brute_moran(x, g, mask), places=10)

    def test_moran_random_near_expectation(self):
        g = G.Grid((40, 40))
        vals = [G.morans_i(np.random.default_rng(s).random(g.shape), g) for s in range(30)]
        self.assertLess(abs(np.mean(vals) - (-1 / (g.size - 1))), 0.01)

    def test_moran_permutation(self):
        g = G.Grid((20, 20))
        clustered = (np.arange(20)[None, :] < 10) * np.ones((20, 1))
        i, p = G.moran_permutation(clustered, g, n_perm=199, seed=0)
        self.assertAlmostEqual(i, G.morans_i(clustered, g))
        self.assertAlmostEqual(p, 1 / 200)
        _, p_rand = G.moran_permutation(np.random.default_rng(1).random((20, 20)), g, n_perm=199, seed=0)
        self.assertGreater(p_rand, 0.01)
        self.assertEqual(G.moran_permutation(clustered, g, n_perm=50, seed=2),
                         G.moran_permutation(clustered, g, n_perm=50, seed=2))

    def test_dissimilarity(self):
        lat = np.zeros((8, 8), int)
        lat[:, 4:] = 1
        self.assertAlmostEqual(G.dissimilarity(lat, 4), 1.0)
        cb = np.indices((8, 8)).sum(0) % 2
        self.assertAlmostEqual(G.dissimilarity(cb, 2), 0.0)
        self.assertAlmostEqual(G.dissimilarity(cb, (2, 4)), 0.0)
        # hand case: tiles (a, b) = (3, 1) and (1, 3) -> 0.5 * (|3/4-1/4| + |1/4-3/4|) = 0.5
        hand = np.array([[0, 0, 1, 1], [0, 1, 1, 0]])
        self.assertAlmostEqual(G.dissimilarity(hand, 2), 0.5)
        self.assertAlmostEqual(G.dissimilarity(np.array([[0, 1, 0]]), (1, 2)), 0.5)   # partial edge tile
        with self.assertRaises(ValueError):
            G.dissimilarity(np.zeros((4, 4), int), 2)

    def test_same_type_share(self):
        g = G.Grid((8, 8))
        lat = np.zeros((8, 8), int)
        self.assertEqual(G.same_type_share(lat, g), 1.0)
        cb = np.indices((8, 8)).sum(0) % 2
        self.assertAlmostEqual(G.same_type_share(cb, g), 0.5)            # Moore: 4 diagonal same, 4 edge other
        self.assertAlmostEqual(G.same_type_share(cb, G.Grid((8, 8), neighbourhood="von_neumann")), 0.0)
        lat = np.full((5, 5), G.EMPTY)
        lat[0, 0], lat[0, 1], lat[4, 4] = 0, 1, 0                         # torus: (4,4) touches (0,0)
        sh = G.same_type_share(lat, G.Grid((5, 5), torus=False), per_cell=True)
        self.assertTrue(np.isnan(sh[4, 4]) and np.isnan(sh[2, 2]))
        self.assertEqual((sh[0, 0], sh[0, 1]), (0.0, 0.0))
        self.assertAlmostEqual(G.same_type_share(lat, G.Grid((5, 5))), (0.5 + 0.0 + 1.0) / 3)
        self.assertTrue(np.isnan(G.same_type_share(np.full((3, 3), G.EMPTY), G.Grid((3, 3)))))

    def test_random_mix_share(self):
        g = G.Grid((60, 60))
        lat = G.random_lattice(g, [0.45, 0.45], seed=0)
        self.assertLess(abs(G.same_type_share(lat, g) - 0.5), 0.02)

    def test_gini(self):
        self.assertEqual(G.gini([2, 2, 2]), 0.0)
        self.assertAlmostEqual(G.gini([0, 0, 0, 5]), 0.75)
        self.assertAlmostEqual(G.gini([1, 2, 3, 4]), 0.25)               # mean abs diff / (2 mean) = 1.25 / 5
        x = np.random.default_rng(0).exponential(size=20000)
        self.assertAlmostEqual(G.gini(x), 0.5, delta=0.01)               # exponential: G = 1/2
        self.assertTrue(np.isnan(G.gini([0, 0])))
        with self.assertRaises(ValueError):
            G.gini([-1, 2])



class TestReviewEdgeCases(unittest.TestCase):
    def test_radius_must_be_integer(self):
        for r in (0, 1.5, -1):
            with self.assertRaises(ValueError):
                G.Grid((9, 9), radius=r)
        self.assertEqual(G.Grid((9, 9), radius=2.0).radius, 2)

    def test_neighbour_sum_dtypes(self):
        g = G.Grid((5, 5))
        self.assertTrue((g.neighbour_sum(np.full((5, 5), 200, np.uint8)) == 1600).all())   # no uint8 wrap
        self.assertEqual(g.neighbour_sum(np.ones((5, 5), np.int8)).dtype, np.int32)
        self.assertEqual(g.neighbour_sum(np.ones((5, 5), np.uint32)).dtype, np.int64)
        self.assertEqual(g.neighbour_sum(np.ones((5, 5), np.uint64)).dtype, np.uint64)     # not float64
        self.assertEqual(g.neighbour_sum(np.ones((5, 5), np.float32)).dtype, np.float32)

    def test_agents_lattice_rejects_bad_cells_and_kinds(self):
        g = G.Grid((3, 3), torus=False)
        with self.assertRaises(ValueError):
            G.Agents([0, -1], [0, 1]).lattice(g)          # -1 from a bounded neighbour_table
        with self.assertRaises(ValueError):
            G.Agents([0, 9], [0, 1]).lattice(g)
        with self.assertRaises(ValueError):
            G.Agents([0, 1], [0, -1]).lattice(g)
        self.assertTrue((G.Agents([], []).lattice(g) == G.EMPTY).all())

    def test_synchronous_result_is_fresh(self):
        s = np.arange(4)
        new = G.synchronous_step(s, lambda old, rng: old, np.random.default_rng(0))
        self.assertTrue(new.flags.writeable)
        new[0] = 7                                        # does not raise, does not touch s
        self.assertEqual(s[0], 0)
        with self.assertRaises(ValueError):
            G.synchronous_step(s, lambda old, rng: old[:2], np.random.default_rng(0))

    def test_moran_permutation_alternatives(self):
        n = 99
        g = G.Grid((10, 10), neighbourhood="von_neumann")
        clustered = (np.arange(10)[None, :] < 5) * np.ones((10, 1))
        cb = (np.indices((10, 10)).sum(0) % 2).astype(float)
        p = {a: G.moran_permutation(clustered, g, n_perm=n, alternative=a)[1]
             for a in ("directed", "greater", "lesser", "two-sided")}
        self.assertEqual(p, {"directed": 1 / 100, "greater": 1 / 100, "lesser": 1.0, "two-sided": 2 / 100})
        q = {a: G.moran_permutation(cb, g, n_perm=n, alternative=a)[1] for a in ("directed", "greater", "lesser")}
        self.assertEqual(q, {"directed": 1 / 100, "greater": 1.0, "lesser": 1 / 100})
        with self.assertRaises(ValueError):
            G.moran_permutation(cb, g, alternative="folded")
        i, pv = G.moran_permutation(np.ones((10, 10)), g, n_perm=9)
        self.assertTrue(np.isnan(i) and np.isnan(pv))
        big = G.Grid((300, 300))                          # chunk is capped by 2**22 cells, result unchanged
        x = np.random.default_rng(0).random(big.shape)
        self.assertEqual(G.moran_permutation(x, big, n_perm=5, chunk=64)[1],
                         G.moran_permutation(x, big, n_perm=5, chunk=5)[1])

    def test_dissimilarity_and_share_input_checks(self):
        with self.assertRaises(ValueError):
            G.dissimilarity(np.zeros((4, 4), int), 0)
        with self.assertRaises(ValueError):
            G.dissimilarity(np.zeros(4, int), 2)
        lat = np.zeros((4, 4), int)
        lat[0, 0] = -2
        with self.assertRaises(ValueError):
            G.same_type_share(lat, G.Grid((4, 4)))

if __name__ == "__main__":
    unittest.main()
