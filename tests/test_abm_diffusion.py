import unittest

import numpy as np

import simengine.abm as A


def synthetic_cascades(n=40, seed=0, horizon=1440):
    """Branching process with a power-law delay kernel; returns delays (min) and follower marks."""
    rng = np.random.default_rng(seed)
    delays, followers, roots = [], [], []
    for k in range(n):
        size_rate = rng.uniform(0.6, 0.9)
        times = [0.0]
        out = []
        i = 0
        while i < len(times) and len(out) < 400:
            kids = rng.poisson(size_rate * (3 if i == 0 else 1))
            for _ in range(kids):
                d = times[i] + (rng.pareto(0.8) + 1) * 2.0 - 2.0
                if d <= horizon:
                    times.append(d)
                    out.append(d)
            i += 1
        out = np.sort(np.array(out))
        delays.append(out)
        followers.append(rng.lognormal(5, 1, len(out)))
        roots.append(float(rng.lognormal(6, 1)))
    return delays, followers, roots


class TestKernels(unittest.TestCase):
    def test_kernels_normalised_monotone(self):
        for F in (A.power_kernel(5, 0.5), A.seismic_kernel()):
            v = F(np.arange(0, 1441, 1.0))
            self.assertAlmostEqual(float(v[0]), 0.0, places=9)
            self.assertAlmostEqual(float(v[-1]), 1.0, places=9)
            self.assertTrue((np.diff(v) >= -1e-12).all())


class TestHawkes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.delays, cls.followers, cls.roots = synthetic_cascades()

    def test_no_events_no_future(self):
        E, p = A.hawkes_mean_field([np.array([])], 60, A.power_kernel(5, 0.5))
        self.assertEqual(float(E.sum()), 0.0)
        self.assertEqual(float(p[0]), 0.0)

    def test_curves_never_below_observed(self):
        t = 60
        obs = [d[d <= t] for d in self.delays]
        N_obs = A.cascade_counts(obs)
        E, _ = A.hawkes_mean_field(obs, t, A.power_kernel(5, 0.5))
        L = A.cumulative_curve(N_obs, t, E)
        i = int(np.searchsorted(A.diffusion.GRID, t))
        self.assertTrue((L[:, i:] >= np.log1p(N_obs[:, i:i + 1]) - 1e-9).all())
        self.assertTrue((np.diff(L, axis=1) >= -1e-9).all())

    def test_select_kernel_and_shape_baseline(self):
        (c, th), scores = A.select_power_kernel(self.delays[:20], t=60, cs=(1, 5), thetas=(0.25, 1.0))
        self.assertIn((c, th), scores)
        N = A.cascade_counts(self.delays)
        g = A.shape_growth(N[:20], 60)
        L = A.shape_forecast(N[20:], 60, g)
        self.assertEqual(L.shape, N[20:].shape)
        self.assertTrue(np.isfinite(A.curve_rmse(L, N[20:])).all())

    def test_seismic_predictor(self):
        t = 60
        sp = A.SeismicPredictor(t=t).fit(self.delays[:25], self.followers[:25], self.roots[:25])
        self.assertGreater(sp.n_star, 0)
        obs = [d[d <= t] for d in self.delays[25:]]
        fol = [f[d <= t] for d, f in zip(self.delays[25:], self.followers[25:])]
        L = sp.predict(obs, fol, self.roots[25:])
        self.assertEqual(L.shape, (15, len(A.diffusion.GRID)))
        self.assertTrue(np.isfinite(L).all())


if __name__ == "__main__":
    unittest.main()
