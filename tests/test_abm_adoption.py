import unittest

import numpy as np
import pandas as pd

import simengine.abm as A


def synthetic_adoption(n_series=8, seed=0, start=1990, end=2021, noise=0.5):
    """Bass-shaped synthetic adoption series (percent), staggered starts."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_series):
        p, q, m = 0.01 * (1 + i % 3), 0.3 + 0.05 * i, 60 + 5 * i
        t = np.arange(start, end, dtype=float)
        y = A.bass_curve(t, m, p, q, start + i) + rng.normal(0, noise, len(t))
        rows += [dict(series=f"s{i}", t=tt, y=float(np.clip(yy, 0, 100))) for tt, yy in zip(t, y)]
    return pd.DataFrame(rows)


SMALL = dict(n_sim=300, n_agents=300)


class TestEngine(unittest.TestCase):
    def test_adoption_sim_shape_monotone_deterministic(self):
        th = A.sample_adoption_prior(5, np.random.default_rng(1))
        a = A.adoption_sim(th, 12, n_agents=200, seed=3)
        b = A.adoption_sim(th, 12, n_agents=200, seed=3)
        self.assertEqual(a.shape, (5, 12))
        np.testing.assert_array_equal(a, b)
        self.assertTrue((np.diff(a, axis=1) >= -1e-9).all())
        self.assertTrue(((a >= 0) & (a <= 100)).all())

    def test_market_sim_discount_raises_sales(self):
        th = np.array([A.MARKET_PRIOR[k][0] / 2 + A.MARKET_PRIOR[k][1] / 2 for k in A.MARKET_PARAMS])
        th[A.MARKET_PARAMS.index("social")] = 0.0
        lp = np.r_[np.zeros(10), -0.3 * np.ones(4)]
        base = A.market_sim(th, np.zeros(14), np.zeros(14), cohort=True)
        disc = A.market_sim(th, lp, np.zeros(14), cohort=True)
        self.assertGreater(disc[0, 10], base[0, 10])

    def test_bank_filters_late_crossings(self):
        B = A.simulation_bank(200, seed=1, n_agents=200)
        self.assertTrue((B["tau1"] <= 28.0).all())
        self.assertEqual(B["C"].shape[0], len(B["theta"]))
        self.assertEqual(B["C"].shape[1], len(A.adoption.GRID))


class TestCurves(unittest.TestCase):
    def test_linear_trend_exact(self):
        t = np.arange(10.0)
        y = 2 + 3 * t
        np.testing.assert_allclose(A.linear_trend(t, y, [10, 11]), [32, 35])

    def test_bass_recovers_noise_free_curve(self):
        t = np.arange(2000, 2012.0)
        y = A.bass_curve(t, 70, 0.01, 0.5, 1999.5)
        h = np.arange(2012, 2020.0)
        pred = A.bass(t, y, h)
        np.testing.assert_allclose(pred, A.bass_curve(h, 70, 0.01, 0.5, 1999.5), atol=1.0)

    def test_wrappers_and_bass_analogy(self):
        df = synthetic_adoption()
        c = 2000.0
        hist = df[df.t <= c]
        for m in [A.CurveBaseline("gompertz"), A.CurveBaseline("damped"), A.BassAnalogy()]:
            p = m.predict(m.fit(hist, "s2", c), [1, 2, 5])
            self.assertEqual(len(p), 3)
            self.assertTrue(((p.y_pred >= 0) & (p.y_pred <= 100)).all())


class TestABC(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_adoption()
        self.t0 = A.adoption_t0(self.df)

    def test_t0_rule(self):
        g = self.df[self.df.series == "s0"]
        self.assertEqual(self.t0["s0"], float(g[g.y >= 1].t.iloc[0]))

    def test_fit_predict_bands_ordered(self):
        for cls in (A.AdoptionABC, A.AdoptionABCAnalogy, A.BassABC, A.BassABCAnalogy):
            m = cls(**SMALL)
            c = self.t0["s3"] + 5
            fit = m.fit(self.df[self.df.t <= c], "s3", c)
            self.assertAlmostEqual(float(fit["w"].sum()), 1.0, places=6)
            p = m.predict(fit, np.arange(1, 11))
            self.assertTrue((p.lo80 <= p.y_pred + 1e-9).all() and (p.y_pred <= p.hi80 + 1e-9).all(), cls.__name__)
            self.assertTrue(((p.y_pred >= 0) & (p.y_pred <= 100)).all())

    def test_fit_does_not_need_future(self):
        m = A.AdoptionABC(**SMALL)
        c = self.t0["s1"] + 3
        a = m.predict(m.fit(self.df[self.df.t <= c], "s1", c), [1, 5])
        b = m.predict(m.fit(self.df, "s1", c), [1, 5])     # future rows present but must be ignored
        np.testing.assert_allclose(a.y_pred, b.y_pred)

    def test_not_reached_one_percent_raises(self):
        m = A.AdoptionABC(**SMALL)
        df = pd.DataFrame(dict(series="x", t=np.arange(5.0), y=[0.1, 0.2, 0.3, 0.4, 0.5]))
        with self.assertRaises(ValueError):
            m.fit(df, "x", 4.0)

    def test_scenario_more_imitation_not_slower(self):
        m = A.AdoptionABC(**SMALL)
        c = self.t0["s4"] + 5
        m.fit(self.df[self.df.t <= c], "s4", c)
        sc = m.scenario(intervention={"imitation": 2.0}, n_draws=30)
        base = sc[sc.scenario == "baseline"].y_pred.to_numpy()
        more = sc[sc.scenario != "baseline"].y_pred.to_numpy()
        self.assertGreaterEqual(float((more - base).mean()), 0.0)
        with self.assertRaises(ValueError):
            m.scenario(intervention={"bogus": 1})

    def test_personas_shares(self):
        m = A.AdoptionABCAnalogy(**SMALL)
        c = self.t0["s5"] + 5
        m.fit(self.df[self.df.t <= c], "s5", c)
        ps = m.personas()
        self.assertEqual(len(ps), 4)
        tot = sum(p["share"] for p in ps)
        self.assertTrue(0.9 <= tot <= 1.1, tot)

    def test_backtest_eval_frame_and_beats_naive_on_bass_data(self):
        methods = {"bass_abc": A.BassABC(**SMALL), "naive": A.CurveBaseline("naive")}
        ev = A.adoption_backtest(self.df, methods, cutoffs=(3, 5))
        for col in ("truth", "pred_bass_abc", "pred_naive", "group", "horizon", "lo80_bass_abc"):
            self.assertIn(col, ev.columns)
        mae = {k: float(np.abs(ev[f"pred_{k}"] - ev.truth).mean()) for k in methods}
        self.assertLess(mae["bass_abc"], mae["naive"])

    def test_turkish_column_alias(self):
        d = self.df.rename(columns={"series": "seri"})
        self.assertEqual(A.adoption_t0(d), self.t0)


if __name__ == "__main__":
    unittest.main()
