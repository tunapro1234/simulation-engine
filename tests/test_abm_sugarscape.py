import json
import os
import re
import tempfile
import unittest

import numpy as np

import simengine.abm.sugarscape as S
from simengine.llm.fake import FakeLLM


def tiny(capacity, n_agents=1, **kw):
    """Model on a hand-made capacity grid; agents are moved with `place`."""
    cfg = S.SugarscapeConfig(n_agents=n_agents, capacity=np.asarray(capacity, float), max_age=None, **kw)
    return S.Sugarscape(cfg, seed=0)


def place(m, i, r, c, vision=None, sugar=None):
    site = r * m.side + c
    m.occ[m.pos[i]] = -1
    m.pos[i], m.occ[site] = site, i
    if vision is not None:
        m.vision[i] = vision
    if sugar is not None:
        m.wealth[i] = sugar


def greedy_responder(system, user):
    """Picks the listed site with the most sugar (first listed on ties): rule M through the prompt."""
    sites = [(int(j), float(s)) for j, s in re.findall(r"^(\d+): [^,]*, sugar ([\d.]+)", user, re.M)]
    best = max(sites, key=lambda t: (t[1], -t[0]))
    return json.dumps({"site": best[0]})


class TestInequality(unittest.TestCase):
    def test_gini_known_values(self):
        self.assertAlmostEqual(S.gini([5, 5, 5, 5]), 0.0)
        self.assertAlmostEqual(S.gini([0, 0, 0, 10]), 0.75)            # (n - 1) / n
        self.assertAlmostEqual(S.gini([1, 2, 3, 4]), 0.25)
        self.assertTrue(np.isnan(S.gini([])))

    def test_lorenz_matches_gini(self):
        w = np.random.default_rng(3).pareto(2.0, 300)
        x, y = S.lorenz(w)
        self.assertEqual((x[0], y[0], x[-1]), (0.0, 0.0, 1.0))
        self.assertAlmostEqual(y[-1], 1.0)
        self.assertTrue(np.all(np.diff(y) >= 0) and np.all(y <= x + 1e-12))
        area = float(((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2).sum())
        self.assertAlmostEqual(S.gini(w), 1 - 2 * area, places=10)


class TestLandscape(unittest.TestCase):
    def test_two_peaks(self):
        cap = S.sugar_capacity()
        self.assertEqual(cap.shape, (50, 50))
        self.assertEqual(set(np.unique(cap)), {0.0, 1.0, 2.0, 3.0, 4.0})
        self.assertTrue((cap[:25, 25:] == 4).any() and (cap[25:, :25] == 4).any())   # NE and SW peaks
        self.assertFalse((cap[:25, :25] == 4).any() or (cap[25:, 25:] == 4).any())
        sp = S.spice_capacity()
        self.assertTrue((sp[:25, :25] == 4).any() and (sp[25:, 25:] == 4).any())     # NW and SE
        self.assertEqual(S.sugar_capacity(100).shape, (100, 100))

    def test_load_capacity(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "map.txt")
            np.savetxt(p, np.arange(9).reshape(3, 3), fmt="%d")
            np.testing.assert_array_equal(S.load_capacity(p), np.arange(9).reshape(3, 3))

    def test_growback(self):
        m = tiny(np.full((5, 5), 3.0), metabolism=(0, 0))
        m.sugar[:] = 0
        w0 = m.wealth[0]
        m.step()
        self.assertEqual(m.sugar.sum(), 24)                           # +1 everywhere, one site harvested
        self.assertEqual(m.wealth[0], w0 + 1)
        m2 = tiny(np.full((5, 5), 3.0), growback=np.inf)
        m2.sugar[:] = 0
        m2.step()
        self.assertEqual(int((m2.sugar == 3).sum()), 24)             # all but the site just harvested


class TestMovement(unittest.TestCase):
    def test_rule_m_picks_most_sugar_lattice_only(self):
        cap = np.zeros((9, 9))
        cap[4, 7] = 2           # east 3
        cap[1, 4] = 3           # north 3
        cap[3, 3] = 4           # diagonal: invisible
        m = tiny(cap)
        place(m, 0, 4, 4, vision=3)
        self.assertEqual(m.rule_m(0), 1 * 9 + 4)
        place(m, 0, 4, 4, vision=2)                                   # nothing visible: stay
        self.assertEqual(m.rule_m(0), 4 * 9 + 4)

    def test_nearest_on_ties_and_occupied_excluded(self):
        cap = np.zeros((9, 9))
        cap[4, 6] = 3           # east 2
        cap[4, 1] = 3           # west 3
        m = tiny(cap, n_agents=2)
        place(m, 0, 4, 4, vision=4)
        place(m, 1, 0, 0)
        self.assertEqual(m.rule_m(0), 4 * 9 + 6)
        place(m, 1, 4, 6)
        self.assertEqual(m.rule_m(0), 4 * 9 + 1)

    def test_non_torus_edges(self):
        m = tiny(np.ones((6, 6)), torus=False)
        place(m, 0, 0, 0, vision=2)
        sites, _ = m.candidates(0)
        self.assertEqual(len(sites), 5)                               # here + 2 south + 2 east
        m2 = tiny(np.ones((6, 6)))
        place(m2, 0, 0, 0, vision=2)
        self.assertEqual(len(m2.candidates(0)[0]), 9)

    def test_harvest_and_metabolism_conserve_sugar(self):
        cfg = S.SugarscapeConfig(n_agents=50, size=20, growback=0, metabolism=(0, 0), max_age=None)
        m = S.Sugarscape(cfg, seed=1)
        before = m.sugar.sum() + m.wealth.sum()
        for _ in range(5):
            m.step()
        self.assertAlmostEqual(m.sugar.sum() + m.wealth.sum(), before)
        cfg = S.SugarscapeConfig(n_agents=30, size=20, growback=0, metabolism=(2, 2), endowment=(100, 100),
                                 max_age=None)
        m = S.Sugarscape(cfg, seed=1)
        before = m.sugar.sum() + m.wealth.sum()
        m.step()
        self.assertAlmostEqual(m.sugar.sum() + m.wealth.sum(), before - 60)

    def test_plan_conflict_falls_back(self):
        cap = np.zeros((7, 7))
        cap[3, 3] = 4
        m = tiny(cap, n_agents=2)
        place(m, 0, 3, 2, vision=1)
        place(m, 1, 3, 4, vision=1)
        ev = m.step(plan={0: 3 * 7 + 3, 1: 3 * 7 + 3})
        self.assertEqual(ev["conflicts"], 1)
        self.assertEqual(m.occ[3 * 7 + 3] in (0, 1), True)


class TestDynamics(unittest.TestCase):
    def test_replacement_keeps_population(self):
        run = S.Sugarscape(S.SugarscapeConfig(n_agents=120, size=30), seed=2).run(120)
        self.assertTrue((run.stats.population == 120).all())
        self.assertGreater(run.stats.aged.sum(), 0)
        self.assertEqual(int(run.stats.births.sum()), int(run.stats.starved.sum() + run.stats.aged.sum()))
        self.assertEqual(set(run.snapshots), {0, 120})

    def test_carrying_capacity_without_replacement(self):
        cfg = S.SugarscapeConfig(n_agents=400, max_age=None, replace=False)
        st = S.Sugarscape(cfg, seed=0).run(150).stats
        self.assertLess(st.population.iloc[-1], 400)
        self.assertGreater(st.population.iloc[-1], 50)
        p = st.population.to_numpy()
        self.assertLess(p[100] - p[150], (p[0] - p[50]) / 10)                       # levelling off
        self.assertLess(st.mean_metabolism.iloc[-1], st.mean_metabolism.iloc[0])   # selection on metabolism

    def test_deterministic(self):
        a = S.Sugarscape(S.SugarscapeConfig(n_agents=60, size=25), seed=5).run(30).stats
        b = S.Sugarscape(S.SugarscapeConfig(n_agents=60, size=25), seed=5).run(30).stats
        self.assertTrue(a.equals(b))

    def test_gini_rises_from_uniform_endowment(self):
        st = S.Sugarscape(S.SugarscapeConfig(n_agents=250), seed=0).run(150).stats
        self.assertLess(st.gini.iloc[0], 0.3)
        self.assertGreater(st.gini.iloc[100:].mean(), st.gini.iloc[0] + 0.1)
        self.assertGreater(st.skewness.iloc[-1], 0)

    def test_gini_trajectory_frame(self):
        d = S.gini_trajectory(seeds=[0, 1], n_steps=20, config=S.SugarscapeConfig(n_agents=40, size=20), at=(0, 20))
        self.assertEqual(len(d), 4)
        self.assertEqual(set(d.columns), {"step", "seed", "gini", "population"})


class TestTrade(unittest.TestCase):
    def test_trade_requires_spice(self):
        with self.assertRaises(ValueError):
            S.Sugarscape(S.SugarscapeConfig(trade=True))

    def test_pair_trade_improves_both(self):
        m = tiny(np.ones((5, 5)), n_agents=2, spice=True, trade=True)
        m.wealth[:], m.wp[:] = [40.0, 4.0], [4.0, 40.0]
        m.metab[:], m.mp[:] = 2.0, 2.0
        w0 = S._welfare(m.wealth, m.wp, m.metab, m.mp)
        tot = (m.wealth.sum(), m.wp.sum())
        n = m._trade_pair(0, 1)
        self.assertGreater(n, 0)
        self.assertTrue(np.all(S._welfare(m.wealth, m.wp, m.metab, m.mp) > w0))
        self.assertAlmostEqual(m.wealth.sum(), tot[0])
        self.assertAlmostEqual(m.wp.sum(), tot[1])
        r0 = S._mrs(m.wealth[0], m.wp[0], 2.0, 2.0)
        r1 = S._mrs(m.wealth[1], m.wp[1], 2.0, 2.0)
        self.assertLess(abs(np.log(r0 / r1)), abs(np.log((4 / 40) / (40 / 4))))  # MRSs moved together

    def test_trade_run(self):
        cfg = S.SugarscapeConfig(n_agents=200, spice=True, trade=True)
        st = S.Sugarscape(cfg, seed=0).run(60).stats
        self.assertGreater(st.trades.sum(), 0)
        self.assertTrue(np.isfinite(st.price_geo_mean.iloc[1:]).any())
        self.assertTrue((st.population == 200).all())
        off = S.Sugarscape(S.SugarscapeConfig(n_agents=200, spice=True), seed=0).run(5).stats
        self.assertNotIn("trades", off.columns)


class TestLLMPolicy(unittest.TestCase):
    def cfg(self):
        return S.SugarscapeConfig(n_agents=30, size=20)

    def test_greedy_llm_runs_like_rule_m(self):
        llm = FakeLLM(greedy_responder)
        pol = S.LLMPolicy(llm, persona=lambda a: f"You have vision {a['vision']}.")
        run = S.Sugarscape(self.cfg(), seed=0, policy=pol).run(15)
        self.assertEqual(run.calls, int(run.stats.population.iloc[:-1].sum()))
        self.assertEqual(run.invalid, 0)
        self.assertEqual(run.stopped, "completed")
        self.assertIn("You have vision", llm.calls[0][0])
        self.assertIn('"site"', llm.calls[0][0])
        base = S.Sugarscape(self.cfg(), seed=0).run(15).stats
        self.assertLess(abs(run.stats.mean_wealth.iloc[-1] - base.mean_wealth.iloc[-1]),
                        0.5 * base.mean_wealth.iloc[-1])

    def test_invalid_replies_fall_back(self):
        run = S.Sugarscape(self.cfg(), seed=0, policy=S.LLMPolicy(FakeLLM(lambda s, u: "no idea"))).run(5)
        self.assertEqual(run.invalid, run.calls)
        self.assertTrue((run.stats.population == 30).all())

    def test_parse(self):
        self.assertEqual(S.LLMPolicy.parse('```json\n{"site": 2}\n```', 3), 2)
        self.assertEqual(S.LLMPolicy.parse("1", 3), 1)
        for bad in ('{"site": 7}', '{"site": true}', "north"):
            with self.assertRaises((ValueError, TypeError)):
                S.LLMPolicy.parse(bad, 3)

    def test_budget_stops(self):
        pol = S.LLMPolicy(FakeLLM(greedy_responder), max_calls=70)
        run = S.Sugarscape(self.cfg(), seed=0, policy=pol).run(10)
        self.assertTrue(run.stopped.startswith("budget"))
        self.assertEqual(run.calls, 60)
        self.assertEqual(len(run.stats), 3)

    def test_spice_prompt(self):
        llm = FakeLLM(greedy_responder)
        cfg = S.SugarscapeConfig(n_agents=10, size=20, spice=True)
        S.Sugarscape(cfg, seed=0, policy=S.LLMPolicy(llm)).run(2)
        self.assertIn("spice", llm.calls[0][1])


if __name__ == "__main__":
    unittest.main()
