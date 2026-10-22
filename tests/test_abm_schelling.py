import json
import unittest

import numpy as np

import simengine.abm.grid as G
import simengine.abm.schelling as S
from simengine.llm.fake import FakeLLM


def stripes(h, w, empty_col=None):
    lat = np.zeros((h, w), int)
    lat[:, w // 2:] = 1
    if empty_col is not None:
        lat[:, empty_col] = G.EMPTY
    return lat


class TestMetrics(unittest.TestCase):
    def test_unlike_pairs_reference_patterns(self):
        g = G.Grid((8, 8), torus=True)
        self.assertEqual(S.unlike_pairs(np.zeros((8, 8), int), g), 0)
        # two vertical stripes on a torus: 2 boundaries x 8 rows x 3 Moore links
        self.assertEqual(S.unlike_pairs(stripes(8, 8), g), 2 * 8 * 3)
        cb = np.indices((8, 8)).sum(0) % 2                       # von Neumann checkerboard: every edge unlike
        self.assertEqual(S.unlike_pairs(cb, G.Grid((8, 8), neighbourhood="von_neumann")), 2 * 64)

    def test_unlike_pairs_brute_force(self):
        g = G.Grid((7, 9), torus=False)
        lat = G.random_lattice(g, [0.4, 0.4], seed=3).ravel()
        nbr = g.neighbour_table()
        n = sum(1 for i in range(lat.size) for j in nbr[i] if j > i and lat[i] >= 0 and lat[j] >= 0
                and lat[i] != lat[j])
        self.assertEqual(S.unlike_pairs(lat.reshape(7, 9), g), n)

    def test_segregation_coefficient(self):
        self.assertAlmostEqual(S.segregation_coefficient(stripes(10, 10), torus=False), 1.0)
        # four stripes are two clusters per type on a torus only after the wrap merge
        lat = np.zeros((10, 12), int)
        lat[:, 3:9] = 1                                         # type 0 split at the edge, joined on a torus
        self.assertAlmostEqual(S.segregation_coefficient(lat, torus=True), 1.0)
        self.assertAlmostEqual(S.segregation_coefficient(lat, torus=False), 2 / 120 ** 2 * (30 ** 2 * 2 + 60 ** 2))
        cb = np.indices((10, 10)).sum(0) % 2                    # all singletons
        self.assertAlmostEqual(S.segregation_coefficient(cb), 2 / 100 ** 2 * 100)


class TestThresholdModel(unittest.TestCase):
    def setUp(self):
        self.g = G.Grid((20, 20), torus=True)

    def test_zero_threshold_nobody_moves(self):
        r = S.Schelling(self.g, threshold=0.0).run(10, seed=1)
        self.assertTrue(r.converged)
        self.assertEqual(r.steps, 1)
        np.testing.assert_array_equal(r.lattice, r.initial)

    def test_half_threshold_segregates_and_conserves(self):
        r = S.Schelling(self.g, threshold=0.5).run(200, seed=2)
        h = r.history
        self.assertTrue(r.converged)
        self.assertEqual(h.happy.iloc[-1], 1.0)
        self.assertLess(h.same_share.iloc[0], 0.6)
        self.assertGreater(h.same_share.iloc[-1], 0.75)
        for t in (0, 1, G.EMPTY):
            self.assertEqual((r.lattice == t).sum(), (r.initial == t).sum())
        self.assertIn("seg_coef", h.columns)

    def test_deterministic_given_seed(self):
        a = S.Schelling(self.g, threshold=0.4).run(50, seed=7)
        b = S.Schelling(self.g, threshold=0.4).run(50, seed=7)
        np.testing.assert_array_equal(a.lattice, b.lattice)

    def test_satisfying_and_nearest_rules_give_happy_movers(self):
        for rule in ("satisfying", "nearest"):
            m = S.Schelling(self.g, threshold=0.5, rule=rule)
            r = m.run(100, seed=4)
            self.assertGreater(r.history.same_share.iloc[-1], r.history.same_share.iloc[0])
            self.assertGreater(r.history.happy.iloc[-1], 0.95, rule)

    def test_nearest_picks_closest_satisfying_cell(self):
        g = G.Grid((5, 9), torus=False)
        lat = np.ones((5, 9), int)
        lat[:, :2] = 0
        lat[2, 6] = 0                                           # lone type-0 agent inside type 1
        lat[2, 2] = G.EMPTY                                     # next to the type-0 block (distance 4)
        lat[0, 8] = G.EMPTY                                     # inside type 1 (distance 4, unsatisfying)
        m = S.Schelling(g, threshold=0.3, rule="nearest")     # (2, 2): 3 of 8 alike; (0, 8): 0 of 3
        st = S._State(g, lat, 2)
        a = int(st.cell_agent[g.ravel(2, 6)])
        self.assertEqual(m._destination(st, a, np.random.default_rng(0)), g.ravel(2, 2))

    def test_like_tot_excludes_own_cell(self):
        g = G.Grid((3, 4), torus=False)
        lat = np.array([[0, -1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        st = S._State(g, lat, 2)
        a = int(st.cell_agent[0])
        like, tot = st.like_tot(a, np.array([1]))
        self.assertEqual((int(like[0]), int(tot[0])), (0, 4))  # 5 occupied around cell 1, minus the mover

    def test_per_type_threshold_and_bounded_grid(self):
        g = G.Grid((15, 15), torus=False)
        m = S.Schelling(g, threshold=[0.0, 0.6])
        r = m.run(100, seed=0)
        st0, st = S._State(g, r.initial, 2), S._State(g, r.lattice, 2)
        h0, h = m.happy_mask(st0), m.happy_mask(st)
        self.assertTrue(h[st.kind == 0].all())                  # threshold 0: always happy, never moves
        self.assertGreater(h[st.kind == 1].mean(), h0[st0.kind == 1].mean())

    def test_movers_all_requires_targeted_rule(self):
        with self.assertRaises(ValueError):
            S.Schelling(self.g, movers="all")
        r = S.Schelling(self.g, threshold=0.5, movers="all", rule="satisfying").run(5, seed=0, stop_when_still=True)
        self.assertEqual(r.steps, 5)                            # happy agents keep moving: never "still"

    def test_tipping_curve_shape(self):
        df = S.tipping_curve([0.2, 0.5, 0.9], seeds=[0, 1], grid=self.g, n_steps=60)
        m = df.groupby("threshold").same_share.mean()
        self.assertGreater(m[0.5], m[0.2])
        self.assertLess(m[0.9], m[0.5])                         # too demanding: no equilibrium, stays mixed
        self.assertFalse(df[df.threshold == 0.9].converged.any())


class TestZhang(unittest.TestCase):
    def test_payoff_matches_table4(self):
        u = S.zhang_payoff(np.arange(9), 4)
        np.testing.assert_allclose(u, [0, .25, .5, .75, 1, .9, .8, .7, .6])
        np.testing.assert_allclose(S.zhang_payoff(np.arange(5), 2), [0, .5, 1, .8, .6])

    def test_incremental_rho_and_like_counts(self):
        g = G.Grid((12, 12), torus=True)
        r = S.ZhangSwap(g, beta=5).run(20000, seed=1, record_every=5000)
        self.assertEqual(int(r.history.rho.iloc[-1]), S.unlike_pairs(r.lattice, g))
        self.assertAlmostEqual(r.history.same_share.iloc[-1], G.same_type_share(r.lattice, g))
        self.assertEqual((r.lattice == 0).sum(), 72)

    def test_beta_drives_segregation(self):
        g = G.Grid((20, 20), torus=True)
        lo = S.ZhangSwap(g, beta=0.0).run(40000, seed=0, record_every=40000).history.rho.iloc[-1]
        hi = S.ZhangSwap(g, beta=10.0).run(40000, seed=0, record_every=40000).history.rho.iloc[-1]
        self.assertLess(hi, 0.5 * lo)

    def test_target_stop_and_guards(self):
        g = G.Grid((10, 10), torus=True)
        r = S.ZhangSwap(g, beta=20).run(10 ** 6, seed=0, record_every=1000, target_rho=150)
        self.assertIsNotNone(r.reached)
        self.assertLessEqual(r.history.rho.iloc[-1], 150)
        with self.assertRaises(ValueError):
            S.ZhangSwap(G.Grid((10, 10), torus=False)).run(10, seed=0)
        with self.assertRaises(ValueError):
            S.ZhangSwap(g, Z=0.5, M=0.6)


class TestParse(unittest.TestCase):
    def test_json_forms(self):
        self.assertEqual(S.parse_decision('{"move": true}'), S.Decision(True))
        self.assertEqual(S.parse_decision('Sure. ```json\n{"move": "no"}\n```'), S.Decision(False))
        self.assertEqual(S.parse_decision('{"choice": 2}', 3), S.Decision(True, 2))
        self.assertEqual(S.parse_decision('{"choice": "stay"}', 3), S.Decision(False))
        self.assertEqual(S.parse_decision('{"choice": 7}', 3), S.Decision(False, valid=False))
        self.assertEqual(S.parse_decision('{"yes_rating": 7, "no_rating": 3}'), S.Decision(True))
        self.assertEqual(S.parse_decision('{"yes_rating": 2, "no_rating": 8}'), S.Decision(False))

    def test_text_forms(self):
        self.assertEqual(S.parse_decision("Yes, I want to move."), S.Decision(True))
        self.assertEqual(S.parse_decision("I will stay here."), S.Decision(False))
        self.assertEqual(S.parse_decision("Option 3 looks best", 3), S.Decision(True, 3))
        self.assertEqual(S.parse_decision("2", 3), S.Decision(True, 2))
        self.assertFalse(S.parse_decision("hmm", 0).valid)
        self.assertFalse(S.parse_decision("", 0).valid)
        self.assertFalse(S.parse_decision(None, 0).valid)


class TestLLMMover(unittest.TestCase):
    def setUp(self):
        self.g = G.Grid((16, 16), torus=True)

    def test_threshold_responder_reproduces_rule(self):
        llm = FakeLLM(S.threshold_responder(0.5))
        mover = S.LLMMover(llm, labels=("blue", "orange"))
        r = S.Schelling(self.g, threshold=0.5, mover=mover).run(60, seed=3)
        self.assertTrue(r.converged)
        self.assertEqual(r.history.happy.iloc[-1], 1.0)
        self.assertGreater(r.history.same_share.iloc[-1], 0.75)
        self.assertEqual(r.llm["invalid"], 0)
        self.assertLess(r.llm["calls"], r.llm["asked"])        # identical neighbourhoods share one call
        self.assertTrue(all("orange" in u or "blue" in u for _, u in llm.calls))

    def test_prompt_describes_neighbourhood(self):
        lat = np.ones((5, 5), int)
        lat[2, 2] = 0
        lat[0, 0] = G.EMPTY
        g = G.Grid((5, 5), torus=True)
        llm = FakeLLM(lambda s, u: '{"move": false}')
        mover = S.LLMMover(llm, labels=("red", "green"))
        S.Schelling(g, threshold=0.5, mover=mover).run(1, seed=0, lattice=lat)
        users = [u for _, u in llm.calls]
        self.assertTrue(any("0 are red households (your group), 8 are green households, 0 are empty" in u
                            for u in users))
        self.assertTrue(all(json.loads('{"move": false}') == {"move": False} for _ in users))

    def test_persona_hook_forms(self):
        for persona, check in (("likes quiet streets", lambda s, i, t: "likes quiet streets" in s),
                               ({0: "P-zero", 1: "P-one"}, lambda s, i, t: f"P-{['zero', 'one'][t]}" in s),
                               (lambda i, t: f"agent {i}", lambda s, i, t: f"agent {i}" in s)):
            llm = FakeLLM(lambda s, u: '{"move": false}')
            mover = S.LLMMover(llm, persona=persona, ask="all")
            r = S.Schelling(self.g, threshold=0.5, mover=mover).run(1, seed=0)
            self.assertEqual(r.llm["asked"], int((r.initial >= 0).sum()))
            st = S._State(self.g, r.initial, 2)
            sys0 = mover.system_prompt(0, int(st.kind[0]))
            self.assertIn("About you:", sys0)
            self.assertTrue(check(sys0, 0, int(st.kind[0])))
            self.assertTrue(any("About you:" in s for s, _ in llm.calls))
        per_agent = [f"person {i}" for i in range(400)]
        mover = S.LLMMover(FakeLLM(lambda s, u: "stay"), persona=per_agent)
        self.assertIn("person 5", mover.system_prompt(5, 0))
        self.assertEqual(S.LLMMover(FakeLLM(), persona=None).system_prompt(0, 0), S.DEFAULT_SYSTEM)

    def test_options_move_to_chosen_vacancy(self):
        llm = FakeLLM(S.threshold_responder(0.5))
        mover = S.LLMMover(llm, n_options=4)
        r = S.Schelling(self.g, threshold=0.5, mover=mover).run(40, seed=1)
        self.assertGreater(r.history.same_share.iloc[-1], r.history.same_share.iloc[0] + 0.1)
        self.assertTrue(any("Option 4:" in u for _, u in llm.calls))
        for t in (0, 1, G.EMPTY):
            self.assertEqual((r.lattice == t).sum(), (r.initial == t).sum())

    def test_options_still_round_does_not_stop(self):
        llm = FakeLLM(lambda s, u: '{"choice": "stay"}')
        r = S.Schelling(self.g, threshold=0.7, mover=S.LLMMover(llm, n_options=2)).run(4, seed=0)
        self.assertEqual(r.steps, 4)
        self.assertFalse(r.converged)

    def test_invalid_replies_stay_and_are_counted(self):
        mover = S.LLMMover(FakeLLM(lambda s, u: "I cannot decide."))
        r = S.Schelling(self.g, threshold=0.5, mover=mover).run(5, seed=0)
        np.testing.assert_array_equal(r.lattice, r.initial)
        self.assertEqual(r.llm["invalid"], r.llm["asked"])
        self.assertEqual(r.llm["invalid_rate"], 1.0)
        self.assertTrue(r.converged)                            # nobody moved -> still

    def test_max_calls_stops_cleanly(self):
        llm = FakeLLM(lambda s, u: '{"move": true}')
        mover = S.LLMMover(llm, max_calls=3, dedupe=False)
        r = S.Schelling(self.g, threshold=0.5, mover=mover).run(5, seed=0)
        self.assertEqual(len(llm.calls), 0)
        self.assertFalse(r.converged)
        self.assertEqual(r.steps, 1)

    def test_no_dedupe_sends_every_prompt(self):
        llm = FakeLLM(lambda s, u: '{"move": false}')
        mover = S.LLMMover(llm, dedupe=False)
        r = S.Schelling(self.g, threshold=0.5, mover=mover).run(1, seed=0)
        self.assertEqual(r.llm["calls"], r.llm["asked"])


if __name__ == "__main__":
    unittest.main()
