import os
import unittest

import numpy as np
import pandas as pd

import simengine.abm.spatial_sir as ep
from simengine.llm import FakeLLM

SEIR = ep.EpiParams.from_r0(2.5, infectious_days=5, latent_days=3)
SIR = ep.EpiParams.from_r0(2.0, infectious_days=4)


class TestODE(unittest.TestCase):
    def test_final_size_and_inverse(self):
        self.assertAlmostEqual(ep.final_size(2.0), 0.7968121, places=6)
        self.assertEqual(ep.final_size(0.9), 0.0)
        for r0 in (1.3, 2.0, 3.0):
            self.assertAlmostEqual(ep.r0_final_size(ep.final_size(r0)), r0, places=6)

    def test_ode_conserves_and_reaches_final_size(self):
        for p in (SIR, SEIR):
            o = ep.seir_ode(p, n=1e6, i0=1, days=400)
            tot = o[["S", "E", "I", "R"]].sum(1)
            self.assertTrue(np.allclose(tot, 1e6))
            self.assertAlmostEqual(1 - o.S.iloc[-1] / 1e6, ep.final_size(p.r0), places=3)
            self.assertAlmostEqual(o.new.sum(), 1e6 - o.S.iloc[-1], delta=1e-3)   # new[0] = seeds
        self.assertTrue((ep.seir_ode(SIR, 100, 1, 10).E == 0).all())

    def test_r0_growth_recovers_ode(self):
        o = ep.seir_ode(SEIR, n=1e6, i0=1, days=200)
        est = ep.r0_growth(o.new.to_numpy(), SEIR.gamma, SEIR.sigma, n=1e6, start_cases=50, stop_share=0.01)
        self.assertAlmostEqual(est["r0"], 2.5, delta=0.05)
        sir = ep.seir_ode(SIR, n=1e6, i0=1, days=200)
        est = ep.r0_growth(sir.new.to_numpy(), SIR.gamma, n=1e6, start_cases=50, stop_share=0.01)
        self.assertAlmostEqual(est["r0"], 2.0, delta=0.05)
        with self.assertRaises(ValueError):
            ep.r0_growth([1, 2, 3], 0.2)


class TestGrid(unittest.TestCase):
    def brute(self, a, r, torus):
        H, W = a.shape[-2:]
        out = np.zeros_like(a, float)
        for i in range(H):
            for j in range(W):
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        ii, jj = i + di, j + dj
                        if torus:
                            out[..., i, j] += a[..., ii % H, jj % W]
                        elif 0 <= ii < H and 0 <= jj < W:
                            out[..., i, j] += a[..., ii, jj]
        return out

    def test_box_sum_matches_brute_force(self):
        a = np.random.default_rng(0).random((2, 7, 9))
        for r in (0, 1, 2):
            for torus in (True, False):
                np.testing.assert_allclose(ep.box_sum(a, r, torus), self.brute(a, r, torus))
        big = ep.box_sum(a, 10, True)                       # window covers the torus: total per replicate
        np.testing.assert_allclose(big, np.broadcast_to(a.sum((1, 2), keepdims=True), a.shape))
        np.testing.assert_allclose(ep.box_sum(a, 10, False), np.broadcast_to(a.sum((1, 2), keepdims=True), a.shape))

    def test_cells_from_xy(self):
        self.assertEqual(ep.cells_from_xy([1, 200], [1, 200], (200, 200), origin=1).tolist(), [0, 39999])
        with self.assertRaises(ValueError):
            ep.cells_from_xy([0], [0], (200, 200), origin=1)

    def test_trajectories_forward_fill(self):
        df = pd.DataFrame({"uid": [7, 7, 9], "d": [0, 1, 0], "t": [1, 0, 0], "x": [0, 1, 2], "y": [0, 0, 0]})
        traj, uids = ep.trajectories_from_records(df, (3, 3), days=2, slots_per_day=2)
        self.assertEqual(uids.tolist(), [7, 9])
        self.assertEqual(traj.tolist(), [[0, 0, 1, 1], [2, 2, 2, 2]])   # leading gap -> first known cell


class TestSpatialSEIR(unittest.TestCase):
    def test_well_mixed_matches_ode(self):
        n = 4000
        m = ep.SpatialSEIR(SEIR, shape=(40, 40), n=n, radius=None, steps_per_day=4)
        run = m.run(days=150, i0=20, reps=6, seed=1)
        tot = run.S + run.E + run.I + run.R
        self.assertTrue((tot == n).all())
        self.assertEqual(run.new.sum(1).tolist(), (n - run.S[:, -1]).tolist())
        agr = ep.ode_agreement(run, ep.seir_ode(SEIR, n, 20, 150))
        self.assertEqual(agr["major"], 1.0)
        self.assertAlmostEqual(agr["attack_rate_abm"], agr["attack_rate_ode"], delta=0.02)
        self.assertAlmostEqual(agr["peak_share_abm"], agr["peak_share_ode"], delta=0.015)
        self.assertLess(abs(agr["peak_day_abm"] - agr["peak_day_ode"]), 6)
        self.assertLess(agr["prevalence_rmse"], 0.01)

    def test_local_contact_slows_the_epidemic(self):
        kw = dict(shape=(60, 60), n=3600, steps_per_day=2)
        mixed = ep.SpatialSEIR(SEIR, radius=None, **kw).run(120, i0=10, reps=3, seed=2).summary()
        local = ep.SpatialSEIR(SEIR, radius=3, **kw).run(120, i0=10, reps=3, seed=2).summary()
        self.assertGreater(local["peak_day"], mixed["peak_day"] + 10)
        self.assertLess(local["peak_share"], mixed["peak_share"])

    def test_deterministic_and_seeds(self):
        m = ep.SpatialSEIR(SIR, shape=(20, 20), n=400, radius=1)
        a, b = m.run(40, i0=5, reps=2, seed=3), m.run(40, i0=5, reps=2, seed=3)
        np.testing.assert_array_equal(a.I, b.I)
        np.testing.assert_array_equal(a.infected_day, b.infected_day)
        run = m.run(40, i0=[0, 1], reps=2, seed=3)
        self.assertTrue((run.infected_day[:, :2] == 0).all())
        self.assertTrue((run.E == 0).all())                  # SIR has no exposed compartment

    def test_index_case_r0(self):
        mixed = ep.r0_index_case(ep.SpatialSEIR(SIR, shape=(30, 30), n=2000, radius=None, steps_per_day=2),
                                 reps=600, seed=0)
        self.assertLess(mixed["lo"], 2.0)
        self.assertGreater(mixed["hi"], 2.0)
        self.assertEqual(len(mixed["secondary"]), 600)

    def test_present_mask_isolates(self):
        m = ep.SpatialSEIR(SIR, shape=(10, 10), n=200, radius=None)
        present = np.ones(200, bool)
        present[100:] = False
        run = m.run(60, i0=[0, 1, 2], reps=2, seed=0, present=present)
        self.assertTrue((run.infected_day[:, 100:] == -1).all())

    def test_mobility_separated_groups(self):
        # group A lives in the left half, group B in the right half; B visits cell 0 only on odd slots when A is
        # elsewhere, so the two groups never share a window at radius 0
        T = 8
        a = np.tile(np.where(np.arange(T) % 2 == 0, 0, 1), (50, 1))
        b = np.tile(np.where(np.arange(T) % 2 == 1, 0, 3), (50, 1))
        m = ep.SpatialSEIR(ep.EpiParams(beta=3.0, gamma=0.2), shape=(2, 2), positions=np.r_[a, b], radius=0,
                           steps_per_day=4)
        run = m.run(60, i0=[0], reps=3, seed=1)
        self.assertTrue((run.infected_day[:, 50:] == -1).all())
        self.assertTrue((run.infected_day[:, :50] >= 0).mean() > 0.5)
        mixed = ep.SpatialSEIR(ep.EpiParams(beta=3.0, gamma=0.2), shape=(2, 2), positions=np.r_[a, b],
                               radius=1, steps_per_day=4).run(60, i0=[0], reps=3, seed=1)
        self.assertTrue((mixed.infected_day[:, 50:] >= 0).any())     # radius 1 covers the 2x2 torus

    def test_bad_positions(self):
        with self.assertRaises(ValueError):
            ep.SpatialSEIR(SIR, shape=(5, 5), positions=[30])
        with self.assertRaises(ValueError):
            ep.SpatialSEIR(SIR, shape=(5, 5))
        with self.assertRaises(ValueError):                  # gamma * dt > 1
            ep.SpatialSEIR(ep.EpiParams(beta=1.0, gamma=2.0), shape=(5, 5), n=5).run(1)
        ep.SpatialSEIR(ep.EpiParams(beta=1.0, gamma=2.0), shape=(5, 5), n=5, steps_per_day=2).run(1)

    @unittest.skipUnless(os.environ.get("SIMENGINE_DATA"), "SIMENGINE_DATA not set")
    def test_yjmob100k_loader(self):
        traj, uids = ep.load_yjmob100k(n_users=30, days=2)
        self.assertEqual(traj.shape, (30, 96))
        self.assertTrue((traj >= 0).all() and (traj < 40000).all())
        run = ep.SpatialSEIR(SIR, shape=(200, 200), positions=traj, radius=1, steps_per_day=48).run(4, i0=3, seed=0)
        self.assertEqual(run.S.shape, (1, 5))


class TestGenerativeEpidemic(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(ep.parse_isolation("Reasoning: has a fever.\nResponse: Yes"), (True, "has a fever."))
        self.assertEqual(ep.parse_isolation("Reasoning: needs money\n**Response:** no")[0], False)
        self.assertEqual(ep.parse_isolation("Reasoning: x\nResponse: [Yes]")[0], True)        # echoes the template
        self.assertEqual(ep.parse_isolation("Reasoning: busy day\n**Response**: No"), (False, "busy day"))
        self.assertIsNone(ep.parse_isolation("I would rather not say.")[0])
        for echo in ("Reasoning: x\nResponse: [Yes or No]", "Reasoning: x\nResponse: Yes/No", "Response: no or yes"):
            self.assertIsNone(ep.parse_isolation(echo)[0])                     # unfilled template -> 'No' later
        self.assertEqual(ep.parse_isolation("Reasoning: x\nResponse: No, staying out.")[0], False)

    def test_prompt_by_condition(self):
        a = ep.epi_personas(1, seed=0)[0]
        self.assertTrue(18 <= a.age <= 64 and len(a.traits) == 5)
        s, u = ep.isolation_prompt(a)
        self.assertNotIn("feels", s)
        self.assertNotIn("newspaper", s)
        self.assertIn("stay at home for the entire day", u)
        s, _ = ep.isolation_prompt(a, health="fever", news_pct=0.7, message="Masks are free at the pharmacy.")
        self.assertIn("has a fever and a cough", s)
        self.assertIn("finds that 0.7% of Dewberry Hollow's population", s)
        self.assertIn("Masks are free", s)

    def test_base_condition_everyone_goes_out(self):
        llm = FakeLLM(ep.rule_responder(news_threshold=0.0))
        res = ep.GenerativeEpidemic(llm, n=40, condition="base", seed=1).run(days=10, stop_when_clear=False)
        self.assertTrue((res.daily.out_share == 1.0).all())
        self.assertEqual(res.calls, 40 * 10)
        self.assertEqual(len(llm.calls), res.calls)
        self.assertEqual(len(res.log), 400)
        self.assertEqual(res.invalid, 0)

    def test_self_health_symptomatic_stay_home(self):
        res = ep.GenerativeEpidemic(FakeLLM(ep.rule_responder()), n=60, condition="self_health", i0=5,
                                    seed=0).run(days=30)
        by = res.stay_home_by_health()
        self.assertEqual(by["normal"], 0.0)
        self.assertEqual(by["fever"], 1.0)
        self.assertEqual(by["cough"], 1.0)
        d = res.daily
        self.assertTrue(((d.S + d.I + d.R) == 60).all())

    def test_isolation_flattens_curve(self):
        def attack(cond, responder):
            ars = []
            for s in range(6):
                r = ep.GenerativeEpidemic(FakeLLM(responder), n=100, condition=cond, seed=s).run(days=80)
                ars.append(1 - r.daily.S.iloc[-1] / 100)
            return np.mean(ars)
        base = attack("base", ep.rule_responder())
        full = attack("full", ep.rule_responder(news_threshold=1.0))
        self.assertGreater(base, 0.5)
        self.assertLess(full, base)

    def test_message_and_invalid_and_budget(self):
        msg = "The council orders a lockdown today."
        llm = FakeLLM(ep.rule_responder(stay_on=(), message_keyword="lockdown"))
        res = ep.GenerativeEpidemic(llm, n=30, condition="full", message=lambda d, pct: msg if d >= 1 else None,
                                    seed=0).run(days=8, stop_when_clear=False)
        self.assertEqual(res.daily.out_share.iloc[0], 1.0)
        self.assertTrue((res.daily.out_share.iloc[1:] == 0.0).all())
        self.assertTrue((res.daily.new_cases.iloc[1:] == 0).all())
        self.assertIn(msg, llm.calls[30][0])

        junk = ep.GenerativeEpidemic(FakeLLM(lambda s, u: "maybe"), n=20, retries=1, seed=0).run(days=3,
                                                                                               stop_when_clear=False)
        self.assertEqual(junk.invalid, 60)
        self.assertEqual(junk.calls, 120)                    # one retry per invalid reply
        self.assertTrue((junk.daily.out_share == 1.0).all())   # non-conforming -> "No" (paper)
        self.assertFalse(junk.log.valid.any())

        b = ep.GenerativeEpidemic(FakeLLM(ep.rule_responder()), n=20, max_calls=50, seed=0).run(days=10)
        self.assertEqual(b.stopped, "budget")
        self.assertEqual(b.calls, 40)

    def test_grid_contact_and_validation(self):
        res = ep.GenerativeEpidemic(FakeLLM(ep.rule_responder()), n=50, contact="grid", shape=(8, 8), radius=1,
                                    condition="self_health", seed=0).run(days=20)
        self.assertGreater(len(res.daily), 0)
        with self.assertRaises(ValueError):
            ep.GenerativeEpidemic(FakeLLM(), condition="news")
        self.assertAlmostEqual(ep.GenerativeEpidemic(FakeLLM(), n=5).r0_nominal, 3.0)


if __name__ == "__main__":
    unittest.main()
