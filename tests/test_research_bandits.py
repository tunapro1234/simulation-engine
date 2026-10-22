"""simengine.research.bandits: schedules, identification rates on simulated arms, regret, LUCB stopping, RacingProposer.

Offline, synthetic arms only (no data rows, no LLM). Replications are vectorised, so the rate tests run in seconds.
"""
import hashlib
import json
import math
import unittest

import numpy as np
import pandas as pd

from simengine.research.bandits import (ALGORITHMS, AUDIBERT_BUBECK_2010, RacingProposer, SimulatedArms, canonical,
                                        compare, identify_best, logbar, lucb, lucb_beta, simulate, sr_schedule,
                                        successive_rejects, thompson_beta, thompson_gaussian,
                                        ucb1)
from simengine.research.loop import Loop
from simengine.research.proposers import ProposalContext
from simengine.research.space import Space


class TestSchedules(unittest.TestCase):
    def test_logbar(self):
        self.assertAlmostEqual(logbar(2), 1.0)
        self.assertAlmostEqual(logbar(4), 0.5 + 1 / 2 + 1 / 3 + 1 / 4)

    def test_sr_schedule_matches_paper_formula_and_budget(self):
        for K, n in [(2, 10), (5, 100), (6, 600), (20, 2000), (30, 12000), (7, 7)]:
            nk = sr_schedule(K, n)
            exp = [math.ceil((n - K) / (logbar(K) * (K + 1 - k))) for k in range(1, K)]
            self.assertEqual(nk.tolist(), exp)
            # phases: arms dismissed at the end of phase k were pulled n_k times; the two finalists n_{K-1} times
            self.assertLessEqual(int(nk.sum() + nk[-1]), n)
            self.assertTrue((np.diff(nk) >= 0).all())

    def test_sr_needs_budget_and_arms(self):
        with self.assertRaises(ValueError):
            sr_schedule(5, 4)
        with self.assertRaises(ValueError):
            sr_schedule(1, 10)

    def test_lucb_beta(self):
        b = lucb_beta(4, 3, 5, 0.1)
        self.assertAlmostEqual(float(b), math.sqrt(math.log(1.25 * 5 * 3 ** 4 / 0.1) / 8))
        self.assertAlmostEqual(float(lucb_beta(4, 3, 5, 0.1, scale=2.0)), 2 * float(b))

    def test_aliases(self):
        self.assertEqual(canonical("SR"), "successive_rejects")
        self.assertEqual(canonical("ts-gaussian"), "thompson_gaussian")
        with self.assertRaises(ValueError):
            canonical("exp3")


class TestSingleRun(unittest.TestCase):
    def test_successive_rejects_allocation(self):
        arms = SimulatedArms([0.2, 0.9, 0.5, 0.4, 0.3], seed=0)
        res = successive_rejects(arms, 5, 100, seed=0)
        nk = sr_schedule(5, 100)
        self.assertEqual(sorted(res.pulls.tolist()), sorted(nk.tolist() + [nk[-1]]))
        self.assertEqual(res.budget_used, int(res.pulls.sum()))
        self.assertLessEqual(res.budget_used, 100)
        self.assertEqual(len(res.rejected), 4)
        self.assertNotIn(res.best, res.rejected)
        self.assertEqual(sorted(res.rejected + [res.best]), list(range(5)))
        self.assertEqual(res.stopped, "phases")
        self.assertEqual(len(res.arms), res.budget_used)

    def test_deterministic_given_seeds(self):
        a = [identify_best(SimulatedArms([0.5, 0.45, 0.4], seed=3), 3, 200, alg, seed=1).to_dict() for alg in
             ALGORITHMS]
        b = [identify_best(SimulatedArms([0.5, 0.45, 0.4], seed=3), 3, 200, alg, seed=1).to_dict() for alg in
             ALGORITHMS]
        self.assertEqual(a, b)

    def test_ucb1_plays_each_arm_once_first(self):
        res = ucb1(SimulatedArms([0.1, 0.9, 0.5], seed=0), 3, 50)
        self.assertEqual(res.arms[:3], [0, 1, 2])
        self.assertEqual(res.budget_used, 50)
        self.assertEqual(res.best, 1)

    def test_lower_is_better_and_units(self):
        rng = np.random.default_rng(0)
        mu = [3.0, 1.0, 2.0]
        pull = lambda i: mu[i] + 0.3 * rng.standard_normal()
        for alg in ALGORITHMS:
            kw = {"bounds": (0.0, 4.0)} if alg != "thompson_gaussian" else {"sigma": 0.3}
            res = identify_best(pull, 3, 300, alg, higher_is_better=False, seed=0, **kw)
            self.assertEqual(res.best, 1, alg)
            self.assertAlmostEqual(float(res.means[res.best]), 1.0, delta=0.3)

    def test_crashed_arm_is_never_recommended(self):
        rng = np.random.default_rng(1)
        pull = lambda i: float("nan") if i == 0 else [0, 0.3, 0.6][i] + 0.1 * rng.standard_normal()
        for alg in ALGORITHMS:
            res = identify_best(pull, 3, 60, alg, seed=0)
            self.assertNotEqual(res.best, 0, alg)
            self.assertTrue(math.isnan(res.rewards[res.arms.index(0)]))

    def test_lucb_stops_when_confident(self):
        res = lucb(SimulatedArms([0.95, 0.1, 0.05], seed=0), 3, delta=0.05)
        self.assertEqual(res.stopped, "confident")
        self.assertEqual(res.best, 0)
        res = lucb(SimulatedArms([0.51, 0.5], seed=0), 2, budget=101, delta=0.05)
        self.assertEqual(res.stopped, "budget")
        self.assertLessEqual(res.budget_used, 101)

    def test_lucb_round_index_follows_paper(self):
        # deterministic rewards 1 / 0: after c two-arm checks each arm has u = c samples and the paper's round index is
        # t = K + c (rounds 1..K are the initial pulls); LUCB1 stops at the first c with 2 beta(c, K + c) < 1
        K, delta = 2, 0.05
        c = next(c for c in range(1, 10_000) if 2 * float(lucb_beta(c, K + c, K, delta)) < 1)
        res = lucb(lambda i: float(i == 0), K, budget=10_000, delta=delta)
        self.assertEqual(res.stopped, "confident")
        self.assertEqual(res.budget_used, K + 2 * (c - 1))
        c_wrong = next(c for c in range(1, 10_000) if 2 * float(lucb_beta(c, c, K, delta)) < 1)
        self.assertNotEqual(c, c_wrong)   # the test can tell t = K + c from t = c

    def test_thompson_beta_lower_is_better_default_bounds(self):
        # losses in [0, 1] without bounds=: the Bernoulli trial uses the negated reward rescaled from (-1, 0), so the
        # sampler still concentrates on the lowest-loss arm (it used to see only failures and pull at random)
        rng = np.random.default_rng(0)
        mu = [0.1, 0.6, 0.7, 0.8]
        res = thompson_beta(lambda i: float(rng.random() < mu[i]), 4, 400, higher_is_better=False, seed=0)
        self.assertEqual(res.best, 0)
        self.assertGreater(res.pulls[0], 0.6 * 400)

    def test_single_arm(self):
        res = identify_best(lambda i: 1.0, 1, 10, "successive_rejects")
        self.assertEqual((res.best, res.stopped), (0, "single arm"))

    def test_thompson_warmup(self):
        # a confident prior far below the rewards: without warmup the first sampled arm is never left (documented)
        mk = lambda: SimulatedArms([0.3, 0.2, 0.25, 0.1], kind="gaussian", sigma=0.05, seed=0)
        cold = thompson_gaussian(mk(), 4, 40, sigma=0.05, prior_mean=-1.0, seed=0)
        warm = thompson_gaussian(mk(), 4, 40, sigma=0.05, prior_mean=-1.0, warmup=1, seed=0)
        self.assertEqual(int((cold.pulls > 0).sum()), 1)
        self.assertEqual(warm.arms[:4], [0, 1, 2, 3])
        self.assertTrue((warm.pulls >= 1).all())
        self.assertEqual(warm.budget_used, 40)

    def test_gaussian_prior_mean(self):
        arms = SimulatedArms([10.0, 10.5, 9.0], kind="gaussian", sigma=1.0, seed=4)
        res = thompson_gaussian(arms, 3, 400, prior_mean=10.0, seed=0)
        self.assertEqual(res.best, 1)


class TestIdentificationRates(unittest.TestCase):
    """Probability of error over 1000 simulated runs (binomial SE <= 0.016)."""

    def test_easy_problem_all_algorithms(self):
        arms = SimulatedArms([0.8, 0.5, 0.4], seed=0)
        df = compare(arms, 300, reps=1000, seed=0)
        self.assertEqual(len(df), len(ALGORITHMS))
        self.assertTrue((df.error_rate <= 0.02).all(), df[["algorithm", "error_rate"]])

    def test_adaptive_beats_uniform_on_ab2010_exp4(self):
        mu, n = AUDIBERT_BUBECK_2010["exp4"]
        df = compare(SimulatedArms(mu, seed=1), n, reps=1000, seed=0).set_index("algorithm")
        unif = df.loc["uniform", "error_rate"]
        self.assertGreater(unif, 0.2)
        for a in ("successive_rejects", "lucb", "ucb1", "thompson_beta", "thompson_gaussian"):
            self.assertLess(df.loc[a, "error_rate"], 0.19, a)
            self.assertLess(df.loc[a, "error_rate"], unif - 0.04, a)

    def test_sr_beats_uniform_on_ab2010_exp1_and_exp7(self):
        for name in ("exp1", "exp7"):
            mu, n = AUDIBERT_BUBECK_2010[name]
            arms = SimulatedArms(mu, seed=2)
            sr = simulate(arms, "successive_rejects", n, reps=1000, seed=0)["error_rate"]
            un = simulate(arms, "uniform", n, reps=1000, seed=0)["error_rate"]
            self.assertLess(sr, 0.25, name)
            self.assertGreater(un - sr, 0.15, name)

    def test_more_budget_fewer_errors(self):
        mu, _ = AUDIBERT_BUBECK_2010["exp7"]
        arms = SimulatedArms(mu, seed=5)
        e = [simulate(arms, "successive_rejects", n, reps=1000, seed=0)["error_rate"] for n in (3000, 6000, 12000)]
        self.assertTrue(e[0] > e[1] > e[2], e)

    def test_lucb_error_below_delta(self):
        arms = SimulatedArms([0.9, 0.5, 0.4], seed=6)
        r = simulate(arms, "lucb", None, reps=1000, seed=0, delta=0.05)
        self.assertEqual(r["confident_rate"], 1.0)
        self.assertLessEqual(r["error_rate"], 0.05)
        self.assertLess(r["mean_pulls"], 3000)

    def test_gaussian_arms(self):
        arms = SimulatedArms([1.0, 0.6, 0.5, 0.2], kind="gaussian", sigma=1.0, seed=3)
        e_ts = simulate(arms, "thompson_gaussian", 300, reps=1000, seed=0)["error_rate"]
        e_un = simulate(arms, "uniform", 300, reps=1000, seed=0)["error_rate"]
        self.assertLess(e_ts, 0.15)
        self.assertLessEqual(e_ts, e_un)

    def test_tied_best_refused(self):
        with self.assertRaises(ValueError):
            simulate(SimulatedArms([0.5, 0.5, 0.1]), "uniform", 30, reps=10)


class TestRegret(unittest.TestCase):
    def test_regret_policies_beat_uniform(self):
        arms = SimulatedArms([0.6, 0.5, 0.4, 0.3], seed=0)
        n = 3000
        reg = {a: simulate(arms, a, n, reps=300, seed=0)["cumulative_regret"]
               for a in ("ucb1", "thompson_beta", "thompson_gaussian", "uniform")}
        self.assertAlmostEqual(reg["uniform"], n / 4 * (0.1 + 0.2 + 0.3), delta=1e-9)
        for a in ("ucb1", "thompson_beta", "thompson_gaussian"):
            self.assertLess(reg[a], reg["uniform"] / 2, (a, reg))
        self.assertLess(reg["thompson_beta"], reg["ucb1"])

    def test_regret_grows_sublinearly(self):
        arms = SimulatedArms([0.6, 0.5, 0.4], seed=1)
        r1 = simulate(arms, "ucb1", 1000, reps=300, seed=0)["cumulative_regret"]
        r4 = simulate(arms, "ucb1", 4000, reps=300, seed=0)["cumulative_regret"]
        self.assertLess(r4, 2.5 * r1)


# ----------------------------------------------------------------- Loop integration

N_DEV = 240
CALLS = []


def _u(text, n):
    return np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)).normal(size=n)


def objective(cfg):
    """Lower is better; optimum x=0.7, method b. With cfg["fidelity"] = j the runner scores mini-batch j: 24 units
    with batch-specific noise (a fresh seed per batch)."""
    CALLS.append(dict(cfg))
    split = cfg["split"]
    effect = 2.0 * (cfg["x"] - 0.7) ** 2 + {"a": 0.15, "b": 0.0}[cfg["method"]]
    key = json.dumps({k: v for k, v in sorted(cfg.items()) if k not in ("split", "fidelity")})
    if "fidelity" in cfg:
        j = int(cfg["fidelity"])
        score = 1.0 + effect + 0.3 * _u(f"{key}-{split}-b{j}", 24)
        return {"per_unit": pd.DataFrame({"unit": [f"b{j}_{i}" for i in range(24)], "err": score})}
    score = 1.0 + _u(f"units-{split}", N_DEV) + effect + 0.05 * _u(f"{key}-{split}", N_DEV)
    return {"per_unit": pd.DataFrame({"unit": [f"u{i}" for i in range(N_DEV)], "err": score})}


class TestRacingProposer(unittest.TestCase):
    def setUp(self):
        CALLS.clear()

    def test_in_loop(self):
        sp = Space({"x": (0.0, 1.0), "method": ["a", "b"]}, defaults={"x": 0.1, "method": "a"})
        prop = RacingProposer("successive_rejects", pulls_per_arm=4, n_grid=5, seed=0)
        loop = Loop(objective, sp, prop, primary="err", higher_is_better=False, stop_after=2, max_turns=6,
                    ablation=False)
        out = loop.run()
        self.assertGreaterEqual(out["kept"], 1)
        props = [t["proposal"] for t in out["history"][1:]]
        self.assertTrue(all(p["rationale"].startswith("won a successive_rejects race") for p in props))
        fid = [c["fidelity"] for c in CALLS if "fidelity" in c]
        self.assertTrue(fid and all(float(f).is_integer() and f >= 1 for f in fid))
        self.assertTrue(all(c["split"] == "dev" for c in CALLS if "fidelity" in c))
        self.assertEqual(out["counters"]["screen_runs"], prop.evaluations)
        self.assertLess(abs(out["best_cfg"]["x"] - 0.7), 0.3)

    def test_direct_pull_picks_best_candidate(self):
        sp = Space({"x": (0.0, 1.0), "method": ["a", "b"]}, defaults={"x": 0.1, "method": "a"})
        inc = sp.validate({})[0]
        ctx = ProposalContext(turn=1, incumbent={"n": 0, "cfg": inc, "metrics": {}}, start=inc, history=[], space=sp,
                              seen=set(), primary="err", higher_is_better=False)
        rng = np.random.default_rng(0)
        pull = lambda cfg, j: 2.0 * (cfg["x"] - 0.7) ** 2 + {"a": 0.15, "b": 0.0}[cfg["method"]] \
            + 0.02 * rng.standard_normal()
        for alg in ("successive_rejects", "lucb", "thompson_gaussian", "ucb1"):
            p = RacingProposer(alg, pull=pull, pulls_per_arm=6, seed=0, **({"scale": 0.1} if alg in ("lucb", "ucb1")
                                                                            else {"sigma": 0.02} if alg ==
                                                                            "thompson_gaussian" else {}))
            out = p.propose(ctx)
            self.assertEqual(out["change"], {"x": 0.75}, alg)
            self.assertEqual(len(out["notes"]["race"]["candidates"]), len(p.candidates(ctx)))
            self.assertLessEqual(p.last.budget_used, 6 * len(p.candidates(ctx)))

    def test_no_screen_falls_back(self):
        sp = Space({"x": (0.0, 1.0)}, defaults={"x": 0.1})
        inc = sp.validate({})[0]
        ctx = ProposalContext(turn=1, incumbent={"n": 0, "cfg": inc, "metrics": {}}, start=inc, history=[], space=sp,
                              seen=set(), primary="err", higher_is_better=False)
        out = RacingProposer().propose(ctx)
        self.assertEqual(out["rationale"], "no screening available")


if __name__ == "__main__":
    unittest.main()
