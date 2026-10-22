"""Hyperband / ASHA on a synthetic noisy objective (offline, FakeLLM, no network, no data rows)."""
import math
import time
import unittest

import numpy as np

from simengine.llm import FakeLLM
from simengine.research import Space, successive_halving
from simengine.research.hyperband import (ASHA, FidelityRunner, Job, MultiFidelity, UnitRunner, asha, hyperband,
                                          hyperband_brackets, pool_sampler, space_sampler)

N = 81
UNIT_NOISE = np.random.default_rng(1).normal(size=N)          # shared per-unit difficulty (common to all configs)


def true_loss(cfg):
    """Lower is better; optimum x=0.7, method 'b'."""
    return (cfg["x"] - 0.7) ** 2 + (0.0 if cfg.get("method", "b") == "b" else 0.05)


def own_noise(cfg, idx):
    seed = int(round(cfg["x"] * 1e6)) + (7 if cfg.get("method") == "a" else 0)
    return np.random.default_rng(seed).normal(size=N)[idx]


def score_units(cfg, idx):
    return true_loss(cfg) + 0.3 * UNIT_NOISE[idx] + 0.3 * own_noise(cfg, idx)


def space():
    return Space({"x": (0.0, 1.0), "method": ["a", "b"]}, defaults={"x": 0.2, "method": "a"})


class TestBrackets(unittest.TestCase):
    def test_algorithm1_plan_r81(self):
        plan = hyperband_brackets(81, 1, 3)
        self.assertEqual([b["s"] for b in plan], [4, 3, 2, 1, 0])
        self.assertEqual([b["n"] for b in plan], [81, 34, 15, 8, 5])          # n = ceil(5 * 3^s / (s + 1))
        self.assertEqual(plan[0]["counts"], [81, 27, 9, 3, 1])
        self.assertEqual(plan[0]["budgets"], [1, 3, 9, 27, 81])
        self.assertEqual(plan[1]["counts"], [34, 11, 3, 1])
        self.assertEqual(plan[2]["counts"], [15, 5, 1])
        self.assertEqual(plan[3]["counts"], [8, 2])
        self.assertEqual([b["cost"] for b in plan], [405, 363, 351, 378, 405])
        self.assertTrue(all(b["budgets"][-1] == 81 for b in plan))

    def test_min_budget_and_non_power(self):
        plan = hyperband_brackets(90, 10, 3)
        self.assertEqual(plan[0]["budgets"], [10, 30, 90])
        self.assertEqual(len(hyperband_brackets(100, 1, 3)), 5)              # floor(log3 100) = 4
        self.assertEqual(len(hyperband_brackets(8, 1, 2)), 4)                # log2 8 = 3 exactly
        with self.assertRaises(ValueError):
            hyperband_brackets(81, 1, 1)
        with self.assertRaises(ValueError):
            hyperband_brackets(5, 10, 3)


class TestFloorRounding(unittest.TestCase):
    def test_floor_vs_ceil(self):
        cands = [{"v": v} for v in range(8)]
        ev = lambda cs, f: [c["v"] for c in cs]
        ceil_ = successive_halving(ev, cands, higher_is_better=False, fidelities=(0.1, 0.5, 1.0), eta=3)
        floor_ = successive_halving(ev, cands, higher_is_better=False, fidelities=(0.1, 0.5, 1.0), eta=3,
                                    rounding="floor")
        self.assertEqual([len(r) for r in ceil_["rungs"]], [8, 3, 1])
        self.assertEqual([len(r) for r in floor_["rungs"]], [8, 2, 1])
        self.assertEqual(floor_["winner"], {"v": 0})
        with self.assertRaises(ValueError):
            successive_halving(ev, cands, higher_is_better=False, rounding="round")


class TestRunners(unittest.TestCase):
    def test_protocol_and_cache(self):
        seen = []
        mf = MultiFidelity(lambda c, b: seen.append(b) or c["x"] * b, 10, unit="llm_calls")
        self.assertIsInstance(mf, FidelityRunner)
        self.assertEqual(mf({"x": 1.0}, 4), 4.0)
        self.assertEqual(mf({"x": 1.0}, 4), 4.0)
        self.assertEqual((mf.spent, mf.calls, seen), (4, 1, [4]))
        self.assertEqual(mf.screen([{"x": 2.0}], 0.5), [10.0])
        self.assertEqual(mf.budget_of(0.01), 1)
        with self.assertRaises(ValueError):
            mf({"x": 1.0}, 11)

    def test_unit_runner_resume_charges_new_units_only(self):
        run = UnitRunner(score_units, N, seed=3)
        cfg = {"x": 0.5, "method": "b"}
        a = run(cfg, 9)
        b = run(cfg, 27)
        self.assertEqual(run.spent, 27)
        self.assertEqual(len(run.per_unit(cfg)), 27)
        np.testing.assert_allclose(a, score_units(cfg, run.order[:9]).mean())
        np.testing.assert_allclose(b, score_units(cfg, run.order[:27]).mean())
        fresh = UnitRunner(score_units, N, seed=3, resume=False)
        fresh(cfg, 9)
        fresh(cfg, 27)
        self.assertEqual(fresh.spent, 36)
        with self.assertRaises(ValueError):
            UnitRunner(score_units, N, max_budget=N + 1)

    def test_unit_runner_wrong_length(self):
        run = UnitRunner(lambda c, idx: np.zeros(len(idx) + 1), N)
        with self.assertRaises(ValueError):
            run({"x": 0.1}, 3)

    def test_budget_is_llm_calls(self):
        llm = FakeLLM(lambda s, u: str(int(u.split()[-1]) % 5))

        def ask(cfg, idx):
            replies = llm.map([("rate", f"x={cfg['x']:.3f} unit {i}") for i in idx])
            return [abs(float(r.text) - 4 * cfg["x"]) for r in replies]

        run = UnitRunner(ask, 27, unit="llm_calls")
        out = hyperband(run, space_sampler(Space({"x": (0.0, 1.0)})), higher_is_better=False, eta=3)
        self.assertEqual(len(llm.calls), run.spent)
        self.assertLessEqual(run.spent, out["budget"])
        self.assertIsNotNone(out["winner"])


class TestHyperband(unittest.TestCase):
    def test_finds_good_config_and_counts_budget(self):
        run = UnitRunner(score_units, N)
        out = hyperband(run, space(), higher_is_better=False, seed=0)
        self.assertEqual(out["budget"], sum(b["cost"] for b in hyperband_brackets(N)))
        self.assertEqual(out["evaluations"], sum(sum(b["counts"]) for b in hyperband_brackets(N)))
        self.assertEqual(len(out["brackets"]), 5)
        self.assertLess(true_loss(out["winner"]), 0.02)
        self.assertLess(run.spent, out["budget"])                    # resume saves the lower-rung units
        best = min(b["score"] for b in out["brackets"])
        self.assertEqual(out["score"], best)

    def test_single_bracket_is_plain_halving(self):
        rng = np.random.default_rng(5)
        pool = [{"x": float(x), "method": "b"} for x in rng.uniform(0, 1, 81)]
        run = UnitRunner(score_units, N)
        hb = hyperband(run, pool_sampler(pool, shuffle=False), higher_is_better=False, brackets=[4])
        run2 = UnitRunner(score_units, N)
        sh = successive_halving(run2.screen, pool, higher_is_better=False,
                                fidelities=tuple(b / 81 for b in (1, 3, 9, 27, 81)), eta=3, rounding="floor")
        self.assertEqual(hb["winner"], sh["winner"])
        self.assertEqual(hb["brackets"][0]["rungs"], sh["rungs"])

    def test_plain_screen_callable_and_higher_is_better(self):
        calls = []

        def screen(cfgs, f):
            calls.append(f)
            return [-(c["x"] - 0.3) ** 2 - 0.05 * (1 - f) * c["x"] for c in cfgs]

        out = hyperband(screen, space_sampler(Space({"x": (0.0, 1.0)})), higher_is_better=True, max_budget=27)
        self.assertLess(abs(out["winner"]["x"] - 0.3), 0.1)
        self.assertAlmostEqual(min(calls), 1 / 27)
        self.assertEqual(max(calls), 1.0)
        with self.assertRaises(ValueError):
            hyperband(screen, space(), higher_is_better=True)        # no max_budget

    def test_pool_exhaustion_and_repeats(self):
        pool = [{"x": x, "method": "b"} for x in np.linspace(0, 1, 20)]
        run = UnitRunner(score_units, N)
        out = hyperband(run, pool, higher_is_better=False, repeats=2)
        self.assertEqual(sum(b["n"] for b in out["brackets"]), 20)   # every candidate drawn exactly once
        self.assertIsNotNone(out["winner"])

    def test_nan_scores(self):
        def fn(cfg, b):
            return float("nan") if cfg["x"] < 0.5 else true_loss(cfg)

        out = hyperband(MultiFidelity(fn, 27), space_sampler(Space({"x": (0.0, 1.0)})), higher_is_better=False)
        self.assertGreaterEqual(out["winner"]["x"], 0.5)
        self.assertFalse(math.isnan(out["score"]))

    def test_deterministic(self):
        a = hyperband(UnitRunner(score_units, N), space(), higher_is_better=False, seed=4)
        b = hyperband(UnitRunner(score_units, N), space(), higher_is_better=False, seed=4)
        self.assertEqual(a["winner"], b["winner"])


class TestASHAScheduler(unittest.TestCase):
    def test_rung_budgets(self):
        self.assertEqual(ASHA(81, 1, 3, higher_is_better=False).budgets, [1, 3, 9, 27, 81])
        self.assertEqual(ASHA(81, 1, 3, s=1, higher_is_better=False).budgets, [3, 9, 27, 81])
        self.assertEqual(ASHA(100, 1, 3, higher_is_better=False).budgets[-1], 100)
        with self.assertRaises(ValueError):
            ASHA(81, 1, 3, s=5, higher_is_better=False)

    def test_promotion_rule_out_of_order(self):
        sch = ASHA(9, 1, 3, higher_is_better=False)
        cfgs = iter([{"x": v} for v in (0.5, 0.1, 0.9, 0.3, 0.8, 0.7)])
        new = lambda: next(cfgs, None)
        jobs = [sch.get_job(new) for _ in range(3)]
        self.assertTrue(all(j.rung == 0 for j in jobs))
        sch.report(jobs[2], 0.9)                  # completions arrive out of order
        sch.report(jobs[0], 0.5)
        j = sch.get_job(new)                      # |rung 0| = 2 -> floor(2/3) = 0 promotable: new config
        self.assertEqual((j.rung, j.trial), (0, 3))
        sch.report(jobs[1], 0.1)                  # |rung 0| = 3 -> top 1 = trial 1
        p = sch.get_job(new)
        self.assertEqual((p.trial, p.rung, p.budget), (1, 1, 3))
        q = sch.get_job(new)                      # trial 1 already promoted -> new config
        self.assertEqual(q.rung, 0)
        sch.report(p, 0.2)
        self.assertEqual(sch.best()["trial"], 1)
        self.assertEqual(sch.best()["rung"], 1)
        self.assertEqual(sch.issued, 1 + 1 + 1 + 1 + 3 + 1)

    def test_nan_never_promoted(self):
        sch = ASHA(3, 1, 3, higher_is_better=True)
        jobs = [sch.get_job(lambda: {"x": 0}) for _ in range(3)]
        for j in jobs:
            sch.report(j, float("nan"))
        self.assertEqual(sch.get_job(lambda: None), None)
        self.assertIsNone(sch.best())


class TestASHARun(unittest.TestCase):
    def test_sequential_budget_and_quality(self):
        run = UnitRunner(score_units, N)
        out = asha(run, space(), higher_is_better=False, total_budget=1902, seed=0)
        self.assertGreaterEqual(out["budget"], 1902)
        self.assertLess(out["budget"] - 1902, N)
        self.assertEqual(out["rung"], 4)
        self.assertLess(true_loss(out["winner"]), 0.02)
        self.assertGreater(out["promotions"], 0)
        again = asha(UnitRunner(score_units, N), space(), higher_is_better=False, total_budget=1902, seed=0)
        self.assertEqual(again["winner"], out["winner"])

    def test_max_trials_and_pool(self):
        out = asha(UnitRunner(score_units, N), space(), higher_is_better=False, max_trials=30)
        self.assertEqual(out["trials"], 30)
        pool = [{"x": x, "method": "b"} for x in np.linspace(0, 1, 9)]
        out = asha(UnitRunner(score_units, N), pool, higher_is_better=False)
        self.assertEqual(out["trials"], 9)
        self.assertEqual([len(r) for r in out["rungs"]], [9, 3, 1, 0, 0])    # nothing reaches 27 with 9 configs
        self.assertEqual(out["rung"], 2)
        with self.assertRaises(ValueError):
            asha(UnitRunner(score_units, N), space(), higher_is_better=False)

    def test_threaded_workers_and_crash(self):
        def fn(cfg, b):
            time.sleep(0.0005 * (1 + (hash(round(cfg["x"], 6)) % 3)))
            if cfg["x"] < 0.05:
                raise RuntimeError("broken config")
            return true_loss(cfg) + 0.01 * np.sin(b * cfg["x"])

        run = MultiFidelity(fn, 27)
        out = asha(run, space_sampler(Space({"x": (0.0, 1.0)})), higher_is_better=False, total_budget=600,
                   n_workers=4, seed=1)
        self.assertLess(out["budget"] - 600, 4 * 27)
        self.assertLess(abs(out["winner"]["x"] - 0.7), 0.15)
        self.assertEqual(out["rung"], len(out["budgets"]) - 1)

    def test_job_dataclass(self):
        j = Job(0, {"x": 1}, 0, 1)
        self.assertEqual(j.budget, 1)


if __name__ == "__main__":
    unittest.main()
