"""Bayesian optimisation proposers (GP-EI/UCB, TPE-lite) on Branin and a mixed categorical toy (synthetic, offline)."""
import hashlib
import math
import unittest

import numpy as np
import pandas as pd

from simengine.research import Loop, RandomProposer, Space
from simengine.research.bayesopt import (BRANIN_MIN, TPE, GPProposer, Parzen1D, SpaceEncoder, TPEProposer, branin,
                                         branin_space, expected_improvement, minimize, mixed_toy, mixed_toy_space,
                                         observations, ucb_beta, upper_confidence_bound)
from simengine.research.bayesopt import _cat_probs

N = 80


def _u(text, n):
    return np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)).normal(size=n)


def branin_runner(cfg):
    """Per-unit Branin loss: shared unit difficulty (sd 5) + small own noise (sd 1); lower is better."""
    s = cfg["split"]
    own = _u(f"{cfg['x1']:.6f},{cfg['x2']:.6f},{s}", N)
    err = float(branin(cfg["x1"], cfg["x2"])) + 5 * _u("units" + s, N) + own
    return {"per_unit": pd.DataFrame({"unit": range(N), "err": err})}


def toy_runner(cfg):
    """Mixed toy as a higher-is-better score; method 'd' crashes."""
    if cfg["method"] == "d":
        raise RuntimeError("method d is broken")
    s = cfg["split"]
    own = 0.01 * _u(f"{sorted(cfg.items())}", N)
    return {"per_unit": pd.DataFrame({"unit": range(N), "score": -mixed_toy(cfg) + 0.2 * _u("u" + s, N) + own})}


def run_loop(runner, space, proposer, primary="err", hib=False, turns=12):
    loop = Loop(runner, space, proposer=proposer, primary=primary, higher_is_better=hib, stop_after=None,
                max_turns=turns)
    return loop.run(finalize=False)


def incumbent(out):
    return [t for t in out["history"] if t["status"] == "keep"][-1]["cfg"]


class TestFunctions(unittest.TestCase):
    def test_branin_minima(self):
        for x1, x2 in [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]:
            self.assertAlmostEqual(float(branin(x1, x2)), BRANIN_MIN, places=5)
        g = np.meshgrid(np.linspace(-5, 10, 301), np.linspace(0, 15, 301))
        self.assertGreaterEqual(float(branin(*g).min()), BRANIN_MIN - 1e-6)

    def test_toy_minimum(self):
        self.assertAlmostEqual(mixed_toy({"method": "b", "x": 0.75, "k": 6}), 0.0)
        self.assertGreater(mixed_toy({"method": "a", "x": 0.75, "k": 6}), 0.5)


class TestEncoder(unittest.TestCase):
    def test_roundtrip_onehot_int(self):
        sp = Space({"x": (0.0, 2.0), "k": (1, 9), "m": ["a", "b", "c"], "f": ["p", "q"]},
                   defaults={"x": 1.0, "k": 3, "m": "a", "f": "p"}, fixed=["f"], forbidden={"m": ["c"]})
        enc = SpaceEncoder(sp)
        self.assertEqual(enc.keys, ["x", "k", "m"])
        self.assertEqual(enc.dim, 4)                      # x, k, one-hot over the 2 allowed methods
        X = enc.encode([{"x": 0.5, "k": 9, "m": "b"}, {"x": 2.0, "k": 1, "m": "c"}])
        np.testing.assert_allclose(X, [[0.25, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
        U, C = enc.unit([{"x": 0.5, "k": 5, "m": "b"}])
        back = enc.from_unit(U, C, {"f": "p"})[0]
        self.assertEqual(back, {"f": "p", "x": 0.5, "k": 5, "m": "b"})
        self.assertEqual(enc.value("k", 0.49), 5)
        self.assertIsInstance(enc.value("k", 0.49), int)
        for c in enc.sample(50, np.random.default_rng(0)):
            self.assertEqual(sp.validate(c)[1], [])
            self.assertNotEqual(c["m"], "c")


class TestAcquisition(unittest.TestCase):
    def test_ei_matches_monte_carlo(self):
        rng = np.random.default_rng(0)
        for mu, sd, best in [(0.0, 1.0, 0.5), (1.0, 0.3, 0.2), (-1.0, 2.0, 0.0)]:
            f = rng.normal(mu, sd, 400_000)
            mc = np.maximum(best - f, 0).mean()
            self.assertAlmostEqual(float(expected_improvement([mu], [sd], best)[0]), mc, delta=0.01)
        np.testing.assert_allclose(expected_improvement([0.0, 1.0], [0.0, 0.0], 0.5), [0.5, 0.0])

    def test_ucb_beta(self):
        self.assertAlmostEqual(ucb_beta(3, 100, delta=0.1, scale=1.0),
                               2 * math.log(100 * 9 * math.pi ** 2 / 0.6))
        self.assertAlmostEqual(ucb_beta(3, 100), 0.2 * ucb_beta(3, 100, scale=1.0))
        np.testing.assert_allclose(upper_confidence_bound([1.0], [2.0], 4.0), [3.0])


class TestParzen(unittest.TestCase):
    def test_density_and_bandwidth(self):
        p = Parzen1D.fit(np.array([0.2, 0.3, 0.9]))
        # neighbours 0 | 0.2 0.3 0.9 | 1: sd = max(left, right) gaps, clipped below at 1/min(100, 1+3)
        np.testing.assert_allclose(p.sds, [1.0, 0.25, 0.6, 0.6])
        u = np.linspace(0, 1, 20001)
        self.assertAlmostEqual(float(getattr(np, "trapezoid", getattr(np, "trapz", None))(np.exp(p.logpdf(u)), u)), 1.0, places=3)
        s = p.sample(5000, np.random.default_rng(1))
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_categorical_posterior(self):
        np.testing.assert_allclose(_cat_probs(np.array([0, 0, 2, -1]), 3), [3 / 6, 1 / 6, 2 / 6])

    def test_tpe_prefers_good_region(self):
        sp = Space({"x": (0.0, 1.0), "m": ["a", "b"]})
        rng = np.random.default_rng(0)
        cfgs = [{"x": float(x), "m": m} for x, m in zip(rng.uniform(size=40), rng.choice(["a", "b"], 40))]
        losses = np.array([(c["x"] - 0.2) ** 2 + (0.5 if c["m"] == "a" else 0) for c in cfgs])
        tpe = TPE(SpaceEncoder(sp), cfgs, losses)
        s = tpe.score([{"x": 0.2, "m": "b"}, {"x": 0.9, "m": "b"}, {"x": 0.2, "m": "a"}])
        self.assertGreater(s[0], s[1])
        self.assertGreater(s[0], s[2])
        draws = tpe.sample(400, np.random.default_rng(2))
        self.assertLess(abs(np.median([d["x"] for d in draws]) - 0.2), 0.15)


class TestObservations(unittest.TestCase):
    def test_crash_and_direction(self):
        hist = [{"status": "keep", "cfg": {"a": 1}, "metrics": {"s": {"value": 2.0}}},
                {"status": "crash", "cfg": {"a": 2}},
                {"status": "discard", "cfg": {"a": 3}, "metrics": {"s": {"value": 5.0}}}]
        cfgs, y = observations(hist, "s", higher_is_better=True)
        self.assertEqual([c["a"] for c in cfgs], [1, 3, 2])
        np.testing.assert_allclose(y, [-2.0, -5.0, -2.0])     # crash = worst loss seen
        self.assertEqual(len(observations(hist, "s", True, crash="skip")[1]), 2)


class TestMinimize(unittest.TestCase):
    def f(self, c):
        return float(branin(c["x1"], c["x2"]))

    def test_gp_and_tpe_beat_random_on_branin(self):
        res = {m: [minimize(self.f, branin_space(), method=m, n_calls=25, seed=s)["best"] for s in range(3)]
               for m in ("random", "gp", "tpe")}
        self.assertLess(np.mean(res["gp"]), np.mean(res["random"]))
        self.assertLess(np.mean(res["gp"]), 1.0)
        self.assertLess(np.mean(res["tpe"]), np.mean(res["random"]))

    def test_mixed_toy_finds_method(self):
        for m in ("gp", "tpe"):
            r = minimize(mixed_toy, mixed_toy_space(), method=m, n_calls=30, seed=0)
            self.assertEqual(r["best_cfg"]["method"], "b", m)
            self.assertLess(r["best"], 0.05, m)

    def test_trace_crash_and_direction(self):
        def obj(c):
            if c["x"] > 0.9:
                raise ValueError("boom")
            return -(c["x"] - 0.5) ** 2

        sp = Space({"x": (0.0, 1.0)})
        r = minimize(obj, sp, method="gp", n_calls=15, seed=1, higher_is_better=True,
                     x0=[{"x": 0.95}, {"x": 0.1}])
        self.assertTrue(math.isnan(r["history"][0]["value"]))
        self.assertEqual(r["history"][0]["method"], "x0")
        self.assertGreater(r["best"], -0.01)
        tr = [t for t in r["trace"] if not math.isnan(t)]
        self.assertEqual(tr, sorted(tr))                     # best-so-far never gets worse
        self.assertEqual(len({h["cfg"]["x"] for h in r["history"]}), len(r["history"]))   # no repeats
        with self.assertRaises(ValueError):
            minimize(obj, sp, method="cmaes")


class TestLoopProposers(unittest.TestCase):
    def _check(self, out, space):
        hist = out["history"]
        self.assertEqual(len(hist), 13)
        hashes = [t["cfg_hash"] for t in hist]
        self.assertEqual(len(hashes), len(set(hashes)))
        for t in hist[1:]:
            self.assertEqual(len(t["proposal"]["change"]), 1)
            self.assertEqual(space.validate(t["cfg"])[1], [])

    def test_gp_and_tpe_in_loop_branin(self):
        start = float(branin(2.5, 7.5))
        for p in (GPProposer(seed=3), GPProposer("ucb", seed=3), TPEProposer(seed=3)):
            out = run_loop(branin_runner, branin_space(), p)
            self._check(out, branin_space())
            inc = incumbent(out)
            self.assertLess(float(branin(inc["x1"], inc["x2"])), start / 3, p.name)
            kinds = [t["proposal"]["kind"] for t in out["history"][1:]]
            self.assertEqual(kinds[0], "random-init")
            self.assertIn(p.name if p.name != "gp" else p.acquisition, kinds)

    def test_deterministic(self):
        a = run_loop(branin_runner, branin_space(), GPProposer(seed=5), turns=6)
        b = run_loop(branin_runner, branin_space(), GPProposer(seed=5), turns=6)
        self.assertEqual([t["cfg"] for t in a["history"]], [t["cfg"] for t in b["history"]])

    def test_categorical_toy_with_crashes(self):
        sp = mixed_toy_space()
        for p in (GPProposer(seed=0), TPEProposer(seed=0)):
            out = run_loop(toy_runner, sp, p, primary="score", hib=True, turns=14)
            hist = out["history"]
            self.assertEqual(len(hist), 15)
            self.assertLess(mixed_toy(incumbent(out)), mixed_toy(sp.validate({})[0]), p.name)
            notes = [t["proposal"].get("notes") for t in hist[1:] if t["proposal"]["kind"] != "random-init"]
            self.assertTrue(all(n and n["n_obs"] >= 3 for n in notes))

    def test_random_proposer_same_budget_runs(self):
        out = run_loop(branin_runner, branin_space(), RandomProposer(seed=3))
        self._check(out, branin_space())


if __name__ == "__main__":
    unittest.main()
