"""Research loop on a synthetic noisy objective (offline, FakeLLM, no network)."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from simengine.llm import FakeLLM
from simengine.research import (GreedyCoordinateProposer, GridProposer, HalvingProposer, Loop, RandomProposer, Space,
                                build_prompt, control_arm, successive_halving)
from simengine.research.proposers import _ClaudePCLI, parse_observer_reply

N_UNITS = {"dev": 240, "select": 120, "test": 200}
CALLS = []
INTERRUPT = {"after": None}


def _u(seed_text, n):
    s = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(s).normal(size=n)


def objective(cfg):
    """Lower is better. Optimum near x=0.7, k=12, method 'b'. Per-unit difficulty is shared across configs
    (so paired comparisons are much tighter than unpaired CIs); each config adds small own noise."""
    CALLS.append(dict(cfg))
    if INTERRUPT["after"] is not None and len(CALLS) > INTERRUPT["after"]:
        raise KeyboardInterrupt("simulated kill")
    split = cfg["split"]
    n = N_UNITS[split]
    if "fidelity" in cfg:
        n = max(10, int(n * cfg["fidelity"]))
    if cfg.get("method") == "c":
        raise RuntimeError("method c is broken")
    base = _u(f"units-{split}", N_UNITS[split])[:n]
    effect = 2.0 * (cfg["x"] - 0.7) ** 2 + 0.004 * (cfg["k"] - 12) ** 2 + {"a": 0.15, "b": 0.0}[cfg["method"]]
    key = json.dumps({k: v for k, v in sorted(cfg.items()) if k not in ("split", "fidelity")})
    noise = 0.05 * _u(f"{key}-{split}", N_UNITS[split])[:n]
    score = 1.0 + base + effect + noise
    units = [f"{split}{i}" for i in range(n)]
    return {"per_unit": pd.DataFrame({"unit": units, "err": score, "grp": [f"g{i % 3}" for i in range(n)]}),
            "metrics": {"effect": effect}}


def flat(cfg):
    """No config has any effect: every change must be discarded."""
    split = cfg["split"]
    n = N_UNITS[split]
    key = json.dumps({k: v for k, v in sorted(cfg.items()) if k != "split"})
    return {"per_unit": pd.DataFrame({"unit": range(n), "err": 1 + _u(f"flat-{split}", n) + 0.05 * _u(key, n)})}


def space():
    return Space({"x": (0.0, 1.0), "k": (1, 24), "method": ["a", "b", "c"]},
                 defaults={"x": 0.2, "k": 4, "method": "a"})


def make(workdir=None, runner=objective, **kw):
    kw.setdefault("primary", "err")
    kw.setdefault("higher_is_better", False)
    return Loop(runner, space(), workdir=workdir, **kw)


class TestLoopRandom(unittest.TestCase):
    def setUp(self):
        CALLS.clear()
        INTERRUPT["after"] = None

    def test_random_improves_and_tests_once(self):
        with tempfile.TemporaryDirectory() as d:
            loop = make(d, proposer=RandomProposer(seed=7), stop_after=4, max_turns=15,
                        baselines={"low_x": {"x": 0.1}, "mid": {"x": 0.5, "k": 10}})
            out = loop.run()
            hist = out["history"]
            self.assertEqual(hist[0]["status"], "keep")
            for t in hist[1:]:
                if t["status"] == "keep":  # acceptance only with the CI below zero (lower is better)
                    self.assertLess(t["diff_inc"]["hi"], 0)
                self.assertEqual(len(config_changes(t["cfg"], hist[t["inc_before"]]["cfg"])), 1)
            self.assertGreaterEqual(out["kept"], 1)
            self.assertLess(out["best_dev"]["value"], hist[0]["metrics"]["err"]["value"])
            test_calls = [c for c in CALLS if c["split"] == "test"]
            # winner + start + two baselines, each exactly once
            self.assertEqual(len(test_calls), 4)
            fin = out["final"]
            self.assertIn("winner-start", fin["test_diff"])
            self.assertTrue(fin["verdict"]["beats_start_on_test"])
            # finalize again: no new test runs
            loop.finalize()
            self.assertEqual(len([c for c in CALLS if c["split"] == "test"]), 4)
            self.assertTrue((Path(d) / "JOURNAL.md").exists())
            j = (Path(d) / "JOURNAL.md").read_text()
            self.assertIn("## Turn 1", j)
            self.assertIn("Final (test split evaluated once)", j)
            tsv = pd.read_csv(Path(d) / "results.tsv", sep="\t")
            self.assertEqual(len(tsv), 2 + len(hist))

    def test_test_split_guarded(self):
        loop = make(None, proposer="random")
        with self.assertRaises(PermissionError):
            loop.evaluate(loop.start, "test")

    def test_crash_is_recorded_not_fatal(self):
        sp = space()
        loop = Loop(objective, sp, proposer=GridProposer(n_grid=3), primary="err", higher_is_better=False,
                    stop_after=3, max_turns=6)
        # force the first proposal to be method c (crash): method first in the sweep, c first among its values
        sp.params = {"method": ["c", "b", "a"], "x": sp.params["x"], "k": sp.params["k"]}
        out = loop.run(finalize=False)
        st = [t["status"] for t in out["history"]]
        self.assertIn("crash", st)
        crash = [t for t in out["history"] if t["status"] == "crash"][0]
        self.assertIn("method c is broken", crash["error"])


def config_changes(a, b):
    return {k for k in a if a[k] != b.get(k)}


class TestStopRule(unittest.TestCase):
    def test_stops_after_k_not_kept(self):
        loop = make(None, runner=flat, proposer=RandomProposer(seed=3), stop_after=3, max_turns=12)
        out = loop.run(finalize=False)
        self.assertEqual(out["turns"], 4)  # turn 0 + 3 discarded
        self.assertTrue(all(t["status"] != "keep" for t in out["history"][1:]))
        self.assertIn("3 consecutive turns not kept", out["stopped"])

    def test_turn_limit(self):
        loop = make(None, runner=flat, proposer=RandomProposer(seed=3), stop_after=None, max_turns=5)
        out = loop.run(finalize=False)
        self.assertEqual(out["turns"], 6)
        self.assertIn("turn limit 5", out["stopped"])

    def test_flat_objective_final_verdict(self):
        loop = make(None, runner=flat, proposer=RandomProposer(seed=3), stop_after=2)
        out = loop.run()
        self.assertEqual(out["final"]["winner_turn"], 0)
        self.assertIn("no turn beat", out["final"]["verdict"]["note"])

    def test_budget_stop(self):
        def responder(system, user):
            return json.dumps({"proposal": {"change": {"x": round(0.21 + 0.01 * user.count("round "), 2)},
                                            "kind": "improve", "hypothesis": "h", "expected": "lower",
                                            "falsifier": "f"}})

        class Paid(FakeLLM):
            def complete(self, system, user, **kw):
                r = super().complete(system, user, **kw)
                r.cost_usd = 0.4
                return r

        loop = make(None, runner=flat, proposer=Paid(responder), stop_after=None, max_turns=20, budget_usd=1.0)
        out = loop.run(finalize=False)
        self.assertIn("budget", out["stopped"])
        self.assertEqual(out["turns"], 4)  # 3 paid proposals reach $1.2 >= $1.0


class TestLLMProposer(unittest.TestCase):
    def setUp(self):
        CALLS.clear()
        INTERRUPT["after"] = None

    def test_fake_llm_with_repair(self):
        plan = iter([
            "not json at all",                                                       # invalid -> repair
            {"change": {"method": "b"}, "kind": "new", "hypothesis": "b is better", "expected": "lower",
             "falsifier": "CI not below 0"},
            {"change": {"x": 0.7}, "kind": "improve", "hypothesis": "x near optimum", "expected": "lower",
             "falsifier": "no drop"},
            {"change": {"x": 0.7, "k": 12}, "kind": "improve", "hypothesis": "two keys", "expected": "lower",
             "falsifier": "-"},                                                      # invalid -> repair
            {"change": {"k": 12}, "kind": "improve", "hypothesis": "k near optimum", "expected": "lower",
             "falsifier": "no drop"},
            {"oneri": {"degisiklik": {"k": 13}, "tur": "iyilestir", "hipotez": "fine", "beklenen_yon": "same",
                       "ne_cikarsa_yanlis": "drop"}},                                 # Turkish keys accepted
            {"change": {"x": 0.69}, "kind": "improve", "hypothesis": "tiny", "expected": "same", "falsifier": "-"},
            {"change": {"x": 0.71}, "kind": "improve", "hypothesis": "tiny", "expected": "same", "falsifier": "-"},
            {"change": {"x": 0.72}, "kind": "improve", "hypothesis": "tiny", "expected": "same", "falsifier": "-"},
        ])
        prompts = []

        def responder(system, user):
            prompts.append(user)
            x = next(plan)
            if isinstance(x, str):
                return x
            if "oneri" in x:
                return "Here you go: " + json.dumps(x)
            return json.dumps({"what_changed": "w", "surprising": "none", "leakage_suspicion": "none", "proposal": x})

        with tempfile.TemporaryDirectory() as d:
            loop = make(d, proposer=FakeLLM(responder), stop_after=3, max_turns=10, task="Synthetic task.")
            out = loop.run()
            c = out["counters"]
            self.assertEqual(c["invalid"], 2)
            self.assertEqual(c["repairs_ok"], 2)
            changes = [t["proposal"]["change"] for t in out["history"][1:]]
            self.assertEqual(changes[0], {"method": "b"})
            self.assertEqual(changes[1], {"x": 0.7})
            self.assertEqual(changes[2], {"k": 12})
            self.assertEqual(changes[3], {"k": 13})
            self.assertEqual(out["best_cfg"], {"x": 0.7, "k": 12, "method": "b"})
            self.assertIsNotNone(out["hit_rate"])
            # the repair prompt carries the rejection message; prompts never contain test numbers
            self.assertTrue(any("previous proposal was rejected" in p for p in prompts))
            self.assertFalse(any("test" in p.lower() for p in prompts))
            self.assertTrue((Path(d) / "observer" / "turn1_prompt.txt").exists())
            self.assertIn("observer: what changed", (Path(d) / "JOURNAL.md").read_text())

    def test_two_invalid_stop_and_test_reference(self):
        replies = iter(['{"proposal": {"change": {"x": 0.5}, "kind": "improve", "hypothesis": "the test set says so",'
                        ' "expected": "lower", "falsifier": "-"}}',
                        '{"proposal": {"change": {"zzz": 1}, "kind": "improve", "hypothesis": "h",'
                        ' "expected": "lower", "falsifier": "-"}}'])
        loop = make(None, proposer=FakeLLM(lambda s, u: next(replies)))
        out = loop.run(finalize=False)
        self.assertIn("proposer invalid twice", out["stopped"])
        self.assertEqual(out["counters"]["test_refs"], 1)
        self.assertEqual(out["turns"], 1)

    def test_prompt_contents(self):
        loop = make(None, proposer="random", stop_after=1, max_turns=1, task="TASK-TEXT")
        loop.run(finalize=False)
        p = build_prompt(loop._context())
        for s in ("TASK-TEXT", "CURRENT BEST", "round 0", "Allowed space", "exactly one key", "lower is better"):
            self.assertIn(s, p)

    def test_parse_reply(self):
        self.assertIsNone(parse_observer_reply("nothing"))
        d = parse_observer_reply('xx {"oneri": {"degisiklik": {"a": 1}, "tur": "onar"}} yy')
        self.assertEqual(d["proposal"]["change"], {"a": 1})
        self.assertEqual(d["proposal"]["kind"], "fix")

    def test_claude_p_fallback_offline(self):
        seen = {}

        class R:
            stdout = json.dumps({"result": '{"proposal": {}}', "is_error": False,
                                 "usage": {"input_tokens": 10, "output_tokens": 3}, "total_cost_usd": 0.001})
            stderr = ""

        def fake_run(cmd, **kw):
            seen["cmd"], seen["input"] = cmd, kw["input"]
            return R()

        llm = _ClaudePCLI(model="claude-haiku-5-5", binary="claude", run=fake_run)
        r = llm.complete("SYS", "USER")
        self.assertEqual(r.text, '{"proposal": {}}')
        self.assertEqual(r.cost_usd, 0.0)
        self.assertIn("--no-session-persistence", seen["cmd"])
        self.assertEqual(seen["cmd"][seen["cmd"].index("--tools") + 1], "")
        self.assertTrue(seen["input"].startswith("SYS"))


class TestResume(unittest.TestCase):
    def setUp(self):
        CALLS.clear()
        INTERRUPT["after"] = None

    def tearDown(self):
        INTERRUPT["after"] = None

    def test_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ref = make(a, proposer=RandomProposer(seed=11), stop_after=3, max_turns=8).run()
            n_ref_calls = len(CALLS)
            CALLS.clear()
            INTERRUPT["after"] = 4
            with self.assertRaises(KeyboardInterrupt):
                make(b, proposer=RandomProposer(seed=11), stop_after=3, max_turns=8).run()
            done = len(CALLS) - 1  # the interrupted call did not finish
            INTERRUPT["after"] = None
            CALLS.clear()
            res = make(b, proposer=RandomProposer(seed=11), stop_after=3, max_turns=8).run()
            self.assertEqual(done + len(CALLS), n_ref_calls)  # nothing finished was re-run
            self.assertEqual([t["cfg"] for t in res["history"]], [t["cfg"] for t in ref["history"]])
            self.assertEqual([t["status"] for t in res["history"]], [t["status"] for t in ref["history"]])
            self.assertEqual(res["final"]["winner_cfg"], ref["final"]["winner_cfg"])
            self.assertAlmostEqual(res["final"]["test"]["winner"]["metrics"]["err"]["value"],
                                   ref["final"]["test"]["winner"]["metrics"]["err"]["value"])

    def test_runner_change_refused(self):
        with tempfile.TemporaryDirectory() as d:
            make(d, proposer="random", stop_after=1, max_turns=1).run(finalize=False)

            def other(cfg):
                return objective(cfg)

            with self.assertRaises(RuntimeError):
                make(d, runner=other, proposer="random")


class TestOtherProposers(unittest.TestCase):
    def setUp(self):
        CALLS.clear()
        INTERRUPT["after"] = None

    def _run(self, proposer, **kw):
        sp = Space({"x": (0.0, 1.0), "k": (1, 24), "method": ["a", "b"]}, defaults={"x": 0.2, "k": 4, "method": "a"})
        loop = Loop(objective, sp, proposer, primary="err", higher_is_better=False, stop_after=3, max_turns=12, **kw)
        return loop, loop.run(finalize=False)

    def test_grid(self):
        _, out = self._run(GridProposer(n_grid=5))
        self.assertGreaterEqual(out["kept"], 1)
        self.assertLess(out["best_dev"]["value"], out["history"][0]["metrics"]["err"]["value"])

    def test_greedy_momentum(self):
        _, out = self._run(GreedyCoordinateProposer(n_steps=8))
        self.assertGreaterEqual(out["kept"], 2)
        reasons = [t["proposal"]["rationale"] for t in out["history"][1:]]
        self.assertTrue(any(r.startswith("momentum") for r in reasons))

    def test_halving(self):
        loop, out = self._run(HalvingProposer(n_candidates=9, fidelities=(0.25, 0.5), eta=3))
        self.assertGreaterEqual(out["kept"], 1)
        self.assertGreater(out["counters"].get("screen_runs", 0), 0)
        fid = [c for c in CALLS if "fidelity" in c]
        self.assertTrue(fid and all(c["split"] == "dev" for c in fid))

    def test_successive_halving_function(self):
        cands = [{"v": v} for v in [5, 1, 3, 4, 2, 9, 7, 8, 6]]
        res = successive_halving(lambda cs, f: [c["v"] + (1 - f) * 0.1 for c in cs], cands, higher_is_better=False)
        self.assertEqual(res["winner"], {"v": 1})
        self.assertEqual(res["evaluations"], 9 + 3 + 1)


class TestSelectionAblationControl(unittest.TestCase):
    def setUp(self):
        CALLS.clear()
        INTERRUPT["after"] = None

    def test_selection_and_ablation(self):
        sp = Space({"x": (0.0, 1.0), "k": (1, 24), "method": ["a", "b"]}, defaults={"x": 0.2, "k": 4, "method": "a"})
        loop = Loop(objective, sp, GridProposer(n_grid=5), primary="err", higher_is_better=False,
                    select_split="select", stop_after=3, max_turns=10, group_col="grp")
        out = loop.run()
        fin = out["final"]
        self.assertTrue(1 <= len(fin["selection"]) <= 3)
        self.assertTrue(all(s["select"]["n"] == N_UNITS["select"] for s in fin["selection"]))
        if fin["winner_turn"] != 0:
            self.assertTrue(fin["ablation"])
            self.assertTrue(all(a["split"] == "select" for a in fin["ablation"]))
        self.assertIn("groups", out["history"][-1])

    def test_control_arm(self):
        with tempfile.TemporaryDirectory() as d:
            main = make(str(Path(d) / "main"), proposer=GridProposer(n_grid=5), stop_after=3, max_turns=6)
            main.run(finalize=False)
            ctrl = control_arm(main, RandomProposer(seed=5), workdir=str(Path(d) / "ctrl"))
            cres = ctrl.run(finalize=False)
            self.assertEqual(cres["turns"], len(main.state["turns"]))
            fin = main.finalize(extra_test={"random_arm": cres["best_cfg"]})
            self.assertIn("winner-random_arm", fin["final"]["test_diff"])


if __name__ == "__main__":
    unittest.main()
