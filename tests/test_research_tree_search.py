"""AIDE-style tree search and MCTS-UCT on synthetic objectives (offline, FakeLLM, no data rows)."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from simengine.llm import FakeLLM
from simengine.research.space import Space
from simengine.research.tree_search import (ConfigOperators, LLMOperators, MCTSSearch, Node, TreeSearch, best_node,
                                            parse_code_reply, parse_config_reply, summarize_tree)

N = {"dev": 200, "test": 150}
CALLS = []


def _u(text, n):
    s = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(s).normal(size=n)


def objective(cfg):
    """Lower is better; optimum x=0.7, k=12, method 'b'; method 'c' crashes. Shared per-unit difficulty."""
    CALLS.append(dict(cfg))
    split = cfg["split"]
    if cfg.get("method") == "c":
        raise RuntimeError("method c is broken")
    n = N[split]
    eff = 2.0 * (cfg["x"] - 0.7) ** 2 + 0.004 * (cfg["k"] - 12) ** 2 + {"a": 0.15, "b": 0.0}[cfg["method"]]
    key = json.dumps({k: v for k, v in sorted(cfg.items()) if k != "split"})
    err = 1.0 + _u(f"u-{split}", n) + eff + 0.02 * _u(key + split, n)
    return {"per_unit": pd.DataFrame({"unit": [f"{split}{i}" for i in range(n)], "err": err}),
            "metrics": {"effect": eff}}


def space():
    return Space({"x": (0.0, 1.0), "k": (1, 24), "method": ["a", "b", "c"]},
                 defaults={"x": 0.2, "k": 4, "method": "a"})


def eff(cfg):
    return 2.0 * (cfg["x"] - 0.7) ** 2 + 0.004 * (cfg["k"] - 12) ** 2 + {"a": 0.15, "b": 0.0}[cfg["method"]]


class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.ts = TreeSearch(objective, ConfigOperators(space()), primary="err", higher_is_better=False,
                             num_drafts=2, debug_prob=1.0, max_debug_depth=2)

    def test_drafts_then_debug_then_improve(self):
        rng = np.random.default_rng(0)
        self.assertEqual(self.ts.policy([], rng), ("draft", None))
        a = Node(0, None, "draft", {"x": 1}, status="ok", score=2.0)
        b = Node(1, None, "draft", {"x": 2}, status="buggy", error="boom")
        op, p = self.ts.policy([a, b], rng)
        self.assertEqual((op, p.id), ("debug", 1))
        b.children = [2]                         # not a leaf any more
        c = Node(2, 1, "debug", {"x": 3}, status="buggy", debug_depth=2)   # depth limit reached
        op, p = self.ts.policy([a, b, c], rng)
        self.assertEqual((op, p.id), ("improve", 0))

    def test_draft_when_no_good_node(self):
        nodes = [Node(i, None, "draft", None, status="buggy", debug_depth=0, children=[9]) for i in range(2)]
        self.assertEqual(self.ts.policy(nodes, np.random.default_rng(0)), ("draft", None))

    def test_best_node_greedy_and_ties(self):
        ns = [Node(0, None, "draft", {}, status="ok", score=1.0), Node(1, None, "draft", {}, status="ok", score=0.5),
              Node(2, None, "draft", {}, status="ok", score=0.5), Node(3, None, "draft", {}, status="buggy")]
        self.assertEqual(best_node(ns, higher_is_better=False).id, 1)
        self.assertEqual(best_node(ns, higher_is_better=True).id, 0)
        self.assertIsNone(best_node(ns[3:], True))


class TestTreeSearchConfig(unittest.TestCase):
    def test_search_improves_and_test_once(self):
        CALLS.clear()
        with tempfile.TemporaryDirectory() as d:
            ts = TreeSearch(objective, ConfigOperators(space()), primary="err", higher_is_better=False, steps=30,
                            workdir=d, seed=1)
            out = ts.run()
            self.assertEqual(out["n_nodes"], 30)
            self.assertTrue(out["stopped"].startswith("step limit"))
            self.assertFalse(any(c["split"] == "test" for c in CALLS))
            nodes = ts.nodes
            self.assertEqual(nodes[0].payload, {"x": 0.2, "k": 4, "method": "a"})   # default first
            self.assertEqual(sum(n.op == "draft" for n in nodes[:5]), 5)
            good = [n for n in nodes if n.status == "ok"]
            self.assertEqual(out["best_dev"]["value"], min(n.score for n in good))
            self.assertLess(eff(out["best_payload"]), eff(nodes[0].payload) - 0.1)
            for n in nodes:                       # improve = exactly one key changed from the parent
                if n.op == "improve":
                    self.assertEqual(len([k for k in n.payload if n.payload[k] != nodes[n.parent].payload[k]]), 1)
                    self.assertIn(n.id, nodes[n.parent].children)
            with self.assertRaises(PermissionError):
                ts.evaluate(nodes[0].payload, "test")
            fin = ts.finalize()["final"]
            n_test = sum(c["split"] == "test" for c in CALLS)
            self.assertEqual(n_test, 2)           # winner + reference (first good node)
            self.assertLess(fin["test"]["winner"]["score"], fin["test"]["reference"]["score"])
            self.assertLess(fin["test_diff"]["winner-reference"]["ci95"]["hi"], 0)
            self.assertIsNotNone(fin["winner_test_minus_dev"])
            ts.finalize()
            self.assertEqual(sum(c["split"] == "test" for c in CALLS), n_test)   # idempotent
            for f in ("state.json", "JOURNAL.md", "results.tsv"):
                self.assertTrue((Path(d) / f).exists())
            j = (Path(d) / "JOURNAL.md").read_text()
            self.assertIn("*best*", j)
            self.assertIn("test split evaluated once", j)

    def test_buggy_nodes_and_debug_depth(self):
        sp = Space({"x": (0.0, 1.0), "k": (1, 24), "method": ["c", "a", "b"]},
                   defaults={"x": 0.2, "k": 4, "method": "c"})     # the default crashes
        ts = TreeSearch(objective, ConfigOperators(sp), primary="err", higher_is_better=False, steps=25,
                        num_drafts=3, debug_prob=0.9, max_debug_depth=2, seed=3)
        ts.run()
        nodes = ts.nodes
        self.assertTrue(nodes[0].buggy)
        self.assertIn("method c is broken", nodes[0].error)
        dbg = [n for n in nodes if n.op == "debug"]
        self.assertTrue(dbg)
        for n in dbg:
            par = nodes[n.parent]
            self.assertTrue(par.buggy)
            self.assertEqual(n.debug_depth, par.debug_depth + 1)
            self.assertLessEqual(n.debug_depth, 2)
            self.assertEqual(len([c for c in par.children if nodes[c].op == "debug"]), 1)  # leaf rule
        # the error names `method`, so a config debug changes that key
        first = next(n for n in dbg if n.parent == 0)
        self.assertNotEqual(first.payload["method"], "c")
        self.assertIsNotNone(ts.best())

    def test_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            kw = dict(primary="err", higher_is_better=False, seed=5)
            TreeSearch(objective, ConfigOperators(space()), steps=8, workdir=d1, **kw).run()
            a = TreeSearch(objective, ConfigOperators(space()), steps=15, workdir=d1, **kw)
            self.assertEqual(len(a.nodes), 8)
            a.run()
            b = TreeSearch(objective, ConfigOperators(space()), steps=15, workdir=d2, **kw)
            b.run()
            self.assertEqual([n.payload for n in a.nodes], [n.payload for n in b.nodes])
            self.assertEqual([n.op for n in a.nodes], [n.op for n in b.nodes])

    def test_resume_refuses_other_direction(self):
        with tempfile.TemporaryDirectory() as d:
            TreeSearch(objective, ConfigOperators(space()), primary="err", higher_is_better=False, steps=3,
                       workdir=d).run()
            with self.assertRaises(RuntimeError):
                TreeSearch(objective, ConfigOperators(space()), primary="err", higher_is_better=True, workdir=d)

    def test_exhausted_operators_stop(self):
        sp = Space({"m": ["a", "b"]}, defaults={"m": "a"})
        ts = TreeSearch(lambda c: {"per_unit": {"u1": 1.0 if c["m"] == "a" else 0.5, "u2": 1.0}},
                        ConfigOperators(sp, tries=20), steps=10, num_drafts=1)
        out = ts.run()
        self.assertEqual(out["n_nodes"], 2)
        self.assertIn("nothing left", out["stopped"])


class TestRunnerContract(unittest.TestCase):
    def test_scalar_score_nan_and_unit_count(self):
        sp = Space({"m": ["ok", "nan", "short"]}, defaults={"m": "ok"})

        def run(c):
            if c["m"] == "ok":
                return {"per_unit": {f"u{i}": float(i) for i in range(5)}}
            if c["m"] == "nan":
                return {"score": float("nan")}
            return {"per_unit": {"u0": 1.0}}

        ts = TreeSearch(run, ConfigOperators(sp), steps=6, num_drafts=3, debug_prob=0.0)
        ts.run()
        by = {n.payload["m"]: n for n in ts.nodes if n.payload}
        self.assertEqual(by["ok"].status, "ok")
        self.assertEqual(by["ok"].score, 2.0)
        self.assertIn("no per_unit and no finite", by["nan"].error)
        self.assertIn("unit count", by["short"].error)

        ts2 = TreeSearch(lambda c: {"score": 0.8}, ConfigOperators(sp), steps=1)
        ts2.run()
        self.assertEqual((ts2.nodes[0].score, ts2.nodes[0].score_source), (0.8, "runner"))


class TestLLMOperatorsConfig(unittest.TestCase):
    def test_fake_llm_tree(self):
        replies = iter([
            '{"plan": "defaults", "config": {"x": 0.2, "k": 4, "method": "a"}}',
            'not json at all',                                                     # buggy: unparsable
            '{"plan": "try b", "config": {"x": 0.5, "k": 10, "method": "b"}}',
            '{"plan": "two keys", "config": {"x": 0.7, "k": 12, "method": "b"}}',  # improve: refused (2 keys)
            '{"plan": "check the test split", "config": {"x": 0.7, "k": 10, "method": "b"}}',  # refused
            '{"plan": "fixed", "config": {"x": 0.6, "k": 10, "method": "b"}}',
        ] + ['{"plan": "x up", "config": {"x": 0.65, "k": 10, "method": "b"}}'] * 10)
        llm = FakeLLM(lambda s, u: next(replies))
        ops = LLMOperators(llm, space=space(), task="Toy regression: lower err is better.")
        with tempfile.TemporaryDirectory() as d:
            ts = TreeSearch(objective, ops, primary="err", higher_is_better=False, steps=7, num_drafts=3,
                            debug_prob=0.0, workdir=d)
            ts.run()
            nodes = ts.nodes
            self.assertEqual([n.op for n in nodes[:3]], ["draft"] * 3)
            self.assertIn("JSON", nodes[1].error)
            self.assertEqual(nodes[3].op, "improve")
            self.assertIn("exactly one key", nodes[3].error)
            self.assertIn("test split", nodes[4].error)
            self.assertEqual(nodes[5].payload, {"x": 0.6, "k": 10, "method": "b"})
            self.assertEqual(nodes[5].parent, 2)
            self.assertIsNotNone(nodes[5].diff_parent)
            self.assertTrue((Path(d) / "operator" / "node5_improve_prompt.txt").exists())
        # prompts carry the tree memory and the space
        last = llm.calls[-1][1]
        self.assertIn("Search memory:", last)
        self.assertIn("Recent failures:", last)
        self.assertIn("IMPROVE node", last)
        self.assertIn('"method": ["a", "b", "c"]', last)

    def test_debug_prompt_has_error(self):
        llm = FakeLLM(lambda s, u: '{"plan": "p", "config": {"x": 0.3, "k": 4, "method": "c"}}' if "DRAFT" in u
                      else '{"plan": "fix", "config": {"x": 0.3, "k": 4, "method": "a"}}')
        ts = TreeSearch(objective, LLMOperators(llm, space=space()), primary="err", higher_is_better=False,
                        steps=2, num_drafts=1, debug_prob=1.0)
        ts.run()
        self.assertEqual([n.op for n in ts.nodes], ["draft", "debug"])
        self.assertIn("method c is broken", llm.calls[1][1])
        self.assertEqual(ts.nodes[1].status, "ok")


# toy code task: fit y = 3x + 1 on synthetic data; the runner execs the snippet (tests only, trusted text)
_X = np.linspace(-1, 1, 60)
_Y = 3 * _X + 1 + 0.1 * _u("noise", 60)


def code_runner(cfg):
    ns = {"np": np}
    exec(cfg["code"], ns)                        # noqa: S102 - FakeLLM text in a unit test
    idx = np.arange(60) % 2 == (0 if cfg["split"] == "dev" else 1)
    pred = np.asarray(ns["predict"](_X[idx]), float)
    return {"per_unit": pd.DataFrame({"unit": np.flatnonzero(idx), "se": (pred - _Y[idx]) ** 2})}


CODE = {
    "d0": "Plan: predict the mean.\n```python\ndef predict(x):\n    return np.full(len(x), 1.0)\n```",
    "d1": "Plan: linear fit.\n```python\ndef predict(x):\n    return slope * x + 1\n```",   # NameError
    "fix": "Plan: define the slope.\n```python\nslope = 2.0\ndef predict(x):\n    return slope * x + 1\n```",
    "imp": "Plan: slope 3.\n```python\nslope = 3.0\ndef predict(x):\n    return slope * x + 1\n```",
}


def code_llm(system, user):
    if "DEBUG" in user:
        return CODE["fix"]
    if "IMPROVE" in user:
        return CODE["imp"]
    return CODE["d0"] if "0 nodes" in user else CODE["d1"]


class TestLLMOperatorsCode(unittest.TestCase):
    def test_code_tree(self):
        llm = FakeLLM(code_llm)
        ts = TreeSearch(code_runner, LLMOperators(llm, mode="code", task="Fit y from x."), primary="se",
                        higher_is_better=False, steps=6, num_drafts=2, debug_prob=1.0)
        out = ts.run()
        ops = [(n.op, n.status) for n in ts.nodes]
        self.assertEqual(ops[:3], [("draft", "ok"), ("draft", "buggy"), ("debug", "ok")])
        self.assertIn("NameError", ts.nodes[1].error)
        self.assertIn("slope = 3.0", out["best_payload"]["code"])
        self.assertEqual(ts.nodes[3].repeat_of, None)
        self.assertTrue(any(n.repeat_of is not None for n in ts.nodes[4:]))   # same code again = repeat, no rerun
        fin = ts.finalize()["final"]
        self.assertLess(fin["test"]["winner"]["score"], 0.05)

    def test_parsers(self):
        code, plan, err = parse_code_reply("idea\n```python\na = 1\n```\nmore\n```py\nb = 2\n```")
        self.assertEqual((code, plan, err), ("b = 2", "idea", None))
        self.assertIsNotNone(parse_code_reply("no code")[2])
        cfg, plan, err = parse_config_reply('x {"plan": "p", "config": {"k": 3}} y')
        self.assertEqual((cfg, plan, err), ({"k": 3}, "p", None))
        self.assertIn("no 'config'", parse_config_reply('{"plan": "p"}')[2])
        with self.assertRaises(ValueError):
            LLMOperators(FakeLLM(), mode="config")


class TestSummary(unittest.TestCase):
    def test_summary(self):
        ns = [Node(0, None, "draft", {"k": 1}, plan="base", status="ok", score=0.9),
              Node(1, 0, "improve", {"k": 2}, plan="k up", status="ok", score=0.7),
              Node(2, None, "draft", {"code": "a\nb"}, status="buggy", error="ValueError: shape")]
        s = summarize_tree(ns, "err", False)
        self.assertIn("3 nodes (2 ok, 1 buggy)", s)
        self.assertIn("best node 1 err=0.7000", s)
        self.assertLess(s.index("node 1 (improve of 0)"), s.index("node 0 (draft)"))
        self.assertIn("ValueError: shape", s)
        self.assertIn("<code, 2 lines>", s)


class TestMCTS(unittest.TestCase):
    def sp(self):
        return Space({"a": [0, 1, 2, 3, 4], "b": [0, 1, 2, 3, 4], "c": ["p", "q"]},
                     defaults={"a": 0, "b": 0, "c": "p"}, check=lambda c: ["bad corner"] if c["a"] == c["b"] == 4 else [])

    @staticmethod
    def run_ab(calls):
        def r(cfg):
            calls.append(dict(cfg))
            n = 50
            v = (cfg["a"] - 3) ** 2 + 0.5 * (cfg["b"] - 1) ** 2 + (0.0 if cfg["c"] == "q" else 1.0)
            key = json.dumps({k: cfg[k] for k in "abc"})
            return {"per_unit": {f"u{i}": x for i, x in enumerate(v + _u("m-units", n) + 0.01 * _u(key, n))}}
        return r

    def test_finds_optimum_and_concentrates(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            m = MCTSSearch(self.run_ab(calls), self.sp(), primary="v", higher_is_better=False, budget=30, seed=2,
                           workdir=d)
            out = m.run()
            self.assertEqual(out["runner_calls"], len(calls))
            self.assertLessEqual(len(calls), 30)
            hashes = [json.dumps(c, sort_keys=True) for c in calls]
            self.assertEqual(len(hashes), len(set(hashes)))          # no config scored twice
            self.assertEqual({k: out["best_payload"][k] for k in "abc"}, {"a": 3, "b": 1, "c": "q"})
            tree = m.state["tree"]
            visits = [tree.get(str(i), [0])[0] for i in range(5)]
            self.assertEqual(int(np.argmax(visits)), 3)              # a=3 subtree visited most
            fin = m.finalize()["final"]
            self.assertEqual(sum(c["split"] == "test" for c in calls), 2)
            self.assertLess(fin["test_diff"]["winner-reference"]["ci95"]["hi"], 0)
            self.assertIn("MCTS-UCT", (Path(d) / "JOURNAL.md").read_text())

    def test_exhausts_small_space_and_invalid(self):
        calls = []
        sp = Space({"a": [3, 4], "b": [4, 1]}, defaults={"a": 3, "b": 4},
                   check=lambda c: ["bad corner"] if c["a"] == c["b"] == 4 else [])

        def r(cfg):
            calls.append(cfg)
            return {"per_unit": {"u0": float(cfg["a"] + cfg["b"]), "u1": 1.0}}

        m = MCTSSearch(r, sp, primary="v", higher_is_better=False, budget=100, seed=0)
        out = m.run()
        self.assertEqual(out["stopped"], "search space exhausted")
        self.assertEqual(len(calls), 3)                              # the invalid corner never reaches the runner
        bad = [n for n in m.nodes if n.buggy]
        self.assertTrue(bad and all("bad corner" in n.error for n in bad))
        self.assertEqual(out["best_payload"], {"a": 3, "b": 1})

    def test_resume(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            kw = dict(primary="v", higher_is_better=False, seed=4)
            MCTSSearch(self.run_ab([]), self.sp(), budget=6, workdir=d1, **kw).run()
            a = MCTSSearch(self.run_ab([]), self.sp(), budget=12, workdir=d1, **kw)
            a.run()
            b = MCTSSearch(self.run_ab([]), self.sp(), budget=12, workdir=d2, **kw)
            b.run()
            self.assertEqual([n.payload for n in a.nodes], [n.payload for n in b.nodes])

    def test_bad_keys(self):
        with self.assertRaises(ValueError):
            MCTSSearch(self.run_ab([]), self.sp(), keys=["zz"])


if __name__ == "__main__":
    unittest.main()
