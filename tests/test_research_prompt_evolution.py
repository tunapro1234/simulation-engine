"""research.prompt_evolution: OPRO / APE / Promptbreeder search, validation selection, sealed single test pass.

Offline: FakeLLM optimizers and scorers on synthetic units (no data rows, no network)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import simengine.research.prompt_evolution as pe
from simengine.llm.fake import FakeLLM
from simengine.research.prereg import Prereg, lock_prereg, verify_seal


def _toy(seed=0, n=300):
    units, score, keys, resp = pe.toy_problem(n_units=n, seed=seed)
    tr, va, te = pe.split_units(units, seed=seed)
    return pe.PromptEvaluator(score, tr, va, te), FakeLLM(resp), keys


def _const(text):
    return FakeLLM(lambda s, u: text)


class TestSplitAndScorer(unittest.TestCase):
    def test_split_sizes_disjoint(self):
        df = pd.DataFrame({"u": range(100)})
        tr, va, te = pe.split_units(df, frac=(0.6, 0.2, 0.2), seed=1)
        self.assertEqual((len(tr), len(va), len(te)), (60, 20, 20))
        self.assertEqual(len(set(tr.u) | set(va.u) | set(te.u)), 100)

    def test_split_group_aware_and_list(self):
        df = pd.DataFrame({"g": np.repeat(np.arange(20), 3), "x": range(60)})
        parts = pe.split_units(df, seed=0, group="g")
        groups = [set(p.g) for p in parts]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        tr, va, te = pe.split_units(list("abcdefgh"), frac=(2, 1, 1))
        self.assertEqual(sorted(tr + va + te), list("abcdefgh"))

    def test_llm_scorer_exact_match_and_position(self):
        units = [{"input": "2+2", "output": "4"}, {"input": "3+3", "output": "6"}]
        llm = FakeLLM(lambda s, u: "4." if "2+2" in u else "five")
        r = pe.make_llm_scorer(llm)("Add the numbers.", units)
        self.assertEqual(r["scores"].tolist(), [1.0, 0.0])
        self.assertEqual(llm.calls[0], ("Add the numbers.", "2+2"))
        pe.make_llm_scorer(llm, position="q_begin")("Add.", units[:1])
        self.assertEqual(llm.calls[-1], ("", "Add.\n2+2"))
        with self.assertRaises(ValueError):
            pe.make_llm_scorer(llm, position="middle")

    def test_persona_text_scorer(self):
        # persona text in the system prompt, numeric survey answers, score = minus absolute error
        items = pd.DataFrame({"input": ["q1", "q2"], "output": [70.0, 20.0]})
        llm = FakeLLM(lambda s, u: "60" if "outdoorsy" in s else "10")
        score = pe.make_llm_scorer(llm, parse=float, metric=lambda y, t: -abs(y - float(t)))
        ev = pe.PromptEvaluator(score, items, items, higher_is_better=True)
        a, b = ev.scores("An outdoorsy person.")["mean"], ev.scores("A homebody.")["mean"]
        self.assertGreater(a, b)
        odd = pe.make_llm_scorer(FakeLLM(lambda s, u: "no idea"), parse=float)("x", items)
        self.assertTrue(np.isnan(odd["scores"]).all())

    def test_evaluator_cache_subset_and_test_guard(self):
        n = {"calls": 0}

        def score(text, units):
            n["calls"] += 1
            return np.arange(len(units), dtype=float)
        ev = pe.PromptEvaluator(score, list(range(10)), list(range(4)), list(range(4)))
        ev.scores("x")
        ev.scores("x")
        sub = ev.scores("x", idx=[2, 5])
        self.assertEqual(sub["scores"].tolist(), [2.0, 5.0])
        self.assertEqual(n["calls"], 1)
        self.assertEqual(ev.unit_evals["train"], 10)
        with self.assertRaises(PermissionError):
            ev.scores("x", "test")
        bad = pe.PromptEvaluator(lambda t, u: [1.0], [1, 2], [1])
        with self.assertRaises(ValueError):
            bad.scores("x")


class TestParsing(unittest.TestCase):
    def test_square_brackets_and_clean(self):
        self.assertEqual(pe.parse_square_brackets("Sure. [Think it through.] and [x]"), "Think it through.")
        self.assertEqual(pe.parse_square_brackets("no brackets"), "")
        self.assertEqual(pe.clean_reply('INSTRUCTION MUTANT: "Solve it carefully."\n\nmore'), "Solve it carefully.")
        self.assertEqual(pe.first_line("\n3. Check units.\n4. Other"), "Check units.")
        self.assertEqual(pe.clean_reply("The instruction was: add them"), "add them")


class TestOPRO(unittest.TestCase):
    def test_meta_prompt_layout(self):
        mk = lambda t, s: pe.Candidate(t, s, s, 10, True, "opro", "x")
        hist = [mk("b", 0.5), mk("a", 0.9), mk("c", 0.1), mk("b", 0.6)]
        p = pe.opro_meta_prompt(hist, [("Q: 1+1", "2")], keep_top=2)
        self.assertTrue(p.startswith("I have some texts along with their corresponding scores."))
        self.assertIn("higher scores indicate better quality", p)
        self.assertNotIn("text:\nc", p)                                  # only the best 2 kept
        self.assertLess(p.index("text:\nb\nscore:\n60"), p.index("text:\na\nscore:\n90"))   # ascending, best last
        self.assertIn("input:\n<INS>\nQ: 1+1\noutput:\n2", p)
        self.assertTrue(p.endswith("has a score as high as possible. Write the text in square brackets."))
        q = pe.opro_meta_prompt(hist, [], show_scores=False, higher_is_better=False, score_scale=None)
        self.assertNotIn("score:", q.split("Write your new text")[0].replace("scores", ""))
        # lower is better (OPRO Fig. 19-20): worst first, so raw scores descend, and the tail asks for a low score
        lo = [pe.Candidate(t, s, -s, 10, True, "opro", "x") for t, s in (("b", 0.5), ("a", 0.9), ("c", 0.1))]
        low = pe.opro_meta_prompt(lo, [], higher_is_better=False, score_scale=None)
        self.assertIn("arranged in descending order based on their scores, where lower scores", low)
        self.assertLess(low.index("score:\n0.9"), low.index("score:\n0.5"))
        self.assertLess(low.index("score:\n0.5"), low.index("score:\n0.1"))
        self.assertIn("has a score as low as possible.", low)
        self.assertNotIn("descending", p)

    def test_search_calls_dedup_and_train_only(self):
        ev, _, _ = _toy()
        llm = _const("[use kw00]")
        out = pe.opro(llm, ev, ["use kw05"], steps=3, per_step=4, parallel=1)
        self.assertEqual(out["llm_calls"], 12)
        self.assertEqual(len(llm.calls), 12)
        self.assertEqual(out["duplicates"], 11)
        self.assertEqual([c.text for c in out["history"]], ["use kw05", "use kw00"])
        self.assertEqual(ev.unit_evals["test"], 0)
        self.assertEqual(ev.unit_evals["val"], 0)
        # exemplars come from the training split only
        val_inputs = set(ev.splits["val"]["input"])
        for _, u in llm.calls:
            self.assertFalse(any(x in u for x in val_inputs))
        bad = pe.opro(_const("no brackets"), ev, ["use kw05"], steps=1, per_step=3, parallel=1)
        self.assertEqual(bad["unparsed"], 3)

    def test_distinct_samples_per_call(self):
        seen = []

        class Rec(FakeLLM):
            def complete(self, system, user, *, max_tokens=256, **kw):
                seen.append(kw.get("sample"))
                return super().complete(system, user, max_tokens=max_tokens, **kw)
        ev, _, _ = _toy()
        pe.opro(Rec(lambda s, u: "[x]"), ev, ["use kw01"], steps=2, per_step=3, parallel=1)
        self.assertEqual(seen, list(range(6)))

    def test_toy_opro_beats_seed_on_test(self):
        ev, llm, _ = _toy()
        res = pe.evolve("opro", llm, ev, seeds=["use kw05"], steps=10, per_step=8, parallel=1)
        self.assertTrue(res.test_diff["winner-seed0"]["better"])
        self.assertGreater(res.test["winner"]["mean"], res.test["seed0"]["mean"])
        self.assertEqual(res.search["unit_evals"]["test"], 2 * ev.n("test"))


class TestAPE(unittest.TestCase):
    def test_forward_template(self):
        p = pe.ape_forward_prompt([("a", "1")] * 5)
        self.assertTrue(p.startswith("I gave a friend an instruction and five inputs. The friend read the "
                                     "instruction and wrote an output for every one of the inputs.\n"
                                     "Here are the input-output pairs:\n\nInput: a\nOutput: 1"))
        self.assertTrue(p.endswith("The instruction was"))

    def test_filter_and_resample(self):
        ev, llm, _ = _toy()
        out = pe.ape(llm, ev, n_candidates=20, subset_size=30, top_frac=0.2, resample_rounds=1, per_survivor=2,
                     parallel=1)
        n_forward = 20
        resample_prompts = [u for _, u in llm.calls if u.startswith("Generate a variation of the following "
                                                                      "instruction while keeping the semantic")]
        self.assertEqual(len(resample_prompts), out["llm_calls"] - n_forward)
        self.assertGreater(len(resample_prompts), 0)
        surv = [c for c in out["history"] if c.meta.get("survivor")]
        self.assertTrue(surv and all(c.full_train for c in surv))
        self.assertTrue(all(c.n_units == 30 for c in out["history"] if not c.meta.get("survivor")))
        self.assertEqual(ev.unit_evals["test"], 0)

    def test_needs_demos(self):
        ev = pe.PromptEvaluator(lambda t, u: np.ones(len(u)), [1, 2], [3])
        with self.assertRaises(ValueError):
            pe.ape(_const("x"), ev)


class TestPromptbreeder(unittest.TestCase):
    def _run(self, ops, **kw):
        ev, llm, _ = _toy(n=120)
        out = pe.promptbreeder(llm, ev, "Find the topic keywords.", population=6, generations=2, operators=ops,
                               parallel=1, seed=3, **kw)
        return ev, llm, out

    def test_operator_templates(self):
        expect = {"zero_order": "Find the topic keywords. A list of 100 hints:",
                  "first_order": "INSTRUCTION MUTANT:",
                  "eda_rank": "A List of Responses in descending order of score.",
                  "lineage": "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY",
                  "first_order_hyper": "Please summarize and improve the following instruction:",
                  "lamarckian": "I gave a friend an instruction and some advice. Here are the correct examples"}
        for op, s in expect.items():
            _, llm, out = self._run([op], crossover_p=0.0)
            body = [u for _, u in llm.calls[6:]]              # after the 6 initialisation calls
            self.assertTrue(any(s in u for u in body), op)
            self.assertEqual(out["operators"][op], 6)          # 2 generations x 3 tournaments
            self.assertEqual(len(out["population"]), 6)
        _, llm, _ = self._run(["eda"], crossover_p=0.0)
        self.assertTrue(all(u.rstrip().endswith(".") for _, u in llm.calls[6:]))

    def test_init_prompt_and_history(self):
        ev, llm, out = self._run(["first_order"])
        u0 = llm.calls[0][1]
        self.assertIn(" INSTRUCTION: Find the topic keywords. INSTRUCTION MUTANT:", u0)
        self.assertTrue(any(u0.startswith(m) for m in pe.MUTATION_PROMPTS))
        self.assertTrue(any(t in u0 for t in pe.THINKING_STYLES))
        self.assertEqual(len(out["history"]), 6 + 6)
        self.assertEqual(ev.unit_evals["test"], 0)

    def test_lamarckian_fallback_without_outputs(self):
        ev = pe.PromptEvaluator(lambda t, u: np.full(len(u), float("kw00" in t)), list(range(8)), list(range(4)))
        out = pe.promptbreeder(_const("use kw00"), ev, "p", population=4, generations=1, operators=["lamarckian"],
                               parallel=1)
        self.assertEqual(out["lamarckian_fallbacks"], 2)

    def test_lamarckian_correct_workings_lower_is_better_and_cached(self):
        # lower is better: "mixed" (0, 1, 0, 1, ...) beats "worse" (all 1); correct workings are its score-0 units,
        # read from its cached batch evaluation (no full-train scoring)
        sizes = []

        def score(text, units):
            sizes.append(len(units))
            s = np.array([float(u % 2) if text == "mixed" else 1.0 for u in units])
            return {"scores": s, "outputs": [f"u{u}:{int(v)}" for u, v in zip(units, s)]}
        ev = pe.PromptEvaluator(score, list(range(8)), list(range(4)), higher_is_better=False)
        llm = _const("new text")
        out = pe.promptbreeder(llm, ev, "p", seeds=["mixed", "worse"], population=2, generations=1,
                               operators=["lamarckian"], batch_size=4, crossover_p=0.0, parallel=1)
        self.assertEqual(out["lamarckian_fallbacks"], 0)
        work = llm.calls[-1][1]
        self.assertIn("correct examples of his workings out", work)
        self.assertIn(":0", work)
        self.assertNotIn(":1", work)
        self.assertNotIn(8, sizes)

    def test_generation_pairs_each_unit_once(self):
        n = {"k": 0}

        def fresh(s, u):
            n["k"] += 1
            return f"text {n['k']}"
        ev = pe.PromptEvaluator(lambda t, u: np.full(len(u), float(len(t))), list(range(8)), list(range(4)))
        seeds = [f"seed {'x' * i}" for i in range(6)]                       # distinct fitness
        for sd in range(10):
            out = pe.promptbreeder(FakeLLM(fresh), ev, "p", seeds=seeds, population=6, generations=1,
                                   operators=["first_order"], crossover_p=0.0, parallel=1, seed=sd)
            kept = [u["task"] for u in out["population"] if u["task"] in seeds]
            self.assertEqual(len(kept), 3, sd)                               # 3 disjoint pairs, 3 losers replaced

    def test_batches_and_bad_operator(self):
        _, _, out = self._run(["first_order"], batch_size=10)
        self.assertTrue(all(c.n_units == 10 for c in out["history"]))
        with self.assertRaises(ValueError):
            self._run(["mutate_everything"])

    def test_diversity_filter(self):
        keep = pe.diversity_filter(["Solve the problem step by step, then check it.",
                                    "solve the problem step by step, then check it",
                                    "Draw a picture first."])
        self.assertEqual(keep, [0, 2])


class TestSelectAndTest(unittest.TestCase):
    def test_seal_prereg_and_once(self):
        ev, llm, _ = _toy()
        with tempfile.TemporaryDirectory() as d:
            p = Prereg(title="toy", question="does OPRO beat the seed?", falsification="winner-seed0 CI includes 0")
            lock_prereg(p.write(Path(d) / "prereg.md"))
            res = pe.evolve("opro", llm, ev, seeds=["use kw05", "use kw09"], steps=3, parallel=1,
                            registry_dir=Path(d) / ".prereg", prereg=Path(d) / "prereg.md")
            self.assertEqual(set(res.test), {"winner", "seed0", "seed1"})
            sealed = {"method": "opro", "winner": res.best.text, "train": res.best.train,
                      "val": float(res.validation.val.iloc[0]), "top_k": res.validation.text.tolist(),
                      "baselines": {"seed0": "use kw05", "seed1": "use kw09"}}
            self.assertTrue(verify_seal(sealed, registry_dir=Path(d) / ".prereg")["matched"])
            b = res.test_diff["winner-seed0"]
            self.assertLessEqual(b["bonferroni"]["lo"], b["ci95"]["lo"])
            with self.assertRaises(PermissionError):
                pe.select_and_test(ev, [res.best])

    def test_edited_prereg_blocks_test(self):
        ev, llm, _ = _toy()
        with tempfile.TemporaryDirectory() as d:
            f = Prereg(title="toy", question="q").write(Path(d) / "p.md")
            lock_prereg(f)
            f.write_text(f.read_text() + "\nedited\n")
            with self.assertRaises(ValueError):
                pe.evolve("opro", llm, ev, seeds=["use kw05"], steps=1, parallel=1, prereg=f)
            self.assertEqual(ev.unit_evals["test"], 0)

    def test_validation_picks_and_lower_is_better(self):
        # train favours "a", validation favours "b"; lower is better
        tr, va, te = list(range(6)), list(range(4)), list(range(4))
        tbl = {("a", 6): 0.1, ("b", 6): 0.3, ("a", 4): 0.9, ("b", 4): 0.2}

        def score(text, units):
            return np.full(len(units), tbl[(text, len(units))])
        ev = pe.PromptEvaluator(score, tr, va, te, higher_is_better=False)
        hist = [ev.candidate("a", "m", "x"), ev.candidate("b", "m", "x")]
        res = pe.select_and_test(ev, hist, baselines={"a": "a"})
        self.assertEqual(res.best.text, "b")
        self.assertAlmostEqual(res.gaps["train_minus_val"], 0.1)
        self.assertTrue(res.test_diff["winner-a"]["better"])

    def test_subset_only_candidates_rescored(self):
        ev, llm, _ = _toy()
        res = pe.evolve("ape", llm, ev, n_candidates=12, subset_size=20, top_frac=0.1, top_k=4, parallel=1)
        full = res.history[res.history.full_train]
        self.assertTrue(set(res.validation.text) <= set(full.text))
        self.assertEqual(len(res.validation), 4)

    def test_promptbreeder_evolve_and_errors(self):
        ev, llm, _ = _toy(n=120)
        res = pe.evolve("promptbreeder", llm, ev, seeds=["use kw05"], problem="Find keywords.", population=6,
                        generations=2, parallel=1)
        self.assertIn("operators", res.search)
        json.dumps(res.test, default=float)
        with self.assertRaises(ValueError):
            pe.evolve("gepa", llm, ev, seeds=["x"])
        with self.assertRaises(ValueError):
            pe.evolve("opro", llm, ev)


if __name__ == "__main__":
    unittest.main()
