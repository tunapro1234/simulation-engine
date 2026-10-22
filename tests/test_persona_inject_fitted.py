import json
import tempfile
import unittest

import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM
from simengine.persona.inject import fitted as F

OPTS = ["yes", "no"]
FITTED = ("fitted_topk", "fitted_stepwise", "fitted_grips", "fitted_opro")
FAST = {"fitted_topk": {"k": 2, "n_estimators": 20}, "fitted_stepwise": {"max_turns": 6, "n_boot": 200},
        "fitted_grips": {"n_iter": 3, "m": 3}, "fitted_opro": {"steps": 2, "per_step": 2}}


def people(n=10):
    out = []
    for i in range(n):
        car = "yes" if i % 2 else "no"
        ans = {f"q{j}": car if j % 2 == 0 else ("no" if i % 2 else "yes") for j in range(8)}
        out.append({"id": f"u{i}", "fields": {"likes_cars": car, "age": 20 + i, "colour": "rgb"[i % 3]},
                    "answers": ans})
    return out


QS = [{"id": f"q{j}", "text": f"Do you own a car? (item {j})" if j % 2 == 0 else f"Do you take the bus? (item {j})",
       "options": OPTS} for j in range(8)]
FIT, HELD = QS[:4], QS[4:]


def oracle(system, user):
    """Scorer: answers from the likes_cars line (no line -> 'no'). Optimizer (OPRO meta-prompt): a bracketed text."""
    if "<INS>" in user or "\ntext:\n" in user:
        return "[You answer as this respondent:]"
    likes = "likes cars: yes" in system
    if "likes cars" not in system:
        return json.dumps({"answer": "no"})
    a = ("yes" if likes else "no") if "own a car" in user else ("no" if likes else "yes")
    return json.dumps({"answer": a})


def fit(name, seed=0, llm=None, **kw):
    inj = pi.get_injector(name, seed=seed, **FAST[name])
    llm = llm or FakeLLM(oracle)
    return inj, inj.fit(llm, people(), FIT, "answers", **kw), llm


class TestRegistry(unittest.TestCase):
    def test_listed_with_papers(self):
        self.assertNotIn("fitted", pi.IMPORT_ERRORS)
        rows = {r["name"]: r for r in pi.list_injectors("fitted")}
        self.assertEqual(set(rows), set(FITTED))
        for r in rows.values():
            self.assertIn("arXiv:", r["paper"])
            self.assertEqual(r["needs"], ["fields"])

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            pi.get_injector("fitted_topk", k=0)
        with self.assertRaises(ValueError):
            pi.get_injector("fitted_stepwise", start="half")
        with self.assertRaises(ValueError):
            pi.get_injector("fitted_grips", val_frac=1.0)
        with self.assertRaises(ValueError):
            pi.get_injector("fitted_opro", seeds=[])

    def test_label_shows_caller_params(self):
        self.assertEqual(pi.get_injector("fitted_topk", k=3).label, "fitted_topk(k=3)")


class TestGuards(unittest.TestCase):
    def test_missing_fields_lists_needs(self):
        inj = pi.get_injector("fitted_stepwise")
        with self.assertRaises(pi.MissingPersonField) as cm:
            inj.fit(FakeLLM(oracle), [{"id": "x", "answers": {"q0": "yes"}}], FIT, "answers")
        self.assertIn("fields", str(cm.exception))
        inj, _, _ = fit("fitted_topk")
        with self.assertRaises(pi.MissingPersonField) as cm:
            inj({"id": "x"}, HELD[0])
        self.assertIn("['fields']", str(cm.exception))

    def test_not_fitted(self):
        with self.assertRaises(F.NotFitted):
            pi.get_injector("fitted_grips")(people()[0], HELD[0])

    def test_fit_questions_refused(self):
        for name in FITTED:
            inj, res, _ = fit(name)
            for q in FIT:
                with self.assertRaises(F.HeldOutViolation, msg=name):
                    inj(people()[0], q)
            self.assertEqual(set(res.train_questions) | set(res.val_questions), {q["id"] for q in FIT})
            inj(people()[0], HELD[0])

    def test_max_calls_refuses_before_any_call(self):
        llm = FakeLLM(oracle)
        with self.assertRaises(ValueError):
            fit("fitted_grips", llm=llm, max_calls=10)
        self.assertEqual(llm.calls, [])

    def test_question_checks(self):
        inj = pi.get_injector("fitted_grips")
        with self.assertRaises(ValueError):
            inj.fit(FakeLLM(oracle), people(), [FIT[0], FIT[0]], "answers")
        with self.assertRaises(ValueError):
            inj.fit(FakeLLM(oracle), people(), FIT, "answers", val_questions=["q7"])
        with self.assertRaises(ValueError):
            inj.fit(FakeLLM(oracle), people(), FIT, "answers", val_questions=[q["id"] for q in FIT])
        with self.assertRaises(ValueError):
            inj.fit(FakeLLM(oracle), people(), FIT, {})            # no cell has truth

    def test_topk_needs_variation(self):
        flat = [{**p, "answers": {q["id"]: "yes" for q in FIT}} for p in people()]
        with self.assertRaises(ValueError):
            pi.get_injector("fitted_topk").fit(None, flat, FIT, "answers")


class TestSplitAndDeterminism(unittest.TestCase):
    def test_split_questions(self):
        ids = [f"q{i}" for i in range(8)]
        tr, va = F.split_questions(ids, 0.25, seed=3)
        self.assertEqual((len(tr), len(va)), (6, 2))
        self.assertEqual(set(tr) | set(va), set(ids))
        self.assertFalse(set(tr) & set(va))
        self.assertEqual(F.split_questions(ids, 0.25, seed=3), (tr, va))
        self.assertEqual(F.split_questions(ids, 0.0), (ids, []))
        tr2, va2 = F.split_questions(["a", "b"], 0.9)
        self.assertEqual((len(tr2), len(va2)), (1, 1))          # always one train question left

    def test_explicit_val_questions(self):
        inj = pi.get_injector("fitted_stepwise", **FAST["fitted_stepwise"])
        res = inj.fit(FakeLLM(oracle), people(), FIT, "answers", val_questions=["q1"])
        self.assertEqual(res.val_questions, ["q1"])
        self.assertEqual(res.train_questions, ["q0", "q2", "q3"])

    def test_same_seed_same_fit(self):
        for name in FITTED:
            _, a, _ = fit(name, seed=7)
            _, b, _ = fit(name, seed=7)
            self.assertEqual(a.config, b.config, name)
            self.assertEqual(a.history, b.history, name)
            self.assertEqual((a.train_questions, a.val_questions), (b.train_questions, b.val_questions), name)

    def test_render_is_pure(self):
        inj, _, _ = fit("fitted_grips")
        p = people()[3]
        self.assertEqual(inj(p, HELD[0]).system, inj(p, HELD[0]).system)


class TestSearch(unittest.TestCase):
    def test_finds_the_informative_attribute(self):
        for name in ("fitted_topk", "fitted_stepwise"):
            _, res, _ = fit(name)
            self.assertIn("likes_cars", res.config["keys"], name)
        _, res, _ = fit("fitted_stepwise")
        self.assertEqual(res.train_score, 1.0)

    def test_topk_weights_rank(self):
        _, res, llm = fit("fitted_topk")
        w = res.meta["weights"]
        self.assertEqual(max(w, key=w.get), "likes_cars")
        self.assertEqual(llm.calls, [])                      # no LLM in topk
        self.assertEqual(res.val_questions, [])

    def test_grips_edit(self):
        import random
        rng = random.Random(0)
        self.assertIsNone(F.FittedGrips.edit([], [], rng))
        k, p, op = F.FittedGrips.edit(["a"], [], rng)
        self.assertEqual((k, p, op), ([], ["a"], "del"))
        k, p, op = F.FittedGrips.edit([], ["b"], rng)
        self.assertEqual((k, p, op), (["b"], [], "add"))
        for _ in range(20):
            k, p, _ = F.FittedGrips.edit(["a", "b", "c"], ["d"], rng)
            self.assertEqual(sorted(k + p), ["a", "b", "c", "d"])

    def test_opro_counts_and_instruction(self):
        inj, res, _ = fit("fitted_opro")
        self.assertEqual(res.optimizer_calls, 4)
        self.assertIn(res.config["instruction"], {F.HEADER, "You answer as this respondent:"})
        self.assertTrue(inj(people()[1], HELD[0]).system.startswith(res.config["instruction"]))

    def test_seal(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("fitted_topk", "fitted_grips"):
                _, res, _ = fit(name, registry_dir=d)
                self.assertIsNotNone(res.seal, name)

    def test_load_round_trip(self):
        inj, res, _ = fit("fitted_grips")
        other = pi.get_injector("fitted_grips").load(json.loads(json.dumps(res.to_dict())))
        p = people()[2]
        self.assertEqual(other(p, HELD[1]).system, inj(p, HELD[1]).system)
        with self.assertRaises(F.HeldOutViolation):
            other(p, FIT[0])
        with self.assertRaises(ValueError):
            pi.get_injector("fitted_topk").load(res.to_dict())


class TestLeakage(unittest.TestCase):
    def test_no_answers_in_prompt(self):
        for name in FITTED:
            inj, res, _ = fit(name)
            p = people()[5]
            for q in HELD:
                pr = inj(p, q)
                self.assertEqual(pr.history, [])
                for qid in p["answers"]:
                    self.assertNotIn(qid, pr.system, name)
                # system = instruction + the chosen attribute lines only
                body = pr.system.splitlines()[1:-1]
                self.assertEqual(len(body), len([k for k in res.config["keys"] if k in p["fields"]]), name)

    def test_fit_never_sees_held_out_questions(self):
        for name in FITTED:
            _, _, llm = fit(name)
            seen = "\n".join(s + "\n" + u for s, u in llm.calls)
            for q in HELD:
                self.assertNotIn(q["text"], seen, name)

    def test_opro_optimizer_sees_train_only(self):
        sent = []

        def spy(system, user):
            if "<INS>" in user or "\ntext:\n" in user:
                sent.append(user)
            return oracle(system, user)
        inj = pi.get_injector("fitted_opro", **FAST["fitted_opro"])
        res = inj.fit(FakeLLM(spy), people(), FIT, "answers")
        self.assertTrue(sent)
        val_texts = [q["text"] for q in FIT if q["id"] in res.val_questions]
        for m in sent:
            for t in val_texts:
                self.assertNotIn(t, m)


class TestCompare(unittest.TestCase):
    def test_compare_on_held_out(self):
        for name in ("fitted_topk", "fitted_stepwise"):
            inj, _, llm = fit(name)
            res = pi.compare_injectors(llm, people(), HELD, "answers", [inj], B=200)
            t = res.table.set_index("arm")
            self.assertEqual(t.loc[inj.label, "score"], 1.0)
            self.assertLess(t.loc["no_persona", "score"], 1.0)
            self.assertLess(t.loc[f"swap_{inj.label}", "score"], 1.0)

    def test_compare_on_fit_questions_refused(self):
        inj, _, llm = fit("fitted_topk")
        with self.assertRaises(F.HeldOutViolation):
            pi.compare_injectors(llm, people(), FIT, "answers", [inj], B=200)


if __name__ == "__main__":
    unittest.main()
