import json
import unittest

import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM
from simengine.persona.inject._base import Injector, Prompt, register

OPTS = ["yes", "no"]


def people(n=8):
    return [{"id": f"u{i}", "fields": {"likes_cars": "yes" if i % 2 else "no", "age": 20 + i},
             "answers": {"q0": "yes" if i % 2 else "no", "q1": "no" if i % 2 else "yes"}} for i in range(n)]


QUESTIONS = [{"id": "q0", "text": "Do you own a car?", "options": OPTS},
             {"id": "q1", "text": "Do you take the bus?", "options": OPTS}]


def oracle(system, user):
    """Answers from the persona's likes_cars field; no persona -> always 'no'."""
    likes = "likes cars: yes" in system
    if "own a car" in user:
        a = "yes" if likes else "no"
    else:
        a = "no" if likes else "yes"
    return json.dumps({"answer": a}) if system else json.dumps({"answer": "no"})


class TestPrompt(unittest.TestCase):
    def test_flatten_and_messages(self):
        p = Prompt("sys", "Q?", history=[("user", "hi"), ("assistant", "hello")])
        flat = p.flatten()
        self.assertIn("Interviewer: hi", flat)
        self.assertIn("You: hello", flat)
        self.assertTrue(flat.endswith("Q?"))
        self.assertEqual(Prompt("s", "u").flatten(), "u")
        self.assertEqual(p.messages()[-1], {"role": "user", "content": "Q?"})
        with self.assertRaises(ValueError):
            Prompt("s", "u", history=[("system", "x")])


class TestRegistry(unittest.TestCase):
    def test_builtins_listed(self):
        rows = {r["name"]: r for r in pi.list_injectors()}
        self.assertIn("no_persona", rows)
        self.assertIn("plain_fields", rows)
        self.assertEqual(set(rows["plain_fields"]), {"name", "family", "paper", "needs", "description"})
        self.assertTrue(all(r["family"] == "control" for r in pi.list_injectors("control")))
        self.assertEqual(pi.IMPORT_ERRORS, {})

    def test_get_and_register(self):
        with self.assertRaises(ValueError):
            pi.get_injector("nope")
        inj = pi.get_injector("plain_fields", keys=["age"])
        self.assertIs(pi.get_injector(inj), inj)
        self.assertEqual(inj.label, "plain_fields(keys=['age'])")
        self.assertEqual(pi.get_injector("plain_fields").label, "plain_fields")

        @register
        class T(Injector):
            name, family, needs = "_test_inject_core", "test", ("text",)

            def render(self, person, question):
                return Prompt(person["text"], self.question_block(question))
        try:
            with self.assertRaises(ValueError):
                register(type("T2", (Injector,), {"name": "_test_inject_core"}))
            with self.assertRaises(pi.MissingPersonField):
                pi.get_injector("_test_inject_core")({"id": "a"}, QUESTIONS[0])
            p = pi.get_injector("_test_inject_core")({"id": "a", "text": "I am A."}, QUESTIONS[0])
            self.assertEqual(p.meta["injector"], "_test_inject_core")
        finally:
            pi.REGISTRY.pop("_test_inject_core", None)


class TestBuiltins(unittest.TestCase):
    def test_no_persona_and_plain_fields(self):
        p0 = pi.get_injector("no_persona")(people()[1], QUESTIONS[0])
        self.assertEqual(p0.system, "")
        self.assertIn('"answer"', p0.user)
        p = pi.get_injector("plain_fields")(people()[1], QUESTIONS[0])
        self.assertIn("- likes cars: yes", p.system)
        self.assertIn("- age: 21", p.system)
        p = pi.get_injector("plain_fields", keys=["age"])(people()[1], QUESTIONS[0])
        self.assertNotIn("likes", p.system)
        flat = pi.get_injector("plain_fields")({"id": "x", "region": "north", "text": "skip me"}, QUESTIONS[0])
        self.assertIn("region: north", flat.system)
        self.assertNotIn("skip me", flat.system)

    def test_known_answers_drops_target(self):
        q = {**QUESTIONS[0], "exclude": ["q1"]}
        self.assertEqual(Injector.known_answers(people()[0], q), {})
        self.assertEqual(Injector.known_answers(people()[0], QUESTIONS[0]), {"q1": "yes"})

    def test_swap_never_self(self):
        ps = people(5)
        d = pi.derangement([p["id"] for p in ps], seed=3)
        self.assertTrue(all(k != v for k, v in d.items()))
        self.assertEqual(sorted(d.values()), sorted(d))
        s = pi.swap_wrapper("plain_fields", ps, seed=3)
        p = s(ps[0], QUESTIONS[0])
        self.assertEqual(p.meta["donor"], d["u0"])
        with self.assertRaises(ValueError):
            pi.swap_wrapper("plain_fields", ps[:1])

    def test_parse_option(self):
        q = QUESTIONS[0]
        self.assertEqual(pi.parse_option('{"answer": "Yes"}', q), "yes")
        self.assertEqual(pi.parse_option("no", q), "no")
        self.assertIsNone(pi.parse_option("I think yes", q))
        self.assertEqual(pi.parse_option('{"answer": 4}', {"text": "n?", "min": 0, "max": 5}), 4)


class TestCompare(unittest.TestCase):
    def test_dry_run_no_calls(self):
        llm = FakeLLM(oracle)
        res = pi.compare_injectors(llm, people(), QUESTIONS, "answers", ["plain_fields"], dry_run=True)
        self.assertEqual(llm.calls, [])
        self.assertEqual(res.arms, ["no_persona", "plain_fields", "swap_plain_fields"])
        self.assertEqual(res.estimate["calls"], 3 * 8 * 2)
        self.assertIsNone(res.table)
        self.assertIsNone(res.estimate["usd_max"])        # fake model has no price

    def test_compare_table(self):
        llm = FakeLLM(oracle)
        res = pi.compare_injectors(llm, people(), QUESTIONS, "answers", ["plain_fields"], B=200)
        self.assertEqual(len(llm.calls), 48)
        t = res.table.set_index("arm")
        self.assertAlmostEqual(t.loc["plain_fields", "score"], 1.0)
        self.assertAlmostEqual(t.loc["no_persona", "score"], 0.5)
        self.assertAlmostEqual(t.loc["plain_fields", "diff_vs_no_persona"], 0.5)
        self.assertGreater(t.loc["plain_fields", "lo_vs_no_persona"], 0)
        self.assertGreater(t.loc["plain_fields", "diff_vs_swap"], 0)  # swapped persona answers for someone else
        self.assertEqual(set(res.frame.columns) >= {"group", "item", "truth", "pred_plain_fields"}, True)
        self.assertEqual(t.loc["plain_fields", "n_people"], 8)

    def test_truth_mapping_unparsed_and_errors(self):
        def flaky(system, user):
            if "age: 20" in system:
                raise RuntimeError("boom")
            return "maybe"
        truth = {("u0", "q0"): "no", ("u1", "q0"): "yes", ("u2", "q0"): "no"}
        res = pi.compare_injectors(FakeLLM(flaky), people(3), QUESTIONS, truth, ["plain_fields"],
                                   controls=("no_persona",), B=100)
        self.assertEqual(res.cells, [("u0", "q0"), ("u1", "q0"), ("u2", "q0")])
        t = res.table.set_index("arm")
        self.assertEqual(t.loc["plain_fields", "errors"], 1)
        self.assertEqual(t.loc["plain_fields", "parsed"], 0)
        self.assertEqual(t.loc["plain_fields", "score"], 0.0)
        self.assertNotIn("swap_plain_fields", t.index)

    def test_history_kw_only_when_supported(self):
        @register
        class H(Injector):
            name, family = "_test_hist", "test"

            def render(self, person, question):
                return Prompt("sys", self.question_block(question), history=[("user", "hi"), ("assistant", "yo")])

        class HistLLM(FakeLLM):
            supports_history = True

            def complete(self, system, user, *, max_tokens=256, **kw):
                self.seen = kw.get("history")
                return super().complete(system, user, max_tokens=max_tokens)
        try:
            plain = FakeLLM(lambda s, u: '{"answer": "yes"}')
            pi.compare_injectors(plain, people(2), QUESTIONS[:1], "answers", ["_test_hist"], controls=(), B=50)
            self.assertIn("Interviewer: hi", plain.calls[0][1])
            h = HistLLM(lambda s, u: '{"answer": "yes"}')
            pi.compare_injectors(h, people(2), QUESTIONS[:1], "answers", ["_test_hist"], controls=(), B=50)
            self.assertEqual(h.seen, [("user", "hi"), ("assistant", "yo")])
            self.assertNotIn("Interviewer", h.calls[0][1])
        finally:
            pi.REGISTRY.pop("_test_hist", None)

    def test_guards(self):
        with self.assertRaises(ValueError):
            pi.compare_injectors(FakeLLM(), people(), QUESTIONS, "answers", ["plain_fields"], controls=("bogus",))
        with self.assertRaises(ValueError):
            pi.compare_injectors(FakeLLM(), people(), QUESTIONS, {}, ["plain_fields"])
        with self.assertRaises(ValueError):
            pi.compare_injectors(FakeLLM(), people(), QUESTIONS, "answers", ["plain_fields", "plain_fields"])
        llm = FakeLLM(model="claude-haiku-5-5")
        with self.assertRaises(ValueError):
            pi.compare_injectors(llm, people(), QUESTIONS, "answers", ["plain_fields"], max_usd=0.0)
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
