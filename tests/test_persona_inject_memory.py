import json
import unittest

import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM
from simengine.persona.inject import memory as mem

QS = [{"id": "q0", "text": "Do you own a car?", "options": ["yes", "no"]},
      {"id": "q1", "text": "How often do you drive your car to work?", "options": ["daily", "never"]},
      {"id": "q2", "text": "Do you read books at night?", "options": ["yes", "no"]},
      {"id": "q3", "text": "Would you buy an electric car?", "options": ["yes", "no"]},
      {"id": "q4", "text": "How many books did you read last year?", "min": 0, "max": 100}]


def person(pid="u1", target_answer="no"):
    return {"id": pid, "answers": {"q0": "yes", "q1": "daily", "q2": "no", "q3": target_answer, "q4": 12}}


class TestRegistry(unittest.TestCase):
    def test_listed(self):
        rows = {r["name"]: r for r in pi.list_injectors("memory")}
        self.assertEqual(set(rows), {"memory_stream", "memory_reflect", "memory_session"})
        for r in rows.values():
            self.assertIn("arXiv:2304.03442", r["paper"])
        self.assertEqual(rows["memory_reflect"]["needs"], ["answers", "reflections"])
        self.assertNotIn("memory", pi.IMPORT_ERRORS)

    def test_label_short_for_long_params(self):
        inj = pi.get_injector("memory_stream", k=3, questions=QS)
        self.assertTrue(inj.label.startswith("memory_stream(k=3,questions=#"))
        self.assertLess(len(inj.label), 50)
        self.assertEqual(inj.label, pi.get_injector("memory_stream", k=3, questions=QS).label)
        self.assertEqual(pi.get_injector("memory_stream").label, "memory_stream")

    def test_bad_params(self):
        for kw in ({"k": 0}, {"decay": 0}, {"weights": (1, 1)}, {"weights": (1, -1, 1)}):
            with self.assertRaises(ValueError):
                pi.get_injector("memory_stream", **kw)
        with self.assertRaises(ValueError):
            pi.get_injector("memory_session", prior_only=True)


class TestMissingFields(unittest.TestCase):
    def test_missing_lists_needs(self):
        for name, missing in (("memory_stream", "answers"), ("memory_session", "answers"),
                              ("memory_reflect", "reflections")):
            p = {"id": "x", "answers": {"q0": "yes"}} if missing == "reflections" else {"id": "x"}
            with self.assertRaises(pi.MissingPersonField) as cm:
                pi.get_injector(name)(p, QS[0])
            self.assertIn(missing, str(cm.exception))
            self.assertIn(name, str(cm.exception))


class TestScoring(unittest.TestCase):
    def test_minmax_and_cosine(self):
        self.assertEqual(mem.minmax([2, 2]), [0.0, 0.0])
        self.assertEqual(mem.minmax([1, 3, 2]), [0.0, 1.0, 0.5])
        self.assertEqual(mem.bow_cosine("the car", "a car"), 1.0)
        self.assertEqual(mem.bow_cosine("the", "car"), 0.0)

    def test_relevance_wins_with_relevance_only(self):
        inj = pi.get_injector("memory_stream", k=2, questions=QS, weights=(0, 0, 1))
        p = inj(person(), QS[3])                         # electric car -> the two car answers
        self.assertEqual(p.meta["memories"], ["q0", "q1"])

    def test_recency_only_keeps_latest(self):
        p = pi.get_injector("memory_stream", k=2, questions=QS, weights=(1, 0, 0))(person(), QS[0])
        self.assertEqual(p.meta["memories"], ["q3", "q4"])

    def test_importance(self):
        imp = {("u1", "q2"): 10, ("u1", "q4"): 9}
        p = pi.get_injector("memory_stream", k=2, questions=QS, weights=(0, 1, 0), importance=imp)(person(), QS[0])
        self.assertEqual(p.meta["memories"], ["q2", "q4"])
        p = pi.get_injector("memory_stream", k=1, weights=(0, 1, 0),
                            importance=lambda per, q, a: 10 if q == "q1" else 1)(person(), QS[0])
        self.assertEqual(p.meta["memories"], ["q1"])

    def test_custom_embed(self):
        emb = lambda t: [1.0, 0.0] if "books" in t else [0.0, 1.0]
        p = pi.get_injector("memory_stream", k=2, questions=QS, weights=(0, 0, 1), embed=emb)(person(), QS[4])
        self.assertEqual(p.meta["memories"], ["q2", "q3"])   # q2 by relevance, q3 wins the zero-relevance tie by time


class TestLeakage(unittest.TestCase):
    def test_target_and_exclude_never_shown(self):
        pp = person(target_answer="SECRETVALUE")
        q = {**QS[3], "exclude": ["q1"]}
        for name, kw in (("memory_stream", {"k": None}), ("memory_session", {}),
                         ("memory_session", {"as_history": False})):
            p = pi.get_injector(name, questions=QS, **kw)({**pp, "reflections": []}, q)
            text = p.system + p.flatten()
            self.assertNotIn("SECRETVALUE", text)
            self.assertNotIn("q3", p.meta["memories"])
            self.assertNotIn("q1", p.meta["memories"])
            self.assertNotIn("drive your car", text)

    def test_reflection_seen_filter(self):
        pp = {**person(), "reflections": [
            {"text": "Cares about cars", "sources": ["q0"], "seen": ["q0", "q1", "q2"]},
            {"text": "Used the target", "sources": ["q0"], "seen": ["q0", "q3"]}]}
        p = pi.get_injector("memory_reflect", k=None, questions=QS)(pp, QS[3])
        self.assertIn("[reflection] Cares about cars", p.system)
        self.assertNotIn("Used the target", p.system)
        self.assertEqual(p.meta["reflections_dropped"], 1)
        p = pi.get_injector("memory_reflect", k=None, questions=QS)(pp, {**QS[4], "exclude": ["q2"]})
        self.assertEqual(p.meta["reflections_dropped"], 1)
        self.assertIn("Used the target", p.system)              # q3 is not the target here
        self.assertNotIn("Cares about cars", p.system)
        for bad in ({"text": "no seen list"}, {"text": "string seen", "seen": "q3"}, {"seen": ["q0"]}, "bare text"):
            with self.assertRaises(ValueError):                  # a string 'q3' must not pass as ('q', '3')
                pi.get_injector("memory_reflect")({**person(), "reflections": [bad]}, QS[3])

    def test_int_qids(self):
        qs = [{**q, "id": i} for i, q in enumerate(QS)]
        pp = {"id": "u1", "answers": {0: "yes", 1: "daily", 2: "no", 3: "SECRETVALUE"}}
        for name, kw in (("memory_stream", {}), ("memory_session", {}), ("memory_session", {"prior_only": True})):
            p = pi.get_injector(name, questions=qs, **kw)(pp, qs[3])
            self.assertEqual(p.meta["memories"], ["0", "1", "2"])
            self.assertNotIn("SECRETVALUE", p.system + p.flatten())

    def test_no_memories_left(self):
        p = pi.get_injector("memory_stream")({"id": "a", "answers": {"q0": "yes"}}, QS[0])
        self.assertIn("(no earlier answers)", p.system)
        self.assertEqual(p.meta["memories"], [])


class TestSession(unittest.TestCase):
    def test_history_turns_prior_only(self):
        p = pi.get_injector("memory_session", questions=QS, prior_only=True)(person(), QS[2])
        self.assertEqual(p.history, [("user", QS[0]["text"]), ("assistant", '{"answer": "yes"}'),
                                     ("user", QS[1]["text"]), ("assistant", '{"answer": "daily"}')])
        self.assertIn("Interviewer: Do you own a car?", p.flatten())
        p = pi.get_injector("memory_session", questions=QS, prior_only=True)(person(), QS[0])
        self.assertEqual(p.history, [])
        with self.assertRaises(ValueError):
            pi.get_injector("memory_session", questions=QS[:2], prior_only=True)(person(), QS[3])

    def test_window(self):
        p = pi.get_injector("memory_session", k=2, questions=QS)(person(), QS[0])
        self.assertEqual(p.meta["memories"], ["q3", "q4"])
        self.assertEqual(json.loads(p.history[-1][1]), {"answer": 12})


class TestDeterminism(unittest.TestCase):
    def test_render_repeatable(self):
        for name in ("memory_stream", "memory_session"):
            a = pi.get_injector(name, questions=QS)(person(), QS[3])
            b = pi.get_injector(name, questions=QS)(person(), QS[3])
            self.assertEqual((a.system, a.flatten(), a.meta), (b.system, b.flatten(), b.meta))
        ppl = [person(f"u{i}") for i in range(4)]
        s1 = pi.swap_wrapper(pi.get_injector("memory_stream", questions=QS), ppl, seed=5)(ppl[0], QS[3])
        s2 = pi.swap_wrapper(pi.get_injector("memory_stream", questions=QS), ppl, seed=5)(ppl[0], QS[3])
        self.assertEqual((s1.system, s1.meta["donor"]), (s2.system, s2.meta["donor"]))

    def test_compare_runs(self):
        ppl = [person(f"u{i}", "yes" if i % 2 else "no") for i in range(6)]
        llm = FakeLLM(lambda s, u: '{"answer": "yes"}')
        res = pi.compare_injectors(llm, ppl, QS[3:4], "answers", ["memory_stream", "memory_session"], B=50)
        self.assertLessEqual({"memory_stream", "memory_session", "swap_memory_stream"}, set(res.table["arm"]))
        self.assertEqual(res.errors.get("memory_stream", 0), 0)


class TestOfflineSteps(unittest.TestCase):
    def test_rate_importance(self):
        llm = FakeLLM(lambda s, u: "8" if "car" in u else "not a number")
        imp = mem.rate_importance(llm, [person()], QS)
        self.assertEqual(imp, {("u1", "q0"): 8, ("u1", "q1"): 8, ("u1", "q3"): 8})
        self.assertIn("poignancy", llm.calls[0][1])
        echo = FakeLLM(lambda s, u: "On the scale of 1 to 10, rating: 7")   # echoed scale is not the rating
        self.assertEqual(mem.rate_importance(echo, [{"id": "a", "answers": {"q0": "yes"}}], QS), {("a", "q0"): 7})

    def test_reflect_cites_and_hides(self):
        def responder(system, user):
            if "salient high-level questions" in user:
                return "1. What does this person think about cars?\n2. Does the person read?"
            return "1. The person relies on a car (because of 1, 2)\n2. no citation here\n3. Bad index (because of 99)"
        llm = FakeLLM(responder)
        refl = mem.reflect(llm, person(target_answer="SECRETVALUE"), QS, hide=["q3"], n_questions=2, k=3)
        self.assertEqual(len(llm.calls), 3)
        self.assertTrue(all("SECRETVALUE" not in u for _, u in llm.calls))
        self.assertEqual(len(refl), 2)
        self.assertEqual(refl[0]["text"], "The person relies on a car")
        self.assertNotIn("q3", refl[0]["seen"])
        self.assertTrue(set(refl[0]["sources"]) <= set(refl[0]["seen"]))
        pp = {**person(), "reflections": refl}
        p = pi.get_injector("memory_reflect", k=None, questions=QS)(pp, QS[3])
        self.assertEqual(p.meta["reflections_dropped"], 0)
        self.assertIn("relies on a car", p.system)
        p = pi.get_injector("memory_reflect", k=None, questions=QS)(pp, QS[0])   # q0 was seen -> dropped
        self.assertEqual(p.meta["reflections_dropped"], 2)
        self.assertEqual(mem.reflect(llm, {"id": "e", "answers": {"q0": "yes"}}, QS, hide=["q0"]), [])


if __name__ == "__main__":
    unittest.main()
