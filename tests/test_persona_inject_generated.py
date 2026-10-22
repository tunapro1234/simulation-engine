import json
import re
import unittest

import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM
from simengine.persona.inject import generated as G

Q = {"id": "q1", "text": "Do you favour more public transit?", "options": ["yes", "no"]}
QX = {"id": "q2", "text": "How many cars does your household own?", "min": 0, "max": 9, "exclude": ["vote"]}
SECRET = "zz-target-answer-zz"

STORY_YOUNG = ("I am twenty three and grew up in a small coastal town where my parents ran a bakery. Moving to the "
               "city for university changed everything for me, and now I care most about my friends, cycling to "
               "work every morning and learning to cook the dishes my grandmother used to make on holidays.")
STORY_OLD = ("I am seventy and spent most of my life on a farm in the hills, raising three children with my late "
             "husband. Church, my garden and the grandchildren matter most to me now, and I still drive into town "
             "every Saturday to sell eggs at the market and catch up with the neighbours I have known for decades.")


def person(i=0, **extra):
    return {"id": f"u{i}", "fields": {"age": 25 if i % 2 == 0 else 70, "gender": "female", "vote": "left"},
            "text": f"Post {i}: the bus was late again and I missed my shift.",
            "answers": {"q1": SECRET, "q2": SECRET, "vote": SECRET},
            "generated": {"backstory": STORY_YOUNG if i % 2 == 0 else STORY_OLD,
                          "persona": f"A night-shift nurse number {i} who relies on buses.",
                          "narrative": f"You are person {i}, a nurse who commutes by bus."},
            **extra}


def pool():
    return [{"id": "young", "text": STORY_YOUNG, "demographics": {"age_bracket": {"18-29": 0.9, "65+": 0.1},
                                                                 "gender": {"female": 0.6, "male": 0.4}}},
            {"id": "old", "text": STORY_OLD, "demographics": {"age_bracket": {"18-29": 0.05, "65+": 0.95},
                                                              "gender": {"female": 0.8, "male": 0.2}}}]


ALL = [("anthology_backstory", {}), ("anthology_backstory", {"placement": "system"}),
       ("anthology_backstory", {"placement": "history", "consistency": False}),
       ("anthology_greedy", {"pool": pool(), "keys": ["age_bracket", "gender"]}),
       ("personahub", {}), ("personahub", {"style": "prefix"}),
       ("llm_narrative", {}), ("llm_narrative", {"with_fields": True})]


class TestRegistry(unittest.TestCase):
    def test_listed_with_arxiv(self):
        rows = {r["name"]: r for r in pi.list_injectors("generated")}
        self.assertEqual(set(rows), {"anthology_backstory", "anthology_greedy", "personahub", "llm_narrative"})
        for r in rows.values():
            self.assertRegex(r["paper"], r"arXiv:\d{4}\.\d{5}")
            self.assertTrue(r["needs"] and r["description"])
        self.assertNotIn("generated", pi.IMPORT_ERRORS)

    def test_labels(self):
        self.assertEqual(pi.get_injector("anthology_backstory").label, "anthology_backstory")
        self.assertEqual(pi.get_injector("anthology_backstory", placement="system").label,
                         "anthology_backstory(placement=system)")
        self.assertEqual(pi.get_injector("personahub", style="prefix").label, "personahub(style=prefix)")
        self.assertEqual(G.PersonaHub(style="prefix").label, "personahub(style=prefix)")
        self.assertEqual(G.PersonaHub().label, "personahub")

    def test_bad_params(self):
        for name, kw in [("anthology_backstory", {"placement": "middle"}), ("personahub", {"style": "vibes"}),
                         ("anthology_greedy", {"pool": pool()})]:
            with self.assertRaises(ValueError, msg=(name, kw)):
                pi.get_injector(name, **kw)

    def test_unknown_pool(self):
        inj = pi.get_injector("anthology_greedy", pool="nope", keys=["gender"])
        with self.assertRaisesRegex(ValueError, "register_pool"):
            inj(person(0), Q)


class TestMissing(unittest.TestCase):
    def test_no_generated_lists_needs(self):
        for name in ("anthology_backstory", "personahub", "llm_narrative"):
            with self.assertRaises(pi.MissingPersonField) as cm:
                pi.get_injector(name)({"id": "u9", "fields": {"age": 30}}, Q)
            self.assertIn("generated", str(cm.exception), name)

    def test_wrong_generated_key_names_builder(self):
        p = person(0)
        p["generated"] = {"persona": "A teacher."}
        with self.assertRaisesRegex(pi.MissingPersonField, r"\['generated'\].*narrate_people"):
            pi.get_injector("llm_narrative")(p, Q)
        with self.assertRaisesRegex(pi.MissingPersonField, "backstory"):
            pi.get_injector("anthology_backstory")(p, Q)

    def test_greedy_needs_fields_and_traits(self):
        inj = pi.get_injector("anthology_greedy", pool=pool(), keys=["age_bracket", "gender"])
        with self.assertRaisesRegex(pi.MissingPersonField, "fields"):
            inj({"id": "u9"}, Q)
        with self.assertRaisesRegex(pi.MissingPersonField, r"\['gender'\]"):
            inj({"id": "u9", "fields": {"age": 40}}, Q)

    def test_narrative_with_fields_needs_fields(self):
        p = person(0)
        del p["fields"]
        p["answers"] = {}
        with self.assertRaisesRegex(pi.MissingPersonField, "fields"):
            pi.get_injector("llm_narrative", with_fields=True)(p, Q)
        pi.get_injector("llm_narrative")(p, Q)               # plain variant does not need them

    def test_narrative_prompt_without_fields(self):
        with self.assertRaises(pi.MissingPersonField):
            G.narrative_prompt({"id": "u9"})


class TestRender(unittest.TestCase):
    def test_backstory_prefix_is_paper_layout(self):
        p = pi.get_injector("anthology_backstory")(person(0), Q)
        self.assertEqual(p.system, "")
        self.assertTrue(p.user.startswith(f"Question: {G.BACKSTORY_QUESTION}\n\nAnswer: {STORY_YOUNG}"))
        self.assertIn(G.CONSISTENCY + "\nQuestion: " + Q["text"], p.user)
        self.assertIn('"answer"', p.user)
        self.assertEqual(p.meta["injector"], "anthology_backstory")

    def test_backstory_system_and_history(self):
        s = pi.get_injector("anthology_backstory", placement="system")(person(1), Q)
        self.assertIn(STORY_OLD, s.system)
        self.assertNotIn(STORY_OLD, s.user)
        h = pi.get_injector("anthology_backstory", placement="history", consistency=False)(person(1), Q)
        self.assertEqual(h.history, [("user", G.BACKSTORY_QUESTION), ("assistant", STORY_OLD)])
        self.assertNotIn(G.CONSISTENCY, h.user)
        self.assertIn(STORY_OLD, h.flatten())

    def test_personahub_styles(self):
        r = pi.get_injector("personahub")(person(2), Q)
        self.assertIn("night-shift nurse number 2", r.system)
        x = pi.get_injector("personahub", style="prefix")(person(2), Q)
        self.assertEqual(x.system, "")
        self.assertTrue(x.user.startswith("Persona: A night-shift nurse number 2"))

    def test_narrative_with_fields_hides_target_and_exclude(self):
        p = person(0)
        p["fields"]["q2"] = 3
        out = pi.get_injector("llm_narrative", with_fields=True)(p, QX)
        self.assertIn("You are person 0", out.system)
        self.assertIn("- age: 25", out.system)
        self.assertNotIn("- q2:", out.system)
        self.assertNotIn("- vote:", out.system)                # QX excludes "vote"
        self.assertIn("- vote: left", pi.get_injector("llm_narrative", with_fields=True)(p, Q).system)

    def test_greedy_picks_by_edge_weight(self):
        inj = pi.get_injector("anthology_greedy", pool=pool(), keys=["age_bracket", "gender"])
        self.assertEqual(inj(person(0), Q).meta["story"], "young")
        old = inj(person(1), Q)
        self.assertEqual(old.meta["story"], "old")
        self.assertAlmostEqual(old.meta["weight"], 0.95 * 0.8)
        self.assertIn(STORY_OLD, old.user)

    def test_greedy_refuses_target_key(self):
        inj = pi.get_injector("anthology_greedy", pool=pool(), keys=["gender", "vote"])
        with self.assertRaisesRegex(ValueError, "target"):
            inj(person(0), QX)

    def test_registered_pool(self):
        G.register_pool("t_pool", pool(), replace=True)
        with self.assertRaises(ValueError):
            G.register_pool("t_pool", pool())
        with self.assertRaises(ValueError):
            G.register_pool("released", pool())
        inj = pi.get_injector("anthology_greedy", pool="t_pool", keys=["age_bracket"])
        self.assertEqual(inj.label, "anthology_greedy(keys=['age_bracket'],pool=t_pool)")
        self.assertEqual(inj(person(1), Q).meta["story"], "old")

    def test_deterministic(self):
        for name, kw in ALL:
            a = pi.get_injector(name, **kw)(person(1), QX)
            b = pi.get_injector(name, **kw)(person(1), QX)
            self.assertEqual((a.system, a.user, a.history), (b.system, b.user, b.history), name)

    def test_no_target_answer_leakage(self):
        for name, kw in ALL:
            for q in (Q, QX):
                p = pi.get_injector(name, **kw)(person(0), q)
                self.assertNotIn(SECRET, p.system + p.flatten(), (name, kw))

    def test_swap_uses_donor_backstory(self):
        ppl = [person(0), person(1)]
        s = pi.swap_wrapper("anthology_backstory", ppl, seed=0)
        self.assertIn(STORY_OLD, s(ppl[0], Q).user)
        self.assertEqual(s(ppl[0], Q).meta["donor"], "u1")


class TestMatching(unittest.TestCase):
    def test_age_bracket(self):
        self.assertEqual([G.age_bracket(a) for a in (18, 29.9, 30, 49, 50, 64, 65, "65+", 17, "x", None)],
                         ["18-29", "18-29", "30-49", "30-49", "50-64", "50-64", "65+", "65+", None, None, None])

    def test_edge_weight_is_product(self):
        f = {"age": 70, "gender": "male"}
        self.assertAlmostEqual(G.edge_weight(f, pool()[1], ["age_bracket", "gender"]), 0.95 * 0.2)
        self.assertEqual(G.edge_weight({"age": 40}, pool()[1], ["age_bracket"]), 0.0)
        self.assertEqual(G.edge_weight({"x": 1}, {"demographics": {"x": 1.0}}, ["x"]), 1.0)  # point value, 1 == 1.0

    def test_greedy_reuses_and_ties_follow_seed(self):
        ppl = [{"id": f"p{i}", "fields": {"age": 22}} for i in range(4)]
        m = G.match_backstories(ppl, pool(), ["age_bracket"])
        self.assertEqual({s["id"] for s in m.values()}, {"young"})       # stories reused
        tied = [{"id": f"s{i}", "text": "x", "demographics": {"age_bracket": "18-29"}} for i in range(6)]
        ppl = [{"id": f"p{i}", "fields": {"age": 22}} for i in range(12)]
        a = G.match_backstories(ppl, tied, ["age_bracket"], seed=1)
        self.assertEqual(a, G.match_backstories(ppl, tied, ["age_bracket"], seed=1))
        picks = {s: {k: v["id"] for k, v in G.match_backstories(ppl, tied, ["age_bracket"], seed=s).items()}
                 for s in range(5)}
        self.assertGreater(len({tuple(sorted(v.items())) for v in picks.values()}), 1)

    def test_hungarian_one_to_one(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy missing")
        ppl = [{"id": "a", "fields": {"age": 23}}, {"id": "b", "fields": {"age": 23}}]
        m = G.match_backstories(ppl, pool(), ["age_bracket"], method="hungarian")
        self.assertEqual(sorted(s["id"] for s in m.values()), ["old", "young"])
        with self.assertRaisesRegex(ValueError, "at least as many"):
            G.match_backstories(ppl * 2, pool(), ["age_bracket"], method="hungarian")

    def test_bad_pool_and_method(self):
        with self.assertRaises(ValueError):
            G.match_backstories([person(0)], [], ["gender"])
        with self.assertRaisesRegex(ValueError, "demographics"):
            G.match_backstories([person(0)], [{"id": "s", "text": "x"}], ["gender"])
        with self.assertRaises(ValueError):
            G.match_backstories([person(0)], pool(), ["gender"], method="auction")


class TestGeneration(unittest.TestCase):
    def test_generate_backstories(self):
        replies = iter([STORY_YOUNG + "\nQuestion: next one?", "too short", STORY_OLD])
        llm = FakeLLM(lambda s, u: next(replies))
        out = G.generate_backstories(llm, 3, parallel=1)
        self.assertEqual([s["id"] for s in out], ["bs0", "bs2"])
        self.assertEqual(out[0]["text"], STORY_YOUNG)                     # stop sequence cut
        self.assertTrue(all(u == f"Question: {G.BACKSTORY_QUESTION}\n\nAnswer:" for _, u in llm.calls))
        with self.assertRaises(ValueError):
            G.generate_backstories(llm, -1)

    def test_estimate_demographics(self):
        def resp(s, u):
            return json.dumps({"answer": "65+" if "farm" in u else "18-29"})
        q = {"id": "age_bracket", "text": "How old are you?", "options": ["18-29", "30-49", "50-64", "65+"]}
        llm = FakeLLM(resp)
        out = G.estimate_demographics(llm, [{"id": "a", "text": STORY_OLD}, {"id": "b", "text": STORY_YOUNG}],
                                      [q], samples=3, parallel=1)
        self.assertEqual(out[0]["demographics"], {"age_bracket": {"65+": 1.0}})
        self.assertEqual(out[1]["demographics"], {"age_bracket": {"18-29": 1.0}})
        self.assertEqual(len(llm.calls), 6)
        self.assertIn(G.CONSISTENCY, llm.calls[0][1])
        with self.assertRaises(ValueError):
            G.estimate_demographics(llm, [], [{"id": "age", "text": "Age?"}])
        none = G.estimate_demographics(FakeLLM(lambda s, u: "unclear"), [{"id": "a", "text": "x"}], [q], samples=2)
        self.assertEqual(none[0]["demographics"], {"age_bracket": {}})

    def test_text_to_persona_never_reads_answers(self):
        llm = FakeLLM(lambda s, u: "  A commuter nurse.  ")
        ppl = [person(0), {"id": "u7", "answers": {"q1": SECRET}}]
        out = G.text_to_persona(llm, ppl, relation="write")
        self.assertEqual(out, {"u0": "A commuter nurse."})
        self.assertIn("Who is likely to write the following text?", llm.calls[0][1])
        self.assertIn("the bus was late", llm.calls[0][1])
        self.assertNotIn(SECRET, "".join(s + u for s, u in llm.calls))
        with self.assertRaises(ValueError):
            G.text_to_persona(llm, ppl, relation="review")
        p = G.attach(ppl, out, "persona")
        self.assertEqual(p[0]["generated"]["persona"], "A commuter nurse.")
        self.assertEqual(p[0]["generated"]["backstory"], STORY_YOUNG)   # other generated keys kept
        self.assertNotIn("generated", p[1])

    def test_persona_to_persona_and_dedup(self):
        n = iter(range(100))
        llm = FakeLLM(lambda s, u: f"A relative number {next(n)}.")
        out = G.persona_to_persona(llm, ["A nurse.", "A driver."], rounds=2, parallel=1)
        self.assertEqual(len(out), 4)
        self.assertIn("Who is in close relationship with the given persona?", llm.calls[0][1])
        self.assertEqual(G.dedup_personas(["A tall nurse", "a tall nurse!", "A bus driver"]),
                         ["A tall nurse", "A bus driver"])

    def test_narrate_people_uses_fields_only(self):
        llm = FakeLLM(lambda s, u: "You are a nurse.")
        out = G.narrate_people(llm, [person(0)], exclude=["vote"], words=80)
        self.assertEqual(out, {"u0": "You are a nurse."})
        s, u = llm.calls[0]
        self.assertEqual(s, G.NARRATIVE_SYSTEM)
        self.assertIn("- age: 25", u)
        self.assertIn("80 words", u)
        self.assertNotIn("vote", u)
        self.assertNotIn(SECRET, s + u)
        self.assertNotIn("bus was late", u)                            # free text is not a field


class TestCompare(unittest.TestCase):
    def test_compare_with_fake_llm(self):
        # oracle: "no" when the persona text mentions a farm (the old story), else "yes"
        def oracle(system, user):
            return json.dumps({"answer": "no" if re.search(r"\bfarm\b", system + user) else "yes"})
        ppl = [person(i) for i in range(6)]
        truth = {(p["id"], "q1"): ("yes" if i % 2 == 0 else "no") for i, p in enumerate(ppl)}
        res = pi.compare_injectors(FakeLLM(oracle), ppl, [Q], truth, ["anthology_backstory"], B=200)
        t = res.table.set_index("arm")
        self.assertAlmostEqual(t.loc["anthology_backstory", "score"], 1.0)
        self.assertAlmostEqual(t.loc["no_persona", "score"], 0.5)
        self.assertLess(t.loc["swap_anthology_backstory", "score"], 1.0)


if __name__ == "__main__":
    unittest.main()
