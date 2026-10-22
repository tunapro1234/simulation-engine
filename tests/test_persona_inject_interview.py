import json
import unittest

import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM
from simengine.persona.inject import interview as iv

OPTS = ["yes", "no"]
SECRET = "zebra-umbrella"          # a target answer that must never reach a prompt


def person(i=0, n_pairs=10):
    pairs = [(f"Question {k}?", f"Answer {k} from person {i}.") for k in range(n_pairs)]
    pairs[1] = ("Do you drive?", "I drive every day." if i % 2 else "I never drive.")
    return {"id": f"u{i}", "interview": pairs,
            "answers": {"q0": "yes" if i % 2 else "no", "q1": "no", "qs": SECRET},
            "interview_summary": {"drives": "daily" if i % 2 else "never", "home_town": "small town"},
            "expert_reflections": {"psychologist": ["Values routine."], "behavioral economist": ["Saves carefully."],
                            "political scientist": ["Votes in every election."], "demographer": ["Lives alone."]}}


Q_CAR = {"id": "q0", "text": "Do you own a car?", "options": OPTS}
Q_SECRET = {"id": "qs", "text": "What is your password word?", "options": [SECRET, "other"]}
Q_NUM = {"id": "qn", "text": "How many hours a week do you drive?", "min": 0, "max": 80}
NAMES = ["interview_transcript", "interview_lesion", "interview_summary", "interview_expert_reflection",
         "interview_plus_survey"]


def oracle(system, user):
    """Reads the persona from the system prompt: drives -> owns a car."""
    yes = "drive every day" in system or "drives: daily" in system
    return "Step 1) ...\n" + json.dumps({"answer": "yes" if yes else "no"})


class TestRegistry(unittest.TestCase):
    def test_registered(self):
        rows = {r["name"]: r for r in pi.list_injectors("interview")}
        self.assertEqual(sorted(rows), sorted(NAMES))
        self.assertNotIn("interview", pi.IMPORT_ERRORS)
        for r in rows.values():
            self.assertIn("2411.10109", r["paper"])
            self.assertTrue(r["description"])

    def test_missing_fields_listed(self):
        for name in NAMES:
            inj = pi.get_injector(name)
            with self.assertRaises(pi.MissingPersonField) as cm:
                inj({"id": "x"}, Q_CAR)
            for f in inj.needs:
                self.assertIn(repr(f), str(cm.exception))
        with self.assertRaises(pi.MissingPersonField):
            pi.get_injector("interview_transcript")({"id": "x", "interview": []}, Q_CAR)

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            pi.get_injector("interview_lesion", drop=1.0)
        with self.assertRaises(ValueError):
            pi.get_injector("interview_transcript", truncate="middle")
        with self.assertRaises(ValueError):
            pi.get_injector("interview_transcript", budget_tokens=0)
        with self.assertRaises(ValueError):
            pi.get_injector("interview_expert_reflection", expert="astrologer")
        self.assertEqual(pi.get_injector("interview_lesion", drop=0.4).label, "interview_lesion(drop=0.4)")
        lab = pi.get_injector("interview_plus_survey", items={"q1": "Do you take the bus?"}).label
        self.assertEqual(lab, "interview_plus_survey(items=1)")


class TestRender(unittest.TestCase):
    def test_paper_prompt(self):
        p = pi.get_injector("interview_transcript")(person(1), Q_CAR)
        self.assertIn("Interviewer: Do you drive?\nParticipant: I drive every day.", p.system)
        self.assertTrue(p.system.endswith("====="))
        self.assertIn("What you see above is an interview transcript", p.user)
        self.assertIn("Option Interpretation", p.user)
        self.assertIn('{"answer": "<your choice>"}', p.user)
        self.assertEqual(p.meta["injector"], "interview_transcript")
        p = pi.get_injector("interview_transcript", cot=False)(person(1), Q_NUM)
        self.assertNotIn("Step 1", p.user)
        self.assertIn("from 0 to 80", p.user)
        self.assertIn("Range Interpretation", pi.get_injector("interview_transcript")(person(1), Q_NUM).user)

    def test_string_interview(self):
        p = pi.get_injector("interview_transcript")({"id": "s", "interview": "Q: hi\nA: hello\n\nQ: job?\nA: none"},
                                                    Q_CAR)
        self.assertIn("Q: hi\nA: hello\n\nQ: job?\nA: none", p.system)
        self.assertEqual(iv.transcript_units("a\nb\n"), ["a", "b"])

    def test_deterministic(self):
        for name, kw in [("interview_lesion", {"drop": 0.6, "seed": 3}),
                         ("interview_transcript", {"budget_tokens": 60, "truncate": "random", "seed": 3})]:
            a = pi.get_injector(name, **kw)(person(1), Q_CAR)
            b = pi.get_injector(name, **kw)(person(1), Q_CAR)
            self.assertEqual((a.system, a.user), (b.system, b.user))
            c = pi.get_injector(name, **{**kw, "seed": 4})(person(1), Q_CAR)
            self.assertNotEqual(a.system, c.system)
        # the lesion is per person, not per question
        les = pi.get_injector("interview_lesion", drop=0.5)
        self.assertEqual(les(person(1), Q_CAR).system, les(person(1), Q_NUM).system)

    def test_lesion_counts(self):
        p = pi.get_injector("interview_lesion", drop=0.8)(person(1), Q_CAR)
        self.assertEqual((p.meta["kept"], p.meta["of"]), (2, 10))
        self.assertEqual(p.system.count("Interviewer:"), 2)
        self.assertEqual(len(iv.lesion(["only"], 0.8)), 1)              # never empties the transcript
        self.assertEqual(iv.lesion(list("abcdef"), 0.0), list("abcdef"))

    def test_budget(self):
        units = iv.transcript_units(person(0, 40)["interview"])
        full = "\n\n".join(units)
        for how in iv.TRUNCATE:
            out = "\n\n".join(iv.fit_budget(units, 400, how, seed=1))
            self.assertLessEqual(len(out), 400, how)
            self.assertLess(len(out), len(full))
        self.assertTrue(iv.fit_budget(units, 400, "head")[0].startswith("Interviewer: Question 0?"))
        self.assertIn("Question 39?", iv.fit_budget(units, 400, "tail")[-1])
        ht = iv.fit_budget(units, 400, "head_tail")
        self.assertIn("Question 0?", ht[0])
        self.assertIn("Question 39?", ht[-1])
        self.assertTrue(any("omitted" in u for u in ht))
        omitted = sum(int(u.split()[1]) for u in ht if u.startswith("[..."))
        self.assertEqual(omitted + sum(not u.startswith("[...") for u in ht), 40)   # markers count what was cut
        rnd = iv.fit_budget(units, 400, "random", seed=1)
        self.assertEqual(rnd, sorted(rnd, key=units.index))                         # original order kept
        self.assertEqual(iv.fit_budget(units, None), units)
        self.assertEqual(iv.fit_budget(["x" * 50], 10, "head"), ["x" * 10])
        p = pi.get_injector("interview_transcript", budget_tokens=100)(person(0, 40), Q_CAR)
        self.assertLessEqual(p.meta["system_chars"], 400)

    def test_expert_routing(self):
        inj = pi.get_injector("interview_expert_reflection")
        q_pol = {"id": "q9", "text": "Should the government spend more on police?", "options": OPTS}
        self.assertEqual(inj(person(), q_pol).meta["experts"], ["political scientist"])
        self.assertEqual(inj(person(), {**Q_CAR, "expert": "demographer"}).meta["experts"], ["demographer"])
        self.assertEqual(iv.keyword_expert({"text": "Are you happy?"}), "psychologist")
        # short keys need a word or an inflection: 'bet' is not 'better', 'law' not 'lawn', 'live' not 'lively'
        for text in ("Are things getting better?", "Is your lawn green?", "Are you lively?", "Is your agent kind?"):
            self.assertEqual(iv.keyword_expert({"text": text}), "psychologist", text)
        pol, eco, dem = "political scientist", "behavioral economist", "demographer"
        for text, want in [("Did you go voting?", pol), ("More public policies?", pol), ("Do you like gambling?", eco),
                           ("Do you place bets?", eco), ("Are you religious?", dem), ("Which cities?", dem)]:
            self.assertEqual(iv.keyword_expert({"text": text}), want, text)
        p = pi.get_injector("interview_expert_reflection", expert="all")(person(), Q_CAR)
        self.assertEqual(p.meta["experts"], list(iv.EXPERTS))
        for note in ("Values routine.", "Saves carefully.", "Votes in every election.", "Lives alone."):
            self.assertIn(note, p.system)
        only = pi.get_injector("interview_expert_reflection", expert="economist", transcript=False)(person(), Q_CAR)
        self.assertNotIn("Interviewer:", only.system)
        self.assertIn("Saves carefully.", only.system)
        # routed expert has no notes -> fall back to what the person has
        fb = inj({"id": "z", "interview": "hi", "expert_reflections": {"demographer": "Lives alone."}}, q_pol)
        self.assertTrue(fb.meta["expert_fallback"])
        self.assertIn("Lives alone.", fb.system)
        # an empty note list counts as missing: no bare header, fall back
        empty = {"id": "z", "interview": "hi", "expert_reflections": {"political scientist": [" "],
                                                                       "demographer": ["Lives alone."]}}
        fb = inj(empty, q_pol)
        self.assertTrue(fb.meta["expert_fallback"])
        self.assertEqual(fb.meta["experts"], ["demographer"])
        self.assertNotIn("political scientist", fb.system)
        with self.assertRaises(TypeError):                         # a memory-style list is refused clearly
            inj({"id": "z", "interview": "hi", "expert_reflections": [{"text": "x"}]}, q_pol)

    def test_plus_survey_reflections(self):
        plain = pi.get_injector("interview_plus_survey")
        self.assertNotIn("expert_reflections", plain.needs)
        self.assertNotIn("Reflections of an expert", plain(person(1), Q_CAR).system)
        inj = pi.get_injector("interview_plus_survey", expert="auto")       # v3: reflections for every agent type
        self.assertEqual(inj.label, "interview_plus_survey(expert=auto)")
        q_pol = {"id": "q9", "text": "Should the government spend more on police?", "options": OPTS}
        p = inj(person(1), q_pol)
        self.assertEqual(p.meta["experts"], ["political scientist"])
        self.assertLess(p.system.index("answers to other survey questions"), p.system.index("Votes in every"))
        self.assertNotIn("Lives alone.", p.system)
        with self.assertRaises(pi.MissingPersonField):
            inj({"id": "x", "interview": "hi", "answers": {"q1": "no"}}, Q_CAR)
        with self.assertRaises(ValueError):
            pi.get_injector("interview_plus_survey", expert="astrologer")

    def test_summary(self):
        p = pi.get_injector("interview_summary")(person(1), Q_CAR)
        self.assertIn("- drives: daily", p.system)
        self.assertIn("- home town: small town", p.system)
        self.assertNotIn("Interviewer:", p.system)


class TestLeakage(unittest.TestCase):
    def test_target_answer_never_shown(self):
        for name in NAMES:
            p = pi.get_injector(name)(person(1), Q_SECRET)
            self.assertNotIn(SECRET, p.system, name)
        s = pi.get_injector("interview_plus_survey", items={"q1": "Do you take the bus?"})(person(1), Q_CAR)
        self.assertIn("- Do you take the bus?: no", s.system)
        self.assertNotIn("- q0:", s.system)                      # the target itself
        self.assertIn(f"- qs: {SECRET}", s.system)                # other answers are fine
        ex = pi.get_injector("interview_plus_survey")(person(1), {**Q_CAR, "exclude": ["qs", "q1"]})
        self.assertIn("(none)", ex.system)
        self.assertEqual(ex.meta["n_answers"], 0)

    def test_answers_not_in_flat_fields(self):
        # dict-valued interview fields never leak into plain_fields of a flat person
        p = pi.get_injector("plain_fields")({**person(1), "age": 30}, Q_CAR)
        self.assertNotIn("daily", p.system)
        self.assertNotIn("routine", p.system)


class TestBuilders(unittest.TestCase):
    def responder(self, system, user):
        if "JSON dictionary of key facts" in user:
            return '```json\n{"drives": "daily", "pets": "a dog"}\n```'
        if "expert demographer" in user:
            return "1. Lives alone.\n2. Works nights.\n"
        if "taking notes" in user:
            return "- A note.\n"
        if "Which domain expert" in user:
            return "Political scientist."
        return ""

    def test_prepare(self):
        llm = FakeLLM(self.responder)
        base = [{"id": "a", "interview": [("Do you drive?", "Every day.")]}, {"id": "b"},
                {"id": "c", "interview": "x", "interview_summary": {"k": "v"}}]
        out = iv.prepare(llm, base)
        self.assertEqual(len(llm.calls), 1 + 4 + 4)               # b has no interview, c keeps its summary
        self.assertEqual(out[0]["interview_summary"], {"drives": "daily", "pets": "a dog"})
        self.assertEqual(out[0]["expert_reflections"]["demographer"], ["Lives alone.", "Works nights."])
        self.assertEqual(out[2]["interview_summary"], {"k": "v"})
        self.assertNotIn("expert_reflections", base[0])                  # inputs untouched
        self.assertIn("(with a PhD) taking notes", llm.calls[1][1])
        p = pi.get_injector("interview_expert_reflection", expert="demographer")(out[0], Q_CAR)
        self.assertIn("- Works nights.", p.system)

    def test_parsers_and_router(self):
        self.assertEqual(iv.parse_summary("- Home town: Small\n- Siblings: none"),
                         {"home_town": "Small", "siblings": "none"})
        self.assertEqual(iv.parse_summary("free text"), {"summary": "free text"})
        self.assertEqual(iv.parse_summary(""), {})
        llm = FakeLLM(self.responder)
        self.assertEqual(iv.route_expert(llm, Q_CAR), "political scientist")
        qs = iv.route_experts(llm, [Q_CAR, {**Q_NUM, "expert": "demographer"}])
        self.assertEqual([q["expert"] for q in qs], ["political scientist", "demographer"])
        self.assertEqual(iv.route_expert(FakeLLM(lambda s, u: "no idea"), {"text": "Do you vote?"}),
                         "political scientist")                    # unusable reply -> keyword rule


class TestCompare(unittest.TestCase):
    def test_compare_beats_controls(self):
        people = [person(i) for i in range(8)]
        llm = FakeLLM(oracle)
        res = pi.compare_injectors(llm, people, [Q_CAR], "answers", ["interview_transcript", "interview_summary"],
                                   B=200, max_tokens=1024)
        t = res.table.set_index("arm")
        self.assertAlmostEqual(t.loc["interview_transcript", "score"], 1.0)
        self.assertAlmostEqual(t.loc["interview_summary", "score"], 1.0)
        self.assertAlmostEqual(t.loc["no_persona", "score"], 0.5)
        self.assertGreater(t.loc["interview_transcript", "diff_vs_swap"], 0)
        dry = pi.compare_injectors(FakeLLM(oracle), people, [Q_CAR], "answers", ["interview_lesion"], dry_run=True)
        self.assertEqual(dry.estimate["calls"], 3 * 8)


if __name__ == "__main__":
    unittest.main()
