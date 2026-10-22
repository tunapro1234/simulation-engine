import json
import unittest

import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM
from simengine.persona.inject.framing import FORMATS, PLACEMENTS, VOICES, grid

OPTS = ["yes", "no"]
Q = {"id": "q0", "text": "Do you own a car?", "options": OPTS}
Q1 = {"id": "q1", "text": "Do you take the bus?", "options": OPTS}
NAMES = ["framed", "argyle_backstory", "hu_collier_profile", "lutz_third_person", "lutz_interview"]


def person(i=1, **extra):
    return {"id": f"u{i}", "fields": {"likes_cars": "yes" if i % 2 else "no", "age": 20 + i, "region": "north",
                                      **extra},
            "answers": {"q0": "SECRET_Q0", "q1": "SECRET_Q1"}}


def every_text(p):
    return "\n".join([p.system, p.user, p.flatten()] + [t for _, t in p.history])


class TestRegistry(unittest.TestCase):
    def test_listed(self):
        rows = {r["name"]: r for r in pi.list_injectors("framing")}
        self.assertEqual(set(rows), set(NAMES))
        self.assertNotIn("framing", pi.IMPORT_ERRORS)
        for r in rows.values():
            self.assertIn("arXiv:", r["paper"])
            self.assertEqual(r["needs"], ["fields"])

    def test_bad_knob(self):
        for kw in ({"placement": "footer"}, {"voice": "royal_we"}, {"format": "yaml"}, {"order": "random"}):
            with self.assertRaises(ValueError):
                pi.get_injector("framed", **kw)

    def test_labels_distinct_in_grid(self):
        arms = grid()
        self.assertEqual(len(arms), len(PLACEMENTS) * len(VOICES) * len(FORMATS))
        self.assertEqual(len({a.label for a in arms}), len(arms))
        self.assertIn("framed(format=json,placement=user,voice=first)", {a.label for a in arms})


class TestNeeds(unittest.TestCase):
    def test_missing_fields_lists_needs(self):
        for n in NAMES:
            with self.assertRaises(pi.MissingPersonField) as cm:
                pi.get_injector(n)({"id": "x", "text": "bio only"}, Q)
            self.assertIn("fields", str(cm.exception))
            self.assertIn(n, str(cm.exception))

    def test_all_values_empty(self):
        with self.assertRaises(pi.MissingPersonField) as cm:
            pi.get_injector("framed")({"id": "x", "fields": {"age": None, "region": "", "z": float("nan")}}, Q)
        self.assertIn("non-empty", str(cm.exception))
        with self.assertRaises(pi.MissingPersonField):     # keys select nothing present
            pi.get_injector("framed", keys=["income"])(person(), Q)

    def test_empty_fragment_omitted(self):
        p = pi.get_injector("argyle_backstory")({"id": "x", "fields": {"age": 30, "income": None}}, Q)
        self.assertIn("My age is 30.", p.user)
        self.assertNotIn("income", p.user)


class TestRender(unittest.TestCase):
    def test_every_cell_renders_and_ends_with_question(self):
        for inj in grid() + [pi.get_injector(n) for n in NAMES]:
            p = inj(person(), Q)
            self.assertTrue(p.user.rstrip().endswith('{"answer": "<your choice>"}'), inj.label)
            self.assertIn("north", every_text(p), inj.label)
            self.assertEqual(p.meta["injector"], inj.name)
            self.assertEqual(pi.parse_option('{"answer": "yes"}', Q), "yes")

    def test_placement(self):
        s = pi.get_injector("framed", placement="system")(person(), Q)
        self.assertIn("region: north", s.system)
        self.assertNotIn("north", s.user)
        u = pi.get_injector("framed", placement="user")(person(), Q)
        self.assertEqual(u.system, "")
        self.assertLess(u.user.index("north"), u.user.index("own a car"))
        t = pi.get_injector("framed", placement="turns")(person(), Q)
        self.assertEqual([r for r, _ in t.history], ["user", "assistant"])
        self.assertIn("north", t.history[0][1])
        self.assertNotIn("north", t.user)

    def test_voice_and_format(self):
        r = lambda **kw: pi.get_injector("framed", placement="user", format="prose", **kw)(person(), Q).user  # noqa: E731
        self.assertIn("Your region is north.", r(voice="second"))
        self.assertIn("My region is north.", r(voice="first"))
        self.assertIn("Their region is north.", r(voice="third"))
        self.assertTrue(r(voice="third").startswith("Imagine a person"))
        j = pi.get_injector("framed", format="json")(person(), Q)
        self.assertEqual(json.loads(j.system.splitlines()[1])["region"], "north")
        qa = pi.get_injector("framed", format="qa", voice="third")(person(), Q).system
        self.assertIn("Q: What is this person's region?\nA: north", qa)
        iv = pi.get_injector("framed", format="qa", voice="interview", placement="user")(person(), Q).user
        self.assertIn("Interviewer: What is your region?\nInterviewee: My region is north.", iv)
        self.assertIn("Interviewer: Do you own a car?", iv)

    def test_presets(self):
        hc = pi.get_injector("hu_collier_profile")(person(), Q)
        self.assertTrue(hc.system.startswith("Your Profile:\nYour likes cars is yes."))
        lt = pi.get_injector("lutz_third_person")(person(), Q)
        self.assertTrue(lt.user.startswith("Think of a person"))
        li = pi.get_injector("lutz_interview")(person(), Q)
        self.assertTrue(li.system.startswith("You are an interviewee."))
        self.assertEqual(len(li.history), 6)                     # 3 attributes x (question, answer)
        self.assertEqual(li.history[-1], ("assistant", "My region is north."))
        ab = pi.get_injector("argyle_backstory", phrases={"age": "In terms of my age, I am {value}."})(person(), Q)
        self.assertIn("In terms of my age, I am 21.", ab.user)
        self.assertEqual(pi.get_injector("argyle_backstory", voice="third").voice, "third")   # knobs still work

    def test_order_and_keys(self):
        f = {"b": 1, "a": 2, "c": 3}
        keys = lambda **kw: pi.get_injector("framed", **kw)({"id": "x", "fields": f}, Q).meta["keys"]  # noqa: E731
        self.assertEqual(keys(), ["b", "a", "c"])
        self.assertEqual(keys(order="sorted"), ["a", "b", "c"])
        self.assertEqual(keys(order="reversed"), ["c", "a", "b"])
        self.assertEqual(keys(keys=["c", "b"]), ["c", "b"])

    def test_shuffle_deterministic_with_seed(self):
        f = {f"k{i}": i for i in range(12)}
        ppl = [{"id": f"p{i}", "fields": f} for i in range(6)]
        a = [pi.get_injector("framed", order="shuffle", seed=7)(p, Q).meta["keys"] for p in ppl]
        b = [pi.get_injector("framed", order="shuffle", seed=7)(p, Q).meta["keys"] for p in ppl]
        c = [pi.get_injector("framed", order="shuffle", seed=8)(p, Q).meta["keys"] for p in ppl]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertGreater(len({tuple(k) for k in a}), 1)       # order differs across people
        self.assertTrue(all(sorted(k) == sorted(f) for k in a))
        p1 = pi.get_injector("framed", order="shuffle", seed=7)(ppl[0], Q)
        p2 = pi.get_injector("framed", order="shuffle", seed=7)(ppl[0], Q)
        self.assertEqual((p1.system, p1.user, p1.history), (p2.system, p2.user, p2.history))


class TestLeakage(unittest.TestCase):
    def test_answers_never_shown(self):
        for inj in grid() + [pi.get_injector(n) for n in NAMES]:
            txt = every_text(inj(person(), Q))
            self.assertNotIn("SECRET", txt, inj.label)

    def test_target_and_excluded_fields_dropped(self):
        p = person(q0="LEAK_TARGET", q1="LEAK_EXCL")
        for inj in grid():
            txt = every_text(inj(p, {**Q, "exclude": ["q1"]}))
            self.assertNotIn("LEAK_TARGET", txt, inj.label)
            self.assertNotIn("LEAK_EXCL", txt, inj.label)
        self.assertIn("LEAK_TARGET", every_text(pi.get_injector("framed")(p, Q1)))   # not the target there


class TestCompare(unittest.TestCase):
    def test_compare_with_fake_llm(self):
        def oracle(system, user):
            text = system + user
            likes = "likes cars: yes" in text or "likes cars is yes" in text or "likes cars?\nA: yes" in text \
                or '"likes_cars": "yes"' in text
            return json.dumps({"answer": "yes" if likes else "no"})
        ppl = [{"id": f"u{i}", "fields": {"likes_cars": "yes" if i % 2 else "no", "age": 20 + i},
                "answers": {"q0": "yes" if i % 2 else "no"}} for i in range(8)]
        arms = grid(placements=("system", "turns"), voices=("second",), formats=("bullets", "json")) + \
            ["hu_collier_profile", "lutz_interview"]
        llm = FakeLLM(oracle)
        plan = pi.compare_injectors(llm, ppl, [Q], "answers", arms, dry_run=True)
        self.assertEqual(llm.calls, [])
        self.assertEqual(len(plan.arms), 1 + 2 * 6)
        res = pi.compare_injectors(llm, ppl, [Q], "answers", arms, B=100)
        t = res.table.set_index("arm")
        for a in ("framed(format=json,placement=turns,voice=second)", "hu_collier_profile", "lutz_interview"):
            self.assertAlmostEqual(t.loc[a, "score"], 1.0, msg=a)
        self.assertAlmostEqual(t.loc["no_persona", "score"], 0.5)


if __name__ == "__main__":
    unittest.main()
