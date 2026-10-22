import json
import re
import unittest

import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM
from simengine.persona.inject.social import (BIAS, CLOSED_WORLD, TONES, attach_neighbors, attach_norms,
                                             neighbors_of, norm_from_dist, prevalence_item)

OPTS = ["yes", "no"]
Q = {"id": "q0", "text": "Would you adopt the new app?", "options": OPTS}
Q1 = {"id": "q1", "text": "Do you take the bus?", "options": OPTS}
QN = {"id": "qn", "text": "How many days a week do you work from home?", "min": 0, "max": 7}
NAMES = ["peer_messages", "neighbor_activity", "asch_majority", "descriptive_norm", "news_feed"]


def nb(i, ans="yes", t=None, **kw):
    n = {"id": f"n{i}", "label": f"Contact {i}", "answers": {"q0": ans, "q1": "no", "qn": i},
         "message": f"msg {i}", "action": "adopted the app" if ans == "yes" else "ignored the app", **kw}
    if t is not None:
        n["time"] = t
    return n


def person(i=1, k=4, **extra):
    return {"id": f"u{i}", "fields": {"age": 20 + i, "region": "north"},
            "answers": {"q0": "SECRET_Q0", "q1": "SECRET_Q1", "qn": 99},
            "neighbors": [nb(j) for j in range(k)],
            "norms": {"q0": {"yes": 62, "no": 38}, "q1": {"yes": 1, "no": 3}, "qn": 2.5},
            "media": [{"source": "the local paper", "text": f"story {j}", "time": j} for j in range(5)],
            **extra}


def every_text(p):
    return "\n".join([p.system, p.user, p.flatten()] + [t for _, t in p.history])


class TestRegistry(unittest.TestCase):
    def test_listed(self):
        self.assertNotIn("social", pi.IMPORT_ERRORS)
        rows = {r["name"]: r for r in pi.list_injectors("social")}
        self.assertEqual(set(rows), set(NAMES))
        for r in rows.values():
            self.assertRegex(r["paper"], r"arXiv:\d{4}\.\d{5}")
            self.assertTrue(r["needs"])
            self.assertTrue(r["description"])

    def test_bad_params(self):
        for name, kw in [("peer_messages", {"bias": "huge"}), ("peer_messages", {"select": "best"}),
                         ("neighbor_activity", {"select": "top"}), ("asch_majority", {"tone": "angry"}),
                         ("asch_majority", {"answer": "yes", "n": 0}), ("descriptive_norm", {"source": "web"}),
                         ("descriptive_norm", {"style": "injunctive"}), ("news_feed", {"select": "all"})]:
            with self.assertRaises(ValueError, msg=(name, kw)):
                pi.get_injector(name, **kw)

    def test_labels(self):
        self.assertEqual(pi.get_injector("peer_messages", bias="weak").label, "peer_messages(bias=weak)")
        self.assertEqual(pi.get_injector("asch_majority").label, "asch_majority")


class TestNeeds(unittest.TestCase):
    def test_missing_field_lists_needs(self):
        bare = {"id": "x", "fields": {"age": 30}}
        for n in NAMES:
            with self.assertRaises(pi.MissingPersonField) as cm:
                pi.get_injector(n)(bare, Q)
            msg = str(cm.exception)
            self.assertIn(n, msg)
            self.assertTrue(any(f in msg for f in ("neighbors", "norms", "media")), msg)

    def test_present_but_unusable(self):
        cases = [("peer_messages", {"neighbors": [{"id": "a", "action": "x"}]}, "message"),
                 ("neighbor_activity", {"neighbors": [{"id": "a", "answers": {"q0": "yes"}}]}, "action"),
                 ("asch_majority", {"neighbors": [{"id": "a", "answers": {"q1": "yes"}}]}, "answers['q0']"),
                 ("descriptive_norm", {"norms": {"q1": {"yes": 1}}}, "norms['q0']"),
                 ("descriptive_norm", {"norms": {"q0": {"yes": 0, "no": 0}}}, "non-empty"),
                 ("news_feed", {"media": [{"source": "tv"}]}, "text")]
        for name, extra, frag in cases:
            with self.assertRaises(pi.MissingPersonField, msg=name) as cm:
                pi.get_injector(name)({"id": "x", **extra}, Q)
            self.assertIn(frag, str(cm.exception), name)
            self.assertIn("needs=", str(cm.exception), name)

    def test_only_self_as_neighbour(self):
        p = {"id": "u1", "neighbors": [{"id": "u1", "message": "me", "answers": {"q0": "yes"}}]}
        for n in ("peer_messages", "asch_majority"):
            with self.assertRaises(pi.MissingPersonField):
                pi.get_injector(n)(p, Q)

    def test_group_placeholder_missing(self):
        inj = pi.get_injector("descriptive_norm", group="people in {city}")
        with self.assertRaises(pi.MissingPersonField) as cm:
            inj(person(), Q)
        self.assertIn("city", str(cm.exception))
        p = inj(person(fields={"city": "Rivertown"}), Q)
        self.assertIn("Among people in Rivertown", p.system)

    def test_fixed_answer_needs_nothing(self):
        inj = pi.get_injector("asch_majority", answer="no", n=3)
        p = inj({"id": "x"}, Q)
        self.assertEqual(p.meta["confederates"], ["no"] * 3)
        with self.assertRaises(ValueError):
            inj({"id": "x"}, {**Q, "options": ["a", "b"]})
        self.assertEqual(pi.list_injectors("social")[0]["needs"], ["neighbors"])   # class needs untouched


class TestRender(unittest.TestCase):
    def test_every_injector_ends_with_answer_line(self):
        for n in NAMES:
            p = pi.get_injector(n)(person(), Q)
            self.assertTrue(p.user.rstrip().endswith('{"answer": "<your choice>"}'), n)
            self.assertEqual(p.meta["injector"], n)
            q = pi.get_injector(n, **({"source": "norms"} if n == "descriptive_norm" else {}))(person(), QN)
            self.assertTrue(q.user.rstrip().endswith('{"answer": <number>}'), n)

    def test_peer_messages(self):
        p = pi.get_injector("peer_messages", k=2, bias="strong")(person(state={"q0": "undecided"}), Q)
        # Chuang et al. persona prompt order: "Role play this person." first, then the persona
        self.assertTrue(p.system.startswith("Role play this person.\n\nYou are a person with these attributes:"))
        self.assertLess(p.system.index("north"), p.system.index("change or maintain your belief"))
        self.assertIn(BIAS["strong"], p.system)
        self.assertIn(CLOSED_WORLD, p.system)
        self.assertIn("Your initial belief: undecided", p.system)
        self.assertIn('- Contact 3: "msg 3"', p.user)
        self.assertNotIn("msg 1", p.user)                      # recent: only the last 2
        self.assertEqual(p.meta["shown"], ["n2", "n3"])
        bare = pi.get_injector("peer_messages", bias="none", closed_world=False, persona=False)(person(), Q)
        self.assertNotIn("confirmation bias", bare.system)
        self.assertNotIn("Internet", bare.system)
        self.assertNotIn("north", bare.system)

    def test_recent_uses_time(self):
        ppl = person(neighbors=[nb(0, t=5), nb(1, t=1), nb(2, t=9), nb(3, t=3)])
        p = pi.get_injector("peer_messages", k=2)(ppl, Q)
        self.assertEqual(p.meta["shown"], ["n0", "n2"])        # times 5, 9, oldest first
        self.assertLess(p.user.index("msg 0"), p.user.index("msg 2"))

    def test_neighbor_activity_exposure(self):
        ppl = person(neighbors=[nb(0), nb(1, "no"), nb(2), {"id": "n3", "action": "adopted the app"}])
        p = pi.get_injector("neighbor_activity", action="adopted the app")(ppl, Q)
        self.assertIn("3 of your 4 contacts have adopted the app.", p.user)
        self.assertEqual(p.meta["exposed"], (3, 4))
        self.assertIn('- Contact 1 ignored the app: "msg 1"', p.user)
        self.assertIn("- Someone you follow adopted the app", p.user)
        self.assertNotIn("contacts have", pi.get_injector("neighbor_activity")(ppl, Q).user)

    def test_asch_tones_and_distill(self):
        ppl = person(neighbors=[nb(0), nb(1), nb(2, "no")])
        for tone, say in TONES.items():
            p = pi.get_injector("asch_majority", tone=tone)(ppl, Q)
            self.assertTrue(p.user.startswith(
                "You are one of the participants in an experiment. There are in total 4 participants. Answer the "
                "questions based on your knowledge. The conversation of the whole experiment is as follows:\n\n"
                "Organiser: Answer the following multiple choice question by giving the most appropriate response. "
                "Answer should be chosen among [yes, no].\n\nQuestion: Would you adopt the new app?\n\n"), tone)
            for i, a in enumerate(p.meta["confederates"]):     # the paper's phrasings in turn
                self.assertIn(f"Participant {i + 1}: {say[i % len(say)].format(a=a)}\n", p.user, tone)
            self.assertIn("Now, as Participant 4, it's your turn:\n", p.user)
            self.assertEqual(p.system, "")                     # persona off by default, as in the paper
        five = pi.get_injector("asch_majority", tone="neutral", answer="no", n=5)({"id": "x"}, Q).user
        self.assertIn("Participant 2: My answer is no\nParticipant 3: no, in my opinion\nParticipant 4: It's no\n"
                      "Participant 5: I think it is no\n", five)
        num = pi.get_injector("asch_majority")(ppl, QN).user
        self.assertIn("Organiser: Answer the following question by giving the most appropriate response.\n\n"
                      "Question: How many days", num)
        d = pi.get_injector("asch_majority", distill=True)(ppl, Q)
        self.assertIn('Organiser: Of the 3 participants before you, 2 chose "yes", 1 chose "no".', d.user)
        self.assertNotIn("Participant 1:", d.user)
        u = pi.get_injector("asch_majority", distill=True, answer="no", n=4)({"id": "x"}, Q)
        self.assertIn('Organiser: All participants before you have chosen "no" as the answer.', u.user)
        self.assertIn("as Participant 5", u.user)
        self.assertIn("north", pi.get_injector("asch_majority", persona=True)(ppl, Q).system)

    def test_asch_k_subsample(self):
        p = pi.get_injector("asch_majority", k=2, seed=3)(person(k=6), Q)
        self.assertEqual(len(p.meta["confederates"]), 2)
        self.assertEqual(len(set(p.meta["shown"])), 2)

    def test_descriptive_norm(self):
        s = pi.get_injector("descriptive_norm")(person(), Q).system
        self.assertIn('Among people in your community, approximately 62% chose "yes" and 38% chose "no".', s)
        self.assertIn("You are free to give any answer.", s)
        m = pi.get_injector("descriptive_norm", style="most", group="people your age")(person(), Q).system
        self.assertIn('Most people your age (62%) chose "yes".', m)
        tie = pi.get_injector("descriptive_norm", style="most")(person(norms={"q0": {"no": 1, "yes": 1}}), Q)
        self.assertIn('The most common answer among people in your community is "yes" (50%).', tie.system)
        num = pi.get_injector("descriptive_norm")(person(), QN).system
        self.assertIn("the average answer to this question is 2.5.", num)
        three = {"id": "q3", "text": "Pick", "options": ["a", "b", "c"]}
        t = pi.get_injector("descriptive_norm", decimals=1)(person(norms={"q3": {"c": 1, "a": 2, "b": 1}}), three)
        self.assertIn('approximately 50.0% chose "a", 25.0% chose "b" and 25.0% chose "c".', t.system)

    def test_descriptive_norm_from_neighbours(self):
        inj = pi.get_injector("descriptive_norm", source="neighbors")
        self.assertEqual(inj.needs, ("neighbors",))
        ppl = person(neighbors=[nb(0), nb(1), nb(2), nb(3, "no")])
        del ppl["norms"]
        self.assertIn('75% chose "yes" and 25% chose "no"', inj(ppl, Q).system)
        self.assertIn("average answer to this question is 1.5.", inj(ppl, QN).system)

    def test_news_feed(self):
        p = pi.get_injector("news_feed", k=2, context="A new virus is spreading.")(person(), Q)
        self.assertIn("Relevant memories:\n- A new virus is spreading.\n- You read in the local paper: story 3",
                      p.system)
        self.assertIn("story 4", p.system)
        self.assertNotIn("story 2", p.system)
        self.assertTrue(p.user.startswith(Q["text"]))
        plain = pi.get_injector("news_feed")({"id": "x", "media": ["Fuel prices rose."]}, Q)
        self.assertIn("- You read in the news: Fuel prices rose.", plain.system)


class TestDeterminism(unittest.TestCase):
    def test_seeded_random_selection(self):
        ppl = [person(i, k=12) for i in range(6)]
        for name in ("peer_messages", "neighbor_activity", "asch_majority"):
            kw = {"k": 4, "seed": 7} | ({} if name == "asch_majority" else {"select": "random"})
            a = [pi.get_injector(name, **kw)(p, Q).meta["shown"] for p in ppl]
            b = [pi.get_injector(name, **kw)(p, Q).meta["shown"] for p in ppl]
            c = [pi.get_injector(name, **{**kw, "seed": 8})(p, Q).meta["shown"] for p in ppl]
            self.assertEqual(a, b, name)
            self.assertNotEqual(a, c, name)
            self.assertGreater(len({tuple(x) for x in a}), 1, name)  # differs across people
        n = pi.get_injector("news_feed", k=2, select="random", seed=1)
        self.assertEqual(n(person(), Q).system, n(person(), Q).system)

    def test_render_is_pure(self):
        p = person()
        before = json.dumps(p, sort_keys=True)
        for name in NAMES:
            x, y = pi.get_injector(name)(p, Q), pi.get_injector(name)(p, Q)
            self.assertEqual((x.system, x.user, x.history), (y.system, y.user, y.history), name)
        self.assertEqual(json.dumps(p, sort_keys=True), before)


class TestLeakage(unittest.TestCase):
    def test_own_answers_never_shown(self):
        for name in NAMES:
            for q in (Q, Q1, QN):
                txt = every_text(pi.get_injector(name)(person(), q))
                self.assertNotIn("SECRET", txt, name)
                self.assertNotIn("99", txt, name)

    def test_self_loop_dropped(self):
        me = {"id": "u1", "label": "LEAK_SELF", "answers": {"q0": "LEAK"}, "message": "LEAK_SELF"}
        ppl = person(neighbors=[nb(0), me, nb(1)])
        self.assertEqual([n["id"] for n in neighbors_of(ppl)], ["n0", "n1"])
        for name in ("peer_messages", "neighbor_activity", "asch_majority"):
            self.assertNotIn("LEAK", every_text(pi.get_injector(name)(ppl, Q)), name)
        self.assertNotIn("LEAK", every_text(pi.get_injector("descriptive_norm", source="neighbors")(ppl, Q)))

    def test_social_keys_not_shown_as_attributes(self):
        flat = {"id": "x", "age": 30, "state": "LEAK_STATE", "group": "LEAK_GROUP",
                "neighbors": [nb(0)], "norms": {"q0": {"yes": 1}}, "media": ["m"]}
        for name in NAMES:
            txt = every_text(pi.get_injector(name, **({"persona": True} if name == "asch_majority" else {}))(flat, Q))
            self.assertIn("age: 30", txt, name)
            self.assertNotIn("LEAK_GROUP", txt, name)
            if name not in ("peer_messages", "neighbor_activity"):  # these show the prior view on purpose
                self.assertNotIn("LEAK_STATE", txt, name)

    def test_attach_norms_leave_one_out(self):
        ppl = [{"id": "a", "fields": {"r": "x"}, "answers": {"q0": "yes", "qn": 1}},
               {"id": "b", "fields": {"r": "x"}, "answers": {"q0": "no", "qn": 3}},
               {"id": "c", "fields": {"r": "x"}, "answers": {"q0": "no"}},
               {"id": "d", "fields": {"r": "y"}, "answers": {"q0": "yes"}}]
        out = {p["id"]: p["norms"] for p in attach_norms(ppl, [Q, QN], by="r")}
        self.assertEqual(out["a"], {"q0": {"no": 2}, "qn": 3.0})
        self.assertEqual(out["b"], {"q0": {"yes": 1, "no": 1}, "qn": 1.0})
        self.assertEqual(out["d"], {})                           # alone in its group
        self.assertNotIn("norms", ppl[0])                        # inputs untouched
        self.assertEqual(attach_norms(ppl, [Q])[0]["norms"]["q0"], {"no": 2, "yes": 1})


class TestEdgeCases(unittest.TestCase):
    def test_k_must_be_positive(self):
        for name in ("peer_messages", "neighbor_activity", "asch_majority", "news_feed"):
            for k in (0, -1, 1.5, True):
                with self.assertRaises(ValueError, msg=(name, k)):
                    pi.get_injector(name, k=k)
        self.assertEqual(len(pi.get_injector("peer_messages", k=None)(person(k=9), Q).meta["shown"]), 9)

    def test_time_order_when_all_shown(self):
        ppl = person(neighbors=[nb(0, t=5), nb(1, t=1), nb(2, t=9)])
        for kw in ({"k": None}, {"k": 10}, {"k": 2, "select": "random"}):
            p = pi.get_injector("peer_messages", **kw)(ppl, Q)
            times = [{"n0": 5, "n1": 1, "n2": 9}[i] for i in p.meta["shown"]]
            self.assertEqual(times, sorted(times), kw)          # "oldest first" holds for every k / select

    def test_blank_and_nan_values_skipped(self):
        nan = float("nan")
        ppl = person(state={"q0": nan}, text="",
                     neighbors=[nb(0), {"id": "a", "message": nan, "answers": {"q0": nan}, "action": ""},
                                {"id": "b", "message": "  ", "answers": {"q0": ""}}, nb(1, "no")],
                     norms={"q0": {"yes": 3, "no": nan}}, media=[{"text": nan}, "real story"])
        texts = "\n".join(every_text(pi.get_injector(n)(ppl, Q)) for n in NAMES)
        self.assertNotIn("nan", texts)
        self.assertNotIn("Your initial belief", texts)
        p = pi.get_injector("asch_majority")(ppl, Q)
        self.assertEqual(sorted(p.meta["confederates"]), ["no", "yes"])
        self.assertIn('approximately 100% chose "yes".',
                      pi.get_injector("descriptive_norm")(ppl, Q).system)
        with self.assertRaises(pi.MissingPersonField):
            pi.get_injector("descriptive_norm")(person(norms={"q0": nan}), Q)

    def test_pandas_na(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        ppl = person(fields={"age": 30, "city": pd.NA}, state=pd.NA,
                     neighbors=[nb(0), {"id": "a", "message": pd.NA, "answers": {"q0": pd.NA}}])
        for n in NAMES:
            self.assertNotIn("<NA>", every_text(pi.get_injector(n, **({"persona": True} if n == "asch_majority"
                                                                          else {}))(ppl, Q)), n)
        out = attach_norms([{"id": "a", "answers": {"q0": pd.NA}}, {"id": "b", "answers": {"q0": "yes"}}], [Q])
        self.assertEqual(out[0]["norms"], {"q0": {"yes": 1}})
        self.assertEqual(out[1]["norms"], {})

    def test_own_action_and_message_not_shown(self):
        # an ABM step writes the person's own action / message onto the person dict (attach_neighbors reads them);
        # for the person themself that can be the target answer, so it is never a persona attribute
        flat = {"id": "x", "age": 30, "action": "LEAK_adopted the app", "message": "LEAK_I adopted it",
                "label": "LEAK_label", "time": 7, "neighbors": [nb(0)], "norms": {"q0": {"yes": 1}}, "media": ["m"]}
        for name in NAMES:
            txt = every_text(pi.get_injector(name, **({"persona": True} if name == "asch_majority" else {}))(flat, Q))
            self.assertIn("age: 30", txt, name)
            self.assertNotIn("LEAK", txt, name)
            self.assertNotIn("time: 7", txt, name)

    def test_bridge_validation(self):
        with self.assertRaises(ValueError):
            attach_neighbors([{"id": "a"}, {"id": "a"}], [("a", "a")])
        with self.assertRaises(ValueError):
            attach_norms([{"id": 1}, {"id": "1"}], [Q])
        for bad in (-0.1, 1.5, 70):
            with self.assertRaises(ValueError, msg=bad):
                prevalence_item(bad, "the town", "fell ill")

    def test_attach_norms_matches_brute_force(self):
        import random
        rng = random.Random(0)
        ppl = [{"id": f"p{i}", "fields": {"g": rng.choice("ab")},
                "answers": {"q0": rng.choice(OPTS + [None]), "qn": rng.choice([1, 2, 5, None])}} for i in range(40)]
        out = attach_norms(ppl, [Q, QN], by="g")
        for p, o in zip(ppl, out):
            peers = [x for x in ppl if x["id"] != p["id"] and x["fields"]["g"] == p["fields"]["g"]]
            ans = [x["answers"]["q0"] for x in peers if x["answers"]["q0"] is not None]
            nums = [x["answers"]["qn"] for x in peers if x["answers"]["qn"] is not None]
            self.assertEqual(o["norms"].get("q0"), {a: ans.count(a) for a in set(ans)} or None)
            self.assertAlmostEqual(o["norms"].get("qn"), sum(nums) / len(nums))


class TestBridges(unittest.TestCase):
    PEOPLE = [{"id": "a", "label": "Ann", "message": "hi from a", "answers": {"q0": "yes"}},
              {"id": "b", "message": "hi from b", "answers": {"q0": "no"}},
              {"id": "c", "answers": {"q0": "yes"}}]

    def test_attach_neighbors(self):
        out = {p["id"]: p for p in attach_neighbors(self.PEOPLE, [("a", "b"), ("b", "c"), ("a", "a"), ("a", "z")])}
        self.assertEqual([n["id"] for n in out["a"]["neighbors"]], ["b"])
        self.assertEqual([n["id"] for n in out["b"]["neighbors"]], ["a", "c"])
        self.assertEqual(out["b"]["neighbors"][0]["label"], "Ann")
        self.assertNotIn("neighbors", self.PEOPLE[0])
        d = {p["id"]: p for p in attach_neighbors(self.PEOPLE, {"a": ["b", "c"]}, directed=True,
                                                     states={"c": {"action": "adopted"}})}
        self.assertEqual([n["id"] for n in d["a"]["neighbors"]], ["b", "c"])
        self.assertEqual(d["a"]["neighbors"][1]["action"], "adopted")
        self.assertEqual(d["b"]["neighbors"], [])
        p = pi.get_injector("asch_majority", tone="plain")(out["b"], Q)
        self.assertEqual(sorted(p.meta["confederates"]), ["yes", "yes"])

    def test_norm_from_dist_and_prevalence(self):
        self.assertEqual(norm_from_dist([0.2, 0.8], OPTS), {"yes": 0.2, "no": 0.8})
        with self.assertRaises(ValueError):
            norm_from_dist([0.2, 0.3, 0.5], OPTS)
        s = pi.get_injector("descriptive_norm")({"id": "x", "norms": {"q0": norm_from_dist([0.2, 0.8], OPTS)}}, Q)
        self.assertIn('20% chose "yes" and 80% chose "no"', s.system)
        item = prevalence_item(0.007, "the town's population", "caught new infections yesterday", time=3)
        self.assertEqual(item["text"], "0.7% of the town's population caught new infections yesterday.")
        self.assertEqual(item["time"], 3)
        n = pi.get_injector("news_feed")({"id": "x", "media": [item]}, Q)
        self.assertIn("You read in the newspaper: 0.7%", n.system)


def follower(system, user):
    """Fake respondent that sides with whatever majority the prompt shows, else says no."""
    m = re.search(r'(\d+)% chose "(yes|no)"', system)
    if m and int(m.group(1)) > 50:
        return json.dumps({"answer": m.group(2)})
    conf = re.findall(r"Participant \d+: .*?\b(yes|no)\b", user)
    if conf:
        return json.dumps({"answer": max(set(conf), key=conf.count)})
    return '{"answer": "no"}'


class TestCompare(unittest.TestCase):
    def people(self):
        # homophily: each person's neighbours mostly share their answer
        out = []
        for i in range(8):
            ans = "yes" if i % 2 else "no"
            other = "no" if ans == "yes" else "yes"
            out.append({"id": f"u{i}", "answers": {"q0": ans},
                        "neighbors": [nb(10 * i + j, ans) for j in range(3)] + [nb(10 * i + 9, other)],
                        "norms": {"q0": {ans: 70, other: 30}}})
        return out

    def test_dry_run(self):
        llm = FakeLLM(follower)
        res = pi.compare_injectors(llm, self.people(), [Q], "answers", ["asch_majority", "descriptive_norm"],
                                   dry_run=True)
        self.assertEqual(llm.calls, [])
        self.assertEqual(res.arms, ["no_persona", "asch_majority", "descriptive_norm", "swap_asch_majority",
                                    "swap_descriptive_norm"])

    def test_social_arms_beat_controls(self):
        llm = FakeLLM(follower)
        res = pi.compare_injectors(llm, self.people(), [Q], "answers",
                                   [("asch_majority", {"tone": "neutral"}), "descriptive_norm"], B=200)
        t = res.table.set_index("arm")
        for arm in ("asch_majority(tone=neutral)", "descriptive_norm"):
            self.assertAlmostEqual(t.loc[arm, "score"], 1.0, msg=arm)
            self.assertAlmostEqual(t.loc[arm, "diff_vs_no_persona"], 0.5, msg=arm)
            self.assertLess(t.loc["swap_" + arm, "score"], 1.0, msg=arm)


if __name__ == "__main__":
    unittest.main()
