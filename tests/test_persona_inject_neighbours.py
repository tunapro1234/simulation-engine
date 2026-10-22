import json
import re
import unittest

import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM

OPTS = ["yes", "no"]
Q_CAR = {"id": "q0", "text": "Do you own a car?", "options": OPTS}
Q_BUS = {"id": "q1", "text": "Do you take the bus?", "options": OPTS}
Q_NUM = {"id": "q2", "text": "How many trips last week?", "min": 0, "max": 9}
QUESTIONS = [Q_CAR, Q_BUS, Q_NUM]


def people(n=12):
    """Rural people own cars, urban ones take the bus; q2 = i % 7."""
    return [{"id": f"u{i}", "fields": {"region": "rural" if i % 2 else "urban", "age": 20 + i},
             "answers": {"q0": "yes" if i % 2 else "no", "q1": "no" if i % 2 else "yes", "q2": i % 7}}
            for i in range(n)]


def all_text(p):
    return p.system + "\n" + p.flatten()


class TestRegistry(unittest.TestCase):
    def test_listed(self):
        self.assertNotIn("neighbours", pi.IMPORT_ERRORS)
        rows = {r["name"]: r for r in pi.list_injectors("neighbours")}
        self.assertEqual(set(rows), {"similar_respondents", "neighbour_votes", "answer_neighbours"})
        for r in rows.values():
            self.assertRegex(r["paper"], r"arXiv:\d{4}\.\d{5}")
            self.assertTrue(r["needs"])

    def test_bad_params(self):
        with self.assertRaisesRegex(ValueError, "pool"):
            pi.get_injector("neighbour_votes")
        for name, kw in [("similar_respondents", {"strategy": "nope"}), ("similar_respondents", {"format": "x"}),
                         ("neighbour_votes", {"show": "x"}), ("neighbour_votes", {"basis": "x"}),
                         ("neighbour_votes", {"k": 0})]:
            with self.assertRaises(ValueError):
                pi.get_injector(name, pool=people(), **kw)
        with self.assertRaisesRegex(ValueError, "unique"):
            pi.get_injector("neighbour_votes", pool=people() + people(1))

    def test_label_summarises_pool(self):
        a = pi.get_injector("neighbour_votes", pool=people(), k=3)
        b = pi.get_injector("neighbour_votes", pool=people(), k=5)
        self.assertRegex(a.label, r"^neighbour_votes\(k=3,pool=n12#[0-9a-f]{6}\)$")
        self.assertNotEqual(a.label, b.label)
        self.assertNotIn("rural", a.label)
        c = pi.get_injector("answer_neighbours", pool=people(), questions=QUESTIONS)
        self.assertRegex(c.label, r"questions=n3#[0-9a-f]{6}")


class TestMissingFields(unittest.TestCase):
    def test_needs_listed_in_error(self):
        inj = pi.get_injector("similar_respondents", pool=people())
        with self.assertRaises(pi.MissingPersonField) as cm:
            inj({"id": "x", "answers": {"q1": "yes"}}, Q_CAR)
        self.assertIn("fields", str(cm.exception))
        both = pi.get_injector("neighbour_votes", pool=people(), basis="both")
        self.assertEqual(both.needs, ("fields", "answers"))
        with self.assertRaisesRegex(pi.MissingPersonField, "answers"):
            both({"id": "x", "fields": {"region": "rural"}}, Q_CAR)
        with self.assertRaisesRegex(pi.MissingPersonField, "answers"):
            pi.get_injector("answer_neighbours", pool=people())({"id": "x", "fields": {"a": 1}}, Q_CAR)

    def test_only_target_answered_is_an_error(self):
        inj = pi.get_injector("answer_neighbours", pool=people())
        with self.assertRaisesRegex(pi.MissingPersonField, "other than the target"):
            inj({"id": "x", "answers": {"q0": "yes"}}, Q_CAR)
        with self.assertRaisesRegex(pi.MissingPersonField, "other than the target"):
            inj({"id": "x", "answers": {"q0": "yes", "q1": "no"}}, {**Q_CAR, "exclude": ["q1"]})


class TestLeakage(unittest.TestCase):
    def setUp(self):
        self.pool = people()
        # the target is in the pool with a unique answer to the target question and a unique excluded answer
        self.target = {"id": "u3", "fields": {"region": "rural", "age": 23},
                       "answers": {"q0": "yes", "q1": "no", "q2": 777, "q9": "SENTINEL-EXCL"}}
        self.pool[3] = self.target

    def test_target_never_neighbour_nor_shown(self):
        q = {**Q_NUM, "exclude": ["q9"]}
        for name, kw in [("similar_respondents", {"k": 11}), ("similar_respondents", {"k": 11, "format": "multi"}),
                         ("similar_respondents", {"k": 11, "strategy": "random"}),
                         ("neighbour_votes", {"k": 11, "basis": "both"}),
                         ("answer_neighbours", {"k": 11, "min_overlap": 1})]:
            p = pi.get_injector(name, pool=self.pool, **kw)(self.target, q)
            self.assertNotIn("u3", p.meta["neighbours"], name)
            self.assertEqual(len(p.meta["neighbours"]), 11, name)
            text = all_text(p)
            self.assertNotIn("777", text, name)                       # no one else answered 777 to q2
            self.assertNotIn("SENTINEL-EXCL", text, name)
            self.assertNotRegex(text, r"\bu\d+\b", name)              # neighbours are never shown by id

    def test_similarity_ignores_target_answer(self):
        """Changing only the target's own answer to the target question must not change the prompt."""
        inj = pi.get_injector("neighbour_votes", pool=self.pool, k=4, basis="answers")
        a = inj(self.target, Q_CAR)
        b = inj({**self.target, "answers": {**self.target["answers"], "q0": "no"}}, Q_CAR)
        self.assertEqual(all_text(a), all_text(b))
        self.assertEqual(a.meta["neighbours"], b.meta["neighbours"])

    def test_int_ids_match(self):
        pool = [{**p, "id": i} for i, p in enumerate(people())]
        p = pi.get_injector("similar_respondents", pool=pool, k=11)({**pool[0], "id": "0"}, Q_CAR)
        self.assertNotIn("0", p.meta["neighbours"])


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_prompt(self):
        for name, kw in [("similar_respondents", {"strategy": "random", "k": 4}), ("neighbour_votes", {"k": 4}),
                         ("answer_neighbours", {"k": 3})]:
            a = pi.get_injector(name, pool=people(), seed=7, **kw)(people()[0], Q_CAR)
            b = pi.get_injector(name, pool=people(), seed=7, **kw)(people()[0], Q_CAR)
            self.assertEqual(all_text(a), all_text(b))
            self.assertEqual(a.meta, b.meta)

    def test_seed_changes_random_strategy_and_ties(self):
        pool = people(40)
        picks = {tuple(pi.get_injector("similar_respondents", pool=pool, strategy="random", k=5, seed=s)
                       (pool[0], Q_CAR).meta["neighbours"]) for s in range(5)}
        self.assertGreater(len(picks), 1)
        # all-equal fields: every candidate ties, the seed alone sets the order
        flat = [{**p, "fields": {"region": "same"}} for p in pool]
        ties = {tuple(pi.get_injector("neighbour_votes", pool=flat, k=3, seed=s)(flat[0], Q_CAR).meta["neighbours"])
                for s in range(5)}
        self.assertGreater(len(ties), 1)

    def test_most_similar_order(self):
        p = pi.get_injector("similar_respondents", pool=people(), k=3)(people()[1], Q_CAR)
        self.assertEqual(p.meta["similarity"], sorted(p.meta["similarity"], reverse=True))
        self.assertIn("selected to be most similar to", p.user)
        least = pi.get_injector("similar_respondents", pool=people(), k=3, strategy="least_similar")(people()[1], Q_CAR)
        self.assertLess(max(least.meta["similarity"]), min(p.meta["similarity"]))


class TestRender(unittest.TestCase):
    def test_similar_respondents_single_and_multi(self):
        p = pi.get_injector("similar_respondents", pool=people(), k=3)(people()[1], Q_CAR)
        self.assertEqual(p.history, [])
        self.assertEqual(len(re.findall(r"^Participant \d+: ", p.user, re.M)), 3)
        self.assertIn("Here is the participant to be classified: region: rural; age: 21", p.user)
        self.assertTrue(p.system.startswith("Please perform a classification task"))
        self.assertIn("Do you own a car? -> yes", p.user)
        m = pi.get_injector("similar_respondents", pool=people(), k=3, format="multi")(people()[1], Q_CAR)
        self.assertEqual([r for r, _ in m.history], ["user", "assistant"] * 3)
        self.assertEqual(json.loads(m.history[1][1]), {"answer": "yes"})
        self.assertTrue(m.user.startswith("Here is the participant to be classified"))

    def test_stratify_shows_every_option(self):
        p = pi.get_injector("similar_respondents", pool=people(), k=4, stratify=True)(people()[1], Q_CAR)
        self.assertEqual(p.user.count("-> yes"), 2)
        self.assertEqual(p.user.count("-> no"), 2)
        plain = pi.get_injector("similar_respondents", pool=people(), k=4)(people()[1], Q_CAR)
        self.assertEqual(plain.user.count("-> no"), 0)                 # nearest rural people all own cars

    def test_votes_counts_share_numbers(self):
        p = pi.get_injector("neighbour_votes", pool=people(), k=5)(people()[1], Q_CAR)
        self.assertIn("answers to the next question were: yes 5, no 0.", p.system)
        s = pi.get_injector("neighbour_votes", pool=people(), k=4, show="share", keys=["age"])(people()[1], Q_CAR)
        self.assertRegex(s.system, r"yes \d+%, no \d+% \(n=4\)")
        self.assertNotIn("region", s.system)
        n = pi.get_injector("neighbour_votes", pool=people(), k=3, keys=["age"])(people()[5], Q_NUM)
        self.assertIn("(median ", n.system)
        self.assertIn('{"answer": <number>}', n.user)

    def test_answer_neighbours_shared_items(self):
        p = pi.get_injector("answer_neighbours", pool=people(), k=2, questions=QUESTIONS)(people()[1], Q_CAR)
        self.assertIn("- Do you take the bus?: no", p.system)
        self.assertIn("agreement 92%", p.system)                     # bus match + trips 2 vs 1 (range 6)
        self.assertIn("this question: yes", p.system)
        self.assertNotIn("- Do you own a car?", p.system)
        none = pi.get_injector("answer_neighbours", pool=people(), min_overlap=5)(people()[1], Q_CAR)
        self.assertIn("No other participant", none.system)
        self.assertEqual(none.meta["neighbours"], [])

    def test_no_candidate_answered(self):
        pool = [{**p, "answers": {"q1": "yes"}} for p in people()]
        p = pi.get_injector("similar_respondents", pool=pool)(people()[1], Q_CAR)
        self.assertIn("No other participant answered", p.user)
        v = pi.get_injector("neighbour_votes", pool=pool)(people()[1], Q_CAR)
        self.assertIn("none of them answered", v.system)


def votes_oracle(system, user):
    """Follows the majority in a neighbour_votes prompt; otherwise 'no'."""
    m = re.search(r"were: yes (\d+), no (\d+)", system)
    a = "yes" if m and int(m.group(1)) > int(m.group(2)) else "no"
    return json.dumps({"answer": a})


class TestCompare(unittest.TestCase):
    def test_votes_beat_no_persona(self):
        ppl = people()
        llm = FakeLLM(votes_oracle)
        res = pi.compare_injectors(llm, ppl, [Q_CAR, Q_BUS], "answers",
                                   [("neighbour_votes", {"pool": ppl, "k": 3})], B=200)
        t = res.table.set_index("arm")
        arm = [a for a in res.arms if a.startswith("neighbour_votes")][0]
        self.assertAlmostEqual(t.loc[arm, "score"], 1.0)
        self.assertAlmostEqual(t.loc["no_persona", "score"], 0.5)
        self.assertGreater(t.loc[arm, "lo_vs_no_persona"], 0)
        self.assertIn(f"swap_{arm}", res.arms)
        self.assertEqual(len(llm.calls), 3 * 12 * 2)


if __name__ == "__main__":
    unittest.main()
