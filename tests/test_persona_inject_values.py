import json
import unittest

import simengine.persona.inject as pi
from simengine.llm.fake import FakeLLM
from simengine.persona.inject import values as V

Q = {"id": "q1", "text": "Is divorce justifiable?", "options": ["never", "sometimes", "always"]}
QN = {"id": "q2", "text": "How many close friends do you have?", "min": 0, "max": 50}
SECRET = "zz-target-answer-zz"


def person(i=0):
    return {"id": f"u{i}", "country": "Atlantis" if i % 2 else "Lemuria",
            "schwartz": {"benevolence": 5.0, "power": 2.0 + i % 2 * 3.5, "self_direction": 4.5},
            "mfq": {"care": 4.0, "fairness": 3.5, "loyalty": 1.0 + i % 2 * 3.5, "authority": 2.0, "sanctity": 1.0},
            "hofstede": {"pdi": 60, "idv": 30},
            "ivs_map": {"traditional_secular": -0.5, "survival_self_expression": 1.2},
            "answers": {"q1": SECRET, "q2": SECRET}}


ALL = [("schwartz_scores", {}), ("schwartz_scores", {"definitions": False, "order": "shuffle", "seed": 2}),
       ("schwartz_anchor", {}), ("schwartz_anchor", {"k": 3}), ("moral_foundations", {}),
       ("moral_foundations", {"style": "scores"}), ("hofstede_country", {}), ("hofstede_country", {"scores": True}),
       ("cultural_prompting", {}), ("cultural_prompting", {"variant": "random", "map": True})]


class TestRegistry(unittest.TestCase):
    def test_listed_with_arxiv(self):
        rows = {r["name"]: r for r in pi.list_injectors("values")}
        self.assertEqual(set(rows), {"schwartz_scores", "schwartz_anchor", "moral_foundations", "hofstede_country",
                                     "cultural_prompting"})
        for r in rows.values():
            self.assertIn("arXiv:", r["paper"])
            self.assertTrue(r["needs"] and r["description"])
        self.assertNotIn("values", pi.IMPORT_ERRORS)

    def test_labels(self):
        self.assertEqual(pi.get_injector("schwartz_anchor", k=2).label, "schwartz_anchor(k=2)")
        self.assertEqual(pi.get_injector("cultural_prompting").label, "cultural_prompting")

    def test_bad_params(self):
        for name, kw in [("schwartz_scores", {"order": "alpha"}), ("schwartz_scores", {"scale": (6, 1)}),
                         ("schwartz_anchor", {"k": 0}), ("moral_foundations", {"style": "vibes"}),
                         ("cultural_prompting", {"variant": 10})]:
            with self.assertRaises(ValueError, msg=(name, kw)):
                pi.get_injector(name, **kw)


class TestRender(unittest.TestCase):
    def test_schwartz_scores_template(self):
        p = pi.get_injector("schwartz_scores")(person(), Q)
        self.assertTrue(p.system.startswith("Let's roleplay."))
        self.assertIn("between 1 and 6", p.system)
        self.assertIn("- Self-Direction: 4.5", p.system)
        self.assertIn("values independent thought", p.system)
        self.assertIn("Please choose only one option", p.system)
        self.assertIn('"answer"', p.user)
        # paper order (alphabetical) by default, rank on request
        self.assertLess(p.system.index("- Benevolence: 5"), p.system.index("- Power: 2"))
        short = pi.get_injector("schwartz_scores", definitions=False, order="rank")(person(1), QN)
        self.assertNotIn("definitions", short.system)
        self.assertNotIn("choose only one option", short.system)       # numeric question
        self.assertEqual(list(short.meta["values"]), ["power", "benevolence", "self-direction"])

    def test_schwartz_anchor(self):
        p = pi.get_injector("schwartz_anchor")(person(), Q)
        self.assertEqual(p.system, f"Answer as a person that is {V.BASIC_ANCHORS['benevolence']}.")
        self.assertEqual(p.meta["anchors"], ["benevolence"])
        refined = pi.get_injector("schwartz_anchor")({"id": "a", "schwartz": "universalism_nature"}, Q)
        self.assertEqual(refined.system, "Answer as a person that is protecting the natural environment from "
                                         "destruction or pollution.")
        two = pi.get_injector("schwartz_anchor", k=2)({"id": "a", "schwartz": ["hedonism", "tradition"]}, Q)
        self.assertIn(" and following cultural family or religious practices.", two.system)   # paper text wins
        with self.assertRaises(ValueError):
            pi.get_injector("schwartz_anchor")({"id": "a", "schwartz": "kindness"}, Q)

    def test_moral_foundations(self):
        p = pi.get_injector("moral_foundations")(person(), Q)
        self.assertEqual(p.system, "You do not like to cause harm.")
        p = pi.get_injector("moral_foundations")(person(1), Q)          # loyalty 4.5 is top
        self.assertEqual(p.system, "You would sacrifice yourself for your country.")
        aliased = pi.get_injector("moral_foundations", k=2)({"id": "a", "mfq": {"purity": 5, "harm": 4}}, Q)
        self.assertEqual(aliased.meta["foundations"], ["sanctity", "care"])
        with self.assertRaises(ValueError):                              # liberty has no published sentence
            pi.get_injector("moral_foundations")({"id": "a", "mfq": {"liberty": 4}}, Q)
        s = pi.get_injector("moral_foundations", style="scores")({"id": "a", "mfq": {"liberty": 4}}, Q)
        self.assertIn("- Liberty: 4", s.system)
        self.assertIn("from 0 (not at all relevant) to 5", s.system)

    def test_hofstede(self):
        p = pi.get_injector("hofstede_country")(person(), Q)
        self.assertEqual(p.system, "Answer the following prompt by imagining that you are proudly and thoroughly "
                                   "from the country of Lemuria.")
        adv = pi.get_injector("hofstede_country", advice=True)(person(), Q)
        self.assertTrue(adv.system.endswith(", providing appropriate advice."))
        sc = pi.get_injector("hofstede_country", scores=True)({"id": "a", "country": "X",
                                                               "hofstede": {"Power distance": 70, "ivr": 40}}, Q)
        self.assertIn("- Power distance: 70 (low:", sc.system)
        self.assertIn("- Indulgence: 40", sc.system)

    def test_cultural_prompting(self):
        p = pi.get_injector("cultural_prompting")(person(), Q)
        self.assertEqual(p.system, "You are an average human being born in Lemuria and living in Lemuria "
                                   "responding to the following survey question.")
        mig = pi.get_injector("cultural_prompting", variant=9)({"id": "a", "country": "X", "birth_country": "Y"}, Q)
        self.assertEqual(mig.system, "You are a world citizen born in Y and living in X responding to the "
                                     "following survey question.")
        m = pi.get_injector("cultural_prompting", map=True)(person(), Q)
        self.assertIn("traditional (negative) to secular-rational (positive): -0.5", m.system)
        self.assertIn("survival (negative) to self-expression (positive): 1.2", m.system)


class TestEdgeCases(unittest.TestCase):
    def test_missing_fields_list_needs(self):
        bare = {"id": "a"}
        for name, kw in ALL:
            inj = pi.get_injector(name, **kw)
            with self.assertRaises(pi.MissingPersonField, msg=inj.label) as cm:
                inj(bare, Q)
            for f in inj.needs:
                self.assertIn(repr(f), str(cm.exception))
        with self.assertRaises(pi.MissingPersonField) as cm:
            pi.get_injector("hofstede_country", scores=True)({"id": "a", "country": "X"}, Q)
        self.assertIn("'hofstede'", str(cm.exception))
        with self.assertRaises(pi.MissingPersonField):                   # all scores None
            pi.get_injector("schwartz_scores")({"id": "a", "schwartz": {"power": None}}, Q)

    def test_bad_scores(self):
        inj = pi.get_injector("schwartz_scores")
        for bad in ({"powr": 3}, {"power": 7}, {"power": "high"}, {"power": True}, {"power": float("nan")}):
            with self.assertRaises(ValueError, msg=bad):
                inj({"id": "a", "schwartz": bad}, Q)
        with self.assertRaises(ValueError):
            inj({"id": "a", "schwartz": [1, 2]}, Q)
        ok = pi.get_injector("schwartz_scores", scale=(-3, 3))({"id": "a", "schwartz": {"power": -1.25}}, Q)
        self.assertIn("- Power: -1.25", ok.system)

    def test_deterministic_and_seeded(self):
        for name, kw in ALL:
            a = pi.get_injector(name, **kw)(person(1), Q)
            b = pi.get_injector(name, **kw)(person(1), Q)
            self.assertEqual((a.system, a.user), (b.system, b.user), name)
        ppl = [{"id": f"p{i}", "country": "X"} for i in range(30)]
        v0 = [pi.get_injector("cultural_prompting", variant="random", seed=0)(p, Q).meta["variant"] for p in ppl]
        v1 = [pi.get_injector("cultural_prompting", variant="random", seed=1)(p, Q).meta["variant"] for p in ppl]
        self.assertNotEqual(v0, v1)
        self.assertGreater(len(set(v0)), 3)
        tie = {"id": "t", "schwartz": {c: 4 for c in V.SCHWARTZ_10}}
        tops = {pi.get_injector("schwartz_anchor", seed=s)(tie, Q).meta["anchors"][0] for s in range(20)}
        self.assertGreater(len(tops), 1)                                 # ties broken by seed, not dict order
        orders = {tuple(pi.get_injector("schwartz_scores", order="shuffle", seed=s)(tie, Q).meta["values"])
                  for s in range(5)}
        self.assertGreater(len(orders), 1)

    def test_no_target_answer_leakage(self):
        for name, kw in ALL:
            for q in (Q, QN):
                p = pi.get_injector(name, **kw)(person(1), q)
                self.assertNotIn(SECRET, p.system + p.flatten(), name)

    def test_swap_uses_donor_values(self):
        ppl = [person(0), person(1)]
        s = pi.swap_wrapper("moral_foundations", ppl, seed=0)
        self.assertEqual(s(ppl[0], Q).system, "You would sacrifice yourself for your country.")


class TestCompare(unittest.TestCase):
    def test_compare_with_fake_llm(self):
        # oracle: answers "always" only when the persona puts benevolence first by rank
        def oracle(system, user):
            return json.dumps({"answer": "always" if "caring for the welfare" in system else "never"})
        ppl = [person(i) for i in range(6)]
        truth = {(p["id"], "q1"): ("never" if i % 2 else "always") for i, p in enumerate(ppl)}
        llm = FakeLLM(oracle)
        res = pi.compare_injectors(llm, ppl, [Q], truth, ["schwartz_anchor"], B=200)
        t = res.table.set_index("arm")
        self.assertAlmostEqual(t.loc["schwartz_anchor", "score"], 1.0)
        self.assertAlmostEqual(t.loc["no_persona", "score"], 0.5)
        self.assertLess(t.loc["swap_schwartz_anchor", "score"], 1.0)


if __name__ == "__main__":
    unittest.main()
