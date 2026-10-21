import collections
import csv
import json
import os
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

import simengine.persona as sp
import simengine.persona.methods as pm
from simengine.persona.methods import big5, construction, demographic, gps, homo_silicus, value_anchor
from simengine.persona.methods.helpers import (RowFragmentRenderer, TraitDialSampler, quota_counts, quota_mixture,
                                               stratified_pool_sample, weighted_indices_no_replacement,
                                               weighted_rows_no_replacement)

DATA_FREE = ["baseline", "homo_silicus", "big5", "value_anchor", "gps", "construction", "bfi2"]
DATA_BACKED = ["demographic", "anthology", "twin2k"]


class TestRegistry(unittest.TestCase):
    def test_methods_listed(self):
        names = [m["name"] for m in pm.list_methods()]
        for n in DATA_FREE + DATA_BACKED:
            self.assertIn(n, names)
        flags = {m["name"]: m["needs_data"] for m in pm.list_methods()}
        self.assertTrue(all(flags[n] for n in DATA_BACKED))
        self.assertFalse(any(flags[n] for n in DATA_FREE))
        self.assertIsInstance(pm.get_method("gps"), pm.PersonaMethod)
        with self.assertRaises(ValueError):
            pm.get_method("nope")

    def test_exports_from_persona(self):
        self.assertIs(sp.sample_texts, pm.sample_texts)
        self.assertIs(sp.AgentBank, pm.AgentBank)

    def test_register_custom(self):
        m = homo_silicus.make({"reciprocal": 1.0}, name="hs_reciprocal_test")
        pm.register_method(m)
        try:
            texts = pm.sample_texts("hs_reciprocal_test", 3, seed=0)
            self.assertTrue(all("reciprocity" in t for t in texts))
            with self.assertRaises(ValueError):
                pm.register_method(m)
        finally:
            pm.METHODS.pop("hs_reciprocal_test", None)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_list_other_seed_differs(self):
        for name in DATA_FREE:
            a = pm.sample_texts(name, 12, seed=3)
            self.assertEqual(a, pm.sample_texts(name, 12, seed=3), name)
            self.assertEqual(len(a), 12)
            if name != "baseline":
                self.assertNotEqual(a, pm.sample_texts(name, 12, seed=4), name)
        self.assertEqual(pm.sample_texts("baseline", 3), ["", "", ""])

    def test_method_personas_shape(self):
        people = pm.method_personas("value_anchor", 5, seed=7)
        self.assertEqual([p["id"] for p in people], [f"value_anchor-7-{k}" for k in range(5)])
        for p in people:
            self.assertEqual(p["persona_method"], "value_anchor")
            self.assertIn(p["params"]["value"], value_anchor.ANCHORS)
            self.assertEqual(p["params"]["seed"], 7)
            self.assertTrue(p["text"].startswith("Answer as a person that is "))
        from simengine.sim.personas import as_personas
        views = as_personas(people)
        self.assertEqual(views[0].pid, "value_anchor-7-0")
        self.assertIn(people[0]["text"], views[0].text())

    def test_params_carry_dials(self):
        p = pm.method_personas("construction", 3, seed=0)
        self.assertEqual({(x["params"]["efficiency"], x["params"]["self_interest"], x["params"]["inequity_aversion"])
                          for x in p}, set(construction.PAPER_DIALS))
        g = pm.method_personas("gps", 2, seed=0)[0]
        self.assertEqual(set(g["params"]["answers"]), set(gps.PREFS))
        self.assertIn(f"your answer: {g['params']['answers']['risk']}", g["text"])
        b = pm.method_personas("big5", 1, seed=0)[0]
        for t in big5.TRAITS:
            self.assertIn(f"Your {t} score is {b['params']['dials'][t]} out of 10.", b["text"])


class TestMixtures(unittest.TestCase):
    def test_homo_silicus_composition_fixed(self):
        comps = {seed: collections.Counter(p["params"]["type"] for p in pm.method_personas("homo_silicus", 50, seed))
                 for seed in (0, 1, 2, 99)}
        first = comps[0]
        self.assertTrue(all(c == first for c in comps.values()))
        self.assertEqual(first, collections.Counter({"self_interested": 18, "efficient": 25, "selfish_but_fair": 7}))
        orders = {tuple(pm.sample_texts("homo_silicus", 10, s)) for s in range(5)}
        self.assertGreater(len(orders), 1)

    def test_quota_largest_remainder(self):
        c = quota_counts({"a": 0.53, "b": 0.37, "c": 0.10}, 4)
        self.assertEqual(sum(c.values()), 4)
        self.assertEqual(c, {"a": 2, "b": 2, "c": 0})
        pool = quota_mixture(["x", "y"], [1, 3], 8, seed=0)
        self.assertEqual(collections.Counter(pool), {"x": 2, "y": 6})
        with self.assertRaises(ValueError):
            quota_counts({"a": 0}, 3)

    def test_value_anchor_frequencies(self):
        people = pm.method_personas("value_anchor", 10000, seed=11)
        freq = collections.Counter(p["params"]["value"] for p in people)
        total = sum(value_anchor.PREVALENCE_WEIGHTS.values())
        for v, w in value_anchor.PREVALENCE_WEIGHTS.items():
            self.assertAlmostEqual(freq[v] / 10000, w / total, delta=0.02, msg=v)

    def test_construction_balanced(self):
        c = collections.Counter(pm.sample_texts("construction", 9, seed=5))
        self.assertEqual(sorted(c.values()), [3, 3, 3])


class TestTraitDialSampler(unittest.TestCase):
    def test_correlation_recovery(self):
        s = TraitDialSampler({t: (0.0, 1.0) for t in big5.TRAITS}, big5.CORR)
        draws = s.sample_dials(5000, seed=1)
        x = np.array([[d[t] for t in big5.TRAITS] for d in draws])
        r = np.corrcoef(x, rowvar=False)
        self.assertLess(np.abs(r - np.array(big5.CORR)).max(), 0.05)
        self.assertLess(np.abs(x.mean(0)).max(), 0.05)

    def test_nested_dict_corr_and_render(self):
        s = TraitDialSampler({"a": (5, 1.5), "b": (5, 1.5)}, {"a": {"b": 0.6}}, template="a={a:.0f} b={b:.0f}",
                             clip=(0, 10))
        draws = s.sample_dials(5000, seed=2)
        x = np.array([[d["a"], d["b"]] for d in draws])
        self.assertAlmostEqual(np.corrcoef(x, rowvar=False)[0, 1], 0.6, delta=0.05)
        self.assertTrue((x[:, 0] >= 0).all() and (x[:, 0] <= 10).all())
        self.assertTrue(s.sample(1, seed=0)[0].startswith("a="))

    def test_not_positive_definite(self):
        with self.assertRaises(ValueError):
            TraitDialSampler({"a": (0, 1), "b": (0, 1)}, [[1, 1.2], [1.2, 1]])

    def test_gps_correlation_recovery(self):
        draws = gps.SAMPLER.sample_dials(1, seed=0)
        self.assertTrue(all(0 <= v <= 10 and isinstance(v, int) for v in draws[0].values()))
        z = np.array([gps.SAMPLER.z(r) for r in [random.Random(3)] for _ in range(5000)])
        self.assertLess(np.abs(np.corrcoef(z, rowvar=False) - np.array(gps.R)).max(), 0.05)


class TestRows(unittest.TestCase):
    def test_weighted_no_replacement(self):
        rows = [{"id": i, "w": w} for i, w in enumerate([1, 2, 0, 3, 4])]
        got = weighted_rows_no_replacement(rows, "w", 4, seed=0)
        ids = [r["id"] for r in got]
        self.assertEqual(len(set(ids)), 4)
        self.assertNotIn(2, ids)
        more = weighted_indices_no_replacement([r["w"] for r in rows], 7, seed=1)   # > eligible: repeats, no hang
        self.assertEqual(len(more), 7)
        self.assertEqual(set(more[:4]), {0, 1, 3, 4})
        self.assertNotIn(2, more)
        first = collections.Counter(weighted_indices_no_replacement([1, 3], 1, seed=s)[0] for s in range(4000))
        self.assertAlmostEqual(first[1] / 4000, 0.75, delta=0.03)
        with self.assertRaises(ValueError):
            weighted_indices_no_replacement([0, 0], 1)
        with self.assertRaises(ValueError):
            weighted_indices_no_replacement([1, -1], 1)

    def test_fragment_renderer(self):
        r = RowFragmentRenderer([("sex", {"male": "I am male."}), ("age", "I am {} years old."),
                                 lambda row: "I vote." if row.get("vote") == "yes" else None],
                                frame='"{backstory}" Be this person.')
        self.assertEqual(r.render({"sex": "male", "age": "40", "vote": "yes"}),
                         '"I am male. I am 40 years old. I vote." Be this person.')
        self.assertEqual(r.fragments({"sex": "other", "age": "", "vote": "no"}), [])

    def test_stratified_pool(self):
        pool = [{"bracket": b, "k": i} for i, b in enumerate(["y", "y", "o"])]
        got = stratified_pool_sample(pool, {"y": 1.0, "o": 0.0}, 5, seed=0)
        self.assertTrue(all(x["bracket"] == "y" for x in got))
        self.assertEqual({x["k"] for x in got[:2]}, {0, 1})
        with self.assertRaises(ValueError):
            stratified_pool_sample(pool, {"y": 1.0, "missing": 1.0}, 2, seed=0)


class TestAgentBank(unittest.TestCase):
    def test_growth_keeps_ids_and_texts(self):
        bank = pm.AgentBank("homo_silicus")
        first = bank.get(4)
        self.assertEqual(list(first), [f"homo_silicus-{i:03d}" for i in range(4)])
        grown = bank.get(10)
        self.assertEqual(len(grown), 10)
        for k, v in first.items():
            self.assertEqual(grown[k], v)
        self.assertEqual(bank.get(3), dict(list(first.items())[:3]))
        self.assertEqual(bank.agents[4]["seed"], 1004)
        self.assertEqual(pm.AgentBank("homo_silicus").get(4), first)

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gps.json")
            a = pm.AgentBank("gps", path=path).get(3)
            b = pm.AgentBank("gps", path=path)
            self.assertEqual(len(b), 3)
            self.assertEqual(b.get(5) and dict(list(b.get(5).items())[:3]), a)
            with self.assertRaises(ValueError):
                pm.AgentBank("big5", path=path)
            legacy = os.path.join(d, "legacy.json")
            with open(legacy, "w") as f:
                json.dump({"big5-000": "old text"}, f)
            bank = pm.AgentBank("big5", path=legacy)
            self.assertEqual(bank.get(2)["big5-000"], "old text")
            self.assertEqual(bank.agents[1]["seed"], 1001)


def _write_gss(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=demographic.COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in demographic.COLUMNS})


class TestDataBacked(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "persona_methods"))

    def tearDown(self):
        self.tmp.cleanup()

    def _p(self, name):
        return os.path.join(self.root, "persona_methods", name)

    def test_missing_data_raises_with_layout(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SIMENGINE_DATA", None)
            for name in DATA_BACKED:
                with self.assertRaises(pm.PersonaDataNotFound) as cm:
                    pm.sample_texts(name, 1)
                self.assertIn("persona_methods", str(cm.exception))
                with self.assertRaises(FileNotFoundError):
                    pm.sample_texts(name, 1, data_root=self.root)

    def test_demographic_synthetic(self):
        _write_gss(self._p("gss_sample.csv"), [
            {"id": "1", "age": "33", "sex": "male", "race": "black", "hispanic": "not hispanic", "born": "yes",
             "degree": "bachelor's", "childs": "2", "income16": "$110,000 to $129,999", "partyid": "strong democrat",
             "region": "northeast", "weight": "2.0"},
            {"id": "2", "age": "89+", "sex": "female", "race": "white", "hispanic": "mexican", "born": "no",
             "childs": "0", "polviews": "conservative", "weight": "1.0"},
            {"id": "3", "sex": "female", "weight": "0"},
        ])
        people = pm.method_personas("demographic", 2, seed=0, data_root=self.root)
        self.assertEqual({p["params"]["row_id"] for p in people}, {"1", "2"})
        by = {p["params"]["row_id"]: p["text"] for p in people}
        self.assertIn('"Racially, I am Black. I am male. I am 33 years old. I live in the Northeast of the United '
                      'States. I have 2 children. I have a bachelor\'s degree. My total family income is $110,000 '
                      'to $129,999 a year. Politically, I am a strong Democrat."', by["1"])
        self.assertTrue(by["1"].startswith("In a survey, you described yourself in these words:"))
        self.assertIn("Racially, I am Hispanic. I was not born in the United States. I am female. "
                      "I am 89 years old or older.", by["2"])
        self.assertIn("I have no children.", by["2"])
        self.assertNotIn("married", by["2"])
        # also works with data_root = the persona_methods folder and via $SIMENGINE_DATA
        self.assertEqual(pm.sample_texts("demographic", 2, 0, data_root=os.path.join(self.root, "persona_methods")),
                         [p["text"] for p in people])
        with mock.patch.dict(os.environ, {"SIMENGINE_DATA": self.root}):
            self.assertEqual(pm.sample_texts("demographic", 2, 0), [p["text"] for p in people])

    def test_anthology_synthetic(self):
        stories = [{"bracket": b, "question": "Tell me about yourself.", "text": f"story {i}", "sid": i}
                   for i, b in enumerate(["18-29", "30-49", "50-64", "65+"] * 3)]
        with open(self._p("anthology_backstories.json"), "w") as f:
            json.dump(stories, f)
        people = pm.method_personas("anthology", 20, seed=1, data_root=self.root)
        self.assertEqual(len(people), 20)
        self.assertTrue(all("Question: Tell me about yourself." in p["text"] for p in people))
        self.assertTrue(all(p["params"]["bracket"] in ("18-29", "30-49", "50-64", "65+") for p in people))
        self.assertEqual(pm.sample_texts("anthology", 20, 1, data_root=self.root), [p["text"] for p in people])

    def test_twin2k_prefix_stable(self):
        with open(self._p("twin2k_personas_sample.json"), "w") as f:
            json.dump({"personas": [{"pid": str(i), "summary": f"person {i}"} for i in range(30)]}, f)
        long = pm.sample_texts("twin2k", 20, seed=5, data_root=self.root)
        self.assertEqual(pm.sample_texts("twin2k", 7, seed=5, data_root=self.root), long[:7])
        self.assertEqual(len(set(long)), 20)
        self.assertTrue(long[0].startswith("## Persona Profile"))
        self.assertEqual(len(pm.sample_texts("twin2k", 40, seed=5, data_root=self.root)), 40)


@unittest.skipUnless(os.environ.get("SIMENGINE_DATA"), "SIMENGINE_DATA not set")
class TestRealData(unittest.TestCase):
    def test_sample_each(self):
        for name in DATA_BACKED:
            try:
                texts = pm.sample_texts(name, 5, seed=1000)
            except pm.PersonaDataNotFound:
                continue
            self.assertEqual(len(texts), 5)
            self.assertTrue(all(texts))


if __name__ == "__main__":
    unittest.main()
