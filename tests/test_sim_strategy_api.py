"""sim.strategy API: scenario formats ({id: text} mapping), TypeError on bad input, control arms, variants alias."""
import json
import unittest

from simengine.llm import FakeLLM
from simengine.sim import schema_responder, strategy


def personas(n=6):
    return [{"id": f"u{i}", "name": f"N{i}", "text": f"age {20 + i}, city {i}", "segment": "A" if i % 2 else "B"}
            for i in range(n)]


def prompts(llm):
    return "\n".join(u for _, u in llm.calls)


class TestScenarioFormats(unittest.TestCase):
    def test_mapping_text_reaches_prompt(self):
        llm = FakeLLM(schema_responder(1))
        r = strategy(personas(), {"price": "Cut price 10% on staples", "ship": "Free delivery over 200"},
                     llm=llm, B=20, controls=[])
        self.assertEqual([s["id"] for s in r["scenarios"]], ["price", "ship"])
        self.assertEqual(r["scenarios"][0]["text"], "Cut price 10% on staples")
        text = prompts(llm)
        self.assertIn("Cut price 10% on staples", text)
        self.assertIn("Free delivery over 200", text)
        self.assertEqual(sorted(r["ranking"]), ["price", "ship"])

    def test_mapping_of_dicts_keeps_key_as_id(self):
        r = strategy(personas(4), {"a": {"text": "one"}, "b": {"text": "two"}}, llm=FakeLLM(schema_responder()),
                     B=10, controls=[])
        self.assertEqual([s["id"] for s in r["scenarios"]], ["a", "b"])

    def test_auto_ids_for_strings(self):
        r = strategy(personas(4), ["x", "y"], llm=FakeLLM(schema_responder()), B=10, controls=[])
        self.assertEqual([s["id"] for s in r["scenarios"]], ["S1", "S2"])

    def test_bad_types_raise(self):
        llm = FakeLLM(schema_responder())
        for bad in ("just one string", 42, ["ok", 3]):
            with self.assertRaises(TypeError, msg=repr(bad)):
                strategy(personas(), bad, llm=llm)
        with self.assertRaises(ValueError):
            strategy(personas(), {"only": "one"}, llm=llm)


class TestControls(unittest.TestCase):
    def test_default_runs_control_arms(self):
        llm = FakeLLM(schema_responder(2))
        r = strategy(personas(), ["cut price", "free delivery"], llm=llm, B=20, truth={"S1": 1, "S2": 2})
        json.dumps(r)
        self.assertEqual(r["controls"]["arms"], ["no_persona", "persona_swap"])
        self.assertEqual(set(r["arms"]), {"persona", "no_persona", "persona_swap"})
        self.assertGreater(r["arms"]["no_persona"]["n"], 0)
        self.assertIn("typical adult consumer", prompts(llm))
        self.assertEqual(set(r["arms_direction_correct"]), set(r["arms"]))
        self.assertEqual(r["arms"]["persona"]["winner"], r["winner"])

    def test_no_controls_warns(self):
        llm = FakeLLM(schema_responder(2))
        r = strategy(personas(), ["cut price", "free delivery"], llm=llm, B=20, controls=[])
        self.assertEqual(set(r["arms"]), {"persona"})
        self.assertNotIn("typical adult consumer", prompts(llm))
        self.assertTrue(any("no persona-control arms" in w for w in r["warnings"]))

    def test_unknown_control(self):
        with self.assertRaises(ValueError):
            strategy(personas(), ["a", "b"], llm=FakeLLM(schema_responder()), controls=["bogus"])


class TestVariantsAlias(unittest.TestCase):
    def test_variants_alias(self):
        a = strategy(personas(), variants=["cut price", "free delivery"], llm=FakeLLM(schema_responder(3)), B=20)
        b = strategy(personas(), ["cut price", "free delivery"], llm=FakeLLM(schema_responder(3)), B=20)
        self.assertEqual(a["ranking"], b["ranking"])
        self.assertEqual(a["scenarios"], b["scenarios"])

    def test_both_or_neither_raise(self):
        llm = FakeLLM(schema_responder())
        with self.assertRaises(TypeError):
            strategy(personas(), ["a", "b"], variants=["c", "d"], llm=llm)
        with self.assertRaises(TypeError):
            strategy(personas(), llm=llm)


if __name__ == "__main__":
    unittest.main()
