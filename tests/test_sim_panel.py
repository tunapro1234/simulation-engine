"""simengine.sim persona adapter, control arms and batched panel questioning."""
import json
import unittest

from simengine.llm import FakeLLM
from simengine.sim import (PersonaView, as_personas, ask_panel, build_prompts, control_personas, control_plan,
                           schema_responder)
from simengine.sim.panel import Usage
from simengine.sim.personas import call_count, choose_lang, normalize_controls

TALP = {
    "identity": {"persona_id": "t1", "name": "Persona T", "age_band": "25-34", "gender": "female", "locale": "en-US",
                 "tagline": "careful saver"},
    "traits": [{"label": f"trait{i}", "value": i} for i in range(15)],
    "authored": {"persona": "x" * 800, "buying_context": "buys groceries weekly"},
    "evidence": {"measured_behavior": {"facets": {"segment": "S1", "orders": 12}}},
}


class Obj:
    segment = "S2"

    def to_dict(self):
        return {"id": "o1", "name": "Persona O", "text": "Retired teacher."}


class Txt:
    pid = "x9"

    def text(self):
        return "A plain object persona."


class TestPersonas(unittest.TestCase):
    def test_talp_dict(self):
        (p,) = as_personas([TALP])
        self.assertEqual((p.pid, p.name, p.segment, p.locale), ("t1", "Persona T", "S1", "en-US"))
        t = p.text()
        self.assertTrue(t.startswith("Persona T, age 25-34, female"))
        self.assertIn("trait11: 11", t)
        self.assertNotIn("trait12", t)              # traits cut at 12
        self.assertNotIn("x" * 601, t)              # authored text cut at 600
        self.assertIn("Measured: segment=S1; orders=12", t)

    def test_flat_object_string_forms(self):
        ps = as_personas([{"pid": "f1", "age": 30, "city": "Izmir", "segment": "A"}, Obj(), Txt(), "Just text"])
        self.assertEqual([p.pid for p in ps], ["f1", "o1", "x9", "p004"])
        self.assertIn("age: 30; city: Izmir", ps[0].text())
        self.assertNotIn("segment", ps[0].text())
        self.assertEqual(ps[1].segment, "S2")
        self.assertIn("A plain object persona.", ps[2].text())
        self.assertEqual(ps[3].seg, "all")

    def test_duplicate_ids(self):
        with self.assertRaises(ValueError):
            as_personas([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}])

    def test_lang(self):
        ps = as_personas([{"id": "a", "text": "x", "locale": "tr-TR"}, {"id": "b", "text": "y"}])
        self.assertEqual(choose_lang(None, ps), "tr")
        self.assertEqual(choose_lang("en", ps), "en")


class TestControls(unittest.TestCase):
    def setUp(self):
        self.ps = [PersonaView(pid=f"p{i}", name=f"N{i}", attrs=[f"age {20 + i}"], body=[f"body {i}"],
                               tagline=f"tag{i}") for i in range(4)]

    def test_swap_keeps_own_identity_takes_next_content(self):
        sw = control_personas("persona_swap", self.ps)
        self.assertEqual([p.pid for p in sw], ["p0", "p1", "p2", "p3"])
        self.assertEqual(sw[0].name, "N0")
        self.assertEqual(sw[0].tagline, "tag0")
        self.assertEqual(sw[0].body, ["body 1"])
        self.assertEqual(sw[3].body, ["body 0"])    # cyclic

    def test_no_persona_neutral(self):
        nc = control_personas("personasiz", self.ps, lang="tr")
        self.assertTrue(all(p.text() == "Tipik bir yetişkin tüketici." for p in nc))
        self.assertEqual(control_personas("persona_swap", self.ps[:1]), [])

    def test_plan_within_main_budget(self):
        self.assertEqual(call_count(25, 10), 3)
        self.assertEqual(call_count(25, 10, splits=2), 4)
        plan = control_plan(40, 10, None)
        self.assertEqual(plan["_main_calls"], 4)
        self.assertEqual(plan["no_persona"]["batch"], 20)
        self.assertLessEqual(plan["no_persona"]["calls"] + plan["persona_swap"]["calls"], plan["_main_calls"])
        self.assertGreaterEqual(plan["persona_swap"]["n"], 1)
        with self.assertRaises(ValueError):
            normalize_controls(["bogus"])
        self.assertEqual(normalize_controls([]), [])


class TestAskPanel(unittest.TestCase):
    def setUp(self):
        self.ps = as_personas([{"id": f"u{i}", "text": f"person {i}"} for i in range(5)])
        self.fields = {"score": "int 1-10", "reason": "str"}

    def test_batches_and_fake_answers(self):
        llm = FakeLLM(schema_responder(0))
        plan = build_prompts(self.ps, "Rate it.", self.fields, batch=2)
        self.assertEqual([ids for _, ids in plan], [["u0", "u1"], ["u2", "u3"], ["u4"]])
        u = Usage()
        out = ask_panel(llm, self.ps, "Rate it.", self.fields, batch=2, usage=u)
        self.assertEqual(sorted(out), [f"u{i}" for i in range(5)])
        self.assertTrue(all(1 <= v["score"] <= 10 for v in out.values()))
        self.assertEqual((u.calls, u.missing), (3, 0))
        again = ask_panel(FakeLLM(schema_responder(0)), self.ps, "Rate it.", self.fields, batch=2)
        self.assertEqual(out, again)                # deterministic

    def test_mapping_reply_wrapper_and_missing(self):
        def responder(system, user):
            if "### u0" in user:
                return json.dumps({"u0": {"score": 3}, "u1": {"score": 4}})
            if "### u2" in user:
                return "```json\n{\"people\": [{\"id\": \"u2\", \"score\": 5}]}\n```"
            return "Sorry, I can't."
        u = Usage()
        out = ask_panel(FakeLLM(responder), self.ps, "t", self.fields, batch=2, usage=u)
        self.assertEqual({k: v["score"] for k, v in out.items()}, {"u0": 3, "u1": 4, "u2": 5})
        self.assertEqual((u.missing, u.unparsed_calls), (2, 1))

    def test_positional_remap_only_when_no_id_matches(self):
        def responder(system, user):
            return json.dumps([{"id": "P001", "score": 1}, {"id": "P002", "score": 2}])
        u = Usage()
        out = ask_panel(FakeLLM(responder), self.ps[:2], "t", self.fields, batch=2, usage=u)
        self.assertEqual(out["u0"]["score"], 1)
        self.assertEqual(out["u1"]["id"], "u1")
        self.assertEqual(u.remapped, 2)


if __name__ == "__main__":
    unittest.main()
