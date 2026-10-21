import json
import unittest

from simengine.persona import (Audience, AudienceSpec, Persona, dump_personas, enum_for, load_personas)

TALP_JSON = {
    "identity": {"persona_id": "p1", "name": "Synthetic One", "gender": "female", "age_band": "25-34",
                 "locale": "tr-TR", "tagline": "night-shift nurse", "demographic_traits": {"region": "north"},
                 "unknown_identity_key": 7},
    "traits": [{"label": "income_tier", "value": "middle", "group": "socioeconomic", "weight": 0.3}],
    "authored": {"persona": "B" * 900, "history_data": "H" * 900,
                 "triggers": {"frustrations": [{"text": "late delivery", "root_trait": "time"}]}},
    "behavioral": {"motivations": ["save time"]},
    "quantitative_dispositions": [{"dimension": "price_sensitivity", "value": 0.7}],
    "evidence": {"measured_behavior": {"facets": {"orders_2021": 12}}},
    "talp_top_level_unknown": {"a": 1},
}


class TestPersonaSchema(unittest.TestCase):
    def test_roundtrip_keeps_unknown_keys(self):
        p = Persona.from_dict(TALP_JSON)
        self.assertEqual(p.pid, "p1")
        self.assertEqual(p.trait("income_tier"), "middle")
        self.assertEqual(p.trait("missing", "x"), "x")
        self.assertEqual(p.authored.triggers.frustrations[0].text, "late delivery")
        self.assertEqual(p.quantitative_dispositions[0].value, 0.7)
        d = p.to_dict()
        self.assertEqual(d["identity"]["unknown_identity_key"], 7)
        self.assertEqual(d["traits"][0]["weight"], 0.3)
        self.assertEqual(d["talp_top_level_unknown"], {"a": 1})
        self.assertNotIn("extra", d)
        self.assertEqual(Persona.from_dict(json.loads(p.to_json())).to_dict(), d)

    def test_to_dict_drops_empty(self):
        d = Persona.from_dict({"identity": {"persona_id": "x"}}).to_dict()
        self.assertNotIn("authored", d)          # only empty sub-objects inside
        self.assertNotIn("traits", d)
        self.assertEqual(d["identity"]["persona_id"], "x")

    def test_load_personas_forms(self):
        flat = load_personas([{"persona_id": "a", "age_band": "18-24"}])
        self.assertEqual(flat[0].identity.age_band, "18-24")
        wrapped = load_personas(json.dumps({"personas": [TALP_JSON, TALP_JSON]}))
        self.assertEqual(len(wrapped), 2)
        self.assertEqual(load_personas(dump_personas(wrapped))[1].to_dict(), wrapped[1].to_dict())
        self.assertEqual(load_personas(TALP_JSON)[0].pid, "p1")

    def test_text_clips_narrative_not_data(self):
        p = Persona.from_dict(TALP_JSON)
        t = p.text()
        self.assertIn("B" * 600, t)
        self.assertNotIn("B" * 601, t)
        self.assertIn("H" * 900, t)                 # data-derived history is never clipped
        self.assertIn("Measured: orders_2021=12", t)
        self.assertTrue(t.startswith("Synthetic One, age 25-34, female"))
        self.assertIn("B" * 900, p.text(clip=None))
        self.assertEqual(p.metin(), t)               # research-code alias

    def test_card(self):
        c = Persona.from_dict(TALP_JSON).card()
        self.assertEqual(c["id"], "p1")
        self.assertEqual(c["evidence"], {"orders_2021": 12})
        self.assertEqual(c["traits"][0], {"label": "income_tier", "value": "middle"})

    def test_validate(self):
        spec = AudienceSpec(target_description="People who always order online", target_size=5,
                            persona_mode="b2x",
                            audience=Audience(income_tier=["lower"], identity_text="", behavioral_anchor="b"))
        w = " | ".join(spec.validate())
        self.assertIn("'lower' is not in the enum", w)
        self.assertIn("target_size 5", w)
        self.assertIn("persona_mode 'b2x'", w)
        self.assertIn("absolute wording (always)", w)
        self.assertIn("identity_text is required", w)
        self.assertNotIn("behavioral_anchor", w)
        ok = AudienceSpec(target_description="Young parents", audience=Audience(
            income_tier=["middle"], identity_text="parents", behavioral_anchor="weekly shop"))
        self.assertEqual(ok.validate(), [])
        self.assertEqual(AudienceSpec(locale="tr-TR").language(), "tr")
        self.assertEqual(ok.language(), "en")

    def test_spec_roundtrip_and_enum_for(self):
        spec = AudienceSpec.from_dict({"target_description": "t", "audience": {"age_range": ["65+"], "x": 1},
                                       "custom": True})
        self.assertEqual(spec.audience.age_range, ["65+"])
        self.assertEqual(spec.to_dict()["audience"]["x"], 1)
        self.assertTrue(spec.to_dict()["custom"])
        self.assertIn("65+", enum_for("age_band"))
        self.assertIn("female", enum_for("gender"))
        self.assertIsNone(enum_for("nope"))


if __name__ == "__main__":
    unittest.main()
