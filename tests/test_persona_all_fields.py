"""Every persona field reaches the prompt by default; explicit caps warn (rehearsal G2: 11 of 21 Twin fields cut)."""
import unittest
import warnings

from simengine.persona import Persona
from simengine.sim.personas import _from_talp


def _talp(n_traits=15, n_facets=21):
    return {"identity": {"persona_id": "p1", "age_band": "25-34", "gender": "female"},
            "traits": [{"label": f"t{i}", "value": f"v{i}"} for i in range(n_traits)],
            "authored": {"consumer_profile": "x" * 900, "history_data": "h" * 900, "persona": "p" * 900},
            "evidence": {"measured_behavior": {"facets": {f"f{i}": i for i in range(n_facets)}}}}


class AllFields(unittest.TestCase):
    def test_view_renders_every_trait_and_facet(self):
        text = _from_talp(_talp()).text()
        self.assertIn("t14: v14", text)
        self.assertIn("f20=20", text)

    def test_view_keeps_data_text_uncut_and_clips_llm_text(self):
        body = _from_talp(_talp()).body
        self.assertIn("x" * 900, body)          # consumer_profile: data-derived, uncut
        self.assertIn("h" * 900, body)
        self.assertIn("p" * 600, body)
        self.assertNotIn("p" * 601, body)

    def test_view_cap_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            text = _from_talp(_talp(), max_facets=10).text()
        self.assertNotIn("f10=10", text)
        self.assertTrue(any("21 facets" in str(x.message) for x in w))

    def test_persona_text_default_all(self):
        p = Persona.from_dict(_talp())
        self.assertIn("f20=20", p.text())
        self.assertIn("t14: v14", p.text())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertNotIn("f5=5", p.text(max_facets=5))
        self.assertTrue(w)

    def test_no_warning_when_within_cap(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _from_talp(_talp(3, 3), max_traits=12, max_facets=10).text()


if __name__ == "__main__":
    unittest.main()
