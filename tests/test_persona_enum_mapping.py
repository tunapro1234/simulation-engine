"""Persona builders: out-of-enum TALP values are kept (facet + one warning per column), the id is not repeated in
`Persona.text()`, and `from_history` applies comma fields to the answers block too."""
import unittest
import warnings

import pandas as pd

from simengine.persona import Mapping, Persona, from_history, from_row, from_table


def table():
    return pd.DataFrame({"uid": ["u1", "u2", "u3", "u4"], "age": [25, 40, 70, 19],
                         "income_tier": ["High", "mid", "low", "high"], "orders": [3, 5, 1, 0],
                         "target": [1, 0, 1, 0]})


class TestOutOfEnum(unittest.TestCase):
    def test_kept_as_facet_with_one_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ps = from_table(table(), id_col="uid", segment="income_tier", forbidden=["target"])
        msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
        self.assertEqual(len(msgs), 1)                              # once per column, not per row
        m = msgs[0]
        self.assertIn("income_tier", m)
        self.assertIn("3 of 4 rows", m)
        self.assertIn("'high' x2", m)
        self.assertIn("'mid' x1", m)
        self.assertIn("low, lower_middle, middle, upper_middle, affluent", m)
        p1 = ps[0]
        self.assertIsNone(p1.trait("income_tier"))
        self.assertEqual(p1.facets["income_tier"], "high")
        self.assertIn("income_tier=high", p1.text())                 # still reaches the prompt
        self.assertEqual(p1.segment, "high")
        self.assertEqual(ps[2].trait("income_tier"), "low")          # in-enum value stays a trait
        self.assertNotIn("income_tier", ps[2].facets)
        self.assertNotIn("target", p1.facets)

    def test_off_enum_facet_survives_max_facets(self):
        df = table().assign(**{f"x{i}": i for i in range(12)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = from_table(df, id_col="uid", forbidden=["target"])[1]
        self.assertIn("income_tier=mid", p.text(max_facets=3))

    def test_in_enum_table_does_not_warn(self):
        df = table().assign(income_tier=["low", "middle", "affluent", "low"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from_table(df, id_col="uid", forbidden=["target"])
        self.assertEqual([x for x in w if "enum" in str(x.message)], [])

    def test_other_age_cuts_and_gender_codes_are_kept(self):
        """Pew/Twin bands ("18-29", "50-64") and gender codes were dropped without a word by auto_mapping."""
        df = pd.DataFrame({"uid": ["a", "b", "c", "d"], "age": ["18-29", "50-64", "25-34", "16"],
                           "sex": [1, "Prefer not to say", "F", None], "target": [0, 1, 0, 1]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ps = from_table(df, id_col="uid", forbidden=["target"])
        msgs = " ".join(str(x.message) for x in w)
        self.assertIn("age_band (column 'age'): 2 of 4 rows", msgs)
        self.assertIn("gender (column 'sex'): 2 of 4 rows", msgs)
        self.assertEqual(ps[0].facets["age_band"], "18-29")
        self.assertIn("age_band=18-29", ps[0].text())
        self.assertEqual(ps[2].identity.age_band, "25-34")
        self.assertNotIn("age_band", ps[3].facets)                   # a numeric age under 18 is still dropped
        self.assertEqual(ps[0].facets["gender"], 1)
        self.assertEqual(ps[1].facets["gender"], "Prefer not to say")
        self.assertEqual(ps[2].identity.gender, "female")
        self.assertNotIn("gender", ps[3].facets)

    def test_from_row_alone_warns(self):
        m = Mapping(id_col="uid", fields={"income_tier": "inc"})
        with self.assertWarns(UserWarning):
            p = from_row({"uid": "a", "inc": "rich"}, m)
        self.assertEqual(p.facets, {"income_tier": "rich"})
        self.assertIn("rich", p.text())


class TestNoDuplicatedId(unittest.TestCase):
    def test_text_has_no_id_line(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = from_table(table(), id_col="uid", forbidden=["target"])[0]
        t = p.text()
        self.assertEqual(p.pid, "u1")
        self.assertTrue(t.startswith("age 25-34"))                  # the exact age 25 is banded
        self.assertNotIn("u1", t)
        block = f"### {p.pid}\n{t}"
        self.assertEqual(block.count("u1"), 1)

    def test_real_name_is_kept(self):
        p = Persona.from_dict({"identity": {"persona_id": "p9", "name": "Deniz", "age_band": "18-24"}})
        self.assertTrue(p.text().startswith("Deniz, age 18-24"))
        self.assertEqual(Persona.from_dict({"identity": {"persona_id": "p9"}}).text(), "")


class TestHistoryIdsAndCommas(unittest.TestCase):
    def survey(self):
        return pd.DataFrame({"rid": ["R1", "R2"], "race": ["White,Asian", "Black"], "freq": ["often", "rarely"]})

    def test_id_out_of_profile_and_text(self):
        ps = from_history(self.survey(), id_col="rid", level="L1",
                          questions={"rid": "Respondent", "freq": "How often", "race": "Race"},
                          comma_fields=["race"])
        self.assertEqual([p.pid for p in ps], ["R1", "R2"])
        self.assertEqual(ps[0].identity.persona_id, "R1")
        t = ps[0].text()
        self.assertNotIn("R1", t)
        self.assertNotIn("Respondent", t)
        self.assertIn("- Race: White, Asian", t)

    def test_comma_fields_in_answers_and_mapping_path(self):
        answers = pd.DataFrame({"rid": ["R1", "R1"], "item": ["race", "q1"], "value": ["White,Asian", None]})
        m = Mapping(id_col="rid", fields={"income_tier": "freq"})
        with self.assertWarns(UserWarning):                          # 'often'/'rarely' are not income tiers
            ps = from_history(self.survey(), id_col="rid", level="L1", mapping=m, answers=answers,
                              questions={"race": "Race"}, comma_fields=["race"], none_fields=["q1"])
        cp = ps[0].authored.consumer_profile
        self.assertEqual(cp.count("White, Asian"), 2)                # table profile and known answers
        self.assertIn("- q1: None", cp)
        self.assertNotIn("White,Asian", ps[0].text().split("Measured:")[0])
        self.assertEqual(ps[0].facets["income_tier"], "often")
        self.assertNotIn("R1", ps[0].text())

    def test_comma_cols_still_works(self):
        ps = from_history(self.survey(), id_col="rid", level="L1", questions={"race": "Race"}, comma_cols=["race"])
        self.assertIn("- Race: White, Asian", ps[0].authored.consumer_profile)


if __name__ == "__main__":
    unittest.main()
