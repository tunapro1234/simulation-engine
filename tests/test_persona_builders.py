import collections
import unittest

import numpy as np
import pandas as pd

from simengine.persona import (FakeSource, Mapping, age_band, from_history, from_row, from_table,
                               purchase_history_text, summarize_purchases, survey_profile_text, to_frames)

AGE = {"18 - 24 years": "18-24", "25 - 34 years": "25-34", "65 and older": "65+"}


def survey(n=30, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "rid": [f"R{i:03d}" for i in range(n)],
        "age_txt": r.choice(list(AGE) + ["prefer not"], n),
        "sex": r.choice(["Female", "Male", "Other"], n),
        "income": r.choice(["Less than $25,000", "$50,000 - $74,999", "$150,000 or more"], n),
        "freq": r.choice(["Less than 5 times per month", "More than 10 times per month"], n),
        "life": r.choice(["Moved place of residence,Lost a job", None], n),
        "target_2022": r.integers(0, 20, n),
    })


INCOME = {"Less than $25,000": "low", "$50,000 - $74,999": "middle", "$150,000 or more": "affluent"}


def mapping():
    return Mapping(id_col="rid",
                   fields={"age_band": ("age_txt", AGE), "gender": ("sex", str.lower),
                           "income_tier": ("income", INCOME),
                           "usage_intensity": ("freq", lambda v: "heavy" if v.startswith("More") else "light"),
                           "household_composition": lambda row: "single_no_kids"},
                   facets={"stated_frequency": "freq", "life_changes": lambda r: r["life"]},
                   text={"buying_context": lambda r: f"orders {r['freq'].lower()}"},
                   segment="income_tier", forbidden=["target_2022"], source_name="synthetic/survey")


class TestFromTable(unittest.TestCase):
    def test_explicit_mapping(self):
        df = survey()
        ps = from_table(df, mapping())
        self.assertEqual(len(ps), len(df))
        p, row = ps[0], df.iloc[0]
        self.assertEqual(p.pid, row["rid"])
        self.assertEqual(p.identity.age_band, AGE.get(row["age_txt"]))
        self.assertEqual(p.trait("income_tier"), INCOME[row["income"]])
        self.assertEqual(p.trait("household_composition"), "single_no_kids")
        self.assertEqual(p.segment, p.trait("income_tier"))
        self.assertEqual(p.facets["stated_frequency"], row["freq"])
        self.assertEqual(p.evidence.measured_behavior["claims"]["stated_frequency"]["contract_id"], "synthetic/survey")
        self.assertTrue(p.authored.buying_context.startswith("orders "))
        self.assertNotIn("target_2022", p.to_json())
        none_life = [q for q, (_, r) in zip(ps, df.iterrows()) if r["life"] is None]
        self.assertTrue(all("life_changes" not in q.facets for q in none_life))
        self.assertTrue(any(q.identity.age_band is None for q in ps))    # "prefer not" -> dropped

    def test_leakage_guard(self):
        m = mapping()
        m.facets["oops"] = "target_2022"
        with self.assertRaises(ValueError):
            from_row(survey().iloc[0], m)

    def test_auto_mapping(self):
        df = pd.DataFrame({"user_id": [1, 2, 3], "age": [19, 44, 80], "gender": ["F", "male", "?"],
                           "income_tier": ["Middle", "low", "rich"], "spend": [10.5, np.nan, 3.0],
                           "bought": [1, 0, 1]})
        ps = from_table(df, id_col="user_id", forbidden=["bought"])
        self.assertEqual([p.identity.age_band for p in ps], ["18-24", "35-44", "65+"])
        self.assertEqual([p.identity.gender for p in ps], ["female", "male", None])
        self.assertEqual([p.trait("income_tier") for p in ps], ["middle", "low", None])
        self.assertEqual(ps[0].facets, {"spend": 10.5})
        self.assertEqual(ps[1].facets, {})
        self.assertIsInstance(ps[0].facets["spend"], float)
        self.assertTrue(all("bought" not in p.facets for p in ps))

    def test_sampling(self):
        df = survey(60)
        ps = from_table(df, mapping(), n=12, seed=1)
        self.assertEqual(len(ps), 12)
        self.assertEqual([p.pid for p in ps], [p.pid for p in from_table(df, mapping(), n=12, seed=1)])
        strat = from_table(df, mapping(), n=12, seed=1, stratify="income")
        share = df["income"].value_counts(normalize=True)
        got = collections.Counter(INCOME_INV[p.trait("income_tier")] for p in strat)
        for k, s in share.items():
            self.assertLessEqual(abs(got[k] - s * 12), 1.01)

    def test_age_band(self):
        self.assertEqual([age_band(x) for x in (17, 18, 24.9, 25, 64, 65, "70", None, "x", np.nan)],
                         [None, "18-24", "18-24", "25-34", "55-64", "65+", "65+", None, None, None])

    def test_narrate_keeps_facets(self):
        ps = from_table(survey(5), mapping())
        before = [dict(p.facets) for p in ps]
        out = FakeSourceNarr().narrate(ps)
        self.assertEqual([p.facets for p in out], before)
        self.assertTrue(all(p.extra.get("narrative") == "llm" for p in out))
        self.assertTrue(out[0].identity.name.startswith("(fake)"))

    def test_to_frames(self):
        ps = from_table(survey(4), mapping())
        ps[0].quantitative_dispositions = []
        d = ps[1].to_dict()
        d["quantitative_dispositions"] = [{"dimension": "Price Sensitivity", "value": "0.4"}]
        people, feats = to_frames([ps[0], d])
        self.assertEqual(list(people["id"]), [ps[0].pid, ps[1].pid])
        self.assertIn("seg_income_tier", people)
        self.assertIn("persona_text", people)
        self.assertEqual(feats.to_dict("records"), [{"id": ps[1].pid, "item": "q_price_sensitivity", "value": 0.4}])


INCOME_INV = {v: k for k, v in INCOME.items()}


class FakeSourceNarr:
    def narrate(self, ps):
        from simengine.persona import narrate
        return narrate(FakeSource(), ps, context="synthetic grocery panel")


def purchases():
    return pd.DataFrame({
        "rid": ["R000", "R000", "R000", "R001", "R002", "R002", "R002", "R003"],
        "cat": ["BOOK", "BOOK", "PEN", "BOOK", "PEN", "BOOK", "CUP", "CUP"],
        "day": ["2022-01-03", "2022-01-03", "2021-03-01", "2022-07-01", "2022-06-30", "2019-05-05",
                "2023-01-01", "2020-02-02"],
        "qty": [1, 2, None, 5, 1, 1, 1, 4],
    })


class TestHistory(unittest.TestCase):
    def test_summarize_uses_only_past(self):
        s = summarize_purchases(purchases(), id_col="rid", category_col="cat", date_col="day",
                                cutoff="2022-07-01", qty_col="qty", ids=["R000", "R001", "R002"])
        self.assertEqual(s.attrs["categories"], ["BOOK", "PEN", "CUP"])      # by distinct buyers, then name
        r = s.set_index(["rid", "category"])
        self.assertEqual(r.loc[("R000", "BOOK"), "items"], 2)
        self.assertEqual(r.loc[("R000", "BOOK"), "quantity"], 3.0)
        self.assertEqual(r.loc[("R000", "BOOK"), "days"], 1)
        self.assertEqual(r.loc[("R000", "BOOK"), "items_recent"], 2)
        self.assertEqual(r.loc[("R000", "PEN"), "items_recent"], 0)
        self.assertEqual(r.loc[("R001", "BOOK"), "items"], 0)               # bought ON the cutoff: excluded
        self.assertIsNone(r.loc[("R001", "BOOK"), "last_purchase"])
        self.assertEqual(r.loc[("R002", "CUP"), "items"], 0)                # future purchase excluded
        self.assertEqual(str(r.loc[("R002", "PEN"), "last_purchase"]), "2022-06-30")
        self.assertEqual(len(s), 9)
        txt = purchase_history_text(s[s.rid == "R000"], recent_label="July 2021 - June 2022")
        self.assertEqual(txt.splitlines()[0],
                         "BOOK: 2 items on 1 different days, last purchase 2022-01-03, 2 items in July 2021 - June 2022")
        self.assertEqual(txt.splitlines()[2], "CUP: never purchased")

    def test_survey_profile_text(self):
        row = pd.Series({"a": "x", "b": None, "life": None, "race": "White,Asian"})
        t = survey_profile_text(row, {"a": "Q a", "b": "Q b", "life": "Life changes", "race": "Race"},
                                none_cols=["life"], comma_cols=["race"])
        self.assertEqual(t.splitlines(), ["- Q a: x", "- Q b: (no answer)", "- Life changes: None",
                                          "- Race: White, Asian"])

    def test_levels(self):
        df = survey(4).drop(columns=["target_2022"])
        kw = dict(id_col="rid", purchases=purchases(), category_col="cat", date_col="day", cutoff="2022-07-01",
                  qty_col="qty", questions={"age_txt": "Age group", "freq": "How often"})
        l0 = from_history(df, level="L0", **kw)
        l1 = from_history(df, level="L1", **kw)
        l2 = from_history(df, level="L2", **kw)
        self.assertEqual([p.pid for p in l2], list(df["rid"]))
        self.assertEqual(len({p.text() for p in l0}), len(l0))          # differ only by id line
        self.assertEqual(len({p.text().split("\n", 1)[1] for p in l0}), 1)
        self.assertIsNone(l1[0].authored.history_data)
        self.assertIn("- Age group: ", l1[0].authored.consumer_profile)
        self.assertNotIn("income", l1[0].authored.consumer_profile.lower())
        h = l2[0].authored.history_data
        self.assertTrue(h.startswith("Purchase history in these categories, May 2019 - June 2022:"))
        self.assertIn("BOOK: 2 items on 1 different days", h)
        self.assertIn("2 items in July 2021 - June 2022", h)
        self.assertEqual(l2[0].extra["level"], "L2")
        self.assertEqual(l2[0].extra["history_cutoff"], "2022-07-01")
        self.assertIn(h, l2[0].text())
        self.assertIn("CUP: never purchased", l2[2].authored.history_data)    # R002's 2023 cup is after cutoff
        self.assertIn("CUP: 1 items on 1 different days, last purchase 2020-02-02", l2[3].authored.history_data)

    def test_mapping_and_answers(self):
        df = survey(3).drop(columns=["target_2022"])
        answers = pd.DataFrame({"rid": ["R000", "R000", "R001"], "item": ["Q1", "TARGET", "Q1"],
                                "value": [3, 5, 1]})
        ps = from_history(df, id_col="rid", level="L1", mapping=mapping(), answers=answers,
                          item_text={"Q1": "How risky are you (1-5)?"}, targets=["TARGET"],
                          questions={"freq": "How often"})
        self.assertEqual(ps[0].trait("income_tier"), INCOME[df.iloc[0]["income"]])
        cp = ps[0].authored.consumer_profile
        self.assertIn("- How risky are you (1-5)?: 3", cp)
        self.assertNotIn("TARGET", cp)
        self.assertNotIn("Known answers", ps[2].authored.consumer_profile)
        only_answers = from_history(id_col="rid", level="L1", answers=answers, targets=["TARGET"])
        self.assertEqual([p.pid for p in only_answers], ["R000", "R001"])
        self.assertIn("- Q1: 3", only_answers[0].authored.consumer_profile)

    def test_guards(self):
        df = survey(3)
        with self.assertRaises(ValueError):
            from_history(df, id_col="rid", level="L1", questions={"target_2022": "x"}, targets=["target_2022"])
        with self.assertRaises(ValueError):
            from_history(df, id_col="rid", level="L2", purchases=purchases(), category_col="cat", date_col="day")
        with self.assertRaises(ValueError):
            from_history(df, id_col="rid", level="L2")
        with self.assertRaises(ValueError):
            from_history(df, id_col="rid", level="L3")
        l1 = from_history(df, id_col="rid", level="L1", targets=["target_2022"])
        self.assertNotIn("target_2022", l1[0].authored.consumer_profile)


if __name__ == "__main__":
    unittest.main()
