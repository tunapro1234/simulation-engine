"""sim.predict_people inputs: Persona lists, {pid: text} cleanup, unlabeled-item warning, summarize_purchases
histories, and the llm= keyword form. Synthetic data only."""
import json
import re
import unittest
import warnings

import pandas as pd

from simengine.llm import FakeLLM
from simengine.persona import Persona, from_history, summarize_purchases
from simengine.sim import AMAZON, PredictionSpec, batch_calibrate, predict_people
from simengine.sim.prediction import histories_from_summary

ITEMS = ["ABIS_BOOK", "CLEANING_AGENT"]


def responder(system, user):
    block = re.search(r"\(key: description\):\n(.*?)\n\n", user, re.S).group(1)
    return json.dumps({ln.split(":", 1)[0]: 40 for ln in block.splitlines()})


def users(llm):
    return [u for _, u in llm.calls]


class TestPersonaPeople(unittest.TestCase):
    TABLE = pd.DataFrame({"uid": ["a1", "a2"], "Q-demos-age": ["25 - 34 years", "65 and older"],
                          "Q-demos-state": ["Ohio", "Texas"]})

    def test_from_history_l1_personas(self):
        ps = from_history(self.TABLE, id_col="uid", level="L1", questions=None)
        llm = FakeLLM(responder)
        r = predict_people(llm, ps, ITEMS, spec=AMAZON)
        self.assertEqual(r["failed"], [])
        self.assertEqual(sorted(r["p"]), ["a1", "a2"])
        us = users(llm)
        self.assertEqual(len(us), 2)
        joined = "\n".join(us)
        self.assertIn("25 - 34 years", joined)
        self.assertIn("Texas", joined)
        for u in us:
            lines = [ln.strip() for ln in u.splitlines()]
            self.assertNotIn("a1", lines)
            self.assertNotIn("a2", lines)
            self.assertNotIn("Survey answers of this person:", lines)   # profile_intro only, not repeated
            self.assertTrue(u.startswith(AMAZON.profile_intro + "\n- "))

    def test_persona_answers_dict(self):
        p = Persona.from_dict({"identity": {"persona_id": "z9", "name": "z9"}})
        p.extra["answers"] = {"Q-demos-age": "35 - 44 years", "Q-demos-race": "White,Asian"}
        llm = FakeLLM(responder)
        predict_people(llm, [p], ITEMS, spec=AMAZON)
        u = users(llm)[0]
        self.assertIn("- Age group: 35 - 44 years", u)
        self.assertIn("- Race: White, Asian", u)
        self.assertNotIn("z9", u)

    def test_narrative_persona_without_profile(self):
        p = Persona.from_dict({"identity": {"persona_id": "n1", "name": "Mara", "age_band": "30-39"},
                               "authored": {"persona": "Runs a small bakery.", "history_data": "BOOKS: 3"}})
        llm = FakeLLM(responder)
        predict_people(llm, [p], ITEMS)
        u = users(llm)[0]
        self.assertIn("Mara, age 30-39\nRuns a small bakery.", u)
        self.assertNotIn("BOOKS: 3", u)                      # history_data goes through histories=
        self.assertNotIn("n1", u)

    def test_text_dict_cleanup(self):
        llm = FakeLLM(responder)
        predict_people(llm, {"u7": "u7\nSurvey answers of this person:\n- Age: 40"}, ITEMS, spec=AMAZON)
        u = users(llm)[0]
        self.assertTrue(u.startswith(AMAZON.profile_intro + "\n- Age: 40\n\nTask:"))
        self.assertNotIn("u7", u)

    def test_unsupported_types(self):
        llm = FakeLLM(responder)
        with self.assertRaisesRegex(TypeError, "list\\[Persona\\]"):
            predict_people(llm, self.TABLE, ITEMS)
        with self.assertRaisesRegex(TypeError, "list\\[Persona\\]"):
            predict_people(llm, [1, 2], ITEMS)
        with self.assertRaisesRegex(TypeError, "list\\[Persona\\]"):
            predict_people(llm, {"u1": 5}, ITEMS)
        with self.assertRaisesRegex(TypeError, "list\\[Persona\\]"):
            predict_people(llm, "u1", ITEMS)
        self.assertEqual(llm.calls, [])

    def test_empty_profile_warns(self):
        ps = from_history(None, id_col="uid", level="L1", ids=["e1"])
        with self.assertWarnsRegex(UserWarning, "empty profile"):
            predict_people(FakeLLM(responder), ps, ITEMS)


class TestUnlabeledWarning(unittest.TestCase):
    def test_amazon_missing_label(self):
        with self.assertWarns(UserWarning) as cm:
            predict_people(FakeLLM(responder), {"u1": "- Age: 30"}, ["ABIS_BOOK", "PET_FOOD", "TOY"], spec=AMAZON)
        msgs = [str(w.message) for w in cm.warnings if "spec.labels" in str(w.message)]
        self.assertEqual(len(msgs), 1)
        self.assertIn("['PET_FOOD', 'TOY']", msgs[0])
        self.assertNotIn("ABIS_BOOK", msgs[0])

    def test_no_warning_when_covered_or_unlabeled_spec(self):
        for spec in (AMAZON, PredictionSpec()):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                predict_people(FakeLLM(responder), {"u1": "- Age: 30"}, ITEMS, spec=spec)
            self.assertFalse([x for x in w if "spec.labels" in str(x.message)])


class TestSummaryHistories(unittest.TestCase):
    def setUp(self):
        buys = pd.DataFrame({"uid": ["a1", "a1", "a1", "a2"],
                             "cat": ["ABIS_BOOK", "ABIS_BOOK", "CLEANING_AGENT", "CLEANING_AGENT"],
                             "day": ["2021-03-01", "2022-03-01", "2022-05-02", "2023-01-01"]})
        self.summary = summarize_purchases(buys, id_col="uid", category_col="cat", date_col="day",
                                           cutoff="2022-07-01", categories=ITEMS, ids=["a1", "a2"])

    def test_mapping(self):
        h = histories_from_summary(self.summary)
        self.assertEqual(h["a1"]["ABIS_BOOK"], {"count": 2, "days": 2, "last": "2022-03-01", "count_recent": 1})
        self.assertEqual(h["a2"]["CLEANING_AGENT"]["count"], 0)          # after the cutoff: not counted
        self.assertIsNone(h["a2"]["CLEANING_AGENT"]["last"])
        with self.assertRaisesRegex(ValueError, "id_col"):
            histories_from_summary(self.summary.assign(extra=1))

    def test_l2_accepts_frame(self):
        llm = FakeLLM(responder)
        r = predict_people(llm, {"a1": "- Age: 30", "a2": "- Age: 60"}, ITEMS, level="L2", spec=AMAZON,
                           histories=self.summary)
        self.assertEqual(r["failed"], [])
        u = dict(zip(["a1", "a2"], users(llm)))
        self.assertIn("ABIS_BOOK: 2 items on 2 different days, last purchase 2022-03-01, 1 items in", u["a1"])
        self.assertIn("CLEANING_AGENT: never purchased", u["a2"])


class TestLLMKeyword(unittest.TestCase):
    PEOPLE = {"u1": "- Age: 30", "u2": "- Age: 50"}

    def test_forms_identical(self):
        a, b, c = FakeLLM(responder), FakeLLM(responder), FakeLLM(responder)
        r1 = predict_people(a, self.PEOPLE, ITEMS, spec=AMAZON)
        r2 = predict_people(self.PEOPLE, ITEMS, llm=b, spec=AMAZON)
        r3 = predict_people(llm=c, people=self.PEOPLE, items=ITEMS, spec=AMAZON)
        self.assertEqual(r1["p"], r2["p"])
        self.assertEqual(r1["p"], r3["p"])
        self.assertEqual(a.calls, b.calls)
        self.assertEqual(a.calls, c.calls)

    def test_l0_keyword(self):
        r = predict_people(None, ITEMS, llm=FakeLLM(responder), level="L0", l0_samples=2)
        self.assertEqual(r["p"]["_all"]["ABIS_BOOK"], 0.4)

    def test_errors(self):
        llm = FakeLLM(responder)
        with self.assertRaisesRegex(TypeError, "both"):
            predict_people(llm, self.PEOPLE, ITEMS, llm=llm)
        with self.assertRaisesRegex(TypeError, "needs an llm"):
            predict_people()
        with self.assertRaisesRegex(TypeError, "multiple values"):
            predict_people(self.PEOPLE, ITEMS, llm=llm, people=self.PEOPLE)


class TestBatchCalibrateOverwrite(unittest.TestCase):
    def test_default_overwrites_copy_only(self):
        f = pd.DataFrame({"item": ["A", "A"], "p": [0.2, 0.6]})
        g = batch_calibrate(f, {"A": 0.5})
        self.assertEqual(list(f["p"]), [0.2, 0.6])             # input untouched
        self.assertNotEqual(list(g["p"]), [0.2, 0.6])          # col replaced in the copy
        h = batch_calibrate(f, {"A": 0.5}, out_col="p_cal")
        self.assertEqual(list(h["p"]), [0.2, 0.6])
        self.assertIn("p_cal", h)


if __name__ == "__main__":
    unittest.main()
