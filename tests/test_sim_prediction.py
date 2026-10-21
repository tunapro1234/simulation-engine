"""simengine.sim per-person prediction prompts (L0/L1/L2/SSR), batch calibration, retest statistics."""
import hashlib
import json
import re
import unittest

import numpy as np
import pandas as pd

from simengine.llm import FakeLLM
from simengine.sim import (AMAZON, SSR, PredictionSpec, batch_calibrate, history_text, predict_people,
                           prediction_frame, profile_text, prompt_l0, prompt_l1, prompt_l2, prompt_ssr,
                           retest_stats, schema_responder)

ITEMS = ["ABIS_BOOK", "PET_FOOD"]


def keys_in(user):
    block = re.search(r"\(key: description\):\n(.*?)\n\n", user, re.S).group(1)
    return [ln.split(":", 1)[0] for ln in block.splitlines()]


class TestPrompts(unittest.TestCase):
    def test_amazon_preset_wording(self):
        s, u = prompt_l0(ITEMS, AMAZON)
        self.assertTrue(s.startswith("You simulate the Amazon shopping behavior of US Amazon customers."))
        self.assertTrue(u.startswith("Task: consider a randomly selected US adult Amazon customer's account during "
                                     "July 1 - December 31, 2022."))
        self.assertIn("ABIS_BOOK: Books\nPET_FOOD: Pet food\n", u)
        self.assertTrue(u.endswith("Return a JSON object with exactly these 2 keys and integer values from 0 to 100."))
        prof = profile_text({"Q-demos-age": "25 - 34 years", "Q-demos-race": "White,Asian"}, AMAZON.questions,
                            none_fields=AMAZON.none_fields, comma_fields=AMAZON.comma_fields)
        self.assertIn("- Age group: 25 - 34 years", prof)
        self.assertIn("- Race: White, Asian", prof)
        self.assertIn("- Life changes in 2021 (self or account sharer): None", prof)
        self.assertIn("- State of residence: (no answer)", prof)
        self.assertEqual(len(prof.splitlines()), 22)
        s1, u1 = prompt_l1(prof, ITEMS, AMAZON)
        self.assertTrue(u1.startswith("Survey answers of this person about themselves and their Amazon account:\n- Age"))
        self.assertIn("consider this person's Amazon account (including anyone who shares it) during", u1)
        _, u2 = prompt_l2(prof, ITEMS, "ABIS_BOOK: never purchased", AMAZON)
        self.assertIn("Their Amazon purchase history in these categories, January 2018 - June 2022:\n"
                      "ABIS_BOOK: never purchased\n\nTask:", u2)
        ss, us = prompt_ssr(prof, ITEMS, AMAZON)
        self.assertIn("speak as this person about their Amazon account (including anyone who shares it) during", us)
        self.assertIn("one sentence (a string) as each value", us)

    def test_generic_spec_and_labels(self):
        spec = PredictionSpec(period="Q3 2025", labels={"GYM": "Gym visit"})
        _, u = prompt_l1("- Age: 30", ["GYM", "late_fee"], spec)
        self.assertIn("during Q3 2025", u)
        self.assertIn("GYM: Gym visit\nlate_fee: Late fee\n", u)

    def test_history_text(self):
        h = history_text({"A": {"count": 3, "days": 2, "last": "2022-05-01", "count_recent": 1}, "B": 0,
                          "C": "bought twice"}, ["A", "B", "C", "D"], recent_period="July 2021 - June 2022")
        self.assertEqual(h.splitlines(), [
            "A: 3 items on 2 different days, last purchase 2022-05-01, 1 items in July 2021 - June 2022",
            "B: never purchased", "C: bought twice", "D: never purchased"])


class TestPredict(unittest.TestCase):
    PEOPLE = {"u1": {"age": "30"}, "u2": "- age: 50", "u3": {"age": "70"}}

    def test_l1_samples_and_frame(self):
        def responder(system, user):
            h = int(hashlib.md5(user.encode()).hexdigest(), 16)
            return json.dumps({k: (h >> (8 * j)) % 101 for j, k in enumerate(keys_in(user))})
        llm = FakeLLM(responder)
        r = predict_people(llm, self.PEOPLE, ITEMS, level="L1", samples=2)
        json.dumps(r)
        self.assertEqual(len(llm.calls), 6)
        self.assertEqual(sorted(r["p"]), ["u1", "u2", "u3"])
        self.assertEqual(len(r["samples"]), 3 * 2 * 2)
        self.assertTrue(all(0 <= v <= 1 for d in r["p"].values() for v in d.values()))
        f = prediction_frame(r, truth={"u1": {"ABIS_BOOK": 1, "PET_FOOD": 0}}, segments={"u1": "young"})
        self.assertEqual(len(f), 6)
        self.assertIn("pred_L1", f.columns)
        self.assertEqual(f.loc[(f.unit == "u1") & (f.item == "ABIS_BOOK"), "truth"].item(), 1.0)
        self.assertEqual(f.loc[f.unit == "u1", "segment"].iloc[0], "young")

    def test_retry_then_fail(self):
        def responder(system, user):
            if "age: 50" in user:
                return "not json"
            if "age\": \"70" in user or "age: 70" in user:
                return json.dumps({"ABIS_BOOK": 10}) if "Attempt" not in user else json.dumps(
                    {"ABIS_BOOK": 10, "PET_FOOD": "20%"})
            return json.dumps({"abis_book": 40, "pet_food": 5})
        llm = FakeLLM(responder)
        r = predict_people(llm, self.PEOPLE, ITEMS, level="L1", retries=1)
        self.assertEqual(r["failed"], ["u2"])
        self.assertAlmostEqual(r["p"]["u3"]["PET_FOOD"], 0.2)
        self.assertEqual(r["retried"], 2)
        self.assertEqual(len(llm.calls), 5)
        self.assertTrue(any("(Attempt 2: reply with the JSON object only.)" in u for _, u in llm.calls))

    def test_l0_broadcast(self):
        llm = FakeLLM(schema_responder(0))
        r = predict_people(llm, None, ITEMS, level="L0", l0_samples=4)
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(list(r["p"]), ["_all"])
        f = prediction_frame(r, units=["a", "b"])
        self.assertEqual(len(f), 4)
        self.assertEqual(f.pred_L0.nunique(), 2)

    def test_l2_needs_histories(self):
        llm = FakeLLM(schema_responder(0))
        with self.assertRaises(ValueError):
            predict_people(llm, self.PEOPLE, ITEMS, level="L2")
        with self.assertRaises(ValueError):
            predict_people(llm, self.PEOPLE, ITEMS, level="L2", histories={"u1": "x"})
        hist = {"u1": {"ABIS_BOOK": {"count": 2}}, "u2": "ABIS_BOOK: never purchased", "u3": {}}
        r = predict_people(llm, self.PEOPLE, ITEMS, level="L2", histories=hist)
        self.assertEqual(len(r["p"]), 3)
        self.assertTrue(any("ABIS_BOOK: 2 items on 2 different days" in u for _, u in llm.calls))

    def test_ssr_level(self):
        def encoder(texts):
            return np.array([[1.0, 0.0] if "probably bought" in t else [0.0, 1.0] for t in texts])
        spec = PredictionSpec(ssr_anchors=["I probably did not buy anything like this.",
                                           "I probably bought something like this."])
        llm = FakeLLM(schema_responder(0))
        r = predict_people(llm, {"u1": "x", "u2": "y"}, ITEMS, level="SSR", spec=spec, ssr=SSR(encoder=encoder))
        for pid, d in r["p"].items():
            for k, v in d.items():
                want = 1.0 if "probably bought" in r["texts"][pid]["0"][k] else 0.0
                self.assertAlmostEqual(v, want)

    def test_bad_args(self):
        llm = FakeLLM(schema_responder(0))
        with self.assertRaises(ValueError):
            predict_people(llm, self.PEOPLE, [], level="L1")
        with self.assertRaises(ValueError):
            predict_people(llm, self.PEOPLE, ITEMS, level="L9")


class TestCalibrateRetest(unittest.TestCase):
    def test_batch_calibrate(self):
        f = pd.DataFrame({"unit": list("aabbcc"), "item": ["X", "Y"] * 3, "p": [0.9, 0.1, 0.8, 0.2, 0.7, 0.0]})
        g = batch_calibrate(f, {"X": 0.2, "Y": 0.5})
        logit = lambda x: np.log(x / (1 - x))  # noqa: E731
        x = g[g.item == "X"].p.to_numpy()
        self.assertAlmostEqual(float(np.mean(logit(x))), float(logit(0.2)), places=9)
        self.assertTrue(np.all(np.diff(x) < 0))            # order within an item kept
        self.assertTrue(np.isfinite(g.p).all())            # p = 0 clipped
        self.assertEqual(f.p.iloc[0], 0.9)                 # input untouched
        h = batch_calibrate(f, {"X": 0.2, "Y": 0.5}, out_col="p_cal")
        self.assertIn("p_cal", h.columns)
        with self.assertRaises(ValueError):
            batch_calibrate(f, {"X": 0.2})

    def test_retest(self):
        rows = []
        for u, base in (("a", 0.2), ("b", 0.5), ("c", 0.9)):
            for s, eps in enumerate((0.0, 0.02, -0.02)):
                rows += [{"unit": u, "item": "X", "sample": s, "p": base + eps},
                         {"unit": u, "item": "Y", "sample": s, "p": 1 - base}]
        st = retest_stats(rows)
        self.assertEqual((st["n_units"], st["k"], st["n_cells"]), (3, 3, 6))
        self.assertAlmostEqual(st["identical_share"], 0.5)
        self.assertGreater(st["retest_r"], 0.9)
        self.assertGreater(st["spearman_brown"], st["item_retest_r"] - 1e-12)
        self.assertEqual(retest_stats([])["k"], 0)


if __name__ == "__main__":
    unittest.main()
