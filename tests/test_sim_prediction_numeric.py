"""sim.predict_people(kind="numeric"): several numeric fields with their own ranges in one JSON reply per person,
asked in one llm.map batch; out-of-range / unparsed values become NaN and are counted per field. FakeLLM and
synthetic people only."""
import json
import math
import re
import unittest

import numpy as np

from simengine.llm import FakeLLM
from simengine.sim import FieldSpec, prediction_frame
from simengine.sim.prediction import NumericField, numeric_fields, predict_people, prompt_numeric

# five targets with different ranges (log scale, 0-10, 0-32, 0-200, share)
FIELDS = [NumericField("log_rate", -5, 1, label="Log10 of the weekly rate"),
          NumericField("score", 0, 10),
          NumericField("days", 0, 32, unit="days"),
          NumericField("spend", 0, 200, unit="USD"),
          NumericField("share", 0, 1)]
PEOPLE = {"p1": "Age 30, lives in a city.", "p2": "Age 61, retired.", "p3": "Age 45, two children."}


def field_names(user):
    return re.findall(r"^(\w+): .*\(number from ", user, re.M)


def mid_responder(system, user):
    """Every requested field at the middle of its range, read back from the prompt."""
    out = {}
    for name, lo, hi in re.findall(r"^(\w+): .*\(number from (-?[\d.]+) to (-?[\d.]+)", user, re.M):
        out[name] = (float(lo) + float(hi)) / 2
    return json.dumps(out)


class MapLLM(FakeLLM):
    """FakeLLM that records the batches and keywords of each map() call."""

    def __init__(self, responder):
        super().__init__(responder)
        self.batches = []

    def map(self, prompts, *, max_tokens=256, parallel=4, **kw):
        self.batches.append((len(prompts), kw.get("return_exceptions")))
        return super().map(prompts, max_tokens=max_tokens, parallel=parallel, **kw)


class TestNumericFields(unittest.TestCase):
    def test_one_call_per_person_all_fields_one_map_batch(self):
        llm = MapLLM(mid_responder)
        r = predict_people(llm, PEOPLE, kind="numeric", fields=FIELDS)
        self.assertEqual(len(llm.calls), 3)                       # one reply per person, all five fields
        self.assertEqual(llm.batches, [(3, True)])                # parallel batch, error slots
        for _, user in llm.calls:
            self.assertEqual(field_names(user), [f.name for f in FIELDS])
            self.assertIn("(number from -5 to 1)", user)
            self.assertIn("(number from 0 to 200 USD)", user)
            self.assertIn("Log10 of the weekly rate", user)
        self.assertEqual(r["kind"], "numeric")
        self.assertEqual(r["items"], [f.name for f in FIELDS])
        self.assertEqual(r["p"]["p1"], {"log_rate": -2.0, "score": 5.0, "days": 16.0, "spend": 100.0, "share": 0.5})
        self.assertEqual(r["invalid"]["spend"], {"ok": 3, "out_of_range": 0, "unparsed": 0, "no_reply": 0})
        self.assertEqual(r["n_invalid"], 0)
        self.assertEqual(r["fields"][0], {"name": "log_rate", "lo": -5.0, "hi": 1.0, "unit": "",
                                          "label": "Log10 of the weekly rate"})

    def test_out_of_range_and_unparsed_become_nan_counted_per_field(self):
        replies = {
            "p1": {"log_rate": -7, "score": "8", "days": 40, "spend": 150, "share": 0.2},       # 2 out of range
            "p2": {"log_rate": 0.5, "score": "about ten", "spend": 250, "share": True},        # days missing
            "p3": {"log_rate": "-1.5", "score": 10, "days": 0, "spend": 0, "share": 1.0},
        }

        def responder(system, user):
            pid = next(p for p, t in PEOPLE.items() if t in user)
            return "Here you go:\n```json\n" + json.dumps(replies[pid]) + "\n```"

        r = predict_people(FakeLLM(responder), PEOPLE, kind="numeric", fields=FIELDS)
        self.assertEqual(r["p"]["p1"], {"log_rate": None, "score": 8.0, "days": None, "spend": 150.0, "share": 0.2})
        self.assertEqual(r["p"]["p2"], {"log_rate": 0.5, "score": None, "days": None, "spend": None, "share": None})
        self.assertEqual(r["p"]["p3"], {"log_rate": -1.5, "score": 10.0, "days": 0.0, "spend": 0.0, "share": 1.0})
        inv = r["invalid"]
        self.assertEqual(inv["log_rate"], {"ok": 2, "out_of_range": 1, "unparsed": 0, "no_reply": 0})
        self.assertEqual(inv["score"], {"ok": 2, "out_of_range": 0, "unparsed": 1, "no_reply": 0})
        self.assertEqual(inv["days"], {"ok": 1, "out_of_range": 1, "unparsed": 1, "no_reply": 0})
        self.assertEqual(inv["spend"], {"ok": 2, "out_of_range": 1, "unparsed": 0, "no_reply": 0})
        self.assertEqual(inv["share"], {"ok": 2, "out_of_range": 0, "unparsed": 1, "no_reply": 0})
        self.assertEqual(r["n_invalid"], 6)
        self.assertEqual(r["failed"], [])
        # one row per person x field, NaN where invalid, range columns kept
        f = prediction_frame(r)
        self.assertEqual(len(f), 15)
        self.assertEqual(list(f.columns[:4]), ["unit", "item", "truth", "pred_L1"])
        row = f[(f.unit == "p1") & (f.item == "days")].iloc[0]
        self.assertTrue(math.isnan(row["pred_L1"]))
        self.assertEqual((row["lo"], row["hi"], row["status"]), (0.0, 32.0, "out_of_range"))
        self.assertEqual(int(f["pred_L1"].isna().sum()), 6)

    def test_unparsable_reply_retried_then_no_reply(self):
        def responder(system, user):
            if "Age 61" in user:
                return "Somewhere around the middle for all of them."
            return mid_responder(system, user)

        llm = FakeLLM(responder)
        r = predict_people(llm, PEOPLE, kind="numeric", fields=FIELDS, retries=1)
        self.assertEqual(len(llm.calls), 4)                       # 3 + one re-ask for p2
        self.assertEqual(r["failed"], ["p2"])
        self.assertEqual(r["retried"], 1)
        self.assertNotIn("p2", r["p"])
        self.assertEqual(r["invalid"]["score"], {"ok": 2, "out_of_range": 0, "unparsed": 0, "no_reply": 1})
        f = prediction_frame(r, units=list(PEOPLE))
        self.assertEqual(len(f), 15)
        self.assertEqual(set(f[f.unit == "p2"]["status"]), {"no_reply"})

    def test_error_slot_from_map_is_no_reply(self):
        from simengine.llm.errors import LLMError

        class ErrLLM(FakeLLM):
            def map(self, prompts, *, max_tokens=256, parallel=4, return_exceptions=False, **kw):
                out = super().map(prompts, max_tokens=max_tokens, parallel=parallel, **kw)
                return [LLMError("boom") if "Age 45" in u else o for (_, u), o in zip(prompts, out)]

        r = predict_people(ErrLLM(mid_responder), PEOPLE, kind="numeric", fields=FIELDS, retries=0)
        self.assertEqual(r["failed"], ["p3"])
        self.assertEqual(r["invalid"]["share"]["no_reply"], 1)
        self.assertEqual(r["p"]["p1"]["share"], 0.5)

    def test_samples_average_valid_values_only(self):
        def responder(system, user):
            d = json.loads(mid_responder(system, user))
            if "Attempt" not in user and FakeLLM_sample[0] % 2:
                d["spend"] = 999                                  # odd samples out of range
            FakeLLM_sample[0] += 1
            return json.dumps(d)

        FakeLLM_sample = [0]
        r = predict_people(FakeLLM(responder), {"p1": "Age 30."}, kind="numeric", fields=FIELDS, samples=2,
                           sample_kw=None)
        self.assertEqual(r["p"]["p1"]["spend"], 100.0)            # mean of the one valid sample
        self.assertEqual(r["invalid"]["spend"], {"ok": 1, "out_of_range": 1, "unparsed": 0, "no_reply": 0})
        self.assertEqual(len(r["samples"]), 10)

    def test_l0_and_l2_numeric(self):
        llm = FakeLLM(mid_responder)
        r0 = predict_people(llm, kind="numeric", fields=FIELDS, level="L0", l0_samples=2)
        self.assertEqual(list(r0["p"]), ["_all"])
        self.assertEqual(r0["p"]["_all"]["days"], 16.0)
        r2 = predict_people(FakeLLM(mid_responder), PEOPLE, kind="numeric", fields=FIELDS, level="L2",
                            histories={p: "spent 40 USD last month" for p in PEOPLE})
        self.assertEqual(len(r2["p"]), 3)

    def test_field_forms_and_errors(self):
        fs = numeric_fields([("a", 0, 10), {"name": "b", "lo": -5, "hi": 1, "unit": "log"},
                             FieldSpec("c", kind="int", min=0, max=32, label="Days"), NumericField("d", 0, 1)])
        self.assertEqual([(f.name, f.lo, f.hi) for f in fs], [("a", 0, 10), ("b", -5, 1), ("c", 0, 32), ("d", 0, 1)])
        self.assertEqual(fs[2].label, "Days")
        for bad in ([("a", 5, 1)], [("a", 0, float("inf"))], [("a", 0, 1), ("a", 0, 2)], [],
                    [FieldSpec("t", kind="text")]):
            with self.assertRaises(ValueError):
                numeric_fields(bad)
        with self.assertRaises(ValueError):
            predict_people(FakeLLM(mid_responder), PEOPLE, kind="numeric")             # no fields
        with self.assertRaises(ValueError):
            predict_people(FakeLLM(mid_responder), PEOPLE, kind="numeric", fields=FIELDS, level="SSR")
        with self.assertRaises(ValueError):
            predict_people(FakeLLM(mid_responder), PEOPLE, ["x"], kind="numeric", fields=FIELDS)
        with self.assertRaises(ValueError):
            predict_people(FakeLLM(mid_responder), PEOPLE, ["x"], kind="count")
        with self.assertRaises(ValueError):
            predict_people(FakeLLM(mid_responder), PEOPLE, ["x"], fields=FIELDS)       # fields need kind

    def test_string_values_read_whole_or_unparsed(self):
        # lenient number search read these as wrong in-range values and counted them ok
        fs = [NumericField("spend", 0, 2000, unit="USD"), NumericField("log_rate", -5, 1), NumericField("score", 0, 10)]
        cases = [({"spend": "1,250", "log_rate": "−0.5", "score": ".5"}, (1250.0, -0.5, 0.5)),
                 ({"spend": "120 USD", "log_rate": "-1.5e0", "score": "7,5"}, (120.0, -1.5, 7.5)),
                 ({"spend": "$120", "log_rate": "-.5", "score": " 8 "}, (120.0, -0.5, 8.0)),
                 ({"spend": [3, 4], "log_rate": {"low": -2, "high": 0}, "score": "5-10"}, (None, None, None)),
                 ({"spend": "45%", "log_rate": "about -2", "score": "7/10"}, (None, None, None))]
        for d, want in cases:
            def responder(system, user, d=d):
                return json.dumps(d)

            r = predict_people(FakeLLM(responder), {"p1": "Age 30."}, kind="numeric", fields=fs)
            self.assertEqual(tuple(r["p"]["p1"][f.name] for f in fs), want, d)
            if want[0] is None:
                self.assertEqual(r["invalid"]["score"]["unparsed"], 1)

    def test_l2_summary_history_lists_its_own_items(self):
        seen = []

        def responder(system, user):
            seen.append(user)
            return mid_responder(system, user)

        hist = {p: {"BOOKS": {"count": 3, "days": 2, "last": "2024-01-01", "count_recent": 1}} for p in PEOPLE}
        predict_people(FakeLLM(responder), PEOPLE, kind="numeric", fields=FIELDS, level="L2", histories=hist)
        self.assertIn("BOOKS: 3 items on 2 different days", seen[0])
        self.assertNotIn("spend: never purchased", seen[0])       # field names are not purchase items

    def test_case_insensitive_duplicate_fields(self):
        with self.assertRaises(ValueError):
            numeric_fields([("Spend", 0, 1), ("spend", 0, 2)])     # both would read the same reply key

    def test_prompt_numeric_and_probability_default_unchanged(self):
        sy, us = prompt_numeric("Age 30.", numeric_fields(FIELDS))
        self.assertIn("numeric estimates", sy)
        self.assertIn("exactly these 5 keys", us)
        self.assertNotIn("probability", us)
        llm = FakeLLM(lambda s, u: json.dumps({"A": 40}))
        r = predict_people(llm, PEOPLE, ["A"])
        self.assertEqual(r["kind"], "probability")
        self.assertAlmostEqual(r["p"]["p1"]["A"], 0.4)
        self.assertTrue(np.isfinite(prediction_frame(r)["pred_L1"]).all())


if __name__ == "__main__":
    unittest.main()
