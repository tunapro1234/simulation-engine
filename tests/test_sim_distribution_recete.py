"""Distribution runners after the recipe rehearsals: segment_distribution repeats (sample= per repeat), filled
share per cell and the filled_<method> frame column, persona_distribution refusal accounting and segment
schemes, person_distribution_frame, retest_stats(kind="distribution")."""
import json
import re
import unittest

import numpy as np

from simengine.llm import FakeLLM
from simengine.sim import distribution_frame, persona_distribution, retest_stats, segment_distribution
from simengine.sim.distribution import person_distribution_frame
from simengine.types import LLMResponse

QS = [{"id": "q1", "text": "Do you approve?", "options": ["Approve", "Neutral", "Disapprove"]},
      {"id": "q2", "text": "Own a car?", "options": ["Yes", "No"]}]
SEGS = {"young": "adults aged 18 to 34", "old": "adults aged 65 or older"}
REFUSAL = "I'm sorry, but I can't help with that request."


class SampleFake(FakeLLM):
    """FakeLLM whose reply depends on `sample`, with a disk-cache stand-in keyed on (user, sample)."""

    def __init__(self):
        super().__init__()
        self.samples: list = []
        self.cache: dict = {}

    def complete(self, system, user, *, max_tokens=256, **kw):
        self.samples.append(kw.get("sample", "-"))
        key = (user, kw.get("sample", 0))
        if key not in self.cache:
            s = int(kw.get("sample", 0))
            k = len(re.findall(r"^\d+\. ", user, re.M))
            pct = [10 + 10 * s if j == 0 else 0 for j in range(k)]
            pct[-1] = 100 - pct[0]
            self.cache[key] = json.dumps({str(j + 1): v for j, v in enumerate(pct)})
        self.calls.append((system, user))
        return LLMResponse(text=self.cache[key], model="fake", input_tokens=1, output_tokens=1)


class TestSegmentRepeats(unittest.TestCase):
    def test_plain_call_has_no_sample(self):
        llm = SampleFake()
        r = segment_distribution(llm, SEGS, QS[:1])
        self.assertEqual(set(llm.samples), {"-"})
        self.assertEqual(r["repeats"], 1)
        self.assertEqual(len(r["sample_dist"]), 1)
        self.assertEqual(r["sample_dist"][0], r["dist"])

    def test_repeats_are_independent_and_averaged(self):
        llm = SampleFake()
        r = segment_distribution(llm, SEGS, QS, repeats=3)
        json.dumps(r)
        self.assertEqual(r["calls"], 3 * 2 * 2 * 3)
        self.assertEqual(len(llm.cache), 3 * 2 * 2 * 3)       # no repeat served from another's cache entry
        self.assertEqual(sorted(set(llm.samples)), [0, 1, 2])
        per = [d["young"]["q1"] for d in r["sample_dist"]]
        np.testing.assert_allclose(per, [[0.1, 0, 0.9], [0.2, 0, 0.8], [0.3, 0, 0.7]])
        np.testing.assert_allclose(r["dist"]["young"]["q1"], [0.2, 0, 0.8])
        self.assertEqual({c["sample"] for c in r["cells"]}, {0, 1, 2})
        self.assertEqual(len(r["samples"]), 3 * 2 * 2)
        st = retest_stats(r, kind="distribution")
        self.assertEqual(st["k"], 3)
        self.assertEqual(st["n_cells"], 4)
        self.assertEqual(st["reduced_to"], "E[position]")
        self.assertGreater(st["within_cell_w1"], 0)
        self.assertAlmostEqual(st["identical_share"], 0.0)

    def test_sample_offset(self):
        llm = SampleFake()
        segment_distribution(llm, SEGS, QS[:1], repeats=2, ask_kw={"sample": 5})
        self.assertEqual(sorted(set(llm.samples)), [5, 6])
        llm = SampleFake()
        segment_distribution(llm, SEGS, QS[:1], ask_kw={"sample": 4})
        self.assertEqual(set(llm.samples), {4})


class TestFilled(unittest.TestCase):
    @staticmethod
    def responder(system, user):
        if "Consider adults aged 18" in user:            # h2 framing for one segment: unusable reply
            return "No idea."
        return json.dumps({"1": 60, "2": 30, "3": 10}) if "3. " in user else json.dumps({"1": 80, "2": 20})

    def test_filled_share_and_frame_column(self):
        r = segment_distribution(FakeLLM(self.responder), SEGS, QS)
        self.assertEqual(r["invalid"], 2)
        self.assertAlmostEqual(r["filled"]["young"]["q1"], 1 / 3)
        self.assertAlmostEqual(r["filled"]["old"]["q2"], 0.0)
        self.assertAlmostEqual(r["filled_share"], 2 / 12)
        self.assertEqual(sum(c["filled"] for c in r["cells"]), 2)
        f = distribution_frame(r, method="c3")
        self.assertIn("filled_c3", f.columns)
        self.assertAlmostEqual(f.loc[f.unit == "young|q1", "filled_c3"].item(), 1 / 3)
        self.assertAlmostEqual(f.loc[f.unit == "old|q1", "filled_c3"].item(), 0.0)

    def test_sample_rows_flag_filler(self):
        r = segment_distribution(FakeLLM(self.responder), SEGS, QS, repeats=2)
        row = [x for x in r["samples"] if x["unit"] == "young" and x["item"] == "q1" and x["sample"] == 1][0]
        self.assertAlmostEqual(row["filled"], 1 / 3)
        self.assertEqual([x["filled"] for x in r["samples"] if x["unit"] == "old"], [0.0] * 4)
        st = retest_stats(r, kind="distribution")
        self.assertAlmostEqual(st["filled_share"], (2 * 2 / 3) / 8)

    def test_retest_without_kind_says_so(self):
        r = segment_distribution(FakeLLM(self.responder), SEGS, QS, repeats=2)
        with self.assertRaisesRegex(ValueError, "kind='distribution'"):
            retest_stats(r)

    def test_skip_fills_nothing(self):
        r = segment_distribution(FakeLLM(self.responder), SEGS, QS, fallback="skip")
        self.assertEqual(r["invalid"], 2)
        self.assertAlmostEqual(r["filled_share"], 0.0)

    def test_legacy_result_gives_nan(self):
        f = distribution_frame({"dist": {"a": {"q1": [0.5, 0.5]}}})
        self.assertTrue(np.isnan(f["filled_llm_dist"].item()))


class TestPersonaAccounting(unittest.TestCase):
    PS = [{"id": f"u{i}", "text": f"person {i}", "segment": "A" if i < 3 else "B",
           "age_band": ["18-34", "35-54", None][i % 3], "hh": 1 + i % 2} for i in range(5)]

    @staticmethod
    def responder(system, user):
        ids = re.search(r"respondent IDs as keys \(([^)]*)\)", user).group(1).split(", ")
        if "u3" in ids:
            return REFUSAL                                   # refused at the first ask and after the re-ask
        return json.dumps({i: {"p": [[50, 30, 20], [100, 0]]} for i in ids if i != "u0"})

    def test_refused_unparsed_ids(self):
        r = persona_distribution(FakeLLM(self.responder), self.PS, QS, batch=2)
        json.dumps(r)
        self.assertEqual(r["refused_ids"], ["u2", "u3"])
        self.assertEqual(r["unparsed_ids"], ["u0"])
        self.assertEqual(r["invalid_ids"], ["u0", "u2", "u3"])
        self.assertEqual((r["refused"], r["unparsed"], r["invalid"]), (2, 1, 3))
        self.assertAlmostEqual(r["refusal_rate"], 1 / 3, places=4)
        self.assertEqual(r["status"]["u1"], "ok")
        self.assertAlmostEqual(r["filled"]["A"]["q1"], 2 / 3)
        self.assertAlmostEqual(r["filled_share"], 3 / 5)
        f = distribution_frame(r)
        self.assertAlmostEqual(f.loc[f.unit == "B|q2", "filled_llm_dist"].item(), 0.5)

    def test_segment_schemes(self):
        llm = FakeLLM(lambda s, u: json.dumps({i: {"p": [[50, 30, 20], [100, 0]]}
                                               for i in re.findall(r"^(u\d): ", u, re.M)}))
        r = persona_distribution(llm, self.PS, QS, segments={"age": "age_band", "hh": lambda d: f"hh{d['hh']}"})
        self.assertEqual(set(r["dist"]), {"A", "B"})                  # persona segment kept
        self.assertEqual(set(r["dist_by"]["age"]), {"18-34", "35-54", "NA"})
        self.assertEqual(set(r["dist_by"]["hh"]), {"hh1", "hh2"})
        self.assertEqual(r["segments_by"]["age"]["u2"], "NA")
        np.testing.assert_allclose(r["dist_by"]["hh"]["hh2"]["q1"], [0.5, 0.3, 0.2])
        self.assertEqual(r["filled_by"]["age"]["18-34"]["q1"], 0.0)
        f = distribution_frame(r, scheme="hh", method="pd")
        self.assertEqual(sorted(f.unit), ["hh1|q1", "hh1|q2", "hh2|q1", "hh2|q2"])
        self.assertIn("filled_pd", f.columns)
        with self.assertRaises(KeyError):
            distribution_frame(r, scheme="income")
        with self.assertRaisesRegex(ValueError, "clash"):
            persona_distribution(llm, self.PS, QS, segments={"segment": "age_band"})
        pf = person_distribution_frame(r)
        self.assertEqual(len(pf), 5 * 2)
        self.assertEqual({"pid", "item", "p", "status", "segment", "age", "hh", "E_pos"} - set(pf.columns), set())
        self.assertAlmostEqual(pf.loc[(pf.pid == "u1") & (pf.item == "q1"), "E_pos"].item(), 0.3 + 0.4)

    def test_person_frame_invalid_rows(self):
        r = persona_distribution(FakeLLM(self.responder), self.PS, QS, batch=2)
        pf = person_distribution_frame(r)
        row = pf[(pf.pid == "u3") & (pf.item == "q1")].iloc[0]
        self.assertEqual(row["status"], "refused")
        self.assertIsNone(row["p"])
        self.assertTrue(np.isnan(row["E_pos"]))


class TestRetestDistribution(unittest.TestCase):
    ROWS = [{"unit": "g1", "item": "q", "sample": 0, "p": [1, 0, 0]},
            {"unit": "g1", "item": "q", "sample": 1, "p": [0, 0, 1]},
            {"unit": "g2", "item": "q", "sample": 0, "p": [0, 1, 0]},
            {"unit": "g2", "item": "q", "sample": 1, "p": [0, 1, 0]}]

    def test_positions_and_w1(self):
        st = retest_stats(self.ROWS, kind="distribution")
        self.assertAlmostEqual(st["within_cell_w1"], 0.5)          # (1.0 + 0.0) / 2
        self.assertAlmostEqual(st["within_cell_w1_max"], 1.0)
        self.assertAlmostEqual(st["identical_share"], 0.5)
        st = retest_stats(self.ROWS, kind="distribution", pos=[0, 9, 10])
        self.assertAlmostEqual(st["within_cell_sd"], (np.std([0, 10], ddof=1) + 0) / 2)
        self.assertAlmostEqual(st["within_cell_w1"], 0.5)
        st2 = retest_stats(self.ROWS, kind="distribution", pos={"q": [0, 9, 10]})
        self.assertEqual(st["within_cell_sd"], st2["within_cell_sd"])

    def test_missing_vector_and_errors(self):
        rows = self.ROWS + [{"unit": "g3", "item": "q", "sample": 0, "p": None},
                            {"unit": "g3", "item": "q", "sample": 1, "p": [0, 0, 1]}]
        st = retest_stats(rows, kind="distribution")
        self.assertEqual(st["n_cells"], 3)
        with self.assertRaises(ValueError):
            retest_stats(self.ROWS, kind="distribution", pos=[0, 1])
        with self.assertRaises(ValueError):
            retest_stats(self.ROWS, kind="vector")

    def test_probability_default_unchanged(self):
        rows = [{"unit": u, "item": "x", "sample": s, "p": 0.1 * (i + s)} for i, u in enumerate("abc")
                for s in range(2)]
        st = retest_stats(rows)
        self.assertNotIn("within_cell_w1", st)
        self.assertAlmostEqual(st["within_cell_sd"], np.std([0, 0.1], ddof=1))


if __name__ == "__main__":
    unittest.main()
