"""DistributionFake (G7 #11): an offline backend that reads the sim.distribution prompts (DIST_TEMPLATE and
PERSON_PROB_TEMPLATE) and answers with well-formed shares, so segment_distribution / persona_distribution
pipelines dry-run end to end without a real LLM; synthetic_distribution is its deterministic Dirichlet default."""
import json
import unittest

import numpy as np

from simengine.llm import DistributionFake, FakeLLM, looks_like_refusal, synthetic_distribution
from simengine.llm.fake_distribution import parse_distribution_prompt, parse_person_prompt
from simengine.sim import distribution_frame, persona_distribution, segment_distribution
from simengine.sim.distribution import FRAMINGS_NAMED, distribution_prompt, person_distribution_frame

QS = [{"id": "q1", "text": "Do you approve of the plan?", "options": ["Approve", "Neutral", "Disapprove"]},
      {"id": "q2", "text": "Own a car?", "options": ["Yes", "No"]},
      {"id": "q3", "text": "How often, e.g. weekly?", "options": ["Never", "Monthly", "Weekly", "Daily"]}]
SEGS = {"young": "adults aged 18 to 34", "old": "adults aged 65 or older, retired"}
PS = [{"id": f"u{i}", "text": f"person {i}, {'urban' if i % 2 else 'rural'}", "segment": "A" if i < 3 else "B"}
      for i in range(6)]


class TestSyntheticDistribution(unittest.TestCase):
    def test_deterministic_simplex(self):
        p = synthetic_distribution(4, "adults", "Own a car?", seed=3)
        self.assertEqual(len(p), 4)
        self.assertAlmostEqual(sum(p), 1.0, places=12)
        self.assertTrue(all(x >= 0 for x in p))
        self.assertEqual(p, synthetic_distribution(4, "adults", "Own a car?", seed=3))
        self.assertNotEqual(p, synthetic_distribution(4, "adults", "Own a car?", seed=4))
        self.assertNotEqual(p, synthetic_distribution(4, "retirees", "Own a car?", seed=3))

    def test_alpha_and_k(self):
        flat = synthetic_distribution(3, "g", "q", alpha=1e4)
        np.testing.assert_allclose(flat, [1 / 3] * 3, atol=0.02)
        with self.assertRaises(ValueError):
            synthetic_distribution(1, "g", "q")


class TestPromptParsing(unittest.TestCase):
    def test_distribution_prompt_fields(self):
        for head in list(FRAMINGS_NAMED.values()) + ["Consider {group}."]:
            u = distribution_prompt(head, SEGS["old"], QS[2], population="US adults", source="a study",
                                    when="2020")
            d = parse_distribution_prompt(u)
            self.assertEqual(d["group"], SEGS["old"])
            self.assertEqual(d["question"], QS[2]["text"])
            self.assertEqual(d["options"], QS[2]["options"])

    def test_person_prompt_fields(self):
        llm = DistributionFake()
        persona_distribution(llm, PS[:2], QS, batch=2)
        d = parse_person_prompt(llm.calls[0][1])
        self.assertEqual(d["ids"], ["u0", "u1"])
        self.assertEqual(d["profiles"]["u1"], "u1; person 1, urban")       # PersonaView.text, newlines -> "; "
        self.assertEqual([q["question"] for q in d["questions"]], [q["text"] for q in QS])
        self.assertEqual([q["options"] for q in d["questions"]], [q["options"] for q in QS])

    def test_other_prompts(self):
        self.assertIsNone(parse_distribution_prompt("hello"))
        self.assertIsNone(parse_person_prompt("hello"))


class TestSegmentDistribution(unittest.TestCase):
    def test_end_to_end_default(self):
        r = segment_distribution(DistributionFake(seed=1), SEGS, QS)
        self.assertEqual(r["invalid"], 0)
        self.assertEqual(r["filled_share"], 0.0)
        for s in SEGS:
            for q in QS:
                p = r["dist"][s][q["id"]]
                self.assertEqual(len(p), len(q["options"]))
                self.assertAlmostEqual(sum(p), 1.0, places=9)
        # noise=0: every framing gets the same vector, which is synthetic_distribution of the prompt's fields
        np.testing.assert_allclose(r["dist"]["old"]["q3"], synthetic_distribution(4, SEGS["old"], QS[2]["text"], 1),
                                   atol=1e-3)
        f = distribution_frame(r, truth={"young": {"q1": [0.2, 0.3, 0.5]}}, method="fake")
        self.assertEqual(len(f), len(SEGS) * len(QS))
        self.assertEqual({"unit", "truth", "pred_fake", "filled_fake", "segment", "item"} - set(f.columns), set())
        for p in f["pred_fake"]:
            self.assertAlmostEqual(sum(p), 1.0, places=9)
        self.assertTrue((f["filled_fake"] == 0).all())

    def test_reply_is_well_formed_json(self):
        llm = DistributionFake()
        u = distribution_prompt("Consider {group}.", "adults", QS[0])
        js = json.loads(llm.complete("", u).text)
        self.assertEqual(sorted(js), ["1", "2", "3"])
        self.assertAlmostEqual(sum(js.values()), 100.0, places=6)

    def test_value_fn_override(self):
        seen = []

        def vf(group, question, options):
            seen.append((group, question, tuple(options)))
            if question == "Own a car?":
                return [0.9, 0.1] if "65" in group else {"Yes": 30, "No": 70}
            return None                                          # default for the rest

        r = segment_distribution(DistributionFake(vf), SEGS, QS)
        np.testing.assert_allclose(r["dist"]["old"]["q2"], [0.9, 0.1], atol=1e-6)
        np.testing.assert_allclose(r["dist"]["young"]["q2"], [0.3, 0.7], atol=1e-6)
        self.assertIn((SEGS["young"], "Own a car?", ("Yes", "No")), seen)
        self.assertAlmostEqual(sum(r["dist"]["young"]["q1"]), 1.0, places=9)

    def test_value_fn_bad_value_raises(self):
        with self.assertRaisesRegex(ValueError, "value_fn"):
            segment_distribution(DistributionFake(lambda g, q, o: [1, 2, 3, 4, 5]), SEGS, QS[:1])

    def test_segments_map_descriptions_to_keys(self):
        r = segment_distribution(DistributionFake(lambda g, q, o: [1, 0] if g == "old" else [0, 1],
                                                  segments=SEGS), SEGS, QS[1:2])
        np.testing.assert_allclose(r["dist"]["old"]["q2"], [1, 0])
        np.testing.assert_allclose(r["dist"]["young"]["q2"], [0, 1])

    def test_noise_varies_framings_and_repeats(self):
        r0 = segment_distribution(DistributionFake(noise=0.0), SEGS, QS[:1], repeats=2)
        r1 = segment_distribution(DistributionFake(noise=0.5), SEGS, QS[:1], repeats=2)
        same = [c["p"] for c in r0["cells"] if c["segment"] == "young"]
        self.assertTrue(all(np.allclose(p, same[0], atol=1e-3) for p in same))
        noisy = [c["p"] for c in r1["cells"] if c["segment"] == "young"]
        self.assertGreater(len({tuple(np.round(p, 3)) for p in noisy}), 1)
        for p in noisy:
            self.assertAlmostEqual(sum(p), 1.0, places=9)
        r1b = segment_distribution(DistributionFake(noise=0.5), SEGS, QS[:1], repeats=2)
        self.assertEqual(r1["dist"], r1b["dist"])                 # still deterministic

    def test_refusal_and_unparsed_knobs(self):
        refuse = DistributionFake(refuse=lambda g, q: q == "Own a car?")
        self.assertTrue(looks_like_refusal(refuse.refusal_text))
        r = segment_distribution(refuse, SEGS, QS)
        n_fr = len(r["framings"])
        self.assertEqual(r["invalid"], len(SEGS) * n_fr)            # every q2 cell, still refused after re-ask
        self.assertEqual(r["usage"]["refused_calls"], len(SEGS) * n_fr)
        self.assertAlmostEqual(r["filled"]["young"]["q2"], 1.0)
        self.assertAlmostEqual(r["filled"]["young"]["q1"], 0.0)
        r = segment_distribution(DistributionFake(unparsed=lambda g, q: "65" in g), SEGS, QS)
        self.assertEqual(r["invalid"], len(QS) * len(r["framings"]))
        self.assertEqual(r["usage"].get("refused_calls", 0), 0)
        self.assertAlmostEqual(r["filled"]["old"]["q3"], 1.0)
        np.testing.assert_allclose(r["dist"]["old"]["q3"], [0.25] * 4)

    def test_rate_knobs(self):
        r = segment_distribution(DistributionFake(unparsed=1.0), SEGS, QS)
        self.assertEqual(r["filled_share"], 1.0)
        r = segment_distribution(DistributionFake(unparsed=0.5, seed=2), {f"g{i}": f"group {i}" for i in range(40)},
                                 QS[:1])
        self.assertTrue(0.2 < r["filled_share"] < 0.8, r["filled_share"])
        with self.assertRaises(ValueError):
            DistributionFake(refuse=1.5)

    def test_other_prompts_fall_through(self):
        llm = DistributionFake()
        self.assertIsInstance(llm, FakeLLM)
        self.assertEqual(llm.complete("", "hello").text, "{}")      # schema_responder's catch-all
        self.assertEqual(llm.stats["calls"], 1)
        self.assertEqual(llm.spent_usd, 0.0)


class TestPersonaDistribution(unittest.TestCase):
    def test_end_to_end_default(self):
        r = persona_distribution(DistributionFake(), PS, QS, batch=4)
        self.assertEqual(r["invalid"], 0)
        self.assertEqual(r["calls"], 2)
        self.assertEqual(set(r["status"].values()), {"ok"})
        for pid, d in r["person"].items():
            for q in QS:
                self.assertAlmostEqual(sum(d[q["id"]]), 1.0, places=9)
        f = distribution_frame(r)
        self.assertEqual(sorted(f.unit), sorted(f"{s}|{q['id']}" for s in "AB" for q in QS))
        for p in f["pred_llm_dist"]:
            self.assertAlmostEqual(sum(p), 1.0, places=9)
        self.assertEqual(len(person_distribution_frame(r)), len(PS) * len(QS))
        self.assertNotEqual(r["person"]["u0"]["q1"], r["person"]["u1"]["q1"])   # keyed on the profile

    def test_value_fn_by_id_and_profile(self):
        r = persona_distribution(DistributionFake(lambda g, q, o: [1, 0] if "urban" in g else [0, 1]),
                                 PS, QS[1:2], batch=3)
        np.testing.assert_allclose(r["person"]["u1"]["q2"], [1, 0])
        np.testing.assert_allclose(r["person"]["u2"]["q2"], [0, 1])
        llm = DistributionFake(lambda g, q, o: [1, 0] if g == "u4" else None, person_key="id")
        r = persona_distribution(llm, PS, QS[1:2], batch=3)
        np.testing.assert_allclose(r["person"]["u4"]["q2"], [1, 0])
        with self.assertRaises(ValueError):
            DistributionFake(person_key="name")

    def test_refusal_and_unparsed_knobs(self):
        r = persona_distribution(DistributionFake(unparsed=lambda g, q: "person 2" in g), PS, QS, batch=3)
        self.assertEqual(r["unparsed_ids"], ["u2"])
        self.assertEqual(r["refused"], 0)
        r = persona_distribution(DistributionFake(refuse=lambda g, q: "person 4" in g), PS, QS, batch=3)
        self.assertEqual(r["refused_ids"], ["u3", "u4", "u5"])        # the whole batch refuses
        self.assertAlmostEqual(r["refusal_rate"], 0.5)
        self.assertAlmostEqual(r["filled"]["B"]["q1"], 1.0)

    def test_rate_knobs_are_per_batch_and_per_respondent(self):
        # a rate used to be drawn per respondent x question and any hit refused the batch: refuse=0.05 with
        # batch=12 and 3 questions refused 80% of 240 personas; now the rate is the share of batches / respondents
        qs = [{"id": f"q{i}", "text": f"Question {i}?", "options": ["a", "b", "c"]} for i in range(3)]
        ps = [{"id": f"u{i}", "text": f"person {i}", "segment": "A"} for i in range(480)]
        r = persona_distribution(DistributionFake(refuse=0.1), ps, qs, batch=12)
        self.assertLess(r["refusal_rate"], 0.3, r["refusal_rate"])            # 40 batches, expected 0.1
        self.assertEqual(r["refused"], 12 * round(r["refusal_rate"] * 40))
        u = persona_distribution(DistributionFake(unparsed=0.1), ps, qs, batch=12)
        self.assertTrue(0.05 < u["unparsed"] / len(ps) < 0.15, u["unparsed"])
        self.assertEqual(persona_distribution(DistributionFake(refuse=1.0), ps[:24], qs, batch=12)["refused"], 24)

    def test_ids_with_commas_colons_and_parentheses(self):
        ps = [{"id": "NY, urban", "text": "x", "segment": "A"}, {"id": "b(2)", "text": "y", "segment": "A"},
              {"id": "c: 3", "text": "z", "segment": "A"}]
        r = persona_distribution(DistributionFake(), ps, QS, batch=3)
        self.assertEqual(r["status"], {p["id"]: "ok" for p in ps})
        r = persona_distribution(DistributionFake(lambda g, q, o: [1, 0] if g == "NY, urban" else None,
                                                  person_key="id"), ps, QS[1:2], batch=3)
        np.testing.assert_allclose(r["person"]["NY, urban"]["q2"], [1, 0])


class TestCustomFraming(unittest.TestCase):
    def test_segments_key_under_a_framing_without_consider(self):
        seen = set()

        def vf(g, q, o):
            seen.add(g)
            return [1, 0] if g == "old" else [0, 1]

        r = segment_distribution(DistributionFake(vf, segments=SEGS), SEGS, QS[1:2],
                                 framings={"x": "Among {group}, a poll was run.", "y": "Consider {group}."})
        self.assertEqual(seen, {"old", "young"})
        np.testing.assert_allclose(r["dist"]["old"]["q2"], [1, 0])
        np.testing.assert_allclose(r["dist"]["young"]["q2"], [0, 1])


if __name__ == "__main__":
    unittest.main()
