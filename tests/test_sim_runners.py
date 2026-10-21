"""simengine.sim runners on FakeLLM: survey (+SSR), A/B, roundtable, strategy."""
import hashlib
import json
import re
import unittest

import numpy as np

from simengine.llm import FakeLLM
from simengine.sim import (SSR, KEEP_VIEW, ab_frame, ab_many, ab_test, confidence_check, paars_decision,
                           roundtable, roundtable_frame, schema_responder, strategy, strategy_frame, survey,
                           survey_frame)


def personas(n=6):
    return [{"id": f"u{i}", "name": f"N{i}", "text": f"age {20 + i}, city {i}", "segment": "A" if i % 2 else "B"}
            for i in range(n)]


def bow_encoder(texts, dim=256):
    """Hashed bag-of-words embedding: identical sentences -> identical vectors, no model download."""
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z']+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    return out


ANCHORS = ["I would definitely not buy it.", "I probably would not buy it.", "I might or might not buy it.",
           "I would probably buy it.", "I would definitely buy it."]


class TestSSR(unittest.TestCase):
    def test_distribution_and_expected(self):
        s = SSR(encoder=bow_encoder)
        P = s.distribution(["I would definitely buy it.", "I would definitely not buy it."], ANCHORS)
        self.assertEqual(P.shape, (2, 5))
        np.testing.assert_allclose(P.sum(axis=1), 1.0)
        self.assertEqual(int(P[0].argmax()), 4)
        self.assertGreater(SSR.expected(P)[0], SSR.expected(P)[1])
        np.testing.assert_allclose(SSR.to_probability(np.eye(5)[[0, 4]]), [0.0, 1.0])
        self.assertEqual(s.distribution([], "intent").shape, (0, 5))


class TestSurvey(unittest.TestCase):
    Q = [{"id": "q1", "text": "Would you buy it?", "scale": "intent"},
         {"id": "q2", "text": "Pick one", "kind": "choice", "options": ["x", "y", "z"]}]

    def test_survey_with_controls_and_events(self):
        llm = FakeLLM(schema_responder(1))
        ev = []
        r = survey(personas(), self.Q, llm=llm, B=50, on_event=ev.append)
        json.dumps(r)
        self.assertEqual(r["n_answered"], 6)
        q1, q2 = r["questions"]
        self.assertEqual(set(q1["arms"]), {"persona", "no_persona", "persona_swap"})
        self.assertEqual(q1["arms"]["persona"]["n"], 6)
        self.assertAlmostEqual(sum(q2["arms"]["persona"]["dist"].values()), 1.0)
        self.assertEqual({e["type"] for e in ev}, {"stage", "row", "done"})
        # swap control prompts carry each persona's own id/name but the next persona's content
        swap = [u for _, u in llm.calls if "### u0\nN0\nage 21, city 1" in u]
        self.assertTrue(swap)
        f = survey_frame(r, truth={"q1": {"u0": 4}})
        self.assertEqual(len(f), 6)
        self.assertEqual(f.loc[f.unit == "u0|q1", "truth"].item(), 4)
        for c in ("pred_score", "pred_ssr", "pred_no_persona_score", "pred_persona_swap_score"):
            self.assertIn(c, f.columns)
        self.assertTrue(f["pred_ssr"].isna().all())      # SSR off by default
        fi = survey_frame(r, level="item", kind="choice")
        self.assertIn("all", set(fi.segment))

    def test_single_question_with_options_and_no_controls(self):
        llm = FakeLLM(schema_responder(2))
        r = survey(personas(4), "Which colour?", ["red", "blue"], llm=llm, controls=[], B=20)
        self.assertEqual(r["questions"][0]["kind"], "choice")
        self.assertEqual(len(llm.calls), 1)

    def test_ssr_scores_free_text(self):
        def responder(system, user):
            ids = re.findall(r"^### (\S+)\s*$", user, re.M)
            return json.dumps([{"id": p, "q1_answer": ANCHORS[4] if p in ("u0", "u2") else ANCHORS[0],
                                "q1_score": 3} for p in ids])
        r = survey(personas(4), [self.Q[0]], llm=FakeLLM(responder), ssr=SSR(encoder=bow_encoder), controls=[],
                   B=20)
        f = survey_frame(r)
        v = dict(zip(f.unit, f.pred_ssr))
        self.assertGreater(v["u0|q1"], v["u1|q1"])
        self.assertTrue(f.pred_ssr.notna().all())


class TestAB(unittest.TestCase):
    V = [{"id": "A", "text": "red button"}, {"id": "B", "text": "blue button"}]

    def test_ab_test(self):
        llm = FakeLLM(schema_responder(1))
        r = ab_test(personas(), self.V, llm=llm, B=50, truth={"A": 0.02, "B": 0.03})
        json.dumps(r)
        self.assertIn(r["winner"], ("A", "B"))
        self.assertEqual(set(r["arms"]), {"persona", "no_persona", "persona_swap"})
        self.assertEqual(r["truth_comparison"]["true_winner"], "B")
        self.assertIn("low", r["confidence"])
        # half of the personas see the variants in reverse order
        self.assertEqual({a["order"] for a in r["answers"].values()}, {"forward", "reverse"})
        self.assertIn("value", r["position_effect"])
        f = ab_frame(r)
        for c in ("pred_score", "pred_share", "pred_win"):
            self.assertIn(c, f.columns)

    def test_confidence_check(self):
        strong = {"persona": {"winner": "A", "variants": {"A": {"share": 0.8}, "B": {"share": 0.2}},
                              "sign_test": {"p": 0.01}},
                  "persona_swap": {"winner": "A", "n": 10}}
        self.assertFalse(confidence_check(strong)["low"])
        flipped = {**strong, "persona_swap": {"winner": "B", "n": 10}}
        c = confidence_check(flipped)
        self.assertTrue(c["low"])
        self.assertEqual(c["reasons"][0]["code"], "arm_persona_swap")

    def test_paars_decision(self):
        d = paars_decision({f"p{i}": {"A": 8, "B": 3} for i in range(9)} | {"p9": {"A": 2, "B": 5}})
        self.assertEqual(d["decision"], "winner_A")
        self.assertAlmostEqual(d["confidence"], 0.9)
        self.assertEqual(paars_decision({"p": {"A": 5, "B": 5}})["decision"], "inconclusive")
        self.assertEqual(paars_decision({"a": {"A": 6, "B": 5}, "b": {"A": 5, "B": 6}})["decision"],
                         "inconclusive")

    def test_ab_many(self):
        def responder(system, user):
            tids = re.findall(r"^- (t\d+): A = ", user, re.M)
            pids = re.findall(r"^- (u\d+): ", user, re.M)
            return json.dumps([{"id": f"{p}|{t}", "a": 7, "b": 3, "choice": "A" if t != "t2" else "B"}
                               for p in pids for t in tids])
        tests = [{"test_id": f"t{i}", "a": f"head A{i}", "b": f"head B{i}"} for i in range(3)]
        llm = FakeLLM(responder)
        df = ab_many(personas(4), tests, llm=llm, tests_per_call=2)
        self.assertEqual(len(llm.calls), 2)
        p = dict(zip(df.test_id, df.p_a))
        self.assertAlmostEqual(p["t0"], 4.5 / 5)
        self.assertAlmostEqual(p["t2"], 0.5 / 5)
        self.assertTrue((df.n_votes == 4).all())


class TestRoundtable(unittest.TestCase):
    def test_rounds_context_and_frame(self):
        llm = FakeLLM(schema_responder(3))
        r = roundtable(personas(), "a neighbourhood tea shop", llm=llm, rounds=3, B=30, extra_instruction=KEEP_VIEW)
        json.dumps(r)
        self.assertEqual(len(r["table"]), 3)
        later = [u for _, u in llm.calls if "round 2 of 3" in u]
        self.assertTrue(later and all("Your previous answer" in u for u in later))
        self.assertTrue(all(KEEP_VIEW.strip() in u for u in later))
        self.assertTrue(all("Your previous answer" not in u for _, u in llm.calls if "round 1 of 3" in u))
        f = roundtable_frame(r, truth={"u0": 60})
        self.assertEqual(list(f.columns[:4]), ["unit", "truth", "pred_round0", "pred_round2"])
        fs = roundtable_frame(r, truth={"A": 50}, level="segment")
        self.assertEqual(set(fs.unit), {"A", "B", "all"})

    def test_missing_answer_carried_forward(self):
        n = {"c": 0}

        def responder(system, user):
            n["c"] += 1
            ids = re.findall(r"^### (\S+)\s*$", user, re.M)
            if "round 2 of 2" in user:
                ids = [i for i in ids if i != "u1"]
            return json.dumps([{"id": i, "intent": 40, "remark": "fine"} for i in ids])
        r = roundtable(personas(3), "topic", llm=FakeLLM(responder), rounds=2, B=20)
        self.assertTrue(r["answers"]["1"]["u1"]["carried"])     # JSON-safe: round keys are strings
        self.assertEqual(roundtable_frame(r).pred_round1.notna().sum(), 3)


class TestStrategy(unittest.TestCase):
    def test_strategy(self):
        llm = FakeLLM(schema_responder(4))
        r = strategy(personas(), ["cut price 10%", "free delivery", {"id": "L", "text": "loyalty points"}],
                     llm=llm, B=30, truth={"S1": 5, "S2": 9, "L": 1})
        json.dumps(r)
        self.assertEqual(sorted(r["ranking"]), ["L", "S1", "S2"])
        self.assertEqual(r["truth_comparison"]["true_winner"], "S2")
        self.assertIn(r["divergence"]["most_divergent"], ("A", "B"))
        f = strategy_frame(r)
        self.assertEqual(len(f), 3 * 3)
        self.assertEqual(f.loc[f.unit == "S2|all", "truth"].item(), 9)
        with self.assertRaises(ValueError):
            strategy(personas(), ["only one"], llm=llm)


if __name__ == "__main__":
    unittest.main()
