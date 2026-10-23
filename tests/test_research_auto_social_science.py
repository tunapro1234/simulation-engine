"""research.hypotheses: automated social science loop (Manning, Zhu & Horton 2024) and the experiment outcome
predictor (Hewitt et al. 2024). Offline: FakeLLM responders only, synthetic fixtures; the one real-data check is
skipped unless $SIMENGINE_DATA is set."""
import importlib.util
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from simengine.llm.fake import FakeLLM
from simengine.research import hypotheses as H

# parquet needs pyarrow or fastparquet; without one the classes that use it skip (nothing gets installed)
PARQUET = any(importlib.util.find_spec(m) for m in ("pyarrow", "fastparquet"))
NO_PARQUET = "no parquet engine (pyarrow / fastparquet)"

SCENARIO = "A buyer and a seller bargain over a used mug."

HYP_JSON = {"hypotheses": [{
    "agents": ["buyer", "seller"], "rationale": "budget raises the price; attachment makes the seller firm",
    "outcome": {"name": "deal_price", "kind": "continuous", "units": "dollars", "agent": "buyer",
                "question": "What price did you agree on, in dollars?", "lo": 0, "hi": 100},
    "causes": [{"name": "buyer_budget", "kind": "continuous", "units": "dollars", "levels": [10, 20, 40],
                "proxy": "Your budget", "agent": "buyer"},
               {"name": "seller_attachment", "kind": "binary", "levels": ["low", "high"],
                "proxy": "Your attachment to the mug", "agent": "seller"}]}]}


def _budget(text):
    m = re.search(r"Your budget: (\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def bargain_responder(system, user):
    """Buyer answers 0.5 * budget + 5, plus 10 if the seller said it is attached (only visible in a conversation)."""
    if system == H.HYPOTHESIS_SYSTEM:
        return json.dumps(HYP_JSON)
    if system == H.PATH_SYSTEM:
        return json.dumps({"predictions": {"buyer_budget": 2.0, "seller_attachment": 8.0},
                           "sig": {"buyer_budget": True, "seller_attachment": False}})
    if system == H.JUDGE_SYSTEM:
        return "yes" if user.count("seller:") >= 1 and user.count("buyer:") >= 1 else "no"
    if "Survey question" in user:
        b = _budget(system) or 0.0
        bonus = 10.0 if "very attached" in user else 0.0
        return json.dumps({"answer": 0.5 * b + 5 + bonus})
    if "You are the seller" in system:
        return "I am very attached to this mug." if "attachment to the mug: high" in system else "Fine, make an offer."
    return "I would like to buy the mug."


def _hyp():
    hyps, problems = H.parse_hypotheses(HYP_JSON, SCENARIO)
    assert not problems, problems
    return hyps[0]


class TestParse(unittest.TestCase):
    def test_repairs_and_rejections(self):
        bad = {"hypotheses": [
            {"agents": ["a"], "outcome": {"name": "Y val", "question": "How much?"},
             "causes": [{"name": "x", "kind": "continuous", "levels": [1, "two", "3", 4, 5, 6, 7]},
                        {"name": "z", "kind": "binary", "levels": ["only"]},
                        {"name": "w", "kind": "weird", "levels": [0, 1], "agent": "ghost"}],
             "interactions": [["x", "nope"], ["x", "w"]]},
            {"agents": ["a"], "outcome": {"name": "y"}, "causes": [{"name": "x", "levels": [0, 1]}]}]}
        hyps, problems = H.parse_hypotheses(json.dumps(bad), SCENARIO, max_levels=5)
        self.assertEqual(len(hyps), 1)
        h = hyps[0]
        self.assertEqual(h.outcome.name, "y_val")
        self.assertEqual(h.outcome.agent, "a")
        self.assertEqual([c.name for c in h.causes], ["x", "w"])
        self.assertEqual(h.cause("x").levels, [1.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(h.cause("w").kind, "continuous")
        self.assertIsNone(h.cause("w").agent)
        self.assertEqual(h.interactions, [("x", "w")])
        text = " | ".join(problems)
        for needle in ("non-numeric", "kept the first 5", "binary with 1", "not in agents", "unknown causes",
                       "no survey question"):
            self.assertIn(needle, text)

    def test_roundtrip_and_id(self):
        h = _hyp()
        h2 = H.Hypothesis.from_dict(h.to_dict())
        self.assertEqual(h.id, h2.id)
        self.assertEqual(h.edges(), ["buyer_budget -> deal_price", "seller_attachment -> deal_price"])

    def test_propose_reasks_once(self):
        replies = iter(["no json here", json.dumps(HYP_JSON)])
        llm = FakeLLM(lambda s, u: next(replies))
        log = H.DecisionLog()
        hyps = H.propose_hypotheses(llm, SCENARIO, log=log)
        self.assertEqual(len(hyps), 1)
        self.assertEqual(len(log.events("propose")), 2)
        self.assertIn("previous reply was rejected", llm.calls[1][1])
        self.assertEqual(len(log.events("hypothesis")), 1)


class TestDesignEstimate(unittest.TestCase):
    def test_factorial_and_subsample(self):
        h = _hyp()
        log = H.DecisionLog()
        d = H.design(h, replicates=3, log=log)
        self.assertEqual(len(d), 3 * 2 * 3)
        self.assertEqual(d.groupby("cell").size().tolist(), [3] * 6)
        d2 = H.design(h, max_cells=4, seed=1, log=log)
        self.assertEqual(len(d2), 4)
        self.assertTrue(log.events("design")[1]["subsampled"])

    def test_encode_nominal_and_interaction(self):
        h = H.Hypothesis(SCENARIO, ["a"], [H.Variable("x", "continuous", [0.0, 1.0]),
                                           H.Variable("g", "nominal", ["r", "s", "t"])],
                         H.Variable("y", question="?"), interactions=[("x", "g")])
        X, owner = H.encode(h, H.design(h))
        self.assertEqual(list(X.columns), ["x", "g[s]", "g[t]", "x:g[s]", "x:g[t]"])
        self.assertEqual(owner["x:g[t]"], "x:g")

    def test_recovers_linear_world(self):
        h = H.Hypothesis(SCENARIO, ["a"], [H.Variable("x", "continuous", [0, 1, 2, 3]),
                                           H.Variable("g", "nominal", ["r", "s"])], H.Variable("y", question="?"))
        d = H.design(h, replicates=50)
        d["y"] = H.linear_world({"x": 1.5}, intercept=2.0, noise=0.5, seed=3, levels={"g": {"s": -1.0}})(h, d)
        p, info = H.estimate(h, d, backend="numpy")
        c = p.set_index("term").coef
        self.assertAlmostEqual(c["x"], 1.5, delta=0.1)
        self.assertAlmostEqual(c["g[s]"], -1.0, delta=0.15)
        self.assertEqual(info["n"], 400)
        try:
            import statsmodels  # noqa: F401
        except ImportError:
            return
        p2, info2 = H.estimate(h, d, backend="statsmodels")
        self.assertEqual(info2["backend"], "statsmodels")
        np.testing.assert_allclose(p.coef, p2.coef, atol=1e-9)
        np.testing.assert_allclose(p.se, p2.se, atol=1e-9)
        np.testing.assert_allclose(p.p, p2.p, atol=1e-9)

    def test_missing_constant_and_rank(self):
        h = H.Hypothesis(SCENARIO, ["a"], [H.Variable("x", "continuous", [0, 1, 2])], H.Variable("y", question="?"))
        d = H.design(h, replicates=4)
        d["y"] = d.x * 2.0 + np.r_[np.zeros(11), np.nan]
        d.loc[0, "y"] = 0.1
        p, info = H.estimate(h, d, backend="numpy")
        self.assertEqual(info["dropped_missing"], 1)
        h2 = H.Hypothesis(SCENARIO, ["a"], [H.Variable("x", "continuous", [0, 1, 2]),
                                            H.Variable("x2", "continuous", [0, 2, 4])], H.Variable("y", question="?"))
        d2 = pd.DataFrame({"x": [0, 1, 2] * 3, "x2": [0, 2, 4] * 3, "y": np.arange(9.0)})
        with self.assertRaises(ValueError):
            H.estimate(h2, d2)


class TestLoop(unittest.TestCase):
    def test_interview_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeLLM(bargain_responder)
            ass = H.AutoSocialScience(llm, SCENARIO, simulator="interview", replicates=2, workdir=tmp)
            res = ass.run()
            self.assertEqual(len(res.runs), 1)
            r = res.runs[0]
            c = r.paths.set_index("term").coef
            self.assertAlmostEqual(c["buyer_budget"], 0.5, places=9)
            self.assertAlmostEqual(c["seller_attachment"], 0.0, places=9)   # private to the seller: buyer never sees it
            self.assertEqual(r.summary["n_paths"], 2)
            self.assertAlmostEqual(r.comparison.set_index("term").ratio["buyer_budget"], 4.0)
            events = [e["event"] for e in res.log.records]
            self.assertLess(events.index("predict_paths"), events.index("simulate"))
            self.assertLess(events.index("simulate"), events.index("fit"))
            lines = (Path(tmp) / "decisions.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), len(res.log.records))
            self.assertTrue((Path(tmp) / "paths.csv").exists())
            self.assertEqual(json.loads((Path(tmp) / "summary.json").read_text())[0]["hypothesis"], r.hypothesis.id)
            self.assertIn("buyer_budget", res.summary.edges.iloc[0])

    def test_conversation_transmits_private_attribute(self):
        llm = FakeLLM(bargain_responder)
        sim = H.ConversationSimulator(llm, max_statements=6)
        h = _hyp()
        d = H.design(h)
        log = H.DecisionLog()
        d["y"] = sim(h, d, log)
        p, _ = H.estimate(h, d)
        c = p.set_index("term").coef
        self.assertAlmostEqual(c["seller_attachment"], 10.0, places=9)
        self.assertAlmostEqual(c["buyer_budget"], 0.5, places=9)
        ev = log.events("simulate")[0]
        self.assertEqual(ev["ended_by_judge"], len(d))
        self.assertEqual(ev["mean_statements"], 2.0)
        self.assertEqual([a for a, _ in sim.transcripts[0]], ["buyer", "seller"])

    def test_random_protocol_and_cap(self):
        llm = FakeLLM(lambda s, u: "no" if s == H.JUDGE_SYSTEM else '{"answer": 3}')
        sim = H.ConversationSimulator(llm, protocol="random", max_statements=4, seed=2)
        h = _hyp()
        y = sim(h, H.design(h).head(3))
        np.testing.assert_array_equal(y, [3.0, 3.0, 3.0])
        self.assertTrue(all(len(t) == 4 for t in sim.transcripts))
        self.assertTrue(all(t[i][0] != t[i + 1][0] for t in sim.transcripts for i in range(3)))
        with self.assertRaises(ValueError):
            H.ConversationSimulator(llm, protocol="debate")

    def test_market_world_signs(self):
        h = H.Hypothesis("a shop runs a promotion", ["shopper"],
                         [H.Variable("price_cut", "continuous", [0.0, 0.1, 0.2, 0.3]),
                          H.Variable("promo", "binary", [0, 1])], H.Variable("sales", question="?"))
        d = H.design(h, replicates=2)
        d["y"] = H.market_world(weeks=3, n_agents=200)(h, d)
        p, _ = H.estimate(h, d)
        c = p.set_index("term").coef
        self.assertGreater(c["price_cut"], 0)
        self.assertGreater(c["promo"], 0)

    def test_failed_hypothesis_is_logged(self):
        h = H.Hypothesis(SCENARIO, ["a"], [H.Variable("x", "continuous", [0, 1])], H.Variable("y", question="?"))
        llm = FakeLLM(lambda s, u: "nothing")
        res = H.AutoSocialScience(llm, SCENARIO, simulator="interview").run(hypotheses=[h])
        self.assertEqual(res.runs, [])
        self.assertEqual(len(res.log.events("failed")), 1)

    def test_compare_and_predict_outcomes(self):
        fitted = pd.DataFrame({"term": ["(intercept)", "a", "b", "c"], "coef": [1.0, 2.0, -1.0, 0.5],
                               "p": [0.0, 0.001, 0.2, 0.01]})
        pred = pd.DataFrame({"term": ["a", "b", "c"], "predicted": [4.0, 1.0, 0.25], "sig_share": [1.0, 0.0, 0.0]})
        m, s = H.compare_paths(fitted, pred)
        self.assertEqual(s["n_paths"], 3)
        self.assertAlmostEqual(s["sign_agreement"], 2 / 3)
        self.assertAlmostEqual(s["sig_agreement"], 2 / 3)
        self.assertAlmostEqual(s["mean_ratio"], (2 + 1 + 0.5) / 3)
        self.assertAlmostEqual(s["overestimate_share"], 1 / 3)
        llm = FakeLLM(lambda s, u: '{"answer": 7}')
        h = _hyp()
        y = H.predict_outcomes(llm, h, H.design(h).head(2), fitted=fitted)
        np.testing.assert_array_equal(y, [7.0, 7.0])
        self.assertIn("Fitted linear SCM", llm.calls[0][1])

    def test_predict_paths_samples(self):
        vals = iter([1.0, 3.0])
        llm = FakeLLM(lambda s, u: json.dumps({"predictions": {"x": next(vals)}, "sig": {"x": "yes"}}))
        h = H.Hypothesis(SCENARIO, ["a"], [H.Variable("x", "continuous", [0, 1])], H.Variable("y", question="?"))
        out = H.predict_paths(llm, h, ["x"], 10, n_samples=2)
        self.assertAlmostEqual(out.predicted[0], 2.0)
        self.assertEqual(out.sig_share[0], 1.0)
        self.assertNotEqual(llm.calls[0][1], llm.calls[1][1])

    def test_parse_number(self):
        self.assertEqual(H.parse_number('{"answer": "12"}'), 12.0)
        self.assertEqual(H.parse_number("about 1,250 dollars"), 1250.0)
        self.assertTrue(np.isnan(H.parse_number("8", lo=0, hi=7)))
        self.assertTrue(np.isnan(H.parse_number("none")))


def survey_responder(system, user):
    """Agreement 4 at baseline, +1 for the 'strong' message; Democrats +0.5 everywhere; 'garbage' for one profile."""
    if system == H.DIRECT_SYSTEM:
        return json.dumps({"means": {"control": 4.0, "weak": 4.2, "strong": 5.5}})
    if "party: Other" in user:
        return "I cannot say"
    base = 4.0 + (1.0 if "STRONG" in user else 0.0) + (0.5 if "party: Democrat" in user else 0.0)
    return json.dumps({"answer": base})


class TestExperimentForecast(unittest.TestCase):
    PROFILES = pd.DataFrame({"age": [30, 50, 70, 40], "party": ["Democrat", "Republican", "Democrat", "Other"]})
    CONDS = {"control": "A neutral note.", "weak": "A mild appeal.", "strong": "A STRONG appeal."}

    def test_simulate_mode(self):
        llm = FakeLLM(survey_responder)
        log = H.DecisionLog()
        fc = H.forecast_experiment(llm, "A study on civic appeals.", self.CONDS, "How much do you agree?",
                                   scale=(1, 7), labels=("disagree", "agree"), control="control",
                                   profiles=self.PROFILES, n_per_condition=40, log=log)
        self.assertEqual(fc.calls, 120)
        e = fc.effects.set_index("condition").effect
        self.assertAlmostEqual(e["strong"], 1.0, places=9)    # same profiles across conditions -> exact
        self.assertAlmostEqual(e["weak"], 0.0, places=9)
        self.assertGreater(fc.conditions.n_invalid.sum(), 0)
        self.assertEqual(log.events("forecast_experiment")[0]["invalid"], int(fc.conditions.n_invalid.sum()))
        systems = {s for s, _ in llm.calls}
        self.assertGreater(len(systems), 1)                    # several templates from the bank

    def test_direct_mode_and_errors(self):
        llm = FakeLLM(survey_responder)
        fc = H.forecast_experiment(llm, "setting", self.CONDS, "Agree?", control="control", mode="direct",
                                   n_samples=3)
        self.assertAlmostEqual(fc.effects.set_index("condition").effect["strong"], 1.5)
        self.assertEqual(fc.calls, 3)
        with self.assertRaises(ValueError):
            H.forecast_experiment(llm, "s", self.CONDS, "q", control="missing")

    def test_scores(self):
        rng = np.random.default_rng(0)
        true = rng.normal(0, 1, 200)
        se = np.full(200, 0.5)
        obs = true + rng.normal(0, 0.5, 200)
        s = H.score_effect_forecasts(true, obs, se)
        self.assertGreater(s["r_adj"], s["r"])
        self.assertAlmostEqual(s["r_adj"], 1.0, delta=0.06)
        self.assertGreater(s["sign_acc_sig"], 0.95)
        self.assertTrue(np.isnan(H.score_effect_forecasts([1, 2], [1, 2])["r"]))

    def test_condition_protocol(self):
        df = pd.DataFrame({"study": np.repeat(["s1", "s2", "s3", "s4"], 3), "condition": list("abc") * 4,
                           "true_mean": [1, 2, 4, 0, 1, 1.5, 3, 2, 5, 1, 1, 2.0],
                           "true_se": 0.01})
        df["pred_mean"] = 2 * df.true_mean + 7          # affine per study -> contrasts perfectly correlated
        s = H.score_condition_forecasts(df, repeats=8)
        self.assertAlmostEqual(s["r"], 1.0)
        self.assertAlmostEqual(s["slope"], 0.5)
        self.assertEqual(s["n"], 8)
        self.assertEqual(s["n_studies"], 4)


class TestReviewRegressions(unittest.TestCase):
    def test_interactions_after_cause_cut(self):
        obj = {"agents": ["a"], "outcome": {"name": "y", "question": "?"},
               "causes": [{"name": n, "levels": [0, 1]} for n in ("p", "q", "r", "s")],
               "interactions": [["p", "s"], ["P", "Q"], ["q", "p"]]}
        hyps, problems = H.parse_hypotheses(obj, SCENARIO, max_causes=3)
        h = hyps[0]
        self.assertEqual([c.name for c in h.causes], ["p", "q", "r"])
        self.assertEqual(h.interactions, [("p", "q")])         # s was cut; "P","Q" normalised; q:p duplicate
        self.assertIn("duplicate", " | ".join(problems))
        self.assertEqual(list(H.encode(h, H.design(h))[0].columns), ["p", "q", "r", "p:q"])

    def test_levels_dedup_after_conversion(self):
        obj = {"agents": ["a"], "outcome": {"name": "y", "question": "?"},
               "causes": [{"name": "x", "levels": [10, "10", "$20"]}]}
        hyps, _ = H.parse_hypotheses(obj, SCENARIO)
        self.assertEqual(hyps[0].cause("x").levels, [10.0, 20.0])
        obj["causes"][0]["levels"] = [5, "5"]
        self.assertEqual(H.parse_hypotheses(obj, SCENARIO)[0], [])

    def test_number_parsing_consistent(self):
        self.assertEqual(H.parse_number('{"answer": "1,250"}'), 1250.0)
        self.assertEqual(H.parse_number("about 1,250 dollars"), 1250.0)
        self.assertEqual(H.parse_number('{"answer": "3,5"}'), 3.5)
        self.assertTrue(np.isnan(H.parse_number('{"answer": null, "confidence": 3}')))

    def test_judge_after_first_statement(self):
        llm = FakeLLM(lambda s, u: "yes" if s == H.JUDGE_SYSTEM else '{"answer": 1}')
        h = _hyp()
        d = H.design(h).head(2)
        log = H.DecisionLog()
        H.ConversationSimulator(llm, max_statements=5)(h, d, log)
        self.assertEqual(log.events("simulate")[0]["mean_statements"], 1.0)
        log2 = H.DecisionLog()
        H.ConversationSimulator(llm, max_statements=5, judge_from=3)(h, d, log2)
        self.assertEqual(log2.events("simulate")[0]["mean_statements"], 3.0)

    def test_outcome_agent_fallback_and_subsample_prompt(self):
        h = H.Hypothesis(SCENARIO, ["bidder"], [H.Variable("x", "continuous", [0, 1, 2])], H.Variable("y", question="?"))
        llm = FakeLLM(lambda s, u: '{"answer": 1}')
        H.InterviewSimulator(llm)(h, H.design(h))
        self.assertTrue(all(s.startswith("You are the bidder") for s, _ in llm.calls))
        self.assertIn("Every combination", H.path_prediction_prompt(h, ["x"], 3))
        self.assertIn("random subset", H.path_prediction_prompt(h, ["x"], 2, all_cells=False))
        llm2 = FakeLLM(lambda s, u: json.dumps({"predictions": {"x": 1.0}}) if s == H.PATH_SYSTEM
                       else '{"answer": 1}')
        H.AutoSocialScience(llm2, SCENARIO, max_cells=2).run(hypotheses=[h])
        self.assertIn("random subset", next(u for s, u in llm2.calls if s == H.PATH_SYSTEM))

    def test_paired_se(self):
        # same profile and template -> the paired difference is exactly 1 for every respondent -> se 0
        llm = FakeLLM(lambda s, u: json.dumps({"answer": (5.0 if "STRONG" in u else 4.0)
                                                + (1.0 if "party: Democrat" in u else 0.0)}))
        prof = pd.DataFrame({"party": ["Democrat", "Republican"]})
        conds = {"control": "note", "strong": "A STRONG appeal."}
        fc = H.forecast_experiment(llm, "s", conds, "q", control="control", profiles=prof, n_per_condition=30)
        self.assertAlmostEqual(fc.effects.se[0], 0.0)
        fc2 = H.forecast_experiment(llm, "s", conds, "q", control="control", profiles=prof, n_per_condition=30,
                                    same_profiles=False)
        self.assertGreater(fc2.effects.se[0], 0.0)

    def test_one_outcome_per_study(self):
        # study s1 has two outcomes; o1 is perfectly forecast, o2 is reversed. Pooling uses both every repeat,
        # the paper's protocol draws one per study, so per-repeat r is +1 or lower and n is one outcome's contrasts.
        rows = []
        for st in ("s1", "s2", "s3"):
            for o in ("o1", "o2"):
                for k, v in enumerate([0.0, 1.0, 3.0]):
                    rows.append({"study": st, "dv": o, "condition": k, "true_mean": v + (st == "s2"),
                                 "pred_mean": -v if o == "o2" else v, "true_se": 0.01})
        df = pd.DataFrame(rows)
        one = H.score_condition_forecasts(df, outcome="dv", repeats=16)
        self.assertEqual(one["n"], 6)                           # 3 studies x 2 contrasts
        self.assertEqual(one["n_units"], 6)
        pooled = H.score_condition_forecasts(df.assign(study=df.study + ":" + df.dv), repeats=16)
        self.assertEqual(pooled["n"], 12)
        # row order does not change the draw
        shuf = df.sample(frac=1, random_state=1)
        self.assertEqual(H.score_condition_forecasts(shuf, outcome="dv", repeats=4),
                         H.score_condition_forecasts(df, outcome="dv", repeats=4))

    def test_log_numpy_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = H.DecisionLog(Path(tmp) / "d.jsonl")
            log.log("x", arr=np.array([1.0, np.nan]), s=pd.Series([1, 2]))
            self.assertEqual(json.loads((Path(tmp) / "d.jsonl").read_text())["arr"], [1.0, None])


@unittest.skipUnless(PARQUET, NO_PARQUET)
@unittest.skipUnless(os.environ.get("SIMENGINE_DATA"), "set $SIMENGINE_DATA for real-data checks")
class TestRealData(unittest.TestCase):
    def test_socsci210_condition_protocol(self):
        """Hewitt protocol on the SocSci210 condition table: true means as the forecast give r = 1 and a
        reliability in (0, 1]."""
        from simengine.bench.base import DataNotFound, resolve_root
        try:
            root = resolve_root(None, "socsci210")
        except DataNotFound as e:
            self.skipTest(str(e))
        path = Path(root) / "ham" / "cond_stats.parquet"
        if not path.exists():
            self.skipTest("socsci210 cond_stats.parquet missing")
        c = pd.read_parquet(path, columns=["study_id", "task_num", "condition_num", "n", "mean", "sd"])
        c["exp"] = c.study_id + ":" + c.task_num.astype(str)
        c["true_se"] = c.sd / np.sqrt(c.n)
        s = H.score_condition_forecasts(c.assign(pred_mean=c["mean"]), study="exp", condition="condition_num",
                                        true="mean", repeats=2)
        self.assertGreater(s["n"], 100)
        self.assertAlmostEqual(s["r"], 1.0)
        self.assertTrue(0 < s["reliability"] <= 1)
        s1 = H.score_condition_forecasts(c.assign(pred_mean=c["mean"]), study="study_id", outcome="task_num",
                                         condition="condition_num", true="mean", repeats=2)
        self.assertLess(s1["n"], s["n"])                       # one outcome per study per repeat
        self.assertGreater(s1["n_studies"], 100)


if __name__ == "__main__":
    unittest.main()
