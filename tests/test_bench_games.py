"""bench tasks built on simengine.games: econ_games_b01, ertan_wtp, beebench_b01_prior. Offline (FakeLLM only).

The real-data check of beebench_b01_prior runs only when the b01 results table is present: $SIMENGINE_B01_DIR
(the directory holding results.csv), else <$SIMENGINE_TEST_DATA or $SIMENGINE_DATA>/beebench_b01/.
"""
import json
import os
import re
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine.games as gm
from simengine import bench
from simengine.bench.tasks import econ_games_b01 as E
from simengine.bench.tasks import ertan_wtp as W
from simengine.llm import FakeLLM


def rows(game, decisions, method="m", model="fake"):
    """decisions: {(agent, run, condition): decision} -> games.play-shaped rows."""
    return pd.DataFrame([{"model": model, "persona_method": method, "agent_id": a, "game": game,
                          "role": gm.GAMES[game]["role"], "condition": c, "wording_id": 0, "run": r, "decision": d,
                          "belief": None, "valid": True, "raw_len": 1, "cost_usd": 0.0}
                         for (a, r, c), d in decisions.items()])


def holdout_results(method="m", yes_low=True, n=4):
    """Synthetic results for the 4 referenced holdout games (and one calibration game)."""
    parts = [rows("beauty_contest", {(f"a{i}", 0, "base"): v for i, v in enumerate([30, 40, 70, 0][:n])},
                  method),
             rows("pg_punishment", {(f"a{i}", 0, c): v for i in range(n)
                                    for c, v in (("contribute", 10), ("punish_freerider", 3),
                                                 ("punish_cooperator", 1 if i == 0 else 0))}, method),
             rows("third_party_punishment", {(f"a{i}", 0, c): (5 if c != "gave50" and i % 2 else 0)
                                             for i in range(n) for c in ("gave0", "gave10", "gave25", "gave50")},
                  method),
             rows("who_to_punish", {(f"a{i}", 0, c): ("Yes" if (c == "ballot_low") == yes_low else "No")
                                    for i in range(n) for c in ("ballot_low", "ballot_avg", "ballot_high")}, method),
             rows("dictator", {(f"a{i}", 0, "base"): 20 for i in range(n)}, method)]
    return pd.concat(parts, ignore_index=True)


class TestEconGamesLoad(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.pop("SIMENGINE_DATA", None)

    def tearDown(self):
        if self._env is not None:
            os.environ["SIMENGINE_DATA"] = self._env

    def test_registered_and_loads_without_data(self):
        self.assertIn("econ_games_b01", bench.list())
        self.assertIn("ertan_wtp", bench.list())
        self.assertIn("beebench_b01_prior", bench.list())
        task = bench.get("econ_games_b01")
        d = task.load()                                     # no data_root, no $SIMENGINE_DATA
        refs = gm.human_refs()
        share = refs[refs.scale == "share"]
        self.assertEqual(len(d.train) + len(d.test), len(share))
        self.assertTrue(d.test["holdout"].all())
        self.assertFalse(d.train["holdout"].any())
        self.assertEqual(set(d.test["game"]), {"beauty_contest", "pg_punishment", "third_party_punishment",
                                               "who_to_punish"})
        row = d.test.set_index("unit").loc["who_to_punish:yes_share_low"]
        self.assertAlmostEqual(row["truth"], 0.641)
        self.assertIn("Table 2", row["citation"])
        self.assertEqual(task.describe()["kind"], "continuous")

    def test_options(self):
        d_all = bench.get("econ_games_b01", scale="all").load()
        self.assertIn("beauty_contest:mean_guess", set(d_all.test["unit"]))
        d = bench.get("econ_games_b01", split="all").load()
        self.assertEqual(len(d.train), 0)
        self.assertIn("dictator:mean_give_share", set(d.test["unit"]))
        with self.assertRaises(ValueError):
            bench.get("econ_games_b01", spilt="all")
        with self.assertRaises(ValueError):
            bench.get("econ_games_b01", split="nope").load()
        with self.assertRaises(ValueError):
            bench.get("econ_games_b01", scale="nope").load()


class TestEconGamesBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = bench.get("econ_games_b01", split="all", scale="all")
        cls.data = cls.task.load()
        base = cls.task.baselines(cls.data)
        cls.base = {k: dict(zip(cls.data.test["unit"], v)) for k, v in base.items()}

    def test_names(self):
        self.assertEqual(set(self.base), {"rational_corner", "midpoint"})     # no llm / results: no bare_model

    def test_rational_corner(self):
        c = self.base["rational_corner"]
        self.assertEqual(c["dictator:mean_give_share"], 0.0)
        self.assertEqual(c["dictator:share_give_zero"], 1.0)
        self.assertEqual(c["trust_sender:mean_sent_share"], 0.0)
        self.assertEqual(c["trust_returner:mean_return_share"], 0.0)
        self.assertEqual(c["beauty_contest:mean_guess"], 0.0)
        self.assertEqual(c["beauty_contest:share_zero"], 1.0)
        self.assertEqual(c["holt_laury:share_4_to_6_safe"], 1.0)            # risk neutral: 4 safe choices
        self.assertAlmostEqual(c["loss_gambles:implied_lambda_median"], 6 / 5)   # first EV-positive rung
        self.assertEqual(c["public_goods_strategy:share_free_riders"], 1.0)
        self.assertEqual(c["who_to_punish:yes_share_low"], 0.0)
        self.assertTrue(np.isnan(c["pg_punishment:perverse_event_share"]))   # no punishment: undefined, not 0

    def test_midpoint(self):
        m = self.base["midpoint"]
        self.assertEqual(m["dictator:share_equal_split"], 1.0)
        self.assertEqual(m["ultimatum_proposer:mean_offer_share"], 0.5)
        self.assertEqual(m["trust_returner:mean_return_share"], 0.5)
        self.assertEqual(m["beauty_contest:mean_guess"], 50.0)
        # uniform random over 10 Holt-Laury rows, exact: P(Bin(10, .5) > 4) = 638/1024
        self.assertAlmostEqual(m["holt_laury:share_more_than_4_safe"], 638 / 1024)
        self.assertAlmostEqual(m["who_to_punish:yes_share_low"], 1 / 3)
        self.assertAlmostEqual(m["ultimatum_responder:reject_rate_low"], 0.5)

    def test_deterministic(self):
        again = self.task.baselines(self.task.load())
        for k, v in again.items():
            np.testing.assert_array_equal(v, np.array([self.base[k][u] for u in self.data.test["unit"]]))


class TestEconGamesFrame(unittest.TestCase):
    def test_results_as_prediction_and_metrics(self):
        task = bench.get("econ_games_b01")
        data = task.load()
        res = holdout_results()
        fr = task.run({"mine": E.results_method(res)}, data=data)
        self.assertEqual(len(fr), len(data.test))
        self.assertEqual(set(fr["split"]), {"test"})
        v = fr.set_index("unit")["pred_mine"]
        m = gm.moments(res).set_index(["game", "moment"])["value"]
        self.assertAlmostEqual(v["who_to_punish:yes_share_low"], 1.0)
        self.assertAlmostEqual(v["beauty_contest:share_above_67"], m[("beauty_contest", "share_above_67")])
        self.assertAlmostEqual(v["pg_punishment:perverse_event_share"], 1 / 5)
        # the results table itself is accepted as the prediction, through games.moment_frame
        fr2 = task.frame(data, {"mine": res})
        np.testing.assert_allclose(fr2["pred_mine"].to_numpy(float), fr["pred_mine"].to_numpy(float))
        fr3 = task.run({"mine": res}, data=data, baselines=False)      # run() passes the table through too
        np.testing.assert_allclose(fr3["pred_mine"].to_numpy(float), fr["pred_mine"].to_numpy(float))
        s = task.score(fr, "mine", B=200)
        self.assertEqual(s["metric"], "mae")
        self.assertFalse(s["higher_is_better"])
        self.assertEqual(s["n"], 4)                                     # four holdout games = clusters
        err = np.abs(fr["pred_mine"] - fr["truth"]).mean()
        self.assertAlmostEqual(s["value"], err)
        self.assertLessEqual(s["lo"], s["value"])
        self.assertGreaterEqual(s["hi"], s["value"])

    def test_perfect_prediction_and_missing_moments(self):
        task = bench.get("econ_games_b01")
        data = task.load()
        truth = data.test["truth"].to_numpy(float)
        part = truth.copy()
        part[:2] = np.nan                                              # moments the method could not compute
        fr = task.frame(data, {"exact": truth, "part": part})
        self.assertEqual(task.score(fr, "exact", B=100)["value"], 0.0)
        self.assertEqual(task.score(fr, "exact", "rmse", B=100)["value"], 0.0)
        cov = task.score(fr, "part", "coverage")
        self.assertAlmostEqual(cov["value"], (len(truth) - 2) / len(truth))
        self.assertEqual(task.score(fr, "part", B=100)["value"], 0.0)  # NaN rows excluded, never filled
        empty = task.frame(data, {"none": rows("dictator", {("a", 0, "base"): 10})})
        self.assertTrue(empty["pred_none"].isna().all())               # no holdout game played
        self.assertTrue(np.isnan(task.score(empty, "none")["value"]))

    def test_several_persona_methods_expand(self):
        task = bench.get("econ_games_b01")
        data = task.load()
        res = pd.concat([holdout_results("good"), holdout_results("bad", yes_low=False)], ignore_index=True)
        fr = task.frame(data, {"b01": res})
        self.assertIn("pred_b01_good", fr.columns)
        self.assertIn("pred_b01_bad", fr.columns)
        with self.assertRaises(ValueError):
            E.results_method(res)(data)
        good = E.results_method(res, "good")(data)
        np.testing.assert_allclose(good, fr["pred_b01_good"].to_numpy(float))

    def test_results_frame_feeds_evaluate(self):
        import simengine.evidence as ev
        task = bench.get("econ_games_b01")
        res = pd.concat([holdout_results("good"), holdout_results("bad", yes_low=False)], ignore_index=True)
        fr = task.results_frame(res)
        self.assertEqual(set(fr["split"]), {"train", "test"})
        self.assertNotIn("item", fr.columns)
        self.assertTrue({"pred_good", "pred_bad", "pred_rational_corner", "pred_midpoint"} <= set(fr.columns))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = ev.evaluate(fr.dropna(), ours="good", task="continuous", B=50, B_aggregate=20)
            fr_test = task.run({"good": E.results_method(res, "good")})
            rep2 = ev.evaluate(fr_test, ours="good", task="continuous", B=50, B_aggregate=20)
        self.assertIn("headline_en", ev.summary(rep))
        self.assertIn("headline_en", ev.summary(rep2))

    def test_leaderboard_runs_without_data(self):
        tab = bench.run(["econ_games_b01", "ertan_wtp"], B=50)
        self.assertTrue((tab["status"] == "ok").all())
        md = bench.to_markdown(tab)
        self.assertIn("### econ_games_b01", md)
        self.assertIn("### ertan_wtp", md)


def game_responder(answers, belief=8):
    """FakeLLM responder: identifies (game, condition) from the rendered text; belief prompts get `belief`."""
    dec, bel = {}, {}
    for name in answers:
        for c in gm.GAMES[name]["conditions"]:
            dec[gm.render(name, c["id"])] = answers[name](c["id"])
            b = gm.render_belief(name, c["id"])
            if b:
                bel[b] = belief

    def respond(system, user):
        for t, v in dec.items():
            if t in user:
                return json.dumps({"answer": v})
        for t, v in bel.items():
            if t in user:
                return json.dumps({"answer": v})
        return "no idea"
    return respond


HOLDOUT_ANSWERS = {
    "beauty_contest": lambda c: 35,
    "pg_punishment": lambda c: {"contribute": 12, "punish_freerider": 4, "punish_cooperator": 0}[c],
    "third_party_punishment": lambda c: 0 if c == "gave50" else 6,
    "who_to_punish": lambda c: "Yes" if c == "ballot_low" else "No",
}


class TestEconGamesLLM(unittest.TestCase):
    def test_play_method_with_fake_llm(self):
        task = bench.get("econ_games_b01")
        data = task.load()
        llm = FakeLLM(game_responder(HOLDOUT_ANSWERS))
        f = E.play_method(llm, {"persona": {"p-000": "You are careful.", "p-001": "You are bold."}}, runs=2)
        v = dict(zip(data.test["unit"], f(data)))
        self.assertAlmostEqual(v["who_to_punish:yes_share_low"], 1.0)
        self.assertAlmostEqual(v["pg_punishment:contribute_share"], 0.6)
        self.assertAlmostEqual(v["pg_punishment:perverse_event_share"], 0.0)
        self.assertAlmostEqual(v["third_party_punishment:share_punish_below_50"], 1.0)
        self.assertTrue(all(s in ("You are careful.", "You are bold.") for s, _ in llm.calls))

    def test_bare_model_from_llm_and_from_results(self):
        llm = FakeLLM(game_responder(HOLDOUT_ANSWERS))
        task = bench.get("econ_games_b01", llm=llm, bare_agents=2, runs=1)
        data = task.load()
        base = task.baselines(data)
        self.assertIn("bare_model", base)
        self.assertTrue(all(s == "" for s, _ in llm.calls))           # no persona: empty system prompt
        self.assertAlmostEqual(dict(zip(data.test["unit"], base["bare_model"]))["who_to_punish:yes_share_low"], 1.0)
        n_calls = len(llm.calls)
        res = pd.concat([holdout_results("baseline", yes_low=False), holdout_results("big5")], ignore_index=True)
        t2 = bench.get("econ_games_b01", results=res)
        b2 = t2.baselines(t2.load())
        self.assertAlmostEqual(dict(zip(t2.load().test["unit"], b2["bare_model"]))["who_to_punish:yes_share_low"],
                               0.0)
        self.assertEqual(len(llm.calls), n_calls)
        t3 = bench.get("econ_games_b01", results=holdout_results("big5"))
        self.assertNotIn("bare_model", t3.baselines(t3.load()))

    def test_cost_gate_applies(self):
        llm = FakeLLM(game_responder(HOLDOUT_ANSWERS))
        task = bench.get("econ_games_b01", llm=llm, bare_agents=1, runs=1,
                         play_kw={"est_cost_per_call": 1.0})
        with self.assertRaises(gm.CostGateError):
            task.baselines(task.load())
        self.assertEqual(llm.calls, [])


# ---------------------------------------------------------------------- ertan_wtp

def human_like_run():
    """Votes no-punishment first, then punish-low; period-1 at 0.73 of the endowment with SD 3; contributions rise
    under punish-low; punishment 1:1 to the shortfall, aimed only at below-average players."""
    c1 = {"A": 10, "B": 10, "C": 2.5, "D": 6.7}         # mean 7.3, SD about 3.08
    periods = []
    for k in range(3):
        periods.append({"contrib": dict(c1) if k == 0 else {"A": 9, "B": 8, "C": 5, "D": 4},
                        "received": {L: 0.0 for L in "ABCD"}, "punish_low": False})
    events = []
    for k in range(3):
        c = {"A": 10, "B": 10, "C": 9 + k * 0.3, "D": 8 + k * 0.5}
        avg = sum(c.values()) / 4
        rec = {L: (avg - v if v < avg else 0.0) for L, v in c.items()}
        periods.append({"contrib": c, "received": rec, "punish_low": True})
        events += [{"target_contribution": v, "avg": avg, "amount": rec[L]} for L, v in c.items() if rec[L] > 0]
    return {"votes": [{"punish_low": False, "punish_high": False}, {"punish_low": True, "punish_high": False}],
            "periods": periods, "punish_events": events}


def society_responder(system, user):
    """Vote 1: nothing passes; vote 2: punish-low passes. Free rider gives 2, others 8 (10 once punish-low is in).
    Every allowed punishment is $2."""
    if "Ballot item" in user:
        m = re.search(r"who contributed (LESS than|exactly|MORE than) the group average\?", user)
        second = "Vote 1 outcome" in user
        return json.dumps({"vote": "Yes" if (second and m.group(1) == "LESS than") else "No"})
    if "How much of your $10" in user:
        if system == "A free rider.":
            return json.dumps({"contribution": 2})
        return json.dumps({"contribution": 10 if "Vote 1 outcome" in user else 8})
    if "reduce their earnings" in user:
        return '{"punish": 2}'
    raise AssertionError(user)


class TestErtanWTP(unittest.TestCase):
    def test_loads_without_data(self):
        task = bench.get("ertan_wtp")
        d = task.load()
        self.assertEqual(list(d.test["unit"]), [n for n, _ in gm.CRITERIA])
        self.assertTrue((d.test["truth"] == 1.0).all())
        self.assertEqual(dict(zip(d.test["unit"], d.test["weight"])), dict(gm.CRITERIA))
        self.assertIn("p.10", d.test.set_index("unit").loc["never_punish_high", "target"])
        self.assertEqual(len(d.train), 0)

    def test_human_like_run_scores_one(self):
        task = bench.get("ertan_wtp")
        data = task.load()
        run = human_like_run()
        s = gm.wtp_scorecard(run)
        self.assertAlmostEqual(s["index"], 1.0)
        fr = task.frame(data, {"human": run, "human_list": [run, run]})
        self.assertTrue((fr["pred_human"] == 1.0).all())
        r = task.score(fr, "human", B=100)
        self.assertEqual((r["metric"], r["value"], r["n"]), ("index", 1.0, 9))
        self.assertEqual(task.score(fr, "human", "wmae", B=100)["value"], 0.0)
        np.testing.assert_allclose(fr["pred_human_list"], fr["pred_human"])

    def test_na_criteria_excluded_and_index_matches_scorecard(self):
        task = bench.get("ertan_wtp")
        data = task.load()
        base = task.baselines(data)
        corner = dict(zip(data.test["unit"], base["rational_corner"]))
        for n in ("treatment_effect", "sustain_under_punish_low", "punishment_usage", "punishment_targeting"):
            self.assertTrue(np.isnan(corner[n]), n)                  # N/A stays NaN, never 0
        fr = task.frame(data, base)
        r = task.score(fr, "rational_corner", B=50)
        s = gm.wtp_scorecard(W.synthetic_run(0, False))
        self.assertAlmostEqual(r["value"], s["index"])
        self.assertEqual(r["n"], s["n_criteria"])
        self.assertAlmostEqual(task.score(fr, "rational_corner", "coverage")["value"], s["n_criteria"] / 9)
        self.assertAlmostEqual(task.score(fr, "rational_corner", "wmae", B=50)["value"], 1 - s["index"])

    def test_criterion_means_over_runs(self):
        a = W.synthetic_run(0, False)             # treatment_effect N/A
        b = human_like_run()
        m = W.criterion_means([a, b])
        self.assertAlmostEqual(m["treatment_effect"], gm.wtp_scorecard(b)["treatment_effect"])
        self.assertAlmostEqual(m["final_vote_punish_low"], 0.5)
        sc = W.scorecards([a, b, {"votes": [], "periods": []}])
        self.assertEqual(len(sc), 2)
        self.assertEqual(sc.attrs["n_unscorable"], 1)

    def test_society_method_and_group_conversion(self):
        llm = FakeLLM(society_responder)
        personas = ["A steady cooperator."] * 3 + ["A free rider."]
        task = bench.get("ertan_wtp")
        data = task.load()
        f = W.society_method(llm, personas, seeds=(0, 1))
        pred = dict(zip(data.test["unit"], f(data)))
        self.assertEqual(pred["final_vote_punish_low"], 1.0)
        self.assertEqual(pred["no_dismantling"], 1.0)
        self.assertEqual(pred["never_punish_high"], 1.0)              # punish_high taken from the group rules
        self.assertEqual(pred["punishment_targeting"], 1.0)           # all $ on the free rider
        from simengine.abm.society import run_who_to_punish
        out = run_who_to_punish(FakeLLM(society_responder), personas, seed=0)
        run = W.group_to_run(out["groups"][0])
        self.assertEqual(run["votes"][1], {"punish_low": True, "punish_high": False})
        self.assertEqual(len(run["punish_events"]), 9)
        fr = task.frame(data, {"soc": out})
        np.testing.assert_allclose(fr["pred_soc"].to_numpy(float), np.array([pred[u] for u in fr["unit"]]))
        fr2 = task.run({"soc": out}, data=data)
        np.testing.assert_allclose(fr2["pred_soc"].to_numpy(float), fr["pred_soc"].to_numpy(float))
        self.assertTrue(all(s in personas for s, _ in llm.calls))

    def test_frame_feeds_evaluate(self):
        import simengine.evidence as ev
        task = bench.get("ertan_wtp")
        data = task.load()
        fr = task.frame(data, {"good": np.linspace(0.6, 1.0, 9), "bad": np.linspace(0.0, 0.4, 9)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = ev.evaluate(fr, ours="good", task="continuous", B=50, B_aggregate=20)
        self.assertIn("headline_en", ev.summary(rep))


# ---------------------------------------------------------------------- beebench_b01_prior

class TestB01Prior(unittest.TestCase):
    def test_skips_when_absent(self):
        env = os.environ.pop("SIMENGINE_DATA", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(bench.DataNotFound):
                    bench.get("beebench_b01_prior").load(tmp)
            with self.assertRaises(bench.DataNotFound):
                bench.get("beebench_b01_prior").load()
            tab = bench.run(["beebench_b01_prior"], B=20)
            self.assertEqual(list(tab["status"]), ["missing_data"])
        finally:
            if env is not None:
                os.environ["SIMENGINE_DATA"] = env

    def test_synthetic_file_in_layout(self):
        res = pd.concat([holdout_results("baseline", yes_low=False), holdout_results("big5")], ignore_index=True)
        res.loc[0, "valid"] = False
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "beebench_b01"))
            res.assign(timestamp="t").to_csv(os.path.join(tmp, "beebench_b01", "results.csv"), index=False)
            task = bench.get("beebench_b01_prior")
            data = task.load(tmp)
            self.assertEqual(data.meta["n_rows"], len(res))
            self.assertEqual(data.meta["n_invalid"], 1)
            fr = task.run(data=data)
            self.assertTrue({"pred_bare_model", "pred_b01_big5", "pred_rational_corner", "pred_midpoint"}
                            <= set(fr.columns))
            v = fr.set_index("unit")
            self.assertEqual(v.loc["who_to_punish:yes_share_low", "pred_b01_big5"], 1.0)
            self.assertEqual(v.loc["who_to_punish:yes_share_low", "pred_bare_model"], 0.0)
            tab = bench.run(["beebench_b01_prior"], data_root=tmp, B=20)
            self.assertTrue((tab["status"] == "ok").all())
            rf = task.results_frame(data=data)
            self.assertIn("pred_b01_big5", rf.columns)
            self.assertEqual(rf.columns[-1], "split")
            only = bench.get("beebench_b01_prior", methods=["big5"]).load(tmp)
            self.assertEqual(only.meta["persona_methods"], ["big5"])


def _b01_dir():
    d = os.environ.get("SIMENGINE_B01_DIR")
    if d:
        return d
    root = os.environ.get("SIMENGINE_TEST_DATA") or os.environ.get("SIMENGINE_DATA")
    return os.path.join(root, "beebench_b01") if root else None


@unittest.skipUnless(_b01_dir() and os.path.exists(os.path.join(_b01_dir(), "results.csv")),
                     "BeeBench b01 results.csv not present")
class TestB01PriorReal(unittest.TestCase):
    def test_prior_file(self):
        task = bench.get("beebench_b01_prior")
        data = task.load(_b01_dir())
        # edsl-test benchmark/beebench/b01 README / ANOMALIES.md: 10,920 rows, 30 invalid, one model
        self.assertEqual(data.meta["n_rows"], 10920)
        self.assertEqual(data.meta["n_invalid"], 30)
        self.assertEqual(len(data.meta["persona_methods"]), 7)
        fr = task.run(data=data)
        for m in ("bare_model", "b01_big5", "b01_twin2k"):
            self.assertEqual(task.score(fr, m, "coverage")["value"], 1.0)


if __name__ == "__main__":
    unittest.main()
