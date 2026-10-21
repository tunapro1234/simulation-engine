"""simengine.games scoring: moments, distance to human, EvalFrame bridge, coherence audit, WTP scorecard."""
import unittest

import numpy as np
import pandas as pd

import simengine.games as gm


def rows(game, decisions, method="m", model="fake", valid=True):
    """decisions: {(agent, run, condition): decision}."""
    out = []
    for (a, r, c), d in decisions.items():
        out.append({"model": model, "persona_method": method, "agent_id": a, "game": game,
                    "role": gm.GAMES[game]["role"], "condition": c, "wording_id": 0, "run": r, "decision": d,
                    "belief": None, "valid": valid if not callable(valid) else valid(d), "raw_len": 10,
                    "cost_usd": 0.0})
    return pd.DataFrame(out)


def eleven_twenty_human(n_scale=1, method="m"):
    dist = gm.HUMAN_DISTRIBUTIONS["eleven_twenty"]
    dec, i = {}, 0
    for v, p in zip(dist["support"], dist["p"]):
        for _ in range(int(round(p * 100)) * n_scale):
            dec[(f"a{i:03d}", 0, "base")] = v
            i += 1
    return rows("eleven_twenty", dec, method=method)


class TestMoments(unittest.TestCase):
    def test_basic_moments(self):
        df = rows("dictator", {("a", 0, "base"): 0, ("b", 0, "base"): 50, ("c", 0, "base"): 25,
                               ("d", 0, "base"): 999}, valid=lambda d: d <= 100)
        m = gm.moments(df).set_index("moment")
        self.assertAlmostEqual(m.loc["mean_give_share", "value"], 0.25)
        self.assertAlmostEqual(m.loc["share_give_zero", "value"], 1 / 3)
        self.assertAlmostEqual(m.loc["share_equal_split", "value"], 1 / 3)
        self.assertEqual(m.loc["mean_give_share", "n_rows"], 4)
        self.assertEqual(m.loc["mean_give_share", "n_valid"], 3)

    def test_holt_laury_and_loss_gambles(self):
        hl = {("a", 0, f"row{k}"): ("Option A" if k <= 6 else "Option B") for k in range(1, 11)}
        hl.update({("b", 0, f"row{k}"): ("Option A" if k <= 4 else "Option B") for k in range(1, 11)})
        hl[("c", 0, "row1")] = "Option A"                     # incomplete schedule: not counted
        m = gm.moments(rows("holt_laury", hl)).set_index("moment")["value"]
        self.assertEqual(m["mean_safe_choices"], 5.0)
        self.assertEqual(m["share_more_than_4_safe"], 0.5)
        gains = [c["id"] for c in gm.GAMES["loss_gambles"]["conditions"]]
        lg = {("a", 0, c): ("Accept" if gm.GAMES["loss_gambles"]["conditions"][i]["gain"] >= 12 else "Reject")
              for i, c in enumerate(gains)}
        lg.update({("b", 0, c): "Reject" for c in gains})      # never accepts: censored at the top of the ladder
        m = gm.moments(rows("loss_gambles", lg)).set_index("moment")["value"]
        self.assertAlmostEqual(m["implied_lambda_median"], (2.4 + 4.0) / 2)
        self.assertEqual(m["share_never_accept"], 0.5)

    def test_schedule_types(self):
        self.assertEqual(gm.classify_schedule([0, 0, 0, 0, 0]), "free_rider")
        self.assertEqual(gm.classify_schedule([0, 5, 10, 15, 20]), "conditional")
        self.assertEqual(gm.classify_schedule([0, 5, 10, 5, 0]), "hump")
        self.assertEqual(gm.classify_schedule([10, 10, 10, 10, 10]), "other")
        self.assertEqual(gm.classify_schedule([20, 0, 20, 0, 20]), "other")

    def test_third_party_slope(self):
        dec = {}
        for i in range(3):
            for c, g in (("gave0", 0), ("gave10", 10), ("gave25", 25), ("gave50", 50)):
                dec[(f"a{i}", 0, c)] = 0.28 * (50 - g)
        m = gm.moments(rows("third_party_punishment", dec)).set_index("moment")["value"]
        self.assertAlmostEqual(m["punish_slope"], 0.28)
        self.assertAlmostEqual(m["mean_spend_gave0"], 14.0)
        self.assertAlmostEqual(m["share_punish_below_50"], 1.0)


class TestDistance(unittest.TestCase):
    def test_zero_when_moments_equal_published_refs(self):
        et = eleven_twenty_human()
        up = rows("ultimatum_proposer", {(f"a{i}", 0, "base"): 40 for i in range(10)})
        wtp = rows("who_to_punish", {(f"a{i:04d}", 0, "ballot_low"): ("Yes" if i < 641 else "No")
                                     for i in range(1000)})
        d = gm.distance_to_human(pd.concat([et, up, wtp], ignore_index=True), B=50)
        got = d.set_index(["game", "moment", "metric"])["abs_error"]
        for key in [("eleven_twenty", "share_17_to_19", "abs_error"), ("eleven_twenty", "share_17_to_20", "abs_error"),
                    ("eleven_twenty", "share_choose_20", "abs_error"), ("eleven_twenty", "mean_request", "abs_error"),
                    ("eleven_twenty", "choice_distribution", "wasserstein"),
                    ("eleven_twenty", "choice_distribution", "ks"),
                    ("ultimatum_proposer", "mean_offer_share", "abs_error"),
                    ("who_to_punish", "yes_share_low", "abs_error")]:
            self.assertAlmostEqual(got[key], 0.0, places=9, msg=key)
        # moments with no data in these rows stay missing, never 0
        self.assertTrue(np.isnan(got[("who_to_punish", "yes_share_high", "abs_error")]))

    def test_zero_against_own_moments_every_game(self):
        rng = np.random.default_rng(1)
        parts = []
        for name, g in gm.GAMES.items():
            dec = {}
            for a in range(4):
                for c in g["conditions"]:
                    if g["question_type"] == "mc":
                        dec[(f"a{a}", 0, c["id"])] = g["options"][int(rng.integers(len(g["options"])))]
                    else:
                        lo, hi = gm.bounds(g, c["id"])
                        dec[(f"a{a}", 0, c["id"])] = int(rng.integers(lo, hi + 1))
            parts.append(rows(name, dec))
        res = pd.concat(parts, ignore_index=True)
        m = gm.moments(res)
        self.assertEqual(set(m["game"]), set(gm.GAMES))
        refs = m.rename(columns={"value": "ref"})[["game", "moment", "ref"]].dropna()
        refs = refs.rename(columns={"ref": "value"}).assign(citation="self", scale="share")
        d = gm.distance_to_human(res, refs=refs, B=0, distributions={})
        self.assertEqual(len(d), len(refs))
        self.assertTrue(np.allclose(d["abs_error"], 0.0))

    def test_bootstrap_ci_resamples_agents(self):
        df = rows("dictator", {(f"a{i}", r, "base"): (10 if i < 5 else 50) for i in range(10) for r in range(3)})
        d = gm.distance_to_human(df, B=300, seed=0).set_index("moment")
        r = d.loc["mean_give_share"]
        self.assertAlmostEqual(r["sim"], 0.30)
        self.assertLessEqual(r["ci_lo"], r["abs_error"])
        self.assertGreaterEqual(r["ci_hi"], r["abs_error"])
        self.assertGreater(r["ci_hi"] - r["ci_lo"], 0)
        self.assertEqual(r["n_agents"], 10)
        d2 = gm.distance_to_human(df, B=300, seed=0)
        pd.testing.assert_frame_equal(d.reset_index(), d2.set_index("moment").reset_index())

    def test_distribution_distance_positive_when_off(self):
        df = rows("eleven_twenty", {(f"a{i}", 0, "base"): 20 for i in range(10)})
        d = gm.distance_to_human(df, B=0).set_index(["moment", "metric"])["abs_error"]
        self.assertGreater(d[("choice_distribution", "wasserstein")], 2.0)
        self.assertAlmostEqual(d[("choice_distribution", "ks")], 0.94)

    def test_distances_match_scipy(self):
        from scipy.stats import ks_2samp, wasserstein_distance
        rng = np.random.default_rng(3)
        sup, p = [11, 13, 17, 20], [0.1, 0.2, 0.4, 0.3]
        for _ in range(5):
            x = rng.integers(10, 22, 25).astype(float)
            self.assertAlmostEqual(gm.wasserstein_to(x, sup, p), wasserstein_distance(x, sup, v_weights=p))
            y = np.repeat(sup, [10, 20, 40, 30]).astype(float)
            self.assertAlmostEqual(gm.ks_to(x, sup, p), ks_2samp(x, y).statistic)
        self.assertTrue(np.isnan(gm.wasserstein_to([], sup, p)))

    def test_human_refs_have_citations(self):
        r = gm.human_refs()
        self.assertEqual(list(r.columns), ["game", "moment", "value", "citation", "scale", "holdout"])
        self.assertTrue((r["citation"].str.len() > 5).all())
        self.assertTrue(set(r["game"]) <= set(gm.GAMES))
        self.assertAlmostEqual(sum(gm.HUMAN_DISTRIBUTIONS["eleven_twenty"]["p"]), 1.0)
        for g, mom in zip(r["game"], r["moment"]):      # every ref names a moment the game computes
            self.assertIn(mom, gm.MOMENTS[g](rows(g, {("a", 0, gm.GAMES[g]["conditions"][0]["id"]): None})
                                             .assign(_agent="a")))


class TestMomentFrame(unittest.TestCase):
    def test_frame_feeds_evaluate(self):
        parts = []
        for meth, offer, give in (("good", 40, 28), ("bad", 0, 0)):
            parts.append(rows("ultimatum_proposer", {(f"{meth}{i}", 0, "base"): offer for i in range(5)}, method=meth))
            parts.append(rows("dictator", {(f"{meth}{i}", 0, "base"): give for i in range(5)}, method=meth))
            parts.append(rows("beauty_contest", {(f"{meth}{i}", 0, "base"): (30 if meth == "good" else 90)
                                                 for i in range(5)}, method=meth))
            parts.append(rows("who_to_punish", {(f"{meth}{i}", 0, c): ("Yes" if (meth == "good") == (c == "ballot_low")
                                                                       else "No")
                                                for i in range(5) for c in ("ballot_low", "ballot_avg", "ballot_high")},
                              method=meth))
        res = pd.concat(parts, ignore_index=True)
        f = gm.moment_frame(res)
        self.assertNotIn("item", f.columns)
        self.assertEqual(set(f.columns), {"unit", "group", "truth", "pred_good", "pred_bad", "split"})
        self.assertIn("dictator:mean_give_share", set(f["unit"]))
        self.assertEqual(set(f.loc[f.group == "who_to_punish", "split"]), {"test"})
        self.assertEqual(set(f.loc[f.group == "dictator", "split"]), {"train"})
        self.assertNotIn("beauty_contest:mean_guess", set(f["unit"]))          # raw scale left out
        self.assertIn("beauty_contest:mean_guess", set(gm.moment_frame(res, include_raw=True)["unit"]))
        self.assertIn("item", gm.moment_frame(res, item=True).columns)
        import warnings
        import simengine.evidence as ev
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # constant synthetic predictions -> undefined correlation
            rep = ev.evaluate(f.dropna(), ours="good", task="continuous", B=50, B_aggregate=20)
        self.assertIn("headline_en", ev.summary(rep))

    def test_several_models_need_model_argument(self):
        a = rows("dictator", {("a", 0, "base"): 10}, model="m1")
        b = rows("dictator", {("a", 0, "base"): 20}, model="m2")
        with self.assertRaises(ValueError):
            gm.moment_frame(pd.concat([a, b]))
        f = gm.moment_frame(pd.concat([a, b]), model="m2")
        self.assertAlmostEqual(f.set_index("unit").loc["dictator:mean_give_share", "pred_m"], 0.2)


class TestCoherence(unittest.TestCase):
    CASES = {
        # flag: (game, bad decisions, clean decisions)
        "hl_multi_switch": ("holt_laury",
                            {f"row{k}": ("Option A" if k % 2 else "Option B") for k in range(1, 11)},
                            {f"row{k}": ("Option A" if k <= 5 else "Option B") for k in range(1, 11)}),
        "lg_non_monotone": ("loss_gambles",
                            {c["id"]: ("Accept" if c["gain"] == 4 else "Reject")
                             for c in gm.GAMES["loss_gambles"]["conditions"]},
                            {c["id"]: ("Accept" if c["gain"] >= 10 else "Reject")
                             for c in gm.GAMES["loss_gambles"]["conditions"]}),
        "tc_future_bias": ("time_choices",
                           {"now110": "Later payment", "later110": "Sooner payment"},
                           {"now110": "Sooner payment", "later110": "Later payment"}),
        "ur_incoherent": ("ultimatum_responder",
                          {"low": "Accept", "mid": "Reject", "fair": "Accept"},
                          {"low": "Reject", "mid": "Accept", "fair": "Accept"}),
        "tr_overreturn": ("trust_returner", {"low": 35, "mid": 30}, {"low": 10, "mid": 30}),
        "bc_dominated": ("beauty_contest", {"base": 80}, {"base": 33}),
        "pp_perverse": ("pg_punishment",
                        {"contribute": 10, "punish_freerider": 1, "punish_cooperator": 4},
                        {"contribute": 10, "punish_freerider": 4, "punish_cooperator": 0}),
        "tpp_inverted": ("third_party_punishment", {"gave0": 0, "gave50": 5}, {"gave0": 10, "gave50": 0}),
    }

    def _df(self, game, dec, agents=("a", "b"), method="m", valid=True):
        return rows(game, {(a, 0, c): v for a in agents for c, v in dec.items()}, method=method, valid=valid)

    def test_each_detector_fires_and_stays_silent(self):
        for flag, (game, bad, clean) in self.CASES.items():
            with self.subTest(flag=flag):
                valid = (lambda d: d <= 30) if game == "trust_returner" else True
                a = gm.coherence_audit(self._df(game, bad, agents=("a",), valid=valid))
                self.assertIn(flag, set(a["flag"]))
                self.assertTrue(a.loc[a.flag == flag, "precedent"].str.len().gt(0).all())
                a = gm.coherence_audit(self._df(game, clean, agents=("a",)))
                self.assertNotIn(flag, set(a["flag"]))

    def test_invalid_rate_and_zero_variance(self):
        df = self._df("dictator", {"base": 0}, agents=[f"a{i}" for i in range(8)])
        df.loc[:1, "valid"] = False
        df.loc[:1, "decision"] = None
        a = gm.coherence_audit(df)
        self.assertEqual(set(a["flag"]), {"invalid_rate", "zero_variance"})
        base = self._df("dictator", {"base": 0}, agents=[f"a{i}" for i in range(8)], method="baseline")
        self.assertEqual(len(gm.coherence_audit(base)), 0)            # baseline exempt from zero_variance
        mixed = rows("dictator", {(f"a{i}", 0, "base"): i * 5 for i in range(8)})
        self.assertEqual(len(gm.coherence_audit(mixed)), 0)

    def test_flags_never_drop_rows(self):
        df = pd.concat([self._df("beauty_contest", {"base": 80}, agents=("a",)),
                        self._df("beauty_contest", {"base": 30}, agents=("b",))], ignore_index=True)
        marked = gm.flag_rows(df)
        self.assertEqual(len(marked), len(df))
        self.assertEqual(marked.set_index("agent_id").loc["a", "coherence_flags"], "bc_dominated")
        self.assertEqual(marked.set_index("agent_id").loc["b", "coherence_flags"], "")


def human_like_run():
    """3 votes: no punishment, then punish-low twice; contributions in the human band and rising under
    punish-low; punishment of the low contributor ~1:1 to the shortfall; all punishment aimed low."""
    p = lambda c, rec, pl: {"contrib": c, "received": rec, "punish_low": pl}  # noqa: E731
    zero = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
    periods = [p({"a": 10, "b": 10, "c": 2, "d": 8}, zero, False),        # mean 7.5 = 0.75, sd 3.28
               p({"a": 9, "b": 8, "c": 1, "d": 6}, zero, False),
               p({"a": 10, "b": 9, "c": 6, "d": 9}, {"a": 0, "b": 0, "c": 2.5, "d": 0}, True),   # mean 8.5
               p({"a": 10, "b": 10, "c": 8, "d": 10}, {"a": 0, "b": 0, "c": 1.5, "d": 0}, True)]  # mean 9.5
    votes = [{"punish_low": False, "punish_high": False}, {"punish_low": True, "punish_high": False},
             {"punish_low": True, "punish_high": False}]
    events = [{"target_contribution": 6, "avg": 8.5, "amount": 2.5}, {"target_contribution": 8, "avg": 9.5,
                                                                       "amount": 1.5}]
    return {"endowment": 10, "votes": votes, "periods": periods, "punish_events": events}


class TestWTP(unittest.TestCase):
    def test_human_like_run_scores_one(self):
        s = gm.wtp_scorecard(human_like_run())
        for name, _ in gm.CRITERIA:
            self.assertEqual(s[name], 1.0, name)
        self.assertEqual(s["index"], 1.0)
        self.assertEqual(s["n_criteria"], 9)

    def test_na_criteria_excluded_not_zero_filled(self):
        run = human_like_run()
        run["votes"] = [{"punish_low": False, "punish_high": False}] * 2
        for p in run["periods"]:
            p["punish_low"] = False
        s = gm.wtp_scorecard(run)
        for name in ("treatment_effect", "sustain_under_punish_low", "punishment_usage", "punishment_targeting"):
            self.assertIsNone(s[name])
        self.assertEqual(s["final_vote_punish_low"], 0.0)
        w = dict(gm.CRITERIA)
        observed = {k: s[k] for k in w if s[k] is not None}
        expect = sum(w[k] * v for k, v in observed.items()) / sum(w[k] for k in observed)
        self.assertAlmostEqual(s["index"], expect)
        self.assertEqual(s["n_criteria"], 5)
        # zero-filling would have given a lower index
        self.assertGreater(s["index"], sum(w[k] * v for k, v in observed.items()) / sum(w.values()))

    def test_violations_lower_the_score(self):
        run = human_like_run()
        run["votes"][1]["punish_high"] = True
        run["votes"][2] = {"punish_low": False, "punish_high": False}
        s = gm.wtp_scorecard(run)
        self.assertEqual(s["never_punish_high"], 0.0)
        self.assertEqual(s["final_vote_punish_low"], 0.0)
        self.assertEqual(s["no_dismantling"], 1.0)        # first vote was no-punishment, so not dismantled
        self.assertLess(s["index"], 1.0)
        run = human_like_run()
        run["punish_events"] = [{"target_contribution": 10, "avg": 8.5, "amount": 3.0}]
        self.assertEqual(gm.wtp_scorecard(run)["punishment_targeting"], 0.0)
        run["punish_events"] = []
        self.assertIsNone(gm.wtp_scorecard(run)["punishment_targeting"])

    def test_regimes_from_votes_and_bool_votes(self):
        run = human_like_run()
        run["votes"] = [False, True]
        for p in run["periods"]:
            del p["punish_low"]
        s = gm.wtp_scorecard(run)
        self.assertIsNone(s["never_punish_high"])         # not recorded -> N/A
        self.assertEqual(s["treatment_effect"], 1.0)
        self.assertEqual(s["n_criteria"], 8)

    def test_targets_carry_cites(self):
        for k, (v, cite) in gm.ERTAN_TARGETS.items():
            self.assertTrue(cite and isinstance(v, (int, float)), k)
        self.assertEqual(gm.WTP_PARAMS["period1_band"], (0.69, 0.77))


if __name__ == "__main__":
    unittest.main()
