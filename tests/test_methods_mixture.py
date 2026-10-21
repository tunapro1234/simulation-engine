"""Theory-persona mixture calibration: simplex grid fit, identifiability, held-out validation, LLM probes."""
import json
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.llm import FakeLLM
from simengine.methods import (CR_GAMES, CR_HUMAN_LEFT, ENGEL_DICTATOR, HUMANIZING_PREFIX, THEORY_TYPES,
                               MixtureFit, UnidentifiedWarning, charness_rabin_games, choice_game, dictator_game,
                               fit_mixture_weights, identifiability, mixture, numeric_game, probe_types)

# edsl-test personas/homo_silicus/weights.json (gpt-oss-120b, 10 reps per cell): measured type behaviour.
SRC_CR = {
    "self_interested": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "efficient": [0.0, 0.0, 1.0, 0.0, 0.0, 0.8],
    "inequity_averse": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "competitive": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    "reciprocal": [0.9, 1.0, 0.4, 0.0, 0.0, 0.0],
    "selfish_but_fair": [0.8, 1.0, 1.0, 0.6, 0.1, 0.0],
}
SRC_DICTATOR = {
    "self_interested": [0.0, 1.0, 0.0], "efficient": [0.0, 1.0, 0.0], "inequity_averse": [0.5, 0.0, 1.0],
    "competitive": [0.4, 0.6, 0.0], "reciprocal": [0.05, 0.9, 0.1], "selfish_but_fair": [0.48, 0.0, 0.9],
}


def synthetic(n_types=4, n_targets=10, seed=0):
    rs = np.random.default_rng(seed)
    B = pd.DataFrame(rs.uniform(0, 1, (n_types, n_targets)), index=[f"t{i}" for i in range(n_types)],
                     columns=[f"g{j}" for j in range(n_targets)])
    return B


class TestFit(unittest.TestCase):
    def test_recovers_on_grid_mixture_exactly(self):
        B = synthetic()
        w = pd.Series([0.30, 0.10, 0.46, 0.14], index=B.index)
        y = mixture(w, B)
        f = fit_mixture_weights(B, y)
        self.assertIsInstance(f, MixtureFit)
        np.testing.assert_allclose(f.weights.to_numpy(), w.to_numpy(), atol=1e-12)
        self.assertLess(f.train_sse, 1e-20)
        self.assertTrue(f.identified)
        self.assertEqual(f.unidentified_pairs, [])
        self.assertAlmostEqual(f.weights.sum(), 1.0)

    def test_recovers_off_grid_mixture_within_grid_step(self):
        for seed in range(3):
            B = synthetic(seed=seed)
            w = pd.Series(np.random.default_rng(100 + seed).dirichlet(np.ones(4)), index=B.index)
            y = mixture(w, B)
            for step in (0.02, 0.05):
                f = fit_mixture_weights(B, y, grid_step=step)
                self.assertLessEqual(np.max(np.abs(f.weights - w)), step + 1e-12, (seed, step))
            c = fit_mixture_weights(B, y, method="slsqp")
            self.assertLessEqual(np.max(np.abs(c.weights - w)), 1e-4)
            self.assertEqual(c.method, "slsqp")
            self.assertIsNone(c.weight_range)

    def test_identical_types_warn_and_are_reported(self):
        B = synthetic(3)
        B.loc["copy"] = B.loc["t1"] + 0.004                   # within ident_tol=0.01
        y = mixture({"t0": 0.2, "t1": 0.5, "t2": 0.3}, B)
        with self.assertWarns(UnidentifiedWarning):
            f = fit_mixture_weights(B, y)
        self.assertEqual(f.unidentified_pairs, [("t1", "copy")])
        self.assertFalse(f.identified)
        # exact duplicates: the split between them is arbitrary, and weight_range shows it
        B2 = synthetic(3)
        B2.loc["copy"] = B2.loc["t1"]
        with self.assertWarns(UnidentifiedWarning):
            f2 = fit_mixture_weights(B2, mixture({"t0": 0.2, "t1": 0.5, "t2": 0.3}, B2))
        self.assertGreater(f2.n_ties, 1)
        self.assertAlmostEqual(f2.weight_range.loc["copy", "min"], 0.0)
        self.assertAlmostEqual(f2.weight_range.loc["copy", "max"], 0.5)
        self.assertAlmostEqual(f2.weights["t1"] + f2.weights["copy"], 0.5)
        self.assertAlmostEqual(f2.weights["t0"], 0.2)

    def test_distinct_types_do_not_warn(self):
        B = synthetic()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UnidentifiedWarning)
            fit_mixture_weights(B, mixture({"t0": 1.0}, B), grid_step=0.1)

    def test_rank_deficiency_warns_without_identical_pair(self):
        B = synthetic(3, 6)
        B.loc["mix"] = 0.5 * B.loc["t0"] + 0.5 * B.loc["t1"]   # a convex combination, not a copy
        with self.assertWarns(UnidentifiedWarning):
            f = fit_mixture_weights(B, B.loc["t2"], grid_step=0.1)
        self.assertEqual(f.unidentified_pairs, [])
        self.assertEqual(f.rank, 3)
        self.assertFalse(f.identified)
        info = identifiability(B)
        self.assertFalse(info["identified"])
        # more types than targets + 1 is never identified
        self.assertFalse(identifiability(synthetic(5, 3))["identified"])

    def test_validation_uses_held_out_targets_only(self):
        B = synthetic(3, 8)
        w = pd.Series([0.5, 0.3, 0.2], index=B.index)
        truth = mixture(w, B)
        train, held = truth[["g0", "g1", "g2", "g3", "g4"]], truth[["g5", "g6", "g7"]] + [0.1, -0.2, 0.05]
        f = fit_mixture_weights(B, train, grid_step=0.1, val=held)
        expect = np.sqrt(np.mean((f.weights.to_numpy() @ B[held.index].to_numpy() - held.to_numpy()) ** 2))
        self.assertAlmostEqual(f.val_rmse, expect)
        self.assertEqual(list(f.val_pred.index), list(held.index))
        self.assertEqual(list(f.train_pred.index), list(train.index))
        # the held-out columns never move the weights or the train SSE
        B2 = B.copy()
        B2[["g5", "g6", "g7"]] = 0.0
        f2 = fit_mixture_weights(B2, train, grid_step=0.1, val=held)
        pd.testing.assert_series_equal(f.weights, f2.weights)
        self.assertEqual(f.train_sse, f2.train_sse)
        self.assertNotAlmostEqual(f.val_rmse, f2.val_rmse)
        # leakage fails fast
        with self.assertRaises(ValueError):
            fit_mixture_weights(B, train, val=truth[["g4", "g5"]])
        # separate validation table (e.g. another probe)
        VB = pd.DataFrame({"v": [0.0, 1.0, 0.5]}, index=B.index)
        f3 = fit_mixture_weights(B, train, grid_step=0.1, val=(VB, pd.Series({"v": 0.4})))
        self.assertAlmostEqual(f3.val_rmse, abs(f3.weights.to_numpy() @ [0.0, 1.0, 0.5] - 0.4))
        self.assertIsNone(fit_mixture_weights(B, train, grid_step=0.1).val_rmse)

    def test_missing_cells_raise_not_imputed(self):
        B = synthetic(3, 4)
        y = mixture({"t0": 1.0}, B)
        B.loc["t1", "g2"] = np.nan
        with self.assertRaises(ValueError) as cm:
            fit_mixture_weights(B, y, grid_step=0.1)
        self.assertIn("t1/g2", str(cm.exception))
        fit_mixture_weights(B, y.drop("g2"), grid_step=0.1)    # an unused column may be missing
        with self.assertRaises(ValueError):
            fit_mixture_weights(synthetic(3, 4), y.where(y.index != "g0"), grid_step=0.1)

    def test_bad_arguments(self):
        B = synthetic(3, 4)
        y = mixture({"t0": 1.0}, B)
        with self.assertRaises(ValueError):
            fit_mixture_weights(B, y, grid_step=0.03)
        with self.assertRaises(ValueError), warnings.catch_warnings():
            warnings.simplefilter("ignore", UnidentifiedWarning)
            fit_mixture_weights(synthetic(12, 4), mixture({"t0": 1.0}, synthetic(12, 4)), max_grid=1000)
        with self.assertRaises(ValueError):
            fit_mixture_weights(B, pd.Series({"nope": 0.1}))
        with self.assertRaises(ValueError):
            fit_mixture_weights(B, y, method="nope")
        one = fit_mixture_weights(B.loc[["t0"]], y)
        self.assertEqual(one.weights.to_dict(), {"t0": 1.0})

    def test_reproduces_edsl_test_homo_silicus_calibration(self):
        B = pd.DataFrame(SRC_CR, index=list(CR_GAMES)).T
        D = pd.DataFrame(SRC_DICTATOR, index=list(ENGEL_DICTATOR.index)).T
        f = fit_mixture_weights(B, CR_HUMAN_LEFT, val=(D, ENGEL_DICTATOR))
        self.assertEqual({k: round(v, 2) for k, v in f.weights.items()},
                         {"self_interested": 0.36, "efficient": 0.5, "inequity_averse": 0.0, "competitive": 0.0,
                          "reciprocal": 0.0, "selfish_but_fair": 0.14})
        self.assertAlmostEqual(f.train_sse, 0.088936, places=6)       # weights.json train_sse
        self.assertAlmostEqual(f.val_rmse, 0.31485491897, places=9)   # weights.json val_rmse
        self.assertTrue(f.identified)
        json.dumps(f.to_dict())


def scripted_responder(choices, gives):
    """type -> fixed answers: choices[type] = {game: 'Left'/'Right'}, gives[type] = reply text."""
    def respond(system, user):
        t = next(k for k in choices if THEORY_TYPES[k] in system)
        if "How much do you give" in user:
            return gives[t]
        for g, (L, R, _) in CR_GAMES.items():
            if f"you get ${L[1]}, Person A gets ${L[0]}." in user and f"you get ${R[1]}, Person A gets ${R[0]}." in user:
                return choices[t][g]
        raise AssertionError(user)
    return respond


class RecordingLLM:
    model = "rec"

    def __init__(self, reply):
        self.reply, self.kw = reply, []

    def complete(self, system, user, *, max_tokens=256, **kw):
        return self.map([(system, user)], max_tokens=max_tokens, **kw)[0]

    def map(self, prompts, *, max_tokens=256, parallel=4, **kw):
        from simengine.types import LLMResponse
        self.kw.append(dict(kw, max_tokens=max_tokens))
        return [LLMResponse(text=self.reply(s, u, kw), model=self.model) for s, u in prompts]


class TestProbe(unittest.TestCase):
    TYPES = {k: THEORY_TYPES[k] for k in ("self_interested", "efficient", "inequity_averse")}

    def test_parsers(self):
        g = choice_game("pick", ("Left", "Right"))
        self.assertEqual(g.parse("Left."), "Left")
        self.assertEqual(g.parse("**right**"), "Right")
        self.assertEqual(g.parse("I choose Left because..."), "Left")
        self.assertIsNone(g.parse("Left or Right, hard to say"))
        self.assertIsNone(g.parse("neither"))
        n = numeric_game("how much", lo=0, hi=100)
        self.assertEqual(n.parse("$30"), 30.0)
        self.assertEqual(n.parse("I give 30 dollars, i.e. 30."), 30.0)
        self.assertIsNone(n.parse("between 20 and 30"))
        self.assertIsNone(n.parse("150"))
        self.assertIsNone(n.parse("nothing"))
        self.assertIn("exactly one word", g.user)

    def test_probe_and_fit_end_to_end(self):
        choices = {
            "self_interested": {g: "Left" for g in CR_GAMES},
            "efficient": {"Berk29": "Right", "Barc2": "Right", "Berk23": "Left", "Barc8": "Right",
                          "Berk15": "Right", "Berk26": "Left"},
            "inequity_averse": {"Berk29": "Left", "Barc2": "Left", "Berk23": "Right", "Barc8": "Right",
                                "Berk15": "Right", "Berk26": "Right"},
        }
        gives = {"self_interested": "0", "efficient": "0", "inequity_averse": "50"}
        llm = FakeLLM(scripted_responder(choices, gives))
        games = {**charness_rabin_games(), "dictator": dictator_game()}
        B, rows = probe_types(llm, self.TYPES, games, reps=3, prefix=HUMANIZING_PREFIX, return_rows=True)
        self.assertEqual(list(B.index), list(self.TYPES))
        self.assertEqual(list(B.columns), list(CR_GAMES) + list(ENGEL_DICTATOR.index))
        self.assertEqual(B.loc["efficient", "Berk26"], 1.0)
        self.assertEqual(B.loc["inequity_averse", "dictator:p_give_50"], 1.0)
        self.assertEqual(B.loc["inequity_averse", "dictator:mean_give_frac"], 0.5)
        self.assertEqual(len(llm.calls), 3 * 3 * 7)
        system, user = llm.calls[0]
        self.assertTrue(system.startswith(HUMANIZING_PREFIX) and system.endswith(THEORY_TYPES["self_interested"]))
        self.assertNotIn(THEORY_TYPES["self_interested"], user)
        for col in ("persona_method", "agent_id", "seed", "persona_params", "max_tokens", "wording_id", "valid"):
            self.assertIn(col, rows.columns)
        self.assertTrue(rows.valid.all())
        self.assertEqual(B.attrs["n_valid"]["efficient/Berk29"], 3)
        # self_interested and efficient are identical on the dictator probe (the edsl-test v1 bug)
        with self.assertWarns(UnidentifiedWarning):
            fd = fit_mixture_weights(B, ENGEL_DICTATOR)
        self.assertIn(("self_interested", "efficient"), fd.unidentified_pairs)
        f = fit_mixture_weights(B, CR_HUMAN_LEFT, val=ENGEL_DICTATOR)
        self.assertTrue(f.identified)
        self.assertIsNotNone(f.val_rmse)

    def test_invalid_answers_are_counted_and_left_missing(self):
        def reply(s, u, kw):
            if THEORY_TYPES["efficient"] in s:
                return "I cannot decide" if "Option Left" in u else "maybe 10 or 20"
            return "Left" if "Option Left" in u else "25"
        llm = RecordingLLM(reply)
        games = {"Berk29": charness_rabin_games()["Berk29"], "dictator": dictator_game()}
        with self.assertWarns(UserWarning):
            B, rows = probe_types(llm, self.TYPES, games, reps=4, return_rows=True)
        self.assertTrue(np.isnan(B.loc["efficient", "Berk29"]))
        self.assertTrue(B.loc["efficient", ["dictator:mean_give_frac", "dictator:p_give_0"]].isna().all())
        self.assertEqual(B.loc["self_interested", "dictator:mean_give_frac"], 0.25)
        self.assertEqual(B.attrs["n_invalid"]["efficient/Berk29"], 4)
        self.assertEqual(int((~rows.valid).sum()), 8)
        self.assertEqual(rows.loc[~rows.valid, "value"].isna().sum(), 8)
        # repetitions are independent draws for caching backends: sample=r per map call
        self.assertEqual([k.get("sample") for k in llm.kw], [0, 1, 2, 3])
        with self.assertRaises(ValueError):                    # no imputation downstream either
            fit_mixture_weights(B, pd.Series({"Berk29": 0.31}))

    def test_min_valid_and_sample_kw_off(self):
        n = {"i": 0}

        def reply(s, u, kw):
            n["i"] += 1
            return "Left" if n["i"] % 2 else "?"
        llm = RecordingLLM(reply)
        g = {"Berk29": charness_rabin_games()["Berk29"]}
        with self.assertWarns(UserWarning):
            B = probe_types(llm, {"a": "A"}, g, reps=4, min_valid=3, sample_kw=None)
        self.assertTrue(np.isnan(B.loc["a", "Berk29"]))
        self.assertTrue(all("sample" not in k for k in llm.kw))
        with self.assertRaises(ValueError):
            probe_types(llm, {"a": "A"}, g, reps=0)


if __name__ == "__main__":
    unittest.main()
