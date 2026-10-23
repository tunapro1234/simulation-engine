"""llm_ceiling_prior: blind / named / perturbed prompts, placebo controls, leak check (FakeLLM only)."""
import json
import re
import unittest

import numpy as np
import pandas as pd

import simengine.abm as A
from simengine.llm import FakeLLM

_LINE = re.compile(r"^\s*(-?\d+):\s*([-+0-9.eE]+)\s*$", re.M)


def panel(n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        kind = "cell" if i % 2 else "web"
        m = 60 + 8 * i
        for t in range(2000, 2011):
            rows.append(dict(series=f"{kind}:c{i}", kind=kind, t=float(t),
                             y=float(m / (1 + np.exp(-0.5 * (t - 2008))) + rng.normal(0, 0.1))))
    return pd.DataFrame(rows)


def values(user):
    return [float(v) for _, v in _LINE.findall(user)]


def shape_responder(system, user):
    """Reasons from the numbers only: 1.5 x the last value."""
    return json.dumps({"ceiling": 1.5 * values(user)[-1]})


def memoriser(table):
    """Recognises the exact history (blind) and answers its 'known' ceiling; a rescaled history is not recognised."""
    def respond(system, user):
        key = tuple(round(v, 3) for v in values(user))
        return json.dumps({"ceiling": table.get(key, 100.0)})
    return respond


class TestCeilingPrior(unittest.TestCase):
    def setUp(self):
        self.df = panel()

    def test_columns_calls_and_prompts(self):
        llm = FakeLLM(shape_responder)
        pr = A.llm_ceiling_prior(llm, self.df, groups="kind")
        n = self.df["series"].nunique()
        self.assertEqual(len(pr), n)
        self.assertEqual(len(llm.calls), 3 * n)                 # named + blind + perturbed, controls need no calls
        for c in ("ceiling", "ceiling_blind", "ceiling_perturbed", "ceiling_shuffled", "ceiling_group_median",
                  "group", "hist_max"):
            self.assertIn(c, pr.columns)
        named = [u for _, u in llm.calls if "Series:" in u]
        blind = [u for _, u in llm.calls if "Series:" not in u]
        self.assertEqual(len(named), n)
        for u in blind:                                          # anonymised: no name, no calendar year
            self.assertNotIn("cell:", u)
            self.assertNotIn("web:", u)
            self.assertNotIn("2005", u)
        last = self.df.sort_values("t").groupby("series")["y"].last()
        np.testing.assert_allclose(pr["ceiling"], 1.5 * last.loc[pr.index], rtol=1e-3)

    def test_perturbed_follows_scaling_no_recognition(self):
        pr = A.llm_ceiling_prior(FakeLLM(shape_responder), self.df, groups="kind")
        np.testing.assert_allclose(pr["ceiling_perturbed"], pr["ceiling_blind"], rtol=1e-3)
        lc = pr.attrs["leak_check"]
        self.assertGreater(lc["rho_perturbed_blind"], 0.95)
        self.assertGreater(lc["scale_elasticity"], 0.9)
        self.assertFalse(lc["recognition"])
        self.assertIn("no sign of recognition", lc["message"])

    def test_memorised_blind_answers_are_flagged(self):
        rng = np.random.default_rng(5)
        table = {}
        for _, g in self.df.sort_values("t").groupby("series"):
            table[tuple(round(float(f"{v:.4g}"), 3) for v in g["y"])] = float(rng.uniform(80, 200))
        pr = A.llm_ceiling_prior(FakeLLM(memoriser(table)), self.df, named=False, groups="kind")
        lc = pr.attrs["leak_check"]
        self.assertTrue(lc["recognition"])
        self.assertLess(lc["scale_elasticity"], 0.5)
        self.assertIn("recogni", lc["message"])

    def test_placebo_controls(self):
        pr = A.llm_ceiling_prior(FakeLLM(shape_responder), self.df, groups="kind", perturb=0)
        self.assertNotIn("ceiling_perturbed", pr.columns)
        for g, sub in pr.groupby("group"):
            for s in sub.index:                                  # leave-one-out: the series' own value is not in it
                self.assertAlmostEqual(sub.loc[s, "ceiling_group_median"], sub["ceiling"].drop(s).median())
            self.assertEqual(sorted(sub["ceiling_shuffled"]), sorted(sub["ceiling"]))
            self.assertTrue((sub["ceiling_shuffled"] != sub["ceiling"]).all())   # no series keeps its own value

    def test_placebos_carry_no_own_value_in_small_groups(self):
        # group of 3: the plain median IS the middle series' own ceiling; a singleton would keep its own in both
        df = self.df[self.df.series.isin(["web:c0", "web:c2", "web:c4", "cell:c1"])]
        pr = A.llm_ceiling_prior(FakeLLM(shape_responder), df, groups="kind", perturb=0)
        web = pr[pr.group == "web"]
        mid = web["ceiling"].sort_values().index[1]
        self.assertNotAlmostEqual(web.loc[mid, "ceiling_group_median"], web.loc[mid, "ceiling"])
        self.assertTrue((web["ceiling_group_median"] != web["ceiling"]).all())
        self.assertTrue(np.isnan(pr.loc["cell:c1", "ceiling_shuffled"]))
        self.assertTrue(np.isnan(pr.loc["cell:c1", "ceiling_group_median"]))
        up = A.ceiling_upper(pr, col="ceiling_shuffled")         # NaN placebo -> observed max, never its own ceiling
        self.assertEqual(up["cell:c1"], pr.loc["cell:c1", "hist_max"])

    def test_last_keeps_full_history_max(self):
        # a series that peaked and fell: last=3 shows only the decline, the floor must still be the real maximum
        rows = [dict(series=f"s{i}", t=float(t), y=float(v + i))
                for i in range(3) for t, v in zip(range(2000, 2008), [5, 20, 60, 90, 80, 70, 65, 60])]
        df = pd.DataFrame(rows)
        llm = FakeLLM(lambda s, u: '{"ceiling": 50}')
        pr = A.llm_ceiling_prior(llm, df, perturb=0, last=3)
        self.assertTrue(all(len(values(u)) == 3 for _, u in llm.calls))
        np.testing.assert_allclose(pr["hist_max"], [90.0, 91.0, 92.0])
        self.assertEqual(pr.attrs["leak_check"]["n_below_hist"], 3)
        self.assertEqual(A.ceiling_upper(pr)["s0"], 90.0)

    def test_named_only_and_blind_only(self):
        llm = FakeLLM(shape_responder)
        pr = A.llm_ceiling_prior(llm, self.df, blind=False, perturb=0)
        self.assertEqual(len(llm.calls), self.df["series"].nunique())
        self.assertNotIn("ceiling_blind", pr.columns)
        pr2 = A.llm_ceiling_prior(FakeLLM(shape_responder), self.df, named=False, perturb=0)
        pd.testing.assert_series_equal(pr2["ceiling"], pr2["ceiling_blind"], check_names=False)
        self.assertEqual(pr2.attrs["leak_check"]["source"], "blind")
        with self.assertRaises(ValueError):
            A.llm_ceiling_prior(FakeLLM(shape_responder), self.df, blind=False, named=False)

    def test_unparseable_is_nan_not_error(self):
        pr = A.llm_ceiling_prior(FakeLLM(lambda s, u: "no idea"), self.df, perturb=0)
        self.assertTrue(pr["ceiling"].isna().all())
        self.assertEqual(pr.attrs["leak_check"]["n_failed"]["named"], self.df["series"].nunique())

    def test_upper_dict_floors_at_history(self):
        low = FakeLLM(lambda s, u: '{"ceiling": 1}')
        pr = A.llm_ceiling_prior(low, self.df, perturb=0)
        up = A.ceiling_upper(pr)
        self.assertEqual(set(up), set(pr.index))
        for s, v in up.items():
            self.assertGreaterEqual(v, pr.loc[s, "hist_max"])
        self.assertEqual(pr.attrs["leak_check"]["n_below_hist"], len(pr))
        self.assertIsInstance(pr["ceiling"].to_dict(), dict)

    def test_future_spearman(self):
        fut = []
        for s, g in self.df.groupby("series"):
            fut.append(dict(series=s, t=2012.0, y=float(g["y"].iloc[-1]) * 1.2))
        pr = A.llm_ceiling_prior(FakeLLM(shape_responder), self.df, future=pd.DataFrame(fut), groups="kind")
        lc = pr.attrs["leak_check"]
        self.assertGreater(lc["rho_future"]["ceiling"], 0.9)
        self.assertIn("blind", lc["rho_future"])

    def test_deterministic_with_seed(self):
        a = A.llm_ceiling_prior(FakeLLM(shape_responder), self.df, groups="kind", seed=3)
        b = A.llm_ceiling_prior(FakeLLM(shape_responder), self.df, groups="kind", seed=3)
        pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    unittest.main()
