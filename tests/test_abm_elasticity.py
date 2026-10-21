import json
import unittest

import numpy as np
import pandas as pd

import simengine.abm as A
from simengine.llm import FakeLLM


def synthetic_events(n_units=30, events_per_unit=4, seed=0):
    """Events with windows t = -25..8; unit elasticity = category level + unit deviation."""
    rng = np.random.default_rng(seed)
    cat_b = {"cereal": -2.0, "soup": -1.0}
    rows, true_e, cats = [], {}, {}
    eid = 0
    for u in range(n_units):
        cat = "cereal" if u % 2 else "soup"
        e = cat_b[cat] + rng.normal(0, 0.8)
        true_e[f"u{u}"], cats[f"u{u}"] = e, cat
        for _ in range(events_per_unit):
            a = rng.normal(2.5, 0.3)
            t = np.arange(-25, 9)
            lp = np.where(rng.random(len(t)) < 0.15, -rng.uniform(0.05, 0.4, len(t)), 0.0)
            lp[t >= 1] = np.where(t[t >= 1] <= 4, -0.25, 0.0)
            promo = (lp < -0.2).astype(float)
            mu = np.exp(a + e * lp + 0.3 * promo)
            y = rng.poisson(mu)
            rows += [dict(series=f"e{eid}", unit=f"u{u}", category=cat, t=int(tt), y=float(yy), lp=float(l),
                          promo=float(p)) for tt, yy, l, p in zip(t, y, lp, promo)]
            eid += 1
    return pd.DataFrame(rows), true_e, cats


class TestElasticity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.true_e, cls.cats = synthetic_events()
        ev = sorted(cls.df.series.unique(), key=lambda s: int(s[1:]))
        test_units = {f"u{u}" for u in range(0, 30, 3)}
        is_test = cls.df.unit.isin(test_units)
        cls.train = cls.df[~is_test]
        cls.target_pre = cls.df[is_test & (cls.df.t <= 0)]
        cls.query = cls.df[is_test & (cls.df.t >= 1)]

    def score(self, m):
        p = m.predict(m.fit(self.train, self.target_pre), self.query)
        return A.wape(p.y_pred, self.query.y)

    def test_category_matches_loglog_fe(self):
        m = A.ElasticityShrinkage("category", with_data=False)
        fit = m.fit(self.train, self.target_pre)
        coefs = A.category_elasticity(self.train)
        st = fit["st"]
        np.testing.assert_allclose(st.b.to_numpy(), [coefs[c][0][0] for c in st.category])
        self.assertEqual(m.kind, "baseline")
        self.assertLess(coefs["cereal"][0][0], coefs["soup"][0][0])     # cereal more elastic

    def test_oracle_prior_beats_category_and_naive(self):
        w_cat = self.score(A.ElasticityShrinkage("category", with_data=False))
        w_orc = self.score(A.ElasticityShrinkage("llm", llm_elasticity=self.true_e, with_data=False))
        w_mean = A.wape(A.PreEventMean(26).predict(A.PreEventMean(26).fit(self.train, self.target_pre),
                                                   self.query).y_pred, self.query.y)
        self.assertLess(w_orc, w_cat)
        self.assertLess(w_cat, w_mean)

    def test_shrinkage_lambda_from_grid_and_all_priors_run(self):
        for prior in ("category", "llm", "llm_deviation", "mix"):
            m = A.ElasticityShrinkage(prior, llm_elasticity=self.true_e, lam_grid=[1, 10, 1e9])
            fit = m.fit(self.train, self.target_pre)
            self.assertIn(fit["lam"], [1, 10, 1e9])
            self.assertTrue(np.isfinite(self.score(m)))

    def test_placebo_shuffles_within_category(self):
        sh = A.shuffle_elasticities(self.true_e, self.cats, seed=3)
        self.assertEqual(set(sh), set(self.true_e))
        for c in ("cereal", "soup"):
            a = sorted(v for u, v in self.true_e.items() if self.cats[u] == c)
            b = sorted(v for u, v in sh.items() if self.cats[u] == c)
            np.testing.assert_allclose(a, b)
        self.assertNotEqual(sh, self.true_e)
        # oracle knowledge must beat its own placebo
        w_true = self.score(A.ElasticityShrinkage("llm", llm_elasticity=self.true_e, with_data=False))
        w_plac = self.score(A.ElasticityShrinkage("llm", llm_elasticity=sh, with_data=False))
        self.assertLess(w_true, w_plac)

    def test_needs_llm_values(self):
        with self.assertRaises(ValueError):
            A.ElasticityShrinkage("llm")

    def test_elicit_with_fake_llm(self):
        def responder(system, user):
            ids = [ln.split(":")[0] for ln in user.splitlines() if ln.startswith("u")]
            return "Here: " + json.dumps([{"id": i, "elasticity": -1.5, "low": -2, "high": -1} for i in ids])
        llm = FakeLLM(responder)
        items = {f"u{i}": f"item {i}, cereal; regular price $3.00" for i in range(25)}
        out = A.elicit_elasticities(llm, items, batch=10)
        self.assertEqual(len(out), 25)
        self.assertEqual(len(llm.calls), 3)
        self.assertIn("own-price elasticity", llm.calls[0][1])
        self.assertEqual(A.elicit_elasticities(FakeLLM(), items, batch=25), {})   # garbage replies are dropped


class TestMarketABM(unittest.TestCase):
    def test_fit_predict_random_search(self):
        rng = np.random.default_rng(0)
        n, n_pre, H = 40, 26, 8
        th = np.array([-1.5, 2.5, 0.5, 2.0, 3.0, 0.7, 1.0, 0.8, 0.8, 0.0])
        LP = np.zeros((n, n_pre + H))
        LP[:, n_pre:n_pre + 4] = -rng.uniform(0.1, 0.3, (n, 1))
        PR = (LP < -0.15).astype(float)
        Y = A.market_sim(np.tile(th, (n, 1)), LP, PR, n_agents=100, cohort=True) * rng.uniform(5, 20, (n, 1))
        cats = np.where(np.arange(n) % 2, "a", "b")
        m = A.MarketABM(n_pre=n_pre, n_trials=6, sample=20, n_agents=100, use_optuna=False).fit(Y, LP, PR, cats)
        Y_pre = Y.copy()
        Y_pre[:, n_pre:] = np.nan
        yh, lo, hi = m.predict(LP, PR, Y_pre, cats, with_band=True)
        self.assertEqual(yh.shape, (n, H))
        self.assertTrue((lo <= hi + 1e-9).all())
        self.assertTrue(np.isfinite(A.wape(yh, Y[:, n_pre:])))
        with self.assertRaises(ValueError):
            m.predict(LP, PR, Y_pre, np.array(["zzz"] * n))


if __name__ == "__main__":
    unittest.main()
