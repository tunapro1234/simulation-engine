"""Repeat-purchase market ABM with pooled per-category calibration (ported from the TALP prep research code
`abm/kod/m2_abm.py`; engine in :func:`simengine.abm.engine.market_sim`).

Forecast for an event: simulate its planned price / promo path, then scale the level to the event's own pre-period:
    y_hat_t = sim_t * (sum of real sales in the last `window` pre weeks / sum of simulated sales in the same weeks).
Calibration: one parameter set per category, chosen by WAPE on (a sample of) training events. The research used optuna
TPE (120-300 trials) seeded with a literature start; optuna is imported lazily and a seeded random search is the
fallback when it is missing.

Measured (makro/abm/RAPOR.md, M2 Dominick's dev, UPC-grouped 5 folds, WAPE 8 weeks): pooled ABM 0.465 (120 TPE) /
0.456 (300 TPE) vs category log-log 0.434 and GBDT 0.373 (+/- 0.0009, 5 seeds) -- the ABM is NOT the most accurate.
C3 strategy ranking test: ABM alone Spearman 0.612 +/- 0.102 (seed sensitive) vs GBDT 0.780; ABM response as a GBDT
feature 0.812, but a plain log-log elasticity feature gives 0.807 (difference +0.005 [-0.020, +0.030]): no separate
contribution of the ABM structure was shown. Use it for mechanism / counterfactual stories (stockpiling, reference
price), not for accuracy.
"""
from __future__ import annotations

import numpy as np

from . import engine

LOWER = np.array([engine.MARKET_PRIOR[k][0] for k in engine.MARKET_PARAMS])
UPPER = np.array([engine.MARKET_PRIOR[k][1] for k in engine.MARKET_PARAMS])
# rough literature start (Bijmolt 2005 elasticity ~ -2.6, loss > gain)
START = dict(a0=-1.5, beta=2.5, sigma=0.5, gain=2.0, loss=3.0, theta=0.7, stock=1.0, inventory=0.8, promo=0.8,
             social=0.3)
WINDOWS = (4, 8, 16, 26)


def simulate(theta, LP, PR, seed=0, n_agents=200, chunk=4000):
    """theta (n, 10) or (10,); LP, PR (n, T). Cohort-mode simulated sales (n, T)."""
    n = len(LP)
    theta = np.broadcast_to(np.atleast_2d(theta), (n, len(LOWER)))
    out = np.empty(np.shape(LP))
    for i in range(0, n, chunk):
        s = slice(i, i + chunk)
        out[s] = engine.market_sim(theta[s], LP[s], PR[s], n_agents=n_agents, seed=seed, cohort=True)
    return out


def scale_to_level(sim, Y, n_pre, window=8):
    """Scale simulated post-period sales to the real pre-period level. Y may hold NaN after n_pre."""
    ratio = np.nansum(Y[:, n_pre - window:n_pre], 1) / np.maximum(sim[:, n_pre - window:n_pre].sum(1), 1e-9)
    return sim[:, n_pre:] * ratio[:, None]


def wape(y_hat, y_true):
    y_true = np.asarray(y_true, float)
    return float(np.abs(np.asarray(y_hat) - y_true).sum() / max(y_true.sum(), 1e-12))


class MarketABM:
    """Pooled-calibration market ABM (research `abm_havuz`).

    fit(Y, LP, PR, categories): training events' full windows, arrays (n, n_pre + H).
    predict(LP, PR, Y_pre, categories): target events (planned price / promo path known, Y only for the pre-period).
    """
    name = "abm_market"
    kind = "method"

    def __init__(self, n_pre=26, n_trials=120, sample=600, seed=0, n_agents=200, use_optuna=True):
        self.n_pre, self.n_trials, self.sample, self.seed = n_pre, n_trials, sample, seed
        self.n_agents, self.use_optuna = n_agents, use_optuna
        self.params: dict = {}

    def _objective(self, th, win, Y, LP, PR):
        return wape(scale_to_level(simulate(th, LP, PR, 0, self.n_agents), Y, self.n_pre, win), Y[:, self.n_pre:])

    def _search(self, Y, LP, PR):
        names = engine.MARKET_PARAMS
        try:
            if not self.use_optuna:
                raise ImportError
            import optuna
        except ImportError:
            rng = np.random.default_rng(self.seed)
            cands = [(np.array([START[k] for k in names]), 8)]
            cands += [(LOWER + (UPPER - LOWER) * rng.random(len(LOWER)), int(rng.choice(WINDOWS)))
                      for _ in range(self.n_trials - 1)]
            scores = [self._objective(th, w, Y, LP, PR) for th, w in cands]
            b = int(np.argmin(scores))
            return cands[b][0], cands[b][1], scores[b]
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def obj(tr):
            th = np.array([tr.suggest_float(k, LOWER[j], UPPER[j]) for j, k in enumerate(names)])
            return self._objective(th, tr.suggest_categorical("window", list(WINDOWS)), Y, LP, PR)

        st = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.seed))
        st.enqueue_trial({**START, "window": 8})
        st.optimize(obj, n_trials=self.n_trials)
        return np.array([st.best_params[k] for k in names]), st.best_params["window"], st.best_value

    def fit(self, Y, LP, PR, categories):
        Y, LP, PR = (np.asarray(a, float) for a in (Y, LP, PR))
        cats = np.asarray(categories)
        self.params = {}
        for c in sorted(set(cats.tolist())):
            ii = np.flatnonzero(cats == c)
            ii = np.random.default_rng(20261020 + self.seed).choice(ii, min(self.sample, len(ii)), replace=False)
            th, win, score = self._search(Y[ii], LP[ii], PR[ii])
            fit = scale_to_level(simulate(th, LP[ii], PR[ii], 0, self.n_agents), Y[ii], self.n_pre, win)
            ratio = (Y[ii, self.n_pre:] + 0.5) / (fit + 0.5)
            self.params[c] = dict(theta=th, window=int(win), wape_calib=float(score),
                                  q10=np.quantile(ratio, 0.1, 0), q90=np.quantile(ratio, 0.9, 0))
        return self

    def predict(self, LP, PR, Y_pre, categories, with_band=False):
        LP, PR, Y_pre = (np.asarray(a, float) for a in (LP, PR, Y_pre))
        cats = np.asarray(categories)
        H = LP.shape[1] - self.n_pre
        out = np.full((len(LP), H), np.nan)
        lo, hi = out.copy(), out.copy()
        for c, p in self.params.items():
            m = cats == c
            if not m.any():
                continue
            out[m] = scale_to_level(simulate(p["theta"], LP[m], PR[m], 0, self.n_agents), Y_pre[m], self.n_pre,
                                    p["window"])
            lo[m], hi[m] = out[m] * p["q10"], out[m] * p["q90"]     # conformal-style 80% band from calibration
        if np.isnan(out).any():
            raise ValueError("predict got a category that was not calibrated")
        return (out, lo, hi) if with_band else out
