"""Self-exciting cascade predictors for share / retweet curves (Hawkes mean field, SEISMIC-style marked kernel) and the
shape baseline. Ported from the TALP prep research code `abm/kod/yayilim24.py` (pre-registered as `yayilim-24s`).

STATUS: NOT YET MEASURED. The research run stopped after the training-set kernel grid search; no dev or test
evaluation of these predictors exists in the sources (makro/onkayit/yayilim-24s.md is the plan, no RAPOR entry).
Treat every number these functions produce as unvalidated until you score it against the shape baseline
(:func:`shape_forecast`) with the evidence pipeline.

Units: time in minutes, horizon 0-1440 min (24 h). Curves are cumulative counts N(s) on GRID (10-min steps); scoring
is on log1p.

* :func:`hawkes_mean_field`: expected future events after observing [0, t]: each observed event (plus the root at 0)
  excites future events through a normalised kernel F; the branching factor p is estimated per cascade as
  (#events) / sum_i w_i F(t - t_i), capped at `cap / n_star`; then the mean-field recursion runs minute by minute.
* :func:`power_kernel` (Hawkes, power-law, c and theta picked on training data) and :func:`seismic_kernel`
  (SEISMIC: constant for s0 = 5 min then power law with theta = 0.242, Zhao et al. KDD 2015 defaults).
* Marks: with follower counts as marks (w_i) and n_star = mean follower count, this is the SEISMIC-style predictor;
  :class:`SeismicPredictor` adds the research's linear correction of the log final size fitted on training cascades.
"""
from __future__ import annotations

import numpy as np

GRID = np.arange(0, 1441, 10)   # 145 points, minutes
HORIZON = 1440


def cascade_counts(delays, grid=GRID):
    """delays: list of sorted arrays of event delays (min) per cascade (root excluded). Returns counts (n, len(grid))."""
    return np.stack([np.searchsorted(np.sort(np.asarray(g, float)), grid, side="right") for g in delays]).astype(float)


def power_kernel(c, theta, horizon=HORIZON):
    """Normalised cumulative power-law kernel F(tau) = (c^-th - (tau + c)^-th) / (c^-th - (H + c)^-th)."""
    a, b = c ** -theta, (horizon + c) ** -theta
    return lambda tau: (a - (np.asarray(tau, float) + c) ** -theta) / (a - b)


def seismic_kernel(s0=5.0, theta=0.242, horizon=HORIZON):
    """SEISMIC memory kernel: constant until s0, then power law; normalised so F(horizon) = 1."""
    def raw(tau):
        tau = np.asarray(tau, float)
        return np.where(tau <= s0, tau, s0 + s0 / theta * (1 - (np.maximum(tau, s0) / s0) ** -theta))
    z = float(raw(float(horizon)))
    return lambda tau: raw(tau) / z


def hawkes_mean_field(delays, t, F, cap=0.99, marks=None, n_star=1.0, horizon=HORIZON):
    """Mean-field Hawkes recursion in 1-minute bins after observing [0, t] minutes.

    delays: list of arrays (event delays, root excluded). marks: optional list of arrays aligned with
    [root] + events observed up to t (e.g. follower counts). Returns (E, p): E (n, horizon) expected events per
    minute bin (zero before t), p the per-cascade branching estimate before capping."""
    T = int(horizon)
    t = int(t)
    dF = np.diff(F(np.arange(T + 1, dtype=float)))
    n = len(delays)
    E = np.zeros((n, T))
    base = np.zeros((n, T))
    par = np.zeros(n)
    mins = np.arange(t, T)
    for k, g in enumerate(delays):
        g = np.asarray(g, float)
        gg = np.concatenate([[0.0], g[g <= t]])
        w = np.ones(len(gg)) if marks is None else np.asarray(marks[k], float)
        num = len(gg) - 1
        den = float(np.sum(w * F(np.minimum(t - gg, T))))
        par[k] = num / den if den > 0 else 0.0
        b = F(np.clip(mins[None, :] + 1 - gg[:, None], 0, T)) - F(np.clip(mins[None, :] - gg[:, None], 0, T))
        base[k, t:] = (w[:, None] * b).sum(0)
    p = np.minimum(par, cap / n_star)
    for j in range(t, T):
        past = E[:, t:j] @ dF[1:j - t + 1][::-1] if j > t else 0.0
        E[:, j] = p * (base[:, j] + n_star * past)
    return E, par


def cumulative_curve(N_obs, t, E, grid=GRID):
    """Combine observed counts (n, len(grid)) up to t with expected future events E -> log1p curve (n, len(grid))."""
    i = int(np.searchsorted(grid, t))
    cum = np.concatenate([np.zeros((len(E), 1)), np.cumsum(E, 1)], 1)
    L = np.log1p(N_obs).copy()
    L[:, i + 1:] = np.log1p(N_obs[:, i:i + 1] + cum[:, grid[i + 1:]])
    return L


def shape_growth(N_train, t, grid=GRID):
    """Median log1p growth after t over training cascades (Szabo-Huberman style shape baseline)."""
    i = int(np.searchsorted(grid, t))
    return np.median(np.log1p(N_train) - np.log1p(N_train[:, i:i + 1]), 0)


def shape_forecast(N_obs, t, growth, grid=GRID):
    """Shape baseline: log1p N(t) + median training growth. Returns a log1p curve (n, len(grid))."""
    i = int(np.searchsorted(grid, t))
    L = np.log1p(N_obs).copy()
    L[:, i + 1:] = np.log1p(N_obs[:, i:i + 1]) + growth[None, i + 1:]
    return L


def curve_rmse(L, N_true):
    """Per-cascade RMSE on log1p between a predicted log curve and true counts."""
    return np.sqrt(((L - np.log1p(N_true)) ** 2).mean(1))


def select_power_kernel(train_delays, t=60, cs=(1, 5, 15), thetas=(0.1, 0.25, 0.5, 1.0), cap=0.99, grid=GRID):
    """Research selection rule: pick (c, theta) by mean log1p curve RMSE on TRAINING cascades observed until t.
    Returns ((c, theta), {(c, theta): rmse})."""
    N = cascade_counts(train_delays, grid)
    scores = {}
    for c in cs:
        for th in thetas:
            E, _ = hawkes_mean_field(train_delays, t, power_kernel(c, th), cap)
            scores[(c, th)] = float(curve_rmse(cumulative_curve(N, t, E, grid), N).mean())
    return min(scores, key=scores.get), scores


class SeismicPredictor:
    """Marked (follower-weighted) mean-field predictor with the SEISMIC kernel plus a linear correction of the log
    final size fitted on training cascades: log1p(N_final) ~ a + b * log1p(N(t) + sum E). NOT YET MEASURED."""
    name = "seismic"
    kind = "baseline"

    def __init__(self, t=60, s0=5.0, theta=0.242, cap=0.99):
        self.t, self.cap = t, cap
        self.F = seismic_kernel(s0, theta)
        self.ab = np.array([0.0, 1.0])
        self.n_star = 1.0

    @staticmethod
    def _marks(delays, followers, root_followers, t):
        return [np.concatenate([[root_followers[k]], np.asarray(followers[k], float)[np.asarray(delays[k]) <= t]])
                for k in range(len(delays))]

    def _raw(self, delays, followers, root_followers):
        N = cascade_counts(delays)
        i = int(np.searchsorted(GRID, self.t))
        E, _ = hawkes_mean_field(delays, self.t, self.F, self.cap,
                                 marks=self._marks(delays, followers, root_followers, self.t), n_star=self.n_star)
        return N, i, E

    def fit(self, delays, followers, root_followers):
        """Training cascades: full delays (to 24 h), follower counts aligned with delays, root follower counts."""
        self.n_star = float(np.mean(np.concatenate([np.asarray(f, float) for f in followers]))) or 1.0
        N_full = cascade_counts(delays)
        obs = [np.asarray(d, float)[np.asarray(d, float) <= self.t] for d in delays]
        fol = [np.asarray(f, float)[np.asarray(d, float) <= self.t] for d, f in zip(delays, followers)]
        N, i, E = self._raw(obs, fol, root_followers)
        raw_final = np.log1p(N[:, i] + E.sum(1))
        A = np.c_[np.ones(len(raw_final)), raw_final]
        self.ab = np.linalg.lstsq(A, np.log1p(N_full[:, -1]), rcond=None)[0]
        return self

    def predict(self, delays, followers, root_followers):
        """delays / followers observed up to self.t. Returns a log1p curve on GRID (n, 145)."""
        N, i, E = self._raw(delays, followers, root_followers)
        raw_final = np.log1p(N[:, i] + E.sum(1))
        final = np.maximum(np.expm1(self.ab[0] + self.ab[1] * raw_final), N[:, i])
        scale = (final - N[:, i]) / np.maximum(E.sum(1), 1e-12)
        return cumulative_curve(N, self.t, E * scale[:, None])
