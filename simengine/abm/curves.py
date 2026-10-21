"""Adoption-curve baselines (ported from the TALP prep kit `baseline.py` and `yontemler/temel.py`).

Every function takes the observed (t, y) of ONE series up to the cutoff and target times, and returns forecasts.
y is a percentage (0-100). The market potential m is estimated from data up to the cutoff, never fixed.

Measured on OWID internet adoption (dev, 122 series, MAE pp; makro/RAPOR.md, M1 table): naive 20.42,
linear trend 12.75, damped trend 13.69, logistic 16.96, Gompertz 12.39, Bass NLS 15.97, Bass + analogy 15.94.
Early in the curve, S-curve fits rarely beat a straight trend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

UPPER = 100.0


def naive(t, y, target):
    """Last value carried forward."""
    return np.full(len(target), np.asarray(y, float)[-1], dtype=float)


def linear_trend(t, y, target, last=5):
    """Straight line through the last `last` points, clipped to [0, 100]."""
    tt, yy = np.asarray(t, float)[-last:], np.asarray(y, float)[-last:]
    if len(tt) < 2:
        return naive(t, y, target)
    b, a = np.polyfit(tt, yy, 1)
    return np.clip(a + b * np.asarray(target, float), 0, UPPER)


def damped_trend(t, y, target, last=5, phi=0.9):
    """Damped trend (Gardner-McKenzie): slope of the last 5 points, damped by phi^i over the horizon. phi fixed."""
    tt, yy = np.asarray(t, float)[-last:], np.asarray(y, float)[-last:]
    if len(tt) < 2:
        return naive(t, y, target)
    b = np.polyfit(tt, yy, 1)[0]
    h = np.maximum(np.asarray(target, float) - tt[-1], 0)
    return np.clip(yy[-1] + b * phi * (1 - phi ** h) / (1 - phi), 0, UPPER)


def gompertz_curve(t, m, b, c, t0):
    return m * np.exp(-b * np.exp(-c * (np.asarray(t, float) - t0)))


def bass_curve(t, m, p, q, tau):
    """Cumulative Bass curve; tau is the start time."""
    s = np.maximum(np.asarray(t, float) - tau, 0)
    e = np.exp(-(p + q) * s)
    return m * (1 - e) / (1 + (q / p) * e)


def logistic_curve(t, m, k, t50):
    return m / (1 + np.exp(-k * (np.asarray(t, float) - t50)))


def _multi_start(x0s, lo, hi, resid):
    best = None
    for x0 in x0s:
        try:
            r = least_squares(resid, x0, bounds=(lo, hi), method="trf")
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def gompertz(t, y, target):
    """Gompertz NLS, multi-start, m lower bound = observed max."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    t0 = t[0]
    lo = [min(max(y.max(), 1e-3), UPPER - 1e-3), 1e-3, 1e-3]
    hi = [UPPER, 50.0, 3.0]
    x0s = [[min(max(y.max() * f, lo[0] + 1e-6), UPPER), b0, c0]
           for f in (1.2, 2.0, 4.0) for b0 in (2.0, 8.0) for c0 in (0.05, 0.3)]
    x0s = [np.clip(x, np.array(lo) + 1e-9, np.array(hi) - 1e-9) for x in x0s]
    r = _multi_start(x0s, lo, hi, lambda x: gompertz_curve(t, x[0], x[1], x[2], t0) - y)
    if r is None:
        return naive(t, y, target)
    return np.clip(gompertz_curve(target, *r.x, t0), 0, UPPER)


def logistic(t, y, target):
    """Logistic NLS, multi-start."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    lo = [min(max(y.max(), 1e-3), UPPER - 1e-3), 1e-3, t[0] - 30]
    hi = [UPPER, 5.0, t[-1] + 60]
    x0s = [[min(max(y.max() * f, lo[0] + 1e-6), UPPER), k, t[-1] + d]
           for f in (1.2, 2.0, 4.0) for k in (0.1, 0.4) for d in (0, 10)]
    x0s = [np.clip(x, np.array(lo) + 1e-9, np.array(hi) - 1e-9) for x in x0s]
    r = _multi_start(x0s, lo, hi, lambda x: logistic_curve(t, *x) - y)
    if r is None:
        return naive(t, y, target)
    return np.clip(logistic_curve(target, *r.x), 0, UPPER)


def bass(t, y, target, prior=None, prior_weight=0.0, return_params=False):
    """Bass NLS. prior=(log p, log q) adds a penalty term (analogy or LLM prior)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    t_first = t[0]

    def resid(x):
        m, lp, lq, tau = x
        r = bass_curve(t, m, np.exp(lp), np.exp(lq), tau) - y
        if prior is not None and prior_weight > 0:
            r = np.concatenate([r, prior_weight * (np.array([lp, lq]) - np.asarray(prior))])
        return r

    m0 = max(y.max() * 1.5, 5.0)
    x0s = [[min(m0, UPPER), np.log(p0), np.log(q0), t_first - d]
           for p0 in (0.003, 0.03) for q0 in (0.2, 0.5) for d in (0.5, 3.0)]
    if prior is not None:
        x0s.append([min(m0, UPPER), prior[0], prior[1], t_first - 1.0])
    lo = [min(max(y.max(), 1e-3), UPPER - 1e-3), np.log(1e-5), np.log(1e-3), t_first - 30]
    hi = [UPPER, np.log(0.5), np.log(3.0), t_first + 0.99]
    x0s = [np.clip(x, np.array(lo) + 1e-9, np.array(hi) - 1e-9) for x in x0s]
    r = _multi_start(x0s, lo, hi, resid)
    if r is None:
        return (naive(t, y, target), None) if return_params else naive(t, y, target)
    m, lp, lq, tau = r.x
    pred = np.clip(bass_curve(target, m, np.exp(lp), np.exp(lq), tau), 0, UPPER)
    if return_params:
        return pred, dict(m=m, p=np.exp(lp), q=np.exp(lq), tau=tau)
    return pred


CURVES = {"naive": naive, "linear": linear_trend, "damped": damped_trend, "logistic": logistic,
          "gompertz": gompertz, "bass": bass}


# ---------------------------------------------------------------- forecaster wrappers (fit / predict)
def _series(history: pd.DataFrame, series):
    g = history[history["series"] == series].sort_values("t")
    return g["t"].to_numpy(float), g["y"].to_numpy(float)


class CurveBaseline:
    """Fits one series only, using its data up to the cutoff (naive, linear, damped, logistic, gompertz, bass)."""
    kind = "baseline"

    def __init__(self, name: str, fn=None):
        self.name = name
        self.fn = fn or CURVES[name]

    def fit(self, history, series, cutoff):
        t, y = _series(history, series)
        return dict(series=series, cutoff=cutoff, t=t, y=y)

    def predict(self, model, horizons):
        h = np.asarray(horizons, float)
        return pd.DataFrame(dict(series=model["series"], cutoff=model["cutoff"], horizon=h,
                                 t=model["cutoff"] + h,
                                 y_pred=self.fn(model["t"], model["y"], model["cutoff"] + h)))


class BassAnalogy:
    """Bass + analogy prior: median Bass (log p, log q) of the OTHER series at the same cutoff becomes a penalty.
    Classic marketing method (Sultan-Farley-Lehmann type), leak free: only data with t <= cutoff."""
    kind = "baseline"
    name = "bass_analogy"

    def __init__(self, weight=1.0, min_points=6):
        self.weight, self.min_points = weight, min_points
        self._cache: dict = {}

    def _prior(self, history, cutoff, exclude):
        key = (cutoff, len(history))
        if key not in self._cache:
            pq = {}
            for s, g in history.groupby("series"):
                if len(g) >= self.min_points and g["y"].max() >= 5:
                    t = g["t"].to_numpy(float)
                    _, p = bass(t, g["y"].to_numpy(float), t[-1:], return_params=True)
                    if p is not None:
                        pq[s] = (np.log(p["p"]), np.log(p["q"]))
            self._cache[key] = pq
        pq = [v for s, v in self._cache[key].items() if s != exclude]
        if len(pq) < 3:
            return None
        return np.median(np.array(pq), axis=0)

    def fit(self, history, series, cutoff):
        t, y = _series(history, series)
        return dict(series=series, cutoff=cutoff, t=t, y=y, prior=self._prior(history, cutoff, series))

    def predict(self, model, horizons):
        h = np.asarray(horizons, float)
        pr = model["prior"]
        y = bass(model["t"], model["y"], model["cutoff"] + h, prior=pr,
                 prior_weight=self.weight if pr is not None else 0)
        return pd.DataFrame(dict(series=model["series"], cutoff=model["cutoff"], horizon=h,
                                 t=model["cutoff"] + h, y_pred=y))
