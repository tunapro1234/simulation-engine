"""Adoption-curve baselines (ported from the TALP prep kit `baseline.py` and `yontemler/temel.py`).

Every function takes the observed (t, y) of ONE series up to the cutoff and target times, and returns forecasts.
y is a percentage (0-100) by default. The market potential m is estimated from data up to the cutoff, never fixed.
`upper=` (default UPPER = 100) is the clip / market-potential bound; raise it for quantities that pass 100 (mobile
subscriptions per 100 people) or pass None for no upper bound.

    import simengine.abm.curves as C
    C.linear_trend(t, y, target, upper=200)          # subscriptions per 100 people can pass 100
    C.constant_growth(t, y, target)                  # last 3 years' log growth, damped (phi 0.8, as in rehearsal D7)
    C.CurveBaseline("growth", phi=1.0, upper=None)   # fit / predict wrapper, keyword arguments go to the curve
    C.BassAnalogy(group_col="kind")                  # analogy prior only from series of the target's own kind

Measured on OWID internet adoption (dev, 122 series, MAE pp; makro/RAPOR.md, M1 table): naive 20.42,
linear trend 12.75, damped trend 13.69, logistic 16.96, Gompertz 12.39, Bass NLS 15.97, Bass + analogy 15.94.
Early in the curve, S-curve fits rarely beat a straight trend.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

UPPER = 100.0


def _upper(upper):
    return np.inf if upper is None else float(upper)


def naive(t, y, target):
    """Last value carried forward."""
    return np.full(len(target), np.asarray(y, float)[-1], dtype=float)


def linear_trend(t, y, target, last=5, upper=UPPER):
    """Straight line through the last `last` points, clipped to [0, upper]."""
    tt, yy = np.asarray(t, float)[-last:], np.asarray(y, float)[-last:]
    if len(tt) < 2:
        return naive(t, y, target)
    b, a = np.polyfit(tt, yy, 1)
    return np.clip(a + b * np.asarray(target, float), 0, _upper(upper))


def _damped_h(h, phi):
    """sum_{i=1..h} phi^i (h itself when phi == 1)."""
    return h if phi == 1 else phi * (1 - phi ** h) / (1 - phi)


def damped_trend(t, y, target, last=5, phi=0.9, upper=UPPER):
    """Damped trend (Gardner-McKenzie): slope of the last 5 points, damped by phi^i over the horizon. phi fixed."""
    tt, yy = np.asarray(t, float)[-last:], np.asarray(y, float)[-last:]
    if len(tt) < 2:
        return naive(t, y, target)
    b = np.polyfit(tt, yy, 1)[0]
    h = np.maximum(np.asarray(target, float) - tt[-1], 0)
    return np.clip(yy[-1] + b * _damped_h(h, phi), 0, _upper(upper))


def constant_growth(t, y, target, last=3, phi=0.8, offset=0.1, upper=UPPER):
    """Constant growth rule: the last observed log growth rate damped by phi^i (default 0.8, the rehearsal D7
    setting) or held (phi = 1).

    g = log((y_T + offset) / (y_b + offset)) / (t_T - t_b), b = the earliest point within `last` years before T;
    forecast (y_T + offset) * exp(g * sum_{i=1..h} phi^i) - offset, clipped to [0, upper]. `offset` (pp) keeps
    zeros finite. With phi = 1 it grows exponentially and overshoots near saturation; upper / phi < 1 hold it back.
    Rehearsal D7: strongest simple baseline on the low-level dev origins, behind linear on the test origin
    (docs/modules/abm.md)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) < 2:
        return naive(t, y, target)
    base = np.flatnonzero((t >= t[-1] - last) & (t < t[-1]))
    if len(base) == 0:
        return naive(t, y, target)
    b = base[0]
    g = np.log((max(y[-1], 0) + offset) / (max(y[b], 0) + offset)) / (t[-1] - t[b])
    h = np.maximum(np.asarray(target, float) - t[-1], 0)
    return np.clip((max(y[-1], 0) + offset) * np.exp(g * _damped_h(h, phi)) - offset, 0, _upper(upper))


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


def gompertz(t, y, target, upper=UPPER):
    """Gompertz NLS, multi-start, m in [observed max, upper]."""
    up = _upper(upper)
    t, y = np.asarray(t, float), np.asarray(y, float)
    t0 = t[0]
    lo = [min(max(y.max(), 1e-3), up - 1e-3), 1e-3, 1e-3]
    hi = [up, 50.0, 3.0]
    x0s = [[min(max(y.max() * f, lo[0] + 1e-6), up), b0, c0]
           for f in (1.2, 2.0, 4.0) for b0 in (2.0, 8.0) for c0 in (0.05, 0.3)]
    x0s = [np.clip(x, np.array(lo) + 1e-9, np.array(hi) - 1e-9) for x in x0s]
    r = _multi_start(x0s, lo, hi, lambda x: gompertz_curve(t, x[0], x[1], x[2], t0) - y)
    if r is None:
        return naive(t, y, target)
    return np.clip(gompertz_curve(target, *r.x, t0), 0, up)


def logistic(t, y, target, upper=UPPER):
    """Logistic NLS, multi-start, m in [observed max, upper]."""
    up = _upper(upper)
    t, y = np.asarray(t, float), np.asarray(y, float)
    lo = [min(max(y.max(), 1e-3), up - 1e-3), 1e-3, t[0] - 30]
    hi = [up, 5.0, t[-1] + 60]
    x0s = [[min(max(y.max() * f, lo[0] + 1e-6), up), k, t[-1] + d]
           for f in (1.2, 2.0, 4.0) for k in (0.1, 0.4) for d in (0, 10)]
    x0s = [np.clip(x, np.array(lo) + 1e-9, np.array(hi) - 1e-9) for x in x0s]
    r = _multi_start(x0s, lo, hi, lambda x: logistic_curve(t, *x) - y)
    if r is None:
        return naive(t, y, target)
    return np.clip(logistic_curve(target, *r.x), 0, up)


def bass(t, y, target, prior=None, prior_weight=0.0, return_params=False, upper=UPPER):
    """Bass NLS, m in [observed max, upper]. prior=(log p, log q) adds a penalty term (analogy or LLM prior)."""
    up = _upper(upper)
    t, y = np.asarray(t, float), np.asarray(y, float)
    t_first = t[0]

    def resid(x):
        m, lp, lq, tau = x
        r = bass_curve(t, m, np.exp(lp), np.exp(lq), tau) - y
        if prior is not None and prior_weight > 0:
            r = np.concatenate([r, prior_weight * (np.array([lp, lq]) - np.asarray(prior))])
        return r

    m0 = max(y.max() * 1.5, 5.0)
    x0s = [[min(m0, up), np.log(p0), np.log(q0), t_first - d]
           for p0 in (0.003, 0.03) for q0 in (0.2, 0.5) for d in (0.5, 3.0)]
    if prior is not None:
        x0s.append([min(m0, up), prior[0], prior[1], t_first - 1.0])
    lo = [min(max(y.max(), 1e-3), up - 1e-3), np.log(1e-5), np.log(1e-3), t_first - 30]
    hi = [up, np.log(0.5), np.log(3.0), t_first + 0.99]
    x0s = [np.clip(x, np.array(lo) + 1e-9, np.array(hi) - 1e-9) for x in x0s]
    r = _multi_start(x0s, lo, hi, resid)
    if r is None:
        return (naive(t, y, target), None) if return_params else naive(t, y, target)
    m, lp, lq, tau = r.x
    pred = np.clip(bass_curve(target, m, np.exp(lp), np.exp(lq), tau), 0, up)
    if return_params:
        return pred, dict(m=m, p=np.exp(lp), q=np.exp(lq), tau=tau)
    return pred


CURVES = {"naive": naive, "linear": linear_trend, "damped": damped_trend, "logistic": logistic,
          "gompertz": gompertz, "bass": bass, "growth": constant_growth}


# ---------------------------------------------------------------- forecaster wrappers (fit / predict)
def _series(history: pd.DataFrame, series):
    g = history[history["series"] == series].sort_values("t")
    return g["t"].to_numpy(float), g["y"].to_numpy(float)


class CurveBaseline:
    """Fits one series only, using its data up to the cutoff (naive, linear, damped, logistic, gompertz, bass,
    growth). Keyword arguments go to the curve function (e.g. upper=200, phi=0.9)."""
    kind = "baseline"

    def __init__(self, name: str, fn=None, **kw):
        self.name = name
        self.fn = fn or CURVES[name]
        self.kw = kw

    def fit(self, history, series, cutoff):
        t, y = _series(history, series)
        return dict(series=series, cutoff=cutoff, t=t, y=y)

    def predict(self, model, horizons):
        h = np.asarray(horizons, float)
        return pd.DataFrame(dict(series=model["series"], cutoff=model["cutoff"], horizon=h,
                                 t=model["cutoff"] + h,
                                 y_pred=self.fn(model["t"], model["y"], model["cutoff"] + h, **self.kw)))


class BassAnalogy:
    """Bass + analogy prior: median Bass (log p, log q) of the OTHER series at the same cutoff becomes a penalty.
    Classic marketing method (Sultan-Farley-Lehmann type), leak free: only data with t <= cutoff.
    `group_col=` (a history column, e.g. "kind") takes donors only from series with the target's value there, as in
    AdoptionABCAnalogy; fewer than 3 donors -> no prior (plain Bass NLS) with a warning. None pools every series."""
    kind = "baseline"
    name = "bass_analogy"

    def __init__(self, weight=1.0, min_points=6, upper=UPPER, group_col=None):
        self.weight, self.min_points, self.upper, self.group_col = weight, min_points, upper, group_col
        self._cache: dict = {}

    def _same_group(self, history, series):
        """Series names in the target's group (None without group_col); a NaN group has no donors."""
        if self.group_col is None:
            return None
        if self.group_col not in history.columns:
            raise ValueError(f"group_col={self.group_col!r} is not a column of history")
        grp = history.groupby("series")[self.group_col].first()
        g = grp.get(series)
        return set() if pd.isna(g) else set(grp.index[grp == g])

    def _prior(self, history, cutoff, exclude):
        key = (cutoff, len(history))
        if key not in self._cache:
            pq = {}
            for s, g in history.groupby("series"):
                if len(g) >= self.min_points and g["y"].max() >= 5:
                    t = g["t"].to_numpy(float)
                    _, p = bass(t, g["y"].to_numpy(float), t[-1:], return_params=True, upper=self.upper)
                    if p is not None:
                        pq[s] = (np.log(p["p"]), np.log(p["q"]))
            self._cache[key] = pq
        same = self._same_group(history, exclude)
        pq = [v for s, v in self._cache[key].items() if s != exclude and (same is None or s in same)]
        if len(pq) < 3:
            if same is not None:
                warnings.warn(f"BassAnalogy: {len(pq)} donor series in {exclude!r}'s {self.group_col} group "
                              f"(< 3); no analogy prior, plain Bass NLS", stacklevel=3)
            return None
        return np.median(np.array(pq), axis=0)

    def fit(self, history, series, cutoff):
        t, y = _series(history, series)
        return dict(series=series, cutoff=cutoff, t=t, y=y, prior=self._prior(history, cutoff, series))

    def predict(self, model, horizons):
        h = np.asarray(horizons, float)
        pr = model["prior"]
        y = bass(model["t"], model["y"], model["cutoff"] + h, prior=pr,
                 prior_weight=self.weight if pr is not None else 0, upper=self.upper)
        return pd.DataFrame(dict(series=model["series"], cutoff=model["cutoff"], horizon=h,
                                 t=model["cutoff"] + h, y_pred=y))
