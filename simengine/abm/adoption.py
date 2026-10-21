"""Adoption-curve ABM + ABC calibration + analogy prior (ported from the TALP prep kit `abm_benimseme.py`).

Methods (all share `fit(history, series, cutoff)` / `predict(model, horizons)` / `scenario` / `personas`):

* :class:`AdoptionABC`        : a simulation bank drawn from the prior (once per seed, cached in memory and optionally
  on disk). Distance = RMSE to the target series' points up to the cutoff (recent points weighted up to 2x). The
  closest 1% of (simulation x time-shift) candidates are accepted (rejection ABC); forecast = median, 80% band =
  10%/90% quantiles.
* :class:`AdoptionABCAnalogy` : empirical prior from the OTHER series at the same calendar cutoff (each fitted with
  ABC on t <= cutoff, >= 6 points, max >= 5%); the target series is excluded from its own prior. The target
  likelihood is a Gaussian kernel on the distance. ABM counterpart of Bass + analogy.
* :class:`BassABC`, :class:`BassABCAnalogy` : the same ABC procedure with a closed-form Bass simulator (control to
  separate "ABM structure" from "calibration + prior").

Alignment: each simulation's 1% crossing time tau1 is found. In the data, t0 = first observed year with y >= 1%; the
true crossing lies in (t0 - d, t0], d = gap to the previous observation (at most 10 years). Calendar year t maps to
simulation time tau1 + (t - t0) + delta, with delta on a 0.25-year grid in [0, d).

Leakage: a fit only sees `history` (all series, t <= cutoff). m is never fixed; it comes from the prior.

Measured (makro/abm/m1/NOTLAR.md and makro/RAPOR.md M1): OWID internet TEST, 7 seeds, MAE pp:
AdoptionABCAnalogy 8.48 +/- 0.16, BassABCAnalogy 9.73 +/- 0.12 (paired diff -1.25, 95% CI [-1.83, -0.82]),
linear trend 10.72, Gompertz 11.61, Bass NLS 14.86, naive 18.39. US household technologies (dev): AdoptionABC
8.54 vs linear 8.70 (not significant) and the analogy prior HURTS there (10.07).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import engine

STEP = 0.25                     # grid (years)
R_MIN, R_MAX = -15.0, 32.0      # t - t0 + delta range
GRID = np.arange(R_MIN, R_MAX + 1e-9, STEP)
SIM_T = 45                      # engine: S = (SIM_T + 15) * substeps -> 60-year path
TAU_MAX = 28.0                  # simulations crossing 1% later than this are dropped (32 years ahead needed)
BASS_PRIOR = {"m": (0.02, 1.0), "p": (0.001, 0.05), "q": (0.1, 0.8)}   # m as ADOPTION_PRIOR; p, q wide uniform
ENGINE_VERSION = "motor-2026-10-20"

_BANKS: dict = {}


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Accept research-code column names (seri) and return series, t, y."""
    if "series" not in df.columns and "seri" in df.columns:
        df = df.rename(columns={"seri": "series"})
    missing = {"series", "t", "y"} - set(df.columns)
    if missing:
        raise ValueError(f"adoption data needs columns series, t, y; missing {sorted(missing)}")
    return df


def adoption_t0(df: pd.DataFrame, threshold=1.0, max_first=20.0, min_points=8) -> dict:
    """{series: t0}, t0 = first year with y >= threshold (%). Series whose first observation is already above
    `max_first` (start not observed) or with fewer than `min_points` points are skipped (research selection rule)."""
    df = _norm(df)
    out = {}
    for s, g in df.sort_values("t").groupby("series"):
        up = g[g["y"] >= threshold]
        if up.empty or g["y"].iloc[0] > max_first or len(g) < min_points:
            continue
        out[s] = float(up["t"].iloc[0])
    return out


# ------------------------------------------------------------------ simulation banks
def _crossing(z0, path):
    tau = np.full(len(path), np.nan)
    for b in range(len(path)):
        k = np.argmax(path[b] >= 1.0)
        if path[b, k] >= 1.0 and k > 0:
            y0, y1 = path[b, k - 1], path[b, k]
            tau[b] = z0[k - 1] + (1.0 - y0) / max(y1 - y0, 1e-9) * (z0[k] - z0[k - 1])
    return tau


def _to_grid(theta, z0, path, inn):
    tau = _crossing(z0, path)
    ok = np.isfinite(tau) & (tau <= TAU_MAX)
    theta, path, inn, tau = theta[ok], path[ok], inn[ok], tau[ok]
    st = tau[:, None] + GRID[None, :]
    C = np.stack([np.interp(st[b], z0, path[b], left=0.0) for b in range(len(tau))]) if len(tau) else \
        np.zeros((0, len(GRID)))
    Y = np.stack([np.interp(st[b], z0, inn[b], left=0.0) for b in range(len(tau))]) if len(tau) else \
        np.zeros((0, len(GRID)))
    return dict(theta=theta, C=C.astype(np.float32), Y=Y.astype(np.float32), tau1=tau)


def simulation_bank(n_sim=2000, seed=0, n_agents=2000, simulator="abm", chunk=250, cache_dir=None):
    """Prior simulation bank: dict theta (B, 8), C (B, G) curve on GRID (%), Y (B, G) innovator part (%), tau1 (B,).
    Only simulations crossing 1% before TAU_MAX are kept. Cached in memory; on disk too if `cache_dir` is given."""
    key = (simulator, n_sim, n_agents if simulator == "abm" else 0, seed)
    if key in _BANKS:
        return _BANKS[key]
    path = None
    if cache_dir is not None:
        tag = hashlib.md5(repr((key, ENGINE_VERSION)).encode()).hexdigest()[:8]
        path = Path(cache_dir) / f"adoption_bank_{simulator}_n{n_sim}_a{key[2]}_s{seed}_{tag}.npz"
        if path.exists():
            d = np.load(path)
            _BANKS[key] = {k: d[k] for k in d.files}
            return _BANKS[key]
    rng = np.random.default_rng(10_000 + seed)
    if simulator == "abm":
        theta = engine.sample_adoption_prior(n_sim, rng)
        theta[:, 7] = 0.0
        paths, inns = [], []
        for i in range(0, n_sim, chunk):
            z, y, yen = engine.adoption_sim(theta[i:i + chunk], SIM_T, n_agents=n_agents,
                                            seed=seed * 7919 + i // chunk, raw=True)
            paths.append(y)
            inns.append(yen)
        z0 = np.r_[0.0, z]
        P = np.c_[np.zeros(n_sim), np.concatenate(paths)]
        Inn = np.c_[np.zeros(n_sim), np.concatenate(inns)]
    elif simulator == "bass":
        lo = np.array([v[0] for v in BASS_PRIOR.values()])
        hi = np.array([v[1] for v in BASS_PRIOR.values()])
        m, p, q = (lo + (hi - lo) * rng.random((n_sim, 3))).T
        z0 = np.arange(0, (SIM_T + 15) * 1.0 + 1e-9, STEP)
        e = np.exp(-(p + q)[:, None] * z0[None, :])
        P = 100 * m[:, None] * (1 - e) / (1 + (q / p)[:, None] * e)
        Inn = np.concatenate([np.zeros((n_sim, 1)), np.cumsum(
            0.5 * STEP * p[:, None] * ((100 * m[:, None] - P[:, 1:]) + (100 * m[:, None] - P[:, :-1])), 1)], 1)
        # engine parametrisation: the mean-field / Bass limit of the ABM
        theta = np.c_[m, np.log(p), np.log(q), np.zeros(n_sim), np.zeros(n_sim), np.ones(n_sim),
                      np.zeros(n_sim), np.zeros(n_sim)]
    else:
        raise ValueError(f"unknown simulator {simulator!r} (abm | bass)")
    out = _to_grid(theta, z0, P, Inn)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **out)
    _BANKS[key] = out
    return out


def _grid_idx(r):
    return np.rint((np.asarray(r, float) - R_MIN) / STEP).astype(int).clip(0, len(GRID) - 1)


def _series_info(g, t0, cutoff):
    """g: one series' data with t <= cutoff. Returns observation times relative to t0, values, allowed delta grid."""
    t, y = g["t"].to_numpy(float), g["y"].to_numpy(float)
    m = (t <= cutoff) & (t >= t0 + R_MIN + 10)
    t, y = t[m], y[m]
    before = t[t < t0]
    gap = min(t0 - before.max(), 10.0) if len(before) else 10.0
    deltas = np.arange(0.0, gap - 1e-9, STEP)
    if len(deltas) == 0:
        deltas = np.array([0.0])
    return t - t0, y, deltas


def abc_distance(C, rel, y, deltas, weights=None):
    """(B, D) weighted RMSE matrix: every bank curve x every delta."""
    w = np.ones_like(y) if weights is None else weights
    w = w / w.sum()
    D = np.empty((C.shape[0], len(deltas)), np.float32)
    for j, d in enumerate(deltas):
        idx = _grid_idx(rel + d)
        D[:, j] = np.sqrt((((C[:, idx] - y[None, :]) ** 2) * w[None, :]).sum(1))
    return D


def _recent_weights(rel, recent_weight):
    """More weight on recent points: w = 1 + recent_weight * (rank / n)."""
    if not recent_weight:
        return None
    r = np.argsort(np.argsort(rel))
    return 1.0 + recent_weight * (r + 1) / len(rel)


def _wquantile(v, w, qs):
    """v: (K, H) values, w: (K,) weights; returns (len(qs), H)."""
    o = np.argsort(v, axis=0)
    out = np.empty((len(qs), v.shape[1]))
    for h in range(v.shape[1]):
        vv, ww = v[o[:, h], h], w[o[:, h]]
        cw = np.cumsum(ww) / ww.sum()
        out[:, h] = np.interp(qs, cw - ww / ww.sum() / 2, vv)
    return out


def _first_cross(g, threshold=1.0):
    g = g.sort_values("t")
    up = g[g["y"] >= threshold]
    return None if up.empty else float(up["t"].iloc[0])


class AdoptionABC:
    """Rejection ABC on a prior simulation bank (research name `abm_abc`).

    Defaults are the dev-selected research settings: bank 2,000 simulations x 2,000 agents, accept 1%, recent
    weight 1. Smaller banks (e.g. n_sim=300, n_agents=300) are fine for smoke tests but were not measured.
    """
    name = "abm_abc"
    kind = "method"
    simulator = "abm"

    def __init__(self, n_sim=2000, n_agents=2000, accept=0.01, recent_weight=1.0, seed=0, cache_dir=None):
        self.n_sim, self.n_agents, self.accept, self.recent_weight = n_sim, n_agents, accept, recent_weight
        self.seed, self.cache_dir = seed, cache_dir
        self._last = None

    def bank(self):
        return simulation_bank(self.n_sim, self.seed, self.n_agents, simulator=self.simulator,
                               cache_dir=self.cache_dir)

    def _target(self, history, series, cutoff):
        history = _norm(history)
        g = history[(history["series"] == series) & (history["t"] <= cutoff)]
        t0 = _first_cross(g)
        if t0 is None:
            raise ValueError(f"series {series!r} has not reached 1% by cutoff {cutoff}; cannot align")
        rel, y, deltas = _series_info(g, t0, cutoff)
        return history, t0, rel, y, deltas

    def _model(self, B, series, cutoff, t0, rel, y, b_idx, d_val, w):
        w = np.asarray(w, float)
        self._last = dict(series=series, cutoff=cutoff, t0=t0, b=np.asarray(b_idx), delta=np.asarray(d_val),
                          w=w / w.sum(), rel=rel, y=y, theta=B["theta"][b_idx], seed=self.seed)
        return self._last

    def fit(self, history, series, cutoff):
        B = self.bank()
        _, t0, rel, y, deltas = self._target(history, series, cutoff)
        D = abc_distance(B["C"], rel, y, deltas, _recent_weights(rel, self.recent_weight))
        k = min(max(int(self.accept * D.size), 20), D.size - 1)
        flat = np.argpartition(D.ravel(), k)[:k]
        b_idx, d_idx = np.unravel_index(flat, D.shape)
        return self._model(B, series, cutoff, t0, rel, y, b_idx, deltas[d_idx], np.ones(k))

    def predict(self, model, horizons):
        B = self.bank()
        h = np.asarray(horizons, float)
        target_rel = model["cutoff"] + h - model["t0"]
        idx = _grid_idx(target_rel[None, :] + model["delta"][:, None])          # (K, H)
        V = B["C"][model["b"][:, None], idx]
        q = _wquantile(V, model["w"], [0.1, 0.5, 0.9])
        return pd.DataFrame(dict(series=model["series"], cutoff=model["cutoff"], horizon=h,
                                 t=model["cutoff"] + h, y_pred=q[1], lo80=q[0], hi80=q[2]))

    # -- counterfactual: re-simulate accepted thetas with an intervention (common random numbers)
    def scenario(self, model=None, intervention=None, n_draws=200):
        """intervention: {"imitation": 1.2} (q x1.2), {"market": 1.1} (m x1.1, capped at 1), {"innovation": 1.5}
        (p x1.5), or set {"threshold"|"expand"|"global"|"het": value}. Returns scenario, t, y_pred, lo80, hi80
        for the baseline and the intervention. Model-implied, not a measured causal effect."""
        m = model or self._last
        if m is None:
            raise ValueError("call fit first")
        intervention = intervention or {}
        rng = np.random.default_rng(1)
        k = min(len(m["b"]), n_draws)
        sel = rng.choice(len(m["b"]), k, replace=len(m["b"]) < k, p=m["w"])
        th = np.array(m["theta"][sel], float)
        th2 = th.copy()
        for name, v in intervention.items():
            if name == "imitation":
                th2[:, 2] += np.log(v)
            elif name == "innovation":
                th2[:, 1] += np.log(v)
            elif name == "market":
                th2[:, 0] = np.minimum(th2[:, 0] * v, 1.0)
            elif name in ("threshold", "expand", "global", "het"):
                th2[:, engine.ADOPTION_PARAMS.index(name)] = v
            else:
                raise ValueError(f"unknown intervention: {name}")
        z, y1, _ = engine.adoption_sim(th, SIM_T, n_agents=self.n_agents, seed=99, raw=True)
        _, y2, _ = engine.adoption_sim(th2, SIM_T, n_agents=self.n_agents, seed=99, raw=True)
        z0 = np.r_[0.0, z]
        ts = np.arange(m["t0"] - 3, m["cutoff"] + 11)
        rows = []
        for label, Y in (("baseline", y1), (str(intervention), y2)):
            vals = []
            for i in range(k):
                yb = np.r_[0.0, y1[i]]
                kk = np.argmax(yb >= 1.0)
                tau1 = z0[kk] if yb[kk] >= 1.0 else z0[-1]
                st = tau1 + (ts - m["t0"]) + m["delta"][sel[i]]
                vals.append(np.interp(st, z0, np.r_[0.0, Y[i]], left=0.0))
            q = np.quantile(np.array(vals), [0.1, 0.5, 0.9], axis=0)
            rows.append(pd.DataFrame(dict(scenario=label, t=ts, y_pred=q[1], lo80=q[0], hi80=q[2])))
        return pd.concat(rows, ignore_index=True)

    def personas(self, model=None):
        """Innovator / early imitator / late imitator / non-adopter shares 10 years after the cutoff (weighted mean
        over accepted samples). These are model-implied attributions, not observed; label them so."""
        m = model or self._last
        if m is None:
            return []
        B = self.bank()
        idx = _grid_idx(m["cutoff"] + 10 - m["t0"] + m["delta"])
        top = B["C"][m["b"], idx] / 100
        inn = B["Y"][m["b"], idx] / 100
        imit = top - inn
        half = np.array([B["C"][b, :i + 1].searchsorted(B["C"][b, i] / 2) for b, i in zip(m["b"], idx)])
        half = half.clip(0, len(GRID) - 1)
        early = np.clip(B["C"][m["b"], half] / 100 - B["Y"][m["b"], half] / 100, 0, None)
        w = m["w"]

        def avg(v):
            return float((v * w).sum())

        yr50 = avg(GRID[half] - m["delta"])
        return [
            dict(name="Innovator (external influence)", share=round(avg(inn), 3),
                 description="Adopts from media/marketing pressure, independent of peers.",
                 response=f"p ~ {np.exp(avg(m['theta'][:, 1])):.4f}/yr"),
            dict(name="Early imitator", share=round(avg(early), 3),
                 description=f"Adopts once neighbours do, before the curve's half point (~ t0+{yr50:.0f} yr).",
                 response=f"q ~ {np.exp(avg(m['theta'][:, 2])):.2f}/yr"),
            dict(name="Late imitator", share=round(max(avg(imit) - avg(early), 0), 3),
                 description="Adopts after the half point, mostly through social pressure.",
                 response=f"threshold ~ {avg(m['theta'][:, 4]):.2f}"),
            dict(name="Non-adopter (by horizon)", share=round(max(1 - avg(top), 0), 3),
                 description=f"Outside the market or still waiting; long-run ceiling m ~ {avg(m['theta'][:, 0]):.2f}.",
                 response="0"),
        ]


class AdoptionABCAnalogy(AdoptionABC):
    """Empirical prior = union of the OTHER series' ABC acceptance sets at the same cutoff (equal weight per series).
    Target likelihood = exp(-0.5 (d/h)^2), h = distance quantile covering the acceptance fraction of the prior
    (research name `abm_abc_analoji`). Works when the series are of one kind (countries x one technology);
    across unrelated technologies it hurts (US households dev 10.07 vs plain ABC 8.54)."""
    name = "abm_abc_analogy"

    def __init__(self, min_points=6, analogy_accept=0.01, **kw):
        super().__init__(**kw)
        self.min_points, self.analogy_accept = min_points, analogy_accept
        self._priors: dict = {}

    def _build_prior(self, history, cutoff):
        B = self.bank()
        h = history[history["t"] <= cutoff]
        key = (cutoff, self.seed, self.n_sim, self.n_agents, len(h), tuple(sorted(map(str, h["series"].unique()))))
        if key in self._priors:
            return self._priors[key]
        nB = len(B["theta"])
        per = {}
        for s2, g in h.groupby("series"):
            if len(g) < self.min_points or g["y"].max() < 5 or g.sort_values("t")["y"].iloc[0] > 20:
                continue
            t0 = _first_cross(g)
            if t0 is None:
                continue
            rel, y, deltas = _series_info(g, t0, cutoff)
            if len(rel) < 3:
                continue
            D = abc_distance(B["C"], rel, y, deltas, _recent_weights(rel, self.recent_weight)).min(1)
            k = min(max(int(self.analogy_accept * nB), 10), nB - 1)
            b = np.argpartition(D, k)[:k]
            v = np.zeros(nB, np.float32)
            v[b] = 1.0 / k
            per[s2] = v
        self._priors[key] = per
        return per

    def fit(self, history, series, cutoff):
        B = self.bank()
        history, t0, rel, y, deltas = self._target(history, series, cutoff)
        per = self._build_prior(history, cutoff)
        others = [v for s2, v in per.items() if s2 != series]
        D = abc_distance(B["C"], rel, y, deltas, _recent_weights(rel, self.recent_weight))
        prior = np.ones(len(B["theta"]), np.float32) if len(others) < 3 else np.sum(others, axis=0)
        cand = np.flatnonzero(prior > 0)
        Da = D[cand]
        wp = np.repeat(prior[cand][:, None], len(deltas), 1)
        o = np.argsort(Da.ravel())
        cw = np.cumsum(wp.ravel()[o]) / wp.sum()
        pos = min(np.searchsorted(cw, max(self.accept * 3, 0.05)), len(o) - 1)
        bw = max(float(Da.ravel()[o][pos]), 0.25)
        lik = np.exp(-0.5 * (Da / bw) ** 2)
        w = (wp * lik).ravel()
        k = min(4000, w.size)
        sel = np.argpartition(-w, k - 1)[:k]
        sel = sel[w[sel] > 0]
        a_idx, d_idx = np.unravel_index(sel, Da.shape)
        return self._model(B, series, cutoff, t0, rel, y, cand[a_idx], deltas[d_idx], w[sel])


class BassABC(AdoptionABC):
    """Control: identical ABC, but the simulator is the closed-form Bass curve (research name `abc_bass`)."""
    name = "abc_bass"
    kind = "control"
    simulator = "bass"


class BassABCAnalogy(AdoptionABCAnalogy):
    """Control: identical analogy ABC with the Bass simulator (research name `abc_bass_analoji`)."""
    name = "abc_bass_analogy"
    kind = "control"
    simulator = "bass"


# ------------------------------------------------------------------ backtest runner
def adoption_backtest(df, methods, cutoffs=(3, 5, 8), horizon=10, series=None, t0=None):
    """Calendar-cutoff backtest (research task M1). For every series s and relative cutoff k, c = t0(s) + k; each
    method sees ALL series' data with t <= c and forecasts the observed points in (c, c + horizon].
    Requires >= 3 points up to c and >= 1 after. Returns a long EvalFrame: series, group, cutoff, cutoff_rel,
    horizon, t, truth, pred_<name> (and lo80_/hi80_ when a method gives bands)."""
    df = _norm(df)
    t0 = t0 or adoption_t0(df)
    series = list(series) if series is not None else sorted(t0)
    methods = methods if isinstance(methods, dict) else {m.name: m for m in methods}
    rows = []
    for s in series:
        g = df[df["series"] == s]
        for k in cutoffs:
            c = t0[s] + k
            before, after = g[g["t"] <= c], g[(g["t"] > c) & (g["t"] <= c + horizon)]
            if len(before) < 3 or len(after) < 1:
                continue
            hist = df[df["t"] <= c]
            h = (after["t"].to_numpy(float) - c)
            out = pd.DataFrame(dict(series=s, group=s, cutoff=c, cutoff_rel=k, horizon=h, t=c + h,
                                    truth=after["y"].to_numpy(float)))
            for name, meth in methods.items():
                p = meth.predict(meth.fit(hist, s, c), h)
                out[f"pred_{name}"] = p["y_pred"].to_numpy()
                if "lo80" in p:
                    out[f"lo80_{name}"] = p["lo80"].to_numpy()
                    out[f"hi80_{name}"] = p["hi80"].to_numpy()
            rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
