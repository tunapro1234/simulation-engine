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

Alignment: each simulation's crossing time tau1 of the alignment level (`threshold`, a fraction, default 0.01 = 1%)
is found. In the data, t0 = first observed year with y >= that level; the true crossing lies in (t0 - d, t0],
d = gap to the previous observation (at most 10 years). Calendar year t maps to simulation time tau1 + (t - t0) +
delta, with delta on a 0.25-year grid in [0, d). A series that has not reached the level by the cutoff cannot be
aligned: `fit` raises :class:`IneligibleSeries` (a ValueError). So does a cutoff or forecast more than R_MAX = 32
years after the crossing (bank window; calendar origins on old series). `threshold=0.001` keeps more early series
in but is worse where 1% works: on synthetic Bass series that had reached both levels, aligning at 0.1% raised
MAE (ABM 7.7 -> 9.0, Bass control 3.8 -> 4.2) and halved the ABM's 80% band coverage (0.50 -> 0.26). Use it only
as the fallback for the series 1% cannot align (on_ineligible="nan", then fill), not for the whole panel.

Backtests: `adoption_backtest` cuts at t0(s) + k (relative, research task M1) or at fixed calendar years
(`calendar_origins=[1991, ..., 1996]`, one origin for every series). `on_ineligible="skip"` drops the
(series, origin) blocks a method cannot align and lists them in `frame.attrs`; `fit_many` does the same for one
cutoff.

    frame = adoption_backtest(df, {"abm": AdoptionABCAnalogy(), "abm01": AdoptionABCAnalogy(threshold=0.001),
                                   "growth": CurveBaseline("growth")},
                              calendar_origins=range(1991, 1997), horizon=5, on_ineligible="nan")
    frame.attrs["n_ineligible"], frame.attrs["ineligible_series"]   # per method: frame.attrs["ineligible"]
    frame["pred_abm_fb"] = frame["pred_abm"].fillna(frame["pred_abm01"]).fillna(frame["pred_growth"])
    fm = fit_many(AdoptionABCAnalogy(), history, cutoff=1997)          # fm.models, fm.ineligible, fm.n_ineligible

Leakage: a fit only sees `history` (all series, t <= cutoff). m is never fixed; it comes from the prior.

Options (defaults leave the output unchanged):

    AdoptionABC(scale=2.0)                    # y per 100 people up to 200: the bank's m <= 1 means 2 x 100
    AdoptionABCAnalogy(scale={s: c / 100 for s, c in prior["ceiling"].items()})   # LLM ceiling (abm.ceiling)
    AdoptionABCAnalogy(group_col="kind")      # analogy donors only from the target's own kind
    sc = abm.scenario(model, {"imitation": 1.2}, anchor=best_forecast)   # y_anchor = best forecast + ABM delta
    sc.attrs["origin_fit_error"]              # the ABM baseline's miss at the last observed year (data units)

Measured (makro/abm/m1/NOTLAR.md and makro/RAPOR.md M1): OWID internet TEST, 7 seeds, MAE pp:
AdoptionABCAnalogy 8.48 +/- 0.16, BassABCAnalogy 9.73 +/- 0.12 (paired diff -1.25 +/- 0.21 per seed;
seed-averaged series bootstrap -1.32, 95% CI [-1.83, -0.82]), linear trend 10.72, Gompertz 11.61, Bass NLS 14.86, naive 18.39. US household technologies (dev): AdoptionABC
8.54 vs linear 8.70 (not significant) and the analogy prior HURTS there (10.07).
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
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
ON_INELIGIBLE = ("raise", "skip", "nan")


class IneligibleSeries(ValueError):
    """The series has not reached the alignment level by the cutoff (ABC methods cannot align it)."""

    def __init__(self, series, cutoff, threshold, reason=None):
        self.series, self.cutoff, self.threshold = series, cutoff, threshold
        super().__init__(reason or (f"series {series!r} has not reached {100 * threshold:g}% by cutoff {cutoff}; "
                                    f"cannot align (lower threshold=, or on_ineligible='skip' in adoption_backtest / "
                                    f"fit_many)"))


def _past_window(series, cutoff, threshold, last_rel):
    """Bank curves end R_MAX years after the alignment crossing; past that _grid_idx would clip silently (a flat
    forecast at the grid-end value). Reachable with calendar origins on series that crossed decades earlier."""
    return IneligibleSeries(series, cutoff, threshold,
                            f"series {series!r} at cutoff {cutoff}: needs the curve {last_rel:g} years after its "
                            f"{100 * threshold:g}% crossing, the bank covers {R_MAX:g}; cannot align")


def _check_scale(scale):
    """None, a positive number, or {series: positive number}."""
    if scale is None:
        return None
    if isinstance(scale, dict):
        out = {k: float(v) for k, v in scale.items()}
        if not all(np.isfinite(v) and v > 0 for v in out.values()):
            raise ValueError("scale values must be positive numbers (data units per percent point / 1)")
        return out
    scale = float(scale)
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be a positive number (2.0 = series up to 200), got {scale!r}")
    return scale


def _anchor_at(anchor, ts) -> np.ndarray:
    """Anchor forecast values at times ts (NaN where missing). Frame with t and y_pred (or y), dict or Series."""
    if isinstance(anchor, pd.DataFrame):
        col = "y_pred" if "y_pred" in anchor.columns else "y"
        if "t" not in anchor.columns or col not in anchor.columns:
            raise ValueError("anchor frame needs columns t and y_pred (or y)")
        anchor = anchor.groupby("t")[col].mean()
    elif isinstance(anchor, dict):
        anchor = pd.Series(anchor, dtype=float)
    elif not isinstance(anchor, pd.Series):
        raise TypeError("anchor: a frame with t, y_pred; a {t: value} dict; or a Series indexed by t")
    a = pd.Series(anchor.to_numpy(float), index=np.asarray(anchor.index, float))
    return a.reindex(np.asarray(ts, float)).to_numpy(float)


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
def _crossing(z0, path, level=1.0):
    tau = np.full(len(path), np.nan)
    for b in range(len(path)):
        k = np.argmax(path[b] >= level)
        if path[b, k] >= level and k > 0:
            y0, y1 = path[b, k - 1], path[b, k]
            tau[b] = z0[k - 1] + (level - y0) / max(y1 - y0, 1e-9) * (z0[k] - z0[k - 1])
    return tau


def _to_grid(theta, z0, path, inn, level=1.0):
    tau = _crossing(z0, path, level)
    ok = np.isfinite(tau) & (tau <= TAU_MAX)
    theta, path, inn, tau = theta[ok], path[ok], inn[ok], tau[ok]
    st = tau[:, None] + GRID[None, :]
    C = np.stack([np.interp(st[b], z0, path[b], left=0.0) for b in range(len(tau))]) if len(tau) else \
        np.zeros((0, len(GRID)))
    Y = np.stack([np.interp(st[b], z0, inn[b], left=0.0) for b in range(len(tau))]) if len(tau) else \
        np.zeros((0, len(GRID)))
    return dict(theta=theta, C=C.astype(np.float32), Y=Y.astype(np.float32), tau1=tau)


def simulation_bank(n_sim=2000, seed=0, n_agents=2000, simulator="abm", chunk=250, cache_dir=None, align=1.0):
    """Prior simulation bank: dict theta (B, 8), C (B, G) curve on GRID (%), Y (B, G) innovator part (%), tau1 (B,).
    Curves are aligned at their crossing of `align` percent (default 1%); only simulations crossing it before
    TAU_MAX are kept. Cached in memory; on disk too if `cache_dir` is given."""
    key = (simulator, n_sim, n_agents if simulator == "abm" else 0, seed)
    if align != 1.0:                                      # default key unchanged: existing disk caches stay valid
        key = key + (float(align),)
    if key in _BANKS:
        return _BANKS[key]
    path = None
    if cache_dir is not None:
        tag = hashlib.md5(repr((key, ENGINE_VERSION)).encode()).hexdigest()[:8]
        lv = "" if align == 1.0 else f"_l{align:g}"
        path = Path(cache_dir) / f"adoption_bank_{simulator}_n{n_sim}_a{key[2]}_s{seed}{lv}_{tag}.npz"
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
    out = _to_grid(theta, z0, P, Inn, align)
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

    def __init__(self, n_sim=2000, n_agents=2000, accept=0.01, recent_weight=1.0, seed=0, cache_dir=None,
                 threshold=0.01, scale=None):
        if not 0 < threshold < 1:
            raise ValueError(f"threshold is a fraction of the population (0.01 = 1%), got {threshold!r}")
        if threshold > 0.05:                              # 0.1 here is 10%, not 0.1% (adoption_t0 takes percent)
            warnings.warn(f"threshold={threshold!r} aligns at {100 * threshold:g}% of the population; "
                          f"for 0.1% pass 0.001", stacklevel=2)
        self.n_sim, self.n_agents, self.accept, self.recent_weight = n_sim, n_agents, accept, recent_weight
        self.seed, self.cache_dir, self.threshold = seed, cache_dir, threshold
        self.scale = _check_scale(scale)
        self._last = None

    def _scale_of(self, series) -> float:
        """Data units per internal percent point for `series` (1.0 = the bank's 0-100 scale)."""
        if self.scale is None:
            return 1.0
        if isinstance(self.scale, dict):
            return float(self.scale.get(series, 1.0))
        return self.scale

    def _scaled(self, history):
        """history with y divided by each series' scale (unchanged object when there is no scale)."""
        history = _norm(history)
        if self.scale is None or self.scale == 1.0:
            return history
        f = history["series"].map(self._scale_of).astype(float)
        return history.assign(y=history["y"] / f)

    @property
    def level(self) -> float:
        """Alignment level in percent (y units)."""
        return 100.0 * self.threshold

    def bank(self):
        return simulation_bank(self.n_sim, self.seed, self.n_agents, simulator=self.simulator,
                               cache_dir=self.cache_dir, align=self.level)

    def eligible(self, history, series, cutoff) -> bool:
        """True if fit will not raise IneligibleSeries: the series reached the alignment level by the cutoff, and the
        cutoff is within the bank window (R_MAX years after that crossing). predict also raises past the window."""
        try:
            self._target(self._scaled(history), series, cutoff)
        except IneligibleSeries:
            return False
        return True

    def _target(self, history, series, cutoff):
        history = _norm(history)
        g = history[(history["series"] == series) & (history["t"] <= cutoff)]
        t0 = _first_cross(g, self.level)
        if t0 is None:
            raise IneligibleSeries(series, cutoff, self.threshold)
        rel, y, deltas = _series_info(g, t0, cutoff)
        if rel.max() + deltas.max() > R_MAX:
            raise _past_window(series, cutoff, self.threshold, rel.max() + deltas.max())
        return history, t0, rel, y, deltas

    def _model(self, B, series, cutoff, t0, rel, y, b_idx, d_val, w):
        w = np.asarray(w, float)
        self._last = dict(series=series, cutoff=cutoff, t0=t0, b=np.asarray(b_idx), delta=np.asarray(d_val),
                          w=w / w.sum(), rel=rel, y=y, theta=B["theta"][b_idx], seed=self.seed)
        if self.scale is not None:
            self._last["scale"] = self._scale_of(series)          # rel / y stay in internal units
        return self._last

    def fit(self, history, series, cutoff):
        B = self.bank()
        _, t0, rel, y, deltas = self._target(self._scaled(history), series, cutoff)
        D = abc_distance(B["C"], rel, y, deltas, _recent_weights(rel, self.recent_weight))
        k = min(max(int(self.accept * D.size), 20), D.size - 1)
        flat = np.argpartition(D.ravel(), k)[:k]
        b_idx, d_idx = np.unravel_index(flat, D.shape)
        return self._model(B, series, cutoff, t0, rel, y, b_idx, deltas[d_idx], np.ones(k))

    def predict(self, model, horizons):
        B = self.bank()
        h = np.asarray(horizons, float)
        target_rel = model["cutoff"] + h - model["t0"]
        if target_rel.max() + model["delta"].max() > R_MAX:
            raise _past_window(model["series"], model["cutoff"], self.threshold,
                               target_rel.max() + model["delta"].max())
        idx = _grid_idx(target_rel[None, :] + model["delta"][:, None])          # (K, H)
        V = B["C"][model["b"][:, None], idx]
        q = _wquantile(V, model["w"], [0.1, 0.5, 0.9])
        sc = model.get("scale", 1.0)
        if sc != 1.0:
            q = q * sc
        return pd.DataFrame(dict(series=model["series"], cutoff=model["cutoff"], horizon=h,
                                 t=model["cutoff"] + h, y_pred=q[1], lo80=q[0], hi80=q[2]))

    # -- counterfactual: re-simulate accepted thetas with an intervention (common random numbers)
    def scenario(self, model=None, intervention=None, n_draws=200, *, delta=False, anchor=None):
        """intervention: {"imitation": 1.2} (q x1.2), {"market": 1.1} (m x1.1, capped at 1), {"innovation": 1.5}
        (p x1.5), or set {"threshold"|"expand"|"global"|"het": value}. Returns scenario, t, y_pred, lo80, hi80
        for the baseline and the intervention. Model-implied, not a measured causal effect.

        The ABM's own level can sit far from the data (G10: its baseline missed the origin year), so read the
        intervention as a change. delta=True adds delta, delta_lo80, delta_hi80: the paired (same draw, same random
        numbers) intervention - baseline difference, median and 10%/90% quantiles; 0 on the baseline rows.
        anchor = your best forecast (a frame with t and y_pred or y, a {t: value} dict or a Series indexed by t)
        implies delta and adds y_anchor = anchor + delta (NaN where the anchor has no t). Either way
        frame.attrs["origin_fit_error"] = baseline median - observed y at the last observed year (data units)."""
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
        sc = m.get("scale", 1.0)
        rows, draws = [], []
        for label, Y in (("baseline", y1), (str(intervention), y2)):
            vals = []
            for i in range(k):
                yb = np.r_[0.0, y1[i]]
                kk = np.argmax(yb >= self.level)
                tau1 = z0[kk] if yb[kk] >= self.level else z0[-1]
                st = tau1 + (ts - m["t0"]) + m["delta"][sel[i]]
                vals.append(np.interp(st, z0, np.r_[0.0, Y[i]], left=0.0))
            vals = np.array(vals)
            if sc != 1.0:
                vals = vals * sc
            draws.append(vals)
            q = np.quantile(vals, [0.1, 0.5, 0.9], axis=0)
            rows.append(pd.DataFrame(dict(scenario=label, t=ts, y_pred=q[1], lo80=q[0], hi80=q[2])))
        if not delta and anchor is None:
            return pd.concat(rows, ignore_index=True)
        d = np.quantile(draws[1] - draws[0], [0.1, 0.5, 0.9], axis=0)
        zero = np.zeros(len(ts))
        for r, (lo, mid, hi) in zip(rows, ((zero, zero, zero), (d[0], d[1], d[2]))):
            r["delta"], r["delta_lo80"], r["delta_hi80"] = mid, lo, hi
        if anchor is not None:
            a = _anchor_at(anchor, ts)
            for r in rows:
                r["y_anchor"] = a + r["delta"].to_numpy()
        out = pd.concat(rows, ignore_index=True)
        last = int(np.argmax(m["rel"]))
        out.attrs["origin_fit_error"] = float(np.interp(m["t0"] + m["rel"][last], ts, rows[0]["y_pred"])
                                              - m["y"][last] * sc)
        return out

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

    def __init__(self, min_points=6, analogy_accept=0.01, group_col=None, **kw):
        super().__init__(**kw)
        self.min_points, self.analogy_accept, self.group_col = min_points, analogy_accept, group_col
        self._priors: dict = {}

    def _donors(self, history, series, per):
        """The other series' prior vectors; with group_col only those of the target's own group (one kind)."""
        if self.group_col is None:
            return [v for s2, v in per.items() if s2 != series]
        if self.group_col not in history.columns:
            raise ValueError(f"group_col={self.group_col!r} is not a column of history")
        grp = history.groupby("series")[self.group_col].first()
        g = grp.get(series)
        return [v for s2, v in per.items() if s2 != series and pd.notna(g) and grp.get(s2) == g]

    def _build_prior(self, history, cutoff):
        B = self.bank()
        h = history[history["t"] <= cutoff]
        key = (cutoff, self.seed, self.n_sim, self.n_agents, self.threshold, len(h),
               tuple(sorted(map(str, h["series"].unique()))))
        if key in self._priors:
            return self._priors[key]
        nB = len(B["theta"])
        per = {}
        for s2, g in h.groupby("series"):
            if len(g) < self.min_points or g["y"].max() < 5 or g.sort_values("t")["y"].iloc[0] > 20:
                continue
            t0 = _first_cross(g, self.level)
            if t0 is None:
                continue
            rel, y, deltas = _series_info(g, t0, cutoff)
            keep = rel + deltas.max() <= R_MAX                  # donor: only the part the bank window covers
            rel, y = rel[keep], y[keep]
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
        history, t0, rel, y, deltas = self._target(self._scaled(history), series, cutoff)
        per = self._build_prior(history, cutoff)
        others = self._donors(history, series, per)
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
@dataclass
class FitMany:
    """`fit_many` result: fitted models by series, and the series the method could not align."""
    models: dict = field(default_factory=dict)
    ineligible: list = field(default_factory=list)

    @property
    def n_ineligible(self) -> int:
        return len(self.ineligible)


def _check_on_ineligible(on_ineligible):
    if on_ineligible not in ON_INELIGIBLE:
        raise ValueError(f"on_ineligible must be one of {ON_INELIGIBLE}, got {on_ineligible!r}")


def fit_many(method, history, cutoff, series=None, on_ineligible="skip") -> FitMany:
    """Fit `method` on every series (default: all series with data up to the cutoff). Series the method cannot align
    (IneligibleSeries) are skipped and listed (on_ineligible="skip") or re-raised ("raise")."""
    if on_ineligible not in ("raise", "skip"):
        raise ValueError(f"fit_many: on_ineligible must be 'raise' or 'skip', got {on_ineligible!r}")
    history = _norm(history)
    hist = history[history["t"] <= cutoff]
    series = list(series) if series is not None else sorted(hist["series"].unique(), key=str)
    out = FitMany()
    for s in series:
        try:
            out.models[s] = method.fit(hist, s, cutoff)
        except IneligibleSeries:
            if on_ineligible == "raise":
                raise
            out.ineligible.append(s)
    return out


def adoption_backtest(df, methods, cutoffs=(3, 5, 8), horizon=10, series=None, t0=None, calendar_origins=None,
                      on_ineligible="raise"):
    """Cutoff backtest. Relative (default, research task M1): for every series s and k in `cutoffs`, c = t0(s) + k
    (t0 = first year >= 1%, `adoption_t0` selection). Calendar: `calendar_origins=[1991, ..., 1996]` replaces the
    relative cutoffs with the same fixed years for every series (default series: all of them); concatenate two
    calls for both. Each method sees ALL series' data with t <= c and forecasts the observed points in
    (c, c + horizon]. Requires >= 3 points up to c and >= 1 after.

    on_ineligible: what to do when a method cannot align a series at c (IneligibleSeries, e.g. ABC below its
    threshold): "raise" (default), "skip" (drop the whole (series, c) block so every method is scored on the same
    rows) or "nan" (keep the block, NaN predictions for that method; fill with a fallback yourself). Either way
    frame.attrs gets "ineligible" (list of {series, cutoff, method}), "ineligible_series" and "n_ineligible"
    (distinct series).

    Returns a long EvalFrame: series, group, cutoff, cutoff_rel (c - t0(s); in calendar mode a label only, NaN when
    s is not in t0), horizon, t, truth, pred_<name> (and lo80_/hi80_ when a method gives bands)."""
    _check_on_ineligible(on_ineligible)
    df = _norm(df)
    t0 = t0 or adoption_t0(df)
    if calendar_origins is not None:
        series = list(series) if series is not None else sorted(df["series"].unique(), key=str)
    else:
        series = list(series) if series is not None else sorted(t0)
    methods = methods if isinstance(methods, dict) else {m.name: m for m in methods}
    rows, bad = [], []
    for s in series:
        g = df[df["series"] == s]
        if calendar_origins is not None:
            cuts = [(float(c), float(c) - t0[s] if s in t0 else np.nan) for c in calendar_origins]
        else:
            cuts = [(t0[s] + k, k) for k in cutoffs]
        for c, k in cuts:
            before, after = g[g["t"] <= c], g[(g["t"] > c) & (g["t"] <= c + horizon)]
            if len(before) < 3 or len(after) < 1:
                continue
            hist = df[df["t"] <= c]
            h = (after["t"].to_numpy(float) - c)
            out = pd.DataFrame(dict(series=s, group=s, cutoff=c, cutoff_rel=k, horizon=h, t=c + h,
                                    truth=after["y"].to_numpy(float)))
            keep = True
            for name, meth in methods.items():
                if not keep:                              # block dropped: only list the other methods that cannot align
                    if hasattr(meth, "eligible") and not meth.eligible(hist, s, c):
                        bad.append(dict(series=s, cutoff=c, method=name))
                    continue
                try:
                    p = meth.predict(meth.fit(hist, s, c), h)
                except IneligibleSeries:
                    if on_ineligible == "raise":
                        raise
                    bad.append(dict(series=s, cutoff=c, method=name))
                    if on_ineligible == "skip":
                        keep = False
                        continue
                    out[f"pred_{name}"] = np.nan
                    continue
                out[f"pred_{name}"] = p["y_pred"].to_numpy()
                if "lo80" in p:
                    out[f"lo80_{name}"] = p["lo80"].to_numpy()
                    out[f"hi80_{name}"] = p["hi80"].to_numpy()
            if keep:
                rows.append(out)
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    ids = sorted({b["series"] for b in bad}, key=str)
    frame.attrs.update(ineligible=bad, ineligible_series=ids, n_ineligible=len(ids))
    return frame
