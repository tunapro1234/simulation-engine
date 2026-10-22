"""Multi-series demand forecasting: rule baselines, leak-free lag features, a direct multi-horizon GBDT,
rolling-origin backtest, metric-optimal point forecasts and an LLM prior feature.

Port of the D4 blind rehearsal code (M5 weekly demand, 4 weeks ahead): ``kod/ozellik.py`` (lag / rolling
features, rule ladder T1-T4), ``kod/ses.py`` (simple exponential smoothing, alpha tuned only on origins whose
targets are <= the forecast origin), ``kod/kos.py`` (one LightGBM per horizon, Tweedie 1.2), ``kod/pm.py``
(Poisson median post-transform for L1 metrics), ``kod/ek.py`` (rule baselines over rolling origins) and
``kod/llm_olay.py`` (named / blind LLM prior arms).

Panel layout (long): one row per (series, time) with columns ``id_col``, ``time_col``, ``y_col`` and optional
exogenous columns. ``time_col`` must be sortable (int weeks, dates). The *origin* is the last time whose
``y`` is known; horizon ``h`` is the h-th period after it. Missing (series, time) cells are read as
``fill_value`` (0.0 = no sales, the demand convention). Rows after the origin may exist with ``y`` NaN to carry
known-in-advance exogenous values (price, calendar).

Every forecaster returns a long frame ``[id_col, "origin", "h", time_col, "pred"]``.
"""
from __future__ import annotations

import inspect
import re
import warnings
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ._util import parse_json

SES_ALPHAS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7)     # kod/ses.py ALFA grid
L1_METRICS = {"mae", "wmape", "wape", "l1", "mad", "smae"}
L2_METRICS = {"mse", "rmse", "l2"}


# ----------------------------------------------------------------------------- panel helpers
def _time_grid(times):
    """Complete regular grid spanning ``times`` (step = smallest gap, int or datetime times), or None when the
    times are not on such a grid (strings, uneven dates)."""
    t = pd.Index(times)
    if len(t) < 2:
        return None
    if pd.api.types.is_integer_dtype(t.dtype):
        v = t.to_numpy(np.int64)
        step = int(np.diff(v).min())
        if step > 0 and np.all((v - v[0]) % step == 0):
            return pd.Index(np.arange(v[0], v[-1] + step, step, dtype=v.dtype))
        return None
    if isinstance(t, pd.DatetimeIndex):
        step = pd.Series(t).diff().min()
        if step <= pd.Timedelta(0):
            return None
        g = pd.date_range(t[0], t[-1], freq=step)
        return g if t.isin(g).all() else None
    return None


class _Wide:
    """Wide view of a long panel: ids x times matrix (raw with NaN, and filled)."""

    def __init__(self, panel: pd.DataFrame, id_col: str, time_col: str, y_col: str, fill_value: float = 0.0):
        for c in (id_col, time_col, y_col):
            if c not in panel.columns:
                raise KeyError(f"panel has no column {c!r}")
        piv = panel.pivot_table(index=id_col, columns=time_col, values=y_col, aggfunc="first", dropna=False)
        piv = piv.sort_index(axis=1)
        grid = _time_grid(piv.columns)
        if grid is not None and len(grid) > len(piv.columns):
            # a time with no rows at all must still be a column, or h shifts by one period
            piv = piv.reindex(columns=grid)
        self.ids = piv.index.to_numpy()
        self.times = piv.columns.to_numpy()
        self.raw = piv.to_numpy(dtype=float)
        self.Y = np.where(np.isnan(self.raw), fill_value, self.raw)
        self.panel, self.id_col, self.time_col, self.y_col = panel, id_col, time_col, y_col
        self._tpos = {t: i for i, t in enumerate(self.times)}
        self._exog: dict[str, pd.DataFrame] = {}

    def exog(self, col: str) -> pd.DataFrame:
        if col not in self._exog:
            self._exog[col] = (self.panel.pivot_table(index=self.id_col, columns=self.time_col, values=col,
                                                      aggfunc="first", dropna=False)
                               .reindex(index=self.ids, columns=self.times))
        return self._exog[col]

    def pos(self, origin) -> int:
        if origin not in self._tpos:
            raise KeyError(f"origin {origin!r} is not a time in the panel")
        return self._tpos[origin]

    def time_at(self, p: int):
        """Time label at position p, extrapolated with the median step for numeric times beyond the panel."""
        if 0 <= p < len(self.times):
            return self.times[p]
        t = self.times
        if len(t) >= 2 and np.issubdtype(np.asarray(t).dtype, np.number):
            step = np.median(np.diff(t.astype(float)))
            out = t[-1] + (p - (len(t) - 1)) * step
            return type(t[-1])(out) if np.issubdtype(np.asarray(t).dtype, np.integer) else out
        if len(t) >= 2 and np.issubdtype(np.asarray(t).dtype, np.datetime64):
            step = pd.Series(t).diff().min()
            return pd.Timestamp(t[-1]) + (p - (len(t) - 1)) * step
        return np.nan

    def frame(self, origin, preds: np.ndarray, horizons: Sequence[int]) -> pd.DataFrame:
        """preds: (n_series, len(horizons)) -> long frame."""
        p = self.pos(origin)
        n = len(self.ids)
        rows = []
        for j, h in enumerate(horizons):
            rows.append(pd.DataFrame({self.id_col: self.ids, "origin": origin, "h": int(h),
                                      self.time_col: [self.time_at(p + h)] * n, "pred": preds[:, j]}))
        return pd.concat(rows, ignore_index=True)


def _horizons(horizon: int | Sequence[int]) -> list[int]:
    hs = list(range(1, int(horizon) + 1)) if np.isscalar(horizon) else [int(h) for h in horizon]
    if not hs or min(hs) < 1:
        raise ValueError("horizons must be >= 1")
    return hs


def mask_future(panel: pd.DataFrame, origin, *, time_col: str = "time", y_col: str = "y") -> pd.DataFrame:
    """Copy of the panel with ``y`` set to NaN after the origin (exogenous columns are kept)."""
    out = panel.copy()
    out.loc[out[time_col] > origin, y_col] = np.nan
    return out


def wmape(truth, pred) -> float:
    """sum|y - y_hat| / sum y (ratio of sums, the M5 / D4 primary metric)."""
    y, p = np.asarray(truth, float), np.asarray(pred, float)
    ok = ~(np.isnan(y) | np.isnan(p))
    return float(np.abs(y[ok] - p[ok]).sum() / y[ok].sum())


def bias(truth, pred) -> float:
    """sum y_hat / sum y - 1 (report next to WMAPE: a median point lowers WMAPE but biases totals down)."""
    y, p = np.asarray(truth, float), np.asarray(pred, float)
    ok = ~(np.isnan(y) | np.isnan(p))
    return float(p[ok].sum() / y[ok].sum() - 1.0)


# ----------------------------------------------------------------------------- rule baselines
def _baseline(compute: Callable[[np.ndarray, int, list[int]], np.ndarray]):
    def run(panel, origin, horizon=4, *, id_col="id", time_col="time", y_col="y", fill_value=0.0):
        w = _Wide(panel, id_col, time_col, y_col, fill_value)
        p, hs = w.pos(origin), _horizons(horizon)
        return w.frame(origin, compute(w.Y[:, :p + 1], p, hs), hs)
    return run


def persistence_forecast(panel: pd.DataFrame, origin, horizon: int | Sequence[int] = 4, *, id_col: str = "id",
                         time_col: str = "time", y_col: str = "y", fill_value: float = 0.0) -> pd.DataFrame:
    """Last known value for every horizon (D4 ladder T1). Hidden D4 dev reference: 0.418 WMAPE."""
    return _baseline(lambda Y, p, hs: np.repeat(Y[:, p:p + 1], len(hs), 1))(
        panel, origin, horizon, id_col=id_col, time_col=time_col, y_col=y_col, fill_value=fill_value)


persistence = persistence_forecast      # module-level alias; the package exports persistence_forecast because
                                        # methods.persistence is the older history-dict baseline


def seasonal_naive(panel: pd.DataFrame, origin, horizon: int | Sequence[int] = 4, *, season: int = 52,
                   id_col: str = "id", time_col: str = "time", y_col: str = "y",
                   fill_value: float = 0.0) -> pd.DataFrame:
    """Value one season before the target (D4 ladder T2, season=52 weeks). NaN when history is too short."""
    def f(Y, p, hs):
        out = np.full((Y.shape[0], len(hs)), np.nan)
        for j, h in enumerate(hs):
            k = p + h - season * int(np.ceil(h / season))
            if k >= 0:
                out[:, j] = Y[:, k]
        return out
    return _baseline(f)(panel, origin, horizon, id_col=id_col, time_col=time_col, y_col=y_col, fill_value=fill_value)


def moving_mean(panel: pd.DataFrame, origin, horizon: int | Sequence[int] = 4, *, window: int = 4,
                id_col: str = "id", time_col: str = "time", y_col: str = "y", fill_value: float = 0.0) -> pd.DataFrame:
    """Mean of the last ``window`` periods up to the origin (D4 ladder T3, window=4: hidden dev ref. 0.379)."""
    def f(Y, p, hs):
        m = Y[:, max(p - window + 1, 0):p + 1].mean(1)
        return np.repeat(m[:, None], len(hs), 1)
    return _baseline(f)(panel, origin, horizon, id_col=id_col, time_col=time_col, y_col=y_col, fill_value=fill_value)


def seasonal_ratio(panel: pd.DataFrame, origin, horizon: int | Sequence[int] = 4, *, window: int = 4,
                   season: int = 52, cap: tuple[float, float] | None = (0.5, 2.0), around: int = 1,
                   smooth: float = 1.0, group_col: str | None = None, id_col: str = "id", time_col: str = "time",
                   y_col: str = "y", fill_value: float = 0.0) -> pd.DataFrame:
    """Last ``window`` periods mean x last year's same-period ratio (D4 ladder T4 / T4b).

    pred(h) = mean(y[p-window+1 .. p]) * ratio(h), p = the origin's position, with
    ratio(h) = (ly(h) + smooth) / (base + smooth) clipped to ``cap`` (None: no clip), where
    base = mean(y[p-window+1-season .. p-season]) (the same window one season back) and
    ly(h) = mean(y[k-around .. k+around]), k = p + h - season (the target period one season back, +-around periods;
    never past the origin). A horizon longer than the season wraps as ``seasonal_naive``: n = ceil(h / season) seasons
    back for both ly and base. Defaults are T4 (window=4, season=52, around=1, smooth=1, cap=(0.5, 2)).
    group_col: pool the ratio over a static grouping column (e.g. store x dept): sum(ly) / sum(base) over the
    group's series; T4b is ``group_col=..., around=0, smooth=0``. A zero or negative denominator gives ratio 1.
    NaN when the history is shorter than one season plus the window. Same output format as ``moving_mean``.
    """
    if window < 1 or season < 1 or around < 0:
        raise ValueError("window and season must be >= 1 and around >= 0")
    w = _Wide(panel, id_col, time_col, y_col, fill_value)
    p, hs = w.pos(origin), _horizons(horizon)
    Y = w.Y[:, :p + 1]
    m = Y[:, max(p - window + 1, 0):p + 1].mean(1)
    b0, b1 = p - window + 1 - season, p - season
    base = Y[:, b0:b1 + 1].mean(1) if b0 >= 0 else np.full(Y.shape[0], np.nan)
    codes = None
    if group_col is not None:
        if group_col not in panel.columns:
            raise KeyError(f"panel has no column {group_col!r}")
        g = panel.groupby(id_col, sort=False)[group_col].first().reindex(w.ids)
        codes = pd.factorize(g)[0]
    out = np.full((Y.shape[0], len(hs)), np.nan)
    for j, h in enumerate(hs):
        n = max(int(np.ceil(h / season)), 1)        # seasons back (as seasonal_naive): 1 unless horizon > season
        k = p + h - season * n
        lo, hi = k - around, min(k + around, p)
        bj0, bas = b0, base
        if n > 1:
            bj0 = p - window + 1 - season * n
            bas = Y[:, bj0:p - season * n + 1].mean(1) if bj0 >= 0 else base
        if lo < 0 or bj0 < 0 or hi < lo:
            continue
        ly = Y[:, lo:hi + 1].mean(1)
        num, den = ly + smooth, bas + smooth
        if codes is not None:
            num = np.bincount(codes, ly, codes.max() + 1)[codes] + smooth
            den = np.bincount(codes, bas, codes.max() + 1)[codes] + smooth
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
        if cap is not None:
            r = np.clip(r, cap[0], cap[1])
        out[:, j] = m * r
    return w.frame(origin, out, hs)


def ses_level(Y: np.ndarray, alpha: float) -> np.ndarray:
    """Simple exponential smoothing level per column (kod/ses.py ses_seviye): L[:, t] = level after time t."""
    Y = np.asarray(Y, float)
    L = np.zeros_like(Y)
    lev = Y[:, 0].copy()
    L[:, 0] = lev
    for j in range(1, Y.shape[1]):
        lev = alpha * Y[:, j] + (1 - alpha) * lev
        L[:, j] = lev
    return L


def ses(panel: pd.DataFrame, origin, horizon: int | Sequence[int] = 4, *, alpha: float | None = None,
        alphas: Sequence[float] = SES_ALPHAS, n_tune_origins: int = 6, tune_step: int | None = None,
        id_col: str = "id", time_col: str = "time", y_col: str = "y", fill_value: float = 0.0,
        return_alpha: bool = False):
    """Simple exponential smoothing, flat forecast (D4 ladder T6).

    ``alpha=None`` picks alpha from ``alphas`` by mean WMAPE over ``n_tune_origins`` earlier origins (step =
    max horizon) whose targets all lie <= ``origin`` - the rehearsal's rule, so the tuning never sees the
    forecast window. ``return_alpha=True`` returns ``(frame, alpha, scores)``."""
    w = _Wide(panel, id_col, time_col, y_col, fill_value)
    p, hs = w.pos(origin), _horizons(horizon)
    Y = w.Y[:, :p + 1]
    H = max(hs)
    scores: dict[float, float] = {}
    if alpha is None:
        step = tune_step or H
        last = p - H                                 # last tuning origin: its targets end at p
        tune = [t for t in range(last, 0, -step)][:n_tune_origins]
        if not tune:
            raise ValueError("not enough history to tune alpha; pass alpha=")
        for a in alphas:
            L = ses_level(Y, a)
            scores[a] = float(np.mean([wmape(Y[:, t + 1:t + H + 1], np.repeat(L[:, t:t + 1], H, 1)) for t in tune]))
        alpha = min(scores, key=scores.get)
    lev = ses_level(Y, alpha)[:, p]
    fr = w.frame(origin, np.repeat(lev[:, None], len(hs), 1), hs)
    return (fr, alpha, scores) if return_alpha else fr


# ----------------------------------------------------------------------------- features
def _feature_block(w: _Wide, p: int, hs: Sequence[int], lags, windows, season, exog, static) -> pd.DataFrame:
    Y = w.Y[:, :p + 1]                      # hard cut: nothing after the origin is visible
    n = Y.shape[0]
    base: dict[str, Any] = {}
    for k in lags:
        base[f"lag{k}"] = Y[:, p - k + 1] if p - k + 1 >= 0 else np.full(n, np.nan)
    for win in windows:
        a = max(p - win + 1, 0)
        base[f"mean{win}"] = Y[:, a:p + 1].mean(1)
        base[f"nz{win}"] = (Y[:, a:p + 1] > 0).sum(1).astype(float)
    if season:
        a, b = p - season - 1, p - season + 3       # same-season level around the origin one season ago
        base["season_base"] = Y[:, max(a, 0):b].mean(1) if b > 0 else np.full(n, np.nan)
    df = pd.DataFrame(base)
    if static:
        st = w.panel.groupby(w.id_col)[list(static)].first().reindex(w.ids)
        for c in static:
            df[c] = pd.Categorical(st[c].to_numpy()).codes.astype(np.int32)
    ex = {c: w.exog(c) for c in exog} if exog else None
    outs = []
    for h in hs:
        d = df.copy()
        d.insert(0, "h", h)
        if season:
            k = p + h - season * int(np.ceil(h / season))
            d["season_lag"] = Y[:, k] if k >= 0 else np.nan
            d["season_ratio"] = (d["season_lag"] + 1.0) / (d["season_base"] + 1.0)
        for c, piv in (ex or {}).items():
            d[c] = piv.iloc[:, p + h].to_numpy() if p + h < piv.shape[1] else np.nan
        d.insert(0, w.id_col, w.ids)
        outs.append(d)
    return pd.concat(outs, ignore_index=True)


def make_features(panel: pd.DataFrame, origin, *, id_col: str = "id", time_col: str = "time", y_col: str = "y",
                  lags: Sequence[int] = (1, 2, 3, 4), windows: Sequence[int] = (4, 8, 13, 26),
                  horizons: int | Sequence[int] = 1, season: int | None = None,
                  exog: Sequence[str] | None = None, static: Sequence[str] | None = None,
                  fill_value: float = 0.0) -> pd.DataFrame:
    """Feature rows at one origin: one row per (series, h), computed only from ``y`` at times <= origin.

    Columns: ``id_col``, ``h``, ``lag{k}`` (lag1 = value at the origin), ``mean{w}`` / ``nz{w}`` (rolling mean
    and non-zero count over the last w periods), with ``season``: ``season_lag`` (value one season before the
    target), ``season_base`` and ``season_ratio`` (kod/ozellik.py yoy / yb); ``exog`` columns are read at the
    *target* time (use only known-in-advance values such as planned price or calendar); ``static`` columns
    (constant per series) become integer category codes."""
    w = _Wide(panel, id_col, time_col, y_col, fill_value)
    return _feature_block(w, w.pos(origin), _horizons(horizons), lags, windows, season, exog, static)


# ----------------------------------------------------------------------------- direct forecaster
def _default_model(model, objective, y_nonneg: bool, seed: int):
    if not isinstance(model, str):
        from sklearn.base import clone
        return clone(model)
    if model == "lightgbm":
        try:
            import lightgbm as lgb
            obj = objective or ("tweedie" if y_nonneg else "regression")
            kw = dict(tweedie_variance_power=1.2) if obj == "tweedie" else {}
            # kod/kos.py defaults (num_leaves 127, lr 0.08), smaller min_data for small panels
            return lgb.LGBMRegressor(objective=obj, n_estimators=300, learning_rate=0.08, num_leaves=63,
                                     min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                     reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=1, **kw)
        except ImportError:
            warnings.warn("lightgbm not installed; DirectForecaster falls back to sklearn "
                          "HistGradientBoostingRegressor", RuntimeWarning, stacklevel=3)
            model = "hgb"
    if model in ("hgb", "sklearn"):
        from sklearn.ensemble import HistGradientBoostingRegressor
        loss = objective if objective in ("poisson", "squared_error", "absolute_error") else (
            "poisson" if y_nonneg else "squared_error")
        return HistGradientBoostingRegressor(loss=loss, max_iter=300, learning_rate=0.08, random_state=seed)
    raise ValueError(f"unknown model {model!r}: 'lightgbm', 'hgb' or an sklearn estimator")


class DirectForecaster:
    """Direct multi-horizon strategy: one global model per horizon h, trained on rows (series, origin T) with
    features at T and target y at T+h, pooled over ``n_train_origins`` origins whose targets are <= the fit
    origin (kod/kos.py). D4 rehearsal, one run, test WMAPE: sales-history-only Tweedie LightGBM (T5) 0.354,
    + price + calendar (B3) 0.348, vs. SES 0.371 and last-4 mean 0.379 (SONUC-MAIN.md).

    ``model``: "lightgbm" (lazy import, falls back to sklearn HistGradientBoostingRegressor with a warning),
    "hgb", or any sklearn regressor (cloned per horizon). ``objective``: LightGBM objective (default "tweedie"
    with variance power 1.2 for non-negative y) or HGB loss. ``clip_min``: lower clip for predictions
    (0.0 for demand)."""

    def __init__(self, model: Any = "lightgbm", horizons: int | Sequence[int] = 4, *,
                 lags: Sequence[int] = (1, 2, 3, 4), windows: Sequence[int] = (4, 8, 13, 26),
                 season: int | None = None, exog: Sequence[str] | None = None, static: Sequence[str] | None = None,
                 objective: str | None = None, n_train_origins: int = 52, origin_step: int = 1,
                 min_history: int | None = None, clip_min: float | None = None, id_col: str = "id",
                 time_col: str = "time", y_col: str = "y", fill_value: float = 0.0, seed: int = 0):
        self.model, self.horizons = model, _horizons(horizons)
        self.lags, self.windows, self.season = tuple(lags), tuple(windows), season
        self.exog, self.static, self.objective = list(exog or []), list(static or []), objective
        self.n_train_origins, self.origin_step = int(n_train_origins), int(origin_step)
        self.min_history = min_history
        self.clip_min, self.seed = clip_min, seed
        self.id_col, self.time_col, self.y_col, self.fill_value = id_col, time_col, y_col, fill_value
        self.models_: dict[int, Any] = {}

    def _wide(self, panel):
        return _Wide(panel, self.id_col, self.time_col, self.y_col, self.fill_value)

    def _X(self, d: pd.DataFrame) -> pd.DataFrame:
        return d.drop(columns=[self.id_col, "h"])

    def fit(self, panel: pd.DataFrame, origin) -> "DirectForecaster":
        w = self._wide(mask_future(panel, origin, time_col=self.time_col, y_col=self.y_col))
        p = w.pos(origin)
        need = self.min_history if self.min_history is not None else max(
            [*self.lags, *self.windows, (self.season or 0) + 4, 1])
        y_nonneg = bool(np.nanmin(w.raw[:, :p + 1]) >= 0)
        self.models_ = {}
        for h in self.horizons:
            Ts = list(range(p - h, need - 2, -self.origin_step))[:self.n_train_origins]
            if not Ts:
                raise ValueError(f"no training origin for h={h}: history too short")
            parts = []
            for T in Ts:
                d = _feature_block(w, T, [h], self.lags, self.windows, self.season, self.exog, self.static)
                d["_y"] = w.raw[:, T + h]
                parts.append(d)
            tr = pd.concat(parts, ignore_index=True)
            tr = tr[~tr["_y"].isna()]
            m = _default_model(self.model, self.objective, y_nonneg, self.seed)
            m.fit(self._X(tr.drop(columns="_y")), tr["_y"].to_numpy())
            self.models_[h] = m
        self.feature_names_ = list(self._X(tr.drop(columns="_y")).columns)
        self.origin_ = origin
        return self

    def predict(self, panel: pd.DataFrame, origin=None) -> pd.DataFrame:
        if not self.models_:
            raise RuntimeError("call fit() first")
        origin = self.origin_ if origin is None else origin
        w = self._wide(mask_future(panel, origin, time_col=self.time_col, y_col=self.y_col))
        p = w.pos(origin)
        preds = np.zeros((len(w.ids), len(self.horizons)))
        for j, h in enumerate(self.horizons):
            d = _feature_block(w, p, [h], self.lags, self.windows, self.season, self.exog, self.static)
            preds[:, j] = self.models_[h].predict(self._X(d))
        if self.clip_min is not None:
            preds = np.maximum(preds, self.clip_min)
        return w.frame(origin, preds, self.horizons)

    def __call__(self, panel, origin, horizon=None, **_):
        """Refit at ``origin`` and predict (so a DirectForecaster can be passed to ``backtest``)."""
        return self.fit(panel, origin).predict(panel, origin)


# ----------------------------------------------------------------------------- backtest
def backtest(model_or_fn: Any, panel: pd.DataFrame, *, origins: Iterable, horizon: int | Sequence[int] = 4,
             id_col: str = "id", time_col: str = "time", y_col: str = "y", dropna: bool = True,
             fill_value: float = 0.0, **kwargs) -> pd.DataFrame:
    """Rolling-origin evaluation -> long frame ``[id_col, origin, h, time_col, truth, pred]``.

    ``model_or_fn`` is an object with ``fit(panel, origin)`` / ``predict(panel, origin)`` (refit at every
    origin), or a callable ``fn(panel, origin, horizon, **kwargs)`` returning a forecast frame (every
    baseline in this module). At each origin the callee receives the panel with ``y`` masked to NaN after
    the origin, so it cannot leak the future; truth comes from the unmasked panel.

    Use several origins: one origin is misleading. In the D4 rehearsal (one run, `kod/ek.py`) the rule
    baselines' WMAPE swung between about 0.38 and 0.45 across origins 229-269, and the dev origin 269 was
    one of the easiest.

    ``fill_value``: truth of an (id, time) cell missing inside the panel (default 0.0, the same "missing cell =
    0" rule the forecasters use, so sparse panels without zero-sales rows score like dense ones). Targets beyond
    the last panel time are always NaN (dropped with ``dropna``). ``fill_value=np.nan`` drops missing cells.
    A forecaster with its own ``horizons`` (DirectForecaster) must cover the requested ``horizon``: missing
    horizons raise a warning and are absent from the result."""
    hs = _horizons(horizon)
    own = getattr(model_or_fn, "horizons", None)
    if own is not None and not set(hs) <= set(own):
        warnings.warn(f"backtest: horizon {hs} is not covered by the forecaster's horizons {list(own)}; "
                      f"missing h={sorted(set(hs) - set(own))} are not evaluated", UserWarning, stacklevel=2)
    truth_w = _Wide(panel, id_col, time_col, y_col, fill_value=fill_value)
    cols = dict(id_col=id_col, time_col=time_col, y_col=y_col)
    out = []
    for o in origins:
        masked = mask_future(panel, o, time_col=time_col, y_col=y_col)
        if hasattr(model_or_fn, "fit") and hasattr(model_or_fn, "predict") and not isinstance(model_or_fn, type):
            fr = model_or_fn.fit(masked, o).predict(masked, o)
        else:
            sig = inspect.signature(model_or_fn).parameters
            kw = {k: v for k, v in {**cols, **kwargs}.items()
                  if k in sig or any(p.kind == p.VAR_KEYWORD for p in sig.values())}
            fr = model_or_fn(masked, o, hs, **kw)
        fr = fr[fr["h"].isin(hs)].copy()
        p = truth_w.pos(o)
        idx = fr[id_col].map({k: i for i, k in enumerate(truth_w.ids)}).to_numpy()
        tp = p + fr["h"].to_numpy()
        ok = tp < truth_w.raw.shape[1]
        tr = np.full(len(fr), np.nan)
        tr[ok] = truth_w.Y[idx[ok], tp[ok]]
        fr["origin"] = o
        fr["truth"] = tr
        out.append(fr[[id_col, "origin", "h", time_col, "truth", "pred"]])
    res = pd.concat(out, ignore_index=True)
    return res.dropna(subset=["truth"]).reset_index(drop=True) if dropna else res


def summarize_backtest(bt: pd.DataFrame, by: Sequence[str] | str | None = "origin") -> pd.DataFrame:
    """WMAPE, bias and row count per group of a backtest frame (``by=None``: one overall row)."""
    def agg(g):
        return pd.Series({"wmape": wmape(g["truth"], g["pred"]), "bias": bias(g["truth"], g["pred"]),
                          "n": int(len(g))})
    if by is None:
        return agg(bt).to_frame().T
    by = [by] if isinstance(by, str) else list(by)
    return bt.groupby(by)[["truth", "pred"]].apply(agg).reset_index()


# ----------------------------------------------------------------------------- metric-optimal point
def point_for_metric(mean_pred, *, metric: str = "mae", dist: str | None = "poisson",
                     threshold: float | None = None) -> np.ndarray:
    """Turn a mean forecast into the point that is optimal for ``metric``.

    L1 metrics (mae, wmape, wape) are minimised by the median, MSE by the mean. With ``dist="poisson"`` the
    median of Poisson(mean) is returned (kod/pm.py); ``threshold`` applies it only where mean < threshold
    (rehearsal B5 = B3 + Poisson median for y_hat < 3; large means keep the mean). The median shifts totals
    down, so report bias (``bias()``) separately. D4 rehearsal (one run): dev WMAPE -0.0020 / -0.0025 on two
    origins, store x dept aggregate WMAPE +0.0013 / +0.0022 (worse), from SIMENGINE-EKSIK.md #9."""
    x = np.asarray(mean_pred, float)
    m = metric.lower()
    if m in L2_METRICS or dist is None:
        return x.copy()
    if m not in L1_METRICS:
        raise ValueError(f"unknown metric {metric!r}: one of {sorted(L1_METRICS | L2_METRICS)}")
    if dist != "poisson":
        raise ValueError("only dist='poisson' (or None) is implemented")
    from scipy.stats import poisson
    med = poisson.median(np.maximum(np.nan_to_num(x, nan=0.0), 1e-9))
    med = np.where(np.isnan(x), np.nan, med)
    return np.where(x < threshold, med, x) if threshold is not None else med


# ----------------------------------------------------------------------------- LLM prior feature
PRIOR_SYSTEM = "You are a careful analyst. Reply with JSON only."
PRIOR_SUFFIX = '\nReturn {"value": <number>} only.'


def _blind_label(i: int) -> str:
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return f"group {s}"


def _parse_reply(text: str, parse: Callable[[Any], Any]):
    try:
        obj = parse_json(text)
        if isinstance(obj, dict):
            obj = obj.get("value", next(iter(obj.values())) if len(obj) == 1 else None)
        return parse(obj)
    except (ValueError, TypeError, KeyError, StopIteration):
        try:
            return parse(text.strip())
        except (ValueError, TypeError):
            m = re.search(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", text)     # first number in a plain-text reply
            try:
                return parse(m.group(0)) if m else np.nan
            except (ValueError, TypeError):
                return np.nan


def llm_prior_feature(llm, keys: Sequence[Any], question: str | Callable[[Any], str], *,
                      blind_question: str | Callable[[Any], str] | None = None, shuffle_placebo: bool = True,
                      parse: Callable[[Any], Any] = float, system: str = PRIOR_SYSTEM, max_tokens: int = 200,
                      thinking: Any = None, tag: str | None = None, seed: int = 0,
                      key_name: str = "key") -> pd.DataFrame:
    """Ask an LLM one number per key and return it as a feature table with its controls.

    ``question``: template with ``{key}`` or ``callable(key) -> prompt``; a ``{"value": <number>}`` JSON
    instruction is appended. ``blind_question``: same, but a str template gets an anonymous label
    ("group A", ...) for ``{key}``, a callable gets the key and must hide the name itself (e.g. keep the event,
    hide the category). Output columns: ``key_name``, ``value`` (named arm), ``value_blind`` (if asked) and
    ``value_shuffled`` (placebo: the named values permuted across keys, no extra calls). A feature that helps
    only as ``value`` and not as ``value_shuffled`` / ``value_blind`` carries information from the name.
    Calls go through ``llm.map`` (one batch). Port of D4 ``kod/llm_olay.py`` (named vs blind arms, category x
    event multipliers, 6 calls)."""
    keys = list(keys)

    def prompts(q, blind: bool):
        out = []
        for i, k in enumerate(keys):
            if callable(q):
                u = q(k)
            else:
                u = q.format(key=_blind_label(i) if blind else k)
            out.append((system, u + PRIOR_SUFFIX))
        return out

    kw: dict[str, Any] = {}
    if thinking is not None:
        kw["thinking"] = thinking
    batch = prompts(question, False)
    if blind_question is not None:
        batch += prompts(blind_question, True)
    if tag is not None:
        kw["tag"] = tag
    resps = llm.map(batch, max_tokens=max_tokens, **kw)
    if len(resps) != len(batch):
        raise RuntimeError(f"llm.map returned {len(resps)} answers for {len(batch)} prompts")
    vals = [_parse_reply(getattr(r, "text", r), parse) for r in resps]
    n_bad = sum(1 for v in vals if isinstance(v, float) and np.isnan(v))
    if n_bad:
        warnings.warn(f"llm_prior_feature: {n_bad}/{len(vals)} replies could not be parsed (NaN)", RuntimeWarning,
                      stacklevel=2)
    out = pd.DataFrame({key_name: keys, "value": vals[:len(keys)]})
    if blind_question is not None:
        out["value_blind"] = vals[len(keys):]
    if shuffle_placebo:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(keys))
        if len(keys) > 1:
            while np.any(perm == np.arange(len(keys))):     # derangement: no key keeps its own value
                perm = rng.permutation(len(keys))
        out["value_shuffled"] = out["value"].to_numpy()[perm]
    return out


__all__ = ["persistence_forecast", "seasonal_naive", "moving_mean", "seasonal_ratio", "ses", "ses_level", "make_features",
           "DirectForecaster", "backtest", "summarize_backtest", "point_for_metric", "llm_prior_feature",
           "mask_future", "wmape", "bias", "SES_ALPHAS"]
