"""Tabular regression baselines and group-level (cell) baselines, all usable directly in ``oof_predict``.

Task shape: a table of rows (people, studies, ads) with categorical and numeric columns and a continuous target;
the honest no-LLM rung is "the best plain regression on the same columns" or "the level of the row's cell"
(outcome x year mean). Every class here has ``fit(X, y, sample_weight=None)`` / ``predict(X)`` on a DataFrame, so
``oof_predict(model, df, features=..., target=..., weight="w")`` passes per-row weights straight through.

    from simengine.methods.tabular import GroupStat, RandomEffectsMean, TabularGBDT, TabularRidge
    from simengine.methods import oof_predict
    oof_r = oof_predict(TabularRidge(cat=["outcome", "year"], num=["n", "pre"]), df,
                        features=["outcome", "year", "n", "pre"], target="ate", group="study", weight="w")
    oof_g = oof_predict(TabularGBDT(cat=["outcome", "year"]), df, features=["outcome", "year", "n", "pre"],
                        target="ate", group="study")
    oof_re = oof_predict(RandomEffectsMean(by=["outcome", "year"], se_col="se"), df,
                         features=["outcome", "year", "se"], target="ate", group="study")
    oof_md = oof_predict(GroupStat(by=["outcome", "year"], stat="median"), df,
                         features=["outcome", "year"], target="ate", group="study")

- ``TabularRidge``: one-hot categoricals + standardised numerics (training mean fill) + ``RidgeCV`` (G2 rehearsal
  hand-wrote it with ``ColumnTransformer``; the scaler here is weighted, which ``make_pipeline`` cannot do).
- ``TabularGBDT``: LightGBM when importable (native categoricals), else sklearn ``HistGradientBoostingRegressor``
  (``backend_`` says which).
- ``RandomEffectsMean``: per cell, the DerSimonian-Laird random-effects mean ``sum(y/(se^2+tau^2)) /
  sum(1/(se^2+tau^2))``; cells with too few rows fall back to coarser cells (prefixes of ``by``), then the grand
  random-effects mean (D5 rehearsal: the standard meta-analysis baseline "outcome x year precision-weighted mean").
- ``GroupStat``: per-cell median / mean / weighted mean with the same fallback (D5 hand-wrote it as ``Duzey``).
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from ._util import lightgbm_available, n_threads

RIDGE_ALPHAS = tuple(np.logspace(-2, 4, 25))
STATS = ("median", "mean", "wmean")


def _weights(sample_weight, n: int) -> np.ndarray | None:
    if sample_weight is None:
        return None
    w = np.asarray(sample_weight, float).ravel()
    if len(w) != n:
        raise ValueError(f"sample_weight has {len(w)} values, X has {n} rows")
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("sample_weight must be finite, non-negative and not all zero")
    return w


def _split_columns(X: pd.DataFrame, cat, num) -> tuple[list[str], list[str]]:
    """cat / num lists; ``None`` = every column of X not listed in the other one, by dtype (numeric -> num,
    anything else -> cat); ``()`` = none. Columns of X in neither list are not used."""
    cat_l = None if cat is None else list(cat)
    num_l = None if num is None else list(num)
    taken = set(cat_l or []) | set(num_l or [])
    rest = [c for c in X.columns if c not in taken]
    if num_l is None:
        num_l = [c for c in rest if pd.api.types.is_numeric_dtype(X[c])]
    if cat_l is None:
        cat_l = [c for c in rest if c not in num_l]
    missing = [c for c in cat_l + num_l if c not in X.columns]
    if missing:
        raise KeyError(f"columns not in X: {missing}")
    both = sorted(set(cat_l) & set(num_l))
    if both:
        raise ValueError(f"columns listed in both cat and num: {both}")
    if not cat_l and not num_l:
        raise ValueError("no feature columns: give cat= / num= or pass X with columns")
    return cat_l, num_l


def _cat_str(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), "__nan__").astype(str)


# ------------------------------------------------------------------------------------------------ regressions

class TabularRidge:
    """One-hot categoricals + standardised numerics + ``RidgeCV`` (continuous target).

    - ``cat`` / ``num``: column lists (``None`` = the remaining columns of ``X`` by dtype, ``()`` = none). Numeric
      codes such as a year go in ``cat`` to get one dummy per value.
    - Numerics: NaN filled with the training (weighted) mean, then centred and scaled by the training (weighted)
      mean / std. Categoricals: one column per training category (NaN is its own category); ``min_count`` drops
      rarer categories; unseen categories at predict time get all-zero dummies (the intercept).
    - ``alphas``: ridge grid, chosen by ``RidgeCV``'s efficient leave-one-out (weighted when ``sample_weight``).

    Attributes after fit: ``alpha_``, ``coef_`` (Series by encoded column name), ``intercept_``.
    """

    def __init__(self, cat: Sequence[str] | None = None, num: Sequence[str] | None = None, *,
                 alphas: Sequence[float] = RIDGE_ALPHAS, min_count: int = 1):
        self.cat, self.num, self.alphas, self.min_count = cat, num, alphas, min_count

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.num_:
            v = X[self.num_].apply(pd.to_numeric, errors="coerce").astype(float)
            v = v.fillna(self.mean_)
            parts.append(((v - self.mean_) / self.scale_).to_numpy(float))
        for c in self.cat_:
            s = _cat_str(X[c]).to_numpy()
            parts.append((s[:, None] == np.asarray(self.categories_[c], dtype=object)[None, :]).astype(float))
        return np.hstack(parts) if parts else np.zeros((len(X), 0))

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        from sklearn.linear_model import RidgeCV
        X = pd.DataFrame(X)
        y = np.asarray(y, float)
        w = _weights(sample_weight, len(X))
        self.cat_, self.num_ = _split_columns(X, self.cat, self.num)
        ww = np.ones(len(X)) if w is None else w
        if self.num_:
            v = X[self.num_].apply(pd.to_numeric, errors="coerce").astype(float)
            ok = v.notna().to_numpy()
            vv = np.nan_to_num(v.to_numpy(float))
            sw = (ok * ww[:, None]).sum(0)
            mean = np.where(sw > 0, (vv * ok * ww[:, None]).sum(0) / np.where(sw > 0, sw, 1), 0.0)
            var = (ok * ww[:, None] * (vv - mean) ** 2).sum(0) / np.where(sw > 0, sw, 1)
            sd = np.sqrt(var)
            self.mean_ = pd.Series(mean, index=self.num_)
            self.scale_ = pd.Series(np.where(sd > 1e-12, sd, 1.0), index=self.num_)
        self.categories_ = {}
        for c in self.cat_:
            vc = _cat_str(X[c]).value_counts()
            self.categories_[c] = sorted(vc.index[vc >= self.min_count])
        Z = self._encode(X)
        self.feature_names_ = list(self.num_) + [f"{c}={k}" for c in self.cat_ for k in self.categories_[c]]
        self.model_ = RidgeCV(alphas=list(self.alphas)).fit(Z, y, sample_weight=w)
        self.alpha_ = float(self.model_.alpha_)
        self.coef_ = pd.Series(self.model_.coef_, index=self.feature_names_)
        self.intercept_ = float(self.model_.intercept_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._encode(pd.DataFrame(X)))


class TabularGBDT:
    """Gradient-boosted trees on categoricals + numerics (continuous target).

    ``backend``: ``"auto"`` (LightGBM when importable, else sklearn ``HistGradientBoostingRegressor`` with a
    ``RuntimeWarning``), ``"lightgbm"`` or ``"sklearn"``. Categoricals use native categorical splits in both
    (training categories; unseen / NaN -> missing). Defaults are for small tables (hundreds to a few thousand rows):
    300 trees, lr 0.05, 15 leaves, min 20 rows per leaf; ``params`` override them (LightGBM names for LightGBM,
    sklearn names for the fallback). ``backend_`` after fit says which ran.
    """

    def __init__(self, cat: Sequence[str] | None = None, num: Sequence[str] | None = None, *,
                 backend: str = "auto", seed: int = 0, **params):
        self.cat, self.num, self.backend, self.seed, self.params = cat, num, backend, seed, params

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        out = pd.DataFrame(index=X.index)
        for c in self.num_:
            out[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        for c in self.cat_:
            s = X[c].astype(object).where(X[c].notna(), None)
            s = s.map(lambda v: v if v is None else str(v))
            cat = pd.Categorical(s, categories=self.categories_[c])
            out[c] = cat if self.backend_ == "lightgbm" else np.where(cat.codes >= 0, cat.codes, np.nan)
        return out[self.num_ + self.cat_]

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        X = pd.DataFrame(X)
        y = np.asarray(y, float)
        w = _weights(sample_weight, len(X))
        self.cat_, self.num_ = _split_columns(X, self.cat, self.num)
        self.categories_ = {c: sorted(X[c].dropna().astype(str).unique()) for c in self.cat_}
        b = self.backend
        if b not in ("auto", "lightgbm", "sklearn"):
            raise ValueError(f"backend must be 'auto', 'lightgbm' or 'sklearn', got {b!r}")
        if b == "auto":
            b = "lightgbm" if lightgbm_available() else "sklearn"
            if b == "sklearn":
                warnings.warn("lightgbm not installed; TabularGBDT falls back to sklearn "
                              "HistGradientBoostingRegressor", RuntimeWarning, stacklevel=2)
        self.backend_ = b
        F = self._frame(X)
        if b == "lightgbm":
            import lightgbm as lgb
            prm = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.8, num_threads=n_threads(len(X)),
                       random_state=self.seed, verbose=-1)
            prm.update(self.params)
            self.model_ = lgb.LGBMRegressor(**prm).fit(F, y, sample_weight=w)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            mask = [c in self.cat_ for c in F.columns]
            prm = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                       random_state=self.seed)
            prm.update(self.params)
            self.model_ = HistGradientBoostingRegressor(categorical_features=mask if any(mask) else None, **prm)
            self.model_.fit(F.to_numpy(float), y, sample_weight=w)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        F = self._frame(X)
        return np.asarray(self.model_.predict(F if self.backend_ == "lightgbm" else F.to_numpy(float)), float)


# ------------------------------------------------------------------------------------------------ cell baselines

def dersimonian_laird(y, se) -> tuple[float, float, float]:
    """DerSimonian-Laird random-effects pooling -> ``(mu, tau2, se_mu)``.

    ``w = 1/se^2``, ``Q = sum w (y - ybar_w)^2``, ``C = sum w - sum w^2 / sum w``, ``tau2 = max(0, (Q - (k-1)) / C)``,
    ``mu = sum(y / (se^2 + tau2)) / sum(1 / (se^2 + tau2))``, ``se_mu = sum(1 / (se^2 + tau2))^-1/2``. One study ->
    ``tau2 = 0``."""
    y, v = np.asarray(y, float), np.asarray(se, float) ** 2
    if len(y) == 0:
        return float("nan"), float("nan"), float("nan")
    w = 1.0 / v
    tau2 = 0.0
    if len(y) > 1:
        ybar = (w * y).sum() / w.sum()
        q = (w * (y - ybar) ** 2).sum()
        c = w.sum() - (w ** 2).sum() / w.sum()
        tau2 = max(0.0, (q - (len(y) - 1)) / c) if c > 0 else 0.0
    ws = 1.0 / (v + tau2)
    return float((ws * y).sum() / ws.sum()), float(tau2), float(ws.sum() ** -0.5)


def _wmedian(y: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(y)
    y, cw = y[o], np.cumsum(w[o])
    half = cw[-1] / 2.0
    i = int(np.searchsorted(cw, half))
    if np.isclose(cw[i], half) and i + 1 < len(y):      # exactly half the weight below: midpoint, as np.median
        return float((y[i] + y[i + 1]) / 2.0)
    return float(y[i])


class _CellBaseline:
    """Shared cell lookup: one table per prefix of ``by`` (finest first), grand value last."""

    by: list
    min_n: int

    def _keys(self, X: pd.DataFrame, cols: list[str]) -> list:
        if len(cols) == 1:
            return list(_cat_str(X[cols[0]]))
        return list(zip(*[_cat_str(X[c]) for c in cols]))

    def _fit_cells(self, X: pd.DataFrame, agg) -> None:
        missing = [c for c in self.by if c not in X.columns]
        if missing:
            raise KeyError(f"by columns not in X: {missing}")
        self.tables_ = []
        for k in range(len(self.by), 0, -1):
            cols = self.by[:k]
            cells: dict = {}
            for i, key in enumerate(self._keys(X, cols)):
                cells.setdefault(key, []).append(i)
            rows = {key: agg(np.asarray(idx)) for key, idx in cells.items() if len(idx) >= self.min_n}
            tab = pd.DataFrame(list(rows.values()), index=pd.Index(list(rows), dtype=object, tupleize_cols=False))
            self.tables_.append((cols, tab))
        self.grand_ = agg(np.arange(len(X)))

    def _lookup(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = pd.DataFrame(X)
        out = np.full(len(X), np.nan)
        lev = np.zeros(len(X), dtype=int)
        for cols, tab in self.tables_:
            if tab.empty:
                continue
            val = tab["value"].to_dict()
            for i, key in enumerate(self._keys(X, cols)):
                if lev[i] == 0 and key in val:
                    out[i], lev[i] = val[key], len(cols)
        out[lev == 0] = self.grand_["value"]
        return out, lev

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._lookup(X)[0]

    def level(self, X: pd.DataFrame) -> np.ndarray:
        """Number of ``by`` columns of the cell each row's prediction came from (0 = grand value)."""
        return self._lookup(X)[1]


class RandomEffectsMean(_CellBaseline):
    """Per-cell DerSimonian-Laird random-effects mean of ``y`` with standard errors from ``X[se_col]``.

    - ``by``: cell columns, finest last; a cell (by[:k]) with fewer than ``min_n`` rows is skipped and its rows use
      the next coarser cell (by[:k-1]), then the grand random-effects mean over all training rows. Unseen cells at
      predict time take the same path.
    - ``tau2``: ``"dl"`` (estimated per cell, DerSimonian-Laird) or a fixed float used in every cell.
    - Rows with a NaN standard error are not used (``RuntimeWarning`` with the count); ``se <= 0`` raises.
    - ``sample_weight`` is not accepted: the weights are ``1/(se^2 + tau2)`` by construction, so
      ``oof_predict(..., weight=)`` raises a clear error for this model.

    Attributes after fit: ``tables_`` (list of ``(cols, DataFrame[value, tau2, se, n])``, finest first), ``grand_``.
    ``level(X)`` tells which cell depth each prediction came from.
    """

    def __init__(self, by: Sequence[str], se_col: str = "se", *, tau2="dl", min_n: int = 2):
        self.by, self.se_col, self.tau2, self.min_n = list(by), se_col, tau2, min_n

    def fit(self, X: pd.DataFrame, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, float)
        if self.se_col not in X.columns:
            raise KeyError(f"se_col {self.se_col!r} not in X (add it to features=)")
        se = pd.to_numeric(X[self.se_col], errors="coerce").to_numpy(float)
        if (se <= 0).any():
            raise ValueError(f"{self.se_col!r} has values <= 0; standard errors must be positive")
        ok = np.isfinite(se) & np.isfinite(y)
        if not ok.all():
            warnings.warn(f"RandomEffectsMean: {int((~ok).sum())} rows with a missing standard error / target are "
                          "not used", RuntimeWarning, stacklevel=2)
        X, y, se = X[ok].reset_index(drop=True), y[ok], se[ok]
        if not len(y):
            raise ValueError("RandomEffectsMean: no rows with a standard error")
        fixed = None if self.tau2 == "dl" else float(self.tau2)

        def agg(idx):
            if fixed is None:
                mu, t2, s = dersimonian_laird(y[idx], se[idx])
            else:
                ws = 1.0 / (se[idx] ** 2 + fixed)
                mu, t2, s = float((ws * y[idx]).sum() / ws.sum()), fixed, float(ws.sum() ** -0.5)
            return {"value": mu, "tau2": t2, "se": s, "n": len(idx)}

        self._fit_cells(X, agg)
        return self


class GroupStat(_CellBaseline):
    """Per-cell median / mean / weighted mean of ``y`` (the D5 ``Duzey`` level baseline).

    - ``stat``: ``"median"``, ``"mean"`` or ``"wmean"``. Weights come from ``X[weight]`` when ``weight`` is given,
      else from ``sample_weight`` (what ``oof_predict(..., weight=)`` passes). When weights are present, ``mean`` and
      ``wmean`` are the weighted mean and ``median`` the weighted median; ``wmean`` without weights raises.
    - ``by``, ``min_n``: as ``RandomEffectsMean`` (fallback to coarser cells, then the grand value).
    """

    def __init__(self, by: Sequence[str], stat: str = "median", *, weight: str | None = None, min_n: int = 1):
        if stat not in STATS:
            raise ValueError(f"stat must be one of {STATS}, got {stat!r}")
        self.by, self.stat, self.weight, self.min_n = list(by), stat, weight, min_n

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, float)
        if self.weight is not None:
            if self.weight not in X.columns:
                raise KeyError(f"weight column {self.weight!r} not in X (add it to features=)")
            w = _weights(pd.to_numeric(X[self.weight], errors="coerce").to_numpy(float), len(X))
        else:
            w = _weights(sample_weight, len(X))
        if self.stat == "wmean" and w is None:
            raise ValueError("GroupStat(stat='wmean') needs weight= (a column of X) or sample_weight")
        ok = np.isfinite(y)
        X, y = X[ok].reset_index(drop=True), y[ok]
        w = None if w is None else w[ok]

        def agg(idx):
            v = y[idx]
            if w is None:
                val = float(np.median(v) if self.stat == "median" else np.mean(v))
            elif self.stat == "median":
                val = _wmedian(v, w[idx]) if w[idx].sum() > 0 else float(np.median(v))
            else:
                val = float(np.average(v, weights=w[idx])) if w[idx].sum() > 0 else float(np.mean(v))
            return {"value": val, "n": len(idx)}

        self._fit_cells(X, agg)
        return self
