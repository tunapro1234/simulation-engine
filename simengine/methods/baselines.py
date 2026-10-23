"""Non-LLM baselines: the ladder every simulation result is reported against.

Ported from the TALP prep research kits (intent kit `yontemler.py`, persona kit `Medyan`, deney kit `KomsuDeney`,
kanit `merdiven.py`, makro `temel_c1.py`, intent/seq `klasik.py`). Algorithms and defaults are kept.

Uniform API: classes have ``fit(...)`` returning ``self`` and ``predict(...)``; binary/multiclass classes also have
``predict_proba``. Distribution and basket baselines are pure functions.

kind: "binary" (P(y=1) out), "multiclass" (probability matrix, columns = ``classes_``), "continuous" (number out).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ._util import n_threads, require_lightgbm

KINDS = ("binary", "multiclass", "continuous")


def infer_kind(y) -> str:
    """binary if values are a subset of {0, 1}; multiclass if non-numeric; continuous otherwise."""
    s = pd.Series(np.asarray(y, dtype=object)).dropna()
    if not len(s):
        return "continuous"
    num = pd.to_numeric(s, errors="coerce")
    if num.isna().any() or s.map(lambda v: isinstance(v, (str, bool, np.bool_))).any():
        return "binary" if set(s.astype(str)) <= {"0", "1", "True", "False"} else "multiclass"
    return "binary" if set(np.unique(num.to_numpy(float))) <= {0.0, 1.0} else "continuous"


def _check_kind(kind, y):
    kind = infer_kind(y) if kind in (None, "auto") else kind
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return kind


def _n(X) -> int:
    return int(X) if np.isscalar(X) else len(X)


def _ndim(a) -> int:
    if a is None:
        return 0
    if isinstance(a, pd.DataFrame):
        return 2
    if isinstance(a, (pd.Series, pd.Index)):
        return 1
    try:
        return int(np.ndim(a))
    except ValueError:                       # ragged list
        return 1


def _ncols(a) -> int:
    return int(np.shape(a)[1]) if _ndim(a) == 2 else 1


def _y_first(y, X):
    """Argument order of ``fit(y, X=None)``. ``methods.transfer.oof_predict`` and sklearn-style callers call
    ``fit(X, y)``; that is detected when the first argument is 2-D (DataFrame / matrix) and the second 1-D, or the
    first has more than one column, and turned around. Both 1-D (equal length) is ambiguous: the old order (y first)
    is kept. A one-column 2-D y (column vector) is flattened."""
    if _ndim(y) == 2 and (_ndim(X) == 1 or _ncols(y) > 1):
        if X is None:
            raise ValueError(f"fit got a {np.shape(y)} table and no y: call fit(y) or fit(X, y)")
        y, X = X, y
    if _ndim(y) == 2 and _ncols(y) == 1:
        y = np.asarray(y, dtype=object)[:, 0]
    return y, X


# ---------------------------------------------------------------- knows nothing

class ConstantBaseline:
    """Same prediction for everyone (intent kit `_sabit`): binary = training rate, multiclass = training class shares,
    continuous = training median (``stat="mean"`` for the mean). ``predict`` accepts a row count or any sized input.

    ``fit(y)`` / ``fit(y, X)``; the sklearn order ``fit(X, y)`` (what ``oof_predict`` calls) is detected by shape
    (first argument 2-D and second 1-D, or first with several columns) and handled. Two 1-D arguments of equal
    length are read as ``(y, X)``; pass ``y=`` by keyword when in doubt."""

    def __init__(self, kind: str = "auto", stat: str = "median"):
        self.kind, self.stat = kind, stat

    def fit(self, y, X=None):
        y, X = _y_first(y, X)
        y = pd.Series(np.asarray(y, dtype=object)).dropna()
        self.kind_ = _check_kind(self.kind, y)
        if self.kind_ == "binary":
            self.value_ = float(pd.to_numeric(y).mean())
        elif self.kind_ == "multiclass":
            sh = y.astype(str).value_counts(normalize=True).sort_index()
            self.classes_, self.value_ = list(sh.index), sh.to_numpy(float)
        else:
            v = pd.to_numeric(y).to_numpy(float)
            self.value_ = float(np.median(v) if self.stat == "median" else np.mean(v))
        return self

    def predict_proba(self, X):
        n = _n(X)
        if self.kind_ == "multiclass":
            return np.tile(self.value_, (n, 1))
        if self.kind_ == "binary":
            return np.full(n, self.value_)
        raise ValueError("predict_proba is for binary/multiclass")

    def predict(self, X):
        n = _n(X)
        if self.kind_ == "multiclass":
            return np.tile(self.value_, (n, 1))
        return np.full(n, self.value_)


class MajorityBaseline:
    """Majority class label (``predict``) and its one-hot (``predict_proba``). Report it next to every accuracy:
    in the Twin-2K-500 persona task the column median alone scores acc 0.755 (uretim/persona/RAPOR.md, T1 table).
    ``fit(y)``, ``fit(y, X)`` or ``fit(X, y)``: argument order as in :class:`ConstantBaseline`."""

    def fit(self, y, X=None):
        y, X = _y_first(y, X)
        s = pd.Series(np.asarray(y, dtype=object)).dropna()
        vc = s.value_counts()
        self.label_ = vc.index[0]
        self.classes_ = sorted(vc.index, key=str)
        return self

    def predict(self, X):
        return np.array([self.label_] * _n(X), dtype=object)

    def predict_proba(self, X):
        P = np.zeros((_n(X), len(self.classes_)))
        P[:, self.classes_.index(self.label_)] = 1.0
        return P


class ItemMarginal:
    """Per-item ("everyone answers this question the same") baseline (kanit `merdiven.py` `marjinal`):
    item mean / median / class shares learned on training rows; unseen items fall back to the global value."""

    def __init__(self, item_col: str = "item", y_col: str = "truth", kind: str = "auto", stat: str = "mean"):
        self.item_col, self.y_col, self.kind, self.stat = item_col, y_col, kind, stat

    def fit(self, df: pd.DataFrame):
        y = df[self.y_col]
        self.kind_ = _check_kind(self.kind, y)
        if self.kind_ == "multiclass":
            self.classes_ = sorted(y.dropna().astype(str).unique())
            ct = pd.crosstab(df[self.item_col], y.astype(str)).reindex(columns=self.classes_, fill_value=0)
            self.table_ = ct.div(ct.sum(1), axis=0)
            self.global_ = y.astype(str).value_counts(normalize=True).reindex(self.classes_).fillna(0).to_numpy()
        else:
            agg = "median" if self.stat == "median" else "mean"
            v = pd.to_numeric(y)
            self.table_ = v.groupby(df[self.item_col]).agg(agg)
            self.global_ = float(getattr(np, agg)(v.dropna().to_numpy(float)))
        return self

    def predict(self, df: pd.DataFrame):
        if self.kind_ == "multiclass":
            T = self.table_.reindex(df[self.item_col].to_numpy())
            M = np.array(T.to_numpy(float))     # writable copy (copy-on-write hands back read-only views)
            bad = np.isnan(M).any(1)
            M[bad] = self.global_
            return M
        return df[self.item_col].map(self.table_).fillna(self.global_).to_numpy(float)

    predict_proba = predict


class SegmentRate:
    """Segment mean shrunk toward the global mean, ``(sum + a*g) / (n + a)`` (intent kit `segment_oran`; kanit uses
    a fixed a=10). With ``valid=(segments, y)`` in ``fit``, ``a`` is picked from (1, 5, 20, 50, 200) by validation
    log-loss (binary/multiclass) or MAE (continuous)."""

    GRID = (1, 5, 20, 50, 200)

    def __init__(self, a: float | None = None, kind: str = "auto"):
        self.a, self.kind = a, kind

    def _pred(self, seg_tr, y_tr, seg, a):
        if self.kind_ == "multiclass":
            prior = y_tr.value_counts(normalize=True).reindex(self.classes_).fillna(0)
            cnt = pd.crosstab(np.asarray(seg_tr), np.asarray(y_tr)).reindex(columns=self.classes_, fill_value=0)
            sh = (cnt + a * prior.to_numpy()).div(cnt.sum(1) + a, axis=0)
            out = np.array(sh.reindex(pd.Series(seg).to_numpy()).to_numpy(float))       # writable copy
            bad = np.isnan(out).any(1)
            out[bad] = prior.to_numpy()
            return out
        g = float(y_tr.mean())
        t = pd.DataFrame({"s": np.asarray(seg_tr), "y": y_tr.to_numpy(float)}).groupby("s")["y"].agg(["sum", "count"])
        p = (t["sum"] + a * g) / (t["count"] + a)
        return pd.Series(np.asarray(seg)).map(p).fillna(g).to_numpy(float)

    def _loss(self, y, p):
        if self.kind_ == "binary":
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if self.kind_ == "multiclass":
            idx = np.array([self.classes_.index(v) if v in self.classes_ else -1 for v in y])
            pp = np.where(idx >= 0, p[np.arange(len(y)), np.maximum(idx, 0)], 1e-6)
            return float(-np.mean(np.log(np.clip(pp, 1e-6, 1))))
        return float(np.mean(np.abs(y - p)))

    def fit(self, segments, y, valid: tuple | None = None):
        y = pd.Series(np.asarray(y, dtype=object))
        self.kind_ = _check_kind(self.kind, y)
        if self.kind_ == "multiclass":
            y = y.astype(str)
            self.classes_ = sorted(y.unique())
        else:
            y = pd.to_numeric(y)
        self.seg_, self.y_ = np.asarray(segments), y.reset_index(drop=True)
        if self.a is not None:
            self.a_ = float(self.a)
        elif valid is not None:
            sv, yv = np.asarray(valid[0]), np.asarray(valid[1], dtype=object)
            yv = yv.astype(str) if self.kind_ == "multiclass" else yv.astype(float)
            self.a_ = float(min(self.GRID, key=lambda a: self._loss(yv, self._pred(self.seg_, self.y_, sv, a))))
        else:
            self.a_ = 10.0
        return self

    def predict(self, segments):
        return self._pred(self.seg_, self.y_, np.asarray(segments), self.a_)

    predict_proba = predict


class ConditionalMode:
    """Most frequent label per key cell (person x slot x day-type, ...) with neighbour smoothing, time decay and a
    backoff chain, fit on a long table by sparse groupby counts (no dense user x period x item tensor). The
    multiclass counterpart of ``SegmentRate``; generalises ``abm.mobility.slot_mode``.

        m = ConditionalMode(["person", "slot", "daytype"], target="place",
                            smooth={"slot": [(-1, 0.5), (1, 0.5)]}, wrap={"slot": 48},
                            backoff=[["person", "daytype"], ["person"], ["slot", "daytype"]],
                            decay=0.95, time_col="day").fit(history)
        labels = m.predict(query)                       # (n,) labels
        top3, level = m.predict(query, k=3, return_level=True)

    smooth: {numeric column: [(offset, weight), ...]}. A cell at value v also counts the rows at v + offset with
            ``weight`` (its own rows weigh 1). Several smoothed columns multiply. Applied on every level that holds
            the column. ``wrap={col: period}`` makes offsets circular (48 half-hour slots, 7 weekdays; keys are
            taken mod period in fit and predict, so 1-based slots 1..48 work).
    backoff: key lists tried in order after ``keys`` when a query cell has no weight (> ``min_weight``); the
            global mode always ends the chain. ``levels_`` lists the chain, ``return_level`` the index used per row.
    decay:  row weight ``decay ** (ref - time)`` with ``ref`` the latest ``time_col`` value in fit (``now=``
            overrides); datetimes count in days. ``weight_col`` multiplies in.
    Ties go to the globally more frequent label, then the smaller label (as str). ``k`` returns an (n, k) object
    array: the answering level's ranking, then the next levels' labels not yet listed, None-padded.
    """

    def __init__(self, keys: Sequence[str], target: str = "label", smooth: dict | None = None,
                 backoff: Sequence[Sequence[str]] | None = None, decay: float | None = None,
                 time_col: str | None = None, wrap: dict | None = None, weight_col: str | None = None,
                 min_weight: float = 0.0):
        self.keys, self.target = [keys] if isinstance(keys, str) else list(keys), target
        self.smooth, self.wrap = dict(smooth or {}), dict(wrap or {})
        self.backoff = [[b] if isinstance(b, str) else list(b) for b in (backoff or [])]
        self.decay, self.time_col, self.weight_col, self.min_weight = decay, time_col, weight_col, min_weight

    def _weights(self, df, now):
        w = np.ones(len(df)) if self.weight_col is None else df[self.weight_col].to_numpy(float)
        if self.decay is None:
            return w
        if self.time_col is None:
            raise ValueError("decay needs time_col")
        if not 0 < self.decay <= 1:
            raise ValueError(f"decay must be in (0, 1], got {self.decay}")
        t = df[self.time_col]
        ref = t.max() if now is None else now
        age = ref - t
        age = age.dt.total_seconds() / 86400 if pd.api.types.is_timedelta64_dtype(age) else age
        return w * self.decay ** np.asarray(age, float)

    def _wrap(self, d: pd.DataFrame) -> pd.DataFrame:
        """Smoothed / wrapped key columns as int64 (uint8 slots would overflow on -1) or float, wrapped columns taken
        mod their period (so 1-based slots 1..48 meet 0 after a shift and queries at 48 find the same cell)."""
        cols = [c for c in d.columns if c in self.smooth or c in self.wrap]
        if not cols:
            return d
        d = d.copy()
        for c in cols:
            v = d[c].to_numpy()
            v = v.astype(np.int64) if np.issubdtype(v.dtype, np.integer) else v.astype(float)
            d[c] = v % self.wrap[c] if c in self.wrap else v
        return d

    def _counts(self, base: pd.DataFrame, level: list) -> pd.DataFrame:
        """Weighted label counts per cell of ``level`` (smoothed), sorted best first within each cell."""
        y = self.target
        c = self._wrap(base.groupby(level + [y], sort=False, observed=True)["_w"].sum().reset_index())
        for col in [x for x in level if x in self.smooth]:
            parts = [c]
            for off, wt in self.smooth[col]:
                s = c.copy()
                s[col] = s[col] - off
                if col in self.wrap:
                    s[col] = s[col] % self.wrap[col]
                s["_w"] = s["_w"] * float(wt)
                parts.append(s)
            c = pd.concat(parts, ignore_index=True).groupby(level + [y], sort=False, observed=True)["_w"].sum()
            c = c.reset_index()
        c = c[c["_w"] > 0]
        tot = c.groupby(level, sort=False, observed=True)["_w"].transform("sum")
        c = c[tot.to_numpy() > self.min_weight]
        c = c.assign(_r=c[y].map(self.rank_).to_numpy())
        return c.sort_values(level + ["_w", "_r"], ascending=[True] * len(level) + [False, True],
                             kind="mergesort").reset_index(drop=True)

    def fit(self, df: pd.DataFrame, now=None):
        y = self.target
        levels = [tuple(self.keys)] + [tuple(b) for b in self.backoff if tuple(b) != tuple(self.keys)]
        self.levels_ = list(dict.fromkeys(levels + [()]))
        cols = sorted({c for lv in self.levels_ for c in lv} | {y}, key=str)
        missing = [c for c in cols + [self.time_col, self.weight_col] if c is not None and c not in df.columns]
        if missing:
            raise KeyError(f"ConditionalMode.fit: missing column(s) {missing}")
        base = df[df[y].notna()]
        base = base[cols].assign(_w=self._weights(base, now))
        g = base.groupby(y, sort=False)["_w"].sum()
        g = g[g > 0]
        order = sorted(g.index, key=lambda v: (-g[v], str(v)))
        if not order:
            raise ValueError("ConditionalMode.fit: no rows with a label and positive weight")
        self.rank_ = {v: i for i, v in enumerate(order)}
        self.global_ = order
        self.tables_ = [self._counts(base, list(lv)) if lv else None for lv in self.levels_]
        return self

    def _lookup(self, q: pd.DataFrame, i: int, k: int):
        """Per query row: list of the top-k labels at level i (None where the cell is unknown)."""
        lv = list(self.levels_[i])
        if not lv:
            return [self.global_[:k]] * len(q)
        t = self.tables_[i]
        top = t.groupby(lv, sort=False, observed=True).head(k)
        lists = top.groupby(lv, sort=False, observed=True)[self.target].agg(list).rename("_l").reset_index()
        m = q[lv].merge(lists, on=lv, how="left")
        return m["_l"].tolist()

    def predict(self, query: pd.DataFrame, k: int | None = None, return_level: bool = False):
        q = query.reset_index(drop=True)
        missing = [c for lv in self.levels_ for c in lv if c not in q.columns]
        if missing:
            raise KeyError(f"ConditionalMode.predict: missing column(s) {sorted(set(missing))}")
        q = self._wrap(q)
        n, kk = len(q), 1 if k is None else int(k)
        level = np.full(n, -1)
        if k is None:                                  # top-1: vectorised merge per level
            out = np.empty(n, dtype=object)
            for i, lv in enumerate(self.levels_):
                todo = np.flatnonzero(level < 0)
                if not len(todo):
                    break
                if not lv:
                    out[todo], level[todo] = self.global_[0], i
                    break
                best = self.tables_[i].drop_duplicates(list(lv))
                lab = best[self.target].to_numpy(dtype=object)        # labels by row number, dtype kept
                pos = q.iloc[todo][list(lv)].merge(best[list(lv)].assign(_i=np.arange(len(best))), on=list(lv),
                                                   how="left")["_i"].to_numpy()
                hit = ~np.isnan(pos)
                out[todo[hit]], level[todo[hit]] = lab[pos[hit].astype(int)], i
            res = pd.Series(out, dtype=object).infer_objects().to_numpy()   # int labels stay int
            return (res, level) if return_level else res
        found = [[] for _ in range(n)]
        for i in range(len(self.levels_)):
            todo = np.flatnonzero(np.fromiter((len(f) < kk for f in found), bool, n))
            if not len(todo):
                break
            for j, lab in zip(todo, self._lookup(q.iloc[todo], i, kk)):
                if not isinstance(lab, list):
                    continue
                if level[j] < 0:
                    level[j] = i
                f = found[j]
                f.extend(v for v in lab if v not in f)
                del f[kk:]
        top = np.full((n, kk), None, dtype=object)
        for j, f in enumerate(found):
            top[j, :len(f)] = f
        return (top, level) if return_level else top


# ---------------------------------------------------------------- MAE ladder rungs (median constant, cell median)

_L1_STATS = {"mae", "wmape", "wape", "l1", "mad", "smae", "zmae", "medae", "nmae"}
_L2_STATS = {"mse", "rmse", "l2", "zmse", "zrmse", "r2"}


def stat_for_metric(metric: str) -> str:
    """The constant that minimises ``metric``: "median" for L1 errors (MAE, WAPE, zMAE), "mean" for L2 (MSE,
    RMSE), as in ``forecast.point_for_metric``. For a pinball / quantile loss pass the level as ``stat=q``."""
    m = str(metric).lower()
    if m in _L1_STATS:
        return "median"
    if m in _L2_STATS:
        return "mean"
    raise ValueError(f"no constant stat for metric {metric!r}: one of {sorted(_L1_STATS | _L2_STATS)}, "
                     "or pass stat='median' / 'mean' / a quantile level")


def _wstat(v: np.ndarray, w: np.ndarray | None, stat) -> float:
    """mean / median / quantile q of v; weighted quantiles are the weighted step quantile (midpoint on an exact
    tie of the cumulative weight, so equal weights give np.median)."""
    if not len(v):
        raise ValueError("no rows with a numeric target (and positive weight) to take the stat of")
    if stat == "mean":
        return float(np.average(v, weights=w))
    q = 0.5 if stat == "median" else float(stat)
    if w is None:
        return float(np.quantile(v, q))
    o = np.argsort(v, kind="mergesort")
    v, cw = v[o], np.cumsum(w[o])
    h = q * cw[-1]
    i = int(np.searchsorted(cw, h))
    if i < len(v) - 1 and np.isclose(cw[i], h, rtol=1e-9, atol=0):      # relative: tiny weights stay exact
        return float((v[i] + v[i + 1]) / 2)
    return float(v[min(i, len(v) - 1)])


def _resolve_stat(stat, metric):
    s = stat_for_metric(metric) if stat is None else stat
    if s not in ("mean", "median") and not (isinstance(s, (int, float)) and 0 <= float(s) <= 1):
        raise ValueError(f"stat must be 'mean', 'median' or a quantile in [0, 1], got {s!r}")
    return s


def _loss(y, p, stat) -> float:
    e = np.asarray(y, float) - np.asarray(p, float)
    if stat == "mean":
        return float(np.mean(e ** 2))
    q = 0.5 if stat == "median" else float(stat)
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


class MedianConstant:
    """The constant rung of an MAE ladder: training median (default, ``metric="mae"``), mean (``metric="mse"``)
    or quantile ``stat=q``, optionally weighted. ``ConstantBaseline`` already has stat="median"; this one picks
    the stat from the metric and takes weights. D12 pilot: the median constant beat the mean by +0.0075 zMAE.

    ``fit(y, X=None, weight=None)``; ``fit(X, y)`` (``oof_predict``, sklearn) is detected as in
    :class:`ConstantBaseline`, and ``sample_weight=`` is an alias of ``weight`` (so ``oof_predict(weight=...)``
    works)."""

    def __init__(self, stat=None, metric: str = "mae"):
        self.stat, self.metric = stat, metric

    def fit(self, y, X=None, weight=None, sample_weight=None):
        if sample_weight is not None:
            if weight is not None:
                raise ValueError("pass weight= or sample_weight=, not both")
            weight = sample_weight
        y, X = _y_first(y, X)
        self.stat_ = _resolve_stat(self.stat, self.metric)
        v = pd.to_numeric(pd.Series(np.asarray(y, dtype=object)), errors="coerce").to_numpy(float)
        w = None if weight is None else np.asarray(weight, float)
        ok = ~np.isnan(v) if w is None else ~np.isnan(v) & (w > 0)
        self.value_ = _wstat(v[ok], None if w is None else w[ok], self.stat_)
        return self

    def predict(self, X):
        return np.full(_n(X), self.value_)


class CellMedian:
    """Cell median (mean / quantile) shrunk toward the global one: ``(n * cell + shrink * global) / (n + shrink)``,
    n = rows in the cell (Kish effective n with ``weight``). shrink=0 is the raw cell stat, large shrink the
    constant. Unseen cells get the global stat. The stat follows ``metric`` (median for MAE) unless ``stat`` is set.
    ``shrink=None`` picks it from ``GRID`` by the metric's loss on ``fit(..., valid=df)`` (pinball for a quantile).

        CellMedian(by=["store", "dow"], y_col="sales", shrink=None).fit(train, valid=dev).predict(test)
    """

    GRID = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500)

    def __init__(self, by, y_col: str = "y", shrink: float | None = 0.0, weight: str | None = None, stat=None,
                 metric: str = "mae"):
        self.by = [by] if isinstance(by, str) else list(by)
        self.y_col, self.shrink, self.weight, self.stat, self.metric = y_col, shrink, weight, stat, metric

    def _cells(self, df):
        v = pd.to_numeric(df[self.y_col], errors="coerce")
        w = None if self.weight is None else df[self.weight].astype(float)
        ok = v.notna() if w is None else v.notna() & (w > 0)
        d = df.loc[ok, self.by].assign(_v=v[ok].to_numpy(float))
        if w is None:
            g = d.groupby(self.by, observed=True)["_v"]
            stat = g.mean() if self.stat_ == "mean" else g.quantile(0.5 if self.stat_ == "median" else self.stat_)
            return stat.rename("_s"), g.size().astype(float).rename("_n"), \
                _wstat(d["_v"].to_numpy(), None, self.stat_)
        d["_w"] = w[ok].to_numpy(float)
        rows = {k: (_wstat(x["_v"].to_numpy(), x["_w"].to_numpy(), self.stat_),
                    x["_w"].sum() ** 2 / (x["_w"] ** 2).sum())
                for k, x in d.groupby(self.by, observed=True)}
        idx = pd.MultiIndex.from_tuples([k if isinstance(k, tuple) else (k,) for k in rows], names=self.by)
        s = pd.Series([r[0] for r in rows.values()], index=idx, name="_s")
        n = pd.Series([r[1] for r in rows.values()], index=idx, name="_n")
        return s, n, _wstat(d["_v"].to_numpy(), d["_w"].to_numpy(), self.stat_)

    def fit(self, df: pd.DataFrame, valid: pd.DataFrame | None = None):
        self.stat_ = _resolve_stat(self.stat, self.metric)
        self.cell_, self.n_, self.global_ = self._cells(df)
        if self.shrink is not None:
            self.shrink_ = float(self.shrink)
            return self
        if valid is None:
            raise ValueError("shrink=None picks shrink on validation data: pass fit(df, valid=...)")
        yv = pd.to_numeric(valid[self.y_col], errors="coerce").to_numpy(float)
        ok = ~np.isnan(yv)
        if not ok.any():
            raise ValueError(f"valid has no numeric {self.y_col!r}: cannot pick shrink")
        self.scores_ = {}
        for a in self.GRID:
            self.shrink_ = float(a)
            self.scores_[a] = _loss(yv[ok], self.predict(valid)[ok], self.stat_)
        self.shrink_ = float(min(self.GRID, key=lambda a: (round(self.scores_[a], 12), -a)))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        t = pd.concat([self.cell_, self.n_], axis=1).reset_index()
        m = df[self.by].reset_index(drop=True).merge(t, on=self.by, how="left")
        s, n = m["_s"].to_numpy(float), m["_n"].to_numpy(float)
        a = self.shrink_
        p = s if a == 0 else (n * s + a * self.global_) / (n + a)
        return np.where(np.isnan(p), self.global_, p)


# ---------------------------------------------------------------- distributions (group / segment level)

def uniform_distribution(k: int) -> np.ndarray:
    return np.full(int(k), 1.0 / int(k))


def weighted_distribution(values, categories: Sequence, weights=None) -> np.ndarray:
    """Weighted answer shares over ``categories`` (missing/out-of-range answers ignored; uniform if nothing left)."""
    v = pd.Series(np.asarray(values, dtype=object))
    w = pd.Series(np.ones(len(v)) if weights is None else np.asarray(weights, float))
    ok = v.isin(list(categories)).to_numpy()
    if not ok.any():
        return uniform_distribution(len(categories))
    s = w[ok].groupby(v[ok].to_numpy()).sum().reindex(list(categories)).fillna(0.0).to_numpy(float)
    return s / s.sum()


def overall_copy(overall, groups: Iterable) -> dict:
    """Copy the known population (overall) distribution to every group. Hard to beat for subgroup opinion
    distributions: OpinionQA groups, dev WD 0.0530 vs uniform 0.1976 (uretim/dongu/GUNLUK.md, oqa_grup_B round 0)."""
    p = np.asarray(overall, float)
    return {g: p.copy() for g in groups}


def _segment_keys(df: pd.DataFrame, segment_col) -> pd.Series:
    """One key per row: the column value, or a tuple for several columns (cross segments)."""
    if isinstance(segment_col, str):
        return df[segment_col]
    cols = list(segment_col)
    return pd.Series(list(zip(*(df[c] for c in cols))), index=df.index, dtype=object)


def _is(keys: pd.Series, s) -> np.ndarray:
    """keys == s as a bool array; element-wise for tuple keys (pandas would treat a tuple as list-like)."""
    if isinstance(s, tuple):
        return np.fromiter((k == s for k in keys), bool, len(keys))
    return (keys == s).to_numpy()


def _effective_n(ok: np.ndarray, w) -> float:
    """Rows with a valid answer; with weights, Kish's effective size (sum w)^2 / sum w^2."""
    if w is None:
        return float(ok.sum())
    ww = np.asarray(w, float)[ok]
    return float(ww.sum() ** 2 / (ww ** 2).sum()) if len(ww) and (ww ** 2).sum() > 0 else 0.0


def segment_distributions(df: pd.DataFrame, value_col: str, segment_col, categories: Sequence,
                          weight_col: str | None = None, mode: str = "overall",
                          pairs: dict | None = None, m: float = 20.0, prior=None,
                          segments: Iterable | None = None) -> dict:
    """Distribution baselines per segment (makro `temel_c1.py`; "shrunk" from the G5 rehearsal ladder, T2b).

    mode: "overall"  every segment gets the weighted overall marginal (segment ignored; population copy, T1),
          "others"   a segment gets the marginal of all OTHER segments (no data from the target segment),
          "pair"     a segment gets the marginal of ``pairs[segment]`` (e.g. the same party's other age group),
          "own"      the segment's own marginal (segment empirical, T2; oracle reference on held-in data),
          "shrunk"   own marginal shrunk toward the overall: p = (n_s * own + m * overall) / (n_s + m), where
                     n_s is the number of valid answers in the segment (Kish effective size with ``weight_col``)
                     and ``m`` a pseudo-count. m=0 is "own", m -> inf is "overall".

    ``segment_col`` may be a list of columns: segments are then cross cells and the keys are tuples. ``prior``
    replaces the overall marginal as the shrinkage target ("shrunk" only): one distribution (e.g. a known
    population), or a dict {segment: distribution}, which gives hierarchical shrinkage, e.g. cross cells toward
    their one-column "shrunk" result. ``segments`` lists the keys to return (default: the ones in ``df``); a key
    with no rows gets the target itself under "shrunk" and the uniform distribution under "own"/"pair".

    "shrunk" is the LLM-free baseline to beat for segment distribution tasks: in the G5 rehearsal (segment x item
    answer distributions, 1-TVD) shrunk segment empirical (m=20) led on seed 0 (0.912 against population copy
    0.906; on seed 1 the population copy led), and the LLM hybrid did not beat it across 5 seeds. Report it next
    to the population copy ("overall"), not instead of it.
    Pick ``m`` on training cells with ``select_m``.
    """
    w = df[weight_col] if weight_col else None
    keys = _segment_keys(df, segment_col)
    seg_list = list(pd.unique(keys)) if segments is None else list(segments)
    if mode == "shrunk":
        if m < 0:
            raise ValueError("m must be >= 0")
        overall = weighted_distribution(df[value_col], categories, w)
        valid = df[value_col].isin(list(categories)).to_numpy()
    out = {}
    for s in seg_list:
        if mode == "overall":
            mask = np.ones(len(df), bool)
        elif mode == "others":
            mask = ~_is(keys, s)
        elif mode == "pair":
            mask = _is(keys, pairs[s])
        elif mode in ("own", "shrunk"):
            mask = _is(keys, s)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        own = weighted_distribution(df.loc[mask, value_col], categories, None if w is None else w[mask])
        if mode != "shrunk":
            out[s] = own
            continue
        target = overall if prior is None else np.asarray(prior[s] if isinstance(prior, dict) else prior, float)
        n = _effective_n(valid & mask, w)
        out[s] = target.copy() if n == 0 else (n * own + m * target) / (n + m)
    return out


def segment_empirical(df: pd.DataFrame, value_col: str, segment_col, categories: Sequence,
                      shrink: float = 0.0, **kw) -> dict:
    """Segment empirical distribution (T2); ``shrink=m`` > 0 shrinks it toward the overall marginal (T2b).
    Alias of ``segment_distributions(mode="own" / "shrunk", m=shrink)``."""
    mode = "shrunk" if shrink > 0 else "own"
    return segment_distributions(df, value_col, segment_col, categories, mode=mode, m=shrink, **kw)


M_GRID = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500)


@dataclass
class ShrinkSelection:
    """``select_m`` result: chosen pseudo-count, mean held-out score per grid value (the criterion, higher is
    better), number of scored cells summed over folds, and mean held-out 1-TVD per grid value (for display)."""
    m: float
    scores: dict
    n_cells: int
    tvd: dict | None = None


def select_m(df: pd.DataFrame, value_col, segment_col, categories, grid: Sequence[float] = M_GRID,
             folds: int | None = 5, unit_col: str | None = None, weight_col: str | None = None,
             min_eval: int = 1, seed: int = 0, score: str = "l2") -> ShrinkSelection:
    """Pick ``m`` for ``mode="shrunk"`` by K-fold cross-validation on training data only.

    Units (rows, or ``unit_col`` groups such as households, so one unit never sits on both sides) are split into
    ``folds`` folds; ``folds=None`` is leave-one-unit-out. Each fold is held out once: shrunk distributions are fit
    on the rest (overall marginal from the rest too) and compared with the held-out segment's own empirical
    marginal. Cells (segment x item) with fewer than ``min_eval`` held-out answers are skipped; the score is the
    mean over scored cells and folds. ``value_col`` may be a list of item columns; ``categories`` is then one
    shared list or a dict {column: categories}. Ties go to the smaller ``m``.

    score: "l2" (default) 1 - squared L2 distance to the held-out empirical. The held-out empirical is a noisy
           copy of the truth and its noise adds the same constant to every candidate, so this ranks m as the
           truth would, whatever the fold size (also leave-one-unit-out).
           "tvd"  1 - TVD to the held-out empirical, the task metric itself. Not a proper score: on small
           held-out cells it rewards sharp predictions, so it drifts toward m = 0; with one answer per cell
           (leave-one-row-out) it always returns an end of the grid. Use only with large held-out cells.
    ``tvd`` on the result always reports the mean held-out 1-TVD, the number a reader expects next to the task.
    """
    if score not in ("l2", "tvd"):
        raise ValueError(f"unknown score {score!r}; use 'l2' or 'tvd'")
    items = [value_col] if isinstance(value_col, str) else list(value_col)
    cats = {c: list(categories[c] if isinstance(categories, dict) else categories) for c in items}
    units = df[unit_col].to_numpy() if unit_col else np.arange(len(df))
    uniq = pd.unique(units)
    k = len(uniq) if folds is None else int(folds)
    if k < 2 or k > len(uniq):
        raise ValueError(f"folds must be in [2, n_units={len(uniq)}], got {k}")
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % k))
    f = np.array([fold_of[u] for u in units])
    keys = _segment_keys(df, segment_col)
    w = df[weight_col] if weight_col else None
    tot = {g: [] for g in grid}
    tvd = {g: [] for g in grid}
    n_cells = 0
    for j in range(k):
        tr, ho = df[f != j], df[f == j]
        k_ho = keys[f == j]
        for c in items:
            valid = ho[c].isin(cats[c]).to_numpy()
            segs = [s for s in pd.unique(k_ho) if (valid & _is(k_ho, s)).sum() >= min_eval]
            if not segs:
                continue
            truth = segment_distributions(ho, c, segment_col, cats[c], weight_col, mode="own", segments=segs)
            n_cells += len(segs)
            for g in grid:
                pred = segment_distributions(tr, c, segment_col, cats[c], weight_col, mode="shrunk", m=g,
                                             segments=segs)
                t = [1.0 - 0.5 * np.abs(pred[s] - truth[s]).sum() for s in segs]
                tvd[g] += t
                tot[g] += t if score == "tvd" else [1.0 - float(((pred[s] - truth[s]) ** 2).sum()) for s in segs]
    if not n_cells:
        raise ValueError("no held-out cell with enough answers; lower min_eval or folds")
    scores = {g: float(np.mean(v)) for g, v in tot.items()}
    best = max(grid, key=lambda g: (round(scores[g], 12), -g))
    return ShrinkSelection(m=float(best), scores=scores, n_cells=n_cells,
                           tvd={g: float(np.mean(v)) for g, v in tvd.items()})


# ---------------------------------------------------------------- persistence (person's own past behaviour)

class PersistenceBaseline:
    """"Tomorrow = yesterday": each unit's own value in the last observed period(s) (intent kit `kalicilik`).

    ``fit(history)`` takes a long table (unit, time, value) observed before the cutoff. The prediction for a unit is
    ``agg`` (mean/sum/last) of its values over the last ``last_n_periods`` distinct time values, times ``scale``
    (e.g. a yearly count / 12 for a monthly rate). Units without rows in that window get ``fill``
    ("zero" for count data, "median" of the predicted units, or a number).

    Measured: order-frequency class, macro-F1 0.781 [0.756, 0.805] vs majority 0.275; the best GBDT with all
    features is not separable from it (+0.001 [-0.022, 0.023]) (uretim/intent/RAPOR.md, G0 table).
    """

    def __init__(self, unit_col: str = "unit", time_col: str = "time", value_col: str = "value",
                 last_n_periods: int = 1, agg: str = "mean", scale: float = 1.0, fill="median"):
        self.unit_col, self.time_col, self.value_col = unit_col, time_col, value_col
        self.last_n_periods, self.agg, self.scale, self.fill = last_n_periods, agg, scale, fill

    def fit(self, history: pd.DataFrame):
        h = history[[self.unit_col, self.time_col, self.value_col]].dropna(subset=[self.value_col])
        times = np.sort(pd.unique(h[self.time_col]))
        win = set(times[-self.last_n_periods:]) if len(times) else set()
        w = h[h[self.time_col].isin(win)].sort_values(self.time_col)
        g = w.groupby(self.unit_col)[self.value_col]
        v = g.last() if self.agg == "last" else g.agg(self.agg)
        self.values_ = v.astype(float) * self.scale
        self.units_ = set(pd.unique(h[self.unit_col]))
        return self

    def predict(self, units) -> np.ndarray:
        u = pd.Series(np.asarray(units, dtype=object))
        p = np.array(u.map(self.values_).to_numpy(float))          # writable copy (copy-on-write)
        miss = np.isnan(p)
        if miss.any():
            if self.fill == "zero":
                f = 0.0
            elif self.fill == "median":
                f = float(np.nanmedian(self.values_.to_numpy())) if len(self.values_) else 0.0
            else:
                f = float(self.fill)
            p[miss] = f
        return p


def persistence(history: pd.DataFrame, units, **kw) -> np.ndarray:
    """Function form of :class:`PersistenceBaseline`."""
    return PersistenceBaseline(**kw).fit(history).predict(units)


# ---------------------------------------------------------------- next basket / repeat purchase (U, T, K tensors)

def basket_tensor(df: pd.DataFrame, user_col: str, period_col: str, item_col: str, count_col: str | None = None,
                  n_periods: int | None = None):
    """Long purchase table -> (X, users, items): X[u, t, k] = count of item k bought by user u in integer period t."""
    users = sorted(pd.unique(df[user_col]), key=str)
    items = sorted(pd.unique(df[item_col]), key=str)
    T = int(n_periods if n_periods is not None else df[period_col].max() + 1)
    ui = pd.Index(users).get_indexer(df[user_col])
    ki = pd.Index(items).get_indexer(df[item_col])
    ti = df[period_col].to_numpy(int)
    X = np.zeros((len(users), T, len(items)), np.float32)
    np.add.at(X, (ui, ti, ki), 1.0 if count_col is None else df[count_col].to_numpy(np.float32))
    return X, users, items


def popularity(X: np.ndarray, cutoff: int, H: int) -> np.ndarray:
    """Share of users who bought each item in the H periods before ``cutoff`` (K,)."""
    return (X[:, max(0, cutoff - H):cutoff] > 0).any(1).mean(0)


def personal_frequency(X: np.ndarray, cutoff: int, H: int, beta: float = 1e-3) -> np.ndarray:
    """P-TopFreq: number of periods with a purchase of the item, plus beta * popularity as tie-break (U, K)."""
    B = (X[:, :cutoff] > 0).sum(1).astype(np.float32)
    return B + beta * popularity(X, cutoff, H)[None]


def decayed_frequency(X: np.ndarray, cutoff: int, H: int, r: float = 0.9, beta: float = 0.05) -> np.ndarray:
    """Time-decayed personal frequency s_k = sum_t r^(c-1-t) 1[X_tk>0] / sum_t r^(c-1-t) + beta * popularity.
    Beats P-TopFreq on test MAP@10: OE +0.023 [0.019, 0.027], CJ +0.015 [0.011, 0.019] (uretim/intent/RAPOR.md)."""
    w = r ** np.arange(cutoff - 1, -1, -1, dtype=np.float32)
    S = np.einsum("utk,t->uk", (X[:, :cutoff] > 0).astype(np.float32), w)
    S /= w.sum()
    return S + beta * popularity(X, cutoff, H)[None]


def tifu_knn(X: np.ndarray, cutoff: int, H: int, r_b: float = 0.9, r_g: float = 0.7, m: int = 6, k: int = 300,
             alpha: float = 0.7, beta: float = 0.01) -> np.ndarray:
    """TIFU-KNN (Hu et al., SIGIR 2020): decayed user vector, alpha*u + (1-alpha)*mean of k nearest users.
    Measured: no gain over decayed frequency (test -0.001 [-0.003, 0.002] OE; uretim/intent/RAPOR.md)."""
    Bx = (X[:, :cutoff] > 0).astype(np.float32)
    full = Bx.sum(2) > 0
    U, _, K = Bx.shape
    u_vec = np.zeros((U, K), np.float32)
    for u in range(U):
        b = Bx[u, full[u]]
        n = len(b)
        if n == 0:
            continue
        G = int(np.ceil(n / m))
        edges = [max(0, n - (G - j) * m) for j in range(G)] + [n]
        v = np.zeros(K, np.float32)
        for j in range(G):
            g = b[edges[j]:edges[j + 1]]
            L = len(g)
            wb = r_b ** np.arange(L - 1, -1, -1, dtype=np.float32)
            v += (r_g ** (G - 1 - j)) * (wb @ g) / L
        u_vec[u] = v / G
    nrm = np.linalg.norm(u_vec, axis=1, keepdims=True)
    un = u_vec / np.maximum(nrm, 1e-9)
    sim = un @ un.T
    np.fill_diagonal(sim, -1)
    sim[nrm[:, 0] == 0] = -1
    sim[:, nrm[:, 0] == 0] = -1
    k = max(1, min(k, U - 1))
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k] if U > 1 else np.zeros((U, 1), int)
    neigh = u_vec[idx].mean(1)
    return alpha * u_vec + (1 - alpha) * neigh + beta * popularity(X, cutoff, H)[None]


# ---------------------------------------------------------------- supervised tabular baselines

def _split_columns(X: pd.DataFrame, categorical: Sequence[str] | None):
    if categorical is None:
        categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) or
                       isinstance(X[c].dtype, pd.CategoricalDtype)]
    numeric = [c for c in X.columns if c not in categorical]
    return list(categorical), numeric


def _preprocessor(cat, num):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    return ColumnTransformer([
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                                            max_categories=60)), cat),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)])


def _proba_out(model, P, kind):
    cl = list(model.classes_)
    if kind == "binary":
        return P[:, cl.index(1)] if 1 in cl else np.zeros(len(P))
    return P


class LogisticBaseline:
    """One-hot categoricals + standardised numerics -> logistic regression (C=1) or ridge (alpha=1) for continuous
    targets (intent kit `lineer`). Train rows above ``max_rows`` are subsampled (seed 0).

    ``fit(X, y, sample_weight=None)``: per-row weights go to the final estimator (the scaler stays unweighted) and
    are subsampled with the rows, so ``oof_predict(weight=...)`` works."""

    def __init__(self, kind: str = "auto", categorical: Sequence[str] | None = None, C: float = 1.0,
                 alpha: float = 1.0, max_rows: int = 300_000):
        self.kind, self.categorical, self.C, self.alpha, self.max_rows = kind, categorical, C, alpha, max_rows

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        X = pd.DataFrame(X)
        y = np.asarray(y)
        w = _sample_weight(sample_weight, len(X))
        self.kind_ = _check_kind(self.kind, y)
        if len(X) > self.max_rows:
            i = np.random.default_rng(0).choice(len(X), self.max_rows, replace=False)
            X, y = X.iloc[i], y[i]
            w = None if w is None else w[i]
        self.cat_, self.num_ = _split_columns(X, self.categorical)
        est = Ridge(alpha=self.alpha) if self.kind_ == "continuous" else LogisticRegression(max_iter=2000, C=self.C)
        if self.kind_ == "binary":
            y = y.astype(int)
        elif self.kind_ == "multiclass":
            y = y.astype(str)
        pipe = make_pipeline(_preprocessor(self.cat_, self.num_), est)
        kw = {} if w is None else {f"{pipe.steps[-1][0]}__sample_weight": w}
        self.model_ = pipe.fit(X[self.cat_ + self.num_], y, **kw)
        if self.kind_ == "multiclass":
            self.classes_ = list(self.model_.classes_)
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X)[self.cat_ + self.num_]
        return _proba_out(self.model_, self.model_.predict_proba(X), self.kind_)

    def predict(self, X):
        if self.kind_ == "continuous":
            return self.model_.predict(pd.DataFrame(X)[self.cat_ + self.num_])
        return self.predict_proba(X)


def _sample_weight(w, n: int):
    if w is None:
        return None
    w = np.array(w, dtype=float).reshape(-1)
    if len(w) != n:
        raise ValueError(f"sample_weight has {len(w)} values for {n} rows")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("sample_weight must be finite and >= 0")
    return w


def _eval_kw(Model, Xv, yv) -> dict:
    """Validation-set keywords for LightGBM's sklearn fit: eval_X/eval_y on versions that have them (4.7+
    deprecates eval_set), else eval_set."""
    import inspect
    if "eval_X" in inspect.signature(Model.fit).parameters:
        return {"eval_X": (Xv,), "eval_y": (yv,)}
    return {"eval_set": [(Xv, yv)]}


RARE_SHARE = 0.02           # LightGBMBaseline: minority class share below this with default min_child_samples warns
_MIN_CHILD_KEYS = ("min_child_samples", "min_data_in_leaf", "min_data_per_leaf", "min_data")
# LightGBMBaseline(refit_full=True): params the refit drops (a tree-count alias would override n_estimators_, an
# early-stopping one needs a validation set the refit no longer has)
_REFIT_DROP_KEYS = ("num_iterations", "num_iteration", "n_iter", "num_tree", "num_trees", "num_round", "num_rounds",
                    "nrounds", "num_boost_round", "max_iter", "early_stopping_round", "early_stopping_rounds",
                    "early_stopping", "n_iter_no_change")


REWEIGHT_ODDS = 1.5        # LightGBMBaseline: sample_weight moving the positive odds by more than this is a reweighting


def _reweighting(params: dict, kind: str = "binary", y=None, w=None) -> str | None:
    """The class-reweighting parameter that is set (LightGBM names and aliases), or None. Binary: also a
    ``sample_weight`` whose weighted positive odds differ from the unweighted odds by more than ``REWEIGHT_ODDS``
    (e.g. ``w = 19`` on positives), the same decalibration by hand. Multiclass: ``class_weight`` only."""
    if params.get("class_weight") is not None:
        return f"class_weight={params['class_weight']!r}"
    if kind != "binary":
        return None
    spw = params.get("scale_pos_weight")
    if spw is not None and float(spw) != 1.0:
        return f"scale_pos_weight={spw}"
    for k in ("is_unbalance", "unbalance", "unbalanced_sets"):
        if params.get(k):
            return f"{k}=True"
    if w is not None and y is not None and w.sum() > 0:
        r, rw = float(np.mean(y)), float(np.sum(w * y) / np.sum(w))
        if 0 < r < 1 and 0 < rw < 1:
            odds = (rw / (1 - rw)) / (r / (1 - r))
            if not 1 / REWEIGHT_ODDS <= odds <= REWEIGHT_ODDS:
                return f"sample_weight (weighted positive share {rw:.3f} vs {r:.3f})"
    return None


def _eval_overlap(X: pd.DataFrame, Xv: pd.DataFrame, labels: bool = True) -> str | None:
    """Why ``eval_set`` looks like it holds training rows, or None. Row labels are compared when both came as
    pandas objects (``labels``) and not both have a fresh 0..n-1 index (no information): a row counts only when its
    label AND its feature values match a training row (a dev frame cut from a separate file shares labels 0..m with
    a train file but not the rows). Otherwise whole feature rows are hashed, if the training rows are mostly
    distinct (repeated rows, e.g. all-categorical features, make identical rows normal)."""
    def fresh(ix):
        return isinstance(ix, pd.RangeIndex) and ix.start == 0 and ix.step == 1
    if not len(X) or not len(Xv):
        return None
    cols = [c for c in X.columns if c in Xv.columns]
    if not cols:
        return None
    if labels and not (fresh(X.index) and fresh(Xv.index)):
        shared = Xv.index.isin(X.index)
        if not shared.any():
            return None
        hv = pd.util.hash_pandas_object(Xv[cols][shared], index=True).to_numpy()
        k = int(np.isin(hv, pd.util.hash_pandas_object(X[cols], index=True).to_numpy()).sum())
        return (f"shares {k} of its {len(Xv)} row labels (index) with the training rows, features identical"
                if k else None)
    h = pd.util.hash_pandas_object(X[cols], index=False).to_numpy()
    if len(np.unique(h)) < 0.9 * len(h):
        return None
    k = int(np.isin(pd.util.hash_pandas_object(Xv[cols], index=False).to_numpy(), h).sum())
    if k and k >= 0.01 * len(Xv):
        return f"has {k} of its {len(Xv)} rows identical to training rows (all {len(cols)} features)"
    return None


class LightGBMBaseline:
    """LightGBM (lazily imported) with the intent kit defaults: 2000 trees max, lr 0.05, 31 leaves,
    min_child_samples 50, subsample/colsample 0.8, early stopping 100 on ``eval_set`` when given (else 300 trees).
    Object/category columns become pandas categoricals with the training categories.

    ``fit(X, y, eval_set=None, sample_weight=None, calibrate_set=None)``; ``sample_weight`` goes to LightGBM (so
    ``oof_predict(weight=...)`` works). Warnings (``UserWarning``), each a silent failure seen in rehearsals:

    - binary minority class below ``RARE_SHARE`` (2%) with min_child_samples left at 50: leaves fit a handful of
      positives and the trees over-fit (D13: ~1% positives, ~6 min per fit, over-fit). Set a larger
      ``min_child_samples`` (200+), ``learning_rate=0.02`` and L2 (``reg_lambda=5``), or start from LogisticBaseline.
    - ``eval_set`` rows that share row labels (index) with the training rows, or (both indexes 0..n-1) are
      identical feature rows: early stopping then follows training rows and the eval score is optimistic.
    - ``scale_pos_weight`` / ``is_unbalance`` / ``class_weight`` set, or (binary) a ``sample_weight`` that moves the
      positive odds by more than ``REWEIGHT_ODDS`` (1.5x): they reweight the classes, so ``predict_proba`` is a
      ranking score, not a probability (D13: no error, the mean prediction moved far above the rate). The warning
      gives the mean in-sample prediction vs the training rate (multiclass: the class with the largest gap).

    ``calibrate="platt"`` (binary only) fits ``methods.hybrid.BinaryCalibrator(method="platt")`` on the raw
    predictions for ``calibrate_set=(X, y)``, else ``eval_set`` (which early stopping also used, mildly optimistic);
    neither given -> ``ValueError``. ``predict`` / ``predict_proba`` then return calibrated probabilities,
    ``predict_proba(X, raw=True)`` the uncalibrated ones. Without it, calibrate held-out predictions yourself with
    ``BinaryCalibrator``.

    ``refit_full=True`` (default off) is the "tree count from early stopping, then train on everything" step: after
    early stopping on ``eval_set`` (required), the model is refit once on train + eval rows with
    ``round(best_iteration_ * refit_scale)`` trees (``refit_scale`` 1.1 by default: ~10% more rows, a few more trees).
    ``best_iteration_`` / ``n_estimators_`` keep both counts (set without refit too: both = trees used); categories
    cover the eval rows too. No held-out score is left after the refit, so read the dev score from a
    ``refit_full=False`` fit. With ``calibrate`` it needs a separate ``calibrate_set`` (the eval rows are now training
    rows); ``sample_weight`` is refused (eval rows have none).

    Usage::

        m = LightGBMBaseline(scale_pos_weight=20, calibrate="platt", min_child_samples=300)
        m.fit(X_tr, y_tr, eval_set=(X_dev, y_dev))
        p = m.predict_proba(X_test)                 # calibrated; raw=True for the reweighted score
        LightGBMBaseline(refit_full=True).fit(X_tr, y_tr, eval_set=(X_dev, y_dev))   # final model on train + dev
    """

    CALIBRATE = (None, "platt")

    def __init__(self, kind: str = "auto", seed: int = 0, objective: str | None = None, calibrate: str | None = None,
                 refit_full: bool = False, refit_scale: float = 1.1, **params):
        if calibrate not in self.CALIBRATE:
            raise ValueError(f"calibrate must be one of {self.CALIBRATE}, got {calibrate!r}")
        if not refit_scale > 0:
            raise ValueError(f"refit_scale must be > 0, got {refit_scale!r}")
        self.kind, self.seed, self.objective, self.calibrate, self.params = kind, seed, objective, calibrate, params
        self.refit_full, self.refit_scale = refit_full, refit_scale

    def _frame(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cat_:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.categories_[c])
        return X[self.columns_]

    def _warn_rare(self, y):
        if self.kind_ != "binary" or not len(y) or any(k in self.params for k in _MIN_CHILD_KEYS):
            return
        rate = float(np.mean(y))
        share, name = (rate, "positive") if rate <= 0.5 else (1 - rate, "negative")
        if 0 < share < RARE_SHARE:
            k = int(round(share * len(y)))
            warnings.warn(f"LightGBMBaseline: {name} share {share:.2%} ({k} of {len(y)} rows) with the default "
                          "min_child_samples=50: leaves fit a few rare rows and the trees over-fit (D13: ~1% "
                          "positives, ~6 min per fit). Try min_child_samples=200+ (a leaf should hold several rare "
                          "rows), learning_rate=0.02, reg_lambda=5 (L2), or LogisticBaseline first",
                          UserWarning, stacklevel=3)

    def fit(self, X, y, eval_set: tuple | None = None, sample_weight=None, calibrate_set: tuple | None = None):
        lgb = require_lightgbm()
        labeled = isinstance(X, (pd.DataFrame, pd.Series))
        X = pd.DataFrame(X)
        y = np.asarray(y)
        w = _sample_weight(sample_weight, len(X))
        self.kind_ = _check_kind(self.kind, y)
        cal_set = calibrate_set if calibrate_set is not None else eval_set
        if self.calibrate and self.kind_ != "binary":
            raise ValueError(f"calibrate={self.calibrate!r} is for a binary target, this one is {self.kind_}")
        if self.calibrate and cal_set is None:
            raise ValueError(f"calibrate={self.calibrate!r} needs held-out rows: pass eval_set= or calibrate_set=")
        if self.refit_full:
            if eval_set is None:
                raise ValueError("refit_full=True needs eval_set= (early stopping picks the tree count)")
            if self.calibrate and calibrate_set is None:
                raise ValueError("refit_full=True trains on the eval rows: calibrate needs a separate calibrate_set=")
            if w is not None:
                raise ValueError("refit_full=True with sample_weight: the eval rows have no weights to refit with")
        self.columns_ = list(X.columns)
        self.cat_ = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        self.categories_ = {c: sorted(X[c].astype(str).unique()) for c in self.cat_}
        prm = dict(n_estimators=2000 if eval_set is not None else 300, learning_rate=0.05, num_leaves=31,
                   min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   num_threads=n_threads(len(X)), random_state=self.seed, verbose=-1)
        if self.objective:
            prm["objective"] = self.objective
        prm.update(self.params)
        Model = lgb.LGBMRegressor if self.kind_ == "continuous" else lgb.LGBMClassifier
        if self.kind_ == "multiclass":
            y = y.astype(str)
        elif self.kind_ == "binary":
            y = y.astype(int)
        self._warn_rare(y)
        kw = {} if w is None else {"sample_weight": w}
        if eval_set is not None:
            why = _eval_overlap(X, pd.DataFrame(eval_set[0]),
                                labels=labeled and isinstance(eval_set[0], (pd.DataFrame, pd.Series)))
            if why:
                warnings.warn(f"LightGBMBaseline: eval_set {why}; early stopping then tracks rows the trees were "
                              "fit on (stops late) and the eval score is optimistic. Split by unit / time first",
                              UserWarning, stacklevel=2)
            yv = np.asarray(eval_set[1])
            yv = yv.astype(str) if self.kind_ == "multiclass" else yv
            kw.update(_eval_kw(Model, self._frame(eval_set[0]), yv))
            if self.kind_ == "multiclass" and "eval_y" in kw:
                # LightGBM 4.7 label-encodes eval_set labels but not eval_y: give it the class codes (sorted order)
                cls = np.unique(y)
                if not np.isin(yv, cls).all():
                    raise ValueError(f"eval_set has classes the training rows lack: {sorted(set(yv) - set(cls))[:5]}")
                kw["eval_y"] = (np.searchsorted(cls, yv),)
            kw["callbacks"] = [lgb.early_stopping(100, verbose=False)]
        self.model_ = Model(**prm).fit(self._frame(X), y, **kw)
        # set in both modes (D13: a refit_full=False fit had them only on model_); no eval_set -> all trees
        self.best_iteration_ = int(self.model_.best_iteration_ or prm["n_estimators"])
        self.n_estimators_ = self.best_iteration_
        if self.refit_full:
            self.n_estimators_ = max(1, int(round(self.best_iteration_ * self.refit_scale)))
            X = pd.concat([X, pd.DataFrame(eval_set[0])[self.columns_]], ignore_index=True)
            # continuous: let numpy promote (int train labels must not truncate float eval labels)
            y = np.concatenate([y, yv if self.kind_ == "continuous" else yv.astype(y.dtype)])
            self.categories_ = {c: sorted(X[c].astype(str).unique()) for c in self.cat_}
            rp = {k: v for k, v in prm.items() if k not in _REFIT_DROP_KEYS}
            self.model_ = Model(**{**rp, "n_estimators": self.n_estimators_}).fit(self._frame(X), y)
        if self.kind_ == "multiclass":
            self.classes_ = list(self.model_.classes_)
        self.calibrator_ = None
        rew = _reweighting(prm, self.kind_, y, w) if self.kind_ != "continuous" else None
        if rew and self.kind_ == "multiclass":
            P = np.asarray(self.predict_proba(X), float)
            share = pd.Series(y).value_counts(normalize=True).reindex(self.classes_, fill_value=0).to_numpy()
            j = int(np.argmax(np.abs(P.mean(0) - share)))
            warnings.warn(f"LightGBMBaseline: {rew} reweights the classes, so predict_proba is not calibrated (class "
                          f"{self.classes_[j]!r}: mean prediction on the training rows {P[:, j].mean():.3f} vs "
                          f"training share {share[j]:.3f}); use it as a ranking score or recalibrate on held-out "
                          "rows", UserWarning, stacklevel=2)
        elif self.calibrate:
            from .hybrid import BinaryCalibrator
            yc = np.asarray(cal_set[1]).astype(int)
            self.calibrator_ = BinaryCalibrator(method=self.calibrate).fit(self.predict_proba(cal_set[0], raw=True),
                                                                           yc)
        elif rew:
            m = float(self.predict_proba(X, raw=True).mean())
            warnings.warn(f"LightGBMBaseline: {rew} reweights the classes, so predict_proba is not calibrated (mean "
                          f"prediction on the training rows {m:.3f} vs training rate {float(np.mean(y)):.3f}); use it "
                          "as a ranking score, or pass calibrate='platt' with eval_set= / calibrate_set= (or fit "
                          "methods.hybrid.BinaryCalibrator on held-out predictions)", UserWarning, stacklevel=2)
        return self

    def predict_proba(self, X, raw: bool = False):
        p = _proba_out(self.model_, self.model_.predict_proba(self._frame(X)), self.kind_)
        cal = getattr(self, "calibrator_", None)
        return p if raw or cal is None else cal.predict(p)

    def predict(self, X):
        if self.kind_ == "continuous":
            return self.model_.predict(self._frame(X))
        return self.predict_proba(X)

    def feature_importance(self, top: int = 10) -> dict:
        imp = pd.Series(self.model_.booster_.feature_importance("gain"), index=self.columns_)
        return {k: float(v) for k, v in imp.sort_values(ascending=False).head(top).items()}


class KNNBaseline:
    """k nearest neighbours in standardised feature space (one-hot categoricals), k=50 as in the persona kit's
    ``knn50`` base. Binary/multiclass -> neighbour class shares, continuous -> neighbour mean."""

    def __init__(self, k: int = 50, kind: str = "auto", categorical: Sequence[str] | None = None):
        self.k, self.kind, self.categorical = k, kind, categorical

    def fit(self, X, y):
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        X = pd.DataFrame(X)
        y = np.asarray(y)
        self.kind_ = _check_kind(self.kind, y)
        self.cat_, self.num_ = _split_columns(X, self.categorical)
        k = max(1, min(self.k, len(X)))
        est = KNeighborsRegressor(k) if self.kind_ == "continuous" else KNeighborsClassifier(k)
        if self.kind_ == "binary":
            y = y.astype(int)
        elif self.kind_ == "multiclass":
            y = y.astype(str)
        self.model_ = make_pipeline(_preprocessor(self.cat_, self.num_), est).fit(X[self.cat_ + self.num_], y)
        if self.kind_ == "multiclass":
            self.classes_ = list(self.model_.classes_)
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X)[self.cat_ + self.num_]
        return _proba_out(self.model_, self.model_.predict_proba(X), self.kind_)

    def predict(self, X):
        if self.kind_ == "continuous":
            return self.model_.predict(pd.DataFrame(X)[self.cat_ + self.num_])
        return self.predict_proba(X)


class TextKNNRegressor:
    """Similar past experiments (deney kit `KomsuDeney`): TF-IDF (1-2 grams, min_df=2, sublinear) cosine similarity,
    similarity-weighted mean outcome of the k=10 most similar training texts from OTHER groups.
    Measured on SocSci210 effect sizes: Pearson r 0.080 [-0.024, 0.232] (uretim/deney/RAPOR.md, 1.4): weak."""

    def __init__(self, k: int = 10, min_df: int = 2):
        self.k, self.min_df = k, min_df

    def fit(self, texts, y, groups=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [str(t) for t in texts]
        self.v_ = TfidfVectorizer(ngram_range=(1, 2), min_df=min(self.min_df, max(1, len(texts))),
                                  sublinear_tf=True)
        self.X_ = self.v_.fit_transform(texts)
        self.y_ = np.asarray(y, float)
        self.g_ = np.asarray(groups) if groups is not None else np.arange(len(texts)).astype(object)
        return self

    def predict(self, texts, groups=None):
        S = (self.v_.transform([str(t) for t in texts]) @ self.X_.T).toarray()
        tg = np.asarray(groups) if groups is not None else np.full(len(S), object())
        out = np.empty(len(S))
        for i in range(len(S)):
            s = np.where(self.g_ == tg[i], -1, S[i])
            j = np.argsort(-s)[: self.k]
            w = np.clip(s[j], 1e-3, None)
            out[i] = np.sum(w * self.y_[j]) / w.sum()
        return out
