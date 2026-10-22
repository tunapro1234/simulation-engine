"""Non-LLM baselines: the ladder every simulation result is reported against.

Ported from the TALP prep research kits (intent kit `yontemler.py`, persona kit `Medyan`, deney kit `KomsuDeney`,
kanit `merdiven.py`, makro `temel_c1.py`, intent/seq `klasik.py`). Algorithms and defaults are kept.

Uniform API: classes have ``fit(...)`` returning ``self`` and ``predict(...)``; binary/multiclass classes also have
``predict_proba``. Distribution and basket baselines are pure functions.

kind: "binary" (P(y=1) out), "multiclass" (probability matrix, columns = ``classes_``), "continuous" (number out).
"""
from __future__ import annotations

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


# ---------------------------------------------------------------- knows nothing

class ConstantBaseline:
    """Same prediction for everyone (intent kit `_sabit`): binary = training rate, multiclass = training class shares,
    continuous = training median (``stat="mean"`` for the mean). ``predict`` accepts a row count or any sized input."""

    def __init__(self, kind: str = "auto", stat: str = "median"):
        self.kind, self.stat = kind, stat

    def fit(self, y, X=None):
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
    in the Twin-2K-500 persona task the column median alone scores acc 0.755 (uretim/persona/RAPOR.md, T1 table)."""

    def fit(self, y, X=None):
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
            M = T.to_numpy(float)
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
            out = sh.reindex(pd.Series(seg).to_numpy()).to_numpy(float)
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
    answer distributions, 1-TVD) shrunk segment empirical (m=20) was the strongest LLM-free arm, above the
    population copy and the raw segment empirical, and the LLM hybrid did not beat it across 5 seeds.
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
    """``select_m`` result: chosen pseudo-count, mean 1-TVD per grid value, number of scored cells per fold."""
    m: float
    scores: dict
    n_cells: int


def select_m(df: pd.DataFrame, value_col, segment_col, categories, grid: Sequence[float] = M_GRID,
             folds: int | None = 5, unit_col: str | None = None, weight_col: str | None = None,
             min_eval: int = 1, seed: int = 0) -> ShrinkSelection:
    """Pick ``m`` for ``mode="shrunk"`` by K-fold 1-TVD on training data only.

    Units (rows, or ``unit_col`` groups such as households, so one unit never sits on both sides) are split into
    ``folds`` folds; ``folds=None`` is leave-one-unit-out. Each fold is held out once: shrunk distributions are fit
    on the rest (overall marginal from the rest too) and scored against the held-out segment's own empirical
    marginal by 1 - TVD. Cells (segment x item) with fewer than ``min_eval`` held-out answers are skipped; the
    score is the mean over scored cells and folds. ``value_col`` may be a list of item columns; ``categories`` is
    then one shared list or a dict {column: categories}. Ties go to the smaller ``m``.
    Note: the held-out empirical is noisy, so small folds favour larger ``m`` a little less than the truth would;
    5 folds is the default for that reason.
    """
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
                tot[g] += [1.0 - 0.5 * np.abs(pred[s] - truth[s]).sum() for s in segs]
    if not n_cells:
        raise ValueError("no held-out cell with enough answers; lower min_eval or folds")
    scores = {g: float(np.mean(v)) for g, v in tot.items()}
    best = max(grid, key=lambda g: (round(scores[g], 12), -g))
    return ShrinkSelection(m=float(best), scores=scores, n_cells=n_cells)


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
        p = u.map(self.values_).to_numpy(float)
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
    targets (intent kit `lineer`). Train rows above ``max_rows`` are subsampled (seed 0)."""

    def __init__(self, kind: str = "auto", categorical: Sequence[str] | None = None, C: float = 1.0,
                 alpha: float = 1.0, max_rows: int = 300_000):
        self.kind, self.categorical, self.C, self.alpha, self.max_rows = kind, categorical, C, alpha, max_rows

    def fit(self, X: pd.DataFrame, y):
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        X = pd.DataFrame(X)
        y = np.asarray(y)
        self.kind_ = _check_kind(self.kind, y)
        if len(X) > self.max_rows:
            i = np.random.default_rng(0).choice(len(X), self.max_rows, replace=False)
            X, y = X.iloc[i], y[i]
        self.cat_, self.num_ = _split_columns(X, self.categorical)
        est = Ridge(alpha=self.alpha) if self.kind_ == "continuous" else LogisticRegression(max_iter=2000, C=self.C)
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


def _eval_kw(Model, Xv, yv) -> dict:
    """Validation-set keywords for LightGBM's sklearn fit: eval_X/eval_y on versions that have them (4.7+
    deprecates eval_set), else eval_set."""
    import inspect
    if "eval_X" in inspect.signature(Model.fit).parameters:
        return {"eval_X": (Xv,), "eval_y": (yv,)}
    return {"eval_set": [(Xv, yv)]}


class LightGBMBaseline:
    """LightGBM (lazily imported) with the intent kit defaults: 2000 trees max, lr 0.05, 31 leaves,
    min_child_samples 50, subsample/colsample 0.8, early stopping 100 on ``eval_set`` when given (else 300 trees).
    Object/category columns become pandas categoricals with the training categories."""

    def __init__(self, kind: str = "auto", seed: int = 0, objective: str | None = None, **params):
        self.kind, self.seed, self.objective, self.params = kind, seed, objective, params

    def _frame(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cat_:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.categories_[c])
        return X[self.columns_]

    def fit(self, X, y, eval_set: tuple | None = None):
        lgb = require_lightgbm()
        X = pd.DataFrame(X)
        y = np.asarray(y)
        self.kind_ = _check_kind(self.kind, y)
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
        kw = {}
        if eval_set is not None:
            yv = np.asarray(eval_set[1])
            yv = yv.astype(str) if self.kind_ == "multiclass" else yv
            kw = _eval_kw(Model, self._frame(eval_set[0]), yv)
            kw["callbacks"] = [lgb.early_stopping(100, verbose=False)]
        self.model_ = Model(**prm).fit(self._frame(X), y, **kw)
        if self.kind_ == "multiclass":
            self.classes_ = list(self.model_.classes_)
        return self

    def predict_proba(self, X):
        return _proba_out(self.model_, self.model_.predict_proba(self._frame(X)), self.kind_)

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
