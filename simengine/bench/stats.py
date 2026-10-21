"""Small, dependency-free metric and bootstrap helpers used by the bench tasks.

Kept local to `simengine.bench` so tasks can score themselves without importing other modules'
internals. Definitions follow the research code they were ported from (see each docstring).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

B_DEFAULT = 1000
SEED_DEFAULT = 42


def ci(point: float, boots: np.ndarray, *, kind: str = "percentile") -> tuple[float, float]:
    """95% interval from bootstrap replicates. kind='percentile' (default) or 'normal' (point +- 1.96 sd),
    the latter used by talp73 for aggregate (distribution / correlation) measures."""
    b = np.asarray(boots, float)
    b = b[np.isfinite(b)]
    if not np.isfinite(point) or len(b) < 10:
        return float("nan"), float("nan")
    if kind == "normal":
        s = float(b.std())
        return float(point - 1.96 * s), float(point + 1.96 * s)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_mean(values: Sequence[float], *, B: int = B_DEFAULT, seed: int = SEED_DEFAULT) -> dict:
    """Mean of per-unit scores with a unit bootstrap (dongu `bootstrap_ort`: default_rng(42), B = 1000)."""
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), (B, len(a)))].mean(1)
    return {"value": float(a.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "n": int(len(a))}


def cluster_bootstrap(fn: Callable[[np.ndarray], float], groups: Sequence, *, B: int = B_DEFAULT,
                      seed: int = SEED_DEFAULT, kind: str = "percentile") -> dict:
    """Point estimate fn(all row indices) and a cluster bootstrap over `groups` (rows of one group move together).
    fn receives an integer index array (with repeats) into the original rows."""
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(uniq)))]
    members = [order[starts[i]:starts[i + 1]] for i in range(len(uniq))]
    point = float(fn(np.arange(len(g))))
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        vals[b] = fn(np.concatenate([members[i] for i in pick]))
    lo, hi = ci(point, vals, kind=kind)
    return {"value": point, "lo": lo, "hi": hi, "n": int(len(uniq))}


def fisher_z(r: Sequence[float]) -> np.ndarray:
    return np.arctanh(np.clip(np.asarray(r, float), -0.999999, 0.999999))


def pearson(a: np.ndarray, b: np.ndarray, min_n: int = 3) -> float:
    """Pearson r on finite pairs; NaN if fewer than min_n pairs or a constant side."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < min_n or a.std() <= 1e-12 or b.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def wasserstein_ordinal(p: np.ndarray, q: np.ndarray, pos: np.ndarray) -> float:
    """Wasserstein-1 between two distributions on option positions scaled to 0-1 (dongu `wd`; SubPOP standard)."""
    pos = np.asarray(pos, float)
    o = np.argsort(pos)
    pos, p, q = pos[o], np.asarray(p, float)[o], np.asarray(q, float)[o]
    return float(np.sum(np.abs(np.cumsum(p)[:-1] - np.cumsum(q)[:-1]) * np.diff(pos)))


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance 0.5 * sum |p - q| (inputs normalised to sum 1)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    return float(0.5 * np.abs(p - q).sum())


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC with ties counted 0.5 (rank formula). NaN if one class is missing."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pair_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """A/B pair accuracy: p > 0.5 predicts A; p == 0.5 scores half (deney `pairwise_accuracy`)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean(np.where(np.isclose(p, 0.5), 0.5, ((p > 0.5) == (y == 1)).astype(float))))
