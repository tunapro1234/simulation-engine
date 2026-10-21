"""Runner helpers ported from sahne/kosucu/ortak.py: JSON-safe conversion, number/choice parsing from LLM
output, percentile bootstrap CIs over personas, bootstrap win probability, sign test, segment breakdown,
representative quotes, event emitter. All CIs here resample personas (the unit that varies in a panel).
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable

import numpy as np

B_DEFAULT = 1000
ALL = "all"


# ---------------------------------------------------------------- types / formats
def to_jsonable(x: Any) -> Any:
    """numpy -> python; NaN/inf -> None; tuple -> list; dict keys -> str."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else f
    return x


def to_number(x: Any, lo: float | None = None, hi: float | None = None) -> float | None:
    """LLM value -> float ("7", "7/10", 7.0, "70%", "7,5"); clipped to [lo, hi]; None if unparseable."""
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
    else:
        m = re.search(r"-?\d+(?:[.,]\d+)?", str(x))
        if not m:
            return None
        v = float(m.group(0).replace(",", "."))
    if math.isnan(v) or math.isinf(v):
        return None
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def field_name(s: Any, i: int) -> str:
    """Safe JSON field stem from a question/variant id (e.g. 'v1', 'q_price')."""
    t = re.sub(r"[^a-zA-Z0-9_]", "_", str(s)).strip("_")[:20]
    return t if t and t[0].isalpha() else (f"x{i}_{t}" if t else f"x{i}")


def resolve_choice(x: Any, ids: list[str]) -> str | None:
    """'B', 'Variant B', 'b' -> 'B' (from the id list); None if no id is found."""
    if x is None:
        return None
    s = str(x).strip()
    if s in ids:
        return s
    low = {k.lower(): k for k in ids}
    if s.lower() in low:
        return low[s.lower()]
    for k in sorted(ids, key=len, reverse=True):
        if re.search(rf"(^|[^a-zA-Z0-9]){re.escape(k)}([^a-zA-Z0-9]|$)", s, re.I):
            return k
    return None


# ---------------------------------------------------------------- statistics
def _clean(values) -> np.ndarray:
    return np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))],
                      dtype=float)


def bootstrap_ci(values, B: int = B_DEFAULT, seed: int = 0, fn: Callable = np.mean, alpha: float = 0.05) -> dict:
    """Persona-level values -> {value, low, high, n} (percentile bootstrap)."""
    x = _clean(values)
    n = len(x)
    if n == 0:
        return {"value": None, "low": None, "high": None, "n": 0}
    d = float(fn(x))
    if n == 1:
        return {"value": d, "low": d, "high": d, "n": 1}
    r = np.random.default_rng(seed)
    idx = r.integers(0, n, size=(max(1, B), n))
    bs = x[idx].mean(axis=1) if fn is np.mean else np.apply_along_axis(fn, 1, x[idx])
    return {"value": d, "low": float(np.quantile(bs, alpha / 2)), "high": float(np.quantile(bs, 1 - alpha / 2)),
            "n": n}


def bootstrap_distribution(P, B: int = B_DEFAULT, seed: int = 0, alpha: float = 0.05) -> dict:
    """Rows = persona probability / one-hot vectors (n, k) -> mean distribution + per-cell CI."""
    P = np.asarray(P, dtype=float)
    if P.size == 0:
        return {"value": [], "low": [], "high": [], "n": 0}
    n = len(P)
    r = np.random.default_rng(seed)
    bs = P[r.integers(0, n, size=(max(1, B), n))].mean(axis=1)
    return {"value": P.mean(axis=0).tolist(), "low": np.quantile(bs, alpha / 2, axis=0).tolist(),
            "high": np.quantile(bs, 1 - alpha / 2, axis=0).tolist(), "n": n}


def bootstrap_win(M, B: int = B_DEFAULT, seed: int = 0) -> list[float]:
    """M (n personas, k options) paired scores -> probability that each column has the highest mean
    (personas resampled; ties split). Rows with NaN are dropped."""
    M = np.asarray(M, dtype=float)
    if M.size:
        M = M[~np.isnan(M).any(axis=1)]
    if M.size == 0:
        return []
    n, _ = M.shape
    r = np.random.default_rng(seed)
    mean = M[r.integers(0, n, size=(max(1, B), n))].mean(axis=1)
    top = mean == mean.max(axis=1, keepdims=True)
    return (top / top.sum(axis=1, keepdims=True)).mean(axis=0).tolist()


def sign_test(diffs) -> dict:
    """Two-sided binomial sign test on paired differences (ties dropped)."""
    f = np.asarray([v for v in diffs if v is not None], dtype=float)
    pos, neg, tie = int((f > 0).sum()), int((f < 0).sum()), int((f == 0).sum())
    if pos + neg == 0:
        p = 1.0
    else:
        from scipy.stats import binomtest
        p = float(binomtest(pos, pos + neg, 0.5).pvalue)
    return {"positive": pos, "negative": neg, "ties": tie, "p": p}


def summary_stats(values) -> dict:
    x = _clean(values)
    if len(x) == 0:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "n": int(len(x))}


def segment_breakdown(value: dict, seg: dict, fn: Callable = bootstrap_ci, **kw) -> dict:
    """value {pid: v}, seg {pid: segment} -> {segment: fn([v...])} sorted by segment name."""
    groups: dict[str, list] = {}
    for pid, v in value.items():
        if v is not None:
            groups.setdefault(seg.get(pid, ALL), []).append(v)
    return {s: fn(groups[s], **kw) for s in sorted(groups)}


def pick_quotes(records: list[dict], n: int = 4) -> list[dict]:
    """records [{pid, segment, text, value}] -> n representative quotes covering the value range and segments
    (lowest, highest, around the median, then unseen segments)."""
    k = [x for x in records if x.get("text")]
    if not k:
        return []
    numeric = sorted([x for x in k if x.get("value") is not None], key=lambda x: x["value"])
    chosen: list[dict] = []

    def add(x):
        if x is not None and x not in chosen and len(chosen) < n:
            chosen.append(x)

    if numeric:
        add(numeric[0])
        add(numeric[-1])
        add(numeric[len(numeric) // 2])
    seen = {x.get("segment") for x in chosen}
    for x in k:
        if x.get("segment") not in seen:
            add(x)
            seen.add(x.get("segment"))
    for x in k:
        add(x)
    return [{"pid": x["pid"], "segment": x.get("segment"), "text": str(x["text"])[:300], "value": x.get("value")}
            for x in chosen]


def split_halves(items):
    """Split personas into two halves (even / odd index) to balance order effects."""
    return items[0::2], items[1::2]


# ---------------------------------------------------------------- events
class Emitter:
    """Runner events: {"type": "row", runner, pid, segment, summary, value}, {"type": "stage", runner, name},
    {"type": "done", runner}. A failing callback never stops the run."""

    def __init__(self, callback: Callable[[dict], None] | None, runner: str):
        self.callback = callback
        self.runner = runner

    def emit(self, kind: str, **kw):
        if self.callback:
            try:
                self.callback(to_jsonable({"type": kind, "runner": self.runner, **kw}))
            except Exception:  # noqa: BLE001
                pass

    def row(self, pid, segment, summary, value=None, **kw):
        self.emit("row", pid=pid, segment=segment, summary=str(summary)[:200], value=value, **kw)

    def stage(self, name, **kw):
        self.emit("stage", name=name, **kw)

    def done(self, **kw):
        self.emit("done", **kw)
