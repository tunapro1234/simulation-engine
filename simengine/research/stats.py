"""Bootstrap helpers for the research loop (ported from uretim/dongu/kod/kosucu.py: bootstrap_fark, bootstrap_ort).

Kept local on purpose: the loop's acceptance rule must not change when another module tunes its own CI code.
Defaults (seed 42, B=1000, percentile CI) are the ones the measured loops used.
"""
from __future__ import annotations

import numpy as np

B_BOOT = 1000


def paired_bootstrap_diff(a, b, seed: int = 42, B: int = B_BOOT, alpha: float = 0.05) -> dict:
    """Paired difference mean(a - b) with a (1 - alpha) percentile bootstrap CI.

    `a`, `b`: per-unit scores in the same unit order. Returns {diff, lo, hi, n, alpha}."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) == 0:
        raise ValueError("paired_bootstrap_diff: no common units")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    bs = d[idx].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(bs, 50 * alpha)),
            "hi": float(np.percentile(bs, 100 - 50 * alpha)), "n": int(len(d)), "alpha": alpha}


def bootstrap_mean(a, seed: int = 42, B: int = B_BOOT, alpha: float = 0.05) -> dict:
    """Mean with a percentile bootstrap CI. Returns {value, lo, hi, n}."""
    a = np.asarray(a, float)
    if len(a) == 0:
        return {"value": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), (B, len(a)))].mean(1)
    return {"value": float(a.mean()), "lo": float(np.percentile(bs, 50 * alpha)),
            "hi": float(np.percentile(bs, 100 - 50 * alpha)), "n": int(len(a))}


def direction(diff: dict) -> str:
    """'lower' if the CI is entirely below 0, 'higher' if entirely above, else 'same'."""
    return "lower" if diff["hi"] < 0 else "higher" if diff["lo"] > 0 else "same"


def is_better(diff: dict, higher_is_better: bool) -> bool:
    """Does the paired diff (a - b) show a significantly better than b (CI excludes 0 on the good side)?"""
    return diff["lo"] > 0 if higher_is_better else diff["hi"] < 0


def is_tie(diff: dict) -> bool:
    return diff["lo"] <= 0 <= diff["hi"]
