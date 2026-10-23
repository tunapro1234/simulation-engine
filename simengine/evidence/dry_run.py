"""Dry run of the evidence pipeline: the code path of `evaluate` on masked truth, never the real numbers.

    import simengine.evidence as ev
    rep = ev.dry_run(frame, title="T2 ladder", ours="t2", seed=0)     # same kwargs as evaluate; truth masked first
    rep["baslik"]          # "DRY RUN: truth masked, numbers are not results | T2 ladder"
    rep["dry_run"]         # {"mode", "level", "seed", "masked_rows", "unchanged_share", "note"}
    ev.card(rep, "dry.md") # the stamp is in the title and the first limit line
    masked, info = ev.mask_truth(frame, seed=0, mode="marginal")      # the masked copy only (input left unchanged)

Why (TALP G2 #1): with FakeLLM the code smoke test still scored against the real labels, so the LLM-free ladder
values were seen before the pre-registration commit. A dry run checks that the pipeline runs end to end (columns,
task detection, baselines, card) without showing a single real-truth number.

Masking. mode="permute" (default) shuffles the truth values within each item (the whole column when there is no
item column); mode="marginal" replaces each value by a draw with replacement from its item's truth values. Item
marginals (class shares, mean) are kept; they are data, not results. Rows with missing truth stay missing. When
the item level cannot hide the truth (more than half of the rows sit in items with one distinct truth value or, with
aggregate_by, one aggregation unit, e.g. one row per item), the mask moves to the whole column, and for
distribution truths of different lengths (or a column that is still frozen) to a per-row shuffle of the vector's
components. Truth that no mask can change (a constant column) raises ValueError: there is no dry run of it.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..types import COLUMN_ALIASES

DRY_RUN_STAMP = "DRY RUN: truth masked, numbers are not results"
MODES = ("permute", "marginal")
FROZEN_MAX = 0.5          # share of rows a level may leave unchangeable before the mask moves one level up
_EN_TO_TR = {v: k for k, v in COLUMN_ALIASES.items()}


def _col(frame: pd.DataFrame, name: str | None) -> str | None:
    """Column of the frame under its English or Turkish (alias) name, or None."""
    if name is None:
        return None
    for c in (name, _EN_TO_TR.get(name), COLUMN_ALIASES.get(name)):
        if c is not None and c in frame.columns:
            return c
    return None


def _missing(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip()) or (np.isscalar(v) and pd.isna(v))


def _vec(v) -> np.ndarray | None:
    """A distribution cell as a float vector, None for scalars and {class: p} dicts."""
    if isinstance(v, str) and v.lstrip().startswith("["):
        v = json.loads(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, dtype=float).ravel()
    return None


def _key(v) -> Any:
    """Hashable comparison key of a truth cell (vectors rounded so float noise is not a change)."""
    x = _vec(v)
    if x is not None:
        return ("v",) + tuple(np.round(x, 12).tolist())
    if isinstance(v, dict):
        return ("d",) + tuple(sorted((str(k), round(float(p), 12)) for k, p in v.items()))
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    try:
        return round(float(v), 12)
    except (TypeError, ValueError):
        return str(v)


def _frozen(keys: list, strata: np.ndarray, units: np.ndarray | None) -> np.ndarray:
    """Rows whose stratum cannot change under a within-stratum mask: one distinct truth value, or (aggregate_by)
    one aggregation unit, whose mean a within-stratum shuffle leaves as it was."""
    out = np.zeros(len(keys), bool)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        one_value = len({keys[i] for i in idx}) < 2
        one_unit = units is not None and len(np.unique(units[idx])) < 2
        out[idx] = one_value or one_unit
    return out


def _shuffle(values: list, strata: np.ndarray, mode: str, rng: np.random.Generator) -> list:
    out = list(values)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        src = rng.permutation(idx) if mode == "permute" else rng.choice(idx, size=len(idx), replace=True)
        for i, j in zip(idx, src):
            out[i] = values[j]
    return out


def _components(values: list, rng: np.random.Generator) -> list:
    """Per-row random order of a distribution's components (lists stay lists, JSON stays JSON)."""
    out = []
    for v in values:
        x = _vec(v)
        y = x[rng.permutation(len(x))]
        out.append(json.dumps(y.tolist()) if isinstance(v, str) else (y.tolist() if isinstance(v, list) else y))
    return out


def mask_truth(frame: pd.DataFrame, *, seed: int = 0, mode: str = "permute", item: str | None = "item",
               aggregate_by: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Copy of `frame` with the truth column masked (see the module docstring) and an info dict:
    mode, level ("item" | "all" | "components"), seed, masked_rows, unchanged_share (rows whose value did not
    change, by chance or because their stratum is frozen), note. `frame` itself is not modified."""
    if mode not in MODES:
        raise ValueError(f"mask_truth: mode must be one of {MODES}, got {mode!r}")
    tcol = _col(frame, "truth")
    if tcol is None:
        raise ValueError("mask_truth: no 'truth' column (alias: gercek); a frame without truth has nothing to mask")
    out = frame.copy()
    have = np.flatnonzero(~np.array([_missing(v) for v in frame[tcol]], dtype=bool))
    if len(have) == 0:
        raise ValueError("mask_truth: every truth value is missing; evaluate gives the 'awaiting truth' report")
    vals = [frame[tcol].iloc[i] for i in have]
    keys = [_key(v) for v in vals]
    icol, acol = _col(frame, item), _col(frame, aggregate_by)
    units = pd.factorize(frame[acol].iloc[have])[0] if acol else None
    vecs = [_vec(v) for v in vals]
    is_vec = all(x is not None for x in vecs)
    same_len = is_vec and len({len(x) for x in vecs}) == 1
    rng = np.random.default_rng(seed)
    levels = ((["item"] if icol else []) + (["all"] if not is_vec or same_len else [])
              + (["components"] if is_vec else []))
    for level in levels:
        if level == "components":
            frozen = np.array([len(np.unique(np.round(x, 12))) < 2 for x in vecs])
        else:
            strata = (pd.factorize(frame[icol].iloc[have])[0] if level == "item" else np.zeros(len(have), int))
            frozen = _frozen(keys, strata, units)
        if frozen.mean() <= FROZEN_MAX:
            break
    else:
        raise ValueError("mask_truth: the truth cannot be masked (constant column, or every row frozen): a dry run "
                         "would show the real numbers")
    new = _components(vals, rng) if level == "components" else _shuffle(vals, strata, mode, rng)
    s = out[tcol].astype(object)
    for i, v in zip(have, new):
        s.iat[i] = v
    out[tcol] = s if frame[tcol].dtype == object else pd.Series(list(s), index=out.index, dtype=frame[tcol].dtype)
    unchanged = float(np.mean([_key(v) == k for v, k in zip(new, keys)]))
    where = {"item": f"within {icol!r}", "all": "across the whole column",
             "components": "by shuffling each distribution's components"}[level]
    how = "masked" if level == "components" else "permuted" if mode == "permute" else "redrawn from the marginal"
    note = f"{DRY_RUN_STAMP}: truth {how} {where} (seed {seed}, {len(have)} rows, {100 * unchanged:.0f} % unchanged)"
    return out, {"mode": mode, "level": level, "seed": int(seed), "masked_rows": int(len(have)),
                 "unchanged_share": unchanged, "note": note}


def dry_run(frame: pd.DataFrame, *, seed: int = 0, mode: str = "permute", item: str | None = "item",
            **evaluate_kw) -> dict:
    """`evaluate(**evaluate_kw)` on `mask_truth(frame)`: the report's title (`baslik`) is stamped
    "DRY RUN: truth masked, numbers are not results", the stamp leads the limits (card), and `report["dry_run"]`
    holds the mask info. seed seeds the mask and is passed to evaluate. The real truth is never scored."""
    if "df" in evaluate_kw:
        raise TypeError("dry_run: pass the frame as the first argument")
    masked, info = mask_truth(frame, seed=seed, mode=mode, item=item, aggregate_by=evaluate_kw.get("aggregate_by"))
    from .api import evaluate              # late: api imports the whole evidence layer

    title = evaluate_kw.pop("title", "") or ""
    stamped = f"{DRY_RUN_STAMP} | {title}" if title else DRY_RUN_STAMP
    evaluate_kw["limits"] = [info["note"]] + list(evaluate_kw.get("limits") or [])
    rep = evaluate(masked, stamped, seed=seed, **evaluate_kw)
    rep["baslik"] = stamped
    lim = list(rep.get("sinirlar") or [])
    if info["note"] in lim:
        lim.remove(info["note"])
    rep["sinirlar"] = [info["note"]] + lim
    rep["dry_run"] = info
    return rep
