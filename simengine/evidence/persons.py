"""Segment x item distribution scores with a person-cluster paired bootstrap and a bias correction.

    import simengine.evidence as ev
    r = ev.evaluate_distribution_persons(people, ["age", "income"], ["q1", "q2"],
                                         {"T1": t1_cells, "llm": {"q1": "llm_q1", "q2": "llm_q2"}}, B=1000)
    r["arms"]["llm"]["point"], r["arms"]["llm"]["ci_bc"], r["diffs"][0]

`evaluate(task="distribution")` takes ready cells and can only resample cells or segments. Here the input is the
holdout *people*: one row per person, a segment column per scheme, the person's answer (an option label) per item.
Truth for a cell (scheme, segment, item) is the share of the cell's people choosing each option; the bootstrap
draws people with replacement (clustered by ``id_col``), rebuilds every cell from the same draw for every arm
(paired) and scores each arm as the mean metric over cells. Several schemes over the same people (age, income,
household) are scored jointly and per scheme; a person counts once per scheme.

Arms (``pred_cols``):
- fixed cell predictions, fit on training people: ``{(scheme, segment, item): vector}`` or a DataFrame with columns
  ``scheme`` (optional with one scheme), ``segment``, ``item`` and ``pred`` (or ``pred_<arm>``, or a single
  ``pred_<method>`` column, e.g. the output of ``sim.distribution_frame``). Vectors follow ``options[item]``.
- person-level predictions: ``{item: column}``; each cell holds a probability vector (list / array / JSON) or an
  option label (one-hot). The arm's cell prediction is the mean over the cell's people, resampled with them.

Bias: 1-TVD between a small sample's empirical distribution and any prediction is biased *down* (sampling noise
adds distance), and the bootstrap repeats that noise, so percentile intervals sit below the point and can exclude
it. The report gives the percentile CI, the basic CI (2*point - quantiles) and a bias-centred percentile CI
(quantiles - bias), plus the bias-corrected point 2*point - mean(boot). ``hold_truth=True`` instead keeps truth at
the full-sample distributions and resamples only person-level predictions (fixed arms then get zero width).
"""
from __future__ import annotations

import json
import warnings
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["evaluate_distribution_persons", "DIST_METRICS"]


def _tvd(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(t - p).sum(axis=-1)


def _w1(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    k = t.shape[-1]
    return np.abs(np.cumsum(t, axis=-1) - np.cumsum(p, axis=-1))[..., :-1].sum(axis=-1) / max(1, k - 1)


# name -> (vectorised distance-or-score over the last axis, higher_is_better)
DIST_METRICS: dict[str, tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], bool]] = {
    "1-tvd": (lambda t, p: 1.0 - _tvd(t, p), True),
    "tvd": (_tvd, False),
    "1-w1": (lambda t, p: 1.0 - _w1(t, p), True),        # ordinal options, normalised 1-Wasserstein
    "w1": (_w1, False),
}
_ALIASES = {"one_minus_tvd": "1-tvd", "bir_eksi_tvd": "1-tvd", "1-wd": "1-w1", "wd": "w1"}


def _vec(v: Any, opts: list, item: str) -> np.ndarray | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, str) and v.strip().startswith("["):
        v = json.loads(v)
    if isinstance(v, Mapping):
        v = [float(v.get(o, v.get(str(o), 0.0))) for o in opts]
    if isinstance(v, (list, tuple, np.ndarray)):
        a = np.asarray(v, dtype=float)
        if a.shape != (len(opts),):
            raise ValueError(f"evaluate_distribution_persons: item {item!r}: vector of length {a.size}, "
                             f"{len(opts)} options {opts}")
        s = a.sum()
        return a / s if s > 0 else None
    if v in opts:                                       # a label -> one-hot
        a = np.zeros(len(opts))
        a[opts.index(v)] = 1.0
        return a
    raise ValueError(f"evaluate_distribution_persons: item {item!r}: {v!r} is not an option {opts}")


def _fixed_table(spec, arm: str, schemes: list[str]) -> dict:
    if isinstance(spec, pd.DataFrame):
        col = "pred" if "pred" in spec.columns else f"pred_{arm}"
        only = [c for c in spec.columns if str(c).startswith("pred_")]
        col = only[0] if col not in spec.columns and len(only) == 1 else col     # one pred_<method> column
        if col not in spec.columns or not {"segment", "item"} <= set(spec.columns):
            raise ValueError(f"evaluate_distribution_persons: arm {arm!r}: need columns segment, item and pred "
                             f"(or pred_{arm})")
        if "scheme" not in spec.columns and len(schemes) > 1:
            raise ValueError(f"evaluate_distribution_persons: arm {arm!r}: several schemes, add a 'scheme' column")
        sc = spec["scheme"] if "scheme" in spec.columns else [schemes[0]] * len(spec)
        return {(str(s), str(g), str(i)): v for s, g, i, v in zip(sc, spec["segment"], spec["item"], spec[col])}
    out = {}
    for k, v in spec.items():
        if len(k) == 2 and len(schemes) == 1:                         # (segment, item) with a single scheme
            k = (schemes[0], *k)
        if len(k) != 3:
            raise ValueError(f"evaluate_distribution_persons: arm {arm!r}: key {k!r}, use (scheme, segment, item)")
        out[(str(k[0]), str(k[1]), str(k[2]))] = v
    return out


def _q(x: np.ndarray, alpha: float) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if not len(x):
        return (float("nan"), float("nan"))
    return float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2))


def _intervals(point: float, boot: np.ndarray, alpha: float) -> dict:
    lo, hi = _q(boot, alpha)
    m = float(np.nanmean(boot)) if np.isfinite(boot).any() else float("nan")
    bias = m - point
    return {"point": point, "boot_mean": m, "bias": bias, "point_bc": point - bias,
            "ci": [lo, hi], "ci_basic": [2 * point - hi, 2 * point - lo], "ci_bc": [lo - bias, hi - bias]}


def evaluate_distribution_persons(person_frame: pd.DataFrame, segment_cols: Sequence[str] | Mapping[str, str],
                                  items: Sequence[str], pred_cols: Mapping[str, Any], *, B: int = 1000,
                                  seed: int = 0, metric: str | Callable = "1-tvd", id_col: str | None = "unit",
                                  options: Mapping[str, Sequence] | None = None, hold_truth: bool = False,
                                  alpha: float = 0.05, min_n: int = 1,
                                  pairs: Sequence[tuple[str, str]] | None = None) -> dict:
    """Person-cluster paired bootstrap of segment x item distribution scores (see the module docstring).

    person_frame: one row per person (rows sharing ``id_col`` are resampled together; None or a missing column ->
    each row is a person). segment_cols: segment columns (one per scheme), or {scheme name: column}. items: answer
    columns (option labels; NaN = no answer). pred_cols: {arm: fixed cell table or {item: person-level column}}.
    metric: "1-tvd" (default), "tvd", "1-w1", "w1", or a callable (truth (k,), pred (k,)) -> float, read as
    higher-is-better. options: {item: ordered options} (default the observed answers, numeric order when they are
    numbers, else alphabetical, which "w1" warns about; fixed vectors must follow this order). Fixed dict keys are
    (scheme, segment, item), or (segment, item) with one scheme.
    min_n: cells with fewer answering people in the full sample are dropped. pairs: (a, b) differences a - b
    (default every pair in arm order).

    Returns a JSON-safe dict: metric, higher_is_better, B, seed, alpha, hold_truth, n_persons, n_cells, schemes,
    arms {arm: point, boot_mean, bias, point_bc, ci, ci_basic, ci_bc, by_scheme {scheme: point}}, diffs [{a, b,
    point, ..., p_two_sided}], cells [{scheme, segment, item, n, <arm>: score}].
    """
    if isinstance(metric, str):
        name = _ALIASES.get(metric, metric)
        if name not in DIST_METRICS:
            raise ValueError(f"evaluate_distribution_persons: metric {metric!r}; one of {sorted(DIST_METRICS)} "
                             "or a callable")
        fn, higher = DIST_METRICS[name]
    else:
        name, higher = getattr(metric, "__name__", "custom"), True
        fn = np.vectorize(lambda t, p: float(metric(t, p)), signature="(k),(k)->()")
    schemes = dict(segment_cols) if isinstance(segment_cols, Mapping) else {str(c): c for c in segment_cols}
    items = [str(i) for i in items]
    df = person_frame.reset_index(drop=True)
    for c in list(schemes.values()) + items:
        if c not in df.columns:
            raise ValueError(f"evaluate_distribution_persons: column {c!r} not in person_frame")
    if not pred_cols:
        raise ValueError("evaluate_distribution_persons: no arms in pred_cols")
    arms = [str(a) for a in pred_cols]

    # people -> clusters
    if id_col is not None and id_col in df.columns:
        codes, uniq = pd.factorize(df[id_col])
    else:
        codes, uniq = np.arange(len(df)), np.arange(len(df))
    P, n_cl = len(df), len(uniq)
    opts, alpha_order = {}, []
    for i in items:
        given = (options or {}).get(i)
        seen = df[i].dropna().unique().tolist()
        if given is not None:
            opts[i] = list(given)
        elif seen and all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                          for v in seen):
            opts[i] = sorted(seen)                                    # numeric answers: numeric order (1, 2, 10)
        else:
            opts[i] = sorted(seen, key=str)
            alpha_order.append(i)
    if alpha_order and name in ("1-w1", "w1"):
        warnings.warn(f"evaluate_distribution_persons: metric {name!r} uses option order, items {alpha_order} have "
                      "label answers sorted alphabetically; pass options={item: [ordered labels]}", stacklevel=2)

    # truth one-hot (P, k) per item; segment membership (P, S) per scheme
    onehot = {}
    for i in items:
        M = np.zeros((P, len(opts[i])))
        for r, v in enumerate(df[i]):
            if v is not None and v == v:
                if v not in opts[i]:
                    raise ValueError(f"evaluate_distribution_persons: item {i!r}: answer {v!r} not in {opts[i]}")
                M[r, opts[i].index(v)] = 1.0
        onehot[i] = M
    answered = {i: onehot[i].sum(axis=1) for i in items}
    member = {}
    for s, col in schemes.items():
        seg = df[col].map(lambda v: None if v is None or v != v else str(v))
        labels = sorted(seg.dropna().unique().tolist())
        member[s] = (labels, np.stack([(seg == g).to_numpy(float) for g in labels], axis=1) if labels
                     else np.zeros((P, 0)))

    # arm specs
    person_level, fixed = {}, {}
    for a, spec in pred_cols.items():
        a = str(a)
        if isinstance(spec, Mapping) and spec and all(not isinstance(k, tuple) for k in spec):
            mats = {}
            for i in items:
                if i not in spec:
                    raise ValueError(f"evaluate_distribution_persons: arm {a!r} has no column for item {i!r}")
                V, ok = np.zeros((P, len(opts[i]))), np.zeros(P)
                for r, v in enumerate(df[spec[i]]):
                    x = _vec(v, opts[i], i)
                    if x is not None:
                        V[r], ok[r] = x, 1.0
                mats[i] = (V, ok)
            person_level[a] = mats
        else:
            fixed[a] = _fixed_table(spec, a, list(schemes))

    # cell list (full-sample n >= min_n) and fixed prediction arrays
    cells = []
    for s, (labels, S) in member.items():
        for i in items:
            n = S.T @ onehot[i].sum(axis=1)
            for g_ix, g in enumerate(labels):
                if n[g_ix] >= max(1, min_n):
                    cells.append((s, g_ix, g, i, int(n[g_ix])))
    if not cells:
        raise ValueError("evaluate_distribution_persons: no cell has min_n answering people")
    fixed_vec = {}
    for a, tab in fixed.items():
        for s, _, g, i, _ in cells:
            if (s, g, i) not in tab:
                raise ValueError(f"evaluate_distribution_persons: arm {a!r} has no prediction for cell {(s, g, i)}")
            fixed_vec[(a, s, g, i)] = _vec(tab[(s, g, i)], opts[i], i)
            if fixed_vec[(a, s, g, i)] is None:
                raise ValueError(f"evaluate_distribution_persons: arm {a!r}: empty prediction for cell {(s, g, i)}")

    # per (scheme, item) design matrices, so one replicate chunk is a few (R, P) @ (P, S*k) products
    design = {}
    for s, i in {(cl[0], cl[3]) for cl in cells}:
        S = member[s][1] * answered[i][:, None]                       # (P, S) the segment's answering people
        k = len(opts[i])
        X = (S[:, :, None] * onehot[i][:, None, :]).reshape(P, -1)
        pl = {}
        for a, mats in person_level.items():
            V, ok = mats[i]
            pl[a] = ((S[:, :, None] * V[:, None, :]).reshape(P, -1), S * ok[:, None])
        design[(s, i)] = (S.shape[1], k, X, pl)

    def _share(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)

    def scores(W: np.ndarray, Wt: np.ndarray) -> np.ndarray:
        """W, Wt: (R, P) person weights for predictions / truth -> (R, n_arms, n_cells) metric values."""
        R = W.shape[0]
        out = np.full((R, len(arms), len(cells)), np.nan)
        cache = {}
        for c, (s, g_ix, g, i, _) in enumerate(cells):
            if (s, i) not in cache:
                n_s, k, X, pl = design[(s, i)]
                cnt = (Wt @ X).reshape(R, n_s, k)
                T = _share(cnt, cnt.sum(axis=2, keepdims=True))
                prs = {a: _share((W @ Xv).reshape(R, n_s, k), (W @ ok)[:, :, None]) for a, (Xv, ok) in pl.items()}
                cache[(s, i)] = (T, prs)
            T_all, prs = cache[(s, i)]
            T = T_all[:, g_ix, :]
            for a_ix, a in enumerate(arms):
                Pr = np.broadcast_to(fixed_vec[(a, s, g, i)], T.shape) if a in fixed else prs[a][:, g_ix, :]
                out[:, a_ix, c] = fn(T, Pr)
        return out

    w_full = np.ones((1, P))
    base = scores(w_full, w_full)[0]                                  # (n_arms, n_cells)
    rng = np.random.default_rng(seed)
    boot = np.empty((B, len(arms), len(cells)))
    chunk = max(1, min(B, 200))
    for b0 in range(0, B, chunk):
        r = min(chunk, B - b0)
        draw = rng.integers(0, n_cl, size=(r, n_cl))
        cw = np.stack([np.bincount(d, minlength=n_cl) for d in draw]).astype(float)
        W = cw[:, codes]
        boot[b0:b0 + r] = scores(W, np.broadcast_to(w_full, W.shape) if hold_truth else W)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)               # all-NaN slices -> NaN
        pt = np.nanmean(base, axis=1)                                 # (n_arms,)
        bt = np.nanmean(boot, axis=2)                                 # (B, n_arms)
    res_arms = {}
    for a_ix, a in enumerate(arms):
        r = _intervals(float(pt[a_ix]), bt[:, a_ix], alpha)
        r["by_scheme"] = {s: float(np.nanmean([base[a_ix, c] for c, cl in enumerate(cells) if cl[0] == s]))
                          for s in schemes if any(cl[0] == s for cl in cells)}
        res_arms[a] = r
    diffs = []
    for a, b in (pairs if pairs is not None else [(arms[x], arms[y]) for x in range(len(arms))
                                                 for y in range(x + 1, len(arms))]):
        ia, ib = arms.index(a), arms.index(b)
        d = bt[:, ia] - bt[:, ib]
        r = _intervals(float(pt[ia] - pt[ib]), d, alpha)
        f = d[np.isfinite(d)]
        r["p_two_sided"] = float(min(1.0, 2 * min((f <= 0).mean(), (f >= 0).mean()))) if len(f) else None
        diffs.append({"a": a, "b": b, **r})
    cell_rows = [{"scheme": s, "segment": g, "item": i, "n": n, **{a: float(base[a_ix, c]) for a_ix, a in
                                                                  enumerate(arms)}}
                 for c, (s, _, g, i, n) in enumerate(cells)]
    note = None
    if hold_truth and fixed:
        note = "hold_truth: truth fixed, fixed-cell arms do not move (zero-width CIs): " + ", ".join(sorted(fixed))
    return {"metric": name, "higher_is_better": higher, "B": int(B), "seed": int(seed), "alpha": alpha,
            "hold_truth": bool(hold_truth), "n_persons": int(n_cl), "n_cells": len(cells), "schemes": list(schemes),
            "arms": res_arms, "diffs": diffs, "cells": cell_rows, "note": note}
