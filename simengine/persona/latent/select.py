"""Validation-only hyperparameter choice (rule R-HP): every config is scored on validation cells only.

    cfgs = grid({"layer": [0.4, 0.55, 0.7], "lr": [1e-4, 3e-4]})
    best, table = choose(cfgs, lambda c: val_metrics_g1(fit_and_predict(c), val_rows), cost_key="cost")
"""
from __future__ import annotations

import itertools
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


def grid(spec: Mapping[str, Sequence]) -> list[dict]:
    """Full product of the spec's value lists, in spec order."""
    keys = list(spec)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(spec[k] for k in keys))]


def kisi_auc(p, y, person) -> float:
    """Within-person AUC (talp73 interpretation 4): per person with both classes, the share of (bought, not
    bought) pairs ranked right, ties 0.5; mean over persons. NaN if no person has both classes."""
    d = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, int), "u": np.asarray(person)})
    d["r"] = d.groupby("u")["p"].rank(method="average")
    g = pd.DataFrame({"n": d.groupby("u").size(), "n1": d.groupby("u")["y"].sum(),
                      "r1": d[d.y == 1].groupby("u")["r"].sum()}).fillna(0.0)
    g["n0"] = g["n"] - g["n1"]
    g = g[(g.n1 > 0) & (g.n0 > 0)]
    if g.empty:
        return float("nan")
    auc = (g.r1 - g.n1 * (g.n1 + 1) / 2) / (g.n1 * g.n0)
    return float(auc.mean())


def acc_1a(p, y) -> float:
    """1a(a) accuracy: decision p >= 0.5 on every row."""
    return float(np.mean((np.asarray(p, float) >= 0.5) == (np.asarray(y) == 1)))


def val_metrics_g1(p, rows: pd.DataFrame) -> dict:
    """kisi_auc, 1a_dogruluk_a and log loss of predictions p on validation rows (uid, y)."""
    y = rows["y"].to_numpy()
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"kisi_auc": kisi_auc(p, y, rows["uid"].to_numpy()), "1a_dogruluk_a": acc_1a(p, y),
            "logloss": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))}


def choose(configs: Sequence[dict], score_fn: Callable[[dict], dict], *, primary: str = "kisi_auc",
           tiebreak: Sequence[str] = ("1a_dogruluk_a",), cost_key: str | None = None, out_path=None
           ) -> tuple[dict, pd.DataFrame]:
    """Score every config with score_fn (validation only) and pick the best by `primary`, then `tiebreak` (higher is
    better), then the lower cost (config[cost_key]), then the earlier config. The table is written to out_path
    (parquet) when given."""
    rows = []
    for i, c in enumerate(configs):
        s = score_fn(c)
        rows.append({"i": i, **{f"cfg_{k}": v for k, v in c.items()}, **s})
    t = pd.DataFrame(rows)
    keys = [primary] + [k for k in tiebreak if k in t]
    asc = [False] * len(keys)
    order = t.copy()
    if cost_key is not None:
        order["_cost"] = [c.get(cost_key, 0) for c in configs]
        keys, asc = keys + ["_cost"], asc + [True]
    order = order.sort_values(keys + ["i"], ascending=asc + [True], na_position="last")
    best = dict(configs[int(order.iloc[0]["i"])])
    t["chosen"] = t["i"] == int(order.iloc[0]["i"])
    if out_path is not None:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        t.astype({c: str for c in t.columns if c.startswith("cfg_")}).to_parquet(out_path, index=False)
    return best, t


__all__ = ["grid", "choose", "kisi_auc", "acc_1a", "val_metrics_g1"]
