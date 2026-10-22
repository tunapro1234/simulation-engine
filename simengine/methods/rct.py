"""Randomised-experiment (RCT / A-B / many-labs) condition effects and the "same test in another market" baseline.

Task shape: one row per person with the condition they were randomised to (``cond``; one label is the control), the
outcome ``y`` and optionally the block they were randomised within (country, site, week) and segment columns
(``by``). The anchor every forecast is scored against is the condition effect: within each block
``mean(y | cond) - mean(y | control)``, averaged over the blocks (size or equal weights), per segment when ``by`` is
given, with the Neyman standard error ``sqrt(sum_b w_b^2 (s2_cond/n_cond + s2_ctrl/n_ctrl)) / sum_b w_b``.

``rct_transfer`` is the strongest data baseline when the same conditions were tested elsewhere: the effects
estimated on the training blocks are the prediction for the held-out blocks, set next to the held-out truth.

Measured (G1 rehearsal, climate many-labs, 11 messages vs control, policy support; 32 training countries -> 31 test
countries, 29,662 test people): transfer Spearman over the 11 conditions 0.845 [0.43, 1.00] against a sampling-noise
ceiling of 0.908; a single training country transferred alone: median 0.28, quartiles [0.06, 0.48]. The LLM arms
(direct forecast 0.073, persona panels -0.127 / 0.227) did not reach it. Usage::

    import simengine.methods as M
    eff = M.rct_effects(df, "condition", "Control", "policy", block="country", by="ideology")
    tr = M.rct_transfer(df, "condition", "Control", "policy", block="country", test_blocks=test_countries)
    tr.attrs["spearman"], tr[["condition", "pred", "truth"]]
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

Z95 = 1.959963984540054
_BLOCK = "__block__"


def _cols(by) -> list[str]:
    if by is None:
        return []
    return [by] if isinstance(by, str) else list(by)


def rct_effects(df: pd.DataFrame, cond: str, control, y: str, *, block: str | None = None, by=None,
                weights=None) -> pd.DataFrame:
    """Condition (x segment) effects against ``control``, within blocks and averaged over them.

    - ``cond``: condition column; ``control``: its control label (must occur). Every other label gets a row.
    - ``block``: randomisation block column (``None`` = one block, i.e. a plain difference in means).
    - ``by``: segment column(s); effects are computed within each segment x block cell.
    - ``weights``: ``None`` / ``"size"`` (block weight = people in the two arms of that block and segment),
      ``"equal"`` (every block counts the same; use it when one block dominates the sample), or a mapping / Series
      block label -> weight (missing labels raise).

    A block counts for a condition only when it has both that condition and the control (in that segment). Rows
    with a missing ``y``, ``cond``, ``block`` or segment are dropped. A cell with one person borrows the arm's pooled
    within-block variance for the SE (``se`` is NaN when that arm has no cell with two people). Returns one row per (segment..., condition): ``effect``, ``se``,
    ``ci_low`` / ``ci_high`` (normal 95%), ``mean_cond`` / ``mean_control`` (block-weighted), ``n_blocks``,
    ``n_cond``, ``n_control``; ``.attrs["blocks"]`` lists the blocks that contributed.
    """
    bys = _cols(by)
    need = [cond, y] + bys + ([block] if block else [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in df: {missing}")
    if block in bys or cond in bys:
        raise ValueError("by must not contain the block or condition column")
    d = df[need].copy()
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d.dropna()
    blk = block or _BLOCK
    if block is None:
        d[_BLOCK] = 0
    if not (d[cond] == control).any():
        raise ValueError(f"control label {control!r} not found in {cond!r}")
    if isinstance(weights, str) and weights not in ("size", "equal"):
        raise ValueError("weights must be None, 'size', 'equal' or a mapping block -> weight")

    st = d.groupby(bys + [blk, cond], observed=True)[y].agg(mean="mean", var="var", n="count").reset_index()
    # cells with one person: pooled within-block variance of that arm (in that segment)
    arm = bys + [cond]
    st["_ss"] = np.where(st["n"] >= 2, st["var"] * (st["n"] - 1), 0.0)
    st["_df"] = np.where(st["n"] >= 2, st["n"] - 1, 0)
    pooled = st.groupby(arm, observed=True)[["_ss", "_df"]].transform("sum")
    st["var"] = st["var"].where(st["n"] >= 2, pooled["_ss"] / pooled["_df"].replace(0, np.nan))

    keys = bys + [blk]
    ctrl = st.loc[st[cond] == control, keys + ["mean", "var", "n"]].rename(
        columns={"mean": "m0", "var": "v0", "n": "n0"})
    m = st.loc[st[cond] != control, keys + [cond, "mean", "var", "n"]].merge(ctrl, on=keys, how="inner")
    out_cols = bys + [cond, "effect", "se", "ci_low", "ci_high", "mean_cond", "mean_control", "n_blocks",
                      "n_cond", "n_control"]
    if m.empty:
        out = pd.DataFrame(columns=out_cols)
        out.attrs["blocks"] = []
        return out
    m["diff"] = m["mean"] - m["m0"]
    m["v"] = m["var"] / m["n"] + m["v0"] / m["n0"]
    if weights is None or weights == "size":
        m["w"] = (m["n"] + m["n0"]).astype(float)
    elif weights == "equal":
        m["w"] = 1.0
    else:
        wmap = pd.Series(weights, dtype=float)
        w = m[blk].map(wmap)
        if w.isna().any():
            raise KeyError(f"weights has no value for blocks {sorted(m.loc[w.isna(), blk].unique())[:5]}")
        m["w"] = w
    m["wd"], m["wv"] = m["w"] * m["diff"], m["w"] ** 2 * m["v"]
    m["wm"], m["wm0"] = m["w"] * m["mean"], m["w"] * m["m0"]
    g = m.groupby(arm, observed=True)
    out = g[["w", "wd", "wv", "wm", "wm0", "n", "n0"]].sum()
    out["n_blocks"] = g[blk].nunique()
    W = out["w"].replace(0, np.nan)
    out["effect"] = out["wd"] / W
    out["se"] = np.sqrt(out["wv"]) / W
    out.loc[m["wv"].isna().groupby([m[c] for c in arm], observed=True).any(), "se"] = np.nan   # sum() would skip a NaN variance
    out["ci_low"], out["ci_high"] = out["effect"] - Z95 * out["se"], out["effect"] + Z95 * out["se"]
    out["mean_cond"], out["mean_control"] = out["wm"] / W, out["wm0"] / W
    out = out.rename(columns={"n": "n_cond", "n0": "n_control"}).reset_index()
    out = out[out_cols].astype({"n_blocks": int, "n_cond": int, "n_control": int})
    out.attrs["blocks"] = [] if block is None else sorted(pd.unique(m[blk]).tolist(), key=str)
    return out


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return float("nan")
    return float(a[ok].rank().corr(b[ok].rank()))


def rct_transfer(df: pd.DataFrame, cond: str, control, y: str, *, block: str, test_blocks: Sequence,
                 train_blocks: Sequence | None = None, by=None, weights=None, per_block: bool = False) -> pd.DataFrame:
    """"Same test in another market": effects on ``train_blocks`` as the prediction for ``test_blocks``.

    ``train_blocks=None`` = every block not in ``test_blocks`` (overlap raises). ``per_block=False``: the truth is
    the effect pooled over the test blocks (``rct_effects`` with the same ``weights``); ``True``: one truth row per
    test block (the prediction is the same pooled training effect for every block). Returns the truth rows
    (segment..., [block], condition) with ``pred``, ``pred_se``, ``truth``, ``truth_se``, ``n_blocks_train`` and
    ``n_blocks_test`` (``pred`` is NaN for a condition / segment never seen with its control in training).
    ``.attrs``: ``spearman`` (rank correlation of pred vs truth over the rows; with ``per_block`` the mean of the
    per-block correlations, which are in the dict ``spearman_by_block``) and ``mae``.
    """
    if block not in df.columns:
        raise KeyError(f"block column {block!r} not in df")
    test_blocks = list(test_blocks)
    if not test_blocks:
        raise ValueError("test_blocks is empty")
    present = set(pd.unique(df[block].dropna()))
    for name, blocks in (("test_blocks", test_blocks), ("train_blocks", train_blocks or [])):
        unknown = [b for b in blocks if b not in present]
        if unknown:     # typically str labels for an int block column (or the reverse)
            raise KeyError(f"{name} not in df[{block!r}]: {sorted(map(repr, unknown))[:5]} (block labels look like "
                           f"{sorted(map(repr, list(present)[:3]))}; check the label type)")
    if train_blocks is None:
        train_blocks = [b for b in pd.unique(df[block].dropna()) if b not in set(test_blocks)]
    else:
        train_blocks = list(train_blocks)
        both = set(train_blocks) & set(test_blocks)
        if both:
            raise ValueError(f"blocks in both train and test: {sorted(both, key=str)[:5]}")
    bys = _cols(by)
    tr = rct_effects(df[df[block].isin(train_blocks)], cond, control, y, block=block, by=bys, weights=weights)
    te_df = df[df[block].isin(test_blocks)]
    if per_block:
        te = rct_effects(te_df, cond, control, y, by=bys + [block])     # one block per cell: plain difference
    else:
        te = rct_effects(te_df, cond, control, y, block=block, by=bys, weights=weights)
    keys = bys + [cond]
    pred = tr[keys + ["effect", "se", "n_blocks"]].rename(
        columns={"effect": "pred", "se": "pred_se", "n_blocks": "n_blocks_train"})
    lead = bys + ([block] if per_block else []) + [cond]
    truth = te[lead + ["effect", "se", "n_blocks"]].rename(
        columns={"effect": "truth", "se": "truth_se", "n_blocks": "n_blocks_test"})
    out = truth.merge(pred, on=keys, how="left")
    out = out[lead + ["pred", "pred_se", "truth", "truth_se", "n_blocks_train", "n_blocks_test"]]
    out["n_blocks_train"] = out["n_blocks_train"].fillna(0).astype(int)
    out.attrs["mae"] = float((out["pred"] - out["truth"]).abs().mean())
    if per_block:
        by_block = out.groupby(block)[["pred", "truth"]].apply(lambda t: _rank_corr(t["pred"], t["truth"]))
        out.attrs["spearman_by_block"] = {b: float(v) for b, v in by_block.items()}   # dict: a Series in attrs
        out.attrs["spearman"] = float(by_block.mean())
    else:
        out.attrs["spearman"] = _rank_corr(out["pred"], out["truth"])
    out.attrs["train_blocks"], out.attrs["test_blocks"] = list(train_blocks), test_blocks
    return out
