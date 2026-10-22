"""Repeated surveys: past waves -> next wave's segment x question answer distributions (LLM-free ladder).

Task shape (rehearsal D6, a CES-type yearly survey): person-level answers for waves 2006..2024, predict the
2025 distribution of each segment x question, often across a regime switch (the president's party changed).
The D6 team wrote this ladder by hand (~17 min); a referee then showed that the "same regime, last 4 waves"
rung beat the preregistered method on dev (0.0990 vs 0.1007 TVD), and nothing in the library had built it.

Input: a long frame, one row per respondent x question, with a wave column, a segment column, an item column and
an answer column (``value``); optional ``weight`` (survey weight), regime and subgroup columns. A wide frame
(one column per question) works with ``item=None, value=[<question columns>]``. Answers outside ``categories``
(NaN, "skipped") are ignored. ``categories`` is one list for every item or a dict {item: list}.

Every predictor uses only waves strictly before ``target`` (default: one after the last wave, i.e. the future
wave), and returns the distribution frame `evaluate(task="distribution")` takes: one row per segment x item with
``unit`` ("<segment>|<item>"), ``truth`` (the target wave's distribution when it is in ``df``, else NaN),
``pred_<method>``, ``segment``, ``item``, ``group`` (= item), ``wave`` (the target), ``n`` (valid answers in the
target cell, for n-weighted national totals) and ``n_train_<method>`` (valid answers behind the prediction).

    from simengine.methods.waves import (wave_persistence, pooled_shrunk, same_regime, subgroup_mixture,
                                         transition_backtest, merge_frames)
    cols = dict(wave="year", segment="seg", item="q", value="answer", categories=CATS, weight="w")
    t0 = wave_persistence(df, **cols, target=2025)                    # T0: copy last wave
    t2 = pooled_shrunk(df, **cols, target=2025, k=4, m=20)            # T2: last 4 waves, shrunk to population
    t4b = same_regime(df, **cols, target=2025, regime=PRES_PARTY, k=4)  # T4b: last 4 waves with an R president
    mx = subgroup_mixture(df, **cols, target=2025, subgroup="pid3", k=1,
                          swap={"Rep": "Dem", "Dem": "Rep"}, swap_items=["approve_pres"])
    frame = merge_frames([t0, t2, t4b, mx])                           # -> se.evaluate(frame, task="distribution")
    bt = transition_backtest(df, waves=[2009, 2017, 2021], methods={
        "t0": (wave_persistence, {}), "t4b": (same_regime, {"regime": PRES_PARTY, "k": 4})}, **cols)
    bt.by_wave            # wave x method mean 1-TVD: the year-to-year spread the single dev year hides

``mixture_reweight`` and ``role_swap`` are the pure pieces under ``subgroup_mixture`` (an MRP-like step: a
segment's distribution = its subgroup composition x subgroup-conditional distributions).

Cautions (D6): pick and report on several past transition years (``transition_backtest``), not one dev year: a
cluster bootstrap over segments in one year does not cover year-to-year (regime) uncertainty. Put the same-regime
rung on the ladder whenever the target crosses a regime switch.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["wave_persistence", "pooled_shrunk", "same_regime", "subgroup_mixture", "mixture_reweight",
           "role_swap", "transition_backtest", "BacktestResult", "merge_frames", "one_minus_tvd"]


# ---------------------------------------------------------------- helpers

def one_minus_tvd(p, q) -> float:
    """1 - total variation distance between two distributions over the same options (NaN if either is missing)."""
    if p is None or q is None or (np.isscalar(p) and pd.isna(p)) or (np.isscalar(q) and pd.isna(q)):
        return float("nan")
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(1.0 - 0.5 * np.abs(p / p.sum() - q / q.sum()).sum())


def _long(df: pd.DataFrame, item, value) -> tuple[pd.DataFrame, str, str]:
    """Wide (item=None, value=[question columns]) -> long with columns '_item', '_value'; long passes through."""
    if item is None:
        if isinstance(value, str):
            raise ValueError("item=None needs value=[question columns] (wide frame)")
        cols = list(value)
        keep = [c for c in df.columns if c not in cols]
        out = df.melt(id_vars=keep, value_vars=cols, var_name="_item", value_name="_value")
        return out, "_item", "_value"
    return df, item, value


def _cats(categories, it) -> list:
    if isinstance(categories, Mapping):
        if it not in categories:
            raise KeyError(f"no categories for item {it!r}")
        return list(categories[it])
    return list(categories)


def _waves_before(df: pd.DataFrame, wave: str, target) -> list:
    ws = sorted(pd.unique(df[wave]))
    return [w for w in ws if w < target]


def _default_target(df: pd.DataFrame, wave: str):
    """One after the last wave for integer waves; otherwise the caller must say."""
    last = max(pd.unique(df[wave]))
    if isinstance(last, (int, np.integer)):
        return last + 1
    raise ValueError("target= is required when waves are not integers")


def _counts(d: pd.DataFrame, keys: list[str], item: str, value: str, categories, weight: str | None,
            items: Sequence) -> dict:
    """{item: (counts, n_eff, n_rows)} over valid answers: counts is a DataFrame (index = keys, columns =
    categories; one row for keys=[]), n_eff the row count or Kish's (sum w)^2 / sum w^2 with weights, n_rows the
    plain row count."""
    out = {}
    for it in items:
        cats = _cats(categories, it)
        di = d[(d[item] == it) & d[value].isin(cats)]
        w = di[weight].astype(float) if weight else pd.Series(1.0, index=di.index)
        if not keys:
            c = w.groupby(di[value]).sum().reindex(cats).fillna(0.0).to_frame().T
            s2 = float((w ** 2).sum())
            out[it] = (c, pd.Series([float(w.sum()) ** 2 / s2 if s2 > 0 else 0.0]), pd.Series([len(di)]))
            continue
        if not len(di):
            idx = pd.MultiIndex.from_tuples([], names=keys) if len(keys) > 1 else pd.Index([], name=keys[0])
            out[it] = (pd.DataFrame(columns=cats, index=idx, dtype=float), pd.Series(dtype=float, index=idx),
                       pd.Series(dtype=float, index=idx))
            continue
        by = [di[k] for k in keys]
        c = w.groupby(by + [di[value]]).sum().unstack(value).reindex(columns=cats).fillna(0.0)
        s1 = w.groupby(by).sum()
        s2 = (w ** 2).groupby(by).sum()
        out[it] = (c, (s1 ** 2 / s2.where(s2 > 0)).fillna(0.0), w.groupby(by).size())
    return out


def _dist(row) -> np.ndarray | None:
    a = np.asarray(row, float)
    s = a.sum()
    return a / s if s > 0 else None


def _cells(df, wave, segment, item, value, categories, target, train, segments, items):
    """Segments and items to predict: given, else those of the target wave when present, else of the window."""
    tgt = df[df[wave] == target]
    src = tgt if len(tgt) else train
    if items is None:
        items = list(pd.unique(src[item]))
    if segments is None:
        segments = list(pd.unique(src[segment].dropna()))
    return list(segments), list(items)


def _frame(preds: dict, ntrain: dict, df, wave, segment, item, value, categories, weight, target,
           segments, items, method) -> pd.DataFrame:
    """Distribution frame rows; truth and n from the target wave when it is in df."""
    tgt = df[df[wave] == target]
    tc = _counts(tgt, [segment], item, value, categories, weight, items) if len(tgt) else {}
    rows = []
    for it in items:
        for s in segments:
            truth, n = np.nan, 0
            if it in tc:
                c, _, nr = tc[it]
                if s in c.index:
                    t = _dist(c.loc[s].to_numpy())
                    if t is not None:
                        truth, n = list(t), int(nr.get(s, 0))
            rows.append({"unit": f"{s}|{it}", "truth": truth, f"pred_{method}": list(preds[(s, it)]),
                         "segment": s, "item": it, "group": it, "wave": target, "n": n,
                         f"n_train_{method}": float(ntrain.get((s, it), 0.0))})
    return pd.DataFrame(rows)


def _prepare(df, wave, segment, item, value, target):
    df, item, value = _long(df, item, value)
    if target is None:
        target = _default_target(df, wave)
    return df, item, value, target


def _shrunk_window(train, segment, item, value, categories, weight, segments, items, m, prior):
    """Per (segment, item): (n_s own + m prior) / (n_s + m) over the window; returns preds, n_train."""
    seg_c = _counts(train, [segment], item, value, categories, weight, items)
    pop_c = _counts(train, [], item, value, categories, weight, items)
    preds, ntrain = {}, {}
    for it in items:
        k = len(_cats(categories, it))
        if isinstance(prior, Mapping):
            pr = np.asarray(prior[it], float)
            pr = pr / pr.sum()
        elif prior is None or (isinstance(prior, str) and prior == "population"):
            pr = _dist(pop_c[it][0].iloc[0].to_numpy(float))
            pr = np.full(k, 1.0 / k) if pr is None else pr
        elif isinstance(prior, str) and prior == "uniform":
            pr = np.full(k, 1.0 / k)
        else:
            raise ValueError("prior must be 'population', 'uniform' or {item: distribution}")
        c, neff, _ = seg_c[it]
        for s in segments:
            own = _dist(c.loc[s].to_numpy()) if s in c.index else None
            ns = float(neff.get(s, 0.0)) if own is not None else 0.0
            if own is None:
                p = pr.copy()
            elif m == 0:
                p = own
            else:
                p = (ns * own + m * pr) / (ns + m)
            preds[(s, it)] = p
            ntrain[(s, it)] = ns
    return preds, ntrain


def _window(df, wave, waves):
    return df[df[wave].isin(list(waves))]


# ---------------------------------------------------------------- predictors

def wave_persistence(df: pd.DataFrame, wave: str, segment: str, item: str | None, value, categories,
                     k: int = 1, target=None, weight: str | None = None, segments: Sequence | None = None,
                     items: Sequence | None = None, method: str = "persistence") -> pd.DataFrame:
    """T0: copy the last wave's segment x item distribution; k > 1 pools the last k waves (counts summed, so
    n-weighted). A segment with no answers in the window gets the window's population distribution
    (``n_train_<method>`` = 0 marks it)."""
    df, item, value, target = _prepare(df, wave, segment, item, value, target)
    if k < 1:
        raise ValueError("k must be >= 1")
    before = _waves_before(df, wave, target)
    if not before:
        raise ValueError(f"no wave before target {target!r}")
    train = _window(df, wave, before[-k:])
    segments, items = _cells(df, wave, segment, item, value, categories, target, train, segments, items)
    preds, ntrain = _shrunk_window(train, segment, item, value, categories, weight, segments, items, 0.0, None)
    return _frame(preds, ntrain, df, wave, segment, item, value, categories, weight, target, segments, items,
                  method)


def pooled_shrunk(df: pd.DataFrame, wave: str, segment: str, item: str | None, value, categories,
                  k: int = 4, m: float = 20.0, prior: Any = "population", target=None, weight: str | None = None,
                  segments: Sequence | None = None, items: Sequence | None = None,
                  method: str = "pooled_shrunk") -> pd.DataFrame:
    """T2: each segment's distribution pooled over the last k waves, shrunk toward ``prior``:
    p = (n_s * own + m * prior) / (n_s + m), n_s = valid answers in the window (Kish effective size with
    ``weight``). prior: "population" (the window's pooled population distribution of the item), "uniform", or
    {item: distribution}. m=0 is the pooled segment empirical; m -> inf the population copy."""
    df, item, value, target = _prepare(df, wave, segment, item, value, target)
    if k < 1 or m < 0:
        raise ValueError("k must be >= 1 and m >= 0")
    before = _waves_before(df, wave, target)
    if not before:
        raise ValueError(f"no wave before target {target!r}")
    train = _window(df, wave, before[-k:])
    segments, items = _cells(df, wave, segment, item, value, categories, target, train, segments, items)
    preds, ntrain = _shrunk_window(train, segment, item, value, categories, weight, segments, items, m, prior)
    return _frame(preds, ntrain, df, wave, segment, item, value, categories, weight, target, segments, items,
                  method)


def _wave_regimes(df: pd.DataFrame, wave: str, regime) -> dict:
    """{wave: regime label} from a mapping or a per-row column (must be constant within a wave)."""
    if isinstance(regime, Mapping):
        return dict(regime)
    if regime not in df.columns:
        raise KeyError(f"regime column {regime!r} not in frame")
    g = df.groupby(wave)[regime].agg(lambda s: list(pd.unique(s.dropna())))
    bad = [w for w, v in g.items() if len(v) > 1]
    if bad:
        raise ValueError(f"regime not constant within waves {bad[:5]}")
    return {w: v[0] for w, v in g.items() if v}


def same_regime(df: pd.DataFrame, wave: str, segment: str, item: str | None, value, categories, regime,
                k: int = 4, m: float = 0.0, target=None, target_regime=None, weight: str | None = None,
                segments: Sequence | None = None, items: Sequence | None = None,
                method: str = "same_regime") -> pd.DataFrame:
    """T4b: pool the last k waves before ``target`` whose regime label equals the target's (e.g. the last 4 years
    under a president of the same party), then shrink with ``m`` toward the window population (m=0: none).

    regime: a {wave: label} mapping or a column constant within each wave. The target's label is
    ``target_regime``, else read from the mapping / the target wave's rows. With no earlier wave of that regime
    it falls back to the last k waves with a warning. The waves used are in ``frame.attrs["waves"]``.
    D6: this rung (k=4) beat the preregistered method on dev, 0.0990 vs 0.1007 TVD."""
    df, item, value, target = _prepare(df, wave, segment, item, value, target)
    if k < 1 or m < 0:
        raise ValueError("k must be >= 1 and m >= 0")
    reg = _wave_regimes(df, wave, regime)
    if target_regime is None:
        if target not in reg:
            raise ValueError(f"regime of target {target!r} unknown: pass target_regime=")
        target_regime = reg[target]
    before = _waves_before(df, wave, target)
    if not before:
        raise ValueError(f"no wave before target {target!r}")
    use = [w for w in before if reg.get(w) == target_regime][-k:]
    if not use:
        warnings.warn(f"no wave before {target!r} with regime {target_regime!r}; using the last {k} waves",
                      UserWarning, stacklevel=2)
        use = before[-k:]
    train = _window(df, wave, use)
    segments, items = _cells(df, wave, segment, item, value, categories, target, train, segments, items)
    preds, ntrain = _shrunk_window(train, segment, item, value, categories, weight, segments, items, m,
                                   "population")
    fr = _frame(preds, ntrain, df, wave, segment, item, value, categories, weight, target, segments, items, method)
    fr.attrs["waves"] = list(use)
    fr.attrs["target_regime"] = target_regime
    return fr


def mixture_reweight(cond: Mapping[Any, Sequence[float]], weights: Mapping[Any, float]) -> np.ndarray:
    """Mixture sum_g w_g * cond[g] / sum_g w_g. Subgroups with zero weight may be absent from ``cond``;
    a positive-weight subgroup missing from ``cond`` is dropped with a warning (the rest renormalised)."""
    miss = [g for g, w in weights.items() if w > 0 and g not in cond]
    if miss:
        warnings.warn(f"no conditional distribution for subgroups {miss}; dropped", UserWarning, stacklevel=2)
    use = {g: float(w) for g, w in weights.items() if w > 0 and g in cond}
    tot = sum(use.values())
    if tot <= 0:
        raise ValueError("no positive weight on a subgroup with a conditional distribution")
    acc = None
    for g, w in use.items():
        c = np.asarray(cond[g], float)
        c = c / c.sum()
        acc = w * c if acc is None else acc + w * c
    return acc / tot


def role_swap(cond: Mapping[Any, Any], mapping: Mapping[Any, Any]) -> dict:
    """Map conditional distributions across a regime switch: out[g] = cond[mapping.get(g, g)].
    E.g. presidential approval after D -> R: {"Rep": "Dem", "Dem": "Rep"} gives Republicans the Democrats' old
    in-party distribution and vice versa; independents keep theirs."""
    out = {}
    for g in cond:
        src = mapping.get(g, g)
        if src not in cond:
            raise KeyError(f"role_swap: source subgroup {src!r} (for {g!r}) not in cond")
        out[g] = cond[src]
    return out


def subgroup_mixture(df: pd.DataFrame, wave: str, segment: str, item: str | None, value, categories,
                     subgroup: str, k: int = 1, weights: Any = None, swap: Mapping | None = None,
                     swap_items: Sequence | None = None, m: float | None = None, target=None,
                     weight: str | None = None, segments: Sequence | None = None,
                     items: Sequence | None = None, method: str = "mixture") -> pd.DataFrame:
    """MRP-like mixture: segment s, item i -> sum_g w[s][g] * P(answer | g, i), with P pooled over the last k
    waves (population level, or segment x subgroup shrunk toward it with pseudo-count ``m``).

    weights: None = each segment's subgroup composition (valid answers, survey weights) in the last wave of the
        window; "target" = its composition in the target wave's rows of each item, answered or not (subgroup labels
        with the answers missing, e.g. the sample frame; in a backtest it uses the target's subgroup labels, so only
        when those are known before the answers); {subgroup: w} for every segment; or {segment: {subgroup: w}}.
    swap: ``role_swap`` mapping applied to the conditionals of ``swap_items``, which must be given with it: swapping
        a stable item (party ID itself, religion) hands each party the other's answers.
    """
    df, item, value, target = _prepare(df, wave, segment, item, value, target)
    if k < 1 or (m is not None and m < 0):
        raise ValueError("k must be >= 1 and m >= 0")
    if isinstance(weights, str) and weights != "target":
        raise ValueError("weights must be None, 'target', {subgroup: w} or {segment: {subgroup: w}}")
    if swap and swap_items is None:
        raise ValueError("swap= needs swap_items= (the regime-dependent items, e.g. presidential approval)")
    before = _waves_before(df, wave, target)
    if not before:
        raise ValueError(f"no wave before target {target!r}")
    train = _window(df, wave, before[-k:])
    segments, items = _cells(df, wave, segment, item, value, categories, target, train, segments, items)
    sub_c = _counts(train, [subgroup], item, value, categories, weight, items)
    seg_sub_c = _counts(train, [segment, subgroup], item, value, categories, weight, items) if m is not None \
        else None
    comp = {}   # {item: weight sum per (segment, subgroup)}
    if weights is None:
        last = _counts(_window(df, wave, [before[-1]]), [segment, subgroup], item, value, categories, weight, items)
        comp = {it: last[it][0].sum(axis=1) for it in items}
    elif isinstance(weights, str):
        tg = df[df[wave] == target]
        if not len(tg):
            raise ValueError("weights='target' needs the target wave's rows (subgroup labels) in df")
        for it in items:   # every row of the item, answered or not: the answers are not used
            di = tg[tg[item] == it]
            w = di[weight].astype(float) if weight else pd.Series(1.0, index=di.index)
            comp[it] = w.groupby([di[segment], di[subgroup]]).sum()
    sw_items = set(swap_items or ())
    preds, ntrain = {}, {}
    for it in items:
        c, sub_n, _ = sub_c[it]
        cond = {g: d for g in c.index if (d := _dist(c.loc[g].to_numpy())) is not None}
        for s in segments:
            cs = dict(cond)
            ns = 0.0
            if seg_sub_c is not None:
                cc, nn, _ = seg_sub_c[it]
                for g in cond:
                    if (s, g) in cc.index:
                        own = _dist(cc.loc[(s, g)].to_numpy())
                        n_sg = float(nn.get((s, g), 0.0))
                        if own is not None:
                            cs[g] = (n_sg * own + m * cond[g]) / (n_sg + m) if n_sg + m > 0 else cond[g]
                            ns += n_sg
            if swap and it in sw_items:
                cs = role_swap(cs, swap)
            if weights is None or isinstance(weights, str):
                ws = {g: float(v) for (s2, g), v in comp[it].items() if s2 == s}
                if not ws:   # segment absent from the composition source: population subgroup shares
                    ws = {g: float(v) for g, v in c.sum(axis=1).items()}
            elif all(isinstance(v, Mapping) for v in weights.values()):
                ws = dict(weights.get(s) or {})
                if not ws:
                    raise KeyError(f"no subgroup weights for segment {s!r}")
            else:
                ws = dict(weights)
            preds[(s, it)] = mixture_reweight(cs, ws)
            ntrain[(s, it)] = ns if seg_sub_c is not None else \
                float(sum(sub_n.get(g, 0.0) for g, v in ws.items() if v > 0 and g in cond))
    fr = _frame(preds, ntrain, df, wave, segment, item, value, categories, weight, target, segments, items, method)
    fr.attrs["waves"] = list(before[-k:])
    return fr


# ---------------------------------------------------------------- combining and backtests

def merge_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Join several predictors' frames for the same target(s) on (wave, unit): one row per cell, every
    pred_<method> / n_train_<method> column side by side; truth, n and labels from the first frame."""
    if not frames:
        raise ValueError("no frames")
    out = frames[0]
    key = ["wave", "unit"] if "wave" in out.columns else ["unit"]
    for f in frames[1:]:
        extra = [c for c in f.columns if c.startswith(("pred_", "n_train_", "filled_")) and c not in out.columns]
        dup = [c for c in f.columns if c.startswith("pred_") and c in out.columns]
        if dup:
            raise ValueError(f"duplicate method columns {dup}: give each predictor its own method=")
        out = out.merge(f[key + extra], on=key, how="outer")
    return out


@dataclass
class BacktestResult:
    """frame: long distribution frame over all backtest waves (``wave`` column; evaluate with group="wave" or
    per wave); by_wave: wave x method mean 1-TVD over scored cells; summary: per method mean, sd and min over
    waves, the number of waves it led, and the mean rank."""
    frame: pd.DataFrame
    by_wave: pd.DataFrame
    summary: pd.DataFrame
    waves_used: dict = field(default_factory=dict)


def transition_backtest(df: pd.DataFrame, waves: Sequence, methods: Mapping[str, Any], wave: str,
                        segment: str, item: str | None, value, categories, weight: str | None = None,
                        segments: Sequence | None = None, items: Sequence | None = None) -> BacktestResult:
    """Score each method on several past target waves (e.g. the earlier regime-switch years), each predicted
    only from waves before it, so the year-to-year spread is visible.

    methods: {name: (fn, kwargs)} with fn one of this module's predictors (called with the column arguments,
    ``target=<wave>`` and ``method=name``), or {name: callable(df, target) -> frame with pred_<name> (or a single
    pred_ column, renamed)}. Cells are the target wave's segment x item cells unless ``segments`` / ``items`` say.
    """
    df, item, value = _long(df, item, value)
    cols = dict(wave=wave, segment=segment, item=item, value=value, categories=categories, weight=weight,
                segments=segments, items=items)
    per_wave, used = [], {}
    for t in waves:
        if not (df[wave] == t).any():
            raise ValueError(f"backtest wave {t!r} not in df (truth needed)")
        frames = []
        for name, spec in methods.items():
            if isinstance(spec, tuple):
                fn, kw = spec
                f = fn(df, **cols, **dict(kw), target=t, method=name)
            else:
                f = spec(df, t)
                pc = [c for c in f.columns if c.startswith("pred_")]
                if f"pred_{name}" not in pc:
                    if len(pc) != 1:
                        raise ValueError(f"method {name!r} returned pred columns {pc}; expected pred_{name}")
                    f = f.rename(columns={pc[0]: f"pred_{name}"})
                if "wave" not in f.columns:
                    f = f.assign(wave=t)
            used[(t, name)] = f.attrs.get("waves")
            frames.append(f)
        per_wave.append(merge_frames(frames))
    frame = pd.concat(per_wave, ignore_index=True)
    names = list(methods)
    scored = frame[frame["truth"].map(lambda v: isinstance(v, list))]
    sc = pd.DataFrame({n: [one_minus_tvd(p, t) for p, t in zip(scored[f"pred_{n}"], scored["truth"])]
                       for n in names}, index=scored.index)
    sc["wave"] = scored["wave"].to_numpy()
    by_wave = sc.groupby("wave")[names].mean().reindex(list(waves))
    ranks = by_wave.rank(axis=1, ascending=False)
    lead = by_wave.idxmax(axis=1).value_counts().reindex(names).fillna(0).astype(int)
    summary = pd.DataFrame({"mean": by_wave.mean(), "sd": by_wave.std(ddof=1), "min": by_wave.min(),
                            "waves_led": lead, "mean_rank": ranks.mean()}).sort_values("mean", ascending=False)
    return BacktestResult(frame=frame, by_wave=by_wave, summary=summary, waves_used=used)
