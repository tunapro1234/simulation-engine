"""Robustness of segment distributions: how much does the answer move when the same prompt is asked again
(retest), asked in other words (framing) or shown with the options reversed (order), compared with how far the
groups sit from the population (the signal)? Port of the G7 rehearsal's `g7.py puanla` (makro C1 protocol:
retest + 2 framings + reversed options, ratio table "axis W1 / signal").

Built on `segment_distribution` (one framing per run, no averaging). Conditions:
  base         the first framing (FRAMINGS_HIDDEN["h1"] by default), options in the given order
  retest:<i>   the base prompt again with `sample=<base sample + i>` in the call keywords, so a cached backend
               (Claude's disk cache keys on `sample`) draws a new reply instead of returning the stored one
  framing:<v>  each other framing in `framings`
  order        the base prompt with the options reversed; the reply vector is mapped back to the original order
Per segment x question x condition: W1(base, condition) (sim.w1_ordinal; TVD with nominal=True) and, for order,
the signed shift E[pos | base] - E[pos | reversed], positions scaled to [0, 1] (negative = pulled toward the
option listed first, i.e. primacy; positive = recency). Signal per segment x question: W1(group, population).
Ratio per condition = mean condition W1 / mean signal W1; summary ratio = the worst condition. Questions are the
bootstrap clusters for the 95% CIs.

    qs = [{"id": "q1", "text": "Do you approve of ...?", "options": ["Strongly approve", ..., "Strongly disapprove"]}]
    segs = {"D-young": "US adults aged 18 to 54 who identify as Democrats", ...}
    rb = robustness(llm, segs, qs, population="US adults")
    rb["conditions"]        # [{"axis", "condition", "w1", "lo", "hi", "ratio", "ratio_lo", "ratio_hi", ...}]
    rb["ratio"], rb["worst"]                 # summary: worst condition noise / group signal
    rb["card"]              # evidence.kart row (tur "robustness"): append it to kart(...)["satirlar"]
    robustness_frame(rb)    # one row per segment x question x condition

signal: "population" (default) asks the population description once as its own segment under the base framing;
"mean" uses the mean of the segments' base vectors (no extra calls: the simulator's own between-group spread);
a mapping {"segments": {seg: {qid: vec}}, "population": {qid: vec}} uses real data (the rehearsal's anchor;
"segments" is required, "population" defaults to the mean of the given groups as in the rehearsal's round 3).
"population" and "mean" are the simulator's own scale; the card note says which one was used.
Calls: (1 + retests + extra framings + order) x segments x questions, plus questions for signal="population".
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._stats import to_jsonable
from .distribution import FRAMINGS_HIDDEN, FRAMINGS_NAMED, _questions, _segments, segment_distribution, w1_ordinal

AXES = ("retest", "framing", "order")
THRESHOLD = 0.5        # G7: above half the group signal, single-prompt subgroup comparisons are not trusted
_SIG_TR = {"given": "gerçek veri", "given-mean": "gerçek veri, grup ortalamasına karşı",
           "population": "simülatör, nüfus istemi", "mean": "simülatör, grup ortalamasına karşı"}


def tvd(p: Any, q: Any) -> float:
    """Total variation distance between two distributions over the same options (unordered), in [0, 1]."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())


def mean_position(p: Any) -> float:
    """E[option position] with positions scaled to [0, 1] (first option 0, last 1)."""
    p = np.asarray(p, float)
    p = p / p.sum()
    return float(p @ np.arange(len(p)) / max(1, len(p) - 1))


def _sum_usage(us: Sequence[dict]) -> dict:
    out: dict[str, Any] = {}
    for u in us:
        for k, v in (u or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "refusal_rate":
                out[k] = out.get(k, 0) + v
            elif v is not None and k not in out:
                out[k] = v
    calls = out.get("first_calls") or out.get("calls") or 0
    if "refused_calls" in out and calls:
        out["refusal_rate"] = round(out["refused_calls"] / calls, 4)
    return out


def _stat(per_q: np.ndarray, sig_q: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    """Mean condition distance over questions, its ratio to the mean signal, and question-clustered percentile
    95% CIs for both (questions with no cell on either side are dropped)."""
    ok = ~np.isnan(per_q) & ~np.isnan(sig_q)
    a, s = per_q[ok], sig_q[ok]
    out: dict[str, Any] = {"w1": float(a.mean()) if len(a) else None, "lo": None, "hi": None, "ratio": None,
                           "ratio_lo": None, "ratio_hi": None}
    if not len(a):
        return out
    if s.mean() > 0:
        out["ratio"] = float(a.mean() / s.mean())
    if len(a) >= 2 and n_boot > 0:
        idx = rng.integers(0, len(a), size=(n_boot, len(a)))
        m, sm = a[idx].mean(axis=1), s[idx].mean(axis=1)
        out["lo"], out["hi"] = np.percentile(m, [2.5, 97.5])
        r = m[sm > 0] / sm[sm > 0]
        if len(r):
            out["ratio_lo"], out["ratio_hi"] = np.percentile(r, [2.5, 97.5])
    return out


def robustness(llm, segments, questions, *, axes: Sequence[str] = AXES,
               framings: Mapping[str, str] | None = None, population: str = "adults", source: str | None = None,
               when: str | None = None, retests: int = 1, signal: str | Mapping = "population",
               nominal: bool = False, threshold: float = THRESHOLD, max_tokens: int = 300, fallback: str = "uniform",
               ask_kw: dict | None = None, n_boot: int = 1000, seed: int = 0,
               include_unsure: bool | None = None) -> dict:
    """Retest / framing / option-order robustness of `segment_distribution`, scaled by the group signal.

    framings: {name: template}; the first is the base, the others are the framing conditions (default
    FRAMINGS_NAMED when `source` is given, else FRAMINGS_HIDDEN). nominal=True: TVD in place of W1 and no order
    shift (unordered options such as brands); the order axis still runs. ask_kw goes to every run (`ask_many`
    options and call keywords); a `sample` there is the base sample, retest i uses sample + i.
    include_unsure goes to every run (segment_distribution, D6 #5), so the prompt measured is the one you run.
    Returns {"conditions": [...], "signal": {...}, "ratio", "worst", "verdict", "card", "cells": [...],
    "runs": {condition: segment_distribution result}, "calls", "invalid", "usage"}."""
    axes = tuple(axes)
    bad = [a for a in axes if a not in AXES]
    if bad:
        raise ValueError(f"unknown axes {bad}; use {AXES}")
    segs = _segments(segments)
    qs = _questions(questions)
    fr = dict(framings or (FRAMINGS_NAMED if source else FRAMINGS_HIDDEN))
    if not fr:
        raise ValueError("framings is empty")
    base_name = next(iter(fr))
    if "framing" in axes and len(fr) < 2:
        raise ValueError("the framing axis needs at least 2 framings (the first is the base)")
    if "retest" in axes and retests < 1:
        raise ValueError("retests must be >= 1 for the retest axis")
    dist = tvd if nominal else w1_ordinal
    akw = dict(ask_kw or {})
    sample0 = int(akw.pop("sample", 0) or 0)
    common = dict(population=population, source=source, when=when, max_tokens=max_tokens, fallback=fallback,
                  include_unsure=include_unsure)

    def run(sg, qq, fv: str, sample: int) -> dict:
        kw = {**akw, "sample": sample} if sample else akw
        return segment_distribution(llm, sg, qq, framings={fv: fr[fv]}, ask_kw=kw or None, **common)

    runs: dict[str, dict] = {"base": run(segs, qs, base_name, sample0)}
    conds: list[tuple[str, str]] = []                     # (axis, condition)
    if "retest" in axes:
        for i in range(1, retests + 1):
            runs[f"retest:{i}"] = run(segs, qs, base_name, sample0 + i)
            conds.append(("retest", f"retest:{i}"))
    if "framing" in axes:
        for v in list(fr)[1:]:
            runs[f"framing:{v}"] = run(segs, qs, v, sample0)
            conds.append(("framing", f"framing:{v}"))
    if "order" in axes:
        rq = [{**q, "options": q["options"][::-1]} for q in qs]
        r = run(segs, rq, base_name, sample0)
        # map each reversed vector back to the original option order
        r["dist"] = {s: {qid: list(p)[::-1] for qid, p in d.items()} for s, d in r["dist"].items()}
        r["cells"] = [{**c, "p": None if c.get("p") is None else list(c["p"])[::-1]} for c in r["cells"]]
        r["options"] = {q["id"]: q["options"] for q in qs}
        r["reversed"] = True
        runs["order"] = r
        conds.append(("order", "order"))
    base = runs["base"]["dist"]

    # signal: W1(group, population) per segment x question
    if isinstance(signal, Mapping):
        # real group vectors are required: W1(simulated group, real population) would count the simulator's own
        # error as group signal and flatter the ratio
        if not signal.get("segments"):
            raise ValueError("a given signal needs real 'segments' vectors {seg: {qid: vec}} "
                             "('population' optional: the mean of the given groups when absent)")
        gsrc = {str(s): {str(k): v for k, v in d.items()} for s, d in signal["segments"].items()}
        pop = {str(k): v for k, v in (signal.get("population") or {}).items()}
        sig_by = "given"
        if not pop:
            pop = {q["id"]: np.mean([gsrc[s][q["id"]] for s in segs if q["id"] in gsrc.get(s, {})], axis=0).tolist()
                   for q in qs if any(q["id"] in gsrc.get(s, {}) for s in segs)}
            sig_by = "given-mean"
    elif signal == "population":
        runs["population"] = run({"population": population}, qs, base_name, sample0)
        pop = runs["population"]["dist"].get("population", {})
        gsrc, sig_by = base, "population"
    elif signal == "mean":
        pop = {q["id"]: np.mean([base[s][q["id"]] for s in segs if q["id"] in base.get(s, {})], axis=0).tolist()
               for q in qs if any(q["id"] in base.get(s, {}) for s in segs)}
        gsrc, sig_by = base, "mean"
    else:
        raise ValueError("signal must be 'population', 'mean' or a mapping with 'segments' (and 'population')")
    sig_cells = []
    for s in segs:
        for q in qs:
            g, p = (gsrc.get(s) or {}).get(q["id"]), pop.get(q["id"])
            if g is not None and p is not None:
                sig_cells.append({"segment": s, "item": q["id"], "w1": dist(g, p)})

    cells = []
    for axis, c in conds:
        d = runs[c]["dist"]
        for s in segs:
            for q in qs:
                a, b = (base.get(s) or {}).get(q["id"]), (d.get(s) or {}).get(q["id"])
                if a is None or b is None:
                    continue
                row = {"axis": axis, "condition": c, "segment": s, "item": q["id"], "w1": dist(a, b)}
                if axis == "order" and not nominal:
                    row["shift"] = mean_position(a) - mean_position(b)
                cells.append(row)

    qids = [q["id"] for q in qs]

    def per_q(rows: list, key: str = "w1") -> np.ndarray:
        return np.array([np.mean([x[key] for x in rows if x["item"] == qi] or [np.nan]) for qi in qids])

    rng = np.random.default_rng(seed)
    sig_q = per_q(sig_cells)
    sig = _stat(sig_q, np.ones_like(sig_q), rng, n_boot)
    summary = []
    for axis, c in conds:
        rows = [x for x in cells if x["condition"] == c]
        st = _stat(per_q(rows), sig_q, rng, n_boot)
        row = {"axis": axis, "condition": c, **st, "n_cells": len(rows), "invalid": runs[c]["invalid"]}
        if axis == "order" and not nominal:
            sh = per_q(rows, "shift")
            sh = sh[~np.isnan(sh)]
            row["shift"] = float(sh.mean()) if len(sh) else None
            row["shift_lo"] = row["shift_hi"] = None
            if len(sh) >= 2 and n_boot > 0:
                bs = sh[rng.integers(0, len(sh), size=(n_boot, len(sh)))].mean(axis=1)
                row["shift_lo"], row["shift_hi"] = np.percentile(bs, [2.5, 97.5])
        summary.append(row)

    rated = [r for r in summary if r["ratio"] is not None]
    worst = max(rated, key=lambda r: r["ratio"]) if rated else None
    ratio = worst["ratio"] if worst else None
    if ratio is None:
        verdict = "no ratio: no condition or zero signal"
    elif ratio > threshold:
        verdict = (f"{worst['condition']} moves the answer by {ratio:.2f} x the group signal (> {threshold}): "
                   "single-prompt subgroup comparisons are not reliable; average framings and both option orders "
                   "and report the average's own retest spread")
    else:
        verdict = f"every condition stays at or under {threshold} x the group signal (worst {worst['condition']} {ratio:.2f})"
    if worst and worst["ratio_lo"] is not None and worst["ratio_lo"] <= threshold <= worst["ratio_hi"]:
        verdict += f" (borderline: the 95% CI [{worst['ratio_lo']:.2f}, {worst['ratio_hi']:.2f}] contains {threshold})"
    n_inv = sum(r["invalid"] for r in runs.values())
    metric = "TVD" if nominal else "W1"
    parts = [f"{r['condition']} {r['ratio']:.2f}" for r in rated]
    o = next((r for r in summary if r.get("shift") is not None), None)
    if o:
        parts.append(f"sıra kayması {o['shift']:+.3f}")
    from ..evidence.kart import satir           # lazy: evidence imports heavier deps
    card = satir("robustness", f"Sağlamlık: koşul oynaması / grup sinyali (en kötü: {worst['condition']})"
                 if worst else "Sağlamlık: koşul oynaması / grup sinyali", ratio,
                 worst["ratio_lo"] if worst else None, worst["ratio_hi"] if worst else None, birim="x sinyal",
                 not_=(f"{metric} oranları: " + ", ".join(parts) + f"; sinyal {metric} {sig['w1']:.3f} ({_SIG_TR[sig_by]})"
                       + (f"; geçersiz cevap {n_inv} ({'tekdüze dolduruldu' if fallback == 'uniform' else 'atlandı'})"
                          if n_inv else "")
                       + (f"; {threshold} üstü: tek istemle alt grup karşılaştırması güvenilmez"
                          if ratio is not None and ratio > threshold else "")) if rated else "ölçülmedi",
                 kaynak="sim.robustness")
    return to_jsonable({
        "method": "robustness", "metric": metric, "base_framing": base_name, "axes": list(axes),
        "conditions": summary, "ratio": ratio, "worst": worst["condition"] if worst else None,
        "threshold": threshold, "verdict": verdict, "card": card,
        "signal": {"by": sig_by, "w1": sig["w1"], "lo": sig["lo"], "hi": sig["hi"], "cells": sig_cells,
                   "population": pop},
        "cells": cells, "runs": runs, "calls": sum(r["calls"] for r in runs.values()),
        "invalid": {k: r["invalid"] for k, r in runs.items()},
        "usage": _sum_usage([r.get("usage") for r in runs.values()])})


def robustness_frame(result: dict) -> pd.DataFrame:
    """One row per segment x question x condition: axis, condition, segment, item, w1 (shift for order), and the
    cell's signal (W1 between the group and the population)."""
    sig = {(x["segment"], x["item"]): x["w1"] for x in (result.get("signal") or {}).get("cells") or []}
    df = pd.DataFrame(result.get("cells") or [])
    if df.empty:
        return df
    df["signal"] = [sig.get((s, i), np.nan) for s, i in zip(df["segment"], df["item"])]
    return df


__all__ = ["robustness", "robustness_frame", "tvd", "mean_position", "AXES", "THRESHOLD"]
