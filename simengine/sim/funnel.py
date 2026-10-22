"""Funnel: a multi-step journey (landing page -> pricing -> checkout ...) asked step by step on a persona panel,
with per-step continue rates, cumulative reach, drop-off and a paired comparison of the final conversion.

    from simengine.sim.funnel import funnel, funnel_frame
    variants = {"A": ["Landing: ...", "Pricing: $9/month ...", "Checkout: card details ..."],
                "B": ["Landing: ...", "Pricing: free trial, then $9/month ...", "Checkout: ..."]}
    res = funnel(personas, variants, llm=llm, context="a budgeting app for students")
    res["variants"]["A"]["steps"][1]["continue"]     # {value, low, high, n}: mean P(continue | reached step 2)
    res["variants"]["A"]["steps"][1]["reach"]        # share of the panel still in the funnel after step 2
    res["variants"]["A"]["conversion"]               # reach after the last step
    res["comparison"]["pairs"]                       # B - A final conversion, paired persona bootstrap CI
    frame = funnel_frame(res)                        # long DataFrame: variant, step, persona, p_continue, reach

Each step is one `simengine.sim.survey` call with a numeric 0-100 question. The step's context shows the earlier
steps of the same variant ("You have already seen: ...", each marked as continued past) and then the current one,
so the answer is read as P(continue | reached this step). Per persona, reach after step k is the product of its
continue probabilities up to k; the variant's reach is the panel mean of those products (not the product of the
step means), with a percentile bootstrap CI over personas. Drop-off at step k is reach(k-1) - reach(k).
Variants are compared on the same personas: the CI of a difference resamples personas, not variants.

Controls: unlike `survey`, no persona-control arm runs by default (`controls=None` = none): a funnel already costs
variants x steps survey calls. Pass controls=["no_persona"] (or both arms) to get each step's control-arm mean
continue rate next to the persona arm; reach and comparisons use the persona arm only.
Steps shared by two variants give identical prompts, so a caching backend answers them once.
Offline: FakeLLM(schema_responder()) runs the same code path (dry runs, not measurement).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._stats import B_DEFAULT, Emitter, bootstrap_ci, bootstrap_win, to_jsonable
from .personas import as_personas
from .survey import survey

QUESTION = "How likely are you to continue to the next step? (0-100)"
QID = "cont"


def _steps(vid: str, steps) -> list[dict]:
    """[str | {"id"/"name", "text"}] -> [{"id", "text"}]."""
    if isinstance(steps, (str, dict)):
        steps = [steps]
    out = []
    for k, s in enumerate(steps or []):
        if isinstance(s, dict):
            sid, text = str(s.get("id", s.get("name", f"s{k + 1}"))), str(s.get("text", ""))
        else:
            sid, text = f"s{k + 1}", str(s)
        if not text.strip():
            raise ValueError(f"variant {vid!r}: step {k + 1} has no text")
        out.append({"id": sid, "text": text})
    if not out:
        raise ValueError(f"variant {vid!r} has no steps")
    if len({s["id"] for s in out}) != len(out):
        raise ValueError(f"variant {vid!r}: step ids must be unique")
    return out


def step_context(steps: list[dict], k: int, context: str = "") -> str:
    """Context shown at step k (0-based): shared context, the earlier steps as already seen, then the current step."""
    n = len(steps)
    parts = [context] if context else []
    if k:
        seen = "\n".join(f"  Step {j + 1}: {steps[j]['text']}" for j in range(k))
        parts.append("You have already seen (and chose to continue past each of these):\n" + seen)
    parts.append(f"Now you are on step {k + 1} of {n}:\n{steps[k]['text']}")
    return "\n\n".join(parts)


def _paired(a: np.ndarray, b: np.ndarray, B: int, seed: int) -> dict:
    """Paired persona bootstrap of mean(a - b) -> {value, low, high, n, p_better} (NaN pairs dropped)."""
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    out = bootstrap_ci(d.tolist(), B=B, seed=seed)
    if len(d) > 1:
        r = np.random.default_rng(seed)
        bs = d[r.integers(0, len(d), size=(max(1, B), len(d)))].mean(axis=1)
        out["p_better"] = float((bs > 0).mean() + 0.5 * (bs == 0).mean())
    else:
        out["p_better"] = None if not len(d) else float(d[0] > 0) + 0.5 * float(d[0] == 0)
    return out


def _usage_sum(us: list[dict]) -> dict:
    out: dict = {}
    for u in us:
        for k, v in u.items():
            if k in ("model", "last_error"):
                out[k] = v or out.get(k)
            elif k != "refusal_rate" and isinstance(v, (int, float)):
                out[k] = out.get(k, 0) + v
    calls = sum(u.get("calls", 0) for u in us)
    out["refusal_rate"] = round(sum(u.get("refusal_rate", 0) * u.get("calls", 0) for u in us) / calls, 4) \
        if calls else 0.0
    out["surveys"] = len(us)
    return out


def funnel(personas, variants: dict, *, llm, question: str = QUESTION, context: str = "", controls=None,
           batch: int = 8, seed: int = 0, B: int = B_DEFAULT, lang: str | None = None, free_text: bool = False,
           parallel: int | None = None, tag: str | None = None, on_event=None) -> dict:
    """Simulated funnel. `variants`: {variant_id: [step_text | {"id", "text"}, ...]} (steps in order; variants
    may differ in length). Every persona is asked every step of every variant (conditional on having reached it).
    Returns a JSON-safe dict:
      variants: {vid: {"steps": [{id, index, text, continue, reach, drop, drop_rate, n, controls?, quotes?}],
                       "conversion", "biggest_drop", "segments": {segment: conversion CI}, "n_complete"}}
      comparison: {"baseline", "pairs": [{variant, baseline, diff, p_better}], "win_prob", "best"}
      answers: {vid: {pid: {"p": [p per step], "reach": [...]}}} (None where unanswered); segments; usage; warnings.
    continue / reach / drop / conversion are {value, low, high, n} on a 0-1 scale (persona bootstrap, B resamples).
    A persona without an answer at some step drops out of that variant's reach from that step on (warned).
    question: the numeric 0-100 question asked at each step. context: a text shown before the steps at every step.
    controls: persona-control arms (default none, see the module docstring). batch / lang / free_text / parallel:
    passed to `survey` (free_text=True adds a one-sentence reason per answer and per-step quotes). tag: ledger
    label; calls are tagged f"{tag}:funnel:{vid}:{step}:survey:{arm}"."""
    if not isinstance(variants, dict) or not variants:
        raise ValueError("variants must be a non-empty {variant_id: [step_text, ...]} dict")
    em = Emitter(on_event, "funnel")
    ps = as_personas(personas)
    pids = [p.pid for p in ps]
    seg = {p.pid: p.seg for p in ps}
    vs = {str(v): _steps(str(v), s) for v, s in variants.items()}
    q = {"id": QID, "text": question, "kind": "numeric", "min": 0, "max": 100, "free_text": free_text}
    ctrl = [] if controls is None else list(controls)
    warnings: list[str] = []
    usages: list[dict] = []
    P: dict[str, np.ndarray] = {}                 # vid -> (n personas, n steps) continue probabilities, NaN = none
    out_v: dict[str, dict] = {}
    for vid, steps in vs.items():
        M = np.full((len(ps), len(steps)), np.nan)
        rows = []
        for k, st in enumerate(steps):
            em.stage("step", variant=vid, step=st["id"], index=k, n=len(ps))
            t = f"{tag}:funnel:{vid}:{st['id']}" if tag else f"funnel:{vid}:{st['id']}"
            r = survey(ps, [q], llm=llm, context=step_context(steps, k, context), lang=lang, batch=batch,
                       controls=ctrl, B=B, free_text=free_text, parallel=parallel, tag=t)
            usages.append(r["usage"])
            warnings += [f"{vid}/{st['id']}: {w}" for w in r["warnings"]]
            for i, pid in enumerate(pids):
                a = (r["answers"].get(pid) or {}).get(QID)
                if a and a.get("value") is not None:
                    M[i, k] = a["value"] / 100.0
            qr = r["questions"][0]
            row = {"id": st["id"], "index": k, "text": st["text"]}
            if ctrl:
                row["controls"] = {arm: {kk: (None if s.get("mean", {}).get(kk) is None
                                              else (s["mean"][kk] / 100.0 if kk != "n" else s["mean"][kk]))
                                         for kk in ("value", "low", "high", "n")}
                                   for arm, s in qr["arms"].items() if arm != "persona"}
            if qr.get("quotes"):
                row["quotes"] = qr["quotes"]
            rows.append(row)
        R = np.cumprod(M, axis=1)                  # NaN propagates: no reach after a missing answer
        prev = np.ones(len(ps))
        for k, row in enumerate(rows):
            col = M[:, k]
            row["continue"] = bootstrap_ci(col[~np.isnan(col)].tolist(), B=B, seed=seed)
            row["reach"] = bootstrap_ci(R[~np.isnan(R[:, k]), k].tolist(), B=B, seed=seed)
            dk = prev - R[:, k]
            row["drop"] = bootstrap_ci(dk[~np.isnan(dk)].tolist(), B=B, seed=seed)
            v = row["continue"]["value"]
            row["drop_rate"] = None if v is None else 1.0 - v
            row["n"] = int((~np.isnan(col)).sum())
            prev = R[:, k]
        final = R[:, -1]
        lost = int(np.isnan(final).sum())
        if lost:
            warnings.append(f"{vid}: {lost}/{len(ps)} personas missing an answer at some step; they are left out of "
                            "reach from that step on")
        drops = [(r["drop"]["value"], r["id"]) for r in rows if r["drop"]["value"] is not None]
        segs = sorted({seg[p] for p in pids})
        out_v[vid] = {"steps": rows, "conversion": rows[-1]["reach"], "n_complete": len(ps) - lost,
                      "biggest_drop": max(drops)[1] if drops else None,
                      "segments": {sg: bootstrap_ci([x for p, x in zip(pids, final) if seg[p] == sg and x == x],
                                                    B=B, seed=seed) for sg in segs},
                      "answers": {pid: {"p": M[i].tolist(), "reach": R[i].tolist()} for i, pid in enumerate(pids)}}
        P[vid] = final
        em.row(vid, None, f"conversion {out_v[vid]['conversion']['value']}", out_v[vid]["conversion"]["value"])
    ids = list(vs)
    base = ids[0]
    pairs = [{"variant": v, "baseline": base, **{"diff": None, "p_better": None}} for v in ids[1:]]
    for pr in pairs:
        d = _paired(P[pr["variant"]], P[base], B, seed)
        pr["p_better"] = d.pop("p_better")
        pr["diff"] = d
    W = np.column_stack([P[v] for v in ids])
    win = bootstrap_win(W, B=B, seed=seed) if len(ids) > 1 else [1.0]
    comparison = {"baseline": base, "pairs": pairs, "win_prob": dict(zip(ids, win)) if win else {},
                  "best": ids[int(np.argmax(win))] if win else None}
    answers = {vid: d.pop("answers") for vid, d in out_v.items()}
    out = {"runner": "funnel", "question": question, "n_personas": len(ps), "variants": out_v,
           "comparison": comparison, "answers": answers, "segments": seg, "controls": ctrl,
           "usage": _usage_sum(usages), "warnings": warnings}
    em.done(n=len(ps))
    return to_jsonable(out)


def funnel_frame(result: dict) -> pd.DataFrame:
    """Long DataFrame from a funnel result: one row per variant x step x persona with columns variant, step,
    step_index, persona, group (= persona, for evidence group metrics), segment, p_continue, reach (NaN where
    unanswered)."""
    seg = result.get("segments", {})
    rows = []
    for vid, v in result.get("variants", {}).items():
        ans = (result.get("answers") or {}).get(vid, {})
        for st in v["steps"]:
            k = st["index"]
            for pid, a in ans.items():
                p, r = a["p"][k], a["reach"][k]
                rows.append({"variant": vid, "step": st["id"], "step_index": k, "persona": pid, "group": pid,
                             "segment": seg.get(pid), "p_continue": np.nan if p is None else p,
                             "reach": np.nan if r is None else r})
    return pd.DataFrame(rows, columns=["variant", "step", "step_index", "persona", "group", "segment", "p_continue",
                                       "reach"])


__all__ = ["funnel", "funnel_frame", "step_context", "QUESTION"]
