"""Segment event-rate prior: ask the model, per segment, what share of units has an event, k samples each, and
return a segment x rate table (rates in [0, 1]) ready as a group feature.

    segs = {"new|fast": "new visitors whose first two page views are under 60 seconds apart", ...}
    prior = segment_rate_prior(llm, segs, "the session ends with a purchase", samples=5,
                               context="An online shop logs product views, add-to-cart and purchase events.")
    prior                  # index segment; mean, sd, n_ok, n_failed (mean/sd as rates in [0, 1])
    GroupTransfer(group_features=prior[["mean"]]).fit(X, y, groups=seg)     # or join prior["mean"] as a feature

    # blind arm: same event, segment replaced by "group A", ... (or your own hidden wording)
    prior = segment_rate_prior(llm, segs, event, blind=True)                # + mean_blind, sd_blind, ...

One call per segment x sample (x arm). samples=k passes `sample=0..k-1` to llm.complete (Claude keys its disk cache
on `sample`, so samples are independent draws; samples=1 is the plain call). Calls go through `sim.panel.ask_many`
(one `llm.map` batch per sample; soft refusals re-asked once, empty replies re-asked with a larger budget). Replies
are read with `simengine.llm.jsonx.json_extract`: `{"pct": 12}` (parse_pct=True, a percentage 0-100) or
`{"rate": 0.12}` (parse_pct=False, a proportion 0-1); a bare number or "12%" string is read too. A reply that is
missing, unparsed or out of range is a failed sample (never clipped), counted in n_failed and warned about once.
Port of the hand-written prior in blind rehearsal D13 (6 segments x 5 samples, `complete` + `json_extract`). That
prior over-estimated the purchase rate about threefold and its AP was at the segment-rate baseline, so treat it as
a feature to test against a blind / shuffled control, not as a forecast.
"""
from __future__ import annotations

import math
import re
import string
import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..llm.jsonx import json_extract
from ._llmarg import MISSING, resolve_llm
from .panel import REFRAME_NOTE_GENERIC, Usage, ask_many

RATE_SYSTEM = "You are a careful analyst. Give calibrated numeric estimates."
RATE_TEMPLATE = """{context}Consider {group}.
Estimate {what} of them for which the following happens: {event}.
Return ONLY a JSON object: {answer}"""
PCT = {"what": "the percentage", "answer": '{"pct": <number between 0 and 100>}', "scale": 100.0}
RATE = {"what": "the share (a proportion between 0 and 1)", "answer": '{"rate": <number between 0 and 1>}',
        "scale": 1.0}
VALUE_KEYS = ("pct", "percent", "percentage", "rate", "share", "proportion", "value", "estimate")
_NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(%?)\s*$")


def _segments(segments) -> dict[str, str]:
    if isinstance(segments, Mapping):
        return {str(k): str(v) for k, v in segments.items()}
    return {str(s): str(s) for s in segments}


def _blind_labels(keys: list[str], blind) -> dict[str, str]:
    if isinstance(blind, Mapping):
        missing = [k for k in keys if k not in {str(b) for b in blind}]
        if missing:
            raise ValueError(f"blind= mapping has no wording for segments {missing[:3]}")
        return {str(k): str(v) for k, v in blind.items() if str(k) in keys}
    letters = string.ascii_uppercase
    return {k: "group " + (letters[i] if i < 26 else f"{letters[i // 26 - 1]}{letters[i % 26]}")
            for i, k in enumerate(keys)}


def rate_prompt(group: str, event: str, *, context: str = "", parse_pct: bool = True,
                template: str = RATE_TEMPLATE) -> str:
    """One rate question. `context` (a setting sentence, no dataset name) goes first when given."""
    form = PCT if parse_pct else RATE
    ctx = context.strip() + "\n" if context and context.strip() else ""
    return template.format(context=ctx, group=group, event=event, what=form["what"], answer=form["answer"])


def parse_rate(text: str | None, *, parse_pct: bool = True) -> float | None:
    """Reply -> rate in [0, 1], or None (no JSON number, or outside 0-100 / 0-1). Never clipped."""
    data = json_extract(text, default=None) if text else None
    if isinstance(data, Mapping):
        low = {str(k).strip().lower(): v for k, v in data.items()}
        val = next((low[k] for k in VALUE_KEYS if k in low), None)
        if val is None and len(low) == 1:
            val = next(iter(low.values()))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        val = data
    else:
        m = _NUM.match(text or "")
        val = m.group(0) if m else None
    if isinstance(val, str):
        m = _NUM.match(val)
        if not m:
            return None
        val = float(m.group(1)) / (100.0 if m.group(2) and not parse_pct else 1.0)    # "4%" as a proportion
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
        return None
    hi = PCT["scale"] if parse_pct else RATE["scale"]
    if not 0 <= val <= hi:
        return None
    return float(val) / hi


def _summary(vals: list[list[float | None]], suffix: str = "") -> dict[str, list]:
    out: dict[str, list] = {f"mean{suffix}": [], f"sd{suffix}": [], f"n_ok{suffix}": [], f"n_failed{suffix}": []}
    for vs in vals:
        ok = np.array([v for v in vs if v is not None], dtype=float)
        out[f"mean{suffix}"].append(float(ok.mean()) if len(ok) else float("nan"))
        out[f"sd{suffix}"].append(float(ok.std(ddof=1)) if len(ok) > 1 else float("nan"))
        out[f"n_ok{suffix}"].append(int(len(ok)))
        out[f"n_failed{suffix}"].append(int(len(vs) - len(ok)))
    return out


def segment_rate_prior(*args, llm=None, segments=MISSING, event=MISSING, samples: int = 5, blind: Any = False,
                       parse_pct: bool = True, context: str = "", system: str = RATE_SYSTEM,
                       template: str = RATE_TEMPLATE, max_tokens: int = 200, ask_kw: dict | None = None,
                       usage: Usage | None = None) -> pd.DataFrame:
    """Per-segment event-rate prior from `samples` independent answers per segment.

    segments: {key: description} or a list of descriptions (key = description). event: what happens to a unit,
    e.g. "the session ends with a purchase". blind: False; True = a control arm with the same event and the segment
    replaced by "group A", "group B", ...; or {key: hidden wording} (e.g. keep the visitor type, hide the timing).
    Returns a DataFrame indexed by segment (name "segment"): mean, sd (ddof=1, NaN under 2 answers), n_ok,
    n_failed, and with blind the same columns suffixed "_blind". Rates are in [0, 1]. attrs: "samples" (long rows:
    segment, arm, sample, rate, text), "prompts", "usage", "warnings". Pass `df[["mean"]]` as
    `GroupTransfer(group_features=)` (every column becomes a feature). llm may be passed first (legacy) or as llm=."""
    llm, (segments, event) = resolve_llm("segment_rate_prior", args, llm, {"segments": segments, "event": event})
    segs = _segments(segments)
    if not segs:
        raise ValueError("segment_rate_prior needs at least one segment")
    keys = list(segs)
    arms = {"named": segs}
    if blind is not False and blind is not None:
        arms["blind"] = _blind_labels(keys, blind)
    jobs = [(a, k) for a in arms for k in keys]
    prompts = [(system, rate_prompt(arms[a][k], str(event), context=context, parse_pct=parse_pct, template=template))
               for a, k in jobs]
    usage = usage if usage is not None else Usage()
    samples = max(1, int(samples))
    kw = dict(ask_kw or {})
    s0 = int(kw.pop("sample", 0))
    vals: dict[tuple[str, str], list] = {j: [] for j in jobs}
    rows = []
    for i in range(samples):
        skw = {**kw, "sample": s0 + i} if (samples > 1 or s0) else kw     # samples=1: the plain call
        errs: list = []
        resp = ask_many(llm, prompts, max_tokens=max_tokens, usage=usage, tag="rate_prior",
                        reframe=REFRAME_NOTE_GENERIC, errors=errs, **skw)
        for (a, k), r in zip(jobs, resp):
            text = getattr(r, "text", None) if r is not None else None
            v = parse_rate(text, parse_pct=parse_pct)
            vals[(a, k)].append(v)
            rows.append({"segment": k, "arm": a, "sample": s0 + i, "rate": v, "text": text})
    df = pd.DataFrame(_summary([vals[("named", k)] for k in keys]), index=pd.Index(keys, name="segment"))
    if "blind" in arms:
        for c, v in _summary([vals[("blind", k)] for k in keys], "_blind").items():
            df[c] = v
    n_bad = sum(r["rate"] is None for r in rows)
    warns = []
    if n_bad:
        warns.append(f"segment_rate_prior: {n_bad} of {len(rows)} sampled answers failed (no reply, no number or "
                     f"outside {'0-100 %' if parse_pct else '0-1'}); counted in n_failed, mean over the rest")
        warnings.warn(warns[0], RuntimeWarning, stacklevel=2)
    df.attrs["samples"] = rows
    df.attrs["prompts"] = [{"segment": k, "arm": a, "system": s, "user": u} for (a, k), (s, u) in zip(jobs, prompts)]
    df.attrs["usage"] = usage.as_dict()
    df.attrs["warnings"] = warns
    return df


__all__ = ["segment_rate_prior", "rate_prompt", "parse_rate", "RATE_SYSTEM", "RATE_TEMPLATE"]
