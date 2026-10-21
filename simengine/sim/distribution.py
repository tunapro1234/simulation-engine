"""Segment-distribution prompting: ask the model directly for each segment's answer distribution (no personas),
plus per-persona probability vectors. Port of makro/llmabm/kod/c1/k2.py (`dag_istem`, `dag_ayikla`, ACIK/GIZLI
framings, OLAS_ISTEM) and makro/kod/kit/yontemler/llm_dagilim.py (variant averaging, invalid -> uniform).

Measured (makro C1, CES 2020, claude-haiku-5-5, 4 segments x 8 items, normalised W1, lower is better): one call
per segment x item x framing, 3 framings averaged, closed 82% of the gap between the general-population
marginal (W1 0.206) and a 37-person real survey of the segment (0.049): W1 0.077, vs 0.120 (55%) for a
148-persona panel; per-persona probability vectors 0.088 (75%). Details and limits in docs/modules/sim.md.

    qs = [{"id": "q1", "text": "Do you approve of ...?", "options": ["Strongly approve", ..., "Strongly disapprove"]}]
    segs = {"D-young": "US adults aged 18 to 54 who identify as Democrats", ...}
    res = segment_distribution(llm, segs, qs, population="US adults")
    res["dist"]["D-young"]["q1"]      # averaged probability vector
    frame = distribution_frame(res, truth={"D-young": {"q1": [0.1, 0.3, ...]}, ...})
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._stats import to_jsonable
from .panel import Usage
from .parse import normalize_probs, parse_distribution, parse_json, unwrap
from .personas import as_personas

_MINORITY = " Think about the full spread of opinion in this group, including minority views."

# k2.py ACIK (named survey + date; pre-registered arm (i)) and GIZLI (no name, no year; arm (ii)). The measured
# wording is reproduced exactly with population="US adults", source="the Cooperative Election Study (CES)",
# when="October 2020"; other studies fill the same slots.
FRAMINGS_NAMED = {
    "v1": "In {source}, a large national survey of {population} fielded in {when}, consider respondents who are "
          "{group}.",
    "v2": "In a national survey of {population} conducted in {when}, consider respondents who are {group}.",
    "v3": "In {source}, a large national survey of {population} fielded in {when}, consider respondents who are "
          "{group}." + _MINORITY,
}
FRAMINGS_HIDDEN = {
    "h1": "In a large national survey of {population}, consider respondents who are {group}.",
    "h2": "Consider {group}.",
    "h3": "In a large national survey of {population}, consider respondents who are {group}." + _MINORITY,
}

DIST_TEMPLATE = """{head}
Question: {stem}
Answer options:
{options}
Estimate the percentage of these respondents who would choose each option (ignore "not sure" answers). Percentages must sum to 100.
Return ONLY a JSON object mapping option number to percentage, e.g. {example}"""

PERSON_PROB_TEMPLATE = """You are simulating {setting}.
Below are {n} survey respondents. For EACH respondent and EACH question, estimate the probability (in percent) that this respondent gives each answer option. They answer independently.
Session: {session}

Respondents:
{people}

Questions (answer options are numbered):
{questions}

Return ONLY a JSON object with exactly these {n} respondent IDs as keys ({ids}). For each respondent, "p" is a list of {nq} lists (one per question, in order), each list giving the percent for options 1..K and summing to 100:
{{"{first}": {{"p": [{skeleton}]}}, ...}}"""


def _questions(questions) -> list[dict]:
    out = []
    for i, q in enumerate(questions):
        opts = [str(x) for x in (q.get("options") or [])]
        if len(opts) < 2:
            raise ValueError(f"question {q.get('id', i)}: needs at least 2 options")
        out.append({"id": str(q.get("id", f"q{i + 1}")), "text": q.get("text", ""), "options": opts})
    if not out:
        raise ValueError("questions is empty")
    return out


def _segments(segments) -> dict[str, str]:
    if isinstance(segments, Mapping):
        return {str(k): str(v) for k, v in segments.items()}
    return {str(s): str(s) for s in segments}


def distribution_prompt(framing: str, group: str, question: dict, *, population: str = "adults",
                        source: str | None = None, when: str | None = None) -> str:
    """One measured-style prompt: framing head + question + numbered options + JSON percentage request."""
    k = len(question["options"])
    head = framing.format(group=group, population=population, source=source or "a large national survey",
                          when=when or "recent years")
    return DIST_TEMPLATE.format(head=head, stem=question["text"],
                                options="\n".join(f"{j + 1}. {x}" for j, x in enumerate(question["options"])),
                                example=json.dumps({str(j + 1): round(100 / k) for j in range(k)}))


def segment_distribution(llm, segments, questions, *, population: str = "adults", source: str | None = None,
                         when: str | None = None, framings: Mapping[str, str] | None = None,
                         max_tokens: int = 300, fallback: str = "uniform") -> dict:
    """One call per segment x question x framing; vectors averaged over framings.

    segments: {key: group description} or a list of descriptions. framings default: FRAMINGS_NAMED when
    `source` is given (the pre-registered arm), else FRAMINGS_HIDDEN. An invalid answer becomes a uniform
    vector before averaging (fallback='uniform', as in the measured kit) or is skipped (fallback='skip').
    Returns {"dist": {segment: {question_id: [p...]}}, "cells": [...], "invalid": n, "calls": n, "usage"}."""
    segs = _segments(segments)
    qs = _questions(questions)
    framings = dict(framings or (FRAMINGS_NAMED if source else FRAMINGS_HIDDEN))
    jobs = [(v, s, q) for v in framings for s in segs for q in qs]
    prompts = [("", distribution_prompt(framings[v], segs[s], q, population=population, source=source, when=when))
               for v, s, q in jobs]
    usage = Usage()
    resp = llm.map(prompts, max_tokens=max_tokens) if prompts else []
    cells, acc, invalid = [], {}, 0
    for (v, s, q), r in zip(jobs, resp):
        usage.add(r, "distribution")
        k = len(q["options"])
        p = parse_distribution(getattr(r, "text", None), k, q["options"])
        if p is None:
            invalid += 1
        cells.append({"variant": v, "segment": s, "item": q["id"], "p": None if p is None else p.tolist()})
        if p is None and fallback == "uniform":
            p = np.full(k, 1.0 / k)
        if p is not None:
            acc.setdefault(s, {}).setdefault(q["id"], []).append(p)
    dist = {s: {qid: np.mean(v, axis=0).tolist() for qid, v in d.items()} for s, d in acc.items()}
    return to_jsonable({"method": "segment_distribution", "dist": dist, "cells": cells, "invalid": invalid,
                        "calls": len(prompts), "framings": list(framings), "options": {q["id"]: q["options"] for q in qs},
                        "usage": usage.as_dict()})


def _person_block(questions) -> str:
    return "\n".join(f"Q{i + 1}. {q['text']} " + ", ".join(f"{j + 1}={x}" for j, x in enumerate(q["options"]))
                     for i, q in enumerate(questions))


def persona_distribution(llm, personas, questions, *, setting: str = "a survey of adults", batch: int = 12,
                         session: str = "s1", max_tokens: int | None = None) -> dict:
    """Per-persona probability vectors, many respondents per call (k2.py arm (iii), OLAS_ISTEM, generalised:
    `setting` replaces the measured "a public opinion survey of US adults in October 2020, ..." line; tables are
    filled within segment).
    Returns {"person": {pid: {question_id: [p...] | None}}, "dist": {segment: {question_id: mean vector}},
    "invalid": n personas with an unusable answer}. Segment means use a uniform vector for invalid personas."""
    ps = as_personas(personas)
    order = {sg: i for i, sg in enumerate(dict.fromkeys(p.seg for p in ps))}
    ps = sorted(ps, key=lambda p: order[p.seg])     # tables within a segment, as in the measured run
    qs = _questions(questions)
    qtext = _person_block(qs)
    skel = ", ".join("[...]" for _ in qs)
    plan = []
    for b in range(0, len(ps), max(1, batch)):
        g = ps[b:b + batch]
        ids = [p.pid for p in g]
        user = PERSON_PROB_TEMPLATE.format(setting=setting, n=len(g), session=f"{session}-table{b // max(1, batch)}",
                                           people="\n".join(f"{p.pid}: " + p.text("en").replace("\n", "; ")
                                                            for p in g),
                                           questions=qtext, ids=", ".join(ids), first=ids[0], nq=len(qs),
                                           skeleton=skel)
        plan.append((("", user), ids))
    if max_tokens is None:
        max_tokens = min(16000, 200 + batch * len(qs) * 12 * max(len(q["options"]) for q in qs))
    usage = Usage()
    person: dict[str, dict] = {}
    invalid = 0
    for (_, ids), r in zip(plan, llm.map([p for p, _ in plan], max_tokens=max_tokens)):
        usage.add(r, "persona_distribution")
        js = parse_json(getattr(r, "text", None))
        js = js if isinstance(js, dict) else {}
        for pid in ids:
            row = js.get(pid)
            L = unwrap(row.get("p") if isinstance(row, dict) else row)
            vecs = None
            if isinstance(L, list) and len(L) == len(qs):
                vecs = [normalize_probs(x, len(q["options"]), q["options"]) for x, q in zip(L, qs)]
                if any(v is None for v in vecs):
                    vecs = None
            if vecs is None:
                invalid += 1
                person[pid] = {q["id"]: None for q in qs}
            else:
                person[pid] = {q["id"]: v.tolist() for q, v in zip(qs, vecs)}
    seg = {p.pid: p.seg for p in ps}
    dist: dict[str, dict] = {}
    for sg in sorted(set(seg.values())):
        for q in qs:
            k = len(q["options"])
            vs = [np.asarray(person[p][q["id"]]) if person.get(p, {}).get(q["id"]) is not None else np.full(k, 1 / k)
                  for p in seg if seg[p] == sg]
            dist.setdefault(sg, {})[q["id"]] = np.mean(vs, axis=0).tolist()
    return to_jsonable({"method": "persona_distribution", "person": person, "dist": dist, "invalid": invalid,
                        "calls": len(plan), "segments": seg, "options": {q["id"]: q["options"] for q in qs},
                        "usage": usage.as_dict()})


def distribution_frame(result: dict, truth: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
                       method: str = "llm_dist") -> pd.DataFrame:
    """EvalFrame (kind 'distribution'), one row per segment x question: truth (list) and pred_<method> (list)."""
    truth = truth or {}
    rows = []
    for s, d in (result.get("dist") or {}).items():
        for qid, p in d.items():
            t = (truth.get(s) or {}).get(qid)
            rows.append({"unit": f"{s}|{qid}", "truth": list(t) if t is not None else np.nan,
                         f"pred_{method}": list(p), "segment": s, "item": qid, "group": qid})
    return pd.DataFrame(rows)


def w1_ordinal(p: Any, q: Any) -> float:
    """Normalised 1-Wasserstein distance between two distributions over the same k ordered options, in [0, 1]
    (sum of |CDF differences| / (k - 1)); the metric used for the makro C1 numbers."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    k = len(p)
    return float(np.abs(np.cumsum(p) - np.cumsum(q))[:-1].sum() / max(1, k - 1))


__all__ = ["segment_distribution", "persona_distribution", "distribution_prompt", "distribution_frame",
           "w1_ordinal", "FRAMINGS_NAMED", "FRAMINGS_HIDDEN", "DIST_TEMPLATE"]
