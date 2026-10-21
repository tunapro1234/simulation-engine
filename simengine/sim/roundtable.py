"""Multi-round roundtable: segmented panel, N discussion rounds, intent per round.
Port of sahne/kod/sahne/kosucu/roundtable.py; the optional stance instructions come from
makro/llmabm/kod/c1/masa.py (CONFIRMATION_BIAS = Chuang et al. 2024 style, KEEP_VIEW).

Round 0: everyone answers alone (intent 0-100 + a short remark for the table). Round t>0: each persona sees its
own previous answer and the previous-round summary of its segment (or of the whole table): mean intent and
2-3 quotes. Output: round x segment mean/sd (a falling sd can be homogenisation / herding), shift, persona
paths, quotes. Measured caveat (makro C1): plain discussion rounds changed 5-8% of answers and did not move
the distribution closer to the real survey; round 0 is the independent measurement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._stats import ALL, Emitter, bootstrap_ci, pick_quotes, summary_stats, to_jsonable, to_number
from .panel import Usage, ask_panel
from .personas import as_personas, choose_lang

DEFAULT_QUESTION = {"en": "How likely are you to buy / sign up / adopt this (0 = certainly not, 100 = certainly yes)?",
                    "tr": "Bunu satın alma / abone olma / benimseme olasılığın ne kadar (0 = kesinlikle hayır, "
                          "100 = kesinlikle evet)?"}
FIELDS = {"intent": "int 0-100", "remark": "str: 1-2 sentences the person says at the table"}
# makro/llmabm/kod/c1/masa.py
CONFIRMATION_BIAS = (" Like real people, every participant has confirmation bias: they give more weight to arguments "
                     "that support what they already believe and tend to dismiss or discount arguments against it.")
KEEP_VIEW = (" Each participant should keep their own view unless they hear an argument that would genuinely "
             "persuade someone with their background.")


def _task(topic, question, t, n_rounds, extra=""):
    if t == 0:
        return (f"Group discussion, round 1 of {n_rounds}. Topic:\n{topic}\n\nQuestion: {question}\n"
                "Each person answers alone first, before hearing anyone else.")
    return (f"Group discussion, round {t + 1} of {n_rounds}. Topic:\n{topic}\n\nQuestion: {question}\n"
            "Each person now hears what their group said in the previous round (see Context). They may keep, "
            "raise or lower their intent; changing only because others did is not required. "
            "Refer to the discussion if it matters to them." + (extra or ""))


def _group_summary(ans: dict, pids, n_quotes: int) -> str:
    v = [ans[p]["intent"] for p in pids if p in ans and ans[p]["intent"] is not None]
    if not v:
        return ""
    k = pick_quotes([{"pid": p, "segment": None, "text": ans[p]["remark"], "value": ans[p]["intent"]}
                     for p in pids if p in ans], n_quotes)
    voices = " | ".join(f'"{x["text"]}" ({x["value"]:.0f})' for x in k if x.get("value") is not None)
    return f"group mean intent {np.mean(v):.0f}/100 (n={len(v)}); voices: {voices}"


def roundtable(personas, topic: str, *, llm, question: str | None = None, rounds: int = 3,
               scope: str = "segment", n_quotes: int = 3, lang: str | None = None, batch: int = 10,
               extra_instruction: str = "", B: int = 1000, on_event=None) -> dict:
    """scope: 'segment' (each persona hears its own segment) or 'all' (whole table).
    extra_instruction: appended to rounds t>0, e.g. CONFIRMATION_BIAS or KEEP_VIEW."""
    em = Emitter(on_event, "roundtable")
    if not topic:
        raise ValueError("topic is empty")
    ps = as_personas(personas)
    lang = choose_lang(lang, ps)
    question = question or DEFAULT_QUESTION.get(lang, DEFAULT_QUESTION["en"])
    n_rounds = max(1, int(rounds))
    scope = "all" if scope in ("all", "hepsi") else "segment"
    seg = {p.pid: p.seg for p in ps}
    pids = [p.pid for p in ps]
    segs = sorted(set(seg.values()))
    usage = Usage()
    history: list[dict] = []

    for t in range(n_rounds):
        em.stage(f"round {t}", round=t)
        context = None
        if t > 0:
            prev = history[-1]
            summ = {sg: _group_summary(prev, [p for p in pids if scope == "all" or seg[p] == sg], n_quotes)
                    for sg in (segs if scope != "all" else [ALL])}
            context = {}
            for p in pids:
                oz = summ.get(seg[p] if scope != "all" else ALL, "")
                own = prev.get(p)
                parts = []
                if own:
                    parts.append(f"Your previous answer: intent {own['intent']:.0f}, you said \"{own['remark']}\"."
                                 if own["intent"] is not None else f"You previously said \"{own['remark']}\".")
                if oz:
                    parts.append(f"Previous round, {'your group' if scope != 'all' else 'the whole table'}: {oz}")
                if parts:
                    context[p] = " ".join(parts)
        r = ask_panel(llm, ps, _task(topic, question, t, n_rounds, extra_instruction), FIELDS, batch=batch,
                      context=context, lang=lang, usage=usage, tag=f"roundtable:round{t}")
        ans = {pid: {"intent": to_number(x.get("intent"), 0, 100), "remark": x.get("remark")} for pid, x in r.items()}
        for pid, a in ans.items():
            em.row(pid, seg[pid], a.get("remark") or "", a.get("intent"), round=t)
        # a persona with no answer keeps its previous-round view (flagged)
        if t > 0:
            for p in pids:
                if p not in ans and p in history[-1]:
                    ans[p] = {**history[-1][p], "carried": True}
        history.append(ans)

    table = []
    for t, cv in enumerate(history):
        row = {"round": t, "overall": summary_stats([cv[p]["intent"] for p in cv]),
               "overall_ci": bootstrap_ci([cv[p]["intent"] for p in cv], B=B), "segments": {}}
        for sg in segs:
            row["segments"][sg] = summary_stats([cv[p]["intent"] for p in cv if seg[p] == sg])
        table.append(row)
    paths = {p: [history[t].get(p, {}).get("intent") for t in range(n_rounds)] for p in pids}
    shift = [v[-1] - v[0] for v in paths.values() if v and v[0] is not None and v[-1] is not None]
    changed = sum(1 for d in shift if abs(d) >= 5)
    last = history[-1]
    first_o, last_o = table[0]["overall"], table[-1]["overall"]
    out = {
        "runner": "roundtable", "lang": lang, "topic": topic, "question": question, "rounds": n_rounds,
        "scope": scope, "n_personas": len(pids), "n_answered": len(last), "table": table,
        "summary": {
            "first": first_o, "last": last_o, "shift": bootstrap_ci(shift, B=B),
            "sd_change": (last_o["sd"] - first_o["sd"]) if last_o["sd"] is not None and first_o["sd"] is not None
            else None,
            "changed_share": changed / len(shift) if shift else None,
            "segment_shift": {sg: (table[-1]["segments"][sg]["mean"] - table[0]["segments"][sg]["mean"])
                              if table[-1]["segments"][sg]["mean"] is not None
                              and table[0]["segments"][sg]["mean"] is not None else None for sg in segs},
        },
        "paths": paths,
        "quotes": [{"round": t, "segments": {sg: pick_quotes([{"pid": p, "segment": sg, "text": cv[p]["remark"],
                                                               "value": cv[p]["intent"]}
                                                              for p in cv if seg[p] == sg], n_quotes) for sg in segs}}
                   for t, cv in enumerate(history)],
        "answers": {t: cv for t, cv in enumerate(history)},
        "segments": seg,
        "warnings": ([f"{len(pids) - len(last)} personas unanswered in the last round"] if len(last) < len(pids)
                     else []) + ["a falling sd across rounds can be herding; round 0 is the independent measurement"],
        "usage": usage.as_dict(),
    }
    em.done(n=len(last))
    return to_jsonable(out)


def roundtable_frame(result: dict, truth: dict | None = None, level: str | None = None) -> pd.DataFrame:
    """level='person' (default when truth keys are persona ids): one row per persona; level='segment': one row
    per segment (+ 'all'). Columns pred_round0 (independent) and pred_round<last> (after discussion), 0-100."""
    truth = truth or {}
    seg = result.get("segments", {})
    n = int(result.get("rounds", 1))
    if level is None:
        level = "segment" if truth and not (set(map(str, truth)) & set(seg)) else "person"
    ans = {int(k): v for k, v in (result.get("answers") or {}).items()}
    rounds = sorted({0, n - 1})
    rows = []
    if level == "person":
        for p, sg in seg.items():
            t = to_number(truth.get(p)) if truth.get(p) is not None else None
            r = {"unit": p, "truth": np.nan if t is None else t}
            for k in rounds:
                v = (ans.get(k, {}).get(p) or {}).get("intent")
                r[f"pred_round{k}"] = np.nan if v is None else v
            r.update({"segment": sg, "item": "roundtable", "group": p})
            rows.append(r)
    else:
        table = result.get("table", [])
        for sg in sorted(set(seg.values())) + [ALL]:
            g = truth.get(sg)
            t = to_number(g) if g is not None else None
            r = {"unit": sg, "truth": np.nan if t is None else t}
            for k in rounds:
                o = table[k]["overall"] if sg == ALL else table[k]["segments"].get(sg, {})
                r[f"pred_round{k}"] = np.nan if o.get("mean") is None else o.get("mean")
            r.update({"segment": sg, "item": "roundtable", "group": sg})
            rows.append(r)
    return pd.DataFrame(rows)


__all__ = ["roundtable", "roundtable_frame", "CONFIRMATION_BIAS", "KEEP_VIEW"]
