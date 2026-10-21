"""Strategy comparison: 2+ scenarios x segment; each persona rates each scenario 0-100 with a short reason.
Port of sahne/kod/sahne/kosucu/strateji.py.

    res = strategy(personas, [{"id": "S1", "text": "Cut price 10%"}, {"id": "S2", "text": "Free delivery"}],
                   llm=llm, context="online grocery, Istanbul")
    res["ranking"], res["divergence"]["most_divergent"], strategy_frame(res)

Order effect: half of the personas see the scenarios reversed. Output: scenario x segment mean table, overall
ranking with CIs and bootstrap win probability, sign test on the top two, the most divergent segment.
Measured caveat (makro M2/C3): no simulated panel ranking was validated against real price events; on
Dominick's the LLM elasticity prior added nothing beyond shrinkage (LLM-swap placebo gave the same gain).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._stats import (ALL, Emitter, bootstrap_ci, bootstrap_win, field_name, pick_quotes, resolve_choice,
                     sign_test, split_halves, to_jsonable, to_number)
from .panel import Usage, ask_panel
from .personas import as_personas, choose_lang

DEFAULT_QUESTION = {"en": "If this were done, how likely would the person be to buy / sign up / stay (0-100)?",
                    "tr": "Bu yapılsaydı bu kişinin satın alma / abone olma / kalma olasılığı ne olurdu (0-100)?"}


def _scenarios(scenarios):
    v = list(scenarios or [])
    if len(v) < 2:
        raise ValueError("scenarios needs at least 2 entries")
    out, seen = [], set()
    for i, x in enumerate(v):
        x = x if isinstance(x, dict) else {"id": f"S{i + 1}", "text": str(x)}
        sid = str(x.get("id", f"S{i + 1}"))
        label = field_name(sid, i)
        if label in seen:
            label = f"{label}_{i}"
        seen.add(label)
        out.append({"id": sid, "text": x.get("text", x.get("metin", "")), "label": label})
    return out


def _task(order, context, question):
    g = "Strategy comparison." + (f" Context: {context}\n" if context else "\n")
    for s in order:
        g += f"\nScenario {s['label']}:\n{s['text']}\n"
    g += (f"\nQuestion: {question}\nRate each scenario separately for each person (0 = certainly not, 100 = certainly), "
          "give a very short reason per scenario, and name the scenario the person would prefer.")
    f = {}
    for s in order:
        f[f"intent_{s['label']}"] = "int 0-100"
        f[f"reason_{s['label']}"] = "str: max 15 words"
    f["preferred"] = "one of " + "|".join(s["label"] for s in order)
    return g, f


def strategy(personas, scenarios, *, llm, context: str = "", question: str | None = None, truth=None,
             lang: str | None = None, batch: int = 8, B: int = 1000, on_event=None) -> dict:
    """truth (optional): {scenario: observed value} -> direction check against the panel winner."""
    em = Emitter(on_event, "strategy")
    ps = as_personas(personas)
    sc = _scenarios(scenarios)
    ids = [s["id"] for s in sc]
    lab = {s["label"]: s["id"] for s in sc}
    lang = choose_lang(lang, ps)
    question = question or DEFAULT_QUESTION.get(lang, DEFAULT_QUESTION["en"])
    seg = {p.pid: p.seg for p in ps}
    pids = [p.pid for p in ps]
    usage = Usage()

    ans = {}
    em.stage("persona", n=len(ps))
    for half, order, name in zip(split_halves(ps), (sc, sc[::-1]), ("forward", "reverse")):
        if not half:
            continue
        task, fields = _task(order, context, question)
        r = ask_panel(llm, half, task, fields, batch=batch, lang=lang, usage=usage, tag=f"strategy:{name}")
        for pid, x in r.items():
            ans[pid] = {"intent": {s["id"]: to_number(x.get(f"intent_{s['label']}"), 0, 100) for s in sc},
                        "reason": {s["id"]: x.get(f"reason_{s['label']}") for s in sc},
                        "preferred": lab.get(resolve_choice(x.get("preferred"), list(lab))), "order": name}
            em.row(pid, seg[pid], " · ".join(f"{k}={v}" for k, v in ans[pid]["intent"].items())
                   + f" -> {ans[pid]['preferred']}", ans[pid]["preferred"])

    def matrix(pp):
        pp = [p for p in pp if p in ans]
        return np.array([[ans[p]["intent"][i] if ans[p]["intent"][i] is not None else np.nan for i in ids]
                         for p in pp], dtype=float).reshape(len(pp), len(ids))

    M = matrix(pids)
    win = bootstrap_win(M, B=B) if len(M) else []
    overall = {i: {**bootstrap_ci(M[:, j].tolist(), B=B), "win_prob": win[j] if win else None,
                   "preference_share": (sum(1 for c in ans.values() if c["preferred"] == i) / len(ans)) if ans
                   else None}
               for j, i in enumerate(ids)}
    ranking = sorted([i for i in ids if overall[i]["value"] is not None], key=lambda i: -overall[i]["value"])
    segs = sorted(set(seg.values()))
    table, div = {}, {}
    for sg in segs:
        Ms = matrix([p for p in pids if seg[p] == sg])
        table[sg] = {i: bootstrap_ci(Ms[:, j].tolist(), B=B) for j, i in enumerate(ids)}
        table[sg]["_winner"] = max((i for i in ids if table[sg][i]["value"] is not None),
                                   key=lambda i: table[sg][i]["value"], default=None)
        # divergence: mean absolute deviation of the segment's scenario profile from the overall one,
        # level difference removed (profile, not level)
        d = [(table[sg][i]["value"] - overall[i]["value"]) for i in ids
             if table[sg][i]["value"] is not None and overall[i]["value"] is not None]
        if d:
            f = np.array(d) - np.mean(d)
            div[sg] = float(np.abs(f).mean())
    most = max(div, key=div.get) if div else None
    top2 = None
    if len(ranking) >= 2:
        a, b = ids.index(ranking[0]), ids.index(ranking[1])
        dd = [x for x in (M[:, a] - M[:, b]) if not np.isnan(x)]
        top2 = {"a": ranking[0], "b": ranking[1], **sign_test(dd), "diff": bootstrap_ci(dd, B=B)}
    out = {
        "runner": "strategy", "lang": lang, "question": question, "n_personas": len(pids), "n_answered": len(ans),
        "scenarios": [{"id": s["id"], "text": s["text"]} for s in sc],
        "overall": overall, "ranking": ranking, "winner": ranking[0] if ranking else None,
        "top_two": top2, "table": table,
        "divergence": {"segments": div, "most_divergent": most,
                       "most_divergent_winner": table.get(most, {}).get("_winner") if most else None,
                       "description": "how far the segment's scenario profile deviates from the overall one "
                                      "(level difference removed, points)"},
        "quotes": {i: pick_quotes([{"pid": p, "segment": seg[p], "text": c["reason"].get(i),
                                    "value": c["intent"].get(i)} for p, c in ans.items()], 3) for i in ids},
        "answers": ans, "segments": seg,
        "warnings": [f"{len(pids) - len(ans)} personas unanswered"] if len(ans) < len(pids) else [],
        "usage": usage.as_dict(),
    }
    if isinstance(truth, dict):
        ts = {k: to_number(v) for k, v in truth.items() if k in ids and to_number(v) is not None}
        if len(ts) >= 2:
            tw = max(ts, key=ts.get)
            out["truth_comparison"] = {"truth": ts, "true_winner": tw, "direction_correct": out["winner"] == tw}
    em.done(n=len(ans), winner=out["winner"])
    return to_jsonable(out)


def strategy_frame(result: dict, truth: dict | None = None) -> pd.DataFrame:
    """One row per scenario x segment ('all' included). truth {scenario: value} or {scenario: {segment: value}}.
    Columns pred_intent (mean 0-100), pred_win (only on 'all' rows)."""
    if truth is None:
        truth = (result.get("truth_comparison") or {}).get("truth") or {}
    rows = []
    for s in result.get("scenarios", []):
        i = s["id"]
        g = truth.get(i)
        for sg in [ALL] + sorted(result.get("table", {})):
            o = result["overall"][i] if sg == ALL else result["table"][sg].get(i, {})
            gv = g.get(sg) if isinstance(g, dict) else (g if sg == ALL else None)
            gv = to_number(gv)
            w = result["overall"][i].get("win_prob") if sg == ALL else None
            rows.append({"unit": f"{i}|{sg}", "truth": np.nan if gv is None else gv,
                         "pred_intent": np.nan if o.get("value") is None else o.get("value"),
                         "pred_win": np.nan if w is None else w,
                         "segment": sg, "item": f"strategy|{sg}", "group": sg})
    return pd.DataFrame(rows)


__all__ = ["strategy", "strategy_frame"]
