"""Strategy comparison: 2+ scenarios x segment; each persona rates each scenario 0-100 with a short reason.
Port of sahne/kod/sahne/kosucu/strateji.py.

    res = strategy(personas, [{"id": "S1", "text": "Cut price 10%"}, {"id": "S2", "text": "Free delivery"}],
                   llm=llm, context="online grocery, Istanbul")
    res = strategy(personas, {"price": "Cut price 10%", "ship": "Free delivery"}, llm=llm)   # {id: text}
    res = strategy(personas, variants=["Cut price 10%", "Free delivery"], llm=llm)           # ab_test keyword
    res["ranking"], res["divergence"]["most_divergent"], res["arms"]["no_persona"], strategy_frame(res)

Order effect: half of the personas see the scenarios reversed. Output: scenario x segment mean table, overall
ranking with CIs and bootstrap win probability, sign test on the top two, the most divergent segment.
Control arms (no_persona, persona_swap) run by default, with the same semantics as `ab_test` / `survey`.
Measured caveat (makro M2/C3): no simulated panel ranking was validated against real price events; on
Dominick's the LLM elasticity prior added nothing beyond shrinkage (LLM-swap placebo gave the same gain).
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ._stats import (ALL, Emitter, bootstrap_ci, bootstrap_win, field_name, pick_quotes, resolve_choice,
                     sign_test, split_halves, to_jsonable, to_number)
from .panel import Usage, ask_panel
from .personas import as_personas, choose_lang, control_personas, control_plan, normalize_controls

DEFAULT_QUESTION = {"en": "If this were done, how likely would the person be to buy / sign up / stay (0-100)?",
                    "tr": "Bu yapılsaydı bu kişinin satın alma / abone olma / kalma olasılığı ne olurdu (0-100)?"}


def _scenarios(scenarios):
    """Normalise to [{id, text, label}]. Accepts a list of str / {id, text} dicts or a {id: text} mapping;
    missing ids become S1, S2, ... by position."""
    if isinstance(scenarios, Mapping):
        scenarios = [dict(v, id=v.get("id", k)) if isinstance(v, Mapping) else {"id": k, "text": v}
                     for k, v in scenarios.items()]
    elif isinstance(scenarios, (str, bytes)) or not hasattr(scenarios, "__iter__"):
        raise TypeError(f"scenarios must be a list of str / {{id, text}} dicts or a {{id: text}} mapping, "
                        f"not {type(scenarios).__name__}")
    v = list(scenarios)
    if len(v) < 2:
        raise ValueError("scenarios needs at least 2 entries")
    out, seen = [], set()
    for i, x in enumerate(v):
        if isinstance(x, str):
            x = {"id": f"S{i + 1}", "text": x}
        elif not isinstance(x, Mapping):
            raise TypeError(f"scenario {i} must be a str or an {{id, text}} dict, not {type(x).__name__}")
        sid = str(x.get("id", f"S{i + 1}"))
        label = field_name(sid, i)
        if label in seen:
            label = f"{label}_{i}"
        seen.add(label)
        out.append({"id": sid, "text": str(x.get("text", x.get("metin", ""))), "label": label})
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


def _arm(llm, personas, sc, context, question, lang, batch, tag, usage) -> dict:
    out = {}
    lab = {s["label"]: s["id"] for s in sc}
    for half, order, name in zip(split_halves(personas), (sc, sc[::-1]), ("forward", "reverse")):
        if not half:
            continue
        task, fields = _task(order, context, question)
        r = ask_panel(llm, half, task, fields, batch=batch, lang=lang, usage=usage, tag=f"{tag}:{name}")
        for pid, x in r.items():
            out[pid] = {"intent": {s["id"]: to_number(x.get(f"intent_{s['label']}"), 0, 100) for s in sc},
                        "reason": {s["id"]: x.get(f"reason_{s['label']}") for s in sc},
                        "preferred": lab.get(resolve_choice(x.get("preferred"), list(lab))), "order": name}
    return out


def _arm_summary(cv: dict, ids, pids, B: int) -> dict:
    pp = [p for p in pids if p in cv]
    mean = {}
    for i in ids:
        v = [cv[p]["intent"][i] for p in pp if cv[p]["intent"].get(i) is not None]
        mean[i] = bootstrap_ci(v, B=B)
    ranking = sorted([i for i in ids if mean[i]["value"] is not None], key=lambda i: -mean[i]["value"])
    return {"n": len(pp), "intent": mean, "ranking": ranking, "winner": ranking[0] if ranking else None}


def strategy(personas, scenarios=None, *, variants=None, llm, context: str = "", question: str | None = None,
             truth=None, lang: str | None = None, batch: int = 8, controls=None, B: int = 1000,
             on_event=None) -> dict:
    """Rank 2+ scenarios on a persona panel; each persona gives a 0-100 intent and a reason per scenario.

    scenarios: at least 2, as a list whose items are a str (id auto S1, S2, ... by position) or an
        {"id", "text"} dict (legacy key "metin" also read), or a {id: text} mapping. Anything else -> TypeError.
    variants: keyword-only alias of `scenarios` (the `ab_test` name); give one of the two, not both.
    context: shared text shown before the scenarios. question: default DEFAULT_QUESTION[lang].
    truth: optional {scenario: observed value} -> `truth_comparison` (direction check against the panel winner)
        and `arms_direction_correct` per arm.
    lang: "en" / "tr"; default from the personas. batch: personas per prompt (default 8).
    controls: control arms, same semantics and default as `ab_test` / `survey`: None runs no_persona and
        persona_swap (call budget from `control_plan`), [] runs none (a warning says so).
    B: bootstrap resamples. on_event: progress callback (Emitter).

    Returns a JSON-safe dict: ranking (ids by mean intent), winner, overall {id: mean, CI, win_prob,
    preference_share}, top_two (sign test + bootstrap diff on the top two), divergence {segments,
    most_divergent, most_divergent_winner}, table {segment: {id: CI, "_winner"}}, quotes, answers (persona arm),
    arms {arm: {n, intent, ranking, winner}}, controls {arms, plan, answers}, warnings, usage, and
    truth_comparison when `truth` is given."""
    if variants is not None:
        if scenarios is not None:
            raise TypeError("give scenarios or variants, not both")
        scenarios = variants
    if scenarios is None:
        raise TypeError("strategy() needs scenarios (or variants=)")
    em = Emitter(on_event, "strategy")
    ps = as_personas(personas)
    sc = _scenarios(scenarios)
    ids = [s["id"] for s in sc]
    lang = choose_lang(lang, ps)
    question = question or DEFAULT_QUESTION.get(lang, DEFAULT_QUESTION["en"])
    seg = {p.pid: p.seg for p in ps}
    pids = [p.pid for p in ps]
    usage = Usage()

    em.stage("persona", n=len(ps))
    ans = _arm(llm, ps, sc, context, question, lang, batch, "strategy", usage)
    for pid, c in ans.items():
        em.row(pid, seg[pid], " · ".join(f"{k}={v}" for k, v in c["intent"].items()) + f" -> {c['preferred']}",
               c["preferred"], arm="persona")
    arms = {"persona": ans}
    ctrl = normalize_controls(controls)
    plan = control_plan(len(ps), batch, ctrl, splits=2)
    for arm in ctrl:
        cp = control_personas(arm, ps, lang, plan[arm]["n"])
        em.stage(arm, n=len(cp), calls=plan[arm]["calls"])
        arms[arm] = _arm(llm, cp, sc, context, question, lang, plan[arm]["batch"], f"strategy:{arm}",
                         usage) if cp else {}
    arm_summary = {arm: _arm_summary(cv, ids, pids, B) for arm, cv in arms.items()}

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
    warnings = []
    if len(ans) < len(pids):
        warnings.append(f"{len(pids) - len(ans)} personas unanswered")
    if not ctrl:
        warnings.append("no persona-control arms run (controls=[]); the ranking may not depend on the personas")
    winner = ranking[0] if ranking else None
    for arm in ctrl:
        o = arm_summary[arm]
        if not o["n"]:
            warnings.append(f"{arm} arm got no answers")
        elif winner and o["winner"] and o["winner"] != winner:
            warnings.append(f"{arm} arm picked {o['winner']} (main arm {winner}); the ranking depends on "
                            "persona assignment, low confidence")
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
        "arms": arm_summary,
        "controls": {"arms": ctrl, "plan": plan, "answers": {k: v for k, v in arms.items() if k != "persona"}},
        "warnings": warnings, "usage": usage.as_dict(),
    }
    if isinstance(truth, dict):
        ts = {k: to_number(v) for k, v in truth.items() if k in ids and to_number(v) is not None}
        if len(ts) >= 2:
            tw = max(ts, key=ts.get)
            out["truth_comparison"] = {"truth": ts, "true_winner": tw, "direction_correct": out["winner"] == tw}
            out["arms_direction_correct"] = {arm: o["winner"] == tw for arm, o in arm_summary.items()}
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
