"""Multi-round roundtable: segmented panel, N discussion rounds, intent per round.
Port of sahne/kod/sahne/kosucu/roundtable.py; the optional stance instructions come from
makro/llmabm/kod/c1/masa.py (CONFIRMATION_BIAS = Chuang et al. 2024 style, KEEP_VIEW).

Round 0: everyone answers alone (intent 0-100 + a short remark for the table). Round t>0: each persona sees its
own previous answer and the previous-round summary of its segment (or of the whole table): mean intent and
2-3 quotes. Output: round x segment mean/sd (a falling sd can be homogenisation / herding), shift, persona
paths, quotes. Measured caveat (makro C1): plain discussion rounds changed 5-8% of answers and did not move
the distribution closer to the real survey; round 0 is the independent measurement.

Strategy comparison at the table: `roundtable_strategy` runs one roundtable per condition on the same panel
(condition fields hidden from the prompt) and returns condition x round x segment means, the paired effect of
each condition vs the control table, and the paired round-0 vs last-round shift with bootstrap CIs.
Not re-exported from `simengine.sim` yet; import from here (and the printer from `_stats`):

    from simengine.sim.roundtable import roundtable_strategy, roundtable_strategy_frame
    from simengine.sim._stats import print_events
    r = roundtable_strategy(personas, {"Control": "...", "Discount": "...", "Story": "..."}, llm=llm, rounds=3,
                            on_event=print_events)
    r["effects"]        # [{condition, round, segment, effect, low, high, n}] vs the Control table
    r["shift"]          # [{condition, segment, value, low, high, n, effect_shift...}] last round - round 0
    roundtable_strategy_frame(r, truth={"Discount": 4.1, "Story": -0.5}, effect=True)   # RCT effects as truth
"""
from __future__ import annotations

import warnings as _warnings
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ._stats import ALL, Emitter, bootstrap_ci, pick_quotes, summary_stats, to_jsonable, to_number
from .panel import Usage, ask_panel
from .personas import PersonaView, as_persona, as_personas, choose_lang
from .strategy import _scenarios

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


# ---------------------------------------------------------------- strategy comparison at the table
_TRAIT_KEYS = ("label", "key", "name", "field")


def _prune(x: Any, fields: set) -> Any:
    """Drop keys named in `fields` (lower case; matched case-insensitively) at any depth; drop list items that
    are {label|key|name|field: <field>} dicts."""
    if isinstance(x, dict):
        return {k: _prune(v, fields) for k, v in x.items() if str(k).lower() not in fields}
    if isinstance(x, list):
        return [_prune(v, fields) for v in x
                if not (isinstance(v, dict) and any(str(v.get(k)).lower() in fields for k in _TRAIT_KEYS
                                                    if v.get(k) is not None))]
    return x


def hide_persona_fields(personas: Iterable[Any], fields: Iterable[str] = ("condition",)) -> list:
    """Persona-like items -> PersonaViews whose prompt text no longer shows `fields` (dict keys at any depth,
    traits with that label, case-insensitive). Id, segment and source are kept from the original item. A
    PersonaView is rebuilt from its dict / to_dict() source; a field still visible in the text afterwards (plain
    strings, text-only objects, a field written into free text) gives a warning."""
    fields = {str(f).lower() for f in (fields or ())}
    out, left = [], set()
    for i, x in enumerate(personas):
        v = as_persona(x, i)
        src = x.source if isinstance(x, PersonaView) else x
        d = src if isinstance(src, dict) else (src.to_dict() if callable(getattr(src, "to_dict", None)) else None)
        if fields and isinstance(d, dict):
            h = as_persona(_prune(d, fields), i)
            h.pid, h.segment, h.source = v.pid, v.segment, v.source
            v = h
        low = v.text().lower()
        left |= {f for f in fields if f"{f}:" in low or f"{f}=" in low}
        out.append(v)
    if left:
        _warnings.warn(f"persona text still shows {sorted(left)} after hide_fields (free text or a persona without a "
                       "dict form); remove it before the run", stacklevel=2)
    return out


def _intents(run: dict, t: int) -> dict:
    return {p: a.get("intent") for p, a in ((run.get("answers") or {}).get(str(t)) or {}).items()}


def _paired(a: dict, b: dict, pids, B: int) -> dict:
    """Mean of a[p] - b[p] over personas answered in both, persona bootstrap CI."""
    return bootstrap_ci([a[p] - b[p] for p in pids if a.get(p) is not None and b.get(p) is not None], B=B)


def roundtable_strategy(personas, scenarios, *, llm, control: str | None = "Control", rounds: int = 3,
                        scope: str = "segment", hide_fields: Iterable[str] = ("condition",), B: int = 1000,
                        on_event=None, **roundtable_kw) -> dict:
    """One roundtable per condition on the same panel; which strategy wins at an interacting table.

    scenarios: {id: text} or a list of str / {id, text} dicts (ids S1, S2, ... by position), at least 2.
    control: id of the control condition (effects are paired per persona: condition table - control table);
      None or an id that is not in `scenarios` skips the effect table (with a warning).
    hide_fields: persona fields kept out of the prompt (e.g. the RCT arm the person was in).
    roundtable_kw: passed to `roundtable` (question, lang, n_quotes, batch, extra_instruction).
    Output: `table` [{condition, round, segment, mean, sd, n}] ('all' = whole table), `effects`
    [{condition, round, segment, effect, low, high, n}], `shift` [{condition, segment, value, low, high, n}] =
    last round - round 0 per persona (+ `effect_shift*` = change of the paired effect), `ranking` by overall mean
    in round 0 and in the last round, `runs` {condition: roundtable result}. Each condition's table is a separate
    conversation; CIs resample personas and ignore the within-table dependence of later rounds."""
    em = Emitter(on_event, "roundtable_strategy")
    sc = _scenarios(scenarios)
    ids = [s["id"] for s in sc]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate condition ids in {ids}")
    ps = hide_persona_fields(personas, hide_fields)
    if not ps:
        raise ValueError("no personas")
    n_rounds = max(1, int(rounds))
    seg = {p.pid: p.seg for p in ps}
    pids = [p.pid for p in ps]
    segs = sorted(set(seg.values()))
    groups = {ALL: pids, **{sg: [p for p in pids if seg[p] == sg] for sg in segs}}
    warnings = []
    if control is not None and control not in ids:
        warnings.append(f"control {control!r} is not a condition; no effect table")
        _warnings.warn(warnings[-1] + f" (conditions: {ids})", stacklevel=2)
    ctrl = control if control in ids else None

    runs = {}
    for k, s in enumerate(sc):
        em.stage(f"condition {s['id']}", condition=s["id"], index=k, of=len(sc))
        runs[s["id"]] = roundtable(ps, s["text"], llm=llm, rounds=n_rounds, scope=scope, B=B, on_event=on_event,
                                   **roundtable_kw)
    last = n_rounds - 1
    iv = {c: [_intents(runs[c], t) for t in range(n_rounds)] for c in ids}
    for c in ids:       # surface per-table problems (the herding note is repeated once below)
        warnings += [f"{c}: {w}" for w in runs[c].get("warnings", []) if "herding" not in w]
        carried = [p for p, a in ((runs[c].get("answers") or {}).get(str(last)) or {}).items() if a.get("carried")]
        if carried and n_rounds > 1:
            warnings.append(f"{c}: {len(carried)} personas did not answer the last round; their earlier answer "
                            "is carried into shift / effects")

    table = [{"condition": c, "round": t, "segment": g, **summary_stats([iv[c][t].get(p) for p in gp])}
             for c in ids for t in range(n_rounds) for g, gp in groups.items()]
    effects, shift = [], []
    for c in ids:
        for g, gp in groups.items():
            row = {"condition": c, "segment": g, **_ci_row(_paired(iv[c][last], iv[c][0], gp, B))}
            if ctrl is not None and c != ctrl:
                d0 = {p: _sub(iv[c][0].get(p), iv[ctrl][0].get(p)) for p in gp}
                d1 = {p: _sub(iv[c][last].get(p), iv[ctrl][last].get(p)) for p in gp}
                row.update({f"effect_shift_{k}": v for k, v in _ci_row(_paired(d1, d0, gp, B)).items()})
            shift.append(row)
            if ctrl is None or c == ctrl:
                continue
            for t in range(n_rounds):
                e = _paired(iv[c][t], iv[ctrl][t], gp, B)
                effects.append({"condition": c, "round": t, "segment": g, "effect": e["value"], "low": e["low"],
                                "high": e["high"], "n": e["n"]})

    def rank(t):
        m = {r["condition"]: r["mean"] for r in table if r["round"] == t and r["segment"] == ALL}
        return sorted([c for c in ids if m.get(c) is not None], key=lambda c: -m[c])

    usage: dict = {}
    for r in runs.values():
        for k, v in (r.get("usage") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                usage[k] = usage.get(k, 0) + v
            elif k == "model" and v:
                usage[k] = v
    out = {"runner": "roundtable_strategy", "conditions": [{"id": s["id"], "text": s["text"]} for s in sc],
           "control": ctrl, "rounds": n_rounds, "scope": runs[ids[0]]["scope"], "n_personas": len(pids),
           "segments": seg, "hidden_fields": sorted(set(hide_fields or ())), "table": table, "effects": effects,
           "shift": shift, "ranking": {"round0": rank(0), "last": rank(last)}, "runs": runs,
           "warnings": warnings + ["each condition is a separate table; round 0 is the independent measurement; a "
                                   "falling sd across rounds can be herding"],
           "usage": usage}
    em.done(n=len(ids))
    return to_jsonable(out)


def _sub(a, b):
    return None if a is None or b is None else a - b


def _ci_row(ci: dict) -> dict:
    return {"value": ci["value"], "low": ci["low"], "high": ci["high"], "n": ci["n"]}


def roundtable_strategy_frame(result: dict, truth: dict | None = None, level: str | None = None,
                              effect: bool = False) -> pd.DataFrame:
    """EvalFrame over conditions via `roundtable_frame`: unit '<condition>|<unit>', item = condition.
    truth: {condition: value} (whole-table value, row 'all'), {condition: {segment: value}} or
    {condition: {persona id: value}}. effect=True subtracts the control condition's predictions per unit
    (pred columns become condition - control, control rows dropped); give truth as observed effects then."""
    truth = truth or {}
    runs = result.get("runs") or {}
    ctrl = result.get("control")
    if effect and ctrl not in runs:
        raise ValueError("effect=True needs a result with a control condition")

    seg = set(result.get("segments") or {})
    given = [v for v in truth.values() if v is not None]
    if level is None:       # scalar truth or segment keys -> segment rows; persona-id keys -> person rows
        level = ("segment" if any(not isinstance(v, dict) or not set(map(str, v)) & seg for v in given)
                 else "person")

    def one(c):
        tv = truth.get(c)
        return roundtable_frame(runs[c], truth=tv if isinstance(tv, dict) else ({ALL: tv} if tv is not None
                                                                              else None), level=level)

    base = one(ctrl) if effect else None
    frames = []
    for c in [x["id"] for x in result.get("conditions", [])]:
        if effect and c == ctrl:
            continue
        f = one(c)
        if effect:
            pc = [k for k in f.columns if k.startswith("pred_")]
            b = base.set_index("unit")[pc].reindex(f["unit"]).to_numpy()
            f[pc] = f[pc].to_numpy(dtype=float) - b
        f["condition"] = c
        f["item"] = c
        f["unit"] = c + "|" + f["unit"].astype(str)
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


roundtable_conditions = roundtable_strategy     # alias: same runner, condition wording

__all__ = ["roundtable", "roundtable_frame", "roundtable_strategy", "roundtable_strategy_frame",
           "roundtable_conditions", "hide_persona_fields", "CONFIRMATION_BIAS", "KEEP_VIEW"]
