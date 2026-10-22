"""A/B variant comparison, PAARS template: every persona scores every variant 1-10, picks one, gives a reason.
Port of sahne/kod/sahne/kosucu/ab.py (single test, controls, confidence check), sahne/prova/upworthy.py
(many pairwise tests per call, `ab_many`) and deney/kod/kit/persona_talp.py (`paars_decision`).

    res = ab_test(personas, [{"id": "A", "text": "..."}, {"id": "B", "text": "..."}], llm=llm,
                  context="push notification on Friday 18:00", truth={"A": 0.012, "B": 0.019})
    res["tendency"], res["confidence"]["low"], ab_frame(res)

Order effect: half of the personas see the variants forward, half reversed; `position_effect` is reported.
The result is a panel *tendency*, not a predicted winner: on 300 real Upworthy A/B tests an 8-persona Haiku
panel matched the real winner 0.522 [0.468; 0.575] of the time (chance 0.50); see docs/modules/sim.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._stats import (ALL, Emitter, bootstrap_ci, bootstrap_distribution, bootstrap_win, field_name, pick_quotes,
                     resolve_choice, sign_test, split_halves, to_jsonable, to_number)
from .panel import REFRAME_NOTE_GENERIC, Usage, ask_many, ask_panel
from .parse import parse_json
from .personas import as_personas, choose_lang, control_personas, control_plan, normalize_controls

DEFAULT_QUESTION = {"en": "How likely would you be to click on / engage with each one?",
                    "tr": "Her birine tıklama / ilgilenme olasılığın ne kadar?"}
LOW_CONFIDENCE = {"tr": "Karar persona atamasına bağlı, güven düşük",
                  "en": "Decision depends on persona assignment; low confidence"}
SHARE_MARGIN = 0.10        # top two preference shares within this margin count as a tie
P_THRESHOLD = 0.05


def _variants(variants):
    v = list(variants or [])
    if len(v) < 2:
        raise ValueError("variants needs at least 2 entries")
    out, seen = [], set()
    for i, x in enumerate(v):
        x = x if isinstance(x, dict) else {"id": chr(65 + i), "text": str(x)}
        vid = str(x.get("id", chr(65 + i)))
        label = field_name(vid, i)
        if label in seen:
            label = f"{label}_{i}"
        seen.add(label)
        out.append({"id": vid, "text": x.get("text", x.get("metin", "")), "label": label})
    return out


def _task(order, context, question):
    g = "A/B test. Each person sees the following variants"
    g += (f" in this context: {context}\n" if context else ".\n")
    for v in order:
        g += f"\nVariant {v['label']}:\n{v['text']}\n"
    g += (f"\nQuestion: {question}\nFor each variant give a 1-10 score (1 = certainly not, 10 = certainly yes), "
          "then pick the single variant the person would choose, and one sentence explaining why "
          "(from their own needs, budget and taste).")
    fields = {f"score_{v['label']}": "int 1-10" for v in order}
    fields["choice"] = "one of " + "|".join(v["label"] for v in order)
    fields["reason"] = "str: one sentence"
    return g, fields


def _arm(llm, personas, variants, context, question, lang, batch, tag, usage) -> dict:
    out = {}
    labels = [v["label"] for v in variants]
    for half, order, name in zip(split_halves(personas), (variants, variants[::-1]), ("forward", "reverse")):
        if not half:
            continue
        task, fields = _task(order, context, question)
        r = ask_panel(llm, half, task, fields, batch=batch, lang=lang, usage=usage, tag=f"{tag}:{name}")
        for pid, x in r.items():
            score = {v["id"]: to_number(x.get(f"score_{v['label']}"), 1, 10) for v in variants}
            s = resolve_choice(x.get("choice"), labels)
            choice = next((v["id"] for v in variants if v["label"] == s), None)
            if choice is None and all(p is None for p in score.values()):
                continue
            out[pid] = {"score": score, "choice": choice, "reason": x.get("reason"), "order": name}
    return out


def _summary(cv: dict, variants, pids, B: int) -> dict:
    ids = [v["id"] for v in variants]
    pids = [p for p in pids if p in cv]
    M = np.array([[cv[p]["score"].get(i) if cv[p]["score"].get(i) is not None else np.nan for i in ids]
                  for p in pids], dtype=float).reshape(len(pids), len(ids))
    choices = [cv[p]["choice"] for p in pids if cv[p].get("choice") in ids]
    share = bootstrap_distribution([np.eye(len(ids))[ids.index(s)] for s in choices], B=B)
    win = bootstrap_win(M, B=B) if len(pids) else []
    var = {}
    for j, i in enumerate(ids):
        var[i] = {"score": bootstrap_ci(M[:, j].tolist() if len(pids) else [], B=B),
                  "share": share["value"][j] if share["n"] else None,
                  "share_ci": [share["low"][j], share["high"][j]] if share["n"] else None,
                  "win_prob": win[j] if win else None}
    mean = {i: var[i]["score"]["value"] for i in ids}
    ranking = sorted([i for i in ids if mean[i] is not None], key=lambda i: -mean[i])
    r = {"variants": var, "n": len(pids), "ranking": ranking, "winner": ranking[0] if ranking else None}
    if share["n"]:
        r["choice_winner"] = ids[int(np.argmax(share["value"]))]
    if len(ranking) >= 2:
        a, b = ids.index(ranking[0]), ids.index(ranking[1])
        r["sign_test"] = {"a": ranking[0], "b": ranking[1],
                          **sign_test([x for x in (M[:, a] - M[:, b]) if not np.isnan(x)])}
    return r


def _direction(summary: dict, truth, ids) -> dict | None:
    if truth is None:
        return None
    if isinstance(truth, dict) and "winner" in truth:
        truth = truth["winner"]
    if isinstance(truth, str):
        tw = truth
        r = {"true_winner": tw}
    else:
        ts = {str(k): to_number(v) for k, v in truth.items() if str(k) in ids and to_number(v) is not None}
        if len(ts) < 2:
            return None
        tw = max(ts, key=ts.get)
        r = {"true_winner": tw, "truth": ts}
        common = [i for i in ids if i in ts and summary["variants"][i]["score"]["value"] is not None]
        if len(common) >= 3:
            from scipy.stats import spearmanr
            r["spearman"] = float(spearmanr([ts[i] for i in common],
                                            [summary["variants"][i]["score"]["value"] for i in common]).statistic)
    r["predicted_winner"] = summary.get("winner")
    r["direction_correct"] = summary.get("winner") == tw
    return r


def confidence_check(summaries: dict, ids: list[str] | None = None) -> dict:
    """How robust the panel tendency is (not a winner claim). Low when any of: top two preference shares within
    ±10 points, sign test p > 0.05, or a control arm (no_persona / persona_swap) picks a different winner."""
    g = summaries.get("persona") or {}
    reasons = []
    shares = sorted([v["share"] for v in (g.get("variants") or {}).values() if v.get("share") is not None],
                    reverse=True)
    if len(shares) >= 2 and shares[0] - shares[1] <= SHARE_MARGIN + 1e-9:
        reasons.append({"code": "close_shares", "gap": shares[0] - shares[1],
                        "en": f"preference shares within ±{SHARE_MARGIN * 100:.0f}% ({shares[0]:.2f} / {shares[1]:.2f})",
                        "tr": f"tercih payları ±%{SHARE_MARGIN * 100:.0f} içinde ({shares[0]:.2f} / {shares[1]:.2f})"})
    st = g.get("sign_test")
    if st and st.get("p") is not None and st["p"] > P_THRESHOLD:
        reasons.append({"code": "sign_test_p", "p": st["p"], "en": f"sign test p = {st['p']:.2f} > {P_THRESHOLD}",
                        "tr": f"işaret testi p = {st['p']:.2f} > {P_THRESHOLD}"})
    for arm in ("no_persona", "persona_swap"):
        o = summaries.get(arm)
        if o and o.get("n") and o.get("winner") and g.get("winner") and o["winner"] != g["winner"]:
            reasons.append({"code": f"arm_{arm}", "arm": arm, "winner": o["winner"],
                            "en": f"{arm} arm picked {o['winner']} (main arm {g['winner']})",
                            "tr": f"{arm} kolu {o['winner']} seçti (ana kol {g['winner']})"})
    return {"low": bool(reasons), "reasons": reasons, "warning": LOW_CONFIDENCE if reasons else None,
            "thresholds": {"share": SHARE_MARGIN, "p": P_THRESHOLD}}


def ab_test(personas, variants, *, llm, context: str = "", question: str | None = None, truth=None,
            test_id: str = "ab", lang: str | None = None, batch: int = 10, controls=None, B: int = 1000,
            on_event=None) -> dict:
    """PAARS A/B panel on one test. `truth`: {variant: CTR/conversion} or the winning id (optional)."""
    em = Emitter(on_event, "ab")
    ps = as_personas(personas)
    vs = _variants(variants)
    lang = choose_lang(lang, ps)
    question = question or DEFAULT_QUESTION.get(lang, DEFAULT_QUESTION["en"])
    seg = {p.pid: p.seg for p in ps}
    ids = [v["id"] for v in vs]
    usage = Usage()

    em.stage("persona", n=len(ps))
    arms = {"persona": _arm(llm, ps, vs, context, question, lang, batch, "ab", usage)}
    for pid, c in arms["persona"].items():
        em.row(pid, seg[pid], f"{c['choice']} ({' · '.join(f'{k}={v}' for k, v in c['score'].items())}) "
               f"{c.get('reason') or ''}", c["choice"], arm="persona")
    ctrl = normalize_controls(controls)
    plan = control_plan(len(ps), batch, ctrl, splits=2)
    for arm in ctrl:
        cp = control_personas(arm, ps, lang, plan[arm]["n"])
        em.stage(arm, n=len(cp), calls=plan[arm]["calls"])
        arms[arm] = _arm(llm, cp, vs, context, question, lang, plan[arm]["batch"], f"ab:{arm}", usage) if cp else {}

    pids = [p.pid for p in ps]
    main = arms["persona"]
    summaries = {arm: _summary(cv, vs, pids, B) for arm, cv in arms.items()}
    overall = summaries["persona"]
    seg_summary = {}
    for sg in sorted({seg[p] for p in pids}):
        o = _summary(main, vs, [p for p in pids if seg[p] == sg], B)
        seg_summary[sg] = {"winner": o["winner"], "n": o["n"],
                           "score": {i: o["variants"][i]["score"]["value"] for i in ids},
                           "win_prob": {i: o["variants"][i]["win_prob"] for i in ids}}
    first, last = ids[0], ids[-1]
    diff = []
    for p, c in main.items():
        a, b = c["score"].get(first), c["score"].get(last)
        if a is not None and b is not None:
            diff.append((a - b) if c["order"] == "forward" else (b - a))
    warnings = []
    if len(main) < len(pids):
        warnings.append(f"{len(pids) - len(main)} personas unanswered")
    conf = confidence_check(summaries, ids)
    if conf["low"]:
        warnings.append(LOW_CONFIDENCE["en"] + ": " + "; ".join(n["en"] for n in conf["reasons"]))
    out = {
        "runner": "ab", "confidence": conf, "tendency": overall["winner"], "lang": lang, "test": test_id,
        "question": question, "n_personas": len(pids), "n_answered": len(main),
        "variants": [{"id": v["id"], "text": v["text"]} for v in vs],
        "overall": overall, "winner": overall["winner"], "segment_summary": seg_summary,
        "arms": {arm: {"winner": o["winner"], "n": o["n"],
                       "score": {i: o["variants"][i]["score"] for i in ids},
                       "share": {i: o["variants"][i]["share"] for i in ids},
                       "win_prob": {i: o["variants"][i]["win_prob"] for i in ids}}
                 for arm, o in summaries.items()},
        "position_effect": {"description": "score of the variant shown first minus shown last (first and last "
                                           "variant)", **bootstrap_ci(diff, B=B)},
        "quotes": {i: pick_quotes([{"pid": p, "segment": seg[p], "text": c.get("reason"), "value": c["score"].get(i)}
                                   for p, c in main.items() if c.get("choice") == i], 3) for i in ids},
        "answers": main,
        "controls": {"arms": ctrl, "plan": plan, "answers": {k: v for k, v in arms.items() if k != "persona"}},
        "segments": seg, "warnings": warnings, "usage": usage.as_dict(),
    }
    d = _direction(overall, truth, ids)
    if d:
        out["direction_correct"] = d["direction_correct"]
        out["truth_comparison"] = d
        out["arms_direction_correct"] = {arm: (_direction(o, truth, ids) or {}).get("direction_correct")
                                         for arm, o in summaries.items()}
    em.done(n=len(main), winner=overall["winner"])
    return to_jsonable(out)


def ab_frame(result: dict, truth=None) -> pd.DataFrame:
    """EvalFrame, one row per variant (item = test). truth: {variant: CTR} or winning id (-> 1/0).
    Columns on a 0-1 scale: pred_score ((mean 1-10 score - 1) / 9), pred_share (preference share), pred_win
    (bootstrap win probability), pred_<control arm> (control arm mean score, same scaling)."""
    def unit01(x):
        x = to_number(x)
        return np.nan if x is None else (x - 1) / 9
    if truth is None:
        tc = result.get("truth_comparison") or {}
        truth = tc.get("truth") or tc.get("true_winner")
    if isinstance(truth, dict) and "winner" in truth:
        truth = truth["winner"]
    test = result.get("test", "ab")
    arms = result.get("arms", {})
    rows = []
    for v in result.get("variants", []):
        i = v["id"]
        if isinstance(truth, str):
            t = 1.0 if i == truth else 0.0
        elif isinstance(truth, dict):
            t = to_number(truth.get(i))
            t = np.nan if t is None else t
        else:
            t = np.nan
        main = arms.get("persona", {})
        r = {"unit": f"{test}|{i}", "truth": t,
             "pred_score": unit01((main.get("score", {}).get(i) or {}).get("value")),
             "pred_share": main.get("share", {}).get(i), "pred_win": main.get("win_prob", {}).get(i)}
        for arm, o in arms.items():
            if arm != "persona":
                r[f"pred_{arm}"] = unit01((o.get("score", {}).get(i) or {}).get("value"))
        r.update({"segment": ALL, "item": test, "group": test})
        rows.append(r)
    return pd.DataFrame(rows)


def paars_decision(scores: dict, threshold: float = 0.65) -> dict:
    """scores {pid: {"A": s, "B": s}} -> forced-choice shares, winner or inconclusive, confidence.
    Confidence = share of personas scoring A above B, distance from 0.5 folded (sign-test logic); ties dropped.
    Port of deney/kod/kit/persona_talp.py `karar`."""
    a = np.array([to_number(v.get("A")) if to_number(v.get("A")) is not None else np.nan
                  for v in scores.values()], float)
    b = np.array([to_number(v.get("B")) if to_number(v.get("B")) is not None else np.nan
                  for v in scores.values()], float)
    m = np.isfinite(a) & np.isfinite(b) & (a != b)
    if m.sum() == 0:
        return {"forced_choice": {"A": 0.5, "B": 0.5}, "decision": "inconclusive", "confidence": 0.0}
    pa = float(np.mean(a[m] > b[m]))
    conf = max(pa, 1 - pa)
    return {"forced_choice": {"A": pa, "B": 1 - pa}, "mean_diff": float(np.nanmean(a - b)),
            "decision": ("winner_A" if pa > 0.5 else "winner_B") if conf >= threshold else "inconclusive",
            "confidence": conf}


# ---------------------------------------------------------------- many pairwise tests (Upworthy prova)
SYSTEM_MANY = ("You simulate a panel of distinct readers scrolling a social feed. Each reader judges headlines "
               "from their own tastes. Output only a JSON array.")
MANY_TEMPLATE = """Two alternative headlines (A and B) were tested for the same story. For EACH reader and EACH test below,
rate how likely the reader is to click each headline (1-10) and say which one they would click.

Readers:
{readers}

Tests:
{tests}

Return a JSON array with one object per (reader, test):
{{"id": "<reader_id>|<test_id>", "a": int 1-10, "b": int 1-10, "choice": "A" or "B"}}
JSON only."""


def ab_many(personas, tests, *, llm, tests_per_call: int = 5, system: str = SYSTEM_MANY,
            template: str = MANY_TEMPLATE, max_tokens: int | None = None, ask_kw: dict | None = None,
            usage: Usage | None = None) -> pd.DataFrame:
    """Many pairwise A/B tests, all personas per call (port of sahne/prova/upworthy.py `oyla`).
    tests: DataFrame or list of dicts with columns test_id, a, b (texts). Returns one row per test:
    test_id, p_a = (votes for A + 0.5) / (votes + 1) (Laplace), n_votes, votes (list of dicts).
    The prompt defaults to the headline wording of the measured run; pass `system`/`template` for other content
    (template fields: {readers}, {tests}).
    Calls go through `sim.panel.ask_many` (soft refusals not cached and re-asked once, empty replies re-asked with
    a larger max_tokens, a failed call gives no votes); `ask_kw` passes its options, `usage` collects the counts."""
    usage = usage if usage is not None else Usage()
    ps = as_personas(personas)
    df = pd.DataFrame(tests).reset_index(drop=True)
    for c in ("test_id", "a", "b"):
        if c not in df:
            raise ValueError(f"tests needs column {c!r}")
    df["test_id"] = df["test_id"].astype(str)
    readers = "\n".join(f"- {p.pid}: " + p.text("en").replace("\n", " · ")[:400] for p in ps)
    prompts = []
    for s in range(0, len(df), max(1, tests_per_call)):
        g = df.iloc[s:s + tests_per_call]
        block = "\n".join(f"- {r.test_id}: A = \"{str(r.a).strip()}\" | B = \"{str(r.b).strip()}\"" for r in g.itertuples())
        prompts.append((system, template.format(readers=readers, tests=block)))
    if max_tokens is None:
        max_tokens = min(16000, 200 + 45 * len(ps) * max(1, tests_per_call))
    votes: dict[str, list] = {}
    for r in ask_many(llm, prompts, max_tokens=max_tokens, usage=usage, tag="ab_many", reframe=REFRAME_NOTE_GENERIC,
                      **(ask_kw or {})):
        data = parse_json(getattr(r, "text", None))
        for x in (data if isinstance(data, list) else []):
            if not isinstance(x, dict) or "|" not in str(x.get("id")):
                continue
            pid, tid = str(x["id"]).split("|", 1)
            c = str(x.get("choice", "")).strip().upper()[:1]
            if c in ("A", "B"):
                votes.setdefault(tid.strip(), []).append({"pid": pid.strip(), "choice": c, "a": x.get("a"),
                                                          "b": x.get("b")})
    rows = []
    for tid in df["test_id"]:
        vs = votes.get(tid, [])
        rows.append({"test_id": tid, "p_a": (sum(v["choice"] == "A" for v in vs) + 0.5) / (len(vs) + 1),
                     "n_votes": len(vs), "votes": vs})
    return pd.DataFrame(rows)


__all__ = ["ab_test", "ab_frame", "ab_many", "confidence_check", "paars_decision", "DEFAULT_QUESTION"]
