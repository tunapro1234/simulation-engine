"""Survey panel: Likert questions (SSR + direct score) and multiple-choice questions.
Port of sahne/kod/sahne/kosucu/panel.py.

    questions = [{"id": "q1", "text": "How likely are you to buy ...?", "kind": "likert5",
                  "scale": "intent|agreement|satisfaction|likelihood|liking", "anchors": [5 sentences] (optional)},
                 {"id": "q2", "text": "...", "kind": "choice", "options": ["...", "..."]}]
    res = survey(personas, questions, llm=llm, context="product description", ssr=SSR())
    frame = survey_frame(res, truth={"q1": {pid: 4, ...}})          # EvalFrame, one row per persona x question

Likert: the persona first answers in its own words (`<f>_answer`; for Turkish also a one-sentence English
paraphrase `<f>_answer_en`), then a 1-5 score. SSR turns the free text into a distribution over the anchors;
the score is the direct measure. SSR is off unless an `SSR` instance (or True) is passed, so a run never
downloads a model by surprise. Control arms (no_persona, persona_swap) run by default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._stats import (ALL, Emitter, bootstrap_ci, bootstrap_distribution, field_name, pick_quotes, to_jsonable,
                     to_number)
from .panel import Usage, ask_panel
from .personas import as_personas, choose_lang, control_personas, control_plan, normalize_controls
from .ssr import SSR, anchor_sets

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _questions(questions, options=None) -> list[dict]:
    if isinstance(questions, (str, dict)):
        questions = [questions]
    if options is not None:
        if len(questions or []) != 1:
            raise ValueError("options= is for a single question; put options inside each question dict")
        q = questions[0]
        q = {"text": q} if isinstance(q, str) else dict(q)
        q.setdefault("kind", "choice")
        q["options"] = list(options)
        questions = [q]
    out = []
    for i, q in enumerate(questions or []):
        if isinstance(q, str):
            q = {"text": q}
        kind = q.get("kind", q.get("tur", "likert5"))
        kind = "choice" if kind in ("choice", "coktan", "mc", "multiple") else "likert5"
        qid = str(q.get("id", f"q{i + 1}"))
        d = {"id": qid, "text": q.get("text", q.get("metin", "")), "kind": kind, "field": field_name(qid, i)}
        if kind == "choice":
            opts = [str(x) for x in (q.get("options") or q.get("secenekler") or [])][:len(LETTERS)]
            if len(opts) < 2:
                raise ValueError(f"question {qid}: a choice question needs at least 2 options")
            d["options"] = opts
        else:
            d["scale"] = q.get("scale", q.get("olcek", "agreement"))
            d["anchors"] = q.get("anchors", q.get("capalar"))
        out.append(d)
    if not out:
        raise ValueError("questions is empty")
    seen = set()
    for i, d in enumerate(out):
        if d["field"] in seen:
            d["field"] = f"{d['field']}_{i}"
        seen.add(d["field"])
    return out


def _anchors(q: dict):
    return q.get("anchors") or q.get("scale") or "agreement"


def _task_fields(group: list[dict], context: str, lang: str):
    lines, fields = [], {}
    for q in group:
        f = q["field"]
        if q["kind"] == "likert5":
            lines.append(f"[{f}] {q['text']}")
            fields[f"{f}_answer"] = "str: 1-2 first-person sentences in the person's own voice answering question " + f
            if lang == "tr":
                fields[f"{f}_answer_en"] = ("str: the same answer as ONE short first-person English sentence "
                                            "starting with I (always English)")
            a0 = anchor_sets(_anchors(q))[0]
            fields[f"{f}_score"] = f"int 1-5 (1 = {a0[0]} 5 = {a0[-1]})"
        else:
            opts = "\n".join(f"   {LETTERS[j]}) {x}" for j, x in enumerate(q["options"]))
            lines.append(f"[{f}] {q['text']}\n{opts}")
            fields[f"{f}_choice"] = "one of " + "|".join(LETTERS[:len(q["options"])])
    task = ("Survey. Each person answers every question below honestly from their own situation; "
            "negative, unsure and indifferent answers are normal. Write the free-text answer before the number.\n")
    if context:
        task += f"\nContext shown to everyone:\n{context}\n"
    task += "\nQuestions:\n" + "\n".join(lines)
    return task, fields


def _arm(llm, personas, questions, lang, batch, group_size, context, tag, usage) -> dict:
    """Run one arm (main or control) -> {pid: {question_id: raw answer dict}}."""
    ans: dict[str, dict] = {p.pid: {} for p in personas}
    for g in range(0, len(questions), group_size):
        group = questions[g:g + group_size]
        task, fields = _task_fields(group, context, lang)
        r = ask_panel(llm, personas, task, fields, batch=batch, lang=lang, usage=usage, tag=f"{tag}:{g}")
        for pid, x in r.items():
            for q in group:
                f = q["field"]
                if q["kind"] == "likert5":
                    t = x.get(f"{f}_answer")
                    if t is None and x.get(f"{f}_score") is None:
                        continue
                    ans[pid][q["id"]] = {"text": t, "en": x.get(f"{f}_answer_en") if lang == "tr" else None,
                                         "score": to_number(x.get(f"{f}_score"), 1, 5)}
                else:
                    h = str(x.get(f"{f}_choice") or "").strip().lower()[:1]
                    j = LETTERS.find(h) if h else -1
                    if 0 <= j < len(q["options"]):
                        ans[pid][q["id"]] = {"choice": q["options"][j], "choice_index": j}
    return {pid: c for pid, c in ans.items() if c}


def _apply_ssr(arms: dict, questions, ssr, warnings: list):
    """Add SSR distributions to every Likert answer in every arm (one embedding batch per question)."""
    if ssr is None or ssr is False:
        return
    if ssr is True:
        ssr = SSR()
    for q in questions:
        if q["kind"] != "likert5":
            continue
        target = [(arm, pid) for arm, cv in arms.items() for pid, c in cv.items()
                  if q["id"] in c and c[q["id"]].get("text")]
        if not target:
            continue
        texts = [arms[a][p][q["id"]].get("en") or arms[a][p][q["id"]]["text"] for a, p in target]
        try:
            P = ssr.distribution(texts, _anchors(q))
        except Exception as e:  # noqa: BLE001 - missing optional model must not kill the run
            warnings.append(f"SSR failed ({q['id']}): {type(e).__name__}: {str(e)[:120]}; direct score only")
            return
        E = SSR.expected(P)
        for (a, p), pv, e in zip(target, P, E):
            arms[a][p][q["id"]]["ssr"] = [round(float(v), 4) for v in pv]
            arms[a][p][q["id"]]["ssr_mean"] = round(float(e), 4)


def _likert_summary(cv: dict, qid: str, pids, B: int) -> dict:
    ssr_P = [cv[p][qid]["ssr"] for p in pids if qid in cv.get(p, {}) and "ssr" in cv[p][qid]]
    ssr_e = [cv[p][qid].get("ssr_mean") for p in pids if qid in cv.get(p, {})]
    score = [cv[p][qid].get("score") for p in pids if qid in cv.get(p, {})]
    onehot = [np.eye(5)[int(round(v)) - 1] for v in score if v is not None]
    return {"ssr_mean": bootstrap_ci(ssr_e, B=B), "ssr_dist": bootstrap_distribution(ssr_P, B=B)["value"],
            "score_mean": bootstrap_ci(score, B=B), "score_dist": bootstrap_distribution(onehot, B=B)["value"],
            "n": sum(1 for p in pids if qid in cv.get(p, {}))}


def _choice_summary(cv: dict, q: dict, pids, B: int) -> dict:
    k = len(q["options"])
    onehot = [np.eye(k)[cv[p][q["id"]]["choice_index"]] for p in pids if q["id"] in cv.get(p, {})]
    d = bootstrap_distribution(onehot, B=B)
    return {"dist": dict(zip(q["options"], d["value"])) if d["n"] else {},
            "ci": {o: [a, b] for o, a, b in zip(q["options"], d["low"], d["high"])} if d["n"] else {},
            "n": d["n"]}


def survey(personas, questions, options=None, *, llm, context: str = "", lang: str | None = None, batch: int = 8,
           question_group: int = 6, controls=None, ssr=None, B: int = 1000, on_event=None) -> dict:
    """Simulated survey. `questions`: a list of question dicts, one dict, or one question string (with
    `options` it becomes a choice question, without them a 5-point Likert one). Returns a JSON-safe dict:
    per question the arm summaries (persona + controls), segment breakdown and quotes; raw answers; control
    plan; usage; warnings."""
    em = Emitter(on_event, "survey")
    ps = as_personas(personas)
    qs = _questions(questions, options)
    lang = choose_lang(lang, ps)
    seg = {p.pid: p.seg for p in ps}
    usage = Usage()
    warnings: list[str] = []

    em.stage("persona", n=len(ps))
    arms = {"persona": _arm(llm, ps, qs, lang, batch, question_group, context, "survey", usage)}
    q0 = qs[0]
    for pid, c in arms["persona"].items():
        a = c.get(q0["id"]) or {}
        em.row(pid, seg[pid], a.get("text") or a.get("choice") or "", a.get("score"), arm="persona")
    ctrl = normalize_controls(controls)
    plan = control_plan(len(ps), batch, ctrl)
    for arm in ctrl:
        cp = control_personas(arm, ps, lang, plan[arm]["n"])
        em.stage(arm, n=len(cp), calls=plan[arm]["calls"])
        arms[arm] = _arm(llm, cp, qs, lang, plan[arm]["batch"], question_group, context, f"survey:{arm}",
                         usage) if cp else {}
    _apply_ssr(arms, qs, ssr, warnings)

    main = arms["persona"]
    missing = [p.pid for p in ps if p.pid not in main]
    if missing:
        warnings.append(f"{len(missing)} personas unanswered")
    pids = [p.pid for p in ps]
    segs = sorted({seg[p] for p in pids})
    results = []
    for q in qs:
        summ = (lambda cv, pp, q=q: _likert_summary(cv, q["id"], pp, B)) if q["kind"] == "likert5" else \
               (lambda cv, pp, q=q: _choice_summary(cv, q, pp, B))
        r = {k: v for k, v in q.items() if k != "field" and v is not None}
        r["arms"] = {arm: summ(cv, pids) for arm, cv in arms.items()}
        r["overall"] = r["arms"]["persona"]
        r["segments"] = {sg: summ(main, [p for p in pids if seg[p] == sg]) for sg in segs}
        recs = []
        for p in pids:
            c = main.get(p, {}).get(q["id"])
            if not c:
                continue
            if q["kind"] == "likert5":
                recs.append({"pid": p, "segment": seg[p], "text": c.get("text"),
                             "value": c.get("ssr_mean") if c.get("ssr_mean") is not None else c.get("score")})
            else:
                recs.append({"pid": p, "segment": seg[p], "text": c.get("choice"), "value": c.get("choice_index")})
        r["quotes"] = pick_quotes(recs, 4)
        results.append(r)
    out = {"runner": "survey", "lang": lang, "n_personas": len(ps), "n_answered": len(main),
           "questions": results, "answers": main,
           "controls": {"arms": ctrl, "plan": plan, "answers": {k: v for k, v in arms.items() if k != "persona"}},
           "segments": seg, "warnings": warnings, "usage": usage.as_dict()}
    em.done(n=len(main))
    return to_jsonable(out)


def survey_frame(result: dict, truth: dict | None = None, level: str = "person", kind: str | None = None
                 ) -> pd.DataFrame:
    """EvalFrame from a survey result.

    level='person': one row per persona x question; truth {question_id: {pid: value}}.
    level='item': one row per question x segment ('all' included); truth {question_id: value} or
    {question_id: {segment: value}}.
    Likert and choice questions are not mixed: `kind` (default: likert5 if present) filters.
    Likert columns: pred_ssr (SSR expected score, NaN when SSR was off), pred_score (direct 1-5), and per control
    arm pred_<arm> (SSR) and pred_<arm>_score. Choice: person level the chosen option, item level the option
    distribution (list in `options` order)."""
    truth = truth or {}
    qs = result.get("questions", [])
    if kind is None:
        kind = "likert5" if any(q["kind"] == "likert5" for q in qs) else "choice"
    qs = [q for q in qs if q["kind"] == kind]
    arms = {"persona": result.get("answers", {}), **(result.get("controls", {}).get("answers") or {})}
    seg = result.get("segments", {})
    rows = []

    def t_get(qid, key):
        g = truth.get(qid)
        if isinstance(g, dict):
            return g.get(key, np.nan)
        return g if (g is not None and key == ALL) else np.nan

    if level == "person":
        for q in qs:
            for pid in seg:
                r = {"unit": f"{pid}|{q['id']}", "truth": t_get(q["id"], pid)}
                for arm, cv in arms.items():
                    c = (cv.get(pid) or {}).get(q["id"])
                    nm = "" if arm == "persona" else arm
                    if kind == "likert5":
                        r[f"pred_{nm or 'ssr'}"] = c.get("ssr_mean") if c and c.get("ssr_mean") is not None \
                            else np.nan
                        r[f"pred_{nm + '_score' if nm else 'score'}"] = c.get("score") if c and \
                            c.get("score") is not None else np.nan
                    else:
                        r[f"pred_{nm or 'choice'}"] = c.get("choice") if c else np.nan
                r.update({"segment": seg[pid], "item": q["id"], "group": pid})
                rows.append(r)
    elif level == "item":
        for q in qs:
            groups = {ALL: list(seg)}
            for pid, sg in seg.items():
                groups.setdefault(sg, []).append(pid)
            for sg, pids in groups.items():
                r = {"unit": f"{q['id']}|{sg}", "truth": t_get(q["id"], sg)}
                for arm, cv in arms.items():
                    nm = "" if arm == "persona" else arm
                    if kind == "likert5":
                        o = _likert_summary(cv, q["id"], pids, B=1)
                        v1, v2 = o["ssr_mean"]["value"], o["score_mean"]["value"]
                        r[f"pred_{nm or 'ssr'}"] = np.nan if v1 is None else v1
                        r[f"pred_{nm + '_score' if nm else 'score'}"] = np.nan if v2 is None else v2
                    else:
                        o = _choice_summary(cv, q, pids, B=1)
                        r[f"pred_{nm or 'dist'}"] = [o["dist"].get(x, 0.0) for x in q["options"]] if o["n"] \
                            else np.nan
                r.update({"segment": sg, "item": q["id"], "group": q["id"]})
                rows.append(r)
    else:
        raise ValueError("level must be 'person' or 'item'")
    return pd.DataFrame(rows)


__all__ = ["survey", "survey_frame"]
