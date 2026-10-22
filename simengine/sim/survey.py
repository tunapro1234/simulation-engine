"""Survey panel: Likert questions (SSR + direct score), numeric/slider questions and multiple-choice questions.
Port of sahne/kod/sahne/kosucu/panel.py.

    questions = [{"id": "q1", "text": "How likely are you to buy ...?", "kind": "likert5",
                  "scale": "intent|agreement|satisfaction|likelihood|liking", "anchors": [5 sentences] (optional)},
                 {"id": "q2", "text": "...", "kind": "choice", "options": ["...", "..."]},
                 {"id": "q3", "text": "...", "kind": "numeric", "min": 0, "max": 100,
                  "labels": ["strongly oppose", "strongly support"] (optional), "free_text": False}]
    res = survey(personas, questions, llm=llm, context="product description", ssr=SSR())
    res = survey(personas, questions, llm=llm, stimulus_col="message")     # each persona sees its own message
    frame = survey_frame(res, truth={"q1": {pid: 4, ...}})          # EvalFrame, one row per persona x question

Likert: the persona first answers in its own words (`<f>_answer`; for Turkish also a one-sentence English
paraphrase `<f>_answer_en`), then a 1-5 score. SSR turns the free text into a distribution over the anchors;
the score is the direct measure. SSR is off unless an `SSR` instance (or True) is passed, so a run never
downloads a model by surprise. Control arms (no_persona, persona_swap) run by default.
Numeric: a number in [min, max] (and an optional one-sentence reason); a value outside the range is unparsed.
`free_text=False` (per question or for the whole survey) drops the free-text field, which is most of the output.
Soft refusals are re-asked once with a research-forecast framing and reported apart from parse failures
(`refused`, `unparsed`, `refusal_rate`, `refusals` in the result).
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
NUMERIC_KINDS = ("numeric", "number", "slider", "scale", "sayisal", "kaydirici")
STIMULUS_BATCHING = ("group", "inline")
_EPS = 1e-9


def _questions(questions, options=None, free_text: bool = True) -> list[dict]:
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
        kind = ("choice" if kind in ("choice", "coktan", "mc", "multiple") else
                "numeric" if kind in NUMERIC_KINDS else "likert5")
        qid = str(q.get("id", f"q{i + 1}"))
        d = {"id": qid, "text": q.get("text", q.get("metin", "")), "kind": kind, "field": field_name(qid, i)}
        if kind == "choice":
            opts = [str(x) for x in (q.get("options") or q.get("secenekler") or [])][:len(LETTERS)]
            if len(opts) < 2:
                raise ValueError(f"question {qid}: a choice question needs at least 2 options")
            d["options"] = opts
        elif kind == "numeric":
            lo, hi = q.get("min", 0), q.get("max", 100)
            try:
                lo, hi = float(lo), float(hi)
            except (TypeError, ValueError):
                raise ValueError(f"question {qid}: min and max must be numbers") from None
            if not lo < hi:
                raise ValueError(f"question {qid}: a numeric question needs min < max (got {lo}, {hi})")
            d["min"], d["max"] = (int(lo) if lo.is_integer() else lo), (int(hi) if hi.is_integer() else hi)
            labels = q.get("labels")
            if labels is not None:
                labels = [str(x) for x in labels]
                if len(labels) != 2:
                    raise ValueError(f"question {qid}: labels must be [label at min, label at max]")
            d["labels"] = labels
        else:
            d["scale"] = q.get("scale", q.get("olcek", "agreement"))
            d["anchors"] = q.get("anchors", q.get("capalar"))
        if kind != "choice":
            d["free_text"] = bool(q.get("free_text", free_text))
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


def _numeric_spec(q: dict) -> str:
    lo, hi = q["min"], q["max"]
    lab = f" ({lo} = {q['labels'][0]}, {hi} = {q['labels'][1]})" if q.get("labels") else ""
    return f"number {lo}-{hi}{lab}"


def _task_fields(group: list[dict], context: str, lang: str, inline_stimulus: bool = False):
    lines, fields = [], {}
    any_text = False
    for q in group:
        f = q["field"]
        ft = q.get("free_text", True)
        if q["kind"] == "likert5":
            lines.append(f"[{f}] {q['text']}")
            if ft:
                any_text = True
                fields[f"{f}_answer"] = ("str: 1-2 first-person sentences in the person's own voice answering "
                                         "question " + f)
                if lang == "tr":
                    fields[f"{f}_answer_en"] = ("str: the same answer as ONE short first-person English sentence "
                                                "starting with I (always English)")
            a0 = anchor_sets(_anchors(q))[0]
            fields[f"{f}_score"] = f"int 1-5 (1 = {a0[0]} 5 = {a0[-1]})"
        elif q["kind"] == "numeric":
            lines.append(f"[{f}] {q['text']} (answer with a number from {q['min']} to {q['max']})")
            if ft:
                any_text = True
                fields[f"{f}_reason"] = "str: 1 short first-person sentence in the person's own voice"
            fields[f"{f}_value"] = _numeric_spec(q)
        else:
            opts = "\n".join(f"   {LETTERS[j]}) {x}" for j, x in enumerate(q["options"]))
            lines.append(f"[{f}] {q['text']}\n{opts}")
            fields[f"{f}_choice"] = "one of " + "|".join(LETTERS[:len(q["options"])])
    task = ("Survey. Each person answers every question below honestly from their own situation; "
            "negative, unsure and indifferent answers are normal."
            + (" Write the free-text answer before the number." if any_text else "") + "\n")
    if context:
        task += f"\nContext shown to everyone:\n{context}\n"
    if inline_stimulus:
        task += ("\nEach person saw their own material, shown under their profile; answer as that person after "
                 "seeing it.\n")
    task += "\nQuestions:\n" + "\n".join(lines)
    return task, fields


def _parse_answer(q: dict, x: dict, lang: str):
    """One persona's raw fields for one question -> answer dict, None (no answer at all) or False (invalid)."""
    f = q["field"]
    if q["kind"] == "likert5":
        t = x.get(f"{f}_answer") if q.get("free_text", True) else None
        sc = to_number(x.get(f"{f}_score"), 1, 5)
        if t is None and sc is None:
            return False if x.get(f"{f}_score") is not None else None
        return {"text": t, "en": x.get(f"{f}_answer_en") if lang == "tr" else None, "score": sc}
    if q["kind"] == "numeric":
        raw = x.get(f"{f}_value")
        if raw is None:
            return None
        v = to_number(raw)
        if v is None or v < q["min"] - _EPS or v > q["max"] + _EPS:
            return False
        v = min(max(v, q["min"]), q["max"])
        return {"value": v, "text": x.get(f"{f}_reason") if q.get("free_text", True) else None}
    h = str(x.get(f"{f}_choice") or "").strip().lower()[:1]
    if not h:
        return None
    j = LETTERS.find(h)
    if 0 <= j < len(q["options"]):
        return {"choice": q["options"][j], "choice_index": j}
    return False


def _stim_groups(personas, context: str, stim: dict | None, mode: str):
    """[(shared context, personas, inline {pid: text} or None)]: one entry per distinct stimulus in 'group' mode."""
    if not stim:
        return [(context, personas, None)]
    if mode == "inline":
        return [(context, personas, {p.pid: stim[p.pid] for p in personas if stim.get(p.pid)})]
    order: dict[str, list] = {}
    for p in personas:
        order.setdefault(stim.get(p.pid) or "", []).append(p)
    return [("\n\n".join(x for x in (context, s) if x), ps, None) for s, ps in order.items()]


def _arm(llm, personas, questions, lang, batch, group_size, context, tag, usage, *, arm: str = "persona",
         stim: dict | None = None, stimulus_batching: str = "group", info: dict | None = None,
         ask_kw: dict | None = None) -> dict:
    """Run one arm (main or control) -> {pid: {question_id: raw answer dict}}. `info` collects this arm's
    refusal / unparsed counts."""
    ans: dict[str, dict] = {p.pid: {} for p in personas}
    refused: set = set()
    bad = {q["id"]: 0 for q in questions}
    before = (usage.first_calls, usage.refused_calls, usage.recovered_calls)
    errors: list = []
    for g in range(0, len(questions), group_size):
        group = questions[g:g + group_size]
        for ctx, ps, inline in _stim_groups(personas, context, stim, stimulus_batching):
            task, fields = _task_fields(group, ctx, lang, inline_stimulus=bool(inline))
            r = ask_panel(llm, ps, task, fields, batch=batch, lang=lang, usage=usage, tag=f"{tag}:{g}",
                          llm_tag=f"survey:{arm}", context=inline, context_label="Shown to this person",
                          refused_ids=refused, errors=errors, **(ask_kw or {}))
            for pid, x in r.items():
                for q in group:
                    a = _parse_answer(q, x, lang)
                    if a is False:
                        bad[q["id"]] += 1
                    elif a is not None:
                        ans[pid][q["id"]] = a
    if info is not None:
        first = usage.first_calls - before[0]
        rc = usage.refused_calls - before[1]
        info.update({"calls": first, "refused_calls": rc, "recovered_calls": usage.recovered_calls - before[2],
                     "refusal_rate": round(rc / first, 4) if first else 0.0, "refused_ids": refused,
                     "unparsed_items": bad, "errors": errors})
    return {pid: c for pid, c in ans.items() if c}


def _apply_ssr(arms: dict, questions, ssr, warnings: list):
    """Add SSR distributions to every Likert answer in every arm (one embedding batch per question)."""
    if ssr is None or ssr is False:
        return
    if ssr is True:
        ssr = SSR()
    off = [q["id"] for q in questions if q["kind"] == "likert5" and not q.get("free_text", True)]
    if off:
        warnings.append(f"SSR skipped for {off}: free_text=False leaves no text to embed; direct score only")
    for q in questions:
        if q["kind"] != "likert5" or not q.get("free_text", True):
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


def _numeric_summary(cv: dict, qid: str, pids, B: int) -> dict:
    v = [cv[p][qid].get("value") for p in pids if qid in cv.get(p, {})]
    x = np.asarray([a for a in v if a is not None], dtype=float)
    return {"mean": bootstrap_ci(x, B=B), "median": float(np.median(x)) if len(x) else None,
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None, "n": int(len(x))}


def _choice_summary(cv: dict, q: dict, pids, B: int) -> dict:
    k = len(q["options"])
    onehot = [np.eye(k)[cv[p][q["id"]]["choice_index"]] for p in pids if q["id"] in cv.get(p, {})]
    d = bootstrap_distribution(onehot, B=B)
    return {"dist": dict(zip(q["options"], d["value"])) if d["n"] else {},
            "ci": {o: [a, b] for o, a, b in zip(q["options"], d["low"], d["high"])} if d["n"] else {},
            "n": d["n"]}


def _summarizer(q: dict, B: int):
    if q["kind"] == "likert5":
        return lambda cv, pp: _likert_summary(cv, q["id"], pp, B)
    if q["kind"] == "numeric":
        return lambda cv, pp: _numeric_summary(cv, q["id"], pp, B)
    return lambda cv, pp: _choice_summary(cv, q, pp, B)


def _blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and v != v:
        return True
    return isinstance(v, str) and not v.strip()


def persona_value(p, col: str):
    """Look up `col` on a persona-like source: flat dict key, TALP facet (evidence.measured_behavior.facets),
    identity field, demographic trait, trait label, authored field, `extra`, or an object attribute."""
    src = p.source if p.source is not None else p
    d = src if isinstance(src, dict) else (src.to_dict() if callable(getattr(src, "to_dict", None)) else None)
    if isinstance(d, dict):
        if col in d and not isinstance(d[col], (dict, list)):
            return d[col]
        facets = ((d.get("evidence") or {}).get("measured_behavior") or {}).get("facets") or {}
        idn = d.get("identity") or {}
        for box in (facets, idn, idn.get("demographic_traits") or {}, d.get("authored") or {},
                    d.get("extra") or {}):
            if isinstance(box, dict) and col in box and not isinstance(box[col], (dict, list)):
                return box[col]
        for t in d.get("traits") or []:
            if isinstance(t, dict) and t.get("label") == col:
                return t.get("value")
    v = getattr(src, col, None)
    return None if callable(v) else v


def _without_field(p, col: str):
    """PersonaView `p` re-rendered without field `col` (flat key, facet, identity / demographic field, authored
    field, extra, trait label), so a stimulus read from `col` is not also shown in the profile text (and is not
    carried into the persona_swap body). Keeps pid, segment, locale and source. Unchanged when the source is not
    dict-like."""
    from .personas import as_persona
    src = p.source if p.source is not None else None
    d = src if isinstance(src, dict) else (src.to_dict() if callable(getattr(src, "to_dict", None)) else None)
    if not isinstance(d, dict):
        return p
    d2 = {k: v for k, v in d.items() if k != col}
    for key in ("identity", "authored", "extra"):
        if isinstance(d2.get(key), dict):
            d2[key] = {k: v for k, v in d2[key].items() if k != col}
    idn = d2.get("identity")
    if isinstance(idn, dict) and isinstance(idn.get("demographic_traits"), dict):
        idn["demographic_traits"] = {k: v for k, v in idn["demographic_traits"].items() if k != col}
    ev = d2.get("evidence")
    if isinstance(ev, dict) and isinstance(ev.get("measured_behavior"), dict):
        mb = dict(ev["measured_behavior"])
        if isinstance(mb.get("facets"), dict):
            mb["facets"] = {k: v for k, v in mb["facets"].items() if k != col}
        d2["evidence"] = {**ev, "measured_behavior": mb}
    if isinstance(d2.get("traits"), list):
        d2["traits"] = [t for t in d2["traits"] if not (isinstance(t, dict) and t.get("label") == col)]
    v = as_persona(d2)
    v.pid, v.segment, v.locale, v.source = p.pid, p.segment, p.locale, p.source
    if p.tagline is not None:
        v.tagline = p.tagline
    return v


def _stimuli(ps, context, stimulus_col, warnings: list):
    """-> (shared context text, {pid: stimulus} or None)."""
    if isinstance(context, dict):
        if stimulus_col is not None:
            raise ValueError("pass per-person stimuli either as context={pid: text} or as stimulus_col=, not both")
        stim = {str(k): str(v) for k, v in context.items() if not _blank(v)}
        unknown = set(stim) - {p.pid for p in ps}
        if unknown:
            warnings.append(f"context has {len(unknown)} ids that are not in the panel (ignored)")
        shared = ""
    elif stimulus_col is not None:
        stim = {}
        for p in ps:
            v = persona_value(p, stimulus_col)
            if not _blank(v):
                stim[p.pid] = str(v)
        shared = context or ""
    else:
        return context or "", None
    lacking = sum(1 for p in ps if p.pid not in stim)
    if lacking:
        warnings.append(f"{lacking} personas have no stimulus (shared context only)")
    return shared, stim


def survey(personas, questions, options=None, *, llm, context: str | dict = "", lang: str | None = None,
           batch: int = 8, question_group: int = 6, controls=None, ssr=None, B: int = 1000, on_event=None,
           stimulus_col: str | None = None, stimulus_batching: str = "group", free_text: bool = True,
           refusal_retry: bool = True, cache_refusals: bool = False, empty_retry: bool = True,
           empty_retry_kw: dict | None = None) -> dict:
    """Simulated survey. `questions`: a list of question dicts, one dict, or one question string (with
    `options` it becomes a choice question, without them a 5-point Likert one). Returns a JSON-safe dict:
    per question the arm summaries (persona + controls), segment breakdown and quotes; raw answers; control
    plan; usage; warnings; refusal accounting.

    context: one text shown to everyone, or {persona_id: text} (a per-person stimulus).
    stimulus_col: name of a persona field / table column holding each persona's own stimulus (e.g. the message
        or condition it saw); `context` (a string) is then still shown to everyone.
    stimulus_batching: "group" (default) batches only personas with the same stimulus, which goes into the
        shared context block; "inline" mixes stimuli in one batch and shows each under its person.
    free_text: default for Likert/numeric questions; False drops the free-text field (fewer output tokens,
        no SSR). A question dict's own "free_text" wins.
    refusal_retry / cache_refusals: see `ask_panel`. The result reports `refused` (personas lost to a refusal
        that persisted after the reframed re-ask), `unparsed` (personas unanswered for any other reason),
        `refusal_rate` (main arm, first pass) and per-arm detail in `refusals`.
    empty_retry / empty_retry_kw: see `ask_many`. A batch whose reply stays empty (EmptyResponseError) or fails
        with another LLMError counts as unparsed (usage `empty_calls` / `error_calls`, plus a warning); the survey
        raises only when no persona in the main arm got an answer."""
    if stimulus_batching not in STIMULUS_BATCHING:
        raise ValueError(f"stimulus_batching must be one of {STIMULUS_BATCHING}")
    em = Emitter(on_event, "survey")
    ps = as_personas(personas)
    qs = _questions(questions, options, free_text=free_text)
    lang = choose_lang(lang, ps)
    seg = {p.pid: p.seg for p in ps}
    usage = Usage()
    warnings: list[str] = []
    shared, stim = _stimuli(ps, context, stimulus_col, warnings)
    if stimulus_col is not None:
        # the stimulus goes in once, as context; drop it from the profile text (all arms, incl. the swap body)
        ps = [_without_field(p, stimulus_col) for p in ps]
    ask_kw = {"refusal_retry": refusal_retry, "cache_refusals": cache_refusals, "empty_retry": empty_retry,
              "empty_retry_kw": empty_retry_kw}
    opts = {"stim": stim, "stimulus_batching": stimulus_batching, "ask_kw": ask_kw}
    info: dict[str, dict] = {"persona": {}}

    em.stage("persona", n=len(ps))
    arms = {"persona": _arm(llm, ps, qs, lang, batch, question_group, shared, "survey", usage, arm="persona",
                            info=info["persona"], **opts)}
    q0 = qs[0]
    for pid, c in arms["persona"].items():
        a = c.get(q0["id"]) or {}
        em.row(pid, seg[pid], a.get("text") or a.get("choice") or "", a.get("score", a.get("value")),
               arm="persona")
    ctrl = normalize_controls(controls)
    plan = control_plan(len(ps), batch, ctrl)
    for arm in ctrl:
        cp = control_personas(arm, ps, lang, plan[arm]["n"])
        em.stage(arm, n=len(cp), calls=plan[arm]["calls"])
        info[arm] = {}
        arms[arm] = _arm(llm, cp, qs, lang, plan[arm]["batch"], question_group, shared, f"survey:{arm}",
                         usage, arm=arm, info=info[arm], **opts) if cp else {}
    _apply_ssr(arms, qs, ssr, warnings)

    main = arms["persona"]
    mi = info["persona"]
    if not main and mi.get("errors"):
        raise mi["errors"][0]          # no persona got any answer: a real failure, not a gap to report
    if usage.empty_calls or usage.error_calls:
        warnings.append(f"{usage.empty_calls} empty replies ({usage.empty_recovered} recovered by a larger-max_tokens "
                        f"re-ask) and {usage.error_calls} failed calls; their personas count as unparsed. Last: "
                        f"{usage.last_error}. For empty replies pass thinking='off' or a larger max_tokens.")
    missing = [p.pid for p in ps if p.pid not in main]
    n_refused = sum(1 for p in missing if p in mi["refused_ids"])
    n_unparsed = len(missing) - n_refused
    if missing:
        warnings.append(f"{len(missing)} personas unanswered ({n_refused} refused after the reframed re-ask, "
                        f"{n_unparsed} unparsed or missing)")
    if mi["refused_calls"]:
        warnings.append(f"{mi['refused_calls']}/{mi['calls']} persona-arm batches refused at the first ask "
                        f"(refusal_rate {mi['refusal_rate']:.2f}); {mi['recovered_calls']} recovered by the "
                        f"research-forecast re-ask" if refusal_retry else
                        f"{mi['refused_calls']}/{mi['calls']} persona-arm batches refused (refusal_retry off)")
    bad_items = sum(mi["unparsed_items"].values())
    if bad_items:
        warnings.append(f"{bad_items} persona answers unparsed (invalid or out-of-range values): "
                        + ", ".join(f"{k}={v}" for k, v in mi["unparsed_items"].items() if v))
    for q in qs:      # a 0-1 fraction on a 0-100 slider is in range, so it parses silently: flag it
        if q["kind"] != "numeric" or q["max"] - q["min"] <= 1:
            continue
        vals = [c[q["id"]]["value"] for c in main.values() if q["id"] in c]
        if len(vals) >= 3 and max(abs(v) for v in vals) <= 1:
            warnings.append(f"{q['id']}: every answer is between -1 and 1 on a {q['min']}-{q['max']} scale; the model "
                            "may be answering with fractions (e.g. 0.75 for 75). Check the question wording.")
    pids = [p.pid for p in ps]
    segs = sorted({seg[p] for p in pids})
    results = []
    for q in qs:
        summ = _summarizer(q, B)
        r = {k: v for k, v in q.items() if k != "field" and v is not None}
        r["arms"] = {arm: summ(cv, pids) for arm, cv in arms.items()}
        r["overall"] = r["arms"]["persona"]
        r["segments"] = {sg: summ(main, [p for p in pids if seg[p] == sg]) for sg in segs}
        r["unparsed"] = mi["unparsed_items"].get(q["id"], 0)
        recs = []
        for p in pids:
            c = main.get(p, {}).get(q["id"])
            if not c:
                continue
            if q["kind"] == "likert5":
                recs.append({"pid": p, "segment": seg[p], "text": c.get("text"),
                             "value": c.get("ssr_mean") if c.get("ssr_mean") is not None else c.get("score")})
            elif q["kind"] == "numeric":
                recs.append({"pid": p, "segment": seg[p], "text": c.get("text"), "value": c.get("value")})
            else:
                recs.append({"pid": p, "segment": seg[p], "text": c.get("choice"), "value": c.get("choice_index")})
        r["quotes"] = pick_quotes(recs, 4)
        results.append(r)
    refusals = {arm: {k: (len(v) if k == "refused_ids" else v) for k, v in d.items()
                      if k not in ("unparsed_items", "errors")} for arm, d in info.items() if d}
    for d in refusals.values():
        d["refused_personas"] = d.pop("refused_ids")
    out = {"runner": "survey", "lang": lang, "n_personas": len(ps), "n_answered": len(main),
           "questions": results, "answers": main,
           "controls": {"arms": ctrl, "plan": plan, "answers": {k: v for k, v in arms.items() if k != "persona"}},
           "segments": seg, "warnings": warnings, "usage": usage.as_dict(),
           "refused": n_refused, "unparsed": n_unparsed, "refusal_rate": mi["refusal_rate"],
           "unparsed_items": bad_items, "refusals": refusals}
    if stim is not None:
        out["stimuli"] = {p.pid: stim.get(p.pid) for p in ps}
    em.done(n=len(main))
    return to_jsonable(out)


def survey_frame(result: dict, truth: dict | None = None, level: str = "person", kind: str | None = None
                 ) -> pd.DataFrame:
    """EvalFrame from a survey result.

    level='person': one row per persona x question; truth {question_id: {pid: value}}.
    level='item': one row per question x segment ('all' included); truth {question_id: value} or
    {question_id: {segment: value}}.
    Likert, numeric and choice questions are not mixed: `kind` (default: likert5 if present, else numeric, else
    choice) filters.
    Likert columns: pred_ssr (SSR expected score, NaN when SSR was off), pred_score (direct 1-5), and per control
    arm pred_<arm> (SSR) and pred_<arm>_score. Numeric: pred_value and pred_<arm> (person value; item level the
    mean). Choice: person level the chosen option, item level the option distribution (list in `options` order).
    With per-person stimuli the person-level frame has a `stimulus` column."""
    truth = truth or {}
    qs = result.get("questions", [])
    if kind is None:
        kinds = {q["kind"] for q in qs}
        kind = "likert5" if "likert5" in kinds else "numeric" if "numeric" in kinds else "choice"
    qs = [q for q in qs if q["kind"] == kind]
    arms = {"persona": result.get("answers", {}), **(result.get("controls", {}).get("answers") or {})}
    seg = result.get("segments", {})
    stimuli = result.get("stimuli")
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
                    elif kind == "numeric":
                        r[f"pred_{nm or 'value'}"] = c.get("value") if c and c.get("value") is not None else np.nan
                    else:
                        r[f"pred_{nm or 'choice'}"] = c.get("choice") if c else np.nan
                r.update({"segment": seg[pid], "item": q["id"], "group": pid})
                if stimuli is not None:
                    r["stimulus"] = stimuli.get(pid)
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
                    elif kind == "numeric":
                        v = _numeric_summary(cv, q["id"], pids, B=1)["mean"]["value"]
                        r[f"pred_{nm or 'value'}"] = np.nan if v is None else v
                    else:
                        o = _choice_summary(cv, q, pids, B=1)
                        r[f"pred_{nm or 'dist'}"] = [o["dist"].get(x, 0.0) for x in q["options"]] if o["n"] \
                            else np.nan
                r.update({"segment": sg, "item": q["id"], "group": q["id"]})
                rows.append(r)
    else:
        raise ValueError("level must be 'person' or 'item'")
    return pd.DataFrame(rows)


__all__ = ["survey", "survey_frame", "persona_value"]
