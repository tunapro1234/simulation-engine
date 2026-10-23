"""Open-ended interview panel: one free-text question per persona, with the no_persona / persona_swap control arms.

    res = interview(personas, "Why did you stop ordering last month?", llm=llm)
    res = interview(personas, question, llm=llm, stimulus_col="message")          # each persona sees its own message
    res = interview(personas, question, llm=llm, stimuli={pid: text}, arms=("persona", "swap"))
    res["answers"]                       # {pid: text}, persona arm
    res["controls"]["answers"]           # {arm: {pid: text}}
    res["refused_ids"], res["empty_ids"], res["error_ids"]                        # main arm; per arm in res["arm_stats"]
    frame = interview_frame(res)         # one row per person x arm (pred_text, status, chars)

`survey` covers Likert / numeric / choice questions; a plain "why?" question had no runner, so its control arms
had to be built by hand with `llm.map` (G6 rehearsal #3). This runner uses the same pieces as `survey`: the batched
panel prompt (`build_prompts` / `ask_many`), `control_personas` / `control_plan` for the control arms (persona_swap
may be a subsample, see result["controls"]["plan"]), the same stimulus handling (`stimulus_col=` or a per-person
dict; the stimulus is dropped from the profile text) and the same ledger tag layout as `survey`: `interview:<arm>`
(`interview:persona`, `interview:no_persona`, `interview:persona_swap`), or `<tag>:interview:<arm>` with `tag=`.

Per arm, every asked persona ends in exactly one bucket:
  answered  a non-blank answer (cut to max_chars; `truncated` counts the cuts)
  refused   its batch stayed refused after the research-forecast re-ask (soft refusal or RefusalError), or its
            answer reads like the model declining ("As an AI I can't answer for this person"; `looks_like_refusal`;
            the text is kept in arm_stats[arm]["refused_answers"]; answer_refusal=False keeps it as an answer)
  error     its batch got no reply (empty reply after the larger-max_tokens re-ask, or another LLMError)
  empty     its batch replied, but this persona's answer is missing, blank or not text
A persona declining in its own voice ("I'd rather not say", "Bunu söylemek istemiyorum") is an answer.
Not the same module as simengine.persona.inject.interview (interview-transcript injectors).
"""
from __future__ import annotations

import pandas as pd

from ..llm.fake import FakeLLM
from ._stats import Emitter, to_jsonable
from .panel import Usage, _collect, ask_many, build_prompts, looks_like_refusal
from .personas import as_personas, choose_lang, control_personas, control_plan, normalize_controls
from .survey import _resolve_parallel, _stim_groups, _stimuli, _without_field

ARMS_DEFAULT = ("persona", "no_persona", "persona_swap")
STIMULUS_BATCHING = ("group", "inline")
STATUSES = ("answered", "refused", "empty", "error", "not_asked")


def _arms(arms) -> list[str]:
    """'persona' first (when asked for), then the control arms in their given order (aliases resolved)."""
    if isinstance(arms, str):
        arms = [arms]
    arms = list(arms or [])
    if not arms:
        raise ValueError(f"arms is empty; use a subset of {ARMS_DEFAULT}")
    ctrl = normalize_controls([a for a in arms if a != "persona"])
    return (["persona"] if "persona" in arms else []) + ctrl


def _task(question: str, context: str, max_chars: int, inline_stimulus: bool) -> str:
    task = ("Interview. Each person answers the question below in their own words, from their own situation, "
            f"in at most {max_chars} characters; negative, unsure, indifferent and short answers are normal.\n")
    if context:
        task += f"\nContext shown to everyone:\n{context}\n"
    if inline_stimulus:
        task += ("\nEach person saw their own material, shown under their profile; answer as that person after "
                 "seeing it.\n")
    return task + f"\nQuestion:\n{question}"


def _text(v) -> str | None:
    """Answer field -> stripped text, or None (missing, blank, or not text)."""
    if isinstance(v, list):
        v = " ".join(str(x) for x in v if isinstance(x, (str, int, float)))
    if not isinstance(v, (str, int, float)) or isinstance(v, bool):
        return None
    s = str(v).strip()
    return s or None


def _max_tokens(plan, max_chars: int) -> int:
    """Room for the largest batch: ~2 characters per token covers Turkish free text."""
    return min(16000, 300 + max(len(ids) for _, ids in plan) * (40 + max_chars // 2))


def _arm(llm, personas, question, lang, batch, shared, stim, mode, max_chars, usage, *, tag, llm_tag,
         parallel, ask_kw, answer_refusal=True) -> tuple[dict, dict]:
    """Ask one arm -> ({pid: text}, stats). All batches (every stimulus group) go out in one `ask_many`."""
    plan = []
    for ctx, ps, inline in _stim_groups(personas, shared, stim, mode):
        plan += build_prompts(ps, _task(question, ctx, max_chars, bool(inline)),
                              {"answer": f"str: the person's own first-person answer, at most {max_chars} characters"},
                              batch=batch, context=inline, lang=lang, context_label="Shown to this person")
    before = (usage.first_calls, usage.refused_calls, usage.recovered_calls)
    st: list = []
    errors: list = []
    raw: dict = {}
    refused: set = set()
    if plan:
        rs = ask_many(llm, [p for p, _ in plan], max_tokens=_max_tokens(plan, max_chars), usage=usage, tag=tag,
                      llm_tag=llm_tag, parallel=parallel, statuses=st, errors=errors, **ask_kw)
        _collect(plan, rs, st, raw, usage, refused)
        failed = {pid for (_, ids), r, s in zip(plan, rs, st) if r is None and s != "refused" for pid in ids}
    else:
        failed = set()
    answers, ids = {}, {"refused": [], "empty": [], "error": []}
    truncated = 0
    declined: dict = {}                             # model-voice refusals inside the answer field
    for p in personas:
        t = _text((raw.get(p.pid) or {}).get("answer"))
        if t is not None and answer_refusal and looks_like_refusal(t):
            declined[p.pid] = t[:max_chars]
            ids["refused"].append(p.pid)
        elif t is not None:
            if len(t) > max_chars:
                t, truncated = t[:max_chars].rstrip(), truncated + 1
            answers[p.pid] = t
        elif p.pid in refused:
            ids["refused"].append(p.pid)
        elif p.pid in failed:
            ids["error"].append(p.pid)
        else:
            ids["empty"].append(p.pid)
    first = usage.first_calls - before[0]
    rc = usage.refused_calls - before[1]
    stats = {"n": len(personas), "answered": len(answers), **{k: len(v) for k, v in ids.items()},
             **{f"{k}_ids": v for k, v in ids.items()}, "refused_in_answer": len(declined),
             "refused_answers": declined, "truncated": truncated, "calls": first,
             "refused_calls": rc, "recovered_calls": usage.recovered_calls - before[2],
             "refusal_rate": round(rc / first, 4) if first else 0.0, "_errors": errors}
    return answers, stats


def interview(personas, question: str, *, llm, context: str | dict = "", stimulus_col: str | None = None,
              stimuli: dict | None = None, arms=ARMS_DEFAULT, max_chars: int = 600, batch: int = 8,
              lang: str | None = None, stimulus_batching: str = "group", refusal_retry: bool = True,
              cache_refusals: bool = False, empty_retry: bool = True, empty_retry_kw: dict | None = None,
              parallel: int | None = None, tag: str | None = None, answer_refusal: bool = True,
              on_event=None) -> dict:
    """Ask every persona one open-ended question; returns a JSON-safe dict with per-person free-text answers per
    arm and refused / empty / error counts and ids (main arm at the top level, every arm in `arm_stats`).

    question: the question text. max_chars: the answer limit stated in the prompt; longer answers are cut.
    arms: any of "persona", "no_persona", "persona_swap" (aliases as in `survey`: "swap", "personasiz",
        "persona_yok"). The control arms follow `control_plan` (no_persona in batches of up to 20, persona_swap
        within the remaining call budget), so the defaults roughly double the calls. The main arm is "persona"
        when run, else the first control arm.
    context: one text shown to everyone, or {pid: text} (a per-person stimulus).
    stimulus_col: a persona field / table column with each persona's own stimulus; stimuli: {pid: text}, the same
        as a dict context (`context` may then stay a shared text). Use one of the two. The stimulus is removed
        from the profile text in every arm (incl. the swap body).
    stimulus_batching, parallel, refusal_retry, cache_refusals, empty_retry, empty_retry_kw: as in `survey`.
    tag: a ledger-tag label, as in `survey`: calls are tagged f"{tag}:interview:{arm}" (default "interview:{arm}").
    answer_refusal: True (default) moves an answer that reads like the model declining (`looks_like_refusal`) to
        refused; its text stays in arm_stats[arm]["refused_answers"]. False keeps it as an answer.
    on_event: callback for {"type": "stage" | "row" | "done", "runner": "interview", ...} events.
    Raises the first LLM error when the main arm got no answer and every failure was an error."""
    if stimulus_batching not in STIMULUS_BATCHING:
        raise ValueError(f"stimulus_batching must be one of {STIMULUS_BATCHING}")
    question = str(question or "").strip()
    if not question:
        raise ValueError("question is empty")
    if int(max_chars) < 1:
        raise ValueError("max_chars must be >= 1")
    max_chars = int(max_chars)
    arm_list = _arms(arms)
    em = Emitter(on_event, "interview")
    ps = as_personas(personas)
    lang = choose_lang(lang, ps)
    seg = {p.pid: p.seg for p in ps}
    usage = Usage()
    warnings: list[str] = []
    if stimuli is not None:
        if isinstance(context, dict) or stimulus_col is not None:
            raise ValueError("pass per-person stimuli once: stimuli=, context={pid: text} or stimulus_col=")
        _, stim = _stimuli(ps, dict(stimuli), None, warnings)
        shared = context or ""
    else:
        shared, stim = _stimuli(ps, context, stimulus_col, warnings)
    if stimulus_col is not None:
        ps = [_without_field(p, stimulus_col) for p in ps]
    ask_kw = {"refusal_retry": refusal_retry, "cache_refusals": cache_refusals, "empty_retry": empty_retry,
              "empty_retry_kw": empty_retry_kw}
    par = _resolve_parallel(llm, parallel)
    ctrl = [a for a in arm_list if a != "persona"]
    plan = control_plan(len(ps), batch, ctrl)
    answers: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    for arm in arm_list:
        if arm == "persona":
            cp, b = ps, batch
        else:
            cp, b = control_personas(arm, ps, lang, plan[arm]["n"]), plan[arm]["batch"]
        em.stage(arm, n=len(cp), calls=plan[arm]["calls"] if arm in plan else plan["_main_calls"])
        answers[arm], stats[arm] = _arm(llm, cp, question, lang, b, shared, stim, stimulus_batching, max_chars,
                                        usage, tag=f"interview:{arm}",
                                        llm_tag=f"{tag}:interview:{arm}" if tag else f"interview:{arm}",
                                        parallel=par, ask_kw=ask_kw, answer_refusal=answer_refusal)
        if not cp and arm == "persona_swap":
            stats[arm]["skipped"] = "persona_swap needs at least 2 personas"
            warnings.append("persona_swap skipped: it needs at least 2 personas")
        for pid, t in answers[arm].items():
            em.row(pid, seg[pid], t, arm=arm)
    main_arm = arm_list[0]
    mi = stats[main_arm]
    errors = [e for d in stats.values() for e in d.pop("_errors")]
    if not answers[main_arm] and mi["error"] and not mi["refused"] and not mi["empty"]:
        raise next(e for e in errors)      # nobody answered and every batch failed: a real failure
    if usage.empty_calls or usage.error_calls:
        warnings.append(f"{usage.empty_calls} empty replies ({usage.empty_recovered} recovered by a larger-max_tokens "
                        f"re-ask) and {usage.error_calls} failed calls; their personas count as error. Last: "
                        f"{usage.last_error}. For empty replies pass thinking='off' or a larger max_tokens.")
    missing = mi["refused"] + mi["empty"] + mi["error"]
    if missing:
        hint = ""
        if not answers[main_arm] and isinstance(llm, FakeLLM) and llm.responder is None:
            hint = "; FakeLLM has no responder; use FakeLLM(simengine.sim.schema_responder())"
        inside = f", {mi['refused_in_answer']} of them inside the answer" if mi["refused_in_answer"] else ""
        warnings.append(f"{missing} personas without an answer in the {main_arm} arm ({mi['refused']} refused"
                        f"{inside}, {mi['empty']} empty, {mi['error']} error){hint}")
    if mi["refused_calls"]:
        warnings.append(f"{mi['refused_calls']}/{mi['calls']} {main_arm}-arm batches refused at the first ask; "
                        f"{mi['recovered_calls']} recovered by the research-forecast re-ask" if refusal_retry else
                        f"{mi['refused_calls']}/{mi['calls']} {main_arm}-arm batches refused (refusal_retry off)")
    cut = sum(d["truncated"] for d in stats.values())
    if cut:
        warnings.append(f"{cut} answers longer than max_chars={max_chars} were cut")
    out = {"runner": "interview", "question": question, "lang": lang, "max_chars": max_chars, "main_arm": main_arm,
           "n_personas": len(ps), "n_answered": len(answers[main_arm]), "answers": answers[main_arm],
           "controls": {"arms": ctrl, "plan": plan, "answers": {a: v for a, v in answers.items() if a != main_arm}},
           "arm_stats": stats, "segments": seg, "warnings": warnings, "usage": usage.as_dict(),
           **{k: mi[k] for k in ("refused", "empty", "error", "refused_ids", "empty_ids", "error_ids",
                                 "refused_in_answer", "refusal_rate", "truncated")}}
    if stim is not None:
        out["stimuli"] = {p.pid: stim.get(p.pid) for p in ps}
    em.done(n=len(answers[main_arm]))
    return to_jsonable(out)


def interview_frame(result: dict, truth: dict | None = None) -> pd.DataFrame:
    """One row per person x arm: unit "<pid>|<arm>", truth (from {pid: value}, else None), pred_text (the answer,
    None when there is none), status (answered | refused | empty | error | not_asked; not_asked = outside a
    subsampled persona_swap arm or a skipped one), chars, arm, segment, item (= arm), group (= pid), and
    `stimulus` when the run had per-person stimuli."""
    truth = truth or {}
    seg = result.get("segments", {})
    stats = result.get("arm_stats", {})
    main = result.get("main_arm", "persona")
    arms = {main: result.get("answers", {}), **(result.get("controls", {}).get("answers") or {})}
    stimuli = result.get("stimuli")
    rows = []
    for arm in stats or arms:
        st = stats.get(arm, {})
        status = {pid: k for k in ("refused", "empty", "error") for pid in st.get(f"{k}_ids", [])}
        cv = arms.get(arm, {})
        for pid in seg:
            t = cv.get(pid)
            s = "answered" if t is not None else status.get(pid, "not_asked")
            r = {"unit": f"{pid}|{arm}", "truth": truth.get(pid), "pred_text": t, "status": s,
                 "chars": len(t) if t is not None else 0, "arm": arm, "segment": seg[pid], "item": arm,
                 "group": pid}
            if stimuli is not None:
                r["stimulus"] = stimuli.get(pid)
            rows.append(r)
    df = pd.DataFrame(rows, columns=["unit", "truth", "pred_text", "status", "chars", "arm", "segment", "item",
                                     "group"] + (["stimulus"] if stimuli is not None else []))
    # object dtype keeps None: pandas 3's inferred str dtype would turn a missing answer / truth / stimulus into NaN.
    # pred_text always; truth and stimulus only when inferred as str (numeric truth stays float with NaN, as in 2.x)
    for c in [c for c in ("truth", "pred_text", "stimulus") if c in df.columns]:
        if c == "pred_text" or isinstance(df[c].dtype, pd.StringDtype):
            df[c] = pd.Series([r.get(c) for r in rows], index=df.index, dtype=object)
    return df


__all__ = ["interview", "interview_frame", "ARMS_DEFAULT", "STATUSES"]
