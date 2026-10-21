"""Schema-constrained answers with validate-and-reprompt feedback.

Port of replicant `src/replicant/otree/bot.py` (pyreplicant 0.3.0, MIT): `FormController.validate`,
`LLMBot._decide_batch`, `_decide_sequential`, `_build_feedback`, `_build_prompt`, `_parse_json_response` and the
retry limit `MAX_LLM_RETRIES = 8`. The oTree client, OpenRouter/`requests` calls and the 429 retry are not
ported: the LLM goes through the `simengine.types.LLM` protocol, and `simengine.llm` owns backoff and budgets.

    fields = [FieldSpec("contribution", "int", min=0, max=20, label="Tokens you put in the group account"),
              FieldSpec("vote", "choice", choices=[("1", "Yes"), ("0", "No")])]
    res = ask_constrained(llm, persona_text, game_text, fields, meta={"condition": "random", "agent_id": "a1"})
    res.completed, res.values, res.failures       # {"contribution": 12, "vote": "1"} when completed
    completion_summary(all_results, by="condition")  # n_planned, n_completed, failed_<field>, missing_<field>

A field that never validates stays missing. Nothing is imputed, salvaged from free text or defaulted: the
failure is logged (`failures`) and counted (`completion_summary`). replicant's `_try_direct_value` (substring and
first-number salvage, which reads "yes" inside "no, yes ...") is deliberately dropped.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .panel import Usage
from .parse import parse_json

MAX_RETRIES = 8                 # replicant MAX_LLM_RETRIES
SEQUENTIAL_THRESHOLD = 4        # replicant: <= 3 fields in one call, 4+ fields one at a time
KINDS = ("int", "float", "choice", "text")

# replicant _build_prompt system tail, verbatim. The persona text (caller's `system`) goes in front of it.
JSON_ONLY = ("CRITICAL INSTRUCTION: Respond with ONLY a valid JSON object. "
             "No explanation, no markdown code blocks, no text before or after. "
             "Just the raw JSON object.")
PARSE_FEEDBACK = ("Your response could not be parsed as JSON. "
                  "Respond with ONLY a JSON object, no markdown or explanation.")


@dataclass(frozen=True)
class FieldSpec:
    """One answer field. `kind`: "int" | "float" | "choice" | "text".

    `choices` (choice only): list of values, list of (value, display) pairs, or a {value: display} dict. The
    prompt shows the display texts; the model may answer with either, case-insensitively, and the cleaned answer
    is always the value. `min` / `max` are inclusive numeric bounds. `label` is the question shown to the model
    (default: the field name with underscores as spaces)."""
    name: str
    kind: str = "text"
    min: float | None = None
    max: float | None = None
    choices: Any = None
    label: str = ""

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("FieldSpec.name must be a non-empty string")
        if self.kind not in KINDS:
            raise ValueError(f"FieldSpec.kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind == "choice":
            pairs = _choice_pairs(self.choices)
            if not pairs:
                raise ValueError(f"choice field {self.name!r} needs choices")
            object.__setattr__(self, "choices", pairs)
        elif self.choices is not None:
            raise ValueError(f"choices given for non-choice field {self.name!r}")
        if self.min is not None and self.max is not None and float(self.min) > float(self.max):
            raise ValueError(f"field {self.name!r}: min > max")

    @property
    def values(self) -> list[str]:
        return [v for v, _ in self.choices or ()]

    @property
    def displays(self) -> list[str]:
        return [d for _, d in self.choices or ()]


def _choice_pairs(choices) -> tuple[tuple[str, str], ...]:
    if choices is None:
        return ()
    if isinstance(choices, Mapping):
        items = list(choices.items())
    else:
        items = [c if isinstance(c, (tuple, list)) and len(c) == 2 else (c, c) for c in choices]
    return tuple((_as_str(v), str(d)) for v, d in items)


def _as_str(v) -> str:
    if isinstance(v, float) and math.isfinite(v) and v == int(v):
        v = int(v)
    return str(v).strip()


def as_fields(fields) -> list[FieldSpec]:
    """FieldSpec list from FieldSpecs or plain dicts ({"name", "kind", "min", "max", "choices", "label"})."""
    if isinstance(fields, (FieldSpec, Mapping)):
        fields = [fields]
    out = [f if isinstance(f, FieldSpec) else FieldSpec(**dict(f)) for f in fields]
    names = [f.name for f in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate field names: {names}")
    return out


# ---------------------------------------------------------------- validation (FormController.validate)

def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and (not v.strip() or v.strip().lower() == "nan")


def _num(v, f: FieldSpec):
    """(number, None) or (None, error). Plain numbers or numeric strings only; "7 tokens" is an error."""
    if isinstance(v, bool) or isinstance(v, (dict, list, tuple)):
        return None, f"{f.name}: '{v}' is not a number"
    try:
        x = float(str(v).strip()) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        return None, f"{f.name}: '{v}' is not a number"
    if not math.isfinite(x):
        return None, f"{f.name}: '{v}' is not a finite number"
    if f.kind == "int":
        if x != int(x):
            return None, f"{f.name}: {x:g} is not a whole number"
        x = int(x)
    if f.min is not None and x < float(f.min):
        return None, f"{f.name}: {x} is below minimum {_fmt(f.min)}"
    if f.max is not None and x > float(f.max):
        return None, f"{f.name}: {x} is above maximum {_fmt(f.max)}"
    return x, None


def _fmt(b) -> str:
    return f"{b:g}" if isinstance(b, float) else str(b)


def _validate(answers: Any, fields: Sequence[FieldSpec]) -> tuple[dict, dict]:
    """(clean {name: value}, errors {name: message})."""
    answers = answers if isinstance(answers, Mapping) else {}
    clean, errors = {}, {}
    for f in fields:
        v = answers.get(f.name)
        if _is_missing(v):
            errors[f.name] = f"Missing required field: {f.name}"
            continue
        if f.kind == "choice":
            s = _as_str(v) if not isinstance(v, (dict, list)) else None
            hit = None
            if s is not None:
                hit = next((val for val, _ in f.choices if val == s), None)
                if hit is None:
                    hit = next((val for val, d in f.choices if d.strip() == s), None)
                if hit is None:            # case-insensitive fallback, value or display text
                    low = s.lower()
                    hit = next((val for val, d in f.choices if d.strip().lower() == low or val.lower() == low),
                               None)
            if hit is None:
                errors[f.name] = f"{f.name}: '{v}' is not a valid choice. Options: {f.displays}"
            else:
                clean[f.name] = hit
        elif f.kind in ("int", "float"):
            x, err = _num(v, f)
            if err:
                errors[f.name] = err
            else:
                clean[f.name] = x
        else:
            if isinstance(v, (dict, list)):
                errors[f.name] = f"{f.name}: expected text, got a JSON {type(v).__name__}"
            else:
                clean[f.name] = str(v).strip()
    return clean, errors


def validate_answers(answers: Any, fields) -> tuple[dict, list[str]]:
    """Validate a parsed answer dict against field specs. Returns (clean, errors).

    Missing (None, "", "nan", NaN) is an error. Choice answers match the value, then the display text, then
    either case-insensitively; the clean value is the choice value. Numbers must parse as a whole number or
    numeric string (no unit or prose salvage); "int" fields reject fractions; bounds are inclusive. Text is kept
    as stripped text. Fields with an error are absent from `clean` (never imputed)."""
    clean, errors = _validate(answers, as_fields(fields))
    return clean, list(errors.values())


# ---------------------------------------------------------------- prompts (LLMBot._build_prompt / _build_feedback)

def _describe(f: FieldSpec) -> str:
    label = f.label or f.name.replace("_", " ")
    if f.kind == "choice":
        options = ", ".join(f'"{d}"' for d in f.displays)
        return f'- "{f.name}": {label} (choose exactly one of: {options})'
    if f.kind in ("int", "float"):
        word = "integer" if f.kind == "int" else "number"
        if f.min is not None and f.max is not None:
            bounds = f" ({word} from {_fmt(f.min)} to {_fmt(f.max)})"
        elif f.min is not None:
            bounds = f" ({word}, min: {_fmt(f.min)})"
        elif f.max is not None:
            bounds = f" ({word}, max: {_fmt(f.max)})"
        else:
            bounds = f" ({word})"
        return f'- "{f.name}": {label}{bounds}'
    return f'- "{f.name}": {label}'


def build_user_prompt(prompt: str, fields: Sequence[FieldSpec], feedback: str = "",
                      prior_answers: Mapping | None = None) -> str:
    """The replicant user prompt: page context, prior answers on this page, field list, feedback, example."""
    fields = as_fields(fields)
    user = f"Page context:\n{prompt}\n\n"
    if prior_answers:
        user += "Your previous answers on this page:\n" + json.dumps(dict(prior_answers), indent=2) + "\n\n"
    user += "Field to answer:\n" + "\n".join(_describe(f) for f in fields)
    if feedback:
        user += f"\n\n{feedback}"
    example = ", ".join(f'"{f.name}": ...' for f in fields)
    return user + f"\n\nRespond: {{{example}}}"


def build_feedback(errors: Sequence[str], parsed: bool, attempt: int, max_retries: int) -> str:
    """Retry feedback for the next attempt (`attempt` is the 1-based number of the next call)."""
    if not parsed:
        body = PARSE_FEEDBACK
    else:
        body = ("Your previous response had errors:\n" + "\n".join(f"  - {e}" for e in errors)
                + "\nPlease fix these and try again.")
    return body + f"\n(Attempt {attempt} of {max_retries}.)"


def _parse_object(text: str | None):
    """(dict, parsed?) from LLM text. A one-element array holding an object counts as that object."""
    data = parse_json(text) if text else None
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    if isinstance(data, dict):
        return dict(data), True
    return {}, False


# ---------------------------------------------------------------- result

@dataclass
class ConstrainedResult:
    """Outcome of one `ask_constrained` call.

    values      validated answers only ({field: value}); a failed or unreached field is absent, never filled
    attempts    LLM calls made
    failures    one entry per field without a value: {"field", "attempt", "error", "last_response_len"};
                attempt = calls spent on that field (0 = never asked because an earlier field failed)
    completed   every field has a validated value
    usage       simengine.sim.Usage (calls, tokens, cost, unparsed_calls)
    mode        "batch" | "sequential"; fields: field names in order; max_tokens: the cap used (it is a
                treatment); meta: caller's tags (condition, agent_id, persona_method, seed, wording_id ...)
    trace       per call: {"attempt", "fields", "errors", "parsed", "response_len"}"""
    values: dict
    attempts: int
    failures: list
    completed: bool
    usage: Usage
    mode: str = "batch"
    fields: list = field(default_factory=list)
    max_tokens: int | None = None
    meta: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.completed

    def as_dict(self) -> dict:
        """JSON-safe dict (usage flattened with Usage.as_dict())."""
        d = asdict(self)
        d["usage"] = self.usage.as_dict()
        return d

    def row(self) -> dict:
        """One flat answer row: meta tags, the validated values, valid, attempts, max_tokens, mode, n_failed."""
        return {**self.meta, **self.values, "valid": self.completed, "attempts": self.attempts,
                "max_tokens": self.max_tokens, "mode": self.mode, "n_failed": len(self.failures)}


# ---------------------------------------------------------------- ask

def ask_constrained(llm, system: str, prompt: str, fields, *, max_retries: int = MAX_RETRIES,
                    sequential_threshold: int | None = SEQUENTIAL_THRESHOLD, max_tokens: int = 1024,
                    system_suffix: str = JSON_ONLY, stop_on_failure: bool = True, meta: Mapping | None = None,
                    usage: Usage | None = None, **kw) -> ConstrainedResult:
    """Ask for schema-valid JSON answers, re-prompting with the validation errors until they pass.

    `system` is the persona text (the JSON-only instruction `system_suffix` is appended; pass "" to drop it);
    `prompt` is the page / game / question text. With fewer than `sequential_threshold` fields all fields go in
    one call, retried up to `max_retries` times. With more, fields are asked one at a time, each with up to
    `max_retries` calls, and every prompt carries the answers already given on this page. In sequential mode the
    page stops at the first field that fails (`stop_on_failure=True`, as in replicant); later fields are logged
    as failures with attempt 0. `sequential_threshold=None` always batches.

    Errors raised by the LLM backend (budget, auth) propagate; an empty or unparseable reply is a failed attempt.
    `**kw` goes to `llm.complete` (e.g. effort, temperature). `meta` is stored on the result only."""
    fields = as_fields(fields)
    if not fields:
        raise ValueError("ask_constrained needs at least one field")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    usage = usage if usage is not None else Usage()
    sys_text = system or ""
    if system_suffix:
        sys_text = (sys_text.rstrip() + "\n\n" + system_suffix) if sys_text.strip() else system_suffix
    sequential = sequential_threshold is not None and len(fields) >= sequential_threshold
    state = {"attempts": 0, "trace": []}

    def call(group: list[FieldSpec], feedback: str, prior: dict) -> tuple[str, dict, bool]:
        user = build_user_prompt(prompt, group, feedback, prior)
        r = llm.complete(sys_text, user, max_tokens=max_tokens, **kw)
        usage.add(r, "constrained")
        state["attempts"] += 1
        text = getattr(r, "text", None) or ""
        raw, parsed = _parse_object(text)
        if not parsed:
            usage.unparsed_calls += 1
        return text, raw, parsed

    def run(group: list[FieldSpec], prior: dict, salvage_single_key: bool):
        """Up to max_retries calls for `group`. Returns (clean, errors, calls, last_text)."""
        feedback, clean, errors, text = "", {}, {}, ""
        for k in range(1, max_retries + 1):
            text, raw, parsed = call(group, feedback, prior)
            if salvage_single_key and parsed and group[0].name not in raw and len(raw) == 1:
                raw = {group[0].name: next(iter(raw.values()))}     # replicant: value under another key
            clean, errors = _validate(raw, group)
            state["trace"].append({"attempt": state["attempts"], "fields": [f.name for f in group],
                                   "errors": list(errors.values()), "parsed": parsed, "response_len": len(text)})
            if not errors:
                return clean, {}, k, text
            if k < max_retries:
                feedback = build_feedback(list(errors.values()), parsed, k + 1, max_retries)
        return clean, errors, max_retries, text

    values, failures = {}, []
    if not sequential:
        clean, errors, n, text = run(fields, {}, salvage_single_key=len(fields) == 1)
        values.update(clean)
        failures += [{"field": name, "attempt": n, "error": err, "last_response_len": len(text)}
                     for name, err in errors.items()]
    else:
        for i, f in enumerate(fields):
            clean, errors, n, text = run([f], dict(values), salvage_single_key=True)
            if not errors:
                values.update(clean)
                continue
            failures.append({"field": f.name, "attempt": n, "error": errors[f.name],
                             "last_response_len": len(text)})
            if stop_on_failure:
                failures += [{"field": g.name, "attempt": 0, "error": f"not asked: field '{f.name}' failed",
                              "last_response_len": None} for g in fields[i + 1:]]
                break
    usage.missing += len(failures)
    return ConstrainedResult(values=values, attempts=state["attempts"], failures=failures,
                             completed=not failures and len(values) == len(fields), usage=usage,
                             mode="sequential" if sequential else "batch", fields=[f.name for f in fields],
                             max_tokens=max_tokens, meta=dict(meta or {}), trace=state["trace"])


# ---------------------------------------------------------------- completion accounting

def _get(r, key: str):
    if isinstance(r, ConstrainedResult):
        r = r.as_dict()
    if key in (r.get("meta") or {}):
        return r["meta"][key]
    return r.get(key)


def _norm(r) -> dict:
    return r.as_dict() if isinstance(r, ConstrainedResult) else dict(r)


def completion_summary(results, by: str | None = "condition", planned: int | Mapping | None = None) -> pd.DataFrame:
    """Completion table: one row per group.

    `results`: ConstrainedResults (or their `as_dict()`), grouped by `meta[by]` (falling back to a top-level key;
    "all" when absent or `by=None`), or a {group: [results]} mapping. `planned`: the number of runs that were
    planned (an int for every group or {group: n}); default = results seen, which hides runs that crashed before
    returning, so pass it when you know it. Columns: <by>, n_planned, n_results, n_completed, completion_rate
    (n_completed / n_planned), attempts, failed_<field> (failed after all retries) and missing_<field> (no value:
    failed or never reached) for every field seen. No result is dropped or imputed."""
    if isinstance(results, Mapping):
        groups = {g: [_norm(r) for r in rs] for g, rs in results.items()}
    else:
        groups: dict = {}
        for r in results:
            g = _get(r, by) if by else None
            groups.setdefault("all" if g is None else g, []).append(_norm(r))
    names: list[str] = []
    for rs in groups.values():
        for r in rs:
            for n in list(r.get("fields") or []) + [x["field"] for x in r.get("failures") or []]:
                if n not in names:
                    names.append(n)
    col = by or "group"
    rows = []
    for g, rs in groups.items():
        n_plan = planned.get(g, len(rs)) if isinstance(planned, Mapping) else (planned if planned is not None
                                                                              else len(rs))
        done = sum(bool(r.get("completed")) for r in rs)
        row = {col: g, "n_planned": int(n_plan), "n_results": len(rs), "n_completed": done,
               "completion_rate": done / n_plan if n_plan else float("nan"),
               "attempts": sum(int(r.get("attempts") or 0) for r in rs)}
        for n in names:
            fails = [x for r in rs for x in r.get("failures") or [] if x["field"] == n]
            row[f"failed_{n}"] = sum(1 for x in fails if x.get("attempt"))
            row[f"missing_{n}"] = len(fails)
        rows.append(row)
    cols = [col, "n_planned", "n_results", "n_completed", "completion_rate", "attempts"] + \
        [c for n in names for c in (f"failed_{n}", f"missing_{n}")]
    return pd.DataFrame(rows, columns=cols)


__all__ = ["FieldSpec", "ConstrainedResult", "validate_answers", "ask_constrained", "completion_summary",
           "build_user_prompt", "build_feedback", "as_fields", "MAX_RETRIES", "SEQUENTIAL_THRESHOLD", "JSON_ONLY"]
