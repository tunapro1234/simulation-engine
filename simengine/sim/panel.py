"""Batched persona questioning over the `LLM` protocol (port of sahne/kaynak.py `PersonaSource.sor`).

One prompt carries `batch` personas; the model returns one JSON object per persona id. The prompt text
(SYSTEM_ASK, ASK_TEMPLATE) is the one used in the measured sahne runs (Haiku 5.5, 21 Oct).

    answers = ask_panel(llm, personas, "Rate this ad 1-10 ...", {"score": "int 1-10", "reason": "str"})
    # {pid: {"id": pid, "score": 7, "reason": "..."}}

Caching, budgets, retries and parallelism belong to the LLM implementation (simengine.llm). Here: prompt
building, robust parsing, id matching, usage accounting. A persona whose answer is missing or unparseable is
simply absent from the result and counted in `Usage.missing`.

Soft refusals (a plain-text "I can't produce this ..." with no JSON, stop_reason end_turn) are detected,
kept out of the disk cache (the backend's `cache_if=` hook) and re-asked once with a research-forecast framing
(`REFRAME_NOTE`). A batch still refused after that is counted in `Usage.refused`, not in `Usage.missing`.
An empty reply (`EmptyResponseError`: thinking used the whole max_tokens) is re-asked once with a larger
max_tokens; a batch that still has no reply (empty, or another `LLMError` slot) is counted in `Usage.missing`
(and `Usage.empty_calls` / `Usage.error_calls`) instead of aborting the run. Only when no prompt got any reply
is the first error raised. `ask_many()` is the same call path for the other runners (distribution, prediction, A/B).
The ledger tag of every call is `llm_tag` (default: `tag`), joined to the backend's own tag if it has one:
simengine's backends get `llm.map(tag=<llm_tag>)` and combine it with their own tag via `simengine.llm.compose_tag`
("<backend tag>:<llm_tag>", the backend tag appears exactly once); any other backend with a `.tag` gets
`llm.map(tag="<backend tag>:<llm_tag>")`. simengine's ledger then prepends its `tag_prefix` (env
SIMENGINE_SPEND_TAG_PREFIX) to the TSV tag column. Example:
`Claude(tag="L1", tag_prefix="d3/")` + survey(tag="dev") writes "d3/L1:dev:survey:persona".

`schema_responder()` is a deterministic offline responder for `FakeLLM` (port of sahne `SahteKaynak`): it reads
ids and field specs back out of the prompt and fills values by hashing. Dry runs only, never a measurement.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .parse import parse_json
from .personas import PersonaView

SYSTEM_ASK = ("You simulate several distinct people. Answer as each person would, from their own situation, "
              "budget and habits; disagreement and indifference are normal. Do not make everyone positive. "
              "Output only a JSON array.")

ASK_TEMPLATE = """{task}

People:
{people}

For EACH person above return one JSON object, in a JSON array, with fields:
  "id": the person id
{fields}
Free-text fields in {lang}. Answer independently for each person. JSON array only."""

# Second try for a refused batch (d2 rehearsal, 22 Oct: Haiku 5.5 answered 59/150 survey batches with a plain-text
# refusal, "I can't ... invent answers for people who don't exist"; stop_reason end_turn, no RefusalError).
REFRAME_NOTE = ("This is a research forecast. Estimate how a person with these characteristics would most likely "
                "answer; give your best guess for every person, in the requested JSON format.")
SYSTEM_REFRAME = SYSTEM_ASK + " " + REFRAME_NOTE

# Local fallback, used only when simengine.llm.looks_like_refusal cannot be imported. Same idea, smaller: a
# first-person negation followed by a task verb, or a fabrication phrase. ask_panel applies it only to replies that
# carry no usable JSON answer, so "I can't afford it" inside a free-text field never counts.
_REFUSAL_RE = re.compile(
    r"(?:\bi\s*(?:can ?not|can't|won't|will not|(?:am|'m) (?:not able to|unable to|not comfortable)|"
    r"(?:do not|don't) feel comfortable|must decline|have to decline)\s+(?:[\w'-]+\s+){0,3}?"
    r"(?:produce|create|generate|provide|simulate|invent|fabricate|make up|role-?play|pretend|impersonate|answer|"
    r"complete|fill|help|assist|speak)\b|\bpeople who (?:don't|do not) exist\b|"
    r"\b(?:invent|fabricat|mak(?:e|ing) up)\w*\s+(?:[\w'-]+\s+){0,3}?(?:answers|responses|data)\b|"
    r"^\W*as an ai\b|\b(?:bunu|bu tür)\s+(?:\w+\s+){0,3}(?:üretemem|uyduramam|oluşturamam)|"
    r"\bmaalesef\b.{0,80}(?:yapamam|yardımcı olamam))", re.I)


def _local_refusal(text) -> bool:
    return bool(text) and bool(_REFUSAL_RE.search(str(text)[:400].replace("\u2019", "'")))


def looks_like_refusal(text) -> bool:
    """True when `text` reads like a soft refusal: simengine.llm.looks_like_refusal (imported lazily), or a small
    local regex if that import fails. ask_panel applies it only to replies without a usable JSON answer."""
    if not text:
        return False
    try:
        from ..llm import looks_like_refusal as llm_check
    except ImportError:
        return _local_refusal(text)
    try:
        return bool(llm_check(str(text)))
    except Exception:  # noqa: BLE001 - a different signature must not break the runner
        return _local_refusal(text)


# Second try for a refused non-panel prompt (segment distributions, per-person predictions, A/B votes).
REFRAME_NOTE_GENERIC = "This is a research forecast. Give your best estimate, in the requested JSON format."


def _is_error_named(e: BaseException, name: str) -> bool:
    return any(c.__name__ == name for c in type(e).__mro__)


def _is_llm_error(e: BaseException) -> bool:
    """A backend error slot (LLMError and subclasses) that should not abort a whole runner."""
    try:
        from ..llm import LLMError
    except ImportError:  # pragma: no cover
        return _is_error_named(e, "LLMError")
    return isinstance(e, LLMError) or _is_error_named(e, "LLMError")


def _is_refusal_error(e: BaseException) -> bool:
    """A backend error slot that means 'refused' (RefusalError: stop_reason refusal)."""
    try:
        from ..llm import RefusalError
    except ImportError:  # pragma: no cover
        RefusalError = ()
    return (bool(RefusalError) and isinstance(e, RefusalError)) or type(e).__name__ == "RefusalError"


@dataclass
class Usage:
    """Call / token / cost accounting summed over LLMResponse objects."""
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    missing: int = 0               # persona answers absent or unparseable (refusals not included)
    remapped: int = 0              # answers whose unknown id was mapped positionally to a missing id
    unparsed_calls: int = 0        # responses with no JSON at all that were not refusals
    model: str | None = None
    tags: dict = field(default_factory=dict)
    first_calls: int = 0           # prompts sent in the first pass (before any refusal re-ask)
    refused_calls: int = 0         # first-pass prompts answered with a refusal
    reframed_calls: int = 0        # refused prompts re-asked with REFRAME_NOTE
    recovered_calls: int = 0       # re-asked prompts that came back without a refusal
    refused: int = 0               # persona answers lost to a refusal that persisted after the re-ask
    empty_calls: int = 0           # replies with no text (EmptyResponseError), any pass
    empty_recovered: int = 0       # empty replies that came back after the larger-max_tokens re-ask
    error_calls: int = 0           # prompts left without a reply by another LLMError (after the backend's retries)
    last_error: str | None = None  # message of the last empty / error slot

    def add(self, r, tag: str | None = None):
        self.calls += 1
        self.cached += int(bool(getattr(r, "cached", False)))
        self.input_tokens += int(getattr(r, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(r, "output_tokens", 0) or 0)
        self.cost_usd += float(getattr(r, "cost_usd", 0.0) or 0.0)
        self.model = getattr(r, "model", None) or self.model
        if tag:
            self.tags[tag] = self.tags.get(tag, 0) + 1

    @property
    def refusal_rate(self) -> float:
        """Share of first-pass prompts that were refused (before the re-ask)."""
        return self.refused_calls / self.first_calls if self.first_calls else 0.0

    def as_dict(self) -> dict:
        return {"model": self.model, "calls": self.calls, "cached": self.cached,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 6), "missing": self.missing, "remapped": self.remapped,
                "unparsed_calls": self.unparsed_calls, "refused_calls": self.refused_calls,
                "reframed_calls": self.reframed_calls, "recovered_calls": self.recovered_calls,
                "refused": self.refused, "refusal_rate": round(self.refusal_rate, 4),
                "empty_calls": self.empty_calls, "empty_recovered": self.empty_recovered,
                "error_calls": self.error_calls, "last_error": self.last_error}

    def note_error(self, e: BaseException, empty: bool) -> None:
        if empty:
            self.empty_calls += 1
        else:
            self.error_calls += 1
        self.last_error = f"{type(e).__name__}: {str(e)[:200]}"


def _items(data, ids: list[str]) -> list[dict]:
    """Accept a JSON array of objects, a single object, or a {pid: {...}} mapping."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if "id" in data:
            return [data]
        if any(str(k) in ids for k in data):
            return [{**v, "id": str(k)} for k, v in data.items() if isinstance(v, dict)]
        for v in data.values():           # {"people": [...]} wrapper
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def _refused_text(text) -> bool:
    """A reply is a refusal when it holds no JSON answer object and reads like a refusal."""
    if not text:
        return False
    data = parse_json(text)
    if isinstance(data, list):
        has_answer = any(isinstance(x, dict) and x for x in data)
    else:
        has_answer = isinstance(data, dict) and bool(data)
    return not has_answer and looks_like_refusal(text)


def cache_ok(text) -> bool:
    """`cache_if=` hook for the backend: False for a refusal, so the backend returns it without caching it."""
    return not _refused_text(text)


def _is_text_spec(spec) -> bool:
    """A field spec that asks for free text ("str ...", "list ...", "text ..."), not a number / choice."""
    return str(spec).lstrip().lower().startswith(("str", "list", "text"))


def build_prompts(personas: list[PersonaView], task: str, fields: dict, *, batch: int = 10,
                  context: dict | None = None, lang: str = "en", context_label: str = "Context"
                  ) -> list[tuple[tuple[str, str], list[str]]]:
    """[((system, user), [ids in this prompt]), ...]. `context`: {pid: text} shown under that person's profile
    as '<context_label>: text'."""
    context = context or {}
    field_text = "\n".join(f'  "{k}": {v}' for k, v in fields.items())
    has_text = any(_is_text_spec(v) for v in fields.values())
    out = []
    for b in range(0, len(personas), max(1, batch)):
        group = personas[b:b + batch]
        block = "\n\n".join(f"### {p.pid}\n{p.text(lang)}" + (f"\n{context_label}: {context[p.pid]}"
                                                              if p.pid in context else "") for p in group)
        user = ASK_TEMPLATE.format(task=task, people=block, fields=field_text,
                                   lang="Turkish" if lang == "tr" else "English")
        if not has_text:
            user = re.sub(r"\nFree-text fields in \w+\. ", "\n", user, count=1)
        out.append(((SYSTEM_ASK, user), [p.pid for p in group]))
    return out


def _ledger_tag(llm, tag: str) -> str:
    """The tag passed to `llm.map`. A simengine backend (CachedBackend) composes '<backend tag>:<tag>' itself with
    `compose_tag`, so it gets the raw call tag (no prefix guessing). Any other backend with a string `.tag` gets
    '<backend tag>:<tag>' composed here (or either alone)."""
    from ..llm._base import CachedBackend
    if isinstance(llm, CachedBackend):
        return tag
    base = getattr(llm, "tag", "") or ""
    if not isinstance(base, str):
        base = ""
    return f"{base}:{tag}" if base and tag else (tag or base)


def _absorb(ids: list[str], text, out: dict, usage: Usage) -> None:
    """Parse one batch reply into `out` (id matching, positional remap) and count what is missing."""
    data = parse_json(text)
    if data is None:
        usage.unparsed_calls += 1
    items = _items(data, ids)
    got, unknown = set(), []
    for x in items:
        xid = str(x.get("id", "")).strip()
        if xid in ids and xid not in got:
            out[xid] = {**x, "id": xid}
            got.add(xid)
        else:
            unknown.append(x)
    missing = [p for p in ids if p not in got]
    # the model renumbered the ids (e.g. P001.. from a template): map positionally only if counts match
    if unknown and missing and len(unknown) == len(missing) and not got:
        for x, p in zip(unknown, missing):
            out[p] = {**x, "id": p}
        usage.remapped += len(missing)
        missing = []
    usage.missing += len(missing)


def ask_many(llm, prompts: list, *, max_tokens: int, usage: Usage | None = None, tag: str = "ask",
             llm_tag: str | None = None, parallel: int | None = None, refusal_retry: bool = True,
             cache_refusals: bool = False, reframe: str | None = None, empty_retry: bool = True,
             empty_retry_kw: dict | None = None, statuses: list | None = None, errors: list | None = None,
             **kw) -> list:
    """Shared call path for the runners: `llm.map` with error slots, soft-refusal handling and empty-reply handling.
    Returns one entry per prompt: the response object, or None when the prompt ended refused / empty / failed.

    - soft refusal (no JSON answer + `looks_like_refusal`) or a RefusalError slot: kept out of the disk cache
      (`cache_if=cache_ok` unless cache_refusals) and, with refusal_retry, re-asked once with `reframe`
      (default REFRAME_NOTE) appended to the system prompt and prepended to the user prompt.
    - EmptyResponseError slot (thinking used max_tokens): with empty_retry, re-asked once with `empty_retry_kw`
      (default {"max_tokens": max(2 * max_tokens, 4000)}, at most 32000; pass e.g. {"thinking": "off"} too).
    - any other LLMError slot: the prompt gets None (counted in usage.error_calls).
    statuses: a list that receives "ok" | "recovered" | "refused" | "empty" | "error" per prompt.
    errors: a list that receives the error objects. Without it, a call where no prompt got a reply and at least
    one failed with an error raises the first error (StopError always propagates from `llm.map`).
    Other exceptions (programming errors) are raised."""
    usage = usage if usage is not None else Usage()
    prompts = list(prompts)
    if not prompts:
        return []
    mkw: dict[str, Any] = {"tag": _ledger_tag(llm, llm_tag if llm_tag is not None else tag), "return_exceptions": True,
                           **kw}
    if parallel is not None:
        mkw["parallel"] = parallel
    if not cache_refusals:
        mkw["cache_if"] = cache_ok
    errs: list = []

    def run(ps, **over):
        rs = llm.map(ps, **{"max_tokens": max_tokens, **mkw, **over})
        if len(rs) != len(ps):
            raise RuntimeError(f"llm.map returned {len(rs)} answers for {len(ps)} prompts")
        return rs

    def classify(r) -> str:
        if isinstance(r, BaseException):
            if _is_refusal_error(r):
                return "refused"
            if not _is_llm_error(r):
                raise r
            empty = _is_error_named(r, "EmptyResponseError")
            usage.note_error(r, empty)
            errs.append(r)
            return "empty" if empty else "error"
        usage.add(r, tag)
        return "refused" if _refused_text(getattr(r, "text", None)) else "ok"

    out: list = [None] * len(prompts)
    st: list = [None] * len(prompts)
    usage.first_calls += len(prompts)
    for i, r in enumerate(run(prompts)):
        st[i] = classify(r)
        if st[i] == "ok":
            out[i] = r
        elif st[i] == "refused":
            usage.refused_calls += 1
    redo = [i for i in range(len(prompts)) if st[i] == "refused"]
    if redo and refusal_retry:
        note = reframe or REFRAME_NOTE
        usage.reframed_calls += len(redo)
        again = [((f"{prompts[i][0]} {note}" if prompts[i][0] else note), f"{note}\n\n{prompts[i][1]}") for i in redo]
        for i, r in zip(redo, run(again)):
            c = classify(r)
            if c == "ok":
                usage.recovered_calls += 1
                out[i], st[i] = r, "recovered"
            elif c != "refused":
                st[i] = c
    redo = [i for i in range(len(prompts)) if st[i] == "empty"]
    if redo and empty_retry:
        over = dict(empty_retry_kw) if empty_retry_kw else {"max_tokens": min(32000, max(2 * max_tokens, 4000))}
        for i, r in zip(redo, run([prompts[i] for i in redo], **over)):
            c = classify(r)
            if c == "ok":
                usage.empty_recovered += 1
                out[i], st[i] = r, "recovered"
            else:
                st[i] = c
    if statuses is not None:
        statuses.extend(st)
    if errors is not None:
        errors.extend(errs)
    elif errs and all(x in ("empty", "error") for x in st):
        raise errs[0]
    return out


def ask_panel(llm, personas: list[PersonaView], task: str, fields: dict, *, batch: int = 10,
              context: dict | None = None, lang: str = "en", max_tokens: int | None = None,
              usage: Usage | None = None, tag: str = "ask", parallel: int = 4, llm_tag: str | None = None,
              refusal_retry: bool = True, cache_refusals: bool = False, reframe: str | None = None,
              refused_ids: set | None = None, context_label: str = "Context", empty_retry: bool = True,
              empty_retry_kw: dict | None = None, errors: list | None = None) -> dict[str, dict]:
    """Ask every persona the same task in batched prompts. Returns {pid: {field: value}}.
    `fields`: {"score": "int 1-10", "reason": "str", "choice": "one of A|B"}; `context`: {pid: extra text}.

    llm_tag: ledger tag passed to `llm.map(tag=...)` (default `tag`; joined to the backend's own tag, see the
        module docstring).
    refusal_retry: re-ask a refused batch once with `reframe` (default REFRAME_NOTE) in system and user prompt.
    cache_refusals: False (default) passes `cache_if=cache_ok`: the backend returns a refusal without writing it
        to the disk cache (no extra call), so a rerun asks again. True leaves the refusal cached (a rerun replays
        it for free, then replays the cached reframed answer).
    refused_ids: a set that receives the ids whose batch stayed refused.
    empty_retry / empty_retry_kw: an empty reply (EmptyResponseError) is re-asked once with a larger max_tokens
        (see `ask_many`); a batch left without a reply counts in usage.missing, not as an abort.
    errors: a list that receives the error slots; without it, a call where every batch failed raises."""
    usage = usage if usage is not None else Usage()
    plan = build_prompts(personas, task, fields, batch=batch, context=context, lang=lang, context_label=context_label)
    if not plan:
        return {}
    if max_tokens is None:
        max_tokens = _panel_max_tokens(plan, fields)
    st: list = []
    rs = ask_many(llm, [p for p, _ in plan], max_tokens=max_tokens, usage=usage, tag=tag,
                  llm_tag=llm_tag if llm_tag is not None else tag, parallel=parallel, refusal_retry=refusal_retry,
                  cache_refusals=cache_refusals, reframe=reframe, empty_retry=empty_retry,
                  empty_retry_kw=empty_retry_kw, statuses=st, errors=errors)
    out: dict[str, dict] = {}
    _collect(plan, rs, st, out, usage, refused_ids)
    return out


def _panel_max_tokens(plan, fields: dict) -> int:
    """ask_panel's default max_tokens for one plan: room for the largest batch's answers."""
    return min(16000, 300 + max(len(ids) for _, ids in plan) * (40 + 90 * len(fields)))


def _collect(plan, replies: list, statuses: list, out: dict, usage: Usage, refused_ids: set | None = None) -> None:
    """Fold `ask_many` replies for a panel plan into `out` ({pid: fields}) and count missing / refused ids."""
    for (_, ids), r, c in zip(plan, replies, statuses):
        if r is not None:
            _absorb(ids, r.text, out, usage)
        elif c == "refused":
            usage.refused += len(ids)
            if refused_ids is not None:
                refused_ids.update(ids)
        else:                      # empty / error: no reply for this batch
            usage.missing += len(ids)


# ---------------------------------------------------------------- offline dry-run responder
def _h(*parts) -> float:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _fake_value(pid, name, spec, seed):
    h = _h(pid, name, seed)
    spec = str(spec)
    m = re.search(r"int\s*(-?\d+)\s*-\s*(-?\d+)", spec)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return a + int(h * (b - a + 1)) if b > a else a
    m = re.search(r"number\s*(-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)", spec)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return round(a + h * (b - a), 1) if b > a else a
    if spec.startswith("float") or "0.0-1.0" in spec or "probability" in spec:
        return round(h, 3)
    m = re.search(r"one of ([^\n]+)", spec)
    if m:
        opts = [s.strip() for s in re.split(r"\||,", m.group(1)) if s.strip()]
        return opts[int(h * len(opts)) % len(opts)]
    if spec.startswith("list"):
        return [f"(fake) {name} {int(h * 100)}"]
    return f"(fake) {name} {pid}"


def schema_responder(seed: int = 0) -> Callable[[str, str], str]:
    """FakeLLM responder that answers simengine.sim prompts with deterministic hashed values:
    panel prompts (ids from '### <id>' lines, fields from the field list), distribution prompts (option
    percentages), per-persona probability vectors, per-person prediction prompts (keyed 0-100 values or SSR
    sentences). For dry runs, not measurement."""

    def respond(system: str, user: str) -> str:
        tag = hashlib.sha256((system + "\x00" + user).encode()).hexdigest()[:6] + str(seed)
        ids = re.findall(r"^### (\S+)\s*$", user, re.M)
        if ids and "with fields:" in user:
            block = user.split("with fields:", 1)[1].split("Free-text fields", 1)[0]
            specs = dict(re.findall(r'^\s+"([^"]+)":\s*(.+)$', block, re.M))
            specs.pop("id", None)
            return json.dumps([{"id": pid, **{k: _fake_value(pid, k, v, tag) for k, v in specs.items()}}
                               for pid in ids])
        m = re.search(r"respondent IDs as keys \(([^)]*)\)", user)
        if m:
            pids = [x.strip() for x in m.group(1).split(",") if x.strip()]
            ks = [len(re.findall(r"(?:^|\s)\d+=", ln)) for ln in re.findall(r"^Q\d+\. .*$", user, re.M)]
            out = {}
            for pid in pids:
                vecs = []
                for qi, k in enumerate(ks):
                    w = [_h(tag, pid, qi, j) + 0.05 for j in range(k)]
                    vecs.append([round(100 * x / sum(w)) for x in w])
                out[pid] = {"p": vecs}
            return json.dumps(out)
        m = re.search(r"Answer options:\n((?:\d+\. .*\n?)+)", user)
        if m and "percentage" in user:
            k = len(re.findall(r"^\d+\. ", m.group(1), re.M))
            w = [_h(tag, j) + 0.05 for j in range(k)]
            return json.dumps({str(j + 1): round(100 * x / sum(w)) for j, x in enumerate(w)})
        m = re.search(r"\(key: description\):\n(.*?)\n\n", user, re.S)
        if m:
            keys = [ln.split(":", 1)[0].strip() for ln in m.group(1).splitlines() if ":" in ln]
            if "one short first-person sentence" in user:
                return json.dumps({k: ("I probably bought something like this." if _h(tag, k) > 0.5
                                       else "I probably did not buy anything like this.") for k in keys})
            return json.dumps({k: int(_h(tag, k) * 100) for k in keys})
        return "{}"

    return respond


__all__ = ["ask_panel", "ask_many", "build_prompts", "Usage", "REFRAME_NOTE_GENERIC", "schema_responder", "SYSTEM_ASK", "ASK_TEMPLATE", "REFRAME_NOTE",
           "SYSTEM_REFRAME", "looks_like_refusal", "cache_ok"]
