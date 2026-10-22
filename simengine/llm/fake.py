"""Deterministic offline LLM for tests and dry runs (no network, no cost).

    llm = FakeLLM()                                        # stable hash text for any prompt
    llm = FakeLLM(lambda system, user: '{"p": 0.5}')       # your own responder
    llm = FakeLLM.survey()                                 # schema-true answers to simengine.sim prompts
    llm = FakeLLM.survey(lambda pid, qid, spec: 70 if spec["kind"] == "numeric" else None, questions=questions)
    res = se.sim.survey(personas, questions, llm=llm)      # numeric values in range, never all "unparsed"

`survey_responder` answers the panel prompt layout that `sim.survey` / `sim.ask_panel` send ('### <id>' blocks,
the '"<field>": <spec>' list, '[<field>] <question>' lines with 'a) option' lines): every scored field
(`<f>_value` "number lo-hi", `<f>_score` "int 1-5", `<f>_choice` "one of a|b") gets a value inside its range,
from `value_fn(persona_id, question_id, spec)` or a deterministic hash. Other simengine prompts fall through to
`simengine.sim.schema_responder`. Dry runs and contract tests only, never a measurement: a value_fn built from the
real answers prints real-looking metrics before the pre-registration is locked (G2 #1), so make it synthetic.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Callable, Sequence

from ..types import LLMResponse
from ._base import Stats


class FakeLLM:
    """Returns `responder(system, user)` if given, otherwise a stable pseudo-random reply derived
    from a hash of the prompt. `calls` records every (system, user) pair. `stats` / `stats()` and `spent_usd`
    mirror the real backends (cost always 0), so a script that prints them also runs dry."""

    def __init__(self, responder: Callable[[str, str], str] | None = None, model: str = "fake"):
        self.model = model
        self.responder = responder
        self.calls: list[tuple[str, str]] = []

    @property
    def stats(self) -> Stats:
        """Same keys as `Claude.stats`; `calls` counts complete() calls, every cost is 0."""
        return Stats(calls=len(self.calls), cache_hits=0, retries=0, errors=0, cost_usd=0.0, ledger_total_usd=0.0)

    @property
    def spent_usd(self) -> float:
        return 0.0

    @classmethod
    def survey(cls, value_fn: Callable[[str, str, dict], Any] | None = None, seed: int = 0,
               model: str = "fake", questions: Any = None) -> "FakeLLM":
        """FakeLLM with `survey_responder(value_fn, seed, questions)`: schema-true offline answers for `sim.survey`."""
        return cls(survey_responder(value_fn, seed, questions), model=model)

    def complete(self, system: str, user: str, *, max_tokens: int = 256, **kw: Any) -> LLMResponse:
        self.calls.append((system, user))
        if self.responder is not None:
            text = self.responder(system, user)
        else:
            text = hashlib.sha256((system + "\x00" + user).encode()).hexdigest()[:16]
        return LLMResponse(text=text, model=self.model, input_tokens=len((system + user).split()),
                           output_tokens=len(text.split()))

    def map(self, prompts: Sequence[tuple[str, str]], *, max_tokens: int = 256, parallel: int = 4,
            **kw: Any) -> list[LLMResponse]:
        return [self.complete(s, u, max_tokens=max_tokens, **kw) for s, u in prompts]


# ---------------------------------------------------------------- survey / panel responder
_SCORED = (("_value", "numeric"), ("_score", "likert5"), ("_choice", "choice"))
_RANGE = re.compile(r"^(int|number)\s*(-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)")


def _field_specs(user: str) -> dict[str, str]:
    """{"<field>": "<spec>"} from the panel prompt's field list (between 'with fields:' and the closing lines)."""
    block = user.split("with fields:", 1)[1]
    block = re.split(r"\n(?:Free-text fields|Answer independently)", block, maxsplit=1)[0]
    specs = dict(re.findall(r'^\s+"([^"]+)":\s*(.+)$', block, re.M))
    specs.pop("id", None)
    return specs


def _questions(user: str) -> dict[str, dict]:
    """{"<field stem>": {"text", "options"}} from the '[<stem>] text' lines and their '   a) option' lines."""
    out: dict[str, dict] = {}
    cur = None
    sec = user.split("\nQuestions:\n", 1)[-1].split("\n\nPeople:\n", 1)[0]
    for ln in sec.splitlines():
        m = re.match(r"^\[([^\]]+)\] (.*)$", ln)
        if m:
            cur = out[m.group(1)] = {"text": m.group(2), "options": []}
            continue
        m = re.match(r"^\s+[a-z]\) (.*)$", ln)
        if m and cur is not None:
            cur["options"].append(m.group(1))
        elif ln.strip() and cur is not None and not ln.startswith(" "):
            cur = None
    return out


def _profiles(user: str) -> dict[str, str]:
    """{"<pid>": "<profile text shown under '### <pid>'>"} from the People section."""
    sec = user.split("\nPeople:\n", 1)[-1].split("\n\nFor EACH person", 1)[0]
    parts = re.split(r"^### (\S+)\s*$", sec, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def _spec(field: str, raw: str, qs: dict) -> dict | None:
    """Spec dict for a scored field ({"field", "stem", "kind", "text", "spec", + range or options}), None for
    free-text fields."""
    for suf, kind in _SCORED:
        if not field.endswith(suf):
            continue
        stem = field[:-len(suf)]
        q = qs.get(stem, {})
        d = {"field": field, "stem": stem, "kind": kind, "text": q.get("text", ""), "spec": raw}
        if kind == "choice":
            letters = re.search(r"one of\s*(\S+)", raw)
            letters = letters.group(1).split("|") if letters else []
            opts = q.get("options") or letters
            d.update(options=list(opts), letters=letters[:len(opts)] or [chr(97 + j) for j in range(len(opts))])
        else:
            m = _RANGE.match(raw)
            if not m:
                return None
            lo, hi = float(m.group(2)), float(m.group(3))
            d.update(min=int(lo) if lo.is_integer() else lo, max=int(hi) if hi.is_integer() else hi,
                     integer=m.group(1) == "int")
        return d
    return None


def _coerce(v: Any, d: dict, h: float, strict: bool = False):
    """A value_fn (`strict`: a bad value raises) or schema_responder value (None or unusable -> the hash `h` in
    [0, 1)) as an in-range field value."""
    if d["kind"] == "choice":
        opts, letters = d["options"], d["letters"]
        if not letters:
            return v
        j = None
        num = isinstance(v, (int, float)) and not isinstance(v, bool) and float(v).is_integer()
        s = str(int(v)) if num else (v.strip() if isinstance(v, str) else None)
        if s is not None and s in opts:              # option text first: options [1..5] and value 3 mean "3"
            j = opts.index(s)
        elif isinstance(v, str) and s.lower() in letters:
            j = letters.index(s.lower())
        elif num:
            j = int(v)
        if j is None or not 0 <= j < len(letters):
            if strict:
                raise ValueError(f"value_fn returned {v!r} for {d['field']}; give an option, its letter or index")
            j = int(h * len(letters)) % len(letters)
        return letters[j]
    lo, hi = d["min"], d["max"]
    try:
        v = float(v)
        if math.isnan(v):
            raise ValueError
    except (TypeError, ValueError):
        if strict:
            raise ValueError(f"value_fn returned {v!r} for {d['field']}; give a number") from None
        v = lo + h * (hi - lo)
    if d["integer"]:
        return int(min(max(round(v), math.ceil(lo)), math.floor(hi)))
    v = min(max(v, lo), hi)
    return int(v) if v.is_integer() else round(v, 4)


def _unit(*parts) -> float:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 0x100000000


def survey_responder(value_fn: Callable[[str, str, dict], Any] | None = None, seed: int = 0, questions: Any = None
                     ) -> Callable[[str, str], str]:
    """FakeLLM responder for `sim.survey` / `sim.ask_panel` prompts with values inside every field's range.

    value_fn(persona_id, question_id, spec) -> value or None. question_id is the question's own id when
    `questions` (the list given to sim.survey) is passed, otherwise the field stem the prompt uses (the id cut to a
    plain identifier of at most 20 characters, see `sim._stats.field_name`: 'q-price' -> 'q_price', '1' -> 'x0_1').
    spec: {"field", "stem", "kind" ("numeric", "likert5", "choice"), "text", "spec", "arm", "profile"; numeric /
    likert5: "min", "max", "integer"; choice: "options" (texts), "letters"}. A number is clamped into [min, max]
    (rounded for int fields); a choice may be the option text (tried first, so options [1, 2, 3] and 3 give "3"),
    its letter or its index. None (or no value_fn) -> `sim.schema_responder(seed)`'s deterministic hashed value,
    clamped the same way. Free text comes from schema_responder too, and so does every non-panel prompt.

    Control arms keep the real persona ids. The no_persona arm (arm "no_persona": the prompt shows only a neutral
    adult) never calls value_fn and gets the hashed default, so a pid-keyed value_fn cannot leak the person into
    it. persona_swap (arm "persona") shows the slot's pid with another persona's profile; key on spec["profile"]
    (the text the LLM sees) if that control should follow the profile, as an LLM would.
    """
    from ..sim.panel import schema_responder          # lazy: simengine.sim imports this module
    from ..sim.personas import NEUTRAL_TEXT
    fallback = schema_responder(seed)
    neutral = set(NEUTRAL_TEXT.values())
    qid_of: dict[str, str] = {}
    if questions is not None:
        from ..sim.survey import _questions as norm_questions
        qid_of = {d["field"]: d["id"] for d in norm_questions(questions)}

    def respond(system: str, user: str) -> str:
        ids = re.findall(r"^### (\S+)\s*$", user, re.M)
        if not (ids and "with fields:" in user):
            return fallback(system, user)
        specs, qs, prof = _field_specs(user), _questions(user), _profiles(user)
        rows = {str(r.get("id")): r for r in json.loads(fallback(system, user)) if isinstance(r, dict)}
        out = []
        for pid in ids:
            row = dict(rows.get(pid) or {"id": pid})
            text = prof.get(pid, "")
            arm = "no_persona" if text.split("\n", 1)[0].strip() in neutral else "persona"
            for f, raw in specs.items():
                d = _spec(f, raw, qs)
                if d is None:      # free text / list: schema_responder's placeholder
                    row.setdefault(f, f"(fake) {f} {pid}")
                    continue
                d.update(arm=arm, profile=text)
                v = (value_fn(pid, qid_of.get(d["stem"], d["stem"]), dict(d))
                     if value_fn is not None and arm != "no_persona" else None)
                row[f] = _coerce(row.get(f) if v is None else v, d, _unit(seed, pid, f), strict=v is not None)
            out.append(row)
        return json.dumps(out)

    return respond
