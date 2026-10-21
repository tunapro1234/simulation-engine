"""Small local answer parser + retry-with-feedback loop for one numerical or multiple-choice answer.

Kept inside `games/` on purpose (no dependency on `simengine.sim`). The logic follows the replicant FormController
idea (validate, then re-ask with the error text in the prompt) but stays strict: the answer must be a JSON
object {"answer": ...}, or the whole reply must be a bare number / an exact option. There is no substring or
first-number salvage (that salvage reads "yes" inside "no, yes ..."). A reply that cannot be used stays missing.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_BARE_NUM = re.compile(r"^\s*\$?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars|tokens|points)?\s*\.?\s*$", re.I)
ANSWER_KEYS = ("answer", "decision", "value", "choice", "prediction")


def instruction(spec: dict) -> str:
    """The answer-format line appended to every prompt."""
    if spec["kind"] == "mc":
        opts = ", ".join(f'"{o}"' for o in spec["options"])
        return f'Choose exactly one of: {opts}. Reply with JSON only: {{"answer": "<your choice>"}}'
    lo, hi = spec.get("min"), spec.get("max")
    rng = f" from {_fmt(lo)} to {_fmt(hi)}" if lo is not None and hi is not None else ""
    return f'Answer with a single number{rng}. Reply with JSON only: {{"answer": <number>}}'


def _fmt(x: Any) -> str:
    return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)


def _json_obj(text: str) -> Any:
    m = _FENCE.search(text)
    s = m.group(1) if m else text
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                v = dec.raw_decode(s[i:])[0]
            except json.JSONDecodeError:
                continue
            if isinstance(v, dict):
                return v
    return None


def _to_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, str):
        m = _BARE_NUM.match(v)
        if m:
            return float(m.group(1).replace(",", ""))
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().strip("\"'`.").strip()).lower()


def parse_answer(text: Any, spec: dict) -> tuple[Any, str | None]:
    """Return (value, error). value is None when nothing usable was found. An out-of-range number is returned
    WITH an error: the caller keeps it as an invalid decision (b01 rule: kept and flagged, never dropped)."""
    if text is None or str(text).strip() == "":
        return None, "empty reply"
    s = str(text)
    obj = _json_obj(s)
    raw: Any = None
    if isinstance(obj, dict):
        low = {str(k).strip().lower(): v for k, v in obj.items()}
        for k in ANSWER_KEYS:
            if k in low:
                raw = low[k]
                break
        if raw is None and len(obj) == 1:
            raw = next(iter(obj.values()))
        if raw is None:
            return None, 'JSON reply has no "answer" field'
    else:
        raw = s
    if spec["kind"] == "mc":
        if not isinstance(raw, str):
            return None, f"answer must be one of the options, got {raw!r}"
        opts = list(spec["options"])
        if raw in opts:
            return raw, None
        hit = [o for o in opts if _norm(o) == _norm(raw)]
        if len(hit) == 1:
            return hit[0], None
        return None, f"answer {raw.strip()[:60]!r} is not one of the options"
    v = _to_number(raw)
    if v is None:
        return None, "answer is not a number"
    v = int(v) if float(v).is_integer() else v
    lo, hi = spec.get("min"), spec.get("max")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        return v, f"answer {v} is outside {_fmt(lo)}-{_fmt(hi)}"
    return v, None


@dataclass
class Answer:
    value: Any = None
    valid: bool = False
    attempts: int = 0
    error: str | None = None
    raw_len: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: list = field(default_factory=list)


def feedback_prompt(user: str, error: str, spec: dict) -> str:
    return f"{user}\n\nYour previous reply could not be used ({error}). {instruction(spec)}"


def ask_many(llm, items: Sequence[tuple[str, str, dict, dict]], *, retries: int = 2, max_tokens: int = 1024,
             parallel: int = 4, **kw: Any) -> list[Answer]:
    """items: (system, user, spec, extra_kw). The user text must already end with `instruction(spec)`.

    One batched `llm.map` per round; only failed items are re-asked, with the error text appended (at most
    `retries` extra rounds). A run of failures ends with value None (unparseable) or the last out-of-range
    number, valid=False, and the errors kept."""
    out = [Answer() for _ in items]
    todo = list(range(len(items)))
    prompts = {i: items[i][1] for i in todo}
    for attempt in range(retries + 1):
        if not todo:
            break
        groups: dict[tuple, list[int]] = {}
        for i in todo:
            key = tuple(sorted((items[i][3] or {}).items()))
            groups.setdefault(key, []).append(i)
        nxt = []
        for key, idx in groups.items():
            resp = llm.map([(items[i][0], prompts[i]) for i in idx], max_tokens=max_tokens, parallel=parallel,
                           **kw, **dict(key))
            for i, r in zip(idx, resp):
                a = out[i]
                a.attempts += 1
                a.cost_usd += float(getattr(r, "cost_usd", 0.0) or 0.0)
                a.input_tokens += int(getattr(r, "input_tokens", 0) or 0)
                a.output_tokens += int(getattr(r, "output_tokens", 0) or 0)
                text = getattr(r, "text", None)
                a.raw_len = len(text or "")
                value, err = parse_answer(text, items[i][2])
                if err is None:
                    a.value, a.valid, a.error = value, True, None
                else:
                    a.errors.append({"attempt": a.attempts, "error": err, "last_response_len": a.raw_len})
                    a.error = err
                    a.value = value              # None, or the out-of-range number (kept, flagged)
                    nxt.append(i)
                    prompts[i] = feedback_prompt(items[i][1], err, items[i][2])
        todo = nxt
    return out


__all__ = ["instruction", "parse_answer", "ask_many", "Answer", "feedback_prompt", "ANSWER_KEYS"]
