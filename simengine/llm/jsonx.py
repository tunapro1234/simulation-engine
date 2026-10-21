"""Robust JSON extraction from LLM text.

Merges three research parsers: talp73/kod/llm/openrouter.py `parse_json` (```json fences, trailing
commas), sahne/kod/sahne/kaynak.py `json_coz` (first {/[ via raw_decode, salvage complete objects
from a truncated array) and makro/llmabm/kod/llm.py `sor_json` (re-ask once with "JSON only").
"""
from __future__ import annotations

import json
import re
from typing import Any

_MISSING = object()
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_FLAT_OBJ = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")   # objects nested at most one level


def _raw(s: str):
    try:
        return json.JSONDecoder().raw_decode(s)[0]
    except json.JSONDecodeError:
        pass
    fixed = _TRAILING_COMMA.sub(r"\1", s)
    if fixed != s:
        try:
            return json.JSONDecoder().raw_decode(fixed)[0]
        except json.JSONDecodeError:
            pass
    return _MISSING


def _salvage(s: str) -> list:
    out = []
    for m in _FLAT_OBJ.finditer(s):
        try:
            out.append(json.loads(_TRAILING_COMMA.sub(r"\1", m.group(0))))
        except json.JSONDecodeError:
            pass
    return out


def _try(s: str, max_starts: int = 50):
    s = s.strip()
    starts = [i for i, c in enumerate(s) if c in "{["][:max_starts]
    if not starts:
        return _MISSING
    v = _raw(s[starts[0]:])
    if v is not _MISSING:
        return v
    if s[starts[0]] == "[":              # truncated array: keep its complete objects
        got = _salvage(s[starts[0]:])
        if got:
            return got
    for i in starts[1:]:
        v = _raw(s[i:])
        if v is not _MISSING:
            return v
    return _MISSING


def json_extract(text: str | None, default: Any = _MISSING) -> Any:
    """Return the first JSON value (object or array) found in `text`.

    Order: each ``` fenced block, then the whole text. Inside each: the value starting at the first
    '{' / '[' (prose before and after is ignored; trailing commas are tolerated); if that is a
    truncated array, its complete objects are salvaged as a list; otherwise later '{' / '['
    positions are tried. Raises ValueError when nothing parses,
    unless `default` is given.
    """
    if text:
        for block in _FENCE.findall(text):
            v = _try(block)
            if v is not _MISSING:
                return v
        v = _try(text)
        if v is not _MISSING:
            return v
    if default is not _MISSING:
        return default
    raise ValueError("no JSON found in LLM output")


JSON_RETRY_SUFFIX = "\n\nReturn only valid JSON, with no other text."


def complete_json(llm, system: str, user: str, *, attempts: int = 2, **kw) -> Any:
    """Call any `LLM` and parse its JSON. On a parse failure re-ask once with a 'JSON only' suffix
    (a different prompt, so a different cache key). Raises ValueError after `attempts`."""
    last: Exception | None = None
    for k in range(max(1, attempts)):
        u = user if k == 0 else user + JSON_RETRY_SUFFIX
        r = llm.complete(system, u, **kw)
        try:
            return json_extract(r.text)
        except ValueError as e:
            last = e
    raise ValueError(f"no JSON after {attempts} attempts: {last}")


__all__ = ["json_extract", "complete_json", "JSON_RETRY_SUFFIX"]
