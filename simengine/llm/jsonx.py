"""Robust JSON extraction from LLM text.

Merges three research parsers: talp73/kod/llm/openrouter.py `parse_json` (```json fences, trailing
commas), sahne/kod/sahne/kaynak.py `json_coz` (first {/[ via raw_decode, salvage complete objects
from a truncated array) and makro/llmabm/kod/llm.py `sor_json` (re-ask once with "JSON only").

    data = complete_json(llm, "Return JSON only.", prompt, max_tokens=600)
    st = {}; data = complete_json(llm, sys, prompt, max_tokens=300, stats=st)   # st["truncated"], ...
    out = complete_json_many(llm, [(sys, p) for p in prompts], max_tokens=600, stats=st)   # llm.parallel threads

Truncation. A reply that stopped at the output budget (`raw["stop_reason"] == "max_tokens"`, or, when the backend
reports no stop reason, a JSON value opened and never closed) is not a parse failure: `complete_json` asks the same
prompt again once with `max_tokens` doubled (capped at `max_tokens_cap`), and an `EmptyResponseError` (thinking ate
the budget) is retried the same way. A `max_tokens` under `JSON_WARN_BELOW` warns (`JsonBudgetWarning`): in the G9
recipe rehearsal Haiku wrote prose despite "JSON only" and 7/48 replies were cut at 300 tokens.
"""
from __future__ import annotations

import json
import re
import warnings
from typing import Any

from .errors import EmptyResponseError

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
JSON_WARN_BELOW = 400          # complete_json with max_tokens under this -> one JsonBudgetWarning
MAX_TOKENS_CAP = 4096          # a truncation retry never asks for more than this by default
TRUNCATED_STOP_REASONS = ("max_tokens", "length")
DEFAULT_MAX_TOKENS = 256       # the LLM protocol default, used when neither the call nor the backend sets one


class JsonBudgetWarning(UserWarning):
    """complete_json got a small max_tokens: a model that adds prose before the JSON gets cut off."""


def json_unclosed(text: str | None) -> bool:
    """True when the first '{' / '[' in `text` opens a JSON value that never closes (strings and escapes are
    respected), i.e. the reply was most likely cut at the output budget. False for no JSON at all."""
    if not text:
        return False
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return False
    depth, in_str, esc = 0, False, False
    for c in text[min(starts):]:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth <= 0:
                return False
    return depth > 0


def is_truncated(r) -> bool:
    """Did this LLMResponse stop at max_tokens? Uses raw["stop_reason"] when the backend reports one, else
    `json_unclosed(r.text)`."""
    stop = (getattr(r, "raw", None) or {}).get("stop_reason")
    if stop is not None:
        return stop in TRUNCATED_STOP_REASONS
    return json_unclosed(getattr(r, "text", ""))


def _takes_max_tokens(llm) -> bool:
    from .claude_code import ClaudeCode      # `claude -p` ignores max_tokens, so a bigger budget cannot help
    return not isinstance(llm, ClaudeCode)


def _count(llm, stats: dict | None, **kw: int) -> None:
    if stats is not None:
        for k, v in kw.items():
            stats[k] = stats.get(k, 0) + v
    bump = getattr(llm, "_bump", None)        # CachedBackend: shows up in llm.stats as json_<key>
    if callable(bump):
        bump(**{f"json_{k}": v for k, v in kw.items()})


def complete_json(llm, system: str, user: str, *, attempts: int = 2, truncation_retries: int = 1,
                  max_tokens_cap: int = MAX_TOKENS_CAP, stats: dict | None = None, **kw) -> Any:
    """Call any `LLM` and parse its JSON. On a parse failure re-ask once with a 'JSON only' suffix
    (a different prompt, so a different cache key). Raises ValueError after `attempts`.

    A truncated reply (`is_truncated`) or an `EmptyResponseError` is retried with the same prompt and
    max_tokens x2 (capped at `max_tokens_cap`), up to `truncation_retries` times; these retries do not use up
    `attempts`. When no larger budget is left, a truncated reply whose complete objects can be salvaged is
    returned (with a RuntimeWarning), else it counts as a failed attempt; a final EmptyResponseError is re-raised.
    `stats` (a dict) gets counts: calls, truncated, truncation_retries, parse_failures. Real backends also add
    them to `llm.stats` as json_calls, json_truncated, ... A max_tokens (from the call, else the backend's
    default) under JSON_WARN_BELOW warns once per message (JsonBudgetWarning)."""
    kw = dict(kw)
    sized = _takes_max_tokens(llm)
    mt = kw.get("max_tokens")
    if mt is None:
        mt = getattr(llm, "max_tokens", None)
    if sized and mt is not None and int(mt) < JSON_WARN_BELOW:
        warnings.warn(f"complete_json on {getattr(llm, 'model', '?')} with max_tokens={mt} (< {JSON_WARN_BELOW}): "
                      "models often write prose before the JSON despite 'JSON only' and get cut off (G9: 7/48 at 300). "
                      "Pass max_tokens=600 or more.", JsonBudgetWarning, stacklevel=2)
    budget = int(mt) if mt is not None else DEFAULT_MAX_TOKENS
    last: Exception | None = None
    k, t = 0, 0
    while k < max(1, attempts):
        u = user if k == 0 else user + JSON_RETRY_SUFFIX
        _count(llm, stats, calls=1)
        try:
            r = llm.complete(system, u, **kw)
        except EmptyResponseError:
            _count(llm, stats, truncated=1)
            if sized and t < truncation_retries and budget < max_tokens_cap:
                t += 1
                budget = min(max_tokens_cap, budget * 2)
                kw["max_tokens"] = budget
                _count(llm, stats, truncation_retries=1)
                continue
            raise
        cut = sized and is_truncated(r)
        try:
            v = json_extract(r.text)
        except ValueError as e:
            v, last = _MISSING, e
        if cut:
            _count(llm, stats, truncated=1)
            if t < truncation_retries and budget < max_tokens_cap:
                t += 1
                budget = min(max_tokens_cap, budget * 2)
                kw["max_tokens"] = budget
                _count(llm, stats, truncation_retries=1)
                continue
            if v is not _MISSING:
                warnings.warn(f"complete_json: reply still cut at max_tokens={budget}; returning the salvaged part",
                              RuntimeWarning, stacklevel=2)
                return v
            last = ValueError(f"reply cut at max_tokens={budget} (truncated JSON)")
        elif v is not _MISSING:
            return v
        _count(llm, stats, parse_failures=1)
        k += 1
    raise ValueError(f"no JSON after {attempts} attempts: {last}")


DEFAULT_PARALLEL = 4           # complete_json_many on a backend without a `parallel` attribute (FakeLLM)


def _request(req) -> tuple[str, str, dict]:
    if isinstance(req, dict):
        extra = {k: v for k, v in req.items() if k not in ("system", "user")}
        return req.get("system", ""), req["user"], extra
    if isinstance(req, (tuple, list)) and len(req) == 2:
        return req[0], req[1], {}
    raise TypeError(f"complete_json_many: a request is (system, user) or a dict with 'user', got {type(req).__name__}")


def complete_json_many(llm, requests, *, parallel: int | None = None, attempts: int = 2,
                       return_exceptions: bool = True, stats: dict | None = None, **kw) -> list[Any]:
    """`complete_json` for each request on `parallel` threads (default `llm.parallel`, else 4); results in input order.

    `requests`: (system, user) pairs or dicts {"system", "user", **per-request kwargs}; per-request kwargs override
    `kw` (max_tokens, tag, truncation_retries, ...). A failed request (ValueError after `attempts`, a final
    EmptyResponseError, any ordinary exception) does not sink the batch: its slot holds the exception
    (return_exceptions=True) or None. A `StopError` (budget, 401, spend limit) cancels the pending requests and is
    raised. `stats` gets the per-request complete_json counts summed (calls, truncated, truncation_retries,
    parse_failures) plus `requests` and `failed`; each request counts into its own dict, so threads never share one."""
    from .parallel import parallel_map
    reqs = [_request(r) for r in requests]
    if parallel is None:
        parallel = getattr(llm, "parallel", None) or DEFAULT_PARALLEL
    per: list[dict] = [{} for _ in reqs]

    def one(i: int) -> Any:
        system, user, extra = reqs[i]
        return complete_json(llm, system, user, attempts=attempts, stats=per[i], **{**kw, **extra})

    out = parallel_map(one, range(len(reqs)), parallel=parallel, return_exceptions=True)
    failed = sum(isinstance(v, Exception) for v in out)
    if stats is not None:
        for d in per + [{"requests": len(reqs), "failed": failed}]:
            for k, v in d.items():
                stats[k] = stats.get(k, 0) + v
    if not return_exceptions:
        out = [None if isinstance(v, Exception) else v for v in out]
    return out


__all__ = ["json_extract", "complete_json", "complete_json_many", "json_unclosed", "is_truncated", "JSON_RETRY_SUFFIX", "JSON_WARN_BELOW",
           "MAX_TOKENS_CAP", "JsonBudgetWarning"]
