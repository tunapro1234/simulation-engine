"""Claude through the official `anthropic` SDK (imported lazily; `pip install simengine[llm]`).

Ported from talp73/kod/llm/claude_api.py (httpx there; the SDK here) and openrouter.py:
- `output_config.effort` (default "low"), small `max_tokens`, no temperature (Haiku/Sonnet 5.5
  return HTTP 400 for a non-default temperature, talp73/PROTOKOL.md v1.4);
- the system prompt is sent as one text block with `cache_control: ephemeral` (not part of the
  local cache key);
- local disk cache keyed by sha256(model + system + user + params); `sample=k` gives independent
  draws of the same prompt (the research code's `ornek`);
- JSONL cost ledger with the per-model price table; budget guard before every real call;
- rate limit (429) / overloaded (529) / 5xx / network errors: linear backoff, honouring retry-after;
- HTTP 401 -> AuthError immediately; spend / usage / credit-limit errors -> SpendLimitError
  immediately (no retry, no fallback key);
- `validate(text)` hook: ValueError/KeyError/TypeError -> the call is retried, nothing cached.
The API key comes from `api_key=`, else ANTHROPIC_API_KEY, else the file named by SIMENGINE_KEY_FILE.
Its value is never logged, cached, put in an exception or in repr().
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

from ..types import LLMResponse
from ._base import CachedBackend
from .errors import AuthError, LLMError, RefusalError, SpendLimitError
from .pricing import PRICES, Price, price_for, usage_cost

LIMIT_MARKERS = ("usage limit", "spend limit", "spending limit", "credit balance", "billing")
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def load_api_key(api_key: str | None = None) -> str:
    """api_key arg > ANTHROPIC_API_KEY > contents of the file named by SIMENGINE_KEY_FILE.
    Raises AuthError (naming only the variables, never a value) when none is available."""
    if api_key:
        return api_key.strip()
    env = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env:
        return env
    kf = os.environ.get("SIMENGINE_KEY_FILE", "").strip()
    if kf:
        try:
            with open(os.path.expanduser(kf), encoding="utf-8") as f:
                val = f.read().strip()
        except OSError as e:
            raise AuthError(f"SIMENGINE_KEY_FILE is set but cannot be read ({type(e).__name__})") from None
        if val:
            return val
        raise AuthError("SIMENGINE_KEY_FILE points to an empty file")
    raise AuthError("no API key: set ANTHROPIC_API_KEY or SIMENGINE_KEY_FILE (cached answers still work)")


def _status(exc: BaseException) -> int | None:
    s = getattr(exc, "status_code", None)
    if s is None:
        resp = getattr(exc, "response", None)
        s = getattr(resp, "status_code", None)
    try:
        return int(s) if s is not None else None
    except (TypeError, ValueError):
        return None


def _message(exc: BaseException) -> str:
    parts = [str(getattr(exc, "message", "") or ""), str(exc)]
    body = getattr(exc, "body", None)
    if body:
        parts.append(str(body))
    return " ".join(parts).lower()


def classify_error(exc: BaseException) -> str:
    """'auth' | 'limit' | 'retry' | 'fatal'. Ported from claude_api._yetki/_sinir_mi:
    401 -> auth; 400/402/403/429 whose message names a usage/spend/credit/billing limit -> limit;
    429 rate limit, 529 overloaded, 5xx, 408/409, connection/timeout errors -> retry; else fatal."""
    st = _status(exc)
    msg = _message(exc)
    if st == 401:
        return "auth"
    if st in (400, 402, 403, 429) and any(k in msg for k in LIMIT_MARKERS):
        return "limit"
    if st in RETRY_STATUS or "overloaded" in msg:
        return "retry"
    name = type(exc).__name__
    if st is None and ("Connection" in name or "Timeout" in name):
        return "retry"
    return "fatal"


def _retry_after(exc: BaseException) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    try:
        v = headers.get("retry-after")
        return float(v) if v is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def _to_dict(msg: Any) -> dict:
    if isinstance(msg, dict):
        return msg
    for attr in ("model_dump", "to_dict"):
        f = getattr(msg, attr, None)
        if callable(f):
            try:
                return f(mode="json") if attr == "model_dump" else f()
            except TypeError:
                return f()
    return dict(vars(msg))


def _text(d: dict) -> str:
    return "".join(b.get("text", "") for b in (d.get("content") or []) if isinstance(b, dict) and b.get("type") == "text")


class Claude(CachedBackend):
    """Claude API backend implementing the `simengine.types.LLM` protocol.

        llm = Claude(model="claude-haiku-5-5", effort="low", cache_dir=".cache", budget_usd=5)
        r = llm.complete("You are a survey respondent.", "Answer A or B.", max_tokens=200)
        rs = llm.map([(sys, u) for u in prompts], parallel=4)

    Parameters
    ----------
    model, effort : model id and output_config.effort (None = model default).
    max_tokens : default for complete()/map(). Thinking tokens count toward it; the research runs
        used 2000 with effort low (talp73/PROTOKOL.md v1.5).
    thinking : None (omit: adaptive on all 5.x models), "adaptive", "off" (Haiku -> disabled,
        Sonnet 5.5 -> between_tools; Opus 5.5 / Fable cannot disable -> ValueError) or a raw dict.
    cache_dir : local disk cache + default ledger at <cache_dir>/ledger.jsonl. None = no cache.
    ledger : CostLedger or path; overrides the default ledger location.
    budget_usd : BudgetExceeded is raised before a call when the ledger total >= this.
    system_cache : put cache_control on the system block (default True).
    fallbacks : None (default) or a server-side fallback spec such as "default"; when set, the beta
        messages endpoint is used with the server-side-fallback beta. Off by default because a
        fallback changes which model answers, which breaks a measured comparison.
    max_attempts, rate_wait, error_wait : retry policy (linear backoff: wait * attempt).
    client : an object with `.messages.create(**params)` (tests inject a fake; default SDK client).
    prices : override the price table (needed for models not in PRICES).
    """

    def __init__(self, model: str = "claude-haiku-5-5", *, effort: str | None = "low", max_tokens: int = 256,
                 thinking: str | dict | None = None, cache_dir=None, ledger=None, budget_usd: float | None = None,
                 system_cache: bool = True, fallbacks: Any = None, max_attempts: int = 4, rate_wait: float = 15.0,
                 error_wait: float = 3.0, timeout: float = 180.0, parallel: int = 4, tag: str = "",
                 api_key: str | None = None, client: Any = None, prices: dict[str, Price] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        super().__init__(cache_dir=cache_dir, ledger=ledger, budget_usd=budget_usd, tag=tag, parallel=parallel)
        if effort is not None and effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS} or None")
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.system_cache = system_cache
        self.fallbacks = fallbacks
        self.max_attempts = max(1, int(max_attempts))
        self.rate_wait = rate_wait
        self.error_wait = error_wait
        self.timeout = timeout
        self.prices = prices if prices is not None else PRICES
        price_for(model, self.prices)            # fail early on an unpriced model (budget would be blind)
        self._api_key_arg = api_key
        self._client = client
        self._client_lock = threading.Lock()
        self._sleep = sleep

    def __repr__(self) -> str:   # never shows the key
        return (f"Claude(model={self.model!r}, effort={self.effort!r}, cache={'on' if self.cache else 'off'}, "
                f"budget_usd={self.budget_usd!r})")

    # ------------------------------------------------------------ client
    def _get_client(self):
        with self._client_lock:
            if self._client is None:
                key = load_api_key(self._api_key_arg)
                try:
                    import anthropic  # lazy: optional dependency
                except ImportError as e:  # pragma: no cover
                    raise LLMError("the 'anthropic' package is required for Claude: pip install simengine[llm]") from e
                # SDK retries off: this class owns retry policy (spend-limit 429s must not be retried)
                self._client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=self.timeout)
            return self._client

    # ------------------------------------------------------------ request
    def _thinking_param(self, thinking):
        if thinking is None:
            return None
        if isinstance(thinking, dict):
            return thinking
        if thinking == "adaptive":
            return {"type": "adaptive"}
        if thinking == "off":
            m = self.model
            if "haiku" in m:
                return {"type": "disabled"}
            if m.startswith("claude-sonnet-5-5"):
                return {"type": "between_tools"}
            raise ValueError(f"thinking cannot be turned off on {m}; lower effort instead")
        raise ValueError("thinking must be None, 'adaptive', 'off' or a dict")

    def build_request(self, system: str, user: str, max_tokens: int | None = None, **kw: Any) -> tuple[dict, dict]:
        """Return (api_params, key_params). api_params go to messages.create; key_params (no
        cache_control, plain system text) feed the local cache key."""
        kw = dict(kw)
        sample = kw.pop("sample", 0)
        effort = kw.pop("effort", self.effort)
        thinking = self._thinking_param(kw.pop("thinking", self.thinking))
        system_cache = kw.pop("system_cache", self.system_cache)
        oc = dict(kw.pop("output_config", None) or {})
        if effort is not None:
            oc.setdefault("effort", effort)
        mt = int(max_tokens if max_tokens is not None else self.max_tokens)
        params: dict[str, Any] = {"model": self.model, "max_tokens": mt,
                                  "messages": [{"role": "user", "content": user}]}
        if oc:
            params["output_config"] = oc
        if thinking is not None:
            params["thinking"] = thinking
        if system:
            params["system"] = ([{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                                if system_cache else system)
        params.update(kw)                        # any other API parameter (stop_sequences, metadata, ...)
        key_params = {k: v for k, v in params.items() if k not in ("model", "messages", "system")}
        key_params["sample"] = sample
        if self.fallbacks is not None:
            key_params["fallbacks"] = self.fallbacks
        return params, key_params

    def _create(self, params: dict):
        c = self._get_client()
        if self.fallbacks is None:
            return c.messages.create(**params)
        return c.beta.messages.create(**params, betas=["server-side-fallback-2026-07-01"], fallbacks=self.fallbacks)

    # ------------------------------------------------------------ LLM protocol
    def complete(self, system: str, user: str, *, max_tokens: int | None = None, **kw: Any) -> LLMResponse:
        """One call (or a cache hit). Extra keywords: tag, sample, validate, effort, thinking,
        system_cache, output_config, and any other messages.create parameter."""
        tag = kw.pop("tag", self.tag)
        validate = kw.pop("validate", None)
        params, key_params = self.build_request(system, user, max_tokens, **kw)
        key = self._key(self.model, system, user, key_params)
        rec = self._cache_get(key)
        if rec is not None:
            self._bump(cache_hits=1)
            return self._hit(rec, self.model)
        last = ""
        for attempt in range(1, self.max_attempts + 1):
            self.check_budget()
            try:
                msg = self._create(params)
            except (AuthError, LLMError):
                raise
            except Exception as e:  # noqa: BLE001 - SDK errors classified by status/message
                kind = classify_error(e)
                st = _status(e)
                if kind == "auth":
                    self._bump(errors=1)
                    raise AuthError(f"HTTP 401 from the Claude API ({type(e).__name__}); not retried") from None
                if kind == "limit":
                    self._bump(errors=1)
                    raise SpendLimitError(f"usage/spend/credit limit (HTTP {st}); not retried") from None
                if kind == "fatal" or attempt == self.max_attempts:
                    self._bump(errors=1)
                    raise LLMError(f"Claude call failed (HTTP {st}, {type(e).__name__}): {str(e)[:200]}") from e
                ra = _retry_after(e)
                base = self.rate_wait if st in (429, 529) or "overloaded" in _message(e) else self.error_wait
                self._bump(retries=1)
                last = f"HTTP {st} {type(e).__name__}"
                self._sleep(ra if ra is not None else base * attempt)
                continue
            d = _to_dict(msg)
            u = d.get("usage") or {}
            served = d.get("model") or self.model
            try:
                cost = usage_cost(served, u, self.prices)
            except KeyError:
                cost = usage_cost(self.model, u, self.prices)
            inp = int(u.get("input_tokens") or 0) + int(u.get("cache_creation_input_tokens") or 0)
            out = int(u.get("output_tokens") or 0)
            cr = int(u.get("cache_read_input_tokens") or 0)
            self.ledger.record(f"anthropic/{self.model}", inp, out, cost, tag, cache_read_tokens=cr,
                               cache_write_tokens=int(u.get("cache_creation_input_tokens") or 0),
                               served_model=served, attempt=attempt)
            self._bump(calls=1, cost_usd=cost)
            raw = {"usage": u, "stop_reason": d.get("stop_reason"), "response_model": served, "id": d.get("id"),
                   "attempt": attempt}
            if d.get("stop_reason") == "refusal":
                self._bump(errors=1)
                raise RefusalError(f"refusal (stop_details={d.get('stop_details')})")
            text = _text(d)
            if validate is not None:
                try:
                    validate(text)
                except (ValueError, KeyError, TypeError) as e:
                    last = f"validation: {e}"[:200]
                    self._bump(retries=1)
                    continue
            self._cache_put(key, {"text": text, "model": self.model, "input_tokens": inp, "output_tokens": out,
                                  "cache_read_tokens": cr, "cost_usd": cost, "raw": raw, "system": system,
                                  "user": user, "params": key_params})
            return LLMResponse(text=text, model=self.model, input_tokens=inp, output_tokens=out, cache_read_tokens=cr,
                               cost_usd=cost, cached=False, raw=raw)
        self._bump(errors=1)
        raise LLMError(f"Claude call failed after {self.max_attempts} attempts: {last}")


__all__ = ["Claude", "load_api_key", "classify_error", "LIMIT_MARKERS"]
