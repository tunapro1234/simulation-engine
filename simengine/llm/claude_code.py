"""ClaudeCode: Claude through the `claude -p` CLI (subscription; no API bill).

Ported from sahne/kod/sahne/kaynak.py (ClaudeKaynak._cagir), makro/llmabm/kod/llm.py (`sor`) and
dongu/kod/gozlemci.sh. Each call runs

    claude -p --model <m> --output-format json --tools "" --no-session-persistence
           --strict-mcp-config --setting-sources "" --settings '{"language":"English"}'
           [--system-prompt <system>]

with the prompt on stdin, in an empty working directory (so no project CLAUDE.md is loaded), with
a timeout. The language setting is passed explicitly because `claude -p` can inherit the user's
language setting (deney/RAPOR.md, 21 Oct) and it enters the cache key. The JSON envelope's
`result` is the text; `total_cost_usd` is a list-price equivalent, not a bill: it is recorded in
the ledger as `list_price_usd` with cost_usd = 0, so it does not count against budget_usd.
`max_calls` caps new (non-cached) calls per instance, like the research code's call budgets.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from ..types import LLMResponse
from ._base import CachedBackend
from .errors import AuthError, BudgetExceeded, LLMError, SpendLimitError

CLI_LIMIT_MARKERS = ("usage limit", "limit reached", "credit balance", "rate limit", "out of extra usage",
                     "spend limit")
CLI_AUTH_MARKERS = ("invalid api key", "please run /login", "not logged in", "authentication", "401")


class ClaudeCode(CachedBackend):
    """`claude -p` backend implementing the `simengine.types.LLM` protocol.

        llm = ClaudeCode(model="claude-haiku-5-5", cache_dir=".cache", max_calls=500)
        r = llm.complete("You are a careful analyst.", "Propose one change as JSON.")

    Use it for long-lived or low-volume agents (an observer in a research loop, a handful of judge
    calls). For thousands of short calls use `Claude` (API): every `claude -p` call carries a large
    tool/system context (see docs/modules/llm.md).
    `max_tokens` is accepted for protocol compatibility but the CLI does not take it.
    """

    def __init__(self, model: str = "claude-haiku-5-5", *, timeout: float = 300.0, language: str = "English",
                 claude_bin: str | None = None, cwd: str | os.PathLike | None = None, cache_dir=None, ledger=None,
                 max_calls: int | None = None, budget_usd: float | None = None, use_system_prompt: bool = True,
                 extra_args: Sequence[str] = (), max_attempts: int = 3, error_wait: float = 10.0,
                 parallel: int = 4, tag: str = "", runner: Callable[..., Any] | None = None,
                 sleep: Callable[[float], None] = time.sleep, spend_log=None, agent: str | None = None):
        super().__init__(cache_dir=cache_dir, ledger=ledger, budget_usd=budget_usd, tag=tag, parallel=parallel,
                         spend_log=spend_log, agent=agent)
        self.model = model
        self.timeout = timeout
        self.language = language
        self.claude_bin = claude_bin or os.environ.get("CLAUDE_BIN", "claude")
        self._cwd = Path(cwd) if cwd is not None else None
        self.max_calls = max_calls
        self.use_system_prompt = use_system_prompt
        self.extra_args = list(extra_args)
        self.max_attempts = max(1, int(max_attempts))
        self.error_wait = error_wait
        self._run = runner or subprocess.run
        self._sleep = sleep
        self._calls_lock = threading.Lock()
        self._new_calls = 0

    def __repr__(self) -> str:
        return f"ClaudeCode(model={self.model!r}, language={self.language!r}, max_calls={self.max_calls!r})"

    # ------------------------------------------------------------ command
    def build_command(self, system: str = "") -> list[str]:
        cmd = [self.claude_bin, "-p", "--model", self.model, "--output-format", "json", "--tools", "",
               "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "",
               "--settings", json.dumps({"language": self.language})]
        if system and self.use_system_prompt:
            cmd += ["--system-prompt", system]
        return cmd + self.extra_args

    def _stdin(self, system: str, user: str) -> str:
        if system and not self.use_system_prompt:
            return f"{system}\n\n{user}"
        return user

    def workdir(self) -> Path:
        """Empty directory the CLI runs in (no project CLAUDE.md / settings picked up)."""
        if self._cwd is None:
            base = (self.cache.dir / "claude-cwd") if self.cache is not None else Path(tempfile.mkdtemp(prefix="simengine-claude-"))
            self._cwd = base
        self._cwd.mkdir(parents=True, exist_ok=True)
        return self._cwd

    def available(self) -> bool:
        return shutil.which(self.claude_bin) is not None or Path(self.claude_bin).exists()

    @staticmethod
    def parse_envelope(stdout: str) -> dict:
        """Parse the `--output-format json` envelope (the last JSON object line on stdout)."""
        s = (stdout or "").strip()
        if not s:
            raise LLMError("claude -p returned empty stdout")
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            for ln in reversed(s.splitlines()):
                ln = ln.strip()
                if ln.startswith("{"):
                    try:
                        return json.loads(ln)
                    except json.JSONDecodeError:
                        continue
        raise LLMError("claude -p stdout is not a JSON envelope")

    def _reserve(self) -> None:
        with self._calls_lock:
            if self.max_calls is not None and self._new_calls >= self.max_calls:
                raise BudgetExceeded(f"call budget reached ({self.max_calls} new claude -p calls)")
            self._new_calls += 1

    # ------------------------------------------------------------ LLM protocol
    def complete(self, system: str, user: str, *, max_tokens: int = 256, **kw: Any) -> LLMResponse:
        tag = kw.pop("tag", self.tag)
        validate = kw.pop("validate", None)
        cache_if = kw.pop("cache_if", None)
        sample = kw.pop("sample", 0)
        if kw:
            raise TypeError(f"ClaudeCode.complete: unsupported arguments {sorted(kw)}")
        key_params = {"backend": "claude-p", "lang": self.language, "sample": sample,
                      "system_mode": "flag" if self.use_system_prompt else "inline", "extra": self.extra_args}
        key = self._key(self.model, system, user, key_params)
        rec = self._cache_get(key)
        if rec is not None:
            self._bump(cache_hits=1)
            return self._hit(rec, self.model)
        self.check_budget()
        self._reserve()
        cmd, stdin, cwd = self.build_command(system), self._stdin(system, user), self.workdir()
        last = ""
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.time()
            try:
                r = self._run(cmd, input=stdin, capture_output=True, text=True, timeout=self.timeout, cwd=str(cwd))
            except subprocess.TimeoutExpired:
                last = f"timeout after {self.timeout:.0f} s"
                self._bump(retries=1)
                self._sleep(self.error_wait)
                continue
            except FileNotFoundError:
                raise LLMError(f"claude CLI not found: {self.claude_bin!r} (set CLAUDE_BIN)") from None
            sec = time.time() - t0
            try:
                env = self.parse_envelope(r.stdout)
            except LLMError as e:
                err = (r.stderr or "")[-300:].lower()
                if any(k in err for k in CLI_AUTH_MARKERS):
                    raise AuthError("claude -p is not authenticated; not retried") from None
                if any(k in err for k in CLI_LIMIT_MARKERS):
                    raise SpendLimitError("claude -p reports a usage limit; not retried") from None
                last = f"{e} (rc={getattr(r, 'returncode', None)})"
                self._bump(retries=1)
                self._sleep(self.error_wait)
                continue
            text = str(env.get("result") or "")
            if env.get("is_error"):
                low = text.lower()
                if any(k in low for k in CLI_AUTH_MARKERS):
                    raise AuthError("claude -p is not authenticated; not retried")
                if any(k in low for k in CLI_LIMIT_MARKERS):
                    raise SpendLimitError("claude -p reports a usage limit; not retried")
                last = f"is_error: {text[:200]}"
                self._bump(retries=1)
                self._sleep(self.error_wait)
                continue
            u = env.get("usage") or {}
            inp = int(u.get("input_tokens") or 0) + int(u.get("cache_creation_input_tokens") or 0)
            cr = int(u.get("cache_read_input_tokens") or 0)
            out = int(u.get("output_tokens") or 0)
            lp = float(env.get("total_cost_usd") or 0.0)
            self.ledger.record(f"claude-p/{self.model}", inp, out, 0.0, tag, cache_read_tokens=cr,
                               cache_write_tokens=int(u.get("cache_creation_input_tokens") or 0), list_price_usd=lp,
                               seconds=round(sec, 2))
            self._bump(calls=1)
            if validate is not None:
                try:
                    validate(text)
                except (ValueError, KeyError, TypeError) as e:
                    last = f"validation: {e}"[:200]
                    self._bump(retries=1)
                    continue
            raw = {"usage": u, "list_price_usd": lp, "seconds": round(sec, 2), "session_id": env.get("session_id"),
                   "attempt": attempt}
            if self._should_cache(cache_if, text):
                self._cache_put(key, {"text": text, "model": self.model, "input_tokens": inp, "output_tokens": out,
                                      "cache_read_tokens": cr, "cost_usd": 0.0, "raw": raw, "system": system,
                                      "user": user, "params": key_params})
            else:
                raw["not_cached"] = True
            return LLMResponse(text=text, model=self.model, input_tokens=inp, output_tokens=out, cache_read_tokens=cr,
                               cost_usd=0.0, cached=False, raw=raw)
        self._bump(errors=1)
        raise LLMError(f"claude -p failed after {self.max_attempts} attempts: {last}")


__all__ = ["ClaudeCode"]
