"""JSONL cost ledger: one line per real (non-cached) call, shared across threads and processes.

Ported from talp73/kod/llm/openrouter.py (`_log_yaz`, `harcama_toplam`: append under flock, the
budget gate sums the whole log) as JSONL instead of TSV. Fields per line:
    time, model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_usd, tag
(+ `list_price_usd` for subscription calls through `claude -p`, whose `cost_usd` is 0).
`path=None` keeps the ledger in memory (budget still enforced for this process only).

Optional shared spend log (TSV, the research fleet's `harcama.log` layout, see `kod/ortak.py`
`HarcamaLedger` in the D2 rehearsal): `tsv_path=` (or env SIMENGINE_SPEND_LOG) appends one tab-separated
line per real call, columns `TSV_COLUMNS`, no header, under flock. `agent=` (or env SIMENGINE_AGENT)
fills the agent column. The JSONL ledger stays the budget authority; the TSV is write-only.

Tags. A backend builds each row's tag with `compose_tag(instance_tag, call_tag)`; `tag_prefix=` (or env
SIMENGINE_SPEND_TAG_PREFIX) is then prepended verbatim, so the TSV tag column is
`<prefix><instance tag>:<call tag>`. The JSONL row keeps `tag` without the prefix (as before) and gets a
`tag_prefix` field when the prefix is not empty.
"""
from __future__ import annotations

import json
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Callable

try:
    import fcntl  # POSIX only
except ImportError:  # pragma: no cover
    fcntl = None


SPEND_LOG_ENV = "SIMENGINE_SPEND_LOG"
AGENT_ENV = "SIMENGINE_AGENT"
TAG_PREFIX_ENV = "SIMENGINE_SPEND_TAG_PREFIX"
DEFAULT_AGENT = "simengine"
# column layout of the shared TSV spend log (one line per real call, no header)
TSV_COLUMNS = ("time", "agent", "model", "input_tokens", "output_tokens", "usd", "tag")


def compose_tag(instance_tag: str | None = "", call_tag: str | None = None) -> str:
    """Combine a backend's own tag with a per-call tag (the prefix is added later by the ledger).

    - no call tag (None) or an empty one -> the instance tag;
    - no instance tag -> the call tag unchanged (the old behaviour);
    - otherwise always "<instance tag>:<call tag>". The call tag is never inspected, so a call tag that happens to
      start with "<instance tag>:" is still prefixed (no guessing; simengine.sim passes its raw call tag to
      simengine backends and lets them compose it here).

        compose_tag("", "survey:persona")          -> "survey:persona"
        compose_tag("L1", "survey:persona")        -> "L1:survey:persona"
        compose_tag("survey", "survey:persona")    -> "survey:survey:persona"
        compose_tag("L1", None)                    -> "L1"
    """
    base = "" if instance_tag is None else str(instance_tag)
    call = "" if call_tag is None else str(call_tag)
    if not base:
        return call
    if not call:
        return base
    return f"{base}:{call}"


def _tsv_field(v) -> str:
    return " ".join(str(v).split("\t")).replace("\r", " ").replace("\n", " ")


def tsv_model(model: str) -> str:
    """Model column of the TSV: a bare Claude id gets the provider prefix ("claude-x" -> "anthropic/claude-x");
    already prefixed names ("anthropic/...", "claude-p/...") are kept."""
    m = str(model)
    return f"anthropic/{m}" if m.startswith("claude-") and "/" not in m else m


def tsv_line(row: dict, agent: str, tag_prefix: str = "") -> str:
    """One TSV line (with newline) for a ledger row, columns TSV_COLUMNS, usd with 6 decimals."""
    vals = {"time": row.get("time", ""), "agent": agent, "model": tsv_model(row.get("model", "")),
            "input_tokens": int(row.get("input_tokens") or 0), "output_tokens": int(row.get("output_tokens") or 0),
            "usd": f"{float(row.get('cost_usd') or 0.0):.6f}", "tag": f"{tag_prefix}{row.get('tag', '')}"}
    return "\t".join(_tsv_field(vals[c]) for c in TSV_COLUMNS) + "\n"


def _tsv(tsv_path) -> Path | None:
    """tsv_path argument -> Path (with ~ expanded) or None (None / False / "" = no TSV)."""
    return Path(os.path.expanduser(str(tsv_path))) if tsv_path not in (None, False, "") else None


def _append_locked(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            if fcntl:
                fcntl.flock(f, fcntl.LOCK_UN)


class CostLedger:
    """JSONL ledger (budget source) plus an optional shared TSV spend log.

        CostLedger(".cache/ledger.jsonl", tsv_path="spend.log", agent="team-a", tag_prefix="d2/")

    tsv_path / agent default to the env vars SIMENGINE_SPEND_LOG / SIMENGINE_AGENT (agent falls back to
    "simengine"). tsv_path=False disables the TSV even when the env var is set. A failed TSV write
    warns once and never fails the call (the spend is already in the JSONL ledger).
    tag_prefix (None = env SIMENGINE_SPEND_TAG_PREFIX, else "") is prepended to the TSV tag column and stored as
    the JSONL field `tag_prefix` when not empty; tag_prefix="" ignores the env var.

    `attach(tsv_path=, agent=, tag_prefix=)` fills settings that were not given explicitly here (env/default values
    are replaced) and raises ValueError only when an explicit value differs; backends call it for
    `Claude(ledger=CostLedger(...), spend_log=..., agent=..., tag_prefix=...)`.
    """

    def __init__(self, path: str | os.PathLike | None = None, *, tsv_path: str | os.PathLike | bool | None = None,
                 agent: str | None = None, tag_prefix: str | None = None):
        self.path = Path(path) if path is not None else None
        # which settings came from an argument (not env/default): attach() may only replace the others
        self._explicit = {"tsv_path": tsv_path is not None, "agent": bool(agent), "tag_prefix": tag_prefix is not None}
        if tsv_path is None:
            tsv_path = os.environ.get(SPEND_LOG_ENV, "").strip() or None
        self.tsv_path = _tsv(tsv_path)
        self.agent = agent if agent else (os.environ.get(AGENT_ENV, "").strip() or DEFAULT_AGENT)
        if tag_prefix is None:
            tag_prefix = os.environ.get(TAG_PREFIX_ENV, "").strip()
        self.tag_prefix = str(tag_prefix)
        self._tsv_warned = False
        self._lock = threading.Lock()
        self._mem: list[dict] = []
        self._offset = 0          # bytes of the file already summed
        self._total = 0.0

    def attach(self, *, tsv_path: str | os.PathLike | bool | None = None, agent: str | None = None,
               tag_prefix: str | None = None) -> "CostLedger":
        """Set tsv_path / agent / tag_prefix (None = leave as is). A setting this ledger got explicitly must match
        (same TSV path, agent, prefix), else ValueError naming the conflict; nothing changes on an error.
        Settings from env vars or defaults are replaced. Attached values count as explicit afterwards, so two
        backends sharing one ledger cannot silently disagree."""
        new = {}
        if tsv_path is not None:
            new["tsv_path"] = _tsv(tsv_path)
        if agent:
            new["agent"] = str(agent)
        if tag_prefix is not None:
            new["tag_prefix"] = str(tag_prefix)
        names = {"tsv_path": "spend_log", "agent": "agent", "tag_prefix": "tag_prefix"}
        clash = [k for k, v in new.items() if self._explicit[k] and getattr(self, k) != v]
        if clash:
            have = ", ".join(f"{k}={getattr(self, k)!s:.80}" for k in clash)
            got = ", ".join(f"{names[k]}={new[k]!s:.80}" for k in clash)
            raise ValueError(f"ledger=CostLedger(...) already has {have}; the backend got {got}. Set it once, e.g. "
                             f"Claude(..., ledger=CostLedger(path, tsv_path=..., agent=..., tag_prefix=...)) or "
                             f"Claude(..., ledger=path, spend_log=..., agent=..., tag_prefix=...)")
        for k, v in new.items():
            setattr(self, k, v)
            self._explicit[k] = True
        return self

    # ------------------------------------------------------------ write
    def record(self, model: str, input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0,
               tag: str = "", cache_read_tokens: int = 0, cache_write_tokens: int = 0, **extra) -> dict:
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": model, "input_tokens": int(input_tokens),
               "output_tokens": int(output_tokens), "cache_read_tokens": int(cache_read_tokens),
               "cache_write_tokens": int(cache_write_tokens), "cost_usd": round(float(cost_usd), 8), "tag": tag,
               **extra}
        if self.tag_prefix:
            row["tag_prefix"] = self.tag_prefix
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            if self.path is None:
                self._mem.append(row)
                self._total += row["cost_usd"]
            else:
                _append_locked(self.path, line)
        if self.tsv_path is not None:
            self._write_tsv(row)
        return row

    def _write_tsv(self, row: dict) -> None:
        try:
            _append_locked(self.tsv_path, tsv_line(row, self.agent, self.tag_prefix))
        except OSError as e:
            if not self._tsv_warned:
                self._tsv_warned = True
                warnings.warn(f"spend log {self.tsv_path} not writable ({type(e).__name__}); "
                              "the JSONL ledger still has the row", RuntimeWarning, stacklevel=3)

    # ------------------------------------------------------------ read
    def rows(self) -> list[dict]:
        if self.path is None:
            with self._lock:
                return list(self._mem)
        if not self.path.exists():
            return []
        out = []
        with self.path.open(encoding="utf-8") as f:
            for ln in f:
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
        return out

    def total(self, model_filter: Callable[[str], bool] | None = None) -> float:
        """Sum of cost_usd over the whole ledger (all processes that write to the same file).
        Without a filter the sum is incremental (only new bytes are read)."""
        if model_filter is not None:
            return sum(float(r.get("cost_usd") or 0) for r in self.rows() if model_filter(str(r.get("model", ""))))
        if self.path is None:
            with self._lock:
                return self._total
        with self._lock:
            if not self.path.exists():
                self._offset, self._total = 0, 0.0
                return 0.0
            size = self.path.stat().st_size
            if size < self._offset:            # truncated / replaced: start over
                self._offset, self._total = 0, 0.0
            with self.path.open("rb") as f:
                f.seek(self._offset)
                chunk = f.read()
            end = chunk.rfind(b"\n") + 1      # only complete lines
            for ln in chunk[:end].splitlines():
                try:
                    self._total += float(json.loads(ln).get("cost_usd") or 0)
                except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                    pass
            self._offset += end
            return self._total

    def summary(self) -> dict:
        """Per-model {calls, input_tokens, output_tokens, cost_usd}."""
        out: dict[str, dict] = {}
        for r in self.rows():
            d = out.setdefault(str(r.get("model")), {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            d["calls"] += 1
            d["input_tokens"] += int(r.get("input_tokens") or 0)
            d["output_tokens"] += int(r.get("output_tokens") or 0)
            d["cost_usd"] += float(r.get("cost_usd") or 0)
        return out


__all__ = ["CostLedger", "TSV_COLUMNS", "SPEND_LOG_ENV", "AGENT_ENV", "TAG_PREFIX_ENV", "compose_tag", "tsv_line",
           "tsv_model"]
