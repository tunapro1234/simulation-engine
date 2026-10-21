"""JSONL cost ledger: one line per real (non-cached) call, shared across threads and processes.

Ported from talp73/kod/llm/openrouter.py (`_log_yaz`, `harcama_toplam`: append under flock, the
budget gate sums the whole log) as JSONL instead of TSV. Fields per line:
    time, model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_usd, tag
(+ `list_price_usd` for subscription calls through `claude -p`, whose `cost_usd` is 0).
`path=None` keeps the ledger in memory (budget still enforced for this process only).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable

try:
    import fcntl  # POSIX only
except ImportError:  # pragma: no cover
    fcntl = None


class CostLedger:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._mem: list[dict] = []
        self._offset = 0          # bytes of the file already summed
        self._total = 0.0

    # ------------------------------------------------------------ write
    def record(self, model: str, input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0,
               tag: str = "", cache_read_tokens: int = 0, cache_write_tokens: int = 0, **extra) -> dict:
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": model, "input_tokens": int(input_tokens),
               "output_tokens": int(output_tokens), "cache_read_tokens": int(cache_read_tokens),
               "cache_write_tokens": int(cache_write_tokens), "cost_usd": round(float(cost_usd), 8), "tag": tag,
               **extra}
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            if self.path is None:
                self._mem.append(row)
                self._total += row["cost_usd"]
                return row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                if fcntl:
                    fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(line)
                    f.flush()
                finally:
                    if fcntl:
                        fcntl.flock(f, fcntl.LOCK_UN)
        return row

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


__all__ = ["CostLedger"]
