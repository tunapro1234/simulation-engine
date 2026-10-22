"""Read the shared TSV spend log back: calls / tokens / USD per tag, agent or model.

    from simengine.llm import spend_summary
    s = spend_summary("spend.log", prefix="recete-G2/")   # path=None -> env SIMENGINE_SPEND_LOG
    print(s)                     # one line per tag + total
    s.total["usd"], s.groups["recete-G2/L1:survey:persona"]["calls"]
    spend_summary("spend.log", by="agent")                # or by="model", by=None (total only), by=callable(row)
    spend_summary(llm, prefix="recete-G2/")                # a backend or CostLedger: reads its spend_log= file

The log is the one `CostLedger(tsv_path=...)` / `Claude(spend_log=...)` appends to (columns `TSV_COLUMNS`, no
header; `usd` is the 6th column). The recipe rehearsal G2 (#7) summed that column with awk for its tag prefix.
Lines with fewer than the 7 columns or whose numbers do not parse are counted in `skipped`, never raised.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .ledger import SPEND_LOG_ENV, TSV_COLUMNS

_NUM = ("calls", "input_tokens", "output_tokens", "usd")


def _zero() -> dict:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}


@dataclass
class SpendSummary:
    """Result of `spend_summary`. groups: {key: {calls, input_tokens, output_tokens, usd}} sorted by usd (desc);
    total: the same over every matching line; skipped: malformed lines."""

    path: str
    prefix: str | None
    by: str | None
    groups: dict[str, dict] = field(default_factory=dict)
    total: dict = field(default_factory=_zero)
    skipped: int = 0

    @property
    def usd(self) -> float:
        return self.total["usd"]

    def to_dict(self) -> dict:
        return {"path": self.path, "prefix": self.prefix, "by": self.by, "groups": self.groups,
                "total": self.total, "skipped": self.skipped}

    def __str__(self) -> str:
        head = f"{self.by or 'all'}\tcalls\tin_tok\tout_tok\tusd"
        rows = [f"{k}\t{d['calls']}\t{d['input_tokens']}\t{d['output_tokens']}\t{d['usd']:.6f}"
                for k, d in self.groups.items()]
        t = self.total
        rows.append(f"TOTAL\t{t['calls']}\t{t['input_tokens']}\t{t['output_tokens']}\t{t['usd']:.6f}")
        if self.skipped:
            rows.append(f"(skipped {self.skipped} malformed lines)")
        return "\n".join([head, *rows])


def _log_of(obj) -> str | os.PathLike | None:
    """A backend (`.ledger`) or a CostLedger (`.tsv_path`) -> its TSV path; anything else unchanged."""
    led = getattr(obj, "ledger", obj)
    if hasattr(led, "tsv_path"):
        if led.tsv_path is None:
            raise ValueError("this backend / ledger writes no TSV spend log (pass spend_log= or set "
                             f"{SPEND_LOG_ENV}); its own spend is llm.spent_usd")
        return led.tsv_path
    return obj


def spend_summary(log_path=None, prefix: str | None = None,
                  by: str | Callable[[dict], str] | None = "tag") -> SpendSummary:
    """Sum the shared TSV spend log for tags starting with `prefix` (None = every line).

    log_path: a path, a backend built with `spend_log=` (e.g. `Claude`) or a `CostLedger` (its TSV file; the whole
    shared file, other agents included, so pass `prefix=`), or None -> env SIMENGINE_SPEND_LOG. A missing file gives
    an empty summary (all zero).
    by: "tag" (default), "agent", "model", any other TSV column name, a callable(row dict) -> key, or None for the
    total only.
    """
    log_path = _log_of(log_path)
    if log_path is None:
        log_path = os.environ.get(SPEND_LOG_ENV, "").strip() or None
    if log_path is None:
        raise ValueError(f"spend_summary needs log_path= or env {SPEND_LOG_ENV}")
    path = Path(os.path.expanduser(str(log_path)))
    if isinstance(by, str) and by not in TSV_COLUMNS:
        raise ValueError(f"by={by!r}: use one of {TSV_COLUMNS}, a callable or None")
    label = by if isinstance(by, str) or by is None else getattr(by, "__name__", "key")
    out = SpendSummary(path=str(path), prefix=prefix, by=label)
    groups: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if not ln.strip():
                    continue
                parts = ln.split("\t", len(TSV_COLUMNS) - 1)     # a tab inside the tag stays in it
                if len(parts) != len(TSV_COLUMNS):
                    out.skipped += 1
                    continue
                row = dict(zip(TSV_COLUMNS, parts))
                if prefix is not None and not row["tag"].startswith(prefix):
                    continue
                try:
                    vals = (1, int(row["input_tokens"]), int(row["output_tokens"]), float(row["usd"]))
                except ValueError:
                    out.skipped += 1
                    continue
                targets = [out.total]
                if by is not None:
                    key = row[by] if isinstance(by, str) else str(by(row))
                    targets.append(groups.setdefault(key, _zero()))
                for d in targets:
                    for k, v in zip(_NUM, vals):
                        d[k] += v
    out.total["usd"] = round(out.total["usd"], 6)
    for d in groups.values():
        d["usd"] = round(d["usd"], 6)
    out.groups = dict(sorted(groups.items(), key=lambda kv: (-kv[1]["usd"], kv[0])))
    return out


__all__ = ["spend_summary", "SpendSummary"]
