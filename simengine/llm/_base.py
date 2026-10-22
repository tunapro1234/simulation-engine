"""Shared machinery of the real backends: disk cache, cost ledger, budget guard, map, stats."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Sequence

from ..types import LLMResponse
from .cache import DiskCache, cache_key
from .errors import BudgetExceeded
from .jsonx import complete_json
from .ledger import CostLedger
from .parallel import parallel_map


class CachedBackend:
    """Base class. Subclasses implement `complete`. Not used by FakeLLM (kept dependency-free)."""

    model: str = "-"

    def __init__(self, *, cache_dir: str | os.PathLike | None = None, ledger: CostLedger | str | os.PathLike | None = None,
                 budget_usd: float | None = None, tag: str = "", parallel: int = 4,
                 spend_log: str | os.PathLike | None = None, agent: str | None = None):
        self.cache = DiskCache(cache_dir) if cache_dir is not None else None
        if isinstance(ledger, CostLedger):
            if spend_log is not None or agent is not None:
                raise ValueError("spend_log/agent configure the default ledger; with ledger=CostLedger(...) pass "
                                 "tsv_path=/agent= to that CostLedger instead")
            self.ledger = ledger
        elif ledger is not None:
            self.ledger = CostLedger(ledger, tsv_path=spend_log, agent=agent)
        elif cache_dir is not None:
            self.ledger = CostLedger(Path(cache_dir) / "ledger.jsonl", tsv_path=spend_log, agent=agent)
        else:
            self.ledger = CostLedger(None, tsv_path=spend_log, agent=agent)
        self.budget_usd = budget_usd
        self.tag = tag
        self.parallel = parallel
        self._stats_lock = threading.Lock()
        self._stats = {"calls": 0, "cache_hits": 0, "retries": 0, "errors": 0, "cost_usd": 0.0}

    # ------------------------------------------------------------ helpers
    def _bump(self, **kw: float) -> None:
        with self._stats_lock:
            for k, v in kw.items():
                self._stats[k] = self._stats.get(k, 0) + v

    @property
    def stats(self) -> dict:
        """This instance's counters: real calls, cache hits, retries, errors, USD spent."""
        with self._stats_lock:
            d = dict(self._stats)
        d["cost_usd"] = round(d["cost_usd"], 6)
        d["ledger_total_usd"] = round(self.ledger.total(), 6)
        return d

    def check_budget(self) -> None:
        """Raise BudgetExceeded when the ledger total is already >= budget_usd."""
        if self.budget_usd is None:
            return
        tot = self.ledger.total()
        if tot >= self.budget_usd:
            raise BudgetExceeded(f"cost ledger total ${tot:.4f} >= budget ${self.budget_usd:.4f}; no new call made")

    def _key(self, key_model: str, system: str, user: str, params: dict) -> str:
        return cache_key(key_model, system, user, params)

    def _cache_get(self, key: str) -> dict | None:
        return self.cache.get(key, self.model) if self.cache is not None else None

    def _cache_put(self, key: str, record: dict) -> None:
        if self.cache is not None:
            self.cache.put(key, record, self.model)

    @staticmethod
    def _should_cache(cache_if, text: str) -> bool:
        """cache_if(text) -> bool decides whether a successful reply is written to the disk cache
        (None = always). An exception inside cache_if counts as False (do not cache)."""
        if cache_if is None:
            return True
        try:
            return bool(cache_if(text))
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _hit(rec: dict, model: str) -> LLMResponse:
        return LLMResponse(text=rec.get("text", ""), model=rec.get("model", model),
                           input_tokens=int(rec.get("input_tokens") or 0), output_tokens=int(rec.get("output_tokens") or 0),
                           cache_read_tokens=int(rec.get("cache_read_tokens") or 0), cost_usd=0.0, cached=True,
                           raw={**(rec.get("raw") or {}), "original_cost_usd": rec.get("cost_usd", 0.0),
                                "saved_at": rec.get("saved_at")})

    # ------------------------------------------------------------ LLM protocol
    def complete(self, system: str, user: str, *, max_tokens: int = 256, **kw: Any) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError

    def map(self, prompts: Sequence[tuple[str, str]], *, max_tokens: int = 256, parallel: int | None = None,
            return_exceptions: bool = False, **kw: Any) -> list[LLMResponse]:
        """Run many (system, user) prompts with a thread pool, results in input order.
        A StopError (budget, 401, spend limit) cancels the rest and is raised; finished answers are
        cached, so the same call resumes. return_exceptions=True puts other errors (LLMError,
        RefusalError, EmptyResponseError, ...) in their slot. Keywords such as tag=, validate=,
        cache_if=, thinking= are passed to every complete() call."""
        p = self.parallel if parallel is None else parallel
        return parallel_map(lambda su: self.complete(su[0], su[1], max_tokens=max_tokens, **kw), list(prompts),
                            parallel=p, return_exceptions=return_exceptions)

    def complete_json(self, system: str, user: str, *, attempts: int = 2, **kw: Any) -> Any:
        """complete() + json_extract(); re-asks once with a 'JSON only' suffix on a parse failure."""
        return complete_json(self, system, user, attempts=attempts, **kw)
