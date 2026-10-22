"""LLM backends for simengine.

- `Claude`      Claude API via the official `anthropic` SDK (lazy import): effort, cache_control,
                disk cache, JSONL cost ledger, budget guard, parallel map, 429/overload backoff,
                immediate stop on 401 and on spend/credit-limit errors.
- `ClaudeCode`  `claude -p` subprocess (subscription, no API cost) for long-lived / low-volume agents.
- `FakeLLM`     deterministic, offline (tests and dry runs).
All three implement `simengine.types.LLM` (complete + map).
Also: `looks_like_refusal(text)` (soft-refusal detector), a shared TSV spend log on `CostLedger`
(`tsv_path=` / SIMENGINE_SPEND_LOG) with `spend_summary(log_path, prefix=)` to sum it per tag, `EmptyResponseError`
for thinking-ate-the-budget replies, `complete_json` retrying a reply cut at max_tokens with a larger budget.
"""
from .cache import DiskCache, cache_key
from .claude import Claude, classify_error, load_api_key
from .claude_code import ClaudeCode
from .errors import (AuthError, BudgetExceeded, EmptyResponseError, LLMError, RefusalError, SpendLimitError, StopError,
                     ThinkingBudgetWarning)
from .fake import FakeLLM
from .jsonx import JsonBudgetWarning, complete_json, is_truncated, json_extract
from .ledger import AGENT_ENV, SPEND_LOG_ENV, TAG_PREFIX_ENV, TSV_COLUMNS, CostLedger, compose_tag
from .parallel import parallel_map
from .refusal import looks_like_refusal
from .spend import SpendSummary, spend_summary
from .pricing import PRICES, Price, cost_usd, price_for, usage_cost

__all__ = [
    "Claude", "ClaudeCode", "FakeLLM",
    "DiskCache", "cache_key", "CostLedger", "TSV_COLUMNS", "SPEND_LOG_ENV", "AGENT_ENV", "TAG_PREFIX_ENV",
    "compose_tag", "looks_like_refusal", "spend_summary", "SpendSummary",
    "PRICES", "Price", "cost_usd", "usage_cost", "price_for",
    "json_extract", "complete_json", "is_truncated", "JsonBudgetWarning", "parallel_map",
    "load_api_key", "classify_error",
    "LLMError", "StopError", "AuthError", "SpendLimitError", "BudgetExceeded", "RefusalError",
    "EmptyResponseError", "ThinkingBudgetWarning",
]
