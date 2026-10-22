"""Exceptions shared by the LLM backends.

`StopError` and its subclasses mean "stop the whole run now, do not retry": a parallel `map`
cancels its pending items when one is raised. Results that already finished stay in the disk
cache, so re-running the same command resumes where it stopped.
"""
from __future__ import annotations


class LLMError(RuntimeError):
    """A call failed after its retries (or failed in a way that retrying cannot fix)."""


class StopError(LLMError):
    """Non-retryable: stop the run (bad key, spend/credit limit, budget)."""


class AuthError(StopError):
    """HTTP 401 or no API key available. Never retried, never falls back to another key."""


class SpendLimitError(StopError):
    """The account hit a usage / spend / credit limit (not a transient rate limit)."""


class BudgetExceeded(StopError):
    """The cost ledger total reached `budget_usd` (checked before every real call)."""


class RefusalError(LLMError):
    """The model answered with stop_reason == "refusal". Not cached."""


class EmptyResponseError(LLMError):
    """The reply has no text and stop_reason == "max_tokens": thinking used the whole output budget.
    The call is billed (ledger row) but not cached and not retried (the same request would fail again)."""


class ThinkingBudgetWarning(UserWarning):
    """max_tokens is small while thinking is on; thinking tokens count toward max_tokens."""


__all__ = ["LLMError", "StopError", "AuthError", "SpendLimitError", "BudgetExceeded", "RefusalError",
           "EmptyResponseError", "ThinkingBudgetWarning"]
