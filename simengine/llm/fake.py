"""Deterministic offline LLM for tests and dry runs (no network, no cost)."""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Sequence

from ..types import LLMResponse


class FakeLLM:
    """Returns `responder(system, user)` if given, otherwise a stable pseudo-random reply derived
    from a hash of the prompt. `calls` records every (system, user) pair."""

    def __init__(self, responder: Callable[[str, str], str] | None = None, model: str = "fake"):
        self.model = model
        self.responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, max_tokens: int = 256, **kw: Any) -> LLMResponse:
        self.calls.append((system, user))
        if self.responder is not None:
            text = self.responder(system, user)
        else:
            text = hashlib.sha256((system + "\x00" + user).encode()).hexdigest()[:16]
        return LLMResponse(text=text, model=self.model, input_tokens=len((system + user).split()),
                           output_tokens=len(text.split()))

    def map(self, prompts: Sequence[tuple[str, str]], *, max_tokens: int = 256, parallel: int = 4,
            **kw: Any) -> list[LLMResponse]:
        return [self.complete(s, u, max_tokens=max_tokens, **kw) for s, u in prompts]
