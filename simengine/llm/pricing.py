"""Per-model list prices and token -> USD cost math.

Prices are $ per million tokens (input, output) for the Claude API, first-party, prompts up to
the model's long-context threshold. Ported from talp73/kod/llm/claude_api.py (FIYAT, _usd) and
extended with Fable 5.1.

Cost formula (same as the research code):
    usd = (input + 1.25 * cache_write + 0.1 * cache_read) * p_in + output * p_out
Cache reads are billed at 0.1x input. For Opus 5.5 the list price of a cache read is $0.20/MTok,
i.e. 0.05x; the research code deliberately kept 0.1x (conservative), and so does this module.
"""
from __future__ import annotations

from dataclasses import dataclass

CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25   # 5-minute ephemeral cache write


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    long_context_threshold: int | None = None     # prompt tokens above which the long price applies
    long_input_per_mtok: float | None = None
    long_output_per_mtok: float | None = None


PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(0.10, 0.50, 100_000, 0.50, 2.50),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-opus-5-5": Price(4.00, 20.00),
    "claude-fable-5-1": Price(10.00, 50.00),
}


def normalize_model(model: str) -> str:
    """'anthropic/claude-haiku-5-5' -> 'claude-haiku-5-5' (ledger lines of the research code use the prefix)."""
    return model.split("/", 1)[1] if model.startswith("anthropic/") else model


def price_for(model: str, prices: dict[str, Price] | None = None) -> Price:
    """Exact match first, then the longest table key that prefixes the model id. KeyError if unknown:
    an unpriced model would silently disable the budget guard, so it is an error, not $0."""
    table = PRICES if prices is None else prices
    m = normalize_model(model)
    if m in table:
        return table[m]
    keys = sorted((k for k in table if m.startswith(k)), key=len, reverse=True)
    if keys:
        return table[keys[0]]
    raise KeyError(f"no price for model {model!r}; pass prices={{...}} (known: {sorted(table)})")


def cost_usd(model: str, input_tokens: int = 0, output_tokens: int = 0, cache_read_tokens: int = 0,
             cache_write_tokens: int = 0, prices: dict[str, Price] | None = None) -> float:
    """USD cost of one call. `input_tokens` is the uncached input (API usage.input_tokens)."""
    p = price_for(model, prices)
    prompt = input_tokens + cache_read_tokens + cache_write_tokens
    pin, pout = p.input_per_mtok, p.output_per_mtok
    if p.long_context_threshold is not None and prompt > p.long_context_threshold:
        pin = p.long_input_per_mtok if p.long_input_per_mtok is not None else pin
        pout = p.long_output_per_mtok if p.long_output_per_mtok is not None else pout
    eff_in = input_tokens + CACHE_WRITE_MULT * cache_write_tokens + CACHE_READ_MULT * cache_read_tokens
    return (eff_in * pin + output_tokens * pout) / 1e6


def usage_cost(model: str, usage: dict, prices: dict[str, Price] | None = None) -> float:
    """Cost from an Anthropic `usage` dict (input_tokens, output_tokens, cache_*_input_tokens)."""
    g = lambda k: int(usage.get(k) or 0)  # noqa: E731
    return cost_usd(model, g("input_tokens"), g("output_tokens"), g("cache_read_input_tokens"),
                    g("cache_creation_input_tokens"), prices)


__all__ = ["Price", "PRICES", "CACHE_READ_MULT", "CACHE_WRITE_MULT", "normalize_model", "price_for",
           "cost_usd", "usage_cost"]
