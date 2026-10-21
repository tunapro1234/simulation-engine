"""Robust parsers for LLM output: JSON extraction and probability normalisation.

Ported from sahne/kaynak.py (`json_coz`: fenced blocks, first array/object, salvage of complete objects from a
truncated array), talp73/kod/llm/openrouter.py (`parse_json`: trailing commas), makro/llmabm/kod/c1/k2.py
(`dag_ayikla`: option-number -> percentage dict) and talp73/kod/llm/istem.py (`dogrulayici`: per-key 0-100
probabilities, single-key wrapper unwrapping).
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Sequence

import numpy as np

from ._stats import to_number

_TRAILING = re.compile(r",\s*([}\]])")
_OBJ = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")


def _strip_trailing_commas(s: str) -> str:
    return _TRAILING.sub(r"\1", s)


def parse_json(text: Any) -> Any:
    """Extract the first JSON value (object or array) from LLM text.

    Handles ```json fences, prose before/after the JSON, trailing commas, and a truncated array (complete
    objects are salvaged and returned as a list). Returns None when nothing parseable is found. Dicts and lists
    are returned unchanged."""
    if text is None:
        return None
    if isinstance(text, (dict, list)):
        return text
    s = str(text)
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m and ("{" in m.group(1) or "[" in m.group(1)):
        s = m.group(1)
    starts = [i for i, c in enumerate(s) if c in "[{"][:50]
    if not starts:
        return None
    dec = json.JSONDecoder()
    for i in starts:
        body = s[i:]
        for cand in (body, _strip_trailing_commas(body)):
            try:
                v = dec.raw_decode(cand)[0]
            except json.JSONDecodeError:
                continue
            if isinstance(v, (dict, list)):
                return v
        if s[i] == "[" and s[i + 1:].lstrip().startswith("{"):
            # an array of objects that does not parse (cut off at max_tokens): keep the complete objects
            out = []
            for mm in _OBJ.finditer(s[i:]):
                try:
                    out.append(json.loads(_strip_trailing_commas(mm.group(0))))
                except json.JSONDecodeError:
                    pass
            if out:
                return out
    return None


def unwrap(d: Any, keys: Sequence[str] | None = None) -> Any:
    """{"probabilities": {...}} -> {...}: unwrap a single-key wrapper dict (one level, repeated) unless that key
    is itself one of the expected `keys`."""
    want = {str(k).strip().lower() for k in (keys or [])}
    while isinstance(d, dict) and len(d) == 1:
        k, v = next(iter(d.items()))
        if str(k).strip().lower() in want or not isinstance(v, (dict, list)):
            break
        d = v
    return d


def _num(v: Any) -> float | None:
    if isinstance(v, str):
        v = v.strip().rstrip("%").strip()
    return to_number(v)


def normalize_probs(values: Any, k: int | None = None, labels: Sequence[str] | None = None) -> np.ndarray | None:
    """Turn an LLM answer into a probability vector summing to 1, or None when invalid.

    `values`: list of numbers/strings ("40%", "0.4"), or a dict keyed by option number ("1".."k", 1-based, as
    in the measured distribution prompt), option letter ("a"/"A"), "option 1", or the option label text.
    Invalid: wrong length, a missing option, a negative value, a non-number, or an all-zero vector.
    Percentages and fractions are both accepted (only the ratios matter)."""
    values = unwrap(values, labels)
    if isinstance(values, dict):
        if k is None:
            k = len(labels) if labels else len(values)
        low = {re.sub(r"\s+", " ", str(key)).strip().lower(): v for key, v in values.items()}
        vec = []
        for j in range(k):
            cands = [str(j + 1), f"option {j + 1}", f"o{j + 1}", chr(97 + j) if j < 26 else None]
            if labels and j < len(labels):
                cands.append(re.sub(r"\s+", " ", str(labels[j])).strip().lower())
            hit = next((low[c] for c in cands if c is not None and c in low), None)
            if hit is None:
                return None
            vec.append(hit)
        values = vec
    if not isinstance(values, (list, tuple, np.ndarray)):
        return None
    xs = [_num(v) for v in values]
    if any(x is None for x in xs) or not xs:
        return None
    v = np.asarray(xs, dtype=float)
    if k is not None and len(v) != k:
        return None
    if (v < 0).any() or not math.isfinite(v.sum()) or v.sum() <= 0:
        return None
    return v / v.sum()


def parse_distribution(text: Any, k: int, labels: Sequence[str] | None = None) -> np.ndarray | None:
    """LLM text with a JSON option->percentage map (or list) -> normalised vector of length k, or None."""
    return normalize_probs(parse_json(text), k, labels)


def parse_item_probs(text: Any, keys: Sequence[str], scale: float = 100.0, strict: bool = True) -> dict:
    """JSON {key: 0-100} -> {key: probability in [0, 1]} (talp73 validator).

    Keys match case-insensitively after stripping; a single-key wrapper is unwrapped; "%" suffixes are accepted.
    strict=True raises ValueError on a missing key / non-number / out-of-range value (so the caller can retry);
    strict=False skips bad keys."""
    d = parse_json(text)
    if not isinstance(d, dict):
        if strict:
            raise ValueError("not a JSON object")
        return {}
    d = unwrap(d, keys)
    if not isinstance(d, dict):
        if strict:
            raise ValueError("not a JSON object")
        return {}
    dn = {str(k).strip().upper(): v for k, v in d.items()}
    out = {}
    for key in keys:
        v = dn.get(str(key).strip().upper())
        x = _num(v) if v is not None else None
        if x is None:
            if strict:
                raise ValueError(f"missing or non-numeric key {key}")
            continue
        if not 0 <= x <= scale:
            if strict:
                raise ValueError(f"out of range {key}={x}")
            continue
        out[key] = x / scale
    return out


def parse_text_fields(text: Any, keys: Sequence[str], strict: bool = True) -> dict:
    """JSON {key: sentence} -> {key: str} (SSR free-text validator, talp73/kod/llm/ssr.py)."""
    d = unwrap(parse_json(text), keys)
    if not isinstance(d, dict):
        if strict:
            raise ValueError("not a JSON object")
        return {}
    dn = {str(k).strip().upper(): v for k, v in d.items()}
    out = {}
    for key in keys:
        v = dn.get(str(key).strip().upper())
        if not isinstance(v, str) or not v.strip():
            if strict:
                raise ValueError(f"missing or empty sentence {key}")
            continue
        out[key] = v.strip()
    return out


__all__ = ["parse_json", "unwrap", "normalize_probs", "parse_distribution", "parse_item_probs",
           "parse_text_fields"]
