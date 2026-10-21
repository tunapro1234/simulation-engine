"""Cross-instrument fidelity check for Big Five personas (ported from replicant `personalities/validation.py`).

Assign a personality with one instrument (BFI-2-Expanded sentences, or Big5-Scaler dials), measure it with a
DIFFERENT one (the 20-item Mini-IPIP, Donnellan et al. 2006), and compare assigned with measured scores:
Pearson r, r^2, MAE and bias (measured - assigned) per domain and pooled.

    from simengine.persona.validate import validate_bigfive, to_card_row
    res = validate_bigfive(llm, n=20, seed=42)          # any simengine.types.LLM; FakeLLM in tests
    res["pooled"], res["metrics"]["agreeableness"]["bias"], res["n_missing"]
    row = to_card_row(res)                              # flat dict for an evidence table

Port fixes against the source:
  * no NaN -> 3 imputation: an unparseable or empty answer stays missing, a domain with a missing item is
    missing (min_items=4), and both are counted (`n_missing`, `n_missing_pairs`); metrics use complete pairs.
  * no EDSL Agent/Survey: persona text is the `system` prompt, each Mini-IPIP item is one `user` prompt sent
    through `llm.map` (the source also asked one item per question).
  * answers are parsed with `simengine.sim.parse.parse_json` (JSON) or a strict bare-number rule; a number
    is never pulled out of prose ("Not 5" is missing, not 5).
Measured with the source code (replicant papers/ertan2009/results/validation_continuous.json, model
stepfun/step-3.5-flash, n = 20, seed 42, NaN answers imputed as 3): pooled r .911, r^2 .830, MAE .666; bias
A +0.735, N -0.692, O +0.660, C +0.444, E -0.034. `validate_bigfive(llm, n=20, seed=42)` assigns the same
scores as that run.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable, Mapping, Sequence

from .bigfive import DOMAINS, METHOD_NAME, bfi2_text, sample_bigfive

# Mini-IPIP (Donnellan, Oswald, Baird & Lucas 2006), 20 items, 4 per domain, 1-5 accuracy scale
MINI_IPIP = [
    {"id": 1, "text": "Am the life of the party", "domain": "extraversion", "reverse": False},
    {"id": 2, "text": "Don't talk a lot", "domain": "extraversion", "reverse": True},
    {"id": 3, "text": "Talk to a lot of different people at parties", "domain": "extraversion", "reverse": False},
    {"id": 4, "text": "Keep in the background", "domain": "extraversion", "reverse": True},
    {"id": 5, "text": "Sympathize with others' feelings", "domain": "agreeableness", "reverse": False},
    {"id": 6, "text": "Am not interested in other people's problems", "domain": "agreeableness", "reverse": True},
    {"id": 7, "text": "Feel others' emotions", "domain": "agreeableness", "reverse": False},
    {"id": 8, "text": "Am not really interested in others", "domain": "agreeableness", "reverse": True},
    {"id": 9, "text": "Get chores done right away", "domain": "conscientiousness", "reverse": False},
    {"id": 10, "text": "Often forget to put things back in their proper place", "domain": "conscientiousness",
     "reverse": True},
    {"id": 11, "text": "Like order", "domain": "conscientiousness", "reverse": False},
    {"id": 12, "text": "Make a mess of things", "domain": "conscientiousness", "reverse": True},
    {"id": 13, "text": "Have frequent mood swings", "domain": "neuroticism", "reverse": False},
    {"id": 14, "text": "Am relaxed most of the time", "domain": "neuroticism", "reverse": True},
    {"id": 15, "text": "Get upset easily", "domain": "neuroticism", "reverse": False},
    {"id": 16, "text": "Seldom feel blue", "domain": "neuroticism", "reverse": True},
    {"id": 17, "text": "Have a vivid imagination", "domain": "openness", "reverse": False},
    {"id": 18, "text": "Am not interested in abstract ideas", "domain": "openness", "reverse": True},
    {"id": 19, "text": "Have difficulty understanding abstract ideas", "domain": "openness", "reverse": True},
    {"id": 20, "text": "Do not have a good imagination", "domain": "openness", "reverse": True},
]
_ITEM = {it["id"]: it for it in MINI_IPIP}

OPTION_LABELS = {1: "Very inaccurate", 2: "Moderately inaccurate", 3: "Neutral", 4: "Moderately accurate",
                 5: "Very accurate"}

VALIDATION_INSTRUCTION = (
    "You are a participant in a psychology study. "
    "Your personality description below reflects who you are. "
    "Let it naturally shape your responses — answer as a real "
    "person with this personality would."
)
WORDING_ID = "mini_ipip_en_v1"


def item_prompt(item: Mapping[str, Any]) -> str:
    """User prompt for one Mini-IPIP item (the source's question text and option labels)."""
    opts = "\n".join(f"{k}: {v}" for k, v in OPTION_LABELS.items())
    return (f'Rate how accurately this describes you: "{item["text"]}"\n\n{opts}\n\n'
            "Answer with a single number from 1 to 5.")


def system_prompt(persona_text: str, instruction: str = VALIDATION_INSTRUCTION) -> str:
    return f"{instruction}\n\nYour personality:\n{persona_text}" if persona_text else instruction


# ---------------------------------------------------------------- parsing and scoring
_LABEL_RE = "|".join(re.escape(v) for v in sorted(OPTION_LABELS.values(), key=len, reverse=True))


def parse_likert(text: Any, lo: float = 1.0, hi: float = 5.0) -> float | None:
    """LLM answer -> number in [lo, hi], or None (missing; never imputed).

    Accepts only a strict answer form: a bare number ("4", "4.0", "4."), optionally prefixed with
    "Answer:"/"Rating:", optionally followed by "/5" or "out of 5" and/or an option label
    ("4 - Moderately accurate", "5 Very accurate"), or JSON ({"answer": 4} or 4, fenced or not).
    A number is never pulled out of prose: "I would say 4.", "Not 5" and "never 1" are None."""
    if text is None:
        return None
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        x = float(text)
        return x if lo <= x <= hi and math.isfinite(x) else None
    s = str(text).strip()
    if not s:
        return None
    if s[0] in "{[" or s.startswith("```"):
        from ..sim.parse import parse_json
        v = parse_json(s)
        if isinstance(v, dict):
            for k in ("answer", "rating", "score", "value", "response"):
                if k in v:
                    return parse_likert(v[k], lo, hi)
            vals = list(v.values())
            return parse_likert(vals[0], lo, hi) if len(vals) == 1 else None
        if isinstance(v, list) and len(v) == 1:
            return parse_likert(v[0], lo, hi)
        return None
    top = re.escape(f"{hi:g}")
    m = re.fullmatch(
        rf"(?:(?:answer|rating)\s*:\s*)?([-+]?\d+(?:\.\d+)?)"
        rf"(?:\s*(?:/|out\s+of)\s*{top}(?:\.0+)?)?"
        rf"(?:\s*[-:)(,]?\s*(?:{_LABEL_RE})\)?)?\s*\.?",
        s, re.I)
    if not m:
        return None
    x = float(m.group(1))
    return x if lo <= x <= hi else None


def _item_id(key: Any) -> int | None:
    if isinstance(key, int):
        return key
    m = re.search(r"(\d+)$", str(key))
    return int(m.group(1)) if m else None


def score_mini_ipip(answers: Mapping[Any, Any], min_items: int = 4) -> dict[str, float]:
    """{item: answer} -> {domain: 1-5 score}. Items are keyed 1..20, "ipip_1", "answer.ipip_1", ...

    Answers are parsed with `parse_likert`; reverse-keyed items score 6 - x. A domain with fewer than
    `min_items` valid answers is NaN (default 4 = all items required). Missing answers are never imputed."""
    vals: dict[int, float] = {}
    for k, v in answers.items():
        i = _item_id(k)
        if i in _ITEM:
            x = parse_likert(v)
            if x is not None:
                vals[i] = 6.0 - x if _ITEM[i]["reverse"] else x
    out = {}
    for d in DOMAINS:
        xs = [vals[it["id"]] for it in MINI_IPIP if it["domain"] == d and it["id"] in vals]
        out[d] = sum(xs) / len(xs) if xs and len(xs) >= min_items else float("nan")
    return out


def compute_metrics(assigned: Sequence[float], measured: Sequence[float]) -> dict[str, float]:
    """Pearson r, r^2, MAE and bias (mean measured - mean assigned) over complete pairs (population moments,
    as in the source). r is NaN with fewer than 2 pairs or zero variance."""
    pairs = [(float(a), float(m)) for a, m in zip(assigned, measured)
             if a is not None and m is not None and not math.isnan(float(a)) and not math.isnan(float(m))]
    n = len(pairs)
    nan = float("nan")
    if n == 0:
        return {"r": nan, "r2": nan, "mae": nan, "bias": nan, "n": 0, "assigned_mean": nan, "measured_mean": nan}
    ma = sum(a for a, _ in pairs) / n
    mm = sum(m for _, m in pairs) / n
    cov = sum((a - ma) * (m - mm) for a, m in pairs) / n
    va = sum((a - ma) ** 2 for a, _ in pairs) / n
    vm = sum((m - mm) ** 2 for _, m in pairs) / n
    r = cov / math.sqrt(va * vm) if n >= 2 and va > 0 and vm > 0 else nan
    return {"r": r, "r2": r * r if not math.isnan(r) else nan, "mae": sum(abs(a - m) for a, m in pairs) / n,
            "bias": mm - ma, "n": n, "assigned_mean": ma, "measured_mean": mm}


# ---------------------------------------------------------------- builders
def _build(builder: str | Callable, n: int, seed: Any, sample_kw: Mapping[str, Any] | None = None
           ) -> tuple[str, list[dict]]:
    """-> (builder name, [{"agent_id", "text", "assigned", "params"}])."""
    kw = dict(sample_kw or {})
    if callable(builder):
        people = sample_bigfive(n, seed=seed, **kw)
        return getattr(builder, "__name__", "custom"), [
            {"agent_id": p["id"], "text": builder(p["scores"]), "assigned": p["scores"], "params": p["params"]}
            for p in people]
    if builder in ("bfi2", METHOD_NAME):
        return "bfi2", [{"agent_id": p["id"], "text": p["text"], "assigned": p["scores"], "params": p["params"]}
                        for p in sample_bigfive(n, seed=seed, **kw)]
    if builder in ("big5", "big5_scaler"):
        if kw:
            raise ValueError("sample_kw applies to the bfi2 and callable builders only")
        from .methods import big5
        from .methods.helpers import as_rng
        rng = as_rng(seed)
        out = []
        for k in range(n):
            d = big5.SAMPLER.dials(rng)
            out.append({"agent_id": f"big5-{seed}-{k}", "text": big5.render(d),
                        "assigned": {t: big5.dial_to_score(d[t]) for t in DOMAINS},
                        "params": {"dials": d, "seed": seed}})
        return "big5_scaler", out
    raise ValueError(f"unknown builder {builder!r}: use 'bfi2', 'big5_scaler' or a callable(scores) -> text")


def validate_bigfive(llm: Any, n: int = 20, seed: Any = 42, builder: str | Callable = "bfi2",
                     max_tokens: int = 64, instruction: str = VALIDATION_INSTRUCTION, min_items: int = 4,
                     parallel: int = 4, sample_kw: Mapping[str, Any] | None = None, **llm_kw: Any) -> dict:
    """Assign n sampled personalities, measure each with the 20 Mini-IPIP items, compare.

    builder: "bfi2" (BFI-2-Expanded sentences, US_ADULT_NORMS, independent domains; seed 42 reproduces the
    source run's assigned scores), "big5_scaler" (edsl-test Big5-Scaler 0-10 dials, Soto & John 2017 norms
    and correlations; assigned = the 1-5 score the dial states), or callable(scores) -> text over
    `sample_bigfive` scores. `sample_kw` is passed to `sample_bigfive` (norms, sd_scale, overrides, corr), e.g.
    to validate a skewed population.

    Returns {"builder", "model", "n", "seed", "max_tokens", "wording_id", "metrics": {domain: {r, r2, mae, bias,
    n, assigned_mean, measured_mean}}, "pooled": {...}, "n_pairs", "n_missing" (item answers that did not
    parse), "n_missing_pairs" (agent-domain pairs dropped), "n_items", "paired": [...], "rows": [...]}. Each
    row is one item answer with persona_method, agent_id, seed, params, max_tokens, wording_id, raw, value and
    valid."""
    name, agents = _build(builder, n, seed, sample_kw)
    prompts = []
    for a in agents:
        sys_p = system_prompt(a["text"], instruction)
        for it in MINI_IPIP:
            prompts.append((sys_p, item_prompt(it)))
    resps = llm.map(prompts, max_tokens=max_tokens, parallel=parallel, **llm_kw) if prompts else []
    if len(resps) != len(prompts):
        raise RuntimeError(f"llm.map returned {len(resps)} responses for {len(prompts)} prompts")
    rows, paired = [], []
    per_domain: dict[str, tuple[list, list]] = {d: ([], []) for d in DOMAINS}
    n_missing = n_missing_pairs = 0
    k = 0
    for a in agents:
        answers = {}
        for it in MINI_IPIP:
            raw = getattr(resps[k], "text", resps[k])
            k += 1
            v = parse_likert(raw)
            valid = v is not None
            n_missing += not valid
            answers[it["id"]] = v
            rows.append({"persona_method": name, "agent_id": a["agent_id"], "seed": seed, "params": a["params"],
                         "max_tokens": max_tokens, "wording_id": WORDING_ID, "item": it["id"],
                         "domain": it["domain"], "reverse": it["reverse"], "raw": raw, "value": v,
                         "valid": valid})
        measured = score_mini_ipip(answers, min_items=min_items)
        for d in DOMAINS:
            m = measured[d]
            ok = not math.isnan(m)
            n_missing_pairs += not ok
            paired.append({"agent_id": a["agent_id"], "domain": d, "assigned": a["assigned"][d],
                           "measured": m if ok else None, "valid": ok})
            if ok:
                per_domain[d][0].append(a["assigned"][d])
                per_domain[d][1].append(m)
    metrics = {d: compute_metrics(*per_domain[d]) for d in DOMAINS}
    all_a = [x for d in DOMAINS for x in per_domain[d][0]]
    all_m = [x for d in DOMAINS for x in per_domain[d][1]]
    return {"builder": name, "model": getattr(llm, "model", ""), "n": n, "seed": seed, "max_tokens": max_tokens,
            "wording_id": WORDING_ID, "metrics": metrics, "pooled": compute_metrics(all_a, all_m),
            "n_pairs": len(all_a), "n_missing": n_missing, "n_missing_pairs": n_missing_pairs,
            "n_items": len(prompts), "paired": paired, "rows": rows}


def to_card_row(result: Mapping[str, Any]) -> dict:
    """Flatten a validate_bigfive result into one row: model, builder, n, seed, pooled_{r,r2,mae,bias},
    <domain>_{r,r2,mae,bias}, n_pairs, n_missing, n_missing_pairs, completion (valid item answers / items)."""
    row = {k: result.get(k) for k in ("model", "builder", "n", "seed", "max_tokens", "wording_id")}
    for m in ("r", "r2", "mae", "bias"):
        row[f"pooled_{m}"] = result["pooled"].get(m)
    for d in DOMAINS:
        for m in ("r", "r2", "mae", "bias"):
            row[f"{d}_{m}"] = result["metrics"][d].get(m)
    row["n_pairs"] = result.get("n_pairs")
    row["n_missing"] = result.get("n_missing")
    row["n_missing_pairs"] = result.get("n_missing_pairs")
    n_items = result.get("n_items") or 0
    row["completion"] = (n_items - (result.get("n_missing") or 0)) / n_items if n_items else float("nan")
    return row


__all__ = ["MINI_IPIP", "OPTION_LABELS", "VALIDATION_INSTRUCTION", "WORDING_ID", "item_prompt", "system_prompt",
           "parse_likert", "score_mini_ipip", "compute_metrics", "validate_bigfive", "to_card_row"]
