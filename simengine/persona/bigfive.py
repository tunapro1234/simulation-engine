"""Big Five personas in the BFI-2-Expanded sentence style (ported from replicant `personalities/factory.py`,
pyreplicant 0.3.0, MIT), plus the Big5-Scaler dial variant (edsl-test `personas/big5`).

Each domain has 7 intensity levels (-3..+3) with 3 sentences (one per facet). A 1-5 score maps to a weight
(score - 3) * 1.5; a fractional weight blends `round(frac * 3)` sentences of the upper level with the rest
from the lower level.

    from simengine.persona import bigfive as bf
    bf.bfi2_text({"agreeableness": 4.5, "neuroticism": 1.5})        # unspecified domains = 3.0 (average)
    people = bf.sample_bigfive(50, seed=1, overrides={"agreeableness": 2.0})
    panel = bf.bigfive_audience(spec, n=40, seed=0, strength=0.5)  # persona-strength knob

Sources: Soto & John (2017), BFI-2; Huang et al. (2024), BFI-2-Expanded format for LLMs; sentence bank from
replicant (calibrated in an earlier, unpublished project of the same owner). The sentence bank is English
only; a Turkish bank does not exist and any translation is unvalidated until `validate_bigfive` is rerun in
Turkish.

US_ADULT_NORMS are replicant's POPULATION_NORMS ("US adults, 1-5 Mini-IPIP scale"). replicant attributes its
sampler to Soto & John (2017), but these numbers are not that paper's Table 5 (SOTO_JOHN_2017_NORMS below);
their primary source is unverified.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from .methods.big5 import CORR as _SJ_CORR, NORMS as _SJ_NORMS, TRAITS as _SJ_TRAITS
from .methods.helpers import as_rng, cholesky, corr_matrix
from .schema import Trait

METHOD_NAME = "bfi2"

# 7-level sentence bank per domain (-3..+3), 3 sentences per level (one per facet); replicant factory.py
SENTENCES = {
    "extraversion": {
        -3: [
            'I am someone who strongly avoids social situations and prefers complete solitude.',
            'I am someone who never takes charge and actively avoids leadership roles.',
            'I am someone who has very low energy and avoids activity whenever possible.',
        ],
        -2: [
            'I am someone who tends to be quiet and avoids social gatherings.',
            'I am someone who prefers to have others take charge.',
            'I am someone who is less active than other people.',
        ],
        -1: [
            'I am someone who is somewhat reserved, though not antisocial.',
            'I am someone who occasionally lets others lead.',
            'I am someone who has moderate energy, leaning quieter.',
        ],
        0: [
            'I am someone who sometimes is outgoing, but not always.',
            'I am someone who sometimes acts as a leader, but not always.',
            'I am someone who sometimes is full of energy, but not always.',
        ],
        1: [
            'I am someone who is fairly sociable and enjoys company.',
            'I am someone who sometimes takes the lead in groups.',
            'I am someone who has good energy most of the time.',
        ],
        2: [
            'I am someone who is outgoing, sociable.',
            'I am someone who is dominant, acts as a leader.',
            'I am someone who is full of energy.',
        ],
        3: [
            'I am someone who is extremely outgoing and thrives on social interaction.',
            'I am someone who always takes charge and naturally dominates groups.',
            'I am someone who has boundless energy and is always on the go.',
        ],
    },
    "agreeableness": {
        -3: [
            "I am someone who is cold, uncaring, and indifferent to others' suffering.",
            'I am someone who is frequently rude and dismissive toward others.',
            'I am someone who is deeply suspicious and always assumes the worst.',
        ],
        -2: [
            'I am someone who can be cold and uncaring.',
            'I am someone who is sometimes rude to others.',
            'I am someone who tends to find fault with others.',
        ],
        -1: [
            'I am someone who is not particularly warm, though not unkind.',
            'I am someone who can be blunt, sometimes to a fault.',
            "I am someone who is somewhat skeptical of others' motives.",
        ],
        0: [
            'I am someone who sometimes is compassionate, but not always.',
            'I am someone who sometimes is respectful, but not always.',
            'I am someone who sometimes assumes the best about people, but not always.',
        ],
        1: [
            'I am someone who is generally kind and considerate.',
            'I am someone who usually treats others with respect.',
            'I am someone who tends to give people the benefit of the doubt.',
        ],
        2: [
            'I am someone who is compassionate, has a soft heart.',
            'I am someone who is respectful, treats others with respect.',
            'I am someone who assumes the best about people.',
        ],
        3: [
            'I am someone who is deeply compassionate and always puts others first.',
            'I am someone who treats everyone with the utmost respect and kindness.',
            'I am someone who always sees the good in people, no matter what.',
        ],
    },
    "conscientiousness": {
        -3: [
            'I am someone who is extremely disorganized and lives in chaos.',
            'I am someone who cannot finish tasks and gives up at the first obstacle.',
            'I am someone who is very careless and unreliable.',
        ],
        -2: [
            'I am someone who tends to be disorganized.',
            'I am someone who has difficulty getting started on tasks.',
            'I am someone who can be somewhat careless.',
        ],
        -1: [
            'I am someone who is not particularly organized, though not messy.',
            'I am someone who sometimes procrastinates.',
            'I am someone who is occasionally careless with details.',
        ],
        0: [
            'I am someone who sometimes keeps things tidy, but not always.',
            'I am someone who sometimes is persistent, but not always.',
            'I am someone who sometimes is reliable, but not always.',
        ],
        1: [
            'I am someone who generally keeps things in order.',
            'I am someone who usually follows through on tasks.',
            'I am someone who is mostly dependable.',
        ],
        2: [
            'I am someone who keeps things neat and tidy.',
            'I am someone who is persistent, works until the task is finished.',
            'I am someone who is reliable, can always be counted on.',
        ],
        3: [
            'I am someone who is meticulously organized and never tolerates disorder.',
            'I am someone who is relentlessly persistent and never leaves a task incomplete.',
            'I am someone who is unfailingly reliable in every situation.',
        ],
    },
    "neuroticism": {
        -3: [
            'I am someone who never worries and is immune to stress.',
            'I am someone who never feels sad or down, always completely content.',
            'I am someone who is unshakably calm and never gets emotional.',
        ],
        -2: [
            'I am someone who is relaxed, handles stress well.',
            'I am someone who feels secure, comfortable with self.',
            'I am someone who is emotionally stable, not easily upset.',
        ],
        -1: [
            "I am someone who doesn't worry much, though not completely carefree.",
            'I am someone who is generally content and secure.',
            'I am someone who is fairly even-tempered most of the time.',
        ],
        0: [
            'I am someone who sometimes worries, but not always.',
            'I am someone who sometimes feels down, but not always.',
            'I am someone who sometimes gets emotional, but not always.',
        ],
        1: [
            'I am someone who tends to worry more than average.',
            'I am someone who sometimes feels a bit down or blue.',
            'I am someone who can be somewhat sensitive emotionally.',
        ],
        2: [
            'I am someone who worries a lot.',
            'I am someone who tends to feel depressed, blue.',
            'I am someone who is temperamental, gets emotional easily.',
        ],
        3: [
            'I am someone who is consumed by worry and anxiety constantly.',
            'I am someone who frequently feels deeply depressed and hopeless.',
            'I am someone who is extremely volatile and emotionally unstable.',
        ],
    },
    "openness": {
        -3: [
            'I am someone who has absolutely no interest in art, music, or literature.',
            'I am someone who actively avoids abstract thinking and complex ideas.',
            'I am someone who has no creativity or imagination whatsoever.',
        ],
        -2: [
            'I am someone who has few artistic interests.',
            'I am someone who has little interest in abstract ideas.',
            'I am someone who has little creativity.',
        ],
        -1: [
            'I am someone who has limited interest in the arts.',
            'I am someone who prefers practical thinking over abstract ideas.',
            'I am someone who is not particularly creative or imaginative.',
        ],
        0: [
            'I am someone who sometimes appreciates art and music, but not always.',
            'I am someone who sometimes enjoys deep thinking, but not always.',
            'I am someone who sometimes comes up with new ideas, but not always.',
        ],
        1: [
            'I am someone who has some appreciation for art and culture.',
            'I am someone who occasionally enjoys exploring abstract ideas.',
            'I am someone who can be somewhat creative at times.',
        ],
        2: [
            'I am someone who is fascinated by art, music, or literature.',
            'I am someone who is complex, a deep thinker.',
            'I am someone who is original, comes up with new ideas.',
        ],
        3: [
            'I am someone who is passionately devoted to art, music, and literature.',
            'I am someone who constantly seeks out complex, abstract ideas.',
            'I am someone who is extraordinarily creative and always generating new ideas.',
        ],
    },
}

DOMAINS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]

# Named profiles on the sentence-bank weight scale (high = +2, low = -2, average = 0). Uncalibrated: replicant
# documents bias rather than compensating for it.
PROFILES = {
    "cooperative": {"extraversion": 0, "agreeableness": 2, "conscientiousness": 2, "neuroticism": -2, "openness": 0},
    "selfish": {"extraversion": 0, "agreeableness": -2, "conscientiousness": -2, "neuroticism": 2, "openness": 0},
    "leader": {"extraversion": 2, "agreeableness": 0, "conscientiousness": 2, "neuroticism": -2, "openness": 2},
    "anxious": {"extraversion": -2, "agreeableness": 0, "conscientiousness": 0, "neuroticism": 2, "openness": -2},
    "average": {"extraversion": 0, "agreeableness": 0, "conscientiousness": 0, "neuroticism": 0, "openness": 0},
}

# replicant POPULATION_NORMS, (mean, sd) on 1-5; order = DOMAINS (the draw order, as in replicant)
US_ADULT_NORMS = {
    "extraversion": (3.3, 0.75),
    "agreeableness": (3.8, 0.60),
    "conscientiousness": (3.7, 0.65),
    "neuroticism": (2.8, 0.75),
    "openness": (3.6, 0.65),
}
# Soto & John (2017) Table 5 (Internet sample, N = 1,000) and Table 2 intercorrelations, via edsl-test big5
SOTO_JOHN_2017_NORMS = dict(_SJ_NORMS)
SOTO_JOHN_2017_CORR = {a: {b: _SJ_CORR[i][j] for j, b in enumerate(_SJ_TRAITS) if b != a}
                       for i, a in enumerate(_SJ_TRAITS)}

DEFAULT_INSTRUCTION = (
    "You are a participant in an experiment. "
    "Your personality description below reflects who you are. "
    "Let it naturally shape your decisions and reasoning — "
    "act as a real person with this personality would."
)


# ---------------------------------------------------------------- scores <-> sentences
def score_to_weight(score: float) -> float:
    """1-5 score -> -3..+3 sentence-bank weight."""
    return (float(score) - 3.0) * 1.5


def weight_to_score(weight: float) -> float:
    """-3..+3 weight -> 1-5 score."""
    return float(weight) / 1.5 + 3.0


def level_from_score(score: float) -> str:
    """high (>= 3.75) / low (<= 2.25) / average, replicant's level-validation cut points."""
    if score >= 3.75:
        return "high"
    if score <= 2.25:
        return "low"
    return "average"


def blend_sentences(domain: str, weight: float) -> list[str]:
    """3 sentences for a (fractional) weight: round(frac * 3) from the upper level, the rest from the lower."""
    weight = max(-3.0, min(3.0, float(weight)))
    lo, hi = int(math.floor(weight)), int(math.ceil(weight))
    if lo == hi:
        return list(SENTENCES[domain][lo])
    n_hi = round((weight - lo) * 3)
    return SENTENCES[domain][lo][:3 - n_hi] + SENTENCES[domain][hi][:n_hi]


def build_description(weights: Mapping[str, float]) -> str:
    """{domain: weight} -> 15 lines (domain order = DOMAINS)."""
    lines = []
    for d in DOMAINS:
        lines.extend(blend_sentences(d, weights[d]))
    return "\n".join(lines)


def _check_scores(scores: Mapping[str, float]) -> dict[str, float]:
    unknown = set(scores) - set(DOMAINS)
    if unknown:
        raise ValueError(f"unknown Big Five domains: {sorted(unknown)} (use {DOMAINS})")
    return {d: float(scores.get(d, 3.0)) for d in DOMAINS}


def bfi2_text(scores: Mapping[str, float]) -> str:
    """OCEAN scores on 1-5 (missing domains = 3.0) -> BFI-2-Expanded persona sentences."""
    s = _check_scores(scores)
    return build_description({d: score_to_weight(v) for d, v in s.items()})


def profile_scores(name: str) -> dict[str, float]:
    """A named PROFILES entry as 1-5 scores (+2 -> 4.33, -2 -> 1.67)."""
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose from {list(PROFILES)}")
    return {d: weight_to_score(w) for d, w in PROFILES[name].items()}


def bigfive_traits(scores: Mapping[str, float]) -> list[Trait]:
    """TALP-schema traits (label = domain, value = score rounded to 2 decimals, group "big_five")."""
    s = _check_scores(scores)
    return [Trait(label=d, value=round(v, 2), group="big_five",
                  extra={"scale": "1-5", "level": level_from_score(v)}) for d, v in s.items()]


# ---------------------------------------------------------------- sampling
def _norms(norms: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    out = {}
    for d, v in norms.items():
        if isinstance(v, Mapping):
            out[d] = (float(v["mean"]), float(v["sd"]))
        else:
            out[d] = (float(v[0]), float(v[1]))
    if set(out) != set(DOMAINS):
        raise ValueError(f"norms must cover exactly {DOMAINS}")
    return out


def sample_bigfive(n: int, seed: Any = 0, norms: Mapping[str, Any] = US_ADULT_NORMS, sd_scale: float = 1.0,
                   overrides: Mapping[str, float] | None = None, corr: Any = None) -> list[dict]:
    """n Big Five personas from population norms, as flat persona dicts.

    Scores ~ Normal(mean, sd * sd_scale) per domain, clipped to [1, 5]. `overrides` replaces domain means
    (1-5). corr=None draws domains independently in the order of `norms` (replicant `sample_personalities`;
    the same seed gives replicant's scores); a correlation (nested dict or matrix in `norms` order, e.g.
    SOTO_JOHN_2017_CORR) draws them jointly via Cholesky.

    Each dict: {"id": f"bfi2-{seed}-{k}", "text", "persona_method": "bfi2", "scores", "params": {"scores",
    "sd_scale", "overrides", "seed"}}."""
    if sd_scale < 0:
        raise ValueError("sd_scale must be >= 0")
    nm = _norms(norms)
    names = list(nm)
    ov = dict(overrides or {})
    bad = set(ov) - set(DOMAINS)
    if bad:
        raise ValueError(f"unknown override domains: {sorted(bad)}")
    means = {d: float(ov.get(d, nm[d][0])) for d in names}
    sds = {d: nm[d][1] * sd_scale for d in names}
    L = cholesky(corr_matrix(names, corr)) if corr is not None else None
    rng = as_rng(seed)
    out = []
    for k in range(n):
        if L is None:
            raw = {d: (rng.gauss(means[d], sds[d]) if sds[d] > 0 else means[d]) for d in names}
        else:
            e = [rng.gauss(0, 1) for _ in names]
            z = [sum(L[i][j] * e[j] for j in range(i + 1)) for i in range(len(names))]
            raw = {d: means[d] + sds[d] * z[i] for i, d in enumerate(names)}
        scores = {d: max(1.0, min(5.0, raw[d])) for d in DOMAINS}
        out.append({"id": f"{METHOD_NAME}-{seed}-{k}", "text": bfi2_text(scores), "persona_method": METHOD_NAME,
                    "scores": scores,
                    "params": {"scores": scores, "sd_scale": sd_scale, "overrides": ov or None, "seed": seed}})
    return out


def _target(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return min(5.0, max(1.0, float(value)))
    v = str(value).strip().lower()
    return {"high": weight_to_score(2), "low": weight_to_score(-2), "average": 3.0, "medium": 3.0}.get(v)


def audience_targets(spec: Any) -> dict[str, float]:
    """Big Five mean targets from `spec.audience.trait_requirements` entries whose dimension is a domain:
    values[0] is a 1-5 number or high (4.33) / average (3.0) / low (1.67)."""
    out = {}
    aud = getattr(spec, "audience", None)
    for req in (getattr(aud, "trait_requirements", None) or []):
        if not isinstance(req, Mapping):
            continue
        dim = str(req.get("dimension", "")).strip().lower()
        vals = req.get("values") or []
        if dim in DOMAINS and vals:
            t = _target(vals[0])
            if t is not None:
                out[dim] = t
    return out


def bigfive_audience(spec: Any, n: int, seed: Any = 0, strength: float = 1.0,
                     norms: Mapping[str, Any] = US_ADULT_NORMS, targets: Mapping[str, float] | str | None = None,
                     corr: Any = None) -> list[dict]:
    """n Big Five personas for an AudienceSpec, with one persona-strength knob.

    Base means = `targets` (dict of 1-5 means or a PROFILES name), else the spec's trait_requirements on Big
    Five domains, else the norms. Then mean = 3 + strength * (base - 3) and sd_scale = strength:
    strength 0 gives everyone the all-average "sometimes ..., but not always" text, 1 the targets/norms as
    they are, > 1 exaggerated personalities. Demographic cells come from `cell_plan(spec, n, seed)` (the same
    quota plan as the LLM sources); they are kept as fields and rendered as one identity line above the
    sentences. Returns flat persona dicts (full text, no 600-character clipping), ready for simengine.sim."""
    from .sources import cell_plan
    if strength < 0:
        raise ValueError("strength must be >= 0")
    nm = _norms(norms)
    if isinstance(targets, str):
        base = profile_scores(targets)
    elif targets is not None:
        base = {d: float(v) for d, v in _check_scores({**{d: nm[d][0] for d in DOMAINS}, **targets}).items()}
    else:
        base = {d: nm[d][0] for d in DOMAINS}
        base.update(audience_targets(spec))
    means = {d: min(5.0, max(1.0, 3.0 + strength * (base[d] - 3.0))) for d in DOMAINS}
    people = sample_bigfive(n, seed=seed, norms=nm, sd_scale=strength, overrides=means, corr=corr)
    cells = cell_plan(spec, n, seed)
    out = []
    for p, cell in zip(people, cells):
        head = ", ".join(x for x in [f"age {cell['age_band']}" if cell.get("age_band") else "",
                                     cell.get("gender") or ""] if x)
        rest = "; ".join(f"{k}: {v}" for k, v in cell.items() if k not in ("age_band", "gender"))
        ident = "; ".join(x for x in (head, rest) if x)
        p = dict(p)
        p["text"] = (ident + "\n" + p["text"]) if ident else p["text"]
        p.update(cell)
        p["segment"] = cell.get("income_tier") or cell.get("age_band")
        p["locale"] = getattr(spec, "locale", "") or ""
        p["params"] = {**p["params"], "strength": strength, "base_means": base, "cell": cell}
        out.append(p)
    return out


__all__ = ["SENTENCES", "DOMAINS", "PROFILES", "US_ADULT_NORMS", "SOTO_JOHN_2017_NORMS", "SOTO_JOHN_2017_CORR",
           "DEFAULT_INSTRUCTION", "METHOD_NAME", "score_to_weight", "weight_to_score", "level_from_score",
           "blend_sentences", "build_description", "bfi2_text", "profile_scores", "bigfive_traits",
           "sample_bigfive", "audience_targets", "bigfive_audience"]
