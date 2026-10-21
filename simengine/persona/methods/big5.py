"""Big Five numeric dials, Big5-Scaler "Simple Prompt" (edsl-test personas/big5; Cho & Cheong 2025,
arXiv:2508.06149, Figure 2).

One sentence per trait stating what a high score means plus "Your X score is {s} out of 10", then the paper's
closing line. The source adds a neutral low-pole description to each sentence (paraphrasing BFI-2
reverse-keyed items), because a population sample draws low scores the high-pole-only template leaves
undefined.

Scores: Soto & John (2017), JPSP 113(1), Internet sample (N = 1,000) domain means/SDs (Table 5, p.128) and
domain intercorrelations (Table 2, p.125), sampled jointly (Cholesky), truncated to the 1-5 BFI-2 scale and
rescaled to integers 0-10. The sample skews young and self-selected. BFI factor structure is not guaranteed to
hold inside an LLM; treat trait -> behavior claims modestly.
"""
from __future__ import annotations

from ._base import FunctionMethod
from .helpers import TraitDialSampler, as_rng

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# Soto & John (2017), Table 5, p.128: BFI-2 domain mean (SD), 1-5 scale, Internet sample, Combined (N = 1,000)
NORMS = {
    "openness": (3.92, 0.65),
    "conscientiousness": (3.43, 0.77),
    "extraversion": (3.23, 0.80),
    "agreeableness": (3.68, 0.64),
    "neuroticism": (3.07, 0.87),
}

# Soto & John (2017), Table 2, p.125: domain intercorrelations, Internet sample; order = TRAITS
CORR = [
    [1.00, -0.02, 0.20, 0.15, -0.06],
    [-0.02, 1.00, 0.22, 0.28, -0.30],
    [0.20, 0.22, 1.00, 0.14, -0.34],
    [0.15, 0.28, 0.14, 1.00, -0.29],
    [-0.06, -0.30, -0.34, -0.29, 1.00],
]

# (high pole verbatim from the paper's Simple Prompt, low pole paraphrasing BFI-2 reverse-keyed items)
GLOSS = {
    "openness": ("are imaginative, curious, and creative",
                 "are practical, conventional, and prefer familiar routines"),
    "conscientiousness": ("are disciplined and dependable",
                          "are easygoing, spontaneous, and less organized"),
    "extraversion": ("are outgoing, enthusiastic, and enjoy social interactions",
                     "are reserved, quiet, and content to let others take charge"),
    "agreeableness": ("prioritize harmony and positive relationships",
                      "are competitive, skeptical of others, and stand their ground"),
    "neuroticism": ("are more emotionally reactive and prone to mood swings",
                    "are calm, secure, and emotionally stable"),
}
CLOSING = "From now on, you are an agent with this personality, and you should respond based on this personality."


def to_dial(x):
    """1-5 score -> integer 0-10 dial."""
    return round((x - 1) / 4 * 10)


def dial_to_score(d):
    """0-10 dial -> the 1-5 score it states (for fidelity checks)."""
    return 1 + 4 * float(d) / 10


def render(dials):
    lines = []
    for trait in TRAITS:
        high, low = GLOSS[trait]
        lines.append(f"People with high {trait} score {high}; people with low {trait} score {low}. "
                     f"Your {trait} score is {dials[trait]} out of 10.")
    lines.append(CLOSING)
    return " ".join(lines)


SAMPLER = TraitDialSampler(NORMS, CORR, template=render, clip=(1.0, 5.0), transform=to_dial)


def _draw(n, seed=None, data_root=None):
    rng = as_rng(seed)
    out = []
    for _ in range(n):
        d = SAMPLER.dials(rng)
        out.append((render(d), {"dials": d}))
    return out


METHOD = FunctionMethod(
    name="big5", draw_fn=_draw,
    description="Big Five 0-10 dials (Big5-Scaler Simple Prompt), jointly sampled from BFI-2 norms.",
    paper="Cho & Cheong 2025 (arXiv:2508.06149); Soto & John 2017 (JPSP 113(1))",
    source="edsl-test personas/big5/__init__.py")
