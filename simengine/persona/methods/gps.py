"""Global Preferences Survey persona (edsl-test personas/gps).

Each agent is one joint draw of the six GPS preferences (patience, risk taking, positive reciprocity, negative
reciprocity, altruism, trust), written out as that person's answers to the GPS's eight qualitative items
(verbatim wordings and scale anchors).

Sources: Falk, Becker, Dohmen, Enke, Huffman & Sunde, "Global Evidence on Economic Preferences", NBER WP 23943
(QJE 2018, 133(4)): item wordings, Appendix A.6 (pp.50-53); individual-level partial correlations between the
six preference measures, Appendix C, Table 12 (p.62), ~80,000 people, conditional on country fixed effects;
latent scale mean 0 / SD 1 (QJE p.1653).

Assumptions of the source (not GPS's): latent z -> 0-10 answer is round(5 + 2z) clipped to [0, 10] (raw item
distributions are unpublished); all items of one preference get the same answer; the correlations of the
composite measures are applied to the qualitative-item z-scores; the "(far) above/below the world average"
tags bucket z at +-0.5 and +-1.5 SD. All marginals are world-level placeholders, not a Turkey population.
"""
from __future__ import annotations

from ._base import FunctionMethod
from .helpers import TraitDialSampler, as_rng

PREFS = ["patience", "risk", "posrec", "negrec", "altruism", "trust"]

# NBER WP 23943, Appendix C, Table 12 (p.62); order = PREFS
R = [
    [1.000, 0.210, 0.084, 0.112, 0.098, 0.044],
    [0.210, 1.000, 0.068, 0.228, 0.106, 0.047],
    [0.084, 0.068, 1.000, 0.010, 0.329, 0.114],
    [0.112, 0.228, 0.010, 1.000, 0.067, 0.075],
    [0.098, 0.106, 0.329, 0.067, 1.000, 0.151],
    [0.044, 0.047, 0.114, 0.075, 0.151, 1.000],
]

WILLINGNESS_ITEMS = [
    ("patience", "How willing are you to give up something that is beneficial "
                 "for you today in order to benefit more from that in the future?"),
    ("altruism", "How willing are you to give to good causes without expecting "
                 "anything in return?"),
    ("negrec", "How willing are you to punish someone who treats you unfairly, "
               "even if there may be costs for you?"),
    ("negrec", "How willing are you to punish someone who treats others unfairly, "
               "even if there may be costs for you?"),
]
RISK_ITEM = "Please tell me, in general, how willing or unwilling you are to take risks."
SELF_ASSESSMENT_ITEMS = [
    ("posrec", "When someone does me a favor I am willing to return it."),
    ("negrec", "If I am treated very unjustly, I will take revenge at the first "
               "occasion, even if there is a cost to do so."),
    ("trust", "I assume that people have only the best intentions."),
]


def to_answer(z):
    """Latent z -> 0-10 integer survey answer (assumes item mean 5, SD 2)."""
    return max(0, min(10, round(5 + 2 * z)))


def level(z):
    """Relative standing in SD units, the way the GPS paper reports differences."""
    if z < -1.5:
        return "far below the world average"
    if z < -0.5:
        return "below the world average"
    if z <= 0.5:
        return "close to the world average"
    if z <= 1.5:
        return "above the world average"
    return "far above the world average"


def render(a):
    """{pref: 0-10 answer} -> persona text."""
    lines = ["You answered the Global Preferences Survey (the standard worldwide "
             "survey of economic preferences) as follows."]
    lines.append('Your willingness to act, from 0 ("completely unwilling to do '
                 'so") to 10 ("very willing to do so"):')
    for p, item in WILLINGNESS_ITEMS:
        lines.append(f'- "{item}" — your answer: {a[p]} ({level((a[p] - 5) / 2)}).')
    lines.append(f'- "{RISK_ITEM}" From 0 ("completely unwilling to take risks") '
                 f'to 10 ("very willing to take risks") — your answer: '
                 f'{a["risk"]} ({level((a["risk"] - 5) / 2)}).')
    lines.append('How well each statement describes you as a person, from 0 '
                 '("does not describe me at all") to 10 ("describes me perfectly"):')
    for p, item in SELF_ASSESSMENT_ITEMS:
        lines.append(f'- "{item}" — your answer: {a[p]} ({level((a[p] - 5) / 2)}).')
    lines.append("You are this person; in every decision, act as they would.")
    return "\n".join(lines)


SAMPLER = TraitDialSampler({p: (0.0, 1.0) for p in PREFS}, R, template=render, transform=to_answer)


def _draw(n, seed=None, data_root=None):
    rng = as_rng(seed)
    out = []
    for _ in range(n):
        a = SAMPLER.dials(rng)
        out.append((render(a), {"answers": a}))
    return out


METHOD = FunctionMethod(
    name="gps", draw_fn=_draw,
    description="GPS economic preferences, jointly sampled, rendered as answers to the 8 qualitative items.",
    paper="Falk et al. 2018 (QJE 133(4)); NBER WP 23943",
    source="edsl-test personas/gps/__init__.py")
