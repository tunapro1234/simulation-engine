"""Schwartz value anchor (edsl-test personas/value_anchor; Rozen et al., arXiv:2407.12878).

Persona = the paper's verbatim template "Answer as a person that is [value]" (p.4), where [value] is one of
the 19 Best-Worst Refined Values descriptions of the paper's Appendix E (the 20th BWVr item, animal welfare,
is excluded). Appendix E swaps the labels of items 6 and 7 (power-dominance / power-resources); the source
keeps the paper's texts but files them under the standard Schwartz value codes, and so do we.

Each anchor is drawn i.i.d. with the probability that a person holds it as their MOST important value:
PREVALENCE_WEIGHTS inlined from edsl-test personas/value_anchor/weights.json ("Lee et al. 2019, Table 2,
Study 2 (N=439)", 1,000,000 simulated draws by its prep_weights.py). Pass weights="uniform" for the paper's own
uniform setup.
"""
from __future__ import annotations

from ._base import FunctionMethod
from .helpers import as_rng

TEMPLATE = "Answer as a person that is {value}."

ANCHORS = {
    "self-direction-thought": "developing your own original ideas and opinions",
    "self-direction-action": "being free to act independently",
    "stimulation": "having an exciting life; having all sorts of new experiences",
    "hedonism": "taking advantage of every opportunity to enjoy life's pleasures",
    "achievement": "being ambitious and successful",
    "power-resources": "having the power that money and possessions can bring",
    "power-dominance": "having the authority to get others to do what you want",
    "face": "protecting your public image and avoiding being shamed",
    "security-personal": "living and acting in ways that ensure that you are personally safe and secure",
    "security-societal": "living in a safe and stable society",
    "tradition": "following cultural family or religious practices",
    "conformity-rules": "obeying all rules and laws",
    "conformity-interpersonal": "making sure you never upset or annoy others",
    "humility": "being humble and avoiding public recognition",
    "benevolence-dependability": "being a completely dependable and trustworthy friend and family member",
    "benevolence-caring": "helping and caring for the wellbeing of those who are close",
    "universalism-concern": "caring and seeking justice for everyone especially the weak and vulnerable in society",
    "universalism-nature": "protecting the natural environment from destruction or pollution",
    "universalism-tolerance": "being open-minded and accepting of people and ideas, even when you disagree with them",
}

# edsl-test personas/value_anchor/weights.json: P(value is the most important one), Lee et al. 2019 Study 2
PREVALENCE_WEIGHTS = {
    "self-direction-thought": 0.0622, "self-direction-action": 0.1165, "stimulation": 0.0483,
    "hedonism": 0.0870, "achievement": 0.0727, "power-resources": 0.0108, "power-dominance": 0.0005,
    "face": 0.0029, "security-personal": 0.0597, "security-societal": 0.0527, "tradition": 0.0111,
    "conformity-rules": 0.0053, "conformity-interpersonal": 0.0041, "humility": 0.0057,
    "benevolence-dependability": 0.1652, "benevolence-caring": 0.1463, "universalism-concern": 0.0661,
    "universalism-nature": 0.0285, "universalism-tolerance": 0.0543,
}


def render(value_code):
    return TEMPLATE.format(value=ANCHORS[value_code])


def make(weights=None, name="value_anchor"):
    """weights: None (prevalence), "uniform", or a dict over ANCHORS codes."""
    if weights is None:
        w = dict(PREVALENCE_WEIGHTS)
    elif weights == "uniform":
        w = {v: 1.0 for v in ANCHORS}
    else:
        w = {v: float(weights.get(v, 0.0)) for v in ANCHORS}
    codes = list(ANCHORS)

    def _draw(n, seed=None, data_root=None):
        rng = as_rng(seed)
        values = rng.choices(codes, weights=[w[v] for v in codes], k=n)
        return [(render(v), {"value": v}) for v in values]

    return FunctionMethod(
        name=name, draw_fn=_draw, info={"weights": w},
        description="One Schwartz value anchor per persona, drawn by human most-important-value prevalence.",
        paper="Rozen et al. 2024 (arXiv:2407.12878); Lee et al. 2019 (BWVr prevalence)",
        source="edsl-test personas/value_anchor/__init__.py, weights.json")


METHOD = make()
