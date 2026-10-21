"""Homo Silicus preference types + calibrated mixture (edsl-test personas/homo_silicus).

Each persona = the General Social Agents "humanizing" prefix sentence + one Horton-style social-preference
one-liner. The population is a mixture over 6 types grounded in Charness & Rabin (2002, QJE):
  self_interested / efficient / inequity_averse : Horton (2023), arXiv:2301.07543, Fig. 3 ("both players" ->
                                                  "all players" for n-player games)
  competitive, reciprocal                        : Charness & Rabin (2002) competitive and reciprocity preferences
  selfish_but_fair                               : verbatim example prompt, Manning et al., General Social
                                                  Agents (arXiv:2508.17407), Sec. 2.1
Sampling is deterministic largest-remainder quotas (`quota_mixture`), shuffled by seed: the composition of a
group of n never depends on the seed.

CALIBRATED_WEIGHTS are inlined from edsl-test personas/homo_silicus/weights.json (written by its calibrate.py,
model gpt-oss-120b: fitted on the 6 Charness-Rabin allocation menus, train SSE 0.089, held-out Engel 2011
dictator RMSE 0.315). The reciprocal type got weight 0, so this mixture can never contain a punisher.
HORTON_WEIGHTS are Horton's published GPT-4o mixture (Sec. 2.2.1), 0 for the types he did not have.
"""
from __future__ import annotations

from ._base import FunctionMethod
from .helpers import quota_mixture

PREFIX = "You are a human being with all the cognitive biases and heuristics that come with it."

TYPES = {
    "self_interested": "You only care about your own pay-off.",
    "efficient": "You only care about the total pay-off of all players.",
    "inequity_averse": "You only care about fairness between players.",
    "competitive": "You only care about doing better than the other players.",
    "reciprocal": ("You only care about reciprocity: you are generous to those who act fairly, and you punish "
                   "unfair behavior even at a cost to yourself."),
    "selfish_but_fair": "You are self-interested but fair.",
}
PERSONAS = {t: f"{PREFIX} {s}" for t, s in TYPES.items()}

# edsl-test personas/homo_silicus/weights.json (gpt-oss-120b calibration; train_sse 0.0889, val_rmse 0.3149)
CALIBRATED_WEIGHTS = {"self_interested": 0.36, "efficient": 0.50, "inequity_averse": 0.0, "competitive": 0.0,
                      "reciprocal": 0.0, "selfish_but_fair": 0.14}
# Horton (2023) GPT-4o mixture, arXiv:2301.07543 Sec. 2.2.1
HORTON_WEIGHTS = {"self_interested": 0.53, "efficient": 0.37, "inequity_averse": 0.10, "competitive": 0.0,
                  "reciprocal": 0.0, "selfish_but_fair": 0.0}


def make(weights=None, name="homo_silicus"):
    """A homo_silicus method with custom mixture weights (dict over TYPES)."""
    w = dict(CALIBRATED_WEIGHTS if weights is None else weights)
    unknown = set(w) - set(TYPES)
    if unknown:
        raise ValueError(f"unknown homo_silicus types: {sorted(unknown)}")

    def _draw(n, seed=None, data_root=None):
        return [(PERSONAS[t], {"type": t}) for t in quota_mixture(list(TYPES), w, n, seed)]

    return FunctionMethod(
        name=name, draw_fn=_draw, info={"weights": w},
        description="One-sentence social-preference types, calibrated mixture with fixed quotas.",
        paper="Horton 2023 (arXiv:2301.07543); Charness & Rabin 2002 (QJE); Manning et al. 2025 (arXiv:2508.17407)",
        source="edsl-test personas/homo_silicus/__init__.py, weights.json")


METHOD = make()
