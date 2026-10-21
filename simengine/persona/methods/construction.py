"""General Social Agents construction: numeric trait dials (edsl-test personas/construction).

Manning & Horton, "General Social Agents", arXiv:2508.17407, Appendix A. One fixed template (Appendix A.2,
p.49, verbatim) with three 1-10 dials from the Charness & Rabin (2002) social-preference model: efficiency,
self-interest, inequity aversion. The population is a balanced mixture over K dial vectors; PAPER_DIALS is the
paper's published optimum for GPT-4o (p.50). The source repo's local re-calibration file (dials.json) was never
written, so the paper's vectors are what it ran with. Its replication run on GPT-4o got MAE 0.347 against the
paper's 0.20 (edsl-test personas/construction/replication.json): it beat the bare model (0.398) but not the
paper.
"""
from __future__ import annotations

from ._base import FunctionMethod
from .helpers import as_rng

TEMPLATE = (
    "On a scale from 1 to 10, your efficiency level is: {eff}. 10 means you "
    "strongly prioritize maximizing combined payoffs, and 1 means you don't care. "
    "On a scale from 1 to 10, your self-interest level is: {self}. 10 means you "
    "strongly prioritize your own payoffs, and 1 means you don't care. "
    "On a scale from 1 to 10, your inequity aversion level is: {ineq}. 10 means "
    "you strongly prioritize fairness between players, and 1 means you don't care."
)

# phi* from the paper (p.50; optimized for GPT-4o); order (efficiency, self-interest, inequity aversion)
PAPER_DIALS = [(7, 10, 10), (3, 1, 3), (1, 10, 2)]


def render(d):
    eff, self_, ineq = d
    return TEMPLATE.format(eff=eff, self=self_, ineq=ineq)


def make(dials=None, name="construction"):
    """dials: list of (eff, self, ineq) vectors (default PAPER_DIALS)."""
    pool = [tuple(int(x) for x in d) for d in (dials or PAPER_DIALS)]
    if not pool or any(len(d) != 3 for d in pool):
        raise ValueError("dials must be a non-empty list of (eff, self, ineq) triples")

    def _draw(n, seed=None, data_root=None):
        rng = as_rng(seed)
        chosen = pool * (n // len(pool)) + rng.sample(pool, n % len(pool))
        rng.shuffle(chosen)
        return [(render(d), {"efficiency": d[0], "self_interest": d[1], "inequity_aversion": d[2]})
                for d in chosen]

    return FunctionMethod(
        name=name, draw_fn=_draw, info={"dials": pool},
        description="Fixed template with three 1-10 social-preference dials; balanced mixture of dial vectors.",
        paper="Manning & Horton 2025, General Social Agents (arXiv:2508.17407), App. A",
        source="edsl-test personas/construction/__init__.py")


METHOD = make()
