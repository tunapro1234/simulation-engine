"""Control: no persona at all, the bare model (edsl-test personas/baseline)."""
from __future__ import annotations

from ._base import FunctionMethod


def _draw(n, seed=None, data_root=None):
    return [("", {}) for _ in range(n)]


METHOD = FunctionMethod(
    name="baseline", draw_fn=_draw,
    description="No persona (empty text): the bare-model control every persona method must beat.",
    source="edsl-test personas/baseline/__init__.py")
