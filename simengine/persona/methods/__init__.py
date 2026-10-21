"""simengine.persona.methods: the persona method zoo (ported from edsl-test `personas/`).

Every method returns n persona texts for a seed; "" means no persona (the baseline control). Persona text goes
into the LLM call as `system`, the game or question as `user`.

    import simengine.persona.methods as pm
    pm.list_methods()                                   # name, needs_data, paper, source
    texts = pm.sample_texts("homo_silicus", n=4, seed=1)
    people = pm.method_personas("big5", n=20, seed=0)   # flat dicts -> simengine.sim runners accept them
    bank = pm.AgentBank("value_anchor"); bank.get(10)   # same agent ids across every game

Data-free: baseline, homo_silicus, big5, value_anchor, gps, construction, bfi2 (simengine.persona.bigfive).
Data-backed (files not shipped; `data_root` or $SIMENGINE_DATA, layout `<root>/persona_methods/<file>`):
demographic (gss_sample.csv), anthology (anthology_backstories.json), twin2k (twin2k_personas_sample.json).
Provenance and measured results per method: docs/modules/persona.md.
"""
from __future__ import annotations

from typing import Any

from . import anthology, baseline, big5, construction, demographic, gps, homo_silicus, twin2k, value_anchor
from ._base import DATA_SUBDIR, ENV_DATA, FunctionMethod, PersonaDataNotFound, PersonaMethod, data_path
from .bank import AgentBank
from .helpers import (RowFragmentRenderer, TraitDialSampler, cholesky, quota_counts, quota_mixture,
                      stratified_pool_sample, weighted_indices_no_replacement, weighted_rows_no_replacement)


def _bfi2_draw(n, seed=None, data_root=None):
    from ..bigfive import sample_bigfive
    return [(p["text"], p["params"]) for p in sample_bigfive(n, seed=seed)]


BFI2 = FunctionMethod(
    name="bfi2", draw_fn=_bfi2_draw,
    description="BFI-2-Expanded sentence bank (3 sentences per domain), scores from US adult norms.",
    paper="Soto & John 2017 (BFI-2); Huang et al. 2024 (BFI-2-Expanded format); validated with the Mini-IPIP "
          "(Donnellan et al. 2006)",
    source="replicant src/replicant/personalities/factory.py")

METHODS: dict[str, PersonaMethod] = {m.name: m for m in (
    baseline.METHOD, homo_silicus.METHOD, big5.METHOD, value_anchor.METHOD, gps.METHOD, construction.METHOD,
    demographic.METHOD, anthology.METHOD, twin2k.METHOD, BFI2)}


def register_method(method: PersonaMethod, replace: bool = False) -> PersonaMethod:
    """Add a method to the registry (e.g. homo_silicus.make(weights=...) under a new name)."""
    if method.name in METHODS and not replace:
        raise ValueError(f"persona method {method.name!r} already registered")
    METHODS[method.name] = method
    return method


def get_method(method: str | PersonaMethod) -> PersonaMethod:
    if not isinstance(method, str):
        if isinstance(method, PersonaMethod):
            return method
        raise TypeError("method must be a name or a PersonaMethod")
    try:
        return METHODS[method]
    except KeyError:
        raise ValueError(f"unknown persona method {method!r}; choose from {list(METHODS)}") from None


def list_methods() -> list[dict]:
    """One dict per registered method: name, needs_data, description, paper, source, data_file."""
    out = []
    for m in METHODS.values():
        d = m.describe() if hasattr(m, "describe") else {"name": m.name, "needs_data": m.needs_data}
        out.append(d)
    return out


def _draw(m: PersonaMethod, n: int, seed: Any, data_root: str | None) -> list[tuple[str, dict]]:
    draw = getattr(m, "draw", None)
    if callable(draw):
        return draw(n, seed, data_root)
    return [(t, {}) for t in m.sample(n, seed, data_root)]


def sample_texts(method: str | PersonaMethod, n: int, seed: Any = 0, data_root: str | None = None) -> list[str]:
    """n persona texts from one method ("" = no persona)."""
    return get_method(method).sample(n, seed, data_root)


def method_personas(method: str | PersonaMethod, n: int, seed: Any = 0, data_root: str | None = None) -> list[dict]:
    """n flat persona dicts: {"id": f"{method}-{seed}-{k}", "text", "persona_method", "params"}; `params`
    holds the method's per-persona parameters plus "seed".

    `simengine.sim` accepts them as flat personas (the `text` key is the persona block). A baseline persona has
    an empty text, and the sim adapter then renders the remaining scalar field ("persona_method: baseline");
    for a clean no-persona arm use the sim runners' `no_persona` control instead."""
    m = get_method(method)
    return [{"id": f"{m.name}-{seed}-{k}", "text": text, "persona_method": m.name,
             "params": {**params, "seed": seed}}
            for k, (text, params) in enumerate(_draw(m, n, seed, data_root))]


__all__ = [
    "PersonaMethod", "FunctionMethod", "PersonaDataNotFound", "METHODS", "BFI2", "list_methods", "get_method",
    "register_method", "sample_texts", "method_personas", "AgentBank", "data_path", "ENV_DATA", "DATA_SUBDIR",
    "TraitDialSampler", "cholesky", "quota_counts", "quota_mixture", "weighted_rows_no_replacement",
    "weighted_indices_no_replacement", "RowFragmentRenderer", "stratified_pool_sample",
    "baseline", "homo_silicus", "big5", "value_anchor", "gps", "construction", "demographic", "anthology",
    "twin2k",
]
