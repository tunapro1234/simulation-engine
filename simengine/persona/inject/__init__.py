"""simengine.persona.inject: persona injection methods from the literature, one module per family.

An injector renders (person, question) -> Prompt(system, user, history, meta); `compare_injectors` runs several
on the same cells with no_persona and swap controls and paired bootstrap CIs (simengine.evidence).

    import simengine.persona.inject as pi
    pi.list_injectors()                                # name, family, paper, needs, description
    inj = pi.get_injector("plain_fields", keys=["age", "region"])
    p = inj({"id": "u1", "fields": {"age": 34, "region": "north"}},
            {"id": "q1", "text": "Do you own a car?", "options": ["yes", "no"]})
    llm.complete(p.system, p.flatten())
    res = pi.compare_injectors(llm, people, questions, "answers", ["plain_fields"], dry_run=True)
    res.estimate                                       # calls and upper-bound USD before spending anything

Family modules (`inject/<family>.py`) are imported automatically (pkgutil), so adding one never edits this file.
A module that fails to import is skipped with a warning and listed in IMPORT_ERRORS.
"""
from __future__ import annotations

import importlib
import pkgutil
import warnings

from ._base import (REGISTRY, RESERVED, ROLES, Injector, MissingPersonField, NoPersona, PlainFields, Prompt, Swap,
                    answer_spec, derangement, get_injector, list_injectors, parse_option, register, swap_wrapper)
from .compare import CONTROLS, UNPARSED, CompareResult, compare_injectors, estimate_cost

IMPORT_ERRORS: dict[str, str] = {}
FAMILIES: list[str] = []

for _m in pkgutil.iter_modules(__path__):
    if _m.name.startswith("_") or _m.name == "compare":
        continue
    try:
        importlib.import_module(f"{__name__}.{_m.name}")
        FAMILIES.append(_m.name)
    except Exception as _e:  # noqa: BLE001 - one broken family must not break the zoo
        IMPORT_ERRORS[_m.name] = f"{type(_e).__name__}: {_e}"
        warnings.warn(f"simengine.persona.inject: skipped family module {_m.name!r} ({IMPORT_ERRORS[_m.name]})",
                      RuntimeWarning, stacklevel=2)

__all__ = ["Prompt", "Injector", "MissingPersonField", "REGISTRY", "register", "list_injectors", "get_injector",
           "NoPersona", "PlainFields", "Swap", "swap_wrapper", "derangement", "answer_spec", "parse_option",
           "compare_injectors", "CompareResult", "estimate_cost", "UNPARSED", "CONTROLS", "IMPORT_ERRORS",
           "FAMILIES", "ROLES", "RESERVED"]
