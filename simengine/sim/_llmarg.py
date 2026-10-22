"""`llm` argument resolution for entry points that took the LLM first and positionally.

survey / ab_test / ab_many / strategy / roundtable take `llm=` keyword-only. The older segment_distribution,
persona_distribution and ask_constrained took it first. Both forms are accepted there:

    def segment_distribution(*args, llm=None, segments=MISSING, questions=MISSING, ...):
        llm, (segments, questions) = resolve_llm("segment_distribution", args, llm,
                                                 {"segments": segments, "questions": questions})

    segment_distribution(llm, segs, qs)         # legacy, positional
    segment_distribution(segs, qs, llm=llm)     # keyword, like survey(...)

Neither or both given -> TypeError.
"""
from __future__ import annotations

from typing import Any, Mapping


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING: Any = _Missing()


def _is_llm(x: Any) -> bool:
    return callable(getattr(x, "complete", None))


def resolve_llm(fn: str, args: tuple, llm: Any, params: Mapping[str, Any]) -> tuple[Any, list]:
    """Split `args` into (llm, [values of params in order]). `params` maps each required parameter name to its
    keyword value (MISSING when not passed by keyword); positional args fill them in order."""
    names = list(params)
    args = list(args)
    if llm is None:
        if not args:
            raise TypeError(f"{fn}() needs an llm: pass it first (legacy) or as llm=")
        llm = args.pop(0)
    elif args and _is_llm(args[0]):
        raise TypeError(f"{fn}() got the llm both positionally and as llm=; pass it one way")
    if len(args) > len(names):
        raise TypeError(f"{fn}() takes {len(names)} positional arguments after the llm "
                        f"({', '.join(names)}) but {len(args)} were given")
    out = []
    for i, name in enumerate(names):
        pos = args[i] if i < len(args) else MISSING
        kw = params[name]
        if pos is not MISSING and kw is not MISSING:
            raise TypeError(f"{fn}() got multiple values for argument '{name}'")
        v = pos if pos is not MISSING else kw
        if v is MISSING:
            raise TypeError(f"{fn}() missing required argument: '{name}'")
        out.append(v)
    return llm, out
