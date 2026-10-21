"""simengine: population simulation with LLM personas, agent-based models and an honest evidence pipeline.

    import simengine as se
    llm = se.FakeLLM(se.sim.schema_responder())          # or se.llm.Claude(model="claude-haiku-5-5", budget_usd=5)
    panel = se.persona.from_table(df, id_col="user_id", forbidden=["target"])
    res = se.sim.survey(panel, "I would try this product", llm=llm)
    frame = se.sim.survey_frame(res, truth={"q1": {...}})
    report = se.evidence.evaluate(frame, ours="score", task="continuous")
    se.evidence.card(report, "card.md"); se.evidence.dashboard([report], "pano.html")

Subpackages are imported lazily on first attribute access, so `import simengine` stays fast and optional
dependencies (anthropic, lightgbm, sentence-transformers) are only needed by the parts that use them.

Subpackages: llm, persona, sim, methods, abm, evidence, research, bench (plus `types` and `cli`).
What works and what does not, with measured numbers: README.md, docs/RESULTS.md, docs/METHODS.md.
"""
from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

SUBPACKAGES = ("llm", "persona", "sim", "methods", "abm", "evidence", "research", "bench", "types", "cli")

# Most used entry points, re-exported lazily: name -> (subpackage, attribute)
_ENTRY_POINTS = {
    # llm
    "Claude": ("llm", "Claude"),
    "ClaudeCode": ("llm", "ClaudeCode"),
    "FakeLLM": ("llm", "FakeLLM"),
    # persona
    "Persona": ("persona", "Persona"),
    "AudienceSpec": ("persona", "AudienceSpec"),
    "from_table": ("persona", "from_table"),
    "from_history": ("persona", "from_history"),
    # sim
    "survey": ("sim", "survey"),
    "ab_test": ("sim", "ab_test"),
    "segment_distribution": ("sim", "segment_distribution"),
    "predict_people": ("sim", "predict_people"),
    # evidence
    "evaluate": ("evidence", "evaluate"),
    "card": ("evidence", "card"),
    "dashboard": ("evidence", "dashboard"),
    "summary": ("evidence", "summary"),
    "worth_humans": ("evidence", "worth_humans"),
    # research
    "Loop": ("research", "Loop"),
    "Space": ("research", "Space"),
    "Prereg": ("research", "Prereg"),
    "lock_prereg": ("research", "lock_prereg"),
    "seal": ("research", "seal"),
    # types
    "LLM": ("types", "LLM"),
    "LLMResponse": ("types", "LLMResponse"),
    "TaskData": ("types", "TaskData"),
    "normalize_frame": ("types", "normalize_frame"),
}

__all__ = ["__version__", *SUBPACKAGES, *_ENTRY_POINTS]


def __getattr__(name: str) -> Any:
    if name in SUBPACKAGES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    if name in _ENTRY_POINTS:
        sub, attr = _ENTRY_POINTS[name]
        val = getattr(importlib.import_module(f"{__name__}.{sub}"), attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
