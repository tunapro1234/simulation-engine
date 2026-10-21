"""Shared contracts between simengine modules.

Every subpackage depends only on this file (plus numpy/pandas) for cross-module types,
so modules can be built, tested and replaced independently.

EvalFrame convention (a long pandas DataFrame, one row per evaluated unit):
    truth            required. label (0/1, class), number, or distribution (list/array/JSON)
    pred_<method>    one or more. probability / label / probability vector / number / score
    segment, seg_*   optional. segment error analysis
    group            optional. cluster key for bootstrap (e.g. person id)
    item             optional. question / test id (per-item baselines, A/B tests)
    split            optional. 'train' | 'test'; metrics use test rows only
    time             optional. for temporal leakage checks
    unit             optional. row id
    retest           optional. second human measurement of the same unit (human ceiling)
The Turkish column names used by the original research code (gercek, tahmin_<m>, bolum=egitim/test,
grup, madde, zaman, birim, tekrar) are accepted everywhere as aliases; see COLUMN_ALIASES.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import pandas as pd

TASK_KINDS = ("binary", "multiclass", "distribution", "continuous", "ab")
PRED_PREFIX = "pred_"

COLUMN_ALIASES = {
    "gercek": "truth",
    "bolum": "split",
    "grup": "group",
    "madde": "item",
    "zaman": "time",
    "birim": "unit",
    "tekrar": "retest",
}
SPLIT_ALIASES = {"egitim": "train", "test": "test"}
PRED_ALIASES = ("tahmin_",)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Turkish research-code column names mapped to the English convention."""
    out = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    ren = {}
    for c in out.columns:
        for p in PRED_ALIASES:
            if isinstance(c, str) and c.startswith(p):
                ren[c] = PRED_PREFIX + c[len(p):]
    out = out.rename(columns=ren)
    if "split" in out.columns:
        out["split"] = out["split"].map(lambda s: SPLIT_ALIASES.get(s, s))
    return out


# ---------------------------------------------------------------- LLM

@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False          # served from simengine's local disk cache (no API call)
    raw: dict | None = None


@runtime_checkable
class LLM(Protocol):
    """Anything that turns (system, user) into text. Implementations: simengine.llm.Claude,
    simengine.llm.ClaudeCode (`claude -p`), simengine.llm.FakeLLM (deterministic, offline)."""

    model: str

    def complete(self, system: str, user: str, *, max_tokens: int = 256, **kw: Any) -> LLMResponse: ...

    def map(self, prompts: Sequence[tuple[str, str]], *, max_tokens: int = 256, parallel: int = 4,
            **kw: Any) -> list[LLMResponse]: ...


# ---------------------------------------------------------------- tasks / benchmark

@dataclass
class TaskData:
    """A task instance after loading and (temporal) splitting. Frames are task specific;
    `meta` documents split rule, n, source, license."""
    train: pd.DataFrame
    test: pd.DataFrame
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Task(Protocol):
    name: str
    kind: str                     # one of TASK_KINDS

    def load(self, data_root: str | None = None) -> TaskData: ...

    def baselines(self, data: TaskData) -> dict[str, Any]: ...

    def frame(self, data: TaskData, preds: dict[str, Any]) -> pd.DataFrame:
        """Build an EvalFrame (test rows) from {method: predictions aligned with data.test}."""
        ...


# ---------------------------------------------------------------- research loop

# A runner takes a config dict and returns {"metrics": {...}, "per_unit": DataFrame|None, ...}.
Runner = Callable[[dict], dict]
