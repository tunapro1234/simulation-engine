"""Task base class, task registry and data-root resolution for `simengine.bench`.

A bench task implements the `simengine.types.Task` protocol:

    data  = task.load(data_root)          # TaskData(train, test, meta), split already applied
    preds = task.baselines(data)          # {name: predictions aligned with data.test rows}
    frame = task.frame(data, preds)       # EvalFrame (test rows; truth, pred_<method>, group, item, ...)

and adds `score(frame, method)` (the task's own metric with a bootstrap CI, ported from the research code)
and `run(methods)` (load + baselines + your methods -> EvalFrame).

Predictions are always aligned with `data.test` rows: a 1-D array for binary / continuous / ab tasks, a list of
probability vectors for distribution tasks.
"""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ..types import PRED_PREFIX, TASK_KINDS, TaskData

ENV_DATA = "SIMENGINE_DATA"


class DataNotFound(FileNotFoundError):
    """Raised when a task's files are not under data_root / $SIMENGINE_DATA."""


def resolve_root(data_root: str | os.PathLike | None, subdir: str | None = None) -> str:
    """data_root argument, else $SIMENGINE_DATA. Accepts either the catalogue root (containing `subdir`) or the
    task's own directory. Raises DataNotFound if nothing usable exists."""
    root = data_root if data_root is not None else os.environ.get(ENV_DATA)
    if not root:
        raise DataNotFound(f"no data_root given and ${ENV_DATA} is not set")
    root = os.fspath(root)
    if subdir and os.path.isdir(os.path.join(root, subdir)):
        return os.path.join(root, subdir)
    if os.path.isdir(root):
        return root
    raise DataNotFound(f"data root does not exist: {root}")


def need(path: str) -> str:
    if not os.path.exists(path):
        raise DataNotFound(f"missing file: {path}")
    return path


@dataclass
class Metric:
    """name, direction and scorer. fn(task, frame, pred_col, B, seed) -> {value, lo, hi, n}."""
    name: str
    higher_is_better: bool
    fn: Callable[..., dict]
    description: str = ""


def as_pred_col(method: str) -> str:
    return method if method.startswith(PRED_PREFIX) else PRED_PREFIX + method


def methods_in(frame: pd.DataFrame) -> list[str]:
    return [c[len(PRED_PREFIX):] for c in frame.columns if isinstance(c, str) and c.startswith(PRED_PREFIX)]


class BenchTask:
    """Base class. Subclasses set name, kind, data_subdir, metrics, layout (doc string of expected files) and
    implement `_load(root)`, `baselines(data)` and `_frame(data)` (the method-independent EvalFrame columns)."""

    name: str = ""
    kind: str = ""
    data_subdir: str | None = None
    layout: str = ""
    source: str = ""
    metrics: tuple[Metric, ...] = ()

    def __init__(self, **options: Any):
        self.options = options

    # ------------------------------------------------------------------ protocol
    def load(self, data_root: str | None = None) -> TaskData:
        data = self._load(resolve_root(data_root, self.data_subdir))
        data.meta.setdefault("task", self.name)
        data.meta.setdefault("kind", self.kind)
        data.meta.setdefault("options", dict(self.options))
        return data

    def baselines(self, data: TaskData) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def frame(self, data: TaskData, preds: dict[str, Any]) -> pd.DataFrame:
        base = self._frame(data).reset_index(drop=True)
        n = len(base)
        for m, p in preds.items():
            col = as_pred_col(m)
            if self.kind == "distribution":
                vals = [np.asarray(x, float) for x in p]
                if len(vals) != n:
                    raise ValueError(f"{m}: {len(vals)} predictions for {n} test rows")
                base[col] = pd.Series(vals, dtype=object)
            else:
                v = np.asarray(p, float).ravel()
                if len(v) != n:
                    raise ValueError(f"{m}: {len(v)} predictions for {n} test rows")
                base[col] = v
        if "split" not in base.columns:
            base["split"] = "test"
        return base

    def _load(self, root: str) -> TaskData:  # pragma: no cover - abstract
        raise NotImplementedError

    def _frame(self, data: TaskData) -> pd.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------ scoring
    @property
    def primary(self) -> Metric:
        return self.metrics[0]

    def metric(self, name: str | None) -> Metric:
        if name is None:
            return self.primary
        for m in self.metrics:
            if m.name == name:
                return m
        raise KeyError(f"{self.name}: unknown metric {name!r}; have {[m.name for m in self.metrics]}")

    def score(self, frame: pd.DataFrame, method: str, metric: str | None = None, *, B: int = 1000,
              seed: int = 42) -> dict:
        """{metric, value, lo, hi, n, higher_is_better} for one method column of an EvalFrame."""
        m = self.metric(metric)
        col = as_pred_col(method)
        if col not in frame.columns:
            raise KeyError(f"{col} not in frame")
        test = frame[frame["split"] == "test"] if "split" in frame.columns else frame
        r = m.fn(self, test.reset_index(drop=True), col, B, seed)
        return {"metric": m.name, "higher_is_better": m.higher_is_better, **r}

    def score_all(self, frame: pd.DataFrame, methods: Iterable[str] | None = None, *, B: int = 1000,
                  seed: int = 42) -> pd.DataFrame:
        rows = []
        for meth in methods or methods_in(frame):
            for i, m in enumerate(self.metrics):
                rows.append({"task": self.name, "method": meth, "primary": i == 0,
                             **self.score(frame, meth, m.name, B=B, seed=seed)})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ convenience
    def predict(self, method: Any, data: TaskData) -> Any:
        """Call a user method: callable(data) or callable(task, data); or an object with .predict(data)."""
        f = method.predict if hasattr(method, "predict") and not callable(method) else method
        if not callable(f):
            raise TypeError(f"method {method!r} is not callable")
        try:
            n = len([p for p in inspect.signature(f).parameters.values()
                     if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
        except (TypeError, ValueError):
            n = 1
        return f(self, data) if n >= 2 else f(data)

    def run(self, methods: Any = None, data_root: str | None = None, *, data: TaskData | None = None,
            baselines: bool | Iterable[str] = True) -> pd.DataFrame:
        """Load (unless data is given), compute baselines and the given methods, return the EvalFrame.
        methods: None, {name: callable}, list of callables (named by __name__) or baseline names."""
        data = data if data is not None else self.load(data_root)
        preds: dict[str, Any] = {}
        base = self.baselines(data) if baselines else {}
        if baselines is not True and baselines:
            keep = set(baselines)
            base = {k: v for k, v in base.items() if k in keep}
        preds.update(base)
        for name, meth in normalize_methods(methods).items():
            if isinstance(meth, str):
                if meth not in base:
                    base = self.baselines(data)
                    if meth not in base:
                        raise KeyError(f"{self.name}: no baseline {meth!r}")
                preds[name] = base[meth]
                continue
            preds[name] = self.predict(meth, data)
        return self.frame(data, preds)

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind, "data_subdir": self.data_subdir, "layout": self.layout.strip(),
                "source": self.source.strip(), "metrics": [m.name for m in self.metrics], "options": dict(self.options)}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, kind={self.kind!r}, options={self.options})"


def normalize_methods(methods: Any) -> dict[str, Any]:
    if methods is None:
        return {}
    if isinstance(methods, dict):
        return dict(methods)
    if isinstance(methods, str) or callable(methods):
        methods = [methods]
    out = {}
    for m in methods:
        if isinstance(m, str):
            out[m] = m
        else:
            out[getattr(m, "name", None) or getattr(m, "__name__", None) or type(m).__name__] = m
    return out


# ---------------------------------------------------------------------- registry
_REGISTRY: dict[str, Callable[..., BenchTask]] = {}


def register(name: str | None = None):
    """Class decorator / function: register a task factory under `name` (default: cls.name)."""
    def deco(cls):
        key = name or cls.name
        if not key:
            raise ValueError("task needs a name")
        if getattr(cls, "kind", None) and cls.kind not in TASK_KINDS:
            raise ValueError(f"{key}: kind {cls.kind!r} not in {TASK_KINDS}")
        _REGISTRY[key] = cls
        return cls
    return deco


def get(name: str, **options: Any) -> BenchTask:
    """Instantiate a registered task, e.g. get("gss", holdout_panel="panel2006")."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown task {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**options)


def list_tasks() -> list[str]:
    return sorted(_REGISTRY)
