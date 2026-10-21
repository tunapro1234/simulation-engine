"""Leaderboard runner: tasks x methods -> long table (+ markdown pivot).

    from simengine import bench
    table = bench.run(["gss", "upworthy"], methods={"mine": my_fn}, data_root="/path/to/catalogue")
    print(bench.to_markdown(table))

A method is a callable `f(data)` or `f(task, data)` returning predictions aligned with `data.test`, an object with
`.predict(data)`, or the name of a task baseline. Methods can be given per task: {"gss": {...}, "upworthy": {...}}.
Tasks whose data is missing are reported with status 'missing_data' instead of raising.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .base import BenchTask, DataNotFound, get, normalize_methods

COLUMNS = ["task", "method", "metric", "value", "lo", "hi", "n", "higher_is_better", "primary", "is_baseline", "status",
           "note"]


def _as_task(t: Any) -> BenchTask:
    if isinstance(t, BenchTask):
        return t
    if isinstance(t, str):
        return get(t)
    if isinstance(t, tuple) and len(t) == 2 and isinstance(t[0], str) and isinstance(t[1], dict):
        return get(t[0], **t[1])
    if all(hasattr(t, a) for a in ("load", "baselines", "frame", "name")):
        return t  # any object implementing the Task protocol (no scoring: needs .score)
    raise TypeError(f"not a task: {t!r}")


def _methods_for(methods: Any, name: str) -> dict:
    if isinstance(methods, dict) and name in methods and isinstance(methods[name], (dict, list, tuple)):
        return normalize_methods(methods[name])
    if isinstance(methods, dict) and methods and all(isinstance(v, (dict, list, tuple)) for v in methods.values()):
        return {}  # per-task mapping without an entry for this task
    return normalize_methods(methods)


def run(tasks: Iterable[Any], methods: Any = None, data_root: str | None = None, *, B: int = 1000, seed: int = 42,
        baselines: bool | Iterable[str] = True, metrics: str = "all", return_frames: bool = False):
    """Run every task with its baselines and the given methods; score each method with each task metric.

    tasks: names ("gss"), (name, options) tuples, or task instances. metrics: 'all' or 'primary'.
    Returns the long table (columns COLUMNS); with return_frames=True also {task: EvalFrame}."""
    rows, frames = [], {}
    if isinstance(tasks, (str, BenchTask)):
        tasks = [tasks]
    for t in tasks:
        task = _as_task(t)
        meth = _methods_for(methods, task.name)
        try:
            data = task.load(data_root)
            frame = task.run(meth, data=data, baselines=baselines)
        except DataNotFound as e:
            rows.append({"task": task.name, "status": "missing_data", "note": str(e)})
            continue
        frames[task.name] = frame
        base_names = set(frame.columns) - {f"pred_{m}" for m in meth}
        mets = task.metrics[:1] if metrics == "primary" else task.metrics
        for col in [c for c in frame.columns if isinstance(c, str) and c.startswith("pred_")]:
            m = col[len("pred_"):]
            for i, mt in enumerate(task.metrics):
                if mt not in mets:
                    continue
                try:
                    r = task.score(frame, m, mt.name, B=B, seed=seed)
                    status, note = "ok", ""
                except Exception as e:  # a metric failing on one method must not hide the others
                    r, status, note = {"metric": mt.name, "higher_is_better": mt.higher_is_better}, "error", repr(e)
                rows.append({"task": task.name, "method": m, "primary": i == 0, "is_baseline": col in base_names,
                             "status": status, "note": note, **r})
    table = pd.DataFrame(rows).reindex(columns=COLUMNS)
    return (table, frames) if return_frames else table


def _fmt(v, lo, hi, digits):
    if v is None or not np.isfinite(v):
        return "-"
    s = f"{v:.{digits}f}"
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        s += f" [{lo:.{digits}f}, {hi:.{digits}f}]"
    return s


def to_markdown(table: pd.DataFrame, *, primary_only: bool = True, digits: int = 3) -> str:
    """Markdown: one section per task, rows = methods (best first), columns = metrics, cells 'value [lo, hi]'."""
    out = []
    ok = table[table["status"] == "ok"]
    if primary_only:
        ok = ok[ok["primary"] == True]  # noqa: E712
    for task, g in ok.groupby("task", sort=False):
        mets = list(dict.fromkeys(g["metric"]))
        first = g[g["metric"] == mets[0]]
        hib = bool(first["higher_is_better"].iloc[0])
        order = first.sort_values("value", ascending=not hib)["method"].tolist()
        out.append(f"### {task}\n")
        hdr = ["method"] + [f"{m} ({'higher' if bool(g[g.metric == m].higher_is_better.iloc[0]) else 'lower'} is better)"
                            for m in mets] + ["n"]
        out.append("| " + " | ".join(hdr) + " |")
        out.append("|" + "---|" * len(hdr))
        for meth in order:
            cells = [meth + ("" if not g[(g.method == meth)].is_baseline.any() else " (baseline)")]
            n = ""
            for m in mets:
                r = g[(g.method == meth) & (g.metric == m)]
                if len(r):
                    r = r.iloc[0]
                    cells.append(_fmt(r["value"], r["lo"], r["hi"], digits))
                    n = n or str(int(r["n"])) if pd.notna(r["n"]) else n
                else:
                    cells.append("-")
            out.append("| " + " | ".join(cells + [n]) + " |")
        out.append("")
    miss = table[table["status"] == "missing_data"]
    for _, r in miss.iterrows():
        out.append(f"- {r['task']}: data not found ({r['note']})")
    return "\n".join(out).rstrip() + "\n"
