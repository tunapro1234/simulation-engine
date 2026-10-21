"""simengine.bench: task registry, local-dataset adapters with temporal / wave splits, baselines and a leaderboard.

    from simengine import bench
    bench.list()                                   # registered task names
    task = bench.get("gss", holdout_panel="panel2006")
    data = task.load("/path/to/catalogue")         # or set $SIMENGINE_DATA
    frame = task.run(methods={"mine": lambda data: ...}, data=data)   # EvalFrame (truth, pred_<method>, ...)
    task.score(frame, "mine")                      # task metric with a bootstrap CI
    table = bench.run(["gss", "upworthy"], methods=...)   # long table; bench.to_markdown(table)

Expected file layout per task: `bench.get(name).layout` (and docs/modules/bench.md).
"""
from .base import (ENV_DATA, BenchTask, DataNotFound, Metric, get, list_tasks, methods_in, register,
                   resolve_root)
from .leaderboard import run, to_markdown
from .tasks import (BankMarketing, BeeBenchB01Prior, EconGamesB01, ErtanWTP, GSSWave, OpenEcommerce, OpinionQAGroups,
                    SimBench, Twin2K500, Upworthy)

list = list_tasks  # noqa: A001  (bench.list() per the module contract)

__all__ = ["ENV_DATA", "BenchTask", "DataNotFound", "Metric", "get", "list", "list_tasks", "methods_in", "register",
           "resolve_root", "run", "to_markdown", "BankMarketing", "BeeBenchB01Prior", "EconGamesB01", "ErtanWTP",
           "GSSWave", "OpenEcommerce", "OpinionQAGroups", "SimBench", "Twin2K500", "Upworthy"]
