"""simengine.bench: task registry, local-dataset adapters with temporal / wave splits, baselines and a leaderboard.

    from simengine import bench
    bench.list()                                   # registered task names (26 built-in)
    task = bench.get("gss", holdout_panel="panel2006")
    data = task.load("/path/to/catalogue")         # or set $SIMENGINE_DATA
    frame = task.run(methods={"mine": lambda data: ...}, data=data)   # EvalFrame (truth, pred_<method>, ...)
    task.score(frame, "mine")                      # task metric with a bootstrap CI
    table = bench.run(["gss", "upworthy"], methods=...)   # long table; bench.to_markdown(table)
    csv = bench.from_csv("train.csv", target="y", group_col="user")   # any labelled table as a task (csvtask.py)

    bench.get("movielens", target="next_item")     # also: instacart, m5, retailrocket_{purchase,next}, choices13k,
    bench.get("cpc18")                             # socsci210, subpop_opinionqa, global_opinionqa, swissmetro,
    bench.get("hillstrom_uplift", arm="womens")    # {hillstrom,criteo,x5}_uplift, webshop_{next_action,buy}

Expected file layout per task: `bench.get(name).layout` (and docs/modules/bench.md, docs/modules/bench/<name>.md).
Module-level helpers of a task (metrics, readers, persona text) stay in `simengine.bench.tasks.<module>`.
"""
from .base import (ENV_DATA, BenchTask, DataNotFound, Metric, get, list_tasks, methods_in, register,
                   resolve_root)
from .csvtask import CsvTask, from_csv
from .leaderboard import run, to_markdown
from .tasks import (CPC18, BankMarketing, BeeBenchB01Prior, Choices13k, CriteoUplift, EconGamesB01, ErtanWTP, GSSWave,
                    GlobalOpinionQA, HillstromUplift, Instacart, M5, MovieLens1M, OpenEcommerce, OpinionQAGroups,
                    RetailRocketNext, RetailRocketPurchase, SimBench, SocSci210, SubPOPOpinionQA, SwissmetroModeChoice,
                    Twin2K500, Upworthy, WebShopBuy, WebShopNextAction, X5Uplift)

list = list_tasks  # noqa: A001  (bench.list() per the module contract)

__all__ = ["ENV_DATA", "BenchTask", "CsvTask", "DataNotFound", "Metric", "from_csv", "get", "list", "list_tasks",
           "methods_in", "register", "resolve_root", "run", "to_markdown", "BankMarketing", "BeeBenchB01Prior",
           "CPC18", "Choices13k", "CriteoUplift", "EconGamesB01", "ErtanWTP", "GSSWave", "GlobalOpinionQA",
           "HillstromUplift", "Instacart", "M5", "MovieLens1M", "OpenEcommerce", "OpinionQAGroups", "RetailRocketNext",
           "RetailRocketPurchase", "SimBench", "SocSci210", "SubPOPOpinionQA", "SwissmetroModeChoice", "Twin2K500",
           "Upworthy", "WebShopBuy", "WebShopNextAction", "X5Uplift"]
