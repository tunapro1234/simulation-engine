"""beebench_b01_prior (continuous): the econ_games_b01 task with the prior BeeBench b01 runs as fixed reference arms.

Same test rows, split, scale and metrics as `econ_games_b01`. `load()` also reads the b01 results table (10,920
rows on the original run: 7 persona methods x 14 games x 10 agents x 3 runs, one model), so the leaderboard and
the evidence card can be produced offline with no API spend. The table is not shipped: point `data_root` (or
$SIMENGINE_DATA) at a directory holding `beebench_b01/results.csv`, or at the directory that holds
`results.csv` itself.

Every persona method in the file becomes a reference arm named `b01_<persona_method>`, except the no-persona
method (`bare_method`, default 'baseline'), which is `bare_model`. They are added next to the LLM-free
`rational_corner` and `midpoint` baselines, so `bench.run` lists them as baselines; they are prior results, not
fitted to this task. All moments are recomputed from the rows by `simengine.games` (valid rows only), so every
number this task reports is simengine's own computation, not a figure from the b01 repository.
"""
from __future__ import annotations

import os
import re

import pandas as pd

from ...types import TaskData
from ..base import DataNotFound, need, register, resolve_root
from .econ_games_b01 import BARE_METHOD, EconGamesB01, _align, moment_values


@register()
class BeeBenchB01Prior(EconGamesB01):
    """econ_games_b01 plus the prior b01 results (`<root>/beebench_b01/results.csv`) as reference arms."""

    name = "beebench_b01_prior"
    data_subdir = "beebench_b01"
    accepted_options = {
        "split": '"holdout" | "all" (default "holdout")',
        "scale": '"share" | "all" (default "share")',
        "bare_method": 'persona_method of the no-persona arm (default "baseline")',
        "model": "model to keep when the file holds several",
        "methods": "persona methods to keep: a list, or 'big5|twin2k' on the command line (default: all)",
    }
    layout = """
    <root>/beebench_b01/results.csv   BeeBench b01 tidy results (model, persona_method, agent_id, game, role,
                                      condition, wording_id, run, decision, belief, valid, raw_len, [timestamp,]
                                      cost_usd); not shipped
    """
    source = ("edsl-test benchmark/beebench/b01/results.csv (prior gpt-oss-120b run); moments recomputed by "
              "simengine.games")

    def load(self, data_root: str | None = None) -> TaskData:
        root = resolve_root(data_root, self.data_subdir)
        data = self._load(root)
        data.meta.setdefault("task", self.name)
        data.meta.setdefault("kind", self.kind)
        data.meta.setdefault("options", dict(self.options))
        return data

    def _load(self, root: str | None) -> TaskData:
        if root is None:
            raise DataNotFound("beebench_b01_prior needs data_root (see layout)")
        import simengine.games as gm
        res = gm.read_results(need(os.path.join(root, "results.csv")))
        keep = self.options.get("methods")
        if keep is not None:
            if isinstance(keep, str):             # CLI form: methods=big5|twin2k
                keep = [m for m in re.split(r"[|+,\s]+", keep) if m]
            res = res[res["persona_method"].astype(str).isin([str(m) for m in keep])]
        if self.options.get("model") is not None:
            res = res[res["model"].astype(str) == str(self.options["model"])]
        data = super()._load(None)
        n_valid = int(res["valid"].sum())
        data.meta.update({"results": res, "n_rows": int(len(res)), "n_invalid": int(len(res) - n_valid),
                          "models": sorted(res["model"].astype(str).unique()),
                          "persona_methods": sorted(res["persona_method"].astype(str).unique()),
                          "source": "prior BeeBench b01 run (edsl-test); human moments inlined in simengine.games"})
        return data

    def _prior_values(self, data: TaskData) -> dict[str, dict[str, float]]:
        res: pd.DataFrame = data.meta["results"]
        if res.empty:
            return {}
        vals = moment_values(res, data.meta["refs"], model=self.options.get("model"))
        bm = self.options.get("bare_method", BARE_METHOD)
        return {("bare_model" if pm == bm else f"b01_{pm}"): v for pm, v in vals.items()}

    def baselines(self, data: TaskData) -> dict:
        out = super().baselines(data)
        cache = data.meta.get("_prior_values")
        if cache is None:
            cache = data.meta["_prior_values"] = self._prior_values(data)
        out.update({k: _align(v, data.test["unit"]) for k, v in cache.items()})
        return out

    def _bare_results(self):
        return None            # the bare model comes from the prior file (see baselines)

    def results_frame(self, results=None, *, data: TaskData | None = None, baselines: bool = True) -> pd.DataFrame:
        """Train + test EvalFrame with the LLM-free baselines, the prior b01 arms and (optionally) new results."""
        data = data if data is not None else self.load()
        fr = super().results_frame(results, data=data, baselines=baselines)
        if baselines:
            units = fr["unit"]
            for k, v in self._prior_values(data).items():
                fr.insert(len(fr.columns) - 1, f"pred_{k}", _align(v, units))
        return fr


__all__ = ["BeeBenchB01Prior"]
