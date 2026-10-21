"""ertan_wtp (continuous): human-likeness of a multi-period who-to-punish society, criterion by criterion.

The 9 criteria of `simengine.games.wtp_scorecard` (port of edsl-test `benchmark/score_wtp.py`), each scored 0-1
against the Ertan, Page & Putterman (2005 WP, EER 2009) targets inlined from edsl-test
`benchmark/human_targets.json` with printed-page cites (`games.ERTAN_TARGETS`, `games.WTP_PARAMS`). One test row
per criterion: truth = 1.0, the full human-likeness credit (each criterion's scoring rule already encodes its
human target, quoted in the `target` column of `data.test`); a method predicts the criterion score of its runs,
averaged over runs. No data is needed; `data_root` is ignored.

Criteria and weights (3 institutional/treatment, 2 behavioural mechanics, 1 distributional shape) are
`games.CRITERIA`. A criterion a run cannot show (single-regime run, punishment never legal, no punishment paid,
no punish_high record) is N/A: it is excluded from the index, never zero-filled, and a criterion that is N/A in
every run stays NaN (it lowers `coverage`).

Metrics:
  index     (primary, higher is better) weighted mean criterion score over the observed criteria, the
            scorecard's own index (equal to `wtp_scorecard(run)["index"]` for one run).
  wmae      weighted mean |1 - score| (lower is better).
  coverage  share of the 9 criteria observed.
The CI resamples criteria (with their weights): it shows how much the index leans on single criteria, not
sampling uncertainty across groups or seeds. For that, run several seeds and look at `scorecards(runs)`.

Methods: a callable returning the 9 criterion scores aligned with `data.test`, or the run output itself as the
prediction: a structured run dict (the wtp_scorecard input), the dict that
`simengine.abm.society.run_who_to_punish` returns (every group is scored), or a list of either. Helpers:
`runs_method(runs)`, `society_method(llm, personas, seeds=...)`, `group_to_run(group)`, `scorecards(runs)`.

Baselines (LLM-free synthetic runs, scored by the same scorecard):
  rational_corner  everyone contributes 0 in every period, every ballot item fails, no punishment.
  full_cooperation everyone contributes the full endowment in every period and punish-low passes every vote
                   (nobody is below average, so nobody is punished).
Both use the Ertan design shape of the simengine society port: 4 players, endowment 10, `votes` votes x
`periods` periods (defaults 2 x 3, options).
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, Metric, register

FAMILY = {3: "institutional", 2: "mechanics", 1: "distribution"}

# What each criterion is scored against (games.ERTAN_TARGETS / WTP_PARAMS; printed pages of the 2005 WP).
TARGETS = {
    "never_punish_high": "no group vote ever allowed punishing high contributors: 0 of 160 (Result 1, p.10)",
    "final_vote_punish_low": "final group vote punish-low in 35 of 40 groups, 0.875 (Figure 2 reading p.11)",
    "no_dismantling": "groups move toward punish-low, not away: first vote punish-low 16/40, final 35/40 "
                      "(Figure 2 reading p.11)",
    "treatment_effect": "contributions higher under punish-low than with no punishment (Figure 3 p.12: about 0.95 "
                        "of endowment in the punish-low steady state); full credit at +2 tokens (scorer choice)",
    "period1_level": "period-1 contributions 0.69-0.77 of the endowment (Figure 3 reading p.12)",
    "sustain_under_punish_low": "flat or rising under punish-low (Figure 3 p.12: 0.95 steady, 0.82 in period 30); "
                                "flat = 1, -2 tokens per period = 0",
    "punishment_usage": "$0.97-1.22 of punishment received per $1 below average under punish-low (Table 4 p.17); "
                        "full credit for a ratio in 0.5-1.5",
    "punishment_targeting": "punishment aimed at below-average contributors: perverse share at most 0.28 (p.19), "
                            "about 0.20 in the literature (p.1-2)",
    "heterogeneity": "humans differ in period 1; full credit at a period-1 SD of 3 tokens (scorer choice)",
}


def _games():
    import simengine.games as gm
    return gm


# ---------------------------------------------------------------------- run normalisation

def group_to_run(group: Mapping, endowment: float | None = None) -> dict:
    """One group of `abm.society.run_who_to_punish(...)["groups"]` -> the wtp_scorecard run dict. Votes carry
    punish_high from the group's rules; invalid punishment answers (amount None) are left out of the events."""
    rules = list(group.get("rules") or [])
    if rules:
        votes = [{"punish_low": bool(r["low"]), "punish_high": bool(r["high"])} for r in rules]
    else:
        votes = [{"punish_low": bool(v)} for v in group.get("votes") or []]
    periods = [{"contrib": dict(p["contrib"]), "received": dict(p.get("received") or {}),
                "punish_low": bool(p["punish_low"])} for p in group.get("periods") or []]
    events = [{"target_contribution": e["target_contribution"], "avg": e["avg"], "amount": e["amount"]}
              for e in group.get("punishments") or [] if e.get("valid", True) and e.get("amount") is not None]
    run = {"votes": votes, "periods": periods, "punish_events": events}
    if endowment is not None:
        run["endowment"] = endowment
    return run


def as_runs(x: Any) -> list[dict]:
    """Normalise to a list of scorecard run dicts: a run dict, a run_who_to_punish output (all its groups) or a
    list of either. Groups without a vote or a completed period cannot be scored and are dropped (counted by
    `scorecards`)."""
    if isinstance(x, Mapping):
        if "groups" in x:
            endow = (x.get("design") or {}).get("endowment")
            return [group_to_run(g, endow) for g in x["groups"]]
        if "periods" in x and ("rules" in x or "punishments" in x):
            return [group_to_run(x)]
        return [dict(x)]
    out: list[dict] = []
    for item in x:
        out += as_runs(item)
    return out


def _scorable(run: Mapping) -> bool:
    return bool(run.get("votes")) and bool(run.get("periods"))


def scorecards(runs: Any, params: Mapping | None = None) -> pd.DataFrame:
    """One row per scorable run: the 9 criteria (NaN = N/A), index, n_criteria. `attrs['n_unscorable']` counts
    runs dropped for having no vote or no completed period."""
    gm = _games()
    rs = as_runs(runs)
    names = [n for n, _ in gm.CRITERIA]
    rows = []
    for i, r in enumerate(rs):
        if not _scorable(r):
            continue
        s = gm.wtp_scorecard(r, params)
        rows.append({"run": i, **{n: (np.nan if s[n] is None else float(s[n])) for n in names},
                     "index": np.nan if s["index"] is None else float(s["index"]), "n_criteria": s["n_criteria"]})
    df = pd.DataFrame(rows, columns=["run", *names, "index", "n_criteria"])
    df.attrs["n_unscorable"] = len(rs) - len(df)
    return df


def criterion_means(runs: Any, params: Mapping | None = None) -> dict[str, float]:
    """Mean score per criterion over runs where it is observed; NaN where it is N/A in every run."""
    sc = scorecards(runs, params)
    out = {}
    for n, _ in _games().CRITERIA:
        v = sc[n].to_numpy(float) if len(sc) else np.array([])
        v = v[np.isfinite(v)]
        out[n] = float(v.mean()) if len(v) else float("nan")
    return out


def is_run_output(x: Any) -> bool:
    if isinstance(x, Mapping):
        return "groups" in x or "periods" in x
    if isinstance(x, (list, tuple)) and x and all(isinstance(i, Mapping) for i in x):
        return all("groups" in i or "periods" in i for i in x)
    return False


def runs_method(runs: Any, params: Mapping | None = None, name: str = "runs") -> Callable:
    """A bench method from run output: f(data) -> criterion means aligned with data.test."""
    def f(data: TaskData):
        m = criterion_means(runs, params)
        return np.array([m[u] for u in data.test["unit"]], dtype=float)
    f.__name__ = name
    return f


def society_method(llm, personas: Any = None, *, seeds: Iterable[int] = (0, 1, 2), params: Mapping | None = None,
                   name: str = "society", **society_kw: Any) -> Callable:
    """A bench method that runs `simengine.abm.society.run_who_to_punish(llm, personas, seed=s, **society_kw)` for
    each seed and scores every group. Each seed is a separate run of the society (the edsl-test protocol was one
    group of 4 per seed, 3 seeds)."""
    def f(data: TaskData):
        from ...abm.society import run_who_to_punish
        outs = [run_who_to_punish(llm, personas, seed=int(s), **society_kw) for s in seeds]
        return runs_method(outs, params)(data)
    f.__name__ = name
    return f


# ---------------------------------------------------------------------- synthetic baseline runs

def synthetic_run(contrib: float, punish_low: bool, *, votes: int = 2, periods: int = 3, players: int = 4,
                  endowment: float = 10) -> dict:
    labels = "ABCDEFGH"[:players]
    return {"endowment": endowment,
            "votes": [{"punish_low": punish_low, "punish_high": False} for _ in range(votes)],
            "periods": [{"contrib": {L: contrib for L in labels}, "received": {L: 0.0 for L in labels},
                         "punish_low": punish_low} for _ in range(votes * periods)],
            "punish_events": []}


# ---------------------------------------------------------------------- metrics

def _weights(frame: pd.DataFrame) -> np.ndarray:
    if "weight" in frame.columns:
        return frame["weight"].to_numpy(float)
    w = dict(_games().CRITERIA)
    return np.array([w.get(u, 1) for u in frame["unit"]], dtype=float)


def _crit_metric(kind: str):
    def fn(task, frame, col, B, seed):
        y, p, w = frame["truth"].to_numpy(float), frame[col].to_numpy(float), _weights(frame)
        if kind == "coverage":
            return {"value": float(np.isfinite(p).mean()) if len(p) else float("nan"), "lo": float("nan"),
                    "hi": float("nan"), "n": int(len(p))}
        ok = np.isfinite(p) & np.isfinite(y)
        y, p, w = y[ok], p[ok], w[ok]
        if not len(p):
            return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
        v = p if kind == "index" else np.abs(y - p)

        def stat(i):
            return float(np.sum(w[i] * v[i]) / np.sum(w[i]))
        point = stat(np.arange(len(v)))
        rng = np.random.default_rng(seed)
        boots = np.array([stat(rng.integers(0, len(v), len(v))) for _ in range(int(B))]) if B else np.array([])
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) >= 10 else (np.nan, np.nan))
        return {"value": point, "lo": float(lo), "hi": float(hi), "n": int(len(v))}
    return fn


METRICS = (Metric("index", True, _crit_metric("index"), "weighted mean criterion score (scorecard index)"),
           Metric("wmae", False, _crit_metric("wmae"), "weighted mean |1 - criterion score|"),
           Metric("coverage", True, _crit_metric("coverage"), "share of the 9 criteria observed"))


# ---------------------------------------------------------------------- task

@register()
class ErtanWTP(BenchTask):
    """Who-to-punish human-likeness, one row per scorecard criterion. Data-free."""

    name = "ertan_wtp"
    kind = "continuous"
    data_subdir = None
    accepted_options = {
        "params": "dict of wtp_scorecard parameter overrides (default games.WTP_PARAMS)",
        "votes": "votes in the synthetic baseline runs (default 2)",
        "periods": "periods per vote in the synthetic baseline runs (default 3)",
    }
    layout = """
    (none) the Ertan, Page & Putterman targets are inlined in simengine.games (ERTAN_TARGETS, WTP_PARAMS);
    data_root is ignored.
    """
    source = ("edsl-test benchmark/score_wtp.py + benchmark/human_targets.json (Ertan, Page & Putterman 2005 WP / "
              "EER 2009); scorecard port simengine.games.wtp")
    metrics = METRICS

    def load(self, data_root: str | None = None) -> TaskData:
        data = self._load(None)
        data.meta.setdefault("task", self.name)
        data.meta.setdefault("kind", self.kind)
        data.meta.setdefault("options", dict(self.options))
        return data

    def _load(self, root: str | None) -> TaskData:
        gm = _games()
        test = pd.DataFrame([{"unit": n, "family": FAMILY[w], "weight": w, "truth": 1.0, "target": TARGETS[n]}
                             for n, w in gm.CRITERIA])
        meta = {"split": "all criteria are test (no training rows)", "n_test": len(test),
                "params": {**gm.WTP_PARAMS, **(self.options.get("params") or {})},
                "source": "Ertan, Page & Putterman 2005 WP (EER 2009) targets via edsl-test human_targets.json"}
        return TaskData(train=test.iloc[:0].copy(), test=test, meta=meta)

    def _params(self) -> Mapping | None:
        return self.options.get("params")

    def baselines(self, data: TaskData) -> dict[str, np.ndarray]:
        kw = {"votes": int(self.options.get("votes", 2)), "periods": int(self.options.get("periods", 3))}
        out = {}
        for name, run in (("rational_corner", synthetic_run(0, False, **kw)),
                          ("full_cooperation", synthetic_run(10, True, **kw))):
            out[name] = runs_method([run], self._params())(data)
        return out

    def predict(self, method: Any, data: TaskData) -> Any:
        """Run output (a run dict, a run_who_to_punish result or a list of them) is passed through to frame()."""
        if is_run_output(method):
            return method
        return super().predict(method, data)

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        return pd.DataFrame({"unit": t["unit"].to_numpy(), "group": t["family"].to_numpy(),
                             "weight": t["weight"].to_numpy(float), "truth": t["truth"].to_numpy(float),
                             "split": "test"})

    def frame(self, data: TaskData, preds: dict[str, Any]) -> pd.DataFrame:
        """EvalFrame of the 9 criteria. A prediction may be a vector aligned with data.test or run output (a run
        dict, a run_who_to_punish result, or a list of them), which is scored here."""
        flat = {m: (runs_method(p, self._params())(data) if is_run_output(p) else p) for m, p in preds.items()}
        return super().frame(data, flat)


__all__ = ["ErtanWTP", "TARGETS", "group_to_run", "as_runs", "scorecards", "criterion_means", "runs_method",
           "society_method", "synthetic_run", "is_run_output"]
