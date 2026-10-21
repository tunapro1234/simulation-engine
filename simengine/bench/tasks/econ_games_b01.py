"""econ_games_b01 (continuous): how close simulated one-shot economic-game behaviour comes to published human moments.

One test row per human-referenced moment of the `simengine.games` battery (BeeBench b01 games plus
`ertan_vignette`), e.g. `dictator:mean_give_share` (Engel 2011: 0.2835) or `who_to_punish:yes_share_low` (Ertan,
Page & Putterman: 0.641). Truth is the cited human value from `games.human_refs()`; a method predicts the moment
value its simulated agents produce. No data is needed: the human values are small published constants inlined in
`simengine.games` (each with its citation), so `load()` ignores `data_root`.

Split (option `split`): 'holdout' (default) puts the calibration-game moments in train and the holdout-game
moments (beauty contest, punishment games, who-to-punish) in test, the b01 rule that holdout games are never
used for persona tuning. 'all' tests every referenced moment (train empty).
Scale (option `scale`): 'share' (default) keeps 0-1 moments only, so the pooled error is in one unit; 'all' adds
raw-unit moments (beauty-contest mean guess, punishment points, lambda), whose errors are NOT comparable with the
shares.

Metrics: `mae` (primary, lower is better) and `rmse` over test moments with a finite prediction, with a cluster
bootstrap over games (all moments of a game move together; n = games); `coverage` = share of test moments the
method produced a finite value for. A moment a method cannot compute (no valid answers, no complete schedule)
stays NaN; it is never filled in, it lowers `coverage` instead.

Baselines (LLM-free unless an LLM or results are given):
  rational_corner  self-interested, risk-neutral payoff maximiser (dictator 0, trust 0, beauty contest 0, ...);
                   decisions in CORNER_DECISIONS, moments computed by `games.moments` on those decisions.
  midpoint         numerical answers at the midpoint of their range (the equal split where the range is a pot);
                   multiple-choice answers uniformly at random, computed exactly by enumerating every answer
                   combination across a game's conditions.
  bare_model       the no-persona model (b01 'baseline' method): from option `results` (rows whose
                   persona_method is `bare_method`, default 'baseline') and/or by playing the games with option
                   `llm` (empty personas, `bare_agents` agents x `runs` runs; the games.play cost gate applies,
                   pass play_kw={'confirm': True} to go over it). Absent when neither is given.

Methods: a callable returning predictions aligned with `data.test` rows, or (simpler) a `games.play` results
DataFrame given as the prediction itself: `frame()` turns it into moments through `games.moment_frame`. Helpers:
`results_method(results)`, `play_method(llm, personas)`, and `task.results_frame(results)` for a train+test
EvalFrame that `simengine.evidence.evaluate` takes directly.
"""
from __future__ import annotations

import itertools
import os
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from ...types import PRED_PREFIX, TaskData
from ..base import BenchTask, Metric, register
from ..stats import cluster_bootstrap

BARE_METHOD = "baseline"          # b01 name of the no-persona persona method
UNIT_COLUMNS = ["unit", "game", "moment", "truth", "scale", "citation", "holdout"]

# Self-interested, risk-neutral, payoff-maximising decision per (game, condition id). `None` key = every condition.
# eleven_twenty has no pure-strategy equilibrium; 20 is the payoff-maximising request ignoring the bonus (the
# level-0 anchor). The ultimatum proposer offers the smallest positive whole-dollar amount (subgame perfect when
# the responder accepts any positive offer). Risk-neutral Holt-Laury: Option A in rows 1-4, B from row 5
# (Holt & Laury 2002 pp.1645-46, quoted in the GAME references). Loss gambles: accept iff the expected value is
# positive (gain > loss; indifference at gain = loss is resolved as Reject). Who-to-punish: punishment is costly
# and a selfish group never uses it, so every ballot item gets No. time_choices and ertan_vignette carry no
# numeric human reference and are left out.
CORNER_DECISIONS: dict[str, Any] = {
    "dictator": {None: 0},
    "eleven_twenty": {None: 20},
    "holt_laury": {f"row{k}": ("Option A" if k <= 4 else "Option B") for k in range(1, 11)},
    "loss_gambles": "ev_positive",
    "public_goods_strategy": {None: 0},
    "trust_sender": {None: 0},
    "trust_returner": {None: 0},
    "ultimatum_proposer": {None: 1},
    "ultimatum_responder": {None: "Accept"},
    "beauty_contest": {None: 0},
    "pg_punishment": {None: 0},
    "third_party_punishment": {None: 0},
    "who_to_punish": {None: "No"},
}


def _games():
    import simengine.games as gm   # lazy: keeps `import simengine.bench` light
    return gm


# ---------------------------------------------------------------------- synthetic decision tables

def _rows(game: str, agent: str, decisions: Mapping[str, Any], method: str) -> list[dict]:
    gm = _games()
    g = gm.GAMES[game]
    return [{"model": "synthetic", "persona_method": method, "agent_id": agent, "game": game, "role": g["role"],
             "condition": str(c), "wording_id": 0, "run": 0, "decision": d, "belief": None, "valid": True,
             "raw_len": 0, "cost_usd": 0.0} for c, d in decisions.items()]


def corner_results(games=None) -> pd.DataFrame:
    """games.play-shaped rows of the rational-corner decisions (one synthetic agent per game)."""
    gm = _games()
    out = []
    for game, rule in CORNER_DECISIONS.items():
        if games is not None and game not in games:
            continue
        conds = [str(c["id"]) for c in gm.GAMES[game]["conditions"]]
        if rule == "ev_positive":
            dec = {str(c["id"]): ("Accept" if float(c["gain"]) > float(c["loss"]) else "Reject")
                   for c in gm.GAMES[game]["conditions"]}
        else:
            dec = {c: rule[c] if c in rule else rule[None] for c in conds}
        out += _rows(game, "corner-000", dec, "rational_corner")
    return pd.DataFrame(out)


def midpoint_results(games=None) -> pd.DataFrame:
    """games.play-shaped rows of the midpoint baseline: numerical games get one agent at the midpoint of each
    condition's bounds; multiple-choice games get one agent per answer combination across the game's conditions
    (every combination once, i.e. the exact uniform-random expectation of each moment)."""
    gm = _games()
    out = []
    for game in gm.list_games():
        if games is not None and game not in games:
            continue
        g = gm.GAMES[game]
        conds = [str(c["id"]) for c in g["conditions"]]
        if g["question_type"] == "numerical":
            dec = {}
            for c in g["conditions"]:
                lo, hi = gm.bounds(g, c)
                dec[str(c["id"])] = (float(lo) + float(hi)) / 2
            out += _rows(game, "midpoint-000", dec, "midpoint")
        else:
            for k, combo in enumerate(itertools.product(g["options"], repeat=len(conds))):
                out += _rows(game, f"midpoint-{k:05d}", dict(zip(conds, combo)), "midpoint")
    return pd.DataFrame(out)


# ---------------------------------------------------------------------- results -> moment values

def _as_results(results: Any) -> pd.DataFrame:
    if isinstance(results, (str, os.PathLike)):
        return _games().read_results(results)
    if not isinstance(results, pd.DataFrame):
        raise TypeError(f"results must be a games.play DataFrame or a CSV path, not {type(results).__name__}")
    return results


def is_results(x: Any) -> bool:
    """True for a games.play results table (as opposed to a prediction vector)."""
    return isinstance(x, pd.DataFrame) and {"game", "condition", "decision", "valid"} <= set(x.columns)


def moment_values(results: Any, refs: pd.DataFrame, *, model: str | None = None) -> dict[str, dict[str, float]]:
    """{persona_method: {unit 'game:moment': simulated value}} through `games.moment_frame` (all scales)."""
    res = _as_results(results)
    if res.empty:
        return {}
    fr = _games().moment_frame(res, refs, model=model, include_raw=True)
    out = {}
    for col in [c for c in fr.columns if c.startswith(PRED_PREFIX)]:
        out[col[len(PRED_PREFIX):]] = dict(zip(fr["unit"], fr[col].astype(float)))
    return out


def _align(values: Mapping[str, float], units) -> np.ndarray:
    return np.array([float(values.get(u, np.nan)) for u in units], dtype=float)


def results_method(results: Any, persona_method: str | None = None, *, model: str | None = None) -> Callable:
    """A bench method from a games.play results table (or CSV path): f(data) -> moment values aligned with
    data.test. With several persona methods in `results`, name one."""
    def f(data: TaskData):
        vals = moment_values(results, data.meta["refs"], model=model)
        if not vals:
            return np.full(len(data.test), np.nan)
        pm = persona_method
        if pm is None:
            if len(vals) > 1:
                raise ValueError(f"results hold several persona methods {sorted(vals)}: pass persona_method=")
            pm = next(iter(vals))
        if pm not in vals:
            raise KeyError(f"persona_method {pm!r} not in results ({sorted(vals)})")
        return _align(vals[pm], data.test["unit"])
    f.__name__ = persona_method or "results"
    return f


def play_method(llm, personas: Any, *, persona_method: str | None = None, games: Any = None,
                **play_kw: Any) -> Callable:
    """A bench method that plays the games behind the test moments with `games.play(llm, personas, ...)` and
    returns the moment values. play_kw go to games.play (runs, seed, max_tokens, confirm, out, ...); the games.play
    cost gate applies."""
    def f(data: TaskData):
        gl = games if games is not None else sorted(set(data.test["game"]))
        res = _games().play(llm, personas, gl, **play_kw)
        return results_method(res, persona_method)(data)
    f.__name__ = persona_method or "play"
    return f


# ---------------------------------------------------------------------- metrics

def _err_metric(kind: str):
    def fn(task, frame, col, B, seed):
        y, p = frame["truth"].to_numpy(float), frame[col].to_numpy(float)
        groups = frame["group"].to_numpy() if "group" in frame.columns else np.arange(len(frame))

        def stat(idx):
            e = p[idx] - y[idx]
            e = e[np.isfinite(e)]
            if not len(e):
                return float("nan")
            return float(np.mean(np.abs(e))) if kind == "mae" else float(np.sqrt(np.mean(e ** 2)))
        ok = np.isfinite(p) & np.isfinite(y)
        if not ok.any():
            return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
        idx = np.flatnonzero(ok)
        return cluster_bootstrap(lambda j: stat(idx[j]), groups[ok], B=B, seed=seed)
    return fn


def _coverage(task, frame, col, B, seed):
    p = frame[col].to_numpy(float)
    v = float(np.isfinite(p).mean()) if len(p) else float("nan")
    return {"value": v, "lo": float("nan"), "hi": float("nan"), "n": int(len(p))}


METRICS = (Metric("mae", False, _err_metric("mae"), "mean |sim - human| over test moments; game-cluster bootstrap"),
           Metric("rmse", False, _err_metric("rmse"), "root mean squared error; game-cluster bootstrap"),
           Metric("coverage", True, _coverage, "share of test moments with a finite prediction"))


# ---------------------------------------------------------------------- task

def _unit_table(scale: str) -> pd.DataFrame:
    refs = _games().human_refs()
    if scale == "share":
        refs = refs[refs["scale"] == "share"]
    elif scale != "all":
        raise ValueError("scale must be 'share' or 'all'")
    t = refs.rename(columns={"value": "truth"}).copy()
    t["unit"] = t["game"] + ":" + t["moment"]
    return t.sort_values(["game", "moment"])[UNIT_COLUMNS].reset_index(drop=True)


@register()
class EconGamesB01(BenchTask):
    """Simulated vs human moments of the simengine.games battery (b01 games + ertan_vignette). Data-free."""

    name = "econ_games_b01"
    kind = "continuous"
    data_subdir = None
    accepted_options = {
        "split": '"holdout" | "all" (default "holdout")',
        "scale": '"share" | "all" (default "share")',
        "results": "games.play results DataFrame or CSV path (adds bare_model from its bare rows)",
        "bare_method": 'persona_method of the bare model in results (default "baseline")',
        "llm": "an LLM: adds bare_model by playing the games with empty personas (cost gate applies)",
        "bare_agents": "no-persona agents when llm is given (default 10)",
        "runs": "runs per cell when llm is given (default 3)",
        "seed": "seed when llm is given (default 0)",
        "play_kw": "dict of extra games.play arguments when llm is given (e.g. confirm, max_tokens, out)",
        "model": "model to keep when results hold several",
    }
    layout = """
    (none) the human moments are inlined in simengine.games (human_refs(), each with its citation);
    data_root is ignored. Optional: option results=<games.play CSV> for the bare_model baseline.
    """
    source = ("edsl-test benchmark/beebench/b01 (games, holdout, references); replicant papers/ertan2009/config.py "
              "(ertan_vignette); scoring is simengine's own (games.moments / moment_frame)")
    metrics = METRICS

    def load(self, data_root: str | None = None) -> TaskData:
        data = self._load(None)
        data.meta.setdefault("task", self.name)
        data.meta.setdefault("kind", self.kind)
        data.meta.setdefault("options", {k: v for k, v in self.options.items() if k not in ("llm", "results")})
        return data

    def _units(self) -> pd.DataFrame:
        return _unit_table(self.options.get("scale", "share"))

    def _split(self, units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        split = self.options.get("split", "holdout")
        if split == "holdout":
            tr, te = units[~units["holdout"]], units[units["holdout"]]
            rule = "holdout: calibration-game moments train, holdout-game moments test (b01 games/holdout split)"
        elif split == "all":
            tr, te = units.iloc[:0], units
            rule = "all: every referenced moment is test"
        else:
            raise ValueError("split must be 'holdout' or 'all'")
        return tr.reset_index(drop=True), te.reset_index(drop=True), rule

    def _load(self, root: str | None) -> TaskData:
        units = self._units()
        tr, te, rule = self._split(units)
        refs = _games().human_refs()
        meta = {"split": rule, "scale": self.options.get("scale", "share"), "n_train": len(tr), "n_test": len(te),
                "refs": refs, "source": "published human moments inlined in simengine.games (see citation column)"}
        return TaskData(train=tr, test=te, meta=meta)

    # ------------------------------------------------------------------ baselines
    def _baseline_values(self, refs: pd.DataFrame) -> dict[str, dict[str, float]]:
        out = {"rational_corner": moment_values(corner_results(), refs)["rational_corner"],
               "midpoint": moment_values(midpoint_results(), refs)["midpoint"]}
        bare = self._bare_results()
        if bare is not None and len(bare):
            vals = moment_values(bare, refs, model=self.options.get("model"))
            bm = self.options.get("bare_method", BARE_METHOD)
            if bm in vals:
                out["bare_model"] = vals[bm]
        return out

    def _bare_results(self) -> pd.DataFrame | None:
        bm = self.options.get("bare_method", BARE_METHOD)
        res = self.options.get("results")
        prior = None
        if res is not None:
            r = _as_results(res)
            prior = r[r["persona_method"].astype(str) == bm]
            if self.options.get("model") is not None:
                prior = prior[prior["model"].astype(str) == str(self.options["model"])]
        llm = self.options.get("llm")
        if llm is None:
            return prior
        n = int(self.options.get("bare_agents", 10))
        personas = {bm: {f"{bm}-{i:03d}": "" for i in range(n)}}
        gl = sorted(set(self._units()["game"]))
        kw = dict(self.options.get("play_kw") or {})
        return _games().play(llm, personas, gl, runs=int(self.options.get("runs", 3)),
                             seed=int(self.options.get("seed", 0)), resume=prior if prior is not None and len(prior)
                             else None, **kw)

    def baselines(self, data: TaskData) -> dict[str, np.ndarray]:
        cache = data.meta.get("_baseline_values")
        if cache is None:
            cache = data.meta["_baseline_values"] = self._baseline_values(data.meta["refs"])
        return {k: _align(v, data.test["unit"]) for k, v in cache.items()}

    # ------------------------------------------------------------------ frames
    def predict(self, method: Any, data: TaskData) -> Any:
        """A games.play results table (or a CSV path to one) is passed through; frame() turns it into moments."""
        if is_results(method):
            return method
        if isinstance(method, (str, os.PathLike)) and str(method).endswith(".csv"):
            return _as_results(method)
        return super().predict(method, data)

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        return pd.DataFrame({"unit": t["unit"].to_numpy(), "group": t["game"].to_numpy(),
                             "truth": t["truth"].to_numpy(float), "split": "test"})

    def frame(self, data: TaskData, preds: dict[str, Any]) -> pd.DataFrame:
        """EvalFrame of test moments. A prediction may be a vector aligned with data.test or a games.play results
        table; a table with several persona methods adds one column per method (`<name>_<persona_method>`)."""
        flat: dict[str, Any] = {}
        for m, p in preds.items():
            if is_results(p):
                vals = moment_values(p, data.meta["refs"], model=self.options.get("model"))
                if not vals:
                    flat[m] = np.full(len(data.test), np.nan)
                elif len(vals) == 1:
                    flat[m] = _align(next(iter(vals.values())), data.test["unit"])
                else:
                    for pm, v in vals.items():
                        flat[f"{m}_{pm}"] = _align(v, data.test["unit"])
            else:
                flat[m] = p
        return super().frame(data, flat)

    def results_frame(self, results: Any = None, *, data: TaskData | None = None,
                      baselines: bool = True) -> pd.DataFrame:
        """Train + test EvalFrame (unit, group, truth, pred_<persona_method>..., pred_<baseline>..., split) for
        `simengine.evidence.evaluate(frame, ours=..., task="continuous")`: the calibration moments let evaluate
        fit its own baselines, the holdout moments are scored."""
        data = data if data is not None else self.load()
        units = pd.concat([data.train.assign(split="train"), data.test.assign(split="test")], ignore_index=True)
        fr = pd.DataFrame({"unit": units["unit"], "group": units["game"], "truth": units["truth"].astype(float)})
        cols: dict[str, dict] = {}
        if baselines:
            cols.update(self._baseline_values(data.meta["refs"]))
        if results is not None:
            cols.update(moment_values(results, data.meta["refs"], model=self.options.get("model")))
        for m, v in cols.items():
            fr[PRED_PREFIX + m] = _align(v, units["unit"])
        fr["split"] = units["split"].to_numpy()
        return fr


__all__ = ["EconGamesB01", "CORNER_DECISIONS", "corner_results", "midpoint_results", "moment_values",
           "results_method", "play_method", "is_results", "BARE_METHOD"]
