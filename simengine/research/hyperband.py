"""Multi-fidelity search: Hyperband and ASHA on top of `successive_halving`.

Hyperband (Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar, JMLR 18(185), 2018; arXiv:1603.06560, Algorithm 1) runs
several brackets of successive halving that trade the number of configs n against the budget each one gets. ASHA
(Li et al., "A System for Massively Parallel Hyperparameter Tuning", MLSys 2020; arXiv:1810.05934, Algorithm 2)
promotes a config to the next rung as soon as it is in the top 1/eta of the completed results of its rung, so workers
never wait for a rung to fill.

The resource ("budget") is an integer count of units: rows of a sample, personas, questions, or LLM calls. A
multi-fidelity runner maps (config, budget) to one primary-metric score:

- `MultiFidelity(fn, max_budget)` wraps `fn(cfg, budget) -> score`, caches repeated (config, budget) pairs and
  counts the resource it charged (`spent`).
- `UnitRunner(score_units, n_units)` builds one from a per-unit scorer `score_units(cfg, idx) -> scores`. Budget b
  scores the first b units of one seeded permutation shared by every config (common random numbers), and a promoted
  config pays only for its new units (`resume=True`, the paper's "resume from checkpoint" case).

    import numpy as np
    from simengine.research import Space
    from simengine.research.hyperband import UnitRunner, asha, hyperband, space_sampler
    noise = np.random.default_rng(0).normal(size=(81,))
    run = UnitRunner(lambda cfg, idx: (cfg["x"] - 0.7) ** 2 + 0.3 * noise[idx], n_units=81, unit="llm_calls")
    sample = space_sampler(Space({"x": (0.0, 1.0)}))
    hb = hyperband(run, sample, higher_is_better=False, eta=3)       # hb["winner"], hb["score"], hb["budget"]
    run2 = UnitRunner(lambda cfg, idx: (cfg["x"] - 0.7) ** 2 + 0.3 * noise[idx], n_units=81, unit="llm_calls")
    out = asha(run2, sample, higher_is_better=False, total_budget=hb["budget"])
"""
from __future__ import annotations

import math
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from .proposers import successive_halving
from .space import Space, config_hash

UNITS = ("samples", "llm_calls")
Sampler = Callable[[np.random.Generator, int], list]


# ===================================================================== multi-fidelity runner protocol

@runtime_checkable
class FidelityRunner(Protocol):
    """(config, budget) -> one primary-metric score. `budget` is an int in [min_budget, max_budget] counted in
    `unit` ("samples" = units of a sample, "llm_calls" = LLM calls, or any label)."""
    max_budget: int
    min_budget: int
    unit: str

    def __call__(self, cfg: dict, budget: int) -> float: ...


class MultiFidelity:
    """Counted multi-fidelity runner around `fn(cfg, budget) -> score`.

    A repeated (config, budget) pair is served from the cache and not charged. `spent` = resource units charged,
    `calls` = runs, `log` = [(cfg_hash, budget, score)]. `screen(configs, fidelity)` is the `successive_halving`
    evaluate with budget = round(fidelity * max_budget). Thread-safe for `asha(n_workers > 1)`."""

    def __init__(self, fn: Callable[[dict, int], float] | None, max_budget: int, *, min_budget: int = 1,
                 unit: str = "samples"):
        if int(max_budget) < 1 or not 1 <= int(min_budget) <= int(max_budget):
            raise ValueError(f"need 1 <= min_budget <= max_budget, got {min_budget}, {max_budget}")
        self.fn, self.max_budget, self.min_budget, self.unit = fn, int(max_budget), int(min_budget), unit
        self.spent, self.calls = 0, 0
        self.log: list[tuple[str, int, float]] = []
        self._cache: dict[tuple[str, int], float] = {}
        self._lock = threading.Lock()

    def budget_of(self, fidelity: float) -> int:
        return int(min(self.max_budget, max(self.min_budget, round(float(fidelity) * self.max_budget))))

    def _charge(self, n: int) -> None:
        with self._lock:
            self.spent += int(n)

    def _run(self, cfg: dict, budget: int) -> float:
        self._charge(budget)
        return self.fn(cfg, budget)

    def __call__(self, cfg: dict, budget: int) -> float:
        budget = int(budget)
        if not self.min_budget <= budget <= self.max_budget:
            raise ValueError(f"budget {budget} outside [{self.min_budget}, {self.max_budget}]")
        key = (config_hash(cfg), budget)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        score = float(self._run(cfg, budget))
        with self._lock:
            self._cache[key] = score
            self.calls += 1
            self.log.append((key[0], budget, score))
        return score

    def screen(self, configs: list, fidelity: float) -> list:
        b = self.budget_of(fidelity)
        return [self(c, b) for c in configs]


class UnitRunner(MultiFidelity):
    """Budget = number of units. `score_units(cfg, idx)` returns one score per unit index in `idx` (one LLM call per
    unit for LLM tasks); the score at budget b is agg(scores of order[:b]) with `order` one seeded permutation of
    range(n_units), shared by all configs. With resume=True a config's per-unit scores are kept, so going from b1
    to b2 charges b2 - b1 units; resume=False re-scores (and charges) all b units on every run."""

    def __init__(self, score_units: Callable[[dict, np.ndarray], Any], n_units: int, *, max_budget: int | None = None,
                 min_budget: int = 1, unit: str = "samples", seed: int = 0, resume: bool = True,
                 agg: Callable[[np.ndarray], float] = np.mean):
        super().__init__(None, max_budget or n_units, min_budget=min_budget, unit=unit)
        if self.max_budget > n_units:
            raise ValueError(f"max_budget {self.max_budget} > n_units {n_units}")
        self.score_units, self.n_units, self.resume, self.agg = score_units, int(n_units), resume, agg
        self.order = np.random.default_rng(seed).permutation(self.n_units)
        self._units: dict[str, np.ndarray] = {}

    def _run(self, cfg: dict, budget: int) -> float:
        h = config_hash(cfg)
        with self._lock:
            have = self._units.get(h, np.empty(0)) if self.resume else np.empty(0)
        if len(have) < budget:
            idx = self.order[len(have):budget]
            new = np.asarray(self.score_units(cfg, idx), float).reshape(-1)
            if len(new) != len(idx):
                raise ValueError(f"score_units returned {len(new)} scores for {len(idx)} units")
            self._charge(len(idx))
            have = np.concatenate([have, new])
            if self.resume:
                with self._lock:
                    if len(have) > len(self._units.get(h, ())):
                        self._units[h] = have
        return float(self.agg(have[:budget]))

    def per_unit(self, cfg: dict) -> np.ndarray:
        """Per-unit scores scored so far for `cfg` (in `order`), empty if none or resume=False."""
        return self._units.get(config_hash(cfg), np.empty(0)).copy()


# ===================================================================== samplers

def space_sampler(space: Space, *, max_tries: int = 1000) -> Sampler:
    """i.i.d. configs from a Space: choices uniform, int ranges uniform inclusive, float ranges uniform; fixed keys keep
    their defaults; forbidden values and configs failing `validate` (including `check`) are redrawn."""
    def sample(rng: np.random.Generator, n: int) -> list:
        out = []
        for _ in range(n):
            for _ in range(max_tries):
                cfg = {}
                for k, u in space.params.items():
                    if k in space.fixed:
                        continue
                    if isinstance(u, list):
                        u = [x for x in u if x not in space.forbidden.get(k, [])]
                        v = u[int(rng.integers(len(u)))] if u else None
                    elif space.is_int_range(k):
                        v = int(rng.integers(u[0], u[1] + 1))
                    else:
                        v = float(rng.uniform(u[0], u[1]))
                    cfg[k] = v.item() if isinstance(v, np.generic) else v
                if any(v in space.forbidden.get(k, []) for k, v in cfg.items()):
                    continue
                full, err = space.validate(cfg)
                if not err:
                    out.append(full)
                    break
            else:
                raise ValueError(f"space_sampler: no valid config in {max_tries} draws")
        return out
    return sample


def pool_sampler(pool: Sequence[dict], *, shuffle: bool = True) -> Sampler:
    """Draw from a finite candidate list without replacement over the whole search; returns fewer configs (then none)
    once the pool is exhausted. The order is drawn from the search's rng on first use."""
    state: dict[str, Any] = {"order": None, "pos": 0}

    def sample(rng: np.random.Generator, n: int) -> list:
        if state["order"] is None:
            state["order"] = rng.permutation(len(pool)) if shuffle else np.arange(len(pool))
        i = state["pos"]
        state["pos"] = min(len(pool), i + n)
        return [dict(pool[j]) for j in state["order"][i:state["pos"]]]
    return sample


def _as_sampler(sample) -> Sampler:
    if isinstance(sample, Space):
        return space_sampler(sample)
    if isinstance(sample, (list, tuple)):
        return pool_sampler(list(sample))
    if callable(sample):
        return sample
    raise TypeError("sample must be a Space, a list of configs or a callable (rng, n) -> configs")


# ===================================================================== Hyperband

def _s_max(ratio: float, eta: int) -> int:
    """floor(log_eta(ratio)) without float log error (log(81)/log(3) is not exactly 4)."""
    s = 0
    while eta ** (s + 1) <= ratio * (1 + 1e-12):
        s += 1
    return s


def _check_eta(eta) -> int:
    if isinstance(eta, bool) or int(eta) != eta or eta < 2:
        raise ValueError(f"eta must be an int >= 2, got {eta!r}")
    return int(eta)


def hyperband_brackets(max_budget: int, min_budget: int = 1, eta: int = 3) -> list[dict]:
    """Bracket plan of Algorithm 1 with R = max_budget / min_budget: s_max = floor(log_eta R), B = (s_max + 1) R;
    bracket s draws n = ceil(B / R * eta^s / (s + 1)) configs and rung i runs floor(n eta^-i) of them at
    max_budget * eta^(i - s) (rounded, at least min_budget). For R = 81, eta = 3 that gives n = 81, 34, 15, 8, 5.
    Returns [{"s", "n", "counts", "budgets", "cost"}] from s = s_max down to 0; cost = sum(counts * budgets)."""
    eta = _check_eta(eta)
    if int(max_budget) < 1 or not 1 <= int(min_budget) <= int(max_budget):
        raise ValueError(f"need 1 <= min_budget <= max_budget, got {min_budget}, {max_budget}")
    smax = _s_max(max_budget / min_budget, eta)
    out = []
    for s in range(smax, -1, -1):
        n = -(-(smax + 1) * eta ** s // (s + 1))        # integer ceil: 5 / 3 * 9 must give 15, not 16
        counts = [n // eta ** i for i in range(s + 1)]
        budgets = [max(int(min_budget), int(round(max_budget * float(eta) ** (i - s)))) for i in range(s + 1)]
        out.append({"s": s, "n": n, "counts": counts, "budgets": budgets,
                    "cost": int(sum(c * b for c, b in zip(counts, budgets)))})
    return out


def _runner_range(runner, max_budget, min_budget) -> tuple[int, int]:
    """(R, r) for a FidelityRunner; refuses budgets the runner would clamp (nominal budget would be wrong)."""
    Rr, rr = int(runner.max_budget), int(getattr(runner, "min_budget", 1))
    R, r = int(max_budget or Rr), int(min_budget or rr)
    if R > Rr or r < rr:
        raise ValueError(f"budgets [{r}, {R}] outside the runner's [{rr}, {Rr}]")
    return R, r


def _as_screen(runner, max_budget, min_budget):
    if hasattr(runner, "max_budget") and callable(runner):         # FidelityRunner: (cfg, budget) -> score
        R, r = _runner_range(runner, max_budget, min_budget)
        if hasattr(runner, "screen") and R == runner.max_budget:
            return runner.screen, R, r
        return (lambda cfgs, f: [runner(c, int(min(R, max(r, round(f * R))))) for c in cfgs]), R, r
    if callable(runner):
        if max_budget is None:
            raise ValueError("a plain screen callable (configs, fidelity) -> scores needs max_budget=")
        return runner, int(max_budget), int(min_budget or 1)
    raise TypeError("runner must be a MultiFidelity / UnitRunner or a screen callable (configs, fidelity) -> scores")


def _better(a: float, b: float, higher_is_better: bool) -> bool:
    if math.isnan(a):
        return False
    if math.isnan(b):
        return True
    return a > b if higher_is_better else a < b


def hyperband(runner, sample, *, higher_is_better: bool, eta: int = 3, max_budget: int | None = None,
              min_budget: int | None = None, seed: int = 0, brackets: Sequence[int] | None = None,
              repeats: int = 1) -> dict:
    """Hyperband, Algorithm 1: for s = s_max..0 run `successive_halving(..., rounding="floor")` over n fresh configs
    with the bracket's budgets; return the best bracket winner. Every bracket ends at max_budget, so the winners are
    compared on full-budget scores. Deviation: Alg. 1 line 11 returns the config with the smallest intermediate loss
    seen so far, which includes low-budget rungs; a noisy small-sample (or early-iteration) score would win that
    minimum by chance, so only max_budget scores are compared here. `brackets` restricts the s values (e.g. [s_max] = plain halving with Hyperband's n), `repeats` runs
    the outer loop again with fresh configs (the paper recommends repeating).

    `runner`: a FidelityRunner `(cfg, budget) -> score` (MultiFidelity, UnitRunner or any object with max_budget;
    max_budget= / min_budget= may narrow but not widen its range), or a plain screen `(configs, fidelity) -> scores`
    with `max_budget=` (fidelity = budget / max_budget). `sample`: a Space, a list of configs (drawn without replacement), or
    `(rng, n) -> configs`. Returns {"winner", "score", "bracket", "brackets": [...], "evaluations", "budget"}; budget =
    nominal resource (sum of budget per evaluation; a resuming UnitRunner charges less, see `runner.spent`)."""
    eta = _check_eta(eta)
    screen, R, r = _as_screen(runner, max_budget, min_budget)
    draw = _as_sampler(sample)
    rng = np.random.default_rng(seed)
    plan = [b for b in hyperband_brackets(R, r, eta) if brackets is None or b["s"] in brackets]
    if not plan:
        raise ValueError(f"no bracket in {brackets} (s_max = {_s_max(R / r, eta)})")
    done, best = [], None
    for rep in range(int(repeats)):
        for b in plan:
            cands = list(draw(rng, b["n"]))
            rec = {"s": b["s"], "repeat": rep, "n": len(cands), "budgets": b["budgets"], "winner": None,
                   "score": float("nan"), "evaluations": 0, "budget": 0, "rungs": []}
            if cands:
                res = successive_halving(screen, cands, higher_is_better=higher_is_better,
                                         fidelities=tuple(x / R for x in b["budgets"]), eta=eta, rounding="floor")
                rec["rungs"] = res["rungs"]
                rec["evaluations"] = res["evaluations"]
                rec["budget"] = int(sum(len(p) * x for p, x in zip(res["rungs"], b["budgets"])))
                if res["winner"] is not None:
                    rec["winner"] = res["winner"]
                    if len(res["rungs"]) == len(b["budgets"]):
                        rec["score"] = float(dict(res["rungs"][-1])[res["winner_index"]])
                    else:               # NaNs left one survivor before the last rung: score it at max_budget
                        rec["score"] = float(screen([res["winner"]], 1.0)[0])
                        rec["evaluations"] += 1
                        rec["budget"] += R
            done.append(rec)
            if rec["winner"] is not None and (best is None or _better(rec["score"], best["score"], higher_is_better)):
                best = rec
    return {"winner": best["winner"] if best else None, "score": best["score"] if best else float("nan"),
            "bracket": best["s"] if best else None, "brackets": done,
            "evaluations": int(sum(d["evaluations"] for d in done)), "budget": int(sum(d["budget"] for d in done)),
            "max_budget": R, "min_budget": r, "eta": eta}


# ===================================================================== ASHA

@dataclass
class Job:
    trial: int          # index into ASHA.trials
    cfg: dict
    rung: int
    budget: int


class ASHA:
    """Algorithm 2 scheduler, driven by `get_job` / `report` from any executor.

    Rungs k = 0..K with K = floor(log_eta(max_budget / min_budget)) - s; rung k runs at max_budget * eta^(k - K)
    (= min_budget * eta^(s + k) when the ratio is a power of eta; anchored at max_budget so the top rung is a full run).
    get_job() scans k = K-1, ..., 0 for a config in the top floor(|rung k| / eta) of rung k (|rung k| = completed
    results) that is not yet promoted and promotes it to k + 1; otherwise it starts a new config in rung 0. NaN scores
    rank last and are never promoted."""

    def __init__(self, max_budget: int, min_budget: int = 1, eta: int = 3, s: int = 0, *, higher_is_better: bool):
        self.eta = _check_eta(eta)
        if int(max_budget) < 1 or not 1 <= int(min_budget) <= int(max_budget):
            raise ValueError(f"need 1 <= min_budget <= max_budget, got {min_budget}, {max_budget}")
        smax = _s_max(max_budget / min_budget, self.eta)
        if not 0 <= s <= smax:
            raise ValueError(f"s must be in [0, {smax}], got {s}")
        self.max_budget, self.min_budget, self.s, self.hib = int(max_budget), int(min_budget), int(s), higher_is_better
        self.K = smax - self.s
        self.budgets = [max(self.min_budget, int(round(max_budget * float(self.eta) ** (k - self.K))))
                        for k in range(self.K + 1)]
        self.trials: list[dict] = []
        self.rungs: list[dict[int, float]] = [{} for _ in self.budgets]     # trial -> score, completion order
        self.promoted: list[set[int]] = [set() for _ in self.budgets]
        self.issued = 0                                                     # nominal budget of issued jobs
        self.n_jobs = 0

    def _top(self, k: int) -> list[int]:
        items = [(t, v) for t, v in self.rungs[k].items() if not math.isnan(v)]
        items.sort(key=lambda p: (-p[1] if self.hib else p[1], p[0]))
        return [t for t, _ in items[:len(self.rungs[k]) // self.eta]]

    def get_job(self, new_config: Callable[[], dict | None] | None = None) -> Job | None:
        """Next job, or None when nothing is promotable and `new_config` returns None (or is not given)."""
        for k in range(self.K - 1, -1, -1):
            for t in self._top(k):
                if t not in self.promoted[k]:
                    self.promoted[k].add(t)
                    return self._issue(Job(t, self.trials[t], k + 1, self.budgets[k + 1]))
        cfg = new_config() if new_config is not None else None
        if cfg is None:
            return None
        self.trials.append(cfg)
        return self._issue(Job(len(self.trials) - 1, cfg, 0, self.budgets[0]))

    def _issue(self, job: Job) -> Job:
        self.issued += job.budget
        self.n_jobs += 1
        return job

    def report(self, job: Job, score: float) -> None:
        self.rungs[job.rung][job.trial] = float(score)

    def best(self) -> dict | None:
        """Best finite score in the highest rung that has one (ASHA's current incumbent)."""
        for k in range(self.K, -1, -1):
            items = [(t, v) for t, v in self.rungs[k].items() if not math.isnan(v)]
            if items:
                t, v = min(items, key=lambda p: (-p[1] if self.hib else p[1], p[0]))
                return {"trial": t, "cfg": self.trials[t], "score": v, "rung": k, "budget": self.budgets[k]}
        return None


def asha(runner: Callable[[dict, int], float], sample, *, higher_is_better: bool, eta: int = 3, s: int = 0,
         max_budget: int | None = None, min_budget: int | None = None, total_budget: int | None = None,
         max_trials: int | None = None, n_workers: int = 1, seed: int = 0) -> dict:
    """Run ASHA until the issued nominal budget reaches `total_budget` (the last jobs may overshoot it by at most
    n_workers * max_budget), `max_trials` configs have been started, or the sampler is exhausted and nothing is promotable.
    `runner(cfg, budget) -> score` (a MultiFidelity / UnitRunner; max_budget/min_budget default to its own).
    n_workers > 1 runs jobs in a thread pool (I/O-bound runners such as LLM calls); completion order then varies, so
    only n_workers=1 is reproducible. Returns {"winner", "score", "rung", "budget", "trials", "jobs", "rungs",
    "budgets", "promotions"}."""
    if hasattr(runner, "max_budget"):
        R, r = _runner_range(runner, max_budget, min_budget)
    else:
        R, r = int(max_budget or 0), int(min_budget or 1)
    if R < 1:
        raise ValueError("asha needs max_budget= (or a runner with .max_budget)")
    draw = _as_sampler(sample)
    if total_budget is None and max_trials is None and not isinstance(sample, (list, tuple)):
        raise ValueError("asha needs total_budget= or max_trials= (an unbounded sampler never runs out)")
    sched = ASHA(R, r, eta, s, higher_is_better=higher_is_better)
    rng = np.random.default_rng(seed)

    def new():
        if max_trials is not None and len(sched.trials) >= max_trials:
            return None
        got = draw(rng, 1)
        return got[0] if got else None

    def stop():
        return total_budget is not None and sched.issued >= total_budget

    if n_workers <= 1:
        while not stop():
            job = sched.get_job(new)
            if job is None:
                break
            sched.report(job, runner(job.cfg, job.budget))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            running: dict = {}
            while True:
                while len(running) < n_workers and not stop():
                    job = sched.get_job(new)
                    if job is None:
                        break
                    running[ex.submit(runner, job.cfg, job.budget)] = job
                if not running:
                    break
                fin, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in fin:
                    job = running.pop(fut)
                    try:
                        score = float(fut.result())
                    except Exception:           # a crashed job ranks last; the search goes on
                        score = float("nan")
                    sched.report(job, score)
    b = sched.best()
    return {"winner": b["cfg"] if b else None, "score": b["score"] if b else float("nan"),
            "rung": b["rung"] if b else None, "budget": sched.issued, "trials": len(sched.trials),
            "jobs": sched.n_jobs, "rungs": [sorted(x.items()) for x in sched.rungs], "budgets": sched.budgets,
            "promotions": int(sum(len(p) for p in sched.promoted)), "scheduler": sched}
