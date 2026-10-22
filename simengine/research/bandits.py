"""Bandits for choosing among methods / configs under an evaluation budget: regret policies (UCB1, Thompson sampling
with Beta or Gaussian priors), best-arm identification (successive rejects, LUCB1, uniform allocation), a vectorised
simulator for identification rates, and `RacingProposer`, which races single changes inside `Loop`.

    from simengine.research.bandits import SimulatedArms, identify_best, simulate, RacingProposer
    arms = SimulatedArms([0.5, 0.45, 0.4], kind="bernoulli", seed=1)
    res = identify_best(arms, 3, budget=600, algorithm="successive_rejects", seed=0)
    res.best, res.pulls, res.means                       # recommended arm, pulls per arm, empirical means
    simulate(arms, "lucb", budget=5000, reps=1000, delta=0.05)["error_rate"]
    loop = Loop(runner, space, proposer=RacingProposer("successive_rejects", pulls_per_arm=4), primary="err",
                higher_is_better=False)                  # runner reads cfg["fidelity"] = 1, 2, ... as a mini-batch index

An arm is pulled through `pull(arm) -> float` (one noisy reward; higher is better unless higher_is_better=False).
Every algorithm maximises; a lower-is-better metric is negated, and a non-finite reward (a crashed evaluation) counts
as -inf, so that arm is never recommended unless every arm crashed.

Algorithms (checked against the papers' algorithm boxes):
- UCB1 (Auer, Cesa-Bianchi & Fischer, Machine Learning 47, 2002, Fig. 1): play each arm once, then the arm maximising
  mean_j + sqrt(2 ln n / n_j), n = plays so far. Rewards in [0, 1]; `scale` = reward range for other supports.
- Thompson sampling, Beta (Agrawal & Goyal, COLT 2012, Algorithms 1-2): Beta(S + 1, F + 1) posteriors; a reward in
  [lo, hi] is rescaled to [0, 1] and turned into a Bernoulli trial with that success probability (Algorithm 2).
- Thompson sampling, Gaussian (Agrawal & Goyal, AISTATS 2013, Algorithm 2): theta_i ~ N(mu_i, sigma^2 / (k_i + 1)),
  mu_i = (prior_mean + sum of rewards) / (k_i + 1); sigma = 1, prior_mean = 0 is the paper's algorithm. The prior has
  the same variance as one reward, so with a small sigma and a prior_mean far from the rewards an unpulled arm is
  almost never sampled again; `warmup=1` (both Thompson variants; not in the papers) pulls every arm once first.
- Successive rejects (Audibert, Bubeck & Munos, COLT 2010, Fig. 3): K - 1 phases, n_k = ceil((n - K) /
  (logbar(K) (K + 1 - k))), logbar(K) = 1/2 + sum_{i=2..K} 1/i; each surviving arm is pulled n_k - n_{k-1} times,
  then the lowest empirical mean is dismissed (ties at random). Parameter-free, never exceeds the budget n.
- LUCB1 (Kalyanakrishnan, Tewari, Auer & Stone, ICML 2012, §3, m = 1): rounds 1..K sample every arm once; each
  later round t = K + 1, K + 2, ... take h = empirical best, l = argmax over the others of mean + beta; stop iff
  (mean_l + beta_l) - (mean_h - beta_h) < eps, else sample h and l (§3.2 footnote 3: two arms per round from round
  K + 1). beta(u, t) = scale * sqrt(ln(k1 K t^4 / delta) / (2u)), k1 = 5/4. A `budget` caps the pulls (then the
  empirical best is returned with stopped="budget").
- uniform: every arm floor(n/K) times (the remainder to random distinct arms), recommend the empirical best; the
  baseline of Audibert & Bubeck's experiments.
Regret policies used for identification recommend the empirical best at the end of the budget.

`simulate` runs `reps` independent replications at once (state arrays of shape (reps, K)), so identification rates
over thousands of runs take seconds. `AUDIBERT_BUBECK_2010` holds the seven Bernoulli problems of that paper's §7.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

ALGORITHMS = ("ucb1", "thompson_beta", "thompson_gaussian", "successive_rejects", "lucb", "uniform")
_ALIASES = {"ucb": "ucb1", "ts": "thompson_beta", "ts_beta": "thompson_beta", "beta": "thompson_beta",
            "ts_gaussian": "thompson_gaussian", "gaussian": "thompson_gaussian", "sr": "successive_rejects",
            "lucb1": "lucb", "unif": "uniform"}
LUCB_K1 = 5 / 4


def _ab_problems() -> dict:
    def mk(best, rest):
        return np.array([best] + list(rest), float)
    return {
        "exp1": (mk(0.5, [0.4] * 19), 2000),
        "exp2": (mk(0.5, [0.42] * 5 + [0.38] * 14), 2000),
        "exp3": (mk(0.5, [0.5 - 0.37 ** i for i in (2, 3, 4)]), 2000),
        "exp4": (mk(0.5, [0.42, 0.4, 0.4, 0.35, 0.35]), 600),
        "exp5": (mk(0.5, [0.5 - 0.025 * i for i in range(2, 16)]), 4000),
        "exp6": (mk(0.5, [0.48] + [0.37] * 18), 6000),
        "exp7": (mk(0.5, [0.45] * 5 + [0.43] * 14 + [0.38] * 10), 6000),
        "exp7_2h": (mk(0.5, [0.45] * 5 + [0.43] * 14 + [0.38] * 10), 12000),
    }


# Bernoulli means (best arm first, mean 0.5) and budget n of Audibert, Bubeck & Munos (COLT 2010) §7 / Figure 5.
AUDIBERT_BUBECK_2010 = _ab_problems()


def canonical(algorithm: str) -> str:
    a = algorithm.lower().replace("-", "_")
    a = _ALIASES.get(a, a)
    if a not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; one of {ALGORITHMS}")
    return a


def logbar(K: int) -> float:
    """logbar(K) = 1/2 + sum_{i=2}^{K} 1/i (Audibert et al. 2010)."""
    return 0.5 + float(np.sum(1.0 / np.arange(2, K + 1)))


def sr_schedule(K: int, budget: int) -> np.ndarray:
    """Cumulative pulls per surviving arm after each phase: n_k, k = 1..K-1 (successive rejects)."""
    if K < 2:
        raise ValueError("successive rejects needs at least 2 arms")
    if budget < K:
        raise ValueError(f"budget {budget} < number of arms {K}")
    k = np.arange(1, K)
    return np.ceil((budget - K) / (logbar(K) * (K + 1 - k)) - 1e-12).astype(int)


def lucb_beta(u, t, K: int, delta: float, scale: float = 1.0):
    """LUCB1 confidence radius beta(u, t) = scale * sqrt(ln(k1 K t^4 / delta) / (2u))."""
    return scale * np.sqrt(np.log(LUCB_K1 * K * np.asarray(t, float) ** 4 / delta) / (2.0 * np.asarray(u, float)))


# ===================================================================== vectorised core

def _argmax_rand(x: np.ndarray, rng) -> np.ndarray:
    """Row-wise argmax with uniform random tie-breaking. x: (R, K)."""
    m = x == x.max(axis=1, keepdims=True)
    return np.argmax(rng.random(x.shape) * m + m, axis=1)


def _means(n, s, empty=-np.inf):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), empty)


class _State:
    """Counts and sums of R independent replications over K arms. draw(rows, arms) -> rewards (maximised)."""

    def __init__(self, draw, R: int, K: int, lo: float = 0.0, hi: float = 1.0):
        self.draw, self.R, self.K = draw, R, K
        self.n = np.zeros((R, K), np.int64)
        self.s = np.zeros((R, K))
        self.lo, self.hi = lo, hi
        self.S = np.zeros((R, K))   # Bernoulli-trial successes (Thompson Beta)

    def pull(self, rows, arms, rng=None, trial=False):
        r = np.asarray(self.draw(rows, arms), float)
        r = np.where(np.isfinite(r), r, -np.inf)
        np.add.at(self.n, (rows, arms), 1)
        np.add.at(self.s, (rows, arms), r)
        if trial:
            y = np.clip((r - self.lo) / (self.hi - self.lo), 0.0, 1.0)
            y = np.where(np.isfinite(r), y, 0.0)
            np.add.at(self.S, (rows, arms), (rng.random(len(y)) < y).astype(float))
        return r

    @property
    def mean(self):
        return _means(self.n, self.s)


def _init_round(st: _State, rows, rng=None, trial=False):
    for a in range(st.K):
        st.pull(rows, np.full(len(rows), a), rng, trial)


def _run_regret(alg: str, st: _State, budget: int, rng, *, scale=1.0, sigma=1.0, prior=(1.0, 1.0),
                prior_mean=0.0, warmup=0) -> dict:
    R, K = st.R, st.K
    rows = np.arange(R)
    t0 = 0
    trial = alg == "thompson_beta"
    w = 1 if alg == "ucb1" else int(warmup)
    for _ in range(w):
        for a in range(K):
            if t0 < budget:
                st.pull(rows, np.full(R, a), rng, trial)
                t0 += 1
    for t in range(t0, budget):
        if alg == "ucb1":
            idx = st.mean + scale * np.sqrt(2.0 * math.log(t) / np.maximum(st.n, 1))
            a = _argmax_rand(idx, rng)
        elif alg == "thompson_beta":
            a = _argmax_rand(rng.beta(st.S + prior[0], st.n - st.S + prior[1]), rng)
        else:
            mu = (prior_mean + st.s) / (st.n + 1)
            a = _argmax_rand(mu + sigma / np.sqrt(st.n + 1) * rng.standard_normal((R, K)), rng)
        st.pull(rows, a, rng, trial)
    return {"stopped": np.array(["budget"] * R, object)}


def _run_sr(st: _State, budget: int, rng) -> dict:
    R, K = st.R, st.K
    sched = sr_schedule(K, budget)
    active = np.ones((R, K), bool)
    order = np.zeros((R, K - 1), int)
    prev = 0
    for k, nk in enumerate(sched):
        m = nk - prev
        prev = nk
        if m > 0:
            rr, aa = np.nonzero(active)
            st.pull(np.repeat(rr, m), np.repeat(aa, m))
        x = np.where(active, st.mean, np.inf)
        drop = _argmax_rand(-x, rng)
        active[np.arange(R), drop] = False
        order[:, k] = drop
    return {"stopped": np.array(["phases"] * R, object), "order": order, "best": np.argmax(active, axis=1)}


def _run_uniform(st: _State, budget: int, rng) -> dict:
    R, K = st.R, st.K
    q, r = divmod(budget, K)
    if q:
        rr, aa = np.nonzero(np.ones((R, K), bool))
        st.pull(np.repeat(rr, q), np.repeat(aa, q))
    if r:
        extra = np.argsort(rng.random((R, K)), axis=1)[:, :r]
        st.pull(np.repeat(np.arange(R), r), extra.ravel())
    return {"stopped": np.array(["budget"] * R, object)}


def _run_lucb(st: _State, budget: int | None, rng, *, delta=0.05, epsilon=0.0, scale=1.0) -> dict:
    R, K = st.R, st.K
    budget = budget if budget is not None else 1000 * K
    if budget < K:
        raise ValueError(f"budget {budget} < number of arms {K}")
    rows = np.arange(R)
    _init_round(st, rows)               # rounds 1..K of the paper
    live = np.ones(R, bool)
    stopped = np.array(["budget"] * R, object)
    best = np.zeros(R, int)
    rounds = np.zeros(R, int)
    t = K
    while live.any():
        t += 1
        lr = np.flatnonzero(live)
        mu = st.mean[lr]
        b = lucb_beta(st.n[lr], t, K, delta, scale)
        h = _argmax_rand(mu, rng)
        ucb = mu + b
        ucb[np.arange(len(lr)), h] = -np.inf
        l = _argmax_rand(ucb, rng)
        i = np.arange(len(lr))
        gap = ucb[i, l] - (mu[i, h] - b[i, h])
        done = gap < epsilon
        out = st.n[lr].sum(1) + 2 > budget
        fin = done | out
        best[lr[fin]] = h[fin]
        stopped[lr[done]] = "confident"
        rounds[lr[fin]] = t - K - 1   # two-arm rounds sampled
        live[lr[fin]] = False
        go = ~fin
        if go.any():
            g = lr[go]
            st.pull(np.concatenate([g, g]), np.concatenate([h[go], l[go]]))
    return {"stopped": stopped, "best": best, "rounds": rounds}


def _run(alg: str, draw, R: int, K: int, budget: int | None, *, seed=0, bounds=(0.0, 1.0), **kw) -> dict:
    alg = canonical(alg)
    if K < 1:
        raise ValueError("need at least one arm")
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    st = _State(draw, R, K, lo, hi)
    if alg != "lucb" and (budget is None or budget < 1):
        raise ValueError(f"{alg} needs a positive budget")
    if K == 1:
        st.pull(np.arange(R), np.zeros(R, int))
        out = {"stopped": np.array(["single arm"] * R, object), "best": np.zeros(R, int)}
    elif alg == "successive_rejects":
        out = _run_sr(st, budget, rng)
    elif alg == "uniform":
        out = _run_uniform(st, budget, rng)
    elif alg == "lucb":
        out = _run_lucb(st, budget, rng, **{k: kw[k] for k in ("delta", "epsilon", "scale") if k in kw})
    else:
        allowed = {"ucb1": ("scale",), "thompson_beta": ("prior", "warmup"),
                   "thompson_gaussian": ("sigma", "prior_mean", "warmup")}[alg]
        out = _run_regret(alg, st, budget, rng, **{k: kw[k] for k in allowed if k in kw})
    if "best" not in out:
        out["best"] = _argmax_rand(st.mean, rng)
    out.update(n=st.n, s=st.s, algorithm=alg)
    return out


# ===================================================================== single run API

@dataclass
class BanditResult:
    algorithm: str
    best: int                       # recommended arm
    pulls: np.ndarray               # (K,) pulls per arm
    means: np.ndarray               # (K,) empirical means in the caller's units (nan = never pulled)
    budget_used: int
    stopped: str                    # "budget" | "phases" | "confident" | "single arm"
    rejected: list = field(default_factory=list)    # successive rejects: dismissal order
    arms: list = field(default_factory=list)        # pulled arm sequence
    rewards: list = field(default_factory=list)     # rewards in the caller's units, same order

    def to_dict(self) -> dict:
        return {"algorithm": self.algorithm, "best": self.best, "pulls": self.pulls.tolist(),
                "means": [None if not np.isfinite(m) else float(m) for m in self.means],
                "budget_used": self.budget_used, "stopped": self.stopped, "rejected": list(self.rejected)}


def identify_best(pull: Callable[[int], float], n_arms: int, budget: int | None, algorithm: str = "successive_rejects",
                  *, higher_is_better: bool = True, seed: int = 0, bounds: tuple | None = None, **kw) -> BanditResult:
    """Run one bandit over `n_arms` arms with `pull(arm) -> reward` for at most `budget` pulls; return the
    recommended arm. `bounds` = (lo, hi) of the reward in the caller's units (default (0, 1); Thompson Beta
    rescaling; also sets `scale` = hi - lo for UCB1 and LUCB unless given). Extra keywords: delta, epsilon (LUCB), scale (UCB1, LUCB),
    sigma, prior_mean (Thompson Gaussian), prior (Thompson Beta)."""
    sign = 1.0 if higher_is_better else -1.0
    arms_log, rew_log = [], []

    def draw(rows, arms):
        out = np.empty(len(arms))
        for i, a in enumerate(arms):
            r = float(pull(int(a)))
            arms_log.append(int(a))
            rew_log.append(r)
            out[i] = sign * r if math.isfinite(r) else -np.inf
        return out

    lo, hi = (0.0, 1.0) if bounds is None else (float(bounds[0]), float(bounds[1]))
    if not hi > lo:
        raise ValueError("bounds must be (lo, hi) with hi > lo")
    b = (lo, hi) if higher_is_better else (-hi, -lo)   # default [0, 1] in the caller's units, also for losses
    if bounds is not None:
        kw.setdefault("scale", hi - lo)
    if canonical(algorithm) == "thompson_gaussian":
        kw.pop("scale", None)
        if not higher_is_better and "prior_mean" in kw:
            kw["prior_mean"] = -kw["prior_mean"]
    out = _run(algorithm, draw, 1, n_arms, budget, seed=seed, bounds=b, **kw)
    n, s = out["n"][0], out["s"][0]
    means = np.where(n > 0, sign * s / np.maximum(n, 1), np.nan)
    return BanditResult(algorithm=out["algorithm"], best=int(out["best"][0]), pulls=n.copy(), means=means,
                        budget_used=int(n.sum()), stopped=str(out["stopped"][0]),
                        rejected=[int(x) for x in out["order"][0]] if "order" in out else [],
                        arms=arms_log, rewards=rew_log)


def ucb1(pull, n_arms, budget, **kw) -> BanditResult:
    return identify_best(pull, n_arms, budget, "ucb1", **kw)


def thompson_beta(pull, n_arms, budget, **kw) -> BanditResult:
    return identify_best(pull, n_arms, budget, "thompson_beta", **kw)


def thompson_gaussian(pull, n_arms, budget, **kw) -> BanditResult:
    return identify_best(pull, n_arms, budget, "thompson_gaussian", **kw)


def successive_rejects(pull, n_arms, budget, **kw) -> BanditResult:
    return identify_best(pull, n_arms, budget, "successive_rejects", **kw)


def lucb(pull, n_arms, budget=None, *, delta: float = 0.05, epsilon: float = 0.0, **kw) -> BanditResult:
    return identify_best(pull, n_arms, budget, "lucb", delta=delta, epsilon=epsilon, **kw)


def uniform(pull, n_arms, budget, **kw) -> BanditResult:
    return identify_best(pull, n_arms, budget, "uniform", **kw)


# ===================================================================== simulation

class SimulatedArms:
    """Synthetic arms with known means. kind "bernoulli" (means in [0, 1]) or "gaussian" (sd `sigma`).
    `arms(i)` draws one reward (a `pull` for identify_best); `draw(rows, arms)` is the vectorised form."""

    def __init__(self, means: Sequence[float], kind: str = "bernoulli", sigma: float = 1.0, seed: int = 0):
        self.means = np.asarray(means, float)
        if kind not in ("bernoulli", "gaussian"):
            raise ValueError("kind must be 'bernoulli' or 'gaussian'")
        if kind == "bernoulli" and ((self.means < 0) | (self.means > 1)).any():
            raise ValueError("Bernoulli means must lie in [0, 1]")
        self.kind, self.sigma = kind, float(sigma)
        self.rng = np.random.default_rng(seed)

    @property
    def K(self) -> int:
        return len(self.means)

    @property
    def best(self) -> int:
        return int(np.argmax(self.means))

    @property
    def gaps(self) -> np.ndarray:
        return self.means.max() - self.means

    def draw(self, rows, arms) -> np.ndarray:
        mu = self.means[np.asarray(arms, int)]
        if self.kind == "bernoulli":
            return (self.rng.random(len(mu)) < mu).astype(float)
        return mu + self.sigma * self.rng.standard_normal(len(mu))

    def __call__(self, arm: int) -> float:
        return float(self.draw(None, [arm])[0])

    def hardness(self) -> dict:
        """H1 = sum_i gap_i^-2 and H2 = max_i i * gap_(i)^-2 over all K arms, the best arm's gap taken as the
        smallest other gap (Audibert et al. 2010 §2)."""
        g = np.sort(self.gaps)[1:]
        if (g <= 0).any():
            return {"H1": math.inf, "H2": math.inf}
        g2 = np.concatenate([[g[0]], g])  # gap_(1) = gap_(2) by convention
        return {"H1": float(np.sum(g2 ** -2.0)), "H2": float(np.max(np.arange(1, len(g2) + 1) * g2 ** -2.0))}


def simulate(arms: SimulatedArms, algorithm: str, budget: int | None, *, reps: int = 1000, seed: int = 0,
             chunk: int = 2000, **kw) -> dict:
    """Run `reps` independent replications (vectorised over replications) and report the probability of error
    (recommended arm is not the best), its binomial standard error, mean pulls used, mean simple regret
    (gap of the recommended arm) and mean cumulative pseudo-regret (sum of pulls x gap)."""
    if arms.K > 1 and int(np.sum(arms.means == arms.means.max())) > 1:
        raise ValueError("the best arm must be unique for an identification rate")
    errs, used, simple, cum, conf = [], [], [], [], []
    for c, r0 in enumerate(range(0, reps, chunk)):
        R = min(chunk, reps - r0)
        out = _run(algorithm, arms.draw, R, arms.K, budget, seed=seed + 7919 * c, **kw)
        b = out["best"]
        errs.append(b != arms.best)
        used.append(out["n"].sum(1))
        simple.append(arms.gaps[b])
        cum.append(out["n"] @ arms.gaps)
        conf.append(out["stopped"] == "confident")
    e = np.concatenate(errs)
    p = float(e.mean())
    return {"algorithm": canonical(algorithm), "K": arms.K, "budget": budget, "reps": reps,
            "error_rate": p, "error_se": math.sqrt(p * (1 - p) / reps),
            "mean_pulls": float(np.concatenate(used).mean()), "simple_regret": float(np.concatenate(simple).mean()),
            "cumulative_regret": float(np.concatenate(cum).mean()),
            "confident_rate": float(np.concatenate(conf).mean())}


def compare(arms: SimulatedArms, budget: int | None, algorithms: Sequence[str] = ALGORITHMS, *, reps: int = 1000,
            seed: int = 0, params: dict[str, dict] | None = None):
    """simulate() for several algorithms on the same arms; returns a DataFrame (one row per algorithm)."""
    import pandas as pd
    rows = []
    for a in algorithms:
        rows.append(simulate(arms, a, budget, reps=reps, seed=seed, **(params or {}).get(a, {})))
    return pd.DataFrame(rows)


# ===================================================================== Loop proposer

class RacingProposer:
    """Race the single-change neighbours of the current best with a bandit and propose the winner.

    Candidates are the grid neighbours of the incumbent (as HalvingProposer), at most `n_candidates`, drawn with
    default_rng(seed + turn). Pull j (0-based) of a candidate config is `pull(cfg, j)` if given, else one loop
    screening run `ctx.screen([cfg], fidelity_of(j))`: by default the runner gets cfg["fidelity"] = j + 1 and should
    return the score on mini-batch j + 1 (a disjoint unit subset or a fresh seed). The loop caches every screening run
    by (config, fidelity), so a resumed loop does not re-run them. `budget` (default pulls_per_arm x K) caps the
    pulls per turn. The proposal is then judged by the loop on the full dev split as usual; every pull is an extra look
    at dev."""
    name = "racing"
    requires_fields = False

    def __init__(self, algorithm: str = "successive_rejects", *, budget: int | None = None, pulls_per_arm: int = 4,
                 n_candidates: int = 9, n_grid: int = 5, pull: Callable[[dict, int], float] | None = None,
                 fidelity_of: Callable[[int], float] | None = None, seed: int = 0, **algo_kw):
        self.algorithm = canonical(algorithm)
        self.budget, self.pulls_per_arm = budget, pulls_per_arm
        self.n_candidates, self.n_grid, self.seed = n_candidates, n_grid, seed
        self.pull, self.fidelity_of = pull, fidelity_of or (lambda j: float(j + 1))
        self.algo_kw = algo_kw
        self.evaluations = 0
        self.last: BanditResult | None = None

    def candidates(self, ctx) -> list:
        cands = []
        for k in ctx.space.used(ctx.incumbent["cfg"]):
            for v in ctx.space.grid_values(k, self.n_grid):
                if ctx.candidate(k, v) is not None:
                    cands.append((k, v))
        rng = np.random.default_rng(self.seed + ctx.turn)
        if len(cands) > self.n_candidates:
            cands = [cands[i] for i in sorted(rng.choice(len(cands), self.n_candidates, replace=False))]
        return cands

    def propose(self, ctx):
        cands = self.candidates(ctx)
        if not cands:
            return None
        if len(cands) == 1 or (self.pull is None and ctx.screen is None):
            (k, v), = cands[:1]
            return {"change": {k: v}, "kind": "racing", "hypothesis": "-", "expected": "-", "falsifier": "-",
                    "rationale": "single candidate" if len(cands) == 1 else "no screening available"}
        cfgs = [ctx.candidate(k, v) for k, v in cands]
        seen = [0] * len(cfgs)

        def arm(i):
            j = seen[i]
            seen[i] += 1
            if self.pull is not None:
                return float(self.pull(cfgs[i], j))
            return float(ctx.screen([cfgs[i]], self.fidelity_of(j))[0])

        K = len(cfgs)
        budget = self.budget or self.pulls_per_arm * K
        res = identify_best(arm, K, budget, self.algorithm, higher_is_better=ctx.higher_is_better,
                            seed=self.seed + ctx.turn, **self.algo_kw)
        self.evaluations += res.budget_used
        self.last = res
        k, v = cands[res.best]
        return {"change": {k: v}, "kind": "racing", "hypothesis": "-", "expected": "-", "falsifier": "-",
                "rationale": f"won a {self.algorithm} race over {K} candidates ({res.budget_used} pulls, "
                             f"stopped: {res.stopped})",
                "notes": {"race": {"candidates": [{k_: v_} for k_, v_ in cands], **res.to_dict()}}}


__all__ = ["ALGORITHMS", "AUDIBERT_BUBECK_2010", "BanditResult", "RacingProposer", "SimulatedArms", "canonical",
           "compare", "identify_best", "logbar", "lucb", "lucb_beta", "simulate", "sr_schedule", "successive_rejects",
           "thompson_beta", "thompson_gaussian", "ucb1", "uniform"]
