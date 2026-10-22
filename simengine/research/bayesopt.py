"""Bayesian optimisation over a research `Space`: a Gaussian-process surrogate with expected improvement or GP-UCB,
and TPE-lite (tree-structured Parzen estimator, Bergstra, Bardenet, Bengio & Kegl, NIPS 2011, §4).

    from simengine.research import Loop, Space
    from simengine.research.bayesopt import GPProposer, TPEProposer, branin, minimize
    loop = Loop(runner, space, proposer=GPProposer(acquisition="ei", seed=0), primary="err",
                higher_is_better=False, stop_after=None, max_turns=20)
    res = minimize(lambda c: branin(c["x1"], c["x2"]), Space({"x1": (-5.0, 10.0), "x2": (0.0, 15.0)}),
                   method="tpe", n_calls=30, seed=0)
    res["best"], res["best_cfg"], res["trace"]          # best loss, its config, best-so-far per call

Encoding (`SpaceEncoder`): every non-fixed key of the space; a (lo, hi) range is scaled to [0, 1] (an int range is
rounded back on decode), a list of choices is one-hot (forbidden values left out). Losses are always minimised: a
higher-is-better metric is negated.

Two ways to use it:
- Proposers for `Loop` (`GPProposer`, `TPEProposer`): the Loop contract is ONE changed key per turn from the current
  best, so the acquisition is maximised over one-change neighbours of the incumbent (all choices of a categorical key,
  random + grid values of a range). The surrogate is fitted on every finished turn's dev primary mean; a crashed turn
  counts as the worst loss seen. The first `n_init` turns are seeded random single changes (as `RandomProposer`).
- `minimize(objective, space, method="gp" | "tpe" | "random")`: plain sequential optimisation where every key moves,
  the setting of the BO literature; used for the Branin and mixed-toy numbers in docs/modules/research/bayesopt.md.

GP: scikit-learn `GaussianProcessRegressor`, Constant * Matern(nu=2.5, one length scale per encoded column) + White
noise, `normalize_y=True`. EI is the closed form under the Gaussian posterior with incumbent y* = min loss seen
(Bergstra et al. 2011 §3.1); UCB picks argmax mu + sqrt(beta_t) sigma on the negated loss (Srinivas et al., ICML 2010,
Algorithm 1), beta_t = 2 log(|D| t^2 pi^2 / (6 delta)) of their Theorem 1 with |D| = number of candidates and
t = observations + 1 (round t uses the posterior after t - 1 points), scaled by 1/5 as in their experiments (§6), unless
`beta` is given.
TPE (§4.1-4.2 and §6 of the paper): split the observations at the gamma quantile (gamma = 0.15); l(x) = Parzen density
of the good part, g(x) of the rest; for a range, an equally weighted mixture of the (uniform) prior and Gaussians at each
point with sd = the larger distance to the left / right neighbour (the bounds count as neighbours), clipped; for a
choice, probabilities proportional to N p_i + C_i; draw `n_candidates` (100) from l and take the largest l(x)/g(x).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from .proposers import ProposalContext, _mk
from .space import Space, _plain, config_hash

BRANIN_MIN = 0.397887  # global minimum of branin() on [-5, 10] x [0, 15]


# ===================================================================== test functions

def branin(x1, x2):
    """Branin-Hoo with the usual constants (a=1, b=5.1/(4 pi^2), c=5/pi, r=6, s=10, t=1/(8 pi)); vectorised.
    Three global minima of 0.397887 at (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    b, c, t = 5.1 / (4 * np.pi ** 2), 5 / np.pi, 1 / (8 * np.pi)
    return (x2 - b * x1 ** 2 + c * x1 - 6) ** 2 + 10 * (1 - t) * np.cos(x1) + 10


def branin_space() -> Space:
    return Space({"x1": (-5.0, 10.0), "x2": (0.0, 15.0)}, defaults={"x1": 2.5, "x2": 7.5})


_TOY_OFFSET = {"a": 0.6, "b": 0.0, "c": 0.3, "d": 0.9}
_TOY_CENTER = {"a": 0.2, "b": 0.75, "c": 0.4, "d": 0.5}


def mixed_toy(cfg: dict) -> float:
    """Mixed categorical/numeric toy loss (minimum 0 at method='b', x=0.75, k=6). The best x depends on the method,
    so a model that treats keys independently is misled."""
    m = cfg["method"]
    return _TOY_OFFSET[m] + 3.0 * (cfg["x"] - _TOY_CENTER[m]) ** 2 + 0.01 * (cfg["k"] - 6) ** 2


def mixed_toy_space() -> Space:
    return Space({"method": ["a", "b", "c", "d"], "x": (0.0, 1.0), "k": (1, 16)},
                 defaults={"method": "a", "x": 0.5, "k": 12})


# ===================================================================== encoding

class SpaceEncoder:
    """Unit-cube / one-hot encoding of the non-fixed keys of a `Space`."""

    def __init__(self, space: Space, digits: int = 6):
        self.space, self.digits = space, digits
        allowed = space.allowed()
        self.keys = list(allowed)
        self.num = [k for k in self.keys if isinstance(space.params[k], tuple)]
        self.cat = {k: list(allowed[k]) for k in self.keys if isinstance(space.params[k], list)}
        self.slices, j = {}, 0
        for k in self.keys:
            w = len(self.cat[k]) if k in self.cat else 1
            self.slices[k] = slice(j, j + w)
            j += w
        self.dim = j

    # numeric part in [0, 1] (n, len(num)) and categorical indices (n, len(cat)); -1 = value not allowed / missing
    def unit(self, cfgs: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
        U = np.empty((len(cfgs), len(self.num)))
        for j, k in enumerate(self.num):
            lo, hi = self.space.params[k]
            v = np.array([float(c[k]) for c in cfgs]) if cfgs else np.empty(0)
            U[:, j] = (v - lo) / (hi - lo) if hi > lo else 0.5
        C = np.full((len(cfgs), len(self.cat)), -1, int)
        for j, (k, ch) in enumerate(self.cat.items()):
            pos = {v: i for i, v in enumerate(ch)}
            C[:, j] = [pos.get(c.get(k), -1) for c in cfgs]
        return U, C

    def encode(self, cfgs: Sequence[dict]) -> np.ndarray:
        U, C = self.unit(cfgs)
        X = np.zeros((len(cfgs), self.dim))
        for j, k in enumerate(self.num):
            X[:, self.slices[k].start] = U[:, j]
        rows = np.arange(len(cfgs))
        for j, k in enumerate(self.cat):
            ok = C[:, j] >= 0
            X[rows[ok], self.slices[k].start + C[ok, j]] = 1.0
        return X

    def value(self, key: str, u: float):
        """Range value from a unit coordinate (clipped; ints rounded, floats rounded to `digits`)."""
        lo, hi = self.space.params[key]
        v = lo + float(np.clip(u, 0.0, 1.0)) * (hi - lo)
        if self.space.is_int_range(key):
            return int(min(max(round(v), lo), hi))
        return round(v, self.digits)

    def from_unit(self, U: np.ndarray, C: np.ndarray, base: dict | None = None) -> list[dict]:
        out = []
        for i in range(len(U)):
            cfg = dict(base or {})
            for j, k in enumerate(self.num):
                cfg[k] = self.value(k, U[i, j])
            for j, (k, ch) in enumerate(self.cat.items()):
                cfg[k] = _plain(ch[int(C[i, j])])
            out.append(cfg)
        return out

    def sample(self, n: int, rng: np.random.Generator, base: dict | None = None) -> list[dict]:
        """n uniform random configs (every encoded key drawn; `base` fills the rest)."""
        U = rng.uniform(size=(n, len(self.num)))
        C = np.column_stack([rng.integers(len(ch), size=n) for ch in self.cat.values()]) if self.cat \
            else np.empty((n, 0), int)
        return self.from_unit(U, C, base)


# ===================================================================== observations

def observations(history: Sequence[dict], primary: str, higher_is_better: bool,
                 crash: str = "worst") -> tuple[list[dict], np.ndarray]:
    """(configs, losses) from Loop turn records (dev primary mean; loss = -metric when higher is better).
    A crashed / non-finite turn gets the worst finite loss seen (`crash="worst"`) or is dropped (`"skip"`)."""
    cfgs, ys, bad = [], [], []
    for t in history:
        v = ((t.get("metrics") or {}).get(primary) or {}).get("value") if t.get("status") != "crash" else None
        v = float(v) if v is not None else float("nan")
        if not math.isfinite(v):
            bad.append(t["cfg"])
            continue
        cfgs.append(t["cfg"])
        ys.append(-v if higher_is_better else v)
    if bad and crash == "worst" and ys:
        worst = max(ys)
        cfgs += bad
        ys += [worst] * len(bad)
    return cfgs, np.asarray(ys, float)


# ===================================================================== GP surrogate + acquisitions

def fit_gp(X: np.ndarray, y: np.ndarray, *, seed: int = 0, n_restarts: int = 2):
    """GaussianProcessRegressor, Constant * Matern(nu=2.5, ARD) + White, normalize_y; ConvergenceWarnings muted."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    d = X.shape[1]
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(d, 0.5), length_scale_bounds=(1e-2, 1e2),
                                                         nu=2.5) + WhiteKernel(1e-3, (1e-6, 1.0)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=n_restarts, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)
    return gp


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.0) -> np.ndarray:
    """EI for minimisation: E[max(best - xi - f, 0)] under f ~ N(mu, sigma^2)."""
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    imp = best - xi - mu
    with np.errstate(divide="ignore", invalid="ignore"):
        z = imp / sigma
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    return np.where(sigma > 1e-12, ei, np.maximum(imp, 0.0))


def ucb_beta(t: int, n_candidates: int, delta: float = 0.1, scale: float = 0.2) -> float:
    """beta_t = 2 log(|D| t^2 pi^2 / (6 delta)) (Srinivas et al. 2010, Theorem 1), times `scale` (their §6: 1/5)."""
    return scale * 2.0 * math.log(max(n_candidates, 1) * max(t, 1) ** 2 * math.pi ** 2 / (6 * delta))


def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray, beta: float) -> np.ndarray:
    """GP-UCB on the negated loss: -mu + sqrt(beta) sigma (larger = propose)."""
    return -np.asarray(mu, float) + math.sqrt(beta) * np.asarray(sigma, float)


def gp_scores(enc: SpaceEncoder, cfgs: Sequence[dict], losses: np.ndarray, candidates: Sequence[dict], *,
              acquisition: str = "ei", xi: float = 0.01, beta: float | None = None, t: int | None = None,
              seed: int = 0, n_restarts: int = 2) -> dict:
    """Fit the GP on (cfgs, losses) and score candidates. Returns {"acq", "mu", "sigma", "beta"} arrays.
    `xi` is in units of the loss standard deviation; `t` is the UCB round (default: number of observations + 1)."""
    gp = fit_gp(enc.encode(cfgs), losses, seed=seed, n_restarts=n_restarts)
    mu, sigma = gp.predict(enc.encode(candidates), return_std=True)
    if acquisition == "ei":
        acq = expected_improvement(mu, sigma, float(losses.min()), xi * float(np.std(losses) or 1.0))
        b = None
    elif acquisition == "ucb":
        b = beta if beta is not None else ucb_beta(t or len(losses) + 1, len(candidates))
        acq = upper_confidence_bound(mu, sigma, b)
    else:
        raise ValueError(f"acquisition must be 'ei' or 'ucb', got {acquisition!r}")
    return {"acq": acq, "mu": mu, "sigma": sigma, "beta": b}


# ===================================================================== TPE-lite

def _trunc_logpdf(u: np.ndarray, mus: np.ndarray, sds: np.ndarray, logw: np.ndarray) -> np.ndarray:
    """log density on [0, 1] of a mixture of Gaussians truncated to [0, 1]; u (n,), components (m,)."""
    mass = norm.cdf((1 - mus) / sds) - norm.cdf((0 - mus) / sds)
    lp = norm.logpdf(u[:, None], mus[None, :], sds[None, :]) - np.log(np.maximum(mass, 1e-300))[None, :]
    return logsumexp(lp + logw[None, :], axis=1)


@dataclass
class Parzen1D:
    """Adaptive Parzen estimator of one range key in unit coordinates (Bergstra et al. 2011 §4.2): an equally weighted
    mixture of the prior N(0.5, 1) and one Gaussian per point, everything truncated to [0, 1]."""
    mus: np.ndarray
    sds: np.ndarray

    @classmethod
    def fit(cls, pts: np.ndarray, prior_sd: float = 1.0) -> "Parzen1D":
        p = np.sort(np.clip(np.asarray(pts, float), 0.0, 1.0))
        k = len(p)
        if k:
            ext = np.concatenate([[0.0], p, [1.0]])            # the bounds count as neighbours
            sd = np.maximum(ext[1:-1] - ext[:-2], ext[2:] - ext[1:-1])
            sd = np.clip(sd, prior_sd / min(100.0, 1.0 + k), prior_sd)
        else:
            sd = np.empty(0)
        return cls(np.concatenate([[0.5], p]), np.concatenate([[prior_sd], sd]))

    def logpdf(self, u: np.ndarray) -> np.ndarray:
        return _trunc_logpdf(np.asarray(u, float), self.mus, self.sds, np.full(len(self.mus), -math.log(len(self.mus))))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        from scipy.stats import truncnorm
        j = rng.integers(len(self.mus), size=n)
        mu, sd = self.mus[j], self.sds[j]
        return truncnorm.rvs((0 - mu) / sd, (1 - mu) / sd, loc=mu, scale=sd, size=n, random_state=rng)


def _cat_probs(idx: np.ndarray, n_choices: int) -> np.ndarray:
    """Posterior over choices proportional to N p_i + C_i with a uniform prior p_i = 1/N (= 1 + C_i)."""
    c = np.bincount(idx[idx >= 0], minlength=n_choices).astype(float)
    return (1.0 + c) / (n_choices + c.sum())


class TPE:
    """l(x) / g(x) densities over an encoded space, one independent factor per key (no conditional keys here)."""

    def __init__(self, enc: SpaceEncoder, cfgs: Sequence[dict], losses: np.ndarray, gamma: float = 0.15):
        self.enc, self.gamma = enc, gamma
        order = np.argsort(np.asarray(losses, float), kind="stable")
        n_below = max(1, int(math.ceil(gamma * len(order))))
        U, C = enc.unit(list(cfgs))
        lo, hi = order[:n_below], order[n_below:]
        self.l_num = [Parzen1D.fit(U[lo, j]) for j in range(U.shape[1])]
        self.g_num = [Parzen1D.fit(U[hi, j]) for j in range(U.shape[1])]
        self.l_cat = [_cat_probs(C[lo, j], len(ch)) for j, ch in enumerate(enc.cat.values())]
        self.g_cat = [_cat_probs(C[hi, j], len(ch)) for j, ch in enumerate(enc.cat.values())]
        self.n_below, self.n_above = len(lo), len(hi)

    def score(self, cfgs: Sequence[dict]) -> np.ndarray:
        """log l(x) - log g(x); larger is better (EI is increasing in l/g, §4.1)."""
        U, C = self.enc.unit(list(cfgs))
        s = np.zeros(len(cfgs))
        for j in range(U.shape[1]):
            s += self.l_num[j].logpdf(U[:, j]) - self.g_num[j].logpdf(U[:, j])
        for j in range(C.shape[1]):
            ok = C[:, j] >= 0
            s[ok] += np.log(self.l_cat[j][C[ok, j]]) - np.log(self.g_cat[j][C[ok, j]])
        return s

    def sample_key(self, key: str, n: int, rng: np.random.Generator) -> list:
        """n values of one key drawn from l(x)."""
        if key in self.enc.cat:
            j = list(self.enc.cat).index(key)
            ch = self.enc.cat[key]
            return [_plain(ch[i]) for i in rng.choice(len(ch), size=n, p=self.l_cat[j])]
        j = self.enc.num.index(key)
        return [self.enc.value(key, u) for u in self.l_num[j].sample(n, rng)]

    def sample(self, n: int, rng: np.random.Generator, base: dict | None = None) -> list[dict]:
        """n full configs drawn from l(x) (every key independently)."""
        U = np.column_stack([p.sample(n, rng) for p in self.l_num]) if self.l_num else np.empty((n, 0))
        C = np.column_stack([rng.choice(len(p), size=n, p=p) for p in self.l_cat]) if self.l_cat \
            else np.empty((n, 0), int)
        return self.enc.from_unit(U, C, base)


# ===================================================================== Loop proposers (one change per turn)

def _neighbours(ctx: ProposalContext, enc: SpaceEncoder, rng: np.random.Generator, n_per_key: int,
                n_grid: int = 5) -> list[tuple[str, Any, dict]]:
    """Valid, unseen one-change neighbours (key, value, full cfg) of the incumbent: every choice of a categorical key;
    n_per_key uniform draws plus the grid of a range (all values of a small int range)."""
    out, seen = [], set()
    for k in ctx.space.used(ctx.incumbent["cfg"]):
        if k not in enc.slices:
            continue
        if k in enc.cat:
            vals = list(enc.cat[k])
        elif ctx.space.is_int_range(k) and ctx.space.params[k][1] - ctx.space.params[k][0] + 1 <= n_per_key:
            lo, hi = ctx.space.params[k]
            vals = list(range(lo, hi + 1))
        else:
            vals = [enc.value(k, u) for u in rng.uniform(size=n_per_key)] + ctx.space.grid_values(k, n_grid)
        for v in vals:
            v = _plain(v)
            if (k, v) in seen:
                continue
            seen.add((k, v))
            full = ctx.candidate(k, v)
            if full is not None:
                out.append((k, v, full))
    return out


class _BOProposer:
    name = "bayesopt"
    requires_fields = False

    def __init__(self, *, n_init: int | None = None, seed: int = 0, crash: str = "worst"):
        self.n_init, self.seed, self.crash = n_init, seed, crash

    def _init_n(self, enc: SpaceEncoder) -> int:
        return self.n_init if self.n_init is not None else max(3, min(10, enc.dim + 1))

    def _warmup(self, cands, rng, n_obs):
        k, v, _ = cands[int(rng.integers(len(cands)))]
        return _mk({k: v}, "random-init", f"warm-up {n_obs} observations < n_init; seed {self.seed}")

    def propose(self, ctx: ProposalContext):
        enc = SpaceEncoder(ctx.space)
        rng = np.random.default_rng(self.seed + 7919 * ctx.turn)
        cfgs, losses = observations(ctx.history, ctx.primary, ctx.higher_is_better, self.crash)
        if len(losses) < self._init_n(enc):
            cands = _neighbours(ctx, enc, rng, 64)
            return self._warmup(cands, rng, len(losses)) if cands else None
        return self._propose(ctx, enc, rng, cfgs, losses)


class GPProposer(_BOProposer):
    """GP surrogate (Matern 5/2 ARD, one-hot categoricals) + EI or GP-UCB, maximised over one-change neighbours of
    the current best (`n_per_key` random values per range key + the grid; every choice of a categorical)."""
    name = "gp"

    def __init__(self, acquisition: str = "ei", *, xi: float = 0.01, beta: float | None = None, n_per_key: int = 256,
                 n_init: int | None = None, seed: int = 0, n_restarts: int = 2, crash: str = "worst"):
        super().__init__(n_init=n_init, seed=seed, crash=crash)
        self.acquisition, self.xi, self.beta, self.n_per_key, self.n_restarts = acquisition, xi, beta, n_per_key, n_restarts

    def _propose(self, ctx, enc, rng, cfgs, losses):
        cands = _neighbours(ctx, enc, rng, self.n_per_key)
        if not cands:
            return None
        sc = gp_scores(enc, cfgs, losses, [c for _, _, c in cands], acquisition=self.acquisition, xi=self.xi,
                       beta=self.beta, t=len(losses) + 1, seed=self.seed + ctx.turn, n_restarts=self.n_restarts)
        i = int(np.argmax(sc["acq"]))
        k, v, _ = cands[i]
        sign = -1.0 if ctx.higher_is_better else 1.0
        p = _mk({k: v}, self.acquisition, f"GP-{self.acquisition.upper()} argmax over {len(cands)} one-change "
                                          f"neighbours, {len(losses)} observations")
        p["notes"] = {"acq": float(sc["acq"][i]), "pred_mean": float(sign * sc["mu"][i]),
                      "pred_sd": float(sc["sigma"][i]), "n_obs": len(losses), "n_candidates": len(cands)}
        return p


class TPEProposer(_BOProposer):
    """TPE-lite: l/g Parzen densities (gamma quantile split); `n_candidates` one-change values drawn from l (spread
    over the used keys), the one with the largest l(x)/g(x) of the full config is proposed."""
    name = "tpe"

    def __init__(self, gamma: float = 0.15, *, n_candidates: int = 100, n_init: int | None = None, seed: int = 0,
                 crash: str = "worst"):
        super().__init__(n_init=n_init, seed=seed, crash=crash)
        self.gamma, self.n_candidates = gamma, n_candidates

    def _propose(self, ctx, enc, rng, cfgs, losses):
        tpe = TPE(enc, cfgs, losses, self.gamma)
        keys = [k for k in ctx.space.used(ctx.incumbent["cfg"]) if k in enc.slices]
        if not keys:
            return None
        m = max(1, int(math.ceil(self.n_candidates / len(keys))))
        cands, seen = [], set()
        for k in keys:
            for v in tpe.sample_key(k, m, rng):
                if (k, v) in seen:
                    continue
                seen.add((k, v))
                full = ctx.candidate(k, v)
                if full is not None:
                    cands.append((k, v, full))
        if not cands:                       # l(x) only hits seen configs: fall back to the uniform neighbourhood
            cands = _neighbours(ctx, enc, rng, m)
            if not cands:
                return None
        s = tpe.score([c for _, _, c in cands])
        i = int(np.argmax(s))
        k, v, _ = cands[i]
        p = _mk({k: v}, "tpe", f"TPE argmax l/g over {len(cands)} draws from l; gamma {self.gamma}, "
                               f"{tpe.n_below} good / {tpe.n_above} other observations")
        p["notes"] = {"log_l_over_g": float(s[i]), "n_obs": len(losses), "n_candidates": len(cands)}
        return p


# ===================================================================== full-space sequential optimisation

def _local(enc: SpaceEncoder, cfgs, losses, n, rng, sd=0.05, top=5):
    """Gaussian perturbations (unit sd) of the numeric keys of the `top` best configs; categoricals kept."""
    if n <= 0 or not enc.num:
        return []
    best = np.argsort(losses, kind="stable")[:top]
    U, C = enc.unit([cfgs[i] for i in best])
    j = rng.integers(len(best), size=n)
    return enc.from_unit(np.clip(U[j] + rng.normal(0, sd, size=(n, U.shape[1])), 0, 1), C[j])


def minimize(objective: Callable[[dict], float], space: Space, *, method: str = "gp", n_calls: int = 30,
             n_init: int | None = None, seed: int = 0, acquisition: str = "ei", n_candidates: int | None = None,
             gamma: float = 0.15, xi: float = 0.01, beta: float | None = None, higher_is_better: bool = False,
             x0: Sequence[dict] = (), tries: int = 50) -> dict:
    """Sequential optimisation of objective(cfg) -> float over full configs (every non-fixed key moves).

    method "random" (uniform draws), "gp" (EI / UCB over `n_candidates` = 2000 uniform + 25% local perturbations of the
    5 best), "tpe" (best l/g of `n_candidates` = 100 draws from l). The first `n_init` calls (`x0`, then uniform draws)
    are shared by all methods under the same seed. A call that raises or returns a non-finite value is recorded with
    value NaN; the surrogate sees it as the worst finite loss seen so far (until a finite loss exists, proposals stay
    uniform). Returns {"best", "best_cfg", "history": [{"cfg", "value", "method"}],
    "trace": best-so-far value per call}."""
    if method not in ("gp", "tpe", "random"):
        raise ValueError(f"method must be gp, tpe or random, got {method!r}")
    enc = SpaceEncoder(space)
    rng = np.random.default_rng(seed)
    base, err = space.validate({})
    base = {k: v for k, v in base.items() if k not in enc.slices}
    if n_init is None:
        n_init = 10 if method == "tpe" else max(3, min(10, 2 * enc.dim + 1))
    sign = -1.0 if higher_is_better else 1.0
    hist, cfgs, raw, hashes = [], [], [], set()      # raw: signed losses, NaN for a crash

    def valid(c):
        full, e = space.validate({**base, **c})
        return None if e or config_hash(full) in hashes else full

    def pick_random():
        for _ in range(tries):
            c = valid(enc.sample(1, rng)[0])
            if c is not None:
                return c, "random"
        return None, "random"

    queue = list(x0)
    for i in range(n_calls):
        cfg, how = None, method
        while queue and cfg is None:
            cfg = valid(queue.pop(0))
            how = "x0" if cfg is not None else method
        R = np.asarray(raw, float)
        fin = np.isfinite(R)
        if cfg is None and (method == "random" or len(raw) < n_init or not fin.any()):
            cfg, how = pick_random()
        elif cfg is None:
            L = np.where(fin, R, R[fin].max())          # crashes = worst finite loss seen so far
            if method == "gp":
                n = n_candidates or 2000
                cand = enc.sample(n, rng) + _local(enc, cfgs, L, n // 4, rng)
                cand = [c for c in (valid(c) for c in cand) if c is not None]
                if cand:
                    acq = gp_scores(enc, cfgs, L, cand, acquisition=acquisition, xi=xi, beta=beta, t=len(L) + 1,
                                    seed=seed + i)["acq"]
                    cfg = cand[int(np.argmax(acq))]
            else:
                tpe = TPE(enc, cfgs, L, gamma)
                cand = [c for c in (valid(c) for c in tpe.sample(n_candidates or 100, rng)) if c is not None]
                if cand:
                    cfg = cand[int(np.argmax(tpe.score(cand)))]
            if cfg is None:
                cfg, how = pick_random()
        if cfg is None:
            break
        try:
            v = float(objective(dict(cfg)))
        except Exception:
            v = float("nan")
        hashes.add(config_hash(cfg))
        hist.append({"cfg": cfg, "value": v, "method": how})
        cfgs.append(cfg)
        raw.append(sign * v if math.isfinite(v) else float("nan"))
    vals = np.array([h["value"] for h in hist], float)
    ok = np.isfinite(vals)
    if not ok.any():
        return {"best": None, "best_cfg": None, "history": hist, "trace": []}
    sv = np.where(ok, sign * vals, np.inf)
    run = np.minimum.accumulate(sv)
    trace = [float(sign * x) if math.isfinite(x) else float("nan") for x in run]
    b = int(np.argmin(sv))
    return {"best": float(vals[b]), "best_cfg": hist[b]["cfg"], "history": hist, "trace": trace}


__all__ = ["SpaceEncoder", "observations", "fit_gp", "expected_improvement", "ucb_beta", "upper_confidence_bound",
           "gp_scores", "Parzen1D", "TPE", "GPProposer", "TPEProposer", "minimize", "branin", "branin_space",
           "BRANIN_MIN", "mixed_toy", "mixed_toy_space"]
