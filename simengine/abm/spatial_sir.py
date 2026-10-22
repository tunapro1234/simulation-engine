"""Spatial SIR / SEIR on a 2-D grid, its mean-field ODE control, R0 estimators, and an LLM-agent isolation variant.

Grid engine (:class:`SpatialSEIR`): n agents on an H x W lattice, a whole BATCH of replicates (B, n) per call.
Contact is frequency dependent and local: a susceptible agent in cell c has hazard
``beta * I_box(c) / N_box(c)`` per day, where ``*_box`` counts present agents in the Moore window of radius r around
c (own cell included; r = 0 is same-cell only, r = None is full mixing, the mean-field control). Positions are
either static home cells ``(n,)`` (contact via the neighbourhood) or mobility trajectories ``(n, T)`` (one cell per
time slot; contact via co-location in the window at each slot, ``steps_per_day`` = slots per day, trajectory index
``t % T``). Transitions are per step with probabilities ``1 - exp(-rate * dt)`` (exponential latent / infectious
periods), so with r = None and enough steps per day the ABM mean converges to :func:`seir_ode`.

R0: :func:`r0_index_case` (one infectious agent in a susceptible population, secondary cases made absorbing, so the
count is the realised R0 of that contact structure), :func:`r0_growth` (early exponential growth rate r ->
R0 = (1 + r/gamma)(1 + r/sigma), exponential periods; Wallinga & Lipsitch 2007), :func:`r0_final_size`
(-ln(1-z)/z from the attack rate z) and :func:`final_size` (the inverse, z = 1 - exp(-R0 z)).

Generative agents (:class:`GenerativeEpidemic`): the design of Williams, Hosseinichimeh, Majumdar & Ghaffarzadegan
2023, "Epidemic modeling with generative agents" (arXiv:2307.04986): each day every agent gets a persona prompt (age,
Big Five traits, a bio with a job to go to) plus memories set by the condition (``base``; ``self_health``: own
symptoms; ``full``: own symptoms + yesterday's share of new cases in the town) and is asked whether to stay home for
the entire day ("Reasoning: ... / Response: Yes|No"; a non-conforming reply counts as "No", as in the paper).
Agents who go out each contact ``contact_rate`` agents among those also out; an infectious agent infects a
susceptible contact with probability ``infectivity``. Disease: 6 days, days 1-2 normal, 3 and 6 light cough, 4-5
fever and cough, then immune; the news number is the share of agents on day 4 of infection. ``message=`` adds a
line to every agent's memories (e.g. a public health appeal), so the effect of a message on isolation and on the
epidemic curve can be simulated. :func:`rule_responder` is a FakeLLM responder with a fixed rule (offline control).

Usage::

    import simengine.abm.spatial_sir as ep
    p = ep.EpiParams.from_r0(2.5, infectious_days=5, latent_days=3)
    ode = ep.seir_ode(p, n=10_000, i0=10, days=160)                            # day, S, E, I, R, new
    m = ep.SpatialSEIR(p, shape=(100, 100), n=10_000, radius=None, steps_per_day=4)   # well mixed
    run = m.run(days=160, i0=10, reps=20, seed=1)
    ep.ode_agreement(run, ode)              # attack rate, peak share / day, curve RMSE vs the ODE
    ep.r0_index_case(ep.SpatialSEIR(p, shape=(100, 100), n=10_000, radius=1), reps=400)

    from simengine.llm import FakeLLM
    g = ep.GenerativeEpidemic(FakeLLM(ep.rule_responder(news_threshold=1.0)), n=100, condition="full",
                              message="The mayor asks everyone with a cough to stay home.")
    res = g.run(days=60)                    # res.daily (S, I, R, new cases, mobility), res.log (one row per decision)

Measured agreement numbers: docs/modules/abm/epidemic.md.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

S, E, I, R = 0, 1, 2, 3
ENV_DATA = "SIMENGINE_DATA"


# ---------------------------------------------------------------- parameters, ODE, final size
@dataclass
class EpiParams:
    beta: float                    # transmission rate per day: hazard = beta * I_local / N_local
    gamma: float                   # recovery rate per day (mean infectious period 1 / gamma)
    sigma: float | None = None     # E -> I rate per day (mean latent period 1 / sigma); None = SIR

    @property
    def model(self) -> str:
        return "SIR" if self.sigma is None else "SEIR"

    @property
    def r0(self) -> float:
        return self.beta / self.gamma

    @classmethod
    def from_r0(cls, r0: float, infectious_days: float, latent_days: float | None = None) -> "EpiParams":
        g = 1.0 / infectious_days
        return cls(beta=r0 * g, gamma=g, sigma=None if latent_days is None else 1.0 / latent_days)


def seir_ode(params: EpiParams, n: float, i0: float, days: int, e0: float = 0.0, dt: float = 0.02) -> pd.DataFrame:
    """Mean-field SIR / SEIR (frequency dependent), RK4 with step dt. One row per day 0..days:
    S, E, I, R (counts) and new (infections during the previous day, S(d-1) - S(d); day 0 = i0 + e0 seeds, as
    in :class:`EpiRun`)."""
    b, g, s = params.beta, params.gamma, params.sigma

    def f(y):
        Sx, Ex, Ix, _ = y
        inf = b * Sx * Ix / n
        if s is None:
            return np.array([-inf, 0.0, inf - g * Ix, g * Ix])
        return np.array([-inf, inf - s * Ex, s * Ex - g * Ix, g * Ix])

    y = np.array([n - i0 - e0, e0, i0, 0.0], float)
    k = int(round(1.0 / dt))
    h = 1.0 / k
    out = [y.copy()]
    for _ in range(days):
        for _ in range(k):
            k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(y.copy())
    a = np.array(out)
    df = pd.DataFrame(a, columns=["S", "E", "I", "R"])
    df.insert(0, "day", np.arange(days + 1))
    df["new"] = np.r_[i0 + e0, -np.diff(a[:, 0])]
    return df


def final_size(r0: float, s0: float = 1.0) -> float:
    """Attack fraction z of the closed SIR/SEIR epidemic: z = s0 * (1 - exp(-R0 z)) (Newton; 0 if R0 * s0 <= 1)."""
    if r0 * s0 <= 1:
        return 0.0
    z = s0
    for _ in range(100):
        fz = z - s0 * (1 - np.exp(-r0 * z))
        z_new = z - fz / (1 - s0 * r0 * np.exp(-r0 * z))
        if abs(z_new - z) < 1e-13:
            break
        z = z_new
    return float(z)


def r0_final_size(attack_rate: float) -> float:
    """R0 implied by a final attack fraction z in a closed, fully susceptible population: -ln(1 - z) / z."""
    z = float(np.clip(attack_rate, 1e-12, 1 - 1e-12))
    return float(-np.log(1 - z) / z)


def r0_growth(new: Sequence[float], gamma: float, sigma: float | None = None, n: float | None = None,
              start_cases: float = 10.0, stop_share: float = 0.05, window: tuple[int, int] | None = None) -> dict:
    """R0 from the early exponential growth rate of daily incidence (log-linear OLS fit).

    window: (first_day, last_day) inclusive; default from the first day with >= start_cases new cases to the last
    day before cumulative cases pass stop_share * n (n required then). R0 = (1 + r/gamma)(1 + r/sigma) for
    exponential latent / infectious periods (SIR: 1 + r/gamma). Returns {r0, r, window, n_days}."""
    x = np.asarray(new, float)
    if window is None:
        if n is None:
            raise ValueError("give n (population) or an explicit window")
        ok = np.flatnonzero(x >= start_cases)
        if not len(ok):
            return {"r0": float("nan"), "r": float("nan"), "window": None, "n_days": 0}
        lo = int(ok[0])
        over = np.flatnonzero(np.cumsum(x) > stop_share * n)
        hi = int(over[0]) - 1 if len(over) else len(x) - 1
        window = (lo, max(hi, lo))
    lo, hi = window
    d = np.arange(lo, hi + 1)
    y = x[lo:hi + 1]
    keep = y > 0
    if keep.sum() < 3:
        return {"r0": float("nan"), "r": float("nan"), "window": window, "n_days": int(keep.sum())}
    r = float(np.polyfit(d[keep], np.log(y[keep]), 1)[0])
    r0 = (1 + r / gamma) * (1 if sigma is None else 1 + r / sigma)
    return {"r0": float(r0), "r": r, "window": window, "n_days": int(keep.sum())}


# ---------------------------------------------------------------- grid helpers
def _box_1d(a: np.ndarray, r: int, axis: int, wrap: bool) -> np.ndarray:
    L = a.shape[axis]
    w = 2 * r + 1
    if wrap and w >= L:                                   # window covers the whole torus axis
        return np.broadcast_to(a.sum(axis, keepdims=True), a.shape).copy()
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="wrap" if wrap else "constant")
    c = np.cumsum(p, axis=axis)
    zero = np.zeros_like(np.take(c, [0], axis=axis))
    c = np.concatenate([zero, c], axis=axis)
    return np.take(c, np.arange(w, w + L), axis=axis) - np.take(c, np.arange(0, L), axis=axis)


def box_sum(a: np.ndarray, radius: int, torus: bool = True) -> np.ndarray:
    """Sum over the (2r+1) x (2r+1) Moore window (centre included) on the last two axes; integral-image cost, so
    large radii are as cheap as small ones. Bounded grids count outside cells as 0."""
    a = np.asarray(a, float)
    if radius <= 0:
        return a.copy()
    return _box_1d(_box_1d(a, radius, -2, torus), radius, -1, torus)


def cells_from_xy(x, y, shape: tuple[int, int], origin: int = 0) -> np.ndarray:
    """Flat cell index (row = y, col = x) from integer coordinates; origin=1 for 1-based data (e.g. YJMob100K)."""
    h, w = shape
    x = np.asarray(x, int) - origin
    y = np.asarray(y, int) - origin
    if (x < 0).any() or (x >= w).any() or (y < 0).any() or (y >= h).any():
        raise ValueError(f"coordinates outside the {shape} grid (origin={origin})")
    return y * w + x


def random_positions(n: int, shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    """n static home cells drawn uniformly (several agents may share a cell)."""
    return np.random.default_rng(seed).integers(0, shape[0] * shape[1], n)


# ---------------------------------------------------------------- spatial engine
@dataclass
class EpiRun:
    """Daily counts per replicate. S, E, I, R: (B, days + 1); new: (B, days + 1), new[:, 0] = initial cases;
    infected_day: (B, n) day of infection (0 for seeds, -1 never)."""
    S: np.ndarray
    E: np.ndarray
    I: np.ndarray
    R: np.ndarray
    new: np.ndarray
    infected_day: np.ndarray
    n: int
    params: EpiParams

    @property
    def days(self) -> np.ndarray:
        return np.arange(self.S.shape[1])

    def mean(self) -> pd.DataFrame:
        """Replicate mean per day (same columns as :func:`seir_ode`)."""
        df = pd.DataFrame({k: getattr(self, k).mean(0) for k in ("S", "E", "I", "R", "new")})
        df.insert(0, "day", self.days)
        return df

    def frame(self) -> pd.DataFrame:
        """Long table: rep, day, S, E, I, R, new."""
        B, D = self.S.shape
        return pd.DataFrame({"rep": np.repeat(np.arange(B), D), "day": np.tile(np.arange(D), B),
                             **{k: getattr(self, k).ravel() for k in ("S", "E", "I", "R", "new")}})

    def attack_rate(self) -> np.ndarray:
        return (self.n - self.S[:, -1]) / self.n

    def summary(self, minor: float = 0.01) -> dict:
        """attack rate, peak prevalence share, peak day (mean, sd over replicates); `major` = share of replicates
        with attack rate > minor (stochastic fade-outs are excluded from the *_major means)."""
        ar = self.attack_rate()
        pk = self.I.max(1) / self.n
        pd_ = self.I.argmax(1).astype(float)
        maj = ar > minor
        out = {"reps": int(len(ar)), "major": float(maj.mean()),
               "attack_rate": float(ar.mean()), "attack_rate_sd": float(ar.std()),
               "peak_share": float(pk.mean()), "peak_day": float(pd_.mean())}
        if maj.any():
            out.update(attack_rate_major=float(ar[maj].mean()), peak_share_major=float(pk[maj].mean()),
                       peak_day_major=float(pd_[maj].mean()), peak_day_major_sd=float(pd_[maj].std()))
        return out


class SpatialSEIR:
    """Stochastic SIR / SEIR on a grid (see module docstring).

    params        : EpiParams (sigma None -> SIR).
    shape         : (H, W) grid.
    positions     : None (n uniform random home cells), (n,) static flat cells, or (n, T) trajectories of flat
                    cells (mobility; slot t uses column t % T).
    radius        : Moore window radius incl. own cell; 0 = same cell only; None = well mixed (mean field).
    steps_per_day : sub-steps per day (static) or trajectory slots per day (mobility, e.g. 48 for 30-min slots).
    torus         : wrap edges (True) or bounded grid.
    """

    def __init__(self, params: EpiParams, shape: tuple[int, int] = (100, 100), positions=None, n: int | None = None,
                 radius: int | None = 1, steps_per_day: int = 1, torus: bool = True, seed: int = 0):
        self.params, self.shape, self.radius, self.torus = params, (int(shape[0]), int(shape[1])), radius, torus
        self.steps_per_day = int(steps_per_day)
        if positions is None:
            if n is None:
                raise ValueError("give positions or n")
            positions = random_positions(n, self.shape, seed)
        pos = np.asarray(positions, np.int64)
        if pos.ndim not in (1, 2):
            raise ValueError("positions must be (n,) cells or (n, T) trajectories")
        if pos.min() < 0 or pos.max() >= self.shape[0] * self.shape[1]:
            raise ValueError("positions must be flat cell indices inside the grid")
        self.positions = pos
        self.n = pos.shape[0]
        self.mobile = pos.ndim == 2

    def cells(self, step: int) -> np.ndarray:
        return self.positions[:, step % self.positions.shape[1]] if self.mobile else self.positions

    def local_share(self, infectious: np.ndarray, present: np.ndarray, step: int) -> np.ndarray:
        """(B, n) share of infectious among present agents in each agent's window at this step."""
        inf = infectious & present
        if self.radius is None:
            num = inf.sum(1, keepdims=True).astype(float)
            den = present.sum(1, keepdims=True).astype(float)
            return np.broadcast_to(num / np.maximum(den, 1), inf.shape)
        B = inf.shape[0]
        H, W = self.shape
        cells = self.cells(step)
        flat = (np.arange(B)[:, None] * (H * W) + cells[None, :]).ravel()
        Ic = np.bincount(flat, weights=inf.ravel(), minlength=B * H * W).reshape(B, H, W)
        Nc = np.bincount(flat, weights=present.ravel(), minlength=B * H * W).reshape(B, H, W)
        if self.radius > 0:
            Ic, Nc = box_sum(Ic, self.radius, self.torus), box_sum(Nc, self.radius, self.torus)
        share = (Ic / np.maximum(Nc, 1)).reshape(B, H * W)
        return np.take_along_axis(share, np.broadcast_to(cells, (B, self.n)), 1)

    def run(self, days: int, i0: int | Sequence[int] = 10, reps: int = 1, seed: int | None = None,
            stop_when_extinct: bool = True, absorb: bool = False, present=None) -> EpiRun:
        """Simulate `reps` replicates for `days` days.

        i0      : number of random initial infectious agents per replicate, or explicit agent indices.
        absorb  : new infections enter E and never progress (index-case R0 experiments).
        present : optional (n,) or (B, n) bool, agents absent from all contact (e.g. isolated); default all."""
        p = self.params
        rng = np.random.default_rng(seed)
        n, B, spd = self.n, int(reps), self.steps_per_day
        dt = 1.0 / spd
        state = np.zeros((B, n), np.int8)
        inf_day = np.full((B, n), -1, np.int32)
        if np.ndim(i0) == 0:
            k = int(i0)
            idx = np.argsort(rng.random((B, n)), 1)[:, :k]
        else:
            idx = np.broadcast_to(np.asarray(i0, int), (B, len(i0)))
        np.put_along_axis(state, idx, I, 1)
        np.put_along_axis(inf_day, idx, 0, 1)
        pres = np.ones((B, n), bool) if present is None else np.broadcast_to(np.asarray(present, bool), (B, n))
        pE = 0.0 if (absorb or p.sigma is None) else -np.expm1(-p.sigma * dt)
        pI = -np.expm1(-p.gamma * dt)
        jump_S = 1 if (absorb or p.sigma is not None) else 2        # SIR: S -> I directly
        counts = np.zeros((4, B, days + 1))
        new = np.zeros((B, days + 1))
        new[:, 0] = (state == I).sum(1)

        def record(d):
            for c in range(4):
                counts[c, :, d] = (state == c).sum(1)

        record(0)
        last = days
        for d in range(days):
            for s in range(spd):
                step = d * spd + s
                is_I = state == I
                share = self.local_share(is_I, pres, step)
                prob = np.where(state == S, -np.expm1(-p.beta * dt * share) * pres,
                                np.where(state == E, pE, np.where(is_I, pI, 0.0)))
                trans = rng.random((B, n)) < prob
                fromS = trans & (state == S)
                state = state + np.where(fromS, jump_S, trans.astype(np.int8)).astype(np.int8)
                inf_day[fromS] = d + 1
                new[:, d + 1] += fromS.sum(1)
            record(d + 1)
            if stop_when_extinct:
                active = (state == I).any() if absorb else ((state == I) | (state == E)).any()
                if not active:
                    last = d + 1
                    break
        if last < days:                                     # carry the final state forward
            counts[:, :, last + 1:] = counts[:, :, last:last + 1]
        return EpiRun(S=counts[S], E=counts[E], I=counts[I], R=counts[R], new=new, infected_day=inf_day,
                      n=n, params=p)


def ode_agreement(run: EpiRun, ode: pd.DataFrame, minor: float = 0.01) -> dict:
    """ABM replicates vs the mean-field ODE (same params, n, i0): attack rate, peak prevalence share and peak day,
    plus RMSE of the replicate-mean prevalence curve I/n (major outbreaks only, attack rate > minor)."""
    D = min(run.S.shape[1], len(ode))
    o_I = ode["I"].to_numpy()[:D] / run.n
    o_ar = 1 - ode["S"].iloc[D - 1] / run.n
    maj = run.attack_rate() > minor
    I = run.I[maj, :D] / run.n if maj.any() else run.I[:, :D] / run.n
    ar = run.attack_rate()[maj] if maj.any() else run.attack_rate()
    return {"reps": int(len(maj)), "major": float(maj.mean()),
            "attack_rate_abm": float(ar.mean()), "attack_rate_ode": float(o_ar),
            "peak_share_abm": float(I.max(1).mean()), "peak_share_ode": float(o_I.max()),
            "peak_day_abm": float(I.argmax(1).mean()), "peak_day_ode": float(o_I.argmax()),
            "prevalence_rmse": float(np.sqrt(((I.mean(0) - o_I) ** 2).mean()))}


def r0_index_case(model: SpatialSEIR, reps: int = 400, max_days: int = 200, seed: int = 0) -> dict:
    """Realised R0 of a contact structure: one random infectious agent per replicate in a fully susceptible
    population, secondary infections absorbing (they never transmit); R0 = mean secondary cases. Normal 95% CI."""
    run = model.run(max_days, i0=1, reps=reps, seed=seed, absorb=True)
    sec = run.E[:, -1]
    m, se = float(sec.mean()), float(sec.std(ddof=1) / np.sqrt(len(sec))) if len(sec) > 1 else 0.0
    return {"r0": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se, "reps": int(reps), "theory": model.params.r0,
            "secondary": sec}


# ---------------------------------------------------------------- mobility data
def trajectories_from_records(df: pd.DataFrame, shape: tuple[int, int], days: int, slots_per_day: int = 48,
                              uid: str = "uid", day: str = "d", slot: str = "t", x: str = "x", y: str = "y",
                              origin: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Long (uid, day, slot, x, y) records -> dense (n_users, days * slots_per_day) flat-cell trajectories.
    Missing slots take the last known cell (forward fill), leading gaps the first known cell. Returns (traj, uids)."""
    d = df[df[day] < days]
    uids, u = np.unique(d[uid].to_numpy(), return_inverse=True)
    T = days * slots_per_day
    col = d[day].to_numpy(int) * slots_per_day + d[slot].to_numpy(int)
    traj = np.full((len(uids), T), -1, np.int64)
    traj[u, col] = cells_from_xy(d[x].to_numpy(), d[y].to_numpy(), shape, origin)
    have = traj >= 0
    idx = np.where(have, np.arange(T)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    first = have.argmax(1)
    idx = np.where(np.maximum.accumulate(have, axis=1), idx, first[:, None])
    return np.take_along_axis(traj, idx, 1), uids


def load_yjmob100k(data_root: str | None = None, n_users: int = 2000, days: int = 14,
                   chunksize: int = 2_000_000) -> tuple[np.ndarray, np.ndarray]:
    """First `n_users` users (by uid) of YJMob100K dataset 1 (Yabe et al. 2024, Sci Data; CC BY 4.0; 200 x 200
    grid of 500 m cells, 30-min slots, 1-based x / y) as trajectories for :class:`SpatialSEIR` (shape (200, 200),
    steps_per_day=48). data_root: the catalogue root holding `yjmob100k/` or that directory itself; default
    $SIMENGINE_DATA. The file is sorted by uid, so reading stops after the first n_users."""
    root = data_root if data_root is not None else os.environ.get(ENV_DATA)
    if not root:
        raise FileNotFoundError(f"no data_root given and ${ENV_DATA} is not set")
    if os.path.isdir(os.path.join(root, "yjmob100k")):
        root = os.path.join(root, "yjmob100k")
    path = os.path.join(root, "yjmob100k-dataset1.csv.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing file: {path}")
    parts, seen = [], set()
    for ch in pd.read_csv(path, chunksize=chunksize, dtype="int32"):
        ch = ch[ch.d < days]
        seen.update(np.unique(ch.uid).tolist())
        parts.append(ch)
        if len(seen) > n_users:
            break
    df = pd.concat(parts)
    keep = np.sort(np.unique(df.uid))[:n_users]
    return trajectories_from_records(df[df.uid.isin(keep)], (200, 200), days, 48, origin=1)


# ---------------------------------------------------------------- generative agents (Williams et al. 2023)
TRAITS = [("extraverted", "introverted"), ("agreeable", "disagreeable"), ("conscientious", "unconscientious"),
          ("emotionally stable", "emotionally unstable"), ("open to experience", "closed to experience")]
# day of infection -> symptom (paper: 6 days; 1-2 asymptomatic but infectious, 3 and 6 light cough, 4-5 fever)
SYMPTOMS = {1: "normal", 2: "normal", 3: "cough", 4: "fever", 5: "fever", 6: "cough"}
HEALTH_TEXT = {"normal": "{name} feels normal.", "cough": "{name} has a light cough.",
               "fever": "{name} has a fever and a cough."}
CONDITIONS = ("base", "self_health", "full")


@dataclass
class EpiAgent:
    id: str
    name: str                      # label used in the prompt (no real names by default)
    age: int
    traits: list[str] = field(default_factory=list)


def epi_personas(n: int, seed: int = 0) -> list[EpiAgent]:
    """Paper-style personas: age uniform in 18..64, each Big Five trait positive or negative with p = 0.5."""
    rng = np.random.default_rng(seed)
    ages = rng.integers(18, 65, n)
    pick = rng.random((n, len(TRAITS))) < 0.5
    return [EpiAgent(f"a{i:04d}", f"Resident {i + 1}", int(ages[i]),
                     [TRAITS[j][0 if pick[i, j] else 1] for j in range(len(TRAITS))]) for i in range(n)]


def isolation_prompt(agent: EpiAgent, health: str | None = None, news_pct: float | None = None,
                     message: str | None = None, town: str = "Dewberry Hollow",
                     virus: str = "Catasat") -> tuple[str, str]:
    """(system, user) for one daily decision; structure follows Williams et al. 2023, Fig. S2 (persona and memories
    in the system message, the question and reply format in the user message). health: 'normal' | 'cough' |
    'fever' | None (not told); news_pct: yesterday's new-case share in % or None (not told)."""
    nm = agent.name
    mem = []
    if health is not None:
        mem.append(HEALTH_TEXT[health].format(name=nm))
    if news_pct is not None:
        mem.append(f"{nm} knows about the {virus} virus spreading across the country. It is an infectious disease "
                   f"that spreads from human to human contact via an airborne virus. The deadliness of the virus is "
                   f"unknown. Scientists are warning about a potential epidemic.")
        mem.append(f"{nm} checks the newspaper and finds that {news_pct:.1f}% of {town}'s population caught new "
                   f"infections of the {virus} virus yesterday.")
    if message:
        mem.append(message)
    mem.append(f"{nm} goes to work to earn money to support {nm}'s self.")
    system = (f"You are {nm}. You are {agent.age} years old.\nYour traits are given below:\n"
              f"{', '.join(agent.traits)}\nYour basic bio is below:\n"
              f"{nm} lives in the town of {town}. {nm} likes the town and has friends who also live there. "
              f"{nm} has a job and goes to the office for work every day.\n"
              f"I will provide {nm}'s relevant memories here:\n" + "\n".join(mem))
    user = (f"Based on the provided memories, should {nm} stay at home for the entire day? Please provide your "
            f"reasoning.\nThe format should be as follows:\nReasoning: [explanation]\nResponse: [Yes or No]\n"
            f"It is important to provide Response in a single word.")
    return system, user


_RESP = re.compile(r"Response\s*[:\-]\s*\**\s*(yes|no)\b", re.I)
_REASON = re.compile(r"Reasoning\s*:\s*(.*?)(?:\n\s*\**\s*Response\s*:|$)", re.I | re.S)


def parse_isolation(text: str) -> tuple[bool | None, str]:
    """(stay_home, reasoning) from a 'Reasoning: ... Response: Yes|No' reply; stay_home None if non-conforming."""
    m = _RESP.search(text or "")
    r = _REASON.search(text or "")
    return (None if m is None else m.group(1).lower() == "yes"), (r.group(1).strip() if r else "")


def rule_responder(stay_on: Sequence[str] = ("cough", "fever"), news_threshold: float | None = None,
                   message_keyword: str | None = None) -> Callable[[str, str], str]:
    """FakeLLM responder with a fixed rule, in the paper's reply format: stay home if told of a symptom in
    `stay_on`, or if the news share >= news_threshold (%), or if the memories contain `message_keyword`."""
    def respond(system: str, user: str) -> str:
        sym = "fever" if "has a fever" in system else "cough" if "has a light cough" in system else "normal"
        m = re.search(r"finds that ([\d.]+)%", system)
        pct = float(m.group(1)) if m else None
        stay = (sym in stay_on or (news_threshold is not None and pct is not None and pct >= news_threshold)
                or (message_keyword is not None and message_keyword in system))
        return f"Reasoning: rule (symptom={sym}, news={pct}).\nResponse: {'Yes' if stay else 'No'}"
    return respond


@dataclass
class GenerativeRun:
    daily: pd.DataFrame            # day, S, I, R, new_cases, day4_cases, news_pct, out_share, invalid
    log: pd.DataFrame              # day, agent, health, news_pct, stay_home, valid, reasoning, prompt_id
    calls: int
    invalid: int
    stopped: str                   # "cleared" | "days" | "budget"

    def stay_home_by_health(self) -> pd.Series:
        """Share of decisions to stay home per health state (cf. the paper's Table 1 effects)."""
        return self.log.groupby("health")["stay_home"].mean()


class GenerativeEpidemic:
    """LLM-agent epidemic after Williams et al. 2023 (see module docstring).

    llm           : LLM protocol (FakeLLM, Claude, ClaudeCode); one call per agent per day through llm.map.
    agents / n    : EpiAgent list or n paper-style personas.
    condition     : 'base' | 'self_health' | 'full'.
    message       : None, a string, or callable(day, news_pct) -> str | None, added to every agent's memories.
    contact       : 'random' (paper: each infectious agent out meets contact_rate distinct agents among those out)
                    or 'grid' (out agents mix in a Moore window of `radius` around `positions` on `shape`;
                    infection probability 1 - (1 - infectivity) ** (contact_rate * local infectious share)).
    i0            : initially infected agents (day 1 of infection). The paper text does not state its value.
    max_calls     : stop (stopped='budget') before a day whose calls would exceed this.
    Invalid replies: retried `retries` times, then counted as 'No' (go out), as in the paper; row valid=False.
    """

    def __init__(self, llm, agents: Sequence[EpiAgent] | None = None, n: int = 100, contact_rate: int = 5,
                 infectivity: float = 0.1, infectious_days: int = 6, condition: str = "full",
                 message: str | Callable[[int, float], str | None] | None = None, i0: int = 3,
                 contact: str = "random", shape: tuple[int, int] = (20, 20), positions=None, radius: int = 1,
                 town: str = "Dewberry Hollow", virus: str = "Catasat", max_tokens: int = 256, parallel: int = 8,
                 retries: int = 0, max_calls: int | None = None, seed: int = 0):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        if contact not in ("random", "grid"):
            raise ValueError("contact must be 'random' or 'grid'")
        self.llm = llm
        self.agents = list(agents) if agents is not None else epi_personas(n, seed)
        self.n = len(self.agents)
        self.k, self.p, self.D = int(contact_rate), float(infectivity), int(infectious_days)
        self.condition, self.message, self.i0, self.contact = condition, message, int(i0), contact
        self.town, self.virus, self.max_tokens, self.parallel = town, virus, max_tokens, parallel
        self.retries, self.max_calls, self.seed = retries, max_calls, seed
        if contact == "grid":
            pos = random_positions(self.n, shape, seed) if positions is None else np.asarray(positions, np.int64)
            self.grid = SpatialSEIR(EpiParams(0.0, 1.0), shape, positions=pos, radius=radius)
        self.r0_nominal = self.k * self.p * self.D           # paper: 5 * 0.1 * 6 = 3

    def _symptom(self, dinf: np.ndarray) -> list[str]:
        if self.D == 6:
            return [SYMPTOMS.get(int(d), "normal") for d in dinf]
        # other durations: scale the paper's 6-day profile
        return [SYMPTOMS.get(int(np.ceil(d * 6 / self.D)), "normal") if d > 0 else "normal" for d in dinf]

    def _message(self, day: int, pct: float) -> str | None:
        return self.message(day, pct) if callable(self.message) else self.message

    def _ask(self, prompts):
        replies = self.llm.map(prompts, max_tokens=self.max_tokens, parallel=self.parallel)
        out = [parse_isolation(r.text) for r in replies]
        calls = len(prompts)
        for _ in range(self.retries):
            bad = [i for i, (s, _) in enumerate(out) if s is None]
            if not bad:
                break
            again = self.llm.map([prompts[i] for i in bad], max_tokens=self.max_tokens, parallel=self.parallel)
            calls += len(bad)
            for i, r in zip(bad, again):
                out[i] = parse_isolation(r.text)
        return out, calls

    def _transmit(self, out: np.ndarray, infectious: np.ndarray, susceptible: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
        if self.contact == "grid":
            share = self.grid.local_share(infectious[None], out[None], 0)[0]
            prob = 1 - (1 - self.p) ** (self.k * share)
            return susceptible & out & (rng.random(self.n) < prob)
        O = np.flatnonzero(out)
        src = np.flatnonzero(out & infectious)
        hit = np.zeros(self.n, bool)
        k = min(self.k, len(O) - 1)
        if k <= 0 or not len(src):
            return hit
        key = rng.random((len(src), len(O)))
        key[np.arange(len(src)), np.searchsorted(O, src)] = 2.0          # no self contact
        partners = O[np.argpartition(key, k - 1, axis=1)[:, :k]]
        ok = susceptible[partners] & (rng.random(partners.shape) < self.p)
        hit[partners[ok]] = True
        return hit

    def run(self, days: int = 60, stop_when_clear: bool = True) -> GenerativeRun:
        """Daily loop (paper Table S4): decide -> contacts among those out -> infections -> progress -> news."""
        rng = np.random.default_rng(self.seed)
        n = self.n
        dinf = np.zeros(n, np.int32)                       # day of infection (0 = not infected)
        rec = np.zeros(n, bool)
        dinf[rng.choice(n, self.i0, replace=False)] = 1
        day4 = self.i0                                     # paper: daily_new_cases starts at initial_infected
        rows, logs, calls, invalid, clear, stopped = [], [], 0, 0, 0, "days"
        for day in range(days):
            if self.max_calls is not None and calls + n > self.max_calls:
                stopped = "budget"
                break
            pct = 100.0 * day4 / n
            sym = self._symptom(dinf)
            health = None if self.condition == "base" else sym
            news = pct if self.condition == "full" else None
            msg = self._message(day, pct)
            prompts = [isolation_prompt(a, None if health is None else health[i], news, msg, self.town, self.virus)
                       for i, a in enumerate(self.agents)]
            answers, c = self._ask(prompts)
            calls += c
            stay = np.array([bool(s) for s, _ in answers])
            valid = np.array([s is not None for s, _ in answers])
            invalid += int((~valid).sum())
            out = ~stay
            infectious = dinf > 0
            susceptible = (dinf == 0) & ~rec
            hit = self._transmit(out, infectious, susceptible, rng)
            # progression: infected advance one day; past D -> recovered; new infections start at day 1
            adv = infectious
            dinf[adv] += 1
            done = dinf > self.D
            rec |= done
            dinf[done] = 0
            dinf[hit] = 1
            day4 = int((dinf == 4).sum())
            for i, a in enumerate(self.agents):
                logs.append((day, a.id, sym[i], news, bool(stay[i]), bool(valid[i]), answers[i][1],
                             hashlib.sha256((prompts[i][0] + "\x00" + prompts[i][1]).encode()).hexdigest()[:12]))
            n_inf = int((dinf > 0).sum())
            rows.append({"day": day, "S": int(((dinf == 0) & ~rec).sum()), "I": n_inf, "R": int(rec.sum()),
                         "new_cases": int(hit.sum()), "day4_cases": day4, "news_pct": pct,
                         "out_share": float(out.mean()), "invalid": int((~valid).sum())})
            clear = clear + 1 if n_inf == 0 else 0
            if stop_when_clear and clear >= 2:            # paper: no infected agents for two consecutive days
                stopped = "cleared"
                break
        log = pd.DataFrame(logs, columns=["day", "agent", "health", "news_pct", "stay_home", "valid", "reasoning",
                                          "prompt_id"])
        return GenerativeRun(pd.DataFrame(rows), log, calls, invalid, stopped)
