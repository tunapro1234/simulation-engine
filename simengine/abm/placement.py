"""Ad / billboard placement on a grid from simulated or observed trajectories: reach and frequency of a cell set,
greedy and lazy-greedy (CELF) maximum coverage under a cell count or a cost budget, an exposure-to-adoption step with
word of mouth to grid neighbours, and a temporal holdout against the usual baselines (top-k visited cells, random).

Trajectory protocol (no dependency on a mobility model): anything with five integer columns `user, day, slot, x, y`
-- a pandas DataFrame (YJMob-style names `uid, d, t` are accepted), a dict of arrays, or an (n, 5) array in that
order. Rows whose x / y fall outside the :class:`CellGrid` (e.g. the 999 mask of YJMob100K) are dropped.

    import simengine.abm.placement as pl
    traj = pl.toy_trajectories(n_users=300, n_days=14, seed=0)            # synthetic fixture, not a mobility model
    grid = pl.CellGrid.fit(traj, size=2)                                  # 2x2 raw cells per billboard cell
    ex = pl.exposures(traj, grid, days=range(10))                         # users x cells, distinct (user, day, cell)
    p = pl.greedy_placement(ex.M, k=5)                                    # lazy greedy, (1 - 1/e) guarantee
    print(p.cells, p.value, p.bound)                                      # bound: online upper bound on the optimum
    print(pl.reach_frequency(ex.M, p.cells))
    print(pl.holdout(traj, grid, train_days=range(10), test_days=range(10, 14), ks=(5, 10)))

Objective. With c_u(S) the exposures of user u to cell set S, the placement maximises
    F(S) = sum_u w_u * min(c_u(S), r) / r
r = 1 is plain reach (maximum coverage); r > 1 is a frequency-capped coverage that credits up to r exposures (a
concave function of a modular one, so still monotone submodular). "Effective reach at r+" (users with c_u >= r) is
reported but not optimised directly: it is not submodular.

Algorithms (method checked against Leskovec et al., "Cost-effective outbreak detection in networks", KDD 2007):
* unit cost, k cells: greedy, F(A_G) >= (1 - 1/e) max_{|A|=k} F(A) (their Theorem 2, citing Nemhauser-Wolsey-Fisher).
* cell costs and a budget B: CELF runs the unit-cost greedy and the benefit/cost greedy (both skipping unaffordable
  cells) and keeps the better; max(F(A_UC), F(A_CB)) >= 1/2 (1 - 1/e) OPT (their Theorem 3; Khuller-Moss-Naor 1999
  for budgeted max cover).
* lazy evaluation: stale marginal gains are upper bounds by submodularity, so only the top of a heap is re-evaluated;
  same output as the plain greedy (ties broken by lowest cell index), far fewer evaluations.
* :attr:`Placement.bound`: their Theorem 4 online bound, OPT <= F(A) + the best (fractional) knapsack of the
  single-cell gains delta_s given A; it applies to any placement (see :func:`online_bound`).

Adoption (:func:`adoption`): campaign exposures convert with 1 - (1 - p_exposure)^f, then word of mouth runs on the
home-cell grid: pressure = adopter share in the 3x3 neighbourhood (Moore), a non-adopter adopts with
1 - exp(-q * pressure). An optional cumulative kernel F (e.g. :func:`simengine.abm.diffusion.power_kernel`) decays an
adopter's influence with age; `wom=` replaces the neighbourhood rule (the hook).

STATUS: measured by this port on a YJMob100K subsample (first 4,000 users, days 0-59 -> 60-74, one seed; tables and
command in docs/modules/abm/placement.md). Test-day reach share at k = 50 on 500 m cells: lazy greedy 0.739, top-50
most visited 0.597 (difference [+0.126, +0.157]), random 0.152, greedy fitted on the test days 0.790. Greedy on
EPR-simulated trajectories (simengine.abm.mobility) 0.741. Not measured against real campaign outcomes: there is no
adoption ground truth here, so the holdout scores reach only.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy import sparse

COLS = ("user", "day", "slot", "x", "y")
ALIASES = {"uid": "user", "d": "day", "t": "slot"}
_ROUND = 9          # gains are rounded before comparisons so plain and lazy greedy break ties identically


# ---------------------------------------------------------------- trajectories and grid

def traj_columns(traj: Any) -> dict[str, np.ndarray]:
    """Normalise the trajectory protocol to {user, day, slot, x, y} int64 arrays."""
    if isinstance(traj, pd.DataFrame):
        df = traj.rename(columns={k: v for k, v in ALIASES.items() if k in traj.columns and v not in traj.columns})
        return {c: df[c].to_numpy(np.int64) for c in COLS}
    if isinstance(traj, dict):
        d = {ALIASES.get(k, k): v for k, v in traj.items()}
        return {c: np.asarray(d[c], np.int64) for c in COLS}
    a = np.asarray(traj)
    if a.ndim != 2 or a.shape[1] != 5:
        raise ValueError("trajectories must be a DataFrame, a dict or an (n, 5) array user, day, slot, x, y")
    return {c: a[:, i].astype(np.int64) for i, c in enumerate(COLS)}


@dataclass(frozen=True)
class CellGrid:
    """Billboard cells: raw coordinates [x0, x0 + nx*size) x [y0, y0 + ny*size) in blocks of size x size.
    cell id = row * nx + col with col = (x - x0) // size, row = (y - y0) // size."""
    x0: int
    y0: int
    nx: int
    ny: int
    size: int = 1

    @classmethod
    def fit(cls, traj: Any, size: int = 1, missing: Iterable[int] = (999,)) -> "CellGrid":
        """Bounding box of the observed coordinates (ignoring `missing` sentinels), in blocks of `size`."""
        c = traj_columns(traj)
        ok = ~(np.isin(c["x"], list(missing)) | np.isin(c["y"], list(missing)))
        x, y = c["x"][ok], c["y"][ok]
        x0, y0 = int(x.min()), int(y.min())
        return cls(x0, y0, -(-(int(x.max()) - x0 + 1) // size), -(-(int(y.max()) - y0 + 1) // size), size)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def cell(self, x, y) -> np.ndarray:
        """Cell id per point, -1 outside the grid."""
        col = (np.asarray(x) - self.x0) // self.size
        row = (np.asarray(y) - self.y0) // self.size
        ok = (col >= 0) & (col < self.nx) & (row >= 0) & (row < self.ny)
        return np.where(ok, row * self.nx + col, -1)

    def xy(self, cells) -> np.ndarray:
        """(n, 2) raw coordinates of each cell's lower-left corner."""
        cells = np.asarray(cells, np.int64)
        return np.c_[self.x0 + (cells % self.nx) * self.size, self.y0 + (cells // self.nx) * self.size]


@dataclass
class Exposure:
    """M: users x cells exposure counts (scipy CSC). users: user id of each row."""
    M: sparse.csc_matrix
    users: np.ndarray
    grid: CellGrid
    unit: str = "day"

    @property
    def active(self) -> np.ndarray:
        """Users with at least one exposure in this window."""
        return np.asarray(self.M.sum(1)).ravel() > 0


def exposures(traj: Any, grid: CellGrid, days: Iterable[int] | None = None, slots: Iterable[int] | None = None,
              unit: str = "day", users: np.ndarray | None = None) -> Exposure:
    """Exposure matrix (users x cells) from trajectories restricted to `days` / `slots`.

    unit: "day" counts distinct (user, day, cell) -- one billboard sighting per day; "slot" counts every ping;
    "user" is 0/1. `users` fixes the row order (pass the same array for train and test windows)."""
    c = traj_columns(traj)
    if users is None:
        users = np.unique(c["user"])
    users = np.asarray(users, np.int64)
    keep = grid.cell(c["x"], c["y"]) >= 0
    if days is not None:
        keep &= np.isin(c["day"], np.fromiter(days, np.int64))
    if slots is not None:
        keep &= np.isin(c["slot"], np.fromiter(slots, np.int64))
    pos = np.searchsorted(users, c["user"][keep])
    pos = np.clip(pos, 0, len(users) - 1)
    known = users[pos] == c["user"][keep]
    row, cell, day = pos[known], grid.cell(c["x"][keep], c["y"][keep])[known], c["day"][keep][known]
    if unit == "day":
        key = np.unique((row * grid.n_cells + cell) * (int(day.max(initial=0)) + 1) + day)
        rc = key // (int(day.max(initial=0)) + 1)
        row, cell = rc // grid.n_cells, rc % grid.n_cells
    elif unit == "user":
        rc = np.unique(row * grid.n_cells + cell)
        row, cell = rc // grid.n_cells, rc % grid.n_cells
    elif unit != "slot":
        raise ValueError("unit must be 'day', 'slot' or 'user'")
    M = sparse.coo_matrix((np.ones(len(row)), (row, cell)), shape=(len(users), grid.n_cells)).tocsc()
    M.sum_duplicates()
    return Exposure(M, users, grid, unit)


def home_cells(traj: Any, grid: CellGrid, users: np.ndarray, night: Iterable[int] = tuple(range(0, 14)) +
               tuple(range(42, 48)), days: Iterable[int] | None = None) -> np.ndarray:
    """Modal cell over night slots (default 0-6:59 and 21:00-23:59 for 30-min slots), else modal cell overall;
    -1 for users never seen. Aligned with `users`."""
    out = np.full(len(users), -1, np.int64)
    for sl in (None, night):                       # overall first, night overwrites where available
        ex = exposures(traj, grid, days=days, slots=sl, unit="slot", users=users).M.tocsr()
        has = np.diff(ex.indptr) > 0
        if has.any():
            am = np.asarray(ex.argmax(1)).ravel()
            out[has] = am[has]
    return out


# ---------------------------------------------------------------- objective and reach / frequency

def _cell_vector(M, cells) -> np.ndarray:
    cells = np.asarray(list(cells), np.int64)
    if len(cells) == 0:
        return np.zeros(M.shape[0])
    return np.asarray(M[:, cells].sum(1)).ravel()


def coverage(M, cells, r: int = 1, weights: np.ndarray | None = None) -> float:
    """F(S) = sum_u w_u min(c_u(S), r) / r."""
    c = _cell_vector(M, cells)
    w = np.ones(M.shape[0]) if weights is None else np.asarray(weights, float)
    return float((w * np.minimum(c, r)).sum() / r)


def reach_frequency(M, cells, min_freq: int = 3, users: np.ndarray | None = None) -> dict:
    """Media-planning summary of a cell set over the users in M (or the boolean / index subset `users`).

    reach (users with >= 1 exposure), reach_share, frequency (mean exposures among reached), effective_reach
    (users with >= min_freq), grp (gross rating points = 100 * total exposures / audience), hist (count of users
    with 0, 1, ..., min_freq+ exposures)."""
    c = _cell_vector(M, cells)
    if users is not None:
        c = c[users]
    n = max(len(c), 1)
    reached = c > 0
    hist = np.bincount(np.minimum(c, min_freq).astype(np.int64), minlength=min_freq + 1)
    return dict(reach=int(reached.sum()), reach_share=float(reached.sum() / n),
                frequency=float(c[reached].mean()) if reached.any() else 0.0,
                effective_reach=int((c >= min_freq).sum()), grp=float(100 * c.sum() / n), audience=int(len(c)),
                hist=hist.tolist())


# ---------------------------------------------------------------- greedy / lazy greedy

@dataclass
class Placement:
    """Chosen cells in pick order, marginal gains, objective value, total cost, online upper bound on the optimum
    (Leskovec et al. 2007 Theorem 4), number of single-cell gain evaluations, method name."""
    cells: list[int]
    gains: list[float]
    value: float
    cost: float
    bound: float
    evaluations: int
    method: str
    extra: dict = field(default_factory=dict)


class _State:
    """Exposure counts c_u(S) and the gain of adding one or all cells (CSC column slices, vectorised)."""

    def __init__(self, M, r, weights):
        self.M = sparse.csc_matrix(M, dtype=float)
        self.r = float(r)
        self.w = np.ones(self.M.shape[0]) if weights is None else np.asarray(weights, float)
        self.c = np.zeros(self.M.shape[0])
        self.col = np.repeat(np.arange(self.M.shape[1]), np.diff(self.M.indptr))
        self.evals = 0

    def _contrib(self, rows, data):
        c = self.c[rows]
        return self.w[rows] * (np.minimum(c + data, self.r) - np.minimum(c, self.r)) / self.r

    def gain(self, j: int) -> float:
        self.evals += 1
        a, b = self.M.indptr[j], self.M.indptr[j + 1]
        g = np.bincount(np.zeros(b - a, np.int64), self._contrib(self.M.indices[a:b], self.M.data[a:b]), 1)[0]
        return round(float(g), _ROUND)

    def gains(self) -> np.ndarray:
        self.evals += self.M.shape[1]
        g = np.bincount(self.col, self._contrib(self.M.indices, self.M.data), self.M.shape[1])
        return np.round(g, _ROUND)

    def add(self, j: int) -> None:
        a, b = self.M.indptr[j], self.M.indptr[j + 1]
        self.c[self.M.indices[a:b]] += self.M.data[a:b]

    def value(self) -> float:
        return float((self.w * np.minimum(self.c, self.r)).sum() / self.r)


def _plain(st: _State, avail: np.ndarray, cost: np.ndarray, budget: float, ratio: bool):
    cells, gains, spent = [], [], 0.0
    avail = avail.copy()
    while True:
        avail &= cost <= budget - spent + 1e-12
        if not avail.any():
            break
        g = st.gains()
        key = np.where(avail, np.round(g / cost, _ROUND) if ratio else g, -np.inf)
        j = int(np.argmax(key))
        if g[j] <= 0:
            break
        st.add(j)
        cells.append(j)
        gains.append(float(g[j]))
        spent += cost[j]
        avail[j] = False
    return cells, gains, spent


def _lazy(st: _State, avail: np.ndarray, cost: np.ndarray, budget: float, ratio: bool):
    g0 = st.gains()
    fresh = {int(j): float(g0[j]) for j in np.flatnonzero(avail)}          # last evaluated gain per cell
    heap = [((-round(g / cost[j], _ROUND) if ratio else -g), j, 0) for j, g in fresh.items()]
    heapq.heapify(heap)
    cells, gains, spent, step = [], [], 0.0, 0
    while heap:
        _, j, at = heapq.heappop(heap)
        if cost[j] > budget - spent + 1e-12:
            continue                                   # remaining budget only shrinks: drop for good
        if at == step:                                 # bound is current: j is the argmax
            if fresh[j] <= 0:
                break
            st.add(j)
            cells.append(j)
            gains.append(fresh[j])
            spent += cost[j]
            step += 1
            continue
        g = fresh[j] = st.gain(j)
        heapq.heappush(heap, ((-round(g / cost[j], _ROUND) if ratio else -g), j, step))
    return cells, gains, spent


def online_bound(M, cells, k: int | None = None, costs: np.ndarray | None = None, budget: float | None = None,
                 r: int = 1, weights: np.ndarray | None = None) -> float:
    """Upper bound on the optimal F for budget k (unit costs) or `budget` (costs), valid for ANY placement `cells`:
    F(A) + the best fractional knapsack of single-cell gains delta_s given A (Leskovec et al. 2007, Theorem 4)."""
    st = _State(M, r, weights)
    for j in cells:
        st.add(int(j))
    d = st.gains()
    d[np.asarray(list(cells), np.int64)] = 0.0
    if costs is None:
        return st.value() + float(np.sort(d)[::-1][:int(k)].clip(0).sum())
    cost = np.asarray(costs, float)
    order = np.argsort(-(d / cost), kind="stable")
    d, cost = d[order], cost[order]
    cum = np.cumsum(cost)
    full = cum <= budget
    extra = float(d[full].sum())
    nxt = int(full.sum())
    if nxt < len(d):
        extra += (budget - (cum[nxt - 1] if nxt else 0.0)) / cost[nxt] * d[nxt]
    return st.value() + extra


def greedy_placement(M, k: int | None = None, *, costs: np.ndarray | None = None, budget: float | None = None,
                     r: int = 1, weights: np.ndarray | None = None, candidates: Iterable[int] | None = None,
                     lazy: bool = True, fill: bool = False) -> Placement:
    """Maximum (capped) coverage under a cell count `k` or a cost `budget`.

    Unit cost: greedy (1 - 1/e). Costs: CELF, better of unit-cost and benefit/cost greedy (1/2 (1 - 1/e)).
    lazy=False runs the plain greedy (one vectorised pass over all cells per pick); same cells.
    Greedy stops when no cell adds value. fill=True then spends the rest of the budget on the remaining candidates
    by distinct visitors (zero gain on these exposures, but a hedge for unseen days); recorded in extra["filled"]."""
    M = sparse.csc_matrix(M, dtype=float)
    n = M.shape[1]
    avail = np.zeros(n, bool)
    avail[np.arange(n) if candidates is None else np.asarray(list(candidates), np.int64)] = True
    if costs is None:
        if k is None:
            raise ValueError("give k (number of cells) or costs + budget")
        cost, B = np.ones(n), float(k)
    else:
        if budget is None:
            raise ValueError("costs need a budget")
        cost, B = np.asarray(costs, float), float(budget)
        if (cost <= 0).any():
            raise ValueError("costs must be positive")
    runs = []
    for ratio in ((False,) if costs is None else (False, True)):
        st = _State(M, r, weights)
        cells, gains, spent = (_lazy if lazy else _plain)(st, avail, cost, B, ratio)
        runs.append((st.value(), cells, gains, spent, st.evals, "benefit_cost" if ratio else "unit_cost"))
    value, cells, gains, spent, evals, which = max(runs, key=lambda t: t[0])
    bound = online_bound(M, cells, k=k, costs=costs, budget=budget, r=r, weights=weights)
    filled: list[int] = []
    if fill:
        rest = avail.copy()
        rest[cells] = False
        order = np.flatnonzero(rest)[np.argsort(-np.diff(M.indptr)[rest], kind="stable")]
        filled = _take_in_order(order, B - len(cells), None, None) if costs is None else \
            _take_in_order(order, None, cost, B - spent)
        cells, gains = cells + filled, gains + [0.0] * len(filled)
        spent += float(cost[filled].sum()) if filled else 0.0
    return Placement(cells, gains, value, float(spent), bound, int(sum(t[4] for t in runs)),
                     ("lazy_greedy" if lazy else "greedy"),
                     dict(rule=which, runs={t[5]: t[0] for t in runs}, filled=filled))


# ---------------------------------------------------------------- baselines

def _take_in_order(order: np.ndarray, k, cost, budget) -> list[int]:
    if cost is None:
        return [int(j) for j in order[:int(k)]]
    out, spent = [], 0.0
    for j in order:
        if spent + cost[j] <= budget + 1e-12:
            out.append(int(j))
            spent += cost[j]
    return out


def top_visited(M, k: int | None = None, *, by: str = "visits", costs: np.ndarray | None = None,
                budget: float | None = None, candidates: Iterable[int] | None = None) -> list[int]:
    """Baseline: cells ranked by total exposures (by="visits") or distinct visitors (by="visitors"); first k, or
    in rank order while the budget allows."""
    M = sparse.csc_matrix(M)
    score = np.asarray(M.sum(0)).ravel() if by == "visits" else np.diff(M.indptr).astype(float)
    if by not in ("visits", "visitors"):
        raise ValueError("by must be 'visits' or 'visitors'")
    cand = np.arange(M.shape[1]) if candidates is None else np.asarray(list(candidates), np.int64)
    order = cand[np.argsort(-score[cand], kind="stable")]
    return _take_in_order(order, k, None if costs is None else np.asarray(costs, float), budget)


def random_cells(M, k: int | None = None, *, seed: int = 0, costs: np.ndarray | None = None,
                 budget: float | None = None, candidates: Iterable[int] | None = None) -> list[int]:
    """Baseline: uniform random cells among `candidates` (default: cells with at least one exposure in M)."""
    M = sparse.csc_matrix(M)
    cand = np.flatnonzero(np.diff(M.indptr) > 0) if candidates is None else np.asarray(list(candidates), np.int64)
    order = np.random.default_rng(seed).permutation(cand)
    return _take_in_order(order, k, None if costs is None else np.asarray(costs, float), budget)


# ---------------------------------------------------------------- exposure -> adoption with word of mouth

def moore_share(adopt_w: np.ndarray, pop: np.ndarray) -> np.ndarray:
    """Default word-of-mouth pressure: adopter weight / population in each cell's 3x3 neighbourhood.
    adopt_w, pop: (R, ny, nx). Returns (R, ny, nx)."""
    def box(a):
        p = np.pad(a, ((0, 0), (1, 1), (1, 1)))
        return sum(p[:, i:i + a.shape[1], j:j + a.shape[2]] for i in range(3) for j in range(3))
    P = box(pop)
    return np.where(P > 0, box(adopt_w) / np.maximum(P, 1e-12), 0.0)


@dataclass
class AdoptionRun:
    """curve (R, steps + 1): cumulative adopters after the campaign step (0) and each word-of-mouth step;
    adopt_step (R, n_users): step of adoption, -1 never; from_ad / from_wom (R,) adopters per channel."""
    curve: np.ndarray
    adopt_step: np.ndarray
    from_ad: np.ndarray
    from_wom: np.ndarray

    @property
    def mean_curve(self) -> np.ndarray:
        return self.curve.mean(0)


def adoption(freq: np.ndarray, home: np.ndarray, grid: CellGrid, *, p_exposure: float = 0.02, q: float = 0.3,
             steps: int = 10, kernel: Callable[[np.ndarray], np.ndarray] | None = None,
             wom: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, runs: int = 1,
             seed: int = 0) -> AdoptionRun:
    """Exposure-to-adoption step plus word of mouth to grid neighbours, `runs` replicates vectorised.

    freq: (n_users,) campaign exposures per user (e.g. ``_cell_vector`` of the chosen cells, or
    ``Exposure.M[:, cells].sum(1)``). home: (n_users,) home cell (-1: no home, never influenced by word of mouth).
    Step 0: adopt with 1 - (1 - p_exposure)^freq. Steps 1..steps: non-adopters adopt with 1 - exp(-q * pressure),
    pressure = wom(adopter weight grid, population grid) at their home cell (default :func:`moore_share`).
    kernel: cumulative influence kernel F on step ages (e.g. ``diffusion.power_kernel(1, 0.5, horizon=steps)``);
    an adopter of age a weighs (F(a + 1) - F(a)) / (F(1) - F(0)). None: constant weight (Bass imitation)."""
    freq = np.asarray(freq, float)
    home = np.asarray(home, np.int64)
    rng = np.random.default_rng(seed)
    n, C = len(freq), grid.n_cells
    wom = wom or moore_share
    housed = home >= 0
    pop = np.bincount(home[housed], minlength=C).astype(float).reshape(grid.ny, grid.nx)
    pop = np.broadcast_to(pop, (runs, grid.ny, grid.nx))
    step_of = np.full((runs, n), -1, np.int64)
    p_ad = 1 - (1 - p_exposure) ** freq
    step_of[rng.random((runs, n)) < p_ad[None, :]] = 0
    from_ad = (step_of == 0).sum(1)
    if kernel is not None:
        dF = np.diff(kernel(np.arange(steps + 2, dtype=float)))
        weight_of_age = dF / max(dF[0], 1e-12)
    curve = np.zeros((runs, steps + 1), np.int64)
    curve[:, 0] = from_ad
    h = np.broadcast_to(np.where(housed, home, 0)[None, :], (runs, n))
    flat = (np.arange(runs) * C)[:, None] + h                                 # (run, home cell) bins
    for t in range(1, steps + 1):
        adopted = step_of >= 0
        if kernel is None:
            w = adopted.astype(float)
        else:
            w = np.where(adopted, weight_of_age[np.clip(t - 1 - step_of, 0, steps)], 0.0)
        aw = np.bincount(flat.ravel(), (w * housed[None, :]).ravel(), runs * C)
        pressure = wom(aw.reshape(runs, grid.ny, grid.nx), pop).reshape(runs, C)
        pr = 1 - np.exp(-q * np.take_along_axis(pressure, h, 1))
        new = (~adopted) & housed[None, :] & (rng.random((runs, n)) < pr)
        step_of[new] = t
        curve[:, t] = (step_of >= 0).sum(1)
    return AdoptionRun(curve, step_of, from_ad, curve[:, -1] - from_ad)


# ---------------------------------------------------------------- temporal holdout

def _paired_ci(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """95% bootstrap CI over users of mean(a) - mean(b) (paired, a and b per-user 0/1)."""
    d = a.astype(float) - b.astype(float)
    if len(d) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def holdout(traj: Any, grid: CellGrid, *, train_days: Iterable[int], test_days: Iterable[int],
            ks: Iterable[int] = (5, 10, 20, 50), unit: str = "day", r: int = 1, min_freq: int = 3,
            random_seeds: int = 10, n_boot: int = 1000, seed: int = 0, oracle: bool = True) -> pd.DataFrame:
    """Place with train days, score reach on test days. One row per (k, method).

    Methods: lazy_greedy (on train reach / capped coverage r; fill=True), top_visits, top_visitors, random (mean of
    `random_seeds` draws among cells seen in train) and, with oracle=True, greedy fitted on the TEST days (in-sample
    ceiling, not a method). Audience: users with at least one test exposure. Columns: test reach share, effective
    reach share (>= min_freq), frequency, train reach share, and the paired user-bootstrap 95% CI of
    lazy_greedy - method on test reach share (`diff_lo`, `diff_hi`)."""
    c = traj_columns(traj)
    users = np.unique(c["user"])
    tr = exposures(c, grid, days=train_days, unit=unit, users=users)
    te = exposures(c, grid, days=test_days, unit=unit, users=users)
    aud = te.active
    rows = []
    for k in ks:
        picks = {"lazy_greedy": greedy_placement(tr.M, k, r=r, fill=True).cells,
                 "top_visits": top_visited(tr.M, k, by="visits"),
                 "top_visitors": top_visited(tr.M, k, by="visitors")}
        rand = [random_cells(tr.M, k, seed=seed + s) for s in range(random_seeds)]
        if oracle:
            picks["oracle_test"] = greedy_placement(te.M, k, r=r, fill=True).cells
        hit = {m: _cell_vector(te.M, cs)[aud] > 0 for m, cs in picks.items()}
        hit_rand = np.mean([_cell_vector(te.M, cs)[aud] > 0 for cs in rand], 0)
        for m, cs in list(picks.items()) + [("random", None)]:
            if m == "random":
                rf = [reach_frequency(te.M, cs, min_freq, aud) for cs in rand]
                rf = {key: float(np.mean([x[key] for x in rf])) for key in ("reach_share", "effective_reach",
                                                                             "frequency")}
                tr_share = float(np.mean([reach_frequency(tr.M, cs, 1, tr.active)["reach_share"] for cs in rand]))
                h = hit_rand
            else:
                rf = reach_frequency(te.M, cs, min_freq, aud)
                tr_share = reach_frequency(tr.M, cs, 1, tr.active)["reach_share"]
                h = hit[m]
            lo, hi = (0.0, 0.0) if m == "lazy_greedy" else _paired_ci(hit["lazy_greedy"], h, n_boot, seed)
            rows.append(dict(k=int(k), method=m, test_reach_share=rf["reach_share"],
                             test_effective_share=rf["effective_reach"] / max(int(aud.sum()), 1),
                             test_frequency=rf["frequency"], train_reach_share=tr_share,
                             diff_lo=lo, diff_hi=hi, audience=int(aud.sum())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- synthetic fixture

def toy_trajectories(n_users: int = 200, n_days: int = 7, n_slots: int = 48, side: int = 24, n_districts: int = 4,
                     n_hubs: int = 4, p_hub: float = 0.15, p_ping: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """Synthetic trajectories for tests and examples (NOT a mobility model). Users live in `n_districts` districts
    of Zipf-like size (district 0 largest); each district has `n_hubs` hubs that only its residents visit. At night
    users are at home, by day at a work cell near home, and in each day slot at one of their district's hubs with
    probability p_hub (personal hub taste). A ping is kept with probability p_ping. Coordinates 1..side.
    Because the big district's hubs collect most visits, top-k-visited picks overlap; coverage spreads out."""
    rng = np.random.default_rng(seed)
    size = 1.0 / np.arange(1, n_districts + 1)
    dist = rng.choice(n_districts, n_users, p=size / size.sum())
    centre = rng.integers(side // 6, side - side // 6 + 1, (n_districts, 2))
    spread = max(side // 8, 1)
    home = np.clip(centre[dist] + rng.integers(-spread, spread + 1, (n_users, 2)), 1, side)
    work = np.clip(home + rng.integers(-spread, spread + 1, (n_users, 2)), 1, side)
    hubs = np.clip(centre[:, None, :] + rng.integers(-spread, spread + 1, (n_districts, n_hubs, 2)), 1, side)
    taste = rng.dirichlet(np.ones(n_hubs), n_users)
    U, D, S = (a.ravel() for a in np.meshgrid(np.arange(n_users), np.arange(n_days), np.arange(n_slots),
                                              indexing="ij"))
    day = (S >= n_slots * 8 // 24) & (S < n_slots * 19 // 24)
    xy = np.where(day[:, None], work[U], home[U])
    go = day & (rng.random(len(U)) < p_hub)
    pick = (rng.random(len(U))[:, None] < np.cumsum(taste, 1)[U]).argmax(1)
    xy[go] = hubs[dist[U[go]], pick[go]]
    keep = rng.random(len(U)) < p_ping
    return pd.DataFrame(dict(user=U[keep], day=D[keep], slot=S[keep], x=xy[keep, 0], y=xy[keep, 1]))
