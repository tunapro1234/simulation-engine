"""2-D lattice core for grid ABMs: environment, agent arrays, schedulers, seeded RNG and spatial metrics.

Other grid models (Schelling, voter, Sugarscape-style) build on this file; it holds no model of its own.

* :class:`Grid`   : H x W lattice, bounded or torus, Moore or von Neumann neighbourhood of any radius. Neighbour
  counts / sums are vectorised shifts (``np.roll`` on a torus, zero padding when bounded) over the last two axes,
  so a stack ``(..., H, W)`` is handled in one call. :meth:`Grid.neighbour_table` gives flat neighbour indices
  for per-agent moves.
* Lattice convention: an int array (H, W), ``EMPTY`` (-1) for a free cell, 0..K-1 for the agent type.
  :func:`random_lattice` fills one with exact type counts; :class:`Agents` is the struct-of-arrays view
  (flat cell index, type, extra per-agent arrays) and converts both ways.
* Schedulers: :func:`random_order_step` (asynchronous, each agent sees the latest state, fresh random order per
  step), :func:`synchronous_step` (new state computed from a frozen copy of the old one) and :func:`run`
  (step loop with records and an early stop). :func:`make_rng` / :func:`spawn_rngs` give seeded generators.
* Metrics: :func:`morans_i` (+ :func:`moran_permutation` pseudo p-value), :func:`dissimilarity` (Duncan index on
  block tiles), :func:`same_type_share` (Schelling's mean share of like neighbours), :func:`gini`.

Usage::

    import simengine.abm.grid as G
    g = G.Grid((50, 50), torus=True, neighbourhood="moore")
    lat = G.random_lattice(g, [0.45, 0.45], seed=0)          # 10% empty
    G.same_type_share(lat, g)                                # ~0.5 for a random 50/50 mix
    G.morans_i((lat == 0).astype(float), g, mask=lat >= 0)   # ~0 (E[I] = -1/(n-1))
    G.dissimilarity(lat, block=5)                            # small but > 0 from sampling noise

Measured numbers (reference configurations, timings) are in docs/modules/abm/grid_core.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

EMPTY = -1
NEIGHBOURHOODS = ("moore", "von_neumann")
SeedLike = int | None | np.random.Generator | np.random.SeedSequence


# ---------------------------------------------------------------- RNG
def make_rng(seed: SeedLike = None) -> np.random.Generator:
    """Generator from an int / None / SeedSequence; a Generator is passed through unchanged (shared stream)."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def spawn_rngs(seed: SeedLike, n: int) -> list[np.random.Generator]:
    """n statistically independent generators (e.g. one per replicate / per agent group), reproducible from seed."""
    if isinstance(seed, np.random.Generator):
        return list(seed.spawn(n))
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    return [np.random.default_rng(s) for s in ss.spawn(n)]


# ---------------------------------------------------------------- environment
def _offsets(neighbourhood: str, radius: int) -> np.ndarray:
    r = np.arange(-radius, radius + 1)
    dr, dc = (a.ravel() for a in np.meshgrid(r, r, indexing="ij"))
    keep = (np.abs(dr) + np.abs(dc) <= radius) if neighbourhood == "von_neumann" else np.ones(dr.shape, bool)
    keep &= (dr != 0) | (dc != 0)
    return np.stack([dr[keep], dc[keep]], 1)


@dataclass(frozen=True)
class Grid:
    """H x W lattice. torus=True wraps both edges; bounded cells near the edge have fewer neighbours.

    neighbourhood: "moore" (Chebyshev distance <= radius; 8 cells at radius 1) or "von_neumann"
    (Manhattan distance <= radius; 4 cells at radius 1). The centre cell is never its own neighbour.
    """
    shape: tuple[int, int]
    torus: bool = True
    neighbourhood: str = "moore"
    radius: int = 1
    offsets: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        h, w = (int(s) for s in self.shape)
        if h < 1 or w < 1:
            raise ValueError(f"shape must be positive, got {self.shape}")
        if self.neighbourhood not in NEIGHBOURHOODS:
            raise ValueError(f"neighbourhood must be one of {NEIGHBOURHOODS}, got {self.neighbourhood!r}")
        if int(self.radius) != self.radius or self.radius < 1:
            raise ValueError(f"radius must be an integer >= 1, got {self.radius!r}")
        object.__setattr__(self, "radius", int(self.radius))
        if self.torus and min(h, w) < 2 * self.radius + 1:
            raise ValueError(f"torus needs each side >= 2*radius+1 = {2 * self.radius + 1} (else a cell would "
                             f"count the same neighbour twice), got {self.shape}")
        object.__setattr__(self, "shape", (h, w))
        object.__setattr__(self, "offsets", _offsets(self.neighbourhood, self.radius))

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def k(self) -> int:
        """Neighbourhood size of an interior cell."""
        return len(self.offsets)

    def kernel(self) -> np.ndarray:
        """(2r+1, 2r+1) 0/1 weight kernel, centre 0 (for scipy.ndimage.correlate style use)."""
        r = self.radius
        out = np.zeros((2 * r + 1, 2 * r + 1), int)
        out[self.offsets[:, 0] + r, self.offsets[:, 1] + r] = 1
        return out

    # --- vectorised neighbourhood sums
    def neighbour_sum(self, values: np.ndarray) -> np.ndarray:
        """Sum of ``values`` over each cell's neighbours, over the last two axes ((..., H, W) -> same shape).

        Bool and ints narrower than 64 bit are summed in (at least) int32 (uint32 in int64) so k * max does not
        wrap; 64-bit ints and floats keep their dtype. Outside cells contribute 0 when bounded.
        """
        a = np.asarray(values)
        if a.shape[-2:] != self.shape:
            raise ValueError(f"last two axes must be {self.shape}, got {a.shape}")
        if a.dtype.kind in "biu" and a.dtype.itemsize < 8:      # not uint64: int32 + uint64 promotes to float64
            a = a.astype(np.promote_types(a.dtype, np.int32), copy=False)
        out = np.zeros_like(a)
        if self.torus:
            for dr, dc in self.offsets:
                out += np.roll(a, (-dr, -dc), axis=(-2, -1))       # out[r, c] += a[r + dr, c + dc]
            return out
        r, (h, w) = self.radius, self.shape
        p = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(r, r), (r, r)])
        for dr, dc in self.offsets:
            out += p[..., r + dr:r + dr + h, r + dc:r + dc + w]
        return out

    def neighbour_count(self, mask: np.ndarray) -> np.ndarray:
        """Number of True neighbours per cell (int32)."""
        return self.neighbour_sum(np.asarray(mask, bool))

    def degree(self) -> np.ndarray:
        """(H, W) number of in-grid neighbours per cell (k everywhere on a torus)."""
        return self.neighbour_count(np.ones(self.shape, bool))

    def type_counts(self, lattice: np.ndarray, n_types: int | None = None) -> np.ndarray:
        """(n_types, H, W): neighbours of each type around every cell (EMPTY cells ignored)."""
        lat = np.asarray(lattice)
        n_types = int(lat.max()) + 1 if n_types is None else n_types
        onehot = lat[None] == np.arange(max(n_types, 0))[:, None, None]
        return self.neighbour_count(onehot)

    # --- index helpers
    def ravel(self, rows, cols) -> np.ndarray:
        return np.ravel_multi_index((np.asarray(rows), np.asarray(cols)), self.shape)

    def unravel(self, cells) -> tuple[np.ndarray, np.ndarray]:
        return np.unravel_index(np.asarray(cells), self.shape)

    def neighbour_table(self) -> np.ndarray:
        """(H*W, k) int64 flat indices of each cell's neighbours, in ``offsets`` order; -1 = outside (bounded)."""
        h, w = self.shape
        rr, cc = np.divmod(np.arange(self.size), w)
        nr = rr[:, None] + self.offsets[None, :, 0]
        nc = cc[:, None] + self.offsets[None, :, 1]
        if self.torus:
            return (nr % h) * w + (nc % w)
        inside = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
        return np.where(inside, nr * w + nc, -1)


# ---------------------------------------------------------------- agents
def random_lattice(grid: Grid, fractions: Sequence[float], seed: SeedLike = None) -> np.ndarray:
    """Int lattice with round(f * H*W) agents of each type (fractions sum <= 1); the rest EMPTY, placed at random."""
    f = np.asarray(fractions, float)
    if (f < 0).any() or f.sum() > 1 + 1e-9:
        raise ValueError(f"fractions must be >= 0 and sum to <= 1, got {fractions}")
    counts = np.round(f * grid.size).astype(int)
    if counts.sum() > grid.size:                       # rounding overshoot: trim the largest type
        counts[np.argmax(counts)] -= counts.sum() - grid.size
    flat = np.full(grid.size, EMPTY, int)
    flat[:counts.sum()] = np.repeat(np.arange(len(f)), counts)
    return make_rng(seed).permutation(flat).reshape(grid.shape)


@dataclass
class Agents:
    """Struct-of-arrays agent set: ``cell`` (N,) flat cell index, ``kind`` (N,) type >= 0, ``data`` extra (N, ...)
    arrays (wealth, threshold, ...). Index i is the agent id; rows stay aligned under :meth:`subset`."""
    cell: np.ndarray
    kind: np.ndarray
    data: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.cell = np.asarray(self.cell, np.int64)
        self.kind = np.asarray(self.kind, np.int64)
        if self.cell.shape != self.kind.shape or self.cell.ndim != 1:
            raise ValueError("cell and kind must be 1-D arrays of the same length")
        for k, v in self.data.items():
            if len(v) != len(self.cell):
                raise ValueError(f"data[{k!r}] has length {len(v)}, expected {len(self.cell)}")

    @property
    def n(self) -> int:
        return len(self.cell)

    @classmethod
    def from_lattice(cls, lattice: np.ndarray, **data) -> "Agents":
        """One agent per occupied cell, in flat (row-major) cell order."""
        flat = np.asarray(lattice).ravel()
        cell = np.flatnonzero(flat != EMPTY)
        return cls(cell, flat[cell], dict(data))

    def lattice(self, grid: Grid) -> np.ndarray:
        """(H, W) int lattice of agent types; raises if two agents share a cell, a cell is off the grid (e.g. a -1
        from a bounded :meth:`Grid.neighbour_table`, which numpy would silently wrap to the last cell) or a kind < 0."""
        if self.n and (self.cell.min() < 0 or self.cell.max() >= grid.size):
            raise ValueError(f"agent cell outside 0..{grid.size - 1}")
        if self.n and self.kind.min() < 0:
            raise ValueError("agent kind must be >= 0 (EMPTY is reserved for free cells)")
        if np.unique(self.cell).size != self.n:
            raise ValueError("two agents share a cell")
        flat = np.full(grid.size, EMPTY, int)
        flat[self.cell] = self.kind
        return flat.reshape(grid.shape)

    def coords(self, grid: Grid) -> np.ndarray:
        """(N, 2) row, col."""
        return np.stack(grid.unravel(self.cell), 1)

    def subset(self, idx) -> "Agents":
        return Agents(self.cell[idx], self.kind[idx], {k: np.asarray(v)[idx] for k, v in self.data.items()})


# ---------------------------------------------------------------- schedulers
def random_order_step(n: int, act: Callable[[int], Any], rng: np.random.Generator) -> np.ndarray:
    """Asynchronous update: call act(i) for every agent id 0..n-1 in a fresh random order; act mutates the shared
    state so later agents see earlier moves. Returns the order used."""
    order = rng.permutation(n)
    for i in order:
        act(int(i))
    return order


def synchronous_step(state: np.ndarray, rule: Callable[[np.ndarray, np.random.Generator], np.ndarray],
                     rng: np.random.Generator) -> np.ndarray:
    """Synchronous update: new = rule(old, rng) with ``old`` a read-only copy, so every cell / agent reacts to the
    same previous state (a rule that writes into it raises). The result is always a fresh writeable array, also
    when the rule returns ``old`` or a view of it."""
    old = np.array(state, copy=True)
    old.flags.writeable = False
    new = np.asarray(rule(old, rng))
    if new.shape != old.shape:
        raise ValueError(f"rule returned shape {new.shape}, expected {old.shape}")
    return new.copy() if (not new.flags.writeable or np.may_share_memory(new, old)) else new


@dataclass
class RunLog:
    records: list = field(default_factory=list)   # record(t) at t = 0 (before the first step) .. steps
    steps: int = 0                                 # steps actually run
    stopped: bool = False                          # True when stop(t) ended the run early


def run(step: Callable[[int, np.random.Generator], Any], n_steps: int, seed: SeedLike = None,
        record: Callable[[int], Any] | None = None, stop: Callable[[int], bool] | None = None) -> RunLog:
    """Call step(t, rng) for t = 1..n_steps with one seeded generator; record(t) after each step (and at t=0);
    stop(t) -> True ends the run after step t (e.g. no agent moved)."""
    rng = make_rng(seed)
    log = RunLog()
    if record is not None:
        log.records.append(record(0))
    for t in range(1, n_steps + 1):
        step(t, rng)
        log.steps = t
        if record is not None:
            log.records.append(record(t))
        if stop is not None and stop(t):
            log.stopped = True
            break
    return log


# ---------------------------------------------------------------- spatial metrics
def _moran_batch(z: np.ndarray, grid: Grid, mask: np.ndarray, s0: float) -> np.ndarray:
    """Moran's I for centred values z (..., H, W), zero outside mask."""
    n = mask.sum()
    num = (z * grid.neighbour_sum(z)).sum(axis=(-2, -1))
    den = (z * z).sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return n / s0 * num / den


def _moran_prep(values, grid: Grid, mask):
    x = np.asarray(values, float)
    if x.shape != grid.shape:
        raise ValueError(f"values must have shape {grid.shape}, got {x.shape}")
    m = np.isfinite(x) if mask is None else (np.asarray(mask, bool) & np.isfinite(x))
    s0 = float((grid.neighbour_count(m) * m).sum())        # sum of binary weights between included cells
    if m.sum() < 2 or s0 == 0:
        raise ValueError("need >= 2 included cells with at least one included neighbour pair")
    return x, m, s0


def morans_i(values: np.ndarray, grid: Grid, mask: np.ndarray | None = None) -> float:
    """Global Moran's I with binary contiguity weights from ``grid`` (Moran 1950; Cliff & Ord form):

        I = n / S0 * sum_ij w_ij z_i z_j / sum_i z_i^2,   z = x - mean(x),  S0 = sum_ij w_ij,

    over the cells in ``mask`` (default: finite values); weights only link included cells. E[I] = -1/(n-1) under
    no spatial association; near +1 clustered, -1 for a von Neumann checkerboard (0 with Moore: the diagonals share
    the type). For a type lattice pass (lat == a) with mask lat >= 0.
    NaN if the included values are constant.
    """
    x, m, s0 = _moran_prep(values, grid, mask)
    z = np.where(m, x - x[m].mean(), 0.0)
    return float(_moran_batch(z, grid, m, s0))


def moran_permutation(values: np.ndarray, grid: Grid, mask: np.ndarray | None = None, n_perm: int = 999,
                      seed: SeedLike = 0, chunk: int = 64, alternative: str = "directed") -> tuple[float, float]:
    """(I, pseudo p): values shuffled over the included cells n_perm times; p = (m + 1) / (n_perm + 1) with m the
    number of permuted I at least as extreme as the observed one, as in PySAL esda's ``calculate_significance``:

    * "directed" (default; esda's legacy ``p_sim``): m = min(#{sim >= I}, n_perm - #{sim >= I}), i.e. the tail the
      observed I lies in, not doubled. esda notes this runs small; use "two-sided" for an unbiased test.
    * "greater" (clustering) m = #{sim >= I}; "lesser" (dispersion) m = #{sim <= I};
    * "two-sided": p = min(2 (min(#{sim >= I}, #{sim <= I}) + 1) / (n_perm + 1), 1).

    Batched in ``chunk`` permutations per vectorised call (capped so a batch holds <= 2**22 cells, ~32 MB per
    temporary). (nan, nan) when the included values are constant."""
    if alternative not in ("directed", "greater", "lesser", "two-sided"):
        raise ValueError(f"alternative must be directed / greater / lesser / two-sided, got {alternative!r}")
    x, m, s0 = _moran_prep(values, grid, mask)
    z = np.where(m, x - x[m].mean(), 0.0)
    obs = float(_moran_batch(z, grid, m, s0))
    if not np.isfinite(obs):
        return obs, float("nan")
    chunk = max(1, min(chunk, 2 ** 22 // grid.size))
    rng, zi, idx = make_rng(seed), z[m], np.flatnonzero(m.ravel())
    sims = []
    for i in range(0, n_perm, chunk):
        b = min(chunk, n_perm - i)
        zz = np.zeros((b, grid.size))
        zz[:, idx] = rng.permuted(np.broadcast_to(zi, (b, zi.size)), axis=1)
        sims.append(_moran_batch(zz.reshape(b, *grid.shape), grid, m, s0))
    sims = np.concatenate(sims) if sims else np.empty(0)
    ge, le = int((sims >= obs).sum()), int((sims <= obs).sum())
    if alternative == "two-sided":
        return obs, min(2 * (min(ge, le) + 1) / (n_perm + 1), 1.0)
    m = {"directed": min(ge, n_perm - ge), "greater": ge, "lesser": le}[alternative]
    return obs, (m + 1) / (n_perm + 1)


def dissimilarity(lattice: np.ndarray, block: int | tuple[int, int], a: int = 0, b: int = 1) -> float:
    """Duncan & Duncan (1955) index D = 1/2 sum_j |a_j / A - b_j / B| over block tiles of the lattice (edge tiles
    may be partial). 0 = every tile has the city-wide a:b mix, 1 = complete separation. Depends on the tile size
    (modifiable areal unit problem) and is > 0 for a random mix on small tiles."""
    lat = np.asarray(lattice)
    bh, bw = (block, block) if np.isscalar(block) else block
    if lat.ndim != 2 or bh < 1 or bw < 1:
        raise ValueError(f"need a 2-D lattice and block sides >= 1, got shape {lat.shape}, block {block}")
    h, w = lat.shape
    tile = (np.arange(h)[:, None] // bh) * (-(-w // bw)) + np.arange(w)[None, :] // bw
    n_t = int(tile.max()) + 1
    ca = np.bincount(tile[lat == a], minlength=n_t)
    cb = np.bincount(tile[lat == b], minlength=n_t)
    if ca.sum() == 0 or cb.sum() == 0:
        raise ValueError(f"both types {a} and {b} must be present")
    return float(0.5 * np.abs(ca / ca.sum() - cb / cb.sum()).sum())


def same_type_share(lattice: np.ndarray, grid: Grid, per_cell: bool = False):
    """Mean like-neighbour share (the quantity behind Schelling's happiness rule): for each agent, like neighbours /
    occupied neighbours; mean of these per-agent ratios over agents with >= 1 occupied neighbour (not the pooled
    ratio sum(like) / sum(occupied)). per_cell=True returns the (H, W) share (NaN for empty / isolated cells)."""
    lat = np.asarray(lattice)
    if lat.size and lat.min() < EMPTY:                 # a -2 would index counts from the end (silent wrong share)
        raise ValueError(f"lattice values must be EMPTY ({EMPTY}) or a type >= 0, got min {lat.min()}")
    occ = lat != EMPTY
    if not occ.any():
        return np.full(lat.shape, np.nan) if per_cell else float("nan")
    counts = grid.type_counts(lat)                                # (K, H, W)
    total = counts.sum(0)
    same = np.take_along_axis(counts, np.where(occ, lat, 0)[None], 0)[0]
    ok = occ & (total > 0)
    share = np.full(lat.shape, np.nan)
    share[ok] = same[ok] / total[ok]
    if per_cell:
        return share
    return float(share[ok].mean()) if ok.any() else float("nan")


def gini(x: np.ndarray) -> float:
    """Population Gini of non-negative values: sum_i (2i - n - 1) x_(i) / (n sum x), x sorted, i = 1..n.
    0 = equal, (n-1)/n = one holds everything. NaN for an all-zero or empty input."""
    v = np.sort(np.asarray(x, float).ravel())
    if (v < 0).any():
        raise ValueError("gini needs non-negative values")
    n, s = v.size, v.sum()
    if n == 0 or s == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(((2 * i - n - 1) * v).sum() / (n * s))
