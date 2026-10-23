# simengine.abm.grid: lattice core

`simengine/abm/grid.py` is the base layer for grid agent-based models. It provides the environment, the agent
arrays, the schedulers, seeded random generators and four spatial metrics. It contains no model of its own:
Schelling, voter and resource models build on it. Its public names are exported from `simengine.abm` (except
`run`, which stays `simengine.abm.grid.run`); the examples import the module as `import simengine.abm.grid as G`.

## API

| name | what it does |
|---|---|
| `Grid(shape, torus=True, neighbourhood="moore", radius=1)` | H x W lattice that is either bounded or wraps at the edges (torus). Moore uses Chebyshev distance <= r (8 cells at r=1) and von Neumann uses Manhattan distance <= r (4 cells at r=1). On a torus each side must be at least 2r+1. |
| `Grid.neighbour_sum(values)` / `neighbour_count(mask)` | Sum of the neighbours' values, or the count of True neighbours, for every cell. Works over the last two axes, so a `(..., H, W)` stack takes one call. It uses `np.roll` on a torus and zero padding when bounded. Bool and ints narrower than 64 bit are summed in at least int32 (so a uint8 lattice cannot wrap); 64-bit ints and floats keep their dtype. |
| `Grid.degree()`, `Grid.k`, `Grid.offsets`, `Grid.kernel()` | Neighbours per cell (fewer at the edges when bounded), the size of an interior neighbourhood, the `(k, 2)` offsets, and the 0/1 kernel for use with `scipy.ndimage`. |
| `Grid.type_counts(lattice, n_types=None)` | `(K, H, W)` array: how many neighbours of each type every cell has. |
| `Grid.neighbour_table()` | `(H*W, k)` flat neighbour indices for moving single agents. When bounded, -1 marks a neighbour outside the grid. |
| `Grid.ravel(rows, cols)` / `Grid.unravel(cells)` | Convert between flat indices and (row, col) coordinates. |
| `EMPTY` (-1), `random_lattice(grid, fractions, seed)` | Lattice convention: an int array with `EMPTY` for a free cell and 0..K-1 for the agent type. `random_lattice` places exactly `round(f * H*W)` agents of each type at random. |
| `Agents(cell, kind, data)` | Struct-of-arrays agent set: flat cell index, type, and extra per-agent arrays. Has `from_lattice`, `lattice(grid)` (raises on collisions, on a cell outside the grid such as the -1 of a bounded `neighbour_table`, and on a negative kind), `coords`, `subset` and `n`. |
| `random_order_step(n, act, rng)` | Asynchronous step: calls `act(i)` once per agent in a fresh random order. Each agent sees the moves made earlier in the same step. |
| `synchronous_step(state, rule, rng)` | Computes `new = rule(old, rng)` from a read-only copy of the state, so every unit reacts to the same previous state. A rule that writes into the old state raises an error. The result is always a fresh writeable array, also when the rule returns `old` itself. |
| `run(step, n_steps, seed, record, stop)` -> `RunLog(records, steps, stopped)` | Step loop with one seeded generator. `record(t)` is taken at t=0 and after every step. `stop(t)` can end the run early. |
| `make_rng(seed)`, `spawn_rngs(seed, n)` | `make_rng` builds a Generator from an int, None or SeedSequence, and passes a Generator through unchanged. `spawn_rngs` gives n independent streams through `SeedSequence.spawn`. |
| `morans_i(values, grid, mask=None)` | Global Moran's I with binary contiguity weights from the grid. Only cells in `mask` are included, and weights link only included cells. |
| `moran_permutation(values, grid, mask, n_perm=999, seed=0, alternative="directed")` | Returns `(I, pseudo p)`. The values are shuffled over the included cells and the permutations are batched in vectorised chunks. `alternative` is `"directed"`, `"greater"`, `"lesser"` or `"two-sided"` (see below). `(nan, nan)` for constant values. |
| `dissimilarity(lattice, block, a=0, b=1)` | Duncan dissimilarity index D computed over block tiles. Edge tiles may be partial. |
| `same_type_share(lattice, grid, per_cell=False)` | Mean of the per-agent shares of like neighbours among occupied neighbours (not the pooled ratio). Agents with no occupied neighbour are left out. Values below `EMPTY` raise. |
| `gini(x)` | Population Gini of non-negative values. |

```python
import numpy as np
import simengine.abm.grid as G

g = G.Grid((50, 50), torus=True, neighbourhood="moore")
lat = G.random_lattice(g, [0.45, 0.45], seed=0)                 # two types, 10% empty
ag = G.Agents.from_lattice(lat, wealth=np.ones((lat >= 0).sum()))
rng = G.make_rng(0)

def act(i):                                                     # e.g. an unhappy agent jumps to a random empty cell
    ...

log = G.run(lambda t, r: G.random_order_step(ag.n, act, r), 50, seed=0,
            record=lambda t: G.same_type_share(ag.lattice(g), g))
```

## Definitions and provenance

These are the textbook definitions. No code was ported from another project.

- **Moran's I** (Moran 1950, *Biometrika* 37:17-23; written here in the Cliff & Ord form):
  `I = n / S0 * sum_ij w_ij z_i z_j / sum_i z_i^2`, where `z = x - mean(x)`, `S0 = sum_ij w_ij` and `w_ij` is 1 for
  grid neighbours. Under no spatial association `E[I] = -1/(n-1)`. The tests check it against a dense
  weight-matrix computation on a masked grid, for both boundary modes and both neighbourhoods.
- **Permutation p-value**: `p = (M + 1) / (n_perm + 1)` with the tail rules of PySAL esda's
  `esda.significance.calculate_significance` (checked against the esda source on GitHub, `main` branch, 2026-10-22;
  no code copied, esda not run here). With `g = #{sim >= I}` and `l = #{sim <= I}`: `"directed"` (the default, and
  esda's legacy `Moran.p_sim`) uses `M = min(g, n_perm - g)`, the tail the observed I falls in, not doubled; esda's
  own source notes that this makes p-values too small. `"greater"` uses `M = g` (test for clustering), `"lesser"`
  uses `M = l` (dispersion), and `"two-sided"` gives `min(2 (min(g, l) + 1) / (n_perm + 1), 1)`. esda's
  `"folded"` (absolute deviation from the permutation mean) is not implemented; the module's earlier docstring
  called the directed rule "folded", which was the wrong esda name. For a test decided in advance, pass
  `"greater"` or `"two-sided"`.
- **Dissimilarity index** (Duncan & Duncan 1955, *American Sociological Review* 20:210-217):
  `D = 1/2 sum_j |a_j/A - b_j/B|` over areal units. Here the areal units are square (or rectangular) tiles of the
  lattice.
- **Same-type neighbour share**: the "like neighbours" fraction that Schelling (1971, *Journal of Mathematical
  Sociology* 1:143-186) uses for agent happiness, averaged over agents.
- **Gini**: `sum_i (2i - n - 1) x_(i) / (n sum x)` with x sorted ascending. This equals the mean absolute difference
  divided by twice the mean. The maximum is `(n-1)/n`, with no small-sample correction.

## Measured numbers (this port)

These come from my own runs on 2026-10-22 (Python 3.12.3, numpy 2.5.4, 32-core AWS box, single thread). The script
lives outside the repo. To reproduce, run it from the repo root:
`PYTHONPATH=. python measure.py` (the scratch copy is kept outside the repo). It contains only the calls
listed below. A reviewer re-ran it the same evening: every reference value and null level matched to the last
digit shown. The timings vary with machine load: on a loaded box (load average ~32 on 32 cores) the 1000x1000
count took 2.2-3.7 ms, `morans_i` 25-31 ms, `same_type_share` 39-44 ms and the 999-permutation test 159-164 ms,
while scipy stayed at 7.9 ms. So treat the speed ratio as roughly 2-3.5x, not an exact figure.

**Null values: a random 45/45/10 lattice, 100x100 torus, Moore, 20 seeds (mean +/- sd).** These are the levels a
segregation model has to move away from:

| metric | value |
|---|---|
| `same_type_share` | 0.5002 +/- 0.0023 |
| `morans_i(lat == 0, mask=lat >= 0)` | 0.0004 +/- 0.0046 (E[I] = -0.0001) |
| `dissimilarity(block=10)` | 0.0825 +/- 0.0055 |
| `dissimilarity(block=5)` | 0.1660 +/- 0.0042 |

D is clearly above 0 for a random mix, and halving the tile side doubles it, so always report D together with its
tile size and this null level.

**Reference patterns, 100x100 torus.** A checkerboard gives I = -1.000 with von Neumann neighbours but I = 0.000
with Moore neighbours, because the four diagonal neighbours share the cell's type. Its share is 0.0 (von Neumann)
or 0.5 (Moore), and D(10) = 0. Two halves give I = 0.970 (Moore) or 0.980 (von Neumann), share 0.985 or 0.990, and
D(10) = 1.

**Timings** (ms per call, mean of 20 calls after one warm-up):

| grid | `neighbour_count` torus / bounded | `same_type_share` | `morans_i` | `dissimilarity(10)` |
|---|---|---|---|---|
| 100x100 | 0.09 / 0.06 | 0.27 | 0.25 | 0.09 |
| 1000x1000 | 2.19 / 2.27 | 31.98 | 23.22 | 10.96 |

`moran_permutation` with 999 permutations on a 100x100 grid takes 139 ms. On the same 1000x1000 count,
`scipy.ndimage.correlate(mode="grid-wrap")` takes 7.71 ms, so the roll/pad shifts were about 3.5x faster at
radius 1 in that run (2-3.5x across re-runs). A test checks that both give identical results at radius 2.

## Limitations

- Only square lattices: there are no hexagonal or irregular graphs, and the weights are binary with no distance
  decay. The shift cost grows with k, which is 24 for Moore at r=2, so at large radii a kernel convolution
  (`Grid.kernel()` + scipy) can be faster.
- `random_order_step` is a Python loop over agents, which is the price of a true asynchronous update. Models that
  can tolerate it should vectorise inside `act`, or use `synchronous_step`.
- `Agents` does not track occupancy by itself. Collisions are detected only when `lattice()` is built, so a model
  that moves agents must keep its own occupancy lattice (or rebuild it).
- `dissimilarity` compares only two types at a time and depends on tile size (the modifiable areal unit problem).
  `moran_permutation` gives a permutation pseudo p-value (default `"directed"`, which runs small), not an
  analytical z-score.
- These are infrastructure numbers: reference values and speed, not predictive accuracy. No real data was used.
