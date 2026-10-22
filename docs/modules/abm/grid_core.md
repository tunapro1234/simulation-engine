# simengine.abm.grid: lattice core

`simengine/abm/grid.py` is the base layer for grid agent-based models. It provides the environment, the agent
arrays, the schedulers, seeded random generators and four spatial metrics. It contains no model of its own:
Schelling, voter and resource models build on it. Import it directly (`import simengine.abm.grid as G`) until the
integrator registers it in `simengine.abm`.

## API

| name | what it does |
|---|---|
| `Grid(shape, torus=True, neighbourhood="moore", radius=1)` | H x W lattice that is either bounded or wraps at the edges (torus). Moore uses Chebyshev distance <= r (8 cells at r=1) and von Neumann uses Manhattan distance <= r (4 cells at r=1). On a torus each side must be at least 2r+1. |
| `Grid.neighbour_sum(values)` / `neighbour_count(mask)` | Sum of the neighbours' values, or the count of True neighbours, for every cell. Works over the last two axes, so a `(..., H, W)` stack takes one call. It uses `np.roll` on a torus and zero padding when bounded. |
| `Grid.degree()`, `Grid.k`, `Grid.offsets`, `Grid.kernel()` | Neighbours per cell (fewer at the edges when bounded), the size of an interior neighbourhood, the `(k, 2)` offsets, and the 0/1 kernel for use with `scipy.ndimage`. |
| `Grid.type_counts(lattice, n_types=None)` | `(K, H, W)` array: how many neighbours of each type every cell has. |
| `Grid.neighbour_table()` | `(H*W, k)` flat neighbour indices for moving single agents. When bounded, -1 marks a neighbour outside the grid. |
| `Grid.ravel(rows, cols)` / `Grid.unravel(cells)` | Convert between flat indices and (row, col) coordinates. |
| `EMPTY` (-1), `random_lattice(grid, fractions, seed)` | Lattice convention: an int array with `EMPTY` for a free cell and 0..K-1 for the agent type. `random_lattice` places exactly `round(f * H*W)` agents of each type at random. |
| `Agents(cell, kind, data)` | Struct-of-arrays agent set: flat cell index, type, and extra per-agent arrays. Has `from_lattice`, `lattice(grid)` (raises on collisions), `coords`, `subset` and `n`. |
| `random_order_step(n, act, rng)` | Asynchronous step: calls `act(i)` once per agent in a fresh random order. Each agent sees the moves made earlier in the same step. |
| `synchronous_step(state, rule, rng)` | Computes `new = rule(old, rng)` from a read-only copy of the state, so every unit reacts to the same previous state. A rule that writes into the old state raises an error. |
| `run(step, n_steps, seed, record, stop)` -> `RunLog(records, steps, stopped)` | Step loop with one seeded generator. `record(t)` is taken at t=0 and after every step. `stop(t)` can end the run early. |
| `make_rng(seed)`, `spawn_rngs(seed, n)` | `make_rng` builds a Generator from an int, None or SeedSequence, and passes a Generator through unchanged. `spawn_rngs` gives n independent streams through `SeedSequence.spawn`. |
| `morans_i(values, grid, mask=None)` | Global Moran's I with binary contiguity weights from the grid. Only cells in `mask` are included, and weights link only included cells. |
| `moran_permutation(values, grid, mask, n_perm=999, seed=0)` | Returns `(I, pseudo p)`. The values are shuffled over the included cells and the permutations are batched in vectorised chunks. |
| `dissimilarity(lattice, block, a=0, b=1)` | Duncan dissimilarity index D computed over block tiles. Edge tiles may be partial. |
| `same_type_share(lattice, grid, per_cell=False)` | Mean share of like neighbours among occupied neighbours. Agents with no occupied neighbour are left out. |
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
- **Permutation p-value**: one-sided and folded, as PySAL esda's `p_sim` describes it. If m is the number of
  permuted I at least as large as the observed I, then `p = (min(m, n_perm - m) + 1) / (n_perm + 1)`. This follows
  the description of that convention. PySAL code was not copied or run here.
- **Dissimilarity index** (Duncan & Duncan 1955, *American Sociological Review* 20:210-217):
  `D = 1/2 sum_j |a_j/A - b_j/B|` over areal units. Here the areal units are square (or rectangular) tiles of the
  lattice.
- **Same-type neighbour share**: the "like neighbours" fraction that Schelling (1971, *Journal of Mathematical
  Sociology* 1:143-186) uses for agent happiness, averaged over agents.
- **Gini**: `sum_i (2i - n - 1) x_(i) / (n sum x)` with x sorted ascending. This equals the mean absolute difference
  divided by twice the mean. The maximum is `(n-1)/n`, with no small-sample correction.

## Limitations

- Only square lattices: there are no hexagonal or irregular graphs, and the weights are binary with no distance
  decay. The shift cost grows with k, which is 24 for Moore at r=2, so at large radii a kernel convolution
  (`Grid.kernel()` + scipy) can be faster.
- `random_order_step` is a Python loop over agents, which is the price of a true asynchronous update. Models that
  can tolerate it should vectorise inside `act`, or use `synchronous_step`.
- `Agents` does not track occupancy by itself. Collisions are detected only when `lattice()` is built, so a model
  that moves agents must keep its own occupancy lattice (or rebuild it).
- `dissimilarity` compares only two types at a time and depends on tile size (the modifiable areal unit problem).
  `moran_permutation` gives a one-sided pseudo p-value, not an analytical z-score.
- These are infrastructure numbers: reference values and speed, not predictive accuracy. No real data was used.
