# simengine.abm.placement

Billboard / ad placement on a grid from trajectories. Given who was where and when, pick the cells that reach the
most people under a budget. Then check the pick on days it never saw, against the simple baselines.

| component | what it is |
|---|---|
| `traj_columns`, `CellGrid`, `exposures`, `home_cells` | trajectory protocol, billboard cells (blocks of raw cells), users x cells exposure matrix, home cell per user |
| `coverage`, `reach_frequency` | objective `F(S)`; media summary: reach, reach share, frequency, effective reach at r+, GRP, frequency histogram |
| `greedy_placement`, `Placement`, `online_bound` | greedy and lazy greedy (CELF) maximum coverage under `k` cells or a cost budget, plus the online upper bound on the optimum |
| `top_visited`, `random_cells` | baselines: top-k cells by visits or distinct visitors, random visited cells |
| `adoption`, `AdoptionRun`, `moore_share` | exposure-to-adoption step, then word of mouth to grid neighbours (replaceable hook, optional decay kernel) |
| `holdout` | temporal holdout: place on train days, score reach on test days, paired user-bootstrap CI against the greedy |
| `toy_trajectories` | synthetic fixture for tests and examples (not a mobility model) |

The module is not registered in `simengine/abm/__init__.py` yet. Import it directly:
`import simengine.abm.placement as pl`.

## Trajectory protocol

Any of the following, with five integer columns `user, day, slot, x, y`:

- a DataFrame (YJMob100K names `uid, d, t, x, y` are accepted, so the frames of `simengine.abm.mobility` and
  the output of `mobility.from_cell` work as they are);
- a dict of arrays;
- an `(n, 5)` array.

`CellGrid(x0, y0, nx, ny, size)` maps raw `(x, y)` to billboard cells of `size x size` raw cells, with
`cell = row * nx + col`. This is not the `mobility.to_cell` code, which is x-major on the raw 200 x 200 grid. Points
outside the grid are dropped, including the YJMob `999` mask. `CellGrid.fit` takes the bounding box and ignores 999.

`exposures(traj, grid, days=, slots=, unit=, users=)` builds a sparse users x cells matrix. The `unit` argument sets
what counts as an exposure:

- `"day"` (default): distinct (user, day, cell), so one sighting per day per billboard;
- `"slot"`: every ping;
- `"user"`: 0/1.

Pass the same `users` array for the train and test windows so the rows line up.

## Objective and algorithms

With `c_u(S)` the exposures of user `u` to the cell set `S`:

    F(S) = sum_u w_u * min(c_u(S), r) / r

- With `r = 1`, `F(S)` is reach, i.e. maximum coverage.
- With `r > 1`, it is frequency-capped coverage: each user counts up to `r` exposures. This is still monotone and
  submodular, because it is a concave function of a modular one.
- "Effective reach at r+" (users with `c_u >= r`) is reported but not optimised directly, because it is not
  submodular.

The methods were checked against Leskovec, Krause, Guestrin, Faloutsos, VanBriesen, Glance, *Cost-effective
Outbreak Detection in Networks*, KDD 2007 (cs.cmu.edu/~jure/pubs/detect-kdd07.pdf):

- **Unit cost, `k` cells.** Greedy gives `F(A_G) >= (1 - 1/e) max_{|A|=k} F(A)`. This is their Theorem 2, which
  cites Nemhauser, Wolsey and Fisher 1978.
- **Cell costs and budget `B`.** CELF runs two greedies, both skipping cells the remaining budget cannot pay for:
  the unit-cost greedy and the benefit/cost greedy. It returns the better of the two, and
  `max(F(A_UC), F(A_CB)) >= 1/2 (1 - 1/e) OPT`. This is their Theorem 3, first proved for budgeted max cover by
  Khuller, Moss and Naor 1999. The test suite reproduces the paper's counterexample: a cheap cell with ratio 2
  against an expensive one with ratio 1. Benefit/cost greedy alone picks the cheap cell and ends with 2. CELF
  picks the expensive cell and ends with 10.
- **Lazy evaluation.** Stale marginal gains are upper bounds, by submodularity. A heap re-evaluates only its top
  entry. The output is the same as the plain greedy: ties go to the lowest cell index, and gains are rounded to
  1e-9 before comparison.
- **`Placement.bound`** is their Theorem 4: `OPT <= F(A) + the best fractional knapsack of single-cell gains
  delta_s given A`. It holds for any placement, so `online_bound(M, top_visited(M, k), k=k)` also bounds OPT for a
  baseline's cells.
- **Zero gain.** Greedy stops when no cell adds value. With `fill=True` it spends the rest of the budget on cells
  ranked by distinct visitors. `holdout` uses `fill=True`, so every method places exactly `k` cells.

## Adoption step

`adoption(freq, home, grid, p_exposure=0.02, q=0.3, steps=10, kernel=None, wom=None, runs=1, seed=0)` simulates
`runs` replicates, vectorised:

1. **Step 0 (campaign).** A user with `f` exposures adopts with probability `1 - (1 - p_exposure)^f`.
2. **Steps 1..steps (word of mouth).** A non-adopter with home cell `h` adopts with probability
   `1 - exp(-q * pressure_h)`. By default, `pressure` is the adopter share in the 3x3 (Moore) neighbourhood of
   home cells (`moore_share`), which is a discrete Bass imitation term on the grid. Users without a home cell are
   never influenced.
3. **`kernel`.** A cumulative influence kernel `F`, for example
   `simengine.abm.diffusion.power_kernel(c, theta, horizon=steps)`. An adopter of age `a` then weighs
   `(F(a+1) - F(a)) / (F(1) - F(0))`, so influence decays. This is where the module reuses
   `simengine.abm.diffusion`. The Hawkes / SEISMIC predictors there fit cascade curves, not grids, so nothing else
   from that file fits here.
4. **`wom(adopter_weight_grid, population_grid) -> pressure_grid`** is the hook. Both arrays have shape
   `(runs, ny, nx)`. Pass a different neighbourhood, a distance kernel or an empirical contact matrix.

`home_cells` gives each user's modal cell over night slots (0-6:59 and 21:00-23:59 for 30-min slots), falling back
to the modal cell overall.

There is no adoption ground truth anywhere in this module. `p_exposure` and `q` are illustrative values, not
calibrated ones.

## Minimal usage

```python
import simengine.abm.placement as pl
traj = pl.toy_trajectories(n_users=300, n_days=14, seed=0)        # or a DataFrame uid, d, t, x, y
grid = pl.CellGrid.fit(traj, size=2)
ex = pl.exposures(traj, grid, days=range(10))
p = pl.greedy_placement(ex.M, k=5)                                 # p.cells, p.value, p.bound, p.evaluations
pl.reach_frequency(ex.M, p.cells, min_freq=3)
cell_costs = 1 + ex.M.sum(0).A.ravel() / 100                       # illustrative: busier cells cost more
pl.greedy_placement(ex.M, costs=cell_costs, budget=20.0)          # CELF; .extra["rule"] says which greedy won
df = pl.holdout(traj, grid, train_days=range(10), test_days=range(10, 14), ks=(5, 10))
freq = ex.M[:, p.cells].sum(1).A.ravel()
run = pl.adoption(freq, pl.home_cells(traj, grid, ex.users), grid, runs=50)   # run.mean_curve
```

`holdout` returns one row per `(k, method)`. The methods are:

- `lazy_greedy`;
- `top_visits` and `top_visitors`;
- `random`: the mean of `random_seeds` draws among cells visited in train;
- `oracle_test`: greedy fitted on the **test** days. This is an in-sample ceiling, not a method.

The audience is the users with at least one test exposure. `diff_lo` / `diff_hi` are the 95% paired
user-bootstrap CI of `test_reach_share(lazy_greedy) - test_reach_share(method)`.

## Provenance

New code in this port, not a port of research code. `simengine.abm.mobility` names
`uretim/tahmin/kod/reklam_yeri.py` of the TALP prep research as the source of its EPR loop. That file was not
available to this unit, so nothing here is taken from it. The algorithms follow the KDD 2007 paper cited above.
The bounds in the tests (greedy `>= (1 - 1/e)` OPT, CELF `>= 1/2 (1 - 1/e)` OPT, `OPT <= bound`) are checked
against brute force on small random instances.

## Limitations

- **Reach is not sales.** The holdout scores audience reach on unseen days only. Nothing here was measured against
  campaign outcomes.
- **Exposure means presence.** Being in a cell (500 m in YJMob) counts as seeing the ad, whatever the visibility,
  direction or dwell time. With `unit="day"`, one sighting per day per cell is counted.
- **Cell costs are the caller's.** No price data is included.
- **Lazy greedy keeps a Python heap.** It needs far fewer gain evaluations, but the plain greedy is one vectorised
  pass per pick. See the timing below for this scale.
- **The adoption step is a mechanism sketch, not a fitted model.** `p_exposure`, `q` and the kernel are
  uncalibrated.
