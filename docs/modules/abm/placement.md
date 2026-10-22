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
  1e-9 before comparison, with one formula for single gains and arrays (Python `round(x, 9)` and `np.round(x, 9)`
  disagree on some halfway values, which made lazy and plain pick different cells on such ties).
- **Input matrix.** `M` is copied to float CSC with duplicate `(user, cell)` entries summed and stored zeros
  dropped. Without this, a duplicate entry was credited twice under the cap and a stored zero counted as a visitor.
- **`Placement.bound`** is their Theorem 4: `OPT <= F(A) + the best fractional knapsack of single-cell gains
  delta_s given A`. It holds for any placement, so `online_bound(M, top_visited(M, k), k=k)` also bounds OPT for a
  baseline's cells. With `candidates`, the bound is for the optimum over those cells only.
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
- `oracle_test`: greedy fitted on the **test** days. This is an in-sample reference, not a method. It is greedy,
  not the optimum, so it is not a strict ceiling: with `r = 3` (500 m, `k = 10`) its test reach is 0.424 against
  0.428 for the train-fitted greedy.

The audience is the users with at least one test exposure. `diff_lo` / `diff_hi` are the 95% paired
user-bootstrap CI of `test_reach_share(lazy_greedy) - test_reach_share(method)`.

## Measured (this port, 22 Oct 2026)

**Data.** YJMob100K dataset1 (Yabe et al. 2024, Sci Data; CC BY 4.0), first 4,000 users (`uid < 4000`),
4,841,963 rows, no masked rows. Train is days 0-59, test is days 60-74, `unit="day"`, `k` cells, one seed. Two
grids were measured:

- `size=1`: the raw 200 x 200 grid, 500 m cells, 23,797 cells visited in train;
- `size=4`: 50 x 50 grid, 2 km cells, 2,108 cells visited.

In every table the audience is all 4,000 users, since all are active in test. The CI is the 95% paired
user-bootstrap (1,000 draws) of greedy minus method.

Command (the data are read at runtime, nothing is shipped):

```python
import pandas as pd, simengine.abm.placement as pl
parts = []
for ch in pd.read_csv("$SIMENGINE_DATA/yjmob100k/yjmob100k-dataset1.csv.gz", chunksize=1_000_000):
    parts.append(ch[ch.uid < 4000])
    if ch.uid.iloc[-1] >= 4000: break
t = pd.concat(parts)
for size in (1, 4):
    g = pl.CellGrid(1, 1, 200 // size, 200 // size, size)
    print(pl.holdout(t, g, train_days=range(60), test_days=range(60, 75), ks=(5, 10, 25, 50, 100),
                     random_seeds=10, n_boot=1000))
```

### Test-day reach share, 500 m cells (`size=1`)

| k | lazy_greedy | top_visits | top_visitors | random (10 draws) | oracle_test (in-sample) | greedy - top_visits [95% CI] |
|---|---|---|---|---|---|---|
| 5 | 0.315 | 0.311 | 0.311 | 0.017 | 0.333 | [-0.007, +0.017] |
| 10 | 0.431 | 0.348 | 0.358 | 0.036 | 0.437 | [+0.070, +0.097] |
| 25 | 0.602 | 0.462 | 0.427 | 0.076 | 0.630 | [+0.124, +0.156] |
| 50 | 0.739 | 0.597 | 0.527 | 0.152 | 0.790 | [+0.126, +0.157] |
| 100 | 0.831 | 0.712 | 0.652 | 0.278 | 0.922 | [+0.106, +0.135] |

### Test-day reach share, 2 km cells (`size=4`)

| k | lazy_greedy | top_visits | top_visitors | random | oracle_test | greedy - top_visits [95% CI] |
|---|---|---|---|---|---|---|
| 5 | 0.557 | 0.478 | 0.478 | 0.100 | 0.557 | [+0.066, +0.091] |
| 10 | 0.681 | 0.517 | 0.546 | 0.181 | 0.695 | [+0.148, +0.179] |
| 25 | 0.850 | 0.653 | 0.641 | 0.369 | 0.870 | [+0.181, +0.212] |
| 50 | 0.923 | 0.821 | 0.723 | 0.563 | 0.962 | [+0.089, +0.113] |
| 100 | 0.958 | 0.887 | 0.824 | 0.767 | 0.998 | [+0.061, +0.081] |

What the tables show:

- **Greedy against the baselines.** Coverage greedy beats top-k-visited by 7-20 points of test reach from `k = 10`
  on, at both grid sizes. At `k = 5` on 500 m cells it ties, with a CI that includes 0. Random is far behind.
- **Cost of the temporal shift.** The in-sample oracle is 0-9 points above greedy, with the gap growing with `k`.
  Train reach overstates test reach: at 500 m and `k = 50`, train reach is 0.954 while test reach is 0.739.
- **Frequency.** Top-k cells deliver about 2x the frequency of greedy. At 500 m and `k = 50`, test frequency is
  9.04 for top_visits against 3.92 for greedy. The effective reach at 3+ is about the same: 0.390 against 0.384.
- **Capped objective (`r=3`).** At 500 m and `k = 50`, test reach is 0.759 and effective reach at 3+ is 0.421,
  against 0.739 / 0.384 for `r = 1`. At 2 km it is 0.948 / 0.818 against 0.923 / 0.776.

### Placement from simulated trajectories (`simengine.abm.mobility`)

Each simulator was fitted on train days and asked for the real test-day `(uid, d, t)` query rows. Greedy then
placed cells on the simulated test-day positions, and the cells were scored on the real test days. The simulator
therefore knows **when** each user was active in test, but not **where**. One seed.

| grid, k | greedy on real train | greedy on sim slot_mode | sim markov_sample | sim EPR | top_visits (train) | oracle_test |
|---|---|---|---|---|---|---|
| 500 m, 10 | 0.431 | 0.395 | 0.430 | 0.416 | 0.348 | 0.437 |
| 500 m, 50 | 0.739 | 0.697 | 0.751 | 0.741 | 0.597 | 0.790 |
| 2 km, 10 | 0.681 | 0.664 | 0.685 | 0.689 | 0.517 | 0.695 |
| 2 km, 50 | 0.923 | 0.923 | 0.941 | 0.933 | 0.821 | 0.962 |

Paired CIs (real-train greedy minus simulated source):

- **slot_mode** is worse in 3 of 4 cases: for example [+0.024, +0.049] at 500 m and `k = 10`.
- **EPR** ties at 500 m and `k = 50`, [-0.017, +0.011]. It is slightly better at 2 km and `k = 50`,
  [-0.020, -0.002].
- **markov_sample** is better at 2 km and `k = 50`, [-0.026, -0.010].

Every source beats the top-k baseline. A stochastic simulator (Markov sample, EPR) is about as good as the raw
history for this purpose. It may be a little better at coarse cells, where sampling smooths over the user's whole
60-day history. This is a single seed.

**Without the timing leak** (review re-run, same data and seed): the query rows were the last 15 train days
(45-59) shifted by 15 days, so the simulators saw nothing from the test days. Test reach share:

| grid, k | real train | slot_mode | markov_sample | EPR |
|---|---|---|---|---|
| 500 m, 10 | 0.431 | 0.390 | 0.429 | 0.419 |
| 500 m, 50 | 0.739 | 0.698 | 0.749 | 0.741 |
| 2 km, 10 | 0.681 | 0.664 | 0.689 | 0.680 |
| 2 km, 50 | 0.923 | 0.922 | 0.949 | 0.937 |

The picture does not change: slot_mode is worse at both `k` on 500 m cells ([+0.028, +0.053] and [+0.027,
+0.055]), and at 2 km and `k = 50` Markov sample and EPR are better than the real history ([-0.034, -0.018] and
[-0.023, -0.006]). The other CIs include 0. The leak was not what made the simulators look good.

### Speed (32-core box shared with other jobs, 3 repeats, `size=1`, 767,420 non-zeros)

| k | lazy greedy | plain greedy | evaluations (lazy / plain) |
|---|---|---|---|
| 50 | 0.30-0.40 s | 0.61-0.82 s | 63,040 / 2,000,000 |
| 200 (stops at 103 cells, full coverage) | 0.47-0.56 s | 1.26-1.33 s | 88,115 / 4,160,000 |

The lazy count includes one full pass over the 40,000 cells. The plain count is 104 passes: the 104th finds no
gain. In the first measurement the first call in a fresh process was slower: 2.4 s lazy against 1.0 s plain, at a
load average of about 70. A review re-run did not reproduce it (load average 22-55, three fresh processes,
`k = 50`): the first call took 0.32-0.36 s when lazy went first and 0.73 s when plain went first. The holdout over
5 values of `k` took 6-10 s per grid.

### Adoption illustration (uncalibrated, no ground truth)

Setup: 2 km cells, `k = 10` chosen on train, exposures counted on test days, home cells from train nights,
`p_exposure = 0.01`, `q = 0.3`, 10 steps, 50 runs.

| cells | adopters from the ad | adopters from word of mouth | total |
|---|---|---|---|
| greedy | 138 | 941 | 1,079 |
| top_visits | 202 | 1,064 | 1,266 |

Under a nearly linear exposure response, total exposures matter more than reach. The reach-optimal set is then not
the adoption-optimal one. Pick the objective (`r`, or adoption itself) to match the assumed response curve.

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
  pass per pick. See the timing above for this scale.
- **The adoption step is a mechanism sketch, not a fitted model.** `p_exposure`, `q` and the kernel are
  uncalibrated.
