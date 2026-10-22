# `simengine.research.hyperband`: Hyperband, ASHA and a multi-fidelity runner protocol

## What

Multi-fidelity configuration search. Each config is scored cheaply at a small budget first, and only the promising
ones get the full budget. The budget is an integer count of resource units: rows of a sample, personas, questions or
LLM calls (one unit = one call). Three pieces:

- **Runner protocol.** `FidelityRunner` is any `(cfg, budget) -> score` object with `max_budget`, `min_budget` and
  `unit`. `MultiFidelity(fn, max_budget)` wraps a plain function, caches repeated (config, budget) pairs and counts
  what it charged (`spent`, `calls`, `log`). `UnitRunner(score_units, n_units)` builds a runner from a per-unit scorer
  `score_units(cfg, idx) -> scores`. Budget b scores the first b units of one seeded permutation that every config
  shares, so comparisons use common random numbers. With `resume=True` a promoted config only pays for its new units.
  This is the "resume from checkpoint" case of both papers.
- **Hyperband** (`hyperband`, `hyperband_brackets`). Runs several brackets of the existing
  `proposers.successive_halving`, each with a different trade-off between n configs and budget per config. The
  halving step is not duplicated: the module adds `rounding="floor"` to `successive_halving` (the default `"ceil"` is
  unchanged) and calls it once per bracket.
- **ASHA** (`ASHA` scheduler, `asha` driver). Asynchronous successive halving: a config moves up a rung as soon as it
  is in the top 1/eta of the completed results of its rung, so a free worker never waits for a rung to fill. The
  scheduler is driven by `get_job()` and `report(job, score)` from any executor. `asha(n_workers=k)` runs jobs in a
  thread pool, which suits I/O-bound runners such as LLM calls.

## API

```python
from simengine.research import Space
from simengine.research.hyperband import (ASHA, FidelityRunner, Job, MultiFidelity, UnitRunner, asha, hyperband,
                                          hyperband_brackets, pool_sampler, space_sampler)

run = UnitRunner(score_units, n_units=81, unit="llm_calls")          # score_units(cfg, idx) -> one score per unit
hb = hyperband(run, Space({...}), higher_is_better=False, eta=3)     # or a list of configs, or (rng, n) -> configs
hb["winner"], hb["score"], hb["bracket"], hb["budget"], run.spent    # nominal budget vs units actually charged
out = asha(UnitRunner(score_units, 81), space, higher_is_better=False, total_budget=hb["budget"], n_workers=1)
```

| name | what it does |
|---|---|
| `hyperband_brackets(max_budget, min_budget=1, eta=3)` | Algorithm 1 plan: `[{"s", "n", "counts", "budgets", "cost"}]` from s_max down to 0. |
| `hyperband(runner, sample, *, higher_is_better, eta, max_budget, min_budget, seed, brackets, repeats)` | Runs the brackets. `brackets=[s_max]` gives plain halving with Hyperband's n. `repeats` reruns the outer loop with fresh configs. `runner` is any `FidelityRunner`; `max_budget=` / `min_budget=` may narrow its budget range but not widen it (a ValueError, since the runner would clamp the budget and the nominal count would be wrong). `runner` can also be a plain screen `(configs, fidelity) -> scores` with `max_budget=` (fidelity = budget / max_budget, the same contract as `ProposalContext.screen`). |
| `ASHA(max_budget, min_budget, eta, s, *, higher_is_better)` | Scheduler: `budgets` (per rung), `get_job(new_config)`, `report(job, score)`, `best()`, `issued` (nominal budget of the jobs handed out). |
| `asha(runner, sample, *, higher_is_better, eta, s, total_budget, max_trials, n_workers, seed)` | Runs until the issued budget reaches `total_budget`, `max_trials` configs have started, or a finite pool is used up and nothing can be promoted. |
| `MultiFidelity`, `UnitRunner`, `FidelityRunner` | Runner protocol (above). `UnitRunner.per_unit(cfg)` returns the per-unit scores so far, which can be paired with `research.stats.paired_bootstrap_diff`. |
| `space_sampler(space)`, `pool_sampler(pool)` | i.i.d. uniform configs from a `Space`; a finite list drawn without replacement over the whole search. |

NaN scores rank last: halving drops them, and ASHA never promotes them. In threaded `asha`, a runner exception
becomes NaN. In sequential mode it propagates.

## Provenance

Checked against the arXiv PDFs before writing (text extracted with `pdftotext`):

- Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar. *Hyperband: A Novel Bandit-Based Approach to Hyperparameter
  Optimization.* JMLR 18 (2018) 1-52, arXiv:1603.06560v4, **Algorithm 1**. Inputs R and eta (default 3).
  s_max = floor(log_eta R), B = (s_max + 1) R. For s = s_max..0: n = ceil(B/R * eta^s/(s+1)), r = R eta^-s. Then for
  i = 0..s: n_i = floor(n eta^-i), r_i = r eta^i, T = top_k(T, L, floor(n_i/eta)). Here R is expressed as
  `max_budget / min_budget` and rung budgets are `max_budget * eta^(i-s)`, rounded to ints.
- Li, Jamieson, Rostamizadeh, Gonina, Ben-Tzur, Hardt, Recht, Talwalkar. *A System for Massively Parallel
  Hyperparameter Tuning.* MLSys 2020, arXiv:1810.05934v5, **Algorithm 2**. Inputs r, R, eta and s. A job at rung k runs
  with r eta^(s+k). `get_job` scans k = floor(log_eta(R/r)) - s - 1, ..., 0 for
  `top_k(rung k, |rung k|/eta)` minus the configs already promoted, and promotes the first one. Otherwise it draws a
  random config into rung 0.

**Discrepancy in the paper.** Table 1 of the JMLR/arXiv v4 PDF (R = 81, eta = 3) lists n = 81, 27, 9, 6, 5 for
s = 4..0. Algorithm 1's formula gives n = 81, 34, 15, 8, 5. This module follows the algorithm. The test
`test_algorithm1_plan_r81` pins 81/34/15/8/5, the rung counts (for example 34, 11, 3, 1) and the bracket costs
405/363/351/378/405, which add up to 1902.

**Choices not fixed by the papers:**
- ASHA rung budgets are anchored at max_budget (`max_budget * eta^(k-K)`), so the top rung is always a full run.
  This equals r eta^(s+k) when R/r is a power of eta.
- `top_k` uses floor(|rung|/eta), counting completed results including NaNs. Ties are broken by trial order.
- Hyperband compares bracket winners on their max_budget scores. This is a deliberate deviation: Algorithm 1
  (line 11) returns "the configuration with the smallest intermediate loss seen so far", and that minimum also runs
  over the low-budget rungs. A small-sample (or early-iteration) score is noisy, so the minimum over hundreds of them
  can be a lucky low-budget draw (not measured here). Only full-budget scores are compared here.
- `asha` stops on a nominal budget. The last jobs can overshoot it by up to `n_workers * max_budget`.

## Measured numbers

Synthetic benchmark, offline, run once on 2026-10-22 (55 s). Command:

```bash
PYTHONPATH=. nice -n 19 python examples/hyperband_vs_halving.py --seeds 200 --out <scratch>/hyperband_vs_halving.json
```

Setup: 3 continuous params and 1 categorical (3 levels); true loss mu = sum (x_i - 0.7)^2 + level offset, with
optimum 0. R = 81, eta = 3. **Regret** is the true mu of the returned config (lower is better). Each column gives a
mean over 200 seeds, with a percentile bootstrap CI (B = 1000). The last column is the paired difference from
Hyperband (negative means better than Hyperband). Methods:

- `random`: 23 configs at the full budget.
- `halving_1/9`: `successive_halving` the way `HalvingProposer` uses it, with fidelities (1/9, 1/3, 1), ceil
  rounding and 70 configs.
- `halving_s4x5`: Hyperband's most aggressive bracket repeated 5 times.
- `hyperband`: one full Hyperband run.
- `asha`: sequential ASHA with s = 0 and total_budget 1902.

The "units charged" column is what a resuming `UnitRunner` actually paid.

**Sample fidelity** (budget = units or LLM calls; low budgets are unbiased but noisy, with per-unit sd about 0.7):

| method | nominal budget | units charged | mean regret [95% CI] | minus Hyperband [95% CI] |
|---|---|---|---|---|
| random | 1863 | 1863 | 0.1046 [0.0955, 0.1137] | +0.0328 [+0.0219, +0.0438] |
| halving_1/9 | 1926 | 1494 | 0.0723 [0.0660, 0.0789] | +0.0005 [-0.0080, +0.0091] |
| halving_s4x5 | 2025 | 1485 | 0.0576 [0.0523, 0.0635] | -0.0142 [-0.0216, -0.0067] |
| hyperband | 1902 | 1581 | 0.0718 [0.0652, 0.0786] | +0.0000 |
| asha | 1918 | 1378 | 0.0553 [0.0500, 0.0608] | -0.0165 [-0.0242, -0.0092] |

**Curve fidelity** (budget = iterations; low budgets mislead because configs near the optimum learn slowly: a
"late bloomer" term a/sqrt(b/R) - a):

| method | nominal budget | units charged | mean regret [95% CI] | minus Hyperband [95% CI] |
|---|---|---|---|---|
| random | 1863 | 1863 | 0.0895 [0.0816, 0.0974] | -0.0171 [-0.0279, -0.0051] |
| halving_1/9 | 1926 | 1926 | 0.1137 [0.1073, 0.1211] | +0.0072 [-0.0025, +0.0176] |
| halving_s4x5 | 2025 | 2025 | 0.2339 [0.2279, 0.2401] | +0.1274 [+0.1182, +0.1374] |
| hyperband | 1902 | 1902 | 0.1065 [0.0984, 0.1146] | +0.0000 |
| asha | 1917 | 1917 | 0.2266 [0.2194, 0.2337] | +0.1201 [+0.1099, +0.1306] |

What this shows, on this synthetic problem only:

- When the low-budget score is an unbiased mean of units, as with sample size or LLM calls, aggressive early
  stopping wins. ASHA (s = 0) and the repeated s = 4 bracket beat Hyperband by about 0.015 regret. ASHA also
  charged the fewest units. Hyperband matches the existing 1/9-ladder halving and beats random search.
- When low budgets mislead, aggressive halving and ASHA (s = 0) are about twice as bad as Hyperband. Hyperband
  hedges between the two cases, but random search at full budget still beats it by 0.017 here. That is the
  "protecting itself" trade-off the JMLR paper describes, not a free win.
- For simengine's usual budgets (sample size, LLM calls) the first case is the relevant one. Prefer `asha` or a
  single aggressive bracket there. Use `hyperband` when the budget is iterations or rounds (debate turns, ABM steps,
  training epochs), where early scores can rank configs wrongly.

Literature (not measured here): the JMLR paper reports Hyperband speed-ups over random search and Bayesian
optimisation on its benchmarks, and the MLSys paper reports ASHA scaling linearly with the number of workers. See
those papers for their numbers.

## Limitations

- Configs are sampled i.i.d. (uniform on a `Space`), as in both papers. There is no model-based sampler (BOHB style).
- Infinite-horizon variants are not implemented. Asynchronous Hyperband, which loops through ASHA brackets, is not
  either; call `asha(s=...)` per bracket yourself.
- Not wired into `Loop` or `make_proposer`. `HalvingProposer` stays the Loop's multi-fidelity proposer. `hyperband`
  accepts the same `(configs, fidelity)` screen contract, so a proposer can be added later.
- Every screened config is another look at the dev split. Multi-fidelity search makes many comparisons, so confirm
  the winner on a held-out split (`Loop(select_split=...)`) before reporting it.
- Threaded `asha` is not reproducible: completion order changes promotions. Use `n_workers=1` for fixed results.
- The benchmark numbers come from one synthetic objective with chosen noise levels. They show the direction of each
  trade-off, not effect sizes for real tasks. No real dataset was run, and this unit has no bench task.

## Review (2026-10-22)

- Algorithm 1, Table 1 and ASHA's Algorithm 2 were rechecked against the `pdftotext` output of the two arXiv PDFs.
  The Table 1 discrepancy is real: the table prints n = 81, 27, 9, 6, 5 (and rung counts 27/9/3/1, 9/3/1, 6/2), while
  line 2's formula ceil(B/R * eta^s/(s+1)) with B = 5R gives 81, 34, 15, 8, 5. ASHA's `get_job` (scan
  k = floor(log_eta(R/r)) - s - 1 .. 0, top |rung k|/eta, first not-yet-promoted, else a new config at rung 0) and the
  rung budget r eta^(s+k) match the code.
- Corrected: the earlier text said the paper's "smallest intermediate loss" rule picks from the same set as comparing
  full-budget scores. It does not, because the paper's rule includes low-budget losses. The code now documents this as
  a deviation.
- Fixed: `hyperband` treated a `FidelityRunner` that is not a `MultiFidelity` (an object with `max_budget` but no
  `screen`) as a `(configs, fidelity)` screen, and refused it. `max_budget=` above the runner's own value (or
  `min_budget=` below it) was clamped by the runner without any error, so `budget` overstated what had been run.
  `hyperband` and `asha` now refuse that range, and protocol objects are wrapped.
- Re-ran the benchmark command above (200 seeds, 62 s under `nice`). Every number in both tables came out identical.
