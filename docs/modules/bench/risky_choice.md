# simengine.bench.tasks.risky_choice

Predict how often people choose Option B over Option A in a choice between two gambles. The unit is an aggregate
choice rate: one problem (choices13k) or one problem x block of five trials (CPC18). Two registered `BenchTask`s:

| task | data | units (default split) | truth |
|---|---|---|---|
| `choices13k` | Peterson et al. 2021 (Science 372:1209), MTurk, 5 trials per problem | 9,831 problems with feedback and no ambiguity; random 90/10 split, 983 test problems | `bRate` (B-rate over subjects and 5 trials) |
| `cpc18` | Choice Prediction Competition 2018 (Plonsky et al., arXiv:1904.06866) incl. CPC15 (Erev et al. 2017) | 270 tasks x 5 blocks; train = sets 1-7 (210 tasks), test = sets 8-9 (the 60 competition tasks, 300 cells) | mean over subjects of the per-subject B-rate in the block |

## API

```python
from simengine import bench
from simengine.bench.tasks import risky_choice as R   # registers both tasks (until the package __init__ imports it)

task = bench.get("cpc18")                    # or R.CPC18(split="cpc15"), R.Choices13k(seed=0)
data = task.load("/path/to/catalogue")       # or $SIMENGINE_DATA; TaskData(train, test, meta)
frame = task.run(methods={"mine": lambda d: my_rates(d.train, d.test)}, data=data)
task.score(frame, "mine")                    # {'metric': 'mse', 'value', 'lo', 'hi', 'n', 'higher_is_better'}
task.score_all(frame, B=1000)                # every method x (mse, completeness, pref_acc)
task.fitted_                                 # parameters of the fitted baselines (ev_logit, pt, cpt)
```

- A method returns one B-rate in [0, 1] per test row. `data.test` / `data.train` carry the CPC problem columns
  (`Ha, pHa, La, Hb, pHb, Lb, LotShapeB, LotNumB, Amb, Corr`, CPC18 also `LotShapeA, LotNumA, Set, GameID`),
  `block`, `y` (observed B-rate), `n` (subjects), `se2` (squared standard error of `y`), and the full outcome
  distributions `dist_a`, `dist_b` as `(outcomes ascending, probabilities)`.
- EvalFrame: `unit`, `group` = `item` = problem (cluster for the bootstrap), `segment` (`risk` / `amb`), `seg_block`,
  `n`, `se2`, `truth`, `pred_<method>`.
- `choices13k` options: `subset` (`"peterson"` default: Feedback and not Amb, 9,831 rows; `"all"`: 14,568 rows),
  `seed` (0), `test_size` (0.1), `train_frac` (1.0; subsample of the train set, for learning curves), `beast`
  (`"sd"` default, `"original"`, `"both"`, `"none"`), `beast_sims` (5000), `beast_protocol` (`"five"` default,
  `"cpc_block"`; see Limitations).
- `cpc18` options: `split` (`"cpc18"` default, `"cpc15"`: sets 1-3 = 90 calibration tasks train, sets 4-5 = the 60
  CPC15 competition tasks test), `subset` (`"all"` default; `"risk_nofb"`: tasks without ambiguity, block 1 only,
  182 train / 48 test tasks, the CPC18 SI "foresight comparison" subset), `beast` (`"both"` default), `beast_sims`.
- Free functions: `cpc_distribution(H, pH, L, shape, num)` (organisers' `CPC18_getDist`), `merge_dist`, `pad_dists`,
  `expected_value`, `gw_weight`, `prospect_value(X, P, alpha, lam, gamma, delta, cumulative=)`, `fit_prospect`,
  `predict_prospect`, `logit_b`, `fit_ev_logit`, `p_better_comonotone`, `fosd`, `completeness(y, p, se2)`,
  `beast_predict(dist_a, dist_b, amb, corr, version="sd" | "original", n_sims=5000, ...)` returning `(P, 5)` B-rates.

## Metrics

- `mse` (primary): mean squared error of the B-rate over test rows, as in Peterson et al. (MSE on ~1000 unseen
  problems) and the CPC (mean over 60 tasks x 5 blocks; the competitions print 100 x MSE). CI: 95% cluster bootstrap
  over problems, B = 1000, `default_rng(42)`.
- `completeness`: (MSE_random - MSE) / (MSE_random - MSE_irreducible), MSE_random from always predicting 0.5 and
  MSE_irreducible = mean of `se2` (Fudenberg et al.; the estimator described in Plonsky et al. Methods). 0 = no better
  than 0.5, 1 = as good as the sampling noise allows.
- `pref_acc`: share of rows where prediction and data fall on the same side of 0.5 (ties count half); Peterson et al.
  call it decision preference accuracy, Plonsky et al. P(agree).

## Baselines

| name | what | fitted on train |
|---|---|---|
| `half` | always 0.5 | - |
| `train_mean` | mean train B-rate | mean |
| `ev_max` | 1 if EV(B) > EV(A), 0 if lower, 0.5 if equal | - |
| `ev_logit` | logistic(mu (EV_B - EV_A)) | mu, least squares |
| `pt` | separable prospect theory: sum w(p_i) u(x_i) | alpha, lambda, gamma, delta, mu by least squares (L-BFGS-B, 3 starts) |
| `cpt` | cumulative prospect theory, rank-dependent weights | same 5 parameters |
| `gbdt_objective` | `HistGradientBoostingRegressor` on the objective task columns + block | yes |
| `beast` | BEAST, CPC15 version (Erev et al. 2017), published parameters | no |
| `beast_sd` | BEAST.sd, the CPC18 baseline (subjective dominance, complexity noise, ambiguity learning), published parameters | no |

PT/CPT follow the specification in the CPC18 SI (Plonsky et al., "Cumulative prospect theory" section): u(x) = x^a for
gains, -lambda (-x)^a for losses (one curvature), Goldstein-Einhorn weighting w(p) = delta p^gamma / (delta p^gamma +
(1-p)^gamma), the same w for gains and losses, logit choice with sensitivity mu. The SI fits by grid search; this
module uses least squares with L-BFGS-B.

BEAST is a numpy re-implementation of the organisers' per-agent Python code (CPC15_BEASTsimulation, CPC18_BEASTsd
simulation), vectorised over problems and simulated agents: 25 trials, feedback from trial 6, five blocks, agent traits
drawn uniformly below the published maxima (BEAST: sigma 7, kappa 3, beta 2.6, gamma 0.5, psi 0.07, theta 1; BEAST.sd:
sigma 13, sigma_comp 35, kappa 3, beta 1.4, gamma 1, psi 0.25, theta 0.7, w_amb 0.25). Differences from the
organisers' code, all equal in distribution: P(A better) / P(B better) for subjective dominance is computed exactly for
comonotone draws (the organisers estimate it with 100,000 draws); random numbers are drawn in batches, so results do
not match theirs draw for draw; each agent draws `kappa` (max) mental samples and averages its own first kappa_i.

## Provenance and checks

- Data layout (read at run time, nothing is shipped): `<root>/choices13k/ham/c13k_selections.csv` and
  `c13k_problems.json` (github jcpeterson/choices13k; no licence file in that repository), `<root>/cpc18/ham/
  cpc18_all_raw.csv` (trial-level CPC15 + CPC18, Zenodo 10.5281/zenodo.2571510, CC BY 4.0).
- `cpc_distribution` reproduces the organisers' `CPC18_getDist` on all 540 CPC options, and on Option B of all 14,568
  choices13k rows it reproduces the outcome lists in `c13k_problems.json` (0 mismatches; check run against the
  organisers' code, not part of the test suite because that code is not in the repository).
- BEAST and BEAST.sd vs the organisers' simulation on 14 CPC tasks (ambiguous, correlated, multi-outcome and plain):
  20,000 vectorised agents vs 4,000 reference agents per task, max absolute difference over the 5 blocks 0.000-0.011
  for BEAST.sd and 0.000-0.012 for BEAST (median about 0.006); with 1,000 reference agents the differences were up to
  0.022, so they shrink with the reference sample size, as Monte Carlo noise does.
- Tests (`tests/test_bench_risky_choice.py`) use synthetic files only; one real-data test (BEAST.sd on the CPC18
  competition set within 0.001 of the published 0.00708) runs when `$SIMENGINE_DATA` is set.

## Limitations

- choices13k has 5 trials per problem with feedback after each, not the CPC 25-trial design. `five` (default)
  simulates 5 trials with feedback from trial 1 (`Feedback` rows) or none; `cpc_block` runs the 25-trial CPC
  simulation and takes the row's `Block`, as the CENTaUR choices13k BEAST.sd script does. Which one Plonsky et al.
  used for their 0.0216 / 65.7% is not stated; `five` lands closer (BEAST 0.0211 / 0.661 vs 0.0233 / 0.608), which
  is why it is the default. Their number may also be over all 9,831 problems rather than a 983-problem test set.
- One random split per seed, not the 50-split average of the papers, so choices13k numbers carry split noise (about
  +-0.0005 across seeds 0-4 for gbdt_objective).
- PT/CPT are fitted by least squares from three starts, not by the SI grid search; the fitted parameters on the
  `risk_nofb` subset are within 0.02 of the SI values, other subsets may have flat directions (lambda and delta trade
  off when few gambles mix gains and losses).
- BEAST is stochastic: with 5,000 agents the B-rate of a cell has Monte Carlo sd below about 0.007, which moves the
  CPC MSE in the fifth decimal. `seed` fixes it.
- No BEAST-GB, Psychological Forest or neural PT here; their numbers above are quoted, not reproduced.
- choices13k carries no licence file; use it for evaluation only and do not redistribute rows.
