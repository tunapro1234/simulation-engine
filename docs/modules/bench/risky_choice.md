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
- EvalFrame: `unit`, `group` = `item` = problem (`Problem` for choices13k, `GameID` for CPC18; cluster for the
  bootstrap), `segment` (`risk` / `amb`), `seg_block`,
  `n`, `se2`, `truth`, `pred_<method>`.
- `choices13k` options: `subset` (`"peterson"` default: Feedback and not Amb, 9,831 rows, one per Problem; `"all"`:
  14,568 rows, 13,006 Problems, 1,562 of them in both feedback conditions), `seed` (0; the split permutes unique
  `Problem` ids, so both rows of a Problem land on the same side), `test_size` (0.1), `train_frac` (1.0; subsample of the train set, for learning curves), `beast`
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
- `fosd` (used for BEAST's "trivial" problems) agrees with the organisers' `CPC15_isStochasticDom` on all 14,838
  option pairs (270 CPC tasks + 14,568 choices13k rows; 2,506 dominated, 0 mismatches; review check, 22 Oct 2026).
- Degenerate options with H == L (CPC18 game 164, a training task; 131 choices13k rows) are merged to one outcome
  here, while the organisers' `CPC18_getDist` keeps two equal rows. The only effect is on BEAST.sd's complexity rule
  (sigma_comp needs one option with more than 2 outcomes and the other with more than 1): it differs on 26 of the
  9,831 `peterson` rows (37 of 14,568), where A is degenerate and B has more than 2 outcomes; no CPC test task.
- Tests (`tests/test_bench_risky_choice.py`) use synthetic files only; one real-data test (BEAST.sd on the CPC18
  competition set within 0.001 of the published 0.00708) runs when `$SIMENGINE_DATA` is set.

## Measured numbers

Command (from the repository root; one run each, 22 Oct 2026, 5,000 BEAST agents per problem):

```bash
SIMENGINE_DATA=<catalogue> PYTHONPATH=. python -W ignore -c "
from simengine.bench.tasks import risky_choice as R
t = R.CPC18()                      # or R.CPC18(split='cpc15'), R.CPC18(subset='risk_nofb'), R.Choices13k(seed=0, ...)
print(t.score_all(t.run(), B=1000).pivot_table(index='method', columns='metric', values='value'))"
```

The CPC18 track I and choices13k seed 0 tables below were re-run by the reviewer on 22 Oct 2026 (after the
Problem-level split change) and reproduce to every printed digit.

CPC18 track I, 60 competition tasks x 5 blocks (wall 11 s, 0.6 GB):

| method | MSE | 95% CI | completeness | pref_acc | published (100 x MSE, Plonsky et al. arXiv:1904.06866) |
|---|---|---|---|---|---|
| beast_sd | 0.00703 | [0.00539, 0.00900] | 0.903 | 0.888 | BEAST.sd 0.708, P(agree) 0.883 (Table S1) |
| beast | 0.00790 | [0.00539, 0.01108] | 0.889 | 0.908 | BEAST 0.79 ("test MSE 0.0079", SI), completeness 88.9% (main text) |
| pt | 0.02334 | [0.01721, 0.03061] | 0.643 | 0.768 | |
| cpt | 0.02414 | [0.01768, 0.03179] | 0.630 | 0.752 | |
| ev_logit | 0.03137 | [0.02237, 0.04157] | 0.515 | 0.707 | |
| gbdt_objective | 0.05025 | [0.03108, 0.07060] | 0.215 | 0.652 | |
| train_mean | 0.06340 | | 0.005 | 0.668 | |
| half | 0.06373 | | 0 | 0.5 | |
| ev_max | 0.16142 | | -1.556 | 0.707 | |

For scale (literature, not run here): CPC18 winner BEAST-GB 0.569 (submitted; 0.556 in the organisers' replication),
Psychological Forest baseline 0.681 (Table S1); Bourgin et al. 2019 (ICML, Table 1) list BEAST18 0.70.

CPC15 split (`split="cpc15"`, 60 CPC15 competition tasks): beast 0.00988 [0.00619, 0.01427] (published: BEAST15
100 x MSE 0.99, CPC15 winner 0.88; Bourgin et al. 2019 Table 1), beast_sd 0.00676, pt 0.02673, cpt 0.02806, ev_logit
0.03014, gbdt_objective 0.06860, half 0.07618.

CPC18 risk without feedback (`subset="risk_nofb"`, 48 test tasks, block 1): beast 0.01008, beast_sd 0.01008, pt
0.01894, cpt 0.01955, gbdt_objective 0.03057, ev_logit 0.03227, half 0.05095. Published (CPC18 SI, same subset):
untrained BEAST 0.0100, stochastic CPT 0.0198 (fitted alpha 0.91, gamma 0.84, delta 0.83, lambda 1.14, mu 0.25;
ours: 0.92, 0.84, 0.81, 1.15, 0.24).

choices13k, seed 0, 8,848 train / 983 test problems (default options: wall 43 s, 0.28 GB):

| method | MSE | 95% CI | completeness | pref_acc |
|---|---|---|---|---|
| gbdt_objective | 0.01224 | [0.01102, 0.01354] | 0.871 | 0.824 |
| cpt | 0.01968 | [0.01807, 0.02142] | 0.695 | 0.829 |
| pt | 0.02005 | [0.01842, 0.02188] | 0.686 | 0.818 |
| beast_sd (`five`, default) | 0.02130 | [0.01950, 0.02320] | 0.656 | 0.801 |
| beast (`five`) | 0.02113 | [0.01932, 0.02312] | 0.661 | 0.814 |
| beast_sd (`cpc_block`) | 0.02259 | [0.02068, 0.02457] | 0.626 | 0.788 |
| beast (`cpc_block`) | 0.02334 | [0.02124, 0.02537] | 0.608 | 0.803 |
| ev_logit | 0.02517 | [0.02322, 0.02720] | 0.565 | 0.764 |
| half | 0.04905 | [0.04570, 0.05275] | 0 | 0.5 |
| train_mean | 0.04907 | | -0.000 | 0.509 |
| ev_max | 0.15227 | | -2.442 | 0.765 |

Seeds 1-4 (`beast="none"`): gbdt_objective 0.01235 / 0.01222 / 0.01280 / 0.01291, cpt 0.02046 / 0.02127 / 0.02098 /
0.02115, pt 0.02034 / 0.02211 / 0.02144 / 0.02151; over seeds 0-4 the mean is 0.0125 (gbdt_objective) and 0.0207
(cpt). Literature for scale (90/10 splits averaged over 50 runs, not run here): Peterson et al. Fig. 2B classic PT
models about 0.021-0.022 at full data (read off the figure); context-dependent network 0.0113 and BEAST-GB 0.00843
(Plonsky et al.); original BEAST on choices13k 0.0216 (Plonsky et al. SI) and 65.7% completeness (main text), protocol and evaluation
set not stated; Bourgin et al. 2019 Table 2
(80/20 splits): random forest 0.0141, MLP 0.0103, sparse MLP 0.0091.

## Limitations

- choices13k has 5 trials per problem with feedback after each, not the CPC 25-trial design (Plonsky et al. SI,
  Choices13k data: "five consecutive choices with full feedback after each choice"). Its `Block` column is 1 for
  no-feedback rows and, for feedback rows, "sampled uniformly at random from {2, 3, 4, 5}" (choices13k README), so
  it is a label rather than a count of earlier trials; `gbdt_objective` still sees it as a feature. `five` (default)
  simulates 5 trials with feedback from trial 1 (`Feedback` rows) or none; `cpc_block` runs the 25-trial CPC
  simulation and takes the row's `Block`, as the CENTaUR choices13k BEAST.sd script does. Which one Plonsky et al.
  used for their 0.0216 / 65.7% is not stated; `five` lands closer (BEAST 0.0211 / 0.661 vs 0.0233 / 0.608), which
  is why it is the default. Their number may also be over all 9,831 problems rather than a 983-problem test set.
- `subset="all"` before the review split rows, so 279-306 test rows per seed (seeds 0-2) had their gamble pair in
  train under the other feedback condition; the measured MSE shift from splitting on Problem was within split noise
  (gbdt_objective 0.01209 -> 0.01194, cpt 0.02258 -> 0.02173 on seed 0, different test sets), but the split now
  never shares a Problem.
- One random split per seed, not the 50-split average of the papers, so choices13k numbers carry split noise (about
  +-0.0005 across seeds 0-4 for gbdt_objective).
- PT/CPT are fitted by least squares from three starts, not by the SI grid search; the fitted parameters on the
  `risk_nofb` subset are within 0.02 of the SI values, other subsets may have flat directions (lambda and delta trade
  off when few gambles mix gains and losses).
- BEAST is stochastic: with 5,000 agents the B-rate of a cell has Monte Carlo sd below about 0.007, which moves the
  CPC MSE in the fifth decimal. `seed` fixes it.
- No BEAST-GB, Psychological Forest or neural PT here; their numbers above are quoted, not reproduced.
- choices13k carries no licence file; use it for evaluation only and do not redistribute rows.
