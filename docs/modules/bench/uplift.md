# simengine.bench.tasks.uplift

Uplift tasks: rank people by how much a treatment raises their chance to respond, using three public randomised
experiments. Each one is a registered `BenchTask` (layout, load, baselines, frame, metric), so `bench.get`, `task.run`,
`task.score` and `bench.run` work as for the other tasks.

| task | dataset | treatment | outcome (default first) | features |
|---|---|---|---|---|
| `hillstrom_uplift` | MineThatData e-mail challenge (Hillstrom 2008), 64,000 customers | e-mail (`arm="any"`: mens or womens vs none; `"mens"`, `"womens"`: that e-mail vs none) | `visit`, `conversion` (2 weeks) | recency, history, history_segment, mens, womens, zip_code, newbie, channel |
| `criteo_uplift` | Criteo Uplift v2.1 (Diemert et al. 2021), 13,979,592 users | targetable for ads (85%) | `visit`, `conversion` | f0..f11 (anonymised, dense) |
| `x5_uplift` | X5 RetailHero (ODS.ai 2019-20), 200,039 clients | SMS (50%) | `target` (purchase) | age, gender, card issue day, first redeem day, redeem delay |

## API

```python
from simengine import bench
from simengine.bench.tasks import uplift      # registers the three tasks (until the package __init__ imports it)

task = bench.get("hillstrom_uplift", arm="womens", outcome="visit")
data = task.load("/path/to/catalogue")        # or $SIMENGINE_DATA; TaskData(train, test, meta)
frame = task.run(methods={"mine": lambda d: my_scores(d.train, d.test)}, data=data)
task.score(frame, "mine")                     # {'metric': 'qini', 'value', 'lo', 'hi', 'n', 'higher_is_better'}
task.score_all(frame, B=1000)                 # every method x (qini, auuc, uplift@30)
```

- Options (all tasks): `test_frac` (default 0.2), `seed` (split seed, default 0), `baselines` (a `+`-joined subset
  of `random`, `t_logistic`, `t_gbdt`, `class_transform`, `s_gbdt`; default all). Hillstrom: `arm`, `outcome`.
  Criteo: `outcome`, `n_rows` (random subsample before the split; default all rows). Unknown options raise.
- `data.train` / `data.test` hold the features plus `t` (treatment 0/1) and `y` (the chosen outcome 0/1).
  `meta` records the split rule, counts, treated share, test-set ATE and the licence line.
- A method returns one uplift score per test row; only the order matters. The EvalFrame has `unit`, `truth`
  (outcome), `treatment`, `segment` (Hillstrom: channel, X5: gender, Criteo: `all`) and `pred_<method>`.
- Free functions: `qini_curve`, `uplift_curve`, `qini_coefficient`, `auuc`, `uplift_at_k` (all take
  `y, treatment, score, weights=None`), learners `two_model`, `s_learner`, `class_transformation`
  (`train, test, treatment=, outcome=, categorical=, numeric=, model="logistic" | "gbdt" | factory`) and
  `stratified_split(df, by, test_frac, seed)`.

## Split and metrics

- Split: stratified random by (treatment, outcome), 20% test, `default_rng(0)`. 80/20 is the protocol of Diemert et
  al. 2021 (arXiv:2111.10106, section on the experimental setup); they tune by grid search with 5-fold CV, which this
  module does not do.
- `qini` (primary): normalised Qini coefficient. Qini curve at x targeted = Y_t(x) - Y_c(x) N_t(x) / N_c(x)
  (Radcliffe 2007); score = (area(model) - area(random line)) / (area(perfect) - area(random line)), the perfect
  ordering putting treated responders first and control responders last (negative effects allowed). This is the
  definition scikit-uplift documents for `qini_auc_score(negative_effect=True)`; scikit-uplift is not installed here,
  so the code is checked against a brute-force re-implementation of the formula (tests), not against the library.
- `auuc`: same normalisation on the uplift curve (Y_t/N_t - Y_c/N_c)(x) * x, with scikit-uplift's
  `perfect_uplift_curve` ordering. Diemert et al. 2021 use a "separate, relative" AUUC without a closed formula in
  the paper, so numbers here are not comparable to theirs.
- `uplift@30`: treated minus control response rate in the top 30% by score (scikit-uplift `uplift_at_k`, overall).
- CI: 95% percentile row bootstrap, `default_rng(42)`, B = 1000. A replicate is a multinomial row-count weight
  vector on the one fixed sort (exactly a row resample; tested), so it costs O(n) instead of a re-sort.
- Ties: scores equal to each other enter the curve together (evaluated only at distinct scores).

Read the normalised numbers with care: the "perfect" curve knows each row's outcome, which no model can, so even the
true uplift function scores far below 1 (about 0.20 on the synthetic test fixture). Compare methods with each other
and with `random`, not with 1.

## Baselines

| name | method |
|---|---|
| `random` | U(0, 1) scores, `default_rng(0)` |
| `t_logistic` | two-model / T-learner: logistic (one-hot + standardised, median impute) on treated and on control, P1 - P0 |
| `t_gbdt` | T-learner with LightGBM, fixed parameters (300 trees, lr 0.05, 31 leaves, min_child_samples 200, row/col subsampling 0.8/0.9, seed 0); sklearn HistGradientBoosting if LightGBM is missing |
| `class_transform` | class variable transformation (Jaskowski & Jaroszewicz 2012): z = y t + (1-y)(1-t), uplift = 2 P(z=1\|x) - 1 with logistic; since P(z=1\|x) = e P(y\|x,1) + (1-e)(1 - P(y\|x,0)) for treated share e, the identity needs e = 1/2, so rows are weighted to make treated and control each half the weight (Hillstrom `arm="any"` 67%, Criteo 85%) |
| `s_gbdt` | S-learner: one LightGBM with the treatment flag as a feature, f(x, 1) - f(x, 0) |

No hyper-parameter search: these are reference points, not tuned competitors.

## Measured numbers

All numbers below come from one run each on the local catalogue on 2026-10-22 (test split as above, B = 1000),
command, from the repository root with the data under `$SIMENGINE_DATA`:

```python
from simengine import bench
from simengine.bench.tasks import uplift
task = bench.get("hillstrom_uplift")            # or ("hillstrom_uplift", arm="womens"), ...
frame = task.run(data_root=None)                # $SIMENGINE_DATA
print(bench.to_markdown(task.score_all(frame, B=1000).assign(status="ok", is_baseline=True), primary_only=False))
```

Cells are `value [95% CI]`.

### hillstrom_uplift, arm="any", outcome="visit" (n_test 12,799; treated share 0.667; test ATE +0.061)

| method | qini | auuc | uplift@30 |
|---|---|---|---|
| t_logistic | 0.033 [0.004, 0.065] | 0.020 [0.003, 0.038] | 0.087 [0.060, 0.112] |
| s_gbdt | 0.030 [0.001, 0.059] | 0.017 [0.001, 0.035] | 0.079 [0.053, 0.102] |
| class_transform | 0.030 [0.002, 0.059] | 0.017 [0.001, 0.034] | 0.083 [0.060, 0.106] |
| t_gbdt | 0.026 [-0.003, 0.054] | 0.015 [-0.002, 0.031] | 0.075 [0.051, 0.099] |
| random | 0.016 [-0.014, 0.045] | 0.008 [-0.009, 0.025] | 0.075 [0.050, 0.097] |

### hillstrom_uplift, arm="womens", outcome="visit" (n_test 8,539; treated share 0.501; test ATE +0.045)

| method | qini | auuc | uplift@30 |
|---|---|---|---|
| s_gbdt | 0.049 [0.013, 0.083] | 0.023 [0.006, 0.039] | 0.081 [0.052, 0.106] |
| t_gbdt | 0.039 [0.004, 0.076] | 0.018 [0.002, 0.035] | 0.064 [0.037, 0.089] |
| class_transform | 0.033 [-0.003, 0.067] | 0.015 [-0.001, 0.030] | 0.064 [0.039, 0.089] |
| t_logistic | 0.031 [-0.006, 0.066] | 0.014 [-0.002, 0.031] | 0.054 [0.026, 0.084] |
| random | 0.024 [-0.007, 0.058] | 0.011 [-0.003, 0.027] | 0.050 [0.025, 0.077] |

### hillstrom_uplift, arm="mens", outcome="visit" (n_test 8,523; treated share 0.500; test ATE +0.077)

| method | qini | auuc | uplift@30 |
|---|---|---|---|
| s_gbdt | -0.001 [-0.034, 0.033] | -0.000 [-0.018, 0.017] | 0.081 [0.050, 0.111] |
| random | -0.010 [-0.043, 0.025] | -0.005 [-0.022, 0.012] | 0.080 [0.051, 0.106] |
| t_gbdt | -0.013 [-0.047, 0.025] | -0.007 [-0.025, 0.012] | 0.069 [0.040, 0.096] |
| class_transform | -0.014 [-0.046, 0.018] | -0.007 [-0.023, 0.009] | 0.067 [0.036, 0.096] |
| t_logistic | -0.020 [-0.054, 0.013] | -0.010 [-0.028, 0.007] | 0.056 [0.028, 0.087] |

### hillstrom_uplift, arm="any", outcome="conversion" (n_test 12,800; test ATE +0.005)

| method | qini | auuc | uplift@30 |
|---|---|---|---|
| class_transform | 0.153 [0.037, 0.276] | 0.007 [0.002, 0.013] | 0.010 [0.006, 0.016] |
| t_gbdt | 0.057 [-0.063, 0.161] | 0.003 [-0.003, 0.008] | 0.007 [0.001, 0.012] |
| t_logistic | 0.050 [-0.092, 0.188] | 0.002 [-0.004, 0.009] | 0.007 [-0.001, 0.014] |
| s_gbdt | 0.019 [-0.099, 0.141] | 0.001 [-0.005, 0.007] | 0.005 [-0.001, 0.011] |
| random | -0.009 [-0.116, 0.097] | -0.000 [-0.005, 0.005] | 0.006 [0.001, 0.011] |

### x5_uplift (n_test 40,008; treated share 0.500; test ATE +0.033)

| method | qini | auuc | uplift@30 |
|---|---|---|---|
| t_logistic | 0.012 [0.006, 0.019] | 0.019 [0.009, 0.029] | 0.065 [0.049, 0.081] |
| class_transform | 0.012 [0.006, 0.018] | 0.019 [0.009, 0.028] | 0.066 [0.050, 0.082] |
| s_gbdt | 0.006 [-0.001, 0.012] | 0.008 [-0.002, 0.018] | 0.045 [0.028, 0.062] |
| t_gbdt | 0.006 [-0.001, 0.012] | 0.008 [-0.003, 0.018] | 0.050 [0.032, 0.067] |
| random | 0.000 [-0.006, 0.007] | 0.001 [-0.009, 0.010] | 0.034 [0.016, 0.051] |

<!-- CRITEO -->

What the numbers say: on Hillstrom and X5 every baseline sits within a few hundredths of `random` and most CIs
overlap it; the mens e-mail arm shows no detectable heterogeneity at all. The intervals are wide at these test sizes
(Diemert et al. 2021 report that AUUC differences between their baselines become separable only from about 1M test
rows on Criteo). Treat a new method's win on Hillstrom or X5 as real only when its CI clears the best baseline.

## Provenance

- New code, not ported from the research branches. Metric definitions follow Radcliffe 2007 (Qini) and the
  scikit-uplift metric documentation (`qini_auc_score`, `uplift_auc_score`, `uplift_at_k`); class transformation
  follows Jaskowski & Jaroszewicz 2012 (ICML workshop on clinical data); T- / S-learner naming follows Künzel, Sekhon, Bickel
  & Yu 2019 (arXiv:1706.03461, PNAS). The Criteo paper (Diemert et al. 2021, arXiv:2111.10106) was read for the split protocol and its
  baseline list (two-model, class variable transformation, modified outcome, shared representation; logistic / ridge).
- Data licences (catalogue lines): Hillstrom, no licence text at the source (public 2008 challenge data); Criteo
  v2.1, **CC BY-NC-SA 4.0** (non-commercial); X5 RetailHero, no licence stated (competition data, research /
  evaluation only, do not redistribute). The repository ships none of the data.

## Limitations

- X5 uses client profile features only; the purchase history (`purchases.csv.gz`, 639 MB) is not read, and winning
  X5 solutions are built on it, so these X5 numbers understate what is reachable.
- Criteo is subsampled non-uniformly by the publisher for privacy, so its ATE and uplift levels are not real ad
  effects (dataset README).
- `kind = "binary"` describes the outcome; predictions are scores, so `simengine.evidence.evaluate`'s binary metrics
  (AUC on truth, Brier) are not meaningful on this frame. Use `task.score`.
- No hyper-parameter search, no cross-fitting, a single random split; the Hillstrom `spend` outcome is not offered.
- Normalised AUUC / Qini depend on the perfect-curve convention; compare only with numbers computed the same way.
- The tasks are registered when `simengine.bench.tasks.uplift` is imported; the package `__init__` does not import
  it yet.
