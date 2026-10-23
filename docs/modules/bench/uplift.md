# simengine.bench.tasks.uplift

Uplift tasks: rank people by how much a treatment raises their chance to respond, using three public randomised
experiments. Each one is a registered `BenchTask` (layout, load, baselines, frame, metric), so `bench.get`, `task.run`,
`task.score` and `bench.run` work as for the other tasks.

| task | dataset | treatment | outcome (default first) | features |
|---|---|---|---|---|
| `hillstrom_uplift` | MineThatData e-mail challenge (Hillstrom 2008), 64,000 customers | e-mail (`arm="any"`: mens or womens vs none; `"mens"`, `"womens"`: that e-mail vs none) | `visit`, `conversion` (2 weeks) | recency, history, history_segment, mens, womens, zip_code, newbie, channel |
| `criteo_uplift` | Criteo Uplift v2.1 (Diemert et al. 2021), 13,979,592 users | targetable for ads (85%) | `visit`, `conversion` | f0..f11 (anonymised, dense) |
| `x5_uplift` | X5 RetailHero (ODS.ai 2019-20), 200,039 clients | SMS (50%) | `target` (purchase) | age, gender, card issue day, first redeem day, redeem delay (a first redemption after 2019-03-18 is masked: post-treatment, see Limitations) |

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
- `uplift@30`: treated minus control response rate in the first int(0.3 n) rows by score (scikit-uplift
  `uplift_at_k`, strategy "overall"; ties broken by sort order).
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

Hillstrom and X5 numbers below were re-run on 2026-10-23 after the X5 post-treatment fix and the uplift@k row count
change (qini / auuc on Hillstrom unchanged to the digit, uplift@30 moved by at most 0.001; X5 GBDT rows dropped, see
below). Criteo numbers are from 2026-10-22 runs (uplift@30 there counted ceil(0.3 n) rows, at most one row more than
now, invisible at three decimals), except the 3M-row conversion table, re-run on 2026-10-23 after the 758b1f8
review. Test split as above, B = 1000 unless stated; command, from the repository root
with the data under `$SIMENGINE_DATA`:

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
| t_logistic | 0.033 [0.004, 0.065] | 0.020 [0.003, 0.038] | 0.087 [0.060, 0.111] |
| s_gbdt | 0.030 [0.001, 0.059] | 0.017 [0.001, 0.035] | 0.079 [0.053, 0.102] |
| class_transform | 0.030 [0.002, 0.059] | 0.017 [0.001, 0.034] | 0.082 [0.060, 0.105] |
| t_gbdt | 0.026 [-0.003, 0.054] | 0.015 [-0.002, 0.031] | 0.075 [0.051, 0.098] |
| random | 0.016 [-0.014, 0.045] | 0.008 [-0.009, 0.025] | 0.075 [0.050, 0.097] |

### hillstrom_uplift, arm="womens", outcome="visit" (n_test 8,539; treated share 0.501; test ATE +0.045)

| method | qini | auuc | uplift@30 |
|---|---|---|---|
| s_gbdt | 0.049 [0.013, 0.083] | 0.023 [0.006, 0.039] | 0.081 [0.052, 0.107] |
| t_gbdt | 0.039 [0.004, 0.076] | 0.018 [0.002, 0.035] | 0.063 [0.037, 0.090] |
| class_transform | 0.033 [-0.003, 0.067] | 0.015 [-0.001, 0.030] | 0.064 [0.040, 0.089] |
| t_logistic | 0.031 [-0.006, 0.066] | 0.014 [-0.002, 0.031] | 0.054 [0.026, 0.083] |
| random | 0.024 [-0.007, 0.058] | 0.011 [-0.003, 0.027] | 0.051 [0.025, 0.077] |

### hillstrom_uplift, arm="mens", outcome="visit" (n_test 8,523; treated share 0.500; test ATE +0.077)

| method | qini | auuc | uplift@30 |
|---|---|---|---|
| s_gbdt | -0.001 [-0.034, 0.033] | -0.000 [-0.018, 0.017] | 0.080 [0.051, 0.111] |
| random | -0.010 [-0.043, 0.025] | -0.005 [-0.022, 0.012] | 0.080 [0.051, 0.107] |
| t_gbdt | -0.013 [-0.047, 0.025] | -0.007 [-0.025, 0.012] | 0.069 [0.040, 0.096] |
| class_transform | -0.014 [-0.046, 0.018] | -0.007 [-0.023, 0.009] | 0.067 [0.036, 0.097] |
| t_logistic | -0.020 [-0.054, 0.013] | -0.010 [-0.028, 0.007] | 0.056 [0.028, 0.088] |

### hillstrom_uplift, arm="any", outcome="conversion" (n_test 12,800; test ATE +0.005)

| method | qini | auuc | uplift@30 |
|---|---|---|---|
| class_transform | 0.153 [0.037, 0.276] | 0.007 [0.002, 0.013] | 0.010 [0.006, 0.016] |
| t_gbdt | 0.057 [-0.063, 0.161] | 0.003 [-0.003, 0.008] | 0.007 [0.001, 0.012] |
| t_logistic | 0.050 [-0.092, 0.188] | 0.002 [-0.004, 0.009] | 0.007 [-0.001, 0.014] |
| s_gbdt | 0.019 [-0.099, 0.141] | 0.001 [-0.005, 0.007] | 0.005 [-0.001, 0.011] |
| random | -0.009 [-0.116, 0.097] | -0.000 [-0.005, 0.005] | 0.006 [0.001, 0.011] |

### x5_uplift (n_test 40,008; treated share 0.500; test ATE +0.033)

Before the redeem-date mask (2026-10-22, same split) t_gbdt scored qini 0.006 and s_gbdt 0.006; the qini of the two
logistic learners was the same. The leaked post-campaign dates helped the trees a little, not the linear models.

| method | qini | auuc | uplift@30 |
|---|---|---|---|
| t_logistic | 0.012 [0.006, 0.019] | 0.019 [0.009, 0.029] | 0.064 [0.048, 0.081] |
| class_transform | 0.012 [0.005, 0.018] | 0.019 [0.008, 0.029] | 0.067 [0.049, 0.083] |
| s_gbdt | 0.005 [-0.001, 0.011] | 0.008 [-0.002, 0.018] | 0.050 [0.032, 0.067] |
| t_gbdt | 0.003 [-0.004, 0.009] | 0.004 [-0.006, 0.014] | 0.040 [0.024, 0.058] |
| random | 0.000 [-0.006, 0.007] | 0.001 [-0.009, 0.010] | 0.034 [0.016, 0.051] |

### criteo_uplift, outcome="visit", all rows (n_test 2,795,918; treated share 0.850; test ATE +0.0103)

`bench.get("criteo_uplift")`, B = 200 (not 1000: scoring the 2.8M-row test set with B = 1000 would take about
an hour). Wall time 16 min (fit 285 s, score 666 s), peak RSS 7.3 GB, `nice -n 19`.

| method | qini | auuc | uplift@30 |
|---|---|---|---|
| s_gbdt | 0.088 [0.082, 0.094] | 0.034 [0.032, 0.037] | 0.029 [0.027, 0.031] |
| t_gbdt | 0.087 [0.081, 0.094] | 0.034 [0.032, 0.037] | 0.029 [0.028, 0.031] |
| t_logistic | 0.085 [0.079, 0.092] | 0.033 [0.031, 0.036] | 0.028 [0.027, 0.030] |
| class_transform | 0.076 [0.070, 0.083] | 0.030 [0.028, 0.032] | 0.028 [0.026, 0.029] |
| random | -0.000 [-0.005, 0.004] | -0.000 [-0.002, 0.002] | 0.010 [0.009, 0.011] |

### criteo_uplift, outcome="visit", n_rows=3,000,000 (n_test 600,000; test ATE +0.0104)

`bench.get("criteo_uplift", n_rows=3_000_000)`, B = 1000; wall time about 10 min.

| method | qini | auuc | uplift@30 |
|---|---|---|---|
| s_gbdt | 0.093 [0.079, 0.108] | 0.036 [0.031, 0.042] | 0.030 [0.025, 0.034] |
| t_logistic | 0.083 [0.069, 0.097] | 0.032 [0.027, 0.038] | 0.029 [0.025, 0.033] |
| class_transform | 0.079 [0.066, 0.095] | 0.031 [0.026, 0.037] | 0.028 [0.024, 0.033] |
| t_gbdt | 0.077 [0.062, 0.092] | 0.030 [0.024, 0.036] | 0.028 [0.024, 0.032] |
| random | -0.005 [-0.015, 0.005] | -0.002 [-0.006, 0.002] | 0.010 [0.007, 0.012] |

### criteo_uplift, outcome="conversion", n_rows=3,000,000 (n_test 600,000; test ATE +0.0011) -- post-review (758b1f8)

`bench.get("criteo_uplift", n_rows=3_000_000, outcome="conversion")`, B = 1000. Re-run on 2026-10-23 at 567c6b2,
after the `methods.uplift` review (758b1f8): every cell identical to the digit to the 2026-10-22 pre-review run.
Wall time 7.7 min (load 1 s, fit 45 s, score 417 s; was 8.8 min), peak RSS 3.0 GB, `nice -n 19`.

| method | qini | auuc | uplift@30 |
|---|---|---|---|
| s_gbdt | 0.192 [0.123, 0.263] | 0.006 [0.004, 0.008] | 0.003 [0.002, 0.005] |
| t_logistic | 0.172 [0.101, 0.240] | 0.006 [0.003, 0.008] | 0.003 [0.002, 0.004] |
| t_gbdt | 0.074 [0.011, 0.134] | 0.002 [0.000, 0.004] | 0.002 [0.001, 0.003] |
| random | 0.021 [-0.018, 0.059] | 0.001 [-0.001, 0.002] | 0.001 [0.001, 0.002] |
| class_transform | -0.083 [-0.135, -0.032] | -0.003 [-0.004, -0.001] | 0.001 [0.000, 0.001] |

A 1M-row visit run (`n_rows=1_000_000`, B = 200) was repeated on 2026-10-23 and gave identical numbers to the
digit (t_logistic qini 0.090 [0.067, 0.118], random -0.006), so the runs are deterministic given the seeds.

Why class transformation fails on Criteo conversion (measured, 2026-10-23): a LightGBM classifier predicting the
treatment flag from f0..f11 (3M random rows, 80/20 split, same GBDT parameters) reaches held-out AUC 0.509, and the
treated share runs from 0.846 to 0.867 across its predicted-propensity deciles (about 60,000 rows each, so a binomial
SE near 0.0015). Class transformation assumes a constant treated share; the reweighting fixes the overall share
but not the share at x. With a rare outcome z is almost 1 - t, so P_w(z=1|x) is about 1 - e_w(x), the reweighted
treated share at x (e_w = 0.5 at e = 0.85, slope de_w/de = 0.25 / (e (1-e)), about 2). A 0.02 swing in e(x) moves
the score by about 0.08 (2 x 2 x 0.02), against a 0.001 conversion uplift, so the method ranks by inverse
propensity. The T- and S-learners do not have this term. With visit (uplift 0.01) the bias is smaller and
class transformation only trails.

What the numbers say: on Hillstrom and X5 every baseline sits within a few hundredths of `random` and most CIs
overlap it; the mens e-mail arm shows no detectable heterogeneity at all. The intervals are wide at these test sizes
(Diemert et al. 2021 report that AUUC differences between their baselines become separable only from about 1M test
rows on Criteo). Treat a new method's win on Hillstrom or X5 as real only when its CI clears the best baseline.
On Criteo visit every learner clears `random` clearly (qini about 0.08-0.09 vs 0.00), but the four learners sit
inside each other's CIs even on the full 2.8M-row test set, apart from class transformation, which trails slightly.
On Criteo conversion the ordering is unstable (wide CIs) and class transformation goes below random (see above).

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

- X5 `first_redeem_date` runs to 2019-11, past the 2019-03-18 end of the purchase history, and spikes on
  2019-03-21/22 (the SMS). A first redemption after the cutoff is post-treatment: among clients with one, 50.8% are
  treated (11,490 vs 11,110, 2.5 binomial SE above one half), among clients with none 48.1% (8,438 vs 9,108, 5 SE
  below), against 50.0% among clients who redeemed before (measured 2026-10-23). The loader masks
  those dates (redeem day and delay become missing, as for a client who has not redeemed yet). The cutoff is
  `X5Uplift.CUTOFF`; the campaign date itself is not in the files.
- X5 uses client profile features only; the purchase history (`purchases.csv.gz`, 639 MB) is not read, and winning
  X5 solutions are built on it, so these X5 numbers understate what is reachable.
- Criteo is subsampled non-uniformly by the publisher for privacy, so its ATE and uplift levels are not real ad
  effects (dataset README). The treated share also depends slightly on the features (held-out AUC 0.509, measured
  above); `class_transform` does not correct for this, so do not use it as the Criteo conversion reference.
- The full-data Criteo run is B = 200, the others B = 1000. The full run needs about 7.3 GB RAM.
- `kind = "binary"` describes the outcome; predictions are scores, so `simengine.evidence.evaluate`'s binary metrics
  (AUC on truth, Brier) are not meaningful on this frame. Use `task.score`.
- No hyper-parameter search, no cross-fitting, a single random split; the Hillstrom `spend` outcome is not offered.
- Normalised AUUC / Qini depend on the perfect-curve convention; compare only with numbers computed the same way.
- The tasks are registered when `simengine.bench.tasks.uplift` is imported; the package `__init__` does not import
  it yet.
