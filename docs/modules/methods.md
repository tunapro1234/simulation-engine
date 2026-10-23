# simengine.methods

This module holds the non-LLM baselines that every simulation result is reported against, and the hybrids
that combine an LLM prediction with a model learned from data.

| component | what it is | verdict from the prep nights |
|---|---|---|
| `ConstantBaseline`, `MajorityBaseline`, `ItemMarginal`, `SegmentRate` | rate / median / class shares, per item or shrunk per segment | always report them; they are hard to beat on accuracy |
| `ConditionalMode`; `MedianConstant`, `CellMedian`, `stat_for_metric` | multiclass mode per key cell (person x slot x day-type) with neighbour smoothing, time decay and a backoff chain, from sparse counts; median / mean / quantile constant and shrunk cell median for MAE ladders | report the median rung next to the mean on any MAE task (D12 pilot: median +0.0075 zMAE better); measured runs pending |
| `uniform_distribution`, `weighted_distribution`, `overall_copy`, `segment_distributions`, `segment_empirical`, `select_m` | distribution baselines for group-level opinion; `mode="shrunk"` is segment empirical shrunk toward the population, `m` picked by K-fold cross-validation (`select_m`) | `overall_copy` wins against the uniform distribution and against neighbour transfer; with segment training data, report shrunk segment empirical next to `overall_copy`: they are the LLM-free arms to beat |
| `PersistenceBaseline`, `persistence` | "tomorrow = yesterday" from the unit's own history | strongest single signal wherever history exists |
| `basket_tensor`, `popularity`, `personal_frequency`, `decayed_frequency`, `tifu_knn` | next-basket / repeat-purchase scorers | decayed frequency beats P-TopFreq; TIFU-KNN adds nothing |
| `LogisticBaseline`, `LightGBMBaseline`, `KNNBaseline`, `TextKNNRegressor` | supervised tabular / text baselines | GBDT ties persistence when history is available; text kNN on effect sizes is weak |
| `LengthAB`, `TfidfAB`, `TfidfWordAB`, `LLMDirectAB`, `LogitAverageAB`, `ABStack` | A/B winner of two texts | stack (TF-IDF + LLM) is the best measured arm, at **0.817** on Upworthy |
| `EarlyTestBayes`, `simulate_saving_calibrated`, ... | LLM prior + small early test, normal-normal Bayes on the log-odds ratio | **about 2.3x** impression saving at 100 impressions/arm, no gain at 800+ |
| `ShrunkDeviation`, `ridge_to_prior`, `select_lambda`, `LLMFeatureGBDT`, `DevStack`, `BinaryCalibrator` | ways of letting an LLM prediction into a data model, tuned on dev | the LLM term mostly adds ~0 once a strong data baseline exists; tuning keeps it from hurting |
| `TwinStack`, `pooled_logistic_oof`, `column_correlation`, ... | person-level digital twin: matrix completion + kNN + optional LLM column, ridge-stacked per column | Twin-2K-500 fair LLM-free baseline **0.342** corr; LLM adds +0.006 |
| `MNL`, `evaluate_mnl`, `null_scores`, `person_holdout`; `conjoint.wtp`, `conjoint.simulate_shares`, `conjoint.elasticity`; `conjoint_hb.HBMNL`, `conjoint_hb.MixedLogit` | conditional multinomial logit for conjoint / discrete choice with Hessian standard errors, WTP (delta / person-cluster bootstrap CI), share simulation and price elasticities; hierarchical Bayes MNL (person posterior for known persons, population mixture for new ones) and an xlogit mixed-logit wrapper | 0.509 new-person accuracy on camera conjoint (HB 0.519); same person, new task MNL 0.482 vs **HB 0.715** |
| `Markov2Baseline`, `NextActionRanker`, `hit_at_k`, `top1`, `next_action.sessions` / `funnel`, ... | next-action ranking: Markov-2 with backoff, LightGBM lambdarank; step table -> session table (first-k prefix, outcome) and per-step funnel | 26.94 vs **31.60** exact next action on OPeRA |
| `fit_mixture_weights`, `probe_types`, `identifiability`, `mixture` | theory-persona mixture calibration (Homo Silicus types, GSA selection): simplex grid fit, identifiability warning, held-out validation | reproduces the edsl-test fit (.36 / .50 / .14, train SSE .089); held-out dictator RMSE **.315**, i.e. the calibrated mixture does not transfer |
| `GroupTransfer`, `oof_predict` | group transfer (predict people in groups never seen in training): ridge group level on group composition (+ optional LLM group prior) + person deviation model; out-of-fold predictions with group-held-out CV | D2 rehearsal, country-held-out CV person Spearman: person-only GBDT 0.197, two-stage 0.257, two-stage + LLM prior **0.295** (oracle level 0.344) |
| `TfidfBinary`, `target_encode` | one text per row -> P(y=1): word 1-2 + char_wb 3-5 TF-IDF (+ extra / target-encoded columns) logistic with in-train C choice; smoothed OOF mean encoding | G4 rehearsal (first search query -> purchase, 30,365 training sessions) test AUC **0.666** [0.617, 0.714]; 500 sessions 0.579 |
| `rct_effects`, `rct_transfer` | RCT condition (x segment) effects vs control within blocks, averaged over blocks, with SE; "same test in another market" transfer baseline | G1 rehearsal (11 messages, 32 -> 31 countries) transfer Spearman **0.845** [0.43, 1.00], noise ceiling 0.908; one country alone median 0.28 |
| `heterogeneity_test`, `segment_uplift`, `shrink_effects`, `split_half_effects`, `TLearner`, `SLearner`, `qini`, `auuc`, `uplift_at_k` (`uplift.py`) | uplift / CATE for randomised experiments: is the effect heterogeneous at all (treatment x feature test + Cochran Q), segment effects with Neyman SE shrunk by Fay-Herriot (REML tau^2), a split-half dev rig that splits units before bootstrapping, T-/S-learners and the Qini / AUUC / uplift@k metrics the bench uplift tasks use | D11 rehearsal (SMS pilot -> rollout, 26 segments): pilot ATE to every segment tied the strongest baseline (MAE 1.538 vs 1.526); the hand T-learner did not finish in 14 min and the test said it was not needed |
| `item_holdout_features`, `MFBaseline`, `procrustes_align` | item cold start (new items with no rating): member statistics + SVD taste vectors rebuilt with each fold's items removed, fold bases aligned by orthogonal Procrustes (rotation + one scale factor); biased MF (ALS) ladder baseline | D3 rehearsal recipe (port of `kod/ozellik.py`); no measured gain reported for the MF rung, and it reduces to member level + constant in pure item cold start |
| `persistence_forecast`, `seasonal_naive`, `moving_mean`, `seasonal_ratio`, `ses`, `make_features`, `DirectForecaster`, `backtest`, `summarize_backtest`, `point_for_metric`, `llm_prior_feature`; `forecast.panel_from_long`, `forecast.extrapolation_check` | multi-series forecasting (`forecast.py`): rule ladder, leak-free lag/rolling features, one GBDT per horizon (level, log-ratio or logit-with-ceiling target; origin-time features), rolling-origin backtest, Poisson median for L1 metrics, LLM prior feature with blind + shuffled controls, long `id, time, variable, value` table -> panel, extrapolation warning | D4 rehearsal (M5 weekly, 4 weeks ahead, one run) test WMAPE: persistence 0.418, last-4 mean 0.379, SES 0.371, GBDT + price + calendar **0.348**; LLM event prior added nothing (named = blind) |
| `wave_persistence`, `pooled_shrunk`, `same_regime`, `subgroup_mixture`, `mixture_reweight`, `role_swap`, `transition_backtest`, `merge_frames` (`waves.py`) | repeated surveys: past waves -> next wave's segment x question distributions; last-wave copy, k-wave pool shrunk to the population, same-regime pool, party-ID mixture with role swap across a regime switch, per-year backtest over past switch years | D6 rehearsal (yearly opinion survey, D -> R president, one run): copying last year 0.1315 test TVD (approval 0.295), role-mirror baseline 0.0839, team 0.0788; on dev the same-regime 4-wave pool beat the preregistered method, 0.0990 vs 0.1007 |

**Provenance.** Everything is ported from the TALP prep research branches under
`talp-simulation-hackathon-2026/uretim/<branch>/`. The algorithms and defaults are unchanged. Shorthand used below:

- **[deney]** = `uretim/deney/RAPOR.md` (A/B, early test, SocSci210, conjoint, risky choice)
- **[intent]** = `uretim/intent/RAPOR.md` (purchase intent, next basket)
- **[persona]** = `uretim/persona/RAPOR.md` (Twin-2K-500)
- **[web]** = `uretim/web/RAPOR.md` (OPeRA next action)
- **[makro]** = `uretim/makro/RAPOR.md`, **[llmabm]** = `uretim/makro/llmabm/RAPOR.md`
- **[dongu]** = `uretim/dongu/GUNLUK.md` (OpinionQA group loop)
- **[prova-d2]** = `uretim/prova/d2/NOTLAR.md` and `SIMENGINE-EKSIK.md` (blind rehearsal: new-country survey answers; code `kod/cv.py`, `cv_biz3.py`, `tahmin.py`)
- **[prova-d4]** = `uretim/prova/d4/NOTLAR.md`, `SONUC-MAIN.md`, `SIMENGINE-EKSIK.md` (blind rehearsal: M5 weekly demand, 30,490 series, 4 weeks ahead, WMAPE; code `kod/ozellik.py`, `ses.py`, `kos.py`, `pm.py`, `ek.py`, `llm_olay.py`)
- **[prova-d3]** = `uretim/prova/d3/SIMENGINE-EKSIK.md` (blind rehearsal: new-film ratings, item cold start; code `kod/ozellik.py`, `kod/model.py`)
- **[prova-d6]** = `uretim/prova/d6/SIMENGINE-EKSIK.md`, `SONUC-MAIN.md` (blind rehearsal: yearly opinion survey 2006-2024 -> 2025 segment x question distributions across a president switch, cell TVD)
- **[prova-d11]** = `uretim/prova/d11/SIMENGINE-EKSIK.md`, `SONUC-MAIN.md` (blind rehearsal: SMS pilot RCT -> rollout segment uplift, 26 segments)
- **[G1]** = `uretim/recete/G1/RECETE.md`, `kart.md` (open recipe rehearsal: climate many-labs, 11 messages vs control, 32 training / 31 test countries); **[G4]** = `uretim/recete/G4/RECETE.md`, `kart.md`, `CAPRAZ-G1.md` (open recipe rehearsal: ShopCART first query -> purchase, 1,000 test sessions)

Every number below comes from those files. This port re-measured nothing. The tests
(`tests/test_methods_*.py`) only check behaviour on synthetic data.

## Minimal usage

```python
import simengine.methods as M
from simengine.llm import FakeLLM          # any simengine LLM protocol object works the same way

# --- A/B: which headline wins?  frame columns text_a, text_b, y (1 = A won), group (Turkish aliases accepted)
llm_arm = M.LLMDirectAB(llm, context="news headlines, outcome = click-through rate")
stack = M.ABStack(llm_method=llm_arm).fit(train)          # holds out 200 groups as dev for the stack weights
p_a = stack.predict(test)                                 # P(A wins)

# --- early test + prior: P(A better) after 100 impressions per arm
b = M.EarlyTestBayes().fit(prior_prob_finished, lor_finished, se_finished)
b.predict(prior_prob_new, clicks_a, 100, clicks_b, 100)

# --- digital twin: person x feature answers, person x target answers (NaN = unknown)
twin = M.TwinStack(seed=0).fit(features, targets, llm=llm_twin_or_None, nominal=["region"])
point, sample = twin.predict(test_person_ids)
M.column_correlation(truth.loc[test_person_ids], point)

# --- conjoint (self-contained): long rows, one per (task, alternative); task ids unique across persons
import numpy as np, pandas as pd
rs = np.random.default_rng(0)
rows = pd.DataFrame([{"person": t // 8, "task": t, "alternative": j, "price": rs.uniform(50, 150),
                      "quality": rs.uniform(0, 3)} for t in range(2000) for j in (1, 2, 3)])
u = -0.03 * rows["price"] + 0.9 * rows["quality"] + rs.gumbel(size=len(rows))
rows["chosen"] = (u == u.groupby(rows["task"]).transform("max")).astype(int)
train_rows, test_rows = M.person_holdout(rows)            # 20% of persons held out
m = M.MNL(["price", "quality"], scale={"price": 100}).fit(train_rows)
m.coef_, m.se_, m.accuracy(test_rows), M.null_scores(test_rows)
from simengine.methods.conjoint import wtp
wtp(m, price="price")                                     # true 0.9 / 0.03 = 30 price units; delta-method CI
from simengine.methods.conjoint_hb import HBMNL
hb = HBMNL(["price", "quality"], scale={"price": 100}, draws=1000, burn=1000, seed=0).fit(train_rows)
hb.log_likelihood(test_rows, new=True)                    # new persons; known persons: hb.predict_person(rows)

# --- next action: steps (ex, session, step, action, user, page) + candidates (ex, key, features...)
te, s = M.Markov2Baseline().fit(tr_steps, tr_cands).score(te_steps, te_cands)
M.hit_at_k(te, s, ks=(1, 5))
```

Conventions: classes have `fit(...) -> self` and `predict(...)`. Binary and multiclass classes also have
`predict_proba`. `kind` is `"binary"` (P(y=1)), `"multiclass"` (probability matrix, columns `classes_`) or
`"continuous"`. The default `"auto"` infers it. LightGBM is imported only when a LightGBM class is fitted
(`lightgbm_available()` tells you whether it is installed). LLM components call only `llm.map(...)` /
`llm.complete(...)` from `simengine.llm`.

## Baselines (`baselines.py`)

Ported from the intent kit (`intent/kod/kit/yontemler.py`: `_sabit`, `segment_oran`, `kalicilik`, `lineer`, the
LightGBM defaults), the persona kit (`Medyan`), the deney kit (`KomsuDeney`), kanit `merdiven.py`, makro
`temel_c1.py` and intent/seq `nbr/klasik.py`.

- **Constant / majority / item marginal.** Report them next to every accuracy. On Twin-2K-500 the column median
  alone scores accuracy 0.755, above the raw LLM twin at 0.717 **[persona]**. This is why the twin section uses
  across-person correlation as its metric.
- **`SegmentRate`** computes `(sum + a*g) / (n + a)`. `a` is picked from (1, 5, 20, 50, 200) on validation; the
  default is `a = 10`, as in kanit.
- **`ConditionalMode`** is the multiclass counterpart for long tables: the most frequent label per key cell,
  `ConditionalMode(["person", "slot", "daytype"], target="place", smooth={"slot": [(-1, .5), (1, .5)]},
  wrap={"slot": 48}, backoff=[["person"], ["slot", "daytype"]], decay=0.95, time_col="day")`. A cell also counts
  its neighbours' rows at the given weights (its own rows weigh 1); cells with no weight fall down the backoff
  chain and finally to the global mode (`levels_`, `predict(..., return_level=True)`). Ties go to the globally
  more frequent label. `predict(q, k=3)` returns the top 3, padded from the next levels. Counts are sparse
  groupbys, so no dense user x period x item tensor is built (1M history rows, 200k queries: fit ~2 s, top-1
  ~0.3 s, top-3 ~5 s on the dev box). It generalises `abm.mobility.slot_mode`; nothing measured with it yet.
- **MAE ladder rungs.** `MedianConstant()` is the training median (`metric="mse"` gives the mean, `stat=0.9` a
  quantile, `fit(y, weight=w)` a weighted one). `CellMedian(by=[...], y_col=..., shrink=k, weight=...)` gives
  `(n * cell + k * global) / (n + k)` per cell, n = rows (Kish n with weights); `shrink=None` with
  `fit(train, valid=dev)` picks k from (0 ... 500) by the metric's loss (absolute, squared or pinball).
  `stat_for_metric` maps MAE / WAPE / zMAE to the median and MSE / RMSE to the mean, as `point_for_metric` does
  for forecasts. On the D12 pilot the median constant beat the mean by +0.0075 zMAE.
- **Distributions.** For OpinionQA subgroups (setting B, Overall known), copying Overall to every group gives
  dev WD 0.0530 [0.0509, 0.0554]. The uniform distribution gives 0.1976 and "sibling" transfer gives 0.0764
  **[dongu]**. Adding neighbour-group deviations on top was significantly worse (+0.0014 [+0.0006, +0.0021]
  WD). For the makro C1 segment task (W1): the overall marginal scores 0.206 and the marginal of the other three
  segments 0.274. The same party's other age group scores 0.040, against a 37-person survey ceiling of 0.050
  **[makro]**. Use `segment_distributions(mode=...)` for these.
- **LLM-free distribution ladder (segment x item answer shares).** T0 uniform, T1 population copy
  (`mode="overall"`), T2 segment empirical (`mode="own"`), T2b shrunk segment empirical (`mode="shrunk"`,
  p = (n_s * own + m * overall) / (n_s + m), n_s = valid answers in the segment, Kish effective n with weights).
  In the G5 rehearsal (1-TVD over segment x item cells, train households only) T2b with m = 20 led on seed 0
  (0.912 against T1 0.906), T1 led on seed 1, and the LLM hybrid did not beat T2b across 5 seeds (H - T2b mean
  -0.004, sd 0.008) **[G5]**. For any distribution task with segment training data report both T1 and T2b and
  take the stronger as the baseline to beat.
  `select_m(train, items, segment, categories, unit_col=...)` picks m by K-fold cross-validation over units
  (households, so one never sits on both sides; `folds=None` is leave-one-unit-out). The default
  `score="l2"` (squared distance to the held-out empirical) is proper: held-out noise shifts every m alike.
  `score="tvd"` uses the task metric but rewards sharp predictions on small held-out cells and, with one answer
  per cell, can only return an end of the grid; `result.tvd` reports held-out 1-TVD either way. Leave-one-unit-out
  refits every grid value per unit, so it is slow on hundreds of households; 5 folds is the default.
  `segment_empirical(..., shrink=m)` is the
  short alias. Cross segments: pass a list of columns (keys become tuples); for hierarchical shrinkage pass
  `prior={cell: parent_distribution}`, e.g. the one-column shrunk result of the cell's first column. `segments=`
  returns cells absent from the training rows (they get the target distribution).
- **Persistence.** On the open e-commerce order-frequency class, persistence scores macro-F1 0.781
  [0.756, 0.805] against 0.275 for majority. The best GBDT with all features was not separable from it
  (+0.001 [-0.022, 0.023]) **[intent]**. On makro per-item opinion drift, the previous year wins on some items
  (Trump 0.025, better than the LLM) and loses to the marginal on others (economy 0.275) **[makro]**. Always
  run it first.
- **Next basket.** `decayed_frequency` (r=0.9, beta=0.05) beats P-TopFreq on test MAP@10: OE +0.023
  [0.019, 0.027], CJ +0.015 [0.011, 0.019]. `tifu_knn` gave no gain. A person x category LightGBM beat decay on OE
  (+0.012 [0.009, 0.015]) and tied on CJ; **it is not ported** **[intent]**.
- **Supervised.** `LightGBMBaseline` uses up to 2000 trees with early stopping 100 when `eval_set` is given
  (else 300), lr 0.05, 31 leaves, min_child 50, subsample/colsample 0.8. It runs single-threaded below 200k rows,
  because a loaded machine was 10-40x slower with OpenMP. `LogisticBaseline` is logistic regression with C=1 on
  one-hot + standardised features (ridge for continuous targets). `KNNBaseline` defaults to k=50.
  `TextKNNRegressor` (TF-IDF neighbours from other groups) predicts SocSci210 effect sizes at r 0.080
  [-0.024, 0.232] **[deney 1.4]**, which is weak. Ridge scored 0.017; an Opus direct estimate scored r 0.509 but
  was judged not presentation-ready.

## A/B winner (`ab.py`)

Port of `deney/kod/kit/yontemler/ab.py`. Measured on the same 300 significant Upworthy tests (chance 0.50)
**[deney 1.1]**:

| arm | accuracy [95% CI] |
|---|---|
| `LengthAB` (longer wins) | 0.533 [0.477, 0.588] |
| LLM direct: Haiku 5.5 / Sonnet 5.5 / Opus 5.5 (`LLMDirectAB`) | 0.608 / 0.688 / 0.765 [0.720, 0.808] |
| Opus persona panel | 0.685 |
| `TfidfAB` (word 1-2 + char 2-5 gram difference, no-intercept LR, C=0.3) | 0.773 [0.727, 0.820] |
| `LogitAverageAB` (fixed 0.5 / 0.5) | 0.803 [0.757, 0.847] |
| **`ABStack`** (no-intercept LR on the two logits, weights from dev) | **0.817 [0.770, 0.860]**, AUC 0.906 |
| `ABStack`, story-clean training | 0.807 [0.760, 0.847] (+4.3 [+0.7, +7.7] over TF-IDF) |

A blind one-shot run on 136 new pairs scored 0.824 [0.757, 0.890], +7.4 [0.7, 14.7] over TF-IDF **[deney 1.3]**.
In the kit rehearsal the stack scored 0.813, TF-IDF 0.780, `TfidfWordAB` 0.760 and the stored weights 0.807
**[deney 3]**.

**When it does not work.**
- Much of the gain over the LLM alone comes from TF-IDF breaking the LLM's exact 0.5 ties. Over the
  "LLM, tie broken by TF-IDF" rule the stack adds +3.7 [1.0, 6.5] on ek600 and +0.7 [-4.4, 6.6] on the one-shot
  set.
- The learner needs training pairs from the same distribution. Without same-story pairs TF-IDF drops to 0.720.
- On the LOLA v1 setup, Opus direct dropped to 0.645, and the stack with Opus (0.775) did not beat bge + TF-IDF
  without an LLM (0.788). The LOLA paper itself reports 0.823 with OpenAI-3072 embeddings on that setup
  **[deney 4.1]**.
- Over all unselected pairs (not only significant ones), content-based prediction reaches 0.618. The noise
  ceiling there is about 0.74, the repeat-winner rate reported by Hagar et al. 2022 **[deney 4]**.

`ABStack` gets its dev set from the `dev` frame you pass, or else holds out `dev_n=200` groups from training.
If neither is possible it falls back, with a warning, to the stored Upworthy dev300 weights (1.3, 2.75). Invalid
LLM answers count as logit 0. `LLMDirectAB` asks 15 pairs per call and, by default, asks again with A and B
swapped (position bias). `coverage_` is the share of pairs with a valid answer. `calls_` counts requests actually
sent and `cache_hits_` answers served from the disk cache, so a fully cached re-run reports `calls_ == 0`.

To report the direct LLM arm next to the stack without asking the LLM about the test pairs twice, pass the direct
arm's probabilities to the stack (`fit` has already asked about dev):

```python
from simengine.methods.ab import ABStack, LLMDirectAB, ab_archive, ab_responder

pairs = ab_archive(raw, test="test_id", text_a="headline", clicks_a="clicks", n_a="impressions")
direct = LLMDirectAB(llm)
stack = ABStack(llm_method=direct).fit(train)
p_direct = direct.predict(test)
p_stack = stack.predict(test, p_llm=p_direct, valid=direct.valid_)
```

`ab_archive` turns a past A/B archive into this standard frame (`group`, `text_a`, `text_b`, `clicks_a`, `n_a`,
`clicks_b`, `n_b`, `ctr_a`, `ctr_b`, `tie`, `y`, `p_value`, `significant`), the frame `LLMDirectAB`, `ABStack` and
`simulate_saving_calibrated(f.ctr_a, f.ctr_b, prior, f.clicks_a, f.clicks_b)` take. Input can be wide (one row per
pair: pass the `_b` columns too; your A/B sides are kept) or long (one row per variant, as in the raw Upworthy
archive: leave the `_b` columns out, and every two variants of a test form a pair). In long input a seeded coin puts
each pair's variants on A or B (`random_sides=True`), so archive row order cannot leak into `y`: on an archive
listed best-first (simulated), "A = earlier row" gives y = 1 for 88% of pairs and an LLM that always says A scores 0.88; with
random sides both are 0.51. By default it also drops pairs with the same text on both sides (image-only tests)
and exact CTR ties, and keeps one seeded random pair per test, picked after the filters. `p_value` is a pooled
two-proportion z-test and `significant` is p < 0.05 (nominal; 5.0% of simulated A/A pairs). The measured numbers
above are on significant pairs, so compare against them with `significant_only=True` (applied before the
one-pair pick); the deney sets used FDR, because Upworthy A/A pairs hit nominal p < .05 14.6% of the time. Clicks
outside [0, n] raise. For a dry run with no API, `FakeLLM(ab_responder(score_fn, noise, seed))` answers
the `LLMDirectAB` prompt with p_a = sigmoid(score(a) - score(b) + noise). With `score_fn=len` the longer text
wins; the default is a hashed score that carries no signal.

## LLM prior + early test (`bayes_prior.py`)

Port of `deney/kod/analiz/ornek_tasarrufu_llm.py` and `ornek_tasarrufu_llm_ga.py`. The prior on the true
log-odds ratio is N(k * logit p, tau^2): `calibrate_prior` fits k through the origin and sets tau^2 =
Var(residual) - mean se^2. An early test's Haldane-corrected LOR is then combined with that prior by precision
weighting. Two replay protocols convert accuracy into an "equivalent impressions" multiplier, and they give
different numbers on the same data:

- `simulate_saving` (port of `ornek_tasarrufu_llm.py`): 2-fold cross-fitted prior (calibrated on half the pairs,
  about 150), 300 binomial replays per pair, seed 11, no CI. Same RNG order as the script, so the same input and
  seed give the same numbers. Saved output `ornek_tasarrufu_llm.json` (llm300, 300 pairs), 100 impressions/arm:
  human 0.713; Opus direct prior 0.815, **2.47x**; TF-IDF prior 2.79x; stack prior 3.68x; Opus at 400: 1.34x.
- `simulate_saving_calibrated` (port of `ornek_tasarrufu_llm_ga.py`): each of 10 blocks is calibrated on
  `n_calib` random pairs from the other blocks, 5 draws, 200 replays, seed 5, pair-bootstrap 95% CI of the
  multiplier at n=100. It also returns the "human decides, prior breaks ties" accuracy. This is the protocol
  behind the RAPOR headline below.

Measured on Upworthy llm300 with `simulate_saving_calibrated` **[deney 2.2-2.3, `ornek_tasarrufu_llm_ga.json`]**:
an Opus direct prior calibrated on 20 tests, at 100 impressions per arm, raises accuracy from 0.718 to 0.808. That
is **2.27x (1.67-3.11)** impressions. Other priors and settings (calibration on 20 tests unless stated):

| prior / setting | multiplier |
|---|---|
| Opus, 150 calibration tests | 2.58x |
| TF-IDF prior | 2.44x |
| stack prior | 3.30x |
| persona panel prior | 1.41x |
| k = 1 (`fix_scale=True`) | 2.55x |
| Opus at 400 impressions/arm | 1.34x |
| ek600 replication | 2.05x (1.70-2.50) |
| split-half (independent) | 1.99-2.28x |

**When it does not work.** There is no gain at 800+ impressions per arm. Most of the gain is tie-breaking in tests
with very few clicks: "human decides, prior breaks ties" alone gives 0.786. The replay is circular, but the
adversarial check found that this biases *against* the hybrid **[deney 2.1]**.

**Comparing priors, weak priors, censoring, the card.** These live in `simengine.methods.bayes_prior`:

- `simulate_saving_calibrated(..., return_pairs=True)` also returns `hybrid_pairs` / `human_pairs` (per-pair
  accuracy per n, means equal the aggregate) and `decisions_n0` (draws x reps x pairs, True = ship A).
- `compare_priors({"tfidf": p1, "llm": p2}, ctr_a, ctr_b)` runs every prior on the same replays, tie coins,
  calibration sets and bootstrap resamples. Each prior's numbers equal a separate `simulate_saving_calibrated`
  call. `diffs` gives accuracy and multiplier minus the first prior, `vs_human` gives accuracy minus human-only,
  both with paired 95% CIs. `cluster=<test id per pair>` resamples tests instead of pairs.
- Weak-prior guard. A RuntimeWarning (also in `warnings`) fires when the free calibration slope k spans 0 over the
  blocks x draws calibration sets (`k_calib`), or when prior-only accuracy is below 0.6. Calibrated on 20 tests,
  such a prior can cost traffic. `mode="tiebreak"` is the safe choice there: the human decides and the prior only
  breaks ties, so it cannot do worse than human-only in expectation as long as the prior is right on at least
  half of the tied tests.
- Censoring. When the hybrid beats the top of the human curve, `multiplier` is None, `multiplier_censored=True`
  and `multiplier_lower_bound = max(ns) / n0` (16x with the defaults). Bootstrap replicates above the curve rank
  above every finite one, and a CI endpoint in that tail is None (`ci_censored`). Earlier versions dropped those
  replicates, which made the CI too low. `multiplier_at_floor` marks a value clamped at the bottom of the curve.
- CI and calibration noise. By default (`boot_calibration=True`) every bootstrap replicate re-draws the
  calibration sets from its resampled pairs, so the CI carries the noise of calibrating on 20 tests. The research
  script held the calibration fixed (`boot_calibration=False`; the 2.27x (1.67-3.11) above is that CI). In a
  simulation (300 synthetic pairs, n_calib=20, 300 samples per setting) the fixed CI under-covered: nominal 95%
  gave 0.82-0.94 for the multiplier and 0.83-0.93 for the accuracy difference between two priors. Recalibrated, it
  gave 0.93-0.96 and 0.96-0.98. Point values are the same either way; only the CI is wider.
- `saving_section(result)` turns either result into an evidence-card section that
  `simengine.evidence.section_markdown` renders. It shows accuracy at n0, the multiplier with its CI and
  censoring, n_calib and the calibration slope, so the traffic multiplier can sit next to the card.

## Hybrids (`hybrid.py`)

These are the three ways an LLM prediction entered a data model during the prep nights, plus the generic
dev stack. PPI++ lives in `simengine.evidence`.

- **`ShrunkDeviation`**: `pred = base + alpha * (llm - ref)`. The level comes from data, the deviation from the
  LLM, and `alpha` is chosen on dev from 0..1. `ref` is the LLM's own group mean (makro `llm_kal`) or an explicit
  reference. For distributions, the result is clipped and renormalised. Measured on makro M2 price elasticity:
  "level from category, deviation from LLM" + data scored test WAPE 0.478, against 0.477 for category + data
  **[llmabm, M2 table]**, i.e. no gain. For OpinionQA groups, neighbour deviations at alpha = 1 were worse than
  copying Overall **[dongu]**. Expect `alpha_` near 0 unless the LLM has item-specific signal.
  The distribution recipe "population + beta * (LLM segment - LLM population)" is one call:
  `ShrunkDeviation(distribution=True).fit(base=pop, llm=llm_seg[train], y=true_seg[train], ref=llm_pop)`, with
  `base` the known overall distribution, `ref` the LLM's own overall distribution and beta = `alpha_` chosen on
  the train cells (pass `alphas=np.linspace(0, 1.5, 16)` to allow beta > 1).
- **`ridge_to_prior` / `select_lambda`**: ridge toward a prior coefficient vector, with lambda chosen on held-out
  rows from (0.3, 1, 3, 10, 30, 100, 1e9). This is the makro "+data" step. The best M2 test WAPE was 0.469
  (LLM expert prior + data), against 0.477 for category + data. A shuffled-LLM placebo reproduced about 85% of
  that gain, so it is shrinkage from averaging priors, not LLM knowledge **[llmabm]**.
- **`LLMFeatureGBDT`**: the LLM prediction as one more LightGBM column. It gave no gain where a strong data
  baseline existed. Intent OE category ranking MAP@10 went from 0.3307 to 0.3303 **[intent]**. CPC18 risky choice
  MSE x100 went from 0.585 to 0.627, i.e. worse **[deney 1.5]**. Always compare against the same model without
  the column.
- **`DevStack`**: logistic regression on logits (binary) or log-probabilities (multiclass), or ridge
  (continuous), over several methods' dev predictions. `dev_pred_` holds honest 5-fold cross-fit dev predictions.
  **`BinaryCalibrator`** uses Platt scaling, or isotonic regression from 1000 dev rows (intent kit `kalibre`).
  `method="ratio"` only rescales the level (p * base_rate / mean(p), clipped to [0, 1]) and keeps the ranking; use
  it for a rate/funnel question or a small dev set. With `monotone=True` (default) a Platt slope <= 0 falls back
  to the ratio method with a warning (`fallback_`), and Platt on fewer than 200 dev rows warns. In the G4
  rehearsal, 6.6% of n = 50 Platt fits had slope <= 0 and turned test AUC 0.624 into 0.376. No separate accuracy
  is claimed for these two; they are the plumbing that stops a weak LLM arm from hurting.

## Digital twin (`twin.py`)

Port of the persona kit `Stack`, on plain wide DataFrames (person x column). The LLM-free bases are
`svdridge_r30` (PCA of the person's other answers + per-column RidgeCV), `softimpute_r10` and `knn50`, plus an
optional LLM column. They are stacked per column with RidgeCV, with missingness indicators. Train persons' base
predictions are 4-fold inner OOF, and a test person's target is never seen. The point prediction is the stack
value, or the binned conditional median when `binned=True`. The sample is a hot-deck donor copy by default. Donor
sampling keeps the correlation structure between answers at 0.894, against 0.464 for the residual bootstrap.

If a column has fewer than 20 labels, `TwinStack` falls back to the calibrated LLM, or to the pooled distribution
when there is no LLM. When an LLM is present, columns with fewer than 200 labels also use the calibrated LLM. That
200-label threshold comes from the k-curve: the stack catches the raw twin at about 150 labels.

Measured on Twin-2K-500 T1 (2058 persons, 122 target columns) **[persona]**:

| method | across-person corr | accuracy |
|---|---|---|
| raw GPT-4.1-mini twin | 0.175 | 0.717 |
| `TwinStack` without LLM (stack_ml) | 0.204 | 0.732 |
| `TwinStack` + GPT-4.1-mini column | 0.275 | 0.735 |
| **fair LLM-free**: stack_ml + pooled logistic for the 40 price columns | **0.342** | 0.739 |
| fair hybrid (same + LLM) | 0.347 | 0.740 |
| test-retest ceiling | 0.565 (sqrt 0.75) | 0.817 |

In the fair LLM-free row, the 82 non-price columns come from stack_ml. The 40 purchase-at-price columns come from
`pooled_logistic_oof`: one logistic regression on product one-hot + price ratio + the stack_ml OOF prediction,
with person folds. It scores 0.536 on those columns, against 0.144 for the per-column stack
(`persona/kod/eval/k2_fiyat.py`).

**When it does not work.**
- The LLM's person-specific contribution over the fair baseline is only +0.006 [+0.004, +0.007]. An earlier
  "+0.071 from the LLM" claim was **retracted**: the LLM was reading the price in the prompt, not knowing the
  person.
- The LLM column pays off only with few labels. On non-price columns its share is +0.035 at 100 labels and
  +0.010 at 800.
- For questions nobody in the panel has answered yet (T2), the calibrated LLM is the only signal (corr 0.161).
  No LLM-free method replaces it there.

The combination of `TwinStack` + `pooled_logistic_oof` is not wrapped as one class: which columns share a pooled
model is dataset-specific.

## Conjoint (`conjoint.py`)

Port of `deney/kod/eval/baseline_choice.py`. Generic attributes (optionally scaled), plus ASCs for every
alternative except the base one, feed an availability-masked softmax per task fitted with L-BFGS-B from zeros.
The port adds the analytic gradient and Hessian; the optimum is the same. `evaluate_mnl` reproduces the source
evaluation: 20% of persons held out, argmax accuracy, mean log-likelihood per task, against the null (equal share)
and an ASC-only model. Each arm has the source keys `acc` / `ll_per_task` and the aliases `accuracy` /
`log_likelihood` (`null_scores` too). Coefficients come back at full precision; the source rounded them to 3
decimals, which moved WTP by about 8%.

Long format, one row per (task, alternative): `task` (unique across persons), `alternative` (the ASC key),
`chosen` (0/1), optional `available` (default 1) and `person`, plus numeric attribute columns; aliases
gorev_id / alternatif_id / secildi / mevcut / birey_id. `fit` checks it:

- exactly one chosen row among the available rows of a task; more than one raises (usually task ids reused
  across persons), none drops that task with a warning. Model "none of these" as its own alternative row
  (attributes 0, its own ASC), not as a task without a chosen row;
- attribute columns listed in `attrs` but absent raise and name the missing ones;
- no attribute columns at all (`attrs=None` looks for `Feat*` columns) raises instead of silently fitting
  constants only; pass `asc_only=True` for that baseline.

Fitted: `coef_` (part-worths per *scaled* unit, then `ASC_<alt>`), `se_` and `cov_` (inverse analytic Hessian),
`attrs_`, `alts_`, `base_alt_`, `converged_` (L-BFGS-B success), `loglik_`, `n_tasks_`.

Measured on the Sawtooth camera conjoint **[deney 1.5]**: for a new person, holdout accuracy is 0.210 with equal
shares, **0.509 with MNL** and 0.519 with hierarchical Bayes. For the same person on a new task, MNL scores 0.482
against **0.715** for HB. Person-level tastes matter there and MNL ignores them. LLM arms did not beat MNL or HB on
these sets.

### Pricing recipe: personas -> choice tasks -> MNL -> WTP -> elasticity

`examples/conjoint_pricing.py` runs the whole chain offline in about 2 s (FakeLLM with a hand-written responder).

1. **Personas.** A panel of persona texts. For a dry run, give each hidden part-worths so the true WTP is known.
2. **Choice tasks.** 3-4 random profiles per task (price on a grid, the other attributes at their levels) plus a
   "none" alternative; 8-12 tasks per persona. Ask each persona each task and keep one `chosen` row per task.
   Every (persona, task) gets its own `task` id.
3. **MNL.** `m = MNL(attrs, scale={"price": 100}).fit(rows)`. Scale price to about unit size for the optimiser;
   `coef_` / `se_` are per scaled unit, everything below undoes the scale.
4. **WTP.** `wtp(m, price="price")` returns `wtp = -beta_attr / beta_price` (price units per raw attribute unit)
   with `se`, `lo`, `hi` from the delta method on `cov_`. `ci="bootstrap", cluster="person", B=200, seed=0`
   refits on persons drawn with replacement (a person drawn twice gets fresh task ids) and returns percentile
   bounds; prefer it when persons answer many tasks, since the Hessian treats tasks as independent.
   `attrs=["ASC_2"]` gives the price premium of an alternative over the base one.
5. **Shares and elasticity.** `market` is a DataFrame with one row per alternative (attribute columns, plus
   `alternative` when the model has ASCs). `simulate_shares(m, market)` gives logit shares;
   `elasticity(m, market, price="price", price_grid=[0.9, 1.1])` returns `share`, `own`
   (beta_p * p * (1 - s)), `cross` (matrix d log s_i / d log p_j, off-diagonal -beta_p * p_j * s_j) and `arc`
   (midpoint arc elasticity when one price alone is multiplied by each grid factor).

Pooled MNL gives one taste for everybody. Under heterogeneous tastes the part-worths are attenuated a little and
the shares carry the IIA pattern (equal cross elasticities in a column); use `conjoint_hb.HBMNL` when
person-level tastes matter.

`games.wtp` / `games.WTP_PARAMS` are a different thing: "who to punish" in the Ertan, Page & Putterman
punishment game, not willingness to pay.

### Hierarchical Bayes and mixed logit (`conjoint_hb.py`) [prova-d8]

Same long rows as `MNL`, plus a required `person` column. Import from the module:
`from simengine.methods.conjoint_hb import HBMNL, MixedLogit, tune_prior_scale`.

`HBMNL(attrs=, scale=, base_alt=, asc=True, draws=500, burn=500, thin=1, seed=0)` is the Allenby-Rossi sampler in
numpy: Gibbs over individual betas, the population mean `mu` (normal, prior N(0, 100 I)) and a full covariance
`Sigma` (inverse Wishart, nu = K + 3, scale nu I). Each person's beta moves by random-walk Metropolis with proposal
covariance step^2 Sigma; the step is adapted per person toward 0.3 acceptance during burn-in and then frozen. All
parameters, ASCs included, are random. A fixed `seed` gives identical draws. The defaults are small; for a real
run use draws and burn of 1000 or more and check `acceptance_` (0.2-0.4 is healthy).
The priors are unit-scale, so pass `scale=` to bring each attribute near unit sd (fit warns outside 0.1-10).
Unscaled price (0-200) inflated the population sd of its coefficient 40x and doubled the person-level beta
error in simulation. Predictions moved little, but WTP from `draws_` would be far too spread. With 10 tasks per
person the prior also pulls `sigma_` up (true sd 0.6 came out near 0.78, and its 90% interval covered the
truth in 26% of 50 simulated panels; 0.67 and 70% with 40 tasks); `mu_` 90% intervals covered 80-92%. Read
`sigma_` as an upper-leaning estimate, not a calibrated interval.

`prior_scale=` (default 1.0) scales the Sigma prior (IW scale nu * prior_scale * I); smaller values shrink persons
toward `mu`. With few tasks per person and many dummy levels the default loses to pooled MNL: 8 tasks, 12 dummies
and 300 simulated persons gave held-out LL per task -0.84 to -0.95 for HB against -0.78 to -0.84 for MNL when tastes
barely varied. `tune_prior_scale(train_rows, grid=(1, 0.3, 0.1, 0.03), **hb_kw)` holds out one task per person
inside the training rows, scores each value with `predict_person` and returns `(best, scores)`; refit on all rows
with `best`. In the same simulation it brought HB to within 0.02 of MNL there, and ahead of the default HB where
tastes varied widely. Report HB next to MNL either way.

- `predict_person(df)`: known person, held-out task. Averages the probabilities over that person's own draws.
  Persons not seen in fit raise, or with `unknown="population"` are scored by `predict_new`.
- `predict_new(df, n=2000)`: new person. Population mixture of betas drawn from N(mu_t, Sigma_t) over the kept
  draws.
- `log_likelihood(df, new=False)` (mean per task) and `accuracy(df, new=False)`.
- Fitted: `draws_` (draws x persons x K), `mu_draws_`, `sigma_draws_`, posterior means `beta_` (persons x names),
  `mu_`, `sigma_`, `names_`, `persons_`, `acceptance_`.

`MixedLogit(attrs=, randvars={attr: "n"}, scale=, asc=True, n_draws=200, seed=0)` wraps `xlogit.MixedLogit`
(panel by person, simulated ML). xlogit is optional and is not installed by simengine; without it `fit` raises an
ImportError that names the package. xlogit itself raises an opaque `IndexError` when rows are not contiguous per
person, so the wrapper sorts rows by person / task / alternative and pads missing alternatives as unavailable
before passing `panels`. `predict_proba` returns population (new-person) probabilities in the input row order.
For a known person, use `HBMNL.predict_person`.

On a synthetic panel with tastes that flip sign across persons (120 persons, 12 training tasks, 4 held out), HB
person-level held-out log-likelihood per task is about -0.83 against -1.04 for pooled MNL. For new persons, HB's
population mixture is level with MNL. In the D8 rehearsal (task B, known persons) HB and mixed logit tied:
HB - MXL -0.001 [-0.028, +0.025] LL per task.

## Next action (`next_action.py`)

Port of `web/kod/yontem/siralayici.py`. Each step is a set of candidate rows. Priors are computed
out-of-fold with GroupKFold by session on train, and from all of train for test:
- global frequency (+0.01 / +1 smoothing)
- page-conditional frequency
- Markov-1 and Markov-2 transitions
- per-user frequency

`Markov2Baseline` scores `f_markov2 + 1e-2 f_markov + 1e-4 f_page`. `NextActionRanker` runs LightGBM lambdarank
(300 trees, lr 0.05, 31 leaves, truncation 20) on the priors plus your own feature columns. Negatives are sampled
per step: the positives, the top 60 by prior, and 40 random candidates; when candidates carry a `kind` column,
non-click candidates (terminate, input) are always kept, as in the source. The published 31.60 is a 3-seed score
ensemble (single seeds 30.38 / 31.26 / 31.60); use `NextActionRanker(n_seeds=3)` to reproduce that protocol (the
default `n_seeds=1` trains one model). The source's dataset-specific features are not ported, so expect a different
number on your own features.

Measured on the OPeRA official test (902 steps, exact next action %) **[web]**:

| method | exact next action % |
|---|---|
| always terminate | 4.43 |
| repeat previous | 13.97 |
| zero-shot Haiku persona agent (n=300 subsample) | 13.67 [9.97, 17.63] |
| Markov-2 + backoff | 26.94 [21.42, 31.72] |
| **LightGBM lambdarank** | **31.60** [26.81, 35.90] |
| literature SFT+RL (Customer-R1) | 39.58 |

LightGBM beats Markov-2 by +4.66 [+2.04, +7.55], and by [+1.68, +7.66] user-clustered.

**When it does not work.** On a user-disjoint split, LightGBM scores 29.57 against 28.11 for Markov-2 (+1.46
[-0.14, +3.11]); the two are not separable. Part of the gain is per-user history. Use `use_user_prior=False` to
ablate it. The source's page-structure, click-type and text-overlap features were dataset-specific and are **not
ported**; pass equivalents as `numeric` / `categorical`.

**Any query x candidate table.** `NextActionRanker(query_col="qid", cand_col="cell", features=["hist", "dist",
"zone"])` skips the session schema: `fit(table)` / `score(table)` take one row per (query, candidate) with a
relevance column `y` (0/1 or graded ints), e.g. person x time queries against candidate cells (D9 needed a
hand-written LightGBM lambdarank for this). No priors are built; bring them as features. Non-numeric features
become categoricals, scores come back in input row order, and every row is trained on unless
`sample_negatives=True` (positives + `top` by `prior_col` + `random` per query). `top1(..., query_col=,
cand_col=)` and `hit_at_k(..., query_col=)` take the same names. The session schema stays the default.

**Session level: conversion / abandonment from the first steps.** `next_action.sessions` and `next_action.funnel`
turn the step table into session rows without building candidates (not re-exported from `simengine.methods`;
import the module):

```python
from simengine.methods import next_action as na
S = na.sessions(steps, outcome_key="purchase", prefix_k=3, text_col="query")   # one row per session
S = S[S.reached_k]                         # honest "from the first 3 steps" task: sessions still open at k
F = na.funnel(steps, outcome_key="purchase")   # per step index: reached, continue / abandon / convert rates
```

`outcome_key` is the conversion action value, a per-step flag column, or a callable on the ordered session rows
returning a label or `(label, position)`. `sessions` columns: `prefix` (list), `prefix_text`, `prefix_len`,
`reached_k`, `n_steps`, `outcome` (`labels`, default purchase / abandon), `converted`, `outcome_step` and, with
`text_col`, the distinct prefix texts. The prefix is cut before the outcome step, so it never holds the outcome
action or anything after it. `n_steps` and `outcome_step` come from the full session: they are labels, not
features. A prefix shorter than k already reveals an early end, hence the `reached_k` filter. In `funnel` a session
leaves at its conversion step, and `continued[i] == reached[i + 1]`. Nothing measured yet; these are data
helpers (G4 rehearsal: ~15 lines of pandas written by hand).

## Theory-persona mixture calibration (`mixture.py`)

Port of edsl-test `personas/homo_silicus/calibrate.py` (`fit_weights`, `mixture` and the duplicate-row
warning). edsl-test has no LICENSE file; it has the same owner and is reused internally with this provenance note. A population is modelled as a mixture over a few theory types ("You only care about your own
pay-off.", ...). Each type's behaviour on calibration targets is measured once, then simplex weights (w >= 0,
sum 1) are fitted so the weighted average of the types matches the human targets. This is the "selection" step of
General Social Agents (Manning, Zhu & Horton) applied to the Homo Silicus types (Horton 2023, arXiv:2301.07543).
Validation runs on targets that were never fitted, in a different game format.

```python
import simengine.methods as M
games = {**M.charness_rabin_games(), "dictator": M.dictator_game()}
B = M.probe_types(llm, M.THEORY_TYPES, games, reps=10, prefix=M.HUMANIZING_PREFIX)   # type x target
fit = M.fit_mixture_weights(B, M.CR_HUMAN_LEFT, grid_step=0.02, val=M.ENGEL_DICTATOR)
fit.weights, fit.train_sse, fit.val_rmse, fit.unidentified_pairs, fit.weight_range
```

- **Fit.** `fit_mixture_weights(type_behavior, human_target, *, grid_step=0.02, val=None)` searches the whole
  simplex grid, as the source does, and minimises the SSE on the targets in `human_target.index`. The grid is
  vectorised and keeps the source's lexicographic order, so ties resolve the same way. Six types at 2% steps are
  3,478,761 points and take about 0.6 s. `method="slsqp"` (scipy) fits continuous weights for libraries too large
  for the grid; past `max_grid` points the grid raises and points you there.
- **Validation.** `val` is either a Series of held-out targets that are columns of `type_behavior`, or a
  `(val_behavior, val_target)` tuple from a separate probe. If a validation target is also a training target,
  the call raises. `val_rmse` is computed on those targets only.
- **Identifiability.** If two types differ by at most `ident_tol` (0.01) on every training target, the pair goes
  into `unidentified_pairs` and an `UnidentifiedWarning` is raised. The same warning fires when the
  (targets + 1) x types matrix [behaviour; 1] is rank deficient, for example when a type is a mix of others or
  there are more types than targets + 1. `weight_range` gives each type's min/max weight over the grid points
  tied with the best SSE. Without the warning you get the edsl-test v1 bug: in a dictator-only probe,
  `self_interested` and `efficient` both keep everything, and the grid silently gives their joint weight to
  whichever type comes first. That zeroes the modal human type (36% of humans give nothing).
- **Probes.** `probe_types(llm, type_texts, games, reps=10)` runs every (type, game) cell `reps` times
  through `llm.map`. The type text goes in as `system` (optionally after `HUMANIZING_PREFIX`) and the game text
  as `user`. Repetition r is passed as `sample=r` so caching backends give independent draws
  (`sample_kw=None` for a backend that rejects the keyword; then effective n may be 1). Probes are `ProbeGame`s
  (`choice_game`, `numeric_game`, or the ready `charness_rabin_games()` and `dictator_game()`). A choice
  reply must name exactly one option. A numeric reply must contain exactly one number in range. Anything else
  is invalid: it stays missing and is counted in `attrs["n_valid"]` / `attrs["n_invalid"]`, and a warning is
  raised. A cell with fewer than `min_valid` valid answers is NaN, and `fit_mixture_weights` refuses NaN cells
  in the targets it uses. `return_rows=True` also returns one row per answer with `persona_method`, `agent_id`,
  `seed`, persona parameters, `max_tokens`, `wording_id` and `valid`. The prompts end with a one-line answer
  instruction ("Answer with exactly one word: Left or Right.", "Answer with a single number only."), where the
  source relied on EDSL's question types.
- **Constants** (inlined with sources in the module docstring): `CR_GAMES` / `CR_HUMAN_LEFT` (Charness & Rabin
  2002, QJE 117(3), Table I; Berk29 is .31, Horton's text vector says .68, a typo), `ENGEL_DICTATOR` (Engel 2011
  meta-analysis: mean give 28.35%, 36.11% give 0, 16.74% split equally), `THEORY_TYPES` (six types: three from
  Horton Fig. 3, competitive and reciprocal from Charness & Rabin, selfish_but_fair from GSA), and
  `HORTON_GPT4O_WEIGHTS` (.53 / .37 / .10, Horton Sec. 2.2.1).

Measured in edsl-test (`personas/homo_silicus/weights.json`, `NOTES.md`; gpt-oss-120b, 10 reps per cell; not
re-measured here). The six-type fit on the CR menus gives self_interested .36, efficient .50, selfish_but_fair
.14 and 0 for the other three, with train SSE .089. `tests/test_methods_mixture.py` reproduces these weights,
the SSE and the validation RMSE exactly from the stored behaviour vectors. All six types were distinct on the
menus, so no identifiability warning was raised.

**When it does not work.**
- The held-out dictator validation is poor: RMSE **.315**. The mixture gives a mean of 6.7% against the human
  28.35%, and P(give 0) .86 against .36. A good train fit does not transfer to a new game format.
- `reciprocal` gets weight 0 for a structural reason. Unilateral menus contain no prior unfair act, so the
  probes cannot identify reciprocity, so no punisher type survives calibration. In the who-to-punish run
  with the calibrated group, both votes banned all punishment (`NOTES.md`, v2 results). The edsl-test fix is to add probes with prior unfair
  behaviour (ultimatum responder, a public-goods punishment stage) and to re-check the warning.
- All evidence comes from one model (gpt-oss-120b) and 10 reps per cell. Nothing was measured on Claude.

## Group transfer and out-of-fold predictions (`transfer.py`)

Task shape: survey answers exist for people in some groups (countries, regions, schools); predict the answers of
people in **new** groups from their demographics. There is no earlier wave, so persistence does not apply, and a
random-row CV leaks each group's level into training. Report group-held-out CV.

```python
import simengine.methods as M
feats = ["age", "education", "income", "gender"]                 # person features; df also has "country", "y"

# out-of-fold predictions, no country in both train and test of a fold (GroupKFold); KFold when group=None
oof_person = M.oof_predict(M.LightGBMBaseline(kind="continuous"), df, features=feats, target="y", group="country")
oof_gt = M.oof_predict(M.GroupTransfer(group_features=llm_country_prior), df,        # prior: DataFrame indexed by country
                       features=feats, target="y", group="country")

# final model on all training countries, predictions for the new ones
gt = M.GroupTransfer("country", group_features=llm_country_prior).fit(train[feats + ["country"]], train.y)
gt.predict(test[feats + ["country"]])          # person-level
gt.predict_groups(test[feats + ["country"]])   # stage A level per new country (rate for binary targets)
```

`oof_predict(model, df, *, features, target, group=None, n_splits=None, seed=0, method="predict", folds=None,
weight=None, n_repeats=1)`
- `model`: any object with `fit(X, y)` / `predict(X)` (sklearn estimator, simengine baseline, `GroupTransfer`) or a
  zero-argument factory. It is cloned (or deep-copied) per fold. If `fit` / `predict` accept `groups=`, the fold's
  group labels are passed, which is how `GroupTransfer` gets the groups.
- `group` given: `GroupKFold(n_splits)` (`n_splits` defaults to 5), deterministic as in the D2 code, so `seed` is unused. Fewer groups than
  `n_splits` raises `ValueError`. Without `group`: `KFold(shuffle=True, random_state=seed)`.
- `folds=`: precomputed folds, used **exactly** instead of `GroupKFold` / `KFold`. Accepted: a column name of `df`,
  a Series of fold ids aligned with `df.index` (by index), a 1-D array of fold ids (one per row), or an iterable of
  `(train_idx, test_idx)` positional pairs (what an sklearn `split` yields). Fold id `-1` = always trained on,
  never predicted (sklearn `PredefinedSplit`); NaN = row left out. `group=` still goes to models that take
  `groups=` (`GroupTransfer` keeps getting the item / country labels), and a `UserWarning` flags folds where a
  group is in both train and test. `n_splits` together with `folds` must equal the number of folds, else
  `ValueError`. `.attrs["fold_ids"]` lists the fold labels in the order of `.attrs["fold"]`.

  Why it matters: when features are rebuilt inside folds (SVD taste vectors, member mean / count, genre affinity;
  `item_holdout_features` below), the model must be scored with the **same** folds. A fresh `GroupKFold` puts rows
  whose features were built while the held-out items were still in the data into training, and dev comes out
  optimistic. In the D3 rehearsal (`uretim/prova/d3/SIMENGINE-EKSIK.md` #9, one run, no CI) the referee's ridge
  proxy dropped from 0.865 to 0.835 on dev once the OOF split matched the feature folds, i.e. about 0.03 of
  optimism.
  ```python
  res = M.item_holdout_features(ratings, user_col="user_id", item_col="movie_id", items=movies, svd_rank=16, cap=200)
  oof = M.oof_predict(M.GroupTransfer("movie_id", group_features=movies), res.dev,
                      features=res.features + ["movie_id"], target="residual", group="movie_id", folds=res.folds)
  ```
- `method="predict_proba"` keeps the positive-class column of a two-column output (sklearn classifiers). simengine
  binary baselines already return P(y=1) from `predict`.
- It returns a float Series aligned with `df.index`. Rows with a missing target (or group) are not trained on and
  stay NaN. `.attrs["fold"]` holds each row's fold number, -1 when the row was not predicted. The Series can go
  straight into an EvalFrame `pred_<method>` column.
- `n_repeats=`: one 5-fold split of ~300 people is noisy, so the split is redrawn `n_repeats` times and the
  predictions are averaged. Repeat `r` uses `seed + r`. Without `group` that is `KFold(shuffle=True,
  random_state=seed + r)`, so repeat 0 is the single run. With `group` it is a seeded shuffle of the group labels
  dealt into `n_splits` folds of near-equal group counts, still group-disjoint; `n_repeats=1` keeps the
  deterministic `GroupKFold`. `.attrs`: `repeats`, `repeat_pred` (rows x repeats), `repeat_fold` (rows x
  repeats, -1 = not predicted), `repeat_sd` (per-row SD over repeats); `fold` / `fold_ids` are repeat 0's. Score
  each `repeat_pred` column to see how much a metric moves between splits. `folds=` together with `n_repeats > 1`
  raises `ValueError`.
- `target=["y1", "y2", ...]`: returns a DataFrame with one column `oof_<target>` per target, each fitted
  separately (its own missing rows) with the same split settings, so it equals one call per target. Frame
  `.attrs`: `targets`, `repeats`, and the other keys as `{target: value}` dicts.
- Inputs are copied where pandas returns a read-only view (copy-on-write, the default in pandas 3), so it also
  runs with `pd.options.mode.copy_on_write = True`.
- `weight=`: per-row training weights, a column name of `df` or an array / Series aligned with `df` (e.g.
  `1/(se^2 + tau^2)` when the noise of the rows differs 30-fold, as in the D5 rehearsal). Each fold's training rows
  go to `fit(..., sample_weight=w)`. An sklearn `Pipeline` (nested too) gets `<last step>__sample_weight`, so
  `make_pipeline(StandardScaler(), Ridge())` works; the scaler itself stays unweighted (`TabularRidge` below
  weights it). If the model's `fit` has no `sample_weight` parameter (`GroupTransfer`, `RandomEffectsMean`, and
  for now `LogisticBaseline` / `LightGBMBaseline`), `TypeError` names the model. Missing, infinite or negative
  weights on rows with a target raise `ValueError`. Predictions are not weighted: report a weighted metric
  separately.
- Group-level scores of pooled group-held-out predictions are biased **negative** when there is no real group
  signal. This is pessimistic, not leakage: a held-out group's prediction leans on the other groups' mean, which
  is lower when the held-out groups' own mean is higher. With a few dozen groups, the group-level Spearman of the
  pooled OOF predictions is clearly below 0 under the null (a unit test checks the sign on synthetic data). So
  compare with a baseline cross-fitted the same way, such as `GroupTransfer(person_model=None, composition=())`,
  not with 0. The same holds for `evaluate(aggregate_by=...)`'s automatic baselines.

`GroupTransfer(group_col=None, *, composition=None, group_features=None, person_model="auto", kind="auto",
alphas=logspace(-1, 3, 20), clip=(0.02, 0.98), seed=0)`: a port of `iki_asama` in `kod/cv.py` **[prova-d2]**.
- **A, group level.** Each training group's mean target is regressed on group features with `StandardScaler` +
  `RidgeCV(alphas)`, one row per group. Binary targets use the logit of the rate, clipped to `clip`. Group
  features are the group's **composition**: means of the `composition` columns over the group's rows (default:
  all numeric person features), category shares for non-numeric columns (top 20), and the missing-value rate of
  numeric columns that have gaps. They are computed from inputs only, so a new group has them too. Any
  `group_features` columns (e.g. an LLM prior per country) are added. Missing values are filled with the
  training-group means. Use `composition=()` for group features only.
- **B, person deviation.** `y - observed group mean` is regressed on the person features (`group_col` excluded).
  `"auto"` picks LightGBM, as in the port, when it is installed and ridge otherwise. You can also pass `"ridge"`,
  `"lightgbm"`, `None` (group level only) or any regressor.
- **Prediction** is `A + B`. For binary targets it is `expit(A) + B`, clipped to [0, 1]. That is fine for ranking
  and AUC, but it is not a calibrated probability.
- Groups come from `groups=` in `fit` / `predict`, or from the `group_col` column of `X`. `group_features=` passed
  to `fit` / `predict` overrides the constructor one. Multiclass targets raise `ValueError`.
- `group_features` must be indexed by the group labels. If fewer than half the groups match exactly but more match
  as strings (int labels against a str-indexed prior, or the reverse), the string match is used and a
  `UserWarning` says so. If most groups still have no row, a `UserWarning` says that the prior has little or no
  effect, because unmatched groups get the training mean.
  ```python
  prior = llm_prior.set_index("country")          # index "1", "2", ... while df.country holds ints 1, 2, ...
  M.GroupTransfer("country", group_features=prior)  # warns "matched ... as strings" and still uses the prior
  ```
- **Composition with fold-built features.** The default `composition=None` averages *every* numeric person feature
  per group. If some columns are rebuilt per fold (SVD taste vectors, member statistics), their per-group means
  are only comparable across folds when the bases agree: SVD factors have arbitrary sign and rotation per fit, so
  align each fold to one reference basis (`item_holdout_features(align="full")` does it with orthogonal
  Procrustes) or leave those columns out with an explicit list, e.g.
  `GroupTransfer("movie_id", composition=["user_mean", "user_n_pct", "age"])`. Prefer relative counts
  (`user_n_pct`) over absolute ones (`user_n`): holding out popular items shifts absolute counts between folds and
  test. No warning is raised for this; it is a documented trap.
- Differences from the research code: composition is computed from the rows passed to `fit` / `predict`, while
  the port pooled train and test inputs. The values are the same when whole groups are passed. NaN fill uses
  training-group means instead of all-group means, and the binary output is clipped.

Measured **[prova-d2]** (43 training countries, 5-fold country-held-out CV, person-level Spearman on a 0-100 policy
score; numbers from `NOTLAR.md`, not re-measured here):

| arm | CV Spearman |
|---|---|
| person-only LightGBM (T3) | 0.197 |
| LightGBM + composition columns (T4) | 0.234 |
| two-stage, no LLM (T5) | 0.257 |
| two-stage + LLM country prior (BIZ2) | 0.295 |
| two-stage + extended LLM prior (BIZ3, 3rd config) | 0.311 |
| oracle country level + T3 (ceiling) | 0.344-0.346 |

In the D2 evidence cards (`SIMENGINE-EKSIK.md` #13), the submitted two-stage arm was `high confidence` against T5 on
the policy score (+0.054 Spearman), but **inconclusive** on the binary share target against LightGBM + composition
(+0.004 AUC; T4 AUC 0.671).

**When it should help.** The answer's group level varies across groups, and that level is predictable from
who lives in the group or from a group-level prior. It also needs enough training groups for a per-group ridge:
D2 fit it on about 34 groups per fold, so keep group features few.

**When it does not help.**
- If the person features already explain the outcome and group levels barely differ, stage A only adds noise.
- With fewer than about 10 training groups, stage A has too few rows for a per-group ridge (with fewer than 3
  groups it falls back to the mean of group means).
- An LLM group prior can leak a memorised published result. D2 ran a named- vs blind-prompt audit and found
  no sign of this, but check it on every new dataset (`simengine.evidence` memorization audits).
- The D2 numbers come from one dataset and one CV split and have no bootstrap CI. `tests/test_methods_transfer.py`
  only checks that, on synthetic groups with a real composition effect, `GroupTransfer` beats a global mean and
  a person-only ridge in group-held-out CV (and that the prior helps when it carries signal).

## Tabular regression and cell baselines (`tabular.py`)

The no-LLM rung for a plain table of categorical + numeric columns and a continuous target: the best ordinary
regression on the same columns, or the level of the row's cell (outcome x year). The G2 and D5 rehearsals each
hand-wrote these (`SIMENGINE-EKSIK.md` G2 #3, D5 #3 and #4). Every class has `fit(X, y[, sample_weight])` /
`predict(X)` on a DataFrame, so it goes straight into `oof_predict`. The classes are not re-exported from
`simengine.methods` yet; import them from `simengine.methods.tabular`.

```python
from simengine.methods import oof_predict
from simengine.methods.tabular import GroupStat, RandomEffectsMean, TabularGBDT, TabularRidge
feats = ["outcome", "year", "n", "pre"]                           # df also has "ate", "se", "w", "study"
oof_ridge = oof_predict(TabularRidge(cat=["outcome", "year"], num=["n", "pre"]), df, features=feats,
                        target="ate", group="study", weight="w")
oof_gbdt = oof_predict(TabularGBDT(cat=["outcome", "year"]), df, features=feats, target="ate", group="study")
oof_re = oof_predict(RandomEffectsMean(by=["outcome", "year"], se_col="se"), df,
                     features=["outcome", "year", "se"], target="ate", group="study")
oof_med = oof_predict(GroupStat(by=["outcome", "year"], stat="median"), df, features=["outcome", "year"],
                      target="ate", group="study")
```

- `TabularRidge(cat=None, num=None, *, alphas=logspace(-2, 4, 25), min_count=1)`: one-hot categoricals (NaN is
  its own category; categories seen fewer than `min_count` times are dropped; unseen ones get all-zero dummies)
  plus numerics filled with the training mean and standardised, then `RidgeCV` (efficient leave-one-out over
  `alphas`). With `sample_weight`, the fill / centring / scaling and the ridge are all weighted. `None` for
  `cat` / `num` = the remaining columns of `X` by dtype (numeric -> `num`), `()` = none. Put numeric codes such as
  a year in `cat` to get one dummy per value. `alpha_`, `coef_` (by encoded column name) and `intercept_` after fit.
  In all four classes an integral float and the same int are one category (2020.0 from a column with a NaN at fit
  is 2020 at predict).
- `TabularGBDT(cat=None, num=None, *, backend="auto", seed=0, **params)`: LightGBM with native categoricals when
  it is importable, else sklearn `HistGradientBoostingRegressor` with a `RuntimeWarning` (`backend_` says which ran;
  `backend="sklearn"` forces the fallback). Small-table defaults: 300 trees, lr 0.05, 15 leaves, 20 rows per leaf.
  `params` override them, using LightGBM names for LightGBM and sklearn names for the fallback.
- `RandomEffectsMean(by, se_col="se", *, tau2="dl", min_n=2)`: per cell of `by`, the DerSimonian-Laird random-effects
  mean, `sum(y/(se^2+tau2)) / sum(1/(se^2+tau2))`, with `tau2 = max(0, (Q-(k-1))/C)` estimated inside the cell
  (`tau2=<float>` fixes it). A cell with fewer than `min_n` rows, and any cell unseen at predict time, falls back to
  the next coarser prefix of `by` (`by[:-1]`, ...), then to the grand random-effects mean. `level(X)` returns the
  cell depth each prediction came from (0 = grand). `se_col` must be in `features=`. Rows with NaN se are dropped
  with a warning, and `se <= 0` raises. There is no `sample_weight`, because the weights are the precisions. The
  pooling function itself is `dersimonian_laird(y, se) -> (mu, tau2, se_mu)`. D5's B3 (one assumed `tau2 = 2` in
  every outcome x year cell) is `RandomEffectsMean(by=["outcome", "year"], tau2=2.0)`.
- `GroupStat(by, stat="median", *, weight=None, min_n=1)`: per-cell `"median"`, `"mean"` or `"wmean"`, with the
  same fallback. Weights come from `X[weight]`, or else from `sample_weight` (`oof_predict(weight=)`). When weights
  are present, `mean` / `wmean` are the weighted mean and `median` is the weighted median. `wmean` without weights
  raises.

**Measured.** These classes have not been run on rehearsal data yet. In D5, the hand-written level baselines
(`kod/ortak.py` `Duzey`) scored dev MAE between 1.524 and 1.549, and their paired CIs overlapped (D5 #10). `tests/test_methods_tabular.py` checks the DL `tau2` / mean against a hand calculation, the cell
fallback, that weighted fits move toward the heavy rows (TabularRidge, both GBDT backends, GroupStat, a
`make_pipeline` routed through `oof_predict`), and the `TypeError` for models without `sample_weight`.

**When it does not help.** With only a handful of rows per cell, a per-cell DL `tau2` is noisy (it is 0 whenever
`Q < k-1`). Raise `min_n` so that small cells use the coarser level. Median or mean can matter for MAE: in D5 the median
scored 1.547 and the weighted mean 1.524, but the paired CI of the difference included 0 (-0.023 [-0.073, +0.018]).
Try both, cross-fit them the same way, and do not read a gap that size as a winner.

## Item cold start (`coldstart.py`)

Task shape: members rated many items; some **new** items have no rating at all. Predict member x new-item ratings and
the item-level residual (the item's mean rating minus the mean of its raters' own means). Treat the item as the
group: `GroupTransfer(group=item)` learns the item level from the audience composition (who rated it) plus item
metadata, and a person model adds the member x item deviation. The dev table must imitate the test condition: the
dev items' ratings must not be in the member features.

```python
import simengine.methods as M
# ratings: user_id, movie_id, rating; movies: indexed by movie_id, 0/1 genre columns (+ year...); users: indexed by user_id
res = M.item_holdout_features(ratings, user_col="user_id", item_col="movie_id", items=movies,
                              genre_cols=genre_cols, users=users[["age", "female"]],
                              n_splits=5, seed=0, svd_rank=16, cap=200, min_item_ratings=100)
dev = res.dev                                   # held-out ratings + features built without their items + "fold"
feats = res.features + ["movie_id"]
gt = M.GroupTransfer("movie_id", group_features=movies[genre_cols])
oof = M.oof_predict(gt, dev, features=feats, target="residual", group="movie_id", folds=res.folds)
mf = M.oof_predict(M.MFBaseline("user_id", "movie_id"), dev, features=["user_id", "movie_id"], target="rating",
                   group="movie_id", folds=res.folds)          # ladder rung: member level + item prior
test_X = res.transform(test_pairs)               # test features from all ratings, same SVD basis as dev
final = gt.fit(dev[feats], dev.residual).predict(test_X[feats])
```

`item_holdout_features(ratings, *, user_col="user_id", item_col="item_id", rating_col="rating", folds=None,
n_splits=None, seed=0, svd_rank=16, cap=None, min_item_ratings=1, items=None, genre_cols=None, users=None,
align="full", align_scale=True, affinity_shrink=5.0, svd_prefix="svd_")`: a port of `build` / `user_stats` / `align` in
`kod/ozellik.py` **[prova-d3]**.
- Folds: items with at least `min_item_ratings` ratings are dealt with `default_rng(seed).permutation(n) %
  n_splits` over the sorted item ids (the port: 5 folds, items with >= 100 ratings). Or pass `folds=` as a Series
  item id -> fold id (not together with `n_splits`).
- For fold k, the fold's items' ratings are removed and, from the remaining ratings, it recomputes per member:
  `user_mean`, `user_n`, `user_sd` (NaN -> 1.0), `user_n_pct` (percentile rank of the count), the genre affinity
  (sum of residuals per genre / (genre count + `affinity_shrink`)) and an SVD taste vector of rank `svd_rank` from
  the member x item residual matrix (`randomized_svd`, `random_state=0`, factors `U*S`). The pair feature
  `affinity` is the member's mean affinity over the item's genres. `users` columns are joined as they are.
- `cap`: at most `cap` held-out ratings per item, sampled with a per-item seed (the port used 200 to mirror the test
  items' audience size). Members whose ratings were all on the fold's items have no fold statistics, so their
  rows are dropped, as in the port.
- `align="full"`: each fold's SVD basis is rotated onto the full-train basis by orthogonal Procrustes on the common
  members, so fold vectors are comparable with each other and with `transform` (test). The rotation is fitted on
  the full basis, which saw the held-out items. That `svd_rank x svd_rank` rotation is the only held-out
  information in a fold's features: norms and inner products of the fold's vectors do not depend on it (a test
  checks this). `align=None` skips it.
- `align_scale=True` (default, with `align="full"`): after the rotation each fold's SVD vectors are multiplied by
  one scalar, ||full vectors|| / ||fold vectors|| on the common members (`procrustes_align(scale=True)`, factor in
  `res.scales[k]`). A fold's SVD sees only about (n_splits - 1)/n_splits of the ratings, so its `U*S` vectors are
  about sqrt of that smaller; with rotation only, the sd of `svd_0..2` in `dev` was 0.89 of the `transform()` sd on
  synthetic data (300 members, 100 items, 5 folds), and a model fitted on `dev` overweights the SVD part at test
  time. Scaled, the ratio is about 0.98 on the same data. The scalar is the only other number fitted on the full
  basis. `align_scale=False` = the port's rotation only. Like the `user_n` vs `user_n_pct` note: features built
  on fewer ratings must be brought to the test-time scale.
  ```python
  res = item_holdout_features(ratings, n_splits=5, svd_rank=16)          # rotated + rescaled (default)
  res.scales                                                             # {fold: factor}, about 1.1 for 5 folds
  ```
- Returns `ItemHoldout`: `dev` (ids, rating, `residual` = rating - fold `user_mean`, `fold`, features), `folds`
  (= `dev["fold"]`, pass it to `oof_predict(folds=...)`), `item_folds`, `members` (full-train member features),
  `features` (suggested inputs; `user_n_pct` rather than the absolute `user_n`), `svd_cols`, `rotations`, `scales`, and
  `transform(pairs)` for test pairs.
- `procrustes_align(frame, ref, cols, *, scale=False)` is the alignment step on its own (rotation in
  `.attrs["rotation"]`; `scale=True` also rescales by the norm ratio, in `.attrs["scale"]`).

`MFBaseline(user_col="user_id", item_col="item_id", *, rank=16, reg=10.0, reg_bias=5.0, n_iter=10, item_prior=None,
seed=0)`: biased matrix factorisation `mu + b_u + b_i + p_u . q_i` fitted by alternating least squares, with
`fit(X, y)` / `predict(X)` on a frame of (member, item) columns, so it drops into `oof_predict`. `rank=0` = biases
only. The defaults are generic starting values, not tuned on any rehearsal. An unseen item gets `mu + b_u +
item_prior_` (the unweighted mean item bias, or `item_prior=`), an unseen member `mu + b_i`, both unseen `mu +
item_prior_`. In pure item cold start every dev / test item is unseen, so MF is a member-level rung: it shows how
much the item side (composition, metadata, LLM prior) adds, not a competitor for the new items. On warm pairs (a
random 80/20 split of the synthetic test data) it beats the member mean and the biases-only model.

**What was measured.** The D3 rehearsal (`uretim/prova/d3/SIMENGINE-EKSIK.md` #1 and #9, one run, no CI) built this
dev table by hand and used `GroupTransfer("movie_id")` as the backbone. Its only reported dev number for the recipe
is the ~0.03 optimism above (referee's ridge proxy, 0.865 -> 0.835) when the OOF split did not match the feature folds. This
port re-measured nothing. `tests/test_methods_coldstart.py` only checks, on synthetic ratings with a taste
structure (selection on taste, item quality tied to taste), that a held-out item's ratings do not change its fold's
features, that Procrustes alignment brings fold vectors to the full basis, and that `GroupTransfer(group=item)` on
these features beats the member-mean baseline (pair RMSE, item-level residual Spearman > 0.5).

**When it does not help.** When the audience of a new item is not known in advance (no raters at all), the
composition is empty and only item metadata remains (`composition=()`). Sparse members (few ratings outside the
fold) get noisy statistics; the SVD rank must stay below the number of items per fold.

## Single-text binary outcome and target encoding (`binary.py`)

For one text per row (first search query, a review, a message) and a 0/1 outcome. `TfidfAB` does not apply: it
scores the difference of two texts and has no intercept.

```python
import simengine.methods as M
tb = M.TfidfBinary("query", extra_cols=("n_steps",), encode_cols=("query",)).fit(train, train.y)
p = tb.predict(test)                                    # P(y = 1); predict_proba gives two columns
oof = M.oof_predict(M.TfidfBinary("query", C=1.0), train, features=["query"], target="y")
tr_enc, te_enc = M.target_encode(train, test, "query", "y", m=20)   # OOF for train rows, full train for test
```

- **`TfidfBinary(text_col, C=None, extra_cols=(), *, encode_cols=(), word_ngrams=(1, 2), char_ngrams=(3, 5), ...)`**:
  word + char_wb TF-IDF (sublinear tf; `min_df` (2, 3), 1 below 200 training rows), numeric extra columns
  mean-imputed and standardised, non-numeric ones one-hot (unseen -> zeros), `encode_cols` target-encoded inside
  `fit` (OOF over `te_folds`, smoothing `te_m`), then an L2 logistic regression with intercept. `C=None` picks `C_`
  from `C_grid` (0.03 ... 10) by inner stratified CV on the training rows (ROC AUC; ties -> the smaller C); the
  scores are in `cv_scores_`. `word_ngrams=None` / `char_ngrams=None` drops that vectoriser. A training set with one
  class predicts its rate. It has `fit(X, y)` / `predict(X)` and no hidden state across fits, so `oof_predict`
  deep-copies it per fold and every vectoriser, scaler and encoding is refit inside each fold.
- **`target_encode(train, test, col, y, *, m=20, oof=True, folds=5, seed=0)`**: value = (sum y + m * prior) /
  (n + m), prior = mean y of the rows the statistics come from, unseen categories -> prior. `col` may be a tuple
  (joint key). Training rows are encoded out of fold (a row never sees its own label; `folds` = number of shuffled
  folds or fold ids, e.g. the same column given to `oof_predict(folds=...)`), test rows from all training rows.
  `oof=False` is in-sample and leaks: on a 300-level noise key it gives AUC > 0.65 where OOF gives 0.5 (test).
  Returns `(train_encoding, test_encoding)` Series aligned with each frame.

**Measured [G4].** ShopCART session -> purchase from the first search query, 30,365 training sessions, 1,000 test
sessions: TF-IDF + query length + exact-query encoding, C chosen in train, test AUC 0.666 [0.617, 0.714]; the
exact-query encoding alone 0.541, query length alone 0.577. Trained on 500 sessions it fell to 0.579 and on
50-200 sessions to 0.51-0.54: the text learner needs thousands of labels.

## RCT condition effects and cross-block transfer (`rct.py`)

For randomised experiments (A/B, many-labs, multi-country trials): one row per person with the condition, the
outcome and optionally the randomisation block (country, site, week) and segment columns.

```python
eff = M.rct_effects(df, "condition", "Control", "policy", block="country", by="ideology")
tr = M.rct_transfer(df, "condition", "Control", "policy", block="country", test_blocks=test_countries)
tr.attrs["spearman"], tr[["condition", "pred", "truth"]]
```

- **`rct_effects(df, cond, control, y, *, block=None, by=None, weights=None)`**: within each block (x segment)
  mean(y | condition) - mean(y | control), averaged over the blocks that have both arms; `weights` `None` / `"size"`
  (people in the two arms), `"equal"` or a mapping block -> weight. SE = sqrt(sum w_b^2 (s2_c/n_c + s2_0/n_0)) /
  sum w_b with a normal 95% CI; a one-person cell borrows the arm's pooled within-block variance (SE NaN when there
  is none). Block-wise differencing removes block offsets even when arm shares differ across blocks, where the
  pooled difference (`block=None`) is confounded. Rows: (segment..., condition) with `effect`, `se`, `ci_low`,
  `ci_high`, `mean_cond`, `mean_control`, `n_blocks`, `n_cond`, `n_control`.
- **`rct_transfer(df, cond, control, y, *, block, test_blocks, train_blocks=None, by=None, weights=None,
  per_block=False)`**: the effects estimated on the training blocks (default: all others) are the prediction for
  the test blocks, next to the test-block truth (`pred`, `pred_se`, `truth`, `truth_se`, `n_blocks_train`,
  `n_blocks_test`; `pred` NaN for a condition never seen with its control in training). `per_block=True` gives one
  truth row per test block. `.attrs`: `spearman` (per-block mean in that mode, the values in the dict
  `spearman_by_block`), `mae`, `train_blocks`, `test_blocks`.

**Measured [G1].** Climate many-labs, 11 messages vs control, policy support, 32 training countries -> 31 test
countries: transfer Spearman over the conditions 0.845 [0.43, 1.00] against a sampling-noise ceiling of 0.908;
equal country weights 0.88-0.92. One training country alone: median 0.28, quartiles 0.06-0.48, so a single-market
history is a weak baseline. Haiku's direct forecast scored 0.073, persona panels -0.127 / 0.227, and adding the LLM
to the transfer baseline made it worse (-0.300 [-0.85, +0.05]).

## Uplift / CATE for randomised experiments (`uplift.py`) [prova-d11]

One row per customer: a 0/1 treatment, an outcome (0/1 or numeric), features, optionally a segment. Test for
heterogeneity before building any per-customer model.

```python
het = M.heterogeneity_test(df, "sms", "buy", ["age", "gender", "points"], seg="segment")
het.p, het.advice, het.q                       # p > 0.2: do not spend time on per-customer uplift
eff = M.segment_uplift(df, "sms", "buy", "segment")
eb = M.shrink_effects(eff.uplift, eff.se)      # .shrunk, .B, .attrs["tau2"]
dev = M.split_half_effects(df, "sms", "buy", "segment", R=50, seed=0)
M.summarize_split_half(dev)                    # mae, mse_d (noise-corrected), spearman per method
score = M.TLearner(model="logistic").fit(train, "sms", "buy", feats).predict(test)
M.uplift_at_k(test.buy, test.sms, score, 0.3), M.qini(test.buy, test.sms, score)
```

- **`heterogeneity_test(df, treat, y, features, *, seg=None, scale="logit", bins=5)`**: joint test of every
  treatment x feature interaction (numerics standardised, median-imputed with a missing flag; categories one-hot).
  `scale="logit"`: likelihood-ratio test, logistic, 0/1 outcome only. `scale="additive"`: linear model, HC1 Wald
  test, i.e. the risk-difference scale uplift@k acts on. A constant risk difference over a varying base rate is an
  interaction on the logit scale in large samples, and vice versa, so pick the scale of the decision. `q`: Cochran Q
  of the raw group effects over `seg` and over each feature (numerics in `bins` quantile bins), with `I2`. The
  rule in the docstring and `advice`: **p > 0.2: do not spend time on per-customer uplift**; report the constant
  effect with random targeting as the baseline. p < 0.05: a model may pay; check uplift@k against random on held-out rows.
- **`segment_uplift(df, treat, y, seg=None)`**: raw treated - control difference per segment, Neyman SE
  sqrt(s1^2/n1 + s0^2/n0), normal 95% CI, indexed by segment. A 0/1 outcome uses (k + 1) / (n + 2) proportions in
  the variance, so a 6/6 cell does not get SE 0. `.attrs["ate"]`, `["ate_se"]`.
- **`shrink_effects(est, se, X=None, *, tau2=None)`**: Fay-Herriot: est ~ N(X beta, se^2 + tau^2), tau^2 by REML
  (the moment estimator inflated with many small segments in D11), beta by GLS; `X=None` shrinks to the
  precision-weighted mean. Columns `est`, `se`, `prior`, `B` (weight on the prior), `shrunk`, `post_se` (ignores
  the uncertainty in beta and tau^2); a row without an estimate gets the prior.
- **`split_half_effects(df, treat, y, seg, R=100, seed=0, *, methods=None, unit=None, boot=True)`**: per rep the
  units are split in two, stratified by treatment first, and only then bootstrapped inside each half. The D11 hand
  version bootstrapped before splitting, so one customer sat in both halves. `methods`: name ->
  callable(half A) -> effects by segment; default `raw`, `pooled` (A's ATE everywhere), `shrunk`. Truth = half B's
  `segment_uplift`, `truth_var` = its SE^2, so `summarize_split_half` reports `mse_d` = mean((pred - truth)^2 -
  truth_var), the MSE against the true effect. `unit=` keeps a customer's rows together.
- **`TLearner(model="logistic" | "gbdt" | estimator | factory, categorical=None, n_jobs=None)`** and
  **`SLearner`** (default `"gbdt"`): `.fit(df, treat, y, features=None)`, `.predict(df)` = f1(x) - f0(x),
  `.predict_arms(df)`. GBDT = LightGBM with the bench parameters, one thread below 200k rows; without LightGBM it
  falls back to sklearn HistGradientBoosting with a warning. A regressor (no `predict_proba`) gives E[y | x, t].
- **Metrics** `qini` (= `qini_coefficient`), `auuc`, `uplift_at_k(y, t, score, k=0.3)`, `qini_curve`,
  `uplift_curve`: moved here from `bench/tasks/uplift.py`, which imports them, so its numbers are unchanged (pinned
  in `tests/test_bench_uplift.py`). Definitions in `docs/modules/bench/uplift.md`.

Simulated check (`tests/test_methods_uplift.py` setting, 100 datasets of 8,000 customers, half for fitting the test
and learner, half for scoring): with an effect of 0.15 tanh(2 x1), the test rejected at 0.05 in 100/100 on both
scales, and the logistic T-learner's uplift@30 was 0.139 against 0.004 for a random score. With a constant +0.06
risk difference it rejected in 4/100 on both scales, p > 0.2 in 84-87/100, and the T-learner's uplift@30 (0.059)
was no better than random (0.062). One logit + additive test pair on 4,000 rows took 0.02 s.

**Measured [prova-d11].** SMS pilot -> rollout, 180,075 rollout customers, 26 segments, segment uplift MAE in
percentage points. Pilot ATE to every segment 1.538 tied the strongest baseline, empirical-Bayes shrinkage to the
mean (1.526). The preregistered Fay-Herriot with covariates scored 1.924 and the raw pilot segment difference 2.451.
Pilot segment order did not hold in the rollout (Spearman <= 0 for every arm). The hand T-learner run (100 reps x 2
LightGBM) did not finish in 14 minutes, and the LR test said the effect was not heterogeneous.

## Not ported / gaps

- The persona panel (LLM personas voting) and PAARS-style agents are left to the `sim` / `persona` modules. Their
  measured A/B numbers are 0.685 (Opus persona panel) and a 1.41x prior.
- BTYD and ReCANet for repeat purchase, and the person x category LightGBM for next basket are not ported
  (hierarchical Bayes / mixed logit for conjoint: `conjoint_hb.py`).
- The research kits' scale conversions (deciles, point units) are not ported in `TwinStack`: targets are used in
  their raw numeric unit.
- PPI++ is in `simengine.evidence`, not here.


## Demand forecasting (`forecast.py`) [prova-d4]

Long panel: one row per (series, time) with `id`, `time`, `y` (column names are keyword arguments). The
*origin* is the last time with known `y`; horizon `h` is the h-th period after it. Missing cells count as 0
(`fill_value=`). Rows after the origin may carry known-in-advance exogenous columns (price, calendar) with
`y` NaN. Every forecaster returns `[id, origin, h, time, pred]`.

```python
import simengine.methods as M

# rule ladder at one origin (week 269), 4 weeks ahead
M.persistence_forecast(panel, 269, 4)
M.seasonal_naive(panel, 269, 4, season=52)
M.moving_mean(panel, 269, 4, window=4)
M.seasonal_ratio(panel, 269, 4, window=4, season=52)          # T4: last-4 mean x last year's same-period ratio
M.seasonal_ratio(panel, 269, 4, group_col="store_dept", around=0, smooth=0)   # T4b: ratio pooled per group
fr, alpha, scores = M.ses(panel, 269, 4, return_alpha=True)  # alpha tuned on earlier origins, targets <= 269

# features: only y at times <= origin; exog read at the target time; static -> category codes
X = M.make_features(panel, 269, lags=(1, 2, 3, 4), windows=(4, 8, 13, 26), horizons=4, season=52,
                    exog=["price", "snap"], static=["store", "dept"])

# direct strategy: one model per horizon (lightgbm lazily, else sklearn HistGradientBoosting with a warning)
gbm = M.DirectForecaster("lightgbm", horizons=4, season=52, exog=["price"], static=["store", "dept"],
                         n_train_origins=52, clip_min=0.0)

# rolling-origin backtest: each callee sees y masked after its origin; truth from the full panel
bt = M.backtest(gbm, panel, origins=range(229, 270, 8), horizon=4)
base = M.backtest(M.moving_mean, panel, origins=range(229, 270, 8), horizon=4, window=4)
M.summarize_backtest(bt, by="origin")          # wmape, bias, n per origin

# metric-optimal point: Poisson median for L1 metrics, only where y_hat < 3 (rehearsal B5)
bt["pred_l1"] = M.point_for_metric(bt["pred"], metric="wmape", dist="poisson", threshold=3)

# LLM prior as a feature, with a blind arm and a shuffled placebo (one llm.map batch)
prior = M.llm_prior_feature(llm, ["FOODS", "HOBBIES", "HOUSEHOLD"],
                            "Weekly unit-sales multiplier for {key} in a Super Bowl week (1.0 = no effect)?",
                            blind_question="Weekly unit-sales multiplier for {key} in a Super Bowl week?",
                            thinking="off")
# -> key, value (named), value_blind ("group A", ...), value_shuffled (values permuted across keys)
```

What was measured (D4, one run, [prova-d4] `SONUC-MAIN.md`, series-level WMAPE on the sealed test weeks):
persistence 0.418, seasonal naive (52) 0.600, last-4 mean 0.379, SES 0.371, LightGBM on sales history only
0.354, LightGBM + price + calendar 0.348, + Poisson median (ŷ < 3) 0.347; the hidden LightGBM reference was
0.357. The price/calendar gain was small per series (0.0053) and much larger at store x dept x week level
(0.0141).

Backtest details: truth of an (id, time) cell missing inside the panel is 0 (`fill_value=0.0`, the same rule
as the forecasters), so a sparse panel without its zero-sales rows scores like the dense one; targets beyond the
last panel time stay NaN. `fill_value=np.nan` restores the old drop. Integer or regular-date times are reindexed
to a complete grid, so a week with no rows at all does not shift h. A `DirectForecaster` whose `horizons` do not
cover `horizon=` raises a `UserWarning` (the missing h are absent from the result).

```python
bt = M.backtest(M.moving_mean, sparse_panel, origins=[30, 33], horizon=4)            # empty cells = 0
bt_nan = M.backtest(M.moving_mean, sparse_panel, origins=[30, 33], horizon=4, fill_value=float("nan"))
```

Cautions:
- **One origin is misleading.** On the rule baselines the D4 team saw WMAPE swing between about 0.38 and 0.45
  across origins 229-269 (`kod/ek.py`, one run), and the dev origin was among the easiest. Use `backtest`
  with several origins and report per-origin numbers.
- **Median vs mean.** `point_for_metric` lowers an L1 metric and biases totals down. In D4 the median step
  improved series WMAPE by 0.0014 but worsened store x dept WMAPE by 0.0036 and moved bias from -3.3% to
  -4.0%. Report bias (`summarize_backtest`) separately and keep the untransformed arm if aggregate levels
  are scored. For MSE it returns the mean unchanged.
- **LLM prior.** In D4 the six-call category x event multiplier added nothing: named and blind arms both
  0.359 on dev, difference to the no-LLM arm +0.0006 [-0.0007, +0.0017] (`NOTLAR.md`). Keep the blind and
  shuffled columns; a prior only counts when `value` beats both.
- **`seasonal_ratio`** (the D4 team's T4 / T4b rungs): mean of the last `window` periods x
  `(ly + smooth) / (base + smooth)` clipped to `cap=(0.5, 2.0)`, where `base` is the same window one `season` back
  and `ly` the target period one season back +- `around` periods (never past the origin). Defaults are T4
  (`window=4, season=52, around=1, smooth=1`); T4b is `group_col=<store x dept>, around=0, smooth=0` (ratio of the
  group's sums). A zero denominator gives ratio 1; NaN when the history is shorter than season + window. Output
  format as `moving_mean`. D4 dev WMAPE (origin 269, one run, the team's code): T4 0.488 (per-series ratio is
  noisy), T4b 0.393, against last-4 mean 0.387 and persistence 0.434; neither beat the plain last-4 mean.
- `seasonal_naive` was the worst rung on M5 weekly data (0.600). It wins only where the season really repeats
  (the synthetic test).

### Growth and saturating targets, origin-time features [prova-d7]

The D7 task was adoption per 100 people for 361 country x technology series from a fixed 1997 origin, 1-5
years ahead. A level-target tree cannot predict past the levels it was trained on (dev 1.776 MAE, behind a
constant-growth rule at 1.731). A log-growth target fixed dev, but at the test origin it took the highest
mobile level (42) to 150 in 2002: dev targets were all low, so dev could not show it. The team switched to a
logit target with a ceiling of 100 (test MAE 3.73 -> 3.11, one run, `SONUC-MAIN.md`).

```python
from simengine.methods import forecast as F

# 'country, year, variable, value' -> one series per (country, target); every variable becomes a column
panel = F.panel_from_long(long_df, id="country", time="year", var="variable", value="value",
                          targets=["cell", "internet"])
# columns: series ("<country>:<target>"), country, variable, year, y, cell, internet, pc, ...

fc = F.DirectForecaster("lightgbm", horizons=5, id_col="series", time_col="year", static=["variable"],
                        target="logit", upper=100.0,                 # predictions stay in (0, 100)
                        origin_features=["cell", "internet", "pc"],  # read at the origin, not the target year
                        check_range=0.25)                            # warn on extrapolation at predict()
fr = fc.fit(panel, 1997).predict(panel, 1997)
fc.range_check_            # lo, hi, pred_min, pred_max, n_above, n_below, share_out, ok
F.extrapolation_check(fr, dev_backtest["truth"], margin=0.25)        # or against the dev target range
```

- `target="level"` (default, unchanged): y[T+h]. `"log_ratio"`: log((y[T+h] + log_offset) / (y[T] +
  log_offset)), mapped back as (y[T] + c) * exp(z) - c; rows with a non-positive ratio are dropped and a series
  with a non-positive base stays flat (`log_offset=1` for counts with zeros). `"logit"`: the change
  logit(y[T+h] / upper) - logit(y[T] / upper) (shares clipped to `[logit_eps, 1 - logit_eps]`), mapped back as
  upper * sigmoid(logit(y[T] / upper) + z), so no prediction can pass `upper`. Observed y above `upper` raises
  a warning (D7: mobile subscriptions can exceed 100 per 100; pick the ceiling from origin-time knowledge).
  Non-level targets train with the "regression" objective (Tweedie needs a non-negative target).
- `origin_features` columns are read at the origin T as `{c}_at_origin` (another technology's level, an
  income group), unlike `exog`, which is read at the target time and so needs known-in-advance values. In
  `fit` / `predict` their values after the origin are masked, so they cannot leak (a column also listed in
  `exog` is left unmasked). `make_features(..., origin_features=)` gives the same columns.
- `check_range=` / `extrapolation_check`: predictions above hi + margin x (hi - lo) or below lo - margin x
  (hi - lo) of the training (or dev) target range raise a `UserWarning`. It only flags; look at the
  out-of-range rows and judge plausibility (the playbook trap: compare the test-origin prediction
  distribution with the dev target range).
- `panel_from_long` is one pivot (D7's first row-by-row `.loc` version took 2+ minutes). Duplicate
  (id, time, variable) rows raise; series whose target is never observed are dropped (`dropna_series`).
  Missing cells inside the panel still count as `fill_value` (0.0) in the lag / rolling features; pass
  `fill_value=float("nan")` when missing means unknown rather than zero. The growth / logit targets take y[T]
  as the last value observed at or before T, so a gap at the origin is not a fall to 0 (before this, a
  `log_ratio` series with no row at the origin was forecast as 0).


## Repeated surveys: next-wave distributions (`waves.py`) [prova-d6]

For "past survey waves -> next wave's segment x question answer distribution", often across a regime switch
(the president's party changed). Not exported from `simengine.methods`; import from `simengine.methods.waves`.
Input is long: one row per respondent x question with wave, segment, item and answer columns (optional survey
weight, regime and subgroup columns). A wide frame works with `item=None, value=[<question columns>]`.
`categories` is one list or `{item: list}`; other answers (NaN, skipped) are ignored. Every predictor uses only
waves strictly before `target` (default: the wave after the last one) and returns the frame
`evaluate(task="distribution")` takes: `unit` ("seg|item"), `truth` (the target wave's distribution when it is in
the data, else NaN), `pred_<method>`, `segment`, `item`, `group` (= item), `wave`, `n` (valid answers in the
target cell) and `n_train_<method>`.

```python
from simengine.methods.waves import (wave_persistence, pooled_shrunk, same_regime, subgroup_mixture,
                                     transition_backtest, merge_frames)
cols = dict(wave="year", segment="seg", item="q", value="answer", categories=CATS, weight="w")
PRES = {y: "D" if 2009 <= y <= 2016 or 2021 <= y <= 2024 else "R" for y in range(2006, 2026)}  # wave -> regime

t0 = wave_persistence(df, **cols, target=2025)                       # copy 2024 (k=3: pool the last 3, n-weighted)
t2 = pooled_shrunk(df, **cols, target=2025, k=4, m=20)               # last 4 waves, shrunk toward the population
t4b = same_regime(df, **cols, target=2025, regime=PRES, k=4)         # last 4 waves with an R president
mx = subgroup_mixture(df, **cols, target=2025, subgroup="pid3", k=1,
                      swap={"Rep": "Dem", "Dem": "Rep"}, swap_items=["approval_pres", "economy_retro"])
frame = merge_frames([t0, t2, t4b, mx])                              # one row per cell, all pred_ columns

bt = transition_backtest(df, waves=[2009, 2017, 2021], **cols, methods={
    "t0": (wave_persistence, {}), "t2": (pooled_shrunk, {"k": 4}),
    "t4b": (same_regime, {"regime": PRES, "k": 4}),
    "mx": (subgroup_mixture, {"subgroup": "pid3", "swap": {"Rep": "Dem", "Dem": "Rep"},
                              "swap_items": ["approval_pres", "economy_retro"]})})
bt.by_wave      # wave x method mean 1-TVD
bt.summary      # mean, sd, min over waves, waves led, mean rank
```

- `pooled_shrunk`: p = (n_s * own + m * prior) / (n_s + m), n_s = valid answers of the segment in the window
  (Kish effective size with weights); `prior="population"` (the window's pooled item distribution), `"uniform"`
  or `{item: distribution}`. A segment with no answers in the window gets the prior (`n_train` 0).
- `same_regime`: `regime` is a `{wave: label}` mapping or a column constant within each wave; the target's label
  comes from `target_regime=`, the mapping or the target wave's rows. With no earlier wave of that regime it
  falls back to the last k waves and warns. `frame.attrs["waves"]` lists the waves pooled.
- `subgroup_mixture`: segment s, item i -> sum_g w[s][g] P(answer | g, i). The conditionals are pooled over the
  last k waves at population level, or per segment x subgroup shrunk toward it with `m=`. Weights default to each
  segment's subgroup composition in the last training wave. `weights="target"` uses the subgroup labels of the target
  wave's rows, answered or not (only when they are known before the answers, e.g. from the sample frame). A dict sets
  fixed weights, for every segment or per segment. `swap` applies `role_swap` (out[g] = cond[mapping[g]]) to
  `swap_items`, which `swap` requires: swapping a stable item (party ID itself, religion) hands each party the
  other's answers.
  `mixture_reweight(cond, weights)` and `role_swap(cond, mapping)` are the pure pieces.
- `transition_backtest`: `methods` maps a name to `(predictor, kwargs)` (the column arguments, `target` and
  `method` are filled in) or to `callable(df, target) -> frame`. Every backtest wave must be in the data. `frame` is
  the long frame over all waves. For one `evaluate` over several years, make `unit` unique
  (prefix the wave). `block_col="wave"` puts the year-to-year spread into the CIs, but with three switch years
  the card says the CI is not reliable across blocks (< 5); report `bt.by_wave` year by year instead.

What was measured (D6, one run, [prova-d6] `SONUC-MAIN.md`, 2025 test, cell TVD, lower is better): copying 2024
0.1315 (presidential approval alone 0.295); a backward-looking rule ladder 0.0892; party-ID-mediated 0.0872; the
hidden role-mirror baseline (in-party / out-party swap, the `role_swap` family) 0.0839; the team's preregistered
arm 0.0788 and its post-referee arm 0.0800. The gain came from the two regime-dependent questions (approval,
economy); the other five were close to the role-mirror baseline. On dev, the same-regime 4-wave pool (T4b) beat the
team's preregistered method, 0.0990 vs 0.1007 TVD: put `same_regime` on the ladder whenever the target crosses
a regime switch.

Cautions:
- **One dev year hides regime uncertainty.** A cluster bootstrap over segments in a single transition year does
  not cover year-to-year spread. Score on several past switch years (`transition_backtest`) and report each year.
- **Post-hoc selection.** In D6 the arm picked after the referee (best of ~40 configurations on two dev years)
  lost to the preregistered arm on test. Seal the preregistered arm separately and report both.
