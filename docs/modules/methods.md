# simengine.methods

This module holds the non-LLM baselines that every simulation result is reported against, and the hybrids
that combine an LLM prediction with a model learned from data.

| component | what it is | verdict from the prep nights |
|---|---|---|
| `ConstantBaseline`, `MajorityBaseline`, `ItemMarginal`, `SegmentRate` | rate / median / class shares, per item or shrunk per segment | always report them; they are hard to beat on accuracy |
| `uniform_distribution`, `weighted_distribution`, `overall_copy`, `segment_distributions` | distribution baselines for group-level opinion | `overall_copy` wins against the uniform distribution and against neighbour transfer |
| `PersistenceBaseline`, `persistence` | "tomorrow = yesterday" from the unit's own history | strongest single signal wherever history exists |
| `basket_tensor`, `popularity`, `personal_frequency`, `decayed_frequency`, `tifu_knn` | next-basket / repeat-purchase scorers | decayed frequency beats P-TopFreq; TIFU-KNN adds nothing |
| `LogisticBaseline`, `LightGBMBaseline`, `KNNBaseline`, `TextKNNRegressor` | supervised tabular / text baselines | GBDT ties persistence when history is available; text kNN on effect sizes is weak |
| `LengthAB`, `TfidfAB`, `TfidfWordAB`, `LLMDirectAB`, `LogitAverageAB`, `ABStack` | A/B winner of two texts | stack (TF-IDF + LLM) is the best measured arm, at **0.817** on Upworthy |
| `EarlyTestBayes`, `simulate_saving_calibrated`, ... | LLM prior + small early test, normal-normal Bayes on the log-odds ratio | **about 2.3x** impression saving at 100 impressions/arm, no gain at 800+ |
| `ShrunkDeviation`, `ridge_to_prior`, `select_lambda`, `LLMFeatureGBDT`, `DevStack`, `BinaryCalibrator` | ways of letting an LLM prediction into a data model, tuned on dev | the LLM term mostly adds ~0 once a strong data baseline exists; tuning keeps it from hurting |
| `TwinStack`, `pooled_logistic_oof`, `column_correlation`, ... | person-level digital twin: matrix completion + kNN + optional LLM column, ridge-stacked per column | Twin-2K-500 fair LLM-free baseline **0.342** corr; LLM adds +0.006 |
| `MNL`, `evaluate_mnl`, `null_scores`, `person_holdout` | conditional multinomial logit for conjoint / discrete choice | 0.509 new-person accuracy on camera conjoint (HB 0.519, HB not ported) |
| `Markov2Baseline`, `NextActionRanker`, `hit_at_k`, `top1`, ... | next-action ranking: Markov-2 with backoff, LightGBM lambdarank | 26.94 vs **31.60** exact next action on OPeRA |
| `fit_mixture_weights`, `probe_types`, `identifiability`, `mixture` | theory-persona mixture calibration (Homo Silicus types, GSA selection): simplex grid fit, identifiability warning, held-out validation | reproduces the edsl-test fit (.36 / .50 / .14, train SSE .089); held-out dictator RMSE **.315**, i.e. the calibrated mixture does not transfer |
| `GroupTransfer`, `oof_predict` | group transfer (predict people in groups never seen in training): ridge group level on group composition (+ optional LLM group prior) + person deviation model; out-of-fold predictions with group-held-out CV | D2 rehearsal, country-held-out CV person Spearman: person-only GBDT 0.197, two-stage 0.257, two-stage + LLM prior **0.295** (oracle level 0.344) |

**Provenance.** Everything is ported from the TALP prep research branches under
`talp-simulation-hackathon-2026/uretim/<branch>/`. The algorithms and defaults are unchanged. Shorthand used below:

- **[deney]** = `uretim/deney/RAPOR.md` (A/B, early test, SocSci210, conjoint, risky choice)
- **[intent]** = `uretim/intent/RAPOR.md` (purchase intent, next basket)
- **[persona]** = `uretim/persona/RAPOR.md` (Twin-2K-500)
- **[web]** = `uretim/web/RAPOR.md` (OPeRA next action)
- **[makro]** = `uretim/makro/RAPOR.md`, **[llmabm]** = `uretim/makro/llmabm/RAPOR.md`
- **[dongu]** = `uretim/dongu/GUNLUK.md` (OpinionQA group loop)
- **[prova-d2]** = `uretim/prova/d2/NOTLAR.md` and `SIMENGINE-EKSIK.md` (blind rehearsal: new-country survey answers; code `kod/cv.py`, `cv_biz3.py`, `tahmin.py`)

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

# --- conjoint
m = M.MNL(["price", "quality"], scale={"price": 100}).fit(train_rows)
m.accuracy(test_rows), M.null_scores(test_rows)

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
- **Distributions.** For OpinionQA subgroups (setting B, Overall known), copying Overall to every group gives
  dev WD 0.0530 [0.0509, 0.0554]. The uniform distribution gives 0.1976 and "sibling" transfer gives 0.0764
  **[dongu]**. Adding neighbour-group deviations on top was significantly worse (+0.0014 [+0.0006, +0.0021]
  WD). For the makro C1 segment task (W1): the overall marginal scores 0.206 and the marginal of the other three
  segments 0.274. The same party's other age group scores 0.040, against a 37-person survey ceiling of 0.050
  **[makro]**. Use `segment_distributions(mode=...)` for these.
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
swapped (position bias). `coverage_` is the share of pairs with a valid answer.

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

## Hybrids (`hybrid.py`)

These are the three ways an LLM prediction entered a data model during the prep nights, plus the generic
dev stack. PPI++ lives in `simengine.evidence`.

- **`ShrunkDeviation`**: `pred = base + alpha * (llm - ref)`. The level comes from data, the deviation from the
  LLM, and `alpha` is chosen on dev from 0..1. `ref` is the LLM's own group mean (makro `llm_kal`) or an explicit
  reference. For distributions, the result is clipped and renormalised. Measured on makro M2 price elasticity:
  "level from category, deviation from LLM" + data scored test WAPE 0.478, against 0.477 for category + data
  **[llmabm, M2 table]**, i.e. no gain. For OpinionQA groups, neighbour deviations at alpha = 1 were worse than
  copying Overall **[dongu]**. Expect `alpha_` near 0 unless the LLM has item-specific signal.
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
  **`BinaryCalibrator`** uses Platt scaling, or isotonic regression from 1000 dev rows (intent kit `kalibre`). No
  separate accuracy is claimed for these two; they are the plumbing that stops a weak LLM arm from hurting.

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
The port adds the analytic gradient; the optimum is the same. `evaluate_mnl` reproduces the source evaluation:
20% of persons held out, argmax accuracy, mean log-likelihood per task, against the null (equal share) and an
ASC-only model.

Measured on the Sawtooth camera conjoint **[deney 1.5]**: for a new person, holdout accuracy is 0.210 with equal
shares, **0.509 with MNL** and 0.519 with hierarchical Bayes. For the same person on a new task, MNL scores 0.482
against **0.715** for HB. Person-level tastes matter there and MNL ignores them. **HB is not ported.** LLM arms did
not beat MNL or HB on these sets.

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

`oof_predict(model, df, *, features, target, group=None, n_splits=5, seed=0, method="predict")`
- `model`: any object with `fit(X, y)` / `predict(X)` (sklearn estimator, simengine baseline, `GroupTransfer`) or a
  zero-argument factory. It is cloned (or deep-copied) per fold. If `fit` / `predict` accept `groups=`, the fold's
  group labels are passed, which is how `GroupTransfer` gets the groups.
- `group` given: `GroupKFold(n_splits)`, deterministic as in the D2 code, so `seed` is unused. Fewer groups than
  `n_splits` raises `ValueError`. Without `group`: `KFold(shuffle=True, random_state=seed)`.
- `method="predict_proba"` keeps the positive-class column of a two-column output (sklearn classifiers). simengine
  binary baselines already return P(y=1) from `predict`.
- It returns a float Series aligned with `df.index`. Rows with a missing target (or group) are not trained on and
  stay NaN. `.attrs["fold"]` holds each row's fold number, -1 when the row was not predicted. The Series can go
  straight into an EvalFrame `pred_<method>` column.
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

## Not ported / gaps

- The persona panel (LLM personas voting) and PAARS-style agents are left to the `sim` / `persona` modules. Their
  measured A/B numbers are 0.685 (Opus persona panel) and a 1.41x prior.
- Hierarchical Bayes / mixed logit for conjoint, BTYD and ReCANet for repeat purchase, and the person x category
  LightGBM for next basket are not ported.
- The research kits' scale conversions (deciles, point units) are not ported in `TwinStack`: targets are used in
  their raw numeric unit.
- PPI++ is in `simengine.evidence`, not here.
