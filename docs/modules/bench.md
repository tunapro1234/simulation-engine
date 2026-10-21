# simengine.bench

`simengine.bench` holds seven benchmark tasks built from public human-behaviour datasets, plus three economic-game
tasks built on `simengine.games` (`econ_games_b01`, `ertan_wtp` and the optional `beebench_b01_prior`; see
[Economic-game tasks](#economic-game-tasks)). Each dataset task loads local files, applies a fixed train/test split
by time, survey wave or person, computes LLM-free baselines and scores any method with the metric and bootstrap used
in the research code it was ported from. The output is the shared
`EvalFrame` (`simengine/types.py`), so `simengine.evidence.evaluate` can take over from there.

## Components

| file | what it does |
|---|---|
| `base.py` | `BenchTask` base class (implements the `types.Task` protocol and adds `score`, `score_all`, `run`, `describe`), `Metric`, the registry (`register`, `get`, `list_tasks`), data-root resolution (`data_root` argument, else `$SIMENGINE_DATA`) and `DataNotFound`. |
| `stats.py` | Small metric helpers with no dependencies beyond numpy/scipy: unit bootstrap (`default_rng(42)`, B = 1000), cluster bootstrap, Fisher z, Pearson, Wasserstein-1 on 0-1 option positions, TVD, rank AUC, A/B pair accuracy. |
| `tasks/*.py` | One adapter per dataset: `load` (with the split), `baselines`, `frame` and the task's metrics. |
| `leaderboard.py` | `run(tasks, methods, data_root)` returns a long table (`task, method, metric, value, lo, hi, n, higher_is_better, primary, is_baseline, status, note`). `to_markdown(table)` renders one section per task, with methods as rows, metrics as columns and `value [lo, hi]` cells. A task whose files are missing becomes one row with `status = missing_data`; it does not raise. |

A method is any of the following:

- a callable `f(data)` or `f(task, data)` returning predictions aligned with `data.test` rows (a 1-D array, or a list
  of probability vectors for distribution tasks);
- an object with `.predict(data)`;
- the name of one of the task's baselines.

Bench code never imports `simengine.methods` internals.

## Usage

```python
from simengine import bench

bench.list()        # ['bank_marketing', 'beebench_b01_prior', 'econ_games_b01', 'ertan_wtp', 'gss', 'open_ecommerce',
                    #  'opinionqa_groups', 'simbench', 'twin2k500', 'upworthy']
task = bench.get("upworthy", split="temporal")
data = task.load("/path/to/catalogue")            # or export SIMENGINE_DATA=/path/to/catalogue
frame = task.run(methods={"mine": lambda d: my_model(d.train, d.test)}, data=data)   # baselines + mine
task.score(frame, "mine")       # {'metric': 'pair_acc', 'value': ..., 'lo': ..., 'hi': ..., 'n': ...}

table = bench.run(["gss", ("opinionqa_groups", {"split": "dev"}), "bank_marketing"], data_root="/path/to/catalogue")
print(bench.to_markdown(table))
```

`bench.get(name).layout` prints the files a task expects. Every loader accepts either the catalogue root (the one
holding `gss/`, `upworthy/`, ...) or the task's own directory.

Each task declares its options (`accepted_options`; shown by `simengine bench list -v` and `describe()`), and an
unknown option key raises `ValueError`, so a typo such as `upworthy:spilt=holdout` fails instead of silently running
the default split.

## Datasets and licences

The repository ships no data. Download each set yourself under its own terms and point `data_root` (or
`SIMENGINE_DATA`) at your copy. Check the terms before you build a prize-bearing or commercial submission on a set:
the non-commercial and academic-only ones may not allow it.

| dataset | task / where results come from | licence / terms | watch out |
|---|---|---|---|
| UCI Bank Marketing | `bank_marketing` | CC BY 4.0 | attribution |
| Upworthy Research Archive | `upworthy` | CC BY 4.0 | attribution |
| Open e-commerce 1.0 (Harvard Dataverse) | `open_ecommerce` | CC BY 4.0 | attribution |
| Twin-2K-500 | `twin2k500` | CC BY 4.0 per dataset card | attribution |
| OPeRA | next-action results (methods) | CC BY 4.0 | attribution |
| SimBench | `simbench` | **CC BY-NC-SA 4.0** | **non-commercial**; share-alike |
| Pew ATP via OpinionQA / SubPOP | `opinionqa_groups` | **research use** (Pew terms) | check before prize or commercial use |
| GSS panels (NORC) | `gss` | free, attribution required, **no redistribution** | do not ship the data with a submission |
| Dominick's | ABM elasticity results (abm) | **academic research use only** | clear it with the organiser before using it at an event |
| SEISMIC (SNAP) | retweet-curve results (abm) | **no licence stated**, research use; cite Zhao et al. 2015 | ask the source before prize use |
| BeeBench b01 results (edsl-test) | `beebench_b01_prior` | same owner, no LICENSE file (all rights reserved by default) | internal reuse only; do not ship the CSV |

Sources: each adapter's `meta["source"]` and the research data notes (`makro/arastirma/veri-ham.md`,
`makro/abm/arastirma/literatur.md`, `web/PROTOKOL.md`). The dataset's own page is what binds.

## Tasks

Unless a note says otherwise, numbers are the research code's own results, with the branch and file under
`talp-simulation-hackathon-2026/uretim/`. A number marked **(re-run)** is one this module computed on the real data
and checked against that source. A number marked **(new)** was computed by this module and has no counterpart in the
source.

### `opinionqa_groups` (distribution): Pew ATP answer distributions of 22 US demographic groups

- **Files:** `opinionqa/islenmis/{sorular,dagilim}.parquet`,
  `subpop-opinionqa/data/subpopulation_metadata/demographics_22.csv` and
  `subpop-opinionqa/data/opinionqa/processed/opinionqa.csv`.
- **Split and options:** `split='test'` (default) uses the 486 SubPOP OpinionQA test questions; `split='dev'` uses
  the pool questions, with `items='climb'|'select'` choosing the dongu hash partition. The wave-out rule applies: a
  question from wave w is predicted only from questions of other waves (`task.history(data, w)`). `setting='B'` also
  reveals the target question's Overall distribution, which enables the `overall` and `sibling` baselines.
- **Metric:** Wasserstein-1 on 0-1 option positions, averaged over groups and then over questions, with a question
  bootstrap (lower is better). Also reported: `tvd1` (1 − TVD) and `rank_rho`.
- **Baselines:** `uniform`, `position` (mean distribution of same-option-count questions from other waves),
  `knn` (TF-IDF over question and options, k = 20). Setting B adds `overall` and `sibling`.

**Measured numbers:**

| what | WD [95% CI], n | source |
|---|---|---|
| test, uniform | 0.1779 [0.1716, 0.1844], n 486 **(re-run, identical)** | dongu `turlar/oqa_grup_A/test/OZET.md` |
| test, position | 0.1645 [0.1570, 0.1724], n 486 **(re-run, identical)** | same |
| test, best LLM-loop configuration (knn + bge embeddings) | 0.1669 [0.1590, 0.1750]; did not beat position (+0.0025 [−0.0018, +0.0070]) | same |
| dev climb, uniform | 0.1976 [0.1908, 0.2044], n 717 **(re-run, identical; data test)** | dongu `turlar/oqa_grup_A/tabanlar.json` |
| dev climb, position | 0.1856 [0.1789, 0.1930] **(re-run, identical)** | same |
| setting B dev, `overall` / `sibling` | 0.0530 / 0.0764 | dongu `turlar/oqa_grup_B/tabanlar.json` |
| dev climb, `knn` TF-IDF | 0.1762 [0.1694, 0.1838] **(new)** | this module |
| test, `knn` TF-IDF | 0.1732 [0.1643, 0.1821] **(new)** | this module |

- **When it works:** as a hard, honest baseline ladder. The position prior is hard to beat without population
  information.
- **When it doesn't:** in setting B, knowing the Overall distribution removes most of the error (WD 0.053), so
  results from setting A and setting B must not be compared.

### `twin2k500` (continuous): a new person's answers on the Twin-2K-500 heuristics-and-biases block

- **Files:** `twin2k500/question_catalog_and_human_response_csv/wave1_3_response.csv` (features),
  `twin2k500/LLM_simulation_results/wave4_formatted_to_catalog_mapping.json`, and the formatted truth and retest CSVs
  in `twin2k500/LLM_simulation_results/GPT4.1-mini-simulation-llm-vs-human/`.
- **Split:** persons are split by a fixed id hash (dongu `kisi_parca`, 25% test; 1,551 train / 507 test persons on
  the real data). With `split='dev'`, a second hash holds out 20% of the dev persons, and test persons are never used.
- **Scoring:** follows the official MAD evaluation as re-implemented in persona `eval/twin_score.py`. Anchoring
  answers are converted to deciles, and only cells with a wave 4 answer count (122 columns on the real data).
- **Metrics:** `mean_z` (Fisher z of per-column between-person r; a constant prediction counts as 0), `corr`, and the
  official task-level accuracy `acc`.
- **Baselines:** `median` and `ridge` (median imputation, standardisation, Ridge α = 300 on 591 features).
- **Predictions:** give raw units, with anchoring items as free numbers. `frame()` converts them.

**Measured numbers:**

- 5-fold CV on all persons: column median acc 0.755 with corr 0; ridge corr 0.154 (persona `RAPOR.md`,
  `baseline/twin_baselines.py`).
- On the hash test persons: median acc 0.757 [0.751, 0.763]; ridge corr 0.156 [0.131, 0.183], acc 0.724; the
  wave 4 retest as a ceiling reaches corr 0.565 and acc 0.818 **(new)**.
- dongu test, with the persona-kit scorer on the same test persons: T-medyan z 0.000, T-kit z 0.218, T-adil z 0.368,
  LLM-loop winner z 0.374 (dongu `turlar/twin_kit/test/OZET.md`).

**When it doesn't work:** accuracy rewards the median (0.757, higher than ridge). Use the correlation metrics to see
whether a method tells persons apart.

### `open_ecommerce` (binary): which of 20 Amazon categories a person buys from in 2022-H2 (TALP "73%" setup)

- **Files:** `open-ecommerce/amazon-purchases.csv` and `open-ecommerce/survey.csv`.
- **Setup** (talp73 `PROTOKOL.md` v1.0):
  - Persons are users fully observed through 2022: 2,174 on the real data.
  - History is everything before 2022-07-01. The test window runs from 2022-07-01 to 2022-12-31.
  - The 20 categories are those with the most distinct buyers in the history.
  - The sample is 400 persons with `seed=s`, giving 8,000 test rows.
  - Classic models train at the 2022-01-01 cut on persons outside the sample.
- **Primary metric:** `acc_1b_c`, a balanced within-person top-half accuracy (chance 0.5) with a person bootstrap by
  weight matrix. Also reported: `acc_1a` (p ≥ 0.5 on all rows) and `person_auc`.
- **Baselines:** `never`, `constant_rate`, `random`, `popularity`, `persistence`, `decayed_count`, plus `survey_lgbm`
  and `history_lgbm` (LightGBM is optional and imported lazily).
- **Re-run on seed 0:** labels, prior rates and all eight baseline prediction vectors match the talp73 output files
  exactly (max difference 2e-16, LightGBM included). The metrics match too:

| metric | sabit | populerlik | kalicilik | sonumlu | anket | gecmis |
|---|---|---|---|---|---|---|
| 1b(c) | 0.5062 | 0.5833 | 0.5862 | 0.6530 | 0.6009 | 0.6750 |
| 1a | 0.7665 | — | — | — | — | 0.8035 |
| person AUC | — | — | — | — | — | 0.7303 |

  The sabit, populerlik, kalicilik, sonumlu, anket and gecmis baselines correspond to never, popularity, persistence,
  decayed_count, survey_lgbm and history_lgbm here. The source is talp73 `sonuc/<method>_s0.json` in the research
  work directory.

**Published 3-seed means** of 1b(c) (talp73 `kart/KART.md`, `kanit/LIG.md`):

| method | 1b(c) [95% CI] |
|---|---|
| sabit | 49.5% [45.8, 52.6] |
| popularity | 57.8% [55.3, 60.3] |
| persistence | 58.4% [54.9, 61.6] |
| survey LightGBM | 58.8% [55.6, 61.9] |
| decayed count | 65.0% [62.8, 67.1] |
| history LightGBM | 67.2% [65.1, 69.3] |
| L1-haiku55 | 56.4% [52.5, 60.2] |
| L1-sonnet55 | 57.4% [54.8, 60.3] |
| L2-haiku55 (LLM persona + history) | 2.0-2.6 points below history LightGBM in each seed |

- **When it works:** whenever purchase history is available. The history features carry the signal, and survey-only
  information sits near popularity.
- **When it doesn't:** `acc_1a` is dominated by "never buys" (0.766), so do not rank methods by it.

### `upworthy` (ab): which of two headlines won an Upworthy A/B test

- **Files:** `upworthy/islenmis/pairwise.parquet` from the research pair builder. Otherwise the three archive package
  CSVs in `upworthy/ham/`, from which pairs are rebuilt: same A/B randomisation (`default_rng(42)`), two-proportion
  p-values, BH-FDR per archive split.
- **Pairs:** headline-only pairs, with ties dropped. Test pairs must pass FDR unless `test_signif='none'`.
- **Splits:** `split='temporal'` (default) puts the newest 20% of tests in the test set. `split='holdout'` uses the
  archive's official holdout.
- **Metric:** pair accuracy with a cluster bootstrap over tests (`n` counts tests). Also reported: AUC and Brier.
- **Baselines:** `coin`, `longer_headline`, `tfidf_bt` (Bradley-Terry logistic on TF-IDF differences, C = 0.3).

**Measured numbers** (deney `etki/kod/d1_sizinti.py`, `RAPOR.md`):

| split | tfidf_bt pair accuracy [95% CI] | pairs | tests | longer_headline (new) |
|---|---|---|---|---|
| temporal | 0.758 [0.739, 0.777] **(re-run, identical with B = 2000, seed 0)** | 3,274 | 1,302 | 0.533 |
| holdout | 0.800 [0.782, 0.819] **(re-run, identical)** | 2,966 | 1,130 | 0.564 |

**When it doesn't work:** headlines get reused across tests, so the official holdout overstates generalisation. The
temporal split is the honest one, which is why it is the default.

### `simbench` (distribution): SimBench group response distributions, 20 datasets

- **Files:** `simbench/SimBenchPop.csv` and `simbench/SimBenchGrouped.csv`.
- **Split:** SimBench publishes **no** train/test split; all of it is test data. This adapter uses the split
  preregistered in persona/kosul `kod/simbench/sb.py`: 20% of questions go to dev, stratified by dataset, with seed 0.
  That gives Pop 1,430 dev / 5,737 test rows and Grouped 1,262 / 5,081 **(re-run, identical counts)**. TODO: add
  `split='official'` if SimBench ever publishes a split.
- **Metric:** S = 100 · (1 − TVD / dataset mean TVD to uniform), averaged per part and then over the two parts, with
  a question-clustered bootstrap. Also reported: `tvd`. Uniform scores 0 when computed on every row.
- **Baselines:** `uniform` and `dev_marginal`. `dev_marginal` is the mean human share per dataset and option label in
  dev, leaving out the row's own question.

**Measured numbers:**

- The uniform mean TVD is 0.335 on Pop and 0.362 on Grouped **(re-run, identical**; persona/kosul
  `kod/simbench/PROTOKOL.md`, which matches the paper's Table 10).
- Published best: Claude-3.7-Sonnet, S = 40.80 (paper, as quoted in the same PROTOKOL.md).
- On the test rows: `dev_marginal` S = 9.51 [5.13, 13.86] and `uniform` S = −0.59 [−3.68, 2.45] **(new)**. Uniform is
  not exactly 0 because the denominator includes the dev rows. The preregistered expectation for `dev_marginal` was
  S 5-20.
- No LLM results exist in the source branch yet.

### `gss` (continuous): a person's GSS attitude answer in the next panel wave

- **Files:** `gss/panel2006.parquet`, `gss/panel2008.parquet`, `gss/panel2010.parquet` and
  `gss/panel2016_2020.parquet` (NORC panels converted to parquet).
- **Split:** train on the old panels (wave 1 → wave 3) and test on the 2016-20 panel (2016 or 2018 answer → 2020
  answer). `holdout_panel='panel2006'` and similar give a leave-one-old-panel-out fold, and
  `tasks.gss.cross_panel_dev()` reproduces the dongu development score. Item selection follows the preregistered rule
  (factual and nominal items stay features only).
- **Metric:** mean Fisher z of per-item between-person r, with an item bootstrap. Also reported: `corr`.
- **Baselines:** `median`, `persistence` (own previous answer), `ridge`, `pca_ridge`.

**Measured numbers:**

| what | z [95% CI], n | source |
|---|---|---|
| dev climb, T-medyan | 0 | dongu `turlar/gss_dalga/tabanlar.json` |
| dev climb, T-kendi (persistence) | 0.4754 (corr 0.4425), n 101 **(re-run, identical; data test)** | same |
| dev climb, T-pca | 0.5437 (corr 0.4958) **(re-run, identical)** | same |
| test, T-kendi | 0.4905 [0.4632, 0.5181], n 154 | dongu `turlar/gss_dalga/test/OZET.md` |
| test, T-ridge | 0.5124 | same |
| test, T-pca | 0.5601 [0.5288, 0.5913] (corr 0.508) | same |
| test, LLM-loop winner | equal to T-pca | same |

- **When it works:** this is a strong-baseline task, because attitudes persist.
- **When it doesn't:** any method has to beat 0.56 z, not the median.

### `bank_marketing` (binary): term-deposit response to a telephone campaign (UCI, 2008-2010)

- **File:** `bank-marketing/bank-additional/bank-additional-full.csv`.
- **Split:** rows are in call order. The first 60% train, the next 20% are dev and the last 20% test (8,238 rows on
  the real data). `duration` is dropped.
- **Metrics:** AUC (primary), PR-AUC and Brier, with a row bootstrap (`default_rng(42)`).
- **Baselines:** `constant`, `persistence` (yes-rate by previous outcome), `logistic`, and `lgbm` (grid with early
  stopping on dev AUC, then a refit).

**Measured numbers** on test (intent `gbdt/lig/bank_lig.json`, written by `gbdt/lig/bank/bank.py kos`, commit
b0aeb08):

| method | AUC [95% CI] | other |
|---|---|---|
| constant | 0.5 | PR-AUC 0.308 |
| persistence | 0.6408 [0.6289, 0.6532] | |
| logistic | 0.7483 [0.7369, 0.7592] | PR-AUC 0.527, Brier 0.191 |
| lgbm | 0.6653 [0.6537, 0.6770] | |

All four prediction vectors and CIs were **re-run and are identical**. The data test checks the logistic and
persistence AUCs.

The LLM persona arms in the source are not part of bench, because they need an LLM:

| arm | AUC |
|---|---|
| persona_haiku | 0.6835 |
| hybrid (i) | 0.6939 |
| Sonnet, 1,000-row subset | 0.7336 |

**When it doesn't work:** the base rate drifts from 4.8% yes in train to 30.8% in test. Under that drift the LightGBM
grid collapses: dev AUC is 0.5 and it keeps a single tree (source `bilgi.lgbm_ayar`). The plain logistic model is the
baseline to beat. Brier punishes every model trained on the old base rate.

## Economic-game tasks

Three tasks score `simengine.games` output against published human numbers. `econ_games_b01` and `ertan_wtp` need
no data: their human values are small published constants inlined in `simengine.games`, each with its citation, and
`load()` ignores `data_root`. Sources (read-only, same owner): edsl-test `benchmark/beebench/b01` (games, holdout,
references), `benchmark/score_wtp.py` and `benchmark/human_targets.json`; replicant `papers/ertan2009/config.py`
(MIT) for `ertan_vignette`. Files: `tasks/econ_games_b01.py`, `tasks/ertan_wtp.py`, `tasks/beebench_b01_prior.py`.

```python
import simengine.games as gm
from simengine import bench
from simengine.bench.tasks import econ_games_b01 as E, ertan_wtp as W

task = bench.get("econ_games_b01")                       # no data needed
res = gm.play(llm, bank, "holdout", runs=3)              # or any games.play results table / CSV
frame = task.run({"mine": res})                          # the results table itself is a valid prediction
task.score(frame, "mine")                                # {'metric': 'mae', 'value': ..., 'lo': ..., 'hi': ..., 'n': 4}
ev_frame = task.results_frame(res)                       # train + test rows -> simengine.evidence.evaluate

wtp = bench.get("ertan_wtp")
frame = wtp.run({"soc": W.society_method(llm, personas, seeds=(0, 1, 2))})   # abm.society.run_who_to_punish
```

### `econ_games_b01` (continuous): simulated vs human moments of the one-shot game battery

- **Rows:** one per human-referenced moment from `games.human_refs()`, `unit = game:moment` (for example
  `dictator:mean_give_share`, truth 0.2835, Engel 2011). `data.test` also carries `scale`, `citation` and
  `holdout`. `group` is the game.
- **Split** (`split=`): `holdout` (default) puts the calibration-game moments in train and the holdout-game moments
  in test. That gives 16 train and 8 test moments over 4 holdout games (beauty contest, pg_punishment,
  third_party_punishment, who_to_punish), following the b01 rule that holdout games are never used for tuning.
  `all` tests all 24 share moments over 12 games. `time_choices` and `ertan_vignette` have moments but no numeric
  human reference, so they have no rows.
- **Scale** (`scale=`): `share` (default) keeps 0-1 moments only. `all` adds 7 raw-unit moments, such as the
  beauty-contest mean guess (36.73), punishment points and lambda. Their errors are in game units and do not pool
  with the shares, so use `all` only for per-moment inspection.
- **Metrics:**
  - `mae` (primary, lower is better) and `rmse` over the test moments that have a finite prediction. The CI is a
    cluster bootstrap over games: all moments of a game move together, and `n` is the number of games.
  - `coverage`: the share of test moments the method produced a value for. A moment that a method cannot compute
    (no valid answers, no complete schedule) stays NaN and is never filled in. It lowers `coverage` and is left out
    of `mae`, so read the two together.
- **Predictions:** a vector aligned with `data.test`, or a `games.play` results table, which `frame()` turns into
  moments through `games.moment_frame`. A table with several persona methods gives one column per method
  (`<name>_<persona_method>`). Helpers:
  - `results_method(results, persona_method=None)` and `play_method(llm, personas, **play_kw)`. The second plays
    only the test games, and the `games.play` cost gate applies.
  - `task.results_frame(results)`: a train + test frame with every persona method and the baselines, which
    `simengine.evidence.evaluate(frame, ours=..., task="continuous")` takes directly.
- **Baselines:**
  - `rational_corner`: a self-interested, risk-neutral payoff maximiser (`CORNER_DECISIONS`): dictator 0, trust 0,
    beauty contest 0, Holt-Laury safe in rows 1-4, accept a gamble only at positive expected value, contribute and
    punish 0, the smallest positive ultimatum offer ($1), accept every offer, vote No on every ballot item.
    `eleven_twenty` has no pure-strategy equilibrium, so the corner requests 20 (the level-0 anchor). Moments are
    computed by `games.moments` on those decisions. `perverse_event_share` is therefore NaN, because nobody
    punishes.
  - `midpoint`: numerical answers at the midpoint of the condition's range, which is the equal split where the
    range is a pot (dictator 50, trust return half, beauty contest 50). Multiple-choice answers are uniform random,
    computed exactly by enumerating every answer combination across a game's conditions (1,024 Holt-Laury
    schedules, for example).
  - `bare_model`: the no-persona model, the b01 `baseline` method. It comes from `results=` (rows whose
    persona_method is `bare_method`) and/or `llm=`, which plays the referenced games with `bare_agents` empty
    personas × `runs` runs, resuming from `results=`. The `games.play` cost gate applies; pass
    `play_kw={"confirm": True}` to go over it. The baseline is absent when neither option is given, because bench
    baselines never call an LLM on their own.

### `ertan_wtp` (continuous): who-to-punish human-likeness, criterion by criterion

- **Rows:** the 9 criteria of `games.wtp_scorecard` (port of edsl-test `score_wtp.py`), each scored 0-1 against the
  Ertan, Page & Putterman (2005 WP, EER 2009) targets from `human_targets.json`.
  - `truth = 1.0` for every criterion: the full human-likeness credit. Each criterion's scoring rule already
    encodes its human target, and `data.test["target"]` quotes the target and its printed page.
  - `weight` is 3/2/1 (institutional, mechanics, distribution). `group` is that family.
  - There are no train rows.
- **Predictions:** the mean criterion score over runs. A criterion that is N/A in a run is excluded from that run's
  mean, never zero-filled. One that is N/A in every run stays NaN. The run output itself can be passed as the
  prediction: a structured run dict, the `abm.society.run_who_to_punish` result (every group is scored), or a list
  of either. Helpers:
  - `runs_method(runs)` and `society_method(llm, personas, seeds=(0, 1, 2))`.
  - `group_to_run(group)`: takes `punish_high` from the group's voted rules and leaves invalid punishment answers
    out of the events.
  - `scorecards(runs)`: one row per run. Use it for the seed-to-seed spread.
  - In `bench.run`, wrap the run output in `runs_method(...)`. The leaderboard reads a methods dict whose values
    are all dicts or lists as a per-task mapping.
- **Metrics:**
  - `index` (primary, higher is better): the weighted mean over the observed criteria. It equals
    `wtp_scorecard(run)["index"]` for one run.
  - `wmae`: the weighted mean |1 − score|.
  - `coverage`: the share of the 9 criteria observed.
  - The CI resamples criteria with their weights. It shows how much the index leans on single criteria, not
    sampling uncertainty across groups or seeds.
- **Baselines:** synthetic runs scored by the same scorecard (4 players, endowment 10, `votes` × `periods`, default
  2 × 3).
  - `rational_corner`: everyone gives 0 and nothing passes. Index 0.455, with 5 of 9 criteria observed.
  - `full_cooperation`: everyone gives 10 and punish-low always passes. Index 0.769, with 6 of 9 criteria
    observed **(new)**.
  - The scorecard is lenient to a constant full-cooperation run. With a single regime, `treatment_effect` is N/A
    and nobody is below average to punish. Read `coverage` next to `index`.
- **evaluate:** the frame passes `simengine.evidence.evaluate` (the `weight` column becomes its row weight). Because
  truth is constant, its "another unit's truth" random baseline is a perfect 1.0, so its baseline ladder means
  nothing here. Use `task.score` for this task.

### `beebench_b01_prior` (continuous, optional): the prior b01 runs as fixed reference arms

- **Files:** `<root>/beebench_b01/results.csv` (or the directory that holds `results.csv`), the b01 tidy results
  table. It is not shipped. The original run is 10,920 rows (7 persona methods × 14 games × 10 agents × 3 runs,
  gpt-oss-120b), with 30 invalid rows.
- **Task:** same rows, split, scale and metrics as `econ_games_b01`. Each persona method in the file becomes a
  reference arm named `b01_<persona_method>`. The no-persona method (`bare_method`, default `baseline`) becomes
  `bare_model`. Options: `methods=` (`big5|twin2k` on the command line) and `model=`. The arms are listed as
  baselines by `bench.run`. They are prior results, not fitted to this task.
- **Moments:** all are recomputed from the rows by `simengine.games`, on valid rows only. The numbers below are
  simengine's own computation, not figures from the b01 repository.

**Measured numbers (new)** on the original b01 file, read in place. Holdout split, share scale, B = 1000,
`default_rng(42)`:

| arm | holdout MAE [95% CI], 8 moments / 4 games | all-games MAE [95% CI], 24 moments / 12 games (`split=all`) |
|---|---|---|
| b01_gps | 0.113 [0.070, 0.187] | 0.171 [0.125, 0.218] |
| b01_twin2k | 0.185 [0.104, 0.239] | 0.179 [0.128, 0.213] |
| midpoint | 0.201 [0.100, 0.316] | 0.254 [0.143, 0.351] |
| b01_anthology | 0.202 [0.116, 0.289] | 0.188 [0.132, 0.251] |
| b01_demographic | 0.212 [0.124, 0.317] | 0.283 [0.209, 0.357] |
| b01_big5 | 0.246 [0.167, 0.363] | 0.220 [0.166, 0.270] |
| b01_value_anchor | 0.250 [0.133, 0.458] | 0.297 [0.187, 0.401] |
| bare_model (b01 `baseline`) | 0.341 [0.212, 0.506] | 0.404 [0.334, 0.470] |
| rational_corner | 0.452 [0.326, 0.590] (coverage 0.875) | 0.456 [0.390, 0.531] (coverage 0.958) |

- **Caveats:**
  - One model.
  - 10 agents × 3 runs per method.
  - The holdout CI rests on 4 game clusters, so the ranking inside the persona methods is not significant. The
    intervals overlap widely.
  - `split=all` includes the calibration games, which were visible when the b01 personas were built.
- **What holds:** the bare model is worse than the LLM-free midpoint on both splits, and every persona method beats
  the bare model on the holdout games. This matches insight 2 of the porting plan: always show the no-persona
  control, and beat the simple baseline too.

## Tests

Run `python -m unittest discover -s tests -t . -p 'test_bench*.py'`. There are 46 tests, taking about 11 s on the
research machine.

- `test_bench_adapters.py` writes tiny synthetic files in each task's layout and checks loading, split invariants,
  baselines, frame shape and metrics for all seven adapters, including the Upworthy raw-CSV fallback.
- `test_bench_core.py` covers the registry, the method-calling forms, frame validation, the leaderboard (including
  missing data) and the stats helpers.
- `test_bench_data.py` runs only when the catalogue is present (`$SIMENGINE_TEST_DATA`, default: the research
  machine's catalogue path). It reproduces the GSS T-kendi dev score, the OpinionQA uniform dev WD and the bank
  logistic/persistence AUCs.
- `test_bench_games.py` covers the three economic-game tasks offline:
  - loading with no data, the options, and the corner and midpoint values;
  - results tables as predictions, and NaN moments that lower coverage and are never filled;
  - `play_method` and the `llm=` bare model on a scripted `FakeLLM`, and the cost gate;
  - the scorecard frame, N/A exclusion, `society_method` on `abm.society`, and `evidence.evaluate` on both frames;
  - the prior task skipping when its file is absent, and the prior task on a synthetic file in its layout.

  The real b01 check (10,920 rows, 30 invalid) runs only when `$SIMENGINE_B01_DIR`, or
  `<$SIMENGINE_TEST_DATA>/beebench_b01/`, holds `results.csv`.

## Known gaps

- `simbench` uses the persona/kosul dev/test split, because SimBench itself has none. The bootstrap is clustered by
  question, but unlike the source it is not stratified by part × dataset, so CIs can differ slightly.
- `open_ecommerce` leaves out talp73 interpretations 1a(b) (calibrated threshold), 2 (hit@k), 3 (aggregate
  distribution) and 5 (segment cells). They need a polars-hash tie-breaker or segment rules that were not ported.
  1b(c), 1a(a) and person AUC are included.
- `twin2k500` does not port the persona kit's stack model (`T-kit`) or the price-logistic `T-adil`. The baselines are
  median and ridge only.
- `econ_games_b01` has only 8 holdout share moments over 4 games, so its CIs are wide. The beauty contest has no
  full human distribution, so this task carries no Wasserstein or KS row (the games module computes W1/KS for
  eleven_twenty only, outside the task).
- `ertan_wtp` scores criteria, not raw Ertan statistics. Its CI is a criterion bootstrap, not a seed bootstrap. The
  `final_vote_punish_low` human rate is 0.875, so a real human group would average below 1.0 on that row.
- `beebench_b01_prior` treats the prior arms as baselines in `bench.run` (`is_baseline` is true).
- No dataset task runs LLM methods itself. `econ_games_b01` calls an LLM only when you pass `llm=`. LLM arms come in as user methods, for example from `simengine.methods`.
