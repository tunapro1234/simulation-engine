# bench task `socsci210`: response distributions in 210 survey experiments

`simengine/bench/tasks/socsci210.py` turns SocSci210 (Kolluri et al. 2025) into a distribution task. SocSci210 holds
2.9M answers from 210 TESS survey experiments. The unit is a cell, meaning one (study, outcome, condition) triple.
Given the stimulus a participant saw and the response scale, a method predicts how the people in that condition
answered: one probability vector over the integer levels of the scale. Scoring follows the paper's distributional
metric. The Wasserstein-1 distance is taken on the scale rescaled to [0, 1], averaged over the cells of a study and
then over studies. Lower is better.

The task follows the `BenchTask` conventions in `simengine/bench/base.py` (`layout`, `load`, `baselines`, `frame`,
`metrics`). It is not in the package registry yet, so import the module directly. Importing it registers the name
`socsci210`.

### What it adds next to the other tasks

- `opinionqa_groups` and `simbench` score how demographic groups answer survey questions (Pew ATP, and the 20
  SimBench datasets). Neither contains TESS experiments. Their unit is a group, not an experimental condition.
- `research.auto_social_science` already reads the SocSci210 condition table (`cond_stats.parquet`). It scores
  forecasts of condition contrasts (effect sizes, Hewitt-style), not answer distributions, and it is not a bench task.
- This task is the only one that asks for the full answer distribution of each experimental condition. It also
  carries the paper's three generalisation splits.

## API

```python
from simengine.bench.tasks import socsci210 as ss

task = ss.SocSci210(split="study")             # kind "distribution"; metrics w1 (primary), w1_cell, mean_err
data = task.load("<catalogue>")                # or $SIMENGINE_DATA; the directory holding socsci210/
data.test[["study_id", "task_num", "condition_num", "stimuli", "lo", "hi", "levels", "n"]]
frame = task.run(methods={"mine": lambda d: [my_vector(s, k) for s, k in zip(d.test.stimuli, d.test.levels)]},
                 data=data)                    # EvalFrame with the baselines as well
task.score(frame, "mine")                      # {'metric': 'w1', 'value', 'lo', 'hi', 'n': studies, 'n_cells'}
task.score_all(frame)
task.reference(frame, "empirical_best")        # the paper's bootstrap floor on the same cells
task.reference(frame, "uniform01")             # the paper's Uniform Guess (W1 to a continuous U(0, 1))

ss.dist_from_responses([4, 5, 5, 3], lo=1, hi=5)   # sampled answers (e.g. an LLM persona panel) -> vector
ss.write_hist("<catalogue>/socsci210", "<catalogue>/socsci210/hist.parquet")   # aggregate the shards once
```

| option | values (default) | meaning |
|---|---|---|
| `split` | `"study"` \| `"condition"` \| `"outcome"` (`"study"`) | the paper's split (below) |
| `max_levels` | int (101) | drop cells whose scale has more levels (e.g. 0-1200 dollars) |
| `min_n` | int (1) | drop cells with fewer answers on the scale; a cell with none is always dropped |
| `min_in_scale` | float (0.9) | drop cells where a smaller share of answers lies on the stated scale |
| `k` | int (10) | `tfidf_knn` neighbours |
| `per_item` | int (1 for `split="study"`, else 0) | `tfidf_knn`: at most this many neighbour cells per train outcome, 0 = no cap |
| `baselines` | `"+"`-joined names (all that apply) | subset of `uniform`, `scale_marginal`, `tfidf_knn`, `pooled` |

An unknown option key raises `ValueError`, as in every bench task. A prediction must have exactly `levels` entries.
It is clipped at 0 and renormalised, and an all-zero vector counts as uniform.

Other public names: `parse_scale`, `positions`, `w1`, `w1_uniform01`, `empirical_best`, `build_hist`, `cells`,
`load_hist`, `scale_marginal`, `pooled`, `tfidf_knn`, `SPLITS`, `BASELINES`, `SHARD_COLUMNS`, `HIST_FILE`.

### Files

```
<root>/socsci210/hist.parquet                         optional; one row per cell, written by write_hist (preferred)
<root>/socsci210/ham/data/train-000NN-of-00017.parquet HF socratesft/SocSci210 shards (or <root>/socsci210/data/)
<root>/socsci210/ham/metadata/{participant,condition,task}_mapping.json   HF metadata/
```

Only `study_id, task_num, condition_num, sample_id, response, stimuli` are read from a shard, one shard at a time.
Any subset of the 17 shards loads, but it covers only the studies in it. The full set is about 1.4 GB. With no
`hist.parquet`, `load` aggregates the shards on every call. Aggregating all 17 took 36 s and peaked at 1.0 GB RSS
here, so run `write_hist` once instead. `hist.parquet` stores the raw answer values and their counts per cell, so a
better scale parser needs no rebuild. An older `hist.parquet` without the `answers` column raises `DataNotFound`.

### Response scale

The paper restricts outcomes to binary and ordinal questions, and the dataset does not record the scale as a field. So
`parse_scale` reads it from the response instruction at the end of the stimulus. It recognises these forms:

- a range, e.g. "Only return an integer from 1 to 7 ..." or "a number between 0 and 100";
- a dash, e.g. "1–4";
- a consecutive list, e.g. "1, 2, 3, or 4";
- enumerated codes, e.g. "1 for Yes or 2 for No", "1 if ...; 2 if ...", "1 to choose ...; 2 to choose ...";
- an anchored scale without an instruction, e.g. "from 1 (very low) to 7 (very high)".

Open amounts have no stated scale and are left out. Examples are "an integer representing dollars" and "a multiple of
100 between 0 and 1000". `cells` then counts the answers on each level. Answers off the stated scale are dropped and
counted in `n_dropped`. A cell keeps its place only if at least `min_in_scale` of its answers lie on the scale.

On the full data, 5,998 cells exist. 75 have no parseable scale (open amounts). 203 have fewer than 90% of their
answers on the stated scale. These are coding mismatches in the source: for example 0/1 answers to a "1 or 2" item,
1-6 answers to a "1 or 2" item, or 0/25/50/75 answers to a 1-5 item. 5,660 cells are kept, holding 2.72M answers.
Four studies lose all their cells: two contain only open amounts, and two contain only mismatched binary items.

### Split

The splits come from the paper's mapping files:

- **`study`** (paper 5.3). Train on the 170 studies listed as `seen` in `participant_mapping.json`, test on the 40
  `unseen` ones. After the filters above this is 167 / 39 studies and 4,950 / 710 cells.
- **`condition`** (paper 5.4). This covers studies with at least 4 conditions. `condition_mapping.json` lists each
  answer's `sample_id` as train or eval. A cell is train when all its answers are train, and test when all are eval.
  On the full data no cell is mixed (`meta["cells_mixed_split"]` = 0). The result is 3,484 / 1,181 cells, with 127
  test studies.
- **`outcome`** (paper 5.4). This uses the same rule with `task_mapping.json`. The result is 2,908 / 995 cells, with
  71 test studies.

`data.meta` records the cell and study counts, the dropped-cell counts and the data source.

### Metrics

| metric | definition |
|---|---|
| `w1` (primary) | W1 between the predicted and the human vector on positions (r - r_min) / (r_max - r_min); averaged over a study's cells, then over studies; bootstrap CI over studies (B = 1000, seed 42) |
| `w1_cell` | the same W1, averaged over cells; cluster bootstrap over studies |
| `mean_err` | \|predicted mean - human mean\| on [0, 1], study-averaged |

`frame` also has two reference columns, which `reference()` averages the same way as `w1`:

- `ref_empirical_best` follows the paper's Empirical Best. It takes 100 with-replacement resamples of the cell's
  answers (a multinomial with the cell's n) and computes the mean W1 to the full sample, with a seeded rng.
- `ref_uniform01` is our reading of the paper's Uniform Guess: the exact W1 between the human distribution and a
  continuous U(0, 1). The `uniform` baseline is the discrete uniform over the scale's levels instead.

### Baselines

All baselines are LLM-free, use train cells only, and never look at test answers.

- `uniform`: equal mass on every level.
- `scale_marginal`: the mean train distribution over cells whose scale has the same number of levels.
- `tfidf_knn`: the mean distribution of the `k` train cells with the most similar stimulus. Similarity is cosine on
  TF-IDF of 1-2 grams, and only cells with the same number of levels count. With `per_item` = 1, one cell per train
  outcome is used. Without that cap on the study split, the 10 slots fill with near-identical conditions of a single
  study (see the numbers below). The per-split default (1 on `study`, 0 elsewhere) was picked after seeing the
  test-split scores of both settings, so `tfidf_knn` is a test-tuned baseline; both settings are reported below.
- `pooled`: for `condition` and `outcome` only. It is the mean of the same study's train cells for the same outcome,
  that is, the other conditions, which amounts to "the treatment does nothing". Without those, it uses the same study
  and level count, and without that, `scale_marginal`.

## Measured numbers

These come from one run on the local copy of all 17 shards (22 Oct 2026, `nice -n 19`). The shards were first
aggregated with `write_hist`. Each value is `value [95% CI]`, and n counts studies. All three splits together took
about 5 s after the aggregation. A review pass rebuilt `hist.parquet` from the shards (37 s, 1.05 GB peak RSS; same
cells, counts and split shares) and re-ran every number below: all reproduced to the printed digits. Command (Python, from the repository root):

```python
from simengine.bench.tasks import socsci210 as ss
for split in ("study", "condition", "outcome"):
    task = ss.SocSci210(split=split)
    data = task.load("<catalogue>")            # <catalogue>/socsci210/hist.parquet from write_hist
    frame = task.run(data=data)
    print(data.meta, task.score_all(frame))
    print(task.reference(frame, "empirical_best"), task.reference(frame, "uniform01"))
```

**`split="study"`** (39 unseen studies, 710 cells):

| method | w1 | w1_cell | mean_err |
|---|---|---|---|
| uniform | 0.1564 [0.1329, 0.1803] | 0.1616 | 0.1312 |
| scale_marginal | **0.1473** [0.1251, 0.1710] | 0.1486 | 0.1282 |
| tfidf_knn (per_item 1) | 0.1591 [0.1361, 0.1862] | 0.1595 | 0.1415 |
| tfidf_knn, per_item 0 | 0.1862 [0.1570, 0.2225] | | |
| reference: Empirical Best | 0.0128 [0.0111, 0.0148] | | |
| reference: W1 to U(0, 1) | 0.1808 [0.1593, 0.2026] | | |

**`split="condition"`** (127 studies, 1,181 test cells):

| method | w1 | w1_cell | mean_err |
|---|---|---|---|
| uniform | 0.1527 [0.1393, 0.1652] | 0.1718 | 0.1311 |
| scale_marginal | 0.1431 [0.1298, 0.1555] | 0.1484 | 0.1283 |
| tfidf_knn (per_item 0) | 0.0886 [0.0804, 0.0970] | 0.0806 | 0.0813 |
| tfidf_knn, per_item 1 | 0.1292 [0.1180, 0.1409] | | |
| pooled | **0.0449** [0.0387, 0.0525] | 0.0541 | 0.0409 |
| reference: Empirical Best | 0.0151 [0.0139, 0.0164] | | |
| reference: W1 to U(0, 1) | 0.1743 [0.1626, 0.1861] | | |

**`split="outcome"`** (71 studies, 995 test cells):

| method | w1 | w1_cell | mean_err |
|---|---|---|---|
| uniform | 0.1531 [0.1349, 0.1726] | 0.1799 | 0.1323 |
| scale_marginal | 0.1508 [0.1327, 0.1738] | 0.1761 | 0.1373 |
| tfidf_knn (per_item 0) | 0.1343 [0.1108, 0.1620] | 0.1755 | 0.1244 |
| pooled | **0.1315** [0.1096, 0.1573] | 0.1732 | 0.1216 |
| reference: Empirical Best | 0.0159 [0.0144, 0.0178] | | |
| reference: W1 to U(0, 1) | 0.1622 [0.1448, 0.1816] | | |

### Published numbers (not measured here)

All figures are Wasserstein distances from Kolluri et al. 2025, arXiv 2509.05830.

| setting | Uniform Guess | Empirical Best | GPT-4o | Socrates-Qwen-14B (SFT) | Socrates-LLaMA-8B (SFT) |
|---|---|---|---|---|---|
| unseen studies (Table 2) | 0.203 | 0.125 | 0.174 | 0.151 | 0.153 |

For the condition and outcome splits, Table 3 reports LLaMA3-8B results only:

| setting | Uniform Guess | Empirical Best | LLaMA3-8B base | + SFT | + SFT w/ reasoning |
|---|---|---|---|---|---|
| condition split | 0.180 | 0.090 | 0.219 | 0.077 | 0.063 |
| outcome split | 0.165 | 0.086 | 0.224 | 0.125 | 0.114 |

### Reading the numbers

- **Uniform Guess matches the paper only roughly.** Our W1 to U(0, 1) gives 0.174 on the condition split and 0.162
  on the outcome split, against the paper's 0.180 and 0.165. On unseen studies the gap is larger: 0.181 against
  0.203. The cell sets differ, since we have 39 of the 40 test studies, filter off-scale cells and exclude scales
  above 101 levels. Dropping both filters (`min_in_scale=0`, `max_levels` unbounded; 758 study-split cells) moves
  the study-split figure only to 0.179, so the filters do not explain the gap (review re-run, 22 Oct 2026). So the
  paper numbers are a guide here, not a replication.
- **Empirical Best does not match.** Our implementation of the paper's description gives 0.013-0.016. The paper
  reports 0.125 / 0.090 / 0.086. With a median of 249 answers per kept cell, a resample of the answers is
  necessarily that close to the full sample. The paper's figure must therefore come from something else that its text
  does not state, such as a resample of the model's sample size or of individual-level predictions. Also, the
  paper's own SFT models beat its Empirical Best on the condition split (0.063 < 0.090). Treat
  `reference("empirical_best")` as a sampling-noise floor, not as the paper's number.
- **The condition split is easy for a no-LLM baseline.** `pooled` predicts the mean of the study's other conditions
  for the same outcome and scores 0.045. That is below the paper's best fine-tuned model (0.063) and below its
  Empirical Best (0.090), on cells that are not identical to the paper's. Treatment effects in TESS are small next
  to differences between outcomes. A method should be compared with `pooled` on this split, not only with uniform.
- **Text similarity does not transfer to unseen studies.** `tfidf_knn` is no better than `uniform`, and
  `scale_marginal` is the only baseline below it. On unseen studies, knowing the scale is worth about 0.01 W1.
  Fine-tuned LLMs in the paper gain about 0.05 over Uniform Guess.

## Provenance

- **Data.** HF `socratesft/SocSci210`, the 17 parquet shards and `metadata/*_mapping.json`. The source experiments
  are TESS (Time-sharing Experiments for the Social Sciences). The HF card states no licence, and this has not been
  verified. No data rows are in this repository. The tests use synthetic shards.
- **Method.** The method follows Kolluri et al. 2025, "Finetuning LLMs for Human Behavior Prediction in Social Science
  Experiments", arXiv 2509.05830. The following were checked against the arXiv HTML on 22 Oct 2026:
  - rescaling to [0, 1] and the cell → study → overall averaging (3.2);
  - Uniform Guess and Empirical Best (5.2);
  - the 170 / 40 study split (5.3);
  - the condition and outcome splits on studies with ≥ 4 conditions or outcomes, 75 / 25 (5.4).
- **Our own choices.** The paper does not specify the following, so they are ours: the parsing of the response scale,
  the `min_in_scale` and `max_levels` filters, the reading of Uniform Guess as W1 to a continuous U(0, 1), and the
  baselines.

## Limitations

- The task works at the distribution level. The paper's models predict one answer per participant, with demographics,
  and its individual accuracy metric is not implemented here. The shards carry `participant` and `demographic`
  columns, but `build_hist` drops them.
- Scale parsing is heuristic. A stated scale that is wrong for its answers is caught only when fewer than 90% of the
  answers fall on it. A wrong scale that happens to contain the answers would pass unnoticed.
- The stimulus of a cell is taken from its first answer row. The text is not always the same across a cell: in
  shard 0, 63 of 302 cells have several variants with randomised details, such as a name or a word in the story. The
  parsed scale was the same within every one of these cells. But `tfidf_knn`, and any method that reads
  `data.test.stimuli`, sees only the first variant.
- A review pass compared each outcome's parsed scale with the answers seen across its conditions: in 19 of 1,140
  kept outcomes the answers cover less than 60% of the stated scale. Some are genuine (a 1-5 item answered only
  with 3), some look like a mis-stated scale in the source (a "1 (Support strongly) to 5" item answered only 1 or
  2). These cells stay in, because their answers lie on the stated scale. In 12 outcomes the stated scale differs
  between conditions (e.g. 0-7 vs 0-8 counts, or a 1-5 vs a 1-7 wording); each cell is scored on its own text.
- Wide scales (0-100) are scored on 101 levels. Their W1 rewards getting the mean right more than the shape.
- Empirical Best does not reproduce the paper's value (see above), and the published numbers are on a slightly
  different cell set.
