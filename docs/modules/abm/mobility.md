# simengine.abm.mobility and bench task `yjmob`

Human mobility on a 200 x 200 grid (500 m cells) with 30-minute slots, as in YJMob100K (Yabe et al. 2024, Sci Data
11:397) and the HuMob Challenge 2023. The task is to say where each person is at every observed slot of 15 future
days. The module provides the HuMob metrics (GEO-BLEU and DTW), three baselines that need no model, and an
exploration and preferential return (EPR) agent simulator. The bench task `yjmob` wraps the HuMob split.

Import directly until the integrator registers it: `import simengine.abm.mobility as M`,
`from simengine.bench.tasks.yjmob import YJMob`.

## API

| name | what it does |
|---|---|
| `to_cell(x, y)`, `from_cell(c)`, `GRID`, `SLOTS`, `NC`, `MASK` | Cell code `(x-1)*200 + (y-1)` and back to 1-based `(x, y)`. `MASK` is 999, the code for a hidden coordinate. |
| `clean(df)` | Drops users with any 999 coordinate, keeps the first row of a duplicate `(uid, d, t)` and sorts the rows. |
| `geobleu(pred, truth, max_n=3, beta=0.5)` | GEO-BLEU of one pair of `(x, y)` sequences (reference `calc_geobleu_orig`, 2025 normalisation). |
| `dtw(pred, truth, scale=2.0, free_start=True)` | DTW with Euclidean cost divided by `scale`. `free_start=True` is the 2023 reference behaviour (see below). |
| `user_scores(uid, d, t, pred_xy, true_xy, variant="humob23", metrics=("geobleu","dtw"))` | Per-user HuMob score: the metric for each day, averaged over the user's days. Returns `uid, n_days, <metric>`; the task score is the column mean. Batched over days with numpy. |
| `VARIANTS` | `humob23`: N = 3, beta = 0.5, DTW in km, first DP row = 0. `humob25` (GIS Cup 2025): N = 5, beta = 0.5, DTW in km, only `[0, 0]` = 0. |
| `slot_mode(train, query, window=0, fallback=None)` | Per user, the most frequent history cell at the query slot (pools slots `t +- window`). Falls back to the user's overall mode, then to the global slot mode. |
| `global_slot_table(train)`, `global_slot_mode(train, query, table=None)` | Everybody's most frequent cell at each slot. |
| `MarkovMobility(decode="argmax" \| "sample", seed=0).fit(train).predict(query)` | First-order Markov chain per user over consecutive same-day records. Each query day starts at the slot-mode cell. |
| `EPRMobility(rho=0.6, gamma=0.21, alpha=0.55, r_min=1, r_max=200, waiting="empirical" \| "powerlaw", beta_wait=0.8, wait_max=34, day_reset=True, seed=0)` | EPR agent simulator started from each user's history. It produces one simulated trajectory. |
| `mobility_baselines(train, query, global_table=None, epr=True, markov=True, seed=0)` | `{name: cells}` for the standard set. |
| `YJMob(users=2000, seed=0, split="test", pool="all", variant="humob23", markov=True, epr=True)` | Bench task (kind `multiclass`, prediction = cell code). Metrics: `geobleu` (primary), `dtw`, `accuracy`. Each is computed per user and averaged over users with a user bootstrap (B = 1000, seed 42). |

Every predictor takes `train` (columns `uid, d, t, x, y`) and `query` (`uid, d, t`). It returns one cell per query
row, in query order.

```python
import simengine.abm.mobility as M
from simengine.bench.tasks.yjmob import YJMob

task = YJMob(users=2000, seed=0)                 # data_root or $SIMENGINE_DATA, containing yjmob100k/
data = task.load()
frame = task.run(data=data)                      # baselines -> EvalFrame (truth = cell code)
print(task.score_all(frame))                     # geobleu / dtw / accuracy per method, user-bootstrap CI
my = M.EPRMobility(rho=0.4, waiting="powerlaw", seed=1).fit(data.train).predict(data.test[["uid", "d", "t"]])
frame = task.frame(data, {"epr_tuned": my})
```

Data layout: `<root>/yjmob100k/yjmob100k-dataset1.csv.gz` (Zenodo record 10836269, CC BY 4.0, about 111.5M rows;
`.csv` also works).

## Definitions and provenance

**Split (`yjmob`).** HuMob days 1-60 are history and days 61-75 are predicted, which is `d = 0..59 -> 60..74` in the
file. `split="dev"` gives `0..44 -> 45..59` and drops rows with `d >= 60` while reading. Users with a 999 cell are
dropped. The local Zenodo copy has none, and no duplicate `(uid, d, t)` rows: 100,000 users (uid 0-99,999), and all
20,000 users with uid >= 80,000 have rows on days 61-75 (checked 22 Oct 2026). `users` people are drawn from the pool with `default_rng(seed)`. The pool is
`all` or `humob_test` (uid >= 80,000, the people HuMob 2023 task 1 masked). The table for `global_slot_mode` is
counted over the history days of every user in the file, not only the sample. It uses no future rows, and duplicate
rows count.

**GEO-BLEU** (Shimizu, Tsubouchi, Yabe, SIGSPATIAL 2022, doi:10.1145/3557915.3560951). This is a reimplementation of
the organisers' `geobleu/seq_eval.py` (github.com/yahoojapan/geobleu, MIT): commit cdaa364 (Sep 2023) for HuMob
2023 and commit d9efb0f (Aug 2025) for GIS Cup 2025. It is computed per user and per day. For n = 1..N:

- The proximity of two n-grams is `prod_k exp(-beta * ||p_k - q_k||)`, with distance in cells.
- p_n matches predicted and true n-grams greedily: the largest proximity first, each n-gram used at most once, ties
  in (predicted, true) index order. The matched sum is divided by the number of predicted n-grams. The 2023 code
  divides by the number of matches instead; the two agree when the lengths are equal, which HuMob enforces.
- The score is brevity penalty x geometric mean of p_1..p_N. It is 0 if any p_n is 0, as `scipy.stats.mstats.gmean`
  gives.

**DTW.** The cost is Euclidean distance / 2 (km), and the score is the standard DTW recursion. The 2023 reference
code sets `dtw_matrix[0] = 0`, which zeroes the whole first row: the predicted sequence may start at any step of the
true one at no cost. The published 2023 numbers were produced with that code, so `humob23` keeps it. The 2025 code
fixes it to `[0, 0] = 0` (`humob25`).

**EPR** (Song, Koren, Wang, Barabasi, Nature Physics 6:818, 2010; arXiv:1010.0436, checked against the paper):

- At a move the agent explores with probability P_new = rho * S^-gamma, where S is the number of distinct cells so
  far. The paper measured gamma = 0.21 +- 0.02 and found rho roughly normal around 0.6.
- An exploration jump has length dr ~ dr^(-1-alpha) with alpha = 0.55 +- 0.05 and a cutoff near 100 km (200 cells),
  in a random direction.
- Otherwise the agent returns to a seen cell with probability proportional to its visit count (Pi_i = f_i).
- Waiting time is P(dt) ~ dt^(-1-beta), with beta = 0.8 +- 0.1 and a 17 h cutoff (`waiting="powerlaw"`, 1..34
  slots).

The default `waiting="empirical"` and the data rules come from the TALP grid ABM for ad placement
(talp-simulation-hackathon-2026 `uretim/tahmin/kod/reklam_yeri.py`, commit 20eaac9: `hazirla`, `EprIst`,
`epr_sim`). Its parts:

- the per-user probability that two consecutive same-day records differ;
- the day start drawn from the user's observed first-of-day cells;
- stays (runs of equal cells) as visit counts;
- a ring buffer of simulated new cells that later returns can pick.

That file explored with a POI-gravity kernel over 5 x 5 super-cells. Here that kernel is replaced by the paper's
isotropic jump. The port differs from the paper in three ways. The simulator emits a cell only at the slots where
the person was observed, because HuMob scores only those. History visit counts stay fixed during the simulation.
Exploration does not reject cells the person already visited.

**Checks** (`tests/test_abm_mobility.py`, synthetic data only):

- The metrics reproduce the printed values of both reference READMEs: humob23 GEO-BLEU 0.21733678721880598 and DTW
  5.889002930255253; humob25 GEO-BLEU 0.07556369896234784.
- The batched matcher equals a literal greedy matcher on tie-heavy random sequences.
- The baselines keep query order and are seeded.
- The task runs end to end on a synthetic CSV, and its metrics refuse predictions that are not cell codes.
- Unequal-length GEO-BLEU (brevity penalty, 2025 normalisation) equals the literal greedy matcher; DTW has an exact
  case where the 2023 free start gives 0 and standard DTW 4.
- Markov follows a deterministic chain and restarts each day at the slot mode; EPR starts each day at one of the
  person's observed first-of-day cells; every predictor accepts an empty query.

A one-off script run against the downloaded reference code on 150 random users x 3 days gave a maximum absolute
difference of 1.1e-16 (GEO-BLEU) and 1.8e-15 (DTW) for both variants. It also gave 0 on 200 unequal-length pairs for
`geobleu` / `dtw` against `calc_geobleu_orig` / `calc_dtw_orig`. A review re-check the same day gave 0 on 300 more
unequal pairs (N = 3 and 5, both DTW rules) and at most 5.7e-14 for `user_scores` against `calc_geobleu_single` /
`calc_dtw_single` on 60 users with 1-9 rows per day.

**Pipeline check against a published number.** The HuMob 2023 README reports an organiser baseline: each of the
first 10,000 users stays at the centre of their days 1-60 cells, scored on days 61-75, giving GEO-BLEU 0.04195 and
DTW 62.67 (task 1). This port gives 0.04195 and 62.67 when the centre is rounded to a cell, and 0.0403 and 62.72 with
an unrounded mean. Measured 22 Oct 2026 on the local Zenodo copy (uid < 10,000, `clean`, `user_scores(...,
"humob23")`).

## Measured numbers

These are 2,000 users drawn with `default_rng(0)`, history d 0-59, test d 60-74. The pool `all` has 1,766,172
history and 464,032 test rows; `humob_test` has 1,674,256 and 436,511. Values are the mean over users with a 95%
user-bootstrap CI (B = 1000). GEO-BLEU is higher-better and DTW (km) lower-better. All are from one run on 22 Oct 2026.
Loading took 74-100 s and all baselines about 30 s. Scoring both variants took about 50 s. Peak RSS was 6.2 GB
(the full CSV is read before sampling). A second run of the same script on 22 Oct 2026 (review) reproduced every
pool-`all` value in the table to the printed digits.

Command (from the repo root, data copy under `<data-root>/yjmob100k/`):

```python
from simengine.bench.tasks.yjmob import YJMob
import simengine.abm.mobility as M
task = YJMob(users=2000, seed=0)                      # pool="humob_test" for the second table
data = task.load("<data-root>"); preds = task.baselines(data)
q = data.test[["uid", "d", "t"]]
preds["slot_mode_w2"] = M.slot_mode(data.train, q, window=2, fallback=data.meta["global_slot_table"])
preds["epr_powerlaw"] = M.EPRMobility(seed=0, waiting="powerlaw").fit(data.train).predict(q)
frame = task.frame(data, preds)
for v in ("humob23", "humob25"):
    task.options["variant"] = v; print(task.score_all(frame))
```

### Pool `all`

| method | GEO-BLEU humob23 | DTW humob23 | GEO-BLEU humob25 | DTW humob25 | exact cell |
|---|---|---|---|---|---|
| slot_mode | **0.298** [0.290, 0.305] | 51.6 [48.8, 54.7] | **0.187** [0.180, 0.194] | 77.1 [73.7, 80.6] | 0.260 |
| slot_mode, window 2 | 0.279 [0.272, 0.286] | **39.6** [37.2, 42.2] | 0.176 [0.170, 0.182] | **76.7** [73.4, 80.0] | **0.268** |
| markov_argmax | 0.208 [0.201, 0.216] | 44.0 [41.1, 46.8] | 0.133 [0.127, 0.139] | 115.7 [110.6, 120.6] | 0.203 |
| markov_sample | 0.225 [0.218, 0.232] | 69.1 [66.2, 71.9] | 0.128 [0.122, 0.134] | 107.3 [103.3, 111.4] | 0.124 |
| epr (empirical waiting) | 0.136 [0.131, 0.141] | 97.9 [94.6, 101.2] | 0.055 [0.052, 0.059] | 119.7 [115.8, 123.4] | 0.075 |
| epr (power-law waiting) | 0.137 [0.132, 0.142] | 86.7 [83.8, 89.6] | 0.066 [0.063, 0.069] | 121.4 [117.7, 125.1] | 0.075 |
| global_slot_mode | 0.004 [0.003, 0.005] | 346.0 [333.8, 358.3] | 0.002 | 406.4 | 0.003 |

EPR seeds 0 / 1 / 2 (empirical waiting): GEO-BLEU 0.1361 / 0.1358 / 0.1362 and DTW 97.9 / 97.8 / 97.8 (humob23).
The spread across seeds is far smaller than the gap to any baseline.

### Pool `humob_test` (uid >= 80,000), humob23

| method | GEO-BLEU | DTW | exact cell |
|---|---|---|---|
| slot_mode | **0.316** [0.307, 0.323] | 45.9 [43.4, 48.6] | 0.271 |
| slot_mode, window 2 | 0.298 [0.290, 0.305] | **34.5** [32.5, 36.8] | **0.280** |
| markov_argmax | 0.228 [0.221, 0.236] | 38.3 [35.8, 41.1] | 0.218 |
| markov_sample | 0.242 [0.234, 0.249] | 60.4 [58.0, 63.1] | 0.136 |
| epr (empirical) | 0.147 [0.141, 0.152] | 87.1 [84.4, 89.8] | 0.081 |
| epr (power-law) | 0.149 [0.144, 0.154] | 77.6 [75.1, 80.1] | 0.083 |
| global_slot_mode | 0.005 | 327.7 | 0.004 |

### Published HuMob 2023 task 1 numbers (not measured here)

| source | GEO-BLEU | DTW |
|---|---|---|
| GeoFormer, final assessment (Solatorio 2023, arXiv:2311.05092, Table 1; a top-3 team) | 0.3160 | 26.22 |
| GeoFormer, validation / test splits (same table) | 0.3047 / 0.3114 | 29.70 / 29.80 |
| Midterm leaderboard #1, MOBB (connection.mit.edu/humob-challenge-2023, "Midterm Evaluation Results") | 0.3212 | 40.74 |
| Organiser centre-point baseline (geobleu README, commit cdaa364) | 0.04195 | 62.67 |

The final 2023 ranking is published only as an image, so no final winner score is quoted.

### Reading

- The per-user slot mode is the bar to clear. On the HuMob test pool it scores 0.316 GEO-BLEU, level with the
  0.316 that a top-3 transformer (GeoFormer) reported for its final submission. Its DTW is much worse (45.9 vs
  26.2). A window of 2 slots trades GEO-BLEU (-0.02) for humob23 DTW (-11 km); on pool `all` the standard
  (humob25) DTW barely moves (77.1 -> 76.7).
- The comparison is not like for like. We score 2,000 sampled users and the challenge scored all 20,000. The
  challenge also allowed training on the other 80,000 users' days 61-75, which a model like GeoFormer can use and
  our baselines do not. Read it as "the simple baseline is in the same range", not as a ranking.
- The humob23 DTW lets the prediction start at any step of the true day for free (see Definitions), so it rewards a
  prediction that stays near where the day ends. Several DTW gaps below hold only under that rule; check the humob25
  (standard DTW) column before reading a DTW difference.
- First-order Markov loses GEO-BLEU to the slot mode. Our reading, not measured separately: the chain has no time of
  day, and greedy decoding tends to settle on a self-loop, which keeps DTW low and GEO-BLEU lower. Its DTW edge over
  the slot mode (44.0 vs 51.6) exists only under humob23; with standard DTW it is far worse (115.7 vs 77.1).
- EPR as a trajectory predictor is far behind every per-user baseline (about half the slot-mode GEO-BLEU, twice
  the DTW). By construction (Song et al.) it targets the statistics of mobility, such as S(t) growth and Zipf
  visit frequencies, not the time of day; those statistics were not checked here. It is here as
  a simulator for scenario and synthetic-population work and as an honest comparison, not as a forecaster.
  Power-law waiting lowers its humob23 DTW (-11 km) and leaves humob23 GEO-BLEU unchanged; under humob25 it raises
  GEO-BLEU (0.066 vs 0.055) and does not lower DTW (121.4 vs 119.7).

## Limitations

- EPR parameters are the paper's population values (rho = 0.6, gamma = 0.21, alpha = 0.55, beta = 0.8). Nothing is
  fitted per user, and nothing was tuned on the dev split.
- `EPRMobility` emits only at observed slots, keeps history visit counts fixed, and does not reject already-visited
  cells on exploration (see Definitions). The jump length is in cells and ignores land use.
- Days carry no weekday. `d % 7` is a relative weekday at best and no baseline uses it.
- The loader reads the whole CSV (peak 6.2 GB RSS) before sampling users. It is fine for one run, but cache the
  result if you load repeatedly.
- `global_slot_mode` uses every user's history days for its table. That is within the rules (no future rows) but it
  is a population statistic, not a per-sample one.
- Stochastic methods (`markov_sample`, `epr`) are scored on one sample. Averaging several samples would not form a
  valid trajectory (a mean of cells is not a cell). Report the seed spread instead, as above.
- Only task 1 (business as usual) is covered. HuMob 2023 task 2 (emergency period, dataset 2) was not downloaded.
