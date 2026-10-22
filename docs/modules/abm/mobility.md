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
dropped (none in the Zenodo file). `users` people are drawn from the pool with `default_rng(seed)`. The pool is
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
- The task runs end to end on a synthetic CSV.

A one-off script run against the downloaded reference code on 150 random users x 3 days gave a maximum absolute
difference of 1.1e-16 (GEO-BLEU) and 1.8e-15 (DTW) for both variants. It also gave 0 on 200 unequal-length pairs for
`geobleu` / `dtw` against `calc_geobleu_orig` / `calc_dtw_orig`.

**Pipeline check against a published number.** The HuMob 2023 README reports an organiser baseline: each of the
first 10,000 users stays at the centre of their days 1-60 cells, scored on days 61-75, giving GEO-BLEU 0.04195 and
DTW 62.67 (task 1). This port gives 0.04195 and 62.67 when the centre is rounded to a cell, and 0.0403 and 62.72 with
an unrounded mean. Measured 22 Oct 2026 on the local Zenodo copy (uid < 10,000, `clean`, `user_scores(...,
"humob23")`).

## Measured numbers

Pending: run of the baselines on a 2,000-user subsample.

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
