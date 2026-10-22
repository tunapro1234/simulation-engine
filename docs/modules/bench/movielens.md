# bench task `movielens`: MovieLens-1M rating and next-item prediction

`simengine/bench/tasks/movielens.py` turns MovieLens-1M (GroupLens: 1,000,209 ratings from 1 to 5 stars, 3,706 rated
movies, 6,040 users with gender, age bracket, occupation and zip code, April 2000 to February 2003) into two bench
targets. Both use a per-user temporal split. The module also ships LLM-free baselines and a helper that writes a
persona text from a user's own training ratings, so an LLM method can be compared with the baselines on the same rows.

The task follows the `BenchTask` conventions in `simengine/bench/base.py` (`layout`, `load`, `baselines`, `frame`,
`metrics`). It is not in the package registry yet, so import the module directly. Importing it registers the name
`movielens`.

## API

```python
from simengine.bench.tasks import movielens as ml

task = ml.MovieLens1M(target="rating")                  # kind continuous; metrics rmse (primary), mae
data = task.load("<catalogue>")                         # or $SIMENGINE_DATA; the directory holding movielens-1m/
frame = task.run(data=data)                             # EvalFrame with the baselines (+ methods={...})
task.score(frame, "item_knn")                           # {'metric': 'rmse', 'value', 'lo', 'hi', 'n': users}
task.score_all(frame)

nxt = ml.MovieLens1M(target="next_item")                # kind binary; metrics hit10 (primary), ndcg10
d2 = nxt.load("<catalogue>")
f2 = nxt.run(data=d2)
ml.full_ranking(d2, nxt.scorer(d2, "item_knn"))         # hit@10 / ndcg@10 against every unrated movie

ml.persona_text(data, user_id)                          # profile text from train ratings only
ml.personas(data)                                       # {user_id: text} for all train users
ml.item_text(data, movie_id)                            # '<title (year)> - genres: A, B'
```

| option | values (default) | used by |
|---|---|---|
| `target` | `"rating"` \| `"next_item"` (`"rating"`) | both |
| `split` | `"test"` \| `"dev"` (`"test"`) | both |
| `test_frac` | float in (0, 1) (0.2) | rating |
| `n_candidates` | int ≥ 2 (100) | next_item |
| `seed` | int (0); seeds negative sampling and the `random` baseline | next_item |
| `lam` | float (5); ridge weight of the bias model | rating |
| `knn_k` | int (40); neighbours of the rating item-kNN | rating |
| `next_k` | int (50; 0 = all); similar movies kept per movie by the next-item item-kNN | next_item |

An unknown option key raises `ValueError`, as in every bench task.

Other public names:

- **Files and split:** `find_dir`, `read_ml1m`, `order_by_user_time`, `tail_mask`.
- **Fitting:** `fit_biases`, `residual_similarity`, `cooccurrence_cosine`, `item_knn_ratings`.
- **Next-item scorers:** `popularity_scorer`, `item_knn_scorer`, `random_scorer`, `pair_scores`.
- **Metrics:** `expected_topk`, `topk_per_user`, `full_ranking`, `RATING_METRICS`, `NEXT_METRICS`.
- **Code tables:** `AGE`, `OCCUPATION`, `GENDER`.

### Files

`<root>/movielens-1m/ml-1m/{ratings,users,movies}.dat` is the GroupLens `ml-1m.zip` extracted (`::`-separated,
latin-1). The `.dat` files may also sit directly in `<root>/movielens-1m/` or in `acik/ml-1m/` below it.
`ratings.dat` and `users.dat` are parsed with the C parser (about 1 s for the 1M ratings). `movies.dat` is parsed with
the Python parser, because titles contain `:`.

### Split

Each user's ratings are ordered by `(timestamp, movie_id)`. ML-1M has many ratings stamped in the same second, so
the movie id is a deterministic tie-break, not real time order.

- **`rating`:** the last `ceil(test_frac · n_u)` ratings of every user are the test set, and the rest are train. On
  the real data that gives 797,758 train and 202,451 test ratings, from all 6,040 users.
- **`next_item`:** leave-last-out. The user's last rating is the target, and all earlier ratings are train (994,169
  rows). The test rows for each user are the target (truth 1) plus `n_candidates − 1` negatives. Negatives are drawn
  uniformly, without replacement, from the movies present in train that the user never rated anywhere in the file.
  The draw uses `default_rng(seed)`, and each user's candidate order is shuffled so the target's position carries no
  signal. On the real data that is 6,040 × 100 = 604,000 rows.
- **`split="dev"`:** removes the test part first, then applies the same rule again inside train. The dev split never
  reads a test rating as truth.

The EvalFrame columns are `unit` (`user|movie`), `group` (user, the bootstrap cluster), `item` (movie), `time` (the
timestamp of the rating; for next_item, the target's), `truth`, `segment` (age bracket), `seg_gender` and `split`.

### Metrics

- **rating:** RMSE (primary) and MAE over all test ratings. The confidence interval comes from a user-cluster
  bootstrap (B = 1000, `default_rng(42)`; `n` = users).
- **next_item:** hit@10 (primary) and NDCG@10 of the target among the user's candidates, averaged over users, with a
  user bootstrap. Ties are scored in expectation: if `g` candidates score above the target and `t` tie with it, the
  rank is uniform on `g+1 … g+t+1`. A constant score therefore gets hit@10 = 10 / `n_candidates` exactly, and a method
  cannot gain from tie order.
- **`full_ranking(data, scorer)`:** ranks the target against every train movie the user has not rated (3,704 movies
  on the real data). A target that never appears in train counts as a miss.

Report the full-ranking numbers alongside the sampled ones. Krichene & Rendle (KDD 2020, "On Sampled Metrics for
Item Recommendation") show that sampled metrics can disagree with exact metrics on the order of methods. The sampled
metric is kept as the primary metric because an LLM can only read a short candidate list.

### Baselines

**Rating:**

- `global_mean`, `user_mean` and `item_mean`, taken from train. Users or movies with no train ratings fall back to
  the global mean.
- `bias`: r̂ = μ + b_u + b_i, from the regularised least-squares objective of Koren (KDD 2008, "Factorization meets
  the neighborhood", sec. 2.1, λ₁). It is solved by exact alternating block updates (20 sweeps). `lam = 5` was chosen
  on `split="dev"` from {1, 5, 10, 25, 50}. The dev RMSE was 0.9128 / 0.9126 / 0.9137 / 0.9197 / 0.9306.
- `item_knn`: Koren 2008 eq. 3, r̂ = b_ui + Σ s_ij (r_uj − b_uj) / Σ s_ij over the k = 40 most similar movies the
  user rated in train, using positive similarities only. The similarity is eq. 2,
  s_ij = n_ij / (n_ij + 100) · ρ_ij. Here ρ_ij is the uncentred correlation of bias residuals over co-raters, not the
  raw-rating Pearson in the paper. k = 40 was compared with 20 and 80 on dev, with residuals from a λ = 10 bias fit
  (RMSE 0.8829 / 0.8807 / 0.8820).
  Predictions are clipped to [1, 5].

**Next item:**

- `random`: a uniform random score.
- `popularity`: the number of train ratings of the movie.
- `item_knn`: score(u, i) = Σ over u's train movies j of cos(i, j). The cosine is taken on the binary user × movie
  matrix, and each movie keeps its `next_k` most similar movies. `next_k = 50` was chosen on dev from {0 (all), 50,
  200}, with dev hit@10 of 0.534 / 0.684 / 0.656.

Scorers are factories (`task.scorer(data, name)`, or `item_knn_scorer(data, k)`) that fit on `data.train` once and
return `f(user_ids, movie_ids) -> matrix`. Any method can implement the same signature and be passed to `pair_scores`
or `full_ranking`.

### Persona text for LLM methods

`personas(data)` / `persona_text(data, user_id)` read only `users.dat` and `data.train`, never `data.test`. A
profile contains:

- gender, age bracket and occupation, decoded from the README codes, with the zip code left out;
- the number of ratings and their mean;
- up to 5 favourite genres by mean rating, counting only genres with at least 3 ratings;
- the most recently liked movies (4-5 stars) and disliked movies (1-2 stars), newest first, with their stars.

`demographics=False` drops the first sentence, and `n_liked`, `n_disliked` and `n_genres` set the list lengths. The
shape, with invented content:

```
Gender: female. Age: 25-34. Occupation: programmer. Has rated 142 movies, mean 3.62 out of 5. Favourite genres (mean
rating, count): Film-Noir 4.5 (4); Drama 4.1 (40); ... Recently liked (newest first): <title (year)> [5]; ...
Recently disliked (newest first): <title (year)> [1]; ...
```

The tests run an LLM method end to end through `FakeLLM`: persona text plus `item_text`, one prompt per test row,
parsed to a number, framed and scored.

## Measured numbers

These are from runs on the local MovieLens-1M copy (22 Oct 2026, `nice -n 19`, `OMP_NUM_THREADS=4`). Every value is shown as `value [95% CI]`, n = 6,040 users. Command (Python, from the repository
root):

```python
from simengine.bench.tasks import movielens as ml
for target in ("rating", "next_item"):
    task = ml.MovieLens1M(target=target)                      # split="dev" for the dev rows
    data = task.load("<catalogue>")
    print(task.score_all(task.run(data=data)))
    if target == "next_item":
        for name in ("popularity", "item_knn", "random"):
            print(name, ml.full_ranking(data, task.scorer(data, name)))
```

**Rating, test split** (202,451 test ratings; about 7 s end to end, 1.0 GB peak RSS):

| baseline | RMSE | MAE |
|---|---|---|
| global_mean | 1.1464 [1.1332, 1.1600] | 0.9438 [0.9331, 0.9542] |
| user_mean | 1.0743 [1.0641, 1.0848] | 0.8491 [0.8403, 0.8580] |
| item_mean | 0.9897 [0.9785, 1.0006] | 0.7871 [0.7784, 0.7960] |
| bias | 0.9210 [0.9119, 0.9300] | 0.7239 [0.7167, 0.7309] |
| item_knn | **0.8827** [0.8747, 0.8902] | **0.6894** [0.6828, 0.6958] |

Dev split with the final defaults (162,035 dev ratings): global_mean 1.1198, user_mean 1.0463, item_mean 0.9847,
bias 0.9126 [0.9035, 0.9218], item_knn 0.8797 [0.8710, 0.8884].

**Next item, test split** (100 candidates; about 12 s, 0.9 GB peak RSS):

| baseline | hit@10 sampled (100 cand.) | NDCG@10 sampled | hit@10 full ranking (3,704 movies) | NDCG@10 full |
|---|---|---|---|---|
| random | 0.0990 [0.0917, 0.1066] | 0.0448 [0.0412, 0.0484] | 0.0038 [0.0025, 0.0055] | 0.0021 [0.0012, 0.0032] |
| popularity | 0.4392 [0.4271, 0.4522] | 0.2400 [0.2322, 0.2480] | 0.0353 [0.0311, 0.0401] | 0.0174 [0.0151, 0.0202] |
| item_knn | **0.6561** [0.6447, 0.6681] | **0.3825** [0.3742, 0.3909] | **0.0715** [0.0656, 0.0785] | **0.0359** [0.0323, 0.0398] |

Dev split with the final defaults (sampled hit@10): random 0.0960, popularity 0.4592 [0.4469, 0.4714], item_knn
0.6839 [0.6720, 0.6960]. The dev split was used to choose `next_k`, so it is optimistic for item_knn.

The random row matches its expected values (hit@10 = 0.1 sampled, 10 / 3,704 ≈ 0.0027 full) within its CI. The
order of the three baselines is the same under both protocols here. The absolute level is not: item_knn finds the
next movie in its top 10 in 66% of users against 100 sampled candidates, but in 7% against the full catalogue.

## Limitations

- **Cross-user leakage:** the split is per user, not a global time cut. Train contains other users' ratings made after
  a given user's test ratings, so popularity and kNN statistics see some of the future. A global-time split would be
  stricter, and is not implemented.
- **Same-second ties:** within a burst of ratings in the same second, the "next" movie is the one with the highest
  movie id. Sequence-aware methods should treat the order inside a second as arbitrary.
- **Sampled metric:** hit@10 over 100 sampled candidates is easier than, and may rank methods differently from, the
  full-ranking metric. Report both.
- **Item-kNN details:** the item-kNN uses an uncentred residual correlation, not Koren's Pearson on raw ratings, and
  `lam`, `knn_k` and `next_k` were each chosen on one dev split by a small grid. No matrix factorisation baseline is
  included here (`se.MFBaseline` exists in `simengine.methods`).
- **Random-split numbers:** published ML-1M results that use random splits are not comparable with these numbers.
  Temporal tails are harder.
- **Licence:** GroupLens terms allow research use only, require citation (Harper & Konstan 2015, ACM TiiS), and
  forbid redistribution or commercial use without permission. The repository ships no rows. Do not ship the data or
  per-row predictions with a submission, and check with the organiser before a prize-bearing use.
