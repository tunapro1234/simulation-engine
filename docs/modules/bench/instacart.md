# bench task `instacart`: next-basket reorder prediction

`simengine/bench/tasks/instacart.py` turns the Instacart Online Grocery Shopping Dataset 2017 (the file layout of the
Kaggle "Instacart Market Basket Analysis" competition: about 3.4M orders from 206k users, 50k products) into a
binary bench task. The question is the competition's: of the products a shopper has bought before, which will be in
their next order? The module ships LLM-free baselines (three reorder rates and a LightGBM model on the usual
features) so an LLM or persona method can be compared with them on the same rows.

The task follows the `BenchTask` conventions in `simengine/bench/base.py` (`layout`, `load`, `baselines`, `frame`,
`metrics`). It is not in the package registry yet, so import the module directly. Importing it registers the name
`instacart`.

## API

```python
from simengine.bench.tasks.instacart import Instacart

task = Instacart(n_users=20000, seed=0)                 # kind binary; metrics f1 (primary), auc
data = task.load("<catalogue>")                         # or $SIMENGINE_DATA; the directory holding instacart/
frame = task.run(data=data)                             # EvalFrame with the baselines (+ methods={...})
task.score(frame, "lgbm")                               # {'metric': 'f1', 'value', 'lo', 'hi', 'n': users}
task.score(frame, "lgbm", "auc")
task.score_all(frame)
```

| option | values (default) | meaning |
|---|---|---|
| `n_users` | int \| None (None) | None = every user whose last order is labelled (131,209); an int samples that many with `default_rng(seed)` |
| `seed` | int (0) | user sample, LightGBM seed and its early-stopping split |
| `threshold` | float (0.2) | F1 decision rule: a product is predicted when its score is ≥ threshold |
| `lgbm` | bool (True) | add the LightGBM baseline; skipped silently when `lightgbm` is not installed |

An unknown option key raises `ValueError`, as in every bench task.

`data.train` and `data.test` are candidate tables with one row per (user, product the user bought before):
`user_id`, `product_id`, `y` and the 23 columns in `FEATURES`. `data.meta` holds the split description, the user
ids, the product table with department names (`products`), the prior order lines (`history`, so a method can build its
own features or a persona text) and the base rates.

Other public names: `read_table`, `history`, `order_days`, `build_table` (one candidate table for one cut),
`per_user_f1`, `weighted_auc_setup`, `weighted_auc`, `FILES`, `FEATURES`, `LGBM_PARAMS`, `THRESHOLD`.

### Files

```
<root>/instacart/orders.parquet|csv                  order_id, user_id, eval_set, order_number, order_dow,
                                                     order_hour_of_day, days_since_prior_order
<root>/instacart/order_products__prior.parquet|csv   order_id, product_id, add_to_cart_order, reordered
<root>/instacart/order_products__train.parquet|csv   same columns; the labelled last orders
<root>/instacart/products.parquet|csv                product_id, product_name, aisle_id, department_id
<root>/instacart/aisles.parquet|csv, departments.parquet|csv
```

The Kaggle CSVs work as they are; a Parquet file with the same name is preferred when both exist. Files are read in
row groups / 2M-row chunks with integer columns downcast, so the 32M-line prior file takes about 0.8 GB in memory.

### Split

Users are those whose last order has `eval_set == "train"` (the Kaggle `test` orders have no labels and are not
used). Candidates for a user are the distinct products in their `prior` orders.

- **test:** each user's last order is the label. A candidate's truth is 1 when the product is in that order.
  Features use all `prior` orders. Products in the last order that the user never bought before are not candidates
  (they are not reorders).
- **train:** the same users one order earlier. Each user's last `prior` order is the label, and features use only the
  orders before it, for every user, including the product statistics. No test label is read when building it.

### Features (`FEATURES`)

- user: number of orders, items, distinct products, mean basket size, reorder rate (orders after the first), mean
  days between orders;
- product, over all users' history before the cut: line count, buyer count, reorder rate, share of buyers who bought
  it at least twice, mean add-to-cart position;
- user × product: times bought, share of the user's orders, share since first purchase, orders and days since last
  purchase, mean cart position, bought in the most recent order;
- label order: days since the prior order, day of week, hour (the competition gives these for the order to predict);
- `aisle_id`, `department_id` (categorical in LightGBM).

### Metrics

- **`f1` (primary):** mean over users of the F1 between the predicted set {score ≥ `threshold`} and the set of
  reordered products. A user with no reorders scores 1 when nothing is predicted and 0 otherwise; a user with
  reorders and an empty prediction scores 0.
- **`auc`:** ROC AUC pooled over all candidate rows (ties count 0.5).

Both use a user-cluster bootstrap for the 95% CI (B = 1000, `default_rng(seed)` with the `score` default seed 42,
implemented as resampling weights; `n` = users).

### Baselines

- `user_reorder_rate`: the user's share of reordered lines in orders after their first. One number per user, so it
  ranks users, not products within a user.
- `product_reorder_rate`: the product's share of reordered lines over all users.
- `user_product_rate`: times the user bought the product / the user's number of orders.
- `lgbm`: LightGBM binary model on `FEATURES`, trained on the train cut with `LGBM_PARAMS` (learning rate 0.1, 63
  leaves, 4 threads), early stopping (50 rounds, at most 1000) on 10% of users held out of the train cut. The
  prediction is the probability.

All baselines go through the same fixed threshold. No baseline tunes a per-user cut-off.

## Measured numbers

Command (Python, from the repository root, `nice -n 19`, `OMP_NUM_THREADS=4`):

```python
from simengine.bench.tasks.instacart import Instacart
task = Instacart(seed=0)                                      # all 131,209 labelled users
data = task.load("<catalogue>")
frame = task.run(data=data)
print(task.score_all(frame, B=1000))
for th in (0.1, 0.15, 0.2, 0.25, 0.3):                        # threshold sweep, point values only
    print(th, {m: Instacart(threshold=th).score(frame, m, B=10)["value"] for m in ("user_product_rate", "lgbm")})
```

## Limitations

- **Fixed threshold, no F1 optimisation.** Strong competition solutions choose each user's set (and whether to add
  "None") by maximising expected F1 under the predicted probabilities. Here every method uses one global threshold.
  The number is therefore a floor for a given probability model, and methods that output calibrated probabilities
  are compared fairly only at the same threshold.
- **"None" handling differs from the competition.** The competition scored "None" as an extra item, so a user could
  predict products and "None" together and get partial credit. Here a user with no reorders scores 1 only for an
  empty prediction. Numbers are not directly comparable with the Kaggle leaderboard, which was also computed on the
  hidden `test` orders, not on the `train` orders used here.
- **One train cut.** LightGBM learns from each user's last prior order only. Using more cuts per user, or the
  competition's usual setup of training on the `train` orders of other users, would give more training rows; not
  done.
- **Candidate set.** Only previously bought products are candidates, so first-time purchases are out of scope by
  design.
- **Licence:** the Instacart dataset was released for non-commercial use under Instacart's terms; the local copy is a
  Hugging Face mirror of the Kaggle files. The repository ships no rows. Do not ship the data or per-row predictions
  with a submission, and check with the organiser before a prize-bearing use.
