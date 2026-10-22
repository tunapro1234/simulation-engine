# bench tasks `retailrocket_purchase` and `retailrocket_next`: RetailRocket session tasks

`simengine/bench/tasks/retailrocket.py` turns the RetailRocket e-commerce event log (view / addtocart / transaction,
2,756,101 events from 1,407,580 visitors, 2015-05-03 to 2015-09-18) into two session-level tasks on one temporal
split:

| task | kind | question | primary metric |
|---|---|---|---|
| `retailrocket_purchase` | binary | will this session end in a purchase, given its events before the first transaction? | PR-AUC |
| `retailrocket_next` | multiclass (ranked item lists) | which item does the visitor open next? | recall@20 |

Both follow the `BenchTask` conventions in `simengine/bench/base.py` (`layout`, `load`, `baselines`, `frame`,
`metrics`). The package `__init__` does not import the module yet, so import it directly. Importing it registers both
names, after which `bench.get("retailrocket_purchase")` works.

## API

```python
from simengine.bench.tasks.retailrocket import RetailRocketPurchase, RetailRocketNext

task = RetailRocketPurchase()                     # options: split, history, lgbm
data = task.load("<catalogue>")                   # or $SIMENGINE_DATA; reads <root>/retailrocket/events.csv
frame = task.run(data=data, methods={"mine": lambda d: my_probs(d)})   # one probability per data.test row
task.score(frame, "mine")                         # {'metric': 'pr_auc', 'value', 'lo', 'hi', 'n', 'n_visitors', ...}
task.score_all(frame)                             # every method x (pr_auc, auc, logloss)

nxt = RetailRocketNext()                          # options: split, history, k
d2 = nxt.load("<catalogue>")
f2 = nxt.run(data=d2)                             # predictions = one ranked item-id list per data.test row
nxt.score_all(f2)                                 # recall@20, mrr@20, recall@20_new
```

| option | values (default) | used by |
|---|---|---|
| `split` | `"test"` \| `"valid"` (`"test"`) | both |
| `history` | `"train"` (item tables from events before 2015-08-21) \| `"all"` (every event before the evaluation window) (`"train"`) | both |
| `lgbm` | bool (`True`; skipped silently if lightgbm is not installed) | purchase |
| `k` | int (20); renames the metrics to `recall@k`, `mrr@k`, `recall@k_new` | next |

An unknown option key raises `ValueError`.

Free functions: `read_events`, `sessionize`, `visitor_history`, `prefix_features`, `item_stats`, `item_features`,
`next_item_targets`. Constants: `EVENT_ORDER`, `GAP_MS`, `VALID_START`, `TEST_START`, `LGBM_FEATURES`,
`END_POSITION`, `LGBM_PARAMS`. `RetailRocketPurchase.features(data)` returns the (train, test) feature tables that
the baselines use, which a stacked method can reuse.

### Files

`<root>/retailrocket/events.csv` is the Kaggle "Retailrocket recommender system dataset" events file. The columns
read are `timestamp` (ms, UTC), `visitorid`, `event` and `itemid`. `item_properties` and `category_tree` are not
used. An event type other than the three known ones raises `ValueError`.

### Sessions and split

- Events are sorted by visitor, timestamp, event order (view < addtocart < transaction), item id, then file row. A
  gap of more than 30 minutes starts a new session. `session_id` is `"<visitorid>_<n>"`, with `n` counted from 0
  per visitor.
- **Prefix:** the session's events before its first transaction, or all of its events if it has none. A session is
  evaluated only when its prefix has at least 2 events. The original protocol filtered on "at least 2 events in the
  session". Under that rule, every session with a prefix of length 1 was a positive (a negative needs 2 events to be
  kept, so a 1-event prefix can only come from "view, transaction"). That leak was 12.7% of the positives in the
  source analysis. It is fixed here by applying the filter to the prefix.
- **Split by session start (UTC):** train before 2015-08-21, valid from 2015-08-21 to before 2015-09-04, and test
  from 2015-09-04. Train rows always come from the train window. `split` picks valid or test as `data.test`.
- **Purchase:** `y = 1` if the session has at least one transaction. The model sees only the prefix
  (`data.meta["prefix"]`) and the `g_*` columns. These are totals over the same visitor's earlier sessions. Sessions
  never overlap, so these totals cannot include the current session.
- **Next item:** for evaluated sessions whose prefix has at least 2 runs of distinct consecutive items, the last run
  is held out. The target is that run's item. The input (`data.meta["input"]`) is the prefix up to the event just
  before that run. The target is therefore never the input's last item, but it may have appeared earlier in the
  input (`repeat = 1`, about 15% of test rows). Sessions that have no such target are dropped.
- `data.meta["history"]` holds every event before the history cut, including transactions. It is meant for item and
  visitor statistics. For train rows, use out-of-fold statistics, as the baselines do (5 folds by `visitorid % 5`).

Counts on the local copy (asserted by `TestRealData`): 1,761,675 sessions; 312,600 train / 34,223 test purchase rows
(1,013 test positives, rate 0.0296; train rate 0.0306). For the next-item task: 216,260 train rows. Test has 23,517
rows out of 34,223 evaluated sessions, with a mean input of 2.67 events and a repeat rate of 0.155. Valid has 22,663
rows out of 33,858 sessions.

### EvalFrame and metrics

- **Purchase:** the frame has `unit` (session_id), `group` (visitorid), `truth` (y), `time`, `segment` (prefix length
  `2` / `3-4` / `5+`) and `seg_addtocart`. The metrics are PR-AUC (average precision, step-wise with tied scores
  treated as one threshold, equal to sklearn's `average_precision_score`), ROC AUC, and log loss clipped at 1e-15.
- **Next:** the frame has `unit`, `group`, `truth` (target item id), `item` (the input's last item), `time`,
  `segment` (`repeat` / `new`) and `seg_input_len`. `pred_<method>` cells hold int arrays. The metrics are recall@20,
  which equals hit rate because there is one target, MRR@20, and recall@20 on `new` rows (targets not seen in the
  input).
- All confidence intervals come from a visitor-cluster bootstrap (B = 1000, seed 42). `n` is rows and `n_visitors`
  is clusters.

## Baselines

**Purchase** (no LLM):

- `constant`: the train positive rate.
- `has_addtocart`: the train positive rate, conditioned on whether the prefix has an addtocart event (0.0022 vs 0.279
  in the source analysis).
- `last_item_conversion`: the smoothed transaction-per-view rate of the prefix's last item. It uses a prior of 20
  pseudo-views at the global rate.
- `counts_logistic`: a logistic regression on log1p of 7 count features (events, views, addtocarts, distinct items,
  duration, earlier sessions, earlier transactions), with standardised inputs.
- `lgbm`: LightGBM on 29 features (`LGBM_FEATURES`). These add time-of-day, end-position, visitor-history and
  item-rate features to the 7 counts. Item rates for train rows are out-of-fold by visitor. Early stopping uses the
  last 14 days of train (from 2015-08-07, so valid is never touched). The model is then refit on all of train with
  1.1 times the best round count.
- `lgbm_no_end_position`: the same model without the 4 `END_POSITION` features (`last_is_atc`, `events_after_atc`,
  `secs_after_atc`, `item_conv_last`), with the same round count.

**Next item:** the input's last item is never proposed. Each list is padded with the most popular items.

- `popularity`: the most viewed items in history.
- `last_viewed`: the input's earlier items, most recent first.
- `cooc`: items that co-occur with the input's last item in history sessions. Each history session contributes its
  first 50 distinct items, and items are ranked by co-occurrence count.
- `transition`: counts of "last item -> next different item" in history sessions, then `cooc`, then popularity.

## Measured numbers

See the next revision of this file (runs on the local copy).

## Provenance

Ported from the TALP preparation research (intent branch: `hazirla_rr.py` for the split, `baseline_rr.py` for the
baselines, `sonraki_metrik.py` for the next-item metric; protocol "RR-oturum" with the leak fix recorded in
`BASELINE_RR.md`). The data is the Retail Rocket "Retailrocket recommender system dataset" on Kaggle, licensed
CC BY-NC-SA 4.0. The copy used for the numbers below came from a Hugging Face mirror of the Kaggle files (events
and category tree only); its row count matches the Kaggle file (2,756,101). The repository ships no rows.

## Limitations

- **Prefix-cut artefact:** a positive prefix ends right before the purchase, while a negative session is complete.
  Features such as "last event is addtocart" and "time since addtocart" therefore partly encode the definition
  rather than intent. This is why `lgbm_no_end_position` is reported. A fixed-cut task (for example, the first 3
  events or the first 5 minutes) would support a real-time intent claim and is not implemented.
- **Time zone:** days in the file roll over at 03:00 UTC (the shop was most likely in UTC+3). The split boundaries
  are in UTC as defined. `hour` / `weekday` features are UTC.
- **No categories:** the item-to-category mapping is in `item_properties`, which is not read, so there are no
  category features or category-level baselines.
- **Single runs:** each test number is one run. LightGBM uses 4 threads with a fixed seed. Its round count is chosen
  inside train and can differ slightly between versions.
- **Not comparable with published session-recommendation numbers:** those use other session definitions, filters
  (minimum item support, session length), splits and candidate sets. Results reported elsewhere on RetailRocket
  should not be placed next to these numbers without rerunning them on this split.
- **Licence:** CC BY-NC-SA 4.0 is non-commercial and share-alike. Do not ship the data or per-row predictions with a
  submission, and check with the organiser before a prize-bearing use.
