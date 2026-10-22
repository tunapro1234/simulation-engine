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
  which equals hit rate because there is one target, MRR@20 (reciprocal rank, 0 outside the top 20), and recall@20
  on `new` rows (targets not seen in the input). A list is ranked after dropping repeated ids (first position counts)
  and negative ids (padding), as the source metric does.
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

- `popularity`: the items with the most events (all three types) in history.
- `last_viewed`: the input's earlier items, most recent first.
- `cooc`: items that co-occur with the input's last item in history sessions. Each history session contributes its
  first 50 distinct items, and items are ranked by co-occurrence count.
- `transition`: counts of "last item -> next different item" in history sessions, then `cooc`, then popularity.

## Measured numbers

These come from runs on the local RetailRocket copy (22 Oct 2026, `nice -n 19`, `OMP_NUM_THREADS=4`, numpy 2.5.4,
pandas 2.3.3, scikit-learn 1.9.1, lightgbm 4.7.0). Every value is shown as `value [95% CI]` (visitor bootstrap).
Command, run from the repository root with `PYTHONPATH=.`:

```python
from simengine.bench.tasks.retailrocket import RetailRocketPurchase, RetailRocketNext
for task in (RetailRocketPurchase(split="test"), RetailRocketNext(split="test")):   # also split="valid", history="all"
    data = task.load("<catalogue>")
    print(task.score_all(task.run(data=data)))
```

**`retailrocket_next`, test** (23,517 rows, 20,724 visitors; `new` rows 19,880; about 12 s end to end, 1.3 GB peak
RSS):

| baseline | recall@20 | MRR@20 | recall@20 on new targets |
|---|---|---|---|
| popularity | 0.0075 [0.0064, 0.0088] | 0.0016 [0.0012, 0.0021] | 0.0059 [0.0047, 0.0072] |
| last_viewed | 0.1592 [0.1549, 0.1645] | 0.1353 [0.1313, 0.1399] | 0.0058 [0.0046, 0.0071] |
| cooc | **0.2973** [0.2906, 0.3038] | **0.1367** [0.1328, 0.1408] | **0.2881** [0.2810, 0.2951] |
| transition | 0.2923 [0.2861, 0.2991] | 0.1307 [0.1269, 0.1345] | 0.2830 [0.2762, 0.2898] |

Valid split (22,663 rows): popularity 0.0096, last_viewed 0.1657, cooc 0.3315 [0.3245, 0.3379] (MRR 0.1538),
transition 0.3261 (MRR 0.1455).

With `history="all"` on test (item tables built from every event before 2015-09-04): cooc 0.3343 [0.3274, 0.3413]
(MRR 0.1551), transition 0.3279, last_viewed 0.1601, popularity 0.0086. The extra two weeks of history raise
co-occurrence recall by 3.7 points, so compare methods only under the same `history` setting.

**`retailrocket_purchase`, test** (34,223 sessions, 30,179 visitors, 1,013 positives; about 4.3 min end to end, of
which 33 s are loading and baselines and the rest is the 1000-draw bootstrap of 18 metric cells; 1.35 GB peak RSS;
LightGBM refit with 426 rounds):

| baseline | PR-AUC | AUC | log loss |
|---|---|---|---|
| constant | 0.0296 [0.0263, 0.0328] | 0.5000 | 0.1334 [0.1219, 0.1443] |
| has_addtocart | 0.2639 [0.2376, 0.2881] | 0.9336 [0.9254, 0.9411] | 0.0717 [0.0668, 0.0766] |
| last_item_conversion | 0.0563 [0.0485, 0.0649] | 0.6272 [0.6089, 0.6454] | 0.1464 [0.1309, 0.1611] |
| counts_logistic | 0.3697 [0.3002, 0.4405] | 0.9448 [0.9354, 0.9529] | 0.0815 [0.0764, 0.0870] |
| lgbm | **0.5727** [0.4936, 0.6402] | **0.9699** [0.9644, 0.9747] | **0.0605** [0.0569, 0.0643] |
| lgbm_no_end_position | 0.5090 [0.4234, 0.5816] | 0.9658 [0.9599, 0.9707] | 0.0642 [0.0606, 0.0682] |

Valid split (33,858 sessions, positive rate 0.0320; PR-AUC / AUC): has_addtocart 0.2642 / 0.9306, counts_logistic
0.3838 [0.3101, 0.4527] / 0.9354, lgbm 0.5656 [0.4900, 0.6323] / 0.9674, lgbm_no_end_position 0.5248 / 0.9632. The
split is the same as test's train side, so the LightGBM round count (426) is the same.

Each task's test split was run twice, at about 20:35 and 21:30 on 22 Oct. The first runs were on a loaded machine
(18 min for purchase, 5 min for next item). Both runs gave identical numbers to 6 decimals, so the pipeline is
deterministic on this machine. The PR-AUC intervals are wide (about ±0.07 for lgbm) because there are only 1,013
positives. Compare methods with a paired test (`se.evaluate`), not by overlapping intervals.

Comparison with the source branch's tables (`BASELINE_RR.md`, 20 Oct 2026; the same definitions run by its own
scripts): the split and all counts are identical, next item matches to 4 decimals on test and valid, and
`has_addtocart` matches exactly (test 0.2639, valid 0.2642). `counts_logistic` is 0.3697 on test here against 0.3698
there (valid 0.3838 in both). The cause is the duration features: the source's polars `total_seconds()` truncates to
whole seconds, while this port keeps milliseconds. With durations truncated to whole seconds, a review run (22 Oct)
reproduced the source's test PR-AUC to 6 decimals (0.369817).

The source lgbm had PR-AUC 0.5715 on test and 0.5674 on valid (375 refit rounds), and 0.5101 / 0.5280 without
end-position features. The same review measured how much small implementation details move the test lgbm PR-AUC
(predictions only, no bootstrap): whole-second durations 0.5733 (328 rounds); plus the source's fold assignment
(polars `hash(seed=11) % 5` instead of `visitorid % 5`) 0.5705 (377 rounds); plus train rows ordered by session start
0.5708 (360 rounds). The port as shipped gives 0.5727 (426 rounds). So lgbm moves by about ±0.002 PR-AUC and its
round count by about ±50 with these choices. That is far inside the ±0.07 interval, and the exact source value was
not recovered.

## Provenance

Ported from the TALP preparation research (intent branch: `hazirla_rr.py` for the split, `baseline_rr.py` for the
baselines, `sonraki_metrik.py` for the next-item metric; protocol "RR-oturum" with the leak fix recorded in
`BASELINE_RR.md`). The data is the Retail Rocket "Retailrocket recommender system dataset" on Kaggle, licensed
CC BY-NC-SA 4.0. The copy used for the numbers above came from a Hugging Face mirror of the Kaggle files (events
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
- **Single configuration:** each number comes from one fixed configuration, run twice with identical results. LightGBM
  uses 4 threads with a fixed seed. Its round count is chosen inside train, and small implementation choices move it
  and the lgbm scores slightly (see the comparison with the source above).
- **Not comparable with published session-recommendation numbers:** those use other session definitions, filters
  (minimum item support, session length), splits and candidate sets. Results reported elsewhere on RetailRocket
  should not be placed next to these numbers without rerunning them on this split.
- **Licence:** CC BY-NC-SA 4.0 is non-commercial and share-alike. Do not ship the data or per-row predictions with a
  submission, and check with the organiser before a prize-bearing use.
