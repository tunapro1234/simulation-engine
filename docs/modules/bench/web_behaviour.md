# simengine.bench.tasks.web_behaviour

Human web-shopping behaviour, step by step. Two registered `BenchTask`s built from the WebShop human trajectories
(MTurk workers buying an instructed product on a simulated Amazon-like site):

| task | kind | row | truth | prediction |
|---|---|---|---|---|
| `webshop_next_action` | multiclass | every action after a session's first | which of 10 actions the shopper takes next | (n, 10) probabilities, columns in `ACTIONS` order |
| `webshop_buy` | binary | actions taken on a product page | the action is `buy` (Buy Now on this product) | one probability per row |

For TALP this is the "simulated site visitor" check: a visitor simulator, LLM agent or persona panel has to say what a
real person does next on a shop page, and how likely they are to buy now, from the same information a human had
(instruction, own clicks so far, current page). It is scored against what people did, next to chains that only count
what people did before.

## API

```python
import simengine as se
from simengine import bench
from simengine.bench.tasks import web_behaviour as wb   # registers both tasks (until the package __init__ imports it)

task = bench.get("webshop_next_action")                 # options: split, test_frac, seed, uxagent
data = task.load("/path/to/catalogue")                  # or $SIMENGINE_DATA
frame = task.run(data=data)                             # uniform, frequency, markov, markov2, logistic, uxagent_markov
task.score_all(frame)                                   # loglik (primary), accuracy, share_tvd; session bootstrap CI

buy = bench.get("webshop_buy")
buy.score(buy.run(data=buy.load()), "logistic")         # auc (primary), logloss, brier

m = wb.llm_method(se.FakeLLM())                         # real: se.llm.Claude(model=..., budget_usd=...)
task.run(methods={"llm": m}, data=data)                 # one prompt per test row; m.last_unparsed counts failures
wb.compare_agents(data)                                 # action shares and per-session means: humans vs UXAgent
```

- Options: `split="batch"` (default) or `"random"`; `test_frac` (0.3) and `seed` (0) for the random split;
  `uxagent=True` (load the UXAgent traces when present). Unknown options raise.
- `data.train` / `data.test`: `session`, `t` (action index in the session), `page` (page the action was taken on),
  `action`, `y` (index into `ACTIONS`), `prev`, `prev2`, `history` (earlier action names), counts so far
  (`n_search`, `n_items`, `n_details`), `n_opts` (options chosen on the current product), `n_goal_opts` (options the
  instruction asks for), `opts_done`, `results_page`, `log_step`, `log_results_page`, `split`, `batch`, `reward`
  (the session's final reward, for segments only, never a feature).
- `meta`: `split` rule, `actions`, `sessions` (one row per session: n_actions, bought, reward, price, batch, date),
  `sequences` ({session: action indices}), `instructions` ({session: goal text}, for prompts), `uxagent` (LLM-agent
  action sequences or None), counts, `test_shares` (next-action task) or `positive_rate` (buy task).
- EvalFrame: `unit` (`session:t`), `group` (session; bootstrap cluster), `truth`, `item` (previous action),
  `segment` (current page), `seg_step` (1-2 / 3-5 / 6+), `seg_success` (session reward = 1 or not), `pred_<method>`.
- Free functions: `classify_step`, `parse_trajectory`, `read_webshop`, `map_uxagent_action`, `uxagent_sequences`,
  `read_uxagent`, `transition_counts`, `sequence_transitions`, `session_summary`, `compare_agents`, `prompts`,
  `parse_action`, `llm_method`.

## Actions

Trajectory files log pages, not clicks, so actions are derived from consecutive pages (`classify_step`):

| action | page move |
|---|---|
| `search` | start page -> results; results / product -> results for other keywords (a new query after a browser back: the start page is not logged then) |
| `next_page` / `prev_page` | results -> results, page number up / down |
| `click_item` | results -> product, or product -> another product |
| `select_option` | product -> same product with other options chosen |
| `view_detail` | -> description / features / reviews sub page |
| `back` | product -> results, sub page -> product ("< Prev") |
| `back_to_search` | -> start page ("Back to Search") |
| `buy` | -> done (Buy Now; ends the episode) |
| `other` | reloads (same page, same content) and odd moves (start page -> product) |

Each session's first action is not a row: it is `search` in 1,559 of the 1,643 human sessions and would add free
hits. In the other 84 it is a start-page reload (`other`), and their first search is a row (at t = 1).

Counts on the 1,643 local trajectories (2026-10-22): 16,802 page moves; the derived `other` covers 120 start-page
reloads, 119 product-page moves with the same product and options, 40 results reloads and 6 start page -> product.
7 of 1,642 purchases are pressed from a sub page (5) or a results page (2, browser back); they are `buy` rows on
`webshop_next_action` but not rows of `webshop_buy`, which only keeps actions taken on a product page.

## What an LLM agent is scored on

`llm_method(llm)` sends one prompt per test row: system text (predict a human shopper's next step, answer with JSON
probabilities), then the instruction, the shopper's actions so far, the current page type, products opened so far,
options chosen on this product vs requested, and the ten actions with one-line meanings. The page contents (result
titles, product text) are not in the prompt; the model sees what the logistic baseline sees plus the instruction
text. Replies go through `parse_action` (JSON keyed by action names, else the first action name, else uniform), are
mixed with 5% uniform (a one-hot miss would otherwise score log 1e-6), and on `webshop_buy` the `buy` column is the
prediction.

Scores to compare with the baselines below:
- `loglik` (primary): mean log probability of the action taken, clipped at 1e-6. Uniform = -2.303.
- `accuracy`: argmax equals the action taken, ties split.
- `share_tvd`: total variation distance between the mean predicted action distribution and the observed one; the
  population-level check (does the simulated crowd click like the real one), independent of per-row skill.
- `webshop_buy`: ROC AUC (primary), log loss, Brier.

CIs: 95% percentile cluster bootstrap over sessions (`stats.cluster_bootstrap`, `default_rng(42)`, B = 1000).
An agent that clears `markov2` and `logistic` on `loglik` has learnt something beyond click habits; one that only
clears `frequency` has not.

## Baselines

| name | method |
|---|---|
| `uniform` | 1/10 each |
| `frequency` | train action shares, add-one |
| `markov` | P(a_t \| a_t-1) from train rows, add-one per row |
| `markov2` | P(a_t \| a_t-2, a_t-1): counts plus 10 pseudo-counts spread as the `markov` row |
| `logistic` | multinomial logistic (sklearn, C = 1) on one-hot previous two actions and page, plus step, counts so far, options chosen / requested, results page (standardised on train) |
| `uxagent_markov` | `markov` learnt from the 60 UXAgent LLM-agent sessions instead of humans (add-one); only when the parquet is present |
| `constant` (buy) | train buy rate |

On `webshop_buy` `markov` and `uxagent_markov` are the `buy` column of the same chains; `logistic` is a binary
logistic on the same features. No tuning.

## Split

Default `split="batch"`: the collection batches are dated in the file names. Sessions from batches dated before
2022-05-13 train (645 sessions), the 2022-05-13 / 2022-05-14 batches test (998). The batch CSVs shipped next to the
2022-05-13 trajectories give the HIT title "Amazon Shopping Game [Some scoring bugs fixed; server speed improved]",
so the test set is a later, slightly changed site: a temporal shift, not an i.i.d. holdout. `split="random"` holds out 30%
of sessions with `default_rng(seed)`.

## Measured numbers

One run each on the local catalogue copy on 2026-10-22, re-run after the review fixes to `classify_step` and the
UXAgent mapping (B = 1000; cells `value [95% CI]`). Command, from the
repository root with the catalogue under `$SIMENGINE_DATA`:

```python
from simengine import bench
from simengine.bench.tasks import web_behaviour as wb
for opts in ({}, {"split": "random", "seed": 0}):
    for T in (wb.WebShopNextAction, wb.WebShopBuy):
        t = T(**opts); fr = t.run(data=t.load(None))
        print(bench.to_markdown(t.score_all(fr, B=1000).assign(status="ok", is_baseline=True), primary_only=False))
```

Data as loaded: 1,643 trajectories, 1,642 ending in `buy`; mean final reward 0.744 and reward = 1 in 47.3% of the
1,643 (train batches 0.728 / 43.1%, test batches 0.755 / 50.0%). Close to the paper's human-evaluation average
(75.5 / 50.0%, Yao et al. 2022 appendix A.6), which is a different set of sessions.

### webshop_next_action, split="batch" (train 5,536 rows / 645 sessions; test 9,623 rows / 998 sessions)

| method | loglik | accuracy | share_tvd |
|---|---|---|---|
| logistic | -0.933 [-0.952, -0.915] | 0.641 [0.630, 0.651] | 0.061 [0.054, 0.068] |
| markov2 | -1.003 [-1.021, -0.985] | 0.616 [0.606, 0.626] | 0.068 [0.058, 0.079] |
| markov | -1.181 [-1.197, -1.164] | 0.530 [0.520, 0.540] | 0.078 [0.066, 0.091] |
| uxagent_markov | -1.967 [-2.003, -1.928] | 0.289 [0.277, 0.301] | 0.218 [0.206, 0.233] |
| frequency | -2.051 [-2.068, -2.035] | 0.243 [0.234, 0.252] | 0.107 [0.090, 0.125] |
| uniform | -2.303 | 0.100 | 0.315 [0.296, 0.335] |

### webshop_buy, split="batch" (train 2,485 rows, buy rate 0.258; test 4,689 rows / 997 sessions, buy rate 0.212)

| method | auc | logloss | brier |
|---|---|---|---|
| logistic | 0.808 [0.793, 0.823] | 0.413 [0.397, 0.432] | 0.132 [0.126, 0.139] |
| markov | 0.719 [0.704, 0.735] | 0.465 [0.451, 0.481] | 0.150 [0.144, 0.157] |
| uxagent_markov | 0.620 [0.598, 0.641] | 0.525 [0.499, 0.552] | 0.165 [0.156, 0.175] |
| constant | 0.500 | 0.522 [0.510, 0.537] | 0.169 [0.164, 0.176] |

### split="random", seed=0 (test 493 sessions)

| method | next: loglik | next: accuracy | next: share_tvd | buy: auc | buy: logloss |
|---|---|---|---|---|---|
| logistic | -0.969 [-1.016, -0.929] | 0.613 [0.595, 0.629] | 0.024 [0.015, 0.040] | 0.798 [0.772, 0.823] | 0.436 [0.409, 0.467] |
| markov2 | -1.025 [-1.060, -0.994] | 0.589 [0.575, 0.604] | 0.019 [0.013, 0.039] | | |
| markov | -1.182 [-1.215, -1.154] | 0.528 [0.513, 0.544] | 0.022 [0.014, 0.047] | 0.711 [0.686, 0.734] | 0.482 [0.459, 0.507] |
| uxagent_markov | -2.027 [-2.087, -1.968] | 0.267 [0.248, 0.285] | 0.230 [0.215, 0.251] | 0.609 [0.579, 0.642] | 0.562 [0.527, 0.598] |
| frequency / constant | -2.082 [-2.126, -2.043] | 0.228 [0.213, 0.241] | 0.035 [0.020, 0.072] | 0.500 | 0.539 [0.518, 0.561] |
| uniform | -2.303 | 0.100 | 0.271 [0.249, 0.306] | | |

(random split: next-action test 4,631 rows, buy test 2,130 rows, buy rate 0.230.)

### Humans vs UXAgent LLM agents (`compare_agents`, batch split)

| per session / share of actions | human train | human test | UXAgent |
|---|---|---|---|
| sessions | 645 | 998 | 60 |
| actions | 9.58 | 10.64 | 12.62 |
| searches | 1.52 | 1.42 | 2.02 |
| products opened | 1.90 | 2.35 | 2.62 |
| detail tabs | 0.63 | 0.90 | 1.92 |
| option clicks | 1.25 | 1.42 | 2.75 |
| bought | 1.000 | 0.999 | 0.867 |
| share search / click_item / select_option | 0.158 / 0.198 / 0.131 | 0.133 / 0.221 / 0.133 | 0.160 / 0.207 / 0.218 |
| share view_detail / buy | 0.066 / 0.104 | 0.084 / 0.094 | 0.152 / 0.145 |
| share next_page / prev_page | 0.093 / 0.033 | 0.055 / 0.026 | 0.009 / 0.001 |
| share back / back_to_search / other | 0.141 / 0.053 / 0.023 | 0.200 / 0.040 / 0.014 | 0.041 / 0.000 / 0.066 |

What the numbers say: the human's own last two clicks carry most of the signal (`markov2` takes loglik from -2.05 to
-1.00); state counts add a little more (`logistic` -0.93). A chain learnt from LLM shopping agents scores barely above
the human marginal on loglik (-1.97 vs -2.05) and is the worst on `share_tvd` after uniform: the agents read details
and pick options about twice as often as people, and almost never page through results or go back. On a random split
every chain-type baseline gets `share_tvd` near 0.02, so a simulated crowd has to come within a few points of the
human action mix to be competitive on that metric.

## Provenance

- New code, not ported from a research branch. Data reading follows the published file format (one JSON line per page:
  `page`, `url`, `goal`, `content`; `reward` and `reward_info` on the final `done` line); the action names follow the
  WebShop action space (search[...], click[...] on result items, options, Description / Features / Reviews,
  < Prev, Next >, Back to Search, Buy Now; Yao et al. 2022 section 3 and Figure 1).
- WebShop: Yao, Chen, Yang & Narasimhan (2022), "WebShop: Towards Scalable Real-World Web Interaction with Grounded
  Language Agents", arXiv:2207.01206 (NeurIPS 2022). Human trajectories from github princeton-nlp/WebShop
  (`all_trajs/`, linked from its README), MIT licence. Paper, section 3.2 and appendix A.6: 13 workers trained and
  selected with a qualification test (average reward > 0.75 on 5 tasks), the top 7 called experts, $0.70 per completed
  trajectory.
- UXAgent: Lu et al. (2025), "UXAgent: An LLM Agent-Based Usability Testing Framework for Web Design", CHI 2025 LBW,
  arXiv:2502.12561; HF dataset NEU-HAI/UXAgent, MIT (dataset card). The 60 rows are LLM agents with generated
  personas, not people; their site has a cart, filters and sorting that WebShop lacks. Mapping (`map_uxagent_action`):
  `add_to_cart` -> buy, cart / checkout / filter / sort / navigate -> other, typing into checkout, payment or
  quantity fields -> other, browser back -> back, typing a query and pressing the search button in the same step ->
  one search.
- Literature numbers (not measured here), Yao et al. 2022: human experts score 82.1 and succeed 59.6% on the 500 test
  instructions (Table 2, section 5.2); the 8 workers in the human evaluation average 75.5 and 50.0% (appendix A.6);
  IL+RL agent 62.4 / 28.7%. Human experts visit 11.3 states, check 1.9 items and search 1.4 times per trajectory on
  average (Table 2).
- The repository ships none of the data; files are read at run time from `data_root` / `$SIMENGINE_DATA`.

## Limitations

- Instructed task, paid workers: every session but one ends in a purchase, so this is not a conversion-rate dataset.
  `webshop_buy` asks "buy this product now or keep going", not "will this visitor buy at all".
- Actions are inferred from page moves. A reload and an option click that changes nothing look alike (`other`);
  `back` from a product page cannot be told apart from a browser back. Option clicks that keep the same option set
  are not visible.
- Sessions by the same worker are not clustered: the trajectory files carry no worker id, so the task bootstraps over
  sessions. The four MTurk batch CSVs next to the trajectories (`wa-human-traj_11..14.csv`) do carry one for 497 of
  the 998 test sessions (batches 20220513_1 to _4); the task does not read them. A one-off check on those 497
  (2026-10-22, B = 1000, worker ids turned into integer codes in memory, nothing kept): 8 workers, one of them behind
  43% of the sessions; `logistic` loglik -0.944 with session CI [-0.973, -0.916] vs worker CI [-1.032, -0.872];
  `webshop_buy` `logistic` AUC 0.788, session CI [0.762, 0.813] vs worker CI [0.674, 0.841]. Worker-level intervals
  are about three times wider (and a percentile bootstrap over 8 clusters is itself rough). Read the session CIs as
  "these sessions", not "new people"; the paper describes a pool of 13 trained workers.
- The batch split puts a scoring fix and a faster server between train and test; shares move (`back` 0.14 -> 0.20,
  `next_page` 0.09 -> 0.06), which is part of what the task measures but also makes `frequency` look worse than on a
  random split.
- `uxagent_markov` is a different site and only 60 sessions; its gap to humans mixes agent behaviour with site
  differences (no "Back to Search" button, a cart step).
- Prompts carry no page text, so an LLM cannot judge whether a product matches; adding result titles would need the
  WebShop product catalogue, which is not in the local copy.
- The tasks are registered when `simengine.bench.tasks.web_behaviour` is imported; the package `__init__` does not
  import it yet.
