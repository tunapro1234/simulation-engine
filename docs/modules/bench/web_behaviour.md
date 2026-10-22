# simengine.bench.tasks.web_behaviour

Human web-shopping behaviour, step by step. Two registered `BenchTask`s built from the WebShop human trajectories
(MTurk workers buying an instructed product on a simulated Amazon-like site):

| task | kind | row | truth | prediction |
|---|---|---|---|---|
| `webshop_next_action` | multiclass | every action after a session's first | which of 10 actions the shopper takes next | (n, 10) probabilities, columns in `ACTIONS` order |
| `webshop_buy` | binary | actions taken on an item page | the action is `buy` (Buy Now on this product) | one probability per row |

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
| `search` | start page -> results |
| `next_page` / `prev_page` | results -> results, page number up / down |
| `click_item` | results -> product, or product -> another product |
| `select_option` | product -> same product with other options chosen |
| `view_detail` | -> description / features / reviews sub page |
| `back` | product -> results, sub page -> product ("< Prev") |
| `back_to_search` | -> start page ("Back to Search") |
| `buy` | -> done (Buy Now; ends the episode) |
| `other` | reloads (same page, same content) and odd moves (start page -> product) |

Each session's first action (always `search` from the start page) is not a row: it would add 1,643 free hits.

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
  `add_to_cart` -> buy, cart / checkout / filter / sort / navigate -> other, browser back -> back, typing a query and
  pressing the search button in the same step -> one search.
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
- Sessions by the same worker are not clustered: the trajectory files carry no worker id (the MTurk batch CSVs do for
  some batches; they are not read). With a pool of 13 trained workers described in the paper, the session bootstrap CIs are too narrow for
  claims about new people.
- The batch split puts a scoring fix and a faster server between train and test; shares move (`back` 0.14 -> 0.20,
  `next_page` 0.09 -> 0.06), which is part of what the task measures but also makes `frequency` look worse than on a
  random split.
- `uxagent_markov` is a different site and only 60 sessions; its gap to humans mixes agent behaviour with site
  differences (no "Back to Search" button, a cart step).
- Prompts carry no page text, so an LLM cannot judge whether a product matches; adding result titles would need the
  WebShop product catalogue, which is not in the local copy.
- The tasks are registered when `simengine.bench.tasks.web_behaviour` is imported; the package `__init__` does not
  import it yet.
