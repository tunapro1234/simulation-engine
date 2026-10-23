# `simengine.research.tree_search`: AIDE-style solution tree search and MCTS-UCT

## What it does

Two searches for automated modelling. Both use the same runner contract and the same workdir layout as `Loop`.

- **`TreeSearch`** is a tree search in the style of AIDE (Jiang et al. 2025, arXiv:2502.13138). Every node is a
  solution: a config dict, or `{"code": str}` for a code snippet. Each step works like this:
  1. A fixed search policy picks one of three operators and, where needed, a parent node: **draft** (a new
     solution), **debug** (repair a buggy node) or **improve** (one atomic change to the best node).
  2. The operator proposes a payload.
  3. The user's runner scores it on the dev split, and the node joins the tree.

  The answer is the good node with the best dev score (greedy pick).
- **`MCTSSearch`** runs Monte Carlo tree search over the discrete choices of a `Space`, with UCB1 at each tree node
  (UCT). It is meant for config spaces where you want a bandit-style search instead of a local one.

Each evaluated payload is a `Node`. `finalize()` runs the winner and a reference once on the test split. A paired
bootstrap gives the difference with its CI. The result also reports `winner_test_minus_dev`, because the winner's dev
score is a maximum over many noisy scores and so reads too good. It also reports `reference_test_minus_dev` (the
same gap for the reference, i.e. the dev -> test shift: other rows, a refit, a later period) and
`excess_test_minus_dev` = winner gap − reference gap, the part of the gap that selection on dev adds on top of the
shift.

## API

```python
from simengine.research.space import Space
from simengine.research.tree_search import (ConfigOperators, LLMOperators, MCTSSearch, TreeSearch,
                                            summarize_tree, best_node)

space = Space({"lr": [0.02, 0.1, 0.3], "leaves": [7, 31, 63]}, defaults={"lr": 0.1, "leaves": 31})

def runner(cfg):            # cfg = payload + {"split": "dev" | "test"}
    ...                     # fit, predict, score each unit
    return {"per_unit": pd.DataFrame({"unit": ids, "brier": (p - y) ** 2}), "metrics": {...}}

ts = TreeSearch(runner, ConfigOperators(space), primary="brier", higher_is_better=False, steps=20,
                num_drafts=5, debug_prob=0.5, max_debug_depth=3, workdir="runs/tree", seed=0)
out = ts.run()              # {"best_node", "best_payload", "best_dev", "counts", "runner_calls", "history", ...}
fin = ts.finalize()["final"]   # test once: winner vs reference (first good node)
# fin: winner_dev, reference_dev, test{winner, reference, ...}, test_diff{winner-reference: {ci95, bonferroni}},
#      winner_test_minus_dev, reference_test_minus_dev, excess_test_minus_dev

m = MCTSSearch(runner, space, primary="brier", higher_is_better=False, budget=40, c=2 ** 0.5, n_grid=5)
m.run(); m.finalize()       # reference = the space defaults

# LLM operators (any simengine LLM; FakeLLM in tests)
ops = LLMOperators(llm, mode="config", space=space, task="Predict y; lower Brier is better.")
ops = LLMOperators(llm, mode="code", task="...", data_preview="columns: ...")   # payload {"code": ...}
```

| name | what it is |
|---|---|
| `TreeSearch(runner, operators, *, primary, higher_is_better, steps=20, num_drafts=5, debug_prob=0.5, max_debug_depth=3, budget_usd=None, task, unit_col, split_key, dev_split, test_split, workdir, seed, n_boot, alpha, check_runner, verbose)` | The AIDE loop. `policy(nodes, rng)` returns `(op, parent)`, `step()` adds one node, and `run(finalize=False)` keeps stepping until `steps` nodes exist, the operator budget runs out, or the operators have nothing left. `finalize(extra_test=None)` runs the test split once and gives the same answer if called again. |
| `MCTSSearch(runner, space, *, primary, higher_is_better, budget=50, max_iter=20*budget, c=sqrt(2), keys=None, n_grid=5, ...)` | Depth d of the tree fixes key d. Each iteration selects a path by UCB1, expands one untried child, fills the remaining keys at random (the rollout), scores the config and backs the reward up the path. Runs stop at `budget` runner calls, at `max_iter`, or when the space is exhausted. |
| `ConfigOperators(space, *, default_first=True, tries=200)` | Operators that need no LLM. **draft**: the space defaults first, then uniform random configs. **improve**: change one key (another choice, or a Gaussian step with sd = range/4). **debug**: change one key, preferring keys or string values that the error text names as whole words. Proposals are always valid and never repeat a payload already seen. |
| `LLMOperators(llm, *, mode="config"\|"code", space=None, task="", data_preview="", atomic=True, max_tokens=2000, system=SYSTEM, forbid_test_refs=None, summary_k=5)` | The three AIDE prompts (draft, improve, debug). Each prompt includes the task, the data preview and `summarize_tree`. In config mode the reply is JSON `{"plan", "config"}`, checked with `space.validate`; when `atomic` is on, improve must change exactly one key. In code mode the reply is a plan plus one ```` ```python ```` block. A reply that cannot be used becomes a buggy node that carries the reason, so debug can repair it. In config mode, replies that mention the test split are refused. |
| `Node` | `id, parent, op, payload, plan, status (ok/buggy), score, lo, hi, n_units, error, metrics, debug_depth, children, payload_hash, repeat_of, diff_parent, score_source, seconds, cost_usd, eval_key`. |
| `summarize_tree(nodes, primary, higher_is_better, k=5)` | The tree memory used in prompts (AIDE's summary step): node counts, the k best nodes with their plans, and the last failures. Dev scores only. |
| `best_node(nodes, higher_is_better)` | The good node with the best score; on a tie, the earlier node wins. |
| `parse_config_reply`, `parse_code_reply` | Reply parsers. Each returns `(payload, plan, error)`. |

**Runner contract.** This is the same contract `Loop` uses. The runner is called as
`runner({**payload, split_key: split})` and returns `{"per_unit": ...}`. The search computes the score as the mean of
the per-unit primary values, with a bootstrap CI. A reply of the form `{"score": x}` is also accepted and marked
`score_source="runner"`. A node is marked buggy in any of these cases:

- the runner raises, or returns `status: "crash"`;
- there is no finite score;
- unit ids are duplicated;
- the dev unit count differs from the first scored node;
- the operator gave no valid payload.

The module never executes code itself. A code runner has to do its own sandboxing and timeouts.

**Search policy.** This is the AIDE rule, taken from `aide/agent.py` `search_policy` in aideml:

1. Draft while fewer than `num_drafts` drafts exist.
2. Otherwise, with probability `debug_prob`, debug a random buggy leaf whose `debug_depth < max_debug_depth`.
3. Otherwise improve the best good node.
4. If there is no good node, draft.

A debug node's depth is its parent's depth + 1; every other node has depth 0. If an operator returns nothing for
debug or improve, the step falls back to a draft. If the draft also returns nothing, the search stops.

A payload that was already scored becomes a node with `repeat_of` set, and the runner is not called again.

**Workdir.** The workdir holds `state.json`, `JOURNAL.md` (an indented tree plus the final table), `results.tsv`,
`units/<split>_<hash>.csv`, and, with `LLMOperators`, `operator/node<i>_<op>_{prompt,reply}.txt`.

To resume, build the search again with the same workdir. Step n uses `default_rng([seed, n])`, so a resumed run
makes the same choices as an uninterrupted one; a test checks this. A larger `steps` or `budget` on resume carries on
past the old limit. The search refuses to resume if the runner's source has changed (pass `check_runner=False` to
allow it), if the primary metric or its direction differs, or if the state was written by the other search kind.
The test split can be reached only from `finalize()`; `evaluate(..., "test")` raises `PermissionError`.

## Provenance

- **AIDE**: Jiang, Schmidt, Srikanth, Xu, Kaplan, Jacenko, Wu, "AIDE: AI-Driven Exploration in the Space of Code",
  arXiv:2502.13138 (read on arxiv.org, 22 Oct 2026). The paper defines the tree of single-file solutions, the three
  operators (draft from a plan, debug from the error and trace while keeping the approach, improve by one "atomic"
  change), the summary operator, a hard-coded greedy policy, and the choice of the best-scoring node as the answer.
  The paper gives no numbers for the policy settings. The defaults here (`num_drafts=5`, `debug_prob=0.5`,
  `max_debug_depth=3`, `steps=20`) come from aideml `aide/utils/config.yaml`. The buggy rule and the leaf/depth
  condition come from aideml `aide/agent.py`: buggy means an exception, a reviewer bug flag, or no metric, and debug
  candidates must satisfy `is_leaf and debug_depth < max_debug_depth`. No aideml code was copied.
  AIDE's LLM reviewer (`is_bug`) is not ported: here a node is buggy only for the mechanical reasons listed above.
- **UCT**: UCB1 (Auer, Cesa-Bianchi, Fischer 2002) used as the tree policy (Kocsis & Szepesvári 2006, "Bandit based
  Monte-Carlo Planning", ECML). The default `c = sqrt(2)` is the UCB1 constant for rewards in [0, 1]. Rewards are
  min-max normalised over the scores seen so far, and buggy or invalid configs count as the worst score seen. Neither
  of these two choices comes from the papers.

## Measured numbers

Every number below comes from a run made for this module on 22 Oct 2026. The machine was shared and under load, so
wall times are only rough.

**Tests:** `python -m unittest tests.test_research_tree_search`: 21 tests, OK in about 1-4 s (18 at first, 3 added
in review on 22 Oct).

**Toy objective, 10 seeds, 20 dev evaluations each.** This uses the objective from the tests, discretised to x in
{0, .2, …, 1}, k in {1, 4, 8, 12, 16, 24} and method in {a, b, c}, where c crashes; lower is better. The best
reachable true effect is 0.020; the defaults have 0.906. The table gives the true effect of the dev winner:

| search (budget 20) | mean | median | min | max |
|---|---|---|---|---|
| `TreeSearch` + `ConfigOperators` (5 drafts, 15 improves) | 0.033 | 0.020 | 0.020 | 0.084 |
| `MCTSSearch` | 0.104 | 0.084 | 0.020 | 0.180 |
| random search (`TreeSearch`, `num_drafts=20`) | 0.134 | 0.127 | 0.020 | 0.244 |
| `Loop` + random proposer (one change, kept only if the paired CI excludes 0) | 0.020 | 0.020 | 0.020 | 0.020 |

On this smooth objective with shared per-unit noise, local one-change search (AIDE-improve, Loop) beats MCTS and
random search at this budget. With a budget of 20, MCTS expands all 6 first-level nodes (x) and 14-16 of the 36
second-level nodes (x, k) in seeds 0-2; the third key (`method`, which carries the largest single effect) is never
expanded and is set only by random rollouts. Key order therefore matters for MCTS at small budgets.

Review re-run (22 Oct, same script, same machine): the four rows above reproduced exactly.

**Bank Marketing (bench task `bank`, CC BY 4.0, row-position split, `duration` dropped).**

- Model: sklearn `HistGradientBoostingClassifier`, searched over 2,560 configs (learning rate × max leaf nodes × min
  samples leaf × L2 × iterations × class weight).
- Fitting: dev models are fit on the train rows. Test models are refit on train + dev with the same config.
- Score: row Brier, lower is better.
- Budget: each search gets 20 dev evaluations, and the winner runs once on test against the sklearn defaults.
- Default config: dev 0.1085, test 0.2430.

Command:

```bash
SIMENGINE_DATA=<data-root> OMP_NUM_THREADS=1 nice -n 19 python examples/tree_search_bank.py --budget 20 --seeds 0 1 2 --out <scratch>
python examples/tree_search_bank.py --synthetic --budget 4 --seeds 0 --out <scratch>   # offline wiring check, ~4 s
```

Without a data root (or with `--synthetic`) the script writes a 3,000-row table in the bank layout to a temporary
directory and runs the same loader and searches on it; those numbers check the wiring only. Such a run
defaults to `--out runs/tree_search_bank-synthetic`, marks its `--out` with a `SYNTHETIC` file and a `data` column in
`summary.tsv`, and real and synthetic runs refuse to share one `--out` (the searches resume from the state there).

| search | seed 0 dev / test | seed 1 dev / test | seed 2 dev / test | test − default (mean of 3) |
|---|---|---|---|---|
| AIDE (`ConfigOperators`) | 0.1019 / 0.2543 | 0.0998 / 0.2569 | 0.0999 / 0.2576 | +0.0133 |
| MCTS-UCT | 0.1033 / 0.2516 | 0.1011 / 0.2535 | 0.1035 / 0.2387 | +0.0050 |
| random | 0.1006 / 0.2533 | 0.1041 / 0.2529 | 0.1025 / 0.2535 | +0.0103 |
| Loop + random proposer | 0.1016 / 0.2580 | 0.1016 / 0.2580 | 0.1016 / 0.2580 | +0.0150 |

Paired 95% CIs of winner − default on test exclude 0 in every run: 11 runs worse than the defaults, and one better
(MCTS seed 2, −0.0042 [−0.0054, −0.0031]). The three Loop seeds took different paths to the same winner, so the 12
runs have 10 distinct winners, 9 of them worse than the defaults on test.

The task's own baselines on the same test rows, from `BankMarketing.baselines`: constant 0.2810, persistence 0.2615,
logistic 0.1906, LightGBM grid 0.2708.

**Reading.** Every search lowers dev Brier by about 0.004 to 0.009. On test, almost every winner is worse than the
defaults, and all 11 GBDT configs that were run on test (the defaults and the 10 distinct winners, best 0.2387) are
worse than plain logistic regression (0.1906). The other 2,549 configs were never run on test. This split has a
strong base-rate shift (the loaded task gives 4.81% yes in train, 11.07% in dev and 30.83% in test). Tuning on dev
fits the dev period, so a lower dev score does not carry over. 9 of the 10 distinct winners use the smallest
`max_iter` (50), and all 10 use `min_samples_leaf` 320 or `l2_regularization` 10, i.e. strong shrinkage; that this
shrinkage is what hurts on the shifted test period is our reading, not tested.

Gap decomposition (computed from the same runs): the defaults go from dev 0.1085 to test 0.2430, a gap of +0.1344
that is pure dev -> test shift. The winners' gaps are larger; the excess over the defaults' gap (mean, min-max over 3
seeds) is AIDE +0.0213 (0.0179-0.0232), random +0.0164 (0.0144-0.0183), MCTS +0.0109 (0.0008-0.0180), Loop +0.0220.
`finalize()` now reports this excess as `excess_test_minus_dev`. The search modules do what they claim; the result
is a reminder that the dev-to-test gap (`winner_test_minus_dev`) and a test-once comparison against a simple baseline
are what decide.

## Limitations

- The greedy pick is optimistic, and nothing corrects for it. `finalize()` only shows the gap on test.
- AIDE's LLM reviewer and its parallel or async workers are not ported. Steps run one after another, with no
  per-node timeout.
- `ConfigOperators` improve is a random local step with no learned model behind it. `LLMOperators` is tested only
  with FakeLLM; this module made no real LLM calls.
- `MCTSSearch` searches every used key of the defaults. It ignores conditional keys (`space.used(cfg)` per config),
  and ranges become `n_grid` grid points. Keys are expanded in order, so at small budgets the last keys are set only
by random rollouts. A resume with other keys, another `n_grid` or other grid values is refused, because tree paths
are indices into the grid. The test-split reference (the space defaults) costs one extra dev runner
  call if the search never visited it. Wall-clock budgets are not supported, only runner-call budgets.
- Code payloads are hashed by their exact text, so a whitespace edit counts as a new solution. The test-reference
  regex is off by default in code mode, because it also matches names such as `train_test_split`.
- `TreeSearch`, `MCTSSearch`, `ConfigOperators`, `LLMOperators` and `summarize_tree` are exported from
  `simengine.research`; the rest (`Node`, `best_node`, the reply parsers) is in `simengine.research.tree_search`.
