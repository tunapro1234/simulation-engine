# `simengine.games`: economic game library, runner, scoring and coherence

This module puts personas through 15 one-shot economic games, scores the results against published human
numbers, and flags incoherent behaviour. It adds the scoring layer that BeeBench b01 lacked. b01 produced
10,920 result rows, but nothing in it measured distance to humans.

| component | file | what it does |
|---|---|---|
| `GAMES`, `list_games`, `get_game`, `check_game` | `library.py` | 15 GAME dicts (10 calibration, 5 holdout), the registry and the contract check |
| `render`, `render_belief`, `bounds`, `answer_spec` | `library.py` | deterministic prompt text, plus per-condition bounds |
| `play`, `estimate_cost`, `CostGateError`, `as_agents`, `read_results` | `runner.py` | personas × games × conditions × runs through any `simengine.types.LLM`, written to the b01 tidy schema |
| `parse_answer`, `instruction` | `_answer.py` | strict JSON answer parser, with retries that show the model its error |
| `moments`, `human_refs`, `human_distributions`, `distance_to_human`, `moment_frame` | `scoring.py` | per-cell moments, cited human values, absolute error with an agent-bootstrap CI, W1/KS, and an EvalFrame bridge |
| `coherence_audit`, `flag_rows`, `PRECEDENT` | `coherence.py` | the 10 b01 detectors; they flag rows and never drop them |
| `wtp_scorecard`, `CRITERIA`, `WTP_PARAMS`, `ERTAN_TARGETS` | `wtp.py` | the 9-criterion who-to-punish human-likeness index |

WTP in this module means who-to-punish (a punishment game), not willingness-to-pay; for pricing and
willingness-to-pay use `methods.conjoint` (`simengine.methods.conjoint.wtp`).

Dependencies: numpy, pandas and `simengine.llm` (for `FakeLLM` and `cost_usd` in the cost gate). scipy is used
only to cross-check the distances in the tests. The module uses `simengine.evidence.evaluate` only through the
frame that `moment_frame` returns. It does not depend on `simengine.sim`: answer parsing is a small local helper.

## Provenance

The source repos are read-only. Nothing was copied from their data or results.

- **edsl-test BeeBench b01** (same owner, no LICENSE file; reused internally):
  - `benchmark/beebench/b01/games/*.py` and `holdout/*.py`: 14 GAME dicts. Text, conditions, bounds and
    references are verbatim. Local file-path mentions were removed, and the deviation notes from the module
    docstrings were merged into the references.
  - `run.py`: the CSV schema, the resume key, the $5 cost gate and the retry of None decisions. The EDSL wiring
    was rewritten onto `LLM.map`.
  - `analyze.py`: the detectors and the precedent notes.
- **edsl-test `benchmark/score_wtp.py` + `human_targets.json`**: the 9 criteria, the weights, the scorer
  parameters, and the Ertan targets with their printed-page cites.
- **replicant** (MIT; credit to its authors): `papers/ertan2009/config.py` supplied the `ertan_vignette`
  parameters. The wording is new. The bundled oTree app has its own oTree license and was not ported.

## The games

| game | role | measures | conditions | holdout |
|---|---|---|---|---|
| dictator | dictator | altruism | 1 | no |
| eleven_twenty | requester | level-k depth | 1 | no |
| holt_laury | chooser | risk aversion | 10 rows (MC) | no |
| loss_gambles | gambler | loss aversion | 9 gambles (MC) | no |
| public_goods_strategy | contributor | conditional cooperation | unconditional + 5 strategy rows; belief | no |
| time_choices | chooser | patience, present bias | 6 (MC) | no |
| trust_returner | returner | positive reciprocity | 3 (per-condition max) | no |
| trust_sender | sender | trust | 1; belief | no |
| ultimatum_proposer | proposer | strategic fairness | 1 | no |
| ultimatum_responder | responder | negative reciprocity | 3 (MC) | no |
| beauty_contest | guesser | strategic depth | 1 | yes |
| pg_punishment | group member | cooperation + altruistic punishment | contribute + 2 punish vignettes; belief | yes |
| third_party_punishment | third party | norm enforcement | 4 transfer levels | yes |
| who_to_punish | voter | punishment norms | 3 ballot items (MC) | yes |
| ertan_vignette | punisher | targeting under a punish-low-only rule | 3 hypothetical profiles | yes |

**`ertan_vignette` is not a replication of Ertan, Page & Putterman (2009).** It is a rescaled one-shot vignette:
a group of 5, an endowment of 20, MPCR 0.4, 1 token spent removes 3, your own contribution is 15, and the budget
is 0-5. The other four contributed 18,17,19,16 (all_cooperate), 15,5,18,2 (mixed) or 17,18,16,2 (one_freerider).
No individual human amounts exist for these profiles, so the game has moments but no human reference. The
replicant config's `PAPER_FINDINGS` value of 0.85 has no page cite and is not used.

## Running

```python
import simengine.games as gm
from simengine.llm import Claude
llm = Claude(model="claude-haiku-5-5", cache_dir=".cache", budget_usd=5)
bank = {"baseline": {"baseline-000": ""},                  # empty text = no persona (the control)
        "big5": {"big5-000": {"text": "...", "params": {"A": 4.5}}, ...}}
print(gm.estimate_cost(llm, bank, "calibration", runs=3))   # upper bound, before any call
res = gm.play(llm, bank, "calibration", runs=3, out="runs/games.csv")   # re-running resumes
```

- **Prompting.** The persona goes in as `system`. The game text plus one answer line
  (`Reply with JSON only: {"answer": ...}`) goes in as `user`. When a game has a belief layer, the belief is asked
  first, and the decision prompt repeats the belief question and the stated answer (as in b01, where EDSL's
  targeted memory did this).
- **Runs** are passed to the backend as `sample=seed*1000+run`. Cached backends therefore return independent
  draws.
- **Invalid answers.**
  - An unusable reply is re-asked up to `retries=2` more times, with the error text appended.
  - If the reply still cannot be parsed, `decision` is None and `valid=False`. It is never $0. Before commit
    15b9531, b01 turned None into $0; that trap is closed here.
  - An out-of-range number is kept as the decision with `valid=False`. Bounds are per condition: trust_returner
    caps at 30/90/150, and the pg_punishment punishment stages cap at 10.
  - The parser does no substring or first-number salvage. "I would give 30 dollars" is not read as 30.
- **Columns.**
  - The b01 schema: `model, persona_method, agent_id, game, role, condition, wording_id, run, decision, belief,
    valid, raw_len, cost_usd`.
  - Extra columns: `seed, max_tokens, persona_params, belief_valid, attempts, error, holdout`.
  - b01 also had `timestamp`. It is dropped so that output stays deterministic.
- **Resume key:** `(model, persona_method, agent_id, game, condition, wording_id, run)`. A `resume=` frame or CSV
  and the `out=` CSV are both read. Rows already present are not asked again, and the returned frame has no
  duplicate keys.
- **Cost gate.**
  - Before the first call, `estimate_cost` computes an upper bound: prompt characters / 4 input tokens, plus
    `max_tokens` output tokens per call, at list price.
  - The run raises `CostGateError` when the bound exceeds `budget_gate_usd=5.0`, unless `confirm=True`.
  - A model with no list price raises unless you pass `est_cost_per_call=`. An unknown price must not switch
    the gate off silently. `FakeLLM` costs 0.
- **`max_tokens` is recorded on every row.** It acts as a treatment: in edsl-test, the bare model punished
  free-riders at 8192 but not at the default cap (single run;
  `ET/experiments/07_who_to_punish/results/2026-06-11_15-08-37_baseline/ANALYSIS.md`).
- **wording_id.** Only wording 0 ships. A game can carry `"wordings": {1: {"text": ...}}` for paraphrase runs.

## Scoring

```python
m = gm.moments(res)                       # per (model, persona_method, game): moment, value, n_agents, n_rows, n_valid
d = gm.distance_to_human(res, B=1000)     # abs_error vs human + agent-bootstrap CI; W1/KS for eleven_twenty
a = gm.coherence_audit(res)               # flags with precedent notes; gm.flag_rows(res) marks rows
frame = gm.moment_frame(res)              # EvalFrame -> simengine.evidence.evaluate(frame, ours="big5", task="continuous")
```

- **Moments use valid rows only.** Invalid rows show up as `n_rows - n_valid`. A moment that cannot be computed
  (no valid rows, or no complete schedule) stays NaN.
- **Schedule-based moments.** Holt-Laury safe choices, the loss-aversion ladder and the public-goods strategy
  type are computed only on complete (agent, run) schedules.
  - `implied_lambda_median` is the lowest accepted gain divided by the $5 loss. An agent who never accepts is
    right-censored at 4.0, the top of the ladder, so the median is then a lower bound. `share_never_accept`
    reports how many agents were censored.
  - Strategy types are a 5-row simplification of the 21-entry Spearman typing in Fischbacher, Gächter & Fehr
    (2001):
    - free_rider: all zeros
    - conditional: weakly increasing and not constant
    - hump: rises to an interior peak, then falls
    - other: anything else
- **Human references** (`human_refs()`): 31 moments, each with a citation and a scale.
  - Each value is a published-literature constant taken from the game's own `references`. These are
    parameters, not data. Approximate values are marked as approximate: Holt-Laury "~2/3", Camerer's "about
    half", and Fehr & Gächter's 6.8 points for a Partner-treatment free-rider.
  - `eleven_twenty:mean_request` = 17.10 is simengine's own computation from Arad & Rubinstein Table 1.
  - time_choices has no numeric human moment. The β meta-estimate is a model parameter, not a choice share.
    ertan_vignette has no numeric human moment either.
- **Full human distribution:** eleven_twenty only (Arad & Rubinstein 2012, Table 1, n=108). For it,
  `distance_to_human` adds Wasserstein-1 and KS rows.
  - Nagel (1995) reports intervals and spikes, not a full table. The beauty contest is therefore scored on its
    mean, median, share above 67 and share choosing 0. No distribution distance is computed for it.
- **The bootstrap CI** resamples agents. All of an agent's runs and conditions move together. The interval is a
  percentile interval of the absolute error.
  - Cost: on the full b01 table (10,920 rows, 7 methods), `distance_to_human(B=1000)` takes about 50 s on the
    build machine. Lower `B` while iterating.
- **`moment_frame`** gives one row per referenced moment: `unit=game:moment`, `group=game`,
  `truth=human value`, `pred_<method>`, and `split` (`test` for holdout games, `train` otherwise).
  - `evaluate` scores the test rows only, so its headline covers the holdout games. The baselines are fitted on
    the calibration moments.
  - By default, only share-scale (0-1) moments are included; `include_raw=True` adds the others.
  - **Deviation from the porting plan:** the `item` column is left out by default (`item=True` adds
    `item=game:moment`). With one row per item, evaluate's continuous random baseline ("another unit's truth
    within the item") copies the truth and scores 1.000. This was observed while building the module.

## Coherence audit

The audit ports the ten detectors in `analyze.py` and keeps the source's rules:
- Flags mark rows; they never filter data. Use `flag_rows` for a `coherence_flags` column.
- Cells are model × method × game.

| flag | fires when | human precedent |
|---|---|---|
| `invalid_rate` | > 10% invalid decisions in a cell | simengine note: a harness or parse failure, and often not random dropout |
| `zero_variance` | ≥ 6 valid decisions, all identical (the `baseline` method is exempt) | simengine note: humans vary in every game here |
| `hl_multi_switch` | more than one A/B switch in a Holt-Laury agent-run | a minority of real subjects switch more than once |
| `lg_non_monotone` | accepts a smaller gain but rejects a larger one | rare; usually noise |
| `tc_future_bias` | later in the now-pair, sooner in the matched later-pair | attested but rare |
| `ur_incoherent` | accepts a lower offer but rejects a higher one | rare; usually noise |
| `tr_overreturn` | returns more than received | simengine note: impossible in the original |
| `bc_dominated` | beauty-contest guess > 67 | a small share of Nagel's subjects did this |
| `pp_perverse` | punishes the cooperator more than the free rider | ~20% of punishment events in the literature |
| `tpp_inverted` | punishes the fair dictator more than the unfair one | antisocial punishment exists |

`tr_overreturn` is checked on every numeric decision, including invalid ones. The per-condition bound already
marks over-returns invalid, so a check on valid rows alone could never fire.

## Who-to-punish scorecard

`wtp_scorecard(run)` scores one multi-period who-to-punish group run.
- **Input** is structured data, not the regex over `summary.txt` that the source used:
  - `votes`: dicts with `punish_low` and optionally `punish_high`, or a list of bools
  - `periods`: `contrib` and `received` per player, optionally `punish_low`
  - `punish_events`: optional, each with `target_contribution`, `avg` and `amount`
- **Criteria and weights:**
  - weight 3: never_punish_high, final_vote_punish_low, treatment_effect
  - weight 2: no_dismantling, period1_level, sustain_under_punish_low, punishment_usage, punishment_targeting
  - weight 1: heterogeneity
- **N/A criteria are excluded from the weighted mean, not zero-filled.**
- **Two deviations from the source**, both from that rule:
  - `never_punish_high` is N/A when no vote records `punish_high`. The source assumed 1.0.
  - `punishment_targeting` is N/A when punishment was legal but none was paid. The source wrote 0.0.
- **Targets** come from Ertan, Page & Putterman (2005 WP, EER 2009) via `human_targets.json`. Page numbers are
  printed pages.
  - Some values are figure readings: the period-1 band 0.69-0.77 (Figure 3) and the first and final vote shares
    (Figure 2).
  - `treatment_full_credit` (2 tokens) and `heterogeneity_full_credit` (SD 3) are scorer choices, not paper
    numbers.

## Measured results from the source (not re-measured here)

These numbers come from the edsl-test runs on gpt-oss-120b. Nothing was measured on Claude, and no ranking
transfers until the games are rerun through `simengine.llm`.

- **b01 battery** (`ET/benchmark/beebench/b01/results.csv`, `ANOMALIES.md`): 10,920 rows (7 methods × 14 games ×
  10 agents × 3 runs), $1.27, and 30 invalid decisions (0.3%). The twin2k method's trust_sender (7/30) and
  pg_punishment (21/90) decisions were 23% invalid, 28 of the 30. The source contains no distance-to-human numbers. Any per-game comparison must be recomputed with
  `distance_to_human` and labelled as simengine's own computation.
- **Who-to-punish index** (`ET/benchmark/REPORT.md`, `wtp_scores.csv`; mean of 3 seeds, 1 group of 4 per seed; value_anchor is the mean of 4 runs because seed 1
  ran twice; computed by the source scorer, whose N/A handling differs from `wtp_scorecard`):

  | method | index |
  |---|---|
  | construction | .813 |
  | demographic | .797 |
  | value_anchor | .786 |
  | twin2k | .783 |
  | baseline | .694 |
  | big5 | .688 |
  | gps | .688 |
  | homo_silicus | .677 |
  | anthology | .617 |

  - Ranks within the top 4 are not significant, and the weights were not sensitivity-checked.
  - baseline and homo_silicus have an effective n of 1 (cache-degenerate across seeds).
  - Those scores came from the source's regex parser. `wtp_scorecard` applies the same formulas to structured
    input, with the two N/A deviations above, so it has not been re-run on those logs.

## Tests

The tests run offline with `FakeLLM` in about 2 s, in three files:

- `tests/test_games_library.py`:
  - the contract check on all 15 games
  - every placeholder is filled by every condition
  - `render` is deterministic
  - the answer parser does no salvage
- `tests/test_games_runner.py`:
  - output follows the schema
  - None answers give `valid=False` and are never 0
  - error feedback reaches the retry prompt
  - the belief is asked first and shown with the decision
  - resume adds no duplicates (from a frame and from a CSV)
  - the cost gate refuses without `confirm` and raises for an unpriced model
  - each run is an independent sample
- `tests/test_games_scoring.py`:
  - each detector fires on a synthetic bad case and stays silent on a clean one
  - `distance_to_human` is 0 when the moments equal the references (published refs, the Table 1 distribution, and
    own moments for all 15 games)
  - W1 and KS match scipy
  - `moment_frame` feeds `evaluate`
  - `wtp_scorecard` gives 1.0 on a human-like run and excludes N/A criteria

