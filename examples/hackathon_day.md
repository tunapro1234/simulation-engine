# Hackathon day playbook

From a brief and a CSV to an evidence card in about 1-2 hours. The numbers that justify each step are in
[docs/RESULTS.md](../docs/RESULTS.md), which also says where each one was measured; the method notes are in
[docs/METHODS.md](../docs/METHODS.md).

> **Prior work.** simengine was written before the event. If you use it at a hackathon, say so in the submission
> and cite the pre-event state so it can be checked: "built on our simengine library, written before the event
> (v0.1.0, commit `<hash>`, `<date>`)", taken from the last commit before the start
> (`git log -1 --format='%h %ad' --date=short`). The library is proprietary and private, not open source: do not call
> it "open", and if the event requires pre-existing code to be public, clear that with the organiser first. Follow
> the event's rules on existing code. What you build on the day is the data work, the configuration, the sealed
> predictions and the card. Check the dataset terms too (README, "Veri setleri ve lisanslar"): SimBench is
> non-commercial, Dominick's and Pew/OpinionQA are research-only, GSS may not be redistributed.

## 0. Before you start (5 min)

```bash
pip install -e .[all]          # anthropic, lightgbm, statsmodels
simengine demo                 # offline check: FakeLLM, about 4 s, writes runs/demo/card.md + dashboard.html
```

No install (dependencies already in the environment): run from the repo root with
`PYTHONPATH=. python -m simengine demo`; every `simengine ...` command below also works as `python -m simengine ...`.
Keep the day's tables in `data/` and outputs in `runs/`: both are git-ignored, so customer data is never committed.

Set the API key in the environment, never in the repo. Use `se.llm.Claude(model="claude-haiku-5-5", effort="low",
thinking="off", max_tokens=400, cache_dir=".cache", budget_usd=...)` for the bulk of the calls. The backend stops on
the first 401 or spend-limit error, and it does not send `temperature`, which Haiku/Sonnet 5.5 reject.

- **Thinking eats `max_tokens`.** With the default (adaptive) thinking and a small `max_tokens`, Haiku can spend the
  whole budget thinking and return empty text: 186 of 189 calls at `max_tokens=300` in the D2 rehearsal. Pass
  `thinking="off"` or raise `max_tokens`. Below 1024 with thinking on you get a `ThinkingBudgetWarning`; an empty
  reply cut at `max_tokens` raises `EmptyResponseError` (billed, not cached; in `map(..., return_exceptions=True)`
  it lands in that item's slot).
- **Shared spend log.** `export SIMENGINE_SPEND_LOG=<path>` and `SIMENGINE_AGENT=<name>` make every real call append
  one tab-separated line (time, agent, model, input, output, usd, tag) next to the JSONL ledger; or pass
  `Claude(..., spend_log=..., agent=...)`. Give each step a `tag=` so the cost splits by step.

## 1. Read the brief and name the task (10 min)

Decide which of these the brief is really asking for, because the measured answer differs:

| the brief asks | task | first choice | why |
|---|---|---|---|
| "what share of segment X would answer Y?" and you have no survey | `distribution` | `sim.segment_distribution` (3 framings) | closed 0.818 of the gap on CES, panel 0.55 |
| "which of these two texts/offers wins?" with past tests | `ab` / `binary` | `methods.ABStack` (TF-IDF + LLM) | 0.817 on Upworthy; Haiku panel 0.522 |
| the same, but with no past tests | `ab` | `LLMDirectAB` with a big model, then `EarlyTestBayes` | 2.27x traffic saving at 100 impressions per arm (prior calibrated on 20 tests) |
| "what will each customer do?" with history | `binary` / `continuous` | persistence, decayed frequency, LightGBM | LLM personas stayed below the history model |
| "predict answers of people in **new** countries/groups" from survey data in other groups (no earlier wave) | `continuous` / `binary` | `methods.GroupTransfer` (group-level ridge + person deviation, optional LLM group prior) scored with `methods.oof_predict(..., group=...)` | D2 rehearsal, group-held-out CV Spearman: person-only 0.197, two-stage 0.257, + LLM prior 0.295-0.311, oracle 0.344; one dataset, no CI |
| "how will adoption / a cascade evolve?" | `continuous` | curves + `abm.AdoptionABCAnalogy` with the Bass control | ABM 8.48 vs Bass control 9.73 MAE |
| "tell us why / give us voices" | qualitative | `sim.survey` / `sim.ab_test` panel with controls | use quotes, not the panel's numbers |

## 2. Pre-register (10 min)

Write the plan before you look at any result: question, data, split, primary metric with direction, baselines,
expected result with numbers, falsification ("the claim fails if ..."), stop rule.

```python
from simengine.research import Prereg, lock_prereg
Prereg(title="...", question="...", data="...", split="time: train <= 2024-06, test after",
       metrics="AUC (higher better)", baselines=["constant", "persistence", "logistic"],
       expected="stack beats logistic by >= 0.01 AUC", falsification="CI of stack - logistic includes 0",
       stop_rule="3 configs on dev, then one test run").write("prereg.md")
lock_prereg("prereg.md")        # SHA-256 + timestamp in prereg.md.lock.json
```

Hold out the test split (preferably by time or by person) and do not open it until the end.

## 3. Load the data, build the ladder (20 min)

Turn the CSV into an EvalFrame: `truth`, `pred_<method>` columns, and `group` / `segment` / `split` / `time` where
they apply (Turkish column aliases are accepted). Put in the label-free baselines first:

- persistence (last period's value): the strongest single baseline in the research;
- item marginal / overall copy / uniform for distributions;
- popularity / decayed frequency for baskets;
- logistic before LightGBM when the periods differ (bank marketing: logistic 0.748, LightGBM 0.665).

Baselines trained on the target labels must be out-of-fold: `methods.oof_predict(model, df, features=..., target=...,
group="country")` (GroupKFold when `group` is given, so no group is in both train and test of a fold). Declare them
as `baselines_oof` in step 5, not as `unlabeled_baselines`.

`simengine bench list` shows the built-in tasks; if one matches your data, `simengine bench run <task>
--data-root <dir>` gives the research baselines in a few seconds.

## 4. Add the LLM where it is measured to help (30-40 min)

- **No data at all:** `sim.segment_distribution(llm, segments, questions, population=...)`. Seal the predictions with
  `se.seal(predictions, "description")` (the same function as `se.evidence.seal`) before you open the truth. It writes
  to `.onkayit/zincir.jsonl` by default; pass `registry=...` for another directory and give the same directory to
  `--prereg`. A `se.research.seal` chain (`.prereg/chain.jsonl`) is read too. Memorisation of public surveys is not
  ruled out: for numeric aggregates (0-100 country means) use `se.correlation_audit(truth, named, blind, llm=...)`
  (named vs blind prompt, Spearman), or `memorization_audit(..., tolerance=, relative=True)` instead of the ±0.02
  exact hit.
- **With a learner:** stack the LLM output with the learner (`ABStack`, `TwinStack`, `DevStack`). Also build the
  stack without the LLM column. On Twin that fair LLM-free stack was what turned a claimed +0.071 into +0.006.
- **As a prior in a small test:** `EarlyTestBayes` after `calibrate_prior` on past tests. The gain disappears at
  800+ impressions per arm.
- **Personas:** `persona.from_table(df, id_col=..., segment=..., forbidden=[target])`, then `sim.survey` /
  `sim.ab_test`. Keep the default control arms (no persona, swapped persona). If the controls agree with the persona
  arm, the persona is not doing the work. Put a behaviour anchor (last period's value) in the persona when you have
  one: OE went from 0.618 to 0.860 with it, which is about persistence (0.848).
  - `from_table` stops on empty or duplicated ids with the counts; `id_col=None` uses row numbers.
  - **Soft refusals.** Haiku refused 59 of 150 survey batches in plain text in D2. `survey` / `ask_panel` now detect
    a refusal (`se.looks_like_refusal`), keep it out of the cache and re-ask once with a research-forecast framing.
    Read `res["refused"]`, `res["unparsed"]` and `res["refusal_rate"]` and put them on the card: the missing
    personas are not missing at random. For your own direct calls, `llm.complete(..., cache_if=lambda t: not
    se.looks_like_refusal(t))`.
  - **Numeric / slider questions:** `{"id": "pol", "text": ..., "kind": "numeric", "min": 0, "max": 100}`; out of
    range counts as unparsed, never clipped. `free_text=False` drops the answer sentence (most of the output tokens;
    SSR is then skipped).
  - **Per-person stimulus** (each person saw a different message): `survey(panel, qs, llm=llm,
    stimulus_col="condName")` or `context={pid: text}`; personas are batched by stimulus.
  - The ledger tag per arm is `survey:persona` / `survey:no_persona` / `survey:persona_swap`.

Skip discussion rounds and agent societies for accuracy: neither moved the result measurably.

## 5. Evidence card (15 min)

```bash
simengine evaluate data/frame.csv --ours stack --raw llm --task binary --primary auc \
    --unlabeled-baselines persistence --prereg .onkayit --out runs/evidence --lang en
```

Always pass `--primary` with the metric you pre-registered in step 2. Without it, `evaluate` grades on the task
default and warns: binary/multiclass `accuracy`, or `macro_f1` when the majority class is above 60 %; distribution
`one_minus_tvd`; continuous `pearson` (`one_minus_mae` when the truth lies in 0-1); ab `pair_sign`. The badge can
flip with the metric, so grade on the one you registered. `--prereg` must be the directory `seal()` wrote to
(default `.onkayit`); `evaluate` stops with an error when it has neither `zincir.jsonl` nor `chain.jsonl`.

- **Declare your baselines.** Label-trained, cross-fitted baselines go in `--baselines-oof T1,T2` (`baselines_oof=`).
  The badge compares ours with the strongest baseline, declared or automatic. With nothing declared the strongest
  automatic rung may be random/marginal; the card then says "only trivial baselines" (in D2 that gave a misleading
  `high confidence` of +0.310 before the fix). Every `pred_` column is on the ladder.
- **Ceiling.** `--ceiling-col pred_oracle --ceiling-label "oracle"` (`ceiling_col=`) uses a model/oracle ceiling
  instead of the human retest; the gap closed is computed from it and the card names the ceiling type.
- **Group level.** `--aggregate-by country` (`aggregate_by=` or `se.evaluate_aggregate(frame, by=...)`) averages
  truth and predictions per group and bootstraps over groups, e.g. a 20-country Spearman. Run it as a second report.
- **Large tables.** In a single-item frame above 2,000 rows the PPI++ multiplier is computed on a fixed-seed
  2,000-row subsample (noted on the card; `--multiplier-max-rows 0` for the whole table, slow). Above 5,000 rows
  Spearman and the binary metrics use a fast bootstrap path. 42k rows x 8 methods took about 11-12 s instead of
  minutes. `--no-multiplier` skips the multiplier.
- **Sealed card before the truth.** Run the same command on the test frame without a `truth` column (or all NaN)
  and with `--prereg`: you get an `awaiting truth` card with the seal hashes and a prediction summary, exit 0.
  Without `--prereg` the card says the predictions are not sealed.

Or in Python: `se.evidence.evaluate(frame, ours=..., raw=..., task=..., primary="auc", unlabeled_baselines=[...],
prereg=".onkayit")`, then `se.evidence.card(report, "runs/card.md", lang="en")` and
`se.evidence.dashboard([report], "runs/dashboard.html")`.
The card gives the baseline ladder with paired CIs, the shuffle control, the split and memorisation audits,
"worth N humans" (PPI++), the sealed prediction hashes with their UTC times, and a badge: `high confidence`,
`inconclusive`, `below baseline` (the paired CI of ours − strongest baseline is entirely below 0) or
`awaiting truth`. Report the badge as it comes, with the difference and its CI. In the research none of the four
example cards reached "high confidence", and "inconclusive" with honest baselines is a stronger pitch than a number
without them. `lang="en"` drops the Turkish badge reason and headline and uses dot decimals; the ladder and audit
labels stay Turkish.

## 6. If there is time: a research loop (optional)

`simengine loop config.json` (see `examples/loop.json` and `examples/research_loop.py`) searches configs on dev with a
paired keep rule, confirms on a selection split and opens test once. Read `dev_improvement` (positive = better)
rather than `dev_gain`, which is winner − start in metric units and is negative for a lower-is-better metric. Random search is enough: the LLM observer was not
better than random in either measured loop. Expect dev gains to shrink on test (OpinionQA A: the dev gain did not
hold).

## 7. Pitch checklist

- One sentence: what was predicted, against which baseline, by how much, with the CI.
- The ladder figure from the dashboard, with persistence on it.
- What the LLM added over the best LLM-free arm, which may be zero.
- The sealed prediction hash and time (in the card's "Sealed predictions" table) and the prereg lock.
- Cost and time: Haiku is about $0.0003 per call; an OE run from raw CSV to card took under 4 minutes for $0.105.
- The prior-work line from the top of this file.
