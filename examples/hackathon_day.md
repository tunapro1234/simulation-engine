# Hackathon day playbook

From a brief and a CSV to an evidence card in about 1-2 hours. The numbers that justify each step are in
[docs/RESULTS.md](../docs/RESULTS.md), which also says where each one was measured; the method notes are in
[docs/METHODS.md](../docs/METHODS.md).

> **Prior work.** simengine was written before the event and is released under AGPL-3.0-or-later. If you use it at
> a hackathon, say so in the submission and cite the pre-event state so it can be checked: "built on our
> open-source simengine library (AGPL-3.0), published before the event: release `v0.1.0`, commit `<hash>`,
> `<date>`". Follow the event's rules on existing code. What you build on the day is the data work, the
> configuration, the sealed predictions and the card.
>
> **Keep it a dependency, not part of the submission.** Some events (TALP among them, as the owner reports from the
> application form; the clause text is not in this repo) take a right to use, develop and commercialise what is
> submitted. Do not copy simengine's code into the submission repo. Install it as a pinned dependency
> (`pip install "git+https://github.com/tunapro1234/simulation-engine@v0.1.0"`) so the submission holds only the
> work done on the day, and the library stays under its own licence. Check the dataset terms too (README,
> "Veri setleri ve lisanslar"): SimBench is non-commercial, Dominick's and Pew/OpinionQA are research-only, GSS
> may not be redistributed.

## 0. Before you start (5 min)

```bash
pip install -e .[all]          # anthropic, lightgbm, statsmodels
simengine demo                 # offline check: FakeLLM, about 4 s, writes runs/demo/card.md + dashboard.html
```

No install (dependencies already in the environment): run from the repo root with
`PYTHONPATH=. python -m simengine demo`; every `simengine ...` command below also works as `python -m simengine ...`.
`python -I` (isolated mode) ignores `PYTHONPATH`, so that form fails with `No module named 'simengine'`: with `-I`
use the installed package or put the repo on `sys.path` at the top of the script. For background runs use
`python -u` or `export PYTHONUNBUFFERED=1`; redirected output is buffered and the log stays empty until exit.
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
  `Claude(..., spend_log=..., agent=...)`. Give each step a `tag=` so the cost splits by step. With your own ledger
  set them once, on either side: `Claude(..., ledger=se.llm.CostLedger(".cache/ledger.jsonl", tsv_path=...,
  agent=...))` or `Claude(..., ledger=".cache/ledger.jsonl", spend_log=..., agent=...)`. Both together work while
  they agree; a different value on each side raises `ValueError`.
- **Counters.** `llm.stats` is a dict (`llm.stats()` gives a copy): `calls`, `cache_hits`, `retries`, `errors`,
  `cost_usd` for this instance and `ledger_total_usd` for everything in its ledger; `llm.spent_usd` = `cost_usd`.
- **Repeats.** The cache returns the same reply for the same prompt; `complete(..., sample=k)` with k = 0, 1, ...
  gives independent draws, each cached under its own key. Re-runs with a cleared cache differ (no `temperature`).
- **Embedding model offline (SSR).** `SSR` loads its embedding model offline. Point `SIMENGINE_HF_CACHE` at the
  local Hugging Face cache (or pass `SSR(cache_folder=...)`); without it a model that is not in the default cache
  fails to load (OSError).
- **Run prefix on the tag.** `export SIMENGINE_SPEND_TAG_PREFIX=prova-d3/` (or `Claude(..., tag_prefix="prova-d3/")`)
  puts the prefix in front of every TSV tag. The backend's own `tag=` and the call's tag are joined:
  `Claude(tag="L1", tag_prefix="d3/")` + `complete(..., tag="prior")` logs `d3/L1:prior`; `survey(..., tag="dev")`
  logs `d3/L1:dev:survey:persona`. Before this, a call tag silently replaced the backend tag (D3 #3).

## 1. Read the brief and name the task (10 min)

Decide which of these the brief is really asking for, because the measured answer differs:

| the brief asks | task | first choice | why |
|---|---|---|---|
| "what share of segment X would answer Y?" and you have no survey | `distribution` | `sim.segment_distribution` (3 framings) | closed 0.818 of the gap on CES, panel 0.55 |
| "which of these two texts/offers wins?" with past tests | `ab` / `binary` | `methods.ABStack` (TF-IDF + LLM) | 0.817 on Upworthy; Haiku panel 0.522 |
| the same, but with no past tests | `ab` | `LLMDirectAB` with a big model, then `EarlyTestBayes` | 2.27x traffic saving at 100 impressions per arm (prior calibrated on 20 tests) |
| "what will each customer do?" with history | `binary` / `continuous` | persistence, decayed frequency, LightGBM | LLM personas stayed below the history model |
| "predict answers of people in **new** countries/groups" from survey data in other groups (no earlier wave) | `continuous` / `binary` | `methods.GroupTransfer` (group-level ridge + person deviation, optional LLM group prior) scored with `methods.oof_predict(..., group=...)` | D2 rehearsal, group-held-out CV Spearman: person-only 0.197, two-stage 0.257, + LLM prior 0.295-0.311, oracle 0.344; one dataset, no CI |
| "predict member ratings for **new** items/films with no ratings" (cold start) and the item-level residual | `continuous` | `methods.GroupTransfer(group=item)` on `methods.item_holdout_features(ratings, ...)` (member stats + SVD taste rebuilt without each fold's items), `oof_predict(..., folds=res.folds)`; ladder rung `methods.MFBaseline`. Score: `evaluate(offset_col="member_mean", aggregate_by=item, noise_ceiling="variance_components")` | D3 rehearsal, one run, 300 dev films, film residual Spearman: label-free member mean 0.839, audience composition 0.845, stack with the LLM title prior and persona panel 0.895; split-half ceiling 0.987 (the rehearsal's own script). The LLM gain was title recognition (named 0.779 vs blind 0.169); no CI here |
| "forecast weekly demand for many series, horizon H" (stores x items, price/calendar known ahead) | `forecast` | `methods.forecast` baselines (`persistence_forecast`, `seasonal_naive`, `moving_mean`, `seasonal_ratio`, `ses`), then `DirectForecaster("lightgbm", horizons=H, exog=[price, calendar])`; compare all of them with `backtest(..., origins=[...])` over several origins. Score: `evaluate(task="forecast", primary="wmape")` (bias is reported next to it), a second report with `aggregate_by=<store_dept_week>, aggregate="sum"` for the hierarchy totals; `point_for_metric(pred, metric="wmape")` only for L1 metrics | D4 rehearsal (M5, one run, the team's own code written before these helpers existed), test series WMAPE: last-4 mean 0.379, SES 0.371, LightGBM + price + calendar with the Poisson median 0.347 [0.334, 0.360]; vs SES −0.024 [−0.030, −0.019]. The LLM event prior added nothing (named = blind) and was dropped |
| "how will adoption / a cascade evolve?" | `continuous` | curves + `abm.AdoptionABCAnalogy` with the Bass control | ABM 8.48 vs Bass control 9.73 MAE |
| "tell us why / give us voices" | qualitative | `sim.survey` / `sim.ab_test` panel with controls | use quotes, not the panel's numbers |

**Rehearsed scenarios.** Each brief type below had a timed 2-hour open rehearsal (22 Oct, not blind, simengine
v0.1.0, one run unless stated; numbers come from each rehearsal's own card). The calls use today's names. A name
given with its module path is not exported from `se.sim` / `se.methods` yet, so import it from that module. The
card scene is the one that held up in rehearsal. Report the baseline column as the arm to beat.

| scenario (the brief asks) | simengine calls, in order | strongest LLM-free baseline to report | card scene that worked | rehearsal result |
|---|---|---|---|---|
| **G0** free theme: "pick a business problem and solve it with simulation" (rehearsal: decide a headline A/B test in the first 100 impressions) | `methods.LengthAB`, `methods.TfidfAB().fit(past_tests)` -> `methods.LLMDirectAB(llm, reverse=True)` -> `methods.LogitAverageAB` / `methods.ABStack` -> `methods.simulate_saving_calibrated(..., n_calib=20, n0=100)` (traffic multiplier) -> `evidence.ab_pairs_frame` -> `evaluate(task="ab", prereg=...)` | TF-IDF prior trained on past tests: on tests significant at full size, 100 impressions per arm are worth 2.29x [1.65, 2.99] as many | slider over real impressions per arm (25-1,600), human-only vs prior + human: at 100 impressions 67% -> 75%, "about 230 impressions with humans alone"; the gain fades by 1,600 | Upworthy holdout, 300 significant tests: Haiku 5.5 prior 0.82x [0.66, 1.03], the same as a shuffled-Haiku placebo (0.83x); stack - TF-IDF +0.007 [-0.006, +0.021] |
| **G1** interacting synthetic panel + strategy test ("which launch / crisis / positioning strategy wins?") | `methods.rct_effects(df, cond, "Control", y, block=...)` (the anchor) -> `methods.rct_transfer(..., block=..., test_blocks=...)` (the baseline) -> `simengine.sim.effect.effect_forecast(llm, conditions, outcome, named=...)` (hidden arm = forecast, named arm = memorisation probe), `effect_frame`, `memorisation_check` -> panels: `sim.survey(..., stimulus_col="condition")`, `sim.strategy(personas, {id: text}, controls=...)` -> `simengine.sim.roundtable.roundtable_strategy(personas, scenarios, llm=llm, control="Control", on_event=print_events)` (`print_events` from `simengine.sim._stats`) -> `evaluate(..., compare=[...])`; split-half noise ceiling by hand | transfer of the same RCT from other blocks (32 other countries): condition Spearman 0.845 [0.43, 1.00], noise ceiling 0.908 | "the interaction rounds do not change the ranking": round 2 - round 0 +0.02 [-0.03, +0.22] | climate many-labs, 11 messages x 31 test countries: hidden LLM 0.073 [-0.61, 0.71], persona panel -0.127, strategy panel 0.227; all three pre-registered falsification conditions met |
| **G2** synthetic panel vs reality / say-do gap ("here is a small table, show the panel holds") | `persona.from_table(tab, id_col=..., forbidden=[say_col, do_col])` -> `sim.survey(ps, numeric_items, free_text=False)` (no_persona, persona_swap arms by default) -> `sim.survey_frame(r, level="person", kind="numeric")` -> `methods.oof_predict` with `simengine.methods.tabular.TabularRidge` and the stack -> `simengine.evidence.extras.say_do(frame, say_col, do_col)` -> `evaluate(task="continuous", primary="spearman", compare=[("S", "T2"), ("L2", "L2swap")])` | out-of-fold ridge on demographics + self-report scales + the person's own stated value: Spearman 0.594 [0.524, 0.655]; show "take what they say" (0.442) and its calibration (MAE 12.4 -> 3.5) as rungs | three bars, mean correct answers: people say 29.1, the panel says 26.0, real 17.3: "the panel simulates what people say, not what they do" | Twin-2K-500, 400-person table: raw panel 0.141 [0.043, 0.237]; stack - ridge -0.003 [-0.009, +0.004] (seed 1: -0.004); persona - swap +0.194 [+0.047, +0.350] |
| **G3** ad / creative / headline reaction | ladder first: `methods.LengthAB` -> `methods.TfidfAB` -> a frozen-embedding learner (by hand: `sim.SSR(model).embed` of A and B, no-intercept logistic on the difference, symmetric copies; choose TF-IDF or embedding by grouped CV on the training tests) -> `methods.LLMDirectAB(llm, reverse=True)` with a strong model -> `methods.ABStack(llm_method=direct, learner=<chosen rung>)` -> `evidence.ab_memorization(llm, pairs, named_context, neutral_context)` -> `evidence.ab_pairs_frame` -> `evaluate(task="ab")` | frozen bge embedding + logistic on 300 past tests: pair accuracy 0.733 [0.688, 0.775] (TF-IDF on the same 300: 0.568; which rung wins depends on the data) | a real pair side by side with P(A) and the real winner, then the memorisation probe: "these are from the archive, recall the winner" changes nothing (Sonnet -0.006 [-0.039, +0.028]). Do not reuse the rehearsal's "+13 points over the text model": that was against the weak TF-IDF rung | Upworthy holdout, 400 new tests: Sonnet 5.5 direct 0.695, Sonnet stack on the embedding learner 0.757, minus the embedding baseline +0.025 [-0.010, +0.062] (inconclusive); Haiku direct 0.559 |
| **G4** persona walk through a product / web flow ("see your app through the user's eyes") | `simengine.methods.next_action.sessions(steps, outcome_key="purchase", prefix_k=k, text_col=...)` and `funnel(...)` -> `methods.oof_predict(methods.TfidfBinary(text, encode_cols=(text,)), ...)` (+ `methods.target_encode`) -> LLM "shopper's eyes" P(buy) per session prefix (raw arm) -> `methods.DevStack` -> `methods.BinaryCalibrator(method="ratio")` on 50-200 real sessions for the level -> `evaluate(task="binary", primary="auc", compare=[("S", "T3"), ("L1", "T3")])`; a screen-by-screen walk (`sim.survey` per step with `stimulus_col=`, control arms on) only for voices and friction | logistic on TF-IDF of the first search query + query length + query target encoding: AUC 0.666 [0.617, 0.714] | "the raw persona walk overstates purchase 3x": 33.9% predicted vs 11.8% real (+0.221 [0.201, 0.241]); 50 real sessions fix most of the level, and that fix comes from the data, not the LLM | ShopCART, 1,000 test sessions: raw LLM 0.606 [0.551, 0.656]; stack - baseline +0.008 [-0.030, +0.041] (claim fell); the unlabelled LLM ranking is worth a text model on about 1,000-2,000 labelled sessions |
| **G5** persona / audience synthesis from data ("build a representative persona pool from this customer table") | `methods.segment_distributions(df, item, seg, cats, mode="overall")` (T1) and `mode="shrunk", m=select_m(...).m` (T2b; `select_m` from `simengine.methods.baselines`) -> `sim.segment_distribution(llm, segments, questions, population=...)` -> `persona.from_table(...)` -> `sim.persona_distribution(llm, personas, questions, segments={scheme: ...})` -> `methods.ShrunkDeviation(distribution=True)` (base = train overall, ref = the LLM's own overall) -> `persona.pool_select` / `persona.pool_curve` for the pool story -> `evidence.evaluate_distribution_persons(holdout_people, schemes, items, arms)`, over 5 seeds | shrunk segment empirical (T2b): 1-TVD 0.912; report it next to the train-overall copy (T1) and take the stronger | "the card's verdict changes with the seed, so the card says inconclusive" | dunnhumby, 335 holdout households, 84 cells: calibrated arm 0.905, minus T2b -0.007 [-0.015, +0.001] (seed 1: +0.009 [+0.001, +0.017]; 5 seeds mean -0.004, sd 0.008); raw persona panel 0.820, below uniform 0.882 |
| **G6** synthetic interview / focus group: is the "why" right? | persona = the person's own earlier texts before a cutoff (`persona.inject` `log_raw`) -> `simengine.sim.interview.interview(personas, question, llm=llm, stimulus_col=...)` (persona, no_persona, persona_swap arms), `interview_frame` -> candidates: the true reason + 3 by other people about the same product with the same rating -> `evidence.why_match({arm: texts, "own_history": past_texts}, candidates, truth_idx, ids=..., groups=...)`, once with TF-IDF and once with `encoder=sim.SSR(model).embed` -> `evaluate(frame, task="multiclass", primary="accuracy", ours=..., compare=[...])`; code the reasons into topics to separate reason from wording | the person's own earlier reviews: 4-way accuracy 0.477 [0.425, 0.530] (0.505 on a second distractor draw), chance 0.25 | "the person's own history is twice as accurate as the persona interview"; say "own words", not "own reason": at topic level both are near chance (-0.005 [-0.068, 0.055]) | Amazon Video Games reviews, 400 people: persona interview 0.273 [0.230, 0.320]; difference -0.205 [-0.273, -0.138] |
| **G7** simulator reliability ("how far can we trust this simulator?") | bench task `opinionqa_groups` (`OpinionQAGroups`, or the day's data in the same shape): `segments()` -> baselines `predict_with(data, "uniform" / "position" / "knn")` (setting A) and `"overall"` / `"sibling"` (setting B, population known) -> `predict_with(data, "llm", llm=llm, framings=..., samples=...)` on a question subset (the full test split is about 32,000 calls) -> `simengine.sim.robustness.robustness(llm, segs, qs, signal={"segments": ..., "population": ...})` with real data as the signal -> `sim.segment_distribution(..., repeats=2)` + `sim.retest_stats(samples, kind="distribution")` -> `evaluate(task="distribution", primary="wasserstein")` | A (nothing known about the target question): option position, W1 0.173 [0.148, 0.202]; B (population distribution known): overall copy, 0.064 [0.057, 0.071] | most of a one-prompt group difference is prompt noise: "reversing the options moves the answer by 0.056; the real group gap is 0.064, so the wobble is 88% of what you want to measure" | SubPOP OpinionQA, 40 questions x 22 groups: LLM (3 hidden framings) 0.100 [0.087, 0.114], beats A by -0.073 [-0.105, -0.048], trails B by +0.036 [+0.023, +0.050]; noise/signal ratio: retest 0.32, wording 0.60-0.73, option order 0.88 |
| **G9** experiment / A-B / price-WTP effect ("what would this experiment show on people?") | `sim.experiment(llm, personas, conditions, question)` per experiment (60 or more personas per condition, no_persona arm) -> `sim.experiment_frame` -> `sim.exaggeration_ratio(sim_g, human_g, family=...)` -> `sim.correct_effects(sim_g, method="ratio", human_g=..., family=...)` (leave-one-out) -> `sim.human_equivalent_n` -> `evaluate(task="continuous", primary="mae", baselines_oof=["T0"])` + `sim.effect_section` on the card | mean human effect of the other experiment families (a past-experiment archive): MAE 0.509 [0.335, 0.808] in Cohen's d | two columns, 16 experiments, people vs simulation: anchoring, base rate and WTA/WTP match, framing biases go to about 0: "the simulation knows the textbook biases but does not live them" | Twin-2K-500, 16 between-subject experiments: persona panel MAE 0.369, no-persona panel 0.334 [0.229, 0.516]; median abs(d_sim)/abs(d_human) 0.43 [0.34, 0.91], so the panel shrinks effects instead of exaggerating them; human retest 0.068 |

## 2. Pre-register (10 min)

Write the plan before you look at any result: question, data, split, primary metric with direction, baselines,
expected result with numbers, falsification ("the claim fails if ..."), stop rule.

```python
from simengine.research import Prereg, lock_prereg
Prereg(title="...", question="...", data="...", split="time: train <= 2024-06, test after",
       metrics="AUC (higher better)", baselines=["constant", "persistence", "logistic"],
       expected="stack beats logistic by >= 0.01 AUC", falsification="CI of stack - logistic includes 0",
       stop_rule="3 configs on dev, then one test run", notes="ours: stack").write(".onkayit/prereg.md")
lock_prereg(".onkayit/prereg.md")   # SHA-256 + timestamp in .onkayit/prereg.md.lock.json
```

- **Put it in the seal registry.** Write the prereg inside the directory you will pass as `prereg=` / `--prereg` in
  step 5 (`.onkayit/` by default). The card's selection-bias check reads the `.md` / `.txt` / `.json` / `.yaml`
  files there that are older than the latest seal and looks for the `ours` arm's name. A prereg kept elsewhere (a
  hand-written plan committed to git) is not seen, and the card says "ours not found in the pre-registration".
  `.onkayit/` is git-ignored; to keep the plan in git too, `git add -f .onkayit/prereg.md .onkayit/prereg.md.lock.json`.
- **Mark the arm: `ours: <arm>`.** When the text also names the baselines, the check needs `ours: stack` (or
  `primary: stack`) to know which arm is yours; without it the row reads "several method names".
- **Register against the strongest single baseline**, not a union or blend of baselines. In one rehearsal the
  registered T1+T2 union came out weaker than T2 alone, so the registered claim was against a weak baseline while
  the badge used T2.

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
  exact hit. `correlation_audit` checks `max_calls` (default 500) against the planned total (arms × keys ×
  samples) before the first call and names the number to pass: 300 keys × 2 arms needs `max_calls=600`. Save the
  result (`se.evidence.save_json(audit, "runs/audit.json")`) for step 5.
- **Licensed data: the seal registry holds a full copy of the predictions** (`<sha>.csv.gz`). Keep the registry
  git-ignored (`.onkayit` and `.prereg` are in this repo's `.gitignore`; add them to your team repo's too) or outside
  the repo, or seal without the copy: `se.seal(pred, "test", registry="runs/.onkayit", store_copy=False)` (chain record
  only) or `archive_dir="<outside the repo>"` (the chain keeps only the file name). CLI: `simengine seal pred.csv
  "test" --registry runs/.onkayit --no-copy`. `evaluate(prereg=)` and `verify_chain` need only the chain.
- **With a learner:** stack the LLM output with the learner (`ABStack`, `TwinStack`, `DevStack`). Also build the
  stack without the LLM column. On Twin that fair LLM-free stack was what turned a claimed +0.071 into +0.006.
  `st = methods.DevStack("binary").fit([learner_dev, llm_dev], y_dev)` then `st.predict([learner_test, llm_test])`
  (cross-fitted dev predictions in `st.dev_pred_`). `learner_dev` must come from a model that never saw the dev rows
  (fit on train, or `oof_predict`): in-sample dev predictions look perfect and the stack trusts them. One probability
  column is calibrated with `methods.BinaryCalibrator().fit(p_dev, y_dev).predict(p_test)` (Platt below 1,000 dev
  rows, isotonic from 1,000). Below 200 dev rows the Platt slope is noisy (it warns); `method="ratio"` fixes the
  level only and keeps the ranking.
- **Distributions with a known population:** `methods.ShrunkDeviation(distribution=True).fit(base=pop, llm=llm_seg,
  y=true_seg, ref=llm_pop)` on training segments keeps the level from the population distribution and adds the LLM's
  segment deviation (`llm_seg − ref`) times an alpha picked on those segments (0 = ignore the LLM); rows are
  renormalised. `predict(base=pop, llm=llm_new, ref=llm_pop)` for the test segments. `pop` must be known before
  the test answers are (a published overall, or the train respondents), never the mean of the held-out cells.
- **As a prior in a small test:** `EarlyTestBayes` after `calibrate_prior` on past tests. The gain disappears at
  800+ impressions per arm.
- **Personas:** `persona.from_table(df, id_col=..., segment=..., forbidden=[target])`, then `sim.survey` /
  `sim.ab_test`. Keep the default control arms (no persona, swapped persona). If the controls agree with the persona
  arm, the persona is not doing the work. Put a behaviour anchor (last period's value) in the persona when you have
  one: OE went from 0.618 to 0.860 with it, which is about persistence (0.848).
  - `from_table` stops on empty or duplicated ids with the counts; `id_col=None` uses row numbers.
  - **Free-text history** (reviews, complaints, support tickets): put each text as an event with `t` and `text` in
    `person["history"]` and render it with the `log_raw` injector (`simengine.persona.inject.from_logs`):
    `import simengine.persona.inject as pi; pi.get_injector("log_raw", n=5)(person, question)`. Events at or after
    the question's `cutoff` are hidden.
  - **Soft refusals.** Haiku refused 59 of 150 survey batches in plain text in D2. `survey` / `ask_panel` now detect
    a refusal (`se.looks_like_refusal`), keep it out of the cache and re-ask once with a research-forecast framing.
    Read `res["refused"]`, `res["unparsed"]` (counts, ints) and `res["refusal_rate"]` and put them on the card: the
    missing personas are not missing at random. The ids are in `res["refused_ids"]` / `res["unparsed_ids"]`. For your own direct calls, `llm.complete(..., cache_if=lambda t: not
    se.looks_like_refusal(t))`.
  - **Numeric / slider questions:** `{"id": "pol", "text": ..., "kind": "numeric", "min": 0, "max": 100}`; out of
    range counts as unparsed, never clipped. `free_text=False` drops the answer sentence (most of the output tokens;
    SSR is then skipped).
  - **Per-person stimulus** (each person saw a different message): `survey(panel, qs, llm=llm,
    stimulus_col="condName")` or `context={pid: text}`; personas are batched by stimulus.
  - **Many stimuli** (e.g. 900 films × 10 personas): all stimulus groups of an arm go out in one `llm.map`, with
    `parallel=` threads (default: the backend's `parallel`, e.g. `Claude(parallel=16)`, else 4). In D3 the old
    group-by-group loop ran at about 1 call/s; no hand-rolled `ThreadPoolExecutor` is needed now. Smoke-test 2
    stimuli first.
  - The ledger tag per arm is `survey:persona` / `survey:no_persona` / `survey:persona_swap`; `survey(..., tag="dev")`
    makes it `dev:survey:<arm>` so dev and test runs split in the spend log.

Skip discussion rounds and agent societies for accuracy: neither moved the result measurably.

## 4b. Trap check before you seal (10 min)

Every item below cost a rehearsal team time or a wrong claim in the blind rehearsals (22 Oct; 12-14 from D4). `tr`/`te` are the
train and test inputs, `FEATS` the columns your model uses, `G` the group column (country, store, wave).

1. **Post-response columns.** Columns asked after the stimulus (trust, identity, follow-ups) make CV look great and
   are missing in test. `sorted(set(FEATS) - set(te.columns))` must be `[]`; derive `FEATS` from `te.columns` and push
   train and test through the same prep function.
2. **CI on the wrong unit.** If test is new groups, split and bootstrap by group, not by person:
   `se.methods.oof_predict(model, tr, features=FEATS, target="y", group=G)` and `evaluate(..., group=G)` or
   `aggregate_by=G`. Person-clustered CIs are too narrow for a "generalises to new countries" claim.
3. **Coded non-answers.** Codes like "does not use social media" (`Share == 2`) are not 0. Check
   `tr[col].value_counts(dropna=False)` against the codebook and filter (`tr[col].isin([0, 1])`) before fitting.
4. **One dominant group.** If one group is ~20 % of rows, a GroupKFold fold is mostly that group and the pooled
   metric is mostly that group. Print `tr[G].value_counts(normalize=True).head(3)`, report the group-equal-weighted
   metric (`tr.assign(p=oof).groupby(G).apply(lambda d: d.y.corr(d.p, method="spearman")).mean()`) next to the pooled
   one, and say when the dominant group is absent from test.
5. **Per-person LLM calls.** 17k calls eat a 2-hour window and the budget, and raw personas were at chance in both
   rehearsals (Spearman 0.025 and −0.073). Measure on ~500-1,000 training people first; if there is no signal, use the
   LLM as a group-level prior (`GroupTransfer(group_features=prior_df)`, one call per group × a few framings) instead.
6. **Relative scales.** Items asked "compared with others in your country" (SES ladder, local income bands) are not
   comparable across groups. Use within-group ranks (`d[c + "_r"] = d.groupby(G)[c].rank(pct=True)`); if a group mean
   of such a column enters a group-level model, write the limitation on the card.
7. **Target leakage into the LLM prior.** Never ask the LLM for the target itself (or a near-synonym) as a "prior
   feature". `grep` the prior prompts for the target's name; if a close indicator stays, register it as its own arm
   with a memorisation probe (`se.evidence.ezber.correlation_audit`).
8. **"We beat it" without a paired CI.** Every "beats" line on the card needs ours − baseline with its CI over the
   right unit; with 20 groups a Spearman difference is very wide (D2: [−0.155, +0.687]). If the CI contains 0, the
   line goes under "inconclusive". Name the baseline the badge is against, and seal that baseline's test column too.
9. **Haiku empty replies and soft refusals.** With small `max_tokens`, Haiku 5.5 thinking can use the whole budget
   (`EmptyResponseError`); pass `thinking="off"` or `max_tokens >= 1024`. Persona panels can get "I can't invent
   answers for people who don't exist"; `survey` re-asks once and reports `refused` separately. Report the refusal
   rate: missing answers are not random.
10. **Background jobs.** Save the PID when you start a long job (`cmd > log/x.log 2>&1 & echo $! > log/x.pid`) and
    wait on `kill -0 "$(cat log/x.pid)"`, not `pgrep -f` (it matches your own shell). Never pipe a long job into
    `| tail` (the log and the exit code are lost). Start it with `python -u` (or `PYTHONUNBUFFERED=1`), or the log
    stays empty until the job exits. Before you submit, stop your own leftover PIDs.
11. **Features rebuilt inside folds** (member stats, SVD taste vectors computed without the fold's items). The OOF
    split must be the same folds: `oof_predict(model, dev, ..., group=item, folds=res.folds)`; its own GroupKFold
    leaves items that shared the SVD basis in training (D3: dev about 0.03 optimistic, ridge proxy 0.865 → 0.835,
    one run). Align each fold's SVD basis (`item_holdout_features(align="full")` / `procrustes_align`), and make
    absolute counts relative (`user_n_pct`, not `user_n`): a fold sees fewer ratings than the full train set.
    `GroupTransfer`'s default composition averages every numeric column, so pass `composition=[...]` if unaligned
    fold columns are in the table.
12. **One forecast origin.** A single dev origin can be lucky: in D4 the rule baselines moved between about 0.38 and
    0.45 WMAPE across origins 229-269 (one run). Use `backtest(model, panel, origins=[...])` and
    `summarize_backtest(bt, by="origin")`; tune anything (SES alpha, model settings) only on origins whose targets
    are before the dev origin.
    On the pooled backtest frame pass `evaluate(..., task="forecast", block_col="origin")` so the CIs resample
    origins and series together (two-way); with one origin the card warns "single forecast origin" instead.
13. **Mean vs median for L1 metrics.** WMAPE/MAE reward the median, which for low counts is below the mean:
    `point_for_metric(pred, metric="wmape", threshold=3)` lowered D4 series WMAPE by 0.0014 but worsened
    store x dept WMAPE by +0.0036 and pushed bias from −3.3 % to −4.0 % (one run). Report bias, and if totals are
    also graded seal the untransformed arm too.
14. **Items not on sale, and features per horizon.** Zeros before launch or while an item has no price are not
    demand: train only on cells where it is on sale, and predict 0 where it is not on sale in the test window.
    Each horizon h needs features known at the origin (`make_features` / `DirectForecaster` cut history at the
    origin); only truly known-ahead columns (planned price, calendar) go in `exog=` at the target week.

Rehearsal facts to calibrate expectations: the team with the plain kit took 63 min, the simengine-only team
37 min; the kit team's CV estimate matched the test (0.308 → 0.306), the simengine team's was optimistic
(0.311 → 0.283, inside the card's CI). The optimism is consistent with trap 4 (USA was ~20 % of the pooled CV rows and
absent from test), but that was not proven. Both teams beat the demographic GBM baseline at person level
(+0.066 [0.026, 0.101] and +0.042 [0.006, 0.084]); neither did at country level or on sharing.
In D3 (new-film cold start, simengine `018b1a2`, one run) the simengine-only team submitted in 19 min. On the test
films, graded by the rehearsal referee: film residual Spearman 0.897 [0.871, 0.917] against the hidden baseline's 0.567
(+0.330 [0.255, 0.411]); noise ceiling 0.956. Its own LLM-free arms were close behind (audience composition 0.880,
the one-line member-mean baseline 0.856), and its registered claim "the LLM adds over the LLM-free stack" went from
+0.050 [0.028, 0.075] on dev to +0.017 [−0.001, 0.035] on test, so its drop condition was met (borderline). Build your strongest LLM-free baseline
yourself, and write "passed on dev", not "passed".

## 5. Evidence card (15 min)

The frame has one row per scored unit: `truth`, one `pred_<arm>` per arm and a `split` column. A test-only frame
still gets `split="test"`; without the column the card's split check is skipped ("girdide 'bolum' yok"). With
`train` rows (truth and the same `pred_` columns) next to the `test` rows, metrics use the test rows only, the
automatic ladder fits on the train rows, and the card checks group overlap (with `group`) and time order (with `time`).

```python
frame = pd.DataFrame({"truth": te.y, "pred_stack": p_stack, "pred_llm": p_llm, "pred_persistence": te.last,
                      "pred_logistic": p_logit, "split": "test"})   # every arm the command below names
frame.to_csv("data/frame.csv", index=False)
```

```bash
simengine evaluate data/frame.csv --ours stack --raw llm --task binary --primary auc \
    --unlabeled-baselines persistence --prereg .onkayit --out runs/evidence --lang en
```

Group residual task (D3 shape) with the memorization audit, the blind arm and the registered drop conditions:

```bash
simengine evaluate data/dev_pairs.csv --ours ours --task continuous --primary spearman \
    --aggregate-by movie_id --offset-col member_mean --noise-ceiling variance_components \
    --baselines-oof T3 --unlabeled-baselines T0b --memorization runs/audit.json --blind-col ours_blind \
    --compare P1:L1 --compare "P2::c: P2 ≈ 0" --out runs/dev-evidence --lang en
```

```python
rep = se.evaluate(dev, ours="ours", task="continuous", primary="spearman", aggregate_by="movie_id",
                  offset_col="member_mean", noise_ceiling="variance_components", baselines_oof=["T3"],
                  unlabeled_baselines=["T0b"], memorization=audit, blind_col="pred_ours_blind",
                  compare=[("P1", "L1"), ("P2", None, "c: P2 ≈ 0")])
se.paired_diff(dev, "P1", "P_nopersona", metric="spearman", aggregate_by="movie_id",
               offset_col="member_mean")         # any other pair on the frame, same group bootstrap
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
- **Memorization.** Pass the step-4 audit (`memorization=audit` / `--memorization runs/audit.json`); without it the
  card says the audit was skipped and flags "not proven: leakage/memorization" (D3 #8). When the audit warns and ours
  uses an LLM, the badge reason says the gain may come from recognition (`--ours-no-llm` / `ours_uses_llm=False`
  when it does not). `blind_col=` adds ours − blind and warns when the blind arm keeps less than half of the gain.
- **Residuals and the noise ceiling.** `offset_col=` subtracts a per-row offset (the member's training mean) from
  truth and predictions before `aggregate_by`, so the card scores the group residual directly.
  `noise_ceiling="variance_components"` (older name `"split_half"`; with `aggregate_by`, spearman/pearson/r2) gives
  the sampling-noise ceiling of the group means over all scored groups (one-row groups included), so "gap closed" is computed without a human retest.
- **Registered comparisons.** `compare=[("P1", "L1"), ("P2", None, "label")]` / `--compare P1:L1` put each
  pre-registered drop condition on the card as a paired difference with its CI (`None`: against 0);
  `se.paired_diff(frame, a, b, aggregate_by=..., offset_col=...)` for any other pair (a report only holds its
  registered pairs).
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

Forecast mode (D4 shape): reshape the `backtest` output (`truth`, one `pred_<arm>` column per arm, the series id as
the bootstrap cluster) and pass the training cutoff so the contamination audit can run:

```bash
simengine evaluate data/bt.csv --ours lgbm --task forecast --primary wmape --unlabeled-baselines mean4,ses \
    --cutoff 2016-01-01 --ci basic --n-configs 6 --ours-no-llm --prereg .onkayit --out runs/fc-evidence --lang en
simengine evaluate data/bt.csv --ours lgbm --task forecast --primary wmape --aggregate-by store_dept_week \
    --aggregate sum --out runs/fc-totals --lang en          # hierarchy totals, second report
```

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
