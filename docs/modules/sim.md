# `simengine.sim`: LLM simulation runners

Runners that put an LLM in front of a question: a persona panel, an A/B test, a roundtable, a strategy
comparison, a direct question about each segment's answer distribution, and per-person prediction prompts.
Each runner takes any object that follows the `simengine.types.LLM` protocol and returns a JSON-safe dict.
A matching `*_frame` function turns that dict into an EvalFrame (`unit, truth, pred_<method>, segment, item,
group`) that `simengine.evidence.evaluate` accepts.

The measured numbers below come from the research branches. They hold on the data where they were measured.
On other data they are a prior, not a promise. The short version:

- **Asking the model directly for each segment's distribution works best for segment-level opinion data.** It
  closed 82% of the gap, against 55% for a persona panel.
- **Persona panels are near chance on A/B headline tests with Haiku** (0.52 direction accuracy).
- **Per-person purchase prediction from survey answers (L1) is below the "always no" rule.** Adding the
  person's own history (L2) helps, but a plain history model is still better.
- **Discussion rounds barely move answers.**

Run the control arms (`no_persona`, `persona_swap`) every time. They show whether the persona carries the
signal.

## Components

| component | file | what it does |
|---|---|---|
| `survey`, `survey_frame` | `survey.py` | Survey panel. Likert questions get a free-text answer plus a direct 1-5 score, and optionally SSR on the text. Numeric questions (`kind="numeric"`, `min`/`max`, e.g. a 0-100 slider) get a number. Multiple-choice questions get an option letter. `free_text=False` drops the free-text field. Each persona can see its own stimulus (`stimulus_col=` or `context={pid: text}`). Soft refusals are re-asked once and counted apart from parse failures. Control arms run by default. The result has bootstrap CIs per arm, a segment breakdown and quotes. See "Survey options" below. |
| `interview`, `interview_frame` | `interview.py` | Open-ended interview: one free-text question per persona, with the `no_persona` / `persona_swap` control arms and per-person stimuli, as in `survey`. Per arm: answers `{pid: text}` and refused / empty / error counts and ids. Import from `simengine.sim.interview` (not `simengine.persona.inject.interview`). See "Open-ended interview" below. |
| `SSR`, `SCALES` | `ssr.py` | Semantic similarity rating (Maier et al. 2025). It embeds the answer text, compares it with 5 anchor sentences, and returns a distribution and an expected score (or `to_probability`). `sentence-transformers` is imported lazily. The model loads offline from a local cache. `encoder=` takes any texts-to-vectors function. |
| `ab_test`, `ab_frame` | `ab.py` | PAARS-style A/B test on a persona panel. Each persona gives a 1-10 score per variant, a choice and a reason. Half of the panel sees the variants reversed, which gives the position effect. Output: winner, preference share, bootstrap win probability, sign test, control arms, `confidence` (`confidence_check`), and a direction check when `truth` is given. With `reference=` also `lift` and `ab_lift_frame` (see A/B options). |
| `paars_decision` | `ab.py` | Forced-choice decision from per-persona A/B scores, with a confidence threshold (0.65). |
| `ab_many` | `ab.py` | Many pairwise tests per call, with all readers in one prompt. The prompt is verbatim from the Upworthy rehearsal. Returns `p_a` with Laplace smoothing. Calls without a parseable vote are counted and warned about. |
| `roundtable`, `roundtable_frame` | `roundtable.py` | Multi-round discussion. Round 0 is independent. In later rounds each persona sees its own previous answer and a summary of its segment (or of the whole table). `CONFIRMATION_BIAS` and `KEEP_VIEW` are the measured stance instructions. |
| `roundtable_strategy` (alias `roundtable_conditions`), `roundtable_strategy_frame`, `hide_persona_fields` | `roundtable.py` | Strategy comparison at an interacting table: one `roundtable` per condition on the same panel (`{id: text}` or a list; `hide_fields=("condition",)` keeps the RCT arm out of the prompt). Output: condition × round × segment means (`table`), paired effect vs the `control` table per round (`effects`), last round − round 0 per persona with bootstrap CI and the change of the effect (`shift`), and `ranking` in round 0 and the last round. The frame helper reuses `roundtable_frame` (unit `<condition>|<unit>`); `effect=True` gives condition − control predictions to score against observed effects. CIs resample personas and ignore the within-table dependence of later rounds. |
| `strategy`, `strategy_frame` | `strategy.py` | Two or more scenarios × segments. Each persona gives a 0-100 intent and a reason per scenario. Output: ranking, win probability, sign test on the top two, and the segment whose profile diverges most. |
| `segment_distribution`, `distribution_prompt`, `FRAMINGS_NAMED`, `FRAMINGS_HIDDEN` | `distribution.py` | No personas. Makes one call per segment × question × framing ("estimate the percentage of these respondents who would choose each option"), then averages the 3 framings. An invalid reply becomes a uniform vector (or is skipped with `fallback="skip"`); the filled share is recorded per cell. `repeats=k` passes `sample=0..k-1` (independent draws, per-sample dists kept). See "Distribution runners" below. |
| `segment_rate_prior`, `rate_prompt`, `parse_rate` | `rate_prior.py` | No personas. Asks each segment's event rate ("the percentage of them for which ... happens", `{"pct": n}`; `parse_pct=False` asks a 0-1 proportion) `samples` times (`sample=0..k-1`, independent draws) and returns a segment × (`mean`, `sd`, `n_ok`, `n_failed`) table, rates in [0, 1], ready as `GroupTransfer(group_features=prior[["mean"]])`. `blind=True` (or `{key: hidden wording}`) adds a `_blind` arm with the same event and an anonymous segment. Failed answers are never clipped. See "Segment rate prior" below. |
| `effect_forecast`, `effect_frame`, `memorisation_check`, `EffectForecast` | `effect.py` | No personas. Asks for every condition's effect against control in one JSON object per arm × framing × sample (independent draws via `sample=k`), averages per condition, and returns a condition × arm table: hidden arm always, named arm (study named) with `named=`, plus named − hidden as a memorisation probe and refused / unparsed / failed / invalid counts. Import from `simengine.sim.effect`. See "Direct effect forecasts" below. |
| `experiment`, `experiment_frame`, `exaggeration_ratio`, `correct_effects`, `human_equivalent_n`, `effect_section` | `experiment.py` | Simulated randomized experiment on a persona panel: each persona is assigned one condition (balanced, seeded; `between=False` = every condition, paired) and answers through `survey`. Per condition mean / sd / n / refused / unparsed, per effect Hedges g with SE and CI and the no_persona arm's g. Calibration against human effects: median \|sim\|/\|human\|, sim ~ human slope with a family-clustered CI, leave-one-family-out ratio / affine correction, "worth ~N humans" and a card section. See "Experiments and effect calibration" below. |
| `persona_distribution` | `distribution.py` | Per-persona probability vectors, many respondents per call, with tables filled within each segment. Also returns the segment means, one set per `segments=` scheme, and `survey`-style refused / unparsed accounting with ids. `person_distribution_frame` gives persona × question rows. |
| `distribution_frame`, `w1_ordinal` | `distribution.py` | Distribution EvalFrame (list-valued `truth` / `pred_llm_dist`, numeric `filled_llm_dist`; `scheme=` picks a `persona_distribution` segment scheme), and the normalised ordinal W1 used for the makro numbers. |
| `robustness`, `robustness_frame` | `robustness.py` | Retest / framing / option-order robustness of `segment_distribution`, scaled by the group signal (W1 between each group and the population). Per condition: mean W1 against the base run, signed order shift, ratio to the signal with question-clustered CIs; summary ratio = worst condition; a card row (`tur="robustness"`). Import from `simengine.sim.robustness`. See "Robustness" below. |
| `predict_people`, `prediction_frame` | `prediction.py` | Per-person prediction. L0 is a no-persona population prior over `l0_samples` samples. L1 uses the person's own survey answers. L2 is L1 plus the person's past behaviour in the same items. SSR asks for one first-person sentence per item. It takes independent `samples` (passed as `sample=j`) and re-asks with "JSON only" when parsing fails. `kind="numeric"` with `fields=[NumericField(name, lo, hi, unit?)]` (`from simengine.sim import NumericField, numeric_fields`; levels L0/L1/L2) asks for every field in one JSON reply per person, all persons in one `llm.map` batch (`ask_many`, `return_exceptions=True`); a value outside its range, absent or not one whole number (strings like `"1,250"`, `"120 USD"` are read; `"5-10"`, `"7/10"`, `"45%"`, lists are not) becomes None/NaN (never clipped) and is counted per field in `invalid` (`ok`, `out_of_range`, `unparsed`, `no_reply`). L2 summary histories list all their own items. `prediction_frame` then gives one row per person × field with `lo`, `hi`, `status`. |
| `PredictionSpec`, `AMAZON`, `prompt_l0/l1/l2/ssr`, `profile_text`, `history_text` | `prediction.py` | All prompt wording. `PredictionSpec()` is neutral. `AMAZON` reproduces the measured talp73 prompts exactly (checked string-for-string against `talp73/kod/llm/istem.py` and `ssr.py`). |
| `batch_calibrate` | `prediction.py` | Label-free batch calibration: `logit(p) − mean_item logit(p) + logit(prior_rate)`, clipped at 1e-3. It matches `talp73/kod/llm/kalibre.py` to 0.0 max abs difference on a random check. |
| `retest_stats` | `prediction.py` | Test-retest of repeated samples: within-cell sd, share of identical cells, pairwise r, per-item signal vs noise, Spearman-Brown. `kind="distribution"` reduces each vector to E[position] first and adds the within-cell W1. |
| `ask_panel`, `build_prompts`, `Usage` | `panel.py` | Batched persona questioning: one JSON object per persona id. It accepts an array, a single object, an `{id: {...}}` mapping or a wrapper. It remaps ids by position only when no id matched and the counts are equal. Missing answers are counted, not invented. Soft refusals are detected, kept out of the disk cache and re-asked once with `REFRAME_NOTE`. Every call carries a ledger `tag`. |
| `looks_like_refusal`, `REFRAME_NOTE` | `panel.py` | Refusal check used by `ask_panel`: `simengine.llm.looks_like_refusal`, or a small local regex if that import fails. It is only applied to replies that have no usable JSON answer. `REFRAME_NOTE` is the research-forecast wording used for the re-ask. |
| `print_events`, `Emitter` | `_stats.py` | Every runner takes `on_event=`. `print_events` is a ready callback that prints one flushed line per event (`HH:MM:SS [runner] stage …` / `row <pid> <segment> <value>: …` / `done …`), so a background run (`python run.py > log`) shows progress. Other prints need `python -u` or `PYTHONUNBUFFERED=1`. |
| `schema_responder` | `panel.py` | Deterministic `FakeLLM` responder for every prompt in this module (dry runs and tests). It is never a measurement. |
| `as_personas`, `PersonaView`, `control_personas`, `control_plan` | `personas.py` | Duck-typed persona adapter for TALP-schema dicts, objects with `to_dict()`, flat dicts and strings. Control arms: `no_persona` gets a neutral "typical adult consumer" text. `persona_swap` applies a cyclic shift: each persona keeps its own id, name and tagline and takes the next persona's content. `control_plan` keeps the control arms within the main arm's call count. |
| `FieldSpec`, `validate_answers`, `ask_constrained`, `ConstrainedResult`, `completion_summary` | `constrained.py` | Schema-constrained JSON answers (int / float / choice / text, bounds, choices by value or display text). A failed check re-prompts with the error text, up to 8 calls; 4+ fields are asked one at a time with the earlier answers in the prompt. A field that never validates stays missing and is logged. `completion_summary` gives n_planned, n_completed and per-field failure counts per condition. |
| `parse_json`, `normalize_probs`, `parse_distribution`, `parse_item_probs`, `parse_text_fields` | `parse.py` | Robust parsers. They handle fences, prose around the JSON, trailing commas, `[see below]`-style brackets in prose, and truncated arrays (complete objects are kept). Probability keys can be `"1"`, `"option 1"`, a letter or a label, and values can carry `%`. A missing option, a negative value, a zero sum or a wrong length gives `None`. |

Ported from (read-only research code, `uretim/<branch>/`):
- `sahne/kod/sahne/kaynak.py` (batched prompts, `json_coz`, the fake source), `sema.py` (persona rendering),
  `ssr.py`, `kosucu/{ortak,panel,ab,roundtable,strateji}.py`, `prova/upworthy.py` (`ab_many` prompt)
- `makro/llmabm/kod/c1/k2.py` (distribution prompt, ACIK/GIZLI framings, OLAS_ISTEM),
  `makro/kod/kit/yontemler/llm_dagilim.py` (framing average, invalid → uniform),
  `makro/llmabm/kod/c1/masa.py` (stance instructions)
- `talp73/kod/llm/{istem.py, kos.py, ssr.py, kalibre.py, tekrar.py}` (L0/L1/L2/SSR prompts, validator,
  samples, calibration, retest)
- `deney/kod/kit/persona_talp.py` (`paars_decision`)
- replicant (pyreplicant 0.3.0, MIT) `src/replicant/otree/bot.py` (`FormController.validate`, `_decide_batch`, `_decide_sequential`, `_build_prompt`, `_build_feedback`, `MAX_LLM_RETRIES`); see "Schema-constrained answers" below

## Minimal usage

```python
from simengine.llm import Claude, FakeLLM
import simengine.sim as sim

llm = Claude(model="claude-haiku-5-5", effort="low", cache_dir=".cache", budget_usd=2)
# llm = FakeLLM(sim.schema_responder())             # offline dry run, same code path

# 1. Segment distribution: the strongest method measured for segment-level opinion data
segs = {"D-young": "Democrats aged 18 to 44", "R-old": "Republicans aged 45 or older"}
qs = [{"id": "econ", "text": "Over the past year the nation's economy has ...",
       "options": ["Gotten much better", "Gotten better", "Stayed the same", "Gotten worse", "Gotten much worse"]}]
d = sim.segment_distribution(llm, segs, qs, population="US adults")
frame = sim.distribution_frame(d, truth=real_segment_shares)          # {segment: {qid: [p...]}}

# 2. Survey panel with controls
panel = [{"id": "p1", "text": "34, nurse, rents in Izmir, price-sensitive", "segment": "young"}, ...]
res = sim.survey(panel, "Would you try a grocery subscription at 299 TL/month?", llm=llm)
res["questions"][0]["arms"]          # persona vs no_persona vs persona_swap

# 3. A/B with a confidence verdict
ab = sim.ab_test(panel, [{"id": "A", "text": "Free delivery over 300 TL"}, {"id": "B", "text": "15% off tonight"}],
                 llm=llm, context="Friday 18:00 push notification")
ab["winner"], ab["confidence"]["low"], ab["confidence"]["reasons"]

# 4. Per-person prediction (survey answers -> purchase probability), then label-free calibration
res = sim.predict_people(llm, {"u1": {"Age group": "25 - 34 years", ...}}, ["ABIS_BOOK", "PET_FOOD"],
                         level="L1", spec=sim.AMAZON)
f = sim.prediction_frame(res, truth=observed)                       # {uid: {item: 0/1}}
f = sim.batch_calibrate(f, prior_rates=out_of_sample_rates, col="pred_L1")

from simengine.evidence import evaluate
report = evaluate(f, title="L1 purchase", ours="L1")
```

## Survey options

Each of these came from the D2 blind rehearsal (22 Oct, `SIMENGINE-EKSIK.md` #6-#9).

**Numeric questions and `free_text=False`.** A question with `kind="numeric"` (also accepted: `number`,
`slider`) asks for a number in `[min, max]`. The defaults are 0 and 100. `labels=[at_min, at_max]` names the
two ends. A value outside the range, or a value that cannot be parsed, counts as unparsed. It is never clipped
into range. `free_text` (per question, or for the whole survey) defaults to True. With False, the Likert answer
sentence or the numeric reason is not asked for, which cuts most of the output tokens. SSR then has no text, so
only the direct score is used, and a warning says so. In the rehearsal, the free text was about 2,600 output
tokens per call, most of the cost.

```python
q = {"id": "pol", "text": "How much do you support this policy?", "kind": "numeric", "min": 0, "max": 100,
     "labels": ["fully oppose", "fully support"], "free_text": False}
res = sim.survey(panel, [q], llm=llm)
res["questions"][0]["overall"]["mean"]      # {value, low, high, n}; also "median", "sd"
res["questions"][0]["unparsed"]             # answers out of range or unreadable
sim.survey_frame(res)                        # pred_value (persona), pred_no_persona, pred_persona_swap
```

**Per-person stimulus.** Use `stimulus_col="message"` to read each persona's own stimulus from a flat-dict key,
a TALP facet (`from_table` puts every column there), an identity field, a trait, or an attribute. You can also
pass `context={persona_id: text}`. A string `context` is still shown to everyone. Control arms keep each id's
own stimulus. With `stimulus_batching="group"` (the default), only personas with the same stimulus share a
batch, and that stimulus goes into the shared context block. This costs a few more calls (at least one per
distinct stimulus). With `"inline"`, stimuli are mixed in one batch, and each one is shown under its person as
`Shown to this person: ...`. The result has `stimuli` ({pid: text}), and the person-level frame has a
`stimulus` column. With `stimulus_col=`, that field is removed from the profile text of every arm, so the
stimulus appears once per person (as context) and the `persona_swap` body does not carry the next persona's
stimulus.

```python
res = sim.survey(panel, ["Would you share this post?"], llm=llm, stimulus_col="condName",
                 context="You are scrolling your social media feed.")
```

**Many stimuli run in parallel (D3 rehearsal #4).** Before this fix, `stimulus_batching="group"` sent one
`ask_panel` per stimulus group and waited for each one. With one batch per stimulus (one film per batch), the
run was in effect single-threaded: in the D3 rehearsal (one run), 900 films × 3 arms went at about 1 call/s.
Now `survey` builds one plan per arm from every batch of every stimulus group and sends it through one
`ask_many` (one `llm.map`). Batches are split only by question group (`question_group=`) and by `max_tokens`
value, because `max_tokens` is part of the cache key. Groups of different sizes get different `max_tokens`.
Prompts, `max_tokens`, answers, refusal, empty-reply and error handling and `usage` are the same as in the
serial version. A test checks this against the old loop. The reframed re-ask and the empty-reply re-ask are
also one map each. `parallel=` sets the thread count. The default, None, uses the backend's own `parallel`
attribute (`Claude(parallel=16)`), or 4 when the backend has none. Before this fix it was always 4.

```python
llm = se.llm.Claude(model="claude-haiku-5-5", thinking="off", parallel=16, cache_dir=".cache")
res = sim.survey(panel, [q], llm=llm, stimulus_col="film", batch=10)      # 16 threads, one map per arm
res = sim.survey(panel, [q], llm=llm, stimulus_col="film", parallel=8)    # override
```

**Soft refusals.** In the rehearsal, Haiku 5.5 answered 59 of 150 survey batches with a plain-text refusal
("I can't produce this as a survey dataset ..."). These had `stop_reason` end_turn and no error. Without the
fix, 137 of 602 personas went unanswered and the refusals were cached. Now `ask_panel` treats a reply with no
usable JSON answer that reads like a refusal (or a `RefusalError` slot) as refused. It passes
`cache_if=` to the backend, so the refusal is returned but not cached and a rerun asks again. It then re-asks
that batch once, with `REFRAME_NOTE` ("This is a research forecast. Estimate how a person with these
characteristics would most likely answer; give your best guess ...") added to the system and user prompts. The
re-ask has a different prompt, so it gets its own cache entry. Options:
`refusal_retry=False` skips the re-ask, and `cache_refusals=True` keeps refusals in the cache.
The survey result reports refusals and parse failures separately:

| key | meaning |
|---|---|
| `refused` | int: number of main-arm personas left unanswered because their batch was still refused after the re-ask |
| `unparsed` | int: number of main-arm personas left unanswered for any other reason (no JSON, missing id, empty or failed call) |
| `refused_ids`, `unparsed_ids` | the same personas as id lists, in panel order (`len(res["refused_ids"]) == res["refused"]`) |
| `refusal_rate` | share of main-arm first-pass batches that were refused |
| `unparsed_items` | persona × question answers that were present but invalid (e.g. out of range) |
| `refusals` | per arm: `calls`, `refused_calls`, `recovered_calls`, `refusal_rate`, `refused_personas` |

The warnings list both counts, for example `137 personas unanswered (12 refused after the reframed re-ask,
125 unparsed or missing)`. `usage` also has `refused_calls`, `reframed_calls`, `recovered_calls`, `refused`
and `refusal_rate`. `Usage.missing` no longer includes refused personas. We have not yet measured how many
refusals the re-ask recovers.

**Empty replies and failed calls.** An `EmptyResponseError` slot (thinking used the whole `max_tokens`) is
re-asked once with `max_tokens=max(2 * max_tokens, 4000)` (at most 32000), or with `empty_retry_kw=`. A batch
that still has no reply, or that failed with another `LLMError`, counts as unparsed instead of aborting the
survey. `usage` reports `empty_calls`, `empty_recovered`, `error_calls` and `last_error`, and a warning is
added. The survey raises the first error only when no main-arm persona got an answer. Programming errors
(anything that is not an `LLMError`) are still raised.

```python
res = sim.survey(panel, qs, llm=llm, empty_retry_kw={"thinking": "off"})   # Haiku: re-ask without thinking
res["usage"]["empty_calls"], res["usage"]["error_calls"]
```

**Numeric answers.** `"1e2"` parses as 100. When every answer is between -1 and 1 on a scale wider than 1
(for example 0.75 on a 0-100 slider), a warning says the model may be answering with fractions.
Prompts without any free-text field no longer carry the "Free-text fields in ..." line.

**Shared call path (`ask_many`).** `segment_distribution`, `persona_distribution`, `predict_people` and
`ab_many` call `sim.panel.ask_many`, the same path as `ask_panel`. Soft refusals are kept out of the cache
and re-asked once with `REFRAME_NOTE_GENERIC` ("This is a research forecast. Give your best estimate, in the
requested JSON format."). Empty replies are re-asked with a larger budget. A cell or person still without a
reply counts as invalid / failed. Each of these runners takes `ask_kw=` (`refusal_retry`, `cache_refusals`,
`empty_retry`, `empty_retry_kw`), and `ab_many` also takes `usage=`.

```python
res = sim.segment_distribution(llm, segs, qs, ask_kw={"empty_retry_kw": {"thinking": "off"}})
res["usage"]["refused_calls"], res["usage"]["empty_calls"], res["invalid"]
statuses = []
replies = sim.ask_many(llm, prompts, max_tokens=300, statuses=statuses)   # None where refused/empty/failed
```

**Distribution runners: repeats, filled cells, refusal accounting** (G5 / G7 recipe rehearsals).

```python
d = sim.segment_distribution(llm, segs, qs, repeats=3)       # sample=0, 1, 2 passed to llm.complete
d["dist"]                     # averaged over framings and repeats, as before
d["sample_dist"], d["samples"]  # one dist per repeat; long rows (unit = segment) for retest_stats
sim.retest_stats(d, kind="distribution")                     # E[position] per vector + within-cell W1
frame = sim.distribution_frame(d, truth=real, method="c3")    # adds filled_c3
p = sim.persona_distribution(llm, personas, qs, segments={"age": "age_band", "income": lambda x: x["inc"]})
p["refused"], p["unparsed"], p["refusal_rate"], p["refused_ids"], p["unparsed_ids"], p["invalid_ids"]
sim.distribution_frame(p, truth=real_by_age, scheme="age")
from simengine.sim.distribution import person_distribution_frame
person_distribution_frame(p)  # pid x item: p, status, segment, one column per scheme, E_pos
```

- **Repeats.** `repeats=k` asks every prompt k times with `sample=i` (offset by `ask_kw["sample"]`). Claude keys
  its disk cache on `sample`, so repeat i is a new draw, not the stored reply; `repeats=1` passes no `sample` and
  keeps the old cache entry. Calls = framings × segments × questions × k. FakeLLM ignores `sample`, so offline
  repeats are identical.
- **"Not sure" options (D6 #5).** The prompt says to estimate each option's share `(ignore "not sure" answers)`,
  the measured wording. When the option list itself scores "Not sure / neither" or "Don't know", that line tells
  the model to put ~0 there. `include_unsure=True` drops the parenthetical (`DIST_TEMPLATE` minus `UNSURE_NOTE`).
  The default `None` leaves the prompt byte-identical, so cache keys and prompt hashes do not change, and warns
  once per call, naming each option that looks like not sure / unsure / don't know / neither / no opinion /
  undecided (a bare "Neither"; the Likert midpoint "Neither agree nor disagree" does not warn). `False` keeps the
  instruction and turns the warning off. `distribution_prompt(..., include_unsure=)` and `robustness(...,
  include_unsure=)` take the same flag, and `DistributionFake` parses both variants.
- **Filled cells.** An invalid answer replaced by a uniform vector (`fallback="uniform"`) is marked in `cells`
  (`filled=True`) and counted: `filled` {segment: {question: share of the cell's answers}} and `filled_share`.
  `distribution_frame` writes it as the numeric `filled_<method>` column (NaN for a result without it), so an
  accuracy gap can be checked against the share that is filler. `fallback="skip"` fills nothing.
- **Persona accounting** follows `survey`: `refused` (batch still refused after the reframed re-ask), `unparsed`
  (any other unusable answer), `refused + unparsed == invalid` (old key kept), id lists in persona order,
  `refusal_rate` (batches refused at the first ask / batches), `status` per persona, and `filled` per cell.
- **Segment schemes.** `segments={scheme: column | callable}` re-aggregates the same persona vectors in each scheme
  (no extra calls): `dist_by`, `filled_by`, `segments_by`. A column is looked up as in `survey(stimulus_col=)`; a
  callable gets the original persona item; a missing value is labelled `"NA"`. `dist` / `segments` stay the
  persona segment.
- **Retest of distributions.** `retest_stats(samples, kind="distribution", pos=None)` reduces each vector to
  E[position] (`pos`: None = option index 0..K-1, a sequence, or {item: positions}) and then runs the usual
  statistics; it adds `within_cell_w1` / `within_cell_w1_max` (mean pairwise W1 between a cell's samples, divided
  by the position range: `w1_ordinal` for equally spaced options). A missing vector becomes NaN. The default
  `kind="probability"` is unchanged, and raises a ValueError naming `kind="distribution"` when `p` holds vectors.
  Each `samples` row carries `filled` (share of that repeat's framing answers that are uniform filler) and the
  retest reports `filled_share`: filler counts as an answer there, so drop rows with `filled > 0` to retest real
  replies only.
- **Scheme names** must not be `pid`, `item`, `p`, `status`, `segment` or `E_pos` (the person frame's columns).

**Cost per arm.** `ask_panel` passes `tag=` to `llm.map`. `survey` uses `survey:persona`,
`survey:no_persona` and `survey:persona_swap`. `survey(..., tag="dev")` makes them `dev:survey:<arm>`, so the
dev and test runs and the named and unnamed arms can be told apart in the shared spend log (D3 rehearsal #3).
If the backend has its own `tag`, it is joined in front once: `<backend tag>:dev:survey:<arm>`. A simengine backend
gets the raw tag and joins it with `compose_tag`; another backend with a string `.tag` gets it joined by `ask_panel`.
The tag text is never inspected, so `Claude(tag="survey")` + `survey()` logs `survey:survey:persona`. The ledger's
`tag_prefix` (`Claude(tag_prefix=...)` or env `SIMENGINE_SPEND_TAG_PREFIX`) is then put in front in the TSV
column. Other runners use their `ask_panel` tag, for example `strategy:<scenario>`. `ask_panel(..., llm_tag=...)`
overrides it.

```python
llm = se.llm.Claude(tag="L1", tag_prefix="d3/", spend_log="spend.tsv", agent="team")
sim.survey(panel, [q], llm=llm, tag="dev")     # TSV tag column: d3/L1:dev:survey:persona, ...:no_persona, ...
```

## A/B options (`ab.py`)

**Variants.** A list of `{"id", "text"}` dicts, a list of plain texts (ids A, B, ...), or a mapping
`{id: text}`. Any other type (a string, a set, a number) raises `TypeError`.

**`controls=` means persona-control arms** (`no_persona`, `persona_swap`), not a control creative. A control
creative is an ordinary variant; name it in `reference=`.

**Lift against a reference.** `ab_test(..., reference="control")` adds `lift`: for every other variant, the
paired mean score difference vs the reference (`abs`, 1-10 points) and relative to the reference mean (`rel`),
each with a paired per-persona percentile bootstrap CI (`abs_ci`, `rel_ci`), `prob_better` (bootstrap
P(mean difference > 0)) and a sign-test `p` on the per-persona differences. `ab_lift_frame(res)` gives one row
per variant. Without `reference` the result is unchanged.

**Pairwise win matrix.** `pairwise_win_prob[a][b]` = P(mean score of a > mean score of b), personas resampled
paired, ties split; also per arm in `arms[arm]["pairwise_win_prob"]`. `win_prob` is P(variant has the highest
mean of all), so it is near 0 for a weak control; use the pairwise matrix as a pairwise prior (e.g. for
`EarlyTestBayes`).

**Names.** `tendency` and `winner` hold the same value (highest mean score). `tendency` is the preferred name;
`winner` is kept as an alias. `overall.choice_winner` is the forced-choice winner; when it differs from the score
winner, `confidence` gets the reason `choice_disagrees`.

**`ab_frame` columns.** One row per variant, aggregated over personas: `unit` = `<test_id>|<variant>` (`ab|A`
with the default test id), `item` and `group` = test id, `segment` = `all`; per-persona answers stay in
`res["answers"]`. `pred_score`, `pred_share`, `pred_win` are the persona arm. `pred_no_persona` and
`pred_persona_swap` are persona-control arms, not competing methods: leave them out of `evaluate(...,
methods=[...])` or pass them as `unlabeled_baselines=["no_persona", "persona_swap"]`.

**`ab_many` vote accounting.** A reply without JSON counts in `usage.unparsed_calls`; each (reader, test) pair
without a vote counts in `usage.missing` (`usage.refused` when the call stayed refused). A call with no vote at
all raises a `RuntimeWarning` and is listed in `df.attrs["warnings"]`; its tests get `p_a = 0.5`, `n_votes = 0`.
`schema_responder` answers the default `ab_many` prompt (one vote per reader and test).

**`ab_many` reader length.** Each reader line is the persona text on one line, cut at `reader_chars=400` characters (`READER_CHARS`, the measured Upworthy run; the default prompt is byte-identical to the earlier versions). When texts are cut, one `UserWarning` gives the count ("3 of 8 persona texts cut ...") and is listed in `df.attrs["warnings"]`; `reader_chars=None` keeps whole texts, at more input tokens per call.

```python
from simengine.sim.ab import ab_lift_frame

res = sim.ab_test(panel, {"control": "Current banner", "C1": "Free delivery tonight", "C2": "15% off"},
                  llm=llm, reference="control")
ab_lift_frame(res)                          # variant, abs, abs_low, abs_high, rel, ..., prob_better, p
res["pairwise_win_prob"]["C1"]["control"]   # P(C1 scores above control)
```

## Open-ended interview (`interview.py`)

A plain "why?" question had no runner: `survey` asks Likert, numeric and choice questions, so the G6 rehearsal
(#3) built the persona, no-persona and swap arms by hand with `llm.map`. `interview` asks one open-ended question
with the same pieces as `survey`: batched panel prompts, `control_personas` / `control_plan` for the control arms,
`stimulus_col=` or a per-person dict for stimuli (the stimulus is dropped from the profile text in every arm), and
the soft-refusal and empty-reply re-asks.

```python
from simengine.sim.interview import interview, interview_frame

res = interview(panel, "Why did you stop ordering last month?", llm=llm, max_chars=400, batch=8)
res = interview(panel, q, llm=llm, stimulus_col="message")               # each persona sees its own message
res = interview(panel, q, llm=llm, stimuli={pid: text}, arms=("persona", "swap"))
res["answers"]                          # {pid: text}, main arm ("persona" when run)
res["controls"]["answers"]["no_persona"]
res["refused_ids"], res["empty_ids"], res["error_ids"]     # main arm; every arm in res["arm_stats"]
interview_frame(res, truth={pid: label})                    # one row per person x arm
```

- **Buckets.** In each arm every asked persona ends in exactly one bucket. *answered*: a non-blank answer, cut to
  `max_chars` (the cuts are counted in `truncated` and reported in a warning). *refused*: its batch stayed refused
  after the research-forecast re-ask, or its answer reads like the model declining ("As an AI I can't answer for
  this person", `looks_like_refusal`; counted in `refused_in_answer`, text kept in
  `arm_stats[arm]["refused_answers"]`, `answer_refusal=False` keeps it as an answer). *error*: its batch got no
  reply (an empty reply after the larger-max_tokens re-ask, or another LLM error). *empty*: the batch replied, but
  this persona's answer is missing, blank or not text. A persona declining in its own voice ("I'd rather not say")
  is an answer. When the main arm has no answer and every failure is an error, the first LLM error is raised.
- **Control arms.** `arms` takes any of `persona`, `no_persona`, `persona_swap` (aliases as in `survey`). The
  control arms follow `control_plan`, so `persona_swap` may run with a larger batch or as a subsample
  (`res["controls"]["plan"]`; check it before reading the persona - swap gap). With one persona the swap arm is
  skipped with a warning. The defaults roughly double the calls.
- **Ledger tags.** As in `survey`: `interview:persona`, `interview:no_persona`, `interview:persona_swap` by
  default; `tag="dev"` gives `dev:interview:persona` (survey: `dev:survey:persona`). The backend tag is joined in
  front as for `survey`.
- **Frame.** `interview_frame` columns: `unit` (`<pid>|<arm>`), `truth`, `pred_text`, `status` (answered | refused
  | empty | error | not_asked; not_asked = outside a subsampled or skipped swap arm), `chars`, `arm`, `segment`,
  `item` (= arm), `group` (= pid), and `stimulus` when the run had per-person stimuli. Scoring free text against a
  truth label (coding, embedding similarity) is left to the caller.
- `simengine.persona.inject.interview` is a different module (interview-transcript persona injectors).
- Tests: `tests/test_sim_interview.py` (arms and aliases, control prompts, ledger tags, the four buckets, refusal
  recovery, refusals inside the answer, truncation, stimuli, events, the frame).

## When it works and when it does not (measured)

All numbers are copied from the branch reports. Nothing here was re-measured by this module.
`[a; b]` is a 95% bootstrap CI.

### Segment distributions (makro C1: CES 2020, 4 party × age segments, 8 items, claude-haiku-5-5)
Source: `makro/RAPOR.md` (C1 table, K2-5, K2-6, paired measurement). Metric: normalised W1, lower is better.
"Gap closed" is measured from the general-population marginal (W1 0.206) to a 37-person real survey of the
segment (0.050).

| method | W1 | gap closed |
|---|---|---|
| general marginal (ignore segment) | 0.206 | 0 |
| **`segment_distribution`, named survey, 3 framings averaged** | **0.077** | **0.818** |
| framings v1 / v2 / v3 ("think about minority views") | 0.077 / 0.075 / 0.096 | 0.82 / 0.83 / 0.70 |
| hidden framing (no survey name, no year) | 0.084 | 0.774 |
| `persona_distribution` (per-persona vectors) | 0.088 | 0.751 |
| persona panel round 0 (148 personas, batched tables) | 0.120 | 0.55 [0.44; 0.60] |
| persona panel, one call per persona | 0.142 | 0.41 |
| same party, other age group, real answers | 0.040 | 1.06 |

- Direct distribution minus panel: ΔW1 −0.043, item-cluster CI [−0.072; −0.017].
- It is worse than the general marginal on 2 of 8 items: Congress approval and household income. On these
  items the segments really do think alike.
- The direct distribution is worth about a 12-15-person real survey of the same segment. The panel is worth
  about 5-6 people.
- Memorisation is not ruled out. The hidden framing is almost as good as the named one, and party-level poll
  numbers for 2020 are widely published. The method was not tested on a wave after the model's training
  cutoff. Seal predictions before looking at the truth.
- Option order reversed in the panel: W1 0.119, no order effect, but 8% of the answers were invalid JSON.

### Persona controls and discussion (same data; `makro/RAPOR.md`, persona control and paired measurement)
- Persona panel closes 0.55 of the gap. With no persona (age and party label only) it closes 0.45. Shuffling
  personas across segments drops it to 0.25. About 0.45 of the gap comes from the segment label alone, and
  the persona detail adds +0.10, which is not significant.
- Individual level: the segment majority rule gets 0.656 exact match, the LLM round 0 gets 0.613. Do not
  claim individual-level accuracy from a panel.
- `roundtable`: 5 plain discussion rounds change 5-8% of answers. Round 5 ΔW1 −0.005 [−0.012; +0.002].
  Discussion does not improve the distribution.
- With the confirmation-bias instruction (`CONFIRMATION_BIAS`, round 3) ΔW1 is −0.007 [−0.012; −0.003]
  (0.120 → 0.113). The gain is small, and there was no multiple-test correction.

### A/B tests (sahne + deney: Upworthy headline tests)
- `sahne/RAPOR.md`: Haiku 8-persona panel direction accuracy 0.522 [0.468; 0.575] on 300 tests. The
  no-persona Haiku readers got 0.528, and chance is 0.50. The deney stack (TF-IDF + Opus) got 0.807
  [0.760; 0.853]. For A/B text tests use the methods module, not a persona panel.
- `deney/RAPOR.md`: Haiku persona panel 0.497-0.550 (chance). An Opus 10-persona panel got 0.685
  [0.633; 0.737], below asking Opus directly (0.765). A calibrated persona-panel prior still saved samples:
  1.41× [1.12; 1.81] (llm300), 1.34× [1.12; 1.57] (ek600).
- `sahne/RAPOR.md` Turkish A/B example: the panel mean favoured A (win probability 0.84), but the preference
  share was 6-6, the no-persona arm picked B (0.65) and the swapped-persona arm picked B (0.94). This is the
  case `confidence_check` flags as "decision depends on persona assignment".

### Survey answers vs behaviour (sahne: Open e-commerce, n = 198, Pearson on monthly order days)
Source: `sahne/RAPOR.md`, rehearsal (i).
- Demographic baseline 0.280. Haiku persona panel 0.618 [0.513; 0.696], below the person's own stated
  answer (0.649).
- Persona plus the 2021 behaviour anchor 0.860 [0.807; 0.899], but label-free persistence gets 0.848. On a
  second sample (seed 1), persistence 0.897 beat persona plus anchor 0.879 (difference −0.018
  [−0.051; +0.019]).
- Persona swap −0.032 / 0.090: the signal comes from the person's own facts.
- SSR embedding model: bge-base Spearman 0.97 vs all-MiniLM-L6-v2 0.62 on 10 hand-labelled sentences. That
  is a tiny check, so it is the reason for the default, not a validation.

### Per-person purchase prediction (talp73: Open e-commerce 1.0, 400 people × 20 categories × 3 seeds)
Sources: `talp73/kart/KART.md`, `talp73/kod/llm/SONUC-v1.5.md`, `SONUC-v1.7c.md`, `LLM-TABLO.md`,
`talp73/KARSIT-1.md`. 1a = person × category accuracy (threshold 0.5). 1b(c) = balanced-subset accuracy (the
primary measure).

| arm | 1a accuracy | note |
|---|---|---|
| always "no" | 0.764 | |
| survey-feature model (no LLM) | 0.771 | |
| purchase-history model | 0.807 | |
| L0 Haiku (no persona) | 0.614-0.647 | |
| **L1 Haiku** | 0.698 / 0.689 / 0.695 (seeds) | below always-no |
| L1 Sonnet | 0.749 / 0.744 / 0.752 | |
| L1 Haiku, 3 samples averaged | 0.704 | |
| **L2 Haiku** | 0.792 / 0.792 / 0.802 | +0.026..+0.039 over always-no; 1b(c) below history model by −0.022 / −0.020 / −0.026 |

- On 1b(c), L1 Haiku scores 0.583 / 0.543 / 0.564 against 0.588 for the survey model.
- L1 − L0 with Sonnet: person AUC +0.017 / +0.009 / +0.013. The 1b(c) difference, +0.008
  [−0.011; +0.026], is not significant. The persona's own answers add a little ranking signal and almost no
  accuracy.
- `batch_calibrate` (KARSIT-1 T1): after calibration the L1 − survey-model gap on 1b(c) includes zero
  (−0.013 / −0.008 / −0.002). The 1a headline moves from 69.4% to 76.5%, but the calibrated prediction is
  then close to always-no.
- SSR arm (`level="SSR"`, Haiku, all-MiniLM-L6-v2 embeddings, talp73 v1.7d, pre-registered; source
  `talp73/kod/llm/SONUC-v1.7d.md`, added by the integrator): calibrated SSR 1b(c) 0.573 / 0.551 / 0.547 and person AUC
  0.584 / 0.583 / 0.571. SSR-cal − L1-cal on 1b(c): −0.015 [−0.039; +0.007], −0.017 [−0.038; +0.006],
  −0.039 [−0.066; −0.015]; person AUC −0.034..−0.046, significant in all seeds. SSR-cal is below the survey model in
  every seed (1b(c) −0.028 / −0.025 / −0.041) and far below the history model (−0.102..−0.125). Free-text sentences
  carry some person-specific signal, but less than the numeric prompt. Cost: 1,043 calls, $0.678.

## Segment rate prior (`rate_prior.py`)

A per-segment event rate as a group feature, with no personas: one call per segment × sample, `{"pct": n}` read
with `simengine.llm.jsonx.json_extract` (a bare number or a `"12%"` string also works). A reply with no number or
outside 0-100 (0-1 with `parse_pct=False`) is a failed sample: it counts in `n_failed`, the mean is over the
rest, and one `RuntimeWarning` gives the count. Calls go through `ask_many` (one `llm.map` batch per sample, soft
refusals re-asked once, an errored call is a failed sample, `usage.error_calls`). When every call of a sample
fails with an error (bad key, backend down) the first error is raised, as in `survey`; pass `errors=[]` to keep
the failed samples and collect the errors instead. The JSON key sets the scale: `pct` / `percent` / `percentage`
are 0-100 and `rate` / `share` / `proportion` are 0-1 whichever form was asked (`{"rate": 0.12}` is 12 %).

```python
segs = {"new|fast": "new visitors whose first two page views are under 60 seconds apart", ...}
prior = sim.segment_rate_prior(llm, segs, "the session ends with a purchase", samples=5, blind=True,
                               context="An online shop logs product views, add-to-cart and purchase events.")
prior[["mean", "sd", "n_ok", "n_failed", "mean_blind"]]
M.GroupTransfer(group_features=prior[["mean"]]).fit(X, y, groups=seg)   # every column becomes a feature
prior.attrs["samples"]     # long rows: segment, arm, sample, rate, text
```

Measured: only once, by hand, in blind rehearsal D13 (6 segments × 5 samples, Haiku, session purchase rate).
The prior was about three times the observed rate and its AP was at the segment-rate baseline, so the arm was
dropped. Treat it as a feature to test against the blind arm and a shuffled placebo (`methods.llm_prior_feature`
does the shuffle), not as a calibrated rate. Do not name the dataset or its source in `context`.

## Direct effect forecasts (`effect.py`)

The model's own judgment of each condition's effect, with no simulated respondents (G1 rehearsal, the direct arm
L1; it was hand-written there as a `complete_json` loop). `segment_distribution` gives answer distributions;
this gives one number per treatment condition: treatment mean − control mean, in outcome units.

```python
from simengine.sim.effect import effect_forecast, effect_frame, memorisation_check

conds = {"Control": "No message.", "norms": "Most people in your country support ...", "fear": "..."}
res = effect_forecast(llm, conds, "support for climate policy (0-100)", scale=(0, 100),
                      framings=3, samples=2, named="<study or dataset name>", context="Online panel.")
res.table        # condition, hidden, sd_hidden, n_hidden (+ named, sd_named, n_named, named_minus_hidden)
res.probe        # named - hidden: mean, mean |diff|, largest gap, Spearman between the arms
res.refused, res.unparsed, res.failed, res.invalid, res.planned, res.calls
effect_frame(res, truth)            # EvalFrame: truth, pred_effect_hidden (+ pred_effect_named)
memorisation_check(res, truth)      # Spearman per arm vs measured effects, named - hidden, paired bootstrap CI
```

- One call per arm × framing × sample returns all conditions at once (`{"effects": {id: number}}`), so the model
  compares them side by side. `correlation_audit` asks per key, which did not fit this arm.
- Three built-in framings (`FRAMINGS`), or pass your own templates with `{study}` and `{context}` slots. Samples
  are passed as `sample=k` (k ≥ 1) for backends that key their cache on it; `sample_kw=None` puts
  "(Forecast k of n.)" in the prompt instead.
- The hidden arm is the forecast to use. The named arm only shows whether naming the study moves the numbers.
  `memorisation_check` warns when the named arm's Spearman is ≥ hidden + 0.2 (the G1 pre-registration rule) or the
  bootstrap CI of the gap lies above 0; under 4 conditions with truth it says `too_few`.
- A refusal is re-asked once with the research-forecast note; a reply with no JSON is re-asked once with "Return only
  valid JSON". What still fails is counted, never imputed. With numeric `scale=(lo, hi)`, an effect larger than
  hi − lo is invalid.
- Simulated respondents instead of a direct question: `simengine.research.hypotheses.forecast_experiment` (Hewitt
  et al. 2024 protocol: profiles × prompt bank per condition, effect = condition mean − control mean).

## Experiments and effect calibration (`experiment.py`)

A two-condition (or k-condition) between-subject experiment on a persona panel, and the step that makes its
effect sizes usable: how much larger they are than human effects, a correction, and how many human respondents
the corrected simulation is worth (G9 rehearsal; it was hand-written there as one `survey` per condition, Cohen d
by hand and a separate ratio script). `effect_forecast` asks the model for the effect directly; this measures it
from simulated respondents.

```python
from simengine import sim

conds = {"control": "No message.", "norms": "Most people in your country support ...", "fear": "..."}
res = sim.experiment(llm, personas, conds, "How much do you support the policy?", scale=(0, 100), seed=1)
res["conditions"]["norms"]       # mean, sd, n, assigned, refused, unparsed, no_persona {mean, sd, n}
res["effects"]["norms"]          # diff, g, d, se, ci, n_treat, n_control, no_persona_g, no_persona_diff
frame = sim.experiment_frame({"study1": res, "study2": res2},
                             human={"study1": {"norms": {"g": 0.21, "se": 0.06}, "fear": -0.05}})
fit = sim.exaggeration_ratio(frame.pred_g, frame.truth, family=frame.group, human_se=frame.truth_se)
corrected = sim.correct_effects(frame.pred_g, human_g=frame.truth, family=frame.group)   # LOO
sim.human_equivalent_n(corrected, frame.truth, frame.truth_se, family=frame.group)
section = sim.effect_section(frame.pred_g, frame.truth, frame.truth_se, family=frame.group)
# simengine.evidence.section_markdown(section) -> the card's "exaggeration / worth ~N humans" block
```

- Assignment: a seeded permutation dealt round-robin (`assign_conditions`), so arm sizes differ by at most one.
  Each persona's prompt shows only its own condition text (as its `survey` stimulus); `control=` defaults to
  the first condition. The question is numeric with `scale=(lo, hi)`, else a 1-5 Likert (SSR mean when
  `ask_kw={"ssr": ..., "free_text": True}`; `free_text` defaults to False here). `ask_kw` passes any other
  `survey` option (batch, parallel, tag, refusal_retry, ...).
- `samples=k` re-asks with the personas reshuffled, so batches (prompts, cache keys) differ, and averages each
  persona's answers. With `batch=1` the prompts repeat and the backend's cache returns the same answer. The
  averaging takes answer noise out of the sd, so g grows with `samples`; a human study asks each person once,
  so compare g with human g at `samples=1` (the default) or compare `diff`.
- The no_persona arm sees the same condition texts. Its g is what a generic respondent gives; a warning fires
  when it is at least the persona arm's.
- Refused and unparsed personas are counted per condition and left out of the means, never imputed.
- `exaggeration_ratio`: median \|sim\|/\|human\| (exact-zero human effects left out of the ratio) with a
  family-clustered percentile CI, `ratio_sum`, OLS slope sim ~ human (1 = right size) and the through-origin
  slope with clustered CIs, Pearson r, sign agreement, verdict `exaggerates` / `understates` / `about right` /
  `too few` (< 3 effects), and the LOO block: the leave-one-family-out ratio scale and MAE raw vs ratio- vs
  affine-corrected. With a single family the LOO leaves out single effects.
- The OLS slope is pulled toward 0 when the human effects carry sampling noise (errors in the regressor): in a
  simulation with se 0.1 and 20 families its CI held the true slope 0.68 of the time. Pass `human_se=` and read
  `slope_corrected` (cov / (var(human) − mean se²), with `reliability`; None below reliability 0.2), which
  covered 0.91-0.95. The ratio kept its coverage in the same runs; the affine correction (human ~ sim, noise in
  the outcome) is not attenuated.
- Every CI here is a family-clustered percentile interval pushed away from the estimate by
  t(G−1)/z · √(G/(G−1)) for G families. With 3-6 families the plain interval covered 0.72-0.90; widened, the
  ratio, slope and N CIs covered 0.89-0.97 at 3-30 families (simulated). With 3 families the factor is 2.7, so
  the CIs are wide: that is the honest width.
- `correct_effects(sim, human_g=..., family=...)` corrects every effect with a fit from the other families
  only, so an effect is never corrected with its own human value. `correct_effects(new_sim, fit)` applies a
  measured ratio (`method="ratio"`, sim / ratio) or affine fit (`method="affine"`, human ~ a + b sim) to new
  effects.
- `human_equivalent_n`: err = mean((sim − human)²) − mean(human_se²), then N = (4 + mean(human²)/2) / err, the
  total two-arm human sample with the same sampling variance (var d ≈ 4/N). It uses the widened cluster CI.
  When err ≤ 0 (the simulation is within the human noise) it returns `n: None` with a note, not infinity. An
  open upper CI end is `None`.
- `effect_section` puts the raw and LOO-corrected N next to a baseline that predicts every effect as the mean
  of the other human effects. When the simulation is not worth more humans than that baseline, a note says the
  N reflects small effects, not information from the model (the G2 lesson: show the multiplier next to the
  strongest baseline). A simulation within the human noise counts as better than any finite N; when the
  baseline itself is within the noise, the note says these effects cannot show added information. With
  `human_se` the section also shows the noise-corrected slope.
- Measured on FakeLLM only (tests: a panel whose effects are 2× the truth gives ratio ≈ 1.9, slope ≈ 2.0, and
  LOO MAE 0.39 → 0.08). No real-model number yet.

## Schema-constrained answers (`constrained.py`)

Ported from replicant's oTree bot (`src/replicant/otree/bot.py`, pyreplicant 0.3.0, MIT; credit to the replicant
project). The bot sat between the LLM and an oTree form: validate the answer, re-prompt with the errors, give up
after 8 tries. Here the same loop runs on any `LLM` and on any page text, without oTree.

```python
from simengine.sim import FieldSpec, ask_constrained, completion_summary

fields = [FieldSpec("contribution", "int", min=0, max=20, label="Tokens you put in the group account"),
          FieldSpec("vote", "choice", choices=[("1", "Yes"), ("0", "No")])]
res = ask_constrained(llm, persona_text, game_text, fields, max_tokens=1024,
                      meta={"condition": "random", "agent_id": "a1", "persona_method": "big5", "seed": 0})
res.completed, res.values, res.failures, res.attempts, res.usage.as_dict()
rows = [r.row() for r in results]            # meta + values + valid + attempts + max_tokens
completion_summary(results, by="condition", planned={"random": 50, "cooperative": 50})
```

- Persona text goes in as `system`, with replicant's "CRITICAL INSTRUCTION: Respond with ONLY a valid JSON
  object ..." line appended (`system_suffix=""` drops it). The page or game text goes in as `user`, inside
  replicant's prompt frame ("Page context:", "Your previous answers on this page:", "Field to answer:",
  "Respond: {...}").
- `validate_answers`: missing (`None`, `""`, `"nan"`, NaN) is an error. A choice matches its value, then its
  display text, then either case-insensitively; the clean answer is the value. Numbers must be a number or a
  numeric string; bounds are inclusive; `int` rejects fractions. Errors use replicant's wording ("contribution:
  25 is above maximum 20", "Missing required field: vote", "... is not a valid choice. Options: [...]").
- Under `sequential_threshold` (4) fields, one call asks for all fields; each failed check is retried with the
  errors, up to `max_retries` (8) calls. From 4 fields on, fields are asked one at a time (8 calls each) and every
  prompt carries the answers already given. As in replicant, the page stops at the first field that fails; the
  fields after it are logged with `attempt: 0` (never asked). `stop_on_failure=False` keeps going.
- `ConstrainedResult.values` holds validated fields only. A field that fails stays missing. It is not imputed,
  not defaulted and not read out of prose. Filter on `completed` / `valid` before analysis, and report the
  completion table next to every result.
- `completion_summary` columns: `n_planned` (pass `planned=`; the default is the number of results seen, which
  hides runs that crashed), `n_results`, `n_completed`, `completion_rate` (completed / planned), `attempts`,
  `failed_<field>` (failed after all retries) and `missing_<field>` (failed or never reached).

Why the completion table matters (re-counted by this module from replicant
`papers/ertan2009/results/otree_{default,random,cooperative}_raw.json`, the logged page answers and
`FIELD_FAILURE` entries; 50 bots per condition, stepfun/step-3.5-flash, a one-shot vignette version of Ertan, Page
& Putterman 2009, not the original repeated game):

| condition | Baseline page | Voting page | Regimes page | Punishment page |
|---|---|---|---|---|
| no persona (default) | 50/50 | 50/50 | 50/50 | 50/50 |
| random Big Five personas | 50/50 | 50/50 | 50/50 | 50/50 |
| hyper-cooperative personas | 50/50 | 50/50 | **30/50** | **7/50** |

- All 43 field failures in the cooperative run (20 on the regimes page, 23 on the punishment page) ended with an
  empty last response after 8 attempts. The cause (for example a reasoning budget used up before any answer
  text) was never confirmed by a rerun.
- The dropout is not random. The 20 bots that failed the regimes page had a mean baseline contribution of 17.6,
  against 15.8 for the 30 that finished it (re-counted from the same raw file). Dropping the failures silently
  would shift the sample.
- replicant's summary JSON for that run still says n=50. That is why `completion_summary` keeps `n_planned` and
  `n_completed` side by side.

Deviations from replicant, all deliberate:
- **Dropped `_try_direct_value`.** It salvaged a value from non-JSON text by substring and first-number matching,
  so it reads "yes" inside "no, yes ...". A non-JSON reply is now a failed attempt. The one-field single-key
  salvage (`{"answer": 4}` for field `contribution`) is kept, as in replicant.
- **No transport code.** OpenRouter, `requests` and the 429 sleep are not ported; `simengine.llm` owns retries,
  backoff and budgets. Backend errors (budget, auth) propagate. replicant turned every exception into an empty
  answer.
- **Feedback.** An unparseable reply gets replicant's "could not be parsed as JSON" message. In replicant this
  branch was unreachable because the missing-field errors came first. Every retry prompt ends with
  "(Attempt k of n.)", so a cached backend does not replay the same bad answer.
- **Field hints.** Float fields say "number from a to b". Fields without bounds say "(integer)" / "(number)".
  replicant only had integer hints. Batch prompts keep replicant's "Field to answer:" heading.
- **`max_tokens` default 1024.** replicant sent 16000 on every call. The cap is a treatment (in edsl-test's who-to-punish run the bare model
  punished the free-rider every period at max_tokens 8192 and never at the default cap; single run, and the
  default-cap run was partly None-corrupted; `experiments/07_who_to_punish/results/2026-06-11_15-08-37_baseline/ANALYSIS.md`), so it is stored on every result and row.
- **Partial values are kept** in `values` with `completed=False`. replicant discarded the whole page.

## Robustness (`robustness.py`)
Port of the G7 rehearsal's hand-written scoring (makro C1 protocol: retest, other framings, reversed options,
"axis W1 / signal" table). It answers one question before a subgroup comparison is reported: does the answer move
more when the prompt is repeated, reworded or reordered than the groups differ from the population?

```python
from simengine.sim.robustness import robustness, robustness_frame
rb = robustness(llm, segs, qs, axes=("retest", "framing", "order"), population="US adults")
rb["conditions"]   # axis, condition, w1 [lo, hi], ratio [ratio_lo, ratio_hi], shift (order), n_cells, invalid
rb["ratio"], rb["worst"], rb["verdict"]
kart_rows.append(rb["card"])                   # evidence.kart row, tur "robustness"
robustness_frame(rb)                           # segment x question x condition, with the cell's signal
```

- **Conditions.** `base` = the first framing (`FRAMINGS_HIDDEN["h1"]`, or `FRAMINGS_NAMED["v1"]` with `source=`).
  `retest:<i>` (`retests=1`) asks the base prompt again with `sample=<base + i>` passed through `ask_kw` to
  `llm.complete`: Claude keys its disk cache on `sample`, so a retest is a new draw, not the stored reply (the
  base run passes no `sample`, so it shares the plain call's cache entry). `framing:<v>` runs each other framing
  in `framings` on its own (no averaging). `order` reverses the options in the prompt and maps the reply vector
  back to the original order before scoring (cells too; `runs["order"]["reversed"]`).
- **Numbers.** Per segment × question × condition: `w1_ordinal(base, condition)`; for `order` also the signed
  shift E[pos | base] − E[pos | reversed] with positions scaled to [0, 1] (negative = pulled toward the option
  listed first). Signal per segment × question: W1(group, population). `signal="population"` (default) asks the
  population description once as its own segment; `"mean"` uses the mean of the segments' base vectors (no extra
  calls); a mapping `{"segments": {seg: {qid: vec}}, "population": {qid: vec}}` uses real data. `segments` is
  required there (W1 between a simulated group and the real population would count the simulator's own error as
  signal); without `population` the real groups are scored against their own mean (`by="given-mean"`, the
  rehearsal's round 3). The card note names the scale: real data or the simulator's own. Condition ratio = mean
  condition W1 / mean signal W1 over questions; questions are the bootstrap clusters (`n_boot=1000`, `seed=0`).
  The summary `ratio` is the worst condition. Above `threshold=0.5` the verdict says single-prompt subgroup
  comparisons are not reliable (the G7 reading), and it adds "borderline" when the worst ratio's CI contains the
  threshold. `nominal=True` uses TVD and drops the order shift. Invalid answers become uniform vectors
  (`fallback="uniform"`, as in the kit), which shows up as movement; the card note gives their count, and
  `fallback="skip"` leaves those cells out of the comparison instead.
- **Calls.** (1 + retests + extra framings + order) × segments × questions, plus one per question for
  `signal="population"`. With a backend that ignores `sample` (FakeLLM) the retest distance is 0 by construction.
- No measured number is quoted here: the module has only been run offline (FakeLLM) in this repository.

## Design notes and deviations from the research code
- **English field names.** The panel field names (`score`, `choice`, `reason`, `intent`, `remark`,
  `<f>_answer`) and runner prompt wording are English translations of the Turkish sahne prompts. The
  batching frame (`SYSTEM_ASK`, `ASK_TEMPLATE`) is the measured one. The distribution, persona-vector,
  Upworthy and talp73 prompts are verbatim. The survey, A/B, roundtable and strategy task texts were never
  measured in this English form.
- **SSR is off unless you pass it in.** Pass `ssr=SSR()` (or `True`) to turn it on, so a run never downloads
  a model by surprise.
- **SSR on long texts and offline.** `SSR(max_chars=2000)` (the default; `None` = no cut, the old behaviour;
  `embed(texts, max_chars=...)` overrides per call) cuts each text before encoding: bge-base reads at most 512
  tokens anyway, so the default loses little and gains little. Speed comes from a lower cut. One run (22 Oct,
  bge-base, CPU, 4 threads, nice 19 on a shared, loaded machine, 96 synthetic English texts of ~10k characters):
  no cut 2.3 texts/s, 2000 chars 3.0, 1000 chars 6.4, 500 chars 13.7 (rates depend on load). `cache_dir=`
  stores one `.npy` per text keyed by sha256(model name + text), written atomically, so reruns and other
  processes skip texts already embedded (with a lambda or nested `encoder=` it needs `cache_key=`, else
  ValueError). `progress=True` prints one stderr line per `progress_every` new texts with texts/s; a callable
  gets `(done, total, texts_per_s)`. `SIMENGINE_HF_CACHE` / `cache_folder` may be the HF_HOME folder or its
  `hub/`. A model missing from the local cache raises an `OSError` naming `SIMENGINE_HF_CACHE`, `cache_folder`,
  `HF_HOME` and `offline=False`.
- **Retries.** `predict_people` re-asks with a "(Attempt n: reply with the JSON object only.)" suffix, so a
  cached backend does not return the same bad answer. talp73 retried inside its backend.
- **Generalised prompts.** `persona_distribution` takes `setting=` in place of the fixed "public opinion survey
  of US adults in October 2020" line. The distribution framings take `population/source/when`. With
  `population="US adults"`, `source="the Cooperative Election Study (CES)"` and `when="October 2020"` they are
  identical to `k2.py`.
- **Not ported here.** Persona builders (`sahne/sentez.py`) belong to `simengine.persona`. Evidence and cards
  (`sahne/pva.py`) belong to `simengine.evidence`. The A/B TF-IDF stack belongs to `simengine.methods`. The
  sealed-predictions helper belongs to `simengine.research`.
- **Outputs are JSON-safe.** Integer keys become strings: `roundtable(...)["answers"]["0"]`.

## Tests
`tests/test_sim_parse.py`, `test_sim_panel.py`, `test_sim_runners.py`, `test_sim_distribution.py`,
`test_sim_prediction.py`, `test_sim_evaluate.py`, `test_sim_constrained.py`, `test_sim_survey_gaps.py` (refusals,
numeric / free_text, per-person stimulus, ledger tags), `test_sim_review_fixes.py` (stimulus not in the
profile, empty / failed batches, shared handling in the other runners, exponent parsing, free-text line),
`test_sim_survey_parallel.py` (one plan per arm, the same results as the serial loop, `parallel=`,
`refused_ids` / `unparsed_ids`, `survey(tag=)` down to the TSV column), `test_sim_effect.py` (effect forecast arms, sample keys, re-asks, memorisation check), `test_sim_rate_prior.py` (rate table mean / sd / counts, sample keys, failed and errored answers, blind arm, GroupTransfer feature, NumericField export, `ab_many` reader_chars warning and byte-identical default prompt), `test_sim_robustness.py` (order mapping, retest `sample` keys, framing axis, real-data signal, ratio, card row, invalid answers), `test_sim_experiment.py` (Hedges / paired g, balanced assignment, between / within runs, the 2× panel: ratio recovered, LOO correction lowers MAE, LOO never uses the own value, human-equivalent N and its noise floor, card section, few-family CI widening, noise-corrected slope, within-noise vs baseline note) (offline, FakeLLM, a few seconds):

```
nice -n 19 python -m unittest discover -s tests -t . -p 'test_sim*.py'
```
