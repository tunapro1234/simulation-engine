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
| `ab_test`, `ab_frame` | `ab.py` | PAARS-style A/B test on a persona panel. Each persona gives a 1-10 score per variant, a choice and a reason. Half of the panel sees the variants reversed, which gives the position effect. Output: winner, preference share, bootstrap win probability, sign test, control arms, `confidence` (`confidence_check`), and a direction check when `truth` is given. |
| `paars_decision` | `ab.py` | Forced-choice decision from per-persona A/B scores, with a confidence threshold (0.65). |
| `ab_many` | `ab.py` | Many pairwise tests per call, with all readers in one prompt. The prompt is verbatim from the Upworthy rehearsal. Returns `p_a` with Laplace smoothing. |
| `roundtable`, `roundtable_frame` | `roundtable.py` | Multi-round discussion. Round 0 is independent. In later rounds each persona sees its own previous answer and a summary of its segment (or of the whole table). `CONFIRMATION_BIAS` and `KEEP_VIEW` are the measured stance instructions. |
| `roundtable_strategy` (alias `roundtable_conditions`), `roundtable_strategy_frame`, `hide_persona_fields` | `roundtable.py` | Strategy comparison at an interacting table: one `roundtable` per condition on the same panel (`{id: text}` or a list; `hide_fields=("condition",)` keeps the RCT arm out of the prompt). Output: condition × round × segment means (`table`), paired effect vs the `control` table per round (`effects`), last round − round 0 per persona with bootstrap CI and the change of the effect (`shift`), and `ranking` in round 0 and the last round. The frame helper reuses `roundtable_frame` (unit `<condition>|<unit>`); `effect=True` gives condition − control predictions to score against observed effects. CIs resample personas and ignore the within-table dependence of later rounds. |
| `strategy`, `strategy_frame` | `strategy.py` | Two or more scenarios × segments. Each persona gives a 0-100 intent and a reason per scenario. Output: ranking, win probability, sign test on the top two, and the segment whose profile diverges most. |
| `segment_distribution`, `distribution_prompt`, `FRAMINGS_NAMED`, `FRAMINGS_HIDDEN` | `distribution.py` | No personas. Makes one call per segment × question × framing ("estimate the percentage of these respondents who would choose each option"), then averages the 3 framings. An invalid reply becomes a uniform vector (or is skipped with `fallback="skip"`); the filled share is recorded per cell. `repeats=k` passes `sample=0..k-1` (independent draws, per-sample dists kept). See "Distribution runners" below. |
| `effect_forecast`, `effect_frame`, `memorisation_check`, `EffectForecast` | `effect.py` | No personas. Asks for every condition's effect against control in one JSON object per arm × framing × sample (independent draws via `sample=k`), averages per condition, and returns a condition × arm table: hidden arm always, named arm (study named) with `named=`, plus named − hidden as a memorisation probe and refused / unparsed / failed / invalid counts. Import from `simengine.sim.effect`. See "Direct effect forecasts" below. |
| `persona_distribution` | `distribution.py` | Per-persona probability vectors, many respondents per call, with tables filled within each segment. Also returns the segment means, one set per `segments=` scheme, and `survey`-style refused / unparsed accounting with ids. `person_distribution_frame` gives persona × question rows. |
| `distribution_frame`, `w1_ordinal` | `distribution.py` | Distribution EvalFrame (list-valued `truth` / `pred_llm_dist`, numeric `filled_llm_dist`; `scheme=` picks a `persona_distribution` segment scheme), and the normalised ordinal W1 used for the makro numbers. |
| `predict_people`, `prediction_frame` | `prediction.py` | Per-person prediction. L0 is a no-persona population prior over `l0_samples` samples. L1 uses the person's own survey answers. L2 is L1 plus the person's past behaviour in the same items. SSR asks for one first-person sentence per item. It takes independent `samples` (passed as `sample=j`) and re-asks with "JSON only" when parsing fails. |
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
  `kind="probability"` is unchanged.

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
  tokens anyway. `cache_dir=` stores one `.npy` per text keyed by sha256(model name + text), written atomically,
  so reruns and other processes skip texts already embedded. `progress=True` prints one stderr line per
  `progress_every` new texts with texts/s; a callable gets `(done, total, texts_per_s)`. No CPU throughput figure
  is given here because none was measured with this code. A model missing from the local cache raises an
  `OSError` naming `SIMENGINE_HF_CACHE`, `cache_folder`, `HF_HOME` and `offline=False`.
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
`refused_ids` / `unparsed_ids`, `survey(tag=)` down to the TSV column), `test_sim_effect.py` (effect forecast arms, sample keys, re-asks, memorisation check) (123 tests, offline, FakeLLM, under 1 s):

```
nice -n 19 python -m unittest discover -s tests -t . -p 'test_sim*.py'
```
