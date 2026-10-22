# `simengine.persona`

Personas in the TALP JSON shape, persona sources (LLM, fake, TALP API slot), and builders that make
personas from survey rows and purchase/answer history.

Ported from the TALP prep branches (read-only research code):

| library file | ported from |
|---|---|
| `schema.py` | `uretim/sahne/kod/sahne/sema.py` |
| `sources.py` | `uretim/sahne/kod/sahne/kaynak.py` |
| `builders.py` | `uretim/sahne/kod/sahne/sentez.py`, `uretim/persona/kod/kit/veri.py` (`talp_persona_oku`) |
| `history.py` | `uretim/talp73/kod/llm/istem.py` (`profil`), `uretim/talp73/kod/llm/kos.py` (`gecmis_metinleri`), `uretim/talp73/kod/panel/panel.py` (purchase summary), `uretim/persona/kod/kit` (twin persona text = known answers) |

## Components

### Schema (`schema.py`)
- `AudienceSpec` / `Audience`: the input brief (TALP `persona-sets`). Enum fields are `income_tier`,
  `household_composition`, `usage_intensity`, `commitment_disposition`, `brand_awareness`, `age_range` and
  `gender_mix`, plus `identity_text`, `behavioral_anchor`, `locale`, `vertical` and `target_size`.
  `spec.validate()` returns the TALP UI warnings: a value outside the enum, a `target_size` outside 10-200, a
  `persona_mode` that is not b2c/b2b, more than 12 sketch sentences, absolute words ("always", "never",
  "her zaman", ...), and a missing `identity_text` or `behavioral_anchor`.
- `Persona`: `identity`, `traits`, `authored` (persona, consumer_profile, history_data, behavior_voice,
  buying_context, triggers), `behavioral`, `quantitative_dispositions` and `evidence`
  (`measured_behavior.facets/claims`, `measured_receipt`). It also has a `segment` field, which is ours.
  The field names are TALP's. Unknown keys are kept in `extra` at every level and written back by `to_dict()`,
  so a TALP JSON survives a round trip.
- `p.text(lang, clip=600)` is the persona block used in prompts. `p.card()` is a UI card.
  `load_personas(...)` accepts a JSON string, a list, `{"personas": [...]}` or the flat form.
  `dump_personas(...)` writes `{"personas": [...]}`.

### Sources (`sources.py`)
- `PersonaSource` (ABC) provides `generate(spec, n, batch=8, seed=0)`, which turns a brief into n personas
  in batched calls, and `ask(personas, task, fields, batch=10, context=None, lang="en")`, which returns
  `{persona_id: {field: value}}`. Both use the research prompts verbatim (`SYSTEM_GENERATE`,
  `GENERATE_PROMPT`, `SYSTEM_ASK`, `ASK_PROMPT`). The shared layer adds a new-call budget (`CallCounter`),
  an optional JSONL log, parallel calls (at most 4, `SIMENGINE_PARALLEL`), an `on_event` callback and
  `check_errors()`. `check_errors()` raises `LLMError` if any call failed, so a run stops instead of writing
  partial predictions.
- `cell_plan(spec, n, seed)` is the quota plan. Each persona gets its target enum cell from round-robin
  assignment, with one seeded shuffle per axis. The LLM is not simply told to "be diverse". Age defaults to
  all six TALP bands and gender to female/male. Out-of-enum values are dropped from the quota.
- `ClaudeSource(llm)` (alias `LLMSource`) works over any `simengine.types.LLM`: `simengine.llm.Claude`,
  `ClaudeCode` or `FakeLLM`. Batches go through `llm.map`, and the backend's cache, backoff and cost ledger
  apply. Replies served from cache are refunded to the budget. Requests are sent in chunks that fit the
  remaining budget. An error whose text looks like auth, limit or budget (401, "unauthorized", "limit",
  "credit", "budget") stops the rest of the run.
- `FakeSource` is offline and deterministic (hash based). It understands field specs such as `int 1-10`,
  `float 0.0-1.0`, `one of A|B` and `list`, and its text is marked "(fake)". Use it for dry runs and tests.
  **Its numbers are not measurements.**
- `TalpSource` is an empty slot for the TALP API. It reads `TALP_API_URL` / `TALP_API_KEY` from the
  environment and always raises `NotImplementedError`. **It sends no request.** Fill it in only if the
  organizers publish the API.
- `build_persona(pid, cell, raw, spec)`: a missing LLM object gives a stub persona marked
  `generation_error: true`. It never gives an invented one.

### Builders from tables (`builders.py`)
- `from_table(df, mapping)` or `from_table(df, id_col=..., fields=..., facets=..., forbidden=[...])`.
  `Mapping.fields` maps a TALP field to a column, `(column, dict|fn)` or `fn(row)`. Values outside the enum are
  dropped. `Mapping.facets` copies measured or stated values **unchanged** into
  `evidence.measured_behavior.facets` and records a claim (`contract_id`, column). `Mapping.text` adds
  template narrative without an LLM. Pass `n`/`seed` to sample, and `stratify=` for a proportional stratified
  sample.
- **Leakage guard:** a facet whose source column is in `forbidden` raises `ValueError`.
- With only `id_col`, `auto_mapping` takes over. Columns named like TALP fields become identity and traits:
  `age` (numeric) becomes an age band, and `gender`/`sex` is normalized. **Every other column becomes a facet,
  so always put the target in `forbidden`.**
- **Ids are checked first.** If the id column has empty/NaN values or duplicates, `from_table` (and
  `from_history` with a `table`) raises `ValueError` with the counts and a fix hint, before building anything.
  An empty id never becomes the string `"None"` or `"nan"`. `check_ids(df, id_col)` runs the same check on its
  own. With `id_col=None` (or `Mapping(id_col=None)`), the rows are numbered instead: `"row0"`, `"row1"`, ...,
  where the number is the row's position in `df` before sampling, so a sampled panel keeps stable ids.

  ```python
  sp.from_table(df, id_col="ResponseId")   # ValueError: id column 'ResponseId' (602 rows): 512 empty/NaN ids. ...
  panel = sp.from_table(df, id_col=None, forbidden=["target"])   # ids row0, row1, ...
  ```
- **Codebook value labels.** `from_table(df, ..., value_labels={"Gender": {1: "male", 2: "female"}})` (or
  `Mapping.value_labels`; the keyword is merged over the Mapping's, per column) replaces codes with labels
  before the enum mapping and the text, so a persona reads `female` and not `Gender: 2`. Codes compare as
  numbers or strings: `2`, `2.0`, `"2"` and `"02"` are one code, and two labels for one code raise
  `ValueError`. Empty cells stay empty. The id column is never relabelled. A code with no label is kept as is,
  with one `UserWarning` per column that gives the counts per code. `from_row` applies the same labels to one
  row. `apply_value_labels(df, labels)` relabels a whole table. `read_codebook(table)` builds the dict from a
  `(variable, code, label)` table, where a blank variable cell repeats the one above, or from a `(code, label)`
  table with `variable=`. The table can be a DataFrame or a .csv/.tsv path, and `columns=` picks the three or
  two columns out of a wider table.

  ```python
  labels = sp.read_codebook(codebook)          # variable, code, label
  panel = sp.from_table(df, id_col="rid", value_labels=labels, forbidden=["target"])
  ```
- **Persona pool selection.** `pool_select(df, K, method=, features=, strata=, seed=0)` returns the index labels
  of K rows in table order, so `df.loc[idx]` goes straight into `from_table`. The methods are:
  - `"random"`: uniform. For one seed, a smaller pool is a subset of a larger one.
  - `"stratified"`: proportional to the `strata` column(s) by largest remainder, so a stratum smaller than n/K
    can get 0 rows, with random rows inside each stratum.
  - `"kmeans"`: K clusters on z-scored `features`, with k-means++ starts and the best of `n_init=4`. Text
    columns are one-hot and NaN is the mean. It picks the row nearest each centroid, and every row picked is
    distinct. `features` is required, so keep ids and targets out of it. Each member stands for a cluster of a
    different size, so an unweighted k-means pool over-represents sparse regions (the tails) and is not a sample
    of the table: use it to cover the range for narration, and take distributions from the table.

  K >= len(df) returns every row. `pool_curve(df, sizes, score_fn, method="random" | [...], draws=20)` scores
  `draws` pools per size with `score_fn(pool_frame) -> float`. It returns `method, size, draws, mean, lo, hi`,
  where lo and hi are the 5-95% interval of the draws, and `long=True` returns one row per draw. A size of at
  least len(df) is the whole table, scored once. The score is the caller's, e.g. 1 - TVD of the pool members'
  own answers against held-out cells. In the G5 rehearsal, k-means pools scored 3-5 points below random ones,
  so put random in every curve as the reference.

  ```python
  idx = sp.pool_select(train, 20, method="kmeans", features=["age", "income", "visits"], seed=0)
  curve = sp.pool_curve(train, [10, 20, 50, 100], score, method=["random", "stratified", "kmeans"],
                        strata="region", features=["age", "income", "visits"])
  ```
- `narrate(source, personas, context=...)`: an LLM writes the name, tagline, backstory and voice, and leaves
  the facets alone. Without `context`, the LLM may invent the sector.
- `to_frames(personas)` returns `(people, features)`. `people` has `id`, `seg_*`, `segment` and
  `persona_text`. `features` is a long frame of `quantitative_dispositions` (`id, item, value`).
- `age_band(age)` maps a numeric age to a TALP AgeBand.

### Builders from history (`history.py`)
This module follows the talp73 levels:

| level | what the persona text holds | field |
|---|---|---|
| `L0` | one neutral description for everyone (control) | `authored.persona` |
| `L1` | the person's own survey answers, as "- question: answer" lines, plus optional twin-style known answers (`answers`, `item_text`, `targets` dropped) | `authored.consumer_profile` |
| `L2` | L1 plus a per-category purchase summary, using only purchases **strictly before `cutoff`** | `authored.history_data` |

- `summarize_purchases(purchases, id_col, category_col, date_col, cutoff, qty_col=None, top_k=20)` returns one
  row per (id, category) with `items`, `quantity`, `days`, `last_purchase` and `items_recent` (default
  window: cutoff minus 1 year). The default categories are the top 20 by distinct buyers before the cutoff.
  This is the panel V4 rule.
- `purchase_history_text(...)` writes lines such as
  `BOOK: 2 items on 1 different days, last purchase 2022-01-03, 2 items in July 2021 - June 2022`, or
  `CUP: never purchased`.
- **Guards:** `questions` that contain a `targets` column raise. Answers on `targets` items are dropped. L2
  with raw purchases and no `cutoff` raises.

## When it works and when it does not (measured)
All numbers below come from the research runs, which used the original `sahne` and `talp73` code with
Claude Haiku/Sonnet 5.5. These are the prompts and builders this module ports. They have **not** been
re-measured through this library.

**Persona panels on real people carry signal, but simple baselines that use the same information match or
beat them.**
- Open e-commerce, monthly order days in 2022, n = 198, Pearson [95% bootstrap CI]: the demographic baseline
  scores 0.28. A Haiku persona panel built by `from_table` (demographics + survey statements) scores
  **0.618 [0.513; 0.696]**. Adding the 2021 measured behavior as a facet anchor gives
  **0.860 [0.807; 0.899]**. The label-free persistence baseline (last year's value) sits at the same place:
  persona + anchor minus persistence is +0.012 [+0.000; +0.022]. In a second sample (seed 1), the persona
  panel scored 0.540 against 0.521 for the raw statement, and persona + anchor minus persistence was
  −0.018 [−0.051; +0.019]. The persona-swap control scored −0.03 and 0.09, so the persona signal is the
  person's own. With an anchor present, the persona text adds no measurable extra signal: anchored no-persona
  scored 0.849 and 0.894. Source: `uretim/sahne/RAPOR.md` (Prova i and its repeat).
- talp73 is a re-creation of TALP's "73%" claim: 400 personas × 20 Amazon categories × 3 seeds, Open
  e-commerce, test 2022-H2. L1 is the survey profile (`from_history(level="L1")`). L1-haiku55 accuracy at
  0.5 was **69.4% [67.7; 71.2]**, against 76.4% for the always-no baseline and 77.1% for the survey model.
  On the balanced subset (within-person top half) it was **56.4% [52.5; 60.2]**, against 50% chance and
  58.8% for the survey model. Its hit@3 was 64.0%, against 71.0% for popularity. Source:
  `uretim/talp73/kart/KART.md`.
- In the same talp73 run, persona text helps the **same** model a little. L1 − L0 for Haiku, within-person
  AUC: +1.3 / +1.6 / +1.4 points, significant in all three seeds. Accuracy at 0.5: +6.1 points [4.3; 9.0].
  Source: `uretim/talp73/kart/KART.md` ("Persona katkısı").
- L2, the profile plus purchase history before the cutoff: L2-haiku55 accuracy at 0.5 was **79.6%
  [77.8; 81.6]**, which beats always-no. The LLM-free history model scored 80.7%, and on the balanced subset
  L2 minus the history model was −2.0 to −2.6 points in all three seeds. History is the signal; the LLM does
  not beat a classic model on it. Source: `uretim/talp73/kart/KART.md` and `uretim/talp73/PROTOKOL.md`
  v1.5/v1.6.1.
- Twin-2K-500 twin persona text (known answers): persona text compared with a demographics-only LLM, across
  people, corr 0.125 → 0.175. A fair LLM-free model still leads, at 0.342 against 0.175 for the raw twin
  (122 columns). The LLM adds only +0.006 [+0.004, +0.007] on top of it. Source:
  `uretim/persona/RAPOR.md` (KARŞIT-2) and `uretim/persona/RAPOR-G2-YENI.md`.

**Generated panels (`generate`) do not predict A/B winners by themselves.** On Upworthy (300 real A/B tests),
an 8-persona Haiku panel picked the winner **0.522 [0.468; 0.575]** of the time (chance 0.50). That rate was
0.50 on the half where the panel was most unanimous. The experiment kit's stack scored 0.81 [0.76; 0.85].
Source: `uretim/sahne/RAPOR.md` (Prova ii).

**Speed and language.** Going from a raw CSV to personas, a Haiku simulation and an evidence table took
3 min 52 s for Open e-commerce (63 calls, $0.105) and 7 min 2 s for Upworthy (121 calls, $0.20). All steps
other than the LLM calls took under 10 s. A Turkish brief produced 12 Istanbul personas in 41 s (2 calls).
`validate()` caught an out-of-enum value, and the quota came out 4/4/4. Source: `uretim/sahne/RAPOR.md`.

**Practical rules from these runs**
- Always report the persona result next to an LLM-free baseline that uses the same information (survey model,
  persistence or history model), and next to the no-persona and persona-swap controls. Those controls belong
  to `simengine.sim`.
- Accuracy alone flatters constant predictors (talp73 always-no scored 76.4%). Use balanced or ranking
  metrics as well.
- Facets must hold only what was known before the prediction time. Use `forbidden` and `targets`, and a
  `cutoff` for history.
- Twin-2K-500 is public, so Claude 5.5 models may have seen it. Person-level memorization was not tested for
  Claude (`uretim/persona/RAPOR.md`, item 8).

## Deviations from the research code
- `ek` is renamed `extra`, and nested `extra` dicts are flattened at every level. The original flattened only
  the top level. `to_dict()` also drops sub-objects that end up empty.
- `Persona.text()` additionally includes `consumer_profile`, and it never clips the data-derived fields
  (`consumer_profile`, `history_data`). Narrative fields are still clipped at 600 characters.
- There is no driver-level disk cache. Caching, the cost ledger and 429 backoff belong to the LLM backend
  (`simengine.llm`). `ClaudeSource` defaults to `retries=1` and stops the run on auth or limit errors.
- The local Qwen driver (`QwenKaynak`) is not ported.
- The `card()` keys are English (`id, name, age, gender, segment, tagline, traits, story, voice, evidence`).

## Minimal usage
```python
import pandas as pd
import simengine.persona as sp
from simengine.llm import FakeLLM          # swap for simengine.llm.Claude(...) for real runs

survey = pd.DataFrame({"uid": ["u1", "u2"], "age": [23, 61], "gender": ["F", "M"],
                       "income_tier": ["middle", "low"], "orders_2021": [14, 3], "orders_2022": [12, 1]})
panel = sp.from_table(survey, id_col="uid", forbidden=["orders_2022"])   # facets: orders_2021 only

orders = pd.DataFrame({"uid": ["u1", "u1", "u2"], "category": ["BOOK", "PEN", "BOOK"],
                       "date": ["2022-03-01", "2022-06-01", "2022-08-01"]})
l2 = sp.from_history(survey.drop(columns="orders_2022"), id_col="uid", level="L2",
                     purchases=orders, cutoff="2022-07-01")
print(l2[0].text())

src = sp.FakeSource()                       # or sp.ClaudeSource(llm, budget=200)
spec = sp.AudienceSpec(target_description="Urban renters", audience=sp.Audience(
    income_tier=["low", "middle"], identity_text="renters in big cities", behavioral_anchor="weekly grocery app"))
assert spec.validate() == []
gen = src.generate(spec, n=10)
answers = src.ask(panel + gen, "Would you try a 15-minute delivery plan?", {"intent": "int 1-10"})
src.check_errors()
```

## Persona method zoo (`methods/`) and Big Five (`bigfive.py`, `validate.py`)

Ported from two read-only source repos: edsl-test (`personas/`, `benchmark/beebench/b01/agents.py`) and
replicant (pyreplicant 0.3.0, MIT: `src/replicant/personalities/factory.py`, `validation.py`). Credit for the
replicant code goes to that package (MIT, declared in its `pyproject.toml`; it has no root LICENSE file).
edsl-test has no LICENSE file; it is from the same owner.

### Persona methods
Every method returns n persona texts for a seed. The text goes into the LLM call as `system`, and the game or
question goes in as `user`. `""` means no persona.

```python
import simengine.persona as sp
sp.list_methods()                                   # name, needs_data, description, paper, source, data_file
texts = sp.sample_texts("homo_silicus", n=4, seed=1)
people = sp.method_personas("gps", n=20, seed=0)    # [{"id": "gps-0-0", "text", "persona_method", "params"}]
bank = sp.AgentBank("value_anchor", seed_base=1000, path="bank_value_anchor.json")
agents = bank.get(10)                               # {agent_id: text}; growing to 20 keeps these 10 unchanged
```

| method | what the persona is | paper | data |
|---|---|---|---|
| `baseline` | no persona (the bare-model control) | — | — |
| `homo_silicus` | GSA "human being with cognitive biases" prefix + one social-preference one-liner; mixture over 6 types with fixed largest-remainder quotas | Horton 2023 (arXiv:2301.07543); Charness & Rabin 2002; Manning et al. 2025 | — |
| `big5` | Big5-Scaler Simple Prompt: 0-10 dials with high- and low-pole glosses | Cho & Cheong 2025 (arXiv:2508.06149); norms Soto & John 2017 | — |
| `value_anchor` | "Answer as a person that is [value]", one of 19 BWVr values, drawn by most-important-value prevalence | Rozen et al. 2024 (arXiv:2407.12878); Lee et al. 2019 | — |
| `gps` | answers to the 8 qualitative Global Preferences Survey items, 6 preferences sampled jointly | Falk et al. 2018 (QJE); NBER WP 23943 | — |
| `construction` | fixed template with 3 Charness-Rabin dials (efficiency, self-interest, inequity aversion), balanced over the paper's 3 optimal vectors | Manning & Horton 2025, General Social Agents (arXiv:2508.17407) | — |
| `bfi2` | BFI-2-Expanded sentence bank, 15 sentences (`bigfive.py`) | Soto & John 2017; Huang et al. 2024 | — |
| `demographic` | first-person Argyle-style backstory from one survey-weighted GSS 2024 row | Argyle et al. 2023 | `gss_sample.csv` |
| `anthology` | one released Anthology backstory, age-stratified to GSS 2024 shares | Moon et al. 2024 (arXiv:2407.06576) | `anthology_backstories.json` |
| `twin2k` | verbatim Twin-2K-500 persona summary of one real respondent (prefix-stable draw) | Toubia et al. 2025 (arXiv:2505.17479) | `twin2k_personas_sample.json` |

**Data-backed methods.** The data files are not shipped. A loader looks in `data_root`, or in
`$SIMENGINE_DATA` when `data_root` is not given, at `<root>/persona_methods/<file>`. `data_root` may also be
the `persona_methods` folder itself. A missing file raises `PersonaDataNotFound` (a `FileNotFoundError`) whose
message gives the layout.
- `gss_sample.csv`: columns `id, age, sex, race, hispanic, born, degree, marital, childs, wrkstat, income16,
  class, relig, attend, partyid, polviews, region, weight`, with GSS 2024 lower-case category labels. An empty
  cell drops that fragment.
- `anthology_backstories.json`: a list of `{bracket, question, text}` (plus optional `sid`/`age`), where
  bracket is one of `18-29`, `30-49`, `50-64`, `65+`.
- `twin2k_personas_sample.json`: `{"personas": [{"pid", "summary"}]}`.

The source repo's `prepare.py` scripts build these from the GSS 2024 public-use file, HF
`SuhongMoon/anthology_backstory` and HF `LLM-Digital-Twin/Twin-2K-500`.

**Inlined parameters.** These are published-literature or calibration constants, not data. Each carries its
citation in the code.
- homo_silicus `CALIBRATED_WEIGHTS`: self_interested .36, efficient .50, selfish_but_fair .14, from
  `personas/homo_silicus/weights.json` (gpt-oss-120b calibration, train SSE .089, held-out dictator RMSE .315).
  The reciprocal type gets weight 0, so this mixture can never contain a punisher. Horton's GPT-4o mixture
  is available as `HORTON_WEIGHTS`.
- value_anchor `PREVALENCE_WEIGHTS`: 19 values from `personas/value_anchor/weights.json` (Lee et al. 2019,
  Table 2, Study 2, N = 439).
- big5 norms and correlations (Soto & John 2017, Tables 5 and 2); GPS correlations (NBER WP 23943, Table 12);
  the construction dial vectors (GSA p.50); anthology GSS 2024 age shares.

**Generic helpers.** These are exposed for new methods:
- `TraitDialSampler(norms, corr, template, clip, transform)`: pure-Python Cholesky, from big5 and gps.
- `quota_mixture(types, weights, n, seed)`: largest-remainder quotas, from homo_silicus.
- `weighted_rows_no_replacement(rows, weight_col, n, seed)` and `RowFragmentRenderer(templates, frame)`, from
  demographic.
- `stratified_pool_sample(pool, strata_weights, n, seed)`, from anthology.
- `register_method(...)`, for example `homo_silicus.make(weights=...)`, `value_anchor.make("uniform")` or
  `construction.make(dials=...)`.

**Equivalence with the source (checked while porting).** `AgentBank(m).get(10)` reproduced the edsl-test b01
bank files `benchmark/beebench/b01/agents_bank/<m>.json` byte for byte for all 9 methods. The three
data-backed methods were checked against the source data files. `sample_bigfive(20, seed=42)` reproduced the
assigned scores of the replicant validation run exactly (`papers/ertan2009/results/validation_continuous.json`).
Both checks read stored outputs. Neither re-runs the source code.

**Per-method replication status.** Source: `personas/*/replication.json`, all marked PASS. Most are
implementation-fidelity checks on small n, each with its own model.

| method | model | check | result |
|---|---|---|---|
| homo_silicus | gpt-4o | Horton Fig. 3 P(Left) pattern on 6 Charness-Rabin menus, MAD ≤ .2 | MAD .00 / .017 / .10 (efficient / inequity_averse / self_interested), 180 calls |
| big5 | phi-4 | single-trait dial 10/10 vs neutral on BFI-2 items | 5/5 traits higher when dialed (paper allows a neuroticism miss), 1,200 calls |
| value_anchor | Llama-3.1-8B, T 0.7 | anchored value rated above its circle-opposites (≥3 of 4 anchors) | 4/4 anchors, mean gap 3.22 |
| gps | Llama-3.1-8B | Table 12 correlation recovery offline (200k draws) + readback probe | worst correlation gap .0026; readback hit share 1.0 |
| construction | gpt-4o | MAE vs humans on 6 CR menus, paper φ* | **.347 vs paper .20** (baseline .398): better than bare, worse than the paper |
| demographic | Llama-3.1-70B | 24 strong-partisan backstories, 2020 vote | 24/24 own-party aligned |
| anthology | Llama-3.1-70B | age-bracket readback of 24 stories | 23/24 (.958) |
| twin2k | gpt-4.1-mini | holdout accuracy, 5 people × 8 questions | .7075 vs paper .6802 (test-retest .7475) |

**Who-to-punish human-likeness.** Source: edsl-test `benchmark/REPORT.md`, with gpt-oss-120b, 9 criteria
against Ertan, Page & Putterman (2009) targets, mean of 3 seeds (value_anchor: 4 runs, seed 1 ran twice), one group of 4 per seed. The mean index was:
construction .813, demographic .797, value_anchor .786, twin2k .783, baseline .694, big5 .688, gps .688,
homo_silicus .677, anthology .617.

Read these numbers with care:
- The ranks inside the top four are not significant.
- Seed ranges reach 0.37 within one method.
- The criterion weights were not sensitivity-checked.
- baseline and homo_silicus have an effective n of 1 (cache).
- The bare model dismantled the punish-low institution in all 3 seeds.
- "Never allow punishing high contributors" held in every run: 56/56 group votes over the 28 runs in
  `wtp_scores.csv` (REPORT.md says 54/54, which does not match the csv).

Use these as priors on which family to try. They do not show that any method works for your task: always
run the `baseline` control next to it.

### Big Five personas
```python
from simengine.llm import FakeLLM
import simengine.persona as sp
sp.bfi2_text({"agreeableness": 4.5, "neuroticism": 1.5})       # other domains = 3.0
people = sp.sample_bigfive(50, seed=1, overrides={"agreeableness": 2.0}, sd_scale=1.0)
people = sp.sample_bigfive(50, seed=1, norms=sp.SOTO_JOHN_2017_NORMS, corr=sp.SOTO_JOHN_2017_CORR)
panel = sp.bigfive_audience(spec, n=40, seed=0, strength=0.5)   # AudienceSpec quota cells + Big Five
res = sp.validate_bigfive(llm, n=20, seed=42)                   # builder="bfi2" | "big5_scaler" | callable
sp.to_card_row(res)                                             # flat row for an evidence table
```
**Scoring.** A 1-5 score maps to the sentence-bank weight (score − 3) × 1.5, on −3..+3 with 7 levels of 3
sentences each. A fractional weight takes `round(frac × 3)` sentences from the upper level. `PROFILES` holds
replicant's uncalibrated named profiles (cooperative, selfish, leader, anxious, average).

**Norms.** `US_ADULT_NORMS` are replicant's POPULATION_NORMS. replicant credits them to Soto & John (2017), but
they are not that paper's Table 5 (`SOTO_JOHN_2017_NORMS`), so their primary source is unverified.

**Persona-strength knob.** In `bigfive_audience`, mean = 3 + strength × (target − 3) and the SD is the norm
SD × strength:
- strength 0 gives everyone the all-average text.
- strength 1 keeps the targets or norms.
- strength above 1 exaggerates.

The targets come from `targets=` (a dict or a profile name), or else from the spec's
`audience.trait_requirements` on Big Five dimensions (`high` 4.33, `average` 3.0, `low` 1.67, or a number).

**Output form.** `bigfive_audience` returns flat dicts with full text, plus one identity line for the quota
cell, rather than `Persona` objects. TALP narrative fields are clipped at 600 characters in prompts, and the
15 sentences (about 900 characters) would lose neuroticism and openness. Use `sp.bigfive_traits(scores)` to
attach the scores to a `Persona` as traits.

**Fidelity check (`validate_bigfive`).** It assigns with one instrument and measures with another, the
20-item Mini-IPIP (Donnellan et al. 2006):
- Each item is one `user` prompt, and the persona sits in `system`.
- The result reports per-domain and pooled r, r², MAE and bias (measured − assigned), plus `n_pairs`,
  `n_missing` (item answers that did not parse) and `n_missing_pairs`.
- Every item answer row stores persona_method, agent_id, seed, params, max_tokens, wording_id, raw, value and
  valid.

**BFI-2 terms.** The level-0 sentences of the 7-level bank follow BFI-2 item stems ("is outgoing", "is
reliable", "worries"). The BFI-2 (Soto & John 2017) is (c) its authors and free for non-commercial research use;
ask the authors before any commercial use (see `NOTICE`).

**Port fixes against replicant.**
- No NaN-to-3 imputation. An unparseable answer stays missing, and a domain with a missing item is missing
  (`min_items=4`).
- No EDSL Agent/Survey wrappers. The source's EDSL agent rendering is replaced by `instruction + "Your
  personality:" + text`.
- JSON answers are parsed with `simengine.sim.parse.parse_json`. Other answers go through a strict
  answer rule (a bare number, optionally with "/5", "out of 5" or its option label). A number is never
  pulled out of prose: "I would say 4." and "Not 5" are missing, not guessed.

**Measured with the source code.** Source: replicant `papers/ertan2009/results/validation_continuous.json`,
model stepfun/step-3.5-flash, N = 20, seed 42, with NaN answers imputed as 3 in that run.
- Pooled: r = .911, R² = .830, MAE = .666.
- Per-domain bias: A +0.735 (4.02 → 4.75), N −0.692 (2.94 → 2.25), O +0.660, C +0.444, E −0.034.
- Per-domain r ranges from .80 (A) to .95 (E).

Rank order survives while levels drift toward warm and stable. Report bias, not just r. The numbers have not
been re-measured through this library. `validate_bigfive(llm, n=20, seed=42)` assigns the same scores, so a
rerun is directly comparable.

**Turkish.** No Turkish sentence bank or Mini-IPIP wording exists here. Any translation is unvalidated until
`validate_bigfive` is rerun in Turkish, and all norms are US or English-speaking samples.
