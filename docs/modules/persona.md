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
