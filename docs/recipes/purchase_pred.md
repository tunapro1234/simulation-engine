# Recipe: individual purchase prediction (person x category, the talp73 setup)

Script: [`examples/recipes/purchase_pred.py`](../../examples/recipes/purchase_pred.py) (under 200 lines, numbered
steps 0-9). Test: `tests/test_recipe_purchase_pred.py`.

```bash
python3 examples/recipes/purchase_pred.py                                   # offline: FakeLLM + synthetic shoppers, ~20 s
python3 examples/recipes/purchase_pred.py --data data/cells.csv --out runs/purchase
python3 examples/recipes/purchase_pred.py --live --max-usd 1 --data data/cells.csv   # Claude Haiku, key from $ANTHROPIC_API_KEY
python3 examples/recipes/purchase_pred.py --live --key-file <path> --max-usd 1 --persons 100   # key from a file, first 100 persons
```

Outputs go to `--out` (default `runs/purchase_pred`, git-ignored): `card.md`, `dashboard.html`, the seal registry
`.onkayit/` and, with `--live`, the reply cache and cost ledger under `.cache/`.

## When to use

- The brief asks "will this customer buy X?" for **individual** people: per person and per product category, from
  who they are (demographics / a short survey) and what they bought before. This is the claim behind numbers like
  "73 % accuracy" (TALP); the recipe is our reproduction of that setup (`talp73`, Open e-commerce 1.0, 400 persons,
  20 categories, a 6-month target window).
- You have, for the same people, purchases before a cutoff and the outcome after it. Without the after-cutoff
  outcome there is nothing to grade; run it on a past cutoff first.
- Not for: population shares ("what share of 25-34s buy baby products?"; ask per segment with
  `sim.segment_distribution`, closed 0.82 of the gap in G1), next-item recommendation over thousands of SKUs
  (`methods.next_action`, `bench/tasks/instacart.py`), or demand forecasting over time (`task="forecast"`).

## Inputs

`--data` is a CSV with **one row per (person, category)**; every person has a row for every category.

| column | type | meaning |
|---|---|---|
| `person_id` | str | person id; the bootstrap cluster; one person is never split across train and test |
| `category` | str | product category |
| `demo_<name>` | str | one or more demographic / survey columns (`demo_age`, `demo_income`, ...), constant within a person; they become the persona text and the survey model's features. **No outcome information** |
| `n_before` | int | purchases in this category in the 12 months before the cutoff |
| `n_recent` | int | of those, purchases in the last 6 months before the cutoff |
| `bought` | 0/1 | at least one purchase in this category in the target window after the cutoff |
| `p_true` | float, optional | simulated data only: the known true P(buy); added to the oracle ceiling's features |

Without `--data` the script builds 240 synthetic shoppers x 8 categories (35 % of cells bought). A hidden person x
category taste drives both past and future purchases, and demographics move a few categories a little, so history
beats demographics as it did in talp73. The fake LLM reads the history, half-reads the demographics and says
"yes" too often.

## Steps (what each one does, what to change on the day, what backs it)

| step | in the script | change on the day | backed by |
|---|---|---|---|
| 0 | `CATS`, `DEMO`, `WINDOW`, `SYSTEM` | the brief's categories, survey items and window; keep the JSON reply format | Park et al. (arXiv:2411.10109) v3: demographics alone 0.74 normalised vs survey 0.82, so short survey-style personas, no invented biography (SubPOP, Funhouse Mirrors r = 0.20) |
| 1 | load + person-disjoint split (30 % of persons test) | a split the brief fixes; never a random cell split | leakage guard of every result in `docs/RESULTS.md` |
| 2 | balanced mask: per person min(5, #bought, #not bought) cells of each class | nothing; it is the talp73 "1b" mask | talp73: on all cells "never buys" scores 76.4 % and the Haiku persona 69.4 %, below the constant; the mask makes chance 0.5 |
| 3 | backend | `--live --max-usd`; Haiku by default (`--model` to change) | Sonnet − Haiku +1.0 point, not significant (talp73); Park: model generation ~1 point |
| 4 | one call per person and arm, all categories in one JSON: L0 no persona (1 call), L1 demographics, L2 demographics + purchase counts | the history text: add recency or spend if the data has it | talp73 1b(c): L0 0.578, L1 0.583, L2 0.653 (Haiku), so the gain is in the history, not the persona; Park: a bullet summary keeps most of the interview's accuracy |
| 5 | LLM-free baselines on train persons: `popularity`, `survey` (logistic on demographics), `history` (logistic on counts), `hist_survey` | add the client's own scoring rule as another `baselines_oof` column | ranked #1: baseline ladder. talp73 1b(c): popularity 0.582, survey 0.599, history 0.675, history+survey 0.672; no LLM arm beat the history model |
| 6 | `llm_l2_cal` (Platt on train) and `stack` (logistic on logits of `hist_survey` and L2) = ours | nothing; `ours` is named before the truth | ranked #2: LLMs lean "yes" (Schoenegger et al.: mean forecast 57 % vs 45 % resolved yes); `sim.batch_calibrate` 69.4 % → 76.5 %. ranked #3: LLM as a feature of the history model; honest gain small (+0.006 in Twin-2K-500) |
| 6b | `oracle`: history + survey (+ `p_true`) fit on the test cells' own labels | nothing; it is the ceiling, never a prediction | ranked #1: gap closed (Park normalises 68.85 % by the 81.25 % retest = 0.85; Twin-2K-500 closes ~56 %). Purchases have no retest, so the ceiling is a model oracle and the card labels it |
| 7 | plain accuracy on all test cells: always-no vs raw L2 vs history | nothing | the "73 %" trap in one line (talp73 1a) |
| 8 | `se.seal` of the test predictions, `ours = stack` in the description | nothing; do it before you look at test outcomes | pre-registration; the selection-bias audit checks that `ours` is in the seal |
| 9 | `se.evidence.evaluate(primary="balanced_accuracy", ceiling_col="pred_oracle")` → ladder, controls, CIs, gap closed, `card.md`, `dashboard.html` | `--B 1000` for the final card; `memorization=` once the audit has run | ranked #1 again: the card is the deliverable |

Skipped on purpose (the literature ranking's "do not spend time on"): reflection / memory streams, rich biography
personas, fine-tuning (Park 0.79, below the prompt), several samples per person (L1x3 − L1 = 0.0), free-text +
embedding scoring (SSR −2.4 in talp73).

## What the jury should see

1. **The trap, then the fix.** "On all cells, 'nobody buys anything' is right 65 % of the time (76 % in talp73) and
   beats the raw LLM. On a balanced mask chance is 50 %, and that is where we score."
2. **The baseline ladder** with person-bootstrap 95 % CIs, from the card. Offline run (FakeLLM, 71 test persons,
   348 masked cells, balanced accuracy): L0 no persona 0.457, shuffled stack 0.480, random 0.500, L1 0.509,
   popularity 0.563, survey 0.563, raw L2 0.615, history 0.649, history+survey 0.649, stack 0.652 [0.607, 0.698],
   calibrated L2 0.675, oracle ceiling 0.718.
3. **Gap closed** = (ours − best baseline) / (oracle − best baseline): 4 % [−10, 25] offline, badge
   "inconclusive". The card itself writes "n/a, ceiling ≈ baseline" because the oracle is not clearly above the
   history model (+0.043 [−0.006, 0.088]); with that little headroom there is no gap to speak of. A tie with the
   history model is the talp73 result too; say so. It is a result, not a failure.
4. **The controls**: L0 (no persona) and the automatic persona shuffle sit at or below 0.5, so whatever the LLM
   adds comes from the person's data, not from category priors in the prompt.
5. **Persona vs history**: L1 − L0 (what demographics add) next to L2 − L1 (what history adds). In talp73 the first
   was +0.5 points and not significant, the second +7.
6. **The seal and the spend**: the hash and time of the sealed test predictions, and with `--live` the ledger total
   against `--max-usd`.

## Pitfalls

- **Leakage.** History must end before the cutoff and the target window must start after it. Demographic / survey
  answers collected after the window can carry the outcome (talp73's survey was answered after the target window,
  which favours the survey model and L1 alike: note it on the card). Baselines, calibration and the stack are fit on
  train persons only; the oracle is the one model that sees test labels, and it is only the ceiling. Do not tune
  `SYSTEM` on test outcomes: tune on train persons, then seal.
- **Imbalanced accuracy.** Most person x category cells are "not bought", so plain accuracy rewards "never buys"
  (step 7 prints it). Score the balanced mask with balanced accuracy, or AUC (`primary="auc"`). Balanced accuracy
  thresholds at 0.5, which is fair here only because every model is fit on balanced cells; if you score all cells,
  refit on all cells or switch to AUC.
- **Pooled vs within-person.** A model that only knows who shops a lot ranks persons, not categories. The mask keeps
  both classes inside each person; for a strictly within-person score use talp73's 1b(c) (top half of each person's
  cells by p is "yes") or the mean within-person AUC.
- **Memorisation.** Public purchase datasets (Open e-commerce 1.0 is CC0, 2024) and well-known category
  popularity can be in the model's training data. The card flags "not proven: leakage/memorisation" until you run
  `se.evidence.memorization_audit` or `recall_probe` and pass the result as `evaluate(..., memorization=...)`;
  prefer a target window after the model's cutoff (Hewitt et al. report post-cutoff studies separately).
- **Gap closed with a model ceiling.** The oracle is fit in-sample on the test cells, so it is optimistic and the
  gap closed is, if anything, understated. With a real ceiling (a retest, a second wave) pass it as `ceiling_col`
  instead; never compare this gap closed with Park's 0.85, whose ceiling is human retest. When the oracle ties the
  strongest baseline exactly, `evaluate` cannot divide; the script then prints "oracle ties the best baseline" and
  builds the card without a ceiling.
- **Unparsed replies.** A missing category or a non-numeric value becomes 0.5 and is counted in the first output
  line; above a few percent, fix `SYSTEM` or `max_tokens` before trusting the arms.
- **Budget.** `--live` refuses to start without `--max-usd`. Calls are 1 + 2 x persons; `--persons` caps them.
  Once the ledger total reaches the cap the backend raises `BudgetExceeded` before any new call (calls already in
  flight finish), finished replies stay in the cache, so the same command with a higher cap resumes. The key comes
  from the environment or `--key-file` and is never printed.
