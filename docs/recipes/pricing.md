# Recipe: price sensitivity and willingness to pay (choice-based conjoint)

Script: [`examples/recipes/pricing.py`](../../examples/recipes/pricing.py) (under 200 lines, numbered steps 0-8).
Test: `tests/test_recipe_pricing.py`.

```bash
python3 examples/recipes/pricing.py                                    # offline: FakeLLM + synthetic CBC, ~15 s
python3 examples/recipes/pricing.py --data data/cbc.csv --out runs/pricing
python3 examples/recipes/pricing.py --live --max-usd 1 --data data/cbc.csv    # Claude Haiku, key from $ANTHROPIC_API_KEY
python3 examples/recipes/pricing.py --live --key-file <path> --max-usd 1     # or the key from a file
```

Outputs go to `--out` (default `runs/pricing`, git-ignored): `card.md`, `dashboard.html`, `price_curve.csv`, the
seal registry `.onkayit/` and, with `--live`, the reply cache and cost ledger under `.cache/`. `--B` sets the
bootstrap replicates (200 by default, 1000 for the final card).

## When to use

- The brief asks "what will people pay for X?", "which feature is worth adding?" or "where should we put the
  price?", and the answer needs to be a number in currency with an interval: willingness to pay (WTP) per feature,
  a share-vs-price curve, a price elasticity and a revenue-maximising price.
- You have (or can collect on the day) a small choice-based conjoint: each respondent sees 6-12 choice sets of 2-3
  product profiles plus a "buy none" option and picks one. 100+ respondents give usable CIs. The human answers are
  the truth, the baseline and the training data; the LLM panel is graded against them.
- No human choices at all: run steps 0-3 and 5 only, and present the silicon WTP as a hypothesis with its
  inflation warning (see Pitfalls). The card cannot be produced without outcomes.
- Not for: a single yes/no "would you buy at price p" question per person (use a binary survey recipe, e.g.
  `examples/recipes/segment_survey.py`, and `se.sim.batch_calibrate`), or price / promotion changes in real sales
  history (`abm.elasticity.ElasticityShrinkage`, with its LLM-swap placebo).
- The library primitives behind this recipe are in
  [`docs/modules/methods.md`](../modules/methods.md#conjoint-conjointpy) (`MNL`, `conjoint.wtp` with delta-method
  or person-bootstrap CIs, `simulate_shares`, `elasticity`) and `examples/conjoint_pricing.py`.

## Inputs

`--data` is a CSV with **one row per (respondent, choice task, alternative shown)**:

| column | type | meaning |
|---|---|---|
| `person` | str | respondent id; the bootstrap cluster; the train/test split never cuts a person (hash of the id, 30 % test) |
| `segment` | str | respondent segment, e.g. `student` |
| `persona` | str | segment description fed to the LLM (first value per segment is used). **No choice information** |
| `task` | str | choice-set id, unique per person (`p001_t03`); two people answering "task 3" need two ids |
| `design` | str, optional | id of the options shown: the same id means the same profiles. The LLM is asked once per segment x design. Missing: design = task, one call per segment x task (set `--max-usd`) |
| `alternative` | str | `A`, `B`, ... and exactly one `none` row per task (the no-buy option: price 0, attributes 0) |
| `price` | num | price of the option, in `CUR` |
| `<ATTRS>` | num | one column per key of `ATTRS` in step 0 (0/1 or numeric, e.g. `meals`, `organic`, `free_cancel`) |
| `chosen` | 0/1 | exactly one 1 per task |

Without `--data` the script builds 400 synthetic respondents x 4 segments, each answering 8 of 16 shared choice
sets for a meal-kit subscription (A, B, none; prices 229-379 TL). True tastes differ by segment (students are
twice as price-sensitive as young professionals; retirees often buy nothing), with person-level noise. At the
average taste the true WTP is about 16 TL per extra weekly meal, 33 TL for organic and 45 TL for "cancel any
time". The fake LLM gets the direction of every attribute right but feels too little price pain and predicts too
few no-buys.

## Steps (what each one does, what to change on the day, what backs it)

Ranks (#1-#8) refer to the ranked list in the literature review (`PARK-HORTON-AARU.md` section 4, in the
hackathon research folder).

| step | in the script | change on the day | backed by |
|---|---|---|---|
| 0 | `PRODUCT`, `CUR`, `ATTRS` (column -> how the LLM reads it), `SEGMENTS`, `MARKET`, `SYSTEM` | the brief's product and currency, one entry per attribute column, segment texts from the client's research, the reference market (your product A, the main competitor B, none); keep the JSON reply format | Meister et al. (arXiv:2411.05403): segment-level distributions beat per-person persona steering; keep persona texts short (rich biographies did worse in SubPOP and Funhouse Mirrors) |
| 1 | load, column check, person-disjoint split | the split the brief fixes, if any; never split one respondent across train and test | the leakage guard behind every result in `docs/RESULTS.md`; the card's group-overlap audit checks it |
| 2 | backend; `--live` refuses to start without `--max-usd` | `--model` only if the brief demands it | Sonnet − Haiku +1.0 point, not significant (talp73); Park et al. (arXiv:2411.10109): model generation ~1 point. Not worth the budget |
| 3 | one prompt per (segment, design) asking for the probability of each option, plus one no-persona prompt per design | nothing; 4 x 16 + 16 = 80 calls, cents on Haiku | **#4** ask per segment for a verbalised JSON distribution (Meister: TV 0.226; `sim.segment_distribution` closed 0.82 of the gap in G1). The no-persona arm is the control that shows whether the persona text matters |
| 4 | CBC logit on train respondents with segment shifts of the buy constant and the price slope (`mnl`, the baseline to beat) and `stack` = the same logit + log LLM share as one more attribute | add segment x attribute terms when the brief says segments value features differently | **#1** the strongest data baseline, declared on the card (`baselines_oof=["mnl"]`); measured: MNL 0.509 vs equal share 0.210 on the camera conjoint, LLM arms did not beat MNL/HB there. **#3** LLM as a feature of the data model (honest gain small: +0.006 on Twin-2K-500); its fitted weight also acts as **#2** calibration of the LLM's "yes" lean (Schoenegger et al., arXiv:2402.19379) |
| 5 | pooled part-worths -> WTP = −β_attr / β_price per attribute, 95 % person-bootstrap CIs, for the human train respondents and for a silicon sample of 120 respondents drawn from the LLM's segment shares (same segment mix) | pass `train[train.segment == "..."]` to `fit` / `boot` for one segment's WTP; `--B 1000` for the slide | **#6** LLM effect sizes are inflated and need shrinking (Manning et al., arXiv:2404.11794: 13.2x; Hewitt et al.: r = .85 on survey experiments, .34 on megastudies). Wang, Zhang & Zhang (arXiv:2412.19363): combining LLM and human conjoint answers saves 24.9-79.8 % of the data, naive substitution saves nothing. So the human column is the number; the silicon column is a check |
| 6 | share of A in `MARKET` as A's price moves over 80-125 % of its reference price, human CI band from the same bootstrap draws, LLM share, revenue (price x share), arc elasticity; `price_curve.csv` | the sweep range must stay inside the prices respondents saw; replace revenue with margin (`(price − unit_cost) x share`) when the brief gives costs | standard logit share simulation (`methods.conjoint.simulate_shares` / `elasticity` do the same with analytic own/cross elasticities); pooled logit carries the IIA pattern, so for person-level tastes use `conjoint_hb.HBMNL` (same person, new task: HB 0.715 vs MNL 0.482 on the camera conjoint) |
| 7 | `se.seal` of the test predictions ({alternative: probability} per task) with "ours = stack" in the description | nothing; seal before looking at the test choices | pre-registration; the card's selection-bias audit finds `ours` in the seal |
| 8 | `se.evidence.evaluate(task="multiclass", primary="log_loss")` -> ladder (uniform, design shares, design x segment shares, CBC logit), no-persona arm, prediction shuffle, paired CIs, audits, `card.md`, `dashboard.html` | `--B 1000`; `memorization=` once the audit has run | **#1** the card is the deliverable; **#7** memorisation probe (Hewitt) |

Skipped on purpose (the literature ranking's "do not spend time on"): reflection / memory streams, rich biography
personas, fine-tuning, many samples per persona, embedding-similarity scoring (SSR −2.4), a bigger model. #5 (type
mixture weights) and #8 (structured persona schema) are not in this recipe.

## What the jury should see

Offline run, default settings (FakeLLM, synthetic CBC, 125 test respondents x 8 tasks, B = 200):

1. **WTP with CIs, human vs LLM** (TL per unit, train respondents vs silicon sample):

   | attribute | human [95 % CI] | LLM silicon [95 % CI] | LLM / human |
   |---|---|---|---|
   | +1 weekly meal | 10.8 [5.0, 17.1] | 49.5 [32.5, 71.2] | 4.6x |
   | organic | 38.8 [24.0, 53.1] | 87.0 [53.4, 137.6] | 2.2x |
   | cancel any time | 41.3 [29.5, 55.3] | 121.3 [84.2, 164.1] | 2.9x |

   The human CIs cover the synthetic truth (16 / 33 / 45 TL). The LLM ranks the features the same way but its WTP
   is two to five times too high, and none of its intervals overlap the human ones. Say it on the slide: "the
   panel knows what people like, not how much they will pay".
2. **The price curve** for A against a competitor at 349 TL: share falls from 0.66 [0.62, 0.72] at 239 TL to 0.30
   [0.26, 0.34] at 374 TL; elasticity goes from −1.0 to −2.8; revenue is flat between 239 and 269 TL and peaks at
   254 TL. The LLM curve is much flatter (0.64 -> 0.45): pricing on it would overprice.
3. **The baseline ladder** (log loss per task on test respondents, lower is better, person-bootstrap 95 % CI):
   uniform 1.099, shuffled stack 1.064, no-persona LLM 1.062, LLM 1.022 [0.973, 1.086], design shares 0.979,
   design x segment shares 0.897, CBC logit 0.878 [0.845, 0.911], stack 0.877 [0.845, 0.911]. Stack − logit
   +0.000 [−0.000, +0.001]: badge "inconclusive". The LLM adds nothing on top of a segmented logit fitted to the
   human answers. That is the expected result (camera conjoint: LLM arms did not beat MNL/HB), and it is the
   honest headline.
4. **The controls**: the no-persona LLM is barely better than uniform (1.062); the persona LLM improves on it by
   0.040 [−0.013, +0.090], not significant at 125 respondents, so do not claim the persona text works. The
   shuffled stack (predictions moved across tasks) falls back to the no-persona level (1.064), so the stack's
   score comes from matching each task, not from overall shares.
5. **The seal and the spend**: the hash and time of the sealed test predictions and, with `--live`, the ledger
   total against `--max-usd`.

## Pitfalls

- **Leakage.** (a) Persona text must not describe choices or prices paid ("always buys the cheapest plan"). (b) Fit
  the logit and the stack on train respondents only; the card scores test respondents. (c) Do not tune `SYSTEM`,
  `ATTRS` wording or `MARKET` on the test choices: tune on train, then seal. (d) The silicon sample must use the
  LLM's shares only; a silicon WTP fitted after "calibrating" toward the human WTP is no longer an independent check.
- **Imbalanced accuracy.** When most respondents pick "none" (or the cheapest position), "always none" scores a high
  hit rate and beats every model. The card therefore uses log loss on the full probability vector; report hit rate
  only next to the share of the majority option. Without `primary=`, `evaluate` switches accuracy to macro-F1 when
  one class holds over 60 %, and the card's class-imbalance audit warns either way.
- **Memorisation.** For a real brand with public list prices the model may recall the price instead of simulating
  a respondent, and its WTP then tracks the list price. Run once with the brand name removed and the attributes
  paraphrased; audit with `se.evidence.memorization_audit` and pass it as `evaluate(..., memorization=...)`. Until
  then the card flags "not proven: leakage/memorization". Prefer products and prices after the model's cutoff
  (Hewitt et al. report post-cutoff studies separately).
- **WTP is a ratio.** When β_price is near zero (a price range too narrow, or a segment that ignores price) WTP
  explodes and its bootstrap interval becomes huge or changes sign. Check that the price part-worth's CI is well
  away from zero before quoting WTP; use the person bootstrap (asymmetric percentile CI), not ± 2 SE.
- **Extrapolation.** The curve is only valid inside the prices respondents saw (229-379 TL in the demo). The script
  flags a revenue maximum at the edge of the sweep; test wider prices instead of extending the curve.
- **Pooled tastes.** The pooled logit gives one WTP for an average respondent and IIA shares. If the segment WTPs
  differ (rerun step 5 per segment), present them per segment, or use `conjoint_hb.HBMNL`.
- **Unparsed replies.** A reply without numeric option keys becomes 1e-3 per option and then uniform after
  renormalising. Many of those flatten the LLM columns; fix `SYSTEM` or `max_tokens` if the LLM curve looks flat.
- **Budget.** `--live` refuses to start without `--max-usd`. Once the ledger total reaches the cap the backend raises
  `BudgetExceeded` before any new call; finished replies stay in the cache, so the same command with a higher cap
  resumes. The key comes from `$ANTHROPIC_API_KEY` or `--key-file` and is never printed.
