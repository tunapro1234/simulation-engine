# Recipe: rank ad creatives for an audience (lift vs control, with CIs)

Script: [`examples/recipes/ad_ab.py`](../../examples/recipes/ad_ab.py) (about 230 lines, numbered steps 0-8).
Test: `tests/test_recipe_ad_ab.py`.

```bash
python3 examples/recipes/ad_ab.py                                   # offline: FakeLLM + synthetic campaign, ~15 s
python3 examples/recipes/ad_ab.py --data data/campaign.csv --audience students,parents --out runs/ad \
    --product "a meal-kit subscription"                              # the product line every prompt shows
python3 examples/recipes/ad_ab.py --live --max-usd 1 --data data/campaign.csv   # Claude Haiku, key from $ANTHROPIC_API_KEY
python3 examples/recipes/ad_ab.py --live --key-file <path> --max-usd 1        # or the key from a file
```

Outputs go to `--out` (default `runs/ad_ab`, git-ignored): `card.md`, `dashboard.html`, the seal registry
`.onkayit/` (with `prereg.md`, which names `ours` before the seal) and, with `--live`, the reply cache and cost ledger under `.cache/`.

## When to use

- The brief has a control ad (the current one, or "no ad") and a handful of new creatives or messages (four in the
  script), a target audience, and the question "which one should we run, and how much better than what we have
  is it?". The answer is a ranking plus a lift in percentage points over the control, each with a 95 % CI.
- You have some past campaign data with randomised arms: one row per person who saw one ad, and whether that person
  converted. A few hundred people per arm is enough to run it; the CIs tell you whether it is enough to decide.
  The past data gives the baselines, the calibration and the test split. Without outcomes there is nothing to grade.
- A creative nobody has shown yet can be scored by the panel, but the card can only grade creatives that have
  outcomes. Present the new one's calibrated prediction as a hypothesis and say so on the slide.
- Not for: headline / copy pairs with thousands of past A/B tests (use `methods.ABStack`, see `examples/ab_test.py`:
  0.817 [0.770, 0.860] on 300 Upworthy tests, where a Haiku persona panel was at chance, 0.522), or funnels with
  several steps per visitor (`examples/recipes/site_visitors.py`).

## Inputs

`--data` is a CSV with **one row per person who was shown one ad**. Arms must be randomised: if an ad was shown
only to one segment, its lift is confounded with that segment.

| column | type | meaning |
|---|---|---|
| `person_id` | str | person id; the bootstrap cluster; one person = one row = one arm |
| `segment` | str | audience segment, e.g. `students` |
| `persona` | str | segment description fed to the LLM (first value per segment is used). **No outcome information** |
| `arm` | str | creative id; the control's id is `control` (`--control base` if yours is called `base`) |
| `message` | str | the ad text, or a text description of the visual, that this person saw |
| `converted` | 0/1 | the KPI: click, sign-up or purchase. Pick one before looking at results |
| `date` | str, optional | ISO date; when present the last 30 % of dates are the test split, else 30 % of persons (hash of `person_id`) |

Text fields that contain commas must be quoted, as in any CSV. `--audience a,b` restricts the lift table to those
segments (the card still scores everyone in the test split).

Without `--data` the script builds 2,500 synthetic people x 4 segments x 5 arms (21 days, ~8 % convert). Price wins
with students, speed with professionals, family with parents, and retirees barely respond. The fake LLM says "yes"
about five times too often (mean 0.41), gets the three big segment x creative effects right, misses the three smaller ones, and wrongly
expects the green ad to work for retirees.

## Steps (what each one does, what to change on the day, what backs it)

| step | in the script | change on the day | backed by |
|---|---|---|---|
| 0 | `PRODUCT` (or `--product`), `ADS`, `SEGMENTS`, `SYSTEM` | the brief's product, the control + creatives, segments from the client's analytics; describe visuals in one line; keep the JSON reply format | Meister et al. (arXiv:2411.05403): segment-level answers beat per-person persona steering; Hu & Collier: persona variables explain < 10 % of the variance, so keep personas short (rich biographies: SubPOP, Funhouse Mirrors r = 0.20) |
| 1 | load + split (time, else person-disjoint) | the test period the brief fixes; never a random row split across one person | the leakage guard behind every result in `docs/RESULTS.md` |
| 2 | backend; `--live` needs `--max-usd` | `--model` if the brief demands another model | Sonnet − Haiku +1.0 point, not significant (talp73); Park et al. (arXiv:2411.10109): model generation ~1 point |
| 3 | one prompt per (segment, ad) cell + a no-persona prompt per ad | nothing; 4 x 5 + 5 = 25 calls, well under a cent on Haiku. A refused or failed call becomes 0.5 and is counted in the first output line; more than 20 % failed stops the run | ranked #4: ask per segment directly (Meister: real segment distributions beat persona steering); the no-persona arm is the control that shows whether the persona text matters (talp73: L1 − L0 +0.5 points, not significant) |
| 4 | `arm_rate`, `cell_rate` from train people, shrunk toward the global rate (k = 5) | add the client's own rule ("last campaign's winner") as another `baselines_oof` column | ranked #1: baseline ladder and gap closed (Park normalisation; Twin-2K-500 closes ~56 % of the gap) |
| 5 | `stack`: logistic on logits of `cell_rate` and the LLM, fit on train with **out-of-fold** cell rates; prints the "yes" bias and the LLM's weight | nothing; with no history, swap in `methods.BinaryCalibrator` on a small labelled pilot | ranked #2: LLMs lean "yes" (Schoenegger et al., arXiv:2402.19379: mean forecast 57 % vs 45 % resolved yes; `sim.batch_calibrate` 69.4 % → 76.5 % in talp73). Ranked #3: LLM as a feature of the data model (honest gain small: +0.006 in Twin-2K-500). Ranked #6: the weight < 1 shrinks the LLM's effects |
| 6 | `prereg.md` (ours, primary metric, baselines, audience) written to the registry, then `se.seal` of the test predictions | nothing; seal before you look at test outcomes | pre-registration; the card's selection-bias audit reads `prereg.md` and passes only if it is older than the seal and names `ours` (sealed column names alone do not count) |
| 7 | lift table for `--audience`: observed lift vs control (bootstrap within each arm), predicted lift of `stack` (CI from refits on resampled train people, cell rates included), raw `llm`, `cell_rate`; ranking and Spearman | `--audience` for the brief's target; which arm is control | ranked #6: LLM effect sizes are inflated (Manning et al., arXiv:2404.11794: 13.2x; Hewitt et al.: r = .85 on survey experiments but only r = .34 on megastudies, which is what "4 creatives, one outcome" is), so the shrunk `stack` lift is the one to quote |
| 8 | `se.evidence.evaluate(primary="auc")` → ladder, no-persona arm, automatic persona shuffle, paired CIs, audits, `card.md`, `dashboard.html` | `--B 1000` for the final card; `multiplier=True` adds "worth N humans" (about 20 s more); `memorization=` once the audit has run | ranked #1 again: the card is the deliverable; ranked #7: memorisation probe |

Skipped on purpose (the literature ranking's "do not spend time on"): reflection / memory streams, rich biography
personas, fine-tuning, many samples per persona, embedding-similarity scoring (SSR −2.4), a bigger model.

## What the jury should see

Offline run, default settings (FakeLLM, synthetic campaign, 693 test people in the last 6 of 21 days, 124 of them in the control arm, B = 300):

1. **The "yes" bias, then the fix.** "Raw panel: 41 % would sign up. Observed: 8.4 %. After calibration the panel's
   effects keep 0.44 of their stated size." One line, from step 5.
2. **The lift table** (all segments, test people):

   | ad | observed lift [95 % CI] | stack lift [95 % CI] | raw llm | cell_rate |
   |---|---|---|---|---|
   | family | +0.072 [+0.000, +0.138] | +0.027 [+0.002, +0.044] | +0.052 | +0.048 |
   | green | +0.006 [−0.047, +0.067] | +0.011 [−0.003, +0.023] | +0.023 | +0.032 |
   | price | +0.020 [−0.042, +0.090] | +0.027 [+0.001, +0.041] | +0.049 | +0.046 |
   | speed | +0.029 [−0.033, +0.091] | +0.019 [−0.001, +0.032] | +0.032 | +0.036 |

   Ranking: stack family > price > speed > green, observed family > speed > price > green (Spearman 0.80). The
   raw LLM and `cell_rate` alone give the same order (0.80 each), so the panel does not improve the ranking over
   past data. Only "family" has an observed CI whose lower end is at zero (+0.000); none clearly clears it. Say "family first, the
   rest not separable at this n", and quote the stack lift (+2.7 points) rather than the observed +7.2.
3. **The baseline ladder** (person-level AUC, test, person-bootstrap 95 % CI): shuffled-persona stack 0.468,
   random 0.500, no-persona LLM 0.508, raw LLM 0.572, `arm_rate` 0.577, segment mean 0.615, stack 0.628
   [0.564, 0.701], `cell_rate` 0.636. Stack − best baseline −0.009 [−0.031, +0.013]: badge "inconclusive" (the card's badge line uses the worst
   of the three tied baselines, CI [−0.071, +0.013]). The panel
   ties past data; it does not beat it. That is the expected result (ranked #3) and it is the honest headline.
4. **The controls**: the no-persona LLM sits at chance (0.508) while the persona LLM reaches 0.572, so the raw
   panel's signal comes from the segment x creative match, not from generic "ads with discounts work" priors. The
   shuffled-persona row shuffles the stack, which is mostly `cell_rate`, so it shows the stack uses per-segment
   information, not that the panel does. Whether the panel adds anything is stack vs `cell_rate`: it does not here.
5. **The seal and the spend**: the hash and time of the sealed test predictions (selection-bias audit "ok":
   `prereg.md` named `stack` before the seal) and, with `--live`, the ledger
   total against `--max-usd`.

## Pitfalls

- **Leakage.** (a) The stack is fit on train rows, and an in-sample `cell_rate` already contains each train row's
  own outcome. Fit on it and the LLM gets zero weight (we saw −0.01 before switching to out-of-fold rates; 0.44
  after). Keep step 5's folds. (b) Persona text must not mention the outcome ("loves our app"). (c) Do not tune
  `SYSTEM` or the ad descriptions on test outcomes: tune on train, then seal. (d) With dates, the later days are the
  test; a random row split lets later weeks inform the predictions for earlier ones.
- **Imbalanced accuracy.** With ~9 % converting in the test days, "nobody converts" scores 91 % plain accuracy and beats every model.
  The card uses AUC; lifts are reported in absolute percentage points, never as "accuracy". A control arm of ~120
  people gives a lift CI about ±0.065 wide: check the CI before ranking, and size the next test from it.
- **Memorisation.** A famous brand's real campaign, or a public A/B archive (Upworthy), may be in the model's training
  data: Opus recognises Upworthy headlines (membership AUC 0.917) and loses 0.076 on paraphrases
  (`docs/RESULTS.md` §2). Paraphrase the creatives and drop the brand name in a second run; for pair data use
  `se.evidence.ab_memorization`, otherwise `se.evidence.memorization_audit`, and pass the result as
  `evaluate(..., memorization=...)`. Until then the card says "not proven: leakage/memorisation". Prefer campaigns
  after the model's cutoff (Hewitt et al. report post-cutoff studies separately).
- **Four items.** A Spearman over four ads has 24 possible orders; a perfect match by chance is ~4 %, and 0.80 (one
  adjacent swap) or better happens by chance 4 times in 24 (~17 %): not evidence on its own. Picking the best of four observed lifts also overstates its lift (winner's
  curse); the shrunk `stack` lift and the CI are the numbers to quote.
- **Non-randomised arms.** If the client targeted each ad at a different audience, the observed lift mixes the ad
  with the audience. The lift table weights segments by the target audience, which helps only if every segment saw
  every arm; otherwise say so on the card.
- **Unparsed replies.** A `"p"` is salvaged even from JSON cut off at `max_tokens`; a reply with no number, a
  refusal or a failed call becomes 0.5, and the first output line counts them (`n unparsed or failed`). A few of
  those flatten the raw lift; fix `SYSTEM` or `max_tokens` if the count is not 0.
- **Budget.** `--live` refuses to start without `--max-usd`. Once the ledger total reaches the cap the backend raises
  `BudgetExceeded` before any new call (calls already in flight finish). Finished replies stay in the cache, so the
  same command with a higher cap resumes. The key comes from the environment or `--key-file` and is never printed.
