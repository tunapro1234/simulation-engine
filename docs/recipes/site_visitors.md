# Recipe: simulated site visitors (landing page / checkout funnel)

Script: [`examples/recipes/site_visitors.py`](../../examples/recipes/site_visitors.py) (under 200 lines, numbered
steps 0-8). Test: `tests/test_recipe_site_visitors.py`.

```bash
python3 examples/recipes/site_visitors.py                                   # offline: FakeLLM + synthetic funnel, ~12 s
python3 examples/recipes/site_visitors.py --data data/funnel.csv --out runs/funnel
python3 examples/recipes/site_visitors.py --live --max-usd 1 --data data/funnel.csv   # Claude Haiku, key from $ANTHROPIC_API_KEY
python3 examples/recipes/site_visitors.py --live --key-file <path> --max-usd 1        # or the key from a file
```

Outputs go to `--out` (default `runs/site_visitors`, git-ignored): `card.md`, `dashboard.html`, the seal registry
`.onkayit/` and, with `--live`, the reply cache and cost ledger under `.cache/`.

## When to use

- The brief has two or more versions of a page or flow (landing page, product page, cart, checkout) and asks which
  converts better, or where visitors drop off, **and** you have some past traffic per step (even a few hundred
  visitors). The past traffic gives the baselines and the test split; without it there is nothing to grade.
- The brief has a new variant nobody has shown to visitors yet. The LLM can still rank it against the old one, but
  the card can only grade the old variants; say so on the slide.
- Not for: a pure headline / ad-copy A/B with thousands of past tests (use `methods.ABStack`, see
  `examples/ab_test.py`: 0.817 on Upworthy where a persona panel was at chance, 0.522), or traffic forecasting over
  time (`task="forecast"`).

## Inputs

`--data` is a CSV with **one row per (visitor, funnel step the visitor reached)**. A visitor who left at step 2 has
two rows: step 1 with `continued=1` and step 2 with `continued=0`.

| column | type | meaning |
|---|---|---|
| `visitor_id` | str | person id; the bootstrap cluster; one visitor is never split across train and test |
| `segment` | str | visitor segment (device x new/returning, traffic source, ...) |
| `persona` | str | segment description given to the LLM; the first value per segment is used; **no outcome information** |
| `variant` | str | page variant id (`A`, `B`, ...); the first two in sorted order are compared, control first |
| `step` | str | funnel step name |
| `step_order` | int | 1, 2, ...; the highest value is the purchase step |
| `page_text` | str | text description of that variant's page at that step (what the visitor sees) |
| `continued` | 0/1 | 1 = went on to the next step (at the last step: purchased), 0 = dropped off here |
| `date` | str, optional | ISO date; when present the last 30 % of dates are the test split, else a fixed 30 % of visitors |

Without `--data` the script builds a synthetic 1,600-visitor, 4-step funnel whose ground truth contains one effect the
fake LLM does not know (autoplay video hurts on phones), so the card shows an LLM that is useful but not complete.

## Steps (what each one does, what to change on the day, what backs it)

| step | in the script | change on the day | backed by |
|---|---|---|---|
| 0 | `PAGES`, `SEGMENTS`, `SYSTEM` | page texts from the client's screenshots; segments from their analytics; keep the JSON reply format | Meister et al. (arXiv:2411.05403): segment-level questions beat per-person role play; structured persona schemas (Ye et al.) are **not** verified, so keep personas short |
| 1 | load + split | if the brief fixes a test period, use it; never a random row split | time / person-disjoint split = the leakage guard of every result in `docs/RESULTS.md` |
| 2 | backend | `--live --max-usd`; Haiku by default (`--model` to change) | Sonnet − Haiku +1.0 point, not significant (our talp73 runs); a bigger model is not where the gain is |
| 3 | one prompt per (segment, variant, step) + no-persona control per (variant, step) | nothing; 4 segments x 2 variants x 4 steps = 40 calls, cost under a cent on Haiku | persona vs no persona vs shuffled persona closed 0.55 / 0.45 / 0.25 of the gap (`docs/RESULTS.md`, CES persona control); many samples per person added 0.0 |
| 4 | `step_rate`, `cell_rate` from train traffic, shrunk toward the global rate | add a baseline the client already uses (last month's rate) as another `baselines_oof` column | ranked #1: baseline ladder and gap closed (Park et al. arXiv:2411.10109 normalisation; Twin-2K-500 gap ~56 %) |
| 5 | `stack`: logistic regression on logits of `cell_rate` and the LLM, fit on train | nothing; it is the "ours" named before the truth | ranked #2 (calibrate the LLM's "yes" bias; `sim.batch_calibrate` 69.4 % → 76.5 %) and #3 (LLM as a feature of a history model; honest gain is small: +0.006 in Twin-2K-500) |
| 6 | `se.seal` of the test predictions, `ours = stack` in the description | nothing; do it before you look at test outcomes | pre-registration; the card's selection-bias audit checks that `ours` was named in the seal |
| 7 | per-variant conversion, observed lift B−A with a visitor-bootstrap CI, predicted lift per arm, continue rate per step | which two variants to compare | ranked #6: LLM effect sizes are inflated (Automated Social Science, arXiv:2404.11794: 13.2x on average; Hewitt et al. r = .85 but with exaggerated effects), so put the predicted lift next to the observed one and trust the shrunk `stack` lift over the raw `llm` lift |
| 8 | `se.evidence.evaluate` → ladder, controls, paired CIs, audits, `card.md`, `dashboard.html` | `--B 1000` for the final card; pass `memorization=` when you have run the audit (see pitfalls) | ranked #1 again: the card is the deliverable |

Skipped on purpose (section "do not spend time on" of the literature ranking): reflection / memory streams, rich
biography personas, fine-tuning, many samples per person, embedding-similarity scoring (SSR −2.4).

## What the jury should see

1. **The variant decision with its uncertainty.** "B converts 17.9 % vs A 8.8 %, lift +9.2 points [+2.8, +15.3] on
   held-out visitors; the simulation predicted +4.3 (raw LLM) / +2.1 (calibrated stack)." Direction right, size
   under-called: say that, it is the honest result.
2. **Where visitors drop off**, per step and variant, observed next to the LLM's estimate. A step where the LLM is far
   off (step 1 on phones in the synthetic data) is the finding: the simulation does not know about that page element,
   go and look at it.
3. **The baseline ladder** from the card: random 0.50, marginal 0.50, step rate, segment mean, cell rate, LLM, stack,
   all with 95 % CIs. If `stack` only ties `cell_rate` (as in the offline run), the badge is "inconclusive" and the
   slide says so. A tie with the strongest baseline is a result, not a failure.
4. **The controls**: the no-persona arm (`llm_nopersona`) and the automatic persona shuffle (`shuffled_stack`) sit
   near 0.5, so the gain comes from who the visitor is, not from the page text alone.
5. **The seal**: the hash and time of the sealed test predictions, earlier than the evaluation.
6. **Spend**: with `--live`, the last line gives this run's cost and the ledger total against `--max-usd`.

## Pitfalls

- **Leakage.** The persona must describe the segment, not its outcome ("has bought before" is history and allowed
  only if it is true before the test dates; "usually abandons at checkout" is the label). `page_text` must not name
  the winning variant. `cell_rate` and the stack are fit on the train split only; with a `date` column the test days
  come last, otherwise whole visitors are held out. Do not tune `SYSTEM` while looking at test outcomes: tune on train,
  then seal.
- **Imbalanced accuracy.** Most visitors never buy (8-18 % in the synthetic funnel), so "nobody buys" scores 85-90 %
  accuracy and says nothing. The recipe uses AUC as the primary metric and shows the marginal-rate baseline;
  `evaluate` switches to macro-F1 by itself when the majority class is above 60 % and no primary is given. Report the
  per-step continue rate and the conversion, never plain accuracy.
- **Memorisation.** A famous public A/B test (a well-known case study, a public archive) may be in the model's
  training data, and the LLM then "predicts" a result it has read. The card flags "not proven: leakage/memorisation"
  until you run `se.evidence.memorization_audit` (hit rate on real vs scrambled items) or `correlation_audit` /
  `recall_probe` (study named vs blind) and pass the result as `evaluate(..., memorization=...)`. Prefer data
  collected after the model's cutoff (Hewitt et al. report post-cutoff studies separately).
- **Counting rows instead of visitors.** Visitors who go further have more rows. The bootstrap clusters on
  `visitor_id` (`group`), and the conversion and lift in step 7 are per visitor.
- **Small cells.** A segment x variant x step cell with a handful of visitors gives a noisy `cell_rate`; the
  recipe shrinks it toward the global rate with 5 pseudo-visitors. With many segments, merge segments rather than
  trust a 3-visitor cell.
- **Budget.** `--live` refuses to start without `--max-usd`; once the ledger total reaches it, the backend raises
  `BudgetExceeded` before any new call (calls already in flight finish, so the total can pass the cap by a few
  calls), and finished replies stay in the cache, so the same command with a higher cap resumes. The key is read from the
  environment or `--key-file` and never printed.
