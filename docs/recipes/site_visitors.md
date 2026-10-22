# Recipe: simulated site visitors (landing page / checkout funnel)

Script: [`examples/recipes/site_visitors.py`](../../examples/recipes/site_visitors.py) (about 220 lines, numbered
steps 0-8). Test: `tests/test_recipe_site_visitors.py`.

```bash
python3 examples/recipes/site_visitors.py                                   # offline: FakeLLM + synthetic funnel, ~14 s
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
| `date` | str, optional | ISO date; when present the visitors who arrived last (~30 %, by each visitor's first date, so a visitor is never cut in two) are the test split, else a fixed 30 % of visitors |

Without `--data` the script builds a synthetic 1,600-visitor, 4-step funnel whose ground truth contains one effect the
fake LLM does not know (autoplay video hurts on phones), so the card shows an LLM that is useful but not complete.

## Steps (what each one does, what to change on the day, what backs it)

| step | in the script | change on the day | backed by |
|---|---|---|---|
| 0 | `PAGES`, `SEGMENTS`, `SYSTEM` | page texts from the client's screenshots; segments from their analytics; keep the JSON reply format | Meister et al. (arXiv:2411.05403): segment-level questions beat per-person role play; structured persona schemas (Ye et al.) are **not** verified, so keep personas short |
| 1 | load + split | if the brief fixes a test period, use it; never a random row split | time / person-disjoint split = the leakage guard of every result in `docs/RESULTS.md` |
| 2 | backend | `--live --max-usd`; Haiku by default, `--model claude-sonnet-5-5` for the final run (the script turns thinking off, which Opus 5.5 cannot do: Opus needs `thinking=None` and a larger `max_tokens`) | model size does matter in this repo's own runs: Upworthy direction Haiku / Sonnet / Opus 5.5 = 0.608 / 0.688 / 0.765 (`docs/RESULTS.md` section 2); Haiku is the cheap first pass |
| 3 | one prompt per (segment, variant, step) + no-persona control per (variant, step) | nothing; 4 segments x 2 variants x 4 steps = 40 calls, cost under a cent on Haiku. A failed or refused call becomes 0.5 in its slot; if more than a quarter fail the run stops with the first error instead of grading a card of 0.5s | persona vs no persona vs shuffled persona closed 0.55 / 0.45 / 0.25 of the gap (`docs/RESULTS.md`, CES persona control); many samples per person added 0.0 |
| 4 | `step_rate`, `cell_rate` from train traffic, shrunk toward the global rate | add a baseline the client already uses (last month's rate) as another `baselines_oof` column | ranked #1: baseline ladder and gap closed (Park et al. arXiv:2411.10109 normalisation; Twin-2K-500 gap ~56 %) |
| 5 | `stack`: logistic regression on logits of `cell_rate` and the LLM, fit on train | nothing; it is the "ours" named before the truth | ranked #2 (calibrate the LLM's "yes" bias; `sim.batch_calibrate` 69.4 % → 76.5 %) and #3 (LLM as a feature of a history model; honest gain is small: +0.006 in Twin-2K-500) |
| 6 | `se.seal` of the test predictions, `ours = stack` in the description | nothing; do it before you look at test outcomes | pre-registration; the card's selection-bias audit checks that `ours` was named in the seal |
| 7 | per-variant conversion, observed lift B−A with a visitor-bootstrap CI, predicted lift per arm, train-visitor lift, continue rate per step, the 3 cells where the LLM is furthest off | which two variants to compare | ranked #6: LLM effect sizes are often inflated (Automated Social Science, arXiv:2404.11794; Hewitt et al. r = .85 but with exaggerated effects), so never quote a predicted lift alone: put it next to the observed test lift (wide CI) and the train-visitor lift (LLM-free, printed on the same line) |
| 8 | `se.evidence.evaluate` → ladder, controls, paired CIs, audits, `card.md`, `dashboard.html` | `--B 1000` for the final card; pass `memorization=` when you have run the audit (see pitfalls) | ranked #1 again: the card is the deliverable |

Skipped on purpose (section "do not spend time on" of the literature ranking): reflection / memory streams, rich
biography personas, fine-tuning, many samples per person, embedding-similarity scoring (SSR −2.4).

## What the jury should see

1. **The variant decision with its uncertainty.** Offline run: "B converts 17.9 % vs A 8.8 % on 440 held-out
   visitors, lift +9.2 points [+2.8, +15.3]; past (train) visitors show +2.9; the simulation predicts +4.3 (raw LLM),
   +2.1 (stack), +2.0 (cell rate)." The synthetic ground truth is +2.7, so the test lift is the lucky one here: a few
   hundred visitors per variant give a CI ten points wide. Say "B wins" with the CI, and do not sell the predicted
   lift as accurate or as under-called from one test split.
2. **Where visitors drop off**, per step and variant, observed next to the LLM's estimate, and the three cells where the
   LLM is furthest off. Those cells are the finding: in the offline run one of them is A's landing page for
   `mobile_new` (0.25 observed vs 0.44 predicted), the autoplay video the fake LLM does not know about; go and look at
   that page.
3. **The baseline ladder** from the card, all with 95 % CIs: random 0.500, marginal (per-page continue rate) 0.576,
   step rate 0.576, LLM 0.626, cell rate 0.649, stack 0.649, segment mean (page x segment, shrunk) 0.653. `stack` does
   not beat the segment mean (−0.004 [−0.011, +0.002]), so the badge is "inconclusive" and the slide says so: the
   fake LLM's stack weight is about 0, past traffic already knows what it knows. A tie with the strongest baseline is
   a result, not a failure.
4. **The controls.** `llm_nopersona` (same pages, "a typical visitor") 0.481: the page text alone ranks nothing.
   `shuffled_stack` 0.556 permutes visitors within each page (the frame's `item` = variant x step), so it keeps the
   page and removes only who the visitor is; the gap to the raw LLM (0.626) is what the persona adds. Without the
   `item` column the shuffle permutes across all rows and lands at 0.5 for free, which proves nothing.
5. **The seal**: the hash and time of the sealed test predictions, earlier than the evaluation.
6. **Spend**: with `--live`, the last line gives this run's cost and the ledger total against `--max-usd`.

## Pitfalls

- **Leakage.** The persona must describe the segment, not its outcome ("has bought before" is history and allowed
  only if it is true before the test dates; "usually abandons at checkout" is the label). `page_text` must not name
  the winning variant. `cell_rate` and the stack are fit on the train split only; with a `date` column the last-arriving
  visitors are held out, otherwise a fixed 30 % of visitors. Do not tune `SYSTEM` while looking at test outcomes: tune on train,
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
