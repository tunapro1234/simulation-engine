# Recipe: cold start (rating or early demand of a brand-new item)

Script: [`examples/recipes/cold_start.py`](../../examples/recipes/cold_start.py) (about 220 lines, numbered steps
0-8). Test: `tests/test_recipe_cold_start.py`.

```bash
python3 examples/recipes/cold_start.py                                   # offline: FakeLLM + synthetic catalogue, ~3 s
python3 examples/recipes/cold_start.py --data data/items.csv --out runs/cold
python3 examples/recipes/cold_start.py --data data/units.csv --metric wmape    # demand (units) instead of stars
python3 examples/recipes/cold_start.py --live --max-usd 1 --data data/items.csv   # Claude Haiku, key from $ANTHROPIC_API_KEY
python3 examples/recipes/cold_start.py --live --key-file <path> --max-usd 1 --cutoff 2025-03-01
```

Outputs go to `--out` (default `runs/cold_start`, git-ignored): `card.md`, `dashboard.html`, the seal registry
`.onkayit/` and, with `--live`, the reply cache and cost ledger under `.cache/`.

## When to use

- The brief asks "how will this new product / film / menu item do?" for an item with **no ratings or sales yet**,
  and you have a catalogue of past items with their text and their outcome (mean rating, units in the first N weeks).
- The LLM reads the new item's own text and gives a prior; similar past items give a data baseline; a blend learned
  on past items decides how much the LLM is worth. The past items replay the launch: the newest 30 % play "new".
- Not for: items that already have some ratings (warm start: use the item's own mean, shrunk; or `MFBaseline`),
  per-member ratings of a new item when you know who will rate it (`methods.coldstart.item_holdout_features` +
  `GroupTransfer`, the D3 rehearsal setup), weekly demand of existing series (`task="forecast"`).

## Inputs

`--data` is a CSV with **one row per item**. Past items have a `target`; the launch items the brief asks about are
rows with an **empty** `target` (they get a prediction and an interval in step 7, and are never scored).

| column | type | meaning |
|---|---|---|
| `item_id` | str | item id; the bootstrap cluster |
| `title` | str | product name as a shopper sees it; hidden in the blind arm |
| `category` | str | category; the category-mean baseline and part of the kNN text |
| `description` | str | the item's own text: features, materials, claims. **No reviews, ratings, sales or awards in it** |
| `price` | float, optional | shown to the LLM only |
| `launch_date` | str, optional | ISO date; the latest 30 % of labelled launches are the test split (else 30 % of items by a hash of `item_id`); enables `--cutoff` |
| `target` | float | mean star rating (1-5) or units sold in the first N weeks (same N for every item); empty = launch item |

For demand, change `SYSTEM` and `KEY` in step 0 to ask for units (`"units"`) and pass `--metric wmape`. Without
`--data` the script builds 240 synthetic items in 4 categories (rating = category level + 3 traits from a list of
10 + hidden quality) plus two launch items; the fake LLM knows 4 of the traits, overstates them 1.5x and rates
everything about half a star too high.

## Steps (what each one does, what to change on the day, what backs it)

| step | in the script | change on the day | backed by |
|---|---|---|---|
| 0 | `SYSTEM`, `KEY`, `NEW_ITEMS` | the brief's question and scale; keep the JSON reply | Silicon crowd / Schoenegger et al. (arXiv:2402.19379): one plain question per item, no persona needed for an item-level prior |
| 1 | load, time split (newest 30 % = test), launch rows = `split == "new"` | a split the brief fixes; never a random split when launch dates exist | leakage guard of every result in `docs/RESULTS.md`; a time split is what "new item" means |
| 2 | backend | `--live --max-usd`; Haiku by default (`--model` to change) | literature ranking "do not spend time on": bigger model ~1 point (Park et al. arXiv:2411.10109; Sonnet − Haiku +1.0, n.s., talp73) |
| 3 | one call per item, named + blind (name hidden); a refused or failed call and a reply without a number count as unparsed (median prior); above 20 % unparsed the run stops instead of writing a card | nothing; the blind arm is the recognition control | ranked #7: memorisation probe. D3 rehearsal (new-film cold start): the LLM gain was title recognition, ours − LLM-free +0.050 on dev, +0.017 [−0.001, 0.035] on test |
| 4 | category mean and TF-IDF kNN over category + description, train rows cross-fitted (5 item folds) | `--k`; add the client's own rule as another `baselines_oof` column; `methods.baselines.TextKNNRegressor` is the library twin when scikit-learn is installed | ranked #1: baseline ladder. Similar-item data is the cold-start baseline the LLM has to beat |
| 5 | ridge `target ~ 1 + knn + cat_mean + llm` on train = `stack` (ours); the same with the blind LLM = `stack_blind`; the same without the LLM = `stack_free`, the LLM-free rung ours has to beat | nothing; `ours` is named before the truth | ranked #2: LLMs lean optimistic (silicon crowd: mean forecast 57 % vs 45 % resolved yes); the intercept and slope absorb it. ranked #3: LLM as a feature of the data model (`methods.hybrid`; honest gain small, +0.006 in Twin-2K-500). ranked #6: shrink the LLM's effects (Manning et al. 13.2x exaggeration; Hewitt et al. effects r = .85 but too large) |
| 6 | `se.seal` of the test predictions, `ours = stack` in the description | nothing; seal before you look at test outcomes | pre-registration; the selection-bias audit checks that `ours` is in the seal |
| 7 | launch items: kNN from all labelled items, stack refit on all labelled items, 90 % interval from test residual quantiles | nothing; this is the answer to the brief | the interval is empirical (test residuals), not a model's own confidence |
| 8 | `se.evidence.evaluate(task="continuous", primary=mae, baselines_oof=[cat_mean, knn, stack_free], blind_col=...)` → ladder, shuffle control, blind arm, paired CIs, `card.md`, `dashboard.html` | `--B 1000` for the final card; `--cutoff <model cutoff>` when launch dates span it | ranked #1 again: the card is the deliverable |

Skipped on purpose (the literature ranking's "do not spend time on"): personas of imagined shoppers (the target is an
item mean, not a person; persona variables explain < 10 % of variance, Hu & Collier), reflection / memory streams,
fine-tuning (Park 0.79, below the prompt), several samples per item.

## What the jury should see

1. **The ladder, LLM-free rungs first**, with item-bootstrap 95 % CIs (MAE, lower is better). Offline run (FakeLLM,
   72 test items): raw LLM 0.615, random 0.587, shuffled stack 0.465, overall mean 0.396, category mean 0.368,
   kNN 0.340, LLM-free blend 0.333, stack without names 0.307, stack 0.303 [0.251, 0.365].
2. **The raw LLM is the worst rung**, and why: it rates everything +0.5 star (4.09 vs 3.56 observed). After the blend
   its weight is 0.39 next to kNN 0.78. The LLM's own contribution is ours − LLM-free blend = +0.030 [+0.003, +0.058]
   (not ours − kNN, +0.037: part of that is blending kNN with the category mean, no LLM needed). The three LLM-free
   rungs are tied within their CIs and the worst case over them is [−0.001, +0.058], so the badge is
   "inconclusive". Say that sentence as it is.
3. **The controls**: the shuffle control (stack predictions permuted across items) collapses to 0.465, so the stack's
   accuracy is item-specific; the blind arm (name hidden) loses nothing (−0.004 [−0.016, +0.008]), so the gain is not
   title recognition. With real products, show this row first: in D3 it was the whole story.
4. **The launch answer**: each new item with its prediction, the 90 % interval and the raw LLM next to it (offline:
   Orbit Buds 3.61 [3.18, 4.44], raw LLM 4.12).
5. **The seal and the spend**: the hash and time of the sealed test predictions, and with `--live` the ledger total
   against `--max-usd`.

## Pitfalls

- **Leakage.** The description must be what was known at launch: no review snippets, "bestseller" badges, award
  lines or sales figures. The kNN and category mean of a train item never see that item (cross-fit), and idf weights
  come from train items only. With launch dates, use the time split: a random split lets next year's sequel teach
  this year's original. Do not tune `SYSTEM` on test outcomes.
- **Imbalanced accuracy (here: imbalanced error).** Most items sit near the catalogue mean, so the overall mean
  already scores well on MAE (0.396 offline) and a model that predicts "average" for everything looks fine. Read
  the ladder against the overall-mean and category-mean rungs, check the "distribution collapse (sd ratio)" audit
  (blended predictions are pulled to the middle and should not be used to pick the extreme winners without the
  interval), and for demand use WMAPE: plain MAE is dominated by the few best sellers.
- **Memorisation.** Famous items (films, flagship phones, bestselling books) are in the model's training data with
  their reception. The blind arm hides the name; if `stack − stack_blind` is clearly better (CI below zero on MAE)
  the card flags it and the gain is recognition, not prediction. Also hide brand and author names from the
  description if they identify the item, pass `--cutoff` with the model's training cutoff so the card compares items
  launched before vs after it (Hewitt et al. report post-cutoff studies separately: r = .90), and only claim a
  forecast for items launched after the cutoff.
- **Optimism and scale.** LLMs rate new products generously and compress the range. Never report the raw LLM number
  as a forecast; report the blend and its interval. For demand, ask for units in the same window as `target` and
  look at the bias line (`--metric wmape` adds it).
- **Small catalogues.** With fewer than ~100 labelled items the kNN and ridge are noisy and the badge will be
  "inconclusive"; keep `--k` below the smallest category size and say the CI out loud.
- **Unparsed replies.** A reply without a numeric `KEY` (quoted numbers and replies cut mid-reason still parse), a
  refusal or a failed call becomes the median prior and is counted in the first output line; above a few percent,
  fix `SYSTEM` before trusting the arms. Above 20 % the script stops and shows the first bad reply.
- **Budget.** `--live` refuses to start without `--max-usd`. Calls are 2 x items (named + blind). The ledger
  (`<out>/.cache/ledger.jsonl`) counts every run into that `--out`; once its total reaches the cap the backend
  raises `BudgetExceeded` before any new call (up to 4 calls already in flight still finish); finished replies stay in the cache,
  so the same command with a higher cap resumes. The key comes from the environment or `--key-file` and is never
  printed.

## Literature behind the ranking

Ranked list: the TALP literature note, section 4 (`#1` baseline ladder and gap closed, `#2` calibrate the yes /
optimism lean, `#3` LLM as a feature or prior of the data model, `#6` shrink effect sizes, `#7` memorisation probe
and post-cutoff data). Sources: Schoenegger et al., silicon crowd, arXiv:2402.19379; Hewitt et al.,
treatmenteffect.app; Manning et al., Automated Social Science, arXiv:2404.11794; Twin-2K-500, arXiv:2505.17479;
Park et al., arXiv:2411.10109. Measured in this repo: the D3 rehearsal row of `docs/RESULTS.md`.
