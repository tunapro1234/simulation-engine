# Recipe: new product adoption forecast

`examples/recipes/adoption.py` answers two questions: what share of a market will have adopted the product N periods
from now, and roughly what more (or less) marketing spend would change. You get a backtested forecast with an
evidence card, plus marketing what-ifs that are labelled as model output.

```bash
python3 examples/recipes/adoption.py                                   # offline: FakeLLM + 10 synthetic markets, ~15 s
python3 examples/recipes/adoption.py --data data/adoption.csv --target m07 --out runs/adoption --B 1000 --bank 2000
python3 examples/recipes/adoption.py --live --max-usd 0.5              # Claude Haiku, key from $ANTHROPIC_API_KEY
python3 examples/recipes/adoption.py --live --key-file <path> --max-usd 0.5
```

## When to use it

- The brief asks for a market-level adoption or diffusion forecast (a new app, payment method, EV charger,
  subscription) and you have cumulative adoption curves for several comparable markets: countries, cities, stores,
  or earlier launches of a similar product.
- The jury wants "what if we spend more on marketing?" and you can only answer that honestly as a model-implied
  scenario.
- **Not** for person-level "will this customer buy?" questions. Those are binary predictions; use
  `examples/recipes/site_visitors.py` or `sim.prediction`. The curve methods also need at least 3 points up to the
  cutoff and a series that has crossed 1 %. A product with no history at all only gets the LLM prior, and the card
  will rightly show that as unproven.

## Inputs

`--data` CSV, one row per (market, period):

| column | type | meaning |
|---|---|---|
| `series` | str | market / country / product id. One cumulative curve each; the bootstrap cluster |
| `t` | float | period: years or quarters, equal spacing of 1. Aggregate monthly data to quarters first |
| `y` | float | cumulative adopters as **% of the population (0-100)**. This is the forecast target |
| `description` | str, optional | market + product text for the LLM (first value per series). No adoption numbers, brand names or launch years |
| `spend` | float, optional | marketing spend per period in one currency. Used only for a printed elasticity check |

Use series of one kind: one product across markets, or close substitutes. Without `description` the LLM arm gets
"an unnamed market" and works as a pure generic prior. Without `spend` the elasticity check is skipped.

Knobs at the top of the script (step 0): `PRODUCT`, `HORIZON` (at most 10, the scenario window), `CUTOFFS` (periods
after the 1 % crossing at which the backtest cuts), `SPEND_WHATIFS`, `SPEND_ELASTICITY`, `OURS`, `COMPARE` and the
`SYSTEM` prompt.

## Steps (as numbered in the script) and the result behind each one

| # | step | what to change on the day | backed by |
|---|---|---|---|
| 1 | Load the CSV or the synthetic markets; check that `y` is a cumulative % | column names, units | (data hygiene) |
| 2 | Backend: FakeLLM offline; `--live` uses Claude Haiku with `budget_usd=--max-usd`. Before each call the cost ledger (`<out>/.cache/ledger.jsonl`, cumulative over every run into the same `--out`) raises `BudgetExceeded` once its total has reached the cap, so calls already in flight (up to 4 in parallel) can overshoot it by a few cents; set the cap with that slack. A refused, empty or unparsable reply falls back to a default prior and is counted on stdout (`unusable replies`); more than one or two means the prompt or `max_tokens` needs fixing, not that the LLM arm lost. The key is read from `--key-file` or `$ANTHROPIC_API_KEY` and never printed | `--model` only if the card says the model matters | Bigger models are on the "not worth the time" list: Park about 1 point; our Sonnet minus Haiku +1.0, not significant (lit. §4, "not worth it") |
| 3 | One LLM call per market asks for Bass (p, q, m) from the description, plus one **generic-market control** call | `PRODUCT`, the descriptions, `SYSTEM` | Hybrid #3: the LLM as a prior for a history model, not as the forecaster (Schoenegger human + machine averaging; the gains from learning from data in Centaur and SubPOP). The generic arm is the no-persona style control |
| 4 | Backtest: every method sees only `t <= cutoff` for all series and forecasts the next `HORIZON` periods. Curve baselines (naive, linear, Gompertz, Bass NLS), Bass + analogy prior, ABM + ABC + analogy prior, the Bass-simulator ABC control, and the LLM arms (`llm_only` = raw, `llm_generic` = generic-text control, `llm_bass` = LLM shape + fitted start and ceiling). `blend` = mean(abm_analogy, llm_bass); `blend_generic` = mean(abm_analogy, llm_generic) is its no-persona twin | `OURS` before the run; `--bank 2000` for the final card | Evidence card #1 (Park normalisation, Twin gap): a baseline ladder. Measured in this engine (OWID internet, test, 7 seeds, MAE pp): ABC + analogy ABM 8.48, Bass-ABC control 9.73, linear 10.72, Gompertz 11.61, Bass NLS 14.86, naive 18.39. Most of the gain came from ABC + the analogy prior, not from the ABM structure. On mixed US household technologies the analogy prior hurt (10.07 vs 8.54). Analogy priors in Bass models are classic marketing practice (Sultan, Farley and Lehmann 1990 meta-analysis) |
| 5 | Seal the forecasts before scoring, with `OURS` and `COMPARE` in the seal text | nothing: run it before you look at the MAEs | Pre-registration; the card's selection-bias audit checks that `ours` was named in advance (evidence card #1) |
| 6 | Marketing what-ifs on the target market: spend x k becomes innovation p x k^`SPEND_ELASTICITY` in the fitted ABM (`scenario`, common random numbers). The adopter-type shares come from `personas()`. A cross-market elasticity is printed as a check only | `SPEND_ELASTICITY` from a real campaign test if you have one; `SPEND_WHATIFS`; `--target` | Shrink effect sizes #6 (Hewitt; Manning, simulated effects 13.2x too large; Park v3): do not take an elasticity from the LLM or a confounded cross-section. Marketing variables enter the hazard as in the Generalized Bass Model (Bass, Krishnan and Jain 1994) |
| 7 | Evidence card: ladder with paired cluster-bootstrap CIs (cluster = series), registered comparisons, audits, the seal chain; writes `card.md` and `dashboard.html` | `--B 1000` for the final card | Evidence card #1 |

LLM agent societies for diffusion are on the "not worth the time" list (AgentTorch, HuMob, and our own tweet and
YJMob measurements). That is why the simulator here is a rule-based ABM calibrated with ABC, and the LLM only
supplies a prior.

## What the jury should see

1. **The ladder:** naive → linear → Gompertz → Bass → Bass + analogy → ABM + analogy → LLM arms → ours, each with
   a 95 % CI over markets. Ours is reported against the **strongest** baseline, not the weakest.
2. **The controls:** `blend − blend_generic` (does the market description carry information? Both sides share
   the ABM half, so only the text differs) and
   `abm_analogy − bass_abc` (does the network ABM add anything beyond calibration plus prior?).
3. **The seal:** the forecasts were hashed before scoring, and `ours` was named in the seal.
4. **The what-ifs** with 80 % bands, stated as "model-implied under an assumed elasticity of 0.5", never as
   "spend 2x and get +6 pp".
5. The offline synthetic run says this about itself: `llm_bass` 3.55 and Gompertz 4.84 beat `blend` 5.02 (MAE pp),
   so the badge is **inconclusive** and the card says so. That is the point. The card does not let a pre-named
   method claim a win it did not earn. Do not swap `OURS` after looking. If you change it, re-seal and say so on
   the card. The text control reads `blend − blend_generic` −2.27 [−3.98, −0.54]: the description helps here.
   Do not quote any offline LLM number as evidence. `fake_analyst` is built to half-know the generator (it reads
   the same young / metro / budget words that set q and p in `synthetic()`), so the LLM arms win offline by
   construction; only a `--live` run on your data says whether a real model's prior helps.

## Pitfalls

- **Leakage.**
  - A curve fitted on the whole series and then "forecast" inside it.
  - An analogy prior that includes the target market.
  - A `description` that names the brand or the launch year, or contains adoption figures.

  The backtest only lets methods see `t <= cutoff`, and the analogy prior excludes the target series. The text you
  write yourself is the remaining leak. The card's automatic "random" and "median" rungs are cross-fitted from the
  test window and marked look-ahead; they are not legitimate baselines.
- **Imbalanced accuracy.** A pooled MAE is dominated by long or late series and by easy short horizons. At h = 1 a
  naive carry-forward is hard to beat. Early in a curve, S-curve fits rarely beat a straight trend (engine
  measurement: OWID dev, linear 12.75 vs logistic 16.96). Read the per-horizon MAE in `frame.groupby("horizon")`.
  Keep one row per (series, cutoff, horizon) and let the cluster bootstrap resample series. If one market holds most
  rows, pass `balance_col="series"` to `evaluate`. For a person-level binary version, accuracy on an imbalanced
  outcome is meaningless: use AUC or macro-F1 and the "always no" rung.
- **Memorisation.** For famous products and countries (internet, mobile phones, EVs in Norway), the LLM may simply
  remember the curve, so a "prior" is really a lookup (memorisation probe #7, Hewitt). Probe with
  `se.evidence.ezber.recall_probe` on `--live`, anonymise the descriptions, and report the probe on the card
  (`memorization=`). Without it the card flags "not proven: leakage/memorisation", and that flag is correct.
- **Elasticity from cross-sections.** Markets with bigger budgets differ in other ways too. The printed cross-market
  elasticity is confounded; in the synthetic run it reads 1.70 against a true 0.5. Use a measured value or keep the
  assumption explicit.
- **Units.** `y` must be cumulative and in percent. Per-period sales or shares in 0-1 silently break both the 1 %
  alignment and the ceiling bounds.
- **Small ABC banks.** `--bank 300` is for smoke runs. The measured default is 2,000 sims x 2,000 agents. Smaller
  banks were not measured, and in the synthetic run `abm_analogy` is weak at 300.
- **Registering against a weak baseline.** `llm_bass − bass` is an ablation (does the LLM prior help Bass?). The
  card warns that `bass` is not the strongest baseline, so register the headline comparison against the strongest
  one as well.

Literature: the hackathon literature survey (Park / Horton / Aaru notes), §4, ranked by expected gain per hour. Engine measurements: `simengine/abm/adoption.py` and `simengine/abm/curves.py` docstrings, and
`docs/RESULTS.md`.
