# simengine.abm

This module holds the agent-based models and the baselines they have to beat:

| component | what it is | status |
|---|---|---|
| `AdoptionABC`, `AdoptionABCAnalogy` | adoption-curve ABM on a small-world network, calibrated with rejection ABC, plus an optional analogy prior from other series | measured; best method on OWID internet |
| `BassABC`, `BassABCAnalogy` | the same ABC with a closed-form Bass simulator (structure control) | measured |
| `llm_ceiling_prior`, `ceiling_upper` | LLM saturation-level prior per series (blind / named / perturbed prompts) with placebo controls and a recognition check | infrastructure; FakeLLM tests only |
| `CurveBaseline`, `BassAnalogy`, `naive`, `linear_trend`, `damped_trend`, `logistic`, `gompertz`, `bass`, `curves.constant_growth` | adoption-curve baselines | measured (constant growth: rehearsal only) |
| `OpinionPanel`, `opinion_baselines`, `BoundedConfidenceABM` | opinion drift between two survey waves: persistence, mean shift and transition baselines vs a Deffuant-Weisbuch ABM | measured; the ABM loses |
| `ElasticityShrinkage`, `shuffle_elasticities`, `elicit_elasticities`, `PreEventMean` | price-elasticity prior (category, LLM or mix), ridge-shrunk toward the event's own history; LLM-swap placebo | measured; the gain comes from shrinkage |
| `MarketABM`, `market_sim` | repeat-purchase market ABM (WTP, reference price, loss aversion, stockpiling, promotion, social effect) with pooled calibration | measured; not the most accurate |
| `hawkes_mean_field`, `power_kernel`, `seismic_kernel`, `SeismicPredictor`, `shape_forecast` | self-exciting cascade predictors for share curves, plus the shape baseline | measured; Hawkes ties and SEISMIC loses to the shape baseline |
| `AgentSociety`, `ShareCascadeWorld`, `ChoiceWorld`, `World` | turn-based LLM agent society: a deterministic world, agents acting through the LLM protocol, a log for every turn | measured; no better than a single LLM score or a calibrated constant curve |
| `run_who_to_punish`, `run_repeated_pd`, `run_trust_interaction` | economic-game LLM societies (Ertan-Page-Putterman who-to-punish with voting, repeated prisoner's dilemma, trust game) returning structured per-answer rows | infrastructure; no measured result in this port |
| `Grid`, `Agents`, `random_lattice`, `random_order_step`, `synchronous_step`, `morans_i`, `moran_permutation`, `dissimilarity`, `same_type_share`, `gini` | lattice core for grid ABMs: torus / bounded grid, agent arrays, schedulers, seeded RNGs, spatial metrics ([abm/grid_core.md](abm/grid_core.md)) | infrastructure; null levels and timings measured by this port |
| `Schelling`, `tipping_curve`, `ZhangSwap`, `LLMMover`, `segregation_coefficient` | Schelling segregation (threshold rule), Zhang's swap game, LLM movers ([abm/schelling.md](abm/schelling.md)) | measured on synthetic lattices; reproduces the tipping shape and Gauvin's T_f, disagrees with Zhang's beta scale |
| `Sugarscape`, `SugarscapeConfig`, `LLMPolicy`, `lorenz`, `gini_trajectory` | Sugarscape (Epstein-Axtell ch. II and IV): growback, movement, replacement, spice and trade, LLM movement ([abm/sugarscape.md](abm/sugarscape.md)) | measured; Gini 0.24 -> 0.47-0.48, no real LLM run |
| `geobleu`, `dtw`, `user_scores`, `slot_mode`, `MarkovMobility`, `EPRMobility`, `mobility_baselines` | HuMob mobility metrics, no-model baselines and an EPR simulator on the YJMob100K grid ([abm/mobility.md](abm/mobility.md)) | measured; the per-user slot mode wins, EPR loses as a forecaster |
| `CellGrid`, `exposures`, `greedy_placement`, `reach_frequency`, `top_visited`, `placement.holdout` | billboard placement: greedy / CELF max coverage under k or a budget, online bound, temporal holdout ([abm/placement.md](abm/placement.md)) | measured; greedy beats top-k visited by 7-20 points of test reach from k = 10 |
| `SpatialSEIR`, `seir_ode`, `r0_index_case`, `r0_growth`, `GenerativeEpidemic` | spatial SIR / SEIR on a grid or on trajectories, its ODE control, R0 estimators, generative-agent isolation ([abm/epidemic.md](abm/epidemic.md)) | measured against the ODE; generative variant only with rule FakeLLMs |

**Provenance.** The components down to the economic-game societies are ported from the TALP prep research branch
`makro` (the games from `edsl-test`, see their section). The algorithms and defaults are unchanged. The grid and
spatial models (the last six rows) were written for this port from the cited papers and measured by it; see
[Grid and spatial models](#grid-and-spatial-models) and their unit docs. Shorthand used for the ported part:
- **[makro RAPOR]** = `uretim/makro/RAPOR.md`
- **[abm RAPOR]** = `uretim/makro/abm/RAPOR.md`
- **[M1 NOTLAR]** = `uretim/makro/abm/m1/NOTLAR.md`
- **[RECETE]** = `uretim/makro/RECETE.md`
- **[prova D7 NOTLAR]** = `uretim/prova/d7/NOTLAR.md` (rehearsal D7, fixed-origin adoption forecast; added with the calendar-origin backtest)
- **[prova D7 SONUC]** = `uretim/prova/d7/SONUC-MAIN.md` (the same rehearsal scored on the held-out 1998-2002 truth) and its sealed test card `uretim/prova/d7/teslim/simengine-test-kart.md`
- **[G10 rehearsal]** = recipe rehearsal G10 (country x technology adoption, fixed calendar origins 2010 and 2015; ladder scoring log), added with the ceiling prior
- **[yayilim24 SONUC]** = `uretim/makro/abm/yayilim24/SONUC.md` (pre-registration `makro/onkayit/yayilim-24s.md`, commit 06224f7; run 21-22 Oct, added by the integrator after this module was written)

Every number in the ported sections comes from those files; this port did not re-measure them. The numbers in
[Grid and spatial models](#grid-and-spatial-models) are this port's own runs.

**Port equivalence** *(this port)*: `tests/test_abm_equivalence.py` imports the original code when
`SIMENGINE_ABM_SOURCE` points at the `makro` branch. It checks that the adoption and market engines, the six curve
baselines and the four ABC methods (`abm_abc`, `abm_abc_analoji`, `abc_bass`, `abc_bass_analoji`) give identical
output on synthetic data. Without that variable the test is skipped. The NPE variant (`abm_npe`, which needs
sbi/torch) was **not ported**. It scored worse than ABC on the internet dev set: 12.30 vs 10.78 for plain ABC and
9.56 for the analogy variant **[M1 NOTLAR]**.

## Minimal usage

```python
import simengine.abm as abm

# --- adoption (long table: series, t, y in percent; Turkish `seri` is accepted)
m = abm.AdoptionABCAnalogy(seed=0, cache_dir=".cache")              # bank 2,000 sims x 2,000 agents (defaults)
fit = m.fit(history[history.t <= 2005], "Turkey", cutoff=2005)      # sees only t <= cutoff
m.predict(fit, horizons=range(1, 11))                               # y_pred, lo80, hi80
m.scenario(fit, {"imitation": 1.2}); m.personas(fit)                # model-implied, label them so
frame = abm.adoption_backtest(df, {"abm": abm.AdoptionABCAnalogy(), "bass_ctrl": abm.BassABCAnalogy(),
                                   "linear": abm.CurveBaseline("linear")})   # EvalFrame: truth, pred_<m>, group
# then: simengine.evidence.evaluate(frame, ours="abm", ...)
# fixed calendar origins (one cut year for every series); 1% alignment where it works, 0.1% / growth as fallback
frame = abm.adoption_backtest(df, {"abm": abm.AdoptionABCAnalogy(), "abm01": abm.AdoptionABCAnalogy(threshold=0.001),
                                   "growth": abm.CurveBaseline("growth", upper=200)},
                              calendar_origins=range(1991, 1997), horizon=5, on_ineligible="nan")
frame.attrs["n_ineligible"], frame.attrs["ineligible"]          # series / (series, cutoff, method) not aligned
frame["pred_abm_fb"] = frame["pred_abm"].fillna(frame["pred_abm01"]).fillna(frame["pred_growth"])

# --- opinion drift (one row per respondent, wave A / wave B columns per item, coded 1..k)
panel = abm.OpinionPanel(df, {"econ": ("econ_20", "econ_22", 5)}, group="segment", weight="w", homophily="party")
pred = abm.opinion_baselines(panel)                                 # persistence, mean_shift, transition
summary, cells = abm.score_opinion(pred, panel)                     # W1 / TVD per method
frame = abm.to_eval_frame(pred, panel)                              # distribution EvalFrame

# --- elasticity prior with shrinkage (events: series, t (<=0 seen, >=1 forecast), y, lp, promo, category, unit)
llm_e = abm.elicit_elasticities(llm, {"u1": "corn flakes 12oz; regular price $3.49", ...})
mix = abm.ElasticityShrinkage("mix", llm_elasticity=llm_e)          # (category + LLM)/2, ridge toward own history
placebo = abm.ElasticityShrinkage("mix", llm_elasticity=abm.shuffle_elasticities(llm_e, unit_category))
yhat = mix.predict(mix.fit(train_full, target_pre), query).y_pred

# --- LLM agent society (FakeLLM, simengine.llm.Claude or simengine.llm.ClaudeCode)
agents = [abm.SocietyAgent("P01", "19, college student, shares memes"), ...]
run = abm.AgentSociety(agents, abm.ShareCascadeWorld("post text"), llm, mode="batch", max_calls=50).run()
run.log, run.states, run.calls, run.invalid

# --- who-to-punish society (4 players per group, 2 votes x 3 periods); feed run["groups"][g] to games.wtp_scorecard
run = abm.run_who_to_punish(llm, personas=None, groups=1, seed=0)     # None = no-persona control
run["groups"][0]["votes"], run["periods"], run["punishments"], run["invalid"]

# --- grid and spatial models (all offline here; examples/abm_new_demo.py runs each one in about 2 s)
g = abm.Grid((50, 50), torus=True)                                   # Moore, radius 1
run = abm.Schelling(g, threshold=1 / 3).run(n_steps=100, seed=0)     # run.history: same_share, happy, moved, ...
sug = abm.Sugarscape(abm.SugarscapeConfig(n_agents=250), seed=0).run(500)   # sug.stats.gini by step
q = test[["uid", "d", "t"]]                                          # mobility frames: uid, d, t, x, y
preds = abm.mobility_baselines(train, q)                             # slot_mode, markov_*, epr, global_slot_mode
s = abm.user_scores(q.uid, q.d, q.t, abm.from_cell(preds["slot_mode"]), test[["x", "y"]].to_numpy())
ex = abm.exposures(traj, abm.CellGrid.fit(traj, size=2), days=range(10))   # traj: user, day, slot, x, y
p = abm.greedy_placement(ex.M, k=5)                                  # p.cells, p.value, p.bound
par = abm.EpiParams.from_r0(2.5, infectious_days=5, latent_days=3)
r = abm.SpatialSEIR(par, shape=(100, 100), n=10_000, radius=3).run(days=200, i0=10, reps=10, seed=1)
abm.ode_agreement(r, abm.seir_ode(par, n=10_000, i0=10, days=200))
```

## Adoption ABM + ABC + analogy prior

**What it does.** The engine (`adoption_sim`) simulates a batch of parameter sets on a small-world network. The
network is half ring neighbours and half random links. Agents carry log-normal innovation p_i and imitate with a
Granovetter threshold, under a mix of local and global influence. The market (m) opens over 15 years.
`simulation_bank` draws 2,000 parameter sets from the prior and simulates 60 years. It drops runs that cross 1%
later than year 28 and stores each curve on a 0.25-year grid aligned at its 1% crossing.

- **ABC.** The fit uses only data up to the cutoff. Distance is RMSE to the target's points, with recent points
  weighted up to 2x. The closest 1% of (simulation x shift) pairs are accepted. The forecast is the median and the
  80% band is the 10%/90% quantiles.
- **Analogy.** The prior is the union of the other series' ABC acceptance sets at the same cutoff. The target
  series is excluded from its own prior.
- **Defaults** (dev-selected, internet): accept 1%, recent weight 1, analogy accept 1% **[M1 NOTLAR]**.

**Measured results, OWID internet, MAE in pp, lower is better.**
- **Test, 7 seeds** **[makro RAPOR M1; M1 NOTLAR]**:
  - `AdoptionABCAnalogy` 8.48 +/- 0.16
  - Bass-simulator control (`BassABCAnalogy`) 9.73 +/- 0.12
  - linear trend 10.72, Gompertz 11.61, Bass NLS 14.86, naive 18.39
  - Paired ABM - Bass control: -1.25 +/- 0.21 per seed; seed-averaged series bootstrap -1.32 [-1.83, -0.82]; the ABM is better on 58 of
    81 series.
- **Where the gain comes from.** Compared with Bass NLS, about 81-85% of the gain comes from ABC plus the analogy
  prior and about 15-19% from the ABM structure; dev and test agree. Caveat: the two simulators' prior ranges
  differ, so the "structure" share also contains that difference.
- **Cheap cross-series baselines do not come close** **[makro RAPOR K2-10]**: pooled LightGBM 12.95, kNN analogy
  14.81 (test).

**When it does not work.**
- **Short horizon.** At h=1, Bass, Gompertz and the linear trend are better (test h=1: ABM 2.64, Gompertz 2.13,
  linear 2.20) **[M1 NOTLAR]**. The gain is at long horizons (h=10: 13.65 vs linear 20.28).
- **Mixed technologies.** On US household technologies (19 dev series) plain ABC gets 8.54 +/- 0.07 vs linear
  8.70, which is not significant. The analogy prior **hurts** there (10.07), because unrelated technologies share
  no curve shape **[M1 NOTLAR]**. Use the analogy only for series of one kind, such as countries x one technology,
  or pass `group_col=` so a mixed panel takes donors of the target's kind only (not measured).
- **Bands.** Plain ABC bands are too narrow (80% coverage 0.54 internet, 0.64 US). The analogy variant is
  calibrated (0.77-0.80) **[M1 NOTLAR]**.
- **Cost.** A cold bank takes 53-85 s at default size on a loaded 4-core server; warm fits take seconds
  **[M1 NOTLAR]**. Smaller banks (`n_sim=300, n_agents=300`, as in the tests) were **not measured**.
- **`personas()` and `scenario()`.** These are model-implied attributions and counterfactuals, not observed
  quantities. Label them so.
- **Series below the alignment level.** A series must have reached `threshold` (a fraction, default 0.01 = 1%) by
  the cutoff; otherwise `fit` raises `IneligibleSeries` (a `ValueError`). With fixed calendar origins this can be
  most of the panel: in rehearsal D7 (country x technology, origin 1997) 198 of 361 series were below 1%.
  `threshold=0.001` aligns the bank at the 0.1% crossing instead (one extra bank per level, cached separately).
  It is **worse where 1% works** *(this port, synthetic check)*: on 25 synthetic Bass series (5% noise) that had
  observed both crossings, cutoff t0(1%) + 4, h = 1..5, aligning at 0.1% raised MAE from 7.73 to 9.01 (ABM, bank
  600 x 2,000; paired diff +1.28 [+0.57, +2.04]) and from 3.83 to 4.23 (Bass control), and the ABM's 80% band
  coverage fell from 0.50 to 0.26. A 10,000-agent bank gave the same picture, so it is not agent granularity. Use
  0.1% only as the fallback for the series 1% cannot align (`on_ineligible="nan"`, then fill, as in the usage
  above), and report how many rows the fallback covers. `threshold` is a fraction: 0.1 means 10% and warns.
  `eligible(history, series, cutoff)` tells you in advance.
- **Series far past the alignment level.** Bank curves end 32 years (`R_MAX`) after the crossing. A cutoff or
  forecast beyond that (calendar origins on a technology that crossed 1% decades earlier) raises `IneligibleSeries`
  too; before this check it returned a flat value at the grid end. Use the curve baselines for those series.

**Backtests** *(this port)*. `adoption_backtest` cuts at t0(s) + k by default (relative cutoffs, research task
M1). `calendar_origins=[1991, ..., 1996]` uses the same fixed calendar years for every series instead (all
series by default; concatenate two calls for both kinds); `cutoff_rel` is then only a label. `on_ineligible`
decides what happens when a method cannot align a (series, origin) block: `"raise"` (default, the old behaviour),
`"skip"` (drop the whole block, so every method is scored on the same rows) or `"nan"` (keep the rows, NaN for that
method, fill with your fallback). Either way `frame.attrs` lists `ineligible` ({series, cutoff, method}, every
method that cannot align the block), `ineligible_series` and `n_ineligible` (distinct series, any method).
"skip" scores the methods only on series the ABC can align, which is the easier, already-adopting part of the panel:
say so next to the numbers, or use "nan" with a fallback to keep every series.
`adoption.fit_many(method, history, cutoff, on_ineligible="skip")` does the same for one cutoff ("skip" or
"raise") and returns `models`, `ineligible`, `n_ineligible`. `fit_many`, `IneligibleSeries` and
`FitMany` live in `simengine.abm.adoption` (not re-exported from `simengine.abm` yet).

## Adoption: LLM ceiling prior, scale, donor groups, scenario anchor

*(this port; the API below is tested with FakeLLM and synthetic series only, no real-model measurement)*

```python
prior = abm.llm_ceiling_prior(llm, hist, groups="kind", perturb=0.15, parallel=8, seed=0)  # hist: t <= origin
prior[["ceiling", "ceiling_blind", "ceiling_perturbed", "ceiling_shuffled", "ceiling_group_median"]]
print(prior.attrs["leak_check"]["message"])
upper = abm.ceiling_upper(prior, col="ceiling_blind")   # blind ceilings: the default "ceiling" is the named (recalled) one
abm.AdoptionABCAnalogy(scale={s: c / 100 for s, c in upper.items()}, group_col="kind")
DirectForecaster("hgb", horizons=4, target="logit", upper=upper)   # per-series ceilings
sc = model.scenario(fit, {"imitation": 1.2}, anchor=best_forecast)   # y_anchor = best + ABM delta
```

- **`llm_ceiling_prior(llm, series_hist, *, blind=True, named=True, perturb=0.15, groups=None, parallel=8,
  seed=0)`.** One JSON answer per series and prompt, in parallel (`parallel_map`, `jsonx.complete_json`). The
  blind prompt shows only the history (years relative to the last one, values, unit); the named prompt adds the
  series name; the perturbed prompt multiplies the history by u ~ U(1 - p, 1 + p) and divides the answer back.
  `ceiling` is the named answer (blind when `named=False`). Unparseable answers are NaN and counted in
  `leak_check["n_failed"]`.
- **Placebo controls, no extra calls.** `ceiling_shuffled` permutes the named ceilings within `groups` (no series
  keeps its own), `ceiling_group_median` gives every series the median of the other members of its group
  (leave-one-out: the plain median holds the series' own value). A series alone in its group gets NaN in both.
  Run the downstream method with all three; a gain that survives the shuffle is not series knowledge (the
  elasticity prior's gain was matched by its LLM-swap placebo, see below).
- **Leak check** (`attrs["leak_check"]`): the named - blind gap (median relative), the Spearman of perturbed vs
  blind and the scale elasticity (slope of log(raw perturbed / blind) on log u: 1 = the answer follows the numbers,
  0 = it ignores them). Elasticity under 0.5 or Spearman under 0.8 (>= 5 series) sets `recognition=True` and the
  message says the blind prompt looks like recognition. A unit with a natural cap (percent of people, answer
  "100") also gives elasticity ~0; read the flag on series far from such a cap. With `future=` (post-origin
  truth, dev only) it adds the Spearman of each ceiling's headroom with the observed post-origin change, and says
  so when the named one is more than 0.2 above the blind one. Thresholds are this port's choice, not calibrated on a real model.
- **Upper.** `ceiling_upper(prior)` returns {series: value}: NaN falls back to the group median, then the observed
  maximum, and every value is at least the observed maximum. Pass the dict to `DirectForecaster(upper=)` (dict or
  panel column, see methods.md); `CurveBaseline(upper=)` takes one float, so loop per series there.
- **`scale=`** on every ABC method (a number or {series: number}): y is divided by it inside the fit and
  predictions, bands and scenarios are multiplied back, so series above 100 (subscriptions per 100 people) need
  no manual divide or clip. The bank's m <= 1 becomes a ceiling of 100 x scale, and `threshold` aligns at
  `threshold` x 100 x scale data units. Without it the ABM's forecast cannot pass 100 (test: series saturating at
  150-190). `scale=None` / `1.0` gives the old output byte for byte.
- **`group_col=`** on `AdoptionABCAnalogy` / `BassABCAnalogy`: analogy donors only from series with the target's
  value in that column (one kind), fewer than 3 donors falls back to the flat prior as before. This is the fix for
  the mixed-technologies failure above; `BassAnalogy` (curves) does not take it yet.
- **`scenario(..., delta=False, anchor=None)`.** `delta=True` adds the paired intervention - baseline difference
  (`delta`, `delta_lo80`, `delta_hi80`; 0 on baseline rows). `anchor=` (a frame with t and y_pred, a {t: value}
  dict or a Series; a frame with a series column is cut to the model's series) implies it and adds
  `y_anchor = anchor + delta`, so the intervention is read as a change on your best forecast, not on the ABM's
  own level. `attrs["origin_fit_error"]` is the baseline's miss at the last
  observed year; report it next to any scenario.
- **The M1 result did not reproduce at a calendar origin.** M1 measured ABM - Bass control = -1.25 MAE at relative
  cutoffs (t0 + k). In rehearsal G10 at fixed calendar origins the analogy ABM was worse than its Bass control:
  +0.28 at 2010 (11.01 vs 10.73) and +1.54 at 2015 (13.32 vs 11.78) **[G10 rehearsal]**. Do not cite -1.25 for a
  calendar-origin task; backtest at the task's own origin.

## Adoption baselines

`curves.py` holds naive, linear trend (last 5 points), damped trend (phi 0.9 fixed), logistic, Gompertz and Bass
NLS. The S-curve fits use multi-start least squares with m bounded below by the observed max. It also holds
`BassAnalogy`, which uses the median (log p, log q) of the other series as a penalty term.

- **`upper=`** *(this port)*. Every curve clips to [0, upper] and bounds m above by it; the default 100 keeps the
  research behaviour. Raise it for quantities that pass 100 (mobile subscriptions per 100 people), or pass `None`
  for no bound. `CurveBaseline(name, **kw)` and `BassAnalogy(upper=)` pass it through.
- **`constant_growth`** (`CurveBaseline("growth")`) *(this port)*. The last observed log growth rate (over the last
  `last=3` years) is damped by phi^i (default `phi=0.8`, the rehearsal setting) or held (`phi=1`); `offset=0.1` pp
  keeps zeros finite (this port's choice). In rehearsal D7 (361 country x technology series, fixed origins
  1991-1996, h=1..5, per 100 people) a hand-written version (last 3 years, phi 0.8) was the strongest simple
  baseline on dev: 1.731 MAE vs Gompertz 1.783, linear 1.885,
  Bass 1.950, naive 2.610 **[prova D7 NOTLAR]**. **It did not hold on the test origin:** at 1997 -> 1998-2002 the
  same rule scored 4.571 MAE, behind the ABM arm with fallback (4.435) and the organiser's simple linear rule
  (4.271), and some of its forecasts hit that run's 150 clip **[prova D7 SONUC]**. The dev origins (1991-1996)
  had low levels; from a higher level, 5 years of held log growth overshoots even with phi 0.8. The library
  version was not re-measured. Keep `upper` at the real ceiling, put `linear` and `gompertz` next to it, and
  check the test-origin forecast range before trusting a dev win.

Measured on the internet dev set (122 series) **[makro RAPOR M1]**: naive 20.42, linear 12.75, damped 13.69,
logistic 16.96, Gompertz 12.39, Bass 15.97, Bass + analogy 15.94. On US households: linear 8.70 is the best
baseline. Early in the curve (cutoff t0+3..8), S-curve fits rarely beat a straight trend **[RECETE]**. Always put
`linear`, `gompertz` and `growth` next to any adoption claim.

## Opinion drift: bounded-confidence ABM vs persistence

**What it does.**
- **Baselines.** `opinion_baselines` gives three: persistence (wave A unchanged), mean shift (the group's mean
  shift on the other items, leave one item out) and a transition matrix (estimated on the other items with the
  same scale).
- **ABM.** In `BoundedConfidenceABM`, each respondent is an agent. Its latent position comes from its wave-A
  category. For 24 steps it meets a weighted partner, drawn from its own homophily class with probability h. It
  moves mu toward the partner when they differ by less than eps. A group shock kappa x delta and noise sigma are
  added.
- **Calibration.** The 480-cell research grid, scored by W1 on the other items (leave one item out). The mu=0
  control comes out as `abm_shock_noise`.

**Measured** **[abm RAPOR C2]**: CES 2020 to 2022 / 2024, 4 segments x 7 items, 3 seeds, W1.

| method | 2022 | 2024 |
|---|---|---|
| mean shift | 0.111 | 0.131 |
| persistence | 0.117 | 0.127 |
| ABM mu=0 | 0.126 | 0.129 |
| bounded-confidence ABM | 0.133 | 0.145 |

**The ABM loses to both baselines.** ABM - mean shift in 2022 is +0.023 [+0.012, +0.036]. Interaction fits the
training items but does not carry over to the held-out item. Cost: 317 s per seed on 4 cores.

**When to use it.** Use persistence and mean shift as the baselines for aggregate drift. The ABM is worth testing
only when real network or interaction data exist (who sees whom). The LLM panel on the same task also lost to
persistence (0.152 / 0.157) **[makro RAPOR C2]**.

## Elasticity prior with shrinkage

**What it does.**
- **Model.** Per event: log1p(y) = a_e + b_e lp + g_e promo.
- **Priors.** `category` is within-event-demeaned OLS per category, the log-log FE baseline. `llm` is a per-unit
  LLM elasticity. `llm_deviation` is the category level plus the LLM's within-category deviation. `mix` is
  (category + LLM)/2.
- **Shrinkage.** A 2x2 ridge toward the prior on the event's own t <= 0 window. lambda comes from the grid
  {0.3 ... 100, 1e9}, chosen by inner validation on training events (fit on t <= 0, forecast 1..H).
- **Back-transform.** Duan smearing.
- **Elicitation.** `elicit_elasticities` sends the research "expert" prompt (anonymised, 20 items per call) and
  drops unparseable items.

**Measured** **[makro RAPOR M2]**: Dominick's, WAPE over 8 weeks.

| method | dev | test |
|---|---|---|
| pre-event 26-week mean | 0.598 | 0.616 |
| category log-log | 0.466 | 0.479 |
| category + own history | 0.466 | 0.477 |
| LLM expert + data | 0.469 | 0.469 |
| (category + LLM)/2 + data | 0.465 | 0.470 |
| LLM archetype + data | 0.479 | 0.482 |

**The Dominick's finding.** The mix gains a little on test: -0.0072 [-0.011, -0.003]. The LLM-swap placebo
(`shuffle_elasticities`, LLM values permuted across units within category) gains the same: -0.0085 / -0.0066 /
-0.0039, and every variant is zero on dev. **The gain comes from shrinkage, not from LLM knowledge.** Do not write
"the LLM prior helps" unless the real prior beats its placebo with a CI that excludes zero. In C3 strategy
ranking, the LLM prior added nothing (Spearman 0.464 vs a fixed theory order of 0.399 on dev)
**[makro RAPOR C3]**.

## Market ABM

`MarketABM` calibrates one parameter set per category by WAPE. It uses optuna TPE when optuna is installed and a
seeded random search otherwise. It then simulates the planned price and promotion path and scales the level to
the event's own pre-period.

**Measured.**
- **M2, dev** **[abm RAPOR]**: WAPE 0.465 with 120 TPE trials and 0.456 with 300. Category log-log gets 0.434,
  GBDT 0.373 +/- 0.0009.
- **C3 strategy ranking, test** **[makro RAPOR C3]**: the ABM alone gets 0.612 +/- 0.102. It is seed sensitive.
- **ABM response as a GBDT feature**: 0.812. A plain log-log feature gives 0.807, a difference of +0.005
  [-0.020, +0.030]. In the temporal test no hybrid beats GBDT (Holm p >= 0.91).

No separate contribution of the ABM structure was shown. Use it for mechanism and counterfactual stories, such
as stockpiling and the dip after a promotion, not for accuracy. Note that the research "price increase" label
was 58% promotion ends; type those separately **[abm RAPOR]**.

## Diffusion: Hawkes / SEISMIC cascade predictors (measured: no gain over the shape baseline)

**What it does.**
- **Recursion.** `hawkes_mean_field` runs the mean-field self-exciting recursion in 1-minute bins after an
  observation window [0, t]. The branching factor is estimated per cascade and capped (0.99 / n*).
- **Kernels.** `power_kernel(c, theta)` is chosen on training data with `select_power_kernel`.
  `seismic_kernel(s0=5, theta=0.242)` uses the SEISMIC defaults.
- **SEISMIC-style predictor.** `SeismicPredictor` weights events by follower counts and adds the research's
  linear correction of the log final size.
- **Shape baseline.** `shape_forecast` is the median training growth after t (Szabo-Huberman, "SH" below).

**Measured** **[yayilim24 SONUC]** (research code `makro/abm/kod/yayilim24.py`, the file this module was ported
from; this port was not re-measured). Metric: log1p RMSE of the 24-hour cumulative curve on 145 points, lower is
better, tree/cascade bootstrap CIs.

| test set | method | information | curve RMSE |
|---|---|---|---|
| Twitter15/16 (418 trees, temporal + event-disjoint) | B0 fixed median curve (train) | none | 0.920 [0.862, 0.980] |
| | SH60 shape baseline (`shape_forecast`) | first hour | 0.391 [0.342, 0.447] |
| | HK60 Hawkes, power kernel c=1, θ=0.5 | first hour | 0.391 [0.341, 0.449] |
| | G60 GBM on early features (**not ported**) | first hour | **0.289** [0.257, 0.323] |
| | SH360 / HK360 / G360 | first 6 hours | 0.130 / 0.114 / 0.113 |
| SEISMIC (39,263 cascades, day ≥ 12) | B0 | none | 0.620 |
| | SH60 | first hour | 0.267 |
| | SM60 SEISMIC-style (`SeismicPredictor`) | first hour | 0.308 |
| | G60 | first hour | **0.187** |

- HK60 − SH60 on Twitter: 0.000 [−0.006, +0.007]. The Hawkes recursion adds nothing over the shape baseline.
- SM60 − SH60 on SEISMIC: +0.041 [+0.038, +0.043]. The SEISMIC-style model is worse. On about 45% of cascades the
  p̂·n* ≥ 0.99 cap fired (n* = 1,298, a heavy-tailed follower mean); this adaptation is not a faithful SEISMIC
  reproduction.
- Early observation beats everything: the first hour takes the curve error from 0.77 (best text-only arm) to 0.29;
  six hours to 0.11. If a diffusion task comes up, the ready arm is a GBM on early features plus `shape_forecast`.
- A date-only shortcut (B3) beat text models (B3 − B1 −0.073 [−0.117, −0.028]), so every diffusion dataset needs a
  temporal split and a date baseline.

## Turn-based LLM agent society

**What it does.** `AgentSociety(agents, world, llm)` runs turns. On each turn:
1. Active agents observe the world state from the start of the turn.
2. They act through `llm.map`: one call per agent (`per_agent`), or one call for the whole society (`batch`, the
   research design, about 20x fewer calls).
3. Replies are parsed as JSON and validated by the world. Invalid replies are retried, then replaced by the
   world's default action and flagged `valid=False`.
4. `World.step` applies the actions in sorted-id order with an RNG seeded from (seed, turn).

The world state is deterministic. A deterministic LLM (FakeLLM, or a disk-cached `Claude` / `ClaudeCode`)
replays the same log exactly. `max_calls` and the backend's own budget guard both stop the loop cleanly, with
`stopped` saying why. The log has turn, agent, prompt id, attempts, valid, action and raw reply.

**Ready-made worlds.**
- **`ShareCascadeWorld`**: the research 20-persona retweet society. At each checkpoint, every agent who has not
  shared yet decides whether to share, seeing who already has. The research batch format `{"share": [ids]}` is
  accepted.
- **`ChoiceWorld`**: repeated choice with social feedback and per-turn context, for adoption, opinion or
  product choice.

For your own world, implement the `World` protocol: `active`, `describe`, `observe`, `parse`, `default_action`,
`step`, `state`, `n_turns`.

**Measured** **[yayilim24 SONUC]**: the research retweet society (20 Haiku personas × 3 checkpoint turns,
`ShareCascadeWorld` design, batch mode) predicting the 24-hour curve from the text alone, Twitter15/16 test, curve RMSE:
- society L 0.771 [0.717, 0.827]; a single Haiku score D 0.772; L − D −0.001 [−0.010, +0.009]. The pre-registered
  falsification condition "society adds nothing over one LLM score" **fired**.
- L beats the TF-IDF model B1 (−0.130 [−0.191, −0.068]), but a post-hoc check shows that almost all of it is level
  calibration: a constant median curve calibrated on the same dev period scores 0.772, and L − that curve is
  −0.001 [−0.010, +0.009]. Against a recalibrated B1 the society keeps −0.010 [−0.018, −0.001]; Haiku's rank signal is
  real but small (Spearman 0.16 [0.07, 0.26]).
- Sonnet 5.5 on a 150-tree subsample: 0.801 vs Haiku 0.785 on the same trees. The bigger model did not help.
- Test-retest of a second independent Haiku run (97 trees): final-size Spearman 0.78.
- Cost: 3,348 calls, $0.94 (Haiku $0.29 + Sonnet $0.63).

Related measured facts:
- In the C1 survey panels, five rounds of discussion barely moved the answer distribution. ΔW1 was -0.005
  [-0.012, +0.002], and 5-8% of answers changed **[makro RAPOR C1]**.
- A persona asked alone collapses to the most likely answer (spread 0.42, closes 0.41 of the gap vs 0.55 at
  tables) **[makro RAPOR K2-6]**.

So:
- Do not expect turns of interaction to fix a distribution.
- Count invalid replies (`run.invalid`). In one research run 8% of answers were invalid, and nobody checked
  whether those drops were selective.
- Prefer batch prompts, which cut the call count.
- Use `claude -p` (`ClaudeCode`) only for small societies, because every call carries a large context
  (docs/modules/llm.md).

## Economic-game societies: who-to-punish, repeated PD, trust

**Provenance.** Ported from the `edsl-test` repository (same owner as `replicant`):
`experiments/07_who_to_punish/who_to_punish.py`, `experiments/05_prisoners_dilemma/prisoners_dilemma.py`,
`experiments/03_trust/trust_interaction.py`, and only the None-retry rule of `experiments/common.py`. The EDSL calls
are rewritten onto the simengine `LLM` protocol: the persona goes in as `system` (re-sent on every call), the game
text as `user`, and every answer is a small JSON object. No code from the source repository is imported.

**Who-to-punish** (`run_who_to_punish(llm, personas, *, groups=1, seed=0, periods=3, votes=2)`). This is a
simplified version of the design in Ertan, Page & Putterman (2009, *European Economic Review* 53(5)). The constants
(`WTP_DESIGN`) are the paper's:
- 4 players and a $10 endowment.
- Each $1 in the fund pays $0.40 to every player.
- Paying $0.25 removes $1 from a target, at most $10 per target.
- Earnings are `max(0, 10 - c + 0.4 * total - 0.25 * given - received)`.

Before each block of `periods` periods, every player votes Yes / No / No preference on three ballot items: may a
player who contributed less than, exactly, or more than the group average be punished? An item passes only when
Yes > No. No preference does not count, and ties fail (footnote 9 of the working paper). Each period, players
contribute, the contributions are revealed (without names), and then each allowed (punisher, target) pair picks a
punishment.

The edsl-test simplification is kept: one fixed group of 4 per `groups` unit, fixed labels, and 2 votes × 3 periods.
The paper ran 40 groups, 3 or 5 votes, 30 periods, and reshuffled labels.

The output is a JSON-serialisable dict:
- **Per-answer rows** in `ballots`, `contributions` and `punishments`. Every row carries `agent_id`,
  `persona_method`, `seed`, `traits`, `max_tokens`, `wording_id`, `valid`, `attempts` and `raw`.
- **`rules`**: one row per (group, vote), with a tally per item.
- **`periods`**: one row per completed period, with contributions, given and received punishment and earnings per
  label.
- **`groups`**: a score-ready dict per group for `games.wtp_scorecard`. It holds `votes` (the punish-low outcome of
  each vote, the shape of edsl-test `score_wtp.py`), `rules`, `periods`, `punishments` (with `target_contribution`
  and `avg` for the targeting criterion), `dismantled`, `ever_punish_high`, `complete`, `stopped`,
  `n_periods_planned` and `n_periods_completed`.

**Missing answers are never $0.** edsl-test's early runs wrote a missing answer into the game history as $0. Its
commit 15b9531 fixed this by raising instead (see the caveat on [RESULTS.md](../RESULTS.md) §8 row 9; `ET/personas/value_anchor/NOTES.md`). Here, an unparseable,
fractional or out-of-range answer is re-asked `retries` times (default 2) with the parse error appended. Values are
never clamped. If the answer is still invalid, it stays `None` with `valid=False` and is counted in `invalid`. What
happens next depends on `on_invalid`:
- `"stop"` (default): that group ends at that stage. Its completed periods are kept, and `complete=False`.
- `"abstain"`: an invalid ballot is not counted. An invalid contribution makes the player sit out the period: they
  are left out of the fund, the average and the punishable set, and their earnings are `None`. An invalid
  punishment answer removes nothing.
- `"raise"`: raise `InvalidAnswer`.

Report `n_periods_completed / n_periods_planned` and the invalid counts next to any score. Dropout is not random
([RESULTS.md](../RESULTS.md) §8 row 3, replicant data).

**Cache and seeds.** With `sample_key=True` (default), `seed * 1_000_003 + group` is passed to the backend as
`sample`. That way a disk-cached `Claude` / `ClaudeCode` does not serve one group's or seed's answers to another.
edsl-test's baseline runs had an effective n of 1 for this reason ([RESULTS.md](../RESULTS.md) §8 row 7 caveat). `FakeLLM` ignores the
key.

**Settings that change behaviour.** `max_tokens` defaults to 1024 and is recorded on every row. Treat it as a
treatment: in an edsl-test single run, the bare model punished free riders at 8192 tokens but not at the default
cap ([RESULTS.md](../RESULTS.md) §8 row 10; that default-cap run was partly None-corrupted). `max_calls` and the backend's budget guard stop the run cleanly, and `stopped` says why.

**Repeated PD and trust** (optional extras):
- **`run_repeated_pd`**: fixed pairs play `rounds` rounds with the full history shown. The payoffs are C/C 3/3,
  D/D 1/1 and D/C 5/0 (the setup of Mei, Xie, Yuan & Jackson 2024 used in edsl-test). An invalid move ends that
  pair, and the round is not scored.
- **`run_trust_interaction`**: the Berg, Dickhaut & McCabe (1995) investment game with a $100 pot tripled. The
  investor and trustee are separate personas. A return above the received amount is invalid, not clamped.

**Measured.** Nothing in this port. The edsl-test who-to-punish index and the bare-model corner behaviour ([RESULTS.md](../RESULTS.md) §8
rows 7-9) came from EDSL runs with gpt-oss-120b, scored by regex over `summary.txt`. They have not been re-run
through this code. Tests (`tests/test_abm_society_games.py`) check only the mechanics, against hand calculations
with scripted `FakeLLM` replies: payoffs, majority with ties failing, dismantling, retries, and that invalid
answers are never counted as 0.

## Grid and spatial models

Six units written for this port, each with its own doc (API table, provenance checked against the source text,
measured numbers, limitations). The sections below are the short version. All names in the table at the top are
exported from `simengine.abm`. Names that would clash or read ambiguously at package level stay in the submodule:
`abm.grid.run` (step loop), `abm.sugarscape.gini` (clips negatives; `abm.gini` is the grid one, which raises on
them), `abm.placement.adoption` / `AdoptionRun` / `holdout` (`abm.adoption` is the adoption-ABM submodule),
`abm.mobility.clean`, `abm.spatial_sir.rule_responder`, and the module constants. Every unit was reviewed after it
was written; the reviews fixed bugs and re-ran the measurements, as noted in each doc.

`examples/abm_new_demo.py` runs all six on synthetic data in about 2 s. Its printed values are smoke checks, not
results.

### Lattice core (`abm.grid`)

Unit doc: [abm/grid_core.md](abm/grid_core.md).

`Grid(shape, torus, neighbourhood, radius)` gives vectorised neighbour sums and counts over `(..., H, W)` stacks,
flat neighbour tables and type counts. The lattice convention is an int array with `EMPTY` (-1) and types 0..K-1.
`random_order_step` (asynchronous, fresh order per step), `synchronous_step` (from a read-only copy) and
`grid.run` are the schedulers. Metrics: Moran's I with a permutation p-value, Duncan's dissimilarity on tiles, the
like-neighbour share and Gini.

- **Null levels** (random 45/45/10 lattice, 100 x 100 Moore torus, 20 seeds): like share 0.5002, Moran's I 0.0004
  (E[I] = -0.0001), D 0.083 on 10 x 10 tiles and 0.166 on 5 x 5. Report D with its tile size and this null.
- `moran_permutation` defaults to esda's legacy "directed" p-value, which runs small. Pass
  `alternative="greater"` or `"two-sided"` for a test decided in advance.
- Speed: 1000 x 1000 neighbour count 2.2-3.7 ms (load dependent), about 2-3.5x faster than
  `scipy.ndimage.correlate` at radius 1.

### Schelling, Zhang swap game, LLM movers (`abm.schelling`)

Unit doc: [abm/schelling.md](abm/schelling.md).

`Schelling(grid, threshold, rule, movers)` covers Schelling's 1971 rule (`rule="nearest"` on his 13 x 16 board),
the Mesa / NetLogo random move and the Gauvin et al. 2009 variant (`movers="all", rule="satisfying"`).
`ZhangSwap` is Zhang's 2004 all-integrationist swap game. `LLMMover` replaces or overrides the threshold with
batched, de-duplicated LLM decisions parsed by `parse_decision`.

- **Tipping curve** (50 x 50 torus, 10 seeds): like share 0.51 at threshold 0.1, 0.75 at 1/3, 0.87 at 0.5, 0.99 at
  0.7, and back to 0.55 at 0.8 where nobody can be satisfied. NetLogo's "about 70%" at 30% wanted: we get 0.75.
- **Gauvin et al.**: frozen-to-segregated jump between T = 1/3 and 3/8, matching their T_f = 3/8.
- **Zhang**: segregation emerges although every agent prefers mixing, but the onset is between beta 5 and 6, not
  near 2 as in his Fig. 8. The model as written is an Ising model whose literature critical point gives
  beta_c = 5.07; stripes melt at beta 4. The paper does not say how its beta or payoffs were scaled.
- **LLM path**: only checked with a rule-following FakeLLM (it reproduces the classic outcome; offering 3 random
  vacancies alone keeps a 0.7-demand town mixed). No real-LLM numbers.

### Sugarscape (`abm.sugarscape`)

Unit doc: [abm/sugarscape.md](abm/sugarscape.md).

Rules G, M, R, spice and T from Epstein & Axtell (1996), with movement and death checked against the NetLogo port.
The default capacity map is parametric (91.6% of cells equal to the transcribed book map, which is CC BY-NC-SA and
not shipped; `load_capacity` reads any map at runtime). `LLMPolicy` replaces rule M with one `llm.map` per step.

- **Wealth distribution** (250 agents, 10 seeds): Gini 0.236 at t = 0, 0.51 at the first age cohort (t = 100),
  0.475 on average over steps 200-500; skewness 0.03 -> 1.29. The transcribed map agrees within one seed sd.
- **Carrying capacity** 230 agents of 400; selection lowers metabolism (2.47 -> 1.79) and raises vision.
  Trade raises carrying capacity (132 -> 151) and inequality, and prices converge (log-price sd 0.59 -> 0.06).
- The book's own numbers were not checked (not open access). Scripted FakeLLM policies show that random movement
  barely changes the Gini plateau but more than doubles starvation, so an LLM-policy study should report survival too.

### Mobility (`abm.mobility`)

Unit doc: [abm/mobility.md](abm/mobility.md).

HuMob GEO-BLEU and DTW reimplemented from the organisers' reference code (both the 2023 and 2025 variants; max
difference on a tie-heavy probe 5.6e-17 / 2.3e-13), the per-user and global slot mode, a first-order Markov chain
and an EPR simulator (Song et al. 2010). The bench task `yjmob` (`simengine.bench.tasks.yjmob`) wraps the split.

- **2,000 YJMob100K users, days 0-59 -> 60-74** (humob23): slot mode 0.298 GEO-BLEU (0.316 on the HuMob test pool,
  level with a top-3 transformer's published 0.316 but not like for like), Markov 0.21-0.23, EPR 0.14,
  global slot mode 0.004.
- EPR reproduces the S(n) growth exponent (0.74-0.84 vs the paper's 0.83) but is not a forecaster; use it as a
  simulator. Its 16-cell return buffer is a documented deviation from the paper.

### Placement (`abm.placement`)

Unit doc: [abm/placement.md](abm/placement.md).

Max coverage (optionally frequency-capped) under k cells or a cost budget: greedy and lazy greedy (CELF) with the
Leskovec et al. 2007 guarantees and their online upper bound, the top-k-visited and random baselines, an
exposure-to-adoption step with word of mouth, and `placement.holdout` (place on train days, score on test days,
paired user-bootstrap CI).

- **YJMob100K, first 4,000 users, test-day reach share at k = 50**: lazy greedy 0.739 vs top-50 visited 0.597 on
  500 m cells (difference [+0.126, +0.157]); 0.923 vs 0.821 on 2 km cells. At k = 5 on 500 m cells it ties.
- Greedy on EPR- or Markov-simulated trajectories places about as well as on the real history, also without the
  test-day timing. One seed; no adoption ground truth, so the adoption step is illustrative only.

### Spatial SEIR and generative-agent epidemic (`abm.spatial_sir`)

Unit doc: [abm/epidemic.md](abm/epidemic.md).

`SpatialSEIR` runs a batch of replicates on a grid (Moore window of any radius, or full mixing) with static homes or
trajectories; `seir_ode` is the mean-field control; `r0_index_case`, `r0_growth` and `r0_final_size` estimate R0.
`GenerativeEpidemic` follows Williams et al. 2023: each agent is asked daily whether to stay home.

- **Well mixed vs ODE** (n = 10,000, 30 reps): attack rate and peak height within 0.004; the peak comes 1.7-2.7
  days later, a lag that shrinks with smaller steps and more initial cases.
- **Local contact** flattens and delays the epidemic long before it changes the final size (radius 3: attack 0.87,
  peak a third of the ODE's, twice as late). The growth-rate R0 reads low at every finite radius (1.25 at radius 2
  where the index case gives 2.24). At radius 1 an index-case R0 of 1.79 still gave no major outbreak.
- **YJMob100K trajectories** (5,000 users): co-location within 1.5-2.5 km gives R0 1.6-2.1 and smaller, slower
  epidemics; `beta` is not fitted to any disease.
- **Generative variant**: only rule FakeLLMs (base attack 0.937 vs final size 0.940 for R0 = 3). No real-LLM run.
