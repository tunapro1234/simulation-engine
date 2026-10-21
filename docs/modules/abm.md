# simengine.abm

This module holds the agent-based models and the baselines they have to beat:

| component | what it is | status |
|---|---|---|
| `AdoptionABC`, `AdoptionABCAnalogy` | adoption-curve ABM on a small-world network, calibrated with rejection ABC, plus an optional analogy prior from other series | measured; best method on OWID internet |
| `BassABC`, `BassABCAnalogy` | the same ABC with a closed-form Bass simulator (structure control) | measured |
| `CurveBaseline`, `BassAnalogy`, `naive`, `linear_trend`, `damped_trend`, `logistic`, `gompertz`, `bass` | adoption-curve baselines | measured |
| `OpinionPanel`, `opinion_baselines`, `BoundedConfidenceABM` | opinion drift between two survey waves: persistence, mean shift and transition baselines vs a Deffuant-Weisbuch ABM | measured; the ABM loses |
| `ElasticityShrinkage`, `shuffle_elasticities`, `elicit_elasticities`, `PreEventMean` | price-elasticity prior (category, LLM or mix), ridge-shrunk toward the event's own history; LLM-swap placebo | measured; the gain comes from shrinkage |
| `MarketABM`, `market_sim` | repeat-purchase market ABM (WTP, reference price, loss aversion, stockpiling, promotion, social effect) with pooled calibration | measured; not the most accurate |
| `hawkes_mean_field`, `power_kernel`, `seismic_kernel`, `SeismicPredictor`, `shape_forecast` | self-exciting cascade predictors for share curves, plus the shape baseline | **not yet measured** |
| `AgentSociety`, `ShareCascadeWorld`, `ChoiceWorld`, `World` | turn-based LLM agent society: a deterministic world, agents acting through the LLM protocol, a log for every turn | infrastructure; no accuracy measured |

**Provenance.** Everything is ported from the TALP prep research branch `makro`. The algorithms and defaults
are unchanged. Shorthand used below:
- **[makro RAPOR]** = `uretim/makro/RAPOR.md`
- **[abm RAPOR]** = `uretim/makro/abm/RAPOR.md`
- **[M1 NOTLAR]** = `uretim/makro/abm/m1/NOTLAR.md`
- **[RECETE]** = `uretim/makro/RECETE.md`

Every number below comes from those files. This port did not re-measure anything.

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
  - Paired ABM - Bass control: -1.25 +/- 0.21; seed-averaged 95% CI [-1.83, -0.82]; the ABM is better on 58 of
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
  no curve shape **[M1 NOTLAR]**. Use the analogy only for series of one kind, such as countries x one technology.
- **Bands.** Plain ABC bands are too narrow (80% coverage 0.54 internet, 0.64 US). The analogy variant is
  calibrated (0.77-0.80) **[M1 NOTLAR]**.
- **Cost.** A cold bank takes 53-85 s at default size on a loaded 4-core server; warm fits take seconds
  **[M1 NOTLAR]**. Smaller banks (`n_sim=300, n_agents=300`, as in the tests) were **not measured**.
- **`personas()` and `scenario()`.** These are model-implied attributions and counterfactuals, not observed
  quantities. Label them so.

## Adoption baselines

`curves.py` holds naive, linear trend (last 5 points), damped trend (phi 0.9 fixed), logistic, Gompertz and Bass
NLS. All of them use multi-start least squares with m bounded below by the observed max. It also holds
`BassAnalogy`, which uses the median (log p, log q) of the other series as a penalty term.

Measured on the internet dev set (122 series) **[makro RAPOR M1]**: naive 20.42, linear 12.75, damped 13.69,
logistic 16.96, Gompertz 12.39, Bass 15.97, Bass + analogy 15.94. On US households: linear 8.70 is the best
baseline. Early in the curve (cutoff t0+3..8), S-curve fits rarely beat a straight trend **[RECETE]**. Always put
`linear` and `gompertz` next to any adoption claim.

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

## Diffusion: Hawkes / SEISMIC cascade predictors (NOT YET MEASURED)

**What it does.**
- **Recursion.** `hawkes_mean_field` runs the mean-field self-exciting recursion in 1-minute bins after an
  observation window [0, t]. The branching factor is estimated per cascade and capped (0.99 / n*).
- **Kernels.** `power_kernel(c, theta)` is chosen on training data with `select_power_kernel`.
  `seismic_kernel(s0=5, theta=0.242)` uses the SEISMIC defaults.
- **SEISMIC-style predictor.** `SeismicPredictor` weights events by follower counts and adds the research's
  linear correction of the log final size.
- **Shape baseline.** `shape_forecast` is the median training growth after t.

**Status.** The code is ported from the pre-registered `yayilim-24s` study (Twitter15/16 and SEISMIC, 24-hour
curves). The run in the sources stopped after the training-set kernel grid search. **No dev or test result
exists**, so this module quotes no accuracy number. Before using any of these on hackathon day, score them
against `shape_forecast` and a fixed median curve with `simengine.evidence`.

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

**Status.** This is infrastructure. The research society for retweet curves was pre-registered, but its
evaluation never ran, so there is **no measured accuracy**. Related measured facts:
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
