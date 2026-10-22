# simengine.abm

This module holds the agent-based models and the baselines they have to beat:

| component | what it is | status |
|---|---|---|
| `AdoptionABC`, `AdoptionABCAnalogy` | adoption-curve ABM on a small-world network, calibrated with rejection ABC, plus an optional analogy prior from other series | measured; best method on OWID internet |
| `BassABC`, `BassABCAnalogy` | the same ABC with a closed-form Bass simulator (structure control) | measured |
| `CurveBaseline`, `BassAnalogy`, `naive`, `linear_trend`, `damped_trend`, `logistic`, `gompertz`, `bass`, `curves.constant_growth` | adoption-curve baselines | measured (constant growth: rehearsal only) |
| `OpinionPanel`, `opinion_baselines`, `BoundedConfidenceABM` | opinion drift between two survey waves: persistence, mean shift and transition baselines vs a Deffuant-Weisbuch ABM | measured; the ABM loses |
| `ElasticityShrinkage`, `shuffle_elasticities`, `elicit_elasticities`, `PreEventMean` | price-elasticity prior (category, LLM or mix), ridge-shrunk toward the event's own history; LLM-swap placebo | measured; the gain comes from shrinkage |
| `MarketABM`, `market_sim` | repeat-purchase market ABM (WTP, reference price, loss aversion, stockpiling, promotion, social effect) with pooled calibration | measured; not the most accurate |
| `hawkes_mean_field`, `power_kernel`, `seismic_kernel`, `SeismicPredictor`, `shape_forecast` | self-exciting cascade predictors for share curves, plus the shape baseline | measured; Hawkes ties and SEISMIC loses to the shape baseline |
| `AgentSociety`, `ShareCascadeWorld`, `ChoiceWorld`, `World` | turn-based LLM agent society: a deterministic world, agents acting through the LLM protocol, a log for every turn | measured; no better than a single LLM score or a calibrated constant curve |
| `run_who_to_punish`, `run_repeated_pd`, `run_trust_interaction` | economic-game LLM societies (Ertan-Page-Putterman who-to-punish with voting, repeated prisoner's dilemma, trust game) returning structured per-answer rows | infrastructure; no measured result in this port |

**Provenance.** Everything is ported from the TALP prep research branch `makro`. The algorithms and defaults
are unchanged. Shorthand used below:
- **[makro RAPOR]** = `uretim/makro/RAPOR.md`
- **[abm RAPOR]** = `uretim/makro/abm/RAPOR.md`
- **[M1 NOTLAR]** = `uretim/makro/abm/m1/NOTLAR.md`
- **[RECETE]** = `uretim/makro/RECETE.md`
- **[prova D7 NOTLAR]** = `uretim/prova/d7/NOTLAR.md` (rehearsal D7, fixed-origin adoption forecast; added with the calendar-origin backtest)
- **[yayilim24 SONUC]** = `uretim/makro/abm/yayilim24/SONUC.md` (pre-registration `makro/onkayit/yayilim-24s.md`, commit 06224f7; run 21-22 Oct, added by the integrator after this module was written)

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
# fixed calendar origins (one cut year for every series), low alignment level, ineligible series counted
frame = abm.adoption_backtest(df, {"abm": abm.AdoptionABCAnalogy(threshold=0.001),
                                   "growth": abm.CurveBaseline("growth", phi=0.9, upper=200)},
                              calendar_origins=range(1991, 1997), horizon=5, on_ineligible="skip")
frame.attrs["n_ineligible"], frame.attrs["ineligible_series"]   # series the ABC could not align

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
  no curve shape **[M1 NOTLAR]**. Use the analogy only for series of one kind, such as countries x one technology.
- **Bands.** Plain ABC bands are too narrow (80% coverage 0.54 internet, 0.64 US). The analogy variant is
  calibrated (0.77-0.80) **[M1 NOTLAR]**.
- **Cost.** A cold bank takes 53-85 s at default size on a loaded 4-core server; warm fits take seconds
  **[M1 NOTLAR]**. Smaller banks (`n_sim=300, n_agents=300`, as in the tests) were **not measured**.
- **`personas()` and `scenario()`.** These are model-implied attributions and counterfactuals, not observed
  quantities. Label them so.
- **Series below the alignment level.** A series must have reached `threshold` (a fraction, default 0.01 = 1%) by
  the cutoff; otherwise `fit` raises `IneligibleSeries` (a `ValueError`). With fixed calendar origins this can be
  most of the panel: in rehearsal D7 (country x technology, origin 1997) 198 of 361 series were below 1%.
  `threshold=0.001` aligns the bank at the 0.1% crossing instead (one extra bank per level, cached separately);
  only the 1% default was measured. `eligible(history, series, cutoff)` tells you in advance.

**Backtests** *(this port)*. `adoption_backtest` cuts at t0(s) + k by default (relative cutoffs, research task
M1). `calendar_origins=[1991, ..., 1996]` uses the same fixed calendar years for every series instead (all
series by default; concatenate two calls for both kinds); `cutoff_rel` is then only a label. `on_ineligible`
decides what happens when a method cannot align a (series, origin) block: `"raise"` (default, the old behaviour),
`"skip"` (drop the whole block, so every method is scored on the same rows) or `"nan"` (keep the rows, NaN for that
method, fill with your fallback). Either way `frame.attrs` lists `ineligible` ({series, cutoff, method}),
`ineligible_series` and `n_ineligible`. `adoption.fit_many(method, history, cutoff, on_ineligible="skip")` does
the same for one cutoff and returns `models`, `ineligible`, `n_ineligible`. `fit_many`, `IneligibleSeries` and
`FitMany` live in `simengine.abm.adoption` (not re-exported from `simengine.abm` yet).

## Adoption baselines

`curves.py` holds naive, linear trend (last 5 points), damped trend (phi 0.9 fixed), logistic, Gompertz and Bass
NLS. The S-curve fits use multi-start least squares with m bounded below by the observed max. It also holds
`BassAnalogy`, which uses the median (log p, log q) of the other series as a penalty term.

- **`upper=`** *(this port)*. Every curve clips to [0, upper] and bounds m above by it; the default 100 keeps the
  research behaviour. Raise it for quantities that pass 100 (mobile subscriptions per 100 people), or pass `None`
  for no bound. `CurveBaseline(name, **kw)` and `BassAnalogy(upper=)` pass it through.
- **`constant_growth`** (`CurveBaseline("growth")`) *(this port)*. The last observed log growth rate (over the last
  `last=3` years) is held (`phi=1`) or damped by phi^i (`phi<1`); `offset=0.1` pp keeps zeros finite. In rehearsal
  D7 (361 country x technology series, fixed origins 1991-1996, h=1..5, per 100 people) a hand-written version
  (last 3 years, damped) was the strongest simple baseline on dev: 1.731 MAE vs Gompertz 1.783, linear 1.885,
  Bass 1.950, naive 2.610 **[prova D7 NOTLAR]**. The library version was not re-measured. Near saturation it
  overshoots (a pooled log-growth model took 42 to 150 there), so keep `upper` and consider `phi<1`.

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
