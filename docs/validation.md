# Validation: what reviewers will recognize, and where simengine does it

"Does the simulation look human?" is a face-validity question, and reviewers have learned to discount it. This
page lists the validation practices from the agent-based-modelling and LLM-simulation literature. Each one is
mapped to the simengine function that implements it or audits it, with a status:
- **present**: in the tree and exported.
- **planned**: specified for a later port but not yet in this tree.
- **in progress**: being ported now. Some functions of the module are in the tree, but its public API is not complete yet.
- **gap**: not implemented, and not planned.

The statuses reflect the tree on 22 Oct 2026 after the second porting round (persona methods, Big Five, games). Check the
live state with `python -c "import simengine.games"`.

Number tags follow [landscape.md](landscape.md#how-to-read-the-tags):
- kind: **paper** or **vendor self-reported**
- check: **verified** (against the arXiv abstract on 22 Oct 2026) or **unverified (vault)**

Numbers measured in the two source repos are tagged **source repo**: `RP` is replicant and `ET` is edsl-test. The
file each one came from is named, and each was re-read from that file. They are single-model runs (stepfun or
gpt-oss-120b) and make no claim about Claude.

## 1. The minimal standard: Larooij & Törnberg's three criteria

Source: Larooij & Törnberg, "Do Large Language Models Solve the Problems of Agent-Based Modeling? A Critical Review
of Generative Social Simulations", arXiv:2504.03274 (v1). The vault files it as the PNAS Nexus 2025 version,
"Validation Is the Central Challenge for Generative Social Simulation"
(`AW/Papers/(PNAS Nexus 2025) Validation Is the Central Challenge ... .md`).

The review covers 35 generative-ABM studies. It finds 15 of 35 relying only on subjective validation (**paper ·
unverified (vault)**: the arXiv v1 abstract gives no counts and says only "many studies relying solely on
subjective assessments of model 'believability'"). It names five weak practices: human judgment, well-known
social patterns, comparison with earlier models, human-generated data, and internal consistency. It proposes
three criteria. The vault says the paper does not engage with ODD, POM or calibration. Sections 2 to 6 bring in
those classic tools.

| criterion | what it asks | simengine | status |
|---|---|---|---|
| **1. Purpose alignment.** Validation targets follow from the mechanism you claim. | Name the targets, the metric and the falsifier **before** the run. Score the mechanism-level patterns, not one aggregate. | `simengine.research.Prereg` (question, metrics, baselines, expected result, falsification, stop rule) and `simengine.research.lock_prereg`. `simengine.research.declare_targets` writes the published findings you will be compared with into the lock, so changing a target changes the hash. `evaluate` warns when `ours` was not named in advance (selection bias). Pattern-level targets: §2 (POM). | present |
| **2. External grounding.** Human data, ideally pre-registered experiments. | Compare with real human outcomes on held-out units, and seal the predictions before the truth is opened. | `simengine.evidence.evaluate` against `truth`; the `simengine.bench` tasks (SimBench, Twin-2K-500, OpinionQA, GSS, ...); `simengine.research.seal` and `simengine.research.grade_sealed` (predictions are joined with the truth only when they match a seal); `simengine.evidence.published_findings` (simulated − published, row by row, with the source); page-cited game references `simengine.games.human_refs` | present |
| **3. Robustness and sensitivity.** Across runs, seeds, wordings and decoding settings. | Repeat runs; vary one knob at a time; report how much the result moves. | paired cluster-bootstrap CIs in `evaluate`; `simengine.research.harness_variants` (axes: wording, max_tokens, temperature, plus `extra` axes such as harness or persona method) and `simengine.research.harness_sweep`; `simengine.research.Loop` (one change per turn, paired-CI acceptance, a single test run with a Bonferroni line). For variance-based sensitivity, see §5. | present (Sobol: gap) |

## 2. Pattern-oriented modelling (POM): many weak patterns at once

Source: Grimm et al., "Pattern-Oriented Modeling of Agent-Based Complex Systems", Science 2005
(`AW/Papers/(Science 2005) ...`, `AW/Concepts/Pattern-Oriented Modeling ... .md`). The idea is to match several
observed patterns across levels at the same time, rather than fitting one aggregate. Several weak filters together
reject wrong model structures. That reduces equifinality, where different models fit the same aggregate.

| use | simengine | status |
|---|---|---|
| Generic multi-criterion index: a weighted mean of 0-1 criterion scores. N/A criteria drop out of both numerator and denominator, and the observed weight share is reported. | `simengine.evidence.stylized_facts_score` | present |
| The 9-criterion who-to-punish scorecard (weights 3/2/1, about 40 page-cited targets from Ertan, Page & Putterman 2009). It is a POM instance. | `simengine.games.wtp_scorecard`, fed by `simengine.abm.run_who_to_punish` | present |
| Per-game moment panels: dictator modal split and zero spike, ultimatum offers and low-offer rejection, PGG conditional cooperation | `simengine.games.moments`, `simengine.games.distance_to_human` | present |
| Several metrics at once on one frame | `evaluate` computes the whole metric family per task (for distributions: 1 − TVD, SimBench score, ordinal W1, JS, KS, mode match) | present |

What POM caught in our sources (**source repo ET**, `ET/benchmark/REPORT.md`):
- On the who-to-punish index, data-anchored personas reach 0.78 to 0.81. The bare model reaches 0.694.
- The bare model scores 0 on the institution criteria in all 3 seeds: it always dismantles punish-low.
- Within-regime criteria alone would have hidden this. Only the multi-pattern score shows it.

## 3. Docking and replication: levels of equivalence

Source: Axtell, Axelrod, Epstein & Cohen, "Aligning Simulation Models", CMOT 1996, and Wilensky & Rand, "Making
Models Match", JASSS 2007 (`AW/Concepts/Docking - Alignment - Replication (equivalence levels).md`). There are
three levels, from strictest to loosest: numerical identity, distributional equivalence and relational
equivalence. The vault recommends distributional equivalence as the target for "human-like".

| level | meaning here | simengine | status |
|---|---|---|---|
| **Numerical identity** (port vs original) | The library port must give the same numbers as the research code. | `tests/test_evidence_equivalence.py` (identical reports apart from the timestamp), `tests/test_abm_equivalence.py`. They skip when the original code is absent. `simengine.llm.FakeLLM` and `simengine.llm.DiskCache` make runs replayable. | present |
| **Distributional equivalence** (simulation vs humans) | Same answer distribution, not the same mean. | `evaluate` with `task="distribution"` (TVD, W1, KS, JS); `simengine.evidence.variance_ratio` (sim SD / human SD with a bootstrap CI); the spread audit (sd ratio < 0.8 = homogenisation); `simengine.games.distance_to_human` (Wasserstein/KS where full human distributions exist) | present |
| **Relational equivalence** (same structure) | Same signs and sizes of relationships: regression coefficients, subgroup differences, correlations across items. | `simengine.evidence.coefficient_recovery` (the same OLS on simulated and human rows: sign agreement, share differing at α, sign flips); the segment analysis in `evaluate` (correlation of segment-mean differences, worst segments); the within-item correlation and de-meaned R² metrics | present |

## 4. Calibration: SMD, ABC, SBI

Source: `AW/Concepts/Calibration - Estimation of ABMs (SMD, indirect inference, Bayesian-ABC, SBI).md`; Grazzini,
Richiardi & Tsionas, "Bayesian Estimation of Agent-Based Models", JEDC 2017; Grazzini & Richiardi, simulated
minimum distance, working paper 2013-2015 (vault titles; venues unverified).

| method | simengine | status |
|---|---|---|
| **Simulated minimum distance / method of moments.** Choose parameters that bring simulated moments closest to human moments. | `simengine.methods.fit_mixture_weights`: simplex weights over theory personas that minimise SSE to human targets. It checks identifiability (`simengine.methods.identifiability`) and supports a `val=` holdout. `simengine.methods.probe_types` measures each type's behaviour through `LLM.map`. | present |
| **Approximate Bayesian computation.** Rejection ABC over a simulation bank. | `simengine.abm.AdoptionABC`, `simengine.abm.AdoptionABCAnalogy`, `simengine.abm.BassABC` (structure control), `simengine.abm.abc_distance` | present |
| **Neural simulation-based inference** (NPE and similar) | not ported. The `makro` NPE variant scored worse than ABC on the internet dev set: 12.30 vs 10.78 ([modules/abm.md](modules/abm.md)). | gap (deliberate) |

A calibration that fails to transfer is still a result. **Source repo ET**, `ET/personas/homo_silicus/weights.json`:
- The homo_silicus weights fit on Charness-Rabin menus are self_interested .36, efficient .50 and
  selfish_but_fair .14.
- The reciprocal type gets weight 0, so the calibrated population can never legalize punishment.
- `fit_mixture_weights` reports weights like these together with an identifiability check, so that two types
  with identical behaviour cannot silently take each other's weight. The v1 bug in edsl-test was exactly that.

## 5. Sensitivity analysis: OAT and Sobol over prompt and model knobs

Source: Ten Broeke, van Voorn & Ligtenberg, "Which Sensitivity Analysis Method Should I Use for My Agent-Based
Model?", JASSS 2016 (`AW/Concepts/Sensitivity & Robustness Analysis ... .md`). For LLM agents the "parameters" are
the wording, framing, temperature, max_tokens, model version and persona attributes.

| method | simengine | status |
|---|---|---|
| **One-at-a-time (OAT).** Change one knob from the default and measure the paired difference. | `simengine.research.harness_variants` (every one-key change is a legal move) + `simengine.research.harness_sweep` (runs all of them, with paired CIs against the default) + `simengine.research.Loop` | present |
| **Variance-based global SA (Sobol first- and total-order indices).** | Not implemented. A full factorial over `harness_variants` axes can be run with the grid proposer, but no Sobol estimator exists. | gap |

Why this matters, in numbers:
- Analytic flexibility: across alternative analysis configurations of one published silicon-sample study, the
  correlation between human and silicon association structures ran from r = .23 to r = .84 (Cummins,
  arXiv:2509.13397 v4; **paper · verified** for the r range). The vault gives 252 configurations in Study 1 and
  66 in Study 2; those counts are **unverified (vault)**.
- Harness: with the same model and temperature, random-persona agents voted punish-low 14% of the time under the
  EDSL harness and 80% under the oTree harness. **Source repo RP**: `RP/papers/ertan2009/results/big5_random_summary.json`
  (0.14) and `otree_random_summary.json` (0.80).
- Decoding: raising max_tokens to 8192 turned a bare model that never punished into one that punishes the
  free-rider every period. **Source repo ET**:
  `ET/experiments/07_who_to_punish/results/2026-06-11_15-08-37_baseline/ANALYSIS.md`. "Decoding parameters are a
  behavioral treatment." Caveat: the default-cap morning run was partly None-corrupted (truncated answers), and
  the source says the "never punishes" claim "conflated model behavior with a truncation artifact". So it is
  evidence that the cap changes the recorded data, not evidence about model behaviour at the default cap.

So every simulated answer row stores `wording_id`, `max_tokens` and `seed`, and the harness
axes are part of the pre-registration.

## 6. Replication first, then novel prediction

Source: Cross, Haber & Yamins, "Validating Generative Agent-Based Models of Social Norm Enforcement: From
Replication to Novel Predictions", arXiv:2507.22049 (v1). **paper · verified** (the abstract describes the two
stages and gives no numbers). Stage 1 finds the minimal agent architecture that reproduces known human results:
third-party punishment, and gossip and ostracism in public goods games. Stage 2 uses that validated architecture
to predict conditions nobody has tested.

| step | simengine | status |
|---|---|---|
| Calibrate on one set, validate on a held-out set | `split` column (`train`/`test`) in every EvalFrame; the dev / select / test splits in `simengine.research.Loop`; `fit_mixture_weights(val=...)`; calibration vs holdout games in `simengine.games.moment_frame` and `simengine.bench.tasks.econ_games_b01` | present |
| Out-of-sample transfer to structurally different settings (Manning & Horton, arXiv:2508.17407; [landscape.md](landscape.md) B11) | mixture weights fit on one game family and scored on another | present (method: `fit_mixture_weights` with `charness_rabin_games` → `dictator_game`); no cross-family measurement here |
| Novel prediction, sealed before the truth exists | `simengine.research.seal` + `simengine.research.verify_chain` + `simengine.research.grade_sealed` | present |

## 7. Accuracy normalized by human test-retest

People do not answer the same question the same way twice. A simulation cannot be expected to beat a person's own
consistency, so accuracy should be reported as a share of the human test-retest ceiling.

| anchor | number | tags |
|---|---|---|
| Park et al., arXiv:2411.10109 | 83% / 82% / 86% of 2-week test-retest (interview / survey / both), vs 74% demographics-only (v3); "85%" in v1 | paper · verified |
| Maier et al. (SSR), arXiv:2510.08338 | 90% of human test-retest reliability | paper · verified |
| Twin-2K-500, arXiv:2505.17479 | the 4th wave repeats earlier tasks to give a test-retest baseline; the abstract gives no figure | paper · verified |

simengine: the `retest` column gives the human ceiling. `evaluate` then reports the ratio to the ceiling and the
**gap closed**, (method − baseline) / (ceiling − baseline). The gap closed is the headline, because the ratio alone
flatters. On Twin-2K-500 the best method reaches 94% of the ceiling, but the demography baseline alone reaches
93%, and the gap closed is 17% [16, 19] ([modules/evidence.md](modules/evidence.md)). The audits flag a method
that beats the ceiling as a leakage sign. `simengine.sim.retest_stats` measures the LLM's own consistency across
repeated samples. Status: present.

## 8. Variance ratio and coefficient recovery

Source: Bisbee, Clinton, Dorff, Kenkel & Larson, "Synthetic Replacements for Human Survey Data? The Perils of Large
Language Models", Political Analysis (not on arXiv; journal page not fetched). Aggregate means can match while
spread collapses and regression coefficients differ. Numbers: 48% of coefficients differ significantly from
ANES, and signs flip 32% of the time. **paper · unverified (vault)**. The vault's copy of the abstract states the
reduced variance and the differing coefficients only qualitatively.

simengine: `simengine.evidence.variance_ratio` and `simengine.evidence.coefficient_recovery` (present). Details
and the other spread failures are in [failure-modes.md](failure-modes.md).

## 9. Seeds, completion and controls

- **At least 3 seeds.** The who-to-punish index ranged by up to 0.37 within a single persona method across 3
  seeds. **Source repo ET**: `ET/benchmark/REPORT.md`. A ranking from one seed misleads. `evaluate` clusters
  bootstrap resamples by `group`, so put the seed or run into `group` when seeds are the unit. Status: present.
- **Completion on the card.** The share of planned runs that finished goes on every card, per condition:
  `simengine.evidence.completion_row`, `simengine.evidence.completion_from_frame`,
  `simengine.sim.completion_summary`. Status: present.
- **No-persona control and baselines.** Every persona result is shown next to a no-persona arm, a persona-swap
  arm and the baseline ladder (random, marginal, demography, plus label-free rules): the `sim` control arms
  `no_persona` and `persona_swap`, the `shuffle_control` audit, and `simengine.evidence.baseline_ladder`.
  Status: present.

## 10. Prediction-powered inference: "worth N humans"

A simulation can be combined with a small human sample instead of replacing it. PPI (Angelopoulos et al.,
arXiv:2301.09633) and PPI++ (Angelopoulos, Duchi & Zrnic, arXiv:2311.01453) give valid confidence intervals that
tighten as the predictions improve. The vault's mixed-subjects-design note (SMR 2025, unverified (vault)) turns
this into an effective sample size.

simengine: `simengine.evidence.worth_humans` (PPI++ points, multiplier = MSE(classical) / MSE(PPI++)) and
`simengine.evidence.sample_multiplier`. Status: present. Our measurement, at n = 100: persistence 4.36x; LLM
persona behaviour prediction 1.06x (README, `kanit RAPOR §1a`).

## 11. Documentation protocols: ODD and TRACE

Sources: Grimm et al., ODD protocol, Ecol. Model. 2006 / 2010 / JASSS 2020, and Grimm et al., TRACE, Ecol. Model.
2014 (vault titles).

simengine has no ODD template. The following cover parts of it:
- the pre-registration (purpose, data, metrics)
- the research-loop journal (`JOURNAL.md`, `results.tsv`, the observer's raw prompts and replies)
- the evidence card's limits section

Status: partial. An ODD-shaped card section is a gap.

## Quick checklist for a hackathon submission

1. Lock the pre-registration with its targets (`Prereg`, `declare_targets`, `lock_prereg`) before running.
2. Seal the predictions before opening the truth (`seal`).
3. Run with control arms (`no_persona`, `persona_swap`), at least 3 seeds, and logged `wording_id` and
   `max_tokens`.
4. `evaluate` with `retest` when you have it. Lead with the gap closed and the CI.
5. Add `variance_ratio`, `coefficient_recovery` (when there are covariates), `completion_row` and
   `stylized_facts_score` to the card.
6. Sweep the harness one knob at a time (`harness_variants` + `harness_sweep`) and report how far the headline
   moves.
