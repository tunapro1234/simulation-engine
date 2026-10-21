# Failure modes of LLM population simulation, and the audit that catches each one

Each entry gives:
- the failure
- the evidence and its number, with tags
- the simengine audit that catches it, with a status

Tags follow [landscape.md](landscape.md#how-to-read-the-tags):
- kind: **paper** or **vendor self-reported**
- check: **verified** (against the arXiv abstract, 22 Oct 2026) or **unverified (vault)**

**Source repo** marks a number measured in replicant (`RP`) or edsl-test (`ET`). It names the file the number was
re-read from. Those runs used a single model (stepfun/step-3.5-flash or gpt-oss-120b) and small n. They show that
a failure can happen. They do not measure how often it happens on Claude.

Status:
- **present**: in this tree
- **planned**: specified for a later port but not yet here
- **in progress**: partly in the tree, public API not complete
- **gap**: not implemented, and not planned

The statuses reflect the tree on 22 Oct 2026 after the second porting round (persona methods, Big Five, games).

Validation methods are on [validation.md](validation.md).

## Summary

| # | failure | headline number | audit | status |
|---|---|---|---|---|
| F1 | homogeneity / variance collapse | regression coefficients differ in 48%, signs flip in 32% (Bisbee, unverified (vault)); GPT-4 answers lean "more altruistic and cooperative" (Mei, verified) | `variance_ratio`, spread audit | present |
| F2 | broken relational structure | r = .23 to .84 across configurations (Cummins, verified) | `coefficient_recovery`, segment-difference correlation | present |
| F3 | social desirability / prosocial drift | 1.20 human SD for GPT-4, 0.98 for Llama 3 (Salecha, verified); agreeableness +0.735, neuroticism −0.692 (RP) | `validate_bigfive` bias per domain, `published_findings` | present |
| F4 | harness and analytic flexibility | 14% vs 80% on the same model (RP); max_tokens flips punishment (ET) | `harness_variants`, `harness_sweep`, `Prereg` lock | present |
| F5 | silent, non-random dropout | 7/50 finished the punishment page (RP) | `completion_row`, `completion_summary` | present |
| F6 | imputation that hides failure | NaN→3 and None→$0 in the sources | `ask_constrained` `valid` flag, no imputation | present |
| F7 | persona makes the fit worse | punish-low votes 94% → 80% → 46% (RP) | `no_persona` / `persona_swap` arms, `shuffle_control`, baseline ladder | present |
| F8 | seed variance read as a ranking | index range up to 0.37 within a method (ET) | bootstrap CIs, ≥3 seeds, `stylized_facts_score` coverage | present |
| F9 | corner play / incoherent choices | dictator $0, bomb exactly 50 with SD 0 (ET) | `games.coherence_audit`, rational-corner baseline | present |
| F10 | caricature and identity flattening | 3,200 participants, 16 identities, 4 LLMs (Wang, verified) | segment analysis (worst segments) | partial; no caricature metric |
| F11 | training-data leakage / memorization | — | `memorization_audit`, contamination audit, `split_audit`, ceiling-exceed audit | present |
| F12 | model drift over time | same prompt differs over 3 months (Bisbee abstract, unverified (vault)) | model id in every row, `seal`, `DiskCache` | present (logging only) |
| F13 | cache-degenerate controls | baseline and homo_silicus effective n = 1 (ET) | `probe_types(sample_kw=...)`, `Loop` "repeat" status, `variance_ratio` | present |
| F14 | dev gains that vanish on test | OpinionQA loop gain did not hold on test (simengine's own run) | `Loop` select split, single test run, Bonferroni | present |
| F15 | position bias in A/B prompts | — | half-reversed panels in `sim.ab_test`, position-swap audit | present |
| F16 | bots in the "human" truth data | 99.8% pass rate on 6,000 attention-check trials (unverified (vault)) | none | gap |

## Details

### F1. Homogeneity: the simulated population is too uniform

- **Evidence.**
  - Bisbee et al. (Political Analysis), "Synthetic Replacements for Human Survey Data?". Aggregate ANES
    thermometer means look close, but variance is compressed. 48% of regression coefficients differ
    significantly, and signs flip 32% of the time. The vault also says the collapsed variance changes power
    calculations, "33 vs 270+ N needed". **paper · unverified (vault)**: not on arXiv, and the abstract states
    the reduced variation only qualitatively.
  - Mei et al., arXiv:2312.00798 (PNAS 2024). GPT-4 is "statistically indistinguishable from a random human",
    but its play tends toward the "more altruistic and cooperative end of the distribution". **paper ·
    verified**.
  - Li, Li & Qiu, arXiv:2507.02919. "Severe homogenization" and minority opinions underrepresented on ANES 2020.
    **paper · verified** (no numbers in the abstract).
  - Shi & Haupt, arXiv:2604.23575 (2026). Over-correlated judgments among 277 philosophers across 7 models,
    checked against the PhilPapers 2020 survey (N = 1,785). **paper · verified** (abstract numbers). The 2026
    arXiv ID still needs a venue check.
- **Audit.**
  - `simengine.evidence.variance_ratio`: sim SD / human SD per item with a bootstrap CI, plus a pooled ratio.
  - The spread audit in `simengine.evidence.evaluate`: sd ratio < 0.8 is a homogenisation warning, > 1.5 is
    over-dispersion.
  - Measured in this library's source: Twin-2K-500 sd ratio 0.75 → warning ([modules/evidence.md](modules/evidence.md)).
- **Status.** present.

### F2. Broken relational structure: right means, wrong relationships

- **Evidence.**
  - Bisbee: the 48% / 32% coefficients above. **unverified (vault)**.
  - Cummins, arXiv:2509.13397 (v4). Defensible configuration choices moved the human-vs-silicon association-structure
    correlation from r = .23 to r = .84. **paper · verified**.
  - Li, Li & Qiu: accuracy "doesn't hold across demographic aggregation levels" (structural inconsistency).
    **paper · verified**.
- **Audit.**
  - `simengine.evidence.coefficient_recovery`: the same OLS on simulated and human rows; sign agreement, share
    that differs at α, sign flips.
  - `evaluate` segment analysis: correlation of segment-mean differences, worst three segments.
  - Within-item correlation and de-meaned R² (person-specific signal).
- **Status.** present.

### F3. Social desirability and prosocial drift

- **Evidence.**
  - Salecha et al., arXiv:2405.06058 (v2). When models infer they are being tested on Big Five personality, they
    skew toward desirable poles: GPT-4 by "1.20 (human) standard deviations", Llama 3 by 0.98. This holds in
    every model tested and survives paraphrasing. **paper · verified**.
  - The "Utopian illusion" study, arXiv:2510.21180: social role bias, primacy effect and positivity bias in
    chatroom simulations. **paper · unverified (vault)** (abstract not fetched).
  - Our sources (**source repo RP**, `RP/papers/ertan2009/results/validation_continuous.json`, stepfun, N = 20,
    seed 42). Personas were assigned with BFI-2 sentences and measured with Mini-IPIP. Pooled r = .911, but
    agreeableness bias +0.735 (4.02 → 4.75) and neuroticism −0.692. Rank order survives while levels drift
    toward warm and stable. The source imputed NaN answers as 3, which the port removes.
- **Audit.**
  - `simengine.persona.validate_bigfive`: per-domain r, MAE and **bias**, with `n_missing`; missing answers are not imputed. Present.
  - `simengine.evidence.published_findings`: signed deltas against published values. Present.
  - Corner and midpoint game baselines in `simengine.bench.tasks.econ_games_b01`. Present.
- **Rule.** Always report bias next to r.

### F4. Harness, wording and decoding sensitivity (analytic flexibility)

- **Evidence.**
  - Cummins: r = .23 to .84 (F2). **paper · verified**.
  - Tjuatja et al. (TACL 2024). LLMs do not reproduce human survey response biases, and they react to
    perturbations that do not move humans. **paper · unverified (vault)** (no numbers taken).
  - **Source repo RP.** Same model and temperature, random personas, punish-low vote 0.14 under EDSL
    (`RP/papers/ertan2009/results/big5_random_summary.json`) vs 0.80 under oTree (`otree_random_summary.json`).
    Single runs.
  - **Source repo ET.** At max_tokens 8192 the bare model punishes the current free-rider $10 every period. At
    the default cap it never punished (9/9 decisions $0, partly None-corrupted).
    `ET/experiments/07_who_to_punish/results/2026-06-11_15-08-37_baseline/ANALYSIS.md`. Single run.
- **Audit.**
  - `simengine.research.harness_variants` + `simengine.research.harness_sweep`: one-knob changes with paired CIs
    against the default.
  - `simengine.research.Loop` (one change per turn).
  - `simengine.research.Prereg` / `simengine.research.lock_prereg`: fix the harness before looking.
  - Rows store `wording_id`, `max_tokens`, `seed`.
- **Status.** present. A variance-based Sobol decomposition is a gap ([validation.md](validation.md) §5).

### F5. Silent dropout, and dropout that is not random

- **Evidence.** **Source repo RP**, `RP/papers/ertan2009/results/otree_cooperative_raw.json`, recounted here.
  - In the hyper-cooperative persona condition all 50 agents finished the baseline and voting pages, 30/50 the
    regime page, and **7/50** the punishment page. The default and random conditions finished 50/50.
  - The summary JSON still says n = 50.
  - Non-completers differ: baseline contribution 17.6 for the 20 who did not finish the regime page vs 15.83
    for the 30 who did; 16.44 for the 43 who did not finish the punishment page vs 17.14 for the 7 who did
    (recounted from `otree_cooperative_raw.json`; [RESULTS.md](RESULTS.md) §8 row 3).
  - A reasoning-budget cause is suspected but was never confirmed by a rerun.
- **Audit.**
  - `simengine.evidence.completion_row` / `simengine.evidence.completion_from_frame`: planned vs completed vs
    parse failures per condition, with a test of equal completion.
  - `simengine.sim.completion_summary` over `simengine.sim.ask_constrained` results.
- **Rule.** Put n_completed / n_planned per condition on every card.
- **Status.** present.

### F6. Imputation that turns failures into data

- **Evidence.**
  - **Source repo RP / ET**: NaN → 3 in replicant's Big Five validation (`RP/src/replicant/personalities/validation.py`);
    None → $0 in edsl-test who-to-punish before commit 15b9531 (`ET/personas/value_anchor/NOTES.md`,
    `ET/personas/README.md`); a substring "direct value" salvage in replicant's form bot (`RP/src/replicant/otree/bot.py`).
  - Each one turns a missing answer into a plausible-looking one. The $0 version fabricated the "rational
    corner" story (F4 evidence above).
- **Audit.**
  - `simengine.sim.ask_constrained`: schema validation with retry feedback; a failed field stays missing and is
    logged; no salvage.
  - The `valid` flag on every row.
  - `completion_row` counts parse failures separately from dropout.
- **Status.** present. `simengine.abm.run_who_to_punish` and the games runner (`simengine.games.play`) are specified with
  tests that a None answer stays invalid and is never counted as $0.

### F7. The persona makes things worse

- **Evidence.**
  - **Source repo RP**, `otree_{default,random,cooperative}_summary.json`, one-shot Ertan vignette, N = 50 each.
  - Punish-low vote: no persona 0.94, random Big Five personas 0.80, hyper-cooperative 0.46. Punish-high is 0 in
    all three.
  - The no-persona agents fit the institutional result best.
  - simengine's own measurement (docs/RESULTS.md, CES 2020): gap closed 0.55 with personas, 0.45 with no
    persona, 0.25 with a cross-segment shuffle.
- **Audit.**
  - The `sim` control arms `no_persona` and `persona_swap` (`simengine.sim.control_personas`).
  - The `shuffle_control` audit in `evaluate`.
  - `simengine.evidence.baseline_ladder` (random, marginal, demography, plus `unlabeled_baselines`).
- **Status.** present.

### F8. Seed variance read as a ranking

- **Evidence.** **Source repo ET**, `ET/benchmark/REPORT.md`, gpt-oss-120b, one group of 4 per seed, 3 seeds.
  - The index ranges up to 0.37 within one method. For example, gps scored 0.57 to 0.90.
  - The order inside the top four (0.78 to 0.81) is not significant.
  - The report: "claim the cluster, not the podium".
- **Audit.**
  - The cluster bootstrap in `evaluate` (put the seed or run into `group`).
  - Run at least 3 seeds.
  - `simengine.evidence.stylized_facts_score` reports weight coverage, so indexes over different criteria
    sets are not compared.
- **Status.** present.

### F9. Corner play and incoherent choices

- **Evidence.** **Source repo ET**, early runs (`ET/experiments/0[3-6]*/results/*/summary.txt`; [RESULTS.md](RESULTS.md) §8 row 9).
  - The bare model gives $0 as dictator, $0 as trust sender, $0/20 in the public goods game, and defects 5/5 in
    PD.
  - It picks exactly 50 boxes in the bomb task, with SD 0.
  - b01 has 30 invalid rows out of 10,920 (0.3%). 28 of them are in two twin2k cells: pg_punishment 21/90
    and trust_sender 7/30, both 23% invalid (`ET/benchmark/beebench/b01/ANOMALIES.md`, `results.csv`).
- **Audit.**
  - `simengine.games.coherence_audit` flags without dropping rows: invalid_rate > 10%, zero_variance, multiple
    switching in Holt-Laury, dominated beauty-contest choices, and others.
  - A "rational corner" baseline in `simengine.bench.tasks.econ_games_b01`.
  - `variance_ratio` also reports a zero simulated SD. Present.
- **Status.** present (`simengine.games.coherence_audit`, `econ_games_b01` baselines).

### F10. Caricature, stereotyping and identity flattening

- **Evidence.**
  - Wang, Morgenstern & Dickerson, arXiv:2402.01908 (v3; Nature Machine Intelligence). LLMs misportray and
    flatten demographic groups. Human studies with 3,200 participants across 16 identities and 4 LLMs. **paper ·
    verified**.
  - Cheng, Piccardi & Yang, CoMPosT (EMNLP 2023). Political and marginalized groups are the most susceptible to
    caricature. **paper · unverified (vault)**.
  - Durmus et al., arXiv:2306.16388. Prompting the model with a country's perspective "can reflect harmful
    cultural stereotypes". **paper · verified**.
- **Audit.**
  - `evaluate` segment analysis: per-segment metric with CI and bias, worst three segments.
  - `variance_ratio` per item within a segment (filter the frame).
- **Status.** partial. There is no individuation or exaggeration (caricature) metric: gap.

### F11. Training-data leakage and memorization

- **Evidence.** The Larooij & Törnberg review lists training-data leakage among the core problems (vault
  summary). The arXiv:2504.03274 abstract was checked only for counts, which it does not give.
- **Audit.**
  - `simengine.evidence.memorization_audit`.
  - The contamination audit: performance before vs after the model's training cutoff, via `time` and `cutoff=`.
  - `simengine.evidence.split_audit`: train/test group overlap, temporal leakage, duplicate units.
  - A method beating the human `retest` ceiling is flagged as a leakage sign.
- **Status.** present.

### F12. Model drift over time

- **Evidence.** Bisbee: "the same prompt yields significantly different results over a 3-month period" (vault
  copy of the abstract). **paper · unverified (vault)**.
- **Audit.**
  - The model id is stored with each response.
  - `simengine.llm.DiskCache` makes a past run replayable.
  - `simengine.research.seal` timestamps the predictions.
- **Status.** present for logging and replay. There is no automatic drift test (gap): rerun and compare with
  `evaluate`.

### F13. Cache-degenerate controls

- **Evidence.** **Source repo ET**, `ET/benchmark/REPORT.md` ([RESULTS.md](RESULTS.md) §8 row 7). In the who-to-punish benchmark, the baseline and homo_silicus
  arms got the same cached answer every time, so their effective n was 1. Their index ranges (0.66 to 0.69 and
  0.68 to 0.68, `ET/benchmark/REPORT.md`) reflect that.
- **Audit.**
  - `simengine.methods.probe_types` passes the repetition index through `sample_kw` so that caching backends give
    independent draws.
  - `simengine.research.Loop` marks per-unit scores that are identical to an earlier turn as "repeat".
  - `variance_ratio` shows a zero spread.
- **Status.** present.

### F14. Gains found on dev that vanish on test

- **Evidence.** simengine's own research loop on OpinionQA setting A. The dev gain over the position prior
  (−0.0106 WD) did not hold on test: winner − baseline was +0.0025 [−0.0018, +0.0070]. Random search was as good
  as the LLM observer ([modules/research.md](modules/research.md)).
- **Audit.** `simengine.research.Loop`: a select split, a single test run, a Bonferroni line, and the random
  control arm `simengine.research.control_arm`.
- **Status.** present.

### F15. Position bias in paired prompts

- **Evidence.** Measured in this library's sources: in A/B panels, half of the panel sees the variants reversed to
  estimate the position effect ([modules/sim.md](modules/sim.md)).
- **Audit.**
  - `simengine.sim.ab_test` (half-reversed panel).
  - The position-swap audit in `evaluate` (`reverse_<method>` columns).
- **Status.** present.

### F16. Synthetic respondents inside the "human" truth data

- **Evidence.** "The potential existential threat of large language models to online survey research", PNAS 2025.
  An autonomous synthetic respondent passed 99.8% of 6,000 standard attention-check trials. **paper ·
  unverified (vault)**: not on arXiv, and doi.org redirects to the journal site, which was not fetched.
- **Audit.** None. simengine assumes `truth` is human. Use panels with verified respondents, and treat
  unsupervised online panel data as possibly contaminated.
- **Status.** gap (out of scope). Roundtable pivoted to exactly this detection problem
  ([landscape.md](landscape.md) §1).
