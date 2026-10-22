# `simengine.research.hypotheses`: automated social science loop and experiment outcome predictor

Module: `simengine/research/hypotheses.py`. Tests: `tests/test_research_auto_social_science.py` (28 tests, FakeLLM
only; one real-data check runs only with `$SIMENGINE_DATA`). The tests need pandas: run them with the project venv. Not yet exported from `simengine.research`; import the
module directly:

```python
from simengine.research import hypotheses as H
```

## What it does

Two modes.

**1. Hypothesis loop** (after Manning, Zhu & Horton 2024). An LLM states a causal hypothesis as a linear
structural causal model (SCM): a scenario, a list of agents, one outcome measured by a survey question put to one
agent, and 1-3 exogenous causes. Each cause has a kind (continuous, count, ordinal, binary, nominal), units, a short
list of treatment levels, the second-person phrase the agent reads ("Your budget"), the agent who holds it, and
whether other agents can see it. The library then:

1. validates and repairs the proposal (`parse_hypotheses`). Every repair or rejection is one logged line.
2. builds the full factorial design over the levels × replicates (`design`). A `max_cells` cap takes a seeded
   subsample, and that is logged.
3. asks the forecaster LLM for every path estimate and its significance (`predict_paths`, the paper's Fig. A.11
   task) **before** any simulation, and logs the forecast's SHA-256.
4. simulates each design row (`InterviewSimulator`, `ConversationSimulator`, or any numpy world through
   `StructuralSimulator`, `linear_world` or `market_world`).
5. fits the SCM by OLS with an intercept (`estimate`). It uses statsmodels if installed and numpy otherwise. Nominal
   causes are dummy coded against their first level, interactions are column products, and both unstandardised and
   standardised (β\*) paths are reported.
6. compares fitted and forecast paths (`compare_paths`): |pred|/|fit| ratio, sign and significance agreement, and r.

`AutoSocialScience(llm, scenario, ...).run()` chains these steps. Every decision goes to a `DecisionLog`, which is
in memory and also written to `workdir/decisions.jsonl`: proposal raw text and problems, accepted hypotheses, design
size, forecast hash, simulator, protocol, call counts, unparsed answers, estimator backend, dropped rows and columns,
and the comparison summary. The workdir also gets `data_<id>.csv`, `paths.csv` and `summary.json`.

**2. Experiment outcome predictor** (after Hewitt, Ashokkumar, Ghezae & Willer 2024). `forecast_experiment` takes a
setting, `{condition: stimulus}`, an outcome question, a scale and the control condition. There are two modes:

- `mode="simulate"`, the paper's strategy. For each condition, `n_per_condition` participants are drawn from a
  profile table. Each prompt has a template drawn from `PROMPT_BANK`, the profile, the stimulus, the question and
  the scale. Answers are averaged per condition.
- `mode="direct"`. The LLM is asked for every condition mean, `n_samples` times, and the answers are averaged.

Effects are condition mean − control mean. Two scorers compare forecasts with measured effects:

- `score_effect_forecasts` reports r, disattenuated r, Spearman, slope and sign accuracy on significant contrasts.
- `score_condition_forecasts` follows the paper's protocol. In each repeat it draws one control condition per study
  at random (and, with `outcome=`, one dependent variable per study, as the paper does), scores all contrasts, and
  reports the median over 16 repeats.

## API

| name | what |
|---|---|
| `Variable`, `Hypothesis` | SCM spec. `Hypothesis.edges()`, `.to_dict()`, `.from_dict()`, stable `id` (hash of the spec) |
| `HYPOTHESIS_SYSTEM`, `hypothesis_prompt`, `parse_hypotheses(obj, scenario, max_causes=3, max_levels=5)` | proposal prompt and validator; returns `(hypotheses, problems)` |
| `propose_hypotheses(llm, scenario, n=1, attempts=2, log=)` | asks the LLM; if no hypothesis is valid, asks once more with the problems listed |
| `design(hyp, replicates=1, max_cells=None, seed=0)` | factorial DataFrame with columns `cell`, `rep` and one per cause |
| `encode(hyp, df)` | design matrix plus `{term: cause}` (`name[level]` for dummies, `a:b` for interactions) |
| `estimate(hyp, data, outcome_col="y", backend="auto")` | `(paths, info)`. paths has term, cause, coef, se, t, p, lo, hi, beta_std, n. info has backend, n, dropped_missing, dropped_constant, r2. Raises on a rank-deficient design |
| `InterviewSimulator(llm)` | one call per row: the outcome agent sees its own and public attributes and answers the survey question |
| `ConversationSimulator(llm, judge_llm=None, protocol="round_robin" or "random", max_statements=20, judge_from=1)` | agents take turns. An LLM judge checks after every statement (from statement `judge_from` on; the paper asks after each one) whether the conversation is over. Then the outcome agent answers the survey with the transcript. All rows advance in lockstep, one `llm.map` per step. `.transcripts` keeps them |
| `StructuralSimulator(fn)`, `linear_world(coefs, intercept, noise, seed, levels)`, `market_world(price=, promo=, theta=, weeks=)` | vectorised worlds. `market_world` runs `simengine.abm.engine.market_sim` (prior midpoint by default) and returns mean weekly sales during the event |
| `PATH_SYSTEM`, `path_prediction_prompt`, `predict_paths(llm, hyp, terms, n_rows, n_samples=1, all_cells=True)` | path forecast: term, predicted, pred_sd, sig_share, n_valid. `all_cells=False` tells the forecaster the design is a `max_cells` subsample (the loop sets it) |
| `compare_paths(fitted, predicted)` | `(table, summary)`. summary has n_paths, sign_agreement, sig_agreement, mean_ratio, median_ratio, overestimate_share, r |
| `predict_outcomes(llm, hyp, cells, fitted=None)` | per-cell outcome forecast, optionally with the fitted SCM shown (the paper's "predict y_i \| β̂" comparison) |
| `AutoSocialScience(llm, scenario, simulator=, sim_llm=, predictor_llm=, replicates=, max_cells=200, n_forecasts=, workdir=)` | `.run(n_hypotheses=1, hypotheses=None)` returns `AutoSocialScienceResult` (`.runs`, `.summary`, `.table`, `.log`). A hypothesis that fails (for example, every answer unparsed) is logged as `failed` and skipped |
| `DecisionLog(path=None)` | `.log(event, **payload)`, `.events(name)`, `.records` |
| `parse_number(text, lo, hi)` | reads `{"answer": x}` or the first number in the text. Returns NaN if nothing is found or the value is outside [lo, hi] |
| `PROMPT_BANK`, `forecast_experiment(...)`, `ExperimentForecast` | Hewitt-style forecaster. `.conditions` has mean, sd, n, n_invalid. `.effects` has effect, se (simulated) and n. With `same_profiles=True` the se is the paired se of the per-respondent differences; otherwise the two-sample se |
| `score_effect_forecasts(pred, true, true_se)`, `score_condition_forecasts(df, study=, condition=, pred=, true=, true_se=, outcome=None, repeats=16)` | forecast accuracy. `outcome=` draws one dependent variable per study per repeat (paper); without it every `study` group is one unit |

Example (offline; with a real backend, replace FakeLLM with `se.llm.Claude(...)`):

```python
from simengine.llm.fake import FakeLLM
from simengine.research import hypotheses as H
ass = H.AutoSocialScience(FakeLLM(my_responder), "A buyer and a seller bargain over a used mug.",
                          simulator="conversation", replicates=2, workdir="runs/ass")
res = ass.run(n_hypotheses=1)
print(res.summary)            # sign agreement, mean |pred|/|fit|, r2, n per hypothesis
```

## Provenance

The method was checked on the paper pages (arXiv HTML of 2404.11794v2; Hewitt et al. preprint PDF dated 8 Aug 2024)
on 22 Oct 2026, re-checked in review the same day (Manning et al. §A.3.1 and §A.4 of the arXiv HTML; Hewitt et al.
main text, Method section). No code was taken from either paper; the module is written from the method descriptions.

- **Manning, Zhu & Horton (2024), "Automated Social Science: Language Models as Scientist and Subjects",
  arXiv:2404.11794.** What matches the paper:
  - the SCM as the blueprint for agents, design and analysis;
  - variable operationalisation (type, units, levels, a second-person proxy, individual or scenario scope,
    public or private);
  - an outcome measured by a survey question to one agent;
  - every combination of treatment values;
  - turn-taking conversations ended by an LLM judge or after 20 statements. The paper's coordinator is asked
    "after each agent speaks", so the judge is asked from the first statement on (`judge_from=1`; an earlier
    version of this module waited for the second);
  - the paper's "ordered" protocol (`round_robin`) and its "random" protocol (random first speaker, then random
    with nobody speaking twice in a row);
  - a forecast of path estimates and significance from a prompt that describes the SCM, operationalisations and
    number of simulations (Fig. A.11). The ratio is |predicted / estimated| per path, as in Tables A.1-A.2;
  - predict-y with and without the fitted SCM (Fig. A.10).

  Differences:
  - The paper fits with the R package lavaan (§A.4: "The system uses the R package lavaan to estimate all paths").
    Here it is OLS with classical standard errors. For an outcome on
    exogenous causes the coefficients are the same, and the standard errors differ by about √(n/(n−k)).
  - The paper lets the LLM choose among six turn-taking protocols (ordered, random, central-ordered,
    central-random, coordinator-before, coordinator-after). Here there are the first two, and the caller chooses.
  - The paper generates the survey questions, has an LLM turn each answer into a number, and lets an LLM pick an
    aggregation (min, max, average, mode, median, sum) for multi-question outcomes. Here there is one question,
    written by the proposer, and answers are parsed as numbers (`parse_number`).
  - There are no mediators or endogenous chains, only causes → one outcome, plus optional two-way interactions.
- **Hewitt, Ashokkumar, Ghezae & Willer (2024), "Predicting Results of Social Science Experiments Using Large
  Language Models"** (preprint). What matches the paper:
  - simulate participants: introductory message with a study description, a demographic profile, the stimulus,
    and the question with its scale;
  - an ensemble over a prompt bank;
  - condition means averaged;
  - one control condition and one dependent variable drawn at random per study, 16 repeats, median r (Method:
    "randomly selected one control condition per study (and one dependent variable, in cases where the original
    study contained multiple)"; use `outcome=`);
  - disattenuated r reported next to raw r.

  Differences:
  - Disattenuation here is Spearman's correction for sampling error in the measured effects only,
    reliability = 1 − mean(se²)/var(effect). The main text sends the details to the SI ("Analysis of Experimental
    Data"); the SI link (treatmenteffect.app/supplement.pdf) redirects to a DocSend viewer that could not be read,
    so the paper's exact formula is not verified.
  - The paper does not say in the main text how effects are put on a common scale; the measurement script
    standardises by the pooled within-experiment sd (our choice).
  - `same_profiles=True` (the default) reuses one profile draw across conditions, and the simulated se is then the
    paired se. The paper does not say it paired profiles across conditions; `same_profiles=False` draws
    independently.

**Literature numbers** (from the papers, not reproduced here):

- Manning et al.: GPT-4's path forecasts averaged 13.2× the fitted estimates (5.3× without the largest), 10 of 12
  were overestimates, 10 of 12 signs were right, and 10 of 12 significance calls were right. In the auction, clearing-price MSE was 8,628 without
  the fitted SCM and 1,505 with it, against 128 for auction theory.
- Hewitt et al.: across 70 experiments (50 TESS, 20 from two replication projects) with 476 effects, r = 0.85
  (r_adj = 0.91), and r = 0.90 (r_adj 0.94) on unpublished studies; 90% correct signs on significant contrasts.
  Lay forecasters reached r = 0.79 (r_adj 0.84).

## Measured numbers

All numbers come from one run on 22 Oct 2026, with no LLM calls (re-run in review the same day after the fixes;
the synthetic and pooled rows reproduced to the digits shown):

```bash
SIMENGINE_DATA=<data-root> PYTHONPATH=. python examples/auto_social_science_measure.py   # ~10 s
```

| check | result |
|---|---|
| OLS recovery on `linear_world` (one continuous cause with 5 levels, one nominal cause with 3 levels; 15 cells × 4 replicates = 60 rows; noise sd 1; 500 seeds) | 95% CI coverage 0.953 over 1,500 coefficient intervals; mean bias 1.9e-5 |
| numpy vs statsmodels backend (same 500 fits) | max abs difference in coef, se or p: 2.8e-15 |
| `market_world` (market_sim at the prior midpoint, 400 cohort agents, 4-week event; price cut {0, 0.1, 0.2, 0.3} × promo {0, 1}) | price_cut path +3.34 (units of mean weekly sales per 100% cut), promo +0.23, R² 0.96 |
| conversation pipeline with the test FakeLLM (bargaining, 6 cells, judge says yes once both agents have spoken) | 32 LLM calls in total (1 proposal, 1 forecast, 12 statements, 12 judge, 6 survey). Before the judge fix it was 26 (judge skipped after the first statement) |
| SocSci210 condition table (Kolluri et al. 2025; 5,937 conditions, 1,187 outcome × study units from 209 studies, 1,009 of them with ≥ 2 conditions, 4,750 contrasts per repeat when every outcome is pooled; means standardised by the pooled within-experiment sd, so contrasts are Cohen's d) | reliability of the measured contrasts 0.943 pooled, 0.969 under the paper's one-outcome draw, so disattenuation changes r by 2-3% |
| SocSci210 under the paper's protocol (`study=study_id, outcome=task_num`: one outcome and one control per study per repeat; 198 studies with a usable outcome, median 901 contrasts and 290 significant ones per repeat) | reliability of the measured contrasts 0.969 |
| baselines on SocSci210, paper protocol (16 repeats, median) | random forecast: r −0.033, sign accuracy on significant contrasts 0.512. TF-IDF ridge on the stimulus text (5-fold out-of-fold, grouped by study): r 0.038, r_adj 0.039, sign accuracy 0.474. Stimulus length: r 0.032, sign accuracy 0.460 |
| same baselines, every outcome pooled (`study=study:task`, 4,750 contrasts, 1,380 significant) | random: r −0.014, sign 0.501. TF-IDF ridge: r 0.025, r_adj 0.026, sign 0.455. Stimulus length: r 0.026, sign 0.430 |

The SocSci210 rows are the floor an LLM forecaster has to clear on this table: text-only learners do not predict
within-experiment contrasts. The differences among these baselines are noise: with about 290 significant contrasts
per repeat, a coin's sign accuracy has a standard deviation of about 0.03. No LLM forecast has been run on it yet (no real LLM calls in this unit), so there is no
measured LLM number here.

## Limitations

- Infrastructure only: no real-LLM run of either mode has been made in this port. Do not quote an accuracy for it.
- Linear SCM with one outcome. There are no mediators or SEM fit indices, and the standard errors are classical
  (not robust). LLM subjects often give the same answer for every replicate (for example, a cached reply or
  temperature 0), which makes the residual variance and the standard errors too small. Use replicates with distinct
  prompts (the simulators add a "participant k" line) and check `fit_info["r2"]` near 1 as a warning sign.
- An LLM forecaster that wrote the hypothesis also knows what it meant to find. The forecast comes before the
  simulation and is hashed, but that is not independence. Pass a separate `predictor_llm` to separate the roles.
- The interview simulator shows the outcome agent only its own and public attributes, so a private attribute of
  another agent has a zero path by construction. The conversation simulator can pass it on through speech. This is
  the intended contrast, but it is easy to misread as a null effect.
- `parse_number` takes the `answer` key of a JSON reply, else the first number in the text, so free text such as
  "on a 1 to 7 scale, 5" reads as 1. The prompts ask for JSON; check `unparsed` / `n_invalid` in the log.
- `forecast_experiment` treats any number on the scale as valid. Refusals or out-of-range answers become NaN and are
  counted (`n_invalid`). If they cluster in one condition they bias that condition's mean.
- SocSci210 licence: the Hugging Face card states none (the catalogue marks it as not verified; the source is TESS
  open data). It is read at runtime only, and no rows are in the repo.
