# `simengine.research`: pre-registration, sealed predictions, turn-based auto-research loop, search modules

Ported from the TALP prep code: the loop driver `uretim/dongu/kod/dongu.py` (rules `uretim/dongu/onkayit/dongu.md`
v1.0 to v1.2), the `claude -p` observer `uretim/dongu/kod/gozlemci.sh`, the bootstrap of `uretim/dongu/kod/kosucu.py`, and
the seal chain `uretim/kanit/kod/kanit/onkayit.py`. Design ideas (three splits, smoke gate, random control arm,
ablation, mini pre-registered proposals, repeat rejection) come from `arastirma/tahmin/notlar/p22-otoarastirma.md` §4.
That note looked at AIDE, AI Scientist v1/v2, Agent Laboratory, autoresearch, AIRA-dojo, OpenEvolve and others. No code
was taken from them; the note's licence check rules AI Scientist code out.

## Components

| file | what it does |
|---|---|
| `prereg.py` | `Prereg` (question, data, split, metrics, baselines, expected, falsification, stop rule) written as markdown or JSON. `lock_prereg` stores its SHA-256 and a timestamp in `<file>.lock.json` and can also commit both to git (`git_commit=True`). `verify_prereg` fails after any edit; `check_git=True` also reports the first commit time and whether the committed copy is identical. Sealed predictions: `seal` hashes a prediction table in canonical form (row and column order do not matter, floats fixed to 10 significant digits). Dicts and other files work too. Each record goes onto a hash chain with a gzip archive copy. Also `verify_seal`, `verify_chain` (detects edited records), and `grade_sealed`, which joins predictions with the truth only if they match a seal. |
| `space.py` | `Space(params, defaults, fixed, forbidden, used, complexity, check)`. Each param is a list of choices or a `(lo, hi)` range; int ends make an int range. `validate` returns the full config and a list of errors. |
| `loop.py` | `Loop` runs one change per turn on the dev split. A change is kept only if the paired bootstrap CI of its improvement over the current best excludes 0. Other statuses are discard, crash, and repeat (per-unit scores identical to an earlier turn). The loop stops after `stop_after` consecutive non-kept turns, `max_turns` turns, the `budget_usd` proposer budget, an exhausted proposer, or two invalid proposals in one turn. Finalisation has three steps. An optional `select_split` re-ranks the top 3 dev configs. An optional ablation reverts each changed key. Then the winner, the start config and the baselines run once on the test split, with 95% and Bonferroni CIs (k = number of proposal turns). The state is saved to `state.json` and can be resumed. The workdir also gets per-unit CSVs, `JOURNAL.md`, `results.tsv`, and the observer's raw prompt and reply. `control_arm(loop)` builds a random arm with the same number of turns. |
| `proposers.py` | `LLMProposer(llm)` is the observer. It sees dev metrics only and must reply with JSON: one change plus `kind`, `hypothesis`, `expected` (lower, same or higher) and `falsifier`. The loop validates the reply. An invalid reply gets one repair call that includes the rejection reason. A reply that mentions the test split is rejected. Turkish keys from the original loop are accepted. `ClaudeCodeProposer` does the same through `claude -p` (tools off, no session persistence), using `simengine.llm.ClaudeCode` if available, else a local fallback. Search proposers: `RandomProposer` (the control arm), `GridProposer`, `GreedyCoordinateProposer` (hill climbing with momentum), and `HalvingProposer`, which uses `successive_halving` to screen single changes at low fidelity on dev. `make_proposer("random"/"grid"/"greedy"/"halving"/"claude-p"/"claude"/llm)`. |
| `prereg.py`, `space.py` (additions) | Declared targets and harness robustness sweeps: see the next section. |
| `prereg.py` (addenda, deviation arms) | `add_addendum`, `verify_addenda`, `read_addendum`, `seal_deviation`, `grade_arms`: see "Addenda and deviation arms" below. |
| `stats.py` | `paired_bootstrap_diff`, `bootstrap_mean` (seed 42, B=1000, percentile CI: the same defaults as the measured loops); `paired_cluster_bootstrap_diff(a, b, groups)` resamples whole groups (used by `grade_arms(cluster=)`). |

**Runner contract** (`simengine.types.Runner`): the runner receives `cfg` with `cfg["split"]` set to `"dev"`, `"select"`
or `"test"` (key names can be changed). With `HalvingProposer` it also gets `cfg["fidelity"]`. It returns
`{"per_unit": DataFrame[unit, <primary>, ...], "metrics": {...}}`. The loop computes every number itself from the
per-unit scores; it never takes a metric the runner reports about itself. Every per-unit numeric column gets a
bootstrap CI in the journal, and the paired comparison uses the primary column on the units both runs share.

## Search and method modules

Six modules sit next to the loop. Each has its own page with the API, provenance checked against the papers,
measured numbers and limitations. Their main names are exported from `simengine.research` lazily, so
`import simengine.research` still loads no scipy or scikit-learn; the module itself is `simengine.research.<name>`.
Generic helpers stay in their module: `bandits.simulate`, `bandits.compare`, `bayesopt.minimize`,
`hypotheses.design` / `encode` / `estimate`, `prompt_evolution.split_units` / `toy_problem`. The Hyperband function is
`research.hyperband.hyperband` (`research.hyperband` is the module). `examples/research_new_demo.py` runs one small
synthetic case of each in about 2 s (FakeLLM, no data).

All numbers below are copied from the unit pages, where the commands are. They come from synthetic problems unless a
dataset is named, and no real LLM was called in any of them.

### `bandits`: choose among K methods under a noisy evaluation budget ([page](research/bandits.md))

UCB1, Thompson sampling (Beta, Gaussian), successive rejects, LUCB1 and uniform allocation; `identify_best(pull,
n_arms, budget, algorithm)`, a vectorised simulator (`SimulatedArms`, `bandits.simulate`) and `RacingProposer`, which
races the one-change candidates of a `Loop` turn over `cfg["fidelity"]` mini-batches. Exported: `SimulatedArms`,
`BanditResult`, `identify_best`, `RacingProposer`, `ucb1`, `thompson_beta`, `thompson_gaussian`, `successive_rejects`,
`lucb`, `lucb_beta`, `sr_schedule`, `AUDIBERT_BUBECK_2010`.
Measured on the Audibert, Bubeck & Munos (2010) problems (2000 runs each): successive rejects, LUCB1 and Thompson-Beta
beat uniform allocation by a wide margin where there are many bad arms (exp1: uniform 0.446, SR 0.187 error). Our
uniform and SR rates are within 0.045 of the bars read off the paper's Figure 5 on six problems; exp3 and exp5 do not
match the printed formulas (unresolved, see the page). LUCB1's round index starts at K + 1 as in the paper.

### `bayesopt`: GP-EI / GP-UCB and TPE-lite ([page](research/bayesopt.md))

`GPProposer` and `TPEProposer` plug into `Loop` (acquisition maximised over the one-change neighbours of the
incumbent); `bayesopt.minimize(objective, space, method="gp" | "tpe" | "random")` moves every key. Exported:
`GPProposer`, `TPEProposer`, `SpaceEncoder`, `TPE`, `Parzen1D`, `fit_gp`, `gp_scores`, `expected_improvement`,
`upper_confidence_bound`, `ucb_beta`, `branin`, `branin_space`, `BRANIN_MIN`, `mixed_toy`, `mixed_toy_space`.
Measured (20 seeds, 30 calls): with every key free, GP-EI and GP-UCB beat random search in 20 of 20 paired seeds on
Branin (median regret 0.036 and 0.008 against 1.31). Inside `Loop` (one key per turn) the GP still beats random on
Branin, but on the mixed toy all three model proposers lose to random (median 0.30 against 0.015): the one-change
rule makes BO a coordinate search, which cannot cross an interaction. A 30-call GP run takes 16-24 s.

### `hyperband`: Hyperband, ASHA and a multi-fidelity runner protocol ([page](research/hyperband.md))

`MultiFidelity(fn, max_budget)` and `UnitRunner(score_units, n_units)` map (config, budget) to a score, with the
budget in units (rows, personas, questions, LLM calls) and common random numbers across configs. `hyperband.hyperband`
follows JMLR Algorithm 1 over `successive_halving`; `asha` follows MLSys Algorithm 2. Exported: `FidelityRunner`,
`MultiFidelity`, `UnitRunner`, `space_sampler`, `pool_sampler`, `hyperband_brackets`, `ASHA`, `asha`.
Measured (200 seeds, one synthetic objective): when a low budget is an unbiased mean of units, ASHA beats Hyperband
by 0.0165 regret [0.0092, 0.0242] and charges the fewest units; when low budgets mislead (late bloomers), ASHA is
about twice as bad and random search at full budget beats Hyperband by 0.017. Not wired into `make_proposer`;
`HalvingProposer` stays the loop's multi-fidelity proposer.

### `tree_search`: AIDE-style tree search and MCTS-UCT ([page](research/tree_search.md))

`TreeSearch(runner, operators)` drafts, debugs and improves nodes with the aideml search policy (greedy best node);
`MCTSSearch(runner, space)` runs UCT over the grid of a `Space`. Both use the loop's runner contract and workdir
layout, and `finalize()` runs the winner and the defaults once on test and reports the winner's test - dev gap next
to the defaults' gap. Operators: `ConfigOperators` (no LLM) and `LLMOperators`. Exported: `TreeSearch`, `MCTSSearch`,
`ConfigOperators`, `LLMOperators`, `summarize_tree`.
Measured: on the toy objective (budget 20, 10 seeds) AIDE-improve reaches 0.033 mean true effect, MCTS 0.104, random
0.134, `Loop` 0.020. On Bank Marketing (row Brier, temporal split, 3 seeds) every search lowered dev Brier and 11 of
12 winners were worse than the sklearn defaults on test; the base rate shifts from 11% in dev to 31% in test.

### `prompt_evolution`: OPRO, APE and Promptbreeder over a text ([page](research/prompt_evolution.md))

Search on the training split, re-rank the top-k on validation, seal the winner, score the winner and the baselines
once on test with paired CIs (`select_and_test`; `evolve` does search and selection in one call). Exported:
`PromptEvaluator`, `PromptEvolutionResult`, `opro`, `ape`, `promptbreeder`, `select_and_test`, `evolve`,
`make_llm_scorer`, `opro_meta_prompt`, `ape_forward_prompt`, `diversity_filter`.
Measured only on the offline keyword toy with a FakeLLM optimiser (5 seeds): every method's winner beats the seed
text on test in 5 of 5 seeds and ends 0.01-0.05 below the oracle text. This checks the pipeline, not LLM
optimisation quality; no real-LLM run is reported.

### `hypotheses`: automated social science and experiment outcome prediction ([page](research/auto_social_science.md))

`AutoSocialScience(llm, scenario, simulator=...)` has an LLM state a linear SCM, builds the factorial design, simulates
every cell (LLM interview, LLM conversation with a judge, or a numpy world such as `linear_world` / `market_world`),
fits it by OLS and compares the fit with the LLM's forecast made before the simulation (Manning, Zhu & Horton 2024).
`forecast_experiment` and `score_condition_forecasts` predict and score survey-experiment effects (Hewitt et al.
2024). Every decision goes to a `DecisionLog`. Exported: `AutoSocialScience`, `AutoSocialScienceResult`,
`HypothesisRun`, `Hypothesis`, `Variable`, `DecisionLog`, `propose_hypotheses`, `parse_hypotheses`,
`InterviewSimulator`, `ConversationSimulator`, `StructuralSimulator`, `make_simulator`, `linear_world`,
`market_world`, `predict_paths`, `compare_paths`, `ExperimentForecast`, `forecast_experiment`,
`score_effect_forecasts`, `score_condition_forecasts`.
Measured: OLS 95% CI coverage 0.953 on `linear_world` (500 seeds). On SocSci210 under the paper's one-outcome draw,
the non-LLM baselines are at chance (random r -0.033, TF-IDF ridge r 0.038, sign accuracy 0.46-0.51); that is
the floor an LLM forecaster has to clear. No LLM forecast has been run.

## Declared targets and harness robustness

Ported from replicant (`papers/TEMPLATE/config.py` `PAPER_FINDINGS`, declared before the run, and the condition label
of `src/replicant/cli.py` `determine_condition`). No replicant code is copied; the oTree/EDSL calls are not ported.

| function | what it does |
|---|---|
| `declare_targets(findings, prereg=None, overwrite=False)` | Normalises the findings a replication is compared against. Accepted forms: `{"metric": (value, "description")}` (the `PAPER_FINDINGS` form), `{"metric": value}`, `{"metric": {"value", "description", "tolerance", "interval", "source", "unit", "notes"}}`, or a list of dicts with `"metric"`. NaN values, unknown fields, `lo > hi` and a tolerance without a numeric value are refused. With a `Prereg` it sets `prereg.targets`; with a prereg file path it rewrites the file, and refuses if that file is already locked (`overwrite=True` rewrites it, after which the old lock fails until you re-lock). |
| `targets_sha256(targets)` | SHA-256 of the canonical targets (key order and float noise past 10 significant digits do not matter). |
| `lock_prereg` / `verify_prereg` | Unchanged API. The targets are part of the prereg file, so they are covered by the file hash. When the file is a `Prereg` with targets, the lock record also stores `targets_sha256` and `n_targets`, and `verify_prereg` returns `targets_sha256`. A prereg without targets is written byte-for-byte as before. |
| `compare_targets(targets, observed)` | One row per declared metric: `pass`/`fail` (inside `interval`, or within `tolerance`; equality for a non-numeric value), `reported` (numeric value with no tolerance or interval, so only the difference is shown), `missing` (absent, None or NaN, never filled in). An observed metric that was not declared gets its own row with status `undeclared`, so post-hoc metrics show up as post-hoc. |
| `harness_variants(wordings, max_tokens_list, temperatures, *, defaults, extra, fixed, check)` | A `HarnessSpace` (a `Space`) with keys `wording_id`, `max_tokens`, `temperature` plus any `extra` axes, e.g. `{"harness": ["edsl", "otree"]}`. The first value of each axis is the default. `wordings` can be a list of ids or `{id: text}` (`space.wording(cfg)` looks the text up, and the fingerprint covers the texts, so a resumed loop notices an edited wording). `max_tokens` takes positive ints or `None` (provider default); `temperature` takes numbers ≥ 0 or `None`. Every one-key change is a legal `Loop` turn. |
| `space.variants(base=None)` / `one_change_variants(space, base)` | The base plus every config that differs from it in exactly one key, labelled `"default"` or `"key=value"`. Works for any `Space` (ranges give `n_grid` points). Candidates that fail `space.validate` are skipped. |
| `harness_sweep(runner, space, primary, split="dev")` | Runs the base and each one-change variant once (same runner contract as `Loop`) and reports, per variant, `n_units`, `n_missing` (NaN or non-numeric unit scores, left out and never imputed), `mean`, the paired difference from the base with a 95% CI and a Bonferroni CI over the number of variants, `direction` from the Bonferroni CI, and `error` for a variant whose runner raised. `df.attrs["mean_range"]` gives the spread of the variant means. |

Why: in the replicant Ertan, Page & Putterman (2009) vignette, the same model and temperature gave a random-persona
punish-low vote of 14% through EDSL and 80% through oTree (`papers/ertan2009/results/big5_random_summary.json`,
`otree_random_summary.json`; single runs). In edsl-test the bare model punished free-riders at max_tokens 8192 but not
at the default cap (single run). A result that holds for one wording, one cap and one harness is a single draw from
this space. Lock the targets and the harness space in the prereg, then show the sweep next to the headline number.
The tests check only synthetic data (`tests/test_research_extras.py`); no new measurement is reported here.

```python
from simengine.research import Prereg, declare_targets, lock_prereg, harness_variants, harness_sweep, compare_targets

p = Prereg(title="ertan vignette", question="...", falsification="...")
declare_targets({"vote_punish_low_pct": (0.85, "late rounds"), "vote_punish_high_pct": (0.0, "never")}, p)
lock_prereg(p.write("prereg.md"))                       # lock record carries targets_sha256

space = harness_variants({"w1": "...", "w2": "..."}, [None, 8192], [0.0, 1.0], extra={"harness": ["a", "b"]})
sweep = harness_sweep(runner, space, primary="punish_low")   # runner reads cfg["wording_id"], ["max_tokens"], ...
table = compare_targets("prereg.md", {"vote_punish_low_pct": 0.80, "vote_punish_high_pct": 0.0})
```

## Addenda and deviation arms

From the blind TALP rehearsals. In D6 and D8 the team had to deviate from a locked prereg. No helper existed:
appending to the file broke its sha256, and `lock_prereg(overwrite=True)` lost the old lock. In D6 and D7 the
pre-deviation arm had been sealed by hand, so both arms could be graded: in D6 the post-referee choice lost to the
pre-registered arm on test, and in D7 the deviation won. Neither outcome can be reported honestly unless the
pre-deviation arm is sealed and graded too.

| function | what it does |
|---|---|
| `add_addendum(parent="ONKAYIT.md", reason=, deviation=, falsification="", path=None, *, changes=None, notes="", git_commit=False)` | Writes a dated addendum `ONKAYIT-EK<k>.md` next to the parent (k = next free number) and leaves the parent file and its lock alone. The addendum's text records the parent's sha256, lock time and lock hash, the previous addendum and its sha256, the reason, the deviation and the new falsification line (empty means the parent's line stands), plus a JSON copy. It is locked with `lock_prereg(..., parent=)`, and that lock is chained to the parent lock hash and to the previous addendum's lock hash. It refuses an unlocked or edited parent, a previous addendum that no longer verifies, a missing reason or deviation, and an existing `path`. Returns the lock record plus `path`. |
| `lock_prereg(path, parent=)` | Writes the same chained lock for an addendum you wrote yourself. Without `parent` it behaves as before. |
| `verify_addenda(parent)` | Checks the parent and every `<stem>-EK<k>` file. For each addendum it reports `ok` (its own lock matches), `parent_ok` (the lock names the parent's current sha256 and lock hash; this fails if the parent was edited and re-locked) and `link_ok` (the lock chains to the previous addendum's lock). An addendum with a plain lock written by hand reports "not chained". |
| `seal_deviation(pre_arm, post_arm, description, registry_dir=".prereg", *, reason, addendum=None, pre_name=None, post_name=None)` | Seals the pre-deviation arm first, with `arm=pre_name`, `role="preregistered"` and `superseded_by`. Then it seals the post-deviation arm in the same chain, with `role="post_deviation"`, `deviation_of`, `deviation_of_record` (the pre record's hash), `deviation_reason`, and the addendum's file name and sha256 (the addendum must verify). Names default to the file stem. Identical arms are refused. If the pre arm's table was already sealed without a name (`seal` / `se.seal` in the same registry), the pre record points at that oldest seal (`first_sealed_at`, `first_sealed_seq`, `first_sealed_record`, `first_sealed_chain`): that time, not the call's, shows the arm was fixed before the deviation. Running it again with the same inputs returns the same records. The registry holds full gzip copies of both arms: for licensed data put `registry_dir` outside the repo. |
| `seal(..., arm=, role=, deviation_of=, meta=)` | The new fields are stored in the record and covered by its record hash. Records without them keep the old shape, and `verify_chain` passes on old chains. A table sealed earlier without an arm can be sealed again under an arm name. |
| `grade_arms(registry_dir, truth, keys, *, column=None, scorer=None, unit_loss=None, predictions=None, arms=None, cluster=None)` | Grades every arm record with one scorer and returns one row per arm, in chain order. Arm tables are read back from the seal archives (checked against the seal) or taken from `predictions={arm: df}` (checked too). Keys are joined in canonical text form, so an int key matches a str key. The default scoring is absolute error on `column`, giving `mae`, `rmse` and `bias`. With per-unit losses, a deviating arm also gets `diff_vs_pre` with a paired bootstrap CI (`diff_lo`, `diff_hi`, `diff_n`, `diff_cluster`, `diff_n_clusters`); a negative value means the deviation helped. Pass `cluster="segment"` (a key column) when each segment has several rows: rows of a segment share its errors, and the row bootstrap then under-covers (simulated, 61 segments x 7 items with a segment-varying arm advantage: 0.67-0.80 instead of 0.95; the segment bootstrap 0.92-0.93, about 0.89 with 12 segments). With several keys and no `cluster` the CI is row-level and a warning says so; `cluster=False` silences it. Arm rows with no truth row are counted in `n_unmatched` and warned about (key format, e.g. "01" vs 1); archived key columns are read back as text, so "01" stays "01". Arms scored on different units get a warning: their `mae`/`rmse` are then not comparable, and `diff_vs_pre` (shared units only) is the number to read. A custom `scorer(table)` adds its own columns; add `unit_loss` as well if you want the paired diff. |

```python
from simengine.research import add_addendum, seal_deviation, grade_arms, verify_addenda

add_addendum("ONKAYIT.md", reason="referee found a stronger base (T4b) on dev",
             deviation="ours = H_joint, picked from ~40 configs after the stop rule; dev gains are post-selection",
             falsification="claim fails if dev H_joint - T4b segment-clustered CI contains 0",
             changes=["C5 sealed separately", "T4b added to the ladder"])          # -> ONKAYIT-EK1.md + lock
seal_deviation("kol_C5.csv", "tahmin.csv", "2025 forecast", ".prereg", reason="post-referee choice",
               addendum="ONKAYIT-EK1.md", pre_name="C5", post_name="H_joint")
assert verify_addenda("ONKAYIT.md")["ok"]
table = grade_arms(".prereg", truth_df, keys=["segment", "item"], column="p",
                   cluster="segment")                                              # C5 and H_joint side by side
```

The evidence card does not show both arms yet. Until it does, report the `grade_arms` table next to the card.
The tests (`tests/test_research_prereg_deviation.py`) use synthetic data only, with one case where the deviation
loses and one where it wins.

## When it works and when it does not (measured)

Source loops: three real tasks, a Claude Haiku observer through `claude -p`, rules pre-registered before turn 0
(`uretim/dongu/onkayit/dongu.md`, `GUNLUK.md`, `turlar/<loop>/test/OZET.md`, `turlar/<loop>/<arm>/sonuclar.tsv`).
In the tables, "dev" is the split the observer climbs on, "selection" is a held-out part of dev, and "test" was run once.

**OpinionQA subgroup distributions, setting A** (`turlar/oqa_grup_A/test/OZET.md`; WD, lower is better):
- On dev, the loop found a significant gain over the pre-registered position-prior baseline. Switching to kNN gave a
  paired WD of -0.0106 [-0.0153, -0.0056]. The loop then stopped after 3 non-kept turns (5 turns in all, no invalid proposals).
- **On test the gain did not hold.** The winner scored 0.1669 [0.1590, 0.1750] against 0.1645 for the baseline. Winner
  minus baseline was +0.0025 [-0.0018, +0.0070]. The pre-registered falsification line (a) fired.
- **The random control arm was enough on selection.** LLM winner minus random winner was -0.0030 [-0.0096, +0.0032].
  On test, the LLM winner beat the random winner: -0.0062 [-0.0109, -0.0017].
- The observer's expected direction matched the outcome in 1 of 4 proposals (`turlar/oqa_grup_A/llm/sonuclar.tsv`).

**OpinionQA setting B** (`turlar/oqa_grup_B/test/OZET.md`):
- No change beat copying the known Overall distribution (dev WD 0.0530). Every move the observer tried was
  significantly worse; for example knn_sapma came out +0.0014 [+0.0006, +0.0021].
- The loop stopped at turn 3. The LLM arm, the random arm and the baseline gave the same test WD, 0.0649.
- 1 of 4 observer proposals was invalid (unparseable JSON) and was repaired.

**Twin-2K-500 kit** (`turlar/twin_kit/test/OZET.md`, loop v1.1; mean Fisher z, higher is better; the test summary
was added by the integrator after this module was written):
- LLM arm: 5 turns, 2 kept, winner `{"svd_r": 60, "knn_k": 100}`. Random arm: 1 kept, winner `{"svd_r": 63}`.
- Test z: LLM winner 0.3738 [0.3389, 0.4112], random winner 0.3720, fair baseline T-adil 0.3679, T-kit 0.2181
  (n = 122 columns).
- LLM winner − T-adil: +0.0058 [+0.0021, +0.0094] (Bonferroni k=4: [+0.0014, +0.0105]), a small real gain.
- **LLM − random winner: +0.0018 [−0.0011, +0.0046]. Random search was enough** (also on selection: +0.0017
  [−0.0010, +0.0048]).
- 3 of 7 observer proposals were invalid (unparseable JSON), all repaired; invalid+repeat rate 0.43 (falsification
  threshold 0.5).

**What this means:**
- The loop is honest machinery, not a source of gains. It finds real but small improvements when the space contains
  them.
- With about 700 dev units, repeated looks at dev can produce a gain that does not survive on test (setting A).
  That is the reason for the selection split, the single test run and the Bonferroni line.
- On these tasks the LLM observer was not clearly better than seeded random search.
- Plan for frequent invalid JSON from a small observer model; the repair attempt handles it.

**Sealed predictions in use** (`uretim/deney/RAPOR.md`, lines 144 and 259):
- In the Upworthy ek600 sample, the expected ranges were sealed before computation: 1.35x [1.15, 1.60] and
  2.45x [2.20, 2.80]. The results, 1.34x and 2.63x, fell inside them.
- In a third independent sample, 5 of 5 sealed expectations fell inside their ranges.
- A seal is the author's own timestamp. `arastirma/tahmin/notlar/karsit-p22.md` (via p22 §3) suggests also posting the
  hash somewhere external.

**Synthetic checks** (this module's tests and a scratch measurement, 22 Oct 2026). The objective is the one in
`tests/test_research_loop.py`: per-unit difficulty is shared by all configs, so the paired comparison is tight.
- **False keeps under the null.** 1000 independent one-turn loops where no config has any effect kept a change 37 times
  (3.7% per turn). The nominal one-sided rate is 2.5%; the percentile bootstrap at n=240 is slightly anti-conservative.
  Across 12 turns that adds up to roughly a 1 in 3 chance of at least one false keep, which is why the final test uses
  Bonferroni.
- **Proposers on the noisy quadratic.** The start config has a true excess loss of 0.906; lower is better.

  | proposer | runs | true excess loss of the winner | proposal turns | runner calls per run | beat the start on test |
  |---|---|---|---|---|---|
  | random | 10 seeds | mean 0.144 (range 0.043 to 0.553) | 8.3 (mean) | 13.3 (mean) | 10 of 10 |
  | grid, 5 points | 1 | 0.255 (stuck on its grid) | 10 | 14 | yes |
  | greedy coordinate | 1 | 0.004 | 12 | 17 | yes |
  | halving | 10 seeds | 0.005 | 6.1 (mean) | 52 (mean) | 10 of 10 |

  Halving spends its extra runner calls on dev screening, so it takes more looks at dev.
  On a smooth objective, greedy works best. Grid is limited by its resolution. Random is cheap but varies a lot.

## Limits
- A turn changes only the configuration. A runner whose source changes between sessions is refused on resume (its
  source fingerprint is checked). Changing the space or the start config also needs a new workdir.
- The paired CI needs the same unit ids across configs. Units missing from one run are dropped from that comparison.
  With fewer than about 50 units, CIs get wide and nearly every turn is discarded.
- The simplicity rule is on by default (`simpler_ties=True`), as in the original loop (`dongu/kod/dongu.py`): a
  candidate whose paired CI against the incumbent contains 0 and that has fewer components is kept. It only acts
  when the space defines `Space(complexity=...)`; pass `simpler_ties=False` to keep strictly-better changes only.
- The test-reference filter is a regex. It is a guard against the observer talking about test results, not proof that
  no leak happened. The loop never shows test numbers to any proposer.
- `ClaudeCodeProposer` and the `"claude"` proposer call a real model. The tests use only `FakeLLM` and a stubbed
  subprocess.

## Minimal usage
```python
import simengine as se
from simengine.research import Loop, Space, Prereg, lock_prereg, seal

Prereg(title="knn vs prior", question="Does kNN beat the position prior?", data="survey questions, wave split",
       split="dev = old waves, test = new waves", metrics=["wd (lower better)"], baselines={"prior": "method=prior"},
       expected="winner wd below prior on test", falsification="winner does not beat prior on test (Bonferroni CI)"
       ).write("prereg.md")
lock_prereg("prereg.md", git_commit=True)

def runner(cfg):                      # fixed code; reads cfg["split"]
    df = score_units(cfg)             # your task: DataFrame[unit, wd]
    return {"per_unit": df}

space = Space({"method": ["prior", "knn"], "k": (1, 200), "gamma": (0.0, 20.0)},
              defaults={"method": "prior", "k": 20, "gamma": 1.0})
loop = Loop(runner, space, proposer="claude-p", primary="wd", higher_is_better=False, stop_after=3, max_turns=12,
            baselines={"prior": {"method": "prior"}}, select_split="select", prereg="prereg.md",
            task="Predict subgroup answer distributions ...", workdir="runs/knn")
out = loop.run()                      # resume: run the same lines again
print(out["final"]["test_diff"], out["final"]["verdict"])
ctrl = se.research.control_arm(loop, "random", workdir="runs/knn-random").run(finalize=False)

seal(predictions_df, "forecast before truth", registry_dir="prereg")   # then grade_sealed(...) once truth exists
```

## Seal: research vs evidence

The top-level `se.seal` is `simengine.evidence.seal`, which writes `.onkayit/zincir.jsonl`.
`simengine.research.seal` is unchanged: it has its own schema and writes `.prereg/chain.jsonl` with English keys.
`verify_seal` and `grade_sealed` also read an evidence-format `zincir.jsonl` (written by `se.seal`) in the same
directory, or in the sibling default directory when `.prereg` holds no chain. Those records get English aliases
(`seq`, `time_utc`, `rows`, `description`, ...) and `"chain": "zincir.jsonl"`. `verify_chain` still checks only
this module's `chain.jsonl`. Example: `se.seal(df, "x"); se.research.verify_seal(df)["matched"]` is True.
`simengine.evidence.evaluate(prereg=dir)` and `evidence.chain(dir)` read both formats, so a research seal is attached
to the evidence report too:
```python
from simengine.research import seal
import simengine.evidence as ev
seal(pred_df, "forecast before truth")                          # .prereg/chain.jsonl
rep = ev.evaluate(df, ours="llm", prereg=".prereg")             # seals listed on the card
```
