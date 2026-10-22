# bench task `swissmetro`: Swissmetro stated mode choice

`simengine/bench/tasks/swissmetro.py` turns the Swissmetro stated preference survey (Bierlaire, Axhausen and Abay
2001, the standard example data of Biogeme) into a three-way choice task: train, Swissmetro (a planned underground
mag-lev) or car. Each of 1,192 respondents answered nine choice situations that varied door-to-door time, cost,
headway and seat layout. The module fits the classic discrete choice baselines with scipy, reproducing the Biogeme
example coefficients exactly. It also writes a persona text and a prompt per situation, so an LLM method can be scored
on the same held-out rows.

The task follows the `BenchTask` conventions in `simengine/bench/base.py` (`layout`, `load`, `baselines`, `frame`,
`metrics`). It is not in the package registry yet, so import the module directly. Importing it registers the name
`swissmetro`.

## API

```python
from simengine.bench.tasks import swissmetro as S

task = S.SwissmetroModeChoice()                 # kind multiclass; 20% of respondents held out
data = task.load("<catalogue>")                 # or $SIMENGINE_DATA; the directory holding swissmetro/
frame = task.run(data=data)                     # uniform, shares, mnl, mnl_rich, nested, gbdt
task.score_all(frame)                           # loglik (primary), accuracy, rho2, share_tvd
data.meta["fits"]                               # coefficient tables of the fitted logit baselines

d = S.prepare(S.read_raw("<catalogue>/swissmetro"))   # Biogeme filter + derived variables
fit = S.fit_mnl(d)                              # Biogeme b01 logit; fit.table(), fit.loglik, fit.rho2
nest = S.fit_nested(d)                          # Biogeme b09 nested logit (train + car nest)
rich = S.fit_mnl(d, S.RICH)                     # any (parameter, alternative, column) term list

S.persona_text(row); S.situation_text(row); S.prompts(df)
task.run(methods={"llm": S.llm_method(llm)}, data=data)   # one prompt per test choice, parsed to probabilities
```

| option | values (default) | notes |
|---|---|---|
| `sample` | `"biogeme"` \| `"all"` (`"biogeme"`) | `biogeme`: commuter or business trips (PURPOSE 1 or 3) with a known choice, 6,768 choices of 752 respondents; `all`: 10,719 choices of 1,191 |
| `split` | `"respondent"` \| `"situation"` (`"respondent"`) | see Split |
| `test_frac` | float (0.2) | share of respondents held out |
| `seed` | int (0) | respondent draw and LightGBM seed |
| `n_test` | int (2) | last situations per respondent held out with `split="situation"` |
| `lgbm` | bool (true) | include `gbdt` |
| `nested` | bool (true) | include `nested` |

An unknown option key raises `ValueError`, as in every bench task.

Other public names: `ALTS` (`("train", "sm", "car")`), `B01`, `RICH`, `NESTS_B09`, `GBDT_FEATURES`, `LogitFit`,
`design`, `availability`, `mask_probs`, `respondent_split`, `situation_split`, `gbdt_probs`, `parse_choice`, `SYSTEM`,
and the code tables `PURPOSE`, `TICKET`, `WHO`, `LUGGAGE`, `AGE`, `INCOME`, `CANTON`.

### Files

`<root>/swissmetro/swissmetro.dat` is the Biogeme file (tab separated, 10,728 rows, 28 columns). The file may also
sit in `<root>/swissmetro/ham/`.

### Preparation

`prepare` follows the Biogeme example script `swissmetro_data.py`. Rows with `CHOICE = 0` are dropped, and the
`biogeme` sample also keeps only PURPOSE 1 or 3. Train and Swissmetro cost are set to 0 for GA (annual pass) holders.
Times, costs and headways are divided by 100 (`*_SCALED`). Train and car availability are multiplied by `SP != 0`.
The function adds `y` (`CHOICE − 1`: 0 train, 1 Swissmetro, 2 car), `av_<alt>`, `respondent`, `situation` (0-8 within
a respondent) and 0/1 covariates (`ga`, `car_user`, `first`, `male`, `employer_pays`, `luggage`, `age_55plus`,
`income_high`, `business`).

### Split

- **`respondent`** (default): `round(test_frac · n)` respondents are drawn with `default_rng(seed)` from the sorted ids,
  and all their situations go to test. The model sees nobody from the test set, so this measures generalisation to
  new people.
- **`situation`**: each respondent's last `n_test` answered situations are the test set. Every test respondent has
  earlier answers in train, which the `own_shares` baseline uses.

The EvalFrame columns are `unit`, `group` (respondent, the bootstrap cluster), `item` (situation index), `truth`
(0/1/2), `segment` (`car_user` / `rail_user`, i.e. the survey group), `seg_ga`, `seg_purpose`, `av_train`, `av_sm`,
`av_car` and `split`. A prediction is an `(n, 3)` probability array in `ALTS` order. `frame` rejects other shapes,
NaN and negative values.

### Metrics

Unavailable alternatives are zeroed and each row is renormalised before scoring (`mask_probs`).

- **`loglik`** (primary): mean log-probability of the chosen alternative, clipped below at 1e-6.
- **`accuracy`**: argmax over available alternatives, with ties split evenly.
- **`rho2`**: 1 − LL / LL₀, where LL₀ gives equal shares over the available alternatives (Biogeme's "null"
  log-likelihood).
- **`share_tvd`**: total variation distance between the mean predicted shares and the observed shares in the test
  set. It measures aggregate market-share error, not per-person error.

All four use a cluster bootstrap over respondents (`simengine.bench.stats.cluster_bootstrap`).

### Baselines

- `uniform`: equal over the available alternatives.
- `shares`: train market shares, restricted to the available alternatives.
- `mnl`: Biogeme example b01 (`b01a_logit`): ASC_TRAIN, ASC_CAR (Swissmetro ASC = 0), generic B_TIME and B_COST on the
  scaled variables. It is fitted by maximum likelihood with BFGS and an analytic gradient. Classic standard errors
  come from the information matrix, robust ones from the sandwich estimator.
- `nested`: Biogeme example b09 (`b09_nested`): b01 utilities, train and car in one nest ("existing"), Swissmetro alone.
  The nest parameter μ ≥ 1 uses the Biogeme normalisation (upper-level scale 1, P(i | m) ∝ exp(μ V_i)). It is fitted
  with L-BFGS-B and numerical derivatives.
- `mnl_rich`: an MNL that adds alternative-specific times, headway, seat layout and respondent covariates (GA, survey
  group, first class, employer pays, luggage, age 55+, high income, gender, business trip). The specification is
  this module's, not a published one.
- `gbdt`: LightGBM multiclass on raw attributes and respondent codes (300 trees, learning rate 0.05, 15 leaves,
  min_child_samples 50, row subsample 0.8, column subsample 0.9). The parameters are fixed and were not tuned. If
  LightGBM is missing, sklearn's HistGradientBoosting is used instead.
- `own_shares` (`split="situation"` only): the respondent's own train choice counts + 1 (Laplace), restricted to the
  available alternatives.

### Persona text for LLM methods

`persona_text(row)` writes a second-person profile from the respondent columns: gender, age bracket, income band,
origin and destination canton, trip purpose, survey group (rail traveller or car driver), class and ticket for rail
travellers, who pays, luggage, GA ownership, and whether a car is available. `situation_text(row)` lists the available
alternatives with time, cost as the respondent sees it ("covered by your GA" for GA holders on train and Swissmetro)
and headway. `prompts(df)` pairs `SYSTEM`, which asks for a JSON probability object, with persona + situation +
question.

`llm_method(llm, smooth=0.05)` sends one prompt per test row through `llm.map`. `parse_choice` reads the JSON object,
or else the first alternative word in the reply. The parsed vector is mixed with `smooth` of the uniform-over-available
vector, so a confident wrong answer does not score log 1e-6. Unparseable replies become uniform and are counted in
`method.last_unparsed`. The tests run it with `FakeLLM`. No real LLM run has been scored on this task yet.

## Provenance

- Data: Swissmetro SP survey, Sept 1997 – Mar 1998 (441 rail travellers surveyed on board, 751 car drivers reached by
  licence plate). It is distributed by EPFL TRANSP-OR as the Biogeme example file (`transp-or.epfl.ch/data/swissmetro.dat`).
  The download page carries no licence text. The data is not in the repository.
- Paper: M. Bierlaire, K. Axhausen, G. Abay (2001), "The acceptance of modal innovation: the case of Swissmetro",
  1st Swiss Transport Research Conference.
- Reference estimates: the Biogeme example outputs `b01a_logit` (LL −5331.252, null LL −6964.663, ASC_TRAIN −0.7012,
  ASC_CAR −0.1546, B_TIME −1.2779, B_COST −1.0838, robust SE of B_TIME 0.104) and `b09_nested` (LL −5236.900,
  μ_existing 2.054). These are taken from the Biogeme example pages and asserted in `TestRealData`.

## Measured numbers

All numbers below are from runs on the local copy of `swissmetro.dat` (22 Oct 2026). Every run takes under 10 s.
Command (Python, from the repository root):

```python
from simengine.bench.tasks import swissmetro as S
d = S.prepare(S.read_raw("<catalogue>/swissmetro"))
for f in (S.fit_mnl(d), S.fit_nested(d), S.fit_mnl(d, S.RICH)):
    print(f.loglik, f.rho2, f.table())
for opts in ({}, {"split": "situation"}, {"sample": "all"}, {"sample": "all", "split": "situation"}):
    task = S.SwissmetroModeChoice(**opts)
    print(task.score_all(task.run(data=task.load("<catalogue>")), B=1000, seed=0))
```

### Biogeme replication (full `biogeme` sample, 6,768 choices)

| model | LL | ρ² | estimates (robust SE) |
|---|---|---|---|
| b01 MNL | −5331.252 | 0.2345 | ASC_TRAIN −0.7012 (0.0826), ASC_CAR −0.1546 (0.0582), B_TIME −1.2779 (0.1043), B_COST −1.0838 (0.0682) |
| b09 nested | −5236.900 | 0.2481 | ASC_TRAIN −0.5119 (0.0791), ASC_CAR −0.1672 (0.0545), B_TIME −0.8987 (0.1071), B_COST −0.8567 (0.0600), μ 2.0541 (0.1642) |
| rich MNL | −4509.466 | 0.3525 | 19 parameters (`fit.table()`) |

The b01 and b09 values match the Biogeme outputs to every printed digit. The real-data test checks the coefficients
to 4 decimals and LL to 2.

### Held-out scores

Every cell is a test-set value. The bracket is the 95% respondent-cluster bootstrap CI of `loglik` (B = 1000, seed 0).

**`sample="biogeme"`, `split="respondent"`** (seed 0: 5,418 train / 1,350 test choices, 150 test respondents):

| method | loglik [95% CI] | accuracy | ρ² | share_tvd |
|---|---|---|---|---|
| gbdt | **−0.6752** [−0.7602, −0.5981] | **0.7059** | **0.3382** | 0.0326 |
| mnl_rich | −0.6801 [−0.7743, −0.5972] | 0.6859 | 0.3334 | **0.0187** |
| nested | −0.7498 [−0.8314, −0.6822] | 0.6644 | 0.2651 | 0.0292 |
| mnl | −0.7708 [−0.8526, −0.6977] | 0.6696 | 0.2445 | 0.0314 |
| shares | −0.8702 [−0.9283, −0.8144] | 0.5956 | 0.1470 | 0.0692 |
| uniform | −1.0202 [−1.0446, −0.9959] | 0.3656 | 0 | 0.2381 |

With seeds 0-4 (B = 200), the mean ± sd of the test loglik across seeds is: gbdt −0.685 ± 0.036, mnl_rich −0.703 ± 0.053,
nested −0.771 ± 0.032, mnl −0.793 ± 0.034, shares −0.874 ± 0.041, uniform −1.029 ± 0.009. gbdt beat mnl_rich and
nested beat mnl on all 5 seeds. Mean accuracy was gbdt 0.705, mnl_rich 0.693, mnl 0.681 and nested 0.671.

**`sample="biogeme"`, `split="situation"`** (5,264 train / 1,504 test choices, all 752 respondents):

| method | loglik [95% CI] | accuracy | ρ² | share_tvd |
|---|---|---|---|---|
| gbdt | **−0.5394** [−0.5784, −0.5049] | **0.7686** | **0.4758** | **0.0175** |
| own_shares | −0.5833 [−0.6101, −0.5591] | 0.7676 | 0.4331 | 0.0385 |
| mnl_rich | −0.6925 [−0.7312, −0.6534] | 0.6915 | 0.3271 | 0.0200 |
| nested | −0.8035 [−0.8423, −0.7676] | 0.6549 | 0.2192 | 0.0335 |
| mnl | −0.8191 [−0.8607, −0.7806] | 0.6602 | 0.2040 | 0.0286 |
| shares | −0.9194 [−0.9538, −0.8844] | 0.5705 | 0.1066 | 0.0802 |
| uniform | −1.0291 [−1.0404, −1.0183] | 0.3619 | 0 | 0.2086 |

**`sample="all"`** (loglik / accuracy):

| method | respondent split (8,577 / 2,142, 238 test respondents) | situation split (8,337 / 2,382) |
|---|---|---|
| gbdt | **−0.6677** / **0.7222** | **−0.5699** / **0.7573** |
| own_shares | – | −0.6063 / 0.7490 |
| mnl_rich | −0.7305 / 0.6690 | −0.7293 / 0.6637 |
| nested | −0.8181 / 0.6359 | −0.8308 / 0.6335 |
| mnl | −0.8409 / 0.6391 | −0.8456 / 0.6385 |
| shares | −0.9116 / 0.5654 | −0.9338 / 0.5491 |
| uniform | −1.0441 / 0.3557 | −1.0349 / 0.3595 |

What these show:

- **Respondent split:** the textbook b01 and b09 models are far from the ceiling. Respondent covariates (`mnl_rich`) or
  a tree model add about 0.07-0.10 nats per choice.
- **Situation split:** a respondent's own two to seven earlier answers (`own_shares`) beat every attribute-only model
  except gbdt. Stated choices are strongly habitual: 30.4% of the 752 `biogeme`-sample respondents (29.1% of 1,191
  in `all`) chose the same alternative in every answered situation.
- **Bar for an LLM persona method:** on the respondent split it has to beat `shares` (−0.870) to show any signal, and
  `mnl_rich` / `gbdt` (about −0.68) to be useful. The persona text contains the same respondent attributes the
  tabular models use.

## Limitations

- **Stated, not revealed, preference:** Swissmetro was never built. The choices are hypothetical, and the car group
  was reached by licence plate, so the sample is not representative of Swiss travellers.
- **Held-out vs in-sample:** the replication numbers are in-sample, as in Biogeme. The held-out numbers come from one
  split per configuration, and the respondent test set is small (150 people), so the seed sd (about 0.03-0.05 nats)
  is comparable to the gaps between neighbouring baselines. Compare methods on the same seed, with the paired
  bootstrap in `simengine.evidence`, or across several seeds.
- **Unpublished specifications:** `mnl_rich` and the gbdt parameters are this module's choices and were not tuned on
  a dev split. Biogeme's other examples (mixed logit with panel random effects, cross-nested logit) are not
  implemented. Panel effects would likely narrow the gap between the logit models and `own_shares` on the situation
  split.
- **Nested logit fitting:** the nested logit uses numerical derivatives. Its standard errors come from a
  finite-difference Hessian (step 1e-4); only the estimates and LL are checked against Biogeme, not the nested
  standard errors.
- **LLM methods:** no measured LLM numbers yet. `parse_choice` falls back to the first alternative word, so a
  free-text reply mentioning several modes is read as its first mention.
