# simengine.evidence

The evidence layer. It takes one long table of real outcomes and predictions and returns a report: an honest
baseline ladder, paired cluster-bootstrap confidence intervals, calibration, segment error, leakage and
memorization audits, a PPI++ "worth N humans" multiplier, an evidence card (Markdown/JSON) and a single-file
HTML dashboard.

It is a faithful port of the `kanit` package from the TALP prep nights (branch `kanit`, `kod/kanit/`).
The algorithms, defaults and seeds are unchanged. `tests/test_evidence_equivalence.py` imports the original
package when it is present on the machine and checks that the reports are **identical**, apart from the
creation timestamp. The check covers continuous data with a ceiling, binary data with a train/test split,
distribution data and multiclass data with a label baseline, plus the PPI++ points, the multiplier block,
the canonical seal bytes, the card dict and the dashboard HTML. All 28 original tests are ported
(`tests/test_evidence_{ezber,kart,onkayit,orneklem,sema_sinif}.py`) and pass.

Provenance shorthand used below: **[kanit RAPOR §x]** = `uretim/kanit/RAPOR.md`, **[kanit RECETE n]** =
`uretim/kanit/RECETE.md` "what we measured" item n. Every number below comes from those files. None was
re-measured here, except where marked *(this port)*.

## Minimal usage

```python
import simengine.evidence as ev

# frame: EvalFrame (truth, pred_<method>, group, item, segment, split, retest, ...) or the Turkish aliases
report = ev.evaluate(frame, title="Twin T1", ours="hybrid", raw="llm",
                     unlabeled_baselines=["persistence"],     # label-free rules count as baselines
                     circular=["gbdt_same_pop"])              # trained on the target labels: no "worth N humans"
print(ev.summary(report)["headline"])                         # English flat view of the headline numbers
ev.card(report, "card.md")                                    # Markdown card ("card.json" -> card dict)
ev.dashboard([report], "pano.html")                           # single HTML file, inline SVG, light/dark
ev.save_json(report, "report.json")

ev.worth_humans(frame, prior="llm", n=100)                    # PPI++ points, one target per item
ev.seal(predictions, "before truth", registry=".onkayit")     # pre-registration, BEFORE the truth is opened
```

Name `ours` before looking at the results. If it is omitted, the last method is used and the audits carry
a "selection bias" warning.

## Input

One row per evaluated unit. The English names come from `simengine.types` (EvalFrame). The Turkish names
of the research code are accepted as aliases.

| English | Turkish alias | role |
|---|---|---|
| `truth` | `gercek` | required: 0/1, class label, number or distribution (list / array / JSON) |
| `pred_<m>` | `tahmin_<m>` | at least one: probability, label, probability vector, number, distribution, score |
| `group` | `grup` | bootstrap cluster and fold unit (person). Cluster priority: group > item > row |
| `item` | `madde` | question / test id: per-item baselines, A/B tests, within-item metrics |
| `segment`, `seg_*` | same | segment error analysis; the first segment also gives the demography baseline |
| `split` (`train`/`test`) | `bolum` (`egitim`/`test`) | metrics on test rows only; train rows fit the baselines |
| `retest` | `tekrar` | second human measurement of the same unit: human ceiling |
| `time` | `zaman` | temporal leakage check and the contamination audit (`cutoff=`) |
| `unit` | `birim` | duplicate-unit check |
| `weight`, `fold`, `arm`, `options`, `reverse_<m>` | `agirlik`, `kat`, `kol`, `secenekler`, `ters_<m>` | weights, CV folds (group-disjoint check), A/B arm, option names, position-swapped prediction |

The task is detected automatically (`binary`, `multiclass`, `distribution`, `continuous`, `ab`), or you
can pass it with `task=`.

## Report

`evaluate` returns the original kanit summary dict. Its keys keep their Turkish names, so that the
dashboard and the equivalence test work unchanged. `ev.summary(report)` gives an English flat view.
The main keys are:
`manset` (headline: ours, raw, best baseline, best other method, ceiling, gap closed, ratio to ceiling),
`merdiven` (the ladder, sorted on the primary metric, with the paired difference to the previous rung),
`sonuclar` (every column x metric with a 95 % CI), `karsilastirmalar` (paired differences ours − X),
`kalibrasyon`, `segment`, `yayilim` (spread), `denetim` (audits: `tamam`/`uyari`/`hata`/`atlandi` =
ok/warning/error/skipped), `orneklem` (multiplier), `kanit_kart` (evidence card), `sinirlar` (limits).

## Components

### Metrics (`metrik.py`; `metric_family`, `compute_metrics`, `hit_at_k`)
The metric family is chosen per task:
- **binary:** accuracy (0.5 earns half a point), balanced accuracy, kappa, AUC, macro-F1, Brier, log loss, ECE.
- **multiclass:** accuracy (ties shared), balanced accuracy, kappa, macro-F1, plus Brier, log loss and ECE
  when the predictions are probability vectors.
- **distribution:** 1 − TVD, SimBench score 100·(1 − TVD/TVD_uniform), ordinal Wasserstein / (K−1),
  Jensen-Shannon, KS, mode match.
- **continuous:** 1 − MAE (when the truth lies in [0,1]), Pearson, Spearman, MAE, RMSE, R², and with
  `item` also within-item correlation and de-meaned R².
- **ab:** winner top-1, pair-sign accuracy, within-test Spearman, effect correlation, scale error.

Default primary metrics: accuracy, 1 − TVD, Pearson (1 − MAE on [0,1] data), pair sign. When the majority
class share is above 60 %, macro-F1 replaces accuracy automatically.
- *Works:* the imbalance switch exists because accuracy rewards "always majority". OE class has a majority
  share of 70 %, so its primary metric became macro-F1 [kanit RAPOR §1b, RECETE 3].
- *Limit:* the pack has no between-person correlation metric as a card metric. On Twin, 1 − MAE cannot see
  person-specific signal. With the fair persona baseline, between-person correlation goes from 0.342 to
  0.347 when the LLM is added, but this appears only in a footnote [kanit RAPOR, KARŞIT-2].
- `hit_at_k(truth, score, item, k)` is a standalone ranking metric. It is not part of `evaluate`. It is
  ported from the next-action ranker evaluation (`web/kod/yontem/siralayici.py`, `topk_isabet`).

### Bootstrap and paired CIs (`bootstrap.py`; `Resampler`, `ci`, `p_value`, `paired_ci`)
Cluster bootstrap with resampling by group (person), falling back to item and then to row. Every column and
every metric uses the **same** resamples, so method differences are paired. Per-unit metrics use the fast
cluster-sum path with B = 1000. Aggregate metrics (AUC, F1, correlations) are recomputed in a loop with
`B_aggregate = 200`. Intervals are 2.5/97.5 % percentiles. The p-value is two-sided.
- *Limit:* the CI generalises to **new people, not new questions**. On Twin the gap closed is [16–19] with
  person clusters and [2–35] with task clusters [kanit RECETE 5, RAPOR §6]. A two-way (person × item)
  bootstrap is not written. If the event brings new questions, read the item axis separately.

### Baseline ladder (`merdiven.py`; `baseline_ladder`, `shuffle_control`)
The ladder is cross-fitted: train rows are used when a split exists, otherwise 5 group-disjoint folds keyed
by crc32 of the group name. Each unit's baseline comes from folds that did not see its group. Rungs:
**random** (0.5, uniform, within-item permutation, or ties for A/B), **marginal** (item mean or median, or
class shares; leave-one-out for distributions; the arm-majority direction for A/B), **demography**
(item × first segment, shrunk toward the item with k = 10). Label-free rules you supply (for example "what
they did last year") join as baselines through `unlabeled_baselines`. A **shuffle control** permutes "ours"
within each item. It keeps the marginal and removes the person information.
- *Works:* the strongest baseline is often not the LLM. On open e-commerce (OE class), label-free
  persistence reaches macro-F1 0.781 and a label-trained LightGBM reaches 0.783 [0.756–0.809]. The paired
  difference is 0.001 [−0.022, 0.023]: the model is only as good as the baseline [kanit RAPOR §1b,
  RECETE 1].
- *Works:* on Twin-2K-500, ratio to ceiling alone misleads. The best method reaches 94 % of the ceiling, but
  the demography baseline alone reaches 93 %. The gap closed (normalised score) is **17 % [16–19]**. The
  headline therefore leads with the gap closed [kanit RAPOR §1b, RECETE 2]. Twin persona controls: ours −
  item mean 0.013 [0.012–0.014], ours − shuffled persona 0.020 [0.019–0.021].
- *Limit:* without `retest` there is no ceiling and no gap closed. The card then says "no ceiling" and does
  not invent one (OE has no retest) [kanit RAPOR §6].

### Calibration (`kalibrasyon.py`; `recalibrate`, `reliability`)
Cross-fitted recalibration, which goes through the same bootstrap as the raw column:
- binary: Platt and isotonic
- multiclass: temperature
- continuous: linear and isotonic
- distribution: temperature on JS

Outputs are the reliability curve (10 bins), ECE, confidence AUROC, AURC and the coverage-accuracy
(abstention) curve.
- *Measured:* on the Upworthy experiment, the stacked model is 0.960 accurate on its most confident half in
  both samples. The experiment CIs are [0.920–0.987] and [0.937–0.980] [kanit RAPOR §1a].
- *Limit:* the real evidence datasets had no probability outputs, so no coverage-accuracy curve was drawn
  for them (synthetic example only) [kanit RAPOR §5].

### Segments and spread (`analiz.py`)
Per-segment metric with CI and bias, the worst three segments, the correlation of segment-mean differences,
and the spread ratio (sd of prediction / sd of truth within item; < 0.8 = homogenisation, > 1.5 =
over-dispersion). It also gives the entropy ratio and W1.
- *Measured:* Twin sd ratio 0.75 → homogenisation warning [kanit RAPOR §1b].

### Audits (`denetim.py`, `ezber.py`; `split_audit`, `memorization_audit`)
The automatic audits are:
- group overlap train ∩ test (an error if any group overlaps), group-disjoint CV folds, temporal leakage,
  duplicated units
- ours vs item mean, ours vs strongest baseline, ours vs strongest other method, persona shuffle
- person-specific signal (within-item correlation, de-meaned R² > 0)
- a method beating the human ceiling (leakage sign), collapse of spread, class imbalance
- contamination before/after `cutoff` (bootstrap CI), position swap (`reverse_<m>`), selection bias

The memorization audit uses the completion gap: hit rate on real items minus hit rate on control
(scrambled) items, with an item-bootstrap CI. A lower bound above 0 gives a warning. At most 500 calls.
- *Rule:* a skipped critical audit (split, group, time, contamination, memorization) is **not neutral**.
  The card then carries the flag "kanıtlanmadı: sızıntı/ezber" (not proven: leakage/memorization)
  [kanit RAPOR §1c].
- *Measured:* Twin had group-disjoint folds (2,058 persons, 0 overlap), and no method exceeded the ceiling.
  It got a warning for de-meaned R² −0.084 [−0.095, −0.074]: person deviations are not yet predicted better
  than the item mean [kanit RAPOR §1b, RECETE 10].
- *Not yet run on real data:* the memorization and contamination audits never ran on the example data. All
  four example cards are yellow for that reason [kanit RAPOR, "Açık süreçler"].
- *Port change:* the original called `claude -p` in a subprocess. Here the caller is any `simengine.types.LLM`
  (Claude, ClaudeCode or FakeLLM) or a `str -> str` callable. Caching, language settings and budget belong
  to the `simengine.llm` backend. RECETE 11 says `claude -p` inherits the user's language setting, so pass
  English explicitly.

### Worth N humans: PPI++ (`bedel.py`, `orneklem.py`; `worth_humans`, `sample_multiplier`)
The target is a population (item) mean. You have n real answers, plus a simulation prior f for everyone.
The PPI++ estimator is λ = Cov(y,f) / ((1 + n/N)·Var(f)), computed from the labelled sample and clipped to
[0,1], and θ̂ = ȳ_n + λ(f̄_N − f̄_n). The multiplier is MSE(classical) / MSE(PPI++), using the closed-form
classical variance and Monte Carlo for PPI++. Theory gives 1/(1 − ρ²). "≈ humans" is the finite-population
equivalent sample size. CIs use an item bootstrap with ≥ 20 items, and a unit bootstrap below that.
`orneklem.carpan` adds a Bayes-shrinkage estimator with cross-fitted n0, and the "simulation alone"
equivalent. Both are hidden when there are fewer than 20 items.

Measured at n = 100 [kanit RAPOR §1a]:

| prior | ρ | multiplier [95 % CI] | theory | ≈ humans |
|---|---|---|---|---|
| persistence, "what they did in 2021" (label-free; OE) | 0.886 | **4.36× [3.79–5.33]** | 4.66 | 377 |
| external LightGBM trained on past purchases only (OE) | 0.891 | ≈4.5× (4.53; no CI) | 4.85 | |
| stated intent in the survey (collected after the target: upper bound) | 0.608 | 1.60× [1.41–1.75] | 1.59 | 156 |
| LLM predictor + stated intent (500 people) | 0.666 | 1.70× [1.49–2.03] | 1.80 | 149 |
| LLM persona predicts behaviour | 0.302 | 1.06× [0.97–1.22] | 1.10 | 105 |
| LLM persona answers the survey | 0.240 | 1.02× [0.95–1.15] | 1.06 | 101 |
| LLM twin GPT-4.1 (Twin-2K-500) | 0.284 | 1.08× [1.06–1.10] | 1.09 | 107 |

- *Works when* the prior correlates strongly with the truth **within** the item. The only lever is ρ.
  Measured points sit on or just below the theory curve [kanit RAPOR §1a].
- *Does not work* for LLM personas answering surveys (about 1.0×). The LLM adds nothing on top of the stated
  intent: 1.69× for the statement alone vs 1.70× with the LLM [kanit RECETE 9]. Always put a "X alone" arm
  next to every "LLM + X" arm.
- *Circular priors* are excluded with `circular=`: models trained on the same population's target labels
  would inflate the multiplier (for example "OE GBDT 5×"). A prior that uses input collected after the
  target also leaks. A version using a later survey statement showed 5.02× and was withdrawn [kanit RECETE 4].
- *Denominator:* λ must use the labelled-sample variance. With the population variance, OE GBDT at n = 25
  shows 2.0× instead of 5.1× [kanit RECETE 6]. The finite-population equivalent is not n × multiplier:
  4.36× gives 377, not 436.
- *Coverage* of the ±1.96·se interval at n = 100 is **93–94 %**, against a nominal 95 % (intent 92.9–93.9 %,
  persona 93.9–94.0 %; OpinionQA 94.1–94.6 % over 1,491 questions, worst question 0.830)
  [kanit RAPOR §1a].
- *Heavy tails break it:* the worst Twin question (Q166) has coverage 0.475–0.490, caused by a single 5000
  outlier. A t interval does not fix it (0.482). Choosing a 99.5 % winsorised mean as the target does
  (0.949). There is no automatic flag, so seal a trimmed target in advance [kanit RAPOR §1a].
- *Different target:* on Upworthy A/B, the gain in winner selection depends on n. The direct model prior
  gives 2.05–2.27× at 100 impressions per arm in two independent samples, 1.17× / 1.12× at 400, and every
  prior gives 0.88–0.99× at 3,200 (all CIs include 1). The value is in early decisions. Never plot it on the 1/(1 − ρ²) curve [kanit RAPOR §1a, RECETE 8].

### Evidence card (`kart.py`; `card`, `card_markdown`, `badge`, `panel_cost`)
The card has six rows: strongest simple baseline → raw → ours → human ceiling → gap closed → worth N humans
(ours at n = 100; if ours is circular, the raw arm's value with a note). The badge rule:
- `bekliyor` (awaiting truth) when there is no truth yet
- `kararsiz` (inconclusive) when the upper bound of ours − baseline is below 0
- `yuksek` (high confidence) only when the CI lower bound is above 0 and the difference is ≥ 0.02; if ours has
  its own worth-N value, its lower bound must also be ≥ 1.10× and above the baseline's lower bound
- another column's multiplier never supports the badge

`panel_cost` converts extra answers to a public panel price ($0.95–2.86 per answer, a calculation).
- *Measured:* under this rule none of the four example cards (Twin, OE value, OE class, OE LLM-500) is
  "high". The methods do not beat the strongest simple baseline by more than 0.02 [kanit RAPOR §1c].
- *This port:* `card()` writes Markdown (the original had no Markdown card) or, for a `*.json` path, the
  original card dict.

### Dashboard (`pano.py`; `dashboard`, `dashboard_html`)
One or more reports become one HTML file: inline SVG, no external script, style or font, light/dark theme,
phone width. It shows the card, headline, multiplier curve, ladder, metric table, paired differences,
calibration, segments, spread, audits, examples, pre-registration chain and limits. The labels are in
Turkish, as in the original.
- *Measured runtime:* `--hizli` (B = 200) on synthetic binary data took 8.6 s for the report. With full
  B = 1000, OE (2,174 persons) took 23–37 s per dashboard, and Twin (2,058 persons × 122 questions) took
  179–219 s at server load ~120 [kanit RAPOR §3].

### Pre-registration (`onkayit.py`; `seal`, `verify`, `verify_chain`, `grade`, `chain`, `canonical_bytes`)
Before the truth is opened, the prediction table is sealed. The seal is the SHA-256 of a canonical CSV
(independent of row and column order, with floats fixed to 10 significant digits), plus an ISO timestamp, a
hash chain and a gzip archive, with an optional git commit. `grade` joins the table with the truth only if
the seal matches. Pass `prereg=<registry>` to `evaluate` to attach the chain to the report. This lives here
because it is part of kanit. The research module's own lock and loop are separate.
- *Measured:* an Upworthy expectation was sealed before computation (seal bad4549ab3a5). Both results fell
  inside the sealed range [kanit RAPOR §1a].

## When not to trust a number
- Few items (< 20): only PPI++ is shown. Bayes shrinkage and "simulation alone" depend on hitting a single mean.
- A method above the human ceiling: check leakage first.
- `ours` chosen after looking at several methods: the result is optimistic. Validate on independent data
  (selection-bias audit).
- New questions on event day: the person-cluster CI is too narrow.

## Port notes
- Verbatim ports: `sema, metrik, bootstrap, merdiven, kalibrasyon, denetim, analiz, orneklem, bedel, kart,
  onkayit, cekirdek, pano`. The English API is in `api.py`.
- Changes:
  - `ezber` uses the LLM protocol instead of a `claude -p` subprocess, and its CLI is dropped.
  - `denetim.ezber` closes its file handle.
  - Docstring and comment paths were neutralised.
- Not ported:
  - the `python -m kanit` CLI (`rapor`, `birlestir`, `bedel`), which belongs to the integrator's
    `simengine evaluate|card` (`worth_humans(df)` reproduces `kanit bedel`)
  - the dataset adaptors (`adaptor/oe.py`, `adaptor/twin.py`, which belong to bench)
  - `kod/lig.py`, which is outside the package
- `python -m simengine.evidence.onkayit {muhurle|dogrula|zincir|notla}` still works.
- Tests *(this port)*: 49 tests run in about 28 s (the 28 ported tests, 15 API tests and 6 equivalence
  tests, which skip when the original is absent).
