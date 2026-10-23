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
- **binary:** accuracy (0.5 earns half a point), balanced accuracy, kappa, AUC, average precision (`ap`; aliases
  `average_precision`, `pr_auc`), macro-F1, Brier, log loss, ECE, `mean_bias` (mean prediction − base rate, signed).
- **multiclass:** accuracy (ties shared), balanced accuracy, kappa, macro-F1, plus Brier, log loss and ECE
  when the predictions are probability vectors.
- **distribution:** 1 − TVD, SimBench score 100·(1 − TVD/TVD_uniform), ordinal Wasserstein on a 0-1 scale
  (numeric option positions when given, else equal spacing / (K−1)), Jensen-Shannon, KS, mode match.
- **continuous:** 1 − MAE (when the truth lies in [0,1]), Pearson, Spearman, MAE, RMSE, R², `mean_bias`, and
  with `item` also within-item correlation and de-meaned R².
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
- *Copies are separate items:* when a cluster is drawn k times, every copy of an item that sits inside one
  cluster gets its own item code (`Resampler.satirlar(b, item_codes)`), so item-grouped aggregate metrics
  (`pair_sign`, `winner_top1`, `within_spearman`) never pair A1 with B2 and the cluster weighs k, not k².
  Items spread over several clusters (person × question) stay shared. This holds for cluster = item, a group
  column equal to the item, `block_col`, `paired_ci`, `compare` and the per-segment rows. Before the fix a
  400-test A/B card showed a 0.147-wide pair-accuracy CI with group = item and 0.319 with 20 blocks (binomial:
  0.092); now 0.090 and 0.157 [rehearsal G3 #3]. Simulated coverage of the 95 % CI (1000 synthetic 400-test
  A/Bs, true accuracy 0.70): 0.952 with group = item (was 0.997), paired difference to a baseline 0.946 (was
  0.997); per-segment rows (200 tests, 100 resamples) 0.93 (was 0.998). With `block_col` and no real block
  effect the CI over-covers (1.000, width 0.149 against binomial 0.09): the two-way bootstrap multiplies block
  and cluster draws, so read it as conservative.
- *Limit:* the CI generalises to **new people, not new questions**. On Twin the gap closed is [16–19] with
  person clusters and [2–35] with task clusters [kanit RECETE 5, RAPOR §6]. A two-way (person × item)
  bootstrap is not written. If the event brings new questions, read the item axis separately.

### Baseline ladder (`merdiven.py`; `baseline_ladder`, `shuffle_control`)
The ladder is cross-fitted: train rows are used when a split exists, otherwise 5 group-disjoint folds keyed
by crc32 of the group name. Each unit's baseline comes from folds that did not see its group. Rungs:
**random** (0.5, uniform, within-item permutation, or ties for A/B), **marginal** (item mean or median, or
class shares; leave-one-out for distributions; the arm-majority direction for A/B), **demography**
(item × first segment, shrunk toward the item with k = 10). Label-free rules you supply (for example "what
they did last year") join as baselines through `unlabeled_baselines`. For a continuous task, a one-row item
cannot be permuted within itself (that would copy its truth: MAE 0, the strongest baseline, and a zero division
in the ceiling ratio). Its random value is instead drawn from the other items' rows (a leave-one-out pool). When
every item has one row the rung is labelled "Rastgele (başka maddenin gerçek değeri; madde başına tek satır)".
A **shuffle control** permutes "ours"
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
- *Honest rungs (rehearsal D9 #4a, G4 #13, D11 §8):* with no split column the labelled rungs say "(cross-fitted,
  5-fold by group|row)". A binary marginal without items (or one row per item) is one constant when cross-fitting:
  per-fold training rates are anti-correlated with the held-out fold (marginal AUC 0.383 on a 60-row frame, now
  0.5); with a split column the item's train rate is kept. Constant truth on the scored rows: the fitted marginal /
  demographic rungs drop (warning row naming only the rungs really dropped) only when their fitting rows share the
  constant, so they cannot copy it and win the badge; the continuous random rung (it shuffles the test truth) drops,
  0.5 / uniform stay. The demographic rung is skipped when no item × segment cell is seen in fitting (segment = unit),
  checked for binary / multiclass / continuous only. Hard-label multiclass with more than `ETIKET_ESIK` = 200 classes:
  the rungs are labels (uniform random label; item or item × segment mode of the training folds, ties broken by
  global frequency), not n × K probability vectors (20,000 rows × 5,000 classes, B=200: 237 s / 4.8 GB -> 13.4 s /
  205 MB with the bincount kappa / balanced accuracy / macro-F1 of `metrik`).

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
over-dispersion). It also gives the entropy ratio and W1. Binary probability predictions use the Bernoulli sd
instead, and distribution tasks add an entropy CI and a between-cell divergence ratio (see the G4/G7 section below).
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
- *Ours next to the strongest baseline:* `bedel.worth_vs_baseline(frame, "ours", "T2", n=100)` (or one or a list
  of `(y, f_ours, f_base)` triples; `from simengine.evidence import bedel`, not exported at package level yet)
  measures both multipliers on the same labelled draws and the same bootstrap
  resamples. It returns `multiplier_ours`, `multiplier_baseline`, their CIs and "≈ humans", the paired
  `difference` with `difference_ci`, and `same_as_baseline` (True when that CI includes 0). `bedel.section(r)`
  turns it into an extras section ("worth ~X humans; strongest baseline alone ~Y; difference ..."), rendered
  with `section_markdown` / `section_html`. In rehearsal G2 the card's 1.51× came from the LLM-free stated-intent
  arm, so the same multiplier held for that arm alone. `olc` and the card's default row are unchanged for now.
  Pass the rows the card scored (e.g. the test split): for a single item at the same seed the ours figure equals
  the card's row. The baseline must not be fitted on the target labels (circular, see below), or ours reads
  "below baseline" against an inflated multiplier. In simulation the paired difference CI covered
  95-98 % with the unit bootstrap (1-3 items, incl. skewed and 0/1 truth) but only 92-93 % with the item
  bootstrap at 25 heterogeneous items: there a CI edge just past 0 is weak evidence. Each item needs at least
  2n complete rows (n = 100 → 200); pass a smaller `n` on a small test split.
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
- `kararsiz` (inconclusive) when the upper bound of ours − baseline is below 0. The internal code stays
  `kararsiz` (research rule), but `summary()["badge"]` and the Markdown card report this case as
  **`below baseline`** (`summary()["below_baseline"]` is True), so a quoted badge does not understate a clear loss
- `yuksek` (high confidence) only when the CI lower bound is above 0 and the difference is ≥ 0.02; if ours has
  its own worth-N value, its lower bound must also be ≥ 1.10× and above the baseline's lower bound
- never `yuksek` when the ours − baseline CI has zero width (degenerate bootstrap) or there are fewer than 10
  clusters (`ab`: the smaller of clusters and tests); it drops to `kararsiz` and the reason says why. Fewer than
  10 clusters also adds the audit row "küme sayısı (bootstrap GA)" and a `UserWarning` ("only N tests; CI is not
  meaningful")
- another column's multiplier never supports the badge

The Markdown card (`card(report, path, lang="mixed" | "en")`) also prints an English one-line summary with dot
decimals (`summary()["headline_en"]`), a "Sealed predictions" table (SHA-256 and UTC time of every seal record
attached via `prereg=`), and, when the primary metric ranks or correlates (AUC, Pearson, Spearman, ...), a Limits
note if a cross-fitted average baseline (`taban_marjinal`, `taban_demografi`) lands below chance: a leave-one-out
mean moves against the held-out truth, which is expected and not a bug. `lang="en"` drops the Turkish badge reason
and original headline; ladder and audit labels stay Turkish. `evaluate(prereg=<dir>)` warns when the directory has
neither `zincir.jsonl` (evidence.seal / `se.seal`) nor `chain.jsonl` (research.seal).

Default primary metric per task when `primary` is not given: binary / multiclass `accuracy` (`macro_f1` when the
majority class is above 60 %), distribution `one_minus_tvd`, continuous `pearson` (`one_minus_mae` when the truth
lies in 0-1), ab `pair_sign`. The CLI warns when `--primary` is missing. `primary` takes the English name, a
common alias (`log_loss`, `roc_auc`, `f1`, `brier_score`, ...; case and hyphens ignored) or the internal name; an
unknown name raises a `ValueError` that lists the task's metrics and suggests the closest one.

The card's limits (`kanit_kart["yanilgi"]`) carry every limit note (duplicates dropped, order kept; it used to keep
only the first three, so prereg and dominant-block notes vanished). On a pending card (no truth yet) the baseline row
reads `report["bildirilen_tabanlar"]` and says "declared, scored when truth arrives: T0, T1" instead of "taban yok".
When the report has `persona_alanlari` (e.g. `persona.field_report(...)`), two rows follow the six: `persona`
(fields that reached the prompt; partly missing fields in the note) and `dusen` (dropped fields). On an evaluated
report use `persona.attach(report, field_rep)`: it edits the stored card in place (`kart.persona_ekle`, idempotent)
and adds the `as_limit` line to the limits; rebuilding the card with `kart.kart(report)` would drop the seal record.
The count is in the label and in `sayi` (the value column stays empty, so 6 fields do not print as 6.000); the
Markdown card names the rows "Persona fields" and "Dropped fields". A declared baseline without a label is skipped on
the pending card instead of printing "None".

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
- **Licensed data warning:** by default the registry `.onkayit` is created in the current working directory (usually
  your repository), and the gzip copy is the full prediction table. Keep the registry git-ignored or outside the repo,
  or seal with `store_copy=False` / `archive_dir=` (see "Options added after rehearsal D3").

## Extras: spread, completion, published findings, stylized facts, coefficients, say-do (`extras.py`)

Six card sections for questions the baseline ladder does not answer. They are new code (not part of the kanit
port), built from the ideas in the replicant `analysis/comparison.py` (PaperComparison), the edsl-test
`benchmark/score_wtp.py` weighting, and the Bisbee and homogeneity notes of the social-simulation reading list.
Each function returns a JSON-friendly dict with `kind`, `title`, `columns`, `rows`, `notes` and its own numbers.
`section_markdown(sections)` and `section_html(sections)` render one section or a list (a Markdown table with
bullet notes, or an HTML `<section>` with no external assets) so it can be appended to `card(...)` output or a
page. Missing values are never imputed: they are dropped, counted and named in the notes.

```python
import simengine.evidence as ev
spread = ev.variance_ratio(frame, "llm")                         # EvalFrame: truth vs pred_llm, per item
done = ev.completion_row(by_condition={"default": (50, 50), "cooperative": (50, 7, 0)})
paper = ev.published_findings([{"finding": "vote punish-low", "published": 0.85, "simulated": 0.94,
                                "source": "<paper, table/page>", "tolerance": 0.10}])
facts = ev.stylized_facts_score([{"name": "never_punish_high", "value": 1.0, "weight": 3},
                                 {"name": "punishment_usage", "value": None, "weight": 2}])   # N/A: excluded
coefs = ev.coefficient_recovery(sim_df, human_df, ("y", "x1", "x2"))
open("card.md", "a").write("\n" + ev.section_markdown([spread, done, paper, facts, coefs]))
```

- **`variance_ratio(df, method, *, item, truth, source, value, human, B=1000, seed=0)`**: simulated SD /
  human SD per item (ddof = 1) with a percentile bootstrap CI, and a pooled ratio
  sqrt(Σ simulated variance / Σ human variance), the same pooling as the `yayilim` spread block of `evaluate`.
  Paired input is an EvalFrame (`truth`, `pred_<method>`; rows resampled jointly within item). Unpaired input is
  a long table with `source=<column>` (`method` vs `human` rows in `value`; each side resampled independently).
  Flags: < 0.8 homogenisation, > 1.5 over-dispersion (point estimate). Items with < `min_n` values or zero human SD
  get no ratio and a reason. *Limit:* the CI is over respondents within the given items, not over new items.
- **`completion_row(n_planned, n_completed, by_condition, *, n_parse_failed)`**: completion rate with a Wilson CI,
  per condition and overall, split into valid answers, parse failures (`valid=False`) and silent dropout. With ≥ 2
  conditions it adds a test of equal completion (Fisher's exact test for 2 × 2, otherwise chi-square), because
  dropout that differs by condition biases any between-condition comparison. `completion_from_frame(df,
  n_planned, condition=, valid="valid")` builds it from answer rows.
  - *Why:* in the replicant Ertan vignette run the hyper-cooperative condition finished the regime page for 30/50
    agents and the punishment page for 7/50, against 50/50 in the default and random conditions, and the
    non-completers were not a random subset ([RESULTS.md](../RESULTS.md) §8 row 3; counts re-checked from
    `papers/ertan2009/results/otree_cooperative_raw.json`). On the punishment-page counts this function gives
    0.14 [0.07–0.26] for that condition and chi-square p ≈ 7e-27 *(this port's computation)*.
- **`published_findings(rows)`**: rows of `finding`, `published`, `simulated` (None = not measured), `source`,
  optional `unit`, `tolerance` and `published_ci`. Gives delta = simulated − published, delta / |published|, a
  match flag (tolerance, else inside the published CI) and a summary (measured, missing, mean |delta|, matches).
  It is the replicant `PaperComparison` as a card section; the mean |delta| only makes sense when units match.
- **`stylized_facts_score(criteria)`**: weighted mean of 0–1 scores. A criterion whose value is None or NaN is
  N/A and drops out of numerator and denominator, as in the edsl-test who-to-punish scorer (weights there: 3
  institutional/treatment, 2 behavioural mechanics, 1 distribution shape). Values outside [0, 1] raise: score
  against the human target inside the criterion. `weight_coverage` reports the observed share of the total
  weight, because an index over fewer criteria is not comparable with one over all of them.
  - *Caveat from the source:* the edsl-test index weights were never sensitivity-checked, and ranks inside its top
    four methods were not significant ([RESULTS.md](../RESULTS.md) §8 row 7). Report the index with its seeds and coverage.
- **`coefficient_recovery(sim_df, human_df, formula_cols, *, alpha=0.05)`**: fits the same OLS with intercept on
  both samples (`{"y": ..., "x": [...]}` or `(y, x1, ...)`, numeric columns, listwise deletion counted) and
  compares coefficients with z = (b_sim − b_human) / sqrt(se_sim² + se_human²). Summary: `sign_agreement`,
  `share_differ` and `sign_flip_when_human_significant`; the intercept is left out unless `include_intercept`.
  Classical SEs: collapsed simulated variance gives small SEs, so read `differs` together with `diff`.
  - *Context, unverified:* the reading list summarises Bisbee et al. ("Synthetic Replacements for Human Survey
    Data", Political Analysis) as 48 % of coefficients differing from ANES and 32 % flipping sign, with compressed
    variance. These figures come from an AI-assembled note; check the paper before citing them.

- **`extras.say_do(frame, say_col, do_col, preds=None, *, cluster="person", item="item", on="level", B=1000,
  seed=0, alpha=0.05)`**: two anchors per person, what they *said* (stated answer, intention) and what they *did*
  (observed behaviour). For each prediction column (`"llm"` → `pred_llm`; None = every `pred_*`) it gives the
  distance of the panel mean to each anchor mean, the person-level MAE to each anchor, and the paired difference
  `say − do` (negative = closer to say) for both, with a percentile CI from a cluster bootstrap over `cluster`
  (rows of one person move together). The default `"person"` falls back to `"group"`, the EvalFrame cluster key
  that `survey_frame(level="person")` fills with the persona id, so its frame needs no renaming; a named column
  that is missing raises; with neither column, or `cluster=None`, rows are resampled (noted: too narrow when a
  person has several rows). The verdict reads the CI of the `on`
  difference: *closer to what people say* / *closer to what people do* / *undecided*. An anchor row gives
  mean do − mean say with its CI; if that CI covers 0 the anchors are not apart and every verdict is undecided.
  Coverage (simulated, 40 persons × 3 rows, 300 runs): level and MAE CIs 0.93-0.95, a midpoint prediction gets a
  wrong verdict 5-6 % of the time; when the panel mean sits exactly on an anchor the level CI's far bound is too
  tight (0.84; the difference is pinned at ∓gap there), the bound near 0 and the verdict are unaffected.
  *Several behaviours per person* (an `item` column, `madde` accepted, with 2+ values, e.g. a person × behaviour
  frame): the level is taken per item and averaged with equal item weight, and the anchor gap is given per item
  (one anchor row each, CI at `alpha / k`, Bonferroni); the anchors are apart when any item's gap is. Pooling would
  hide the gap whenever people over-report one behaviour and under-report another (+1 and −1 cancel: a prediction
  that tracks `say` exactly read *undecided*, "anchors not apart"). The `mean` / `mean say` / `mean do` columns
  stay pooled for reference; MAE pools the rows. Put the items on one scale first (z-score each item, as a zMAE
  run does): both the item average and the MAE mix them. `item=None` pools every row; a named item column that is
  missing raises. Simulated (60 persons × 3 items with gaps +1/−1/+0.5, person-shared errors, 400 runs, B = 1000):
  level CI coverage 0.94-0.95, a per-item midpoint prediction gets a wrong verdict 5.8 % of the time; with no gap
  on any of 5 items the anchors read apart 6 % of the time (1 item: 5 %).
  Each prediction carries a `card_line` for the evidence card. Missing values are dropped per prediction and
  counted. Gap G2 #2 (rehearsal: the person bootstrap and the "which anchor" scene were hand-written).
  *Export:* import from `simengine.evidence.extras` for now; the package-level `ev.say_do` lands with the
  `evidence/__init__` change, and the main card does not print the line yet.

  ```python
  from simengine.evidence import extras
  sd = extras.say_do(frame, "said", "did", preds=["llm", "persistence"], cluster="person")
  print(extras.section_markdown(sd)); [r["card_line"] for r in sd["predictions"]]
  ```

Tests: `tests/test_evidence_say_do.py` (9 tests, synthetic, < 1 s): a prediction equal to `say` plus noise reads
closer to say, `do` plus noise closer to do, the midpoint undecided; exact point values; anchors not apart →
undecided; person-cluster CI wider than the row CI when errors are tied within person; `group` (and
`grup`) used as the cluster when there is no `person` column, a missing named cluster raises; items with
opposite say−do gaps read per item (pooled they cancel to "not apart"), one item equals no item column; Turkish
`tahmin_` alias, missing rows, no cluster column, errors and rendering.

Tests: `tests/test_evidence_extras.py` (21 tests, synthetic data, < 1 s): variance ratio 1 on identical data and
0.5 on halved spread (point and CI), unpaired input, missing and degenerate items, Turkish aliases; completion
counts, tests and md/html rendering; published-finding deltas with missing values; N/A exclusion; coefficient
recovery on same vs sign-flipped processes.

## Options added after the port (gap fixes)

All are keyword arguments with backward-compatible defaults; report keys stay Turkish, `summary()` adds English
fields.

**One seal.** `se.seal` is `simengine.evidence.seal` (registry `.onkayit/zincir.jsonl`).
`simengine.research.seal` keeps its own chain (`.prereg/chain.jsonl`). `ev.chain(dir)` / `evaluate(prereg=dir)`
read both formats. Research records keep their English keys and also get `sira`, `zaman_utc`, `satir`, `aciklama`
and `"zincir": "chain.jsonl"`. When a default directory (`.onkayit` / `.prereg`) holds no chain, the sibling default
is tried. `ev.chain_paths(dir)` lists the chain files that were found. `seal`, `verify`, `verify_chain` and
`grade` accept `registry_dir=` as an alias of `registry=` (the name older `se.seal` calls used; passing two
different directories raises `TypeError`). `research.verify_seal` / `grade_sealed` also find seals made with
`se.seal` (the `zincir.jsonl` in the same directory, or in the sibling default directory).
```python
se.seal(pred_df, "before truth")                    # .onkayit/zincir.jsonl
se.seal(pred_df, "before truth", registry_dir=".prereg")   # alias: .prereg/zincir.jsonl
se.research.verify_seal(pred_df, registry_dir=".prereg")["matched"]   # True: reads both chain files
ev.evaluate(df, ours="llm", prereg=".prereg")      # a research.seal chain is attached too
```

**Memorization audit for numbers.** `memorization_audit(..., tolerance=0.02, relative=False)`: with
`relative=True` the tolerance is a share of the expected value (0.05 = ±5 %), for 0-100 scales. An item may carry its
own `"tolerance"`. For aggregates (country means and similar), use `correlation_audit(truth, named, blind, llm=...)`.
It asks for each key twice, once with the study named and once blind. It compares Spearman(named, truth) −
Spearman(blind, truth) with a paired key bootstrap, and gives `durum="uyari"` when the CI lower bound is above
`margin`. Each arm is an answer dict or a `{key}` template. Templates get a "best guess" sentence, so models do not
refuse with "I can't recall". `audit_prompts(study, quantity)` builds both templates. The result can be passed to
`evaluate(memorization=...)`. Defaults: `max_tokens=1024` and `thinking="auto"`, which passes `thinking="off"`
when the backend accepts it (Claude Haiku / Sonnet), so thinking cannot use up the budget of a one-number reply
(`thinking=None` keeps the backend default; any other value is passed as is). An empty reply
(`EmptyResponseError`) or another `LLMError` for one key makes that key unparsed instead of aborting the audit:
`n_failed` counts them and `errors` holds up to five messages.
```python
named, blind = ev.audit_prompts("'<study title>' (<authors>, <journal> <year>)", "mean support (0-100)")
audit = ev.correlation_audit(truth_by_country, named, blind, llm=llm, samples=3)
audit["n_failed"], audit["n_unparsed_named"]
```

**One-call recall probe (`ezber.recall_probe`).** `recall_probe(llm, source, groups, measures, truth=None,
best_estimate_retry=True, max_tokens=2048, tag=None, scale=None, warn_spearman=0.7, warn_p=0.05, reference=None)`
makes one persona-free call: "recall the published
results of `<source>` for these groups and measures" as JSON (`{"remember": ..., "results": {group: {measure: n}}}`).
`recall_refusal` classifies the reply, including refusals inside JSON (`"remember": false`, empty tables, all
nulls), which `looks_like_refusal` never flags because it skips JSON replies. Kinds: `error`, `empty_reply`,
`text`, `unparsed`, `remember_false`, `empty`, `all_null`. When the first reply has no numbers, one retry asks for a
"best estimate" (tag `<tag>:retry`). Those numbers are estimates, not recall. The result has `raw` / `raw_retry`,
`recalls` (only when the first reply gives numbers without `"remember": false`), `refusal`, `used` (`first` /
`retry` / None), `values` and `calls`. With `truth` (`{group: {measure: v}}`, `{group: v}` or a DataFrame indexed
by group) it also has `per_group` MAE, `mae`, `pearson` and `spearman` (pooled over all cells), `spearman_within`
/ `p_within` (groups ranked inside each measure, pooled, with a permutation p), `spearman_by_measure`,
`scale_warning` (0-1 against 0-100) and `audit`, which is the dict `evaluate(memorization=...)` reads. The audit
uses the within-measure Spearman, not the pooled one. Pooling mostly ranks measure levels ("support" above
"oppose"), which takes no group knowledge: in a 12 groups x 3 measures simulation, a probe that knew only the
measure means had pooled Spearman >= 0.7 in 88 % of runs and within-measure 0 % (61 x 7: 86 % and 0 %). `audit` is
`"uyari"` when the within-measure Spearman is at least `warn_spearman` (0.7) and `p_within < warn_p` (0.05),
whichever reply the numbers came from; at most 5 % false warnings by construction (simulated 0-4.5 % for 4 to 61
groups), and a probe that knows the group pattern warns in 96-100 % of runs from 6 groups or 4 groups x 2 measures
up. A high match with too few groups to tell from chance (one measure with 4 groups needs a perfect match) is
`"atlandi"`, as is a table where no measure has 3 comparable groups. A single probe has no blind control, and
general knowledge ("young voters lean left") also ranks groups, so for a controlled test use `correlation_audit`.
While the truth is sealed, pass `reference=` (your own forecast, same shapes) to get `vs_reference`: the same
comparison against your forecast, without an audit. In D6, Haiku first returned `remember: false` with empty
tables, then gave numbers when asked for a best estimate; the team compared them with its forecast by hand.
```python
probe = ev.ezber.recall_probe(llm, "the <survey> <year> wave", groups, ["support", "oppose"],
                              truth=truth, scale="percent, 0-100", tag="probe")
probe["refusal"], probe["used"], probe["per_group"], probe["spearman_within"], probe["p_within"]
ev.evaluate(df, ours="llm", memorization=probe["audit"])
# truth sealed: how far is our forecast from what the model says unprompted?
probe = ev.ezber.recall_probe(llm, "the <survey> <year> wave", groups, measures, reference=our_forecast)
probe["vs_reference"]["per_group"], probe["vs_reference"]["spearman_within"]
```

**Speed on large frames.**
- *Multiplier subsample.* For single-item frames above `multiplier_max_rows=2000` rows, the PPI++ multiplier runs
  on a fixed-seed subsample of that many rows. The report records this in `orneklem["alt_orneklem"]`, the card
  notes it, and `None` uses the whole table. Large frames with `multiplier=True` raise a `UserWarning`.
- *Unit bootstrap.* From 2,000 rows (`bedel.HIZLI_ESIK`, equal to `MULTIPLIER_MAX_ROWS`), `bedel.olc` draws one
  labelled index set per column and reuses it across the unit bootstrap (`hizli_bootstrap: True`). So it applies to
  the 2,000-row multiplier subsample and to larger tables. Below 2,000 rows, the point estimate and the CI are
  identical to the original (`hizli=` forces either path).
- *Aggregate metrics.* Above 5,000 rows, Spearman and the binary aggregate metrics (AUC, macro-F1, kappa, balanced
  accuracy, ECE) are bootstrapped from row multiplicities instead of expanded resamples. The values are the same
  up to float rounding.
- *Measured.* With 42,000 rows, 8 methods and the default B, a continuous/Spearman run takes about 11 s and a
  binary/AUC run about 12 s, against 34 s and 82 s before. The multiplier at 2,000 rows went from 62 s to 6 s.

**Declared baselines and the full ladder.** `baselines_oof=[...]` names prediction columns that were trained on the
target labels but cross-fitted (for example a GBDT or ridge). Like `unlabeled_baselines`, they become `baseline`
rungs, and the badge and headline compare ours with the strongest baseline, declared or automatic. An unknown name
only warns. When nothing is declared and the strongest baseline is random or marginal, the following are added: an
audit "yalnız basit tabanlar", a limit line, the card flag "yalnız basit tabanlar", and
`summary()["only_trivial_baselines"]`. `report["tabanlar"]` lists declared, automatic and strongest baselines. The
ladder now lists every `pred_` column, where the original showed at most 6.
```python
ev.evaluate(df, ours="hybrid", raw="llm", baselines_oof=["ridge"], unlabeled_baselines=["persistence"])
```

**Strongest baseline uncertain (selection noise).** With two or more setting-A baselines, the audit "en güçlü taban
belirsiz mi" (`denetim.taban_belirsizligi`) takes the paired bootstrap difference between the best baseline (by point
estimate) and every other one. When the runner-up is not separated (its CI includes 0), the row is a warning:
"strongest baseline uncertain: B3, B2, B1 within CI". It lists the tied set (`esitler`), ours − baseline in the worst
case over that set (`en_kotu`: per replicate, ours against the best tied baseline, with a paired CI, plus the tied
baseline with the lowest single lower bound), and a selection-noise note with the number of baselines. A separated
runner-up gives an `ok` row. With the warning, the badge is graded against the worst tied baseline (`en_kotu`;
the badge reason says so), the strongest-baseline row notes the tied set and the card
gets the flag "en güçlü taban belirsiz". `fark` and the headline stay ours − the single strongest baseline. The
point gap is the same (the point-best baseline is the closest one); the worst case widens the CI, since each
replicate takes its own best tied baseline. The badge's lower bound is the lower of the two (worst case and single
strongest baseline), so it is never easier than against that one baseline: `ci="basic"` reflects the per-replicate
minimum around the point and can put the worst-case lower bound above the single one. In simulation (4 tied
baselines, 300 rows, 300-600 runs) the badge's lower bound sat above the true worst-case gap in at most 0.7% of runs
(nominal 2.5%), with either CI. The percentile worst-case CI is too low as a two-sided interval (covers about 76-79%
when the baselines are truly equal, 95% when they differ a little); `ci="basic"` covers about 89-94%, but its own lower
bound alone missed in up to 4%. It cannot see a baseline that is not on the card. In the D5
rehearsal, four baselines were tied on dev, the card picked one, and a hidden learned baseline won on test.

**Model/oracle ceiling.** `ceiling_col="pred_oracle"` (or any column name) gives a model ceiling in place of the
human `retest`. The column leaves the method list and gives the ceiling and gap closed for every metric.
`report["tavan_turu"]` is `"model"` and `summary()["ceiling_kind"]` is `"oracle/model"`, where a retest gives
`"human retest"`. The card row reads "Model/oracle tavanı", and the leakage audit is phrased for a model ceiling.
`ceiling_label=` names the ceiling. If a `retest` column is also present, it is ignored and a note says so.
```python
ev.evaluate(df, ours="hybrid", raw="llm", ceiling_col="pred_oracle", ceiling_label="GBDT on all features")
```

**Group-level evaluation.** `aggregate_by="group"` (or any column, such as `"country"`) averages truth, predictions,
retest and model ceiling per group (and per item / split when present). The task becomes continuous and the
bootstrap resamples groups. `report["toplama"]` holds `{by, n_rows, n_groups, n_eval_rows}`, and a limit line notes
the reduction. This is for binary and continuous truth only. `ev.evaluate_aggregate(df, "country", ...)` is a
shorthand. The automatic baselines (overall mean, demographic segment) are then cross-fitted over group rows. The
out-of-fold mean moves against the held-out groups' own mean, so on a rank metric (Spearman) these rungs can be
clearly below 0 even with no information. The limit line says so and `report["toplama"]` has
`auto_baselines_cross_fitted_bias: True`. For a meaningful comparison, declare a group-level baseline that was
cross-fitted the same way (`baselines_oof=` / `unlabeled_baselines=`).
```python
ev.evaluate(df, ours="llm", aggregate_by="country", primary="spearman")   # e.g. 20 countries
```

**Awaiting truth.** Rows with missing truth are dropped, and the note gives how many. When the truth column is
absent or entirely missing (on test rows), `evaluate` returns an "awaiting truth" report with the same keys:
- the badge `bekliyor` (`summary()["badge"] == "awaiting truth"`)
- `gercek_bekleniyor: True`
- `tahmin_ozeti`, the per-method prediction summary (`summary()["predictions"]`): numeric columns get mean / sd /
  quartiles, multiclass tasks and integer class codes (3-20 distinct integers, no task given) get the top labels and
  the number of distinct labels instead of a meaningless mean; with an `item` column also per item (`maddeler`, up
  to 100 items, beyond that only the constant ones) and `n_sabit_madde`, with a "sabit tahmin (sd 0)" audit warning
  for items where a method predicts one value (rank metrics will be undefined there)
- the attached seals; the card header and `summary()["headline_en"]` name **ours' seal**, not the chain's last
  record: description marker (`ours: <name>`) > the seal whose canonical sha256 matches the frame's columns > name
  in the description (unless the seal holds only other methods' `pred_` columns) > a seal holding `pred_<ours>`;
  none of these -> the last record with the note "ours not matched" (`report["bizim_muhur"]`,
  `summary()["ours_seal"]`). An audit row compares the frame with ours' sealed file. The card says when the
  predictions are not sealed
- declared baselines (`unlabeled_baselines=`, `baselines_oof=`) kept in `tabanlar` / `bildirilen_tabanlar`
- an empty ladder and empty results

A truth column that is present but entirely missing also emits a `UserWarning` (often a failed join on `unit`);
frames built with `truth=None` (the `sim.*_frame` helpers) trigger it too, so drop the column to stay silent.

The Markdown card adds a "Predictions (awaiting truth)" table (distribution or top labels, distinct labels, items and
constant items per column, then the constant items by name) and, with an `item` column, a "Predictions per item"
table (mean ± sd, or the top label with its share and distinct count; up to 30 items, constant ones first), in both
`lang="en"` and `lang="mixed"`. The dashboard shows a predictions section, and
`simengine evaluate` accepts such a frame. A frame that has truth but whose `split` (`bolum`) column has no test
row (accepted: `test`, `deger`, `degerlendirme`, `holdout`; for example `split="cv"`) raises
`ValueError("no test rows in split ...")`. It is not reported as awaiting truth.
```python
se.seal(pred_df, "before truth"); rep = ev.evaluate(pred_df, ours="llm", prereg=".onkayit")  # no truth column
```

## Options added after rehearsal D3 (new-item cold start, group residual metric)

Blind rehearsal D3 used only this library for a "new film launch" task (member ratings for films with no ratings;
primary metric = Spearman of the film residual). Its gap list asked for the six options below. All are keyword
arguments with backward-compatible defaults; a report made without them has no new keys. Tests:
`tests/test_evidence_d3_gaps.py` (27 tests, synthetic data and FakeLLM, about 11 s).

**Offset group metric (`offset_col=`).** A per-row offset (for example the member's training mean `um`) is subtracted
from truth, retest, the model ceiling and every prediction before aggregation. With `aggregate_by="movie_id"` the
card then scores the film residual directly: mean(rating − um) per film against mean(pred − um) per film. The task
becomes continuous, `report["ofset"]` / `summary()["offset"]` describe the offset, and a limit line notes it. Rows
with no offset lose their truth (dropped, with a warning). Multiclass, distribution and A/B tasks raise `ValueError`.
```python
# pairs: one row per (member, film); um = the member's training mean rating
rep = ev.evaluate(pairs, ours="stack", raw="persona", offset_col="um", aggregate_by="movie_id",
                  primary="spearman", baselines_oof=["ridge"])          # = film residual Spearman
```

**Sampling-noise ceiling (`noise_ceiling="variance_components"`, older name `"split_half"`).** Needs
`aggregate_by`; both names run the same estimator. A group mean carries within-group sampling noise, so even the true
group value cannot correlate perfectly with it. The ceiling is computed over **every scored group** from variance
components: s2_g = within-group variance (ddof=1; groups with one row use the pooled within variance), reliability =
1 − mean(s2_g / n_g) / var(group means) · (G − 3)/(G − 1) (the last factor is the normal-theory small-sample
correction). Ceiling = sqrt(reliability) for `pearson`, reliability for `r2`, and for `spearman` the expected
Spearman between the true value and the noisy mean under a normal model with the estimated true variance and each
group's own size (Monte Carlo, 400 draws, fixed seed; it differs from Pearson when group sizes are mixed, e.g. with
one-row groups). Other primary metrics (`mae`, `rmse`, `one_minus_mae`) raise `NotImplementedError`. Each bootstrap
replicate recomputes the reliability on the same group resample as the methods, so the ceiling CI is paired with
ours. On synthetic data the estimate sits within about 0.01-0.03 of what a predictor that knows the true group values
reaches (slightly low for ~80 groups of 2-3 rows); such an oracle sits at the ceiling in expectation, so the
"exceeds the ceiling" audit can fire for it by chance.

The earlier split-half estimate (random within-group halves, Spearman-Brown) is still reported as a cross-check,
`noise_ceiling_repeats=200` splits: `yari_korelasyon`, `spearman_brown`, `yari_tavan`, `n_grup_yari`,
`bolme_yayilimi`. It is not used: it leaves one-row groups out although they are scored (ceiling too high), and a
3-row group splits 1 against 2 while Spearman-Brown assumes equal halves (ceiling too low).
`report["tavan_turu"] == "orneklem_gurultusu"`, `summary()["ceiling_kind"] == "sampling noise (variance components)"`,
`report["gurultu_tavani"]` holds `guvenilirlik`, `tavan`, `n_grup` (scored groups), `n_tek_satir` (one-row groups),
`ic_varyans_havuz` and the split-half fields. The card row reads "Sampling-noise ceiling (variance components)"; one-row
groups get a card note. An audit flags methods that beat this ceiling significantly (a leakage sign). A `retest`
column is then ignored (noted). It cannot be combined with `ceiling_col`.
```python
rep = ev.evaluate(pairs, ours="stack", offset_col="um", aggregate_by="movie_id", primary="spearman",
                  noise_ceiling="variance_components")
ev.summary(rep)["ceiling"], ev.summary(rep)["gap_closed"], rep["gurultu_tavani"]["n_tek_satir"]
```
- *One rehearsal run (D3 dev, 300 films, no CI):* half Spearman 0.950 → Spearman-Brown 0.975 → ceiling ≈ 0.987.
  These numbers come from the rehearsal's own split-half script, not from this function.

**Sampling-noise ceiling of an error metric (`se_col=`, D5 #8d / D11 #5c / D3 #7).** `noise_ceiling=` covers
correlation metrics only. When the truth is an estimate with a per-row standard error (a survey share, a model
estimate), `evaluate(df, primary="mae", se_col="se")` (also `"rmse"`, `"wmape"`; `task="forecast"` defaults to
wmape) builds the oracle that knows the latent value: its error against the observed truth is N(0, se), so the column
truth + sign·se·a (a = |z| averaged over 400 seeded draws, sqrt(mean z²) for rmse: the draw-averaged metric) is
routed through the `ceiling_col` machinery as "Model/oracle ceiling (sampling noise (SE column))" with a bootstrap CI
and the gap closed. The analytic floor on the scored rows (test rows, `weight` honoured) sits next to it: mae
sum(w·sqrt(2/π)·se)/sum(w), rmse sqrt(sum(w·se²)/sum(w)), wmape sum(w·sqrt(2/π)·se)/sum(w·|y|).
`report["se_tavani"]` / `summary()["se_ceiling"]` (`analytic`, `value`, `lo`, `hi`, `draws`, `seed`, `n`),
`summary()["ceiling_kind"] == "sampling noise (SE column)"`, a card line. No `aggregate_by` needed. With it the
noise is drawn per aggregated cell (the `aggregate_by` keys): the cell mean carries sd sqrt(Σse²)/n, so every row of
the cell gets the same offset and `analytic` is computed on the cells (`n_hucre` / `cells`). Per-row offsets with
random signs would average to a mae floor shrunk by about sqrt(2/π) (synthetic 200 groups x 40 rows: 0.123 against the
true 0.167; now 0.168). NaN or negative SEs on scored rows (train rows may leave it empty), a correlation primary, or
combining it with `ceiling_col` / `noise_ceiling` raise `ValueError`. Clearly beating this floor points to leakage or
an SE column that overstates the noise; the card limit says so in place of the model/oracle "exceeding it does not
mean leakage". Measured on a synthetic 800-row frame (se uniform 0.2-1.0): rmse ceiling 0.643 [0.627, 0.662] against
the analytic 0.644.
```python
rep = ev.evaluate(df, ours="model", primary="mae", se_col="se", split="test")
ev.summary(rep)["se_ceiling"]["analytic"], ev.summary(rep)["ceiling"]
```

**Call budget first (`correlation_audit`).** `max_calls` is compared with the planned total (prompt arms × keys ×
samples) before any call. The error names the number needed (`pass max_calls=600`). Cached replies are counted,
because the LLM backend has no public cache probe; a re-run from the cache costs nothing. An answer-dict arm needs no
calls and is not counted. In D3 the old check fired after the named arm had spent its 300 calls.

**Seal without a copy (`seal(..., store_copy=True, archive_dir=None)`).** `store_copy=False` writes only the
hash-chain record (`"arsiv": None`). `verify`, `verify_chain`, `evaluate(prereg=)` and `research.verify_seal` /
`grade_sealed` work from the chain alone, and `grade` then needs the predictions passed back in. `archive_dir=`
writes the canonical copy to another directory (for example outside the repo). The record keeps only the file name
and `"arsiv_konum": "dis"`, never the path. `grade(None, truth, keys, registry, archive_dir=...)` reopens the copy and
checks its bytes against the sealed SHA-256 (a tampered copy raises `ValueError`, a missing one `FileNotFoundError`).
The default (copy in the registry) is unchanged. CLI: `python -m simengine.evidence.onkayit muhurle f.csv --aciklama
.. --kopyasiz` or `--arsiv <dir>`. The registry still defaults to `.onkayit` in the working directory: keep it
git-ignored or outside the repo for licensed data.
```python
se.seal(pred, "test predictions, before truth", registry="teslim/.onkayit", store_copy=False)  # chain only
se.seal(pred, "before truth", registry=".onkayit", archive_dir="<outside the repo>")
ev.grade(None, truth, ["pair_id"], registry=".onkayit", archive_dir="<outside the repo>")
```

**Memorization caveat on the badge, blind arm (`ours_uses_llm=None`, `blind_col=`).** When the memorization audit
passed to `evaluate(memorization=...)` is in warning state (for `correlation_audit`: the named − blind Spearman CI lies
above the margin) and ours includes an LLM arm, the badge reason gets "Uyarı: kazanç ezber/tanımadan gelebilir" and
the card flag "kazanç ezber/tanımadan olabilir". `summary()["badge_caveat"]` and the Markdown card (both `lang`
modes) add "Caveat: gain may come from recognition/memorization (...)". The badge itself is not changed. The caveat
is only added when there is a gain: when ours − strongest baseline is ≤ 0 (e.g. badge "below baseline") it is left
out (the audit row stays); with no baseline it is kept. The mixed-language card carries it twice on purpose (Turkish
badge reason + English caveat).
`ours_uses_llm=None` assumes ours uses an LLM whenever a memorization result is passed; `False` turns the caveat off.
`blind_col="pred_ours_blind"` names the blind twin of ours (same pipeline without names or titles). It adds a paired
row ours − blind to the registered comparisons, and `report["kor_kol"]` gives the share of ours' gain over the
strongest baseline that the blind arm keeps. Below half, an audit warns and the same badge caveat is added.
```python
rep = ev.evaluate(pairs, ours="stack", offset_col="um", aggregate_by="movie_id", primary="spearman",
                  memorization=audit, blind_col="stack_blind", baselines_oof=["ridge"])
ev.summary(rep)["badge_caveat"], rep["kor_kol"]["korunan_pay"]
```
- *One rehearsal run (D3 dev):* the title-recognition audit gave named 0.779 vs blind 0.169, difference +0.610
  [+0.493; +0.720]; ours − ours-blind was +0.046, which was the whole of the gain over the baseline. The old badge
  still read plain "high confidence".

**Arbitrary paired differences (`paired_diff`, `evaluate(compare=)`).**
`ev.paired_diff(frame_or_report, a, b, *, metric=None, aggregate_by=None, group=None, B=2000, seed=0,
offset_col=None, task=None)` returns `diff` (raw metric(a) − metric(b)), `lo`, `hi`, `p`, `n` (clusters), `cluster`,
`a_better`, and the two values. With a frame it runs a cluster bootstrap with B resamples for every metric type.
`aggregate_by` averages per group first, so the clusters are the groups. `group=` chooses the cluster column; with
`aggregate_by` it must be constant within each group. `b=None` gives a's own value and CI (e.g. "P2 ≈ 0"). With a
report it looks the pair up among the registered comparisons and the ours − X comparisons, in either order.
`evaluate(compare=[("P1", "L1"), ("P2", None, "c: P2 ≈ 0"), ...])` computes each pair on the ladder's bootstrap
resamples. The results go to `report["kayitli_karsilastirmalar"]`, `summary()["comparisons"]` (with a verdict: a
better / b better / unclear; a `(a, None)` row has no direction: `a_better` is None and the verdict is above 0 /
below 0, also on the dashboard, whatever the metric direction), a "Registered comparisons / drop conditions" table on the Markdown card, the card dict
(`kayitli`) and the dashboard's paired-difference table. An unknown column raises `ValueError`. This ports the
rehearsal's `esli()` (film-clustered bootstrap, B = 2000); the point difference is the same.
```python
ev.paired_diff(pairs, "P1", "L1", aggregate_by="movie_id", offset_col="um", metric="spearman")
rep = ev.evaluate(pairs, ours="stack", offset_col="um", aggregate_by="movie_id", primary="spearman",
                  compare=[("P1", "L1", "b: P1 − L1"), ("P1", "P_nopersona"), ("P2", None, "c: P2 ≈ 0")])
ev.paired_diff(rep, "L1", "P1")          # read back from the report, sign flipped
```

## Pre-registration documents, declared split, dominant block (G1 #9 #11, G4 #7 #10, G5 #4 #5)

- **Pre-registration document linked to the seals.** `prereg=` also takes a text file or a list/tuple mixing
  registry directories, text files and seal records (a `zincir.jsonl` / `chain.jsonl` path stands for its
  directory); `prereg_doc=` names the document(s) separately (missing file: `FileNotFoundError`). The selection-bias
  audit searches these texts for the ours name like the registry's own text files. A text counts only when it is not
  newer than the latest seal, and its time is, in order: the `research.lock_prereg` lock (`<doc>.lock.json`) when the
  lock's sha256 matches the file; the earliest seal that recorded the document's sha256; the file mtime. A git
  checkout or copy resets the mtime, so lock or seal the document. A lock whose sha256 no longer matches raises a
  `UserWarning`, adds a card limit and falls back to the mtime. The card lists every document with its sha256 and
  time source (`summary(rep)["prereg_docs"]`).
  ```python
  research.lock_prereg("ONKAYIT.md")                          # ONKAYIT.md.lock.json: sha256 + locked_at
  ev.seal(test_preds, "ours: pred_S", registry=".onkayit", prereg_doc="ONKAYIT.md")
  rep = ev.evaluate(df, ours="S", prereg=[".onkayit", "ONKAYIT.md"])   # or prereg=".onkayit" alone, see below
  ```
  `seal(..., prereg_doc=)` (CLI `muhurle --belge`) writes `"onkayit_belgesi": {"dosya", "sha256"}` (file name only)
  into the chain record and appends `[ön kayıt belgesi <name> sha256:<hash>]` to the description. `evaluate(prereg=
  <registry>)` then picks the document up from the registry's parent directory on its own.
- **Declared vs scored primary metric (D10 metric trap).** The primary metric declared in the documents is read by
  `api.declared_primary(text)`: the `research.Prereg` JSON block or a `Prereg.write("x.json")` file (`metrics` as
  `{"primary": ...}`, a list (the entry that says "primary", else the first), or text), a `Primary metric: ...` /
  `Birincil metrik: ...` line, or the `## Metrics` section line that says "primary" (else its first line); a bare
  JSON file counts only with Prereg's `title` / `question` fields (a results file does not); the first known metric
  name wins, and the latest declaring document (an addendum) counts.
  When it differs from the scored primary (`mae` and `1-mae` count as the same), `evaluate` raises a `UserWarning`,
  adds an audit warning row "birincil metrik beyanı (ön kayıt)" (English: "declared primary metric
  (pre-registration)") and puts the card limit "primary metric differs from the declared one: ..." first; a match
  writes an "ok" row. No document, no row.
- **Declared split.** `split="test"` (or `"holdout"`, `"deger"`, `"degerlendirme"`) writes a constant split column
  when the frame has none: TEST-only frames and `sim.distribution_frame` cells (all held out) then run the split
  audit instead of skipping it ("bölme bilgisi" no longer flags leakage as unproven), and an audit row "bölme beyanı
  (split=)" says the split was declared by the caller, not derived. `split="oof"` / `"cv"` / `"time"` declare
  out-of-fold, cross-validated or time-split predictions (D12 #22, D9 #6, D6 #3b, D7 #12): every row is scored as an
  evaluation row and the audit row reads "cross-fitted (declared)". A frame that already has a split column keeps it
  (warning). Other labels raise `ValueError`.
- **One sign rule on the card (D5 #8c).** The "Ours − strongest baseline" line is in the metric direction (positive
  = better) and now also prints the raw difference: for lower-is-better metrics `raw ours − baseline = -0.010
  [...] (mae, lower is better: negative favours ours, ...)`, the convention of the registered-comparison rows (raw
  A − B). The English card gives a significant-but-small difference its reason and the threshold
  (`kart.ANLAMLI_FARK` = 0.02) instead of an empty line (D12 #10).
- **Dominant block.** `balance_col="country"` (default: `block_col` when given) checks the shares of the scored
  rows. When one block holds more than max(`dominance_share`=0.20, 1.5 / n_blocks), an audit warning row, a card
  note and `report["baskin_blok"]` / `summary(rep)["dominant_block"]` give the primary metric of every method both
  person-weighted (the reported value) and block-equal-weighted. With `aggregate_by=` each group value becomes the
  equal-weighted mean of its per-block means (not computed for `aggregate="sum"`); `balance_col` may vary within the
  aggregate groups (unlike `block_col`). Without `aggregate_by`, per-row metrics (MAE, Brier, 1-TVD, ...) use row
  weights 1 / n_block and whole-sample metrics (Pearson, Spearman, AUC, ...) the mean of per-block values. The
  latter is a within-block value: it leaves out what the method gets right between blocks (an arm that tracks the
  country gap loses, one that ignores it gains), so read a gap there as "how much rides on between-block signal",
  not as a reweighting of the same number. An
  explicit `balance_col` writes an "ok" row when no block dominates; the implicit `block_col` check stays silent then.
  ```python
  rep = ev.evaluate(persons, ours="L1_blind", aggregate_by="condName", balance_col="country", split="test")
  ```

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
  - Additive gap fixes (see "Options added after the port"): missing-truth handling in `sema`, the ladder lists every
    method, the multiplicity bootstrap path in `bootstrap`, the reused index draw in `bedel`, and the declared-baseline,
    model-ceiling, subsample and awaiting-truth branches in `cekirdek`. The equivalence test removes only these
    additive keys and audits before it compares with the original.
- Not ported:
  - the `python -m kanit` CLI (`rapor`, `birlestir`, `bedel`), which belongs to the integrator's
    `simengine evaluate|card` (`worth_humans(df)` reproduces `kanit bedel`)
  - the dataset adaptors (`adaptor/oe.py`, `adaptor/twin.py`, which belong to bench)
  - `kod/lig.py`, which is outside the package
- `python -m simengine.evidence.onkayit {muhurle|dogrula|zincir|notla}` still works.
- Tests *(this port)*: 137 evidence tests run in about 50 s (ported, API, extras, the gap tests in
  `test_evidence_gaps.py` and `test_evidence_d3_gaps.py`, and 6 equivalence tests, which skip when the original is
  absent; with `SIMENGINE_KANIT_SRC` set they pass after the D3 options, which add keys only when used).

## Demand forecasting: `task="forecast"`, WMAPE, bias, totals (D4 rehearsal gaps)

Added after the D4 blind rehearsal (M5 weekly demand, 4 weeks ahead), where the team had to write WMAPE, an item-
clustered bootstrap and a store x dept x week total themselves.

```python
rep = ev.evaluate(fr, "M5 4-week", ours="lgbm", task="forecast",      # continuous + forecast behaviour
                  unlabeled_baselines=["last4_mean"],                   # cluster = the frame's group > item column
                  ours_uses_llm=False,                                   # no LLM arm: memorization audit n/a
                  prereg="teslim/onkayit", n_configs=4, ci="basic")
tot = ev.evaluate(fr, "store x dept x week", ours="lgbm", task="forecast",
                  aggregate_by="store_dept_week", aggregate="sum")      # hierarchical totals, not means
d = ev.paired_diff(fr, "lgbm", "last4_mean", metric="wmape", group="item")
ev.seal("teslim/tahmin.csv", "final", registry="teslim/onkayit")      # records sha256 AND dosya_sha256
```

- **Metrics.** `primary="wmape"` (alias `"wape"`): sum|y - p| / sum|y|, lower is better. `"bias"`: sum p / sum y - 1,
  signed, a diagnostic only (it is refused as primary). They are a new metric kind `oran` (ratio): the cluster
  bootstrap resamples numerator and denominator with the same cluster counts and takes the ratio of sums per
  replicate. Both are added to any continuous evaluation whose primary is a ratio metric, and always with
  `task="forecast"`. `paired_diff` and `paired_ci` accept them on any continuous frame.
- **`aggregate="sum"`** with `aggregate_by=` sums truth and predictions per group (default `"mean"` unchanged).
  The sum is the group total even when the frame has an `item` column (the item is kept as a column only when
  it is constant within the group; the bootstrap clusters are the groups). A group in which a column has any
  missing row gets NaN for that column, so a missing prediction never counts as a zero forecast; the group then
  drops out of the common rows. `noise_ceiling=` is refused with `"sum"` (its half-split means are not totals).
- **Bias has no direction.** `bias` is signed: in `paired_diff` / `paired_ci` / the comparison table, "a better"
  means `|bias_a| < |bias_b|`, and the report's metric table gives it `yon=0`.
- **Forecast-mode wording.** With `task="forecast"` the within-item shuffle row (which would compare ours with
  itself) is not written, the de-meaned R² row is "seri içi sinyal (demeaned_r2 > 0)", and the automatic
  baselines (random, item median/mean) are labelled "look-ahead": they are cross-fitted on the test-window truth,
  so they are not legitimate rungs. Pass origin-time baselines (persistence, last-k mean) as `pred_` columns, or
  `baselines=False`.
- **`block_col=`** (e.g. `"origin"` or the week column; `"time"` accepted): two-way block bootstrap. Each distinct
  value is one block; in every replicate the blocks and the clusters (grup / item / row) are resampled
  independently with replacement and a row's weight is block count × cluster count, so column, ladder, headline and
  paired CIs (`compare=`, `blind_col=`) cover block-to-block variation and series dependence across blocks (a
  one-way block resample could come out narrower than the default when series carry a persistent bias). The card
  meta says "bootstrap: two-way, blocks of `<col>` (n blocks) × clusters"; `summary()["block_bootstrap"]` gives the
  block count, `kart.n_kume` stays the cluster count. Blocks are non-overlapping; for a moving-block flavour with
  length L pass a coarser column (week // L). With `aggregate_by=` the column must be constant within each group
  (else `ValueError`); NaN in the column raises `ValueError`. A one-valued `block_col` is not applied (default
  clusters). Fewer than 10 blocks: audit warning "blok bootstrap blok sayısı" and a limits note; below 5 the card
  meta says "CI not reliable across blocks". Default `None`: clusters as before.
- **Single forecast origin.** With `task="forecast"`, when the `origin` column (else `time`) has one distinct value,
  whatever `block_col` is, an audit warning row "tek tahmin kökeni (single origin)" and a card note say "single
  forecast origin: CIs cover series sampling only, not origin-to-origin variation; use
  methods.forecast.backtest(origins=[...])" (`summary()["single_origin"]`). D4 `SIMENGINE-EKSIK.md` #8.
- **Mean vs median row.** Opt-in. With `task="forecast"` and an L1 primary (`mae` / `wmape`),
  `point_pairs=[(mean_col, median_col)]` or `point_pairs="auto"` (pairs ours with `pred_<ours>_median` if present)
  adds a card table with the primary metric and bias of both, and a sentence with the observed direction (L1
  lower/higher with the median, |bias| smaller/larger) (`summary()["mean_median"]`). Explicit columns are always
  validated; outside forecast + L1 an explicit list is skipped with a `UserWarning`. Default `None`: no row. D4 #9.
- **`ci="basic"`** (2·θ − bootstrap quantiles) for column, ladder, headline and paired CIs; default `"percentile"`.
  When a primary-metric point estimate falls outside its percentile CI (possible for ratio/aggregate metrics), a
  `UserWarning` is raised, an audit row "bootstrap GA (nokta tahmini yüzdelik GA dışında)" is added and the card gets
  a flag. The rehearsal saw this on its own code (point 0.0979, percentile CI [0.0965, 0.1114]; one run, D4
  `SIMENGINE-EKSIK.md` #6); BCa is not implemented.
- **`task="forecast"`** maps to the continuous task internally (`types.TASK_KINDS` is unchanged), default primary
  wmape, and skips the persona shuffle control, the human test-retest ceiling and the PPI++ "worth N humans"
  multiplier: the card rows say "uygulanamaz / not applicable: forecast task" and an audit row explains why. A model
  ceiling still works through `ceiling_col=`. The awaiting-truth card does the same.
- **Contamination audit.** The English `time` column is the internal `zaman`; the audit runs only when the LLM
  training cutoff is given, e.g. `cutoff="2025-03-01"`. With `time` but no cutoff the row now says so.
- **No LLM arm.** `ours_uses_llm=False` labels the missing memorization / contamination audits "no LLM arm: not
  applicable" and the card no longer raises "kanıtlanmadı: sızıntı/ezber" for them (skipped split/group/time leakage
  audits still flag "kanıtlanmadı: sızıntı (...)").
- **Selection bias.** The audit writes "named in advance" only when the ours name (with or without `pred_`; names
  of 1-2 characters need the prefix) occurs in a seal *description* or in the small text files (`.md .txt .yaml
  .yml .json .toml`) of the `prereg=` directory. Sealed column names and source paths do not count: sealing a
  table with every candidate column names nothing in advance. A text file newer (mtime) than the latest seal does
  not count ("pre-registration text is newer than the seal"), and a text naming several methods without marking
  ours (`ours: pred_x`, `primary: pred_x`) gives "several method names". Otherwise "not verified: ours not found in
  the pre-registration" (warning) or "not verified: no pre-registration" (skipped).
  ```python
  Path("reg/PREREG.md").write_text("ours: pred_lgbm, primary WMAPE\n")   # write it BEFORE sealing
  ev.seal(test_preds, "test predictions", registry="reg")
  ```
  `n_configs=` is written into the row and counts toward the "ours is the best of >= 4" optimism note.
- **Seal hashes.** `seal(path)` also stores `dosya_sha256` (raw file bytes, what `sha256sum` prints) next to the
  canonical CSV `sha256` (what `verify`/`grade` check); the card's seal table shows both columns, labelled.

## 'Why' recognition and person-level distribution bootstrap (G6 #1, G5 #3)

**`why_match(generated, candidates, truth_idx, *, encoder="tfidf", ids=None, groups=None, seed=0)`** (`why.py`) asks whether a
generated reason is closer to the person's own real reason than to other people's. One row per unit; `candidates[i]`
holds N real reasons, `truth_idx[i]` marks the person's own (one int is accepted, but shuffle positions). Each arm
"chooses" the most similar candidate (cosine); ties are broken at random with `seed` and the reciprocal rank averages
over tied positions, so an empty text scores chance rather than candidate 0. Output is an EvalFrame: `truth`,
`pred_<arm>` (chosen index), `rr_<arm>`, `n_candidates`, `chance` (1/N), `unit` from `ids` (repeats get a `#k`
suffix) and `group` from `groups` (default `ids`; pass person ids when a person has several rows). Chance: accuracy
1/N, MRR H_N/N. A constant `truth_idx` over 10+ rows warns: evaluate's majority baseline would then score 1.0.
```python
f = ev.why_match({"persona": gen_p, "no_persona": gen_0, "swap": gen_s}, candidates, truth_idx, ids=person_ids)
rep = ev.evaluate(f, task="multiclass", primary="accuracy", ours="persona", raw="no_persona")
ev.why_summary(f)          # {"arms": {arm: {accuracy, mrr}}, "chance_accuracy", "chance_mrr", "n"}
f = ev.why_match(gen, candidates, truth_idx, encoder=sim.SSR(model).embed)   # any texts -> (n, d) callable
```
Distractor recipe: N-1 reasons written by *other* people about the *same product* with the *same rating*. Holding
product and rating fixed leaves only reason content, so a text cannot win by guessing sentiment or naming the product.
Draw and seal the candidate sets (with truth positions) before generating. The default encoder is word 1-2 gram +
char_wb 2-5 gram TF-IDF fit on the candidate texts, in pure numpy with sklearn-compatible defaults (`TfidfEncoder`;
fit it on a larger corpus and pass it as `encoder=` if wanted). `encoder="bge"` is not built in: pass an embedding
callable. TF-IDF rewards shared wording (a person's own vocabulary and style) as much as the reason itself, so
report an embedding row next to it before reading a win as "the reason is right"; it has no stop-word list and
no language-specific tokenizer (char n-grams carry Turkish suffixes).

**`evaluate_distribution_persons(person_frame, segment_cols, items, pred_cols, *, B=1000, seed=0, metric="1-tvd",
id_col="unit", options=None, hold_truth=False, alpha=0.05, min_n=1, pairs=None)`** (`persons.py`) scores segment x
item distributions from the holdout *people* instead of ready cells. One row per person, one segment column per
scheme (`["age", "income"]` or `{name: column}`; schemes are scored jointly and per scheme), one answer column per
item. The bootstrap draws people with replacement (clustered on `id_col`), rebuilds every cell's truth and every
person-level arm from the same draw (paired across arms) and scores the mean metric over cells. Arms are fixed cell
tables (`{(scheme, segment, item): vector}` or a DataFrame `scheme?, segment, item, pred|pred_<arm>`, e.g.
`sim.distribution_frame`) or person-level columns `{item: column}` (probability vector or label per person).
Metrics: `1-tvd`, `tvd`, `1-w1`, `w1`, or a callable (higher is better). Default option order is the observed
answers, numeric when they are numbers; `w1` warns on alphabetical labels, pass `options={item: [ordered]}`. 3000
people x 120 cells x B=1000 runs in ~2 s.

1-TVD between a small segment's empirical distribution and any prediction is biased down, and the bootstrap repeats
the noise, so the percentile CI sits below the point and can exclude it (the G5 rehearsal saw this). Each arm and each
paired difference reports `point`, `ci` (percentile), `ci_basic` (2·point − quantiles), `ci_bc` (percentile − bias),
`bias` (bootstrap mean − point) and `point_bc` (point − bias); differences also give `p_two_sided`. `hold_truth=True`
keeps truth at the full-sample distributions and resamples only person-level predictions (fixed-cell arms then have
zero width; `note` says so). `cells` lists the per-cell scores and counts; cells with fewer than `min_n` answering
people are dropped.

## Card rules from the recipe rehearsals: information level, post-hoc rows, shuffle scope, filled share (G1 #10, G5 #10, G6 #7 #8, G7 #6 #7 #9)

No new `evaluate()` arguments: these rules run on the inputs the report already has.

- **Information level of the marginal baseline.** On distribution tasks `taban_marjinal` (item-wide leave-one-out)
  reads the target question's answers in the *other* cells: setting B in bench terms, information ours does not see
  (setting A: other questions only). With overlapping segment schemes (age / income / household) the held-out cell's
  people can sit inside those other cells. The ladder row carries `bilgi: "B"` and a "setting B" label, the
  "item mean" audit row says so, and the strongest-baseline headline, gap closed and badge use the strongest
  setting-A baseline only. When the setting-B rung beats it, an audit row "bilgi düzeyi (ayar A / ayar B)" and a card
  flag say so; the card's baseline row gets a note. `report["bilgi_duzeyi"]` = `{"B", "B_ustte", "rozet_tabani"}`.
  Other task types are unchanged.
- **Registered vs post hoc.** When pre-registration text exists (seal descriptions, `prereg=` text files / registry
  texts), each `compare=` row's a and b names are looked up in it: a row counts as registered when one paragraph
  (split at blank lines and headings) names both (b None: a alone) or a text contains the row's own label. A whole
  ONKAYIT.md names every arm somewhere, so a and b in different paragraphs (ours in the arms list, b only in the
  baseline list) do not count. Two-character names (`L3`, `T3`) match without the `pred_` prefix here, since both
  have to occur together. To mark a row registered whatever the wording, give it a label that the pre-registration
  uses (`compare=[("L3", "T3", "C3")]` with "C3: L3 − T3" in ONKAYIT.md). A row that is not found gets `tur="posthoc"`, `kayit="sonradan"`, a
  "post hoc (not in the pre-registration)" label suffix, `kind: "posthoc"` in `summary()["comparisons"]`, a
  "sonradan ·" prefix on the dashboard and a card flag. Without any text the rows stay as before (`tur="kayitli"`,
  no `kayit` key). `degerlendir(kiyaslar=)` also takes `(a, b, label, posthoc)` or `{"a", "b", "label", "posthoc"}`
  to set it explicitly; `evaluate(compare=)` still takes 2- or 3-tuples.
- **Weak baseline in a compare row.** If b is a baseline weaker than the strongest (setting-A) baseline, the row
  gets `zayif_taban = {"b", "b_deger", "en_guclu", "en_guclu_etiket", "en_guclu_deger"}`, an audit warning names the
  strongest one, and the card flags it: register against the strongest single baseline, not a weaker combination.
- **Shuffle-control scope.** When ours is constant across units within every item (an arm without person
  information, for example a direct item-level guess), the shuffle control is not built: the audit row is
  `atlandi` with "predictions constant within item (no person information)". The same skip applies to group-level
  rows without an item column (`aggregate_by=` over conditions or countries, no `item`): a shuffle would only permute
  the groups' predictions, a permutation null rather than a persona check. On multiclass tasks the control is
  labelled a *prediction* shuffle, not a persona swap (audit "tahmin karıştırma (persona takası değil)"); pass a real
  swap arm via `compare=(ours, swap, "persona swap")`.
- **Filled share.** Numeric `filled_<method>` columns (from `sim.distribution_frame`: the share of a cell's answers
  that were invalid and replaced by a uniform vector) give one audit row per arm with the mean share. Above 5 %
  (`cekirdek.DOLGU_ESIK`) the row warns and the card flags "dolgu payı yüksek". `report["dolgu"]["kollar"]` holds
  mean, max and cell counts per arm.

## Calibration-in-the-large, option positions, binary spread, entropy ratio (G4 #8 #9, G7 #4 #8)

- **`mean_bias`** (binary and continuous): mean prediction − mean truth per arm, i.e. predicted rate − base rate
  on a binary task ("how many will buy"). It is a per-row metric, so every arm gets a cluster-bootstrap CI in the
  metric table (`rep["sonuclar"][col]["metrikler"]["mean_bias"]`). It is signed (`yon=0`; "a better" = closer to
  0) and a diagnostic, not a primary metric: `evaluate(primary="mean_bias")` raises `ValueError`, as `bias` does
  (a "lower is better" ranking would crown a large negative bias). The Markdown card (`card_markdown`) lists it
  for every baseline, method and calibrated arm in a "Calibration-in-the-large" table. Paired difference of two arms:
  ```python
  ev.paired_ci(frame, "ours", "raw", metric="mean_bias")     # diff = mean(pred_ours) - mean(pred_raw), with CI
  ```
  The continuous ratio form (`bias` = sum p / sum y − 1) stays as before.
- **Wasserstein on option positions.** The distribution metric `wasserstein` (same name, no rename) uses numeric
  option positions when the frame has them and falls back to equal spacing 0..K−1 otherwise. Sources, first
  numeric one wins: a per-row `pos` / `positions` / `konum` column (list per row), a numeric `options` column,
  numeric `classes=`. Positions are sorted and rescaled to 0-1 (min → 0, max → 1), so with bench positions that
  span 0-1 the number equals bench `wd` (`bench.stats.wasserstein_ordinal`). Equally spaced numeric options
  (1..5) give the old number exactly; text options keep equal spacing. `metrik.wasserstein_konumlu(a, b, pos)` is
  the function; `wasserstein_ordinal` (equal spacing) is unchanged.
  ```python
  df["pos"] = [opt_pos[q] for q in df["item"]]                # per row, e.g. [0, .2, .5, 1]
  rep = ev.evaluate(df, ours="llm", primary="wasserstein")
  ```
- **Binary spread.** With probability predictions (any value strictly between 0 and 1) the spread block no longer
  compares sd(p) with the sd of the 0/1 truth (always small, it raised "collapse" on every AUC task). It reports
  `sd_bernoulli` = sqrt(Σ var_item(p) / Σ p̄_item(1 − p̄_item)) with a CI (0 = constant prediction, 1 = certain
  0/1) and leaves `sd_orani` empty (not applicable; the dashboard spread chart skips the arm). The collapse audit
  "dağılım çöküşü (Bernoulli sd oranı)" warns only below 0.05 (near-constant prediction). Hard 0/1 predictions keep
  the old sd ratio.
- **Class imbalance audit.** `dengesizlik(veri, birincil=None)` warns (majority share > 70 %) only when an
  accuracy-type metric (accuracy, macro-F1) is in play: with binary probability predictions, or a primary that is
  not accuracy-type, it is a `tamam` note (AUC, Brier, log loss and ECE are not misled by the class share).
  Multiclass and hard-label binary keep the warning. `evaluate` passes its primary metric, so the warning stays
  when the primary is accuracy or macro-F1, including the automatic macro-F1 switch above a 60 % majority; the
  probability-prediction note applies only to direct calls without `birincil`.
- **Distribution entropy ratio** = mean entropy(prediction) / mean entropy(truth) over cells (ratio of means,
  `entropi_olcut="ortalamalarin_orani"`), with a cluster-bootstrap CI (`entropi_orani_alt/_ust`). The mean of
  per-cell ratios is not used: it explodes on small groups whose truth entropy is ~0. `ayrisma_orani` is the
  between-cell divergence ratio: within each item, sd across cells (groups) of E[position] for the prediction over
  the same for the truth (pooled; < 1 = groups pulled together). E[position], the within-cell sd ratio, the
  distribution bias and the segment-gap correlation use the numeric option positions above when the frame has
  them (rescaled to 0-1), else equal spacing. The collapse audit row prints both with CIs.

## A/B pairs: pair table -> variant frame, A/B memorization probe (G0b #3 #7, G3 #2 #6)

`evaluate(task="ab")` scores variant rows (`item` = test, `truth` = each variant's outcome) and builds the pairs
itself. The bench upworthy task, `methods.LLMDirectAB` and `ABStack` give one row per pair instead (`y` = A won,
prediction = P(A wins)). **`ab_pairs_frame(pairs, pred_cols, y="y", group="test_id", *, a_wins=None, keep=(),
drop_ties=True)`** (`ab.py`) writes each pair as two rows: A gets truth y and pred p, B gets 1 - y and 1 - p.
`pred_cols` is a column, a list (`p_llm` -> arm `llm`) or `{arm: column or array}`. A labelled truth column (the
bench `winner`: a / b / tie) works as is; labels compare case-insensitively, and any other label raises unless
`a_wins=` names the label meaning A won (then everything else but "tie" / "draw" is a B win). Ties carry no pair-sign signal and are dropped
unless `drop_ties=False`. The item is the *pair*, not the test: Upworthy tests hold several pairs, and one item per
test would make `pair_sign` compare A of one pair with B of another. The test goes to `group`, the bootstrap
cluster, so all pairs of a test move together. `arm` (A | B) feeds the arm-majority baseline. `pair_sign` then
equals the direct pair accuracy (p = 0.5 counts as a half). With `group=None` every pair is its own cluster.
```python
f = ev.ab_pairs_frame(test_pairs, {"llm": "p_llm", "stack": "p_stack", "tfidf": "p_tfidf"}, group="test_id")
rep = ev.evaluate(f, task="ab", ours="stack", raw="llm")      # primary pair_sign; AUC/Brier stay on the pair table
```
The CI with `group=` set uses the copy-aware cluster bootstrap (G3 #3 fix above): each pair is nested in one test,
so a test drawn twice gives two separate copies of its pairs. Simulated coverage of the 95 % pair_sign CI
(300 synthetic runs, 150 tests with 1-4 pairs each, test-level accuracy Beta around 0.65): 0.943, width 0.111.
With `group=None` the CI is the binomial one.

**`ab_memorization(llm, pairs, named_context, neutral_context, *, n=100, paraphrase=False, margin=0.0, seed=0,
B=1000, batch=15, reverse=True, max_calls=500)`** asks the same n real pairs `LLMDirectAB`-style twice: once with
the study and source named in the context (for example "Upworthy Research Archive headline tests"), once neutral
("A/B test of two headlines"). Taste for good headlines helps both arms equally. If naming the source raises pair
accuracy, the model may remember the published winners. `deger` = accuracy(named) - accuracy(neutral), with a
paired cluster bootstrap over tests. `durum` is "uyari" when the lower bound is above `margin` (the threshold,
returned as `esik`), else "tamam"; it is "atlandi" below 5 pairs. The dict also has `accuracy_named` /
`accuracy_neutral` with CIs, coverage, `n_pairs`, `n_tests` and `calls`, and it is what `evaluate(memorization=...)`
reads: the card's memorization row shows the named - neutral gap, its CI and the threshold, and "uyari" adds the
caveat "gain may come from recognition/memorization" to the badge reason (the "not proven: leakage/memorization"
flag stays while the split or contamination audit is skipped). `paraphrase=True` (the LLM rewrites every text, batched) or a `str -> str`
callable adds a third arm: the named context on rewritten texts. Memorised wording is lost there, so
accuracy(named) - accuracy(named, paraphrased) is a second gap (`paraphrase_gap`, CI); its lower bound above
`margin` also gives "uyari". `paraphrased_share` is the share of texts the rewrite actually changed: an unparsed
paraphrase reply keeps the original text and pulls `paraphrase_gap` toward 0, so a low share makes that gap no
evidence. The planned call count is checked against `max_calls` before any call (cached replies count too).
Simulated, test-clustered gap CI (synthetic hits, 25-150 tests with 1-4 pairs): with no memorization the
"uyari" rate is 2.8-3.5 % and coverage 0.94-0.95; with a true +0.05 gap coverage is 0.92-0.95 and the lower bound
lies above the truth in 2.4 % of runs (the misses are on the upper side, so the warning is not inflated).

**`ab_memorization_items(pairs, n=100, seed=0)`** builds items for `memorization_audit` (the completion gap): each
real pair ("which text won? reply with it", expected = the real winner) plus a control pair whose two texts come
from two different tests, so no one can know a winner (expected = a coin flip). **Taste confound:** real pairs
have a true winner and controls do not, so any better-than-chance model with no memory at all beats the controls
(pair accuracy 0.65 gives a gap of about +0.15) and `memorization_audit` returns "uyari". That is not a
memorization verdict: pass the `ab_memorization` dict, which cancels taste, to `evaluate(memorization=...)`, and
read this gap only against the neutral-context pair accuracy - 0.5. Text columns may be
`text_a/b`, `headline_a/b` or `metin_a/b`.
```python
mem = ev.ab_memorization(llm, test_pairs, "Upworthy Research Archive (2013-2015) headline tests",
                         "A/B test of two headlines shown to real users", n=100)
rep = ev.evaluate(f, task="ab", ours="stack", raw="llm", memorization=mem)
```
Not yet run on real data. `evaluate(task="ab_pair")` (taking the pair table directly) is still to come.

## Frame helpers: repeated splits, cell counts, wide-to-long, segment bias (D9 #5 #6, D11 #5d, D12 #3 #7 #20 #21)

`frames.py`. Four bridges the TALP rehearsals wrote by hand: a seed loop over random splits (the dev card showed one
split), `value_counts` + concat to turn per-person argmax into cell counts, melting a sealed wide multi-target file and
dividing by the training sd, and the brief's segment error.

```python
ss = ev.split_summary(lambda s: ev.evaluate(make_split(s), ours="t1", multiplier=False), seeds=range(100),
                      pairs=[("t1", "o1")])        # or a list / {name: x} of reports or frames
print(ss.markdown)                                 # card section; ev.evaluate(..., limits=ss.limits) puts it on the card
cf = ev.counts_frame(people, {"m": "p_vec"}, by=["area"], probs=True, classes=cls, argmax_too=True)
ev.evaluate(cf, ours="m", task="forecast", primary="wmape")          # pred_m (expected) vs pred_m_argmax counts
lf = ev.long_frame("sealed_wide.csv", truth_wide, "id", scale="sd_train", train=train_wide)
ev.evaluate(lf, ours="ours", task="continuous", primary="mae")      # MAE in training-sd units (zMAE)
sb = ev.segment_bias(frame, "region", ["ours", "base"])             # sb["arms"], sb["diff"], sb["segments"]
```

- **`split_summary(reports_or_frames, metric=None, pairs=None, *, seeds=None, task=None, names=None)`** returns a
  `SplitSummary`: `per_split` (split x column value, split x pair difference), `table` (median, 2.5 / 97.5 % over
  splits, mean, `share_a_better` for pairs: ties count half, signed metrics compare |value|), `markdown`, `limits`.
  Reports are read from `sonuclar` (columns by method name, baseline columns such as `taban_marjinal`, and `ours`,
  `raw`, `best_baseline` = that split's strongest baseline); frames are scored with point metrics only (no
  bootstrap). Default pair for reports: `("ours", "best_baseline")`. The quantiles are split-to-split spread, not a CI.
- **`counts_frame(rows, pred_cols, by, *, truth="truth", cluster=None, probs=False, classes=None, weight=None,
  argmax_too=False)`**: one row per observed by-cell x class (zeros kept) with `truth` count, `pred_<arm>` count, `n`,
  `item` = class and `group` = cluster (default the by-cell). Label arms count persons per predicted class; probability
  arms (`probs=True` or a list of arms; a vector column with `classes=`, a `{class: p}` column, or `{class: column}`)
  sum probabilities into expected counts. `argmax_too=True` adds `pred_<arm>_argmax`: the trap where argmax wins
  person accuracy and loses the count WMAPE because every cell collapses onto its majority class. A cell where an arm
  has a missing prediction gets NaN for that arm, never a zero count.
- **`long_frame(pred_wide, truth_wide, id_col, scale=None, *, train=None, targets=None, arm="ours")`**: wide files
  (DataFrame or CSV / parquet path; `{arm: ...}` for several arms) to `group` = unit, `item` = target, `truth`,
  `pred_<arm>`. `scale="sd_train"` divides each target by its training sd (ddof 1, from `train=`), or pass
  `{target: s}`; zero or missing scales raise. Truth units without a prediction are dropped with a warning. The scale
  used is in `frame.attrs["scale"]`.
- **`segment_bias(frame, seg, arms, weight=None, *, cluster=None, B=2000, seed=0, ci="percentile", min_n=1)`**:
  Σ_s w_s |mean pred_s − mean truth_s|, w_s = n_s / n (`weight="equal"`: 1 / S; a column name: row weights). The
  cluster bootstrap (group > item > row) resamples truth and every arm together; pairs are `arms[0]` − each other arm.
  **A single-arm CI of an absolute-value metric is biased**: each resample adds noise inside |·|, so its distribution
  sits above the point when the true error is near 0. Synthetic check *(this port)*: 60 segments x 10 rows, no true
  error, 30 seeds: the point fell below its single-arm percentile CI in 29 of 30, while the paired difference CI of
  two unbiased arms covered 0 in 30 of 30. The result flags `point_outside_ci` (with a `UserWarning`) and carries the
  note; judge arms on `diff`.

## Average precision, rate error rows, not-applicable audits (D5 #8, D9 #9, D11 #7, D12 #12 #14 #16 #18, D13 #5 #10)

```python
rep = ev.evaluate(df, ours="kb1", primary="ap")                       # also "average_precision", "pr_auc"
rep = ev.evaluate(df, ours="kb1", primary="auc", rate_error_by="segment")   # or a column name / list of columns
ev.summary(rep)["rate_error"]                                          # rows, ours − other differences
```

- **Average precision** (`metrik._ap`, binary family, higher is better): step-wise area under the precision-recall
  curve, tied scores form one threshold, equal to `sklearn.metrics.average_precision_score` (to 1e-12 on 20 random
  tie-heavy cases); no positive, an empty sample or a non-finite score -> NaN. It is an aggregate metric, so its CI
  comes from the `B_aggregate` loop; above `HIZLI_SIRA_ESIK` rows it takes the multiplicity-weighted fast path next
  to AUC (`bootstrap._ap_cokluk`, equal to the expanded resample to 1e-12): 20,000 binary rows × 6 methods
  (B=300, B_aggregate=100) evaluate 5.0 s -> 3.6 s, same CI to the last digit *(this port, nice 19)*.
- **`rate_error_by=`** (binary / continuous): for every method and baseline column, total rate error
  |Σp − Σy| / n (= |mean_bias|) and n-weighted segment rate error Σ_s |Σ_s p − Σ_s y| / n (= `segment_bias` with
  `weight=None`), weighted by the `weight` column when present; lower is better. The CIs and the ours − other
  differences (p, `a_iyi`) use the ladder's paired cluster bootstrap (same clusters, `block_col` blocks, seed, `B`,
  `ci=`). `"segment"` means the frame's `segment` column, else its `seg_*` columns (each its own column of the table).
  Report `oran_hatasi` (`satirlar`, `farklar`, `segmentler`), `summary()["rate_error"]`, card table "Rate error
  (lower is better)". The single-arm CI of an absolute value lies above the point when the true gap is near 0
  (`point_outside_ci` flag; see `segment_bias`): judge arms on the paired differences. Unknown columns and other
  tasks raise `ValueError`; with `aggregate_by=` only the columns kept by the aggregation can be used.
- **Not applicable instead of failing** (368e5a8, review 36ea5d5): `level="segment" / "study" / "effect"`, or one row
  per segment in every item where the segment cells form a grid repeated across items (distinct cells ≤ max rows per
  item; an explicit other `level=` such as `"person"` turns the detection off), marks the persona shuffle and the
  PPI++ "worth N humans" multiplier "not applicable" with the reason (audit rows, the worth row's note,
  `report["carpan_uygulanamaz"]`). "Strongest other method" warns only when the paired CI excludes 0 in the bad
  direction, otherwise "tied"; without a CI it warns on the point sign, worded "point estimate only". A low sd ratio
  under an L1 primary (mae / one_minus_mae / wmape) is an informational note (`bilgi_notu`), over-dispersion still
  warns. Per-item constant predictions give a warning row with the item count and `report["sira_tanimsiz"]`
  (constant / NaN rank-metric items per method). With `ours_uses_llm=False` the `raw=` arm keeps the memorization and
  contamination audits.

## CLI block bootstrap, point pairs, declared split; seal source name (D4 #8, D6 #9)

- **`simengine evaluate --block-col / --point-pairs / --split`** pass `block_col=`, `point_pairs=` and `split=` to
  `evaluate`. `--point-pairs` is `auto` or `MEAN:MEDIAN` pairs, comma separated (`ours:ours_median,b:b_median`;
  with or without `pred_`); a malformed value is a one-line error, exit 2. `--split` accepts the `split=` labels
  (`test`, `holdout`, `deger`, `degerlendirme`, `oof`, `cv`, `time`).
  ```
  simengine evaluate bt.csv --ours lgbm --task forecast --primary mae --block-col origin --point-pairs auto --split time
  ```
- **Seal source is a file name.** The chain record's `"kaynak"` holds only the file name (`preds.csv`), so a committed
  registry carries no machine path. `onkayit.muhurle(..., kaynak_yolu="tam")` (alias `"full"`; CLI
  `muhurle --tam-yol`) keeps the path as given. Older records with full paths still verify (`zincir_dogrula`,
  `dogrula`) and new records append to them; readers (selection-bias text match, card) use the name only.

## Cluster override, per-unit mean, dropped units, paired_diff blocks, badge threshold (G3 #13, D7 #9, D9 #3b #4d, D12 #11)

```python
rep = ev.evaluate(df, ours="m", cluster="item")                     # bootstrap clusters = items, not cells
```

| option | what it does | source rehearsal |
|---|---|---|
| `evaluate(cluster="item")` | bootstrap cluster column ("item", "group" or any column), overriding group > item > row; group still sets the cross-fitting folds; constant within `aggregate_by` groups; the cluster side of the `block_col` two-way bootstrap; card "clusters: N (cluster=`item`)", `summary()["cluster"]`, `["n_clusters"]` | G3 #13: the same item (question × amount) in 18 countries, unit bootstrap [-0.117, -0.037] vs item-clustered [-0.188, +0.007] |
| `evaluate(aggregate_by="person", aggregate="unit_mean")` | rows stay rows, weight 1/n_unit (× `weight`), unit = bootstrap group: a per-row metric is the mean within each unit, then the mean of the units; the task is kept; rank metrics are one weighted value | D9 #3b: per-person accuracy, "within the person first, then the mean of persons" |
| dropped units (every `aggregate_by` mode) | `report["toplama"]` `n_units`, `n_units_scored`, `n_dropped`, `n_rows_dropped`; when units are dropped (no test-split row, truth or predictions missing): UserWarning, a limit line, card meta "k of N units dropped" | D9 #4d: 15 persons had no DEV rows, aggregate_by scored 2,985 clusters silently |
| `paired_diff(frame, a, b, block_col="origin")` | the two-way block bootstrap of `evaluate(block_col=)` (same default: off; same CI as the report's registered pair for the same B and seed); with `aggregate_by` the column must be constant within groups; on a report the report's bootstrap is reused and its `block_col` returned, a different `block_col` raises | D7 #9: evaluate(block_col="origin") [-0.059, +0.015] vs paired_diff [-0.041, -0.003], a pre-registered decision flipped with the bootstrap type |
| `card_markdown(lang="en")` audit names | `CHECK_TO_ENGLISH` covers the rows added in f948774 / 9aa3d47 (memorization warning and ours, time column parsing, frame ↔ sealed file, effective sample multiplier, constant prediction); an unknown "Turkish (English)" name prints its English part; a free text falling back to its English part keeps the Turkish part in brackets when numbers would be lost | W8 review (deferred) |
| `evaluate(min_effect=0.009)` | smallest ours − strongest baseline difference (metric units and direction) for the "high" badge instead of `kart.ANLAMLI_FARK = 0.02` (default unchanged); card meta "min_effect=x", the badge reason names the threshold (Turkish and English), `report["min_effect"]`, `summary()["min_effect"]`; `kart.rozet(..., anlamli_fark=)` | D12 #11: a z-MAE gain of 0.009 was called "small" against the fixed 0.02 |
