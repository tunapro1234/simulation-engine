# Measured results

Every number in this file was measured by the TALP prep research code (October 2026) and is copied from a report in
the research repository `talp-simulation-hackathon-2026/uretim/<branch>/`. The **source** column names the branch and
file. This library is a port of that code; unless a row says **(re-run)** or **(new)**, the library itself did not
re-measure the number. **(re-run)** means the library reproduced it on the real data; **(new)** means the library
computed it and the research code has no counterpart. Section 8 is the exception: its rows come from two earlier
repositories (replicant and edsl-test) run on non-Claude models, with sources relative to those repos.

`[a, b]` is a 95% bootstrap CI. "Gap closed" is (method − baseline) / (ceiling − baseline).

The short version:

- **Simple baselines are strong.** Persistence ("same as last time"), the item marginal, popularity and a plain
  logistic/GBDT model win or tie in most tasks. Report them next to every LLM number.
- **Pure persona panels rarely beat those baselines.** A/B headline panels are at chance with Haiku; per-person purchase
  prediction from survey answers is below "always no"; discussion rounds and agent societies add nothing measurable.
- **The LLM adds value in three places:** with no data at all (segment distributions asked directly), stacked with a
  learner (A/B TF-IDF + LLM), and as a prior in small early tests (about 2x fewer impressions at 100 per arm).
- **Gains found on dev often shrink on test.** The research loop's dev gain on OpinionQA did not hold on test; random
  search was as good as the LLM observer.

## 1. Survey and opinion distributions

| setting | method | result | source |
|---|---|---|---|
| CES 2020, 4 party × age segments, 8 items, Haiku; W1 (lower better); general marginal 0.206, 37-person real survey 0.050 | `sim.segment_distribution`, 3 named framings averaged | W1 0.077, gap closed 0.818 | makro `RAPOR.md` (C1, K2-5) |
| same | framings v1 / v2 / v3 | 0.077 / 0.075 / 0.096 | makro `RAPOR.md` (K2-5) |
| same | hidden framing (no survey name, no year) | 0.084 (0.774) | makro `RAPOR.md` (K2-5) |
| same | per-persona probability vectors (`persona_distribution`) | 0.088 (0.751) | makro `RAPOR.md` (K2-6) |
| same | persona panel, round 0, batched tables | 0.120, gap closed 0.55 [0.44, 0.60] | makro `RAPOR.md` (C1) |
| same | persona panel, one call per persona | 0.142 (0.41) | makro `RAPOR.md` (K2-6) |
| same | direct distribution − panel | ΔW1 −0.043 [−0.072, −0.017] | makro `RAPOR.md` (K2-5) |
| same | direct distribution vs general marginal | worse on 2 of 8 items (Congress approval, household income); worth about a 12-15-person survey; panel about 5-6 | makro `RAPOR.md` (K2-5) |
| same | persona vs no persona vs cross-segment shuffle | gap closed 0.55 / 0.45 / 0.25 | makro `RAPOR.md` (persona control) |
| same, individual level | segment majority vs LLM round 0 | 0.656 vs 0.613 exact match | makro `RAPOR.md` (persona control) |
| same | 5 discussion rounds (`roundtable`) | 5-8% of answers change; R5 ΔW1 −0.005 [−0.012, +0.002] | makro `RAPOR.md` (paired measurement) |
| same | confirmation-bias instruction, round 3 | ΔW1 −0.007 [−0.012, −0.003] (0.120 → 0.113) | makro `RAPOR.md` |
| OpinionQA subgroups, setting B (Overall known), dev WD | copy Overall / uniform / sibling | 0.0530 [0.0509, 0.0554] / 0.1976 / 0.0764 | dongu `GUNLUK.md`, `turlar/oqa_grup_B/tabanlar.json` |
| OpinionQA setting A test (486 questions), WD | uniform / position prior | 0.1779 [0.1716, 0.1844] / 0.1645 [0.1570, 0.1724] **(re-run)** | dongu `turlar/oqa_grup_A/test/OZET.md` |
| same | TF-IDF kNN (k = 20) | test 0.1732 [0.1643, 0.1821], dev 0.1762 **(new)** | bench module run |
| SimBench (20 datasets), S score | dev marginal / uniform | 9.51 [5.13, 13.86] / −0.59 **(new)**; published best (Claude-3.7-Sonnet) 40.80 | persona/kosul `kod/simbench/PROTOKOL.md`; bench module run |
| SimBench full data (13,510 rows), S score, direct distribution, temperature 0 | Haiku 5.5 / **Sonnet 5.5** | 35.66 [33.02, 38.46] / **46.29 [43.60, 48.75]**, above the published best 40.80 (quoted, not a paired run) | persona/kosul `kod/simbench/SONUC.md` (66481eb), kanit `LIG.md` |
| same, 15% stratified subset (2,027 rows) | Opus 5.5 | 52.14 [49.18, 54.85]; Opus − Sonnet on the subset +5.99 [4.29, 7.62] | same |
| same | dev-fitted calibration | adds nothing: Sonnet +0.31 [−0.14, 0.77], Opus +0.11 [−0.57, 0.78] | same |
| same, where the gain comes from | non-public-survey datasets | tie with Claude-3.7: 14 sets 27.95 vs 27.98 (3.7 values read off the paper's Fig. 3); the gain is in public surveys. No SimBench row memorization (0/327 probes); source-survey knowledge not tested | persona/kosul `kod/simbench/SONUC.md` (26e85b2), `arastirma/karsit/SIMBENCH-KARSIT-2.md` (11cbbed) |
| CES 2020 → 2022 / 2024 opinion drift, W1 | mean shift / persistence / ABM μ=0 / bounded-confidence ABM | 0.111/0.131, 0.117/0.127, 0.126/0.129, 0.133/0.145; ABM − mean shift (2022) +0.023 [+0.012, +0.036] | makro `abm/RAPOR.md` (C2) |
| same | LLM panel | 0.152 / 0.157 (loses to persistence) | makro `RAPOR.md` (C2) |

## 2. A/B tests (Upworthy headline pairs)

| setting | method | result | source |
|---|---|---|---|
| 300 significant tests (llm300), direction accuracy, chance 0.50 | longer headline wins | 0.533 [0.477, 0.588] | deney `RAPOR.md` 1.1 |
| same | LLM direct: Haiku / Sonnet / Opus 5.5 | 0.608 / 0.688 / 0.765 [0.720, 0.808] | deney `RAPOR.md` 1.1 |
| same | TF-IDF (`methods.TfidfAB`) | 0.773 [0.727, 0.820] | deney `RAPOR.md` 1.1 |
| same | logit average 0.5/0.5 | 0.803 [0.757, 0.847] | deney `RAPOR.md` 1.1 |
| same | **stack TF-IDF + Opus (`methods.ABStack`)** | **0.817 [0.770, 0.860]**, AUC 0.906 | deney `RAPOR.md` 1.1 |
| blind one-shot, 136 new pairs | stack | 0.824 [0.757, 0.890], +7.4 [0.7, 14.7] over TF-IDF | deney `RAPOR.md` 1.3 |
| ek600 / one-shot | stack over "LLM, tie broken by TF-IDF" | +3.7 [1.0, 6.5] / +0.7 [−4.4, 6.6] | deney `RAPOR.md` 1.3 |
| LOLA v1 setup | Opus direct / bge + TF-IDF / stack + Opus | 0.645 / 0.788 / 0.775 | deney `RAPOR.md` 4.1 |
| 300 tests | Haiku 8-persona panel (`sim.ab_test`) | 0.522 [0.468, 0.575]; no-persona readers 0.528 | sahne `RAPOR.md` (Prova ii) |
| same | Haiku persona panel / Opus 10-persona panel | 0.497-0.550 / 0.685 [0.633, 0.737] | deney `RAPOR.md` |
| temporal split (newest 20% of tests) | TF-IDF Bradley-Terry | 0.758 [0.739, 0.777], 3,274 pairs **(re-run)** | deney `etki/kod/d1_sizinti.py`, `RAPOR.md` |
| official holdout, FDR | TF-IDF Bradley-Terry | 0.800 [0.782, 0.819] **(re-run)** | same |
| temporal league (test period later than the archive) | hybrid: Opus added to TF-IDF + bge | +0.041 [0.023, 0.059] accuracy; Opus alone does not beat TF-IDF + bge. **Memorization caveat:** Opus recognises Upworthy headlines (membership AUC 0.917) and loses 0.076 on paraphrases, so the hybrid gain is confounded with memorization | intent/gbdt `lig/upworthy_lig.json`, kanit `LIG.md` (probes) |
| Turkish example test | panel mean favours A (0.84), no-persona arm B (0.65), swapped-persona arm B (0.94) | flagged low confidence | sahne `RAPOR.md` |

### LLM prior + early test (`methods.EarlyTestBayes`)

| setting | result | source |
|---|---|---|
| Opus prior calibrated on 20 tests, 100 impressions/arm (`simulate_saving_calibrated`) | accuracy 0.718 → 0.808, **2.27x (1.67-3.11)** impressions | deney `RAPOR.md` 2.2-2.3, `kod/analiz/ornek_tasarrufu_llm_ga.json` |
| same protocol: Opus 150 calibration tests / TF-IDF prior / stack prior / persona-panel prior | 2.58x / 2.44x / 3.30x / 1.41x | deney `RAPOR.md` 2.2-2.3 |
| 2-fold cross-fit protocol (`simulate_saving`), 100 impressions/arm: Opus / TF-IDF / stack prior | human 0.713; 2.47x / 2.79x / 3.68x (no CI) | deney `kod/analiz/ornek_tasarrufu_llm.json` |
| 400 impressions/arm; ek600 replication; split-half | 1.34x; 2.05x (1.70-2.50); 1.99-2.28x | deney `RAPOR.md` 2.2-2.3 |
| 800+ impressions/arm | no gain; "human decides, prior breaks ties" alone 0.786 | deney `RAPOR.md` 2.1-2.3 |
| sealed expectations (ek600): 1.35x [1.15, 1.60] and 2.45x [2.20, 2.80] | results 1.34x and 2.63x, inside; third sample 5/5 inside | deney `RAPOR.md` (lines 144, 259) |

## 3. Per-person prediction

### talp73: which of 20 Amazon categories a person buys in 2022-H2 (400 people × 3 seeds, Open e-commerce)

1a = accuracy at 0.5; 1b(c) = balanced within-person top-half accuracy (primary, chance 0.5).

| method | 1a | 1b(c) | source |
|---|---|---|---|
| always "no" | 0.764 | 0.495 [0.458, 0.526] | talp73 `kart/KART.md` |
| survey-feature model (no LLM) | 0.771 | 0.588 [0.556, 0.619] | talp73 `kart/KART.md` |
| popularity | | 0.578 [0.553, 0.603]; hit@3 71.0% | talp73 `kart/KART.md` |
| decayed count | | 0.650 [0.628, 0.671] | talp73 `kart/KART.md` |
| purchase-history LightGBM | 0.807 | 0.672 [0.651, 0.693] | talp73 `kart/KART.md` |
| L0 Haiku (no persona) | 0.614-0.647 | | talp73 `kod/llm/LLM-TABLO.md` |
| **L1 Haiku (survey persona)** | **0.694 [0.677, 0.712]** | 0.564 [0.525, 0.602]; hit@3 64.0% | talp73 `kart/KART.md` |
| L1 Sonnet | 0.749 / 0.744 / 0.752 | 0.574 [0.548, 0.603] | talp73 `kod/llm/LLM-TABLO.md`, `kart/KART.md` |
| L2 Haiku (persona + history) | 0.796 [0.778, 0.816] | 2.0-2.6 points below the history model in every seed | talp73 `kart/KART.md`, `PROTOKOL.md` v1.5/v1.6.1 |
| L1 − L0 Haiku | +6.1 points [4.3, 9.0] | within-person AUC +1.3 / +1.6 / +1.4 points | talp73 `kart/KART.md` |
| L1 − L0 Sonnet | | person AUC +0.017/+0.009/+0.013; 1b(c) +0.008 [−0.011, +0.026] | talp73 `kod/llm/SONUC-v1.7c.md` |
| batch calibration (`sim.batch_calibrate`) | 69.4% → 76.5% (close to always-no) | L1 − survey model −0.013 / −0.008 / −0.002 (CI includes 0) | talp73 `KARSIT-1.md` (T1) |
| SSR (free text + embedding), calibrated | | 0.573 / 0.551 / 0.547; − L1-cal −0.015 / −0.017 / −0.039; below the survey model in every seed | talp73 `kod/llm/SONUC-v1.7d.md` |
| bench re-run, seed 0 (all eight baselines identical to talp73 output) | never 0.7665, history 0.8035 | constant 0.5062, popularity 0.5833, persistence 0.5862, decayed 0.6530, survey 0.6009, history 0.6750 **(re-run)** | talp73 `sonuc/<method>_s0.json` |

### Open e-commerce monthly order days 2022 (n = 198, Pearson)

| method | result | source |
|---|---|---|
| demographic baseline | 0.280 | sahne `RAPOR.md` (rehearsal i) |
| Haiku persona panel from `persona.from_table` | 0.618 [0.513, 0.696] (stated answer 0.649) | sahne `RAPOR.md` |
| persona + 2021 behaviour anchor | 0.860 [0.807, 0.899]; persistence 0.848; difference +0.012 [+0.000, +0.022] | sahne `RAPOR.md` |
| repeat with seed 1 | persistence 0.897 vs persona + anchor 0.879, −0.018 [−0.051, +0.019] | sahne `RAPOR.md` |
| persona-swap control / anchored no-persona | −0.032 and 0.090 / 0.849 and 0.894 | sahne `RAPOR.md` |
| order-frequency class, macro-F1 | persistence 0.781 [0.756, 0.805], majority 0.275, best GBDT +0.001 [−0.022, 0.023] | intent `RAPOR.md`; kanit `RAPOR.md` §1b |
| next basket MAP@10 | decayed frequency over P-TopFreq: OE +0.023 [0.019, 0.027], CJ +0.015 [0.011, 0.019]; LLM as a GBDT feature 0.3307 → 0.3303 | intent `RAPOR.md` |

### Twin-2K-500 (2,058 persons, 122 target columns; across-person correlation)

| method | corr | acc | source |
|---|---|---|---|
| column median | 0 | 0.755 | persona `RAPOR.md` |
| raw GPT-4.1-mini twin | 0.175 | 0.717 | persona `RAPOR.md` |
| `TwinStack` without LLM | 0.204 | 0.732 | persona `RAPOR.md` |
| `TwinStack` + LLM column | 0.275 | 0.735 | persona `RAPOR.md` |
| **fair LLM-free** (stack + pooled price logistic) | **0.342** | 0.739 | persona `RAPOR.md` |
| fair hybrid (+ LLM) | 0.347; LLM adds +0.006 [+0.004, +0.007] (an earlier +0.071 claim was retracted) | 0.740 | persona `RAPOR.md`, `RAPOR-G2-YENI.md` |
| test-retest ceiling | 0.565 | 0.817 | persona `RAPOR.md` |
| demographics-only LLM → twin persona text | 0.125 → 0.175 | | persona `RAPOR.md` (KARŞIT-2) |
| evidence card | gap closed 17% [16-19] though ratio to ceiling is 94% (demography alone 93%); de-meaned R² −0.084 [−0.095, −0.074]; sd ratio 0.75 | | kanit `RAPOR.md` §1b |
| bench hash test persons | median acc 0.757; ridge corr 0.156 [0.131, 0.183]; retest ceiling corr 0.565 **(new)** | | bench module run |

### Other person-level tasks

| task | result | source |
|---|---|---|
| Bank marketing, test AUC | constant 0.5, persistence 0.6408, **logistic 0.7483 [0.7369, 0.7592]**, LightGBM 0.6653 (collapsed under base-rate drift) **(re-run)**; LLM persona Haiku 0.6835, hybrid 0.6939, Sonnet (1,000 rows) 0.7336 | intent `gbdt/lig/bank_lig.json` |
| GSS next-wave attitudes, test mean z | persistence 0.4905 [0.4632, 0.5181], ridge 0.5124, **PCA ridge 0.5601 [0.5288, 0.5913]**; LLM-loop winner = PCA ridge **(re-run)** | dongu `turlar/gss_dalga/test/OZET.md` |
| OPeRA next web action, exact % | always terminate 4.43, repeat 13.97, zero-shot Haiku persona agent 13.67 (n=300 subsample), Markov-2 26.94 [21.42, 31.72], **LightGBM lambdarank 31.60 [26.81, 35.90]** (3-seed score ensemble); user-disjoint 29.57 vs 28.11 (not separable) | web `RAPOR.md` |
| Camera conjoint | new person: equal share 0.210, MNL 0.509, HB 0.519; same person new task: MNL 0.482, HB 0.715 (HB not ported) | deney `RAPOR.md` 1.5 |
| CPC18 risky choice | LLM as GBDT feature MSE×100 0.585 → 0.627 (worse) | deney `RAPOR.md` 1.5 |
| SocSci210 effect sizes | text kNN r 0.080 [−0.024, 0.232], ridge 0.017, Opus direct 0.509 (not presentation-ready) | deney `RAPOR.md` 1.4 |

## 4. "Worth N humans" (PPI++, n = 100)

| prior | ρ | multiplier | ≈ humans | source |
|---|---|---|---|---|
| persistence, what they did in 2021 (label-free, OE) | 0.886 | **4.36x [3.79, 5.33]** (theory 4.66) | 377 | kanit `RAPOR.md` §1a |
| LightGBM on past purchases only (OE) | 0.891 | ≈4.5x | | kanit `RAPOR.md` §1a |
| stated intent (collected after the target: upper bound) | 0.608 | 1.60x [1.41, 1.75] | 156 | kanit `RAPOR.md` §1a |
| LLM predictor + stated intent | 0.666 | 1.70x [1.49, 2.03] (statement alone 1.69x) | 149 | kanit `RAPOR.md` §1a, `RECETE.md` 9 |
| LLM persona predicts behaviour | 0.302 | 1.06x [0.97, 1.22] | 105 | kanit `RAPOR.md` §1a |
| LLM persona answers the survey | 0.240 | 1.02x [0.95, 1.15] | 101 | kanit `RAPOR.md` §1a |
| LLM twin GPT-4.1 (Twin-2K-500) | 0.284 | 1.08x [1.06, 1.10] | 107 | kanit `RAPOR.md` §1a |

- Interval coverage at n = 100: 93-94% (nominal 95%); OpinionQA 94.1-94.6% over 1,491 questions. A heavy-tailed Twin
  question (Q166) had 0.475-0.490; a 99.5% winsorised target fixes it (0.949). kanit `RAPOR.md` §1a.
- Using the population variance in λ shows OE GBDT at n = 25 as 2.0x instead of 5.1x. kanit `RECETE.md` 6.
- Upworthy winner selection: direct model prior 2.05-2.27x at 100 impressions, 1.17x / 1.12x at 400, 0.88-0.99x at
  3,200. kanit `RAPOR.md` §1a.
- Under the badge rule, none of the four example cards (Twin, OE value, OE class, OE LLM-500) reaches "high
  confidence". kanit `RAPOR.md` §1c.

## 5. Agent-based models

| setting | result | source |
|---|---|---|
| OWID internet adoption, test MAE (pp), 7 seeds | **ABM + ABC + analogy 8.48 ± 0.16**; Bass-simulator control 9.73 ± 0.12; linear 10.72; Gompertz 11.61; Bass NLS 14.86; naive 18.39; paired ABM − Bass control −1.25 ± 0.21 per seed; seed-averaged series bootstrap −1.32 [−1.83, −0.82]; better on 58/81 series | makro `RAPOR.md` (M1), `abm/m1/NOTLAR.md` |
| same | 81-85% of the gain over Bass NLS from ABC + analogy prior, 15-19% from the ABM structure | makro `abm/m1/NOTLAR.md` |
| same, h = 1 | ABM 2.64, Gompertz 2.13, linear 2.20 (simple baselines win) | makro `abm/m1/NOTLAR.md` |
| US household technologies | plain ABC 8.54 vs linear 8.70 (n.s.); analogy prior hurts (10.07) | makro `abm/m1/NOTLAR.md` |
| 80% band coverage | plain ABC 0.54 (internet) / 0.64 (US); analogy 0.77-0.80 | makro `abm/m1/NOTLAR.md` |
| NPE variant (not ported) | 12.30 vs ABC 10.78 / analogy 9.56 (internet dev) | makro `abm/m1/NOTLAR.md` |
| Dominick's price elasticity, WAPE dev/test | pre-event mean 0.598/0.616; category log-log 0.466/0.479; LLM expert + data 0.469/0.469; (category + LLM)/2 + data 0.465/0.470 | makro `RAPOR.md` (M2) |
| same | mix test gain −0.0072 [−0.011, −0.003] equals the LLM-swap placebo (−0.0085 / −0.0066 / −0.0039): the gain is shrinkage, not LLM knowledge | makro `llmabm/RAPOR.md` |
| market ABM | dev WAPE 0.465 (120 TPE) / 0.456 (300 TPE) vs category log-log 0.434 and GBDT 0.373; C3 test 0.612 ± 0.102; as GBDT feature +0.005 [−0.020, +0.030] | makro `abm/RAPOR.md`, `RAPOR.md` (C3) |
| retweet curves, Twitter15/16 test, 24 h curve RMSE | fixed median curve 0.920; shape baseline (first hour) 0.391; Hawkes 0.391 (difference 0.000 [−0.006, +0.007]); GBM on early features (not ported) 0.289 | makro `abm/yayilim24/SONUC.md` |
| same, text only | LLM agent society (20 Haiku personas × 3 turns) 0.771; single Haiku score 0.772 (society − single −0.001 [−0.010, +0.009]); dev-calibrated constant curve 0.772; Sonnet subsample 0.801 vs Haiku 0.785 | makro `abm/yayilim24/SONUC.md` |
| SEISMIC, 39,263 cascades | shape baseline 0.267; SEISMIC-style 0.308 (+0.041 [+0.038, +0.043], worse); GBM 0.187 | makro `abm/yayilim24/SONUC.md` |

## 6. Research loop

| loop | result | source |
|---|---|---|
| OpinionQA setting A | dev gain kNN vs position prior −0.0106 [−0.0153, −0.0056]; **test** winner − baseline +0.0025 [−0.0018, +0.0070] (did not hold); LLM − random winner on selection −0.0030 [−0.0096, +0.0032], on test −0.0062 [−0.0109, −0.0017]; observer direction right 1/4 | dongu `turlar/oqa_grup_A/test/OZET.md`, `llm/sonuclar.tsv` |
| OpinionQA setting B | nothing beat copying Overall; LLM = random = baseline on test (WD 0.0649); 1/4 proposals invalid | dongu `turlar/oqa_grup_B/test/OZET.md` |
| Twin-2K-500 kit | test z: LLM winner 0.3738, random winner 0.3720, fair baseline 0.3679; winner − baseline +0.0058 [+0.0021, +0.0094]; **LLM − random +0.0018 [−0.0011, +0.0046]**; 3/7 proposals invalid, repaired | dongu `turlar/twin_kit/test/OZET.md` |
| GSS waves | LLM-loop winner = PCA ridge baseline | dongu `turlar/gss_dalga/test/OZET.md` |
| null calibration (synthetic, this library) | 1000 one-turn loops with no effect kept 37 (3.7% per turn, nominal 2.5%) **(new)** | research module scratch run, `docs/modules/research.md` |

## 7. Cost and speed

| item | result | source |
|---|---|---|
| Haiku 5.5, effort low, per call | L1 $0.000265, L2 $0.000517 (dry runs); full L1 run 971 calls $0.2577, L2 $0.5315 | talp73 `PROTOKOL.md` v1.4, `kod/llm/LLM-TABLO.md` |
| Sonnet 5.5 L1 | $0.00507/call; full run 976 calls $4.9455 | same |
| raw CSV → personas → Haiku sim → evidence | OE 3 min 52 s (63 calls, $0.105); Upworthy 7 min 2 s (121 calls, $0.20); non-LLM steps < 10 s | sahne `RAPOR.md` |
| Turkish brief → 12 personas | 41 s, 2 calls; `validate()` caught an out-of-enum value | sahne `RAPOR.md` |
| `claude -p` | sahne clean clone 103 calls, 0 errors, $0.155 list price; each call writes ~2.4K tokens to cache, never re-read | sahne `RAPOR.md`, deney `RAPOR.md` |
| evidence report | `--hizli` 8.6 s synthetic; full B = 1000: OE 23-37 s, Twin 179-219 s | kanit `RAPOR.md` §3 |
| ABM bank | cold 53-85 s at default size; warm fits seconds | makro `abm/m1/NOTLAR.md` |
| retweet society | 3,348 calls, $0.94 | makro `abm/yayilim24/SONUC.md` |

## 8. Economic games, persona methods and Big Five fidelity (earlier repos, non-Claude models)

**These rows are results from two earlier repositories, not from the TALP research code, and none was measured on
Claude.** replicant ran on stepfun/step-3.5-flash; edsl-test ran on gpt-oss-120b, except the per-method paper
replications, which used the model each paper used (listed in the row). No ranking here transfers to Claude until it
is rerun through `simengine.llm`. Sources are written relative to each repo's root (`replicant: ...`,
`edsl-test: ...`). The ported code lives in `persona.methods`, `persona.validate`, `sim.constrained`, `games`,
`abm.run_who_to_punish`, `methods.mixture` and the `econ_games_b01` / `ertan_wtp` / `beebench_b01_prior` bench
tasks. **Checked** means the number was re-read from the source file during the port (the port's own check, not a
re-run); rows without it are carried from the porting notes.

| # | result | caveat | source |
|---|---|---|---|
| 1 | Persona fidelity, BFI-2 sentences assigned, Mini-IPIP measured: pooled r = .911, R² = .830, MAE = .666. Bias A +0.735 (4.02 → 4.75), N −0.692 (2.94 → 2.25), O +0.660, C +0.444, E −0.034 (**checked**; `sample_bigfive(20, seed=42)` reproduces the assigned scores exactly) | stepfun/step-3.5-flash, N = 20, seed 42; NaN answers were imputed as 3 in the source (simengine's `validate_bigfive` does not impute). Rank order survives while levels drift | replicant: `papers/ertan2009/results/validation_continuous.json` |
| 2 | Ertan vignette: no-persona agents fit the institutional result best, vote punish-low 0.94 (47/50), punish-high 0.00; random Big Five personas 0.80 / 0.00; hyper-cooperative personas 0.46 (**checked**) | one-shot individual vignettes, not the original repeated game (never call it a replication); individual votes are compared with group votes | replicant: `papers/ertan2009/results/otree_{default,random,cooperative}_summary.json`, `comparison.json` |
| 3 | Completion cliff, hyper-cooperative condition: 30/50 finished the regime page and 7/50 the punishment page, vs 50/50 in default and random. Dropout is not random: on the regime page, non-completers' baseline contribution was 17.6 (n = 20) vs 15.83 for completers (n = 30); on the punishment page, 16.44 for non-completers (n = 43) vs 17.14 for completers (n = 7) (**checked**, recounted from the raw JSON); this port's Wilson CI for 7/50 is 0.14 [0.07, 0.26] **(new)** | the summary JSON wrongly says n = 50; all 43 field failures ended with an empty last response, but the reasoning-budget cause was never confirmed by a rerun | replicant: `papers/ertan2009/results/otree_cooperative_raw.json`, `otree_cooperative_flat.csv`, `papers/ertan2009/replicated/paper.tex` |
| 4 | Harness sensitivity, same model and temperature: random-persona vote punish-low 0.14 (EDSL harness) vs 0.80 (oTree harness); default-condition baseline contribution 0 vs 1.0 (**checked**) | single runs | replicant: `papers/ertan2009/results/big5_random_summary.json`, `default_summary.json`, `otree_random_summary.json`, `otree_default_summary.json` |
| 5 | Legacy trait-behaviour correlation with baseline contribution: A r = +0.54, N r = −0.43 | traits were rebuilt by re-seeding the RNG (fragile); N = 50; about 30 uncorrected p-values in that repo | replicant: `papers/ertan2009/results/big5_random_baseline.csv`, `papers/ertan2009/legacy/fig_personality.py` |
| 6 | Cost: output tokens about 20× input tokens on a reasoning model | the logged $13.10 uses placeholder $1/$1 per M prices and the paper's $0.45 has no log, so only the token ratio is reported | replicant: `papers/ertan2009/results/*_{baseline,voting,regimes,punishment}.csv`, `papers/ertan2009/replicated/paper.tex` |
| 7 | Who-to-punish human-likeness index (9 criteria, mean of 3 seeds per method; value_anchor is the mean of 4 runs because seed 1 ran twice, .659 and .968): construction .813, demographic .797, value_anchor .786, twin2k .783, baseline .694, big5 .688, gps .688, homo_silicus .677, anthology .617 (**checked**); within-method range up to 0.37 across seeds | gpt-oss-120b; 1 group of 4 per seed; ranks inside the top 4 are not significant; criterion weights not sensitivity-checked; baseline and homo_silicus have effective n = 1 (cached answers). `ertan_wtp` ports the scorer, but these indices come from the source's original scorer: simengine makes punishment_targeting N/A where the source wrote 0.0 and never_punish_high N/A when no vote records punish_high, so runs with no paid punishment (e.g. anthology seed 2 and big5 seed 2, targeting 0.000 in the source) get different index values here. A constant full-cooperation run scores 0.769 on 6 of 9 criteria **(new)**, so read coverage next to the index | edsl-test: `benchmark/REPORT.md`, `benchmark/wtp_scores.csv` |
| 8 | The bare model always dismantles the punish-low institution (final_vote_punish_low and no_dismantling = 0 in all 3 seeds); "never allow punishing high contributors" held in every run (never_punish_high = 1.000 in all 28 runs of `wtp_scores.csv`, i.e. 56/56 group votes at 2 votes per run; REPORT.md says 54/54, which predates the csv's extra value_anchor run and does not match it) (**checked**) | same as row 7 | edsl-test: `benchmark/REPORT.md`, `personas/RESULTS.md` |
| 9 | Bare-model corner behaviour: dictator $0, trust $0 sent (10/10), public goods $0 of $20 (4/4), PD defect 5/5 rounds, bomb risk exactly 50 boxes with SD 0 (trust, PG, PD, bomb **checked**) | early single runs; who-to-punish runs before commit 15b9531 turned None answers into $0 (simengine never does) | edsl-test: `experiments/0[3-6]*/results/*/summary.txt`, `personas/METHODS.md` §C |
| 10 | `max_tokens` acts as a treatment: the bare model punishes free-riders at 8192 but not at the default cap | single run; the default-cap morning run (07-37-23) was partly None-corrupted by truncation, so it is not evidence about model behaviour: the source says the "never punishes" claim "conflated model behavior with a truncation artifact". `games.play` and `abm.run_who_to_punish` store `max_tokens` on every row for this reason | edsl-test: `experiments/07_who_to_punish/results/2026-06-11_15-08-37_baseline/ANALYSIS.md`, `personas/README.md` |
| 11 | BeeBench b01: 10,920 rows (7 methods × 14 games × 10 agents × 3 runs), $1.27, 30 invalid (0.27%); twin2k trust_sender 23% invalid (7/30) and twin2k pg_punishment 23% invalid (21/90), so 28 of the 30 invalid rows are in these two cells (**checked**; `games.read_results` parses all 10,920 rows) | b01 itself has no human-distance scoring; per-game numbers must come from `games.distance_to_human` (see the next table) | edsl-test: `benchmark/beebench/b01/results.csv`, `benchmark/beebench/b01/ANOMALIES.md` |
| 12 | Paper replications 8/8 PASS (**checked**: every `replication.json` says pass), e.g. twin2k holdout accuracy .7075 vs paper .6802 (test-retest .7475; gpt-4.1-mini); demographic 24/24 strong partisans aligned (Llama-3.1-70B); construction MAE .347 vs paper .20, i.e. worse (gpt-4o) | small n; mostly implementation-fidelity checks; models differ by method (gpt-4o, gpt-4.1-mini, Llama-3.1-8B/70B, phi-4) | edsl-test: `personas/*/replication.json`, `personas/*/REPLICATION.md` |
| 13 | homo_silicus calibrated weights: self_interested .36, efficient .50, selfish_but_fair .14; train SSE .089; held-out dictator RMSE .315; the reciprocal type gets 0, so the method can never legalise punishment (`methods.fit_mixture_weights` reproduces .36/.50/.14, SSE .088936, RMSE .31485 from the stored behaviour vectors **(re-run)**) | gpt-oss-120b behaviour vectors; one grid fit | edsl-test: `personas/homo_silicus/weights.json`, `personas/homo_silicus/NOTES.md` |
| 14 | Literature anchors (paper numbers, not repo runs): Park et al. 1,052 people, 85% of test-retest (v1; v3 says 83%); SSR 90% of test-retest; SimBench 40.80/100; analytic flexibility r .23 to .84; social desirability 1.20 SD (GPT-4); conjoint augmentation saves 24.9-79.8% | verified against the arXiv abstracts where `docs/landscape.md` says "verified". **Do not cite** Bisbee 48% / 32% and Larooij–Törnberg 15/35 until checked against the paper; General Social Agents 53-73%, Twin-2K ~87% and Bisbee "33 vs 270+" come from vault summaries, not abstracts | awesome-llm-social-simulation notes vault; per-number tags in `docs/landscape.md`, `docs/validation.md`, `docs/failure-modes.md` |

**Distance to human on the b01 table (new).** simengine's own computation on the edsl-test b01 results (read in
place, not copied) with `bench.get("beebench_b01_prior")`: mean absolute error of share-scale moments against the
cited human values, B = 1000 resampling whole games. Lower is better.

| arm | holdout MAE, 8 moments / 4 games | all-games MAE, 24 moments / 12 games |
|---|---|---|
| b01 gps | 0.113 [0.070, 0.187] | 0.171 [0.125, 0.218] |
| b01 twin2k | 0.185 [0.104, 0.239] | 0.179 [0.128, 0.213] |
| midpoint baseline (uniform answers) | 0.201 [0.100, 0.316] | 0.254 [0.143, 0.351] |
| b01 anthology | 0.202 [0.116, 0.289] | 0.188 [0.132, 0.251] |
| b01 demographic | 0.212 [0.124, 0.317] | 0.283 [0.209, 0.357] |
| b01 big5 | 0.246 [0.167, 0.363] | 0.220 [0.166, 0.270] |
| b01 value_anchor | 0.250 [0.133, 0.458] | 0.297 [0.187, 0.401] |
| bare model (b01 `baseline`) | 0.341 [0.212, 0.506] | 0.404 [0.334, 0.470] |
| rational corner baseline | 0.452 [0.326, 0.590] | 0.456 [0.390, 0.531] |

Caveats: one model (gpt-oss-120b), 10 agents × 3 runs per method, the holdout CI rests on 4 game clusters, so the
order among persona methods is not significant; several human references are approximate (see
`docs/modules/games.md`). The one robust reading: the bare model is worse than uniform-random answers on both
splits. Source: edsl-test: `benchmark/beebench/b01/results.csv`; computation `docs/modules/bench.md`
("Economic-game tasks").

## 9. Blind rehearsals of this library (22 Oct 2026)

A fresh agent got a task it had not seen, a 2-hour clock, and hidden test answers. D1 used the earlier team kit; D2-D4
used only this library (the version named in the row). It sealed its predictions before the truth was opened, and the
lead agent scored them with its own script (cluster bootstrap, B = 1,000). One run each; the gaps each team logged
were fixed in the next library version. Source: prova `d<N>/SONUC-MAIN.md`.

| run | task | time used | primary metric, team | vs hidden baseline | LLM role, cost |
|---|---|---|---|---|---|
| D1 (kit) | person policy support from survey answers, unseen countries | 63 min | Spearman 0.306 [0.250, 0.342] | B1 0.242; +0.066 [0.026, 0.101] | country prior; $0.15 |
| D2 (simengine dc2021c) | same task | 37 min | Spearman 0.283 [0.232, 0.320] | +0.042 [0.006, 0.084]; dev CV 0.311 was optimistic | country prior; persona at chance, dropped; $0.32 |
| D3 (simengine 018b1a2) | new-film cold start, film residual | 19 min | Spearman 0.897 [0.871, 0.917] (ceiling 0.956) | B1 0.567; +0.330 [0.255, 0.411] | the LLM gain was title recognition: ours − LLM-free arm +0.050 on dev, +0.017 [−0.001, 0.035] on test; $0.74 |
| D4 (simengine 000c581) | weekly demand, 30,490 series × 4 weeks | 20 min | WMAPE 0.347 [0.334, 0.360] (oracle-level 0.253) | hidden LightGBM 0.357; −0.0097 [−0.0120, −0.0076] | event prior, named + blind; failed its pre-registered drop condition, not submitted; $0.0015 |

What carried over: every team built its LLM-free ladder first and sealed before the truth; the D3 and D4 cards say
"passed on dev", not "passed". What did not: the LLM arm never beat the strongest LLM-free arm on test by a clear
margin, and D4's bias expectation (< 3%) failed (−4.0%).

## Measurement facts that shape the defaults

- Haiku/Sonnet 5.5 reject a non-default `temperature` with HTTP 400; the Claude backend never sends one
  (talp73 `PROTOKOL.md` v1.4).
- `cache_control` on a ~50-token system prompt does nothing (`cache_creation_input_tokens = 0`)
  (talp73 `kod/llm/PROTOKOL-v1.5-ek.md`).
- Haiku adaptive thinking at effort low used 0 thinking tokens on L1; r = 0.90 vs thinking off (talp73 `PROTOKOL.md`).
- A key returned 401 mid-run and two arms never ran; hence stop-on-401 (talp73 `kod/llm/SONUC-v1.6-L1sonnet.md`).
- `claude -p` can inherit the user's language setting; with `--setting-sources ""` 2,362 cached answers had no Turkish
  characters (deney `RAPOR.md`; sahne `RAPOR.md`).
- SSR embeddings: bge-base Spearman 0.97 vs MiniLM 0.62 on 10 hand-labelled sentences (a tiny check) (sahne `RAPOR.md`).
