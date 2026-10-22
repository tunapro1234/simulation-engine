# Methods

One paragraph per method: the idea, when it works, and when it fails. Every number comes from
[RESULTS.md](RESULTS.md), which names the branch and file each one was measured in. The per-module docs in
`docs/modules/` give the API.

## Baselines (`simengine.methods`)

**Persistence** (`PersistenceBaseline`, `persistence`). Predict that each unit does what it did last time. It works
whenever behaviour is habitual and there is a past period: OE monthly order days (0.848-0.897 Pearson), order-frequency
class (macro-F1 0.781, which no GBDT beat), and as a label-free PPI prior worth 4.36x the humans. It fails when there
is no history (new people, new products) or when the target is a change rather than a level. Always put it in the
ladder; it was the number to beat in most of the research.

**Item marginal / overall copy / uniform** (`ItemMarginal`, `overall_copy`, `uniform_distribution`). Predict the
answer shares of the whole population for every segment. If the overall marginal is known it is very hard to beat:
in OpinionQA setting B nothing beat copying Overall. Uniform is the floor. Use these as the ladder for any distribution
task.

**Popularity / decayed frequency / TIFU-kNN** (`popularity`, `decayed_frequency`, `tifu_knn`). Basket and next-purchase
baselines. Decayed frequency beat P-TopFreq on next-basket MAP@10 and reached 0.650 1b(c) on talp73; an LLM added as
a GBDT feature did not help (0.3307 → 0.3303). They fail for cold-start users.

**Logistic / LightGBM / kNN** (`LogisticBaseline`, `LightGBMBaseline`, `KNNBaseline`, `TextKNNRegressor`). The plain
learners. Logistic regression was the best arm on bank marketing (AUC 0.748) while LightGBM collapsed under base-rate
drift (0.665), so try the simple model first when train and test periods differ. The purchase-history LightGBM is the
strongest person-level model on talp73 (0.672). Text kNN on SocSci210 effect sizes was near zero (r 0.080).

**Single-text binary** (`TfidfBinary`, `target_encode`). Word + character TF-IDF logistic on one text per row
(a first search query, a review) for a 0/1 outcome, with optional extra and out-of-fold target-encoded columns and C
chosen inside the training rows. On ShopCART query -> purchase it was the strongest LLM-free arm (test AUC 0.666 from
30,365 training sessions), but it needs thousands of labels: 0.579 from 500 sessions, 0.51-0.54 from 50-200.

**RCT transfer** (`rct_effects`, `rct_transfer`). When the same conditions were randomised in other blocks
(countries, sites, periods), the within-block effects from those blocks are the forecast for new ones. On the
climate many-labs data 32 countries ranked 11 messages at Spearman 0.845 in 31 new countries (noise ceiling 0.908);
no LLM arm came close, and adding one hurt. A single source country is weak (median 0.28).

## A/B (`simengine.methods`)

**TF-IDF pair model** (`TfidfAB`, `TfidfWordAB`). A logistic model on the TF-IDF difference of two texts. With a few
hundred past tests from the same platform it is strong: 0.773 on 300 Upworthy tests, 0.758 on a temporal split. It
needs past tests with the same outcome; with no history it has nothing to learn from.

**LLM direct** (`LLMDirectAB`). Ask the model which text wins and by how much. It scales with model size (Haiku 0.608,
Sonnet 0.688, Opus 0.765 on Upworthy) and needs no training data. On its own it is below TF-IDF; on LOLA v1 Opus
direct (0.645) was well below bge + TF-IDF (0.788). Expect memorisation risk on public data.

**Stack** (`ABStack`, `LogitAverageAB`). A logistic stack of the TF-IDF and LLM logits. This is where the LLM paid off:
0.817 on Upworthy and 0.824 on a blind one-shot of 136 new pairs (+7.4 points over TF-IDF). The gain over the simpler
"LLM, tie broken by TF-IDF" rule was small (+3.7 on ek600, +0.7 on the one-shot, CI includes 0). It needs both training
pairs and LLM calls.

**LLM prior + early test** (`EarlyTestBayes`, `calibrate_prior`, `simulate_saving_calibrated`, `simulate_saving`). Use the model's prediction as a
Bayesian prior on the log-odds ratio and update it with the first impressions of a live test. With 100
impressions per arm it was worth 2.27x the traffic (Opus prior calibrated on 20 tests, `simulate_saving_calibrated`;
the 2-fold `simulate_saving` protocol gives 2.47x on the same data); the stack prior gave 3.30x,
a persona-panel prior 1.41x. The gain shrinks as traffic grows (1.34x at 400 per arm) and disappears at 800+. The
prior must be calibrated on past tests first.

## Persona panels (`simengine.persona`, `simengine.sim`)

**Personas from a table or history** (`persona.from_table`, `from_row`, `from_history`). Turn CSV rows or purchase
histories into persona text with the forbidden columns removed; `validate()` catches out-of-enum values. This is
plumbing: it makes a panel quickly (a Turkish brief to 12 personas in 41 s), but the persona itself rarely carries
the signal. Swapping personas changed little on OE (−0.032, 0.090), and on Twin a demographics-only persona gave 0.125
against 0.175 for the full text.

**Survey panel** (`sim.survey`, `SSR`). Each persona answers a question, either as a structured score or as free text
mapped to a scale by embedding similarity (SSR). With a behaviour anchor (last year's value in the persona) it matched
persistence (0.860 vs 0.848, OE), but without the anchor it was far below (0.618). For per-person purchase
prediction the L1 persona was below "always no" on accuracy and below the survey-feature model on 1b(c); SSR was lower
still. Use it for reasons and quotes, or as a feature, not as the predictor.

**A/B panel** (`sim.ab_test`, `confidence_check`). Personas vote between two texts; controls (no persona, swapped
persona) feed a confidence verdict. Haiku panels were at chance on Upworthy (0.522) and no better than readers with no
persona (0.528); an Opus 10-persona panel reached 0.685, still below TF-IDF. The confidence check did its job on the
Turkish example: the arms disagreed, so it flagged low confidence. Do not let the panel make the decision.

**Roundtable** (`sim.roundtable`). Personas see each other's answers and revise. 5-8% of answers changed and the
distribution did not move measurably (R5 ΔW1 −0.005, CI includes 0); a confirmation-bias instruction gave a small gain
(−0.007). Use it for the demo story, not for accuracy.

**Direct segment distribution** (`sim.segment_distribution`, `persona_distribution`). Skip the personas and ask the
model for each segment's answer shares, averaging named framings. On CES 2020 it closed 0.818 of the gap from the
general marginal to a 37-person survey (W1 0.077), against 0.55 for the persona panel. It worked best when there was
no data at all, but it lost to the general marginal on 2 of 8 items, and memorisation of a public survey is not ruled
out. Seal the prediction before opening the truth. A hidden framing (no survey name, no year) still reached 0.774.

**Per-person prediction** (`sim.predict_people`, `batch_calibrate`). L0/L1/L2 prompts (no persona, survey persona,
persona + history) for "will this person buy category X". The persona adds a measurable but small within-person signal
(L1 − L0 +6.1 points accuracy for Haiku), but it never beat the history model; L2 stayed 2.0-2.6 points below it in
every seed. Batch calibration lifts accuracy to about the always-no level and does not change the ranking.

## Stacking and shrinkage (`simengine.methods`)

**TwinStack / DevStack / pooled logistic** (`TwinStack`, `DevStack`, `pooled_logistic_oof`). Stack column-level
baselines with optional LLM columns on dev, then predict. On Twin-2K-500 the fair LLM-free stack reached corr 0.342
and adding the LLM column gave only +0.006. An earlier +0.071 claim for the LLM was retracted once the LLM-free stack
was made fair. Always compare against a stack built without the LLM.
`DevStack` stacks a learner's dev predictions with the LLM's (logistic on logits for binary) and keeps the
cross-fitted dev predictions; `BinaryCalibrator` recalibrates one probability column on dev (Platt, isotonic
from 1,000 rows).

**Shrinkage toward a prior** (`ShrunkDeviation`, `ridge_to_prior`, `abm.ElasticityShrinkage`). Shrink noisy per-unit
estimates toward a prior (category, LLM expert). On Dominick's elasticities the LLM-plus-category mix helped slightly
on test, but an LLM-swap placebo helped just as much: the gain is shrinkage, not LLM knowledge. Run the placebo before
crediting the LLM.
For segment distributions `ShrunkDeviation(distribution=True)` keeps the known population distribution as the
level and adds alpha times the LLM's segment deviation from its own population answer (alpha picked on training
segments, rows renormalised).

**Choice models** (`MNL`). Multinomial logit for conjoint. It works for new people (0.509 vs equal share 0.210), but
for repeat respondents hierarchical Bayes is much better (0.715 vs 0.482), and HB is not ported.

**Next action** (`Markov2Baseline`, `NextActionRanker`). Markov-2 transitions plus a LightGBM lambdarank over
candidate actions. On OPeRA it reached 31.60% exact against 13.67% for a zero-shot Haiku persona agent (that one on an n=300 subsample). With
user-disjoint splits the two learned models could not be separated (29.57 vs 28.11).
For session outcomes (convert or abandon, judged from the first k steps) `next_action.sessions` builds the
leak-free session table (prefix cut before the outcome step) and `next_action.funnel` the per-step funnel.

## Agent-based models (`simengine.abm`)

**Adoption ABM + ABC + analogy** (`AdoptionABCAnalogy`, `adoption_backtest`). A small adoption simulator whose
parameters are fitted by approximate Bayesian computation, with a prior borrowed from analogous series. On OWID internet
adoption it had the best test MAE (8.48 vs the Bass-simulator control 9.73), but 81-85% of the gain came from ABC +
analogy, not the agent structure. At h = 1 simple curves win, on US household technologies the analogy prior hurt, and
plain ABC bands under-cover (0.54). Always run the Bass control (`BassABCAnalogy`).

**Opinion drift ABM** (`OpinionPanel`, `BoundedConfidenceABM`). Bounded-confidence dynamics over a segment panel.
On CES 2020 → 2022/2024 it lost to the mean-shift baseline (+0.023 W1 in 2022), but the LLM panel did worse (0.152).
Treat this as a scenario tool, not a forecaster.

**Market ABM** (`MarketABM`, `market_sim`). Agents with price sensitivity reacting to promotions. Its dev WAPE (0.456-0.465)
was behind both category log-log (0.434) and GBDT (0.373), and as a GBDT feature it added nothing (+0.005, CI includes 0).

**Retweet curves** (`hawkes_mean_field`, `SeismicPredictor`, `shape_forecast`). Predict a cascade's 24 h curve from its
first hour. A shape baseline (scale the median shape by the first-hour count) and a Hawkes mean-field model tied
(0.391 each); the SEISMIC-style predictor was worse than the shape baseline. A GBM on early features (0.289) was best
and is not ported.

**Agent society** (`AgentSociety`, `ShareCascadeWorld`). 20 LLM personas over 3 turns decide whether to share a text.
From text alone it equalled a single Haiku score (0.771 vs 0.772), and both equalled a dev-calibrated constant curve.
The planned falsification fired: the society adds nothing over one call. Use it for illustration only.

## Evidence (`simengine.evidence`)

**Baseline ladder + card** (`evaluate`, `card`, `dashboard`, `baseline_ladder`). One call scores the method, its raw
version and every baseline with paired bootstrap CIs, a shuffle control, recalibration, split and memorisation audits,
and gives a badge. Its value is honesty: none of the four example cards in the research reached "high confidence".
It cannot make a weak method look good; it can only say so in a form judges trust.

**Worth N humans** (`worth_humans`, PPI++). Combine a small labelled sample with predictions on many unlabeled units;
the multiplier says how many humans the predictor is worth. Coverage at n = 100 was 93-94%. Persistence was worth 4.36x;
LLM persona predictions about 1.0-1.1x, so in the measured tasks LLM personas did not replace humans. Heavy-tailed
targets need winsorising (Q166 coverage 0.475 → 0.949).

**Seal / pre-registration** (`seal`, `verify`, `research.Prereg`, `lock_prereg`). Hash the predictions and the plan
before the truth is seen. The sealed early-test expectations held (1.34x and 2.63x inside their sealed ranges). It costs
nothing and answers "did you tune on the test?".

## Research loop (`simengine.research`)

**Loop with proposers** (`Loop`, `RandomProposer`, `GridProposer`, `GreedyCoordinateProposer`, `HalvingProposer`,
`LLMProposer`, `ClaudeCodeProposer`). Propose a config, run it on dev, keep it only if a paired bootstrap says it beats
the incumbent, confirm once on a selection split, and open the test split at the end. The loop's value is the
discipline: on OpinionQA setting A a dev gain did not hold on test (+0.0025, CI includes 0). An LLM observer was not
better than random search (Twin kit LLM − random +0.0018, CI includes 0; on OpinionQA A random was better on test).
In synthetic null runs the per-turn false keep rate was 3.7% against a nominal 2.5%, so treat a single keep as weak
evidence.

## Benchmarks (`simengine.bench`)

**Bench tasks** (`bench.run`, `list`). Seven tasks (bank marketing, GSS waves, Open e-commerce, OpinionQA groups,
SimBench, Twin-2K-500, Upworthy) with the research splits and baselines. Several baseline rows were re-run and match the
research numbers (GSS PCA ridge 0.560, talp73 baselines identical). The data is not shipped; a missing data root
gives `missing_data`, not an error.
