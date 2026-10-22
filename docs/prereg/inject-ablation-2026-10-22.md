# Pre-registration: Persona-injection ablation on real purchase and survey data (claude-haiku-5-5), 2026-10-22

_Written and locked before any result was computed. Never edited after the lock: changes go into dated, separately locked addenda next to this file (`<name>-EK<k>`, research.add_addendum)._

## Question
Which persona-injection methods of simengine.persona.inject carry person-specific predictive signal (beat the same injector fed another person's persona, and beat a no-persona prompt) when a real LLM predicts (A) next-6-month category purchases in the talp73 setup and (B) Twin-2K-500 wave-4 answers; and does any of them beat the strongest LLM-free baseline on the same cells?

## Data
- A: Open e-commerce 1.0 (Berke et al. 2024, CC0), exactly as prepared by the TALP talp73 pipeline (talp-simulation-hackathon-2026 uretim/talp73, PROTOKOL v1.0 V1-V5, degerlendir.py 71a4e82): fully observed users, the 20 fixed categories, seed-0 sample of 400 persons. Person fields = the 13 talp73 survey items (istem.SORULAR, exact labels). History = that person's purchases before 2022-07-01 (date, title cut to 90 characters, category, unit price, quantity). Target = at least one purchase in the category 2022-07-01 .. 2022-12-31.
- A cells: the talp73 1b balanced mask (per person min(5, bought, not bought) of each, default_rng(1000)), minus the 3 persons with no purchase before the cutoff (log injectors need a history): 359 persons, 2708 cells; cells sha256 04d044090eb0f5a497dfe812d70ee8d2f03c45e56025d2419770b2f2c79543ec.
- B: Twin-2K-500 (Toubia et al. 2025, CC BY 4.0). simengine.bench Twin2K500 split='test' (25% of persons by the fixed id hash); 80 test persons drawn with numpy default_rng(0) (sorted ids). Target items = the scored heuristics-and-biases columns that were asked again in wave 4, pricing block (*_Q295) excluded: 82 items. Truth = the person's wave-4 answer (raw code; anchoring items in scored deciles at scoring time). Only cells with a wave-4 answer: 4360 cells; cells sha256 48961005ebe6588dd72967cb350c8f160d1107c46c15d77dd6f7c7cefe5b817c.
- B persona content: wave 1-3 answers of the Demographics, Personality and Economic preferences blocks (546 bank items; Demographics as person fields, the rest as answers). The person's own wave 1-3 answers to the target items are NOT shown to any injector.
- Licensed data and every prediction stay in the TALP work directory outside this repo; this file holds no data rows, person ids or answers.

## Split
- A: same persons, categories, window and 1b mask as talp73 seed 0; nothing is trained on these persons.
- A fitted_topk is fitted LLM-free (random forest) on the talp73 fully observed users NOT in the seed-0 sample, labels = category purchase in 2022-01 .. 2022-06 (the window talp73's classical baselines train on), fit question ids differ from the target ids. Chosen keys (fixed now): ['State of residence', 'How often the account orders Amazon deliveries', 'Household income before taxes, past 12 months', 'Number of people who log in and order with this Amazon account', 'Age group'].
- A llm_narrative: one generation call per person from the person fields only (150 words, max_tokens 400), before any target call; the narrative never sees history or targets.
- B neighbour pool: 300 Twin dev persons (default_rng(1)) with their own wave-4 answers to the target items; no test person is in the pool, and the swap fix (c09ccc5) keeps the scored person out.
- Swap control: compare_injectors' swap_<injector> (seeded derangement, seed 0): same prompt with another person's persona, scored against the original person.

## Metrics
- **A primary**: talp73 1b(c) accuracy: within each person, the top half of their 1b cells by p is 'yes' (ties broken by talp73's G matrix, default_rng(1000)); share of correct cells. Higher is better, chance 0.5. p = parsed answer / 100.
- **A secondary**: mean within-person AUC (talp73 interpretation 4) on the same cells; 1b(a) accuracy (p >= 0.5).
- **B primary**: official Twin-2K-500 MAD task accuracy against the wave-4 answer: per cell 1 - |pred - truth| / range in scored units (anchoring items to wave 1-3 deciles, bench twin to_scored), mean over a person's tasks, then over persons. Higher is better. Option text -> its 1-based position, the code the wave-4 file uses (checked on the dev pool before any test reply is parsed).
- **B secondary**: between-person correlation per column (mean Fisher z, bench 'mean_z'); exact-match accuracy (compare_injectors' own table).
- **Unparsed / failed replies**: A: p = 0.5; B: the dev-person median of that item. Parse rate is reported per arm; an arm under 95% parsed is flagged 'format failure' and its verdict is not interpreted.
- **Uncertainty**: person-level bootstrap, B = 1000, numpy default_rng(42) (talp73 degerlendir.agirliklar weights for A, the same resamples for every arm); differences are paired on the same resamples; 95% percentile CI.

## Baselines
- Controls (both parts): no_persona (bare question) and swap_<injector> for every injector.
- A, LLM-free, talp73 seed-0 prediction files scored on these exact 2708 cells (already published talp73 numbers, seen before this file): constant 0.5059 [0.4858, 0.5259]; popularity 0.5820 [0.5632, 0.6006]; survey model 0.5990 [0.5812, 0.6164]; history model 0.6750 [0.6574, 0.6938]; history+survey 0.6721 [0.6531, 0.6906].
- A, earlier talp73 LLM arms on the same cells (one call per person for all 20 categories, so the prompt format differs): L0-haiku55 0.5783 [0.5575, 0.5961]; L1-haiku55 (survey persona) 0.5827 [0.5641, 0.6012]; L2-haiku55 (persona + purchase summary) 0.6529 [0.6335, 0.6716].
- B, LLM-free (bench twin, trained on dev persons only, scored on these cells after the lock): median, ridge (alpha 300 on wave 1-3 features); reference row: the person's own wave 1-3 answer (test-retest, uses information no injector sees).

## Expected result
- A L1-type arms (plain_fields, argyle_backstory, fitted_topk, llm_narrative; survey only): 1b(c) 0.56-0.61; injector - swap between 0 and +0.03, at most one significant after Holm; none above the survey model.
- A L2-type arms (own_answers_all = survey + per-category purchase counts; memory_stream over the same bank; log_stats; log_retrieved; history-bearing): 1b(c) 0.62-0.68; injector - swap >= +0.05 and significant for all four; none significantly above the history model (0.675); best within +-0.03 of L2-haiku55.
- A no_persona: 0.55-0.59 (near L0 and popularity).
- B: plain_fields and argyle_backstory - swap within +-0.01 (demographics say little about H&B answers); own_answers_rag and memory_stream - swap between 0 and +0.02; neighbour_votes close to the median baseline; no arm above ridge; every arm below the test-retest reference.

## Falsification
- Claim: on these data persona injection adds person-specific signal only through behavioural history (A), and no injector beats the strongest LLM-free baseline.
- Fails if any A L1-type arm beats the survey model, or any arm beats the history model (A) or ridge (B), with the paired 95% CI above 0.
- Fails ('history carries the signal' half) if any A L2-type arm's difference to its own swap has a CI that includes 0 or is negative.
- Fails ('survey persona carries ~nothing' half) if two or more A L1-type arms beat their swap after Holm.

## Stop rule
- Primary comparisons, fixed: P1 = injector - swap_injector on the primary metric, per injector, Holm across the injectors of the part (A 8, B 5) at family-wise 0.05 (two-sided bootstrap p of evidence.paired_diff). P2 = injector - no_persona, same correction. Everything else (vs baselines, vs talp73 L-rows, best arm) is secondary and reported with its selection caveat.
- Budget: stage cap 10 USD on claude-haiku-5-5 only (no other provider). simengine.llm.Claude with the key file mechanism (SIMENGINE_KEY_FILE), disk cache, JSONL cost ledger at <cache_dir>/ledger.jsonl and the shared TSV spend log (SIMENGINE_SPEND_LOG, agent inject-ablation), budget_usd = 9.5: the backend refuses new calls once the ledger reaches it.
- Order: A narrative generation, then all A arms, then all B arms. A part that stops early is reported as incomplete and not scored on a subset. No prompt or parameter change after the first real call; a rerun replays the disk cache.
- Settings: thinking off, effort low (backend default), no temperature (Haiku 5.5 rejects non-default), max_tokens A 24 / B 48, parallel 4, seed 0.
- Predictions are sealed (research.prereg.seal) per arm before they are joined with the truth.

## Notes
- Survey answers in A were given in 2023-Q1, after the target window (talp73 v1.6.1 V7); this favours survey-based arms and the survey model alike.
- One call per (person, category) cell here; talp73 asked all 20 categories in one call. L-row comparisons are therefore secondary.
- Planned but dropped at planning: answer_neighbours in B (~4,900 input tokens per cell).
- Cost check: compare_injectors' estimate_cost prices the summed tokens of all calls as ONE call, which crosses Haiku's 100k long-context threshold and applies the 5x long price (A 14.73 USD, B 12.68 USD at n=60 in the first dry run). The figures under Extra price each call on its own (simengine.llm.pricing.cost_usd).
- llm_narrative prompt hashes above are of the dry-run placeholder narrative (same text for everyone, so the direct and swap hashes coincide); the generated narratives are sealed before the target calls.
- Runner: ablation.py in the TALP work directory, sha256 c48fe31c00a9386d7ab8ce8a149b6b9a411c968cec2bb4c3b66424fede9e5870; simengine at the commit of this file; persona.inject at c09ccc5 or later.

## Extra
- **model**: claude-haiku-5-5
- **estimated cost (dry run, chars/4 tokens, every reply at max_tokens)**: A 2.94 + narrative generation 0.10 + B 4.00 = 7.04 USD upper bound (planned limit 8 USD). With +25% input tokens 8.37 USD, still under the 10 USD stage cap and the 9.5 USD backend guard. Replies are ~8-15 tokens, not max_tokens, so the expected spend is lower than the bound.
- **A calls**: 46036 + 359 generation calls
- **B calls**: 47960
- **A arms**: 
  - no_persona: 2708 calls, ~240,252 input tokens, <= $0.056 ($0.062 with +25% tokens); prompts sha256 873fb1b8759f
  - plain_fields: 2708 calls, ~1,150,786 input tokens, <= $0.147 ($0.176 with +25% tokens); prompts sha256 f4162922476d
  - argyle_backstory: 2708 calls, ~1,194,791 input tokens, <= $0.152 ($0.182 with +25% tokens); prompts sha256 7724c7dc4adb
  - fitted_topk(k=5): 2708 calls, ~473,797 input tokens, <= $0.080 ($0.091 with +25% tokens); prompts sha256 3ba902f45a8f
  - llm_narrative(style=system): 2708 calls, ~937,562 input tokens, <= $0.126 ($0.149 with +25% tokens); prompts sha256 53ee1440952d
  - own_answers_all(questions=62q): 2708 calls, ~4,613,866 input tokens, <= $0.494 ($0.609 with +25% tokens); prompts sha256 41297d380fb4
  - memory_stream(k=10,questions=#728b46): 2708 calls, ~1,365,884 input tokens, <= $0.169 ($0.203 with +25% tokens); prompts sha256 a2cd7b1eaed5
  - log_stats(top_k=5): 2708 calls, ~924,084 input tokens, <= $0.125 ($0.148 with +25% tokens); prompts sha256 e38efc7b9636
  - log_retrieved(k=10): 2708 calls, ~1,193,125 input tokens, <= $0.152 ($0.181 with +25% tokens); prompts sha256 c7a640d719a7
  - swap_plain_fields: 2708 calls, ~1,151,421 input tokens, <= $0.147 ($0.176 with +25% tokens); prompts sha256 2c546aa564e6
  - swap_argyle_backstory: 2708 calls, ~1,195,426 input tokens, <= $0.152 ($0.182 with +25% tokens); prompts sha256 02c42e6dbccf
  - swap_fitted_topk(k=5): 2708 calls, ~474,183 input tokens, <= $0.080 ($0.091 with +25% tokens); prompts sha256 1676ae2de08d
  - swap_llm_narrative(style=system): 2708 calls, ~937,562 input tokens, <= $0.126 ($0.149 with +25% tokens); prompts sha256 53ee1440952d
  - swap_own_answers_all(questions=62q): 2708 calls, ~4,614,275 input tokens, <= $0.494 ($0.609 with +25% tokens); prompts sha256 d1271a124920
  - swap_memory_stream(k=10,questions=#728b46): 2708 calls, ~1,365,899 input tokens, <= $0.169 ($0.203 with +25% tokens); prompts sha256 f220309194c9
  - swap_log_stats(top_k=5): 2708 calls, ~922,229 input tokens, <= $0.125 ($0.147 with +25% tokens); prompts sha256 235732f3026e
  - swap_log_retrieved(k=10): 2708 calls, ~1,183,281 input tokens, <= $0.151 ($0.180 with +25% tokens); prompts sha256 f3a1f87a9b89
- **B arms**: 
  - no_persona: 4360 calls, ~468,450 input tokens, <= $0.151 ($0.163 with +25% tokens); prompts sha256 9ced401bdeda
  - plain_fields: 4360 calls, ~1,560,284 input tokens, <= $0.261 ($0.299 with +25% tokens); prompts sha256 7ae9d5982260
  - argyle_backstory: 4360 calls, ~1,596,254 input tokens, <= $0.264 ($0.304 with +25% tokens); prompts sha256 46aff933dbfc
  - own_answers_rag(k=8,questions=546q,scorer=bm25): 4360 calls, ~4,474,056 input tokens, <= $0.552 ($0.663 with +25% tokens); prompts sha256 caf6131711e7
  - memory_stream(k=10,questions=#20d2c6): 4360 calls, ~4,570,720 input tokens, <= $0.561 ($0.676 with +25% tokens); prompts sha256 f51f9dd167dc
  - neighbour_votes(k=10,pool=n300#5f5f83): 4360 calls, ~1,792,593 input tokens, <= $0.284 ($0.328 with +25% tokens); prompts sha256 193b3cc333ba
  - swap_plain_fields: 4360 calls, ~1,560,459 input tokens, <= $0.261 ($0.299 with +25% tokens); prompts sha256 fde5e9794e55
  - swap_argyle_backstory: 4360 calls, ~1,596,429 input tokens, <= $0.264 ($0.304 with +25% tokens); prompts sha256 6b07f50630ea
  - swap_own_answers_rag(k=8,questions=546q,scorer=bm25): 4360 calls, ~4,471,490 input tokens, <= $0.552 ($0.663 with +25% tokens); prompts sha256 6c4109d9bcc4
  - swap_memory_stream(k=10,questions=#20d2c6): 4360 calls, ~4,581,100 input tokens, <= $0.563 ($0.677 with +25% tokens); prompts sha256 ea3fb642be77
  - swap_neighbour_votes(k=10,pool=n300#5f5f83): 4360 calls, ~1,792,771 input tokens, <= $0.284 ($0.328 with +25% tokens); prompts sha256 55c94ca7f5ea

## Machine-readable copy
```json
{
 "baselines": [
  "Controls (both parts): no_persona (bare question) and swap_<injector> for every injector.",
  "A, LLM-free, talp73 seed-0 prediction files scored on these exact 2708 cells (already published talp73 numbers, seen before this file): constant 0.5059 [0.4858, 0.5259]; popularity 0.5820 [0.5632, 0.6006]; survey model 0.5990 [0.5812, 0.6164]; history model 0.6750 [0.6574, 0.6938]; history+survey 0.6721 [0.6531, 0.6906].",
  "A, earlier talp73 LLM arms on the same cells (one call per person for all 20 categories, so the prompt format differs): L0-haiku55 0.5783 [0.5575, 0.5961]; L1-haiku55 (survey persona) 0.5827 [0.5641, 0.6012]; L2-haiku55 (persona + purchase summary) 0.6529 [0.6335, 0.6716].",
  "B, LLM-free (bench twin, trained on dev persons only, scored on these cells after the lock): median, ridge (alpha 300 on wave 1-3 features); reference row: the person's own wave 1-3 answer (test-retest, uses information no injector sees)."
 ],
 "data": [
  "A: Open e-commerce 1.0 (Berke et al. 2024, CC0), exactly as prepared by the TALP talp73 pipeline (talp-simulation-hackathon-2026 uretim/talp73, PROTOKOL v1.0 V1-V5, degerlendir.py 71a4e82): fully observed users, the 20 fixed categories, seed-0 sample of 400 persons. Person fields = the 13 talp73 survey items (istem.SORULAR, exact labels). History = that person's purchases before 2022-07-01 (date, title cut to 90 characters, category, unit price, quantity). Target = at least one purchase in the category 2022-07-01 .. 2022-12-31.",
  "A cells: the talp73 1b balanced mask (per person min(5, bought, not bought) of each, default_rng(1000)), minus the 3 persons with no purchase before the cutoff (log injectors need a history): 359 persons, 2708 cells; cells sha256 04d044090eb0f5a497dfe812d70ee8d2f03c45e56025d2419770b2f2c79543ec.",
  "B: Twin-2K-500 (Toubia et al. 2025, CC BY 4.0). simengine.bench Twin2K500 split='test' (25% of persons by the fixed id hash); 80 test persons drawn with numpy default_rng(0) (sorted ids). Target items = the scored heuristics-and-biases columns that were asked again in wave 4, pricing block (*_Q295) excluded: 82 items. Truth = the person's wave-4 answer (raw code; anchoring items in scored deciles at scoring time). Only cells with a wave-4 answer: 4360 cells; cells sha256 48961005ebe6588dd72967cb350c8f160d1107c46c15d77dd6f7c7cefe5b817c.",
  "B persona content: wave 1-3 answers of the Demographics, Personality and Economic preferences blocks (546 bank items; Demographics as person fields, the rest as answers). The person's own wave 1-3 answers to the target items are NOT shown to any injector.",
  "Licensed data and every prediction stay in the TALP work directory outside this repo; this file holds no data rows, person ids or answers."
 ],
 "expected": [
  "A L1-type arms (plain_fields, argyle_backstory, fitted_topk, llm_narrative; survey only): 1b(c) 0.56-0.61; injector - swap between 0 and +0.03, at most one significant after Holm; none above the survey model.",
  "A L2-type arms (own_answers_all = survey + per-category purchase counts; memory_stream over the same bank; log_stats; log_retrieved; history-bearing): 1b(c) 0.62-0.68; injector - swap >= +0.05 and significant for all four; none significantly above the history model (0.675); best within +-0.03 of L2-haiku55.",
  "A no_persona: 0.55-0.59 (near L0 and popularity).",
  "B: plain_fields and argyle_backstory - swap within +-0.01 (demographics say little about H&B answers); own_answers_rag and memory_stream - swap between 0 and +0.02; neighbour_votes close to the median baseline; no arm above ridge; every arm below the test-retest reference."
 ],
 "extra": {
  "A arms": "\n  - no_persona: 2708 calls, ~240,252 input tokens, <= $0.056 ($0.062 with +25% tokens); prompts sha256 873fb1b8759f\n  - plain_fields: 2708 calls, ~1,150,786 input tokens, <= $0.147 ($0.176 with +25% tokens); prompts sha256 f4162922476d\n  - argyle_backstory: 2708 calls, ~1,194,791 input tokens, <= $0.152 ($0.182 with +25% tokens); prompts sha256 7724c7dc4adb\n  - fitted_topk(k=5): 2708 calls, ~473,797 input tokens, <= $0.080 ($0.091 with +25% tokens); prompts sha256 3ba902f45a8f\n  - llm_narrative(style=system): 2708 calls, ~937,562 input tokens, <= $0.126 ($0.149 with +25% tokens); prompts sha256 53ee1440952d\n  - own_answers_all(questions=62q): 2708 calls, ~4,613,866 input tokens, <= $0.494 ($0.609 with +25% tokens); prompts sha256 41297d380fb4\n  - memory_stream(k=10,questions=#728b46): 2708 calls, ~1,365,884 input tokens, <= $0.169 ($0.203 with +25% tokens); prompts sha256 a2cd7b1eaed5\n  - log_stats(top_k=5): 2708 calls, ~924,084 input tokens, <= $0.125 ($0.148 with +25% tokens); prompts sha256 e38efc7b9636\n  - log_retrieved(k=10): 2708 calls, ~1,193,125 input tokens, <= $0.152 ($0.181 with +25% tokens); prompts sha256 c7a640d719a7\n  - swap_plain_fields: 2708 calls, ~1,151,421 input tokens, <= $0.147 ($0.176 with +25% tokens); prompts sha256 2c546aa564e6\n  - swap_argyle_backstory: 2708 calls, ~1,195,426 input tokens, <= $0.152 ($0.182 with +25% tokens); prompts sha256 02c42e6dbccf\n  - swap_fitted_topk(k=5): 2708 calls, ~474,183 input tokens, <= $0.080 ($0.091 with +25% tokens); prompts sha256 1676ae2de08d\n  - swap_llm_narrative(style=system): 2708 calls, ~937,562 input tokens, <= $0.126 ($0.149 with +25% tokens); prompts sha256 53ee1440952d\n  - swap_own_answers_all(questions=62q): 2708 calls, ~4,614,275 input tokens, <= $0.494 ($0.609 with +25% tokens); prompts sha256 d1271a124920\n  - swap_memory_stream(k=10,questions=#728b46): 2708 calls, ~1,365,899 input tokens, <= $0.169 ($0.203 with +25% tokens); prompts sha256 f220309194c9\n  - swap_log_stats(top_k=5): 2708 calls, ~922,229 input tokens, <= $0.125 ($0.147 with +25% tokens); prompts sha256 235732f3026e\n  - swap_log_retrieved(k=10): 2708 calls, ~1,183,281 input tokens, <= $0.151 ($0.180 with +25% tokens); prompts sha256 f3a1f87a9b89",
  "A calls": "46036 + 359 generation calls",
  "B arms": "\n  - no_persona: 4360 calls, ~468,450 input tokens, <= $0.151 ($0.163 with +25% tokens); prompts sha256 9ced401bdeda\n  - plain_fields: 4360 calls, ~1,560,284 input tokens, <= $0.261 ($0.299 with +25% tokens); prompts sha256 7ae9d5982260\n  - argyle_backstory: 4360 calls, ~1,596,254 input tokens, <= $0.264 ($0.304 with +25% tokens); prompts sha256 46aff933dbfc\n  - own_answers_rag(k=8,questions=546q,scorer=bm25): 4360 calls, ~4,474,056 input tokens, <= $0.552 ($0.663 with +25% tokens); prompts sha256 caf6131711e7\n  - memory_stream(k=10,questions=#20d2c6): 4360 calls, ~4,570,720 input tokens, <= $0.561 ($0.676 with +25% tokens); prompts sha256 f51f9dd167dc\n  - neighbour_votes(k=10,pool=n300#5f5f83): 4360 calls, ~1,792,593 input tokens, <= $0.284 ($0.328 with +25% tokens); prompts sha256 193b3cc333ba\n  - swap_plain_fields: 4360 calls, ~1,560,459 input tokens, <= $0.261 ($0.299 with +25% tokens); prompts sha256 fde5e9794e55\n  - swap_argyle_backstory: 4360 calls, ~1,596,429 input tokens, <= $0.264 ($0.304 with +25% tokens); prompts sha256 6b07f50630ea\n  - swap_own_answers_rag(k=8,questions=546q,scorer=bm25): 4360 calls, ~4,471,490 input tokens, <= $0.552 ($0.663 with +25% tokens); prompts sha256 6c4109d9bcc4\n  - swap_memory_stream(k=10,questions=#20d2c6): 4360 calls, ~4,581,100 input tokens, <= $0.563 ($0.677 with +25% tokens); prompts sha256 ea3fb642be77\n  - swap_neighbour_votes(k=10,pool=n300#5f5f83): 4360 calls, ~1,792,771 input tokens, <= $0.284 ($0.328 with +25% tokens); prompts sha256 55c94ca7f5ea",
  "B calls": "47960",
  "estimated cost (dry run, chars/4 tokens, every reply at max_tokens)": "A 2.94 + narrative generation 0.10 + B 4.00 = 7.04 USD upper bound (planned limit 8 USD). With +25% input tokens 8.37 USD, still under the 10 USD stage cap and the 9.5 USD backend guard. Replies are ~8-15 tokens, not max_tokens, so the expected spend is lower than the bound.",
  "model": "claude-haiku-5-5"
 },
 "falsification": [
  "Claim: on these data persona injection adds person-specific signal only through behavioural history (A), and no injector beats the strongest LLM-free baseline.",
  "Fails if any A L1-type arm beats the survey model, or any arm beats the history model (A) or ridge (B), with the paired 95% CI above 0.",
  "Fails ('history carries the signal' half) if any A L2-type arm's difference to its own swap has a CI that includes 0 or is negative.",
  "Fails ('survey persona carries ~nothing' half) if two or more A L1-type arms beat their swap after Holm."
 ],
 "metrics": {
  "A primary": "talp73 1b(c) accuracy: within each person, the top half of their 1b cells by p is 'yes' (ties broken by talp73's G matrix, default_rng(1000)); share of correct cells. Higher is better, chance 0.5. p = parsed answer / 100.",
  "A secondary": "mean within-person AUC (talp73 interpretation 4) on the same cells; 1b(a) accuracy (p >= 0.5).",
  "B primary": "official Twin-2K-500 MAD task accuracy against the wave-4 answer: per cell 1 - |pred - truth| / range in scored units (anchoring items to wave 1-3 deciles, bench twin to_scored), mean over a person's tasks, then over persons. Higher is better. Option text -> its 1-based position, the code the wave-4 file uses (checked on the dev pool before any test reply is parsed).",
  "B secondary": "between-person correlation per column (mean Fisher z, bench 'mean_z'); exact-match accuracy (compare_injectors' own table).",
  "Uncertainty": "person-level bootstrap, B = 1000, numpy default_rng(42) (talp73 degerlendir.agirliklar weights for A, the same resamples for every arm); differences are paired on the same resamples; 95% percentile CI.",
  "Unparsed / failed replies": "A: p = 0.5; B: the dev-person median of that item. Parse rate is reported per arm; an arm under 95% parsed is flagged 'format failure' and its verdict is not interpreted."
 },
 "notes": [
  "Survey answers in A were given in 2023-Q1, after the target window (talp73 v1.6.1 V7); this favours survey-based arms and the survey model alike.",
  "One call per (person, category) cell here; talp73 asked all 20 categories in one call. L-row comparisons are therefore secondary.",
  "Planned but dropped at planning: answer_neighbours in B (~4,900 input tokens per cell).",
  "Cost check: compare_injectors' estimate_cost prices the summed tokens of all calls as ONE call, which crosses Haiku's 100k long-context threshold and applies the 5x long price (A 14.73 USD, B 12.68 USD at n=60 in the first dry run). The figures under Extra price each call on its own (simengine.llm.pricing.cost_usd).",
  "llm_narrative prompt hashes above are of the dry-run placeholder narrative (same text for everyone, so the direct and swap hashes coincide); the generated narratives are sealed before the target calls.",
  "Runner: ablation.py in the TALP work directory, sha256 c48fe31c00a9386d7ab8ce8a149b6b9a411c968cec2bb4c3b66424fede9e5870; simengine at the commit of this file; persona.inject at c09ccc5 or later."
 ],
 "question": "Which persona-injection methods of simengine.persona.inject carry person-specific predictive signal (beat the same injector fed another person's persona, and beat a no-persona prompt) when a real LLM predicts (A) next-6-month category purchases in the talp73 setup and (B) Twin-2K-500 wave-4 answers; and does any of them beat the strongest LLM-free baseline on the same cells?",
 "split": [
  "A: same persons, categories, window and 1b mask as talp73 seed 0; nothing is trained on these persons.",
  "A fitted_topk is fitted LLM-free (random forest) on the talp73 fully observed users NOT in the seed-0 sample, labels = category purchase in 2022-01 .. 2022-06 (the window talp73's classical baselines train on), fit question ids differ from the target ids. Chosen keys (fixed now): ['State of residence', 'How often the account orders Amazon deliveries', 'Household income before taxes, past 12 months', 'Number of people who log in and order with this Amazon account', 'Age group'].",
  "A llm_narrative: one generation call per person from the person fields only (150 words, max_tokens 400), before any target call; the narrative never sees history or targets.",
  "B neighbour pool: 300 Twin dev persons (default_rng(1)) with their own wave-4 answers to the target items; no test person is in the pool, and the swap fix (c09ccc5) keeps the scored person out.",
  "Swap control: compare_injectors' swap_<injector> (seeded derangement, seed 0): same prompt with another person's persona, scored against the original person."
 ],
 "stop_rule": [
  "Primary comparisons, fixed: P1 = injector - swap_injector on the primary metric, per injector, Holm across the injectors of the part (A 8, B 5) at family-wise 0.05 (two-sided bootstrap p of evidence.paired_diff). P2 = injector - no_persona, same correction. Everything else (vs baselines, vs talp73 L-rows, best arm) is secondary and reported with its selection caveat.",
  "Budget: stage cap 10 USD on claude-haiku-5-5 only (no other provider). simengine.llm.Claude with the key file mechanism (SIMENGINE_KEY_FILE), disk cache, JSONL cost ledger at <cache_dir>/ledger.jsonl and the shared TSV spend log (SIMENGINE_SPEND_LOG, agent inject-ablation), budget_usd = 9.5: the backend refuses new calls once the ledger reaches it.",
  "Order: A narrative generation, then all A arms, then all B arms. A part that stops early is reported as incomplete and not scored on a subset. No prompt or parameter change after the first real call; a rerun replays the disk cache.",
  "Settings: thinking off, effort low (backend default), no temperature (Haiku 5.5 rejects non-default), max_tokens A 24 / B 48, parallel 4, seed 0.",
  "Predictions are sealed (research.prereg.seal) per arm before they are joined with the truth."
 ],
 "title": "Persona-injection ablation on real purchase and survey data (claude-haiku-5-5), 2026-10-22"
}
```
