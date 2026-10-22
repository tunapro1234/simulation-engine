# `simengine.research.prompt_evolution`: LLM-driven prompt and persona-text optimisation

## What

The module searches for a text (an instruction, a system prompt or a persona description) that scores well under a
fixed scorer, and it follows the same discipline as the research loop. The search sees the training split only. The
top-k texts are re-ranked on a validation split. The winner is sealed, and a locked prereg is checked. Then the test
split is scored once, for the winner and for the baselines. It ships three search methods:

| method | source | what is implemented |
|---|---|---|
| `opro` | Yang et al. 2023, *Large Language Models as Optimizers*, arXiv:2309.03409 (§4, Fig. 3) | The meta-prompt holds the best `keep_top` (20) texts so far with their training scores, sorted ascending so the best comes last. Scores are rounded to integers 0-100 (`score_scale`). It also holds `n_exemplars` (3) training exemplars, sampled again at every step, with `<INS>` at the insertion point (`position="q_begin"`, or `"q_end"`). The meta-instruction is quoted from Fig. 3: "Write your new text that is different from the old ones and has a score as high as possible. Write the text in square brackets." Each step draws `per_step` (8) samples, and the first `[...]` span is parsed. `show_scores=False` gives the paper's "no scores" ablation. |
| `ape` | Zhou et al. 2022, *Large Language Models Are Human-Level Prompt Engineers*, arXiv:2211.01910 (Alg. 1, Fig. 2-3) | Forward-mode generation template ("I gave a friend an instruction and five inputs. ... The instruction was"). Each candidate is generated from its own random `n_demos` (5) training demonstrations, with `n_candidates` (50, the paper's default) in total. Candidates are scored on a random training subset (`subset_size`), the top `top_frac` are kept, and the survivors are scored on the full training split. `resample_rounds > 0` turns on iterative APE ("Generate a variation of the following instruction while keeping the semantic meaning.") with a new random subset each round. |
| `promptbreeder` | Fernando et al. 2023, *Promptbreeder*, arXiv:2309.16797 (§3) | A binary-tournament GA over units of (task-prompt, mutation-prompt). Initialisation sends "<mutation-prompt> <thinking-style> INSTRUCTION: <problem> INSTRUCTION MUTANT:". Each replication draws one operator uniformly from nine operators in five classes: zero-order generation ("A list of 100 hints:"), first-order generation, EDA (diversity-filtered list in random order), EDA rank and index (ascending list, deliberately headed "descending order of score"), lineage ("GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY"), zero-order hyper-mutation (problem + thinking-style gives a new mutation-prompt), first-order hyper-mutation ("Please summarize and improve the following instruction:"), and Lamarckian working-out-to-task-prompt ("I gave a friend an instruction and some advice. ..."). After the operator, a 10% prompt crossover (fitness-proportionate donor) may replace the task-prompt. The mutant overwrites the loser. |

The template strings were checked against the papers' text (arXiv PDFs, read 22 Oct 2026). `MUTATION_PROMPTS` and
`THINKING_STYLES` are short subsets quoted from Promptbreeder Appendix C/D; pass your own lists for real runs.

## API

```python
import simengine.research.prompt_evolution as pe

tr, va, te = pe.split_units(df, frac=(0.5, 0.25, 0.25), seed=0, group=None)   # DataFrame or list
score = pe.make_llm_scorer(scorer_llm, input_key="input", target_key="output", position="system",
                           parse=None, metric=pe.exact_match, max_tokens=64)  # -> {"scores", "outputs"}
ev = pe.PromptEvaluator(score, tr, va, te, higher_is_better=True)

res = pe.evolve("opro", optimizer_llm, ev, seeds=["Answer the question."], steps=10, per_step=8,
                top_k=5, registry_dir="runs/pe/.prereg", prereg="runs/pe/prereg.md")
res.best.text; res.validation; res.test["winner"]; res.test_diff["winner-seed0"]; res.gaps; res.search
```

- **Scorer contract.** `score_fn(text, units)` returns per-unit scores (an array), or `{"scores": array, "outputs":
  [str]}`. The outputs feed the Lamarckian operator. NaN means a failed unit, and it is left out of the means.
  Scores are higher-is-better unless `higher_is_better=False`, in which case every ranking uses the negated mean
  while reports keep the raw value. `make_llm_scorer` runs the text as the system prompt (or `q_begin` / `q_end` in
  the user message, as in OPRO §4.1). Its default metric is exact match after trimming, case-folding and stripping
  trailing punctuation.
- **Persona text.** Use a scorer that puts the persona text in the system prompt and grades each survey item against
  the human answer, e.g. `make_llm_scorer(llm, parse=float, metric=lambda y, t: -abs(y - float(t)))`.
- **`PromptEvaluator`** caches per-unit scores per (text, split, subset), counts unit evaluations per split, and
  raises `PermissionError` on any read of `"test"` outside `select_and_test`.
- **Search functions** `opro(llm, ev, seeds, ...)`, `ape(llm, ev, ...)` and `promptbreeder(llm, ev, problem, ...)`
  return `{"history": [Candidate], "llm_calls", "cost_usd", ...}`. Each optimizer call gets a distinct
  `sample=` index, so repeated identical meta-prompts are independent draws and not cache hits in
  `simengine.llm.Claude` / `ClaudeCode`. No `temperature` is sent; the Claude 5.5 backends reject non-default
  values. OPRO used optimizer temperature 1.0.
- **`select_and_test(ev, history, top_k=5, baselines=..., registry_dir=..., prereg=...)`** works in five steps:
  1. Takes the top-k distinct texts by training fitness. A text scored only on a subset is rescored on the full
     training split first.
  2. Scores them on validation. The best validation score wins; ties go to the higher training fitness, then to
     the shorter text.
  3. Seals `{method, winner, train, val, top_k, baselines}` with `simengine.research.seal`.
  4. Runs `verify_prereg` and refuses on a failure.
  5. Scores the winner and every baseline once on test, with bootstrap CIs for the means and paired bootstrap
     winner-minus-baseline CIs (95% and Bonferroni over the baselines).

  `gaps` reports train minus val and val minus test for the winner. A second call on the same evaluator raises.
- **`evolve(method, llm, ev, seeds=..., ...)`** runs search and then `select_and_test`. The baselines default to
  the seeds (`seed0`, `seed1`, ...).
- **`toy_problem(...)`** is an offline synthetic task: (units, score_fn, keywords, FakeLLM responder). It is used
  by the tests and the measurement below.

## Measured numbers

See the next commit.

## Limitations

- Promptbreeder is simplified. It evolves one task-prompt per unit (the paper evolves two in sequence), has no
  few-shot context shuffling, and uses character 2-4-gram TF-IDF cosine instead of BERT embeddings for the 0.95
  diversity filter. The default population (20) and generation count (10) are below the paper's 50 and 20-30.
  `batch_size=None` scores fitness on the full training split; the paper uses a fresh batch of 100.
- In `promptbreeder`, a Lamarckian draw without scorer outputs falls back to first-order generation
  (`lamarckian_fallbacks`). By default a "correct" working is any unit at the maximum observed score
  (`correct_at=`).
- APE has no reverse (infilling) mode and no log-probability score, because the Claude backends expose neither. The
  adaptive filter is a single stage per round; the paper uses repeated stages on non-overlapping subsets.
- OPRO keeps a single best score per text. A paraphrase is a new text, so near-duplicates can fill the top-20 list.
- Searching more candidates means more looks at the training split, and top-k selection also looks at validation
  `top_k` times. Read `gaps["train_minus_val"]` before you trust a training score. The test split is the only
  untouched estimate.
- No real-LLM measurement is reported. Before you claim a gain, run it with `simengine.llm.Claude(budget_usd=...)`,
  a locked prereg and the seed prompt as the baseline.

Tests: `python3 -m unittest tests.test_research_prompt_evolution` (23 tests, synthetic units, FakeLLM).
