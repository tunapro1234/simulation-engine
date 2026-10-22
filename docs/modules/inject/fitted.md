# `persona.inject` family: `fitted`

The persona prompt is fitted to people's real answers on a set of FIT questions and then used on other questions.
All four injectors render the same `key: value` attribute lines as `plain_fields`. What gets fitted is which
attributes are shown, in what order, or the framing sentence above them. A fit is population-level: one config
for everybody, and no injector here shows a person's own answers. Module: `simengine/persona/inject/fitted.py`.

```python
import simengine.persona.inject as pi
inj = pi.get_injector("fitted_stepwise", seed=0)
fit = inj.fit(llm, people, fit_questions, "answers", max_calls=2000)    # LLM calls on fit questions only
fit.config, fit.train_score, fit.val_score, fit.calls
res = pi.compare_injectors(llm, people, test_questions, "answers", [inj, "plain_fields"])
```

How the held-out discipline works:

- **Question-level split.** `fit()` splits the fit questions into train and validation questions (`val_frac`=0.25,
  seeded, or `val_questions=[...]`). Search reads train cells only. The best `top_k` configs are then re-ranked on
  the validation questions with `simengine.research.prompt_evolution.select_and_test`.
- **Held-out guard.** After a fit, `render` raises `HeldOutViolation` for every fit question, train or
  validation. `compare_injectors` therefore can only score a fitted injector on questions it never saw. Before a
  fit (or `load()`), `render` raises `NotFitted`.
- **Seal and prereg.** `registry_dir=` seals the winning config (`research.prereg.seal`) before it touches a
  held-out question, and `prereg=` verifies a locked pre-registration.
- **Budget.** `max_calls=` refuses to start when the upper-bound LLM call count is above it. The count is
  `FitResult.meta["estimate"]`.
- **Missing fields.** Every injector needs `fields`. Without them, `fit` and `render` raise `MissingPersonField`.

Scores are per (person, train question) cell: 1 if the parsed reply equals the truth, else 0. Unparsed replies and
failed calls count as wrong, as in `compare_injectors`. `fit.to_dict()` / `inj.load(d)` restore a fit, and the
fit questions stay refused after a load.

| injector | what is fitted | LLM calls (upper bound) | params |
|---|---|---|---|
| `fitted_topk` | the top k attributes by random-forest importance for the fit questions' answers | none | `k`=5, `per_model`=10, `n_estimators`=200, `seed` |
| `fitted_stepwise` | attribute inclusion, by coordinate ascent in `research.Loop` | (turns + 1) x train cells + top_k x val cells | `start` empty\|all, `max_turns` (3 x attributes), `alpha`, `simpler_ties`, `n_boot` |
| `fitted_grips` | attribute lines, by GrIPS edits (delete / swap / add back) | (1 + n_iter x m) x train cells + top_k x val cells | `m`=5, `l`=1, `n_iter`=10, `patience`=2, `start` all\|empty |
| `fitted_opro` | the framing sentence above the lines, by OPRO | (seeds + steps x per_step) x train cells + top_k x val cells, plus steps x per_step optimizer calls | `steps`=10, `per_step`=8, `keep_top`=20, `n_exemplars`=3, `seeds` |

The shared params are `keys` (the candidate attributes; default: every attribute any person has), `val_frac`,
`top_k`=5 (the validation re-rank) and `seed`. `fitted_topk` does not use a validation split, because it makes no
LLM call to validate.

## Literature

**`fitted_topk`: Fröhling et al. 2026 (arXiv:2609.02526) and Rupprecht et al. 2025 (arXiv:2511.21722).**
- *Fröhling et al.:* four general social surveys, two countries, six LLMs and 20 target questions per survey. They
  compare six ways to pick k=5 persona attributes per target question. Two are statistical: random-forest feature
  importance for the target, and correlation. The others use LLM set selection, LLM scoring, embedding similarity
  and human response variation.
- *Fröhling et al., published effect:* "statistical baselines outperform LLM-based approaches for attribute
  selection" (Jensen-Shannon distance, figure 3). The text gives no table of numbers. They also report that more
  attributes do not reliably help.
- *Rupprecht et al.:* German GSS (ALLBUS) personas. They fit one forest per candidate variable, predicting it from
  all other variables, and sum each forest's top-10 importances into a single ranking (TOP-k). Alignment does not
  improve monotonically with k, and TOP-2 aligned best in their runs.
- *Adaptation here:* target questions are held out, so one forest is fit per fit question. Each forest's top
  `per_model` importances are summed (Rupprecht's aggregation), and one top-k set serves every held-out question.
  The default k=5 follows Fröhling et al.

**`fitted_stepwise`: wrapper / stepwise attribute selection (Efroymson 1960; Kohavi & John 1997; no arXiv id).**
- *Search:* starts from no attributes (or all of them). Each turn flips one attribute in or out
  (`research.proposers.GreedyCoordinateProposer`).
- *Acceptance:* `research.Loop` keeps a flip only when the paired bootstrap CI over train cells excludes 0, or when
  it ties with fewer attributes (`simpler_ties`).
- *Selection:* the kept path is re-ranked on the validation questions.
- *Effect:* no published persona-prompt effect; the attribute-selection question it answers is the one studied by
  Fröhling et al.

**`fitted_grips`: Prasad et al. 2022, GrIPS (arXiv:2203.07281).**
- *Paper:* gradient-free, edit-based instruction search, run on eight Natural Instructions classification tasks.
  At each iteration it samples m candidates of l phrase-level edits (delete, swap, paraphrase, add a deleted phrase
  back). The best candidate replaces the base only if it scores higher, and the search stops after P iterations
  without improvement.
- *Published effect:* up to +4.30 points average accuracy with InstructGPT, by the paper's abstract.
- *Here:* the phrases are attribute lines. Paraphrase is left out, since `key: value` lines have nothing to
  paraphrase. The score is cell accuracy, not the paper's balanced accuracy + entropy term.

**`fitted_opro`: Yang et al. 2023, OPRO (arXiv:2309.03409).**
- *Method:* `research.prompt_evolution.opro`. The meta-prompt holds the best texts so far with integer train
  scores in ascending order, plus `n_exemplars` train cells (attribute lines + question -> real answer). Each step
  draws `per_step` samples.
- *Here:* only the framing sentence is optimised; the attribute lines and the closing line stay fixed. The proposer
  LLM sees train scores and train exemplars, never validation or held-out answers. `optimizer=` sets a separate
  proposer LLM.
- *Published effect:* prompts beat human-written ones by "up to 8% on GSM8K, and by up to 50% on Big-Bench
  Hard tasks" (abstract). Those are reasoning benchmarks, not persona simulation.

**Not yet measured in simengine.** None of the four has been run against real panel data here; the tests use
FakeLLM only. A fitted arm is only meaningful against `plain_fields` (all attributes) and `no_persona` on the same
held-out questions. The swap control applies as usual.
