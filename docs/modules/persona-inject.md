# `simengine.persona.inject`

Persona injection methods from the literature: 52 injectors in 12 families plus 2 controls (54 registered),
all compared on the same cells. An injector turns one `(person, question)` into a `Prompt(system, user, history, meta)` built from that
person's own data: attributes, other answers, an interview, a behaviour log, their neighbours, their group.
`compare_injectors` runs several injectors with a `no_persona` and a `swap` control and reports paired
cluster-bootstrap CIs (`simengine.evidence.paired_diff`, cluster = person).

**Nothing in this package has been measured with a real LLM in simengine yet.** Every effect size below is the
cited paper's own number, on its own model, data and metric. The tests run on `FakeLLM` only. They check
rendering, leakage, determinism and the compare plumbing.

```python
import simengine.persona as sp                 # sp.inject is the package; the main names are re-exported
sp.list_injectors()                             # name, family, paper, needs, description
p = sp.get_injector("own_answers_rag", questions=bank, k=3)(person, question)
llm.complete(p.system, p.flatten())
plan = sp.compare_injectors(llm, people, bank, "answers", ["plain_fields", ("own_answers_rag", {"questions": bank})],
                            dry_run=True)
plan.estimate                                   # calls, input tokens (chars/4), upper-bound USD; no call made
res = sp.compare_injectors(llm, people, bank, "answers", ["plain_fields"], max_usd=2.0)
res.table                                       # arm, score [lo, hi], diff/CI vs no_persona and vs swap_<arm>
```

Runnable offline: `python examples/persona_inject_compare.py` (five injectors, synthetic panel, FakeLLM, ~2 s).
Family pages: [interview](inject/interview.md), [own_answers](inject/own_answers.md),
[neighbours](inject/neighbours.md), [framing](inject/framing.md), [generated](inject/generated.md),
[fitted](inject/fitted.md), [population](inject/population.md), [values](inject/values.md),
[from_logs](inject/from_logs.md), [memory](inject/memory.md), [social](inject/social.md),
[finetune](inject/finetune.md).

## API

| name | what it is |
|---|---|
| `list_injectors(family=None)` | one row per registered injector: `name`, `family`, `paper`, `needs`, `description` |
| `get_injector(name, **params)` | an instance; the params are the method's knobs and form the arm label (`own_answers_rag(k=3,questions=8q)`) |
| `compare_injectors(llm, people, questions, truth, injectors, ...)` | renders every cell with each injector plus the controls, calls the LLM, parses, scores, paired CIs; returns `CompareResult` |
| `CompareResult` | `table` (one row per arm), `frame` (EvalFrame: `group`, `item`, `truth`, `pred_<arm>`), `estimate`, `prompts`, `replies`, `pairs`, `errors` |
| `Prompt` | `system`, `user`, `history` [(role, text)], `meta`; `flatten()` folds the history into the user text, `messages()` gives an Anthropic message list |
| `Injector`, `register` | base class and class decorator for new injectors (see "Adding a family") |
| `MissingPersonField` | raised when a person lacks a field the injector `needs`; the message names the field and the builder that makes it |
| `swap_wrapper`, `derangement` | the swap control by hand: each person gets one fixed donor, never themself |
| `parse_option`, `answer_spec` | strict reply parser (`{"answer": ...}`, an exact option or a bare number), the same as `simengine.games` |
| `IMPORT_ERRORS`, `FAMILIES` | family modules that failed to import (should be `{}`), and those that loaded |

`compare_injectors` keywords:
- `injectors`: names, instances, or `(name, params)` tuples.
- `truth`: `"answers"` (`person["answers"][qid]`), a `{(person_id, qid): answer}` map, or a callable. Cells
  without truth are not called.
- `controls=("no_persona", "swap")`: a `swap_<arm>` is added for every injector.
- `metric="accuracy"` (any `evidence` multiclass metric), `parse=parse_option`, `max_tokens=256`.
- `dry_run=True` renders and estimates only. `max_usd` refuses to start above the upper-bound estimate.
- `estimate_cost` prices every prompt as its own call (`simengine.llm.pricing.cost_usd`, reply at `max_tokens`) and sums: `usd_max` total, `usd_per_call_max` dearest call, both None for an unpriced model. The long-context price applies only to a single prompt over the threshold, not to the summed tokens.
- `llm_kw` passes extra keywords (`tag=`, `thinking=`, temperature for sampling arms) to every call.

**Reading the table.** `diff_vs_no_persona` asks "does this persona content help at all?". `diff_vs_swap` asks "does it
help because it is *this person's* data?": the swap arm renders the same injector with another person's data
and scores it against the original person. An arm that beats `no_persona` but not its swap has learnt the
population, not the person. `res.frame` is an EvalFrame, so baseline columns can be added and the whole frame
passed to `se.evaluate` (e.g. `frame["pred_persistence"] = ...`, `se.evaluate(frame, ours=arm, task="multiclass")`).
README's measured lesson applies here too: pure persona panels rarely beat simple baselines, so always score the
injectors against `simengine.methods` baselines as well, not only against each other.

## Which injection when

Pick the row for the data you have. "Expected effect" is the strongest published comparison for that family. The
metrics differ between rows, so compare within a row, not across rows. Costs are per (person, question) cell
unless stated. Use `dry_run=True` to get the real numbers before spending anything.

| data you have | recommended injectors | expected effect (literature) | cost | risk |
|---|---|---|---|---|
| **Demographics only** (`fields`) | `plain_fields`; framing presets `argyle_backstory`, `hu_collier_profile`, `lutz_interview`; `fitted_topk` / `fitted_stepwise` to choose which attributes to show; `cultural_prompting` / `hofstede_country` when country is the main attribute | Small. Persona variables explain under 10% of annotation variance; a 70B model with persona prompting captures 81% of what a regression on the same variables captures (Hu & Collier 2024, arXiv:2402.10811). OpinionQA: no persona 0.43, demographics + ideology 0.47 (Hwang et al. 2023, arXiv:2305.14929). Park et al. demographics-only agents: 0.71 (v1) / 0.74 (v3) normalized GSS accuracy against 0.85 / 0.83 for interview agents (arXiv:2411.10109). Argyle et al.: vote tetrachoric 0.90-0.94 from a 10-variable backstory (arXiv:2209.06899) | one short call; `fitted_topk` fits a forest and makes no LLM call; `fitted_stepwise` / `grips` / `opro` add (turns + 1) x train cells of fit calls (`max_calls=` caps them) | Stereotyped answers: Lutz et al.'s interview format lowered stereotyping in all 5 models, the third-person format did not (arXiv:2507.16076). LLM-written narratives drift: more generated content moved every state to one party (Li et al. 2025, arXiv:2503.16527), so prefer fields over `llm_narrative`. A field that copies the target is dropped only when its key equals the question id |
| **Survey answers** (`answers` to other items) | `own_answers_all` (bank of a few hundred items), `own_answers_rag` (k = 3-8 closest items), `own_answers_fewshot`; `memory_stream`; `answer_neighbours` / `similar_respondents` with a pool of other respondents; `interview_plus_survey` if you also have interviews | The strongest prompt-only signal in this package. Twin-2K-500: 71.72% accuracy with all 412 other items, against 81.72% human test-retest and 59.17% random; a persona summary alone scored 68.02% (Toubia et al. 2025, arXiv:2505.17479). OpinionQA: demographics + ideology + all opinions 0.51, top-8 retrieved opinions alone 0.52, demographics + ideology + top-8 0.54, against 0.43 with no persona (Hwang et al.). Park et al. v3 survey-only agents 0.82 normalized. Holtdirk et al. 2026 (arXiv:2606.09351): neighbours as in-context examples about halve the error of a downstream coefficient against MICE (0.033 vs 0.068, MCAR) | `own_answers_all` input grows with the bank; `rag` and `fewshot` stay at k items; neighbour arms add k profiles per call | Leakage is the main risk. A retest copy of the target under another id is dropped only when its text is identical (`drop_same_text`); a reworded one is not. Held-out item blocks need `exclude` (the Twin replication must also hide the other 87 held-out items). Public panels may be memorised. Neighbour arms show other people's answers to the target on purpose: evaluate with a pool that excludes the scored people or rely on `exclude_people` |
| **Interviews** (transcript, optional notes) | `interview_transcript`, `interview_expert_reflection` (the paper's full agent), `interview_summary`, `interview_plus_survey`; `interview_lesion` as a dose check | Park et al. 2024 (arXiv:2411.10109), GSS accuracy normalized by participants' own two-week retest: interview + reflection 0.85 (v1) / 0.83 (v3); survey + interview 0.86 (v3); summary agents 0.83 / 0.81; 80% of the transcript removed 0.79; demographics 0.71 / 0.74. Big Five normalized correlation 0.80 against 0.55-0.61 for demographic agents | the whole transcript in every call (the paper's averaged 6,491 words); `prepare` costs 1 + 4 calls per person, each over the whole transcript; with `cot=True` use `max_tokens >= 1024` | The interview can contain the answer verbatim: then the score measures retrieval, not simulation (v3 removed 27 near-duplicate GSS items; simengine does no such screening). The default hold-out drops only the target, the paper's weaker variant. Short or synthetic transcripts are a different regime from two-hour interviews |
| **Behaviour logs** (`history` of events) | `log_stats` first (no LLM summary, the baseline), then `log_raw`, `log_retrieved` (BM25), `log_paars`, `log_task_summary`, `log_agent4rec` | Mixed. PAARS purchase prediction 25.5% with no context, 41.1% with history, 47.3% with the full persona (Mansour et al. 2025, arXiv:2503.24228). LaMP-1 accuracy 0.518 → 0.649 with BM25 retrieval (Salemi et al. 2024, arXiv:2304.11406). A task-aware summary + 1 item matched or beat 4 retrieved items (0.743 vs 0.709 on LaMP-1; Richardson et al. 2023, arXiv:2310.20081). But zero-shot ChatGPT rating prediction (RMSE 1.008) trails a supervised model (0.885) and even the item-average heuristic (0.975) (Kang et al. 2023, arXiv:2305.06474) | raw and stats: one call; summaries are offline (PAARS 2 calls per person, others 1), made once before the run | Future events leak unless the question carries a `cutoff` (summaries record theirs and are refused when later). The target item is hidden by default. Supervised recommenders and simple baselines usually win on logs: README's OPeRA next action is LightGBM 31.6% against 13.7% for a zero-shot persona agent |
| **Population target** (a group's answer distribution) | `cell_verbalized` (best format in the paper), `cell_fewshot` (with other questions' real shares), `cell_sequence`, `cell_sample`; `simulate_population` + `tvd`; `ppi_distribution` / `ppi_mean` to correct with a few real labels | Meister et al. 2024 (arXiv:2411.05403), mean total variation (lower is better): verbalized 0.226-0.244, 30-sample sequence 0.278-0.328, uniform 0.363; few-shot steering beat persona steering for all models but GPT-3.5; human baseline 0.250. Measured in the research code behind simengine (not with this package): CES 2020 segment distribution, a direct distribution closed 0.818 of the gap against 0.55 for a persona panel (README, docs/RESULTS.md) | one call per cell, not per person; PPI needs a labelled random subset | Individual variation inside a cell is invisible: per-person accuracy is capped by the cell's majority share, so score with TVD. PPI CIs assume the labelled people are a random subset. The `overall` weights of simulated and real distributions differ when missingness varies by cell |
| **Open model** (weights you can train or steer) | `subpop_lora` (group distributions, forward KL), `centaur_qlora` (one transcript of answers per person), `caa_steer` / `persona_vector_steer` (activation steering), `soft_prompt` (one tuned prompt per group); `ft.recipe`, `ft.train_colab_t4` | SubPOP: Wasserstein distance 32-46% lower than the best prompt baseline on OpinionQA; Llama-3-70B 0.138 → 0.094 (Suh et al. 2025, arXiv:2502.16761). Centaur: held-out NLL 0.44 against 0.58 for the base model and 0.56 for cognitive models (Binz et al. 2025, arXiv:2410.20268). The steering papers report behaviour shifts and r = 0.75-0.83 between projection and trait expression (arXiv:2312.06681, arXiv:2507.21509), not survey accuracy | a Colab T4 run per adapter (default Qwen2.5-1.5B in 4-bit); inference needs a locally served model; the API arms only show the prompt | `train_colab_t4`, `steering_vector` and `steer` have never run here (no torch). The papers used 7B-70B models. Centaur rows repeat each label once per question. With an API model the steering and soft-prompt arms send the bare question, so their scores are meaningless |

Two more inputs have their own families:
- **Network or ABM state** (neighbours' posts, actions, answers; group norms; news): `social` (`peer_messages`,
  `neighbor_activity`, `asch_majority`, `descriptive_norm`, `news_feed`), with `attach_neighbors` / `attach_norms`
  as the bridge from `simengine.abm`. The papers report conformity and norm effects (e.g. a repost-dominant norm
  raised repost against like with OR 89.1; Peng et al. 2026, arXiv:2603.20911), not accuracy on a person. Compare
  against `swap` to separate the network signal from homophily.
- **Value or culture scores** (Schwartz, MFQ, Hofstede, IVS): `values`. All five papers are group-level (Kang
  et al.'s prompt baselines reach NMSE 0.146-0.196 on cluster and country profiles against 0.034 fine-tuned;
  arXiv:2310.17857). A score computed from items that include the target carries the answer.

A first run that answers most questions in one call:
`["plain_fields", ("own_answers_rag", {"questions": bank, "k": 3}), ("similar_respondents", {"pool": train_people}),
("cell_verbalized", {"keys": [...]})]` plus the default controls, on held-out questions, then the best arm
against a `simengine.methods` baseline through `se.evaluate`.

## Families

| family | injectors | reads | page |
|---|---|---|---|
| control | `no_persona`, `plain_fields` | `fields` (or flat scalars) | this page |
| framing | `framed` (placement x voice x format x order grid), `argyle_backstory`, `hu_collier_profile`, `lutz_interview`, `lutz_third_person` | `fields` | [framing](inject/framing.md) |
| fitted | `fitted_topk`, `fitted_stepwise`, `fitted_grips`, `fitted_opro` | `fields`; fit on other questions first | [fitted](inject/fitted.md) |
| generated | `anthology_backstory`, `anthology_greedy`, `personahub`, `llm_narrative` | `generated` texts made offline | [generated](inject/generated.md) |
| values | `schwartz_scores`, `schwartz_anchor`, `moral_foundations`, `hofstede_country`, `cultural_prompting` | `schwartz`, `mfq`, `country`, ... | [values](inject/values.md) |
| own_answers | `own_answers_all`, `own_answers_fewshot`, `own_answers_rag` | `answers` | [own_answers](inject/own_answers.md) |
| memory | `memory_stream`, `memory_reflect`, `memory_session` | `answers`, `reflections` | [memory](inject/memory.md) |
| neighbours | `similar_respondents`, `neighbour_votes`, `answer_neighbours` | `fields` / `answers` + a `pool` | [neighbours](inject/neighbours.md) |
| interview | `interview_transcript`, `interview_lesion`, `interview_summary`, `interview_expert_reflection`, `interview_plus_survey` | `interview` (+ offline notes) | [interview](inject/interview.md) |
| from_logs | `log_raw`, `log_retrieved`, `log_stats`, `log_agent4rec`, `log_paars`, `log_task_summary` | `history` events (+ offline summaries) | [from_logs](inject/from_logs.md) |
| population | `cell_verbalized`, `cell_sequence`, `cell_sample`, `cell_fewshot` | `fields` (the cell keys) | [population](inject/population.md) |
| social | `peer_messages`, `neighbor_activity`, `asch_majority`, `descriptive_norm`, `news_feed` | `neighbors`, `norms`, `media` | [social](inject/social.md) |
| finetune | `subpop_lora`, `centaur_qlora`, `caa_steer`, `persona_vector_steer`, `soft_prompt` | `fields` / `answers`; an open model | [finetune](inject/finetune.md) |

Injectors that need a constructor argument say so when they are built: `pool=` (neighbours), `questions=`
(`own_answers_rag`, and the text of past items elsewhere), `people=` (`cell_fewshot`), `traits=` (steering), a
`.fit()` (fitted) or a registered backstory pool (`anthology_greedy`). Offline builders that cost LLM calls
(`interview.prepare`, `from_logs.summarize_logs`, `generated.narrate_people`, `memory.reflect`, ...) run once
before the comparison and store their output in the person dict, so `render` stays pure and a dry run is free.
`tests/test_persona_inject_zoo.py` builds every registered injector on one synthetic panel and is a working
reference for the arguments.

## `persona.inject` and `persona.methods` (the zoo)

They answer different questions and compose.

| | `persona.methods` (10 methods) | `persona.inject` (52 injectors + 2 controls) |
|---|---|---|
| question | what persona text should a synthetic agent have? | how should a *real* person's data reach the prompt? |
| input | a seed (and for 3 methods a released data file) | one person dict with their own fields, answers, interview, log, ... |
| output | `n` persona texts (`sample_texts`, `method_personas`, `AgentBank`) | a `Prompt` per (person, question) |
| truth | none per person; scored as a population (games' distance to human numbers, Big Five fidelity) | that person's real answer; paired CIs per arm |
| controls | `baseline` ("" persona) | `no_persona` and `swap_<arm>` |
| typical use | `simengine.games`, `abm` societies, `sim` panels without data | per-person prediction, survey simulation from a panel, digital twins |

How they compose:
- **A zoo text as an injector arm.** Put the text where a `generated` injector reads it:
  `{"id": ..., "generated": {"narrative": text}}` with `llm_narrative(style="system")` gives the zoo's own layout
  (text as the system prompt, then "Answer as this person would."). A released Anthology story from
  `methods.anthology` goes in `generated["backstory"]` for `anthology_backstory`, which renders it as the answer
  to the paper's elicitation question; `anthology_greedy(pool="released")` matches released stories to people by
  age bracket (the data file is not shipped; see `persona.methods`).
- **Same paper, two sides.** `methods.value_anchor` samples "Answer as a person that is ..." anchors by
  prevalence; `inject.values.schwartz_anchor` builds the same sentence (the same `ANCHORS`) from a person's
  measured top value. `methods.twin2k` returns the released Twin-2K-500 summary of a respondent; `own_answers_all`
  shows that respondent's full answers. On Twin-2K-500 that is the paper's summary-against-full comparison
  (68.02% against 71.72%).
- **Zoo agents with real context.** `method_personas` gives `{"id", "text", ...}` dicts. Add `neighbors`,
  `norms` or `media` (for example from an ABM step through `social.attach_neighbors`) and the `social` injectors
  render the zoo text as the persona block (`persona=True` reads `text`).
- **Persona objects from tables.** `sp.from_table` / `sp.from_history` build `Persona` objects for the sim
  runners. Injectors take plain dicts, so build `{"id": uid, "fields": {...}, "answers": {...}}` from the same
  table, keeping the target columns out of `fields`.

## Conventions

Person dict: `id` (required), `text` (narrative), `fields` ({attribute: value}), `answers` ({qid: answer}),
`interview` (str or [(q, a)]), `history` ([(role, text)] or event dicts). An injector lists what it reads in
`needs`; a missing field raises `MissingPersonField`.

Question dict: `id`, `text`, `options` (choice) or none (number, optional `min`/`max`), `exclude` (qids that
must not be shown either), `exclude_people` (person ids never shown as neighbours or peers).
`Injector.question_block(q)` appends the strict answer line that `parse_option` reads (`{"answer": ...}`, same
parser as `simengine.games`).

Leakage: injectors showing a person's own answers use `Injector.known_answers(person, question)`, which drops
the target and `exclude` ids. The swap control adds the scored person to `exclude_people`, so a pool-based
injector never shows them to their own donor. `tests/test_persona_inject_zoo.py` checks, for every registered
injector and its swap control, that changing the scored person's own answer to the target leaves the prompt
unchanged.

History: multi-turn prompts are flattened (`Prompt.flatten()`) unless the LLM sets `supports_history = True`
(then `history=` is passed). `simengine.llm.Claude` does not set it: its cache key omits `messages`.

## Adding a family

`simengine/persona/inject/<family>.py`, tests in `tests/test_persona_inject_<family>.py`, a page in
`docs/modules/inject/<family>.md`, and one row per new injector in `CONFIG` of `tests/test_persona_inject_zoo.py`
(the zoo test fails until it is there). The package imports every non-underscore module itself (pkgutil): a family
never edits `__init__.py`, `_base.py` or `compare.py`. A module that fails to import is skipped with a warning
and listed in `pi.IMPORT_ERRORS`.

```python
from ._base import Injector, Prompt, register

@register
class Bio(Injector):
    name, family, paper, needs = "bio", "narrative", "Author et al. 2024", ("text",)
    description = "Narrative persona in the system prompt."

    def __init__(self, style: str = "first", **params):
        super().__init__(style=style, **params)   # arm label = name + params given to get_injector
        self.style = style

    def render(self, person, question):
        return Prompt(system=person["text"], user=self.question_block(question))
```

`get_injector("bio")` is the arm `bio`; `get_injector("bio", style="third")` is `bio(style=third)`. Two arms
with the same label are refused. Keep `render` deterministic (randomness only from a `seed` param) so reruns hit
the LLM disk cache.

Heavy dependencies (torch, transformers, peft) are imported inside functions only, with a clear
ImportError; their tests skip when missing.

## Known gaps

- No injector has been run against real people in simengine; the playbook is the literature's ranking, to be
  checked on your data with `compare_injectors`.
- Retrieval and similarity are lexical (TF-IDF, BM25, Gower match), not the papers' embeddings.
- `compare_injectors(max_tokens=256)` is too short for `interview_*` with `cot=True`: pass `max_tokens=1024` or
  `cot=False`, or the replies score as unparsed.
- The held-out guards match ids and exact text only. A reworded copy of the target, or a field named differently
  from its question id, still leaks; drop those by hand.
- Unknown constructor params are accepted silently by most families (framing refuses them) and only show up in the
  arm label.
