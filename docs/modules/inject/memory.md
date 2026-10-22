# `persona.inject` family: memory

`simengine/persona/inject/memory.py`. This family treats the person's earlier answers in the session as a memory stream, following Generative Agents (Park et al. 2023, arXiv:2304.03442). Each known answer becomes one memory record, and the injector retrieves a subset of them for the target question. Retrieval excludes the target question and its `exclude` ids, because the records come from `Injector.known_answers`.

```python
import simengine.persona.inject as pi
from simengine.persona.inject.memory import rate_importance, reflect

inj = pi.get_injector("memory_stream", k=5, questions=qs)              # qs: ordered question dicts
p = inj(person, qs[3]); llm.complete(p.system, p.flatten())
imp = rate_importance(llm, people, qs)                                  # offline, the paper's poignancy prompt
inj = pi.get_injector("memory_stream", questions=qs, importance=imp)
person["reflections"] = reflect(llm, person, qs, hide=test_qids)       # offline; insights that never saw test_qids
pi.compare_injectors(llm, people, qs, "answers", ["memory_stream", "memory_reflect", "memory_session"], dry_run=True)
```

## Injectors

| name | needs | what it renders |
|---|---|---|
| `memory_stream` | `answers` | System prompt: the top `k` answers (default 5) by the paper's retrieval score, listed in session order |
| `memory_reflect` | `answers`, `reflections` | `memory_stream`, with reflections from `reflect()` competing in the same top `k` |
| `memory_session` | `answers` | The questionnaire so far as a multi-turn history. Interviewer turns hold the question; the person's turns hold `{"answer": ...}`. There is no retrieval. `k` keeps the most recent turns, and `prior_only=True` keeps only questions asked before the target |

**Retrieval score.** The score follows the paper's section 4.1:

`a_rec*recency + a_imp*importance + a_rel*relevance`

Each term is min-max scaled to [0, 1]. A term that is constant across all memories scales to 0, so it does not change the ranking.

- **Weights:** `weights=(1, 1, 1)` by default, as in the paper.
- **Recency:** `decay ** (now - time)`, with `decay=0.995` by default (the paper's value). Here `time` is the questionnaire position from `questions=`, or the insertion order of `person["answers"]` when `questions=` is not given. Because of the min-max scaling, `decay` changes only the shape of the curve, not its range.
- **Importance:** comes from `importance=`, which can be a `{(person_id, qid): 1-10}` map, a `{qid: score}` map, or a callable `(person, qid, answer)`. `rate_importance` builds the map with the paper's prompt ("On the scale of 1 to 10, where 1 is purely mundane ... rate the likely poignancy ..."). Without it, importance is a constant 5 and has no effect.
- **Relevance:** bag-of-words cosine between the target question text and the memory's question and answer, with stopwords removed. You can pass a pure `embed=` callable for embedding cosine, which is what the paper used.

The paper resets recency each time a memory is retrieved. This module does not, because `render` must stay pure.

**Reflection.** `reflect()` follows section 4.2 of the paper:

1. From the 100 most recent records, it asks for the "3 most salient high-level questions" about the person.
2. For each of those questions, it retrieves the top `k` records.
3. It asks for "5 high-level insights ... (example format: insight (because of 1, 5, 3))".

Any insight without a valid citation is dropped. Each reflection stores:

- `sources`: the qids it cites.
- `seen`: every qid the LLM was shown.

`memory_reflect` drops any reflection whose `seen` list contains the target or one of its `exclude` ids. It records the number dropped in `meta["reflections_dropped"]`. A reflection without a `seen` list raises an error.

To evaluate on held-out questions, call `reflect(..., hide=test_qids)` once per person. A strict leave-one-out run would need one `reflect` call per target.

The paper triggers reflection when the summed importance of recent events passes 150. Here, reflection runs once, offline. Reflection trees, where a reflection cites earlier reflections, are not implemented.

**Session.** `memory_session` presents the stream in raw chronological order, which corresponds to the paper's observation log without retrieval. With `as_history=False`, it renders the same turns as `- question: answer` lines in the system prompt. `compare_injectors` flattens the history unless the LLM sets `supports_history`.

Every `render` is deterministic: it has no random component, and ties break by later time, then by id. Long parameter values, such as question lists and maps, appear in arm labels as a 6-character hash, so labels stay short and stable.

## Published effect sizes

The paper measured believability in an agent sandbox, not survey prediction. Participants ranked the conditions, and the scores are TrueSkill μ from the paper's controlled evaluation (100 participants):

| condition | μ |
|---|---|
| full architecture | 29.89 |
| no reflection | 26.88 |
| no reflection, no planning | 25.64 |
| human crowdworker | 22.95 |
| no memory, no reflection, no planning | 21.21 |

The paper reports d = 8.16 for the full architecture against the fully ablated baseline. These are believability rankings and do not transfer to answer accuracy.

**In simengine:** not yet measured. No real LLM run has been made, only FakeLLM tests (`tests/test_persona_inject_memory.py`).

## Limitations

- The memories are survey answers, not lived observations. Without `rate_importance`, the importance term is flat.
- Bag-of-words relevance ignores synonyms. Supply `embed=` for semantic matching.
- `rate_importance` and `reflect` make LLM calls, so they cost money. Run them once, offline, and store the results in the person dict or in the `importance=` map.
