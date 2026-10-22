# `simengine.persona.inject`

Persona injection methods from the literature, one module per family, compared on the same cells.
An injector renders `(person, question) -> Prompt(system, user, history, meta)`; `compare_injectors` runs
several with a `no_persona` and a `swap` control and reports paired cluster-bootstrap CIs
(`simengine.evidence.paired_diff`, cluster = person).

```python
import simengine.persona.inject as pi
pi.list_injectors()                         # name, family, paper, needs, description
p = pi.get_injector("plain_fields")(person, question)
llm.complete(p.system, p.flatten())
plan = pi.compare_injectors(llm, people, questions, "answers", ["plain_fields"], dry_run=True)
plan.estimate                               # calls, input tokens (chars/4), upper-bound USD; no call made
res = pi.compare_injectors(llm, people, questions, "answers", ["plain_fields"], max_usd=2.0)
res.table                                   # arm, score [lo, hi], diff/CI vs no_persona and vs swap_<arm>
```

## Conventions

Person dict: `id` (required), `text` (narrative), `fields` ({attribute: value}), `answers` ({qid: answer}),
`interview` (str or [(q, a)]), `history` ([(role, text)] or event dicts). An injector lists what it reads in
`needs`; a missing field raises `MissingPersonField`.

Question dict: `id`, `text`, `options` (choice) or none (number, optional `min`/`max`), `exclude` (qids that
must not be shown either). `Injector.question_block(q)` appends the strict answer line that `parse_option`
reads (`{"answer": ...}`, same parser as `simengine.games`).

Leakage: injectors showing a person's own answers use `Injector.known_answers(person, question)`, which drops
the target and `exclude` ids.

History: multi-turn prompts are flattened (`Prompt.flatten()`) unless the LLM sets `supports_history = True`
(then `history=` is passed). `simengine.llm.Claude` does not set it: its cache key omits `messages`.

## Adding a family

`simengine/persona/inject/<family>.py`, tests in `tests/test_persona_inject_<family>.py`. The package imports
every non-underscore module itself (pkgutil): never edit `__init__.py`, `_base.py` or `compare.py`. A module that
fails to import is skipped with a warning and listed in `pi.IMPORT_ERRORS`.

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
