# `persona.inject` family: `generated`

The persona is text an LLM wrote about the person beforehand: an Anthology backstory, a Persona Hub description or a
narrative written from structured fields. Generation is a separate offline step (one LLM pass per person, disk
cached). The injectors only read what it produced, so `render` stays pure. Generated texts live in
`person["generated"] = {"backstory": str, "persona": str, "narrative": str}`; `attach()` writes them there.
Module: `simengine/persona/inject/generated.py`.

```python
import simengine.persona.inject as pi
from simengine.persona.inject import generated as g

stories = g.generate_backstories(base_llm, 500)                     # "Question: Tell me about yourself...\n\nAnswer:"
stories = g.estimate_demographics(llm, stories, demo_questions)      # each trait question asked 40 times per story
people = g.attach(people, g.match_backstories(people, stories, ["age_bracket", "gender"]), "backstory")
people = g.attach(people, g.text_to_persona(llm, people), "persona")                  # Persona Hub
people = g.attach(people, g.narrate_people(llm, people, exclude=["vote"]), "narrative")
g.register_pool("w34", stories)
arms = ["anthology_backstory", ("anthology_greedy", {"pool": "w34", "keys": ["age_bracket", "gender"]}),
        "personahub", ("llm_narrative", {"with_fields": True})]
res = pi.compare_injectors(llm, people, questions, "answers", arms, dry_run=True)
```

| injector | what goes in the prompt | needs | params |
|---|---|---|---|
| `anthology_backstory` | The person's assigned backstory as the answer to the elicitation question, then the consistency line and the question | `generated` (`backstory`) | `placement` prefix (paper, user turn) \| system \| history, `consistency`, `elicitation` |
| `anthology_greedy` | Same layout; the story is picked at render time from a pool, by greedy matching on the person's fields | `fields` (+ every trait in `keys`) | `pool` name \| list \| `"released"`, `keys`, `seed` (ties), placement params |
| `personahub` | A one-paragraph persona description from `text_to_persona` | `generated` (`persona`) | `style` roleplay (system) \| prefix (user turn) |
| `llm_narrative` | A second-person narrative written from the person's fields, in the system prompt | `generated` (`narrative`), plus `fields` with `with_fields=True` | `with_fields`, `keys` |

A missing generated text raises `MissingPersonField`. The message lists the injector's `needs` and names the
builder that makes the text (`generate_backstories/match_backstories`, `text_to_persona` or `narrate_people`).

Leakage: the generators read only `fields` and `text`, never `answers`. At render time, a field whose name is the
target question or one of its `exclude` ids is never shown (`llm_narrative(with_fields=True)`). It is never matched
on either: `anthology_greedy` refuses such a key. Generation runs before any question, so a field that *is* a
target has to be dropped by hand with `narrate_people(exclude=[...])` or `keys=`. `text_to_persona` shows the
person's own text verbatim, so that text has to predate the survey.

Builders:

- `generate_backstories(llm, n)`: call *i* uses `sample=i`, so the disk cache keeps the calls apart. The output is
  cut at the paper's stop sequences, and stories under `min_words` are dropped. The paper used a base model
  (completion prompt). With a chat model, the same text goes in as the user turn.
- `estimate_demographics(llm, stories, questions, samples=40)`: question `id` is the trait name, and `options`
  are its values. The result is `{trait: {value: share}}` over the parsed replies. At temperature 0 each share
  collapses to a point mass, so this step needs a sampling LLM.
- `match_backstories(people, pool, keys, method="greedy"|"hungarian")`: the edge weight is
  `prod_k P_story(person's value of trait k)`. `greedy` takes the argmax per person, and stories can be reused.
  Ties are broken by a stable hash of (seed, person, story). `hungarian` is a one-to-one maximum weight matching
  (scipy). The `age_bracket` trait is derived from a numeric `age` field (18-29, 30-49, 50-64, 65+).
- `released_pool()`: the published Anthology stories, read through `persona.methods.anthology` (data under
  `$SIMENGINE_DATA`, not shipped). They are labelled only by age bracket, so this pool matches on
  `age_bracket` alone.
- `text_to_persona(llm, people, relation="write")`, `persona_to_persona(llm, personas, rounds)` and
  `dedup_personas(personas, 0.9)`.
- `narrate_people(llm, people, keys=None, exclude=(), words=150)`.

## Literature

**`anthology_backstory`, `anthology_greedy`: Moon et al. 2024, Virtual Personas for Language Models via an
Anthology of Backstories (arXiv:2407.06576).**
- *Method:* open-ended backstories come from a base model given "Tell me about yourself..." with no conditioning
  (the code follows the released CannyLab/anthology config wording). Each story's demographics are estimated by
  asking the demographic survey 40 times with the story as context. Stories are then matched to the target human
  sample, either greedily or by maximum weight matching. The edge weight is the product of trait probabilities.
- *Evaluation:* Pew ATP waves 34, 92 and 99, scored by Wasserstein distance, Frobenius norm and Cronbach's alpha.
- *Published effect:* the abstract reports "up to 18%" better match to human response distributions and 27%
  better consistency.
  - Llama-3-70B, Table 1, WD for Bio / QA / greedy Anthology: wave 34 0.254 / 0.238 / 0.227, wave 92
    0.348 / 0.371 / 0.313, wave 99 0.296 / 0.327 / 0.288.
  - Greedy matching usually beats maximum weight matching on WD. Neither variant wins on every metric.
- *Here:* `anthology_backstory` is the full estimate-then-match pipeline. `anthology_greedy` does the matching
  inline from a pool. The `system` placement reuses `persona.methods.anthology.TEMPLATE`, which bridges the
  backstory for instruction-tuned models.

**`personahub`: Ge et al. 2024, Scaling Synthetic Data Creation with 1,000,000,000 Personas (arXiv:2406.20094).**
- *Method:* text-to-persona asks "Who is likely to [read|write|like|dislike|...] the text?" about a web text.
  Persona-to-persona asks "Who is in close relationship with the given persona?" (up to six rounds). The personas
  are deduplicated with MinHash on 1-grams at 0.9; here exact Jaccard is used.
- The paper says its printed prompts are simplified, so the wording around those questions is ours.
- *Published effect:* the paper uses personas to diversify synthetic data (math problems, instructions, NPCs). It
  does not use them to predict a specific person's answers. It reports no survey or individual-prediction effect.
- *Here:* the persona is inferred from the person's own text, a person-level use the paper did not test.

**`llm_narrative`: Li et al. 2025, LLM Generated Persona is a Promise with a Catch (arXiv:2503.16527).**
- *Method:* four persona tiers with more and more LLM-written content: meta (sampled demographics only), objective
  tabular, subjective tabular and descriptive (a free narrative). The generator system line is the paper's own
  (Appendix B.1). The user prompt is ours, because the paper's descriptive-persona prompt is not printed in
  readable form.
- *Published effect (a warning):* in their 2024 US presidential election simulation (Llama 3.1 70B), results moved
  steadily left as the share of LLM-generated content rose. With descriptive personas, every state voted for the
  Democratic candidate. Meta personas came closest to the real outcome, and 2016 and 2020 showed the same drift. The
  paper gives these results as figures, not as a numeric table.
- *Here:* `with_fields=True` adds the source fields back. It is the natural ablation of that drift: narrative
  against `plain_fields` against narrative plus fields.

**In simengine:** not yet measured in simengine. No generated-persona arm has been run against human answers here.
