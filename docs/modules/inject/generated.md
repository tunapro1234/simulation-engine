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
        "personahub", "llm_narrative", ("llm_narrative", {"with_fields": True})]
res = pi.compare_injectors(llm, people, questions, "answers", arms, dry_run=True)
```

| injector | what goes in the prompt | needs | params |
|---|---|---|---|
| `anthology_backstory` | The person's assigned backstory as the answer to the elicitation question, then the consistency line and the question | `generated` (`backstory`) | `placement` prefix (paper, user turn) \| system \| history, `consistency`, `elicitation` |
| `anthology_greedy` | Same layout, after the story's own elicitation question; the story is picked at render time from a pool, by greedy matching on the person's fields | `fields` (+ every trait in `keys`) | `pool` name \| list \| `"released"`, `keys`, `seed` (ties), placement params |
| `personahub` | A one-paragraph persona description from `text_to_persona` | `generated` (`persona`) | `style` roleplay (system) \| prefix (user turn) |
| `llm_narrative` | A descriptive persona written from the person's fields. `paper`: the Li et al. B.5 opinion-simulation prompt (guidelines in system, `### PERSONA ###` and `### QUESTION ###` in user). `system`: the narrative as the system prompt | `generated` (`narrative`), plus `fields` with `with_fields=True` | `style` paper \| system, `with_fields`, `keys` |

A missing generated text raises `MissingPersonField`. The message lists the injector's `needs` and names the
builder that makes the text (`generate_backstories/match_backstories`, `text_to_persona` or `narrate_people`).

Leakage: the generators read only `fields` and `text`, never `answers`. At render time, a field whose name is the
target question or one of its `exclude` ids is never shown (`llm_narrative(with_fields=True)`). It is never matched
on either: `anthology_greedy` refuses such a key, and also a derived key whose source field is the target
(`age_bracket` when the target or an `exclude` id is `age`). Generation runs before any question, so a field that *is* a
target has to be dropped by hand with `narrate_people(exclude=[...])` or `keys=`. `text_to_persona` shows the
person's own text verbatim, so that text has to predate the survey.

Builders:

- `generate_backstories(llm, n, preamble=None)`: call *i* uses `sample=i`, so the disk cache keeps the calls apart.
  The output is cut at the released code's stop strings ("Question", "Q:", "Interview", wherever they occur, as
  an API `stop=` would). Stories under `min_words` (default 30, about the released loader's 40 tokens) are
  dropped. Each story keeps its elicitation `question`. The paper used a base model (davinci-002, completion
  prompt). With a chat model, the same text goes in as the user turn. The released default config first asks
  "Are you a US citizen?" and resamples until the answer contains "yes"; `preamble=US_PREAMBLE` reproduces it
  with a fixed "Yes." (off by default, since the target population need not be US).
- `estimate_demographics(llm, stories, questions, samples=40)`: question `id` is the trait name, and `options`
  are its values. Each story is shown after its own elicitation question. The result is `{trait: {value: share}}` over the parsed replies. At temperature 0 each share
  collapses to a point mass, so this step needs a sampling LLM.
- `match_backstories(people, pool, keys, method="greedy"|"hungarian")`: the edge weight is
  `prod_k P_story(person's value of trait k)`. `greedy` takes the argmax per person, and stories can be reused.
  Ties are broken by a stable hash of (seed, person, story). `hungarian` is a one-to-one maximum weight matching
  (scipy). The `age_bracket` trait is derived from a numeric `age` field (18-29, 30-49, 50-64, 65+). A person
  who lacks a matched trait is refused (`MissingPersonField`) rather than given an arbitrary story at weight 0.
- `released_pool()`: the published Anthology stories, read through `persona.methods.anthology` (data under
  `$SIMENGINE_DATA`, not shipped). They are labelled only by age bracket, so this pool matches on
  `age_bracket` alone.
- `text_to_persona(llm, people, relation="write")`, `persona_to_persona(llm, personas, rounds)` and
  `dedup_personas(personas, 0.9)`.
- `narrate_people(llm, people, keys=None, exclude=(), words=None)`: the paper's prompts (B.1 system, B.4 user)
  with the person's fields as JSON metadata. `words` adds a length line the paper does not have. The leading
  "Persona:" label the prompt asks for is stripped.

## Literature

**`anthology_backstory`, `anthology_greedy`: Moon et al. 2024, Virtual Personas for Language Models via an
Anthology of Backstories (arXiv:2407.06576).**
- *Method:* open-ended backstories come from a base model given "Tell me about yourself..." with no conditioning
  (the code follows the released CannyLab/anthology config wording, stop strings and > 40-token filter). Each story's demographics are estimated by
  asking the demographic survey 40 times with the story as context. Stories are then matched to the target human
  sample, either greedily or by maximum weight matching. The edge weight is the product of trait probabilities.
- *Evaluation:* Pew ATP waves 34, 92 and 99, scored by Wasserstein distance, Frobenius norm and Cronbach's alpha.
- *Published effect:* the abstract reports "up to 18%" better match to human response distributions and 27%
  better consistency.
  - Llama-3-70B, Table 1, WD for Bio / QA / greedy Anthology: wave 34 0.254 / 0.238 / 0.227, wave 92
    0.348 / 0.371 / 0.313, wave 99 0.296 / 0.327 / 0.288.
  - Maximum weight matching, same row: WD 0.229 / 0.337 / 0.327. Greedy beats it on WD in all three waves here,
    but not on every metric (wave 99: Frobenius 0.686 vs 0.765, alpha 0.756 vs 0.744 for maximum weight).
- *Here:* `anthology_backstory` is the full estimate-then-match pipeline. `anthology_greedy` does the matching
  inline from a pool. The `system` placement reuses `persona.methods.anthology.TEMPLATE`, which bridges the
  backstory for instruction-tuned models.

**`personahub`: Ge et al. 2024, Scaling Synthetic Data Creation with 1,000,000,000 Personas (arXiv:2406.20094).**
- *Method:* text-to-persona asks "Who is likely to [read|write|like|dislike|...] the text?" about a web text.
  Persona-to-persona asks "Who is in close relationship with the given persona?" (six rounds per persona). The
  personas are deduplicated with MinHash on 1-grams (128 permutations) at 0.9, then by embedding cosine > 0.9.
  Here the first pass is exact Jaccard and the embedding pass is not done.
- The paper says its printed prompts are simplified, so the wording around those questions is ours.
- *Published effect:* the paper uses personas to diversify synthetic data (math problems, instructions, NPCs). It
  does not use them to predict a specific person's answers. It reports no survey or individual-prediction effect.
- *Here:* the persona is inferred from the person's own text, a person-level use the paper did not test.

**`llm_narrative`: Li et al. 2025, LLM Generated Persona is a Promise with a Catch (arXiv:2503.16527).**
- *Method:* four persona tiers with more and more LLM-written content: meta (sampled demographics only), objective
  tabular, subjective tabular and descriptive (a free narrative). The prompts here are the paper's own, verbatim:
  Appendix B.1 system, B.4 descriptive-persona instructions, and B.5 for answering. Two things are ours: the
  metadata is the person's own fields as JSON (the paper fills it from sampled Census-style demographics), and the B.5 line "Output
  format: 'Answer: [Letter]'" is replaced by simengine's strict answer line. The B.1 system line also mentions a
  JSON template. The paper printed one system prompt for all tiers, so it is kept as is.
- *Published effect (a warning):* in their 2024 US presidential election simulation (Llama 3.1 70B), results moved
  steadily left as the share of LLM-generated content rose. With descriptive personas, every state voted for the
  Democratic candidate. Meta personas came closest to the real outcome, and 2016 and 2020 showed the same drift. The
  paper gives these results as figures, not as a numeric table.
- *Here:* `with_fields=True` adds the source fields back. It is the natural ablation of that drift: narrative
  against `plain_fields` against narrative plus fields. `style="system"` is a non-paper layout for comparison.

**In simengine:** not yet measured in simengine. No generated-persona arm has been run against human answers here.
