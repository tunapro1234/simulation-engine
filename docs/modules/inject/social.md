# `simengine.persona.inject.social`

Social context next to the persona: what the person's contacts posted or did, what earlier participants answered,
how common each answer is in a reference group, and what the person read in the news. The family exists so that
an ABM step (`simengine.abm`) can hand each agent its neighbours' current states and the LLM answers in that context.

```python
import simengine.persona.inject as pi
from simengine.persona.inject.social import attach_neighbors, attach_norms, norm_from_dist, prevalence_item

people = attach_neighbors(people, edges, states={"u7": {"action": "adopted the app"}})   # graph -> neighbors
people = attach_norms(people, questions, by="region")      # leave-one-out answer shares per region
pi.get_injector("asch_majority", tone="confident")(people[0], question)
pi.compare_injectors(llm, people, questions, "answers",
                     ["descriptive_norm", ("peer_messages", {"bias": "weak"}), "asch_majority"], dry_run=True)
```

Person fields the family reads:

| field | shape | used by |
|---|---|---|
| `neighbors` | `[{id, label, answers: {qid: answer}, message, action, time}]`; a bare string is a message; oldest first | `peer_messages`, `neighbor_activity`, `asch_majority`, `descriptive_norm(source="neighbors")` |
| `norms` | `{qid: {option: share or count} | number}` | `descriptive_norm` |
| `media` | `[str | {source, text, time}]` | `news_feed` |
| `state` | str, or `{qid: value}`: the person's prior view (last wave, last ABM step); blank / NaN is skipped | `peer_messages` (as "Your initial belief"), `neighbor_activity` |
| `fields`, `text` | persona block, on by default except `asch_majority` | all (`persona=True/False`) |

## Injectors

| name | needs | knobs | paper |
|---|---|---|---|
| `peer_messages` | `neighbors` | `k=5`, `select` (`recent`/`random`), `bias` (`none`/`weak`/`strong`), `closed_world=True`, `persona`, `seed` | Chuang et al. 2023, arXiv:2311.09618 |
| `neighbor_activity` | `neighbors` | `k=10`, `action=` (adds "n of your m contacts have ..."), `select`, `persona`, `seed` | Gao et al. 2023 (S3), arXiv:2307.14984 |
| `asch_majority` | `neighbors` (none with `answer=`) | `tone` (`plain`/`neutral`/`confident`/`uncertain`), `distill`, `answer` + `n`, `k`, `persona=False`, `seed` | Zhu et al. 2024, arXiv:2410.12428 |
| `descriptive_norm` | `norms` (or `neighbors`) | `source` (`norms`/`neighbors`), `group` (may hold `{field}`), `style` (`shares`/`most`), `decimals`, `persona` | Peng et al. 2026, arXiv:2603.20911 |
| `news_feed` | `media` | `k=3`, `context`, `select`, `persona`, `seed` | Williams et al. 2023, arXiv:2307.04986 |

Items are shown oldest first: sorted by `time` when every item has one, else in list order. `select="recent"`
keeps the last `k`; `select="random"` draws a sample seeded by (`seed`, person id, question id), so a rerun
renders the same text. `k` is a positive int or `None` (all). Blank, NaN and pandas NA values (answers,
messages, norms, media text, `state`) count as missing.

### `peer_messages`

What the paper did: each agent got "Role play this person." plus a persona with an initial belief about a
topic, and was told it was talking with strangers on Twitter. At each step a random pair met: one agent wrote a
tweet, the other read it and reported its belief, rated on a 5-point scale. Memory was either cumulative (every
experience appended) or reflective (a rolling summary). Two prompt add-ons were tested: confirmation bias,
weak or strong (Appendix F), and a closed-world line, "alone in your room with limited access to the
Internet" (Appendix G).

How this injector differs: it is one-shot. The k most recent (or sampled) neighbour messages appear as a block of
posts already read, like the cumulative memory, and the question follows. As in the paper's persona prompt, "Role
play this person." comes first, then the persona, then the framing ("you have been reading posts from people you
follow ... change or maintain your belief"; the paper's is "interacting with other strangers on Twitter"). The
bias lines are Appendix F verbatim. The closed-world line is Appendix G verbatim except that "XYZ" becomes "this
topic" and "other strangers on Twitter" becomes "the people you follow". `state` is shown as "Your initial
belief: ...", the paper's initial-opinion slot.

Published effect (Table 1, 15 topics, cumulative memory): without confirmation bias the agents converged on
the factual position. Final mean opinion B was -1.33 under false framing and 0.52 under true framing.
Confirmation bias raised the spread D, the standard deviation of final opinions. Under false framing D went
from 0.60 (none) to 0.87 (weak) and 1.24 (strong); under true framing from 0.66 to 0.95 and 1.52. One-way
ANOVAs of D over the three bias levels gave F(2) = 25.43 (cumulative) and 17.81 (reflective), both p < .001.
Agents were gpt-3.5-turbo-16k. Source: arXiv:2311.09618, Table 1 and Appendix (statistical tests), checked
against the PDF.

### `neighbor_activity`

What the paper did: S3 agents decided their emotion, attitude and whether to repost or post. The prompt held
the user's demographics, their current state and the posts received from followees. The paper describes these
inputs but publishes no verbatim template.

How this injector differs: the template is ours. It shows "<who> <action>: <message>" lines, with an optional
current attitude (`state`). With `action=`, it adds an exposure count over all neighbours, not only the ones
shown. That count is the input of a threshold or complex-contagion model, and S3 does not have it.

Published effect (Table 2): attitude change was predicted with accuracy 83.9% (AUC 0.865) and repost
interaction with accuracy 69.5% (Nuclear Energy) and 66.2% (Gender Discrimination). Those numbers are for the
S3 system as a whole, not for this prompt. Source: arXiv:2307.14984, Table 2.

### `asch_majority`

What the paper did: it adapted the Asch paradigm. The model read "You are one of the participants in an
experiment. There are in total <K> participants. Answer the questions based on your knowledge. The conversation
of the whole experiment is as follows:", then "Organiser: Answer the following multiple choice question by giving
the most appropriate response. Answer should be chosen among [A, B, C, D].", the question and lettered options,
one line per earlier participant ("Participant i: ...") and "Now, as Participant <K>, it's your turn:" (Appendix,
Tables 5 and 6). Models were Llama-3-8B, Mistral-7B-v0.3, Qwen2-7B and Gemma-2-9B, base and instruction-tuned,
greedy decoding. The tested conditions were unanimous wrong confederates, 2 to 10 participants, uniformly random
"diverse" answers, and four tones, each with several phrasings that vary across participants (Table 1): plain
"Kings"; neutral "I think it is Kings", "My answer is Kings", "Kings, in my opinion", ..., "It's Kings";
confident "I am sure it is Kings", "Kings, of course", ..., "Sure thing it's Kings"; uncertain "I am not sure if
it's Kings", "I guess it's Kings", ..., "Kings? perhaps". The paper also proposed Question Distillation, which
replaces the participant lines with "Organiser: All participants before you have chosen <ANSWER> as the answer."

How this injector differs: the template is the paper's, with option text in place of letters ("Answer should be
chosen among [yes, no]") and our strict JSON answer line after "it's your turn:"; a numeric question drops
"multiple choice" and the option list. Each tone cycles through the phrasings printed in Table 1 (participant i
gets phrasing i mod n); the paper elides the rest of its list with "...", so ours is shorter. Confederate answers
come from the neighbours' real answers to the same question, in seeded order, so it simulates social influence
rather than a wrong-answer test. `answer=` with `n=` reproduces the paper's unanimous manipulation.
`distill=True` uses the paper's line when the answers agree. When they disagree, it uses a count line of our own
("Of the 3 participants before you, 2 chose ...").

Published effect: all tested models conformed to the majority. Conformity rose with the number of confederates.
Unanimous-plain confederates lowered accuracy, and diverse-plain answers barely changed it. Neutral raised
conformity over plain, and confident raised it over neutral, in every model; uncertain lowered it for Gemma-2 and
Llama-3 but raised it for Mistral and Qwen2. Instruction tuning lowered conformity (strongly for Gemma-2 and
Llama-3, little for Mistral and Qwen2). Lower accuracy went with more conformity, with correlation -0.777 across
MMLU subjects for Llama-3-8B-Instruct. Per-model rates are in figures only. Table 3 is a different setting (the
Sharma et al. sycophancy pipeline on TriviaQA, not the Asch prompt): Llama-3-8B was sycophantic in 63.2% of cases
(base) against 25.2% (instruct), and 41.4% / 12.4% with Devil's Advocate. Source: arXiv:2410.12428, sections 3
and 5, Figure 6, Table 3, checked against the PDF.

### `descriptive_norm`

What the paper did: in a Weibo-like feed, descriptive norms went in the system prompt as approximate prevalence
of each engagement form. The like-dominant condition said about 80% like, 15% repost and 5% quote; the
repost-dominant condition said about 90% repost and 10% quote. The paper calls these "heuristic prevalence cues
rather than strict behavioral targets"; every option stayed available. It also argues that a study needs an
explicit no-norm baseline, because a default prompt is not a blank control. Agents were Qwen-8B (temperature 0.6)
with English system prompts and Chinese posts. The paper does not print its prompt text.

How this injector differs: the prevalence is over answers to the question, from `person["norms"]` (for example
built with `attach_norms`, which leaves the person's own answer out) or from the neighbours' answers. Since the
paper prints no prompt, the sentence is ours: `style="shares"` follows the paper's "approximately <option>
<share>%" description. `style="most"` ("Most people your age (62%) chose ...") is the common norm message from
the persuasion literature and is not Peng et al.'s. The closing line "You are free to give any answer." stands
for "all options stayed available". The no-norm baseline the paper asks for is the same persona without the norm
line: `plain_fields` (same attribute block) against `descriptive_norm`, or `descriptive_norm(persona=False)`
against the `no_persona` control. Comparing the default `descriptive_norm` (persona on) with `no_persona` mixes
the persona effect into the norm effect.

Published effect: against no norm, the like-dominant norm raised the odds of engaging at all (binary logit B =
0.291, OR 1.338) and the repost-dominant norm lowered them (B = -0.465, OR 0.628). Among engagers, the
repost-dominant norm raised repost against like with OR 89.1 (B = 4.490, multinomial logit). These are main
effects in models that also hold load and popularity-cue interactions. Observed liking under the like-dominant
norm was 14.5% to 24.8% of actions, against 4.1% to 9.5% with no norm. Source: arXiv:2603.20911, Tables 2 and 3
and section 4.1, checked against the PDF.

### `news_feed`

What the paper did: each agent's ChatGPT (gpt-3.5-turbo-0301) prompt held name, age, Big Five traits, a short bio
and "relevant memories", and asked whether to stay home all day. In the full-feedback condition the memories
included "[name] checks the newspaper and finds that [X]% of Dewberry Hollow's population caught new infections
of the Catasat virus yesterday." In `simengine`, `prevalence_item` builds that sentence from an ABM share.

How this injector differs: media items are listed as "You read in <source>: <text>" under "Relevant
memories:", with an optional `context` line in front (the paper's scenario sentence). The question and strict
answer line go in the user turn.

Published effect: in fixed-effect logits of the stay-home decision (Table 1), societal health information had
coefficient 1.59 (regression 2) and 3.97 with a squared term of -0.65 (regression 3), N = 66,960. The pseudo R²
rose from 0.12 (own health only) to 0.41 and 0.46. Agents self-isolated as reported cases rose, and that
flattened the epidemic curve. Source: arXiv:2307.04986, Table 1 and Results.

## Bridges to `simengine.abm`

- `attach_neighbors(people, edges, directed=False, keys=..., states=None)`: edges are `[(a, b)]` or
  `{a: [b, ...]}`. Each neighbour carries `label`, `answers`, `message`, `action` and `time` from its own
  person dict, updated with `states[id]`, for example the agent's action at the current ABM step. Self-loops and
  unknown ids are dropped, duplicate person ids raise, and the inputs are not changed.
- `attach_norms(people, questions, by=None)`: counts of the other people's answers (or their mean for numeric
  questions) within the same `by` group. The person's own answer never enters their own norm. Linear in the
  number of people; blank / NaN answers are skipped; duplicate ids raise.
- `norm_from_dist(dist, options)`: a K-category distribution, such as `abm.opinion.distribution` output, as
  `{option: share}` for `person["norms"][qid]`.
- `prevalence_item(share, population, event)`: an adoption or epidemic share as a Williams-style news item.

## Leakage

No injector here reads `person["answers"]`. A neighbour whose id equals the person's id is dropped, so a
self-loop cannot show the target answer. `state`, `norms`, `neighbors`, `media` and `group` never appear as
persona attributes, and neither do the person's own `label`, `message`, `action` and `time`: an ABM step
writes those onto every person dict for `attach_neighbors`, and the person's own current action can be the
target answer. What the caller must keep honest is the following:
- `state` must be a prior view, not the target answer.
- `norms` must be built without the person (`attach_norms` does this).
- Neighbour answers for the target question are real peer information. With strong homophily they predict
  the target well, which is the effect being measured and not a leak. To check that the effect comes from the
  network and not from the person, compare against the `swap` control (another person's neighbours).

## Measured in simengine

Not yet measured in simengine. The tests use FakeLLM only. They check that the prompt wording follows the
papers, that missing fields raise `MissingPersonField` naming the field, that seeded selection is
reproducible, that the target answer does not leak, and that the bridges work. A FakeLLM that follows the
shown majority scores 1.0 under homophily, which only checks the plumbing. They report no LLM effect.

## Limitations

- All injectors are one-shot. The multi-step loop (agent answers, state updates, neighbours re-read) belongs
  to the ABM driver.
- `neighbor_activity` has no published template behind it. Its wording is ours.
- The exposure count in `neighbor_activity` and the mixed-answer `distill` line in `asch_majority` are
  extensions beyond the papers.
- `asch_majority` tones use only the phrasings Zhu et al. print in Table 1; their full lists are longer.
- The Chuang et al. effects come from agents that wrote and read one tweet per step over many steps. A single
  prompt with k posts is a weaker dose.
