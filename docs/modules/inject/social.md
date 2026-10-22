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
| `state` | str, or `{qid: value}`: the person's prior view (last wave, last ABM step) | `peer_messages`, `neighbor_activity` |
| `fields`, `text` | persona block, on by default except `asch_majority` | all (`persona=True/False`) |

## Injectors

| name | needs | knobs | paper |
|---|---|---|---|
| `peer_messages` | `neighbors` | `k=5`, `select` (`recent`/`random`), `bias` (`none`/`weak`/`strong`), `closed_world=True`, `persona`, `seed` | Chuang et al. 2023, arXiv:2311.09618 |
| `neighbor_activity` | `neighbors` | `k=10`, `action=` (adds "n of your m contacts have ..."), `select`, `persona`, `seed` | Gao et al. 2023 (S3), arXiv:2307.14984 |
| `asch_majority` | `neighbors` (none with `answer=`) | `tone` (`plain`/`neutral`/`confident`/`uncertain`), `distill`, `answer` + `n`, `k`, `persona=False`, `seed` | Zhu et al. 2024, arXiv:2410.12428 |
| `descriptive_norm` | `norms` (or `neighbors`) | `source` (`norms`/`neighbors`), `group` (may hold `{field}`), `style` (`shares`/`most`), `decimals`, `persona` | Peng et al. 2026, arXiv:2603.20911 |
| `news_feed` | `media` | `k=3`, `context`, `select`, `persona`, `seed` | Williams et al. 2023, arXiv:2307.04986 |

`select="recent"` keeps the last `k` items (sorted by `time` when every item has one). `select="random"`
draws a sample seeded by (`seed`, person id, question id), so a rerun renders the same text.

### `peer_messages`

What the paper did: each agent got "Role play this person." plus a persona with an initial belief about a
topic, and was told it was talking with strangers on Twitter. At each step a random pair met: one agent wrote a
tweet, the other read it and reported its belief, rated on a 5-point scale. Memory was either cumulative (every
experience appended) or reflective (a rolling summary). Two prompt add-ons were tested: confirmation bias,
weak or strong (Appendix F), and a closed-world line, "alone in your room with limited access to the
Internet" (Appendix G).

How this injector differs: it is one-shot. The k most recent (or sampled) neighbour messages appear as a block
of posts already read, like the cumulative memory, and the question follows. The bias lines keep the paper's
opening sentences. In the closed-world line, "XYZ" becomes "this topic" and "tweets" become "posts".

Published effect (Table 1, 15 topics, cumulative memory): without confirmation bias the agents converged on
the factual position. Final mean opinion B was -1.33 under false framing and 0.52 under true framing.
Confirmation bias raised the spread D, the standard deviation of final opinions. Under false framing D went
from 0.60 (none) to 0.87 (weak) and 1.24 (strong); under true framing from 0.66 to 0.95 and 1.52. A one-way
ANOVA gave F(2) = 25.43, p < .001. Source: arXiv:2311.09618, Table 1, HTML version.

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
experiment. There are in total <K> participants.", the organiser's multiple-choice question, one line per
earlier participant ("Participant i: ...") and "Now, as Participant <K>, it's your turn." The tested
conditions were unanimous wrong confederates, 2 to 10 participants, uniformly random "diverse" answers, and
four tones: plain "Kings", neutral "I think it is Kings", confident "I am sure it is Kings" and uncertain
"I am not sure if it's Kings". The paper also proposed Question Distillation, which replaces the participant
lines with "Organiser: All participants before you have chosen <ANSWER> as the answer."

How this injector differs: confederate answers come from the neighbours' real answers to the same question,
in seeded order, so it simulates social influence rather than a wrong-answer test. `answer=` with `n=`
reproduces the paper's unanimous manipulation. `distill=True` uses the paper's line when the answers agree.
When they disagree, it uses a count line of our own ("Of the 3 participants before you, 2 chose ..."). Answers
are option text, not letters, and the reply is strict JSON.

Published effect: all tested models conformed to the majority. Conformity rose with the number of
confederates. Unanimous-plain confederates lowered accuracy, and diverse-plain answers barely changed it. The
neutral and confident tones raised conformity over plain. Instruction tuning lowered conformity. Lower
accuracy went with more conformity, with correlation -0.777 across MMLU subjects for Llama-3-8B-Instruct.
Per-model rates are in figures only. On TriviaQA, Llama-3-8B was sycophantic in 63.2% of cases (base) against
25.2% (instruct). Source: arXiv:2410.12428, sections 3 and 5, Table 3.

### `descriptive_norm`

What the paper did: in a Weibo-like feed, descriptive norms went in the system prompt as approximate
prevalence of each engagement form. The like-dominant condition said about 80% like, 15% repost and 5% quote;
the repost-dominant condition said about 90% repost and 10% quote. The paper calls these "heuristic prevalence
cues rather than strict behavioral targets"; every option stayed available. It also argues that a study needs
an explicit no-norm baseline, because a default prompt is not a blank control.

How this injector differs: the prevalence is over answers to the question, from `person["norms"]` (for
example built with `attach_norms`, which leaves the person's own answer out) or from the neighbours' answers.
`style="shares"` follows the paper's percentage form. `style="most"` ("Most people your age (62%) chose ...")
is the common norm message from the persuasion literature and is not Peng et al.'s wording. The closing line
"You are free to give any answer." stands for "all options stayed available". The `no_persona` control in
`compare_injectors` is the no-norm baseline the paper asks for.

Published effect: against no norm, the like-dominant norm raised the odds of engaging at all (binary logit
B = 0.291, OR 1.338) and the repost-dominant norm lowered them (B = -0.465, OR 0.628). Among engagers, the
repost-dominant norm raised repost against like with OR 89.1. Observed liking under the like-dominant norm was
14.5% to 24.8% of actions, against 4.1% to 9.5% with no norm. Source: arXiv:2603.20911, Tables 2 and 3.

### `news_feed`

What the paper did: each agent was a ChatGPT system message with name, age, Big Five traits, a short bio and
"relevant memories", and was asked whether to stay home all day. In the full-feedback condition the memories
included "[name] checks the newspaper and finds that [X]% of Dewberry Hollow's population caught new infections
of the Catasat virus yesterday." In `simengine`, `prevalence_item` builds that sentence from an ABM share.

How this injector differs: media items are listed as "You read in <source>: <text>" under "Relevant
memories:", with an optional `context` line in front (the paper's scenario sentence). The question and strict
answer line go in the user turn.

Published effect: in fixed-effect logits of the stay-home decision (Table 1, N about 67k agent-days),
societal health information had coefficient 1.59 (regression 2) and 3.97 with a squared term of -0.65
(regression 3). The pseudo R² rose from 0.12 (own health only) to 0.41 and 0.46. Agents self-isolated as
reported cases rose, and that flattened the epidemic curve. Source: arXiv:2307.04986, Table 1 and Results.

## Bridges to `simengine.abm`

- `attach_neighbors(people, edges, directed=False, keys=..., states=None)`: edges are `[(a, b)]` or
  `{a: [b, ...]}`. Each neighbour carries `label`, `answers`, `message`, `action` and `time` from its own
  person dict, updated with `states[id]`, for example the agent's action at the current ABM step. Self-loops and
  unknown ids are dropped, and the inputs are not changed.
- `attach_norms(people, questions, by=None)`: counts of the other people's answers (or their mean for numeric
  questions) within the same `by` group. The person's own answer never enters their own norm.
- `norm_from_dist(dist, options)`: a K-category distribution, such as `abm.opinion.distribution` output, as
  `{option: share}` for `person["norms"][qid]`.
- `prevalence_item(share, population, event)`: an adoption or epidemic share as a Williams-style news item.

## Leakage

No injector here reads `person["answers"]`. A neighbour whose id equals the person's id is dropped, so a
self-loop cannot show the target answer. `state`, `norms`, `neighbors`, `media` and `group` never appear as
persona attributes. What the caller must keep honest is the following:
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
- The Chuang et al. effects come from agents that wrote and read one tweet per step over many steps. A single
  prompt with k posts is a weaker dose.
