# `simengine.persona.inject.framing`

The persona content stays the same (`person["fields"]`). What changes is where it goes, which grammatical person
it is written in, and how it is laid out. These are three orthogonal knobs plus attribute order. Each has a
fixed set of values, so a framing ablation is one `grid()` call.

```python
import simengine.persona.inject as pi
from simengine.persona.inject.framing import grid

pi.get_injector("framed", placement="turns", voice="first", format="prose")(person, question)
arms = grid(placements=("system", "user", "turns"), voices=("second", "first", "third"), formats=("bullets", "prose"))
plan = pi.compare_injectors(llm, people, questions, "answers", arms + ["argyle_backstory"], dry_run=True)
```

| knob | values | effect |
|---|---|---|
| `placement` | `system`, `user`, `turns` | system prompt; prefix in the user turn before the question; an earlier user/assistant exchange (`Prompt.history`) |
| `voice` | `second`, `first`, `third`, `interview` | "You are ... / Your age is 34."; "My age is 34."; "Imagine a person ... / Their age is 34."; interviewee roleplay (the persona is the interviewee's earlier answers and the question is the last interviewer turn) |
| `format` | `bullets`, `prose`, `json`, `qa` | `- age: 34` lines; sentences; one JSON object; question/answer pairs, one per attribute |
| `order` | `given`, `sorted`, `reversed`, `shuffle` | `shuffle` is seeded per (`seed`, person id), so a rerun renders the same text |

Other params: `keys=` (select and order attributes; duplicates count once), `header=` (replace the opening line),
`header_sep=` (between the header and the attribute block, default a newline), `closing=` (the line after the
block, default "Answer as this person would." or its voice variant), `lead=` (text put before the question
itself) and `phrases=` (`{attr: "In terms of my age, I am {value}."}`, per-attribute sentence templates used
wherever a sentence is rendered; only `{value}` is substituted). An unknown param raises `TypeError`, so a
typo cannot produce a second arm that is identical to the default.

`placement="turns"` works by voice:
- `second` / `third`: the user turn gives the persona and the assistant replies "Understood. I will answer as
  this person."
- `first`: the user asks "tell me about yourself" and the assistant tells its own story.
- `interview` + `qa`: one interviewer/interviewee exchange per attribute.

`compare_injectors` flattens the history into the user text unless the LLM sets `supports_history`. Today this
means `turns` differs from `user` only in its wording, not in the message structure. See
[persona-inject.md](../persona-inject.md).

Leakage: the family reads only `fields`, never `answers`. A field keyed by the target question id or one of its
`exclude` ids is dropped, so a prep step that copied the label into `fields` cannot reach the prompt.

Needs: `("fields",)`, as a dict. A person without it, or one whose selected values are all empty, raises
`MissingPersonField`. As in Argyle et al., an empty value (`None`, `""`, NaN, `pd.NA`, `NaT`, an empty list)
is omitted, not printed.

## Injectors

| name | defaults | paper |
|---|---|---|
| `framed` | system, second, bullets | the grid itself; Tseng et al. 2024, arXiv:2406.01171 (survey of role-playing and personalization personas) |
| `argyle_backstory` | user, first, prose | Argyle et al. 2023, arXiv:2209.06899 |
| `hu_collier_profile` | system, second, prose, header "Your Profile:" on the same line, no closing | Hu & Collier 2024, arXiv:2402.10811 |
| `lutz_third_person` | user, third, prose, header "Think of a person with the following profile.", lead "How would this person answer the following question:" | Lutz et al. 2025, arXiv:2507.16076 |
| `lutz_interview` | user, interview, qa, lead "How would you answer the following question:" | Lutz et al. 2025, arXiv:2507.16076 |

Every preset accepts all the knobs, so `get_injector("argyle_backstory", order="shuffle", seed=1)` is a valid arm.

### `argyle_backstory`

What the paper did: for each ANES respondent, the authors built a first-person backstory from up to 10 survey
variables, with one template fragment per variable ("Ideologically, I describe myself as conservative.
Politically, I am a strong Republican. Racially, I am white. ..."). A fragment was omitted when its variable was
missing. The backstory was the prefix of a GPT-3 completion prompt, and the vote was read from next-token
probabilities.

How this injector differs: it uses a chat setup with a strict JSON answer, and the generic sentence is "My
<attr> is <value>." (`phrases=` restores per-variable wording). It also adds a closing line, "Answer as the
person described above would.", which a completion model did not need.

Published effect (Study 2, GPT-3 vote against self-reported vote): tetrachoric correlation 0.90 (2012), 0.92
(2016) and 0.94 (2020) across all respondents. Source: arXiv:2209.06899, Table 1 and text.

### `hu_collier_profile`

What the paper did: the persona was written as "Your Profile:" followed on the same line by second-person
declarative sentences, one per annotator variable (Kumar et al. template: "Your Profile: You race is %s. Your
gender is %s. ..."; ANES: "It is 2012. Your Profile: Racially, you are %s. ..."). For Llama-2 base models it was
prepended to the prompt. For chat or tuned models (Llama-2-chat, Tulu-2) it went in the system variable, and
the reply was read from the next token after "Answer: (". The paper also tried 5 attribute orders and 5
paraphrases (Appendix B.1, C).

How this injector differs: the reply is a strict JSON answer, not a next-token option letter, and the generic
sentence is "Your <attr> is <value>." (`phrases=` restores per-variable wording such as "Racially, you are
...").

Published effect: persona variables explain under 10% of the annotation variance in the datasets studied.
Persona prompting gives "modest but statistically significant" gains, and a 70B model with persona prompting
captures 81% of the variance that a linear regression on the ground truth captures. Source: arXiv:2402.10811,
abstract (the 81% is Tulu-2-70b-dpo). Order and paraphrase sensitivity for Llama-2-70b-chat over the 10
variants: R² 0.09 to 0.12 on Kumar et al. (default 0.10) and 0.27 to 0.39 on ANES (default 0.30). Source:
arXiv:2402.10811, Table 6 (HTML).

### `lutz_third_person`, `lutz_interview`

What the paper did: it compared three role-adoption formats. All of them go in the user prompt, not the
system prompt (Appendix B, Table 3):
- direct: "You are a [R] [G1] who lives in the U.S. [TASK]"
- third person: "Think of a [R] [G1] who lives in the U.S. [TASK]", with the task asked about that person
  ("How would [P1] answer the following question: ...", Table 4)
- interview: one user message that opens with "You are an interviewee. Based on your previous answers, provide
  an answer to the last question.", then "Interviewer: Do you identify as ...? / Interviewee: I identify as a
  ...", one exchange per attribute plus "Where do you live? / I live in the U.S.", then "Interviewer: [TASK]"
  and an open "Interviewee:" line. The task is "How would you answer the following question: ..."

It crossed these with three demographic primings: explicit, name and structured.

How these injectors differ: the attributes are generic fields ("What is your age? / My age is 34.") instead of
the paper's gender and race items, and the US-residence clause and the name priming are not added. The prompt
ends with the strict JSON answer line instead of an open "Interviewee:" line. `lutz_interview` keeps the
paper's layout of a single user prompt; `placement="turns"` splits the exchanges into `Prompt.history`, which is
a variant the paper did not test. This module's grid covers the direct format as
`framed(placement=user, voice=second, format=prose)`.

Published effect: the paper fit an OLS model on the standard deviation of marked-word counts across
demographic groups (lower means less stereotyping), against explicit priming with the direct format. The
interview coefficient was -2.6 to -10.0, significant in all 5 models. The third-person coefficient was -0.9 to
+0.8, not significant in any model. The interview format also lowered opinion distance on OpinionsQA for one
model (OLMo-2-7B, Figure 3). Source: arXiv:2507.16076 v2, Table 1 (HTML).

### `framed`

The grid. The Tseng et al. survey covers persona assignment broadly but gives no single framing effect.
Related placement evidence: Zheng et al. 2024 (arXiv:2311.10054) found that system-prompt personas do not
improve accuracy on 2,410 factual questions. In the same work, audience framing ("You are talking to a ...")
beat speaker framing by a small, statistically significant margin. That was a task-accuracy setting, not a
simulation of individuals.

## Measured in simengine

Not yet measured in simengine. The tests use FakeLLM only and check that every arm renders, hides the target
and keeps the same text across reruns. They report no LLM effect.

## Limitations

- `turns` reaches the model as flattened text until a backend sets `supports_history`.
- The generic sentence "Your <attr> is <value>." is clumsy for some attributes. Pass `phrases=` for
  publication-grade wording.
- `interview` with a non-qa format puts the block under an interviewee instruction. This is a grid cell
  without a direct paper counterpart.
