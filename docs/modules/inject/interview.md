# `persona.inject` family: interview

`simengine/persona/inject/interview.py`. Injectors that ground the persona in the person's own interview
transcript, following Park et al. 2024, *Generative Agent Simulations of 1,000 People*, retitled in v3 (June 2026)
*LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals* (arXiv:2411.10109).
In the paper, 1,052 US adults each gave a two-hour, voice-to-voice interview to an AI interviewer that followed the
American Voices Project protocol (6,491 words per transcript on average). Each agent was a GPT-4o prompt holding
the full transcript.

```python
import simengine.persona.inject as pi
from simengine.persona.inject import interview as iv

people = iv.prepare(llm, people)               # offline LLM calls: interview_summary + expert_reflections
questions = iv.route_experts(llm, questions)   # offline: question["expert"], the paper's router
p = pi.get_injector("interview_expert_reflection")(people[0], questions[0])
llm.complete(p.system, p.flatten(), max_tokens=1024)
pi.compare_injectors(llm, people, questions, "answers",
                     ["interview_transcript", "interview_expert_reflection", "interview_summary",
                      pi.get_injector("interview_lesion", drop=0.8)], max_tokens=1024, dry_run=True)
```

## Injectors

All five put the persona material in the system prompt, followed by the paper's `=====` separator. The user turn
holds the paper's prediction task ("What you see above is an interview transcript. Based on the interview
transcript, I want you to predict the participant's survey responses ..."), then "Here are the questions:", the
question, and the strict JSON answer line that `parse_option` reads.

With `cot=True` (the default), the prompt adds the paper's step list. Choice questions get four steps: Option
Interpretation, Option Choice, Reasoning and Response. Numeric questions get three: Range Interpretation,
Reasoning and Response. The paper reuses the choice task line for numeric items; the "The question asks for a
number." wording is ours. The reply must end with a JSON line, so call with `max_tokens >= 1024`. With
`compare_injectors`' default of 256 tokens, the reasoning gets cut off before the JSON and the reply scores as
unparsed. `cot=False` keeps the task wording without the steps. Use it when comparing with families that do not
reason, so that the CoT effect is not mixed into the injection effect.

| name | needs | what it renders | paper variant |
|---|---|---|---|
| `interview_transcript` | `interview` | The transcript. `budget_tokens` truncates it by whole exchanges | interview agent (without reflections) |
| `interview_lesion` | `interview` | The transcript with `round(drop*n)` exchanges removed at random (default `drop=0.8`), with at least one kept | random-lesion agents (0, 20, 40, 60, 80 %) |
| `interview_summary` | `interview_summary` | A `- key: value` dictionary of facts in place of the transcript | summary agents |
| `interview_expert_reflection` | `interview`, `expert_reflections` | The transcript plus the notes of the expert who fits the question | the paper's full agent architecture |
| `interview_plus_survey` | `interview`, `answers` | The transcript plus the person's other answers through `known_answers`; `expert="auto"` (or `"all"`, a name) also appends expert notes and then needs `expert_reflections` | survey-plus-interview agents (v3) |

Common params: `cot=True`, `budget_tokens=None`, `truncate="head_tail"`, `seed=0`.

**Person fields.**
- `interview` is a `str` or `[(question, answer)]`. A string is split into exchanges on blank lines, or on lines
  when it has no blank line.
- `interview_summary` is `{key: fact}`.
- `expert_reflections` is `{expert: [note, ...]}`. It is a separate key from the memory family's `reflections`
  list. A non-dict value raises `TypeError`.

Both derived fields are dicts, so `plain_fields` never prints them.

**Expert reflection.** In the paper, GPT-4o took the transcript and wrote between 5 and 20 notes in each of four
expert personas: psychologist, behavioral economist, political scientist and demographer. This happened once per
person. At answer time, an LLM picked the one expert best suited to the question, and that expert's notes were
appended to the transcript.

Here, `reflect_experts` and `prepare` make the notes offline. `REFLECTION_PROMPT` quotes the paper's demographer
prompt ("Imagine you are an expert demographer (with a PhD) taking notes while observing this interview ..."). The
paper gives no focus wording for the other three experts, so `EXPERT_FOCUS` supplies our own, written in the same
form.

`render` must not call an LLM, so routing uses `question["expert"]` when it is set. `route_experts` fills that key
with the paper's LLM router. When the key is missing, a keyword rule chooses the expert (`keyword_expert`, with
psychologist as the fallback).

Expert options:
- `expert="all"` appends all four sets of notes.
- `expert="<name>"` fixes one expert.
- `transcript=False` sends the notes alone, as a compressed variant that the paper did not run.

If the routed expert has no notes (a missing key or an empty list), the injector uses the notes the person has
and sets `meta["expert_fallback"]`.

In v3, every agent type, the survey-plus-interview agents included, appended the routed expert's reflections, written
over that agent's self-report data. `interview_plus_survey` leaves them out by default; `expert="auto"` gives the v3
agent, with notes made from the transcript alone (`prepare` does not see the survey answers).

**Summary.** The paper asked GPT-4o to turn the transcript into "bullet-pointed dictionaries of key response pairs"
(for example `{"childhood_town": "Small town", "siblings": "Only child", ...}`). This keeps the facts and drops the
interviewee's wording. `summarize_interview` reproduces this with our prompt wording, because the paper does not
quote its summary prompt.

**Truncation** (`fit_budget`). The budget is `budget_tokens * 4` characters, the same rough rule as
`compare.estimate_cost`. Exchanges are kept whole. Strategies:
- `head`: the earliest exchanges.
- `tail`: the most recent exchanges. The paper's AI interviewer kept its reflection notes plus the last 5,000
  characters of the conversation. That was the interviewer's context, not the agent's, so
  `interview_expert_reflection(truncate="tail")` is an adaptation of it.
- `head_tail`: the default. Half the budget from each end, with an `[... n interview exchange(s) omitted ...]`
  marker between.
- `random`: a seeded random subset in the original order, which is a lesion sized to the budget.

If not even one exchange fits, the first exchange kept is cut to the budget. The paper itself never truncated: it
fit the full transcript into GPT-4o's context.

**Determinism.** `lesion` and `random` use `random.Random(f"{seed}:{person_id}")`. The same person therefore gets the
same lesion for every question, which matches the paper (one shortened interview per agent), while different
people get different lesions.

**Leakage.** A transcript is not the person's answer list, so `known_answers` does not apply to it. In
`interview_plus_survey`, the shown answers always go through `known_answers`, which drops the target and
`question["exclude"]`. By default only the target is dropped, which is v3's robustness variant (0.85 below). For
its main survey-plus-interview result, v3 removed from the input every GSS item it evaluated on, and for the Big
Five the whole block of the predicted item's trait. To reproduce that, pass those qids in `exclude`. (v1's
robustness agents excluded "data from the same category as the question being predicted".)

The interview can still contain the answer in plain words. v3 screened all GSS-item and interview-question pairs
for duplicate or near-duplicate wording and removed the 27 GSS items it flagged from the evaluation set; simengine
does no such screening, so drop such items from `questions` yourself. v3 also names direct answer retrieval from
the transcript as one mechanism behind the gain: on a 59-agent subset, interview agents lose accuracy as the items
most likely to have a verbatim answer are removed, and still reach 0.79 against 0.71 for demographic agents with
80 such items dropped. When a held-out item is asked in the interview, the score measures retrieval, not
simulation.

## Published effect sizes (Park et al., GSS normalized accuracy = agent accuracy / participant two-week test-retest)

| condition | v1 (Nov 2024) | v3 (Jun 2026) |
|---|---|---|
| interview (transcript + expert reflection) | 0.85 (raw 68.85 %) | 0.83 (raw 65.67 %) |
| survey-plus-interview (all evaluated GSS items held out) | n/a | 0.86 |
| survey-plus-interview, only the target held out (subsample) | n/a | 0.85 |
| survey only (GSS + Big Five) | n/a | 0.82 |
| demographics only | 0.71 | 0.74 |
| persona paragraph (self-written) | 0.70 | 0.71 |
| summary agents (100-person subsample) | 0.83 | 0.81 |
| random lesion, 80 % removed (subsample) | 0.79 | 0.79 |
| interview on that subsample (reference) | 0.85 | 0.82 |

Source: v1 main text and SI Table 6; v3 main text and SI robustness section. Big Five normalized correlation for
interview agents is 0.80, against 0.55 (v1) or 0.61 (v3) for demographic agents and 0.75 for persona agents.

For economic games, interview agents reached a normalized correlation of 0.66. v3 reports no significant difference
between agent types on that construct (F(4, 5255) = 1.63, p = 0.16).

For the lesion runs, v1 Table 6 lists GSS normalized accuracy of 0.85, 0.84, 0.83, 0.83 and 0.79 at 0, 20, 40, 60
and 80 % removed. That is roughly linear.

All results come from GPT-4o with the paper's chain of thought. Neither version reports reflection against no
reflection on its own. The nearest is v3's fine-tuning check on 552 agents: GPT-4o with reasoning and reflection
reached 0.84, and the same model with both removed from the prompt 0.82 (roughly `interview_transcript(cot=False)`).

**In simengine:** not yet measured. Only FakeLLM tests exist so far (`tests/test_persona_inject_interview.py`),
and no real LLM run has been made.

## Limitations

- simengine ships no interview data. The paper's transcripts are access-controlled, and the person dicts must
  bring their own.
- `prepare` costs 1 + 4 calls per person, each with the whole transcript as input. Run it once and store the
  result.
- The summary prompt and the focus wording for three of the experts are ours, as is the keyword router.
- The paper's numbers rest on two-hour interviews. Short or synthetic transcripts are a different regime.
