"""Interview-transcript injectors (Park et al. 2024, arXiv:2411.10109): the person's interview in context, with
expert reflections, bullet-dictionary summaries, random lesions and token-budget truncation.

The paper's agents put the whole ~2-hour transcript (6,491 words on average) into one prompt, append the reflections
of one social-science expert chosen for the question, and ask for a 4-step chain of thought before the answer.
Reflections and summaries are LLM-made once per person, so they are built offline (`prepare`) and stored on the
person dict; `render` stays pure.

    import simengine.persona.inject as pi
    from simengine.persona.inject import interview as iv
    person = {"id": "u1", "interview": [("Tell me about your life.", "I grew up in a small town ..."), ...]}
    people = iv.prepare(llm, [person])                 # adds interview_summary + reflections (LLM calls)
    questions = iv.route_experts(llm, questions)       # optional: question["expert"] as in the paper
    p = pi.get_injector("interview_transcript", budget_tokens=8000)(people[0], question)
    res = pi.compare_injectors(llm, people, questions, "answers",
                               ["interview_transcript", "interview_expert_reflection", "interview_summary",
                                pi.get_injector("interview_lesion", drop=0.8)], max_tokens=1024)

Person fields: `interview` (str or [(question, answer)]), `interview_summary` ({key: value}, from
`summarize_interview`), `reflections` ({expert: [note, ...]}, from `expert_reflections`), `answers` (only
interview_plus_survey, through `known_answers`). Dict-valued so `plain_fields` never prints them.

cot=True (the paper's prompt) needs room for the reasoning: call with max_tokens >= 1024, or a truncated reply
loses its final JSON line and scores as unparsed.
"""
from __future__ import annotations

import random
import re
from typing import Any, Mapping, Sequence

from ...games._answer import instruction
from ...llm.jsonx import json_extract
from ._base import Injector, Prompt, answer_spec, register

PAPER = "Park et al. 2024, arXiv:2411.10109"
EXPERTS = ("psychologist", "behavioral economist", "political scientist", "demographer")
# What each expert writes about. Only the demographer line is quoted from the paper (SM 3); the others follow it.
EXPERT_FOCUS = {
    "psychologist": "the interviewee's personality, values, motivations and emotional patterns",
    "behavioral economist": "the interviewee's economic decision-making: money, risk, time, fairness and trust",
    "political scientist": "the interviewee's political views, ideology, party identification and civic attitudes",
    "demographer": "the interviewee's demographic traits and social status",
}
# Offline stand-in for the paper's LLM router (see route_expert); first match wins, order matters.
EXPERT_KEYWORDS = {
    "political scientist": ("government", "president", "party", "vote", "election", "liberal", "conservative",
                            "democrat", "republican", "police", "law", "court", "spending", "tax", "immigra",
                            "abortion", "rights", "policy", "military", "congress"),
    "behavioral economist": ("money", "dollar", "income", "pay", "price", "invest", "save", "spend", "risk",
                             "bet", "gamble", "lottery", "trust", "share", "split", "give", "game", "job"),
    "demographer": ("age", "born", "married", "children", "household", "education", "degree", "school",
                    "religion", "race", "region", "city", "live", "work", "occupation", "hours"),
    "psychologist": (),
}
CHARS_PER_TOKEN = 4                      # same rough rule as compare.estimate_cost
TRUNCATE = ("head", "tail", "head_tail", "random")
OMIT = "[... {n} interview exchange(s) omitted ...]"

TASK_MC = (
    "Task: What you see above is an interview transcript. Based on the interview transcript, I want you to predict "
    "the participant's survey responses. All questions are multiple choice, and you must guess from one of the "
    "options presented.")
STEPS_MC = (
    "As you answer, I want you to take the following steps:\n"
    "Step 1) Describe in a few sentences the kind of person that would choose each of the response options. "
    "(\"Option Interpretation\")\n"
    "Step 2) For each response option, reason about why the Participant might answer with that particular option. "
    "(\"Option Choice\")\n"
    "Step 3) Write a few sentences reasoning on which of the options best predicts the participant's response. "
    "(\"Reasoning\")\n"
    "Step 4) Predict how the participant will actually respond in the survey. Predict based on the interview and "
    "your thoughts. (\"Response\")")
TASK_NUM = (
    "Task: What you see above is an interview transcript. Based on the interview transcript, I want you to predict "
    "the participant's survey responses. The question asks for a number.")
STEPS_NUM = (
    "As you answer, I want you to take the following steps:\n"
    "Step 1) Describe in a few sentences the kind of person that would choose each end of the range. "
    "(\"Range Interpretation\")\n"
    "Step 2) Write a few sentences reasoning on which option best predicts the participant's response. "
    "(\"Reasoning\")\n"
    "Step 3) Predict how the participant will actually respond. Predict based on the interview and your thoughts. "
    "(\"Response\")")


# ---------------------------------------------------------------- transcript handling

def transcript_units(interview: Any) -> list[str]:
    """Interview as a list of exchanges. [(q, a)] -> "Interviewer: q\\nParticipant: a"; a string is split on
    blank lines (one exchange per paragraph), or on lines when it has no blank line."""
    if isinstance(interview, str):
        parts = [p.strip() for p in re.split(r"\n\s*\n", interview.strip())]
        if len(parts) == 1:
            parts = [ln.strip() for ln in interview.splitlines()]
        return [p for p in parts if p]
    units = []
    for item in interview or ():
        if isinstance(item, str):
            units.append(item.strip())
            continue
        q, a = item
        units.append(f"Interviewer: {str(q).strip()}\nParticipant: {str(a).strip()}")
    return [u for u in units if u]


def _join(units: Sequence[str]) -> str:
    return "\n\n".join(units)


def fit_budget(units: Sequence[str], max_chars: int | None, how: str = "head_tail", seed: Any = 0) -> list[str]:
    """Keep whole exchanges until `max_chars` (joined with blank lines) is reached.
    head: earliest first. tail: most recent (the paper's AI interviewer kept the last 5,000 characters).
    head_tail: half the budget from each end, with an omission marker between. random: a seeded random subset,
    in original order (a random lesion sized to the budget). If not even one exchange fits, the first kept one
    is cut to the budget."""
    units = list(units)
    if how not in TRUNCATE:
        raise ValueError(f"truncate must be one of {TRUNCATE}, not {how!r}")
    if max_chars is None or len(_join(units)) <= max_chars:
        return units
    if max_chars <= 0:
        raise ValueError("budget must be positive")

    def take(order: Sequence[int], budget: int) -> list[int]:
        kept, used = [], 0
        for i in order:
            cost = len(units[i]) + (2 if kept else 0)
            if used + cost > budget:
                if how != "random":
                    break
                continue
            kept.append(i)
            used += cost
        return kept

    n = len(units)
    if how == "head":
        keep = take(range(n), max_chars)
    elif how == "tail":
        keep = take(range(n - 1, -1, -1), max_chars)
    elif how == "random":
        order = list(range(n))
        random.Random(str(seed)).shuffle(order)
        keep = take(order, max_chars)
    else:
        room = max_chars - len(OMIT.format(n=n)) - 4
        front = take(range(n), room // 2)
        back = take([i for i in range(n - 1, -1, -1) if i not in front], room - room // 2)
        keep = front + back
        if keep:
            keep = sorted(keep)
            out, prev = [], -1
            for i in keep:
                if i != prev + 1:
                    out.append(OMIT.format(n=i - prev - 1))
                out.append(units[i])
                prev = i
            if prev != n - 1:
                out.append(OMIT.format(n=n - 1 - prev))
            return out
    if not keep:
        first = units[n - 1] if how == "tail" else units[0]
        return [first[:max_chars] if how != "tail" else first[-max_chars:]]
    return [units[i] for i in sorted(keep)]


def lesion(units: Sequence[str], drop: float, seed: Any = 0) -> list[str]:
    """Randomly remove round(drop * n) exchanges (at least one is kept), order preserved: the paper's random-lesion
    agents (0/20/40/60/80% of question-response pairs removed)."""
    if not 0 <= drop < 1:
        raise ValueError("drop must be in [0, 1)")
    n = len(units)
    k = min(int(round(drop * n)), max(n - 1, 0))
    gone = set(random.Random(str(seed)).sample(range(n), k))
    return [u for i, u in enumerate(units) if i not in gone]


def keyword_expert(question: dict, default: str = "psychologist") -> str:
    """Deterministic expert for a question: question['expert'] if set, else the first EXPERT_KEYWORDS match."""
    if question.get("expert"):
        return _expert_name(question["expert"])
    words = re.findall(r"[a-z]+", str(question.get("text", "")).lower())
    for expert, keys in EXPERT_KEYWORDS.items():
        if any(w.startswith(k) for w in words for k in keys):
            return expert
    return default


def _expert_name(x: str) -> str:
    s = str(x).strip().lower().replace("_", " ")
    for e in EXPERTS:
        if s == e or s.replace(" ", "") == e.replace(" ", "") or s == e.split()[-1]:
            return e
    raise ValueError(f"unknown expert {x!r}; choose from {EXPERTS}")


# ---------------------------------------------------------------- shared rendering

class _InterviewBase(Injector):
    family = "interview"
    paper = PAPER

    def __init__(self, cot: bool = True, budget_tokens: int | None = None, truncate: str = "head_tail",
                 seed: int = 0, **params: Any):
        if truncate not in TRUNCATE:
            raise ValueError(f"truncate must be one of {TRUNCATE}, not {truncate!r}")
        if budget_tokens is not None and budget_tokens <= 0:
            raise ValueError("budget_tokens must be positive")
        super().__init__(cot=cot, budget_tokens=budget_tokens, truncate=truncate, seed=seed, **params)
        self.cot, self.budget_tokens, self.truncate, self.seed = cot, budget_tokens, truncate, seed

    def transcript(self, person: dict, units: Sequence[str] | None = None) -> str:
        units = transcript_units(person["interview"]) if units is None else units
        cap = None if self.budget_tokens is None else self.budget_tokens * CHARS_PER_TOKEN
        return _join(fit_budget(units, cap, self.truncate, seed=f"{self.seed}:{person.get('id')}"))

    def task_block(self, question: dict) -> str:
        """The paper's prediction prompt around one question, ending in the strict JSON line parse_option reads."""
        mc = bool(question.get("options"))
        line = instruction(answer_spec(question))
        if not self.cot:
            return f"{TASK_MC if mc else TASK_NUM}\n\nHere are the questions:\n\n{question['text']}\n\n{line}"
        line = line.replace("Reply with JSON only:", "After your steps, end your reply with one line of JSON:")
        return (f"{TASK_MC if mc else TASK_NUM}\n\n{STEPS_MC if mc else STEPS_NUM}\n\n"
                f"Here are the questions:\n\n{question['text']}\n\n{line}")

    def prompt(self, system: str, question: dict, **meta: Any) -> Prompt:
        meta["system_chars"] = len(system)
        return Prompt(system=system + "\n\n=====", user=self.task_block(question), meta=meta)


# ---------------------------------------------------------------- injectors

@register
class InterviewTranscript(_InterviewBase):
    """Full transcript in context (the paper's interview agent without reflections); budget_tokens truncates."""
    name = "interview_transcript"
    needs = ("interview",)
    description = ("Interview transcript in the system prompt + the paper's 4-step chain-of-thought prediction "
                   "task; budget_tokens (chars/4) truncates by whole exchanges (head, tail, head_tail, random).")

    def render(self, person: dict, question: dict) -> Prompt:
        return self.prompt(self.transcript(person), question)


@register
class InterviewLesion(_InterviewBase):
    """Random-lesion agent: a seeded random fraction of exchanges removed (per person, fixed across questions)."""
    name = "interview_lesion"
    needs = ("interview",)
    description = ("Transcript with round(drop*n) exchanges randomly removed (paper: 0.2-0.8), seeded per person; "
                   "then the same prompt as interview_transcript.")

    def __init__(self, drop: float = 0.8, **params: Any):
        if not 0 <= drop < 1:
            raise ValueError("drop must be in [0, 1)")
        super().__init__(drop=drop, **params)
        self.drop = drop

    def render(self, person: dict, question: dict) -> Prompt:
        units = transcript_units(person["interview"])
        kept = lesion(units, self.drop, seed=f"{self.seed}:{person.get('id')}")
        return self.prompt(self.transcript(person, kept), question, kept=len(kept), of=len(units))


@register
class InterviewSummary(_InterviewBase):
    """Summary agent: the transcript replaced by a bullet dictionary of key facts (no linguistic features)."""
    name = "interview_summary"
    needs = ("interview_summary",)
    description = ("Bullet-pointed dictionary of key facts from the interview (person['interview_summary'], built "
                   "once by summarize_interview) in place of the transcript.")

    def render(self, person: dict, question: dict) -> Prompt:
        s = person["interview_summary"]
        body = ("\n".join(f"- {str(k).replace('_', ' ')}: {v}" for k, v in s.items())
                if isinstance(s, Mapping) else str(s).strip())
        return self.prompt("Summary of the participant's interview:\n" + body, question)


@register
class InterviewExpertReflection(_InterviewBase):
    """The paper's full agent: transcript + the reflections of the expert best suited to the question."""
    name = "interview_expert_reflection"
    needs = ("interview", "reflections")
    description = ("Transcript + one expert's reflections (psychologist, behavioral economist, political scientist, "
                   "demographer). expert='auto' uses question['expert'] (route_experts) or a keyword rule; 'all' "
                   "appends all four; transcript=False keeps the reflections only (compressed).")

    def __init__(self, expert: str = "auto", transcript: bool = True, **params: Any):
        if expert not in ("auto", "all"):
            expert = _expert_name(expert)
        super().__init__(expert=expert, transcript=transcript, **params)
        self.expert, self.with_transcript = expert, transcript

    def render(self, person: dict, question: dict) -> Prompt:
        refl = {_expert_name(k): v for k, v in person["reflections"].items()}
        chosen = list(refl) if self.expert == "all" else [
            keyword_expert(question) if self.expert == "auto" else self.expert]
        fallback = not any(e in refl for e in chosen)
        if fallback:                      # no notes from the routed expert: use what the person has
            chosen = list(refl)
        blocks = []
        for e in [e for e in EXPERTS if e in chosen]:
            notes = refl[e]
            notes = [notes] if isinstance(notes, str) else list(notes)
            blocks.append(f"Reflections of an expert {e}:\n" + "\n".join(f"- {str(n).strip()}" for n in notes))
        parts = ([self.transcript(person)] if self.with_transcript else []) + blocks
        return self.prompt(_join(parts), question, experts=[e for e in EXPERTS if e in chosen],
                           expert_fallback=fallback)


@register
class InterviewPlusSurvey(_InterviewBase):
    """Survey-plus-interview agent (v3 of the paper): transcript + the person's other survey answers, target and
    question['exclude'] removed through known_answers (the paper excluded the item's whole category)."""
    name = "interview_plus_survey"
    needs = ("interview", "answers")
    description = ("Transcript + the person's other survey answers (known_answers: target and exclude dropped); "
                   "items={qid: question text} labels them, else the qid is shown.")

    def __init__(self, items: Mapping[str, str] | None = None, **params: Any):
        super().__init__(**({"items": dict(items)} if items else {}), **params)
        self.items = {str(k): str(v) for k, v in (items or {}).items()}

    @property
    def label(self) -> str:
        if "items" not in self.params:
            return super().label
        p = {**self.params, "items": len(self.params["items"])}   # keep arm labels short
        return f"{self.name}({','.join(f'{k}={v}' for k, v in sorted(p.items()))})"

    def render(self, person: dict, question: dict) -> Prompt:
        known = self.known_answers(person, question)
        lines = [f"- {self.items.get(str(k), k)}: {v}" for k, v in known.items()]
        survey = "The participant's answers to other survey questions:\n" + ("\n".join(lines) or "(none)")
        return self.prompt(_join([self.transcript(person), survey]), question, n_answers=len(known))


# ---------------------------------------------------------------- offline builders (LLM calls, run once)

SUMMARY_PROMPT = (
    "Convert the interview transcript above into a JSON dictionary of key facts about the interviewee, one "
    "short key per topic and a short factual value (e.g. {\"childhood_town\": \"Small town\", \"siblings\": "
    "\"Only child\"}). Keep the factual content, drop the interviewee's wording. Reply with the JSON object only.")
REFLECTION_PROMPT = (
    "Imagine you are an expert {expert} (with a PhD) taking notes while observing this interview. Write "
    "observations/reflections about {focus}. (You should make more than 5 observations and fewer than 20. Choose "
    "the number that makes sense given the depth of the interview content above.) One observation per line.")
ROUTE_PROMPT = (
    "Which domain expert would best answer the survey question below: {experts}? Reply with the expert's name only."
    "\n\nQuestion: {question}")


def _source(person: dict, budget_tokens: int | None) -> str:
    units = transcript_units(person["interview"])
    cap = None if budget_tokens is None else budget_tokens * CHARS_PER_TOKEN
    return _join(fit_budget(units, cap, "head_tail"))


def summary_prompt(person: dict, budget_tokens: int | None = None) -> tuple[str, str]:
    return _source(person, budget_tokens) + "\n\n=====", SUMMARY_PROMPT


def reflection_prompt(person: dict, expert: str, budget_tokens: int | None = None) -> tuple[str, str]:
    e = _expert_name(expert)
    return _source(person, budget_tokens) + "\n\n=====", REFLECTION_PROMPT.format(expert=e, focus=EXPERT_FOCUS[e])


def parse_summary(text: str) -> dict:
    """JSON object from the reply; else 'key: value' / bullet lines; else {'summary': text}."""
    v = json_extract(text, None)
    if isinstance(v, dict) and v:
        return {str(k): v[k] for k in v}
    out = {}
    for ln in str(text or "").splitlines():
        m = re.match(r"^\s*(?:[-*•]|\d+[.)])?\s*([^:]{1,60}):\s*(.+)$", ln)
        if m:
            out[m.group(1).strip().lower().replace(" ", "_")] = m.group(2).strip()
    return out or ({"summary": str(text).strip()} if str(text or "").strip() else {})


def parse_notes(text: str) -> list[str]:
    """One note per non-empty line, bullets and numbering stripped."""
    notes = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in str(text or "").splitlines()]
    return [n for n in notes if n]


def summarize_interview(llm, person: dict, *, budget_tokens: int | None = None, max_tokens: int = 2048,
                        **kw: Any) -> dict:
    """The paper's summary-agent input: one LLM call -> {key: fact}."""
    s, u = summary_prompt(person, budget_tokens)
    return parse_summary(llm.complete(s, u, max_tokens=max_tokens, **kw).text)


def expert_reflections(llm, person: dict, experts: Sequence[str] = EXPERTS, *, budget_tokens: int | None = None,
                       max_tokens: int = 2048, **kw: Any) -> dict[str, list[str]]:
    """One call per expert -> {expert: [note, ...]} (paper: 4 experts, up to 20 notes each, made once)."""
    exps = [_expert_name(e) for e in experts]
    rs = llm.map([reflection_prompt(person, e, budget_tokens) for e in exps], max_tokens=max_tokens, **kw)
    return {e: parse_notes(r.text) for e, r in zip(exps, rs)}


def route_expert(llm, question: dict, *, max_tokens: int = 16, **kw: Any) -> str:
    """The paper's router: ask the LLM which expert fits the question; keyword_expert when the reply is unusable."""
    u = ROUTE_PROMPT.format(experts=", ".join(EXPERTS), question=question["text"])
    text = llm.complete("", u, max_tokens=max_tokens, **kw).text
    try:
        return _expert_name(text.strip().strip(".").strip())
    except ValueError:
        hits = [e for e in EXPERTS if e in str(text).lower()]
        return hits[0] if len(hits) == 1 else keyword_expert({"text": question["text"]})


def route_experts(llm, questions: Sequence[dict], **kw: Any) -> list[dict]:
    """Copies of `questions` with 'expert' set by route_expert (questions that already have one are kept)."""
    return [q if q.get("expert") else {**q, "expert": route_expert(llm, q, **kw)} for q in questions]


def prepare(llm, people: Sequence[dict], *, summary: bool = True, reflections: bool = True,
            experts: Sequence[str] = EXPERTS, budget_tokens: int | None = None, max_tokens: int = 2048,
            parallel: int = 4, **kw: Any) -> list[dict]:
    """Copies of `people` with interview_summary and/or reflections added (fields already present are kept).
    Calls: one summary + len(experts) reflections per person that lacks them."""
    exps = [_expert_name(e) for e in experts]
    jobs: list[tuple[int, str]] = []
    prompts: list[tuple[str, str]] = []
    for i, p in enumerate(people):
        if not p.get("interview"):
            continue
        if summary and not p.get("interview_summary"):
            jobs.append((i, "summary"))
            prompts.append(summary_prompt(p, budget_tokens))
        if reflections and not p.get("reflections"):
            for e in exps:
                jobs.append((i, e))
                prompts.append(reflection_prompt(p, e, budget_tokens))
    replies = llm.map(prompts, max_tokens=max_tokens, parallel=parallel, **kw) if prompts else []
    out = [dict(p) for p in people]
    for (i, what), r in zip(jobs, replies):
        if what == "summary":
            out[i]["interview_summary"] = parse_summary(r.text)
        else:
            out[i].setdefault("reflections", {})[what] = parse_notes(r.text)
    return out


__all__ = ["InterviewTranscript", "InterviewLesion", "InterviewSummary", "InterviewExpertReflection",
           "InterviewPlusSurvey", "EXPERTS", "EXPERT_FOCUS", "TRUNCATE", "transcript_units", "fit_budget", "lesion",
           "keyword_expert", "summarize_interview", "expert_reflections", "route_expert", "route_experts",
           "prepare", "parse_summary", "parse_notes", "summary_prompt", "reflection_prompt"]
