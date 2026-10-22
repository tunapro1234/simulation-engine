"""Framing family: the same persona attributes rendered with three orthogonal knobs, placement x voice x format,
plus attribute order. Presets reproduce the framings of specific papers; `grid()` spans the knobs for an ablation.

    import simengine.persona.inject as pi
    from simengine.persona.inject.framing import grid
    person = {"id": "u1", "fields": {"age": 34, "region": "north"}}
    q = {"id": "q1", "text": "Do you own a car?", "options": ["yes", "no"]}
    pi.get_injector("framed", placement="user", voice="first", format="prose")(person, q).user
    # 'My age is 34. My region is north.\nAnswer as the person described above would.\n\nDo you own a car?...'
    arms = grid(placements=("system", "user", "turns"), voices=("second", "first"), formats=("bullets", "prose"))
    pi.compare_injectors(llm, people, questions, "answers", arms, dry_run=True)

Knobs (all constructor params, so they show in the arm label):
  placement  system | user (prefix before the question) | turns (persona as an earlier user/assistant exchange)
  voice      second ("You are ...") | first ("I am ...") | third ("Imagine a person ...") | interview (roleplay:
             the persona is the interviewee's earlier answers, the question comes from the interviewer)
  format     bullets | prose | json | qa (question/answer pairs, one per attribute)
  order      given | sorted | reversed | shuffle (seeded per person, deterministic); `keys` picks and orders.
Only person["fields"] is read. A field keyed by the target question id or its `exclude` ids is dropped (leakage
guard: a prep step that copied the target into fields does not reach the prompt).
"""
from __future__ import annotations

import json
import math
import random
from typing import Any, Iterable, Sequence

from ._base import Injector, MissingPersonField, Prompt, get_injector, register

PLACEMENTS = ("system", "user", "turns")
VOICES = ("second", "first", "third", "interview")
FORMATS = ("bullets", "prose", "json", "qa")
ORDERS = ("given", "sorted", "reversed", "shuffle")

INTERVIEWEE = "You are an interviewee. Based on your previous answers, provide an answer to the last question."
HEADERS = {   # opening line of the persona block; first person prose stands alone
    "second": "You are a person with the following profile.",
    "first": "About me:",
    "third": "Imagine a person with the following profile.",
    "interview": INTERVIEWEE,
}
CLOSINGS = {
    "second": "Answer as this person would.",
    "first": "Answer as the person described above would.",
    "third": "Answer as this person would.",
    "interview": "",
}
ACKS = {   # assistant turn after a second/third person persona in placement="turns"
    "second": "Understood. I will answer as this person.",
    "third": "Understood. I will answer as that person would.",
}
INTRO = "Before we begin, please tell me about yourself."


def _value(v: Any) -> str | None:
    """Display string; None for empty / NaN (the fragment is omitted, as in Argyle et al.)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (list, tuple, set)):
        v = ", ".join(str(x) for x in v)
    s = str(v).strip()
    return s or None


def _label(k: str) -> str:
    return str(k).replace("_", " ")


@register
class Framed(Injector):
    """Persona fields rendered by placement x voice x format (see module docstring)."""
    name = "framed"
    family = "framing"
    paper = "Tseng et al. 2024, arXiv:2406.01171 (survey); placement/voice/format ablation grid"
    needs = ("fields",)
    description = ("person['fields'] rendered with placement (system/user/turns), voice (second/first/third/interview), "
                   "format (bullets/prose/json/qa) and order (given/sorted/reversed/shuffle+seed); keys= subsets.")
    defaults: dict = {"placement": "system", "voice": "second", "format": "bullets"}

    def __init__(self, placement: str | None = None, voice: str | None = None, format: str | None = None,
                 order: str = "given", keys: Sequence[str] | None = None, seed: int = 0,
                 header: str | None = None, phrases: dict[str, str] | None = None, **params: Any):
        d = type(self).defaults
        self.placement = placement or d["placement"]
        self.voice = voice or d["voice"]
        self.fmt = format or d["format"]
        self.order = order
        for knob, val, allowed in (("placement", self.placement, PLACEMENTS), ("voice", self.voice, VOICES),
                                   ("format", self.fmt, FORMATS), ("order", order, ORDERS)):
            if val not in allowed:
                raise ValueError(f"{self.name}: {knob} must be one of {allowed}, not {val!r}")
        self.keys = list(keys) if keys else None
        self.seed = seed
        self.header = d.get("header") if header is None else header
        self.phrases = dict(phrases or d.get("phrases") or {})   # {attr: "In terms of my age, I am {value}."}
        super().__init__(placement=self.placement, voice=self.voice, format=self.fmt, order=order,
                         keys=self.keys, seed=seed, **params)

    # ------------------------------------------------------------ attribute selection
    def attributes(self, person: dict, question: dict) -> list[tuple[str, str]]:
        """[(key, value)] in render order, empty values and leaked keys removed."""
        f = person.get("fields")
        if not isinstance(f, dict):
            raise MissingPersonField(f"injector {self.name!r} needs person field(s) {list(self.needs)} as a dict "
                                     f"(person {person.get('id')!r})")
        drop = {str(x) for x in (question.get("exclude") or ())}
        if question.get("id") is not None:
            drop.add(str(question["id"]))
        keys = self.keys if self.keys is not None else list(f)
        items = [(k, _value(f[k])) for k in keys if k in f and str(k) not in drop]
        items = [(k, v) for k, v in items if v is not None]
        if not items:
            raise MissingPersonField(f"injector {self.name!r} needs person field(s) {list(self.needs)} with at least "
                                     f"one non-empty value (keys={self.keys}, person {person.get('id')!r})")
        if self.order == "sorted":
            items.sort(key=lambda kv: str(kv[0]))
        elif self.order == "reversed":
            items.reverse()
        elif self.order == "shuffle":
            random.Random(f"{self.seed}:{person.get('id')}").shuffle(items)   # str seed: stable across runs
        return items

    # ------------------------------------------------------------ text pieces
    def sentence(self, k: str, v: str) -> str:
        if k in self.phrases:
            return self.phrases[k].format(value=v)
        poss = {"second": "Your", "third": "Their"}.get(self.voice, "My")
        return f"{poss} {_label(k)} is {v}."

    def qa_pair(self, k: str, v: str) -> tuple[str, str]:
        q = f"What is this person's {_label(k)}?" if self.voice == "third" else f"What is your {_label(k)}?"
        a = self.sentence(k, v) if self.voice in ("first", "interview") else v
        return q, a

    def block(self, items: list[tuple[str, str]]) -> str:
        """The attribute lines alone (no header / closing)."""
        if self.fmt == "bullets":
            return "\n".join(f"- {_label(k)}: {v}" for k, v in items)
        if self.fmt == "prose":
            return " ".join(self.sentence(k, v) for k, v in items)
        if self.fmt == "json":
            return json.dumps({str(k): v for k, v in items}, ensure_ascii=False)
        tags = ("Interviewer", "Interviewee") if self.voice == "interview" else ("Q", "A")
        return "\n".join(f"{tags[0]}: {q}\n{tags[1]}: {a}" for q, a in (self.qa_pair(k, v) for k, v in items))

    def heading(self) -> str:
        if self.header is not None:
            return self.header
        if self.voice == "first" and self.fmt == "prose":
            return ""
        return HEADERS[self.voice]

    @staticmethod
    def _join(*parts: str, sep: str = "\n") -> str:
        return sep.join(p for p in parts if p)

    # ------------------------------------------------------------ render
    def render(self, person: dict, question: dict) -> Prompt:
        items = self.attributes(person, question)
        qtext = self.question_block(question)
        body = self.block(items)
        meta = {"framing": {"placement": self.placement, "voice": self.voice, "format": self.fmt,
                            "order": self.order}, "keys": [k for k, _ in items]}
        persona = self._join(self.heading(), body, CLOSINGS[self.voice])
        if self.placement in ("system", "user"):
            if self.voice == "interview":
                qtext = "Interviewer: " + qtext
            if self.placement == "system":
                return Prompt(system=persona, user=qtext, meta=meta)
            return Prompt(system="", user=persona + "\n\n" + qtext, meta=meta)
        # placement == "turns": the persona is an earlier exchange; flatten() / messages() label the roles
        if self.voice == "interview" and self.fmt == "qa":       # one exchange per attribute (Lutz et al.)
            hist = [t for k, v in items for t in zip(("user", "assistant"), self.qa_pair(k, v))]
            return Prompt(system=self.header if self.header is not None else INTERVIEWEE, user=qtext,
                          history=hist, meta=meta)
        if self.voice in ("first", "interview"):                 # the model itself told its story
            told = self._join(self.heading(), body) if self.voice == "first" else body
            system = "" if self.voice == "first" else (self.header if self.header is not None else INTERVIEWEE)
            user = qtext if self.voice == "interview" else "Answer as the person you just described would.\n\n" + qtext
            return Prompt(system=system, user=user, history=[("user", INTRO), ("assistant", told)], meta=meta)
        return Prompt(system="", user=qtext, history=[("user", persona), ("assistant", ACKS[self.voice])], meta=meta)


# ---------------------------------------------------------------- literature presets (same knobs, other defaults)

@register
class ArgyleBackstory(Framed):
    """First-person backstory as a prefix before the question (GPT-3 completion setup in the paper)."""
    name = "argyle_backstory"
    paper = "Argyle et al. 2023, Out of One, Many, arXiv:2209.06899"
    description = ("First-person backstory sentences ('My age is 34.'; phrases= for per-attribute templates), "
                   "missing fragments omitted, as a user-turn prefix before the question.")
    defaults = {"placement": "user", "voice": "first", "format": "prose"}


@register
class HuCollierProfile(Framed):
    """'Your Profile:' + second-person declarative sentences in the system prompt (chat models in the paper)."""
    name = "hu_collier_profile"
    paper = "Hu & Collier 2024, Quantifying the Persona Effect in LLM Simulations, arXiv:2402.10811"
    description = ("System: 'Your Profile:' + second-person sentences ('Your age is 34.'); the paper also varied "
                   "attribute order (order=shuffle, seed=...).")
    defaults = {"placement": "system", "voice": "second", "format": "prose", "header": "Your Profile:"}


@register
class LutzThirdPerson(Framed):
    """Third-person role adoption ('Think of a ...') in the user prompt."""
    name = "lutz_third_person"
    paper = "Lutz et al. 2025, The Prompt Makes the Person(a), arXiv:2507.16076 (third-person role adoption)"
    description = "User turn: 'Think of a person with the following profile.' + third-person sentences, then the question."
    defaults = {"placement": "user", "voice": "third", "format": "prose",
                "header": "Think of a person with the following profile."}


@register
class LutzInterview(Framed):
    """Interview role adoption: one interviewer/interviewee exchange per attribute, then the question."""
    name = "lutz_interview"
    paper = "Lutz et al. 2025, The Prompt Makes the Person(a), arXiv:2507.16076 (interview role adoption)"
    description = ("System: 'You are an interviewee ...'; prior turns 'What is your age?' / 'My age is 34.' per "
                   "attribute; the question is the last interviewer turn.")
    defaults = {"placement": "turns", "voice": "interview", "format": "qa"}


def grid(placements: Iterable[str] = PLACEMENTS, voices: Iterable[str] = VOICES, formats: Iterable[str] = FORMATS,
         name: str = "framed", **params: Any) -> list[Injector]:
    """One injector per (placement, voice, format) cell; labels carry the knobs, e.g.
    framed(format=json,placement=user,voice=first). params (order, keys, seed, ...) go to every cell."""
    return [get_injector(name, placement=p, voice=v, format=f, **params)
            for p in placements for v in voices for f in formats]


__all__ = ["Framed", "ArgyleBackstory", "HuCollierProfile", "LutzThirdPerson", "LutzInterview", "grid",
           "PLACEMENTS", "VOICES", "FORMATS", "ORDERS"]
