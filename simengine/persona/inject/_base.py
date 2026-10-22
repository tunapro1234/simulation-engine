"""Core of the persona injection zoo: the Prompt an injector renders, the Injector base class and the registry.

An injector turns (person, question) into a Prompt. Family modules (`inject/<family>.py`) subclass Injector,
decorate it with `@register` and are imported automatically by `simengine.persona.inject`.

    from simengine.persona.inject._base import Injector, Prompt, register

    @register
    class Bio(Injector):
        name, family, paper, needs = "bio", "narrative", "Argyle et al. 2023", ("text",)
        def render(self, person, question):
            return Prompt(system=f"You are this person:\n{person['text']}", user=self.question_block(question))

    p = get_injector("bio").render({"id": "u1", "text": "..."}, {"id": "q1", "text": "Own a car?", "options": ["yes", "no"]})
    llm.complete(p.system, p.flatten())

Person dict:   {"id": str, "text": str, "fields": {k: v}, "answers": {qid: answer}, "interview": str | [(q, a)],
                "history": [(role, text)] | [event dicts], ...}; only "id" is required, `needs` lists the rest.
Question dict: {"id": str, "text": str, "options": [str, ...] (choice) | absent (number, optional "min"/"max"),
                "exclude": [qid, ...] (other items hidden from answers= as well)}.
LEAKAGE RULE: an injector that shows a person's own answers uses `known_answers(person, question)`, which drops
the target question and `exclude` ids.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Sequence

from ...games._answer import instruction, parse_answer

ROLES = ("user", "assistant")
# person keys that are structure, not displayable attributes (plain_fields skips them)
RESERVED = ("id", "text", "fields", "answers", "interview", "history", "params", "meta")


class MissingPersonField(KeyError):
    """The person dict lacks a field the injector `needs`."""


@dataclass
class Prompt:
    """One rendered call. `history` holds prior turns [(role, text)] with role in ROLES, oldest first; the final
    turn is always `user`. `meta` is free (injector name, donor id for swap, token notes, ...)."""
    system: str
    user: str
    history: list[tuple[str, str]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for role, _ in self.history:
            if role not in ROLES:
                raise ValueError(f"history role must be one of {ROLES}, not {role!r}")

    def flatten(self) -> str:
        """User text with the history folded in, for LLMs that take only (system, user)."""
        if not self.history:
            return self.user
        lines = ["Earlier in this conversation:"]
        lines += [f"{'Interviewer' if r == 'user' else 'You'}: {t}" for r, t in self.history]
        return "\n".join(lines) + "\n\nNow:\n" + self.user

    def messages(self) -> list[dict]:
        """Anthropic-style message list (history + final user turn)."""
        return [{"role": r, "content": t} for r, t in self.history] + [{"role": "user", "content": self.user}]


def answer_spec(question: dict) -> dict:
    """games._answer spec for a question: choice when it has options, else a number (min/max optional)."""
    if question.get("options"):
        return {"kind": "mc", "options": [str(o) for o in question["options"]]}
    return {"kind": "num", "min": question.get("min"), "max": question.get("max")}


def parse_option(text: Any, question: dict) -> Any:
    """Strict parse of one reply against the question (JSON {"answer": ...}, exact option or bare number);
    None when unusable. Same rules as simengine.games (no substring salvage)."""
    value, err = parse_answer(text, answer_spec(question))
    return None if err else value


class Injector:
    """Base class. Subclasses set the class attributes and implement `render`; constructor keywords are the
    method's parameters (get_injector(name, **params) passes them). Keep render() pure and deterministic:
    randomness only from `self.seed`-style params, so a rerun hits the LLM disk cache."""
    name: ClassVar[str] = ""
    family: ClassVar[str] = ""
    paper: ClassVar[str] = ""
    needs: ClassVar[tuple[str, ...]] = ()
    description: ClassVar[str] = ""

    def __init__(self, **params: Any):
        self.params = dict(params)

    @property
    def label(self) -> str:
        """Arm name in comparisons: name, plus the params given to get_injector (e.g. "fewshot(k=3)")."""
        if not self.params:
            return self.name
        return f"{self.name}({','.join(f'{k}={v}' for k, v in sorted(self.params.items()))})"

    def render(self, person: dict, question: dict) -> Prompt:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, person: dict, question: dict) -> Prompt:
        self.check(person)
        p = self.render(person, question)
        p.meta.setdefault("injector", self.name)
        return p

    def check(self, person: dict) -> None:
        missing = [f for f in self.needs if _empty(person.get(f))]
        if missing:
            raise MissingPersonField(f"injector {self.name!r} needs person field(s) {missing} "
                                     f"(person {person.get('id')!r})")

    @classmethod
    def describe(cls) -> dict:
        return {"name": cls.name, "family": cls.family, "paper": cls.paper, "needs": list(cls.needs),
                "description": cls.description}

    # ------------------------------------------------------------ helpers for subclasses
    @staticmethod
    def question_block(question: dict) -> str:
        """Question text + the strict answer-format line that `parse_option` reads."""
        return f"{question['text']}\n\n{instruction(answer_spec(question))}"

    @staticmethod
    def known_answers(person: dict, question: dict) -> dict:
        """person['answers'] without the target question and its `exclude` ids (leakage rule)."""
        drop = {str(x) for x in (question.get("exclude") or ())}
        if question.get("id") is not None:
            drop.add(str(question["id"]))
        return {k: v for k, v in (person.get("answers") or {}).items() if str(k) not in drop}

    @staticmethod
    def person_fields(person: dict) -> dict:
        """person['fields'] if present, else the scalar top-level keys that are not RESERVED."""
        if isinstance(person.get("fields"), dict):
            return dict(person["fields"])
        return {k: v for k, v in person.items()
                if k not in RESERVED and isinstance(v, (str, int, float, bool)) and v != ""}


def _empty(v: Any) -> bool:
    if v is None:
        return True
    try:
        return len(v) == 0
    except TypeError:
        return False


REGISTRY: dict[str, type[Injector]] = {}


def register(cls: type[Injector] | None = None, *, replace: bool = False) -> Callable | type[Injector]:
    """Class decorator: `@register` or `@register(replace=True)`. The class needs a unique `name`."""
    def deco(c: type[Injector]) -> type[Injector]:
        if not (isinstance(c, type) and issubclass(c, Injector)):
            raise TypeError("register() decorates Injector subclasses")
        if not c.name:
            raise ValueError(f"{c.__name__} has no `name`")
        if c.name in REGISTRY and REGISTRY[c.name] is not c and not replace:
            raise ValueError(f"injector {c.name!r} already registered by {REGISTRY[c.name].__module__}")
        REGISTRY[c.name] = c
        return c
    return deco if cls is None else deco(cls)


def list_injectors(family: str | None = None) -> list[dict]:
    """One row per registered injector: name, family, paper, needs, description (sorted by family, name)."""
    rows = [c.describe() for c in REGISTRY.values() if family is None or c.family == family]
    return sorted(rows, key=lambda r: (r["family"], r["name"]))


def get_injector(name: str | Injector, **params: Any) -> Injector:
    """Instance by registry name (params go to the constructor); an Injector instance is returned as is."""
    if isinstance(name, Injector):
        if params:
            raise ValueError("params are only accepted with an injector name")
        return name
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown injector {name!r}; choose from {sorted(REGISTRY)}") from None
    inst = cls(**params)
    inst.params = dict(params)          # the arm label shows what the caller set, not constructor defaults
    return inst


# ---------------------------------------------------------------- built-ins

@register
class NoPersona(Injector):
    """Control: the bare question, no persona at all."""
    name = "no_persona"
    family = "control"
    paper = "control arm (every persona study)"
    description = "Empty system prompt, question only."

    def render(self, person: dict, question: dict) -> Prompt:
        return Prompt(system="", user=self.question_block(question))


@register
class PlainFields(Injector):
    """`key: value` lines of the person's attributes in the system prompt."""
    name = "plain_fields"
    family = "control"
    paper = "attribute-list baseline (e.g. Argyle et al. 2023; Santurkar et al. 2023)"
    description = "System: 'You are a person with these attributes:' + key: value lines (person['fields'] or flat scalars)."

    def __init__(self, keys: Sequence[str] | None = None, header: str = "You are a person with these attributes:",
                 **params: Any):
        super().__init__(**({"keys": list(keys)} if keys else {}), **params)
        self.keys = list(keys) if keys else None
        self.header = header

    def render(self, person: dict, question: dict) -> Prompt:
        f = self.person_fields(person)
        keys = self.keys if self.keys is not None else list(f)
        lines = [f"- {k.replace('_', ' ')}: {f[k]}" for k in keys if k in f and f[k] not in (None, "")]
        system = self.header + "\n" + "\n".join(lines) + "\nAnswer as this person would." if lines else ""
        return Prompt(system=system, user=self.question_block(question))


class Swap(Injector):
    """Control wrapper: render `inner` with a different person's persona (a fixed derangement over `people`), so
    the reply is scored against the original person's truth. Not registered (needs the people list)."""
    family = "control"
    paper = "persona-swap / shuffle control"

    def __init__(self, inner: Injector, donors: dict[str, dict]):
        self.inner = inner
        self.donors = donors
        self.params = {}
        self.name = f"swap_{inner.label}"
        self.needs = inner.needs

    @property
    def label(self) -> str:
        return self.name

    def check(self, person: dict) -> None:
        self.inner.check(self.donors[str(person["id"])])

    def render(self, person: dict, question: dict) -> Prompt:
        donor = self.donors[str(person["id"])]
        p = self.inner(donor, question)
        p.meta.update(donor=donor.get("id"), swap_of=self.inner.name)
        return p


def derangement(ids: Sequence[str], seed: int = 0) -> dict[str, str]:
    """{id: other id}: a random permutation with no fixed point (needs >= 2 distinct ids)."""
    import random

    ids = [str(i) for i in ids]
    if len(set(ids)) != len(ids):
        raise ValueError("person ids must be unique")
    if len(ids) < 2:
        raise ValueError("swap control needs at least 2 people")
    rng = random.Random(seed)
    order = ids[:]
    rng.shuffle(order)
    return {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}   # one cycle: never itself


def swap_wrapper(inner: Injector | str, people: Sequence[dict], seed: int = 0, **params: Any) -> Swap:
    """Swap control for `inner` over `people` (each person gets one fixed donor, never themself)."""
    inj = get_injector(inner, **params)
    by_id = {str(p["id"]): p for p in people}
    pairs = derangement([str(p["id"]) for p in people], seed)
    return Swap(inj, {k: by_id[v] for k, v in pairs.items()})


__all__ = ["Prompt", "Injector", "MissingPersonField", "REGISTRY", "register", "list_injectors", "get_injector",
           "NoPersona", "PlainFields", "Swap", "swap_wrapper", "derangement", "answer_spec", "parse_option",
           "ROLES", "RESERVED"]
