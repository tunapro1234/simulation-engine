"""Memory-family injectors: the person's earlier answers in the session as a memory stream (Park et al. 2023,
Generative Agents, arXiv:2304.03442), retrieved by recency + importance + relevance, optionally with reflections.

Paper method (section 4): every observation is a memory record; retrieval score = a_rec*recency +
a_imp*importance + a_rel*relevance with every a = 1, each term min-max scaled to [0, 1]; recency decays
exponentially (factor 0.995 per sandbox hour since last access), importance is an LLM 1-10 poignancy rating,
relevance is embedding cosine to the query. Reflection: from the 100 most recent records ask for the 3 most
salient high-level questions, retrieve for each, ask for 5 insights cited as "(because of 1, 5, 3)".

Survey mapping here: a memory = one known answer (`known_answers`, so the target and `exclude` ids never enter);
time = questionnaire order (`questions=`) or the insertion order of person["answers"]; relevance = bag-of-words
cosine (or a pure `embed` callable); importance = a {(person_id, qid): 1-10} map or callable (`rate_importance`
builds one with the paper's prompt), else constant. LLM steps (`rate_importance`, `reflect`) run offline before
rendering, so `render` stays pure and deterministic.

    import simengine.persona.inject as pi
    from simengine.persona.inject.memory import reflect, rate_importance
    qs = [{"id": "q0", "text": "Do you own a car?", "options": ["yes", "no"]}, ...]
    p = pi.get_injector("memory_stream", k=5, questions=qs)(person, qs[0])
    p = pi.get_injector("memory_session", questions=qs, prior_only=True)(person, qs[3])   # multi-turn history
    person["reflections"] = reflect(llm, person, qs, hide=["q3"])     # insights that never saw q3
    p = pi.get_injector("memory_reflect", questions=qs)(person, qs[3])
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ._base import Injector, Prompt, register

PAPER = "Park et al. 2023, Generative Agents, arXiv:2304.03442"
DECAY = 0.995                     # paper: per sandbox hour since last retrieval
WEIGHTS = (1.0, 1.0, 1.0)         # (recency, importance, relevance), paper: all 1
DEFAULT_IMPORTANCE = 5.0

IMPORTANCE_PROMPT = (
    "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely "
    "poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory.\n"
    "Memory: {memory}\nRating: <fill in>")
QUESTIONS_PROMPT = ("{statements}\n\nGiven only the information above, what are {n} most salient high-level "
                    "questions we can answer about the subjects in the statements? One question per line.")
INSIGHTS_PROMPT = ("Statements about the person:\n{statements}\n\nWhat {n} high-level insights can you infer from the "
                   "above statements? One per line. (example format: insight (because of 1, 5, 3))")

_STOP = frozenset(
    "a an and are as at be been but by can do does did for from had has have how i if in into is it its me my "
    "no not of on or our so than that the their them then there these they this to was we were what when where "
    "which who why will with would you your yes".split())
_WORD = re.compile(r"[a-z0-9]+")
_CITE = re.compile(r"\(\s*because of\s*([\d,\s and]+)\)\s*\.?\s*$", re.I)
_NUM_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


@dataclass(frozen=True)
class Memory:
    """One memory record. `sources` are cited qids (reflections), `seen` every qid its writer was shown."""
    id: str
    text: str
    time: float
    importance: float = DEFAULT_IMPORTANCE
    kind: str = "answer"              # "answer" | "reflection"
    query_text: str = ""              # text used for relevance (question + answer, without the template)
    sources: tuple[str, ...] = ()
    seen: tuple[str, ...] = ()


# ---------------------------------------------------------------- scoring helpers
def tokens(text: str) -> list[str]:
    """Lowercase alphanumeric words without stopwords (relevance features)."""
    return [w for w in _WORD.findall(str(text).lower()) if w not in _STOP and len(w) > 1]


def bow_cosine(a: str, b: str) -> float:
    """Cosine of term-count vectors; 0 when either side has no content words."""
    ca, cb = Counter(tokens(a)), Counter(tokens(b))
    dot = sum(ca[w] * cb[w] for w in ca.keys() & cb.keys())
    na, nb = math.sqrt(sum(v * v for v in ca.values())), math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def _vec_cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v))
    nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(y * y for y in v))
    return dot / (nu * nv) if nu and nv else 0.0


def minmax(xs: Sequence[float]) -> list[float]:
    """Min-max scale to [0, 1]; a constant column scales to 0 (it cannot change the ranking)."""
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    return [0.0] * len(xs) if hi == lo else [(x - lo) / (hi - lo) for x in xs]


def retrieve(memories: Sequence[Memory], query: str, k: int | None, *, decay: float = DECAY,
             weights: Sequence[float] = WEIGHTS, embed: Callable[[str], Sequence[float]] | None = None,
             now: float | None = None) -> list[tuple[Memory, float]]:
    """Top-k (memory, score) by the paper's score; ties broken by later time, then id. k None = all."""
    if not memories:
        return []
    now = max(m.time for m in memories) if now is None else now
    rec = minmax([decay ** max(now - m.time, 0.0) for m in memories])
    imp = minmax([float(m.importance) for m in memories])
    if embed is None:
        rel = minmax([bow_cosine(query, m.query_text or m.text) for m in memories])
    else:
        qv = embed(query)
        rel = minmax([_vec_cosine(qv, embed(m.query_text or m.text)) for m in memories])
    wr, wi, wl = (float(w) for w in weights)
    scored = [(m, wr * r + wi * i + wl * l) for m, r, i, l in zip(memories, rec, imp, rel)]
    scored.sort(key=lambda t: (-t[1], -t[0].time, t[0].id))
    return scored if k is None else scored[:k]


# ---------------------------------------------------------------- memory construction
def question_texts(questions: Any) -> tuple[dict[str, str], dict[str, int]]:
    """`questions` (list of question dicts, {qid: text} or {qid: question dict}) -> ({qid: text}, {qid: order})."""
    if questions is None:
        return {}, {}
    items = questions.items() if isinstance(questions, Mapping) else ((q["id"], q) for q in questions)
    text, order = {}, {}
    for i, (qid, q) in enumerate(items):
        text[str(qid)] = q.get("text", str(qid)) if isinstance(q, Mapping) else str(q)
        order[str(qid)] = i
    return text, order


def _importance(importance: Any, person: dict, qid: str, answer: Any) -> float:
    if importance is None:
        return DEFAULT_IMPORTANCE
    if callable(importance):
        v = importance(person, qid, answer)
    else:
        v = importance.get((str(person.get("id")), qid), importance.get(qid))
    return DEFAULT_IMPORTANCE if v is None else float(v)


def answer_memories(person: dict, question: dict, questions: Any = None, importance: Any = None,
                    hide: Sequence[str] = ()) -> list[Memory]:
    """Known answers (target, `exclude` and `hide` dropped) as memory records, oldest first."""
    texts, order = question_texts(questions)
    known = Injector.known_answers(person, {**question, "exclude": [*(question.get("exclude") or ()), *hide]})
    rows = []
    for pos, (qid, a) in enumerate(known.items()):
        qid = str(qid)
        t = float(order.get(qid, len(order) + pos))       # questionnaire order, unknown ids after it
        qt = texts.get(qid, qid)
        rows.append(Memory(id=qid, text=f'I was asked "{qt}" and answered: {a}', time=t,
                           importance=_importance(importance, person, qid, a), query_text=f"{qt} {a}"))
    return sorted(rows, key=lambda m: (m.time, m.id))


def reflection_memories(person: dict, question: dict, now: float) -> tuple[list[Memory], int]:
    """person["reflections"] usable for this target: those whose `seen` excludes the target and `exclude` ids.
    Returns (memories, n_dropped). A reflection without `seen` cannot be checked for leakage and raises."""
    drop = {str(x) for x in (question.get("exclude") or ())} | {str(question.get("id"))}
    out, dropped = [], 0
    for i, r in enumerate(person.get("reflections") or ()):
        if isinstance(r, str) or "seen" not in r:
            raise ValueError(f"reflection {i} of person {person.get('id')!r} has no 'seen' list, so leakage cannot "
                             "be checked; build reflections with memory.reflect()")
        seen = tuple(str(s) for s in r["seen"])
        if drop & set(seen):
            dropped += 1
            continue
        out.append(Memory(id=f"r{i}", text=str(r["text"]), time=now, kind="reflection",
                          importance=float(r.get("importance", DEFAULT_IMPORTANCE)), query_text=str(r["text"]),
                          sources=tuple(str(s) for s in r.get("sources") or ()), seen=seen))
    return out, dropped


def _short(v: Any) -> str:
    if callable(v):
        return getattr(v, "__name__", type(v).__name__)
    if isinstance(v, (list, tuple, dict)) and len(repr(v)) > 40:
        return f"#{hashlib.sha1(repr(v).encode()).hexdigest()[:6]}"
    return str(v)


class _MemoryBase(Injector):
    family = "memory"
    paper = PAPER

    @property
    def label(self) -> str:
        """Like Injector.label, but long values (question lists, maps) shown as a short hash, callables by name."""
        if not self.params:
            return self.name
        return f"{self.name}({','.join(f'{k}={_short(v)}' for k, v in sorted(self.params.items()))})"


def _check_k(k: Any) -> None:
    if k is not None and (not isinstance(k, int) or k < 1):
        raise ValueError(f"k must be a positive int or None, not {k!r}")


# ---------------------------------------------------------------- injectors
@register
class MemoryStream(_MemoryBase):
    """Top-k earlier answers by recency + importance + relevance (paper section 4.1), shown oldest first."""
    name = "memory_stream"
    needs = ("answers",)
    description = ("System: the k known answers with the highest min-max-scaled recency + importance + relevance "
                   "score (bag-of-words cosine to the question), listed in session order.")
    header = ("You are a survey respondent answering a questionnaire in one session. Memories retrieved from your "
              "memory stream (earlier in this session):")

    def __init__(self, k: int | None = 5, decay: float = DECAY, weights: Sequence[float] = WEIGHTS,
                 questions: Any = None, importance: Any = None, embed: Callable | None = None, **params: Any):
        super().__init__(k=k, **params)
        _check_k(k)
        if not 0 < decay <= 1:
            raise ValueError(f"decay must be in (0, 1], not {decay!r}")
        if len(weights) != 3 or any(w < 0 for w in weights):
            raise ValueError("weights = (recency, importance, relevance), three non-negative numbers")
        self.k, self.decay, self.weights = k, decay, tuple(float(w) for w in weights)
        self.questions, self.importance, self.embed = questions, importance, embed

    def memories(self, person: dict, question: dict) -> tuple[list[Memory], dict]:
        return answer_memories(person, question, self.questions, self.importance), {}

    def render(self, person: dict, question: dict) -> Prompt:
        mems, meta = self.memories(person, question)
        top = retrieve(mems, question["text"], self.k, decay=self.decay, weights=self.weights, embed=self.embed)
        shown = sorted(top, key=lambda t: (t[0].kind == "reflection", t[0].time, t[0].id))
        lines = [f"- {'[reflection] ' if m.kind == 'reflection' else ''}{m.text}" for m, _ in shown]
        system = "\n".join([self.header, *(lines or ["- (no earlier answers)"]),
                            "Answer the next question as this person would, consistent with these memories."])
        meta.update(memories=[m.id for m, _ in shown], scores={m.id: round(s, 6) for m, s in top})
        return Prompt(system=system, user=self.question_block(question), meta=meta)


@register
class MemoryReflect(MemoryStream):
    """Memory stream plus reflections (paper section 4.2) retrieved in the same top-k as the answers."""
    name = "memory_reflect"
    needs = ("answers", "reflections")
    description = ("memory_stream over known answers + person['reflections'] (insights from memory.reflect); a "
                   "reflection whose writer saw the target or an `exclude` id is dropped.")
    header = ("You are a survey respondent answering a questionnaire in one session. Memories and reflections "
              "retrieved from your memory stream:")

    def memories(self, person: dict, question: dict) -> tuple[list[Memory], dict]:
        mems = answer_memories(person, question, self.questions, self.importance)
        now = max([m.time for m in mems], default=0.0) + 1.0          # reflections are written after the answers
        refl, dropped = reflection_memories(person, question, now)
        return mems + refl, {"reflections_dropped": dropped}


@register
class MemorySession(_MemoryBase):
    """The questionnaire so far as a multi-turn conversation: the memory stream in raw chronological order with
    no retrieval (the paper's observation log); `k` keeps the most recent turns only."""
    name = "memory_session"
    needs = ("answers",)
    description = ("History: earlier questions as interviewer turns and the person's answers as JSON replies, in "
                   "questionnaire order; prior_only keeps only questions before the target.")
    system = "You are a survey respondent answering a questionnaire. Stay consistent with your earlier answers."

    def __init__(self, k: int | None = None, questions: Any = None, prior_only: bool = False,
                 as_history: bool = True, **params: Any):
        super().__init__(k=k, **params)
        _check_k(k)
        self.k, self.questions, self.prior_only, self.as_history = k, questions, prior_only, as_history
        self._texts, self._order = question_texts(questions)
        if prior_only and not self._order:
            raise ValueError("prior_only needs questions= (the questionnaire order)")

    def render(self, person: dict, question: dict) -> Prompt:
        mems = answer_memories(person, question, self.questions)
        if self.prior_only:
            qid = str(question.get("id"))
            if qid not in self._order:
                raise ValueError(f"prior_only: target {qid!r} is not in questions=")
            mems = [m for m in mems if m.id in self._order and self._order[m.id] < self._order[qid]]
        if self.k is not None:
            mems = mems[-self.k:]
        known = Injector.known_answers(person, question)
        turns = [(self._texts.get(m.id, m.id), known[m.id]) for m in mems]
        meta = {"memories": [m.id for m in mems]}
        if self.as_history:
            hist = [t for q, a in turns for t in (("user", q), ("assistant", json.dumps({"answer": a})))]
            return Prompt(system=self.system, user=self.question_block(question), history=hist, meta=meta)
        lines = [f"- {q}: {a}" for q, a in turns] or ["- (no earlier answers)"]
        system = "\n".join([self.system, "Your earlier answers in this session:", *lines])
        return Prompt(system=system, user=self.question_block(question), meta=meta)


# ---------------------------------------------------------------- offline LLM steps (run before rendering)
def _first_int(text: str, lo: int = 1, hi: int = 10) -> int | None:
    m = re.search(r"-?\d+", str(text))
    if not m:
        return None
    v = int(m.group())
    return v if lo <= v <= hi else None


def rate_importance(llm: Any, people: Sequence[dict], questions: Any = None, *, max_tokens: int = 8,
                    **llm_kw: Any) -> dict[tuple[str, str], int]:
    """Paper's poignancy prompt for every (person, answered question): {(person_id, qid): 1..10}. Unparsed
    ratings are left out (they fall back to the constant). Pass the result as `importance=`."""
    texts, _ = question_texts(questions)
    keys, prompts = [], []
    for p in people:
        for qid, a in (p.get("answers") or {}).items():
            qt = texts.get(str(qid), str(qid))
            keys.append((str(p["id"]), str(qid)))
            prompts.append(("", IMPORTANCE_PROMPT.format(memory=f'I was asked "{qt}" and answered: {a}')))
    out = {}
    for key, r in zip(keys, llm.map(prompts, max_tokens=max_tokens, **llm_kw)):
        v = _first_int(r.text)
        if v is not None:
            out[key] = v
    return out


def _lines(text: str) -> list[str]:
    return [s for s in (_NUM_PREFIX.sub("", ln).strip() for ln in str(text).splitlines()) if s]


def reflect(llm: Any, person: dict, questions: Any = None, *, hide: Sequence[str] = (), n_questions: int = 3,
            n_insights: int = 5, recent: int = 100, k: int = 10, importance: Any = None, max_tokens: int = 512,
            **llm_kw: Any) -> list[dict]:
    """The paper's reflection step over the person's answers minus `hide` (e.g. every test qid). Returns
    [{"text", "sources": cited qids, "seen": every qid shown to the LLM, "query"}]; insights without a valid
    "(because of ...)" citation are dropped. Store as person["reflections"] for memory_reflect."""
    mems = answer_memories(person, {"id": None}, questions, importance, hide=[str(h) for h in hide])[-recent:]
    if not mems:
        return []
    seen = [m.id for m in mems]
    numbered = "\n".join(f"{i}. {m.text}" for i, m in enumerate(mems, 1))
    r = llm.complete("", QUESTIONS_PROMPT.format(statements=numbered, n=n_questions), max_tokens=max_tokens, **llm_kw)
    out: list[dict] = []
    for q in _lines(r.text)[:n_questions]:
        top = sorted((m for m, _ in retrieve(mems, q, k)), key=lambda m: (m.time, m.id))
        stmts = "\n".join(f"{i}. {m.text}" for i, m in enumerate(top, 1))
        r = llm.complete("", INSIGHTS_PROMPT.format(statements=stmts, n=n_insights), max_tokens=max_tokens, **llm_kw)
        for line in _lines(r.text)[:n_insights]:
            m = _CITE.search(line)
            if not m:
                continue
            idx = [int(x) for x in re.findall(r"\d+", m.group(1))]
            src = sorted({top[i - 1].id for i in idx if 1 <= i <= len(top)})
            text = line[:m.start()].strip().rstrip(".")
            if src and text:
                out.append({"text": text, "sources": src, "seen": seen, "query": q})
    return out


__all__ = ["Memory", "MemoryStream", "MemoryReflect", "MemorySession", "retrieve", "answer_memories",
           "reflection_memories", "rate_importance", "reflect", "bow_cosine", "minmax", "tokens", "PAPER"]
