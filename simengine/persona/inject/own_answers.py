"""Own-answers injectors: the person's own other survey answers as the persona (family "own_answers").

Three variants from the literature, all built on `Injector.known_answers` (target and `exclude` ids never shown):
  own_answers_all      every known answer as question/options/answer text in the system prompt
                       (Twin-2K-500, Toubia et al. 2025, arXiv:2505.17479)
  own_answers_fewshot  k answers as earlier interview turns (user: question, assistant: {"answer": ...}),
                       picked by random / recent / first (in-context demos; LaMP Random/Recency selectors)
  own_answers_rag      the k answers whose questions are most similar to the target question (TF-IDF cosine,
                       BM25 or token overlap; pure Python, no embeddings), order ablation
                       (Hwang et al. 2023, arXiv:2305.14929; LaMP, Salemi et al. 2023, arXiv:2304.11406)

`questions` is the question bank (list of question dicts, or {qid: dict | text}) used to show past questions
by text and options and, for rag, to retrieve by text; bank order is the "time" order (later = more recent).
Without a bank a past question is shown by its id (all, fewshot only).

    import simengine.persona.inject as pi
    bank = [{"id": "q1", "text": "Do you own a car?", "options": ["yes", "no"]}, ...]
    inj = pi.get_injector("own_answers_rag", questions=bank, k=3, scorer="bm25")
    p = inj({"id": "u1", "answers": {"q1": "yes", "q2": "no"}}, bank[2])
    p.meta["shown"]                                   # qids put in context, most similar first
    pi.compare_injectors(llm, people, bank, "answers",
                         [("own_answers_rag", {"questions": bank, "k": k}) for k in (3, 8)] + ["own_answers_all"])
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter
from typing import Any, Mapping, Sequence

from ._base import Injector, Prompt, register

FAMILY = "own_answers"
TWIN2K = "Toubia et al. 2025, Twin-2K-500 (arXiv:2505.17479)"
LAMP = "Salemi et al. 2023, LaMP (arXiv:2304.11406)"
HWANG = "Hwang et al. 2023, Aligning Language Models to User Opinions (arXiv:2305.14929)"

SYSTEM_ALL = ("Below are one real person's answers to earlier survey questions.\n"
              "You are this person. Answer the next question as this person would, staying consistent with "
              "their earlier answers.")
SYSTEM_FEWSHOT = ("You are one specific person taking a survey. Your earlier answers are shown in the "
                  "conversation; answer the next question as yourself, consistent with them.")
SYSTEM_RAG = ("Below are this person's answers to the earlier survey questions most related to the next one.\n"
              "You are this person. Answer the next question as this person would, staying consistent with "
              "these answers.")
NONE_KNOWN = "(no earlier answers of this person are available)"

_TOKEN = re.compile(r"[^\W_]+")
_STOP = frozenset("a an and are as at be by do does for from how i if in is it of on or that the this to was "
                  "what when which who will with would you your".split())


# ---------------------------------------------------------------- bank + rendering
def as_bank(questions: Sequence[dict] | Mapping[str, Any] | None) -> dict[str, dict]:
    """Question bank -> {qid: question dict} in the given order (insertion order = time order)."""
    if not questions:
        return {}
    items = questions.items() if isinstance(questions, Mapping) else ((q["id"], q) for q in questions)
    out = {}
    for qid, q in items:
        q = {"text": q} if isinstance(q, str) else dict(q)
        q.setdefault("id", qid)
        q.setdefault("text", str(qid))
        out[str(qid)] = q
    return out


def _qa_text(q: dict, answer: Any, include_options: bool) -> str:
    lines = [f"Q: {q['text']}"]
    if include_options and q.get("options"):
        lines.append("Options: " + " / ".join(str(o) for o in q["options"]))
    lines.append(f"A: {answer}")
    return "\n".join(lines)


def _fold(text: Any) -> str:
    """Lowercase, accents folded (NFKD minus combining marks), so "Araç"/"arac" and "İstanbul"/"istanbul" match."""
    t = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def tokens(text: str) -> list[str]:
    """Lowercase, accent-folded word tokens (any script) minus a small English stopword list, plural "s" stripped
    (retrieval only)."""
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t
            for t in _TOKEN.findall(_fold(text)) if t not in _STOP]


def _missing(v: Any) -> bool:
    """None, NaN (float / numpy) or pandas NA: an unanswered item, not an answer."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return True
    return type(v).__name__ in ("NAType", "NaTType")


def _json_default(o: Any) -> Any:
    return o.item() if hasattr(o, "item") else str(o)       # numpy scalars from pandas rows


# ---------------------------------------------------------------- embedding-free scorers
def score_tfidf(query: str, docs: Sequence[str]) -> list[float]:
    """Cosine of l2-normalised TF-IDF vectors; idf = ln((1+n)/(1+df)) + 1 over docs (sklearn's smooth idf). As with
    a vectorizer fit on `docs`, query terms that occur in no doc are ignored."""
    toks = [Counter(tokens(d)) for d in docs]
    n = len(docs)
    df = Counter(t for c in toks for t in c)
    idf = {t: math.log((1 + n) / (1 + df[t])) + 1 for t in df}

    def vec(c: Counter) -> dict:
        v = {t: tf * idf[t] for t, tf in c.items() if t in idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}
    qv = vec(Counter(tokens(query)))
    return [sum(w * dv.get(t, 0.0) for t, w in qv.items()) for dv in map(vec, toks)]


def score_bm25(query: str, docs: Sequence[str], k1: float = 1.2, b: float = 0.75) -> list[float]:
    """Okapi BM25 (LaMP's term-matching retriever), idf = ln(1 + (n - df + .5) / (df + .5))."""
    toks = [Counter(tokens(d)) for d in docs]
    n = len(docs)
    avg = (sum(sum(c.values()) for c in toks) / n) if n else 0.0
    df = Counter(t for c in toks for t in c)
    q = set(tokens(query))
    out = []
    for c in toks:
        dl = sum(c.values())
        s = 0.0
        for t in q:
            if t in c:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * c[t] * (k1 + 1) / (c[t] + k1 * (1 - b + b * dl / (avg or 1.0)))
        out.append(s)
    return out


def score_overlap(query: str, docs: Sequence[str]) -> list[float]:
    """Jaccard overlap of token sets."""
    q = set(tokens(query))
    out = []
    for d in docs:
        s = set(tokens(d))
        out.append(len(q & s) / len(q | s) if q | s else 0.0)
    return out


SCORERS = {"tfidf": score_tfidf, "bm25": score_bm25, "overlap": score_overlap}


# ---------------------------------------------------------------- shared base
class _OwnAnswers(Injector):
    """Shared bank handling, answer listing and a short label (the bank shows as its size)."""
    family = FAMILY
    needs = ("answers",)

    def __init__(self, questions: Sequence[dict] | Mapping[str, Any] | None = None, include_options: bool = True,
                 **params: Any):
        super().__init__(**({"questions": questions} if questions else {}), include_options=include_options,
                         **params)
        self.bank = as_bank(questions)
        self.include_options = include_options

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        shown = {k: (f"{len(v)}q" if k == "questions" else v) for k, v in self.params.items()}
        return f"{self.name}({','.join(f'{k}={v}' for k, v in sorted(shown.items()))})"

    def known(self, person: dict, question: dict) -> list[tuple[str, Any]]:
        """Known answers (leakage rule applied, None/NaN skipped) in bank order; ids missing from the bank follow,
        sorted."""
        ka = {str(k): v for k, v in self.known_answers(person, question).items() if not _missing(v)}
        rank = {qid: i for i, qid in enumerate(self.bank)}
        return sorted(ka.items(), key=lambda kv: (rank.get(kv[0], len(rank)), kv[0]))

    def qdict(self, qid: str) -> dict:
        return self.bank.get(qid) or {"id": qid, "text": qid}

    def listing(self, items: Sequence[tuple[str, Any]]) -> str:
        if not items:
            return NONE_KNOWN
        return "\n\n".join(_qa_text(self.qdict(q), a, self.include_options) for q, a in items)

    @staticmethod
    def ordered(items: list, order: str, seed: int) -> list:
        if order in ("bank", "desc"):
            return list(items)
        if order in ("reverse", "asc"):
            return list(reversed(items))
        if order == "random":
            out = list(items)
            random.Random(seed).shuffle(out)
            return out
        raise ValueError(f"unknown order {order!r}")


def _check_choice(name: str, value: Any, choices: Sequence) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, not {value!r}")


# ---------------------------------------------------------------- injectors
@register
class OwnAnswersAll(_OwnAnswers):
    """Twin-2K-500 digital twin: all non-held-out answers (question, options, answer) in the system prompt."""
    name = "own_answers_all"
    paper = TWIN2K
    description = ("All of the person's known answers (target and exclude dropped) as Q/Options/A text in the "
                   "system prompt, answer-as-this-person framing; order bank|reverse|random, optional max_items.")

    def __init__(self, questions=None, order: str = "bank", max_items: int | None = None, seed: int = 0,
                 **params: Any):
        _check_choice("order", order, ("bank", "reverse", "random"))
        if max_items is not None and max_items < 1:
            raise ValueError("max_items must be >= 1 or None")
        super().__init__(questions=questions, order=order, max_items=max_items, seed=seed, **params)
        self.order, self.max_items, self.seed = order, max_items, seed

    def render(self, person: dict, question: dict) -> Prompt:
        items = self.known(person, question)
        if self.max_items is not None:
            items = items[-self.max_items:]             # truncation keeps the most recent answers
        items = self.ordered(items, self.order, self.seed)
        system = SYSTEM_ALL + "\n\n" + self.listing(items)
        return Prompt(system=system, user=self.question_block(question),
                      meta={"shown": [q for q, _ in items], "n_shown": len(items)})


@register
class OwnAnswersFewshot(_OwnAnswers):
    """k own answers as earlier dialogue turns (in-context demonstrations of this one person)."""
    name = "own_answers_fewshot"
    paper = f"in-context demonstrations (Brown et al. 2020, arXiv:2005.14165); Random/Recency selection: {LAMP}"
    description = ("k known answers as prior turns: user = past question block, assistant = {\"answer\": ...}; "
                   "select random|recent|first (bank order = time), order bank|reverse|random.")

    def __init__(self, k: int = 8, questions=None, select: str = "random", order: str = "bank", seed: int = 0,
                 **params: Any):
        _check_choice("select", select, ("random", "recent", "first"))
        _check_choice("order", order, ("bank", "reverse", "random"))
        if k < 1:
            raise ValueError("k must be >= 1")
        super().__init__(questions=questions, k=k, select=select, order=order, seed=seed, **params)
        self.k, self.select, self.order, self.seed = k, select, order, seed

    def render(self, person: dict, question: dict) -> Prompt:
        items = self.known(person, question)
        if self.select == "recent":
            picked = items[-self.k:]
        elif self.select == "first":
            picked = items[:self.k]
        else:   # k smallest hash(seed, person, qid): stable across targets (only the target itself drops out)
            key = {q: hashlib.sha256(f"{self.seed}:{person.get('id')}:{q}".encode()).hexdigest() for q, _ in items}
            keep = set(sorted(key, key=key.get)[:self.k])
            picked = [it for it in items if it[0] in keep]
        picked = self.ordered(picked, self.order, self.seed)
        history: list[tuple[str, str]] = []
        for qid, a in picked:
            q = self.qdict(qid)
            q_shown = q if self.include_options else {k: v for k, v in q.items() if k != "options"}
            reply = json.dumps({"answer": a}, ensure_ascii=False, default=_json_default)
            history += [("user", self.question_block(q_shown)), ("assistant", reply)]
        system = SYSTEM_FEWSHOT if picked else SYSTEM_FEWSHOT + "\n" + NONE_KNOWN
        return Prompt(system=system, user=self.question_block(question), history=history,
                      meta={"shown": [q for q, _ in picked], "n_shown": len(picked)})


@register
class OwnAnswersRag(_OwnAnswers):
    """Top-k own answers retrieved by lexical similarity of their questions to the target question."""
    name = "own_answers_rag"
    paper = f"{HWANG}; retriever and k ablation: {LAMP}"
    description = ("k known answers whose question text is most similar to the target (scorer tfidf|bm25|"
                   "overlap, match question|question_answer), listed desc (most similar first), asc or bank order; "
                   "needs a questions= bank.")

    def __init__(self, k: int = 8, questions=None, scorer: str = "tfidf", order: str = "desc",
                 match: str = "question", **params: Any):
        _check_choice("scorer", scorer, tuple(SCORERS))
        _check_choice("order", order, ("desc", "asc", "bank"))
        _check_choice("match", match, ("question", "question_answer"))
        if k < 1:
            raise ValueError("k must be >= 1")
        if not questions:
            raise ValueError("own_answers_rag needs questions= (the question bank) to retrieve past answers by text")
        super().__init__(questions=questions, k=k, scorer=scorer, order=order, match=match, **params)
        self.k, self.scorer, self.order, self.match = k, scorer, order, match

    def retrieve(self, person: dict, question: dict) -> list[tuple[str, Any, float]]:
        """[(qid, answer, score)] most similar first; ties keep bank order (deterministic)."""
        items = self.known(person, question)
        if not items:
            return []
        docs = [self.qdict(q)["text"] + (f" {a}" if self.match == "question_answer" else "") for q, a in items]
        scores = SCORERS[self.scorer](question["text"], docs)
        ranked = sorted(range(len(items)), key=lambda i: (-scores[i], i))[:self.k]
        return [(items[i][0], items[i][1], scores[i]) for i in ranked]

    def render(self, person: dict, question: dict) -> Prompt:
        hits = self.retrieve(person, question)
        if self.order == "bank":
            rank = {q: i for i, (q, _) in enumerate(self.known(person, question))}
            hits = sorted(hits, key=lambda h: rank[h[0]])
        elif self.order == "asc":
            hits = hits[::-1]                           # most similar last, next to the question
        system = SYSTEM_RAG + "\n\n" + self.listing([(q, a) for q, a, _ in hits])
        return Prompt(system=system, user=self.question_block(question),
                      meta={"shown": [q for q, _, _ in hits], "n_shown": len(hits),
                            "scores": [round(s, 4) for _, _, s in hits]})


__all__ = ["OwnAnswersAll", "OwnAnswersFewshot", "OwnAnswersRag", "SCORERS", "as_bank", "tokens",
           "score_tfidf", "score_bm25", "score_overlap"]
