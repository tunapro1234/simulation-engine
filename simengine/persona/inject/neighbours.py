"""Neighbour injectors: answers of similar *other* people in the prompt (kNN / collaborative-filtering style).

Every injector here takes a `pool` of person dicts at construction; the target person (and any id in the
question's optional "exclude_people") is never in the neighbour set (matched by id) and similarity is computed
from the target's fields and `known_answers` only, so the target's own answer to the target question is never
shown nor used. Question ids and answer keys compare as strings, as in `known_answers`.

    import simengine.persona.inject as pi
    inj = pi.get_injector("similar_respondents", pool=people, k=5)          # neighbours' profiles + answers
    p = inj(people[0], {"id": "q1", "text": "Do you own a car?", "options": ["yes", "no"]})
    llm.complete(p.system, p.flatten())
    pi.get_injector("neighbour_votes", pool=people, k=10)                  # "people like you: yes 7, no 3"
    pi.get_injector("answer_neighbours", pool=people, questions=questions)  # CF over answer vectors
    pi.compare_injectors(llm, people, questions, "answers", [("neighbour_votes", {"pool": people})])

Neighbours are shown as "Participant 1..k", never by id. Similarity is a Gower-style mean over the target's
keys: categorical 1/0 match, numeric 1 - |a - b| / range (range over the pool), a key the neighbour lacks
counts 0. Ties are broken by a hash of (seed, person, question, neighbour), so `seed` is the only randomness.
"""
from __future__ import annotations

import hashlib
import json
import random
from statistics import median
from typing import Any, Sequence

from ._base import Injector, MissingPersonField, Prompt, register

BASES = ("fields", "answers", "both")
STRATEGIES = ("most_similar", "least_similar", "random")
FORMATS = ("single", "multi")
SHOWS = ("counts", "share")
_SUMMARISED = ("pool", "questions")   # long params: the arm label shows size + digest, not the content


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt(v: Any) -> str:
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def _skeys(d: Any) -> dict[str, Any]:
    """Dict with str keys (question and field ids compare as strings, as in `known_answers`)."""
    return {str(k): v for k, v in (d or {}).items()}


def _digest(v: Any) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:6]


class _Neighbours(Injector):
    """Shared machinery: pool index, similarity, ranking. Not registered."""
    family = "neighbours"

    def __init__(self, pool: Sequence[dict] | None = None, *, k: int, basis: str = "fields",
                 keys: Sequence[str] | None = None, seed: int = 0, **params: Any):
        if not pool:
            raise ValueError(f"injector {self.name!r} needs pool=[person, ...] (the other people to draw "
                             "neighbours from)")
        if basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}, not {basis!r}")
        if int(k) < 1:
            raise ValueError("k must be >= 1")
        given = {"basis": basis} if basis != "fields" else {}
        given.update({"keys": list(keys)} if keys else {}, **({"seed": seed} if seed else {}))
        super().__init__(pool=pool, k=k, **given, **params)
        self.pool = list(pool)
        ids = [str(p["id"]) for p in self.pool]
        if len(set(ids)) != len(ids):
            raise ValueError("pool person ids must be unique")
        self.k, self.basis, self.seed = int(k), basis, seed
        self.keys = list(keys) if keys else None
        self.needs = {"fields": ("fields",), "answers": ("answers",), "both": ("fields", "answers")}[basis]
        self._digests = {"pool": _digest(self.pool)}
        self._ranges = self._numeric_ranges()

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        parts = []
        for key, v in sorted(self.params.items()):
            if key in _SUMMARISED:
                d = self._digests.get(key) or _digest(v)
                parts.append(f"{key}=n{len(v)}#{d}")
            else:
                parts.append(f"{key}={v}")
        return f"{self.name}({','.join(parts)})"

    # ------------------------------------------------------------ vectors and similarity
    def _numeric_ranges(self) -> dict[tuple[str, str], float]:
        vals: dict[tuple[str, str], list[float]] = {}
        for p in self.pool:
            for kind, d in (("f", self.person_fields(p)), ("a", p.get("answers") or {})):
                for key, v in d.items():
                    x = _num(v)
                    if x is not None:
                        vals.setdefault((kind, str(key)), []).append(x)
        return {key: max(xs) - min(xs) for key, xs in vals.items() if len(xs) > 1}

    def _fields(self, person: dict) -> dict:
        f = self.person_fields(person)
        keys = self.keys if self.keys is not None else list(f)
        return {k: f[k] for k in keys if k in f and f[k] not in (None, "")}

    def _vector(self, person: dict, question: dict) -> dict[tuple[str, str], Any]:
        """Target vector: fields and/or known answers (never the target question or `exclude` ids)."""
        vec: dict[tuple[str, str], Any] = {}
        if self.basis in ("fields", "both"):
            vec.update({("f", str(k)): v for k, v in self._fields(person).items()})
        if self.basis in ("answers", "both"):
            vec.update({("a", str(k)): v for k, v in self.known_answers(person, question).items()})
        if not vec:
            raise MissingPersonField(f"injector {self.name!r} (basis={self.basis!r}) found nothing to match on for "
                                     f"person {person.get('id')!r}: needs {list(self.needs)} other than the target "
                                     "question and its exclude ids")
        return vec

    def _sim1(self, key: tuple[str, str], a: Any, b: Any) -> float:
        xa, xb = _num(a), _num(b)
        if xa is not None and xb is not None:
            r = self._ranges.get(key)
            if not r:
                return float(xa == xb)
            return max(0.0, 1.0 - abs(xa - xb) / r)
        return float(str(a).strip().lower() == str(b).strip().lower())

    def similarity(self, vec: dict[tuple[str, str], Any], other: dict) -> tuple[float, int]:
        """(mean similarity over the target's keys, number of keys the other person has)."""
        f, a = _skeys(self.person_fields(other)), _skeys(other.get("answers"))
        total, overlap = 0.0, 0
        for key, v in vec.items():
            src = f if key[0] == "f" else a
            if key[1] in src and src[key[1]] not in (None, ""):
                total += self._sim1(key, v, src[key[1]])
                overlap += 1
        return total / len(vec), overlap

    def _tie(self, person: dict, question: dict, other: dict) -> str:
        return hashlib.sha256(f"{self.seed}|{person.get('id')}|{question.get('id')}|{other['id']}".encode()).hexdigest()

    def candidates(self, person: dict, question: dict, min_overlap: int = 1) -> list[tuple[float, dict]]:
        """Pool members other than the target (and question["exclude_people"]) who answered the target question,
        with similarity, most similar first (ties by seeded hash)."""
        skip = {str(person.get("id"))} | {str(x) for x in question.get("exclude_people") or ()}
        vec = self._vector(person, question)
        out = []
        for o in self.pool:
            if str(o["id"]) in skip:
                continue                                   # leakage guard: never the target person
            ans = self._answer(o, question)
            if ans in (None, ""):
                continue                                   # complete cases on the target item only
            s, ov = self.similarity(vec, o)
            if ov >= min_overlap:
                out.append((s, o))
        out.sort(key=lambda t: (-t[0], self._tie(person, question, t[1])))
        return out

    @staticmethod
    def _answer(other: dict, question: dict) -> Any:
        qid = question.get("id")
        return None if qid is None else _skeys(other.get("answers")).get(str(qid))

    def _profile(self, person: dict) -> str:
        return "; ".join(f"{str(k).replace('_', ' ')}: {v}" for k, v in self._fields(person).items())


@register
class SimilarRespondents(_Neighbours):
    """k retrieved respondents (profile + their answer to this question) as in-context examples, then the
    target's profile. Holtdirk et al. 2026's ICL imputation (their best spec: single-turn, most similar, no
    stratification); EmbeddingGemma cosine retrieval replaced by Gower similarity, the 'diverse' (k-medoids)
    strategy and the verbalized-distribution output are not implemented."""
    name = "similar_respondents"
    paper = "Holtdirk, Ahnert, Sakshaug & Haensch 2026, ICL for imputing public opinion data (arXiv:2606.09351)"
    needs = ("fields",)
    description = ("k other respondents who answered this question (most/least similar or random, optional "
                   "stratification by answer) shown with their attributes and answer; single- or multi-turn.")

    def __init__(self, pool: Sequence[dict] | None = None, k: int = 10, strategy: str = "most_similar",
                 stratify: bool = False, format: str = "single", **params: Any):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, not {strategy!r}")
        if format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, not {format!r}")
        super().__init__(pool, k=k, strategy=strategy, stratify=stratify, format=format, **params)
        self.strategy, self.stratify, self.format = strategy, bool(stratify), format

    def _rank(self, person: dict, question: dict, cands: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
        if self.strategy == "least_similar":
            return list(reversed(cands))
        if self.strategy == "random":
            rng = random.Random(f"{self.seed}|{person.get('id')}|{question.get('id')}")
            cands = cands[:]
            rng.shuffle(cands)
        return cands

    def select(self, person: dict, question: dict) -> list[tuple[float, dict]]:
        ranked = self._rank(person, question, self.candidates(person, question))
        opts = [str(o) for o in question.get("options") or ()]
        if not (self.stratify and opts):
            return ranked[:self.k]
        per, extra = divmod(self.k, len(opts))       # budget split evenly over options (remainder to the first)
        out = []
        for i, opt in enumerate(opts):
            hits = [c for c in ranked if str(self._answer(c[1], question)).strip().lower() == opt.lower()]
            out += hits[:per + (1 if i < extra else 0)]
        order = {id(c[1]): j for j, c in enumerate(ranked)}
        return sorted(out, key=lambda c: order[id(c[1])])

    def render(self, person: dict, question: dict) -> Prompt:
        chosen = self.select(person, question)
        how = {"most_similar": "to be most similar to", "least_similar": "to be least similar to",
               "random": "at random, not by similarity to"}[self.strategy]
        system = "Please perform a classification task: predict how a survey participant answers."   # paper wording
        target = f"Here is the participant to be classified: {self._profile(person) or '(no attributes)'}"
        meta = {"neighbours": [str(o["id"]) for _, o in chosen], "similarity": [round(s, 4) for s, _ in chosen]}
        if self.format == "multi":
            hist: list[tuple[str, str]] = []
            for _, o in chosen:
                hist.append(("user", f"Participant: {self._profile(o)}\n\n{self.question_block(question)}"))
                hist.append(("assistant", json.dumps({"answer": self._answer(o, question)})))
            return Prompt(system=system, user=f"{target}\n\n{self.question_block(question)}", history=hist,
                          meta=meta)
        lines = [f"Here is a selection of {len(chosen)} survey participants. They are selected {how} the "
                 "participant to be classified."]
        for i, (_, o) in enumerate(chosen, 1):
            lines.append(f"Participant {i}: {self._profile(o)}. {question['text']} -> {_fmt(self._answer(o, question))}")
        if not chosen:
            lines = ["No other participant answered this question."]
        return Prompt(system=system, user="\n".join(lines) + f"\n\n{target}\n\n{self.question_block(question)}",
                      meta=meta)


@register
class NeighbourVotes(_Neighbours):
    """Persona + an aggregate of the k nearest neighbours' answers to this question ("people like you answered:
    yes 7, no 3"): user-based kNN collaborative filtering put in the prompt as evidence, cf. CoRAL."""
    name = "neighbour_votes"
    paper = ("user-based kNN CF (Resnick et al. 1994, GroupLens) as prompt evidence, cf. Wu et al. 2024 CoRAL "
             "(arXiv:2403.06447)")
    needs = ("fields",)
    description = ("Persona attributes + counts (or shares) of the k most similar other people's answers to "
                   "this question; numbers as sorted values and median.")

    def __init__(self, pool: Sequence[dict] | None = None, k: int = 10, show: str = "counts", **params: Any):
        if show not in SHOWS:
            raise ValueError(f"show must be one of {SHOWS}, not {show!r}")
        super().__init__(pool, k=k, show=show, **params)
        self.show = show

    def tally(self, person: dict, question: dict) -> tuple[list[tuple[float, dict]], str]:
        chosen = self.candidates(person, question)[:self.k]
        vals = [self._answer(o, question) for _, o in chosen]
        if not vals:
            return chosen, "none of them answered this question"
        opts = [str(o) for o in question.get("options") or ()]
        if not opts:
            xs = sorted(x for x in (_num(v) for v in vals) if x is not None)
            if xs:
                return chosen, f"{', '.join(_fmt(x) for x in xs)} (median {_fmt(float(median(xs)))})"
            opts = sorted({str(v) for v in vals})
        counts = {o: 0 for o in opts}
        other = 0
        for v in vals:
            hit = next((o for o in opts if o.lower() == str(v).strip().lower()), None)
            if hit is None:
                other += 1
            else:
                counts[hit] += 1
        if other:
            counts["other"] = other
        n = len(vals)
        if self.show == "share":
            return chosen, ", ".join(f"{o} {round(100 * c / n)}%" for o, c in counts.items()) + f" (n={n})"
        return chosen, ", ".join(f"{o} {c}" for o, c in counts.items())

    def render(self, person: dict, question: dict) -> Prompt:
        chosen, summary = self.tally(person, question)
        on = {"fields": "attributes", "answers": "earlier answers", "both": "attributes and earlier answers"}
        prof = self._profile(person)
        lines = [f"You are a person with these attributes: {prof}." if prof else "You are a survey participant."]
        lines.append(f"Among the {len(chosen)} other people most similar to you in {on[self.basis]}, the answers to "
                     f"the next question were: {summary}.")
        lines.append("Answer as this person would; the others are evidence, not a rule.")
        return Prompt(system="\n".join(lines), user=self.question_block(question),
                      meta={"neighbours": [str(o["id"]) for _, o in chosen]})


@register
class AnswerNeighbours(_Neighbours):
    """Collaborative filtering over answer vectors: the target's own earlier answers, then the k people who
    answered most like them on those items, with their answers there and on this question."""
    name = "answer_neighbours"
    paper = ("similar users' histories in context: Shi et al. 2025 CFRAG (arXiv:2504.05731); Wu et al. 2024 CoRAL "
             "(arXiv:2403.06447)")
    needs = ("answers",)
    description = ("Target's known answers + k nearest people by answer agreement (min_overlap shared items), "
                   "each with their answers on the shared items and on this question.")

    def __init__(self, pool: Sequence[dict] | None = None, k: int = 5, min_overlap: int = 2,
                 questions: Sequence[dict] | dict | None = None, max_items: int = 20, **params: Any):
        if questions is not None:
            params["questions"] = questions
        super().__init__(pool, k=k, basis="answers", min_overlap=min_overlap, max_items=max_items, **params)
        self.params.pop("basis", None)                 # fixed for this injector, not a caller param
        self.min_overlap, self.max_items = int(min_overlap), int(max_items)
        if isinstance(questions, dict):
            self.qtext = {str(k): str(v) for k, v in questions.items()}
        else:
            self.qtext = {str(q["id"]): str(q.get("text", q["id"])) for q in questions or () if "id" in q}
        if questions is not None:
            self._digests["questions"] = _digest(questions)

    def _item(self, qid: Any) -> str:
        return self.qtext.get(str(qid), str(qid))

    def render(self, person: dict, question: dict) -> Prompt:
        own = self.known_answers(person, question)
        items = sorted(own, key=str)[:self.max_items]
        chosen = self.candidates(person, question, min_overlap=self.min_overlap)[:self.k]
        lines = ["You are a survey participant. Your earlier answers:"]
        lines += [f"- {self._item(i)}: {_fmt(own[i])}" for i in items]
        if chosen:
            lines.append(f"\n{len(chosen)} other participants who answered most like you:")
            for n, (s, o) in enumerate(chosen, 1):
                oa = _skeys(o.get("answers"))
                shared = [i for i in items if oa.get(str(i)) not in (None, "")]
                parts = "; ".join(f"{self._item(i)}: {_fmt(oa[str(i)])}" for i in shared)
                lines.append(f"Participant {n} (agreement {round(100 * s)}%): {parts}; "
                             f"this question: {_fmt(self._answer(o, question))}")
        else:
            lines.append(f"\nNo other participant shares at least {self.min_overlap} answered items with you.")
        lines.append("\nAnswer as yourself; the others are evidence, not a rule.")
        return Prompt(system="\n".join(lines), user=self.question_block(question),
                      meta={"neighbours": [str(o["id"]) for _, o in chosen]})


__all__ = ["SimilarRespondents", "NeighbourVotes", "AnswerNeighbours", "BASES", "STRATEGIES", "FORMATS", "SHOWS"]
