"""Social-context injectors: what the person's network, peers and media say, put next to the persona (family "social").

Five variants from the literature:
  peer_messages      neighbours' posts as read messages, optional confirmation-bias and closed-world framing
                     (Chuang et al. 2023, arXiv:2311.09618)
  neighbor_activity  recent actions of followed accounts (reposted, adopted, ...) plus an optional exposure count
                     "n of your m contacts have ..." (S3, Gao et al. 2023, arXiv:2307.14984)
  asch_majority      Asch-style confederates: earlier participants' answers to the same question, in a tone, or
                     distilled to one summary line (Zhu et al. 2024, arXiv:2410.12428)
  descriptive_norm   prevalence of each answer among a reference group, "approximately 62% chose ..." or
                     "most people ... (62%)" (Peng et al. 2026, arXiv:2603.20911)
  news_feed          media items as relevant memories in the system prompt, e.g. a newspaper prevalence figure
                     (Williams et al. 2023, arXiv:2307.04986)

Person fields read here: `neighbors` [{id, label, answers: {qid: answer}, message, action, time}] (a bare string is
a message; oldest first), `norms` {qid: {option: share or count} | number}, `media` [str | {source, text, time}],
`state` (str, or {qid: value}: the person's PRIOR view, e.g. last wave or last ABM step, never the target answer),
plus `fields` / `text` for the optional persona block. A neighbour with the person's own id is dropped.

Bridges for simengine.abm (pure Python): `attach_neighbors` (edges -> neighbors), `attach_norms` (leave-one-out
answer shares per group), `norm_from_dist` (a K-category distribution such as abm.opinion's -> {option: share}),
`prevalence_item` (an epidemic / adoption share -> a news item).

    import simengine.persona.inject as pi
    from simengine.persona.inject.social import attach_neighbors, attach_norms
    people = attach_neighbors(people, [("u1", "u2"), ("u2", "u3")])
    p = pi.get_injector("asch_majority", tone="confident")(people[0], question)
    people = attach_norms(people, questions, by="region")
    pi.compare_injectors(llm, people, questions, "answers", ["descriptive_norm", ("peer_messages", {"bias": "weak"})])
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from ...games._answer import instruction
from ._base import Injector, MissingPersonField, Prompt, answer_spec, register

FAMILY = "social"
CHUANG = "Chuang et al. 2023, Simulating Opinion Dynamics with Networks of LLM-based Agents (arXiv:2311.09618)"
GAO = "Gao et al. 2023, S3: Social-network Simulation System with LLM-Empowered Agents (arXiv:2307.14984)"
ZHU = "Zhu et al. 2024, Conformity in Large Language Models (arXiv:2410.12428)"
PENG = ("Peng et al. 2026, Do LLM-Driven Agents Exhibit Engagement Mechanisms? Information Load, Descriptive Norms "
        "and Popularity Cues (arXiv:2603.20911)")
WILLIAMS = "Williams et al. 2023, Epidemic Modeling with Generative Agents (arXiv:2307.04986)"

SELECT = ("recent", "random")
# never shown as persona attributes: social structure, and the person's own current message / action (an ABM step
# writes those onto the person dict for attach_neighbors; shown as an attribute they can be the target answer)
SOCIAL_KEYS = ("neighbors", "norms", "media", "state", "group", "label", "message", "action", "time")
# Chuang et al. 2023: Appendix F (bias, verbatim) and G (closed world, verbatim but "XYZ" -> "this topic" and
# "other strangers on Twitter" -> "the people you follow")
BIAS = {
    "none": "",
    "weak": ("Remember, you are role-playing as a real person. Like humans, you have confirmation bias. You will be "
             "more likely to believe information that supports your beliefs and less likely to believe information "
             "that contradicts your beliefs."),
    "strong": ("Remember, you are role-playing as a real person. You have a strong confirmation bias. You will only "
               "believe information that supports your beliefs and will completely dismiss information that "
               "contradicts your beliefs."),
}
CLOSED_WORLD = ("Remember, throughout the interactions, you are alone in your room with limited access to the "
                "Internet. You cannot search for information about this topic on the Internet. You can not go out to "
                "ask other people about this topic. Because you are alone in your room, you can not leave your room "
                "to seek information about this topic. To form your belief about this topic, you can only rely on "
                "your initial belief about this topic, along with the information you received from the people you "
                "follow.")
# Zhu et al. 2024, Table 1: the published example phrasings per tone; participant i uses variant i mod len
TONES = {"plain": ("{a}",),
         "neutral": ("I think it is {a}", "My answer is {a}", "{a}, in my opinion", "It's {a}"),
         "confident": ("I am sure it is {a}", "{a}, of course", "Sure thing it's {a}"),
         "uncertain": ("I am not sure if it's {a}", "I guess it's {a}", "{a}? perhaps")}


# ---------------------------------------------------------------- helpers
def _blank(v: Any) -> bool:
    """None, "", NaN (float / numpy) or pandas NA / NaT: no value."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return type(v).__name__ in ("NAType", "NaTType")


def _check_k(k: int | None) -> None:
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
        raise ValueError(f"k must be a positive int or None (all), not {k!r}")


def _rng(seed: int, person: dict, question: dict) -> random.Random:
    # str seed -> sha512 in Random.seed: stable across runs and processes
    return random.Random(f"{seed}|{person.get('id')}|{question.get('id')}")


def neighbors_of(person: dict) -> list[dict]:
    """person['neighbors'] as fresh dicts (a string is a message), oldest first, without the person themself."""
    pid = str(person.get("id"))
    out = []
    for n in person.get("neighbors") or ():
        n = {"message": n} if isinstance(n, str) else dict(n)
        if n.get("id") is not None and str(n["id"]) == pid:
            continue                                # self-loop would show the person's own answer
        out.append(n)
    return out


def _pick(items: list, k: int | None, select: str, rng: random.Random) -> list:
    """k items in oldest-first order (by `time` when every item has one, else list order): the last k (recent) or
    a seeded sample (random)."""
    if items and all(i.get("time") is not None for i in items):
        items = sorted(items, key=lambda i: i["time"])          # stable: equal times keep list order
    if k is None or len(items) <= k:
        return list(items)
    if select == "recent":
        return list(items[-k:])
    idx = sorted(rng.sample(range(len(items)), k))
    return [items[i] for i in idx]


def _persona_lines(inj: Injector, person: dict) -> list[str]:
    """Optional persona block: the narrative text and the attribute lines, when present."""
    out = []
    if not _blank(person.get("text")):
        out.append(str(person["text"]))
    f = inj.person_fields(person)
    attrs = [f"- {k.replace('_', ' ')}: {v}" for k, v in f.items() if k not in SOCIAL_KEYS and not _blank(v)]
    if attrs:
        out.append("You are a person with these attributes:\n" + "\n".join(attrs))
    return out


def _state(person: dict, question: dict) -> Any:
    s = person.get("state")
    if isinstance(s, Mapping):
        qid = question.get("id")
        s = s[qid] if qid in s else s.get(str(qid))
    return None if _blank(s) else s


def _neighbor_answer(n: dict, question: dict) -> Any:
    ans = n.get("answers") or {}
    qid = question.get("id")
    v = ans[qid] if qid in ans else ans.get(str(qid))
    return None if _blank(v) else v


def _quote(v: Any) -> str:
    return f'"{v}"'


def _pct(x: float, decimals: int) -> str:
    return f"{100 * x:.{decimals}f}%"


def _ordered(counts: Mapping[Any, float], question: dict) -> list[tuple[Any, float]]:
    """(option, value) in the question's option order, then any other keys sorted by text."""
    opts = [str(o) for o in question.get("options") or ()]
    by = {str(k): v for k, v in counts.items()}
    rest = sorted(k for k in by if k not in opts)
    return [(k, by[k]) for k in opts + rest if k in by]


def _shares(counts: Mapping[Any, float], question: dict) -> list[tuple[Any, float]]:
    rows = [(k, float(v)) for k, v in _ordered(counts, question) if not _blank(v) and float(v) > 0]
    tot = sum(v for _, v in rows)
    if tot <= 0:
        return []
    return [(k, v / tot) for k, v in rows]


def _missing(inj: Injector, person: dict, what: str) -> MissingPersonField:
    return MissingPersonField(f"injector {inj.name!r} needs {what} (needs={list(inj.needs)}; person "
                              f"{person.get('id')!r})")


class _Social(Injector):
    family = FAMILY

    def _system(self, person: dict, *parts: str, lead: str = "") -> str:
        persona = _persona_lines(self, person) if getattr(self, "persona", False) else []
        return "\n\n".join([lead] * bool(lead) + persona + [p for p in parts if p])


# ---------------------------------------------------------------- injectors
@register
class PeerMessages(_Social):
    """Neighbours' posts shown as messages the person has read (cumulative memory), as in Chuang et al. 2023."""
    name = "peer_messages"
    paper = CHUANG
    needs = ("neighbors",)
    description = ("System: 'Role play this person.' + optional persona + 'you have been reading posts from people "
                   "you follow' + optional "
                   "confirmation-bias (weak/strong) and closed-world lines; user: the k neighbour messages "
                   "(recent or seeded random), then the question. Reads neighbors[].message, optional state, fields, text.")

    def __init__(self, k: int | None = 5, bias: str = "none", closed_world: bool = True, select: str = "recent",
                 persona: bool = True, seed: int = 0, **params: Any):
        if bias not in BIAS:
            raise ValueError(f"bias must be one of {sorted(BIAS)}")
        if select not in SELECT:
            raise ValueError(f"select must be one of {SELECT}")
        _check_k(k)
        super().__init__(k=k, bias=bias, closed_world=closed_world, select=select, persona=persona, seed=seed,
                         **params)
        self.k, self.bias, self.closed_world, self.select = k, bias, closed_world, select
        self.persona, self.seed = persona, seed

    def render(self, person: dict, question: dict) -> Prompt:
        msgs = [n for n in neighbors_of(person) if not _blank(n.get("message"))]
        if not msgs:
            raise _missing(self, person, "at least one neighbour with a 'message'")
        shown = _pick(msgs, self.k, self.select, _rng(self.seed, person, question))
        st = _state(person, question)
        system = self._system(
            person,
            "Now, you have been reading posts from people you follow on a social network. You can decide to change "
            "or maintain your belief about this topic after reading them.",
            f"Your initial belief: {st}" if st is not None else "",
            BIAS[self.bias], CLOSED_WORLD if self.closed_world else "", lead="Role play this person.")
        lines = [f"- {n.get('label') or 'Someone you follow'}: {_quote(n['message'])}" for n in shown]
        user = "Posts you have read (oldest first):\n" + "\n".join(lines) + "\n\n" + self.question_block(question)
        return Prompt(system=system, user=user, meta={"shown": [n.get("id") for n in shown]})


@register
class NeighborActivity(_Social):
    """Followed accounts' recent actions (S3: demographics + current state + received posts) with an optional
    exposure count line for one action, the input a threshold / complex-contagion model works with."""
    name = "neighbor_activity"
    paper = GAO
    needs = ("neighbors",)
    description = ("System: optional persona + current state + 'below is recent activity from accounts you follow'; "
                   "user: '- <who> <action>: \"<message>\"' lines (k, recent or seeded random) and, with action=, "
                   "'n of your m contacts have <action>.' Reads neighbors[].action/message, optional state, fields, text.")

    def __init__(self, k: int | None = 10, action: str | None = None, select: str = "recent",
                 persona: bool = True, seed: int = 0, **params: Any):
        if select not in SELECT:
            raise ValueError(f"select must be one of {SELECT}")
        _check_k(k)
        super().__init__(k=k, action=action, select=select, persona=persona, seed=seed, **params)
        self.k, self.action, self.select, self.persona, self.seed = k, action, select, persona, seed

    def render(self, person: dict, question: dict) -> Prompt:
        nb = neighbors_of(person)
        acts = [n for n in nb if not (_blank(n.get("action")) and _blank(n.get("message")))]
        if not acts:
            raise _missing(self, person, "at least one neighbour with an 'action' or 'message'")
        shown = _pick(acts, self.k, self.select, _rng(self.seed, person, question))
        st = _state(person, question)
        system = self._system(person,
                              f"Your current attitude: {st}" if st is not None else "",
                              "You use a social network. Below is recent activity from accounts you follow.")
        lines = []
        for n in shown:
            who = n.get("label") or "Someone you follow"
            verb = "posted" if _blank(n.get("action")) else n["action"]
            lines.append(f"- {who} {verb}" if _blank(n.get("message")) else f"- {who} {verb}: {_quote(n['message'])}")
        user = "Recent activity from accounts you follow (oldest first):\n" + "\n".join(lines)
        meta = {"shown": [n.get("id") for n in shown]}
        if self.action:
            hit = sum(1 for n in nb if n.get("action") == self.action)
            user += f"\n\n{hit} of your {len(nb)} contacts have {self.action}."
            meta["exposed"] = (hit, len(nb))
        return Prompt(system=system, user=user + "\n\n" + self.question_block(question), meta=meta)


@register
class AschMajority(_Social):
    """Asch-paradigm conformity prompt (Zhu et al. 2024): the person is the last of N+1 participants; the earlier
    ones' answers come from neighbours' answers to the same question, or from a fixed `answer` repeated `n` times
    (the paper's unanimous-confederate manipulation)."""
    name = "asch_majority"
    paper = ZHU
    needs = ("neighbors",)
    description = ("User: Zhu et al.'s template ('You are one of the participants in an experiment. There are in "
                   "total N+1 participants. ...', 'Organiser: ...', 'Question: ...'), 'Participant i: <tone>' lines "
                   "(plain / neutral / confident / uncertain, the paper's example phrasings in turn) or one distilled "
                   "organiser line, then 'Now, as Participant N+1, it's your turn:'. Confederate "
                   "answers = neighbors[].answers[qid] (seeded order) or a fixed answer= x n (then needs nothing). "
                   "persona=True adds fields/text.")

    def __init__(self, tone: str = "neutral", distill: bool = False, answer: Any = None, n: int = 5,
                 k: int | None = None, persona: bool = False, seed: int = 0, **params: Any):
        if tone not in TONES:
            raise ValueError(f"tone must be one of {sorted(TONES)}")
        _check_k(k)
        if answer is not None and n < 1:
            raise ValueError("n (number of confederates) must be >= 1")
        super().__init__(tone=tone, distill=distill, answer=answer, n=n, k=k, persona=persona, seed=seed, **params)
        self.tone, self.distill, self.answer, self.n, self.k = tone, distill, answer, n, k
        self.persona, self.seed = persona, seed
        if answer is not None:
            self.needs = ()

    def _confederates(self, person: dict, question: dict) -> tuple[list[Any], list[Any]]:
        if self.answer is not None:
            opts = [str(o) for o in question.get("options") or ()]
            if opts and str(self.answer) not in opts:
                raise ValueError(f"answer {self.answer!r} is not an option of question {question.get('id')!r}")
            return [self.answer] * self.n, []
        rng = _rng(self.seed, person, question)
        pairs = [(n.get("id"), a) for n in neighbors_of(person) if (a := _neighbor_answer(n, question)) is not None]
        if not pairs:
            raise _missing(self, person, f"neighbours with answers[{question.get('id')!r}]")
        if self.k is not None and len(pairs) > self.k:
            pairs = rng.sample(pairs, self.k)
        rng.shuffle(pairs)
        return [a for _, a in pairs], [i for i, _ in pairs]

    def render(self, person: dict, question: dict) -> Prompt:
        answers, ids = self._confederates(person, question)
        m = len(answers)
        organiser = ("Answer the following multiple choice question by giving the most appropriate response. "
                     "Answer should be chosen among [" + ", ".join(str(o) for o in question["options"]) + "]."
                     if question.get("options") else
                     "Answer the following question by giving the most appropriate response.")
        head = (f"You are one of the participants in an experiment. There are in total {m + 1} participants. Answer "
                f"the questions based on your knowledge. The conversation of the whole experiment is as follows:\n\n"
                f"Organiser: {organiser}\n\nQuestion: {question['text']}")
        if self.distill:
            cnt = Counter(str(a) for a in answers)
            if len(cnt) == 1:
                body = f"Organiser: All participants before you have chosen {_quote(answers[0])} as the answer."
            else:
                rows = sorted(_ordered(cnt, question), key=lambda kv: -kv[1])   # stable: ties keep option order
                body = (f"Organiser: Of the {m} participants before you, "
                        + ", ".join(f"{c} chose {_quote(a)}" for a, c in rows) + ".")
        else:
            say = TONES[self.tone]
            body = "\n".join(f"Participant {i + 1}: {say[i % len(say)].format(a=a)}" for i, a in enumerate(answers))
        user = (f"{head}\n\n{body}\n\nNow, as Participant {m + 1}, it's your turn:\n"
                f"{instruction(answer_spec(question))}")
        return Prompt(system=self._system(person), user=user,
                      meta={"confederates": [str(a) for a in answers], "shown": ids})


@register
class DescriptiveNorm(_Social):
    """Descriptive norm as answer prevalence in a reference group, stated in the system prompt (Peng et al. 2026
    gave approximate percentages of each engagement form there, all options still available). style="most" is the
    common 'most people ...' norm message; it is not Peng et al.'s wording."""
    name = "descriptive_norm"
    paper = PENG
    needs = ("norms",)
    description = ("System: optional persona + 'Among <group>, approximately 62% chose \"yes\" and 38% chose \"no\".' "
                   "(shares) or 'Most <group> (62%) chose \"yes\".' (most); numbers: the group mean. Norm from "
                   "person['norms'][qid] or, with source='neighbors', the neighbours' answers. group may hold "
                   "{field} placeholders filled from the person's fields.")

    def __init__(self, source: str = "norms", group: str = "people in your community", style: str = "shares",
                 decimals: int = 0, persona: bool = True, **params: Any):
        if source not in ("norms", "neighbors"):
            raise ValueError("source must be 'norms' or 'neighbors'")
        if style not in ("shares", "most"):
            raise ValueError("style must be 'shares' or 'most'")
        super().__init__(source=source, group=group, style=style, decimals=decimals, persona=persona, **params)
        self.source, self.group, self.style, self.decimals, self.persona = source, group, style, decimals, persona
        if source == "neighbors":
            self.needs = ("neighbors",)

    def _norm(self, person: dict, question: dict) -> Any:
        qid = question.get("id")
        if self.source == "norms":
            norms = person.get("norms") or {}
            v = norms[qid] if qid in norms else norms.get(str(qid))
            if _blank(v) or (isinstance(v, Mapping) and not v):
                raise _missing(self, person, f"norms[{qid!r}]")
            return v
        vals = [a for n in neighbors_of(person) if (a := _neighbor_answer(n, question)) is not None]
        if not vals:
            raise _missing(self, person, f"neighbours with answers[{qid!r}]")
        if question.get("options"):
            return Counter(str(a) for a in vals)
        return sum(float(a) for a in vals) / len(vals)

    def _group(self, person: dict) -> str:
        try:
            return self.group.format(**self.person_fields(person))
        except KeyError as e:
            raise _missing(self, person, f"field {e.args[0]!r} for group={self.group!r}") from None

    def render(self, person: dict, question: dict) -> Prompt:
        norm, group = self._norm(person, question), self._group(person)
        if isinstance(norm, Mapping):
            sh = _shares(norm, question)
            if not sh:
                raise _missing(self, person, f"a non-empty norm for {question.get('id')!r}")
            if self.style == "most":
                top, s = max(sh, key=lambda kv: kv[1])          # first maximum = option order on ties
                line = (f"Most {group} ({_pct(s, self.decimals)}) chose {_quote(top)}." if s > 0.5 else
                        f"The most common answer among {group} is {_quote(top)} ({_pct(s, self.decimals)}).")
            else:
                parts = [f"{_pct(s, self.decimals)} chose {_quote(k)}" for k, s in sh]
                line = f"Among {group}, approximately " + (" and ".join(parts) if len(parts) < 3 else
                                                            ", ".join(parts[:-1]) + " and " + parts[-1]) + "."
        else:
            line = f"Among {group}, the average answer to this question is {float(norm):g}."
        system = self._system(person, f"What is typical: {line}\nYou are free to give any answer.")
        return Prompt(system=system, user=self.question_block(question), meta={"norm": line})


@register
class NewsFeed(_Social):
    """Media exposure as relevant memories in the system prompt (Williams et al. 2023: persona + memories + 'checks
    the newspaper and finds that X% of the town's population caught new infections yesterday')."""
    name = "news_feed"
    paper = WILLIAMS
    needs = ("media",)
    description = ("System: optional persona + 'Relevant memories:' (optional context line, then 'You read in "
                   "<source>: <text>' for the k most recent or seeded random media items); user: the question. "
                   "Reads media (str or {source, text, time}), optional fields, text.")

    def __init__(self, k: int | None = 3, context: str = "", select: str = "recent", persona: bool = True,
                 seed: int = 0, **params: Any):
        if select not in SELECT:
            raise ValueError(f"select must be one of {SELECT}")
        _check_k(k)
        super().__init__(k=k, context=context, select=select, persona=persona, seed=seed, **params)
        self.k, self.context, self.select, self.persona, self.seed = k, context, select, persona, seed

    def render(self, person: dict, question: dict) -> Prompt:
        items = [{"text": m} if isinstance(m, str) else dict(m) for m in person.get("media") or ()]
        items = [m for m in items if not _blank(m.get("text"))]
        if not items:
            raise _missing(self, person, "at least one media item with 'text'")
        shown = _pick(items, self.k, self.select, _rng(self.seed, person, question))
        mem = ([f"- {self.context}"] if self.context else []) + [
            f"- You read in {m.get('source') or 'the news'}: {m['text']}" for m in shown]
        system = self._system(person, "Relevant memories:\n" + "\n".join(mem))
        return Prompt(system=system, user=self.question_block(question),
                      meta={"shown": [m.get("id", m.get("source")) for m in shown]})


# ---------------------------------------------------------------- bridges (abm states -> person fields)
def attach_neighbors(people: Sequence[dict], edges: Iterable[tuple[Any, Any]] | Mapping[Any, Iterable[Any]], *,
                     directed: bool = False, keys: Sequence[str] = ("label", "answers", "message", "action", "time"),
                     states: Mapping[Any, Mapping[str, Any]] | None = None) -> list[dict]:
    """Copies of `people` with `neighbors` built from a graph. edges: [(a, b), ...] or {a: [b, ...]}; directed
    means a sees b (a follows b). Each neighbour carries its id and `keys` from the neighbour's person dict,
    updated with `states[id]` (e.g. {"action": "adopted"} from an ABM step). Self-loops are dropped."""
    by_id = {str(p["id"]): p for p in people}
    if len(by_id) != len(people):
        raise ValueError("person ids must be unique")
    adj: dict[str, list[str]] = {i: [] for i in by_id}
    pairs = ((a, b) for a, bs in edges.items() for b in bs) if isinstance(edges, Mapping) else edges
    for a, b in pairs:
        a, b = str(a), str(b)
        if a == b:
            continue
        for x, y in ((a, b),) if directed else ((a, b), (b, a)):
            if x in adj and y in by_id and y not in adj[x]:
                adj[x].append(y)
    out = []
    for pid, p in by_id.items():
        nbs = []
        for j in adj[pid]:
            n = {"id": by_id[j]["id"], **{k: by_id[j][k] for k in keys if k in by_id[j]}}
            if states and (states.get(j) or states.get(by_id[j]["id"])):
                n.update(states.get(j) or states.get(by_id[j]["id"]))
            nbs.append(n)
        out.append({**p, "neighbors": nbs})
    return out


def attach_norms(people: Sequence[dict], questions: Sequence[dict], *, by: str | None = None) -> list[dict]:
    """Copies of `people` with `norms` {qid: {option: count}} (choice) or the mean (number) of the OTHER people's
    answers (leave-one-out, so a person's own answer never enters their norm), within the same `by` field value."""
    def key(p: dict) -> Any:
        if by is None:
            return None
        f = p.get("fields") if isinstance(p.get("fields"), dict) else p
        return f.get(by)

    if len({str(p["id"]) for p in people}) != len(people):
        raise ValueError("person ids must be unique")

    def val(p: dict, q: dict) -> Any:
        v = (p.get("answers") or {}).get(q["id"])
        if _blank(v):
            return None
        return float(v) if not q.get("options") else str(v)

    # group totals once, then subtract the person's own answer (leave-one-out) in O(people x questions)
    tot: dict[tuple[Any, str], Any] = {}
    for p in people:
        for q in questions:
            if (v := val(p, q)) is None:
                continue
            t = tot.setdefault((key(p), q["id"]), Counter() if q.get("options") else [0.0, 0])
            if q.get("options"):
                t[v] += 1
            else:
                t[0] += v
                t[1] += 1
    out = []
    for p in people:
        norms: dict[str, Any] = {}
        for q in questions:
            t, v = tot.get((key(p), q["id"])), val(p, q)
            if t is None:
                continue
            if q.get("options"):
                c = Counter(t)
                if v is not None:
                    c[v] -= 1
                c = {k: n for k, n in c.items() if n > 0}
                if c:
                    norms[q["id"]] = c
            else:
                s, n = (t[0] - v, t[1] - 1) if v is not None else (t[0], t[1])
                if n > 0:
                    norms[q["id"]] = s / n
        out.append({**p, "norms": norms})
    return out


def norm_from_dist(dist: Sequence[float], options: Sequence[Any]) -> dict[str, float]:
    """A K-category distribution (category i -> options[i], e.g. abm.opinion.distribution output) as a norm."""
    dist = [float(x) for x in dist]
    if len(dist) != len(options):
        raise ValueError(f"distribution has {len(dist)} categories, question has {len(options)} options")
    return {str(o): x for o, x in zip(options, dist)}


def prevalence_item(share: float, population: str, event: str, *, source: str = "the newspaper",
                    decimals: int = 1, time: Any = None) -> dict:
    """News item in the Williams et al. template: '<X>% of <population> <event>.' (share in 0-1)."""
    share = float(share)
    if not 0 <= share <= 1:
        raise ValueError(f"share must be in [0, 1] (a proportion, not a percentage), not {share!r}")
    item = {"source": source, "text": f"{_pct(share, decimals)} of {population} {event}."}
    if time is not None:
        item["time"] = time
    return item


__all__ = ["PeerMessages", "NeighborActivity", "AschMajority", "DescriptiveNorm", "NewsFeed", "neighbors_of",
           "attach_neighbors", "attach_norms", "norm_from_dist", "prevalence_item", "BIAS", "TONES", "FAMILY"]
