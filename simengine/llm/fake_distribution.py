"""Offline backend for the distribution runners (G7 #11): `DistributionFake` reads the `sim.distribution` prompts
(DIST_TEMPLATE: one segment x question; PERSON_PROB_TEMPLATE: a table of respondents x questions), extracts group,
question and options, and answers with well-formed JSON percentages, so `segment_distribution` /
`persona_distribution` pipelines run end to end without a real LLM.

    llm = DistributionFake()                                   # synthetic_distribution per (group, question)
    llm = DistributionFake(lambda group, question, options: [0.2, 0.3, 0.5] if "65" in group else None)
    llm = DistributionFake(segments=segs, noise=0.3, unparsed=0.1, refuse=lambda g, q: q.startswith("Did you vote"))
    res = se.sim.segment_distribution(segs, questions, llm=llm)
    frame = se.sim.distribution_frame(res)

group is what the prompt shows: the segment description (or its key when `segments=` maps descriptions back to
keys) for segment prompts; the respondent's profile line (or id, `person_key="id"`) for persona tables. Every
other prompt falls through to `sim.schema_responder(seed)`. Dry runs and contract tests only, never a measurement:
a value_fn built from real answers prints real-looking metrics (G2 #1), so keep it synthetic.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Mapping, Sequence

from ..types import LLMResponse
from .fake import FakeLLM

ValueFn = Callable[[str, str, list], Any]
Knob = Callable[[str, str], bool] | float | None

REFUSAL_TEXT = "I'm sorry, but I can't produce this survey data for these respondents."
UNPARSED_TEXT = "Opinion in this group is split; most people would pick one of the middle options."

_DIST = re.compile(r"^(?P<head>.*?)\nQuestion: (?P<stem>.*?)\nAnswer options:\n(?P<opts>(?:\d+\. [^\n]*\n)+)"
                   r"Estimate the percentage", re.S)
_MINORITY = " Think about the full spread of opinion in this group, including minority views."
_GROUP = re.compile(r"\b[Cc]onsider (?:respondents who are )?(?P<g>.+?)\.(?:" + re.escape(_MINORITY) + r")?\s*$")


def _seed_int(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("\x00".join(map(str, parts)).encode()).digest()[:8], "big")


def _unit(*parts: Any) -> float:
    return _seed_int(*parts) / 2 ** 64


def synthetic_distribution(k: int, group: str, question: str, seed: int = 0, alpha: float = 1.0) -> list[float]:
    """Deterministic Dirichlet(alpha, ..., alpha) draw over k options keyed on (seed, group, question, k): shares
    summing to 1. alpha=1 is uniform over the simplex; a large alpha pulls towards 1/k."""
    import numpy as np
    k = int(k)
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    rng = np.random.default_rng(_seed_int(seed, group, question, k))
    p = rng.dirichlet([float(alpha)] * k)
    return (p / p.sum()).tolist()


def parse_distribution_prompt(user: str) -> dict | None:
    """{"group", "question", "options", "head"} from a DIST_TEMPLATE prompt (a reframe note before it is fine),
    None for any other prompt. A custom framing without "consider ..." gives its whole line as the group."""
    m = _DIST.search(user)
    if not m:
        return None
    head = m.group("head").rstrip().rsplit("\n", 1)[-1].strip()
    g = _GROUP.search(head)
    opts = [re.sub(r"^\d+\. ", "", ln) for ln in m.group("opts").splitlines() if ln.strip()]
    return {"group": g.group("g").strip() if g else head, "question": m.group("stem").strip(), "options": opts,
            "head": head}


def _split_options(line: str) -> tuple[str, list[str]] | None:
    """'text 1=a, 2=b, 3=c' -> ('text', ['a', 'b', 'c']): the first ' 1=' after which ', 2=', ', 3=' ... follow."""
    for m in re.finditer(r"(?:^| )1=", line):
        parts = re.split(r", (\d+)=", line[m.end():])
        nums = parts[1::2]
        if nums == [str(j + 2) for j in range(len(nums))] and nums:
            return line[:m.start()].strip(), parts[0::2]
    return None


def _table_ids(joined: str, lines: list[str]) -> list[str] | None:
    """Ids of a respondent table, walking the "pid: profile" lines against the ", "-joined id list, so ids that
    contain ", ", ": " or parentheses survive. None when the lines do not spell out the list."""
    ids, pos = [], 0
    for ln in lines:
        hit = next((ln[:j] for j in (k.start() for k in re.finditer(": ", ln))
                    if joined.startswith(ln[:j], pos) and joined[pos + j:pos + j + 2] in ("", ", ")), None)
        if hit is None:
            return None
        ids.append(hit)
        pos += len(hit) + 2
    return ids if ids and pos - 2 == len(joined) else None


def parse_person_prompt(user: str) -> dict | None:
    """{"ids", "profiles" {id: line}, "questions" [{"question", "options"}], "session"} from a
    PERSON_PROB_TEMPLATE prompt, None for any other prompt."""
    m = re.search(r"respondent IDs as keys \((.*?)\)\. For each respondent", user, re.S) or \
        re.search(r"respondent IDs as keys \(([^)]*)\)", user)
    sec = re.search(r"\nRespondents:\n(.*?)\n\nQuestions \(answer options are numbered\):\n(.*?)\n\nReturn ONLY",
                    user, re.S)
    if not (m and sec):
        return None
    lines = sec.group(1).splitlines()
    ids = _table_ids(m.group(1), lines) or [x.strip() for x in m.group(1).split(",") if x.strip()]
    profiles = {}
    for pid in ids:
        hit = next((ln for ln in lines if ln.startswith(f"{pid}: ")), None)
        profiles[pid] = hit[len(pid) + 2:].strip() if hit is not None else ""
    qs = []
    for block in re.split(r"^Q\d+\. ", sec.group(2), flags=re.M)[1:]:
        got = _split_options(block.strip().replace("\n", " "))
        if got is None:
            return None
        qs.append({"question": got[0], "options": got[1]})
    sess = re.search(r"^Session: (.*)$", user, re.M)
    return {"ids": ids, "profiles": profiles, "questions": qs, "session": sess.group(1) if sess else ""}


def _knob(name: str, v: Knob) -> Knob:
    if v is None or callable(v):
        return v
    v = float(v)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be a rate in [0, 1] or a callable(group, question) -> bool, got {v}")
    return v or None


def _pct(p: Sequence[float]) -> list[float]:
    """Shares -> percentages rounded to 4 decimals that sum to exactly 100 (the rounding rest on the largest)."""
    out = [round(100.0 * x, 4) for x in p]
    j = max(range(len(out)), key=out.__getitem__)
    out[j] = round(out[j] + 100.0 - sum(out), 4)
    return out


class DistributionFake(FakeLLM):
    """FakeLLM-compatible backend for `sim.segment_distribution` / `sim.persona_distribution` prompts.

    value_fn(group, question, options) -> shares (list of k non-negative numbers, percentages or fractions) or a
    dict keyed by option number / letter / text, or None for the default `synthetic_distribution(k, group,
    question, seed)`. An unusable value raises ValueError. noise in [0, 1] mixes in a second deterministic
    Dirichlet draw keyed on the framing line (or table session), the repeat's `sample` and the respondent, so
    framings and repeats differ; noise=0 gives every framing the same vector.
    refuse / unparsed: a rate in [0, 1] (deterministic hash per segment x question; in a persona table per batch
    for refuse and per respondent for unparsed, so the rate is the share of batches / respondents) or
    callable(group, question) -> bool. A refused segment prompt gets `refusal_text` (a soft refusal; it refuses
    the re-ask too); an unparsed one gets plain prose. In a persona table a callable true for any respondent x
    question refuses the whole batch, and an unparsed respondent is left out of the JSON (status "unparsed").
    refuse is checked before unparsed.
    segments: {key: description} as given to segment_distribution, so value_fn and the knobs see the key (also
    under a custom framing: the longest description found in the framing line).
    person_key: "profile" (the respondent's line in the prompt, what an LLM sees) or "id"."""

    def __init__(self, value_fn: ValueFn | None = None, seed: int = 0, noise: float = 0.0, *,
                 refuse: Knob = None, unparsed: Knob = None, segments: Mapping[str, str] | None = None,
                 person_key: str = "profile", alpha: float = 1.0, model: str = "fake"):
        super().__init__(None, model=model)
        if not 0.0 <= float(noise) <= 1.0:
            raise ValueError(f"noise must be in [0, 1], got {noise}")
        if person_key not in ("profile", "id"):
            raise ValueError(f"person_key must be 'profile' or 'id', got {person_key!r}")
        self.value_fn, self.seed, self.noise, self.alpha = value_fn, seed, float(noise), alpha
        self.refuse, self.unparsed = _knob("refuse", refuse), _knob("unparsed", unparsed)
        self.segment_key = {str(v): str(k) for k, v in (segments or {}).items()}
        self.person_key = person_key
        self.refusal_text, self.unparsed_text = REFUSAL_TEXT, UNPARSED_TEXT
        from ..sim.panel import schema_responder          # lazy: simengine.sim imports simengine.llm
        self._fallback = schema_responder(seed)

    def _hit(self, knob: Knob, what: str, group: str, question: str) -> bool:
        if knob is None:
            return False
        if callable(knob):
            return bool(knob(group, question))
        return _unit(self.seed, what, group, question) < knob

    def _table_hit(self, knob: Knob, what: str, groups: list[str], qs: list[dict]) -> bool:
        if callable(knob):
            return any(bool(knob(g, q["question"])) for g in groups for q in qs)
        return self._hit(knob, what, "\x00".join(groups), "")

    def _segment(self, group: str, head: str) -> str:
        """The segments= key for a DIST prompt: exact description, else the longest description inside the
        framing line (a custom framing without "consider ..."), else the group as parsed."""
        if group in self.segment_key:
            return self.segment_key[group]
        inside = [d for d in self.segment_key if d and d in head]
        return self.segment_key[max(inside, key=len)] if inside else group

    def shares(self, group: str, question: str, options: list[str], salt: Sequence[Any] = ()) -> list[float]:
        """The vector this fake answers for one (group, question): value_fn or synthetic_distribution, then
        noise keyed on `salt`."""
        from ..sim.parse import normalize_probs
        k = len(options)
        v = self.value_fn(group, question, list(options)) if self.value_fn is not None else None
        if v is None:
            p = synthetic_distribution(k, group, question, self.seed, self.alpha)
        else:
            arr = normalize_probs(v, k, options)
            if arr is None:
                raise ValueError(f"value_fn returned {v!r} for group {group!r}, question {question!r}; give {k} "
                                 f"non-negative numbers (not all zero) or a dict keyed by option number/text")
            p = arr.tolist()
        if self.noise:
            q = synthetic_distribution(k, group, question, (self.seed, "noise", *salt), self.alpha)
            p = [(1 - self.noise) * a + self.noise * b for a, b in zip(p, q)]
        return p

    def respond(self, system: str, user: str, sample: Any = 0) -> str:
        d = parse_distribution_prompt(user)
        if d is not None:
            g = self._segment(d["group"], d["head"])
            if self._hit(self.refuse, "refuse", g, d["question"]):
                return self.refusal_text
            if self._hit(self.unparsed, "unparsed", g, d["question"]):
                return self.unparsed_text
            p = self.shares(g, d["question"], d["options"], (d["head"], sample))
            return json.dumps({str(j + 1): x for j, x in enumerate(_pct(p))})
        t = parse_person_prompt(user)
        if t is not None:
            keys = {pid: pid if self.person_key == "id" else t["profiles"][pid] for pid in t["ids"]}
            qs = t["questions"]
            # a rate is drawn once per batch (refuse) / respondent (unparsed), so refuse=0.1 refuses ~10% of the
            # batches rather than compounding over every respondent x question cell; a callable sees each cell
            if self._table_hit(self.refuse, "refuse", [keys[pid] for pid in t["ids"]], qs):
                return self.refusal_text
            out = {}
            for pid in t["ids"]:
                if self._table_hit(self.unparsed, "unparsed", [keys[pid]], qs):
                    continue
                out[pid] = {"p": [_pct(self.shares(keys[pid], q["question"], q["options"],
                                                   (t["session"], pid, sample))) for q in qs]}
            return json.dumps(out)
        return self._fallback(system, user)

    def complete(self, system: str, user: str, *, max_tokens: int = 256, **kw: Any) -> LLMResponse:
        self.calls.append((system, user))
        text = self.respond(system, user, kw.get("sample", 0))
        return LLMResponse(text=text, model=self.model, input_tokens=len((system + user).split()),
                           output_tokens=len(text.split()))


__all__ = ["DistributionFake", "synthetic_distribution", "parse_distribution_prompt", "parse_person_prompt",
           "REFUSAL_TEXT", "UNPARSED_TEXT"]
