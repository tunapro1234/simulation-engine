"""Batched persona questioning over the `LLM` protocol (port of sahne/kaynak.py `PersonaSource.sor`).

One prompt carries `batch` personas; the model returns one JSON object per persona id. The prompt text
(SYSTEM_ASK, ASK_TEMPLATE) is the one used in the measured sahne runs (Haiku 5.5, 21 Oct).

    answers = ask_panel(llm, personas, "Rate this ad 1-10 ...", {"score": "int 1-10", "reason": "str"})
    # {pid: {"id": pid, "score": 7, "reason": "..."}}

Caching, budgets, retries and parallelism belong to the LLM implementation (simengine.llm). Here: prompt
building, robust parsing, id matching, usage accounting. A persona whose answer is missing or unparseable is
simply absent from the result and counted in `Usage.missing`.

`schema_responder()` is a deterministic offline responder for `FakeLLM` (port of sahne `SahteKaynak`): it reads
ids and field specs back out of the prompt and fills values by hashing. Dry runs only, never a measurement.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .parse import parse_json
from .personas import PersonaView

SYSTEM_ASK = ("You simulate several distinct people. Answer as each person would, from their own situation, "
              "budget and habits; disagreement and indifference are normal. Do not make everyone positive. "
              "Output only a JSON array.")

ASK_TEMPLATE = """{task}

People:
{people}

For EACH person above return one JSON object, in a JSON array, with fields:
  "id": the person id
{fields}
Free-text fields in {lang}. Answer independently for each person. JSON array only."""


@dataclass
class Usage:
    """Call / token / cost accounting summed over LLMResponse objects."""
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    missing: int = 0               # persona answers absent or unparseable
    remapped: int = 0              # answers whose unknown id was mapped positionally to a missing id
    unparsed_calls: int = 0        # responses with no JSON at all
    model: str | None = None
    tags: dict = field(default_factory=dict)

    def add(self, r, tag: str | None = None):
        self.calls += 1
        self.cached += int(bool(getattr(r, "cached", False)))
        self.input_tokens += int(getattr(r, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(r, "output_tokens", 0) or 0)
        self.cost_usd += float(getattr(r, "cost_usd", 0.0) or 0.0)
        self.model = getattr(r, "model", None) or self.model
        if tag:
            self.tags[tag] = self.tags.get(tag, 0) + 1

    def as_dict(self) -> dict:
        return {"model": self.model, "calls": self.calls, "cached": self.cached,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 6), "missing": self.missing, "remapped": self.remapped,
                "unparsed_calls": self.unparsed_calls}


def _items(data, ids: list[str]) -> list[dict]:
    """Accept a JSON array of objects, a single object, or a {pid: {...}} mapping."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if "id" in data:
            return [data]
        if any(str(k) in ids for k in data):
            return [{**v, "id": str(k)} for k, v in data.items() if isinstance(v, dict)]
        for v in data.values():           # {"people": [...]} wrapper
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def build_prompts(personas: list[PersonaView], task: str, fields: dict, *, batch: int = 10,
                  context: dict | None = None, lang: str = "en") -> list[tuple[tuple[str, str], list[str]]]:
    """[((system, user), [ids in this prompt]), ...]."""
    context = context or {}
    field_text = "\n".join(f'  "{k}": {v}' for k, v in fields.items())
    out = []
    for b in range(0, len(personas), max(1, batch)):
        group = personas[b:b + batch]
        block = "\n\n".join(f"### {p.pid}\n{p.text(lang)}" + (f"\nContext: {context[p.pid]}" if p.pid in context
                                                              else "") for p in group)
        user = ASK_TEMPLATE.format(task=task, people=block, fields=field_text,
                                   lang="Turkish" if lang == "tr" else "English")
        out.append(((SYSTEM_ASK, user), [p.pid for p in group]))
    return out


def ask_panel(llm, personas: list[PersonaView], task: str, fields: dict, *, batch: int = 10,
              context: dict | None = None, lang: str = "en", max_tokens: int | None = None,
              usage: Usage | None = None, tag: str = "ask", parallel: int = 4) -> dict[str, dict]:
    """Ask every persona the same task in batched prompts. Returns {pid: {field: value}}.
    `fields`: {"score": "int 1-10", "reason": "str", "choice": "one of A|B"}; `context`: {pid: extra text}."""
    usage = usage if usage is not None else Usage()
    plan = build_prompts(personas, task, fields, batch=batch, context=context, lang=lang)
    if not plan:
        return {}
    if max_tokens is None:
        max_tokens = min(16000, 300 + max(len(ids) for _, ids in plan) * (40 + 90 * len(fields)))
    responses = llm.map([p for p, _ in plan], max_tokens=max_tokens, parallel=parallel)
    out: dict[str, dict] = {}
    for (_, ids), r in zip(plan, responses):
        usage.add(r, tag)
        data = parse_json(getattr(r, "text", None))
        if data is None:
            usage.unparsed_calls += 1
        items = _items(data, ids)
        got, unknown = set(), []
        for x in items:
            xid = str(x.get("id", "")).strip()
            if xid in ids and xid not in got:
                out[xid] = {**x, "id": xid}
                got.add(xid)
            else:
                unknown.append(x)
        missing = [p for p in ids if p not in got]
        # the model renumbered the ids (e.g. P001.. from a template): map positionally only if counts match
        if unknown and missing and len(unknown) == len(missing) and not got:
            for x, p in zip(unknown, missing):
                out[p] = {**x, "id": p}
            usage.remapped += len(missing)
            missing = []
        usage.missing += len(missing)
    return out


# ---------------------------------------------------------------- offline dry-run responder
def _h(*parts) -> float:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _fake_value(pid, name, spec, seed):
    h = _h(pid, name, seed)
    spec = str(spec)
    m = re.search(r"int\s*(-?\d+)\s*-\s*(-?\d+)", spec)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return a + int(h * (b - a + 1)) if b > a else a
    if spec.startswith("float") or "0.0-1.0" in spec or "probability" in spec:
        return round(h, 3)
    m = re.search(r"one of ([^\n]+)", spec)
    if m:
        opts = [s.strip() for s in re.split(r"\||,", m.group(1)) if s.strip()]
        return opts[int(h * len(opts)) % len(opts)]
    if spec.startswith("list"):
        return [f"(fake) {name} {int(h * 100)}"]
    return f"(fake) {name} {pid}"


def schema_responder(seed: int = 0) -> Callable[[str, str], str]:
    """FakeLLM responder that answers simengine.sim prompts with deterministic hashed values:
    panel prompts (ids from '### <id>' lines, fields from the field list), distribution prompts (option
    percentages), per-persona probability vectors, per-person prediction prompts (keyed 0-100 values or SSR
    sentences). For dry runs, not measurement."""

    def respond(system: str, user: str) -> str:
        tag = hashlib.sha256((system + "\x00" + user).encode()).hexdigest()[:6] + str(seed)
        ids = re.findall(r"^### (\S+)\s*$", user, re.M)
        if ids and "with fields:" in user:
            block = user.split("with fields:", 1)[1].split("Free-text fields", 1)[0]
            specs = dict(re.findall(r'^\s+"([^"]+)":\s*(.+)$', block, re.M))
            specs.pop("id", None)
            return json.dumps([{"id": pid, **{k: _fake_value(pid, k, v, tag) for k, v in specs.items()}}
                               for pid in ids])
        m = re.search(r"respondent IDs as keys \(([^)]*)\)", user)
        if m:
            pids = [x.strip() for x in m.group(1).split(",") if x.strip()]
            ks = [len(re.findall(r"(?:^|\s)\d+=", ln)) for ln in re.findall(r"^Q\d+\. .*$", user, re.M)]
            out = {}
            for pid in pids:
                vecs = []
                for qi, k in enumerate(ks):
                    w = [_h(tag, pid, qi, j) + 0.05 for j in range(k)]
                    vecs.append([round(100 * x / sum(w)) for x in w])
                out[pid] = {"p": vecs}
            return json.dumps(out)
        m = re.search(r"Answer options:\n((?:\d+\. .*\n?)+)", user)
        if m and "percentage" in user:
            k = len(re.findall(r"^\d+\. ", m.group(1), re.M))
            w = [_h(tag, j) + 0.05 for j in range(k)]
            return json.dumps({str(j + 1): round(100 * x / sum(w)) for j, x in enumerate(w)})
        m = re.search(r"\(key: description\):\n(.*?)\n\n", user, re.S)
        if m:
            keys = [ln.split(":", 1)[0].strip() for ln in m.group(1).splitlines() if ":" in ln]
            if "one short first-person sentence" in user:
                return json.dumps({k: ("I probably bought something like this." if _h(tag, k) > 0.5
                                       else "I probably did not buy anything like this.") for k in keys})
            return json.dumps({k: int(_h(tag, k) * 100) for k in keys})
        return "{}"

    return respond


__all__ = ["ask_panel", "build_prompts", "Usage", "schema_responder", "SYSTEM_ASK", "ASK_TEMPLATE"]
