"""PersonaSource: an abstract source that generates personas and asks personas questions, plus drivers.
Ported from the TALP prep `sahne/kod/sahne/kaynak.py`.

Drivers:
  ClaudeSource  any `simengine.types.LLM` (simengine.llm.Claude, ClaudeCode, FakeLLM); batched prompts, JSON out
  TalpSource    TALP API slot. Empty on purpose: it raises NotImplementedError and sends NO request
  FakeSource    deterministic, offline (UI work, tests, dry runs). Its numbers are NOT measurements

Shared layer (all drivers): a new-call budget, a call/cost counter, an optional JSONL log of every prompt
and reply, parallel calls (<= 4 by default) and an optional live event callback. The disk cache lives in
the LLM backend (`simengine.llm.Claude(cache_dir=...)`); replies it serves from cache do not use budget.

    src = ClaudeSource(llm, budget=200)
    personas = src.generate(spec, n=20)
    answers = src.ask(personas, "Rate this ad 1-10 ...", {"score": "int 1-10", "why": "str"})
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .schema import AGE_BAND, AudienceSpec, Persona, enum_for

PARALLEL_CAP = int(os.environ.get("SIMENGINE_PARALLEL", "4"))     # server load rule in the research runs: <= 4


class LLMError(RuntimeError):
    """At least one LLM call failed in a run: do not write predictions, stop with an error."""


class BudgetExceeded(RuntimeError):
    pass


# ---------------------------------------------------------------- shared layer
def prompt_key(model: str, system: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\n\x00{system}\n\x00{prompt}".encode()).hexdigest()[:32]


class JsonlLog:
    """One JSONL line per prompt/reply (cached replies too, with `cached: true`)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def write(self, record: dict):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"time": time.strftime("%F %T"), **record}, ensure_ascii=False,
                                   default=str) + "\n")


@dataclass
class CallCounter:
    budget: int | None = 200         # cap on new (non-cached) calls; None = unlimited
    calls: int = 0                   # new calls
    cached: int = 0                  # served from cache
    errors: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve(self):
        with self.lock:
            if self.budget is not None and self.calls >= self.budget:
                raise BudgetExceeded(f"call budget used up ({self.budget})")
            self.calls += 1

    def refund_as_cached(self):
        with self.lock:
            self.calls -= 1
            self.cached += 1

    def add(self, usd: float | None, seconds: float):
        with self.lock:
            self.usd += usd or 0.0
            self.seconds += seconds

    def summary(self) -> dict:
        return {"calls": self.calls, "cached": self.cached, "errors": self.errors,
                "usd": round(self.usd, 4), "llm_seconds": round(self.seconds, 1), "budget": self.budget}


@dataclass
class Request:
    prompt: str
    system: str = ""
    tag: str = ""                    # grouping in the log: "generate", "ask:ab", "round3" ...
    fields: dict | None = None       # expected JSON fields (for the fake driver and the parser)
    subjects: list[str] | None = None  # persona ids in this prompt (for the fake driver)


@dataclass
class Reply:
    text: str | None
    data: object = None              # parsed JSON (array/object) or None
    cached: bool = False
    seconds: float = 0.0
    cost_usd: float | None = None
    error: str | None = None


# ---------------------------------------------------------------- JSON parser (LLM output)
def json_extract(text: str | None):
    """First JSON array/object in an LLM reply (code fences allowed). A truncated array keeps its complete
    objects. None when nothing parses."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    s = m.group(1) if m else text
    i = min([j for j in (s.find("["), s.find("{")) if j >= 0], default=-1)
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(s[i:])[0]
    except json.JSONDecodeError:
        pass
    out = []
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", s):
        try:
            out.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass
    return out or None


def _as_list(data) -> list:
    return data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])


# ---------------------------------------------------------------- abstract source
class PersonaSource(ABC):
    """Common interface of every driver. A subclass writes only `_call` (and generate/ask if needed)."""

    name = "abstract"
    model = "-"

    def __init__(self, budget: int | None = 200, parallel: int = PARALLEL_CAP, log_path: str | Path | None = None,
                 on_event: Callable[[dict], None] | None = None, retries: int = 3, retry_wait: float = 2.0):
        self.counter = CallCounter(budget=budget)
        self.parallel = max(1, min(parallel, PARALLEL_CAP))
        self.log = JsonlLog(log_path) if log_path else None
        self.on_event = on_event                 # live stream: send events to a UI (e.g. SSE)
        self.retries = max(1, retries)
        self.retry_wait = retry_wait
        self.last_error: str | None = None

    # -- the one job of a driver: return (text, usd)
    @abstractmethod
    def _call(self, request: Request) -> tuple[str, float | None]:
        ...

    def _emit(self, kind: str, **kw):
        if self.on_event:
            try:
                self.on_event({"kind": kind, **kw})
            except Exception:  # noqa: BLE001 — a UI error does not stop the run
                pass

    def _log(self, record: dict):
        if self.log:
            self.log.write({"driver": self.name, "model": self.model, **record})

    def _record_error(self, request: Request, err: str):
        with self.counter.lock:
            self.counter.errors += 1
            self.last_error = err
        self._log({"tag": request.tag, "key": prompt_key(self.model, request.system, request.prompt), "error": err})

    def one(self, request: Request) -> Reply:
        """One request with retries. Raises BudgetExceeded when the budget is used up."""
        key = prompt_key(self.model, request.system, request.prompt)
        self.counter.reserve()
        last = None
        for i in range(self.retries):
            t0 = time.time()
            try:
                text, usd = self._call(request)
                sec = time.time() - t0
                self.counter.add(usd, sec)
                self._log({"tag": request.tag, "key": key, "system": request.system, "prompt": request.prompt,
                           "reply": text, "seconds": round(sec, 2), "usd": usd})
                return Reply(text, json_extract(text), seconds=sec, cost_usd=usd)
            except BudgetExceeded:
                raise
            except NotImplementedError:
                raise
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {str(e)[:200]}"
                if i + 1 < self.retries and self.retry_wait:
                    time.sleep(self.retry_wait * (i + 1))
        self._record_error(request, last)
        return Reply(None, error=last)

    def complete(self, requests: list[Request]) -> list[Reply]:
        """Parallel, order preserved. When the budget runs out the remaining requests come back as errors."""
        def f(req):
            try:
                r = self.one(req)
            except BudgetExceeded as e:
                r = Reply(None, error=str(e))
            self._emit("reply", tag=req.tag, subjects=req.subjects, data=r.data, error=r.error,
                       cached=r.cached, counter=self.counter.summary())
            return r
        if self.parallel == 1 or len(requests) <= 1:
            return [f(r) for r in requests]
        with ThreadPoolExecutor(self.parallel) as ex:
            return list(ex.map(f, requests))

    # ------------------------------------------------------------ high level: generate personas
    def generate(self, spec: AudienceSpec, n: int, batch: int = 8, seed: int = 0) -> list[Persona]:
        """Brief -> n personas. Quota: the chosen enum values are spread over the batches (instead of asking the
        LLM to "be diverse", we hand each persona its target cell, so the spread is kept)."""
        cells = cell_plan(spec, n, seed)
        lang = spec.language()
        requests = []
        for b in range(0, n, batch):
            group = cells[b:b + batch]
            ids = [f"p{b + j + 1:03d}" for j in range(len(group))]
            rows = "\n".join(f"- id {pid}: " + ", ".join(f"{k}={v}" for k, v in c.items())
                             for pid, c in zip(ids, group))
            prompt = GENERATE_PROMPT.format(
                brief=spec.target_description or spec.audience.identity_text, identity=spec.audience.identity_text,
                anchor=spec.audience.behavioral_anchor, vertical=spec.vertical, locale=spec.locale,
                behavior="\n".join("- " + (s if isinstance(s, str) else s.get("text", ""))
                                   for s in spec.behavior_sketch) or "-",
                rows=rows, language="Turkish" if lang == "tr" else "English")
            requests.append(Request(prompt, SYSTEM_GENERATE, tag="generate", subjects=ids,
                                    fields={"__persona__": True, "cells": dict(zip(ids, group))}))
        self._emit("stage", name="generate", n=n, calls=len(requests))
        out = []
        for req, rep in zip(requests, self.complete(requests)):
            items = _as_list(rep.data)
            cell = req.fields["cells"]
            for pid in req.subjects:
                raw = next((x for x in items if isinstance(x, dict)
                            and str(x.get("id", x.get("persona_id"))) == pid), None)
                out.append(build_persona(pid, cell[pid], raw, spec))
        return out

    # ------------------------------------------------------------ high level: ask personas
    def ask(self, personas: list[Persona], task: str, fields: dict, batch: int = 10, tag: str = "ask",
            context: dict | None = None, lang: str = "en") -> dict[str, dict]:
        """Ask every persona the same task; batched prompt (`batch` people per call). Returns {persona_id: {field: value}}.
        `fields`: {"score": "int 1-10", "why": "str", "choice": "one of A|B"}, used both in the prompt and by the
        fake driver. `context`: {persona_id: extra text} (e.g. previous round, measured history)."""
        context = context or {}
        field_text = "\n".join(f'  "{k}": {v}' for k, v in fields.items())
        requests = []
        for b in range(0, len(personas), batch):
            group = personas[b:b + batch]
            block = "\n\n".join(f"### {p.pid}\n{p.text(lang)}" + (f"\nContext: {context[p.pid]}" if p.pid in context else "")
                                for p in group)
            prompt = ASK_PROMPT.format(task=task, people=block, fields=field_text,
                                       language="Turkish" if lang == "tr" else "English")
            requests.append(Request(prompt, SYSTEM_ASK, tag=tag, fields=fields, subjects=[p.pid for p in group]))
        out: dict[str, dict] = {}
        for req, rep in zip(requests, self.complete(requests)):
            for x in _as_list(rep.data):
                if isinstance(x, dict) and str(x.get("id")) in req.subjects:
                    out[str(x["id"])] = x
        return out

    def summary(self) -> dict:
        return {"driver": self.name, "model": self.model, **self.counter.summary()}

    def check_errors(self, since: int = 0, what: str = "run") -> None:
        """Raise LLMError if the error counter grew since `since`: better stop than build a card on missing answers."""
        n = self.counter.errors - since
        if n > 0:
            raise LLMError(f"{what}: {n} LLM call(s) failed, no predictions written. Last error: {self.last_error}")


# ---------------------------------------------------------------- prompts (verbatim from the research code)
SYSTEM_GENERATE = ("You write realistic, specific consumer personas for a simulation panel. Each persona must be a "
                   "distinct, plausible individual consistent with the assigned attributes. Avoid stereotypes and "
                   "avoid making everyone agreeable. Output only JSON.")
SYSTEM_ASK = ("You simulate several distinct people. Answer as each person would, from their own situation, "
              "budget and habits; disagreement and indifference are normal. Do not make everyone positive. "
              "Output only a JSON array.")

GENERATE_PROMPT = """Audience brief: {brief}
Identity: {identity}
Behavioral anchor: {anchor}
Vertical: {vertical} · locale: {locale}
How this audience behaves:
{behavior}

Write one persona for each id below, respecting its assigned attributes exactly:
{rows}

Return a JSON array; one object per id:
{{"id": "...", "name": "...", "tagline": "...", "persona": "3-4 sentence backstory", "behavior_voice": "1-2 sentences in first person",
 "buying_context": "how and why they buy in this vertical", "motivations": ["..."], "frustrations": ["..."],
 "media_habits": ["..."], "price_sensitivity": 0.0-1.0, "novelty_seeking": 0.0-1.0}}
Write the text fields in {language}. JSON only."""

ASK_PROMPT = """{task}

People:
{people}

For EACH person above return one JSON object, in a JSON array, with fields:
  "id": the person id
{fields}
Free-text fields in {language}. Answer independently for each person. JSON array only."""


# ---------------------------------------------------------------- quota plan and persona assembly
CELL_AXES = ("income_tier", "household_composition", "usage_intensity", "commitment_disposition",
             "brand_awareness", "age_band", "gender")


def cell_plan(spec: AudienceSpec, n: int, seed: int = 0) -> list[dict]:
    """Spread the selected enum values evenly over n personas (round-robin per axis, then a seeded shuffle per
    axis so the axes are not locked together). Age defaults to all TALP bands, gender to female/male.
    Out-of-enum values are dropped from the quota (`AudienceSpec.validate()` reports them)."""
    r = random.Random(seed)
    a = spec.audience
    axes = {
        "income_tier": a.income_tier, "household_composition": a.household_composition,
        "usage_intensity": a.usage_intensity, "commitment_disposition": a.commitment_disposition,
        "brand_awareness": a.brand_awareness, "age_band": a.age_range or list(AGE_BAND),
        "gender": a.gender_mix or ["female", "male"],
    }
    plan = [{} for _ in range(n)]
    for name, values in axes.items():
        allowed = enum_for(name)
        if allowed:
            values = [v for v in values if v in allowed]
        if not values:
            continue
        order = [values[i % len(values)] for i in range(n)]
        r.shuffle(order)
        for p, v in zip(plan, order):
            p[name] = v
    return plan


def build_persona(pid: str, cell: dict, raw: dict | None, spec: AudienceSpec) -> Persona:
    """Cell (assigned attributes) + raw LLM object -> Persona. A missing object gives a stub persona marked
    `generation_error: true` (visible in the JSON), never an invented one."""
    raw = raw or {}
    traits = [{"label": k, "value": v, "group": "socioeconomic" if k == "income_tier" else None}
              for k, v in cell.items() if k not in ("age_band", "gender")]
    d = {
        "identity": {"persona_id": pid, "name": raw.get("name") or pid, "gender": cell.get("gender"),
                     "age_band": cell.get("age_band"), "persona_mode": spec.persona_mode, "locale": spec.locale,
                     "tagline": raw.get("tagline")},
        "traits": traits,
        "authored": {"persona": raw.get("persona", ""), "behavior_voice": raw.get("behavior_voice"),
                     "buying_context": raw.get("buying_context")},
        "behavioral": {"motivations": raw.get("motivations") or [], "frustrations": raw.get("frustrations") or [],
                       "media_habits": raw.get("media_habits") or []},
        "quantitative_dispositions": [{"dimension": k, "value": float(raw[k])}
                                      for k in ("price_sensitivity", "novelty_seeking")
                                      if isinstance(raw.get(k), (int, float)) and not isinstance(raw.get(k), bool)],
        "segment": cell.get("income_tier") or cell.get("age_band"),
    }
    if not raw:
        d["extra"] = {"generation_error": True}
    return Persona.from_dict(d)


# ---------------------------------------------------------------- drivers
def _is_stop_error(exc: BaseException) -> bool:
    """True for simengine.llm stop errors (AuthError, SpendLimitError, BudgetExceeded): stop the run, no retry."""
    try:
        from ..llm.errors import StopError
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, StopError)


class ClaudeSource(PersonaSource):
    """Driver over any `simengine.types.LLM`. Batched prompts go through `llm.map` (parallel, the backend's
    cache/backoff/cost ledger). The backend owns retries, so `retries` defaults to 1 here. An error whose
    text looks like authentication / spend limit / budget (or any simengine.llm StopError) stops the remaining
    requests of the run."""

    name = "claude"
    STOP_MARKERS = ("401", "authentication", "unauthorized", "budget", "limit", "credit")

    def __init__(self, llm, *, max_tokens: int = 4096, retries: int = 1, llm_kwargs: dict | None = None, **kw):
        self.llm = llm
        self.model = getattr(llm, "model", "llm")
        self.max_tokens = max_tokens
        self.llm_kwargs = dict(llm_kwargs or {})
        self.stopped: str | None = None
        super().__init__(retries=retries, **kw)

    def _call(self, request: Request):
        r = self.llm.complete(request.system, request.prompt, max_tokens=self.max_tokens, **self.llm_kwargs)
        return r.text, r.cost_usd

    def _is_stop(self, err: str) -> bool:
        low = err.lower()
        return any(m in low for m in self.STOP_MARKERS)

    def _reply_from(self, req: Request, resp, sec: float) -> Reply:
        if getattr(resp, "cached", False):
            self.counter.refund_as_cached()
        else:
            self.counter.add(resp.cost_usd, sec)
        key = prompt_key(self.model, req.system, req.prompt)
        self._log({"tag": req.tag, "key": key, "system": req.system, "prompt": req.prompt, "reply": resp.text,
                   "cached": bool(getattr(resp, "cached", False)), "usd": resp.cost_usd})
        return Reply(resp.text, json_extract(resp.text), cached=bool(getattr(resp, "cached", False)),
                     seconds=sec, cost_usd=resp.cost_usd)

    def complete(self, requests: list[Request]) -> list[Reply]:
        """Send the requests in chunks that fit the remaining budget; cached replies are refunded, so the next
        chunk can use that budget. Requests beyond the budget come back as errors (not counted as LLM errors)."""
        replies: list[Reply | None] = [None] * len(requests)
        pending = list(range(len(requests)))
        while pending and not self.stopped:
            todo = []
            while pending:
                try:
                    self.counter.reserve()
                except BudgetExceeded:
                    break
                todo.append(pending.pop(0))
            if not todo:
                break
            self._send(requests, todo, replies)
        for i in pending:
            replies[i] = Reply(None, error=f"stopped: {self.stopped}" if self.stopped
                               else f"call budget used up ({self.counter.budget})")
        for req, rep in zip(requests, replies):
            self._emit("reply", tag=req.tag, subjects=req.subjects, data=rep.data, error=rep.error,
                       cached=rep.cached, counter=self.counter.summary())
        return replies

    def _send(self, requests: list[Request], todo: list[int], replies: list) -> None:
        if todo:
            t0 = time.time()
            try:
                resps = self.llm.map([(requests[i].system, requests[i].prompt) for i in todo],
                                     max_tokens=self.max_tokens, parallel=self.parallel, **self.llm_kwargs)
                sec = (time.time() - t0) / max(1, len(todo))
                for i, resp in zip(todo, resps):
                    replies[i] = self._reply_from(requests[i], resp, sec)
                for i in todo[len(resps):]:              # backend returned fewer replies than prompts
                    replies[i] = Reply(None, error="no reply from LLM.map")
                    self._record_error(requests[i], replies[i].error)
            except Exception:  # noqa: BLE001 — fall back to one-by-one so a single failure is isolated
                for i in todo:
                    req = requests[i]
                    if self.stopped:
                        replies[i] = Reply(None, error=f"stopped: {self.stopped}")
                        self._record_error(req, replies[i].error)
                        continue
                    last = None
                    for _ in range(self.retries):
                        t1 = time.time()
                        try:
                            resp = self.llm.complete(req.system, req.prompt, max_tokens=self.max_tokens,
                                                     **self.llm_kwargs)
                            replies[i] = self._reply_from(req, resp, time.time() - t1)
                            break
                        except Exception as e:  # noqa: BLE001
                            last = f"{type(e).__name__}: {str(e)[:200]}"
                            if _is_stop_error(e) or self._is_stop(last):
                                self.stopped = last
                                break
                    if replies[i] is None:
                        replies[i] = Reply(None, error=last)
                        self._record_error(req, last)


LLMSource = ClaudeSource   # the driver works with any LLM backend


class TalpSource(PersonaSource):
    """TALP API driver (EMPTY). To be filled if the organizers hand out a key and documentation.
    Known surface (TALP persona schema notes): POST /persona-sets (AudienceSpec), GET .../personas/:id (Persona),
    ad study create/report, MCP: plan/launch/report/follow-up. This class sends no network request."""

    name = "talp"
    model = "talp-api"

    def __init__(self, *a, base_url: str | None = None, api_key: str | None = None, **kw):
        self.base_url = base_url or os.environ.get("TALP_API_URL")
        self._api_key = api_key or os.environ.get("TALP_API_KEY")
        super().__init__(*a, **kw)

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self._api_key)

    def _ready(self):
        if not self.configured:
            raise NotImplementedError("TALP driver is empty: if TALP_API_URL and TALP_API_KEY are provided, "
                                      "fill simengine.persona.sources.TalpSource. Use ClaudeSource or FakeSource now.")
        raise NotImplementedError("TALP API endpoints are written from the organizers' docs (empty on purpose).")

    def _call(self, request: Request):
        self._ready()

    def generate(self, spec, n, batch=8, seed=0):
        self._ready()

    def ask(self, personas, task, fields, batch=10, tag="ask", context=None, lang="en"):
        self._ready()


class FakeSource(PersonaSource):
    """Offline, deterministic driver. Produces hash-based values instead of an LLM; NOT for measurement, only for
    UI/pipeline dry runs. Its text is always marked "(fake)"."""

    name = "fake"
    model = "fake-0"

    def __init__(self, *a, delay: float = 0.0, **kw):
        self.delay = delay
        super().__init__(*a, **kw)

    @staticmethod
    def _h(*parts) -> float:
        return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    def _value(self, pid, name, kind, seed):
        h = self._h(pid, name, seed)
        kind = str(kind)
        m = re.search(r"int\s*(-?\d+)\s*-\s*(-?\d+)", kind)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return a + int(h * (b - a + 1)) if b > a else a
        if kind.startswith("float") or "0.0-1.0" in kind or "probability" in kind:
            return round(h, 3)
        m = re.search(r"one of ([^\n]+)", kind)
        if m:
            options = [s.strip() for s in re.split(r"\||,", m.group(1)) if s.strip()]
            return options[int(h * len(options)) % len(options)]
        if kind.startswith("list"):
            return [f"(fake) {name} {int(h * 100)}"]
        return f"(fake) {name} · {pid}"

    def _call(self, request: Request):
        if self.delay:
            time.sleep(self.delay)
        seed = prompt_key(self.model, request.system, request.prompt)[:6]
        fl = request.fields or {}
        out = []
        for pid in request.subjects or ["x"]:
            if fl.get("__persona__"):
                c = fl["cells"][pid]
                out.append({"id": pid, "name": f"Persona {pid[1:]}",
                            "tagline": f"{c.get('income_tier', '')} {c.get('household_composition', '')}".strip(),
                            "persona": f"(fake) {pid}: " + ", ".join(f"{k}={v}" for k, v in c.items()),
                            "behavior_voice": "(fake)", "buying_context": "(fake)",
                            "motivations": ["(fake)"], "frustrations": ["(fake)"], "media_habits": [],
                            "price_sensitivity": round(self._h(pid, "ps"), 2),
                            "novelty_seeking": round(self._h(pid, "ns"), 2)})
            else:
                out.append({"id": pid, **{k: self._value(pid, k, v, seed) for k, v in fl.items()}})
        return json.dumps(out, ensure_ascii=False), 0.0


SOURCES = {"claude": ClaudeSource, "llm": ClaudeSource, "talp": TalpSource, "fake": FakeSource}


def open_source(name: str, **kw) -> PersonaSource:
    """Driver by name: 'claude'/'llm' (needs llm=...), 'talp', 'fake'."""
    if name not in SOURCES:
        raise ValueError(f"unknown driver {name}: {list(SOURCES)}")
    return SOURCES[name](**kw)


__all__ = ["PersonaSource", "ClaudeSource", "LLMSource", "TalpSource", "FakeSource", "open_source", "Request",
           "Reply", "CallCounter", "BudgetExceeded", "LLMError", "json_extract", "cell_plan", "build_persona",
           "prompt_key", "SYSTEM_GENERATE", "SYSTEM_ASK", "GENERATE_PROMPT", "ASK_PROMPT"]
