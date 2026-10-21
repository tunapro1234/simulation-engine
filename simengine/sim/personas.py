"""Persona adapter for the runners, plus the no-persona and persona-swap control arms.

The runners accept anything persona-like and never import simengine.persona (duck typing):
  * TALP-schema dicts {"identity": {...}, "traits": [...], "authored": {...}, "evidence": {...}, "segment": ...}
  * objects with `to_dict()` returning such a dict (e.g. simengine.persona.Persona), optionally `.segment`
  * flat dicts {"id"|"persona_id"|"pid", "name", "text"|"description"|"persona", "segment", "locale", ...}
    (without a text key, every other scalar field is rendered as "key: value"; pass only fields that are known
    before the prediction moment, never the target)
  * plain strings (one persona description each; ids p001, p002, ...)

Prompt rendering ports sahne/sema.py `Persona.metin` (name/age/gender line, up to 12 traits, authored text cut
at 600 chars, up to 10 measured facets). Controls port sahne/kosucu/ortak.py (`kontrol_personalari`,
`kontrol_plani`): the swap arm keeps each persona's own id, name and tagline and takes the next persona's
content (cyclic shift); the fix that keeps the identity line is from 21 Oct (intent: Haiku labelled 6/200
answers with the swapped identity before the fix).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

ALL = "all"
CONTROLS_DEFAULT = ("no_persona", "persona_swap")
CONTROL_ALIASES = {"personasiz": "no_persona", "persona_yok": "no_persona", "no_persona": "no_persona",
                   "persona_swap": "persona_swap", "swap": "persona_swap"}
CONTROL_BATCH_MAX = 20          # control arms: at most this many personas per prompt
NEUTRAL_TEXT = {"en": "A typical adult consumer.", "tr": "Tipik bir yetişkin tüketici."}
_FLAT_TEXT_KEYS = ("text", "description", "persona", "profile", "prompt")
_FLAT_RESERVED = {"id", "pid", "persona_id", "uid", "name", "segment", "locale", "tagline"} | set(_FLAT_TEXT_KEYS)


@dataclass
class PersonaView:
    """What a runner needs from a persona: id, segment, locale and prompt text (name line + details)."""
    pid: str
    name: str = ""
    attrs: list[str] = field(default_factory=list)     # rest of the first line (age, gender)
    body: list[str] = field(default_factory=list)      # traits, backstory, measured facts
    segment: str | None = None
    locale: str = ""
    tagline: str | None = None
    source: Any = field(default=None, repr=False)

    def text(self, lang: str = "en") -> str:
        head = ", ".join(x for x in [self.name or self.pid, *self.attrs] if x)
        return "\n".join([head, *[b for b in self.body if b]])

    @property
    def seg(self) -> str:
        return self.segment or ALL


def _blank(v) -> bool:
    return v is None or v == "" or (isinstance(v, float) and math.isnan(v))


def _from_talp(d: dict, cut: int = 600) -> PersonaView:
    i = d.get("identity") or {}
    dem = d.get("demographics") or {}
    pid = str(i.get("persona_id") or d.get("persona_id") or d.get("id") or "")
    age = i.get("age_band") or dem.get("age")
    attrs = [f"age {age}" if age else "", i.get("gender") or ""]
    body = []
    traits = [t for t in (d.get("traits") or []) if isinstance(t, dict)]
    if traits:
        body.append("; ".join(f"{t.get('label')}: {t.get('value')}" for t in traits[:12]))
    a = d.get("authored") or {}
    for k in ("persona", "buying_context", "history_data", "behavior_voice"):
        if a.get(k):
            body.append(str(a[k])[:cut])
    mb = ((d.get("evidence") or {}).get("measured_behavior") or {})
    facets = mb.get("facets") or {}
    if facets:
        body.append("Measured: " + "; ".join(f"{k}={v}" for k, v in list(facets.items())[:10]))
    seg = d.get("segment")
    if _blank(seg):
        seg = facets.get("segment")
    return PersonaView(pid=pid, name=str(i.get("name") or ""), attrs=[x for x in attrs if x], body=body,
                       segment=None if _blank(seg) else str(seg), locale=str(i.get("locale") or ""),
                       tagline=i.get("tagline"), source=d)


def _from_flat(d: dict) -> PersonaView:
    pid = next((str(d[k]) for k in ("persona_id", "pid", "id", "uid") if not _blank(d.get(k))), "")
    text = next((str(d[k]) for k in _FLAT_TEXT_KEYS if not _blank(d.get(k))), None)
    if text is not None:
        body = [text]
    else:
        body = ["; ".join(f"{k}: {v}" for k, v in d.items()
                          if k not in _FLAT_RESERVED and not _blank(v) and not isinstance(v, (dict, list)))]
    seg = d.get("segment")
    return PersonaView(pid=pid, name=str(d.get("name") or ""), body=body,
                       segment=None if _blank(seg) else str(seg), locale=str(d.get("locale") or ""),
                       tagline=d.get("tagline"), source=d)


def as_persona(x: Any, i: int = 0) -> PersonaView:
    """One persona-like item -> PersonaView (see module docstring for accepted forms)."""
    if isinstance(x, PersonaView):
        return x
    if isinstance(x, str):
        return PersonaView(pid=f"p{i + 1:03d}", body=[x], source=x)
    d = None
    if isinstance(x, dict):
        d = x
    elif hasattr(x, "to_dict") and callable(x.to_dict):
        d = x.to_dict()
    if isinstance(d, dict):
        v = _from_talp(d) if "identity" in d else _from_flat(d)
        if not isinstance(x, dict):
            v.source = x
            seg = getattr(x, "segment", None)
            if not _blank(seg):
                v.segment = str(seg)
            pid = getattr(x, "pid", None)
            if not _blank(pid):
                v.pid = str(pid)
    else:
        render = getattr(x, "text", None) or getattr(x, "metin", None)
        txt = render() if callable(render) else str(x)
        v = PersonaView(pid=str(getattr(x, "pid", "") or ""), body=[txt], source=x,
                        segment=getattr(x, "segment", None))
    if not v.pid:
        v.pid = f"p{i + 1:03d}"
    return v


def as_personas(items: Iterable[Any]) -> list[PersonaView]:
    """Persona-like items -> PersonaView list; ids must be unique."""
    out = [as_persona(x, i) for i, x in enumerate(items)]
    seen: set[str] = set()
    for p in out:
        if p.pid in seen:
            raise ValueError(f"duplicate persona id {p.pid!r}")
        seen.add(p.pid)
    return out


def choose_lang(lang: str | None, personas: list[PersonaView]) -> str:
    """'tr' or 'en': explicit value wins; otherwise 'tr' when at least half of the personas have a tr locale."""
    if lang in ("tr", "en"):
        return lang
    tr = sum(1 for p in personas if str(p.locale or "").lower().startswith("tr"))
    return "tr" if personas and tr * 2 >= len(personas) else "en"


# ---------------------------------------------------------------- control arms
def normalize_controls(controls) -> list[str]:
    if controls is None:
        return list(CONTROLS_DEFAULT)
    out = []
    for c in controls:
        k = CONTROL_ALIASES.get(str(c))
        if k is None:
            raise ValueError(f"unknown control arm {c!r}; use {CONTROLS_DEFAULT}")
        if k not in out:
            out.append(k)
    return out


def call_count(n: int, batch: int, splits: int = 1) -> int:
    """Calls needed to ask n personas split into `splits` order groups with `batch` personas per prompt."""
    if n <= 0:
        return 0
    sizes = [len(range(i, n, splits)) for i in range(splits)]
    return sum(math.ceil(b / batch) for b in sizes if b)


def control_plan(n: int, batch: int, controls, splits: int = 1) -> dict:
    """Keep control arms within the main arm's call count where possible: no_persona uses large batches (<=20,
    cheap); persona_swap gets the remaining budget, with a larger batch or a subsample, and always at least one
    full pass. Returns {arm: {"batch", "n", "calls"}} plus "_main_calls"."""
    controls = [c for c in normalize_controls(controls)]
    main = call_count(n, batch, splits)
    plan: dict = {"_main_calls": main}
    left = main
    if "no_persona" in controls:
        b = min(CONTROL_BATCH_MAX, max(batch, n))
        c = call_count(n, b, splits)
        plan["no_persona"] = {"batch": b, "n": n, "calls": c}
        left -= c
    if "persona_swap" in controls:
        left = max(left, call_count(n, CONTROL_BATCH_MAX, splits))
        if left == 0:
            plan["persona_swap"] = {"batch": CONTROL_BATCH_MAX, "n": 0, "calls": 0, "note": "no budget left"}
        else:
            b = batch
            while b < CONTROL_BATCH_MAX and call_count(n, b, splits) > left:
                b += 1
            m = n
            while m > 1 and call_count(m, b, splits) > left:
                m -= 1
            plan["persona_swap"] = {"batch": b, "n": m, "calls": call_count(m, b, splits)}
    return plan


def control_personas(arm: str, personas: list[PersonaView], lang: str = "en", n: int | None = None
                     ) -> list[PersonaView]:
    """Persona list for a control arm; ids (pid) stay the real persona's, only the text changes.
    no_persona: neutral 'typical adult consumer' for each id. persona_swap: cyclic shift (i gets i+1's content,
    keeps its own name and tagline)."""
    arm = CONTROL_ALIASES.get(arm, arm)
    n = len(personas) if n is None else n
    target = list(personas)[:n]
    if arm == "no_persona":
        return [PersonaView(pid=p.pid, name=NEUTRAL_TEXT.get(lang, NEUTRAL_TEXT["en"]), segment=p.segment,
                            locale=p.locale) for p in target]
    if arm == "persona_swap":
        allp = list(personas)
        if len(allp) < 2:
            return []
        out = []
        for i, p in enumerate(target):
            q = allp[(i + 1) % len(allp)]
            out.append(PersonaView(pid=p.pid, name=p.name, attrs=list(q.attrs), body=list(q.body),
                                   segment=p.segment, locale=p.locale, tagline=p.tagline))
        return out
    raise ValueError(f"unknown control arm {arm}")


__all__ = ["PersonaView", "as_persona", "as_personas", "choose_lang", "control_personas", "control_plan",
           "call_count", "normalize_controls", "CONTROLS_DEFAULT", "NEUTRAL_TEXT"]
