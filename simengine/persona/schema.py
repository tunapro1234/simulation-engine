"""TALP-compatible persona schema (ported from the TALP prep `sahne/kod/sahne/sema.py`).

Field names are kept identical to the names seen in TALP's public JS bundles, so a TALP persona JSON
can be read with `Persona.from_dict` and what we produce is written in the same shape.

    spec = AudienceSpec(target_description="young parents in Istanbul", locale="tr-TR",
                        audience=Audience(income_tier=["middle"], household_composition=["parents"],
                                          identity_text="...", behavioral_anchor="..."))
    p = Persona.from_dict(json.loads(text)); p.to_dict()

Every dataclass has an `extra` dict: unknown keys from the input land there (no data loss) and are
written back at the top level by `to_dict`. `None` and empty values are dropped from `to_dict`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

# ---- enum values (TALP persona-briefs/options)
INCOME_TIER = ("low", "lower_middle", "middle", "upper_middle", "affluent")
HOUSEHOLD = ("single_no_kids", "couple_no_kids", "parents", "empty_nesters", "multigen", "other")
USAGE = ("light", "regular", "heavy")
COMMITMENT = ("multi_platform", "single_platform_loyal", "occasional", "new_to_category")
BRAND_AWARENESS = ("unaware", "aware_not_examined", "evaluating", "experienced")
AGE_BAND = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")
GENDER = ("male", "female", "other")
PERSONA_MODE = ("b2c", "b2b")

# audience field -> allowed values
ENUMS = {
    "income_tier": INCOME_TIER,
    "household_composition": HOUSEHOLD,
    "usage_intensity": USAGE,
    "commitment_disposition": COMMITMENT,
    "brand_awareness": BRAND_AWARENESS,
    "age_range": AGE_BAND,
    "gender_mix": GENDER,
}
# persona-level field name -> audience-level enum name (identity uses age_band/gender)
PERSONA_TO_AUDIENCE = {"age_band": "age_range", "gender": "gender_mix"}

# readable labels for UI and prompts (TR/EN)
LABELS = {
    "income_tier": {"tr": "Gelir", "en": "Income"},
    "household_composition": {"tr": "Hane", "en": "Household"},
    "usage_intensity": {"tr": "Kullanım", "en": "Usage"},
    "commitment_disposition": {"tr": "Bağlılık", "en": "Commitment"},
    "brand_awareness": {"tr": "Kategori farkındalığı", "en": "Brand awareness"},
    "age_band": {"tr": "Yaş", "en": "Age"},
    "gender": {"tr": "Cinsiyet", "en": "Gender"},
}

# absolute words: TALP reads them as "everyone in the panel without exception"
ABSOLUTE_WORDS = ("never", "always", "all ", "only ", "asla", "her zaman", "hepsi", "sadece", "hiçbir zaman")


def enum_for(name: str) -> tuple | None:
    """Allowed values for a persona/audience field name (age_band and gender map to the audience enums)."""
    return ENUMS.get(PERSONA_TO_AUDIENCE.get(name, name))


def _clean(d: Any) -> Any:
    """Drop None and empty fields (keeps the TALP JSON lean). Cleans children first, so a sub-object that
    only held empty fields disappears too."""
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            v = _clean(v)
            if v is None or (isinstance(v, (str, list, dict)) and len(v) == 0):
                continue
            out[k] = v
        return out
    if isinstance(d, list):
        return [_clean(v) for v in d]
    return d


def _load(cls, d: dict | None):
    """Build a dataclass from a dict; unknown keys go to `extra` (no data loss)."""
    if d is None:
        return cls()
    if isinstance(d, cls):
        return d
    names = {f.name for f in fields(cls)}
    kw, extra = {}, {}
    for k, v in d.items():
        if k == "extra" and isinstance(v, dict):
            extra.update(v)
        elif k in names:
            sub = SUBCLASS.get((cls.__name__, k))
            if sub is not None:
                v = [_load(sub, x) for x in v] if isinstance(v, list) else _load(sub, v)
            kw[k] = v
        else:
            extra[k] = v
    if "extra" in names:
        kw["extra"] = extra
    return cls(**kw)


def _plain(obj: Any) -> Any:
    """Dataclass tree -> plain dicts/lists; every `extra` dict is merged into its own object's level."""
    if is_dataclass(obj) and not isinstance(obj, type):
        d = {f.name: _plain(getattr(obj, f.name)) for f in fields(obj) if f.name != "extra"}
        for k, v in (getattr(obj, "extra", None) or {}).items():
            d.setdefault(k, _plain(v))
        return d
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    return obj


class _DictMixin:
    def to_dict(self) -> dict:
        return _clean(_plain(self))

    @classmethod
    def from_dict(cls, d: dict):
        return _load(cls, d)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)


# ---------------------------------------------------------------- input: audience panel brief
@dataclass
class Audience(_DictMixin):
    income_tier: list[str] = field(default_factory=list)
    household_composition: list[str] = field(default_factory=list)
    usage_intensity: list[str] = field(default_factory=list)
    commitment_disposition: list[str] = field(default_factory=list)
    brand_awareness: list[str] = field(default_factory=list)
    age_range: list[str] = field(default_factory=list)
    gender_mix: list[str] = field(default_factory=list)
    trait_requirements: list[dict] = field(default_factory=list)   # [{dimension, values[], required}]
    identity_text: str = ""
    behavioral_anchor: str = ""
    market_context_hint: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class AudienceSpec(_DictMixin):
    target_description: str = ""
    persona_mode: str = "b2c"
    locale: str = "en-US"
    vertical: str = "general"
    occupation_pool: list[str] = field(default_factory=list)
    source_brief: str | None = None
    target_size: int = 50
    audience: Audience = field(default_factory=Audience)
    behavior_sketch: list = field(default_factory=list)            # <=12 sentences; str or {text, drafter_invented}
    extra: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Counterpart of the TALP UI warnings: out-of-enum value, panel size, absolute wording, missing anchors."""
        warn = []
        a = self.audience
        for name, allowed in ENUMS.items():
            for v in getattr(a, name) or []:
                if v not in allowed:
                    warn.append(f"{name}: '{v}' is not in the enum ({', '.join(allowed)})")
        if not 10 <= self.target_size <= 200:
            warn.append(f"target_size {self.target_size}: TALP accepts 10-200")
        if self.persona_mode not in PERSONA_MODE:
            warn.append(f"persona_mode '{self.persona_mode}' is not b2c/b2b")
        if len(self.behavior_sketch) > 12:
            warn.append("behavior_sketch has more than 12 sentences")
        for c in [self.target_description, a.identity_text, a.behavioral_anchor] + [
                s if isinstance(s, str) else s.get("text", "") for s in self.behavior_sketch]:
            low = f" {c.lower()} "
            hit = [m.strip() for m in ABSOLUTE_WORDS if m in low]
            if hit:
                warn.append(f"absolute wording ({', '.join(hit)}): read as everyone in the panel without "
                            f"exception -> «{c[:60]}»")
        if not a.identity_text:
            warn.append("audience.identity_text is required and empty")
        if not a.behavioral_anchor:
            warn.append("audience.behavioral_anchor is required and empty")
        return warn

    def language(self) -> str:
        """'tr' for a Turkish locale, otherwise 'en'."""
        return "tr" if self.locale.lower().startswith("tr") else "en"

    # aliases used by the TALP research code
    denetle = validate
    dil = language


# ---------------------------------------------------------------- persona object
@dataclass
class Identity(_DictMixin):
    persona_id: str = ""
    name: str = ""
    gender: str | None = None
    age_band: str | None = None
    persona_mode: str = "b2c"
    locale: str = "en-US"
    tagline: str | None = None
    demographic_traits: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


@dataclass
class Trait(_DictMixin):
    label: str = ""
    value: Any = None
    group: str | None = None      # "socioeconomic" is shown separately in TALP
    extra: dict = field(default_factory=dict)


@dataclass
class Trigger(_DictMixin):
    text: str | None = None
    ui_element: str | None = None
    personal_stake: str | None = None
    root_trait: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Triggers(_DictMixin):
    frustrations: list[Trigger] = field(default_factory=list)
    satisfactions: list[Trigger] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class Authored(_DictMixin):
    persona: str = ""                     # "Who they are" (backstory)
    consumer_profile: str | None = None
    history_data: str | None = None       # "Usage history"
    behavior_voice: str | None = None     # "In their words"
    emotional_biography: str | None = None
    buying_context: str | None = None
    triggers: Triggers = field(default_factory=Triggers)
    extra: dict = field(default_factory=dict)


@dataclass
class Behavioral(_DictMixin):
    media_habits: list[str] = field(default_factory=list)
    motivations: list[str] = field(default_factory=list)
    frustrations: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class Disposition(_DictMixin):
    dimension: str = ""
    value: float | None = None
    unit: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Evidence(_DictMixin):
    """Measured facts sampled/derived from a source. Personas synthesized from data put the source row here."""
    measured_behavior: dict = field(default_factory=dict)   # {facets{}, claims{}, labels{}}
    measured_receipt: dict = field(default_factory=dict)    # {draws:[{contract_id, contract_version}]}
    extra: dict = field(default_factory=dict)


@dataclass
class Persona(_DictMixin):
    identity: Identity = field(default_factory=Identity)
    traits: list[Trait] = field(default_factory=list)
    authored: Authored = field(default_factory=Authored)
    behavioral: Behavioral = field(default_factory=Behavioral)
    quantitative_dispositions: list[Disposition] = field(default_factory=list)
    evidence: Evidence = field(default_factory=Evidence)
    segment: str | None = None            # simengine field: panel/evidence segment key
    extra: dict = field(default_factory=dict)

    @property
    def pid(self) -> str:
        return self.identity.persona_id

    def trait(self, label: str, default=None):
        for t in self.traits:
            if t.label == label:
                return t.value
        return default

    @property
    def facets(self) -> dict:
        """Measured facts (evidence.measured_behavior.facets); empty dict when none."""
        return (self.evidence.measured_behavior or {}).get("facets", {}) or {}

    def text(self, lang: str = "en", clip: int | None = 600, max_facets: int = 10) -> str:
        """Short persona description used inside prompts.

        Narrative fields written by an LLM (persona, buying_context, behavior_voice) are clipped to `clip`
        characters (None = no clipping), as in the research code. Data-derived fields
        (consumer_profile, history_data; filled by `from_history`) are never clipped."""
        i = self.identity
        head = [f"{i.name or i.persona_id}", i.age_band and f"age {i.age_band}", i.gender]
        lines = [", ".join(x for x in head if x)]
        if self.traits:
            lines.append("; ".join(f"{t.label}: {t.value}" for t in self.traits[:12]))
        a = self.authored
        cut = (lambda s: s) if clip is None else (lambda s: s[:clip])
        for x, clipped in ((a.persona, True), (a.buying_context, True), (a.consumer_profile, False),
                           (a.history_data, False), (a.behavior_voice, True)):
            if x:
                lines.append(cut(x) if clipped else x)
        f = self.facets
        if f and max_facets:
            lines.append("Measured: " + "; ".join(f"{k}={v}" for k, v in list(f.items())[:max_facets]))
        return "\n".join(lines)

    def card(self) -> dict:
        """UI card (a server can put this into JSON)."""
        return {"id": self.pid, "name": self.identity.name, "age": self.identity.age_band,
                "gender": self.identity.gender, "segment": self.segment, "tagline": self.identity.tagline,
                "traits": [{"label": t.label, "value": t.value} for t in self.traits],
                "story": self.authored.persona, "voice": self.authored.behavior_voice,
                "evidence": self.facets}

    def copy(self) -> "Persona":
        return Persona.from_dict(json.loads(json.dumps(self.to_dict(), default=str)))

    # aliases used by the TALP research code
    metin = text
    kart = card


# sub-dataclass map (so _load can build nested objects)
SUBCLASS = {
    ("AudienceSpec", "audience"): Audience,
    ("Persona", "identity"): Identity,
    ("Persona", "traits"): Trait,
    ("Persona", "authored"): Authored,
    ("Persona", "behavioral"): Behavioral,
    ("Persona", "quantitative_dispositions"): Disposition,
    ("Persona", "evidence"): Evidence,
    ("Authored", "triggers"): Triggers,
    ("Triggers", "frustrations"): Trigger,
    ("Triggers", "satisfactions"): Trigger,
}


def load_personas(data) -> list[Persona]:
    """JSON text / parsed file content / list / {"personas": [...]} -> list of Persona.
    A flat item (identity fields at the top level, with `persona_id`) is accepted too."""
    if isinstance(data, (bytes, bytearray)):
        data = data.decode()
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict):
        data = data.get("personas") or data.get("items") or [data]
    out = []
    for x in data:
        if is_dataclass(x):
            out.append(x)
        elif isinstance(x, dict) and "identity" not in x and "persona_id" in x:
            out.append(Persona.from_dict({"identity": x}))   # flat form: identity fields at the top
        else:
            out.append(Persona.from_dict(x))
    return out


def dump_personas(personas, **kw) -> str:
    """List of Persona -> TALP-style JSON text {"personas": [...]}."""
    return json.dumps({"personas": [p.to_dict() for p in personas]}, ensure_ascii=False, default=str, **kw)


__all__ = ["Audience", "AudienceSpec", "Identity", "Trait", "Trigger", "Triggers", "Authored", "Behavioral",
           "Disposition", "Evidence", "Persona", "load_personas", "dump_personas", "enum_for", "ENUMS", "LABELS",
           "ABSOLUTE_WORDS", "INCOME_TIER", "HOUSEHOLD", "USAGE", "COMMITMENT", "BRAND_AWARENESS", "AGE_BAND",
           "GENDER", "PERSONA_MODE"]
