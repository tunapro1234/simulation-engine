"""Personas from data: a real table row (demographics + survey answers + measured history) -> TALP Persona.
Ported from the TALP prep `sahne/kod/sahne/sentez.py` ("persona synthesis from data").

Same split as TALP's own approach: measured facts go to `evidence.measured_behavior.facets` (from the source,
unchanged), LLM-written narrative goes to `authored`. A card can show "from data" and "invented" separately.

    m = Mapping(
        id_col="Survey ResponseID",
        fields={"age_band": ("Q-demos-age", {"18 - 24 years": "18-24", ...}),
                "income_tier": ("Q-demos-income", income_fn)},
        facets={"stated_orders": "Q-amazon-use-howmany"},      # measured or stated; copied to facets as is
        segment="income_tier", forbidden=["target_col"])
    personas = from_table(df, m, n=40, seed=0)
    personas = narrate(source, personas, lang="tr")             # optional: LLM backstory (batched, cheap)

    personas = from_table(df, id_col="user_id", forbidden=["bought_next_month"])   # automatic mapping

LEAKAGE RULE: facets hold only what is known BEFORE the prediction time. The target label never goes in;
`from_row` raises if a facet source column is listed in `forbidden`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .schema import AGE_BAND, Persona, enum_for

TALP_FIELDS = ("age_band", "gender", "income_tier", "household_composition", "usage_intensity",
               "commitment_disposition", "brand_awareness")


@dataclass
class Mapping:
    id_col: str                                          # row id column
    fields: dict = field(default_factory=dict)           # talp_field -> column | (column, dict | fn | None) | fn(row)
    facets: dict = field(default_factory=dict)           # facet_name -> column | fn(row)
    text: dict = field(default_factory=dict)             # authored field -> fn(row) (template narrative, no LLM)
    segment: str | Callable | None = None                # talp field name, raw column name or fn(row)
    forbidden: list = field(default_factory=list)        # target/future columns: error if used as a facet
    locale: str = "en-US"
    source_name: str = "data"                            # evidence.measured_receipt contract_id


def age_band(age) -> str | None:
    """Numeric age -> TALP AgeBand (None under 18 or unparseable)."""
    try:
        y = float(age)
    except (TypeError, ValueError):
        return None
    if math.isnan(y) or y < 18:
        return None
    for low, b in ((65, "65+"), (55, "55-64"), (45, "45-54"), (35, "35-44"), (25, "25-34"), (18, "18-24")):
        if y >= low:
            return b
    return None


_GENDER = {"f": "female", "female": "female", "woman": "female", "w": "female",
           "m": "male", "male": "male", "man": "male", "other": "other", "non-binary": "other", "nonbinary": "other"}


def gender_value(v) -> str | None:
    """Common gender spellings -> TALP Gender; unknown -> None."""
    if _is_empty(v):
        return None
    return _GENDER.get(str(v).strip().lower())


def _age_value(v):
    if isinstance(v, str) and v.strip() in AGE_BAND:
        return v.strip()
    return age_band(v)


def _apply(convert, value):
    if convert is None:
        return value
    if isinstance(convert, dict):
        return convert.get(value, convert.get(str(value)))
    return convert(value)


def _is_empty(v) -> bool:
    if v is None or (isinstance(v, str) and v == ""):
        return True
    if isinstance(v, (list, tuple, dict, set)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _plain(v):
    return v.item() if hasattr(v, "item") and not isinstance(v, (str, bytes)) else v


def from_row(row, mapping: Mapping) -> Persona:
    """One row (dict or pandas Series) -> Persona. Out-of-enum TALP values are dropped silently."""
    get = row.get if hasattr(row, "get") else row.__getitem__
    cell = {}
    for name, spec in mapping.fields.items():
        col, convert = spec if isinstance(spec, tuple) else (spec, None)
        v = col(row) if callable(col) else _apply(convert, get(col))
        if _is_empty(v):
            continue
        allowed = enum_for(name)
        if allowed and v not in allowed:
            continue
        cell[name] = v
    facets, claims = {}, {}
    for name, src in mapping.facets.items():
        if not callable(src) and src in mapping.forbidden:
            raise ValueError(f"leakage: '{src}' is a target/future column and cannot be a facet")
        v = src(row) if callable(src) else get(src)
        if _is_empty(v):
            continue
        facets[name] = _plain(v)
        claims[name] = {"contract_id": mapping.source_name, "scope": src if isinstance(src, str) else name}
    pid = str(get(mapping.id_col))
    authored = {k: f(row) for k, f in mapping.text.items()}
    seg = mapping.segment
    if callable(seg):
        segment = seg(row)
    elif seg:
        # a mapped TALP field (enum-checked) wins; otherwise the raw column, so `segment="region"` works and an
        # out-of-enum value (dropped from the traits) still segments the panel
        segment = cell.get(seg)
        if _is_empty(segment):
            segment = get(seg)
    else:
        segment = None
    return Persona.from_dict({
        "identity": {"persona_id": pid, "name": pid, "age_band": cell.get("age_band"), "gender": cell.get("gender"),
                     "locale": mapping.locale, "persona_mode": "b2c"},
        "traits": [{"label": k, "value": v, "group": "socioeconomic" if k == "income_tier" else None}
                   for k, v in cell.items() if k not in ("age_band", "gender")],
        "authored": authored,
        "evidence": {"measured_behavior": {"facets": facets, "claims": claims},
                     "measured_receipt": {"draws": [{"contract_id": mapping.source_name, "contract_version": "1"}]}},
        "segment": None if _is_empty(segment) else str(segment),
    })


def auto_mapping(df: pd.DataFrame, id_col: str, forbidden=(), locale: str = "en-US",
                 source_name: str = "data", segment=None) -> Mapping:
    """Guess a Mapping: columns named like TALP fields (and `age`, `sex`) become identity/traits, every other
    column except `id_col` and `forbidden` becomes a facet. Pass the target column(s) in `forbidden`."""
    forbidden = list(forbidden)
    fields, facets = {}, {}
    for col in df.columns:
        if col == id_col or col in forbidden:
            continue
        low = str(col).strip().lower()
        if low in ("age_band", "age") and "age_band" not in fields:
            fields["age_band"] = (col, _age_value)
        elif low in ("gender", "sex") and "gender" not in fields:
            fields["gender"] = (col, gender_value)
        elif low in TALP_FIELDS and low not in fields:
            fields[low] = (col, lambda v: v.strip().lower() if isinstance(v, str) else v)
        else:
            facets[str(col)] = col
    return Mapping(id_col=id_col, fields=fields, facets=facets, segment=segment, forbidden=forbidden,
                   locale=locale, source_name=source_name)


def from_table(df: pd.DataFrame, mapping: Mapping | None = None, *, id_col: str | None = None,
               fields: dict | None = None, facets: dict | None = None, text: dict | None = None,
               segment=None, forbidden=(), locale: str = "en-US", source_name: str = "data",
               n: int | None = None, seed: int = 0, stratify: str | None = None) -> list[Persona]:
    """Table -> list of Persona.

    Either pass a `Mapping`, or keyword pieces of one (`id_col` + `fields`/`facets`/...). With only `id_col`
    the mapping is guessed by `auto_mapping` (every non-forbidden column becomes a facet, so put the target in
    `forbidden`). `n` samples n rows (`seed`); with `stratify` (a column) the sample is proportional-stratified."""
    if mapping is None:
        if id_col is None:
            raise ValueError("from_table needs a Mapping or id_col")
        if fields is None and facets is None:
            mapping = auto_mapping(df, id_col, forbidden, locale, source_name, segment)
            if text:
                mapping.text = dict(text)
        else:
            mapping = Mapping(id_col=id_col, fields=dict(fields or {}), facets=dict(facets or {}),
                              text=dict(text or {}), segment=segment, forbidden=list(forbidden),
                              locale=locale, source_name=source_name)
    if n is not None and n < len(df):
        if stratify and stratify in df:
            share = df[stratify].value_counts(normalize=True)
            parts = [df[df[stratify] == k].sample(max(1, round(p * n)), random_state=seed)
                     for k, p in share.items() if (df[stratify] == k).any()]
            df = pd.concat(parts).head(n)
        else:
            df = df.sample(n, random_state=seed)
    return [from_row(r, mapping) for _, r in df.iterrows()]


NARRATE_TASK = ("Each person below is a REAL survey respondent; the 'Measured' line lists recorded facts. "
                "Write a short, plausible backstory and first-person voice CONSISTENT with these facts. "
                "Do not invent purchases, numbers or facts that contradict them; keep it modest.")


def narrate(source, personas: list[Persona], lang: str = "en", batch: int = 10, context: str = "") -> list[Persona]:
    """Fill the `authored` fields with an LLM (name, tagline, backstory, voice). Measured facts are untouched.
    `context`: what the data is (e.g. "Amazon.com online retail customers, survey 2023"); without it the LLM may
    invent the sector. Personas are changed in place and marked `extra["narrative"] = "llm"`."""
    task = NARRATE_TASK + (f" Context of the data: {context}. Stay within this context." if context else "")
    answers = source.ask(personas, task,
                         {"name": "str (a plausible first name)", "tagline": "str (≤12 words)",
                          "persona": "str (2-3 sentences)", "behavior_voice": "str (1 sentence, first person)"},
                         batch=batch, tag="narrate", lang=lang)
    for p in personas:
        c = answers.get(p.pid)
        if not c:
            continue
        p.identity.name = c.get("name") or p.identity.name
        p.identity.tagline = c.get("tagline")
        p.authored.persona = c.get("persona") or p.authored.persona
        p.authored.behavior_voice = c.get("behavior_voice")
        p.extra["narrative"] = "llm"
    return personas


def _snake(s) -> str:
    import re
    return re.sub(r"[^0-9a-z]+", "_", str(s).lower()).strip("_")


def to_frames(personas) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Personas (objects or TALP dicts) -> (people, features).
    people: one row per persona: `id`, `seg_<field>` for identity fields, demographic_traits and traits, and
    `persona_text` (tagline + authored texts). features: long `id, item, value` from quantitative_dispositions
    (`q_<dimension>`). Ported from the persona kit's TALP reader (`talp_persona_oku`)."""
    rows, feats = [], []
    for p in personas:
        d = p.to_dict() if isinstance(p, Persona) else p
        idn = d.get("identity", {})
        pid = idn.get("persona_id") or d.get("persona_id")
        r = {"id": pid}
        for k in ("age_band", "gender", "locale", "persona_mode"):
            if idn.get(k) is not None:
                r[f"seg_{k}"] = idn[k]
        for k, v in (idn.get("demographic_traits") or {}).items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                r[f"seg_{_snake(k)}"] = v
        for t in d.get("traits") or []:
            r[f"seg_{_snake(t.get('label'))}"] = t.get("value")
        if d.get("segment") is not None:
            r["segment"] = d["segment"]
        au = d.get("authored") or {}
        parts = [f"{k}: {v}" for k, v in au.items() if isinstance(v, str) and v]
        if idn.get("tagline"):
            parts.insert(0, idn["tagline"])
        r["persona_text"] = "\n".join(parts)
        rows.append(r)
        for q in d.get("quantitative_dispositions") or []:
            try:
                feats.append({"id": pid, "item": f"q_{_snake(q['dimension'])}", "value": float(q["value"])})
            except (KeyError, TypeError, ValueError):
                pass
    return pd.DataFrame(rows), pd.DataFrame(feats, columns=["id", "item", "value"])


__all__ = ["Mapping", "from_row", "from_table", "auto_mapping", "narrate", "age_band", "gender_value",
           "to_frames", "TALP_FIELDS", "NARRATE_TASK"]
