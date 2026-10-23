"""Which persona fields reached the prompt, and which source columns were dropped on the way.

A mapping that names a column the table does not have (``"Age"`` vs ``age``), a column left out of an explicit
Mapping, or values that are empty for part of the rows all lose information silently: the personas still build,
`Persona.text()` just says less (G5#12: 46/801 households lost their age field before the from_table fix). This
report counts it so the evidence card and the limits can say so.

    import simengine.persona as sp
    people = sp.from_table(df, m)
    rep = sp.field_report(people, df, m)     # {"n_fields", "used", "dropped", "per_field_missing", "why", "n_people"}
    report = se.evaluate(..., limits=[sp.as_limit(rep)])   # one line on the card's limits, or:
    sp.attach(report, rep)                     # "persona" + "dusen" rows (+ the limit line) on report["kanit_kart"]
    se.card(report)                            # the Markdown card now shows them

Setting `report["persona_alanlari"]` alone does not reach `se.card(report)`: the card is the stored
`report["kanit_kart"]`, built inside evaluate(); rebuilding it with `kart.kart(report)` loses the seal record and the
notes added after it. `attach` edits the stored card in place (idempotent: a second call replaces the rows).

"used" are the field names present in at least one persona (identity age_band/gender, traits, facets; the
names `Persona.text()` renders), in that order. "per_field_missing" counts, per used field, the personas without
it. "dropped" are names that reached no persona: a mapped field whose column is missing or empty, and a source
column that no mapping entry names (id and `forbidden` columns are not dropped; without a mapping the forbidden
list is unknown, so a target column is listed and the reason says so). "why" gives the reason per dropped name.
"""
from __future__ import annotations

import pandas as pd

from .builders import TALP_FIELDS, Mapping, _id_text, _is_empty


def _present(p) -> list[str]:
    """Field names of one persona that `Persona.text()` renders (non-empty values only)."""
    out = [k for k in ("age_band", "gender") if not _is_empty(getattr(p.identity, k, None))]
    out += [t.label for t in p.traits if not _is_empty(t.value)]
    out += [str(k) for k, v in p.facets.items() if not _is_empty(v)]
    return out


def _aliases(col) -> set[str]:
    """Persona field names `auto_mapping` can give a column."""
    low = str(col).strip().lower()
    al = {str(col)}
    if low in ("age_band", "age"):
        al.add("age_band")
    elif low in ("gender", "sex"):
        al.add("gender")
    elif low in TALP_FIELDS:
        al.add(low)
    return al


def _source(spec):
    col = spec[0] if isinstance(spec, tuple) else spec
    return None if callable(col) or col is None else col


def field_report(people, source_df: pd.DataFrame | None = None, mapping: Mapping | None = None) -> dict:
    """Personas (+ optional source table and Mapping) -> {"n_fields", "used", "dropped", "per_field_missing",
    "why", "n_people"}. See the module docstring."""
    people = list(people)
    seen: dict[str, int] = {}
    for p in people:
        for k in dict.fromkeys(_present(p)):
            seen[k] = seen.get(k, 0) + 1
    used = list(seen)
    why: dict[str, str] = {}
    cols = list(source_df.columns) if source_df is not None else []
    if mapping is not None:
        named = set()
        for name, spec in list(mapping.fields.items()) + list(mapping.facets.items()):
            src = _source(spec)
            if src is not None:
                named.add(src)
            if name in seen:
                continue
            if src is not None and source_df is not None and src not in source_df.columns:
                why[name] = f"mapped to {src!r}, not a column of the table"
            else:
                why[name] = "empty or unmapped in every row"
        skip = named | set(mapping.forbidden) | ({mapping.id_col} if mapping.id_col is not None else set())
        callables = bool(mapping.text) or any(_source(s) is None
                                               for s in list(mapping.fields.values()) + list(mapping.facets.values()))
        for c in cols:
            if c not in skip and str(c) not in why and not (_aliases(c) & set(used)):
                why[str(c)] = ("not in the mapping by name (a callable field or text may still read it)" if callables
                               else "not in the mapping")
    elif source_df is not None:
        pids = {p.pid for p in people}
        for c in cols:
            if _aliases(c) & set(used):
                continue
            col = source_df[c]
            if pids and pids <= {_id_text(v) for v in col if not _is_empty(v)}:
                continue                                   # the id column
            why[str(c)] = "no persona has it: empty, or a target/forbidden column"
    n = len(people)
    return {"n_fields": len(used), "used": used, "dropped": list(why), "why": why,
            "per_field_missing": {k: n - seen[k] for k in used}, "n_people": n}


def as_limit(report: dict) -> str:
    """One line for evaluate(limits=[...]): fields that reached the prompt, partly missing ones, dropped ones."""
    n = report.get("n_people")
    parts = [f"Persona fields: {report.get('n_fields')} reached the prompt"
             + (f" for {n} people" if n is not None else "")
             + (f" ({', '.join(report.get('used') or [])})" if report.get("used") else "")]
    miss = [f"{k} {m}/{n}" if n else f"{k} {m}" for k, m in (report.get("per_field_missing") or {}).items() if m]
    if miss:
        parts.append("missing in some personas: " + ", ".join(miss))
    why = report.get("why") or {}
    dropped = report.get("dropped") or []
    parts.append("dropped: " + (", ".join(f"{d} ({why[d]})" if d in why else str(d) for d in dropped)
                                if dropped else "none"))
    return "; ".join(parts) + "."


def attach(report: dict, field_rep: dict, limit: bool = True) -> dict:
    """Put a `field_report` on an evaluate() report: report["persona_alanlari"], the "persona" / "dusen" rows on the
    stored card (report["kanit_kart"], edited in place, so the seal and later notes stay) and, with `limit`, the
    `as_limit` line in the limits (report["sinirlar"] and the card's). Idempotent. Returns the report."""
    from ..evidence import kart as _kart         # lazy: persona does not import evidence at module load
    report["persona_alanlari"] = field_rep
    k = report.get("kanit_kart")
    if k is None:
        report["kanit_kart"] = k = _kart.kart(report)
    _kart.persona_ekle(k, field_rep)
    if limit:
        old = [x for x in report.get("sinirlar") or [] if not str(x).startswith("Persona fields:")]
        line = as_limit(field_rep)
        report["sinirlar"] = old + [line]
        k["yanilgi"] = [x for x in k.get("yanilgi") or [] if not str(x).startswith("Persona fields:")] + [line]
    return report
