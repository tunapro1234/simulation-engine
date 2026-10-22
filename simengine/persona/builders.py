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
    personas = from_table(df, id_col=None, forbidden=["target"])     # no usable id column: ids row0, row1, ...

IDS: `from_table` checks the id column first and raises ValueError with the number of empty/NaN and duplicated
ids (an empty id never becomes the string "None" or "nan"). `id_col=None` uses the row number (position in the
table before sampling) as the id: "row0", "row1", ...

VALUE LABELS (codebook codes): `from_table(df, ..., value_labels={"Gender": {1: "male", 2: "female"}})` (or
`Mapping.value_labels`) turns codes into labels before enum mapping and text rendering, so the persona reads
"female" and not "Gender: 2". Codes match as numbers or strings (2, 2.0, "2" and "02" are one code); an unknown
code is kept as is with one warning per column (counts per code). The id column is never relabelled.
`read_codebook(table)` builds the dict from a (variable, code, label) or (code, label) table;
`apply_value_labels(df, labels)` relabels a whole table.

    labels = read_codebook(codebook_df)                       # columns: variable, code, label
    personas = from_table(df, id_col="rid", value_labels=labels, forbidden=["target"])

POOL: `pool_select(df, K, method="random"|"stratified"|"kmeans", features=, strata=, seed=)` picks K rows (index
labels) for a small persona pool; k-means takes the row nearest each centroid on standardised features.
`pool_curve(df, sizes, score_fn, method=...)` scores pools of growing size (how big must the pool be?).

    idx = pool_select(df, 20, method="kmeans", features=["visits", "spend"], seed=0)
    panel = from_table(df.loc[idx], id_col="rid", forbidden=["target"])
    curve = pool_curve(df, [10, 20, 50], score, method=["random", "kmeans"], features=["visits", "spend"])

LEAKAGE RULE: facets hold only what is known BEFORE the prediction time. The target label never goes in;
`from_row` raises if a facet source column is listed in `forbidden`.
"""
from __future__ import annotations

import math
import os
import warnings
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from .schema import AGE_BAND, Persona, enum_for

TALP_FIELDS = ("age_band", "gender", "income_tier", "household_composition", "usage_intensity",
               "commitment_disposition", "brand_awareness")


@dataclass
class Mapping:
    id_col: str | None                                   # row id column; None = row number ("row<i>")
    fields: dict = field(default_factory=dict)           # talp_field -> column | (column, dict | fn | None) | fn(row)
    facets: dict = field(default_factory=dict)           # facet_name -> column | fn(row)
    text: dict = field(default_factory=dict)             # authored field -> fn(row) (template narrative, no LLM)
    segment: str | Callable | None = None                # talp field name, raw column name or fn(row)
    forbidden: list = field(default_factory=list)        # target/future columns: error if used as a facet
    locale: str = "en-US"
    source_name: str = "data"                            # evidence.measured_receipt contract_id
    value_labels: dict = field(default_factory=dict)     # column -> {code: label}, applied before everything else


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


_GENDER_OTHER = ("non-binary", "nonbinary", "non binary", "third gender", "genderqueer", "gender non-conforming")


def gender_value(v) -> str | None:
    """Common gender spellings -> TALP Gender; codebook wordings such as "Non-binary / third gender" -> "other";
    unknown (e.g. "Prefer not to say") -> None."""
    if _is_empty(v):
        return None
    s = str(v).strip().lower()
    if s in _GENDER:
        return _GENDER[s]
    return "other" if any(w in s for w in _GENDER_OTHER) else None


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


def _code_key(v):
    """Codebook code -> comparison key: 2, 2.0, "2", " 02 " and np.int64(2) all give 2; other text is stripped."""
    v = _plain(v)
    if isinstance(v, (bool, int)):
        return v
    if isinstance(v, str):
        s = v.strip()
        try:
            v = float(s)
        except ValueError:
            return s
        if not math.isfinite(v):
            return s
    if isinstance(v, float):
        return int(v) if v.is_integer() else v
    return v


def _label_map(labels: dict, column=None) -> dict:
    out = {}
    for code, label in labels.items():
        k = _code_key(code)
        if k in out and out[k] != label:
            raise ValueError(f"value_labels[{column!r}]: codes {code!r} and an earlier one are the same code "
                             f"({k!r}) with different labels ({out[k]!r} vs {label!r})")
        out[k] = label
    return out


def _label_value(v, lm: dict, unknown: Counter):
    """Label one cell: empty stays empty, a known code becomes its label, anything else is kept and counted."""
    if _is_empty(v):
        return v
    try:
        return lm[_code_key(v)]
    except (KeyError, TypeError):
        unknown[str(_plain(v))] += 1
        return _plain(v)


def warn_unlabelled(unlabelled: dict, value_labels: dict | None = None, n_rows: int | None = None) -> None:
    """One UserWarning per column whose codes have no label: counts per code (the values are kept as is).
    `unlabelled`: {column: Counter(code -> rows)}, filled by `from_row(..., unlabelled=...)`."""
    for col, cnt in unlabelled.items():
        if not cnt:
            continue
        total = sum(cnt.values())
        vals = ", ".join(f"{v!r} x{k}" for v, k in cnt.most_common(8)) + (", ..." if len(cnt) > 8 else "")
        known = list((value_labels or {}).get(col, {}))
        of = f" of {n_rows}" if n_rows is not None else ""
        warnings.warn(f"value_labels[{col!r}]: {total}{of} rows have codes without a label ({vals}); kept as is. "
                      f"Labelled codes: {', '.join(map(repr, known[:12]))}{', ...' if len(known) > 12 else ''}.",
                      UserWarning, stacklevel=3)


def apply_value_labels(df: pd.DataFrame, value_labels: dict, *, skip=()) -> pd.DataFrame:
    """Copy of `df` with codebook codes replaced by labels; `value_labels` = {column: {code: label}}.
    Codes match as numbers or strings (2, 2.0, "2" and "02" are one code). Empty cells stay empty. A code without a
    label is kept as is, with one UserWarning per column (counts per code); a column missing from `df` warns too.
    `skip`: columns to leave alone (e.g. the id column)."""
    out = df.copy()
    missing = [c for c in value_labels if c not in skip and c not in set(out.columns)]
    if missing:
        warnings.warn(f"value_labels: columns not in the table, ignored: {', '.join(map(repr, missing))}",
                      UserWarning, stacklevel=2)
    unlabelled: dict = {}
    for col, labels in value_labels.items():
        if col in skip or col not in out.columns:
            continue
        lm, unknown = _label_map(labels, col), Counter()
        out[col] = out[col].astype(object).map(lambda v: _label_value(v, lm, unknown))
        unlabelled[col] = unknown
    warn_unlabelled(unlabelled, value_labels, len(out))
    return out


def _labelled_row(row, mapping: "Mapping", unlabelled: dict):
    """Row with `mapping.value_labels` applied (the id column is never relabelled); same type as the input."""
    labels = {c: (_label_map(m, c), unlabelled.setdefault(c, Counter())) for c, m in mapping.value_labels.items()
              if c != mapping.id_col}

    def lab(c, v):
        return _label_value(v, *labels[c]) if c in labels else v

    if isinstance(row, pd.Series):
        return pd.Series({c: lab(c, v) for c, v in row.items()}, dtype=object, name=row.name)
    if hasattr(row, "items"):
        return {c: lab(c, v) for c, v in row.items()}
    raise TypeError("value_labels need a dict or pandas Series row")


def read_codebook(table, *, variable: str | None = None, columns=None) -> dict:
    """Codebook table -> `value_labels` dict {variable: {code: label}}.
    Three columns (variable, code, label): one entry per variable; a blank variable cell repeats the one above
    (codebooks often list the variable once). Two columns (code, label): pass `variable=`. `columns` picks
    (variable, code, label) or (code, label) out of a wider table. `table`: DataFrame or a .csv/.tsv path.
    Codes are normalised like `from_table` compares them (2, "2" and "02" -> 2); one code with two different
    labels raises ValueError."""
    if isinstance(table, (str, os.PathLike)):
        path = str(table)
        sep = "\t" if path.lower().endswith((".tsv", ".tab", ".txt")) else ","
        table = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    t = table[list(columns)] if columns is not None else table
    if t.shape[1] == 2:
        if variable is None:
            raise ValueError("a two-column codebook (code, label) needs variable=")
        rows = [(variable, c, lab) for c, lab in t.itertuples(index=False)]
    elif t.shape[1] == 3:
        var = t.iloc[:, 0].map(lambda v: None if _is_empty(v) or not str(v).strip() else str(v).strip()).ffill()
        rows = list(zip(var, t.iloc[:, 1], t.iloc[:, 2]))
    else:
        raise ValueError(f"codebook has {t.shape[1]} columns; pass columns=(variable, code, label) or (code, label)")
    out: dict = {}
    for var, code, label in rows:
        if _is_empty(var) or _is_empty(code) or _is_empty(label) or not str(code).strip():
            continue
        label = label.strip() if isinstance(label, str) else _plain(label)
        m, k = out.setdefault(var, {}), _code_key(code)
        if k in m and m[k] != label:
            raise ValueError(f"codebook {var!r}: code {k!r} has two labels ({m[k]!r}, {label!r})")
        m[k] = label
    return out


ROW_ID_PREFIX = "row"


def _id_text(v) -> str:
    return str(_plain(v))


def from_row(row, mapping: Mapping, *, pid: str | None = None,
             unlabelled: dict | None = None) -> Persona:
    """One row (dict or pandas Series) -> Persona. Out-of-enum TALP values are dropped silently.
    `pid` overrides the id column (from_table passes the row number when `mapping.id_col` is None).
    An empty/NaN id raises ValueError (it never becomes the string "None").
    `mapping.value_labels` are applied to the row first; `unlabelled`: a dict collecting {column: Counter(code)}
    of codes without a label for one aggregated warning (`warn_unlabelled`); None = warn here for this row."""
    if mapping.value_labels:
        seen = {} if unlabelled is None else unlabelled
        row = _labelled_row(row, mapping, seen)
        if unlabelled is None:
            warn_unlabelled(seen, mapping.value_labels)
    get = row.get if hasattr(row, "get") else row.__getitem__
    if pid is None:
        if mapping.id_col is None:
            raise ValueError("mapping.id_col is None: pass pid= (from_table numbers the rows for you)")
        raw = get(mapping.id_col)
        if _is_empty(raw) or not _id_text(raw).strip():
            raise ValueError(f"empty id in column {mapping.id_col!r}; fill it, drop the row, or use "
                             "from_table(..., id_col=None) to number the rows")
        pid = _id_text(raw)
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


def auto_mapping(df: pd.DataFrame, id_col: str | None, forbidden=(), locale: str = "en-US",
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


def check_ids(df: pd.DataFrame, id_col: str) -> None:
    """Raise ValueError when `id_col` is missing or has empty/NaN or duplicated values (counts in the message)."""
    if id_col not in df.columns:
        raise ValueError(f"id column {id_col!r} is not in the table; pass id_col=None to number the rows")
    col = df[id_col]
    empty = col.map(lambda v: _is_empty(v) or not _id_text(v).strip())
    keys = col[~empty].map(_id_text)
    dup_rows = keys.duplicated(keep=False)
    n_empty, n_dup_rows = int(empty.sum()), int(dup_rows.sum())
    if n_empty or n_dup_rows:
        parts = []
        if n_empty:
            parts.append(f"{n_empty} empty/NaN ids")
        if n_dup_rows:
            parts.append(f"{n_dup_rows} rows share {int(keys[dup_rows].nunique())} duplicated ids "
                         f"(e.g. {keys[dup_rows].iloc[0]!r})")
        raise ValueError(f"id column {id_col!r} ({len(df)} rows): " + "; ".join(parts) +
                         f". Drop or fix those rows (e.g. df.dropna(subset=[{id_col!r}]).drop_duplicates({id_col!r})),"
                         " or pass id_col=None to use row numbers as ids.")


def from_table(df: pd.DataFrame, mapping: Mapping | None = None, *, id_col: str | None = None,
               fields: dict | None = None, facets: dict | None = None, text: dict | None = None,
               segment=None, forbidden=(), locale: str = "en-US", source_name: str = "data",
               n: int | None = None, seed: int = 0, stratify: str | None = None,
               value_labels: dict | None = None) -> list[Persona]:
    """Table -> list of Persona.

    Either pass a `Mapping`, or keyword pieces of one (`id_col` + `fields`/`facets`/...). With only `id_col`
    the mapping is guessed by `auto_mapping` (every non-forbidden column becomes a facet, so put the target in
    `forbidden`). `n` samples n rows (`seed`); with `stratify` (a column) the sample is proportional-stratified.

    `value_labels` {column: {code: label}} (merged over `mapping.value_labels`) relabels codebook codes before
    the enum mapping and the text; unknown codes are kept with one warning per column (see the module docstring).
    The id column is checked first: empty/NaN or duplicated ids raise ValueError with their counts. `id_col=None`
    (or a Mapping with id_col None) numbers the rows instead: "row<i>", i = position in `df` before sampling."""
    if mapping is None:
        if fields is None and facets is None:
            mapping = auto_mapping(df, id_col, forbidden, locale, source_name, segment)
            if text:
                mapping.text = dict(text)
        else:
            mapping = Mapping(id_col=id_col, fields=dict(fields or {}), facets=dict(facets or {}),
                              text=dict(text or {}), segment=segment, forbidden=list(forbidden),
                              locale=locale, source_name=source_name)
    if value_labels:
        mapping = replace(mapping, value_labels={**mapping.value_labels, **value_labels})
    if mapping.id_col is None:
        df = df.reset_index(drop=True)          # index label = row number, kept through sampling
    else:
        check_ids(df, mapping.id_col)
    if n is not None and n < len(df):
        if stratify and stratify in df:
            share = df[stratify].value_counts(normalize=True)
            parts = [df[df[stratify] == k].sample(max(1, round(p * n)), random_state=seed)
                     for k, p in share.items() if (df[stratify] == k).any()]
            df = pd.concat(parts).head(n)
        else:
            df = df.sample(n, random_state=seed)
    if mapping.value_labels:                    # once for the table: one warning per column
        df = apply_value_labels(df, mapping.value_labels, skip=(mapping.id_col,))
        mapping = replace(mapping, value_labels={})
    if mapping.id_col is None:
        return [from_row(r, mapping, pid=f"{ROW_ID_PREFIX}{i}") for i, r in df.iterrows()]
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


POOL_METHODS = ("random", "stratified", "kmeans")


def _strata_positions(df: pd.DataFrame, K: int, strata, rng) -> np.ndarray:
    """Proportional allocation (largest remainder; a stratum smaller than n/K may get 0), random rows within."""
    if strata is None:
        raise ValueError("method='stratified' needs strata= (a column or a list of columns)")
    cols = [strata] if isinstance(strata, str) else list(strata)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"strata columns not in the table: {missing}")
    keys = df[cols].astype(str).agg("|".join, axis=1) if len(cols) > 1 else df[cols[0]].astype(str)
    groups = [np.flatnonzero((keys == k).to_numpy()) for k in sorted(keys.unique())]
    sizes = np.array([len(g) for g in groups])
    quota = K * sizes / len(df)
    alloc = np.floor(quota).astype(int)
    for j in np.argsort(-(quota - alloc), kind="stable")[:K - int(alloc.sum())]:
        alloc[j] += 1                           # quota < size (K < n), so alloc never exceeds the stratum
    return np.concatenate([rng.choice(g, a, replace=False) for g, a in zip(groups, alloc) if a])


def _feature_matrix(df: pd.DataFrame, features) -> np.ndarray:
    """Standardised (z-score) feature matrix; text columns are one-hot, NaN -> column mean (0)."""
    if features is None:
        raise ValueError("method='kmeans' needs features= (the columns to cluster on; keep ids and targets out)")
    cols = [features] if isinstance(features, str) else list(features)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"features not in the table: {missing}")
    parts = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            parts.append(pd.to_numeric(s, errors="coerce").astype(float).to_frame())
        else:
            parts.append(pd.get_dummies(s.astype("string"), prefix=str(c), dtype=float))
    X = pd.concat(parts, axis=1).to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # all-NaN column
        mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    X = (X - mu) / sd
    X[~np.isfinite(X)] = 0.0
    return X


def _sqdist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    d = (X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None, :]
    return np.maximum(d, 0.0)


def _kmeans(X: np.ndarray, K: int, rng, n_init: int = 4, max_iter: int = 100) -> np.ndarray:
    """Lloyd's k-means with k-means++ starts (numpy only); returns the centroids of the best of `n_init` runs."""
    n, best = len(X), None
    for _ in range(max(1, n_init)):
        idx = [int(rng.integers(n))]
        d2 = _sqdist(X, X[idx])[:, 0]
        for _ in range(1, K):
            p = d2.copy()
            p[idx] = 0.0
            if p.sum() <= 0:                    # fewer distinct rows than K: any unused row
                p = np.ones(n)
                p[idx] = 0.0
            idx.append(int(rng.choice(n, p=p / p.sum())))
            d2 = np.minimum(d2, _sqdist(X, X[idx[-1:]])[:, 0])
        C = X[idx].copy()
        for _ in range(max_iter):
            d = _sqdist(X, C)
            lab = d.argmin(1)
            counts = np.bincount(lab, minlength=K)
            sums = np.zeros_like(C)
            np.add.at(sums, lab, X)
            new = C.copy()
            full = counts > 0
            new[full] = sums[full] / counts[full, None]
            if (~full).any():                   # empty cluster: restart it at the worst-fitted rows
                far = np.argsort(-d[np.arange(n), lab], kind="stable")[:int((~full).sum())]
                new[~full] = X[far]
            done = np.allclose(new, C)
            C = new
            if done:
                break
        inertia = float(_sqdist(X, C).min(1).sum())
        if best is None or inertia < best[0]:
            best = (inertia, C)
    return best[1]


def _nearest_unique(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """One distinct row per centroid: centroids with the closest row choose first."""
    d = _sqdist(X, C)
    out = []
    for k in np.argsort(d.min(0), kind="stable"):
        col = d[:, k]
        i = int(col.argmin())
        out.append(i)
        d[i, :] = np.inf
    return np.array(out)


def pool_select(df: pd.DataFrame, K: int, *, method: str = "random", features=None, strata=None, seed: int = 0,
                n_init: int = 4, max_iter: int = 100) -> pd.Index:
    """Pick K rows of `df` for a persona pool; returns their index labels in table order (`df.loc[idx]`).

    random      K rows uniformly; for one seed a smaller pool is a subset of a larger one (nested curves).
    stratified  proportional to the `strata` column(s) (largest remainder), random rows within each stratum.
    kmeans      k-means (K clusters, k-means++ starts, `n_init`) on standardised `features` (text columns
                one-hot, NaN = mean); the row nearest each centroid, all distinct. Each member stands for a
                cluster of a different size, so an unweighted k-means pool over-represents sparse regions (tails)
                and is not a sample of the table.
    K >= len(df) returns every row. `df.index` must be unique."""
    if method not in POOL_METHODS:
        raise ValueError(f"method must be one of {POOL_METHODS}, got {method!r}")
    if isinstance(K, bool) or int(K) != K or K < 1:
        raise ValueError(f"K must be a positive int, got {K!r}")
    if not df.index.is_unique:
        raise ValueError("df.index has duplicates; df.reset_index(drop=True) first")
    K, n = int(K), len(df)
    if K >= n:
        return df.index.copy()
    rng = np.random.default_rng(seed)
    if method == "random":
        pos = rng.permutation(n)[:K]
    elif method == "stratified":
        pos = _strata_positions(df, K, strata, rng)
    else:
        X = _feature_matrix(df, features)
        pos = _nearest_unique(X, _kmeans(X, K, rng, n_init, max_iter))
    return df.index[np.sort(pos)]


def pool_curve(df: pd.DataFrame, sizes, score_fn: Callable, *, method="random", draws: int = 20, seed: int = 0,
               features=None, strata=None, level: float = 0.9, long: bool = False) -> pd.DataFrame:
    """Pool-size curve: for each method and K in `sizes`, `draws` pools (seeds seed, seed+1, ...) from
    `pool_select`, each scored by `score_fn(df.loc[idx]) -> float` (e.g. 1 - TVD of the pool's answers against
    held-out people). `method`: one name or a list (one curve each). A K >= len(df) is the whole table (one draw).
    Returns one row per (method, size): draws, mean, lo, hi (central `level` interval of the draws, default
    5%-95%); `long=True` returns the per-draw scores (method, size, seed, score) instead."""
    methods = [method] if isinstance(method, str) else list(method)
    rows = []
    for m in methods:
        for K in sizes:
            for d in range(1 if K >= len(df) else draws):
                idx = pool_select(df, K, method=m, features=features, strata=strata, seed=seed + d)
                rows.append({"method": m, "size": int(K), "seed": seed + d, "score": float(score_fn(df.loc[idx]))})
    raw = pd.DataFrame(rows, columns=["method", "size", "seed", "score"])
    if long:
        return raw
    a = (1 - level) / 2
    g = raw.groupby(["method", "size"], sort=False)["score"]
    return pd.DataFrame({"draws": g.size(), "mean": g.mean(), "lo": g.quantile(a), "hi": g.quantile(1 - a)}
                        ).reset_index()


__all__ = ["Mapping", "from_row", "from_table", "check_ids", "auto_mapping", "narrate", "age_band", "gender_value",
           "to_frames", "TALP_FIELDS", "NARRATE_TASK",
           "apply_value_labels", "read_codebook", "warn_unlabelled", "pool_select", "pool_curve", "POOL_METHODS"]
