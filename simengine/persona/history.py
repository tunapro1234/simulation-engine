"""Personas with survey-answer and purchase/answer history text.

Ported from the TALP prep branches:
  talp73/kod/llm/istem.py   `profil`: survey answers rendered as "- question: answer" lines (the L1 persona)
  talp73/kod/llm/kos.py     `gecmis_metinleri`: per-category purchase summary text (the L2 add-on)
  talp73/kod/panel/panel.py purchase summary before the cutoff (items, quantity, days, last purchase, last-12-month items)
  persona/kod/kit           twin persona text = the person's known answers (question text + answer), targets excluded

Levels (talp73 protocol):
  L0  no persona: one neutral description for everyone (the "random customer" control)
  L1  survey profile: the person's own survey answers                -> authored.consumer_profile
  L2  L1 + purchase history strictly before `cutoff`                 -> authored.history_data

    personas = from_history(survey, id_col="uid", level="L2", purchases=orders, cutoff="2022-07-01",
                            category_col="category", date_col="order_date", questions={"age": "Age group", ...})
    print(personas[0].pid, personas[0].text())

IDS: the person id is `p.pid` (= `p.identity.persona_id`); it is kept out of `Persona.text()` and of the profile
body (the id column is never rendered as an answer), so label prompt blocks with `p.pid` yourself.
MULTI-SELECT: `comma_cols` (alias `comma_fields`, the `sim.prediction.PredictionSpec` name) renders "a,b" as
"a, b" in the table profile and in the known-answers block alike; `none_cols` (alias `none_fields`) likewise.

LEAKAGE RULE: purchases on/after `cutoff` are never used; `targets` (items/columns to be predicted) are dropped
from answer histories and raise if listed in `questions`.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .builders import Mapping, _is_empty, check_ids, from_row, warn_off_enum
from .schema import Persona

LEVELS = ("L0", "L1", "L2")
PROFILE_HEADER = "Survey answers of this person:"
ANSWERS_HEADER = "Known answers of this person:"
HISTORY_HEADER = "Purchase history in these categories, {range}:"
NEUTRAL_TEXT = "A randomly selected adult customer; no individual information is given."


# ---------------------------------------------------------------- text renderers
def survey_profile_text(row, questions: dict | None = None, *, none_cols: Iterable = (),
                        comma_cols: Iterable = (), missing: str = "(no answer)", skip: Iterable = ()) -> str:
    """Survey row -> "- question: answer" lines (talp73 `profil`).
    `questions`: column -> short question text (None = every column, the column name as question).
    `none_cols`: columns whose missing value means "None" (e.g. a multi-select "life changes" question).
    `comma_cols`: multi-select columns stored as "a,b,c" -> "a, b, c"."""
    none_cols, comma_cols, skip = set(none_cols), set(comma_cols), set(skip)
    get = row.get if hasattr(row, "get") else row.__getitem__
    if questions is None:
        keys = row.index if hasattr(row, "index") and not isinstance(row, dict) else row.keys()
        questions = {c: str(c) for c in keys if c not in skip}
    out = []
    for c, q in questions.items():
        v = get(c)
        if _is_empty(v):
            v = "None" if c in none_cols else missing
        else:
            v = str(v).strip()
            if c in comma_cols:
                v = v.replace(",", ", ")
        out.append(f"- {q}: {v}")
    return "\n".join(out)


def summarize_purchases(purchases: pd.DataFrame, *, id_col: str, category_col: str, date_col: str, cutoff,
                        qty_col: str | None = None, recent_start=None, categories: list | None = None,
                        top_k: int = 20, ids: Iterable | None = None) -> pd.DataFrame:
    """Purchase rows -> one row per (id, category), using only purchases strictly before `cutoff`
    (talp73 `panel.py` history summary). Columns: id_col, category, items (row count), quantity (sum of
    `qty_col`, missing = 1), days (distinct purchase days), last_purchase (date or None), items_recent
    (rows on/after `recent_start`, default cutoff - 1 year). Persons with no purchase in a category get 0.
    `categories` None: the `top_k` categories with the most distinct buyers before the cutoff (ties by name)."""
    cutoff = pd.Timestamp(cutoff)
    recent_start = pd.Timestamp(recent_start) if recent_start is not None else cutoff - pd.DateOffset(years=1)
    a = purchases[[c for c in (id_col, category_col, date_col, qty_col) if c]].copy()
    a[date_col] = pd.to_datetime(a[date_col], errors="coerce")
    a = a.dropna(subset=[date_col])
    past = a[(a[date_col] < cutoff) & a[category_col].notna()]
    if categories is None:
        buyers = past[[id_col, category_col]].drop_duplicates().groupby(category_col).size()
        buyers = buyers.rename("n").reset_index().sort_values(["n", category_col], ascending=[False, True])
        categories = list(buyers[category_col].head(top_k))
    categories = list(categories)
    past = past[past[category_col].isin(categories)]
    ids = list(dict.fromkeys(ids)) if ids is not None else sorted(a[id_col].dropna().unique().tolist())
    if len(past):
        g = past.groupby([id_col, category_col])
        qty = (past[qty_col].fillna(1) if qty_col else pd.Series(1.0, index=past.index)).astype(float)
        agg = pd.DataFrame({
            "items": g.size(),
            "quantity": qty.groupby([past[id_col], past[category_col]]).sum(),
            "days": g[date_col].apply(lambda s: s.dt.normalize().nunique()),
            "last_purchase": g[date_col].max(),
            "items_recent": (past[date_col] >= recent_start).groupby([past[id_col], past[category_col]]).sum(),
        })
    else:
        agg = pd.DataFrame(columns=["items", "quantity", "days", "last_purchase", "items_recent"])
    grid = pd.MultiIndex.from_product([ids, categories], names=[id_col, "category"])
    agg.index = agg.index.set_names([id_col, "category"])
    out = agg.reindex(grid)
    for c in ("items", "days", "items_recent"):
        out[c] = out[c].fillna(0).astype(int)
    out["quantity"] = out["quantity"].fillna(0.0).astype(float)
    out["last_purchase"] = out["last_purchase"].map(lambda t: None if _is_empty(t) else pd.Timestamp(t).date())
    out = out.reset_index()
    out.attrs.update({"cutoff": str(cutoff.date()), "recent_start": str(recent_start.date()),
                      "first_date": None if past.empty else str(past[date_col].min().date()),
                      "categories": categories})
    return out


def _month_range(start, end) -> str:
    return f"{pd.Timestamp(start):%B %Y} - {pd.Timestamp(end):%B %Y}"


def purchase_history_text(summary: pd.DataFrame, categories: list | None = None, *, recent_label: str = "the last 12 months",
                          labels: dict | None = None) -> str:
    """One person's summary rows (from `summarize_purchases`) -> one line per category (talp73 `gecmis_metinleri`):
    "CAT: never purchased" or "CAT: 3 items on 2 different days, last purchase 2022-05-03, 1 items in <recent>"."""
    s = summary.set_index("category")
    cats = list(categories) if categories is not None else list(s.index)
    labels = labels or {}
    lines = []
    for k in cats:
        name = labels.get(k, k)
        if k not in s.index or int(s.loc[k, "items"]) == 0:
            lines.append(f"{name}: never purchased")
        else:
            r = s.loc[k]
            lines.append(f"{name}: {int(r['items'])} items on {int(r['days'])} different days, last purchase "
                         f"{r['last_purchase']}, {int(r['items_recent'])} items in {recent_label}")
    return "\n".join(lines)


# ---------------------------------------------------------------- builder
def from_history(table: pd.DataFrame | None = None, *, id_col: str, level: str = "L2",
                 questions: dict | None = None, mapping: Mapping | None = None,
                 purchases: pd.DataFrame | None = None, purchase_summary: pd.DataFrame | None = None,
                 category_col: str = "category", date_col: str = "date", qty_col: str | None = None,
                 cutoff=None, recent_start=None, categories: list | None = None, top_k: int = 20,
                 category_labels: dict | None = None, history_label: str | None = None,
                 recent_label: str | None = None,
                 answers: pd.DataFrame | None = None, item_col: str = "item", value_col: str = "value",
                 item_text: dict | None = None, targets: Iterable = (),
                 none_cols: Iterable = (), comma_cols: Iterable = (), profile_header: str = PROFILE_HEADER,
                 answers_header: str = ANSWERS_HEADER, history_header: str = HISTORY_HEADER,
                 neutral_text: str = NEUTRAL_TEXT, locale: str = "en-US", source_name: str = "data",
                 ids: Iterable | None = None, comma_fields: Iterable = (),
                 none_fields: Iterable = ()) -> list[Persona]:
    """Personas whose prompt text is the person's own survey answers (L1) plus purchase history (L2).

    table      survey/demographic rows (one per person, `id_col`); rendered by `survey_profile_text` with
               `questions` (None = every column except id/targets).
    mapping    optional `Mapping`: identity/traits/facets from the same row (as `from_table`).
    purchases  raw purchase rows (`id_col`, `category_col`, `date_col`, optional `qty_col`) summarized before
               `cutoff`; or pass a ready `purchase_summary` (output of `summarize_purchases`).
    answers    long frame (`id_col`, `item_col`, `value_col`) of known past answers (twin-style persona text);
               `item_text` maps item -> question text; `targets` are dropped.
    level      'L0' neutral control, 'L1' profile (+ answers), 'L2' L1 + purchase history.
    comma_cols / none_cols (aliases comma_fields / none_fields): applied to the table profile and the answers.
    Returns personas in `ids` order (default: table ids, else purchase/answer ids); the id is `p.pid`
    (`identity.persona_id`) and does not appear in `p.text()`."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    comma_cols = set(comma_cols) | set(comma_fields)
    none_cols = set(none_cols) | set(none_fields)
    targets = set(targets)
    if questions is not None:
        bad = targets & set(questions)
        if bad:
            raise ValueError(f"leakage: target column(s) {sorted(bad)} are in questions")
    rows = {}
    if table is not None:
        check_ids(table, id_col)                 # empty/NaN or duplicated ids -> ValueError with counts
        rows = {str(r[id_col]): r for _, r in table.iterrows()}
    summary, cats = None, None
    if level == "L2":
        if purchase_summary is None and purchases is None:
            raise ValueError("level L2 needs purchases (with cutoff) or purchase_summary")
        if purchase_summary is None:
            if cutoff is None:
                raise ValueError("purchases need a cutoff (only purchases strictly before it are used)")
            purchase_summary = summarize_purchases(purchases, id_col=id_col, category_col=category_col,
                                                   date_col=date_col, cutoff=cutoff, qty_col=qty_col,
                                                   recent_start=recent_start, categories=categories, top_k=top_k,
                                                   ids=ids if ids is not None else (rows and list(table[id_col])) or None)
        summary = {str(k): g for k, g in purchase_summary.groupby(id_col, sort=False)}
        cats = categories or purchase_summary.attrs.get("categories") or list(dict.fromkeys(purchase_summary["category"]))
        at = purchase_summary.attrs
        c_raw = cutoff if cutoff is not None else at.get("cutoff")
        c = pd.Timestamp(c_raw) if c_raw is not None else None
        if recent_label is None:
            rs = recent_start if recent_start is not None else at.get("recent_start")
            recent_label = _month_range(rs, c - pd.DateOffset(days=1)) if (rs is not None and c is not None) \
                else "the last 12 months"
        if history_label is None:
            first = at.get("first_date")
            history_label = (_month_range(first, c - pd.DateOffset(days=1)) if first and c is not None
                             else (f"before {c.date()}" if c is not None else "before the cutoff"))
    ans = {}
    if answers is not None and level != "L0":
        a = answers[~answers[item_col].isin(targets)]
        for pid, g in a.groupby(id_col, sort=False):
            ans[str(pid)] = g
    if ids is None:
        ids = list(rows) or (list(summary) if summary else list(ans))
    ids = [str(i) for i in ids]

    if questions is not None and id_col in questions:      # the id is p.pid, never an answer line
        questions = {c: q for c, q in questions.items() if c != id_col}
    out, off_enum = [], {}
    for pid in ids:
        row = rows.get(pid)
        if level == "L0":
            p = Persona.from_dict({"identity": {"persona_id": pid, "name": pid, "locale": locale},
                                   "authored": {"persona": neutral_text}})
        elif mapping is not None and row is not None:
            p = from_row(row, mapping, pid=pid, off_enum=off_enum)
        else:
            p = Persona.from_dict({"identity": {"persona_id": pid, "name": pid, "locale": locale},
                                   "evidence": {"measured_receipt": {"draws": [{"contract_id": source_name,
                                                                                "contract_version": "1"}]}}})
        if level in ("L1", "L2"):
            parts = []
            if row is not None:
                skip = {id_col} | targets
                parts.append(profile_header + "\n" + survey_profile_text(
                    row, questions, none_cols=none_cols, comma_cols=comma_cols, skip=skip))
            if pid in ans:
                g = ans[pid]
                q = {it: (item_text or {}).get(it, str(it)) for it in g[item_col]}
                vals = dict(zip(g[item_col], g[value_col]))
                parts.append(answers_header + "\n" + survey_profile_text(vals, q, none_cols=none_cols,
                                                                          comma_cols=comma_cols))
            if parts:
                p.authored.consumer_profile = "\n\n".join(parts)
        if level == "L2":
            s = summary.get(pid)
            body = (purchase_history_text(s, cats, recent_label=recent_label, labels=category_labels)
                    if s is not None else "\n".join(f"{(category_labels or {}).get(k, k)}: never purchased"
                                                     for k in cats))
            p.authored.history_data = history_header.format(range=history_label) + "\n" + body
        p.extra["level"] = level
        if level == "L2":
            p.extra["history_cutoff"] = str(pd.Timestamp(cutoff).date()) if cutoff is not None else \
                purchase_summary.attrs.get("cutoff")
        out.append(p)
    warn_off_enum(off_enum, mapping, len(out))
    return out


__all__ = ["from_history", "survey_profile_text", "summarize_purchases", "purchase_history_text", "LEVELS",
           "PROFILE_HEADER", "ANSWERS_HEADER", "HISTORY_HEADER", "NEUTRAL_TEXT"]
