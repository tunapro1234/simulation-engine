"""Evidence extras: spread, completion, published-finding deltas, stylized-fact scores, coefficient recovery, say-do.

Usage::

    from simengine.evidence import extras
    sec = extras.say_do(frame, "said", "did", preds=["llm", "persistence"], cluster="person")
    print(extras.section_markdown(sec))

These are card *sections* that sit next to the main evidence card (`card`, `card_markdown`). They answer the
questions a jury asks about synthetic respondents that the baseline ladder does not:

- ``variance_ratio``: is the simulated spread as wide as the human spread? (homogenisation; Bisbee et al.,
  "Synthetic Replacements for Human Survey Data", Political Analysis)
- ``completion_row``: how many planned agents produced a valid answer, per condition? Silent dropout and parse
  failures are counted and reported, never imputed.
- ``published_findings``: simulated value vs published value for each finding (a delta table; idea from the
  replicant ``PaperComparison``).
- ``stylized_facts_score``: a weighted 0-1 index over named criteria where N/A criteria drop out of both the
  numerator and the denominator (generalised from the edsl-test who-to-punish scorer).
- ``coefficient_recovery``: fit the same OLS on simulated and human rows and report sign agreement and the
  share of coefficients that differ significantly (Bisbee et al.).
- ``say_do``: with two anchors per person (what they said, what they did), is each prediction's panel mean
  (and person-level error) closer to the stated or to the observed value? Person-cluster bootstrap CI and a
  verdict line for the card.

Every function returns a plain, JSON-friendly dict with ``kind``, ``title``, ``columns``, ``rows`` and ``notes``
(plus its own numbers). ``section_markdown`` / ``section_html`` render one section or a list of sections, so they
can be appended to a Markdown card or embedded in an HTML page.
"""
from __future__ import annotations

import html as _html
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..types import COLUMN_ALIASES, PRED_ALIASES, PRED_PREFIX, normalize_frame

__all__ = [
    "variance_ratio", "completion_row", "completion_from_frame", "published_findings", "stylized_facts_score",
    "coefficient_recovery", "say_do", "section_markdown", "section_html",
]

HOMOGENISATION_BELOW = 0.8      # same thresholds as the spread block of `evaluate` (analiz.yayilim)
OVERDISPERSION_ABOVE = 1.5


# ------------------------------------------------------------------ helpers
def _num(x) -> float | None:
    """float or None (None/NaN/inf/unparseable all become None: missing stays missing)."""
    if x is None or isinstance(x, bool):
        return None if x is None else float(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def _numeric(series: pd.Series, name: str) -> np.ndarray:
    """Numeric array with NaN for missing; raises if a non-missing value is not a number (no silent coercion)."""
    conv = pd.to_numeric(series, errors="coerce")
    bad = conv.isna() & series.notna()
    if bad.any():
        ex = series[bad].iloc[0]
        raise ValueError(f"column {name!r} has a non-numeric value ({ex!r}); encode it before calling")
    return conv.to_numpy(dtype=float)


def _percentile_ci(draws: np.ndarray, alpha: float) -> tuple[float | None, float | None]:
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return None, None
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def _fmt(x, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if math.isnan(x):
            return "—"
        return f"{x:.{digits}f}"
    return str(x)


def _section(kind: str, title: str, columns: list[str], rows: list[list], notes: list[str], **extra) -> dict:
    return {"kind": kind, "title": title, "columns": columns, "rows": rows, "notes": notes, **extra}


def _wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


# ------------------------------------------------------------------ variance ratio
def _resolve_pred(df: pd.DataFrame, method: str) -> str:
    for c in (method if str(method).startswith(PRED_PREFIX) else PRED_PREFIX + str(method), method):
        if c in df.columns:
            return c
    raise KeyError(f"no column for method {method!r} (looked for {PRED_PREFIX}{method} and {method})")


def variance_ratio(df: pd.DataFrame, method: str, *, item: str = "item", truth: str = "truth",
                   source: str | None = None, value: str = "value", human: str = "human",
                   B: int = 1000, seed: int = 0, alpha: float = 0.05, min_n: int = 2) -> dict:
    """Simulated SD / human SD per item, with a percentile bootstrap CI, plus a pooled ratio.

    Two input shapes:

    - **EvalFrame (paired, default):** one row per unit with ``truth`` and ``pred_<method>`` (or a column named
      ``method``). Turkish aliases are accepted. Rows missing either value are dropped *per item* and counted.
      The bootstrap resamples rows within each item jointly (the pairing is kept).
    - **Long (unpaired):** pass ``source=<column>``; rows with ``df[source] == method`` are simulated answers and
      rows with ``df[source] == human`` are human answers, both in ``value``. Each side is resampled independently
      within the item.

    Without an ``item`` column all rows form one item. SDs use ddof=1. The pooled ratio is
    sqrt(sum of simulated variances / sum of human variances) over the usable items, the same pooling as the
    spread block of ``evaluate``. Items with fewer than ``min_n`` values on a side, or with zero human SD, get
    ``ratio=None`` and a reason; they are left out of the pooled ratio. Ratios below 0.8 are flagged as
    homogenisation and above 1.5 as over-dispersion (point estimate).

    The CI covers resampling respondents within the given items, not new items.
    """
    if B < 1:
        raise ValueError("B must be >= 1")
    if source is None:
        d = normalize_frame(df)
        col = _resolve_pred(d, method)
        if truth not in d.columns:
            raise KeyError(f"no truth column {truth!r}")
        s_all = _numeric(d[col], col)
        h_all = _numeric(d[truth], truth)
        items = d[item].to_numpy() if item in d.columns else np.zeros(len(d), dtype=int)
        paired = True
    else:
        if source not in df.columns or value not in df.columns:
            raise KeyError(f"long input needs columns {source!r} and {value!r}")
        d = df
        sub = d[d[source].isin([method, human])]
        if not (sub[source] == method).any():
            raise KeyError(f"no rows with {source} == {method!r}")
        if not (sub[source] == human).any():
            raise KeyError(f"no rows with {source} == {human!r}")
        vals = _numeric(sub[value], value)
        items = sub[item].to_numpy() if item in sub.columns else np.zeros(len(sub), dtype=int)
        is_sim = (sub[source] == method).to_numpy()
        paired = False

    rng = np.random.default_rng(seed)
    out_items: list[dict] = []
    boot_vs, boot_vh = [], []          # per usable item: B variances
    vs_pt, vh_pt = [], []
    n_missing_sim = n_missing_human = 0
    for key in pd.unique(items):
        m = items == key
        if paired:
            s, h = s_all[m], h_all[m]
            ms, mh = np.isnan(s), np.isnan(h)
            n_missing_sim += int(ms.sum())
            n_missing_human += int(mh.sum())
            ok = ~(ms | mh)
            s, h = s[ok], h[ok]
        else:
            v = vals[m]
            sm = is_sim[m]
            s, h = v[sm], v[~sm]
            n_missing_sim += int(np.isnan(s).sum())
            n_missing_human += int(np.isnan(h).sum())
            s, h = s[~np.isnan(s)], h[~np.isnan(h)]
        row = {"item": key.item() if hasattr(key, "item") else key, "n_sim": int(s.size), "n_human": int(h.size),
               "sd_sim": None, "sd_human": None, "ratio": None, "lo": None, "hi": None, "reason": None}
        if s.size < min_n or h.size < min_n:
            row["reason"] = f"fewer than {min_n} values"
            out_items.append(row)
            continue
        sd_s, sd_h = float(np.std(s, ddof=1)), float(np.std(h, ddof=1))
        row["sd_sim"], row["sd_human"] = sd_s, sd_h
        if sd_h == 0:
            row["reason"] = "human SD is 0"
            out_items.append(row)
            continue
        row["ratio"] = sd_s / sd_h
        if paired:
            idx = rng.integers(0, s.size, size=(B, s.size))
            bs, bh = s[idx], h[idx]
        else:
            bs = s[rng.integers(0, s.size, size=(B, s.size))]
            bh = h[rng.integers(0, h.size, size=(B, h.size))]
        var_s, var_h = bs.var(axis=1, ddof=1), bh.var(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.sqrt(var_s / var_h)
        r[var_h == 0] = np.nan
        row["lo"], row["hi"] = _percentile_ci(r, alpha)
        boot_vs.append(var_s)
        boot_vh.append(var_h)
        vs_pt.append(sd_s ** 2)
        vh_pt.append(sd_h ** 2)
        out_items.append(row)

    pooled = {"ratio": None, "lo": None, "hi": None, "n_items": len(vs_pt)}
    if vs_pt:
        pooled["ratio"] = float(math.sqrt(sum(vs_pt) / sum(vh_pt)))
        tot_s, tot_h = np.sum(boot_vs, axis=0), np.sum(boot_vh, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            pr = np.sqrt(tot_s / tot_h)
        pr[tot_h == 0] = np.nan
        pooled["lo"], pooled["hi"] = _percentile_ci(pr, alpha)
    ratios = [r["ratio"] for r in out_items if r["ratio"] is not None]
    median = float(np.median(ratios)) if ratios else None
    flag = None
    if pooled["ratio"] is not None:
        if pooled["ratio"] < HOMOGENISATION_BELOW:
            flag = "homogenisation"
        elif pooled["ratio"] > OVERDISPERSION_ABOVE:
            flag = "over-dispersion"

    notes = [f"Pooled ratio = sqrt(sum simulated variance / sum human variance) over {len(vs_pt)} item(s); "
             f"{int((1 - alpha) * 100)} % percentile bootstrap, B = {B}, seed = {seed}, "
             f"{'paired rows' if paired else 'independent samples'} within item.",
             f"< {HOMOGENISATION_BELOW} = homogenisation, > {OVERDISPERSION_ABOVE} = over-dispersion."]
    if n_missing_sim or n_missing_human:
        notes.append(f"Missing values dropped (not imputed): simulated {n_missing_sim}, human {n_missing_human}.")
    skipped = [r for r in out_items if r["ratio"] is None]
    if skipped:
        notes.append(f"{len(skipped)} item(s) without a ratio: " +
                     "; ".join(f"{r['item']} ({r['reason']})" for r in skipped[:5]) + (" ..." if len(skipped) > 5 else ""))
    rows = [[r["item"], r["n_sim"], r["n_human"], r["sd_sim"], r["sd_human"], r["ratio"], r["lo"], r["hi"]]
            for r in out_items]
    rows.append(["pooled", sum(r["n_sim"] for r in out_items), sum(r["n_human"] for r in out_items), None, None,
                 pooled["ratio"], pooled["lo"], pooled["hi"]])
    return _section("variance_ratio", f"Spread: simulated SD / human SD ({method})",
                    ["item", "n sim", "n human", "SD sim", "SD human", "ratio", "lo", "hi"], rows, notes,
                    method=str(method), paired=paired, items=out_items, pooled=pooled, median_item_ratio=median,
                    flag=flag, n_missing_sim=n_missing_sim, n_missing_human=n_missing_human, B=B, seed=seed,
                    alpha=alpha)


# ------------------------------------------------------------------ completion
def _cond_counts(v) -> tuple[int, int, int | None]:
    if isinstance(v, Mapping):
        planned, completed = v.get("planned", v.get("n_planned")), v.get("completed", v.get("n_completed"))
        pf = v.get("parse_failed", v.get("n_parse_failed"))
    else:
        seq = list(v)
        if len(seq) not in (2, 3):
            raise ValueError("a condition is (planned, completed[, parse_failed]) or a dict")
        planned, completed = seq[0], seq[1]
        pf = seq[2] if len(seq) == 3 else None
    if planned is None or completed is None:
        raise ValueError("each condition needs planned and completed counts")
    planned, completed = int(planned), int(completed)
    pf = None if pf is None else int(pf)
    if planned < 0 or completed < 0 or completed > planned:
        raise ValueError(f"need 0 <= completed <= planned, got {completed}/{planned}")
    if pf is not None and (pf < 0 or completed + pf > planned):
        raise ValueError(f"completed + parse_failed exceeds planned ({completed}+{pf} > {planned})")
    return planned, completed, pf


def completion_row(n_planned: int | None = None, n_completed: int | None = None,
                   by_condition: Mapping[str, Any] | None = None, *, n_parse_failed: int | None = None,
                   label: str = "Completion", alpha: float = 0.05) -> dict:
    """Evidence-card row for completion and parse failure.

    ``n_completed`` counts valid answers. ``n_parse_failed`` counts answers that came back but did not parse
    (``valid=False``); planned − completed − parse_failed is silent dropout (no answer at all). Nothing is imputed.

    ``by_condition`` maps a condition name to ``{"planned", "completed", "parse_failed"?}`` or to a
    ``(planned, completed[, parse_failed])`` tuple. When the totals are omitted they are summed from it. With two
    or more conditions a chi-square test of equal completion is reported (Fisher's exact test for 2×2 tables),
    because dropout that differs by condition biases the comparison between conditions.
    """
    conds: list[tuple[str, int, int, int | None]] = []
    if by_condition:
        for name, v in by_condition.items():
            p, c, pf = _cond_counts(v)
            conds.append((str(name), p, c, pf))
        sp, sc = sum(c[1] for c in conds), sum(c[2] for c in conds)
        spf = None if any(c[3] is None for c in conds) else sum(c[3] for c in conds)
        n_planned = sp if n_planned is None else n_planned
        n_completed = sc if n_completed is None else n_completed
        if n_parse_failed is None:
            n_parse_failed = spf
    if n_planned is None or n_completed is None:
        raise ValueError("give n_planned and n_completed, or by_condition")
    total = {"n_planned": n_planned, "n_completed": n_completed, "parse_failed": n_parse_failed}
    n_planned, n_completed, n_parse_failed = _cond_counts(
        {"planned": n_planned, "completed": n_completed, "parse_failed": n_parse_failed})
    notes: list[str] = []
    if conds:
        sp, sc = sum(c[1] for c in conds), sum(c[2] for c in conds)
        if (sp, sc) != (n_planned, n_completed):
            notes.append(f"Totals ({n_completed}/{n_planned}) differ from the sum over conditions ({sc}/{sp}).")

    def one(name, p, c, pf):
        lo, hi = _wilson(c, p, alpha)
        drop = None if pf is None else p - c - pf
        return {"condition": name, "n_planned": p, "n_completed": c, "n_parse_failed": pf, "n_dropout": drop,
                "rate": (c / p) if p else None, "lo": lo, "hi": hi,
                "parse_failure_rate": (pf / p) if (p and pf is not None) else None}

    overall = one("all", n_planned, n_completed, n_parse_failed)
    per = [one(*c) for c in conds]
    test = None
    if len(per) >= 2:
        table = np.array([[c["n_completed"], c["n_planned"] - c["n_completed"]] for c in per])
        if (table.sum(axis=0) == 0).any():
            test = {"test": "none", "p_value": None, "reason": "completion is identical (all or none) everywhere"}
        elif table.shape == (2, 2):
            _, p = stats.fisher_exact(table)
            test = {"test": "fisher_exact", "p_value": float(p)}
        else:
            chi2, p, dof, _ = stats.chi2_contingency(table)
            test = {"test": "chi2", "p_value": float(p), "chi2": float(chi2), "dof": int(dof)}
        if test.get("p_value") is not None and test["p_value"] < alpha:
            notes.append(f"Completion differs across conditions ({test['test']} p = {test['p_value']:.3g}): "
                         "dropout is not ignorable; compare completers' covariates with non-completers'.")
    if overall["rate"] is not None and overall["rate"] < 1:
        notes.append("Missing answers are reported as missing, never imputed.")
    notes.append(f"Rate CI: Wilson {int((1 - alpha) * 100)} %.")
    cols = ["condition", "planned", "completed", "parse failed", "dropout", "rate", "lo", "hi"]
    rows = [[r["condition"], r["n_planned"], r["n_completed"], r["n_parse_failed"], r["n_dropout"], r["rate"],
             r["lo"], r["hi"]] for r in per + [overall]]
    worst = min(per, key=lambda r: (r["rate"] if r["rate"] is not None else 2.0)) if per else None
    return _section("completion", label, cols, rows, notes, overall=overall, by_condition=per, test=test,
                    worst_condition=worst["condition"] if worst else None, input=total, alpha=alpha)


def completion_from_frame(df: pd.DataFrame, n_planned: int | Mapping[str, int], *, condition: str | None = None,
                          valid: str = "valid", label: str = "Completion", alpha: float = 0.05) -> dict:
    """``completion_row`` from answer rows: each row is one returned answer with a boolean ``valid`` column.

    ``n_planned`` is an int (no conditions) or ``{condition: planned}``. Rows with ``valid`` False count as parse
    failures; planned rows that never came back are dropout. Pass ``condition`` to split by a column.
    """
    if valid not in df.columns:
        raise KeyError(f"no {valid!r} column")
    ok = df[valid].fillna(False).astype(bool)
    if condition is None:
        if isinstance(n_planned, Mapping):
            raise ValueError("a per-condition n_planned needs condition=")
        return completion_row(int(n_planned), int(ok.sum()), n_parse_failed=int((~ok).sum()), label=label,
                              alpha=alpha)
    if not isinstance(n_planned, Mapping):
        raise ValueError("with condition=, n_planned is {condition: planned}")
    by = {}
    for name, p in n_planned.items():
        m = df[condition] == name
        by[name] = {"planned": int(p), "completed": int(ok[m].sum()), "parse_failed": int((~ok[m]).sum())}
    unknown = set(pd.unique(df[condition])) - set(n_planned)
    sec = completion_row(by_condition=by, label=label, alpha=alpha)
    if unknown:
        sec["notes"].append(f"Rows with unplanned conditions ignored: {sorted(map(str, unknown))}.")
    return sec


# ------------------------------------------------------------------ published findings
def published_findings(rows: Iterable[Mapping[str, Any]], *, title: str = "Published findings vs simulation",
                       digits: int = 3) -> dict:
    """Delta table: simulated − published for each finding.

    Each row: ``finding``, ``published``, ``simulated`` (None when not measured), ``source`` (citation), and
    optionally ``unit``, ``tolerance`` (|delta| <= tolerance counts as a match) and ``published_ci`` ((lo, hi):
    reports whether the simulated value lies inside it). A missing simulated value stays missing: it is counted
    and shown as "—", and it is left out of the summary statistics.
    """
    out_rows, recs = [], []
    for r in rows:
        if "finding" not in r or "published" not in r:
            raise ValueError("each row needs 'finding' and 'published'")
        pub, sim = _num(r.get("published")), _num(r.get("simulated"))
        if pub is None:
            raise ValueError(f"finding {r['finding']!r}: published value is missing or not a number")
        delta = None if sim is None else sim - pub
        rel = None if (delta is None or pub == 0) else delta / abs(pub)
        tol = _num(r.get("tolerance"))
        match = None if (delta is None or tol is None) else abs(delta) <= tol
        ci = r.get("published_ci")
        inside = None
        if ci is not None and sim is not None:
            lo, hi = _num(ci[0]), _num(ci[1])
            if lo is not None and hi is not None:
                inside = lo <= sim <= hi
        rec = {"finding": str(r["finding"]), "published": pub, "simulated": sim, "delta": delta, "rel_delta": rel,
               "unit": r.get("unit", ""), "tolerance": tol, "match": match, "inside_published_ci": inside,
               "source": r.get("source")}
        recs.append(rec)
        out_rows.append([rec["finding"], pub, sim, delta, rel, match if tol is not None else inside,
                         rec["source"] or ""])
    measured = [x for x in recs if x["delta"] is not None]
    judged = [x for x in measured if x["match"] is not None or x["inside_published_ci"] is not None]
    hits = sum(1 for x in judged if (x["match"] if x["match"] is not None else x["inside_published_ci"]))
    summary = {"n_findings": len(recs), "n_measured": len(measured), "n_missing": len(recs) - len(measured),
               "mean_abs_delta": float(np.mean([abs(x["delta"]) for x in measured])) if measured else None,
               "n_judged": len(judged), "n_match": hits}
    notes = []
    if summary["n_missing"]:
        notes.append(f"{summary['n_missing']} finding(s) not measured in the simulation (shown as —, not imputed).")
    if judged:
        notes.append(f"Matched {hits}/{len(judged)} finding(s) with a tolerance or published CI.")
    notes.append("delta = simulated − published; rel = delta / |published|. Mixed units: compare rows, not the mean.")
    return _section("published_findings", title,
                    ["finding", "published", "simulated", "delta", "rel", "match", "source"], out_rows, notes,
                    findings=recs, summary=summary, digits=digits)


# ------------------------------------------------------------------ stylized facts
def stylized_facts_score(criteria: Iterable[Mapping[str, Any]], *, title: str = "Stylized facts index") -> dict:
    """Weighted mean of 0-1 criterion scores; N/A criteria are excluded from numerator and denominator.

    Each criterion: ``name``, ``value`` (a score in [0, 1], or None/NaN when the criterion is not observable in
    this run), ``weight`` (> 0, default 1) and optional ``note`` / ``target`` (text, e.g. the human benchmark it
    was scored against). Values outside [0, 1] raise: clamp or rescale them in the criterion itself, where the
    human target is known. The result reports how much of the total weight was actually observed
    (``weight_coverage``), because an index over fewer criteria is not comparable to one over all of them.
    """
    recs, num, den, tot = [], 0.0, 0.0, 0.0
    for c in criteria:
        if "name" not in c:
            raise ValueError("each criterion needs a 'name'")
        w = _num(c.get("weight", 1.0))
        if w is None or w <= 0:
            raise ValueError(f"criterion {c['name']!r}: weight must be > 0")
        v = _num(c.get("value"))
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"criterion {c['name']!r}: value {v} is outside [0, 1]")
        tot += w
        if v is not None:
            num += w * v
            den += w
        recs.append({"name": str(c["name"]), "value": v, "weight": w, "applicable": v is not None,
                     "target": c.get("target"), "note": c.get("note")})
    if not recs:
        raise ValueError("no criteria")
    index = num / den if den > 0 else None
    excluded = [r["name"] for r in recs if not r["applicable"]]
    notes = [f"Index = sum(weight × value) / sum(weight) over {len(recs) - len(excluded)} applicable criteria."]
    if excluded:
        notes.append(f"N/A, excluded: {', '.join(excluded)} (weight coverage {den / tot:.0%}).")
    if index is None:
        notes.append("No criterion was observable: no index.")
    rows = [[r["name"], r["weight"], r["value"] if r["applicable"] else "N/A", r["target"] or "", r["note"] or ""]
            for r in recs]
    rows.append(["index", den, index, "", ""])
    return _section("stylized_facts", title, ["criterion", "weight", "score", "target", "note"], rows, notes,
                    index=index, n_criteria=len(recs), n_applicable=len(recs) - len(excluded), excluded=excluded,
                    weight_coverage=den / tot, criteria=recs)


# ------------------------------------------------------------------ coefficient recovery
def _split_formula(formula_cols) -> tuple[str, list[str]]:
    if isinstance(formula_cols, Mapping):
        y, x = formula_cols.get("y"), list(formula_cols.get("x") or [])
    else:
        seq = list(formula_cols)
        y, x = (seq[0], seq[1:]) if seq else (None, [])
    if not y or not x:
        raise ValueError("formula_cols is {'y': outcome, 'x': [predictors]} or (outcome, predictor, ...)")
    return str(y), [str(c) for c in x]


def _ols(df: pd.DataFrame, y: str, xs: list[str], label: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    for c in [y] + xs:
        if c not in df.columns:
            raise KeyError(f"{label}: no column {c!r}")
    Y = _numeric(df[y], y)
    X = np.column_stack([_numeric(df[c], c) for c in xs])
    ok = ~(np.isnan(Y) | np.isnan(X).any(axis=1))
    dropped = int((~ok).sum())
    Y, X = Y[ok], np.column_stack([np.ones(ok.sum()), X[ok]])
    n, k = X.shape
    if n <= k:
        raise ValueError(f"{label}: {n} complete rows for {k} coefficients")
    if np.linalg.matrix_rank(X) < k:
        raise ValueError(f"{label}: predictors are collinear (or constant)")
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    sigma2 = float(resid @ resid) / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    return beta, se, n, dropped


def coefficient_recovery(sim_df: pd.DataFrame, human_df: pd.DataFrame, formula_cols, *, alpha: float = 0.05,
                         include_intercept: bool = False) -> dict:
    """Fit the same OLS (with intercept) on simulated and on human rows and compare the coefficients.

    ``formula_cols``: ``{"y": outcome, "x": [predictors]}`` or ``(outcome, predictor, ...)``; numeric columns only
    (dummy-encode categories first). Rows with a missing value are dropped listwise and counted. Per coefficient:
    both estimates, a two-sided z test of equality using sqrt(se_sim² + se_human²) (the samples are independent),
    and whether the signs agree. Summary: ``sign_agreement`` (share of coefficients with the same sign),
    ``share_differ`` (share with p < alpha), and ``sign_flip_when_human_significant`` (sign flips among the
    coefficients that are significant in the human data). The intercept is reported but left out of the
    summary unless ``include_intercept``.
    """
    y, xs = _split_formula(formula_cols)
    bs, ses, ns, ds = _ols(sim_df, y, xs, "sim_df")
    bh, seh, nh, dh = _ols(human_df, y, xs, "human_df")
    z = float(stats.norm.ppf(1 - alpha / 2))
    names = ["(intercept)"] + xs
    coefs = []
    for i, name in enumerate(names):
        se = math.sqrt(ses[i] ** 2 + seh[i] ** 2)
        zz = (bs[i] - bh[i]) / se if se > 0 else 0.0
        p = float(2 * stats.norm.sf(abs(zz)))
        h_sig = bool(seh[i] > 0 and abs(bh[i] / seh[i]) > z)
        coefs.append({"name": name, "sim": float(bs[i]), "sim_se": float(ses[i]), "human": float(bh[i]),
                      "human_se": float(seh[i]), "diff": float(bs[i] - bh[i]), "p_diff": p, "differs": p < alpha,
                      "same_sign": bool(np.sign(bs[i]) == np.sign(bh[i])), "human_significant": h_sig})
    used = coefs if include_intercept else coefs[1:]
    sig = [c for c in used if c["human_significant"]]
    summary = {"n_coefficients": len(used),
               "sign_agreement": float(np.mean([c["same_sign"] for c in used])),
               "share_differ": float(np.mean([c["differs"] for c in used])),
               "sign_flip_when_human_significant": (float(np.mean([not c["same_sign"] for c in sig]))
                                                    if sig else None),
               "n_human_significant": len(sig), "n_sim": ns, "n_human": nh, "dropped_sim": ds, "dropped_human": dh}
    notes = [f"OLS {y} ~ {' + '.join(xs)}; difference test z = (b_sim − b_human) / sqrt(se_sim² + se_human²), "
             f"alpha = {alpha}.",
             f"Signs agree on {summary['sign_agreement']:.0%} and {summary['share_differ']:.0%} of coefficients "
             f"differ significantly ({len(used)} coefficient(s){'' if include_intercept else ', intercept excluded'})."]
    if ds or dh:
        notes.append(f"Rows with missing values dropped (not imputed): simulated {ds}, human {dh}.")
    notes.append("Classical (homoskedastic) standard errors; a simulated sample with collapsed variance "
                 "can make tiny SEs, so read 'differs' together with the size of 'diff'.")
    rows = [[c["name"], c["sim"], c["human"], c["diff"], c["p_diff"], c["differs"], c["same_sign"]] for c in coefs]
    return _section("coefficient_recovery", f"Coefficient recovery: {y}",
                    ["coefficient", "sim", "human", "diff", "p", "differs", "same sign"], rows, notes,
                    coefficients=coefs, summary=summary, formula={"y": y, "x": xs}, alpha=alpha)


# ------------------------------------------------------------------ say-do anchors
SAY, DO, UNDECIDED = "closer to what people say", "closer to what people do", "undecided"


def _col(d: pd.DataFrame, name: str, what: str) -> str:
    """Column name after normalize_frame (Turkish aliases are renamed there)."""
    for c in (name, COLUMN_ALIASES.get(name, name)):
        if c in d.columns:
            return c
    raise KeyError(f"no {what} column {name!r}")


def _resolve_pred_alias(d: pd.DataFrame, method: str) -> str:
    m = str(method)
    for p in PRED_ALIASES:                       # "tahmin_x" was renamed to "pred_x" by normalize_frame
        if m.startswith(p):
            m = PRED_PREFIX + m[len(p):]
    return _resolve_pred(d, m)


def say_do(frame: pd.DataFrame, say_col: str, do_col: str, preds: Sequence[str] | None = None, *,
           cluster: str | None = "person", on: str = "level", B: int = 1000, seed: int = 0,
           alpha: float = 0.05) -> dict:
    """Is a prediction closer to what people *say* (stated) or to what they *do* (observed)?

    ``frame`` has one row per person (or per person × item) with a numeric ``say_col`` anchor (stated answer,
    intention), a numeric ``do_col`` anchor (observed behaviour) and one or more prediction columns. ``preds`` are
    method names or columns (``"llm"`` → ``pred_llm``); None takes every ``pred_*`` column. For each prediction:

    - **level**: distance of the panel mean to each anchor mean, ``|mean(pred) − mean(say)|`` and
      ``|mean(pred) − mean(do)|``, and their difference ``say − do`` (negative = closer to say);
    - **MAE**: person-level mean absolute error to each anchor and the same ``say − do`` difference.

    Both differences get a percentile CI from a cluster bootstrap over ``cluster`` (persons; rows of a person
    move together; the default ``"person"`` falls back to ``"group"``, the EvalFrame cluster key that
    ``survey_frame`` fills with the persona id; a named column that is missing raises). The same resamples are used
    for every prediction and for the anchor gap, so they are paired.
    The verdict reads the CI of the ``on`` difference (``"level"``, default, or ``"mae"``): wholly below 0 →
    closer to what people say, wholly above 0 → closer to what people do, otherwise undecided. Rows missing the
    prediction or either anchor are dropped for that prediction and counted. With ``cluster=None`` (or no
    person/group column, noted) every row is its own cluster. If the anchors themselves are not apart (the gap
    CI covers 0) the verdict is forced to undecided: there is nothing to be closer to. The level difference is
    ``−gap`` once the panel mean reaches the say anchor (``+gap`` at do), so when a prediction sits on an anchor
    the CI bound away from 0 is too tight (simulated coverage ~0.84 there, ~0.94 elsewhere); the bound next to 0,
    and so the verdict, holds.
    """
    if B < 1:
        raise ValueError("B must be >= 1")
    if on not in ("level", "mae"):
        raise ValueError("on must be 'level' or 'mae'")
    d = normalize_frame(frame)
    sc, dc = _col(d, say_col, "say"), _col(d, do_col, "do")
    if preds is None:
        cols = [c for c in d.columns if isinstance(c, str) and c.startswith(PRED_PREFIX) and c not in (sc, dc)]
        if not cols:
            raise KeyError(f"no {PRED_PREFIX}* columns; pass preds=[...]")
    else:
        cols = [_resolve_pred_alias(d, p) for p in ([preds] if isinstance(preds, str) else preds)]
    notes = []
    cl = None
    if cluster is not None:
        # "grup" was renamed to "group" by normalize_frame; the default "person" falls back to "group", the
        # EvalFrame cluster key (survey_frame puts the persona id there and has no "person" column)
        for c in (cluster, COLUMN_ALIASES.get(cluster, cluster)) + (("group",) if cluster == "person" else ()):
            if c in d.columns:
                cl = c
                break
        if cl is None and cluster != "person":
            raise KeyError(f"no cluster column {cluster!r}; pass cluster=None for a row bootstrap")
    if cl is not None:
        if d[cl].isna().any():
            raise ValueError(f"cluster column {cl!r} has missing values; fill or drop them first")
        codes, uniq = pd.factorize(d[cl], sort=False)
        unit = f"{cl} cluster"
    else:
        codes, uniq = np.arange(len(d)), np.arange(len(d))
        unit = "row"
        if cluster is not None:
            notes.append("No 'person' or 'group' column: every row is its own cluster (rows of one person are "
                         "not tied; the CI is too narrow if a person has several rows).")
    G = len(uniq)
    say, do = _numeric(d[sc], sc), _numeric(d[dc], dc)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(float) if G else np.zeros((B, 0))

    def csum(x: np.ndarray, m: np.ndarray) -> np.ndarray:
        return np.bincount(codes[m], weights=x[m], minlength=G)

    def wmean(x: np.ndarray, m: np.ndarray) -> tuple[float, np.ndarray]:
        s, n = csum(x, m), np.bincount(codes[m], minlength=G).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(s.sum() / n.sum()) if n.sum() else math.nan, (W @ s) / (W @ n)

    pct = int(round((1 - alpha) * 100))
    # anchor gap on rows with both anchors
    ma = ~(np.isnan(say) | np.isnan(do))
    gap_pt, gap_b = wmean(do - say, ma)
    g_lo, g_hi = _percentile_ci(gap_b, alpha) if G >= 2 else (None, None)
    anchors = {"mean_say": _num(np.nanmean(say[ma])) if ma.any() else None,
               "mean_do": _num(np.nanmean(do[ma])) if ma.any() else None,
               "gap": _num(gap_pt), "lo": g_lo, "hi": g_hi, "n": int(ma.sum()),
               "apart": g_lo is not None and (g_lo > 0 or g_hi < 0)}

    out: list[dict] = []
    for c in cols:
        p = _numeric(d[c], c)
        m = ma & ~np.isnan(p)
        r = {"prediction": c, "n": int(m.sum()), "n_missing": int(len(d) - m.sum()),
             "clusters": int(np.unique(codes[m]).size), "mean": None, "mean_say": None, "mean_do": None,
             "dist_say": None, "dist_do": None, "diff": None, "lo": None, "hi": None,
             "mae_say": None, "mae_do": None, "mae_diff": None, "mae_lo": None, "mae_hi": None,
             "verdict": UNDECIDED, "reason": None, "card_line": None}
        if not m.any():
            r["reason"] = "no rows with prediction and both anchors"
            out.append(r)
            continue
        mp, bp = wmean(p, m)
        ms, bs = wmean(say, m)
        md, bd = wmean(do, m)
        es, bes = wmean(np.abs(p - say), m)
        ed, bed = wmean(np.abs(p - do), m)
        r.update(mean=mp, mean_say=ms, mean_do=md, dist_say=abs(mp - ms), dist_do=abs(mp - md),
                 mae_say=es, mae_do=ed, mae_diff=es - ed)
        r["diff"] = r["dist_say"] - r["dist_do"]
        if r["clusters"] >= 2:
            r["lo"], r["hi"] = _percentile_ci(np.abs(bp - bs) - np.abs(bp - bd), alpha)
            r["mae_lo"], r["mae_hi"] = _percentile_ci(bes - bed, alpha)
        lo, hi = (r["lo"], r["hi"]) if on == "level" else (r["mae_lo"], r["mae_hi"])
        if lo is None:
            r["reason"] = "fewer than 2 clusters: no CI"
        elif not anchors["apart"]:
            r["reason"] = "anchors not apart (do − say gap CI covers 0)"
        elif hi < 0:
            r["verdict"] = SAY
        elif lo > 0:
            r["verdict"] = DO
        else:
            r["reason"] = f"{pct} % CI of the {on} difference covers 0"
        key = (r["diff"], lo, hi) if on == "level" else (r["mae_diff"], lo, hi)
        r["card_line"] = (f"{c[len(PRED_PREFIX):] if c.startswith(PRED_PREFIX) else c}: {r['verdict']} "
                          f"({on} say − do = {_fmt(key[0])} [{_fmt(key[1])}, {_fmt(key[2])}])")
        out.append(r)

    notes = [f"diff = distance to say − distance to do (negative = closer to say); level uses panel means, "
             f"MAE uses person-level absolute errors. {pct} % percentile bootstrap over {G} {unit}(s), "
             f"B = {B}, seed = {seed}; verdict reads the {on} CI.",
             f"Anchors: mean say {_fmt(anchors['mean_say'])}, mean do {_fmt(anchors['mean_do'])}, "
             f"do − say {_fmt(anchors['gap'])} [{_fmt(g_lo)}, {_fmt(g_hi)}]"
             + ("" if anchors["apart"] else " — not apart: every verdict is undecided.")] + notes
    miss = [f"{r['prediction']} {r['n_missing']}" for r in out if r["n_missing"]]
    if miss:
        notes.append("Rows dropped (missing prediction or anchor, not imputed): " + ", ".join(miss) + ".")
    rows = [["anchors (do − say)", anchors["n"], None, anchors["mean_say"], anchors["mean_do"], None, None,
             anchors["gap"], g_lo, g_hi, None, None, None, None, None, "apart" if anchors["apart"] else "not apart"]]
    rows += [[r["prediction"], r["n"], r["mean"], r["mean_say"], r["mean_do"], r["dist_say"], r["dist_do"],
              r["diff"], r["lo"], r["hi"], r["mae_say"], r["mae_do"], r["mae_diff"], r["mae_lo"], r["mae_hi"],
              r["verdict"]] for r in out]
    return _section("say_do", "Say or do: which anchor the prediction is closer to",
                    ["prediction", "n", "mean", "mean say", "mean do", "|mean − say|", "|mean − do|", "diff", "lo",
                     "hi", "MAE say", "MAE do", "MAE diff", "MAE lo", "MAE hi", "verdict"], rows, notes,
                    predictions=out, anchors=anchors, say_col=sc, do_col=dc, cluster=cl,
                    n_clusters=G, on=on, B=B, seed=seed, alpha=alpha)


# ------------------------------------------------------------------ rendering
def _as_list(sections) -> list[dict]:
    if isinstance(sections, Mapping):
        return [dict(sections)]
    return [dict(s) for s in sections]


def _md_cell(x, digits) -> str:
    return _fmt(x, digits).replace("|", "\\|").replace("\n", " ")


def section_markdown(sections: Mapping | Sequence[Mapping], *, level: int = 3) -> str:
    """Markdown for one section (or a list of them): a heading, a table and the notes as bullets."""
    parts = []
    for s in _as_list(sections):
        digits = int(s.get("digits", 3))
        lines = [f"{'#' * level} {s.get('title', s.get('kind', 'section'))}", ""]
        cols = list(s.get("columns", []))
        if cols:
            lines.append("| " + " | ".join(_md_cell(c, digits) for c in cols) + " |")
            lines.append("|" + "|".join("---" for _ in cols) + "|")
            for r in s.get("rows", []):
                lines.append("| " + " | ".join(_md_cell(v, digits) for v in r) + " |")
            lines.append("")
        for n in s.get("notes", []):
            lines.append(f"- {n}")
        parts.append("\n".join(lines).rstrip() + "\n")
    return "\n".join(parts)


def section_html(sections: Mapping | Sequence[Mapping], *, level: int = 3) -> str:
    """HTML fragment (no external assets) for one section or a list: heading, table, notes."""
    esc = _html.escape
    level = min(max(int(level), 1), 6)
    parts = []
    for s in _as_list(sections):
        digits = int(s.get("digits", 3))
        h = [f'<section class="se-extra se-{esc(str(s.get("kind", "section")))}">',
             f"<h{level}>{esc(str(s.get('title', s.get('kind', 'section'))))}</h{level}>"]
        cols = list(s.get("columns", []))
        if cols:
            h.append("<table><thead><tr>" + "".join(f"<th>{esc(str(c))}</th>" for c in cols) + "</tr></thead><tbody>")
            for r in s.get("rows", []):
                h.append("<tr>" + "".join(f"<td>{esc(_fmt(v, digits))}</td>" for v in r) + "</tr>")
            h.append("</tbody></table>")
        notes = s.get("notes", [])
        if notes:
            h.append("<ul>" + "".join(f"<li>{esc(str(n))}</li>" for n in notes) + "</ul>")
        h.append("</section>")
        parts.append("\n".join(h))
    return "\n".join(parts) + "\n"
