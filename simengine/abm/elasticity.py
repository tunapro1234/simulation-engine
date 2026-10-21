"""Price-elasticity prior + ridge shrinkage toward the event's own history (ported from the TALP prep kit
`yontemler/llm_onsel.py` and `temel_m2.py`).

Model per event e:  log1p(y_t) = a_e + b_e * lp_t + g_e * promo_t   (lp = log(price / pre-event regular price)).

Priors (b0, g0):
* ``category``      : category OLS on the training events' full windows, demeaned within event (= log-log FE baseline).
* ``llm``           : per-unit elasticity from an LLM (b0), g0 from the category.
* ``llm_deviation`` : category level + the LLM's within-category deviation (b0 = b_cat + e_llm - mean_cat(e_llm)).
* ``mix``           : (category + llm) / 2.
"with data": ridge toward the prior on the target event's own t <= 0 window,
(Z'Z + lam I) theta = Z'y + lam theta0 (demeaned within event); lam is picked by inner validation on training events
(fit on t <= 0, forecast t = 1..H, raw WAPE). Back-transform with Duan smearing.

Measured (makro/RAPOR.md, M2 Dominick's, WAPE 8 weeks, dev / test): pre-event 26-week mean 0.598 / 0.616,
category log-log 0.466 / 0.479, category + own history 0.466 / 0.477, LLM expert prior + data 0.469 / 0.469,
(category + LLM)/2 + data 0.465 / 0.470, LLM archetype prior + data 0.479 / 0.482. The small test gain of the mix
(-0.0072 [-0.011, -0.003]) is matched by an LLM-SWAP PLACEBO (LLM elasticities shuffled across units within category:
-0.0085 / -0.0066 / -0.0039). The gain comes from shrinkage, not from LLM knowledge. Always run
:func:`shuffle_elasticities` as a control before crediting an LLM prior.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

LAM_GRID = [0.3, 1, 3, 10, 30, 100, 1e9]          # 1e9 ~ prior only
INNER_SAMPLE = 3000
PRIORS = ("category", "llm", "llm_deviation", "mix")


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    ren = {"seri": "series", "kat": "category"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns and v not in df.columns})
    need = {"series", "t", "y", "lp", "promo", "category"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"event table needs columns {sorted(need)}; missing {sorted(miss)}")
    return df.assign(ly=np.log1p(df["y"].astype(float)), lp=df["lp"].astype(float), promo=df["promo"].astype(float))


def category_elasticity(train_full: pd.DataFrame) -> dict:
    """{category: (array([b, g]), smearing)}: within-event demeaned OLS per category + Duan smearing factor."""
    d = _prep(train_full)
    out = {}
    for cat, g in d.groupby("category"):
        X = g[["lp", "promo"]]
        Xd = (X - X.groupby(g["series"]).transform("mean")).to_numpy()
        yd = (g["ly"] - g["ly"].groupby(g["series"]).transform("mean")).to_numpy()
        b, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
        out[cat] = (b, float(np.mean(np.exp(yd - Xd @ b))))
    return out


def _sufficient(pre: pd.DataFrame) -> pd.DataFrame:
    """Per event: demeaned Z'Z, Z'y on the t <= 0 window, plus means; plus category and unit."""
    m = pre.groupby("series")[["lp", "promo", "ly"]].transform("mean")
    zx, zs, zy = pre["lp"] - m["lp"], pre["promo"] - m["promo"], pre["ly"] - m["ly"]
    s = pd.DataFrame({"series": pre["series"], "xx": zx * zx, "xs": zx * zs, "ss": zs * zs, "xy": zx * zy,
                      "sy": zs * zy, "mx": pre["lp"], "ms": pre["promo"], "my": pre["ly"]}).groupby("series")
    st = s[["xx", "xs", "ss", "xy", "sy"]].sum().join(s[["mx", "ms", "my"]].mean())
    first = pre.groupby("series").first()
    st["category"] = first["category"]
    st["unit"] = first["unit"] if "unit" in first else first.index
    return st


def _ridge(st, b0, g0, lam):
    """Vectorised 2x2 ridge: one row per event."""
    a11, a12, a22 = st["xx"].to_numpy() + lam, st["xs"].to_numpy(), st["ss"].to_numpy() + lam
    r1, r2 = st["xy"].to_numpy() + lam * b0, st["sy"].to_numpy() + lam * g0
    det = a11 * a22 - a12 * a12
    return (a22 * r1 - a12 * r2) / det, (a11 * r2 - a12 * r1) / det


def shuffle_elasticities(llm_elasticity: dict, categories: dict, seed=1) -> dict:
    """LLM-swap placebo: permute the LLM elasticities across units WITHIN each category.
    llm_elasticity: {unit: e}; categories: {unit: category}."""
    rng = np.random.default_rng(seed)
    out = {}
    by = {}
    for u, e in llm_elasticity.items():
        by.setdefault(categories.get(u), []).append((u, e))
    for _, lst in by.items():
        us = [u for u, _ in lst]
        es = rng.permutation(np.array([e for _, e in lst], float))
        out.update(dict(zip(us, es.tolist())))
    return out


class ElasticityShrinkage:
    """Elasticity prior (category / LLM / mix) with ridge shrinkage toward the event's own pre-period."""
    kind = "method"

    def __init__(self, prior="category", llm_elasticity: dict | None = None, with_data=True, lam_grid=None,
                 inner_sample=INNER_SAMPLE, seed=20261021, name=None):
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}")
        if prior != "category" and not llm_elasticity:
            raise ValueError(f"prior={prior!r} needs llm_elasticity {{unit: elasticity}}")
        self.prior, self.llm, self.with_data = prior, dict(llm_elasticity or {}), with_data
        self.lam_grid = list(lam_grid or LAM_GRID)
        self.inner_sample, self.seed = inner_sample, seed
        self.name = name or (prior + ("+data" if with_data else ""))
        if prior == "category" and not with_data:
            self.kind = "baseline"

    def _prior(self, st, coefs):
        bk = np.array([coefs[c][0][0] for c in st["category"]])
        g0 = np.array([coefs[c][0][1] for c in st["category"]])
        if self.prior == "category":
            return bk, g0
        el = np.array([self.llm.get(u, np.nan) for u in st["unit"]], float)
        if self.prior == "llm":
            b0 = el
        elif self.prior == "mix":
            b0 = (bk + el) / 2
        else:  # llm_deviation: level from data, deviation from the LLM (relative to its category mean over units)
            cm = getattr(self, "_cat_mean", {})
            b0 = bk + el - np.array([cm.get(c, np.nan) for c in st["category"]])
        return np.where(np.isnan(b0), bk, b0), g0

    @staticmethod
    def _yhat(query, st, coefs):
        s = st.reindex(query["series"])
        b, g = s["b"].to_numpy(), s["g"].to_numpy()
        a = s["my"].to_numpy() - b * s["mx"].to_numpy() - g * s["ms"].to_numpy()
        ly = a + b * query["lp"].to_numpy(float) + g * query["promo"].to_numpy(float)
        smear = np.array([coefs[c][1] for c in s["category"]])
        return np.maximum(np.expm1(ly) * smear, 0)

    def fit(self, train_full: pd.DataFrame, target_pre: pd.DataFrame) -> dict:
        """train_full: training events' full windows (t <= 0 and t >= 1). target_pre: target events, t <= 0 only."""
        tr = _prep(train_full)
        tp = _prep(target_pre)
        coefs = category_elasticity(train_full)
        if self.prior == "llm_deviation":
            ucol = "unit" if "unit" in tr.columns else "series"
            uc = pd.concat([tr[[ucol, "category"]], tp[[ucol, "category"]]]).drop_duplicates(ucol)
            uc = uc[uc[ucol].isin(set(self.llm))]
            self._cat_mean = {c: float(np.mean([self.llm[u] for u in g[ucol]])) for c, g in uc.groupby("category")}
        lam = 1e9
        if self.with_data:
            ids = sorted(tr["series"].unique())
            if len(ids) > self.inner_sample:
                ids = list(np.random.default_rng(self.seed).choice(ids, self.inner_sample, replace=False))
            inner = tr[tr["series"].isin(set(ids))]
            st = _sufficient(inner[inner["t"] <= 0])
            fut = inner[inner["t"] >= 1]
            b0, g0 = self._prior(st, coefs)
            best = None
            for l in self.lam_grid:
                b, g = _ridge(st, b0, g0, l)
                yh = self._yhat(fut, st.assign(b=b, g=g), coefs)
                w = np.abs(yh - fut["y"].to_numpy(float)).sum() / max(fut["y"].sum(), 1e-12)
                if best is None or w < best:
                    best, lam = w, l
        st = _sufficient(tp[tp["t"] <= 0])
        b0, g0 = self._prior(st, coefs)
        b, g = _ridge(st, b0, g0, lam) if self.with_data else (b0, g0)
        return dict(st=st.assign(b=b, g=g), coefs=coefs, lam=lam)

    def predict(self, model, query: pd.DataFrame) -> pd.DataFrame:
        """query: series, lp, promo (+ any columns, e.g. t / horizon). Adds y_pred."""
        q = query.rename(columns={"seri": "series"}) if "seri" in query.columns else query
        return q.assign(y_pred=self._yhat(q, model["st"], model["coefs"]))


class PreEventMean:
    """Know-nothing baseline: mean of the last `weeks` pre-event weeks (research `bilmeyen_ort26` / `ort8`)."""
    kind = "baseline"

    def __init__(self, weeks=26):
        self.weeks = weeks
        self.name = f"pre_mean{weeks}"

    def fit(self, train_full, target_pre):
        d = target_pre.rename(columns={"seri": "series"}) if "seri" in target_pre.columns else target_pre
        d = d[(d["t"] <= 0) & (d["t"] > -self.weeks)]
        return d.groupby("series")["y"].mean()

    def predict(self, model, query):
        q = query.rename(columns={"seri": "series"}) if "seri" in query.columns else query
        return q.assign(y_pred=q["series"].map(model).fillna(0).to_numpy())


def wape(y_pred, y_true) -> float:
    y_true = np.asarray(y_true, float)
    return float(np.abs(np.asarray(y_pred, float) - y_true).sum() / max(y_true.sum(), 1e-12))


# ------------------------------------------------------------------ LLM elicitation
ELICIT_CONTEXT = ("Setting: a large supermarket chain. Prices include temporary price reductions; assume no extra "
                  "display or feature advertising.")


def elasticity_prompt(items: dict, context: str = ELICIT_CONTEXT) -> str:
    """The research 'expert' prompt (anonymised: no chain or store names). items: {id: description}."""
    return (f"You are a retail pricing analyst. {context}\nFor each item, estimate the short-run own-price "
            "elasticity of WEEKLY unit sales of THIS item at one store (percent change in units for a 1% change "
            "in its price, competing items' prices unchanged), with an 80% interval.\n\n"
            + "\n".join(f"{i}: {d}" for i, d in items.items())
            + '\n\nReturn ONLY JSON: [{"id": "<id>", "elasticity": <float>, "low": <float>, "high": <float>}, ...]')


def _parse_list(text: str):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        v = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return v if isinstance(v, list) else []


def elicit_elasticities(llm: Any, items: dict, context: str = ELICIT_CONTEXT, batch=20, max_tokens=2000,
                        **kw) -> dict:
    """Ask an LLM (simengine.types.LLM protocol) for per-item elasticities, `batch` items per call.
    Returns {id: elasticity}; unparseable items are dropped (count them before trusting the prior)."""
    ids = list(items)
    prompts = [("", elasticity_prompt({i: items[i] for i in ids[j:j + batch]}, context))
               for j in range(0, len(ids), batch)]
    out = {}
    for r in llm.map(prompts, max_tokens=max_tokens, **kw):
        for x in _parse_list(r.text):
            try:
                key = x["id"]
                key = next((i for i in ids if str(i) == str(key)), key)
                out[key] = float(x["elasticity"])
            except (KeyError, TypeError, ValueError):
                continue
    return out
