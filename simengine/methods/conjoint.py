"""Conditional multinomial logit (MNL) for discrete-choice / conjoint data, with pricing inference.

Port of `uretim/deney/kod/eval/baseline_choice.py` (`_design`, `_nll`, `fit_mnl`, `eval_mnl`): generic attributes
(optionally divided by a scale) + alternative-specific constants for every alternative except ``base_alt``,
availability-masked softmax per task, L-BFGS-B from zeros. Evaluation: 20% of persons held out, accuracy = argmax
alternative per task, mean log-likelihood per task, compared with the null (equal share) and an ASC-only model.
(The port adds the analytic gradient and Hessian; the optimum is the same.)

Long input format, one row per (task, alternative):

- ``task``: choice-set id, unique across persons (two persons answering "task 1" need two different ids);
- ``alternative``: alternative id within the task (the key of the ASCs, e.g. a brand or a position);
- ``chosen``: 0/1, exactly one 1 among the available rows of a task. A "none of these" option is modelled as its
  own alternative row (attributes 0, its own ASC), not as a task with no chosen row;
- ``available`` (optional, default 1): 0 removes the row from the choice set;
- ``person`` (optional): respondent id, used by `person_holdout`, `evaluate_mnl` and the cluster bootstrap;
- attribute columns (numeric).

Research aliases gorev_id / alternatif_id / secildi / mevcut / birey_id are accepted.

Pricing (all prices and attributes in their raw units; ``scale`` is undone):

    m = MNL(["price", "quality"], scale={"price": 100}).fit(rows)
    m.coef_, m.se_                                   # part-worths (scaled units) and Hessian standard errors
    wtp(m, price="price")                            # -beta_attr / beta_price, delta-method CI
    wtp(m, ci="bootstrap", cluster="person", B=200)  # person-cluster bootstrap CI
    market = pd.DataFrame({"alternative": [1, 2, 3], "price": [90, 110, 100], "quality": [1, 1.5, 1]})
    simulate_shares(m, market)                       # logit shares
    elasticity(m, market)                            # own / cross analytic + arc elasticities

Measured (uretim/deney/RAPOR.md 1.5, Sawtooth camera conjoint): new person, holdout accuracy equal share 0.210,
MNL 0.509, hierarchical Bayes 0.519; same person, new task: MNL 0.482 vs HB 0.715 (person-level tastes matter;
HB is not ported). LLM arms did not beat MNL/HB on these sets.
"""
from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from ._util import rename_aliases

ALIASES = {"gorev_id": "task", "alternatif_id": "alternative", "secildi": "chosen", "mevcut": "available",
           "birey_id": "person"}
REQUIRED = ("task", "alternative", "chosen")


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_aliases(df, ALIASES)
    if "available" not in df.columns:
        df = df.assign(available=1)
    return df


def _check_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the long format on available rows; drop (with a warning) tasks whose chosen row is unavailable."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"MNL: missing required column(s) {missing}; long format is one row per (task, "
                         "alternative) with task, alternative, chosen (0/1)")
    bad = ~df["chosen"].isin([0, 1, True, False])
    if bad.any():
        raise ValueError(f"MNL: 'chosen' must be 0/1, got {sorted(df.loc[bad, 'chosen'].unique().tolist())[:5]}")
    n = df.groupby("task")["chosen"].sum()
    many = n[n > 1]
    if len(many):
        raise ValueError(f"MNL: {len(many)} task(s) have more than one chosen alternative (e.g. task "
                         f"{many.index[0]!r} has {int(many.iloc[0])}); task ids must be unique across persons")
    none = n[n == 0]
    if len(none):
        warnings.warn(f"MNL: dropped {len(none)} task(s) with no chosen available alternative (e.g. task "
                      f"{none.index[0]!r}); model a no-choice option as its own alternative row", stacklevel=3)
        df = df[~df["task"].isin(none.index)]
    if df.empty:
        raise ValueError("MNL: no task with a chosen available alternative")
    return df


class MNL:
    """Conditional logit: fit(df) / predict_proba(df) / predict(df) / log_likelihood(df) / accuracy(df).

    ``attrs``: generic attribute columns (None -> all columns starting with "Feat", as in the source);
    ``scale``: {attr: divisor}; ``base_alt``: alternative without an ASC (default: smallest id);
    ``asc``: include alternative-specific constants; ``asc_only``: constants only (the ASC-only baseline).

    Fitted attributes: ``coef_`` (pd.Series: attribute part-worths per *scaled* unit, then ``ASC_<alt>``),
    ``se_`` (pd.Series, standard errors from the inverse analytic Hessian), ``cov_`` (pd.DataFrame, that inverse),
    ``attrs_`` (attribute columns used), ``alts_`` (sorted alternative ids seen), ``base_alt_``, ``converged_``
    (L-BFGS-B success flag), ``loglik_`` (total log-likelihood at the optimum), ``n_tasks_``.
    """

    def __init__(self, attrs: Sequence[str] | None = None, scale: Mapping[str, float] | None = None,
                 base_alt=None, asc: bool = True, asc_only: bool = False):
        self.attrs, self.scale, self.base_alt = attrs, dict(scale or {}), base_alt
        self.asc, self.asc_only = asc, asc_only

    def _design(self, df):
        if not self.asc_only:
            missing = [a for a in self.attrs_ if a not in df.columns]
            if missing:
                raise ValueError(f"MNL: missing attribute column(s) {missing}")
        X = [] if self.asc_only else [df[a].astype(float).fillna(0).to_numpy() / self.scale.get(a, 1.0)
                                      for a in self.attrs_]
        if self.asc or self.asc_only:
            for j in self.alts_:
                if j != self.base_alt_:
                    X.append((df["alternative"] == j).astype(float).to_numpy())
        return np.column_stack(X) if X else np.zeros((len(df), 0))

    @staticmethod
    def _logp(beta, X, g, av):
        v = np.where(av > 0, X @ beta, -1e9)
        vmax = np.full(g.max() + 1, -np.inf)
        np.maximum.at(vmax, g, v)
        ev = np.exp(v - vmax[g])
        den = np.bincount(g, weights=ev, minlength=len(vmax))
        return (v - vmax[g]) - np.log(den[g])

    def fit(self, df: pd.DataFrame):
        """Maximum likelihood on long-format rows (see the module docstring); returns self."""
        df = _prep(df)
        df = _check_tasks(df[df["available"] > 0].reset_index(drop=True)).reset_index(drop=True)
        if self.attrs is not None:
            self.attrs_ = list(self.attrs)
        else:
            self.attrs_ = [c for c in df.columns if str(c).startswith("Feat")]
        if not self.attrs_ and not self.asc_only:
            raise ValueError("MNL: no attribute columns (attrs=None looks for columns starting with 'Feat'); pass "
                             "attrs=[...] or asc_only=True for the constants-only baseline")
        self.alts_ = sorted(df["alternative"].unique())
        self.base_alt_ = self.base_alt if self.base_alt is not None else self.alts_[0]
        names = ([] if self.asc_only else list(self.attrs_))
        if self.asc or self.asc_only:
            names += [f"ASC_{j}" for j in self.alts_ if j != self.base_alt_]
        X = self._design(df)
        g = pd.factorize(df["task"])[0]
        y, av = df["chosen"].to_numpy(float), df["available"].to_numpy(float)
        sp = np.bincount(g, weights=y)                        # chosen count per task (1 after validation)

        def f(b):
            lp = self._logp(b, X, g, av)
            p = np.exp(lp)
            grad = -(X.T @ (y - p * sp[g]))
            return -(lp * y).sum(), grad

        r = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
        self.coef_ = pd.Series(r.x, index=names, dtype=float)
        self.converged_ = bool(r.success)
        self.loglik_ = float(-r.fun)
        self.n_tasks_ = int(len(sp))
        # Hessian of -LL: sum_t sp_t * X_t' (diag p_t - p_t p_t') X_t
        p = np.exp(self._logp(r.x, X, g, av))
        xbar = np.zeros((len(sp), X.shape[1]))
        np.add.at(xbar, g, p[:, None] * X)
        H = (X * (p * sp[g])[:, None]).T @ X - (xbar * sp[:, None]).T @ xbar
        try:
            cov = np.linalg.inv(H) if np.linalg.cond(H) < 1e12 else np.full_like(H, np.nan)
        except np.linalg.LinAlgError:
            cov = np.full_like(H, np.nan)
        self.cov_ = pd.DataFrame(cov, index=names, columns=names)
        self.se_ = pd.Series(np.sqrt(np.clip(np.diag(cov), 0, None)), index=names)
        self._fit_data = df                                   # for the bootstrap in `wtp`
        return self

    def _lp(self, df):
        df = _prep(df).reset_index(drop=True)
        g = pd.factorize(df["task"])[0]
        return df, self._logp(self.coef_.to_numpy(), self._design(df), g, df["available"].to_numpy(float))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Choice probability per row, in row order (0 for unavailable rows)."""
        return np.exp(self._lp(df)[1])

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Most likely alternative per task (index = task)."""
        d, lp = self._lp(df)
        d = d.assign(_lp=lp)
        return d.loc[d.groupby("task")["_lp"].idxmax()].set_index("task")["alternative"]

    def log_likelihood(self, df: pd.DataFrame) -> float:
        """Mean log-likelihood of the chosen alternatives per task."""
        d, lp = self._lp(df)
        avail = d["available"].to_numpy() > 0
        return float((lp[avail] * d["chosen"].to_numpy()[avail]).sum() / d["task"].nunique())

    def accuracy(self, df: pd.DataFrame) -> float:
        """Share of tasks where the most likely alternative is the chosen one."""
        d, lp = self._lp(df)
        d = d.assign(_lp=lp)
        return float(d.loc[d.groupby("task")["_lp"].idxmax(), "chosen"].mean())


def _raw_coef(model: MNL, name: str) -> float:
    """Part-worth per raw unit (the ``scale`` divisor undone); ASCs are unscaled."""
    return float(model.coef_[name]) / model.scale.get(name, 1.0)


def _price_coef(model: MNL, price: str) -> float:
    """Raw-unit price coefficient, with an error naming the fitted coefficients when ``price`` is not one."""
    if price not in getattr(model, "coef_", pd.Series(dtype=float)).index:
        raise ValueError(f"price column {price!r} is not a fitted attribute (fitted: {list(model.coef_.index)})")
    return _raw_coef(model, price)


def _resample(df: pd.DataFrame, cluster: str | None, rng) -> pd.DataFrame:
    """One bootstrap replicate: clusters (or tasks) drawn with replacement; duplicate draws get new task ids."""
    key = cluster if cluster is not None else "task"
    codes, uniq = pd.factorize(df[key])
    if (codes < 0).any():
        raise ValueError(f"wtp: missing values in the bootstrap cluster column {key!r}")
    tcode, tuniq = pd.factorize(df["task"])
    order = np.argsort(codes, kind="stable")
    rows = np.split(order, np.cumsum(np.bincount(codes, minlength=len(uniq)))[:-1])
    draw = rng.integers(0, len(uniq), len(uniq))
    idx = np.concatenate([rows[i] for i in draw])
    rep = np.repeat(np.arange(len(draw)), [len(rows[i]) for i in draw])
    out = df.iloc[idx].copy()
    out["task"] = rep * len(tuniq) + tcode[idx]
    if cluster is not None:
        out[cluster] = rep
    return out


def wtp(model: MNL, price: str = "price", ci: str = "delta", cluster: str | None = "person", B: int = 200,
        seed: int = 0, level: float = 0.95, attrs: Sequence[str] | None = None,
        data: pd.DataFrame | None = None) -> pd.DataFrame:
    """Willingness to pay per attribute, -beta_attr / beta_price, in price units per raw attribute unit.

    ``attrs``: coefficient names (default all fitted attributes except ``price``; ``ASC_<alt>`` names allowed,
    giving the price premium of that alternative over the base one). ``ci="delta"``: delta method on ``cov_``;
    ``ci="bootstrap"``: refit on ``B`` replicates that resample ``cluster`` (None = tasks) with replacement from
    ``data`` (default the training rows); resampled duplicates get fresh task ids. Returns a DataFrame indexed by
    attribute with columns wtp, se, lo, hi (``level`` interval; percentile for the bootstrap).
    """
    bp = _price_coef(model, price)
    names = list(attrs) if attrs is not None else [a for a in model.coef_.index
                                                   if a != price and not str(a).startswith("ASC_")]
    unknown = [a for a in names if a not in model.coef_.index]
    if unknown:
        raise ValueError(f"wtp: unknown coefficient(s) {unknown}")
    w = pd.Series({a: -_raw_coef(model, a) / bp for a in names}, dtype=float)
    z = norm.ppf(0.5 + level / 2)
    if ci == "delta":
        sp, se = model.scale.get(price, 1.0), {}
        for a in names:
            sa = model.scale.get(a, 1.0)
            ba, bq = float(model.coef_[a]), float(model.coef_[price])
            grad = np.array([-(sp / sa) / bq, (sp / sa) * ba / bq ** 2])
            C = model.cov_.loc[[a, price], [a, price]].to_numpy()
            se[a] = float(np.sqrt(max(grad @ C @ grad, 0.0)))
        se = pd.Series(se, dtype=float)
        return pd.DataFrame({"wtp": w, "se": se, "lo": w - z * se, "hi": w + z * se})
    if ci != "bootstrap":
        raise ValueError(f"wtp: ci must be 'delta' or 'bootstrap', got {ci!r}")
    df = _prep(data) if data is not None else model._fit_data
    if cluster is not None and cluster not in df.columns:
        raise ValueError(f"wtp: cluster column {cluster!r} not in the data (cluster=None resamples tasks)")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        m = MNL(model.attrs_, model.scale, model.base_alt_, model.asc, model.asc_only)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(_resample(df, cluster, rng))
        draws.append([-_raw_coef(m, a) / _raw_coef(m, price) for a in names])
    D = np.asarray(draws)
    a = (1 - level) / 2
    return pd.DataFrame({"wtp": w, "se": D.std(0, ddof=1), "lo": np.quantile(D, a, axis=0),
                         "hi": np.quantile(D, 1 - a, axis=0)}, index=names)


def _market(model: MNL, market: pd.DataFrame) -> pd.DataFrame:
    """Market rows as a long frame: aliases renamed, ``task`` (one market) and ``alternative`` filled, alternatives
    checked against the fitted ASCs."""
    m = _prep(market)
    if "task" not in m.columns:
        m = m.assign(task=0)
    uses_asc = (model.asc or model.asc_only) and len(model.alts_) > 1
    if "alternative" not in m.columns:
        if uses_asc:
            raise ValueError("market needs an 'alternative' column: the model has alternative-specific constants")
        m = m.assign(alternative=model.base_alt_)
    elif uses_asc:
        unknown = sorted(set(m["alternative"]) - set(model.alts_), key=str)
        if unknown:
            raise ValueError(f"market alternative(s) {unknown} were not seen in fit (known: {model.alts_})")
    return m


def simulate_shares(model: MNL, market: pd.DataFrame) -> pd.Series:
    """Logit shares of the alternatives in ``market`` (rows = alternatives with the attribute columns; optional
    ``alternative``, ``available`` and ``task`` = market id, default one market). Returns a Series on market.index."""
    return pd.Series(model.predict_proba(_market(model, market)), index=market.index, name="share")


def elasticity(model: MNL, market: pd.DataFrame, price: str = "price",
               price_grid: Sequence[float] | None = None) -> dict:
    """Price elasticities of the shares in one market.

    Returns {"share": shares, "own": analytic own-price elasticity beta_p * p_i * (1 - s_i), "cross": matrix
    E[i, j] = d log s_i / d log p_j (diagonal = own, off-diagonal -beta_p * p_j * s_j), "arc": DataFrame of
    midpoint arc elasticities of s_i when p_i alone is multiplied by each ``price_grid`` factor (relative to the
    market price, default 0.9, 0.95, 1.05, 1.1)}. ``beta_p`` is per raw price unit. Indexed by alternative.
    """
    bp = _price_coef(model, price)
    m = _market(model, market)
    if m["task"].nunique() != 1:
        raise ValueError("elasticity: pass one market (a single task)")
    if price not in m.columns:
        raise ValueError(f"elasticity: market has no {price!r} column")
    named = "alternative" in rename_aliases(market, ALIASES).columns
    labels = pd.Index(m["alternative"] if named else market.index, name="alternative")
    s = model.predict_proba(m)
    p = m[price].to_numpy(float)
    E = np.tile(-bp * p * s, (len(s), 1))
    np.fill_diagonal(E, bp * p * (1 - s))
    grid = [1.0 + d for d in (-0.1, -0.05, 0.05, 0.1)] if price_grid is None else list(price_grid)
    rows = []
    for i in range(len(s)):
        for k in grid:
            if k == 1.0:
                continue
            m2 = m.copy()
            m2.iloc[i, m2.columns.get_loc(price)] = p[i] * k
            s2 = model.predict_proba(m2)[i]
            dp = (p[i] * k - p[i]) / ((p[i] * k + p[i]) / 2)
            ds = (s2 - s[i]) / ((s2 + s[i]) / 2) if s2 + s[i] > 0 else 0.0
            rows.append({"alternative": labels[i], "factor": k, "price": p[i] * k, "share": s2,
                         "arc": ds / dp if dp else np.nan})
    return {"share": pd.Series(s, index=labels, name="share"),
            "own": pd.Series(np.diag(E).copy(), index=labels, name="own"),
            "cross": pd.DataFrame(E, index=labels, columns=labels),
            "arc": pd.DataFrame(rows, columns=["alternative", "factor", "price", "share", "arc"])}


def null_scores(df: pd.DataFrame) -> dict:
    """Equal-share baseline: accuracy = mean 1/n_alternatives, log-likelihood per task = -mean log n_alternatives.
    Keys acc / ll_per_task (source names) and their aliases accuracy / log_likelihood."""
    df = _prep(df)
    n = df[df["available"] > 0].groupby("task").size()
    acc, ll = float((1 / n).mean()), float(-np.log(n).mean())
    return {"acc": acc, "ll_per_task": ll, "accuracy": acc, "log_likelihood": ll}


def person_holdout(df: pd.DataFrame, frac: float = 0.2, seed: int = 0):
    """Split rows by person (``frac`` of persons to test), as in the source evaluation. Returns (train, test)."""
    df = _prep(df)
    if "person" not in df.columns:
        raise ValueError("person_holdout: needs a 'person' column")
    ids = df["person"].unique()
    rng = np.random.default_rng(seed)
    test_ids = set(rng.choice(ids, size=max(1, int(len(ids) * frac)), replace=False))
    te = df["person"].isin(test_ids).to_numpy()
    return df[~te], df[te]


def evaluate_mnl(df: pd.DataFrame, attrs: Sequence[str] | None = None, scale: Mapping[str, float] | None = None,
                 base_alt=None, frac: float = 0.2, seed: int = 0) -> dict:
    """Source `eval_mnl`: person holdout; MNL, ASC-only and null accuracy / log-likelihood per task on the test
    persons, plus the MNL coefficients (full precision; the source rounded them to 3 decimals). Each arm has keys
    acc / ll_per_task and the aliases accuracy / log_likelihood."""
    df = _prep(df)
    df = df[df["available"] > 0]
    tr, te = person_holdout(df, frac, seed)
    out = {}
    for label, kw in (("asc_only", dict(asc_only=True)), ("mnl", {})):
        m = MNL(attrs, scale, base_alt, **kw).fit(tr)
        acc, ll = m.accuracy(te), m.log_likelihood(te)
        out[label] = {"acc": acc, "ll_per_task": ll, "accuracy": acc, "log_likelihood": ll,
                      "coef": m.coef_.to_dict(), "se": m.se_.to_dict()}
    out["null"] = null_scores(te)
    out["n"] = {"test_tasks": int(te["task"].nunique()), "test_persons": int(te["person"].nunique()),
                "train_tasks": int(tr["task"].nunique())}
    return out
