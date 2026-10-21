"""Conditional multinomial logit (MNL) for discrete-choice / conjoint data.

Port of `uretim/deney/kod/eval/baseline_choice.py` (`_design`, `_nll`, `fit_mnl`, `eval_mnl`): generic attributes
(optionally divided by a scale) + alternative-specific constants for every alternative except ``base_alt``,
availability-masked softmax per task, L-BFGS-B from zeros. Evaluation: 20% of persons held out, accuracy = argmax
alternative per task, mean log-likelihood per task, compared with the null (equal share) and an ASC-only model.
(The port adds the analytic gradient; the optimum is the same.)

Long input, one row per (task, alternative): columns ``task``, ``alternative``, ``chosen`` (0/1), optional
``available`` (default 1) and ``person``; research aliases gorev_id / alternatif_id / secildi / mevcut / birey_id.

Measured (uretim/deney/RAPOR.md 1.5, Sawtooth camera conjoint): new person, holdout accuracy equal share 0.210,
MNL 0.509, hierarchical Bayes 0.519; same person, new task: MNL 0.482 vs HB 0.715 (person-level tastes matter;
HB is not ported). LLM arms did not beat MNL/HB on these sets.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ._util import rename_aliases

ALIASES = {"gorev_id": "task", "alternatif_id": "alternative", "secildi": "chosen", "mevcut": "available",
           "birey_id": "person"}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_aliases(df, ALIASES)
    if "available" not in df.columns:
        df = df.assign(available=1)
    return df


class MNL:
    """fit(df) / predict_proba(df) (choice probability per row) / predict(df) (chosen alternative per task) /
    log_likelihood(df) (mean per task) / accuracy(df).

    ``attrs``: generic attribute columns (None -> all columns starting with "Feat", as in the source);
    ``scale``: {attr: divisor}; ``base_alt``: alternative without an ASC (default: smallest id);
    ``asc``: include alternative-specific constants; ``asc_only``: constants only (the ASC-only baseline).
    """

    def __init__(self, attrs: Sequence[str] | None = None, scale: Mapping[str, float] | None = None,
                 base_alt=None, asc: bool = True, asc_only: bool = False):
        self.attrs, self.scale, self.base_alt = attrs, dict(scale or {}), base_alt
        self.asc, self.asc_only = asc, asc_only

    def _design(self, df):
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
        df = _prep(df)
        df = df[df["available"] > 0].reset_index(drop=True)
        self.attrs_ = list(self.attrs) if self.attrs is not None else [c for c in df.columns if c.startswith("Feat")]
        self.alts_ = sorted(df["alternative"].unique())
        self.base_alt_ = self.base_alt if self.base_alt is not None else self.alts_[0]
        names = ([] if self.asc_only else list(self.attrs_))
        if self.asc or self.asc_only:
            names += [f"ASC_{j}" for j in self.alts_ if j != self.base_alt_]
        X = self._design(df)
        g = pd.factorize(df["task"])[0]
        y, av = df["chosen"].to_numpy(float), df["available"].to_numpy(float)

        def f(b):
            lp = self._logp(b, X, g, av)
            p = np.exp(lp)
            sp = np.bincount(g, weights=y)                    # chosen count per task (normally 1)
            grad = -(X.T @ (y - p * sp[g]))
            return -(lp * y).sum(), grad

        r = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
        self.coef_ = pd.Series(r.x, index=names)
        self.converged_ = bool(r.success)
        return self

    def _lp(self, df):
        df = _prep(df).reset_index(drop=True)
        g = pd.factorize(df["task"])[0]
        return df, self._logp(self.coef_.to_numpy(), self._design(df), g, df["available"].to_numpy(float))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self._lp(df)[1])

    def predict(self, df: pd.DataFrame) -> pd.Series:
        d, lp = self._lp(df)
        d = d.assign(_lp=lp)
        return d.loc[d.groupby("task")["_lp"].idxmax()].set_index("task")["alternative"]

    def log_likelihood(self, df: pd.DataFrame) -> float:
        d, lp = self._lp(df)
        avail = d["available"].to_numpy() > 0
        return float((lp[avail] * d["chosen"].to_numpy()[avail]).sum() / d["task"].nunique())

    def accuracy(self, df: pd.DataFrame) -> float:
        d, lp = self._lp(df)
        d = d.assign(_lp=lp)
        return float(d.loc[d.groupby("task")["_lp"].idxmax(), "chosen"].mean())


def null_scores(df: pd.DataFrame) -> dict:
    """Equal-share baseline: accuracy = mean 1/n_alternatives, log-likelihood per task = -mean log n_alternatives."""
    df = _prep(df)
    n = df[df["available"] > 0].groupby("task").size()
    return {"acc": float((1 / n).mean()), "ll_per_task": float(-np.log(n).mean())}


def person_holdout(df: pd.DataFrame, frac: float = 0.2, seed: int = 0):
    """Split rows by person (``frac`` of persons to test), as in the source evaluation. Returns (train, test)."""
    df = _prep(df)
    ids = df["person"].unique()
    rng = np.random.default_rng(seed)
    test_ids = set(rng.choice(ids, size=max(1, int(len(ids) * frac)), replace=False))
    te = df["person"].isin(test_ids).to_numpy()
    return df[~te], df[te]


def evaluate_mnl(df: pd.DataFrame, attrs: Sequence[str] | None = None, scale: Mapping[str, float] | None = None,
                 base_alt=None, frac: float = 0.2, seed: int = 0) -> dict:
    """Source `eval_mnl`: person holdout; MNL, ASC-only and null accuracy / log-likelihood per task on the test
    persons, plus the MNL coefficients."""
    df = _prep(df)
    df = df[df["available"] > 0]
    tr, te = person_holdout(df, frac, seed)
    out = {}
    for label, kw in (("asc_only", dict(asc_only=True)), ("mnl", {})):
        m = MNL(attrs, scale, base_alt, **kw).fit(tr)
        out[label] = {"acc": m.accuracy(te), "ll_per_task": m.log_likelihood(te), "coef": m.coef_.round(3).to_dict()}
    out["null"] = null_scores(te)
    out["n"] = {"test_tasks": int(te["task"].nunique()), "test_persons": int(te["person"].nunique()),
                "train_tasks": int(tr["task"].nunique())}
    return out
