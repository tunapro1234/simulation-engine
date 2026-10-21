"""Next-action prediction as candidate ranking: at each step every possible action is a candidate row, the model
scores candidates and the top one is the prediction.

Port of `uretim/web/kod/yontem/siralayici.py` (OPeRA web shopping sessions): transition priors (global and
page-conditional frequency with +0.01 / +1 smoothing, Markov-1, Markov-2, per-user frequency) computed out-of-fold
on train with GroupKFold by session (5 folds) and from all train for test; Markov-2 baseline score
``f_markov2 + 1e-2 f_markov + 1e-4 f_page``; LightGBM lambdarank on priors + caller features after negative sampling
(positives + top 60 by prior + 40 random per step). The dataset-specific features of the source (page element
structure, click-type priors, text overlap) are not ported: pass your own as ``numeric`` / ``categorical`` columns.

Measured (uretim/web/RAPOR.md, OPeRA official test, 902 steps, exact next action %): always-terminate 4.43,
repeat previous 13.97, Markov-2 + backoff 26.94 [21.42, 31.72], LightGBM lambdarank 31.60 [26.81, 35.90]
(+4.66 [+2.04, +7.55] over Markov-2; user-clustered [+1.68, +7.66]); zero-shot Haiku persona agent 13.67.
On a user-disjoint split LightGBM 29.57 vs Markov-2 28.11 (+1.46 [-0.14, +3.11], not separable): part of the gain
is per-user history. Literature SFT+RL (Customer-R1) 39.58 on the same split.

Inputs (English names; research aliases accepted):
  steps: one row per step: ``ex`` (step id), ``session``, ``step`` (order in session), ``action`` (key of the
         action taken), optional ``user``, ``page``.  (aliases: session_id, adim, user_id, sayfa, hk)
  candidates: one row per (ex, ``key``) with optional ``y`` (1 = taken; computed from ``action`` if absent) and
         any feature columns.  (alias: k -> key)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._util import n_threads, rename_aliases, require_lightgbm

STEP_ALIASES = {"session_id": "session", "adim": "step", "user_id": "user", "sayfa": "page", "hk": "action"}
CAND_ALIASES = {"k": "key"}
PRIORS = ("f_global", "f_page", "f_markov", "f_markov2", "f_user")
ALPHA = 1.0


def add_history(steps: pd.DataFrame) -> pd.DataFrame:
    """Previous and second-previous action key within the session (``prev_key``, ``prev2_key``; "START" at the
    beginning). Missing ``user`` / ``page`` columns are filled with a single value."""
    s = rename_aliases(steps, STEP_ALIASES).copy()
    for c, v in (("user", "_"), ("page", "_")):
        if c not in s.columns:
            s[c] = v
    s = s.sort_values(["session", "step"])
    g = s.groupby("session")["action"]
    s["prev_key"] = g.shift(1).fillna("START")
    s["prev2_key"] = g.shift(2).fillna("START")
    return s


def transition_priors(stat_steps: pd.DataFrame, stat_cands: pd.DataFrame) -> dict:
    """Prior tables from statistics steps (with history) and their available candidate keys (ex, key)."""
    a = stat_steps
    hit = a["action"].value_counts().rename("h")
    avail = stat_cands.groupby("key").size().rename("m")
    g = pd.concat([avail, hit], axis=1).fillna(0)
    p_global = ((g.h + 0.01) / (g.m + ALPHA)).rename("f_global").rename_axis("key").reset_index()
    sa = stat_cands.merge(a[["ex", "page"]], on="ex")
    g = pd.concat([sa.groupby(["page", "key"]).size().rename("m"),
                   a.rename(columns={"action": "key"}).groupby(["page", "key"]).size().rename("h")], axis=1).fillna(0)
    p_page = ((g.h + 0.01) / (g.m + ALPHA)).rename("f_page").reset_index()
    b = a.rename(columns={"action": "key"})
    markov = (b.groupby(["prev_key", "key"]).size() / b.groupby("prev_key").size()).rename("f_markov").reset_index()
    markov2 = (b.groupby(["prev2_key", "prev_key", "key"]).size() / b.groupby(["prev2_key", "prev_key"]).size()
               ).rename("f_markov2").reset_index()
    user = (b.groupby(["user", "key"]).size() / b.groupby("user").size()).rename("f_user").reset_index()
    return {"p_global": p_global, "p_page": p_page, "markov": markov, "markov2": markov2, "user": user}


def attach_priors(cands: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """Merge prior features onto candidates that already carry the step's ``page``, ``user``, ``prev_key``,
    ``prev2_key`` columns; unseen combinations get 0."""
    n = len(cands)
    c = cands.merge(priors["p_global"], on="key", how="left")
    c = c.merge(priors["p_page"], on=["page", "key"], how="left")
    c = c.merge(priors["markov"], on=["prev_key", "key"], how="left")
    c = c.merge(priors["markov2"], on=["prev2_key", "prev_key", "key"], how="left")
    c = c.merge(priors["user"], on=["user", "key"], how="left")
    assert len(c) == n
    c[list(PRIORS)] = c[list(PRIORS)].fillna(0.0)
    return c


def _join_steps(cands, steps):
    c = rename_aliases(cands, CAND_ALIASES)
    drop = [x for x in ("session", "step", "user", "page", "prev_key", "prev2_key", "action") if x in c.columns]
    c = c.drop(columns=drop).merge(steps[["ex", "session", "step", "user", "page", "prev_key", "prev2_key",
                                          "action"]], on="ex", how="left")
    if "y" not in c.columns:
        c["y"] = (c["key"] == c["action"]).astype(int)
    return c


def build_features(train_steps, train_cands, test_steps=None, test_cands=None, n_folds: int = 5):
    """Train candidates with out-of-fold priors (GroupKFold by session) and, if given, test candidates with priors
    from all train. Returns (train, test) candidate tables."""
    from sklearn.model_selection import GroupKFold
    ts = add_history(train_steps)
    tr = _join_steps(train_cands, ts)
    parts = []
    folds = min(n_folds, ts["session"].nunique())
    for f_tr, f_va in GroupKFold(n_splits=folds).split(ts, groups=ts["session"].to_numpy()):
        st = ts.iloc[f_tr]
        va = set(ts.iloc[f_va]["ex"])
        pri = transition_priors(st, tr[tr["ex"].isin(set(st["ex"]))][["ex", "key"]].drop_duplicates())
        parts.append(attach_priors(tr[tr["ex"].isin(va)].copy(), pri))
    tr_out = pd.concat(parts, ignore_index=True)
    te_out = None
    if test_steps is not None:
        es = add_history(test_steps)
        te = _join_steps(test_cands, es)
        te_out = attach_priors(te, transition_priors(ts, tr[["ex", "key"]].drop_duplicates()))
    return tr_out, te_out


def markov2_score(c: pd.DataFrame) -> np.ndarray:
    """Markov-2 with backoff: f_markov2 + 1e-2 f_markov + 1e-4 f_page."""
    return (c["f_markov2"] + 1e-2 * c["f_markov"] + 1e-4 * c["f_page"]).to_numpy()


class Markov2Baseline:
    """fit(train_steps, train_cands) stores the train statistics; score(test_steps, test_cands) -> (candidates with
    priors, scores)."""

    def fit(self, train_steps, train_cands):
        self.ts_ = add_history(train_steps)
        self.tr_ = _join_steps(train_cands, self.ts_)
        self.priors_ = transition_priors(self.ts_, self.tr_[["ex", "key"]].drop_duplicates())
        return self

    def score(self, test_steps, test_cands):
        te = attach_priors(_join_steps(test_cands, add_history(test_steps)), self.priors_)
        return te, markov2_score(te)


def negative_sample(tr: pd.DataFrame, top: int = 60, random: int = 40, seed: int = 0) -> pd.DataFrame:
    """Per step keep positives + the ``top`` candidates by prior (f_page + f_markov + f_user) + ``random`` others."""
    prior = tr["f_page"] + tr["f_markov"] + tr["f_user"]
    r = tr.assign(_o=prior.to_numpy(), _r=np.random.default_rng(seed).random(len(tr)))
    ro = r.groupby("ex")["_o"].rank(ascending=False, method="first")
    rr = r.groupby("ex")["_r"].rank(method="first")
    return tr[((r["y"] == 1) | (ro <= top) | (rr <= random)).to_numpy()]


class NextActionRanker:
    """LightGBM lambdarank over candidates (lightgbm imported lazily).

    fit(train_steps, train_cands, numeric=(), categorical=()) -> builds OOF priors, samples negatives, trains;
    score(test_steps, test_cands) -> (candidates with priors, scores). Feature set = priors + ``numeric`` +
    ``categorical`` caller columns (categories fixed from train)."""

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, num_leaves: int = 31,
                 min_child_samples: int = 20, top: int = 60, random: int = 40, n_folds: int = 5, seed: int = 0,
                 use_user_prior: bool = True):
        self.params = dict(objective="lambdarank", n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, min_child_samples=min_child_samples, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, lambdarank_truncation_level=20,
                           random_state=seed, verbose=-1)
        self.top, self.random, self.n_folds, self.seed = top, random, n_folds, seed
        self.use_user_prior = use_user_prior

    def _cat(self, c):
        c = c.copy()
        for k in self.categorical:
            c[k] = pd.Categorical(c[k].astype(str).fillna("NA"), categories=self.cats_[k])
        return c

    def fit(self, train_steps, train_cands, numeric: Sequence[str] = (), categorical: Sequence[str] = ()):
        lgb = require_lightgbm()
        self.train_steps_, self.train_cands_ = train_steps, train_cands
        self.numeric, self.categorical = list(numeric), list(categorical)
        pri = [p for p in PRIORS if self.use_user_prior or p != "f_user"]
        self.features_ = list(self.categorical) + pri + self.numeric
        tr, _ = build_features(train_steps, train_cands, n_folds=self.n_folds)
        self.cats_ = {k: sorted(tr[k].astype(str).fillna("NA").unique().tolist()) for k in self.categorical}
        tr = self._cat(negative_sample(tr, self.top, self.random, self.seed).sort_values("ex"))
        grp = tr.groupby("ex", sort=True).size().to_numpy()
        self.model_ = lgb.LGBMRanker(num_threads=n_threads(len(tr)), **self.params)
        self.model_.fit(tr[self.features_], tr["y"], group=grp)
        return self

    def score(self, test_steps, test_cands):
        _, te = build_features(self.train_steps_, self.train_cands_, test_steps, test_cands, n_folds=self.n_folds)
        te = self._cat(te)
        return te, self.model_.predict(te[self.features_])


def top1(cands: pd.DataFrame, scores) -> pd.DataFrame:
    """Highest-scoring candidate per step: DataFrame with ex, key."""
    c = cands.assign(_s=np.asarray(scores, float)).sort_values(["ex", "_s"], ascending=[True, False])
    return c.groupby("ex").head(1)[["ex", "key"]].reset_index(drop=True)


def hit_at_k(cands: pd.DataFrame, scores, ks: Sequence[int] = (1, 5, 10)) -> dict:
    """Share of steps whose taken action ranks within the top k (steps without a positive candidate count as
    misses)."""
    c = cands.assign(_s=np.asarray(scores, float))
    c["_r"] = c.groupby("ex")["_s"].rank(ascending=False, method="first")
    r = c[c["y"] == 1].groupby("ex")["_r"].min()
    n = c["ex"].nunique()
    return {f"hit@{k}": float((r <= k).sum() / n) for k in ks}
