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
(3-seed score ensemble, `n_seeds=3`; +4.66 [+2.04, +7.55] over Markov-2; user-clustered [+1.68, +7.66]);
zero-shot Haiku persona agent 13.67 on an n=300 subsample.
On a user-disjoint split LightGBM 29.57 vs Markov-2 28.11 (+1.46 [-0.14, +3.11], not separable): part of the gain
is per-user history. Literature SFT+RL (Customer-R1) 39.58 on the same split.

Inputs (English names; research aliases accepted):
  steps: one row per step: ``ex`` (step id), ``session``, ``step`` (order in session), ``action`` (key of the
         action taken), optional ``user``, ``page``.  (aliases: session_id, adim, user_id, sayfa, hk)
  candidates: one row per (ex, ``key``) with optional ``y`` (1 = taken; computed from ``action`` if absent) and
         any feature columns.  (alias: k -> key)

Session level (no candidates needed): ``sessions(steps, outcome_key="purchase", prefix_k=3)`` turns the step
table into one row per session (first-k action prefix, outcome label, length, optional query text) for "will this
session convert / be abandoned, judged from its first steps"; ``funnel(steps, outcome_key="purchase")`` gives per
step index how many sessions reach it and the continue / abandon / convert rates.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Sequence, Union

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


def negative_sample(tr: pd.DataFrame, top: int = 60, random: int = 40, seed: int = 0,
                    keep_non_click: bool = True) -> pd.DataFrame:
    """Per step keep positives + the ``top`` candidates by prior (f_page + f_markov + f_user) + ``random`` others.
    When the candidates have a ``kind`` column, non-click candidates (terminate, input, ...) are always kept, as in
    the source (`siralayici.py` `negatif_ornekle`: ``| (r.kind != "click")``)."""
    prior = tr["f_page"] + tr["f_markov"] + tr["f_user"]
    r = tr.assign(_o=prior.to_numpy(), _r=np.random.default_rng(seed).random(len(tr)))
    ro = r.groupby("ex")["_o"].rank(ascending=False, method="first")
    rr = r.groupby("ex")["_r"].rank(method="first")
    m = (r["y"] == 1) | (ro <= top) | (rr <= random)
    if keep_non_click and "kind" in tr.columns:
        m = m | (tr["kind"] != "click")
    return tr[m.to_numpy()]


class NextActionRanker:
    """LightGBM lambdarank over candidates (lightgbm imported lazily).

    fit(train_steps, train_cands, numeric=(), categorical=()) -> builds OOF priors, samples negatives, trains;
    score(test_steps, test_cands) -> (candidates with priors, scores). Feature set = priors + ``numeric`` +
    ``categorical`` caller columns (categories fixed from train).

    ``n_seeds``: train that many models (seeds ``seed .. seed + n_seeds - 1``, each with its own negative sample)
    and average their scores. The published 31.60 is a 3-seed average (``n_seeds=3``); the default 1 is faster."""

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, num_leaves: int = 31,
                 min_child_samples: int = 20, top: int = 60, random: int = 40, n_folds: int = 5, seed: int = 0,
                 use_user_prior: bool = True, n_seeds: int = 1):
        self.params = dict(objective="lambdarank", n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, min_child_samples=min_child_samples, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, lambdarank_truncation_level=20,
                           random_state=seed, verbose=-1)
        self.top, self.random, self.n_folds, self.seed = top, random, n_folds, seed
        self.use_user_prior = use_user_prior
        self.n_seeds = max(int(n_seeds), 1)

    def _cat(self, c):
        c = c.copy()
        for k in self.categorical:
            c[k] = pd.Categorical(c[k].fillna("NA").astype(str), categories=self.cats_[k])
        return c

    def fit(self, train_steps, train_cands, numeric: Sequence[str] = (), categorical: Sequence[str] = ()):
        lgb = require_lightgbm()
        self.train_steps_, self.train_cands_ = train_steps, train_cands
        self.numeric, self.categorical = list(numeric), list(categorical)
        pri = [p for p in PRIORS if self.use_user_prior or p != "f_user"]
        self.features_ = list(self.categorical) + pri + self.numeric
        tr, _ = build_features(train_steps, train_cands, n_folds=self.n_folds)
        self.cats_ = {k: sorted(tr[k].fillna("NA").astype(str).unique().tolist()) for k in self.categorical}
        self.models_ = []
        for i in range(self.n_seeds):
            sd = self.seed + i
            t = self._cat(negative_sample(tr, self.top, self.random, sd).sort_values("ex"))
            grp = t.groupby("ex", sort=True).size().to_numpy()
            m = lgb.LGBMRanker(num_threads=n_threads(len(t)), **{**self.params, "random_state": sd})
            self.models_.append(m.fit(t[self.features_], t["y"], group=grp))
        self.model_ = self.models_[0]
        return self

    def score(self, test_steps, test_cands):
        _, te = build_features(self.train_steps_, self.train_cands_, test_steps, test_cands, n_folds=self.n_folds)
        te = self._cat(te)
        return te, np.mean([m.predict(te[self.features_]) for m in self.models_], axis=0)


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


# ---- session level: step table -> session table / funnel ------------------------------------------------------

OutcomeKey = Union[str, Callable[[pd.DataFrame], Any]]


def _ordered_steps(steps: pd.DataFrame, session_col: str, order_col: str, action_col: str) -> pd.DataFrame:
    al = {k: v for k, v in STEP_ALIASES.items() if v in (session_col, order_col, action_col)}
    s = rename_aliases(steps, al)
    missing = [c for c in (session_col, order_col, action_col) if c not in s.columns]
    if missing:
        raise KeyError(f"steps lacks column(s) {missing}")
    s = s[s[session_col].notna()].sort_values([session_col, order_col], kind="mergesort").reset_index(drop=True)
    s["_pos"] = s.groupby(session_col, sort=False).cumcount()
    return s


def _session_outcomes(s: pd.DataFrame, session_col: str, action_col: str, outcome_key: OutcomeKey,
                      labels: tuple) -> pd.DataFrame:
    """Per session: n_steps, outcome, converted (bool), at (0-based position of the outcome step, -1 = none)."""
    pos_label, neg_label = labels
    g = s.groupby(session_col, sort=False)
    n = g.size()
    if callable(outcome_key):
        lab, conv, at = [], [], []
        for _, grp in g:
            r = outcome_key(grp.drop(columns="_pos"))
            lb, a = r if isinstance(r, tuple) else (r, None)
            c = bool(lb == pos_label)
            if a is None:
                a = len(grp) - 1 if c else -1          # converted without a position: the last step is the outcome
            a = int(a)
            if not -1 <= a < len(grp):
                raise ValueError(f"outcome position {a} outside session of {len(grp)} steps")
            if c and a < 0:                            # a conversion must sit on a step, else the prefix holds it
                raise ValueError(f"callable returned {pos_label!r} with position -1; give its step or None")
            lab.append(lb)
            conv.append(c)
            at.append(a)
        o = pd.DataFrame({"outcome": lab, "converted": conv, "at": at}, index=n.index)
    else:
        if outcome_key in s.columns:                   # a per-step flag column (truthy = conversion step)
            hit = s[outcome_key].fillna(0).astype(bool)
            share = hit.groupby(s[session_col], sort=False).agg(["any", "all"])
            if share["any"].any() and share["all"][share["any"]].all() and (n[share["any"]] > 1).any():
                # a session-level label repeated on every row: the first step would count as the conversion,
                # every converted prefix would be empty and reached_k would drop them all
                raise ValueError(f"{outcome_key!r} is truthy on every step of each flagged session: a session "
                                 "label, not a per-step flag. Use outcome_key=lambda g: "
                                 f"'{pos_label}' if g[{outcome_key!r}].iloc[0] else '{neg_label}' (last step = outcome)"
                                 " or the conversion action value")
        else:                                          # an action value (e.g. "purchase")
            hit = s[action_col] == outcome_key
            if len(s) and not hit.any():
                top = s[action_col].value_counts().index[:8].tolist()
                warnings.warn(f"no step has {action_col} == {outcome_key!r}: every session is {neg_label!r} "
                              f"(most common actions: {top})", stacklevel=3)
        first = s["_pos"].where(hit).groupby(s[session_col], sort=False).min().reindex(n.index)
        conv = first.notna()
        o = pd.DataFrame({"outcome": np.where(conv, pos_label, neg_label), "converted": conv.to_numpy(),
                          "at": first.fillna(-1).astype(int).to_numpy()}, index=n.index)
    o["n_steps"] = n.to_numpy()
    return o


def sessions(steps: pd.DataFrame, *, outcome_key: OutcomeKey, prefix_k: int = 3, session_col: str = "session",
             order_col: str = "step", action_col: str = "action", text_col: str | None = None,
             labels: tuple = ("purchase", "abandon"), sep: str = " > ", text_sep: str = " | ") -> pd.DataFrame:
    """One row per session from a step table, for predicting the session outcome from its first steps.

    ``outcome_key``:
      - str naming a column of ``steps``: per-step flag, the first truthy step is the conversion step (a session
        label repeated on every row raises: it would make every converted prefix empty; use a callable);
      - other str: the action value that marks conversion (e.g. ``"purchase"``), first occurrence counts (warns
        when no step matches, e.g. a case typo);
      - callable(session_steps_ordered) -> label, or (label, position) with the 0-based position of the outcome
        step (``-1`` / ``None`` = no outcome step; a ``labels[0]`` label without a position means the last step).
    Sessions with a conversion step get ``labels[0]``, the rest ``labels[1]`` (callables return their own label).

    Columns: ``session_col``, ``prefix`` (list of the first actions), ``prefix_text`` (joined with ``sep``),
    ``prefix_len``, ``reached_k`` (prefix_len == prefix_k), ``n_steps``, ``outcome``, ``converted`` (0/1),
    ``outcome_step`` (0-based position, <NA> if none) and, with ``text_col``, ``text``: the distinct non-empty
    texts (e.g. search queries) of the prefix steps in order, joined with ``text_sep``.

    Leakage: the prefix is the first ``prefix_k`` steps **cut before the outcome step**, so it never contains the
    outcome action or anything after it; ``text`` uses the same steps. ``n_steps``, ``outcome_step`` and
    ``outcome`` are computed from the full session: they are labels / descriptors, not features. A prefix shorter
    than ``prefix_k`` itself reveals that the session ended (or converted) early, so for an honest "from the first
    k steps" task keep ``df[df.reached_k]`` (the sessions still open after k steps) or report the two groups apart.
    """
    if prefix_k < 1:
        raise ValueError("prefix_k must be >= 1")
    s = _ordered_steps(steps, session_col, order_col, action_col)
    o = _session_outcomes(s, session_col, action_col, outcome_key, labels)
    cut = pd.Series(np.where(o["at"] >= 0, np.minimum(prefix_k, o["at"]), prefix_k), index=o.index)
    pre = s[s["_pos"].to_numpy() < cut.reindex(s[session_col]).to_numpy()]
    gp = pre.groupby(session_col, sort=False)
    acts = gp[action_col].agg(list).reindex(o.index)
    acts = [a if isinstance(a, list) else [] for a in acts]
    out = pd.DataFrame({session_col: o.index.to_numpy(), "prefix": acts,
                        "prefix_text": [sep.join(map(str, a)) for a in acts],
                        "prefix_len": [len(a) for a in acts]})
    out["reached_k"] = out["prefix_len"] == prefix_k
    out["n_steps"] = o["n_steps"].to_numpy()
    out["outcome"] = o["outcome"].to_numpy()
    out["converted"] = o["converted"].astype(int).to_numpy()
    out["outcome_step"] = pd.array([a if a >= 0 else pd.NA for a in o["at"]], dtype="Int64")
    if text_col is not None:
        def _join(v):
            seen, keep = set(), []
            for t in v:
                if isinstance(t, str) and t.strip() and t not in seen:
                    seen.add(t)
                    keep.append(t)
            return text_sep.join(keep)
        tx = gp[text_col].agg(_join).reindex(o.index).fillna("")
        out["text"] = tx.to_numpy()
    return out


def funnel(steps: pd.DataFrame, *, outcome_key: OutcomeKey, session_col: str = "session", order_col: str = "step",
           action_col: str = "action", labels: tuple = ("purchase", "abandon"),
           max_steps: int | None = None) -> pd.DataFrame:
    """Per-step funnel: row i = sessions that reach their i-th step (0-based ``step_index``) still unconverted.

    Of the ``reached`` sessions at step i, ``converted`` have their conversion step there, ``abandoned`` end there
    without converting and ``continued`` go on to step i + 1 (so ``continued[i] == reached[i + 1]``). A session
    leaves the funnel at its conversion step; steps after it are ignored. Rates are over ``reached``;
    ``reach_rate`` is over all sessions. ``outcome_key`` / ``labels`` as in ``sessions``. ``max_steps`` keeps the
    first rows only (the counts are unchanged)."""
    s = _ordered_steps(steps, session_col, order_col, action_col)
    o = _session_outcomes(s, session_col, action_col, outcome_key, labels)
    conv = o["converted"].to_numpy(bool)
    at, n = o["at"].to_numpy(), o["n_steps"].to_numpy()
    depth = np.where(conv, at + 1, n)                  # steps spent in the funnel
    m = int(depth.max()) if len(depth) else 0
    idx = np.arange(m)
    reached = (depth[None, :] > idx[:, None]).sum(axis=1)
    converted = np.bincount(at[conv], minlength=m)[:m]
    abandoned = np.bincount(n[~conv] - 1, minlength=m)[:m]
    f = pd.DataFrame({"step_index": idx, "reached": reached, "continued": reached - converted - abandoned,
                      "abandoned": abandoned, "converted": converted})
    den = np.maximum(f["reached"].to_numpy(), 1)
    f["reach_rate"] = f["reached"] / max(len(o), 1)
    f["continue_rate"] = f["continued"] / den
    f["abandon_rate"] = f["abandoned"] / den
    f["convert_rate"] = f["converted"] / den
    return f if max_steps is None else f.head(max_steps).reset_index(drop=True)
