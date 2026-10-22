"""Group transfer: predict person-level answers in groups never seen in training, plus an out-of-fold helper.

Task shape ("group transfer"): survey answers exist for people in some groups (e.g. countries); predict the answers
of people in NEW groups from their demographics. Persistence does not apply (no earlier wave of the same people) and
a random-row CV leaks the group level, so the honest check is group-held-out CV (``oof_predict(..., group=...)``).

``GroupTransfer`` ports the two-stage model of the D2 rehearsal (``uretim/prova/d2/kod/cv.py`` ``iki_asama``,
used by ``cv_biz3.py`` and ``tahmin.py``):

A. group level: ridge (``StandardScaler`` + ``RidgeCV``, alphas logspace(-1, 3, 20)) of the training groups' mean
   target (logit of the clipped rate for binary targets) on group features: the group's composition (means of the
   person features over the group's rows, computed from inputs only, so it is available for a new group) and,
   optionally, extra group-level columns such as an LLM group prior.
B. person deviation: a person-level regressor (LightGBM, as in the port, or ridge) of ``y - observed group mean``
   on the individual features.

prediction = A(group features) + B(person features); for binary targets ``expit(A) + B``, clipped to [0, 1].

Measured (D2 rehearsal, ``uretim/prova/d2/NOTLAR.md``, 43 training countries, 5-fold country-held-out CV, person
Spearman on a 0-100 policy score): person-only LightGBM 0.197, LightGBM + composition columns 0.234, two-stage
without LLM 0.257, two-stage + LLM country prior 0.295 (extended prior 0.311), oracle country level 0.344-0.346.
"""
from __future__ import annotations

import copy
import inspect
from typing import Sequence

import numpy as np
import pandas as pd

from ._util import lightgbm_available, logit, sigmoid
from .baselines import LightGBMBaseline, LogisticBaseline, infer_kind

RIDGE_ALPHAS = tuple(np.logspace(-1, 3, 20))     # cv.py iki_asama: RidgeCV(alphas=np.logspace(-1, 3, 20))
RATE_CLIP = (0.02, 0.98)                         # cv.py iki_asama: cm.clip(0.02, 0.98) before logit
MAX_CATEGORIES = 20


# ------------------------------------------------------------------------------------------------ out-of-fold

def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _weight_kw(m, w: np.ndarray) -> dict:
    """``fit`` keyword(s) that pass sample weights to ``m``: ``sample_weight=`` when ``m.fit`` takes it; for an sklearn
    ``Pipeline`` (nested too) ``<last step>__sample_weight=`` (the last step's fit must take it). Else TypeError."""
    steps = getattr(m, "steps", None)
    if isinstance(steps, list) and steps and all(isinstance(t, tuple) and len(t) == 2 for t in steps):
        name, last = steps[-1]
        if last is None or last == "passthrough":
            raise TypeError("oof_predict(weight=...): the pipeline's last step is 'passthrough'")
        return {f"{name}__{k}": v for k, v in _weight_kw(last, w).items()}
    if _accepts(m.fit, "sample_weight"):
        return {"sample_weight": w}
    raise TypeError(f"oof_predict(weight=...): {type(m).__name__}.fit takes no sample_weight. Use a model whose fit "
                    "accepts sample_weight (sklearn regressors, methods.tabular TabularRidge / TabularGBDT / "
                    "GroupStat; LogisticBaseline / LightGBMBaseline do not yet) or drop weight=")


def _fresh(model):
    """Unfitted copy: a factory is called, sklearn estimators are cloned, anything else is deep-copied."""
    if not hasattr(model, "fit") and callable(model):
        return model()
    try:
        from sklearn.base import clone
        return clone(model)
    except Exception:
        return copy.deepcopy(model)


def _as_1d(p, method: str) -> np.ndarray:
    p = np.asarray(p)
    if p.ndim == 2 and p.shape[1] == 2 and method == "predict_proba":
        return p[:, 1].astype(float)
    if p.ndim == 2 and p.shape[1] == 1:
        return p[:, 0]
    if p.ndim != 1:
        raise ValueError(f"model.{method} returned shape {p.shape}; oof_predict needs one value per row")
    return p


def _fold_splits(folds, df: pd.DataFrame, pos: np.ndarray):
    """Precomputed folds -> (list of (train, test) positions into ``df.iloc[pos]``, fold labels).

    ``folds``: a column name of ``df``, a Series (aligned by index), a 1-D array of fold ids (one per row of ``df``),
    or an iterable of ``(train_idx, test_idx)`` positional index pairs into ``df`` (what ``sklearn`` ``split``
    yields). Fold id ``-1`` = always in training, never predicted (sklearn ``PredefinedSplit``); NaN = row left out.
    """
    n = len(df)
    inv = np.full(n, -1, dtype=int)
    inv[pos] = np.arange(len(pos))
    if isinstance(folds, str):
        if folds not in df.columns:
            raise KeyError(f"folds column {folds!r} not in df")
        folds = df[folds]
    if isinstance(folds, pd.Series):
        if not folds.index.equals(df.index):
            if not df.index.isin(folds.index).all():
                raise ValueError("folds Series must have a fold id for every row of df (index alignment); pass "
                                 "folds.to_numpy() to align by position")
            folds = folds.reindex(df.index)
        ids = folds.to_numpy()
    elif isinstance(folds, np.ndarray) and folds.ndim == 1 and folds.dtype != object:
        ids = folds
    else:
        lst = list(folds)
        pairish = bool(lst) and all(isinstance(t, (tuple, list)) and len(t) == 2 and np.ndim(t[0]) == 1
                                   for t in lst)
        if not pairish:
            ids = np.asarray(lst)
        else:
            splits, seen = [], np.zeros(n, dtype=int)
            for k, (tr, te) in enumerate(lst):
                tr, te = np.asarray(tr, dtype=int), np.asarray(te, dtype=int)
                if (tr.size and (tr.min() < -n or tr.max() >= n)) or (te.size and (te.min() < -n or te.max() >= n)):
                    raise IndexError(f"fold {k}: positional index out of range for df with {n} rows")
                tr, te = np.mod(tr, n), np.mod(te, n)
                if np.intersect1d(tr, te).size:
                    raise ValueError(f"fold {k}: a row is in both its train and test indices")
                seen[te] += 1
                a, b = inv[tr], inv[te]
                splits.append((a[a >= 0], b[b >= 0]))
            if (seen > 1).any():
                raise ValueError("a row is in the test indices of more than one fold")
            return splits, list(range(len(splits)))
    if len(ids) != n:
        raise ValueError(f"folds has {len(ids)} fold ids, df has {n} rows")
    ids = pd.Series(ids).iloc[pos].reset_index(drop=True)
    keep = ids.notna().to_numpy()
    always = (ids == -1).to_numpy() & keep
    labels = sorted(pd.unique(ids[keep & ~always]), key=lambda v: (str(type(v)), v))
    splits = []
    for lab in labels:
        test = (ids == lab).to_numpy() & keep
        splits.append((np.flatnonzero(keep & ~test), np.flatnonzero(test)))
    return splits, labels


def oof_predict(model, df: pd.DataFrame, *, features: Sequence[str], target: str, group: str | None = None,
                n_splits: int | None = None, seed: int = 0, method: str = "predict", folds=None,
                weight=None) -> pd.Series:
    """Out-of-fold predictions: every row is predicted by a model that did not see its fold.

    ``group`` given -> ``GroupKFold(n_splits)`` on ``df[group]`` (no group is in both train and test of a fold;
    deterministic, as in the D2 port, so ``seed`` is unused); otherwise ``KFold(n_splits, shuffle=True,
    random_state=seed)``. ``n_splits`` defaults to 5. ``model`` is any object with ``fit(X, y)`` / ``predict(X)``
    (sklearn estimator, a simengine baseline, ``GroupTransfer``) or a zero-argument factory; it is cloned per fold. If
    ``fit`` / ``predict`` accept a ``groups`` keyword (``GroupTransfer`` does) the fold's group labels are passed.
    ``method="predict_proba"`` takes the positive-class column of a two-column output. Rows with a missing target
    are neither trained on nor predicted (NaN). Returns a float Series aligned with ``df.index``; ``.attrs["fold"]``
    holds each row's fold number (-1 = not predicted) and ``.attrs["fold_ids"]`` the fold labels in that order.

    ``folds``: precomputed folds, used EXACTLY instead of GroupKFold / KFold: a column name of ``df``, a Series of
    fold ids aligned with ``df.index``, a 1-D array of fold ids (one per row), or an iterable of ``(train_idx,
    test_idx)`` positional index pairs. Fold id ``-1`` = always trained on, never predicted (sklearn
    ``PredefinedSplit``); NaN = left out. ``group`` still goes to models that take ``groups=`` (``GroupTransfer``);
    a ``UserWarning`` is raised when a group falls in both train and test of a fold. ``n_splits`` given together
    with ``folds`` must equal the number of folds (else ``ValueError``). Why: features rebuilt inside folds (SVD
    taste vectors, member statistics; see ``item_holdout_features``) must be scored with the SAME folds, or rows
    whose features were built with the held-out items still present end up in training and dev is optimistic.

    ``weight``: per-row training weights (a column name of ``df``, or an array / Series aligned with ``df``), passed
    to ``fit`` as ``sample_weight=`` for each fold's training rows (e.g. ``1/(se^2 + tau^2)`` when row noise
    varies). sklearn pipelines get it routed to their last step (``<step>__sample_weight``); a model whose fit
    takes no ``sample_weight`` raises ``TypeError``. Missing or negative weights on rows with a target raise
    ``ValueError``. Predictions are not weighted; weight the metric separately.

    Group-level scores of pooled group-held-out predictions are biased NEGATIVE under the null (pessimistic, not
    leakage): a held-out group's prediction leans on the mean of the other groups, which is lower when the held-out
    groups' own mean is higher. With a few dozen groups and no real group signal in the features, the group-level
    Spearman of the pooled OOF predictions is clearly below 0, not 0. Compare against a baseline cross-fitted the
    same way (e.g. ``GroupTransfer(person_model=None, composition=())``), not against 0.
    """
    import warnings
    from sklearn.model_selection import GroupKFold, KFold
    features = list(features)
    missing = [c for c in features + [target] + ([group] if group else []) if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in df: {missing}")
    ok = df[target].notna().to_numpy()
    if group is not None:
        ok &= df[group].notna().to_numpy()
    pos = np.flatnonzero(ok)
    d = df.iloc[pos]
    X, y = d[features], d[target].to_numpy()
    g = d[group].to_numpy() if group is not None else None
    w = None
    if weight is not None:
        if isinstance(weight, str):
            if weight not in df.columns:
                raise KeyError(f"weight column {weight!r} not in df")
            wv = df[weight]
        elif isinstance(weight, pd.Series):
            if not df.index.isin(weight.index).all():
                raise ValueError("weight Series must have a value for every row of df (index alignment)")
            wv = weight.reindex(df.index) if not weight.index.equals(df.index) else weight
        else:
            wv = np.asarray(weight)
            if wv.ndim != 1 or len(wv) != len(df):
                raise ValueError(f"weight has shape {wv.shape}; need one value per row of df ({len(df)})")
        w = pd.to_numeric(pd.Series(np.asarray(wv)), errors="coerce").to_numpy(float)[pos]
        bad = ~np.isfinite(w) | (w < 0)
        if bad.any():
            raise ValueError(f"weight: {int(bad.sum())} rows with a target have a missing, infinite or negative "
                             "weight")
    if folds is not None:
        splits, labels = _fold_splits(folds, df, pos)
        if n_splits is not None and n_splits != len(splits):
            raise ValueError(f"n_splits={n_splits} conflicts with the {len(splits)} precomputed folds; pass only "
                             "folds= (the split is taken from it)")
        if not splits:
            raise ValueError("folds= gives no fold with a test row")
        if g is not None:
            leaky = [labels[k] for k, (a, b) in enumerate(splits) if np.intersect1d(g[a], g[b]).size]
            if leaky:
                warnings.warn(f"oof_predict: groups of {group!r} fall in both train and test of folds {leaky[:5]}; "
                              "the group level leaks into training for those rows (folds are used as given)",
                              UserWarning, stacklevel=2)
    else:
        n_splits = 5 if n_splits is None else n_splits
        labels = list(range(n_splits))
        if g is not None:
            n_groups = len(pd.unique(g))
            if n_groups < n_splits:
                raise ValueError(f"group-held-out CV needs at least n_splits={n_splits} groups, got {n_groups}")
            splits = GroupKFold(n_splits).split(X, y, g)
        else:
            splits = KFold(n_splits, shuffle=True, random_state=seed).split(X)
    out = np.full(len(df), np.nan)
    fold = np.full(len(df), -1, dtype=int)
    for k, (a, b) in enumerate(splits):
        if len(b) == 0:
            continue
        if len(a) == 0:
            raise ValueError(f"fold {labels[k]!r} has no training rows")
        m = _fresh(model)
        fkw = {"groups": g[a]} if g is not None and _accepts(m.fit, "groups") else {}
        if w is not None:
            fkw.update(_weight_kw(m, w[a]))
        m.fit(X.iloc[a], y[a], **fkw)
        pred = getattr(m, method)
        pkw = {"groups": g[b]} if g is not None and _accepts(pred, "groups") else {}
        out[pos[b]] = _as_1d(pred(X.iloc[b], **pkw), method)
        fold[pos[b]] = k
    s = pd.Series(out, index=df.index, name=f"oof_{target}")
    s.attrs["fold"] = fold
    s.attrs["fold_ids"] = list(labels)
    return s


# ------------------------------------------------------------------------------------------------ group transfer

GF_MIN_MATCH = 0.5      # below this share of groups found in group_features: warn (labels probably differ)


def _align_group_features(gf: pd.DataFrame, labels: pd.Index) -> pd.DataFrame:
    """group_features re-keyed so its index matches the group labels. Exact labels first; when fewer than
    GF_MIN_MATCH of the groups match and string labels match more (int groups vs a str-indexed prior, or the
    reverse), the string match is used and a warning says so. Still mostly unmatched -> warning (the unmatched
    groups get the training mean, i.e. the prior has no effect for them)."""
    import warnings
    n = len(labels)
    if n == 0:
        return gf
    hit = float(labels.isin(gf.index).mean())
    if hit >= GF_MIN_MATCH:
        return gf
    s_gf = gf.copy()
    s_gf.index = s_gf.index.map(str)
    s_lab = labels.map(str)
    s_hit = float(s_lab.isin(s_gf.index).mean())
    if s_hit > hit and not s_gf.index.duplicated().any():
        warnings.warn(f"GroupTransfer: group_features index matched {hit:.0%} of the group labels exactly and "
                      f"{s_hit:.0%} as strings; matching by string labels (index dtype {gf.index.dtype} vs group "
                      f"labels {labels.dtype})", UserWarning, stacklevel=4)
        return s_gf.rename(index=dict(zip(s_lab, labels))).reindex(labels)
    warnings.warn(f"GroupTransfer: only {hit:.0%} of the {n} groups have a row in group_features (e.g. label "
                  f"{labels[0]!r} vs index {list(gf.index[:3])}); unmatched groups get the training mean, so the "
                  "group prior has little or no effect", UserWarning, stacklevel=4)
    return gf

class GroupTransfer:
    """Two-stage group-transfer model: group level from group features (ridge) + person deviation (see module doc).

    Parameters
    - ``group_col``: name of the group column inside ``X`` (used when ``groups=`` is not passed); it is never used as
      a person feature.
    - ``composition``: columns of ``X`` whose per-group means (numeric) or category shares (non-numeric, top
      ``MAX_CATEGORIES`` training categories) are the group's composition features. ``None`` = all numeric person
      features; ``()`` = no composition (group features only). For numeric composition columns with missing values
      in training, the group's missing rate is added too (the port's ``c_pol_na``). Group size is not used (the port
      dropped ``c_n``).
    - ``group_features``: optional DataFrame indexed by group label with extra group-level columns, e.g. an LLM group
      prior (``fit`` / ``predict`` also take it, overriding this one). Missing groups / values are filled with the
      training-group means.
    - ``person_model``: ``"auto"`` (LightGBM as in the port when installed, else ridge), ``"lightgbm"``, ``"ridge"``
      (``LogisticBaseline(kind="continuous")``), ``None`` (group level only), or any regressor with fit/predict.
    - ``kind``: ``"continuous"``, ``"binary"`` or ``"auto"``. ``alphas``: ridge grid of stage A. ``clip``: rate clip
      before logit for binary targets.

    Attributes after fit: ``group_means_`` (observed training-group means), ``level_model_`` (stage A pipeline),
    ``group_feature_names_``, ``person_model_``.
    """

    def __init__(self, group_col: str | None = None, *, composition: Sequence[str] | None = None,
                 group_features: pd.DataFrame | None = None, person_model="auto", kind: str = "auto",
                 alphas: Sequence[float] = RIDGE_ALPHAS, clip: tuple[float, float] = RATE_CLIP, seed: int = 0):
        self.group_col, self.composition, self.group_features = group_col, composition, group_features
        self.person_model, self.kind, self.alphas, self.clip, self.seed = person_model, kind, alphas, clip, seed

    # -- helpers
    def _groups(self, X: pd.DataFrame, groups):
        if groups is not None:
            g = np.asarray(groups)
            if len(g) != len(X):
                raise ValueError("groups must have one label per row of X")
            return g
        if self.group_col is not None and self.group_col in X.columns:
            return X[self.group_col].to_numpy()
        raise ValueError("GroupTransfer needs groups=... or a group_col present in X")

    def _person_X(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[self.group_col]) if self.group_col in X.columns else X

    def _composition(self, Xp: pd.DataFrame, g) -> pd.DataFrame:
        parts = []
        if self.num_comp_:
            num = Xp[self.num_comp_].apply(pd.to_numeric, errors="coerce")
            parts.append(num.groupby(g).mean().add_prefix("c_"))
            if self.na_comp_:
                parts.append(num[self.na_comp_].isna().groupby(g).mean().add_prefix("c_na_"))
        for c, cats in self.cat_comp_.items():
            v = Xp[c].astype(str)
            sh = pd.DataFrame({f"c_{c}={k}": (v == k).to_numpy(float) for k in cats}, index=Xp.index)
            parts.append(sh.groupby(g).mean())
        if not parts:
            return pd.DataFrame(index=pd.Index(pd.unique(g)))
        return pd.concat(parts, axis=1)

    def _group_frame(self, Xp: pd.DataFrame, g, group_features) -> pd.DataFrame:
        F = self._composition(Xp, g)
        if group_features is not None:
            gf = _align_group_features(pd.DataFrame(group_features).add_prefix("g_"), F.index)
            F = F.join(gf.reindex(F.index))
        return F

    def _make_person_model(self):
        pm = self.person_model
        if pm == "auto":
            pm = "lightgbm" if lightgbm_available() else "ridge"
        if pm == "lightgbm":
            return LightGBMBaseline(kind="continuous", seed=self.seed)
        if pm == "ridge":
            return LogisticBaseline(kind="continuous")
        if pm is None:
            return None
        return _fresh(pm)

    # -- API
    def fit(self, X: pd.DataFrame, y, groups=None, group_features: pd.DataFrame | None = None):
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X = pd.DataFrame(X)
        y = np.asarray(y, float)
        g = self._groups(X, groups)
        self.kind_ = infer_kind(y) if self.kind in (None, "auto") else self.kind
        if self.kind_ not in ("continuous", "binary"):
            raise ValueError(f"GroupTransfer supports continuous and binary targets, got {self.kind_!r}")
        Xp = self._person_X(X)
        comp = ([c for c in Xp.columns if pd.api.types.is_numeric_dtype(Xp[c])]
                if self.composition is None else [c for c in self.composition if c != self.group_col])
        absent = [c for c in comp if c not in Xp.columns]
        if absent:
            raise KeyError(f"composition columns not in X: {absent}")
        self.num_comp_ = [c for c in comp if pd.api.types.is_numeric_dtype(Xp[c])]
        self.na_comp_ = [c for c in self.num_comp_ if Xp[c].isna().any()]
        self.cat_comp_ = {c: list(Xp[c].astype(str).value_counts().index[:MAX_CATEGORIES])
                          for c in comp if c not in self.num_comp_}
        gf = group_features if group_features is not None else self.group_features

        cm = pd.Series(y).groupby(g).mean()
        self.group_means_ = cm
        target = logit(cm.clip(*self.clip), eps=0) if self.kind_ == "binary" else cm.to_numpy()
        F = self._group_frame(Xp, g, gf).loc[cm.index]
        self.group_feature_names_ = list(F.columns)
        self.fill_ = F.mean()
        F = F.fillna(self.fill_).fillna(0.0)
        if F.shape[1] == 0 or len(cm) < 3:
            self.level_model_ = None                  # nothing to learn from: overall mean of group means
            self.level_const_ = float(np.mean(target))
        else:
            self.level_model_ = make_pipeline(StandardScaler(), RidgeCV(alphas=list(self.alphas)))
            self.level_model_.fit(F.to_numpy(float), np.asarray(target, float))
        self.person_model_ = self._make_person_model()
        if self.person_model_ is not None:
            dev = y - pd.Series(g).map(cm).to_numpy()
            self.person_model_.fit(Xp, dev)
        return self

    def predict_groups(self, X: pd.DataFrame, groups=None, group_features: pd.DataFrame | None = None) -> pd.Series:
        """Stage A only: predicted level per group in ``X`` (rate for binary), indexed by group label."""
        X = pd.DataFrame(X)
        g = self._groups(X, groups)
        gf = group_features if group_features is not None else self.group_features
        F = self._group_frame(self._person_X(X), g, gf).reindex(columns=self.group_feature_names_)
        if self.level_model_ is None:
            a = np.full(len(F), self.level_const_)
        else:
            a = self.level_model_.predict(F.fillna(self.fill_).fillna(0.0).to_numpy(float))
        if self.kind_ == "binary":
            a = sigmoid(a)
        return pd.Series(a, index=F.index, name="group_level")

    def predict(self, X: pd.DataFrame, groups=None, group_features: pd.DataFrame | None = None) -> np.ndarray:
        X = pd.DataFrame(X)
        g = self._groups(X, groups)
        level = self.predict_groups(X, g, group_features)
        p = pd.Series(g).map(level).to_numpy(float)
        if self.person_model_ is not None:
            p = p + np.asarray(self.person_model_.predict(self._person_X(X)), float)
        if self.kind_ == "binary":
            p = np.clip(p, 0.0, 1.0)
        return p
