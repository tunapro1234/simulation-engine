"""Uplift / CATE for randomised experiments: how much a treatment (an e-mail, an ad, an SMS) moves each segment or
customer, and whether a per-customer model is worth building at all.

Task shape: one row per unit (customer) with a 0/1 treatment column, an outcome (0/1 or numeric), feature columns
and optionally a segment column. Run order that saves time (D11 rehearsal: a hand T-learner, 100 reps x 2 LightGBM
models, did not finish in 14 min; a treatment x feature LR test would have said it was not needed)::

    import simengine.methods as M
    het = M.heterogeneity_test(df, "sms", "buy", ["age", "gender", "points"], seg="segment")
    het.p, het.advice                                   # p > 0.2: do not spend time on per-customer uplift
    eff = M.segment_uplift(df, "sms", "buy", "segment")  # raw difference + Neyman SE, indexed by segment
    eb = M.shrink_effects(eff.uplift, eff.se)            # Fay-Herriot / empirical Bayes, REML tau^2
    dev = M.split_half_effects(df, "sms", "buy", "segment", R=50, seed=0)   # rep, method, segment, pred, truth, truth_var
    M.summarize_split_half(dev)                          # MAE and noise-corrected MSE per method over the reps
    tl = M.TLearner(model="logistic").fit(train, "sms", "buy", ["age", "gender", "points"])
    score = tl.predict(test)                             # P(y | x, t=1) - P(y | x, t=0)
    M.uplift_at_k(test.buy, test.sms, score, 0.3), M.qini(test.buy, test.sms, score), M.auuc(test.buy, test.sms, score)

Metrics (also used by the bench uplift tasks, `simengine.bench.tasks.uplift`, which import them from here):

    qini / qini_coefficient  normalised Qini coefficient: (area under the Qini curve - random) / (perfect - random), the
                             perfect curve allowing negative effects (Radcliffe 2007; scikit-uplift `qini_auc_score`)
    auuc                     normalised area under the uplift curve, same normalisation (scikit-uplift `uplift_auc_score`)
    uplift_at_k              treated minus control response rate among the top int(k n) rows by score (scikit-uplift
                             `uplift_at_k`, strategy "overall")

A score only has to rank (higher = treat first). Random targeting has uplift@k = the ATE in expectation and Qini 0.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ._util import n_threads

Z95 = 1.959963984540054
TOP_K = 0.3
GBDT_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.9, random_state=0, verbose=-1, n_jobs=4)


# ---------------------------------------------------------------------- curves and metrics
def _thresholds(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descending order and the last index of every distinct-score block (ties move together)."""
    order = np.argsort(score, kind="mergesort")[::-1]
    s = score[order]
    return order, np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]


def _cum(y, t, w, order, thr):
    """Weighted cumulative counts at the thresholds: (n_all, n_treated, y_treated, n_control, y_control)."""
    y, t, w = y[order], t[order], w[order]
    na, nt = np.cumsum(w)[thr], np.cumsum(w * t)[thr]
    yt, yc = np.cumsum(w * y * t)[thr], np.cumsum(w * y * (1 - t))[thr]
    return na, nt, yt, na - nt, yc


def _div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def _curve(kind, y, t, w, order, thr):
    na, nt, yt, nc, yc = _cum(y, t, w, order, thr)
    v = yt - yc * _div(nt, nc) if kind == "qini" else (_div(yt, nt) - _div(yc, nc)) * na
    return np.r_[0.0, na], np.r_[0.0, v]


def _area(x, v):
    return float(np.sum(np.diff(x) * (v[1:] + v[:-1]) / 2))


def _norm_area(kind, y, t, w, model, perfect):
    """(area(model) - area(random)) / (area(perfect) - area(random)); random = straight line to the end point."""
    x, v = _curve(kind, y, t, w, *model)
    xp, vp = _curve(kind, y, t, w, *perfect)
    base = 0.5 * xp[-1] * vp[-1]
    den = _area(xp, vp) - base
    return (_area(x, v) - base) / den if den != 0 else float("nan")


def _perfect_qini_score(y, t):
    return y * t - y * (1 - t)  # treated responders first, control responders last


def _perfect_uplift_scores(y, t):
    """The two candidate perfect orderings of scikit-uplift `perfect_uplift_curve` (chosen per sample)."""
    eq = 2.0 * (y == t)
    return eq + y, eq + t  # summand y if control responders > treated non-responders, else t


def qini_curve(y, treatment, score, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """Qini curve: x = number targeted, value = Y_t(x) - Y_c(x) * N_t(x) / N_c(x) (Radcliffe 2007), at every
    distinct score, starting at (0, 0)."""
    y, t, s = (np.asarray(a, float) for a in (y, treatment, score))
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    return _curve("qini", y, t, w, *_thresholds(s))


def uplift_curve(y, treatment, score, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """Uplift curve: x = number targeted, value = (Y_t / N_t - Y_c / N_c)(x) * x, starting at (0, 0)."""
    y, t, s = (np.asarray(a, float) for a in (y, treatment, score))
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    return _curve("uplift", y, t, w, *_thresholds(s))


def qini_coefficient(y, treatment, score, weights=None) -> float:
    """Normalised Qini coefficient: 1 = perfect ranking (negative effects allowed), 0 = random."""
    y, t, s = (np.asarray(a, float) for a in (y, treatment, score))
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    return _norm_area("qini", y, t, w, _thresholds(s), _thresholds(_perfect_qini_score(y, t)))


qini = qini_coefficient


def auuc(y, treatment, score, weights=None) -> float:
    """Normalised area under the uplift curve: 1 = perfect ranking, 0 = random."""
    y, t, s = (np.asarray(a, float) for a in (y, treatment, score))
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    py, pt = _perfect_uplift_scores(y, t)
    return _auuc_w(y, t, w, _thresholds(s), _thresholds(py), _thresholds(pt))


def _auuc_w(y, t, w, model, perf_y, perf_t):
    cr = float(np.sum(w * y * (1 - t)))          # control responders
    tn = float(np.sum(w * (1 - y) * t))          # treated non-responders
    return _norm_area("uplift", y, t, w, model, perf_y if cr > tn else perf_t)


def uplift_at_k(y, treatment, score, k: float = TOP_K, weights=None) -> float:
    """Treated minus control response rate among the first int(k n) rows by score (ties broken by sort order)."""
    y, t, s = (np.asarray(a, float) for a in (y, treatment, score))
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    return _uak(y, t, w, np.argsort(s, kind="mergesort")[::-1], k)


def _uak(y, t, w, order, k):
    y, t, w = y[order], t[order], w[order]
    top = np.cumsum(w) <= k * w.sum()  # unit weights: the first int(k n) rows, as scikit-uplift
    wt, wc = w * top * t, w * top * (1 - t)
    if wt.sum() == 0 or wc.sum() == 0:
        return float("nan")
    return float((wt * y).sum() / wt.sum() - (wc * y).sum() / wc.sum())


# ---------------------------------------------------------------------- learners
def _gbdt(n_jobs: int | None = None):
    params = GBDT_PARAMS if n_jobs is None else {**GBDT_PARAMS, "n_jobs": n_jobs}
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        warnings.warn("lightgbm not installed; using sklearn HistGradientBoostingClassifier", RuntimeWarning,
                      stacklevel=3)
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=200, random_state=0)


def _logistic(categorical, numeric):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    parts = [("s", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric)]
    if categorical:
        parts.append(("k", OneHotEncoder(handle_unknown="ignore"), categorical))
    return make_pipeline(ColumnTransformer(parts), LogisticRegression(C=1.0, max_iter=2000))


def _make(model, categorical, numeric, n_jobs: int | None = None):
    if hasattr(model, "fit"):  # an estimator instance: a fresh unfitted copy per arm
        from sklearn.base import clone
        return clone(model)
    if callable(model):
        return model()
    if model == "logistic":
        return _logistic(categorical, numeric)
    if model == "gbdt":
        return _gbdt(n_jobs)
    raise ValueError(f"model must be 'logistic', 'gbdt', an estimator or a factory; got {model!r}")


def _X(df, categorical, numeric, model, vocab):
    x = df[numeric + categorical].copy()
    if model != "logistic":  # tree models: categories as integer codes on the training vocabulary
        for c in categorical:
            x[c] = pd.Categorical(x[c].astype(str), categories=vocab[c]).codes
    return x


def _vocab(train, categorical):
    return {c: sorted(train[c].astype(str).unique()) for c in categorical}


def _p1(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else np.asarray(m.predict(X), float)


def two_model(train, test, *, treatment: str, outcome: str, categorical: list[str], numeric: list[str],
              model="logistic") -> np.ndarray:
    """T-learner: separate outcome models on treated and control rows; uplift = P(y | x, t=1) - P(y | x, t=0)."""
    tr = train[treatment].to_numpy() == 1
    v = _vocab(train, categorical)
    Xa, Xt = _X(train, categorical, numeric, model, v), _X(test, categorical, numeric, model, v)
    m1 = _make(model, categorical, numeric).fit(Xa[tr], train[outcome].to_numpy()[tr])
    m0 = _make(model, categorical, numeric).fit(Xa[~tr], train[outcome].to_numpy()[~tr])
    return _p1(m1, Xt) - _p1(m0, Xt)


def s_learner(train, test, *, treatment: str, outcome: str, categorical: list[str], numeric: list[str],
              model="gbdt") -> np.ndarray:
    """S-learner: one outcome model with the treatment flag as a feature; uplift = f(x, 1) - f(x, 0)."""
    num, v = numeric + ["_t"], _vocab(train, categorical)
    Xa = _X(train.assign(_t=train[treatment].astype(float)), categorical, num, model, v)
    m = _make(model, categorical, num).fit(Xa, train[outcome].to_numpy())
    Xt = _X(test.assign(_t=1.0), categorical, num, model, v)
    return _p1(m, Xt) - _p1(m, Xt.assign(_t=0.0))


def class_transformation(train, test, *, treatment: str, outcome: str, categorical: list[str], numeric: list[str],
                         model="logistic") -> np.ndarray:
    """Class variable transformation (Jaskowski & Jaroszewicz 2012): z = y t + (1 - y)(1 - t), uplift = 2 P(z=1|x) - 1.
    The identity needs P(t=1) = 1/2; with any other share the rows are weighted so treated and control each carry
    half of the total weight (then 2 P_w(z=1|x) - 1 equals the uplift again)."""
    t, y = train[treatment].to_numpy(float), train[outcome].to_numpy(float)
    z = (y * t + (1 - y) * (1 - t)).astype(int)
    e = t.mean()
    w = np.where(t == 1, 0.5 / e, 0.5 / (1 - e))
    m, v = _make(model, categorical, numeric), _vocab(train, categorical)
    Xa = _X(train, categorical, numeric, model, v)
    if hasattr(m, "steps"):
        m.fit(Xa, z, **{f"{m.steps[-1][0]}__sample_weight": w})
    else:
        m.fit(Xa, z, sample_weight=w)
    return 2 * _p1(m, _X(test, categorical, numeric, model, v)) - 1


def _split_features(df, treat, y, features, categorical):
    feats = [c for c in df.columns if c not in (treat, y)] if features is None else list(features)
    missing = [c for c in feats + [treat, y] if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in df: {missing}")
    if categorical is None:
        categorical = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == bool]
    cat = [c for c in feats if c in set(categorical)]
    return feats, cat, [c for c in feats if c not in set(cat)]


@dataclass
class TLearner:
    """Two outcome models, one per arm; uplift = f1(x) - f0(x).

    ``model``: ``"logistic"`` (default; median-imputed, scaled numerics + one-hot categories), ``"gbdt"`` (LightGBM
    with the bench parameters, sklearn HistGradientBoosting with a warning if LightGBM is missing; one thread below
    200k rows), an sklearn estimator instance (cloned per arm) or a zero-argument factory. Classifiers give
    P(y=1 | x, t); an estimator without ``predict_proba`` (a regressor) gives E[y | x, t]. ``categorical``: feature
    columns to treat as categories (``None``: the non-numeric ones).

    Cost: two fits per call. Run ``heterogeneity_test`` first; with p > 0.2 the score ranks noise.
    """

    model: Any = "logistic"
    categorical: Sequence[str] | None = None
    n_jobs: int | None = None
    features_: list[str] = field(default_factory=list, init=False, repr=False)
    categorical_: list[str] = field(default_factory=list, init=False, repr=False)

    def _setup(self, df, treat, y, features):
        self.features_, self.categorical_, self.numeric_ = _split_features(df, treat, y, features, self.categorical)
        self.vocab_ = _vocab(df, self.categorical_)
        t = df[treat].to_numpy()
        if not np.isin(t, (0, 1)).all():
            raise ValueError(f"{treat} must be 0/1")
        return t == 1

    def _make(self, n):
        jobs = self.n_jobs if self.n_jobs is not None else n_threads(n)
        return _make(self.model, self.categorical_, self.numeric_, jobs)

    def _x(self, df):
        return _X(df, self.categorical_, self.numeric_, self.model, self.vocab_)

    def fit(self, df: pd.DataFrame, treat: str, y: str, features: Sequence[str] | None = None) -> "TLearner":
        """Fit on ``df``; ``features`` default: every column except ``treat`` and ``y``."""
        tr = self._setup(df, treat, y, features)
        X, yy = self._x(df), df[y].to_numpy()
        self.models_ = (self._make(len(df)).fit(X[~tr], yy[~tr]), self._make(len(df)).fit(X[tr], yy[tr]))
        return self

    def predict_arms(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(f1(x), f0(x)) for every row of ``df``."""
        if not hasattr(self, "models_"):
            raise RuntimeError("call fit() first")
        X = self._x(df)
        return _p1(self.models_[1], X), _p1(self.models_[0], X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Uplift score per row: f1(x) - f0(x)."""
        p1, p0 = self.predict_arms(df)
        return p1 - p0


@dataclass
class SLearner(TLearner):
    """One outcome model with the treatment flag as a feature; uplift = f(x, 1) - f(x, 0). Default ``"gbdt"``: a
    linear model without interactions would give every row the same uplift (up to the link)."""

    model: Any = "gbdt"

    def fit(self, df: pd.DataFrame, treat: str, y: str, features: Sequence[str] | None = None) -> "SLearner":
        tr = self._setup(df, treat, y, features)
        self.numeric_ = self.numeric_ + ["_t"]  # the flag goes through the numeric branch of the model
        X = self._x(df.assign(_t=tr.astype(float)))
        self.models_ = (self._make(len(df)).fit(X, df[y].to_numpy()),)
        return self

    def predict_arms(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "models_"):
            raise RuntimeError("call fit() first")
        X = self._x(df.assign(_t=1.0))
        return _p1(self.models_[0], X), _p1(self.models_[0], X.assign(_t=0.0))


# ---------------------------------------------------------------------- segment effects
def _cols(seg) -> list[str]:
    if seg is None:
        return []
    return [seg] if isinstance(seg, str) else list(seg)


def _binary(v: np.ndarray) -> bool:
    return bool(np.isin(v, (0, 1)).all())


def _arm_stats(y: np.ndarray, binary: bool) -> tuple[float, float, int]:
    """(mean, variance for the SE, n). Binary: q(1 - q) with q = (k + 1) / (n + 2), so a 0/6 or 6/6 cell does not
    get SE 0; numeric: sample variance (NaN for one row)."""
    n = len(y)
    if n == 0:
        return float("nan"), float("nan"), 0
    if binary:
        q = (y.sum() + 1) / (n + 2)
        return float(y.mean()), float(q * (1 - q)), n
    return float(y.mean()), float(np.var(y, ddof=1)) if n > 1 else float("nan"), n


def segment_uplift(df: pd.DataFrame, treat: str, y: str, seg=None) -> pd.DataFrame:
    """Raw treated - control difference per segment with the Neyman SE sqrt(s1^2/n1 + s0^2/n0).

    ``seg``: segment column(s) or ``None`` (one row, "all"). ``treat`` must be 0/1 (bool accepted); rows with a
    missing treatment, outcome or segment are dropped. A 0/1 outcome uses (k + 1) / (n + 2) proportions in the
    variance. Returns a frame indexed by segment: ``uplift``, ``se``, ``ci_low`` / ``ci_high`` (normal 95%),
    ``mean_treat``, ``mean_control``, ``n_treat``, ``n_control``; a segment without both arms has NaN ``uplift`` and
    ``se``. ``.attrs``: ``ate`` and ``ate_se`` over all rows.
    """
    cols = _cols(seg)
    missing = [c for c in [treat, y] + cols if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in df: {missing}")
    d = df[[treat, y] + cols].dropna()
    t = d[treat].astype(float).to_numpy()
    if not _binary(t):
        raise ValueError(f"{treat} must be 0/1")
    yy = d[y].astype(float).to_numpy()
    binary = _binary(yy)

    def row(mask):
        m1, v1, n1 = _arm_stats(yy[mask & (t == 1)], binary)
        m0, v0, n0 = _arm_stats(yy[mask & (t == 0)], binary)
        u = m1 - m0 if n1 and n0 else float("nan")
        se = float(np.sqrt(v1 / n1 + v0 / n0)) if n1 and n0 else float("nan")
        return {"uplift": u, "se": se, "ci_low": u - Z95 * se, "ci_high": u + Z95 * se, "mean_treat": m1,
                "mean_control": m0, "n_treat": n1, "n_control": n0}

    if cols:
        groups = d.reset_index(drop=True).groupby(cols, sort=True).indices
        keys = list(groups)
        rows = []
        for k in keys:
            mask = np.zeros(len(d), bool)
            mask[groups[k]] = True
            rows.append(row(mask))
        idx = pd.MultiIndex.from_tuples(keys, names=cols) if len(cols) > 1 else pd.Index(
            [k[0] if isinstance(k, tuple) else k for k in keys], name=cols[0])
        out = pd.DataFrame(rows, index=idx)
    else:
        out = pd.DataFrame([row(np.ones(len(d), bool))], index=pd.Index(["all"], name="segment"))
    allr = row(np.ones(len(d), bool))
    out.attrs.update(ate=allr["uplift"], ate_se=allr["se"])
    return out


def _reml(u, v, X, t2):
    W = 1.0 / (v + t2)
    XtWX = X.T @ (X * W[:, None])
    beta = np.linalg.solve(XtWX, X.T @ (W * u))
    r = u - X @ beta
    ll = -0.5 * (np.sum(np.log(v + t2)) + np.linalg.slogdet(XtWX)[1] + np.sum(W * r * r))
    return ll, beta


def shrink_effects(est, se, X=None, *, tau2: float | None = None) -> pd.DataFrame:
    """Fay-Herriot / empirical Bayes shrinkage of noisy effects toward a (regression) prior.

    Model: est_i ~ N(X_i beta, se_i^2 + tau^2). ``tau^2`` by REML (grid over [0, upper] + bounded refinement; the
    moment estimator inflates with many small segments, D11 #3), ``beta`` by GLS. ``X``: covariate matrix (rows
    aligned with ``est``; include an intercept column yourself), ``None`` = intercept only (shrink to the
    precision-weighted mean). ``tau2`` fixes tau^2 instead of estimating it.

    Rows with a missing estimate or SE do not enter the fit and get the prior (B = 1). Returns a frame on ``est``'s
    index: ``est``, ``se``, ``prior`` (X beta), ``B`` = se^2 / (se^2 + tau^2) (weight on the prior), ``shrunk`` =
    (1 - B) est + B prior, ``post_se`` = sqrt((1 - B) se^2) (ignores the uncertainty in beta and tau^2).
    ``.attrs``: ``tau2``, ``beta``, ``n_used``.
    """
    idx = est.index if isinstance(est, pd.Series) else pd.RangeIndex(len(np.asarray(est)))
    u, s = np.asarray(est, float), np.asarray(se, float)
    Xa = np.ones((len(u), 1)) if X is None else np.asarray(X, float).reshape(len(u), -1)
    if len(s) != len(u) or len(Xa) != len(u):
        raise ValueError("est, se and X must have the same number of rows")
    ok = np.isfinite(u) & np.isfinite(s) & (s >= 0) & np.isfinite(Xa).all(axis=1)
    p = Xa.shape[1]
    if ok.sum() <= p:
        raise ValueError(f"need more than {p} usable estimates, got {int(ok.sum())}")
    uo, vo, Xo = u[ok], np.maximum(s[ok] ** 2, 1e-12), Xa[ok]
    if tau2 is None:
        from scipy.optimize import minimize_scalar
        upper = max(10 * float(np.var(uo)), 10 * float(vo.max()), 1e-9)
        grid = np.r_[0.0, np.geomspace(upper * 1e-8, upper, 200)]
        ll = np.array([_reml(uo, vo, Xo, g)[0] for g in grid])
        i = int(np.argmax(ll))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        best = grid[i]
        if hi > lo:
            r = minimize_scalar(lambda g: -_reml(uo, vo, Xo, g)[0], bounds=(lo, hi), method="bounded")
            if r.success and -r.fun > ll[i]:
                best = float(r.x)
        tau2 = best
    tau2 = float(max(tau2, 0.0))
    beta = _reml(uo, vo, Xo, tau2)[1]
    prior = Xa @ beta
    v = s ** 2
    B = np.where(ok, np.divide(v, v + tau2, out=np.ones_like(v), where=(v + tau2) > 0), 1.0)
    shrunk = np.where(ok, (1 - B) * np.nan_to_num(u) + B * prior, prior)
    out = pd.DataFrame({"est": u, "se": s, "prior": prior, "B": B, "shrunk": shrunk,
                        "post_se": np.where(ok, np.sqrt((1 - B) * v), np.nan)}, index=idx)
    out.attrs.update(tau2=tau2, beta=beta, n_used=int(ok.sum()))
    return out


# ---------------------------------------------------------------------- split-half dev rig
def _halves(t: np.ndarray, unit: np.ndarray | None, rng: np.random.Generator, boot: bool):
    """Row indices of half A and half B. Units are split first, stratified by treatment (each arm's units halved,
    A gets floor), then each half is bootstrapped over its own units (``boot``), so no unit is in both halves."""
    if unit is None:
        uid, ut, rows = np.arange(len(t)), t, None
    else:
        codes, uid = pd.factorize(unit, sort=False)
        n_units = len(uid)
        uid = np.arange(n_units)
        lo = np.full(n_units, np.inf)
        hi = np.full(n_units, -np.inf)
        np.minimum.at(lo, codes, t)
        np.maximum.at(hi, codes, t)
        if (lo != hi).any():
            raise ValueError("the treatment must be constant within a unit")
        ut = lo
        order = np.argsort(codes, kind="mergesort")
        starts = np.r_[0, np.cumsum(np.bincount(codes, minlength=n_units))]
        rows = (order, starts)
    a_units, b_units = [], []
    for arm in (0, 1):
        ix = rng.permutation(uid[ut == arm])
        h = len(ix) // 2
        a_units.append(ix[:h])
        b_units.append(ix[h:])
    halves = []
    for us in (np.concatenate(a_units), np.concatenate(b_units)):
        if boot:  # resample within the half, per arm so both arms keep their size
            us = np.concatenate([rng.choice(us[ut[us] == arm], (ut[us] == arm).sum(), replace=True)
                                 for arm in (0, 1) if (ut[us] == arm).any()])
        if rows is None:
            halves.append(us)
        else:
            order, starts = rows
            halves.append(np.concatenate([order[starts[u]:starts[u + 1]] for u in us]) if len(us) else
                          np.array([], int))
    return halves[0], halves[1]


def _default_methods(treat, y, seg):
    def raw(a):
        return segment_uplift(a, treat, y, seg).uplift

    def pooled(a):
        r = segment_uplift(a, treat, y, seg)
        return pd.Series(r.attrs["ate"], index=r.index)

    def shrunk(a):
        r = segment_uplift(a, treat, y, seg)
        return shrink_effects(r.uplift, r.se).shrunk

    return {"raw": raw, "pooled": pooled, "shrunk": shrunk}


def split_half_effects(df: pd.DataFrame, treat: str, y: str, seg, R: int = 100, seed: int = 0, *,
                       methods: Mapping[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
                       unit: str | None = None, boot: bool = True) -> pd.DataFrame:
    """Dev rig for segment effects without a labelled truth: predict on half A, score against half B's raw difference.

    Each of the ``R`` reps (``default_rng(seed)``) splits the units in two, stratified by treatment FIRST (each arm
    halved), then bootstraps each half over its own units (``boot``; never before the split, or one customer lands
    in both halves, D11 #3). ``unit``: id column when a unit has several rows (default: one row per unit).

    ``methods``: name -> callable(half_A_frame) -> Series of effects indexed by segment; default ``raw`` (segment
    difference), ``pooled`` (A's ATE everywhere) and ``shrunk`` (``shrink_effects`` to the precision-weighted mean).
    Truth = ``segment_uplift`` on half B; ``truth_var`` = its SE^2, so mean((pred - truth)^2 - truth_var) estimates
    the MSE against the true effect (``summarize_split_half``). Returns a long frame: ``rep``, ``method``, segment
    column(s), ``pred``, ``truth``, ``truth_var``, ``n_treat``, ``n_control`` (half B); segments without both arms in
    B are left out.
    """
    cols = _cols(seg)
    if not cols:
        raise ValueError("seg is required")
    t = df[treat].astype(float).to_numpy()
    if not _binary(t):
        raise ValueError(f"{treat} must be 0/1")
    methods = dict(methods) if methods is not None else _default_methods(treat, y, seg)
    uvals = None if unit is None else df[unit].to_numpy()
    rng = np.random.default_rng(seed)
    d = df.reset_index(drop=True)
    out = []
    for r in range(int(R)):
        a, b = _halves(t, uvals, rng, boot)
        A, B = d.iloc[a].reset_index(drop=True), d.iloc[b].reset_index(drop=True)
        truth = segment_uplift(B, treat, y, seg).dropna(subset=["uplift", "se"])
        for name, fn in methods.items():
            pred = pd.Series(fn(A)).reindex(truth.index)
            fr = pd.DataFrame({"rep": r, "method": name, "pred": pred.to_numpy(float),
                               "truth": truth.uplift.to_numpy(), "truth_var": truth.se.to_numpy() ** 2,
                               "n_treat": truth.n_treat.to_numpy(), "n_control": truth.n_control.to_numpy()},
                              index=truth.index)
            out.append(fr.reset_index())
    res = pd.concat(out, ignore_index=True)
    res = res[["rep", "method"] + cols + ["pred", "truth", "truth_var", "n_treat", "n_control"]]
    res.attrs.update(R=int(R), seed=seed, boot=boot)
    return res


def summarize_split_half(frame: pd.DataFrame, weight: str | None = None) -> pd.DataFrame:
    """Per method over the reps: ``mae``, ``mse_d`` (noise-corrected MSE = mean((pred - truth)^2 - truth_var)) and
    ``spearman`` with their 2.5 / 97.5% rep quantiles (``*_lo`` / ``*_hi``). ``weight``: a column of segment
    weights (e.g. deployment sizes merged in), default equal. Paired differences: pivot ``frame`` by rep."""
    f = frame.dropna(subset=["pred", "truth"])
    w = np.ones(len(f)) if weight is None else f[weight].to_numpy(float)
    e = (f.pred - f.truth).to_numpy()
    f = f.assign(_w=w, _ae=w * np.abs(e), _sd=w * (e ** 2 - f.truth_var.to_numpy()))
    rows = []
    for (m, _), g in f.groupby(["method", "rep"], sort=True):
        sp = g.pred.rank().corr(g.truth.rank()) if len(g) > 2 else np.nan
        rows.append({"method": m, "mae": g._ae.sum() / g._w.sum(), "mse_d": g._sd.sum() / g._w.sum(),
                     "spearman": sp})
    per = pd.DataFrame(rows)
    out = {}
    for m, g in per.groupby("method", sort=True):
        row = {}
        for c in ("mae", "mse_d", "spearman"):
            row[c] = g[c].mean()
            row[f"{c}_lo"], row[f"{c}_hi"] = g[c].quantile(0.025), g[c].quantile(0.975)
        row["reps"] = len(g)
        out[m] = row
    return pd.DataFrame.from_dict(out, orient="index").rename_axis("method")


# ---------------------------------------------------------------------- heterogeneity test
@dataclass
class HeterogeneityResult:
    """``stat`` / ``df`` / ``p``: the joint treatment x feature test; ``q``: Cochran Q per grouping (``by``, ``k``
    groups, ``Q``, ``df``, ``p``, ``I2``); ``advice``: what to do next."""

    stat: float
    df: int
    p: float
    scale: str
    n: int
    q: pd.DataFrame
    advice: str


def _design(d: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for c in features:
        x = d[c]
        if pd.api.types.is_numeric_dtype(x) and x.dtype != bool:
            v = x.to_numpy(float)
            miss = np.isnan(v)
            if miss.all():
                continue
            v = np.where(miss, np.nanmedian(v), v)
            if v.std() > 0:
                cols.append((v - v.mean()) / v.std())
                names.append(c)
            if miss.any() and not miss.all():
                cols.append(miss.astype(float))
                names.append(f"{c}:missing")
        else:
            dm = pd.get_dummies(x.astype(str).fillna("nan"), prefix=c, prefix_sep="=", drop_first=True)
            for k in dm.columns:
                cols.append(dm[k].to_numpy(float))
                names.append(str(k))
    F = np.column_stack(cols) if cols else np.zeros((len(d), 0))
    keep = F.std(axis=0) > 0 if F.shape[1] else np.zeros(0, bool)
    return F[:, keep], [n for n, k in zip(names, keep) if k]


def _logit_ll(X: np.ndarray, y: np.ndarray, iters: int = 100) -> float:
    """Max log-likelihood of a logistic regression by Newton steps (tiny ridge on the slopes against separation)."""
    beta = np.zeros(X.shape[1])
    m = y.mean()
    beta[0] = np.log(m / (1 - m))
    pen = np.r_[0.0, np.full(X.shape[1] - 1, 1e-8)]

    def ll(b):
        z = X @ b
        return float(np.sum(y * z - np.logaddexp(0, z)) - 0.5 * np.sum(pen * b * b))

    cur = ll(beta)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (y - p) - pen * beta
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.lstsq(H, g, rcond=None)[0]
        s = 1.0
        while s > 1e-6:
            nb = beta + s * step
            nv = ll(nb)
            if nv >= cur - 1e-12:
                break
            s /= 2
        beta, prev, cur = nb, cur, nv
        if abs(cur - prev) < 1e-10:
            break
    return cur


def _wald_hc1(X: np.ndarray, y: np.ndarray, test: np.ndarray) -> float:
    """OLS + HC1 sandwich covariance; Wald statistic for the coefficients in ``test`` being 0."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    r = y - X @ b
    meat = (X * (r ** 2)[:, None]).T @ X
    V = XtX_inv @ meat @ XtX_inv * n / max(n - k, 1)
    bt, Vt = b[test], V[np.ix_(test, test)]
    return float(bt @ np.linalg.pinv(Vt) @ bt)


def _cochran(d, treat, y, by_vals: pd.Series, name: str) -> dict:
    r = segment_uplift(d.assign(_g=by_vals.to_numpy()), treat, y, "_g")
    r = r[np.isfinite(r.uplift) & (r.se > 0)]
    k = len(r)
    if k < 2:
        return {"by": name, "k": k, "Q": np.nan, "df": k - 1, "p": np.nan, "I2": np.nan}
    from scipy.stats import chi2
    w = 1 / r.se.to_numpy() ** 2
    u = r.uplift.to_numpy()
    Q = float(np.sum(w * (u - np.sum(w * u) / w.sum()) ** 2))
    return {"by": name, "k": k, "Q": Q, "df": k - 1, "p": float(chi2.sf(Q, k - 1)),
            "I2": max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0}


def heterogeneity_test(df: pd.DataFrame, treat: str, y: str, features: Sequence[str], *, seg=None,
                       scale: str = "logit", bins: int = 5) -> HeterogeneityResult:
    """Is the treatment effect different across customers at all? Run it before any per-customer uplift model.

    Rule: p > 0.2: do not spend time on per-customer uplift; report the constant effect (pooled ATE, or segment
    effects shrunk to it) with random targeting as the baseline. p < 0.05: a model (``TLearner`` / ``SLearner``)
    may pay; check uplift@k against random on held-out rows. In between: weak evidence; a cheap logistic T-learner
    at most. (D11: a hand T-learner, 100 reps x 2 LightGBM models, ran 14 minutes without finishing; this test
    would have shown it was unnecessary.)

    Joint test of all treatment x feature interactions: numeric features standardised (median-imputed, plus a
    missing flag), categorical ones one-hot (first level dropped). ``scale="logit"`` (0/1 outcome): likelihood
    ratio test of y ~ t + F + t:F against y ~ t + F, logistic. ``scale="additive"``: linear model with HC1 robust
    covariance, Wald test of the t:F terms (the risk-difference scale uplift@k acts on; any outcome). On the logit
    scale a constant risk difference over a varying base rate shows up as interaction in large samples, and vice
    versa: pick the scale the decision lives on. Both are chi^2 with ``df`` = number of interaction terms.

    Cochran Q (``q``): the per-group raw effects (``segment_uplift``) over ``seg`` (if given) and over each feature
    (categories as they are, numerics in ``bins`` quantile bins), Q = sum w (u - u_bar)^2 with w = 1 / SE^2,
    chi^2 on k - 1, I2 = (Q - df) / Q.
    """
    feats = list(features)
    cols = _cols(seg)
    missing = [c for c in [treat, y] + feats + cols if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in df: {missing}")
    if scale not in ("logit", "additive"):
        raise ValueError("scale must be 'logit' or 'additive'")
    d = df.dropna(subset=[treat, y]).reset_index(drop=True)
    t = d[treat].astype(float).to_numpy()
    yy = d[y].astype(float).to_numpy()
    if not _binary(t):
        raise ValueError(f"{treat} must be 0/1")
    if scale == "logit" and not _binary(yy):
        raise ValueError("scale='logit' needs a 0/1 outcome; use scale='additive'")
    from scipy.stats import chi2
    F, _names = _design(d, feats)
    k = F.shape[1]
    base = np.column_stack([np.ones(len(d)), t, F])
    full = np.column_stack([base, F * t[:, None]])
    if k == 0:
        stat, p = 0.0, 1.0
    elif scale == "logit":
        stat = max(0.0, 2 * (_logit_ll(full, yy) - _logit_ll(base, yy)))
        p = float(chi2.sf(stat, k))
    else:
        stat = _wald_hc1(full, yy, np.arange(base.shape[1], full.shape[1]))
        p = float(chi2.sf(stat, k))
    qs = []
    if cols:
        g = d[cols].astype(str).agg("|".join, axis=1)
        qs.append(_cochran(d, treat, y, g, "|".join(cols)))
    for c in feats:
        x = d[c]
        if pd.api.types.is_numeric_dtype(x) and x.dtype != bool and x.nunique() > bins:
            g = pd.qcut(x, bins, labels=False, duplicates="drop").astype("float").fillna(-1)
        else:
            g = x.astype(str)
        qs.append(_cochran(d, treat, y, g, c))
    q = pd.DataFrame(qs, columns=["by", "k", "Q", "df", "p", "I2"])
    if p > 0.2:
        advice = (f"no evidence of heterogeneity (p = {p:.3g} > 0.2): do not spend time on per-customer uplift; "
                  "report the constant effect with random targeting as the baseline")
    elif p < 0.05:
        advice = (f"heterogeneous effect (p = {p:.3g}): a per-customer uplift model may pay; check uplift@k "
                  "against random targeting on held-out rows")
    else:
        advice = (f"weak evidence of heterogeneity (p = {p:.3g}): at most a cheap logistic T-learner, and only "
                  "if it beats random targeting on held-out rows")
    return HeterogeneityResult(stat=float(stat), df=int(k), p=float(p), scale=scale, n=len(d), q=q, advice=advice)


__all__ = ["GBDT_PARAMS", "HeterogeneityResult", "SLearner", "TLearner", "TOP_K", "auuc", "class_transformation",
           "heterogeneity_test", "qini", "qini_coefficient", "qini_curve", "s_learner", "segment_uplift",
           "shrink_effects", "split_half_effects", "summarize_split_half", "two_model", "uplift_at_k",
           "uplift_curve"]
