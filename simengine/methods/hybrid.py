"""Hybrids: how an LLM prediction enters a model that learns from data.

Three forms were measured during the prep nights (kanit `LIG-SEMA.md`): (i) shrunk deviation, (ii) LLM prediction as
a GBDT feature, (iii) PPI++ (lives in ``simengine.evidence``). Plus the generic dev-learned stack and calibration of
the intent kit (`uretim/intent/kod/kit/yontemler.py` `valid_ogren`, `yigin`, `kalibre`).

Honest summary: in most measured tasks the LLM term adds little once a strong data baseline exists, and the
shrinkage/stack tuned on dev is what keeps it from hurting. See docs/modules/methods.md for numbers.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ._util import EPS, logit
from .baselines import LightGBMBaseline

ALPHAS = tuple(np.round(np.linspace(0, 1, 11), 2))
LAMBDAS = (0.3, 1, 3, 10, 30, 100, 1e9)      # makro llm_onsel IZGARA; 1e9 ~ prior only


def _mse(y, p):
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def _renorm(P):
    P = np.clip(P, 0, None)
    s = P.sum(-1, keepdims=True)
    return np.where(s > 0, P / np.where(s > 0, s, 1), 1.0 / P.shape[-1])


class ShrunkDeviation:
    """Level from data, deviation from the LLM, shrunk: ``pred = base + alpha * (llm - ref)``.

    ``ref`` is the LLM's own reference level: the mean LLM prediction within ``groups`` (e.g. the category, as in the
    makro ``llm_kal`` prior: b0 = b_category + e_llm - mean_category(e_llm)), or an explicit ``ref`` array (e.g. the
    LLM's overall distribution when ``base`` is the known overall distribution of a subgroup task).
    ``alpha`` is chosen on dev from ``alphas`` (0 = ignore the LLM) by ``loss`` (MSE default). Works for scalars
    (n,) and distributions (n, K); distributions are clipped at 0 and renormalised.

    Measured: makro M2 price elasticity, "level from category, deviation from LLM" + data: test WAPE 0.478 vs
    category + data 0.477 (no gain); the small test gain of averaged priors is reproduced by a shuffled-LLM placebo,
    i.e. it is shrinkage, not LLM knowledge (uretim/makro/llmabm/RAPOR.md 2a'). OpinionQA groups: transferring
    neighbour deviations at alpha=1 was significantly worse than copying Overall (+0.0014 [+0.0006, +0.0021] WD;
    uretim/dongu/GUNLUK.md). Tune alpha on dev; expect alpha near 0 unless the LLM has item-specific signal.
    """

    def __init__(self, alphas: Sequence[float] = ALPHAS, loss: Callable | None = None, distribution: bool = False):
        self.alphas, self.loss, self.distribution = tuple(alphas), loss or _mse, distribution

    @staticmethod
    def deviation(llm, groups=None, ref=None):
        L = np.asarray(llm, float)
        if ref is not None:
            return L - np.asarray(ref, float)
        if groups is None:
            return L - L.mean(0, keepdims=True)
        g = pd.Series(np.asarray(groups))
        if L.ndim == 1:
            return L - pd.Series(L).groupby(g).transform("mean").to_numpy()
        D = pd.DataFrame(L)
        return L - D.groupby(g.to_numpy()).transform("mean").to_numpy()

    def _combine(self, base, dev, a):
        P = np.asarray(base, float) + a * dev
        return _renorm(P) if self.distribution else P

    def fit(self, base, llm, y, groups=None, ref=None):
        dev = self.deviation(llm, groups, ref)
        self.losses_ = {a: self.loss(y, self._combine(base, dev, a)) for a in self.alphas}
        self.alpha_ = float(min(self.losses_, key=self.losses_.get))
        return self

    def predict(self, base, llm, groups=None, ref=None):
        return self._combine(base, self.deviation(llm, groups, ref), self.alpha_)


def ridge_to_prior(Z, y, theta0, lam: float):
    """Ridge regression shrunk toward a prior coefficient vector: (Z'Z + lam I) theta = Z'y + lam theta0.
    lam -> 0: data only; lam -> inf: prior only (the makro "+data" step; centre Z and y first for an
    unpenalised intercept)."""
    Z, y, t0 = np.asarray(Z, float), np.asarray(y, float), np.asarray(theta0, float)
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T @ y + lam * t0)


def select_lambda(fit_predict: Callable[[float], np.ndarray], y, lams: Sequence[float] = LAMBDAS,
                  loss: Callable | None = None):
    """Pick the shrinkage strength on held-out rows: ``fit_predict(lam)`` returns held-out predictions."""
    loss = loss or _mse
    scores = {lam: loss(y, fit_predict(lam)) for lam in lams}
    return min(scores, key=scores.get), scores


class LLMFeatureGBDT(LightGBMBaseline):
    """LightGBM with the LLM prediction as one more feature column (``llm_pred``).

    Measured: no gain where a strong data baseline exists. intent OE category ranking MAP@10 0.3307 -> 0.3303
    (uretim/intent/RAPOR.md); deney D4 risky choice (CPC18) GBDT MSE x100 0.585 -> 0.627, i.e. worse
    (uretim/deney/RAPOR.md 1.5). Always compare against the same model without the column (``LightGBMBaseline``)."""

    def fit(self, X, y, llm_pred, eval_set: tuple | None = None):
        X = pd.DataFrame(X).assign(llm_pred=np.asarray(llm_pred, float))
        ev = None
        if eval_set is not None:
            Xv, yv, lv = eval_set
            ev = (pd.DataFrame(Xv).assign(llm_pred=np.asarray(lv, float)), yv)
        return super().fit(X, y, eval_set=ev)

    def predict_proba(self, X, llm_pred):
        return super().predict_proba(pd.DataFrame(X).assign(llm_pred=np.asarray(llm_pred, float)))

    def predict(self, X, llm_pred):
        return super().predict(pd.DataFrame(X).assign(llm_pred=np.asarray(llm_pred, float)))


# ---------------------------------------------------------------- dev-learned stack / calibration (intent kit)

def _z(kind, preds):
    if kind == "binary":
        return np.column_stack([logit(p, EPS) for p in preds])
    if kind == "multiclass":
        return np.column_stack([np.log(np.clip(np.asarray(p, float), EPS, 1)) for p in preds])
    return np.column_stack([np.asarray(p, float) for p in preds])


def _new(kind, method):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression, Ridge
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    return Ridge(alpha=1.0) if kind == "continuous" else LogisticRegression(max_iter=2000, C=1.0)


def _align(P, model_classes, classes):
    """Put predict_proba columns in ``classes`` order; a class unseen in training gets 0 (intent kit `_hizala`)."""
    out = np.zeros((P.shape[0], len(classes)))
    idx = {c: i for i, c in enumerate(classes)}
    for j, c in enumerate(model_classes):
        out[:, idx[c]] = P[:, j]
    return out


def _fit_apply(kind, method, Za, ya, Zb, classes=None):
    m = _new(kind, method)
    if method == "isotonic":
        return m.fit(Za[:, 0], ya).predict(Zb[:, 0])
    m.fit(Za, ya)
    if kind == "continuous":
        return m.predict(Zb)
    P = m.predict_proba(Zb)
    if kind == "binary":
        return P[:, list(m.classes_).index(1)]
    return P if classes is None else _align(P, list(m.classes_), list(classes))


class DevStack:
    """Stack several methods' predictions with a model learned on dev: logistic on logits (binary), on log
    probabilities (multiclass), ridge (continuous). ``fit`` also stores honest dev predictions (``dev_pred_``) from a
    5-fold cross-fit, so the dev score of the stack is not optimistic (intent kit `valid_ogren`). Multiclass
    probabilities are aligned to ``classes_`` (sorted labels of ``y_dev``); a class missing from a training fold gets
    probability 0 in that fold, as in the kit's `_hizala`."""

    def __init__(self, kind: str = "binary", folds: int = 5, seed: int = 0):
        self.kind, self.folds, self.seed = kind, folds, seed

    def fit(self, dev_preds: Sequence, y_dev):
        from sklearn.model_selection import KFold
        Z, y = _z(self.kind, dev_preds), np.asarray(y_dev)
        self.Z_, self.y_ = Z, y
        self.classes_ = np.unique(y) if self.kind == "multiclass" else None
        out = np.zeros((len(y), len(self.classes_))) if self.kind == "multiclass" else np.zeros(len(y))
        for a, b in KFold(self.folds, shuffle=True, random_state=self.seed).split(Z):
            out[b] = _fit_apply(self.kind, "stack", Z[a], y[a], Z[b], self.classes_)
        self.dev_pred_ = out
        return self

    def predict(self, test_preds: Sequence):
        return _fit_apply(self.kind, "stack", self.Z_, self.y_, _z(self.kind, test_preds), self.classes_)


class BinaryCalibrator:
    """Platt scaling, or isotonic when the dev set has >= 1000 rows (intent kit `kalibre`)."""

    def __init__(self, method: str = "auto"):
        self.method = method

    def fit(self, p_dev, y_dev):
        y = np.asarray(y_dev).astype(int)
        self.method_ = ("isotonic" if len(y) >= 1000 else "platt") if self.method == "auto" else self.method
        self.Z_ = np.asarray(p_dev, float)[:, None] if self.method_ == "isotonic" else _z("binary", [p_dev])
        self.y_ = y
        return self

    def predict(self, p):
        Zb = np.asarray(p, float)[:, None] if self.method_ == "isotonic" else _z("binary", [p])
        return _fit_apply("binary", "isotonic" if self.method_ == "isotonic" else "stack", self.Z_, self.y_, Zb)
