"""Single-text binary outcome baseline and smoothed target encoding.

Task shape: one text per row (first search query, a review, a message) and a 0/1 outcome (purchase, churn, click);
predict P(y=1) for new rows. ``TfidfAB`` does not apply (it scores a difference of two texts and has no intercept).

- ``TfidfBinary``: word 1-2 gram + char_wb 3-5 gram TF-IDF (+ optional numeric / categorical extra columns and
  in-fold target-encoded columns) -> logistic regression with intercept; ``C`` picked by inner stratified CV on the
  training rows when ``C=None``. ``fit(X, y)`` / ``predict(X)`` (P(y=1)) / ``predict_proba(X)`` (two columns), so it
  drops into ``oof_predict(..., method="predict")``; the vectorisers, the scaler and the encodings are refit inside
  every fold.
- ``target_encode``: smoothed mean encoding ``(sum + m * prior) / (n + m)`` of one column (or a column tuple), with
  out-of-fold values for the training rows (a row never sees its own label) and full-training values for the test
  rows; unseen categories get the prior.

Measured (G4 rehearsal, ShopCART session -> purchase, first search query, 30,365 training sessions, 1,000 test
sessions): TF-IDF word + char logistic with exact-query target encoding and in-train C choice, test AUC 0.666
[0.617, 0.714]; trained on 500 sessions 0.579, on 50-200 sessions 0.51-0.54 (the text learner needs thousands of
labels). Usage::

    import simengine.methods as M
    tb = M.TfidfBinary("query", extra_cols=("n_steps",), encode_cols=("query",)).fit(train, train.y)
    p = tb.predict(test)                                              # P(y = 1)
    oof = M.oof_predict(M.TfidfBinary("query"), train, features=["query"], target="y")
    tr_enc, te_enc = M.target_encode(train, test, "query", "y", m=20)  # OOF for train, full-train for test
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
SMALL_TEXTS = 200          # below this many training rows min_df drops to 1 (as in TfidfAB)


# ------------------------------------------------------------------------------------------------ target encoding

def _key(df: pd.DataFrame, col) -> pd.Series:
    """Category key per row; a column tuple becomes one joint key. NaN is its own category."""
    cols = [col] if isinstance(col, str) else list(col)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in df: {missing}")
    if len(cols) == 1:
        return df[cols[0]].astype(object).where(df[cols[0]].notna(), "<nan>").astype(str)
    parts = [df[c].astype(object).where(df[c].notna(), "<nan>").astype(str) for c in cols]
    return pd.Series(["\x1f".join(t) for t in zip(*parts)], index=df.index)


def _encode(k_fit: pd.Series, y_fit: np.ndarray, k_new: pd.Series, m: float, prior: float | None = None):
    ok = ~np.isnan(y_fit)
    yk = pd.Series(y_fit[ok], index=k_fit.index[ok])
    if prior is None:
        prior = float(yk.mean()) if ok.any() else np.nan
    st = yk.groupby(k_fit[ok].to_numpy()).agg(["sum", "count"])
    enc = (st["sum"] + m * prior) / (st["count"] + m)
    return k_new.map(enc).fillna(prior).to_numpy(float), prior


def _fold_ids(folds, df: pd.DataFrame, seed: int) -> np.ndarray:
    """int -> shuffled KFold ids; column name / Series / array -> fold ids as given (-1 = never held out)."""
    n = len(df)
    if isinstance(folds, (int, np.integer)):
        if folds < 2:
            raise ValueError("folds must be >= 2 (or oof=False)")
        if n < folds:
            raise ValueError(f"folds={folds} needs at least that many training rows, got {n}")
        ids = np.empty(n, dtype=int)
        perm = np.random.default_rng(seed).permutation(n)
        for k, part in enumerate(np.array_split(perm, folds)):
            ids[part] = k
        return ids
    if isinstance(folds, str):
        folds = df[folds]
    if isinstance(folds, pd.Series):
        folds = folds.reindex(df.index).to_numpy() if not folds.index.equals(df.index) else folds.to_numpy()
    ids = np.asarray(folds)
    if len(ids) != n:
        raise ValueError(f"folds has {len(ids)} fold ids, train has {n} rows")
    return ids


def target_encode(train: pd.DataFrame, test: pd.DataFrame | None, col, y: str, *, m: float = 20, oof: bool = True,
                  folds=5, seed: int = 0) -> tuple[pd.Series, pd.Series | None]:
    """Smoothed mean target encoding of ``col`` (a column name or a tuple of names = joint key).

    value(category) = (sum of y + m * prior) / (count + m), prior = mean of y over the rows the statistics come from;
    categories never seen there get the prior. ``train`` rows: out-of-fold (``oof=True``; each row is encoded from
    the other folds only, so it never sees its own label) or in-sample (``oof=False``, leaks: use only for a quick
    look). ``test`` rows: statistics of all training rows. Rows with a missing ``y`` are not counted but still get an
    encoding. ``folds``: number of shuffled folds (``seed``), or fold ids (a column name, Series or array, one per
    training row; use the same ids as ``oof_predict(folds=...)``; -1 = never held out, encoded from all training
    rows; NaN = encoded from all training rows). Returns ``(train_encoding, test_encoding)`` float Series aligned
    with each frame's index (``test_encoding`` is None when ``test`` is None).
    """
    if y not in train.columns:
        raise KeyError(f"target column {y!r} not in train")
    if m < 0:
        raise ValueError("m must be >= 0")
    k_tr = _key(train, col)
    y_tr = pd.to_numeric(train[y], errors="coerce").to_numpy(float)
    name = f"te_{col if isinstance(col, str) else '_'.join(col)}"
    full, prior = _encode(k_tr, y_tr, k_tr, m)
    tr = full.copy()
    if oof:
        ids = pd.Series(_fold_ids(folds, train, seed))
        for lab in pd.unique(ids.dropna()):
            if lab == -1:
                continue
            te = (ids == lab).to_numpy()
            tr[te], _ = _encode(k_tr[~te], y_tr[~te], k_tr[te], m)
    out_tr = pd.Series(tr, index=train.index, name=name)
    if test is None:
        return out_tr, None
    out_te = pd.Series(_encode(k_tr, y_tr, _key(test, col), m, prior)[0], index=test.index, name=name)
    return out_tr, out_te


# ------------------------------------------------------------------------------------------------ TF-IDF binary

def _texts(X, col: str) -> pd.Series:
    if isinstance(X, pd.DataFrame):
        if col not in X.columns:
            raise KeyError(f"text column {col!r} not in X")
        s = X[col]
    else:
        s = pd.Series(list(X) if not isinstance(X, pd.Series) else X)
    return s.fillna("").astype(str)


class TfidfBinary:
    """TF-IDF (word + char_wb n-grams) logistic regression for one text column and a 0/1 target (see module doc).

    Parameters
    - ``text_col``: text column of ``X`` (``X`` may also be a list / Series of texts when no extra columns are used).
    - ``C``: inverse L2 strength; ``None`` = chosen from ``C_grid`` by inner ``StratifiedKFold(inner_cv)`` on the
      training rows (``scoring``, default ROC AUC; vectorisers refit per inner fold). Chosen value: ``C_``.
    - ``extra_cols``: other columns of ``X``. Numeric ones are mean-imputed and standardised; non-numeric ones are
      one-hot encoded (unseen categories -> all zeros).
    - ``encode_cols``: columns (or column tuples) target-encoded inside ``fit`` (OOF over ``te_folds`` folds on the
      training rows, smoothing ``te_m``), then standardised like a numeric extra column. ``("query",)`` with
      ``text_col="query"`` is the exact-query encoding of the G4 rehearsal.
    - ``word_ngrams`` / ``char_ngrams``: n-gram ranges; ``None`` drops that vectoriser. ``min_df`` = (word, char),
      lowered to 1 below ``SMALL_TEXTS`` training rows. ``class_weight`` goes to ``LogisticRegression``.

    One class only in training -> constant prediction of that class rate (no error), ``C_`` = None.
    """

    def __init__(self, text_col: str = "text", C: float | None = None, extra_cols: Sequence[str] = (), *,
                 encode_cols: Sequence = (), word_ngrams: tuple[int, int] | None = (1, 2),
                 char_ngrams: tuple[int, int] | None = (3, 5), min_df: tuple[int, int] = (2, 3),
                 max_features: int = 200_000, C_grid: Sequence[float] = C_GRID, inner_cv: int = 3,
                 scoring: str = "roc_auc", te_m: float = 20, te_folds: int = 5, class_weight=None, seed: int = 0):
        self.text_col, self.C, self.extra_cols, self.encode_cols = text_col, C, extra_cols, encode_cols
        self.word_ngrams, self.char_ngrams, self.min_df, self.max_features = word_ngrams, char_ngrams, min_df, \
            max_features
        self.C_grid, self.inner_cv, self.scoring = C_grid, inner_cv, scoring
        self.te_m, self.te_folds, self.class_weight, self.seed = te_m, te_folds, class_weight, seed

    # -- features
    def _pipeline(self, n_rows: int, C: float):
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        small = n_rows < SMALL_TEXTS
        parts = []
        if self.word_ngrams is not None:
            parts.append(("word", TfidfVectorizer(ngram_range=tuple(self.word_ngrams), min_df=1 if small else
                                                  self.min_df[0], max_features=self.max_features, sublinear_tf=True),
                          "__text__"))
        if self.char_ngrams is not None:
            parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(self.char_ngrams),
                                                  min_df=1 if small else self.min_df[1],
                                                  max_features=self.max_features, sublinear_tf=True), "__text__"))
        if self.num_cols_:
            parts.append(("num", make_pipeline(SimpleImputer(strategy="mean", keep_empty_features=True),
                                               StandardScaler()), self.num_cols_))
        if self.cat_cols_:
            parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_cols_))
        if not parts:
            raise ValueError("TfidfBinary has no features: both n-gram ranges are None and no extra/encode columns")
        ct = ColumnTransformer(parts, sparse_threshold=1.0)
        lr = LogisticRegression(C=C, max_iter=3000, class_weight=self.class_weight)
        return Pipeline([("x", ct), ("lr", lr)])

    def _frame(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            if self.extra_cols or self.encode_cols:
                raise ValueError("extra_cols / encode_cols need X as a DataFrame")
            X = pd.DataFrame({self.text_col: list(X)})
        missing = [c for c in self.extra_cols if c not in X.columns]
        if missing:
            raise KeyError(f"extra_cols not in X: {missing}")
        out = pd.DataFrame({"__text__": _texts(X, self.text_col).to_numpy()}, index=X.index)
        cats = getattr(self, "cat_cols_", ())
        for c in self.extra_cols:
            v = X[c]
            out[c] = (v.astype(object).where(v.notna(), "<nan>").astype(str) if c in cats else v).to_numpy()
        return out

    def _with_encodings(self, F: pd.DataFrame, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if not self.encode_cols:
            return F
        F = F.copy()
        if y is not None:                         # fit: OOF encodings for the training rows, keep the training table
            self.te_train_ = X.assign(__y__=np.asarray(y, float))
        for j, c in enumerate(self.encode_cols):
            if y is not None:
                enc, _ = target_encode(self.te_train_, None, c, "__y__", m=self.te_m, folds=self.te_folds,
                                       seed=self.seed)
            else:
                _, enc = target_encode(self.te_train_, X, c, "__y__", m=self.te_m, oof=False)
            F[f"__te{j}__"] = enc.to_numpy()
        return F

    # -- API
    def fit(self, X, y):
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        y = np.asarray(y).astype(float)
        if np.isnan(y).any():
            raise ValueError("y has missing values; drop those rows first (oof_predict does)")
        if not set(np.unique(y)) <= {0.0, 1.0}:
            raise ValueError("TfidfBinary needs a 0/1 target")
        y = y.astype(int)
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.extra_cols if c not in X.columns]
            if missing:
                raise KeyError(f"extra_cols not in X: {missing}")
            self.cat_cols_ = [c for c in self.extra_cols if not pd.api.types.is_numeric_dtype(X[c])]
        else:
            self.cat_cols_ = []
        self.num_cols_ = [c for c in self.extra_cols if c not in self.cat_cols_]
        self.num_cols_ += [f"__te{j}__" for j in range(len(self.encode_cols))]
        F = self._frame(X)
        self.rate_ = float(y.mean()) if len(y) else 0.5
        self.cv_scores_ = {}
        if len(np.unique(y)) < 2:
            self.C_, self.model_ = None, None
            return self
        F = self._with_encodings(F, X, y)     # the inner C search reuses these OOF encodings (C choice only)
        if self.C is not None:
            self.C_ = float(self.C)
        else:
            k = min(self.inner_cv, int(np.bincount(y).min()))
            if k < 2:
                self.C_ = 1.0
            else:
                cv = StratifiedKFold(k, shuffle=True, random_state=self.seed)
                for c in self.C_grid:
                    self.cv_scores_[c] = float(np.mean(cross_val_score(self._pipeline(len(F), c), F, y, cv=cv,
                                                                       scoring=self.scoring)))
                self.C_ = max(self.cv_scores_, key=lambda c: (self.cv_scores_[c], -c))   # tie -> stronger penalty
        self.model_ = self._pipeline(len(F), self.C_).fit(F, y)
        return self

    def predict(self, X) -> np.ndarray:
        """P(y = 1) per row."""
        F = self._frame(X)
        if self.model_ is None:
            return np.full(len(F), self.rate_)
        F = self._with_encodings(F, X)
        return self.model_.predict_proba(F)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        p = self.predict(X)
        return np.column_stack([1 - p, p])
