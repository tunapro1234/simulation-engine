"""Person-level "digital twin" prediction: what will this person answer to the target questions, given their answers
to other questions (and optionally one LLM twin run)?

Port of the persona kit (`uretim/persona/kod/kit/yontemler.py` `Stack`, `taban_tahmin`, `_soft_impute`,
`_ridge_yigma`, `_kutu_medyani`, `_bagisci`; constants MIN_ETIKET=20, MIN_YIGMA=200, bagisci_k=5, aday=30,
bases svdridge_r30 / softimpute_r10 / knn50, 4 inner folds) on plain wide DataFrames (person x column). The research
kit's scale conversions (deciles, point units) are not ported: targets are used in their raw numeric unit.

Measured on Twin-2K-500 T1 (2058 persons, 122 target columns; uretim/persona/RAPOR.md):
  * ``TwinStack`` without LLM (stack_ml): mean accuracy 0.732, across-person correlation 0.204;
    with one GPT-4.1-mini twin column: 0.735 / 0.275 (nested CV, seed 0).
  * The fair LLM-free baseline reaches correlation **0.342**: stack_ml for the 82 non-price columns plus, for the 40
    purchase-at-price columns, a pooled logistic on product one-hot + price ratio + stack_ml OOF prediction
    (``pooled_logistic_oof``; `uretim/persona/kod/eval/k2_fiyat.py`). Fair hybrid 0.347; the LLM's person-specific
    contribution is +0.006 [+0.004, +0.007]. The raw LLM twin alone: 0.175. Earlier "+0.071 from the LLM" was
    retracted: the LLM was reading the price in the prompt, not knowing the person.
  * Ceiling: test-retest correlation 0.565 (sqrt 0.75), test-retest accuracy 0.817.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

MIN_LABELS = 20      # MIN_ETIKET: fewer labelled persons -> fallback
MIN_STACK = 200      # MIN_YIGMA: with an LLM, columns with fewer labels use the calibrated LLM (k-curve threshold)
BASES = ("svdridge_r30", "softimpute_r10", "knn50")


def _soft_impute(M, rank, iters=60, tol=1e-4):
    """Hard-rank iterative SVD (Mazumder et al. 2010) on a z-scaled array with NaNs."""
    from sklearn.utils.extmath import randomized_svd
    miss = np.isnan(M)
    Z = np.where(miss, 0.0, M)
    prev = None
    for _ in range(iters):
        U, s, Vt = randomized_svd(Z, rank, random_state=0, n_iter=4)
        Znew = np.where(miss, (U * s) @ Vt, M)
        if prev is not None and np.linalg.norm(Znew - prev) / (np.linalg.norm(prev) + 1e-9) < tol:
            return Znew
        prev = Z = Znew
    return Z


def base_predict(name: str, Xz: pd.DataFrame, Yz: pd.DataFrame, train_ids, test_ids) -> pd.DataFrame:
    """LLM-free base predictor in z space (port of `taban_tahmin`).

    Xz: person x feature (z-scaled, NaN allowed); Yz: person x target (only ``train_ids`` rows are used).
    ``name``: ``softimpute_r<k>`` | ``svdridge_r<k>`` (PCA of features + per-column RidgeCV) | ``knn<k>``.
    Returns test x target z predictions."""
    from sklearn.decomposition import PCA
    tr, te = pd.Index(train_ids), pd.Index(test_ids)
    if name.startswith("softimpute"):
        rows = tr.append(te)
        Mt = Yz.reindex(rows).copy()
        Mt.loc[te] = np.nan
        M = pd.concat([Xz.reindex(rows), Mt], axis=1).to_numpy(dtype=float)
        Z = _soft_impute(M, max(1, min(int(name.split("_r")[1]), min(M.shape) - 1)))
        return pd.DataFrame(Z[len(tr):, Xz.shape[1]:], index=te, columns=Yz.columns)
    A = Xz.fillna(0)
    k = int(name.split("_r")[1]) if name.startswith("svdridge") else 50
    k = max(1, min(k, len(tr) - 1, A.shape[1]))
    pca = PCA(k, random_state=0).fit(A.loc[tr].to_numpy())
    Etr, Ete = pca.transform(A.loc[tr].to_numpy()), pca.transform(A.loc[te].to_numpy())
    if name.startswith("svdridge"):
        from sklearn.linear_model import RidgeCV
        P = pd.DataFrame(0.0, index=te, columns=Yz.columns)
        for c in Yz.columns:
            y = Yz.loc[tr, c]
            ok = y.notna().to_numpy()
            if ok.sum() >= 30:
                P[c] = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(Etr[ok], y[ok]).predict(Ete)
        return P
    if name.startswith("knn"):
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=max(1, min(int(name[3:]), len(tr)))).fit(Etr)
        _, idx = nn.kneighbors(Ete)
        import warnings
        with warnings.catch_warnings():          # all-NaN neighbour slices -> NaN -> 0 (z mean)
            warnings.simplefilter("ignore", RuntimeWarning)
            V = np.nanmean(Yz.loc[tr].to_numpy()[idx], axis=1)
        return pd.DataFrame(np.nan_to_num(V), index=te, columns=Yz.columns)
    raise ValueError(f"unknown base: {name}")


def _ridge_stack(Ftr: pd.DataFrame, ytr: pd.Series, Fte: pd.DataFrame):
    """Per-column linear stack: RidgeCV(base predictions + missingness indicators). Returns (train fit, test pred)."""
    from sklearn.linear_model import RidgeCV
    mu = Ftr.mean()
    keep = mu.notna()
    Ftr, Fte, mu = Ftr.loc[:, keep], Fte.loc[:, keep], mu[keep]
    ind = [c for c in Ftr.columns if Ftr[c].isna().any() or Fte[c].isna().any()]
    A = pd.concat([Ftr.fillna(mu), Ftr[ind].isna().astype(float).add_suffix("_na")], axis=1)
    B = pd.concat([Fte.fillna(mu), Fte[ind].isna().astype(float).add_suffix("_na")], axis=1)
    ok = ytr.notna().to_numpy()
    if ok.sum() < 30 or A.shape[1] == 0:
        m = float(np.nanmean(ytr))
        return np.full(len(Ftr), m), np.full(len(Fte), m)
    r = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(A[ok], ytr[ok])
    return r.predict(A), r.predict(B)


def binned_median(p_train, y_train, p_test, n_bins: int = 20):
    """Conditional median for accuracy-type scoring: quantile bins of the (OOF) train prediction, median true answer
    per bin, each test row gets its bin's median (port of `_kutu_medyani`; < 50 rows -> p_test unchanged)."""
    ptr, ytr, pte = (np.asarray(v, float) for v in (p_train, y_train, p_test))
    ok = ~np.isnan(ptr) & ~np.isnan(ytr)
    p, y = ptr[ok], ytr[ok]
    if len(y) < 50:
        return pte
    q = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    nb = max(len(q) - 2, 0)
    bt = np.clip(np.searchsorted(q, p, side="right") - 1, 0, nb)
    be = np.clip(np.searchsorted(q, pte, side="right") - 1, 0, nb)
    med = {b: np.median(y[bt == b]) for b in np.unique(bt)}
    keys = np.array(sorted(med))
    return np.array([med[b] if b in med else med[keys[np.abs(keys - b).argmin()]] for b in be])


def donor_sample(Ztr: pd.DataFrame, Zte: pd.DataFrame, Ytr: pd.DataFrame, k: int, rng, candidates: int = 30):
    """Hot-deck: per test person, the k nearest train persons in stack-prediction space in random order, then the
    remaining candidates; per column the first non-missing real answer is copied (port of `_bagisci`).
    Keeps the correlation structure between answers (Twin T1: 0.894 vs 0.464 for residual bootstrap)."""
    from sklearn.neighbors import NearestNeighbors
    mu, sd = Ztr.mean(), Ztr.std().replace(0, 1)
    A, B = ((Ztr - mu) / sd).fillna(0).to_numpy(), ((Zte - mu) / sd).fillna(0).to_numpy()
    _, idx = NearestNeighbors(n_neighbors=min(max(k, candidates), len(A))).fit(A).kneighbors(B)
    Y = Ytr.to_numpy(float)
    out = np.full((len(B), Y.shape[1]), np.nan)
    k = min(k, idx.shape[1])
    for i, row in enumerate(idx):
        cand = Y[np.concatenate([row[rng.permutation(k)], row[k:]])]
        full = ~np.isnan(cand)
        out[i] = np.where(full.any(0), cand[full.argmax(0), np.arange(Y.shape[1])], np.nan)
    return pd.DataFrame(out, index=Zte.index, columns=Ytr.columns)


def _snap(o, support):
    d = np.sort(np.asarray(support, float))
    if len(d) < 2:
        return np.full_like(o, d[0] if len(d) else np.nan)
    i = np.clip(np.searchsorted(d, o), 1, len(d) - 1)
    return np.where(np.abs(o - d[i - 1]) <= np.abs(d[i] - o), d[i - 1], d[i])


class TwinStack:
    """Hybrid stack: LLM-free bases (svdridge, softimpute, kNN on the person's other answers) + optionally one LLM
    column -> per-column ridge stack, with nested CV (train persons' base predictions are inner-fold OOF; a test
    person's target is never seen).

    fit(features, targets, llm=None, bounds=None, nominal=()):
      features: person x feature DataFrame (numeric, NaN allowed): what is known about every person.
      targets:  person x target DataFrame in raw units; rows of persons later passed to ``predict`` are ignored.
      llm:      optional person x target DataFrame of LLM twin predictions (raw units).
      bounds:   {column: (lo, hi)}; default observed min/max.  nominal: categorical target columns.
    predict(test_ids) -> (point, sample): point = stack prediction (or ``binned_median``); sample = one draw per
      person (``sample="donor"``: hot-deck copy of a near train person's real answers; ``"residual"``: point +
      stack residual bootstrap snapped to the observed support).
    Columns with < ``min_labels`` labels: calibrated LLM if given, else the pooled 0-1 distribution of the other
    targets. With an LLM, columns with < ``min_stack`` labels also use the calibrated LLM. Nominal columns: mode
    (point) and donor copy / marginal draw (sample); the stack is for numeric columns only.
    """

    def __init__(self, bases: Sequence[str] = BASES, inner_folds: int = 4, seed: int = 0, sample: str = "donor",
                 binned: bool = False, n_bins: int = 20, donor_k: int = 5, min_labels: int = MIN_LABELS,
                 min_stack: int = MIN_STACK):
        if sample not in ("donor", "residual"):
            raise ValueError("sample must be 'donor' or 'residual'")
        self.bases, self.inner_folds, self.seed, self.sample = list(bases), inner_folds, seed, sample
        self.binned, self.n_bins, self.donor_k = binned, n_bins, donor_k
        self.min_labels, self.min_stack = min_labels, min_stack

    # ------------------------------------------------------------------ fit
    def fit(self, features: pd.DataFrame, targets: pd.DataFrame, llm: pd.DataFrame | None = None,
            bounds: Mapping[str, tuple] | None = None, nominal: Iterable[str] = ()):
        if features.shape[1] == 0:
            raise ValueError("features needs at least one column")
        # sorted feature columns, as the source pivot: the randomized SVD / KFold results depend on column order
        self.features = features.apply(pd.to_numeric, errors="coerce").sort_index(axis=1)
        self.Y = targets.astype(float)
        self.targets = list(targets.columns)
        self.nominal = set(nominal)
        self.llm = None if llm is None else llm.astype(float)
        bounds = dict(bounds or {})
        self.bounds = {c: tuple(map(float, bounds.get(c, (self.Y[c].min(), self.Y[c].max())))) for c in self.targets}
        self.rng = np.random.default_rng(self.seed)
        return self

    def _labels(self, c, exclude):
        y = self.Y[c].drop(index=exclude, errors="ignore").dropna()
        return y if len(y) >= self.min_labels else None

    def _scale01(self, D: pd.DataFrame, cols):
        lo = pd.Series({c: self.bounds[c][0] for c in cols})
        hi = pd.Series({c: self.bounds[c][1] for c in cols})
        return (D[cols] - lo) / (hi - lo).replace(0, 1)

    def _fallback_setup(self, te):
        num = [c for c in self.targets if c not in self.nominal and self.bounds[c][1] > self.bounds[c][0]
               and self._labels(c, te) is not None]
        pools = [self._scale01(self.Y.drop(index=te, errors="ignore"), [c])[c].dropna().to_numpy() for c in num]
        self.pool_ = np.concatenate(pools) if pools else np.array([0.5])
        self.pool_median_ = float(np.median([np.median(p) for p in pools])) if pools else 0.5
        self.calib_ = (0.0, 1.0)
        if self.llm is not None:
            cols = [c for c in num if c in self.llm.columns]
            if cols:
                T = self._scale01(self.Y.drop(index=te, errors="ignore"), cols)
                L = self._scale01(self.llm.reindex(index=T.index, columns=cols), cols)
                m = T.notna() & L.notna()
                x, y = L.where(m).stack().to_numpy(), T.where(m).stack().to_numpy()
                if len(x) > 50 and x.var() > 0:
                    b = np.cov(x, y)[0, 1] / x.var()
                    self.calib_ = (y.mean() - b * x.mean(), b)

    def _fallback(self, c, te):
        n = len(te)
        lo, hi = self.bounds[c]
        if self.llm is not None and c in self.llm.columns:
            p = self.llm[c].reindex(te).to_numpy(float)
            if c in self.nominal or hi <= lo:
                return p, p.copy()
            q = lo + (self.calib_[0] + self.calib_[1] * (p - lo) / (hi - lo)) * (hi - lo)
            return np.clip(q, lo, hi), p.copy()
        if c in self.nominal:
            return np.full(n, np.nan), np.full(n, np.nan)
        p = np.full(n, lo + self.pool_median_ * (hi - lo))
        o = lo + self.rng.choice(self.pool_, n) * (hi - lo)
        obs = self.Y[c].dropna()
        if len(obs) and np.allclose(obs, np.round(obs)):
            o = np.clip(np.round(o), lo, hi)
        return p, o

    def _X(self, ids):
        X = self.features.reindex(ids)
        X = X.loc[:, X.std() > 0]
        return (X - X.mean()) / X.std()

    # ------------------------------------------------------------------ predict
    def predict(self, test_ids):
        from sklearn.model_selection import KFold
        te = pd.Index(test_ids)
        self._fallback_setup(te)
        P = pd.DataFrame(np.nan, index=te, columns=self.targets)
        O = P.copy()
        has_llm = lambda c: self.llm is not None and c in self.llm.columns
        lab, self.llm_fallback_ = [], []
        for c in self.targets:
            y = self._labels(c, te)
            if y is None:
                P[c], O[c] = self._fallback(c, te)
            elif c in self.nominal:
                P[c], O[c] = y.mode().iloc[0], self.rng.choice(y.to_numpy(), len(te))
            elif has_llm(c) and len(y) < self.min_stack:
                self.llm_fallback_.append(c)
                P[c], O[c] = self._fallback(c, te)
            else:
                lab.append(c)
        self.stacked_ = lab
        if not lab:
            return P, O
        Yl = self.Y[lab].drop(index=te, errors="ignore")
        tr = Yl.index[Yl.notna().any(axis=1)]
        try:                         # sorted train ids, as the source's Index.difference(te): order-independent folds
            tr = tr.sort_values()
        except TypeError:            # mixed, unorderable id types: keep the caller's order
            pass
        mu, sd = Yl.loc[tr].mean(), Yl.loc[tr].std().replace(0, 1)
        Yz = (Yl - mu) / sd
        Xz = self._X(tr.append(te))
        lo = pd.Series({c: self.bounds[c][0] for c in lab})
        hi = pd.Series({c: self.bounds[c][1] for c in lab})
        rng_ = (hi - lo).replace(0, 1)
        s01 = lambda Z: ((Z * sd + mu).clip(lo, hi, axis=1) - lo) / rng_
        base_te = {b: s01(base_predict(b, Xz, Yz, tr, te)) for b in self.bases}
        base_tr = {b: pd.DataFrame(np.nan, index=tr, columns=lab) for b in self.bases}
        for itr, ite in KFold(self.inner_folds, shuffle=True, random_state=self.seed).split(tr):
            a, b_ = tr[itr], tr[ite]
            for b in self.bases:
                base_tr[b].loc[b_] = s01(base_predict(b, Xz.loc[a.append(b_)], Yz, a, b_)).to_numpy()
        L01 = None
        if self.llm is not None:
            Lc = self.llm.reindex(index=tr.append(te), columns=lab)
            L01 = (Lc - lo) / rng_
        Y01 = (Yl.loc[tr] - lo) / rng_
        Ztr = pd.DataFrame(index=tr, columns=lab, dtype=float)
        Zte = pd.DataFrame(index=te, columns=lab, dtype=float)
        for c in lab:
            Ftr = pd.DataFrame({b: base_tr[b][c] for b in self.bases})
            Fte = pd.DataFrame({b: base_te[b][c] for b in self.bases})
            if L01 is not None:
                Ftr["llm"], Fte["llm"] = L01.loc[tr, c].to_numpy(), L01.loc[te, c].to_numpy()
            ftr, fte = _ridge_stack(Ftr, Y01[c], Fte)
            fte = np.clip(fte, 0, 1)
            Ztr[c], Zte[c] = np.clip(ftr, 0, 1), fte
            pt = binned_median(Ztr[c], Y01[c], fte, self.n_bins) if self.binned else fte
            P[c] = lo[c] + pt * (hi[c] - lo[c])
            if self.sample == "residual":
                res = (Y01[c] - ftr).dropna().to_numpy()
                s = np.clip(fte + self.rng.choice(res, len(te)), 0, 1) if len(res) else fte
                o = lo[c] + s * (hi[c] - lo[c])
                obs = Yl[c].dropna()
                if np.allclose(obs, np.round(obs)):
                    o = _snap(o, obs.unique())
                O[c] = o
        if self.sample == "donor":
            nom = [c for c in self.targets if c in self.nominal and self._labels(c, te) is not None]
            copy = lab + nom
            D = donor_sample(Ztr, Zte, self.Y.reindex(index=tr, columns=copy), self.donor_k, self.rng)
            for c in copy:
                empty = D[c].isna().to_numpy()
                if empty.any():
                    D.loc[empty, c] = self.rng.choice(self._labels(c, te).to_numpy(), int(empty.sum()))
            O[copy] = D[copy].to_numpy()
        return P, O


def pooled_logistic_oof(df: pd.DataFrame, y_col: str, group_col: str, categorical: Sequence[str] = (),
                        numeric: Sequence[str] = (), folds: int = 5, seed: int = 0, C: float = 1.0) -> np.ndarray:
    """One logistic regression pooled over many binary columns of the same kind (long table: one row per person x
    column), features = one-hot ``categorical`` + ``numeric``; out-of-fold P(y=1) with folds over ``group_col``
    (persons), so a person's own answers never train their prediction.

    This is how the fair Twin baseline handled the 40 purchase-at-price columns (categorical = product, numeric =
    price / product median price + stack_ml OOF prediction): correlation on those columns 0.536 vs 0.144 for the
    per-column stack (uretim/persona/kod/eval/k2_fiyat.py, RAPOR.md)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    parts = [pd.get_dummies(df[list(categorical)].astype(str), dtype=float).to_numpy()] if categorical else []
    if numeric:
        parts.append(df[list(numeric)].to_numpy(float))
    if not parts:
        raise ValueError("give at least one categorical or numeric feature")
    X = np.hstack(parts)
    y = df[y_col].to_numpy().astype(int)
    g = df[group_col].to_numpy()
    persons = pd.unique(g)
    p = np.full(len(df), np.nan)
    for a, b in KFold(folds, shuffle=True, random_state=seed).split(persons):
        tr, te = np.isin(g, persons[a]), np.isin(g, persons[b])
        m = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def column_correlation(truth: pd.DataFrame, pred: pd.DataFrame, min_rows: int = 10) -> float:
    """Across-person correlation: Pearson per column (NaN-masked; constant or tiny columns count as 0), averaged
    in Fisher-z space (uretim/persona/kod/eval/k2_fiyat.py `fisher`)."""
    r = []
    for c in truth.columns:
        if c not in pred.columns:
            continue
        p, t = pred[c].to_numpy(float), truth[c].reindex(pred.index).to_numpy(float)
        m = ~np.isnan(p) & ~np.isnan(t)
        ok = m.sum() > min_rows and p[m].std() > 0 and t[m].std() > 0
        r.append(np.corrcoef(p[m], t[m])[0, 1] if ok else 0.0)
    if not r:
        return float("nan")
    return float(np.tanh(np.mean(np.arctanh(np.clip(r, -0.999999, 0.999999)))))
