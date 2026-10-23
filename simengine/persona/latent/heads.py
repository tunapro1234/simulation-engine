"""LLM-free heads on person vectors (the required encoder-only controls), probes on LLM states, LightGBM stacking.

Inputs are per person, cells index into them: X (n_persons, d), rows (n_cells,) person index, cat (n_cells,)
category index, y (n_cells,) 0/1. No per-cell copy of X is built (5k persons x 20 categories x 5120 dims would not
fit comfortably).

    h = EncoderHead("logistic", C=0.1, seed=0).fit(X, rows, cat, y)
    p = h.predict(X_test, rows_test, cat_test)
    pr = Probe("logistic", pca=256, C=0.1, seed=0).fit(H, rows, cat, y)      # H = LLM states
    cal = platt(p_val, y_val, cat_val); p_cal = cal(p_test, cat_test)
"""
from __future__ import annotations

from typing import Callable

import numpy as np

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


class EncoderHead:
    """enc-<u>: per-category L2 logistic regression on u ("logistic"), or one MLP over [u, category embedding(16)]
    with hidden 256 ("mlp", early-stopped on the validation cells by log loss, <= `epochs`)."""

    def __init__(self, kind: str = "logistic", C: float = 0.1, seed: int = 0, *, hidden: int = 256, cat_dim: int = 16,
                 epochs: int = 30, lr: float = 1e-3, standardize: bool = True):
        if kind not in ("logistic", "mlp"):
            raise ValueError("kind must be 'logistic' or 'mlp'")
        self.kind, self.C, self.seed = kind, C, seed
        self.hidden, self.cat_dim, self.epochs, self.lr, self.standardize = hidden, cat_dim, epochs, lr, standardize

    def _prep(self, X, fit=False, rows=None):
        X = np.asarray(X, np.float32)
        if not self.standardize:
            return X
        if fit:                                   # statistics of the persons the fit rows use, never of all of X
            F = X if rows is None else X[np.unique(np.asarray(rows))]
            self.mu_, sd = F.mean(0), F.std(0)
            sd[sd == 0] = 1.0
            self.sd_ = sd
        return (X - self.mu_) / self.sd_

    def fit(self, X, rows, cat, y, *, val=None) -> "EncoderHead":
        """X holds every person (test persons included); only X[rows] enters the fit, standardisation included.
        val = (X_val, rows_val, cat_val, y_val) for MLP early stopping (ignored by logistic)."""
        Z = self._prep(X, fit=True, rows=rows)
        rows, cat, y = np.asarray(rows), np.asarray(cat), np.asarray(y, float)
        self.n_cat_ = int(cat.max()) + 1
        if self.kind == "logistic":
            from sklearn.linear_model import LogisticRegression
            self.models_ = {}
            for c in range(self.n_cat_):
                m = cat == c
                yc = y[m]
                if len(np.unique(yc)) < 2:
                    self.models_[c] = float(yc.mean()) if len(yc) else 0.5
                    continue
                lr = LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed)
                self.models_[c] = lr.fit(Z[rows[m]], yc)
            return self
        return self._fit_mlp(Z, rows, cat, y, val)

    def _fit_mlp(self, Z, rows, cat, y, val):
        import torch
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        nn = torch.nn
        d = Z.shape[1]
        self.emb_ = nn.Embedding(self.n_cat_, self.cat_dim)
        self.net_ = nn.Sequential(nn.Linear(d + self.cat_dim, self.hidden), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(self.hidden, 1))
        params = list(self.emb_.parameters()) + list(self.net_.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-2)
        Zt = torch.as_tensor(Z)
        r, c, yt = (torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cat, dtype=torch.long),
                    torch.as_tensor(y, dtype=torch.float32))
        best, best_state, bad = np.inf, None, 0
        for _ep in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(len(r), generator=g)
            for i in range(0, len(r), 512):
                b = perm[i:i + 512]
                out = self.net_(torch.cat([Zt[r[b]], self.emb_(c[b])], 1)).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out, yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if val is None:
                continue
            pv = self.predict(*val[:3])
            ll = -np.mean(val[3] * np.log(np.clip(pv, EPS, 1)) + (1 - val[3]) * np.log(np.clip(1 - pv, EPS, 1)))
            if ll < best - 1e-5:
                best, bad = ll, 0
                best_state = ([p.detach().clone() for p in self.emb_.parameters()],
                              [p.detach().clone() for p in self.net_.parameters()])
            else:
                bad += 1
                if bad >= 3:
                    break
        if best_state is not None:
            with torch.no_grad():
                for p, q in zip(self.emb_.parameters(), best_state[0]):
                    p.copy_(q)
                for p, q in zip(self.net_.parameters(), best_state[1]):
                    p.copy_(q)
        self.val_logloss_ = best
        return self

    def predict(self, X, rows, cat) -> np.ndarray:
        Z = self._prep(X)
        rows, cat = np.asarray(rows), np.asarray(cat)
        if self.kind == "logistic":
            p = np.empty(len(rows))
            for c in np.unique(cat):
                m = cat == c
                mod = self.models_.get(int(c), 0.5)
                p[m] = mod if isinstance(mod, float) else mod.predict_proba(Z[rows[m]])[:, 1]
            return p
        import torch
        self.net_.eval()
        with torch.no_grad():
            Zt = torch.as_tensor(Z)
            out = self.net_(torch.cat([Zt[torch.as_tensor(rows, dtype=torch.long)],
                                       self.emb_(torch.as_tensor(cat, dtype=torch.long))], 1)).squeeze(1)
        return _sigmoid(out.numpy())


class Probe(EncoderHead):
    """prb: PCA(k) of LLM states then per-category logistic ("logistic"), or the E6 mass-mean direction per category
    (buyers - non-buyers, in PCA space) with a per-category Platt map ("massmean"). PCA is fitted on the fit
    persons only."""

    def __init__(self, estimator: str = "logistic", pca: int = 256, C: float = 0.1, seed: int = 0):
        super().__init__("logistic", C=C, seed=seed)
        if estimator not in ("logistic", "massmean"):
            raise ValueError("estimator must be 'logistic' or 'massmean'")
        self.estimator, self.pca = estimator, pca

    def _project(self, X, fit=False, rows=None):
        from sklearn.decomposition import PCA
        X = np.asarray(X, np.float32)
        if fit:
            F = X if rows is None else X[np.unique(np.asarray(rows))]
            k = int(min(self.pca, F.shape[0] - 1, F.shape[1]))
            self.pca_ = PCA(k, random_state=self.seed, svd_solver="randomized" if k < F.shape[1] // 2 else "auto")
            self.pca_.fit(F)
        return self.pca_.transform(X).astype(np.float32)

    def fit(self, X, rows, cat, y, *, val=None) -> "Probe":
        P = self._project(X, fit=True, rows=rows)
        if self.estimator == "logistic":
            return super().fit(P, rows, cat, y)
        Z = self._prep(P, fit=True, rows=rows)
        rows, cat, y = np.asarray(rows), np.asarray(cat), np.asarray(y, float)
        self.n_cat_ = int(cat.max()) + 1
        self.dirs_ = np.zeros((self.n_cat_, Z.shape[1]), np.float32)
        for c in range(self.n_cat_):
            m = cat == c
            pos, neg = Z[rows[m & (y == 1)]], Z[rows[m & (y == 0)]]
            if len(pos) and len(neg):
                self.dirs_[c] = pos.mean(0) - neg.mean(0)
        s = (Z[rows] * self.dirs_[cat]).sum(1)
        self.cal_ = platt_scores(s, y, cat)
        return self

    def predict(self, X, rows, cat) -> np.ndarray:
        P = self._project(X)
        if self.estimator == "logistic":
            return super().predict(P, rows, cat)
        Z = self._prep(P)
        rows, cat = np.asarray(rows), np.asarray(cat)
        return self.cal_((Z[rows] * self.dirs_[cat]).sum(1), cat)


def platt_scores(s, y, cat=None) -> Callable:
    """Per-category (or global) logistic map of a raw score s -> probability, fitted on (s, y)."""
    from sklearn.linear_model import LogisticRegression
    s, y = np.asarray(s, float), np.asarray(y, float)
    cat = np.zeros(len(s), int) if cat is None else np.asarray(cat)
    maps = {}
    for c in np.unique(cat):
        m = cat == c
        if len(np.unique(y[m])) < 2:
            maps[int(c)] = float(y[m].mean())
            continue
        maps[int(c)] = LogisticRegression(C=1e4, max_iter=1000).fit(s[m, None], y[m])
    glob = LogisticRegression(C=1e4, max_iter=1000).fit(s[:, None], y) if len(np.unique(y)) > 1 else float(y.mean())

    def f(s_new, cat_new=None):
        s_new = np.asarray(s_new, float)
        cat_new = np.zeros(len(s_new), int) if cat_new is None else np.asarray(cat_new)
        out = np.empty(len(s_new))
        for c in np.unique(cat_new):
            m = cat_new == c
            mod = maps.get(int(c), glob)
            out[m] = mod if isinstance(mod, float) else mod.predict_proba(s_new[m, None])[:, 1]
        return out
    return f


def platt(p_val, y_val, cat_val=None) -> Callable:
    """Platt map on logit(p) fitted on validation cells (per category when cat_val is given)."""
    f = platt_scores(_logit(p_val), y_val, cat_val)
    return lambda p, cat=None: f(_logit(p), cat)


LGBM_PARAM = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)


def stack_lgbm(F_fit, y_fit, F_val, y_val, *, seed: int = 0, params: dict | None = None, rounds: int = 3000):
    """stk: LightGBM with klasik.py's T3 settings on row features (history features of the cell's category + PCA of
    a person embedding + category code). Early stopping on the validation rows, then a refit on fit + val with the
    best round count (as klasik.lgbm). Returns (booster, info)."""
    import lightgbm as lgb
    p = dict(LGBM_PARAM, seed=seed, **(params or {}))
    b = lgb.train(p, lgb.Dataset(F_fit, y_fit), rounds, valid_sets=[lgb.Dataset(F_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    best = max(int(b.best_iteration or rounds), 1)
    full = lgb.train(p, lgb.Dataset(np.vstack([F_fit, F_val]), np.concatenate([y_fit, y_val])), best)
    return full, {"rounds": best, "val_pred": b.predict(F_val, num_iteration=best)}


__all__ = ["EncoderHead", "Probe", "platt", "platt_scores", "stack_lgbm", "LGBM_PARAM"]
