"""Item cold start: member features rebuilt with the held-out items removed, and a matrix-factorisation baseline.

Task shape ("new item launch"): members have ratings of many items; some NEW items have no rating at all. Predict
member x new-item ratings, and the item-level residual (item mean rating minus its audience's mean rating).

``item_holdout_features`` ports the dev-table builder of the D3 rehearsal (``uretim/prova/d3/kod/ozellik.py``
``build`` / ``user_stats`` / ``align`` / ``pair_features``): eligible items are dealt into ``n_splits`` folds; for fold
k its items play "new item": their ratings are removed, member statistics (mean, count, sd, relative activity, genre
affinity) and an SVD taste vector are recomputed from the remaining ratings, and each fold's SVD basis is rotated
onto the full-train basis with orthogonal Procrustes (SVD factors have arbitrary sign / rotation per fit) and,
by default, rescaled to the full basis' overall norm (a fold's SVD sees fewer ratings). The
held-out ratings become the dev rows, with the fold id that ``oof_predict(..., folds=...)`` must reuse.

``MFBaseline`` is a standard biased matrix factorisation (ALS) for the ladder. For an item without training ratings
it can only fall back to the member's bias plus a global item prior, i.e. it is a "member mean" rung in pure item
cold start; it shows how much the item side (metadata, audience composition) adds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

AFFINITY_SHRINK = 5.0     # ozellik.py user_stats: aff = s / (c + 5.0)
SVD_RANK = 16             # ozellik.py K_SVD = 16


# ------------------------------------------------------------------------------------------------ alignment

def procrustes_align(frame: pd.DataFrame, ref: pd.DataFrame, cols: Sequence[str], *, scale: bool = False) -> pd.DataFrame:
    """Rotate ``frame[cols]`` onto ``ref[cols]`` with orthogonal Procrustes fitted on the common index.

    Port of ``ozellik.py`` ``align``. Returns a copy; the rotation matrix is in ``.attrs["rotation"]``. Fewer common
    rows than columns -> ``ValueError`` (the rotation is not identified).
    ``scale=True`` also multiplies the rotated columns by one scalar s = ||ref[cols]|| / ||frame[cols]|| (Frobenius
    norms on the common rows), in ``.attrs["scale"]``: an SVD fitted on a subset of the ratings has smaller
    singular values (about sqrt(share of ratings kept)), which a rotation cannot fix.
    """
    from scipy.linalg import orthogonal_procrustes
    cols = list(cols)
    common = frame.index.intersection(ref.index)
    if len(common) < len(cols):
        raise ValueError(f"procrustes_align needs at least {len(cols)} common rows, got {len(common)}")
    A, B = frame.loc[common, cols].to_numpy(float), ref.loc[common, cols].to_numpy(float)
    R, _ = orthogonal_procrustes(A, B)
    s = 1.0
    if scale:
        na = float(np.sqrt((A ** 2).sum()))
        s = float(np.sqrt((B ** 2).sum())) / na if na > 0 else 1.0
    out = frame.copy()
    out[cols] = frame[cols].to_numpy(float) @ R * s
    out.attrs["rotation"] = R
    if scale:
        out.attrs["scale"] = s
    return out


# ------------------------------------------------------------------------------------------------ member statistics

def _member_stats(tr: pd.DataFrame, u: str, i: str, r: str, genres: pd.DataFrame | None, rank: int,
                  shrink: float, prefix: str):
    """ozellik.py user_stats: member mean / count / sd / relative activity, genre affinity, SVD taste vector."""
    from scipy.sparse import csr_matrix
    from sklearn.utils.extmath import randomized_svd
    g = tr.groupby(u)[r]
    us = pd.DataFrame({"user_mean": g.mean(), "user_n": g.size().astype(float), "user_sd": g.std().fillna(1.0)})
    # relative activity: when popular items are held out, absolute counts shift between folds and test
    us["user_n_pct"] = us["user_n"].rank(pct=True)
    res = tr[r].to_numpy(float) - tr[u].map(us["user_mean"]).to_numpy(float)
    aff = None
    if genres is not None:
        G = genres.reindex(tr[i]).fillna(0.0).to_numpy(float)
        s = pd.DataFrame(G * res[:, None], columns=genres.columns).groupby(tr[u].to_numpy()).sum()
        c = pd.DataFrame(G, columns=genres.columns).groupby(tr[u].to_numpy()).sum()
        aff = s / (c + shrink)
    fac = None
    if rank > 0:
        uid = pd.Index(us.index)
        items = pd.Index(np.sort(tr[i].unique()))
        R = csr_matrix((res, (uid.get_indexer(tr[u]), items.get_indexer(tr[i]))), shape=(len(uid), len(items)))
        if rank >= min(R.shape):
            raise ValueError(f"svd_rank={rank} needs more than {rank} members and items per fold "
                             f"(got {R.shape[0]} members x {R.shape[1]} items)")
        U, S, _ = randomized_svd(R, rank, random_state=0)
        fac = pd.DataFrame(U * S, index=uid, columns=[f"{prefix}{k}" for k in range(rank)])
        us = us.join(fac)
    return us, aff


def _pair_frame(pairs: pd.DataFrame, us: pd.DataFrame, aff, genres, u: str, i: str, users) -> pd.DataFrame:
    """ozellik.py pair_features: member features joined to (member, item) pairs + pair genre affinity."""
    x = pairs.join(us, on=u)
    if users is not None:
        x = x.join(users, on=u)
    if aff is not None:
        fg = genres.reindex(pairs[i]).fillna(0.0).to_numpy(float)
        A = aff.reindex(pairs[u]).fillna(0.0).to_numpy(float)
        x["affinity"] = (fg * A).sum(1) / np.maximum(fg.sum(1), 1)
    return x


@dataclass
class ItemHoldout:
    """Result of ``item_holdout_features``.

    - ``dev``: one row per held-out (member, item) rating: the ids, the rating, ``residual`` (= rating minus the
      member's fold mean; the natural target for an item-residual metric), ``fold`` and the feature columns.
    - ``folds``: ``dev["fold"]`` (pass it as ``oof_predict(..., folds=res.folds)``).
    - ``item_folds``: fold id per eligible item.
    - ``members``: member features from ALL ratings (the basis the folds are aligned to); ``transform`` uses them.
    - ``features``: suggested model inputs (relative count ``user_n_pct`` instead of the absolute ``user_n``).
    - ``svd_cols``: the SVD taste-vector columns.
    - ``rotations`` / ``scales``: per fold, the Procrustes rotation and (``align_scale=True``) the scale factor.
    """
    dev: pd.DataFrame
    folds: pd.Series
    item_folds: pd.Series
    members: pd.DataFrame
    features: list
    svd_cols: list
    rotations: dict = field(default_factory=dict)
    scales: dict = field(default_factory=dict)
    _aff: pd.DataFrame | None = None
    _genres: pd.DataFrame | None = None
    _users: pd.DataFrame | None = None
    _cols: tuple = ("user_id", "item_id", "rating")

    def transform(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Features for new (member, item) pairs (e.g. the test items) from the full-train member statistics,
        in the same basis as ``dev`` (rotation, and with ``align_scale=True`` also the overall SVD scale; with
        ``align_scale=False`` the dev SVD vectors are about sqrt(share of ratings per fold) smaller). Members without
        any training rating get NaN member features."""
        u, i, _ = self._cols
        return _pair_frame(pairs.copy(), self.members, self._aff, self._genres, u, i, self._users)


def item_holdout_features(ratings: pd.DataFrame, *, user_col: str = "user_id", item_col: str = "item_id",
                          rating_col: str = "rating", folds=None, n_splits: int | None = None, seed: int = 0,
                          svd_rank: int = SVD_RANK, cap: int | None = None, min_item_ratings: int = 1,
                          items: pd.DataFrame | None = None, genre_cols: Sequence[str] | None = None,
                          users: pd.DataFrame | None = None, align: str | None = "full", align_scale: bool = True,
                          affinity_shrink: float = AFFINITY_SHRINK, svd_prefix: str = "svd_") -> ItemHoldout:
    """Item-held-out dev table for item cold start (port of D3 ``kod/ozellik.py``; see module doc).

    - ``ratings``: one row per (member, item) rating with ``user_col``, ``item_col``, ``rating_col``.
    - ``folds``: optional Series item id -> fold id (precomputed item folds). Otherwise items with at least
      ``min_item_ratings`` ratings are dealt into ``n_splits`` (default 5) folds as in the port:
      ``default_rng(seed).permutation(n_items) % n_splits`` over the sorted item ids.
    - ``cap``: keep at most ``cap`` ratings per held-out item (random sample, seeded per item; the D3 port used 200
      to mirror the test items' audience size). ``None`` = all.
    - ``items`` + ``genre_cols``: item metadata indexed by item id with 0/1 genre columns -> per-member genre
      affinity (shrunk mean residual per genre, ``/(count + affinity_shrink)``) and the pair feature ``affinity``.
      ``genre_cols=None`` with ``items`` given = all numeric columns of ``items``.
    - ``users``: optional member metadata indexed by member id, joined as is (and added to ``features`` when
      numeric).
    - ``svd_rank``: SVD taste vector size (0 = none), columns ``f"{svd_prefix}{k}"``, from the member x item
      residual matrix (rating minus member mean).
    - ``align="full"``: rotate each fold's SVD basis onto the full-train basis (orthogonal Procrustes on common
      members), so fold vectors and the test-time vectors from ``transform`` are comparable. The rotation is fitted
      on the full basis, which saw the held-out items: the only held-out information in a fold's features is that
      one ``svd_rank x svd_rank`` rotation (norms and inner products of the fold's vectors do not depend on it).
    - ``align_scale=True`` (with ``align="full"``): also rescale each fold's SVD vectors by one scalar so their overall
      norm on common members matches the full basis (``procrustes_align(scale=True)``). A fold's SVD sees only the
      ratings outside its items (about (n_splits - 1) / n_splits of them), so its vectors are about sqrt(that share)
      smaller (sd ratio about 0.89 for 5 folds on synthetic data); a model fitted on ``dev`` would then overweight
      the SVD part at ``transform`` time. Like the rotation, the scalar is fitted on the full basis; with the
      rotation it is the only held-out information in a fold's features.
      ``False`` = rotation only (the D3 port's behaviour).
      ``align=None`` = no rotation (fold bases are then not comparable with each other or with test).

    Members whose ratings are all on the fold's held-out items have no fold statistics; their dev rows are dropped
    (as in the port). Evaluate models on ``dev`` with ``oof_predict(..., folds=res.folds)``, never with a fresh
    split: the features of fold k were built without fold k's items only.
    """
    u, i, r = user_col, item_col, rating_col
    miss = [c for c in (u, i, r) if c not in ratings.columns]
    if miss:
        raise KeyError(f"columns not in ratings: {miss}")
    if align not in ("full", None):
        raise ValueError("align must be 'full' or None")
    e = ratings[[u, i, r]].dropna().reset_index(drop=True)
    genres = None
    if items is not None:
        gc = list(genre_cols) if genre_cols is not None else [c for c in items.columns
                                                                if pd.api.types.is_numeric_dtype(items[c])]
        genres = items[gc].astype(float)
    if users is not None:
        users = users.drop(columns=[c for c in users.columns if c in (i, r)], errors="ignore")
    if folds is not None:
        if n_splits is not None:
            raise ValueError("pass folds= or n_splits=, not both")
        item_folds = pd.Series(folds).dropna()
    else:
        n_splits = 5 if n_splits is None else n_splits
        cnt = e[i].value_counts()
        elig = np.sort(cnt[cnt >= min_item_ratings].index.to_numpy())
        if len(elig) < n_splits:
            raise ValueError(f"need at least n_splits={n_splits} items with >= {min_item_ratings} ratings, "
                             f"got {len(elig)}")
        rng = np.random.default_rng(seed)
        item_folds = pd.Series(rng.permutation(len(elig)) % n_splits, index=elig)
    item_folds.name = "fold"
    rank = int(svd_rank)
    svd_cols = [f"{svd_prefix}{k}" for k in range(rank)]
    members, aff_full = _member_stats(e, u, i, r, genres, rank, affinity_shrink, svd_prefix)
    order = {x: n for n, x in enumerate(np.sort(item_folds.index.to_numpy()))}
    dev, rotations, scales = [], {}, {}
    for k in sorted(pd.unique(item_folds)):
        held = item_folds.index[item_folds == k]
        is_held = e[i].isin(held)
        hid, tr = e[is_held], e[~is_held]
        if cap is not None and len(hid):
            parts = []
            for it, d in hid.groupby(i, sort=True):
                if len(d) > cap:          # seeded per item: the sample does not depend on the rating values
                    d = d.sample(cap, random_state=np.random.default_rng([abs(int(seed)), order[it]]))
                parts.append(d)
            hid = pd.concat(parts)
        us, aff = _member_stats(tr, u, i, r, genres, rank, affinity_shrink, svd_prefix)
        if rank > 0 and align == "full":
            us = procrustes_align(us, members, svd_cols, scale=bool(align_scale))
            rotations[k] = us.attrs.pop("rotation")
            if align_scale:
                scales[k] = us.attrs.pop("scale")
        hid = hid[hid[u].isin(us.index)]
        x = _pair_frame(hid.reset_index(drop=True), us, aff, genres, u, i, users)
        x["residual"] = x[r] - x["user_mean"]
        x["fold"] = k
        dev.append(x)
    dev = pd.concat(dev, ignore_index=True)
    feats = ["user_mean", "user_n_pct", "user_sd"] + (["affinity"] if genres is not None else []) + svd_cols
    if users is not None:
        feats += [c for c in users.columns if pd.api.types.is_numeric_dtype(users[c]) and c not in feats]
    return ItemHoldout(dev=dev, folds=dev["fold"], item_folds=item_folds, members=members, features=feats,
                       svd_cols=svd_cols, rotations=rotations, scales=scales, _aff=aff_full, _genres=genres, _users=users,
                       _cols=(u, i, r))


# ------------------------------------------------------------------------------------------------ MF baseline

class MFBaseline:
    """Biased matrix factorisation fitted by alternating least squares: ``mu + b_u + b_i + p_u . q_i``.

    ``fit(X, y)`` / ``predict(X)`` with ``X`` holding ``user_col`` and ``item_col`` (other columns are ignored), so
    it drops into ``oof_predict``. ``rank=0`` = biases only. ``reg`` / ``reg_bias``: ridge penalties (sum-of-squares
    scale) on the factors / biases; ``n_iter`` ALS sweeps. These defaults are generic starting values, not tuned on
    any rehearsal.

    Fallbacks: unseen item -> ``mu + b_u + item_prior_`` (the member's own level plus the unweighted mean item bias,
    or ``item_prior=`` if given); unseen member -> ``mu + b_i``; both unseen -> ``mu + item_prior_``. In pure item
    cold start every test item is unseen, so the prediction is the member's level plus one constant.
    """

    def __init__(self, user_col: str = "user_id", item_col: str = "item_id", *, rank: int = 16, reg: float = 10.0,
                 reg_bias: float = 5.0, n_iter: int = 10, item_prior: float | None = None, seed: int = 0):
        self.user_col, self.item_col, self.rank, self.reg, self.reg_bias = user_col, item_col, rank, reg, reg_bias
        self.n_iter, self.item_prior, self.seed = n_iter, item_prior, seed

    def get_params(self, deep: bool = True) -> dict:
        return dict(user_col=self.user_col, item_col=self.item_col, rank=self.rank, reg=self.reg,
                    reg_bias=self.reg_bias, n_iter=self.n_iter, item_prior=self.item_prior, seed=self.seed)

    def set_params(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    @staticmethod
    def _solve(idx_own, idx_other, r, other_f, other_b, n_own, k, reg, reg_bias):
        """For each owner: ridge of (r - other bias) on [other factors, 1] -> (factors, bias)."""
        F = np.zeros((n_own, k))
        b = np.zeros(n_own)
        order = np.argsort(idx_own, kind="stable")
        bounds = np.searchsorted(idx_own[order], np.arange(n_own + 1))
        pen = np.diag(np.r_[np.full(k, reg), reg_bias])
        for o in range(n_own):
            rows = order[bounds[o]:bounds[o + 1]]
            if rows.size == 0:
                continue
            j = idx_other[rows]
            A = np.c_[other_f[j], np.ones(rows.size)]
            t = r[rows] - other_b[j]
            sol = np.linalg.solve(A.T @ A + pen, A.T @ t)
            F[o], b[o] = sol[:k], sol[k]
        return F, b

    def fit(self, X: pd.DataFrame, y):
        X = pd.DataFrame(X)
        y = np.asarray(y, float)
        ok = np.isfinite(y) & X[self.user_col].notna().to_numpy() & X[self.item_col].notna().to_numpy()
        users = pd.Index(pd.unique(X[self.user_col][ok]))
        items = pd.Index(pd.unique(X[self.item_col][ok]))
        ui = users.get_indexer(X[self.user_col][ok])
        ii = items.get_indexer(X[self.item_col][ok])
        yy = y[ok]
        self.mu_ = float(yy.mean()) if yy.size else 0.0
        r = yy - self.mu_
        k = int(self.rank)
        rs = np.random.default_rng(self.seed)
        P = rs.normal(0, 0.1, (len(users), k))
        Q = rs.normal(0, 0.1, (len(items), k))
        bu, bi = np.zeros(len(users)), np.zeros(len(items))
        for _ in range(max(1, int(self.n_iter))):
            P, bu = self._solve(ui, ii, r, Q, bi, len(users), k, self.reg, self.reg_bias)
            Q, bi = self._solve(ii, ui, r, P, bu, len(items), k, self.reg, self.reg_bias)
        self.users_, self.items_ = users, items
        self.P_, self.Q_, self.bu_, self.bi_ = P, Q, bu, bi
        self.item_prior_ = float(bi.mean()) if self.item_prior is None and bi.size else float(self.item_prior or 0.0)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        ui = self.users_.get_indexer(X[self.user_col])
        ii = self.items_.get_indexer(X[self.item_col])
        su, si = ui >= 0, ii >= 0
        p = np.full(len(X), self.mu_)
        p[su] += self.bu_[ui[su]]
        p[si] += self.bi_[ii[si]]
        p[~si] += self.item_prior_
        both = su & si
        if self.rank > 0:
            p[both] += np.einsum("ij,ij->i", self.P_[ui[both]], self.Q_[ii[both]])
        return p
