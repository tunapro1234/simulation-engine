"""movielens: MovieLens-1M (GroupLens; 1,000,209 ratings of 3,706 movies by 6,040 users, 2000-04 -> 2003-02) with a
per-user temporal split, two targets and LLM-free baselines.

    from simengine.bench.tasks import movielens as ml
    task = ml.MovieLens1M(target="rating")             # or target="next_item"
    data = task.load("/path/to/catalogue")             # or $SIMENGINE_DATA
    frame = task.run(data=data)                        # baselines + your methods -> EvalFrame
    task.score(frame, "bias")                          # {'metric': 'rmse', 'value', 'lo', 'hi', 'n': users}
    ml.persona_text(data, int(data.test["user_id"][0]))   # train-only profile text for an LLM method

Each user's ratings are ordered by (timestamp, movie_id); ML-1M has many ratings in the same second, so the movie id
breaks ties deterministically.

target='rating' (kind continuous): the last ceil(test_frac * n_u) ratings of every user are test (default 0.2), the
    rest train. Metrics RMSE (primary) and MAE, user-cluster bootstrap. Baselines global_mean, user_mean, item_mean,
    bias (mu + b_u + b_i, ridge objective of Koren 2008 sec. 2.1 solved by exact coordinate steps) and item_knn
    (Koren 2008 eq. 2-3: baseline + similarity-weighted residuals of the k most similar movies the user rated, shrunk
    correlation n_ij / (n_ij + 100) * rho_ij; rho_ij here is the uncentred correlation of bias residuals over
    co-raters).
target='next_item' (kind binary): leave-last-out; the user's last rating is the target, everything before is train.
    Test rows per user = target (truth 1) + n_candidates - 1 movies the user never rated, drawn uniformly from the
    movies present in train (default_rng(seed)), in shuffled order. Metrics hit@10 (primary) and ndcg@10 within each
    user, as expectations under a uniform random tie-break (a constant score gets hit@10 = 10 / n_candidates).
    Baselines random, popularity (train rating count), item_knn (sum over the user's history of binary co-rating
    cosine, each movie keeping its next_k = 50 most similar). `full_ranking` scores against every unrated movie
    instead: sampled metrics can reorder methods (Krichene & Rendle, KDD 2020).
split='dev' applies the same rule once more inside train (train -> dev); test rows are then never loaded as truth.

The split is per user, not a global time cut: train holds other users' later ratings (cross-user leakage).
"""
from __future__ import annotations

import os
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, DataNotFound, Metric, need, register

AGE = {1: "under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
OCCUPATION = {0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
              4: "college/grad student", 5: "customer service", 6: "doctor/health care", 7: "executive/managerial",
              8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired",
              14: "sales/marketing", 15: "scientist", 16: "self-employed", 17: "technician/engineer",
              18: "tradesman/craftsman", 19: "unemployed", 20: "writer"}
GENDER = {"M": "male", "F": "female"}
TARGETS = ("rating", "next_item")
TOP_K = 10


# ---------------------------------------------------------------------- files
def find_dir(root: str) -> str:
    """Directory holding ratings.dat: root itself, root/ml-1m (the GroupLens zip) or root/acik/ml-1m."""
    for d in (root, os.path.join(root, "ml-1m"), os.path.join(root, "acik", "ml-1m")):
        if os.path.exists(os.path.join(d, "ratings.dat")):
            return d
    raise DataNotFound(f"ratings.dat not found under {root} (looked in ., ml-1m/, acik/ml-1m/)")


def _read_colon(path: str, names: list[str]) -> pd.DataFrame:
    # '::' separated with no ':' inside fields -> fast C parser on ':' and every second column
    return pd.read_csv(need(path), sep=":", header=None, usecols=range(0, 2 * len(names), 2), names=None,
                       encoding="latin-1", engine="c").set_axis(names, axis=1)


def read_ml1m(root: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(ratings[user_id, movie_id, rating, timestamp], users[gender, age, occupation, zip] by user_id,
    movies[title, genres] by movie_id)."""
    d = find_dir(root)
    r = _read_colon(os.path.join(d, "ratings.dat"), ["user_id", "movie_id", "rating", "timestamp"])
    u = pd.read_csv(need(os.path.join(d, "users.dat")), sep="::", engine="python", header=None, encoding="latin-1",
                    names=["user_id", "gender", "age", "occupation", "zip"], dtype={"zip": str})
    m = pd.read_csv(need(os.path.join(d, "movies.dat")), sep="::", engine="python", header=None, encoding="latin-1",
                    names=["movie_id", "title", "genres"])
    r = r.astype({"user_id": np.int64, "movie_id": np.int64, "rating": float, "timestamp": np.int64})
    return r, u.set_index("user_id"), m.set_index("movie_id")


# ---------------------------------------------------------------------- split
def order_by_user_time(r: pd.DataFrame) -> pd.DataFrame:
    return r.sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort").reset_index(drop=True)


def tail_mask(r: pd.DataFrame, frac: float | None = None, k: int | None = None) -> np.ndarray:
    """True for each user's last ceil(frac * n_u) (or last k) rows; r must be ordered by user, then time.
    At least one row per user stays out of the tail."""
    n = r.groupby("user_id", sort=False)["user_id"].transform("size").to_numpy()
    pos = r.groupby("user_id", sort=False).cumcount().to_numpy()
    m = np.ceil(frac * n).astype(int) if k is None else np.full(len(n), int(k))
    m = np.minimum(m, n - 1)
    return pos >= n - m


# ---------------------------------------------------------------------- shared fits
def _index(values, universe: np.ndarray) -> np.ndarray:
    """Position of each value in sorted `universe`, -1 if absent."""
    v = np.asarray(values)
    j = np.clip(np.searchsorted(universe, v), 0, max(len(universe) - 1, 0))
    return np.where((len(universe) > 0) & (universe[j] == v), j, -1)


def fit_biases(train: pd.DataFrame, lam: float = 5.0, iters: int = 20) -> tuple[float, pd.Series, pd.Series]:
    """mu, b_u (by user_id), b_i (by movie_id) minimising sum (r - mu - b_u - b_i)^2 + lam (sum b_u^2 + sum b_i^2)
    (Koren 2008 sec. 2.1, lambda_1), by alternating exact block updates."""
    users, u = np.unique(train["user_id"].to_numpy(), return_inverse=True)
    items, i = np.unique(train["movie_id"].to_numpy(), return_inverse=True)
    r = train["rating"].to_numpy(float)
    mu = float(r.mean())
    nu, ni = np.bincount(u, minlength=len(users)), np.bincount(i, minlength=len(items))
    bu = np.zeros(len(users))
    for _ in range(iters):
        bi = np.bincount(i, r - mu - bu[u], minlength=len(items)) / (lam + ni)
        bu = np.bincount(u, r - mu - bi[i], minlength=len(users)) / (lam + nu)
    return mu, pd.Series(bu, index=users), pd.Series(bi, index=items)


def _sparse(train: pd.DataFrame, users: np.ndarray, items: np.ndarray, values=None):
    from scipy.sparse import csr_matrix
    u, i = _index(train["user_id"], users), _index(train["movie_id"], items)
    v = np.ones(len(train)) if values is None else np.asarray(values, float)
    return csr_matrix((v, (u, i)), shape=(len(users), len(items)))


def residual_similarity(train: pd.DataFrame, mu: float, bu: pd.Series, bi: pd.Series, shrink: float = 100.0):
    """(items, S): S_ij = n_ij / (n_ij + shrink) * rho_ij, rho_ij the correlation of bias residuals over the users who
    rated both i and j (uncentred), diagonal 0."""
    users, items = bu.index.to_numpy(), bi.index.to_numpy()
    e = train["rating"].to_numpy(float) - (mu + bu.reindex(train["user_id"]).to_numpy()
                                           + bi.reindex(train["movie_id"]).to_numpy())
    R, Bm, R2 = _sparse(train, users, items, e), _sparse(train, users, items), _sparse(train, users, items, e ** 2)
    num = (R.T @ R).toarray()
    co = (Bm.T @ Bm).toarray()
    q = (R2.T @ Bm).toarray()                 # q_ij = sum over co-raters of e_ui^2
    den = np.sqrt(q * q.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(den > 0, num / den, 0.0) * (co / (co + shrink))
    np.fill_diagonal(S, 0.0)
    return items, S.astype(np.float32)


def cooccurrence_cosine(train: pd.DataFrame, items: np.ndarray, k: int = 0) -> np.ndarray:
    """Binary co-rating cosine n_ij / sqrt(n_i n_j) over `items`, diagonal 0; k > 0 keeps each column's k largest."""
    users = np.unique(train["user_id"].to_numpy())
    Bm = _sparse(train[_index(train["movie_id"], items) >= 0], users, items)
    co = (Bm.T @ Bm).toarray()
    d = np.sqrt(np.diag(co).copy())
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(np.outer(d, d) > 0, co / np.outer(d, d), 0.0)
    np.fill_diagonal(S, 0.0)
    if 0 < k < len(items) - 1:
        cut = np.partition(S, len(items) - k, axis=0)[len(items) - k]
        S = np.where(S >= cut[None, :], S, 0.0)
    return S.astype(np.float32)


# ---------------------------------------------------------------------- rating predictors
def item_knn_ratings(train: pd.DataFrame, test: pd.DataFrame, k: int = 40, shrink: float = 100.0,
                     lam: float = 5.0) -> np.ndarray:
    """Koren 2008 eq. 3: b_ui + sum_{j in S^k(i;u)} s_ij (r_uj - b_uj) / sum s_ij over the k most similar movies u
    rated in train (positive s only); falls back to b_ui. Clipped to [1, 5]."""
    mu, bu, bi = fit_biases(train, lam)
    items, S = residual_similarity(train, mu, bu, bi, shrink)
    base = _bias_pred(test, mu, bu, bi)
    tr_i = _index(train["movie_id"], items)
    tr_e = train["rating"].to_numpy(float) - _bias_pred(train, mu, bu, bi)
    te_i = _index(test["movie_id"], items)
    out = base.copy()
    tr_groups = _groups(train["user_id"].to_numpy())
    for uid, rows in _groups(test["user_id"].to_numpy()).items():
        h = tr_groups.get(uid)
        rows = rows[te_i[rows] >= 0]
        if h is None or not len(rows):
            continue
        s = S[np.ix_(te_i[rows], tr_i[h])].astype(float)
        s[s < 0] = 0.0
        if s.shape[1] > k:
            cut = np.partition(s, s.shape[1] - k, axis=1)[:, s.shape[1] - k]
            s = np.where(s >= cut[:, None], s, 0.0)
        w = s.sum(1)
        ok = w > 0
        out[rows[ok]] += (s[ok] @ tr_e[h]) / w[ok]
    return np.clip(out, 1.0, 5.0)


def _bias_pred(df: pd.DataFrame, mu: float, bu: pd.Series, bi: pd.Series) -> np.ndarray:
    return (mu + bu.reindex(df["user_id"]).fillna(0.0).to_numpy()
            + bi.reindex(df["movie_id"]).fillna(0.0).to_numpy())


def _groups(keys: np.ndarray) -> dict:
    """key -> row indices (input order kept within a key)."""
    o = np.argsort(keys, kind="stable")
    k = keys[o]
    cut = np.flatnonzero(np.diff(k)) + 1
    return {kk: idx for kk, idx in zip(k[np.r_[0, cut]] if len(k) else [], np.split(o, cut) if len(k) else [])}


# ---------------------------------------------------------------------- next-item scorers
# A scorer is f(user_ids, movie_ids) -> score matrix (len(user_ids), len(movie_ids)); higher = more likely next.
# The factories below fit on data.train once and return such an f.
def popularity_scorer(data: TaskData) -> Callable:
    c = data.train["movie_id"].value_counts()
    return lambda user_ids, movie_ids: np.tile(c.reindex(np.asarray(movie_ids)).fillna(0).to_numpy(float),
                                               (len(user_ids), 1))


def item_knn_scorer(data: TaskData, k: int = 0) -> Callable:
    """score(u, i) = sum over u's train movies j of cosine(i, j) on the binary user x movie matrix."""
    tr = data.train
    items = np.unique(tr["movie_id"].to_numpy())
    S = cooccurrence_cosine(tr, items, k)

    def f(user_ids, movie_ids):
        users = np.asarray(user_ids)
        uu = np.unique(users)
        H = _sparse(tr[_index(tr["user_id"], uu) >= 0], uu, items)
        sc = np.asarray(H @ S)[_index(users, uu)]
        j = _index(movie_ids, items)
        return np.where(j[None, :] >= 0, sc[:, np.maximum(j, 0)], 0.0)
    return f


def random_scorer(seed: int = 0) -> Callable:
    rng = np.random.default_rng(seed)
    return lambda user_ids, movie_ids: rng.random((len(user_ids), len(movie_ids)))


def pair_scores(data: TaskData, scorer: Callable, test: pd.DataFrame | None = None, block: int = 1024) -> np.ndarray:
    """Evaluate a matrix scorer on the (user_id, movie_id) rows of data.test, `block` users at a time."""
    t = data.test if test is None else test
    out = np.empty(len(t))
    tu, tm = t["user_id"].to_numpy(), t["movie_id"].to_numpy()
    movies, users = np.unique(tm), np.unique(tu)
    for lo in range(0, len(users), block):
        us = users[lo:lo + block]
        M = np.asarray(scorer(us, movies), float)
        m = np.isin(tu, us)
        out[m] = M[_index(tu[m], us), _index(tm[m], movies)]
    return out


# ---------------------------------------------------------------------- metrics
def expected_topk(n_greater: np.ndarray, n_tied: np.ndarray, k: int = TOP_K) -> tuple[np.ndarray, np.ndarray]:
    """(hit@k, ndcg@k) of one relevant item when n_greater candidates score above it and n_tied tie with it; the rank is
    uniform on n_greater + 1 .. n_greater + n_tied + 1."""
    g, t = np.asarray(n_greater, float), np.asarray(n_tied, float)
    hit = np.clip(k - g, 0, t + 1) / (t + 1)
    D = np.r_[0.0, np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))]       # D[r] = sum_{s<=r} 1/log2(s+1), r <= k
    hi = np.minimum(g + t + 1, k).astype(int)
    lo = np.minimum(g, k).astype(int)
    return hit, (D[hi] - D[lo]) / (t + 1)


def _boot(per_user: dict[str, np.ndarray], stat: Callable, B: int, seed: int) -> dict:
    n = len(next(iter(per_user.values())))
    point = float(stat({k: v for k, v in per_user.items()}))
    if n == 0:
        return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    idx = np.random.default_rng(seed).integers(0, n, (B, n))
    boots = stat({k: v[idx] for k, v in per_user.items()})
    b = np.asarray(boots, float)
    b = b[np.isfinite(b)]
    lo, hi = np.percentile(b, [2.5, 97.5]) if len(b) >= 10 else (np.nan, np.nan)
    return {"value": point, "lo": float(lo), "hi": float(hi), "n": int(n)}


def _err_metric(power: int):
    def metric(task, frame, col, B, seed):
        e = np.abs(frame[col].to_numpy(float) - frame["truth"].to_numpy(float)) ** power
        g = pd.factorize(frame["group"])[0]
        per = {"s": np.bincount(g, e), "n": np.bincount(g).astype(float)}

        def stat(d):
            v = d["s"].sum(-1) / d["n"].sum(-1)
            return np.sqrt(v) if power == 2 else v
        return _boot(per, stat, B, seed)
    return metric


def topk_per_user(frame: pd.DataFrame, col: str, k: int = TOP_K) -> pd.DataFrame:
    """Per user (group): n_greater, n_tied, hit, ndcg of the truth == 1 row among that user's rows."""
    s = frame[col].to_numpy(float)
    g = pd.factorize(frame["group"])[0]
    y = frame["truth"].to_numpy(float) == 1
    tgt = np.full(g.max() + 1 if len(g) else 0, np.nan)
    tgt[g[y]] = s[y]
    ts = tgt[g]
    gt = np.bincount(g, (s > ts) & ~y, minlength=len(tgt))
    tie = np.bincount(g, (s == ts) & ~y, minlength=len(tgt))
    hit, ndcg = expected_topk(gt, tie, k)
    ok = np.isfinite(tgt)
    return pd.DataFrame({"n_greater": gt, "n_tied": tie, "hit": hit, "ndcg": ndcg})[ok]


def _topk_metric(which: str, k: int = TOP_K):
    def metric(task, frame, col, B, seed):
        v = topk_per_user(frame, col, k)[which].to_numpy(float)
        return _boot({"v": v}, lambda d: d["v"].mean(-1), B, seed)
    return metric


RATING_METRICS = (Metric("rmse", False, _err_metric(2), "root mean squared error, user-cluster bootstrap"),
                  Metric("mae", False, _err_metric(1), "mean absolute error, user-cluster bootstrap"))
NEXT_METRICS = (Metric("hit10", True, _topk_metric("hit"), "held-out next movie in the top 10 of its candidates"),
                Metric("ndcg10", True, _topk_metric("ndcg"), "NDCG@10 of the held-out next movie"))


def full_ranking(data: TaskData, scorer: Callable, k: int = TOP_K, *, B: int = 1000, seed: int = 42) -> dict:
    """next_item only: rank each test user's target among all train movies the user has not rated in train (ties
    expected). scorer(user_ids, movie_ids) -> matrix (e.g. `item_knn_scorer(data)`).
    Returns {'hit10': {...}, 'ndcg10': {...}, 'n_items': ...}."""
    tgt = data.test[data.test["y"] == 1].sort_values("user_id", kind="mergesort")
    items = np.unique(data.train["movie_id"].to_numpy())
    users = tgt["user_id"].to_numpy()
    seen = _sparse(data.train[data.train["user_id"].isin(users)], users, items).tocsr()
    gt, tie = np.empty(len(users)), np.empty(len(users))
    ti = _index(tgt["movie_id"], items)
    for lo in range(0, len(users), 1024):
        sl = slice(lo, lo + 1024)
        M = np.asarray(scorer(users[sl], items), float)
        M[seen[sl].toarray() > 0] = -np.inf
        r = np.arange(M.shape[0])
        t = np.where(ti[sl] >= 0, M[r, np.maximum(ti[sl], 0)], -np.inf)
        cold = ti[sl] < 0
        M[r[~cold], ti[sl][~cold]] = -np.inf               # the target is not its own competitor
        gt[sl] = np.where(cold, np.inf, (M > t[:, None]).sum(1))
        tie[sl] = np.where(cold, 0, (M == t[:, None]).sum(1))
    hit, ndcg = expected_topk(np.minimum(gt, 10 ** 9), tie, k)
    hit[~np.isfinite(gt)], ndcg[~np.isfinite(gt)] = 0.0, 0.0
    mean = lambda d: d["v"].mean(-1)  # noqa: E731
    return {f"hit{k}": _boot({"v": hit}, mean, B, seed), f"ndcg{k}": _boot({"v": ndcg}, mean, B, seed),
            "n_items": len(items)}


# ---------------------------------------------------------------------- persona text for LLM methods
def item_text(data: TaskData, movie_id: int) -> str:
    """'<title> - genres: A, B' (the movies.dat title already carries the year)."""
    m = data.meta["movies"]
    if movie_id not in m.index:
        return f"movie {movie_id}"
    row = m.loc[movie_id]
    return f"{row['title']} - genres: {', '.join(str(row['genres']).split('|'))}"


def personas(data: TaskData, user_ids: Iterable[int] | None = None, *, n_liked: int = 10, n_disliked: int = 5,
             n_genres: int = 5, demographics: bool = True) -> dict[int, str]:
    """user_id -> profile text built from users.dat and the user's train ratings only (never data.test):
    demographics (no zip code), rating count and mean, top genres by mean rating (>= 3 ratings), most recent liked
    (4-5) and disliked (1-2) movies."""
    tr = data.train
    if user_ids is not None:
        want = np.asarray(list(user_ids))
        tr = tr[tr["user_id"].isin(want)]
    mv, us = data.meta["movies"], data.meta["users"]
    tr = tr.sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort")
    title = tr["movie_id"].map(mv["title"]).fillna(tr["movie_id"].map(lambda x: f"movie {x}"))
    genres = tr["movie_id"].map(mv["genres"]).fillna("")
    out = {}
    for uid, g in tr.assign(_t=title, _g=genres).groupby("user_id", sort=True):
        parts = []
        if demographics and uid in us.index:
            u = us.loc[uid]
            parts.append(f"Gender: {GENDER.get(u['gender'], u['gender'])}. Age: {AGE.get(int(u['age']), u['age'])}. "
                         f"Occupation: {OCCUPATION.get(int(u['occupation']), u['occupation'])}.")
        r = g["rating"].to_numpy(float)
        parts.append(f"Has rated {len(r)} movies, mean {r.mean():.2f} out of 5.")
        ge = g[["_g", "rating"]].assign(_g=g["_g"].str.split("|")).explode("_g")
        ge = ge[ge["_g"] != ""].groupby("_g")["rating"].agg(["mean", "size"])
        ge = ge[ge["size"] >= 3].sort_values(["mean", "size"], ascending=[False, False]).head(n_genres)
        if len(ge):
            parts.append("Favourite genres (mean rating, count): "
                         + "; ".join(f"{k} {v['mean']:.1f} ({int(v['size'])})" for k, v in ge.iterrows()) + ".")
        for lab, sel, n in (("Recently liked", g["rating"] >= 4, n_liked), ("Recently disliked", g["rating"] <= 2,
                                                                             n_disliked)):
            h = g[sel].tail(n).iloc[::-1]
            if n and len(h):
                parts.append(f"{lab} (newest first): "
                             + "; ".join(f"{t} [{int(x)}]" for t, x in zip(h["_t"], h["rating"])) + ".")
        out[int(uid)] = " ".join(parts)
    return out


def persona_text(data: TaskData, user_id: int, **kw) -> str:
    """One user's profile text (see `personas`)."""
    return personas(data, [user_id], **kw).get(int(user_id), "")


# ---------------------------------------------------------------------- task
@register()
class MovieLens1M(BenchTask):
    """Options: target='rating' | 'next_item'; split='test' | 'dev'; test_frac=0.2 (rating); n_candidates=100 and
    seed=0 (next_item negatives); lam=5 (bias ridge); knn_k=40 (rating neighbours); next_k=50 (next_item: keep each
    movie's k most similar, 0 = all)."""

    name = "movielens"
    kind = "continuous"
    data_subdir = "movielens-1m"
    accepted_options = {
        "target": '"rating" | "next_item" (default "rating")',
        "split": '"test" | "dev" (default "test")',
        "test_frac": "float in (0, 1) (default 0.2; rating only)",
        "n_candidates": "int >= 2 (default 100; next_item only)",
        "seed": "int (default 0; next_item negatives)",
        "lam": "float (default 5, chosen on split=dev; bias ridge)",
        "knn_k": "int (default 40; rating item_knn neighbours)",
        "next_k": "int (default 50, chosen on split=dev; next_item item_knn neighbours per movie, 0 = all)",
    }
    layout = """
    <root>/movielens-1m/ml-1m/{ratings,users,movies}.dat   GroupLens ml-1m.zip extracted ('::' separated, latin-1);
        the .dat files may also sit directly in <root>/movielens-1m/ (or in acik/ml-1m/)
    """
    source = "GroupLens MovieLens 1M (Harper & Konstan 2015, ACM TiiS); research use, citation required, no redistribution"

    def __init__(self, **options):
        super().__init__(**options)
        t = self.options.get("target", "rating")
        if t not in TARGETS:
            raise ValueError(f"movielens: target must be one of {TARGETS}")
        if self.options.get("split", "test") not in ("test", "dev"):
            raise ValueError("movielens: split must be 'test' or 'dev'")
        self.target = t
        self.kind = "continuous" if t == "rating" else "binary"
        self.metrics = RATING_METRICS if t == "rating" else NEXT_METRICS

    # ------------------------------------------------------------------ load
    def _load(self, root: str) -> TaskData:
        r, users, movies = read_ml1m(root)
        r = order_by_user_time(r)
        rated = r.groupby("user_id")["movie_id"].agg(lambda s: set(s.tolist()))   # all ratings, for negatives
        dev = self.options.get("split", "test") == "dev"
        frac = float(self.options.get("test_frac", 0.2))
        if not 0 < frac < 1:
            raise ValueError("movielens: test_frac must be in (0, 1)")
        k = 1 if self.target == "next_item" else None
        if dev:
            r = r[~tail_mask(r, frac, k)].reset_index(drop=True)
        hold = tail_mask(r, frac, k)
        train, test = r[~hold].reset_index(drop=True), r[hold].reset_index(drop=True)
        rule = (f"last ceil({frac} n_u) ratings per user" if k is None else "last rating per user (leave-last-out)")
        meta = {"target": self.target, "split": f"{'dev' if dev else 'test'}: {rule} by (timestamp, movie_id); "
                + ("the test part was removed first" if dev else "train = the earlier ratings"),
                "users": users, "movies": movies, "n_ratings": len(r), "n_users": int(r["user_id"].nunique()),
                "n_items": int(r["movie_id"].nunique()), "source": self.source.strip()}
        if self.target == "next_item":
            test = self._candidates(train, rated, test, meta)
        return TaskData(train=train, test=test, meta=meta)

    def _candidates(self, train, rated, target, meta) -> pd.DataFrame:
        n_c = int(self.options.get("n_candidates", 100))
        if n_c < 2:
            raise ValueError("movielens: n_candidates must be >= 2")
        rng = np.random.default_rng(int(self.options.get("seed", 0)))
        pool = np.unique(train["movie_id"].to_numpy())
        rows = []
        for t in target.itertuples(index=False):
            seen = np.fromiter(rated[t.user_id], np.int64)
            # rejection sampling when the user rated a small share of the pool, else draw from the free list
            if len(seen) > len(pool) // 4 or 2 * n_c > len(pool):
                free = pool[~np.isin(pool, seen)]
                neg = rng.choice(free, min(n_c - 1, len(free)), replace=False)
            else:
                neg = np.empty(0, int)
                while len(neg) < n_c - 1:
                    d = rng.choice(pool, 2 * n_c, replace=True)
                    d = pd.unique(np.r_[neg, d[~np.isin(d, seen)]])
                    neg = d[:n_c - 1]
            ids = np.r_[t.movie_id, neg]
            y = np.r_[1, np.zeros(len(neg), int)]
            o = rng.permutation(len(ids))
            rows.append(pd.DataFrame({"user_id": t.user_id, "movie_id": ids[o], "y": y[o], "timestamp": t.timestamp,
                                      "rating": np.where(y[o] == 1, t.rating, np.nan)}))
        meta["n_candidates"] = n_c
        meta["negatives"] = "uniform over train movies the user never rated (anywhere in the file), default_rng(seed)"
        cols = ["user_id", "movie_id", "y", "timestamp", "rating"]
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)

    # ------------------------------------------------------------------ baselines
    def baselines(self, data: TaskData) -> dict:
        tr, te = data.train, data.test
        if self.target == "next_item":
            return {name: pair_scores(data, self.scorer(data, name)) for name in ("random", "popularity", "item_knn")}
        lam = float(self.options.get("lam", 5.0))
        mu = float(tr["rating"].mean())
        mu_b, bu, bi = fit_biases(tr, lam)
        return {"global_mean": np.full(len(te), mu),
                "user_mean": te["user_id"].map(tr.groupby("user_id")["rating"].mean()).fillna(mu).to_numpy(float),
                "item_mean": te["movie_id"].map(tr.groupby("movie_id")["rating"].mean()).fillna(mu).to_numpy(float),
                "bias": np.clip(_bias_pred(te, mu_b, bu, bi), 1.0, 5.0),
                "item_knn": item_knn_ratings(tr, te, int(self.options.get("knn_k", 40)), lam=lam)}

    def scorer(self, data: TaskData, name: str) -> Callable:
        """next_item matrix scorer of a baseline fitted on data.train, e.g.
        `full_ranking(data, task.scorer(data, "item_knn"))`."""
        if name == "random":
            return random_scorer(int(self.options.get("seed", 0)))
        if name == "popularity":
            return popularity_scorer(data)
        if name == "item_knn":
            return item_knn_scorer(data, int(self.options.get("next_k", 50)))
        raise KeyError(f"movielens: no next_item scorer {name!r}; have ['item_knn', 'popularity', 'random']")

    # ------------------------------------------------------------------ frame
    def _frame(self, data: TaskData) -> pd.DataFrame:
        t, us = data.test, data.meta["users"]
        truth = t["y"].to_numpy(int) if self.target == "next_item" else t["rating"].to_numpy(float)
        return pd.DataFrame({"unit": t["user_id"].astype(str) + "|" + t["movie_id"].astype(str),
                             "group": t["user_id"].to_numpy(), "item": t["movie_id"].to_numpy(),
                             "time": t["timestamp"].to_numpy(), "truth": truth,
                             "segment": t["user_id"].map(us["age"]).map(AGE).fillna("NA").to_numpy(),
                             "seg_gender": t["user_id"].map(us["gender"]).fillna("NA").to_numpy()})
