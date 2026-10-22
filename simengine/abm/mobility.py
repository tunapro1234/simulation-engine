"""Human mobility on a 200 x 200 grid with 30-minute slots (YJMob100K / HuMob Challenge): trajectory metrics, the
baselines that are hard to beat, and an EPR agent simulator.

Data convention: a DataFrame with columns ``uid, d, t, x, y`` (day, slot 0..47, cell coordinates 1..200 as in
YJMob100K, Yabe et al. 2024 Sci Data 11:397). A cell is coded ``c = (x - 1) * 200 + (y - 1)`` (:func:`to_cell`).
Predictors take ``train`` (history rows) and ``query`` (``uid, d, t`` rows to fill) and return one cell per query row,
in query order.

* Metrics, reimplemented from the organisers' reference code (github.com/yahoojapan/geobleu, MIT; GEO-BLEU from
  Shimizu, Tsubouchi, Yabe, SIGSPATIAL 2022, doi:10.1145/3557915.3560951). Per user and per day:
  :func:`geobleu` = brevity penalty x geometric mean over n = 1..N of p_n, where p_n greedily matches predicted and
  true n-grams by proximity prod_k exp(-beta * dist_k) (largest first, each n-gram used once, ties in row-major
  order) and divides the matched sum by the number of predicted n-grams; :func:`dtw` = dynamic time warping with
  Euclidean cell distance / ``scale``. A user's score is the mean over its days, the task score the mean over users
  (:func:`user_scores`). Two parameter sets (:data:`VARIANTS`): ``humob23`` (HuMob 2023, commit cdaa364: N = 3,
  beta = 0.5, DTW in km = cells / 2, first DP row initialised to 0 so the prediction may start anywhere in the truth)
  and ``humob25`` (GIS Cup 2025, commit d9efb0f: N = 5, beta = 0.5, standard DTW with only [0, 0] = 0).
* Baselines: :func:`slot_mode` (per user, the most frequent cell at that slot), :func:`global_slot_mode` (the most
  frequent cell of everybody at that slot), :class:`MarkovMobility` (first-order transitions between consecutive
  same-day records of each user; ``decode='argmax'`` or ``'sample'``).
* :class:`EPRMobility`: exploration and preferential return (Song, Koren, Wang, Barabasi, Nature Physics 6:818,
  2010; arXiv:1010.0436). At each move the agent explores with P_new = rho * S^-gamma (S = distinct cells seen so
  far; defaults rho = 0.6, gamma = 0.21 from the paper), jumping a distance dr ~ dr^(-1-alpha) (alpha = 0.55,
  1 cell to 200 cells = 100 km) in a random direction; otherwise it returns to a seen cell with probability
  proportional to its visit count. When to move: ``waiting='empirical'`` (per-user share of consecutive same-day
  records that change cell, the TALP port) or ``'powerlaw'`` (Song's P(dt) ~ dt^(-1-beta), beta = 0.8, 1 slot to
  17 h). Each day starts from one of the user's observed first-of-day cells (``day_reset=True``).
  Ported from talp-simulation-hackathon-2026 ``uretim/tahmin/kod/reklam_yeri.py`` (commit 20eaac9: ``hazirla`` data
  cleaning, ``EprIst`` statistics, ``epr_sim`` loop); the POI-gravity exploration kernel of that file is replaced by
  the paper's isotropic Levy jump.

Usage::

    import simengine.abm.mobility as M
    train, test = df[df.d < 60], df[df.d >= 60]                    # uid, d, t, x, y
    q = test[["uid", "d", "t"]]
    preds = {"slot_mode": M.slot_mode(train, q), "global": M.global_slot_mode(train, q),
             "markov": M.MarkovMobility().fit(train).predict(q), "epr": M.EPRMobility(seed=0).fit(train).predict(q)}
    for name, c in preds.items():
        s = M.user_scores(q.uid, q.d, q.t, M.from_cell(c), test[["x", "y"]].to_numpy(), variant="humob23")
        print(name, s.geobleu.mean(), s.dtw.mean())

Measured numbers (2,000-user YJMob100K subsample, days 0-59 -> 60-74) are in docs/modules/abm/mobility.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GRID = 200
SLOTS = 48
NC = GRID * GRID
MASK = 999                       # YJMob100K / HuMob code for a hidden coordinate

VARIANTS = {
    "humob23": {"max_n": 3, "beta": 0.5, "dtw_scale": 2.0, "dtw_free_start": True},
    "humob25": {"max_n": 5, "beta": 0.5, "dtw_scale": 2.0, "dtw_free_start": False},
}


# ------------------------------------------------------------------ cells
def to_cell(x, y) -> np.ndarray:
    """1-based (x, y) -> cell code 0..39,999."""
    return (np.asarray(x, np.int64) - 1) * GRID + (np.asarray(y, np.int64) - 1)


def from_cell(c) -> np.ndarray:
    """Cell code -> (n, 2) int array of 1-based (x, y)."""
    c = np.asarray(c, np.int64)
    return np.stack([c // GRID + 1, c % GRID + 1], -1)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop users with any masked (999) coordinate, keep the first row of duplicate (uid, d, t), sort by uid, d, t
    (the data rules of the TALP port)."""
    bad = df.loc[(df.x == MASK) | (df.y == MASK), "uid"].unique()
    out = df[~df.uid.isin(bad)]
    out = out.sort_values(["uid", "d", "t"], kind="stable").drop_duplicates(["uid", "d", "t"], keep="first")
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ metrics
def _geo_pn(sys_xy: np.ndarray, ans_xy: np.ndarray, n: int, beta: float) -> float:
    """Greedy n-gram matching precision for one pair (any lengths)."""
    ms, ma = len(sys_xy) - n + 1, len(ans_xy) - n + 1
    dx = sys_xy[:, None, 0].astype(float) - ans_xy[None, :, 0]
    dy = sys_xy[:, None, 1].astype(float) - ans_xy[None, :, 1]
    e = np.exp(-beta * np.sqrt(dx ** 2. + dy ** 2.))
    p = e[:ms, :ma].copy()
    for k in range(1, n):
        p = p * e[k:k + ms, k:k + ma]
    p = p.ravel()
    total, used_r, used_c = 0.0, np.zeros(ms, bool), np.zeros(ma, bool)
    for idx in np.argsort(-p, kind="stable"):
        i, j = divmod(int(idx), ma)
        if used_r[i] or used_c[j]:
            continue
        total += p[idx]
        used_r[i] = used_c[j] = True
        if used_r.all() or used_c.all():
            break
    return total / ms


def _gmean(p) -> float:
    p = np.asarray(p, float)
    return 0.0 if (p <= 0).any() else float(np.exp(np.log(p).mean()))


def geobleu(pred, truth, max_n: int = 3, beta: float = 0.5) -> float:
    """GEO-BLEU of one predicted vs true sequence of (x, y) (reference ``calc_geobleu_orig``)."""
    s, a = np.asarray(pred, float).reshape(-1, 2), np.asarray(truth, float).reshape(-1, 2)
    if len(s) == 0 or len(a) == 0:
        raise ValueError("empty trajectory")
    nmax = min(max_n, len(s), len(a))
    pn = [_geo_pn(s, a, n, beta) for n in range(1, nmax + 1)]
    bp = 1.0 if len(s) > len(a) else float(np.exp(1.0 - len(a) / len(s)))
    return bp * _gmean(pn)


def dtw(pred, truth, scale: float = 2.0, free_start: bool = True) -> float:
    """DTW with Euclidean cost / scale (reference ``calc_dtw_orig``). free_start=True reproduces the HuMob 2023 code
    (``dtw_matrix[0] = 0``: the whole first row is 0); False is the 2025 fix (only [0, 0] = 0)."""
    s, a = np.asarray(pred, float).reshape(-1, 2), np.asarray(truth, float).reshape(-1, 2)
    return float(_dtw_batch(s[None], a[None], scale, free_start)[0])


def _geobleu_batch(S: np.ndarray, A: np.ndarray, max_n: int, beta: float) -> np.ndarray:
    """GEO-BLEU for B equal-length pairs, S, A (B, L, 2). Same greedy order as the reference: the largest remaining
    proximity first, ties broken by (pred n-gram, true n-gram) index."""
    B, L, _ = S.shape
    dx = S[:, :, None, 0] - A[:, None, :, 0]
    dy = S[:, :, None, 1] - A[:, None, :, 1]
    e = np.exp(-beta * np.sqrt(dx ** 2. + dy ** 2.))
    logs = np.zeros(B)
    zero = np.zeros(B, bool)
    p = e
    nmax = min(max_n, L)
    rows = np.arange(B)
    for n in range(1, nmax + 1):
        if n > 1:
            p = p[:, :-1, :-1] * e[:, n - 1:, n - 1:]
        m = L - n + 1
        q = p.copy().reshape(B, m * m)
        tot = np.zeros(B)
        for _ in range(m):
            k = q.argmax(1)
            tot += q[rows, k]
            i, j = k // m, k % m
            q3 = q.reshape(B, m, m)
            q3[rows, i, :] = -1.0
            q3[rows, :, j] = -1.0
        pn = tot / m
        zero |= pn <= 0
        logs += np.log(np.where(pn > 0, pn, 1.0))
    return np.where(zero, 0.0, np.exp(logs / nmax))


def _dtw_batch(S: np.ndarray, A: np.ndarray, scale: float, free_start: bool) -> np.ndarray:
    """DTW for B pairs of equal shapes S (B, n, 2), A (B, m, 2)."""
    B, n, _ = S.shape
    m = A.shape[1]
    c = np.sqrt((S[:, :, None, 0] - A[:, None, :, 0]) ** 2. + (S[:, :, None, 1] - A[:, None, :, 1]) ** 2.) / scale
    prev = np.zeros((B, m + 1)) if free_start else np.r_[0.0, np.full(m, np.inf)][None].repeat(B, 0)
    for i in range(n):
        cur = np.full((B, m + 1), np.inf)
        best_up = np.minimum(prev[:, 1:], prev[:, :-1])          # (i-1, j) and (i-1, j-1)
        for j in range(m):
            cur[:, j + 1] = c[:, i, j] + np.minimum(best_up[:, j], cur[:, j])
        prev = cur
    return prev[:, -1]


def user_scores(uid, d, t, pred_xy, true_xy, variant: str = "humob23", metrics=("geobleu", "dtw"),
                chunk: int = 2048) -> pd.DataFrame:
    """Per-user HuMob scores: each metric computed day by day on the (d, t)-sorted rows and averaged over the user's
    days. Returns DataFrame uid, n_days, <metrics>; the task score is its column mean."""
    cfg = VARIANTS[variant]
    uid, d, t = np.asarray(uid), np.asarray(d, np.int64), np.asarray(t, np.int64)
    P, T = np.asarray(pred_xy, float).reshape(-1, 2), np.asarray(true_xy, float).reshape(-1, 2)
    if not (len(uid) == len(d) == len(t) == len(P) == len(T)):
        raise ValueError("uid, d, t, pred_xy, true_xy must have the same length")
    if np.isnan(P).any():
        raise ValueError("pred_xy has NaN")
    o = np.lexsort((t, d, uid))
    uid, d, P, T = uid[o], d[o], P[o], T[o]
    new = np.r_[True, (uid[1:] != uid[:-1]) | (d[1:] != d[:-1])]
    start = np.flatnonzero(new)
    length = np.diff(np.r_[start, len(uid)])
    day_uid = uid[start]
    out = {m: np.empty(len(start)) for m in metrics}
    for L in np.unique(length):
        sel = np.flatnonzero(length == L)
        for b0 in range(0, len(sel), chunk):
            s = sel[b0:b0 + chunk]
            idx = start[s][:, None] + np.arange(L)[None]
            Sx, Ax = P[idx], T[idx]
            if "geobleu" in out:
                out["geobleu"][s] = _geobleu_batch(Sx, Ax, cfg["max_n"], cfg["beta"])
            if "dtw" in out:
                out["dtw"][s] = _dtw_batch(Sx, Ax, cfg["dtw_scale"], cfg["dtw_free_start"])
    days = pd.DataFrame({"uid": day_uid, **out})
    res = days.groupby("uid", sort=True).agg(n_days=(metrics[0], "size"), **{m: (m, "mean") for m in metrics})
    return res.reset_index()


# ------------------------------------------------------------------ baselines
def _mode_by(keys: np.ndarray, cells: np.ndarray):
    """Unique keys and their most frequent cell (ties -> smallest cell)."""
    code = np.asarray(keys, np.int64) * NC + np.asarray(cells, np.int64)
    uq, n = np.unique(code, return_counts=True)
    k, c = uq // NC, uq % NC
    o = np.lexsort((c, -n, k))
    k, c = k[o], c[o]
    first = np.r_[True, k[1:] != k[:-1]]
    return k[first], c[first]


def _lookup(keys_tab, vals_tab, q, fill=-1):
    q = np.asarray(q, np.int64)
    if not len(keys_tab):
        return np.full(len(q), fill, np.int64)
    i = np.minimum(np.searchsorted(keys_tab, q), len(keys_tab) - 1)
    return np.where(keys_tab[i] == q, vals_tab[i], fill)


def _cells(df: pd.DataFrame) -> np.ndarray:
    return to_cell(df["x"].to_numpy(), df["y"].to_numpy())


def global_slot_table(train: pd.DataFrame) -> np.ndarray:
    """(48,) most frequent cell over everybody at each slot (-1 for a slot nobody was seen at)."""
    k, c = _mode_by(train["t"].to_numpy(), _cells(train))
    tab = np.full(SLOTS, -1, np.int64)
    tab[k] = c
    if (tab < 0).any():
        tab[tab < 0] = _mode_by(np.zeros(len(train)), _cells(train))[1][0]
    return tab


def global_slot_mode(train: pd.DataFrame, query: pd.DataFrame, table: np.ndarray | None = None) -> np.ndarray:
    """Everybody's most frequent cell at the query slot. `table` (48,) may come from a larger population
    (e.g. :func:`global_slot_table` of all users' history)."""
    tab = global_slot_table(train) if table is None else np.asarray(table, np.int64)
    return tab[query["t"].to_numpy(np.int64)]


def slot_mode(train: pd.DataFrame, query: pd.DataFrame, window: int = 0, fallback: np.ndarray | None = None
              ) -> np.ndarray:
    """Per user, the most frequent history cell at the query slot (pooling slots t +- window). Falls back to the
    user's overall mode, then to `fallback` (48,) / the global slot mode for users without history."""
    u, t, c = train["uid"].to_numpy(np.int64), train["t"].to_numpy(np.int64), _cells(train)
    if window:
        off = np.arange(-window, window + 1)
        tt = (t[:, None] + off[None]).ravel()
        ok = (tt >= 0) & (tt < SLOTS)
        u, c, t = np.repeat(u, len(off))[ok], np.repeat(c, len(off))[ok], tt[ok]
    k, m = _mode_by(u * SLOTS + t, c)
    qu, qt = query["uid"].to_numpy(np.int64), query["t"].to_numpy(np.int64)
    out = _lookup(k, m, qu * SLOTS + qt)
    ku, mu = _mode_by(train["uid"].to_numpy(np.int64), _cells(train))
    miss = out < 0
    out[miss] = _lookup(ku, mu, qu[miss])
    miss = out < 0
    if miss.any():
        tab = global_slot_table(train) if fallback is None else np.asarray(fallback, np.int64)
        out[miss] = tab[qt[miss]]
    return out


def _sorted_query(query: pd.DataFrame):
    qu, qd, qt = (query[c].to_numpy(np.int64) for c in ("uid", "d", "t"))
    o = np.lexsort((qt, qd, qu))
    return qu, qd, qt, o


class _History:
    """Per-user statistics of the history period (the TALP ``EprIst`` port): user index, move probability between
    consecutive same-day records, first-of-day cells, stay counts per cell (a stay = a run of equal cells)."""

    def __init__(self, train: pd.DataFrame):
        df = train.sort_values(["uid", "d", "t"], kind="stable")
        self.uids = np.unique(df["uid"].to_numpy(np.int64))
        u = np.searchsorted(self.uids, df["uid"].to_numpy(np.int64))
        d, c = df["d"].to_numpy(np.int64), _cells(df)
        N = self.N = len(self.uids)
        same = (u[1:] == u[:-1]) & (d[1:] == d[:-1])
        moved = same & (c[1:] != c[:-1])
        npair = np.bincount(u[1:][same], minlength=N)
        self.m = np.where(npair > 0, np.bincount(u[1:][moved], minlength=N) / np.maximum(npair, 1), 0.0)
        new_day = np.r_[True, ~same]
        self.fd_c = c[new_day]
        self.fd_ptr = np.r_[0, np.cumsum(np.bincount(u[new_day], minlength=N))]
        stay = new_day | np.r_[True, c[1:] != c[:-1]]
        code = u[stay] * NC + c[stay]
        uq, n = np.unique(code, return_counts=True)
        hu = uq // NC
        self.h_c = uq % NC
        self.h_ptr = np.r_[0, np.cumsum(np.bincount(hu, minlength=N))]
        self.h_cum = np.cumsum(n.astype(float))
        self.h_tot = np.bincount(hu, weights=n, minlength=N).astype(float)
        self.S = np.bincount(hu, minlength=N).astype(float)
        # consecutive same-day transitions (for the Markov baseline)
        self.tr_from, self.tr_to, self.tr_u = c[:-1][same], c[1:][same], u[1:][same]

    def index(self, uid):
        return _lookup(self.uids, np.arange(self.N), uid)

    def first_of_day(self, users, rng):
        s, n = self.fd_ptr[users], self.fd_ptr[users + 1] - self.fd_ptr[users]
        return self.fd_c[s + np.minimum((rng.random(len(users)) * n).astype(np.int64), n - 1)]

    def weighted_stay(self, users, rng):
        s, e = self.h_ptr[users], self.h_ptr[users + 1]
        lo = np.where(s > 0, self.h_cum[np.maximum(s - 1, 0)], 0.0)
        target = lo + rng.random(len(users)) * (self.h_cum[e - 1] - lo)
        i = np.clip(np.searchsorted(self.h_cum, target, side="right"), s, e - 1)
        return self.h_c[i]


def _by_slot(qd, qt, o):
    """Groups of sorted-query positions by absolute slot d * 48 + t (each user at most once per group)."""
    s = (qd * SLOTS + qt)[o]
    if not len(s):
        return []
    order = np.argsort(s, kind="stable")
    ss = s[order]
    cut = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1], True])
    return [(int(ss[cut[i]]), o[order[cut[i]:cut[i + 1]]]) for i in range(len(cut) - 1)]


@dataclass
class MarkovMobility:
    """First-order Markov chain per user over consecutive same-day records. Each query day starts at the user's
    slot-mode cell for its first query slot, then follows the most likely next cell (``decode='argmax'``, ties ->
    smallest cell) or a sampled one (``'sample'``). A cell with no observed exit keeps the user in place."""

    decode: str = "argmax"
    seed: int = 0

    def fit(self, train: pd.DataFrame) -> "MarkovMobility":
        if self.decode not in ("argmax", "sample"):
            raise ValueError("decode must be 'argmax' or 'sample'")
        self.train_ = train
        h = self.h_ = _History(train)
        key = h.tr_u * NC + h.tr_from
        code = key * NC + h.tr_to                        # < N * NC^2, fits int64 up to ~5.7e9 users
        uq, n = np.unique(code, return_counts=True)
        self.k_, self.to_, self.cum_ = uq // NC, uq % NC, np.cumsum(n.astype(float))
        st = np.flatnonzero(np.r_[True, self.k_[1:] != self.k_[:-1]])
        self.keys_, self.ptr_ = self.k_[st], np.r_[st, len(self.k_)]
        bk, bc = _mode_by(key, h.tr_to)
        self.best_ = bc                                   # aligned with keys_ (same sorted unique keys)
        return self

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        h, rng = self.h_, np.random.default_rng(self.seed)
        qu, qd, qt, o = _sorted_query(query)
        start = slot_mode(self.train_, query)
        out = start.copy()
        ui = h.index(qu)
        cur = np.full(max(h.N, 1), -1, np.int64)
        last_d = np.full(max(h.N, 1), -1, np.int64)
        for _, rows in _by_slot(qd, qt, o):
            rows = rows[ui[rows] >= 0]
            if not len(rows):
                continue
            u = ui[rows]
            newd = last_d[u] != qd[rows]
            nxt = start[rows].copy()
            go = ~newd
            if go.any():
                key = u[go] * NC + cur[u[go]]
                j = np.searchsorted(self.keys_, key)
                j = np.minimum(j, len(self.keys_) - 1)
                hit = self.keys_[j] == key if len(self.keys_) else np.zeros(len(key), bool)
                step = cur[u[go]].copy()
                if self.decode == "argmax":
                    step[hit] = self.best_[j[hit]]
                else:
                    s, e = self.ptr_[j[hit]], self.ptr_[j[hit] + 1]
                    lo = np.where(s > 0, self.cum_[np.maximum(s - 1, 0)], 0.0)
                    tg = lo + rng.random(len(s)) * (self.cum_[e - 1] - lo)
                    step[hit] = self.to_[np.clip(np.searchsorted(self.cum_, tg, side="right"), s, e - 1)]
                nxt[go] = step
            cur[u], last_d[u] = nxt, qd[rows]
            out[rows] = nxt
        return out


@dataclass
class EPRMobility:
    """Exploration and preferential return agent simulator (Song et al. 2010), started from each user's history.

    rho, gamma : P_new = rho * S^-gamma (paper: rho ~ 0.6, gamma = 0.21).
    alpha, r_min, r_max : exploration jump length P(dr) ~ dr^(-1-alpha) in cells (paper: alpha = 0.55, cutoff
        ~100 km = 200 cells), random direction; a jump leaving the grid is redrawn (up to 10 times, then clipped).
    waiting : 'empirical' (move between two consecutive records with the user's history change rate) or
        'powerlaw' (waiting time ~ dt^(-1-beta_wait) drawn on [1, wait_max] slots and floored, so 1 .. wait_max - 1
        whole slots; paper beta = 0.8, cutoff 17 h = 34 slots).
    day_reset : start each day from one of the user's observed first-of-day cells (TALP port).
    Returns of a simulated new cell use its simulated visit count; history counts stay fixed (TALP port).
    Users without history get the global slot mode of the history. Exploration does not reject cells the user
    already visited (on 40,000 cells this is rare; S still grows by one).
    """

    rho: float = 0.6
    gamma: float = 0.21
    alpha: float = 0.55
    r_min: float = 1.0
    r_max: float = 200.0
    waiting: str = "empirical"
    beta_wait: float = 0.8
    wait_max: int = 34
    day_reset: bool = True
    buffer: int = 16
    seed: int = 0

    def fit(self, train: pd.DataFrame) -> "EPRMobility":
        if self.waiting not in ("empirical", "powerlaw"):
            raise ValueError("waiting must be 'empirical' or 'powerlaw'")
        self.h_ = _History(train)
        self.global_ = global_slot_table(train)
        return self

    def _jump(self, cur, rng):
        a, lo, hi = self.alpha, self.r_min ** -self.alpha, self.r_max ** -self.alpha
        x0, y0 = cur // GRID, cur % GRID
        nx, ny = x0.copy(), y0.copy()
        todo = np.ones(len(cur), bool)
        for _ in range(10):
            k = np.flatnonzero(todo)
            if not len(k):
                break
            r = (lo - rng.random(len(k)) * (lo - hi)) ** (-1.0 / a)
            th = rng.random(len(k)) * 2 * np.pi
            xx = np.rint(x0[k] + r * np.cos(th)).astype(np.int64)
            yy = np.rint(y0[k] + r * np.sin(th)).astype(np.int64)
            ok = (xx >= 0) & (xx < GRID) & (yy >= 0) & (yy < GRID)
            nx[k], ny[k] = np.clip(xx, 0, GRID - 1), np.clip(yy, 0, GRID - 1)
            todo[k[ok]] = False
        return nx * GRID + ny

    def _wait(self, n, rng):
        b, lo, hi = self.beta_wait, 1.0, float(self.wait_max)
        return np.floor((lo ** -b - rng.random(n) * (lo ** -b - hi ** -b)) ** (-1.0 / b)).astype(np.int64)

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        h, rng = self.h_, np.random.default_rng(self.seed)
        qu, qd, qt, o = _sorted_query(query)
        out = self.global_[qt].copy()
        ui = h.index(qu)
        N, Y = max(h.N, 1), self.buffer
        cur = np.full(N, -1, np.int64)
        last_d = np.full(N, -1, np.int64)
        S = h.S.copy() if h.N else np.zeros(1)
        yc = np.full((N, Y), -1, np.int64)
        yn = np.zeros((N, Y))
        ny = np.zeros(N, np.int64)
        due = np.zeros(N, np.int64)
        for s_abs, rows in _by_slot(qd, qt, o):
            rows = rows[ui[rows] >= 0]
            if not len(rows):
                continue
            u = ui[rows]
            newd = (last_d[u] != qd[rows]) & (self.day_reset | (cur[u] < 0))
            first = u[newd]
            if len(first):
                cur[first] = h.first_of_day(first, rng)
                if self.waiting == "powerlaw":
                    due[first] = s_abs + self._wait(len(first), rng)
            dev = u[~newd]
            if len(dev):
                if self.waiting == "empirical":
                    mv = dev[rng.random(len(dev)) < h.m[dev]]
                else:
                    mv = dev[due[dev] <= s_abs]
                    due[mv] = s_abs + self._wait(len(mv), rng)
                if len(mv):
                    p_new = np.minimum(self.rho * np.maximum(S[mv], 1.0) ** -self.gamma, 1.0)
                    ex = rng.random(len(mv)) < p_new
                    e, r = mv[ex], mv[~ex]
                    if len(e):
                        new = self._jump(cur[e], rng)
                        cur[e] = new
                        S[e] += 1.0
                        pos = ny[e] % Y                      # ring buffer of simulated new cells
                        yc[e, pos], yn[e, pos] = new, 1.0
                        ny[e] += 1
                    if len(r):
                        tn = yn[r].sum(1)
                        from_new = rng.random(len(r)) < tn / (h.h_tot[r] + tn)
                        a = r[from_new]
                        if len(a):
                            cw = np.cumsum(yn[a], 1)
                            col = np.minimum((cw < (rng.random(len(a)) * cw[:, -1])[:, None]).sum(1), Y - 1)
                            cur[a] = yc[a, col]
                            yn[a, col] += 1.0
                        b = r[~from_new]
                        if len(b):
                            cur[b] = h.weighted_stay(b, rng)
            last_d[u] = qd[rows]
            out[rows] = cur[u]
        return out


def mobility_baselines(train: pd.DataFrame, query: pd.DataFrame, *, global_table: np.ndarray | None = None,
                       epr: bool = True, markov: bool = True, seed: int = 0) -> dict[str, np.ndarray]:
    """{name: cells aligned with query rows} for the standard baseline set."""
    out = {"slot_mode": slot_mode(train, query, fallback=global_table),
           "global_slot_mode": global_slot_mode(train, query, global_table)}
    if markov:
        out["markov_argmax"] = MarkovMobility("argmax", seed).fit(train).predict(query)
        out["markov_sample"] = MarkovMobility("sample", seed).fit(train).predict(query)
    if epr:
        out["epr"] = EPRMobility(seed=seed).fit(train).predict(query)
    return out
