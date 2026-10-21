"""Opinion drift between two survey waves: bounded-confidence ABM + the baselines that beat it.

Ported from the TALP prep research code `abm/kod/c2_abm.py` (ABM) and `llmabm/kod/c2/c2.py` (baselines).

Task: respondents answered K-category items in wave A; predict each group's (segment's) answer distribution in
wave B. Methods:

* ``persistence``     : wave-A distribution unchanged.
* ``mean_shift``      : wave-A distribution shifted by the group's mean shift on the OTHER items (leave one item out).
* ``transition``      : wave-A distribution times a person-level transition matrix estimated on the other items with
  the same scale (leave one item out).
* :class:`BoundedConfidenceABM` (Deffuant-Weisbuch): each respondent is an agent with a latent position from the wave-A
  category; for `n_steps` steps it meets a weighted partner (from its own homophily class with probability h) and
  moves mu of the way when |difference| < eps; plus a group shock kappa * delta (delta = mean shift of the other items)
  and noise sigma. Parameters are chosen on a grid by W1 on the other items (leave one item out); the same calibration
  forced to mu = 0 is reported as ``abm_shock_noise`` (structure control).

Measured (makro/abm/RAPOR.md, C2; CES 2020 -> 2022 / 2024, 4 segments x 7 items, 3 seeds, W1 lower is better):
mean shift 0.111 / 0.131, persistence 0.117 / 0.127, ABM mu=0 0.126 / 0.129, bounded-confidence ABM 0.133 / 0.145.
The ABM LOST to both baselines; interaction fitted the training items but did not transfer (2022: ABM - mean shift
+0.023 [+0.012, +0.036]). Use persistence / mean shift as the baseline; the ABM is here for honest comparison and for
settings that have real network / interaction data.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_GRID = list(itertools.product([0, 0.05, 0.15, 0.3], [0.2, 0.5, 1.0], [0.5, 0.9], [0, 0.05, 0.1, 0.2],
                                      [0, 0.5, 1, 1.5]))   # (mu, eps, h, sigma, kappa), research grid


# ------------------------------------------------------------------ distribution helpers
def distribution(x, w, k):
    """Weighted category distribution of answers coded 1..k (NaN dropped)."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    i = ~np.isnan(x)
    p = np.bincount(x[i].astype(int) - 1, weights=w[i], minlength=k)[:k]
    s = p.sum()
    return p / s if s > 0 else np.full(k, 1.0 / k)


def w1(p, q):
    """Normalised Wasserstein-1 on an ordinal scale (0-1)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(np.abs(np.cumsum(p - q))[:-1].sum() / (len(p) - 1))


def tvd(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def mean_position(p):
    """Mean on a 0-1 scale."""
    p = np.asarray(p, float)
    return float((np.arange(len(p)) / (len(p) - 1) * p).sum())


def shift_distribution(p, d):
    """Shift a distribution by d on the 0-1 scale (linear mass transfer between neighbouring categories)."""
    k = len(p)
    v = np.clip(np.arange(k) / (k - 1) + d, 0, 1) * (k - 1)
    q = np.zeros(k)
    for i, x in enumerate(v):
        lo = int(np.floor(x))
        hi = min(lo + 1, k - 1)
        f = x - lo
        q[lo] += p[i] * (1 - f)
        q[hi] += p[i] * f
    return q


# ------------------------------------------------------------------ panel
@dataclass
class OpinionPanel:
    """Respondent-level two-wave panel.

    df      : one row per respondent.
    items   : {item: (col_wave_a, col_wave_b, k)}; answers coded 1..k, NaN allowed.
    group   : column with the segment whose distribution is predicted.
    weight  : optional survey weight column.
    homophily: optional column for partner choice in the ABM (e.g. party); defaults to `group`.
    """
    df: pd.DataFrame
    items: dict
    group: str = "segment"
    weight: str | None = None
    homophily: str | None = None
    groups: list = field(default_factory=list)

    def __post_init__(self):
        if not self.groups:
            self.groups = sorted(self.df[self.group].dropna().unique().tolist())

    def w(self, rows=None):
        d = self.df if rows is None else self.df.loc[rows]
        return d[self.weight].to_numpy(float) if self.weight else np.ones(len(d))

    def dist(self, item, wave, g):
        a, b, k = self.items[item]
        col = a if wave == "a" else b
        h = self.df[self.df[self.group] == g]
        return distribution(h[col], self.w(h.index), k)

    def truth(self):
        """{(item, group): (wave-A dist, wave-B dist)}."""
        return {(m, g): (self.dist(m, "a", g), self.dist(m, "b", g)) for m in self.items for g in self.groups}

    def group_shift(self, g, exclude):
        """Mean 0-1 shift of group g over the items not in `exclude`."""
        v = [mean_position(self.dist(x, "b", g)) - mean_position(self.dist(x, "a", g))
             for x in self.items if x not in exclude]
        return float(np.mean(v)) if v else 0.0


def _rows(item, group, method, q):
    return [dict(item=item, group=group, method=method, category=c + 1, p=float(v)) for c, v in enumerate(q)]


def opinion_baselines(panel: OpinionPanel, methods=("persistence", "mean_shift", "transition")) -> pd.DataFrame:
    """Long frame item, group, method, category, p for the LLM-free baselines (leave one item out)."""
    out = []
    for g in panel.groups:
        h = panel.df[panel.df[panel.group] == g]
        wt = panel.w(h.index)
        for m, (a, b, k) in panel.items.items():
            pa = distribution(h[a], wt, k)
            if "persistence" in methods:
                out += _rows(m, g, "persistence", pa)
            if "mean_shift" in methods:
                out += _rows(m, g, "mean_shift", shift_distribution(pa, panel.group_shift(g, {m})))
            if "transition" in methods:
                T = np.zeros((k, k))
                for x, (xa, xb, kx) in panel.items.items():
                    if x == m or kx != k:
                        continue
                    i = (h[xa].notna() & h[xb].notna()).to_numpy()
                    np.add.at(T, (h[xa].to_numpy()[i].astype(int) - 1, h[xb].to_numpy()[i].astype(int) - 1), wt[i])
                T = (T + 1e-3) / (T + 1e-3).sum(1, keepdims=True)
                out += _rows(m, g, "transition", pa @ T)
    return pd.DataFrame(out)


class _Item:
    """One item's wave-A respondents: latent start, homophily class, weight, group."""

    def __init__(self, panel: OpinionPanel, item, seed):
        a, _, k = panel.items[item]
        h = panel.df[panel.df[a].notna()]
        self.k = k
        self.grp = h[panel.group].to_numpy()
        self.w = panel.w(h.index)
        self.cls = h[panel.homophily or panel.group].to_numpy()
        rng = np.random.default_rng(seed)
        # uniform inside the category box [c/k, (c+1)/k)
        self.u0 = (h[a].to_numpy(int) - 1 + rng.random(len(h))) / k
        self.grp_i = {g: np.flatnonzero(self.grp == g) for g in panel.groups}
        self.pool = {}
        for c in np.unique(self.cls):
            same = np.flatnonzero(self.cls == c)
            other = np.flatnonzero(self.cls != c)
            self.pool[c] = tuple((i, np.cumsum(self.w[i]) / self.w[i].sum()) if len(i) else (i, None)
                                 for i in (same, other))

    def partners(self, rng, same):
        """Weight-proportional partner per agent: own class if same[i], otherwise another class."""
        j = np.arange(len(self.u0))
        for c, pools in self.pool.items():
            mine = self.cls == c
            for which, sel in ((0, mine & same), (1, mine & ~same)):
                n = int(sel.sum())
                idx, cw = pools[which]
                if n and cw is not None:
                    j[sel] = idx[np.minimum(np.searchsorted(cw, rng.random(n)), len(idx) - 1)]
        return j

    def simulate(self, mu, eps, h, sigma, kappa, delta, seed, n_steps=24):
        """delta: {group: shift on 0-1}. Returns {group: category distribution}."""
        rng = np.random.default_rng(seed)
        u = self.u0.copy()
        d = np.zeros(len(u))
        for g, i in self.grp_i.items():
            d[i] = delta.get(g, 0.0)
        for _ in range(n_steps):
            if mu > 0:
                j = self.partners(rng, rng.random(len(u)) < h)
                diff = u[j] - u
                u = u + mu * diff * (np.abs(diff) < eps)
            u = u + kappa * d / n_steps
            if sigma > 0:
                u = u + rng.normal(0, sigma / np.sqrt(n_steps), len(u))
            u = np.clip(u, 0, 1 - 1e-9)
        c = np.floor(u * self.k).astype(int)
        return {g: (np.bincount(c[i], weights=self.w[i], minlength=self.k) / self.w[i].sum() if len(i)
                    else np.full(self.k, 1.0 / self.k)) for g, i in self.grp_i.items()}


class BoundedConfidenceABM:
    """Deffuant-Weisbuch opinion ABM with leave-one-item-out grid calibration (research `abm_sinirli_guven`)."""
    name = "abm_bounded_confidence"
    kind = "method"

    def __init__(self, grid=None, n_steps=24, seed=20261021):
        self.grid = list(grid) if grid is not None else DEFAULT_GRID
        self.n_steps, self.seed = n_steps, seed
        self.calibration: list[dict] = []

    def fit_predict(self, panel: OpinionPanel) -> pd.DataFrame:
        """Returns long predictions for `abm_bounded_confidence` and the mu=0 control `abm_shock_noise`."""
        items = list(panel.items)
        sims = {m: _Item(panel, m, self.seed + i) for i, m in enumerate(items)}
        truth = panel.truth()
        out, self.calibration = [], []
        zero_ok = [a for a in self.grid if a[0] == 0]
        for m in items:
            train = [x for x in items if x != m]
            dx = {x: {g: panel.group_shift(g, {x, m}) for g in panel.groups} for x in train}
            scores = []
            for (mu, eps, h, sg, ka) in self.grid:
                if mu == 0 and zero_ok and (eps, h) != (zero_ok[0][1], zero_ok[0][2]):
                    scores.append(np.inf)          # eps, h have no effect at mu = 0; skip duplicates
                    continue
                w = []
                for x in train:
                    q = sims[x].simulate(mu, eps, h, sg, ka, dx[x], self.seed + 7, self.n_steps)
                    w += [w1(q[g], truth[(x, g)][1]) for g in panel.groups]
                scores.append(float(np.mean(w)) if w else np.inf)
            best = int(np.argmin(scores))
            mu, eps, h, sg, ka = self.grid[best]
            delta = {g: panel.group_shift(g, {m}) for g in panel.groups}
            q = sims[m].simulate(mu, eps, h, sg, ka, delta, self.seed + 7, self.n_steps)
            for g in panel.groups:
                out += _rows(m, g, self.name, q[g])
            rec = dict(item=m, mu=mu, eps=eps, h=h, sigma=sg, kappa=ka, w1_train=scores[best])
            z = [(s, a) for s, a in zip(scores, self.grid) if a[0] == 0 and np.isfinite(s)]
            if z:
                s0, a0 = min(z, key=lambda t: t[0])
                q0 = sims[m].simulate(*a0, delta, self.seed + 7, self.n_steps)
                for g in panel.groups:
                    out += _rows(m, g, "abm_shock_noise", q0[g])
                rec.update(mu0_sigma=a0[3], mu0_kappa=a0[4], mu0_w1_train=s0)
            self.calibration.append(rec)
        return pd.DataFrame(out)


def score_opinion(pred: pd.DataFrame, panel: OpinionPanel):
    """W1 / TVD of long predictions against wave-B truth. Returns (summary per method, per-cell frame)."""
    truth = panel.truth()
    cells = []
    for (m, g, meth), d in pred.groupby(["item", "group", "method"]):
        q = d.sort_values("category")["p"].to_numpy()
        pa, pb = truth[(m, g)]
        cells.append(dict(item=m, group=g, method=meth, w1=w1(q, pb), tvd=tvd(q, pb),
                          shift_pred=mean_position(q) - mean_position(pa), shift_true=mean_position(pb) - mean_position(pa)))
    cells = pd.DataFrame(cells)
    summ = cells.groupby("method").agg(W1=("w1", "mean"), TVD=("tvd", "mean"), n=("w1", "size")).reset_index()
    return summ.sort_values("W1").reset_index(drop=True), cells


def to_eval_frame(pred: pd.DataFrame, panel: OpinionPanel) -> pd.DataFrame:
    """Distribution EvalFrame (truth = wave-B distribution, pred_<method> = predicted distribution), one row per
    item x group, item -> `item`, group -> `segment`."""
    truth = panel.truth()
    wide = {}
    for (m, g, meth), d in pred.groupby(["item", "group", "method"]):
        wide.setdefault((m, g), {})[f"pred_{meth}"] = d.sort_values("category")["p"].to_numpy().tolist()
    rows = [dict(item=m, segment=g, unit=f"{m}|{g}", truth=truth[(m, g)][1].tolist(), **v)
            for (m, g), v in wide.items()]
    return pd.DataFrame(rows)
