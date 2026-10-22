"""risky_choice: aggregate choice rates between two gambles, choices13k (Peterson et al. 2021) and CPC18 (Plonsky et al.).

    from simengine.bench.tasks import risky_choice as R
    task = R.Choices13k(seed=0)                     # bench name "choices13k"; random 90/10 split of 9,831 problems
    data = task.load("/path/to/catalogue")          # or $SIMENGINE_DATA
    frame = task.run(data=data)                     # half, train_mean, ev_max, ev_logit, pt, cpt, gbdt_objective, beast_sd
    task.score_all(frame)                           # mse (primary), completeness, pref_acc
    R.CPC18().run(data_root=...)                    # bench name "cpc18"; 210 train / 60 test tasks x 5 blocks
    R.beast_predict(dist_a, dist_b, amb, corr)      # (P, 5) B-rates of BEAST.sd, vectorised over problems and agents

Each unit is one choice problem (choices13k) or one problem x block of five trials (CPC18); `truth` is the observed
rate of choosing Option B. The metric is the papers' MSE between predicted and observed rates (CPC15/CPC18: the mean
over the 5 blocks x 60 test problems; the competitions report 100 x MSE), with a cluster bootstrap over problems.
`completeness` is Fudenberg et al.'s (MSE_random - MSE) / (MSE_random - MSE_irreducible), MSE_random from always 0.5
and MSE_irreducible the mean squared standard error of the observed rates (Plonsky et al., Methods).

Baselines: constant 0.5 and train mean; deterministic and logit expected value; separable prospect theory and
cumulative prospect theory (power utility with one curvature, loss aversion, Goldstein-Einhorn weighting, logit choice;
the CPC18 SI specification) fitted by least squares on the train rows; a gradient-boosted tree on the objective task
dimensions; and BEAST (Erev et al. 2017) in its CPC15 form (`beast`, CPC18 split only) and the CPC18 baseline BEAST.sd
(`beast_sd`), both with the organisers' published parameters (no fitting). BEAST is a vectorised re-implementation
of the organisers' per-agent simulation; see docs/modules/bench/risky_choice.md for the deviations.
"""
from __future__ import annotations

import json
import os
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, DataNotFound, Metric, need, register
from ..stats import cluster_bootstrap

Dist = tuple[np.ndarray, np.ndarray]  # (outcomes ascending, probabilities)

SHAPES = {"-": "-", "Symm": "Symm", "R-skew": "R-skew", "L-skew": "L-skew", 0: "-", 1: "Symm", 2: "R-skew",
          3: "L-skew"}


# ---------------------------------------------------------------------- gamble distributions
def merge_dist(x: Sequence[float], p: Sequence[float]) -> Dist:
    """Sort outcomes ascending, add up equal outcomes, drop zero-probability ones."""
    x, p = np.asarray(x, float), np.asarray(p, float)
    keep = p > 0
    u, inv = np.unique(x[keep], return_inverse=True)
    return u, np.bincount(inv, weights=p[keep], minlength=len(u))


def cpc_distribution(H: float, pH: float, L: float, shape: Any = "-", num: int = 1) -> Dist:
    """Full outcome distribution of a CPC15/CPC18 option (Erev et al. 2017, Appendix A; organisers' CPC18_getDist).

    With probability pH the option pays a lottery with expected value H, otherwise L. shape '-' (num 1): the lottery
    is H itself. 'Symm': num outcomes H - k/2 + i, Binomial(k = num - 1, 0.5) weights. 'R-skew' / 'L-skew':
    outcomes H + c +- 2^i with weights 2^-i (i = 1..num, last weight doubled), c = -1 - num / 1 + num."""
    shape = SHAPES.get(shape, shape)
    num = int(num)
    if shape == "-" or num <= 1:
        return merge_dist([L, H], [1 - pH, pH])
    i = np.arange(num)
    if shape == "Symm":
        from scipy.stats import binom
        k = num - 1
        hx, hp = H - k / 2 + i, binom.pmf(i, k, 0.5)
    elif shape in ("R-skew", "L-skew"):
        j = i + 1
        c, sgn = (-1 - num, 1) if shape == "R-skew" else (1 + num, -1)
        hx, hp = H + c + sgn * 2.0 ** j, 0.5 ** j
        hp[-1] *= 2
    else:
        raise ValueError(f"unknown lottery shape {shape!r}")
    return merge_dist(np.r_[hx, L], np.r_[pH * hp, 1 - pH])


def pad_dists(dists: Sequence[Dist], k: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, P, n): (N, K) outcomes padded with the row's maximum (probability 0, so order and CDFs are unchanged)."""
    n = np.array([len(d[0]) for d in dists], int)
    k = int(max(n.max(), k or 1))
    X = np.empty((len(dists), k))
    P = np.zeros((len(dists), k))
    for r, (x, p) in enumerate(dists):
        X[r, :len(x)], X[r, len(x):], P[r, :len(x)] = x, x[-1], p
    return X, P, n


def _pad_pair(dist_a, dist_b):
    XA, PA, nA = pad_dists(dist_a)
    XB, PB, nB = pad_dists(dist_b)
    k = max(XA.shape[1], XB.shape[1])
    if XA.shape[1] < k:
        XA, PA, nA = pad_dists(dist_a, k)
    if XB.shape[1] < k:
        XB, PB, nB = pad_dists(dist_b, k)
    return XA, PA, nA, XB, PB, nB


def _sample_idx(cum: np.ndarray, u: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Index of the first cumulative probability >= u (the organisers' distSample). cum (c, K) or (c, S, K)."""
    c = cum[:, None, :] if cum.ndim == 2 else cum
    return np.minimum((c < u[..., None]).sum(-1), (n - 1)[:, None])


def p_better_comonotone(XA, PA, XB, PB) -> tuple[np.ndarray, np.ndarray]:
    """Exact P(A > B), P(B > A) when both options are drawn with the same uniform number (the CPC18 BEAST.sd code
    estimates this with 100,000 draws and corr=1)."""
    cA, cB = np.cumsum(PA, 1), np.cumsum(PB, 1)
    bp = np.sort(np.concatenate([np.zeros((len(XA), 1)), np.minimum(cA, 1), np.minimum(cB, 1)], 1), 1)
    mid, w = (bp[:, 1:] + bp[:, :-1]) / 2, np.diff(bp, axis=1)
    nA, nB = (PA > 0).sum(1), (PB > 0).sum(1)
    r = np.arange(len(XA))[:, None]
    qa = XA[r, np.minimum((cA[:, None, :] < mid[..., None]).sum(-1), (nA - 1)[:, None])]
    qb = XB[r, np.minimum((cB[:, None, :] < mid[..., None]).sum(-1), (nB - 1)[:, None])]
    return (w * (qa > qb)).sum(1), (w * (qb > qa)).sum(1)


def fosd(XA, PA, XB, PB, tol: float = 1e-9) -> np.ndarray:
    """True where one option first-order stochastically dominates the other (CDFs ordered everywhere, not equal)."""
    pts = np.concatenate([XA, XB], 1)
    FA = (PA[:, None, :] * (XA[:, None, :] <= pts[..., None] + tol)).sum(-1)
    FB = (PB[:, None, :] * (XB[:, None, :] <= pts[..., None] + tol)).sum(-1)
    a = (FA <= FB + tol).all(1) & (FA < FB - tol).any(1)
    b = (FB <= FA + tol).all(1) & (FB < FA - tol).any(1)
    return a | b


# ---------------------------------------------------------------------- value models
def expected_value(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    return (np.asarray(X) * np.asarray(P)).sum(-1)


def gw_weight(p: np.ndarray, gamma: float, delta: float) -> np.ndarray:
    """Goldstein-Einhorn / Lattimore two-parameter weighting w(p) = d p^g / (d p^g + (1 - p)^g)."""
    p = np.clip(p, 0.0, 1.0)
    a = delta * p ** gamma
    with np.errstate(invalid="ignore", divide="ignore"):
        w = a / (a + (1 - p) ** gamma)
    return np.where(p <= 0, 0.0, np.where(p >= 1, 1.0, w))


def prospect_value(X: np.ndarray, P: np.ndarray, alpha: float, lam: float, gamma: float, delta: float,
                   cumulative: bool = True) -> np.ndarray:
    """Weighted value of each row's gamble. u(x) = x^a (x >= 0), -lam (-x)^a (x < 0). cumulative=True: rank-dependent
    decision weights (Tversky & Kahneman 1992, same w for gains and losses as in the CPC18 SI); False: separable
    w(p_i) per outcome (Kahneman & Tversky 1979 form). X must be ascending per row (pad_dists output)."""
    X, P = np.asarray(X, float), np.asarray(P, float)
    ax = np.abs(X) ** alpha
    u = np.where(X >= 0, ax, -lam * ax)
    if not cumulative:
        return (gw_weight(P, gamma, delta) * u).sum(-1)
    le = np.minimum(np.cumsum(P, -1), 1.0)  # P(X <= x_i)
    lt = np.maximum(le - P, 0.0)  # P(X < x_i)
    wl = gw_weight(le, gamma, delta) - gw_weight(lt, gamma, delta)
    wg = gw_weight(1 - lt, gamma, delta) - gw_weight(1 - le, gamma, delta)
    return (np.where(X < 0, wl, wg) * u).sum(-1)


def logit_b(va: np.ndarray, vb: np.ndarray, mu: float) -> np.ndarray:
    """P(B) = e^(mu vB) / (e^(mu vA) + e^(mu vB))."""
    z = np.clip(mu * (np.asarray(vb) - np.asarray(va)), -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


PT_BOUNDS = {"alpha": (0.05, 1.5), "lam": (0.1, 5.0), "gamma": (0.1, 2.5), "delta": (0.05, 5.0), "mu": (1e-4, 20.0)}
PT_STARTS = ({"alpha": 0.9, "lam": 1.2, "gamma": 0.9, "delta": 0.9, "mu": 0.25},
             {"alpha": 0.6, "lam": 2.0, "gamma": 0.6, "delta": 0.8, "mu": 0.5},
             {"alpha": 1.0, "lam": 1.0, "gamma": 1.0, "delta": 1.0, "mu": 0.05})


def fit_prospect(XA, PA, XB, PB, y, *, cumulative: bool = True, weight: np.ndarray | None = None) -> dict:
    """Least-squares fit of (alpha, lam, gamma, delta, mu) to observed B-rates; L-BFGS-B from three starts."""
    from scipy.optimize import minimize
    y = np.asarray(y, float)
    w = np.ones_like(y) if weight is None else np.asarray(weight, float)
    names = list(PT_BOUNDS)

    def loss(v):
        a, l_, g, d, mu = v
        pb = logit_b(prospect_value(XA, PA, a, l_, g, d, cumulative), prospect_value(XB, PB, a, l_, g, d, cumulative),
                     mu)
        return float(np.sum(w * (pb - y) ** 2) / w.sum())

    best = None
    for s in PT_STARTS:
        r = minimize(loss, [s[k] for k in names], method="L-BFGS-B", bounds=[PT_BOUNDS[k] for k in names])
        if best is None or r.fun < best.fun:
            best = r
    return {**dict(zip(names, map(float, best.x))), "train_mse": float(best.fun), "cumulative": cumulative}


def predict_prospect(params: dict, XA, PA, XB, PB) -> np.ndarray:
    a, l_, g, d, mu = (params[k] for k in PT_BOUNDS)
    c = params.get("cumulative", True)
    return logit_b(prospect_value(XA, PA, a, l_, g, d, c), prospect_value(XB, PB, a, l_, g, d, c), mu)


def fit_ev_logit(ev_a, ev_b, y) -> float:
    """Sensitivity mu of P(B) = logistic(mu (EV_B - EV_A)), least squares over log mu."""
    from scipy.optimize import minimize_scalar
    y = np.asarray(y, float)
    r = minimize_scalar(lambda lm: float(np.mean((logit_b(ev_a, ev_b, np.exp(lm)) - y) ** 2)), bounds=(-12, 4),
                        method="bounded")
    return float(np.exp(r.x))


# ---------------------------------------------------------------------- BEAST
BEAST_PARAMS = {
    # Erev et al. (2017) Psychological Review 124:369-409, fitted to the 90 CPC15 calibration problems
    "original": {"sigma": 7.0, "kappa": 3, "beta": 2.6, "gamma": 0.5, "psi": 0.07, "theta": 1.0},
    # CPC18 baseline BEAST.sd (organisers' code): subjective dominance, complexity noise, ambiguity learning weight
    "sd": {"sigma": 13.0, "kappa": 3, "beta": 1.4, "gamma": 1.0, "psi": 0.25, "theta": 0.7, "sigma_comp": 35.0,
           "wamb": 0.25},
}


def beast_predict(dist_a: Sequence[Dist], dist_b: Sequence[Dist], amb: Sequence[int] | None = None,
                  corr: Sequence[int] | None = None, *, version: str = "sd", n_sims: int = 5000, n_trials: int = 25,
                  first_feedback: int = 6, n_blocks: int = 5, seed: int = 0, chunk: int = 32,
                  params: dict | None = None) -> np.ndarray:
    """Mean B-rate per block, (P, n_blocks), of n_sims simulated BEAST agents per problem.

    version 'original' = BEAST of CPC15 (Erev et al. 2017), 'sd' = BEAST.sd, the CPC18 baseline. Trials 1..n_trials;
    feedback (obtained and forgone payoffs) follows every choice from trial `first_feedback` on (CPC: 25, 6, 5 blocks).
    Agents draw traits sigma_i ~ U(0, sigma), kappa_i ~ U{1..kappa}, beta_i, gamma_i, psi_i, theta_i (and w_amb,i)
    ~ U(0, .); each choice compares best-estimate EVs plus the mean of kappa_i mental draws from four sampling tools
    (unbiased / uniform / contingent pessimism / sign) plus N(0, sigma_t) noise."""
    prm = {**BEAST_PARAMS[version], **(params or {})}
    P = len(dist_a)
    amb = np.zeros(P, int) if amb is None else np.asarray(amb, int)
    corr = np.zeros(P, int) if corr is None else np.asarray(corr, int)
    XA, PA, nA, XB, PB, nB = _pad_pair(dist_a, dist_b)
    rng = np.random.default_rng(seed)
    out = np.empty((P, n_blocks))
    for s in range(0, P, chunk):
        e = min(P, s + chunk)
        out[s:e] = _beast_chunk(XA[s:e], PA[s:e], nA[s:e], XB[s:e], PB[s:e], nB[s:e], amb[s:e] == 1, corr[s:e],
                                prm, version == "sd", n_sims, n_trials, first_feedback, n_blocks, rng)
    return out


def _beast_chunk(XA, PA, nA, XB, PB, nB, amb, corr, prm, sd, S, T, F, NB, rng):
    c, K = XA.shape
    r = np.arange(c)[:, None]
    cumA, cumB = np.cumsum(PA, 1), np.cumsum(PB, 1)
    minA, minB = XA[:, 0], XB[:, 0]
    maxA, maxB = XA[r[:, 0], nA - 1], XB[r[:, 0], nB - 1]
    max_out = np.maximum(maxA, maxB)
    sign_max = np.sign(max_out)
    same = np.sign(minA) == np.sign(minB)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.minimum(np.abs(minA), np.abs(minB)) / np.maximum(np.abs(minA), np.abs(minB))
    ratio_min = np.where(minA == minB, 1.0, np.where(same, ratio, 0.0))
    rng_ = (max_out - np.minimum(minA, minB))[:, None]
    ev_a, ev_b = expected_value(XA, PA), expected_value(XB, PB)
    mask_b = np.arange(K)[None, :] < nB[:, None]
    uev_a = (XA * (np.arange(K)[None, :] < nA[:, None])).sum(1) / nA
    uev_b = (XB * mask_b).sum(1) / nB
    amb = amb & (nB > 1)

    # agent traits
    U = lambda: rng.random((c, S))  # noqa: E731
    if sd:
        sig_max = np.where((np.maximum(nA, nB) > 2) & (np.minimum(nA, nB) > 1), prm["sigma_comp"], prm["sigma"])
    else:
        sig_max = np.full(c, prm["sigma"])
    sigma = sig_max[:, None] * U()
    kappa = rng.integers(1, int(prm["kappa"]) + 1, (c, S))
    beta, gama, psi, theta = prm["beta"] * U(), prm["gamma"] * U(), prm["psi"] * U(), prm["theta"] * U()
    wamb = prm.get("wamb", 0.0) * U() if sd else None

    bev_a = np.broadcast_to(ev_a[:, None], (c, S))
    bev_b = np.where(amb[:, None], (1 - psi) * (uev_b + ev_a)[:, None] / 2 + psi * minB[:, None], ev_b[:, None])
    # estimated probabilities of B (ambiguity: P(min) solves BEV_B, the rest share the remainder equally)
    cum_est = np.broadcast_to(cumB[:, None, :], (c, S, K))
    if amb.any():
        rest = ((XB * mask_b).sum(1) - minB) / np.maximum(nB - 1, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.clip((bev_b - rest[:, None]) / (minB - rest)[:, None], 0, 1)
        pe = np.where(np.arange(K)[None, None, :] == 0, t[..., None],
                      (1 - t[..., None]) / np.maximum(nB - 1, 1)[:, None, None]) * mask_b[:, None, :]
        cum_est = np.where(amb[:, None, None], np.cumsum(pe, -1), cum_est)

    dom_support = (minA > maxB) | (minB > maxA)
    if sd:
        pa_b, pb_b = p_better_comonotone(XA, PA, XB, PB)
        subj = np.where((ev_a > ev_b) & (uev_a >= uev_b), 1 - pb_b, np.where((ev_a < ev_b) & (uev_a <= uev_b),
                                                                              1 - pa_b, 0.0))
        subj = np.where(amb, 0.0, subj)
        subj = np.where(dom_support, 1.0, subj)
        sigma = sigma * (1 - subj)[:, None]
    else:
        subj = None
        sigma = np.where(fosd(XA, PA, XB, PB)[:, None], 0.0, sigma)  # "trivial" problems: no error term
    sigmat = sigma.copy()

    n_obs = max(T - F + 1, 0)
    obs_a, obs_b = np.zeros((c, S, max(n_obs, 1))), np.zeros((c, S, max(n_obs, 1)))
    n_a_win, n_b_win, sum_b = np.zeros((c, S)), np.zeros((c, S)), np.zeros((c, S))
    block = T // NB
    rates = np.zeros((c, S, NB))
    pess_ok = (sign_max > 0)[:, None] & (ratio_min[:, None] < gama)
    nfeed = 0
    for trial in range(T):
        pbias = beta / (beta + 1 + (nfeed ** theta if nfeed else 0.0))
        st_a, st_b = np.zeros((c, S)), np.zeros((c, S))
        for k in range(int(prm["kappa"])):
            r0, r1 = U(), U()
            ia, ib = (np.minimum(np.ceil(r1 * nA[:, None]) - 1, (nA - 1)[:, None]).clip(0).astype(int),
                      np.minimum(np.ceil(r1 * nB[:, None]) - 1, (nB - 1)[:, None]).clip(0).astype(int))
            uni_a, uni_b = XA[r, ia], XB[r, ib]
            if nfeed == 0:
                ja, jb = _sample_idx(cumA, r1, nA), _sample_idx(cum_est, r1, nB)
                unb_a, unb_b = XA[r, ja], XB[r, jb]
            else:
                j = np.minimum(np.ceil(r1 * nfeed) - 1, nfeed - 1).clip(0).astype(int)
                unb_a = np.take_along_axis(obs_a, j[..., None], -1)[..., 0]
                unb_b = np.take_along_axis(obs_b, j[..., None], -1)[..., 0]
            sgn_a, sgn_b = rng_ * np.sign(unb_a), rng_ * np.sign(unb_b)
            cp_a = np.where(pess_ok, minA[:, None], uni_a)
            cp_b = np.where(pess_ok, minB[:, None], uni_b)
            tool = np.where(r0 > pbias, 0, np.where(r0 > 2 / 3 * pbias, 1, np.where(r0 > pbias / 3, 2, 3)))
            oa = np.choose(tool, [unb_a, uni_a, cp_a, sgn_a])
            ob = np.choose(tool, [unb_b, uni_b, cp_b, sgn_b])
            on = k < kappa
            st_a += np.where(on, oa, 0.0)
            st_b += np.where(on, ob, 0.0)
        v = (bev_a - bev_b) + (st_a - st_b) / kappa + sigmat * rng.standard_normal((c, S))
        dec = (v < 0) | ((v == 0) & (U() < 0.5))
        if trial // block < NB:
            rates[..., trial // block] += dec
        if trial >= F - 1 and nfeed < n_obs:
            u = U()
            oa = XA[r, _sample_idx(cumA, u, nA)]
            ub = np.where(corr[:, None] == 1, u, np.where(corr[:, None] == -1, 1 - u, U()))
            ob = XB[r, _sample_idx(cumB, ub, nB)]
            obs_a[..., nfeed], obs_b[..., nfeed] = oa, ob
            nfeed += 1
            if sd:
                n_a_win += oa >= ob
                n_b_win += ob >= oa
                sum_b += ob
                if amb.any():
                    bev_b = np.where(amb[:, None], (1 - wamb) * bev_b + wamb * ob, bev_b)
                    avg_b = sum_b / nfeed
                    upd = (amb & (subj != 1))[:, None]
                    ca = (bev_a > avg_b) & (uev_a >= uev_b)[:, None]
                    cb = (bev_a < avg_b) & (uev_a <= uev_b)[:, None]
                    fac = np.where(ca, 1 - n_a_win / nfeed, np.where(cb, 1 - n_b_win / nfeed, 1.0))
                    sigmat = np.where(upd, sigmat * fac, sigmat)
            elif amb.any():
                w = 1.0 / n_obs
                bev_b = np.where(amb[:, None], (1 - w) * bev_b + w * ob, bev_b)
    return (rates / block).mean(1)


# ---------------------------------------------------------------------- metrics
def _arrays(frame, col):
    return frame["truth"].to_numpy(float), frame[col].to_numpy(float), frame["group"].to_numpy()


def _mse(task, frame, col, B, seed):
    y, p, g = _arrays(frame, col)
    return cluster_bootstrap(lambda i: float(np.mean((p[i] - y[i]) ** 2)), g, B=B, seed=seed)


def completeness(y: np.ndarray, p: np.ndarray, se2: np.ndarray) -> float:
    """(MSE_random - MSE) / (MSE_random - MSE_irreducible); random = always 0.5, irreducible = mean se^2."""
    m_r, m = np.mean((0.5 - y) ** 2), np.mean((p - y) ** 2)
    return float((m_r - m) / (m_r - np.mean(se2)))


def _completeness(task, frame, col, B, seed):
    y, p, g = _arrays(frame, col)
    se2 = frame["se2"].to_numpy(float)
    return cluster_bootstrap(lambda i: completeness(y[i], p[i], se2[i]), g, B=B, seed=seed)


def _pref_acc(task, frame, col, B, seed):
    """Share of units where prediction and data agree on the majority option (0.5 on either side counts half)."""
    y, p, g = _arrays(frame, col)
    s = np.where((p == 0.5) | (y == 0.5), 0.5, ((p > 0.5) == (y > 0.5)).astype(float))
    return cluster_bootstrap(lambda i: float(s[i].mean()), g, B=B, seed=seed)


METRICS = (Metric("mse", False, _mse, "mean squared error of the B-rate; cluster bootstrap over problems"),
           Metric("completeness", True, _completeness,
                  "(MSE_0.5 - MSE) / (MSE_0.5 - mean se^2) (Fudenberg et al.; Plonsky et al. Methods)"),
           Metric("pref_acc", True, _pref_acc, "agreement on the majority option (Peterson et al. 'decision "
                                               "preference accuracy', Plonsky et al. 'P(agree)')"))


# ---------------------------------------------------------------------- tasks
class _RiskyChoice(BenchTask):
    kind = "continuous"
    metrics = METRICS
    objective: tuple[str, ...] = ()

    def _arrays(self, df):
        return _pad_pair(list(df["dist_a"]), list(df["dist_b"]))

    def _beast(self, data: TaskData, version: str) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def baselines(self, data: TaskData) -> dict:
        tr, te = data.train, data.test
        XA, PA, _, XB, PB, _ = self._arrays(tr)
        TA, TPA, _, TB, TPB, _ = self._arrays(te)
        ya = tr["y"].to_numpy(float)
        ev_a, ev_b = expected_value(TA, TPA), expected_value(TB, TPB)
        mu = fit_ev_logit(expected_value(XA, PA), expected_value(XB, PB), ya)
        out = {"half": np.full(len(te), 0.5), "train_mean": np.full(len(te), ya.mean()),
               "ev_max": np.where(ev_b > ev_a, 1.0, np.where(ev_b < ev_a, 0.0, 0.5)),
               "ev_logit": logit_b(ev_a, ev_b, mu)}
        self.fitted_ = {"ev_logit": {"mu": mu}}
        for name, cum in (("pt", False), ("cpt", True)):
            prm = fit_prospect(XA, PA, XB, PB, ya, cumulative=cum)
            self.fitted_[name] = prm
            out[name] = predict_prospect(prm, TA, TPA, TB, TPB)
        out["gbdt_objective"] = self._gbdt(tr, te)
        for v in self._beast_versions():
            out["beast" if v == "original" else "beast_sd"] = self._beast(data, v)
        return out

    def _beast_versions(self) -> list[str]:
        b = self.options.get("beast", "sd")
        return {"none": [], "sd": ["sd"], "original": ["original"], "both": ["original", "sd"]}[b]

    def _gbdt(self, tr, te) -> np.ndarray:
        from sklearn.ensemble import HistGradientBoostingRegressor
        cols = list(self.objective)
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        m.fit(tr[cols].to_numpy(float), tr["y"].to_numpy(float))
        return np.clip(m.predict(te[cols].to_numpy(float)), 0, 1)

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        return pd.DataFrame({"unit": np.arange(len(t)), "group": t["problem"].to_numpy(),
                             "item": t["problem"].to_numpy(), "segment": np.where(t["Amb"] == 1, "amb", "risk"),
                             "seg_block": t["block"].to_numpy(int), "n": t["n"].to_numpy(float),
                             "se2": t["se2"].to_numpy(float), "truth": t["y"].to_numpy(float)})


def _locate(root: str, names: Sequence[str]) -> str:
    for d in (os.path.join(root, "ham"), root):
        if all(os.path.exists(os.path.join(d, n)) for n in names):
            return d
    raise DataNotFound(f"{list(names)} not found under {root}/ham or {root}")


@register()
class Choices13k(_RiskyChoice):
    """Options: subset='peterson' (default; feedback, unambiguous: 9,831 problems) | 'all' (14,568 rows);
    seed=0, test_size=0.1 (Peterson et al.: random 90/10, 983 test problems), train_frac=1.0 (subsample of train);
    beast='sd' (default) | 'original' | 'both' | 'none'; beast_sims=5000; beast_protocol='five' (default; the
    choices13k design: 5 trials, feedback after each on Feedback rows) | 'cpc_block' (25-trial CPC simulation, the
    row's Block, as the CENTaUR choices13k BEAST.sd script)."""

    name = "choices13k"
    accepted_options = {
        "subset": '"peterson" | "all" (default "peterson")',
        "seed": "int (default 0)",
        "test_size": "float in (0, 1) (default 0.1)",
        "train_frac": "float in (0, 1] (default 1.0)",
        "beast": '"sd" | "original" | "both" | "none" (default "sd")',
        "beast_sims": "int (default 5000)",
        "beast_protocol": '"five" | "cpc_block" (default "five")',
    }
    data_subdir = "choices13k"
    layout = """
    <root>/choices13k/ham/c13k_selections.csv   Problem, Feedback, n, Block, Ha, pHa, La, Hb, pHb, Lb, LotShapeB,
                                                LotNumB, Amb, Corr, bRate, bRate_std (github jcpeterson/choices13k)
    <root>/choices13k/ham/c13k_problems.json    {row index: {"A": [[p, x], ...], "B": [[p, x], ...]}}
    """
    source = "Peterson, Bourgin, Agrawal, Reichman & Griffiths (2021) Science 372:1209; github jcpeterson/choices13k"
    objective = ("Ha", "pHa", "La", "Hb", "pHb", "Lb", "LotShapeB", "LotNumB", "Amb", "Corr", "Feedback", "block")

    def _load(self, root: str) -> TaskData:
        d = _locate(root, ["c13k_selections.csv", "c13k_problems.json"])
        sel = pd.read_csv(need(os.path.join(d, "c13k_selections.csv")))
        with open(os.path.join(d, "c13k_problems.json"), encoding="utf-8") as f:
            probs = json.load(f)
        sel["dist_a"] = [merge_dist([x for _, x in probs[str(i)]["A"]], [p for p, _ in probs[str(i)]["A"]])
                         for i in range(len(sel))]
        sel["dist_b"] = [merge_dist([x for _, x in probs[str(i)]["B"]], [p for p, _ in probs[str(i)]["B"]])
                         for i in range(len(sel))]
        sel["Feedback"] = sel["Feedback"].astype(bool).astype(int)
        sel["Amb"] = sel["Amb"].astype(bool).astype(int)
        sel["row"] = np.arange(len(sel))
        subset = self.options.get("subset", "peterson")
        if subset == "peterson":
            sel = sel[(sel.Feedback == 1) & (sel.Amb == 0)]
        elif subset != "all":
            raise ValueError("subset must be 'peterson' or 'all'")
        sel = sel.reset_index(drop=True)
        sel["problem"] = sel["row"]  # Problem ids repeat across feedback conditions; the row is the unit
        sel["block"] = sel["Block"].astype(int)
        sel["y"] = sel["bRate"].astype(float)
        sel["se2"] = sel["bRate_std"].astype(float) ** 2 / sel["n"]
        seed, ts = int(self.options.get("seed", 0)), float(self.options.get("test_size", 0.1))
        perm = np.random.default_rng(seed).permutation(len(sel))
        n_te = int(round(ts * len(sel)))
        te, tr = sel.iloc[np.sort(perm[:n_te])], sel.iloc[np.sort(perm[n_te:])]
        frac = float(self.options.get("train_frac", 1.0))
        if frac < 1:
            tr = tr.sample(frac=frac, random_state=seed)
        meta = {"split": f"random {1 - ts:.0%}/{ts:.0%} of {len(sel)} problems, default_rng({seed}).permutation"
                         + (f"; train subsample {frac}" if frac < 1 else ""),
                "subset": subset, "n_train": len(tr), "n_test": len(te),
                "source": "choices13k (Peterson et al. 2021); no licence file in the repository"}
        return TaskData(train=tr.reset_index(drop=True), test=te.reset_index(drop=True), meta=meta)

    def _beast(self, data: TaskData, version: str) -> np.ndarray:
        t = data.test
        sims = int(self.options.get("beast_sims", 5000))
        proto = self.options.get("beast_protocol", "five")
        da, db, amb, corr = list(t.dist_a), list(t.dist_b), t.Amb.to_numpy(), t.Corr.to_numpy()
        if proto == "cpc_block":
            pr = beast_predict(da, db, amb, corr, version=version, n_sims=sims)
            return pr[np.arange(len(t)), t["block"].to_numpy(int) - 1]
        if proto != "five":
            raise ValueError("beast_protocol must be 'cpc_block' or 'five'")
        out = np.empty(len(t))
        for fb in (0, 1):
            m = (t.Feedback.to_numpy() == fb)
            if m.any():
                idx = np.flatnonzero(m)
                out[idx] = beast_predict([da[i] for i in idx], [db[i] for i in idx], amb[idx], corr[idx],
                                         version=version, n_sims=sims, n_trials=5, first_feedback=1 if fb else 6,
                                         n_blocks=1)[:, 0]
        return out


@register()
class CPC18(_RiskyChoice):
    """Options: split='cpc18' (default; sets 1-7 = 210 train tasks, sets 8-9 = the 60 competition tasks) | 'cpc15'
    (sets 1-3 = 90 calibration, sets 4-5 = 60 competition tasks of CPC15); subset='all' (default; 5 blocks per task) |
    'risk_nofb' (unambiguous tasks, first block only: the CPC18 SI foresight comparison, 182 / 48 tasks);
    beast='both' (default) | 'sd' | 'original' | 'none'; beast_sims=5000."""

    name = "cpc18"
    accepted_options = {
        "split": '"cpc18" | "cpc15" (default "cpc18")',
        "subset": '"all" | "risk_nofb" (default "all")',
        "beast": '"both" | "sd" | "original" | "none" (default "both")',
        "beast_sims": "int (default 5000)",
    }
    data_subdir = "cpc18"
    layout = """
    <root>/cpc18/ham/cpc18_all_raw.csv   trial-level CPC15 + CPC18 data (Zenodo 10.5281/zenodo.2571510): SubjID, Set,
                                         GameID, Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB, Amb,
                                         Corr, Trial, B (1 = chose B), block (1-5)
    """
    source = ("Plonsky et al. (2019/2025) arXiv:1904.06866, CPC18; Erev, Ert, Plonsky, Cohen & Cohen (2017) Psych. "
              "Review 124:369, CPC15; data Zenodo 10.5281/zenodo.2571510 (CC BY 4.0)")
    objective = ("Ha", "pHa", "La", "LotShapeA_i", "LotNumA", "Hb", "pHb", "Lb", "LotShapeB_i", "LotNumB", "Amb",
                 "Corr", "block")
    KEYS = ["Ha", "pHa", "La", "LotShapeA", "LotNumA", "Hb", "pHb", "Lb", "LotShapeB", "LotNumB", "Amb", "Corr"]

    def _beast_versions(self) -> list[str]:
        b = self.options.get("beast", "both")
        return {"none": [], "sd": ["sd"], "original": ["original"], "both": ["original", "sd"]}[b]

    @classmethod
    def aggregate(cls, raw: pd.DataFrame) -> pd.DataFrame:
        """One row per (GameID, block): problem dimensions, B-rate, n subjects, squared standard error."""
        per = raw.groupby(["GameID", "block", "SubjID"], sort=True).B.mean().rename("rate").reset_index()
        cell = per.groupby(["GameID", "block"]).rate.agg(y="mean", n="size", var=lambda v: v.var(ddof=1))
        par = raw.groupby("GameID")[cls.KEYS + ["Set"]].first()
        out = cell.reset_index().merge(par.reset_index(), on="GameID")
        out["se2"] = out["var"].fillna(0) / out["n"]
        return out.drop(columns="var")

    def _load(self, root: str) -> TaskData:
        d = _locate(root, ["cpc18_all_raw.csv"])
        raw = pd.read_csv(need(os.path.join(d, "cpc18_all_raw.csv")),
                          usecols=["SubjID", "Set", "GameID", *self.KEYS, "B", "block"], low_memory=False)
        al = self.aggregate(raw)
        games = al.drop_duplicates("GameID").set_index("GameID")
        dists = {g: (cpc_distribution(r.Ha, r.pHa, r.La, r.LotShapeA, r.LotNumA),
                     cpc_distribution(r.Hb, r.pHb, r.Lb, r.LotShapeB, r.LotNumB)) for g, r in games.iterrows()}
        al["dist_a"] = [dists[g][0] for g in al.GameID]
        al["dist_b"] = [dists[g][1] for g in al.GameID]
        code = {"-": 0, "Symm": 1, "R-skew": 2, "L-skew": 3}
        al["LotShapeA_i"] = al.LotShapeA.map(code)
        al["LotShapeB_i"] = al.LotShapeB.map(code)
        al["problem"] = al["GameID"]
        split = self.options.get("split", "cpc18")
        if split == "cpc18":
            tr, te = al[al.Set <= 7], al[al.Set.isin([8, 9])]
            rule = "CPC18 track I: sets 1-7 (210 tasks) train, sets 8-9 (60 competition tasks) test"
        elif split == "cpc15":
            tr, te = al[al.Set <= 3], al[al.Set.isin([4, 5])]
            rule = "CPC15: sets 1-3 (90 calibration tasks) train, sets 4-5 (60 competition tasks) test"
        else:
            raise ValueError("split must be 'cpc18' or 'cpc15'")
        subset = self.options.get("subset", "all")
        if subset == "risk_nofb":
            tr, te = tr[(tr.Amb == 0) & (tr.block == 1)], te[(te.Amb == 0) & (te.block == 1)]
        elif subset != "all":
            raise ValueError("subset must be 'all' or 'risk_nofb'")
        meta = {"split": rule, "subset": subset, "n_train_tasks": int(tr.GameID.nunique()),
                "n_test_tasks": int(te.GameID.nunique()), "n_test_cells": len(te),
                "source": "CPC15/CPC18 raw data, Zenodo 10.5281/zenodo.2571510, CC BY 4.0"}
        return TaskData(train=tr.reset_index(drop=True), test=te.reset_index(drop=True), meta=meta)

    def _beast(self, data: TaskData, version: str) -> np.ndarray:
        t = data.test
        g = t.drop_duplicates("GameID").reset_index(drop=True)
        pr = beast_predict(list(g.dist_a), list(g.dist_b), g.Amb.to_numpy(), g.Corr.to_numpy(), version=version,
                           n_sims=int(self.options.get("beast_sims", 5000)))
        pos = pd.Series(np.arange(len(g)), index=g.GameID)
        return pr[pos.loc[t.GameID].to_numpy(), t["block"].to_numpy(int) - 1]
