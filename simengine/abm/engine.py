"""Fast vectorised numpy ABM engines (ported from the TALP prep research engine `motor.py`).

Two modes, same agent structure, both simulate a whole BATCH of parameter sets (B, ...) in one call so a
calibration can run thousands of simulations at once. Agent randomness comes from `seed` (common random numbers),
so parameter differences are separated from noise.

* :func:`market_sim`  : repeat-purchase market (price / promotion response). Heterogeneous willingness to pay,
  reference price with gain / loss asymmetry, promotion effect, stockpiling, inventory, social influence.
* :func:`adoption_sim`: durable-good adoption on a small-world network. Heterogeneous innovators (p_i),
  imitation with a Granovetter threshold, local / global mixing, market that opens up over 15 years.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- repeat-purchase market (M2)
# Parameters act on log-relative price (lp = log(price / pre-event regular price)); unit free.
MARKET_PRIOR = {
    #  name       low     high    meaning
    "a0":       (-4.0,   1.0),   # base purchase propensity (logit)
    "beta":     (0.0,    8.0),   # price sensitivity (WTP - price)
    "sigma":    (0.05,   1.5),   # WTP heterogeneity (log-scale std)
    "gain":     (0.0,    8.0),   # gain from a price below the reference (deal perception)
    "loss":     (0.0,   12.0),   # loss from a price above the reference (loss aversion)
    "theta":    (0.0,    0.98),  # reference-price memory (weekly exponential smoothing)
    "stock":    (0.0,    4.0),   # stockpiling: extra weeks of consumption per 10% discount
    "inventory": (0.0,   4.0),   # drop in purchase wish while holding stock
    "promo":    (0.0,    3.0),   # display / feature effect (logit)
    "social":   (0.0,    3.0),   # effect of neighbours' purchase share last week
}
MARKET_PARAMS = list(MARKET_PRIOR)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def market_sim(theta, lp, promo, n_agents=400, seed=0, warmup=12, k_neighbors=8, return_rate=False,
               cohort=False):
    """Batch market simulation.

    theta : (B, 10) parameter matrix in MARKET_PARAMS order, or a single (10,) vector.
    lp    : (T,) or (B, T) log-relative shelf price path.
    promo : (T,) or (B, T) promotion intensity (0-1).
    Returns (B, T) weekly sales (mean units per agent); with return_rate also the purchase share.
    Warm-up: `warmup` weeks at the first price so inventory and reference settle.
    cohort=True: each agent is a type cohort; purchase uses the probability (expected value) instead of a
    Bernoulli draw, which makes calibration smooth (this is what the research calibration used).
    """
    theta = np.atleast_2d(np.asarray(theta, float))
    B = theta.shape[0]
    lp = np.broadcast_to(np.atleast_2d(np.asarray(lp, float)), (B, np.shape(lp)[-1]))
    promo = np.broadcast_to(np.atleast_2d(np.asarray(promo, float)), lp.shape)
    T = lp.shape[1]
    a0, beta, sigma, gain, loss, mem, stok, env, pro, sos = (theta[:, i:i + 1] for i in range(10))

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_agents)                     # WTP rank (common random numbers)
    c = rng.gamma(2.0, 0.5, n_agents)                     # weekly consumption, mean 1
    nb = (np.arange(n_agents)[:, None] + rng.integers(1, n_agents, (n_agents, k_neighbors))) % n_agents

    wtp = sigma * z[None, :]
    inv = np.broadcast_to(c, (B, n_agents)).copy()        # inventory: one week
    ref = np.broadcast_to(lp[:, :1], (B, n_agents)).copy()
    last = np.zeros((B, n_agents))
    sales = np.zeros((B, T))
    rate = np.zeros((B, T))

    u_all = None if cohort else rng.random((warmup + T, B, n_agents))
    for t in range(-warmup, T):
        ti = max(t, 0)
        p = lp[:, ti:ti + 1] if t >= 0 else lp[:, :1]
        pr = promo[:, ti:ti + 1] if t >= 0 else 0.0
        gap = ref - p                                       # > 0: a deal
        social = last[:, nb].mean(-1)
        u = (a0 + beta * (wtp - p) + gain * np.maximum(gap, 0) - loss * np.maximum(-gap, 0)
             + pro * pr - env * inv / c + sos * social)
        buy = _sigmoid(u) if cohort else (u_all[t + warmup] < _sigmoid(u)).astype(float)
        q = buy * c * (1.0 + stok * 10.0 * np.maximum(gap, 0))
        inv = np.maximum(inv + q - c, 0.0)
        ref = mem * ref + (1 - mem) * p
        last = buy
        if t >= 0:
            sales[:, t] = q.mean(1)
            rate[:, t] = buy.mean(1)
    if return_rate:
        return sales, rate
    return sales


def sample_market_prior(n, rng, prior=None):
    prior = prior or MARKET_PRIOR
    lo = np.array([prior[k][0] for k in MARKET_PARAMS])
    hi = np.array([prior[k][1] for k in MARKET_PARAMS])
    return lo + (hi - lo) * rng.random((n, len(lo)))


# ---------------------------------------------------------------- durable adoption (M1)
ADOPTION_PRIOR = {
    "m":         (0.02,  1.0),   # long-run market share ceiling (final adopter share)
    "log_p":     (-9.0, -1.0),   # log p: mean external influence (innovation), per year
    "log_q":     (-3.5,  1.0),   # log q: imitation (per adopted-neighbour share), per year
    "het":       (0.0,   2.0),   # innovation heterogeneity (log-normal std)
    "threshold": (0.0,   0.5),   # imitation threshold: no effect below this neighbour share (Granovetter)
    "global":    (0.0,   1.0),   # share of social influence that is global (1 = full mixing, Bass)
    "expand":    (0.0,   1.0),   # market expansion: this fraction of m is closed at t=0 and opens linearly in 15 yr
    "shift":     (0.0,  15.0),   # where data year 0 sits on the simulation clock (years)
}
ADOPTION_PARAMS = list(ADOPTION_PRIOR)


def adoption_sim(theta, T, substeps=4, n_agents=2000, seed=0, k_neighbors=10, raw=False):
    """Batch adoption simulation. Returns (B, T): adopter share (%, 0-100) at the end of data years 0..T-1.
    Data year j sits at simulation time shift + j.

    Market entry: u_i < m(t) (u_i agent rank); m(t) = m * (1 - expand * max(0, 1 - t/15)).
    Adoption hazard: p_i + q * max(nb_share - threshold, 0) / (1 - threshold), nb_share a local/global mix.
    raw=True: `shift` is ignored; returns (time (S,), path (B,S) %, innovators (B,S) %), where innovators is the
    expected cumulative part of adoption attributed to external influence p_i.
    """
    theta = np.atleast_2d(np.asarray(theta, float))
    B = theta.shape[0]
    m, lp, lq, het, esik, kur, gen, kay = (theta[:, i:i + 1] for i in range(8))
    rng = np.random.default_rng(seed)
    zi = rng.standard_normal(n_agents)
    rank = rng.random(n_agents)
    # small world: half ring neighbours, half random
    half = k_neighbors // 2
    ring = (np.arange(n_agents)[:, None] + np.r_[1:half // 2 + 1, -(np.r_[1:half - half // 2 + 1])]) % n_agents
    rand = rng.integers(0, n_agents, (n_agents, k_neighbors - half))
    nb = np.concatenate([ring, rand], 1)

    p_i = np.exp(lp + het * zi[None, :] - het ** 2 / 2)   # keeps mean exp(lp)
    q = np.exp(lq)
    dt = 1.0 / substeps
    adopted = np.zeros((B, n_agents), bool)
    S = int(np.ceil((T + ADOPTION_PRIOR["shift"][1]) * substeps))
    path = np.zeros((B, S))
    inn = np.zeros((B, S)) if raw else None
    inn_cum = np.zeros(B)
    u_all = rng.random((S, n_agents))
    for s in range(S):
        yr = (s + 1) * dt
        m_t = m * (1 - gen * max(0.0, 1 - yr / 15.0))
        in_market = rank[None, :] < m_t
        bf = adopted.astype(float)
        local = bf[:, nb].mean(-1)
        glob = bf.mean(1, keepdims=True)
        share = (1 - kur) * local + kur * glob
        eff = np.maximum(share - esik, 0) / np.maximum(1 - esik, 1e-6)
        h = (p_i + q * eff) * dt
        new = in_market & ~adopted & (u_all[s][None, :] < 1 - np.exp(-h))
        adopted |= new
        path[:, s] = 100.0 * adopted.mean(1)
        if raw:
            frac = p_i / np.maximum(p_i + q * eff, 1e-12)
            inn_cum += (new * frac).sum(1)
            inn[:, s] = 100.0 * inn_cum / n_agents
    time = (np.arange(S) + 1) * dt
    if raw:
        return time, path, inn
    target = kay + np.arange(T)[None, :] + 1.0
    return np.stack([np.interp(target[b], time, path[b]) for b in range(B)])


def sample_adoption_prior(n, rng, prior=None):
    prior = prior or ADOPTION_PRIOR
    lo = np.array([prior[k][0] for k in ADOPTION_PARAMS])
    hi = np.array([prior[k][1] for k in ADOPTION_PARAMS])
    return lo + (hi - lo) * rng.random((n, len(lo)))
