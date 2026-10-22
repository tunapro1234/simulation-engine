"""LLM (or learner) prior + early A/B test, combined by a normal-normal Bayes update on the log-odds ratio.

"First 100 people + simulation = a 200+ person test": the prior N(k * logit p, tau^2) on the true log-odds ratio
(A vs B) is calibrated on a few finished tests (k: scale, tau^2: how much to trust the prior), then combined with
the noisy log-odds ratio of a small early test.

Two replay protocols are ported, and they give different numbers on the same data:

* ``simulate_saving``: `uretim/deney/kod/analiz/ornek_tasarrufu_llm.py` (2-fold cross-fit calibration on half the
  pairs, Haldane +0.5 correction, 300 binomial replays per pair, seed 11). Its saved output
  (`ornek_tasarrufu_llm.json`, Upworthy llm300, 300 pairs) at 100 impressions/arm: human 0.713; Opus 5.5 direct
  prior -> hybrid 0.815, 2.47x saving (1.34x at 400, none at 800); TF-IDF prior 2.79x; TF-IDF + Opus stack 3.68x.
* ``simulate_saving_calibrated``: `uretim/deney/kod/analiz/ornek_tasarrufu_llm_ga.py` (each of 10 blocks is
  calibrated on ``n_calib`` random pairs from the other blocks, 5 draws, 200 replays, seed 5, pair-bootstrap 95% CI
  of the multiplier at n=100). This is the headline of uretim/deney/RAPOR.md 2.2-2.3: no archive, Opus 5.5 direct
  prior calibrated on 20 tests: accuracy 0.718 -> 0.808, 2.27x (CI 1.67-3.11); TF-IDF 2.44x; stack 3.30x
  (`ornek_tasarrufu_llm_ga.json`). Replicated on ek600 at 2.05x (1.70-2.50).

Most of the gain comes from breaking ties in tests with very few clicks ("human decides, prior breaks ties"
alone gives 0.786).

Usage::

    b = EarlyTestBayes().fit(prior_prob_finished, lor_finished, se_finished)
    b.predict(prior_prob_new, clicks_a, 100, clicks_b, 100)      # P(A better)
    b.posterior(prior_prob_new, clicks_a, 100, clicks_b, 100)    # {"lor", "lo", "hi", "p_better"}

Return conventions differ on purpose (kept for compatibility): ``true_lor`` returns ``(lor, se)`` (a standard
error), ``early_lor`` returns ``(lor, var)`` (a variance). Both take a keyword to return the other quantity.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm

NS = (50, 100, 200, 400, 800)


def _logit(p, eps=1e-3):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def true_lor(ctr_a, ctr_b, clicks_a=None, clicks_b=None, return_var: bool = False):
    """Full-data log-odds ratio (A vs B). Returns ``(lor, se)``: ``se`` is a STANDARD ERROR on the log-odds scale
    (0 when click counts are not given), unlike ``early_lor`` which returns a variance.
    ``return_var=True`` returns ``(lor, se ** 2)`` instead."""
    lor = _logit(ctr_a) - _logit(ctr_b)
    if clicks_a is None or clicks_b is None:
        return lor, np.zeros_like(lor)
    se = np.sqrt(1 / (np.asarray(clicks_a, float) + .5) + 1 / (np.asarray(clicks_b, float) + .5))
    return (lor, se ** 2) if return_var else (lor, se)


def calibrate_prior(prior_prob, lor, se=None, fix_scale: bool = False):
    """Fit k (prior logit -> LOR scale, through the origin) and tau^2 = max(Var(lor - k z) - mean(se^2), 1e-3).
    ``fix_scale=True`` uses k = 1 (measured: same or slightly better saving, Opus logit is already near LOR scale)."""
    z = _logit(prior_prob)
    lor = np.asarray(lor, float)
    se = np.zeros_like(lor) if se is None else np.asarray(se, float)
    k = 1.0 if fix_scale else float(np.sum(z * lor) / max(np.sum(z ** 2), 1e-12))
    tau2 = max(float(np.var(lor - k * z) - np.mean(se ** 2)), 1e-3)
    return k, tau2


def early_lor(clicks_a, n_a, clicks_b, n_b, return_se: bool = False):
    """Early-test log-odds ratio (A vs B) with Haldane +0.5 correction. Returns ``(lor, var)``: ``var`` is the
    sampling VARIANCE on the log-odds scale (squared units), unlike ``true_lor`` which returns a standard error.
    ``return_se=True`` returns ``(lor, sqrt(var))`` instead."""
    ca, cb = np.asarray(clicks_a, float), np.asarray(clicks_b, float)
    na, nb = np.asarray(n_a, float), np.asarray(n_b, float)
    pa, pb = (ca + .5) / (na + 1), (cb + .5) / (nb + 1)
    lor = _logit(pa) - _logit(pb)
    se2 = 1 / (ca + .5) + 1 / (na - ca + .5) + 1 / (cb + .5) + 1 / (nb - cb + .5)
    return (lor, np.sqrt(se2)) if return_se else (lor, se2)


def posterior_lor(mu, tau2, lor, se2):
    """Precision-weighted posterior of the LOR from prior N(mu, tau2) and data N(lor, se2); both second arguments
    are variances. Returns ``(mean, var)``."""
    prec = 1 / np.asarray(tau2, float) + 1 / np.asarray(se2, float)
    mean = (np.asarray(mu, float) / tau2 + np.asarray(lor, float) / se2) / prec
    return mean, 1 / prec


def _out(x):
    x = np.asarray(x, float)
    return float(x) if x.ndim == 0 else x


class EarlyTestBayes:
    """LLM (or learner) prior + small early A/B test, normal-normal update on the log-odds ratio (LOR, A vs B).

    ``fit(prior_prob, lor, se)`` on finished calibration tests, then for a new early test:
    ``predict`` = P(A better) under the posterior, ``decide`` = posterior sign (True = A), ``posterior`` = point
    estimate, credible interval and P(A better). All accept scalars or equal-length arrays (``n_a``/``n_b`` may be
    scalars broadcast over the tests).

    Fitted attributes:
        k_: dimensionless scale from the prior's logit to the LOR (prior mean = ``k_ * logit(prior_prob)``,
            in log-odds units). 1 with ``fix_scale=True``; ~0 means the prior carries no information.
        tau2_: prior variance of the true LOR around that mean, in squared log-odds units (floor 1e-3). Small =
            trust the prior, large = the early data dominate.
    """

    def __init__(self, fix_scale: bool = False):
        self.fix_scale = fix_scale

    def fit(self, prior_prob, lor, se=None):
        """Calibrate ``k_`` and ``tau2_`` on finished tests.

        prior_prob: the prior's P(A better) per finished test; lor: their full-data LOR (A vs B); se: STANDARD
        errors of ``lor`` (e.g. the second value of ``true_lor``), subtracted as noise from tau^2; None = 0.
        Warns when the prior carries no information (constant ``prior_prob`` or fitted ``k_`` ~ 0)."""
        self.k_, self.tau2_ = calibrate_prior(prior_prob, lor, se, self.fix_scale)
        z = _logit(prior_prob)
        if abs(self.k_) < 1e-6 or (z.size and float(np.var(z)) < 1e-8):
            warnings.warn("prior carries no information; k_=0, the posterior equals the early-data estimate"
                          if abs(self.k_) < 1e-6 else
                          f"prior carries no information (prior_prob is constant); k_={self.k_:.3g} only shifts "
                          "every test by the same amount", RuntimeWarning, stacklevel=2)
        return self

    def posterior_moments(self, prior_prob, clicks_a, n_a, clicks_b, n_b):
        """Posterior ``(mean, var)`` of the LOR (A vs B), var in squared log-odds units.
        This is what ``posterior`` returned before it returned a dict."""
        mu = self.k_ * _logit(prior_prob)
        lor, se2 = early_lor(clicks_a, n_a, clicks_b, n_b)
        return posterior_lor(mu, self.tau2_, lor, se2)

    def posterior(self, prior_prob, clicks_a, n_a, clicks_b, n_b, alpha: float = 0.05) -> dict:
        """Posterior summary of the LOR (A vs B) for an early test with ``clicks_*`` out of ``n_*`` impressions.

        Returns ``{"lor", "lo", "hi", "p_better", "var"}``: posterior mean LOR, the (1 - alpha) central credible
        interval [lo, hi] (95% by default), P(A better) (= ``predict``) and the posterior variance. Floats for
        scalar input, arrays otherwise. exp() of lor/lo/hi gives the odds-ratio lift and its interval.
        Note: before this method returned ``(mean, var)``; that tuple is now ``posterior_moments``."""
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        m, v = self.posterior_moments(prior_prob, clicks_a, n_a, clicks_b, n_b)
        sd, zq = np.sqrt(v), norm.ppf(1 - alpha / 2)
        return {"lor": _out(m), "lo": _out(m - zq * sd), "hi": _out(m + zq * sd),
                "p_better": _out(norm.cdf(m / sd)), "var": _out(v)}

    def predict(self, prior_prob, clicks_a, n_a, clicks_b, n_b):
        """P(A better) = P(LOR > 0) under the posterior, per test (array, or 0-d for scalar input)."""
        m, v = self.posterior_moments(prior_prob, clicks_a, n_a, clicks_b, n_b)
        return norm.cdf(m / np.sqrt(v))

    def decide(self, prior_prob, clicks_a, n_a, clicks_b, n_b):
        """Posterior decision: True = ship A (posterior mean LOR > 0), False = B; same as ``predict > 0.5``."""
        return self.posterior_moments(prior_prob, clicks_a, n_a, clicks_b, n_b)[0] > 0


def equivalent_multiplier(acc_hybrid: float, human_curve, ns=NS, n: int = 100):
    """How many impressions a human-only test needs to match ``acc_hybrid``, divided by ``n``
    (log-linear interpolation on the human accuracy curve; None if beyond the largest measured n)."""
    hy, xs = np.asarray(human_curve, float), np.log(np.asarray(ns, float))
    if acc_hybrid > hy.max():
        return None
    return float(np.exp(np.interp(acc_hybrid, hy, xs)) / n)


def simulate_saving(ctr_a, ctr_b, prior_prob, clicks_a=None, clicks_b=None, ns=NS, reps: int = 300,
                    seed: int = 11, fix_scale: bool = False) -> dict:
    """Offline replay of the impression-saving experiment.

    For each pair the "truth" is the full-data winner; small tests with n impressions per arm are replayed
    ``reps`` times from Binomial(n, ctr). Human-only decides by the higher sample CTR (coin flip on ties); the
    hybrid uses the 2-fold cross-fitted prior (a pair's calibration never sees its own result).
    Returns accuracies per n, prior-only accuracy and the equivalent-impression multiplier per n.
    Note: the replay is circular (truth and replay come from the same data); the adversarial check found this
    biases AGAINST the hybrid (independent split-half: 3.1-3.3x) (uretim/deney/RAPOR.md 2.1).
    Seed-for-seed equal to `ornek_tasarrufu_llm.py` (same RNG consumption order). For the RAPOR headline
    (calibration on 20 tests, CI) use ``simulate_saving_calibrated``."""
    ctr_a, ctr_b = np.asarray(ctr_a, float), np.asarray(ctr_b, float)
    y = ctr_a > ctr_b
    lor_t, se_t = true_lor(ctr_a, ctr_b, clicks_a, clicks_b)
    n_pairs = len(y)
    fold = np.arange(n_pairs) % 2
    mu, tau2 = np.zeros(n_pairs), np.zeros(n_pairs)
    for f in (0, 1):
        m = fold != f
        if m.sum() == 0:
            m = np.ones(n_pairs, bool)
        k, t2 = calibrate_prior(np.asarray(prior_prob)[m], lor_t[m], se_t[m], fix_scale)
        mu[fold == f], tau2[fold == f] = k * _logit(np.asarray(prior_prob)[fold == f]), t2
    # RNG order as in the research script: every binomial replay for every n first, then the tie-break draws.
    rs = np.random.default_rng(seed)
    sims = [(rs.binomial(n, ctr_a, size=(reps, n_pairs)), rs.binomial(n, ctr_b, size=(reps, n_pairs))) for n in ns]
    human, hybrid = [], []
    for ca, cb in sims:
        d = ca - cb
        dec = np.where(d == 0, rs.random(d.shape) < 0.5, d > 0)
        human.append(float(np.mean(dec == y)))
    for n, (ca, cb) in zip(ns, sims):
        lor, se2 = early_lor(ca, n, cb, n)
        post, _ = posterior_lor(mu, tau2, lor, se2)
        hybrid.append(float(np.mean((post > 0) == y)))
    p = np.asarray(prior_prob, float)
    prior_acc = float(np.mean(np.where(p == .5, .5, (p > .5) == y)))
    mult = [equivalent_multiplier(h, human, ns, n) for h, n in zip(hybrid, ns)]
    return {"n_pairs": int(n_pairs), "ns": list(ns), "human": dict(zip(map(str, ns), human)),
            "hybrid": dict(zip(map(str, ns), hybrid)), "prior_only": prior_acc,
            "multiplier": dict(zip(map(str, ns), mult))}


NS_CALIBRATED = (25, 50, 100, 200, 400, 800, 1600)


def simulate_saving_calibrated(ctr_a, ctr_b, prior_prob, clicks_a=None, clicks_b=None, n_calib: int = 20,
                               ns=NS_CALIBRATED, reps: int = 200, draws: int = 5, blocks: int = 10,
                               n_boot: int = 500, n0: int = 100, seed: int = 5, fix_scale: bool = False) -> dict:
    """Impression saving when only ``n_calib`` finished tests are available to calibrate the prior, with a
    pair-bootstrap 95% CI of the multiplier at ``n0`` impressions/arm (port of `ornek_tasarrufu_llm_ga.py`).

    Pairs are split into ``blocks`` blocks; each block's k and tau^2 come from ``n_calib`` pairs drawn at random from
    the other blocks (one draw per block, ``draws`` independent draws averaged, draw seeds 0..draws-1). Small tests
    are replayed ``reps`` times. The human curve is made monotone (running max) before the inverse interpolation.
    Ties at ``n0``: ``human_decides_prior_breaks_ties`` is the accuracy of "human decides, prior breaks ties".
    Measured (Upworthy llm300, Opus 5.5 direct, n_calib=20): human 0.718 -> hybrid 0.808, 2.27x [1.67, 3.11];
    n_calib=150: 2.58x [1.88, 3.42] (uretim/deney/RAPOR.md 2.2-2.3, `ornek_tasarrufu_llm_ga.json`)."""
    ctr_a, ctr_b = np.asarray(ctr_a, float), np.asarray(ctr_b, float)
    p = np.asarray(prior_prob, float)
    y = ctr_a > ctr_b
    n_pairs = len(y)
    if n0 not in ns:
        raise ValueError(f"n0={n0} must be one of ns={tuple(ns)}")
    lor_t, se_t = true_lor(ctr_a, ctr_b, clicks_a, clicks_b)
    z = _logit(p)
    rs = np.random.default_rng(seed)
    sims = {n: (rs.binomial(n, ctr_a, size=(reps, n_pairs)), rs.binomial(n, ctr_b, size=(reps, n_pairs))) for n in ns}
    tie = rs.random((reps, n_pairs)) < 0.5
    human_dec = {n: np.where(sims[n][0] == sims[n][1], tie, sims[n][0] > sims[n][1]) for n in ns}
    human_acc = {n: np.mean(human_dec[n] == y, axis=0) for n in ns}
    blk = np.arange(n_pairs) % blocks

    def hybrid_acc(draw_seed):
        r = np.random.default_rng(draw_seed)
        mu, tau2 = np.zeros(n_pairs), np.zeros(n_pairs)
        for b in range(blocks):
            cand = np.flatnonzero(blk != b)
            if len(cand) == 0:
                continue
            m = r.choice(cand, min(n_calib, len(cand)), replace=False)
            k, t2 = calibrate_prior(p[m], lor_t[m], se_t[m], fix_scale)
            mu[blk == b], tau2[blk == b] = k * z[blk == b], t2
        out = {}
        for n in ns:
            ca, cb = sims[n]
            lor, se2 = early_lor(ca, n, cb, n)
            post, _ = posterior_lor(mu, tau2, lor, se2)
            out[n] = np.mean((post > 0) == y, axis=0)
        return out

    hs = [hybrid_acc(s) for s in range(draws)]
    h = {n: np.mean([x[n] for x in hs], axis=0) for n in ns}
    logn = np.log(np.asarray(ns, float))

    def gain(idx):
        hy = np.maximum.accumulate(np.array([human_acc[n][idx].mean() for n in ns]))
        target = h[n0][idx].mean()
        if target > hy.max():
            return np.nan
        return float(np.exp(np.interp(target, hy, logn)) / n0)

    point = gain(np.arange(n_pairs))
    bs = np.array([gain(rs.integers(0, n_pairs, n_pairs)) for _ in range(n_boot)])
    bs = bs[np.isfinite(bs)]
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) else [None, None]
    ties = sims[n0][0] == sims[n0][1]
    rule = float(np.mean(np.where(ties, np.where(p == .5, tie, p > .5), human_dec[n0]) == y))
    return {"n_pairs": int(n_pairs), "n_calib": int(n_calib), "ns": list(ns), "n0": int(n0),
            "human": {str(n): float(human_acc[n].mean()) for n in ns},
            "hybrid": {str(n): float(h[n].mean()) for n in ns},
            "prior_only": float(np.mean(np.where(p == .5, .5, (p > .5) == y))),
            "multiplier": None if not np.isfinite(point) else point, "ci": ci,
            "tie_rate": float(ties.mean()), "human_decides_prior_breaks_ties": rule}
