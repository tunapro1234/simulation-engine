"""LLM (or learner) prior + early A/B test, combined by a normal-normal Bayes update on the log-odds ratio.

"First 100 people + simulation = a 200+ person test": the prior N(k * logit p, tau^2) on the true log-odds ratio
(A vs B) is calibrated on a few finished tests (k: scale, tau^2: how much to trust the prior), then combined with
the noisy log-odds ratio of a small early test. Ported from `uretim/deney/kod/analiz/ornek_tasarrufu_llm.py`
(2-fold cross-fit calibration, Haldane +0.5 correction, 300 binomial replays per pair).

Measured on Upworthy llm300 (uretim/deney/RAPOR.md 2.2-2.3): no archive, Opus 5.5 direct prior calibrated on 20
tests, 100 impressions/arm (about 1.5 clicks): accuracy 0.718 -> 0.808, about 2.3x impression saving
(2.27x, CI 1.67-3.11); 1.3x at 400 impressions; no gain at 800+. Replicated on ek600 at 2.05x (1.70-2.50).
Most of the gain comes from breaking ties in tests with very few clicks ("human decides, prior breaks ties"
alone gives 0.786).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

NS = (50, 100, 200, 400, 800)


def _logit(p, eps=1e-3):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def true_lor(ctr_a, ctr_b, clicks_a=None, clicks_b=None):
    """Full-data log-odds ratio (A vs B) and its standard error (0 when click counts are not given)."""
    lor = _logit(ctr_a) - _logit(ctr_b)
    if clicks_a is None or clicks_b is None:
        return lor, np.zeros_like(lor)
    se = np.sqrt(1 / (np.asarray(clicks_a, float) + .5) + 1 / (np.asarray(clicks_b, float) + .5))
    return lor, se


def calibrate_prior(prior_prob, lor, se=None, fix_scale: bool = False):
    """Fit k (prior logit -> LOR scale, through the origin) and tau^2 = max(Var(lor - k z) - mean(se^2), 1e-3).
    ``fix_scale=True`` uses k = 1 (measured: same or slightly better saving, Opus logit is already near LOR scale)."""
    z = _logit(prior_prob)
    lor = np.asarray(lor, float)
    se = np.zeros_like(lor) if se is None else np.asarray(se, float)
    k = 1.0 if fix_scale else float(np.sum(z * lor) / max(np.sum(z ** 2), 1e-12))
    tau2 = max(float(np.var(lor - k * z) - np.mean(se ** 2)), 1e-3)
    return k, tau2


def early_lor(clicks_a, n_a, clicks_b, n_b):
    """Early-test log-odds ratio with Haldane +0.5 correction and its sampling variance."""
    ca, cb = np.asarray(clicks_a, float), np.asarray(clicks_b, float)
    na, nb = np.asarray(n_a, float), np.asarray(n_b, float)
    pa, pb = (ca + .5) / (na + 1), (cb + .5) / (nb + 1)
    lor = _logit(pa) - _logit(pb)
    se2 = 1 / (ca + .5) + 1 / (na - ca + .5) + 1 / (cb + .5) + 1 / (nb - cb + .5)
    return lor, se2


def posterior_lor(mu, tau2, lor, se2):
    """Precision-weighted posterior mean and variance of the LOR."""
    prec = 1 / np.asarray(tau2, float) + 1 / np.asarray(se2, float)
    mean = (np.asarray(mu, float) / tau2 + np.asarray(lor, float) / se2) / prec
    return mean, 1 / prec


class EarlyTestBayes:
    """fit(prior_prob, lor, se) on finished calibration tests -> predict(prior_prob, clicks/impressions of an early
    test) = P(A better) under the posterior. ``decide`` returns the posterior sign (True = A)."""

    def __init__(self, fix_scale: bool = False):
        self.fix_scale = fix_scale

    def fit(self, prior_prob, lor, se=None):
        self.k_, self.tau2_ = calibrate_prior(prior_prob, lor, se, self.fix_scale)
        return self

    def posterior(self, prior_prob, clicks_a, n_a, clicks_b, n_b):
        mu = self.k_ * _logit(prior_prob)
        lor, se2 = early_lor(clicks_a, n_a, clicks_b, n_b)
        return posterior_lor(mu, self.tau2_, lor, se2)

    def predict(self, prior_prob, clicks_a, n_a, clicks_b, n_b):
        m, v = self.posterior(prior_prob, clicks_a, n_a, clicks_b, n_b)
        return norm.cdf(m / np.sqrt(v))

    def decide(self, prior_prob, clicks_a, n_a, clicks_b, n_b):
        return self.posterior(prior_prob, clicks_a, n_a, clicks_b, n_b)[0] > 0


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
    biases AGAINST the hybrid (independent split-half: 3.1-3.3x) (uretim/deney/RAPOR.md 2.1)."""
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
    rs = np.random.default_rng(seed)
    human, hybrid = [], []
    for n in ns:
        ca = rs.binomial(n, ctr_a, size=(reps, n_pairs))
        cb = rs.binomial(n, ctr_b, size=(reps, n_pairs))
        d = ca - cb
        dec = np.where(d == 0, rs.random(d.shape) < 0.5, d > 0)
        human.append(float(np.mean(dec == y)))
        lor, se2 = early_lor(ca, n, cb, n)
        post, _ = posterior_lor(mu, tau2, lor, se2)
        hybrid.append(float(np.mean((post > 0) == y)))
    p = np.asarray(prior_prob, float)
    prior_acc = float(np.mean(np.where(p == .5, .5, (p > .5) == y)))
    mult = [equivalent_multiplier(h, human, ns, n) for h, n in zip(hybrid, ns)]
    return {"n_pairs": int(n_pairs), "ns": list(ns), "human": dict(zip(map(str, ns), human)),
            "hybrid": dict(zip(map(str, ns), hybrid)), "prior_only": prior_acc,
            "multiplier": dict(zip(map(str, ns), mult))}
