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

    from simengine.methods.bayes_prior import compare_priors, saving_section
    r = simulate_saving_calibrated(ctr_a, ctr_b, prior, n_calib=20)   # warns on a weak prior; mode="tiebreak"
    c = compare_priors({"tfidf": p_tfidf, "llm": p_llm}, ctr_a, ctr_b)  # same replays: paired diffs + CIs
    section_markdown(saving_section(r))                                # card section (simengine.evidence)

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
MODES = ("bayes", "tiebreak")
WEAK_PRIOR_ACCURACY = 0.6      # prior-only accuracy below this -> weak-prior warning


def _boot_groups(cluster, n_pairs):
    """Index arrays of the bootstrap units: None = one unit per pair, else one per distinct ``cluster`` value."""
    if cluster is None:
        return None
    c = np.asarray(cluster)
    if len(c) != n_pairs:
        raise ValueError(f"cluster has {len(c)} values for {n_pairs} pairs")
    _, inv = np.unique(c, return_inverse=True)
    return [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]


def _boot_indices(rs, groups, n_pairs, n_boot):
    # cluster=None draws exactly what the research script drew (rs.integers(0, n, n) per replicate)
    if groups is None:
        return [rs.integers(0, n_pairs, n_pairs) for _ in range(n_boot)]
    g = len(groups)
    return [np.concatenate([groups[i] for i in rs.integers(0, g, g)]) for _ in range(n_boot)]


def _ci(bs, cap):
    """95% percentile CI of bootstrap multipliers; NaN = censored above the human curve, ranked above every finite
    value. An endpoint in the censored tail is None (``> cap``). Returns ``(ci, censored_endpoints, n_censored)``."""
    bs = np.asarray(bs, float)
    if not len(bs):
        return [None, None], [False, False], 0
    cen = ~np.isfinite(bs)
    if cen.all():
        return [None, None], [True, True], int(cen.sum())
    big = cap * 1e9
    x = np.where(cen, big, bs)
    lo, hi = float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))
    out = [None if v > cap * (1 + 1e-9) and cen.any() else v for v in (lo, hi)]
    return out, [v is None for v in out], int(cen.sum())


def _replay(ctr_a, ctr_b, ns, reps, seed):
    """Binomial replays of every n (in order), then the tie coin: the RNG order of `ornek_tasarrufu_llm_ga.py`."""
    rs = np.random.default_rng(seed)
    n_pairs = len(ctr_a)
    sims = {n: (rs.binomial(n, ctr_a, size=(reps, n_pairs)), rs.binomial(n, ctr_b, size=(reps, n_pairs))) for n in ns}
    tie = rs.random((reps, n_pairs)) < 0.5
    human_dec = {n: np.where(sims[n][0] == sims[n][1], tie, sims[n][0] > sims[n][1]) for n in ns}
    return rs, sims, tie, human_dec


def _hybrid(p, lor_t, se_t, y, sims, tie, human_dec, ns, n0, n_calib, draws, blocks, fix_scale, mode):
    """Per-pair hybrid accuracy per n (mean over draws and replays), the n0 decisions (draws, reps, pairs) and the
    free calibration slope of every calibration set (for the weak-prior guard)."""
    n_pairs = len(y)
    z = _logit(p)
    blk = np.arange(n_pairs) % blocks
    acc = {n: np.zeros(n_pairs) for n in ns}
    dec0, ks, per_draw = [], [], {n: [] for n in ns}
    if mode == "tiebreak":
        # human decides; on a tie the prior's side wins (coin when the prior is exactly 0.5); no calibration used
        pick = np.where(p == .5, tie, p > .5)
        for n in ns:
            d = np.where(sims[n][0] == sims[n][1], pick, human_dec[n])
            acc[n] = np.mean(d == y, axis=0)
            if n == n0:
                dec0 = np.repeat(d[None], draws, axis=0)
    for s in range(draws):
        r = np.random.default_rng(s)
        mu, tau2 = np.zeros(n_pairs), np.zeros(n_pairs)
        for b in range(blocks):
            cand = np.flatnonzero(blk != b)
            if len(cand) == 0:
                continue
            m = r.choice(cand, min(n_calib, len(cand)), replace=False)
            k, t2 = calibrate_prior(p[m], lor_t[m], se_t[m], fix_scale)
            ks.append(k if not fix_scale else calibrate_prior(p[m], lor_t[m], se_t[m], False)[0])
            mu[blk == b], tau2[blk == b] = k * z[blk == b], t2
        if mode == "tiebreak":
            continue
        for n in ns:
            ca, cb = sims[n]
            lor, se2 = early_lor(ca, n, cb, n)
            d = posterior_lor(mu, tau2, lor, se2)[0] > 0
            per_draw[n].append(np.mean(d == y, axis=0))
            if n == n0:
                dec0.append(d)
    if mode == "bayes":
        acc = {n: np.mean(per_draw[n], axis=0) for n in ns}
    return acc, np.asarray(dec0, bool), np.asarray(ks, float)


def _gain(human_acc, hyb_n0, idx, ns, n0):
    hy = np.maximum.accumulate(np.array([human_acc[n][idx].mean() for n in ns]))
    target = hyb_n0[idx].mean()
    if target > hy.max():
        return np.nan
    return float(np.exp(np.interp(target, hy, np.log(np.asarray(ns, float)))) / n0)


def _guard(name, mode, prior_only, ks, n_calib):
    """Weak-prior warnings: the free slope k spans 0 over the calibration sets, or prior-only accuracy < 0.6."""
    out, who = [], f"prior {name!r}: " if name else ""
    lo, hi = (float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))) if len(ks) else (np.nan, np.nan)
    if mode == "tiebreak":
        if prior_only < 0.5:
            out.append(f"{who}prior-only accuracy {prior_only:.3f} < 0.5: breaking ties with it is worse than a coin")
        return out
    if len(ks) and lo <= 0 <= hi:
        out.append(f"{who}weak prior: the calibration slope k spans 0 over {len(ks)} calibration sets of {n_calib} "
                   f"tests ([{lo:.2f}, {hi:.2f}]); the hybrid can be worse than human-only. Use mode='tiebreak' "
                   "(human decides, prior breaks ties) or more calibration tests")
    if prior_only < WEAK_PRIOR_ACCURACY:
        out.append(f"{who}weak prior: prior-only accuracy {prior_only:.3f} < {WEAK_PRIOR_ACCURACY}; the hybrid can "
                   "be worse than human-only. Use mode='tiebreak' or more calibration tests")
    return out


def _calibrated(priors: dict, ctr_a, ctr_b, clicks_a, clicks_b, n_calib, ns, reps, draws, blocks, n_boot, n0, seed,
                fix_scale, mode, cluster):
    """Shared engine of ``simulate_saving_calibrated`` and ``compare_priors``: one replay, every prior on it."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    ctr_a, ctr_b = np.asarray(ctr_a, float), np.asarray(ctr_b, float)
    y = ctr_a > ctr_b
    n_pairs = len(y)
    if n0 not in ns:
        raise ValueError(f"n0={n0} must be one of ns={tuple(ns)}")
    ps = {k: np.asarray(v, float) for k, v in priors.items()}
    for k, v in ps.items():
        if v.shape != y.shape:
            raise ValueError(f"prior {k!r} has shape {v.shape}, expected ({n_pairs},)")
    lor_t, se_t = true_lor(ctr_a, ctr_b, clicks_a, clicks_b)
    rs, sims, tie, human_dec = _replay(ctr_a, ctr_b, ns, reps, seed)
    human_acc = {n: np.mean(human_dec[n] == y, axis=0) for n in ns}
    groups = _boot_groups(cluster, n_pairs)
    boot = _boot_indices(rs, groups, n_pairs, n_boot)
    ties = sims[n0][0] == sims[n0][1]
    cap = max(ns) / n0
    shared = {"n_pairs": int(n_pairs), "n_calib": int(n_calib), "ns": list(ns), "n0": int(n0), "mode": mode,
              "reps": int(reps), "draws": int(draws), "blocks": int(blocks),
              "human": {str(n): float(human_acc[n].mean()) for n in ns}, "tie_rate": float(ties.mean()),
              "bootstrap": "pair" if groups is None else f"cluster ({len(groups)} clusters)"}
    per, raw = {}, {}
    for name, p in ps.items():
        acc, dec0, ks = _hybrid(p, lor_t, se_t, y, sims, tie, human_dec, ns, n0, n_calib, draws, blocks, fix_scale,
                                mode)
        point = _gain(human_acc, acc[n0], np.arange(n_pairs), ns, n0)
        bs = np.array([_gain(human_acc, acc[n0], i, ns, n0) for i in boot])
        ci, ci_cen, n_cen = _ci(bs, cap)
        prior_only = float(np.mean(np.where(p == .5, .5, (p > .5) == y)))
        rule = float(np.mean(np.where(ties, np.where(p == .5, tie, p > .5), human_dec[n0]) == y))
        censored = not np.isfinite(point)
        lo_lim = float(min(ns) / n0)
        ws = _guard(name if len(ps) > 1 else "", mode, prior_only, ks, n_calib)
        per[name] = {"hybrid": {str(n): float(acc[n].mean()) for n in ns}, "prior_only": prior_only,
                     "multiplier": None if censored else point, "ci": ci,
                     "multiplier_censored": bool(censored),
                     "multiplier_lower_bound": float(cap) if censored else None,
                     "multiplier_at_floor": bool(not censored and point <= lo_lim * (1 + 1e-9)),
                     "ci_censored": ci_cen, "n_boot_censored": n_cen,
                     "human_decides_prior_breaks_ties": rule,
                     "k_calib": {"median": float(np.median(ks)) if len(ks) else None,
                                 "lo": float(np.percentile(ks, 2.5)) if len(ks) else None,
                                 "hi": float(np.percentile(ks, 97.5)) if len(ks) else None,
                                 "n_sets": int(len(ks))},
                     "warnings": ws}
        raw[name] = {"acc": acc, "dec0": dec0, "bs": bs}
    return shared, per, raw, {"y": y, "human_acc": human_acc, "boot": boot, "cap": cap}


def simulate_saving_calibrated(ctr_a, ctr_b, prior_prob, clicks_a=None, clicks_b=None, n_calib: int = 20,
                               ns=NS_CALIBRATED, reps: int = 200, draws: int = 5, blocks: int = 10,
                               n_boot: int = 500, n0: int = 100, seed: int = 5, fix_scale: bool = False,
                               return_pairs: bool = False, mode: str = "bayes", cluster=None) -> dict:
    """Impression saving when only ``n_calib`` finished tests are available to calibrate the prior, with a
    pair-bootstrap 95% CI of the multiplier at ``n0`` impressions/arm (port of `ornek_tasarrufu_llm_ga.py`).

    Pairs are split into ``blocks`` blocks; each block's k and tau^2 come from ``n_calib`` pairs drawn at random from
    the other blocks (one draw per block, ``draws`` independent draws averaged, draw seeds 0..draws-1). Small tests
    are replayed ``reps`` times. The human curve is made monotone (running max) before the inverse interpolation.
    Ties at ``n0``: ``human_decides_prior_breaks_ties`` is the accuracy of "human decides, prior breaks ties".
    Measured (Upworthy llm300, Opus 5.5 direct, n_calib=20): human 0.718 -> hybrid 0.808, 2.27x [1.67, 3.11];
    n_calib=150: 2.58x [1.88, 3.42] (uretim/deney/RAPOR.md 2.2-2.3, `ornek_tasarrufu_llm_ga.json`).

    mode: ``"bayes"`` (default, the normal-normal hybrid) or ``"tiebreak"`` (human decides by the higher sample
        CTR, the prior only breaks ties; no worse than human-only in expectation whenever the prior is right on
        at least half of the tied tests).
    return_pairs: also return ``hybrid_pairs`` / ``human_pairs`` ({str(n): per-pair accuracy}, their means are
        ``hybrid`` / ``human``), ``decisions_n0`` (draws x reps x pairs bool, True = ship A) and ``a_better``.
    cluster: optional test id per pair; the bootstrap then resamples tests, not pairs (several pairs per test).

    Censoring: when the hybrid beats the top of the human curve, ``multiplier`` is None,
    ``multiplier_censored=True`` and ``multiplier_lower_bound = max(ns) / n0``; bootstrap replicates above the
    curve rank above every finite one, and a CI endpoint in that tail is None (``ci_censored``).
    ``multiplier_at_floor``: the hybrid is at or below the human curve's bottom (the value is an upper bound).
    Weak-prior guard: a RuntimeWarning (also in ``warnings``) when the free calibration slope k spans 0 over the
    ``blocks x draws`` calibration sets (``k_calib``: its median and 2.5-97.5% spread), or prior-only accuracy
    < 0.6."""
    shared, per, raw, aux = _calibrated({"prior": prior_prob}, ctr_a, ctr_b, clicks_a, clicks_b, n_calib, ns, reps,
                                        draws, blocks, n_boot, n0, seed, fix_scale, mode, cluster)
    out = {**shared, **per["prior"]}
    for w in out["warnings"]:
        warnings.warn(w, RuntimeWarning, stacklevel=2)
    if return_pairs:
        out["hybrid_pairs"] = {str(n): a for n, a in raw["prior"]["acc"].items()}
        out["human_pairs"] = {str(n): a for n, a in aux["human_acc"].items()}
        out["decisions_n0"] = raw["prior"]["dec0"]
        out["a_better"] = aux["y"]
    return out


def compare_priors(priors: dict, ctr_a, ctr_b, clicks_a=None, clicks_b=None, n_calib: int = 20,
                   ns=NS_CALIBRATED, reps: int = 200, draws: int = 5, blocks: int = 10, n_boot: int = 500,
                   n0: int = 100, seed: int = 5, fix_scale: bool = False, mode: str = "bayes",
                   cluster=None) -> dict:
    """Several priors on ONE replay: same binomial replays, tie coins, calibration sets and bootstrap resamples,
    so their differences are paired.

    priors: ``{name: prior_prob}`` (P(A better) per pair); the first one is the reference. Every prior's entry in
    ``priors`` equals ``simulate_saving_calibrated`` on the same arguments (point values and CI). ``diffs[name]``
    (names after the first): hybrid accuracy at ``n0`` and multiplier minus the reference's, with paired 95%
    bootstrap CIs (pair bootstrap, or test-clustered with ``cluster=`` test ids). ``vs_human[name]``: hybrid minus
    human-only accuracy at ``n0`` with its CI. Multiplier differences drop replicates where either side is
    censored (``n_boot_used``). Weak-prior warnings are raised per prior.

    Usage::

        r = compare_priors({"tfidf": p_tfidf, "llm": p_llm, "stack": p_stack}, ctr_a, ctr_b, n_calib=20)
        r["diffs"]["llm"]            # {"vs": "tfidf", "accuracy", "accuracy_ci", "multiplier", "multiplier_ci", ...}
    """
    if not priors:
        raise ValueError("priors is empty")
    shared, per, raw, aux = _calibrated(dict(priors), ctr_a, ctr_b, clicks_a, clicks_b, n_calib, ns, reps, draws,
                                        blocks, n_boot, n0, seed, fix_scale, mode, cluster)
    names = list(per)
    ref = names[0]
    hum = aux["human_acc"][n0]
    accs = {k: raw[k]["acc"][n0] for k in names}
    boot_acc = {k: np.array([accs[k][i].mean() for i in aux["boot"]]) for k in names}
    boot_hum = np.array([hum[i].mean() for i in aux["boot"]])

    def pct(x):
        x = x[np.isfinite(x)]
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None]

    diffs, vs_human = {}, {}
    for k in names:
        vs_human[k] = {"accuracy": float(accs[k].mean() - hum.mean()), "accuracy_ci": pct(boot_acc[k] - boot_hum)}
        for w in per[k]["warnings"]:
            warnings.warn(w, RuntimeWarning, stacklevel=2)
    for k in names[1:]:
        m, m0 = per[k]["multiplier"], per[ref]["multiplier"]
        dm = raw[k]["bs"] - raw[ref]["bs"]
        diffs[k] = {"vs": ref, "accuracy": float(accs[k].mean() - accs[ref].mean()),
                    "accuracy_ci": pct(boot_acc[k] - boot_acc[ref]),
                    "multiplier": None if m is None or m0 is None else float(m - m0), "multiplier_ci": pct(dm),
                    "n_boot_used": int(np.isfinite(dm).sum())}
    return {**shared, "reference": ref, "human_n0": float(hum.mean()), "priors": per, "diffs": diffs,
            "vs_human": vs_human}


# ------------------------------------------------------------------ card section
def _mult_txt(m, lb=None):
    if m is None:
        return f"> {lb:g}x (above the human curve)" if lb is not None else "—"
    return f"{m:.2f}x"


def _ci_txt(ci, cap):
    lo, hi = ci
    if lo is None and hi is None:
        return "—" if cap is None else f"> {cap:g}x"
    return f"[{_mult_txt(lo)}, {_mult_txt(hi) if hi is not None else f'> {cap:g}x'}]"


def _acc_ci(ci):
    return "—" if ci[0] is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def saving_section(result: dict, title: str | None = None) -> dict:
    """Evidence-card section (same shape as ``simengine.evidence.extras`` sections: kind/title/columns/rows/notes,
    render with ``simengine.evidence.section_markdown``) for a ``simulate_saving_calibrated`` or
    ``compare_priors`` result: accuracy at ``n0``, the impression multiplier with its CI and censoring, n_calib.

    Usage::

        r = simulate_saving_calibrated(ctr_a, ctr_b, prior, n_calib=20)
        md = card_md + "\\n" + section_markdown(saving_section(r))
    """
    n0, cap = result["n0"], max(result["ns"]) / result["n0"]
    head = (f"{result['n_pairs']} test pairs, prior calibrated on {result['n_calib']} finished tests "
            f"({result['blocks']} blocks x {result['draws']} draws, {result['reps']} replays), mode={result['mode']}")
    notes = [head, "Offline replay: the truth is each pair's full-data winner; small tests are resampled from the "
             f"full-data CTRs. CI: {result['bootstrap']} bootstrap, 95%."]
    if "priors" in result:
        ref = result["reference"]
        cols = ["arm", f"accuracy @ {n0}/arm", "vs human (95% CI)", f"vs {ref} (95% CI)", "multiplier",
                "multiplier 95% CI"]
        rows = [["human only", result["human_n0"], None, None, "1.00x", None]]
        for k, r in result["priors"].items():
            vh, d = result["vs_human"][k], result["diffs"].get(k)
            rows.append([k, r["hybrid"][str(n0)], f"{vh['accuracy']:+.3f} {_acc_ci(vh['accuracy_ci'])}",
                         None if d is None else f"{d['accuracy']:+.3f} {_acc_ci(d['accuracy_ci'])}",
                         _mult_txt(r["multiplier"], r["multiplier_lower_bound"]), _ci_txt(r["ci"], cap)])
            notes += r["warnings"]
            if r["multiplier_censored"]:
                notes.append(f"{k}: hybrid accuracy is above the human curve's top ({max(result['ns'])}/arm); "
                             f"the multiplier is at least {cap:g}x.")
        return {"kind": "ab_saving_compare", "title": title or "Impression saving: priors compared",
                "columns": cols, "rows": rows, "notes": notes, "n0": n0, "n_calib": result["n_calib"],
                "priors": {k: {x: r[x] for x in ("multiplier", "ci", "multiplier_censored",
                                                  "multiplier_lower_bound")} for k, r in result["priors"].items()}}
    m, lb, k = result["multiplier"], result["multiplier_lower_bound"], result["k_calib"]
    rows = [["human-only accuracy", result["human"][str(n0)], f"{n0} impressions per arm"],
            ["hybrid accuracy", result["hybrid"][str(n0)], f"mode={result['mode']}"],
            ["prior-only accuracy", result["prior_only"], ""],
            ["human decides, prior breaks ties", result["human_decides_prior_breaks_ties"],
             f"tie rate {result['tie_rate']:.3f}"],
            ["impression multiplier", _mult_txt(m, lb), "human-only impressions needed / n0"],
            ["multiplier 95% CI", _ci_txt(result["ci"], cap),
             f"{result['n_boot_censored']} bootstrap replicates above the curve" if result["n_boot_censored"] else ""],
            ["calibration tests (n_calib)", result["n_calib"], ""],
            ["calibration slope k", k["median"],
             "" if k["lo"] is None else f"2.5-97.5% over {k['n_sets']} sets: [{k['lo']:.2f}, {k['hi']:.2f}]"]]
    if result["multiplier_censored"]:
        notes.append(f"The hybrid beats the human curve's top ({max(result['ns'])}/arm): the multiplier is censored, "
                     f"at least {cap:g}x.")
    if result.get("multiplier_at_floor"):
        notes.append(f"The hybrid is at or below the human curve's bottom ({min(result['ns'])}/arm): "
                     "the multiplier is an upper bound.")
    notes += result["warnings"]
    return {"kind": "ab_saving", "title": title or f"Impression saving at {n0} impressions per arm",
            "columns": ["metric", "value", "note"], "rows": rows, "notes": notes, "n0": n0,
            "n_calib": result["n_calib"], "multiplier": m, "ci": result["ci"],
            "multiplier_censored": result["multiplier_censored"], "multiplier_lower_bound": lb}

