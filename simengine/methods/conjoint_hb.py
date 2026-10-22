"""Person-level taste models for conjoint / discrete choice: hierarchical Bayes MNL and a mixed-logit wrapper.

Same long input as `conjoint.MNL`: one row per (task, alternative) with ``task``, ``alternative``, ``chosen`` (0/1),
optional ``available`` (default 1), ``person`` (required here) and numeric attribute columns; research aliases
gorev_id / alternatif_id / secildi / mevcut / birey_id are accepted. Task ids are unique across persons.

Two prediction targets, as in the conjoint brief:

- **known person, held-out task** (`predict_person`): the person's own posterior draws. This is where person-level
  tastes pay off ([deney 1.5] camera conjoint, same person / new task: MNL 0.482 vs HB 0.715 accuracy);
- **new person** (`predict_new`): the population mixture, beta ~ N(mu, Sigma) integrated over the posterior of
  (mu, Sigma). Close to a pooled MNL there (0.509 vs 0.519).

    from simengine.methods.conjoint_hb import HBMNL, MixedLogit, tune_prior_scale
    hb = HBMNL(["price", "quality"], scale={"price": 100}, draws=1000, burn=1000, seed=0).fit(train_rows)
    hb.predict_person(test_rows_known)        # P(row chosen), person posterior predictive
    hb.predict_new(test_rows_new)             # P(row chosen), population mixture
    hb.log_likelihood(test_rows_known)        # mean per task; new=True for the population mixture
    hb.draws_, hb.beta_, hb.mu_, hb.sigma_    # (draws, persons, K) betas; posterior means
    best, scores = tune_prior_scale(train_rows, attrs=["price", "quality"], scale={"price": 100})  # few tasks
    mx = MixedLogit(["price", "quality"], randvars={"price": "n", "quality": "n"}).fit(train_rows)  # needs xlogit
    mx.predict_proba(test_rows_new)

`HBMNL` is the Allenby-Rossi sampler (numpy only): Gibbs over (betas, mu, Sigma) with a random-walk Metropolis step
per person (proposal covariance step_i^2 Sigma, step_i adapted toward ``target_accept`` during burn-in, then fixed),
mu | . normal under a N(0, prior_var I) prior, Sigma | . inverse Wishart IW(nu + N, nu I + S) with nu = K + 3.
All parameters (attributes and ASCs) are random. Deterministic for a fixed ``seed``. The priors are unit-scale, so
attributes should be near unit sd after ``scale`` (fit warns outside 0.1-10): an unscaled price (0-200) got a
population sd 40x the true one and twice the person-level beta error. With few tasks per person the prior pulls
``sigma_`` up (true sd 0.6 -> about 0.78 with 10 tasks, 0.67 with 40, K = 4; 90% intervals covered the true sd
26% / 70% of the time over 50 / 30 simulated panels); ``mu_`` 90% intervals covered 80-92% in the same runs.

`MixedLogit` wraps ``xlogit.MixedLogit`` (an optional package; ImportError if missing). It sorts rows by person /
task / alternative and pads missing alternatives as unavailable before passing panels: xlogit itself fails with an
opaque ``IndexError`` on rows that are not contiguous per person [prova-d8 #6].
"""
from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .conjoint import _prep

RESERVED = {"task", "alternative", "chosen", "available", "person"}


def _check(df: pd.DataFrame, who: str) -> pd.DataFrame:
    missing = [c for c in ("task", "alternative", "chosen", "person") if c not in df.columns]
    if missing:
        raise ValueError(f"{who}: missing column(s) {missing}; long format is one row per (task, alternative) "
                         "with task, alternative, chosen (0/1), person")
    n = df.groupby("task")["chosen"].sum()
    bad = n[n != 1]
    if len(bad):
        raise ValueError(f"{who}: {len(bad)} task(s) do not have exactly one chosen available alternative (e.g. "
                         f"task {bad.index[0]!r}: {int(bad.iloc[0])})")
    pp = df.groupby("task")["person"].nunique()
    if (pp > 1).any():
        raise ValueError(f"{who}: task {pp[pp > 1].index[0]!r} spans several persons; task ids must be unique "
                         "across persons")
    return df


def _attrs(df: pd.DataFrame, attrs, who: str) -> list:
    if attrs is not None:
        out = list(attrs)
        missing = [a for a in out if a not in df.columns]
        if missing:
            raise ValueError(f"{who}: missing attribute column(s) {missing}")
        return out
    out = [c for c in df.columns if c not in RESERVED and pd.api.types.is_numeric_dtype(df[c])]
    if not out:
        raise ValueError(f"{who}: no numeric attribute columns; pass attrs=[...]")
    return out


def _logp(V: np.ndarray, g: np.ndarray, av: np.ndarray) -> np.ndarray:
    """Log choice probability per row, softmax within task ``g``; V is (rows,) or (rows, R)."""
    one = V.ndim == 1
    V = V[:, None] if one else V
    V = np.where(av[:, None] > 0, V, -1e9)
    G = int(g.max()) + 1
    vmax = np.full((G, V.shape[1]), -np.inf)
    np.maximum.at(vmax, g, V)
    E = np.exp(V - vmax[g])
    den = np.zeros((G, V.shape[1]))
    np.add.at(den, g, E)
    out = V - vmax[g] - np.log(den[g])
    return out[:, 0] if one else out


def _iwishart(df: float, scale: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sigma ~ IW(df, scale) via Bartlett: Sigma^-1 ~ Wishart(df, scale^-1)."""
    K = len(scale)
    C = np.linalg.cholesky(np.linalg.inv(scale))
    A = np.zeros((K, K))
    A[np.diag_indices(K)] = np.sqrt(rng.chisquare(df - np.arange(K)))
    A[np.tril_indices(K, -1)] = rng.standard_normal(K * (K - 1) // 2)
    CA = C @ A
    return np.linalg.inv(CA @ CA.T)


def _task_scores(d: pd.DataFrame, p: np.ndarray) -> tuple[float, float]:
    """(mean log-likelihood per task, argmax accuracy) from per-row probabilities."""
    d = d.assign(_p=p)
    d = d[d["available"] > 0]
    ch = d[d["chosen"] > 0]
    ll = float(np.log(np.clip(ch["_p"].to_numpy(), 1e-300, None)).sum() / ch["task"].nunique())
    acc = float(d.loc[d.groupby("task")["_p"].idxmax(), "chosen"].mean())
    return ll, acc


class HBMNL:
    """Hierarchical Bayes MNL: fit(df) / predict_person(df) / predict_new(df) / log_likelihood / accuracy.

    ``attrs``: attribute columns (None -> every numeric column other than task/alternative/chosen/available/person);
    ``scale``: {attr: divisor}; ``base_alt``: alternative without an ASC (default smallest id); ``asc``: include
    ASCs (random like the attributes); ``draws``: kept draws after ``burn`` (every ``thin``-th); ``seed``;
    ``prior_scale``: Sigma prior scale (IW scale nu * prior_scale * I; smaller shrinks persons toward ``mu``),
    choose it with `tune_prior_scale` when persons have few tasks.

    Fitted: ``names_`` (parameter names, attributes then ``ASC_<alt>``), ``persons_`` (person ids, row order of
    the draws), ``draws_`` (n_draws, n_persons, K) individual betas, ``mu_draws_`` (n_draws, K),
    ``sigma_draws_`` (n_draws, K, K), posterior means ``beta_`` (DataFrame persons x names), ``mu_`` (Series),
    ``sigma_`` (DataFrame), ``acceptance_`` (mean Metropolis acceptance after burn-in).
    """

    def __init__(self, attrs: Sequence[str] | None = None, scale: Mapping[str, float] | None = None,
                 base_alt=None, asc: bool = True, draws: int = 500, burn: int = 500, thin: int = 1, seed: int = 0,
                 prior_var: float = 100.0, nu: float | None = None, prior_scale: float = 1.0,
                 target_accept: float = 0.3):
        self.attrs, self.scale, self.base_alt, self.asc = attrs, dict(scale or {}), base_alt, asc
        self.draws, self.burn, self.thin, self.seed = int(draws), int(burn), max(1, int(thin)), seed
        self.prior_var, self.nu, self.target_accept = prior_var, nu, target_accept
        self.prior_scale = float(prior_scale)

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = [df[a].astype(float).fillna(0).to_numpy() / self.scale.get(a, 1.0) for a in self.attrs_]
        if self.asc:
            X += [(df["alternative"] == j).astype(float).to_numpy() for j in self.alts_ if j != self.base_alt_]
        return np.column_stack(X)

    def fit(self, df: pd.DataFrame):
        df = _prep(df)
        df = _check(df[df["available"] > 0].reset_index(drop=True), "HBMNL")
        self.attrs_ = _attrs(df, self.attrs, "HBMNL")
        self.alts_ = sorted(df["alternative"].unique())
        self.base_alt_ = self.base_alt if self.base_alt is not None else self.alts_[0]
        self.names_ = list(self.attrs_) + ([f"ASC_{j}" for j in self.alts_ if j != self.base_alt_] if self.asc
                                           else [])
        X = self._design(df)
        sd = X[:, :len(self.attrs_)].std(0)
        off = [f"{a} (sd {v:.3g})" for a, v in zip(self.attrs_, sd) if v > 10 or 0 < v < 0.1]
        if off:                                              # unit-scale IW prior: heterogeneity would be inflated
            warnings.warn(f"HBMNL: attribute(s) far from unit scale: {', '.join(off)}; the priors assume unit-scale "
                          "attributes, pass scale={attr: divisor} (e.g. the sd)", stacklevel=2)
        g = pd.factorize(df["task"])[0]
        pid, persons = pd.factorize(df["person"])
        y, av = df["chosen"].to_numpy(float), np.ones(len(df))
        N, K = len(persons), X.shape[1]
        rng = np.random.default_rng(self.seed)
        nu = self.nu if self.nu is not None else K + 3
        V0, A, I = nu * self.prior_scale * np.eye(K), 1.0 / self.prior_var, np.eye(K)

        def ll_person(B):
            return np.bincount(pid, weights=_logp((X * B[pid]).sum(1), g, av) * y, minlength=N)

        B, mu, Sig = np.zeros((N, K)), np.zeros(K), np.eye(K)
        cur = ll_person(B)
        step = np.full(N, 2.38 / np.sqrt(K))
        win, acc_win, acc_post = 50, np.zeros(N), np.zeros(N)
        keep = []
        for it in range(self.burn + self.draws * self.thin):
            L, Sinv = np.linalg.cholesky(Sig), np.linalg.inv(Sig)
            prop = B + step[:, None] * (rng.standard_normal((N, K)) @ L.T)
            new = ll_person(prop)
            d0, d1 = B - mu, prop - mu
            log_r = (new - 0.5 * np.einsum("ij,jk,ik->i", d1, Sinv, d1)
                     - cur + 0.5 * np.einsum("ij,jk,ik->i", d0, Sinv, d0))
            ok = np.log(rng.random(N)) < log_r
            B[ok], cur[ok] = prop[ok], new[ok]
            # mu | B, Sigma ~ N(cov Sinv sum(B), cov), cov = (N Sinv + A I)^-1
            cov = np.linalg.inv(N * Sinv + A * I)
            mu = cov @ (Sinv @ B.sum(0)) + np.linalg.cholesky(cov) @ rng.standard_normal(K)
            D = B - mu
            Sig = _iwishart(nu + N, V0 + D.T @ D, rng)
            if it < self.burn:
                acc_win += ok
                if (it + 1) % win == 0:                      # Robbins-Monro style step adaptation, frozen after burn
                    step *= np.exp(acc_win / win - self.target_accept)
                    acc_win[:] = 0
            else:
                acc_post += ok
                if (it - self.burn) % self.thin == 0:
                    keep.append((B.copy(), mu.copy(), Sig.copy()))
        self.persons_ = np.asarray(persons)
        self.draws_ = np.stack([k[0] for k in keep])
        self.mu_draws_ = np.stack([k[1] for k in keep])
        self.sigma_draws_ = np.stack([k[2] for k in keep])
        self.beta_ = pd.DataFrame(self.draws_.mean(0), index=self.persons_, columns=self.names_)
        self.mu_ = pd.Series(self.mu_draws_.mean(0), index=self.names_)
        self.sigma_ = pd.DataFrame(self.sigma_draws_.mean(0), index=self.names_, columns=self.names_)
        self.acceptance_ = float(acc_post.sum() / max(1, N * self.draws * self.thin))
        self.step_ = step
        return self

    def _rows(self, df):
        d = _prep(df).reset_index(drop=True)
        return d, self._design(d), pd.factorize(d["task"])[0], d["available"].to_numpy(float)

    def predict_person(self, df: pd.DataFrame, unknown: str = "raise", chunk: int = 50) -> np.ndarray:
        """P(row chosen) for known persons from their own posterior draws, in input row order.

        ``unknown``: "raise" (default) for rows of persons not seen in fit, or "population" to score those
        rows with `predict_new`."""
        d, X, g, av = self._rows(df)
        if "person" not in d.columns:
            raise ValueError("HBMNL.predict_person: needs a 'person' column; use predict_new for new persons")
        codes = pd.Index(self.persons_).get_indexer(d["person"])
        unk = codes < 0
        if unk.any() and unknown != "population":
            raise ValueError(f"HBMNL.predict_person: {d.loc[unk, 'person'].nunique()} person(s) not seen in fit "
                             f"(e.g. {d.loc[unk, 'person'].iloc[0]!r}); use predict_new or unknown='population'")
        out = np.zeros(len(d))
        k = ~unk
        if k.any():
            Xk, ck, avk = X[k], codes[k], av[k]
            gk = pd.factorize(g[k])[0]
            tot = np.zeros(k.sum())
            for s in range(0, len(self.draws_), chunk):
                Bt = self.draws_[s:s + chunk][:, ck]                  # (t, rows, K)
                tot += np.exp(_logp(np.einsum("rk,trk->rt", Xk, Bt), gk, avk)).sum(1)
            out[k] = tot / len(self.draws_)
        if unk.any():
            out[unk] = self.predict_new(d[unk].reset_index(drop=True))
        return out

    def _pop_betas(self, n: int) -> np.ndarray:
        rng = np.random.default_rng([int(self.seed) & 0xFFFFFFFF, 1])
        T = len(self.mu_draws_)
        ts = np.arange(T) if n >= T else np.sort(rng.choice(T, n, replace=False))
        per = max(1, n // len(ts))
        out = [self.mu_draws_[t] + rng.standard_normal((per, len(self.names_))) @
               np.linalg.cholesky(self.sigma_draws_[t]).T for t in ts]
        return np.concatenate(out)

    def predict_new(self, df: pd.DataFrame, n: int = 2000, chunk: int = 250) -> np.ndarray:
        """P(row chosen) for a new person: population mixture over ``n`` betas drawn from N(mu_t, Sigma_t)."""
        d, X, g, av = self._rows(df)
        Bp = self._pop_betas(n)
        tot = np.zeros(len(d))
        for s in range(0, len(Bp), chunk):
            tot += np.exp(_logp(X @ Bp[s:s + chunk].T, g, av)).sum(1)
        return tot / len(Bp)

    def log_likelihood(self, df: pd.DataFrame, new: bool = False) -> float:
        """Mean log-likelihood per task; ``new=True`` scores with the population mixture."""
        p = self.predict_new(df) if new else self.predict_person(df)
        return _task_scores(_prep(df).reset_index(drop=True), p)[0]

    def accuracy(self, df: pd.DataFrame, new: bool = False) -> float:
        p = self.predict_new(df) if new else self.predict_person(df)
        return _task_scores(_prep(df).reset_index(drop=True), p)[1]



def tune_prior_scale(df: pd.DataFrame, grid: Sequence[float] = (1.0, 0.3, 0.1, 0.03), seed: int = 0,
                     **hb_kw) -> tuple[float, pd.Series]:
    """Pick ``prior_scale`` inside the training rows: hold out one random task of every person with 2+ tasks,
    fit `HBMNL` on the rest for each value in ``grid``, score the held-out tasks with `predict_person` (mean
    log-likelihood per task). Returns (best scale, Series scale -> score); refit on all rows with the best one.
    ``hb_kw`` goes to `HBMNL` (attrs, scale, draws, burn, ...). With few tasks per person and many dummy levels
    the default 1.0 can lose to a pooled MNL; compare against `MNL` on the same split too.

        best, scores = tune_prior_scale(train_rows, attrs=ATTRS, draws=500, burn=1000)
        hb = HBMNL(ATTRS, prior_scale=best, draws=1000, burn=1000).fit(train_rows)
    """
    d = _check(_prep(df).reset_index(drop=True), "tune_prior_scale")
    tasks = d.groupby("task", sort=False)["person"].first()
    rng = np.random.default_rng(seed)
    held = [rng.choice(t.index.to_numpy()) for _, t in tasks.groupby(tasks) if len(t) >= 2]
    if not held:
        raise ValueError("tune_prior_scale: no person has 2+ tasks to hold one out")
    va = d["task"].isin(held)
    out = pd.Series({float(s): HBMNL(prior_scale=s, seed=seed, **hb_kw).fit(d[~va]).log_likelihood(d[va])
                     for s in grid}, name="loglik_per_task")
    out.index.name = "prior_scale"
    return float(out.idxmax()), out


def _require_xlogit():
    try:
        import xlogit  # noqa: F401
    except ImportError as e:
        raise ImportError("MixedLogit needs the optional 'xlogit' package (pip install xlogit); HBMNL needs only "
                          "numpy") from e
    return xlogit


class MixedLogit:
    """Mixed logit (simulated ML, panel by person) through ``xlogit``: fit(df) / predict_proba(df) /
    log_likelihood(df) / accuracy(df). Predictions are population (new-person) probabilities.

    ``attrs`` (None -> numeric non-reserved columns), ``randvars`` {attr: "n" | "ln" | "t" | "tn" | "u"} (default:
    every attribute normal), ``scale`` {attr: divisor}, ``asc`` (fixed ASCs), ``base_alt``, ``n_draws``, ``seed``;
    ``fit_kw`` is passed to ``xlogit.MixedLogit.fit``. Fitted: ``model_`` (the xlogit model), ``coef_`` (Series),
    ``alts_``, ``attrs_``.
    """

    def __init__(self, attrs: Sequence[str] | None = None, randvars: Mapping[str, str] | None = None,
                 scale: Mapping[str, float] | None = None, asc: bool = True, base_alt=None, n_draws: int = 200,
                 seed: int = 0, **fit_kw):
        self.attrs, self.randvars, self.scale = attrs, randvars, dict(scale or {})
        self.asc, self.base_alt, self.n_draws, self.seed, self.fit_kw = asc, base_alt, n_draws, seed, fit_kw

    def _long(self, d: pd.DataFrame) -> pd.DataFrame:
        """Pad every task to all alternatives (pads unavailable) and sort person / task / alternative."""
        d = d.assign(_row=np.arange(len(d)))
        tp = d.groupby("task", sort=False)["person"].first()
        grid = pd.MultiIndex.from_product([tp.index, self.alts_], names=["task", "alternative"]).to_frame(index=False)
        full = grid.merge(d, on=["task", "alternative"], how="left")
        full["person"] = full["task"].map(tp)
        full["_row"] = full["_row"].fillna(-1).astype(int)
        full["available"] = full["available"].fillna(0)
        full["chosen"] = full["chosen"].fillna(0).astype(int)
        for a in self.attrs_:
            full[a] = full[a].astype(float).fillna(0) / self.scale.get(a, 1.0)
        full = full.sort_values(["person", "task", "alternative"], kind="mergesort").reset_index(drop=True)
        full["_pid"] = pd.factorize(full["person"])[0]
        full["_tid"] = pd.factorize(full["task"])[0]
        return full

    def fit(self, df: pd.DataFrame):
        xl = _require_xlogit()
        d = _check(_prep(df).query("available > 0").reset_index(drop=True), "MixedLogit")
        self.attrs_ = _attrs(d, self.attrs, "MixedLogit")
        self.alts_ = sorted(d["alternative"].unique())
        self.base_alt_ = self.base_alt if self.base_alt is not None else self.alts_[0]
        rv = dict(self.randvars) if self.randvars is not None else {a: "n" for a in self.attrs_}
        f = self._long(d)
        self.model_ = xl.MixedLogit()
        self.model_.fit(X=f[self.attrs_].to_numpy(float), y=f["chosen"].to_numpy(int), varnames=self.attrs_,
                        alts=f["alternative"].to_numpy(), ids=f["_tid"].to_numpy(), randvars=rv,
                        panels=f["_pid"].to_numpy(), avail=f["available"].to_numpy(int),
                        fit_intercept=self.asc, base_alt=self.base_alt_, n_draws=self.n_draws,
                        random_state=self.seed, verbose=0, **self.fit_kw)
        self.coef_ = pd.Series(np.asarray(self.model_.coeff_), index=[str(n) for n in self.model_.coeff_names])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(row chosen) in input row order (population mixture; alternatives unseen in fit get 0)."""
        _require_xlogit()
        d = _prep(df).reset_index(drop=True)
        f = self._long(d.assign(person=d["person"] if "person" in d.columns else d["task"]))
        P = self.model_.predict(X=f[self.attrs_].to_numpy(float), varnames=self.attrs_,
                                alts=f["alternative"].to_numpy(), ids=f["_tid"].to_numpy(),
                                panels=f["_pid"].to_numpy(), avail=f["available"].to_numpy(int),
                                n_draws=self.n_draws, random_state=self.seed, verbose=0, return_proba=True)[1]
        p = np.asarray(P).reshape(-1)
        out = np.zeros(len(d))
        m = f["_row"].to_numpy() >= 0
        out[f["_row"].to_numpy()[m]] = p[m]
        return np.where(d["available"].to_numpy() > 0, out, 0.0)

    def log_likelihood(self, df: pd.DataFrame) -> float:
        return _task_scores(_prep(df).reset_index(drop=True), self.predict_proba(df))[0]

    def accuracy(self, df: pd.DataFrame) -> float:
        return _task_scores(_prep(df).reset_index(drop=True), self.predict_proba(df))[1]
