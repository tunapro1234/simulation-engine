"""A/B winner prediction: which of two variants (e.g. headlines) wins? Every method returns P(A wins).

Ported from the deney kit (`uretim/deney/kod/kit/yontemler/ab.py`). Input frame columns (Turkish aliases accepted):
    text_a (metin_a), text_b (metin_b), y (1 = A won; train/dev only), group (grup; test id, optional),
    significant (anlamli; optional, 1 = the test result was statistically significant).

Measured on Upworthy, the same 300 significant headline tests (chance 0.50; uretim/deney/RAPOR.md 1.1):
    longer headline wins 0.533 · TF-IDF on past tests 0.773 · Opus 5.5 direct 0.765 ·
    stack (TF-IDF + Opus) 0.817 [0.770, 0.860], AUC 0.906; story-clean training 0.807 [0.760, 0.847];
    blind one-shot on 136 new pairs 0.824 [0.757, 0.890].
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ._util import logit, parse_json, rename_aliases, sigmoid

ALIASES = {"metin_a": "text_a", "metin_b": "text_b", "grup": "group", "anlamli": "significant", "gercek": "y"}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_aliases(pd.DataFrame(df), ALIASES)
    for c in ("text_a", "text_b"):
        if c not in df:
            raise KeyError(f"A/B frame needs column {c!r} (or its alias)")
    return df


def _texts(df, col):
    return df[col].fillna("").astype(str)


class LengthAB:
    """Heuristic baseline: the longer text wins (0.6 / 0.4 / 0.5 on ties). Upworthy llm300: 0.533 [0.477, 0.588]."""

    def fit(self, train=None):
        return self

    def predict(self, test):
        test = _prep(test)
        d = _texts(test, "text_a").str.len().to_numpy() - _texts(test, "text_b").str.len().to_numpy()
        return np.where(d > 0, 0.6, np.where(d < 0, 0.4, 0.5))


class TfidfAB:
    """Learner on past tests: word 1-2 gram + char_wb 2-5 gram TF-IDF difference vector (A - B), logistic regression
    without intercept, C=0.3 (Bradley-Terry symmetry: swapping A and B flips the prediction exactly).
    Measured: 0.773 [0.727, 0.820] on llm300; 0.760 [0.752, 0.768] in 5-fold story-grouped CV over 19,656 pairs;
    0.720 when same-story training pairs are removed (uretim/deney/RAPOR.md 1.1-1.2).
    ``min_df`` defaults (3 word / 5 char) are the measured ones; they are lowered automatically on tiny data."""

    def __init__(self, C: float = 0.3, min_df_word: int = 3, min_df_char: int = 5, max_features: int = 200_000):
        self.C, self.min_df_word, self.min_df_char, self.max_features = C, min_df_word, min_df_char, max_features

    def _X(self, df):
        from scipy.sparse import hstack
        f = lambda t: hstack([self.w_.transform(t), self.c_.transform(t)]).tocsr()  # noqa: E731
        return f(_texts(df, "text_a")) - f(_texts(df, "text_b"))

    def fit(self, train):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        train = _prep(train)
        tx = pd.unique(np.concatenate([_texts(train, "text_a").to_numpy(), _texts(train, "text_b").to_numpy()]))
        small = len(tx) < 200
        self.w_ = TfidfVectorizer(ngram_range=(1, 2), min_df=1 if small else self.min_df_word,
                                  max_features=self.max_features, sublinear_tf=True).fit(tx)
        self.c_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1 if small else self.min_df_char,
                                  max_features=self.max_features, sublinear_tf=True).fit(tx)
        self.m_ = LogisticRegression(C=self.C, fit_intercept=False, max_iter=3000).fit(
            self._X(train), train["y"].to_numpy().astype(int))
        return self

    def predict(self, test):
        return self.m_.predict_proba(self._X(_prep(test)))[:, 1]


class TfidfWordAB:
    """Older kit variant (for comparison): word 1-2 grams, LR with intercept C=1, symmetric duplication.
    Measured 0.760 [0.710, 0.807] on llm300 (uretim/deney/RAPOR.md 3, kit rehearsal)."""

    def fit(self, train):
        from scipy.sparse import vstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        train = _prep(train)
        a, b = _texts(train, "text_a"), _texts(train, "text_b")
        self.v_ = TfidfVectorizer(ngram_range=(1, 2), min_df=2 if len(train) >= 100 else 1, sublinear_tf=True)
        self.v_.fit(np.concatenate([a.to_numpy(), b.to_numpy()]))
        X = self.v_.transform(a) - self.v_.transform(b)
        y = train["y"].to_numpy().astype(int)
        self.m_ = LogisticRegression(C=1.0, max_iter=2000).fit(vstack([X, -X]), np.concatenate([y, 1 - y]))
        return self

    def predict(self, test):
        test = _prep(test)
        X = self.v_.transform(_texts(test, "text_a")) - self.v_.transform(_texts(test, "text_b"))
        return self.m_.predict_proba(X)[:, 1]


DEFAULT_CONTEXT = "A/B test of two variants shown to real users; outcome = click-through rate."
SYSTEM = "You are a careful forecaster. Reply with JSON only."


class LLMDirectAB:
    """Ask the LLM directly: probability that A gets the higher click rate, 15 pairs per call. With ``reverse=True``
    (default) the same pairs are asked again with A and B swapped and P(A) = mean(p, 1 - p_swapped) (position bias).
    Invalid or missing answers stay at 0.5 and are counted in ``coverage_`` (share of valid predictions).

    Measured (uretim/deney/RAPOR.md 1.1, llm300): Opus 5.5 0.765 [0.720, 0.808], Sonnet 5.5 0.688, Haiku 5.5 0.608.
    Works with any ``simengine.types.LLM`` (use ``FakeLLM`` offline)."""

    def __init__(self, llm, context: str | None = None, batch: int = 15, reverse: bool = True,
                 max_tokens: int = 1024, parallel: int = 4, **llm_kw: Any):
        self.llm, self.context, self.batch, self.reverse = llm, context, batch, reverse
        self.max_tokens, self.parallel, self.llm_kw = max_tokens, parallel, llm_kw
        self.calls_ = 0

    def fit(self, train=None):
        return self

    def prompt(self, rows: pd.DataFrame) -> str:
        lines = "\n".join(f'{i}. A: "{a}" | B: "{b}"' for i, (a, b) in enumerate(zip(rows.text_a, rows.text_b)))
        return (f"Context: {self.context or DEFAULT_CONTEXT}\n"
                "For each pair, estimate the probability that variant A gets the HIGHER click rate than B in the real test.\n"
                "Think about what real people actually click, not what is 'better writing'.\n"
                f"{lines}\n\nReturn JSON list: [{{\"i\": 0, \"p_a\": 0.63}}, ...] for all {len(rows)} pairs.")

    def _run(self, df):
        n = len(df)
        chunks = [df.iloc[i:i + self.batch] for i in range(0, n, self.batch)]
        outs = self.llm.map([(SYSTEM, self.prompt(c)) for c in chunks], max_tokens=self.max_tokens,
                            parallel=self.parallel, **self.llm_kw)
        self.calls_ += len(chunks)
        p, ok = np.full(n, 0.5), np.zeros(n, bool)
        for ci, (ch, r) in enumerate(zip(chunks, outs)):
            try:
                items = parse_json(r.text)
            except (ValueError, TypeError):
                continue
            for d in items if isinstance(items, list) else []:
                try:
                    j, v = int(d["i"]), float(d["p_a"])
                except (KeyError, ValueError, TypeError):
                    continue
                if 0 <= j < len(ch) and 0 <= v <= 1:
                    p[ci * self.batch + j], ok[ci * self.batch + j] = v, True
        return p, ok

    def predict(self, test):
        test = _prep(test).reset_index(drop=True)
        p1, g1 = self._run(test)
        if not self.reverse:
            self.valid_, self.coverage_ = g1, float(g1.mean()) if len(g1) else np.nan
            return p1
        p2, g2 = self._run(test.rename(columns={"text_a": "text_b", "text_b": "text_a"}))
        p2 = 1 - p2
        p = np.where(g1 & g2, (p1 + p2) / 2, np.where(g1, p1, np.where(g2, p2, 0.5)))
        self.valid_ = g1 | g2
        self.coverage_ = float(self.valid_.mean()) if len(test) else np.nan
        return p


class LogitAverageAB:
    """Fixed logit average of a learner and an LLM (weight 0.5, no tuning). llm300: 0.803 [0.757, 0.847]."""

    def __init__(self, learner, llm_method, w: float = 0.5):
        self.learner, self.llm_method, self.w = learner, llm_method, w

    def fit(self, train):
        self.learner.fit(train)
        return self

    def predict(self, test):
        z = self.w * logit(self.learner.predict(test)) + (1 - self.w) * logit(self.llm_method.predict(test))
        return sigmoid(z)


class ABStack:
    """The best measured A/B arm (deney `HibritIstif`): no-intercept logistic regression on
    [logit P_learner, logit P_llm], weights learned on a dev set (Bradley-Terry symmetry is kept).

    dev source, in order: ``dev`` frame passed to ``fit`` -> else ``dev_n`` groups held out of training (one pair per
    group, significant pairs preferred; the learner's dev prediction comes from a model that did not see those
    groups) -> else the stored Upworthy dev300 weights (1.3, 2.75) with a warning (untuned on other data).
    Invalid LLM predictions count as logit 0 (only the learner speaks for that pair).

    ``learner`` defaults to :class:`TfidfAB`; ``llm_method`` is any object with ``predict`` and (optionally)
    ``valid_`` such as :class:`LLMDirectAB`, or ``None`` with ``llm_prob`` columns supplied via ``predict_from``.
    Measured: 0.817 [0.770, 0.860] llm300; blind one-shot 0.824 [0.757, 0.890], +7.4 [0.7, 14.7] over TF-IDF
    (uretim/deney/RAPOR.md 1.1, 1.3). Caveat: much of the gain over the LLM alone is the learner breaking the
    LLM's exact-0.5 ties; on the LOLA v1 setup the stack with Opus (0.775) was not better than embeddings+TF-IDF.
    """

    MIN_DEV = 20
    W0 = np.array([1.3, 2.75])

    def __init__(self, llm_method=None, learner=None, dev_n: int = 200, seed: int = 7):
        self.llm_method, self.learner = llm_method, learner if learner is not None else TfidfAB()
        self.dev_n, self.seed = dev_n, seed
        self.w_, self.note_ = self.W0.copy(), ""

    def _default(self, why):
        self.w_ = self.W0.copy()
        self.note_ = f"stack not learned ({why}); stored weights (1.3, 2.75) from Upworthy dev300"
        warnings.warn(f"ABStack: {why}; using stored Upworthy dev300 weights learner 1.3 / llm 2.75 "
                      "(untuned on this data)", stacklevel=3)

    @staticmethod
    def learn_weights(p_learner, p_llm, y, valid=None) -> np.ndarray:
        """No-intercept LR on [logit p_learner, logit p_llm] (rows with valid LLM predictions only)."""
        from sklearn.linear_model import LogisticRegression
        m = np.ones(len(y), bool) if valid is None else np.asarray(valid, bool)
        X = np.c_[logit(p_learner), logit(p_llm)][m]
        return LogisticRegression(fit_intercept=False, C=1.0).fit(X, np.asarray(y, int)[m]).coef_[0]

    def fit_weights(self, p_learner, p_llm, y, valid=None):
        """Learn the stack weights from precomputed dev predictions (e.g. cached LLM answers)."""
        valid = np.ones(len(y), bool) if valid is None else np.asarray(valid, bool)
        if valid.sum() < self.MIN_DEV or len(np.unique(np.asarray(y)[valid])) < 2:
            self._default(f"only {int(valid.sum())} valid LLM dev predictions (< {self.MIN_DEV})")
            return self
        self.w_ = self.learn_weights(p_learner, p_llm, y, valid)
        self.note_ = f"stack w(learner, llm)=({self.w_[0]:.2f}, {self.w_[1]:.2f}); precomputed dev"
        return self

    def fit(self, train, dev=None):
        train = _prep(train)
        p_l_dev = None
        if dev is not None and len(dev):
            dev, src = _prep(dev).reset_index(drop=True), f"dev frame, {len(dev)} pairs"
        elif self.dev_n > 0 and "group" in train and train.group.nunique() >= 2 * self.dev_n:
            pool = train
            if "significant" in train and (train.significant == 1).sum() >= self.dev_n:
                pool = train[train.significant == 1]
            g = np.random.default_rng(self.seed).choice(pool.group.unique(), min(self.dev_n, pool.group.nunique()),
                                                        replace=False)
            dev = pool[pool.group.isin(g)].groupby("group").sample(1, random_state=self.seed).reset_index(drop=True)
            p_l_dev = type(self.learner)().fit(train[~train.group.isin(g)]).predict(dev)  # leak-free
            src = f"{len(dev)} pairs held out of training"
        else:
            dev = None
        self.learner.fit(train)
        if dev is None or self.llm_method is None:
            self._default("no dev set and training too small to split" if dev is None else "no llm_method")
            return self
        if p_l_dev is None:
            p_l_dev = self.learner.predict(dev)
        p_llm_dev = self.llm_method.predict(dev)
        valid = getattr(self.llm_method, "valid_", np.ones(len(dev), bool))
        y = dev["y"].to_numpy().astype(int)
        if valid.sum() < self.MIN_DEV or len(np.unique(y[valid])) < 2:
            self._default(f"only {int(valid.sum())} valid LLM dev predictions (< {self.MIN_DEV})")
            return self
        self.w_ = self.learn_weights(p_l_dev, p_llm_dev, y, valid)
        self.note_ = f"stack w(learner, llm)=({self.w_[0]:.2f}, {self.w_[1]:.2f}); {src}"
        return self

    def combine(self, p_learner, p_llm, valid=None) -> np.ndarray:
        valid = np.ones(len(p_llm), bool) if valid is None else np.asarray(valid, bool)
        z = np.c_[logit(p_learner), np.where(valid, logit(p_llm), 0.0)] @ self.w_
        return sigmoid(z)

    def predict(self, test):
        test = _prep(test)
        pt = self.learner.predict(test)
        if self.llm_method is None:
            raise ValueError("ABStack.predict needs llm_method; use combine(p_learner, p_llm) for precomputed LLM")
        pl = self.llm_method.predict(test)
        return self.combine(pt, pl, getattr(self.llm_method, "valid_", None))
