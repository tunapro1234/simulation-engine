"""upworthy: which of two headlines got the higher click-through rate in an Upworthy A/B test (Matias et al. 2021).

Ported from the deney branch (`kod/eval/build_upworthy.py`, `data_upworthy.py`, `etki/kod/d1_upworthy.py`,
`etki/kod/d1_sizinti.py`). Pairs = all arm pairs of a test whose packages differ only in the headline
(`headline_only`), ties dropped, A/B order randomised once (default_rng(42)). Test pairs must be significant after a
Benjamini-Hochberg correction within the archive split (`signif_fdr`).

Splits: 'temporal' (default; d1_sizinti `zaman`): tests ordered by first date, the newest 20% of tests are test,
earlier tests train. 'holdout': the archive's official exploratory + confirmatory -> holdout split.
Metric: pair accuracy (p = 0.5 scores half), cluster bootstrap over tests (deney metrics: B = 2000, seed 0).
"""
from __future__ import annotations

import itertools
import os
import re

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, DataNotFound, Metric, need, register
from ..stats import auc, cluster_bootstrap, pair_accuracy

RAW_SPLITS = ["exploratory", "confirmatory", "holdout"]
RAW_NAME = "upworthy-archive-{}-packages-03.12.2020.csv"


def _norm(s):
    return re.sub(r"\s+", " ", s).strip().lower() if isinstance(s, str) else ""


def bh_reject(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up (same decisions as statsmodels multipletests 'fdr_bh')."""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    o = np.argsort(p, kind="mergesort")
    ok = p[o] <= alpha * np.arange(1, n + 1) / n
    rej = np.zeros(n, bool)
    if ok.any():
        rej[o[: np.flatnonzero(ok).max() + 1]] = True
    return rej


def two_prop_p(c1, n1, c2, n2):
    from scipy import stats
    p = (c1 + c2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (c1 / n1 - c2 / n2) / se
        pv = 2 * stats.norm.sf(np.abs(z))
    return np.where(se > 0, pv, 1.0)


def pairs_from_raw(raw_dir: str) -> pd.DataFrame:
    """Rebuild the pairwise table from the archive package CSVs (build_upworthy.py, the columns this task needs)."""
    dfs = []
    for s in RAW_SPLITS:
        df = pd.read_csv(need(os.path.join(raw_dir, RAW_NAME.format(s))), index_col=0, low_memory=False)
        df["split"] = s
        dfs.append(df)
    pk = pd.concat(dfs, ignore_index=True)
    pk["created_at"] = pd.to_datetime(pk["created_at"], format="mixed")
    pk = pk.rename(columns={"clickability_test_id": "test_id"})
    pk = pk[pk.impressions > 0]
    pk = pk[~pk.duplicated(subset=["test_id", "headline", "excerpt", "lede", "eyecatcher_id", "impressions", "clicks"])].copy()
    for c in ["headline", "excerpt", "lede", "eyecatcher_id"]:
        pk[c + "_n"] = pk[c].map(_norm)
    pk["test_date"] = pk.groupby("test_id").created_at.transform("min")
    ia, ib, tids = [], [], []
    for tid, g in pk.groupby("test_id", sort=False):
        idx = g.index.to_numpy()
        for i, j in itertools.combinations(range(len(idx)), 2):
            ia.append(idx[i])
            ib.append(idx[j])
            tids.append(tid)
    ia, ib = np.array(ia, dtype=int), np.array(ib, dtype=int)
    swap = np.random.default_rng(42).random(len(ia)) >= 0.5  # same stream as one rng.random() per pair
    a_idx, b_idx = np.where(swap, ib, ia), np.where(swap, ia, ib)
    A, Bp = pk.loc[a_idx].reset_index(drop=True), pk.loc[b_idx].reset_index(drop=True)
    pw = pd.DataFrame({"test_id": A.test_id, "split": A.split, "test_date": A.test_date,
                       "headline_a": A.headline, "headline_b": Bp.headline,
                       "clicks_a": A.clicks.astype(int), "clicks_b": Bp.clicks.astype(int),
                       "n_a": A.impressions.astype(int), "n_b": Bp.impressions.astype(int)})
    same = {c: (A[c + "_n"].to_numpy() == Bp[c + "_n"].to_numpy()) for c in ["headline", "excerpt", "lede", "eyecatcher_id"]}
    pw["headline_only"] = ~same["headline"] & same["excerpt"] & same["lede"] & same["eyecatcher_id"]
    ca, cb = pw.clicks_a / pw.n_a, pw.clicks_b / pw.n_b
    pw["winner"] = np.where(ca > cb, "a", np.where(ca < cb, "b", "tie"))
    pw["p_value"] = two_prop_p(pw.clicks_a.to_numpy(float), pw.n_a.to_numpy(float), pw.clicks_b.to_numpy(float),
                               pw.n_b.to_numpy(float))
    pw["signif_fdr"] = False
    for s in RAW_SPLITS:
        m = (pw.split == s).to_numpy()
        pw.loc[m, "signif_fdr"] = bh_reject(pw.loc[m, "p_value"].to_numpy())
    return pw


def _acc(task, frame, col, B, seed):
    y, p = frame["truth"].to_numpy(float), frame[col].to_numpy(float)
    return cluster_bootstrap(lambda i: pair_accuracy(y[i], p[i]), frame["group"].to_numpy(), B=B, seed=seed)


def _auc(task, frame, col, B, seed):
    y, p = frame["truth"].to_numpy(float), frame[col].to_numpy(float)
    return cluster_bootstrap(lambda i: auc(y[i], p[i]), frame["group"].to_numpy(), B=B, seed=seed)


def _brier(task, frame, col, B, seed):
    y, p = frame["truth"].to_numpy(float), np.clip(frame[col].to_numpy(float), 0, 1)
    return cluster_bootstrap(lambda i: float(np.mean((p[i] - y[i]) ** 2)), frame["group"].to_numpy(), B=B, seed=seed)


@register()
class Upworthy(BenchTask):
    """Options: split='temporal' (default) | 'holdout'; test_signif='fdr' (default) | 'none' (all test pairs)."""

    name = "upworthy"
    accepted_options = {
        'split': '"temporal" | "holdout" (default "temporal")',
        'test_signif': '"fdr" | "none" (default "fdr")',
    }
    kind = "ab"
    data_subdir = "upworthy"
    layout = """
    <root>/upworthy/islenmis/pairwise.parquet   test_id, split, test_date, headline_a, headline_b, winner (a|b|tie),
                                                headline_only, signif_fdr (output of the research pair builder), or
    <root>/upworthy/ham/upworthy-archive-{exploratory,confirmatory,holdout}-packages-03.12.2020.csv
                                                (Upworthy Research Archive; pairs are rebuilt, slower)
    """
    source = ("talp-simulation-hackathon-2026 uretim/deney/kod/eval/build_upworthy.py, data_upworthy.py; "
              "uretim/deney/etki/kod/d1_upworthy.py, d1_sizinti.py")
    metrics = (Metric("pair_acc", True, _acc, "pair accuracy, p = 0.5 counts half; cluster bootstrap over tests"),
               Metric("auc", True, _auc, "ROC AUC of P(A wins)"),
               Metric("brier", False, _brier, "Brier score of P(A wins)"))

    def _pairs(self, root: str) -> pd.DataFrame:
        pq = os.path.join(root, "islenmis", "pairwise.parquet")
        if os.path.exists(pq):
            return pd.read_parquet(pq, columns=["test_id", "split", "test_date", "headline_a", "headline_b", "winner",
                                                "headline_only", "signif_fdr"])
        raw = os.path.join(root, "ham")
        if os.path.isdir(raw):
            return pairs_from_raw(raw)
        raise DataNotFound(f"neither {pq} nor {raw}/ found")

    def _load(self, root: str) -> TaskData:
        al = self._pairs(root)
        al = al[(al.winner != "tie") & al.headline_only].reset_index(drop=True)
        al["y"] = (al.winner == "a").astype(int)
        split = self.options.get("split", "temporal")
        if split == "temporal":
            td = al.groupby("test_id").test_date.min().sort_values()
            cut = td.iloc[int(len(td) * 0.8)]
            tr, te = al[al.test_date < cut], al[al.test_date >= cut]
            rule = f"temporal: tests by first date, newest 20% (test_date >= {pd.Timestamp(cut).date()}) are test"
        elif split == "holdout":
            tr, te = al[al.split != "holdout"], al[al.split == "holdout"]
            rule = "official: exploratory + confirmatory -> holdout"
        else:
            raise ValueError("split must be 'temporal' or 'holdout'")
        if self.options.get("test_signif", "fdr") == "fdr":
            te = te[te.signif_fdr]
        meta = {"split": rule + "; headline-only pairs, ties dropped; test pairs BH-FDR significant",
                "n_train_pairs": len(tr), "n_test_pairs": len(te), "n_test_tests": int(te.test_id.nunique()),
                "source": "Upworthy Research Archive (Matias, Munger, Aubin Le Quere, Ebersole 2021); CC BY 4.0"}
        return TaskData(train=tr.reset_index(drop=True), test=te.reset_index(drop=True), meta=meta)

    # ------------------------------------------------------------------ baselines
    @staticmethod
    def tfidf_bt(train: pd.DataFrame, test: pd.DataFrame, C: float = 0.3) -> np.ndarray:
        """Bradley-Terry logistic on TF-IDF difference vectors (word 1-2 gram + char_wb 2-5 gram), no intercept."""
        import scipy.sparse as sp
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        texts = pd.concat([train.headline_a, train.headline_b]).fillna("").unique().tolist()
        w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True).fit(texts)
        c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=200_000,
                            sublinear_tf=True).fit(texts)

        def diff(df):
            t = pd.concat([df.headline_a, df.headline_b]).fillna("").tolist()
            uniq = sorted(set(t))
            pos = {x: i for i, x in enumerate(uniq)}
            X = sp.hstack([w.transform(uniq), c.transform(uniq)]).tocsr()
            ia = [pos[x] for x in df.headline_a.fillna("")]
            ib = [pos[x] for x in df.headline_b.fillna("")]
            return (X[ia] - X[ib]).tocsr()

        m = LogisticRegression(C=C, fit_intercept=False, max_iter=3000, solver="lbfgs").fit(diff(train), train.y.values)
        return m.predict_proba(diff(test))[:, 1]

    def baselines(self, data: TaskData) -> dict:
        t = data.test
        la, lb = t.headline_a.fillna("").str.len().to_numpy(), t.headline_b.fillna("").str.len().to_numpy()
        return {"coin": np.full(len(t), 0.5), "longer_headline": np.where(la > lb, 1.0, np.where(la < lb, 0.0, 0.5)),
                "tfidf_bt": self.tfidf_bt(data.train, t)}

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        return pd.DataFrame({"unit": np.arange(len(t)), "group": t["test_id"].to_numpy(), "item": t["test_id"].to_numpy(),
                             "time": t["test_date"].to_numpy(), "segment": t["split"].to_numpy(),
                             "truth": t["y"].to_numpy(int)})
