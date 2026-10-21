"""bank_marketing: does a customer subscribe to a term deposit after a bank's telephone campaign call (UCI Bank Marketing,
bank-additional-full, Portugal 2008-2010, Moro et al. 2014).

Ported from the intent branch (`gbdt/lig/bank/bank.py`, preregistration `gbdt/onkayit/bank-lig.md`). The file is in
call order (May 2008 -> Nov 2010), so the split is by row position: train rows 0-24,711 (60%), dev 24,712-32,949
(20%), test 32,950-41,187 (20%, mostly 2010). `duration` is dropped (known only after the call). The base rate shifts
strongly over time (train ~4.8% yes, test ~30.8%), which is what makes this split hard.
Metrics: AUC (primary), PR-AUC, Brier; row bootstrap with default_rng(42), B = 1000 (as in the source).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, Metric, need, register
from ..stats import auc, ci

N_TRAIN, N_DEV = 24712, 8238  # = int(0.6 n), round(0.2 n) for the full file (n = 41,188)
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
               "poutcome"]
NUMERIC = ["age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m",
           "nr.employed"]


def _ap(y, p):
    """Average precision (step-wise PR-AUC, same as sklearn average_precision_score)."""
    o = np.argsort(-p, kind="mergesort")
    y, p = y[o], p[o]
    distinct = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    if tp[-1] == 0:
        return float("nan")
    prec, rec = tp / (tp + fp), tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def _row_boot(fn, const_value=None):
    def metric(task, frame, col, B, seed):
        y, p = frame["truth"].to_numpy(float), frame[col].to_numpy(float)
        if const_value is not None and np.std(p) == 0:
            v = const_value(y)
            return {"value": v, "lo": v, "hi": v, "n": len(y)}
        idx = np.random.default_rng(seed).integers(0, len(y), (B, len(y)))
        point = fn(y, p)
        lo, hi = ci(point, np.array([fn(y[i], p[i]) for i in idx]))
        return {"value": float(point), "lo": lo, "hi": hi, "n": len(y)}
    return metric


@register()
class BankMarketing(BenchTask):
    """Options: split='test' (default; train+dev -> test) | 'dev' (train -> dev); lgbm=True."""

    name = "bank_marketing"
    accepted_options = {
        'split': '"test" | "dev" (default "test")',
        'lgbm': 'true | false (default true)',
    }
    kind = "binary"
    data_subdir = "bank-marketing"
    layout = """
    <root>/bank-marketing/bank-additional/bank-additional-full.csv   UCI Bank Marketing (';' separated, 41,188 rows,
                                                                     original row order = call order)
    """
    source = "talp-simulation-hackathon-2026 uretim/intent/gbdt/lig/bank/bank.py; uretim/intent/gbdt/onkayit/bank-lig.md"
    metrics = (Metric("auc", True, _row_boot(auc, lambda y: 0.5), "ROC AUC"),
               Metric("pr_auc", True, _row_boot(_ap, lambda y: float(y.mean())), "average precision"),
               Metric("brier", False, _row_boot(lambda y, p: float(np.mean((p - y) ** 2))), "Brier score"))

    def _load(self, root: str) -> TaskData:
        path = os.path.join(root, "bank-additional", "bank-additional-full.csv")
        if not os.path.exists(path):
            path = os.path.join(root, "bank-additional-full.csv")
        d = pd.read_csv(need(path), sep=";").drop(columns=["duration"])
        d["y"] = (d["y"] == "yes").astype(int)
        n_tr, n_dv = int(0.6 * len(d)), int(round(0.2 * len(d)))
        d["part"] = np.where(d.index < n_tr, "train", np.where(d.index < n_tr + n_dv, "dev", "test"))
        split = self.options.get("split", "test")
        if split == "test":
            tr, te = d[d.part != "test"], d[d.part == "test"]
        elif split == "dev":
            tr, te = d[d.part == "train"], d[d.part == "dev"]
        else:
            raise ValueError("split must be 'test' or 'dev'")
        meta = {"split": f"{split}: by row position (call order); train rows < {n_tr}, dev < {n_tr + n_dv}, "
                         "test the rest", "categorical": CATEGORICAL, "numeric": NUMERIC,
                "source": "UCI Bank Marketing (Moro, Cortez, Rita 2014); CC BY 4.0"}
        return TaskData(train=tr.reset_index(drop=True), test=te.reset_index(drop=True), meta=meta)

    def baselines(self, data: TaskData) -> dict:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        tr, te = data.train, data.test
        base = tr[tr.part == "train"] if (tr.part == "train").any() else tr
        out = {"constant": np.full(len(te), float(base["y"].mean())),
               "persistence": te["poutcome"].map(base.groupby("poutcome")["y"].mean()).fillna(base["y"].mean())
               .to_numpy(float)}
        pre = ColumnTransformer([("k", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                                 ("s", StandardScaler(), NUMERIC)])
        lo = make_pipeline(pre, LogisticRegression(C=1.0, max_iter=3000)).fit(tr[CATEGORICAL + NUMERIC], tr["y"])
        out["logistic"] = lo.predict_proba(te[CATEGORICAL + NUMERIC])[:, 1]
        if self.options.get("lgbm", True):
            p = self._lgbm(tr, te)
            if p is not None:
                out["lgbm"] = p
        return out

    @staticmethod
    def _lgbm(tr, te):
        """Grid num_leaves {15, 31, 63} x lr {0.03, 0.1}, early stopping on dev AUC, refit on train + dev with the chosen
        tree count (source `lgbm_izgara`). Needs a 'dev' part in train (split='test'); else returns None."""
        try:
            import lightgbm as lgb
        except ImportError:
            return None
        from sklearn.metrics import roc_auc_score
        if not (tr.part == "dev").any():
            return None

        def X(df):
            x = df[CATEGORICAL + NUMERIC].copy()
            for c in CATEGORICAL:
                x[c] = pd.Categorical(x[c], categories=sorted(pd.concat([tr[c], te[c]]).unique()))
            return x
        a, dv = tr[tr.part == "train"], tr[tr.part == "dev"]
        import inspect
        # LightGBM 4.7+ deprecates eval_set in favour of eval_X / eval_y (same early stopping)
        ev = ({"eval_X": (X(dv),), "eval_y": (dv["y"],)} if "eval_X" in inspect.signature(lgb.LGBMClassifier.fit).parameters
              else {"eval_set": [(X(dv), dv["y"])]})
        best = None
        for nl in (15, 31, 63):
            for lr in (0.03, 0.1):
                m = lgb.LGBMClassifier(n_estimators=2000, num_leaves=nl, learning_rate=lr, random_state=0, verbose=-1,
                                       n_jobs=4)
                m.fit(X(a), a["y"], **ev, eval_metric="auc",
                      callbacks=[lgb.early_stopping(100, verbose=False)])
                s = roc_auc_score(dv["y"], m.predict_proba(X(dv))[:, 1])
                if best is None or s > best[0]:
                    best = (s, nl, lr, max(m.best_iteration_ or 1, 1))
        _, nl, lr, n = best
        m = lgb.LGBMClassifier(n_estimators=n, num_leaves=nl, learning_rate=lr, random_state=0, verbose=-1, n_jobs=4)
        return m.fit(X(tr), tr["y"]).predict_proba(X(te))[:, 1]

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        return pd.DataFrame({"unit": np.arange(len(t)), "time": np.arange(len(t)), "segment": t["month"].to_numpy(),
                             "seg_contact": t["contact"].to_numpy(), "truth": t["y"].to_numpy(int)})
