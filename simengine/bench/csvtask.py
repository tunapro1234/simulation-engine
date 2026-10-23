"""from_csv: turn any labelled table (a competition's train.csv, a survey export) into a bench task.

    from simengine import bench
    task = bench.from_csv("train.csv", target="label", group_col="user_id", item_col="question", test_frac=0.2)
    data = task.load()                                 # TaskData; target encoded (binary 0/1, multiclass codes)
    frame = task.run({"mine": lambda d: my_model(d.train, d.test)})   # baselines + mine -> EvalFrame
    task.score(frame, "mine")                          # evidence metric, bootstrap CI over groups
    bench.run([task], methods=...)                     # leaderboard tooling works unchanged

Split: `split_col` (test / holdout / eval / true / 1 = test; train / training / dev / false / 0 = train; any other
value raises, so a fold number or an inverted `is_train` flag cannot pass silently) or a seeded random `test_frac` split, by group when `group_col` is given (no group on both sides). Rows with a
missing target are dropped (count in meta). Metrics are the `simengine.evidence` family for the task kind
(`metrik.aile`, English names), primary = evidence default or `metric=` (English, sklearn-style or internal name).
The CI is a cluster bootstrap over `group_col` (else over rows).

Baselines (fit on train only): `constant` (train mean / class shares), `majority` (binary / multiclass: the most
frequent class as a hard 0/1 or one-hot) or `median` (continuous), and `item_marginal` (per-item train mean / class
shares, unseen items fall back to the constant) when `item_col` is given.

Predictions: binary and continuous a 1-D array; multiclass a (n, K) probability matrix / list of vectors, or labels
scored as one-hot: class codes 0..K-1 (what `load()` puts in the target column) or the original labels. When the
classes are themselves numbers other than 0..K-1 (ratings 1..5) and the predictions fit both readings, it raises
instead of guessing; pass `np.eye(K)[codes]` or the labels as strings. Rows with a NaN prediction are left out of
the score; `coverage` in the result is the scored share of test rows.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..types import TaskData
from .base import BenchTask, DataNotFound, Metric, as_pred_col
from .stats import cluster_bootstrap

KINDS = {"binary": "ikili", "multiclass": "cok_sinif", "continuous": "surekli"}
TEST_VALUES = {"test", "holdout", "eval", "evaluation", "degerlendirme", "true", "1", "1.0"}
TRAIN_VALUES = {"train", "training", "fit", "dev", "egitim", "eğitim", "false", "0", "0.0"}


def _evidence_metrics(kind: str, item: bool) -> list:
    from ..evidence.metrik import aile
    return aile(KINDS[kind], olasilik=True, madde=item and kind != "multiclass")


def _english(ad: str) -> str:
    from ..evidence.api import METRIC_TO_ENGLISH
    return METRIC_TO_ENGLISH.get(ad, ad)


def _metric(m, kind: str) -> Metric:
    """Bench Metric around an evidence Metrik: point value on all rows, cluster bootstrap over frame['group']."""
    def fn(task, frame, col, B, seed):
        y, p, madde = task._arrays(frame, col)
        g = pd.factorize(frame["group"])[0] if "group" in frame else np.arange(len(frame))
        ok = np.isfinite(p).all(1) if p.ndim == 2 else np.isfinite(p)
        y, p, g = y[ok], p[ok], g[ok]
        madde = None if madde is None else madde[ok]
        cov = float(ok.mean()) if len(ok) else float("nan")
        if not len(y):
            return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "n_rows": 0, "coverage": cov}

        def stat(idx):
            try:
                return m.deger(y[idx], p[idx], None if madde is None else madde[idx])
            except (ValueError, ZeroDivisionError, IndexError):
                return float("nan")
        r = cluster_bootstrap(stat, g, B=B, seed=seed)
        if r["n"] < 2:  # one cluster: every replicate is the point estimate, a zero-width CI would claim certainty
            r["lo"] = r["hi"] = float("nan")
        return {**r, "n_rows": int(len(y)), "coverage": cov}
    return Metric(_english(m.ad), m.yon > 0, fn, m.aciklama)


class CsvTask(BenchTask):
    """A bench task built by `from_csv` (not registered: it carries its own data source)."""

    accepted_options = {
        "target": "column to predict (required)", "task": '"binary" | "multiclass" | "continuous"',
        "id_col": "row id column (EvalFrame unit)", "group_col": "cluster column (split + bootstrap unit)",
        "item_col": "item / question column (item_marginal baseline, within-item metrics)",
        "split_col": "column marking test rows", "test_frac": "test share when no split_col (default 0.2)",
        "metric": "primary metric (English or internal name)", "features": "feature columns (default: the rest)",
        "positive": "binary: the positive label", "seed": "split seed (default 0)", "read_kw": "pandas.read_csv kwargs",
    }

    def __init__(self, source: str | os.PathLike | pd.DataFrame, *, name: str | None = None, **options: Any):
        super().__init__(**options)
        o = self.options
        if "target" not in o:
            raise ValueError("from_csv: target= is required")
        self.kind = o.get("task", "binary")
        if self.kind not in KINDS:
            raise ValueError(f"from_csv: task must be one of {sorted(KINDS)}, not {self.kind!r}")
        frac = o.get("test_frac", 0.2)
        if o.get("split_col") is None and not 0 < frac < 1:
            raise ValueError(f"from_csv: test_frac must be in (0, 1), not {frac!r}")
        self._src = source
        self.source = "DataFrame" if isinstance(source, pd.DataFrame) else os.fspath(source)
        if name is None:
            name = (os.path.splitext(os.path.basename(os.fspath(source)))[0] if not isinstance(source, pd.DataFrame)
                    else "csv")
        self.name = name
        self.layout = "a single table: one row per labelled unit; see options"
        ms = _evidence_metrics(self.kind, o.get("item_col") is not None)
        primary = o.get("metric")
        if primary is not None:
            from ..evidence.cekirdek import birincil_ic
            from ..evidence.metrik import ISARETLI_METRIKLER
            ic = birincil_ic(primary)
            if ic in ISARETLI_METRIKLER:  # signed (0 is best): no direction to rank a leaderboard by
                raise ValueError(f"from_csv: {_english(ic)!r} is signed (0 is best) and cannot be the primary metric")
            if ic not in [m.ad for m in ms]:
                raise ValueError(f"from_csv: metric {primary!r} is not available for task {self.kind!r}: "
                                 f"{[_english(m.ad) for m in ms]}")
            ms = sorted(ms, key=lambda m: m.ad != ic)
        else:
            from ..evidence.metrik import BIRINCIL
            ms = sorted(ms, key=lambda m: m.ad != BIRINCIL[KINDS[self.kind]])
        self.metrics = tuple(_metric(m, self.kind) for m in ms)

    # ------------------------------------------------------------------ load
    def load(self, data_root: str | None = None) -> TaskData:
        o = self.options
        df = self._read(data_root)
        target = o["target"]
        cols = [c for c in (o.get("id_col"), o.get("group_col"), o.get("item_col"), o.get("split_col")) if c]
        missing = [c for c in [target, *cols, *(o.get("features") or [])] if c not in df.columns]
        if missing:
            raise ValueError(f"{self.name}: column(s) {missing} not in table; have {list(df.columns)}")
        lab = df[target].notna()
        n_drop = int((~lab).sum())
        df = df[lab].reset_index(drop=True)
        feats = list(o.get("features") or [c for c in df.columns if c != target and c not in cols])
        y, enc = self._encode(df[target])
        test = self._test_mask(df)
        if test.all() or not test.any():
            raise ValueError(f"{self.name}: split leaves {'no train' if test.all() else 'no test'} rows")
        keep = list(dict.fromkeys(cols + feats + [target]))
        d = df[keep].copy()
        d[target] = y
        meta = {"target": target, "features": feats, "id_col": o.get("id_col"), "group_col": o.get("group_col"),
                "item_col": o.get("item_col"), "n_unlabeled_dropped": n_drop,
                "split": (f"split_col {o['split_col']!r}" if o.get("split_col") else
                          f"random test_frac={o.get('test_frac', 0.2)} seed={o.get('seed', 0)}"
                          + (f" by {o['group_col']!r}" if o.get("group_col") else "")),
                "task": self.name, "kind": self.kind, "options": {k: v for k, v in o.items() if k != "read_kw"},
                **enc}
        return TaskData(train=d[~test].reset_index(drop=True), test=d[test].reset_index(drop=True), meta=meta)

    def _read(self, data_root: str | None) -> pd.DataFrame:
        if isinstance(self._src, pd.DataFrame):
            return self._src.copy()
        path = os.fspath(self._src)
        if data_root is not None and not os.path.isabs(path):
            path = os.path.join(os.fspath(data_root), path)
        if not os.path.exists(path):
            raise DataNotFound(f"missing file: {path}")
        return pd.read_csv(path, **(self.options.get("read_kw") or {}))

    def _encode(self, s: pd.Series) -> tuple[np.ndarray, dict]:
        if self.kind == "continuous":
            v = pd.to_numeric(s, errors="coerce")
            if v.isna().any():
                raise ValueError(f"{self.name}: target has non-numeric values for a continuous task")
            return v.to_numpy(float), {}
        vals = sorted(s.unique(), key=lambda x: (str(type(x)), x))
        if self.kind == "multiclass":
            if len(vals) < 2:
                raise ValueError(f"{self.name}: multiclass target has a single value {vals}")
            return pd.Categorical(s, categories=vals).codes.astype(int), {"classes": [_py(v) for v in vals]}
        pos = self.options.get("positive")
        if pos is not None:
            if not (s == pos).any():  # a typo or a type mismatch ("1" vs 1) would make every row negative
                raise ValueError(f"{self.name}: positive={pos!r} is not a target value; values {vals[:10]}")
        else:
            if len(vals) != 2:
                raise ValueError(f"{self.name}: binary target has {len(vals)} value(s) {vals[:5]}; "
                                 + ("use task='multiclass'" if len(vals) > 2 else "pass positive= if this is intended"))
            if set(vals) <= {0, 1}:
                pos = 1
            else:
                pos = vals[-1]  # sorted: yes > no, True > False
        return (s == pos).to_numpy(int), {"positive": _py(pos)}

    def _test_mask(self, df: pd.DataFrame) -> np.ndarray:
        o = self.options
        if o.get("split_col"):
            col = o["split_col"]
            v = df[col].astype(str).str.strip().str.lower()
            bad = sorted(set(v) - TEST_VALUES - TRAIN_VALUES)
            if bad:
                raise ValueError(f"{self.name}: split_col {col!r} has value(s) {bad[:10]} that are neither test "
                                 f"({sorted(TEST_VALUES)}) nor train ({sorted(TRAIN_VALUES)}); map it first")
            if "train" in str(col).lower() and set(v) <= {"true", "false", "0", "1", "0.0", "1.0"}:
                raise ValueError(f"{self.name}: split_col {col!r} looks like a train flag, but true / 1 marks TEST "
                                 f"rows; pass a test flag (e.g. df.assign(is_test=df[{col!r}] == 0))")
            return v.isin(TEST_VALUES).to_numpy()
        rng = np.random.default_rng(o.get("seed", 0))
        frac = o.get("test_frac", 0.2)
        if o.get("group_col"):
            groups = pd.unique(df[o["group_col"]])
            k = min(max(int(round(frac * len(groups))), 1), len(groups) - 1)
            pick = set(groups[rng.permutation(len(groups))[:k]].tolist())
            return df[o["group_col"]].isin(pick).to_numpy()
        k = min(max(int(round(frac * len(df))), 1), len(df) - 1)
        m = np.zeros(len(df), bool)
        m[rng.permutation(len(df))[:k]] = True
        return m

    # ------------------------------------------------------------------ baselines
    def baselines(self, data: TaskData) -> dict[str, Any]:
        tr, te, target = data.train, data.test, data.meta["target"]
        item = data.meta.get("item_col")
        y = tr[target].to_numpy(float if self.kind == "continuous" else int)
        n = len(te)
        if self.kind == "multiclass":
            K = len(data.meta["classes"])
            share = np.bincount(y, minlength=K) / len(y)
            out = {"constant": [share] * n, "majority": [np.eye(K)[int(share.argmax())]] * n}
            if item:
                per = {k: np.bincount(g.to_numpy(int), minlength=K) / len(g) for k, g in tr.groupby(item)[target]}
                out["item_marginal"] = [per.get(k, share) for k in te[item]]
            return out
        mean = float(y.mean())
        out = {"constant": np.full(n, mean)}
        if self.kind == "binary":
            out["majority"] = np.full(n, float(mean >= 0.5))
        else:
            out["median"] = np.full(n, float(np.median(y)))
        if item:
            out["item_marginal"] = te[item].map(tr.groupby(item)[target].mean()).fillna(mean).to_numpy(float)
        return out

    # ------------------------------------------------------------------ frame / scoring
    def frame(self, data: TaskData, preds: dict[str, Any]) -> pd.DataFrame:
        if self.kind != "multiclass":
            return super().frame(data, preds)
        base = self._frame(data)
        K = len(data.meta["classes"])
        for m, p in preds.items():
            base[as_pred_col(m)] = pd.Series(list(self._vectors(m, p, K, data.meta["classes"], len(base))),
                                             dtype=object)
        base["split"] = "test"
        return base

    @staticmethod
    def _vectors(m: str, p: Any, K: int, classes: Sequence, n: int) -> np.ndarray:
        if hasattr(p, "to_numpy"):  # Series / DataFrame: positional, whatever the index
            p = p.to_numpy()
        if len(p) != n:
            raise ValueError(f"{m}: {len(p)} predictions for {n} test rows")
        first = np.asarray(p[0]) if n else np.zeros(0)
        if first.ndim == 1 and first.size == K and K > 1:
            v = np.vstack([np.asarray(x, float) for x in p])
            return v
        if first.ndim >= 1 and first.size > 1:
            raise ValueError(f"{m}: probability vectors have {first.size} entries; the task has {K} classes "
                             f"{list(classes)} (a class missing from train still needs its column)")
        a = np.asarray(p).ravel()
        if np.issubdtype(a.dtype, np.number) and not np.issubdtype(a.dtype, np.bool_):
            fin = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.floating) else a
            as_code = bool(np.allclose(fin, np.round(fin)) and (fin >= 0).all() and (fin < K).all())
            numeric = all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in classes)
            as_label = numeric and set(fin.tolist()) <= set(classes)
            if len(fin) and numeric and list(classes) != list(range(K)) and as_code and as_label:
                raise ValueError(f"{m}: numeric predictions fit both class codes 0..{K - 1} and the labels "
                                 f"{list(classes)}; pass np.eye({K})[codes] or the labels as strings")
            if as_code:
                codes = np.where(np.isfinite(a), a, 0).astype(int) if a.dtype.kind == "f" else a.astype(int)
            elif as_label:
                idx = {c: i for i, c in enumerate(classes)}
                codes = np.array([idx.get(x, 0) for x in a.tolist()], int)
            else:
                raise ValueError(f"{m}: numeric multiclass labels must be class codes 0..{K - 1}"
                                 + (f" or the labels {list(classes)}" if numeric else ""))
            out = np.eye(K)[codes]
            out[~np.isfinite(a.astype(float))] = np.nan  # NaN prediction stays missing (left out of the score)
            return out
        else:
            idx = {str(c): i for i, c in enumerate(classes)}
            bad = sorted({str(x) for x in a} - set(idx))
            if bad:
                raise ValueError(f"{m}: unknown class label(s) {bad[:5]}; classes {list(classes)}")
            codes = np.array([idx[str(x)] for x in a], int)
        return np.eye(K)[codes]

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t, meta = data.test, data.meta
        unit = t[meta["id_col"]].to_numpy() if meta.get("id_col") else np.arange(len(t))
        out = {"unit": unit, "group": t[meta["group_col"]].to_numpy() if meta.get("group_col") else unit}
        if meta.get("item_col"):
            out["item"] = t[meta["item_col"]].to_numpy()
        out["truth"] = t[meta["target"]].to_numpy()
        return pd.DataFrame(out)

    def _arrays(self, frame: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        madde = pd.factorize(frame["item"])[0] if "item" in frame else None
        if self.kind == "multiclass":
            p = np.vstack(frame[col].to_list()) if len(frame) else np.zeros((0, 0))
            return frame["truth"].to_numpy(int), p, madde
        return frame["truth"].to_numpy(float), frame[col].to_numpy(float), madde


def _py(v: Any) -> Any:
    return v.item() if isinstance(v, np.generic) else v


def from_csv(path_or_df: str | os.PathLike | pd.DataFrame, *, target: str, id_col: str | None = None,
             group_col: str | None = None, item_col: str | None = None, split_col: str | None = None,
             test_frac: float = 0.2, task: str = "binary", metric: str | None = None,
             features: Sequence[str] | None = None, positive: Any = None, seed: int = 0, name: str | None = None,
             read_kw: dict | None = None) -> CsvTask:
    """A bench task over one table (CSV path, relative paths resolve under load's data_root, or a DataFrame).
    See the module docstring for split, metrics and baselines. Nothing is read until load()."""
    opts = {"target": target, "task": task, "id_col": id_col, "group_col": group_col, "item_col": item_col,
            "split_col": split_col, "test_frac": test_frac, "metric": metric,
            "features": list(features) if features is not None else None, "positive": positive, "seed": seed,
            "read_kw": read_kw}
    return CsvTask(path_or_df, name=name, **opts)
