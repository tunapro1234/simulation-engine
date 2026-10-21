"""twin2k500: Twin-2K-500 person-level prediction of a new person's answers (heuristics-and-biases block).

Task (Toubia et al. 2025 "Twin-2K-500", repeated questions): predict a person's wave 1-3 answer on the scored columns
that were asked again in wave 4, from that person's other wave 1-3 answers. Persons are split by a fixed id hash
(dongu `kisi_parca`: 25% test). Scoring follows the official MAD evaluation as re-implemented in the persona branch
(`eval/twin_score.py`): anchoring items are converted to deciles of the wave 1-3 answers, only cells with a wave 4
retest answer count, per-column between-person Pearson r (Fisher z, mean over columns; constant prediction -> 0) and
task-level accuracy 1 - |pred - truth| / range (mean over a person's tasks, then over persons).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, Metric, need, register
from ..stats import bootstrap_mean, fisher_z

FMT = os.path.join("LLM_simulation_results", "GPT4.1-mini-simulation-llm-vs-human")
QDIR = "question_catalog_and_human_response_csv"
DECILE_GROUPS = [["Q164", "Q166"], ["Q168", "Q170"]]
ANCHORS = {c for g in DECILE_GROUPS for c in g}


def person_part(pid) -> str:
    """dongu v1.1 person split: 'test' (25%) or 'dev'."""
    return "test" if int(hashlib.sha256(f"dongu-v1.1-twin:{pid}".encode()).hexdigest()[:8], 16) % 4 == 0 else "dev"


def column_ranges() -> dict:
    m = {}
    m.update({f"FALSE CONS. SELF _{i}": (1, 5) for i in range(1, 11)})
    m.update({f"FALSE CONS. OTHERS _{i}": (0, 100) for i in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]})
    m["Q156_1"] = (0, 100)
    m["FORM A _1"] = (0, 100)
    for c in ["157", "158"] + [f"160_{i}" for i in (1, 2, 3)] + [f"159_{i}" for i in (1, 2, 3)]:
        m[f"Q{c}"] = (1, 6)
    for c in ("161", "162"):
        m[f"Q{c}"] = (1, 7)
    for c in ("164", "166", "168", "170"):
        m[f"Q{c}"] = (1, 10)
    for c in ("171", "172", "173", "174", "175", "176"):
        m[f"Q{c}"] = (1, 5)
    for c in ("177", "178", "179"):
        m[f"Q{c}"] = (1, 6)
    m["Q181"] = (0, 20)
    m["Q182"] = (0, 20)
    for c in ("183", "184", "192", "193"):
        m[f"Q{c}"] = (1, 2)
    for c in ("189", "190", "191"):
        m[f"Q{c}"] = (1, 10)
    for c in ("194", "195"):
        m[f"Q{c}"] = (1, 6)
    m.update({f"Q198_{i}": (1, 2) for i in range(1, 11)})
    m.update({f"Q203_{i}": (1, 2) for i in range(1, 7)})
    m.update({f"NONSEPARABILTY BENE _{i}": (1, 7) for i in range(1, 5)})
    m.update({f"NONSEPARABILITY RIS _{i}": (1, 7) for i in range(1, 5)})
    m["OMISSION BIAS "] = (1, 4)
    m["DENOMINATOR NEGLECT "] = (1, 2)
    m.update({f"{i}_Q295": (1, 2) for i in range(1, 41)})
    return m


def col_to_task() -> dict:
    raw = {
        **{f"FALSE CONS. SELF _{i}": "false consensus" for i in range(1, 11)},
        **{f"FALSE CONS. OTHERS _{i}": "false consensus" for i in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]},
        "Q156_1": "base rate", "FORM A _1": "base rate", "Q157": "framing problem", "Q158": "framing problem",
        **{f"Q160_{i}": "conjunction problem (Linda)" for i in [1, 2, 3]},
        **{f"Q159_{i}": "conjunction problem (Linda)" for i in [1, 2, 3]},
        "Q161": "outcome bias", "Q162": "outcome bias",
        **{f"Q{i}": "anchoring and adjustment" for i in (164, 166, 168, 170)},
        **{f"Q17{i}": "less is more" for i in range(1, 10)},
        "Q181": "sunk cost fallacy", "Q182": "sunk cost fallacy",
        "Q183": "absolute vs. relative savings", "Q184": "absolute vs. relative savings",
        "Q189": "WTA/WTP-Thaler", "Q190": "WTA/WTP-Thaler", "Q191": "WTA/WTP-Thaler",
        "Q192": "Allais", "Q193": "Allais", "Q194": "myside", "Q195": "myside",
        **{f"Q198_{i}": "prob matching vs. max" for i in range(1, 11)},
        **{f"Q203_{i}": "prob matching vs. max" for i in range(1, 7)},
        **{f"NONSEPARABILTY BENE _{i}": "non-separability of risks and benefits" for i in range(1, 5)},
        **{f"NONSEPARABILITY RIS _{i}": "non-separability of risks and benefits" for i in range(1, 5)},
        "OMISSION BIAS ": "omission", "DENOMINATOR NEGLECT ": "denominator neglect",
        **{f"{i}_Q295": "pricing" for i in range(1, 41)},
    }
    return raw


def read_formatted(path: str) -> pd.DataFrame:
    df = pd.read_csv(need(path), skiprows=[1], low_memory=False)
    df.columns = df.columns.str.upper()
    df = df.drop(columns=[c for c in df.columns if c == "WAVE"], errors="ignore")
    return df.apply(pd.to_numeric, errors="coerce").set_index("TWIN_ID")


def _lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x


def load_features(root: str) -> pd.DataFrame:
    """Numeric wave 1-3 answers except the target columns (persona `twin_baselines.load_features`)."""
    w13 = pd.read_csv(need(os.path.join(root, QDIR, "wave1_3_response.csv")), low_memory=False).set_index("pid")
    mp = json.load(open(need(os.path.join(root, "LLM_simulation_results", "wave4_formatted_to_catalog_mapping.json"))))
    tgt = {x["catalog_csv_column"] for x in mp}
    X = w13[[c for c in w13.columns if c not in tgt]].apply(pd.to_numeric, errors="coerce")
    cat_path = os.path.join(root, QDIR, "question_catalog.json")
    multi = set()
    if os.path.exists(cat_path):
        for e in json.load(open(cat_path)):
            st = _lit(e.get("Settings", {})) or {}
            if isinstance(st, dict) and st.get("Selector") in ("MAVR", "MAHR"):
                multi.update(_lit(e.get("csv_columns")) or [])
    for c in X.columns:
        if c in multi:
            X[c] = X[c].fillna(0)
    return X.loc[:, X.notna().mean() > 0.5]


def decile_thresholds(truth: pd.DataFrame) -> dict:
    th = {}
    for grp in DECILE_GROUPS:
        vals = pd.concat([truth[c] for c in grp if c in truth.columns]).dropna()
        if len(vals):
            th[tuple(grp)] = np.percentile(vals, np.arange(10, 100, 10))
    return th


def to_scored(values: np.ndarray, item: np.ndarray, th: dict) -> np.ndarray:
    """Anchoring answers (free numbers) -> deciles of the wave 1-3 answers (official code); other items unchanged."""
    v = np.asarray(values, float).copy()
    item = np.asarray(item)
    for grp, t in th.items():
        m = np.isin(item, grp) & np.isfinite(v)
        v[m] = np.minimum(np.searchsorted(t, v[m], side="left") + 1.0, 10)
    return v


def _col_r(frame: pd.DataFrame, col: str) -> pd.Series:
    out = {}
    for k, g in frame.groupby("item", sort=False):
        t, p = g["truth"].to_numpy(float), g[col].to_numpy(float)
        m = np.isfinite(t) & np.isfinite(p)
        out[k] = float(np.corrcoef(p[m], t[m])[0, 1]) if m.sum() > 10 and p[m].std() > 0 and t[m].std() > 0 else 0.0
    return pd.Series(out)


def _mean_z(task, frame, col, B, seed):
    return bootstrap_mean(fisher_z(_col_r(frame, col).to_numpy()), B=B, seed=seed)


def _corr(task, frame, col, B, seed):
    z = _mean_z(task, frame, col, B, seed)
    return {k: (float(np.tanh(v)) if k in ("value", "lo", "hi") else v) for k, v in z.items()}


def _acc(task, frame, col, B, seed):
    a = 1 - np.abs(frame[col].to_numpy(float) - frame["truth"].to_numpy(float)) / frame["range"].to_numpy(float)
    d = pd.DataFrame({"g": frame["group"].to_numpy(), "t": frame["segment"].to_numpy(), "a": a})
    per_person = d.groupby(["g", "t"])["a"].mean().groupby(level=0).mean()
    return bootstrap_mean(per_person.to_numpy(), B=B, seed=seed)


@register()
class Twin2K500(BenchTask):
    """Options: split='test' (train on the 75% dev persons, test on the 25% hash test persons; default) |
    'dev' (inside the dev persons only: a second fixed hash holds out 20% of them; test persons never used)."""

    name = "twin2k500"
    kind = "continuous"
    data_subdir = "twin2k500"
    layout = """
    <root>/twin2k500/question_catalog_and_human_response_csv/wave1_3_response.csv   pid x numeric answers (features)
    <root>/twin2k500/question_catalog_and_human_response_csv/question_catalog.json  (optional; multi-select columns)
    <root>/twin2k500/LLM_simulation_results/wave4_formatted_to_catalog_mapping.json  target columns to exclude
    <root>/twin2k500/LLM_simulation_results/GPT4.1-mini-simulation-llm-vs-human/responses_wave1_3_formatted.csv  truth
    <root>/twin2k500/LLM_simulation_results/GPT4.1-mini-simulation-llm-vs-human/responses_wave4_formatted.csv  retest
    """
    source = ("talp-simulation-hackathon-2026 uretim/persona/kod/eval/twin_score.py, baseline/twin_baselines.py; "
              "uretim/dongu/kod/hedef_twin.py (person split)")
    metrics = (Metric("mean_z", True, _mean_z, "mean over columns of Fisher z of between-person Pearson r"),
               Metric("corr", True, _corr, "tanh(mean z) (official 'corr')"),
               Metric("acc", True, _acc, "official MAD task-level accuracy (person bootstrap)"))

    def _load(self, root: str) -> TaskData:
        truth = read_formatted(os.path.join(root, FMT, "responses_wave1_3_formatted.csv"))
        retest = read_formatted(os.path.join(root, FMT, "responses_wave4_formatted.csv"))
        rng = column_ranges()
        cols = [c for c in rng if c in truth.columns and c in retest.columns]
        truth, retest = truth[cols], retest.reindex(truth.index)[cols]
        X = load_features(root)
        ids = truth.index.intersection(X.index)
        split = self.options.get("split", "test")
        part = pd.Series([person_part(k) for k in ids], index=ids)
        if split == "test":
            tr_ids, te_ids = ids[part.to_numpy() == "dev"], ids[part.to_numpy() == "test"]
        elif split == "dev":
            dev = ids[part.to_numpy() == "dev"]
            inner = np.array([int(hashlib.sha256(f"simengine-twin-dev:{k}".encode()).hexdigest()[:8], 16) % 5 == 0
                              for k in dev])
            tr_ids, te_ids = dev[~inner], dev[inner]
        else:
            raise ValueError("split must be 'test' or 'dev'")
        tasks = col_to_task()

        def long(pids, with_retest):
            T = truth.loc[pids]
            d = T.rename_axis("person").reset_index().melt(id_vars="person", var_name="item", value_name="truth")
            if with_retest:
                R = retest.loc[pids].rename_axis("person").reset_index().melt(id_vars="person", var_name="item",
                                                                              value_name="retest")
                d = d.merge(R, on=["person", "item"])
                d = d[d["truth"].notna() & d["retest"].notna()]  # official: only cells with a wave 4 answer
            else:
                d = d[d["truth"].notna()]
            d["task"] = d["item"].map(tasks)
            d["range"] = d["item"].map(lambda c: rng[c][1] - rng[c][0]).astype(float)
            return d.reset_index(drop=True)

        meta = {"split": f"{split}: persons by fixed id hash; train {len(tr_ids)} persons, test {len(te_ids)}",
                "features": X, "targets": cols, "ranges": {c: rng[c] for c in cols},
                "deciles": decile_thresholds(truth), "train_ids": list(tr_ids), "test_ids": list(te_ids),
                "source": "Twin-2K-500 (Toubia et al. 2025); CC BY 4.0 per dataset card"}
        return TaskData(train=long(tr_ids, False), test=long(te_ids, True), meta=meta)

    # ------------------------------------------------------------------ baselines
    def _wide_train(self, data):
        return data.train.pivot(index="person", columns="item", values="truth").reindex(
            index=data.meta["train_ids"], columns=data.meta["targets"])

    def _pick(self, data, P: pd.DataFrame) -> np.ndarray:
        t = data.test
        M = P.reindex(index=data.meta["test_ids"], columns=data.meta["targets"])
        ri = {k: i for i, k in enumerate(M.index)}
        ci = {k: i for i, k in enumerate(M.columns)}
        return M.to_numpy(float)[t["person"].map(ri).to_numpy(int), t["item"].map(ci).to_numpy(int)]

    def predict_median(self, data) -> np.ndarray:
        Y = self._wide_train(data)
        med = Y.median()
        return data.test["item"].map(med).to_numpy(float)

    def predict_ridge(self, data, alpha: float = 300.0) -> np.ndarray:
        """SimpleImputer(median) + StandardScaler + Ridge(alpha=300) per target column (persona baseline)."""
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        X = data.meta["features"]
        Y = self._wide_train(data)
        tr, te = data.meta["train_ids"], data.meta["test_ids"]
        imp = SimpleImputer(strategy="median").fit(X.loc[tr])
        sc = StandardScaler().fit(imp.transform(X.loc[tr]))
        A, Bm = sc.transform(imp.transform(X.loc[tr])), sc.transform(imp.transform(X.loc[te]))
        A, Bm = np.nan_to_num(A), np.nan_to_num(Bm)
        P = pd.DataFrame(np.nan, index=te, columns=Y.columns)
        rng = data.meta["ranges"]
        for c in Y.columns:
            ok = Y[c].notna().to_numpy()
            if ok.sum() < 20:
                continue
            p = Ridge(alpha=alpha).fit(A[ok], Y[c].to_numpy()[ok]).predict(Bm)
            if c not in ANCHORS:
                p = np.clip(p, *rng[c])
            P[c] = p
        return self._pick(data, P)

    def baselines(self, data: TaskData) -> dict:
        return {"median": self.predict_median(data), "ridge": self.predict_ridge(data)}

    def frame(self, data: TaskData, preds: dict) -> pd.DataFrame:
        """Predictions in raw units (anchoring items as free numbers); converted to scored units here."""
        th, item = data.meta["deciles"], data.test["item"].to_numpy()
        return super().frame(data, {m: to_scored(np.asarray(p, float), item, th) for m, p in preds.items()})

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t, th = data.test, data.meta["deciles"]
        return pd.DataFrame({"unit": t["person"].astype(str) + "|" + t["item"], "group": t["person"].astype(str),
                             "item": t["item"], "segment": t["task"], "range": t["range"],
                             "truth": to_scored(t["truth"].to_numpy(float), t["item"].to_numpy(), th),
                             "retest": to_scored(t["retest"].to_numpy(float), t["item"].to_numpy(), th)})
