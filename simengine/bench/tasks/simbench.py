"""simbench: SimBench (Hu et al. 2025, arXiv 2510.17516) group response distributions, 20 datasets.

SimBench ships no train/dev split: both parts (SimBenchPop 7,167 rows, SimBenchGrouped 6,343 rows) are test data.
This adapter uses the dev/test split preregistered in the persona/kosul branch (`kod/simbench/sb.py`): split at the
question-text level (all groups of a question on the same side), stratified by dataset, 20% dev, seed 0
(np.random.default_rng(0).permutation over sorted question keys per dataset). TODO: if SimBench later publishes an
official split, add it as split='official'.

Score (paper, sb.py): S_i = 100 * (1 - TVD(P_i, Q_i) / mean_{j in dataset D, same part} TVD(P_j, U_j)); part score =
mean S_i; overall = mean of the two part scores. Uniform scores 0 by construction. The denominator uses every row of the
dataset in that part (dev + test, label-free scale constant).
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, Metric, need, register
from ..stats import cluster_bootstrap, tvd

PARTS = {"pop": "SimBenchPop.csv", "grp": "SimBenchGrouped.csv"}
DEV_SHARE = 0.2


def load_part(root: str, part: str) -> pd.DataFrame:
    d = pd.read_csv(need(os.path.join(root, PARTS[part])))
    ha = d.human_answer.map(ast.literal_eval)
    gm = d.group_prompt_variable_map.map(ast.literal_eval)
    out = pd.DataFrame({
        "id": [f"{part}{i:05d}" for i in range(len(d))], "part": part, "dataset": d.dataset_name,
        "group_prompt": [t.format(**m) if m else t for t, m in zip(d.group_prompt_template, gm)],
        "question": d.input_template, "options": ha.map(list),
        "P": ha.map(lambda h: np.array(list(h.values()), float) / sum(h.values())), "n_humans": d.group_size})
    out["qkey"] = out.dataset + "\x00" + out.question
    dev = set()
    for _, g in out.groupby("dataset", sort=True):
        q = np.array(sorted(g.qkey.unique()))
        p = np.random.default_rng(0).permutation(len(q))
        dev |= set(q[p[: int(round(DEV_SHARE * len(q)))]])
    out["dev"] = out.qkey.isin(dev)
    u = out.P.map(lambda p: tvd(p, np.full(len(p), 1 / len(p))))
    out["tvd_uniform"] = u
    out["denominator"] = out.groupby("dataset").tvd_uniform.transform("mean")
    return out


def dev_marginal(df: pd.DataFrame) -> list:
    """'sabit' baseline: per dataset x option label mean human share over dev rows, leaving out the row's own question;
    missing label -> 1/k; renormalised (persona/kosul degerlendir.py)."""
    dv = df[df.dev]
    top, cnt, qtop, qcnt = {}, {}, {}, {}
    for ds, qk, ks, p in zip(dv.dataset, dv.qkey, dv.options, dv.P):
        for k, v in zip(ks, p):
            top[ds, k] = top.get((ds, k), 0) + v
            cnt[ds, k] = cnt.get((ds, k), 0) + 1
            qtop[qk, k] = qtop.get((qk, k), 0) + v
            qcnt[qk, k] = qcnt.get((qk, k), 0) + 1
    out = []
    for r in df.itertuples():
        v = []
        for k in r.options:
            n = cnt.get((r.dataset, k), 0) - qcnt.get((r.qkey, k), 0)
            v.append((top.get((r.dataset, k), 0) - qtop.get((r.qkey, k), 0)) / n if n > 0 else 1 / len(r.options))
        v = np.clip(np.array(v), 0, None)
        out.append(v / v.sum() if v.sum() > 0 else np.full(len(v), 1 / len(v)))
    return out


def _s_values(frame, col):
    t = np.array([tvd(p, q) for p, q in zip(frame["truth"], frame[col])])
    return 100 * (1 - t / frame["denominator"].to_numpy(float))


def _score(task, frame, col, B, seed):
    s = _s_values(frame, col)
    part = frame["segment"].to_numpy()

    def f(i):
        return float(np.mean([s[i][part[i] == p].mean() for p in np.unique(part[i])]))
    return cluster_bootstrap(f, frame["group"].to_numpy(), B=B, seed=seed)


def _tvd(task, frame, col, B, seed):
    t = np.array([tvd(p, q) for p, q in zip(frame["truth"], frame[col])])
    return cluster_bootstrap(lambda i: float(t[i].mean()), frame["group"].to_numpy(), B=B, seed=seed)


@register()
class SimBench(BenchTask):
    """Options: split='test' (default; non-dev rows) | 'dev'; part='both' (default) | 'pop' | 'grp'."""

    name = "simbench"
    accepted_options = {
        'part': '"both" | "pop" | "grp" (default "both")',
        'split': '"test" | "dev" (default "test")',
    }
    kind = "distribution"
    data_subdir = "simbench"
    layout = """
    <root>/simbench/SimBenchPop.csv, SimBenchGrouped.csv   HF pitehu/SimBench CSVs: dataset_name, group_prompt_template,
        group_prompt_variable_map, input_template, human_answer (dict of option -> percent), group_size
    """
    source = "talp-simulation-hackathon-2026 uretim/persona/kosul/kod/simbench/sb.py, degerlendir.py, PROTOKOL.md"
    metrics = (Metric("S", True, _score, "SimBench S (0 = uniform, 100 = perfect), mean of part scores"),
               Metric("tvd", False, _tvd, "mean total variation distance"))

    def _load(self, root: str) -> TaskData:
        part = self.options.get("part", "both")
        parts = ["pop", "grp"] if part == "both" else [part]
        df = pd.concat([load_part(root, p) for p in parts], ignore_index=True)
        split = self.options.get("split", "test")
        if split not in ("test", "dev"):
            raise ValueError("split must be 'test' or 'dev'")
        test = df[~df.dev] if split == "test" else df[df.dev]
        meta = {"split": f"{split}: question-level, dataset-stratified 20% dev (seed 0; SimBench has no official split)",
                "parts": parts, "all_rows": df,
                "source": "SimBench (Hu et al. 2025), HF pitehu/SimBench; CC BY-NC-SA 4.0"}
        return TaskData(train=df[df.dev].reset_index(drop=True), test=test.reset_index(drop=True), meta=meta)

    def baselines(self, data: TaskData) -> dict:
        t = data.test
        allr = data.meta["all_rows"]
        marg = dict(zip(allr.id, dev_marginal(allr)))
        return {"uniform": [np.full(len(p), 1 / len(p)) for p in t.P], "dev_marginal": [marg[i] for i in t.id]}

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        return pd.DataFrame({"unit": t["id"], "item": t["qkey"], "group": t["part"] + "\x00" + t["qkey"],
                             "segment": t["part"], "seg_dataset": t["dataset"], "denominator": t["denominator"],
                             "truth": pd.Series([np.asarray(p, float) for p in t["P"]], dtype=object)})
