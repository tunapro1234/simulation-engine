"""yjmob: predict where each person is, slot by slot, for 15 future days (YJMob100K dataset 1, Yabe et al. 2024
Sci Data 11:397; the task of HuMob Challenge 2023, task 1).

100,000 anonymised people, 75 days, 30-minute slots (t = 0..47), cells of a 200 x 200 grid (x, y = 1..200, 500 m).
Split (HuMob): days 1-60 (d = 0..59) are history, days 61-75 (d = 60..74) are predicted at every (uid, d, t) row the
person was observed. split='dev' uses d = 0..44 -> 45..59 and drops rows with d >= 60 as they are read.
Users with any masked (999) coordinate are dropped and duplicate (uid, d, t) rows keep the first (TALP port rules);
then `users` people (default 2,000) are drawn with default_rng(seed) from the pool ('all', or 'humob_test' = uid >=
80,000, the people HuMob 2023 masked). The global slot table of the `global_slot_mode` baseline is counted over the
history days of all users without masked rows (no future rows; duplicate rows counted).

Predictions are cell codes (x - 1) * 200 + (y - 1), one per test row (see simengine.abm.mobility.to_cell).
Metrics (HuMob definitions, simengine.abm.mobility.user_scores): GEO-BLEU (primary, higher is better) and DTW (km,
lower is better), each per user as the mean over its days, then averaged over users with a user bootstrap
(B = 1000, default_rng(42)); plus exact-cell accuracy. variant='humob23' (default, N = 3 and the 2023 DTW start
rule) or 'humob25' (GIS Cup 2025: N = 5, standard DTW).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ...abm import mobility as M
from ...types import TaskData
from ..base import BenchTask, DataNotFound, Metric, register
from ..stats import bootstrap_mean

FILES = ("yjmob100k-dataset1.csv.gz", "yjmob100k-dataset1.csv")
SPLITS = {"test": (60, 75), "dev": (45, 60)}        # (first test day, end), 0-based d
HUMOB_TEST_UID = 80000
DTYPES = {"uid": np.int32, "d": np.int16, "t": np.int16, "x": np.int16, "y": np.int16}


def _path(root: str) -> str:
    for f in FILES:
        p = os.path.join(root, f)
        if os.path.exists(p):
            return p
    raise DataNotFound(f"none of {FILES} under {root}")


def read_dataset(path: str, max_day: int | None = None, chunksize: int = 5_000_000) -> pd.DataFrame:
    """Read uid, d, t, x, y with compact dtypes in chunks; rows with d >= max_day are dropped as they are read."""
    parts = []
    for ch in pd.read_csv(path, usecols=list(DTYPES), dtype=DTYPES, chunksize=chunksize):
        if max_day is not None:
            ch = ch[ch.d < max_day]
        parts.append(ch)
    return pd.concat(parts, ignore_index=True)


def _user_metric(name):
    def fn(task, frame, col, B, seed):
        p = frame[col].to_numpy(float)
        if np.isnan(p).any():
            raise ValueError(f"{col}: NaN predictions")
        truth = frame["truth"].to_numpy(np.int64)
        if name == "accuracy":
            per = pd.Series(p.astype(np.int64) == truth).groupby(frame["uid"].to_numpy()).mean().to_numpy()
        else:
            s = M.user_scores(frame["uid"], frame["d"], frame["t"], M.from_cell(p.astype(np.int64)),
                              M.from_cell(truth), variant=task.options.get("variant", "humob23"), metrics=(name,))
            per = s[name].to_numpy()
        return bootstrap_mean(per, B=B, seed=seed)
    return fn


@register()
class YJMob(BenchTask):
    """Options: users=2000 | 'all'; seed=0; split='test' | 'dev'; pool='all' | 'humob_test'; variant='humob23' |
    'humob25'; markov=True; epr=True."""

    name = "yjmob"
    kind = "multiclass"
    data_subdir = "yjmob100k"
    accepted_options = {
        "users": 'int | "all" (default 2000)',
        "seed": "int (default 0; user sample and stochastic baselines)",
        "split": '"test" (d 0-59 -> 60-74) | "dev" (d 0-44 -> 45-59) (default "test")',
        "pool": '"all" | "humob_test" (uid >= 80000) (default "all")',
        "variant": '"humob23" | "humob25" (default "humob23")',
        "markov": "true | false (default true)",
        "epr": "true | false (default true)",
    }
    layout = """
    <root>/yjmob100k/yjmob100k-dataset1.csv.gz   YJMob100K dataset 1 (Zenodo 10836269; columns uid,d,t,x,y;
                                                 ~111.5M rows; .csv also accepted)
    """
    source = ("Yabe et al. 2024 Sci Data 11:397 (YJMob100K, CC BY 4.0); HuMob Challenge 2023 task 1; metrics from "
              "github.com/yahoojapan/geobleu (commits cdaa364 / d9efb0f); EPR port of talp-simulation-hackathon-2026 "
              "uretim/tahmin/kod/reklam_yeri.py (20eaac9)")
    metrics = (Metric("geobleu", True, _user_metric("geobleu"), "GEO-BLEU, mean over days then users"),
               Metric("dtw", False, _user_metric("dtw"), "DTW in km, mean over days then users"),
               Metric("accuracy", True, _user_metric("accuracy"), "exact cell match share per user"))

    def _load(self, root: str) -> TaskData:
        split = self.options.get("split", "test")
        if split not in SPLITS:
            raise ValueError("split must be 'test' or 'dev'")
        first, end = SPLITS[split]
        df = read_dataset(_path(root), max_day=end)
        bad = np.unique(df.uid.to_numpy()[(df.x.to_numpy() == M.MASK) | (df.y.to_numpy() == M.MASK)])
        good = ~np.isin(df.uid.to_numpy(), bad)
        table = M.global_slot_table(df[good & (df.d.to_numpy() < first)])
        uids = np.setdiff1d(np.unique(df.uid.to_numpy()), bad)
        pool = self.options.get("pool", "all")
        if pool == "humob_test":
            uids = uids[uids >= HUMOB_TEST_UID]
        elif pool != "all":
            raise ValueError("pool must be 'all' or 'humob_test'")
        n = self.options.get("users", 2000)
        seed = int(self.options.get("seed", 0))
        if n != "all" and int(n) < len(uids):
            uids = np.sort(np.random.default_rng(seed).choice(uids, int(n), replace=False))
        df = M.clean(df[df.uid.isin(uids)])
        tr = df[df.d < first].reset_index(drop=True)
        te = df[(df.d >= first) & (df.d < end)].reset_index(drop=True)
        te = te[te.uid.isin(tr.uid.unique())].reset_index(drop=True)
        meta = {"split": f"{split}: history d < {first}, test {first} <= d < {end} (0-based days)",
                "users": int(te.uid.nunique()), "pool": pool, "seed": seed, "train_rows": len(tr),
                "test_rows": len(te), "dropped_masked_users": int(len(bad)), "global_slot_table": table.tolist(),
                "source": "YJMob100K dataset 1 (Yabe et al. 2024); CC BY 4.0"}
        return TaskData(train=tr, test=te, meta=meta)

    def baselines(self, data: TaskData) -> dict:
        q = data.test[["uid", "d", "t"]]
        table = np.asarray(data.meta["global_slot_table"], np.int64) if "global_slot_table" in data.meta else None
        return M.mobility_baselines(data.train, q, global_table=table, epr=self.options.get("epr", True),
                                    markov=self.options.get("markov", True), seed=int(self.options.get("seed", 0)))

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        return pd.DataFrame({"unit": np.arange(len(t)), "uid": t.uid.to_numpy(), "d": t.d.to_numpy(),
                             "t": t.t.to_numpy(), "group": t.uid.to_numpy(), "item": t.d.to_numpy(),
                             "time": t.d.to_numpy(np.int64) * M.SLOTS + t.t.to_numpy(),
                             "truth": M.to_cell(t.x.to_numpy(), t.y.to_numpy())})
