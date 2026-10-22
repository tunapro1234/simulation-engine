"""Equal-budget comparison of search strategies on UCI Bank Marketing (bench task `bank`, row-position split).

    SIMENGINE_DATA=<data-root> OMP_NUM_THREADS=1 python examples/tree_search_bank.py --budget 20 --seeds 0 1 2 \
        --out runs/tsb

Space: sklearn HistGradientBoostingClassifier settings (5 x 4 x 4 x 4 x 4 x 2 = 2,560 configs). Per-unit metric:
row Brier score (lower is better). dev = the bank task's dev rows (model fit on train rows); test = the task's test
rows (model refit on train + dev with the same config). Strategies, each allowed `budget` runner calls on dev:
AIDE tree search (ConfigOperators), MCTS-UCT, random search (TreeSearch with only drafts), and Loop with a random
proposer (keeps a change only when the paired CI excludes 0). Each winner runs once on test. Output: a TSV per run and
summary.tsv under --out; nothing is written into the repo.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from simengine.bench.tasks.bank import CATEGORICAL, NUMERIC, BankMarketing
from simengine.research.loop import Loop
from simengine.research.space import Space
from simengine.research.tree_search import ConfigOperators, MCTSSearch, TreeSearch

SPACE = dict(learning_rate=[0.02, 0.05, 0.1, 0.2, 0.4], max_leaf_nodes=[7, 15, 31, 63],
             min_samples_leaf=[5, 20, 80, 320], l2_regularization=[0.0, 0.1, 1.0, 10.0],
             max_iter=[50, 100, 200, 400], class_weight=[None, "balanced"])
DEFAULTS = dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0, max_iter=100,
                class_weight=None)   # sklearn defaults


def make_runner(data):
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr, te = data.train, data.test
    allx = pd.concat([tr, te])
    cats = {c: sorted(allx[c].astype(str).unique()) for c in CATEGORICAL}

    def X(df):
        x = df[NUMERIC].astype(float).copy()
        for c in CATEGORICAL:
            x[c] = pd.Categorical(df[c].astype(str), categories=cats[c]).codes
        return x

    parts = {"train": tr[tr.part == "train"], "dev": tr[tr.part == "dev"], "all": tr, "test": te}
    xs = {k: X(v) for k, v in parts.items()}
    mask = [False] * len(NUMERIC) + [True] * len(CATEGORICAL)
    memo = {}

    def runner(cfg):
        split = cfg["split"]
        hp = {k: cfg[k] for k in SPACE}
        key = (json.dumps(hp, sort_keys=True), split)
        if key not in memo:
            fit, ev = ("train", "dev") if split == "dev" else ("all", "test")
            m = HistGradientBoostingClassifier(**hp, categorical_features=mask, early_stopping=False, random_state=0)
            m.fit(xs[fit], parts[fit]["y"])
            p = m.predict_proba(xs[ev])[:, 1]
            y = parts[ev]["y"].to_numpy(float)
            memo[key] = pd.DataFrame({"unit": np.arange(len(y)), "brier": (p - y) ** 2})
        return {"per_unit": memo[key]}

    runner.memo = memo
    return runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/tree_search_bank")
    a = ap.parse_args()
    data = BankMarketing(split="test").load(a.data_root)
    out = Path(a.out)
    runner = make_runner(data)
    sp = Space(SPACE, defaults=DEFAULTS)
    rows = []
    for seed in a.seeds:
        B = a.budget
        runs = {
            "aide": lambda d: TreeSearch(runner, ConfigOperators(sp), primary="brier", higher_is_better=False,
                                         steps=B, seed=seed, workdir=d, check_runner=False),
            "mcts": lambda d: MCTSSearch(runner, sp, primary="brier", higher_is_better=False, budget=B, seed=seed,
                                         workdir=d, check_runner=False),
            "random": lambda d: TreeSearch(runner, ConfigOperators(sp), primary="brier", higher_is_better=False,
                                           steps=B, num_drafts=B, seed=seed, workdir=d, check_runner=False),
        }
        for name, mk in runs.items():
            t0 = time.time()
            s = mk(out / f"{name}-s{seed}")
            r = s.run(finalize=True)
            f = r["final"]
            d = f["test_diff"]["winner-reference"]["ci95"]
            rows.append({"method": name, "seed": seed, "dev_evals": sum(n["eval_key"] is not None and
                                                                        n["repeat_of"] is None for n in r["history"]),
                         "winner_dev": f["winner_dev"], "winner_test": f["test"]["winner"]["score"],
                         "default_test": f["test"]["reference"]["score"], "test_diff": d["diff"],
                         "diff_lo": d["lo"], "diff_hi": d["hi"], "winner": json.dumps(f["winner_payload"]),
                         "seconds": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
        t0 = time.time()
        lp = Loop(runner, sp, "random", primary="brier", higher_is_better=False, stop_after=None, max_turns=B - 1,
                  workdir=out / f"loop-s{seed}", check_runner=False, ablation=False,
                  proposer_kw={"seed": 1000 + seed})
        r = lp.run()
        f = r["final"]
        d = f["test_diff"]["winner-start"]["ci95"]
        rows.append({"method": "loop-random", "seed": seed, "dev_evals": len(r["history"]),
                     "winner_dev": (r["best_dev"] or {}).get("value"),
                     "winner_test": f["test"]["winner"]["metrics"]["brier"]["value"],
                     "default_test": f["test"]["start"]["metrics"]["brier"]["value"], "test_diff": d["diff"],
                     "diff_lo": d["lo"], "diff_hi": d["hi"], "winner": json.dumps(f["winner_cfg"]),
                     "seconds": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.tsv", sep="\t", index=False, float_format="%.5f")
    print(df.groupby("method")[["winner_dev", "winner_test", "default_test", "test_diff"]].agg(["mean", "min", "max"])
          .round(5).to_string())
    # task baselines on the same test rows (row Brier)
    y = data.test["y"].to_numpy(float)
    for k, p in BankMarketing(split="test").baselines(data).items():
        print(f"baseline {k}: test brier {np.mean((np.asarray(p, float) - y) ** 2):.5f}")


if __name__ == "__main__":
    main()
