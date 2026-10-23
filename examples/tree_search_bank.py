"""Equal-budget comparison of search strategies on UCI Bank Marketing (bench task `bank`, row-position split).

    SIMENGINE_DATA=<data-root> OMP_NUM_THREADS=1 python examples/tree_search_bank.py --budget 20 --seeds 0 1 2 \
        --out runs/tsb
    python examples/tree_search_bank.py --synthetic --budget 4 --seeds 0      # offline smoke run, a few seconds

Space: sklearn HistGradientBoostingClassifier settings (5 x 4 x 4 x 4 x 4 x 2 = 2,560 configs). Per-unit metric:
row Brier score (lower is better). dev = the bank task's dev rows (model fit on train rows); test = the task's test
rows (model refit on train + dev with the same config). Strategies, each allowed `budget` runner calls on dev:
AIDE tree search (ConfigOperators), MCTS-UCT, random search (TreeSearch with only drafts), and Loop with a random
proposer (keeps a change only when the paired CI excludes 0). Each winner runs once on test. Output: a TSV per run and
summary.tsv under --out; nothing is written into the repo.

--synthetic (also used when no --data-root is given and $SIMENGINE_DATA has no bank-marketing file) writes a
3,000-row table in the same layout (same columns, call order, a base rate that rises over time) to a temporary
directory and loads it through the same task loader. It checks the wiring only; its numbers mean nothing. A
synthetic run defaults to --out runs/tree_search_bank-synthetic, leaves a SYNTHETIC marker there and adds a `data`
column to summary.tsv; real and synthetic runs refuse to share an --out (the searches resume from state in --out).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # run from a checkout without pip install
from simengine.bench.base import DataNotFound
from simengine.bench.tasks.bank import CATEGORICAL, NUMERIC, BankMarketing
from simengine.research.loop import Loop
from simengine.research.space import Space
from simengine.research.tree_search import ConfigOperators, MCTSSearch, TreeSearch

SPACE = dict(learning_rate=[0.02, 0.05, 0.1, 0.2, 0.4], max_leaf_nodes=[7, 15, 31, 63],
             min_samples_leaf=[5, 20, 80, 320], l2_regularization=[0.0, 0.1, 1.0, 10.0],
             max_iter=[50, 100, 200, 400], class_weight=[None, "balanced"])
DEFAULTS = dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0, max_iter=100,
                class_weight=None)   # sklearn defaults


def write_synthetic_bank(path: Path, n: int = 3000, seed: int = 0) -> Path:
    """A bank-additional-full.csv look-alike (';' separated, rows in call order, `duration` and y yes/no present).
    Planted signal: previous success, cellular contact and a falling euribor (later rows) raise the yes rate."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)                                  # call order: the base rate drifts upward over time
    pick = lambda vals, p=None: rng.choice(vals, n, p=p)      # noqa: E731
    d = pd.DataFrame({
        "age": rng.integers(18, 80, n), "job": pick(["admin.", "blue-collar", "technician", "services", "retired"]),
        "marital": pick(["married", "single", "divorced"]), "education": pick(["basic.9y", "high.school",
                                                                               "university.degree"]),
        "default": pick(["no", "unknown"], [0.8, 0.2]), "housing": pick(["yes", "no"]), "loan": pick(["no", "yes"]),
        "contact": pick(["cellular", "telephone"]), "month": pick(["may", "jun", "jul", "aug", "nov"]),
        "day_of_week": pick(["mon", "tue", "wed", "thu", "fri"]), "duration": rng.integers(0, 1000, n),
        "campaign": rng.integers(1, 8, n), "pdays": np.where(rng.random(n) < 0.9, 999, rng.integers(0, 20, n)),
        "previous": rng.poisson(0.2, n), "poutcome": pick(["nonexistent", "failure", "success"], [0.85, 0.1, 0.05]),
        "emp.var.rate": 1.4 - 3.0 * t + rng.normal(0, 0.2, n), "cons.price.idx": 93.5 + rng.normal(0, 0.5, n),
        "cons.conf.idx": -40 + rng.normal(0, 4, n), "euribor3m": 4.9 - 4.0 * t + rng.normal(0, 0.2, n),
        "nr.employed": 5200 - 200 * t + rng.normal(0, 10, n)})
    logit = (-1.0 - 0.6 * d["euribor3m"] + 2.0 * (d.poutcome == "success") + 0.5 * (d.contact == "cellular")
             - 0.1 * d.campaign + 0.01 * (d.age - 40))
    d["y"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    path.mkdir(parents=True, exist_ok=True)
    d.to_csv(path / "bank-additional-full.csv", sep=";", index=False)
    return path


def load_data(data_root, synthetic: bool):
    """(TaskData, synthetic?) from the data root, else from a temporary synthetic file."""
    if not synthetic:
        try:
            return BankMarketing(split="test").load(data_root), False
        except DataNotFound as e:
            if data_root is not None:
                raise
            print(f"no bank data ({e}); running on synthetic data, the numbers mean nothing", flush=True)
    with tempfile.TemporaryDirectory(prefix="simengine-bank-") as tmp:
        return BankMarketing(split="test").load(str(write_synthetic_bank(Path(tmp)))), True


def check_out(out: Path, synthetic: bool) -> None:
    """Keep real and synthetic runs apart: the searches resume from the state.json files under --out (with
    check_runner=False), so a real run started in a synthetic run's --out, or the reverse, would reuse its evals."""
    marker = out / "SYNTHETIC"
    used = out.is_dir() and any(out.iterdir())
    if not synthetic and marker.exists():
        raise SystemExit(f"{out} holds a synthetic smoke run (SYNTHETIC marker); pass a new --out")
    if synthetic and used and not marker.exists():
        raise SystemExit(f"{out} holds a run on real data; pass a new --out for the synthetic smoke run")
    if synthetic:
        out.mkdir(parents=True, exist_ok=True)
        marker.write_text("synthetic bank-layout table: wiring check, not real numbers\n", encoding="utf-8")


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
    ap.add_argument("--out", default=None, help="default runs/tree_search_bank (-synthetic for a smoke run)")
    ap.add_argument("--synthetic", action="store_true",
                    help="offline smoke run on a synthetic table in the same layout")
    a = ap.parse_args()
    data, synthetic = load_data(a.data_root, a.synthetic)
    if synthetic:
        print(f"SMOKE RUN on synthetic data ({len(data.train)} train+dev / {len(data.test)} test rows): wiring check, "
              "not the docs numbers", flush=True)
    out = Path(a.out or ("runs/tree_search_bank-synthetic" if synthetic else "runs/tree_search_bank"))
    check_out(out, synthetic)
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
    if synthetic:
        df.insert(0, "data", "synthetic")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.tsv", sep="\t", index=False, float_format="%.5f")
    print(df.groupby("method")[["winner_dev", "winner_test", "default_test", "test_diff"]].agg(["mean", "min", "max"])
          .round(5).to_string())
    # task baselines on the same test rows (row Brier)
    y = data.test["y"].to_numpy(float)
    for k, p in BankMarketing(split="test", lgbm=not synthetic).baselines(data).items():   # LightGBM: real data only
        print(f"baseline {k}: test brier {np.mean((np.asarray(p, float) - y) ** 2):.5f}")


if __name__ == "__main__":
    main()
