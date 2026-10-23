"""Hyperband and ASHA vs plain successive halving and random search at an equal nominal budget (synthetic, offline).

Two fidelity types, both with max_budget R = 81 and eta = 3 (Hyperband budget 1902 units):
- "sample": budget = number of units (rows, personas or LLM calls). The score at budget b is the mean of b noisy
  per-unit scores, so low budgets are unbiased but noisy (UnitRunner, resume on).
- "curve": budget = iterations. The score at b is mu + a / sqrt(b / R) - a with a per-config slowness a that is
  larger for configs near the optimum, so low budgets mislead (late bloomers); small noise on top.
Regret = true full-budget loss of the returned config (the optimum is 0). Lower is better.

    python examples/hyperband_vs_halving.py --seeds 200 --out hb.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # run from a checkout without pip install
from simengine.research import Space, successive_halving
from simengine.research.hyperband import MultiFidelity, UnitRunner, asha, hyperband, space_sampler
from simengine.research.stats import bootstrap_mean, paired_bootstrap_diff

R, ETA, N_UNITS = 81, 3, 81
SPACE = Space({"x0": (0.0, 1.0), "x1": (0.0, 1.0), "x2": (0.0, 1.0), "m": ["a", "b", "c"]})
OFFSET = {"a": 0.0, "b": 0.08, "c": 0.2}


def mu(cfg):
    return sum((cfg[f"x{i}"] - 0.7) ** 2 for i in range(3)) + OFFSET[cfg["m"]]


def slowness(cfg):
    return 0.6 * np.exp(-8 * (cfg["x0"] - 0.7) ** 2)          # near-optimal x0 learns slowly


def make_runner(kind: str, seed: int):
    rng = np.random.default_rng(10_000 + seed)
    unit = rng.normal(0, 0.5, N_UNITS)                         # shared per-unit difficulty

    def own(cfg, n):
        key = [seed, "abc".index(cfg["m"])] + [int(round(cfg[f"x{i}"] * 1e9)) for i in range(3)]
        return np.random.default_rng(key).normal(0, 0.5, n)          # same config -> same noise, every process

    if kind == "sample":
        return UnitRunner(lambda cfg, idx: mu(cfg) + unit[idx] + own(cfg, N_UNITS)[idx], N_UNITS, seed=seed,
                          unit="samples")

    def curve(cfg, b):
        a = slowness(cfg)
        return mu(cfg) + a / np.sqrt(b / R) - a + 0.05 * own(cfg, 1)[0] / np.sqrt(b / R)
    return MultiFidelity(curve, R, unit="iterations")


def run_all(kind: str, seed: int) -> dict:
    out = {}
    sample = space_sampler(SPACE)
    hb_budget = 1902
    # random search: floor(B / R) configs at full budget
    run = make_runner(kind, seed)
    cfgs = sample(np.random.default_rng(seed), hb_budget // R)
    sc = run.screen(cfgs, 1.0)
    out["random"] = (cfgs[int(np.argmin(sc))], len(cfgs) * R, run.spent)
    # plain halving as HalvingProposer uses it: fidelities (1/9, 1/3, 1), ceil rounding, n = 70 (budget 1926)
    run = make_runner(kind, seed)
    cfgs = sample(np.random.default_rng(seed), 70)
    sh = successive_halving(run.screen, cfgs, higher_is_better=False, fidelities=(1 / 9, 1 / 3, 1.0), eta=ETA)
    out["halving_1/9"] = (sh["winner"], int(sum(len(p) * b for p, b in zip(sh["rungs"], (9, 27, 81)))), run.spent)
    # most aggressive bracket s = 4 repeated 5 times (budget 2025)
    run = make_runner(kind, seed)
    hb4 = hyperband(run, sample, higher_is_better=False, eta=ETA, brackets=[4], repeats=5, seed=seed)
    out["halving_s4x5"] = (hb4["winner"], hb4["budget"], run.spent)
    run = make_runner(kind, seed)
    hb = hyperband(run, sample, higher_is_better=False, eta=ETA, seed=seed)
    assert hb["budget"] == hb_budget
    out["hyperband"] = (hb["winner"], hb["budget"], run.spent)
    run = make_runner(kind, seed)
    a = asha(run, sample, higher_is_better=False, eta=ETA, total_budget=hb_budget, seed=seed)
    out["asha"] = (a["winner"], a["budget"], run.spent)
    return {m: {"regret": mu(w), "budget": b, "spent": s} for m, (w, b, s) in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    report = {}
    for kind in ("sample", "curve"):
        rows = [run_all(kind, s) for s in range(args.seeds)]
        methods = list(rows[0])
        reg = {m: np.array([r[m]["regret"] for r in rows]) for m in methods}
        res = {}
        for m in methods:
            ci = bootstrap_mean(reg[m])
            d = paired_bootstrap_diff(reg[m], reg["hyperband"])
            res[m] = {"regret": ci["value"], "lo": ci["lo"], "hi": ci["hi"],
                      "vs_hyperband": d["diff"], "vs_lo": d["lo"], "vs_hi": d["hi"],
                      "budget": float(np.mean([r[m]["budget"] for r in rows])),
                      "spent": float(np.mean([r[m]["spent"] for r in rows]))}
        report[kind] = res
        print(f"\n## {kind} fidelity, {args.seeds} seeds")
        print("| method | nominal budget | units charged | mean regret [95% CI] | minus Hyperband [95% CI] |")
        print("|---|---|---|---|---|")
        for m, v in res.items():
            print(f"| {m} | {v['budget']:.0f} | {v['spent']:.0f} | {v['regret']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}] "
                  f"| {v['vs_hyperband']:+.4f} [{v['vs_lo']:+.4f}, {v['vs_hi']:+.4f}] |")
    print(f"\n{time.time() - t0:.1f} s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
