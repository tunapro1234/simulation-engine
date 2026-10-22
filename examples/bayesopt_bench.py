"""Bayesian optimisation vs random search at equal budget (synthetic; the numbers in docs/modules/research/bayesopt.md).

    OMP_NUM_THREADS=1 python examples/bayesopt_bench.py --seeds 20 --loop-seeds 10 --jobs 8 --out runs/bo.json

A. `minimize` over full configs (every key moves), `--calls` objective calls: Branin and the mixed categorical toy;
   random, GP-EI, GP-UCB, TPE. Regret = best value found - known minimum (Branin 0.397887, toy 0).
B. `Loop` with one change per turn (the proposer interface), `--turns` turns, stop_after=None, no finalize: noisy
   per-unit Branin / toy runners; RandomProposer vs GPProposer (EI, UCB) vs TPEProposer. Score = true (noise-free)
   loss of the final incumbent minus the known minimum.
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from simengine.research import Loop, RandomProposer
from simengine.research.bayesopt import (BRANIN_MIN, GPProposer, TPEProposer, branin, branin_space, minimize,
                                         mixed_toy, mixed_toy_space)

N_UNITS = 100
FUNCS = {"branin": (lambda c: float(branin(c["x1"], c["x2"])), branin_space, BRANIN_MIN),
         "toy": (mixed_toy, mixed_toy_space, 0.0)}
FULL = {"random": dict(method="random"), "gp-ei": dict(method="gp", acquisition="ei"),
        "gp-ucb": dict(method="gp", acquisition="ucb"), "tpe": dict(method="tpe")}
NOISE = {"branin": (5.0, 1.0), "toy": (0.2, 0.02)}   # sd of shared unit difficulty, sd of per-config noise


def _u(text, n):
    return np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)).normal(size=n)


def _runner(fname):
    f, _, _ = FUNCS[fname]
    sd_unit, sd_own = NOISE[fname]

    def run(cfg):
        s = cfg["split"]
        key = json.dumps({k: v for k, v in sorted(cfg.items()) if k != "split"})
        err = f(cfg) + sd_unit * _u("units" + s, N_UNITS) + sd_own * _u(key + s, N_UNITS)
        return {"per_unit": pd.DataFrame({"unit": range(N_UNITS), "err": err})}
    return run


def _proposer(name, seed):
    return {"random": lambda: RandomProposer(seed=1000 + seed), "gp-ei": lambda: GPProposer("ei", seed=seed),
            "gp-ucb": lambda: GPProposer("ucb", seed=seed), "tpe": lambda: TPEProposer(seed=seed)}[name]()


def job_full(args):
    fname, method, seed, calls = args
    f, sp, fmin = FUNCS[fname]
    t0 = time.time()
    r = minimize(f, sp(), n_calls=calls, seed=seed, **FULL[method])
    return {"part": "full", "func": fname, "method": method, "seed": seed, "regret": [t - fmin for t in r["trace"]],
            "seconds": round(time.time() - t0, 2)}


def job_loop(args):
    fname, method, seed, turns = args
    f, sp, fmin = FUNCS[fname]
    t0 = time.time()
    out = Loop(_runner(fname), sp(), proposer=_proposer(method, seed), primary="err", higher_is_better=False,
               stop_after=None, max_turns=turns).run(finalize=False)
    keeps = [t for t in out["history"] if t["status"] == "keep"]
    return {"part": "loop", "func": fname, "method": method, "seed": seed, "regret": f(keeps[-1]["cfg"]) - fmin,
            "kept": len(keeps) - 1, "turns": len(out["history"]) - 1, "seconds": round(time.time() - t0, 2)}


def summarise(rows, calls):
    out = {}
    for r in rows:
        key = f"{r['part']}/{r['func']}/{r['method']}"
        out.setdefault(key, []).append(r)
    table = {}
    for key, rs in sorted(out.items()):
        if rs[0]["part"] == "full":
            R = np.array([r["regret"] for r in rs])
            table[key] = {"n": len(rs), **{f"median@{c}": float(np.median(R[:, c - 1])) for c in (10, 20, calls)},
                          f"mean@{calls}": float(R[:, -1].mean()), f"share<0.1@{calls}": float((R[:, -1] < 0.1).mean()),
                          "sec": float(np.mean([r["seconds"] for r in rs]))}
        else:
            R = np.array([r["regret"] for r in rs])
            table[key] = {"n": len(rs), "median": float(np.median(R)), "mean": float(R.mean()),
                          "kept": float(np.mean([r["kept"] for r in rs])), "sec": float(np.mean([r["seconds"] for r in rs]))}
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--loop-seeds", type=int, default=10)
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    jobs_full = [(f, m, s, a.calls) for f in FUNCS for m in FULL for s in range(a.seeds)]
    jobs_loop = [(f, m, s, a.turns) for f in FUNCS for m in FULL for s in range(a.loop_seeds)]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(job_full, jobs_full)) + list(ex.map(job_loop, jobs_loop))
    table = summarise(rows, a.calls)
    for k, v in table.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()})
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"args": vars(a), "table": table, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
