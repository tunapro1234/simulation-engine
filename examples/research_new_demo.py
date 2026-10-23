"""One tour of the research search modules on synthetic problems (offline, FakeLLM, no data, about 10 s).

1. bandits: which of 4 methods is best when every evaluation is a noisy 0/1 score (successive rejects vs uniform).
2. bayesopt: TPE-lite vs random search on Branin at 30 calls, and GP-EI at 15.
3. hyperband: Hyperband and ASHA over a per-unit scorer where budget = units (rows, personas or LLM calls).
4. tree_search: AIDE-style draft / debug / improve over a config space, winner run once on test.
5. prompt_evolution: OPRO on the keyword toy task (FakeLLM optimiser), validation pick, one test pass.
6. hypotheses: an LLM-proposed linear SCM fitted on a known linear world (FakeLLM proposer and forecaster).

Every number printed here is from synthetic problems with a scripted FakeLLM; it checks the wiring, not how well any
method works on real tasks (measured numbers: docs/modules/research/<module>.md).

    PYTHONPATH=. python examples/research_new_demo.py
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time

import numpy as np
import pandas as pd

import simengine.research as R
from simengine.llm.fake import FakeLLM


def demo_bandits():
    arms = R.SimulatedArms([0.65, 0.55, 0.50, 0.45], kind="bernoulli", seed=1)   # true accuracy of 4 methods
    res = R.identify_best(arms, 4, budget=400, algorithm="successive_rejects", seed=0)
    rates = {a: R.bandits.simulate(arms, a, budget=400, reps=500, seed=0)["error_rate"]
             for a in ("uniform", "successive_rejects")}
    print(f"[bandits] SR picked arm {res.best} (true best 0), pulls {res.pulls.tolist()}; "
          f"error over 500 runs: uniform {rates['uniform']:.3f}, SR {rates['successive_rejects']:.3f}")


def demo_bayesopt():
    space = R.branin_space()
    f = lambda c: R.branin(c["x1"], c["x2"])                                    # noqa: E731
    out = {m: R.bayesopt.minimize(f, space, method=m, n_calls=30, seed=0)["best"] - R.BRANIN_MIN
           for m in ("random", "tpe")}
    out["gp@15"] = R.bayesopt.minimize(f, space, method="gp", n_calls=15, seed=0)["best"] - R.BRANIN_MIN
    print("[bayesopt] Branin regret, seed 0: " + ", ".join(f"{k} {v:.3f}" for k, v in out.items()))


def demo_hyperband():
    noise = np.random.default_rng(0).normal(size=81)
    score = lambda cfg, idx: (cfg["x"] - 0.7) ** 2 + 0.3 * noise[idx]            # noqa: E731
    sample = R.space_sampler(R.Space({"x": (0.0, 1.0)}))
    hb = R.hyperband.hyperband(R.UnitRunner(score, n_units=81, unit="llm_calls"), sample,
                               higher_is_better=False, eta=3, seed=0)
    run = R.UnitRunner(score, n_units=81, unit="llm_calls")
    a = R.asha(run, sample, higher_is_better=False, total_budget=hb["budget"], seed=0)
    print(f"[hyperband] Hyperband x={hb['winner']['x']:.3f} (nominal budget {hb['budget']}); "
          f"ASHA x={a['winner']['x']:.3f}, units charged {run.spent} (optimum x=0.7)")


def _unit_noise(text: str, n: int) -> np.ndarray:
    return np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)).normal(size=n)


def tree_runner(cfg):
    """Lower is better: per-unit error with shared unit difficulty; method 'c' crashes."""
    if cfg["method"] == "c":
        raise RuntimeError("method c is broken")
    n = {"dev": 200, "test": 400}[cfg["split"]]
    eff = 2.0 * (cfg["x"] - 0.7) ** 2 + 0.004 * (cfg["k"] - 12) ** 2 + {"a": 0.15, "b": 0.0}[cfg["method"]]
    key = json.dumps({k: v for k, v in sorted(cfg.items()) if k != "split"})
    err = 1.0 + _unit_noise("u-" + cfg["split"], n) + eff + 0.02 * _unit_noise(key + cfg["split"], n)
    return {"per_unit": pd.DataFrame({"unit": np.arange(n), "err": err})}


def demo_tree_search(tmp: str):
    space = R.Space({"x": (0.0, 1.0), "k": (1, 24), "method": ["a", "b", "c"]},
                    defaults={"x": 0.1, "k": 4, "method": "a"})
    ts = R.TreeSearch(tree_runner, R.ConfigOperators(space), primary="err", higher_is_better=False, steps=15,
                      workdir=f"{tmp}/tree", seed=0)
    out = ts.run()
    fin = ts.finalize()["final"]
    d = fin["test_diff"]["winner-reference"]["ci95"]                           # reference = the space defaults
    print(f"[tree_search] {out['n_nodes']} nodes, winner {out['best_payload']}; test winner - defaults "
          f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], winner test - dev {fin['winner_test_minus_dev']:+.3f}")


def demo_prompt_evolution(tmp: str):
    units, score, keys, responder = R.prompt_evolution.toy_problem(n_units=240, seed=0)
    tr, va, te = R.prompt_evolution.split_units(units, seed=0)
    ev = R.PromptEvaluator(score, tr, va, te)
    res = R.evolve("opro", FakeLLM(responder), ev, seeds=["use kw05"], steps=4, per_step=4,
                   registry_dir=f"{tmp}/pe/.prereg")
    w, s = res.test["winner"]["mean"], res.test["seed0"]["mean"]
    print(f"[prompt_evolution] OPRO winner {res.best.text!r}: test {w:.3f} vs seed text {s:.3f}")


HYP = {"hypotheses": [{
    "agents": ["shopper"], "rationale": "a deeper discount raises intent; a long form lowers it",
    "outcome": {"name": "intent", "kind": "continuous", "units": "points", "agent": "shopper",
                "question": "How likely are you to buy, 0-10?", "lo": 0, "hi": 10},
    "causes": [{"name": "discount", "kind": "continuous", "units": "percent", "levels": [0, 10, 20],
                "proxy": "Discount shown", "agent": "shopper"},
               {"name": "form", "kind": "binary", "levels": ["short", "long"], "proxy": "Checkout form",
                "agent": "shopper"}]}]}


def ass_responder(system, user):
    if system == R.hypotheses.HYPOTHESIS_SYSTEM:
        return json.dumps(HYP)
    if system == R.hypotheses.PATH_SYSTEM:
        return json.dumps({"predictions": {"discount": 0.1, "form": -1.0},
                           "sig": {"discount": True, "form": True}})
    return "{}"


def demo_hypotheses(tmp: str):
    world = R.linear_world({"discount": 0.08}, intercept=4.0, noise=0.5, seed=0, levels={"form": {"long": -0.6}})
    ass = R.AutoSocialScience(FakeLLM(ass_responder), "An online shop tests its checkout page.", simulator=world,
                              replicates=20, workdir=f"{tmp}/ass")
    res = ass.run(n_hypotheses=1)
    c = res.runs[0].comparison
    rows = ", ".join(f"{r.term} fit {r.coef:+.3f} (LLM said {r.predicted:+.2f})" for r in c.itertuples())
    print(f"[hypotheses] {rows}; truth discount +0.080, form (long) -0.600")


def main():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        demo_bandits()
        demo_bayesopt()
        demo_hyperband()
        demo_tree_search(tmp)
        demo_prompt_evolution(tmp)
        demo_hypotheses(tmp)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
