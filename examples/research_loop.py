"""Research loop workflow: pre-register, then let a proposer change one setting per turn on dev; a change is kept
only if its paired bootstrap CI excludes zero; the winner runs once on test. Offline (random proposer).

    python examples/research_loop.py [workdir]
    simengine loop examples/loop.json --workdir runs/loop-demo        # same loop from the CLI

Swap proposer="random" for "claude-p" (claude -p observer) or se.research.LLMProposer(llm) for an LLM observer.
Measured guidance (docs/RESULTS.md, dongu): on OpinionQA setting A a dev gain (-0.0106 WD) did NOT hold on test
(+0.0025 [-0.0018, +0.0070]), and the LLM observer was not clearly better than seeded random search. The loop is
honest machinery, not a source of gains: always run the random control arm next to it.
"""
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # run from a checkout without pip install
import simengine as se

N_UNITS = {"dev": 240, "select": 120, "test": 200}


def _noise(text, n):
    return np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)).normal(size=n)


def runner(cfg):
    """Fixed task code. Reads cfg['split']; returns per-unit errors (lower is better). Synthetic: the optimum is
    near shrink=0.7, k=12, method='knn'; per-unit difficulty is shared by all configs (paired comparisons)."""
    split, n = cfg["split"], N_UNITS[cfg["split"]]
    effect = 2.0 * (cfg["shrink"] - 0.7) ** 2 + 0.004 * (cfg["k"] - 12) ** 2 + {"prior": 0.15, "knn": 0.0}[cfg["method"]]
    own = 0.05 * _noise(f"{sorted(cfg.items())}", n)
    err = 1.0 + _noise(f"units-{split}", n) + effect + own
    return {"per_unit": pd.DataFrame({"unit": [f"{split}{i}" for i in range(n)], "err": err})}


def space():
    return se.research.Space({"shrink": (0.0, 1.0), "k": (1, 24), "method": ["prior", "knn"]},
                             defaults={"shrink": 0.2, "k": 4, "method": "prior"})


if __name__ == "__main__":
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="simengine-loop-"))
    pre = se.research.Prereg(
        title="shrink/k/method search", question="Does any single-setting change beat the prior on test?",
        data="synthetic units, fixed dev/select/test split", split="dev 240 / select 120 / test 200 units",
        metrics=["err (lower is better)"], baselines={"prior": "method=prior, defaults"},
        expected="winner err below the prior on test", falsification="winner - prior CI on test includes 0",
        stop_rule="3 non-kept turns in a row or 12 turns")
    pre.write(work / "prereg.md")
    se.research.lock_prereg(work / "prereg.md")               # git_commit=True inside a repo

    loop = se.research.Loop(runner, space(), proposer="random", primary="err", higher_is_better=False,
                            stop_after=3, max_turns=12, baselines={"prior": {"method": "prior"}},
                            select_split="select", prereg=work / "prereg.md", workdir=work / "random",
                            task="Minimise per-unit error")
    out = loop.run()
    v = (out.get("final") or {}).get("verdict") or {}
    print({k: out[k] for k in ("stopped", "turns")}, "verdict:", v)
    imp = v.get("dev_improvement")
    if imp:   # err is lower-is-better: dev_gain (winner - start) is negative when the loop improved things
        print(f"dev improvement (positive = better): {imp['diff']:+.3f} [{imp['lo']:+.3f}, {imp['hi']:+.3f}]")
    print("journal:", work / "random" / "JOURNAL.md")
