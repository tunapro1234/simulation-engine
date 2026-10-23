"""Economic games: persona method zoo -> one-shot game battery -> distance to published human numbers, coherence
audit, benchmark score and the who-to-punish society scorecard. Offline: a FakeLLM plays every game.

    python examples/econ_games.py

The FakeLLM answers are hash-seeded noise with a nudge for the "generous" persona, so every number printed here is
about the plumbing, not about any model. Measured guidance (docs/RESULTS.md, "Economic games"): earlier repos on
non-Claude models found that the bare model sits at the self-interested corner and that no-persona agents can fit the
institutional result better than persona agents. Rerun through `simengine.llm` before claiming anything about Claude.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # run from a checkout without pip install
import simengine as se
import simengine.games as gm
from simengine import bench
from simengine.bench.tasks import ertan_wtp as W

_NUM = re.compile(r"single number from (-?[\d.]+) to (-?[\d.]+)")
_MC = re.compile(r"Choose exactly one of: (.*?)\. Reply")


def responder(system, user):
    """Read the answer-format line the game appended and answer inside it (pseudo-random, persona-nudged)."""
    h = int(hashlib.sha256((system + "\x00" + user).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if "generous" in system:
        h = 0.5 + h / 2
    m = _MC.search(user)
    if m:
        opts = re.findall(r'"([^"]+)"', m.group(1))
        return json.dumps({"answer": opts[int(h * len(opts)) % len(opts)]})
    m = _NUM.search(user)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return json.dumps({"answer": round(lo + h * (hi - lo))})
    return "I am not sure."                                     # unusable reply: stays missing, never imputed


llm = se.FakeLLM(responder)

# 1. Personas: the bare-model control (empty text), a toy persona, and 4 draws from the method zoo.
bank = {"baseline": {"baseline-000": ""},
        "toy": {"toy-000": {"text": "You are generous and trusting.", "params": {"generous": 1}}},
        "big5": {p["id"]: {"text": p["text"], "params": p["params"]}
                 for p in se.persona.method_personas("big5", 4, seed=0)}}
print("persona methods in the zoo:", ", ".join(m["name"] for m in se.persona.list_methods()))

# 2. Play the battery (calibration + holdout). The cost gate runs first; FakeLLM costs 0.
print("cost estimate:", {k: v for k, v in gm.estimate_cost(llm, bank, "all", runs=2).items() if k in ("calls", "usd")})
res = gm.play(llm, bank, "all", runs=2, seed=0)
print(f"rows: {len(res)}  invalid: {int((~res['valid'].astype(bool)).sum())}")

# 3. Distance to human: moments vs cited human references, agent-bootstrap CI, W1/KS for eleven_twenty.
dist = gm.distance_to_human(res, B=100)
show = dist[dist.persona_method == "big5"].dropna(subset=["abs_error"]).head(6)
print(show[["game", "moment", "metric", "sim", "human", "abs_error", "ci_lo", "ci_hi"]].round(3).to_string(index=False))

# 4. Coherence audit: flags mark rows, they never drop them.
audit = gm.coherence_audit(res)
print("coherence flags:", audit.groupby("flag").size().to_dict() if len(audit) else {})

# 5. Benchmark: holdout-game MAE against the rational-corner and midpoint baselines.
task = bench.get("econ_games_b01")
frame = task.run({m: res[res.persona_method == m] for m in bank})
for m in ["rational_corner", "midpoint", *bank]:
    s = task.score(frame, m)
    print(f"econ_games_b01 MAE {m:16s} {s['value']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}] n={s['n']}")


# 6. Who-to-punish society (Ertan, Page & Putterman 2009 design) -> 9-criterion human-likeness scorecard.
def society(system, user):
    if "Ballot item" in user:
        low = "LESS than" in user
        return json.dumps({"vote": "Yes" if low else "No"})
    if "How much of your $10" in user:
        return json.dumps({"contribution": 3 if "free" in system else 8})
    if "reduce their earnings" in user:
        return json.dumps({"punish": 1})
    return "?"


run = se.abm.run_who_to_punish(se.FakeLLM(society), ["A steady cooperator."] * 3 + ["A free rider."], seed=0)
card = gm.wtp_scorecard(W.group_to_run(run["groups"][0]))
print(f"who-to-punish index {card['index']:.3f} over {card['n_criteria']} observed criteria;",
      f"invalid answers {run['invalid']['total']}")
