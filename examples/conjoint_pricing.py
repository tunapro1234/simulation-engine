"""Pricing with a choice-based conjoint: personas -> choice tasks -> MNL -> WTP -> shares and elasticity.
Offline and fast (FakeLLM with a hand-written responder; about 2 s).

    python examples/conjoint_pricing.py

Each synthetic persona carries hidden part-worths (price per TL, quality per point on a 1-5 scale, warranty per
year), drawn around known population values. The FakeLLM responder plays the persona: it reads the options from
the prompt and picks the one with the highest utility plus Gumbel noise, so the true WTP is known and the script
can print it next to the estimate. With a real model, swap the FakeLLM for `se.Claude(...)` and keep everything
else; the estimate then measures what the model says, which still has to be checked against people.
Pooled MNL ignores the per-persona spread, so its part-worths are slightly attenuated; with the seeds below the
warranty WTP lands at the lower end, with the truth inside its 95% CI.

A "none of these" option is its own alternative (id 0, all attributes 0, the base of the ASCs), so every task has
exactly one chosen row, as `MNL.fit` requires.
"""
import json
import re
import zlib

import numpy as np
import pandas as pd

import simengine as se
from simengine.methods.conjoint import MNL, elasticity, simulate_shares, wtp

TRUE = {"price": -0.004, "quality": 0.5, "warranty": 0.35}     # population part-worths per raw unit
PRODUCT_ASC = 2.0                                              # any product vs "none"
N_PERSONS, N_TASKS = 200, 8
OPT = re.compile(r"^Option (\d+): price (\d+) TL, quality (\d)/5, warranty (\d) years?$", re.M)


def personas(n, seed=0):
    """Persona text plus hidden part-worths (10% spread around TRUE); the text never mentions the numbers."""
    rs = np.random.default_rng(seed)
    jobs = ["nurse", "student", "engineer", "shop owner", "teacher", "retired driver", "designer", "cook"]
    out = []
    for i in range(n):
        w = {k: v * rs.lognormal(0, 0.1) for k, v in TRUE.items()}
        out.append({"id": f"p{i}", "text": f"{int(rs.integers(19, 70))}, {jobs[i % len(jobs)]}", "worths": w})
    return out


def tasks(n_persons, n_tasks, seed=1):
    """Long design rows: 3 random product profiles per task plus the none option (alternative 0)."""
    rs = np.random.default_rng(seed)
    rows, tid = [], 0
    for p in range(n_persons):
        for _ in range(n_tasks):
            rows.append({"person": f"p{p}", "task": tid, "alternative": 0, "price": 0, "quality": 0, "warranty": 0})
            for j in (1, 2, 3):
                rows.append({"person": f"p{p}", "task": tid, "alternative": j,
                             "price": int(rs.choice(np.arange(600, 1201, 50))),
                             "quality": int(rs.integers(1, 6)), "warranty": int(rs.integers(0, 4))})
            tid += 1
    return pd.DataFrame(rows)


def prompt(persona, rows):
    system = (f"Respondent {persona['id']}: {persona['text']}. You are shopping for a kettle. Answer as JSON "
              '{"choice": <option number>}.')
    lines = [f"Option {r.alternative}: price {r.price} TL, quality {r.quality}/5, warranty {r.warranty} "
             f"year{'s' if r.warranty != 1 else ''}" for r in rows.itertuples() if r.alternative > 0]
    return system, "\n".join(lines + ["Option 0: none of these, keep the money"])


def responder(by_id):
    """Hand-written stand-in for an LLM: utility from the persona's hidden part-worths + Gumbel noise (seeded by
    the prompt, so reruns give the same answers)."""
    def answer(system, user):
        w = by_id[re.search(r"Respondent (\S+):", system).group(1)]["worths"]
        rs = np.random.default_rng(zlib.crc32((system + user).encode()))
        best, best_u = 0, rs.gumbel()                                  # none: utility 0 + noise
        for opt, price, qual, war in OPT.findall(user):
            u = PRODUCT_ASC + w["price"] * int(price) + w["quality"] * int(qual) + w["warranty"] * int(war)
            u += rs.gumbel()
            if u > best_u:
                best, best_u = int(opt), u
        return json.dumps({"choice": best})
    return answer


panel = personas(N_PERSONS)
by_id = {p["id"]: p for p in panel}
design = tasks(N_PERSONS, N_TASKS)
llm = se.FakeLLM(responder(by_id))

# 1. Ask every persona every task; one chosen row per task (the none option included).
picks = {}
for (pid, tid), rows in design.groupby(["person", "task"], sort=False):
    reply = llm.complete(*prompt(by_id[pid], rows)).text
    picks[tid] = json.loads(reply)["choice"]
design["chosen"] = (design["alternative"] == design["task"].map(picks)).astype(int)
print(f"{design.task.nunique()} tasks from {N_PERSONS} personas, {len(llm.calls)} FakeLLM calls; "
      f"none chosen in {design.query('alternative == 0').chosen.mean():.0%}")

# 2. MNL with standard errors (price per 100 TL keeps the optimiser well scaled; wtp/elasticity undo the scale).
m = MNL(["price", "quality", "warranty"], scale={"price": 100}).fit(design)
truth = {"price": TRUE["price"] * 100, "quality": TRUE["quality"], "warranty": TRUE["warranty"]}
coef = pd.DataFrame({"true": pd.Series(truth), "coef": m.coef_, "se": m.se_}).loc[list(truth)]
print("\npart-worths (price per 100 TL):\n" + coef.round(3).to_string())

# 3. WTP in TL per unit, delta-method CI and a person-cluster bootstrap CI.
true_wtp = pd.Series({a: -TRUE[a] / TRUE["price"] for a in ("quality", "warranty")}, name="true")
delta = wtp(m, price="price")
boot = wtp(m, price="price", ci="bootstrap", cluster="person", B=100, seed=0)
table = pd.concat([true_wtp, delta[["wtp", "lo", "hi"]], boot[["lo", "hi"]].add_prefix("boot_")], axis=1)
print("\nWTP, TL per quality point / per warranty year (95% CI):\n" + table.round(1).to_string())

# 4. A market: three kettles and the none option; shares and price elasticities.
market = pd.DataFrame({"alternative": [1, 2, 3, 0], "name": ["basic", "mid", "premium", "none"],
                       "price": [700, 900, 1150, 0], "quality": [2, 3, 5, 0], "warranty": [1, 2, 3, 0]})
e = elasticity(m, market, price="price", price_grid=[0.9, 1.1])
out = market.set_index("alternative")[["name", "price"]].assign(share=e["share"], own=e["own"])
arc = e["arc"].pivot(index="alternative", columns="factor", values="arc").add_prefix("arc_x")
print("\nshares and own-price elasticity:\n" + out.join(arc).round(3).to_string())
print("\ncross-price elasticity d log s_row / d log p_col:\n"
      + e["cross"].rename(index=out["name"], columns=out["name"]).round(3).to_string())

# 5. What-if: premium at 999 TL.
what_if = market.assign(price=market["price"].where(market["name"] != "premium", 999))
print("\npremium at 999 TL, shares:", simulate_shares(m, what_if).set_axis(market["name"]).round(3).to_dict())
