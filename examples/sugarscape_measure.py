"""Sugarscape measurements quoted in docs/modules/abm/sugarscape.md (offline, ~17 min on one busy core).

    python examples/sugarscape_measure.py [capacity_grid.txt]
    python examples/sugarscape_measure.py --quick       # smoke run, < 30 s: same tables, NOT the docs numbers

The optional argument is a whitespace capacity grid (e.g. the NetLogo transcription of the book map) read at runtime.
--quick checks that every section runs and prints the same tables: 2 seeds, 60 steps (36 for section D), a 25x25 map
with a quarter of the agents, every reported step scaled by 60/500 (column names carry the scaled step). Its values
are small-scale smoke numbers, not the measurements in the docs.
"""
import argparse, hashlib, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # run from a checkout without pip install
from simengine.abm.sugarscape import (Sugarscape, SugarscapeConfig, LLMPolicy, gini_trajectory, load_capacity,
                                      sugar_capacity, gini)
from simengine.llm.fake import FakeLLM

ap = argparse.ArgumentParser(description="Sugarscape measurements of docs/modules/abm/sugarscape.md")
ap.add_argument("capacity_grid", nargs="?", default=None, help="whitespace capacity grid; adds row A'")
ap.add_argument("--quick", action="store_true", help="smoke run (< 30 s): small map, 2 seeds, 60 steps")
args = ap.parse_args()
alt_map = args.capacity_grid

pd.set_option("display.width", 200)
# full run = the docs numbers; quick = same sections at small scale (steps scaled by N / 500)
N = 60 if args.quick else 500
SEEDS = range(2) if args.quick else range(10)
D_SEEDS = range(2) if args.quick else range(5)
SIZE, N_WEALTH, N_CC = (25, 62, 100) if args.quick else (50, 250, 400)
def at(t):                                                    # docs step t -> step of this run
    return round(t * N / 500)
AT = tuple(at(t) for t in (0, 10, 25, 50, 100, 150, 200, 300, 400, 500))
N_D = at(300)
def cfg(**kw):
    return SugarscapeConfig(**{"size": SIZE, "n_agents": N_WEALTH, **kw})
if args.quick:
    print(f"SMOKE RUN (--quick): {SIZE}x{SIZE} map, {len(SEEDS)} seeds, {N} steps; not the docs numbers\n")
t0 = time.time()
U = np.arange(5, 26)
print("Gini of U{5..25} (exact, population):", round(gini(np.repeat(U, 1000)), 4))

def traj(c, label):
    d = gini_trajectory(seeds=SEEDS, n_steps=N, config=c, at=AT)
    g = d.groupby("step").gini.agg(["mean", "std"]).round(3)
    print(f"\n[{label}] Gini by step, mean / sd over {len(SEEDS)} seeds\n", g.T.to_string())
    return d

traj(cfg(), f"A wealth: {N_WEALTH} agents, G1, M, R[60,100], parametric map")
if alt_map:
    traj(cfg(capacity=load_capacity(alt_map)), "A' same, transcribed map read at runtime")

# end-of-run shape
rows = []
for s in SEEDS:
    st = Sugarscape(cfg(), seed=s).run(N).stats
    late = st.iloc[at(200):]
    rows.append({"seed": s, f"gini_{at(200)}_{N}": late.gini.mean(), f"skew_{N}": st.skewness.iloc[-1],
                 f"top10_{N}": st.top10_share.iloc[-1], "skew_0": st.skewness.iloc[0],
                 "top10_0": st.top10_share.iloc[0]})
r = pd.DataFrame(rows)
print(f"\n[A] steps {at(200)}-{N} mean Gini (per-seed mean of the window), skewness and top-10% share\n",
      r.drop(columns="seed").agg(["mean", "std"]).round(3).to_string())

# carrying capacity, no ageing
cc = []
for s in SEEDS:
    st = Sugarscape(cfg(n_agents=N_CC, max_age=None, replace=False), seed=s).run(N).stats
    cc.append({f"pop_{at(100)}": st.population.iloc[at(100)], f"pop_{N}": st.population.iloc[N],
               f"gini_{N}": st.gini.iloc[N], "vis_0": st.mean_vision.iloc[0], f"vis_{N}": st.mean_vision.iloc[N],
               "met_0": st.mean_metabolism.iloc[0], f"met_{N}": st.mean_metabolism.iloc[N]})
print(f"\n[B] carrying capacity: {N_CC} agents, G1, M, no ageing / replacement\n",
      pd.DataFrame(cc).agg(["mean", "std"]).round(3).to_string())

# spice with / without trade
for trade in (False, True):
    cc = []
    for s in SEEDS:
        st = Sugarscape(cfg(n_agents=N_CC, max_age=None, replace=False, spice=True, trade=trade),
                        seed=s).run(N).stats
        row = {f"pop_{N}": st.population.iloc[N], f"gini_{N}": st.gini.iloc[N]}
        if trade:
            row.update({f"trades_{N}": st.trades.iloc[N], f"plogsd_{at(10)}": st.price_log_sd.iloc[at(10)],
                        f"plogsd_{at(100)}": st.price_log_sd.iloc[at(100)], f"plogsd_{N}": st.price_log_sd.iloc[N],
                        f"pgeo_{N}": st.price_geo_mean.iloc[N]})
        cc.append(row)
    print(f"\n[C] sugar+spice carrying capacity, {N_CC} agents, no ageing, trade={trade}\n",
          pd.DataFrame(cc).agg(["mean", "std"]).round(3).to_string())
    g = gini_trajectory(seeds=SEEDS, n_steps=N, config=cfg(spice=True, trade=trade), at=AT)
    print(f"[C] Gini (sugar + spice holdings), {N_WEALTH} agents, R[60,100], trade={trade}\n",
          g.groupby("step").gini.agg(["mean", "std"]).round(3).T.to_string())

# LLM policy plumbing with FakeLLM: greedy responder and a random-site responder
def greedy(system, user):
    sites = [(int(j), float(v)) for j, v in re.findall(r"^(\d+): [^,]*, sugar ([\d.]+)", user, re.M)]
    return json.dumps({"site": max(sites, key=lambda t: (t[1], -t[0]))[0]})

def randsite(system, user):
    n = len(re.findall(r"^\d+: ", user, re.M))
    return json.dumps({"site": int(hashlib.sha256(user.encode()).hexdigest(), 16) % n})

for name, resp in (("greedy", greedy), ("random-site", randsite)):
    gs = []
    for s in D_SEEDS:
        pol = LLMPolicy(FakeLLM(resp))
        run = Sugarscape(cfg(), seed=s, policy=pol).run(N_D)
        gs.append(dict({f"gini_{at(100)}": run.stats.gini.iloc[at(100)], f"gini_{N_D}": run.stats.gini.iloc[N_D]},
                       starved=run.stats.starved.sum(), calls=run.calls, invalid=run.invalid,
                       conflicts=run.conflicts))
    print(f"\n[D] LLMPolicy + FakeLLM {name}, {N_WEALTH} agents, {N_D} steps, {len(D_SEEDS)} seeds\n",
          pd.DataFrame(gs).agg(["mean", "std"]).round(3).to_string())
base = [Sugarscape(cfg(), seed=s).run(N_D).stats for s in D_SEEDS]
print("[D] rule M same seeds: gini_%d %.3f gini_%d %.3f starved %.1f" % (
    at(100), np.mean([b.gini.iloc[at(100)] for b in base]), N_D, np.mean([b.gini.iloc[N_D] for b in base]),
    np.mean([b.starved.sum() for b in base])))
print(f"\nwall {time.time() - t0:.0f} s" + ("  (smoke run, not the docs numbers)" if args.quick else ""))
