"""Sugarscape measurements quoted in docs/modules/abm/sugarscape.md (offline, ~17 min on one busy core).

    PYTHONPATH=. python examples/sugarscape_measure.py [capacity_grid.txt]

The optional argument is a whitespace capacity grid (e.g. the NetLogo transcription of the book map) read at runtime.
"""
import hashlib, json, re, sys, time
import numpy as np, pandas as pd
from simengine.abm.sugarscape import (Sugarscape, SugarscapeConfig, LLMPolicy, gini_trajectory, load_capacity,
                                      sugar_capacity, gini)
from simengine.llm.fake import FakeLLM

pd.set_option("display.width", 200)
SEEDS = range(10)
AT = (0, 10, 25, 50, 100, 150, 200, 300, 400, 500)
alt_map = sys.argv[1] if len(sys.argv) > 1 else None
t0 = time.time()
U = np.arange(5, 26)
print("Gini of U{5..25} (exact, population):", round(gini(np.repeat(U, 1000)), 4))

def traj(cfg, label):
    d = gini_trajectory(seeds=SEEDS, n_steps=500, config=cfg, at=AT)
    g = d.groupby("step").gini.agg(["mean", "std"]).round(3)
    print(f"\n[{label}] Gini by step, mean / sd over {len(SEEDS)} seeds\n", g.T.to_string())
    return d

traj(SugarscapeConfig(), "A wealth: 250 agents, G1, M, R[60,100], parametric map")
if alt_map:
    traj(SugarscapeConfig(capacity=load_capacity(alt_map)), "A' same, transcribed map read at runtime")

# end-of-run shape
rows = []
for s in SEEDS:
    st = Sugarscape(SugarscapeConfig(), seed=s).run(500).stats
    late = st.iloc[200:]
    rows.append(dict(seed=s, gini_200_500=late.gini.mean(), skew_500=st.skewness.iloc[-1],
                     top10_500=st.top10_share.iloc[-1], skew_0=st.skewness.iloc[0], top10_0=st.top10_share.iloc[0]))
r = pd.DataFrame(rows)
print("\n[A] steps 200-500 mean Gini (per-seed mean of the window), skewness and top-10% share\n",
      r.drop(columns="seed").agg(["mean", "std"]).round(3).to_string())

# carrying capacity, no ageing
cc = []
for s in SEEDS:
    st = Sugarscape(SugarscapeConfig(n_agents=400, max_age=None, replace=False), seed=s).run(500).stats
    cc.append(dict(pop_100=st.population.iloc[100], pop_500=st.population.iloc[500], gini_500=st.gini.iloc[500],
                   vis_0=st.mean_vision.iloc[0], vis_500=st.mean_vision.iloc[500],
                   met_0=st.mean_metabolism.iloc[0], met_500=st.mean_metabolism.iloc[500]))
print("\n[B] carrying capacity: 400 agents, G1, M, no ageing / replacement\n",
      pd.DataFrame(cc).agg(["mean", "std"]).round(3).to_string())

# spice with / without trade
for trade in (False, True):
    cc = []
    for s in SEEDS:
        st = Sugarscape(SugarscapeConfig(n_agents=400, max_age=None, replace=False, spice=True, trade=trade),
                        seed=s).run(500).stats
        row = dict(pop_500=st.population.iloc[500], gini_500=st.gini.iloc[500])
        if trade:
            row.update(trades_500=st.trades.iloc[500], plogsd_10=st.price_log_sd.iloc[10],
                       plogsd_100=st.price_log_sd.iloc[100], plogsd_500=st.price_log_sd.iloc[500],
                       pgeo_500=st.price_geo_mean.iloc[500])
        cc.append(row)
    print(f"\n[C] sugar+spice carrying capacity, 400 agents, no ageing, trade={trade}\n",
          pd.DataFrame(cc).agg(["mean", "std"]).round(3).to_string())
    g = gini_trajectory(seeds=SEEDS, n_steps=500, config=SugarscapeConfig(spice=True, trade=trade), at=AT)
    print(f"[C] Gini (sugar + spice holdings), 250 agents, R[60,100], trade={trade}\n",
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
    for s in range(5):
        pol = LLMPolicy(FakeLLM(resp))
        run = Sugarscape(SugarscapeConfig(), seed=s, policy=pol).run(300)
        gs.append(dict(gini_100=run.stats.gini.iloc[100], gini_300=run.stats.gini.iloc[300],
                       starved=run.stats.starved.sum(), calls=run.calls, invalid=run.invalid,
                       conflicts=run.conflicts))
    print(f"\n[D] LLMPolicy + FakeLLM {name}, 250 agents, 300 steps, 5 seeds\n",
          pd.DataFrame(gs).agg(["mean", "std"]).round(3).to_string())
base = [Sugarscape(SugarscapeConfig(), seed=s).run(300).stats for s in range(5)]
print("[D] rule M same seeds: gini_100 %.3f gini_300 %.3f starved %.1f" % (
    np.mean([b.gini.iloc[100] for b in base]), np.mean([b.gini.iloc[300] for b in base]),
    np.mean([b.starved.sum() for b in base])))
print(f"\nwall {time.time() - t0:.0f} s")
