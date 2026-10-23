"""Grid and spatial ABMs in one pass: Schelling tipping, Sugarscape inequality, mobility baselines, billboard
placement on a temporal holdout and a spatial SEIR against its ODE (plus the generative-agent variant with a rule
FakeLLM). Offline, synthetic data only, small sizes so it runs in well under 30 s.

    python examples/abm_new_demo.py

The numbers printed here are small-scale smoke values on synthetic data, not results. Measured numbers and their
provenance are in docs/modules/abm.md and the unit docs under docs/modules/abm/.
"""
import numpy as np

import simengine.abm as abm
from simengine.abm import placement as pl
from simengine.abm import spatial_sir as ep
from simengine.llm import FakeLLM

# --- 1. Schelling on the grid core: a mild demand (1/3 alike) still segregates
g = abm.Grid((30, 30), torus=True)
lat0 = abm.random_lattice(g, [0.45, 0.45], seed=0)
print(f"random lattice: like share {abm.same_type_share(lat0, g):.3f}, "
      f"Moran's I {abm.morans_i((lat0 == 0).astype(float), g, mask=lat0 >= 0):+.3f}")
for thr in (0.2, 1 / 3, 0.5):
    run = abm.Schelling(g, threshold=thr).run(n_steps=60, seed=0)
    last = run.history.iloc[-1]
    print(f"schelling threshold {thr:.2f}: like share {last.same_share:.3f}, happy {last.happy:.2f}, "
          f"steps {run.steps}, converged {run.converged}")
mover = abm.LLMMover(FakeLLM(abm.threshold_responder(1 / 3)), labels=("blue", "orange"))
run = abm.Schelling(g, threshold=1 / 3, mover=mover).run(n_steps=30, seed=0)
print(f"schelling via FakeLLM rule: like share {run.history.iloc[-1].same_share:.3f}, calls {run.llm['calls']}")

# --- 2. Sugarscape: near-uniform endowments, skewed wealth after a few dozen steps
sug = abm.Sugarscape(abm.SugarscapeConfig(size=30, n_agents=120), seed=0).run(60)
print(f"sugarscape gini t=0 {sug.stats.gini.iloc[0]:.3f} -> t=60 {sug.stats.gini.iloc[-1]:.3f}")

# --- 3. Mobility: per-user slot mode vs Markov vs EPR on synthetic trajectories (HuMob metrics)
traj = abm.toy_trajectories(n_users=60, n_days=14, seed=0)
df = traj.rename(columns={"user": "uid", "day": "d", "slot": "t"})
train, test = df[df.d < 10], df[df.d >= 10]
q = test[["uid", "d", "t"]]
preds = abm.mobility_baselines(train, q, seed=0)
for name in ("slot_mode", "markov_argmax", "epr", "global_slot_mode"):
    s = abm.user_scores(q.uid, q.d, q.t, abm.from_cell(preds[name]), test[["x", "y"]].to_numpy())
    print(f"mobility {name:16s} GEO-BLEU {s.geobleu.mean():.3f}  DTW {s.dtw.mean():6.2f} km")

# --- 4. Placement: lazy greedy coverage vs top-visited, scored on unseen days
grid = pl.CellGrid.fit(traj, size=2)
ex = abm.exposures(traj, grid, days=range(10))
p = abm.greedy_placement(ex.M, k=5)
print(f"placement k=5: train reach {p.value:.0f} users, online bound {p.bound:.1f}")
h = pl.holdout(traj, grid, train_days=range(10), test_days=range(10, 14), ks=(5,), random_seeds=3, n_boot=200)
for _, row in h.iterrows():
    print(f"  holdout {row.method:12s} test reach share {row.test_reach_share:.3f}")

# --- 5. Spatial SEIR: well mixed tracks the ODE, local contact (radius 3) flattens and delays it
par = abm.EpiParams.from_r0(2.5, infectious_days=5, latent_days=3)
ode = abm.seir_ode(par, n=2_000, i0=10, days=120)
for radius in (None, 3):
    r = abm.SpatialSEIR(par, shape=(45, 45), n=2_000, radius=radius, steps_per_day=2).run(days=120, i0=10, reps=5,
                                                                                         seed=1)
    a = abm.ode_agreement(r, ode)
    print(f"seir radius {radius}: attack {a['attack_rate_abm']:.3f} (ODE {a['attack_rate_ode']:.3f}), "
          f"peak day {a['peak_day_abm']:.1f} (ODE {a['peak_day_ode']:.0f})")
print(f"final-size formula for R0 2.5: {abm.final_size(2.5):.3f}")

# --- 6. Generative-agent epidemic with a rule FakeLLM (checks the loop, not LLM behaviour)
for cond, rule in (("base", ep.rule_responder(stay_on=())), ("self_health", ep.rule_responder())):
    res = abm.GenerativeEpidemic(FakeLLM(rule), n=60, condition=cond, seed=0).run(days=40)
    d = res.daily
    print(f"generative {cond:11s}: attack {1 - d.S.iloc[-1] / 60:.2f}, mean out share {d.out_share.mean():.2f}, "
          f"calls {res.calls}, invalid {res.invalid}")
print("done")
