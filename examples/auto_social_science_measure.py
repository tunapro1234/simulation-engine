"""Measured numbers of docs/modules/research/auto_social_science.md (no LLM calls, ~10 s).

    SIMENGINE_DATA=<data-root> PYTHONPATH=. python examples/auto_social_science_measure.py
Sections 1-3 are synthetic; section 4 reads <data-root>/socsci210/ham/cond_stats.parquet (skipped if absent).
"""
import json, sys, time
import numpy as np, pandas as pd
from simengine.research import hypotheses as H
from simengine.llm.fake import FakeLLM

out = {}
# 1. coefficient recovery + CI coverage on a ground-truth linear SCM
h = H.Hypothesis("s", ["a"], [H.Variable("x", "continuous", [0, 1, 2, 3, 4]), H.Variable("g", "nominal", ["r", "s", "t"])],
                 H.Variable("y", question="?"))
cover, err, diff = [], [], 0.0
for seed in range(500):
    d = H.design(h, replicates=4)
    d["y"] = H.linear_world({"x": 0.3}, 1.0, noise=1.0, seed=seed, levels={"g": {"s": 0.5, "t": -0.2}})(h, d)
    p = H.estimate(h, d, backend="numpy")[0].set_index("term")
    q = H.estimate(h, d, backend="statsmodels")[0].set_index("term")
    diff = max(diff, float(np.abs(p[["coef", "se", "p"]].to_numpy() - q[["coef", "se", "p"]].to_numpy()).max()))
    truth = {"x": 0.3, "g[s]": 0.5, "g[t]": -0.2}
    cover += [(p.lo[t] <= v <= p.hi[t]) for t, v in truth.items()]
    err += [p.coef[t] - v for t, v in truth.items()]
out["recovery"] = {"seeds": 500, "rows": 60, "ci95_coverage": float(np.mean(cover)), "mean_bias": float(np.mean(err)),
                   "max_abs_numpy_vs_statsmodels": diff}

# 2. market_world (abm.market_sim, prior midpoint) fitted paths
hm = H.Hypothesis("shop", ["shopper"], [H.Variable("price_cut", "continuous", [0.0, 0.1, 0.2, 0.3]),
                                         H.Variable("promo", "binary", [0, 1])], H.Variable("sales", question="?"))
dm = H.design(hm)
t0 = time.time()
dm["y"] = H.market_world(weeks=4, n_agents=400)(hm, dm)
pm, im = H.estimate(hm, dm)
out["market_world"] = {"cells": len(dm), "seconds": round(time.time() - t0, 3), "r2": im["r2"],
                       **{t: round(float(c), 4) for t, c in zip(pm.term, pm.coef)}}

# 3. FakeLLM conversation pipeline cost (calls) on the 6-cell bargaining design of the tests
from tests.test_research_auto_social_science import bargain_responder
llm = FakeLLM(bargain_responder)
t0 = time.time()
res = H.AutoSocialScience(llm, "bargain", simulator="conversation", replicates=1,
                          simulator_kw={"max_statements": 20}).run()
out["conversation_fake"] = {"llm_calls": len(llm.calls), "seconds": round(time.time() - t0, 3),
                            **res.runs[0].summary}

# 4. SocSci210 condition table: reliability and non-LLM baselines under the Hewitt protocol
from pathlib import Path
from simengine.bench.base import DataNotFound, resolve_root
try:
    path = Path(resolve_root(None, "socsci210")) / "ham" / "cond_stats.parquet"
except DataNotFound:
    path = None
if path is None or not path.exists():
    print(json.dumps(out, indent=1, default=float)); sys.exit(0)
c = pd.read_parquet(path, columns=["study_id", "task_num", "condition_num", "n", "mean", "sd", "stimuli"])
c["exp"] = c.study_id + ":" + c.task_num.astype(str)
g = c.groupby("exp")
pooled = np.sqrt(g.apply(lambda x: np.average(x.sd ** 2, weights=x.n), include_groups=False))
c["z"] = c["mean"] / c.exp.map(pooled)                     # standardised: contrasts become Cohen's d
c["z_se"] = 1 / np.sqrt(c.n)
c = c[np.isfinite(c.z)]
rng = np.random.default_rng(0)
base = dict(study="exp", condition="condition_num", true="z", true_se="z_se", repeats=16, seed=0)
out["socsci210_rows"] = {"conditions": int(len(c)), "experiments": int(c.exp.nunique()), "studies": int(c.study_id.nunique())}
out["socsci210_random"] = H.score_condition_forecasts(c.assign(pred_mean=rng.standard_normal(len(c))), **base)
# TF-IDF ridge on the stimulus text, target = standardised deviation from the experiment mean, study-grouped 5-fold OOF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
c = c.reset_index(drop=True)
dev = (c.z - c.groupby("exp").z.transform("mean")).to_numpy()
oof = np.zeros(len(c))
for tr, te in GroupKFold(5).split(c, groups=c.study_id):
    vec = TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True).fit(c.stimuli.iloc[tr])
    X = vec.transform(c.stimuli)
    m = Ridge(alpha=1.0).fit(X[tr], dev[tr])
    oof[te] = m.predict(X[te])
out["socsci210_tfidf_ridge_oof"] = H.score_condition_forecasts(c.assign(pred_mean=oof), **base)
# length-difference baseline: longer stimulus -> different answer? (sign-agnostic check)
out["socsci210_stimulus_length"] = H.score_condition_forecasts(c.assign(pred_mean=c.stimuli.str.len().astype(float)), **base)
print(json.dumps(out, indent=1, default=float))
