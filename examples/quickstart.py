"""Quickstart: synthetic survey table -> personas -> simulated survey (FakeLLM) -> EvalFrame -> evidence card.

Offline, no API key, runs in a few seconds:

    python examples/quickstart.py [out_dir]

Swap `FakeLLM(...)` for `se.llm.Claude(model="claude-haiku-5-5", effort="low", cache_dir=".cache", budget_usd=2)`
to run it for real. FakeLLM answers are deterministic hashed noise, so the card SHOULD say "inconclusive": the
label-free persistence baseline (last year's orders) beats a method with no signal. That is the evidence
pipeline doing its job.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import simengine as se

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="simengine-quickstart-"))

# 1. A raw table: one row per person. `enjoys_online` is the target we want to predict (1-5 Likert).
rng = np.random.default_rng(0)
n = 100
df = pd.DataFrame({"uid": [f"u{i:03d}" for i in range(n)], "age": rng.integers(18, 75, n),
                   "gender": rng.choice(["female", "male"], n), "income_tier": rng.choice(["low", "middle", "affluent"], n),
                   "orders_last_year": rng.poisson(6, n)})
df["enjoys_online"] = np.clip(np.round(1.5 + df.orders_last_year / 3 + rng.normal(0, 0.8, n)), 1, 5)

# 2. Personas from data. The target goes in `forbidden`: a facet built from it raises (leakage guard).
#    income_tier uses the TALP enum (low..affluent): a value outside it would be a facet plus a warning.
panel = se.persona.from_table(df, id_col="uid", segment="income_tier", forbidden=["enjoys_online"])

# 3. Simulated survey with the no-persona and persona-swap control arms (on by default).
llm = se.FakeLLM(se.sim.schema_responder())
res = se.sim.survey(panel, [{"id": "q1", "text": "I enjoy shopping online.", "kind": "likert5"}], llm=llm, B=300)

# 4. Seal the predictions before the truth is joined (one seal API: se.seal = se.evidence.seal).
registry = out / ".onkayit"
rec = se.seal(se.sim.survey_frame(res), "quickstart predictions before truth", registry=registry)
print("sealed:", str(rec.get("sha256"))[:12])

# 5. EvalFrame: truth + pred_<method> per person, plus a label-free baseline the LLM must beat.
frame = se.sim.survey_frame(res, truth={"q1": dict(zip(df.uid, df.enjoys_online.astype(float)))})
frame = frame.dropna(axis=1, how="all")                       # SSR columns are empty when SSR is off
last = dict(zip(df.uid, df.orders_last_year))
frame["pred_persistence"] = [float(np.clip(1.5 + last[g] / 3, 1, 5)) for g in frame.group]

# 6. Evidence: baseline ladder, paired cluster-bootstrap CIs, audits, seal table, card, dashboard.
report = se.evidence.evaluate(frame, title="quickstart (FakeLLM)", ours="score", task="continuous",
                              unlabeled_baselines=["persistence"], B=300, prereg=registry)
print(se.evidence.summary(report)["headline"])
print(se.evidence.card(report, out / "card.md")[:600])
se.evidence.dashboard([report], out / "dashboard.html")
print(f"\nwritten: {out / 'card.md'}, {out / 'dashboard.html'}  ({len(llm.calls)} fake LLM calls)")
