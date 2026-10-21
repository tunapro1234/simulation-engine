"""Segment distribution workflow: ask the model directly for each segment's answer shares, then score it
against the distribution baselines with the evidence pipeline. Offline (FakeLLM), synthetic survey.

    python examples/segment_distribution.py

Measured guidance (docs/RESULTS.md, makro/RAPOR.md C1): on CES 2020 segments the direct segment distribution
(3 framings averaged, Haiku) reached W1 0.077 and closed 0.818 of the gap from the general marginal (0.206) to a
37-person real survey (0.050); a persona panel closed only 0.55. It lost to the general marginal on 2 of 8 items,
and memorisation is not ruled out: seal predictions before you open the truth.
"""
import tempfile

import numpy as np
import pandas as pd

import simengine as se
import simengine.methods as M

rng = np.random.default_rng(0)
OPTIONS = ["Much better", "Better", "Same", "Worse", "Much worse"]
segments = {"young_urban": "adults aged 18-34 living in large cities", "old_rural": "adults aged 55+ in rural areas",
            "mid_suburb": "adults aged 35-54 in suburbs", "students": "full-time university students"}
questions = [{"id": f"q{i}", "text": t, "options": OPTIONS} for i, t in enumerate(
    ["Over the next year, will your household finances get ...", "Will public transport in your area get ...",
     "Will the cost of housing for people like you get ...", "Will your job prospects get ..."])]

# Synthetic "real survey": 80 respondents per segment, each segment with its own tilt.
rows = []
for si, s in enumerate(segments):
    for q in questions:
        tilt = np.roll([0.30, 0.25, 0.20, 0.15, 0.10], si + int(q["id"][1:]) % 2)
        for a in rng.choice(OPTIONS, 80, p=tilt):
            rows.append({"segment": s, "qid": q["id"], "answer": a})
survey = pd.DataFrame(rows)
truth = {s: {q["id"]: M.weighted_distribution(survey[(survey.segment == s) & (survey.qid == q["id"])].answer,
                                              OPTIONS).tolist() for q in questions} for s in segments}

# 1. Seal the prediction BEFORE looking at the truth (pre-registration).
llm = se.FakeLLM(se.sim.schema_responder())          # se.llm.Claude(model="claude-haiku-5-5", ...) for real
pred = se.sim.segment_distribution(llm, segments, questions, population="adults in the country")
registry = tempfile.mkdtemp(prefix="simengine-seal-")
frame = se.sim.distribution_frame(pred, truth=None)
rec = se.evidence.seal(frame[["unit", "pred_llm_dist"]].astype(str), "segment distribution before truth",
                       registry=registry)
print("sealed:", str(rec.get("sha256"))[:12])

# 2. Truth + baselines: uniform, and the overall marginal copied to every segment (segment ignored). The overall
#    marginal comes from the same survey, so it is an upper-information baseline (like OpinionQA setting B).
frame = se.sim.distribution_frame(pred, truth=truth)
overall = {q["id"]: M.weighted_distribution(survey[survey.qid == q["id"]].answer, OPTIONS) for q in questions}
frame["pred_uniform"] = [M.uniform_distribution(len(OPTIONS)).tolist() for _ in range(len(frame))]
frame["pred_overall"] = [overall[i].tolist() for i in frame["item"]]

report = se.evidence.evaluate(frame, title="segment distribution (FakeLLM)", ours="llm_dist", task="distribution",
                              unlabeled_baselines=["uniform"], prereg=registry, B=300)
print(se.evidence.summary(report)["headline"])
print("calls:", pred["calls"], "invalid replies:", pred["invalid"])
