"""A/B workflow: which of two texts wins? Offline (FakeLLM), synthetic pairs.

    python examples/ab_test.py

Measured guidance (docs/RESULTS.md): on 300 real Upworthy headline tests the TF-IDF + LLM stack (`ABStack`) was the
best arm, 0.817 [0.770, 0.860]; a Haiku persona panel was at chance, 0.522 [0.468, 0.575] (deney/RAPOR.md 1.1,
sahne/RAPOR.md). So: use the methods stack for the decision, and use the persona panel only for reasons/quotes,
with its confidence check.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # run from a checkout without pip install
import simengine as se
import simengine.methods as M

WORDS = ["river", "garden", "window", "letter", "bridge", "market", "winter", "silver", "forest", "engine"]
HOOK = ["shocking", "secret", "unbelievable"]


def pairs(n, seed):
    """Synthetic A/B pairs: A tends to win when it has a hook word and B does not (with noise); equal word counts."""
    rs = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        ha, hb = rs.random() < 0.5, rs.random() < 0.5
        mk = lambda h: " ".join(list(rs.choice(WORDS, 4 if h else 5)) + ([str(rs.choice(HOOK))] if h else []))
        rows.append({"text_a": mk(ha), "text_b": mk(hb), "y": int((ha - hb) + rs.normal(0, 0.5) > 0),
                     "group": f"t{seed}_{i}"})
    return pd.DataFrame(rows)


LINE = re.compile(r'^(\d+)\. A: "(.*?)" \| B: "(.*?)"$', re.M)


def fake_judge(system, user):
    """Stands in for an LLM that half-knows the rule (it misses 'secret')."""
    out = []
    for i, a, b in LINE.findall(user):
        ha, hb = any(h in a for h in HOOK[::2]), any(h in b for h in HOOK[::2])
        out.append({"i": int(i), "p_a": 0.5 + 0.3 * (ha - hb)})
    return json.dumps(out)


train, test = pairs(600, 0), pairs(200, 1)

# 1. The decision: baselines + LLM direct + stack, all predicting P(A wins).
llm_arm = M.LLMDirectAB(se.FakeLLM(fake_judge), context="news headlines, outcome = click-through rate")
arms = {"length": M.LengthAB().fit(train), "tfidf": M.TfidfAB().fit(train), "llm": llm_arm.fit(train),
        "stack": M.ABStack(llm_method=llm_arm).fit(train)}
frame = pd.DataFrame({"truth": test.y, "group": test.group})
for name, m in arms.items():
    frame[f"pred_{name}"] = m.predict(test)

report = se.evidence.evaluate(frame, title="A/B stack (synthetic)", ours="stack", raw="llm", task="binary",
                              unlabeled_baselines=["length"], B=300)
print(se.evidence.summary(report)["headline"])

# 2. The story: a small persona panel with controls and a confidence verdict (reasons, not the decision).
panel = [{"id": f"p{i}", "text": t, "segment": s} for i, (t, s) in enumerate([
    ("34, nurse, rents in a big city, price-sensitive", "young"), ("58, retired teacher, reads the news daily", "old"),
    ("27, software developer, commutes by bike", "young"), ("66, shop owner, careful with money", "old"),
    ("41, parent of two, shops online at night", "mid"), ("23, student, lives with flatmates", "young")])]
ab = se.sim.ab_test(panel, [{"id": "A", "text": "Free delivery over 300 TL"}, {"id": "B", "text": "15% off tonight"}],
                    llm=se.FakeLLM(se.sim.schema_responder()), context="Friday 18:00 push notification", B=300)
print("panel winner:", ab["winner"], "| low confidence:", ab["confidence"]["low"], ab["confidence"]["reasons"])
