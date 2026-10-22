"""Recipe: survey answer distributions by segment. How would each segment answer these questions, and what is the
population-level share? The LLM states each segment's answer shares directly (verbalised distribution, 3 framings
averaged); a small real survey gives the baselines; a held-out survey is the truth; a prediction-powered (PPI++)
estimate combines both for the population; the evidence card decides.

    python3 examples/recipes/segment_survey.py                        # offline: FakeLLM + synthetic survey, ~10 s
    python3 examples/recipes/segment_survey.py --data data/survey.csv --out runs/survey
    python3 examples/recipes/segment_survey.py --live --max-usd 1     # Claude Haiku, key from $ANTHROPIC_API_KEY
    python3 examples/recipes/segment_survey.py --live --key-file <path> --max-usd 1

--data CSV, one row per (respondent, question answered):
    respondent_id  str  person id (bootstrap cluster; one respondent is never split across train and holdout)
    segment        str  segment key, e.g. young_urban
    segment_desc   str  segment description fed to the LLM (first value per segment; NO answer info in it)
    question_id    str  e.g. q1
    question_text  str  the question as asked (first value per question)
    options        str  answer options in their natural order, "|"-separated (first value per question)
    answer         str  the option the respondent chose (one of options)
    split          str  optional train / holdout; else 30 % of respondents (by id hash) are the small train survey
Guide, pitfalls and the literature behind each step: docs/recipes/segment_survey.md.
"""
import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # run from a checkout without pip install
import simengine as se                                              # noqa: E402

try:                                                                # PPI++ helper (optional; step 9 skips without it)
    from simengine.persona.inject.population import ppi_distribution  # noqa: E402
except ImportError:
    ppi_distribution = None

# 0. Day-of knobs: POPULATION goes into every prompt; segments and questions come from the brief. K_PRIOR = how many
#    real answers the LLM prior is worth in "ours" (fix it BEFORE looking at the holdout; 20 is a neutral start).
POPULATION, TRAIN_SHARE, K_PRIOR = "adults in the country", 0.3, 20
LIKERT = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
SEGMENTS = {"young_urban": "adults aged 18-34 living in large cities", "mid_suburb": "adults aged 35-54 in suburbs",
            "old_rural": "adults aged 55 and over in rural areas", "students": "full-time university students"}
QUESTIONS = {"q1": "I would pay a monthly fee for faster home delivery.", "q2": "I trust online reviews when buying "
             "electronics.", "q3": "I would switch to a cheaper mobile plan even if coverage is worse.",
             "q4": "I prefer paying in cash over cards or phone apps.", "q5": "I am worried about my household "
             "budget for the next year."}
LEAN = {"young_urban": [0.5, 0.6, 0.1, -0.7, 0.0], "mid_suburb": [0.1, 0.1, -0.4, -0.2, 0.1],  # -1 .. +1 = agree
        "old_rural": [-0.6, -0.5, -0.1, 0.7, 0.6], "students": [0.0, 0.4, 0.8, -0.5, 0.7]}
SIZES = {"young_urban": 300, "mid_suburb": 220, "old_rural": 120, "students": 60}       # imbalanced on purpose


def shape(k, lean):
    """Unimodal answer shares over k ordered options, centre moved by lean in [-1, 1]."""
    p = np.exp(-(np.arange(k) - (k - 1) / 2 * (1 + lean)) ** 2 / (2 * (k / 4) ** 2))
    return p / p.sum()


def synthetic(seed=0):
    rs = np.random.default_rng(seed)
    return pd.DataFrame([{"respondent_id": f"{s}-{i:04d}", "segment": s, "segment_desc": SEGMENTS[s], "question_id": q,
                          "question_text": text, "options": "|".join(LIKERT),
                          "answer": rs.choice(LIKERT, p=shape(5, LEAN[s][j]))}
                         for s, n in SIZES.items() for i in range(n) for j, (q, text) in enumerate(QUESTIONS.items())])


def fake_llm(system, user):
    """Stands in for an LLM that half-knows the segments (it misses the budget worry, q5, entirely)."""
    seg = next((s for s, d in SEGMENTS.items() if d in user), None)
    j = next((j for j, t in enumerate(QUESTIONS.values()) if t in user), None)
    lean = 0.7 * LEAN[seg][j] if seg and j is not None and j != 4 else 0.0
    lean += (zlib.crc32(user.encode()) % 100 - 50) / 250                      # stable per-prompt noise
    k = len(re.findall(r"^\d+\. ", user, re.M)) or 5
    return json.dumps({str(i + 1): round(100 * p, 1) for i, p in enumerate(shape(k, float(np.clip(lean, -1, 1))))})


def shares(d, q):                                                    # answer shares in option order (opts: step 1)
    return (d.answer.value_counts().reindex(opts[q], fill_value=0) / max(len(d), 1)).to_numpy(float)


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--data", help="your survey CSV (columns in the module docstring)")
ap.add_argument("--live", action="store_true", help="use the Claude backend instead of FakeLLM")
ap.add_argument("--key-file", help="file holding the API key (else $ANTHROPIC_API_KEY)")
ap.add_argument("--model", default="claude-haiku-5-5")
ap.add_argument("--max-usd", type=float, default=None, help="spend cap for --live (required with --live)")
ap.add_argument("--out", default="runs/segment_survey", help="card, dashboard, seals, cache and ledger go here")
ap.add_argument("--B", type=int, default=300, help="bootstrap replicates (1000 for the final card)")
a = ap.parse_args()
(out := Path(a.out)).mkdir(parents=True, exist_ok=True)

# 1. Data: your CSV or the synthetic survey. Small train survey = baselines + PPI labels; holdout = truth.
df = pd.read_csv(a.data, dtype=str) if a.data else synthetic()
need = {"respondent_id", "segment", "segment_desc", "question_id", "question_text", "options", "answer"}
if need - set(df.columns):
    sys.exit(f"--data is missing columns: {sorted(need - set(df.columns))}")
if "split" not in df:                                                # respondent-disjoint split, stable by id
    df["split"] = ["train" if zlib.crc32(str(r).encode()) % 100 < 100 * TRAIN_SHARE else "holdout"
                   for r in df.respondent_id]
qs = [{"id": q, "text": g.question_text.iloc[0], "options": g.options.iloc[0].split("|")}
      for q, g in df.groupby("question_id", sort=False)]
opts = {q["id"]: q["options"] for q in qs}
if bad := df[[x not in opts[q] for q, x in zip(df.question_id, df.answer)]].shape[0]:
    sys.exit(f"{bad} answers are not among their question's options (check spelling / the '|' separator)")
segs = df.groupby("segment", sort=False).segment_desc.first().to_dict()
train, hold = df[df.split == "train"], df[df.split != "train"]

# 2. Backend: FakeLLM offline; --live = Claude with a hard budget (BudgetExceeded stops new calls).
if a.live and a.max_usd is None:
    sys.exit("--live needs --max-usd (e.g. --max-usd 1); the ledger stops new calls at that total")
if a.live:
    key = Path(a.key_file).expanduser().read_text().strip() if a.key_file else None   # never printed
    llm = se.llm.Claude(model=a.model, effort="low", thinking="off", max_tokens=300, cache_dir=out / ".cache",
                        budget_usd=a.max_usd, api_key=key, tag="segment_survey")
else:
    llm = se.FakeLLM(fake_llm)

# 3. Ask: verbalised shares per segment x question (3 framings averaged) + the no-segment control (population only).
try:
    pred = se.sim.segment_distribution(segs, qs, llm=llm, population=POPULATION)
    ctrl = se.sim.segment_distribution({"all": POPULATION}, qs, llm=llm, population=POPULATION)
except se.llm.BudgetExceeded as e:
    sys.exit(f"stopped by the budget: {e}")
print(f"LLM calls: {pred['calls']} + {ctrl['calls']} control | invalid replies (filled uniform): {pred['invalid']}")

# 4. Baselines from the TRAIN survey only: overall marginal (segment ignored), segment shares shrunk toward it.
arms = {m: {} for m in ("uniform", "train_overall", "train_segment", "llm_nosegment", "llm_dist", "llm_prior")}
for q in opts:
    tq, k = train[train.question_id == q], len(opts[q])
    overall = shares(tq, q) if len(tq) else np.full(k, 1 / k)
    for s in segs:
        ts, llm_v = tq[tq.segment == s], np.asarray(pred["dist"][s][q])
        arms["uniform"][s, q], arms["train_overall"][s, q] = np.full(k, 1 / k), overall
        arms["train_segment"][s, q] = (shares(ts, q) * len(ts) + 5 * overall) / (len(ts) + 5)
        arms["llm_nosegment"][s, q], arms["llm_dist"][s, q] = np.asarray(ctrl["dist"]["all"][q]), llm_v
        # 5. Ours: the LLM as a Dirichlet prior worth K_PRIOR answers, updated with the segment's train answers.
        arms["llm_prior"][s, q] = (K_PRIOR * llm_v + shares(ts, q) * len(ts)) / (K_PRIOR + len(ts))
cells = [(s, q) for s in segs for q in opts]
frame = pd.DataFrame({"unit": [f"{s}|{q}" for s, q in cells], "segment": [s for s, _ in cells],
                      "item": [q for _, q in cells], "group": [q for _, q in cells],
                      **{f"pred_{m}": [arms[m][c].tolist() for c in cells] for m in arms if m != "uniform"}})

# 6. Seal every arm BEFORE the holdout truth is joined (pre-registration; the card shows the seal).
registry = out / ".onkayit"
se.seal(frame.astype(str), "segment_survey predictions; ours = llm_prior, raw = llm_dist", registry=registry)
hc = {c: g for c, g in hold.groupby(["segment", "question_id"])}  # a cell with no holdout answers gets no truth
frame["truth"] = [shares(hc[c], c[1]).tolist() if c in hc else None for c in cells]
frame = frame[frame.truth.notna()]

# 7. Evidence card: 1-TVD ladder, no-segment control, shuffle; CIs over questions (few -> warning; see step 8).
report = se.evidence.evaluate(frame, title="segment_survey distributions", ours="llm_prior", raw="llm_dist",
                              task="distribution", baselines_oof=["train_overall", "train_segment"], B=a.B,
                              prereg=registry)
s = se.evidence.summary(report)
print("\n" + str(s["headline"]))
for r in s["ladder"]:
    print(f"  {str(r['label'] or r['column']):40} {r['value']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
print("badge:", s["badge"], "| audits with warnings:", [d["check"] for d in s["audits"] if d["status"] == "warning"])

# 8. W1 (ordinal options) with a respondent bootstrap: CIs generalise to new people from the same segments.
wide = hold.pivot_table(index=["respondent_id", "segment"], columns="question_id", values="answer",
                        aggfunc="first").reset_index()
w1 = se.evidence.evaluate_distribution_persons(
    wide, ["segment"], list(opts), {m: arms[m] for m in arms}, metric="w1", id_col="respondent_id", options=opts,
    B=a.B, pairs=[("llm_prior", "train_segment"), ("llm_dist", "llm_nosegment"), ("llm_dist", "train_overall")])
print("\nW1 by arm (lower is better), respondent bootstrap 95% CI:")
for m, r in w1["arms"].items():
    print(f"  {m:16} {r['point']:.3f} [{r['ci_basic'][0]:.3f}, {r['ci_basic'][1]:.3f}]")   # basic CI: small-cell bias
for d in w1["diffs"]:
    print(f"  {d['a']} - {d['b']}: {d['point']:+.3f} [{d['ci_basic'][0]:+.3f}, {d['ci_basic'][1]:+.3f}]")
bad = [f"{c['segment']}/{c['item']}" for c in w1["cells"] if c["llm_dist"] > c["train_overall"]]
print("cells where the raw LLM is worse than the train marginal (show these):", bad or "none")

# 9. Population share per option: PPI++ = train answers + LLM shares for everyone (here the holdout plays the
#    unlabelled population; on the day use census segment counts). Raw LLM only: llm_prior has seen train labels.
if ppi_distribution is None:
    print("\nPPI++ skipped: simengine.persona.inject.population.ppi_distribution not available")
else:
    rows = []
    print("\npopulation share, top box, 95% CI:")
    for q in opts:
        tq, hq = train[train.question_id == q], hold[hold.question_id == q]
        f = lambda d: [dict(zip(opts[q], arms["llm_dist"][sg, q])) for sg in d.segment]
        res, true = ppi_distribution(tq.answer, f(tq), f(hq), opts[q]), shares(hq, q)
        rows += list(zip(res.values(), true))
        r = res[opts[q][-1]]                                                 # top box (last option) per question
        print(f"{q} '{opts[q][-1]}': holdout {true[-1]:.3f} | PPI++ {r['estimate']:.3f} [{r['lo']:.3f}, "
              f"{r['hi']:.3f}] | train only {r['classical']:.3f} [{r['classical_lo']:.3f}, {r['classical_hi']:.3f}]")
    print(f"PPI++ CI covers the holdout share in {sum(r['lo'] <= t <= r['hi'] for r, t in rows)}/{len(rows)} options "
          f"(train only {sum(r['classical_lo'] <= t <= r['classical_hi'] for r, t in rows)}/{len(rows)}); median CI "
          f"width ratio PPI/classical {np.median([r['width_ratio'] or 1.0 for r, _ in rows]):.2f}")

# 10. Write the card + dashboard; spend line for --live.
se.evidence.card(report, out / "card.md", lang="en")
se.evidence.dashboard([report], out / "dashboard.html")
cost = f" | spend ${llm.ledger.total():.4f} of ${a.max_usd}" if a.live else ""
print(f"written: {out / 'card.md'}, {out / 'dashboard.html'}{cost}")
