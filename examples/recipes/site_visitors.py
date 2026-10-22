"""Recipe: simulated visitors on a landing page / checkout funnel. Which page variant converts better, and where
do visitors drop off? Personas walk through text descriptions of each variant's pages; the LLM predicts
P(continue) per step; past traffic gives the baselines; the evidence card decides.

    python3 examples/recipes/site_visitors.py                       # offline: FakeLLM + synthetic funnel
    python3 examples/recipes/site_visitors.py --data data/funnel.csv --out runs/funnel
    python3 examples/recipes/site_visitors.py --live --max-usd 1    # Claude Haiku; key: $ANTHROPIC_API_KEY or --key-file

--data CSV, one row per (visitor, funnel step the visitor REACHED):
    visitor_id  str   person id (bootstrap cluster; one visitor is never split across train and test)
    segment     str   visitor segment, e.g. mobile_new
    persona     str   segment description fed to the LLM (first value per segment is used; NO outcome info in it)
    variant     str   page variant id, e.g. A / B
    step        str   funnel step name, e.g. landing / product / cart / checkout
    step_order  int   1, 2, ... (order of the step in the funnel)
    page_text   str   text description of that variant's page at that step
    continued   0/1   1 = clicked through to the next step (at the last step: purchased); 0 = dropped off here
    date        str   optional ISO date; when present the last 30 % of dates are the test split (else 30 % of visitors)
Guide, pitfalls and the literature behind each step: docs/recipes/site_visitors.md.
"""
import argparse
import functools
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # run from a checkout without pip install
import simengine as se                                              # noqa: E402

# 0. Day-of knobs: page texts, segments and the system prompt are what you replace with the brief's own.
PAGES = {  # variant -> step -> what the visitor sees (write these from the client's real pages / screenshots)
    "A": {"landing": "Hero banner with a 20% off code and a big autoplay video", "product": "Price shown; shipping "
          "cost appears only at checkout", "cart": "Cart with an upsell carousel", "checkout": "Checkout requires "
          "you to create an account before paying"},
    "B": {"landing": "Hero banner with a 20% off code and a static photo", "product": "Price shown with 'free "
          "shipping over 300 TL' next to it", "cart": "Cart with an upsell carousel", "checkout": "Guest checkout: "
          "pay with an email address, no account needed"}}
SEGMENTS = {"mobile_new": "first visit, on a phone, came from a social ad, price-sensitive, in a hurry",
            "desktop_new": "first visit, on a laptop, came from a search ad, compares prices across shops",
            "mobile_returning": "returning customer on a phone, has bought before, browses in the evening",
            "desktop_returning": "returning customer on a laptop, has bought before, card saved in the browser"}
SYSTEM = ("You simulate one website visitor. Read the visitor and the page, then estimate the probability that this "
          "visitor continues to the next step. Reply with JSON only: {\"p_continue\": <0-1>, \"reason\": \"...\"}")
NO_PERSONA = "a typical visitor to this shop"                        # control arm: same pages, no persona


def synthetic(n=1600, seed=0):
    """Ground-truth funnel: returning > new; B's free shipping helps price-sensitive visitors, guest checkout helps
    new ones, and A's autoplay video hurts on phones (the fake LLM below does not know the last one)."""
    rs, rows, steps = np.random.default_rng(seed), [], list(PAGES["A"])
    base = {"landing": 0.2, "product": -0.2, "cart": 0.5, "checkout": 0.7}
    for i in range(n):
        seg, var, day = rs.choice(list(SEGMENTS)), rs.choice(list(PAGES)), int(rs.integers(1, 15))
        for k, step in enumerate(steps, 1):
            z = base[step] + (0.6 if "returning" in seg else -0.4)
            z += {"landing": -0.5 * (var == "A" and "mobile" in seg), "product": 0.4 * (var == "B" and seg == "mobile_new"),
                  "checkout": (0.9 if "new" in seg else 0.1) * (var == "B")}.get(step, 0.0)
            go = int(rs.random() < 1 / (1 + np.exp(-z)))
            rows.append({"visitor_id": f"v{i:05d}", "segment": seg, "persona": SEGMENTS[seg], "variant": var,
                         "step": step, "step_order": k, "page_text": PAGES[var][step], "continued": go,
                         "date": f"2026-10-{day:02d}"})
            if not go:
                break
    return pd.DataFrame(rows)


def fake_visitor(system, user):
    """Stands in for an LLM that half-knows the funnel (it misses the autoplay video on phones)."""
    p = 0.55 + 0.15 * ("returning" in user) - 0.10 * ("first visit" in user)
    p += 0.15 * ("guest checkout" in user.lower()) * ("first visit" in user) - 0.10 * ("create an account" in user)
    p += 0.10 * ("free shipping" in user) * ("price-sensitive" in user)
    p += (zlib.crc32(user.encode()) % 100 - 50) / 1000                       # stable per-prompt noise
    return json.dumps({"p_continue": round(float(np.clip(p, 0.02, 0.98)), 3), "reason": "fake"})


def p_of(r):
    j = None if isinstance(r, BaseException) else se.sim.parse_json(r.text)
    return float(np.clip(j.get("p_continue", 0.5), 0.01, 0.99)) if isinstance(j, dict) else 0.5   # unparsed -> 0.5


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--data", help="your funnel CSV (columns in the module docstring)")
ap.add_argument("--live", action="store_true", help="use the Claude backend instead of FakeLLM")
ap.add_argument("--key-file", help="file holding the API key (else $ANTHROPIC_API_KEY); never printed")
ap.add_argument("--model", default="claude-haiku-5-5")
ap.add_argument("--max-usd", type=float, default=None, help="spend cap for --live (required with --live)")
ap.add_argument("--out", default="runs/site_visitors", help="card, dashboard, seals, cache and ledger go here")
ap.add_argument("--B", type=int, default=300, help="bootstrap replicates (1000 for the final card)")
a = ap.parse_args()
(out := Path(a.out)).mkdir(parents=True, exist_ok=True)

# 1. Data: your CSV or the synthetic funnel. Change the split rule here if the brief fixes one.
df = pd.read_csv(a.data) if a.data else synthetic()
need = {"visitor_id", "segment", "persona", "variant", "step", "step_order", "page_text", "continued"}
if need - set(df.columns):
    sys.exit(f"--data is missing columns: {sorted(need - set(df.columns))}")
df = df.sort_values(["visitor_id", "step_order"]).reset_index(drop=True)
n_steps = int(df.step_order.max())
if "date" in df:                                                     # time split: the test days come last
    t = pd.to_datetime(df.date).astype("int64")
    df["split"] = np.where(t > np.quantile(t, 0.7), "test", "train")
else:                                                                # else visitor-disjoint split
    df["split"] = ["test" if zlib.crc32(str(v).encode()) % 10 < 3 else "train" for v in df.visitor_id]
train = df[df.split == "train"]

# 2. Backend: FakeLLM offline; --live = Claude with a hard budget (BudgetExceeded stops new calls).
if a.live:
    key = Path(a.key_file).expanduser().read_text().strip() if a.key_file else os.environ.get("ANTHROPIC_API_KEY")
    if a.max_usd is None or not key:                                 # the ledger stops new calls at --max-usd
        sys.exit("--live needs --max-usd (e.g. --max-usd 1) and $ANTHROPIC_API_KEY or --key-file <path>")
    llm = se.llm.Claude(model=a.model, effort="low", thinking="off", max_tokens=200, cache_dir=out / ".cache",
                        budget_usd=a.max_usd, api_key=key, tag="site_visitors")
else:
    llm = se.FakeLLM(fake_visitor)

# 3. Ask: one prompt per (segment, variant, step) cell, plus the no-persona control per (variant, step).
personas = df.groupby("segment").persona.first().to_dict()
cells = df.drop_duplicates(["segment", "variant", "step"])[["segment", "variant", "step", "page_text"]].copy()
ctrl = cells.drop_duplicates(["variant", "step"]).copy()
ask = lambda who, r: (SYSTEM, f"Visitor: {who}\nFunnel step: {r.step}\nPage: {r.page_text}")
jobs = [ask(personas[r.segment], r) for r in cells.itertuples()] + [ask(NO_PERSONA, r) for r in ctrl.itertuples()]
try:
    replies = llm.map(jobs, max_tokens=200, return_exceptions=True)      # a refusal fills its slot, not a crash
except se.llm.BudgetExceeded as e:
    sys.exit(f"stopped by the budget (answers so far are cached; rerun resumes): {e}")

ps = [p_of(r) for r in replies]
cells["p_llm"], ctrl["p_np"] = ps[:len(cells)], ps[len(cells):]
print(f"LLM calls: {len(jobs)} ({len(cells)} cells + {len(ctrl)} no-persona controls); set to 0.5 (unparsed or "
      f"failed): {sum(p == 0.5 for p in ps)}")
df = df.merge(cells.drop(columns="page_text"), on=["segment", "variant", "step"], how="left")
df = df.merge(ctrl[["variant", "step", "p_np"]], on=["variant", "step"], how="left")

# 4. Baselines from past traffic (train split only): per-step rate and per-cell rate.
glob = train.continued.mean()
for name, keys in {"step_rate": ["step"], "cell_rate": ["segment", "variant", "step"]}.items():
    g = train.groupby(keys).continued.agg(["sum", "count"]).reset_index()
    g[name] = (g["sum"] + 5 * glob) / (g["count"] + 5)                # shrink small cells toward the global rate
    df = df.merge(g[keys + [name]], on=keys, how="left").fillna({name: glob})

# 5. Ours: stack = logistic regression on logits of [cell_rate, llm], fit on train (calibrates the LLM's "yes" bias).
L = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
X = np.column_stack([np.ones(len(df)), L(df.cell_rate), L(df.p_llm)])
tr, y, w = (df.split == "train").to_numpy(), df.continued.to_numpy(), np.zeros(3)
for _ in range(25):                                                   # Newton steps, small ridge for stability
    p = 1 / (1 + np.exp(-X[tr] @ w))
    H = X[tr].T @ (X[tr] * (p * (1 - p))[:, None]) + 1e-3 * np.eye(3)
    w -= np.linalg.solve(H, X[tr].T @ (p - y[tr]) + 1e-3 * w)
df["p_stack"] = 1 / (1 + np.exp(-X @ w))

# 6. Seal the test predictions BEFORE the truth is joined (pre-registration; the card shows the seal).
frame = pd.DataFrame({"truth": df.continued, "group": df.visitor_id, "segment": df.segment, "split": df.split,
                      "pred_step_rate": df.step_rate, "pred_cell_rate": df.cell_rate,
                      "pred_llm_nopersona": df.p_np, "pred_llm": df.p_llm, "pred_stack": df.p_stack})
registry = out / ".onkayit"
se.seal(frame[frame.split == "test"].drop(columns="truth"), "site_visitors test predictions; ours = stack, raw = llm",
        registry=registry)                                           # name ours in the seal, before the truth

# 7. Compare variants on test visitors: observed conversion and lift (visitor bootstrap) vs each arm's prediction.
test = df[df.split == "test"]
vis = test.groupby("visitor_id").agg(variant=("variant", "first"), segment=("segment", "first"),
                                     bought=("continued", lambda s: int(len(s) == n_steps and s.iloc[-1] == 1)))
arms = {"llm": "p_llm", "no-persona": "p_np", "stack": "p_stack", "cell_rate": "cell_rate"}
conv = df.groupby(["segment", "variant", "step"])[list(arms.values())].first().groupby(["segment", "variant"]).prod()
pred = vis.join(conv, on=["segment", "variant"]).groupby("variant")[list(arms.values())].mean()
A, B = sorted(vis.variant.unique())[:2]                              # control first, challenger second
rs, va, vb = np.random.default_rng(0), vis.bought[vis.variant == A].to_numpy(), vis.bought[vis.variant == B].to_numpy()
lift = np.array([rs.choice(vb, len(vb)).mean() - rs.choice(va, len(va)).mean() for _ in range(a.B)])
print("\nvariant  observed conversion   predicted: " + "  ".join(arms))
for v, g in vis.groupby("variant"):
    print(f"{v:8} {g.bought.mean():.3f} (n={len(g):4})     " + "  ".join(f"{pred.loc[v, c]:.3f}" for c in arms.values()))
print(f"lift {B}-{A}: observed {vb.mean() - va.mean():+.3f} [{np.quantile(lift, .025):+.3f}, "
      f"{np.quantile(lift, .975):+.3f}] | predicted " + ", ".join(f"{k} {pred.loc[B, c] - pred.loc[A, c]:+.3f}"
                                                                  for k, c in arms.items()))
drop = test.pivot_table(index="step_order", columns="variant", values=["continued", "p_llm"], aggfunc="mean")
print("\ncontinue rate by step (observed `continued` vs LLM `p_llm`, test rows):\n" + drop.round(3).to_string())

# 8. Evidence card: baseline ladder, controls (no-persona arm + automatic persona shuffle), paired CIs, audits.
report = se.evidence.evaluate(frame, title="site_visitors funnel", ours="stack", raw="llm", task="binary",
                              primary="auc", baselines_oof=["step_rate", "cell_rate"], B=a.B, prereg=registry,
                              multiplier=False)                       # True adds "worth N humans" (slow)
s = se.evidence.summary(report)
EN = {"tahmin_": "", "taban_": "baseline_", "kontrol_karistir_": "shuffled_", "rastgele": "random", "marjinal":
      "marginal", "demografi": "segment_mean"}                       # internal ladder column names -> English
print("\n" + str(s["headline_en"] or s["headline"]))
for r in s["ladder"]:
    name = functools.reduce(lambda n, kv: n.replace(*kv), EN.items(), str(r["column"] or r["label"]))
    print(f"  {name:34} {r['value']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
print("badge:", s["badge"], "| audits with warnings:", [d["check"] for d in s["audits"] if d["status"] == "warning"])
se.evidence.card(report, out / "card.md", lang="en")
se.evidence.dashboard([report], out / "dashboard.html")
cost = f" | spend this run ${llm.spent_usd:.4f}, ledger ${llm.ledger.total():.4f} of ${a.max_usd}" if a.live else ""
print(f"written: {out / 'card.md'}, {out / 'dashboard.html'}{cost}")
