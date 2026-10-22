"""Recipe: will this person buy in this category in the next window? Persona (demographics) + purchase history ->
LLM P(buy) per (person, category), against LLM-free baselines on a balanced cell mask (the talp73 setup), with
calibration, balanced accuracy and the gap-closed evidence card.

    python3 examples/recipes/purchase_pred.py                       # offline: FakeLLM + synthetic shoppers, ~20 s
    python3 examples/recipes/purchase_pred.py --data data/cells.csv --out runs/purchase
    python3 examples/recipes/purchase_pred.py --live --max-usd 1    # Claude Haiku, key from $ANTHROPIC_API_KEY
    python3 examples/recipes/purchase_pred.py --live --key-file <path> --max-usd 1 --persons 100

--data CSV, one row per (person, category); every person has a row for every category:
    person_id    str   person id (bootstrap cluster; a person is never split across train and test)
    category     str   product category, e.g. groceries
    demo_<name>  str   one or more demographic columns, e.g. demo_age, demo_income (constant within a person;
                       they become the persona text and the survey model's features; NO outcome info in them)
    n_before     int   purchases in this category in the 12 months before the cutoff
    n_recent     int   of those, purchases in the last 6 months before the cutoff
    bought       0/1   at least one purchase in this category in the target window after the cutoff
    p_true       float optional, simulated data only: the known true P(buy); feeds the oracle ceiling
Guide, pitfalls and the literature behind each step: docs/recipes/purchase_pred.md.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # run from a checkout without pip install
import simengine as se                                              # noqa: E402

# 0. Day-of knobs: categories, demographics and the window wording are the things to replace with the brief's own.
CATS = ["groceries", "electronics", "baby products", "books", "beauty", "pet supplies", "sports", "home & kitchen"]
DEMO = {"age": ["18-24", "25-34", "35-54", "55+"], "income": ["low", "middle", "high"],
        "household": ["1 person", "2 people", "3+ people"], "orders": ["rarely", "monthly", "weekly"]}
WINDOW = "the next 6 months"
SYSTEM = ("You simulate one online shopper. For each listed category give the probability (0-100) that this shopper "
          f"buys at least one item in it in {WINDOW}. Reply with JSON only: {{\"<category>\": <0-100>, ...}}")


def synthetic(n=240, seed=0):
    """Ground truth: demographics shift category rates a little; a hidden person x category taste drives both the
    past and the future purchases (so history beats demographics, as in talp73)."""
    rs, rows = np.random.default_rng(seed), []
    base = rs.normal(-1.3, 0.5, len(CATS))
    for i in range(n):
        d = {k: rs.choice(v) for k, v in DEMO.items()}
        act = rs.normal(0, 0.5) + {"rarely": -0.6, "monthly": 0, "weekly": 0.6}[d["orders"]]
        for c, cat in enumerate(CATS):
            z = base[c] + act + rs.normal(0, 1.0)                            # hidden taste
            z += 1.2 * (cat == "baby products") * (d["household"] == "3+ people") - 0.8 * (cat == "baby products")
            z += 0.5 * (cat == "electronics") * (d["income"] == "high") + 0.5 * (cat == "books") * (d["age"] == "55+")
            lam = np.exp(z)                                                  # purchases per 6 months
            n_before = int(rs.poisson(2 * lam))
            rows.append({"person_id": f"p{i:04d}", "category": cat, **{f"demo_{k}": v for k, v in d.items()},
                         "n_before": n_before, "n_recent": int(rs.binomial(n_before, 0.5)),
                         "bought": int(rs.poisson(lam) > 0), "p_true": 1 - np.exp(-lam)})
    return pd.DataFrame(rows)


def fake_shopper(system, user):
    """Stands in for an LLM that reads the history, half-reads the demographics and says 'yes' too often."""
    cats = user.split("Categories: ")[-1].split(" | ")
    out = {}
    for cat in cats:
        n = int(user.split(f"- {cat}: ")[1].split(" ")[0]) if f"- {cat}: " in user else -1   # -1 = no history shown
        p = 0.55 + (0.25 * min(n, 3) / 3 - 0.10 * (n == 0) if n >= 0 else 0.0)
        p += 0.15 * (cat == "baby products") * ("3+ people" in user) + 0.10 * (cat == "books") * ("55+" in user)
        p += (zlib.crc32((user + cat).encode()) % 100 - 50) / 500                # stable per-cell noise
        out[cat] = round(100 * float(np.clip(p, 0.02, 0.98)))
    return json.dumps(out)


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--data", help="your (person, category) CSV (columns in the module docstring)")
ap.add_argument("--live", action="store_true", help="use the Claude backend instead of FakeLLM")
ap.add_argument("--key-file", help="file holding the API key (else $ANTHROPIC_API_KEY)")
ap.add_argument("--model", default="claude-haiku-5-5")
ap.add_argument("--max-usd", type=float, default=None, help="spend cap for --live (required with --live)")
ap.add_argument("--persons", type=int, default=None, help="use only the first N persons (cost control)")
ap.add_argument("--out", default="runs/purchase_pred", help="card, dashboard, seals, cache and ledger go here")
ap.add_argument("--B", type=int, default=300, help="bootstrap replicates (1000 for the final card)")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

# 1. Data + person-disjoint split (30 % of persons are test). Change the split rule here if the brief fixes one.
df = pd.read_csv(a.data) if a.data else synthetic()
demo = [c for c in df.columns if c.startswith("demo_")]
need = {"person_id", "category", "n_before", "n_recent", "bought"}
if need - set(df.columns) or not demo:
    sys.exit(f"--data is missing columns: {sorted(need - set(df.columns)) or ['demo_<name>']}")
if a.persons:
    df = df[df.person_id.isin(df.person_id.drop_duplicates()[:a.persons])]
df = df.reset_index(drop=True)
df["split"] = ["test" if zlib.crc32(str(p).encode()) % 10 < 3 else "train" for p in df.person_id]

# 2. Balanced mask (talp73 1b): per person min(5, #bought, #not bought) cells of each class. Accuracy on all cells
#    rewards "never buys"; on the mask chance is 0.5 and balanced accuracy = accuracy. Everything below uses it.
rs = np.random.default_rng(1000)
keep = []
for _, g in df.groupby("person_id"):
    k = min(5, int(g.bought.sum()), int((1 - g.bought).sum()))
    keep += [i for cls in (0, 1) for i in rs.permutation(g.index[g.bought == cls])[:k]]
df["mask"] = df.index.isin(keep)

# 3. Backend: FakeLLM offline; --live = Claude with a hard budget (BudgetExceeded stops new calls).
if a.live:
    if a.max_usd is None:
        sys.exit("--live needs --max-usd (e.g. --max-usd 1); the ledger stops new calls at that total")
    key = Path(a.key_file).expanduser().read_text().strip() if a.key_file else None   # never printed
    llm = se.llm.Claude(model=a.model, effort="low", thinking="off", max_tokens=400, cache_dir=out / ".cache",
                        budget_usd=a.max_usd, api_key=key, tag="purchase_pred")
else:
    llm = se.FakeLLM(fake_shopper)

# 4. Ask: one call per person and arm, all categories at once. L1 = demographics persona, L2 = persona + history;
#    L0 = one no-persona call (the control: any L1 gain over it is the persona's own contribution).
people, cats = df.groupby("person_id").first(), sorted(df.category.unique())
hist = df.pivot(index="person_id", columns="category", values=["n_before", "n_recent"])
persona = lambda p: "; ".join(f"{c[5:]}: {people.at[p, c]}" for c in demo)
history = lambda p: "\n".join(f"- {c}: {hist.at[p, ('n_before', c)]} purchases, {hist.at[p, ('n_recent', c)]} in "
                              "the last 6 months" for c in cats)
ask = lambda who, past="": (SYSTEM, f"Shopper: {who}\n" + (f"Purchases in the 12 months before today:\n{past}\n"
                                                         if past else "") + "Categories: " + " | ".join(cats))
jobs = [ask("a typical online shopper")] + [ask(persona(p)) for p in people.index]
jobs += [ask(persona(p), history(p)) for p in people.index]
try:
    replies = llm.map(jobs, max_tokens=400)
except se.llm.BudgetExceeded as e:
    sys.exit(f"stopped by the budget: {e}")
P = []
for r in replies:                                                    # unparsed reply / category -> NaN, then 0.5
    j = se.sim.parse_json(r.text)
    P.append({c: pd.to_numeric((j if isinstance(j, dict) else {}).get(c), errors="coerce") for c in cats})
P = (pd.DataFrame(P) / 100).clip(0.01, 0.99)
bad, n, A = int(P.isna().sum().sum()), len(people), P.fillna(0.5).to_numpy()
row = df.person_id.map({p: i for i, p in enumerate(people.index)}).to_numpy()
col = [cats.index(c) for c in df.category]
df["llm_l0"], df["llm_l1"], df["llm_l2"] = A[0, col], A[1 + row, col], A[1 + n + row, col]
print(f"LLM calls: {len(jobs)} (1 no-persona + {n} L1 + {n} L2) | unparsed category answers: {bad}")

# 5. LLM-free baselines, fit on train persons' masked cells only: popularity, survey model, history model.
X_demo = pd.get_dummies(df[demo + ["category"]].astype(str)).astype(float)
X_hist = pd.concat([pd.get_dummies(df.category).astype(float), np.log1p(df[["n_before", "n_recent"]]),
                    (df.n_before > 0).astype(float).rename("ever")], axis=1)
fit = (df.split == "train") & df["mask"]
X_both = pd.concat([X_demo, X_hist.drop(columns=cats, errors="ignore")], axis=1)
lr = lambda X, rows=fit: LogisticRegression(max_iter=500).fit(X.loc[rows], df.bought.loc[rows]).predict_proba(X)[:, 1]
df["popularity"] = df.category.map(df[fit].groupby("category").bought.mean()).fillna(0.5)
df["survey"], df["history"] = lr(X_demo), lr(X_hist)
df["hist_survey"] = lr(X_both)

# 6. Calibration (Platt on train) fixes the LLM's yes-bias; ours = stack of history model + L2 (hybrid).
L = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
df["llm_l2_cal"] = se.methods.hybrid.BinaryCalibrator("platt").fit(df.llm_l2[fit], df.bought[fit]).predict(df.llm_l2)
df["stack"] = lr(pd.DataFrame({"hs": L(df.hist_survey), "l2": L(df.llm_l2)}))
te = df[(df.split == "test") & df["mask"]].copy()
# ceiling: history+survey (+ p_true when known) fit on the TEST cells' own labels: in-sample oracle, an upper bound
X_orc = X_both.assign(p_true=L(df.p_true)) if "p_true" in df else X_both
te["oracle"] = lr(X_orc, te.index)[te.index]

# 7. The imbalanced-accuracy trap, in one line: plain accuracy on ALL test cells.
allt = df[df.split == "test"]
acc = lambda p: float(((allt[p] > 0.5) == allt.bought).mean())
print(f"accuracy on all test cells (base rate {allt.bought.mean():.2f}): always-no {1 - allt.bought.mean():.3f}, "
      f"raw L2 {acc('llm_l2'):.3f}, history {acc('history'):.3f}  <- why we score the balanced mask")

# 8. Seal the test predictions BEFORE the truth is joined (pre-registration; the card shows the seal).
arms = ["popularity", "survey", "history", "hist_survey", "llm_l0", "llm_l1", "llm_l2", "llm_l2_cal", "stack"]
frame = pd.DataFrame({"truth": te.bought, "group": te.person_id, "segment": te.category, "split": te.split,
                      **{f"pred_{c}": te[c] for c in arms}, "pred_oracle": te.oracle})
registry = out / ".onkayit"
se.seal(frame.drop(columns="truth"), "purchase_pred test predictions (ours = stack)", registry=registry)

# 9. Evidence card: baseline ladder, controls (L0 no-persona arm + automatic persona shuffle), person-bootstrap CIs,
#    gap closed = (ours - best baseline) / (oracle - best baseline).
kw = dict(title="purchase_pred (person x category, balanced mask)", ours="stack", raw="llm_l2", task="binary",
          primary="balanced_accuracy", B=a.B, prereg=registry, baselines_oof=arms[:4])
try:
    report = se.evidence.evaluate(frame, **kw, ceiling_col="pred_oracle", ceiling_label="in-sample oracle")
except ZeroDivisionError:                                             # oracle ties the best baseline: no gap to close
    print("oracle ties the best baseline exactly: gap closed undefined, card without a ceiling")
    report = se.evidence.evaluate(frame.drop(columns="pred_oracle"), **kw)
s = se.evidence.summary(report)
print(f"\n{s['headline']}\nbalanced accuracy, test persons' masked cells ({len(te)} cells, {te.person_id.nunique()} "
      "persons), 95% person-bootstrap CI:")
for r in s["ladder"]:
    print(f"  {str(r['label'] or r['column']):40} {r['value']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
g, pct = s["gap_closed"] or {}, lambda v: "?" if v is None else f"{100 * v:.0f}"
print("gap closed:", "n/a" if g.get("deger") is None else f"{pct(g['deger'])} % [{pct(g.get('alt'))}, "
      f"{pct(g.get('ust'))}]", "| badge:", s["badge"], "| audits with warnings:",
      [d["check"] for d in s["audits"] if d["status"] == "warning"])
se.evidence.card(report, out / "card.md", lang="en")
se.evidence.dashboard([report], out / "dashboard.html")
cost = f" | spend ${llm.ledger.total():.4f} of ${a.max_usd}" if a.live else ""
print(f"written: {out / 'card.md'}, {out / 'dashboard.html'}{cost}")
