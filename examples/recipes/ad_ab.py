"""Recipe: rank 4 ad creatives / messages for a target audience and estimate each one's lift over the control ad,
with CIs. A persona panel scores every (segment, creative) cell; past campaign data gives the baselines and the
calibration; the evidence card decides whether the panel added anything.

    python3 examples/recipes/ad_ab.py                                   # offline: FakeLLM + synthetic campaign
    python3 examples/recipes/ad_ab.py --data data/campaign.csv --audience students,parents --out runs/ad
    python3 examples/recipes/ad_ab.py --live --max-usd 1 [--key-file <path>]   # Claude Haiku; else $ANTHROPIC_API_KEY

--data CSV, one row per person who was shown one ad (randomised arms):
    person_id  str   person id (bootstrap cluster; one person = one row = one arm)
    segment    str   audience segment, e.g. students
    persona    str   segment description fed to the LLM (first value per segment is used; NO outcome info in it)
    arm        str   creative id; the control ad's id is `control` (change with --control)
    message    str   the ad text / creative description that person saw
    converted  0/1   the KPI: 1 = clicked / signed up / bought (pick ONE before looking at results)
    date       str   optional ISO date; when present the last 30 % of dates are the test split (else 30 % of people)
Guide, pitfalls and the literature behind each step: docs/recipes/ad_ab.md.
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

# 0. Day-of knobs: product, creatives and segments are the things to replace with the brief's own.
PRODUCT = "a grocery delivery app"
ADS = {"control": "Groceries delivered. (the current ad: logo and app-store button)",
       "price": "First order 30% off, free delivery over 250 TL", "speed": "Your groceries at the door in 15 minutes",
       "family": "Plan the week's family meals in one tap; the basket refills itself",
       "green": "Fresh from local farms, plastic-free bags, carbon-neutral delivery"}
SEGMENTS = {"students": "university student, 21, shares a flat, tight monthly budget, orders late at night",
            "professionals": "young professional, 29, works long hours, values time over money, lives alone",
            "parents": "parent of two, 38, plans meals for the week, shops in bulk, careful with money",
            "retirees": "retired teacher, 67, shops at the local market, cautious about new apps"}
SYSTEM = ("You simulate one person seeing a mobile ad. Read the person and the ad, then estimate the probability "
          "that this person taps the ad and signs up. Reply with JSON only: {\"p\": <0-1>, \"reason\": \"<10 words>\"}")
EN = {"tahmin_": "", "taban_": "baseline_", "kontrol_karistir_": "shuffled_", "rastgele": "random", "marjinal": "mean",
      "demografi": "segment_mean"}


def synthetic(n=2500, seed=0):
    """Ground truth: ~8 % base rate; price wins with students, speed with professionals, family with parents;
    retirees barely respond to anything (the fake LLM below wrongly expects green to work for them)."""
    rs, rows = np.random.default_rng(seed), []
    seg_z = {"students": 0.2, "professionals": 0.1, "parents": 0.0, "retirees": -0.8}
    bonus = {("price", "students"): 0.9, ("speed", "professionals"): 0.8, ("family", "parents"): 0.9,
             ("price", "parents"): 0.4, ("speed", "students"): 0.3, ("green", "professionals"): 0.2}
    for i in range(n):
        seg, arm, day = rs.choice(list(SEGMENTS)), rs.choice(list(ADS)), int(rs.integers(1, 22))
        z = -2.6 + seg_z[seg] + (arm != "control") * 0.15 + bonus.get((arm, seg), 0.0)
        rows.append({"person_id": f"p{i:05d}", "segment": seg, "persona": SEGMENTS[seg], "arm": arm,
                     "message": ADS[arm], "converted": int(rs.random() < 1 / (1 + np.exp(-z))),
                     "date": f"2026-10-{day:02d}"})
    return pd.DataFrame(rows)


def fake_panel(system, user):
    """Stands in for an LLM with a 'yes' bias (~0.4 vs ~0.08 truth) and exaggerated, half-right effects."""
    u, p = user.lower(), 0.35
    p += 0.20 * ("30% off" in u) * ("budget" in u) + 0.20 * ("15 minutes" in u) * ("long hours" in u)
    p += 0.25 * ("family meals" in u) * ("parent" in u) + 0.10 * ("local farms" in u) * ("retired" in u)
    p += (zlib.crc32(user.encode()) % 100 - 50) / 1000                      # stable per-prompt noise
    return json.dumps({"p": round(float(np.clip(p, 0.02, 0.98)), 3), "reason": "fake"})


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--data", help="your campaign CSV (columns in the module docstring)")
ap.add_argument("--audience", help="comma-separated segments to rank for (default: everyone)")
ap.add_argument("--control", default="control", help="arm id of the control ad")
ap.add_argument("--product", default=PRODUCT, help="one line on the product, shown in every prompt (step 0)")
ap.add_argument("--live", action="store_true", help="use the Claude backend instead of FakeLLM")
ap.add_argument("--key-file", help="file holding the API key (else $ANTHROPIC_API_KEY)")
ap.add_argument("--model", default="claude-haiku-5-5")
ap.add_argument("--max-usd", type=float, default=None, help="spend cap for --live (required with --live)")
ap.add_argument("--out", default="runs/ad_ab", help="card, dashboard, seals, cache and ledger go here")
ap.add_argument("--B", type=int, default=300, help="bootstrap replicates (1000 for the final card)")
a = ap.parse_args()
(out := Path(a.out)).mkdir(parents=True, exist_ok=True)

# 1. Data: your CSV or the synthetic campaign. Change the split rule here if the brief fixes one.
df = pd.read_csv(a.data) if a.data else synthetic()
need = {"person_id", "segment", "persona", "arm", "message", "converted"}
if need - set(df.columns) or a.control not in set(df.get("arm", [])):
    sys.exit(f"--data needs columns {sorted(need)} and rows with arm == {a.control!r} (see --control)")
if "date" in df:                                                     # time split: the test days come last
    t = pd.to_datetime(df.date).astype("int64")
    df["split"] = np.where(t > np.quantile(t, 0.7), "test", "train")
else:                                                                # else person-disjoint split
    df["split"] = ["test" if zlib.crc32(str(v).encode()) % 10 < 3 else "train" for v in df.person_id]

# 2. Backend: FakeLLM offline; --live = Claude with a hard budget (BudgetExceeded stops new calls).
if a.live and a.max_usd is None:
    sys.exit("--live needs --max-usd (e.g. --max-usd 1); the ledger stops new calls at that total")
key = Path(a.key_file).expanduser().read_text().strip() if a.live and a.key_file else None   # never printed
if key == "":
    sys.exit("--key-file is empty")
llm = se.llm.Claude(model=a.model, effort="low", thinking="off", max_tokens=150, cache_dir=out / ".cache",
                    budget_usd=a.max_usd, api_key=key, tag="ad_ab") if a.live else se.FakeLLM(fake_panel)

# 3. Panel: one prompt per (segment, ad) cell, plus the no-persona control arm per ad (does the persona matter?).
personas = df.groupby("segment").persona.first().to_dict()
cells = df[["segment", "arm", "message"]].drop_duplicates(["segment", "arm"]).copy()
ask = lambda who, ad: (SYSTEM, f"Product: {a.product}\nPerson: {who}\nAd: {ad}")
jobs = [ask(personas[r.segment], r.message) for r in cells.itertuples()]
ctrl = cells.drop_duplicates("arm").copy()
jobs += [ask("a typical adult in this market", r.message) for r in ctrl.itertuples()]
try:
    replies = llm.map(jobs, max_tokens=150, return_exceptions=True)  # one refusal / 5xx must not kill the run
except se.llm.StopError as e:                                         # BudgetExceeded, no / bad key, spend limit
    sys.exit(f"stopped ({type(e).__name__}): {e} (finished replies are cached; rerun resumes)")
if len(bad := [r for r in replies if isinstance(r, Exception)]) > len(jobs) // 5:
    sys.exit(f"{len(bad)}/{len(jobs)} calls failed, e.g. {type(bad[0]).__name__}: {str(bad[0])[:200]}")


def pj(r):
    """Reply -> p; a cut-off JSON still yields its "p"; a failed call or no number -> None (counted, then 0.5)."""
    if isinstance(r, Exception):
        return None
    j, m = se.sim.parse_json(r.text), re.search(r'"p"\s*:\s*"?([0-9.]+)', r.text)
    try:
        return float(np.clip(float(j["p"] if isinstance(j, dict) and "p" in j else m.group(1)), 0.01, 0.99))
    except (TypeError, ValueError, AttributeError):
        return None


ps = [pj(r) for r in replies]
n_bad, ps = ps.count(None), [0.5 if p is None else p for p in ps]
cells["p_llm"], ctrl["p_np"] = ps[:len(cells)], ps[len(cells):]
df = df.merge(cells[["segment", "arm", "p_llm"]], on=["segment", "arm"]).merge(ctrl[["arm", "p_np"]], on="arm")

# 4. Baselines from past data (train split only): per-ad rate and per-(segment, ad) rate, shrunk to the global rate.
y, tr = df.converted.to_numpy(float), np.flatnonzero(df.split == "train")
aid, cid = df.groupby("arm").ngroup().to_numpy(), df.groupby(["segment", "arm"]).ngroup().to_numpy()
rate = lambda code, idx, k=5: ((np.bincount(code[idx], y[idx], code.max() + 1) + k * y[idx].mean())
                               / (np.bincount(code[idx], minlength=code.max() + 1) + k))[code]
df["arm_rate"], df["cell_rate"] = rate(aid, tr), rate(cid, tr)

# 5. Calibrate + shrink: logistic on logits of [cell_rate, llm], fit on train. Fixes the 'yes' bias (intercept)
#    and shrinks the LLM's exaggerated effects (its weight). Swap in methods.hybrid.BinaryCalibrator if no history.
L = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
fold = np.array([zlib.crc32(str(v).encode()) % 5 for v in df.person_id])


def fit(idx):
    """Stack refit on train rows idx -> P(convert), all rows. Out-of-fold cell rates on train (else leakage)."""
    c = rate(cid, idx)
    for k in range(5):
        c[idx[fold[idx] == k]] = rate(cid, idx[fold[idx] != k])[idx[fold[idx] == k]]
    X, w = np.column_stack([np.ones(len(df)), L(c), L(df.p_llm)]), np.zeros(3)
    for _ in range(25):                                               # Newton steps, small ridge for stability
        p = 1 / (1 + np.exp(-X[idx] @ w))
        H = X[idx].T @ (X[idx] * (p * (1 - p))[:, None]) + 1e-3 * np.eye(3)
        w -= np.linalg.solve(H, X[idx].T @ (p - y[idx]) + 1e-3 * w)
    return 1 / (1 + np.exp(-X @ w)), w


df["p_stack"], w = fit(tr)
print(f"LLM calls: {len(jobs)} ({len(ctrl)} no-persona, {n_bad} unparsed or failed -> 0.5) | 'yes' bias: raw LLM mean {df.p_llm.iloc[tr].mean():.3f} vs "
      f"observed {y[tr].mean():.3f} | LLM weight after calibration {w[2]:.2f} (1 = effects as stated, 0 = ignored)")

# 6. Pre-register which column is "ours" and seal the test predictions BEFORE the truth is joined; the card's
#    selection-bias audit reads the note (it must be older than the seal) and shows the seal's hash and time.
frame = pd.DataFrame({"truth": df.converted, "group": df.person_id, "segment": df.segment, "split": df.split,
                      "time": df.get("date"), "pred_arm_rate": df.arm_rate, "pred_cell_rate": df.cell_rate,
                      "pred_llm_nopersona": df.p_np, "pred_llm": df.p_llm, "pred_stack": df.p_stack})
(registry := out / ".onkayit").mkdir(exist_ok=True)
(registry / "prereg.md").write_text("ours: pred_stack | primary: auc | baselines: pred_arm_rate, pred_cell_rate | "
                                    f"lift for audience {a.audience or 'all'} vs control {a.control}\n")
se.seal(frame[frame.split == "test"].drop(columns="truth"), "ad_ab test predictions, ours: pred_stack",
        registry=registry)

# 7. Rank + lift vs control for the target audience (test people). Observed lift: bootstrap within each arm.
#    Predicted lift: audience-weighted (segment, ad) minus (segment, control); CI from stack refits on resampled train.
aud = a.audience.split(",") if a.audience else sorted(df.segment.unique())
test = df[(df.split == "test") & df.segment.isin(aud)]
mix = test.segment.value_counts(normalize=True)                       # audience weights per segment
lift = lambda p, v: sum(mix[s] * (p.loc[(s, v)] - p.loc[(s, a.control)]) for s in mix.index)
cellp = lambda p: pd.Series(np.asarray(p), index=pd.MultiIndex.from_frame(df[["segment", "arm"]])).groupby(
    level=[0, 1]).first()                                             # one value per (segment, ad) cell
rs, arms = np.random.default_rng(0), [v for v in sorted(df.arm.unique()) if v != a.control]
by_arm = {v: g.converted.to_numpy() for v, g in test.groupby("arm")}
if set(arms + [a.control]) - set(by_arm):
    sys.exit(f"no test rows for {sorted(set(arms + [a.control]) - set(by_arm))} in audience {aud}; widen --audience")
if gaps := [(s, v) for s in mix.index for v in arms + [a.control] if (s, v) not in set(zip(df.segment, df.arm))]:
    sys.exit(f"segment x ad cells with no rows (arms not randomised?): {gaps[:5]}; drop them from --audience")
boot = [cellp(fit(rs.choice(tr, len(tr)))[0]) for _ in range(a.B)]
pred = {c: cellp(df[c]) for c in ("p_stack", "p_llm", "cell_rate")}
draw = lambda v: rs.choice(by_arm[v], len(by_arm[v])).mean()
ci = lambda x: f"[{np.quantile(x, .025):+.3f}, {np.quantile(x, .975):+.3f}]"
print(f"\naudience: {', '.join(aud)} | control rate {by_arm[a.control].mean():.3f} (n={len(by_arm[a.control])})\n"
      "ad        observed lift [95% CI]        stack lift [95% CI]           raw llm  cell_rate")
rows = []
for v in arms:
    bs, pb = [draw(v) - draw(a.control) for _ in range(a.B)], [lift(p, v) for p in boot]
    rows.append((v, by_arm[v].mean() - by_arm[a.control].mean(), lift(pred["p_stack"], v), lift(pred["p_llm"], v),
                 lift(pred["cell_rate"], v)))
    print(f"{v:9} {rows[-1][1]:+.3f} {ci(bs)}      {rows[-1][2]:+.3f} {ci(pb)}      {rows[-1][3]:+.3f}   "
          f"{rows[-1][4]:+.3f}")
r = pd.DataFrame(rows, columns=["ad", "observed", "ours", "llm", "cell"]).set_index("ad")
top = lambda c: " > ".join(map(str, r[c].sort_values(ascending=False).index))
sp = lambda c: f"{r[c].corr(r.observed, 'spearman'):.2f}"
print(f"ranking: stack {top('ours')} vs observed {top('observed')} (Spearman {sp('ours')}; raw llm {sp('llm')}; "
      f"cell_rate alone {sp('cell')}; with 4 ads only a perfect match is informative)")

# 8. Evidence card: baseline ladder, controls (no-persona arm + automatic persona shuffle), paired CIs, audits.
report = se.evidence.evaluate(frame, title="ad_ab creative ranking", ours="stack", raw="llm", task="binary",
                              primary="auc", baselines_oof=["arm_rate", "cell_rate"], B=a.B, prereg=registry,
                              multiplier=False)                       # True adds "worth N humans" (~+20 s)
s = se.evidence.summary(report)
o, b, d = s["ours"], s["best_baseline"], s["ours_minus_baseline"] or {}
print(f"\n{s['primary_metric']} (person level, test): stack {o['value']:.3f} [{o['lo']:.3f}, {o['hi']:.3f}] vs best "
      f"baseline {b['label'] or b['column']} {b['value']:.3f}; difference {d.get('fark', np.nan):+.3f} "
      f"[{d.get('alt', np.nan):+.3f}, {d.get('ust', np.nan):+.3f}]")
for row in s["ladder"]:                                              # lowest first; column names in English
    name = re.sub("|".join(EN), lambda m: EN[m.group()], str(row["column"] or row["label"]))
    print(f"  {name:34} {row['value']:.3f} [{row['lo']:.3f}, {row['hi']:.3f}]")
print("badge:", s["badge"], "| audits with warnings:", [d["name"] or d["check"] for d in s["audits"]
                                                        if d["status"] == "warning"])
se.evidence.card(report, out / "card.md", lang="en")
se.evidence.dashboard([report], out / "dashboard.html")
print(f"written: {out / 'card.md'}, {out / 'dashboard.html'}" + (f" | spend ${llm.ledger.total():.4f} of "
                                                                f"${a.max_usd}" if a.live else ""))
