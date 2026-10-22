"""Recipe: price sensitivity and willingness to pay from a choice-based conjoint (CBC). Personas answer the same
choice sets the respondents saw; a conditional logit gives part-worths, WTP with person-bootstrap CIs and a price
elasticity curve, for humans and for the LLM side by side; the evidence card decides whether the LLM adds anything.

    python3 examples/recipes/pricing.py                       # offline: FakeLLM + synthetic CBC, ~15 s
    python3 examples/recipes/pricing.py --data data/cbc.csv --out runs/pricing
    python3 examples/recipes/pricing.py --live --max-usd 1 [--key-file <path>]   # Claude Haiku; else $ANTHROPIC_API_KEY

--data CSV, one row per (respondent, choice task, alternative shown):
    person       str  respondent id (bootstrap cluster; the train/test split never cuts a person)
    segment      str  respondent segment, e.g. student
    persona      str  segment description fed to the LLM (first value per segment is used; NO choice info in it)
    task         str  choice-set id, unique per person (e.g. p001_t03)
    design       str  optional: id of the options shown; the LLM is asked once per segment x design (default: task)
    alternative  str  A / B / ... and one "none" row per task (the no-buy option: price 0, attributes 0)
    price, <ATTRS>  num  price in CUR; one column per key of ATTRS below (0/1 or numeric)
    chosen       0/1  exactly one 1 per task
Guide, pitfalls and the literature behind each step: docs/recipes/pricing.md.
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

# 0. Day-of knobs: product, currency, attributes (column -> how the LLM reads it), segments and reference market.
PRODUCT, CUR, PER = "a weekly meal-kit subscription", "TL", "per month"   # PER: what one price buys
ATTRS = {"meals": lambda v: f"{v:.0f} meals a week", "organic": lambda v: "organic" if v else "not organic",
         "free_cancel": lambda v: "cancel any time" if v else "12-month contract"}
TASTE = ("price", "meals", "organic", "free_cancel", "none")  # synthetic only: price per 100 CUR, none = no-buy utility
SEGMENTS = {  # segment -> (persona text fed to the LLM, synthetic true tastes in TASTE order)
    "student": ("university student, tight monthly budget, rarely cooks", (-2.2, .15, .2, 1.0, -4.5)),
    "young_pro": ("works full time in the city, no time to cook, often eats out", (-0.9, .25, .7, .5, -2.0)),
    "family": ("parent of two, plans meals for the family, watches the grocery bill", (-1.3, .45, .6, .4, -2.5)),
    "retiree": ("retired, cooks at home every day, wary of new subscriptions", (-1.6, .10, .5, .8, -2.5))}
MARKET = pd.DataFrame({"alternative": ["A", "B", "none"], "price": [299, 349, 0], "meals": [5, 3, 0],
                       "organic": [1, 0, 0], "free_cancel": [1, 1, 0]})  # A = ours (price swept), B = competitor
SYSTEM = ("You simulate one consumer in a choice survey. Give the probability that this consumer picks each option. "
          "JSON only, one key per option: {\"A\": p, ..., \"none\": p}")


def synthetic(n=400, n_designs=16, per_person=8, seed=0):
    """Ground truth: segment tastes + person noise, Gumbel choice among A, B and no-buy."""
    rs, rows, zero = np.random.default_rng(seed), [], dict.fromkeys(TASTE[:4], 0.0)
    designs = [{alt: dict(zip(TASTE[:4], map(float, (rs.choice([229, 269, 299, 339, 379]), rs.choice([3, 5]),
                *rs.integers(0, 2, 2))))) for alt in "AB"} for _ in range(n_designs)]   # price, meals, organic, cancel
    for i in range(n):
        seg = list(SEGMENTS)[i % len(SEGMENTS)]
        w = {k: v + rs.normal(0, 0.25 * abs(v)) for k, v in zip(TASTE, SEGMENTS[seg][1])}
        for d in rs.choice(n_designs, per_person, replace=False):
            opts = {**designs[d], "none": zero}
            u = {a: (w["none"] if a == "none" else sum(w[k] * (x / 100 if k == "price" else x) for k, x in o.items()))
                 + rs.gumbel() for a, o in opts.items()}
            rows += [{"person": f"p{i:04d}", "segment": seg, "persona": SEGMENTS[seg][0], "task": f"p{i:04d}_d{d}",
                      "design": f"d{d:02d}", "alternative": a, **o, "chosen": int(a == max(u, key=u.get))}
                     for a, o in opts.items()]
    return pd.DataFrame(rows)


def fake_consumer(system, user):
    """Stands in for an LLM that half-knows the market: right direction, too little price pain, too few no-buys.
    Tuned to the demo ATTRS; with other ATTRS it still answers (price only), so an offline dry run works."""
    sens, u = (1.6 if "student" in user else 0.7), {"none": -0.8}
    for alt, txt in re.findall(r"^([A-Z]): (.*)$", user, re.M):
        meals = re.search(r"(\d+) meals", txt)
        u[alt] = (-sens * float(re.search(r"(\d+)", txt)[1]) / 100 + 0.3 * float(meals[1] if meals else 0)
                  + 0.6 * ("not organic" not in txt and "organic" in txt) + 0.6 * ("any time" in txt)
                  + (zlib.crc32(txt.encode()) % 40 - 20) / 100)                     # stable per-prompt noise
    e = {k: np.exp(v - max(u.values())) for k, v in u.items()}
    return json.dumps({k: round(v / sum(e.values()), 3) for k, v in e.items()})


def boot(rows, B, seed=0):
    """Person-cluster bootstrap of the pooled CBC coefficients (a resampled person gets fresh task ids)."""
    rs, codes, tcode = np.random.default_rng(seed), pd.factorize(rows.person)[0], pd.factorize(rows.task)[0]
    by, draws = [np.flatnonzero(codes == i) for i in range(codes.max() + 1)], []
    for _ in range(B):
        pick = rs.integers(0, len(by), len(by))
        idx, rep = np.concatenate([by[i] for i in pick]), np.repeat(np.arange(len(pick)), [len(by[i]) for i in pick])
        draws.append(fit(rows.iloc[idx].assign(task=rep * (tcode.max() + 1) + tcode[idx])).coef_)
    return pd.DataFrame(draws)


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--data", help="your CBC CSV (columns in the module docstring)")
ap.add_argument("--live", action="store_true", help="use the Claude backend instead of FakeLLM")
ap.add_argument("--key-file", help="file holding the API key (else $ANTHROPIC_API_KEY)")
ap.add_argument("--model", default="claude-haiku-5-5")
ap.add_argument("--max-usd", type=float, default=None, help="spend cap for --live (required with --live)")
ap.add_argument("--out", default="runs/pricing", help="card, dashboard, curve, seals, cache and ledger go here")
ap.add_argument("--B", type=int, default=200, help="bootstrap replicates (1000 for the final card)")
a = ap.parse_args()
(out := Path(a.out)).mkdir(parents=True, exist_ok=True)

# 1. Data + person-disjoint split (30 % of respondents are test). Change the split here if the brief fixes one.
df = pd.read_csv(a.data) if a.data else synthetic()
need = {"person", "segment", "persona", "task", "alternative", "price", "chosen", *ATTRS}
if need - set(df.columns) or "none" not in set(df.alternative):
    sys.exit(f"--data needs a 'none' row per task; missing columns: {sorted(need - set(df.columns))}")
if set(ATTRS) - set(MARKET):
    sys.exit(f"MARKET (step 0) lacks the ATTRS columns {sorted(set(ATTRS) - set(MARKET))}")
df["design"] = df["design"] if "design" in df else df.task
df["split"] = ["test" if zlib.crc32(str(p).encode()) % 10 < 3 else "train" for p in df.person]

# 2. Backend: FakeLLM offline; --live = Claude with a hard budget (BudgetExceeded stops new calls).
if a.live and a.max_usd is None:
    sys.exit("--live needs --max-usd (e.g. --max-usd 1); the ledger stops new calls at that total")
key = Path(a.key_file).expanduser().read_text().strip() if a.key_file else None   # never printed
llm = (se.llm.Claude(model=a.model, effort="low", thinking="off", max_tokens=120, cache_dir=out / ".cache",
                     budget_usd=a.max_usd, api_key=key, tag="pricing") if a.live else se.FakeLLM(fake_consumer))

# 3. Ask: one prompt per (segment, design) for the persona arm, one per design for the no-persona control.
personas = df.groupby("segment").persona.first().to_dict()
opts = df.drop_duplicates(["design", "alternative"]).set_index("design")
show = lambda d: "\n".join(f"{r.alternative}: {r.price:.0f} {CUR} {PER}; " + "; ".join(
    f(getattr(r, k)) for k, f in ATTRS.items()) for r in opts.loc[[d]].itertuples() if r.alternative != "none")
ask = lambda who, d: (SYSTEM, f"Consumer: {who}\nProduct: {PRODUCT}\n{show(d)}\nnone: buy none of these")
cells, designs = list(df[["segment", "design"]].drop_duplicates().itertuples(index=False)), sorted(set(df.design))
jobs = [ask(personas[s], d) for s, d in cells] + [ask("a typical shopper", d) for d in designs]
try:
    replies = llm.map(jobs, max_tokens=120, return_exceptions=True)   # one refusal / 5xx must not kill the run
except se.llm.StopError as e:                                        # budget / 401 / spend limit; finished replies are cached
    sys.exit(f"stopped by the {'budget' if isinstance(e, se.llm.BudgetExceeded) else 'API'}: {e}")
num = lambda p: {k: v for k, v in p.items() if isinstance(v, (int, float))} if isinstance(p, dict) else {}
probs = [{} if isinstance(r, Exception) else num(se.sim.parse_json(r.text)) for r in replies]  # -> uniform below
if bad := sum(not p for p in probs):
    print(f"WARNING: {bad}/{len(jobs)} replies failed or had no numeric option keys; those tasks fall back to uniform")
p_llm = {(s, d, k): v for (s, d), p in zip(cells, probs) for k, v in p.items()}
p_np = {(d, k): v for d, p in zip(designs, probs[len(cells):]) for k, v in p.items()}
df["llm"] = [p_llm.get(k, 0) for k in zip(df.segment, df.design, df.alternative)]
df["llm_np"] = [p_np.get(k, 0) for k in zip(df.design, df.alternative)]
for c in ("llm", "llm_np"):                                                     # renormalise within each task
    df[c] = df[c].clip(1e-3) / df[c].clip(1e-3).groupby(df.task).transform("sum")
print(f"LLM calls: {len(jobs)} ({len(cells)} segment x design + {len(designs)} no-persona controls)")

# 4. Fit on train respondents: CBC logit with segment shifts of the buy constant and price slope (the baseline
#    to beat) and the stack = same logit + log LLM share as one more attribute. Add segment x attribute terms here.
fit = lambda rows, extra=(): se.methods.MNL(cs := list(ATTRS) + ["price", *extra], base_alt="none",
                                            scale={c: 100 for c in cs if c.startswith("price")}).fit(rows)
segs = sorted(df.segment.unique())
for s in segs[1:]:
    buy = ((df.segment == s) & (df.alternative != "none")).astype(float)
    df[f"buy_{s}"], df[f"price_{s}"] = buy, df.price * buy
SEGX, df["logllm"] = [f"{k}_{s}" for s in segs[1:] for k in ("buy", "price")], np.log(df.llm)
train = df[df.split == "train"]
df["p_mnl"], df["p_stack"] = fit(train, SEGX).predict_proba(df), fit(train, SEGX + ["logllm"]).predict_proba(df)

# 5. Pooled part-worths -> WTP (CUR per attribute unit), 95 % person-bootstrap CIs: humans vs a silicon sample
#    drawn from the LLM shares: 120 respondents, same segment mix and tasks per person as the humans (its CI is
#    draw noise at that n, not uncertainty about the LLM). For one segment's WTP pass train[train.segment == ...].
rs = np.random.default_rng(1)
k_seg = (train.groupby("segment").person.nunique() / train.person.nunique() * 120).round().astype(int)
n_task = int(train.groupby("person").task.nunique().median())
by_sd = dict(list(df.drop_duplicates(["segment", "design", "alternative"]).groupby(["segment", "design"])))
seg_d = {s: [d for t, d in by_sd if t == s] for s in k_seg.index}
sil = pd.concat([g.assign(person=f"{s}-{k}", task=f"{s}-{k}-{d}", chosen=np.eye(len(g), dtype=int)[rs.choice(
    len(g), p=g.llm / g.llm.sum())]) for s in k_seg.index for k in range(k_seg[s])
    for d in rs.permutation(seg_d[s])[:n_task] for g in [by_sd[s, d]]], ignore_index=True)
wtp = lambda c: pd.DataFrame({k: -c[k] / (c["price"] / 100) for k in ATTRS})
arms = {"human": (fit(train), boot(train, a.B)), "llm": (fit(sil), boot(sil, a.B, seed=2))}
print(f"\nWTP in {CUR} per unit [95% CI]   human (train respondents)   llm (silicon, n={sil.person.nunique()})")
for k in ATTRS:
    cell = [(wtp(m.coef_.to_frame().T)[k].iloc[0], *np.quantile(wtp(bs)[k], [.025, .975])) for m, bs in arms.values()]
    print(f"  {k:28} " + "".join(f"{v:8.1f} [{lo:7.1f}, {hi:7.1f}]       " for v, lo, hi in cell))
for arm, (_, bs) in arms.items():                       # WTP divides by the price part-worth: it must stay < 0
    if (bs["price"] >= 0).mean() > .025:
        print(f"WARNING: the {arm} price part-worth CI reaches 0: its WTP is a ratio over ~0, do not quote it")

# 6. Price curve for A in MARKET over the tested price range: share, revenue, arc elasticity, human CI band and the
#    revenue-max price with the spread of its bootstrap draws (same draws as the WTP CIs).
share = lambda c, mk: float(np.exp(v := sum(c[k] * mk[k] for k in ATTRS) + c["price"] / 100 * mk.price
                                   + mk.alternative.map(lambda j: c.get(f"ASC_{j}", 0.0))).iloc[0] / np.exp(v).sum())
tested, curve, D = df.loc[df.alternative != "none", "price"], [], []
for p in np.linspace(tested.min(), tested.max(), 11).round():   # only the prices respondents saw: no extrapolation
    mk = MARKET.assign(price=MARKET.price.where(MARKET.alternative != "A", p))
    D.append(draws := [share(c, mk) for _, c in arms["human"][1].iterrows()])
    curve.append({"price": mk.price.iloc[0], "human": share(arms["human"][0].coef_, mk),
                  "lo": np.quantile(draws, .025), "hi": np.quantile(draws, .975),
                  "llm": share(arms["llm"][0].coef_, mk)})
curve = pd.DataFrame(curve).assign(revenue=lambda c: c.price * c.human)        # elasticity: arc (midpoint) formula
curve["elasticity"] = (d := curve.diff() / curve.rolling(2).mean()).human / d.price
curve.to_csv(out / "price_curve.csv", index=False)
print("\nprice curve for A (human share, 95% CI, llm share):", curve.round(3).to_string(index=False), sep="\n")
pmax = curve.price.to_numpy()[np.argmax(np.array(D).T * curve.price.to_numpy(), axis=1)]   # argmax per draw
lo_p, hi_p = np.quantile(pmax, [.025, .975], method="inverted_cdf")
edge = bool(lo_p <= curve.price.min() or hi_p >= curve.price.max())   # a draw's optimum at the edge may lie beyond it
print(f"revenue-max price (human): {curve.price[curve.revenue.idxmax()]:.0f} {CUR}, 95% of bootstrap draws in "
      f"[{lo_p:.0f}, {hi_p:.0f}]" + (" - reaches the edge of the sweep: test wider prices before quoting it" * edge))

# 7. Seal the test predictions BEFORE the truth is joined (one row per task: {alternative: probability}).
tasks = df[df.chosen == 1].set_index("task")
vec = {f"pred_{m}": df.groupby("task")[["alternative", c]].apply(lambda g: dict(zip(*g.T.values)))
       for m, c in dict(mnl="p_mnl", llm_nopersona="llm_np", llm="llm", stack="p_stack").items()}
frame = pd.DataFrame({"truth": tasks.alternative, "group": tasks.person, "segment": tasks.segment,
                      "item": tasks.design, "split": tasks.split, **vec}).reset_index(drop=True)
reg = out / ".onkayit"
se.seal(frame[frame.split == "test"].drop(columns="truth"), "pricing test predictions, ours = stack; registered "
        "comparison: llm vs llm_nopersona (does the persona text help)", registry=reg)

# 8. Evidence card: baseline ladder (uniform, design (x segment) shares, CBC logit), shuffle control, paired CIs,
#    and the sealed persona vs no-persona comparison (does the persona text carry signal at all?).
report = se.evidence.evaluate(frame, title="pricing CBC", ours="stack", raw="llm", task="multiclass",
                              primary="log_loss", baselines_oof=["mnl"], B=a.B, prereg=reg, multiplier=False,
                              compare=[("llm", "llm_nopersona", "llm − llm_nopersona (< 0: the persona text helps)")])
print("\n" + str((s := se.evidence.summary(report)).get("headline_en") or s["headline"]))
EN = {"taban_rastgele": "uniform", "taban_marjinal": "design shares", "taban_demografi": "design x segment shares"}
for r in s["ladder"]:                                   # short English names; card.md has the full labels
    name = EN.get(r["column"]) or str(r["column"]).replace("kontrol_karistir_", "shuffled ").replace("tahmin_", "")
    print(f"  {name:34} {r['value']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
for c in s.get("comparisons") or []:
    print(f"  {c['label']}: {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] -> {c['verdict']}")
print("badge:", s["badge"], "| audits with warnings:",
      [d.get("name") or d["check"] for d in s["audits"] if d["status"] == "warning"])
se.evidence.card(report, out / "card.md", lang="en")
se.evidence.dashboard([report], out / "dashboard.html")
print(f"written: {out}/card.md, dashboard.html, price_curve.csv"
      + (f" | spend ${llm.ledger.total():.4f} of ${a.max_usd}" if a.live else ""))
