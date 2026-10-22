"""Recipe: cold start. Predict the rating (or early demand) of a brand-new item with no history: an LLM prior from
the item's own text, blended with what similar past items scored; the evidence card decides whether the LLM added
anything over the similar-item baseline.

    python3 examples/recipes/cold_start.py                                # offline: FakeLLM + synthetic catalogue
    python3 examples/recipes/cold_start.py --data data/items.csv --out runs/cold   # add --metric wmape for demand
    python3 examples/recipes/cold_start.py --live --max-usd 1             # Claude Haiku, key from $ANTHROPIC_API_KEY
    python3 examples/recipes/cold_start.py --live --key-file <path> --max-usd 1

--data CSV, one row per item (past items have a target; the brand-new items to predict leave it EMPTY):
    item_id      str    item id (bootstrap cluster)
    title        str    product name as a shopper sees it (hidden in the blind control arm)
    category     str    category, e.g. headphones
    description  str    the item's own text: features, materials, claims (NO reviews, ratings or sales in it)
    price        float  optional; shown to the LLM only
    launch_date  str    optional ISO date; when present the latest 30 % of launches are the test split (else 30 % of
                        items by a hash of item_id); also enables --cutoff (memorisation check by date)
    target       float  mean star rating (1-5) or units sold in the first N weeks; empty = new item to predict
Guide, pitfalls and the literature behind each step: docs/recipes/cold_start.md.
"""
import argparse
import re
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # run from a checkout without pip install
import simengine as se                                              # noqa: E402

warnings.filterwarnings("ignore", message="An input array is constant")   # the constant rungs of the ladder
# 0. Day-of knobs: the question, the scale and the launch items are what you replace with the brief's own.
SYSTEM = ("You estimate how shoppers will rate a product that has just launched and has no reviews yet. Read the "
          "product and reply with JSON only: {\"rating\": <average star rating, 1-5>, \"reason\": \"...\"}")
KEY = "rating"                                                       # JSON key to read (e.g. "units" for demand)
NEW_ITEMS = pd.DataFrame([("new1", "Orbit Buds", "headphones", 79, "aluminium; app required"),   # synthetic mode's
                          ("new2", "QuickPress", "kitchen", 45, "dishwasher-safe; plastic hinge")],  # launch items
                         columns=["item_id", "title", "category", "price", "description"])  # --data: empty-target rows
CATS = {"headphones": 0.0, "kitchen": 0.15, "fitness": -0.1, "toys": -0.2}
TRAITS = {"aluminium": 0.25, "two-year warranty": 0.3, "replaceable battery": 0.2, "dishwasher-safe": 0.2, "compact": 0,
          "plastic hinge": -0.35, "app required": -0.3, "subscription": -0.4, "one-size": -0.2, "gift box": 0.05}
EN = {"tahmin_": "", "taban_": "baseline_", "kontrol_karistir_": "shuffled_", "rastgele": "random", "marjinal": "mean"}


def synthetic(n=240, seed=0):
    """Ground truth: star rating = category level + trait effects + a latent quality nobody sees. The fake LLM
    below knows half of the traits (and overstates them) and rates everything ~0.5 star too high."""
    rs, rows, names = np.random.default_rng(seed), [], ["Nova", "Pulse", "Terra", "Vega", "Kite", "Lumo"]
    for i in range(n):
        cat, tr = rs.choice(list(CATS)), list(rs.choice(list(TRAITS), 3, replace=False))
        z = 3.7 + CATS[cat] + sum(TRAITS[t] for t in tr) + rs.normal(0, 0.3)
        rows.append({"item_id": f"i{i:04d}", "title": f"{rs.choice(names)} {cat[:4].title()} {i}", "category": cat,
                     "price": float(rs.integers(15, 200)), "description": "; ".join(tr),
                     "launch_date": f"2026-{1 + i * 9 // n:02d}-{1 + i % 28:02d}", "target": float(np.clip(z, 1, 5))})
    return pd.concat([pd.DataFrame(rows), NEW_ITEMS], ignore_index=True)


def fake_llm(system, user):
    """Stands in for an LLM: +0.5 star optimism, right sign on 4 traits (x1.5), blind to the other 4."""
    r = 4.2 + 1.5 * sum(TRAITS[t] for t in ("aluminium", "two-year warranty", "app required", "subscription")
                        if t in user.lower())
    r += (zlib.crc32(user.encode()) % 100 - 50) / 250                      # stable per-prompt noise
    return '{"%s": %.2f, "reason": "fake"}' % (KEY, np.clip(r, 1, 5))


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--data", help="your item CSV (columns in the module docstring)")
ap.add_argument("--metric", default="mae", help="primary metric: mae (ratings) | wmape (demand) | spearman")
ap.add_argument("--k", type=int, default=15, help="neighbours in the similar-item baseline")
ap.add_argument("--cutoff", help="LLM training cutoff (ISO date): metric before vs after it; needs launch_date")
ap.add_argument("--live", action="store_true", help="use the Claude backend instead of FakeLLM")
ap.add_argument("--key-file", help="file holding the API key (else $ANTHROPIC_API_KEY)")
ap.add_argument("--model", default="claude-haiku-5-5")
ap.add_argument("--max-usd", type=float, default=None, help="spend cap for --live (required with --live)")
ap.add_argument("--out", default="runs/cold_start", help="card, dashboard, seals, cache and ledger go here")
ap.add_argument("--B", type=int, default=300, help="bootstrap replicates (1000 for the final card)")
a = ap.parse_args()
(out := Path(a.out)).mkdir(parents=True, exist_ok=True)

# 1. Data: your CSV or the synthetic catalogue. Rows without a target are the new items; the rest are split.
df = pd.read_csv(a.data) if a.data else synthetic()
if miss := {"item_id", "title", "category", "description", "target"} - set(df.columns):
    sys.exit(f"--data is missing columns: {sorted(miss)}")
df["price"] = df.price if "price" in df else np.nan
if "launch_date" in df:                                              # time split: the newest launches are the test
    t = pd.to_datetime(df.launch_date)                               # a row without a date stays in train
    df["split"] = np.where(t > t[df.target.notna()].quantile(0.7), "test", "train")
else:                                                                # else an item-disjoint hash split
    df["split"] = ["test" if zlib.crc32(str(v).encode()) % 10 < 3 else "train" for v in df.item_id]
df.loc[df.target.isna(), "split"] = "new"
tr, te, new = (np.flatnonzero(df.split == s) for s in ("train", "test", "new"))
y = df.target.to_numpy(float)

# 2. Backend: FakeLLM offline; --live = Claude with a hard budget (BudgetExceeded stops new calls).
if a.live:
    if a.max_usd is None:
        sys.exit("--live needs --max-usd (e.g. --max-usd 1); the ledger stops new calls at that total")
    key = Path(a.key_file).expanduser().read_text().strip() if a.key_file else None   # never printed
    llm = se.llm.Claude(model=a.model, effort="low", thinking="off", max_tokens=120, cache_dir=out / ".cache",
                        budget_usd=a.max_usd, api_key=key, tag="cold_start")
else:
    llm = se.FakeLLM(fake_llm)

# 3. LLM prior per item, plus the BLIND arm (name hidden): if the gain disappears, the LLM recognised the items.
price = lambda p: "" if pd.isna(p) else f"\nPrice: {p:.0f}"
ask = lambda r, name: (SYSTEM, f"Product: {name}\nCategory: {r.category}{price(r.price)}\nDescription: {r.description}")
jobs = [ask(r, r.title) for r in df.itertuples()] + [ask(r, "(name hidden)") for r in df.itertuples()]
try:                                                                 # a refused / failed call -> unparsed, not a crash
    replies = llm.map(jobs, max_tokens=120, **({"return_exceptions": True} if a.live else {}))
except se.llm.StopError as e:                                        # budget cap, missing / bad key, account limit
    sys.exit(f"stopped by the budget: {e}" if isinstance(e, se.llm.BudgetExceeded) else f"stopped: {e}")
texts = ["" if isinstance(r, Exception) else r.text for r in replies]


def num(text):
    """KEY from a reply: the JSON value, else a `"key": 4.2` / `"key": "4.2"` pattern (quoted or cut-off replies)."""
    j = se.sim.parse_json(text)
    if isinstance(j, dict) and isinstance(j.get(KEY), (int, float)) and not isinstance(j.get(KEY), bool):
        return float(j[KEY])
    m = re.search(r'"%s"\s*:\s*"?(-?\d+(?:\.\d+)?)' % re.escape(KEY), text)
    return float(m.group(1)) if m else np.nan


v = np.array([num(t) for t in texts])
print(f"LLM calls: {len(jobs)} ({len(df)} items x named + blind); unparsed or failed: {int(np.isnan(v).sum())}")
if np.isnan(v).mean() > 0.2:                                         # a card on imputed medians would be noise
    sys.exit(f"{int(np.isnan(v).sum())} of {len(v)} replies have no numeric {KEY!r}; fix SYSTEM / KEY. "
             f"First one: {next(t for t, x in zip(texts, v) if np.isnan(x))[:200]!r}")
df["llm"], df["llm_blind"] = np.where(np.isnan(v), np.nanmedian(v), v).reshape(2, -1)   # unparsed -> median prior

# 4. Similar-item baselines from labelled items only (train rows cross-fitted: no item sees its own target):
#    category mean, and k nearest neighbours on TF-IDF of category + description (cosine, similarity-weighted).
tok = [re.findall(r"[a-z0-9]+", f"{c} {d}".lower()) for c, d in zip(df.category, df.description)]
vocab = {w: j for j, w in enumerate(sorted({w for t in tok for w in t}))}
M = np.array([[t.count(w) for w in vocab] for t in tok], float)          # term counts per item
M *= np.log((1 + len(tr)) / (1 + (M[tr] > 0).sum(0))) + 1                    # idf from train items only
M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
fold = np.array([zlib.crc32(f"fold{v}".encode()) % 5 for v in df.item_id])   # salted: independent of the hash split


def from_pool(rows, pool):
    """kNN and category-mean predictions for `rows` from the targets of `pool` only."""
    S = M[rows] @ M[pool].T
    top = np.argsort(-S, axis=1)[:, :a.k]
    w = np.take_along_axis(S, top, 1) + 1e-6
    cm = df.iloc[pool].groupby("category").target.mean()
    return np.column_stack([(w * y[pool][top]).sum(1) / w.sum(1),
                            df.category.iloc[rows].map(cm).fillna(y[pool].mean()).to_numpy()])


df["knn"], df["cat_mean"] = np.nan, np.nan
for k in range(5):                                                   # cross-fit the train rows
    df.loc[df.index[tr[fold[tr] == k]], ["knn", "cat_mean"]] = from_pool(tr[fold[tr] == k], tr[fold[tr] != k])
df.loc[df.index[te], ["knn", "cat_mean"]] = from_pool(te, tr)


# 5. Blend: ridge of target on [1, knn, cat_mean, llm] fitted on train. Intercept + slope absorb the LLM's optimism;
#    a small LLM weight = it adds little beyond similar items. The same fit on llm_blind gives the blind twin, and
#    without the LLM the LLM-free twin: ours - stack_free is what the LLM adds (blending two baselines alone helps).
def ridge(cols, rows, lam=1.0):
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])
    P = lam * np.diag([0.0] + [1.0] * len(cols))                      # do not shrink the intercept
    w = np.linalg.solve(X[rows].T @ X[rows] + P, X[rows].T @ y[rows])
    return X @ w, w


def blend(cols):
    """Ridge on all train rows for test (and later new) rows; train rows get cross-fitted predictions."""
    p, w = ridge(cols, tr)
    for k in range(5):
        rows = tr[fold[tr] == k]
        p[rows] = ridge(cols, tr[fold[tr] != k])[0][rows]
    return p, w


df["stack"], w = blend(["knn", "cat_mean", "llm"])
df["stack_blind"], df["stack_free"] = blend(["knn", "cat_mean", "llm_blind"])[0], blend(["knn", "cat_mean"])[0]
print(f"optimism: raw LLM mean {df.llm.iloc[tr].mean():.2f} vs observed {y[tr].mean():.2f}; "
      f"weights knn {w[1]:.2f}, cat_mean {w[2]:.2f}, llm {w[3]:.2f}")

# 6. Seal the test predictions BEFORE the truth is joined (pre-registration; the card shows the seal).
frame = pd.DataFrame({"truth": df.target, "group": df.item_id, "split": df.split, "time": df.get("launch_date"),
                      "pred_cat_mean": df.cat_mean, "pred_knn": df.knn, "pred_llm": df.llm,
                      "pred_stack_free": df.stack_free, "pred_stack": df["stack"],
                      "pred_stack_blind": df["stack_blind"]})[df.split != "new"]
registry = out / ".onkayit"
se.seal(frame[frame.split == "test"].drop(columns="truth"), "cold_start test predictions; ours = stack, raw = llm",
        registry=registry)

# 7. The launch question: new items from ALL labelled items, 90 % interval from the test residuals.
if len(new):
    lab = np.flatnonzero(df.split != "new")
    q = np.quantile(y[te] - df["stack"].to_numpy()[te], [0.05, 0.95])
    df.loc[df.index[new], ["knn", "cat_mean"]] = from_pool(new, lab)
    p_new = ridge(["knn", "cat_mean", "llm"], lab)[0][new]
    print("\nnew item        prediction [90% interval]   raw llm   knn")
    for i, p in zip(new, p_new):
        print(f"{str(df.title.iloc[i])[:15]:15} {p:6.2f} [{p + q[0]:.2f}, {p + q[1]:.2f}]      "
              f"{df.llm.iloc[i]:5.2f}  {df.knn.iloc[i]:5.2f}")

# 8. Evidence card: LLM-free ladder (cat_mean, knn, stack_free), shuffle control, blind arm, paired CIs, audits.
report = se.evidence.evaluate(frame, title="cold_start new-item ratings", ours="stack", raw="llm", task="continuous",
                              primary=a.metric, baselines_oof=["cat_mean", "knn", "stack_free"], blind_col="pred_stack_blind",
                              B=a.B, prereg=registry, cutoff=a.cutoff, multiplier=False)
s = se.evidence.summary(report)
print("\n" + str(s["headline_en"] or s["headline"]))
for r in s["ladder"]:
    name = re.sub("|".join(EN), lambda m: EN[m.group()], str(r["column"] or r["label"]))
    print(f"  {name:34} {r['value']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
for c in s.get("comparisons") or []:                                 # the blind arm: ours - ours without names
    print(f"  stack - stack_blind: {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] ({c['metric']}; {c['verdict']})")
print("badge:", s["badge"], "| audits with warnings:", [d["name"] or d["check"] for d in s["audits"] if d["status"] == "warning"])
se.evidence.card(report, out / "card.md", lang="en")
se.evidence.dashboard([report], out / "dashboard.html")
cost = f" | spend this run ${llm.spent_usd:.4f}, ledger ${llm.ledger.total():.4f} of ${a.max_usd}" if a.live else ""
print(f"written: {out / 'card.md'}, {out / 'dashboard.html'}{cost}")
