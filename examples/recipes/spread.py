"""Recipe: a message spreading on a social network / grid: who shares, how far it goes, which seeding strategy reaches
most people. An independent-cascade (IC) model is fitted on past cascades, the LLM predicts P(share | a friend shared
it) per (segment, message), the evidence card decides if the LLM adds anything. Guide: docs/recipes/spread.md.

    python3 examples/recipes/spread.py                    # offline: FakeLLM + synthetic 15x15 grid, ~30 s
    python3 examples/recipes/spread.py --data data/cascades.csv --message "..." --out runs/spread
    python3 examples/recipes/spread.py --live --max-usd 1 [--key-file <path>]   # Claude Haiku, $ANTHROPIC_API_KEY

--data CSV, one row per (cascade, node): EVERY node in every cascade, also those who never shared:
    cascade_id, message     one spread of one message (bootstrap cluster, never split) and the message text
    node, segment, persona  person id, segment, segment description for the LLM (first per segment; NO outcomes)
    neighbours              space-separated node ids this node passes messages to (followers; same in every cascade)
    seed, shared, step      0/1 organiser gave it at the start, 0/1 shared it, hop it was shared at (seed 0, else empty)
"""
import argparse
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # run from a checkout without pip install
import simengine as se                                              # noqa: E402

# 0. Day-of knobs: segments, messages and the campaign message are the things to replace with the brief's own.
SEGMENTS = {"student": "university student in a dorm, online all day, cares about rent and exams",
            "parent": "parent of two school kids, works full time, cares about school fees and safety",
            "retiree": "retired teacher, reads the news every morning, cares about pension and health",
            "local_media": "runs a local news page with many followers, posts anything newsworthy"}
TOPICS = {"rent": "Dorm rent goes up 40% next term, sign the petition", "school": "School fees rise again, parents "
          "meet on Monday", "pension": "Pension top-up form closes on Sunday", "health": "Free flu shots at the health "
          "centre for over-60s", "flood": "Flood warning tonight, move cars from the riverside"}
CAMPAIGN = "Rent support for students opens Friday: share so nobody misses it"   # message to seed (--message)
SYSTEM = ("You simulate one person on a social network. A friend just shared the post below. Estimate the "
          "probability that this person shares it too. Reply with JSON only: {\"p_share\": <0-1>, \"reason\": \"...\"}")


def fake_llm(system, user):
    """Stands in for an LLM that half-knows who cares (misses retirees on health) and says yes far too often;
    stable per-prompt noise from a hash."""
    persona, msg = user.removeprefix("Person: ").lower().split("\npost: ")
    hit = any(k in persona and k in f" {msg}" for k in (" rent", " school", " pension"))   # word starts: not "parent"
    p = .45 + .25 * hit + .15 * ("flood" in msg) + .1 * ("news" in persona) + zlib.crc32(user.encode()) % 100 / 1e3
    return json.dumps({"p_share": round(float(np.clip(p - .05, .02, .98)), 3)})


def truth_p(seg, msg):
    """Synthetic world only: P(share | exposed) per receiver segment; fans of the topic share 6x more."""
    topic = next((t for t in TOPICS if f" {t}" in f" {msg.lower()}"), None)
    fan = {"rent": "student", "school": "parent", "pension": "retiree", "health": "retiree"}.get(topic)
    return 0.05 + 0.25 * (fan == seg) + 0.15 * (topic == "flood") + 0.10 * (seg == "local_media")


def simulate(p, seeds, rng=np.random.default_rng(0), R=200):          # one shared stream unless rng is given
    """IC Monte Carlo on SRC -> DST, p[j] = P(j shares | reached); seeds (k,) or (R, k) -> (R, N) hop, -1 never."""
    hop, h = np.full((R, N), -1), 0
    hop[np.arange(R)[:, None], np.broadcast_to(np.atleast_2d(seeds), (R, np.atleast_2d(seeds).shape[1]))] = 0
    front = hop == 0
    while front.any():
        r, e = np.nonzero(front[:, SRC])
        ok, front, h = rng.random(len(e)) < p[DST[e]], np.zeros_like(front), h + 1
        front[r[ok], DST[e[ok]]] = True
        front &= hop < 0
        hop[front] = h
    return hop


def synthetic(side=15, n_casc=60, seed=0):
    """side x side grid of neighbours; local-media hubs also reach 8 random people; 3 random seeds per cascade."""
    global SRC, DST, N
    rs, N = np.random.default_rng(seed), side * side
    seg = rs.choice(list(SEGMENTS), N, p=[.3, .3, .3, .1])
    nb = [[j for j in (i - 1 if i % side else -1, i + 1 if (i + 1) % side else -1, i - side, i + side) if 0 <= j < N]
          + (list(rs.choice(N, 8, replace=False)) if seg[i] == "local_media" else []) for i in range(N)]
    SRC, DST = map(np.array, zip(*[(i, j) for i in range(N) for j in nb[i] if j != i]))
    rows, town = [], "Kadikoy Besiktas Uskudar Sisli Fatih Beykoz Sariyer Maltepe Kartal Pendik Bakirkoy Eyup".split()
    for c in range(n_casc):
        msg, sd = f"{list(TOPICS.values())[c % 5]} ({town[c % 12]})", rs.choice(N, 3, replace=False)
        hop = simulate(np.array([truth_p(s, msg) for s in seg]), sd, rs, R=1)[0]
        rows += [{"cascade_id": f"c{c:02d}", "message": msg, "node": f"n{i:03d}", "segment": seg[i],
                  "persona": SEGMENTS[seg[i]], "neighbours": " ".join(f"n{j:03d}" for j in nb[i]), "seed": int(i in sd),
                  "shared": int(hop[i] >= 0), "step": hop[i] if hop[i] >= 0 else None} for i in range(N)]
    return pd.DataFrame(rows)


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--data", help="your cascades CSV (columns in the module docstring)")
ap.add_argument("--message", default=CAMPAIGN, help="campaign message for the seeding comparison (step 8)")
ap.add_argument("--k", type=int, default=5, help="seed budget: how many people get the message first")
ap.add_argument("--live", action="store_true", help="use the Claude backend instead of FakeLLM")
ap.add_argument("--key-file", type=Path, help="file holding the API key (else $ANTHROPIC_API_KEY)")
ap.add_argument("--model", default="claude-haiku-5-5")
ap.add_argument("--max-usd", type=float, default=None, help="spend cap for --live (required with --live)")
ap.add_argument("--out", default="runs/spread", help="card, dashboard, seals, cache and ledger go here")
ap.add_argument("--B", type=int, default=200, help="bootstrap replicates (1000 for the final card)")
a = ap.parse_args()
(out := Path(a.out)).mkdir(parents=True, exist_ok=True)

# 1. Data + graph: your CSV or the synthetic grid. Split by cascade (a cascade's rows depend on each other).
df = synthetic() if not a.data else pd.read_csv(a.data, dtype={"node": str, "cascade_id": str})
if {"cascade_id", "message", "node", "segment", "persona", "neighbours", "seed", "shared", "step"} - set(df.columns):
    sys.exit("--data is missing columns; expected the ones listed in the module docstring")
nodes = df.drop_duplicates("node").reset_index(drop=True)
idx, N, seg_of = {n: i for i, n in enumerate(nodes.node)}, len(nodes), nodes.segment.to_numpy()
nb = nodes.neighbours.fillna("").astype(str).str.split()
SRC, DST = map(np.array, zip(*[(i, idx[j]) for i, js in enumerate(nb) for j in js if j in idx]))
df["i"] = df.node.map(idx)
df["split"] = ["test" if zlib.crc32(str(c).encode()) % 10 < 3 else "train" for c in df.cascade_id]

# 2. Exposures: k = in-neighbours that had shared before this node decided (the IC likelihood needs it).
for c, g in df.groupby("cascade_id"):
    hop = np.bincount(g.i, np.where(g.shared == 1, g.step.fillna(0), -1) + 1, minlength=N) - 1   # -1 = never
    lit = (hop[SRC] >= 0) & ((hop[SRC] < hop[DST]) | (hop[DST] < 0))                # sharer reached me first
    df.loc[g.index, "k"] = np.bincount(DST[lit], minlength=N)[g.i]

# 3. Backend: FakeLLM offline; --live = Claude with a hard budget (BudgetExceeded stops new calls).
try:                                                                # --live only; the key is never printed
    key = a.live and (a.key_file.expanduser().read_text().strip() if a.key_file else os.getenv("ANTHROPIC_API_KEY"))
except OSError as e:
    sys.exit(f"--key-file cannot be read ({type(e).__name__})")
if a.live and (a.max_usd is None or not key):                       # the ledger stops new calls at --max-usd
    sys.exit("--live needs --max-usd (e.g. --max-usd 1) and $ANTHROPIC_API_KEY or --key-file <path>")
llm = se.FakeLLM(fake_llm) if not a.live else se.llm.Claude(
    model=a.model, effort="low", thinking="off", max_tokens=200, cache_dir=out / ".cache", budget_usd=a.max_usd,
    api_key=key, tag="spread")

# 4. Ask: one prompt per (segment, message) cell, plus the no-persona control per message.
personas = df.groupby("segment").persona.first().to_dict()
msgs = list(dict.fromkeys(list(df.message) + [a.message]))
cells = [(s, m) for m in msgs for s in [*personas, None]]           # None = the no-persona control
jobs = [(SYSTEM, f"Person: {personas.get(s, 'a typical user of this network')}\nPost: {m}") for s, m in cells]
try:
    replies = llm.map(jobs, max_tokens=200, return_exceptions=True)   # a refusal fills its slot, not a crash
except se.llm.StopError as e:                                       # budget, bad key (401) or account spend limit
    sys.exit(f"stopped ({type(e).__name__}; answers so far are cached, a rerun resumes): {e}")
bad = [r for r in replies if isinstance(r, BaseException)]
pj = [None if isinstance(r, BaseException) else se.sim.parse_json(r.text) for r in replies]
ps = [float(np.clip(j.get("p_share", .5), .01, .99)) if isinstance(j, dict) else .5 for j in pj]   # unparsed -> .5
P_LLM = dict(zip(cells, ps))
print(f"LLM calls: {len(jobs)}; failed {len(bad)}, unparsed {sum(not isinstance(j, dict) for j in pj) - len(bad)}")
if len(bad) > len(jobs) // 4:                                       # a few refusals are data; most failing is a bug
    sys.exit(f"{len(bad)} of {len(jobs)} calls failed, first: {type(bad[0]).__name__}: {str(bad[0])[:200]}")

# 5. Fit on TRAIN exposures: IC global p, IC per-segment p, ours = segment p shifted by the calibrated LLM logit.
L, S = (lambda p: np.log(np.clip(p, 1e-3, .999) / np.clip(1 - p, 1e-3, .999))), (lambda z: 1 / (1 + np.exp(-z)))
ex = df[(df.split == "train") & (df.seed == 0) & (df.k > 0)]
ll = lambda p, d: np.sum(d.shared * np.log(1 - (1 - p) ** d.k + 1e-12) + (1 - d.shared) * d.k * np.log(1 - p))
p_glob = (grid := np.linspace(.005, .6, 120))[np.argmax([ll(p, ex) for p in grid])]
p_seg = {str(s): float(grid[np.argmax([ll(p, ex[ex.segment == s]) for p in grid])]) for s in personas}
zs, P_SEG = L(ex.segment.map(p_seg).to_numpy()), np.array([p_seg[s] for s in seg_of])


def shifted(score, name):                                           # score(segment, message) -> LLM P(share)
    """S(logit segment p + c + b (logit score - train mean)), c, b by likelihood; c = b = 0 IS ic_segment."""
    z = L(np.array([score(s, m) for s, m in zip(ex.segment, ex.message)]))
    c, b = max(((c, b) for c in np.linspace(-2, 2, 41) for b in np.linspace(0, 2, 21)),
               key=lambda cb: ll(S(zs + cb[0] + cb[1] * (z - z.mean())), ex))
    print(f"fitted {name}: shift {c:+.2f}, LLM slope {b:.2f}")
    return lambda m: S(L(P_SEG) + c + b * (L(np.array([score(s, m) for s in seg_of])) - z.mean()))


print(f"fitted: p_global {p_glob:.3f}, p_segment {({s: round(v, 3) for s, v in p_seg.items()})}")
METHODS = {"ic_global": lambda m: np.full(N, p_glob), "ic_segment": lambda m: P_SEG,      # LLM-free baselines
           "llm_nopersona": lambda m: np.full(N, P_LLM[None, m]),                         # raw LLM scores
           "llm": lambda m: np.array([P_LLM[s, m] for s in seg_of]),
           "llm_nopersona_calibrated": shifted(lambda s, m: P_LLM[None, m], "no-persona"),  # persona control
           "llm_calibrated": shifted(lambda s, m: P_LLM[s, m], "ours")}

# 6. Predict who shares in every cascade from its seeds (Monte Carlo IC), then seal the test rows BEFORE scoring.
for c_id, g in df.groupby("cascade_id"):        # same seed per cascade for every method; R=500 (at 200 MC noise alone
    sd, rs = g.i[g.seed == 1], zlib.crc32(str(c_id).encode())                       # moved baseline AUCs by 0.03-0.06)
    for name, f in METHODS.items():
        hop = simulate(f(g.message.iloc[0]), sd, np.random.default_rng(rs), 500)
        df.loc[g.index, "pred_" + name] = (hop >= 0).mean(0)[g.i]
frame = df[df.seed == 0].rename(columns={"shared": "truth", "cascade_id": "group"})
frame = frame[["truth", "group", "segment", "split"] + ["pred_" + m for m in METHODS]]
se.seal(frame[frame.split == "test"].drop(columns="truth"), "spread; ours: llm_calibrated", registry=out / ".onkayit")

# 7. Evidence card: IC baselines, calibrated no-persona control, shuffle control, cascade-bootstrap CIs, audits.
sz = frame[frame.split == "test"].groupby("group").sum(numeric_only=True)
sz["pred_train_mean"] = frame[frame.split == "train"].groupby("group").truth.sum().mean()   # constant, LLM-free
print(f"\ntest cascade size (non-seed sharers, observed mean {sz.truth.mean():.1f}): predicted mean | abs error")
print("\n".join(f"  {m[5:]:24} {v.mean():6.1f} | {(v - sz.truth).abs().mean():.1f}" for m, v in sz.iloc[:, 1:].items()))
report = se.evidence.evaluate(frame, "spread: who shares", "llm_calibrated", "llm", primary="auc", B=a.B,
                              baselines_oof=["ic_global", "ic_segment"], prereg=out / ".onkayit", multiplier=False)
s = se.evidence.summary(report)
print("\n" + str(s["headline_en"]))
print("\n".join(f"  {r['column']!s:34} {r['value']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]" for r in s["ladder"]))
print("badge:", s["badge"], "| audits with warnings:", [d["check"] for d in s["audits"] if d["status"] == "warning"])
se.evidence.card(report, out / "card.md", lang="en")
se.evidence.dashboard([report], out / "dashboard.html")

# 8. Seeding: k seeds for the campaign under the fitted model (synthetic: also the true world). Add strategies here.
deg, R, p_ours, G = np.bincount(SRC, minlength=N), 1000, METHODS["llm_calibrated"](a.message), []
reach = lambda p, sd, r=R: (simulate(p, sd, np.random.default_rng(1), r) >= 0).sum(1)   # same draws per strategy
fans = seg_of == max(personas, key=lambda q: P_LLM[q, a.message])                        # segment the LLM rates top
plan = {"random": np.argsort(np.random.default_rng(2).random((R, N)), 1)[:, :a.k], "top_degree": np.argsort(-deg)[:a.k],
        "fans_by_degree": np.argsort(-(deg + 1000 * fans))[:a.k], "greedy_model": G}
for _ in range(a.k):                                                # greedy on the model over the top-30 hubs
    G.append(max((j for j in np.argsort(-deg)[:30] if j not in G), key=lambda j: reach(p_ours, G + [j], 100).mean()))
true_p = None if a.data else np.array([truth_p(q, a.message) for q in seg_of])
ci = lambda v: f"{v.mean():6.1f} +- {1.96 * v.std() / R ** .5:4.1f}"
print(f"\nseeding k={a.k} for {a.message!r}: people reached, mean +- Monte Carlo 95% CI over {R} runs (sim noise only)")
for name, sd in plan.items():
    x, tr = reach(p_ours, sd), f"   true world {ci(reach(true_p, sd))}" if true_p is not None else ""
    print(f"  {name:15} model {ci(x)}  minus random {ci(x - reach(p_ours, plan['random']))}{tr}")
print(f"written: {out}/card.md, dashboard.html" + (f" | spend ${llm.ledger.total():.4f}" if a.live else ""))
