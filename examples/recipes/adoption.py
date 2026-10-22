"""Recipe: new product adoption forecast. How many will have adopted N periods from now, and what does more
marketing buy? Curve baselines (naive, linear, Gompertz, Bass) and analogy-prior Bass / ABM (ABC) forecasts are
backtested on past markets; the LLM reads each market's description and proposes a Bass prior (p, q, m); the
evidence card decides. Marketing what-ifs re-simulate the fitted ABM with a higher innovation (external) rate.

    python3 examples/recipes/adoption.py                          # offline: FakeLLM + 10 synthetic markets, ~15 s
    python3 examples/recipes/adoption.py --data data/adoption.csv --target m07 --out runs/adoption
    python3 examples/recipes/adoption.py --live --max-usd 0.5      # Claude Haiku, key from $ANTHROPIC_API_KEY
    python3 examples/recipes/adoption.py --live --key-file <path> --max-usd 0.5

--data CSV, one row per (market, period):
    series       str    market / country / product id: one cumulative adoption curve each (the bootstrap cluster)
    t            float  period (year or quarter, equal spacing; monthly data: aggregate to quarters first)
    y            float  cumulative adopters as % of the population (0-100): the forecast target
    description  str    optional: market + product text for the LLM (first value per series). NO adoption numbers,
                        brand names or launch years in it: the LLM may remember the real curve
    spend        float  optional: marketing spend per period (one currency); used for the what-if elasticity
Use series of ONE kind (one product across markets): the analogy prior hurts across unrelated technologies.
Guide, pitfalls and the literature behind each step: docs/recipes/adoption.md.
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
import simengine.abm as abm                                         # noqa: E402

# 0. Day-of knobs: product text, horizon, cutoffs and the spend what-ifs are the things to replace with the brief's.
PRODUCT = "a mobile payment wallet: pay in shops with the phone, free to install, needs a bank card"
HORIZON, CUTOFFS = 8, (3, 5)          # periods ahead; backtest cutoffs = periods after the series crossed 1 %
SPEND_WHATIFS = (0.5, 1.5, 2.0)       # marketing spend multipliers tried on the target market
SPEND_ELASTICITY = 0.5                # assumed: spend x k -> innovation p x k^0.5; put a measured value here
OURS = "blend"                        # named BEFORE looking at results: mean of abm_analogy and llm_bass
COMPARE = [(OURS, "llm_generic"), ("abm_analogy", "bass_abc"), ("llm_bass", "bass")]   # registered in the seal
SYSTEM = ("You are a diffusion-of-innovations analyst. Given a product and a market, give Bass model parameters per "
          "period: p (innovation, 0.001-0.05), q (imitation, 0.05-0.8) and m (long-run ceiling, % of population). "
          "Reply with JSON only: {\"p\": <num>, \"q\": <num>, \"m\": <num>, \"reason\": \"...\"}")


def synthetic(n=10, T=26, seed=0):
    """Ground truth: young / metro markets spread faster by word of mouth (q), spend drives p (elasticity 0.5),
    cash-heavy shops cap the ceiling m (the fake LLM below does not know the last one)."""
    rs, rows = np.random.default_rng(seed), []
    for i in range(n):
        metro, young, cash, spend = rs.random() < .5, rs.random() < .5, rs.random() < .4, float(rs.choice([1, 2, 4]))
        m, p, q = 75 - 30 * cash, 0.004 * spend ** 0.5, 0.25 + 0.15 * young + 0.10 * metro
        desc = (f"{'large metro' if metro else 'mid-size city'}, {'young' if young else 'ageing'} population, "
                f"{'cash-heavy' if cash else 'card-heavy'} shops, marketing budget level {spend:g}")
        t = np.arange(T, dtype=float)
        tau = float(rs.integers(-4, 6)) if i < n - 1 else T - 9.0     # the last market launched recently
        y = abm.bass_curve(t, m, p, q, tau) + rs.normal(0, 0.4, T)
        rows += [dict(series=f"m{i:02d}", t=tt, y=float(np.clip(yy, 0, 100)), description=desc, spend=spend)
                 for tt, yy in zip(t, y)]
    return pd.DataFrame(rows)


def fake_analyst(system, user):
    """Stands in for an LLM that half-knows diffusion (word of mouth and budget yes, the cash ceiling no)."""
    lvl = re.search(r"budget level (\d+)", user)
    p = 0.003 * (float(lvl.group(1)) if lvl else 2.0) ** 0.5
    q = 0.28 + 0.12 * ("young" in user) + 0.08 * ("metro" in user)
    q *= 1 + (zlib.crc32(user.encode()) % 100 - 50) / 250                    # stable per-prompt noise (+-20 %)
    return json.dumps({"p": round(p, 4), "q": round(q, 3), "m": 65, "reason": "fake"})


class LLMBass:
    """Backtest method from the LLM's per-series Bass guess: the LLM fixes the shape (p, q); the start time and
    ceiling m are fitted on t <= cutoff (m pulled to the LLM's m by `weight`). raw=True: only the start is fitted."""
    def __init__(self, priors, raw=False, weight=1.0):
        self.priors, self.raw, self.weight = priors, raw, weight

    def fit(self, history, series, cutoff):
        g, pr = history[history.series == series].sort_values("t"), self.priors[series]
        t, y, best = g.t.to_numpy(float), g.y.to_numpy(float), (np.inf, 0.0, pr["m"])
        for tau in np.arange(t[0] - 30, t[-1], 0.25):                 # grid over the start; m in closed form
            f = abm.bass_curve(t, 1.0, pr["p"], pr["q"], tau)
            m = pr["m"] if self.raw else np.clip((f @ y + self.weight * pr["m"]) / (f @ f + self.weight), y.max(), 100)
            best = min(best, (((m * f - y) ** 2).sum(), tau, m))
        return dict(series=series, cutoff=cutoff, tau=best[1], m=best[2], **{k: pr[k] for k in "pq"})

    def predict(self, model, horizons):
        h, (m, p, q, tau) = np.asarray(horizons, float), (model[k] for k in ("m", "p", "q", "tau"))
        return pd.DataFrame(dict(horizon=h, y_pred=abm.bass_curve(model["cutoff"] + h, m, p, q, tau)))


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--data", help="your adoption CSV (columns in the module docstring)")
ap.add_argument("--target", help="series for the marketing what-ifs (default: the most recent launch)")
ap.add_argument("--live", action="store_true", help="use the Claude backend instead of FakeLLM")
ap.add_argument("--key-file", help="file holding the API key (else $ANTHROPIC_API_KEY)")
ap.add_argument("--model", default="claude-haiku-5-5")
ap.add_argument("--max-usd", type=float, default=None, help="spend cap for --live (required with --live)")
ap.add_argument("--out", default="runs/adoption", help="card, dashboard, seals, cache and ledger go here")
ap.add_argument("--bank", type=int, default=300, help="ABC bank: sims = agents (2000 = the measured default, slow)")
ap.add_argument("--B", type=int, default=300, help="bootstrap replicates (1000 for the final card)")
a = ap.parse_args()
(out := Path(a.out)).mkdir(parents=True, exist_ok=True)

# 1. Data: your CSV or the synthetic markets. y must be cumulative % adopters; check the units before anything else.
df = pd.read_csv(a.data) if a.data else synthetic()
if miss := {"series", "t", "y"} - set(df.columns):
    sys.exit(f"--data is missing columns: {sorted(miss)}")
df["series"] = df.series.astype(str)
df["description"] = df.get("description", "an unnamed market")       # no text: the LLM prior is uninformative
desc = df.groupby("series").description.first().to_dict()

# 2. Backend: FakeLLM offline; --live = Claude with a hard budget (BudgetExceeded stops new calls).
if a.live:
    if a.max_usd is None:
        sys.exit("--live needs --max-usd (e.g. --max-usd 0.5); the ledger stops new calls at that total")
    key = Path(a.key_file).expanduser().read_text().strip() if a.key_file else None   # never printed
    llm = se.llm.Claude(model=a.model, effort="low", thinking="off", max_tokens=200, cache_dir=out / ".cache",
                        budget_usd=a.max_usd, api_key=key, tag="adoption")
else:
    llm = se.FakeLLM(fake_analyst)

# 3. Ask: one Bass guess per market from its description, plus the generic-market control (does the text matter?).
ask = lambda market: (SYSTEM, f"Product: {PRODUCT}\nMarket: {market}")
names = sorted(desc)
try:
    replies = llm.map([ask(desc[s]) for s in names] + [ask("a typical market in this country")], max_tokens=200)
except (se.llm.BudgetExceeded, se.llm.AuthError) as e:          # budget hit / no key: no further calls
    sys.exit(f"stopped before the LLM step finished: {e}")

def bass_of(r):
    j = se.sim.parse_json(r.text)
    j = j if isinstance(j, dict) else {}                              # unparsed -> mid-prior defaults
    num = lambda k, d, lo, hi: float(np.clip(pd.to_numeric(j.get(k, d), errors="coerce") or d, lo, hi))
    return dict(p=num("p", 0.005, 1e-4, 0.2), q=num("q", 0.3, 0.01, 2.0), m=num("m", 50, 1, 100))

guess = [bass_of(r) for r in replies]
priors, generic = dict(zip(names, guess)), {s: guess[-1] for s in names}
print(f"LLM calls: {len(replies)} ({len(names)} markets + 1 generic control)")

# 4. Backtest: every method sees only t <= cutoff (all series), forecasts the next HORIZON periods.
small = dict(n_sim=a.bank, n_agents=a.bank, seed=0)
methods = {"naive": abm.CurveBaseline("naive"), "linear": abm.CurveBaseline("linear"),
           "gompertz": abm.CurveBaseline("gompertz"), "bass": abm.CurveBaseline("bass"),
           "bass_analogy": abm.BassAnalogy(),                         # classic: other markets' median (p, q) as prior
           "abm_analogy": abm.AdoptionABCAnalogy(**small),            # ABM + ABC with the analogy prior
           "bass_abc": abm.BassABCAnalogy(**small),                   # control: same ABC, Bass simulator (no network)
           "llm_only": LLMBass(priors, raw=True), "llm_generic": LLMBass(generic), "llm_bass": LLMBass(priors)}
frame = abm.adoption_backtest(df, methods, cutoffs=CUTOFFS, horizon=HORIZON, on_ineligible="nan")
frame = frame.drop(columns=[c for c in frame.columns if c.startswith(("lo80_", "hi80_"))])
for m in ("abm_analogy", "bass_abc"):                                 # series ABC cannot align: Bass-analogy fallback
    frame[f"pred_{m}"] = frame[f"pred_{m}"].fillna(frame.pred_bass_analogy)
frame["pred_blend"] = (frame.pred_abm_analogy + frame.pred_llm_bass) / 2
print(f"backtest rows: {len(frame)} | series {frame.series.nunique()} | ABC fallback: {frame.attrs['n_ineligible']}")

# 5. Seal the forecasts BEFORE scoring (pre-registration; the card shows the seal and that OURS was named first).
registry = out / ".onkayit"
se.seal(frame.drop(columns="truth"), f"adoption backtest forecasts; ours={OURS}; compare: "
        + ", ".join(f"{x} vs {y}" for x, y in COMPARE), registry=registry)
for c in sorted(c for c in frame.columns if c.startswith("pred_")):
    print(f"  MAE {c[5:]:13} {np.abs(frame[c] - frame.truth).mean():6.2f} pp")

# 6. Marketing what-ifs on the target market: spend x k -> innovation p x k^elasticity (model-implied, not causal).
t0 = abm.adoption_t0(df)                                              # first period at >= 1 % per series
target = a.target or max(t0, key=t0.get)                              # default: the most recent launch
if "spend" in df and df.groupby("series").spend.mean().nunique() >= 3:   # data check only: confounded by market
    fit_p = {s: abm.bass(g.t.to_numpy(float), g.y.to_numpy(float), [0.0], return_params=True)[1]
             for s, g in df.sort_values("t").groupby("series") if s in t0}
    sp = df.groupby("series").spend.mean()[[s for s, f in fit_p.items() if f]]
    el = np.polyfit(np.log(sp), np.log([fit_p[s]["p"] for s in sp.index]), 1)[0]
    print(f"\ncross-market spend elasticity of p (check only, confounded): {el:.2f}; used: {SPEND_ELASTICITY}")
cut, abm_m = df[df.series == target].t.max(), methods["abm_analogy"]
fit = abm_m.fit(df, target, cut)                                      # all data up to today; HORIZON <= 10 here
print(f"what-ifs for {target} (median adoption % at +{HORIZON // 2} / +{HORIZON} periods, 80 % band at +{HORIZON}):")
for k in SPEND_WHATIFS:
    sc = abm_m.scenario(fit, {"innovation": k ** SPEND_ELASTICITY}, n_draws=100)
    for lab, g in ([("x1 (now)", sc[sc.scenario == "baseline"])] if k == SPEND_WHATIFS[0] else []) + \
            [(f"x{k:g}", sc[sc.scenario != "baseline"])]:
        at = lambda h: g.iloc[int((g.t - cut - h).abs().argmin())]    # nearest simulated period
        print(f"  spend {lab:9} {at(HORIZON // 2).y_pred:6.1f} {at(HORIZON).y_pred:6.1f} "
              f"[{at(HORIZON).lo80:.1f}, {at(HORIZON).hi80:.1f}]")
print("adopter types (model-implied):", [(p["name"], p["share"]) for p in abm_m.personas(fit)])

# 7. Evidence card: baseline ladder, controls (generic-text LLM, Bass-simulator ABC), paired CIs, audits.
report = se.evidence.evaluate(frame, title="adoption backtest", ours=OURS, raw="llm_only", task="forecast",
                              unlabeled_baselines=["naive", "linear", "gompertz", "bass", "bass_analogy"],
                              compare=COMPARE, primary="mae", split="test", B=a.B, prereg=registry, ours_uses_llm=True)
s = se.evidence.summary(report)
print("\n" + str(s["headline"]))
for r in s["ladder"]:
    print(f"  {str(r['label'] or r['column']):34} {r['value']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")
for c in s.get("comparisons") or []:
    print(f"  {c['a'][7:]} - {c['b'][7:]}: {c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['verdict']}")
print("badge:", s["badge"], "| audits with warnings:", [d["check"] for d in s["audits"] if d["status"] == "warning"])
se.evidence.card(report, out / "card.md", lang="en")
se.evidence.dashboard([report], out / "dashboard.html")
print(f"written: {out / 'card.md'}, {out / 'dashboard.html'}"
      + (f" | spend ${llm.ledger.total():.4f} of ${a.max_usd}" if a.live else ""))
