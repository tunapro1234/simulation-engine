"""Scoring layer for game results: per-cell moments, human reference moments, distance to human with an
agent-bootstrap CI, and an EvalFrame bridge to `simengine.evidence.evaluate`.

b01 produced 10,920 result rows but no distance-to-human scoring; this module is that missing layer (it is
simengine's own computation, not a number from the source repo). Human values are small published-literature
constants inlined from the GAME `references` (each with its citation); they are parameters, not data.

Rules: moments use valid rows only (invalid rows are counted in n_rows - n_valid, never filled in). A moment that
cannot be computed (no valid rows, no complete schedule) is NaN and stays NaN.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from . import library as L

CELL = ["model", "persona_method", "game"]


# ---------------------------------------------------------------- helpers

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _cparam(game: str, key: str) -> dict:
    return {str(c["id"]): c.get(key) for c in L.GAMES[game]["conditions"]}


def _mean(x) -> float:
    v = np.asarray(x, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else float("nan")


def _schedules(d: pd.DataFrame, order: Mapping[str, Any]) -> list[list]:
    """Complete per (agent, run) schedules ordered by `order` (condition id -> sort key)."""
    acc: dict = {}
    for a, r, c, v in zip(d["_agent"].tolist(), d["run"].tolist(), d["condition"].tolist(), d["decision"].tolist()):
        if c in order:
            acc.setdefault((a, r), {})[c] = v
    keys = sorted(order, key=order.get)
    return [[dd[k] for k in keys] for dd in acc.values() if all(k in dd for k in keys)]


# ---------------------------------------------------------------- per-game moments

def _m_dictator(d):
    pot = d["condition"].map(_cparam("dictator", "pot")).astype(float)
    x = _num(d["decision"])
    return {"mean_give_share": _mean(x / pot), "share_give_zero": _mean(x == 0) if len(x) else np.nan,
            "share_equal_split": _mean(x == pot / 2) if len(x) else np.nan}


def _m_eleven_twenty(d):
    x = _num(d["decision"]).dropna()
    if not len(x):
        return {k: np.nan for k in ("share_17_to_19", "share_17_to_20", "share_choose_20", "mean_request")}
    return {"share_17_to_19": float(x.between(17, 19).mean()), "share_17_to_20": float(x.between(17, 20).mean()),
            "share_choose_20": float((x == 20).mean()), "mean_request": float(x.mean())}


_HL_ORDER = {f"row{k}": k for k in range(1, 11)}


def _m_holt_laury(d):
    sch = _schedules(d, _HL_ORDER)
    safe = np.array([sum(v == "Option A" for v in s) for s in sch], dtype=float)
    if not len(safe):
        return {"mean_safe_choices": np.nan, "share_more_than_4_safe": np.nan, "share_4_to_6_safe": np.nan}
    return {"mean_safe_choices": float(safe.mean()), "share_more_than_4_safe": float((safe > 4).mean()),
            "share_4_to_6_safe": float(((safe >= 4) & (safe <= 6)).mean())}


_LG_GAIN = {c["id"]: c["gain"] for c in L.GAMES["loss_gambles"]["conditions"]}
_LG_LOSS = 5.0
LAMBDA_CAP = max(_LG_GAIN.values()) / _LG_LOSS     # 4.0: top of the ladder (right-censoring point)


def _m_loss_gambles(d):  # noqa: D401
    """implied lambda = lowest accepted gain / loss per complete (agent, run) ladder; an agent who never accepts
    is right-censored at the top of the ladder (LAMBDA_CAP = 4.0), so the median is a lower bound then."""
    gains = sorted(_LG_GAIN.values())
    lam, never = [], []
    for s in _schedules(d, _LG_GAIN):
        acc = [g for g, v in zip(gains, s) if v == "Accept"]
        lam.append(min(acc) / _LG_LOSS if acc else LAMBDA_CAP)
        never.append(not acc)
    acc10 = d.loc[d["condition"] == "gain10", "decision"]
    return {"implied_lambda_median": float(np.median(lam)) if lam else np.nan,
            "share_never_accept": float(np.mean(never)) if never else np.nan,
            "accept_rate_gain10": float((acc10 == "Accept").mean()) if len(acc10) else np.nan}


_PG_ORDER = {f"cond{x}": x for x in (0, 5, 10, 15, 20)}


def classify_schedule(s) -> str:
    """FGF-style type of a 5-row strategy schedule (others' average 0, 5, 10, 15, 20): free_rider (all zero),
    conditional (weakly increasing, not constant), hump (rises to an interior peak, then falls), other.
    Simplified from Fischbacher, Gächter & Fehr 2001, who typed 21-entry schedules with a Spearman test."""
    s = [float(v) for v in s]
    if all(v == 0 for v in s):
        return "free_rider"
    inc = all(b >= a for a, b in zip(s, s[1:]))
    if inc and max(s) > min(s):
        return "conditional"
    k = int(np.argmax(s))
    if 0 < k < len(s) - 1 and s[k] > s[0] and s[k] > s[-1] and all(b >= a for a, b in zip(s[:k], s[1:k + 1])) \
            and all(b <= a for a, b in zip(s[k:], s[k + 1:])):
        return "hump"
    return "other"


def _m_public_goods_strategy(d):
    u = _num(d.loc[d["condition"] == "unconditional", "decision"]) / 20.0
    d2 = d.copy()
    d2["decision"] = _num(d2["decision"])
    types = [classify_schedule(s) for s in _schedules(d2, _PG_ORDER)]
    share = (lambda t: float(np.mean([x == t for x in types])) if types else np.nan)
    return {"mean_unconditional_share": _mean(u), "share_conditional_cooperators": share("conditional"),
            "share_free_riders": share("free_rider"), "share_hump_shaped": share("hump")}


def _m_time_choices(d):
    now = d[d["condition"].str.startswith("now")]["decision"]
    later = d[d["condition"].str.startswith("later")]["decision"]
    pb = fb = n = 0
    acc: dict = {}
    for a, r, c, v in zip(d["_agent"].tolist(), d["run"].tolist(), d["condition"].tolist(), d["decision"].tolist()):
        acc.setdefault((a, r), {})[c] = v
    for dd in acc.values():
        for amt in ("103", "110", "125"):
            a, b = dd.get(f"now{amt}"), dd.get(f"later{amt}")
            if a is None or b is None:
                continue
            n += 1
            pb += (a == "Sooner payment" and b == "Later payment")
            fb += (a == "Later payment" and b == "Sooner payment")
    return {"later_share_now_pairs": float((now == "Later payment").mean()) if len(now) else np.nan,
            "later_share_later_pairs": float((later == "Later payment").mean()) if len(later) else np.nan,
            "present_bias_share": pb / n if n else np.nan, "future_bias_share": fb / n if n else np.nan}


def _m_trust_returner(d):
    rec = d["condition"].map(_cparam("trust_returner", "received")).astype(float)
    return {"mean_return_share": _mean(_num(d["decision"]) / rec)}


def _m_trust_sender(d):
    x = _num(d["decision"])
    return {"mean_sent_share": _mean(x / 100.0), "share_send_zero": _mean(x == 0) if len(x) else np.nan}


def _m_ultimatum_proposer(d):
    x = _num(d["decision"])
    return {"mean_offer_share": _mean(x / 100.0), "share_offer_half": _mean(x == 50) if len(x) else np.nan}


def _m_ultimatum_responder(d):
    out = {}
    for c in ("low", "mid", "fair"):
        v = d.loc[d["condition"] == c, "decision"]
        out[f"reject_rate_{c}"] = float((v == "Reject").mean()) if len(v) else np.nan
    return out


def _m_beauty_contest(d):
    x = _num(d["decision"]).dropna()
    if not len(x):
        return {k: np.nan for k in ("mean_guess", "median_guess", "share_above_67", "share_zero")}
    return {"mean_guess": float(x.mean()), "median_guess": float(x.median()),
            "share_above_67": float((x > 67).mean()), "share_zero": float((x == 0).mean())}


def _m_pg_punishment(d):
    x = _num(d["decision"])
    c = d["condition"]
    fr, co = x[c == "punish_freerider"], x[c == "punish_cooperator"]
    events = int((fr > 0).sum() + (co > 0).sum())
    return {"contribute_share": _mean(x[c == "contribute"] / 20.0), "mean_punish_freerider": _mean(fr),
            "mean_punish_cooperator": _mean(co),
            "perverse_event_share": float((co > 0).sum() / events) if events else np.nan}


def _m_third_party_punishment(d):
    x = _num(d["decision"])
    gave = d["condition"].map(_cparam("third_party_punishment", "gave")).astype(float)
    out = {f"mean_spend_{c}": _mean(x[d["condition"] == c]) for c in ("gave0", "gave10", "gave25", "gave50")}
    below = x[gave < 50]
    out["share_punish_below_50"] = float((below > 0).mean()) if len(below.dropna()) else np.nan
    ok = x.notna()
    xx, yy = (50 - gave[ok]).to_numpy(float), x[ok].to_numpy(float)
    out["punish_slope"] = float(np.polyfit(xx, yy, 1)[0]) if len(np.unique(xx)) >= 2 else np.nan
    return out


def _m_who_to_punish(d):
    out = {}
    for k in ("low", "avg", "high"):
        v = d.loc[d["condition"] == f"ballot_{k}", "decision"]
        out[f"yes_share_{k}"] = float((v == "Yes").mean()) if len(v) else np.nan
    return out


def _m_ertan_vignette(d):
    x = _num(d["decision"])
    out = {}
    for p in L.ERTAN_PROFILES:
        v = x[d["condition"] == p]
        out[f"mean_punish_{p}"] = _mean(v)
        out[f"share_punish_{p}"] = float((v > 0).mean()) if len(v.dropna()) else np.nan
    return out


MOMENTS: dict[str, Callable[[pd.DataFrame], dict]] = {
    "dictator": _m_dictator, "eleven_twenty": _m_eleven_twenty, "holt_laury": _m_holt_laury,
    "loss_gambles": _m_loss_gambles, "public_goods_strategy": _m_public_goods_strategy,
    "time_choices": _m_time_choices, "trust_returner": _m_trust_returner, "trust_sender": _m_trust_sender,
    "ultimatum_proposer": _m_ultimatum_proposer, "ultimatum_responder": _m_ultimatum_responder,
    "beauty_contest": _m_beauty_contest, "pg_punishment": _m_pg_punishment,
    "third_party_punishment": _m_third_party_punishment, "who_to_punish": _m_who_to_punish,
    "ertan_vignette": _m_ertan_vignette,
}


# ---------------------------------------------------------------- human references

# (game, moment, value, scale, citation). scale "share" = 0-1 quantity; "raw" = game units.
_REFS = [
    ("dictator", "mean_give_share", 0.2835, "share", "Engel 2011 dictator meta-analysis (MPI preprint 2010/07): 28.35% of the pie"),
    ("dictator", "share_give_zero", 0.3611, "share", "Engel 2011, same source"),
    ("dictator", "share_equal_split", 0.1674, "share", "Engel 2011, same source"),
    ("eleven_twenty", "share_17_to_19", 0.74, "share", "Arad & Rubinstein 2012 AER 102(7), p.3565"),
    ("eleven_twenty", "share_17_to_20", 0.80, "share", "Arad & Rubinstein 2012, p.3566"),
    ("eleven_twenty", "share_choose_20", 0.06, "share", "Arad & Rubinstein 2012, Table 1 p.3565"),
    ("eleven_twenty", "mean_request", 17.10, "raw", "derived by simengine from Arad & Rubinstein 2012 Table 1 p.3565 (n=108)"),
    ("holt_laury", "share_more_than_4_safe", 2 / 3, "share", "Holt & Laury 2002 AER 92(5), p.1648 ('~2/3', approximate)"),
    ("holt_laury", "share_4_to_6_safe", 0.75, "share", "Holt & Laury 2002, Table 3 p.1649, low real column"),
    ("loss_gambles", "implied_lambda_median", 2.25, "raw", "Tversky & Kahneman 1992, JRU 5:297-323, pp.311-312 (median lambda; different elicitation)"),
    ("public_goods_strategy", "mean_unconditional_share", 0.335, "share", "Fischbacher, Gächter & Fehr 2001, Econ. Letters 71, p.401 (MPCR 0.4 vs 0.5 here)"),
    ("public_goods_strategy", "share_conditional_cooperators", 0.50, "share", "FGF 2001, Section 3 p.401 & Fig. 1 (22/44)"),
    ("public_goods_strategy", "share_free_riders", 0.295, "share", "FGF 2001, Section 3 p.401 (13/44)"),
    ("public_goods_strategy", "share_hump_shaped", 0.136, "share", "FGF 2001, Section 3 p.401 (6/44)"),
    ("trust_sender", "mean_sent_share", 0.509, "share", "Johnson & Mislin 2011, J. Econ. Psych. 32(5); WP Table 1 sentfraction"),
    ("trust_returner", "mean_return_share", 0.365, "share", "Johnson & Mislin 2011; WP Table 1 retavail"),
    ("ultimatum_proposer", "mean_offer_share", 0.40, "share", "Oosterbeek, Sloof & van de Kuilen 2004, Exp. Econ. 7(2), abstract"),
    ("ultimatum_responder", "reject_rate_low", 0.50, "share", "Camerer 2003, Behavioral Game Theory ch. 2: offers below 20% rejected 'about half' (approximate)"),
    ("beauty_contest", "mean_guess", 36.73, "raw", "Nagel 1995 AER 85(5), Figure 1B p.1316"),
    ("beauty_contest", "median_guess", 33.0, "raw", "Nagel 1995, Figure 1B p.1316"),
    ("beauty_contest", "share_above_67", 0.10, "share", "Nagel 1995, p.1316"),
    ("beauty_contest", "share_zero", 0.0, "share", "Nagel 1995, p.1316 ('No subject chose 0')"),
    ("pg_punishment", "contribute_share", 0.58, "share", "Fehr & Gächter 2000 AER 90(4), Result 1 p.984 (Stranger)"),
    ("pg_punishment", "mean_punish_freerider", 6.8, "raw", "Fehr & Gächter 2000, Figure 5 p.991 (Partner, 14-20 below average; approximate match)"),
    ("pg_punishment", "perverse_event_share", 0.20, "share", "Ertan, Page & Putterman 2005 WP printed p.1, citing Fehr & Gächter 2000 p.990 ('about 20% or more')"),
    ("third_party_punishment", "mean_spend_gave0", 14.0, "raw", "Fehr & Fischbacher 2004, Evol. Hum. Behav. 25(2), p.68 (points)"),
    ("third_party_punishment", "share_punish_below_50", 0.60, "share", "Fehr & Fischbacher 2004, Sec. 2.2 p.68 ('roughly 60%')"),
    ("third_party_punishment", "punish_slope", 0.28, "raw", "Fehr & Fischbacher 2004, p.68 (OLS on 50 - transfer)"),
    ("who_to_punish", "yes_share_low", 0.641, "share", "Ertan, Page & Putterman 2005 WP (EER 2009), Table 2 p.10 (410/640)"),
    ("who_to_punish", "yes_share_avg", 0.072, "share", "Ertan, Page & Putterman, Table 2 p.10 (46/640)"),
    ("who_to_punish", "yes_share_high", 0.173, "share", "Ertan, Page & Putterman, Table 2 p.10 (111/640)"),
]

# Full human choice distributions (support -> probability).
HUMAN_DISTRIBUTIONS: dict[str, dict] = {
    "eleven_twenty": {
        "support": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "p": [0.04, 0.00, 0.03, 0.06, 0.01, 0.06, 0.32, 0.30, 0.12, 0.06],
        "citation": "Arad & Rubinstein 2012 AER 102(7), Table 1 p.3565 (basic version, n=108)",
    },
}


def human_refs() -> pd.DataFrame:
    """Human reference moments: game, moment, value, citation, scale ('share' | 'raw'), holdout."""
    df = pd.DataFrame(_REFS, columns=["game", "moment", "value", "scale", "citation"])
    df["holdout"] = df["game"].map(lambda g: bool(L.GAMES[g]["holdout"]))
    return df[["game", "moment", "value", "citation", "scale", "holdout"]]


def human_distributions() -> dict:
    return {k: dict(v) for k, v in HUMAN_DISTRIBUTIONS.items()}


# ---------------------------------------------------------------- moments

def _prep(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    for c in ("model", "persona_method", "agent_id", "game", "condition"):
        df[c] = df[c].astype(str)
    if "run" not in df.columns:
        df["run"] = 0
    v = df["valid"]
    df["valid"] = v if v.dtype == bool else v.map(lambda x: str(x).strip().lower() in ("1", "true", "1.0"))
    df["_agent"] = df["agent_id"]
    return df


def _cell_moments(game: str, valid: pd.DataFrame) -> dict:
    f = MOMENTS.get(game)
    return {} if f is None else f(valid)


def moments(results: pd.DataFrame) -> pd.DataFrame:
    """Per (model, persona_method, game) moments. Columns: model, persona_method, game, moment, value,
    n_agents, n_rows, n_valid. Unknown games are skipped; invalid rows only enter n_rows."""
    df = _prep(results)
    rows = []
    for (model, meth, game), g in df.groupby(CELL, sort=True):
        valid = g[g["valid"]]
        for k, v in _cell_moments(game, valid).items():
            rows.append({"model": model, "persona_method": meth, "game": game, "moment": k, "value": v,
                         "n_agents": int(g["agent_id"].nunique()), "n_rows": int(len(g)), "n_valid": int(len(valid))})
    return pd.DataFrame(rows, columns=["model", "persona_method", "game", "moment", "value", "n_agents", "n_rows",
                                       "n_valid"])


# ---------------------------------------------------------------- distances

def _cdfs(values, support, p):
    x = np.sort(np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float))
    if not len(x):
        return None
    sup, pp = np.asarray(support, float), np.asarray(p, float)
    pts = np.union1d(sup, x)
    f_sim = np.searchsorted(x, pts, side="right") / len(x)
    order = np.argsort(sup)
    f_hum = np.concatenate([[0.0], np.cumsum(pp[order])])[np.searchsorted(sup[order], pts, side="right")]
    return pts, f_sim, f_hum


def wasserstein_to(values, support, p) -> float:
    """Wasserstein-1 between the sample `values` and the human distribution (support, p):
    the integral of |F_sim - F_human| (exact for step CDFs; equals scipy.stats.wasserstein_distance)."""
    c = _cdfs(values, support, p)
    if c is None:
        return float("nan")
    pts, f_sim, f_hum = c
    return float(np.sum(np.abs(f_sim - f_hum)[:-1] * np.diff(pts)))


def ks_to(values, support, p) -> float:
    """sup |ECDF(values) - F_human| over all jump points (exact for two step CDFs)."""
    c = _cdfs(values, support, p)
    if c is None:
        return float("nan")
    _, f_sim, f_hum = c
    return float(np.max(np.abs(f_sim - f_hum)))


def _dist_metrics(game: str, valid: pd.DataFrame, dists: Mapping) -> dict:
    if game not in dists:
        return {}
    d = dists[game]
    vals = valid["decision"]
    return {("choice_distribution", "wasserstein"): wasserstein_to(vals, d["support"], d["p"]),
            ("choice_distribution", "ks"): ks_to(vals, d["support"], d["p"])}


def distance_to_human(results: pd.DataFrame, refs: pd.DataFrame | None = None, B: int = 1000, seed: int = 0,
                      distributions: Mapping | None = None, level: float = 0.95) -> pd.DataFrame:
    """Distance between simulated and human moments, per (model, persona_method, game, moment).

    abs_error = |sim - human| with a percentile bootstrap CI that resamples agents (all of an agent's rows, runs
    and conditions move together). For games with a full human distribution (eleven_twenty, Arad & Rubinstein
    Table 1) it adds Wasserstein-1 and KS rows (metric 'wasserstein' / 'ks', moment 'choice_distribution').
    No full distribution is inlined for the beauty contest: Nagel 1995 reports intervals, not a full table.
    Columns: model, persona_method, game, moment, metric, sim, human, abs_error, ci_lo, ci_hi, n_agents,
    n_valid, scale, holdout, citation."""
    refs = human_refs() if refs is None else refs
    dists = HUMAN_DISTRIBUTIONS if distributions is None else distributions
    ref_map = {(r.game, r.moment): r for r in refs.itertuples(index=False)}
    df = _prep(results)
    rng = np.random.default_rng(seed)
    a_lo, a_hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out = []
    for (model, meth, game), g in df.groupby(CELL, sort=True):
        valid = g[g["valid"]].reset_index(drop=True)
        want = [k for (gm, k) in ref_map if gm == game]
        if not want and game not in dists:
            continue
        point = _cell_moments(game, valid)
        pdist = _dist_metrics(game, valid, dists)
        boots: dict = {k: [] for k in want}
        boots.update({k: [] for k in pdist})
        codes = {a: i for i, a in enumerate(valid["_agent"].unique())}
        k_ag = len(codes)
        if B and k_ag:
            idx_by = [np.flatnonzero(valid["_agent"].map(codes).to_numpy() == i) for i in range(k_ag)]
            lens = np.array([len(a) for a in idx_by])
            for _ in range(int(B)):
                pick = rng.integers(0, k_ag, k_ag)
                sub = valid.iloc[np.concatenate([idx_by[i] for i in pick])].copy()
                sub["_agent"] = np.repeat(np.arange(k_ag), lens[pick])
                bm = _cell_moments(game, sub)
                for k in want:
                    r = ref_map[(game, k)]
                    boots[k].append(abs(bm.get(k, np.nan) - float(r.value)))
                for k, v in _dist_metrics(game, sub, dists).items():
                    boots[k].append(v)
        base = {"model": model, "persona_method": meth, "game": game, "n_agents": int(g["agent_id"].nunique()),
                "n_valid": int(len(valid)), "holdout": bool(L.GAMES[game]["holdout"]) if game in L.GAMES else False}

        def ci(vals):
            v = np.asarray(vals, float)
            v = v[np.isfinite(v)]
            return (float(np.percentile(v, a_lo)), float(np.percentile(v, a_hi))) if len(v) else (np.nan, np.nan)

        for k in want:
            r = ref_map[(game, k)]
            sim = point.get(k, np.nan)
            lo, hi = ci(boots[k])
            out.append({**base, "moment": k, "metric": "abs_error", "sim": sim, "human": float(r.value),
                        "abs_error": abs(sim - float(r.value)) if np.isfinite(sim) else np.nan,
                        "ci_lo": lo, "ci_hi": hi, "scale": getattr(r, "scale", ""),
                        "citation": getattr(r, "citation", "")})
        for (mom, metric), v in pdist.items():
            lo, hi = ci(boots[(mom, metric)])
            out.append({**base, "moment": mom, "metric": metric, "sim": np.nan, "human": np.nan, "abs_error": v,
                        "ci_lo": lo, "ci_hi": hi, "scale": "distribution",
                        "citation": dists[game].get("citation", "")})
    cols = ["model", "persona_method", "game", "moment", "metric", "sim", "human", "abs_error", "ci_lo", "ci_hi",
            "n_agents", "n_valid", "scale", "holdout", "citation"]
    return pd.DataFrame(out, columns=cols)


# ---------------------------------------------------------------- EvalFrame bridge

def moment_frame(results: pd.DataFrame, refs: pd.DataFrame | None = None, *, model: str | None = None,
                 include_raw: bool = False, item: bool = False) -> pd.DataFrame:
    """EvalFrame of simulated vs human moments for `simengine.evidence.evaluate(task="continuous")`.

    One row per human-referenced moment: unit=`game:moment`, group=game, truth=human value,
    pred_<persona_method>=simulated value, split='test' for holdout games and 'train' otherwise (evaluate scores
    the test rows, so the headline is the holdout games; the baselines are fitted on the calibration moments).

    item=False (default) leaves the `item` column out. With one row per item, evaluate's continuous random
    baseline ("another unit's truth within the same item") would copy the truth itself and score perfectly.
    item=True adds item=`game:moment` anyway (the layout named in the porting plan) for callers that do not use
    the baseline ladder.

    Only 'share'-scale moments by default (mixing 0-1 shares with raw units such as the beauty-contest mean
    would make the pooled error meaningless); include_raw=True keeps all. With several models in `results`,
    pass model=."""
    refs = human_refs() if refs is None else refs
    if not include_raw and "scale" in refs.columns:
        refs = refs[refs["scale"] == "share"]
    m = moments(results)
    models = sorted(m["model"].unique())
    if model is None:
        if len(models) > 1:
            raise ValueError(f"results hold several models {models}: pass model=")
    else:
        m = m[m["model"] == str(model)]
    m = m.merge(refs[["game", "moment", "value"]].rename(columns={"value": "truth"}), on=["game", "moment"])
    base_cols = ["unit", *(["item"] if item else []), "group", "truth"]
    if m.empty:
        return pd.DataFrame(columns=[*base_cols, "split"])
    wide = (m.drop_duplicates(["game", "moment", "persona_method"])
            .set_index(["game", "moment", "persona_method"])["value"].unstack("persona_method"))
    wide.columns = [f"pred_{c}" for c in wide.columns]
    wide = wide.reset_index().merge(m[["game", "moment", "truth"]].drop_duplicates(["game", "moment"]),
                                    on=["game", "moment"]).sort_values(["game", "moment"])
    wide["unit"] = wide["game"] + ":" + wide["moment"]
    if item:
        wide["item"] = wide["unit"]
    wide["group"] = wide["game"]
    wide["split"] = wide["game"].map(lambda g: "test" if L.GAMES.get(g, {}).get("holdout") else "train")
    preds = [c for c in wide.columns if c.startswith("pred_")]
    return wide[[*base_cols, *preds, "split"]].reset_index(drop=True)


__all__ = ["moments", "human_refs", "human_distributions", "distance_to_human", "moment_frame", "MOMENTS",
           "HUMAN_DISTRIBUTIONS", "classify_schedule", "wasserstein_to", "ks_to", "LAMBDA_CAP"]
