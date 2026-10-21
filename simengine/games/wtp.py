"""Who-to-punish human-likeness scorecard (port of edsl-test `benchmark/score_wtp.py`).

Scores one multi-period who-to-punish group run (votes on punishment rules, contributions, punishment) on 9
criteria, each 0-1, weighted 3/2/1 into one index: institutional/treatment fidelity 3, behavioural mechanics 2,
distributional shape 1. Criteria that cannot be observed in a run (N/A) are excluded from the weighted mean,
never zero-filled.

Input is structured run output (not a regex over summary.txt as in the source):

    run = {
      "endowment": 10,                                     optional, default 10 (Ertan design)
      "votes": [{"punish_low": False, "punish_high": False}, ...]   or a list of bools (punish_low only)
      "periods": [{"contrib": {"p1": 8, ...}, "received": {"p1": 0.0, ...}, "punish_low": False}, ...]
                 ("punish_low" may be left out: periods are then split evenly over the votes, as in the source)
      "punish_events": [{"target_contribution": 2, "avg": 7.5, "amount": 1.0}, ...]   optional
    }

Two documented deviations from the source script: `never_punish_high` is N/A when no vote records
`punish_high` (the source assumed 1.0 when its regex found nothing), and `punishment_targeting` is N/A when
punishment was legal but no punishment was paid or `punish_events` is missing (the source wrote 0.0).

Targets: inlined from edsl-test `benchmark/human_targets.json`, extracted from Ertan, Page & Putterman (2005 WP,
published EER 2009) with printed page cites; page numbers are printed pages (PDF page = printed + 1).
"""
from __future__ import annotations

from typing import Any, Mapping

# (name, weight): 3 = institutional/treatment (primary), 2 = behavioural mechanics, 1 = distributional shape
CRITERIA = [
    ("never_punish_high", 3),          # Ertan Result 1: no group ever allowed punishing high contributors
    ("final_vote_punish_low", 3),      # Ertan Result 2: nearly all final votes = punish-low
    ("no_dismantling", 2),             # humans move TOWARD punish-low, not away
    ("treatment_effect", 3),           # contributions higher under punish-low than with no punishment
    ("period1_level", 2),              # period-1 mean in the human band
    ("sustain_under_punish_low", 2),   # rising/flat under punish-low (decay = bad)
    ("punishment_usage", 2),           # costly punishment actually paid when allowed, ~1:1 to shortfall
    ("punishment_targeting", 2),       # aimed at below-average contributors
    ("heterogeneity", 1),              # period-1 spread; humans are heterogeneous
]

# Scorer parameters (human_targets.json `_scorer_params`).
WTP_PARAMS = {
    "period1_band": (0.69, 0.77),          # Figure 3 p.12: period-1 contributions 0.69 (3-Vote) to 0.75-0.77 (5-Vote)
    "treatment_full_credit": 2.0,          # tokens of punish-low minus no-punishment for full credit (scorer choice)
    "usage_ratio_full_band": (0.5, 1.5),   # Table 4 p.17: $0.97-1.22 punishment received per $1 below average
    "heterogeneity_full_credit": 3.0,      # period-1 SD (tokens) for full credit (scorer choice)
}

# Selected machine-readable targets (human_targets.json `machine_readable`), value and printed-page cite.
ERTAN_TARGETS = {
    "final_vote_punish_low_share": (0.875, "Figure 2 reading p.11, reconciles with p.10 totals (35/40)"),
    "first_vote_no_punish_share": (0.575, "Figure 2 reading p.11 (23/40)"),
    "first_vote_punish_low_share": (0.40, "Figure 2 reading p.11 (16/40)"),
    "group_vote_punish_high_share": (0.0, "Result 1, p.10 (0 of 160 group votes)"),
    "group_votes_total": (160, "Table 1, p.9"),
    "group_vote_no_punish_share": (0.35, "p.10 (56/160)"),
    "group_vote_punish_low_only_share": (0.6125, "p.10 (98/160)"),
    "indiv_vote_yes_punish_low_share": (0.641, "Table 2, p.10 (410/640)"),
    "indiv_vote_yes_punish_avg_share": (0.072, "Table 2, p.10 (46/640)"),
    "indiv_vote_yes_punish_high_share": (0.173, "Table 2, p.10 (111/640); '17%' p.22"),
    "period1_contrib_frac_low": (0.69, "Figure 3 reading p.12 (3-Vote)"),
    "period1_contrib_frac_high": (0.77, "Figure 3 reading p.12 (5-Vote)"),
    "contrib_frac_punishlow_steady": (0.95, "Figure 3 reading p.12"),
    "contrib_frac_punishlow_period30": (0.82, "Figure 3 reading p.12"),
    "perverse_punish_share_max": (0.28, "p.19, unrestricted phase, 129 events"),
    "perverse_punish_share_lit": (0.20, "p.1-2 literature review, citing Fehr & Gächter 2000 p.990"),
    "punish_received_per_dollar_below_avg_unrestricted": (0.888, "Table 4 col 1, p.17"),
    "punish_received_per_dollar_below_avg_punishlow_min": (0.967, "Table 4, p.17"),
    "punish_received_per_dollar_below_avg_punishlow_max": (1.217, "Table 4, p.17"),
    "freeride_freq_unrestricted": (0.72, "Figure 5 + p.15 (148/205)"),
    "freeride_freq_punishlow": (0.35, "Figure 5 + p.15 (of 103)"),
}
ENDOWMENT = 10


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _mean_contrib(p: Mapping) -> float:
    c = p["contrib"]
    return sum(c.values()) / len(c)


def _votes(run: Mapping) -> list[dict]:
    out = []
    for v in run.get("votes") or []:
        out.append(dict(v) if isinstance(v, Mapping) else {"punish_low": bool(v)})
    return out


def _periods(run: Mapping, votes: list[dict]) -> list[dict]:
    ps = [dict(p) for p in run.get("periods") or []]
    if ps and any("punish_low" not in p for p in ps):
        if not votes:
            raise ValueError("periods without 'punish_low' need votes to assign regimes")
        per_vote = max(1, len(ps) // len(votes))
        for i, p in enumerate(ps):
            p.setdefault("punish_low", bool(votes[min(i // per_vote, len(votes) - 1)]["punish_low"]))
    return ps


def targeting_share(events) -> tuple[float | None, float]:
    """Share of punishment dollars aimed at below-average contributors: (share, total$); (None, 0) if none paid."""
    low = total = 0.0
    for e in events or []:
        amt = float(e.get("amount", e.get("punish", 0)) or 0)
        if amt <= 0:
            continue
        total += amt
        if float(e["target_contribution"]) < float(e["avg"]):
            low += amt
    return (low / total, total) if total > 0 else (None, 0.0)


def wtp_scorecard(run: Mapping, params: Mapping | None = None) -> dict:
    """Score one structured who-to-punish run. Returns {criterion: score | None, "index": weighted mean over
    observable criteria, "n_criteria": number observed, "weights": {...}}."""
    t = {**WTP_PARAMS, **(params or {})}
    endow = float(run.get("endowment", ENDOWMENT))
    votes = _votes(run)
    periods = _periods(run, votes)
    if not votes or not periods:
        raise ValueError("run needs at least one vote and one period")
    s: dict[str, Any] = {}
    pl_votes = [bool(v["punish_low"]) for v in votes]

    highs = [v["punish_high"] for v in votes if "punish_high" in v]
    s["never_punish_high"] = None if not highs else (0.0 if any(highs) else 1.0)
    s["final_vote_punish_low"] = 1.0 if pl_votes[-1] else 0.0
    s["no_dismantling"] = 0.0 if (pl_votes[0] and not pl_votes[-1]) else 1.0

    pl = [_mean_contrib(p) for p in periods if p["punish_low"]]
    npn = [_mean_contrib(p) for p in periods if not p["punish_low"]]
    # last period of each regime (quasi steady state), so recovery-from-decay paths are not penalised
    s["treatment_effect"] = clamp01((pl[-1] - npn[-1]) / t["treatment_full_credit"]) if pl and npn else None

    frac = _mean_contrib(periods[0]) / endow
    lo, hi = t["period1_band"]
    s["period1_level"] = 1.0 if lo <= frac <= hi else \
        clamp01(1 - (lo - frac) / lo if frac < lo else 1 - (frac - hi) / (1 - hi))

    if len(pl) >= 2:
        slope = (pl[-1] - pl[0]) / (len(pl) - 1)      # tokens per period under punish-low
        s["sustain_under_punish_low"] = clamp01(1 + slope / 2)   # flat = 1, -2 per period = 0
    else:
        s["sustain_under_punish_low"] = None

    if any(p["punish_low"] for p in periods):
        shortfall = punished = 0.0
        for p in periods:
            if not p["punish_low"]:
                continue
            avg = _mean_contrib(p)
            for player, c in p["contrib"].items():
                if c < avg:
                    shortfall += avg - c
                    punished += float((p.get("received") or {}).get(player, 0.0))
        if shortfall > 0:
            ratio = punished / shortfall
            lo_r, hi_r = t["usage_ratio_full_band"]
            s["punishment_usage"] = 1.0 if lo_r <= ratio <= hi_r else \
                clamp01(ratio / lo_r if ratio < lo_r else 1 - (ratio - hi_r))
        else:
            s["punishment_usage"] = None        # nobody ever below average
        share, _ = targeting_share(run.get("punish_events"))
        s["punishment_targeting"] = share       # None when no punishment was paid / no event data
    else:
        s["punishment_usage"] = None            # punishment never legal: unobservable
        s["punishment_targeting"] = None

    vals = list(periods[0]["contrib"].values())
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    s["heterogeneity"] = clamp01(sd / t["heterogeneity_full_credit"])

    num = den = 0.0
    for name, w in CRITERIA:
        if s[name] is not None:
            num += w * s[name]
            den += w
    s["index"] = num / den if den else None
    s["n_criteria"] = sum(s[n] is not None for n, _ in CRITERIA)
    s["weights"] = dict(CRITERIA)
    return s


__all__ = ["wtp_scorecard", "targeting_share", "CRITERIA", "WTP_PARAMS", "ERTAN_TARGETS"]
