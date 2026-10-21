"""Deterministic coherence detectors for game results (port of edsl-test b01 `analyze.py`).

Flags MARK rows for inspection; they never filter or drop data. Several flagged behaviours are documented in
humans; each flag carries that precedent note. The interesting quantity is the RATE compared with the human
rate, not the existence of a flag.

Detectors (cell = model x persona_method x game; agent-run = one agent's run):
- invalid_rate    : > 10% invalid decisions in a cell
- zero_variance   : every valid decision in a cell identical (>= 6 decisions; the bare-model baseline is exempt)
- hl_multi_switch : Holt-Laury A/B pattern switches more than once within an agent-run
- lg_non_monotone : loss_gambles accepts a worse gamble while rejecting a better one
- tc_future_bias  : time_choices later in the now-pair but sooner in the matched later-pair
- ur_incoherent   : ultimatum_responder accepts a lower offer while rejecting a higher one
- tr_overreturn   : trust_returner returns more than it received (checked on every numeric decision, valid or not:
                    the per-condition bound already marks these invalid, so a valid-only check could never fire)
- bc_dominated    : beauty_contest guesses > 67
- pp_perverse     : pg_punishment punishes the high contributor more than the free rider
- tpp_inverted    : third_party punishes the fair dictator (gave 50) more than the unfair one (gave 0)
"""
from __future__ import annotations

import pandas as pd

from . import library as L

# Human precedent notes, verbatim from b01 analyze.py; the last three (marked "simengine note") are ours.
PRECEDENT = {
    "pp_perverse": "human-attested: ~20% of punishment events target high contributors (Ertan p.1 lit review; 28% by event in their unrestricted phase, p.19)",
    "tpp_inverted": "human-attested in spirit: antisocial/perverse punishment exists (Herrmann et al. 2008; Ertan ~20%)",
    "tc_future_bias": "human-attested but rare (reverse present bias / 'future bias' documented in the intertemporal literature)",
    "hl_multi_switch": "human-attested: a minority of real Holt-Laury subjects switch multiple times",
    "ur_incoherent": "rare in humans; usually noise",
    "lg_non_monotone": "rare in humans; usually noise",
    "bc_dominated": "human-attested: a small share of Nagel 1995 subjects guess above 67",
    "invalid_rate": "simengine note: not a behaviour; harness or parse failure, and dropout is often not random",
    "zero_variance": "simengine note: humans vary in every game of this battery; a constant cell signals collapse",
    "tr_overreturn": "simengine note: impossible in the original game (cannot return more than received)",
}

FLAGS = tuple(PRECEDENT)
_ROW_ORDER = {f"row{k}": k for k in range(1, 11)}
_GAIN = {c["id"]: c["gain"] for c in L.GAMES["loss_gambles"]["conditions"]}
_OFFER_RANK = {"low": 0, "mid": 1, "fair": 2}
_RECEIVED = {c["id"]: c["received"] for c in L.GAMES["trust_returner"]["conditions"]}
COLUMNS = ["model", "persona_method", "game", "flag", "agent_id", "run", "detail", "precedent"]


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _valid(series) -> pd.Series:
    return series if series.dtype == bool else series.map(lambda x: str(x).strip().lower() in ("1", "true", "1.0"))


def detect(game: str, method: str, rows: pd.DataFrame, *, exempt_zero_variance=("baseline",),
           invalid_threshold: float = 0.10, min_zero_variance: int = 6) -> list[dict]:
    """Flags for one cell. rows: that cell's result rows (valid and invalid)."""
    flags: list[dict] = []

    def add(flag, detail, agent_id=None, run=None):
        flags.append({"flag": flag, "agent_id": agent_id, "run": run, "detail": detail,
                      "precedent": PRECEDENT.get(flag, "")})

    ok = _valid(rows["valid"])
    valid = rows[ok]
    if len(rows) and len(valid) / len(rows) < 1 - invalid_threshold:
        add("invalid_rate", f"invalid_rate {1 - len(valid) / len(rows):.0%} ({len(rows) - len(valid)}/{len(rows)})")
    decisions = [str(d) for d in valid["decision"]]
    if method not in exempt_zero_variance and len(decisions) >= min_zero_variance and len(set(decisions)) == 1:
        add("zero_variance", f"zero_variance (all {decisions[0]!r}, n={len(decisions)})")

    per = valid.groupby(["agent_id", "run"], sort=True)
    if game == "holt_laury":
        for (a, r), rs in per:
            seq = [d for _, d in sorted(zip(rs["condition"].map(lambda c: _ROW_ORDER.get(c, 0)), rs["decision"]))]
            switches = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
            if switches > 1:
                add("hl_multi_switch", f"{switches} switches", a, r)
    elif game == "loss_gambles":
        for (a, r), rs in per:
            acc = {_GAIN[c]: d == "Accept" for c, d in zip(rs["condition"], rs["decision"]) if c in _GAIN}
            xs = sorted(acc)
            if any(acc[xs[i]] and not acc[xs[j]] for i in range(len(xs)) for j in range(i + 1, len(xs))):
                add("lg_non_monotone", "accepts a smaller gain but rejects a larger one", a, r)
    elif game == "time_choices":
        for (a, r), rs in per:
            d = dict(zip(rs["condition"], rs["decision"]))
            for amt in ("103", "110", "125"):
                if d.get(f"now{amt}") == "Later payment" and d.get(f"later{amt}") == "Sooner payment":
                    add("tc_future_bias", f"@{amt}", a, r)
    elif game == "ultimatum_responder":
        for (a, r), rs in per:
            d = {_OFFER_RANK[c]: v for c, v in zip(rs["condition"], rs["decision"]) if c in _OFFER_RANK}
            if any(d.get(i) == "Accept" and d.get(j) == "Reject" for i in d for j in d if i < j):
                add("ur_incoherent", "accepts a lower offer but rejects a higher one", a, r)
    elif game == "trust_returner":
        for _, row in rows.iterrows():
            v, cap = _num(row["decision"]), _RECEIVED.get(str(row["condition"]))
            if v is not None and cap is not None and v > cap:
                add("tr_overreturn", f"{v:g} > {cap}", row["agent_id"], row["run"])
    elif game == "beauty_contest":
        for _, row in valid.iterrows():
            v = _num(row["decision"])
            if v is not None and v > 67:
                add("bc_dominated", f"guess {v:g} > 67", row["agent_id"], row["run"])
    elif game == "pg_punishment":
        for (a, r), rs in per:
            d = {c: _num(v) for c, v in zip(rs["condition"], rs["decision"])}
            f_, c_ = d.get("punish_freerider"), d.get("punish_cooperator")
            if f_ is not None and c_ is not None and c_ > f_:
                add("pp_perverse", f"cooperator {c_:g} > freerider {f_:g}", a, r)
    elif game == "third_party_punishment":
        for (a, r), rs in per:
            d = {c: _num(v) for c, v in zip(rs["condition"], rs["decision"])}
            lo, hi = d.get("gave0"), d.get("gave50")
            if lo is not None and hi is not None and hi > lo:
                add("tpp_inverted", f"fair {hi:g} > unfair {lo:g}", a, r)
    return flags


def coherence_audit(results: pd.DataFrame, **kw) -> pd.DataFrame:
    """All flags, one row per flag: model, persona_method, game, flag, agent_id, run, detail, precedent.
    Cell-level flags (invalid_rate, zero_variance) have agent_id/run None. The input is not modified."""
    df = results.copy()
    for c in ("model", "persona_method", "agent_id", "game", "condition"):
        df[c] = df[c].astype(str)
    out = []
    for (model, meth, game), g in df.groupby(["model", "persona_method", "game"], sort=True):
        for f in detect(game, meth, g, **kw):
            out.append({"model": model, "persona_method": meth, "game": game, **f})
    return pd.DataFrame(out, columns=COLUMNS)


def flag_rows(results: pd.DataFrame, audit: pd.DataFrame | None = None) -> pd.DataFrame:
    """Copy of `results` with a `coherence_flags` column ('; '-joined flag names touching each row; a cell-level
    flag touches every row of its cell). No row is removed."""
    audit = coherence_audit(results) if audit is None else audit
    df = results.copy()
    keys = list(zip(df["model"].astype(str), df["persona_method"].astype(str), df["game"].astype(str)))
    ag = df["agent_id"].astype(str).tolist()
    rn = df["run"].astype(str).tolist() if "run" in df.columns else [""] * len(df)
    tags: list[list[str]] = [[] for _ in range(len(df))]
    for f in audit.itertuples(index=False):
        cell = (f.model, f.persona_method, f.game)
        for i, k in enumerate(keys):
            if k != cell:
                continue
            if f.agent_id is None or (isinstance(f.agent_id, float) and f.agent_id != f.agent_id) or (
                    ag[i] == str(f.agent_id) and rn[i] == str(f.run)):
                if f.flag not in tags[i]:
                    tags[i].append(f.flag)
    df["coherence_flags"] = ["; ".join(t) for t in tags]
    return df


__all__ = ["coherence_audit", "detect", "flag_rows", "PRECEDENT", "FLAGS"]
