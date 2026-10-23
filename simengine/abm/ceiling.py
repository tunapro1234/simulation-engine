"""LLM saturation-level (ceiling) prior for adoption series, with placebo controls and a leak check.

One call per series and prompt: the **blind** prompt shows only the anonymised history (relative years, values, unit),
the **named** prompt adds the series name, the **perturbed** prompt is the blind one with the history multiplied by
u ~ U(1 - perturb, 1 + perturb) per series (the answer is divided back by u). Calls run through
`simengine.llm.parallel.parallel_map` and are parsed with `simengine.llm.jsonx.complete_json`; an unparseable
answer is NaN (counted in the leak check), never an exception.

    prior = llm_ceiling_prior(llm, history[history.t <= origin], groups="kind")    # long frame series, t, y
    prior["ceiling"]                      # per series (index), named prompt; ceiling_blind, ceiling_perturbed
    prior["ceiling_shuffled"]             # placebo: named ceilings permuted within group (no series keeps its own)
    prior["ceiling_group_median"]         # placebo: median of the OTHER members' ceilings (leave-one-out)
    prior.attrs["leak_check"]["message"]  # plain sentence; also gap, Spearman, scale elasticity, n_failed
    upper = ceiling_upper(prior)          # {series: ceiling}, floored at the observed max, NaN -> group median
    DirectForecaster("hgb", target="logit", upper=upper)                         # or prior["ceiling"] as a column
    AdoptionABCAnalogy(scale={s: c / 100 for s, c in upper.items()})              # ABM ceiling m <= 1 -> c

A series alone in its group gets NaN in both placebos (there is no other member); ceiling_upper then falls back to
the observed maximum.

Placebo controls cost no calls: run the downstream method with `ceiling`, `ceiling_shuffled` and
`ceiling_group_median`; a gain that survives the shuffle is not series knowledge (the elasticity prior's LLM-swap
placebo matched its gain, docs/modules/abm.md).

Leak check (heuristic thresholds, this port's choice, not calibrated on a real model):

* named - blind gap: what the name adds. A large gap is knowledge about that series, and for an origin inside the
  model's training data possibly knowledge of the outcome.
* perturbed vs blind: an answer reasoned from the numbers scales with u (scale elasticity ~1, Spearman ~1); an answer
  from recognising the exact numbers ignores the rescaling (elasticity ~0). Flagged as recognition when the
  elasticity is under 0.5 or the Spearman under 0.8 (>= 5 series). A unit with a natural cap (percent of people:
  "it saturates at 100") also gives elasticity ~0 without any recognition; read the flag on series far from that
  cap, or in a unit without one.
* `future=` (the post-origin truth, dev only): Spearman of each ceiling's headroom (ceiling - last value) with the
  observed post-origin change (mean future y - last value). Named far above blind means the name carries the
  outcome.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ..llm.jsonx import complete_json
from ..llm.parallel import parallel_map
from .adoption import _norm

SYSTEM = ("You estimate the long-run saturation level (ceiling) of technology adoption series. "
          "Answer with JSON only.")
RECOGNITION_ELASTICITY = 0.5
RECOGNITION_RHO = 0.8
MIN_SERIES = 5


def ceiling_prompt(t, y, unit: str, name: str | None = None) -> str:
    """The user prompt: relative years (0 = last observation) and values; `name` only in the named prompt."""
    t = np.asarray(t, float)
    rel = np.rint(t - t.max()).astype(int)
    lines = "\n".join(f"{r}: {v:.4g}" for r, v in zip(rel, y))
    head = f"Series: {name}\n" if name is not None else ""
    return (f"{head}An adoption series ({unit}), one value per year. Years are relative to the last observation "
            f"(0 = last).\n{lines}\n\nWhat level will this series saturate at in the long run? Give the ceiling in "
            f"the same unit; it may exceed 100 when the unit allows more than one per person. "
            f'Return JSON only: {{"ceiling": <number>}}')


def _parse(v: Any) -> float:
    if isinstance(v, dict):
        v = next((v[k] for k in v if str(k).lower() in ("ceiling", "saturation", "level")), None)
    if isinstance(v, list) and len(v) == 1:
        return _parse(v[0])
    try:
        x = float(v)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) and x > 0 else np.nan


def _spearman(a, b) -> float:
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return float("nan")
    ra, rb = a[ok].rank(), b[ok].rank()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _groups(hist: pd.DataFrame, series: list, groups) -> pd.Series:
    if groups is None:
        return pd.Series("all", index=series, dtype=object)
    if isinstance(groups, str):
        if groups not in hist.columns:
            raise ValueError(f"groups={groups!r} is not a column of series_hist")
        return hist.groupby("series")[groups].first().reindex(series)
    if callable(groups):
        return pd.Series([groups(s) for s in series], index=series, dtype=object)
    return pd.Series([groups.get(s, "all") for s in series], index=series, dtype=object)


def _shuffle_within(values: pd.Series, grp: pd.Series, rng) -> pd.Series:
    """Cyclic shift along a random order inside each group: every series gets another member's value. A series alone
    in its group has no other member to borrow from: NaN (keeping its own value would leak it into the placebo)."""
    out = values.copy()
    for _, idx in values.groupby(grp.fillna("__na__"), sort=True).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            out.loc[idx] = np.nan
            continue
        order = [idx[i] for i in rng.permutation(len(idx))]
        out.loc[order] = values.loc[order[1:] + order[:1]].to_numpy()
    return out


def _loo_median(values: pd.Series, grp: pd.Series) -> pd.Series:
    """Median of the OTHER members' values in each group (leave-one-out; NaN when there is none). The plain group
    median contains the series' own value, and in a group of 3 it is that value for the middle series."""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    for _, idx in values.groupby(grp.fillna("__na__"), sort=True).groups.items():
        v = values.loc[list(idx)].to_numpy(float)
        for j, s in enumerate(idx):
            rest = np.delete(v, j)
            rest = rest[np.isfinite(rest)]
            out.loc[s] = float(np.median(rest)) if len(rest) else np.nan
    return out


def llm_ceiling_prior(llm, series_hist: pd.DataFrame, *, blind: bool = True, named: bool = True,
                      perturb: float = 0.15, groups: str | dict | Callable | None = None, parallel: int = 8,
                      seed: int = 0, unit: str = "per 100 people", future: pd.DataFrame | None = None,
                      max_tokens: int = 600, last: int | None = None) -> pd.DataFrame:
    """Ask `llm` (simengine.types.LLM protocol) for one saturation level per series of `series_hist` (long frame
    series, t, y; only data up to the origin, `seri` accepted). `groups`: a column of series_hist, a {series: group}
    dict or a function of the name (None = one group), used by the placebo controls. `last` keeps only the last
    N points per series in the prompts. `perturb=0` skips the perturbed prompt.

    Returns a frame indexed by series: group, last_t, last_y, hist_max, ceiling (named; blind when named=False),
    ceiling_blind, ceiling_perturbed and u (when perturb > 0 and blind), ceiling_shuffled, ceiling_group_median
    (leave-one-out). hist_max is over the whole history, also with `last`.
    attrs["leak_check"]: source, n_series, n_failed (per prompt), n_below_hist, named_blind_gap (median relative),
    named_blind_abs_gap, rho_perturbed_blind, scale_elasticity, rho_future ({column: rho} or None), recognition
    (True / False / None when it cannot be judged) and message (plain sentences)."""
    if not (blind or named):
        raise ValueError("llm_ceiling_prior: ask at least one of blind=True, named=True")
    if not 0 <= perturb < 1:
        raise ValueError(f"perturb is a fraction in [0, 1), got {perturb!r}")
    hist = _norm(series_hist).dropna(subset=["y"]).sort_values(["series", "t"], kind="mergesort")
    hist = hist.drop_duplicates(["series", "t"], keep="last")
    series = sorted(hist["series"].unique(), key=str)
    if not series:
        raise ValueError("llm_ceiling_prior: series_hist has no rows")
    rng = np.random.default_rng(seed)
    use_pert = perturb > 0 and blind
    u = pd.Series(rng.uniform(1 - perturb, 1 + perturb, len(series)) if use_pert else np.ones(len(series)),
                  index=series)

    items = []                                                     # (kind, series, user prompt)
    info = {}
    for s, g in hist.groupby("series", sort=False):
        info[s] = dict(last_t=float(g["t"].iloc[-1]), last_y=float(g["y"].iloc[-1]), hist_max=float(g["y"].max()))
        g = g.tail(last) if last else g                            # prompt only; hist_max stays the full history's
        t, y = g["t"].to_numpy(float), g["y"].to_numpy(float)
        if named:
            items.append(("named", s, ceiling_prompt(t, y, unit, name=str(s))))
        if blind:
            items.append(("blind", s, ceiling_prompt(t, y, unit)))
        if use_pert:
            items.append(("perturbed", s, ceiling_prompt(t, y * u[s], unit)))

    def ask(it):
        return complete_json(llm, SYSTEM, it[2], max_tokens=max_tokens)

    answers = parallel_map(ask, items, parallel=parallel, return_exceptions=True)
    got: dict = {}
    failed = {k: 0 for k in ("named", "blind", "perturbed")}
    for (kind, s, _), a in zip(items, answers):
        v = np.nan if isinstance(a, Exception) else _parse(a)
        failed[kind] += int(np.isnan(v))
        got[(kind, s)] = v

    out = pd.DataFrame(info).T.reindex(series)
    out.index.name = "series"
    out = out.astype(float)
    out.insert(0, "group", _groups(hist, series, groups).to_numpy())
    col = lambda kind: pd.Series([got[(kind, s)] for s in series], index=series, dtype=float)   # noqa: E731
    out["ceiling"] = col("named") if named else col("blind")
    if blind:
        out["ceiling_blind"] = col("blind")
    if use_pert:
        out["ceiling_perturbed"] = col("perturbed") / u
        out["u"] = u
    out["ceiling_shuffled"] = _shuffle_within(out["ceiling"], out["group"], rng)
    out["ceiling_group_median"] = _loo_median(out["ceiling"], out["group"])
    out.attrs["leak_check"] = _leak_check(out, named, blind, use_pert, failed, future, perturb)
    return out


def _leak_check(out, named, blind, use_pert, failed, future, perturb) -> dict:
    n = len(out)
    lc: dict = dict(source="named" if named else "blind", n_series=n,
                    n_failed={k: v for k, v in failed.items() if (k != "named" or named)
                              and (k != "blind" or blind) and (k != "perturbed" or use_pert)},
                    n_below_hist=int((out["ceiling"] < out["hist_max"]).sum()),
                    named_blind_gap=None, named_blind_abs_gap=None, rho_perturbed_blind=None,
                    scale_elasticity=None, rho_future=None, recognition=None)
    msg = []
    if named and blind:
        rel = (out["ceiling"] - out["ceiling_blind"]) / out["ceiling_blind"]
        lc["named_blind_gap"] = float(rel.median()) if rel.notna().any() else None
        lc["named_blind_abs_gap"] = float(rel.abs().median()) if rel.notna().any() else None
        if lc["named_blind_abs_gap"] is not None:
            msg.append(f"Naming the series moves the ceiling by a median {100 * lc['named_blind_abs_gap']:.0f}% "
                       f"(signed {100 * lc['named_blind_gap']:+.0f}%) against the blind prompt.")
    if use_pert:
        b, p = out["ceiling_blind"], out["ceiling_perturbed"]
        lc["rho_perturbed_blind"] = _spearman(p, b)
        raw = p * out["u"]                                          # the answer before dividing back by u
        x, z = np.log(out["u"]), np.log(raw / b)
        ok = x.notna() & z.notna() & np.isfinite(z)
        if ok.sum() >= 3 and x[ok].var() > 0:
            lc["scale_elasticity"] = float(np.cov(x[ok], z[ok])[0, 1] / x[ok].var())
        if ok.sum() >= MIN_SERIES and lc["scale_elasticity"] is not None:
            rho = lc["rho_perturbed_blind"]
            rec = lc["scale_elasticity"] < RECOGNITION_ELASTICITY or (np.isfinite(rho) and rho < RECOGNITION_RHO)
            lc["recognition"] = bool(rec)
            if rec:
                msg.append(f"The blind prompt looks like recognition: rescaling the history by +/-{100 * perturb:.0f}% "
                           f"does not move the answers with it (scale elasticity {lc['scale_elasticity']:.2f}, "
                           f"Spearman perturbed vs blind {rho:.2f}). The model may know these exact numbers; treat "
                           f"even the blind ceiling as possibly leaked and lean on the placebo controls.")
            else:
                msg.append(f"Blind answers follow a +/-{100 * perturb:.0f}% rescaling of the history (scale elasticity "
                           f"{lc['scale_elasticity']:.2f}, Spearman {rho:.2f}): no sign of recognition from the "
                           f"numbers.")
        else:
            msg.append(f"Too few parsed series ({int(ok.sum())} < {MIN_SERIES}) to judge recognition.")
    if future is not None:
        fut = _norm(future)
        fut = fut[fut["t"] > fut["series"].map(out["last_t"])]
        dev = fut.groupby("series")["y"].mean().reindex(out.index) - out["last_y"]
        cols = [c for c in ("ceiling", "ceiling_blind", "ceiling_perturbed") if c in out.columns]
        lc["rho_future"] = {c.replace("ceiling_", "") if c != "ceiling" else "ceiling":
                            _spearman(out[c] - out["last_y"], dev) for c in cols}
        rf = lc["rho_future"]
        if named and blind and np.isfinite(rf.get("ceiling", np.nan)) and np.isfinite(rf.get("blind", np.nan)):
            if rf["ceiling"] - rf["blind"] > 0.2:
                msg.append(f"Named ceilings track the post-origin change better than blind ones (Spearman "
                           f"{rf['ceiling']:.2f} vs {rf['blind']:.2f}): the name carries information about the "
                           f"outcome; if the forecast window is inside the model's training data, use the blind prior.")
    if lc["n_below_hist"]:
        msg.append(f"{lc['n_below_hist']} of {n} ceilings are below the series' observed maximum "
                   f"(ceiling_upper floors them).")
    lc["message"] = " ".join(msg) if msg else "Only one prompt asked; nothing to compare."
    return lc


def ceiling_upper(prior: pd.DataFrame, col: str = "ceiling", floor: bool = True) -> dict:
    """{series: upper} from a llm_ceiling_prior frame, for DirectForecaster(upper=) / CurveBaseline(upper=) or an
    ABM scale (upper / 100). NaN falls back to the group median, then to the observed maximum; floor=True raises
    every value to at least the observed maximum (a logit target cannot sit below the data). A floored value equals the
    maximum exactly, so a logit forecaster cannot grow that series past it."""
    v = prior[col].fillna(prior["ceiling_group_median"]).fillna(prior["hist_max"])
    if floor:
        v = np.maximum(v, prior["hist_max"])
    return {s: float(x) for s, x in v.items()}


__all__ = ["llm_ceiling_prior", "ceiling_upper", "ceiling_prompt"]
