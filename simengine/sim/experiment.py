"""Two-condition (or k-condition) experiments on a persona panel, and effect-size calibration against human
effects: exaggeration ratio, LOO correction, and how many human respondents the simulation is worth.

    conds = {"control": "No message.", "norms": "Most people in your country support ...", "fear": "..."}
    res = experiment(llm, personas, conds, "How much do you support the policy?", scale=(0, 100), seed=1)
    res["effects"]["norms"]      # diff, g (Hedges), se, ci, n_treat / n_control, no_persona_g
    res["conditions"]            # per condition: mean, sd, n, refused, unparsed (+ the no_persona arm)
    frame = experiment_frame({"study1": res, "study2": res2}, human={"study1": {"norms": 0.21}})

    fit = exaggeration_ratio(sim_g, human_g, family=study, human_se=se)  # median |sim|/|human|, slope, LOO
    corrected = correct_effects(sim_g, human_g=human_g, family=study)   # LOO: never uses its own human value
    human_equivalent_n(corrected, human_g, human_se, family=study)      # {"n": ..., "ci": [...]} or n None
    effect_section(sim_g, human_g, human_se, family=study)              # evidence-extras section for the card

`experiment` is between-subject by default: each persona is assigned one condition (balanced, seeded) and
answers once through `sim.survey` with that condition's text as its own stimulus, so no persona sees the other
arms. The no_persona control arm answers the same condition texts; its g shows how much of the effect a generic
respondent already gives. `between=False` asks every persona under every condition in separate prompts (paired
g, d_av). Refusals and unparsed answers are counted per condition, never imputed.

Effect calibration (G9 rehearsal): simulated effects tend to be larger than human ones. `exaggeration_ratio`
measures that, `correct_effects` divides it out (or fits human ~ a + b sim), always leave-one-out (by family
when given), and `human_equivalent_n` turns the remaining error into a human sample size: a two-arm study with N
people in total has var(d) ~ 4/N, so the simulation's error variance (after subtracting the human sampling noise
mean(se^2)) is worth N = 4 / err humans. When the error is not above the noise, N is None (not "infinite").
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._stats import to_jsonable
from .personas import as_personas
from .survey import survey

Z95 = 1.959963984540054


# ---------------------------------------------------------------- effect sizes
def _j(df: float) -> float:
    """Hedges small-sample correction J(df)."""
    return 1.0 - 3.0 / (4.0 * df - 1.0) if df > 0.25 else float("nan")


def hedges_g(treat, control, alpha: float = 0.05) -> dict:
    """Between-subject Hedges g of `treat` vs `control` (pooled sd), with SE and normal CI.
    {g, d, se, ci, diff, n_treat, n_control}; g None when an arm has < 2 values or the pooled sd is 0."""
    a = np.asarray([x for x in treat if x is not None and not _nan(x)], float)
    b = np.asarray([x for x in control if x is not None and not _nan(x)], float)
    n1, n0 = len(a), len(b)
    out = {"g": None, "d": None, "se": None, "ci": [None, None], "diff": None, "n_treat": n1, "n_control": n0}
    if n1 and n0:
        out["diff"] = float(a.mean() - b.mean())
    if n1 < 2 or n0 < 2:
        return out
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n0 - 1) * b.var(ddof=1)) / (n1 + n0 - 2))
    if sp <= 0:
        return out
    d = (a.mean() - b.mean()) / sp
    j = _j(n1 + n0 - 2)
    g = j * d
    se = j * math.sqrt((n1 + n0) / (n1 * n0) + d * d / (2 * (n1 + n0)))
    z = _z(alpha)
    return {**out, "g": g, "d": d, "se": se, "ci": [g - z * se, g + z * se]}


def paired_g(treat, control, alpha: float = 0.05) -> dict:
    """Within-subject g (d_av: mean difference / sqrt((sd_t^2 + sd_c^2) / 2), Hedges-corrected) over paired
    values (None / NaN pairs dropped), SE from the paired-difference correlation."""
    pairs = [(x, y) for x, y in zip(treat, control) if x is not None and y is not None and not _nan(x)
             and not _nan(y)]
    n = len(pairs)
    out = {"g": None, "d": None, "se": None, "ci": [None, None], "diff": None, "n_treat": n, "n_control": n}
    if not n:
        return out
    a, b = np.array([p[0] for p in pairs], float), np.array([p[1] for p in pairs], float)
    out["diff"] = float((a - b).mean())
    if n < 3:
        return out
    s = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if s <= 0:
        return out
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    d = (a.mean() - b.mean()) / s
    j = _j(n - 1)
    g = j * d
    se = j * math.sqrt(max(2 * (1 - r), 0.0) / n + d * d / (2 * n))
    z = _z(alpha)
    return {**out, "g": g, "d": d, "se": se, "ci": [g - z * se, g + z * se], "r": r}


def _z(alpha: float) -> float:
    if abs(alpha - 0.05) < 1e-12:
        return Z95
    from scipy.stats import norm
    return float(norm.ppf(1 - alpha / 2))


def _nan(x) -> bool:
    return isinstance(x, float) and x != x


def _arm_stats(vals: list) -> dict:
    v = np.asarray([x for x in vals if x is not None], float)
    return {"mean": float(v.mean()) if len(v) else None, "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
            "n": int(len(v))}


# ---------------------------------------------------------------- experiment runner
def _question(question, scale) -> dict:
    q = {"text": question} if isinstance(question, str) else dict(question)
    q.setdefault("id", "outcome")
    if isinstance(scale, (tuple, list)):
        q["kind"], q["min"], q["max"] = "numeric", scale[0], scale[1]
    elif isinstance(scale, str):
        q.setdefault("kind", "likert5")
        q["scale"] = scale
    if q.get("kind", "likert5") not in ("likert5", "likert", "numeric", "number", "slider", "scale"):
        raise ValueError("experiment needs a numeric or Likert question (an effect is a mean difference)")
    return q


def _value(a: dict | None):
    if not a:
        return None
    if a.get("ssr_mean") is not None:
        return a["ssr_mean"]
    return a.get("value", a.get("score"))


def assign_conditions(pids: Sequence[str], conditions: Sequence[str], seed: int = 0) -> dict:
    """Balanced seeded assignment {pid: condition}: a random permutation dealt round-robin (arm sizes differ by
    at most 1)."""
    order = np.random.default_rng(seed).permutation(len(pids))
    names = list(conditions)
    return {pids[i]: names[r % len(names)] for r, i in enumerate(order)}


def experiment(llm, personas, conditions: Mapping[str, str], question, *, between: bool = True,
               control: str | None = None, scale=None, seed: int = 0, samples: int = 1, no_persona: bool = True,
               alpha: float = 0.05, ask_kw: dict | None = None, on_event=None) -> dict:
    """Simulated randomized experiment. `conditions`: {name: stimulus text} (2+; the text is what that arm's
    personas see before the question). `question`: a question string or a `survey` question dict; `scale`
    (lo, hi) makes it numeric, a scale name keeps it a 1-5 Likert (the default; SSR mean when `ask_kw` passes
    ssr=). `control`: the reference condition (default: the first). `between=True`: each persona gets one
    condition (`assign_conditions`, seeded); False: every persona answers under every condition, paired g.
    `samples` > 1 repeats the ask with the personas reshuffled (different batches, so different prompts and
    cache keys) and averages each persona's answers; the averaging removes answer noise from the sd, so g grows
    with `samples` (a human study asks once: compare g with human g at samples=1). `no_persona`: also run survey's no_persona arm on the same
    condition texts and report its g per effect. `ask_kw`: extra `survey` keyword arguments (batch, ssr,
    free_text, parallel, tag, refusal_retry, ...).

    Returns a JSON-safe dict: conditions {name: {mean, sd, n, assigned, refused, unparsed, no_persona}},
    effects {name: {diff, g, d, se, ci, n_treat, n_control, no_persona_g, no_persona_diff}} for every
    non-control condition, assignment {pid: condition} (between only), refused / unparsed totals, warnings,
    usage (calls summed over the survey runs)."""
    names = [str(k) for k in conditions]
    if len(names) < 2:
        raise ValueError("experiment needs at least 2 conditions")
    control = names[0] if control is None else str(control)
    if control not in names:
        raise ValueError(f"control {control!r} is not one of the conditions {names}")
    texts = {str(k): str(v) for k, v in conditions.items()}
    q = _question(question, scale)
    ps = as_personas(personas)
    pids = [p.pid for p in ps]
    kw = dict(ask_kw or {})
    for bad in ("context", "stimulus_col", "controls", "llm", "on_event"):
        if bad in kw:
            raise ValueError(f"ask_kw={bad!r} is set by experiment itself")
    kw.setdefault("free_text", False)
    controls = ["no_persona"] if no_persona else []
    rng = np.random.default_rng(seed)
    assign = assign_conditions(pids, names, seed) if between else None
    warnings: list[str] = []
    usage = {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    # values[arm][cond][pid] -> list of answers over samples
    values: dict[str, dict[str, dict[str, list]]] = {"persona": {c: {} for c in names},
                                                     "no_persona": {c: {} for c in names}}
    missing: dict[str, dict[str, set]] = {c: {"refused": set(), "unparsed": set()} for c in names}
    runs = [(None, {pid: texts[assign[pid]] for pid in pids})] if between else [(c, texts[c]) for c in names]
    for k in range(max(1, int(samples))):
        order = list(ps) if k == 0 else [ps[i] for i in rng.permutation(len(ps))]
        for cond, ctx in runs:
            res = survey(order, [q], llm=llm, context=ctx, controls=controls, on_event=on_event, **kw)
            for key in usage:
                usage[key] += (res.get("usage") or {}).get(key, 0) or 0
            warnings.extend(w for w in res.get("warnings", []) if w not in warnings)
            arms = {"persona": res.get("answers", {}),
                    "no_persona": (res.get("controls", {}).get("answers") or {}).get("no_persona", {})}
            for arm, ans in arms.items():
                for pid in pids:
                    c = cond if cond is not None else assign[pid]
                    v = _value((ans.get(pid) or {}).get(q["id"]))
                    if v is not None:
                        values[arm][c].setdefault(pid, []).append(float(v))
            for kind in ("refused", "unparsed"):
                for pid in res.get(f"{kind}_ids", []):
                    missing[cond if cond is not None else assign[pid]][kind].add(pid)

    def per_person(arm, c):
        return {pid: float(np.mean(v)) for pid, v in values[arm][c].items() if v}

    cond_out = {}
    for c in names:
        pp, npp = per_person("persona", c), per_person("no_persona", c)
        assigned = sum(1 for p in pids if assign[p] == c) if between else len(pids)
        lost = {kind: len(missing[c][kind] - set(pp)) for kind in ("refused", "unparsed")}
        cond_out[c] = {**_arm_stats(list(pp.values())), "assigned": assigned, **lost,
                       "no_persona": _arm_stats(list(npp.values())) if no_persona else None}
    effects = {}
    for c in names:
        if c == control:
            continue
        eff = {}
        for arm in ("persona", "no_persona") if no_persona else ("persona",):
            t, ctl = per_person(arm, c), per_person(arm, control)
            if between:
                e = hedges_g(list(t.values()), list(ctl.values()), alpha)
            else:
                e = paired_g([t.get(p) for p in pids], [ctl.get(p) for p in pids], alpha)
            if arm == "persona":
                eff.update(e)
            else:
                eff["no_persona_g"], eff["no_persona_diff"] = e["g"], e["diff"]
        if eff.get("g") is None:
            warnings.append(f"{c}: g not computed (an arm has < 2 answers or no spread)")
        elif no_persona and eff.get("no_persona_g") is not None and abs(eff["no_persona_g"]) >= abs(eff["g"]):
            warnings.append(f"{c}: the no_persona arm's |g| ({abs(eff['no_persona_g']):.2f}) is at least the "
                            f"persona arm's ({abs(eff['g']):.2f}); the personas do not carry this effect")
        effects[c] = eff
    out = {"runner": "experiment", "design": "between" if between else "within", "control": control,
           "question": q, "conditions": cond_out, "effects": effects, "n_personas": len(pids),
           "samples": max(1, int(samples)), "seed": seed,
           "refused": sum(v["refused"] for v in cond_out.values()),
           "unparsed": sum(v["unparsed"] for v in cond_out.values()),
           "warnings": warnings, "usage": usage}
    if between:
        out["assignment"] = assign
    return to_jsonable(out)


def _lookup(human, study, cond):
    if human is None:
        return None
    for key in ((study, cond), f"{study}|{cond}"):
        if isinstance(human, Mapping) and key in human:
            return human[key]
    if isinstance(human, Mapping) and isinstance(human.get(study), Mapping):
        return human[study].get(cond)
    if isinstance(human, Mapping) and cond in human and not isinstance(human[cond], Mapping):
        return human[cond]
    if isinstance(human, Mapping) and isinstance(human.get(cond), Mapping) and "g" in human[cond]:
        return human[cond]
    return None


def experiment_frame(results, human=None) -> pd.DataFrame:
    """EvalFrame (kind 'continuous'), one row per effect: `results` is one `experiment` result, a list of them
    or {study: result}. human: {condition: g}, {study: {condition: g}}, {(study, condition): g} or
    {"study|condition": g}; a value may be {"g": .., "se": ..}. Columns: unit "<study>|<condition>", item
    (condition), group (study = the family for `exaggeration_ratio`), truth (human g, NaN when absent),
    truth_se, pred_g (simulated g), pred_no_persona_g, sim_se, sim_diff, n_treat, n_control."""
    if isinstance(results, Mapping) and results.get("runner") == "experiment":
        items = [("study1", results)]
    elif isinstance(results, Mapping):
        items = [(str(k), v) for k, v in results.items()]
    else:
        items = [(f"study{i + 1}", r) for i, r in enumerate(results)]
    rows = []
    for study, res in items:
        for cond, e in (res.get("effects") or {}).items():
            h = _lookup(human, study, cond)
            hg, hse = (h.get("g"), h.get("se")) if isinstance(h, Mapping) else (h, None)
            rows.append({"unit": f"{study}|{cond}", "item": cond, "group": study,
                         "truth": float(hg) if hg is not None else np.nan,
                         "truth_se": float(hse) if hse is not None else np.nan,
                         "pred_g": e.get("g") if e.get("g") is not None else np.nan,
                         "pred_no_persona_g": e.get("no_persona_g") if e.get("no_persona_g") is not None
                         else np.nan,
                         "sim_se": e.get("se") if e.get("se") is not None else np.nan,
                         "sim_diff": e.get("diff") if e.get("diff") is not None else np.nan,
                         "n_treat": e.get("n_treat"), "n_control": e.get("n_control")})
    cols = ["unit", "item", "group", "truth", "truth_se", "pred_g", "pred_no_persona_g", "sim_se", "sim_diff",
            "n_treat", "n_control"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- calibration against human effects
def _keys_values(x, keys=None):
    if isinstance(x, pd.Series):
        return (list(x.index), x.to_numpy(float)) if keys is None else (keys, x.reindex(keys).to_numpy(float))
    if isinstance(x, Mapping):
        if keys is None:
            keys = list(x)
        return keys, np.array([np.nan if x.get(k) is None else float(x.get(k)) for k in keys], float)
    arr = np.asarray(x, dtype=float).ravel()
    return keys, arr


def _align(sim, human, *others):
    """-> (keys or None, sim, human, others...) as float arrays over the shared, non-missing effects."""
    keyed = isinstance(sim, (Mapping, pd.Series))
    if keyed:
        ks, s = _keys_values(sim)
        hk = list(human.index) if isinstance(human, pd.Series) else list(human) if isinstance(human, Mapping) else None
        if hk is None:
            raise ValueError("sim_g is keyed (dict / Series); pass human_g keyed the same way")
        ks = [k for k in ks if k in set(hk)]
        _, s = _keys_values(sim, ks)
        _, h = _keys_values(human, ks)
    else:
        ks, s = None, np.asarray(sim, float).ravel()
        h = np.asarray(human, float).ravel()
        if len(h) != len(s):
            raise ValueError(f"sim_g has {len(s)} effects, human_g {len(h)}")
    rest = []
    for o in others:
        if o is None:
            rest.append(None)
        elif np.isscalar(o):
            rest.append(np.full(len(s), float(o)) if not isinstance(o, str) else np.full(len(s), o, object))
        elif keyed and isinstance(o, (Mapping, pd.Series)):
            rest.append(np.array([(o.get(k) if isinstance(o, Mapping) else o.get(k)) for k in ks], object))
        else:
            a = np.asarray(o, object).ravel()
            if len(a) != len(s):
                raise ValueError(f"an aligned argument has {len(a)} values for {len(s)} effects")
            rest.append(a)
    ok = np.isfinite(s) & np.isfinite(h)
    ks = [k for k, m in zip(ks, ok) if m] if ks is not None else None
    return (ks, s[ok], h[ok], *[None if r is None else r[ok] for r in rest])


def _f(a) -> np.ndarray:
    """Float array with None / unparseable -> NaN."""
    return np.array([np.nan if x is None else float(x) for x in np.asarray(a, object).ravel()], float)


def _clusters(family, k: int) -> np.ndarray:
    if family is None:
        return np.arange(k)
    _, inv = np.unique(np.asarray([str(f) for f in family]), return_inverse=True)
    return inv


def _cluster_draws(cl: np.ndarray, B: int, seed: int):
    """B index arrays: clusters resampled with replacement, members concatenated."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cl == c) for c in np.unique(cl)]
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        yield np.concatenate([groups[i] for i in pick])


def _stretch(n_clusters: int, alpha: float) -> float:
    """Few-cluster widening of a cluster-percentile CI: t(G-1) / z * sqrt(G / (G-1)). The plain percentile
    interval covered 0.72-0.90 with 3-6 families (simulated); stretched around the estimate the ratio, slope and
    human-N CIs cover 0.89-0.97 at 3-30 families; the factor tends to 1 as G grows."""
    g = int(n_clusters)
    if g < 2:
        return float("nan")
    from scipy.stats import t
    return float(t.ppf(1 - alpha / 2, g - 1) / _z(alpha) * math.sqrt(g / (g - 1)))


def _pct(draws, alpha: float, est=None, n_clusters: int | None = None):
    """Percentile CI of bootstrap draws; with `est` and `n_clusters` both ends are pushed away from the estimate
    by `_stretch(n_clusters)` (few-cluster correction)."""
    d = np.asarray([x for x in draws if x is not None and np.isfinite(x)], float)
    if len(d) < 10:
        return [None, None]
    lo, hi = float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))
    if est is not None and n_clusters is not None:
        c = _stretch(n_clusters, alpha)
        if not np.isfinite(c):
            return [None, None]
        lo, hi = min(lo, est - c * (est - lo)), max(hi, est + c * (hi - est))
    return [lo, hi]


def _ratio(s, h, min_human: float = 0.0):
    m = np.abs(h) > max(min_human, 1e-12)
    return float(np.median(np.abs(s[m]) / np.abs(h[m]))) if m.any() else None


def _slope(s, h):
    """OLS sim = a + b human -> (b, a); (None, None) without spread in human."""
    if len(h) < 2 or np.var(h) <= 0:
        return None, None
    b = float(np.cov(h, s, ddof=0)[0, 1] / np.var(h))
    return b, float(s.mean() - b * h.mean())


def _slope_corrected(s, h, se2: float, min_rel: float = 0.2):
    """Slope sim ~ human corrected for the human sampling noise (errors in the regressor attenuate OLS by the
    reliability (var(h) - mean se^2) / var(h)) -> (slope, reliability); slope None when reliability < min_rel."""
    if len(h) < 2 or np.var(h, ddof=1) <= 0:
        return None, None
    vh = float(np.var(h, ddof=1))
    rel = (vh - se2) / vh
    if rel < min_rel:
        return None, rel
    return float(np.cov(h, s, ddof=1)[0, 1] / (vh - se2)), rel


def _affine(s, h):
    """OLS human = a + b sim (the prediction direction) -> (a, b); None without spread in sim."""
    if len(s) < 2 or np.var(s) <= 0:
        return None
    b = float(np.cov(s, h, ddof=0)[0, 1] / np.var(s))
    return float(h.mean() - b * s.mean()), b


def _loo(s, h, cl, min_human: float = 0.0):
    """Leave-one-cluster-out ratio and affine predictions for every effect (NaN when the rest cannot fit)."""
    pr, pa, scales = np.full(len(s), np.nan), np.full(len(s), np.nan), np.full(len(s), np.nan)
    for c in np.unique(cl):
        out, keep = cl == c, cl != c
        r = _ratio(s[keep], h[keep], min_human)
        if r:
            pr[out], scales[out] = s[out] / r, r
        f = _affine(s[keep], h[keep])
        if f is not None:
            pa[out] = f[0] + f[1] * s[out]
    return pr, pa, scales


def exaggeration_ratio(sim_g, human_g, family=None, *, human_se=None, B: int = 2000, seed: int = 0,
                       alpha: float = 0.05, min_human: float = 0.0) -> dict:
    """How much larger the simulated effects are than the human ones. sim_g / human_g: aligned sequences, or
    dicts / Series keyed by effect (intersected); family: a label per effect (study / paper; same alignment)
    for the clustered CIs and the leave-one-family-out (LOO) scale. Effects with |human| <= min_human (and
    exact zeros) are left out of the ratio only. human_se (one value or one per effect, same alignment): adds
    slope_corrected, the slope with the human sampling noise taken out of var(human) (OLS on noisy human
    effects is pulled toward 0, so it understates the exaggeration).

    Returns {k, families, ratio (median |sim| / |human|), ratio_ci, ratio_sum (sum |sim| / sum |human|),
    slope / intercept (OLS sim ~ human; 1 = right size) with slope_ci, slope0 (through the origin) with
    slope0_ci, r (Pearson), sign_agreement, affine {a, b} (human ~ a + b sim, the correction fit),
    verdict ("exaggerates" | "understates" | "about right" | "too few"), loo {scale: median / min / max of
    the LOO ratios, mae_raw, mae_ratio, mae_affine}}, and with human_se: slope_corrected, slope_corrected_ci,
    reliability. CIs: percentile over clusters resampled with replacement, widened for few clusters (`_stretch`;
    simulated coverage at 3-30 families: ratio 0.93-0.97, corrected slope 0.91-0.95; the plain percentile
    interval covered 0.72-0.90 at 3-6). The raw slope's CI misses the true slope when the human SEs are not
    small (0.68 at 20 families with se 0.1): read slope_corrected then."""
    _, s, h, fam, hse = _align(sim_g, human_g, family, human_se)
    if hse is not None:
        hse = _f(hse)
        hse = hse if np.isfinite(hse).any() else None      # missing SEs: mean se^2 over the effects that have one
    k = len(s)
    cl = _clusters(fam, k)
    nf = int(len(np.unique(cl))) if k else 0
    out: dict[str, Any] = {"k": k, "families": nf if family is not None else None}
    if k < 3:
        out.update({"ratio": _ratio(s, h, min_human) if k else None, "verdict": "too few",
                    "note": "fewer than 3 effects with both a simulated and a human value"})
        return to_jsonable(out)
    ratio = _ratio(s, h, min_human)
    slope, icpt = _slope(s, h)
    slope0 = float((s * h).sum() / (h * h).sum()) if (h * h).sum() > 0 else None
    aff = _affine(s, h)
    sc_fit = _slope_corrected(s, h, float(np.nanmean(hse ** 2))) if hse is not None else (None, None)
    rd, sd, s0d, scd = [], [], [], []
    for idx in _cluster_draws(cl, B, seed):
        rd.append(_ratio(s[idx], h[idx], min_human))
        sd.append(_slope(s[idx], h[idx])[0])
        hh = (h[idx] ** 2).sum()
        s0d.append(float((s[idx] * h[idx]).sum() / hh) if hh > 0 else None)
        if sc_fit[0] is not None:
            se2 = hse[idx] ** 2
            scd.append(_slope_corrected(s[idx], h[idx], float(np.nanmean(se2)) if np.isfinite(se2).any() else 0.0)[0])
    ratio_ci = _pct(rd, alpha, ratio, nf)
    pr, pa, scales = _loo(s, h, cl, min_human)
    if ratio_ci[0] is not None and ratio_ci[0] > 1:
        verdict = "exaggerates"
    elif ratio_ci[1] is not None and ratio_ci[1] < 1:
        verdict = "understates"
    else:
        verdict = "about right"
    sc = scales[np.isfinite(scales)]
    out.update({
        "ratio": ratio, "ratio_ci": ratio_ci,
        "ratio_sum": float(np.abs(s).sum() / np.abs(h).sum()) if np.abs(h).sum() > 0 else None,
        "slope": slope, "intercept": icpt, "slope_ci": _pct(sd, alpha, slope, nf), "slope0": slope0,
        "slope0_ci": _pct(s0d, alpha, slope0, nf),
        "r": float(np.corrcoef(s, h)[0, 1]) if s.std() > 0 and h.std() > 0 else None,
        "sign_agreement": float(np.mean(np.sign(s) == np.sign(h))),
        "affine": {"a": aff[0], "b": aff[1]} if aff else None, "verdict": verdict,
        "loo": {"by": "family" if family is not None and nf > 1 else "effect",
                "scale": {"median": float(np.median(sc)), "min": float(sc.min()), "max": float(sc.max())}
                if len(sc) else None,
                "mae_raw": float(np.mean(np.abs(s - h))),
                "mae_ratio": float(np.nanmean(np.abs(pr - h))) if np.isfinite(pr).any() else None,
                "mae_affine": float(np.nanmean(np.abs(pa - h))) if np.isfinite(pa).any() else None},
    })
    if hse is not None:
        out.update({"slope_corrected": sc_fit[0], "reliability": sc_fit[1],
                    "slope_corrected_ci": [None, None]})
        if sc_fit[0] is None:
            out["note_slope"] = "the human effects barely vary beyond their own noise (reliability < 0.2): no corrected slope"
        elif sum(x is None for x in scd) > alpha / 2 * len(scd):
            out["note_slope"] = "reliability < 0.2 in too many bootstrap draws: the corrected slope has no CI"
        else:
            out["slope_corrected_ci"] = _pct(scd, alpha, sc_fit[0], nf)
    if family is not None and nf < 2:
        out["note"] = "one family only: the LOO scale leaves out single effects and the CI resamples effects"
    return to_jsonable(out)


def correct_effects(sim_g, fit=None, *, method: str = "ratio", human_g=None, family=None, min_human: float = 0.0):
    """Shrink simulated effects by the measured exaggeration. Two ways:

    - human_g given (LOO, the honest one): every effect is corrected with a ratio / affine fit from the OTHER
      effects (other families when `family` is given), never with its own human value. Effects whose rest
      cannot be fitted come back NaN.
    - fit given (applying an earlier measurement to new effects): a ratio (float, or an `exaggeration_ratio`
      dict) with method='ratio' -> sim / ratio; an (a, b) pair or the dict with method='affine' ->
      a + b * sim.
    Returns the same shape as sim_g (dict / Series keyed the same way, else an array)."""
    if method not in ("ratio", "affine"):
        raise ValueError("method must be 'ratio' or 'affine'")
    keys, s = _keys_values(sim_g)
    if human_g is not None:
        ks, s_ok, h, fam = _align(sim_g, human_g, family)
        cl = _clusters(fam, len(s_ok))
        pr, pa, _ = _loo(s_ok, h, cl, min_human)
        pred = pr if method == "ratio" else pa
        if keys is not None:
            got = dict(zip(ks, pred))
            vals = np.array([got.get(k, np.nan) for k in keys], float)
        else:
            vals = np.full(len(s), np.nan)
            ok = np.isfinite(s) & np.isfinite(np.asarray(human_g, float).ravel())
            vals[ok] = pred
    else:
        if fit is None:
            raise ValueError("pass human_g= (LOO) or fit= (a ratio, an (a, b) pair or an exaggeration_ratio dict)")
        if method == "ratio":
            r = fit.get("ratio") if isinstance(fit, Mapping) else float(fit)
            if not r:
                raise ValueError("the fit has no usable ratio")
            vals = s / float(r)
        else:
            if isinstance(fit, Mapping):
                aff = fit.get("affine") or fit
                a, b = aff["a"], aff["b"]
            else:
                a, b = fit
            vals = float(a) + float(b) * s
    if isinstance(sim_g, pd.Series):
        return pd.Series(vals, index=sim_g.index, name=sim_g.name)
    if isinstance(sim_g, Mapping):
        return {k: (None if not np.isfinite(v) else float(v)) for k, v in zip(keys, vals)}
    return vals


def _n_from(err: float, h2: float):
    return (4.0 + h2 / 2.0) / err if err > 0 else None


def human_equivalent_n(sim_g, human_g, human_se, family=None, *, B: int = 2000, seed: int = 0,
                       alpha: float = 0.05) -> dict:
    """Total human sample size (two equal arms) whose sampling error matches the simulation's error.

    err = mean((sim - human)^2) - mean(human_se^2): the squared distance to the human estimates minus the part
    that is the human studies' own noise. A two-arm study with N people has var(d) ~ (4 + d^2/2) / N, so the
    simulation is worth N = (4 + mean(human^2)/2) / err humans per effect. err <= 0 (the simulation is within
    the human noise) gives n None with a note, not infinity. human_se: one value or one per effect. CI: cluster
    bootstrap over `family` (else effects) of err, widened for few clusters like `exaggeration_ratio`'s CIs
    (simulated coverage of the true N 0.89-0.96 at 3-30 families; the plain percentile 0.76-0.79 at 4);
    an err CI end <= 0 is an open upper end (ci[1] None).
    Returns {n, n_per_arm, ci, mse, noise_var, err_var, k, note}."""
    _, s, h, se, fam = _align(sim_g, human_g, human_se, family)
    se = _f(se)
    ok = np.isfinite(se)
    s, h, se = s[ok], h[ok], se[ok]
    fam = fam[ok] if fam is not None else None
    k = len(s)
    out: dict[str, Any] = {"k": k, "n": None, "n_per_arm": None, "ci": [None, None]}
    if k < 2:
        out["note"] = "fewer than 2 effects with a simulated value, a human value and a human SE"
        return to_jsonable(out)
    mse, noise = float(np.mean((s - h) ** 2)), float(np.mean(se ** 2))
    h2 = float(np.mean(h ** 2))
    err = mse - noise
    out.update({"mse": mse, "noise_var": noise, "err_var": err})
    n = _n_from(err, h2)
    if n is None:
        out["note"] = ("the simulation's error is not above the human sampling noise (mse <= mean se^2): it cannot "
                       "be told apart from a replication, so no finite N is claimed")
        return to_jsonable(out)
    cl = _clusters(fam, k)
    errs = []
    for idx in _cluster_draws(cl, B, seed):
        errs.append(float(np.mean((s[idx] - h[idx]) ** 2) - np.mean(se[idx] ** 2)))
    lo_err, hi_err = _pct(errs, alpha, err, len(np.unique(cl)))
    out.update({"n": n, "n_per_arm": n / 2, "ci": [_n_from(float(hi_err), h2), _n_from(float(lo_err), h2)],
                "note": "N = total two-arm human sample; ci[1] None = the error may be within the human noise"
                if lo_err <= 0 else "N = total two-arm human sample"})
    return to_jsonable(out)


def _fmt(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.{digits}f}"


def _ci(ci, fmt=None) -> str:
    fmt = fmt or _fmt
    if not ci or ci[0] is None and ci[1] is None:
        return "–"
    return f"[{fmt(ci[0])}, {fmt(ci[1]) if ci[1] is not None else 'open'}]"


def _fmt_n(x) -> str:
    return "–" if x is None else f"{x:,.0f}"


def effect_section(sim_g, human_g, human_se=None, family=None, *, B: int = 2000, seed: int = 0,
                   alpha: float = 0.05, title: str = "Effect sizes: exaggeration and human equivalent") -> dict:
    """Evidence-extras section (kind 'effect_calibration'; render with `evidence.section_markdown`): the
    exaggeration ratio and slope with family-clustered CIs, MAE raw vs LOO ratio / affine correction, and
    'worth ~N humans' for the raw and the LOO-corrected simulation next to the baseline that predicts every
    effect as the mean of the other human effects (LOO). When the simulation is not worth more humans than that
    baseline, a note says so. Extra keys: fit (exaggeration_ratio), human_n {raw, corrected, baseline}."""
    fit = exaggeration_ratio(sim_g, human_g, family, human_se=human_se, B=B, seed=seed, alpha=alpha)
    rows = [["exaggeration ratio (median |sim| / |human|)", _fmt(fit.get("ratio")), _ci(fit.get("ratio_ci")),
             fit.get("verdict", "")],
            ["slope sim ~ human (1 = right size)", _fmt(fit.get("slope")), _ci(fit.get("slope_ci")),
             "family-clustered CI" if family is not None else "effect-resampled CI"]]
    if fit.get("slope_corrected") is not None:
        rows.append(["slope corrected for human noise", _fmt(fit["slope_corrected"]), _ci(fit.get("slope_corrected_ci")),
                     f"reliability {_fmt(fit.get('reliability'))}"])
    loo = fit.get("loo") or {}
    notes = [f"{fit['k']} effects" + (f" in {fit['families']} families" if fit.get("families") else "")
             + "; corrections are leave-one-" + ("family" if loo.get("by") == "family" else "effect")
             + "-out, so no effect is corrected with its own human value."]
    if loo:
        rows += [["MAE raw (g units)", _fmt(loo.get("mae_raw"), 3), "–", ""],
                 ["MAE ratio-corrected (LOO)", _fmt(loo.get("mae_ratio"), 3), "–",
                  f"scale {_fmt((loo.get('scale') or {}).get('median'))}"],
                 ["MAE affine-corrected (LOO)", _fmt(loo.get("mae_affine"), 3), "–", ""]]
    human_n = {}
    if human_se is not None and fit["k"] >= 3:
        ks, s, h, se, fam = _align(sim_g, human_g, human_se, family)
        cl = _clusters(fam, len(s))
        corr, _, _ = _loo(s, h, cl)
        base = np.array([h[cl != c].mean() if (cl != c).any() else np.nan for c in cl], float)
        for name, pred, label in (("raw", s, "worth ~N humans (raw simulation)"),
                                  ("corrected", corr, "worth ~N humans (ratio-corrected, LOO)"),
                                  ("baseline", base, "baseline: mean of the other human effects (LOO)")):
            m = np.isfinite(pred)
            r = human_equivalent_n(pred[m], h[m], _f(se)[m],
                                   None if fam is None else fam[m], B=B, seed=seed, alpha=alpha)
            human_n[name] = r
            rows.append([label, _fmt_n(r.get("n")) if r.get("n") is not None else "none", _ci(r.get("ci"), _fmt_n),
                         "within human noise" if r.get("n") is None and r.get("err_var") is not None
                         and r["err_var"] <= 0 else ""])
        def worth(r):          # within the human noise beats any finite N; no estimate (too few) is 0
            if r.get("n") is not None:
                return r["n"]
            return math.inf if r.get("err_var") is not None and r["err_var"] <= 0 else 0.0
        best, bn = max(worth(human_n["raw"]), worth(human_n["corrected"])), worth(human_n["baseline"])
        if math.isinf(bn):
            notes.append("The mean-effect baseline is already within the human noise: these effects cannot show "
                         "that the simulation adds information.")
        elif best <= bn:
            notes.append(f"The simulation is not worth more humans than the mean-effect baseline (~{_fmt_n(bn)}): its "
                         "N reflects that effects are small, not information from the model.")
        notes.append("N = total two-arm human sample whose sampling error matches the error left after removing "
                     "the human studies' own noise (var d ~ 4/N).")
    else:
        notes.append("No human SEs given: the 'worth ~N humans' rows need human_se.")
    return {"kind": "effect_calibration", "title": title, "columns": ["measure", "value", "95% CI", "note"],
            "rows": rows, "notes": notes, "fit": fit, "human_n": to_jsonable(human_n)}


__all__ = ["experiment", "experiment_frame", "assign_conditions", "hedges_g", "paired_g", "exaggeration_ratio",
           "correct_effects", "human_equivalent_n", "effect_section"]
