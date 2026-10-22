"""Direct effect forecasts: ask the model for every condition's effect against control (no personas), over a few
framings x independent samples, with a hidden arm (study not named) and an optional named arm (study named) whose
difference is a memorisation probe.

One call per arm x framing x sample returns all conditions in one JSON object ({"effects": {id: number}}), so the
model compares the conditions side by side. Answers are averaged per condition and arm. The hidden arm is the
forecast; the named arm only exists to show whether naming the study moves the numbers (recognition /
memorisation). With measured effects, `memorisation_check` scores both arms and reports named - hidden accuracy.

    conds = {"Control": "No message.", "norms": "Most people in your country support ...", "fear": "..."}
    res = effect_forecast(llm, conds, "support for climate policy (0-100)", scale=(0, 100),
                          named="Climate Psychology Many Labs (Vlasceanu et al.)")
    res.table                # condition, hidden, named, named_minus_hidden, sd_<arm>, n_<arm>
    res.probe                # named - hidden: mean, mean |diff|, largest gap, Spearman between arms
    res.refused, res.unparsed, res.failed, res.invalid
    frame = effect_frame(res, truth={"norms": 2.1, "fear": -0.4})     # EvalFrame: pred_effect_hidden / _named
    memorisation_check(res, truth)       # Spearman per arm, named - hidden gap, paired bootstrap CI over conditions

Profile simulation instead of a direct question: `simengine.research.hypotheses.forecast_experiment` (Hewitt et
al. 2024 protocol: simulated participants per condition from demographic profiles and a prompt bank, effects =
condition mean - control mean, paired se). Use that when the effect should come from simulated respondents; use
this when the forecast is the model's own judgment of the effect (G1 rehearsal: the direct arm, L1).

Refusals (no JSON, refusal wording) are re-asked once with a research-forecast note (`ask_many`); a reply with no
JSON is re-asked once with "Return only valid JSON". What still fails is counted, never imputed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._stats import to_jsonable, to_number
from .panel import REFRAME_NOTE_GENERIC, Usage, ask_many
from .parse import parse_json

ARMS = ("hidden", "named")
SYSTEM = ("You forecast the results of social-science experiments. Give your best numeric estimate for every "
          "condition, even when unsure. Respond with ONLY a JSON object.")
JSON_ONLY_SUFFIX = "\n\nReturn only valid JSON, with no other text."

# {study}: "" in the hidden arm, " (<named>)" in the named arm. {context}: " " + context, or "".
FRAMINGS = (
    "A randomized experiment{study} tested several interventions against a control condition.{context} "
    "Each participant was randomly assigned to one condition, and the outcome was measured afterwards.",
    "You are forecasting the results of a randomized experiment{study}.{context} Each participant saw exactly "
    "one of the conditions below.",
    "Researchers compared several messages in a randomized trial{study}.{context} Think about how large effects "
    "usually are in experiments like this: most are small, and some are zero or negative.",
)

TEMPLATE = """{head}

Control condition ({control}): {control_text}
Treatment conditions:
{treatments}

Outcome: {outcome}{scale_line}
For each treatment condition, estimate its average effect on the outcome compared with the control group: the treatment group's mean minus the control group's mean, in outcome units (negative if it lowers the outcome).
Return ONLY a JSON object with exactly these keys ({ids}):
{{"effects": {example}}}"""

DEFAULT_CONTROL_TEXT = "no intervention"


@dataclass
class EffectForecast:
    """Result of `effect_forecast`. `table`: one row per treatment condition (condition, hidden, sd_hidden,
    n_hidden, and with a named arm named, sd_named, n_named, named_minus_hidden). `draws`: one row per valid
    number (arm, framing, sample, condition, effect). Counts are per call: `refused` (still refused after the
    re-ask), `unparsed` (no usable JSON after the JSON-only re-ask), `failed` (empty / backend error);
    `invalid` counts single condition values that were missing, non-numeric or out of range."""
    table: pd.DataFrame
    draws: pd.DataFrame
    probe: dict
    conditions: list[str]
    control: str
    named: str | None
    planned: int
    calls: int
    refused: int
    unparsed: int
    failed: int
    invalid: int
    usage: dict = field(default_factory=dict)
    prompts: dict = field(default_factory=dict)   # {arm: [user prompt per framing]}

    def to_dict(self) -> dict:
        return to_jsonable({
            "method": "effect_forecast", "conditions": self.conditions, "control": self.control,
            "named": self.named, "table": self.table.to_dict("records"), "draws": self.draws.to_dict("records"),
            "probe": self.probe, "planned": self.planned, "calls": self.calls, "refused": self.refused,
            "unparsed": self.unparsed, "failed": self.failed, "invalid": self.invalid, "usage": self.usage,
            "prompts": self.prompts})


def _scale_line(scale) -> tuple[str, float | None]:
    """(prompt line, largest possible |effect|) from scale=None | (lo, hi) | (lo, hi, lo_label, hi_label) | str."""
    if scale is None:
        return "", None
    if isinstance(scale, str):
        return f"\nOutcome scale: {scale}.", None
    lo, hi = float(scale[0]), float(scale[1])
    if not hi > lo:
        raise ValueError(f"scale must be (lo, hi) with hi > lo, got {scale!r}")
    lab = f" ({scale[2]} = {lo:g}, {scale[3]} = {hi:g})" if len(scale) >= 4 else ""
    return (f"\nOutcome scale: {lo:g} to {hi:g}{lab}; an effect can range from {lo - hi:g} to {hi - lo:g}.",
            hi - lo)


def effect_prompt(conditions: Mapping[str, str], outcome: str, *, control: str = "Control", framing: str = FRAMINGS[0],
                  named: str | None = None, scale=None, context: str = "") -> str:
    """User prompt for one arm x framing. `named` set -> the named arm (study name in the framing)."""
    treat = [c for c in conditions if c != control]
    head = framing.format(study=f" ({named})" if named else "", context=f" {context.strip()}" if context else "")
    example = "{" + ", ".join(f'"{c}": 0.0' for c in treat[:2]) + (", ..." if len(treat) > 2 else "") + "}"
    return TEMPLATE.format(
        head=head, control=control, control_text=conditions.get(control) or DEFAULT_CONTROL_TEXT,
        treatments="\n".join(f"- {c}: {conditions[c]}" for c in treat), outcome=outcome,
        scale_line=_scale_line(scale)[0], ids=", ".join(f'"{c}"' for c in treat), example=example)


def _effects(obj) -> dict | None:
    """{"effects": {...}}, a flat {id: value} mapping, or a list of {"condition", "effect"} rows -> {id: raw}."""
    if isinstance(obj, Mapping):
        inner = obj.get("effects", obj)
        if isinstance(inner, Mapping):
            return dict(inner)
        obj = inner
    if isinstance(obj, list):
        out = {}
        for row in obj:
            if isinstance(row, Mapping) and "condition" in row:
                out[str(row["condition"])] = row.get("effect")
        return out or None
    return None


def _value(raw, bound: float | None) -> float | None:
    if isinstance(raw, Mapping):
        raw = raw.get("effect", raw.get("value"))
    v = to_number(raw)
    if v is None or (bound is not None and abs(v) > bound):
        return None
    return v


def _spearman(a, b) -> float:
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    ok = a.notna() & b.notna()
    if ok.sum() < 3 or a[ok].nunique() < 2 or b[ok].nunique() < 2:
        return float("nan")
    return float(a[ok].rank().corr(b[ok].rank()))


def effect_forecast(llm, conditions: Mapping[str, str], outcome: str, *, control: str = "Control",
                    framings: int | Sequence[str] = 3, samples: int = 2, named: str | None = None, scale=None,
                    context: str = "", max_tokens: int = 1024, parallel: int | None = None,
                    sample_kw: str | None = "sample", tag: str = "effect_forecast",
                    system: str = SYSTEM) -> EffectForecast:
    """Forecast every condition's effect against `control` directly (no personas).

    conditions: {id: stimulus text}; `control` may be one of them (its text describes the control group) or
        absent ("no intervention"). Every other id is a treatment and gets one effect.
    outcome: what was measured, e.g. "support for climate policy, 0-100 slider".
    framings: how many of the built-in FRAMINGS (1-3), or a sequence of templates with {study} and {context} slots.
    samples: independent draws per framing, passed as `sample=k` (k >= 1) to backends that key their cache on it;
        with sample_kw=None the prompt carries "(Forecast k of n.)" instead.
    named: study / dataset name. Given -> a second, named arm with the same framings and samples; `probe` then
        holds named - hidden per condition. The hidden arm is always run and is the forecast to use.
    scale: None, (lo, hi), (lo, hi, lo_label, hi_label) or a free-text description; with numbers, an effect whose
        size exceeds hi - lo counts as invalid.
    Calls: arms x framings x samples (+ one re-ask per refused or unparsable reply)."""
    if not conditions:
        raise ValueError("conditions is empty")
    conditions = {str(k): str(v) for k, v in conditions.items()}
    treat = [c for c in conditions if c != control]
    if not treat:
        raise ValueError("conditions has no treatment besides the control")
    tpls = list(FRAMINGS[:int(framings)]) if isinstance(framings, int) else [str(f) for f in framings]
    if not tpls or (isinstance(framings, int) and not 1 <= framings <= len(FRAMINGS)):
        raise ValueError(f"framings must be 1..{len(FRAMINGS)} or a non-empty list of templates")
    samples = int(samples)
    if samples < 1:
        raise ValueError("samples must be >= 1")
    bound = _scale_line(scale)[1]
    arms = [("hidden", None)] + ([("named", named)] if named else [])

    jobs, prompts = [], {}                     # jobs: (arm, framing index, sample, user)
    for arm, name in arms:
        prompts[arm] = [effect_prompt(conditions, outcome, control=control, framing=t, named=name, scale=scale,
                                      context=context) for t in tpls]
        for fi, user in enumerate(prompts[arm]):
            for k in range(samples):
                if sample_kw is None and samples > 1:
                    user_k = f"{user}\n(Forecast {k + 1} of {samples}.)"
                else:
                    user_k = user
                jobs.append((arm, fi, k, user_k))

    usage = Usage()
    replies: dict[int, dict] = {}
    refused = failed = 0
    retry: list[int] = []
    by_sample: dict[int, list[int]] = {}
    for i, j in enumerate(jobs):
        by_sample.setdefault(j[2], []).append(i)
    for k, idx in by_sample.items():
        kw = {sample_kw: k} if sample_kw and k else {}
        st: list = []
        rs = ask_many(llm, [(system, jobs[i][3]) for i in idx], max_tokens=max_tokens, usage=usage, tag=tag,
                      reframe=REFRAME_NOTE_GENERIC, parallel=parallel, statuses=st, errors=[], **kw)
        for i, r, s in zip(idx, rs, st):
            if s == "refused":
                refused += 1
            elif r is None:
                failed += 1
            else:
                got = _effects(parse_json(getattr(r, "text", None)))
                if got is None:
                    retry.append(i)
                else:
                    replies[i] = got
    unparsed = 0
    if retry:                                  # one JSON-only re-ask (a different prompt, so a different cache key)
        by_sample = {}
        for i in retry:
            by_sample.setdefault(jobs[i][2], []).append(i)
        for k, idx in by_sample.items():
            kw = {sample_kw: k} if sample_kw and k else {}
            st = []
            rs = ask_many(llm, [(system, jobs[i][3] + JSON_ONLY_SUFFIX) for i in idx], max_tokens=max_tokens,
                          usage=usage, tag=tag, reframe=REFRAME_NOTE_GENERIC, parallel=parallel, statuses=st,
                          errors=[], **kw)
            for i, r, s in zip(idx, rs, st):
                got = _effects(parse_json(getattr(r, "text", None))) if r is not None else None
                if got is None:
                    unparsed += 1
                    usage.unparsed_calls += 1
                else:
                    replies[i] = got

    rows, invalid = [], 0
    for i, got in replies.items():
        arm, fi, k, _ = jobs[i]
        low = {str(key).strip().lower(): v for key, v in got.items()}
        for c in treat:
            v = _value(got[c] if c in got else low.get(c.lower()), bound)
            if v is None:
                invalid += 1
            else:
                rows.append({"arm": arm, "framing": fi, "sample": k, "condition": c, "effect": v})
    draws = pd.DataFrame(rows, columns=["arm", "framing", "sample", "condition", "effect"])
    planned_per_arm = len(tpls) * samples

    table = pd.DataFrame({"condition": treat})
    for arm, _ in arms:
        g = draws[draws.arm == arm].groupby("condition")["effect"]
        table[arm] = table.condition.map(g.mean())
        table[f"sd_{arm}"] = table.condition.map(g.std(ddof=1))
        table[f"n_{arm}"] = table.condition.map(g.size()).fillna(0).astype(int)
    probe: dict[str, Any] = {"named": named, "status": "not_run", "n_planned_per_arm": planned_per_arm}
    if named:
        table["named_minus_hidden"] = table["named"] - table["hidden"]
        d = table["named_minus_hidden"].dropna()
        probe.update(status="ok" if len(d) else "no_data", n=int(len(d)),
                     mean_diff=float(d.mean()) if len(d) else math.nan,
                     mean_abs_diff=float(d.abs().mean()) if len(d) else math.nan,
                     largest=(str(table.loc[d.abs().idxmax(), "condition"]) if len(d) else None),
                     largest_diff=float(d.loc[d.abs().idxmax()]) if len(d) else math.nan,
                     spearman_arms=_spearman(table["hidden"], table["named"]))
    return EffectForecast(table=table, draws=draws, probe=to_jsonable(probe), conditions=treat, control=control,
                          named=named, planned=len(jobs), calls=usage.calls, refused=refused, unparsed=unparsed,
                          failed=failed, invalid=invalid, usage=usage.as_dict(), prompts=prompts)


def effect_frame(res: EffectForecast, truth: Mapping[str, float] | None = None) -> pd.DataFrame:
    """EvalFrame (kind 'continuous'), one row per treatment condition: truth (measured effect, NaN when not
    given) and pred_effect_hidden (+ pred_effect_named with a named arm). Compare the arms in `evaluate` with
    compare=[("pred_effect_named", "pred_effect_hidden")]."""
    truth = truth or {}
    t = res.table
    out = pd.DataFrame({"unit": t.condition, "item": t.condition, "group": t.condition,
                        "truth": [float(truth[c]) if c in truth and truth[c] is not None else np.nan
                                  for c in t.condition],
                        "pred_effect_hidden": t["hidden"]})
    if "named" in t:
        out["pred_effect_named"] = t["named"]
    return out.reset_index(drop=True)


def memorisation_check(res: EffectForecast, truth: Mapping[str, float], *, threshold: float = 0.2, B: int = 1000,
                       seed: int = 0, alpha: float = 0.05) -> dict:
    """Accuracy of each arm against measured effects (Spearman over conditions) and named - hidden, with a paired
    percentile bootstrap CI that resamples conditions. status "warning" when the gap is >= `threshold` (the G1
    pre-registration rule: named Spearman >= hidden + 0.2 reads as possible recognition) or its CI lies above 0;
    "ok" otherwise; "not_run" without a named arm; "too_few" under 4 conditions with truth and both arms."""
    out: dict[str, Any] = {"threshold": threshold, "named": res.named}
    if not res.named:
        return {**out, "status": "not_run"}
    f = effect_frame(res, truth).dropna(subset=["truth", "pred_effect_hidden", "pred_effect_named"])
    n = len(f)
    out["n"] = n
    if n < 4:
        return {**out, "status": "too_few"}
    y, h, m = (f[c].to_numpy(float) for c in ("truth", "pred_effect_hidden", "pred_effect_named"))
    sh, sm = _spearman(h, y), _spearman(m, y)
    gap = sm - sh
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        g = _spearman(m[i], y[i]) - _spearman(h[i], y[i])
        if math.isfinite(g):
            boots.append(g)
    lo, hi = (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))) if boots \
        else (math.nan, math.nan)
    warn = (math.isfinite(gap) and gap >= threshold) or (math.isfinite(lo) and lo > 0)
    return to_jsonable({**out, "status": "warning" if warn else "ok", "spearman_hidden": sh, "spearman_named": sm,
                        "named_minus_hidden": gap, "ci": [lo, hi], "B": len(boots)})


__all__ = ["effect_forecast", "effect_prompt", "effect_frame", "memorisation_check", "EffectForecast", "FRAMINGS",
           "SYSTEM", "TEMPLATE"]
