"""Per-person prediction prompts: L0 (no persona), L1 (persona = the person's own survey answers),
L2 (L1 + the person's past behaviour in the same items), and an SSR variant (one first-person sentence per
item, mapped to a probability by embedding similarity). Port of talp73/kod/llm/{istem.py, kos.py, ssr.py,
kalibre.py, tekrar.py}.

Output for every person x item: probability (0-1) that the event happens in the target window.

    spec = PredictionSpec()                                    # generic wording; AMAZON = the measured prompts
    res = predict_people(llm, people, ["BOOKS", "COFFEE"], level="L1", spec=spec)
    frame = prediction_frame(res, truth={"u1": {"BOOKS": 1, "COFFEE": 0}, ...})
    frame = batch_calibrate(frame, prior_rates={"BOOKS": 0.3, "COFFEE": 0.1}, col="pred_L1")

`people`: {person_id: profile} where profile is a rendered string or a mapping {question: answer}
(rendered by `profile_text`), or a list of persona-like objects (simengine.sim.personas.as_personas).
Profiles must hold only what is known before the prediction window.

Measured (talp73, Open e-commerce 1.0, 400 people x 20 categories x 3 seeds, claude-haiku-5-5): L1 accuracy
0.689-0.698, below "always no" (0.764) and a survey-feature model (0.771); L2 0.792-0.802 beats always-no but
stays below a purchase-history model. After batch calibration the L1-vs-survey-model gap includes zero.
See docs/modules/sim.md for the full table and provenance.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ._stats import to_jsonable
from .panel import REFRAME_NOTE_GENERIC, Usage, ask_many
from .parse import parse_item_probs, parse_text_fields
from .personas import as_personas
from .ssr import SCALES, SSR

RETRY_SUFFIX = "\n\n(Attempt {n}: reply with the JSON object only.)"


@dataclass
class PredictionSpec:
    """All wording of the L0/L1/L2/SSR prompts. Placeholders: {subject}, {period}, {items}, {n},
    {history_period}. Defaults are a neutral purchase-style wording; AMAZON reproduces the measured prompts."""
    system: str = ("You simulate the behavior of one specific real person. "
                   "Give your best calibrated probability estimates. Reply with a JSON object only.")
    system_l0: str = ("You simulate the behavior of people from the target population. "
                      "Give your best calibrated probability estimates. Reply with a JSON object only.")
    system_ssr: str = ("You simulate one specific real person and answer in their own voice. "
                       "Reply with a JSON object only.")
    profile_intro: str = "Survey answers of this person about themselves:"
    history_intro: str = "Their past behavior for these items, {history_period}:"
    subject: str = "this person"
    subject_l0: str = "a randomly selected person from the target population"
    period: str = "the next 6 months"
    history_period: str = "before the prediction window"
    task: str = ("Task: consider {subject} during {period}. For each item below, estimate the probability (0-100) "
                 "that the event described happens at least once during that period.\n\n"
                 "Items (key: description):\n{items}\n\n"
                 "Return a JSON object with exactly these {n} keys and integer values from 0 to 100.")
    task_ssr: str = ("Task: speak as this person about themselves during {period}. For each item below, write one "
                     "short first-person sentence (at most 20 words, no numbers or percentages) saying whether the "
                     "event described happened at least once during that period.\n\n"
                     "Items (key: description):\n{items}\n\n"
                     "Return a JSON object with exactly these {n} keys and one sentence (a string) as each value.")
    ssr_anchors: list = field(default_factory=lambda: SCALES["purchased"][0])
    labels: dict = field(default_factory=dict)          # item key -> description
    questions: dict | None = None                       # profile column -> question label (profile_text)
    none_fields: tuple = ()                             # missing answer rendered "None"
    comma_fields: tuple = ()                            # "a,b" rendered "a, b"

    def label(self, key: str) -> str:
        return self.labels.get(key, str(key).replace("_", " ").capitalize())


# talp73 v1.0 protocol, verbatim (window 2022-H2). AMAZON_QUESTIONS/LABELS are the public field labels of the
# Open e-commerce 1.0 survey and Amazon category codes; no respondent data.
AMAZON = PredictionSpec(
    system=("You simulate the Amazon shopping behavior of one specific real person, a US Amazon customer. "
            "Give your best calibrated probability estimates. Reply with a JSON object only."),
    system_l0=("You simulate the Amazon shopping behavior of US Amazon customers. "
               "Give your best calibrated probability estimates. Reply with a JSON object only."),
    system_ssr=("You simulate one specific real person, a US Amazon customer, and answer in their own voice. "
                "Reply with a JSON object only."),
    profile_intro="Survey answers of this person about themselves and their Amazon account:",
    history_intro="Their Amazon purchase history in these categories, {history_period}:",
    subject="this person's Amazon account (including anyone who shares it)",
    subject_l0="a randomly selected US adult Amazon customer's account",
    period="July 1 - December 31, 2022",
    history_period="January 2018 - June 2022",
    task=("Task: consider {subject} during {period}. For each product category below, estimate the probability "
          "(0-100) that the account purchased at least one item in that category on Amazon during those 6 months.\n\n"
          "Categories (key: description):\n{items}\n\n"
          "Return a JSON object with exactly these {n} keys and integer values from 0 to 100."),
    task_ssr=("Task: speak as this person about their Amazon account (including anyone who shares it) "
              "during {period}. For each product category below, write one short "
              "first-person sentence (at most 20 words, no numbers or percentages) saying whether the account "
              "bought at least one item in that category on Amazon during those 6 months.\n\n"
              "Categories (key: description):\n{items}\n\n"
              "Return a JSON object with exactly these {n} keys and one sentence (a string) as each value."),
    labels={
        "ABIS_BOOK": "Books",
        "HEALTH_PERSONAL_CARE": "Health & personal care products",
        "PORTABLE_ELECTRONIC_DEVICE_COVER": "Covers/cases for tablets, e-readers and other portable devices",
        "CELLULAR_PHONE_CASE": "Cell phone cases",
        "ELECTRONIC_CABLE": "Electronic cables (USB, HDMI, etc.)",
        "CHARGING_ADAPTER": "Charging adapters / chargers",
        "NUTRITIONAL_SUPPLEMENT": "Nutritional supplements / vitamins",
        "WRITING_INSTRUMENT": "Pens, pencils, markers",
        "DRINKING_CUP": "Drinking cups, mugs, tumblers",
        "CLEANING_AGENT": "Cleaning products",
    },
)

AMAZON_QUESTIONS = {
    "Q-demos-age": "Age group",
    "Q-demos-hispanic": "Of Spanish, Hispanic, or Latino origin",
    "Q-demos-race": "Race",
    "Q-demos-education": "Highest education completed",
    "Q-demos-income": "Household income before taxes, past 12 months",
    "Q-demos-gender": "Gender",
    "Q-sexual-orientation": "Sexual orientation",
    "Q-demos-state": "State of residence",
    "Q-amazon-use-howmany": "Number of people who log in and order with this Amazon account",
    "Q-amazon-use-hh-size": "Household size",
    "Q-amazon-use-how-oft": "How often the account orders Amazon deliveries",
    "Q-substance-use-cigarettes": "Someone in the household / on the account smokes cigarettes regularly",
    "Q-substance-use-marijuana": "Someone in the household / on the account smokes marijuana regularly",
    "Q-substance-use-alcohol": "Someone in the household / on the account drinks alcohol regularly",
    "Q-personal-diabetes": "Someone in the household / on the account has diabetes",
    "Q-personal-wheelchair": "Someone in the household / on the account uses a wheelchair",
    "Q-life-changes": "Life changes in 2021 (self or account sharer)",
    "Q-sell-YOUR-data": "Should Amazon be able to sell YOUR purchase data to other companies",
    "Q-sell-consumer-data": "Should companies be able to sell consumer purchase data",
    "Q-small-biz-use": "Should small businesses get free access to purchase data",
    "Q-census-use": "Should the U.S. Census Bureau use purchase data",
    "Q-research-society": "Should researchers be able to use purchase data to study societal change",
}
AMAZON_NONE_FIELDS = ("Q-life-changes",)                  # missing means "None", not "(no answer)"
AMAZON_COMMA_FIELDS = ("Q-life-changes", "Q-demos-race")  # multi-select "a,b" rendered "a, b"
AMAZON = replace(AMAZON, questions=AMAZON_QUESTIONS, none_fields=AMAZON_NONE_FIELDS, comma_fields=AMAZON_COMMA_FIELDS)


def _missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def profile_text(answers: Mapping[str, Any], questions: Mapping[str, str] | None = None, *,
                 none_fields: Sequence[str] = (), comma_fields: Sequence[str] = ()) -> str:
    """Survey answers -> "- <question>: <answer>" lines (talp73 istem.profil). With `questions`
    (column -> label) the lines follow its order and include unanswered ones as "(no answer)"."""
    cols = list(questions) if questions is not None else list(answers)
    out = []
    for c in cols:
        q = questions[c] if questions is not None else c
        v = answers.get(c)
        if _missing(v):
            v = "None" if c in none_fields else "(no answer)"
        else:
            v = str(v).strip()
            if c in comma_fields:
                v = v.replace(",", ", ")
        out.append(f"- {q}: {v}")
    return "\n".join(out)


def history_text(summary: Mapping[str, Any], items: Sequence[str], *, recent_period: str = "the last 12 months"
                 ) -> str:
    """Per-item past-behaviour summary -> L2 history lines (talp73 kos.gecmis_metinleri).
    summary[item]: a string (used as is), 0/None/missing ("never purchased"), or a mapping with
    count, days, last, count_recent."""
    out = []
    for k in items:
        r = summary.get(k)
        if isinstance(r, str):
            out.append(f"{k}: {r}")
        elif not r or (isinstance(r, Mapping) and not r.get("count")):
            out.append(f"{k}: never purchased")
        else:
            out.append(f"{k}: {int(r['count'])} items on {int(r.get('days', r['count']))} different days, "
                       f"last purchase {r.get('last', 'unknown')}, {int(r.get('count_recent', 0))} items in "
                       f"{recent_period}")
    return "\n".join(out)


def _items_block(items: Sequence[str], spec: PredictionSpec) -> str:
    return "\n".join(f"{k}: {spec.label(k)}" for k in items)


def _task(spec: PredictionSpec, items, subject: str, ssr: bool = False) -> str:
    t = spec.task_ssr if ssr else spec.task
    return t.format(subject=subject, period=spec.period, items=_items_block(items, spec), n=len(items))


def prompt_l0(items: Sequence[str], spec: PredictionSpec | None = None) -> tuple[str, str]:
    """(system, user) for the no-persona population prior."""
    spec = spec or PredictionSpec()
    return spec.system_l0, _task(spec, items, spec.subject_l0)


def prompt_l1(profile: str, items: Sequence[str], spec: PredictionSpec | None = None) -> tuple[str, str]:
    """(system, user) with the person's own survey answers as persona."""
    spec = spec or PredictionSpec()
    return spec.system, f"{spec.profile_intro}\n{profile}\n\n" + _task(spec, items, spec.subject)


def prompt_l2(profile: str, items: Sequence[str], history: str, spec: PredictionSpec | None = None
              ) -> tuple[str, str]:
    """(system, user) with survey answers + past behaviour in the same items."""
    spec = spec or PredictionSpec()
    return spec.system, (f"{spec.profile_intro}\n{profile}\n\n"
                         f"{spec.history_intro.format(history_period=spec.history_period)}\n{history}\n\n"
                         + _task(spec, items, spec.subject))


def prompt_ssr(profile: str, items: Sequence[str], spec: PredictionSpec | None = None) -> tuple[str, str]:
    """(system, user) asking for one first-person sentence per item (SSR arm, talp73 v1.7d)."""
    spec = spec or PredictionSpec()
    return spec.system_ssr, f"{spec.profile_intro}\n{profile}\n\n" + _task(spec, items, spec.subject, ssr=True)


def _profiles(people, spec: PredictionSpec, questions=None) -> dict[str, str]:
    if isinstance(people, Mapping):
        q = questions if questions is not None else spec.questions
        out = {}
        for pid, v in people.items():
            out[str(pid)] = (profile_text(v, q, none_fields=spec.none_fields, comma_fields=spec.comma_fields)
                             if isinstance(v, Mapping) else str(v))
        return out
    return {p.pid: p.text("en") for p in as_personas(people)}


def _call_all(llm, jobs, parse, *, max_tokens, retries, sample_kw, usage, tag, ask_kw=None):
    """jobs: [(key, sample, (system, user))] -> {key: {sample: parsed}}, failures, retried count."""
    out: dict = {}
    pending = list(jobs)
    failed, retried = [], 0
    for attempt in range(retries + 1):
        if not pending:
            break
        by_sample: dict[int, list] = {}
        for j in pending:
            by_sample.setdefault(j[1], []).append(j)
        nxt = []
        for s, group in by_sample.items():
            prompts = [(sy, us + (RETRY_SUFFIX.format(n=attempt + 1) if attempt else "")) for _, _, (sy, us) in group]
            kw = {sample_kw: s} if sample_kw and s else {}
            rs = ask_many(llm, prompts, max_tokens=max_tokens, usage=usage, tag=tag, reframe=REFRAME_NOTE_GENERIC,
                          **{**kw, **(ask_kw or {})})
            for (key, smp, pr), r in zip(group, rs):
                try:
                    out.setdefault(key, {})[smp] = parse(getattr(r, "text", None))
                except ValueError as e:
                    nxt.append((key, smp, pr, str(e)))
        retried += len(nxt) if attempt < retries else 0
        pending = [(k, s, p) for k, s, p, _ in nxt]
        failed = [(k, s, err) for k, s, _, err in nxt]
    return out, failed, retried


def predict_people(llm, people=None, items: Sequence[str] = (), *, level: str = "L1",
                   spec: PredictionSpec | None = None, histories: Mapping[str, Any] | None = None,
                   questions: Mapping[str, str] | None = None, samples: int = 1, l0_samples: int = 10,
                   max_tokens: int = 2000, retries: int = 1, sample_kw: str | None = "sample",
                   ssr: SSR | None = None, ask_kw: dict | None = None) -> dict:
    """Run one prediction arm. level: 'L0' (people optional; l0_samples independent samples, averaged, one
    prior for everyone), 'L1', 'L2' (needs `histories`: {pid: text or history_text summary}), 'SSR'.

    samples: independent samples per person (passed as `sample=j` to backends that key their cache on it;
    set sample_kw=None for a backend that rejects the keyword). An unparsable or incomplete reply is re-asked up
    to `retries` times with a short 'JSON only' suffix; persons still failing are listed in 'failed'.
    Returns {"level", "items", "p": {pid | "_all": {item: p}}, "samples": [{unit, sample, item, p}], "failed",
    "retried", "usage"} (JSON-serialisable).
    Calls go through `sim.panel.ask_many`: a soft refusal is not cached and is re-asked once with a
    research-forecast note, an empty reply (EmptyResponseError) once with a larger max_tokens; a person still
    without a reply goes through the normal `retries` and then into 'failed'. `ask_kw` passes ask_many options."""
    level = level.upper()
    if level not in ("L0", "L1", "L2", "SSR"):
        raise ValueError("level must be L0, L1, L2 or SSR")
    items = [str(k) for k in items]
    if not items:
        raise ValueError("items is empty")
    spec = spec or PredictionSpec()
    usage = Usage()
    jobs = []
    if level == "L0":
        pr = prompt_l0(items, spec)
        jobs = [("_all", j, pr) for j in range(max(1, l0_samples))]
    else:
        prof = _profiles(people if people is not None else {}, spec, questions)
        if not prof:
            raise ValueError("people is empty")
        if level == "L2":
            if histories is None:
                raise ValueError("L2 needs histories")
            missing = [p for p in prof if p not in histories]
            if missing:
                raise ValueError(f"no history for {len(missing)} people (e.g. {missing[0]})")
        for pid, pf in prof.items():
            if level == "L1":
                pr = prompt_l1(pf, items, spec)
            elif level == "L2":
                h = histories[pid]
                pr = prompt_l2(pf, items, h if isinstance(h, str) else history_text(h, items), spec)
            else:
                pr = prompt_ssr(pf, items, spec)
            jobs += [(pid, j, pr) for j in range(max(1, samples))]
    if level == "SSR":
        parse: Callable = lambda t: parse_text_fields(t, items, strict=True)  # noqa: E731
    else:
        parse = lambda t: parse_item_probs(t, items, strict=True)  # noqa: E731
    got, failed, retried = _call_all(llm, jobs, parse, max_tokens=max_tokens, retries=retries,
                                     sample_kw=sample_kw, usage=usage, tag=f"predict:{level}", ask_kw=ask_kw)
    if level == "SSR":
        ssr = ssr or SSR()
        cells = [(k, s, it, d[it]) for k, by in got.items() for s, d in by.items() for it in items]
        if cells:
            P = ssr.distribution([c[3] for c in cells], [list(spec.ssr_anchors)])
            pv = SSR.to_probability(P)
            texts = {}
            for (k, s, it, txt), v in zip(cells, pv):
                got[k][s][it] = float(v)
                texts.setdefault(k, {}).setdefault(s, {})[it] = txt
        else:
            texts = {}
    rows = [{"unit": k, "sample": s, "item": it, "p": float(d[it])}
            for k, by in got.items() for s, d in sorted(by.items()) for it in items if it in d]
    p = {}
    for k, by in got.items():
        p[k] = {it: float(np.mean([d[it] for d in by.values() if it in d])) for it in items
                if any(it in d for d in by.values())}
    fail_units = sorted({k for k, _, _ in failed} - set(p))
    out = {"method": f"predict_{level}", "level": level, "items": items, "p": p, "samples": rows,
           "n_samples": max(1, l0_samples if level == "L0" else samples),
           "failed": fail_units, "failed_samples": [{"unit": k, "sample": s, "error": e} for k, s, e in failed],
           "retried": retried, "calls": len(jobs) + retried, "usage": usage.as_dict()}
    if level == "SSR":
        out["texts"] = texts
    return to_jsonable(out)


def prediction_frame(result: dict, truth: Mapping[str, Mapping[str, Any]] | None = None,
                     units: Sequence[str] | None = None, method: str | None = None,
                     segments: Mapping[str, str] | None = None) -> pd.DataFrame:
    """EvalFrame (kind 'binary'), one row per person x item: unit, item, truth, pred_<method> (default the
    level), segment, group (= item). An L0 result is broadcast to `units` (or to the persons in `truth`)."""
    method = method or result.get("level", "llm")
    p = result.get("p") or {}
    truth = truth or {}
    if "_all" in p and len(p) == 1:
        us = list(units or truth)
        if not us:
            us = ["_all"]
        p = {u: p["_all"] for u in us}
    elif units is not None:
        p = {u: p.get(u, {}) for u in units}
    segments = segments or {}
    rows = []
    for u, d in p.items():
        for it in result.get("items", list(d)):
            t = (truth.get(u) or {}).get(it)
            v = d.get(it)
            rows.append({"unit": u, "item": it, "truth": np.nan if t is None else float(t),
                         f"pred_{method}": np.nan if v is None else float(v),
                         "segment": segments.get(u, "all"), "group": it})
    return pd.DataFrame(rows)


def _logit(x, clip):
    x = np.clip(np.asarray(x, float), clip, 1 - clip)
    return np.log(x / (1 - x))


def batch_calibrate(frame: pd.DataFrame, prior_rates: Mapping[str, float], *, col: str = "p",
                    item_col: str = "item", out_col: str | None = None, clip: float = 1e-3) -> pd.DataFrame:
    """Batch (contextual) calibration, talp73 v1.7a: z = logit(p) - mean_item(logit p) + logit(prior_rate[item]),
    p_cal = sigmoid(z). Uses no labels of the scored people: prior_rates must come from outside the sample
    (e.g. other people's base rate in the previous period). The per-item mean is over the rows in `frame`, so
    score the whole batch at once. Returns a copy with `out_col` (default: `col` overwritten)."""
    f = frame.copy()
    missing = set(f[item_col]) - set(prior_rates)
    if missing:
        raise ValueError(f"prior_rates missing items: {sorted(missing)[:5]}")
    z = pd.Series(_logit(f[col].to_numpy(), clip), index=f.index)
    z = z - z.groupby(f[item_col]).transform("mean")
    lo = f[item_col].map(lambda k: float(_logit(prior_rates[k], clip)))
    f[out_col or col] = 1 / (1 + np.exp(-(z + lo)))
    return f


def retest_stats(samples, *, unit: str = "unit", item: str = "item", sample: str = "sample", p: str = "p"
                 ) -> dict:
    """Test-retest of repeated samples (talp73 tekrar.py). `samples`: long table or a predict_people result.
    Returns within-cell sd, share of cells with identical samples, mean pairwise correlation over cells,
    per-item between-person sd vs within-cell sd and retest r, and the Spearman-Brown reliability of the
    k-sample mean."""
    if isinstance(samples, dict):
        samples = samples.get("samples") or []
    d = pd.DataFrame(samples)
    if d.empty:
        return {"n_units": 0, "k": 0}
    d = d.drop_duplicates([unit, sample, item])
    w = d.pivot_table(index=[unit, item], columns=sample, values=p)
    K = w.shape[1]
    out = {"n_units": int(w.index.get_level_values(0).nunique()), "n_cells": int(len(w)), "k": int(K),
           "within_cell_sd": float(w.std(axis=1).mean()) if K > 1 else None,
           "identical_share": float((w.nunique(axis=1) == 1).mean())}
    if K < 2:
        return out

    def pair_r(m):
        rs = []
        for i, j in itertools.combinations(m.columns, 2):
            a, b = m[i].to_numpy(), m[j].to_numpy()
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() > 1 and a[ok].std() > 0 and b[ok].std() > 0:
                rs.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
        return float(np.mean(rs)) if rs else None

    out["retest_r"] = pair_r(w)
    per = []
    for k, g in w.groupby(level=1):
        per.append({"item": k, "between_person_sd": float(g.mean(axis=1).std()),
                    "within_cell_sd": float(g.std(axis=1).mean()), "retest_r": pair_r(g)})
    out["items"] = per
    rr = [x["retest_r"] for x in per if x["retest_r"] is not None]
    if rr:
        r = float(np.mean(rr))
        out["item_retest_r"] = r
        out["spearman_brown"] = (K * r) / (1 + (K - 1) * r) if not math.isclose(1 + (K - 1) * r, 0) else None
    return out


__all__ = ["PredictionSpec", "AMAZON", "AMAZON_QUESTIONS", "AMAZON_NONE_FIELDS", "AMAZON_COMMA_FIELDS",
           "profile_text", "history_text", "prompt_l0", "prompt_l1", "prompt_l2", "prompt_ssr",
           "predict_people", "prediction_frame", "batch_calibrate", "retest_stats"]
