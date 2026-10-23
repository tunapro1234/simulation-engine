"""Per-person prediction prompts: L0 (no persona), L1 (persona = the person's own survey answers),
L2 (L1 + the person's past behaviour in the same items), and an SSR variant (one first-person sentence per
item, mapped to a probability by embedding similarity). Port of talp73/kod/llm/{istem.py, kos.py, ssr.py,
kalibre.py, tekrar.py}.

Output for every person x item: probability (0-1) that the event happens in the target window; with
kind="numeric", a number per person x field inside that field's range (else None / NaN).

    spec = PredictionSpec()                                    # generic wording; AMAZON = the measured prompts
    res = predict_people(llm, people, ["BOOKS", "COFFEE"], level="L1", spec=spec)
    res = predict_people(people, ["BOOKS", "COFFEE"], llm=llm, level="L1")   # same, keyword form
    frame = prediction_frame(res, truth={"u1": {"BOOKS": 1, "COFFEE": 0}, ...})
    frame = batch_calibrate(frame, prior_rates={"BOOKS": 0.3, "COFFEE": 0.1}, col="pred_L1", out_col="pred_L1_cal")
    fields = [NumericField("log_rate", -5, 1), NumericField("spend", 0, 200, unit="USD")]
    res = predict_people(people, llm=llm, kind="numeric", fields=fields)   # one JSON reply per person, all fields
    frame = prediction_frame(res)        # person x field; out of range / unparsed -> NaN, counted in res["invalid"]

`people` (accepted types):
  * list[Persona] (e.g. simengine.persona.from_history(level="L1")) or other persona-like items
    (simengine.sim.personas.as_persona forms): the persona's answers dict (`.answers` or extra["answers"])
    rendered by `profile_text`, else its profile text (authored.consumer_profile; otherwise name/traits/
    narrative) without the id line and without the survey header. history_data is left out: pass `histories=`.
  * {pid: {column: answer}}: rendered by `profile_text` (with `questions` / spec.questions).
  * {pid: text}: used as is, minus a leading id line or survey header (spec.profile_intro is added anyway).
Anything else raises TypeError. Profiles must hold only what is known before the prediction window.
`histories` (L2) may also be a persona.summarize_purchases frame (see `histories_from_summary`).

AMAZON.labels covers 10 of the 20 measured categories (books, health & personal care, portable device covers,
phone cases, electronic cables, charging adapters, nutritional supplements, writing instruments, drinking cups,
cleaning agents); other item keys are described by their key name and predict_people warns, listing them.

Measured (talp73, Open e-commerce 1.0, 400 people x 20 categories x 3 seeds, claude-haiku-5-5): L1 accuracy
0.689-0.698, below "always no" (0.764) and a survey-feature model (0.771); L2 0.792-0.802 beats always-no but
stays below a purchase-history model. After batch calibration the L1-vs-survey-model gap includes zero.
See docs/modules/sim.md for the full table and provenance.
"""
from __future__ import annotations

import itertools
import math
import re
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..llm.jsonx import json_extract
from ._stats import to_jsonable
from .panel import REFRAME_NOTE_GENERIC, Usage, ask_many
from .parse import parse_item_probs, parse_text_fields, unwrap
from .personas import PersonaView, as_persona
from .ssr import SCALES, SSR

RETRY_SUFFIX = "\n\n(Attempt {n}: reply with the JSON object only.)"
SURVEY_HEADERS = ("Survey answers of this person:",)  # simengine.persona.history.PROFILE_HEADER
# summarize_purchases column -> history_text key
SUMMARY_TO_HISTORY = {"items": "count", "days": "days", "last_purchase": "last", "items_recent": "count_recent"}
_SUMMARY_COLS = {"category", "quantity", *SUMMARY_TO_HISTORY}
PEOPLE_TYPES = "people must be list[Persona], {pid: {column: answer}} or {pid: text}"


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
    system_numeric: str = ("You simulate the behavior of one specific real person. "
                           "Give your best numeric estimates. Reply with a JSON object only.")
    system_numeric_l0: str = ("You simulate the behavior of people from the target population. "
                              "Give your best numeric estimates. Reply with a JSON object only.")
    task_numeric: str = ("Task: consider {subject} during {period}. For each quantity below, give your best "
                         "estimate as a number inside the stated range.\n\n"
                         "Quantities (key: description (range)):\n{items}\n\n"
                         "Return a JSON object with exactly these {n} keys and a number as each value.")
    ssr_anchors: list = field(default_factory=lambda: SCALES["purchased"][0])
    labels: dict = field(default_factory=dict)          # item key -> description
    questions: dict | None = None                       # profile column -> question label (profile_text)
    none_fields: tuple = ()                             # missing answer rendered "None"
    comma_fields: tuple = ()                            # "a,b" rendered "a, b"

    def label(self, key: str) -> str:
        return self.labels.get(key, str(key).replace("_", " ").capitalize())

    def unlabeled(self, items: Sequence[str]) -> list[str]:
        """Items with no explicit entry in `labels` (described by their key name)."""
        return [str(k) for k in items if k not in self.labels]


# talp73 v1.0 protocol, verbatim (window 2022-H2). AMAZON_QUESTIONS/LABELS are the public field labels of the
# Open e-commerce 1.0 survey and Amazon category codes; no respondent data.
# AMAZON.labels covers 10 of the 20 measured categories: books, health & personal care, portable device covers,
# phone cases, electronic cables, charging adapters, nutritional supplements, writing instruments, drinking
# cups, cleaning agents. Other keys are described by their key name and predict_people warns.
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
    count, days, last, count_recent (`histories_from_summary` builds these from a summarize_purchases frame).
    Lines use the item key; for display names use simengine.persona.purchase_history_text(rows, labels=...)
    and pass its text per person."""
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


@dataclass(frozen=True)
class NumericField:
    """One numeric target of predict_people(kind="numeric"): inclusive range [lo, hi], optional unit (shown after
    the range) and label (default spec.label(name))."""
    name: str
    lo: float
    hi: float
    unit: str = ""
    label: str = ""

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("NumericField.name must be a non-empty string")
        lo, hi = float(self.lo), float(self.hi)
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
            raise ValueError(f"field {self.name!r}: need finite lo < hi, got {self.lo}..{self.hi}")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)


def numeric_fields(fields) -> list[NumericField]:
    """NumericField, (name, lo, hi[, unit]) tuples, {name, lo, hi, unit?, label?} dicts, or sim.FieldSpec with
    kind int/float and min/max -> validated list (non-empty, unique names)."""
    out: list[NumericField] = []
    for f in fields or []:
        if isinstance(f, NumericField):
            out.append(f)
        elif isinstance(f, Mapping):
            out.append(NumericField(**f))
        elif isinstance(f, (tuple, list)):
            out.append(NumericField(*f))
        elif getattr(f, "kind", None) in ("int", "float"):      # sim.constrained.FieldSpec
            if f.min is None or f.max is None:
                raise ValueError(f"field {f.name!r}: numeric prediction needs both min and max")
            out.append(NumericField(f.name, f.min, f.max, label=getattr(f, "label", "") or ""))
        else:
            raise ValueError(f"not a numeric field: {f!r} (NumericField, (name, lo, hi), dict, or int/float FieldSpec)")
    if not out:
        raise ValueError("kind='numeric' needs fields=[NumericField(name, lo, hi), ...]")
    names = [f.name.strip().lower() for f in out]               # replies are matched case-insensitively
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate field names {dup}")
    return out


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def _fields_block(fields: Sequence[NumericField], spec: PredictionSpec) -> str:
    return "\n".join(f"{f.name}: {f.label or spec.label(f.name)} (number from {_fmt(f.lo)} to {_fmt(f.hi)}"
                     f"{' ' + f.unit if f.unit else ''})" for f in fields)


def prompt_numeric(profile: str | None, fields: Sequence[NumericField], spec: PredictionSpec | None = None,
                   history: str | None = None) -> tuple[str, str]:
    """(system, user) asking for every numeric field in one JSON object. profile None: the L0 population prior;
    history: the L2 past-behaviour block."""
    spec = spec or PredictionSpec()
    task = spec.task_numeric.format(subject=spec.subject if profile is not None else spec.subject_l0,
                                    period=spec.period, items=_fields_block(fields, spec), n=len(fields))
    if profile is None:
        return spec.system_numeric_l0, task
    head = f"{spec.profile_intro}\n{profile}\n\n"
    if history is not None:
        head += f"{spec.history_intro.format(history_period=spec.history_period)}\n{history}\n\n"
    return spec.system_numeric, head + task


def _parse_numeric(text) -> dict:
    """Reply -> the raw JSON object (ValueError when there is none, so _call_all re-asks); values are checked
    per field afterwards."""
    d = unwrap(json_extract(text) if text else None, None)
    if not isinstance(d, dict) or not d:
        raise ValueError("no JSON object in the reply")
    return d


_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_THOUSANDS_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_DECIMAL_COMMA_RE = re.compile(r"[-+]?\d+,\d+")


def _strict_num(v: Any, unit: str = "") -> float | None:
    """A JSON number, or a string that is one whole number ("-1.5", "1,250", "7,5", "120 USD", "$120", Unicode
    minus); None otherwise. Stricter than parse._num: "5-10", "7/10", "1,250" (-> 1.25), ".5" (-> 5), "−0.5"
    (-> 0.5), "45%" or a list / dict would otherwise be read as a wrong in-range number and counted ok."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.strip().replace("\u2212", "-")
    if unit and s.lower().endswith(unit.strip().lower()):
        s = s[: len(s) - len(unit.strip())].strip()
    s = s.lstrip("$\u20ac\u00a3\u20ba").strip()
    if _THOUSANDS_RE.fullmatch(s):
        s = s.replace(",", "")
    elif _DECIMAL_COMMA_RE.fullmatch(s):
        s = s.replace(",", ".")
    return float(s) if _NUM_RE.fullmatch(s) else None


def _check_numeric(d: Mapping, fields: Sequence[NumericField]) -> dict[str, tuple[float | None, str]]:
    """{field: (value or None, status)}; status ok | out_of_range | unparsed (key absent or not one number,
    see _strict_num)."""
    dn = {str(k).strip().lower(): v for k, v in d.items()}
    out = {}
    for f in fields:
        v = dn.get(f.name.strip().lower())
        x = _strict_num(v, f.unit)
        if x is None or not math.isfinite(x):
            out[f.name] = (None, "unparsed")
        elif not f.lo <= x <= f.hi:
            out[f.name] = (None, "out_of_range")
        else:
            out[f.name] = (x, "ok")
    return out


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


def _is_llm(x: Any) -> bool:
    return callable(getattr(x, "complete", None))


def _clean_profile(text: str, pid: str | None, spec: PredictionSpec) -> str:
    """Drop leading id lines ("u1", "### u1", "id: u1") and survey headers: the prompt adds spec.profile_intro
    itself and the raw id is not something the person said about themselves."""
    heads = {h.strip() for h in (*SURVEY_HEADERS, spec.profile_intro) if h}
    ids = set() if not pid else {pid, f"id: {pid}", f"pid: {pid}", f"persona_id: {pid}"}
    lines = str(text).strip("\n").splitlines()
    while lines and (not lines[0].strip() or lines[0].strip() in heads or lines[0].strip().lstrip("#").strip() in ids):
        lines.pop(0)
    return "\n".join(lines).strip()


def _get(obj, key: str):
    return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)


def _persona_answers(x) -> Mapping | None:
    ans = _get(x, "answers")
    if not isinstance(ans, Mapping):
        extra = _get(x, "extra")
        ans = extra.get("answers") if isinstance(extra, Mapping) else None
    return ans if isinstance(ans, Mapping) and ans else None


def _persona_profile(x, i: int, spec: PredictionSpec, q) -> tuple[str, str]:
    """One persona-like item -> (pid, profile): its answers dict (profile_text), else authored.consumer_profile,
    else the persona text (name/traits/narrative) without the id line and without history_data."""
    if not (isinstance(x, (str, Mapping, PersonaView)) or callable(getattr(x, "to_dict", None))
            or callable(getattr(x, "text", None)) or callable(getattr(x, "metin", None))):
        raise TypeError(f"{PEOPLE_TYPES}; item {i} is {type(x).__name__}")
    v = as_persona(x, i)
    ans = _persona_answers(x)
    if ans is not None:
        return v.pid, profile_text(ans, q, none_fields=spec.none_fields, comma_fields=spec.comma_fields)
    a = _get(x, "authored")
    if a is None and not isinstance(x, Mapping) and callable(getattr(x, "to_dict", None)):
        a = (x.to_dict() or {}).get("authored")
    cp = _get(a, "consumer_profile") if a is not None else None
    if cp:
        return v.pid, _clean_profile(cp, v.pid, spec)
    hist = _get(a, "history_data") if a is not None else None
    name = v.name if v.name and v.name != v.pid else ""
    head = ", ".join(t for t in (name, *v.attrs) if t)
    body = [b for b in v.body if b and b != hist]
    return v.pid, _clean_profile("\n".join([head, *body] if head else body), v.pid, spec)


def _profiles(people, spec: PredictionSpec, questions=None) -> dict[str, str]:
    """`people` (see module docstring) -> {pid: profile text}. Unsupported types raise TypeError."""
    q = questions if questions is not None else spec.questions
    if isinstance(people, pd.DataFrame) or isinstance(people, (str, bytes)):
        raise TypeError(f"{PEOPLE_TYPES}, got {type(people).__name__}")
    out: dict[str, str] = {}
    if isinstance(people, Mapping):
        for pid, v in people.items():
            pid = str(pid)
            if isinstance(v, Mapping) and _persona_answers(v) is None and "identity" not in v:
                out[pid] = profile_text(v, q, none_fields=spec.none_fields, comma_fields=spec.comma_fields)
            elif isinstance(v, str):
                out[pid] = _clean_profile(v, pid, spec)
            elif v is not None and not isinstance(v, (int, float, bool, list, tuple)):
                out[pid] = _persona_profile(v, 0, spec, q)[1]     # {pid: Persona}: the key is the id
            else:
                raise TypeError(f"{PEOPLE_TYPES}; value for {pid!r} is {type(v).__name__}")
        return out
    if not isinstance(people, (list, tuple)) and not hasattr(people, "__iter__"):
        raise TypeError(f"{PEOPLE_TYPES}, got {type(people).__name__}")
    for i, x in enumerate(people):
        pid, text = _persona_profile(x, i, spec, q)
        if pid in out:
            raise ValueError(f"duplicate persona id {pid!r}")
        out[pid] = text
    empty = [k for k, t in out.items() if not t.strip()]
    if empty:
        warnings.warn(f"{len(empty)} of {len(out)} people have an empty profile (e.g. {empty[0]!r}): the prompt "
                      "holds no persona; build personas with a survey table or answers (from_history level L1)",
                      stacklevel=3)
    return out


def histories_from_summary(summary: pd.DataFrame, id_col: str | None = None) -> dict[str, dict]:
    """`persona.summarize_purchases` frame -> {pid: {category: {count, days, last, count_recent}}} for
    `history_text` / predict_people(histories=...). Mapping (SUMMARY_TO_HISTORY): items -> count, days -> days,
    last_purchase -> last (str, None when never purchased), items_recent -> count_recent; quantity is unused.
    `id_col`: default the one column that is not a summary column."""
    if id_col is None:
        rest = [c for c in summary.columns if c not in _SUMMARY_COLS]
        if len(rest) != 1:
            raise ValueError(f"cannot tell the id column of the purchase summary (candidates {rest}); pass id_col")
        id_col = rest[0]
    need = {id_col, "category", *SUMMARY_TO_HISTORY} - set(summary.columns)
    if need:
        raise ValueError(f"purchase summary lacks columns {sorted(need)} (expected persona.summarize_purchases)")
    out: dict[str, dict] = {}
    for r in summary.to_dict("records"):
        d = {k: r[c] for c, k in SUMMARY_TO_HISTORY.items()}
        d["last"] = None if _missing(d["last"]) else str(d["last"])
        for k in ("count", "days", "count_recent"):
            d[k] = 0 if _missing(d[k]) else int(d[k])
        out.setdefault(str(r[id_col]), {})[str(r["category"])] = d
    return out


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


def predict_people(*args, llm=None, people=None, items: Sequence[str] | None = None, level: str = "L1",
                   spec: PredictionSpec | None = None, histories: Mapping[str, Any] | pd.DataFrame | None = None,
                   questions: Mapping[str, str] | None = None, samples: int = 1, l0_samples: int = 10,
                   max_tokens: int = 2000, retries: int = 1, sample_kw: str | None = "sample",
                   ssr: SSR | None = None, ask_kw: dict | None = None, kind: str = "probability",
                   fields: Sequence[Any] | None = None) -> dict:
    """Run one prediction arm. level: 'L0' (people optional; l0_samples independent samples, averaged, one
    prior for everyone), 'L1', 'L2' (needs `histories`), 'SSR'.

    kind="numeric" (levels L0/L1/L2): `fields` = [NumericField(name, lo, hi, unit?, label?)] (or tuples / dicts /
    int-float sim.FieldSpec, see `numeric_fields`) replaces `items`. One prompt per person asks for every field
    in one JSON object; all prompts go out in one `ask_many` batch (llm.map, return_exceptions=True). A value
    outside [lo, hi], absent or not a number is None (NaN in prediction_frame), never clipped or re-asked; a
    reply with no JSON object is re-asked like the probability arm. The result adds "kind", "fields" and
    "invalid" {field: {ok, out_of_range, unparsed, no_reply}} counted over replies (persons x samples) plus
    "n_invalid"; "p" holds the mean of the valid samples per person and field (None if none was valid) and
    "samples" rows carry "status". prediction_frame(result) gives one row per person x field with lo, hi, status.

    Call forms: predict_people(llm, people, items, ...) (legacy, llm first) or predict_people(people, items,
    llm=llm, ...) like survey / ab_test; giving the llm both ways raises TypeError.
    people: list[Persona] (e.g. persona.from_history(level="L1")), {pid: {column: answer}} or {pid: text}; see
    the module docstring. Anything else raises TypeError.
    histories: {pid: text or history_text summary {item: {count, days, last, count_recent}}}, or a
    persona.summarize_purchases frame (columns items, days, last_purchase, items_recent -> count, days, last,
    count_recent; see `histories_from_summary`). History lines use the item key; for display names render
    each person with persona.purchase_history_text(rows, labels=...) and pass {pid: text}.
    Items with no entry in spec.labels are described by their key name; when spec.labels is not empty (e.g.
    AMAZON, whose labels cover 10 of the 20 measured categories) one warning lists them. Labels are never invented.

    samples: independent samples per person (passed as `sample=j` to backends that key their cache on it;
    set sample_kw=None for a backend that rejects the keyword). An unparsable or incomplete reply is re-asked up
    to `retries` times with a short 'JSON only' suffix; persons still failing are listed in 'failed'.
    Returns {"level", "items", "p": {pid | "_all": {item: p}}, "samples": [{unit, sample, item, p}], "failed",
    "retried", "usage"} (JSON-serialisable).
    Calls go through `sim.panel.ask_many`: a soft refusal is not cached and is re-asked once with a
    research-forecast note, an empty reply (EmptyResponseError) once with a larger max_tokens; a person still
    without a reply goes through the normal `retries` and then into 'failed'. `ask_kw` passes ask_many options."""
    args = list(args)
    if llm is None:
        if not args:
            raise TypeError("predict_people() needs an llm: pass it first (legacy) or as llm=")
        llm = args.pop(0)
    elif args and _is_llm(args[0]):
        raise TypeError("predict_people() got the llm both positionally and as llm=; pass it one way")
    if len(args) > 2:
        raise TypeError(f"predict_people() takes people, items after the llm but {len(args)} were given")
    if len(args) >= 1:
        if people is not None:
            raise TypeError("predict_people() got multiple values for argument 'people'")
        people = args[0]
    if len(args) == 2:
        if items is not None:
            raise TypeError("predict_people() got multiple values for argument 'items'")
        items = args[1]
    level = level.upper()
    if level not in ("L0", "L1", "L2", "SSR"):
        raise ValueError("level must be L0, L1, L2 or SSR")
    if kind not in ("probability", "numeric"):
        raise ValueError(f"kind must be 'probability' or 'numeric', got {kind!r}")
    if kind == "numeric":
        return _predict_numeric(llm, people, items, fields, level=level, spec=spec or PredictionSpec(),
                                histories=histories, questions=questions, samples=samples, l0_samples=l0_samples,
                                max_tokens=max_tokens, retries=retries, sample_kw=sample_kw, ask_kw=ask_kw)
    if fields is not None:
        raise ValueError("fields= is for kind='numeric'; probability items go in items=")
    items = [] if items is None else [str(k) for k in items]     # arrays / Index / Series as before
    if not items:
        raise ValueError("items is empty")
    spec = spec or PredictionSpec()
    unlabeled = spec.unlabeled(items) if spec.labels else []    # no labels at all: key names by design
    if unlabeled:
        warnings.warn(f"{len(unlabeled)} of {len(items)} items have no entry in spec.labels and are described by "
                      f"their key name: {unlabeled}", stacklevel=2)
    if isinstance(histories, pd.DataFrame):
        histories = histories_from_summary(histories)
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
    out = {"method": f"predict_{level}", "level": level, "kind": "probability", "items": items, "p": p,
           "samples": rows,
           "n_samples": max(1, l0_samples if level == "L0" else samples),
           "failed": fail_units, "failed_samples": [{"unit": k, "sample": s, "error": e} for k, s, e in failed],
           "retried": retried, "calls": len(jobs) + retried, "usage": usage.as_dict()}
    if level == "SSR":
        out["texts"] = texts
    return to_jsonable(out)


def _predict_numeric(llm, people, items, fields, *, level, spec, histories, questions, samples, l0_samples,
                     max_tokens, retries, sample_kw, ask_kw) -> dict:
    """predict_people(kind="numeric"): one JSON object per person with every field (see predict_people)."""
    if level == "SSR":
        raise ValueError("kind='numeric' supports levels L0, L1, L2 (SSR maps sentences to probabilities)")
    if items is not None:
        raise ValueError("kind='numeric' takes fields=[NumericField(...)], not items")
    fs = numeric_fields(fields)
    names = [f.name for f in fs]
    if isinstance(histories, pd.DataFrame):
        histories = histories_from_summary(histories)
    jobs = []
    if level == "L0":
        pr = prompt_numeric(None, fs, spec)
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
            h = None
            if level == "L2":
                h = histories[pid]
                # a summary renders all its own items (field names are not purchase items)
                h = h if isinstance(h, str) else history_text(h, list(h))
            jobs += [(pid, j, prompt_numeric(pf, fs, spec, h)) for j in range(max(1, samples))]
    usage = Usage()
    got, failed, retried = _call_all(llm, jobs, _parse_numeric, max_tokens=max_tokens, retries=retries,
                                     sample_kw=sample_kw, usage=usage, tag=f"predict:numeric:{level}", ask_kw=ask_kw)
    inv = {n: {"ok": 0, "out_of_range": 0, "unparsed": 0, "no_reply": 0} for n in names}
    rows, p = [], {}
    for k, by in got.items():
        vals: dict[str, list] = {n: [] for n in names}
        for s, d in sorted(by.items()):
            for n, (x, st) in _check_numeric(d, fs).items():
                inv[n][st] += 1
                rows.append({"unit": k, "sample": s, "item": n, "p": x, "status": st})
                if x is not None:
                    vals[n].append(x)
        p[k] = {n: float(np.mean(v)) if v else None for n, v in vals.items()}
    for _ in failed:
        for n in names:
            inv[n]["no_reply"] += 1
    fail_units = sorted({k for k, _, _ in failed} - set(p))
    out = {"method": f"predict_numeric_{level}", "level": level, "kind": "numeric", "items": names,
           "fields": [{"name": f.name, "lo": f.lo, "hi": f.hi, "unit": f.unit, "label": f.label} for f in fs],
           "p": p, "samples": rows, "n_samples": max(1, l0_samples if level == "L0" else samples),
           "invalid": inv, "n_invalid": sum(c[s] for c in inv.values() for s in ("out_of_range", "unparsed",
                                                                                  "no_reply")),
           "failed": fail_units, "failed_samples": [{"unit": k, "sample": s, "error": e} for k, s, e in failed],
           "retried": retried, "calls": len(jobs) + retried, "usage": usage.as_dict()}
    return to_jsonable(out)


def prediction_frame(result: dict, truth: Mapping[str, Mapping[str, Any]] | None = None,
                     units: Sequence[str] | None = None, method: str | None = None,
                     segments: Mapping[str, str] | None = None) -> pd.DataFrame:
    """EvalFrame (kind 'binary'), one row per person x item: unit, item, truth, pred_<method> (default the
    level), segment, group (= item). An L0 result is broadcast to `units` (or to the persons in `truth`).
    A kind="numeric" result (EvalFrame kind 'numeric') adds lo, hi and status (ok | out_of_range | unparsed |
    no_reply; with several samples "ok" when any sample was valid, else the first sample's status)."""
    method = method or result.get("level", "llm")
    p = result.get("p") or {}
    truth = truth or {}
    numeric = result.get("kind") == "numeric"
    status: dict = {}
    if numeric:
        for r in sorted(result.get("samples") or [], key=lambda r: r.get("status") != "ok"):
            status.setdefault((r["unit"], r["item"]), r.get("status"))
    if "_all" in p and len(p) == 1:
        us = list(units or truth)
        if not us:
            us = ["_all"]
        status = {(u, it): st for u in us for (_, it), st in status.items()}
        p = {u: p["_all"] for u in us}
    elif units is not None:
        p = {u: p.get(u, {}) for u in units}
    segments = segments or {}
    rng = {f["name"]: f for f in result.get("fields") or []}
    rows = []
    for u, d in p.items():
        for it in result.get("items", list(d)):
            t = (truth.get(u) or {}).get(it)
            v = d.get(it)
            row = {"unit": u, "item": it, "truth": np.nan if t is None else float(t),
                   f"pred_{method}": np.nan if v is None else float(v),
                   "segment": segments.get(u, "all"), "group": it}
            if numeric:
                row.update(lo=rng.get(it, {}).get("lo"), hi=rng.get(it, {}).get("hi"),
                           status=status.get((u, it), "no_reply"))
            rows.append(row)
    return pd.DataFrame(rows)


def _logit(x, clip):
    x = np.clip(np.asarray(x, float), clip, 1 - clip)
    return np.log(x / (1 - x))


def batch_calibrate(frame: pd.DataFrame, prior_rates: Mapping[str, float], *, col: str = "p",
                    item_col: str = "item", out_col: str | None = None, clip: float = 1e-3) -> pd.DataFrame:
    """Batch (contextual) calibration, talp73 v1.7a: z = logit(p) - mean_item(logit p) + logit(prior_rate[item]),
    p_cal = sigmoid(z). Uses no labels of the scored people: prior_rates must come from outside the sample
    (e.g. other people's base rate in the previous period). The per-item mean is over the rows in `frame`, so
    score the whole batch at once. Returns a copy; `frame` itself is not modified. Overwrite: with out_col=None
    (the default) the calibrated values replace `col` in the copy, so the raw predictions are gone from it;
    pass out_col (e.g. "pred_L1_cal") to keep both columns side by side."""
    f = frame.copy()
    missing = set(f[item_col]) - set(prior_rates)
    if missing:
        raise ValueError(f"prior_rates missing items: {sorted(missing)[:5]}")
    z = pd.Series(_logit(f[col].to_numpy(), clip), index=f.index)
    z = z - z.groupby(f[item_col]).transform("mean")
    lo = f[item_col].map(lambda k: float(_logit(prior_rates[k], clip)))
    f[out_col or col] = 1 / (1 + np.exp(-(z + lo)))
    return f


def _positions(pos, item, k: int) -> np.ndarray:
    """Option positions for one item: pos None -> 0..k-1; a sequence; or {item: sequence}."""
    if isinstance(pos, Mapping):
        pos = pos.get(item)
    x = np.arange(k, dtype=float) if pos is None else np.asarray(pos, float)
    if len(x) != k:
        raise ValueError(f"item {item!r}: {len(x)} positions for {k} options")
    return x


def _w1_pos(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """1-Wasserstein distance between two vectors over options at positions x, divided by the position range
    (equals `w1_ordinal` for equally spaced positions)."""
    a, b = a / a.sum(), b / b.sum()
    span = float(x.max() - x.min()) or 1.0
    return float((np.abs(np.cumsum(a) - np.cumsum(b))[:-1] * np.diff(x)).sum() / span)


def _reduce_distribution(d: pd.DataFrame, unit: str, item: str, sample: str, p: str, pos) -> tuple:
    """Distribution rows -> (rows with scalar E[position] in `p`, within-cell W1 summary)."""
    d = d.copy()
    vec = [None if v is None or (isinstance(v, float) and v != v) else np.asarray(v, float) for v in d[p]]
    d[p] = [np.nan if v is None else float(v @ _positions(pos, it, len(v)) / v.sum())
            for v, it in zip(vec, d[item])]
    d["_vec"] = vec
    w1 = []
    for (_, it), g in d[[unit, item, "_vec"]].groupby([unit, item], sort=False):
        vs = [v for v in g["_vec"] if v is not None]
        pairs = [_w1_pos(a, b, _positions(pos, it, len(a))) for a, b in itertools.combinations(vs, 2)]
        if pairs:
            w1.append(float(np.mean(pairs)))
    return d.drop(columns="_vec"), {"within_cell_w1": float(np.mean(w1)) if w1 else None,
                                    "within_cell_w1_max": float(np.max(w1)) if w1 else None}


def retest_stats(samples, *, unit: str = "unit", item: str = "item", sample: str = "sample", p: str = "p",
                 kind: str = "probability", pos=None) -> dict:
    """Test-retest of repeated samples (talp73 tekrar.py). `samples`: long table or a predict_people result.
    Returns within-cell sd, share of cells with identical samples, mean pairwise correlation over cells,
    per-item between-person sd vs within-cell sd and retest r, and the Spearman-Brown reliability of the
    k-sample mean.
    kind="distribution": `p` holds a probability vector per row (e.g. `segment_distribution(..., repeats=k)`
    result, whose "samples" rows use unit = segment). Each vector is reduced to E[position] (pos: None -> option
    index 0..K-1, a sequence of positions, or {item: positions}) before the statistics above, and the result
    adds `within_cell_w1` (mean pairwise W1 between the samples of a cell, divided by the position range; the
    `w1_ordinal` scale for equally spaced options) and `within_cell_w1_max`; rows with a `filled` column add
    `filled_share` (uniform filler counts as an answer here: drop rows with filled > 0 to retest real replies)."""
    if kind not in ("probability", "distribution"):
        raise ValueError(f"kind must be 'probability' or 'distribution', got {kind!r}")
    if isinstance(samples, dict):
        samples = samples.get("samples") or []
    d = pd.DataFrame(samples)
    if d.empty:
        return {"n_units": 0, "k": 0}
    d = d.drop_duplicates([unit, sample, item])
    extra: dict = {}
    if kind == "distribution":
        filled = d["filled"] if "filled" in d.columns else None
        d, extra = _reduce_distribution(d, unit, item, sample, p, pos)
        extra["reduced_to"] = "E[position]"
        if filled is not None:                   # segment_distribution rows: share of the vectors that is filler
            extra["filled_share"] = float(pd.to_numeric(filled, errors="coerce").mean())
    elif d[p].map(lambda v: isinstance(v, (list, tuple, np.ndarray))).any():
        raise ValueError(f"column {p!r} holds probability vectors; pass kind='distribution'")
    w = d.pivot_table(index=[unit, item], columns=sample, values=p)
    K = w.shape[1]
    out = {"n_units": int(w.index.get_level_values(0).nunique()), "n_cells": int(len(w)), "k": int(K),
           "within_cell_sd": float(w.std(axis=1).mean()) if K > 1 else None,
           "identical_share": float((w.nunique(axis=1) == 1).mean()), **extra}
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


__all__ = ["PredictionSpec", "NumericField", "numeric_fields", "prompt_numeric", "AMAZON", "AMAZON_QUESTIONS",
           "AMAZON_NONE_FIELDS", "AMAZON_COMMA_FIELDS",
           "profile_text", "history_text", "histories_from_summary", "prompt_l0", "prompt_l1", "prompt_l2", "prompt_ssr",
           "predict_people", "prediction_frame", "batch_calibrate", "retest_stats"]
