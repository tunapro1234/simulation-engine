"""Automated social science: an LLM proposes causal hypotheses, the library runs the experiment and fits it.

Two modes.

1. **Hypothesis loop** (Manning, Zhu & Horton 2024, "Automated Social Science: Language Models as Scientist and
   Subjects", arXiv:2404.11794). The LLM states a hypothesis as a linear structural causal model (SCM): a scenario,
   agents, one outcome measured by a survey question, and 1-3 exogenous causes, each operationalised with a kind,
   units and a short list of treatment levels. The library builds the full factorial design, simulates every cell
   (LLM interview, multi-agent LLM conversation ended by an LLM judge, or any vectorised numpy world such as
   `simengine.abm.market_sim`), fits the SCM by OLS (statsmodels if installed, else numpy) and compares the fitted
   path estimates with the LLM's own forecast of them (made before the simulation, as in the paper's Fig. A.11 task).
2. **Experiment outcome predictor** (Hewitt, Ashokkumar, Ghezae & Willer 2024, "Predicting Results of Social Science
   Experiments Using Large Language Models"). For each condition of a survey experiment the LLM simulates participants
   drawn from a profile table (prompt templates drawn from a bank, responses averaged per condition), or is asked
   for the effects directly; treatment effects are condition means minus the control mean. `score_effect_forecasts`
   and `score_condition_forecasts` give r, disattenuated r and sign accuracy against measured effects.

Every decision (accepted / rejected hypothesis parts and why, design size and subsampling, protocol, parse failures,
estimator, prediction hash before fitting) goes to a `DecisionLog` (`decisions.jsonl` in `workdir`).

    from simengine.llm.fake import FakeLLM
    from simengine.research.hypotheses import AutoSocialScience, linear_world
    llm = FakeLLM(my_responder)                     # real: se.llm.Claude(model=..., budget_usd=...)
    ass = AutoSocialScience(llm, "A buyer and a seller bargain over a used mug.", workdir="runs/ass",
                            simulator="interview", replicates=2)
    res = ass.run(n_hypotheses=1)
    res.runs[0].comparison      # term, coef, se, p, beta_std, predicted, ratio, sign_match, sig_match
    res.summary                 # per hypothesis: sign agreement, mean |pred|/|fit|, r2, n

    fc = forecast_experiment(llm, setting, {"control": "...", "t1": "..."}, "How much do you agree ...?",
                             scale=(1, 7), control="control", profiles=profile_df, n_per_condition=50)
    fc.effects                  # condition, effect, se (simulated), n
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ..llm.jsonx import json_extract

VARIABLE_KINDS = ("continuous", "count", "ordinal", "binary", "nominal")
NUMERIC_KINDS = ("continuous", "count")
PROTOCOLS = ("round_robin", "random")


# ===================================================================== decision log

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, np.ndarray, pd.Series)):
        return [_jsonable(v) for v in (o.tolist() if hasattr(o, "tolist") else o)]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict("records"))
    return o


class DecisionLog:
    """Append-only event log. `log(event, **payload)` keeps a record in memory and, with `path`, appends a JSON line."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload) -> dict:
        rec = {"seq": len(self.records), "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event,
               **_jsonable(payload)}
        self.records.append(rec)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def events(self, event: str) -> list[dict]:
        return [r for r in self.records if r["event"] == event]


def _log(log: DecisionLog | None) -> DecisionLog:
    return log if log is not None else DecisionLog()


# ===================================================================== hypothesis = linear SCM

@dataclass
class Variable:
    """One SCM variable. `levels` are the treatment values of a cause (all combinations are run); for the outcome they
    are optional. `proxy` is the second-person phrasing shown to the agent ("Your budget", or a template with
    "{value}"). `agent` = the agent who holds the attribute (None: a scenario-level fact everyone sees); `public` makes
    an individual attribute visible to the other agents. `question` (outcome only) is the survey question; `lo`/`hi`
    bound valid answers."""
    name: str
    kind: str = "continuous"
    levels: list = field(default_factory=list)
    units: str = ""
    proxy: str = ""
    agent: str | None = None
    public: bool = False
    question: str = ""
    lo: float | None = None
    hi: float | None = None

    def phrase(self, value) -> str:
        p = self.proxy or self.name.replace("_", " ")
        if "{value}" in p:
            return p.format(value=value)
        return f"{p}: {value}" + (f" {self.units}" if self.units else "")


@dataclass
class Hypothesis:
    """Linear SCM: causes -> outcome (plus optional two-way interactions)."""
    scenario: str
    agents: list[str]
    causes: list[Variable]
    outcome: Variable
    interactions: list[tuple[str, str]] = field(default_factory=list)
    rationale: str = ""
    id: str = ""

    def __post_init__(self):
        self.interactions = [tuple(p) for p in self.interactions]
        if not self.id:
            self.id = "h" + hashlib.sha256(json.dumps(self.to_dict(include_id=False), sort_keys=True,
                                                      default=str).encode()).hexdigest()[:8]

    def cause(self, name: str) -> Variable:
        return next(c for c in self.causes if c.name == name)

    def edges(self) -> list[str]:
        return [f"{c.name} -> {self.outcome.name}" for c in self.causes] + \
               [f"{a}:{b} -> {self.outcome.name}" for a, b in self.interactions]

    def to_dict(self, include_id: bool = True) -> dict:
        d = {"scenario": self.scenario, "agents": list(self.agents), "causes": [asdict(c) for c in self.causes],
             "outcome": asdict(self.outcome), "interactions": [list(p) for p in self.interactions],
             "rationale": self.rationale}
        if include_id:
            d["id"] = self.id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        return cls(scenario=d["scenario"], agents=list(d["agents"]), causes=[Variable(**c) for c in d["causes"]],
                   outcome=Variable(**d["outcome"]), interactions=[tuple(p) for p in d.get("interactions", [])],
                   rationale=d.get("rationale", ""), id=d.get("id", ""))


# ===================================================================== 1. proposal

HYPOTHESIS_SYSTEM = (
    "You are a social scientist designing an in-silico experiment with LLM agents. State each hypothesis as a linear "
    "structural causal model: one outcome measured by a survey question put to one agent, and exogenous causes that "
    "the experimenter can set. Operationalise every variable: kind (continuous, count, ordinal, binary, nominal), "
    "units, and a short list of treatment levels (all combinations will be run). Write `proxy` as the second-person "
    "phrase the agent will read (e.g. \"Your budget\"). Reply with JSON only.")

_HYP_FORMAT = (
    '{"hypotheses": [{"agents": ["buyer", "seller"], "rationale": "...",\n'
    '  "outcome": {"name": "deal_price", "kind": "continuous", "units": "dollars", "agent": "buyer",\n'
    '              "question": "What price did you agree on, in dollars?", "lo": 0, "hi": 100},\n'
    '  "causes": [{"name": "buyer_budget", "kind": "continuous", "units": "dollars", "levels": [5, 20, 40],\n'
    '              "proxy": "Your budget", "agent": "buyer", "public": false}],\n'
    '  "interactions": []}]}')


def hypothesis_prompt(scenario: str, n: int = 1, max_causes: int = 3, max_levels: int = 5) -> str:
    return (f"Scenario: {scenario}\n\nPropose {n} hypothesis(es). Each has 1 to {max_causes} causes with 2 to "
            f"{max_levels} levels each; levels of continuous/count causes are numbers. Format:\n{_HYP_FORMAT}")


# one number: thousands groups "1,250" / "1,250.5" first, then a plain or decimal-comma number ("3,5")
_NUM = r"-?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)"


def _to_float(s: str) -> float:
    return float(s.replace(",", "") if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s) else s.replace(",", "."))


def _numeric(v) -> float | None:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float, np.number)) and math.isfinite(float(v)):
        return float(v)
    if isinstance(v, str):
        m = re.fullmatch(rf"\s*\$?\s*({_NUM})\s*%?\s*", v)
        if m:
            return _to_float(m.group(1))
    return None


def _norm_name(s) -> str:
    return re.sub(r"\W+", "_", str(s).strip()).strip("_").lower()


def _clean_variable(d: dict, role: str, max_levels: int, problems: list) -> Variable | None:
    if not isinstance(d, dict) or not str(d.get("name", "")).strip():
        problems.append(f"{role}: missing name")
        return None
    name = _norm_name(d["name"])
    kind = str(d.get("kind", "continuous")).lower()
    if kind not in VARIABLE_KINDS:
        problems.append(f"{role} {name}: kind {kind!r} -> continuous")
        kind = "continuous"
    levels = list(d.get("levels") or [])
    if role == "cause" and kind in NUMERIC_KINDS:
        nums = [_numeric(v) for v in levels]
        bad = [v for v, x in zip(levels, nums) if x is None]
        if bad:
            problems.append(f"cause {name}: dropped non-numeric levels {bad}")
        levels = [x for x in nums if x is not None]
    dedup = []
    for v in levels:                    # after conversion, so 10 and "10" are one level
        if v not in dedup:
            dedup.append(v)
    levels = dedup
    if role == "cause":
        if len(levels) > max_levels:
            problems.append(f"cause {name}: {len(levels)} levels, kept the first {max_levels}")
            levels = levels[:max_levels]
        if kind == "binary" and len(levels) != 2:
            problems.append(f"cause {name}: binary with {len(levels)} levels -> " +
                            ("nominal" if len(levels) > 2 else "rejected"))
            kind = "nominal" if len(levels) > 2 else kind
        if len(levels) < 2:
            problems.append(f"cause {name}: fewer than 2 distinct levels, rejected")
            return None
    elif not str(d.get("question", "")).strip():
        problems.append(f"outcome {name}: no survey question, rejected")
        return None
    lo, hi = _numeric(d.get("lo")), _numeric(d.get("hi"))
    return Variable(name=name, kind=kind, levels=levels, units=str(d.get("units", "") or ""),
                    proxy=str(d.get("proxy", "") or ""), agent=d.get("agent") or None, public=bool(d.get("public")),
                    question=str(d.get("question", "") or ""), lo=lo, hi=hi)


def parse_hypotheses(obj: Any, scenario: str, max_causes: int = 3, max_levels: int = 5
                     ) -> tuple[list[Hypothesis], list[str]]:
    """Validate an LLM reply (dict / list / JSON text). Returns (hypotheses, problems); every repair or rejection is
    one line in `problems`."""
    problems: list[str] = []
    if isinstance(obj, str):
        obj = json_extract(obj, default=None)
    items = obj.get("hypotheses", [obj]) if isinstance(obj, dict) else obj if isinstance(obj, list) else []
    out = []
    for i, h in enumerate(items):
        p: list[str] = []
        if not isinstance(h, dict):
            problems.append(f"hypothesis {i}: not an object")
            continue
        outcome = _clean_variable(h.get("outcome"), "outcome", max_levels, p)
        causes, seen = [], set()
        for c in h.get("causes") or []:
            v = _clean_variable(c, "cause", max_levels, p)
            if v is None:
                continue
            if v.name in seen or (outcome and v.name == outcome.name):
                p.append(f"cause {v.name}: duplicate name, rejected")
                continue
            seen.add(v.name)
            causes.append(v)
        if len(causes) > max_causes:
            p.append(f"{len(causes)} causes, kept the first {max_causes}")
            causes = causes[:max_causes]
        agents = [str(a) for a in (h.get("agents") or [])] or ["participant"]
        for v in causes + ([outcome] if outcome else []):
            if v.agent is not None and v.agent not in agents:
                p.append(f"{v.name}: agent {v.agent!r} not in agents, made scenario-level")
                v.agent = None
        if outcome is not None and outcome.agent is None:
            outcome.agent = agents[0]
        kept = [c.name for c in causes]          # after the max_causes cut, so no interaction names a dropped cause
        inter = []
        for pair in h.get("interactions") or []:
            ab = [_norm_name(x) for x in pair] if isinstance(pair, (list, tuple)) and len(pair) == 2 else []
            if len(ab) == 2 and all(x in kept for x in ab) and ab[0] != ab[1]:
                if (ab[0], ab[1]) in inter or (ab[1], ab[0]) in inter:
                    p.append(f"interaction {pair}: duplicate, dropped")
                else:
                    inter.append((ab[0], ab[1]))
            else:
                p.append(f"interaction {pair}: unknown causes, dropped")
        problems += [f"hypothesis {i}: {s}" for s in p]
        if outcome is None or not causes:
            problems.append(f"hypothesis {i}: rejected (no valid outcome or causes)")
            continue
        out.append(Hypothesis(scenario=scenario, agents=agents, causes=causes, outcome=outcome, interactions=inter,
                              rationale=str(h.get("rationale", ""))))
    return out, problems


def propose_hypotheses(llm, scenario: str, n: int = 1, *, max_causes: int = 3, max_levels: int = 5,
                       attempts: int = 2, max_tokens: int = 1500, log: DecisionLog | None = None,
                       **kw) -> list[Hypothesis]:
    """Ask the LLM for `n` hypotheses. A reply with no valid hypothesis is re-asked once with the problems listed."""
    log = _log(log)
    user, hyps = hypothesis_prompt(scenario, n, max_causes, max_levels), []
    for k in range(max(1, attempts)):
        r = llm.complete(HYPOTHESIS_SYSTEM, user, max_tokens=max_tokens, **kw)
        hyps, problems = parse_hypotheses(r.text, scenario, max_causes, max_levels)
        log.log("propose", attempt=k, raw=r.text, cost_usd=r.cost_usd, accepted=[h.id for h in hyps],
                problems=problems)
        if hyps:
            break
        user = (hypothesis_prompt(scenario, n, max_causes, max_levels) + "\n\nYour previous reply was rejected: "
                + "; ".join(problems or ["no JSON"]) + "\nReturn only valid JSON.")
    for h in hyps[:n]:
        log.log("hypothesis", **h.to_dict())
    return hyps[:n]


# ===================================================================== 2. design and encoding

def design(hyp: Hypothesis, replicates: int = 1, *, max_cells: int | None = None, seed: int = 0,
           log: DecisionLog | None = None) -> pd.DataFrame:
    """Full factorial over the causes' levels x `replicates`. `max_cells` caps the number of cells (seeded subsample,
    logged). Columns: cell, rep, one column per cause."""
    names = [c.name for c in hyp.causes]
    cells = list(itertools.product(*[c.levels for c in hyp.causes]))
    full = len(cells)
    if max_cells is not None and full > max_cells:
        idx = np.sort(np.random.default_rng(seed).choice(full, max_cells, replace=False))
        cells = [cells[i] for i in idx]
    df = pd.DataFrame(cells, columns=names)
    df.insert(0, "cell", np.arange(len(df)))
    df = df.loc[df.index.repeat(replicates)].reset_index(drop=True)
    df.insert(1, "rep", np.tile(np.arange(replicates), len(cells)))
    _log(log).log("design", hypothesis=hyp.id, factorial_cells=full, cells=len(cells), replicates=replicates,
                  rows=len(df), subsampled=len(cells) < full, seed=seed)
    return df


def _code(c: Variable) -> dict | None:
    """Numeric code per level for numeric/binary/ordinal causes; None for nominal (dummy coded)."""
    nums = [_numeric(v) for v in c.levels]
    if c.kind in NUMERIC_KINDS or (c.kind in ("ordinal", "binary") and all(x is not None for x in nums)):
        return {v: x for v, x in zip(c.levels, nums)}
    if c.kind in ("ordinal", "binary"):
        return {v: float(i) for i, v in enumerate(c.levels)}     # declared order; binary: first level = 0
    return None


def encode(hyp: Hypothesis, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Design matrix (no intercept) and {term: cause}. Nominal causes are treatment coded against their first level
    (term `name[level]`); interactions are products of the coded columns (`a:b`)."""
    X, owner, cols = {}, {}, {}
    for c in hyp.causes:
        code = _code(c)
        if code is None:
            cols[c.name] = []
            for lev in c.levels[1:]:
                t = f"{c.name}[{lev}]"
                X[t] = (df[c.name] == lev).astype(float).to_numpy()
                owner[t] = c.name
                cols[c.name].append(t)
        else:
            X[c.name] = df[c.name].map(code).astype(float).to_numpy()
            owner[c.name] = c.name
            cols[c.name] = [c.name]
    for a, b in hyp.interactions:
        for ta in cols[a]:
            for tb in cols[b]:
                t = f"{ta}:{tb}"
                X[t] = X[ta] * X[tb]
                owner[t] = f"{a}:{b}"
    return pd.DataFrame(X, index=df.index), owner


# ===================================================================== 3. estimation

def _have_statsmodels() -> bool:
    try:
        import statsmodels.api  # noqa: F401
        return True
    except ImportError:
        return False


def estimate(hyp: Hypothesis, data: pd.DataFrame, outcome_col: str = "y", *, backend: str = "auto",
             alpha: float = 0.05, log: DecisionLog | None = None) -> tuple[pd.DataFrame, dict]:
    """OLS of the outcome on the encoded causes (+ intercept), classical standard errors.

    Returns (paths, info). paths: term, cause, coef, se, t, p, lo, hi, beta_std (coef * sd(x) / sd(y)), n.
    Rows with a missing outcome are dropped and counted; constant design columns are dropped and logged.
    backend: "auto" (statsmodels if installed, else numpy), "statsmodels", "numpy"."""
    from scipy import stats as st
    log = _log(log)
    X, owner = encode(hyp, data)
    y = pd.to_numeric(data[outcome_col], errors="coerce").to_numpy(float)
    keep = np.isfinite(y)
    X, y = X[keep], y[keep]
    const = [t for t in X.columns if X[t].nunique() < 2]
    X = X.drop(columns=const)
    use = backend if backend != "auto" else ("statsmodels" if _have_statsmodels() else "numpy")
    n, k = len(y), X.shape[1] + 1
    if n <= k:
        raise ValueError(f"estimate: {n} usable rows for {k} parameters")
    A = np.column_stack([np.ones(n), X.to_numpy(float)])
    if np.linalg.matrix_rank(A) < k:
        raise ValueError("estimate: design matrix is rank deficient (collinear causes)")
    if use == "statsmodels":
        import statsmodels.api as sm
        res = sm.OLS(y, A).fit()
        coef, se, r2 = np.asarray(res.params), np.asarray(res.bse), float(res.rsquared)
    elif use == "numpy":
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        s2 = resid @ resid / (n - k)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
        tss = ((y - y.mean()) ** 2).sum()
        r2 = float(1 - resid @ resid / tss) if tss > 0 else float("nan")
    else:
        raise ValueError(f"unknown backend {backend!r}")
    with np.errstate(divide="ignore", invalid="ignore"):
        t = coef / se
    p = 2 * st.t.sf(np.abs(t), n - k)
    q = st.t.ppf(1 - alpha / 2, n - k)
    sdy = y.std(ddof=1)
    terms = ["(intercept)"] + list(X.columns)
    sdx = np.r_[np.nan, X.std(ddof=1).to_numpy()]
    paths = pd.DataFrame({"term": terms, "cause": ["(intercept)"] + [owner[c] for c in X.columns], "coef": coef,
                          "se": se, "t": t, "p": p, "lo": coef - q * se, "hi": coef + q * se,
                          "beta_std": coef * sdx / sdy if sdy > 0 else np.nan, "n": n})
    info = {"backend": use, "n": n, "dropped_missing": int((~keep).sum()), "dropped_constant": const, "r2": r2,
            "alpha": alpha}
    log.log("fit", hypothesis=hyp.id, **info)
    return paths, info


# ===================================================================== 4. simulators

def _agent_facts(hyp: Hypothesis, row: dict, agent: str) -> list[str]:
    """Attributes the agent sees: scenario-level causes, its own causes, other agents' public causes."""
    out = []
    for c in hyp.causes:
        if c.agent is None or c.agent == agent:
            out.append(c.phrase(row[c.name]))
        elif c.public:
            out.append(f"The {c.agent}'s {c.name.replace('_', ' ')}: {row[c.name]}" + (f" {c.units}" if c.units else ""))
    return out


def _outcome_agent(hyp: Hypothesis) -> str:
    """The agent who answers the survey (parse_hypotheses sets it; hand-built hypotheses fall back to agents[0])."""
    return hyp.outcome.agent or (hyp.agents[0] if hyp.agents else "participant")


def _persona(hyp: Hypothesis, row: dict, agent: str) -> str:
    facts = _agent_facts(hyp, row, agent)
    return (f"You are the {agent} in this scenario: {hyp.scenario}\n" +
            ("What you know:\n" + "\n".join(f"- {f}" for f in facts) if facts else ""))


ANSWER_FORMAT = 'Reply with JSON only: {"answer": <number>}.'


def parse_number(text: str | None, lo: float | None = None, hi: float | None = None) -> float:
    """{"answer": x} or the first number in the text; NaN if none or outside [lo, hi]. A JSON reply whose answer
    key holds no number (null, "unsure") is NaN, not the first number elsewhere in the text. "1,250" reads as
    1250 and "3,5" as 3.5, in JSON values and free text alike."""
    obj = json_extract(text, default=None) if text else None
    v, keyed = None, False
    if isinstance(obj, dict):
        for key in ("answer", "value", "prediction", "estimate"):
            if key in obj:
                v, keyed = _numeric(obj[key]), True
                break
    if v is None and text and not keyed:
        m = re.search(_NUM, str(text))
        v = _to_float(m.group(0)) if m else None
    if v is None or (lo is not None and v < lo) or (hi is not None and v > hi):
        return float("nan")
    return float(v)


def _survey_user(hyp: Hypothesis, transcript: str = "") -> str:
    o = hyp.outcome
    rng = f" (a number from {o.lo:g} to {o.hi:g})" if o.lo is not None and o.hi is not None else ""
    return ((f"Conversation so far:\n{transcript}\n\n" if transcript else "") +
            f"Survey question: {o.question}{rng}\n{ANSWER_FORMAT}")


class InterviewSimulator:
    """One LLM call per design row: the outcome agent reads its attributes and answers the survey question
    (no conversation). Cheapest LLM-subject simulator; replicates differ by a "participant k" line so cached
    backends do not return the same reply."""
    name = "interview"

    def __init__(self, llm, *, max_tokens: int = 200, parallel: int = 4, **kw):
        self.llm, self.max_tokens, self.parallel, self.kw = llm, max_tokens, parallel, kw

    def __call__(self, hyp: Hypothesis, df: pd.DataFrame, log: DecisionLog | None = None) -> np.ndarray:
        log = _log(log)
        rows = df.to_dict("records")
        prompts = [(_persona(hyp, r, _outcome_agent(hyp)) + f"\n(Participant {int(r.get('rep', 0)) + 1}.)",
                    _survey_user(hyp)) for r in rows]
        resp = self.llm.map(prompts, max_tokens=self.max_tokens, parallel=self.parallel, **self.kw)
        y = np.array([parse_number(r.text, hyp.outcome.lo, hyp.outcome.hi) for r in resp])
        log.log("simulate", hypothesis=hyp.id, simulator=self.name, rows=len(rows), calls=len(resp),
                unparsed=int(np.isnan(y).sum()), cost_usd=sum(r.cost_usd for r in resp))
        return y


JUDGE_SYSTEM = ("You monitor a conversation between agents in a social science simulation. Decide whether it has "
                "reached its natural end (agreement, decision, or a clear breakdown). Answer only yes or no.")


class ConversationSimulator:
    """Multi-agent LLM conversation per design row, as in Manning et al. (2024): agents speak in turn
    (`protocol` "round_robin" = the paper's "ordered", or seeded "random" = random first speaker, nobody twice in a
    row), after every statement an external LLM judge (the paper's coordinator) says whether the conversation is over
    (`judge_from` = number of statements before it is first asked; paper: 1, after each agent speaks), it stops at
    `max_statements` (paper: 20); then the outcome agent answers the survey question with the transcript in view.
    All rows advance in lockstep, one `llm.map` per step."""
    name = "conversation"

    def __init__(self, llm, *, judge_llm=None, protocol: str = "round_robin", max_statements: int = 20,
                 judge_from: int = 1, max_tokens: int = 200, parallel: int = 4, seed: int = 0,
                 keep_transcripts: bool = True, **kw):
        if protocol not in PROTOCOLS:
            raise ValueError(f"protocol must be one of {PROTOCOLS}")
        self.llm, self.judge = llm, judge_llm or llm
        self.protocol, self.max_statements, self.max_tokens = protocol, max_statements, max_tokens
        self.judge_from = max(1, int(judge_from))
        self.parallel, self.seed, self.keep, self.kw = parallel, seed, keep_transcripts, kw
        self.transcripts: list[list[tuple[str, str]]] = []

    def _speaker(self, agents, step, rng, last):
        if self.protocol == "round_robin":
            return agents[step % len(agents)]
        pool = [a for a in agents if a != last] or agents
        return pool[int(rng.integers(len(pool)))]

    def __call__(self, hyp: Hypothesis, df: pd.DataFrame, log: DecisionLog | None = None) -> np.ndarray:
        log = _log(log)
        rows = df.to_dict("records")
        n, agents = len(rows), hyp.agents
        rngs = [np.random.default_rng([self.seed, i]) for i in range(n)]
        tr: list[list[tuple[str, str]]] = [[] for _ in range(n)]
        active, last, calls, ended_by = list(range(n)), [None] * n, 0, ["max_statements"] * n
        txt = lambda i: "\n".join(f"{a}: {s}" for a, s in tr[i])
        for step in range(self.max_statements):
            if not active:
                break
            who = {i: self._speaker(agents, step, rngs[i], last[i]) for i in active}
            prompts = [(_persona(hyp, rows[i], who[i]) + "\nSpeak in one or two sentences, in character.",
                        (f"Conversation so far:\n{txt(i)}\n\n" if tr[i] else "The conversation starts now.\n\n")
                        + f"Your turn, {who[i]}:") for i in active]
            resp = self.llm.map(prompts, max_tokens=self.max_tokens, parallel=self.parallel, **self.kw)
            calls += len(resp)
            for i, r in zip(active, resp):
                tr[i].append((who[i], (r.text or "").strip()))
                last[i] = who[i]
            if step + 1 >= self.judge_from:
                jr = self.judge.map([(JUDGE_SYSTEM, f"{hyp.scenario}\n\n{txt(i)}\n\nIs the conversation over?")
                                     for i in active], max_tokens=5, parallel=self.parallel)
                calls += len(jr)
                done = {i for i, r in zip(active, jr) if (r.text or "").strip().lower().startswith("yes")}
                for i in done:
                    ended_by[i] = "judge"
                active = [i for i in active if i not in done]
        resp = self.llm.map([(_persona(hyp, rows[i], _outcome_agent(hyp)), _survey_user(hyp, txt(i)))
                             for i in range(n)], max_tokens=self.max_tokens, parallel=self.parallel, **self.kw)
        calls += len(resp)
        y = np.array([parse_number(r.text, hyp.outcome.lo, hyp.outcome.hi) for r in resp])
        if self.keep:
            self.transcripts = tr
        lens = [len(t) for t in tr]
        log.log("simulate", hypothesis=hyp.id, simulator=self.name, protocol=self.protocol, judge_from=self.judge_from, rows=n, calls=calls,
                ended_by_judge=ended_by.count("judge"), mean_statements=float(np.mean(lens)) if lens else 0.0,
                unparsed=int(np.isnan(y).sum()))
        return y


class StructuralSimulator:
    """Wraps a vectorised world `fn(df) -> (n,) outcomes` (numpy ABM, ground-truth SCM, ...)."""
    name = "structural"

    def __init__(self, fn: Callable[[pd.DataFrame], np.ndarray], name: str | None = None):
        self.fn = fn
        if name:
            self.name = name

    def __call__(self, hyp: Hypothesis, df: pd.DataFrame, log: DecisionLog | None = None) -> np.ndarray:
        y = np.asarray(self.fn(df), float).reshape(-1)
        if len(y) != len(df):
            raise ValueError(f"{self.name}: returned {len(y)} outcomes for {len(df)} rows")
        _log(log).log("simulate", hypothesis=hyp.id, simulator=self.name, rows=len(df), calls=0,
                      unparsed=int(np.isnan(y).sum()))
        return y


def linear_world(coefs: dict[str, float], intercept: float = 0.0, noise: float = 1.0, seed: int = 0,
                 levels: dict[str, dict] | None = None) -> StructuralSimulator:
    """Ground-truth linear SCM y = intercept + sum coef * x + N(0, noise^2). `levels` maps a nominal cause to
    {level: effect}. For coefficient-recovery checks of the whole pipeline."""
    def fn(df):
        rng = np.random.default_rng(seed)
        y = np.full(len(df), float(intercept))
        for k, b in coefs.items():
            y += b * pd.to_numeric(df[k]).to_numpy(float)
        for k, m in (levels or {}).items():
            y += df[k].map(m).fillna(0.0).to_numpy(float)
        return y + noise * rng.standard_normal(len(df))
    return StructuralSimulator(fn, "linear_world")


def market_world(*, price: str = "price_cut", promo: str | None = "promo", theta: Sequence[float] | None = None,
                 weeks: int = 4, n_agents: int = 400, seed: int = 0, cohort: bool = True) -> StructuralSimulator:
    """Outcome = mean weekly sales of the `simengine.abm.market_sim` repeat-purchase market during an event of
    `weeks` weeks after a regular-price warm-up. Cause `price` = fractional price cut (0.2 = 20% off), `promo` =
    promotion intensity 0-1. `theta` = market parameters (MARKET_PARAMS order; default the prior midpoint)."""
    from ..abm.engine import MARKET_PRIOR, market_sim
    th = np.asarray(theta if theta is not None else [(lo + hi) / 2 for lo, hi in MARKET_PRIOR.values()], float)

    def fn(df):
        B = len(df)
        cut = pd.to_numeric(df[price]).to_numpy(float) if price in df else np.zeros(B)
        pr = pd.to_numeric(df[promo]).to_numpy(float) if promo and promo in df else np.zeros(B)
        lp = np.zeros((B, weeks + 1))
        lp[:, 1:] = np.log(np.clip(1 - cut, 1e-3, None))[:, None]
        pm = np.zeros((B, weeks + 1))
        pm[:, 1:] = pr[:, None]
        sales = market_sim(np.repeat(th[None], B, 0), lp, pm, n_agents=n_agents, seed=seed, cohort=cohort)
        return sales[:, 1:].mean(1)
    return StructuralSimulator(fn, "market_world")


def make_simulator(sim, llm=None, **kw):
    if callable(sim) and not isinstance(sim, str):
        return sim
    if sim == "interview":
        return InterviewSimulator(llm, **kw)
    if sim == "conversation":
        return ConversationSimulator(llm, **kw)
    raise ValueError(f"unknown simulator {sim!r} (interview, conversation, or a callable)")


# ===================================================================== 5. LLM forecasts of the paths

PATH_SYSTEM = ("You are an expert social scientist. Your task is to predict the point estimates for the paths in "
               "the SCM as accurately as possible based on the experiment described. Reply with JSON only.")


def _describe(hyp: Hypothesis, terms: Sequence[str], n_rows: int, all_cells: bool = True) -> str:
    lines = [f"Scenario: {hyp.scenario}", f"Agents: {', '.join(hyp.agents)}",
             f"Outcome: {hyp.outcome.name} ({hyp.outcome.kind}, {hyp.outcome.units or 'no units'}), measured by "
             f"asking the {_outcome_agent(hyp)}: \"{hyp.outcome.question}\"", "Causes (treatment values):"]
    for c in hyp.causes:
        coding = "dummy coded vs " + repr(c.levels[0]) if _code(c) is None else "numeric value"
        lines.append(f"- {c.name} ({c.kind}, {c.units or 'no units'}, {coding}), held by "
                     f"{c.agent or 'the scenario'}: {c.levels}")
    lines.append(f"Every combination of treatment values was simulated; {n_rows} simulations in total." if all_cells
                 else f"A random subset of the combinations of treatment values was simulated; {n_rows} simulations "
                      "in total.")
    lines.append("The SCM is a linear regression of the outcome on the causes with an intercept. Paths "
                 "(unstandardised coefficients, outcome units per cause unit): " + ", ".join(terms))
    return "\n".join(lines)


def path_prediction_prompt(hyp: Hypothesis, terms: Sequence[str], n_rows: int, all_cells: bool = True) -> str:
    fmt = json.dumps({"predictions": {t: 0.0 for t in terms}, "sig": {t: True for t in terms}})
    return (_describe(hyp, terms, n_rows, all_cells) + "\n\nPredict each path estimate and whether it is significant at "
            f"the 5% level. Format: {fmt}")


def predict_paths(llm, hyp: Hypothesis, terms: Sequence[str], n_rows: int, *, n_samples: int = 1,
                  all_cells: bool = True, max_tokens: int = 600, log: DecisionLog | None = None,
                  **kw) -> pd.DataFrame:
    """LLM forecast of each path (paper Fig. A.11 task). n_samples > 1 asks the same question with a sample number
    appended (distinct cache keys; pass temperature via **kw) and averages. Returns term, predicted, pred_sd,
    sig_share, n_valid; the forecast's sha256 is logged before any fit. all_cells=False tells the forecaster the
    design is a subsample of the factorial (max_cells cut)."""
    log = _log(log)
    base = path_prediction_prompt(hyp, terms, n_rows, all_cells)
    users = [base if n_samples == 1 else f"{base}\n(forecast {k + 1} of {n_samples})" for k in range(n_samples)]
    resp = llm.map([(PATH_SYSTEM, u) for u in users], max_tokens=max_tokens, **kw)
    vals = {t: [] for t in terms}
    sig = {t: [] for t in terms}
    for r in resp:
        obj = json_extract(r.text, default=None)
        if not isinstance(obj, dict):
            continue
        pr, sg = obj.get("predictions") or {}, obj.get("sig") or {}
        for t in terms:
            v = _numeric(pr.get(t)) if isinstance(pr, dict) else None
            if v is not None:
                vals[t].append(v)
            if isinstance(sg, dict) and t in sg:
                sig[t].append(bool(sg[t]) if not isinstance(sg[t], str) else sg[t].lower() in ("true", "yes"))
    out = pd.DataFrame({"term": list(terms),
                        "predicted": [float(np.mean(vals[t])) if vals[t] else np.nan for t in terms],
                        "pred_sd": [float(np.std(vals[t], ddof=1)) if len(vals[t]) > 1 else np.nan for t in terms],
                        "sig_share": [float(np.mean(sig[t])) if sig[t] else np.nan for t in terms],
                        "n_valid": [len(vals[t]) for t in terms]})
    digest = hashlib.sha256(out.to_json(orient="records", double_precision=10).encode()).hexdigest()
    log.log("predict_paths", hypothesis=hyp.id, samples=n_samples, sha256=digest, raw=[r.text for r in resp],
            predictions=out, cost_usd=sum(r.cost_usd for r in resp))
    return out


def compare_paths(fitted: pd.DataFrame, predicted: pd.DataFrame, alpha: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """Join fitted and predicted paths (intercept excluded). ratio = |predicted| / |coef|; sign_match; sig_match
    (fitted p < alpha vs the majority of the LLM's sig votes). Summary: n_paths, sign_agreement, sig_agreement,
    mean_ratio, median_ratio, overestimate_share, r (Pearson, >= 3 paths)."""
    f = fitted[fitted.term != "(intercept)"]
    m = f.merge(predicted, on="term", how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        m["ratio"] = np.abs(m.predicted) / np.abs(m.coef)
    m["sign_match"] = np.where(m.predicted.notna(), np.sign(m.predicted) == np.sign(m.coef), np.nan)
    m["fit_sig"] = m.p < alpha
    m["sig_match"] = np.where(m.sig_share.notna(), (m.sig_share >= 0.5) == m.fit_sig, np.nan)
    ok = m.predicted.notna()
    rat = m.ratio[ok & np.isfinite(m.ratio)]
    summ = {"n_paths": int(len(m)), "n_predicted": int(ok.sum()),
            "sign_agreement": float(m.sign_match[ok].mean()) if ok.any() else None,
            "sig_agreement": float(pd.to_numeric(m.sig_match, errors="coerce").mean())
            if m.sig_share.notna().any() else None,
            "mean_ratio": float(rat.mean()) if len(rat) else None,
            "median_ratio": float(rat.median()) if len(rat) else None,
            "overestimate_share": float((rat > 1).mean()) if len(rat) else None,
            "r": float(np.corrcoef(m.coef[ok], m.predicted[ok])[0, 1]) if ok.sum() >= 3 else None}
    return m, summ


def predict_outcomes(llm, hyp: Hypothesis, cells: pd.DataFrame, *, fitted: pd.DataFrame | None = None,
                     max_tokens: int = 100, log: DecisionLog | None = None, **kw) -> np.ndarray:
    """Per-cell outcome forecast ("predict y_i"); with `fitted` the fitted path estimates are shown first
    ("predict y_i | beta-hat", the paper's auction comparison). Returns one prediction per row of `cells`."""
    names = [c.name for c in hyp.causes]
    head = f"Scenario: {hyp.scenario}\nOutcome: {hyp.outcome.question}"
    if fitted is not None:
        head += "\nFitted linear SCM from earlier simulations: " + ", ".join(
            f"{t} = {c:.4g}" for t, c in zip(fitted.term, fitted.coef))
    users = [head + "\nTreatment: " + ", ".join(f"{k} = {r[k]}" for k in names) +
             "\nPredict the average outcome. " + ANSWER_FORMAT for r in cells.to_dict("records")]
    resp = llm.map([(PATH_SYSTEM.split(" Your task")[0], u) for u in users], max_tokens=max_tokens, **kw)
    y = np.array([parse_number(r.text) for r in resp])
    _log(log).log("predict_outcomes", hypothesis=hyp.id, conditioned=fitted is not None, rows=len(users),
                  unparsed=int(np.isnan(y).sum()))
    return y


# ===================================================================== orchestration

@dataclass
class HypothesisRun:
    hypothesis: Hypothesis
    data: pd.DataFrame              # design + outcome column "y"
    paths: pd.DataFrame             # fitted
    fit_info: dict
    predicted: pd.DataFrame
    comparison: pd.DataFrame
    summary: dict


@dataclass
class AutoSocialScienceResult:
    runs: list[HypothesisRun]
    log: DecisionLog

    @property
    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{"hypothesis": r.hypothesis.id, "edges": "; ".join(r.hypothesis.edges()),
                              "n": r.fit_info["n"], "r2": r.fit_info["r2"], **r.summary} for r in self.runs])

    @property
    def table(self) -> pd.DataFrame:
        return pd.concat([r.comparison.assign(hypothesis=r.hypothesis.id) for r in self.runs], ignore_index=True) \
            if self.runs else pd.DataFrame()


class AutoSocialScience:
    """propose -> design -> forecast paths -> simulate -> fit -> compare, per hypothesis.

    llm: proposer (and default subject / forecaster). simulator: "interview", "conversation", or any
    `callable(hyp, df, log) -> outcomes` (StructuralSimulator, linear_world, market_world). sim_llm /
    predictor_llm: separate backends for subjects and forecaster (e.g. a cheaper model for the subjects).
    The forecast is made and hashed BEFORE the simulation, so it cannot see the data."""

    def __init__(self, llm, scenario: str, *, simulator: Any = "interview", sim_llm=None, predictor_llm=None,
                 replicates: int = 1, max_cells: int | None = 200, max_causes: int = 3, max_levels: int = 5,
                 n_forecasts: int = 1, backend: str = "auto", alpha: float = 0.05, seed: int = 0,
                 workdir: str | Path | None = None, simulator_kw: dict | None = None):
        self.llm, self.scenario = llm, scenario
        self.sim_llm, self.pred_llm = sim_llm or llm, predictor_llm or llm
        self.simulator = make_simulator(simulator, self.sim_llm, **(simulator_kw or {}))
        self.replicates, self.max_cells, self.max_causes, self.max_levels = replicates, max_cells, max_causes, max_levels
        self.n_forecasts, self.backend, self.alpha, self.seed = n_forecasts, backend, alpha, seed
        self.workdir = Path(workdir) if workdir else None
        self.log = DecisionLog(self.workdir / "decisions.jsonl" if self.workdir else None)

    def run_one(self, hyp: Hypothesis) -> HypothesisRun:
        df = design(hyp, self.replicates, max_cells=self.max_cells, seed=self.seed, log=self.log)
        X, _ = encode(hyp, df)
        terms = [t for t in X.columns if X[t].nunique() > 1]
        full = math.prod(len(c.levels) for c in hyp.causes)
        pred = predict_paths(self.pred_llm, hyp, terms, len(df), n_samples=self.n_forecasts,
                             all_cells=df["cell"].nunique() == full, log=self.log)
        df["y"] = self.simulator(hyp, df, self.log)
        paths, info = estimate(hyp, df, "y", backend=self.backend, alpha=self.alpha, log=self.log)
        comp, summ = compare_paths(paths, pred, self.alpha)
        self.log.log("compare", hypothesis=hyp.id, **summ)
        if self.workdir:
            df.to_csv(self.workdir / f"data_{hyp.id}.csv", index=False)
        return HypothesisRun(hyp, df, paths, info, pred, comp, summ)

    def run(self, n_hypotheses: int = 1, hypotheses: Sequence[Hypothesis] | None = None) -> AutoSocialScienceResult:
        self.log.log("start", scenario=self.scenario, simulator=getattr(self.simulator, "name", repr(self.simulator)),
                     replicates=self.replicates, max_cells=self.max_cells, backend=self.backend, seed=self.seed,
                     given_hypotheses=hypotheses is not None)
        hyps = list(hypotheses) if hypotheses is not None else propose_hypotheses(
            self.llm, self.scenario, n_hypotheses, max_causes=self.max_causes, max_levels=self.max_levels,
            log=self.log)
        runs = []
        for h in hyps:
            try:
                runs.append(self.run_one(h))
            except ValueError as e:              # rank-deficient design, too few parsed answers, ...
                self.log.log("failed", hypothesis=h.id, error=str(e))
        res = AutoSocialScienceResult(runs, self.log)
        if self.workdir and runs:
            res.table.to_csv(self.workdir / "paths.csv", index=False)
            (self.workdir / "summary.json").write_text(json.dumps(_jsonable(res.summary.to_dict("records")),
                                                                  indent=2, ensure_ascii=False))
        self.log.log("done", runs=len(runs), failed=len(hyps) - len(runs))
        return res


# ===================================================================== Hewitt et al.: experiment outcome predictor

PROMPT_BANK = (
    "You will be asked to predict how people respond to various messages. {setting}",
    "A researcher wants to anticipate how a survey participant answers after reading some material. {setting}",
    "Imagine the person described below took part in a survey study. {setting}",
    "You are simulating a participant in an online survey experiment. {setting}",
)
DIRECT_SYSTEM = ("You are an expert in survey experiments. Predict the results of the experiment described. "
                 "Reply with JSON only.")


def profile_text(p: dict) -> str:
    return "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in p.items() if v is not None and v == v)


@dataclass
class ExperimentForecast:
    mode: str
    conditions: pd.DataFrame        # condition, mean, sd, n, n_invalid
    effects: pd.DataFrame           # condition, effect, se, n  (treatment mean - control mean)
    calls: int


def forecast_experiment(llm, setting: str, conditions: dict[str, str], question: str, *, scale: tuple = (1, 7),
                        labels: tuple[str, str] | None = None, control: str, profiles: Any = None,
                        n_per_condition: int = 30, mode: str = "simulate", prompt_bank: Sequence[str] = PROMPT_BANK,
                        same_profiles: bool = True, n_samples: int = 5, seed: int = 0, max_tokens: int = 100,
                        parallel: int = 4, log: DecisionLog | None = None, **kw) -> ExperimentForecast:
    """Forecast condition means and treatment effects of a survey experiment.

    mode="simulate" (Hewitt et al. 2024): per condition, `n_per_condition` participants drawn from `profiles`
    (DataFrame / list of dicts; demographic fields), each prompt = a template from `prompt_bank` + profile + stimulus
    + question and scale; answers averaged per condition. `same_profiles` reuses one draw across conditions (paired,
    less noise; the paper drew independently). mode="direct": ask the LLM `n_samples` times for every condition's
    mean and average. Effects = condition mean - `control` mean. Simulated se: with `same_profiles` the paired se,
    sd(y_t - y_c) / sqrt(pairs) over respondents valid in both conditions (respondent j has the same profile and
    template in every condition); otherwise sqrt(sd_t^2/n_t + sd_c^2/n_c). Direct mode: se = NaN."""
    log = _log(log)
    if control not in conditions:
        raise ValueError(f"control {control!r} not in conditions")
    lo, hi = float(scale[0]), float(scale[1])
    lab = f" ({labels[0]} = {lo:g}, {labels[1]} = {hi:g})" if labels else ""
    rng = np.random.default_rng(seed)
    names = list(conditions)
    rows, ys = [], {}
    if mode == "simulate":
        prof = (profiles.to_dict("records") if isinstance(profiles, pd.DataFrame) else list(profiles or [{}]))
        draw = lambda: rng.integers(0, len(prof), n_per_condition)
        tpl = lambda: rng.integers(0, len(prompt_bank), n_per_condition)
        shared = (draw(), tpl()) if same_profiles else None
        prompts, owner = [], []
        for c in names:
            pi, ti = shared if shared else (draw(), tpl())
            for j, (a, b) in enumerate(zip(pi, ti)):
                sys_ = prompt_bank[b].format(setting=setting).strip()
                user = (f"Participant: {profile_text(prof[a]) or 'an adult survey respondent'}\n\n"
                        f"The participant read:\n\"{conditions[c]}\"\n\nQuestion: {question}\n"
                        f"Scale: {lo:g} to {hi:g}{lab}.\n(Respondent {j + 1}.) How would this participant answer? "
                        + ANSWER_FORMAT)
                prompts.append((sys_, user))
                owner.append(c)
        resp = llm.map(prompts, max_tokens=max_tokens, parallel=parallel, **kw)
        y = np.array([parse_number(r.text, lo, hi) for r in resp])
        own = np.array(owner)
        for c in names:
            v = ys[c] = y[own == c]
            ok = v[np.isfinite(v)]
            rows.append({"condition": c, "mean": float(ok.mean()) if len(ok) else np.nan,
                         "sd": float(ok.std(ddof=1)) if len(ok) > 1 else np.nan, "n": int(len(ok)),
                         "n_invalid": int(len(v) - len(ok))})
        calls = len(resp)
        log.log("forecast_experiment", mode=mode, conditions=names, control=control, n_per_condition=n_per_condition,
                same_profiles=same_profiles, profiles=len(prof), templates=len(prompt_bank),
                invalid=int(np.isnan(y).sum()), seed=seed)
    elif mode == "direct":
        desc = "\n".join(f"- {c}{' (control)' if c == control else ''}: \"{conditions[c]}\"" for c in names)
        fmt = json.dumps({"means": {c: 0.0 for c in names}})
        base = (f"{setting}\nParticipants were randomly assigned to read one of these:\n{desc}\n\nThen they "
                f"answered: {question} on a scale from {lo:g} to {hi:g}{lab}.\nPredict the average answer in each "
                f"condition. Format: {fmt}")
        resp = llm.map([(DIRECT_SYSTEM, base if n_samples == 1 else f"{base}\n(forecast {k + 1} of {n_samples})")
                        for k in range(n_samples)], max_tokens=max_tokens * 4, parallel=parallel, **kw)
        vals = {c: [] for c in names}
        for r in resp:
            obj = json_extract(r.text, default=None)
            m = obj.get("means", obj) if isinstance(obj, dict) else {}
            for c in names:
                v = _numeric(m.get(c)) if isinstance(m, dict) else None
                if v is not None and lo <= v <= hi:
                    vals[c].append(v)
        for c in names:
            v = np.array(vals[c])
            rows.append({"condition": c, "mean": float(v.mean()) if len(v) else np.nan,
                         "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan, "n": int(len(v)),
                         "n_invalid": int(n_samples - len(v))})
        calls = len(resp)
        log.log("forecast_experiment", mode=mode, conditions=names, control=control, samples=n_samples,
                raw=[r.text for r in resp])
    else:
        raise ValueError("mode must be 'simulate' or 'direct'")
    cond = pd.DataFrame(rows)
    c0 = cond.set_index("condition").loc[control]
    eff = cond[cond.condition != control].copy()
    eff["effect"] = eff["mean"] - c0["mean"]
    if mode == "simulate" and same_profiles:
        def paired_se(c):
            d = ys[c] - ys[control]
            d = d[np.isfinite(d)]
            return float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else np.nan
        eff["se"] = [paired_se(c) for c in eff.condition]
    elif mode == "simulate":
        eff["se"] = np.sqrt(eff.sd ** 2 / eff.n.clip(lower=1) + c0.sd ** 2 / max(c0.n, 1))
    else:
        eff["se"] = np.nan
    return ExperimentForecast(mode, cond, eff[["condition", "effect", "se", "n"]].reset_index(drop=True), calls)


def score_effect_forecasts(pred, true, true_se=None, *, alpha: float = 0.05) -> dict:
    """Accuracy of effect forecasts against measured effects.

    r = Pearson(pred, true); r_adj = r / sqrt(reliability), reliability = 1 - mean(se^2) / var(true) (Spearman's
    correction for sampling error in the measured effects only; NaN when reliability <= 0; may exceed 1).
    sign_acc_sig = share of correct signs among measured effects significant at `alpha` (|true/se| > z).
    Also spearman, slope (OLS of true on pred: magnitude calibration) and n."""
    from scipy import stats as st
    p, t = np.asarray(pred, float), np.asarray(true, float)
    se = np.asarray(true_se, float) if true_se is not None else None
    ok = np.isfinite(p) & np.isfinite(t) & (np.isfinite(se) if se is not None else True)
    p, t = p[ok], t[ok]
    se = se[ok] if se is not None else None
    out = {"n": int(len(p)), "r": np.nan, "r_adj": np.nan, "reliability": np.nan, "spearman": np.nan,
           "slope": np.nan, "sign_acc_sig": np.nan, "n_sig": 0}
    if len(p) < 3 or p.std() == 0 or t.std() == 0:
        return out
    out["r"] = float(np.corrcoef(p, t)[0, 1])
    out["spearman"] = float(st.spearmanr(p, t).statistic)
    out["slope"] = float(np.polyfit(p, t, 1)[0])
    if se is not None:
        rel = 1 - np.mean(se ** 2) / t.var(ddof=1)
        out["reliability"] = float(rel)
        out["r_adj"] = float(out["r"] / math.sqrt(rel)) if rel > 0 else np.nan
        sig = np.abs(t / np.where(se > 0, se, np.nan)) > st.norm.ppf(1 - alpha / 2)
        out["n_sig"] = int(sig.sum())
        if sig.any():
            out["sign_acc_sig"] = float((np.sign(p[sig]) == np.sign(t[sig])).mean())
    return out


def score_condition_forecasts(df: pd.DataFrame, *, study: str = "study", condition: str = "condition",
                              pred: str = "pred_mean", true: str = "true_mean", true_se: str | None = "true_se",
                              outcome: str | None = None, repeats: int = 16, seed: int = 0,
                              alpha: float = 0.05) -> dict:
    """Hewitt et al. (2024) protocol on condition-level means: per repeat pick one control condition per study at
    random (and, with `outcome`, one dependent variable per study, as the paper does when a study has several),
    form every other condition minus that control (predicted and measured; se of a contrast = sqrt(se_t^2 + se_c^2)),
    pool across studies and score; report the median of each score over `repeats` (paper: 16). Rows are ordered by
    `condition` inside a study, so the draw does not depend on row order. Without `outcome`, every row group of
    `study` is one unit (pass a study x outcome key to pool all outcomes). Units with one condition are skipped."""
    rng = np.random.default_rng(seed)
    keys = [study] + ([outcome] if outcome else [])
    units = {k: g.sort_values(condition) for k, g in df.groupby(keys, sort=True) if len(g) >= 2}
    by_study: dict = {}
    for k in units:
        by_study.setdefault(k[0] if isinstance(k, tuple) else k, []).append(k)
    reps = []
    for _ in range(repeats):
        P, T, S = [], [], []
        for ks in by_study.values():
            g = units[ks[int(rng.integers(len(ks)))]] if outcome else units[ks[0]]
            c = int(rng.integers(len(g)))
            gp, gt = g[pred].to_numpy(float), g[true].to_numpy(float)
            m = np.arange(len(g)) != c
            P.append(gp[m] - gp[c])
            T.append(gt[m] - gt[c])
            if true_se:
                gs = g[true_se].to_numpy(float)
                S.append(np.sqrt(gs[m] ** 2 + gs[c] ** 2))
        if not P:
            break
        reps.append(score_effect_forecasts(np.concatenate(P), np.concatenate(T),
                                           np.concatenate(S) if true_se else None, alpha=alpha))
    if not reps:
        return {"n": 0, "repeats": 0}
    out = {k: float(np.nanmedian([r[k] for r in reps])) if any(np.isfinite(r[k]) for r in reps) else np.nan
           for k in ("r", "r_adj", "spearman", "slope", "sign_acc_sig", "reliability")}
    out.update(n=int(np.median([r["n"] for r in reps])), n_sig=int(np.median([r["n_sig"] for r in reps])),
               n_studies=len(by_study), n_units=len(units), repeats=len(reps))
    return out


__all__ = [
    "Variable", "Hypothesis", "DecisionLog", "VARIABLE_KINDS", "PROTOCOLS", "HYPOTHESIS_SYSTEM", "hypothesis_prompt",
    "parse_hypotheses", "propose_hypotheses", "design", "encode", "estimate", "parse_number",
    "InterviewSimulator", "ConversationSimulator", "JUDGE_SYSTEM", "StructuralSimulator", "linear_world", "market_world",
    "make_simulator", "PATH_SYSTEM", "path_prediction_prompt", "predict_paths", "compare_paths", "predict_outcomes",
    "HypothesisRun", "AutoSocialScienceResult", "AutoSocialScience",
    "PROMPT_BANK", "DIRECT_SYSTEM", "profile_text", "ExperimentForecast", "forecast_experiment",
    "score_effect_forecasts", "score_condition_forecasts",
]
