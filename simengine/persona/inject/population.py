"""Population-level injection: ask about a demographic cell, not one person, then rectify with a few real labels.

Injectors (family "population") describe the person's cell (the `keys` fields) and either ask for the cell's
answer distribution (verbalized JSON or a sequence of draws, Meister et al. 2024) or for one random member of the
cell (silicon sampling with a per-person draw index). Everyone in a cell gets the same distribution prompt, so a
cell costs one call (the LLM disk cache serves the rest). The JSON keeps an "answer" field (the modal option), so
`compare_injectors` scores them per person with the default parser; `parse_distribution` reads the distribution.

    import simengine.persona.inject as pi
    from simengine.persona.inject import population as pop
    inj = pi.get_injector("cell_verbalized", keys=["age_band", "region"])
    res = pop.simulate_population(llm, people, question, inj)       # one call per cell
    res.dist["age_band=30-39|region=north"]                         # {option: share}
    truth = pop.empirical_distribution(people, question, keys=["age_band", "region"])
    pop.tvd(res.overall, truth.overall)                             # Meister et al.'s metric
    lab = people[:50]                                               # the few people with a real answer
    pop.ppi_distribution([p["answers"][question["id"]] for p in lab], [res.per_person[p["id"]] for p in lab],
                         [res.per_person[p["id"]] for p in people[50:]], question["options"])

Estimators: `ensemble` (linear pool across injectors / models), `calibrate` (Platt or isotonic for individual
probabilities) and `ppi_mean` / `ppi_distribution` (PPI and PPI++: few real labels + many simulated ones,
rectified estimate with a normal CI).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from ...sim.parse import normalize_probs, parse_json
from ._base import REGISTRY, Injector, MissingPersonField, Prompt, get_injector, parse_option, register

FAMILY = "population"
MEISTER = "Meister, Guestrin & Hashimoto 2024, Benchmarking Distributional Alignment of LLMs (arXiv:2411.05403)"
SUBPOP = "Suh et al. 2025, SubPOP (arXiv:2502.16761)"
ARGYLE = "Argyle et al. 2023, Out of One, Many: silicon sampling (arXiv:2209.06899)"


# ---------------------------------------------------------------- cells

def cell_fields(person: dict, keys: Sequence[str] | None) -> dict:
    """The person's cell attributes (all fields when keys is None); a missing key raises MissingPersonField."""
    f = Injector.person_fields(person)
    if keys is None:
        return {k: v for k, v in f.items() if v not in (None, "")}
    missing = [k for k in keys if f.get(k) in (None, "")]
    if missing:
        raise MissingPersonField(f"cell keys {missing} missing from person {person.get('id')!r} fields "
                                 f"(cell = {list(keys)})")
    return {k: f[k] for k in keys}


def cell_key(person: dict, keys: Sequence[str] | None) -> str:
    """Stable cell id, e.g. "age_band=30-39|region=north"."""
    return "|".join(f"{k}={v}" for k, v in cell_fields(person, keys).items())


def describe_cell(cf: Mapping) -> str:
    return "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in cf.items()) or "the general population"


def _options(question: dict, who: str) -> list[str]:
    opts = [str(o) for o in (question.get("options") or [])]
    if len(opts) < 2:
        raise ValueError(f"{who} needs a choice question with >= 2 options (question {question.get('id')!r})")
    return opts


def _opt_lines(opts: Sequence[str]) -> str:
    return "\n".join(f"- {o}" for o in opts)


def _draw(seed: int, pid: Any, draws: int) -> int:
    """Deterministic 1-based draw index for a person (sha256, not Python's salted hash)."""
    h = hashlib.sha256(f"{seed}\x00{pid}".encode()).hexdigest()
    return int(h[:12], 16) % draws + 1


class _Cell(Injector):
    """Shared constructor: `keys` = the fields that define the cell (None = all fields)."""
    family = FAMILY
    needs = ("fields",)

    def __init__(self, keys: Sequence[str] | None = None, **params: Any):
        super().__init__(**({"keys": list(keys)} if keys else {}), **params)
        self.keys = list(keys) if keys else None

    def group_system(self, person: dict) -> tuple[str, dict]:
        cf = cell_fields(person, self.keys)
        return ("You simulate the answers of a group of survey respondents, not of one person. Please simulate "
                f"the answers of a group of people with these attributes: {describe_cell(cf)}.\n"
                "Think about the full spread of opinion in this group, including minority views."), cf


# ---------------------------------------------------------------- injectors

@register
class CellVerbalized(_Cell):
    """Meister et al.'s best expression method: the cell's distribution as JSON percentages."""
    name = "cell_verbalized"
    paper = MEISTER
    description = ("Persona-steered group prompt for the person's cell (keys=); asks for the share choosing each "
                   "option as JSON {'distribution': {option: percent}, 'answer': modal option}. One call per cell.")

    def render(self, person: dict, question: dict) -> Prompt:
        opts = _options(question, self.name)
        system, cf = self.group_system(person)
        example = json.dumps({"distribution": {o: "<percent>" for o in opts}, "answer": "<most chosen option>"})
        user = (f"Question: {question['text']}\nAnswer options:\n{_opt_lines(opts)}\n\n"
                "Estimate the percentage of this group that would choose each option. Percentages must sum to 100. "
                f"Reply with JSON only: {example}")
        return Prompt(system, user, meta={"cell": describe_cell(cf), "kind": "distribution"})


@register
class CellSequence(_Cell):
    """Meister et al.'s "sequence" method: n independent draws from the group, written out."""
    name = "cell_sequence"
    paper = MEISTER
    description = ("Group prompt for the cell; asks for n (default 30) answers from n random members as a JSON "
                   "list, plus the modal answer. parse_distribution counts the draws. One call per cell.")

    def __init__(self, n: int = 30, keys: Sequence[str] | None = None, **params: Any):
        if int(n) < 2:
            raise ValueError("cell_sequence needs n >= 2 draws")
        super().__init__(keys=keys, n=n, **params)
        self.n = int(n)

    def render(self, person: dict, question: dict) -> Prompt:
        opts = _options(question, self.name)
        system, cf = self.group_system(person)
        user = (f"Question: {question['text']}\nAnswer options:\n{_opt_lines(opts)}\n\n"
                f"Draw {self.n} respondents at random from this group and write down the option each one chooses, "
                f"in order (exactly {self.n} entries, each an exact option text). Reply with JSON only: "
                '{"samples": ["<option>", ...], "answer": "<the most frequent option in samples>"}')
        return Prompt(system, user, meta={"cell": describe_cell(cf), "kind": "sequence", "n": self.n})


@register
class CellSample(_Cell):
    """Silicon sampling of one cell member; the draw index varies the prompt per person (seeded)."""
    name = "cell_sample"
    paper = ARGYLE + "; seeded draws over " + MEISTER
    description = ("First-person: 'you are respondent #j, drawn at random from people with <cell>'; j comes "
                   "from sha256(seed, person id) mod draws, so cell members get distinct prompts (and cache "
                   "entries). Works for numeric questions too. Pair with temperature > 0 via llm_kw.")

    def __init__(self, draws: int = 30, seed: int = 0, keys: Sequence[str] | None = None, **params: Any):
        if int(draws) < 1:
            raise ValueError("cell_sample needs draws >= 1")
        super().__init__(keys=keys, draws=draws, seed=seed, **params)
        self.draws, self.seed = int(draws), int(seed)

    def render(self, person: dict, question: dict) -> Prompt:
        cf = cell_fields(person, self.keys)
        j = _draw(self.seed, person.get("id"), self.draws)
        system = (f"You are respondent #{j} in a random sample of people with these attributes: {describe_cell(cf)}. "
                  "You are one particular member of this group, not its average; answer as this respondent would.")
        return Prompt(system, self.question_block(question),
                      meta={"cell": describe_cell(cf), "kind": "sample", "draw": j})


@register
class CellFewshot(_Cell):
    """Meister et al.'s few-shot steering: the cell's real distributions on related questions as examples."""
    name = "cell_fewshot"
    paper = MEISTER
    description = ("Verbalized cell distribution plus k example questions with the cell's observed answer shares "
                   "(from people=, questions=; none given -> zero-shot). Examples: the 2k most similar questions (word overlap), then k by "
                   "max-min TVD diversity. Target and exclude ids never used (known_answers per member).")

    def __init__(self, people: Sequence[dict] = (), questions: Sequence[dict] = (), k: int = 5, min_n: int = 3,
                 keys: Sequence[str] | None = None, **params: Any):
        # without people= / questions= there are no examples and the prompt equals cell_verbalized's task
        super().__init__(keys=keys, k=k, min_n=min_n, **params)
        self.people, self.questions = list(people), [q for q in questions if q.get("options")]
        self.k, self.min_n = int(k), int(min_n)

    @property
    def label(self) -> str:
        shown = {k: v for k, v in self.params.items() if k not in ("people", "questions")}
        tail = ",".join(f"{k}={v}" for k, v in sorted(shown.items()))
        return f"{self.name}({tail})" if tail else self.name

    def examples(self, person: dict, question: dict) -> list[tuple[dict, dict[str, float], int]]:
        """[(example question, cell distribution, n)] for the person's cell; never the target or exclude ids."""
        key = cell_key(person, self.keys)
        members = [p for p in self.people if _safe_key(p, self.keys) == key]
        drop = {str(x) for x in (question.get("exclude") or ())} | {str(question.get("id"))}
        cands = []
        for q in self.questions:
            if str(q.get("id")) in drop:
                continue
            opts = [str(o) for o in q["options"]]
            # leakage rule: each member's answers pass through known_answers for the TARGET question
            ans = [Injector.known_answers(m, question).get(q.get("id")) for m in members]
            ans = [str(a) for a in ans if a is not None and str(a) in opts]
            if len(ans) < self.min_n:
                continue
            d = {o: ans.count(o) / len(ans) for o in opts}
            cands.append((_overlap(question["text"], q["text"]), str(q.get("id")), q, d, len(ans)))
        cands.sort(key=lambda c: (-c[0], c[1]))
        pool = cands[:2 * self.k]
        chosen = pool[:1]
        while len(chosen) < min(self.k, len(pool)):
            rest = [c for c in pool if c not in chosen]
            chosen.append(max(rest, key=lambda c: (min(_tvd_any(c[3], s[3]) for s in chosen), c[0], c[1])))
        return [(c[2], c[3], c[4]) for c in chosen]

    def render(self, person: dict, question: dict) -> Prompt:
        opts = _options(question, self.name)
        system, cf = self.group_system(person)
        ex = self.examples(person, question)
        blocks = []
        for q, d, n in ex:
            shares = json.dumps({o: round(100 * v) for o, v in d.items()})
            blocks.append(f"Question: {q['text']}\nAnswer options:\n{_opt_lines([str(o) for o in q['options']])}\n"
                          f"Observed shares in this group (n={n}): {shares}")
        head = ("Here is how this group actually answered some related questions:\n\n" + "\n\n".join(blocks) + "\n\n"
                if blocks else "")
        example = json.dumps({"distribution": {o: "<percent>" for o in opts}, "answer": "<most chosen option>"})
        user = (f"{head}Now estimate for this question.\nQuestion: {question['text']}\nAnswer options:\n"
                f"{_opt_lines(opts)}\n\nEstimate the percentage of this group that would choose each option. "
                f"Percentages must sum to 100. Reply with JSON only: {example}")
        return Prompt(system, user, meta={"cell": describe_cell(cf), "kind": "distribution",
                                          "examples": [str(q.get("id")) for q, _, _ in ex]})


def _safe_key(person: dict, keys) -> str | None:
    try:
        return cell_key(person, keys)
    except MissingPersonField:
        return None


_WORD = re.compile(r"[a-z0-9]+")


def _overlap(a: str, b: str) -> float:
    """Jaccard word overlap (stand-in for the paper's text-embedding similarity)."""
    sa, sb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def _tvd_any(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    return tvd(p, q) if set(p) == set(q) else 1.0


# ---------------------------------------------------------------- parsing

def parse_distribution(text: Any, question: dict) -> dict[str, float] | None:
    """{option: share} from a reply: a "distribution" map (percent or fraction, keyed by option text or 1-based
    number), or a "samples" list (counted; entries that are not options are dropped). None when unusable."""
    opts = [str(o) for o in (question.get("options") or [])]
    if not opts:
        return None
    obj = parse_json(text)
    if not isinstance(obj, dict):
        return None
    low = {str(k).strip().lower(): v for k, v in obj.items()}
    if "distribution" in low:
        v = normalize_probs(low["distribution"], len(opts), opts)
        return None if v is None else dict(zip(opts, map(float, v)))
    if isinstance(low.get("samples"), list):
        norm = {o.strip().lower(): o for o in opts}
        hits = [norm.get(str(s).strip().lower()) for s in low["samples"]]
        hits = [h for h in hits if h is not None]
        if not hits:
            return None
        return {o: hits.count(o) / len(hits) for o in opts}
    return None


def parse_modal(text: Any, question: dict) -> Any:
    """compare_injectors parser: the modal option of a parsed distribution (ties: first option), else
    parse_option (so cell_sample and other families score the same way)."""
    d = parse_distribution(text, question)
    if d:
        return max(d, key=d.get)
    return parse_option(text, question)


# ---------------------------------------------------------------- population runs

@dataclass
class PopulationResult:
    dist: dict[str, dict[str, float]]          # cell -> {option: share}; parsed members only
    overall: dict[str, float]                  # people-weighted pool of the cells
    n: dict[str, int]                          # people per cell
    parsed: dict[str, int]                     # people per cell with a usable reply
    per_person: dict[str, dict[str, float]]    # person id -> the distribution their cell/draw gave
    calls: int                                 # unique prompts sent
    prompts: list[Prompt] = field(default_factory=list)
    replies: list[str | None] = field(default_factory=list)


def _vec(d: Mapping[str, float] | None, opts: Sequence[str]) -> np.ndarray | None:
    return None if d is None else np.array([float(d.get(o, 0.0)) for o in opts])


def _pool(vecs: list[np.ndarray], w: Sequence[float] | None = None) -> np.ndarray:
    w = np.ones(len(vecs)) if w is None else np.asarray(w, float)
    v = (np.vstack(vecs) * w[:, None]).sum(0) / w.sum()
    return v / v.sum() if v.sum() > 0 else v


def simulate_population(llm, people: Sequence[dict], question: dict, injector: Injector | str = "cell_verbalized",
                        *, keys: Sequence[str] | None = None, max_tokens: int = 512, parallel: int = 4,
                        llm_kw: dict | None = None, **params: Any) -> PopulationResult:
    """Render every person, send each distinct prompt once, and pool per cell. A distribution reply counts as
    that person's vector; a single answer counts as one-hot (so cell_sample and individual injectors work).
    keys: cell grouping (default: the injector's keys; also passed to a cell injector given by name)."""
    from ...llm.parallel import parallel_map

    if isinstance(injector, str) and keys is not None and "keys" not in params and \
            issubclass(REGISTRY.get(injector, Injector), _Cell):
        params["keys"] = list(keys)             # a cell injector by name: group the prompt by the same keys
    inj = get_injector(injector, **params)
    opts = _options(question, "simulate_population")
    keys = keys if keys is not None else getattr(inj, "keys", None)
    people = list(people)
    rendered = [inj(p, question) for p in people]
    uniq: dict[tuple[str, str], int] = {}
    for pr in rendered:
        uniq.setdefault((pr.system, pr.flatten()), len(uniq))
    order = list(uniq)
    kw = dict(llm_kw or {})
    out = parallel_map(lambda sp: llm.complete(sp[0], sp[1], max_tokens=max_tokens, **kw), order,
                       parallel=parallel, return_exceptions=True)
    texts = [None if isinstance(r, Exception) else r.text for r in out]
    cells: dict[str, list[np.ndarray]] = {}
    n: dict[str, int] = {}
    per_person = {}
    replies = []
    for p, pr in zip(people, rendered):
        t = texts[uniq[(pr.system, pr.flatten())]]
        replies.append(t)
        ck = cell_key(p, keys)
        n[ck] = n.get(ck, 0) + 1
        d = parse_distribution(t, question)
        if d is None:
            a = parse_option(t, question)
            d = {o: float(o == a) for o in opts} if a is not None else None
        if d is None:
            continue
        per_person[str(p["id"])] = d
        cells.setdefault(ck, []).append(_vec(d, opts))
    dist = {c: dict(zip(opts, map(float, _pool(v)))) for c, v in cells.items()}
    parsed = {c: len(v) for c, v in cells.items()}
    overall = (dict(zip(opts, map(float, _pool([_vec(dist[c], opts) for c in dist], [n[c] for c in dist])))) if dist
               else {})
    return PopulationResult(dist=dist, overall=overall, n=n, parsed=parsed, per_person=per_person,
                            calls=len(order), prompts=rendered, replies=replies)


def empirical_distribution(people: Sequence[dict], question: dict, keys: Sequence[str] | None = None,
                           *, answers: Mapping[str, Any] | None = None) -> PopulationResult:
    """Real answer shares per cell (truth side). answers: {person_id: answer}, default person['answers'][qid].
    Never pass this to a prompt for the same question; it is the scoring target."""
    opts = _options(question, "empirical_distribution")
    qid = question.get("id")
    cells: dict[str, list[str]] = {}
    n: dict[str, int] = {}
    for p in people:
        ck = cell_key(p, keys)
        n[ck] = n.get(ck, 0) + 1
        a = answers.get(str(p["id"])) if answers is not None else (p.get("answers") or {}).get(qid)
        if a is not None and str(a) in opts:
            cells.setdefault(ck, []).append(str(a))
    dist = {c: {o: v.count(o) / len(v) for o in opts} for c, v in cells.items()}
    tot = sum(len(v) for v in cells.values())
    overall = {o: sum(v.count(o) for v in cells.values()) / tot for o in opts} if tot else {}
    return PopulationResult(dist=dist, overall=overall, n=n, parsed={c: len(v) for c, v in cells.items()},
                            per_person={}, calls=0)


# ---------------------------------------------------------------- estimators

def tvd(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Total variation distance 0.5 * sum |p - q| over the union of options (Meister et al.'s metric)."""
    ks = set(p) | set(q)
    return 0.5 * sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in ks)


def ensemble(dists: Sequence[Mapping], weights: Sequence[float] | None = None) -> dict:
    """Linear opinion pool. Each item is {option: share} or {cell: {option: share}} (e.g. PopulationResult.dist
    from several injectors or models); cells missing in some members pool over the members that have them."""
    if not dists:
        raise ValueError("ensemble needs at least one distribution")
    w = [1.0] * len(dists) if weights is None else [float(x) for x in weights]
    if len(w) != len(dists) or min(w) < 0 or sum(w) <= 0:
        raise ValueError("weights must be non-negative, one per distribution, not all zero")
    nested = all(isinstance(v, Mapping) for d in dists for v in d.values())
    if not nested:
        opts = list(dict.fromkeys(o for d in dists for o in d))
        return dict(zip(opts, map(float, _pool([_vec(d, opts) for d in dists], w))))
    cells = list(dict.fromkeys(c for d in dists for c in d))
    return {c: ensemble([d[c] for d in dists if c in d], [wi for d, wi in zip(dists, w) if c in d]) for c in cells}


def calibrate(p_train: Sequence[float], y_train: Sequence[int], p_apply: Sequence[float],
              method: str = "platt") -> np.ndarray:
    """Recalibrate simulated individual probabilities of a binary outcome on a labelled subset.
    platt: logistic regression on logit(p) (Platt 1999); isotonic: monotone step map (Zadrozny & Elkan 2002).
    Fit on people whose labels are NOT used for the final estimate (or cross-fit), else the CI is optimistic."""
    p, y, pa = np.asarray(p_train, float), np.asarray(y_train, float), np.asarray(p_apply, float)
    if len(p) != len(y) or len(p) < 2:
        raise ValueError("calibrate needs >= 2 aligned (p, y) training pairs")
    if len(np.unique(y)) < 2:
        raise ValueError("calibrate needs both outcomes (0 and 1) in y_train")
    eps = 1e-6
    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        z = lambda v: np.log(np.clip(v, eps, 1 - eps) / (1 - np.clip(v, eps, 1 - eps))).reshape(-1, 1)  # noqa: E731
        return LogisticRegression(C=1e4, max_iter=500).fit(z(p), y).predict_proba(z(pa))[:, 1]
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y).predict(pa)
    raise ValueError("method must be 'platt' or 'isotonic'")


def ppi_mean(y: Sequence[float], f_labeled: Sequence[float], f_unlabeled: Sequence[float], *,
             lam: float | str = "auto", alpha: float = 0.05) -> dict:
    """Prediction-powered mean with a normal CI.

    theta = mean(y) + lam * (mean(f_unlabeled) - mean(f_labeled)), with n labelled pairs (y, f_labeled) and N
    unlabelled predictions. lam=1: PPI (Angelopoulos et al. 2023, arXiv:2301.09633); lam=0: labels only;
    "auto": PPI++ power tuning lam = Cov(y, f) / ((1 + n/N) Var(f)) with Var over all n + N predictions
    (Angelopoulos, Duchi & Zrnic 2023, arXiv:2311.01453, Example 6.1; not clipped, the mean case is convex).
    se^2 = lam^2 Var(f_unlabeled) / N + Var(y - lam f_labeled) / n."""
    y, fl, fu = (np.asarray(v, float) for v in (y, f_labeled, f_unlabeled))
    n, N = len(y), len(fu)
    if len(fl) != n or n < 2 or N < 2:
        raise ValueError("ppi_mean needs >= 2 aligned labelled pairs and >= 2 unlabelled predictions")
    if not all(np.isfinite(v).all() for v in (y, fl, fu)):
        raise ValueError("ppi_mean inputs must be finite (drop unparsed rows first)")
    if lam == "auto":
        vf = np.var(np.r_[fl, fu], ddof=1)
        lam_v = float(np.cov(y, fl, ddof=1)[0, 1] / ((1 + n / N) * vf)) if vf > 0 else 0.0
    else:
        lam_v = float(lam)
    est = float(y.mean() + lam_v * (fu.mean() - fl.mean()))
    se = float(np.sqrt(lam_v ** 2 * np.var(fu, ddof=1) / N + np.var(y - lam_v * fl, ddof=1) / n))
    z = NormalDist().inv_cdf(1 - alpha / 2)
    cse = float(np.std(y, ddof=1) / np.sqrt(n))
    return {"estimate": est, "lo": est - z * se, "hi": est + z * se, "se": se, "lam": lam_v,
            "classical": float(y.mean()), "classical_lo": float(y.mean() - z * cse),
            "classical_hi": float(y.mean() + z * cse), "n": n, "N": N,
            "width_ratio": se / cse if cse > 0 else None}


def ppi_distribution(labels: Sequence[Any], f_labeled: Sequence[Mapping[str, float]],
                     f_unlabeled: Sequence[Mapping[str, float]], options: Sequence[Any], *,
                     lam: float | str = "auto", alpha: float = 0.05) -> dict[str, dict]:
    """Per-option share of a choice question: ppi_mean on one-hot real labels vs simulated per-person
    probabilities (e.g. PopulationResult.per_person). Each option gets its own lam and CI (marginal, not
    simultaneous; shares need not sum to 1)."""
    opts = [str(o) for o in options]
    lab = [str(a) for a in labels]
    bad = sorted({a for a in lab if a not in opts})
    if bad:
        raise ValueError(f"labels not among options: {bad[:5]}")
    return {o: ppi_mean([float(a == o) for a in lab], [float(d.get(o, 0.0)) for d in f_labeled],
                        [float(d.get(o, 0.0)) for d in f_unlabeled], lam=lam, alpha=alpha) for o in opts}


__all__ = ["CellVerbalized", "CellSequence", "CellSample", "CellFewshot", "cell_key", "cell_fields",
           "describe_cell", "parse_distribution", "parse_modal", "PopulationResult", "simulate_population",
           "empirical_distribution", "tvd", "ensemble", "calibrate", "ppi_mean", "ppi_distribution"]
