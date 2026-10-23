"""Run several persona injectors on the same (person, question) cells and compare them with paired CIs.

    import simengine.persona.inject as pi
    plan = pi.compare_injectors(llm, people, questions, truth, ["plain_fields", "bio"], dry_run=True)
    plan.estimate                     # {"calls": ..., "input_tokens_est": ..., "usd_max": ...}; no LLM call
    res = pi.compare_injectors(llm, people, questions, truth, ["plain_fields", "bio"], max_usd=2.0)
    res.table                         # one row per arm: score, CI, diff vs no_persona and vs its swap

Arms: every injector, `no_persona` (bare question) and `swap_<injector>` (the injector with another person's
persona, same question, scored against the original person). Per-cell answers form an EvalFrame
(group = person id, item = question id, truth, pred_<arm>); CIs come from `simengine.evidence.paired_diff`
(cluster bootstrap over people). An unparsed or failed reply scores as wrong (sentinel UNPARSED).

truth: {(person_id, question_id): answer}, a callable(person, question) -> answer | None, or "answers"
(person["answers"][question_id]). Cells without truth are not called.

History: a Prompt with prior turns goes out flattened (`Prompt.flatten()`) unless the LLM sets
`supports_history = True`, then it gets `history=[(role, text), ...]` as a keyword. No backend sets it today:
simengine.llm.Claude would accept `messages=` but leaves messages out of its cache key, so distinct histories
could collide in the disk cache.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from ...llm.parallel import parallel_map
from ._base import Injector, Prompt, get_injector, parse_option, swap_wrapper

UNPARSED = "<unparsed>"
CONTROLS = ("no_persona", "swap")
CHARS_PER_TOKEN = 4


@dataclass
class CompareResult:
    table: pd.DataFrame | None            # one row per arm (None when dry_run)
    frame: pd.DataFrame | None            # EvalFrame, one row per (person, question) cell
    estimate: dict                        # cost estimate made before any call
    arms: list[str]
    cells: list[tuple[str, str]]          # (person_id, question_id) that have truth
    prompts: dict[str, list[Prompt]]      # arm -> prompts aligned with cells
    replies: dict[str, list[str | None]] = field(default_factory=dict)
    pairs: list[dict] = field(default_factory=list)   # raw paired_diff dicts
    errors: dict[str, int] = field(default_factory=dict)


def _truth_fn(truth) -> Callable[[dict, dict], Any]:
    if callable(truth):
        return truth
    if truth == "answers":
        return lambda p, q: (p.get("answers") or {}).get(q.get("id"))
    if isinstance(truth, Mapping):
        return lambda p, q: truth.get((str(p["id"]), str(q.get("id"))))
    raise TypeError('truth must be a {(person_id, question_id): answer} mapping, a callable or "answers"')


def _key(v: Any) -> Any:
    """Comparable label: 3.0 -> "3", None -> UNPARSED, everything else str."""
    if v is None:
        return UNPARSED
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v)


def estimate_cost(llm, prompts: Sequence[Prompt], max_tokens: int) -> dict:
    """Calls, rough input tokens (chars / 4) and an upper-bound USD (every reply at max_tokens) when the model
    is in simengine.llm.pricing; usd_max / usd_per_call_max None otherwise.

    Each prompt is priced as its own call and summed, so the long-context price applies only to a prompt that
    is itself over the model's threshold (summing first and pricing once overstated Haiku runs ~5x).
    usd_per_call_max: the dearest single call."""
    mt = int(max_tokens)
    tins = [(len(p.system) + len(p.flatten())) // CHARS_PER_TOKEN for p in prompts]
    usd = per_call = None
    try:
        from ...llm.pricing import cost_usd
        model = getattr(llm, "model", "-")
        costs = [cost_usd(model, t, mt) for t in tins]
        usd, per_call = round(sum(costs), 4), round(max(costs, default=0.0), 6)
    except KeyError:
        pass
    return {"calls": len(prompts), "input_tokens_est": sum(tins), "output_tokens_max": len(prompts) * mt,
            "usd_max": usd, "usd_per_call_max": per_call, "model": getattr(llm, "model", None)}


def _resolve(injectors) -> list[Injector]:
    out = []
    for x in injectors:
        if isinstance(x, tuple):
            out.append(get_injector(x[0], **(x[1] if len(x) > 1 else {})))
        else:
            out.append(get_injector(x))
    labels = [i.label for i in out]
    dup = {l for l in labels if labels.count(l) > 1}
    if dup:
        raise ValueError(f"duplicate injector arms {sorted(dup)}: give them different params")
    return out


def compare_injectors(llm, people: Sequence[dict], questions: Sequence[dict], truth, injectors: Sequence, *,
                      parse: Callable[[Any, dict], Any] = parse_option, metric: str = "accuracy",
                      controls: Sequence[str] = CONTROLS, seed: int = 0, parallel: int = 4, max_tokens: int = 256,
                      B: int = 1000, task: str = "multiclass", dry_run: bool = False, max_usd: float | None = None,
                      llm_kw: dict | None = None) -> CompareResult:
    """Render every (person, question) with each injector (+ controls), call the LLM, parse, score, paired CIs.

    injectors: names, Injector instances or (name, params) tuples. controls: subset of ("no_persona", "swap");
    the swap control is added for every injector. parse(text, question) -> answer | None. metric: any
    evidence metric for `task` (multiclass: accuracy, balanced_accuracy, kappa, macro_f1). dry_run: render and
    estimate only. max_usd: refuse to start when the upper-bound estimate is above it (unpriced model: no check).
    llm_kw: extra keywords for every complete() call (tag=, thinking=, ...)."""
    bad = set(controls) - set(CONTROLS)
    if bad:
        raise ValueError(f"unknown controls {sorted(bad)}; choose from {CONTROLS}")
    people = list(people)
    questions = list(questions)
    for q in questions:
        if "text" not in q:
            raise ValueError("every question needs 'text'")
    qid = lambda i, q: str(q.get("id", f"q{i}"))  # noqa: E731
    questions = [{**q, "id": qid(i, q)} for i, q in enumerate(questions)]
    given = _resolve(injectors)
    inj = [i for i in given if i.name != "no_persona"]
    arms: list[tuple[str, Injector]] = [(i.label, i) for i in inj]
    if "no_persona" in controls or len(inj) < len(given):
        arms.insert(0, ("no_persona", get_injector("no_persona")))
    if "swap" in controls and inj:
        if len(people) < 2:
            raise ValueError("swap control needs at least 2 people (or pass controls=('no_persona',))")
        for i in inj:
            s = swap_wrapper(i, people, seed=seed)
            arms.append((s.label, s))

    tf = _truth_fn(truth)
    cells, cell_obj, truths = [], [], []
    for p in people:
        for q in questions:
            t = tf(p, q)
            if t is None:
                continue
            cells.append((str(p["id"]), q["id"]))
            cell_obj.append((p, q))
            truths.append(t)
    if not cells:
        raise ValueError("no (person, question) cell has truth")

    prompts = {name: [a(p, q) for p, q in cell_obj] for name, a in arms}
    for name, ps in prompts.items():
        for pr in ps:
            pr.meta.setdefault("arm", name)
    flat = [pr for ps in prompts.values() for pr in ps]
    est = estimate_cost(llm, flat, max_tokens)
    est["arms"] = len(arms)
    est["cells"] = len(cells)
    res = CompareResult(table=None, frame=None, estimate=est, arms=[n for n, _ in arms], cells=cells, prompts=prompts)
    if dry_run:
        return res
    if max_usd is not None and est["usd_max"] is not None and est["usd_max"] > max_usd:
        raise ValueError(f"estimated upper bound ${est['usd_max']} > max_usd ${max_usd} ({est['calls']} calls); "
                         "lower max_tokens, cells or arms")

    kw = dict(llm_kw or {})
    hist = bool(getattr(llm, "supports_history", False))

    def call(pr: Prompt):
        if hist and pr.history:
            return llm.complete(pr.system, pr.user, max_tokens=max_tokens, history=list(pr.history), **kw)
        return llm.complete(pr.system, pr.flatten(), max_tokens=max_tokens, **kw)

    out = parallel_map(call, flat, parallel=parallel, return_exceptions=True)
    frame = pd.DataFrame({"group": [c[0] for c in cells], "item": [c[1] for c in cells],
                          "truth": pd.Series([_key(t) for t in truths], dtype=object)})
    k = 0
    for name, _ in arms:
        texts, preds, errs = [], [], 0
        for (p, q) in cell_obj:
            r = out[k]
            k += 1
            if isinstance(r, Exception):
                texts.append(None)
                preds.append(UNPARSED)
                errs += 1
                continue
            texts.append(r.text)
            try:
                preds.append(_key(parse(r.text, q)))
            except Exception:  # noqa: BLE001 - a broken parser cell counts as unparsed
                preds.append(UNPARSED)
        res.replies[name] = texts
        res.errors[name] = errs
        frame["pred_" + name] = pd.Series(preds, dtype=object)
    res.frame = frame
    res.table = _table(res, arms, metric, B, seed, task)
    return res


def _table(res: CompareResult, arms, metric: str, B: int, seed: int, task: str) -> pd.DataFrame:
    from ...evidence import paired_diff

    f = res.frame
    rows = []
    names = [n for n, _ in arms]
    for name, a in arms:
        own = paired_diff(f, name, None, metric=metric, B=B, seed=seed, task=task)
        res.pairs.append(own)
        kind = "control" if name == "no_persona" or name.startswith("swap_") else "injector"
        row = {"arm": name, "kind": kind, "family": a.family, "n": len(f), "n_people": f["group"].nunique(),
               "parsed": int((f["pred_" + name] != UNPARSED).sum()), "errors": res.errors.get(name, 0),
               "metric": own["metric"], "score": own["value_a"], "lo": own["lo"], "hi": own["hi"]}
        for ref, tag in (("no_persona", "vs_no_persona"), (f"swap_{name}", "vs_swap")):
            if ref in names and ref != name:
                d = paired_diff(f, name, ref, metric=metric, B=B, seed=seed, task=task)
                res.pairs.append(d)
                row.update({f"diff_{tag}": d["diff"], f"lo_{tag}": d["lo"], f"hi_{tag}": d["hi"], f"p_{tag}": d["p"]})
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["compare_injectors", "CompareResult", "estimate_cost", "UNPARSED", "CONTROLS"]
