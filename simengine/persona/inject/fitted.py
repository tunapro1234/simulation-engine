"""Fitted injectors: the persona prompt is fitted on known answers to TRAIN questions, then used on held-out ones.

Each injector searches a persona configuration (which attributes to show, in which order, or the framing text) that
maximises agreement with people's real answers on a train split of questions, picks the winner on a validation
split of questions (`simengine.research.prompt_evolution.select_and_test`, optional seal of the winner), and
refuses to render any question it was fitted on, so `compare_injectors` can only score it on held-out questions.

    import simengine.persona.inject as pi
    inj = pi.get_injector("fitted_stepwise", seed=0)
    fit = inj.fit(llm, people, train_questions, "answers", max_calls=2000)   # LLM calls on fit questions only
    fit.config, fit.train_score, fit.val_score                              # chosen keys, accuracies
    res = pi.compare_injectors(llm, people, test_questions, "answers", [inj, "plain_fields"])
    inj(people[0], train_questions[0])                                      # HeldOutViolation

Injectors (family "fitted"; all render `key: value` lines like plain_fields, so the all-keys config IS plain_fields):
- fitted_topk: no LLM. Random-forest importance of each attribute for predicting each fit question's answer
  (Fröhling et al. 2026, arXiv:2609.02526, "feature importance", k=5), summed over questions from each forest's top
  `per_model` (Rupprecht et al. 2025, arXiv:2511.21722, TOP-k), keep the top k.
- fitted_stepwise: coordinate ascent over attribute inclusion with `simengine.research.Loop` (one flip per turn,
  kept only if the paired bootstrap CI on train cells excludes 0; simpler ties kept).
- fitted_grips: GrIPS edit search over attribute lines (delete / swap / add-back), m candidates per iteration,
  patience P (Prasad et al. 2022, arXiv:2203.07281).
- fitted_opro: OPRO over the framing sentence above the attribute lines (`simengine.research.prompt_evolution.opro`;
  Yang et al. 2023, arXiv:2309.03409).

Scores: per (person, train question) cell, 1 if `parse(reply)` equals the truth, else 0 (unparsed and failed calls
count as wrong, as in compare_injectors). Fit state is population-level (one config for everybody); no injector
here shows a person's own answers. `fit(max_calls=)` refuses to start when the upper-bound call count is above it.
LEAKAGE: a field named like the question being asked (or one of its `exclude` ids) is never shown, in fit scoring
and at render, and is never a forest feature for that question (`hidden(question)`, as in inject.generated).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ._base import Injector, Prompt, parse_option, register
from .compare import _key, _truth_fn

HEADER = "You are a person with these attributes:"
TAIL = "Answer as this person would."


class NotFitted(RuntimeError):
    """render() before fit() / load()."""


class HeldOutViolation(ValueError):
    """render() on a question the injector was fitted on (train or validation)."""


@dataclass
class FitResult:
    method: str
    config: dict                          # {"keys": [...]} (+ "instruction" for opro, "weights" for topk)
    train_questions: list[str]
    val_questions: list[str]
    train_score: float | None             # mean cell accuracy of the winner on train questions (None: no LLM)
    val_score: float | None               # same on validation questions (None: no validation split)
    calls: int = 0                        # scorer LLM calls (person x question cells evaluated)
    optimizer_calls: int = 0              # proposer LLM calls (opro)
    history: list[dict] = field(default_factory=list)   # visited configs: text, train, origin, step
    seal: dict | None = None
    meta: dict = field(default_factory=dict)

    @property
    def fit_questions(self) -> set[str]:
        return set(self.train_questions) | set(self.val_questions)

    def to_dict(self) -> dict:
        return {"method": self.method, "config": self.config, "train_questions": self.train_questions,
                "val_questions": self.val_questions, "train_score": self.train_score, "val_score": self.val_score}


def split_questions(qids: Sequence[str], val_frac: float, seed: int = 0) -> tuple[list[str], list[str]]:
    """Random question-level split. val_frac > 0 with >= 2 questions gives at least one of each."""
    qids = list(qids)
    if val_frac <= 0 or len(qids) < 2:
        return qids, []
    order = qids[:]
    random.Random(seed).shuffle(order)
    n_val = min(max(1, round(val_frac * len(qids))), len(qids) - 1)
    val = set(order[:n_val])
    return [q for q in qids if q not in val], [q for q in qids if q in val]


def hidden(question: dict) -> set[str]:
    """Field names never shown for `question`: its id and its `exclude` ids (a field may BE the target answer)."""
    out = {str(x) for x in (question.get("exclude") or ())}
    if question.get("id") is not None:
        out.add(str(question["id"]))
    return out


class _Fitted(Injector):
    """Shared machinery: question split, cells, cached scorer, held-out guard. Not registered."""
    family = "fitted"
    needs = ("fields",)
    llm_free = False

    def __init__(self, keys: Sequence[str] | None = None, val_frac: float = 0.25, top_k: int = 5, seed: int = 0,
                 **params: Any):
        super().__init__(**({"keys": list(keys)} if keys else {}), val_frac=val_frac, top_k=top_k, seed=seed,
                         **params)
        if not 0 <= val_frac < 1:
            raise ValueError("val_frac must be in [0, 1)")
        self.keys = list(keys) if keys else None
        self.val_frac, self.top_k, self.seed = float(val_frac), int(top_k), int(seed)
        self.result: FitResult | None = None

    # ------------------------------------------------------------ rendering
    def system_for(self, person: dict, cfg: dict, question: dict | None = None) -> str:
        f = self.person_fields(person)
        drop = hidden(question) if question is not None else set()
        lines = [f"- {k.replace('_', ' ')}: {f[k]}" for k in cfg["keys"]
                 if k in f and f[k] not in (None, "") and str(k) not in drop]
        if not lines:
            return ""
        return cfg.get("instruction", HEADER) + "\n" + "\n".join(lines) + "\n" + TAIL

    def _prompt(self, person: dict, question: dict, cfg: dict) -> Prompt:
        return Prompt(system=self.system_for(person, cfg, question), user=self.question_block(question))

    def render(self, person: dict, question: dict) -> Prompt:
        if self.result is None:
            raise NotFitted(f"injector {self.name!r} is not fitted: call .fit(llm, people, train_questions, truth) "
                            "(or .load(fit_dict)) and pass the instance to compare_injectors")
        qid = str(question.get("id"))
        if qid in self.result.fit_questions:
            raise HeldOutViolation(f"injector {self.name!r} was fitted on question {qid!r}; score it on held-out "
                                   "questions only")
        p = self._prompt(person, question, self.result.config)
        p.meta["fitted"] = {"method": self.result.method, "keys": list(self.result.config["keys"])}
        return p

    def load(self, fit: FitResult | dict) -> "_Fitted":
        """Restore a fit (FitResult or its to_dict()); the fit questions stay refused."""
        if isinstance(fit, dict):
            fit = FitResult(method=fit["method"], config=dict(fit["config"]),
                            train_questions=list(fit["train_questions"]), val_questions=list(fit["val_questions"]),
                            train_score=fit.get("train_score"), val_score=fit.get("val_score"))
        if fit.method != self.name:
            raise ValueError(f"fit is from {fit.method!r}, not {self.name!r}")
        self.result = fit
        return self

    # ------------------------------------------------------------ fitting
    def estimate_calls(self, n_attrs: int, n_train: int, n_val: int) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, llm, people: Sequence[dict], questions: Sequence[dict], truth="answers", *,
            val_questions: Sequence[str] | None = None, parse: Callable[[Any, dict], Any] = parse_option,
            parallel: int = 4, max_tokens: int = 256, max_calls: int | None = None, registry_dir=None, prereg=None,
            llm_kw: dict | None = None, optimizer=None) -> FitResult:
        """Fit on `questions` (every one is then refused by render). val_questions: explicit validation ids
        (default: a seeded `val_frac` split). truth as in compare_injectors. max_calls: refuse to start when the
        upper-bound number of LLM calls is above it. registry_dir / prereg: seal the winner / verify a locked
        prereg before the fit is returned (simengine.research). optimizer: proposer LLM for opro (default llm)."""
        people = list(people)
        for p in people:
            self.check(p)
        qs = []
        for q in questions:
            if q.get("id") is None or "text" not in q:
                raise ValueError("every fit question needs 'id' and 'text'")
            qs.append({**q, "id": str(q["id"])})
        if len({q["id"] for q in qs}) != len(qs):
            raise ValueError("fit question ids must be unique")
        qids = [q["id"] for q in qs]
        if val_questions is not None:
            val = [str(v) for v in val_questions]
            bad = set(val) - set(qids)
            if bad:
                raise ValueError(f"val_questions {sorted(bad)} are not among the fit questions")
            train = [q for q in qids if q not in set(val)]
        else:
            train, val = split_questions(qids, 0.0 if self.llm_free else self.val_frac, self.seed)
        if not train:
            raise ValueError("no train question left after the validation split")
        attrs = self.keys or list(dict.fromkeys(k for p in people for k in self.person_fields(p)))
        if not attrs:
            raise ValueError(f"injector {self.name!r}: the people have no attributes to fit")
        tf = _truth_fn(truth)
        by_id = {q["id"]: q for q in qs}
        cells = {s: [] for s in ("train", "val")}
        for split, ids in (("train", train), ("val", val)):
            for qid in ids:
                q = by_id[qid]
                for p in people:
                    t = tf(p, q)
                    if t is not None:
                        cells[split].append({"unit": f"{p['id']}|{qid}", "person": p, "question": q, "truth": t})
        if not cells["train"]:
            raise ValueError("no (person, train question) cell has truth")
        est = self.estimate_calls(len(attrs), len(cells["train"]), len(cells["val"]))
        if max_calls is not None and est["calls"] > max_calls:
            raise ValueError(f"{self.name}: upper bound {est['calls']} LLM calls > max_calls {max_calls}; "
                             "lower the search budget, people or fit questions")
        ctx = _Ctx(self, llm, parse, parallel, max_tokens, dict(llm_kw or {}), attrs, cells, train, val)
        res = self._search(ctx, optimizer=optimizer, registry_dir=registry_dir, prereg=prereg)
        res.meta.setdefault("estimate", est)
        res.meta.setdefault("attributes", attrs)
        self.result = res
        return res

    def _search(self, ctx: "_Ctx", **kw) -> FitResult:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------ shared: config text <-> config
    def cfg_of(self, text: str) -> dict:
        return {"keys": json.loads(text)}

    @staticmethod
    def text_of(keys: Sequence[str]) -> str:
        return json.dumps(list(keys))


class _Ctx:
    """One fit: cells per split, the scorer, a PromptEvaluator over config texts (cache + split discipline)."""

    def __init__(self, inj: _Fitted, llm, parse, parallel, max_tokens, llm_kw, attrs, cells, train, val):
        self.inj, self.llm, self.parse = inj, llm, parse
        self.parallel, self.max_tokens, self.llm_kw = parallel, max_tokens, llm_kw
        self.attrs, self.cells, self.train_q, self.val_q = attrs, cells, train, val
        self.calls = 0
        self._ev = None

    def score(self, text: str, units) -> list[float]:
        from ...llm.parallel import parallel_map

        cfg = self.inj.cfg_of(text)
        units = list(units)
        prompts = [self.inj._prompt(u["person"], u["question"], cfg) for u in units]
        out = parallel_map(lambda pr: self.llm.complete(pr.system, pr.flatten(), max_tokens=self.max_tokens,
                                                        **self.llm_kw),
                           prompts, parallel=self.parallel, return_exceptions=True)
        self.calls += len(prompts)
        sc = []
        for u, r in zip(units, out):
            if isinstance(r, Exception):
                sc.append(0.0)
                continue
            try:
                sc.append(float(_key(self.parse(r.text, u["question"])) == _key(u["truth"])))
            except Exception:  # noqa: BLE001 - a broken parser cell counts as wrong
                sc.append(0.0)
        return sc

    @property
    def ev(self):
        """research.prompt_evolution.PromptEvaluator over config texts; with no validation split the val slot
        holds the train cells but is never read (selection then uses train only)."""
        if self._ev is None:
            from ...research.prompt_evolution import PromptEvaluator
            val = self.cells["val"] or self.cells["train"]
            self._ev = PromptEvaluator(self.score, self.cells["train"], val, None)
        return self._ev

    def train_mean(self, text: str) -> float:
        return float(self.ev.scores(text, "train")["mean"])

    def finish(self, history, method: str, registry_dir=None, prereg=None, optimizer_calls: int = 0,
               meta: dict | None = None) -> FitResult:
        """Pick the winner (validation re-rank of the top_k train configs, or best train), seal, verify prereg."""
        from ...research.prereg import seal, verify_prereg
        from ...research.prompt_evolution import select_and_test

        rec, val_score = None, None
        if self.cells["val"]:
            sel = select_and_test(self.ev, history, method=method, top_k=self.inj.top_k, baselines={},
                                  registry_dir=registry_dir, prereg=prereg, seed=self.inj.seed)
            win, val_score, rec = sel.best, float(sel.validation.val.iloc[0]), sel.seal
        else:
            best = {}
            for c in history:
                if c.text not in best or c.fitness > best[c.text].fitness:
                    best[c.text] = c
            win = sorted(best.values(), key=lambda c: (-c.fitness, len(c.text), c.text))[0]
            if registry_dir is not None:
                rec = seal({"method": method, "winner": win.text, "train": win.train},
                           f"persona.inject {method}: fitted config before held-out questions", registry_dir)
            if prereg is not None:
                chk = verify_prereg(prereg)
                if not chk["ok"]:
                    raise ValueError(f"prereg check failed before held-out use: {chk['reason']}")
        hist = [{"text": c.text, "train": c.train, "origin": c.origin, "step": c.step} for c in history]
        return FitResult(method=method, config=self.inj.cfg_of(win.text), train_questions=list(self.train_q),
                         val_questions=list(self.val_q), train_score=float(win.train), val_score=val_score,
                         calls=self.calls, optimizer_calls=optimizer_calls, history=hist, seal=rec,
                         meta=dict(meta or {}))


# ---------------------------------------------------------------- injectors

@register
class FittedTopK(_Fitted):
    """Random-forest attribute selection. Fröhling et al. 2026 (§3.2, "feature importance" baseline) fit one forest
    per target question from the other variables and keep its k=5 most important; Rupprecht et al. 2025 (§3) sum the
    top-10 importances of many forests into one ranking (TOP-k). Held-out questions have no answers at fit time, so
    here one forest per FIT question, the `per_model` most important attributes of each add their importance to a
    global ranking, keep the top k (one set for every held-out question). No LLM call; one-hot columns are summed back
    to their attribute."""
    name = "fitted_topk"
    paper = ("Fröhling et al. 2026, arXiv:2609.02526 (RF feature-importance selection, k=5); Rupprecht et al. 2025, "
             "arXiv:2511.21722 (summed top-10 importances, TOP-k)")
    description = ("No-LLM fit: random forest per fit question predicts its answer from the attributes; summed top-"
                   "per_model importances rank the attributes; system shows the top k as key: value lines.")
    llm_free = True

    def __init__(self, k: int = 5, per_model: int = 10, n_estimators: int = 200, **params: Any):
        super().__init__(k=k, per_model=per_model, n_estimators=n_estimators, **params)
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k, self.per_model, self.n_estimators = int(k), int(per_model), int(n_estimators)

    def estimate_calls(self, n_attrs, n_train, n_val):
        return {"calls": 0, "optimizer_calls": 0}

    def _search(self, ctx: _Ctx, **kw) -> FitResult:
        weights = forest_importance(ctx.cells["train"], ctx.attrs, self.person_fields, per_model=self.per_model,
                                    n_estimators=self.n_estimators, seed=self.seed)
        rank = sorted(ctx.attrs, key=lambda a: (-weights[a], ctx.attrs.index(a)))
        keys = [a for a in rank if weights[a] > 0][:self.k]
        if not keys:
            raise ValueError("fitted_topk: no fit question has two or more distinct answers to learn from")
        # explicit val_questions are unused here (no LLM to validate) but stay refused like every fit question
        res = FitResult(method=self.name, config={"keys": keys}, train_questions=list(ctx.train_q),
                        val_questions=list(ctx.val_q), train_score=None, val_score=None,
                        meta={"weights": {a: round(weights[a], 6) for a in rank}})
        reg = kw.get("registry_dir")
        if reg is not None:
            from ...research.prereg import seal
            res.seal = seal({"method": self.name, "keys": keys, "weights": res.meta["weights"]},
                            f"persona.inject {self.name}: fitted config before held-out questions", reg)
        return res


def forest_importance(cells: Sequence[dict], attrs: Sequence[str], fields_of: Callable[[dict], dict], *,
                      per_model: int = 10, n_estimators: int = 200, seed: int = 0) -> dict[str, float]:
    """{attribute: summed importance}: one RandomForestClassifier per question over the cells' people, numeric
    attributes as numbers (missing -> median), others one-hot (missing is its own level)."""
    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as e:  # pragma: no cover - core deps
        raise ImportError("fitted_topk needs numpy, pandas and scikit-learn (pip install scikit-learn)") from e

    by_q: dict[str, list[dict]] = {}
    for c in cells:
        by_q.setdefault(str(c["question"]["id"]), []).append(c)
    total = {a: 0.0 for a in attrs}
    for qid, cs in by_q.items():
        y = [_key(c["truth"]) for c in cs]
        if len(set(y)) < 2:
            continue
        rows = [fields_of(c["person"]) for c in cs]
        cols, owner = [], []
        drop = hidden(cs[0]["question"])                 # the target is never its own predictor
        for a in attrs:
            if str(a) in drop:
                continue
            vals = [r.get(a) for r in rows]
            nums = [v for v in vals if v is not None and not isinstance(v, bool) and isinstance(v, (int, float))]
            if nums and len(nums) == sum(v is not None for v in vals):
                med = float(np.median(nums))
                cols.append(pd.Series([float(v) if v is not None else med for v in vals], name=a))
                owner.append(a)
            else:
                d = pd.get_dummies(pd.Series(["<missing>" if v is None else str(v) for v in vals]), prefix=a)
                for c in d.columns:
                    cols.append(d[c].astype(float))
                    owner.append(a)
        if not cols:
            continue
        X = pd.concat(cols, axis=1).to_numpy(float)
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed).fit(X, y)
        imp: dict[str, float] = {}
        for o, v in zip(owner, rf.feature_importances_):
            imp[o] = imp.get(o, 0.0) + float(v)
        for a in sorted(imp, key=lambda a: (-imp[a], attrs.index(a)))[:per_model]:
            total[a] += imp[a]
    return total


@register
class FittedStepwise(_Fitted):
    """Coordinate ascent over attribute inclusion run by simengine.research.Loop: a binary Space (one key per
    attribute), GreedyCoordinateProposer flips one attribute per turn, a flip is kept only when the paired
    bootstrap CI over train cells excludes 0 (or ties with fewer attributes). Stops when no unseen flip is left or
    after max_turns. The kept path is re-ranked on the validation questions."""
    name = "fitted_stepwise"
    paper = ("stepwise / wrapper attribute selection (Efroymson 1960; Kohavi & John 1997; no arXiv id) with "
             "simengine.research.Loop acceptance; attribute-selection question: Fröhling et al. 2026, arXiv:2609.02526")
    description = ("Coordinate ascent over which attributes to show (one flip per turn, kept if the paired CI on "
                   "train cells excludes 0); validation re-rank; system shows the chosen key: value lines.")

    def __init__(self, start: str = "empty", max_turns: int | None = None, alpha: float = 0.05,
                 simpler_ties: bool = True, n_boot: int = 1000, **params: Any):
        if start not in ("empty", "all"):
            raise ValueError("start must be 'empty' or 'all'")
        super().__init__(start=start, **({"max_turns": max_turns} if max_turns is not None else {}), alpha=alpha,
                         simpler_ties=simpler_ties, n_boot=n_boot, **params)
        self.start, self.max_turns, self.alpha = start, max_turns, float(alpha)
        self.simpler_ties, self.n_boot = bool(simpler_ties), int(n_boot)

    def _turns(self, n_attrs: int) -> int:
        return self.max_turns if self.max_turns is not None else 3 * n_attrs

    def estimate_calls(self, n_attrs, n_train, n_val):
        return {"calls": (self._turns(n_attrs) + 1) * n_train + self.top_k * n_val, "optimizer_calls": 0}

    def _search(self, ctx: _Ctx, registry_dir=None, prereg=None, **kw) -> FitResult:
        from ...research.loop import Loop
        from ...research.proposers import GreedyCoordinateProposer
        from ...research.space import Space

        slot = {f"a{i}": a for i, a in enumerate(ctx.attrs)}     # Space keys never clash with Loop's own keys
        space = Space({s: [False, True] for s in slot}, complexity=lambda c: sum(bool(c[s]) for s in slot))
        on = self.start == "all"
        split_map = {"dev": "train", "select": "val"}

        def text(cfg):
            return self.text_of([slot[s] for s in slot if cfg.get(s)])

        def runner(cfg):
            r = ctx.ev.scores(text(cfg), split_map[cfg["split"]])
            units = ctx.cells[split_map[cfg["split"]]]
            return {"per_unit": {u["unit"]: s for u, s in zip(units, r["scores"])}}

        loop = Loop(runner, space, GreedyCoordinateProposer(), start={s: on for s in slot}, primary="score",
                    higher_is_better=True, stop_after=None, max_turns=self._turns(len(slot)), alpha=self.alpha,
                    workdir=None, select_split=None, test_split="never", simpler_ties=self.simpler_ties,
                    ablation=False, seed=self.seed, n_boot=self.n_boot)
        out = loop.run(finalize=False)
        kept = [t for t in out["history"] if t["status"] == "keep"]
        hist = [ctx.ev.candidate(text(t["cfg"]), self.name, "start" if t["n"] == 0 else "keep", step=t["n"])
                for t in kept]
        res = ctx.finish(hist, self.name, registry_dir, prereg,
                         meta={"stopped": out["stopped"], "turns": out["turns"], "kept": out["kept"]})
        res.history = [{"text": text(t["cfg"]), "train": (t.get("metrics") or {}).get("score", {}).get("value"),
                        "origin": t["status"], "step": t["n"]} for t in out["history"]]
        return res


@register
class FittedGrips(_Fitted):
    """GrIPS (Prasad et al. 2022, Alg. 1) with attribute lines as the phrases: each iteration samples m candidates,
    each with l edits from {delete a line (kept for later), swap two lines, add back a deleted line at a random
    position}; the best candidate replaces the base only if it scores higher on train cells; patience P.
    Paraphrase (PEGASUS in the paper) has no meaning for `key: value` lines and is left out; the score is cell
    accuracy, not the paper's balanced accuracy + 10 x answer entropy."""
    name = "fitted_grips"
    paper = "Prasad et al. 2022, GrIPS, arXiv:2203.07281"
    description = ("Edit search over the attribute lines (delete / swap / add-back; m candidates per iteration, "
                   "patience P, accept only a higher train score); validation re-rank of the visited configs.")
    OPS = ("del", "swap", "add")

    def __init__(self, m: int = 5, l: int = 1, n_iter: int = 10, patience: int = 2, start: str = "all",
                 **params: Any):
        if start not in ("empty", "all"):
            raise ValueError("start must be 'empty' or 'all'")
        super().__init__(m=m, l=l, n_iter=n_iter, patience=patience, start=start, **params)
        self.m, self.l, self.n_iter, self.patience, self.start = int(m), int(l), int(n_iter), int(patience), start

    def estimate_calls(self, n_attrs, n_train, n_val):
        return {"calls": (1 + self.n_iter * self.m) * n_train + self.top_k * n_val, "optimizer_calls": 0}

    @classmethod
    def edit(cls, keys: list[str], pool: list[str], rng: random.Random) -> tuple[list[str], list[str], str] | None:
        """One random valid edit of (keys, deleted pool) -> (keys, pool, op), or None when nothing applies."""
        ops = [o for o in cls.OPS if (o == "del" and keys) or (o == "swap" and len(keys) >= 2) or (o == "add" and pool)]
        if not ops:
            return None
        op = rng.choice(ops)
        keys, pool = keys[:], pool[:]
        if op == "del":
            pool.append(keys.pop(rng.randrange(len(keys))))
        elif op == "swap":
            i, j = rng.sample(range(len(keys)), 2)
            keys[i], keys[j] = keys[j], keys[i]
        else:
            keys.insert(rng.randrange(len(keys) + 1), pool.pop(rng.randrange(len(pool))))
        return keys, pool, op

    def _search(self, ctx: _Ctx, registry_dir=None, prereg=None, **kw) -> FitResult:
        rng = random.Random(self.seed)
        base = list(ctx.attrs) if self.start == "all" else []
        pool = [] if self.start == "all" else list(ctx.attrs)
        base_c = ctx.ev.candidate(self.text_of(base), self.name, "start")
        hist, patience, accepted = [base_c], self.patience, 0
        for it in range(1, self.n_iter + 1):
            cands = []
            for _ in range(self.m):
                k, p, ops = base, pool, []
                for _ in range(self.l):
                    e = self.edit(k, p, rng)
                    if e is None:
                        break
                    k, p, op = e
                    ops.append(op)
                if ops:
                    c = ctx.ev.candidate(self.text_of(k), self.name, "+".join(ops), step=it)
                    hist.append(c)
                    cands.append((c, p))
            if not cands:
                break
            best, best_pool = max(cands, key=lambda cp: cp[0].fitness)
            if best.fitness > base_c.fitness:
                base, pool, base_c, patience = self.cfg_of(best.text)["keys"], best_pool, best, self.patience
                accepted += 1
            else:
                patience -= 1
                if patience <= 0:
                    break
        return ctx.finish(hist, self.name, registry_dir, prereg,
                          meta={"iterations": it if self.n_iter else 0, "accepted": accepted,
                                "base": self.cfg_of(base_c.text)["keys"]})


@register
class FittedOpro(_Fitted):
    """OPRO over the framing sentence above the attribute lines: simengine.research.prompt_evolution.opro (meta-
    prompt = best texts so far with integer train scores, ascending, + n_exemplars train cells with <INS>; per_step
    samples per step), then the validation re-rank. Attribute lines (all, or `keys`) and the closing line are fixed."""
    name = "fitted_opro"
    paper = "Yang et al. 2023, OPRO, arXiv:2309.03409"
    description = ("OPRO-optimised framing sentence before the key: value lines (proposer LLM sees train scores "
                   "and train exemplars only); validation re-rank.")

    def __init__(self, steps: int = 10, per_step: int = 8, keep_top: int = 20, n_exemplars: int = 3,
                 seeds: Sequence[str] = (HEADER,), **params: Any):
        super().__init__(steps=steps, per_step=per_step, keep_top=keep_top, n_exemplars=n_exemplars,
                         **({"seeds": list(seeds)} if tuple(seeds) != (HEADER,) else {}), **params)
        if not seeds:
            raise ValueError("fitted_opro needs at least one seed text")
        self.steps, self.per_step, self.keep_top, self.n_exemplars = int(steps), int(per_step), int(keep_top), int(n_exemplars)
        self.seeds = list(seeds)
        self._attrs: list[str] = []

    def cfg_of(self, text: str) -> dict:
        return {"instruction": text, "keys": list(self._attrs)}

    def estimate_calls(self, n_attrs, n_train, n_val):
        n_opt = self.steps * self.per_step
        return {"calls": (len(self.seeds) + n_opt) * n_train + self.top_k * n_val, "optimizer_calls": n_opt}

    def _search(self, ctx: _Ctx, optimizer=None, registry_dir=None, prereg=None, **kw) -> FitResult:
        from ...research.prompt_evolution import opro

        self._attrs = list(ctx.attrs)
        for u in ctx.cells["train"]:          # OPRO exemplars: persona lines + question -> real answer (train only)
            u.setdefault("input", self.system_for(u["person"], {"instruction": "", "keys": ctx.attrs},
                                                  u["question"]).strip()
                         + "\n" + u["question"]["text"])
            u.setdefault("output", _key(u["truth"]))
        task = ("Each text is the opening sentence of a system prompt that describes a survey respondent by "
                "attribute lines; the model then answers survey questions as that person.")
        s = opro(optimizer or ctx.llm, ctx.ev, self.seeds, steps=self.steps, per_step=self.per_step,
                 keep_top=self.keep_top, n_exemplars=self.n_exemplars, position="q_begin", task=task, seed=self.seed)
        return ctx.finish(s["history"], self.name, registry_dir, prereg, optimizer_calls=s["llm_calls"],
                          meta={"duplicates": s["duplicates"], "unparsed": s["unparsed"], "cost_usd": s["cost_usd"]})


__all__ = ["FittedTopK", "FittedStepwise", "FittedGrips", "FittedOpro", "FitResult", "NotFitted", "HeldOutViolation",
           "split_questions", "forest_importance", "hidden", "HEADER", "TAIL"]
