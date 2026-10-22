"""LLM-driven prompt / persona-text optimisation with a train / validation / held-out test protocol.

Three search methods over a text (an instruction, a system prompt or a persona description), each scored by a fixed,
user-supplied scorer on per-unit data:

- `opro`: OPRO (Yang et al. 2023, arXiv:2309.03409). Meta-prompt = the best `keep_top` texts so far with their
  training scores in ascending order (best last), a few random training exemplars with `<INS>` at the insertion
  point, and the meta-instruction "Write your new text that is different from the old ones and has a score as high
  as possible. Write the text in square brackets." `per_step` samples per step; scores rounded to integers 0-100.
- `ape`: APE (Zhou et al. 2022, arXiv:2211.01910). Forward-mode generation template from demonstrations, scoring on a
  random training subset, keep the top `top_frac`, optional iterative Monte Carlo resampling ("Generate a variation
  of the following instruction while keeping the semantic meaning."), survivors scored on the full training split.
- `promptbreeder`: Promptbreeder-style binary-tournament GA (Fernando et al. 2023, arXiv:2309.16797). Units carry a
  task-prompt and a mutation-prompt; nine operators in five classes (zero/first-order generation, EDA, EDA rank and
  index, lineage, zero/first-order hyper-mutation, Lamarckian working-out-to-prompt) plus 10% prompt crossover.

Search sees the training split only. `select_and_test` re-ranks the top-k texts on the validation split, seals the
chosen text with `simengine.research.seal` (and checks a locked prereg if given), then scores the winner and the
baselines ONCE on the test split with paired bootstrap CIs. `evolve` runs search + selection in one call.

    import simengine.research.prompt_evolution as pe
    tr, va, te = pe.split_units(df, frac=(0.5, 0.25, 0.25), seed=0)          # rows with "input", "output"
    score = pe.make_llm_scorer(scorer_llm)                                  # system = candidate text, exact match
    ev = pe.PromptEvaluator(score, tr, va, te)
    res = pe.evolve("opro", optimizer_llm, ev, seeds=["Answer the question."], steps=10,
                    registry_dir="runs/pe/.prereg")
    res.best.text, res.test["winner"]["mean"], res.test_diff["winner-seed0"]

Scorer contract: `score_fn(text, units) -> array of per-unit scores` (or `{"scores": array, "outputs": [str]}`;
outputs feed the Lamarckian operator). `units` is a DataFrame slice or a list. Scores are higher-is-better unless
`PromptEvaluator(higher_is_better=False)`. For persona text, let the scorer put the text in the system prompt and
score each survey item against the human answer (e.g. minus the absolute error).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ..types import LLM
from .prereg import seal, verify_prereg
from .stats import bootstrap_mean, is_better, paired_bootstrap_diff

SYSTEM = "Follow the request in the user message. Reply with the requested text only."

# OPRO Fig. 3 (GSM8K meta-prompt), generalised to any exemplar format. Lower-is-better follows Fig. 19-20: the
# list is still worst-to-best, so the raw scores run in descending order and the tail asks for a lower score.
OPRO_HEAD = ("I have some texts along with their corresponding scores. The texts are arranged in {order} order "
             "based on their scores, where {better} scores indicate better quality.")
OPRO_HEAD_NOSCORE = "I have some texts. The texts are arranged in ascending order of quality."
OPRO_EXEMPLARS = ("The following exemplars show how to apply your text: you replace {tag} in each input with your "
                  "text, then read the input and give an output. We say your output is wrong if your output is "
                  "different from the given output, and we say your output is correct if they are the same.")
OPRO_TAIL = ("Write your new text that is different from the old ones and has a score as {high} as possible. "
             "Write the text in square brackets.")

# APE Fig. 2 (top) and Fig. 3 / Appendix templates.
APE_FORWARD = ("I gave a friend an instruction and {n} inputs. The friend read the instruction and wrote an output "
               "for every one of the inputs.\nHere are the input-output pairs:\n\n{pairs}\n\nThe instruction was")
APE_RESAMPLE = ("Generate a variation of the following instruction while keeping the semantic meaning.\n\n"
                "Input: {instruction}\nOutput:")

# Promptbreeder §3 strings.
PB_HINTS = "A list of 100 hints:"
PB_HYPER = "Please summarize and improve the following instruction:"
PB_LAMARCK = ("I gave a friend an instruction and some advice. Here are the correct examples of his workings out:\n"
              "{workings}\nThe instruction was:")
PB_LINEAGE = "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY"
# Short subsets quoted from Promptbreeder Appendix C (mutation-prompts) and D (thinking-styles); pass your own lists.
MUTATION_PROMPTS = (
    "Modify the following instruction creatively, giving some advice on how to solve it:",
    "Just change this instruction to make it more fun, think WELL outside the box:",
    "How would you help an LLM to follow the instruction?",
    "Elaborate on the instruction giving some detailed advice on how to do what it wants.",
    "Say that instruction again in another way. DON'T use any of the words in the original instruction there's a "
    "good chap.",
    "Make a variant of the prompt.",
)
THINKING_STYLES = (
    "Let's think step by step.",
    "How could I devise an experiment to help solve that problem?",
    "How can I simplify the problem so that it is easier to solve?",
    "What is the core issue or problem that needs to be addressed?",
    "Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem.",
    "Let's make a step by step plan and implement it with good notion and explanation.",
)
PB_OPERATORS = ("zero_order", "first_order", "eda", "eda_rank", "lineage", "zero_order_hyper", "first_order_hyper",
                "lamarckian")

_NUM = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


# ===================================================================== data, scoring

def split_units(units, frac=(0.5, 0.25, 0.25), seed: int = 0, group: str | None = None):
    """Random train / validation / test split of a DataFrame or list (group-aware for a DataFrame `group` column)."""
    f = np.asarray(frac, float)
    if len(f) != 3 or (f < 0).any() or f.sum() <= 0:
        raise ValueError("frac must be three non-negative numbers")
    f = f / f.sum()
    rng = np.random.default_rng(seed)
    n = len(units)
    if group is not None:
        keys = pd.Index(pd.unique(units[group]))
        perm = rng.permutation(len(keys))
        cut = np.round(np.cumsum(f)[:2] * len(keys)).astype(int)
        lab = np.empty(len(keys), int)
        lab[perm[:cut[0]]], lab[perm[cut[0]:cut[1]]], lab[perm[cut[1]:]] = 0, 1, 2
        row = lab[keys.get_indexer(units[group])]
    else:
        perm = rng.permutation(n)
        cut = np.round(np.cumsum(f)[:2] * n).astype(int)
        row = np.empty(n, int)
        row[perm[:cut[0]]], row[perm[cut[0]:cut[1]]], row[perm[cut[1]:]] = 0, 1, 2
    return tuple(_take(units, np.flatnonzero(row == k)) for k in range(3))


def _take(units, idx):
    if isinstance(units, pd.DataFrame):
        return units.iloc[np.asarray(idx, int)]
    return [units[int(i)] for i in idx]


def _records(units) -> list:
    return units.to_dict("records") if isinstance(units, pd.DataFrame) else list(units)


def normalise_answer(s) -> str:
    return re.sub(r"[\s\.\!\?\"']+$", "", str(s).strip()).strip().casefold()


def exact_match(output: str, target) -> float:
    return float(normalise_answer(output) == normalise_answer(target))


def make_llm_scorer(llm: LLM, *, input_key: str = "input", target_key: str = "output",
                    position: str = "system", render: Callable[[str, dict], tuple[str, str]] | None = None,
                    parse: Callable[[str], Any] | None = None, metric: Callable[[Any, Any], float] = exact_match,
                    max_tokens: int = 64, parallel: int = 4, **complete_kw) -> Callable:
    """Scorer that runs the candidate text on every unit with `llm` and grades the reply.

    position: "system" (text = system prompt, input = user), "q_begin" / "q_end" (OPRO §4.1: text before / after
    the input in the user message). `render(text, unit) -> (system, user)` overrides it. Returns
    {"scores", "outputs"}; a failed call scores NaN."""
    if position not in ("system", "q_begin", "q_end"):
        raise ValueError("position must be system, q_begin or q_end")

    def default_render(text, u):
        x = str(u[input_key])
        if position == "system":
            return text, x
        return "", (f"{text}\n{x}" if position == "q_begin" else f"{x}\n{text}")

    rd = render or default_render

    def score(text: str, units) -> dict:
        recs = _records(units)
        res = llm.map([rd(text, u) for u in recs], max_tokens=max_tokens, parallel=parallel,
                      **({"return_exceptions": True} if _accepts(llm.map, "return_exceptions") else {}),
                      **complete_kw)
        outs, sc = [], []
        for u, r in zip(recs, res):
            if isinstance(r, Exception):
                outs.append(""), sc.append(np.nan)
                continue
            t = r.text or ""
            outs.append(t)
            try:
                sc.append(float(metric(parse(t) if parse else t, u[target_key])))
            except (ValueError, TypeError, KeyError):          # unparseable reply: NaN, left out of means
                sc.append(np.nan)
        return {"scores": np.asarray(sc, float), "outputs": outs}
    return score


def _accepts(fn, name) -> bool:
    import inspect
    try:
        p = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in p


@dataclass
class Candidate:
    text: str
    train: float                  # mean score on the units it was scored on (raw direction)
    fitness: float                # train if higher_is_better else -train
    n_units: int
    full_train: bool              # scored on the whole training split
    method: str
    origin: str                   # seed | opro | ape-forward | ape-resample | pb-<operator> | pb-init
    step: int = 0
    parent: str | None = None
    meta: dict = field(default_factory=dict)


class PromptEvaluator:
    """Caches per-unit scores per (text, split, subset). Search may read "train" (and subsets of it), selection
    reads "val"; "test" is refused except through `select_and_test` (once)."""

    def __init__(self, score_fn: Callable, train, val, test=None, *, higher_is_better: bool = True):
        if len(train) == 0 or len(val) == 0:
            raise ValueError("train and val must be non-empty")
        self.score_fn, self.hib = score_fn, higher_is_better
        self.splits = {"train": train, "val": val, "test": test}
        self._cache: dict = {}
        self.unit_evals = {"train": 0, "val": 0, "test": 0}
        self.test_done = False

    def n(self, split: str) -> int:
        s = self.splits[split]
        return 0 if s is None else len(s)

    def fitness(self, mean: float) -> float:
        if mean is None or not np.isfinite(mean):
            return -math.inf
        return mean if self.hib else -mean

    def scores(self, text: str, split: str = "train", idx=None, *, _allow_test: bool = False) -> dict:
        """{"scores": per-unit array, "outputs": list | None, "mean": nan-mean} for `text` on split[idx]."""
        if split == "test" and not _allow_test:
            raise PermissionError("the test split is scored once, by select_and_test()")
        units = self.splits[split]
        if units is None:
            raise ValueError(f"no {split} split")
        if idx is not None:
            idx = tuple(int(i) for i in idx)
            full = self._cache.get((text, split, None))
            if full is not None:                      # subset of a full run: no new call
                s = full["scores"][list(idx)]
                o = None if full["outputs"] is None else [full["outputs"][i] for i in idx]
                return {"scores": s, "outputs": o, "mean": _nanmean(s)}
        key = (text, split, idx)
        if key not in self._cache:
            sub = units if idx is None else _take(units, idx)
            r = self.score_fn(text, sub)
            if isinstance(r, dict):
                s, o = np.asarray(r["scores"], float), r.get("outputs")
            else:
                s, o = np.asarray(r, float), None
            if s.shape != (len(sub),):
                raise ValueError(f"score_fn returned {s.shape} scores for {len(sub)} units")
            self.unit_evals[split] += len(sub)
            self._cache[key] = {"scores": s, "outputs": None if o is None else list(o), "mean": _nanmean(s)}
        return self._cache[key]

    def cached(self, text: str, split: str = "train") -> dict | None:
        """The cached run of `text` on `split` that covers the most units (None if never scored)."""
        hits = [v for (t, sp, _), v in self._cache.items() if t == text and sp == split]
        return max(hits, key=lambda v: len(v["scores"]), default=None)

    def best_score(self, split: str = "train") -> float:
        """Best finite per-unit score cached on `split` over all texts (raw direction; nan if none)."""
        v = [x["scores"][np.isfinite(x["scores"])] for (_, sp, _), x in self._cache.items() if sp == split]
        v = np.concatenate(v) if v else np.empty(0)
        return float("nan") if not len(v) else float(v.max() if self.hib else v.min())

    def candidate(self, text: str, method: str, origin: str, *, idx=None, step: int = 0, parent=None,
                  **meta) -> Candidate:
        r = self.scores(text, "train", idx)
        n = len(r["scores"])
        return Candidate(text, r["mean"], self.fitness(r["mean"]), n, idx is None or n == self.n("train"),
                         method, origin, step, parent, meta)


def _nanmean(s) -> float:
    s = np.asarray(s, float)
    return float(np.nanmean(s)) if np.isfinite(s).any() else float("nan")


# ===================================================================== LLM helpers

class _Calls:
    """Counts optimizer LLM calls and cost; gives every call a distinct `sample` index (independent cache entries)."""

    def __init__(self, llm: LLM, system: str, max_tokens: int, parallel: int, complete_kw: dict):
        self.llm, self.system, self.max_tokens, self.parallel, self.kw = llm, system, max_tokens, parallel, complete_kw
        self.n, self.cost, self._sample = 0, 0.0, 0

    def many(self, users: Sequence[str]) -> list[str]:
        from ..llm.parallel import parallel_map
        base = self._sample
        self._sample += len(users)
        res = parallel_map(lambda iu: self.llm.complete(self.system, iu[1], max_tokens=self.max_tokens,
                                                        sample=base + iu[0], **self.kw),
                           list(enumerate(users)), parallel=self.parallel)
        self.n += len(res)
        self.cost += float(sum(float(getattr(r, "cost_usd", 0.0) or 0.0) for r in res))
        return [r.text or "" for r in res]

    def one(self, user: str) -> str:
        return self.many([user])[0]


_LEAD = re.compile(r"^\s*(?:[-*•]|\(?\d+[\.\):]|INSTRUCTION MUTANT\s*[:=]?|INSTRUCTION\s*[:=]|"
                   r"(?:the )?instruction (?:was|is)\s*:?|output\s*:)\s*", re.I)


def clean_reply(text: str, max_chars: int = 2000) -> str:
    """First non-empty paragraph, numbering / control labels / enclosing quotes or brackets stripped."""
    t = (text or "").strip()
    for para in re.split(r"\n\s*\n", t):
        p = para.strip()
        for _ in range(3):
            p = _LEAD.sub("", p).strip()
        if len(p) >= 2 and p[0] + p[-1] in ('""', "''", "[]", "``", "<>"):
            p = p[1:-1].strip()
        if p:
            return p[:max_chars]
    return ""


def first_line(text: str, max_chars: int = 2000) -> str:
    for line in (text or "").splitlines():
        p = clean_reply(line, max_chars)
        if p:
            return p
    return ""


def parse_square_brackets(text: str) -> str:
    """OPRO's output format: the first [...] span (empty string if none)."""
    m = re.search(r"\[(.*?)\]", text or "", re.S)
    return m.group(1).strip() if m else ""


def _exemplar(u: dict, input_key: str, target_key: str) -> tuple[str, str] | None:
    try:
        return str(u[input_key]), str(u[target_key])
    except (KeyError, TypeError, IndexError):
        return None


# ===================================================================== OPRO

def opro_meta_prompt(history: Sequence[Candidate], exemplars: Sequence[tuple[str, str]], *, keep_top: int = 20,
                     score_scale: float | None = 100, higher_is_better: bool = True, show_scores: bool = True,
                     tag: str = "<INS>", position: str = "q_begin", task: str = "") -> str:
    """OPRO meta-prompt (Fig. 3): best `keep_top` texts ascending by quality, exemplars, meta-instruction."""
    best = {}
    for c in history:
        if c.text and (c.text not in best or c.fitness > best[c.text].fitness):
            best[c.text] = c
    top = sorted(best.values(), key=lambda c: c.fitness)[-keep_top:] if keep_top else []

    def fmt(v):
        if not np.isfinite(v):
            return "nan"
        return str(int(round(v * score_scale))) if score_scale else f"{v:.4g}"
    L = [task] if task else []
    hib = higher_is_better
    L.append(OPRO_HEAD.format(order="ascending" if hib else "descending", better="higher" if hib else "lower")
             if show_scores else OPRO_HEAD_NOSCORE)
    for c in top:
        L.append(f"\ntext:\n{c.text}" + (f"\nscore:\n{fmt(c.train)}" if show_scores else ""))
    if exemplars:
        L.append("\n" + OPRO_EXEMPLARS.format(tag=tag))
        for x, y in exemplars:
            inp = f"{tag}\n{x}" if position == "q_begin" else f"{x}\n{tag}"
            L.append(f"\ninput:\n{inp}\noutput:\n{y}")
    L.append("\n" + OPRO_TAIL.format(high="high" if hib else "low"))
    return "\n".join(L)


def opro(llm: LLM, ev: PromptEvaluator, seeds: Sequence[str], *, steps: int = 10, per_step: int = 8,
         keep_top: int = 20, n_exemplars: int = 3, score_scale: float | None = 100, show_scores: bool = True,
         input_key: str = "input", target_key: str = "output", position: str = "q_begin", task: str = "",
         seed: int = 0, max_tokens: int = 400, parallel: int = 4, system: str = SYSTEM,
         **complete_kw) -> dict:
    """OPRO search on the training split. Defaults follow the paper (8 samples/step, top 20 kept, 3 exemplars sampled
    each step, integer scores); the paper runs ~200 steps, the default here is 10.
    Returns {"history": [Candidate], "llm_calls", "cost_usd", "duplicates", "unparsed"}."""
    rng = np.random.default_rng(seed)
    calls = _Calls(llm, system, max_tokens, parallel, complete_kw)
    hist = [ev.candidate(s, "opro", "seed") for s in dict.fromkeys(seeds)]
    seen = {c.text for c in hist}
    train = _records(ev.splits["train"])
    dup = unparsed = 0
    for step in range(1, steps + 1):
        ex = []
        if n_exemplars and train:
            for i in rng.choice(len(train), min(n_exemplars, len(train)), replace=False):
                e = _exemplar(train[int(i)], input_key, target_key)
                if e:
                    ex.append(e)
        meta = opro_meta_prompt(hist, ex, keep_top=keep_top, score_scale=score_scale, higher_is_better=ev.hib,
                                show_scores=show_scores, position=position, task=task)
        for raw in calls.many([meta] * per_step):
            t = parse_square_brackets(raw)
            if not t:
                unparsed += 1
                continue
            if t in seen:
                dup += 1
                continue
            seen.add(t)
            hist.append(ev.candidate(t, "opro", "opro", step=step))
    return {"history": hist, "llm_calls": calls.n, "cost_usd": calls.cost, "duplicates": dup, "unparsed": unparsed}


# ===================================================================== APE

def ape_forward_prompt(demos: Sequence[tuple[str, str]]) -> str:
    n = len(demos)
    pairs = "\n\n".join(f"Input: {x}\nOutput: {y}" for x, y in demos)
    return APE_FORWARD.format(n=_NUM[n] if n < len(_NUM) else n, pairs=pairs)


def ape(llm: LLM, ev: PromptEvaluator, *, n_candidates: int = 50, n_demos: int = 5, subset_size: int | None = 20,
        top_frac: float = 0.1, resample_rounds: int = 0, per_survivor: int = 4, seeds: Sequence[str] = (),
        input_key: str = "input", target_key: str = "output", seed: int = 0, max_tokens: int = 200,
        parallel: int = 4, system: str = SYSTEM, **complete_kw) -> dict:
    """APE (Algorithm 1): forward-mode proposals (each from its own random `n_demos` training demonstrations), score
    on a random training subset, keep the top `top_frac`, optionally resample variations of the survivors for
    `resample_rounds` rounds (iterative APE, new random subset each round), survivors scored on the full training
    split. The paper samples 50 candidates by default and uses plain APE (no resampling) by default."""
    rng = np.random.default_rng(seed)
    calls = _Calls(llm, system, max_tokens, parallel, complete_kw)
    train = _records(ev.splits["train"])
    n_tr = len(train)
    pool = [e for e in (_exemplar(u, input_key, target_key) for u in train) if e]
    if not pool and not seeds:
        raise ValueError(f"APE needs training units with {input_key!r} and {target_key!r}")
    users = []
    for _ in range(n_candidates if pool else 0):
        k = min(n_demos, len(pool))
        users.append(ape_forward_prompt([pool[int(i)] for i in rng.choice(len(pool), k, replace=False)]))
    texts = list(dict.fromkeys([*seeds, *(clean_reply(t) for t in calls.many(users))]))
    texts = [t for t in texts if t]
    origin = {t: ("seed" if t in seeds else "ape-forward") for t in texts}
    hist: list[Candidate] = []

    def stage(cands, rnd):
        idx = None
        if subset_size and subset_size < n_tr:
            idx = np.sort(rng.choice(n_tr, subset_size, replace=False))
        scored = [ev.candidate(t, "ape", origin[t], idx=idx, step=rnd) for t in cands]
        hist.extend(scored)
        scored.sort(key=lambda c: -c.fitness)
        return [c.text for c in scored[:max(1, math.ceil(top_frac * len(scored)))]]

    surv = stage(texts, 0)
    for rnd in range(1, resample_rounds + 1):
        outs = calls.many([APE_RESAMPLE.format(instruction=t) for t in surv for _ in range(per_survivor)])
        new = [t for t in dict.fromkeys(clean_reply(o) for o in outs) if t and t not in origin]
        for t in new:
            origin[t] = "ape-resample"
        surv = stage(list(dict.fromkeys([*surv, *new])), rnd)
    for t in surv:
        hist.append(ev.candidate(t, "ape", origin[t], step=resample_rounds + 1, survivor=True))
    return {"history": hist, "llm_calls": calls.n, "cost_usd": calls.cost, "survivors": surv}


# ===================================================================== Promptbreeder

def diversity_filter(texts: Sequence[str], threshold: float = 0.95) -> list[int]:
    """Indices of texts kept when a text more than `threshold` cosine-similar to an earlier kept one is dropped.
    Character 2-4-gram TF-IDF stands in for the paper's BERT embeddings."""
    if len(texts) < 2:
        return list(range(len(texts)))
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform([t or " " for t in texts])
    S = (X @ X.T).toarray()
    keep: list[int] = []
    for i in range(len(texts)):
        if not keep or S[i, keep].max() <= threshold:
            keep.append(i)
    return keep


@dataclass
class _Unit:
    task: str
    mutation: str
    fitness: float
    train: float
    lineage: list = field(default_factory=list)
    uid: int = 0


def promptbreeder(llm: LLM, ev: PromptEvaluator, problem: str, *, population: int = 20, generations: int = 10,
                  seeds: Sequence[str] = (), mutation_prompts: Sequence[str] = MUTATION_PROMPTS,
                  thinking_styles: Sequence[str] = THINKING_STYLES, operators: Sequence[str] = PB_OPERATORS,
                  batch_size: int | None = None, crossover_p: float = 0.1, diversity_threshold: float = 0.95,
                  correct_at: float | None = None, n_workings: int = 2, seed: int = 0, max_tokens: int = 300,
                  parallel: int = 4, system: str = SYSTEM, **complete_kw) -> dict:
    """Promptbreeder-style binary tournament GA on the training split.

    One generation pairs the whole population at random (paper §4; an odd unit sits out) and runs one tournament
    per pair: mutate a copy of the fitter unit with an operator drawn uniformly from `operators`, apply prompt
    crossover with prob. `crossover_p` (task-prompt replaced by one of
    another unit chosen fitness-proportionately), score it and overwrite the loser. Fitness = mean score on a fresh
    random batch of `batch_size` training units (paper: 100), or on the full training split if None (deterministic).
    Lamarckian mutation reads the scorer outputs of the parent's cached evaluation (no new scoring); units scoring
    at least `correct_at` (at most, if lower is better; default: the best per-unit score cached on train over all
    texts, i.e. 1 for exact match) count as correct workings. Without outputs or correct units it falls back to
    first-order generation.
    Differences from the paper: one task-prompt per unit (the paper evolves two in sequence), no few-shot context
    shuffling, TF-IDF instead of BERT for the 0.95 diversity filter. Paper defaults: population 50, 20-30 generations."""
    if population < 2:
        raise ValueError("population must be >= 2")
    bad = set(operators) - set(PB_OPERATORS)
    if bad:
        raise ValueError(f"unknown operators {sorted(bad)}")
    rng = np.random.default_rng(seed)
    calls = _Calls(llm, system, max_tokens, parallel, complete_kw)
    n_tr = ev.n("train")
    hist: list[Candidate] = []
    op_count = {o: 0 for o in operators}
    fallbacks = 0
    uid = [0]

    def pick(seq):
        return seq[int(rng.integers(len(seq)))]

    def score(text, origin, step, parent=None, **meta) -> Candidate:
        idx = None if not batch_size or batch_size >= n_tr else np.sort(rng.choice(n_tr, batch_size, replace=False))
        c = ev.candidate(text, "promptbreeder", origin, idx=idx, step=step, parent=parent, **meta)
        hist.append(c)
        return c

    def unit(task, mut, c: Candidate, lineage=None) -> _Unit:
        uid[0] += 1
        return _Unit(task, mut, c.fitness, c.train, list(lineage or []), uid[0])

    # initialisation: mutation-prompt + thinking-style + "INSTRUCTION: <problem>" + "INSTRUCTION MUTANT:"
    init = [(pick(mutation_prompts), pick(thinking_styles)) for _ in range(max(0, population - len(seeds)))]
    outs = calls.many([f"{m} {t} INSTRUCTION: {problem} INSTRUCTION MUTANT:" for m, t in init])
    pop: list[_Unit] = []
    for s in seeds[:population]:
        pop.append(unit(s, pick(mutation_prompts), score(s, "seed", 0)))
    for (m, t), o in zip(init, outs):
        task = clean_reply(o) or problem
        pop.append(unit(task, m, score(task, "pb-init", 0, thinking_style=t)))
    best_fit = max(u.fitness for u in pop)
    for u in pop:
        if u.fitness == best_fit:
            u.lineage.append(u.task)

    def mutate(w: _Unit, op: str) -> tuple[str, str, str]:
        """(new task-prompt, new mutation-prompt, operator actually used)."""
        nonlocal fallbacks
        if op == "zero_order":
            return first_line(calls.one(f"{problem} {PB_HINTS}")), w.mutation, op
        if op == "first_order":
            return clean_reply(calls.one(f"{w.mutation} INSTRUCTION: {w.task} INSTRUCTION MUTANT:")), w.mutation, op
        if op in ("eda", "eda_rank"):
            order = sorted(pop, key=lambda u: u.fitness) if op == "eda_rank" else \
                [pop[int(i)] for i in rng.permutation(len(pop))]
            tasks = [u.task for u in order]
            tasks = [tasks[i] for i in diversity_filter(tasks, diversity_threshold)]
            lst = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(tasks))
            n = len(tasks)
            if op == "eda":
                user = f"{lst}\n{n + 1}."
            else:
                user = (f"INSTRUCTION: {w.mutation}\n A List of Responses in descending order of score. {n + 1} is "
                        f"the best response. It resembles {n} more than it does (1)\n{lst}\n{n + 1}.")
            return first_line(calls.one(user)), w.mutation, op
        if op == "lineage":
            lin = w.lineage or [w.task]
            lst = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lin))
            return first_line(calls.one(f"{PB_LINEAGE}\n{lst}\n{len(lin) + 1}.")), w.mutation, op
        if op in ("zero_order_hyper", "first_order_hyper"):
            hyper = f"{problem} {pick(thinking_styles)}" if op == "zero_order_hyper" else f"{PB_HYPER} {w.mutation}"
            m2 = clean_reply(calls.one(hyper)) or w.mutation
            return clean_reply(calls.one(f"{m2} INSTRUCTION: {w.task} INSTRUCTION MUTANT:")), m2, op
        if op == "lamarckian":
            r = ev.cached(w.task, "train") or ev.scores(w.task, "train")
            outs_w, s = r["outputs"], r["scores"]
            thr = ev.best_score("train") if correct_at is None else correct_at
            if outs_w and np.isfinite(thr):
                with np.errstate(invalid="ignore"):
                    good = s >= thr if ev.hib else s <= thr
                ok = [i for i in np.flatnonzero(good) if outs_w[i]]
                if ok:
                    pickd = rng.choice(ok, min(n_workings, len(ok)), replace=False)
                    work = "\n".join(str(outs_w[int(i)]) for i in pickd)
                    return clean_reply(calls.one(PB_LAMARCK.format(workings=work))), w.mutation, op
            fallbacks += 1
            return mutate(w, "first_order")[0], w.mutation, "first_order"
        raise ValueError(op)

    for g in range(1, generations + 1):
        perm = rng.permutation(len(pop))
        for i, j in zip(perm[0::2], perm[1::2]):
            a, b = pop[int(i)], pop[int(j)]
            if a.fitness == b.fitness:
                win, lose = (int(i), int(j)) if rng.random() < 0.5 else (int(j), int(i))
            else:
                win, lose = (int(i), int(j)) if a.fitness > b.fitness else (int(j), int(i))
            w = pop[win]
            op = pick(list(operators))
            task, mut, used = mutate(w, op)
            op_count[op] += 1
            xo = False
            if crossover_p and rng.random() < crossover_p:
                others = [k for k in range(len(pop)) if k not in (win, lose)]
                if others:
                    f = np.array([pop[k].fitness for k in others], float)
                    fin = np.isfinite(f)
                    wgt = np.where(fin, f - (f[fin].min() if fin.any() else 0.0) + 1e-9, 0.0)
                    p = wgt / wgt.sum() if wgt.sum() > 0 else None      # fitness proportionate (shifted >= 0)
                    task, xo = pop[others[int(rng.choice(len(others), p=p))]].task, True
            task = task or w.task
            c = score(task, f"pb-{used}", g, parent=w.task, crossover=xo)
            child = unit(task, mut, c, w.lineage)
            if child.fitness > max(u.fitness for k, u in enumerate(pop) if k != lose):
                child.lineage.append(task)            # a new population elite joins its lineage
            pop[lose] = child
    return {"history": hist, "llm_calls": calls.n, "cost_usd": calls.cost, "operators": op_count,
            "lamarckian_fallbacks": fallbacks,
            "population": [{"task": u.task, "mutation": u.mutation, "train": u.train} for u in pop]}


# ===================================================================== selection and the single test

@dataclass
class PromptEvolutionResult:
    method: str
    best: Candidate
    history: pd.DataFrame
    validation: pd.DataFrame
    test: dict                          # name -> {"mean", "ci", "n"}
    test_diff: dict                     # "winner-<name>" -> {"ci95", "bonferroni", "better"}
    seal: dict | None
    search: dict                        # llm_calls, cost_usd, unit evaluations, method extras
    gaps: dict                          # train - val, val - test of the winner (raw direction)


def history_frame(history: Sequence[Candidate]) -> pd.DataFrame:
    cols = ["text", "train", "fitness", "n_units", "full_train", "method", "origin", "step", "parent"]
    return pd.DataFrame([{k: getattr(c, k) for k in cols} for c in history], columns=cols)


def select_and_test(ev: PromptEvaluator, history: Sequence[Candidate], *, method: str = "", top_k: int = 5,
                    baselines: dict[str, str] | None = None, registry_dir=None, prereg=None, alpha: float = 0.05,
                    seed: int = 42, n_boot: int = 1000, search: dict | None = None) -> PromptEvolutionResult:
    """Top-k distinct texts by training fitness (full-train scores preferred) -> validation -> the best validation
    text wins (ties: higher train fitness, then shorter). The winner is sealed (`registry_dir`) and a locked prereg is
    verified before the test split is scored ONCE for the winner and every baseline. Paired bootstrap winner minus
    baseline with a 95% and a Bonferroni (k = number of baselines) CI."""
    if ev.test_done:
        raise PermissionError("the test split was already scored for this evaluator")
    if not history:
        raise ValueError("empty history")
    best_by_text: dict[str, Candidate] = {}
    for c in history:
        if not c.text:
            continue
        o = best_by_text.get(c.text)
        if o is None or (c.full_train, c.fitness) > (o.full_train, o.fitness):
            best_by_text[c.text] = c
    ranked = sorted(best_by_text.values(), key=lambda c: (c.full_train, c.fitness), reverse=True)[:top_k]
    rescored = []
    for i, c in enumerate(ranked):                   # subset-only scores (APE stages, PB batches): rescore full train
        if not c.full_train:
            ranked[i] = best_by_text[c.text] = ev.candidate(c.text, c.method, c.origin, step=c.step, parent=c.parent,
                                                            rescored=True)
            rescored.append(ranked[i])
    rows = []
    for c in ranked:
        v = ev.scores(c.text, "val")
        rows.append({"text": c.text, "train": c.train, "val": v["mean"], "val_fitness": ev.fitness(v["mean"]),
                     "origin": c.origin, "n_val": int(np.isfinite(v["scores"]).sum())})
    val = pd.DataFrame(rows)
    val["_len"] = val.text.str.len()
    val["_tf"] = [ev.fitness(x) for x in val.train]
    val = val.sort_values(["val_fitness", "_tf", "_len"], ascending=[False, False, True]).drop(columns=["_len", "_tf"])
    val = val.reset_index(drop=True)
    win = best_by_text[val.text.iloc[0]]
    rec = None
    if registry_dir is not None:
        rec = seal({"method": method, "winner": win.text, "train": win.train, "val": float(val.val.iloc[0]),
                    "top_k": val.text.tolist(), "baselines": dict(baselines or {})},
                   f"prompt_evolution {method}: winner before the test split", registry_dir)
    if prereg is not None:
        chk = verify_prereg(prereg)
        if not chk["ok"]:
            raise ValueError(f"prereg check failed before the test split: {chk['reason']}")
    out = PromptEvolutionResult(method, win, history_frame([*history, *rescored]), val, {}, {}, rec, dict(search or {}), {})
    if ev.n("test") == 0:
        return out
    ev.test_done = True
    names = {"winner": win.text, **{k: v for k, v in (baselines or {}).items()}}
    per = {k: ev.scores(t, "test", _allow_test=True)["scores"] for k, t in names.items()}
    for k, s in per.items():
        m = bootstrap_mean(s[np.isfinite(s)], seed=seed, B=n_boot, alpha=alpha)
        out.test[k] = {"text": names[k], "mean": m["value"], "ci": (m["lo"], m["hi"]), "n": m["n"]}
    k_b = max(1, len(names) - 1)
    for k in names:
        if k == "winner":
            continue
        ok = np.isfinite(per["winner"]) & np.isfinite(per[k])
        if not ok.any():
            continue
        d = paired_bootstrap_diff(per["winner"][ok], per[k][ok], seed=seed, B=n_boot, alpha=alpha)
        db = paired_bootstrap_diff(per["winner"][ok], per[k][ok], seed=seed, B=n_boot, alpha=alpha / k_b)
        out.test_diff[f"winner-{k}"] = {"ci95": d, "bonferroni": db, "better": is_better(db, ev.hib)}
    vt = out.test["winner"]["mean"]
    out.gaps = {"train_minus_val": win.train - float(val.val.iloc[0]),
                "val_minus_test": float(val.val.iloc[0]) - vt if vt is not None else None}
    out.search["unit_evals"] = dict(ev.unit_evals)
    return out


METHODS = {"opro": opro, "ape": ape, "promptbreeder": promptbreeder}


def evolve(method: str, llm: LLM, ev: PromptEvaluator, *, seeds: Sequence[str] = (), problem: str | None = None,
           top_k: int = 5, baselines: dict[str, str] | None = None, registry_dir=None, prereg=None,
           alpha: float = 0.05, **kw) -> PromptEvolutionResult:
    """Search with `method` ("opro" | "ape" | "promptbreeder") on train, then `select_and_test`. Baselines default to
    the seed texts ({"seed0": seeds[0], ...}). `problem` is Promptbreeder's problem description (default: seeds[0])."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    if method == "opro":
        if not seeds:
            raise ValueError("opro needs at least one seed text")
        s = opro(llm, ev, seeds, **kw)
    elif method == "ape":
        s = ape(llm, ev, seeds=seeds, **kw)
    else:
        prob = problem or (seeds[0] if seeds else None)
        if not prob:
            raise ValueError("promptbreeder needs a problem description (problem= or a seed)")
        s = promptbreeder(llm, ev, prob, seeds=seeds, **kw)
    base = baselines if baselines is not None else {f"seed{i}": t for i, t in enumerate(dict.fromkeys(seeds))}
    hist = s.pop("history")
    return select_and_test(ev, hist, method=method, top_k=top_k, baselines=base, registry_dir=registry_dir,
                           prereg=prereg, alpha=alpha, search=s)


# ===================================================================== offline toy problem

def toy_problem(n_units: int = 300, n_keys: int = 12, cap: int = 4, noise: float = 0.05, seed: int = 0):
    """Synthetic task for tests and dry runs (no LLM). Each unit needs one keyword (Zipf-like frequencies); a text
    scores 1 on a unit if it contains that keyword and at most `cap` distinct keywords, flipped with prob `noise`.
    The optimum is the `cap` most frequent keywords. Returns (units DataFrame, score_fn, keywords, optimizer
    responder for FakeLLM: it reads the last listed text / demonstrations and adds, drops or swaps one keyword)."""
    rng = np.random.default_rng(seed)
    keys = [f"kw{i:02d}" for i in range(n_keys)]
    p = 1.0 / np.arange(1, n_keys + 1)
    p /= p.sum()
    need = rng.choice(n_keys, n_units, p=p)
    flip = rng.random(n_units) < noise
    units = pd.DataFrame({"unit": np.arange(n_units), "input": [f"case {i} about {keys[k]}" for i, k in
                                                                 enumerate(need)],
                          "output": [keys[k] for k in need], "_need": need, "_flip": flip})
    kre = re.compile(r"kw\d\d")

    def score_fn(text, sub):
        have = set(kre.findall(text or ""))
        hit = np.array([keys[k] in have for k in sub["_need"]], float) if len(have) <= cap else np.zeros(len(sub))
        s = np.where(sub["_flip"].to_numpy(), 1 - hit, hit)
        return {"scores": s, "outputs": [f"working: {keys[k]}" if v else "" for k, v in zip(sub["_need"], s)]}

    state = {"n": 0}

    def responder(system, user):
        state["n"] += 1
        r = np.random.default_rng(seed * 7919 + state["n"])
        blocks = re.findall(r"(?:^|\n)(?:text:\n|\d+\. |Input: )(.+)", user)
        if "Here are the input-output pairs" in user:          # APE forward: induce from demonstrations
            outs = re.findall(r"Output: (kw\d\d)", user)
            vals, cnt = np.unique(outs, return_counts=True)
            cur = list(vals[np.argsort(-cnt)][:int(r.integers(1, cap + 2))])
        else:
            last = blocks[-1] if blocks else user
            cur = list(dict.fromkeys(kre.findall(last)))
            mv = r.random()
            if (mv < 0.5 or not cur) and len(cur) < n_keys:
                cur.append(keys[int(r.choice(n_keys, p=p))])
            elif mv < 0.75 and len(cur) > 1:
                cur.pop(int(r.integers(len(cur))))
            elif cur:
                cur[int(r.integers(len(cur)))] = keys[int(r.integers(n_keys))]
            cur = list(dict.fromkeys(cur))
        return "[" + "use " + " ".join(cur) + "]" if "square brackets" in user else "use " + " ".join(cur)

    return units, score_fn, keys, responder


__all__ = ["PromptEvaluator", "Candidate", "PromptEvolutionResult", "split_units", "make_llm_scorer", "exact_match",
           "normalise_answer", "opro", "opro_meta_prompt", "ape", "ape_forward_prompt", "promptbreeder",
           "diversity_filter", "select_and_test", "evolve", "history_frame", "clean_reply", "first_line",
           "parse_square_brackets", "toy_problem", "MUTATION_PROMPTS", "THINKING_STYLES", "PB_OPERATORS", "METHODS"]
