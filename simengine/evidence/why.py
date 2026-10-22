"""N-way 'why' recognition: is a generated reason closer to the person's real reason than to other people's?

    import simengine.evidence as ev
    frame = ev.why_match({"persona": gen_p, "no_persona": gen_0}, candidates, truth_idx, ids=person_ids)
    report = ev.evaluate(frame, task="multiclass", primary="accuracy", ours="persona", raw="no_persona")
    ev.why_summary(frame)       # {arm: accuracy, MRR, chance levels} without the bootstrap

One row per unit (a person x product, say): ``generated[arm][i]`` is the reason an arm wrote, ``candidates[i]`` are N
real reasons, exactly one of them (``truth_idx[i]``) the person's own. Each arm "chooses" the candidate most similar
to its text; accuracy is the share of rows where it chose the true one, the reciprocal rank (``rr_<arm>``) says how
far down the true one sat. Chance: accuracy 1/N, MRR H_N/N (H_N the N-th harmonic number).

Distractor recipe: for each unit take N-1 reasons written by *other* people about the *same product* with the *same
rating* as the true reason. Holding product and rating fixed leaves only reason content to tell them apart, so a
generated text cannot win by guessing sentiment or naming the product. Draw distractors before generating, seal the
candidate sets with the truth position, and shuffle positions (the truth index must not be constant).

Similarity is cosine. ``encoder="tfidf"`` (default) fits word 1-2 gram + char_wb 2-5 gram TF-IDF on the candidate
texts (all rows; generated texts are only transformed) and concatenates the two blocks with equal weight (pure
numpy, sklearn-compatible defaults; `TfidfEncoder` can also be fit on a larger corpus and passed in). Any
callable ``texts -> (n, d) array`` works, e.g. ``sim.SSR(model).embed`` for a sentence encoder; tests pass a toy
callable so no model is downloaded. Ties (e.g. an empty generated text) are broken at random with ``seed`` and the
reciprocal rank uses the average rank over ties, so a blank text scores chance, not candidate 0. TF-IDF rewards
shared wording (a person's own vocabulary and style) as much as the reason; report an embedding row next to it.
"""
from __future__ import annotations

import math
import re
import warnings
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..types import PRED_PREFIX

__all__ = ["why_match", "why_summary", "tfidf_encoder", "TfidfEncoder", "RR_PREFIX"]

RR_PREFIX = "rr_"
Encoder = Callable[[list[str]], np.ndarray]


_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def _word_grams(text: str) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


def _char_grams(text: str, lo: int = 2, hi: int = 5) -> list[str]:
    out = []
    for w in text.lower().split():                     # char_wb: n-grams inside space-padded words
        w = f" {w} "
        for n in range(lo, hi + 1):
            out.extend(w[i:i + n] for i in range(max(1, len(w) - n + 1)))
    return out


class TfidfEncoder:
    """[word 1-2 gram | char_wb 2-5 gram] TF-IDF fit on `corpus` (sklearn-style: lowercase, \\w\\w+ tokens, smooth
    idf, sublinear tf, each block l2-normalised, blocks weighted equally). Terms absent from the corpus are ignored.
    Pure python/numpy (no sklearn). ``vectors(texts)`` -> sparse {term: weight} dicts; calling the encoder gives a
    dense (n, d) array whose columns are the terms present in *those* texts (consistent within one call only)."""

    def __init__(self, corpus: Sequence[str]):
        corpus = [str(t or "") for t in corpus]
        self.idf = []
        for grams in (_word_grams, _char_grams):
            df: dict[str, int] = {}
            for t in corpus:
                for g in set(grams(t)):
                    df[g] = df.get(g, 0) + 1
            n = len(corpus)
            self.idf.append({g: math.log((1 + n) / (1 + k)) + 1.0 for g, k in df.items()})

    def vectors(self, texts: Sequence[str]) -> list[dict]:
        out = []
        for t in texts:
            v: dict = {}
            for b, (grams, idf) in enumerate(zip((_word_grams, _char_grams), self.idf)):
                tf: dict[str, int] = {}
                for g in grams(str(t or "")):
                    if g in idf:
                        tf[g] = tf.get(g, 0) + 1
                blk = {(b, g): (1.0 + math.log(k)) * idf[g] for g, k in tf.items()}
                norm = math.sqrt(sum(x * x for x in blk.values())) * math.sqrt(2.0)
                v.update({k: x / norm for k, x in blk.items()} if norm > 0 else {})
            out.append(v)
        return out

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        vs = self.vectors(texts)
        cols = {k: j for j, k in enumerate(sorted({k for v in vs for k in v}))}
        E = np.zeros((len(vs), max(1, len(cols))))
        for i, v in enumerate(vs):
            for k, x in v.items():
                E[i, cols[k]] = x
        return E


def tfidf_encoder(corpus: Sequence[str]) -> TfidfEncoder:
    """`TfidfEncoder` fit on `corpus`."""
    return TfidfEncoder(corpus)


def _dot(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return float(sum(x * b.get(k, 0.0) for k, x in a.items()))


def _unit(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=float)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.where(n > 0, n, 1.0)


def _arms(generated) -> dict[str, list]:
    if isinstance(generated, Mapping):
        return {str(k): list(v) for k, v in generated.items()}
    if isinstance(generated, pd.DataFrame):
        return {str(c): generated[c].tolist() for c in generated.columns}
    return {"gen": list(generated)}


def _choose(scores: np.ndarray, rng: np.random.Generator) -> int:
    """argmax with a random tie-break."""
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    return int(best[rng.integers(len(best))]) if len(best) > 1 else int(best[0])


def why_match(generated, candidates: Sequence[Sequence[str]], truth_idx: Sequence[int] | int, *,
              encoder: str | Encoder = "tfidf", ids: Sequence | None = None, groups: Sequence | None = None,
              seed: int = 0) -> pd.DataFrame:
    """Score each arm's generated reason against N candidate real reasons -> EvalFrame (task 'multiclass').

    generated: {arm: [text per row]} (or a DataFrame with one column per arm, or one list -> arm "gen").
    candidates: per row, the N candidate texts (N may differ across rows). truth_idx: index of the true candidate per
    row (or one int for all rows; shuffle positions so it is not constant). encoder: "tfidf" or a callable
    texts -> (n, d) array (an object with ``.embed`` also works). ids: id per row -> ``unit`` (repeats get a
    ``#k`` suffix so units stay unique) and, unless ``groups`` is given, ``group``. groups: bootstrap cluster per
    row (the person id when a person has several rows, e.g. person x product units).

    Columns: unit, group, truth (true index), n_candidates, chance (1/N), pred_<arm> (chosen index), rr_<arm>
    (reciprocal rank of the true candidate, average rank over ties). ``frame.attrs["why"]`` holds the encoder name
    and chance levels. Feed it to ``evaluate(frame, task="multiclass", primary="accuracy")``; MRR via `why_summary`.
    """
    arms = _arms(generated)
    cands = [[str(c or "") for c in row] for row in candidates]
    n = len(cands)
    for a, g in arms.items():
        if len(g) != n:
            raise ValueError(f"why_match: arm {a!r} has {len(g)} texts, candidates has {n} rows")
    truth = np.full(n, int(truth_idx)) if np.ndim(truth_idx) == 0 else np.asarray(truth_idx, dtype=int)
    if len(truth) != n:
        raise ValueError(f"why_match: truth_idx has {len(truth)} entries, candidates has {n} rows")
    sizes = np.array([len(r) for r in cands])
    if n and (sizes < 2).any():
        raise ValueError("why_match: every row needs at least 2 candidates")
    if ((truth < 0) | (truth >= sizes)).any():
        raise ValueError("why_match: truth_idx out of range for its candidate list")
    for nm, v in (("ids", ids), ("groups", groups)):
        if v is not None and len(v) != n:
            raise ValueError(f"why_match: {nm} has {len(v)} entries, candidates has {n} rows")
    if n >= 10 and len(np.unique(truth)) == 1:
        warnings.warn("why_match: truth_idx is the same position in every row; evaluate's majority-class baseline "
                      "then scores 1.0. Shuffle candidate positions when building the sets", stacklevel=2)

    if isinstance(encoder, str):
        if encoder != "tfidf":
            raise ValueError(f"why_match: encoder {encoder!r}: use 'tfidf' or a callable texts -> (n, d) array "
                             "(e.g. sim.SSR(model).embed)")
        enc, enc_name = tfidf_encoder([c for r in cands for c in r]), "tfidf"
    else:
        enc = getattr(encoder, "embed", encoder)
        enc_name = getattr(enc, "__qualname__", type(encoder).__name__)

    start = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    flat = [c for r in cands for c in r]
    if isinstance(enc, TfidfEncoder):
        Cv = enc.vectors(flat)
        sim_rows = lambda Gv: [np.array([_dot(Cv[j], Gv[i]) for j in range(start[i], start[i + 1])])
                               for i in range(n)]
        embed = enc.vectors
    else:
        Cv = _unit(enc(flat)) if n else np.zeros((0, 1))
        sim_rows = lambda G: [Cv[start[i]:start[i + 1]] @ G[i] for i in range(n)]
        embed = lambda texts: _unit(enc(texts))
    rng = np.random.default_rng(seed)
    unit = _unique(list(ids)) if ids is not None else list(range(n))
    group = list(groups) if groups is not None else (list(ids) if ids is not None else unit)
    out = pd.DataFrame({"unit": unit, "group": group, "truth": truth, "n_candidates": sizes, "chance": 1.0 / sizes})
    for a, g in arms.items():
        texts = [str(t) if t is not None and t == t else "" for t in g]
        rows = sim_rows(embed(texts)) if n else []
        pred, rr = np.empty(n, dtype=int), np.empty(n)
        for i, s in enumerate(rows):
            pred[i] = _choose(s, rng)
            t = s[truth[i]]
            above = int((s > t + 1e-12).sum())
            ties = int((np.abs(s - t) <= 1e-12).sum()) - 1
            rr[i] = np.mean(1.0 / (above + 1 + np.arange(ties + 1)))   # average of 1/rank over tied positions
        out[PRED_PREFIX + a] = pred
        out[RR_PREFIX + a] = rr
    out.attrs["why"] = {"encoder": enc_name, "arms": list(arms), "chance_accuracy": float(np.mean(1.0 / sizes)) if n
                        else None, "chance_mrr": float(np.mean([_harmonic(k) / k for k in sizes])) if n else None}
    return out


def _unique(ids: list) -> list:
    """Repeated ids -> "<id>#<k>" from the second occurrence on (evaluate audits repeated units)."""
    seen: dict = {}
    out = []
    for u in ids:
        k = seen.get(u, 0)
        seen[u] = k + 1
        out.append(u if k == 0 else f"{u}#{k}")
    return out


def _harmonic(k: int) -> float:
    return float(np.sum(1.0 / np.arange(1, k + 1)))


def why_summary(frame: pd.DataFrame, arms: Sequence[str] | None = None) -> dict:
    """{arm: {n, accuracy, mrr}} plus chance levels, from a `why_match` frame (point values; CIs come from
    `evaluate`)."""
    arms = list(arms) if arms is not None else [c[len(PRED_PREFIX):] for c in frame.columns
                                                  if str(c).startswith(PRED_PREFIX)]
    sizes = frame["n_candidates"].to_numpy()
    res = {"n": int(len(frame)), "chance_accuracy": float(np.mean(1.0 / sizes)) if len(frame) else None,
           "chance_mrr": float(np.mean([_harmonic(int(k)) / k for k in sizes])) if len(frame) else None, "arms": {}}
    for a in arms:
        hit = frame[PRED_PREFIX + a].to_numpy() == frame["truth"].to_numpy()
        res["arms"][a] = {"accuracy": float(hit.mean()) if len(frame) else None,
                          "mrr": float(frame[RR_PREFIX + a].mean()) if len(frame) else None}
    return res
