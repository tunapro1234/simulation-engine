"""socsci210: response distribution of each (condition, outcome) cell of 210 TESS survey experiments.

SocSci210 (Kolluri et al. 2025, arXiv 2509.05830; HF socratesft/SocSci210) holds 2.9M individual answers from 210
experiments. This task scores the paper's distributional alignment: for every (study, outcome, condition) cell the
human answers form a distribution on the stated response scale, rescaled to [0, 1] by (r - r_min) / (r_max - r_min);
the score is the Wasserstein-1 distance between predicted and human distribution, averaged over the cells of a study
and then over studies (lower is better). Splits follow the paper's mapping files:

    split="study"      (default, paper 5.3) 170 seen studies train, 40 unseen studies test
    split="condition"  (paper 5.4) studies with >= 4 conditions; 75% of conditions train, 25% test
    split="outcome"    (paper 5.4) same on outcomes (task_num)

    from simengine import bench
    from simengine.bench.tasks import socsci210          # registers "socsci210"
    task = bench.get("socsci210", split="study")
    data = task.load()                                   # $SIMENGINE_DATA; data.test: stimuli, lo, hi, levels, P
    frame = task.run(methods={"mine": lambda d: [my_dist(s, k) for s, k in zip(d.test.stimuli, d.test.levels)]},
                     data=data)
    task.score(frame, "mine")                            # {'metric': 'w1', 'value', 'lo', 'hi', 'n', ...}
    task.reference(frame, "empirical_best")              # the paper's bootstrap floor on the same cells

A prediction is one probability vector per test cell over the integer levels lo..hi (length `levels`);
`dist_from_responses` turns sampled integer answers (e.g. from an LLM persona panel) into such a vector.

The raw shards carry one row per answer; `build_hist` aggregates them into one row per cell (raw answer counts),
which `load` prefers when `<root>/socsci210/hist.parquet` exists; `cells` applies the scale parsed from the stimulus
(`parse_scale`) and drops off-scale answers. Open-ended amounts (dollars, minutes) have no stated scale and are left
out, as in the paper (ordinal and binary outcomes only).
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, DataNotFound, Metric, need, register
from ..stats import bootstrap_mean, cluster_bootstrap

SPLITS = ("study", "condition", "outcome")
BASELINES = ("uniform", "scale_marginal", "tfidf_knn", "pooled")
SHARD_COLUMNS = ["study_id", "task_num", "condition_num", "sample_id", "response", "stimuli"]
HIST_FILE = "hist.parquet"

# "an integer from 1 (Strongly disagree) to 7 (...)", "a number between 0 and 100", "from 1 to 7, where 1 means ..."
_RANGE = re.compile(r"(?:integer|number|value|whole number)s? (?:from|between) (-?\d+)(?:\s*\([^)]*\))?"
                    r"(?:,? where [^.]*?)? (?:to|and) (-?\d+)")
# "1 for Yes or 2 for No", "1 if you ...; 2 if ...", "1 to choose the first ...; 2 to choose ..."
_ENUM = re.compile(r"(-?\d+) (?:for|=|means|if|to [a-z])")
# "Only return 1, 2, 3, or 4" / "Only return 1-4"
_LIST = re.compile(r"^Only return (-?\d+(?:(?:,? or |, )-?\d+)+),? nothing else")
_DASH = re.compile(r"^Only return (-?\d+)\s*[-\u2013]\s*(-?\d+),? nothing else")
# no instruction: "on a scale from 1 (very negative) to 7 (very positive)"
_ANCHORED = re.compile(r"(-?\d+) \([^)]*\) to (-?\d+) \(")

# ---------------------------------------------------------------------- scale and distributions
def parse_scale(text: str) -> tuple[int, int] | None:
    """(r_min, r_max) stated in the response instruction ("Only return ..."), or None (open / unparsed answer).
    Tried in order: a range ("an integer from 1 to 7"), a dash or a consecutive list ("1-4", "1, 2, or 3"), the
    enumerated codes ("1 for Yes or 2 for No"); without an instruction, an anchored scale ("from 1 (low) to 7
    (high)"). Open amounts ("an integer representing dollars") and non-consecutive lists give None."""
    if not isinstance(text, str):
        return None
    i = text.rfind("Only return")
    if i < 0:
        m = _ANCHORED.findall(text)
        lo, hi = (int(v) for v in m[-1]) if m else (0, 0)
        return (lo, hi) if hi > lo else None
    tail = text[i:]
    m = _RANGE.findall(tail)
    if m:
        lo, hi = (int(v) for v in m[-1])
        return (lo, hi) if hi > lo else None
    m = _DASH.match(tail)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (lo, hi) if hi > lo else None
    m = _LIST.match(tail)
    if m:
        v = sorted(int(x) for x in re.findall(r"-?\d+", m.group(1)))
        return (v[0], v[-1]) if v == list(range(v[0], v[-1] + 1)) else None
    codes = [int(v) for v in _ENUM.findall(tail)]
    return (min(codes), max(codes)) if len(codes) >= 2 and max(codes) > min(codes) else None


def positions(levels: int) -> np.ndarray:
    """Level positions on [0, 1]: (r - r_min) / (r_max - r_min) for r = r_min..r_max."""
    return np.linspace(0.0, 1.0, int(levels))


def dist_from_responses(values: Iterable[float], lo: int, hi: int) -> np.ndarray:
    """Integer answers -> probability vector over lo..hi (answers rounded; out-of-range / NaN answers ignored;
    uniform if none is left)."""
    v = np.asarray(list(values), float)
    v = np.round(v[np.isfinite(v)])
    v = v[(v >= lo) & (v <= hi)].astype(int) - int(lo)
    k = int(hi) - int(lo) + 1
    c = np.bincount(v, minlength=k).astype(float)
    return c / c.sum() if c.sum() > 0 else np.full(k, 1.0 / k)


def _normalise(p, k: int) -> np.ndarray:
    p = np.clip(np.asarray(p, float).ravel(), 0, None)
    if len(p) != k:
        raise ValueError(f"prediction has {len(p)} levels, cell has {k}")
    s = p.sum()
    return p / s if s > 0 and np.isfinite(s) else np.full(k, 1.0 / k)


def w1(p: np.ndarray, q: np.ndarray, pos: np.ndarray | None = None) -> float:
    """Wasserstein-1 between two distributions on sorted positions (default: evenly spaced on [0, 1])."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    pos = positions(len(p)) if pos is None else np.asarray(pos, float)
    return float(np.sum(np.abs(np.cumsum(p)[:-1] - np.cumsum(q)[:-1]) * np.diff(pos)))


def w1_uniform01(p: np.ndarray, pos: np.ndarray | None = None) -> float:
    """Wasserstein-1 between a discrete distribution on positions in [0, 1] and the continuous U(0, 1): the
    integral of |F(x) - x| over [0, 1], F the step CDF (exact). Our reading of the paper's "Uniform Guess"."""
    p = np.asarray(p, float)
    pos = positions(len(p)) if pos is None else np.asarray(pos, float)
    a, b, c = pos[:-1], pos[1:], np.cumsum(p)[:-1]   # F = c on [a, b)
    below = ((b - c) ** 2 - (a - c) ** 2) / 2
    above = ((c - a) ** 2 - (c - b) ** 2) / 2
    inside = ((c - a) ** 2 + (b - c) ** 2) / 2
    seg = np.where(c <= a, below, np.where(c >= b, above, inside))
    return float(seg.sum() + pos[0] ** 2 / 2 + (1 - pos[-1]) ** 2 / 2)


def empirical_best(p: np.ndarray, n: int, *, reps: int = 100, rng: np.random.Generator | None = None,
                   pos: np.ndarray | None = None) -> float:
    """Paper's "Empirical Best": mean W1 between `reps` with-replacement resamples of the n answers and the full
    sample (multinomial(n, p) draws)."""
    p = np.asarray(p, float)
    pos = positions(len(p)) if pos is None else np.asarray(pos, float)
    rng = rng if rng is not None else np.random.default_rng(0)
    if n <= 0:
        return float("nan")
    r = rng.multinomial(int(n), p / p.sum(), size=reps) / n
    return float((np.abs(np.cumsum(r, 1)[:, :-1] - np.cumsum(p)[:-1]) @ np.diff(pos)).mean())


# ---------------------------------------------------------------------- raw shards -> one row per cell
def _find(root: str, *rels: str) -> str | None:
    for rel in rels:
        for base in (root, os.path.join(root, "ham"), os.path.join(root, "islenmis")):
            p = os.path.join(base, rel)
            if os.path.exists(p):
                return p
    return None


def _eval_share(d: pd.DataFrame, mapping: dict | None, key: str) -> pd.Series:
    """Per row: 1 if its sample_id is in mapping[study]['eval'], 0 if in 'train', NaN if the study is not mapped."""
    if not mapping:
        return pd.Series(np.nan, index=d.index)
    rows = [(s, int(i), 1.0) for s, m in mapping.items() for i in m.get("eval", [])]
    rows += [(s, int(i), 0.0) for s, m in mapping.items() for i in m.get("train", [])]
    ref = pd.DataFrame(rows, columns=["study_id", "sample_id", key]).drop_duplicates(["study_id", "sample_id"])
    return d[["study_id", "sample_id"]].merge(ref, how="left", on=["study_id", "sample_id"])[key].set_axis(d.index)


def build_hist(shards: Sequence[str], metadata_dir: str | None = None) -> pd.DataFrame:
    """Aggregate HF shards (one row per answer) to one row per (study_id, task_num, condition_num) cell:
    stimuli, answers / answer_counts (every distinct raw answer and its count, ascending), seen (study in the
    paper's 170 training studies; None if unmapped), cond_eval / task_eval (share of the cell's answers in the
    condition / outcome mapping's eval part; NaN if the study is not in that mapping). The response scale is
    applied later by `cells`, so a better `parse_scale` needs no rebuild. Reads only SHARD_COLUMNS, one shard
    (~170k answers) at a time."""
    maps = {}
    for name in ("participant", "condition", "task"):
        p = os.path.join(metadata_dir, f"{name}_mapping.json") if metadata_dir else None
        if p and os.path.exists(p):
            with open(p) as f:
                maps[name] = json.load(f)
        else:
            maps[name] = None
    keys = ["study_id", "task_num", "condition_num"]
    counts, first, shares = [], [], []
    for path in shards:
        d = pd.read_parquet(path, columns=SHARD_COLUMNS)
        d["cond_eval"] = _eval_share(d, maps["condition"], "cond_eval")
        d["task_eval"] = _eval_share(d, maps["task"], "task_eval")
        counts.append(d.groupby(keys + ["response"]).size().rename("c").reset_index())
        first.append(d.drop_duplicates(keys)[keys + ["stimuli"]])
        shares.append(d.groupby(keys)[["cond_eval", "task_eval"]].agg(["sum", "count"]))
        del d
    if not counts:
        raise DataNotFound("socsci210: no shards given")
    c = pd.concat(counts).groupby(keys + ["response"], as_index=False).c.sum().sort_values(keys + ["response"])
    out = pd.concat(first).drop_duplicates(keys).sort_values(keys).reset_index(drop=True)
    sh = pd.concat(shares).groupby(level=list(range(3))).sum()
    for col in ("cond_eval", "task_eval"):
        num, den = sh[(col, "sum")], sh[(col, "count")]
        out[col] = out.set_index(keys).index.map((num / den.where(den > 0)).to_dict())
    vals = c.groupby(keys).agg(answers=("response", list), answer_counts=("c", list))
    out = out.merge(vals, how="left", left_on=keys, right_index=True)
    seen = set(maps["participant"]["seen"]) if maps["participant"] else set()
    unseen = set(maps["participant"]["unseen"]) if maps["participant"] else set()
    out["seen"] = out.study_id.map(lambda s: True if s in seen else (False if s in unseen else None))
    return out


def cells(h: pd.DataFrame) -> pd.DataFrame:
    """Apply the stated scale to a `build_hist` table: adds lo, hi, levels (0 if unparsed), counts (per integer
    level lo..hi), n (answers on the scale), n_dropped (answers off the scale or non-integer) and in_scale (n / all
    answers). Off-scale answers are mostly a coding mismatch in the source (e.g. 0/1 answers to a "1 or 2" item)."""
    out = h.copy()
    sc = out.stimuli.map(parse_scale)
    out["lo"] = [s[0] if s else np.nan for s in sc]
    out["hi"] = [s[1] if s else np.nan for s in sc]
    out["levels"] = (out.hi - out.lo + 1).fillna(0).astype(int)
    vecs, ns, tots = [], [], []
    for lo, k, v, c in zip(out.lo, out.levels, out["answers"], out["answer_counts"]):
        v = np.asarray(v if v is not None and np.ndim(v) else [], float)
        c = np.asarray(c if c is not None and np.ndim(c) else [], float)
        tots.append(int(c.sum()))
        ok = (v == np.round(v)) & (v >= lo) & (v < lo + k) if k > 0 else np.zeros(len(v), bool)
        cnt = np.bincount((v[ok] - lo).astype(int), weights=c[ok], minlength=k).astype(np.int64) if k > 0 else \
            np.zeros(0, np.int64)
        vecs.append(cnt.tolist())
        ns.append(int(cnt.sum()))
    out["counts"], out["n"] = vecs, ns
    out["n_dropped"] = np.asarray(tots) - out.n
    out["in_scale"] = out.n / np.maximum(tots, 1)
    return out


def write_hist(root: str, out_path: str) -> pd.DataFrame:
    """build_hist over every `train-*.parquet` under <root>/(ham/)data with <root>/(ham/)metadata; write parquet."""
    shards = _shards(root)
    h = build_hist(shards, _find(root, "metadata"))
    h.to_parquet(out_path, index=False)
    return h


def _shards(root: str) -> list[str]:
    d = _find(root, "data")
    shards = sorted(glob.glob(os.path.join(d, "train-*.parquet"))) if d else []
    if not shards:
        raise DataNotFound(f"socsci210: no {HIST_FILE} and no data/train-*.parquet under {root}")
    return shards


def load_hist(root: str) -> tuple[pd.DataFrame, str]:
    """(`cells` table, provenance string): <root>/hist.parquet (or islenmis/hist.parquet) if present, else the
    shards (slow: about 3 minutes for all 17)."""
    p = _find(root, HIST_FILE)
    if p:
        h, src = pd.read_parquet(p), p
        if "answers" not in h.columns:
            raise DataNotFound(f"{p} has no answers/answer_counts columns; rebuild it with socsci210.write_hist")
    else:
        shards = _shards(root)
        meta = need(_find(root, "metadata") or os.path.join(root, "ham", "metadata"))
        h, src = build_hist(shards, meta), f"{len(shards)} shard(s) under {os.path.dirname(shards[0])}"
    return cells(h), src


# ---------------------------------------------------------------------- baselines
def _mean_vec(ps: Sequence[np.ndarray], k: int) -> np.ndarray:
    return np.mean(np.stack(ps), 0) if len(ps) else np.full(k, 1.0 / k)


def scale_marginal(train: pd.DataFrame, test: pd.DataFrame) -> list[np.ndarray]:
    """Mean train distribution over cells with the same number of levels (each cell weighted equally)."""
    by_k = {k: _mean_vec(list(g.P), k) for k, g in train.groupby("levels")}
    return [by_k.get(k, np.full(k, 1.0 / k)) for k in test.levels]


def pooled(train: pd.DataFrame, test: pd.DataFrame) -> list[np.ndarray]:
    """Nearest pool of train cells: same study and outcome (other conditions = 'the treatment does nothing'), else
    same study and level count, else scale_marginal."""
    so = {k: list(g.P) for k, g in train.groupby(["study_id", "task_num", "levels"])}
    sk = {k: list(g.P) for k, g in train.groupby(["study_id", "levels"])}
    marg = dict(zip(test.index, scale_marginal(train, test)))
    out = []
    for i, r in zip(test.index, test.itertuples()):
        ps = so.get((r.study_id, r.task_num, r.levels)) or sk.get((r.study_id, r.levels))
        out.append(_mean_vec(ps, r.levels) if ps else marg[i])
    return out


def tfidf_knn(train: pd.DataFrame, test: pd.DataFrame, k: int = 10, *, per_item: int = 1) -> list[np.ndarray]:
    """Mean distribution of the k most similar train cells by stimulus text (TF-IDF 1-2 grams, cosine) among cells
    with the same number of levels, taking at most `per_item` cells per train outcome (study/task; 0 = no cap) so
    one study's near-identical conditions do not fill all k slots; scale_marginal where no such cell exists."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(train.stimuli)
    Xtr, Xte = vec.transform(train.stimuli), vec.transform(test.stimuli)
    trP, trK, teK = list(train.P), train.levels.to_numpy(), test.levels.to_numpy()
    tr_item = pd.factorize(train["study_id"].astype(str) + "/" + train["task_num"].astype(str))[0]
    out: list = [None] * len(test)
    marg = scale_marginal(train, test)
    for lv in np.unique(teK):
        ti, ri = np.flatnonzero(teK == lv), np.flatnonzero(trK == lv)
        if len(ri) == 0:
            for j in ti:
                out[j] = marg[j]
            continue
        S = (Xte[ti] @ Xtr[ri].T).toarray()
        order = np.argsort(-S, axis=1, kind="stable")
        items = tr_item[ri]
        for j, o in zip(ti, order):
            if per_item > 0:   # rank of each cell within its item along this row's order; keep the first per_item
                it = items[o]
                rank = pd.Series(it).groupby(it).cumcount().to_numpy()
                o = o[rank < per_item]
            out[j] = _mean_vec([trP[ri[m]] for m in o[:k]], lv)
    return out


# ---------------------------------------------------------------------- metrics
def _pair_values(frame: pd.DataFrame, col: str, what: str = "w1") -> np.ndarray:
    out = np.empty(len(frame))
    for i, (t, p, k) in enumerate(zip(frame["truth"], frame[col], frame["levels"])):
        q, pos = _normalise(p, k), positions(k)
        out[i] = w1(t, q, pos) if what == "w1" else abs(float(pos @ q) - float(pos @ t))
    return out


def _study_avg(vals: np.ndarray, study: np.ndarray, B: int, seed: int) -> dict:
    """Mean over cells within a study, then over studies; bootstrap over studies."""
    sm = pd.Series(vals).groupby(np.asarray(study)).mean()
    r = bootstrap_mean(sm.to_numpy(), B=B, seed=seed)
    return {**r, "n_cells": int(np.isfinite(vals).sum())}


def _w1_study(task, frame, col, B, seed):
    return _study_avg(_pair_values(frame, col), frame["group"].to_numpy(), B, seed)


def _w1_cell(task, frame, col, B, seed):
    v = _pair_values(frame, col)
    return cluster_bootstrap(lambda i: float(v[i].mean()), frame["group"].to_numpy(), B=B, seed=seed)


def _mean_err(task, frame, col, B, seed):
    return _study_avg(_pair_values(frame, col, "mean"), frame["group"].to_numpy(), B, seed)


def _segment(k: int) -> str:
    return "binary" if k == 2 else ("ordinal" if k <= 11 else "wide")


@register()
class SocSci210(BenchTask):
    """Options: split, max_levels, min_n, min_in_scale, k, per_item, baselines (see accepted_options)."""

    name = "socsci210"
    kind = "distribution"
    data_subdir = "socsci210"
    accepted_options = {
        "split": '"study" | "condition" | "outcome" (default "study")',
        "max_levels": "drop cells whose scale has more levels (default 101)",
        "min_n": "drop cells with fewer in-scale answers (default 1; a cell with none is always dropped)",
        "min_in_scale": "drop cells where a smaller share of answers lies on the stated scale (default 0.9)",
        "k": "tfidf_knn neighbours (default 10)",
        "per_item": 'tfidf_knn: max neighbour cells per train outcome, 0 = no cap (default 1 for split="study", '
                    'else 0)',
        "baselines": '"+"-joined subset of uniform, scale_marginal, tfidf_knn, pooled (default all that apply)',
    }
    layout = """
    <root>/socsci210/hist.parquet   optional, one row per cell (raw answer counts), written by socsci210.write_hist;
        preferred if present (also <root>/socsci210/islenmis/hist.parquet)
    <root>/socsci210/ham/data/train-000NN-of-00017.parquet  HF socratesft/SocSci210 shards (or <root>/socsci210/data/):
        study_id, task_num, condition_num, sample_id, response, stimuli (other columns ignored); any subset of the
        17 shards loads, covering only the studies in it
    <root>/socsci210/ham/metadata/participant_mapping.json, condition_mapping.json, task_mapping.json   (HF metadata/)
    """
    source = ("Kolluri et al. 2025, arXiv 2509.05830 (sections 5.3-5.4: Wasserstein on [0, 1], cell -> study -> "
              "overall mean; Uniform Guess, Empirical Best); HF socratesft/SocSci210 (TESS experiments)")
    metrics = (Metric("w1", False, _w1_study, "Wasserstein-1 on [0, 1], mean over a study's cells then over studies"),
               Metric("w1_cell", False, _w1_cell, "Wasserstein-1 on [0, 1], mean over cells (study cluster CI)"),
               Metric("mean_err", False, _mean_err, "|predicted mean - human mean| on [0, 1], study-averaged"))

    def _load(self, root: str) -> TaskData:
        split = self.options.get("split", "study")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        h, src = load_hist(root)
        max_k, min_n = int(self.options.get("max_levels", 101)), max(int(self.options.get("min_n", 1)), 1)
        min_share = float(self.options.get("min_in_scale", 0.9))
        n_all, n_unparsed = len(h), int((h.levels < 2).sum())
        n_offscale = int(((h.levels >= 2) & (h.in_scale < min_share)).sum())
        h = h[(h.levels >= 2) & (h.levels <= max_k) & (h.n >= min_n) & (h.in_scale >= min_share)].copy()
        h["P"] = [np.asarray(c, float) / max(sum(c), 1) for c in h.counts]
        h["unit"] = h.study_id + "/t" + h.task_num.astype(str) + "/c" + h.condition_num.astype(str)
        h["item"] = h.study_id + "/t" + h.task_num.astype(str)
        if split == "study":
            known = h.seen.notna()
            train, test = h[known & h.seen.astype(bool)], h[known & ~h.seen.fillna(False).astype(bool)]
            mixed = 0
        else:
            s = h["cond_eval" if split == "condition" else "task_eval"]
            mixed = int((s.notna() & (s > 0) & (s < 1)).sum())
            train, test = h[s == 0], h[s == 1]
        meta = {"split": {"study": "paper 5.3: participant_mapping seen (train) / unseen (test) studies",
                          "condition": "paper 5.4: condition_mapping eval conditions (studies with >= 4 conditions)",
                          "outcome": "paper 5.4: task_mapping eval outcomes (studies with >= 4 outcomes)"}[split],
                "split_name": split, "cells_total": n_all, "cells_unparsed": n_unparsed,
                "cells_off_scale": n_offscale, "cells_kept": int(len(h)), "cells_mixed_split": mixed,
                "studies_train": int(train.study_id.nunique()), "studies_test": int(test.study_id.nunique()),
                "cells_train": int(len(train)), "cells_test": int(len(test)), "data": src,
                "source": "SocSci210 (Kolluri et al. 2025), HF socratesft/SocSci210; TESS experiments; the HF card "
                          "states no licence (not verified)"}
        return TaskData(train=train.reset_index(drop=True), test=test.reset_index(drop=True), meta=meta)

    def baselines(self, data: TaskData) -> dict:
        tr, te = data.train, data.test
        want = self.options.get("baselines")
        names = want.split("+") if want else [b for b in BASELINES if b != "pooled" or
                                               data.meta.get("split_name") != "study"]
        unknown = set(names) - set(BASELINES)
        if unknown:
            raise ValueError(f"unknown baselines {sorted(unknown)}; have {BASELINES}")
        fns = {"uniform": lambda: [np.full(k, 1.0 / k) for k in te.levels],
               "scale_marginal": lambda: scale_marginal(tr, te),
               "tfidf_knn": lambda: tfidf_knn(tr, te, int(self.options.get("k", 10)),
                                              per_item=int(self.options.get(
                                                  "per_item", 1 if data.meta.get("split_name") == "study" else 0))),
               "pooled": lambda: pooled(tr, te)}
        return {b: fns[b]() for b in names}

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        rng = np.random.default_rng(0)
        eb = [empirical_best(p, n, rng=rng) for p, n in zip(t.P, t.n)]
        return pd.DataFrame({"unit": t.unit, "item": t.item, "group": t.study_id, "levels": t.levels,
                             "segment": t.levels.map(_segment), "n": t.n,
                             "truth": pd.Series([np.asarray(p, float) for p in t.P], dtype=object),
                             "ref_empirical_best": eb, "ref_uniform01": [w1_uniform01(p) for p in t.P]})

    def reference(self, frame: pd.DataFrame, which: str = "empirical_best", *, B: int = 1000,
                  seed: int = 42) -> dict:
        """Study-averaged paper reference on the frame's cells: 'empirical_best' (bootstrap floor) or 'uniform01'
        (W1 to the continuous U(0, 1), the paper's Uniform Guess)."""
        col = f"ref_{which}"
        if col not in frame.columns:
            raise KeyError(f"{col} not in frame; which = 'empirical_best' | 'uniform01'")
        test = frame[frame["split"] == "test"] if "split" in frame.columns else frame
        return {"metric": "w1", "reference": which,
                **_study_avg(test[col].to_numpy(float), test["group"].to_numpy(), B, seed)}
