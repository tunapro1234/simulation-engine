"""opinionqa_groups: answer distribution of 22 US demographic groups on Pew ATP questions, survey wave held out.

Ported from the dongu branch (`kosucu.py` oqa_*, preregistration `onkayit/oqa-grup.md`). Unit = question; per question
the Wasserstein-1 distance on 0-1 option positions is averaged over its groups, then averaged over questions with a
question bootstrap (lower is better; SubPOP standard). Wave-out rule: a prediction for a question of wave w may use
only questions of other waves (`OpinionQAGroups.history(data, wave)` returns exactly that).

Setting A (default): only other-wave questions are known. Setting B: the target question's Overall (population)
distribution is also known (`data.test['overall']`), enabling the `overall` / `sibling` baselines.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, Metric, need, register
from ..stats import bootstrap_mean, wasserstein_ordinal
from .gss import parca


def _scale(o):
    o = np.asarray(o, float)
    return (o - o.min()) / (o.max() - o.min())


def _groups(sub_root: str) -> list[tuple[str, str]]:
    g22 = pd.read_csv(need(os.path.join(sub_root, "subpopulation_metadata", "demographics_22.csv")))
    return sorted((r.attribute, g) for r in g22.itertuples() for g in ast.literal_eval(r.group))


def load_questions(root: str):
    """(pool, test, groups): {qkey: dict(wave, text, options, pos, dist{(attr, grp): p}, overall)}."""
    oqa = os.path.join(root, "opinionqa", "islenmis")
    sub = os.path.join(root, "subpop-opinionqa", "data")
    groups = _groups(sub)
    gset = set(groups)
    s = pd.read_parquet(need(os.path.join(oqa, "sorular.parquet")))
    d = pd.read_parquet(need(os.path.join(oqa, "dagilim.parquet")))
    q_info = s.set_index("qkey")
    pool, test = {}, {}
    for q, grp in d[~d.qkey.isin(s[s.dis500].qkey)].groupby("qkey"):
        r = q_info.loc[q]
        ordn = np.asarray(r.ordinal, float)
        m = len(ordn)
        if m < 2 or ordn.max() == ordn.min():
            continue
        dist, ov = {}, None
        for x in grp.itertuples():
            p = np.asarray(x.responses, float)[:m]  # trailing non-ordinal options (Not sure, ...) dropped, as SubPOP
            if p.sum() <= 0:
                continue
            p = p / p.sum()
            if x.attribute == "Overall":
                ov = p
            elif (x.attribute, x.group) in gset:
                dist[(x.attribute, x.group)] = p
        if ov is None or not dist:
            continue
        pool[q] = dict(wave=r.wave, text=r.question, options=list(r.options)[:m], pos=_scale(ordn), dist=dist, overall=ov)
    o = pd.read_csv(need(os.path.join(sub, "opinionqa", "processed", "opinionqa.csv")))
    wave = q_info["wave"]
    for q, grp in o.groupby("qkey"):
        x0 = grp.iloc[0]
        ordn = np.array(ast.literal_eval(x0.ordinal), float)
        if len(ordn) < 2 or ordn.max() == ordn.min():
            continue
        opts = [z for z in ast.literal_eval(x0.options) if z != "Refused"][:len(ordn)]
        dist, ov = {}, None
        for x in grp.itertuples():
            p = np.array(ast.literal_eval(x.responses), float)
            p = p / p.sum()
            if x.attribute == "Overall":
                ov = p
            elif (x.attribute, x.group) in gset:
                dist[(x.attribute, x.group)] = p
        test[q] = dict(wave=wave.get(q, str(q).rsplit("_", 1)[-1]), text=x0.question, options=opts, pos=_scale(ordn),
                       dist=dist, overall=ov)
    return pool, test, groups


def _long(qs: dict) -> pd.DataFrame:
    rows = []
    for q in sorted(qs):
        v = qs[q]
        for (a, g), p in v["dist"].items():
            rows.append({"qkey": q, "wave": v["wave"], "attribute": a, "group": g, "question": v["text"],
                         "options": v["options"], "pos": v["pos"], "dist": p, "overall": v["overall"]})
    return pd.DataFrame(rows, columns=["qkey", "wave", "attribute", "group", "question", "options", "pos", "dist",
                                       "overall"])


def rebin(p, pos_src, pos_dst):
    """Share mass at source positions linearly onto target positions (both 0-1)."""
    out = np.zeros(len(pos_dst))
    for m, x in zip(p, pos_src):
        j = np.searchsorted(pos_dst, x)
        if j <= 0:
            out[0] += m
        elif j >= len(pos_dst):
            out[-1] += m
        else:
            a, b = pos_dst[j - 1], pos_dst[j]
            w = (x - a) / (b - a)
            out[j - 1] += m * (1 - w)
            out[j] += m * w
    return out


def _qdict(df: pd.DataFrame) -> dict:
    out = {}
    for q, g in df.groupby("qkey", sort=False):
        r = g.iloc[0]
        out[q] = dict(wave=r.wave, text=r.question, options=r.options, pos=np.asarray(r.pos, float),
                      overall=np.asarray(r.overall, float),
                      dist={(x.attribute, x.group): np.asarray(x.dist, float) for x in g.itertuples()})
    return out


def _move(p, src, tgt_pos):
    return p if len(src["pos"]) == len(tgt_pos) and np.allclose(src["pos"], tgt_pos) else rebin(p, src["pos"], tgt_pos)


def predict_question(target: dict, pool: dict, method: str, groups, *, sim: dict | None = None, k: int = 20,
                     gamma: float = 1.0, cache: dict | None = None) -> dict:
    """{group: p} for one target question. pool must already exclude the target's wave.
    method: uniform | position (mean of same-option-count questions per group) | overall | sibling (setting B) | knn."""
    m = len(target["pos"])
    if method == "uniform":
        return {g: np.ones(m) / m for g in groups}
    if method == "overall":
        return {g: target["overall"] for g in groups}
    if method == "sibling":
        out = {}
        for g in groups:
            sib = [target["dist"][h] for h in target["dist"] if h[0] == g[0] and h != g]
            out[g] = np.mean(sib, 0) if sib else target["overall"]
        return out
    cache = {} if cache is None else cache
    key = (target["wave"], m, tuple(np.round(target["pos"], 6)))
    if key not in cache:
        same = [q for q in pool if len(pool[q]["pos"]) == m] or list(pool)
        K = {}
        for g in groups:
            ps = [_move(pool[q]["dist"][g], pool[q], target["pos"]) for q in same if g in pool[q]["dist"]]
            K[g] = np.mean(ps, 0) if ps else np.ones(m) / m
        K["Overall"] = np.mean([_move(pool[q]["overall"], pool[q], target["pos"]) for q in same], 0)
        cache[key] = K
    K = cache[key]
    if method == "position":
        return {g: K[g] for g in groups}
    if method != "knn":
        raise ValueError(f"unknown method {method!r}")
    cand = [q for q in pool if len(pool[q]["pos"]) == m] or list(pool)
    v0 = sim[target["qkey"]]
    s = np.stack([sim[q] for q in cand]) @ v0
    order = np.argsort(-s, kind="stable")[:k]
    w = np.clip(s[order], 1e-6, None) ** gamma
    w = w / w.sum()
    qs = [cand[i] for i in order]
    ov_hat = sum(wi * _move(pool[q]["overall"], pool[q], target["pos"]) for q, wi in zip(qs, w))
    out = {}
    for g in groups:
        ps = [(wi, _move(pool[q]["dist"][g], pool[q], target["pos"])) for q, wi in zip(qs, w) if g in pool[q]["dist"]]
        p = sum(wi * pi for wi, pi in ps) / sum(wi for wi, _ in ps) if ps else ov_hat
        p = np.clip(p, 1e-9, None)
        out[g] = p / p.sum()
    return out


def tfidf_vectors(questions: dict) -> dict:
    """L2-normalised TF-IDF of question text + options (dongu oqa_temsil, tfidf)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    qs = sorted(questions)
    txt = [questions[q]["text"] + " Options: " + "; ".join(questions[q]["options"]) for q in qs]
    X = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, stop_words="english").fit_transform(txt).toarray()
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return {q: X[i] for i, q in enumerate(qs)}


def _per_question(frame, col, fn):
    vals = [fn(np.asarray(t, float), np.asarray(p, float), np.asarray(pos, float))
            for t, p, pos in zip(frame["truth"], frame[col], frame["pos"])]
    return pd.Series(vals).groupby(frame["item"].to_numpy()).mean()


def _wd(task, frame, col, B, seed):
    return bootstrap_mean(_per_question(frame, col, lambda t, p, pos: wasserstein_ordinal(t, p, pos)).to_numpy(), B=B,
                          seed=seed)


def _tvd1(task, frame, col, B, seed):
    return bootstrap_mean(_per_question(frame, col, lambda t, p, pos: 1 - 0.5 * np.abs(t - p).sum()).to_numpy(), B=B,
                          seed=seed)


def _rank_rho(task, frame, col, B, seed):
    """Spearman rho over a question's groups between true and predicted expected position, mean over questions."""
    rh = []
    for _, g in frame.groupby("item", sort=False):
        if len(g) < 3:
            continue
        tr = np.array([np.dot(t, pos) for t, pos in zip(g["truth"], g["pos"])])
        pr = np.array([np.dot(p, pos) for p, pos in zip(g[col], g["pos"])])
        r = pd.Series(tr).corr(pd.Series(pr), method="spearman")
        if np.isfinite(r):
            rh.append(r)
    return bootstrap_mean(rh, B=B, seed=seed)


@register()
class OpinionQAGroups(BenchTask):
    """Options: split='test' (SubPOP OpinionQA test questions; default) | 'dev' (pool questions, dongu dev set);
    items='all'|'climb'|'select' (dev only, dongu hash partition); setting='A'|'B'."""

    name = "opinionqa_groups"
    accepted_options = {
        'split': '"test" | "dev" (default "test")',
        'items': '"all" | "climb" | "select", dev split only (default "all")',
        'setting': '"A" (no Overall given) | "B" (Overall given) (default "A")',
    }
    kind = "distribution"
    data_subdir = None
    layout = """
    <root>/opinionqa/islenmis/sorular.parquet   qkey, wave, question, options, ordinal, dis500 (Pew ATP W26-W92 questions)
    <root>/opinionqa/islenmis/dagilim.parquet   qkey, attribute, group, responses (weighted answer shares per group)
    <root>/subpop-opinionqa/data/subpopulation_metadata/demographics_22.csv   the 22 evaluation groups
    <root>/subpop-opinionqa/data/opinionqa/processed/opinionqa.csv            SubPOP OpinionQA test questions
    """
    source = "talp-simulation-hackathon-2026 uretim/dongu/kod/kosucu.py (oqa_*); onkayit/oqa-grup.md"
    metrics = (Metric("wd", False, _wd, "Wasserstein-1 on 0-1 option positions, mean over groups then questions"),
               Metric("tvd1", True, _tvd1, "1 - total variation distance"),
               Metric("rank_rho", True, _rank_rho, "Spearman over groups of expected position, mean over questions"))

    def _load(self, root: str) -> TaskData:
        pool, test, groups = load_questions(root)
        split = self.options.get("split", "test")
        if split == "test":
            targets = test
        elif split == "dev":
            items = self.options.get("items", "all")
            targets = {q: v for q, v in pool.items() if items == "all" or parca(q) == items}
        else:
            raise ValueError("split must be 'test' or 'dev'")
        setting = self.options.get("setting", "A")
        tdf = _long(targets)
        if setting == "A":
            tdf = tdf.assign(overall=None)
        meta = {"split": f"{split}: targets = {'SubPOP OpinionQA test' if split == 'test' else 'pool'} questions; "
                         "wave-out rule (use only other-wave train questions)", "setting": setting, "groups": groups,
                "n_questions": len(targets), "n_pool": len(pool),
                "source": "Pew ATP via OpinionQA (Santurkar et al. 2023) / SubPOP (Suh et al. 2025); research use"}
        return TaskData(train=_long(pool), test=tdf, meta=meta)

    @staticmethod
    def history(data: TaskData, wave: str) -> pd.DataFrame:
        """Train rows allowed for a target of `wave` (wave-out rule)."""
        return data.train[data.train["wave"] != wave]

    def predict_with(self, data: TaskData, method: str, **kw) -> list:
        pool = _qdict(data.train)
        tq = _qdict(data.test)
        groups = [tuple(g) for g in data.meta["groups"]]
        if method in ("overall", "sibling") and data.meta.get("setting") != "B":
            raise ValueError(f"{method} needs setting='B' (population distribution known)")
        sim = tfidf_vectors({**pool, **tq}) if method == "knn" else None
        by_wave: dict = {}
        preds = {}
        for q in sorted(tq):
            h = dict(tq[q], qkey=q)
            w = h["wave"]
            if w not in by_wave:
                by_wave[w] = ({k: v for k, v in pool.items() if v["wave"] != w}, {})
            p_other, cache = by_wave[w]
            P = predict_question(h, p_other, method, groups, sim=sim, cache=cache, **kw)
            for g in h["dist"]:
                preds[(q, g)] = P[g]
        return [preds[(r.qkey, (r.attribute, r.group))] for r in data.test.itertuples()]

    def baselines(self, data: TaskData) -> dict:
        names = ["uniform", "position", "knn"]
        if data.meta.get("setting") == "B":
            names += ["overall", "sibling"]
        return {n: self.predict_with(data, n) for n in names}

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        return pd.DataFrame({"unit": t["qkey"] + "|" + t["attribute"] + ":" + t["group"], "item": t["qkey"],
                             "group": t["qkey"], "segment": t["attribute"], "seg_wave": t["wave"],
                             "truth": pd.Series([np.asarray(x, float) for x in t["dist"]], dtype=object),
                             "pos": pd.Series([np.asarray(x, float) for x in t["pos"]], dtype=object)})
