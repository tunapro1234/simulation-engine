"""subpop_opinionqa / global_opinionqa: answer distributions of demographic groups (SubPOP-OpinionQA, 22 or 61 US
groups on Pew ATP questions) and of countries (GlobalOpinionQA, Pew GAS + WVS), one row per (question, group).

    from simengine.bench.tasks import subpop as sp
    task = sp.SubPOPOpinionQA()                    # or sp.GlobalOpinionQA(); setting="B" (default) | "A"
    data = task.load("<catalogue>")                # or $SIMENGINE_DATA; holds subpop-opinionqa/ (globalopinionqa/)
    frame = task.run(data=data)                    # uniform, overall, nearest_group (+ methods={...})
    task.score_all(frame)                          # wd / tvd (SubPOP); tvd / js_sim (GlobalOpinionQA)
    task.others(data, item, group, attribute)      # rows a setting-B prediction for (item, group) may read

Split: questions are hashed into test (default 50%) and train; `split="dev"` hashes train again, so dev never reads a
test question. Setting A: only train questions are known (uniform is SubPOP's "upper bound" baseline). Setting B
(group held out): for target (q, g) q's population marginal and the distributions on q of the groups outside g's
block are also known, never g's own row. Block (`block` column): the attribute for SubPOP (Overall = weighted sum of
an attribute's groups, so with g's block mates known g would follow by subtraction), the country without its
"(Non-national sample)" suffix for GlobalOpinionQA. `overall`: Pew's weighted Overall row for SubPOP; the unweighted
mean of the other blocks' rows for GlobalOpinionQA. `peers="all"` opens every other group (leaks through that identity).

Metrics. SubPOP (Suh et al. 2025, arXiv:2502.16761, `subpop.utils.survey_utils.ordinal_emd`): Wasserstein-1 on the
hand-made OpinionQA ordinal values divided by their range, options with a negative ordinal dropped, NaN when every
ordinal is equal; averaged over all (question, group) rows (Table 1). A prediction with no mass on the ordered options
(or a NaN) counts as uniform here; SubPOP's code returns NaN, which would drop the row. GlobalOpinionQA (Durmus et al. 2023,
arXiv:2306.16388, sec. 2.3): similarity 1 - Jensen-Shannon distance over all options incl. DK/Refused (log base 2 here;
the paper does not state the base). Total variation distance is reported for both. CIs: question-cluster bootstrap.
For `opinionqa_groups` (wave-out, knn/position baselines on the larger OpinionQA pool) see `opinionqa.py`.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, Metric, need, register
from ..stats import cluster_bootstrap

TEST_SHARE = 0.5
SALT = "opinion_dist-v1"
COLUMNS = ["item", "qkey", "source", "attribute", "group", "question", "options", "ordinal", "dist", "overall",
           "block"]
PEERS = ("blocks", "all")


# ---------------------------------------------------------------------- distances
def _norm(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 0, None)
    s = p.sum()
    return p / s if s > 0 else np.full(len(p), 1 / len(p))


def ordinal_wd(p, q, ordinal) -> float:
    """SubPOP `ordinal_emd`: W1 between p and q on ordinal values / (max - min); negative ordinals (no order, e.g.
    'Not sure') dropped and the rest renormalised; NaN if fewer than two distinct ordinals remain."""
    o = np.asarray(ordinal, float)
    keep = o >= 0
    o = o[keep]
    if len(o) < 2 or o.max() == o.min():
        return float("nan")
    srt = np.argsort(o, kind="stable")
    p, q = _norm(np.asarray(p, float)[keep])[srt], _norm(np.asarray(q, float)[keep])[srt]
    w = np.diff(o[srt]) / (o.max() - o.min())
    return float(np.abs(np.cumsum(p)[:-1] - np.cumsum(q)[:-1]) @ w)


def tvd(p, q) -> float:
    return float(0.5 * np.abs(_norm(p) - _norm(q)).sum())


def js_distance(p, q, base: float = 2.0) -> float:
    """Jensen-Shannon distance (square root of the JS divergence), log base `base` (2 -> bounded by 1)."""
    p, q = _norm(p), _norm(q)
    m = 0.5 * (p + q)

    def kl(a):
        nz = a > 0
        return float(np.sum(a[nz] * np.log(a[nz] / m[nz])))
    return float(np.sqrt(max(0.5 * kl(p) + 0.5 * kl(q), 0.0) / np.log(base)))


def pairwise(P: np.ndarray, ordinal=None) -> np.ndarray:
    """n x n distance matrix between the rows of P (distributions of one question): ordinal_wd if `ordinal` is given,
    else total variation."""
    P = np.apply_along_axis(_norm, 1, np.asarray(P, float))
    if ordinal is None:
        return 0.5 * np.abs(P[:, None, :] - P[None, :, :]).sum(-1)
    o = np.asarray(ordinal, float)
    keep = o >= 0
    o = o[keep]
    if len(o) < 2 or o.max() == o.min():
        return np.full((len(P), len(P)), np.nan)
    srt = np.argsort(o, kind="stable")
    P = np.apply_along_axis(_norm, 1, P[:, keep])[:, srt]
    C = np.cumsum(P, 1)[:, :-1]
    w = np.diff(o[srt]) / (o.max() - o.min())
    return np.abs(C[:, None, :] - C[None, :, :]) @ w


# ---------------------------------------------------------------------- files
def _lit(s):
    return ast.literal_eval(s) if isinstance(s, str) else s


def load_subpop_opinionqa(root: str, groups: str = "22") -> pd.DataFrame:
    """Long table (COLUMNS) from SubPOP's `opinionqa.csv`; groups '22' (SubPOP training groups) or '61' (all).
    A qkey asked with several texts (IDIMPORT_W43) becomes items `<qkey>#<k>`. Questions whose ordinals are all equal
    are dropped (SubPOP scores them NaN)."""
    meta = os.path.join(root, "data", "subpopulation_metadata", f"demographics_{groups}.csv")
    gm = pd.read_csv(need(meta))
    gset = {(r.attribute, g) for r in gm.itertuples() for g in _lit(r.group)}
    o = pd.read_csv(need(os.path.join(root, "data", "opinionqa", "processed", "opinionqa.csv")))
    o["question"] = o["question"].astype(str)
    nt = o.groupby("qkey")["question"].transform("nunique")
    k = o.groupby("qkey")["question"].transform(lambda s: pd.factorize(s)[0])
    o["item"] = np.where(nt > 1, o["qkey"] + "#" + k.astype(str), o["qkey"])
    rows = []
    for it, g in o.groupby("item", sort=True):
        x0 = g.iloc[0]
        ordn = [float(v) for v in _lit(x0.ordinal)]
        pos = [v for v in ordn if v >= 0]
        if len(pos) < 2 or max(pos) == min(pos):
            continue
        opts = [str(z) for z in _lit(x0.options) if z != "Refused"][:len(ordn)]
        ov = g[g.attribute == "Overall"]
        ov = _norm(_lit(ov.iloc[0].responses)) if len(ov) else None
        seen = set()
        for x in g.itertuples():
            key = (x.attribute, x.group)
            if key not in gset or key in seen:
                continue
            p = np.asarray(_lit(x.responses), float)
            if len(p) != len(ordn) or p.sum() <= 0:
                continue
            seen.add(key)
            rows.append({"item": it, "qkey": x.qkey, "source": str(x.qkey).rsplit("_", 1)[-1], "attribute": x.attribute,
                         "group": x.group, "question": x.question, "options": opts, "ordinal": ordn, "dist": _norm(p),
                         "overall": ov, "block": x.attribute})
    return pd.DataFrame(rows, columns=COLUMNS)


def _selections(s: str) -> dict:
    s = s.strip()
    if s.startswith("defaultdict"):
        s = s[s.index("{"):s.rindex("}") + 1]
    return ast.literal_eval(s)


def load_global_opinionqa(root: str) -> pd.DataFrame:
    """Long table (COLUMNS) from GlobalOpinionQA `global_opinions.csv`; group = country, attribute = 'country',
    item = '<GAS|WVS>:<sha256(question)[:12]>', block = country without '(Non-national sample)'. Rows whose shares
    sum to 0 are dropped (Durmus: n_c|q > 0); the others are renormalised. No ordinal (options are treated as
    categories); `overall` is filled per row in setting B."""
    d = pd.read_csv(need(os.path.join(root, "data", "global_opinions.csv")))
    rows = []
    for x in d.itertuples():
        opts = [str(z) for z in _lit(x.options)]
        it = f"{x.source}:{hashlib.sha256(str(x.question).encode()).hexdigest()[:12]}"
        for c, p in sorted(_selections(x.selections).items()):
            p = np.asarray(p, float)
            if len(p) != len(opts) or not np.isfinite(p).all() or p.sum() <= 0:
                continue
            rows.append({"item": it, "qkey": it, "source": x.source, "attribute": "country", "group": c,
                         "question": x.question, "options": opts, "ordinal": None, "dist": _norm(p), "overall": None,
                         "block": re.sub(r"\s*\(Non-national sample\)$", "", c)})
    return pd.DataFrame(rows, columns=COLUMNS)


def split_items(items, test_share: float = TEST_SHARE, salt: str = SALT) -> set:
    """Items on the test side of a fixed sha256 hash partition (all groups of a question on one side)."""
    cut = int(test_share * 10_000)
    return {i for i in items if int(hashlib.sha256(f"{salt}:{i}".encode()).hexdigest()[:8], 16) % 10_000 < cut}


# ---------------------------------------------------------------------- baselines
def _gkey(df: pd.DataFrame) -> pd.Series:
    return df["attribute"] + ":" + df["group"]


def group_distances(train: pd.DataFrame, ordinal: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(mean distance, shared-question count) between every pair of groups over the train questions both answered.
    Distance = ordinal_wd if `ordinal` (SubPOP) else total variation. Index/columns are 'attribute:group'."""
    keys = sorted(_gkey(train).unique())
    ix = {k: i for i, k in enumerate(keys)}
    S, N = np.zeros((len(keys),) * 2), np.zeros((len(keys),) * 2)
    for _, g in train.assign(_k=_gkey(train)).groupby("item", sort=False):
        if len(g) < 2:
            continue
        D = pairwise(np.stack(g["dist"].to_numpy()), g["ordinal"].iloc[0] if ordinal else None)
        if not np.isfinite(D).all():
            continue
        j = np.array([ix[k] for k in g["_k"]])
        S[np.ix_(j, j)] += D
        N[np.ix_(j, j)] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(N > 0, S / N, np.nan)
    return pd.DataFrame(M, keys, keys), pd.DataFrame(N, keys, keys).astype(int)


def loo_marginal(test: pd.DataFrame, by_block: bool = True) -> list:
    """Unweighted mean of the other rows' distributions on the same item, rows of the target's block left out if
    `by_block` (else only the row itself); uniform if nothing is left."""
    out = [None] * len(test)
    t = test.reset_index(drop=True)
    blk = t["block"].to_numpy() if by_block else np.arange(len(t))
    for _, g in t.groupby("item", sort=False):
        P, b = np.stack(g["dist"].to_numpy()), blk[g.index]
        for i, r in enumerate(g.index):
            m = b != b[i]
            out[r] = _norm(P[m].mean(0)) if m.any() else np.full(P.shape[1], 1 / P.shape[1])
    return out


def nearest_group(data: TaskData, ordinal: bool = True, min_shared: int = 5, peers: str = "blocks"
                  ) -> tuple[list, int]:
    """Setting B: predict (q, g) with the distribution on q of the group h closest to g on train questions
    (>= min_shared shared questions; ties -> alphabetical), h outside g's block (peers='blocks') or any h != g
    (peers='all'). Falls back to the `overall` column. Returns (preds, n_fallback)."""
    D, N = group_distances(data.train, ordinal)
    D = D.where(N >= min_shared)
    t = data.test.reset_index(drop=True).assign(_k=lambda d: _gkey(d))
    out, nfb = [None] * len(t), 0
    for _, g in t.groupby("item", sort=False):
        have = dict(zip(g["_k"], g["dist"]))
        blk = dict(zip(g["_k"], g["block"])) if peers == "blocks" else {k: k for k in have}
        for r, k, ov in zip(g.index, g["_k"], g["overall"]):
            cand = [h for h in have if blk[h] != blk[k] and k in D.index and h in D.columns
                    and np.isfinite(D.at[k, h])]
            if cand:
                out[r] = have[min(cand, key=lambda h: (D.at[k, h], h))]
            else:
                nfb += 1
                out[r] = ov if ov is not None else np.full(len(have[k]), 1 / len(have[k]))
    return out, nfb


# ---------------------------------------------------------------------- metrics
def _rows(frame, col, fn) -> np.ndarray:
    bad = [i for i, (t, p) in enumerate(zip(frame["truth"], frame[col])) if np.size(p) != len(t)]
    if bad:  # tvd / js would broadcast a short vector silently
        i = bad[0]
        raise ValueError(f"{col}: row {i} has {np.size(frame[col].iloc[i])} probabilities, the question has "
                         f"{len(frame['truth'].iloc[i])} options ({len(bad)} such rows)")
    if "ordinal" in frame.columns and fn is ordinal_wd:
        return np.array([fn(t, p, o) for t, p, o in zip(frame["truth"], frame[col], frame["ordinal"])], float)
    return np.array([fn(t, p) for t, p in zip(frame["truth"], frame[col])], float)


def _scorer(fn, sign: float = 1.0, shift: float = 0.0):
    def score(task, frame, col, B, seed):
        v = shift + sign * _rows(frame, col, fn)
        ok = np.isfinite(v)
        v, items = v[ok], frame["item"].to_numpy()[ok]
        if not len(v):
            return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
        return cluster_bootstrap(lambda ix: v[ix].mean(), items, B=B, seed=seed)
    return score


WD = Metric("wd", False, _scorer(ordinal_wd), "SubPOP ordinal Wasserstein-1 / ordinal range, mean over (question, "
            "group) rows; question-cluster bootstrap")
TVD = Metric("tvd", False, _scorer(tvd), "total variation distance, mean over rows; question-cluster bootstrap")
JS_SIM = Metric("js_sim", True, _scorer(js_distance, -1.0, 1.0), "1 - Jensen-Shannon distance (base 2), Durmus et al. "
                "2023; mean over rows; question-cluster bootstrap")


# ---------------------------------------------------------------------- tasks
class OpinionDistTask(BenchTask):
    """Shared split / settings / baselines of the two group-distribution tasks."""

    kind = "distribution"
    ordinal = True
    accepted_options = {
        "split": '"test" | "dev" (default "test")',
        "setting": '"A" (only train questions known) | "B" (other groups on the question known) (default "B")',
        "test_share": "float in (0, 1), share of questions hashed to test (default 0.5)",
        "min_shared": "int, shared train questions needed to call a group nearest (default 5)",
        "peers": '"blocks" (setting B reads groups outside the target\'s block) | "all" (every other group; leaky) '
                 '(default "blocks")',
    }

    def _read(self, root: str) -> pd.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError

    def _load(self, root: str) -> TaskData:
        df = self._read(root)
        split = self.options.get("split", "test")
        setting = self.options.get("setting", "B")
        share = float(self.options.get("test_share", TEST_SHARE))
        peers = self.options.get("peers", "blocks")
        if peers not in PEERS:
            raise ValueError("peers must be 'blocks' or 'all'")
        if split not in ("test", "dev"):
            raise ValueError("split must be 'test' or 'dev'")
        if setting not in ("A", "B"):
            raise ValueError("setting must be 'A' or 'B'")
        if not 0 < share < 1:
            raise ValueError("test_share must be in (0, 1)")
        # only questions answered by >= 2 groups can be compared between groups
        n = df.groupby("item")["group"].transform("size")
        df = df[n >= 2].reset_index(drop=True)
        items = sorted(df["item"].unique())
        te = split_items(items, share)
        if split == "dev":
            rest = [i for i in items if i not in te]
            te = split_items(rest, share, SALT + ":dev")
            df = df[~df["item"].isin(split_items(items, share))]
        train = df[~df["item"].isin(te)].reset_index(drop=True)
        test = df[df["item"].isin(te)].reset_index(drop=True)
        if setting == "A":
            test = test.assign(overall=None)
        else:  # the file's population row if it has one (SubPOP), else the mean of the readable rows
            loo = loo_marginal(test, peers == "blocks")
            test = test.assign(overall=[o if o is not None else m for o, m in zip(test["overall"], loo)])
        meta = {"split": f"{split}: sha256 question hash ({SALT}{':dev' if split == 'dev' else ''}), "
                         f"test_share={share}; all groups of a question on one side",
                "setting": setting, "peers": peers, "groups": sorted(set(zip(df["attribute"], df["group"]))),
                "n_items": int(test["item"].nunique()), "n_train_items": int(train["item"].nunique())}
        return TaskData(train=train, test=test, meta=meta)

    @staticmethod
    def others(data: TaskData, item: str, group: str, attribute: str | None = None) -> pd.DataFrame:
        """Test rows a setting-B prediction for (item, group) may read: same item, outside the target's block
        (peers='blocks') or every other row (peers='all'). `attribute` is needed when a group label occurs under
        two attributes (SubPOP 61: 'Other' in RACE and RELIG). Empty in setting A."""
        t = data.test[data.test["item"] == item]
        me = (t["group"] == group) & ((t["attribute"] == attribute) if attribute is not None else True)
        if me.sum() != 1:
            raise ValueError(f"{item!r}: {int(me.sum())} rows match group {group!r}"
                             + ("" if attribute is not None else "; pass attribute="))
        if data.meta.get("setting") != "B":
            return t.iloc[:0]
        if data.meta.get("peers", "blocks") == "all":
            return t[~me]
        return t[t["block"] != t.loc[me, "block"].iloc[0]]

    def baselines(self, data: TaskData) -> dict:
        m = [len(o) for o in data.test["options"]]
        out = {"uniform": [np.full(k, 1 / k) for k in m]}
        if data.meta.get("setting") == "B":
            out["overall"] = [np.asarray(o, float) for o in data.test["overall"]]
            out["nearest_group"], nfb = nearest_group(data, self.ordinal, int(self.options.get("min_shared", 5)),
                                                      data.meta.get("peers", "blocks"))
            data.meta["nearest_group_fallbacks"] = nfb
        return out

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        return pd.DataFrame({"unit": t["item"] + "|" + t["attribute"] + ":" + t["group"], "item": t["item"],
                             "group": t["item"], "segment": t["attribute"], "seg_group": t["group"],
                             "seg_source": t["source"],
                             "truth": pd.Series([np.asarray(x, float) for x in t["dist"]], dtype=object),
                             "ordinal": pd.Series(list(t["ordinal"]), dtype=object)})


@register()
class SubPOPOpinionQA(OpinionDistTask):
    """SubPOP-OpinionQA: 494 Pew ATP questions x 22 (or 61) groups. Options as OpinionDistTask plus groups='22'|'61'."""

    name = "subpop_opinionqa"
    data_subdir = "subpop-opinionqa"
    accepted_options = {**OpinionDistTask.accepted_options,
                        "groups": '"22" (SubPOP groups) | "61" (all OpinionQA groups) (default "22")'}
    layout = """
    <root>/subpop-opinionqa/data/opinionqa/processed/opinionqa.csv              qkey, attribute, group, responses,
                                                                              refusal_rate, ordinal, question, options
    <root>/subpop-opinionqa/data/subpopulation_metadata/demographics_22.csv    attribute, group (list) - 22 groups
    <root>/subpop-opinionqa/data/subpopulation_metadata/demographics_61.csv    (groups="61")
    """
    source = ("Suh et al. 2025, SubPOP (arXiv:2502.16761), github JosephJeesungSuh/subpop data/ (code BSD-3); "
              "distributions from Pew ATP via OpinionQA (Santurkar et al. 2023); Pew terms: research, attribution")
    metrics = (WD, TVD, JS_SIM)

    def _read(self, root: str) -> pd.DataFrame:
        g = str(self.options.get("groups", "22"))
        if g not in ("22", "61"):
            raise ValueError("groups must be '22' or '61'")
        return load_subpop_opinionqa(root, g)


@register()
class GlobalOpinionQA(OpinionDistTask):
    """GlobalOpinionQA: 2,556 Pew GAS / WVS questions x countries (138 country labels). Setting B's `overall` is the
    unweighted mean of the other countries on the question (a country's '(Non-national sample)' row is its block mate,
    so neither is read for the other)."""

    name = "global_opinionqa"
    data_subdir = "globalopinionqa"
    ordinal = False
    layout = """
    <root>/globalopinionqa/data/global_opinions.csv   question, selections ({country: shares}), options, source
    """
    source = ("Durmus et al. 2023 (arXiv:2306.16388), HF Anthropic/llm_global_opinions; CC-BY-NC-SA-4.0; "
              "underlying Pew Global Attitudes and World Values Survey")
    metrics = (TVD, JS_SIM)

    def _read(self, root: str) -> pd.DataFrame:
        return load_global_opinionqa(root)
