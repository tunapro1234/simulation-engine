"""gss: GSS panels, a person's attitude answer in the next wave (old waves -> new wave), LLM-free baselines.

Ported from the dongu branch (`hedef_gss.py`, preregistration `onkayit/gss-dalga.md`). Train on the old panels
(2006, 2008, 2010; wave 1 -> wave 3, four years), test on the 2016-20 panel (2016 or 2018 answer -> 2020 answer,
panstat == 1). Unit of evaluation = target item; score = between-person Pearson r per item, Fisher z, mean over items
(higher is better), bootstrap over items.
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import pandas as pd

from ...types import TaskData
from ..base import BenchTask, Metric, need, register
from ..stats import bootstrap_mean, fisher_z, pearson

OLD = ["panel2006", "panel2008", "panel2010"]
NEW = "panel2016_2020"
DEFAULT_CFG = {"yontem": "pca_ridge", "alfa": 100.0, "eksik_bayrak": True, "pca_r": 20, "kendi_agirlik": 0.0,
               "hgb_iter": 100, "hgb_lr": 0.1, "hgb_yaprak": 15}
# Preregistered item rules (gss-dalga.md): factual/demographic and nominal-coded items are not targets (stay features).
OLGU = set("""sex spaneng region degree spdeg childs earnrs marital widowed divorce evwork spevwork wrkgovt wrkslf spwrkslf
wksub wksubs wksup wksups partfull localnum unemp union hunt dwelown raclive racwork xmovie owngun pistol rifle shotgun rowngun
wrkstat spwrksta feeused reborn savesoul""".split())
NOMINAL = set("courts divlaw finalter racopen kidssol chldidel letin1a aged fair helpful trust".split())
EKSIK_SAY = {"partyid": [7]}  # 7 = other party: unordered, treated as missing


def parca(unit: str) -> str:
    """dongu v1.1 dev partition of a unit by fixed hash: 'climb' (70%) or 'select' (30%)."""
    return "select" if int(hashlib.sha256(f"dongu-v1.1:{unit}".encode()).hexdigest()[:8], 16) % 10 < 3 else "climb"


def _roots(d: pd.DataFrame) -> dict[str, set]:
    b: dict[str, set] = {}
    for c in d.columns:
        m = re.match(r"(.*)_(\d[ab]?)$", str(c))
        if m:
            b.setdefault(m.group(1), set()).add(m.group(2))
    return b


def select_items(old: dict, new: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(candidate feature items, target items); rule from the preregistration (hedef_gss.maddeler)."""
    y1 = new[new["panstat"] == 1]
    common = {k for k, v in _roots(new).items() if {"1a", "1b", "2"} <= v}
    for d in old.values():
        common &= {k for k, v in _roots(d).items() if {"1", "3"} <= v}
    cand = []
    for k in sorted(common):
        y, x = y1[f"{k}_2"], y1[f"{k}_1a"].where(y1["samptype"] == 2016, y1[f"{k}_1b"])
        nt = int((y.notna() & x.notna()).sum())
        nd = sum(int((d[f"{k}_1"].notna() & d[f"{k}_3"].notna()).sum()) for d in old.values())
        vs = set(y.dropna().unique())
        vo = set().union(*[set(d[f"{k}_3"].dropna().unique()) for d in old.values()])
        if nt >= 200 and nd >= 300 and 2 <= len(vs) <= 11 and vs <= vo:
            cand.append(k)
    return cand, [k for k in cand if k not in OLGU and k not in NOMINAL]


def wave_pairs(old: dict, new: pd.DataFrame, cand: list[str]) -> dict:
    """{panel: (X previous wave [cand + age], Y next wave [cand])}; 'new': (X, Y, cohort)."""
    oz = cand + ["age"]
    P = {}
    for a, d in old.items():
        d = d[d[[f"{k}_3" for k in cand]].notna().any(axis=1)]
        X = pd.DataFrame({k: d[f"{k}_1"].to_numpy(float) for k in oz})
        Y = pd.DataFrame({k: d[f"{k}_3"].to_numpy(float) for k in cand})
        P[a] = (X, Y)
    y1 = new[new["panstat"] == 1]
    s16 = (y1["samptype"] == 2016).to_numpy()
    X = pd.DataFrame({k: np.where(s16, y1[f"{k}_1a"], y1[f"{k}_1b"]).astype(float) for k in oz})
    Y = pd.DataFrame({k: y1[f"{k}_2"].to_numpy(float) for k in cand})
    P["new"] = (X, Y, pd.Series(np.where(s16, "2016", "2018")))
    for k, codes in EKSIK_SAY.items():
        for v in P.values():
            for M in v[:2]:
                if k in M:
                    M.loc[M[k].isin(codes), k] = np.nan
    return P


def _design(Xtr, Xte, cfg):
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1).fillna(1)
    A, B = ((Xtr - mu) / sd), ((Xte - mu) / sd)
    Ea, Eb = A.isna().astype(float), B.isna().astype(float)
    A, B = A.fillna(0.0), B.fillna(0.0)
    if cfg["eksik_bayrak"]:
        A, B = pd.concat([A, Ea.add_suffix("__e")], axis=1), pd.concat([B, Eb.add_suffix("__e")], axis=1)
    return A, B


def predict_items(Xtr, Ytr, Xte, targets, cfg):
    """Port of hedef_gss.tahmin: wide predictions (Xte rows x targets). cfg['yontem'] in
    medyan | kendi (persistence: own previous answer) | ridge | pca_ridge | hgb."""
    from sklearn.linear_model import Ridge
    cfg = {**DEFAULT_CFG, **cfg}
    P = pd.DataFrame(np.nan, index=Xte.index, columns=targets)
    y_mean = Ytr[targets].mean()
    own = {k: Xte[k].fillna(y_mean[k]).to_numpy() for k in targets}
    y = cfg["yontem"]
    if y == "medyan":
        for k in targets:
            P[k] = float(Ytr[k].median())
        return P
    if y == "kendi":
        for k in targets:
            P[k] = own[k]
        return P
    if y in ("ridge", "pca_ridge"):
        A, B = _design(Xtr, Xte, cfg)
        if y == "pca_ridge":
            from sklearn.decomposition import PCA
            main = [c for c in A.columns if not c.endswith("__e")]
            pca = PCA(n_components=min(int(cfg["pca_r"]), len(main)), random_state=0).fit(A[main].to_numpy())
            Za, Zb = pca.transform(A[main].to_numpy()), pca.transform(B[main].to_numpy())
        for k in targets:
            m = Ytr[k].notna().to_numpy()
            if y == "ridge":
                Fa, Fb = A.to_numpy(), B.to_numpy()
            else:
                ek = [k] + ([f"{k}__e"] if cfg["eksik_bayrak"] else [])
                Fa, Fb = np.hstack([Za, A[ek].to_numpy()]), np.hstack([Zb, B[ek].to_numpy()])
            r = Ridge(alpha=float(cfg["alfa"])).fit(Fa[m], Ytr[k].to_numpy()[m])
            P[k] = r.predict(Fb)
    elif y == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        A, B = Xtr.to_numpy(float), Xte.to_numpy(float)
        for k in targets:
            m = Ytr[k].notna().to_numpy()
            Am = A[m]
            live = np.array([np.unique(c[np.isfinite(c)]).size >= 2 for c in Am.T])
            h = HistGradientBoostingRegressor(max_iter=int(cfg["hgb_iter"]), learning_rate=float(cfg["hgb_lr"]),
                                              max_leaf_nodes=int(cfg["hgb_yaprak"]), random_state=0)
            h.fit(Am[:, live], Ytr[k].to_numpy()[m])
            P[k] = h.predict(B[:, live])
    else:
        raise ValueError(f"unknown method {y!r}")
    w = float(cfg["kendi_agirlik"])
    if w:
        for k in targets:
            P[k] = w * own[k] + (1 - w) * P[k].to_numpy()
    return P


def _item_r(frame: pd.DataFrame, col: str, min_n: int = 30) -> pd.Series:
    out = {}
    for k, g in frame.groupby("item", sort=False):
        t, p = g["truth"].to_numpy(float), g[col].to_numpy(float)
        m = np.isfinite(t) & np.isfinite(p)
        out[k] = pearson(p[m], t[m]) if m.sum() >= min_n else float("nan")
    return pd.Series(out)


def _mean_z(task, frame, col, B, seed):
    r = _item_r(frame, col).fillna(0.0)  # constant prediction: r undefined, preregistered as 0
    return bootstrap_mean(fisher_z(r.to_numpy()), B=B, seed=seed)


def _corr(task, frame, col, B, seed):
    z = _mean_z(task, frame, col, B, seed)
    return {k: (float(np.tanh(v)) if k in ("value", "lo", "hi") else v) for k, v in z.items()}


@register()
class GSSWave(BenchTask):
    """Options: holdout_panel=None (default: train old panels, test 2016-20 panel) or one of panel2006/2008/2010
    (leave-one-old-panel-out development fold); items='all'|'climb'|'select' (dongu dev partition by item hash)."""

    name = "gss"
    kind = "continuous"
    data_subdir = "gss"
    layout = """
    <root>/gss/panel2006.parquet, panel2008.parquet, panel2010.parquet   GSS panel files (NORC Stata -> parquet), one
        row per respondent, columns <var>_1, <var>_2, <var>_3 (wave suffix)
    <root>/gss/panel2016_2020.parquet   columns <var>_1a, <var>_1b (2016 / 2018 sample), <var>_2 (2020), panstat, samptype
    """
    source = "talp-simulation-hackathon-2026 uretim/dongu/kod/hedef_gss.py; onkayit/gss-dalga.md"
    metrics = (Metric("mean_z", True, _mean_z, "mean over items of Fisher z of between-person Pearson r"),
               Metric("corr", True, _corr, "tanh(mean z)"))

    def _load(self, root: str) -> TaskData:
        old = {a: pd.read_parquet(need(os.path.join(root, f"{a}.parquet"))) for a in OLD}
        new = pd.read_parquet(need(os.path.join(root, f"{NEW}.parquet")))
        cand, targets = select_items(old, new)
        items = self.options.get("items", "all")
        if items in ("climb", "select"):
            targets = [k for k in targets if parca(k) == items]
        P = wave_pairs(old, new, cand)
        hold = self.options.get("holdout_panel")
        if hold is None:
            tr = OLD
            Xte, Yte, cohort = P["new"]
            test_name = "panel2016_2020"
        else:
            if hold not in OLD:
                raise ValueError(f"holdout_panel must be one of {OLD}")
            tr = [a for a in OLD if a != hold]
            Xte, Yte = P[hold]
            cohort = pd.Series([hold] * len(Xte))
            test_name = hold
        Xtr = pd.concat([P[a][0] for a in tr], ignore_index=True)
        Ytr = pd.concat([P[a][1] for a in tr], ignore_index=True)
        tr_panel = np.concatenate([[a] * len(P[a][0]) for a in tr])
        train = _long(Ytr[targets], [f"{a}:{i}" for a, i in zip(tr_panel, _within(tr_panel))])
        test = _long(Yte[targets], [f"{test_name}:{i}" for i in range(len(Yte))], cohort.to_numpy())
        meta = {"split": f"train {'+'.join(tr)} (wave 1 -> 3); test {test_name}", "targets": targets, "features": cand + ["age"],
                "X_train": Xtr, "Y_train": Ytr, "X_test": Xte, "n_train_persons": len(Xtr), "n_test_persons": len(Xte),
                "source": "GSS panels (NORC); terms: free, attribution required, no redistribution"}
        return TaskData(train=train, test=test, meta=meta)

    def predict_wide(self, data: TaskData, **cfg) -> pd.DataFrame:
        return predict_items(data.meta["X_train"], data.meta["Y_train"], data.meta["X_test"], data.meta["targets"], cfg)

    def baselines(self, data: TaskData) -> dict:
        out = {}
        for name, cfg in (("median", {"yontem": "medyan"}), ("persistence", {"yontem": "kendi"}),
                          ("ridge", {"yontem": "ridge"}), ("pca_ridge", {"yontem": "pca_ridge"})):
            out[name] = rows_from_wide(self.predict_wide(data, **cfg), data.test)
        return out

    def _frame(self, data: TaskData) -> pd.DataFrame:
        t = data.test
        return pd.DataFrame({"unit": t["person"] + "|" + t["item"], "group": t["person"], "item": t["item"],
                             "segment": t["cohort"], "truth": t["truth"].to_numpy(float)})


def _within(panels):
    out, seen = [], {}
    for a in panels:
        out.append(seen.get(a, 0))
        seen[a] = seen.get(a, 0) + 1
    return out


def _long(Y: pd.DataFrame, persons, cohort=None) -> pd.DataFrame:
    rows = []
    persons = np.asarray(persons)
    for k in Y.columns:
        v = Y[k].to_numpy(float)
        m = np.isfinite(v)
        d = pd.DataFrame({"person": persons[m], "row": np.flatnonzero(m), "item": k, "truth": v[m]})
        if cohort is not None:
            d["cohort"] = np.asarray(cohort)[m]
        rows.append(d)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["person", "row", "item", "truth"])
    if "cohort" not in out.columns:
        out["cohort"] = ""
    return out


def rows_from_wide(P: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Pick wide predictions (positional rows x items) for the long test rows."""
    cols = {c: i for i, c in enumerate(P.columns)}
    M = P.to_numpy(float)
    return M[test["row"].to_numpy(int), test["item"].map(cols).to_numpy(int)]


def cross_panel_dev(data_root: str | None = None, cfg: dict | None = None, items: str = "climb", B: int = 1000,
                    seed: int = 42) -> dict:
    """dongu development score: each old panel held out in turn (trained on the other two), per-item r averaged
    over folds in Fisher z, then mean z over items with an item bootstrap. Never touches the 2016-20 panel."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    zs: dict[str, list] = {}
    for hold in OLD:
        task = GSSWave(holdout_panel=hold, items=items)
        data = task.load(data_root)
        p = rows_from_wide(task.predict_wide(data, **cfg), data.test)
        fr = task.frame(data, {"m": p})
        r = _item_r(fr, "pred_m")
        for k, v in r.items():
            if np.isfinite(v) or cfg["yontem"] == "medyan":
                zs.setdefault(k, []).append(0.0 if not np.isfinite(v) else v)
    per = {k: float(np.tanh(np.mean(fisher_z(v)))) for k, v in zs.items() if v}
    z = fisher_z(np.array(list(per.values())))
    res = bootstrap_mean(z, B=B, seed=seed)
    return {"mean_z": res, "corr": {k: (float(np.tanh(v)) if k in ("value", "lo", "hi") else v) for k, v in res.items()},
            "per_item_r": per}
