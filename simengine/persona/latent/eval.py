"""Scoring of the latent-persona arms: G1 through the TALP talp73 evaluator (subprocess, unchanged), G2 MAD
accuracy, G3 counterfactual price slope, all with person-level bootstrap (B = 1000, default_rng(42)).

    res = score_g1(evaluator, [pred_path], bases=["enc-angc", "gecmis"], sonuc_dir=..., B=1000)
    g2 = score_g2(pred, truth, base=enc_pred)            # MAD accuracy and paired difference CI
    g3 = score_g3(sweep, truth_test)                     # s_hi, s_true_hi, ratio, CI, D5 verdict
    decide([(lo, hi) for each seed], "D1")               # all CIs above 0 -> pass
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def score_g1(evaluator, preds: Sequence, *, bases: Sequence[str] = (), sonuc_dir=None, panel_dir=None,
             B: int = 1000, python: str | None = None) -> list[dict]:
    """Run talp73 degerlendir.py on prediction files (named <arm>_s<seed>.parquet; base files <base>_s<seed> in the
    same folder). TALP73_SONUC / TALP73_PANEL are set when given. Returns the result json of each file."""
    env = dict(os.environ)
    if sonuc_dir is not None:
        env["TALP73_SONUC"] = str(sonuc_dir)
        Path(sonuc_dir).mkdir(parents=True, exist_ok=True)
    if panel_dir is not None:
        env["TALP73_PANEL"] = str(panel_dir)
    cmd = [python or sys.executable, str(evaluator), *map(str, preds), "--B", str(B)]
    if bases:
        cmd += ["--taban", *bases]
    subprocess.run(cmd, env=env, check=True, cwd=str(Path(evaluator).parent))
    out = []
    son = Path(env.get("TALP73_SONUC", "."))
    for p in preds:
        j = son / (Path(p).stem + ".json")
        out.append(json.loads(j.read_text()) if j.exists() else {"missing": str(j)})
    return out


def _boot_idx(n: int, B: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n))


def _ci(v) -> tuple[float, float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def mad_accuracy(pred: pd.DataFrame, truth: pd.DataFrame, *, items: Sequence[str] | None = None) -> pd.Series:
    """Per-person MAD accuracy 1 - mean |y01_hat - y01| over the person's items (pred: uid, oge, p; truth: uid,
    oge, y01)."""
    d = truth.merge(pred, on=["uid", "oge"], how="inner")
    if items is not None:
        d = d[d.oge.isin(set(items))]
    return 1 - (d["p"] - d["y01"]).abs().groupby(d["uid"]).mean()


def score_g2(pred: pd.DataFrame, truth: pd.DataFrame, *, items: Sequence[str] | None = None, B: int = 1000,
             seed: int = 42, base: pd.DataFrame | None = None) -> dict:
    """MAD accuracy (person mean, then over persons) with a person bootstrap CI; with base, the paired difference."""
    a = mad_accuracy(pred, truth, items=items)
    out = {"mad_acc": float(a.mean()), "n_persons": int(len(a))}
    idx = _boot_idx(len(a), B, seed)
    av = a.to_numpy()
    out["ci"] = _ci(av[idx].mean(1))
    if base is not None:
        b = mad_accuracy(base, truth, items=items).reindex(a.index)
        dv = av - b.to_numpy()
        out["diff"] = float(np.nanmean(dv))
        out["diff_ci"] = _ci(np.nanmean(dv[idx], 1))
    return out


def _slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm = x - x.mean()
    return float((xm * (y - y.mean())).sum() / (xm ** 2).sum())


def demand_curve(df: pd.DataFrame, col: str) -> pd.Series:
    """Mean of `col` per carpan level, averaging within person first (person-weighted)."""
    return df.groupby(["carpan_duzey", "uid"])[col].mean().groupby("carpan_duzey").mean()


def score_g3(sweep: pd.DataFrame, truth: pd.DataFrame, *, lo: float = 1.0, B: int = 1000, seed: int = 42) -> dict:
    """Counterfactual price check. sweep: uid, oge, carpan_duzey, p (each test person x target product x level);
    truth: observed test rows uid, oge, carpan_duzey, y. s_hi = OLS slope of the mean predicted P(buy) over levels
    >= lo; s_true_hi = the same on the observed buy rate. Person bootstrap for the CI of s_hi and of the ratio.
    D5: s_hi CI upper bound < 0 and s_hi / s_true_hi in [0.5, 1.5]."""
    persons = np.array(sorted(set(sweep.uid) & set(truth.uid)))
    sw = sweep[sweep.uid.isin(persons)]
    tr = truth[truth.uid.isin(persons)]
    lv = sorted(l for l in sw.carpan_duzey.unique() if l >= lo - 1e-9)
    # per person x level means, so a bootstrap is a weighted average of rows of these tables
    P = sw.groupby(["uid", "carpan_duzey"])["p"].mean().unstack().reindex(persons)
    T = tr.assign(y=tr["y"].astype(float)).groupby(["uid", "carpan_duzey"])["y"].agg(["sum", "count"])
    Ts = T["sum"].unstack().reindex(persons).fillna(0.0)
    Tc = T["count"].unstack().reindex(persons).fillna(0.0)

    def slopes(w):
        cp = (P.mul(w, axis=0)).sum() / w.sum()
        ct = (Ts.mul(w, axis=0)).sum() / (Tc.mul(w, axis=0)).sum()
        s_hi = _slope(lv, cp.reindex(lv))
        st = ct.reindex(lv)
        ok = st.notna()
        s_true = _slope(np.array(lv)[ok.to_numpy()], st[ok]) if ok.sum() >= 2 else np.nan
        s_all = _slope(cp.index, cp.to_numpy())
        return s_hi, s_true, s_all, cp, ct

    w1 = np.ones(len(persons))
    s_hi, s_true, s_all, cp, ct = slopes(pd.Series(w1, index=persons))
    idx = _boot_idx(len(persons), B, seed)
    bs = []
    for r in idx:
        w = pd.Series(np.bincount(r, minlength=len(persons)).astype(float), index=persons)
        a, t, _, _, _ = slopes(w)
        bs.append((a, t, a / t if t else np.nan))
    bs = np.array(bs)
    ratio = s_hi / s_true if s_true else np.nan
    ci = _ci(bs[:, 0])
    out = {"s_hi": s_hi, "s_hi_ci": ci, "s_true_hi": s_true, "s_true_hi_ci": _ci(bs[:, 1]), "ratio": ratio,
           "ratio_ci": _ci(bs[:, 2]), "s_all": s_all, "curve": {float(k): float(v) for k, v in cp.items()},
           "curve_true": {float(k): float(v) for k, v in ct.items()},
           "pearson_curve": float(np.corrcoef(cp.reindex(ct.index), ct)[0, 1]) if len(ct) > 2 else np.nan,
           "n_persons": int(len(persons))}
    out["D5"] = bool(ci[1] < 0 and 0.5 <= ratio <= 1.5)
    return out


def decide(cis: Sequence[tuple[float, float]], rule: str = "D1", *, band: float = 0.01,
           acc_cis: Sequence[tuple[float, float]] | None = None) -> dict:
    """Verdict from per-seed paired CIs (arm - reference) on kisi_auc. D1: pass when every CI's lower bound > 0 and,
    with acc_cis (1a(a)), every upper bound >= 0; 'adds signal' when the 1a(a) CIs are all > 0 too, else 'adds
    ranking signal'. D2/D3/D6: every lower bound > 0. D4: 'better' (all > 0), 'equivalent' (all inside
    [-band, band]) or 'worse/unclear'."""
    cis = [tuple(map(float, c)) for c in cis]
    if not cis:
        return {"rule": rule, "verdict": "not run"}
    above = all(lo > 0 for lo, _ in cis)
    if rule == "D4":
        if above:
            v = "better"
        elif all(-band <= lo and hi <= band for lo, hi in cis):
            v = "equivalent"
        else:
            v = "worse/unclear"
        return {"rule": rule, "verdict": v, "cis": cis}
    out = {"rule": rule, "verdict": "pass" if above else "fail", "cis": cis, "n_seeds": len(cis)}
    if rule == "D1" and acc_cis is not None:
        acc = [tuple(map(float, c)) for c in acc_cis]
        ok = above and all(hi >= 0 for _, hi in acc)
        out.update(verdict="pass" if ok else "fail", acc_cis=acc)
        if ok:
            out["label"] = "adds signal" if all(lo > 0 for lo, _ in acc) else "adds ranking signal"
    return out


__all__ = ["score_g1", "score_g2", "score_g3", "mad_accuracy", "demand_curve", "decide"]
