"""Evidence frame helpers: repeated-split summary, cell counts, wide-to-long, segment bias, pooled dists, choices.

    import simengine.evidence as ev
    reps = [ev.evaluate(make_split(s), ours="t1", multiplier=False) for s in range(100)]
    ss = ev.split_summary(reps, pairs=[("t1", "o1")])          # per-split values + paired diffs over splits
    ss.table; ss.markdown; ev.evaluate(frame, ours="t1", limits=ss.limits)   # card section or card limits
    ev.split_summary(make_split_frame, seeds=range(100), metric="mae", pairs=[("t1", "o1")])  # seed loop here

    cf = ev.counts_frame(people, {"argmax": "pred_label"}, by=["area"], truth="truth")   # one row per area x class
    cf = ev.counts_frame(people, {"probs": "p_vec"}, by=["area"], probs=True, classes=["a", "b", "c"])
    ev.evaluate(cf, ours="probs", task="forecast", primary="wmape")                   # density WMAPE on counts

    lf = ev.long_frame("sealed_wide.csv", truth_wide, "id", scale="sd_train", train=train_wide)  # item = target
    ev.evaluate(lf, ours="ours", task="continuous", primary="mae")                    # MAE in training-sd units

    sb = ev.segment_bias(frame, "region", ["ours", "base"])   # n-weighted |mean pred - mean truth| per segment
    sb["diff"]                                                 # paired cluster bootstrap ours - base

    nat = ev.pooled_distribution(cells, weight_col="n", by="item")   # item x segment dists -> n-weighted national
    ev.evaluate(nat, ours="llm", task="distribution", primary="one_minus_tvd")

    cf = ev.choice_frame(long, "task", "alt", "chosen", ["pred_mnl"])   # (task, alternative) rows -> one per task
    ev.evaluate(cf, ours="mnl", task="multiclass", primary="logloss", classes=cf.attrs["classes"])

Why these exist (TALP rehearsals): a dev card tied to one random split hid split-to-split spread (D11 #5d, D12 #7);
per-person argmax beat the baseline on accuracy but lost on the cell-count WMAPE, and the count frame was built by
hand (D9 #5 #6); a sealed wide multi-target file had to be melted and scaled by the training sd by hand (D12 #20);
the brief's segment error had no function and its single-arm bootstrap CI excluded the point (D12 #3 #21);
the national TVD was written by hand because aggregate_by averages groups instead of n-weighting them (D6 #3a);
the long (task, alternative) choice table had no converter to a multiclass frame (D8 #5).
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..types import COLUMN_ALIASES, PRED_PREFIX, normalize_frame
from . import bootstrap as _boot
from . import metrik as _metrik
from .api import (FORECAST_TASKS, METRIC_TO_ENGLISH, _method, _metric, _task, prepare)
from .cekirdek import AZ_KUME
from .sema import ON_EK

ABS_CI_NOTE = ("a single-arm bootstrap CI of an absolute-value metric is biased: |mean error| of a resample is "
               "never below 0 and sits above the point when the true error is near 0, so the percentile CI can "
               "exclude the point; compare arms with the paired difference CI")
SPLIT_NOTE = "spread over repeated splits of the same data (split-to-split variation), not a CI of the population value"
BEST_BASELINE = "best_baseline"


def _pct(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.0f} %"


def _num(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.4g}"


def _diff_num(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.4g}"


def _better(direction: int, signed: bool, a: float, b: float) -> float:
    """1.0 when a beats b, 0.5 on a tie, 0.0 otherwise; NaN when either is missing."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    if signed:
        a, b, direction = abs(a), abs(b), -1
    if a == b:
        return 0.5
    return 1.0 if (a - b) * direction > 0 else 0.0


# ------------------------------------------------------------------ split_summary
@dataclass
class SplitSummary:
    """Result of `split_summary`. per_split: one row per split x column (value) and split x pair (diff);
    table: median and 2.5 / 97.5 % over splits per column and per pair (pairs: share of splits favouring a);
    markdown: a card section; limits: one line per pair for `evaluate(limits=...)`."""
    metric: str
    direction: int
    n_splits: int
    per_split: pd.DataFrame
    table: pd.DataFrame
    markdown: str
    limits: list[str] = field(default_factory=list)


def _directions() -> dict[str, int]:
    out: dict[str, int] = {}
    for g in ("ikili", "cok_sinif", "dagilim", "surekli", "ab"):
        for olas in (True, False):
            for madde in (True, False):
                try:
                    fam = _metrik.aile(g, olas, madde, False, oran=True)
                except Exception:                                   # noqa: BLE001 (task without that flavour)
                    continue
                for m in fam:
                    out.setdefault(m.ad, m.yon)
    return out


def _report_values(rep: dict, metric: str | None) -> tuple[str, int, dict[str, float]]:
    ad = _metric(metric) or rep.get("birincil")
    if not ad:
        raise ValueError("split_summary: the report has no primary metric; pass metric=")
    yon = next((m.get("yon") for m in rep.get("metrikler") or [] if m.get("ad") == ad), None)
    if yon is None:
        yon = _directions().get(ad, 1)
    vals = {}
    for col, r in (rep.get("sonuclar") or {}).items():
        m = (r.get("metrikler") or {}).get(ad)
        if m is not None:
            vals[col] = float(m["deger"]) if m.get("deger") is not None else np.nan
    if not vals:
        raise ValueError(f"split_summary: metric {metric or ad!r} is not in the report's results")
    man = rep.get("manset") or {}
    for key, alias in (("bizim", "ours"), ("ham", "raw"), ("en_iyi_baseline", BEST_BASELINE)):
        col = (man.get(key) or {}).get("sutun")
        if col in vals:
            vals[alias] = vals[col]
    if ad != rep.get("birincil"):
        # the headline's best baseline is picked on the primary; on another metric it can be the weakest one
        # (pearson-best "random" vs mae-best "marginal"): pick the strongest baseline on this metric instead
        tab = [(c, vals[c]) for c, r in (rep.get("sonuclar") or {}).items()
               if r.get("tur") == "baseline" and c in vals and np.isfinite(vals[c])]
        vals.pop(BEST_BASELINE, None)
        if tab:
            key = (lambda cv: abs(cv[1])) if ad in _metrik.ISARETLI_METRIKLER else (lambda cv: -yon * cv[1])
            vals[BEST_BASELINE] = min(tab, key=key)[1]
    return ad, int(yon), vals


def _frame_values(frame: pd.DataFrame, metric: str | None, task: str | None) -> tuple[str, int, dict[str, float]]:
    v = prepare(frame, _task(task))
    d = v.df
    madde = pd.factorize(d["madde"])[0] if "madde" in d else None
    olas = v.gorev != "cok_sinif" or isinstance(d[ON_EK + v.yontemler[0]].iloc[0], np.ndarray)
    birim = v.gorev == "surekli" and d["gercek"].between(0, 1).all()
    fam = {m.ad: m for m in _metrik.aile(v.gorev, olas, madde is not None and v.gorev in ("ikili", "surekli"),
                                         birim, oran=True)}
    ad = _metric(metric) or ("wmape" if task in FORECAST_TASKS else
                             "bir_eksi_mae" if birim and "bir_eksi_mae" in fam else _metrik.BIRINCIL[v.gorev])
    if ad not in fam:
        raise ValueError(f"split_summary: metric {metric or ad!r} is not defined for task {v.gorev!r}; "
                         f"use one of {sorted(METRIC_TO_ENGLISH.get(k, k) for k in fam)}")
    M = fam[ad]
    w = d["agirlik"].to_numpy(float) if "agirlik" in d else None
    vals = {}
    for y_ in v.yontemler:
        y, p = _metrik.diziler(v, ON_EK + y_)
        try:
            vals[ON_EK + y_] = float(M.deger(y, p, madde) if M.tur == "toplu" else M.deger(y, p, w=w))
        except (ValueError, ZeroDivisionError, IndexError):
            vals[ON_EK + y_] = np.nan
    return ad, int(M.yon), vals


def _lookup(vals: dict[str, float], name: str) -> float:
    for k in (name, ON_EK + (_method(name) or ""), name.replace("baseline_", "taban_", 1)):
        if k in vals:
            return vals[k]
    raise KeyError(name)


def _label(col: str) -> str:
    return col[len(ON_EK):] if col.startswith(ON_EK) else col


def split_summary(reports_or_frames, metric: str | None = None, pairs: Sequence[tuple[str, str]] | None = None, *,
                  seeds: Iterable[int] | None = None, task: str | None = None, names: Sequence | None = None,
                  title: str = "Repeated splits") -> SplitSummary:
    """Per-split values and paired differences over repeated splits (or seeds) of the same evaluation.

    reports_or_frames: a list (or {split name: x} dict) of `evaluate` reports or of EvalFrames (one per split; frames
        are scored with point metrics only, no bootstrap), or a callable seed -> report | frame together with seeds=
        (the seed loop runs here).
    metric: English or internal name; default the report's primary (frames: the task's primary, wmape for
        task="forecast").
    pairs: [(a, b), ...] method names (with or without `pred_`; reports also take baseline columns such as
        "taban_marjinal", "ours", "raw" and "best_baseline" = that split's strongest baseline on `metric`, which is
        the headline's best baseline when metric is the report's primary). Default: reports with
        an ours column -> [("ours", "best_baseline")]; frames -> none.
    Returns a SplitSummary: `table` has, per column and per pair a − b, the median and the 2.5 / 97.5 % quantiles over
    splits and, for pairs, the share of splits in which a is better (ties count half; signed metrics compare |value|).
    The quantiles describe split-to-split spread, not a CI. `markdown` is a card section; `limits` one line per pair
    for evaluate(limits=...).
    """
    if callable(reports_or_frames):
        if seeds is None:
            raise ValueError("split_summary: a callable needs seeds= (e.g. seeds=range(100))")
        seeds = list(seeds)
        items = [reports_or_frames(s) for s in seeds]
        keys = list(names) if names is not None else seeds
    elif isinstance(reports_or_frames, Mapping):
        keys, items = list(reports_or_frames.keys()), list(reports_or_frames.values())
    else:
        items = list(reports_or_frames)
        keys = list(names) if names is not None else list(range(len(items)))
    if len(keys) != len(items):
        raise ValueError("split_summary: names= must have one name per split")
    if len(items) < 2:
        raise ValueError("split_summary: needs at least 2 splits")
    ad = yon = None
    per_vals = []
    for it in items:
        if isinstance(it, dict):
            a_, y_, vals = _report_values(it, metric)
        elif isinstance(it, pd.DataFrame):
            a_, y_, vals = _frame_values(it, metric, task)
        else:
            raise TypeError(f"split_summary: expected evaluate reports or DataFrames, got {type(it).__name__}")
        if ad is not None and a_ != ad:
            raise ValueError(f"split_summary: splits disagree on the metric ({ad!r} vs {a_!r}); pass metric=")
        ad, yon = a_, y_
        per_vals.append(vals)
    signed = ad in _metrik.ISARETLI_METRIKLER
    if pairs is None:
        pairs = [("ours", BEST_BASELINE)] if all("ours" in v and BEST_BASELINE in v for v in per_vals) else []
    pairs = [tuple(p) for p in pairs]
    for a, b in pairs:
        for v in per_vals:
            for nm in (a, b):
                try:
                    _lookup(v, nm)
                except KeyError:
                    raise ValueError(f"split_summary: column {nm!r} is missing in a split; columns: "
                                     f"{sorted(_label(c) for c in v)}") from None
    cols = [c for c in per_vals[0] if c not in ("ours", "raw", BEST_BASELINE)
            and all(c in v for v in per_vals[1:])]
    rows = []
    for k, v in zip(keys, per_vals):
        rows += [{"split": k, "kind": "value", "name": _label(c), "value": v[c], "a_better": np.nan} for c in cols]
        for a, b in pairs:
            va, vb = _lookup(v, a), _lookup(v, b)
            rows.append({"split": k, "kind": "diff", "name": f"{_label(a)} - {_label(b)}", "value": va - vb,
                         "a_better": _better(yon, signed, va, vb)})
    per = pd.DataFrame(rows, columns=["split", "kind", "name", "value", "a_better"])
    summ = []
    for (kind, name), g in per.groupby(["kind", "name"], sort=False):
        x = g["value"].to_numpy(float)
        x = x[np.isfinite(x)]
        q = np.quantile(x, [0.025, 0.5, 0.975]) if len(x) else [np.nan] * 3
        summ.append({"kind": kind, "name": name, "median": float(q[1]), "q025": float(q[0]), "q975": float(q[2]),
                     "mean": float(x.mean()) if len(x) else np.nan, "n_splits": int(len(x)),
                     "share_a_better": float(np.nanmean(g["a_better"])) if kind == "diff" else np.nan})
    table = pd.DataFrame(summ)
    m_en = METRIC_TO_ENGLISH.get(ad, ad)
    n = len(items)
    lines = [f"### {title}: {m_en} over {n} splits", "",
             f"Median and 2.5 / 97.5 % over splits: {SPLIT_NOTE}.", "",
             "| column / pair | median | 2.5 % | 97.5 % | a better in |", "|---|---|---|---|---|"]
    limits = []
    for r in table.itertuples():
        diff = r.kind == "diff"
        f = _diff_num if diff else _num
        lines.append(f"| {r.name} | {f(r.median)} | {f(r.q025)} | {f(r.q975)} | "
                     f"{_pct(r.share_a_better) + ' of splits' if diff else ''} |")
        if diff:
            limits.append(f"repeated splits (n={r.n_splits}): {r.name} {m_en} median {_diff_num(r.median)}, "
                          f"2.5-97.5 % over splits [{_diff_num(r.q025)}, {_diff_num(r.q975)}], "
                          f"a better in {_pct(r.share_a_better)} of splits")
    better = "closer to 0" if signed else "lower" if yon < 0 else "higher"
    lines += ["", f"{m_en}: {better} is better. A single split's card shows one draw from this spread."]
    return SplitSummary(metric=m_en, direction=yon, n_splits=n, per_split=per, table=table,
                        markdown="\n".join(lines) + "\n", limits=limits)


# ------------------------------------------------------------------ counts_frame
def _arm_map(pred_cols) -> dict[str, Any]:
    if isinstance(pred_cols, str):
        pred_cols = [pred_cols]
    if isinstance(pred_cols, Mapping):
        return {str(k): v for k, v in pred_cols.items()}
    out = {}
    for c in pred_cols:
        arm = re.sub(r"^(pred_|tahmin_|p_)", "", str(c)) or str(c)
        if arm in out:
            raise ValueError(f"counts_frame: two prediction columns map to the arm {arm!r}; pass a dict {{arm: column}}")
        out[arm] = c
    return out


def _prob_matrix(rows: pd.DataFrame, src, classes: list | None) -> tuple[np.ndarray, list]:
    """Probability source -> (n x K matrix, class order). src: {class: column}, a column of {class: p} dicts, or a
    column of vectors (needs classes=)."""
    if isinstance(src, Mapping):
        cl = list(src.keys())
        return np.column_stack([rows[c].to_numpy(float) for c in src.values()]), cl
    s = rows[src]
    first = next((x for x in s if x is not None and not (np.isscalar(x) and pd.isna(x))), None)
    if isinstance(first, Mapping):
        cl = list(classes) if classes is not None else sorted({k for x in s if isinstance(x, Mapping) for k in x})
        M = np.array([[float(x.get(c, 0.0)) for c in cl] if isinstance(x, Mapping) else [np.nan] * len(cl)
                      for x in s], float)
        return M, cl
    if classes is None:
        raise ValueError(f"counts_frame: column {src!r} holds probability vectors; pass classes= (their order)")
    cl = list(classes)
    M = np.array([np.asarray(x, float) if x is not None and not (np.isscalar(x) and pd.isna(x))
                  else np.full(len(cl), np.nan) for x in s], float)
    if M.ndim != 2 or M.shape[1] != len(cl):
        raise ValueError(f"counts_frame: vectors in {src!r} do not have len(classes)={len(cl)} entries")
    return M, cl


def counts_frame(rows: pd.DataFrame, pred_cols, by: Sequence[str] | str, *, truth: str = "truth",
                 cluster: str | None = None, probs: bool | Sequence[str] = False, classes: Sequence | None = None,
                 weight: str | None = None, argmax_too: bool = False) -> pd.DataFrame:
    """Per-person predictions -> one row per cell (by values x class) with the truth count and each arm's predicted
    count, ready for evaluate(task="forecast", primary="wmape") (density / cell-count error).

    rows: one row per person with the true class in `truth` (alias "gercek").
    pred_cols: column, list (a `pred_` / `p_` prefix is dropped from the arm name) or {arm: source}. A label arm counts
        persons per predicted class. A probability arm (probs=True for all arms, or a list of arm names) sums the
        probabilities: expected counts. Probability source: a column of vectors (with classes=), a column of
        {class: p} dicts, or {class: column}. Rows that do not sum to 1 warn.
    by: cell columns (e.g. ["area"] or ["area", "week"]); cells are the observed by-combinations x every class, zeros
        included (a class predicted where none occurred counts).
    cluster: bootstrap cluster written to `group` (a by column, or any column constant within cells); default the
        by-cell, so all classes of one cell move together.
    weight: optional person weight column (weighted counts).
    argmax_too: for probability arms, also add `pred_<arm>_argmax` (argmax label counts), the per-person argmax trap:
        argmax wins accuracy but piles every cell onto its majority class.
    Rows with missing truth are dropped. A cell in which an arm has a missing prediction gets NaN for that arm (a
    missing prediction is not a zero count). Columns: by..., `class`, `item` (= class, per-class baselines),
    `group`, `truth`, `pred_<arm>`..., `n` (persons in the cell).
    """
    by = [by] if isinstance(by, str) else list(by)
    d = rows
    if truth == "truth" and "truth" not in d.columns and "gercek" in d.columns:
        truth = "gercek"
    if truth not in d.columns:
        raise ValueError(f"counts_frame: truth column {truth!r} not found")
    missing = [c for c in by if c not in d.columns]
    if missing:
        raise ValueError(f"counts_frame: by columns {missing} not found")
    d = d[d[truth].notna()].reset_index(drop=True)
    arms = _arm_map(pred_cols)
    prob_arms = set(arms) if probs is True else set() if probs is False else {str(a) for a in probs}
    if prob_arms - set(arms):
        raise ValueError(f"counts_frame: probs names arms {sorted(prob_arms - set(arms))} not in pred_cols")
    w = np.ones(len(d)) if weight is None else d[weight].to_numpy(float)
    cell_key = d[by].astype(str).agg("\x1f".join, axis=1) if len(by) > 1 else d[by[0]].astype(str)
    cells = d[by].drop_duplicates().reset_index(drop=True)
    ckey = cells.astype(str).agg("\x1f".join, axis=1) if len(by) > 1 else cells[by[0]].astype(str)
    cidx = pd.Index(ckey).get_indexer(cell_key)

    mats: dict[str, np.ndarray] = {}
    lab_preds: dict[str, pd.Series] = {}
    cl_set = list(classes) if classes is not None else list(pd.unique(d[truth]))
    for arm, src in arms.items():
        if arm in prob_arms:
            M, cl = _prob_matrix(d, src, None if classes is None else list(classes))
            rs = M[np.isfinite(M).all(1)].sum(1)
            if len(rs) and not np.allclose(rs, 1.0, atol=1e-3):
                # percentages, scores or a partial class list: the expected counts no longer add up to the persons
                warnings.warn(f"counts_frame: probabilities of arm {arm!r} do not sum to 1 per person (range "
                              f"{rs.min():.3g} to {rs.max():.3g}); expected counts are off by that factor",
                              UserWarning, stacklevel=2)
            for c in cl:
                if c not in cl_set:
                    cl_set.append(c)
            mats[arm] = (M, cl)
            if argmax_too:
                am = np.where(np.isfinite(M).all(1), np.argmax(np.where(np.isfinite(M), M, -np.inf), 1), -1)
                lab_preds[arm + "_argmax"] = pd.Series([cl[i] if i >= 0 else np.nan for i in am])
        else:
            s = d[src] if isinstance(src, str) else pd.Series(np.asarray(src))
            if len(s) != len(d):
                raise ValueError(f"counts_frame: arm {arm!r} has {len(s)} values for {len(d)} rows")
            lab_preds[arm] = s.reset_index(drop=True)
            if classes is None:
                for c in pd.unique(s.dropna()):
                    if c not in cl_set:
                        cl_set.append(c)
    if classes is None:
        try:
            cl_set = sorted(cl_set)
        except TypeError:
            pass
    K, C = len(cl_set), len(cells)
    kpos = {c: i for i, c in enumerate(cl_set)}

    def counts(labels: pd.Series) -> np.ndarray:
        out = np.zeros((C, K))
        ok = labels.notna().to_numpy()
        k = labels.map(kpos)
        if ok.any() and k[ok].isna().any():
            raise ValueError(f"counts_frame: labels {sorted(set(labels[ok][k[ok].isna()]))[:5]} not in classes=")
        np.add.at(out, (cidx[ok], k[ok].to_numpy(int)), w[ok])
        bad = np.zeros(C, bool)
        np.logical_or.at(bad, cidx[~ok], True)
        out[bad] = np.nan
        return out

    truth_c = counts(d[truth])
    preds: dict[str, np.ndarray] = {}
    for arm, (M, cl) in mats.items():
        out = np.zeros((C, K))
        cols = [kpos[c] for c in cl]
        bad_row = ~np.isfinite(M).all(1)
        Mz = np.where(np.isfinite(M), M, 0.0) * w[:, None]
        for j, k in enumerate(cols):
            np.add.at(out[:, k], cidx, Mz[:, j])
        bad = np.zeros(C, bool)
        np.logical_or.at(bad, cidx[bad_row], True)
        out[bad] = np.nan
        preds[arm] = out
    for arm, s in lab_preds.items():
        preds[arm] = counts(s)
    if cluster is not None:
        if cluster not in d.columns:
            raise ValueError(f"counts_frame: cluster column {cluster!r} not found")
        per_cell = d.groupby(cidx)[cluster].nunique()
        if (per_cell > 1).any():
            raise ValueError(f"counts_frame: cluster {cluster!r} is not constant within the by-cells")
        grp = d.groupby(cidx)[cluster].first().reindex(range(C)).to_numpy()
    else:
        grp = ckey.to_numpy()
    n_cell = np.bincount(cidx, minlength=C)
    out = cells.loc[np.repeat(np.arange(C), K)].reset_index(drop=True)
    out["class"] = np.tile(np.array(cl_set, dtype=object), C)
    out["item"] = out["class"]
    out["group"] = np.repeat(grp, K)
    out["truth"] = truth_c.ravel()
    for arm, P in preds.items():
        out[PRED_PREFIX + arm] = P.ravel()
    out["n"] = np.repeat(n_cell, K)
    return out


# ------------------------------------------------------------------ long_frame
def _read(x, what: str, id_col: str | None = None) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, (str, Path)):
        p = Path(x)
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p, dtype={id_col: str} if id_col else None)     # ids as text: "007" stays "007"
    raise TypeError(f"long_frame: {what} must be a DataFrame or a CSV / parquet path")


def long_frame(pred_wide, truth_wide, id_col: str, scale: None | str | Mapping[str, float] | pd.Series = None, *,
               train=None, targets: Sequence[str] | None = None, arm: str = "ours") -> pd.DataFrame:
    """Wide multi-target predictions and truth (one row per unit, one column per target) -> long EvalFrame with
    one row per unit x target: `group` = unit id (the bootstrap cluster: all targets of a unit move together),
    `item` = target, `truth`, `pred_<arm>`.

    pred_wide: a DataFrame or CSV / parquet path (e.g. the sealed prediction file), or {arm: DataFrame | path}.
    scale: None (raw units), {target: s} / Series, or "sd_train": every target's truth and predictions are divided by
        the sample sd (ddof=1) of that target in `train=` (wide training truth), so MAE becomes zMAE (sd units) and
        targets on different scales weigh alike. A zero or missing scale raises.
    targets: default the numeric non-id truth columns present in every prediction frame.
    Ids are read as text from CSV; when one side has numeric ids and the other text, they are matched as text.
    Units of the truth without a prediction are dropped with a warning (no shared unit raises); the scale used is in
    `frame.attrs["scale"]`.
    """
    truth = _read(truth_wide, "truth_wide", id_col)
    arms = ({str(k): _read(v, f"pred_wide[{k!r}]", id_col) for k, v in pred_wide.items()}
            if isinstance(pred_wide, Mapping) else {arm: _read(pred_wide, "pred_wide", id_col)})
    for nm, f in {"truth_wide": truth, **arms}.items():
        if id_col not in f.columns:
            raise ValueError(f"long_frame: id column {id_col!r} not in {nm}")
    back = None
    if len({pd.api.types.is_numeric_dtype(f[id_col]) for f in (truth, *arms.values())}) > 1:
        # numeric ids on one side, text on the other (a CSV read back): match on the text form, keep truth's ids
        back = dict(zip(truth[id_col].astype(str), truth[id_col]))
        truth = truth.assign(**{id_col: truth[id_col].astype(str)})
        arms = {k: f.assign(**{id_col: f[id_col].astype(str)}) for k, f in arms.items()}
    for nm, f in {"truth_wide": truth, **arms}.items():
        if f[id_col].duplicated().any():
            raise ValueError(f"long_frame: duplicate ids in {nm}")
    if targets is None:
        targets = [c for c in truth.columns if c != id_col and all(c in f.columns for f in arms.values())
                   and (pd.api.types.is_numeric_dtype(truth[c]) or pd.api.types.is_bool_dtype(truth[c]))]
        if not targets:
            raise ValueError("long_frame: no target column is shared by the truth and the predictions")
    else:
        targets = list(targets)
        for nm, f in {"truth_wide": truth, **arms}.items():
            miss = [t for t in targets if t not in f.columns]
            if miss:
                raise ValueError(f"long_frame: targets {miss} not in {nm}")
    if isinstance(scale, str):
        if scale != "sd_train":
            raise ValueError(f"long_frame: scale must be None, a mapping or 'sd_train', not {scale!r}")
        if train is None:
            raise ValueError("long_frame: scale='sd_train' needs train= (wide training truth)")
        tr = _read(train, "train")
        miss = [t for t in targets if t not in tr.columns]
        if miss:
            raise ValueError(f"long_frame: targets {miss} not in train")
        sc = {t: float(pd.to_numeric(tr[t], errors="coerce").std(ddof=1)) for t in targets}
    elif scale is None:
        sc = {t: 1.0 for t in targets}
    else:
        sc = {t: float(dict(scale)[t]) if t in dict(scale) else np.nan for t in targets}
    bad = [t for t, s in sc.items() if not np.isfinite(s) or s <= 0]
    if bad:
        raise ValueError(f"long_frame: scale is zero or missing for targets {bad}")
    base = truth[[id_col] + targets]
    ids = set(base[id_col])
    for nm, f in arms.items():
        lost = ids - set(f[id_col])
        if lost:
            warnings.warn(f"long_frame: {len(lost)} of {len(ids)} truth units have no prediction in arm {nm!r}; "
                          "they are dropped", UserWarning, stacklevel=2)
        ids &= set(f[id_col])
    if not ids:
        raise ValueError(f"long_frame: no unit id is shared by the truth and the predictions (id dtypes: truth "
                         f"{truth[id_col].dtype}, " + ", ".join(f"{k} {f[id_col].dtype}" for k, f in arms.items()) +
                         "); check the id column and its format")
    base = base[base[id_col].isin(ids)]
    out = base.melt(id_vars=id_col, value_vars=targets, var_name="item", value_name="truth")
    for nm, f in arms.items():
        lf = f[[id_col] + targets].melt(id_vars=id_col, value_vars=targets, var_name="item",
                                        value_name=PRED_PREFIX + nm)
        out = out.merge(lf, on=[id_col, "item"], how="left")
    s = out["item"].map(sc).to_numpy(float)
    for c in ["truth"] + [PRED_PREFIX + nm for nm in arms]:
        out[c] = pd.to_numeric(out[c], errors="raise").to_numpy(float) / s
    if back is not None:
        out[id_col] = out[id_col].map(back)
    out = out.rename(columns={id_col: "group"}) if id_col != "group" else out
    out = out[["group", "item", "truth"] + [PRED_PREFIX + nm for nm in arms]]
    out.attrs["scale"] = dict(sc)
    return out


# ------------------------------------------------------------------ segment_bias
def _seg_stats(cnt: np.ndarray, Sw: np.ndarray, Sy: np.ndarray, Sp: dict[str, np.ndarray], equal: bool):
    """Cluster counts (R x C) and cluster x segment sums -> {arm: (R,) weighted mean over segments of |mean p - mean y|}."""
    W = cnt @ Sw                                   # R x S
    ok = W > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        my = (cnt @ Sy) / W
        segw = ok.astype(float) if equal else W
        segw = segw / segw.sum(1, keepdims=True)
        out = {}
        for a, sp in Sp.items():
            e = np.abs((cnt @ sp) / W - my)
            out[a] = np.where(ok, e * segw, 0.0).sum(1)
    return out


def segment_bias(frame: pd.DataFrame, seg: str | Sequence[str], arms: Sequence[str], weight: str | None = None, *,
                 cluster: str | None = None, B: int = 2000, seed: int = 0, ci: str = "percentile",
                 min_n: int = 1) -> dict:
    """Segment error |mean pred_s − mean truth_s|, averaged over segments with weight n_s (the brief's metric), per
    arm, with a paired cluster bootstrap of the arm differences.

    frame: EvalFrame (`truth`, `pred_<arm>`; Turkish aliases accepted). seg: segment column(s) (several are combined).
    arms: method names (with or without `pred_`); the first is "ours": pairs are arms[0] − each other arm.
    weight: None -> segments weighted by row count n_s; "equal" -> every segment alike; a column name -> row weights
        (weighted means, segment weight = its weight sum).
    cluster: bootstrap cluster column (default group > item > row); every arm and the truth are resampled together.
        Fewer than 10 clusters warns; a single cluster leaves the CIs and p empty (NaN).
    min_n: segments with fewer rows are left out (listed in `dropped`). Rows missing the truth or any arm are dropped
        for every arm (paired).
    Returns {"metric", "arms": {arm: {"value", "lo", "hi", "point_outside_ci"}}, "diff": [{"a", "b", "diff", "lo",
    "hi", "p", "a_better"}], "segments": DataFrame (segment, n, weight, mean truth, mean and |error| per arm),
    "cluster", "n_clusters", "B", "dropped", "note"}. The single-arm CI is shown for completeness only: for an
    absolute-value metric it is biased upward (see `note`); judge arms on the paired difference CI.
    """
    d = normalize_frame(frame)
    segs = [seg] if isinstance(seg, str) else list(seg)
    for c in segs:
        if c not in d.columns:
            raise ValueError(f"segment_bias: segment column {c!r} not found")
    if "truth" not in d.columns:
        raise ValueError("segment_bias: the frame has no truth column")
    cols = {}
    for a in arms:
        nm = _method(a)
        c = PRED_PREFIX + nm
        if c not in d.columns:
            raise ValueError(f"segment_bias: prediction column {c!r} not found")
        cols[nm] = c
    if len(cols) < 1:
        raise ValueError("segment_bias: pass at least one arm")
    if ci not in ("percentile", "basic"):
        raise ValueError(f"segment_bias: ci must be 'percentile' or 'basic', not {ci!r}")
    equal = weight == "equal"
    keep = d["truth"].notna() & d[segs].notna().all(axis=1)
    for c in cols.values():
        keep &= d[c].notna()
    d = d[keep].reset_index(drop=True)
    seg_key = d[segs].astype(str).agg(" | ".join, axis=1) if len(segs) > 1 else d[segs[0]]
    sizes = seg_key.value_counts()
    dropped = sorted(map(str, sizes[sizes < min_n].index))
    d = d[seg_key.map(sizes) >= min_n].reset_index(drop=True)
    seg_key = seg_key[seg_key.map(sizes) >= min_n].reset_index(drop=True)
    if d.empty:
        raise ValueError("segment_bias: no rows left (missing values or min_n)")
    w = (d[weight].to_numpy(float) if weight not in (None, "equal") else np.ones(len(d)))
    clu_name = cluster or next((c for c in ("group", "item") if c in d.columns), None)
    if clu_name is not None and clu_name not in d.columns:
        raise ValueError(f"segment_bias: cluster column {clu_name!r} not found")
    kc = pd.factorize(d[clu_name])[0] if clu_name else np.arange(len(d))
    ks, seg_names = pd.factorize(seg_key)
    C, S = int(kc.max()) + 1, len(seg_names)
    y = pd.to_numeric(d["truth"]).to_numpy(float)

    def sums(v):
        out = np.zeros((C, S))
        np.add.at(out, (kc, ks), v)
        return out

    Sw, Sy = sums(w), sums(w * y)
    Sp = {a: sums(w * pd.to_numeric(d[c]).to_numpy(float)) for a, c in cols.items()}
    point = {a: float(v[0]) for a, v in _seg_stats(np.ones((1, C)), Sw, Sy, Sp, equal).items()}
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(C, np.full(C, 1.0 / C), size=B).astype(float)
    reps = _seg_stats(cnt, Sw, Sy, Sp, equal)
    if C < 2:
        # one cluster: every resample is the sample itself, the CI collapses on the point and p would read 1/B
        reps = {a: np.full(B, np.nan) for a in reps}
    if C < AZ_KUME:
        warnings.warn(f"segment_bias: only {C} bootstrap clusters ({clu_name or 'row'}); the CI is not meaningful "
                      f"below {AZ_KUME}" + ("; CI and p are left empty" if C < 2 else ""), UserWarning, stacklevel=2)
    res = {}
    for a in cols:
        lo, hi = _boot.ga(reps[a], nokta=point[a], yontem=ci)
        res[a] = {"value": point[a], "lo": lo, "hi": hi,
                  "point_outside_ci": bool(np.isfinite(lo) and not lo <= point[a] <= hi)}
    names = list(cols)
    diffs = []
    for b in names[1:]:
        a = names[0]
        f = reps[a] - reps[b]
        lo, hi = _boot.ga(f, nokta=point[a] - point[b], yontem=ci)
        diffs.append({"a": a, "b": b, "diff": point[a] - point[b], "lo": lo, "hi": hi, "p": _boot.p_degeri(f),
                      "a_better": None if point[a] == point[b] else bool(point[a] < point[b])})
    W = Sw.sum(0)
    tab = pd.DataFrame({"segment": list(map(str, seg_names)), "n": np.bincount(ks, minlength=S),
                        "weight": (np.full(S, 1.0 / S) if equal else W / W.sum()), "truth_mean": Sy.sum(0) / W})
    for a in names:
        tab[f"{a}_mean"] = Sp[a].sum(0) / W
        tab[f"{a}_abs_err"] = np.abs(tab[f"{a}_mean"] - tab["truth_mean"])
    if any(r["point_outside_ci"] for r in res.values()):
        warnings.warn("segment_bias: a single-arm point estimate lies outside its bootstrap CI (absolute-value "
                      "metric); use the paired difference", UserWarning, stacklevel=2)
    return {"metric": "segment_bias_" + ("equal" if equal else "n_weighted" if weight is None else "weighted"),
            "arms": res, "diff": diffs, "segments": tab, "cluster": clu_name or "row", "n_clusters": C, "B": B,
            "dropped": dropped, "note": ABS_CI_NOTE}


# ------------------------------------------------------------------ pooled_distribution
def _dist_vec(v, where: str) -> np.ndarray:
    if isinstance(v, str):
        v = json.loads(v)
    if v is None or isinstance(v, dict) or np.isscalar(v):
        raise ValueError(f"pooled_distribution: {where} is not a distribution vector (list / array / JSON list): {v!r}")
    x = np.asarray(v, dtype=float).ravel()
    if not np.isfinite(x).all() or (x < 0).any() or x.sum() <= 0:
        raise ValueError(f"pooled_distribution: {where} needs finite shares >= 0 with a positive sum, got {x.tolist()}")
    return x / x.sum()


def _alias(frame: pd.DataFrame, name: str) -> str:
    """Column name in the frame, accepting the Turkish alias of an English EvalFrame name (and back)."""
    if name in frame.columns:
        return name
    for c in ({v: k for k, v in COLUMN_ALIASES.items()}.get(name), COLUMN_ALIASES.get(name)):
        if c is not None and c in frame.columns:
            return c
    raise ValueError(f"column {name!r} not in the frame")


def pooled_distribution(frame: pd.DataFrame, weight_col: str = "n", by: str | Sequence[str] = "item") -> pd.DataFrame:
    """Segment-level distribution rows (one row per item x segment) -> one national row per item: the truth and
    every pred_ column become sum_s n_s p_s / sum_s n_s, each segment vector normalised to sum 1 first.

    weight_col: the segment size column (e.g. respondents); >= 0, finite, positive total per item.
    by: the item key (str or list); a split column is added to the key when present (train and test cells are
    never pooled together). A retest column with vectors is pooled too. Vectors of one item must have the same
    length in every row and column; a missing cell raises. Output columns: by, [split], truth, pred_*, [retest],
    weight_col (summed), n_cells. Lists are returned (JSON-ready).

        nat = ev.pooled_distribution(cells, weight_col="n", by="item")
        ev.evaluate(nat, ours="llm", task="distribution", primary="one_minus_tvd")    # national 1 - TVD

    evaluate(aggregate_by=) averages groups with equal weight and is for binary / continuous tasks; this is the
    n-weighted national distribution of TALP D6 (#3a: national TVD and net approval were written by hand).
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("pooled_distribution: frame must be a non-empty DataFrame")
    keys = [_alias(frame, b) for b in ([by] if isinstance(by, str) else list(by))]
    for s in ("split", "bolum"):
        if s in frame.columns and s not in keys:
            keys.append(s)
    wcol = _alias(frame, weight_col)
    w = pd.to_numeric(frame[wcol], errors="coerce").to_numpy(float)
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError(f"pooled_distribution: weights {weight_col!r} must be finite and >= 0")
    tcol = _alias(frame, "truth")
    cols = [tcol] + [c for c in frame.columns if isinstance(c, str) and c.startswith((PRED_PREFIX, ON_EK))]
    for r in ("retest", "tekrar"):
        if r in frame.columns and frame[r].map(lambda v: isinstance(v, (list, tuple, np.ndarray, str))).all():
            cols.append(r)
    if len(cols) < 2:
        raise ValueError("pooled_distribution: at least one pred_<method> column is required")
    rows = []
    for key, idx in frame.groupby(keys, sort=False, dropna=False).indices.items():
        key = key if isinstance(key, tuple) else (key,)
        name = ", ".join(f"{k}={v!r}" for k, v in zip(keys, key))
        ww = w[idx]
        if ww.sum() <= 0:
            raise ValueError(f"pooled_distribution: weights of {name} sum to 0")
        row = dict(zip(keys, key))
        L = None
        for c in cols:
            vs = [_dist_vec(frame[c].iat[i], f"{c} at {name}") for i in idx]
            lens = {len(v) for v in vs} | ({L} if L is not None else set())
            if len(lens) != 1:
                raise ValueError(f"pooled_distribution: vectors of {name} do not align (lengths {sorted(lens)} "
                                 f"across rows or columns; column {c!r})")
            L = len(vs[0])
            row[c] = (ww @ np.vstack(vs) / ww.sum()).tolist()
        row[wcol] = float(ww.sum())
        row["n_cells"] = int(len(idx))
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ choice_frame
def _softmax(u: np.ndarray) -> np.ndarray:
    e = np.exp(u - u.max())
    return e / e.sum()


def choice_frame(long_df: pd.DataFrame, task_col: str, alt_col: str, chosen_col: str,
                 pred_cols: str | Sequence[str] | Mapping[str, str], *, scores: str = "prob") -> pd.DataFrame:
    """Long choice table (one row per task x alternative, a 0/1 chosen flag, one score column per method) -> one
    row per task: truth = the chosen alternative, pred_<method> = probability vector over all alternatives, for
    evaluate(task="multiclass", primary="logloss").

    pred_cols: column name(s) (a pred_ prefix is dropped from the method name) or {method: column}.
    scores: "prob" (>= 0, renormalised within the task) or "utility" (softmax within the task, e.g. logit utilities).
    Classes = every alternative in the table, sorted as text (the order evaluate uses), in `frame.attrs["classes"]`;
    an alternative missing from a task gets probability 0 there. Pass the classes on, or an alternative that is never
    chosen drops out of the class set: ev.evaluate(cf, task="multiclass", primary="logloss",
    classes=cf.attrs["classes"]). Columns constant within every task (person id, split, ...) are carried over.
    Exactly one chosen alternative per task, unique (task, alternative) pairs and complete scores are required.

        cf = ev.choice_frame(long, "task", "alt", "chosen", ["pred_llm", "pred_mnl"], scores="prob")
        ev.evaluate(cf, ours="llm", task="multiclass", primary="logloss", classes=cf.attrs["classes"])

    Why: TALP D8 #5, the long (task, alternative) format had no converter.
    """
    if scores not in ("prob", "utility"):
        raise ValueError("choice_frame: scores must be 'prob' or 'utility'")
    for c in (task_col, alt_col, chosen_col):
        if c not in long_df.columns:
            raise ValueError(f"choice_frame: column {c!r} not in the table")
    if isinstance(pred_cols, Mapping):
        arms = {str(k): v for k, v in pred_cols.items()}
    else:
        arms = {(c[len(PRED_PREFIX):] if c.startswith(PRED_PREFIX) else c): c
                for c in ([pred_cols] if isinstance(pred_cols, str) else list(pred_cols))}
    missing = [c for c in arms.values() if c not in long_df.columns]
    if missing or not arms:
        raise ValueError(f"choice_frame: prediction columns missing: {missing or 'none given'}")
    d = long_df.reset_index(drop=True)
    dup = d.duplicated([task_col, alt_col])
    if dup.any():
        raise ValueError(f"choice_frame: duplicate (task, alternative) rows, e.g. "
                         f"{d.loc[dup, [task_col, alt_col]].iloc[0].tolist()}")
    ch = pd.to_numeric(d[chosen_col].astype(object).map(lambda v: float(v) if isinstance(v, (bool, np.bool_)) else v),
                       errors="coerce")
    if ch.isna().any() or not ch.isin([0, 1]).all():
        raise ValueError(f"choice_frame: {chosen_col!r} must be 0/1 (or bool)")
    per = ch.groupby(d[task_col], sort=False).sum()
    bad = per[per != 1]
    if len(bad):
        raise ValueError(f"choice_frame: {len(bad)} task(s) without exactly one chosen alternative, e.g. "
                         f"{bad.index[0]!r} ({int(bad.iloc[0])} chosen)")
    classes = sorted(pd.unique(d[alt_col]).tolist(), key=str)
    pos = {a: i for i, a in enumerate(classes)}
    skip = {task_col, alt_col, chosen_col, *arms.values()}
    other = [c for c in d.columns if c not in skip]
    const = [c for c in other if (d.groupby(task_col, sort=False)[c].nunique(dropna=False) <= 1).all()]
    rows = []
    for t, idx in d.groupby(task_col, sort=False).indices.items():
        g = d.iloc[idx]
        row = {task_col: t, **{c: g[c].iat[0] for c in const}}
        row["truth"] = g[alt_col].to_numpy()[ch.to_numpy()[idx] == 1][0]
        at = np.array([pos[a] for a in g[alt_col]])
        for m, c in arms.items():
            s = pd.to_numeric(g[c], errors="coerce").to_numpy(float)
            if not np.isfinite(s).all():
                raise ValueError(f"choice_frame: {c!r} has a missing or non-finite score in task {t!r}")
            if scores == "prob":
                if (s < 0).any() or s.sum() <= 0:
                    raise ValueError(f"choice_frame: {c!r} needs probabilities >= 0 with a positive sum in task {t!r}")
                p = s / s.sum()
            else:
                p = _softmax(s)
            v = np.zeros(len(classes))
            v[at] = p
            row[PRED_PREFIX + m] = v
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["classes"] = classes
    return out
