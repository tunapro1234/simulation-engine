"""English entry points over the ported kanit evidence layer.

The algorithms live in the verbatim-ported modules of this package (Turkish internal names):
    sema        input schema, task detection          metrik      metric families per task
    bootstrap   paired cluster bootstrap              merdiven    cross-fitted baseline ladder + shuffle control
    kalibrasyon recalibration, reliability, ECE       analiz      segment error, spread (homogenisation)
    denetim     leakage / honesty audits              ezber       memorization audit (LLM knows the data?)
    orneklem    effective sample multiplier           bedel       PPI++ "worth N humans"
    kart        evidence card dict + badge rule       onkayit     pre-registration seal (SHA-256 chain)
    cekirdek    degerlendir(): table -> report dict   pano        single-file HTML dashboard

This module only translates names: English EvalFrame columns (simengine.types) and the Turkish research
aliases are both accepted; arguments are English. The report dict keeps the original schema
(see docs/modules/evidence.md and `summary()` for an English flat view), so results equal the original
kanit package on the same input.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..types import PRED_PREFIX, normalize_frame
from . import bedel as _bedel
from . import bootstrap as _boot
from . import denetim as _denetim
from . import ezber as _ezber
from . import kalibrasyon as _kal
from . import kart as _kart
from . import merdiven as _merdiven
from . import metrik as _metrik
from . import onkayit as _onkayit
from . import orneklem as _orneklem
from . import pano as _pano
from .cekirdek import SURUM, degerlendir
from .sema import ON_EK, TEST_ADLARI, gercek_yok, hazirla

VERSION = SURUM
DOMINANT_BLOCK_SHARE = 0.20  # balance_col: one block above this share of scored rows -> block-equal-weighted check
MULTIPLIER_MAX_ROWS = 2000   # single-item PPI++ multiplier above this many rows: fixed-seed subsample

# ------------------------------------------------------------------ name translation
TASK_TO_INTERNAL = {"binary": "ikili", "multiclass": "cok_sinif", "distribution": "dagilim",
                    "continuous": "surekli", "ab": "ab"}
TASK_TO_ENGLISH = {v: k for k, v in TASK_TO_INTERNAL.items()}

# English EvalFrame column -> internal (kanit) column. `truth`/`pred_` etc. come from simengine.types;
# the rest are kanit extras with an English spelling.
COLUMN_TO_INTERNAL = {
    "truth": "gercek", "split": "bolum", "group": "grup", "item": "madde", "time": "zaman",
    "unit": "birim", "retest": "tekrar",
    "weight": "agirlik", "fold": "kat", "arm": "kol", "options": "secenekler",
}
SPLIT_TO_INTERNAL = {"train": "egitim"}
REVERSE_PREFIX = ("reverse_", "ters_")   # position-swap column for pair prompts (A/B order swapped)

METRIC_TO_INTERNAL = {
    "accuracy": "dogruluk", "balanced_accuracy": "dengeli_dogruluk", "kappa": "kappa", "auc": "auc",
    "macro_f1": "macro_f1", "brier": "brier", "logloss": "logloss", "ece": "ece",
    "one_minus_tvd": "bir_eksi_tvd", "1-tvd": "bir_eksi_tvd", "simbench": "simbench", "wasserstein": "wasserstein",
    "js": "js", "ks": "ks", "mode_match": "mod_eslesme",
    "one_minus_mae": "bir_eksi_mae", "1-mae": "bir_eksi_mae", "pearson": "pearson", "spearman": "spearman",
    "mae": "mae", "rmse": "rmse", "r2": "r2", "within_item_corr": "madde_ici_kor", "demeaned_r2": "demeaned_r2",
    "winner_top1": "kazanan_top1", "pair_sign": "cift_isaret",
    "within_spearman": "spearman_ici", "effect_corr": "etki_korelasyonu", "scale_error": "olcek_sapmasi",
    "wmape": "wmape", "wape": "wmape", "bias": "yanlilik",
}
METRIC_TO_ENGLISH = {}
for _en, _tr in METRIC_TO_INTERNAL.items():
    METRIC_TO_ENGLISH.setdefault(_tr, _en)

BADGE_TO_ENGLISH = {"yuksek": "high confidence", "kararsiz": "inconclusive", "bekliyor": "awaiting truth"}
BELOW_BASELINE = "below baseline"
# primary metrics that rank or correlate: a cross-fitted constant baseline can score below chance on these
RANK_METRICS = {"auc": 0.5, "pearson": 0.0, "spearman": 0.0, "madde_ici_kor": 0.0, "spearman_ici": 0.0,
                "etki_korelasyonu": 0.0, "kappa": 0.0, "r2": 0.0, "demeaned_r2": 0.0}
STATUS_TO_ENGLISH = {"tamam": "ok", "uyari": "warning", "hata": "error", "atlandi": "skipped"}
CEILING_KIND = {"insan_tekrar": "human retest", "model": "oracle/model", "orneklem_gurultusu": "sampling noise (variance components)"}
NOISE_CEILINGS = ("variance_components", "split_half")   # same estimator; split_half = older name
NOISE_CEILING_REPEATS = 200     # random half splits (the D3 rehearsal tavan.py used 200)
MEMORIZATION_CAVEAT_EN = "gain may come from recognition/memorization"
CAVEAT_REASON_EN = {"ezber_denetimi": "memorization audit warning", "kor_kol": "the blind arm loses most of the gain"}
ROW_TO_ENGLISH = {"taban": "Strongest simple baseline", "ham": "Raw method", "kalibre": "Ours",
                  "tavan": "Human retest ceiling", "bosluk": "Gap closed (ours - base)/(ceiling - base)",
                  "bedel": "Worth N humans (+100 real answers)"}


def to_internal(df: pd.DataFrame) -> pd.DataFrame:
    """English EvalFrame or Turkish research frame -> the internal kanit column convention."""
    out = normalize_frame(df)          # Turkish aliases -> English (simengine.types contract)
    ren = {c: COLUMN_TO_INTERNAL[c] for c in out.columns if c in COLUMN_TO_INTERNAL}
    for c in out.columns:
        if not isinstance(c, str):
            continue
        if c.startswith(PRED_PREFIX):
            ren[c] = ON_EK + c[len(PRED_PREFIX):]
        elif c.startswith(REVERSE_PREFIX[0]):
            ren[c] = "ters_" + c[len(REVERSE_PREFIX[0]):]
    out = out.rename(columns=ren)
    if "bolum" in out.columns:
        out["bolum"] = out["bolum"].map(lambda s: SPLIT_TO_INTERNAL.get(s, s))
    return out


def _method(name: str | None) -> str | None:
    if name is None:
        return None
    for p in (PRED_PREFIX, ON_EK):
        if name.startswith(p):
            return name[len(p):]
    return name


def _methods(names: Iterable[str] | None) -> list[str] | None:
    return None if names is None else [_method(n) for n in names]


FORECAST_TASKS = ("forecast", "tahmin")   # handled inside evidence: continuous + forecast behaviour (see evaluate)
AGGREGATES = ("mean", "sum")
CI_METHODS = ("percentile", "basic")


def _task(task: str | None) -> str | None:
    if task in FORECAST_TASKS:
        return "surekli"
    return None if task is None else TASK_TO_INTERNAL.get(task, task)


PREREG_TEXT_SUFFIXES = (".md", ".txt", ".yaml", ".yml", ".json", ".toml")
PREREG_TEXT_MAX_BYTES = 1_000_000


def _iso_time(z) -> float | None:
    try:
        return datetime.fromisoformat(str(z)).timestamp() if z else None
    except ValueError:
        return None


def _prereg_doc_entry(f: Path, records=()) -> dict | None:
    """One pre-registration text -> {"metin", "zaman", "dosya", "sha256", "zaman_kaynagi"}. zaman is
    - the lock time of a research.lock_prereg lock `<doc>.lock.json` whose sha256 matches the file ("kilit"), else
    - the earliest seal record that carries this file's sha256 (seal(prereg_doc=)) ("muhur"), else
    - the file mtime ("mtime"; a git checkout or copy resets it, so a locked/sealed doc is safer).
    A lock with a different sha256 sets "kilit_uyusmuyor" (content changed after the lock)."""
    try:
        if not f.is_file() or f.suffix.lower() not in PREREG_TEXT_SUFFIXES or f.name.endswith(".lock.json") \
                or f.stat().st_size > PREREG_TEXT_MAX_BYTES:
            return None
        raw = f.read_bytes()
        ent = {"metin": raw.decode("utf-8", errors="replace"), "zaman": f.stat().st_mtime, "dosya": f.name,
               "sha256": hashlib.sha256(raw).hexdigest(), "zaman_kaynagi": "mtime"}
    except OSError:
        return None
    lp = Path(str(f) + ".lock.json")
    if lp.is_file():
        try:
            lock = json.loads(lp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            lock = None
        if isinstance(lock, dict) and lock.get("sha256") == ent["sha256"]:
            z = _iso_time(lock.get("locked_at_utc") or lock.get("locked_at"))
            if z is not None:
                ent.update(zaman=z, zaman_kaynagi="kilit")
        elif isinstance(lock, dict):
            ent["kilit_uyusmuyor"] = True
    if ent["zaman_kaynagi"] == "mtime":
        z = [_iso_time(r.get("zaman_utc") or r.get("zaman")) for r in records
             if isinstance(r, dict) and (r.get("onkayit_belgesi") or {}).get("sha256") == ent["sha256"]]
        z = [x for x in z if x is not None]
        if z:
            ent.update(zaman=min(z), zaman_kaynagi="muhur")
    return ent


def _prereg_texts(prereg, records=()) -> list[dict]:
    """Small text files (.md/.txt/.yaml/.yml/.json/.toml) of a prereg registry directory, or the file itself:
    searched for the `ours` name by the selection-bias audit (see `_prereg_doc_entry` for the time used). A text
    newer than the latest seal does not count as pre-registration. Seal chains (.jsonl) and lock files are skipped."""
    if prereg is None or isinstance(prereg, (list, tuple)):
        return []
    p = Path(prereg)
    files = [p] if p.is_file() else (sorted(p.iterdir()) if p.is_dir() else [])
    return [e for e in (_prereg_doc_entry(f, records) for f in files) if e is not None]


def _as_list(x) -> list:
    return [] if x is None else list(x) if isinstance(x, (list, tuple)) else [x]


def _prereg_sources(prereg, prereg_doc) -> tuple[list | None, list[dict]]:
    """prereg= / prereg_doc= -> (seal records or None, text entries). prereg: seal record dicts, registry
    directories (their chain + text files) and text files, alone or mixed in a list. A chain file (zincir.jsonl /
    chain.jsonl) stands for its directory. A seal that recorded a document (seal(prereg_doc=)) pulls that document
    in from the registry's parent directory when it exists there."""
    items, docs = _as_list(prereg), [Path(x) for x in _as_list(prereg_doc)]
    for f in docs:
        if not f.is_file():
            raise FileNotFoundError(f"prereg_doc {f.name!r} does not exist")
    records, dirs, files = [], [], []
    for x in items:
        if isinstance(x, dict):
            records.append(x)
            continue
        q = Path(x)
        if q.is_file() and q.name in (_onkayit.ZINCIR_ADI, RESEARCH_CHAIN):
            q = q.parent
        (files if q.is_file() else dirs).append(q)
    for q in dirs:
        if not chain_paths(q):
            warnings.warn(f"prereg registry {str(q)!r} has no {_onkayit.ZINCIR_ADI} (evidence.seal / se.seal) "
                          f"or {RESEARCH_CHAIN} (research.seal): no seal is attached", UserWarning, stacklevel=3)
        records.extend(chain(q))
    if len(dirs) + bool(any(isinstance(x, dict) for x in items)) > 1:
        records.sort(key=lambda r: str(r.get("zaman_utc") or r.get("zaman") or ""))
    for q in dirs:   # documents named by a seal record, next to the registry
        for r in chain(q):
            ad = (r.get("onkayit_belgesi") or {}).get("dosya")
            if ad and (q.parent / ad).is_file():
                files.append(q.parent / ad)
    texts, seen = [], set()
    for q in dirs:
        for f in sorted(q.iterdir()) if q.is_dir() else []:
            if f.is_file() and f.resolve() not in seen and (e := _prereg_doc_entry(f, records)) is not None:
                seen.add(f.resolve())
                texts.append(e)
    for f in files + docs:
        key = f.resolve()
        if key in seen:
            continue
        seen.add(key)
        e = _prereg_doc_entry(f, records)
        if e is not None:
            texts.append(e)
    return (records if prereg is not None or docs else None), texts


def _metric(name: str | None) -> str | None:
    return None if name is None else METRIC_TO_INTERNAL.get(name, name)


def prepare(df: pd.DataFrame, task: str | None = None, methods: Sequence[str] | None = None,
            classes: list | None = None):
    """Validate the table, detect the task, decode vector columns -> `sema.Veri` (internal names)."""
    return hazirla(to_internal(df), _task(task), _methods(methods), classes)


# ------------------------------------------------------------------ evaluate
def evaluate(df: pd.DataFrame, title: str = "", ours: str | None = None, raw: str | None = None, *,
             dataset: str = "", task: str | None = None, methods: Sequence[str] | None = None,
             labels: dict | None = None, primary: str | None = None, B: int = 1000, B_aggregate: int = 200,
             seed: int = 0, calibrate: Sequence[str] | None = None, cutoff: str | None = None,
             memorization: str | dict | None = None, multiplier: bool = True, limits: Sequence[str] | None = None,
             prereg: list | str | os.PathLike | None = None, classes: list | None = None, baselines: bool = True,
             multiplier_methods: Sequence[str] | None = None, circular: Sequence[str] | None = None,
             level: str | None = None, source: str | None = None,
             unlabeled_baselines: Sequence[str] | None = None, baselines_oof: Sequence[str] | None = None,
             ceiling_col: str | None = None, ceiling_label: str | None = None, aggregate_by: str | None = None,
             multiplier_max_rows: int | None = MULTIPLIER_MAX_ROWS, offset_col: str | None = None,
             noise_ceiling: str | None = None, noise_ceiling_repeats: int = NOISE_CEILING_REPEATS,
             blind_col: str | None = None, compare: Sequence | None = None, ours_uses_llm: bool | None = None,
             aggregate: str = "mean", ci: str = "percentile", n_configs: int | None = None,
             block_col: str | None = None, point_pairs: Sequence | None = None,
             prereg_doc: str | os.PathLike | Sequence | None = None, split: str | None = None,
             balance_col: str | None = None, dominance_share: float = DOMINANT_BLOCK_SHARE) -> dict:
    """Long table -> evidence report (baseline ladder, paired cluster-bootstrap CIs, calibration, segments,
    audits, effective-sample multiplier, evidence card). Same numbers as kanit.degerlendir.

    ours / raw: method names without the `pred_` prefix. Name `ours` BEFORE looking at results: if omitted the
    last method is used and a selection-bias warning is added to the audits.
    task: binary | multiclass | distribution | continuous | ab (auto-detected when None) | forecast.
        forecast (demand / time-series forecasting): continuous metrics plus wmape and bias, default primary wmape;
        the persona shuffle control, the human retest ceiling and the PPI++ "worth N humans" multiplier are skipped
        and the card says "not applicable" (no human answers, no person information). A model ceiling still works
        via ceiling_col=.
    primary: metric name (English alias or internal); default per task, macro-F1 when the majority class > 60 %.
        Continuous ratio metrics: "wmape" (= "wape", sum|y - p| / sum|y|, lower is better; the cluster bootstrap
        resamples numerator and denominator together, ratio of sums per replicate) and "bias" (sum p / sum y - 1,
        signed diagnostic, never primary). Passing primary="wmape" adds both to any continuous evaluation.
    B / B_aggregate: bootstrap replicates for per-unit / aggregate metrics. Cluster = group > item > row.
    memorization: result dict of `memorization_audit` or a path to its JSON.
    prereg: seal records, a registry directory (its chain is attached to the report and its small text files are
        searched for the ours name), a text file (e.g. "ONKAYIT.md"), or a list/tuple mixing them
        (prereg=[".onkayit", "ONKAYIT.md"]; a zincir.jsonl / chain.jsonl path stands for its directory).
    prereg_doc: pre-registration document(s) (hand-written ONKAYIT.md or research.Prereg(...).write + lock_prereg),
        path or list; FileNotFoundError when missing. A text counts only when written no later than the latest seal.
        Its time is the research.lock_prereg lock (`<doc>.lock.json`) when the lock's sha256 matches the file, else
        the earliest seal that recorded the document's sha256 (seal(..., prereg_doc=)), else the file mtime (a git
        checkout resets it, so an unlocked, unsealed doc can look "newer than the seal"). A lock whose sha256 does
        not match warns and adds a card limit. Documents named by a seal are picked up next to the registry
        directory automatically. The card lists the documents with their sha256 and time source.
    split: a constant split label ("test" / "holdout" / "deger" / "degerlendirme") written into a split column
        when the frame has none (TEST-only frames, distribution frames whose cells are all held out), so the split
        audit runs; an audit row records that the split was declared, not derived. Ignored with a warning when the
        frame already has a split column; other labels raise ValueError.
    balance_col: block column whose shares are checked (e.g. "country"; default: block_col when given). When one
        block holds more than max(dominance_share, 1.5 / n_blocks) of the scored rows (default share 0.20), an
        audit warning row and a card note give the primary metric of every method person-weighted (as reported)
        and block-equal-weighted: with aggregate_by, each group's value is the equal-weighted mean of its per-block
        means (not computed for aggregate="sum"); without it, row weights 1/n_block for per-row metrics and the
        mean of per-block values for whole-sample metrics (rank / correlation / AUC). `report["baskin_blok"]`,
        `summary()["dominant_block"]`. With an explicit balance_col an "ok" row is written when no block dominates.
    circular: priors trained on the same population's target labels; excluded from "worth N humans".
    unlabeled_baselines: prediction columns that never saw target labels (e.g. persistence) -> baseline rungs.
    baselines_oof: prediction columns trained on the target labels but cross-fitted (out-of-fold) -> baseline rungs.
        The badge and headline compare ours with the strongest baseline (automatic + declared). When nothing is
        declared and the strongest is the random/marginal rung, the card says "only trivial baselines".
    ceiling_col: a model/oracle ceiling column ("pred_x" or any column name) instead of the human `retest`
        ceiling; the card labels the ceiling type and the gap closed is computed from it. ceiling_label names it.
    aggregate_by: evaluate at group level ("group" or any column): truth and predictions are averaged per group
        (and per item / split when present) and the bootstrap resamples groups. Binary/continuous tasks only.
    aggregate: "mean" (default) or "sum" with aggregate_by: sum per group instead of the mean (hierarchical totals,
        e.g. aggregate_by="store_dept_week" for a store x dept x week WMAPE).
    ci: "percentile" (default) or "basic" (2*theta - bootstrap quantiles) for column, ladder, headline and paired
        CIs. When the point estimate falls outside the percentile CI (possible for ratio/aggregate metrics) a warning
        is raised and the card gets an audit row and a flag.
    n_configs: number of configurations tried before picking ours; written in the selection-bias audit (and >= 4
        with ours the best adds the optimism note). The audit says ours was "named in advance" only when the ours
        name occurs in the seal records or the text files of the prereg= registry; otherwise "not verified".
    multiplier_max_rows: single-item frames larger than this compute the PPI++ multiplier on a fixed-seed
        subsample of this many rows (None: whole table, slow). A warning is raised for large frames.
    offset_col: a per-row offset column (e.g. the member's training mean): subtracted from truth, retest, the
        model ceiling and every prediction BEFORE aggregation, so aggregate_by="movie_id" scores the group residual
        (mean(truth − offset) vs mean(pred − offset)). Numeric truth only; the task becomes continuous.
    noise_ceiling: "variance_components" (or its older name "split_half", same estimator; needs aggregate_by):
        sampling-noise ceiling of group means over ALL scored groups. reliability = 1 - mean(s2_g / n_g) / var(group
        means), s2_g = within-group variance (pooled within variance for one-row groups); ceiling = sqrt(reliability)
        for pearson, (6/pi) asin(sqrt(reliability) / 2) for spearman (normal theory), reliability for r2. Other
        primaries raise NotImplementedError. A split-half cross-check (`noise_ceiling_repeats` random within-group
        half splits, Spearman-Brown) is reported but not used: it is biased low for small or odd-sized groups. The
        card labels the ceiling type and the gap closed uses it.
    blind_col: the blind (no names / titles) twin of ours, e.g. "pred_ours_blind": adds a paired row ours − blind
        and flags when the blind arm keeps less than half of ours' gain over the strongest baseline.
    compare: registered comparisons / drop conditions, [("P1", "L1"), ("P2", None, "P2 ≈ 0"), ...]: each (a, b)
        pair gets a paired difference on the same bootstrap resamples (b None: a's value against 0); shown in the
        report (`kayitli_karsilastirmalar`), `summary()["comparisons"]`, the card and the dashboard.
    ours_uses_llm: when the memorization audit is in warning state and ours includes an LLM arm, the badge reason
        gets the caveat "gain may come from recognition/memorization". None (default) assumes yes whenever a
        memorization result is passed; False turns the caveat off, and the missing memorization / contamination
        audits are then labelled "no LLM arm: not applicable" instead of raising the "not proven: leakage/memorization"
        flag.
    cutoff: the LLM's training cutoff date (e.g. cutoff="2025-03-01") for the contamination audit; it needs a
        `time` (or `zaman`) column and compares the primary metric before vs after the cutoff.
    block_col: block bootstrap column (e.g. the forecast origin or week column, "time" accepted): every distinct value
        is one block and the bootstrap is two-way: in each replicate the blocks and the clusters (grup / item / row)
        are resampled independently with replacement and a row's weight is block count × cluster count, so the column,
        ladder, headline and paired-difference CIs (compare=, blind_col=) cover both block-to-block variation and
        series dependence across blocks. Non-overlapping blocks; for a moving-block flavour with block length L pass a
        coarser column (e.g. week // L). With aggregate_by= the column must be constant within each group. NaN in the
        column raises ValueError. One distinct value: not applied (default clusters) and an audit row says so; fewer
        than 10 blocks: audit warning + card note (below 5: "CI not reliable across blocks").
    Single forecast origin (task="forecast"): when an `origin` (else `time`) column has one distinct value, whatever
        block_col is, an audit warning row and a card note say "single forecast origin: CIs cover series sampling only,
        not origin-to-origin variation; use methods.forecast.backtest(origins=[...])".
    point_pairs: None (default: off), "auto" (pair ours with `<ours>_median` if that prediction column exists) or
        [(mean_col, median_col), ...] prediction columns (with or without `pred_`) of the same model as a mean and a
        median point forecast. In forecast mode with an L1 primary (mae / wmape) the card shows a row per pair with the
        primary metric and bias of both, plus the observed direction. Explicit columns are always validated
        (ValueError); outside forecast + L1 an explicit list is skipped with a UserWarning.
    Missing truth: rows with no truth are dropped (noted); when the truth column is absent or all missing the
    report is an "awaiting truth" card (seals + prediction summary; no ladder, calibration or bootstrap).
    """
    if aggregate not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}, not {aggregate!r}")
    if ci not in CI_METHODS:
        raise ValueError(f"ci must be one of {CI_METHODS}, not {ci!r}")
    forecast = task in FORECAST_TASKS
    if forecast:
        multiplier = False
        task = "continuous"
    prereg, prereg_texts = _prereg_sources(prereg, prereg_doc)
    doc_limits = []
    for e in prereg_texts:
        if e.get("kilit_uyusmuyor"):
            warnings.warn(f"evaluate: pre-registration document {e['dosya']!r} differs from its lock "
                          f"({e['dosya']}.lock.json): its file time is used", UserWarning, stacklevel=2)
            doc_limits.append(f"ön kayıt belgesi {e['dosya']} kilidinden sonra değişmiş (sha256 eşleşmiyor); "
                              "zamanı dosya zamanı (pre-registration document changed after its lock)")
    tmp = None
    ezber_json = None
    if isinstance(memorization, dict):
        fd, tmp = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(memorization, f, ensure_ascii=False)
        ezber_json = tmp
    elif memorization is not None:
        ezber_json = str(memorization)
    if noise_ceiling is not None:
        if noise_ceiling not in NOISE_CEILINGS:
            raise ValueError(f"noise_ceiling must be one of {NOISE_CEILINGS}, not {noise_ceiling!r}")
        if aggregate_by is None:
            raise ValueError(f"noise_ceiling={noise_ceiling!r} needs aggregate_by= (the groups whose means carry the noise)")
        if ceiling_col is not None:
            raise ValueError("pass noise_ceiling or ceiling_col, not both")
        if int(noise_ceiling_repeats) < 1:
            raise ValueError("noise_ceiling_repeats must be >= 1")
    d = to_internal(df)
    split_declared = None
    if split is not None:
        sp = str(SPLIT_TO_INTERNAL.get(split, split))
        if sp.lower() not in TEST_ADLARI:
            raise ValueError(f"split= fills a constant test split; use one of {TEST_ADLARI}, not {split!r}")
        if "bolum" in d.columns:
            warnings.warn("evaluate: the frame already has a split column; split= is ignored", UserWarning, stacklevel=2)
        else:
            d = d.assign(bolum=sp)
            split_declared = sp
    if doc_limits:
        limits = list(limits or []) + doc_limits
    blok = None
    if block_col is not None:
        blok = COLUMN_TO_INTERNAL.get(block_col, block_col)
        if blok not in d.columns:
            raise ValueError(f"block_col {block_col!r} is not a column of the frame")
    denge = blok
    if balance_col is not None:
        denge = COLUMN_TO_INTERNAL.get(balance_col, balance_col)
        if denge not in d.columns:
            raise ValueError(f"balance_col {balance_col!r} is not a column of the frame")
    methods_i = _methods(methods)
    labels_i = {_method(k): v for k, v in (labels or {}).items()}
    tavan_sutun = tavan_etiket = None
    if ceiling_col is not None:
        d, tavan_sutun, tavan_etiket = _ceiling_column(d, ceiling_col, ceiling_label, labels_i, ours, raw)
        if methods_i:
            methods_i = [m for m in methods_i if ON_EK + m != ceiling_col and m != _method(ceiling_col)]
    ofset = None
    if offset_col is not None:
        d, ofset = _offset(d, offset_col, _task(task))
        task = "continuous"
        limits = list(limits or []) + [ofset["note"]]
    toplama = None
    tavan_yarim = None
    d_rows = d          # row (person) level frame, before aggregate_by
    if aggregate_by is not None:
        d, toplama = _aggregate(d, aggregate_by, _task(task),
                                halves=int(noise_ceiling_repeats) if noise_ceiling else None, seed=seed, how=aggregate,
                                keep=(blok,) if blok else ())
        tavan_yarim = toplama.pop("halves", None)
        task = "continuous"
        limits = list(limits or []) + [toplama["note"]]
        level = level or "grup"
    kiyas = _compare_list(compare)
    if multiplier and len(d) > (multiplier_max_rows or MULTIPLIER_MAX_ROWS) and not gercek_yok(d):
        tek = "madde" not in d or d["madde"].nunique() <= 1
        warnings.warn(
            f"evaluate: {len(d)} rows with multiplier=True. " + (
                f"The PPI++ 'worth N humans' multiplier is computed on a fixed-seed {multiplier_max_rows}-row "
                "subsample (noted on the card)." if tek and multiplier_max_rows else
                "The PPI++ multiplier runs on the whole table and can take minutes.")
            + " Pass multiplier=False to skip it.", UserWarning, stacklevel=2)
    try:
        rep = degerlendir(
            d, title, dataset, gorev=_task(task), yontemler=methods_i, bizim=_method(ours),
            ham=_method(raw), etiketler=labels_i, birincil=_metric(primary),
            B=B, B_toplu=B_aggregate, tohum=seed, kalibre=_methods(calibrate), kesim=cutoff, ezber_json=ezber_json,
            carpan=multiplier, sinirlar=list(limits) if limits else None, onkayit=prereg, siniflar=classes,
            baseline=baselines, carpan_yontemler=_methods(multiplier_methods), dongusel=_methods(circular),
            duzey=level, girdi_kaynagi=source, etiketsiz_tabanlar=_methods(unlabeled_baselines),
            oof_tabanlar=_methods(baselines_oof), tavan_sutun=tavan_sutun, tavan_etiket=tavan_etiket,
            carpan_en_fazla=multiplier_max_rows, tavan_yarim=tavan_yarim, kor=_method(blind_col),
            kiyaslar=kiyas, ezber_llm=ours_uses_llm, tahmin_gorevi=forecast, ga_yontemi=ci,
            n_yapilandirma=n_configs, onkayit_metin=prereg_texts, blok=blok)
    finally:
        if tmp:
            os.unlink(tmp)
    if toplama:
        rep["toplama"] = {k: v for k, v in toplama.items() if k != "note"}
    if prereg_texts:
        rep["onkayit_belgeleri"] = [{"dosya": e["dosya"], "sha256": e["sha256"], "zaman_kaynagi": e["zaman_kaynagi"],
                                     "zaman_utc": datetime.fromtimestamp(e["zaman"]).astimezone().isoformat(timespec="seconds"),
                                     **({"kilit_uyusmuyor": True} if e.get("kilit_uyusmuyor") else {})}
                                    for e in prereg_texts]
    if split_declared and isinstance(rep.get("denetim"), list):
        rep["denetim"].append({"kontrol": "bölme beyanı (split=)", "durum": "tamam", "deger": split_declared,
                               "ayrinti": f"split={split!r} verildi: bütün satırlar '{split_declared}' (değerlendirme) "
                                          "sayıldı; eğitim satırı yok, bölme beyanı kullanıcınındır "
                                          "(declared split: every row is held out)"})
    if denge is not None and rep.get("sonuclar"):
        _dominant_block(rep, d_rows, denge, explicit=balance_col is not None, by=aggregate_by, how=aggregate,
                        share=dominance_share)
    if ofset:
        rep["ofset"] = {k: v for k, v in ofset.items() if k != "note"}
    if point_pairs is not None and rep.get("sonuclar"):
        rep["_forecast"] = bool(forecast)
        try:
            pairs = _point_pairs(rep, point_pairs, ours)
        finally:
            rep.pop("_forecast", None)
        if pairs:
            rep["ortalama_medyan"] = pairs
    return rep


L1_METRICS = ("mae", "wmape")


def _mean_median_here(r: dict) -> str:
    """Observed direction for one mean/median row (L1 lower/higher, |bias| smaller/larger)."""
    a, b, ba, bb = r.get("mean_value"), r.get("median_value"), r.get("mean_bias"), r.get("median_bias")
    parts = []
    if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
        parts.append(f"{r['metric']} {'lower' if b < a else 'higher' if b > a else 'equal'} with the median "
                     f"({b - a:+.3f})")
    if ba is not None and bb is not None and np.isfinite(ba) and np.isfinite(bb):
        parts.append(f"|bias| {'smaller' if abs(bb) < abs(ba) else 'larger' if abs(bb) > abs(ba) else 'equal'} "
                     f"({ba:+.3f} → {bb:+.3f})")
    return f"`{r['median']}` vs `{r['mean']}`: " + (", ".join(parts) or "not computable")


def _point_pairs(rep: dict, point_pairs, ours) -> list[dict]:
    """Mean vs median point forecast rows (forecast mode, L1 primary): primary metric and bias of both columns.

    point_pairs None: off (default, old behaviour). "auto": pair ours with `<ours>_median` when that column exists.
    An explicit list is always validated; when the mode or primary metric means it is skipped, a UserWarning says so."""
    if point_pairs is None:
        return []
    res = rep.get("sonuclar") or {}
    col = lambda c: ON_EK + _method(c)
    if isinstance(point_pairs, str):
        if point_pairs != "auto":
            raise ValueError(f"point_pairs must be None, 'auto' or [(mean_col, median_col), ...], not {point_pairs!r}")
        o = _method(ours) if ours else (rep.get("kart") or {}).get("bizim")
        pairs = [(o, o + "_median")] if o and col(o + "_median") in res else []
        explicit = False
    else:
        pairs = [tuple(p) for p in point_pairs]
        explicit = True
    for a, b in pairs:
        for name in (a, b):
            if col(name) not in res:
                raise ValueError(f"point_pairs column {name!r} is not an evaluated prediction column")
    if not pairs:
        return []
    if not rep.get("_forecast") or rep.get("birincil") not in L1_METRICS:
        if explicit:
            warnings.warn("point_pairs ignored: the mean/median row needs task='forecast' and an L1 primary "
                          f"(mae / wmape); here primary={rep.get('birincil')!r}"
                          + ("" if rep.get("_forecast") else ", task is not forecast"), UserWarning, stacklevel=3)
        return []
    out = []
    for a, b in pairs:
        ca, cb = col(a), col(b)
        m = lambda c, ad: (res[c]["metrikler"].get(ad) or {})
        out.append({"ortalama": ca, "medyan": cb, "metrik": rep["birincil"],
                    "ortalama_deger": m(ca, rep["birincil"]).get("deger"), "medyan_deger": m(cb, rep["birincil"]).get("deger"),
                    "ortalama_yanlilik": m(ca, "yanlilik").get("deger"), "medyan_yanlilik": m(cb, "yanlilik").get("deger")})
    return out


def _compare_list(compare) -> list | None:
    """compare=[(a, b), (a, b, label), {"a", "b", "label"}] -> [(a, b | None, label | None)] (prefix-free names)."""
    if not compare:
        return None
    out = []
    for c in compare:
        if isinstance(c, dict):
            a, b, lab = c.get("a"), c.get("b"), c.get("label")
        elif isinstance(c, (tuple, list)) and len(c) in (2, 3):
            a, b, lab = c[0], c[1], (c[2] if len(c) == 3 else None)
        else:
            raise ValueError(f"compare entries are (a, b) or (a, b, label) tuples, not {c!r}")
        if a is None:
            raise ValueError(f"compare entry {c!r} has no first column")
        out.append((_method(a), _method(b), lab))
    return out


def _offset(d: pd.DataFrame, col: str, task: str | None):
    """Subtract a per-row offset from truth, retest, model ceiling and every prediction (residual evaluation)."""
    if task in ("cok_sinif", "dagilim", "ab"):
        raise ValueError(f"offset_col works for numeric (binary/continuous) truth, not {TASK_TO_ENGLISH.get(task, task)}")
    src = next((c for c in (col, COLUMN_TO_INTERNAL.get(col, col)) if c in d.columns), None)
    if src is None:
        raise ValueError(f"offset_col {col!r} is not a column of the frame")
    if src == "gercek" or (isinstance(src, str) and src.startswith(ON_EK)):
        raise ValueError(f"offset_col {col!r} is the truth or a prediction column")
    off = pd.to_numeric(d[src], errors="coerce")
    if off.isna().all() and d[src].notna().any():
        raise ValueError(f"offset_col {col!r} is not numeric")
    d = d.copy()
    cols = [c for c in d.columns if c in ("gercek", "tekrar", "tavan_model") or (isinstance(c, str) and c.startswith(ON_EK))]
    for c in cols:
        x = pd.to_numeric(d[c], errors="coerce")
        if c != "gercek" and x.isna().all() and d[c].notna().any():
            raise ValueError(f"offset_col needs numeric predictions; {c!r} is not numeric")
        if c == "gercek" and x.isna().all() and d[c].notna().any():
            raise ValueError("offset_col needs a numeric truth")
        d[c] = x - off
    n_na = int(off.isna().sum())
    if n_na:
        warnings.warn(f"offset_col {col!r}: {n_na} row(s) have no offset; their truth and predictions become missing "
                      "(rows with missing truth are dropped)", UserWarning, stacklevel=3)
    note = (f"Ofsetli değerlendirme (offset_col={col}): gerçek ve bütün tahminlerden satır ofseti çıkarıldı "
            f"(ortalama ofset {float(off.mean()):.3f}) ve grup düzeyine indirmeden önce; metrikler artık "
            "(gerçek − ofset) üzerinde.")
    return d, {"column": col, "n_rows": int(len(d)), "n_missing": n_na, "mean": float(off.mean()) if off.notna().any() else None,
               "columns": [str(c) for c in cols], "note": note}


def evaluate_aggregate(df: pd.DataFrame, by: str = "group", title: str = "", ours: str | None = None,
                       raw: str | None = None, **kw) -> dict:
    """Group-level evaluation: `evaluate(df, ..., aggregate_by=by)` (e.g. 20 countries, Spearman, bootstrap over
    groups)."""
    return evaluate(df, title, ours, raw, aggregate_by=by, **kw)


def _ceiling_column(d: pd.DataFrame, col: str, label, labels: dict, ours, raw):
    """Move the ceiling column out of the method columns -> (frame, internal column, label)."""
    cands = [col, COLUMN_TO_INTERNAL.get(col, col)]
    if col.startswith(PRED_PREFIX):
        cands.append(ON_EK + col[len(PRED_PREFIX):])
    cands.append(ON_EK + col)
    src = next((c for c in cands if c in d.columns), None)
    if src is None:
        raise ValueError(f"ceiling_col {col!r} is not a column of the frame")
    name = _method(src) if src.startswith(ON_EK) else src
    if name in (_method(ours), _method(raw)):
        raise ValueError(f"ceiling_col {col!r} is also ours/raw")
    d = d.rename(columns={src: "tavan_model"})
    return d, "tavan_model", label or labels.get(name, name)


def _split_halves(y: np.ndarray, codes: np.ndarray, G: int, R: int, seed: int):
    """R random within-group half splits -> (A, B) half means, R x G (NaN for groups with < 2 rows)."""
    ok = np.isfinite(y) & (codes >= 0)
    gi, yi = codes[ok], y[ok]
    n_g = np.bincount(gi, minlength=G)
    start = np.concatenate([[0], np.cumsum(n_g)[:-1]])
    half_n = n_g // 2
    rng = np.random.default_rng([seed, 2112])
    A, B = np.full((R, G), np.nan), np.full((R, G), np.nan)
    rank = np.empty(len(yi), int)
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in range(R):
            o = np.lexsort((rng.random(len(yi)), gi))
            rank[o] = np.arange(len(yi)) - start[gi[o]]
            h = rank < half_n[gi]
            ca, cb = np.bincount(gi[h], minlength=G), np.bincount(gi[~h], minlength=G)
            a = np.bincount(gi[h], yi[h], G) / ca
            b = np.bincount(gi[~h], yi[~h], G) / cb
            a[(ca == 0) | (cb == 0)] = np.nan
            b[(ca == 0) | (cb == 0)] = np.nan
            A[s], B[s] = a, b
    return A, B, n_g


def _within_variance(y: np.ndarray, codes: np.ndarray, G: int):
    """Per group: number of finite truth rows and within-group variance (ddof=1; NaN for groups with < 2 rows)."""
    ok = np.isfinite(y) & (codes >= 0)
    gi, yi = codes[ok], y[ok]
    n = np.bincount(gi, minlength=G).astype(float)
    m = np.bincount(gi, yi, G) / np.where(n > 0, n, 1)
    ss = np.bincount(gi, (yi - m[gi]) ** 2, G)
    with np.errstate(invalid="ignore", divide="ignore"):
        s2 = np.where(n >= 2, ss / (n - 1), np.nan)
    return n, s2


def _aggregate(d: pd.DataFrame, by: str, task: str | None, halves: int | None = None, seed: int = 0,
               how: str = "mean", keep: Sequence[str] = ()):
    """Average (how="mean") or sum (how="sum") truth, predictions, retest and model ceiling per group (x item x split
    when present). halves: number of random within-group half splits for the split-half noise ceiling (None: not
    computed)."""
    if how not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}, not {how!r}")
    col = COLUMN_TO_INTERNAL.get(by, by)
    if col not in d.columns:
        raise ValueError(f"aggregate_by {by!r} is not a column of the frame")
    if task in ("cok_sinif", "dagilim", "ab"):
        raise ValueError(f"aggregate_by works for binary/continuous tasks, not {TASK_TO_ENGLISH.get(task, task)}")
    # sum = hierarchical total of the group (e.g. store x dept x week): never split by item, otherwise each
    # (group, item) cell is one row and nothing is summed. The item survives only when constant within the group.
    keys = [col] + [k for k in (("bolum",) if how == "sum" else ("madde", "bolum")) if k in d.columns and k != col]
    if how == "sum" and halves:
        raise ValueError("noise_ceiling works on group means; it is not available with aggregate='sum' "
                         "(the half-split means are on a different scale from the group totals)")
    num = [c for c in d.columns if c in ("gercek", "tekrar", "tavan_model") or (isinstance(c, str) and c.startswith(ON_EK))]
    out = d[keys].copy()
    for c in num:
        x = pd.to_numeric(d[c], errors="coerce")
        if c != "gercek" and x.isna().all() and d[c].notna().any():
            raise ValueError(f"aggregate_by needs numeric columns; {c!r} is not numeric")
        out[c] = x
    if "gercek" in d.columns and out["gercek"].isna().all() and d["gercek"].notna().any():
        raise ValueError("aggregate_by needs a numeric truth (binary or continuous)")
    g = out.groupby(keys, sort=True, dropna=False)
    if how == "sum":
        # a group total is defined only when every row of the column is present: a missing prediction must not
        # count as a zero forecast while the truth is summed over all rows (biased WMAPE / bias)
        agg = g[num].sum(min_count=1)
        eksik = out[num].isna().groupby([out[k] for k in keys], sort=True, dropna=False).any()
        agg = agg.mask(eksik.reindex(agg.index).fillna(False).astype(bool)).reset_index()
    else:
        agg = g[num].mean().reset_index()
    agg.insert(len(keys), "n_satir", g.size().to_numpy())
    if how == "sum" and "madde" in d.columns and col != "madde":
        m = d.groupby(keys, sort=True, dropna=False)["madde"]
        if (m.nunique(dropna=False) <= 1).all():
            agg["madde"] = m.first().to_numpy()
    # segment columns constant within the group survive (first value)
    segs = [c for c in d.columns if isinstance(c, str) and (c == "segment" or c.startswith("seg_"))]
    for c in segs:
        s = d.groupby(keys, sort=True, dropna=False)[c]
        if (s.nunique(dropna=False) <= 1).all():
            agg[c] = s.first().to_numpy()
    for c in keep:   # e.g. block_col: must be constant within the group
        if c in agg.columns:
            continue
        s = d.groupby(keys, sort=True, dropna=False)[c]
        if not (s.nunique(dropna=False) <= 1).all():
            raise ValueError(f"block_col {c!r} varies within aggregate_by groups; aggregate by a finer key or "
                             "use a block column that is constant within each group")
        agg[c] = s.first().to_numpy()
    if col != "grup":
        agg["grup"] = agg[col].astype(str)
    n_groups = int(agg[col].nunique())
    yarim = None
    if halves:
        codes = g.ngroup().to_numpy()
        A, B, n_g = _split_halves(out["gercek"].to_numpy(float) if "gercek" in out else np.full(len(out), np.nan),
                                  codes, len(agg), halves, seed)
        agg["_tavan_satir"] = np.arange(len(agg))
        n_v, s2 = _within_variance(out["gercek"].to_numpy(float) if "gercek" in out else np.full(len(out), np.nan),
                                   codes, len(agg))
        yarim = {"A": A, "B": B, "n": n_v, "s2": s2, "sutun": "_tavan_satir", "R": int(halves), "tohum": seed,
                 "ortalama_grup_boyu": float(n_g[n_g > 0].mean()) if (n_g > 0).any() else None}
    note = (f"Grup düzeyi değerlendirme (aggregate_by={by}): {len(d)} satır {n_groups} grubun "
            + ("toplamına (aggregate='sum', hiyerarşik toplam; madde ayrımı yok, bir sütunda grubun herhangi bir "
               "satırı eksikse o grubun o sütun toplamı eksik sayılır) indirildi " if how == "sum"
               else "ortalamasına indirildi ")
            + 
            f"(eşit satır ağırlığı, {len(agg)} değerlendirme satırı); bootstrap gruplar üzerinden. Grup içi örneklem "
            "gürültüsü ortalamalarda kalır. Otomatik tabanlar (genel ortalama, demografi) grup satırları üzerinde "
            "çapraz uydurmalıdır: kat dışı ortalama tutulan grupların ortalamasıyla ters ilişkili olduğundan sıra "
            "metriklerinde (Spearman) bilgisizken bile sıfırın belirgin altında çıkabilir; anlamlı karşılaştırma için "
            "baselines_oof / unlabeled_baselines ile grup düzeyi bir taban bildirin.")
    info = {"by": by, "column": col, "n_rows": int(len(d)), "n_groups": n_groups, "n_eval_rows": int(len(agg)),
            "aggregate": how,
            "auto_baselines_cross_fitted_bias": True, "note": note}
    if yarim is not None:
        info["halves"] = yarim
    return agg, info


def _metric_points(frame: pd.DataFrame, gorev: str | None, metric: str, cols: list[str], weight_col=None,
                   block_col=None) -> dict:
    """Point value of one metric per prediction column on an internal frame (no bootstrap). weight_col /
    block_col: per-row metrics are weighted 1 / n_block, whole-sample metrics are the mean of per-block values."""
    v = hazirla(frame, gorev, [c[len(ON_EK):] for c in cols])
    d = v.df
    madde = pd.factorize(d["madde"])[0] if "madde" in d else None
    m = next((x for x in _metrik.aile(v.gorev, True, madde is not None, True, True) if x.ad == metric), None)
    if m is None:
        return {}
    out = {}
    for c in cols:
        try:
            y, p = _metrik.diziler(v, c)
            if block_col is None:
                val = _boot._guvenli(m, y, p, madde) if m.tur == "toplu" else m.deger(y, p)
            elif m.tur == "toplu":
                b = d[block_col].astype(str).to_numpy()
                vals = []
                for k in pd.unique(b):
                    i = np.flatnonzero(b == k)
                    if len(i) >= 2:
                        vals.append(_boot._guvenli(m, y[i], p[i], None if madde is None else madde[i]))
                vals = [x for x in vals if x is not None and np.isfinite(x)]
                val = float(np.mean(vals)) if vals else np.nan
            else:
                n_b = d.groupby(block_col, dropna=False)[block_col].transform("size").to_numpy(float)
                val = m.deger(y, p, w=1.0 / n_b)
            out[c] = None if val is None or not np.isfinite(val) else float(val)
        except (ValueError, TypeError, ZeroDivisionError, IndexError, FloatingPointError):
            out[c] = None
    return out


def _dominant_block(rep: dict, d: pd.DataFrame, col: str, *, explicit: bool, by: str | None, how: str,
                    share: float = DOMINANT_BLOCK_SHARE) -> None:
    """Dominant-block check on the row-level frame: audit row, card note and `rep["baskin_blok"]` (in place)."""
    rows = d
    if "bolum" in rows.columns:
        rows = rows[rows["bolum"].astype(str).str.lower().isin(TEST_ADLARI).to_numpy()]
    if "gercek" in rows.columns:
        rows = rows[rows["gercek"].notna().to_numpy()]
    counts = rows[col].astype(str).value_counts(dropna=False)
    if len(counts) < 2:
        return
    n_blocks, top, top_share = int(len(counts)), str(counts.index[0]), float(counts.iloc[0] / counts.sum())
    thr = max(float(share), 1.5 / n_blocks)
    dominant = top_share > thr
    if not (dominant or explicit):
        return
    metric = rep.get("birincil")
    cols = [c for c in rows.columns if isinstance(c, str) and c.startswith(ON_EK)]
    gorev = rep.get("gorev")
    values, kind = {}, None
    if dominant and metric and cols:
        x = rows.assign(**{col: rows[col].astype(str)})
        if by is not None:
            agg_col = COLUMN_TO_INTERNAL.get(by, by)
            keys = [agg_col] + [k for k in ("madde", "bolum") if k in x.columns and k != agg_col]
            if how == "mean" and col not in keys:
                num = ["gercek"] + cols
                x = x[keys + [col]].join(x[num].apply(pd.to_numeric, errors="coerce"))
                person = x.groupby(keys, dropna=False)[num].mean().reset_index()
                equal = (x.groupby(keys + [col], dropna=False)[num].mean()
                         .groupby(level=list(range(len(keys))), dropna=False).mean().reset_index())
                a = _metric_points(person, "surekli", metric, cols)
                b = _metric_points(equal, "surekli", metric, cols)
                kind = "iki_adim"   # group value = equal-weighted mean of per-block means
        else:
            a = _metric_points(x, gorev, metric, cols)
            m = next((q for q in _metrik.aile(gorev or "surekli", True, "madde" in x, True, True) if q.ad == metric), None)
            b = _metric_points(x, gorev, metric, cols, block_col=col)
            kind = "blok_ortalamasi" if m is not None and m.tur == "toplu" else "agirlik_1_n_blok"
        if kind:
            values = {c[len(ON_EK):]: {"kisi_agirlikli": a.get(c), "blok_esit": b.get(c)} for c in cols}
    f3 = lambda v: "—" if v is None else f"{v:.3f}".replace(".", ",")
    ours = ((rep.get("manset") or {}).get("bizim") or {}).get("sutun") or ""
    ours = ours[len(ON_EK):] if ours.startswith(ON_EK) else ours
    pct = f"{100 * top_share:.1f}".replace(".", ",")
    detail = f"{col} = {top}: değerlendirilen satırların %{pct}'i ({n_blocks} blok; eşik %{100 * thr:.0f})"
    kind_tr = {"iki_adim": "grup değeri blok içi ortalamaların eşit ağırlıklı ortalaması",
               "agirlik_1_n_blok": "satır ağırlığı 1/n_blok", "blok_ortalamasi": "blok değerlerinin ortalaması"}
    if dominant and ours in values:
        v = values[ours]
        detail += (f"; {metric} bizim ({ours}) kişi ağırlıklı {f3(v['kisi_agirlikli'])}, blok eşit ağırlıklı "
                   f"{f3(v['blok_esit'])} ({kind_tr[kind]}) (dominant block: person-weighted vs block-equal-weighted)")
    elif dominant and by is not None and how == "sum":
        detail += "; aggregate='sum' ile blok eşit ağırlıklı değer hesaplanmaz (dominant block)"
    elif dominant:
        detail += " (dominant block)"
    else:
        detail += " (no dominant block)"
    rep.setdefault("denetim", []).append({"kontrol": "baskın blok (dominant block)", "durum": "uyari" if dominant else "tamam",
                                          "ayrinti": detail, "deger": top_share})
    rep["baskin_blok"] = {"sutun": col, "blok": top, "pay": top_share, "n_blok": n_blocks, "esik": thr,
                          "baskin": dominant, "metrik": metric, "agirlik": kind, "degerler": values}
    if dominant:
        note = "Baskın blok: " + detail
        rep["sinirlar"] = list(rep.get("sinirlar") or []) + [note]
        if isinstance(rep.get("kanit_kart"), dict):
            rep["kanit_kart"]["yanilgi"] = list(rep["kanit_kart"].get("yanilgi") or []) + [note]


def save_json(report: dict, path) -> Path:
    """Write the report as JSON (NaN already mapped to null)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return p


def _box(d):
    return None if not d else {"column": d.get("sutun"), "label": d.get("etiket"), "value": d.get("deger"),
                               "lo": d.get("alt"), "hi": d.get("ust")}


def _below_baseline(k: dict) -> bool:
    f = k.get("fark") or {}
    return f.get("ust") is not None and f["ust"] < 0


def _badge_english(k: dict) -> str | None:
    """English badge. The research rule (kart.rozet) gives "kararsiz" when ours − baseline lies entirely below 0;
    in English that case is reported as "below baseline" so a quoted badge does not understate a clear loss.
    The internal code (`kanit_kart["rozet"]`) is unchanged."""
    if k.get("rozet") == "kararsiz" and _below_baseline(k):
        return BELOW_BASELINE
    return BADGE_TO_ENGLISH.get(k.get("rozet"), k.get("rozet"))


def _seals(report: dict) -> list[dict]:
    return [{"index": r.get("sira"), "sha256": r.get("sha256"), "time_utc": r.get("zaman_utc") or r.get("zaman"),
             "rows": r.get("satir"), "description": r.get("aciklama"), "file_sha256": r.get("dosya_sha256")}
            for r in report.get("onkayit") or [] if isinstance(r, dict)]


def _f3(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.3f}"


def _headline_en(s: dict) -> str | None:
    """English one-line headline built from the summary fields (dot decimals)."""
    o, b = s.get("ours"), s.get("best_baseline")
    if s.get("awaiting_truth"):
        z = s.get("seals") or []
        return (f"awaiting truth: predictions of {len(s.get('predictions') or {})} method(s)"
                + (f", sealed ({len(z)} record(s), last sha256 {str(z[-1].get('sha256'))[:12]})" if z else ", NOT sealed")
                + "; no metric until the truth is opened")
    if not o or o.get("value") is None:
        return None
    ci = lambda d: f" [{_f3(d.get('lo'))}, {_f3(d.get('hi'))}]" if d.get("lo") is not None and d.get("hi") is not None else ""
    out = f"{s.get('primary_metric')}: ours ({o.get('column')}) {_f3(o['value'])}{ci(o)}"
    if b and b.get("value") is not None:
        out += f" vs strongest simple baseline ({b.get('column')}) {_f3(b['value'])}{ci(b)}"
    d = s.get("ours_minus_baseline") or {}
    if d.get("fark") is not None:
        out += f"; ours − baseline {d['fark']:+.3f}"
        if d.get("alt") is not None and d.get("ust") is not None:
            out += f" [{d['alt']:+.3f}, {d['ust']:+.3f}] (paired, positive = better)"
    c = s.get("ceiling")
    kind = s.get("ceiling_kind") or "human retest"
    out += f"; {kind} ceiling {_f3(c['value'])}" if c and c.get("value") is not None else "; no human retest ceiling"
    if s.get("only_trivial_baselines"):
        out += "; only trivial baselines (random/marginal) were compared"
    return out


def summary(report: dict) -> dict:
    """English flat view of the headline numbers of a report (no new computation)."""
    m = report.get("manset") or {}
    k = report.get("kanit_kart") or {}
    worth = next((r for r in k.get("satirlar") or [] if r.get("tur") == "bedel"), None)
    out = {
        "title": report.get("baslik"),
        "task": "forecast" if report.get("tahmin_gorevi") else TASK_TO_ENGLISH.get(report.get("gorev"), report.get("gorev")),
        "primary_metric": METRIC_TO_ENGLISH.get(report.get("birincil"), report.get("birincil")),
        "primary_metric_internal": report.get("birincil"),
        "ours": _box(m.get("bizim")), "raw": _box(m.get("ham")), "best_baseline": _box(m.get("en_iyi_baseline")),
        "best_other_method": _box(m.get("en_iyi_rakip")), "ceiling": _box(m.get("tavan")),
        "gap_closed": m.get("bosluk_kapanan"), "ratio_to_ceiling": m.get("tavana_oran"),
        "baseline_ratio_to_ceiling": m.get("taban_tavana_oran"),
        "worth_humans_at_100": None if not worth or worth.get("deger") is None else
        {"multiplier": worth["deger"], "lo": worth.get("alt"), "hi": worth.get("ust"), "note": worth.get("not")},
        "ours_minus_baseline": k.get("fark"),
        "badge": _badge_english(k), "badge_code": k.get("rozet"), "below_baseline": _below_baseline(k),
        "badge_reason": k.get("rozet_gerekce"),
        "flags": k.get("etiketler") or [],
        "audits": [{"check": d.get("kontrol"), "status": STATUS_TO_ENGLISH.get(d.get("durum"), d.get("durum")),
                    "detail": d.get("ayrinti"), "value": d.get("deger")} for d in report.get("denetim") or []],
        "ladder": [{"column": r.get("sutun"), "label": r.get("etiket"), "kind": r.get("tur"), "value": r.get("deger"),
                    "lo": r.get("alt"), "hi": r.get("ust"), "diff_prev": r.get("fark_onceki"), "p": r.get("p")}
                   for r in report.get("merdiven") or []],
        "headline": m.get("cumle"),
        "seals": _seals(report),
        "ceiling_kind": CEILING_KIND.get(report.get("tavan_turu") or ("insan_tekrar" if m.get("tavan") else None)),
        "declared_baselines": list((report.get("tabanlar") or {}).get("bildirilen") or []),
        "only_trivial_baselines": bool((report.get("tabanlar") or {}).get("yalniz_basit")),
        "awaiting_truth": bool(report.get("gercek_bekleniyor")),
        "predictions": report.get("tahmin_ozeti"),
        "aggregate": report.get("toplama"),
        "offset": report.get("ofset"),
        "noise_ceiling": report.get("gurultu_tavani"),
        "comparisons": [_comparison_en(c) for c in report.get("kayitli_karsilastirmalar") or []],
        "blind_arm": report.get("kor_kol"),
        "badge_caveat": _caveat_en(report),
    }
    if report.get("blok_bootstrap"):
        b = report["blok_bootstrap"]
        out["block_bootstrap"] = {"column": b.get("sutun"), "n_blocks": b.get("n_blok")}
    if report.get("baskin_blok"):
        b = report["baskin_blok"]
        out["dominant_block"] = {
            "column": b.get("sutun"), "block": b.get("blok"), "share": b.get("pay"), "n_blocks": b.get("n_blok"),
            "threshold": b.get("esik"), "dominant": b.get("baskin"),
            "metric": METRIC_TO_ENGLISH.get(b.get("metrik"), b.get("metrik")),
            "weighting": {"iki_adim": "two-step mean (block means, then equal weights)",
                          "agirlik_1_n_blok": "row weight 1/n_block",
                          "blok_ortalamasi": "mean of per-block values"}.get(b.get("agirlik")),
            "values": {m: {"person_weighted": v.get("kisi_agirlikli"), "block_equal": v.get("blok_esit")}
                       for m, v in (b.get("degerler") or {}).items()}}
    if report.get("onkayit_belgeleri"):
        out["prereg_docs"] = [{"file": e.get("dosya"), "sha256": e.get("sha256"), "time": e.get("zaman_utc"),
                               "time_source": {"kilit": "lock", "muhur": "seal", "mtime": "file mtime"}.get(
                                   e.get("zaman_kaynagi"), e.get("zaman_kaynagi")),
                               "lock_mismatch": bool(e.get("kilit_uyusmuyor"))}
                              for e in report["onkayit_belgeleri"]]
    if report.get("tek_koken"):
        t = report["tek_koken"]
        out["single_origin"] = {"column": t.get("sutun"), "value": t.get("deger"), "note": t.get("not")}
    if report.get("ortalama_medyan"):
        out["mean_median"] = [{"mean": _method(r["ortalama"]), "median": _method(r["medyan"]),
                               "metric": METRIC_TO_ENGLISH.get(r["metrik"], r["metrik"]),
                               "mean_value": r["ortalama_deger"], "median_value": r["medyan_deger"],
                               "mean_bias": r["ortalama_yanlilik"], "median_bias": r["medyan_yanlilik"]}
                              for r in report["ortalama_medyan"]]
    out["headline_en"] = _headline_en(out)
    return out


def _comparison_en(c: dict) -> dict:
    v = c.get("sonuc") or ""
    verdict = ("a better (CI above 0)" if v.startswith("A iyi") else "b better (CI below 0)" if v.startswith("B iyi")
               else "above 0" if v.startswith("0'ın üstünde") else "below 0" if v.startswith("0'ın altında")
               else "no CI" if v == "GA yok" else "unclear (CI includes 0)")
    return {"label": c.get("etiket"), "a": c.get("a"), "b": c.get("b"), "kind": {"kor": "blind", "kayitli": "registered"}
            .get(c.get("tur"), c.get("tur")), "metric": METRIC_TO_ENGLISH.get(c.get("metrik"), c.get("metrik")),
            "diff": c.get("fark"), "lo": c.get("alt"), "hi": c.get("ust"), "p": c.get("p"), "n": c.get("n"),
            "a_better": c.get("a_iyi"), "verdict": verdict}


def _caveat_en(report: dict) -> str | None:
    ez = report.get("ezber_uyarisi")
    if not ez or report.get("gercek_bekleniyor"):
        return None
    k = report.get("kanit_kart") or {}
    if k.get("satirlar") is not None and not _kart.kazanc_var(k.get("fark")):   # no gain over the baseline
        return None
    return f"{MEMORIZATION_CAVEAT_EN} ({', '.join(CAVEAT_REASON_EN.get(n, n) for n in ez.get('nedenler') or [])})"


# ------------------------------------------------------------------ card (markdown) and dashboard
def _num(x, unit=""):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    if unit == "%":
        return f"{100 * x:.0f}%"
    if unit == "×":
        return f"{x:.2f}×"
    return f"{x:.3f}"


def _signed(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.3f}"


def _ci(lo, hi, unit=""):
    if lo is None or hi is None:
        return ""
    return f" [{_num(lo, unit)}, {_num(hi, unit)}]"


def _md(s) -> str:
    return "" if s is None else str(s).replace("|", "\\|").replace("\n", " ")


def _dot(s, lang):
    """lang="en": Turkish comma decimals -> dot decimals inside free text."""
    s = _md(s)
    return re.sub(r"(?<=\d),(?=\d)", ".", s) if lang == "en" else s


def _rank_note(report: dict) -> str | None:
    chance = RANK_METRICS.get(report.get("birincil"))
    if chance is None:
        return None
    low = [r for r in report.get("merdiven") or [] if r.get("sutun") in ("taban_marjinal", "taban_demografi")
           and r.get("deger") is not None and r["deger"] < chance]
    if not low:
        return None
    names = ", ".join(f"`{r['sutun']}`" for r in low)
    return (f"Average baselines ({names}) score below chance ({chance:g}) on {report.get('birincil')}. This is "
            "expected, not a bug: they are cross-fitted (a row never sees its own label), so a leave-one-out "
            "mean or rate moves against the held-out truth. On ranking and correlation metrics, compare ours "
            "with the informative rungs (persistence, learners) rather than with these.")


def card_markdown(report: dict, lang: str = "mixed") -> str:
    """Evidence card as Markdown: badge, the six card rows, the ladder, the audits and the seals.

    lang="mixed" (default) keeps the research card's Turkish labels next to the English ones;
    lang="en" drops the Turkish badge reason and original headline and uses dot decimals in the audit details
    (ladder and audit labels stay as computed)."""
    k = report.get("kanit_kart") or _kart.kart(report)
    s = summary(report)
    en = lang == "en"
    out = [f"# Evidence card: {_md(report.get('baslik') or report.get('veri_adi') or '')}", ""]
    reason = s.get("headline_en") if en else _md(k.get("rozet_gerekce"))
    badge = _badge_english(k)
    if s.get("below_baseline"):
        badge += " (ours − strongest baseline CI is entirely below 0; research rule: inconclusive)"
    if s.get("only_trivial_baselines"):
        badge += (" (only trivial baselines: random/marginal; declare real ones with unlabeled_baselines= or "
                  "baselines_oof=)")
    cav = s.get("badge_caveat")
    out.append(f"**Badge: {badge}.** " + ("" if en else reason) + (f" Caveat: {cav}." if cav else ""))
    if k.get("etiketler"):
        out.append("")
        out.append("Flags: " + ", ".join(_dot(e, lang) for e in k["etiketler"]))
    meta = (f"Task: {s['task'] or '—'} · primary metric: {s['primary_metric'] or '—'}"
            + (f" (`{s['primary_metric_internal']}`)" if s['primary_metric_internal'] else "") + f" · clusters: {k.get('n')}")
    if k.get("kaynak"):
        meta += f" · source: {_md(k.get('kaynak'))}"
    if s.get("aggregate"):
        a = s["aggregate"]
        meta += f" · level: group (aggregate_by={a.get('by')}: {a.get('n_rows')} rows -> {a.get('n_groups')} groups)"
    if s.get("block_bootstrap"):
        b = s["block_bootstrap"]
        meta += (f" · bootstrap: two-way, blocks of `{b['column']}` ({b['n_blocks']} blocks) × clusters"
                 + (" — CI not reliable across blocks (< 5 blocks)" if (b.get("n_blocks") or 0) < 5 else
                    " — few blocks (< 10): coarse CIs" if (b.get("n_blocks") or 0) < 10 else ""))
    if s.get("ceiling_kind"):
        meta += f" · ceiling: {s['ceiling_kind']}"
    if s.get("declared_baselines"):
        meta += " · declared baselines: " + ", ".join(f"`{c}`" for c in s["declared_baselines"])
    out += ["", meta, ""]
    if s.get("single_origin"):
        so = s["single_origin"]
        out += [f"Note: {so['note']} (`{so['column']}` = {_md(so['value'])}).", ""]
    db = s.get("dominant_block")
    if db and db.get("dominant"):
        ours_v = (db.get("values") or {}).get(_method(((report.get("manset") or {}).get("bizim") or {}).get("sutun")) or "")
        out += [f"Note: dominant block `{_md(db['column'])}` = {_md(db['block'])} holds {100 * db['share']:.1f}% of the "
                f"scored rows (threshold {100 * db['threshold']:.0f}%)"
                + (f"; {db['metric']} of ours person-weighted {_num(ours_v['person_weighted'])} vs block-equal-weighted "
                   f"{_num(ours_v['block_equal'])} ({db['weighting']})" if ours_v else "") + ".", ""]
    if s.get("headline_en"):
        out += [f"Summary: {s['headline_en']}", ""]
    out += ["| Row | Method | Value [95% CI] | Note |", "|---|---|---|---|"]
    for r in k.get("satirlar") or []:
        u = r.get("birim") or ""
        row = ROW_TO_ENGLISH.get(r['tur'], r['tur'])
        if r["tur"] == "tavan" and s.get("ceiling_kind") == "oracle/model":
            row = "Oracle/model ceiling"
        elif r["tur"] == "tavan" and s.get("ceiling_kind") == CEILING_KIND["orneklem_gurultusu"]:
            row = "Sampling-noise ceiling (variance components)"
        out.append(f"| {row} | {_md(r.get('etiket'))} | "
                   f"{_num(r.get('deger'), u)}{_ci(r.get('alt'), r.get('ust'), u)} | {_dot(r.get('not'), lang)} |")
    f = k.get("fark")
    if f and f.get("fark") is not None:
        out += ["", f"Ours − strongest baseline (paired, metric direction, positive = better): {f['fark']:+.3f}"
                    + (f" [{f['alt']:+.3f}, {f['ust']:+.3f}]" if f.get("alt") is not None and f.get("ust") is not None else "")]
    if s.get("noise_ceiling") and s["noise_ceiling"].get("tavan") is not None:
        nc = s["noise_ceiling"]
        conv = {"guvenilirlik": "reliability", "karekok_guvenilirlik": "sqrt(reliability)"}.get(
            nc.get("olcek"), "(6/pi) asin(sqrt(reliability)/2)")
        out += ["", f"Ceiling type: sampling noise (variance components) over all {nc.get('n_grup')} scored groups "
                    f"({nc.get('n_tek_satir')} with one row, pooled within variance): reliability {_f3(nc.get('guvenilirlik'))} "
                    f"-> ceiling {_f3(nc.get('tavan'))} ({conv}). Split-half check (not used; biased low for small groups): "
                    f"half-mean {nc.get('korelasyon')} {_f3(nc.get('yari_korelasyon'))} over {nc.get('n_grup_yari')} groups "
                    f"({nc.get('R')} splits) -> Spearman-Brown {_f3(nc.get('spearman_brown'))}."]
    if s.get("mean_median"):
        out += ["", "| Point forecast | Mean column | Median column | " + "Primary (mean → median) | Bias (mean → median) |",
                "|---|---|---|---|---|"]
        for r in s["mean_median"]:
            out.append(f"| mean vs median ({r['metric']}) | `{r['mean']}` | `{r['median']}` | "
                       f"{_num(r['mean_value'])} → {_num(r['median_value'])} | "
                       f"{_signed(r['mean_bias'])} → {_signed(r['median_bias'])} |")
        out += ["", "A median point forecast can lower L1 metrics (MAE / WMAPE) but usually biases totals down "
                    "(bias = sum pred / sum truth − 1); here: " + "; ".join(_mean_median_here(r) for r in s["mean_median"])
                    + ". Report both and pick by the decision's loss."]
    if s.get("comparisons"):
        out += ["", "## Registered comparisons / drop conditions", "",
                "| Comparison | Kind | Metric | A − B [95% CI] | p | n clusters | Verdict |", "|---|---|---|---|---|---|---|"]
        for c in s["comparisons"]:
            ci_ = (f" [{c['lo']:+.3f}, {c['hi']:+.3f}]" if c.get("lo") is not None and c.get("hi") is not None else "")
            dv = "—" if c.get("diff") is None else f"{c['diff']:+.3f}{ci_}"
            p = "—" if c.get("p") is None else f"{c['p']:.3f}"
            out.append(f"| {_md(c.get('label'))} | {c['kind']} | {c.get('metric')} | {dv} | {p} | {c.get('n')} | {c['verdict']} |")
        out += ["", "A − B on the same cluster-bootstrap resamples as the ladder (b empty: A's value against 0). "
                    "For lower-is-better metrics a negative difference favours A."]
    if s["ladder"]:
        out += ["", "## Baseline ladder", "", "| Rung | Kind | Value [95% CI] | Δ vs previous [95% CI] | p |",
                "|---|---|---|---|---|"]
        for r in report.get("merdiven") or []:
            d = "—" if r.get("fark_onceki") is None else f"{r['fark_onceki']:+.3f}{_ci(r.get('fark_alt'), r.get('fark_ust'))}"
            p = "—" if r.get("p") is None else f"{r['p']:.3f}"
            out.append(f"| {_md(r.get('etiket'))} (`{r.get('sutun')}`) | {r.get('tur')} | "
                       f"{_num(r.get('deger'))}{_ci(r.get('alt'), r.get('ust'))} | {d} | {p} |")
    if s.get("predictions"):
        out += ["", "## Predictions (awaiting truth)", "", "| Column | n | missing | mean | sd | min | median | max | top labels |",
                "|---|---|---|---|---|---|---|---|---|"]
        for c, p in s["predictions"].items():
            top = ", ".join(f"{_md(a)} {100 * b:.0f}%" for a, b in (p.get("en_sik") or {}).items())
            out.append(f"| {_md(p.get('etiket'))} (`{c}`) | {p.get('n')} | {p.get('eksik')} | {_num(p.get('ortalama'))} | "
                       f"{_num(p.get('sd'))} | {_num(p.get('en_kucuk'))} | {_num(p.get('medyan'))} | "
                       f"{_num(p.get('en_buyuk'))} | {top or ('vector' if p.get('tur') == 'vektor' else '—')} |")
    if s["audits"]:
        out += ["", "## Audits", "", "| Check | Status | Detail |", "|---|---|---|"]
        for a in s["audits"]:
            out.append(f"| {_md(a['check'])} | {a['status']} | {_dot(a['detail'], lang)} |")
    if s["seals"]:
        out += ["", "## Sealed predictions (pre-registration chain)", "",
                "| # | SHA-256 (canonical CSV) | SHA-256 (file bytes) | Sealed at (UTC) | Rows | Description |",
                "|---|---|---|---|---|---|"]
        for z in s["seals"]:
            fb = f"`{z['file_sha256']}`" if z.get("file_sha256") else "— (sealed from a DataFrame)"
            out.append(f"| {z['index']} | `{z['sha256']}` | {fb} | {_md(z['time_utc'])} | {z['rows']} | {_md(z['description'])} |")
        out += ["", "Canonical CSV = hash of the sorted, normalised table (what verify/grade check); file bytes = sha256 "
                    "of the submitted file as written on disk (what `sha256sum <file>` prints)."]
    else:
        out += ["", "Sealed predictions: none attached (pass prereg=<registry dir> to evaluate)."]
    if s.get("prereg_docs"):
        out += ["", "Pre-registration documents searched for the ours name: " + "; ".join(
            f"{_md(e['file'])} (sha256 `{str(e['sha256'])[:12]}`, time {_md(e['time'])} from {e['time_source']}"
            + (", CHANGED AFTER LOCK" if e["lock_mismatch"] else "") + ")" for e in s["prereg_docs"]) + "."]
    limits = list(k.get("yanilgi") or [])
    if db and db.get("dominant"):   # already shown as the dominant-block note above
        limits = [x for x in limits if not str(x).startswith("Baskın blok:")]
    note = _rank_note(report)
    if limits or note:
        out += ["", "## Limits", ""] + [f"- {_dot(x, lang)}" for x in limits] + ([f"- {note}"] if note else [])
    if s["headline"] and not en:
        out += ["", "Headline (original wording): " + _md(s["headline"])]
    return "\n".join(out) + "\n"


def card(report: dict, path=None, lang: str = "mixed") -> str | dict:
    """Evidence card. path *.json -> the card dict (stage schema, `kanit_kart`); otherwise Markdown
    (lang "mixed" or "en", see `card_markdown`). Returns the content; writes it when `path` is given."""
    if path is not None and str(path).endswith(".json"):
        k = report.get("kanit_kart") or _kart.kart(report)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(k, ensure_ascii=False, indent=1))
        return k
    md = card_markdown(report, lang=lang)
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")
    return md


def dashboard(reports: dict | list[dict], path, title: str = "Kanıt panosu", worth: dict | None = None) -> Path:
    """One or more reports -> single-file HTML dashboard (inline SVG, no external resources, light/dark).
    worth: optional {"n": .., "noktalar": [...]} block of `worth_humans` points shown on top."""
    return _pano.yaz(reports if isinstance(reports, list) else [reports], path, baslik=title,
                     **({"bedel": worth} if worth else {}))


def dashboard_html(reports: dict | list[dict], title: str = "Kanıt panosu", worth: dict | None = None) -> str:
    return _pano.html(reports, title, worth)


# ------------------------------------------------------------------ worth N humans (PPI++)
def worth_humans(data, prior: str | Sequence[str] | None = None, *, n: int = 100, R: int = 400, B: int = 200,
                 seed: int = 0, circular: Sequence[str] = (), style: str = "", dataset: str = "") -> dict | list[dict]:
    """PPI++ "worth N humans": how many real answers is (simulation prior + n real answers) worth when
    estimating a population mean? lambda = Cov(y,f)/((1+n/N)Var(f)) from the labelled sample, clipped to [0,1].

    data: a DataFrame (truth + pred_<prior> columns, optional item: one target per item), a list of
    (y, f) column pairs, or a single (y, f) pair.
    Returns the measured point dict (multiplier `carpan`, CI `carpan_ga`, rho, theory 1/(1-rho^2), coverage,
    n_esdeger) for pairs; for a DataFrame a list of points, one per prior (circular priors skipped).
    """
    if isinstance(data, pd.DataFrame):
        d = to_internal(data)
        yon = [c[len(ON_EK):] for c in d.columns if isinstance(c, str) and c.startswith(ON_EK)]
        sec = [prior] if isinstance(prior, str) else (list(_methods(prior)) if prior else yon)
        sec = [_method(y) for y in sec if _method(y) not in set(_methods(circular) or [])]
        out = []
        for y in sec:
            c = ON_EK + y
            if not (pd.api.types.is_numeric_dtype(d["gercek"]) and pd.api.types.is_numeric_dtype(d[c])):
                continue
            gruplar = [x for _, x in d.groupby("madde")] if "madde" in d else [d]
            q = _bedel.olc([(x["gercek"].to_numpy(float), x[c].to_numpy(float)) for x in gruplar], n=n, R=R, B=B,
                           tohum=seed)
            q.update({"tarz": style, "veri": dataset, "onsel": y, "kaynak": f"{dataset} [{c}]"})
            out.append(q)
        return out
    if isinstance(data, tuple) and len(data) == 2 and np.ndim(data[0]) == 1:
        data = [data]
    return _bedel.olc(list(data), n=n, R=R, B=B, tohum=seed)


def worth_theory_curve(n: int | None = None, N: float | None = None, steps: int = 101) -> list[dict]:
    """Theory curve 1/(1 - k rho^2), k = 1 - n/N (or 1)."""
    return _bedel.egri(n, N, steps)


def sample_multiplier(df: pd.DataFrame, methods: Sequence[str] | dict, *, n_list=(10, 25, 50, 100, 200), R: int = 40,
                      seed: int = 0, B: int = 300, task: str | None = None) -> dict | None:
    """Effective sample multiplier per item (PPI++ and Bayes shrinkage) -> report 'orneklem' block."""
    v = prepare(df, task)
    cols = methods if isinstance(methods, dict) else {m: m for m in methods}
    cols = {ON_EK + _method(c): lab for c, lab in cols.items()}
    return _orneklem.carpan(v, cols, n_listesi=tuple(n_list), R=R, tohum=seed, B=B)


# ------------------------------------------------------------------ metrics, bootstrap, baselines, calibration
def metric_family(task: str, probabilistic: bool = True, item: bool = False, unit_range: bool = False):
    """Metric objects for a task (internal `Metrik`: .ad, .tur birim|toplu, .yon +1/-1, .deger(y, p, item))."""
    return _metrik.aile(_task(task), probabilistic, item, unit_range)


def compute_metrics(df: pd.DataFrame, task: str | None = None, methods: Sequence[str] | None = None) -> dict:
    """Point estimates of the task's metric family for every pred_ column (no bootstrap, no baselines)."""
    v = prepare(df, task, methods)
    d = v.df
    madde = pd.factorize(d["madde"])[0] if "madde" in d else None
    olas = v.gorev != "cok_sinif" or isinstance(d[ON_EK + v.yontemler[0]].iloc[0], np.ndarray)
    birim = v.gorev == "surekli" and d["gercek"].between(0, 1).all()
    fam = _metrik.aile(v.gorev, olas, madde is not None and v.gorev in ("ikili", "surekli"), birim)
    out = {}
    for y_ in v.yontemler:
        y, p = _metrik.diziler(v, ON_EK + y_)
        out[y_] = {METRIC_TO_ENGLISH.get(m.ad, m.ad): _boot._guvenli(m, y, p, madde) if m.tur == "toplu"
                   else m.deger(y, p) for m in fam}
    return out


def hit_at_k(truth, score, item, k: int | Sequence[int] = (1, 3, 5)) -> dict:
    """Share of items whose top-k scored candidates contain a positive (truth == 1). Ties broken by row order.
    Ported from the next-action ranker evaluation (web branch, `topk_isabet`)."""
    t = pd.DataFrame({"y": np.asarray(truth), "s": np.asarray(score, float), "ex": np.asarray(item)})
    t["_r"] = t.groupby("ex")["s"].rank(ascending=False, method="first")
    r = t[t.y == 1].groupby("ex")["_r"].min()
    n = t.ex.nunique()
    ks = [k] if isinstance(k, int) else list(k)
    return {f"hit@{q}": float((r <= q).sum() / n) for q in ks}


Resampler = _boot.Orneklem
ci = _boot.ga
p_value = _boot.p_degeri
bootstrap_metrics = _boot.hesapla


def paired_ci(df: pd.DataFrame, a: str, b: str, metric: str | None = None, *, B: int = 1000, B_aggregate: int = 200,
              seed: int = 0, task: str | None = None) -> dict:
    """Paired cluster bootstrap of metric(a) − metric(b) on the same resamples (cluster = group > item > row)."""
    v = prepare(df, task)
    d = v.df
    madde_var = "madde" in d
    olas = v.gorev != "cok_sinif" or isinstance(d[ON_EK + v.yontemler[0]].iloc[0], np.ndarray)
    birim = v.gorev == "surekli" and d["gercek"].between(0, 1).all()
    fam = {m.ad: m for m in _metrik.aile(v.gorev, olas, madde_var and v.gorev in ("ikili", "surekli"), birim, oran=True)}
    ad = _metric(metric) or ("wmape" if task in FORECAST_TASKS else
                             "bir_eksi_mae" if birim else _metrik.BIRINCIL[v.gorev])
    M = fam[ad]
    ca, cb = ON_EK + _method(a), ON_EK + _method(b)
    girdi = {c: (*_metrik.diziler(v, c), None) for c in (ca, cb)}
    orn = _boot.Orneklem(v.kume, B=B, B_toplu=B_aggregate, tohum=seed, madde_kume=v.kume_adi == "madde")
    nokta, tekrar = _boot.hesapla(orn, girdi, [M], pd.factorize(d["madde"])[0] if madde_var else None,
                                  d["agirlik"].to_numpy(float) if "agirlik" in d else None)
    f = tekrar[ca][ad] - tekrar[cb][ad]
    lo, hi = _boot.ga(f)
    return {"metric": METRIC_TO_ENGLISH.get(ad, ad), "a": nokta[ca][ad], "b": nokta[cb][ad],
            "diff": nokta[ca][ad] - nokta[cb][ad], "lo": lo, "hi": hi, "p": _boot.p_degeri(f),
            "a_better": _a_better(M, nokta[ca][ad], nokta[cb][ad]), "cluster": v.kume_adi,
            "n_clusters": int(v.kume.max()) + 1}


def _a_better(M, va, vb) -> bool | None:
    """a better than b: by direction, or for a signed metric (bias) the one closer to 0 (|a| < |b|)."""
    if M.ad in _metrik.ISARETLI_METRIKLER:
        if va is None or vb is None or not (np.isfinite(va) and np.isfinite(vb)):
            return None
        return bool(abs(va) < abs(vb))
    return bool(M.yon * (va - vb) > 0)


def paired_diff(frame_or_report, a: str, b: str | None, *, metric: str | None = None, aggregate_by: str | None = None,
                group: str | None = None, B: int = 2000, seed: int = 0, offset_col: str | None = None,
                task: str | None = None, aggregate: str = "mean", ci: str = "percentile") -> dict:
    """Paired difference metric(a) − metric(b) with a cluster-bootstrap 95 % CI, for ANY two columns (a drop
    condition such as P1 − L1, or a control such as P1 − P_shuffled). b=None: a's own value and CI (e.g. "P2 ≈ 0").

    frame_or_report: an EvalFrame (or Turkish frame) -> computed here; or a report from `evaluate` -> the pair is
        looked up in its registered comparisons (`evaluate(compare=...)`, `blind_col=`) or in its ours − X
        comparisons, either order (B and seed are then the report's).
    metric: English or internal name (default: the task's primary). aggregate_by: average per group first (as
        `evaluate(aggregate_by=)`; the clusters are then the groups). group: the bootstrap cluster column (default
        group > item > row; with aggregate_by it must be constant within each aggregate group). offset_col: as in
        `evaluate`, subtracted before aggregation. All percentile CIs use B resamples for every metric type.
    metric="wmape" / "bias" work on continuous (and task="forecast") frames: numerator and denominator are resampled
        together. aggregate="sum": sum per aggregate_by group instead of the mean. ci="basic": 2*diff - quantiles.
    Returns {"metric", "a", "b", "value_a", "value_b", "diff", "lo", "hi", "p", "n" (clusters), "cluster",
    "a_better", "B", "source"}. diff is raw a − b: for lower-is-better metrics a negative diff favours a."""
    if isinstance(frame_or_report, dict):
        return _paired_from_report(frame_or_report, a, b, metric)
    d = to_internal(frame_or_report)
    task_i = _task(task)
    if offset_col is not None:
        d, _ = _offset(d, offset_col, task_i)
        task_i = "surekli"
    gc = None
    if group is not None:
        gc = next((c for c in (group, COLUMN_TO_INTERNAL.get(group, group)) if c in d.columns), None)
        if gc is None:
            raise ValueError(f"group {group!r} is not a column of the frame")
    if aggregate_by is not None:
        col = COLUMN_TO_INTERNAL.get(aggregate_by, aggregate_by)
        cmap = None
        if gc is not None and gc != col:
            if col not in d.columns:
                raise ValueError(f"aggregate_by {aggregate_by!r} is not a column of the frame")
            if (d.groupby(col, dropna=False)[gc].nunique(dropna=False) > 1).any():
                raise ValueError(f"group {group!r} must be constant within each aggregate_by={aggregate_by!r} group")
            cmap = d.groupby(col, dropna=False)[gc].first()
        d, _ = _aggregate(d, aggregate_by, task_i, how=aggregate)
        task_i = "surekli"
        if cmap is not None:
            d["grup"] = d[col].map(cmap).astype(str)
    elif gc is not None and gc != "grup":
        d["grup"] = d[gc].astype(str)
    v = hazirla(d, task_i)
    dd = v.df
    madde_var = "madde" in dd
    olas = v.gorev != "cok_sinif" or isinstance(dd[ON_EK + v.yontemler[0]].iloc[0], np.ndarray)
    birim = v.gorev == "surekli" and dd["gercek"].between(0, 1).all()
    if ci not in CI_METHODS:
        raise ValueError(f"ci must be one of {CI_METHODS}, not {ci!r}")
    fam = {m.ad: m for m in _metrik.aile(v.gorev, olas, madde_var and v.gorev in ("ikili", "surekli"), birim, oran=True)}
    ad = _metric(metric) or ("wmape" if task in FORECAST_TASKS else
                             "bir_eksi_mae" if birim else _metrik.BIRINCIL[v.gorev])
    if ad not in fam:
        raise ValueError(f"metric {metric!r} is not available for this task: {sorted(fam)}")
    M = fam[ad]
    cols = [ON_EK + _method(x) for x in (a, b) if x is not None]
    for c in cols:
        if c not in dd:
            raise ValueError(f"column {c!r} is not in the frame (need pred_<name>)")
    girdi = {c: (*_metrik.diziler(v, c), None) for c in cols}
    orn = _boot.Orneklem(v.kume, B=B, B_toplu=B, tohum=seed, madde_kume=v.kume_adi == "madde")
    nokta, tekrar = _boot.hesapla(orn, girdi, [M], pd.factorize(dd["madde"])[0] if madde_var else None,
                                  dd["agirlik"].to_numpy(float) if "agirlik" in dd else None)
    ca = cols[0]
    cb = cols[1] if len(cols) > 1 else None
    va = nokta[ca][ad]
    vb = nokta[cb][ad] if cb else None
    f = tekrar[ca][ad] - (tekrar[cb][ad] if cb else 0.0)
    diff = va - (vb if cb else 0.0)
    lo, hi = _boot.ga(f, nokta=diff, yontem=ci)
    return {"metric": METRIC_TO_ENGLISH.get(ad, ad), "a": _method(a), "b": _method(b), "value_a": va, "value_b": vb,
            "diff": diff, "lo": lo, "hi": hi, "p": _boot.p_degeri(f),
            "a_better": _a_better(M, va, vb) if cb else None, "n": int(v.kume.max()) + 1, "cluster": v.kume_adi,
            "B": int(B), "source": "frame"}


def _paired_from_report(report: dict, a: str, b: str | None, metric: str | None) -> dict:
    met = _metric(metric) or report.get("birincil")
    ca = ON_EK + _method(a)
    cb = None if b is None else ON_EK + _method(b)
    names = lambda x: {x, _method(x)} if x else {None}
    rows = list(report.get("kayitli_karsilastirmalar") or []) + list(report.get("karsilastirmalar") or [])
    n = (report.get("kart") or {}).get("n_kume")
    for r in rows:
        if r.get("metrik") != met:
            continue
        ra, rb = r.get("a"), r.get("b")
        sign = None
        if ra in names(ca) | {a} and (rb in (names(cb) | {b}) if cb else rb is None):
            sign = 1
        elif cb and ra in names(cb) | {b} and rb in names(ca) | {a}:
            sign = -1
        if sign is None:
            continue
        lo, hi = r.get("alt"), r.get("ust")
        if sign < 0:
            lo, hi = (None if hi is None else -hi), (None if lo is None else -lo)
        diff = None if r.get("fark") is None else sign * r["fark"]
        yon = next((m.get("yon", 1) for m in report.get("metrikler") or [] if m.get("ad") == met), 1)
        if met in _metrik.ISARETLI_METRIKLER:
            val = lambda c: (((report.get("sonuclar") or {}).get(c) or {}).get("metrikler") or {}).get(met, {}).get("deger")
            va_, vb_ = val(ca), (val(cb) if cb else None)
            better = None if va_ is None or vb_ is None else bool(abs(va_) < abs(vb_))
        else:
            better = None if (diff is None or b is None) else bool(yon * diff > 0)
        return {"metric": METRIC_TO_ENGLISH.get(met, met), "a": _method(a), "b": _method(b), "value_a": None,
                "value_b": None, "diff": diff, "lo": lo, "hi": hi, "p": r.get("p"),
                "a_better": better, "n": r.get("n", n),
                "cluster": report.get("tutma_birimi"), "B": None, "source": "report"}
    raise ValueError(f"pair ({a!r}, {b!r}) on {met!r} is not in the report: pass the frame, or evaluate(..., "
                     f"compare=[({a!r}, {b!r})])")


def baseline_ladder(df: pd.DataFrame, task: str | None = None, summary_stat: str = "mean", seed: int = 0) -> dict:
    """Cross-fitted label-free baselines {name: (label, values aligned with test rows)}:
    random, marginal (item mean / class shares), demography (item x segment, shrunk k=10) when a segment exists."""
    return _merdiven.baselineler(prepare(df, task), summary_stat, seed)


def shuffle_control(df: pd.DataFrame, method: str, seed: int = 1, task: str | None = None):
    """Predictions of `method` shuffled within item (persona-swap control)."""
    return _merdiven.karistir(prepare(df, task), ON_EK + _method(method), seed)


def recalibrate(df: pd.DataFrame, method: str, task: str | None = None) -> dict:
    """Cross-fitted recalibration: binary Platt + isotonic, multiclass temperature, continuous linear + isotonic,
    distribution temperature (JS)."""
    return _kal.yeniden(prepare(df, task), ON_EK + _method(method))


def reliability(df: pd.DataFrame, method: str, task: str | None = None, bins: int = 10) -> dict | None:
    """Reliability curve, ECE, confidence AUROC, AURC and coverage-accuracy (abstention) curve."""
    v = prepare(df, task)
    y, p = _metrik.diziler(v, ON_EK + _method(method))
    return _kal.egri(v, y, p, bins)


ece = _metrik.ece
tvd = _metrik.tvd
wasserstein_ordinal = _metrik.wasserstein_sirali
jensen_shannon = _metrik.js
ks_distance = _metrik.ks


# ------------------------------------------------------------------ audits
def split_audit(df: pd.DataFrame, task: str | None = None) -> list[dict]:
    """Group overlap train ∩ test, group-disjoint CV folds, temporal leakage, duplicated test units."""
    return _denetim.bolme(prepare(df, task))


def memorization_audit(items: list[dict], llm_or_caller, *, max_calls: int = _ezber.EN_FAZLA, B: int = 1000,
                       seed: int = 0, tolerance: float = 0.02, relative: bool = False) -> dict:
    """Completion-gap memorization test: hit rate on real items − hit rate on control (scrambled) items, item
    bootstrap 95 % CI; lower bound > 0 -> 'uyari' (memorization suspected).
    items: [{"prompt"|"istem", "expected"|"beklenen", "control"|"kontrol": bool}].
    llm_or_caller: a simengine LLM (e.g. FakeLLM) or a str -> str callable.
    tolerance / relative: numeric hit tolerance (absolute ±0.02 by default, for 0-1 shares); relative=True makes it
    a share of the expected value (0-100 or any scale). For numeric aggregates use `correlation_audit` instead."""
    og = [{"istem": o.get("istem", o.get("prompt")), "beklenen": o.get("beklenen", o.get("expected")),
           "kontrol": bool(o.get("kontrol", o.get("control", False)))} for o in items]
    cagir = _ezber.llm_caller(llm_or_caller) if hasattr(llm_or_caller, "complete") else llm_or_caller
    return _ezber.olc(og, cagir, en_fazla=max_calls, B=B, tohum=seed, tolerance=tolerance, relative=relative)


answer_hit = _ezber.isabet
correlation_audit = _ezber.correlation_audit
audit_prompts = _ezber.audit_prompts

# ------------------------------------------------------------------ pre-registration (kanit.onkayit)
canonical_bytes = _onkayit.kanonik_bayt


def _registry(registry, registry_dir):
    """`registry_dir=` is accepted as an alias of `registry=` (the name research.seal and older se.seal used)."""
    if registry_dir is None:
        return registry if registry is not None else ".onkayit"
    if registry is not None and Path(registry) != Path(registry_dir):
        raise TypeError("pass registry= or registry_dir= (alias), not both")
    return registry_dir


def seal(predictions, description: str, registry: str | os.PathLike | None = None, git_commit: bool = False, *,
         registry_dir: str | os.PathLike | None = None, store_copy: bool = True,
         archive_dir: str | os.PathLike | None = None, prereg_doc: str | os.PathLike | None = None) -> dict:
    """Seal a prediction table before the truth is revealed: SHA-256 of a canonical CSV + timestamp + hash chain.
    registry (default '.onkayit'; `registry_dir=` is an accepted alias) gets `zincir.jsonl` + a gzip copy.

    WARNING: the default registry '.onkayit' is created in the current working directory, usually inside your
    repository, and the gzip copy is the FULL prediction table. For licensed data keep the registry git-ignored
    or outside the repo, or pass store_copy=False.
    store_copy=False: write only the hash-chain record (no canonical copy; the record has "arsiv": None).
    verify / verify_chain work from the chain alone; grade then needs the predictions passed back in.
    archive_dir: write the canonical copy there instead of the registry (the record keeps only the file name
    and "arsiv_konum": "dis", no path); grade(None, truth, keys, archive_dir=...) can reopen it.
    A file path also records the raw file bytes' sha256 as "dosya_sha256" (file_sha256) next to the canonical
    "sha256"; the card's seal table shows both, labelled. Report the canonical one for verify/grade, the file one
    for `sha256sum` checks of the submitted file.
    prereg_doc: the hand-written / research.Prereg pre-registration document (e.g. "ONKAYIT.md"): its raw-byte
    sha256 goes into the record ("onkayit_belgesi": {"dosya", "sha256"}, file name only) and the description gets
    "[ön kayıt belgesi <name> sha256:<hash>]". evaluate(prereg=<registry>) then finds the document next to the
    registry and counts it as written no later than this seal while its bytes still match."""
    if not store_copy and archive_dir is not None:
        raise ValueError("store_copy=False and archive_dir= exclude each other")
    return _onkayit.muhurle(predictions, description, _registry(registry, registry_dir), git_commit,
                            kopya=store_copy, arsiv_dizini=archive_dir, belge=prereg_doc)


def verify(predictions, sha256: str | None = None, registry: str | os.PathLike | None = None, *,
           registry_dir: str | os.PathLike | None = None) -> dict:
    """Does the table still match a sealed hash? -> {"eslesti" (matched), "kayit" (record), "sha256"}."""
    return _onkayit.dogrula(predictions, sha256, _registry(registry, registry_dir))


def verify_chain(registry: str | os.PathLike | None = None, *, registry_dir: str | os.PathLike | None = None) -> dict:
    """Recompute every link of the seal chain -> {"gecerli" (valid), "n", "kirik" (broken indices)}."""
    return _onkayit.zincir_dogrula(_registry(registry, registry_dir))


def grade(predictions, truth, keys: list[str], registry: str | os.PathLike | None = None, *,
          registry_dir: str | os.PathLike | None = None, archive_dir: str | os.PathLike | None = None, **kw) -> dict:
    """Verify the seal, then join predictions with the truth on `keys` (raises ValueError if not sealed).
    predictions=None: reopen the sealed canonical copy (the last record, or `sha256=`) from the registry or
    `archive_dir`; its bytes are checked against the record. A seal made with store_copy=False has no copy, so
    pass the predictions (FileNotFoundError otherwise)."""
    if archive_dir is not None:
        kw["arsiv_dizini"] = archive_dir
    return _onkayit.notla(predictions, truth, keys, _registry(registry, registry_dir), **kw)


RESEARCH_CHAIN = "chain.jsonl"          # simengine.research.seal (default registry '.prereg')
_DEFAULT_REGISTRIES = (".onkayit", ".prereg")
_RESEARCH_TO_INTERNAL = {"seq": "sira", "time": "zaman", "time_utc": "zaman_utc", "rows": "satir", "cols": "sutun",
                         "columns": "sutunlar", "description": "aciklama", "source": "kaynak", "archive": "arsiv",
                         "prev": "onceki", "prev_record": "onceki_kayit", "record_hash": "kayit_hash"}


def chain_paths(registry: str | os.PathLike = ".onkayit") -> list[Path]:
    """Existing seal-chain files for a registry directory: `zincir.jsonl` (evidence.seal, also `se.seal`) and
    `chain.jsonl` (research.seal). When the directory holds neither and is one of the two default names
    ('.onkayit' / '.prereg'), the sibling default directory is tried, so a seal made with either API is found."""
    p = Path(registry)
    found = [f for f in (p / _onkayit.ZINCIR_ADI, p / RESEARCH_CHAIN) if f.exists()]
    if not found and p.name in _DEFAULT_REGISTRIES:
        for other in _DEFAULT_REGISTRIES:
            if other != p.name:
                q = p.parent / other
                found = [f for f in (q / _onkayit.ZINCIR_ADI, q / RESEARCH_CHAIN) if f.exists()]
    return found


def chain(registry: str | os.PathLike = ".onkayit") -> list[dict]:
    """Seal records of a registry (pass to evaluate(prereg=...)). Reads both chain formats (see `chain_paths`):
    evidence records are returned unchanged; research records keep their English keys and get the evidence names
    (`sira`, `zaman_utc`, `satir`, `aciklama`, ...) added, plus `"zincir": "chain.jsonl"`."""
    out = []
    files = chain_paths(registry)
    for f in files:
        if f.name == _onkayit.ZINCIR_ADI:
            out.extend(_onkayit._zincir_oku(f.parent))
        else:
            with open(f, encoding="utf-8") as fh:
                recs = [json.loads(s) for s in fh if s.strip()]
            for r in recs:
                r2 = dict(r)
                for en, tr in _RESEARCH_TO_INTERNAL.items():
                    if en in r and tr not in r2:
                        r2[tr] = r[en]
                r2["zincir"] = RESEARCH_CHAIN
                out.append(r2)
    if len(files) > 1:
        out.sort(key=lambda r: str(r.get("zaman_utc") or r.get("zaman") or ""))
    return out


# ------------------------------------------------------------------ card pieces
badge = _kart.rozet
card_dict = _kart.kart
panel_cost = _kart.dolar
