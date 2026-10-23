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
from .sema import ON_EK, TEST_ADLARI, _eksik, gercek_yok, gorev_bul, hazirla

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
INTERNAL_TO_COLUMN = {"satir": "row"}       # bootstrap cluster names in summary() / the card
for _en, _tr in COLUMN_TO_INTERNAL.items():
    INTERNAL_TO_COLUMN.setdefault(_tr, _en)
REVERSE_PREFIX = ("reverse_", "ters_")   # position-swap column for pair prompts (A/B order swapped)

METRIC_TO_INTERNAL = {
    "accuracy": "dogruluk", "balanced_accuracy": "dengeli_dogruluk", "kappa": "kappa", "auc": "auc",
    "average_precision": "ap", "pr_auc": "ap", "ap": "ap",
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
# se_col=: sampling-noise ceiling of an error metric from a per-row standard-error column. The oracle knows the latent
# value, so its error against the observed truth is N(0, se): pred = truth + sign * se * a, a = |z| averaged over R
# draws (mae, wmape: mean of R per-draw values) or sqrt(mean z^2) (rmse: R-averaged MSE).
SE_CEILING_METRICS = ("mae", "rmse", "wmape")
SE_CEILING_DRAWS = 400
SE_CEILING_LABEL = "sampling noise (SE column)"
MODEL_CEILING_LIMIT_HEAD = "Tavan bir model/oracle tahminidir"     # cekirdek's limit for a ceiling_col
SE_CEILING_LIMIT = ("Tavan örneklem gürültüsü tavanıdır (SE sütunu): gerçek değeri bilen bir oracle'ın gözlenen gerçeğe "
                    "hatası N(0, se); gürültülü gerçeğe bundan anlamlı yakın bir tahmin sızıntıya ya da SE sütununun "
                    "gürültüyü abarttığına işaret eder.")
SE_CEILING_NOTE = "tavan türü: örneklem gürültüsü (SE sütunu)"
DECLARED_PRIMARY_CHECK = "birincil metrik beyanı (ön kayıt)"
DECLARED_PRIMARY_LIMIT = "primary metric differs from the declared one"
METRIC_EQUIVALENT = {"bir_eksi_mae": "mae"}     # same ranking, different scale: not a different metric
# split= labels: constant test labels, or declarations that every row is an evaluation row of a cross-fitted /
# time-split protocol (out-of-fold predictions, CV folds, a rolling time split) -> audit text
SPLIT_DECLARED = {"oof": "çapraz uydurmalı, kat dışı tahmin (cross-fitted (declared): every prediction is out-of-fold)",
                  "cv": "çapraz doğrulama katları (cross-fitted (declared): every prediction comes from a CV fold "
                        "that did not train on it)",
                  "time": "zaman bölmeli (cross-fitted (declared): time split, every prediction made before its "
                          "truth window)"}
MEMORIZATION_CAVEAT_EN = "gain may come from recognition/memorization"
CAVEAT_REASON_EN = {"ezber_denetimi": "memorization audit warning", "kor_kol": "the blind arm loses most of the gain"}
ROW_TO_ENGLISH = {"taban": "Strongest simple baseline", "ham": "Raw method", "kalibre": "Ours",
                  "tavan": "Human retest ceiling", "bosluk": "Gap closed (ours - base)/(ceiling - base)",
                  "bedel": "Worth N humans (+100 real answers)", "persona": "Persona fields",
                  "dusen": "Dropped fields"}


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
UNIT_MEAN = "unit_mean"      # evaluate(aggregate=): per-row metric within each unit, then the mean of the units
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


_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)
_PRIMARY_LINE = re.compile(r"^\W*(?:primary(?:\s+metric)?|birincil(?:\s+metrik)?)\W*?\s*[:=]\s*(.+)$", re.I | re.M)
_METRICS_SECTION = re.compile(r"^#+\s*(?:metrics?|metrikler?)\s*$\n+(.+?)(?:\n\s*\n|\n#|\Z)", re.I | re.M | re.S)
_PRIMARY_WORD = re.compile(r"\b(?:primary|birincil)\b", re.I)
_METRIC_NAMES = sorted(set(METRIC_TO_INTERNAL) | set(METRIC_TO_INTERNAL.values()) | {"wape"}, key=len, reverse=True)
_METRIC_TOKEN = re.compile(r"(?<![\w-])(" + "|".join(re.escape(m) for m in _METRIC_NAMES) + r")(?![\w-])", re.I)


def _first_metric(text) -> str | None:
    m = _METRIC_TOKEN.search(str(text))
    return None if m is None else _metric(m.group(1).lower())


def _marked_primary(items) -> object:
    """The entry that says "primary" / "birincil", else the first one (Prereg: primary first)."""
    items = list(items)
    return next((v for v in items if _PRIMARY_WORD.search(str(v))), items[0] if items else None)


def declared_primary(text: str) -> str | None:
    """Primary metric declared in a pre-registration text -> internal metric name, or None.

    Order: the machine-readable copy of research.Prereg (the ```json block of the markdown, or the whole file when
    Prereg.write wrote .json): "metrics": {"primary": ..} / the list entry that says "primary" (else the first) /
    text; a "Primary metric: ..." / "Birincil metrik: ..." line; then the "## Metrics" section's line that says
    "primary" (else its first line: Prereg writes the primary first). The first known metric name in that text
    wins."""
    blocks = [m.group(1) for m in _JSON_BLOCK.finditer(text)]
    if text.lstrip().startswith("{"):
        blocks.insert(0, text)
    for b in blocks:
        try:
            doc = json.loads(b)
        except ValueError:
            continue
        if not isinstance(doc, dict) or (b is text and not ({"question", "title"} & set(doc))):
            continue                     # a bare .json is a prereg only with Prereg's fields (not a results file)
        mt = doc.get("metrics")
        if isinstance(mt, dict):
            mt = next((v for k, v in mt.items() if str(k).lower() in ("primary", "birincil")), None) or \
                next(iter(mt), None)
        elif isinstance(mt, (list, tuple)):
            mt = _marked_primary(mt)
        if mt and (x := _first_metric(mt)):
            return x
    for rx in (_PRIMARY_LINE, _METRICS_SECTION):
        m = rx.search(text)
        if m and (x := _first_metric(_marked_primary(m.group(1).strip().splitlines()))):
            return x
    return None


def _declared_primary_check(rep: dict, texts: list[dict]) -> None:
    """Scored primary vs the primary declared in the pre-registration documents (the latest declaring document,
    e.g. an addendum, wins): warning + audit row + card limit when they differ, an "ok" row when they match."""
    dec = [(e["zaman"], e["dosya"], x) for e in texts if (x := declared_primary(e.get("metin") or ""))]
    scored = rep.get("birincil")
    if not dec or not scored or not isinstance(rep.get("denetim"), list):
        return
    _, doc, want = max(dec, key=lambda t: t[0])
    same = METRIC_EQUIVALENT.get(want, want) == METRIC_EQUIVALENT.get(scored, scored)
    en = lambda x: METRIC_TO_ENGLISH.get(x, x)
    row = {"kontrol": DECLARED_PRIMARY_CHECK, "durum": "tamam" if same else "uyari", "deger": want,
           "ayrinti": (f"ön kayıtta beyan edilen birincil metrik {want} ({doc}); puanlanan {scored} "
                       f"(declared primary {en(want)} in {doc}, scored {en(scored)})")}
    rep["denetim"].append(row)
    if same:
        return
    msg = (f"{DECLARED_PRIMARY_LIMIT}: the card scores {en(scored)}, {doc} declares {en(want)}; "
           "pass primary= the declared metric or register the change in a dated addendum")
    warnings.warn(f"evaluate: {msg}", UserWarning, stacklevel=3)
    rep.setdefault("sinirlar", []).insert(0, msg)
    k = rep.get("kanit_kart")
    if isinstance(k, dict):
        k["yanilgi"] = [msg] + list(k.get("yanilgi") or [])


def prepare(df: pd.DataFrame, task: str | None = None, methods: Sequence[str] | None = None,
            classes: list | None = None):
    """Validate the table, detect the task, decode vector columns -> `sema.Veri` (internal names)."""
    return hazirla(to_internal(df), _task(task), _methods(methods), classes)


def _multiplier_task(d: pd.DataFrame, task: str | None) -> str | None:
    """Internal task for the multiplier default: the declared one, else auto-detected as degerlendir would."""
    if task is not None:
        return _task(task)
    cols = [c for c in d.columns if str(c).startswith(ON_EK)]
    if "gercek" not in d.columns or not cols:
        return None
    try:
        return gorev_bul(d, cols)
    except (TypeError, ValueError):
        return None


def _multiplier_not_applicable(rep: dict, note: str) -> None:
    """Default-skipped PPI++ multiplier: audit row, report key and the card's bedel row say 'not applicable'
    (same shape as degerlendir's segment / study level rule)."""
    rep["carpan_uygulanamaz"] = note
    rep.setdefault("denetim", []).append({"kontrol": "etkin örneklem çarpanı (PPI++ kaç insana bedel)",
                                          "durum": "atlandi", "deger": None, "uygulanamaz": True, "ayrinti": note})
    for r in (rep.get("kanit_kart") or {}).get("satirlar") or []:
        if r.get("tur") == "bedel":
            r.update(deger=None, alt=None, ust=None, birim="", **{"not": note})


# ------------------------------------------------------------------ evaluate
def evaluate(df: pd.DataFrame, title: str = "", ours: str | None = None, raw: str | None = None, *,
             dataset: str = "", task: str | None = None, methods: Sequence[str] | None = None,
             labels: dict | None = None, primary: str | None = None, B: int = 1000, B_aggregate: int = 200,
             seed: int = 0, calibrate: Sequence[str] | None = None, cutoff: str | None = None,
             memorization: str | dict | None = None, multiplier: bool | None = None,
             limits: Sequence[str] | None = None,
             prereg: list | str | os.PathLike | None = None, classes: list | None = None, baselines: bool = True,
             multiplier_methods: Sequence[str] | None = None, circular: Sequence[str] | None = None,
             level: str | None = None, source: str | None = None,
             unlabeled_baselines: Sequence[str] | None = None, baselines_oof: Sequence[str] | None = None,
             ceiling_col: str | None = None, ceiling_label: str | None = None, aggregate_by: str | None = None,
             multiplier_max_rows: int | None = MULTIPLIER_MAX_ROWS, offset_col: str | None = None,
             noise_ceiling: str | None = None, noise_ceiling_repeats: int = NOISE_CEILING_REPEATS,
             se_col: str | None = None, blind_col: str | None = None, compare: Sequence | None = None, ours_uses_llm: bool | None = None,
             aggregate: str = "mean", ci: str = "percentile", n_configs: int | None = None,
             block_col: str | None = None, point_pairs: Sequence | None = None,
             prereg_doc: str | os.PathLike | Sequence | None = None, split: str | None = None,
             balance_col: str | None = None, dominance_share: float = DOMINANT_BLOCK_SHARE,
             rate_error_by: str | Sequence[str] | None = None, cluster: str | None = None,
             min_effect: float | None = None) -> dict:
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
        Binary also "ap" (= "average_precision", "pr_auc"): step-wise average precision, tied scores one threshold
        (sklearn average_precision_score), CI from the B_aggregate loop.
        Continuous ratio metrics: "wmape" (= "wape", sum|y - p| / sum|y|, lower is better; the cluster bootstrap
        resamples numerator and denominator together, ratio of sums per replicate) and "bias" (sum p / sum y - 1,
        signed diagnostic, never primary). Passing primary="wmape" adds both to any continuous evaluation.
    B / B_aggregate: bootstrap replicates for per-unit / aggregate metrics. Cluster = group > item > row.
    cluster: bootstrap cluster column ("item", "group" or any column) overriding group > item > row; every value is one
        cluster (G3 #13: the same item repeated in 18 countries needs cluster="item"; the unit bootstrap gave
        [-0.117, -0.037], the item-clustered one [-0.188, +0.007]). The group column still sets the cross-fitting
        folds. With aggregate_by= it must be constant within each group; with block_col= it is the cluster side of
        the two-way bootstrap. NaN in the column raises ValueError. Named on the card ("clusters: N (cluster=`x`)"),
        `summary()["cluster"]`, `summary()["n_clusters"]`.
    min_effect: smallest ours − strongest baseline difference (metric units, metric direction) for the "high" badge,
        replacing kart.ANLAMLI_FARK = 0.02 (D12 #11: a z-MAE gain of 0.009 was called "small" against a fixed 0.02).
        Printed on the card meta line ("min_effect=x"), in the badge reason and `report["min_effect"]`. None: 0.02.
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
        The primary metric declared in the documents (`declared_primary`: research.Prereg "metrics", primary first,
        or a "Primary metric: ..." line; the latest declaring document wins) is checked against the scored one: when
        they differ, a UserWarning, an audit warning row and the card limit "primary metric differs from the
        declared one".
    split: a constant split label ("test" / "holdout" / "deger" / "degerlendirme") written into a split column
        when the frame has none (TEST-only frames, distribution frames whose cells are all held out), so the split
        audit runs; an audit row records that the split was declared, not derived. "oof" / "cv" / "time" declare
        out-of-fold, cross-validated or time-split predictions: every row is an evaluation row and the audit row
        reads "cross-fitted (declared)". Ignored with a warning when the frame already has a split column; other
        labels raise ValueError.
    balance_col: block column whose shares are checked (e.g. "country"; default: block_col when given). When one
        block holds more than max(dominance_share, 1.5 / n_blocks) of the scored rows (default share 0.20), an
        audit warning row and a card note give the primary metric of every method person-weighted (as reported)
        and block-equal-weighted: with aggregate_by, each group's value is the equal-weighted mean of its per-block
        means (not computed for aggregate="sum"); without it, row weights 1/n_block for per-row metrics and the
        mean of per-block values for whole-sample metrics (rank / correlation / AUC; within-block values leave the
        between-block signal out, so a gap there is not only weighting). `report["baskin_blok"]`,
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
        "unit_mean": rows are not averaged; each scored row gets weight 1 / n_unit (times the weight column) and the
        unit is the bootstrap group, so a per-row metric (accuracy, MAE, Brier) is the mean within each unit, then the
        equal-weighted mean of the units (D9 #3b, per-person accuracy); the task is kept (binary stays binary).
        Rank / correlation metrics are one weighted sample value. Every aggregate_by mode counts the units with no
        scored row (no test-split row, truth or predictions missing): `report["toplama"]` n_units, n_units_scored,
        n_dropped, n_rows_dropped, n_dropped_no_test_row (train-only units), n_dropped_missing (test rows without
        truth / predictions); when n_dropped > 0 a UserWarning, a limit line and the card meta say
        "k of N units dropped" (D9 #4d: 15 persons had no DEV rows and were dropped silently).
        A pooled primary (Pearson, Spearman, AUC, macro-F1, ECE: no row weights) is not a unit mean: unit_mean then
        warns, adds a limit line, sets `report["toplama"]["pooled_primary"]` and the card says "NOT a unit mean".
    ci: "percentile" (default) or "basic" (2*theta - bootstrap quantiles) for column, ladder, headline and paired
        CIs. When the point estimate falls outside the percentile CI (possible for ratio/aggregate metrics) a warning
        is raised and the card gets an audit row and a flag.
    n_configs: number of configurations tried before picking ours; written in the selection-bias audit (and >= 4
        with ours the best adds the optimism note). The audit says ours was "named in advance" only when the ours
        name occurs in the seal records or the text files of the prereg= registry; otherwise "not verified".
    multiplier: the PPI++ "worth N humans" effective-sample multiplier. None (default) computes it, except on a
        binary task without a human panel (no LLM arm: no raw= and ours_uses_llm=False, no blind_col, no item
        column or a single item, no retest column), e.g. an event log: there is no human answer to stand in for, so it
        is skipped and the card / `report["carpan_uygulanamaz"]` say "not applicable" (multiplier_methods= counts as
        a request: computed). True always computes it, False always skips it (no note).
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
    se_col: per-row standard error of the truth (e.g. a survey estimate's SE) for an error-metric primary (mae, rmse,
        wmape; primary= required, or task="forecast" for its default wmape): a seeded oracle column truth + N(0, se)
        (SE_CEILING_DRAWS draws averaged in the metric) becomes the model ceiling labelled "sampling noise (SE
        column)"; the analytic value (mae: sum(w sqrt(2/pi) se) / sum(w)) is in `report["se_tavani"]`,
        `summary()["se_ceiling"]` and on the card. No aggregate_by needed; with it the noise is drawn per
        aggregated cell with sd sqrt(sum se^2)/n (the cell mean's SE), the same offset on every row of the cell.
        Clearly beating this floor is a leakage sign (the card limit says so). NaN or negative SEs on scored rows
        raise ValueError (train rows may leave it empty); not with ceiling_col / noise_ceiling.
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
    rate_error_by: None (default: off), "segment" (the frame's `segment` / `seg_*` columns) or column name(s), binary
        and continuous tasks: rows of total rate error |sum p − sum y| / n (= |mean_bias|) and n-weighted segment rate
        error sum_s |sum_s p − sum_s y| / n (= sum_s (n_s/n) |mean_s p − mean_s y|) for every method and baseline
        column (agirlik/weight column: weighted sums), lower is better. CIs and ours − other differences (p, a_iyi) on
        the paired cluster bootstrap of the ladder (same clusters, blocks, seed, B, ci=). `report["oran_hatasi"]`,
        `summary()["rate_error"]`, a "Rate error" card table. A missing column raises ValueError, as do other tasks.
    Missing truth: rows with no truth are dropped (noted); when the truth column is absent or all missing the
    report is an "awaiting truth" card (seals + prediction summary; no ladder, calibration or bootstrap).
    """
    if aggregate not in AGGREGATES + (UNIT_MEAN,):
        raise ValueError(f"aggregate must be one of {AGGREGATES + (UNIT_MEAN,)}, not {aggregate!r}")
    if aggregate == UNIT_MEAN and aggregate_by is None:
        raise ValueError("aggregate='unit_mean' needs aggregate_by= (the unit column, e.g. the person)")
    if aggregate == UNIT_MEAN and noise_ceiling is not None:
        raise ValueError("noise_ceiling works on group means; use aggregate='mean' with it, not 'unit_mean'")
    if ci not in CI_METHODS:
        raise ValueError(f"ci must be one of {CI_METHODS}, not {ci!r}")
    if min_effect is not None:
        if isinstance(min_effect, bool) or not np.isfinite(float(min_effect)) or float(min_effect) < 0:
            raise ValueError(f"min_effect must be a finite number >= 0 (metric units), not {min_effect!r}")
        min_effect = float(min_effect)
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
    se_metric = None
    if se_col is not None:
        if ceiling_col is not None or noise_ceiling is not None:
            raise ValueError("pass se_col, ceiling_col or noise_ceiling, not several")
        se_metric = _metric(primary) or ("wmape" if forecast else None)
        if se_metric not in SE_CEILING_METRICS:
            raise ValueError(f"se_col gives a sampling-noise ceiling for error metrics {SE_CEILING_METRICS}; pass "
                             f"primary= one of them (primary is {primary!r}). Correlation metrics: "
                             "noise_ceiling='variance_components' with aggregate_by=")
    d = to_internal(df)
    split_declared = None
    if split is not None:
        sp = str(SPLIT_TO_INTERNAL.get(split, split))
        if sp.lower() not in TEST_ADLARI and sp.lower() not in SPLIT_DECLARED:
            raise ValueError(f"split= fills a constant test split or declares a cross-fitted one; use one of "
                             f"{TEST_ADLARI + list(SPLIT_DECLARED)}, not {split!r}")
        if "bolum" in d.columns:
            warnings.warn("evaluate: the frame already has a split column; split= is ignored", UserWarning, stacklevel=2)
        else:
            d = d.assign(bolum=sp if sp.lower() in TEST_ADLARI else "test")   # oof / cv / time: every row is scored
            split_declared = sp
    if doc_limits:
        limits = list(limits or []) + doc_limits
    blok = None
    if block_col is not None:
        blok = COLUMN_TO_INTERNAL.get(block_col, block_col)
        if blok not in d.columns:
            raise ValueError(f"block_col {block_col!r} is not a column of the frame")
    kume = None
    if cluster is not None:
        kume = COLUMN_TO_INTERNAL.get(cluster, cluster) if cluster not in d.columns else cluster
        if kume not in d.columns:
            raise ValueError(f"cluster {cluster!r} is not a column of the frame")
        if d[kume].isna().any():
            raise ValueError(f"cluster column {cluster!r} has {int(d[kume].isna().sum())} missing (NaN) rows")
    denge = blok
    if balance_col is not None:
        denge = COLUMN_TO_INTERNAL.get(balance_col, balance_col)
        if denge not in d.columns:
            raise ValueError(f"balance_col {balance_col!r} is not a column of the frame")
    methods_i = _methods(methods)
    labels_i = {_method(k): v for k, v in (labels or {}).items()}
    tavan_sutun = tavan_etiket = None
    se_info = None
    if se_metric is not None:
        d, se_info = _se_ceiling(d, se_col, se_metric, seed, by=None if aggregate == UNIT_MEAN else aggregate_by,
                                 how=aggregate)
        tavan_sutun, tavan_etiket = "tavan_model", SE_CEILING_LABEL
        if methods_i:
            methods_i = [m for m in methods_i if m != _method(se_col)]
    elif ceiling_col is not None:
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
    if aggregate_by is not None and aggregate == UNIT_MEAN:
        d, toplama = _unit_mean(d, aggregate_by, _task(task), methods_i)
        limits = list(limits or []) + [toplama["note"]]
    elif aggregate_by is not None:
        d, toplama = _aggregate(d, aggregate_by, _task(task),
                                halves=int(noise_ceiling_repeats) if noise_ceiling else None, seed=seed, how=aggregate,
                                keep={**({kume: "cluster"} if kume else {}), **({blok: "block_col"} if blok else {})})
        tavan_yarim = toplama.pop("halves", None)
        task = "continuous"
        limits = list(limits or []) + [toplama["note"]]
        level = level or "grup"
    if toplama is not None:
        dus = _dropped_units(d_rows, d, toplama["column"], methods_i)
        toplama.update(dus)
        if dus.get("n_dropped"):
            n, N, r = dus["n_dropped"], dus["n_units"], dus["n_rows_dropped"]
            why = f"{dus['n_dropped_no_test_row']} with no test-split row, {dus['n_dropped_missing']} with truth / " \
                  "predictions missing"
            warnings.warn(f"evaluate: aggregate_by={aggregate_by!r}: {n} of {N} units dropped (no scored row: {why}; "
                          f"{r} input rows); the metric covers the other {N - n} units", UserWarning, stacklevel=2)
            limits = list(limits or []) + [
                f"aggregate_by={aggregate_by}: {N} birimden {n} tanesinin değerlendirilen satırı yok, düşürüldü "
                f"({r} girdi satırı) ({n} of {N} units dropped by aggregate_by={aggregate_by}: no scored row, "
                f"{r} input rows)"]
    kiyas = _compare_list(compare)
    oran_seg = _rate_error_columns(d, rate_error_by, aggregated=toplama is not None)
    carpan_na = None
    if multiplier is None:
        # D13-W8 #7: binary task without a human panel (event log): PPI++ 'worth N humans' has nothing to stand in for;
        # multiplier_methods= is an explicit request for it
        multiplier = True
        na = None if multiplier_methods else _orneklem.panel_yok(d, _multiplier_task(d, task), ham_var=raw is not None or blind_col is not None,
                                 llm_yok=ours_uses_llm is False)
        if na:
            multiplier = False
            carpan_na = (f"uygulanamaz: {na[0]}; PPI++ 'kaç insana bedel' atlandı, istenirse multiplier=True "
                         f"(not applicable: {na[1]}; PPI++ 'worth N humans' skipped, pass multiplier=True to "
                         "compute it)")
    if multiplier and len(d) > (multiplier_max_rows or MULTIPLIER_MAX_ROWS) and not gercek_yok(d):
        tek = "madde" not in d or d["madde"].nunique() <= 1
        warnings.warn(
            f"evaluate: {len(d)} rows with multiplier=True. " + (
                f"The PPI++ 'worth N humans' multiplier is computed on a fixed-seed {multiplier_max_rows}-row "
                "subsample (noted on the card)." if tek and multiplier_max_rows else
                "The PPI++ multiplier runs on the whole table and can take minutes.")
            + " Pass multiplier=False to skip it.", UserWarning, stacklevel=2)
    if kume is not None:
        d = d.copy(deep=False)
        d.attrs = {**d.attrs, "kume_sutun": kume}     # sema.hazirla reads it: the bootstrap cluster column
    esik_token = _kart.EN_KUCUK_ETKI.set(min_effect)   # kart.kart reads it inside degerlendir
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
        _kart.EN_KUCUK_ETKI.reset(esik_token)
        if tmp:
            os.unlink(tmp)
    if carpan_na and rep.get("orneklem") is None and "carpan_uygulanamaz" not in rep:
        if rep.get("toplu_duzey"):   # segment / study / effect / choice rows: degerlendir's reason, as with True
            t = rep["toplu_duzey"]
            carpan_na = (f"uygulanamaz: {t['neden']}; PPI++ 'kaç insana bedel' atlandı (not applicable: {t['reason']}; "
                         "PPI++ 'worth N humans' skipped)")
        _multiplier_not_applicable(rep, carpan_na)
    if min_effect is not None:
        rep["min_effect"] = min_effect
    if toplama:
        rep["toplama"] = {k: v for k, v in toplama.items() if k != "note"}
        if aggregate == UNIT_MEAN:
            _unit_mean_pooled_primary(rep, aggregate_by, primary)
    if kume is not None:
        rep["kume_sutunu"] = kume
    if prereg_texts:
        rep["onkayit_belgeleri"] = [{"dosya": e["dosya"], "sha256": e["sha256"], "zaman_kaynagi": e["zaman_kaynagi"],
                                     "zaman_utc": datetime.fromtimestamp(e["zaman"]).astimezone().isoformat(timespec="seconds"),
                                     **({"kilit_uyusmuyor": True} if e.get("kilit_uyusmuyor") else {})}
                                    for e in prereg_texts]
    if split_declared and isinstance(rep.get("denetim"), list):
        cross = SPLIT_DECLARED.get(split_declared.lower())
        rep["denetim"].append({"kontrol": "bölme beyanı (split=)", "durum": "tamam", "deger": split_declared,
                               "ayrinti": f"split={split!r} verildi: " + (
                                   f"bütün satırlar değerlendirme satırı, {cross}" if cross else
                                   f"bütün satırlar '{split_declared}' (değerlendirme) sayıldı; eğitim satırı yok, "
                                   "bölme beyanı kullanıcınındır (declared split: every row is held out)")})
    if se_info is not None:
        c = (rep.get("manset") or {}).get("tavan") or {}
        se_info.update(deger=c.get("deger"), alt=c.get("alt"), ust=c.get("ust"))
        rep["se_tavani"] = se_info
        _se_ceiling_wording(rep)
    if prereg_texts:
        _declared_primary_check(rep, prereg_texts)
    if denge is not None and rep.get("sonuclar"):
        _dominant_block(rep, d_rows, denge, explicit=balance_col is not None,
                        by=None if aggregate == UNIT_MEAN else aggregate_by, how=aggregate,
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
    if oran_seg and rep.get("sonuclar") and not rep.get("gercek_bekleniyor"):
        rep["oran_hatasi"] = _rate_error(rep, d, oran_seg, methods=methods_i, classes=classes, block=blok, B=B,
                                         seed=seed, ci=ci)
    return rep


RATE_ERROR_TASKS = ("ikili", "surekli")


def _rate_error_columns(d: pd.DataFrame, by, aggregated: bool = False) -> list[str] | None:
    """rate_error_by= -> internal segment column names, checked against the evaluated frame before scoring."""
    if by is None:
        return None
    names = [by] if isinstance(by, str) else list(by)
    out = []
    for x in names:
        if x == "segment" and "segment" not in d.columns:
            segs = [c for c in d.columns if isinstance(c, str) and c.startswith("seg_")]
            if not segs:
                raise ValueError("rate_error_by='segment' needs a 'segment' or 'seg_*' column in the evaluated frame "
                                 "(or pass a column name)" + (" (aggregate_by= keeps only the group, item, split "
                                                              "and block columns)" if aggregated else ""))
            out += segs
            continue
        c = COLUMN_TO_INTERNAL.get(x, x)
        if c not in d.columns:
            raise ValueError(f"rate_error_by: {x!r} is not a column of the evaluated frame"
                             + (" (aggregate_by= keeps only the group, item, split and block columns)"
                                if aggregated else ""))
        out.append(c)
    return list(dict.fromkeys(out))


def _rate_error(rep: dict, d: pd.DataFrame, segs: list[str], *, methods, classes, block, B: int, seed: int,
                ci: str) -> dict:
    """Total and n-weighted segment rate error of every method / baseline column on the paired cluster bootstrap
    of the ladder (same Orneklem: clusters, two-way blocks, seed). Lower is better; differences are ours − other."""
    from scipy import sparse
    from .cekirdek import _blok_kur
    if rep.get("gorev") not in RATE_ERROR_TASKS:
        raise ValueError(f"rate_error_by needs a binary or continuous task (sums of predictions are rates or totals), "
                         f"not {TASK_TO_ENGLISH.get(rep.get('gorev'), rep.get('gorev'))!r}")
    veri = hazirla(d, rep["gorev"], methods, classes)
    _blok_kur(veri, block, False)
    dd = veri.df
    y = dd["gercek"].to_numpy(float)
    w = dd["agirlik"].to_numpy(float) if "agirlik" in dd else np.ones(len(dd))
    sonuc = rep.get("sonuclar") or {}
    taban = {}
    if any(str(c).startswith("taban_") for c in sonuc):    # same cross-fitted rungs (same function, summary, seed)
        ozet_s = "median" if rep.get("birincil") in ("mae", "bir_eksi_mae", "wmape") else "mean"
        taban = {f"taban_{ad}": deg for ad, (_, deg) in _merdiven.baselineler(veri, ozet_s, seed).items()}
    kol, atlanan = {}, []
    for c, v in sonuc.items():
        if v.get("tur") not in ("baseline", "yontem"):
            continue
        p = taban.get(c, dd[c] if c in dd else None)
        try:
            p = np.asarray(p, float)
        except (TypeError, ValueError):
            p = None
        if p is None or p.shape != y.shape or not np.isfinite(p).all():
            atlanan.append(c)
            continue
        kol[c] = p
    cols = list(kol)
    if not cols:
        return {"segmentler": segs, "n": int(len(dd)), "B": B, "ga_yontemi": ci, "bizim": None, "satirlar": [],
                "farklar": [], "atlanan": atlanan}
    orn = _boot.Orneklem(veri.kume, B=B, tohum=seed, iki_yon=veri.iki_yon)
    G, C, k = orn.G, len(cols), orn.kume
    E = np.column_stack([w * (kol[c] - y) for c in cols])                    # rows × C
    Wn = np.bincount(k, w, minlength=G)
    T = np.column_stack([np.bincount(k, E[:, j], minlength=G) for j in range(C)])   # G × C
    S = {}
    for s in segs:
        kod, _ = pd.factorize(dd[s], use_na_sentinel=False)
        K = int(kod.max()) + 1
        satir = np.repeat(k, C)
        sutun = (np.arange(C)[None, :] * K + kod[:, None]).reshape(-1)
        S[s] = (sparse.csr_matrix((E.reshape(-1), (satir, sutun)), shape=(G, C * K)).T.tocsr(), K)   # (C·K) × G

    def olc(c):
        n = c @ Wn
        out = {"toplam": np.abs(c @ T) / n}
        for s, (M, K) in S.items():
            out[s] = np.abs(M @ c).reshape(C, K).sum(1) / n
        return out

    nokta = olc(np.ones(G))
    R = {o: np.empty((B, C)) for o in nokta}
    for b in range(B):
        for o, v in olc(orn.sayim(b)).items():
            R[o][b] = v

    def kutu(o, j):
        a, u = (_finite(x) for x in _boot.ga(R[o][:, j], nokta=nokta[o][j], yontem=ci))
        v = float(nokta[o][j])
        return {"deger": v, "alt": a, "ust": u,
                **({"nokta_ga_disi": True} if a is not None and u is not None and not a <= v <= u else {})}

    satirlar = [{"sutun": c, "etiket": sonuc[c].get("etiket"), "tur": sonuc[c].get("tur"), "toplam": kutu("toplam", j),
                 "segment": {s: {**kutu(s, j), "n_segment": S[s][1]} for s in segs}}
                for j, c in enumerate(cols)]
    bc = ((rep.get("manset") or {}).get("bizim") or {}).get("sutun")
    farklar = []
    if bc in cols:
        i = cols.index(bc)
        for j, c in enumerate(cols):
            if j == i:
                continue
            for o in nokta:
                f = R[o][:, i] - R[o][:, j]
                fn = float(nokta[o][i] - nokta[o][j])
                a, u = (_finite(x) for x in _boot.ga(f, nokta=fn, yontem=ci))
                farklar.append({"a": bc, "b": c, "etiket": sonuc[c].get("etiket"), "olcu": o, "fark": fn,
                                "alt": a, "ust": u, "p": _finite(_boot.p_degeri(f)),
                                "a_iyi": True if u is not None and u < 0 else False if a is not None and a > 0 else None})
    out = {"segmentler": segs, "n": int(len(dd)), "B": B, "ga_yontemi": ci, "bizim": bc, "satirlar": satirlar,
           "farklar": farklar,
           "tanim": "toplam = |Σtahmin − Σgerçek| / n; segment = Σ_s |Σ_s tahmin − Σ_s gerçek| / n (n ağırlıklı); küçük iyi. "
                    "Mutlak değer her tekrarda ≥ 0: gerçek fark 0'a yakınken yüzdelik GA noktanın üstünde kalabilir; "
                    "kollar arası karar eşleşik farktan (farklar) okunur"}
    if atlanan:
        out["atlanan"] = atlanan
    return out


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


def _se_ceiling(d: pd.DataFrame, se_col: str, metric: str, seed: int, by: str | None = None, how: str = "mean",
                draws: int = SE_CEILING_DRAWS) -> tuple[pd.DataFrame, dict]:
    """se_col -> (frame with the oracle column "tavan_model", info). The oracle's error against the observed truth is
    N(0, se_i); its column is truth + sign * se * a with a = |z| averaged over `draws` seeded draws (mae / wmape) or
    sqrt(mean z^2) (rmse), so the metric of the column is the draw-averaged metric. With `by` (aggregate_by) the
    noise is drawn per aggregated cell (the `_aggregate` keys): the cell mean of the truth carries sd sqrt(sum se^2)/n
    (the cell total sqrt(sum se^2)), so every row of the cell gets the same offset sign * a * sqrt(sum se^2) / n;
    per-row offsets with random signs would average to a (2/pi)-shrunk mae ceiling. Analytic value on the scored
    rows / cells (test rows when a split column exists; the row weight column honoured without `by`): mae
    sum(w sqrt(2/pi) sd) / sum(w), rmse sqrt(sum(w sd^2) / sum(w)), wmape sum(w sqrt(2/pi) sd) / sum(w |y|).
    The SE must be present and >= 0 on the scored rows; rows outside the test split may leave it empty."""
    col = COLUMN_TO_INTERNAL.get(se_col, se_col)
    if col not in d.columns:
        raise ValueError(f"se_col {se_col!r} is not a column of the frame")
    if "gercek" not in d.columns:
        raise ValueError("se_col needs a truth column")
    y = pd.to_numeric(d["gercek"], errors="coerce").to_numpy(float)
    se = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
    test = (d["bolum"].astype(str).str.lower().isin(TEST_ADLARI).to_numpy() if "bolum" in d.columns
            else np.ones(len(d), bool))
    scored = np.isfinite(y) & test
    if not np.isfinite(se[scored]).all():
        raise ValueError(f"se_col {se_col!r} has missing or non-numeric values on scored rows with a truth")
    if (se[np.isfinite(se)] < 0).any():
        raise ValueError(f"se_col {se_col!r} has negative standard errors")
    if by is None:
        codes, G = np.arange(len(d)), len(d)
    else:
        bc = COLUMN_TO_INTERNAL.get(by, by)
        if bc not in d.columns:
            raise ValueError(f"aggregate_by {by!r} is not a column of the frame")
        keys = [bc] + [k for k in (("bolum",) if how == "sum" else ("madde", "bolum")) if k in d.columns and k != bc]
        codes = d.groupby(keys, sort=True, dropna=False).ngroup().to_numpy()
        G = int(codes.max()) + 1 if len(codes) else 0
    ok = np.isfinite(y) & np.isfinite(se)
    n_c = np.bincount(codes[ok], minlength=G).astype(float)
    ss = np.bincount(codes[ok], weights=se[ok] ** 2, minlength=G)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(ss) / n_c               # sd of the cell mean (the cell total's sd / n): per-row offset
    rng = np.random.default_rng([abs(int(seed or 0)), 5113])
    acc = np.zeros(G)
    for _ in range(int(draws)):          # one vector per draw: memory stays O(rows)
        z = rng.standard_normal(G)
        acc += z * z if metric == "rmse" else np.abs(z)
    a = np.sqrt(acc / draws) if metric == "rmse" else acc / draws
    sign = np.where(rng.random(G) < 0.5, -1.0, 1.0)
    off = (sign * sd * a)[codes]
    out = d.drop(columns=[col]).assign(tavan_model=np.where(ok, y + off, np.nan))
    k = math.sqrt(2 / math.pi)
    if by is None:
        w = (pd.to_numeric(d["agirlik"], errors="coerce").to_numpy(float) if "agirlik" in d.columns
             else np.ones(len(d)))[scored]
        s, yy = se[scored], y[scored]
    else:                                # cells of scored rows, equal cell weight (as `_aggregate`)
        cs = np.unique(codes[scored])
        s = sd[cs]
        ysum = np.bincount(codes[ok], weights=y[ok], minlength=G)[cs]
        yy = ysum if how == "sum" else ysum / n_c[cs]
        if how == "sum":
            s = s * n_c[cs]              # the cell total's sd
        w = np.ones(len(cs))
    analytic = None
    if len(s) and w.sum() > 0:
        den = (w * np.abs(yy)).sum() if metric == "wmape" else w.sum()
        analytic = (float(np.sqrt((w * s * s).sum() / w.sum())) if metric == "rmse"
                    else float((w * k * s).sum() / den) if den > 0 else None)
    info = {"sutun": se_col, "metrik": metric, "R": int(draws), "tohum": seed, "analitik": analytic,
            "n": int(scored.sum()), "etiket": SE_CEILING_LABEL, **({"n_hucre": int(len(s))} if by is not None else {})}
    return out, info


def _se_ceiling_wording(rep: dict) -> None:
    """The core labels any ceiling_col as a model/oracle reference ("exceeding it does not mean leakage"); an SE
    ceiling is a sampling-noise floor, where clearly beating it is a leakage sign: swap the limit and the card note."""
    def swap(xs):
        return [SE_CEILING_LIMIT if str(x).startswith(MODEL_CEILING_LIMIT_HEAD) else x for x in xs or []]
    rep["sinirlar"] = swap(rep.get("sinirlar"))
    k = rep.get("kanit_kart")
    if isinstance(k, dict):
        k["yanilgi"] = swap(k.get("yanilgi"))
        for r in k.get("satirlar") or []:
            if r.get("tur") == "tavan" and r.get("deger") is not None:
                r["not"] = SE_CEILING_NOTE


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


def _scored_rows(d: pd.DataFrame, methods: Sequence[str] | None) -> np.ndarray:
    """Rows sema.hazirla scores: test split (when a split column exists), truth and every prediction present."""
    m = np.ones(len(d), bool)
    if "bolum" in d.columns:
        m &= d["bolum"].astype(str).str.lower().isin(TEST_ADLARI).to_numpy()
    m &= ~_eksik(d["gercek"]) if "gercek" in d.columns else False
    cols = ([ON_EK + x for x in methods] if methods else
            [c for c in d.columns if isinstance(c, str) and c.startswith(ON_EK)])
    for c in cols:
        if c in d.columns:
            m &= ~_eksik(d[c])
    return m


def _dropped_units(rows: pd.DataFrame, scored: pd.DataFrame, col: str, methods) -> dict:
    """Units of the input (`rows`) with no scored row in the evaluated frame (aggregate_by drops them silently: D9
    #4d, 15 persons without DEV rows). Empty when there is no truth at all (awaiting-truth card)."""
    if col not in rows.columns or col not in scored.columns or "gercek" not in rows.columns \
            or bool(_eksik(rows["gercek"]).all()):
        return {}
    keep = set(scored.loc[_scored_rows(scored, methods), col].tolist())
    units = rows[col]
    n_units = int(units.nunique(dropna=False))
    lost = ~units.isin(keep).to_numpy()
    n_drop = int(units[lost].nunique(dropna=False))
    # split of the dropped units: no test-split row at all (train-only, e.g. D9 persons without DEV rows, or the
    # training weeks of a forecast frame) vs test rows whose truth / predictions are missing
    n_no_test = 0
    if "bolum" in rows.columns and n_drop:
        test = set(units[rows["bolum"].astype(str).str.lower().isin(TEST_ADLARI).to_numpy()].tolist())
        n_no_test = int(units[lost & ~units.isin(test).to_numpy()].nunique(dropna=False))
    return {"n_units": n_units, "n_units_scored": n_units - n_drop, "n_dropped": n_drop,
            "n_rows_dropped": int(lost.sum()), "n_dropped_no_test_row": n_no_test,
            "n_dropped_missing": n_drop - n_no_test}


def _unit_mean(d: pd.DataFrame, by: str, task: str | None, methods) -> tuple[pd.DataFrame, dict]:
    """aggregate="unit_mean": rows stay rows; scored rows get weight w / (sum of w in their unit) (the weight column
    or 1) and the unit becomes the bootstrap group, so a per-row metric is the mean within each unit, then the
    equal-weighted mean of the units (D9 #3b: per-person accuracy)."""
    col = COLUMN_TO_INTERNAL.get(by, by) if by not in d.columns else by
    if col not in d.columns:
        raise ValueError(f"aggregate_by {by!r} is not a column of the frame")
    if task == "ab":
        raise ValueError("aggregate='unit_mean' does not apply to the ab task")
    if d[col].isna().any():
        raise ValueError(f"aggregate_by {by!r} has {int(d[col].isna().sum())} missing (NaN) rows")
    d = d.copy()
    w = pd.to_numeric(d["agirlik"], errors="coerce").to_numpy(float) if "agirlik" in d.columns else np.ones(len(d))
    ok = _scored_rows(d, methods)
    tot = pd.Series(np.where(ok, w, 0.0)).groupby(d[col].to_numpy()).transform("sum").to_numpy()
    d["agirlik"] = np.where(ok & (tot > 0), w / np.where(tot > 0, tot, 1.0), w)
    if col != "grup":
        d["grup"] = d[col].astype(str)
    n_units = int(d.loc[ok, col].nunique())
    note = (f"Birim ortalaması (aggregate_by={by}, aggregate='unit_mean'): {int(ok.sum())} değerlendirme satırında "
            f"satır metriği önce her birimin içinde, sonra {n_units} birim eşit ağırlıkla ortalandı (satır ağırlığı "
            "1/n_birim); bootstrap birimler üzerinden. Toplu metrikler (Pearson, Spearman, AUC, macro-F1, ECE) ağırlık "
            "almaz: birim ortalaması değil, bütün satırların tek değeridir "
            f"(unit mean, aggregate_by={by}: a per-row metric averaged within each of {n_units} units over "
            f"{int(ok.sum())} scored rows, then across units with equal weight, row weight 1/n_unit; bootstrap over "
            "units; pooled metrics "
            "such as Pearson, Spearman, AUC, macro-F1 and ECE take no weights: one value over all rows, not a mean of "
            "units)")
    info = {"by": by, "column": col, "n_rows": int(len(d)), "n_groups": n_units, "n_eval_rows": int(ok.sum()),
            "aggregate": UNIT_MEAN, "note": note}
    return d, info


def _unit_mean_pooled_primary(rep: dict, by: str, primary: str | None) -> None:
    """aggregate="unit_mean" weights rows 1/n_unit, but pooled ("toplu") metrics take no weights: a pooled primary
    (continuous default Pearson; binary macro-F1 picked for class imbalance) left the headline equal to the plain
    row-level value with no sign on the card. Flags it: toplama["pooled_primary"], a UserWarning and a limit line."""
    met = rep.get("birincil")
    gorev = rep.get("gorev")
    if not met or not gorev or not rep.get("sonuclar"):
        return
    try:
        fam = _metrik.aile(gorev, True, True, True, True, True)
    except ValueError:
        return
    M = next((m for m in fam if m.ad == met), None)
    if M is None or M.tur != "toplu":
        return
    en = METRIC_TO_ENGLISH.get(met, met)
    picked = "picked by default" if primary is None else "passed as primary"
    msg = (f"aggregate='unit_mean': birincil metrik {en} toplu bir metrik, ağırlık almaz; manşet birim ortalaması "
           f"değil, satır düzeyindeki değerdir (aggregate='unit_mean' with aggregate_by={by} does not apply to the "
           f"primary metric {en}, {picked}: pooled metrics take no row weights, so the headline is the row-level "
           "value; pass a per-row primary such as accuracy, brier, mae or rmse for a mean of units)")
    warnings.warn(f"evaluate: {msg}", UserWarning, stacklevel=3)
    rep["toplama"]["pooled_primary"] = en
    rep["sinirlar"] = list(rep.get("sinirlar") or []) + [msg]
    if isinstance(rep.get("kanit_kart"), dict):
        rep["kanit_kart"]["yanilgi"] = list(rep["kanit_kart"].get("yanilgi") or []) + [msg]


def _aggregate(d: pd.DataFrame, by: str, task: str | None, halves: int | None = None, seed: int = 0,
               how: str = "mean", keep: Sequence[str] | dict = ()):
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
            what = keep.get(c, "block_col") if isinstance(keep, dict) else "block_col"
            raise ValueError(f"{what} {c!r} varies within aggregate_by groups; aggregate by a finer key or "
                             f"use a {'block' if what == 'block_col' else what} column that is constant within "
                             "each group")
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
               "agirlik_1_n_blok": "satır ağırlığı 1/n_blok", "blok_ortalamasi": "blok içi değerlerin ortalaması; bloklar arası fark dışarıda, yeniden ağırlıklama değil"}
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


def _ours_seal(report: dict) -> dict | None:
    """report["bizim_muhur"] in English: the seal of ours (index, sha256, match class; "note" when ours matched no
    seal and the chain's last record stands in)."""
    b = report.get("bizim_muhur")
    if not b:
        return None
    out = {"index": b.get("sira"), "sha256": b.get("sha256"), "match": b.get("eslesme"), "n_records": b.get("n_kayit")}
    if b.get("not"):
        m = re.search(r"\((ours not matched[^()]*(?:\([^()]*\)[^()]*)*)\)\s*$", str(b["not"]))
        out["note"] = m.group(1) if m else _en_text(b["not"])
    return out


def _f3(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.3f}"


# ------------------------------------------------------------------ English rendering (headline_en, card(lang="en"))
# The research layer writes Turkish labels, flags, audit names and details. These tables translate the common ones
# for the English views; anything not covered stays in its Turkish wording (never an error).
KIND_TO_ENGLISH = {"baseline": "baseline", "yontem": "method", "kontrol": "control", "kalibre": "calibrated",
                   "tavan": "ceiling"}
AUTO_BASELINE_TO_ENGLISH = {"rastgele": "random", "marjinal": "marginal", "demografi": "demographic"}
RECALIBRATION_TO_ENGLISH = {"izotonik": "isotonic", "sicaklik": "temperature", "dogrusal": "linear", "platt": "platt"}
GAP_NA_NOTE = "n/a, ceiling ≈ baseline"
CHECK_TO_ENGLISH = {
    "grup-ayrık katlar (CV)": "group-disjoint folds (CV)",
    "bölme bilgisi (eğitim/test)": "split information (train/test)",
    "grup çakışması (eğitim ∩ test)": "group overlap (train ∩ test)",
    "zaman sızıntısı (eğitim testten sonra mı)": "temporal leakage (train after test?)",
    "birim tekrarı (testte)": "duplicated units (test)",
    "tavanı aşan yöntem (sızıntı işareti)": "method above the ceiling (leakage sign)",
    "model/oracle tavanını aşan yöntem": "method above the model/oracle ceiling",
    "örneklem gürültüsü tavanını aşan yöntem (sızıntı işareti)": "method above the sampling-noise ceiling (leakage sign)",
    "dağılım çöküşü (Bernoulli sd oranı)": "distribution collapse (Bernoulli sd ratio)",
    "dağılım çöküşü (entropi oranı)": "distribution collapse (entropy ratio)",
    "dağılım çöküşü (sd oranı)": "distribution collapse (sd ratio)",
    "sınıf dengesizliği": "class imbalance",
    "kontaminasyon (kesim öncesi/sonrası)": "contamination (before/after cutoff)",
    "pozisyon takası": "position swap",
    "ezber testi (LLM veriyi biliyor mu)": "memorization test (does the LLM know the data?)",
    "en güçlü taban belirsiz mi (tabanlar arası seçim gürültüsü)": "strongest baseline uncertain? (selection noise among baselines)",
    "madde ortalamasını geçiyor mu": "beats the item mean?",
    "en güçlü basit tabanı geçiyor mu": "beats the strongest simple baseline?",
    "en güçlü diğer yöntem": "strongest other method",
    "tahmin karıştırma (persona takası değil)": "prediction shuffle (not a persona swap)",
    "persona karıştırma (birey bilgisi kullanılıyor mu)": "persona shuffle (is individual information used?)",
    "kişiye özgü sinyal (madde_ici_kor)": "person-specific signal (within_item_corr)",
    "kişiye özgü sinyal (demeaned_r2 > 0)": "person-specific signal (demeaned_r2 > 0)",
    "seri içi sinyal (demeaned_r2 > 0)": "within-series signal (demeaned_r2 > 0)",
    "seçim yanlılığı": "selection bias",
    "kör kol (tanıma kontrolü)": "blind arm (recognition check)",
    "yalnız basit tabanlar": "only trivial baselines",
    "tek tahmin kökeni (single origin)": "single forecast origin",
    "mühür": "seal", "kontrol": "control", "gerçek": "truth",
    "kayıtlı karşılaştırma zayıf tabana karşı": "registered comparison against a weak baseline",
    "görev türü: tahmin": "task type: forecast",
    "girdi notu": "input note",
    "etkin örneklem çarpanı": "effective sample multiplier",
    "döngüsel önsel (çarpan dışı)": "circular prior (excluded from the multiplier)",
    "bölme beyanı (split=)": "declared split (split=)",
    "birincil metrik beyanı (ön kayıt)": "declared primary metric (pre-registration)",
    "bootstrap GA (nokta tahmini yüzdelik GA dışında)": "bootstrap CI (point estimate outside the percentile CI)",
    "blok bootstrap": "block bootstrap",
    "blok bootstrap blok sayısı": "block bootstrap: number of blocks",
    "bilgi düzeyi (ayar A / ayar B)": "information level (setting A / setting B)",
    "baskın blok (dominant block)": "dominant block",
    "küme sayısı (bootstrap GA)": "number of clusters (bootstrap CI)",
    # audit rows added in f948774 / 9aa3d47 and the constant-prediction rows (W8: printed in Turkish with lang="en")
    "sabit tahmin (sd 0)": "constant prediction (sd 0)",
    "sabit tahmin: sıra metrikleri tanımsız (constant prediction)": "constant prediction: rank metrics undefined",
    "etkin örneklem çarpanı (PPI++ kaç insana bedel)": "effective sample multiplier (PPI++ worth N humans)",
    "ezber uyarısı ve bizim kol (memorization warning and ours)": "memorization warning and ours",
    "çerçeve ↔ mühürlü dosya": "frame ↔ sealed file",
    "zaman sütunu çözümü (time column parsing)": "time column parsing",
}
CHECK_PATTERNS_EN = ((r"^pozisyon takası \((.+)\)$", r"position swap (\1)"),
                     (r"^dolgu payı \(geçersiz → tekdüze\): (.+)$", r"fill share (invalid → uniform): \1"))
# card row labels ("Method" column): the row name is already in the "Row" column
ROW_LABEL_TO_ENGLISH = {
    "En güçlü basit taban": "—", "Ham yöntem": "—", "Bizim yöntem": "—", "İnsan tekrar tavanı": "human test-retest",
    "Model/oracle tavanı": "model/oracle", "Örneklem gürültüsü tavanı (varyans bileşenleri)": "variance components",
    "Kapanan boşluk (bizim − taban)/(tavan − taban)": "(ours − baseline)/(ceiling − baseline)",
    "Kaç insana bedel": "PPI++ multiplier",
}
ROW_LABEL_PREFIXES_EN = ((r"^En güçlü basit taban: ", ""), (r"^Ham: ", ""), (r"^Bizim: ", ""),
                         (r"^Kaç insana bedel: (.*) \+ 100 gerçek yanıt$", r"\1 + 100 real answers"))

# Ordered (pattern, replacement): whole sentences first, then fragments. Applied after dot decimals.
_PHRASES_EN = [
    # badge reasons (kart.rozet)
    (r"Gerçek sonuç henüz yok ve tahmin mühürlenmemiş: gerçek açılmadan önce seal\(\) ile mühürleyin\.",
     "No truth yet and the predictions are not sealed: seal them with seal() before the truth is opened."),
    (r"Gerçek sonuç henüz yok: tahmin mühürlü, kart gerçek gelince dolar\.",
     "No truth yet: the predictions are sealed; the card fills in when the truth arrives."),
    (r"Bizim yöntem en güçlü basit tabanın altında: fark (\S+), GA \[(\S+); (\S+)\]\.",
     r"Ours is below the strongest simple baseline: difference \1, CI [\2, \3]."),
    (r"Tabana göre fark (\S+) anlamlı, ama ", r"The difference to the baseline (\1) is significant, but "),
    (r"bizim yöntemin bedel alt sınırı (\S+)× < (\S+)×", r"the lower bound of ours' worth-N-humans multiplier is \1× < \2×"),
    (r"bizim yöntemin bedel alt sınırı (\S+)× tabanınkini \((\S+)×\) geçmiyor",
     r"the lower bound of ours' worth-N-humans multiplier (\1×) does not exceed the baseline's (\2×)"),
    (r"Tabana göre fark (\S+), GA \[(\S+); (\S+)\] sıfırın üstünde\.",
     r"Difference to the baseline \1, CI [\2, \3] above zero."),
    (r"Fark anlamlı ama küçük \((\S+), GA \[(\S+); (\S+)\]\); en güçlü basit taban neredeyse aynı yerde"
     r"(?: \(en küçük etki eşiği (\S+)\))?\.",
     lambda m: f"Significant but small difference ({m.group(1)}, CI [{m.group(2)}, {m.group(3)}]); the strongest "
               "simple baseline is almost level" + (f" (minimum effect {m.group(4)})" if m.group(4) else "") + "."),
    (r" En küçük etki eşiği \(min_effect\) (\S+)\.", r" Minimum effect (min_effect) \1."),
    (r"Tabana göre fark (\S+), GA \[(\S+); (\S+)\] sıfırı içeriyor\.",
     r"Difference to the baseline \1, CI [\2, \3] includes zero."),
    (r"En güçlü basit tabana göre eşleşik farkın güven aralığı ölçülmedi\.",
     "No paired CI against the strongest simple baseline was measured."),
    (r"Tabana göre fark yok; ", "No difference to a baseline; "),
    (r"Bizim yöntemin kaç insana bedel GA'sı (\S+)×'ten başlıyor \(≥ (\S+)×\)\.",
     r"Ours' worth-N-humans CI starts at \1× (≥ \2×)."),
    (r"Tabana göre farkın güven aralığı yok ve bizim yöntemin kendi bedeli ölçülmedi\.",
     "No CI for the difference to a baseline, and ours' own multiplier was not measured."),
    (r"\(Rozet eşit tabanların en kötüsüne karşı: (.*?); badge against the worst tied baseline\.\)",
     r"(Badge against the worst tied baseline: \1.)"),
    (r"Yalnız (\d+) küme/test: bootstrap GA anlamsız \(< (\d+)\); yüksek güven verilmez \([^()]*\)\.",
     r"Only \1 clusters/tests: the bootstrap CI is not meaningful (< \2); no 'high confidence'."),
    (r"; yalnız \d+ '\w+' kümesi, GA anlamsız\)", ")"),
    (r"'gercek' \(truth\) sütunu yok: ", "no truth column: "),
    (r"'gercek' \(truth\) sütununun tamamı eksik: ", "the truth column is all missing: "),
    (r"kalibrasyon, taban merdiveni, bootstrap ve çarpan atlandı", "calibration, baseline ladder, bootstrap and multiplier skipped"),
    (r"(\d+) mühür kaydı eklendi; son sha256 ", r"\1 seal record(s) attached; last sha256 "),
    (r"mühür kaydı yok: tahmini gerçek açılmadan önce seal\(\) ile mühürleyin ve evaluate\(prereg=<dizin>\) verin",
     "no seal record: seal the predictions with seal() before the truth is opened and pass evaluate(prereg=<dir>)"),
    (r"'bizim' yöntem açıkça belirtilmedi \(son sütun alındı\); yöntemi önceden belirleyin",
     "ours was not named (the last column was taken); name it in advance"),
    (r"Uyarı: kazanç ezber/tanımadan gelebilir", "Caveat: gain may come from recognition/memorization"),
    (r"ezber denetimi uyarıda", "memorization audit warning"),
    (r"kör kol kazancın çoğunu kaybediyor", "the blind arm loses most of the gain"),
    # flags (kart.kart)
    (r"kanıtlanmadı: sızıntı/ezber", "not proven: leakage/memorization"),
    (r"kanıtlanmadı: sızıntı \((.*) atlandı\)", lambda m: "not proven: leakage ("
     + ", ".join(_check_en(x) for x in m.group(1).split(", ")) + " skipped)"),
    (r"LLM kolu yok: ezber denetimi uygulanamaz \(no LLM arm: memorization audit not applicable\)",
     "no LLM arm: memorization audit not applicable"),
    (r"uygulanamaz: ikili görev, insan paneli yok .*? \(not applicable: (binary task without a human panel .*? to "
     r"compute it)\)", r"not applicable: \1"),
    (r"yüzdelik GA nokta tahminini dışarıda bırakıyor \((.*?), (\d+) sütun\)",
     r"percentile CI excludes the point estimate (\1, \2 columns)"),
    (r"; ci='basic' deneyin", "; try ci='basic'"),
    (r"; temel \(basic\) GA kullanıldı", "; basic CI used"),
    (r"yalnız basit tabanlar \(rastgele/marjinal\)", "only trivial baselines (random/marginal)"),
    (r"en güçlü taban belirsiz \((\d+) taban GA içinde\): seçim gürültüsü",
     r"strongest baseline uncertain (\1 baselines within CI): selection noise"),
    (r"ayar-B tabanı \((.*?)\) daha güçlü: rozet ayar-A tabanına karşı",
     r"setting-B baseline (\1) is stronger: the badge is against the setting-A baseline"),
    (r"dolgu payı yüksek: (\S+) %(\d+) \(geçersiz → tekdüze\)", r"high fill share: \1 \2% (invalid → uniform)"),
    (r"sonradan karşılaştırma: (\d+) satır ön kayıtta yok \(post hoc\)",
     r"post hoc comparisons: \1 row(s) not in the pre-registration"),
    (r"kayıtlı karşılaştırma zayıf tabana karşı \(en güçlü: (.*?)\)",
     r"registered comparison against a weak baseline (strongest: \1)"),
    (r"kazanç ezber/tanımadan olabilir", "gain may come from recognition/memorization"),
    # card row notes (kart.kart)
    (r"tavan yok \(test-tekrar ölçülmemiş\)", "no ceiling (no test-retest measured)"),
    (r"tavan yok: hesaplanamaz", "no ceiling: not computable"),
    (r"^taban yok$", "no baseline"), (r"^ham kol yok$", "no raw arm"),
    (r"tür: model/oracle \(insan test-retest değil\)", "type: model/oracle (not a human test-retest)"),
    (r"^tavan türü: örneklem gürültüsü \(SE sütunu\)$", "ceiling type: sampling noise (SE column)"),
    (r"tavan türü: örneklem gürültüsü \(varyans bileşenleri, tüm gruplar\)",
     "ceiling type: sampling noise (variance components, all groups)"),
    (r"≈ (\S+) gerçek yanıt \(PPI\+\+, sonlu kitle\)", r"≈ \1 real answers (PPI++, finite population)"),
    (r"bizim yöntemin kendi çarpanı yok \(aynı kitlenin etiketiyle eğitildi, döngüsel\); satır ham kolun",
     "ours has no multiplier of its own (trained on the same population's labels, circular); the row is the raw arm's"),
    (r"en yüksek çarpan (.*?) (\S+)× \(rozete dayanak değil\)", r"highest multiplier \1 \2× (not used for the badge)"),
    (r"^ölçülmedi \(bizim ve ham kolun çarpanı yok; en yüksek çarpan (.*?) (\S+)×, dayanak değil\)",
     r"not measured (no multiplier for ours or the raw arm; highest multiplier \1 \2×, not used)"),
    (r"^ölçülmedi$", "not measured"),
    (r"~(\d+) ek yanıt ≈ \$(\S+) \[hesap: yanıt başı \$(\S+)\]", r"~\1 extra answers ≈ $\2 [at $\3 per answer]"),
    (r"belirsiz: (.*?) GA içinde \(strongest baseline uncertain\)", r"uncertain: \1 within CI"),
    # baseline / control / ceiling labels (merdiven.baselineler, cekirdek)
    (r"Rastgele \(başka maddenin gerçek değeri; madde başına tek satır\)", "Random (another item's truth; one row per item)"),
    (r"Rastgele \(madde içi gerçek değer\)", "Random (within-item truth)"),
    (r"Rastgele \(hep beraberlik\)", "Random (always a tie)"),
    (r"Rastgele \(tekdüze\)", "Random (uniform)"), (r"Rastgele \(0\.5\)", "Random (0.5)"),
    (r"Demografi ortalaması \(", "Demographic mean ("), (r"Demografi sınıf payları \(", "Demographic class shares ("),
    (r"Demografi \(", "Demographic ("),
    (r"Madde sınıf payları", "Item class shares"), (r"Genel sınıf payları", "Overall class shares"),
    (r"Madde ortalaması", "Item mean"), (r"Madde medyanı", "Item median"), (r"Genel ortalama", "Overall mean"),
    (r"Genel medyan", "Overall median"), (r"Genel oran", "Overall rate"), (r" \(madde başına tek satır\)", " (one row per item)"),
    (r"Tekdüze dağılım", "Uniform distribution"), (r"Madde geneli \(birini dışarıda bırak\)", "Item overall (leave one out)"),
    (r"Çoğunluk yönü \(kol ortalaması\)", "Majority direction (arm mean)"),
    (r" \(ileriye bakan / look-ahead\)", " (look-ahead)"),
    (r"ayar B: hedef sorunun öbür hücrelerinin cevabı \(setting B\)", "setting B: the target question's other cells"),
    (r"İnsan tavanı \(test-retest\)", "Human ceiling (test-retest)"),
    (r"Model/oracle tavanı \(", "Model/oracle ceiling ("),
    (r"Örneklem gürültüsü tavanı \(varyans bileşenleri\)", "Sampling-noise ceiling (variance components)"),
    (r" \+ izotonik\b", " + isotonic"), (r" \+ sicaklik\b", " + temperature"), (r" \+ dogrusal\b", " + linear"),
    (r"\(kör kol: ad/başlık yok\)", "(blind arm: no names/titles)"),
    # limits (cekirdek / api)
    (r"Bootstrap birimi '(\w+)' \((\d+) küme", lambda m: "Bootstrap unit '" + {"grup": "group", "madde": "item",
     "satir": "row"}.get(m.group(1), m.group(1)) + f"' ({m.group(2)} clusters"),
    (r" × (\d+) '(.*?)' bloğu", r" × \1 '\2' blocks"),
    (r"; toplu metriklerde (\d+)\)\. ", r"; \1 for aggregate metrics). "),
    (r"Maddeler sabit kabul edilir: GA yeni kişilere genellemeyi ölçer, yeni sorulara değil\.",
     "Items are taken as fixed: the CI measures generalisation to new people, not to new questions."),
    (r"Blok × küme iki yönlü örneklenir \(aşağıda\)\.", "Blocks × clusters are resampled two-way (below)."),
    (r"Seriler \(madde\) sabit kabul edilir: .*? backtest kullanın\.",
     "Series (items) are taken as fixed: the CI measures generalisation to new clusters (series/groups) on the same "
     "series; one forecast start (origin) does not measure generalisation in time, use a backtest over several."),
    (r"Baseline'lar ve kalibrasyon çapraz uydurmalıdır \(aynı grubun satırı kendi eşlemesini görmez\)\.",
     "Baselines and calibration are cross-fitted (a group's rows never see their own fit)."),
    (r"Tahmin görevi: otomatik tabanlar .*? baselines=False kullanın\.",
     "Forecast task: the automatic baselines (random / item median or mean) are cross-fitted from the truth of the test "
     "window, so they look ahead of the forecast origin and are not a legitimate ladder rung. Pass baselines known at "
     "the origin (persistence, last-k mean: methods.forecast) as pred_ columns, or use baselines=False."),
    (r"Tavan örneklem gürültüsü tavanıdır .*? sızıntıya işaret eder\.",
     "The ceiling is a sampling-noise ceiling (variance components): reliability = 1 − mean(within-group variance / "
     "group size) / var(group means) over all scored groups; ceiling = √reliability (Spearman: (6/π)·asin(√reliability/2) "
     "under normal theory; r2: reliability). Only the within-group sample separates a noisy group mean from the true "
     "group value; clearly exceeding the ceiling points to leakage."),
    (r"Tavan örneklem gürültüsü tavanıdır \(SE sütunu\): .*? işaret eder\.",
     "The ceiling is a sampling-noise floor (SE column): an oracle that knows the true value misses the observed truth "
     "by N(0, se); a prediction clearly closer than that to the noisy truth points to leakage or to an SE column that "
     "overstates the noise."),
    (r"Tavan bir model/oracle tahminidir .*? sızıntı anlamına gelmez\.",
     "The ceiling is a model/oracle prediction (not a human test-retest): a reference point, not an attainable upper "
     "bound; the gap closed is relative to it, and exceeding it does not mean leakage."),
    (r"Tavan, aynı birimin ikinci ölçümüyle uyumdur\. .*? önce sızıntıya bakılır\.",
     "The ceiling is agreement with a second measurement of the same unit. A noise-free predictor can exceed it; if "
     "the ratio to the ceiling passes 100%, check for leakage first."),
    (r"Doğruluk sınıf payına duyarlıdır; kappa ve dengeli doğruluk birlikte okunmalı\.",
     "Accuracy depends on the class shares; read it together with kappa and balanced accuracy."),
    (r"Yalnız basit tabanlar \(rastgele/marjinal\): bildirilmiş güçlü bir taban olmadan rozet iyimserdir\.",
     "Only trivial baselines (random/marginal): without a declared strong baseline the badge is optimistic."),
    (r"Gerçek sonuç yok: bu kart yalnız mühür ve tahmin özetidir, başarı iddiası taşımaz\.",
     "No truth yet: this card is only the seal and a prediction summary; it makes no performance claim."),
    (r"(?s)Grup düzeyi değerlendirme \(aggregate_by=(.*?)\): (\d+) satır (\d+) grubun (toplamına|ortalamasına).*",
     lambda m: f"Group-level evaluation (aggregate_by={m.group(1)}): {m.group(2)} rows reduced to the "
     + ("sum" if m.group(4) == "toplamına" else "mean") + f" of {m.group(3)} groups; bootstrap over groups. Within-group "
     "sampling noise stays in the means. Automatic baselines (overall mean, demography) are cross-fitted over group "
     "rows and can score clearly below zero on rank metrics (Spearman) even when uninformative; declare a group-level "
     "baseline with baselines_oof / unlabeled_baselines for a meaningful comparison."),
    (r"(?s)Ofsetli değerlendirme \(offset_col=(.*?)\): .*?\(ortalama ofset (\S+)\).*",
     r"Offset evaluation (offset_col=\1): the row offset (mean \2) was subtracted from the truth and every prediction "
     r"before any aggregation; metrics are on (truth − offset)."),
    (r"ön kayıt belgesi (\S+) kilidinden sonra değişmiş \(sha256 eşleşmiyor\); zamanı dosya zamanı "
     r"\(pre-registration document changed after its lock\)",
     r"pre-registration document \1 changed after its lock (sha256 mismatch); its file time is used"),
    (r"^Baskın blok: ", "Dominant block: "),
    (r"Bootstrap iki yönlü \(block_col=(.*?)\): (\d+) blok ve (\d+) '(\w+)' kümesi .*? kapsar\.",
     r"Two-way bootstrap (block_col=\1): \2 blocks and \3 '\4' clusters are resampled independently with replacement in "
     r"every replicate, row weight = block count × cluster count; the CIs cover variation between blocks (e.g. origins "
     r"or weeks) and between series/clusters, including series dependence across blocks."),
    (r"Yalnız (\d+) blok \(< (\d+)\): blok bileşeninin GA katkısı kaba", r"Only \1 blocks (< \2): the block part of the CI is coarse"),
    (r"; (\d+) bloktan az olduğu için bloklar arası değişkenlik bu GA'larla tahmin edilemez \(GA güvenilir değil: CI not "
     r"reliable across blocks\), daha çok köken/hafta ile backtest edin\.",
     r"; with fewer than \1 blocks the variation between blocks cannot be estimated (CI not reliable across blocks): "
     r"backtest over more origins/weeks."),
    (r"; blok sayısını artırmak \(daha çok köken/hafta\) GA'yı sağlamlaştırır\.",
     "; more blocks (more origins/weeks) make the CI sturdier."),
    (r"block_col=(.*?) tek değerli: blok bootstrap uygulanmadı, varsayılan kümeler kullanıldı",
     r"block_col=\1 has one value: block bootstrap not applied, the default clusters are used"),
    (r"^'(.*?)' tek değerli \((.*?)\): ", r"'\1' has one value (\2): "),
    (r"; block_col=(.*?) blokları köken değil, kökenler arası değişkenlik yine kapsanmaz",
     r"; the block_col=\1 blocks are not origins, so variation between origins is still not covered"),
    (r"talep/zaman serisi tahmini \(task=forecast\): .*? Metrikler WMAPE ve yanlılık içerir\.",
     "demand / time-series forecast (task=forecast): the persona shuffle control, the human test-retest ceiling and "
     "the PPI++ 'worth N humans' multiplier do not apply (no human answers or person information) and are skipped. "
     "Metrics include WMAPE and bias."),
    # audit details (denetim, cekirdek, api)
    (r"^(\d+) ortak grup", r"\1 shared group(s)"), (r"\(ör\. ", "(e.g. "),
    (r"'grup' sütunu yok: aynı kişinin satırları iki yana düşebilir", "no group column: one person's rows may fall on both sides"),
    (r"girdide 'bolum' yok; tahminlerin hangi bölmeyle üretildiğini kaynak kod/protokol söylemeli",
     "no split column in the input; the code or protocol must say which split produced the predictions"),
    (r"(\d+) kat · (\d+) grup · (\d+) grup birden çok katta", r"\1 folds · \2 groups · \3 groups in more than one fold"),
    (r"eğitim son (\S+) · test ilk (\S+) · testten sonraki eğitim payı %(\S+)",
     r"last train \1 · first test \2 · share of train rows after the test start \3%"),
    (r"^(\d+) tekrar eden birim$", r"\1 duplicated unit(s)"),
    (r"^sütun yok$", "column missing"),
    (r"^tekrar ölçümü yok$", "no retest measurement"),
    (r"hiçbir yöntem test-retest tavanını anlamlı aşmıyor", "no method is significantly above the test-retest ceiling"),
    (r"hiçbir yöntem model/oracle tavanını anlamlı aşmıyor", "no method is significantly above the model/oracle ceiling"),
    (r"model/oracle tavanını anlamlı aşan: (.*?); tavan zayıf ya da yöntem tavanın görmediği bilgiyi kullanıyor "
     r"\(sızıntıya bakın\)",
     r"significantly above the model/oracle ceiling: \1; the ceiling is weak or the method uses information the "
     r"ceiling does not see (check for leakage)"),
    (r"insan tekrar ölçümünü anlamlı aşan: (.*?)\. Gürültüsüz gerçek tahmini tavanı aşabilir ama önce sızıntıyı ele",
     r"significantly above the human retest: \1. A noise-free estimate can exceed the ceiling, but rule out leakage first"),
    (r": olasılık tahmini, ", ": probability forecast, "),
    (r"; 0/1 gerçeğin sd'siyle kıyas uygulanamaz \(0 = sabit tahmin, 1 = kesin 0/1\); <(\S+) neredeyse sabit tahmin",
     r"; not comparable with the sd of a 0/1 truth (0 = constant prediction, 1 = sharp 0/1); <\1 almost constant"),
    (r": sd oranı ", ": sd ratio "), (r"<(\S+) homojenleşme, >(\S+) aşırı yayılım", r"<\1 homogenisation, >\2 over-dispersion"),
    (r"entropi oranı \(ortalamaların oranı\)", "entropy ratio (ratio of means)"),
    (r"entropi oranı (\S+) \(1 = insan kadar çeşitli\)", r"entropy ratio \1 (1 = as diverse as humans)"),
    (r"gruplar arası ayrışma oranı", "between-group separation ratio"), (r"^hesaplanamadı$", "not computable"),
    (r"en sık sınıf payı %(\S+?)(;|$)", r"majority class share \1%\2"),
    (r"; birincil metrik (\S+) doğruluk türü değil", r"; primary metric \1 is not accuracy-type"),
    (r"; olasılık tahmini: AUC / Brier / logloss / ECE sınıf payından yanıltılmaz",
     "; probability forecast: AUC / Brier / logloss / ECE are not misled by the class share"),
    (r"; tablodaki doğruluk ve F1 \(eşik 0\.5\) için kappa ve dengeli doğruluğa bak",
     "; for accuracy and F1 in the table (threshold 0.5) read kappa and balanced accuracy"),
    (r"; doğruluk yanıltıcı olabilir, kappa ve dengeli doğruluğa bak", "; accuracy can mislead, read kappa and balanced accuracy"),
    (r"kesim tarihi \(cutoff=\) ve 'zaman' \(time\) sütunu yok", "no cutoff date (cutoff=) and no time column"),
    (r"'zaman' \(time\) sütunu var ama LLM eğitim kesim tarihi verilmedi: (.*?) ile kesim öncesi/sonrası karşılaştırılır",
     r"a time column exists but no LLM training cutoff was given: \1 compares before/after the cutoff"),
    (r"^'zaman' \(time\) sütunu yok$", "no time column"),
    (r"öncesi (\d+), sonrası (\d+) birim: az", r"\1 units before, \2 after the cutoff: too few"),
    (r"^kesim (\S+) · ", r"cutoff \1 · "), (r": önce (\S+) / sonra (\S+), fark ", r": before \1 / after \2, difference "),
    (r" · kesimden önceki birimlerde anlamlı daha iyi: ezber şüphesi",
     " · significantly better on units before the cutoff: memorization suspected"),
    (r"ezber testi sonucu verilmedi \(kanit\.ezber\)", "no memorization test result given (memorization=)"),
    (r"^okunamadı: ", "unreadable: "), (r"sonuç dosyasında 'durum' yok", "no status in the result file"),
    (r"sıra değişince aynı kalan %(\S+); tutarlı doğruluk %(\S+)",
     r"unchanged when the order is swapped \1%; consistent accuracy \2%"),
    (r"'ters_<yontem>' sütunu yok", "no reverse_<method> column"),
    (r"en güçlü taban belirsiz: (.*?) GA içinde \(strongest baseline uncertain: .*? within CI\)",
     r"strongest baseline uncertain: \1 within CI"),
    (r" · bizim − eşitlerin en kötüsü (.*?) \(worst case ours − tied baselines, paired\)",
     r" · worst case ours − tied baselines \1 (paired)"),
    (r"; tek tek en düşük alt sınır: ", "; lowest single lower bound: "),
    (r" · (\d+) taban arasından nokta tahminiyle en iyisini seçmek seçim gürültüsü taşır: rozet ve 'en güçlü taban' "
     r"satırını eşitlerin hepsine karşı oku \((.*?)\)$", r" · \2"),
    (r"en güçlü taban (.*?) ikinciden ayrışıyor \((.*?); strongest baseline separated from the runner-up\) · (\d+) taban",
     r"strongest baseline \1 separated from the runner-up (\2) · \3 baselines"),
    (r"bildirilmiş taban yok; en güçlü taban (.*?) \(otomatik rastgele/marjinal\)\. Rozet bu basit tabana karşı: gerçek "
     r"tabanları unlabeled_baselines= \(etiketsiz\) ya da baselines_oof= \(etiketle eğitilmiş, çapraz uydurmalı\) ile "
     r"bildirin", r"no declared baseline; the strongest baseline is \1 (automatic random/marginal). The badge is against "
     r"this trivial baseline: declare real ones with unlabeled_baselines= (label-free) or baselines_oof= (trained on the "
     r"labels, cross-fitted)"),
    (r"doğrulanmadı: ön kayıt verilmedi \(prereg=\); 'bizim' \((.*?)\) adının önceden seçildiği kanıtlanamaz "
     r"\(not verified: no pre-registration\)",
     r"not verified: no pre-registration (prereg=); ours (\1) cannot be shown to have been named in advance"),
    (r"^(\d+) yöntem; ", r"\1 methods; "),
    (r"bolum sütunu: (\d+) test, (\d+) eğitim satırı; metrikler yalnız testte",
     r"split column: \1 test, \2 train rows; metrics on test rows only"),
    (r" · (?:seri içi|kişiye özgü) sapmalar (?:seri|madde) ortalamasından kötü ya da ayırt edilemiyor; genlik "
     r"kalibrasyonu gerekebilir", " · deviations from the mean are worse or indistinguishable; amplitude calibration "
     "may be needed"),
    (r": değerlendirilen satırların %(\S+?)'i \((\d+) blok; eşik %(\d+)\)", r": \1% of the scored rows (\2 blocks; threshold \3%)"),
    (r"; (\S+) bizim \((.*?)\) kişi ağırlıklı (\S+), blok eşit ağırlıklı (\S+) \((.*)\) \(dominant block: person-weighted "
     r"vs block-equal-weighted\)", r"; \1 of ours (\2) person-weighted \3, block-equal-weighted \4 (\5)"),
    (r"grup değeri blok içi ortalamaların eşit ağırlıklı ortalaması", "group value = equal-weighted mean of the block means"),
    (r"satır ağırlığı 1/n_blok", "row weight 1/n_block"),
    (r"blok içi değerlerin ortalaması; bloklar arası fark dışarıda, yeniden ağırlıklama değil",
     "mean of within-block values; between-block differences left out, not a reweighting"),
    (r"; aggregate='sum' ile blok eşit ağırlıklı değer hesaplanmaz \(dominant block\)",
     "; no block-equal-weighted value with aggregate='sum'"),
    (r"^split=('\w+') verildi: bütün satırlar değerlendirme satırı, .*?\((cross-fitted \(declared\): [^()]*)\)$",
     r"split=\1 given: every row is an evaluation row, \2"),
    (r"^eksik \(missing\): ", "missing: "), (r"^yok \(none\)$", "none"),
    # generic fragments (last)
    (r"bizim − taban\b", "ours − baseline"), (r"\bkarıştırılmış tahmin\b", "shuffled prediction"),
    (r"\bkarıştırılmış\b", "shuffled"), (r"− marjinal\b", "− marginal"), (r"'bizim'", "'ours'"), (r"\bbizim\b", "ours"),
    (r" farkı ", " difference "),
]
_PHRASES_EN += [   # constant-prediction audit detail (denetim): its nested-bracket tail defeats the fallback
    (r"(\d+) madde sabit tahmin \(sd 0\), (\d+)/(\d+) maddede sıra/korelasyon metriği tanımsız",
     r"\1 items with a constant prediction (sd 0), rank/correlation metric undefined in \2/\3 items"),
    (r" · NaN maddeler madde ortalamasından \(madde_ici_kor, madde başına Spearman\) sessizce düşer "
     r"\(constant prediction: rank metrics undefined \(\d+ items\); NaN items drop out of per-item means\)",
     r" · NaN items drop out of per-item means (within-item correlation, per-item Spearman) silently"),
]
_PHRASES_EN = [(re.compile(p), r) for p, r in _PHRASES_EN]
_TR_LETTERS = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_TAIL_EN = re.compile(r"^(.*\S)\s+\(([^()]*)\)\.?$", re.S)
_NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_NAME_RE = re.compile(r"(?<![A-Za-z0-9_])(?:tahmin|taban|kontrol|kalib)_[^\s,;:()\[\]`'\"|]+")
_METRIC_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k, v in METRIC_TO_ENGLISH.items() if k != v),
                                                  key=len, reverse=True)) + r")\b")


def column_en(name):
    """Internal column name -> the user's spelling: tahmin_x -> pred_x, taban_x -> 'baseline x' (automatic rungs;
    declared baselines keep their pred_ column), kontrol_karistir_x -> 'control shuffled pred_x', kalib_x+k ->
    'calibrated pred_x + k', tavan -> 'ceiling'. Anything else is returned unchanged."""
    if not isinstance(name, str):
        return name
    if name == "tavan":
        return "ceiling"
    if name == "tavan_model":
        return "ceiling (model)"
    if name.startswith(ON_EK):
        return PRED_PREFIX + name[len(ON_EK):]
    if name.startswith("taban_"):
        x = name[len("taban_"):]
        return "baseline " + AUTO_BASELINE_TO_ENGLISH.get(x, x)
    if name.startswith("kontrol_karistir_"):
        return "control shuffled " + PRED_PREFIX + name[len("kontrol_karistir_"):]
    if name.startswith("kontrol_"):
        return "control " + name[len("kontrol_"):]
    if name.startswith("kalib_"):
        m, plus, how = name[len("kalib_"):].rpartition("+")
        return (f"calibrated {PRED_PREFIX}{m} + {RECALIBRATION_TO_ENGLISH.get(how, how)}" if plus
                else f"calibrated {PRED_PREFIX}{how}")
    return name


def _names_en(text: str) -> str:
    def one(m):
        s = m.group(0)
        tail = len(s) - len(s.rstrip("."))
        return column_en(s[:len(s) - tail]) + s[len(s) - tail:]
    return _NAME_RE.sub(one, text)


def _check_en(name) -> str:
    """Audit check name -> English (unknown names stay as they are)."""
    if not isinstance(name, str):
        return name
    if name in CHECK_TO_ENGLISH:
        return CHECK_TO_ENGLISH[name]
    for p, r in CHECK_PATTERNS_EN:
        if re.match(p, name):
            return re.sub(p, r, name)
    m = _TAIL_EN.match(name)    # a new "Turkish (English)" name: its English part
    if m and _TR_LETTERS.search(m.group(1)) and not _TR_LETTERS.search(m.group(2)) and len(m.group(2).split()) >= 2:
        return m.group(2)
    return name


def _en_tail_keep_numbers(tr: str, en: str) -> str:
    """English part of a "Turkish (English)" text without losing numbers only the Turkish part has (W8). Turkish
    decimal commas count as points. Language-neutral " · " segments of the Turkish part (no Turkish letters, e.g.
    "mae: ours − b = -0.12 [-0.25, 0.02], p=0.200") are kept before the English part; numbers still missing follow
    in brackets. The Turkish words themselves never reach the English view (the W8 fix first bracketed the whole
    Turkish part: "tied: paired CI includes 0 [... · berabere: eşleşik GA 0'ı içeriyor]")."""
    nums = lambda x: _NUM_RE.findall(re.sub(r"(?<=\d),(?=\d)", ".", x))
    if not set(nums(tr)) - set(nums(en)):
        return en
    keep = [g for g in tr.split(" · ") if not _TR_LETTERS.search(g)]
    out = " · ".join(keep + [en])
    have = set(nums(out))
    lost = list(dict.fromkeys(x for x in nums(tr) if x not in have))
    return out + (f" [{', '.join(lost)}]" if lost else "")


def _en_text(s) -> str:
    """Free research text (flag, note, audit detail, limit, badge reason) -> English where a table covers it.
    Dot decimals and pred_ names always; a sentence written as "Turkish (English)" falls back to its English part;
    anything else keeps the Turkish wording."""
    if s is None:
        return ""
    t = re.sub(r"(?<=\d),(?=\d)", ".", str(s))
    if t in CHECK_TO_ENGLISH:
        return CHECK_TO_ENGLISH[t]
    for p, r in _PHRASES_EN:
        t = p.sub(r, t)
    for p, r in CHECK_PATTERNS_EN:
        t = re.sub(p, r, t)
    for tr, en in sorted(CHECK_TO_ENGLISH.items(), key=lambda kv: -len(kv[0])):
        if len(tr) > 12 and tr in t:
            t = t.replace(tr, en)
    m = _TAIL_EN.match(t)
    if m and _TR_LETTERS.search(m.group(1)) and not _TR_LETTERS.search(m.group(2)) and len(m.group(2).split()) >= 3:
        t = _en_tail_keep_numbers(m.group(1), m.group(2))
    t = re.sub(r"(?<![\w%])%(\d+(?:\.\d+)?)", r"\1%", t)
    return _METRIC_RE.sub(lambda q: METRIC_TO_ENGLISH[q.group(1)], _names_en(t))


def _label_en(label) -> str:
    """Card row / ladder / calibration label -> English (method names are the user's and stay)."""
    if label is None:
        return ""
    t = str(label)
    if t in ROW_LABEL_TO_ENGLISH:
        return ROW_LABEL_TO_ENGLISH[t]
    for p, r in ROW_LABEL_PREFIXES_EN:
        t = re.sub(p, r, t)
    return _en_text(t)


def _kind_en(kind) -> str:
    return KIND_TO_ENGLISH.get(kind, kind) if isinstance(kind, str) else kind


def _finite(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _ceiling_near_baseline(m: dict) -> bool:
    """ceiling − strongest baseline is about 0, or smaller than its CI width (approximated from the two CIs as
    sqrt(w_ceiling² + w_baseline²), unpaired): the gap-closed ratio then divides by noise."""
    c, b = m.get("tavan") or {}, m.get("en_iyi_baseline") or {}
    cv, bv = _finite(c.get("deger")), _finite(b.get("deger"))
    if cv is None or bv is None:
        return False
    gap = abs(cv - bv)
    if gap <= 1e-9 * max(1.0, abs(cv), abs(bv)):
        return True
    w = [_finite(x.get("ust")) - _finite(x.get("alt")) for x in (c, b)
         if _finite(x.get("alt")) is not None and _finite(x.get("ust")) is not None]
    return bool(w) and gap < math.sqrt(sum(v * v for v in w))


def _gap_closed(report: dict) -> dict | None:
    """summary()['gap_closed']: {"value", "lo", "hi"} of (ours − baseline)/(ceiling − baseline); value None with
    "note": GAP_NA_NOTE when the ceiling is about the baseline. The Turkish keys "deger" / "alt" / "ust" are kept as
    deprecated aliases holding the raw research numbers (use value / lo / hi)."""
    m = report.get("manset") or {}
    g = m.get("bosluk_kapanan")
    if not g:
        return None
    out = {"value": g.get("deger"), "lo": g.get("alt"), "hi": g.get("ust")}
    if _ceiling_near_baseline(m):
        out.update(value=None, lo=None, hi=None, note=GAP_NA_NOTE)
    out.update(deger=g.get("deger"), alt=g.get("alt"), ust=g.get("ust"))
    return out


def _headline_en(s: dict) -> str | None:
    """English one-line headline built from the summary fields (dot decimals)."""
    o, b = s.get("ours"), s.get("best_baseline")
    if s.get("awaiting_truth"):
        z = s.get("seals") or []
        bm = s.get("ours_seal")
        if bm and bm.get("sha256"):   # the seal the Turkish card shows (cekirdek._bizim_muhuru), not the chain's last
            ad = "last record" if bm.get("match") == "son" else "ours' seal"
            muhur = (f", sealed ({len(z)} record(s), {ad} #{bm.get('index')} sha256 {str(bm['sha256'])[:12]}"
                     + (f"; {bm['note']}" if bm.get("note") else "") + ")")
        else:
            muhur = f", sealed ({len(z)} record(s), last sha256 {str(z[-1].get('sha256'))[:12]})" if z else ", NOT sealed"
        return (f"awaiting truth: predictions of {len(s.get('predictions') or {})} method(s)" + muhur
                + "; no metric until the truth is opened")
    if not o or o.get("value") is None:
        return None
    ci = lambda d: f" [{_f3(d.get('lo'))}, {_f3(d.get('hi'))}]" if d.get("lo") is not None and d.get("hi") is not None else ""
    out = f"{s.get('primary_metric')}: ours ({column_en(o.get('column'))}) {_f3(o['value'])}{ci(o)}"
    if b and b.get("value") is not None:
        out += f" vs strongest simple baseline ({column_en(b.get('column'))}) {_f3(b['value'])}{ci(b)}"
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
        "gap_closed": _gap_closed(report), "ratio_to_ceiling": m.get("tavana_oran"),
        "baseline_ratio_to_ceiling": m.get("taban_tavana_oran"),
        "worth_humans_at_100": None if not worth or worth.get("deger") is None else
        {"multiplier": worth["deger"], "lo": worth.get("alt"), "hi": worth.get("ust"), "note": worth.get("not")},
        "ours_minus_baseline": k.get("fark"),
        "badge": _badge_english(k), "badge_code": k.get("rozet"), "below_baseline": _below_baseline(k),
        "badge_reason": k.get("rozet_gerekce"),
        "badge_reason_en": _en_text(k.get("rozet_gerekce")) if k.get("rozet_gerekce") else None,
        "flags": k.get("etiketler") or [],
        "flags_en": [_en_text(e) for e in k.get("etiketler") or []],
        # name / status / detail in English (untranslated details stay Turkish); "check" (Turkish name) and
        # "detail_tr" keep the research wording ("check" is the older key, kept for compatibility)
        "audits": [{"name": _check_en(d.get("kontrol")), "status": STATUS_TO_ENGLISH.get(d.get("durum"), d.get("durum")),
                    "detail": _en_text(d.get("ayrinti")) if d.get("ayrinti") is not None else None,
                    "check": d.get("kontrol"), "detail_tr": d.get("ayrinti"), "value": d.get("deger")}
                   for d in report.get("denetim") or []],
        "ladder": [{"column": r.get("sutun"), "label": r.get("etiket"), "kind": r.get("tur"), "value": r.get("deger"),
                    "lo": r.get("alt"), "hi": r.get("ust"), "diff_prev": r.get("fark_onceki"), "p": r.get("p")}
                   for r in report.get("merdiven") or []],
        "headline": m.get("cumle"),
        "seals": _seals(report),
        "ours_seal": _ours_seal(report),
        "ceiling_kind": SE_CEILING_LABEL if report.get("se_tavani") and m.get("tavan") else
        CEILING_KIND.get(report.get("tavan_turu") or ("insan_tekrar" if m.get("tavan") else None)),
        "declared_baselines": list((report.get("tabanlar") or {}).get("bildirilen") or []),
        "only_trivial_baselines": bool((report.get("tabanlar") or {}).get("yalniz_basit")),
        "awaiting_truth": bool(report.get("gercek_bekleniyor")),
        "predictions": report.get("tahmin_ozeti"),
        "aggregate": report.get("toplama"),
        "cluster": INTERNAL_TO_COLUMN.get(report.get("tutma_birimi"), report.get("tutma_birimi")),
        "n_clusters": (report.get("kart") or {}).get("n_kume"),
        "cluster_override": report.get("kume_sutunu") is not None,
        "min_effect": report.get("min_effect", (report.get("kanit_kart") or {}).get("min_effect")),
        "offset": report.get("ofset"),
        "noise_ceiling": report.get("gurultu_tavani"),
        "se_ceiling": _se_ceiling_en(report.get("se_tavani")),
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
                          "blok_ortalamasi": "mean of within-block values (between-block differences left out; not a reweighting)"}.get(b.get("agirlik")),
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
    if report.get("oran_hatasi"):
        out["rate_error"] = _rate_error_en(report["oran_hatasi"])
    if report.get("ortalama_medyan"):
        out["mean_median"] = [{"mean": _method(r["ortalama"]), "median": _method(r["medyan"]),
                               "metric": METRIC_TO_ENGLISH.get(r["metrik"], r["metrik"]),
                               "mean_value": r["ortalama_deger"], "median_value": r["medyan_deger"],
                               "mean_bias": r["ortalama_yanlilik"], "median_bias": r["medyan_yanlilik"]}
                              for r in report["ortalama_medyan"]]
    out["headline_en"] = _headline_en(out)
    return out


def _rate_error_en(o: dict) -> dict:
    """report["oran_hatasi"] -> summary()["rate_error"] (English keys, user column names)."""
    box = lambda b: {"value": b.get("deger"), "lo": b.get("alt"), "hi": b.get("ust"),
                     "point_outside_ci": bool(b.get("nokta_ga_disi"))}
    return {"segment_columns": [column_en(s) for s in o.get("segmentler") or []], "n": o.get("n"), "B": o.get("B"),
            "ours": column_en(o.get("bizim")),
            "definition": "total = |sum pred − sum truth| / n; segment = sum over segments of |sum pred − sum truth| "
                          "/ n (n-weighted mean of per-segment rate gaps); lower is better",
            "rows": [{"column": column_en(r["sutun"]), "label": r.get("etiket"), "kind": _kind_en(r.get("tur")),
                      "total": box(r["toplam"]),
                      "segment": {column_en(s): {**box(v), "n_segments": v.get("n_segment")}
                                  for s, v in (r.get("segment") or {}).items()}} for r in o.get("satirlar") or []],
            "differences": [{"a": column_en(f["a"]), "b": column_en(f["b"]), "label": f.get("etiket"),
                             "measure": "total" if f["olcu"] == "toplam" else column_en(f["olcu"]), "diff": f["fark"],
                             "lo": f.get("alt"), "hi": f.get("ust"), "p": f.get("p"), "a_better": f.get("a_iyi")}
                            for f in o.get("farklar") or []],
            "skipped": [column_en(c) for c in o.get("atlanan") or []]}


def _rate_error_md(re_: dict, en: bool) -> list[str]:
    """Card table of summary()["rate_error"]: per-column total / segment rate error, then ours − other (paired)."""
    segs = re_["segment_columns"]
    out = ["", "## Rate error (lower is better)", "",
           "| Method | Kind | Total abs(Σp − Σy)/n [95% CI] | "
           + " | ".join(f"Segment `{_md(s)}` Σ abs(Σp − Σy)/n [95% CI]" for s in segs) + " |",
           "|---|---|---|" + "---|" * len(segs)]
    for r in re_["rows"]:
        lab = _label_en(r.get("label")) if en else r.get("label")
        cells = [r["total"]] + [r["segment"].get(s) or {} for s in segs]
        out.append(f"| {_md(lab)} (`{r['column']}`) | {r['kind']} | "
                   + " | ".join(f"{_num(c.get('value'))}{_ci(c.get('lo'), c.get('hi'))}" for c in cells) + " |")
    if re_["differences"]:
        out += ["", "| Ours − other | Measure | Difference [95% CI] | p | Verdict |", "|---|---|---|---|---|"]
        for f in re_["differences"]:
            ci_ = f" [{f['lo']:+.3f}, {f['hi']:+.3f}]" if f.get("lo") is not None and f.get("hi") is not None else ""
            v = {True: "ours lower (CI below 0)", False: "ours higher (CI above 0)"}.get(f.get("a_better"),
                                                                                       "unclear (CI includes 0)")
            out.append(f"| `{f['a']}` − `{f['b']}` | {_md(f['measure'])} | {f['diff']:+.3f}{ci_} | "
                       + ("—" if f.get("p") is None else f"{f['p']:.3f}") + f" | {v} |")
    out += ["", f"Total = |sum pred − sum truth| / n (= |mean_bias|); segment = sum over segments of |sum pred − sum "
                f"truth| / n, the n-weighted mean of per-segment rate gaps. Paired cluster bootstrap (B={re_['B']}, "
                "the ladder's clusters and seed); abs(·) is ≥ 0 in every replicate, so near a zero gap the CI can lie above "
                "the point: read the paired differences."]
    if re_.get("skipped"):
        out += ["", "Not scored (non-numeric or missing values): " + ", ".join(f"`{c}`" for c in re_["skipped"]) + "."]
    return out


def _comparison_en(c: dict) -> dict:
    v = c.get("sonuc") or ""
    yer = "below" if "altında" in v else "above"      # where the printed (raw) CI lies; lower-is-better flips it
    verdict = (f"a better (CI {yer} 0)" if v.startswith("A iyi") else f"b better (CI {yer} 0)" if v.startswith("B iyi")
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


def _rank_note(report: dict, lang: str = "mixed") -> str | None:
    chance = RANK_METRICS.get(report.get("birincil"))
    if chance is None:
        return None
    low = [r for r in report.get("merdiven") or [] if r.get("sutun") in ("taban_marjinal", "taban_demografi")
           and r.get("deger") is not None and r["deger"] < chance]
    if not low:
        return None
    en = lang == "en"
    names = ", ".join(f"`{column_en(r['sutun']) if en else r['sutun']}`" for r in low)
    metric = METRIC_TO_ENGLISH.get(report.get("birincil"), report.get("birincil")) if en else report.get("birincil")
    return (f"Average baselines ({names}) score below chance ({chance:g}) on {metric}. This is "
            "expected, not a bug: they are cross-fitted (a row never sees its own label), so a leave-one-out "
            "mean or rate moves against the held-out truth. On ranking and correlation metrics, compare ours "
            "with the informative rungs (persistence, learners) rather than with these.")


def _direction(metric: str | None) -> int | None:
    """+1 higher is better, -1 lower is better, None unknown (internal metric name)."""
    for g in TASK_TO_ENGLISH:
        m = next((x for x in _metrik.aile(g, True, True, True, oran=True) if x.ad == metric), None)
        if m is not None:
            return m.yon
    return None


def _raw_diff_note(f: dict, metric: str | None) -> str:
    """Raw ours − baseline next to the metric-direction difference, so the headline uses the same sign rule as the
    registered comparisons (raw A − B)."""
    yon = _direction(metric)
    if yon is None:
        return ""
    name = METRIC_TO_ENGLISH.get(metric, metric)
    if yon > 0:
        return f"; raw ours − baseline = {f['fark']:+.3f} (same sign: {name}, higher is better)"
    ci = (f" [{-f['ust']:+.3f}, {-f['alt']:+.3f}]" if f.get("alt") is not None and f.get("ust") is not None else "")
    return (f"; raw ours − baseline = {-f['fark']:+.3f}{ci} ({name}, lower is better: negative favours ours, the "
            "sign rule of the registered comparisons)")


def _small_gain_reason(f: dict | None, reason: str, min_effect: float | None = None) -> str:
    """English badge reason when the paired difference is significant but below the threshold (min_effect, else
    ANLAMLI_FARK; was empty in D12)."""
    esik = _kart.ANLAMLI_FARK if min_effect is None else min_effect
    if not f or any(f.get(x) is None for x in ("fark", "alt", "ust")) or not (f["alt"] > 0 and f["fark"] < esik):
        return reason
    if reason.strip() and not reason.startswith("Significant but small"):
        return reason
    base = reason.strip() or (f"Significant but small difference ({f['fark']:+.3f}, CI [{f['alt']:+.3f}, "
                              f"{f['ust']:+.3f}]); the strongest simple baseline is almost level.")
    src = "ANLAMLI_FARK" if min_effect is None else f"min_effect={esik:g}"
    return f"{base} 'High' needs a difference of at least {esik:g} ({src}, metric direction) with the CI above 0."


def _se_ceiling_en(x: dict | None) -> dict | None:
    if not x:
        return None
    return {"column": x.get("sutun"), "metric": METRIC_TO_ENGLISH.get(x.get("metrik"), x.get("metrik")),
            "label": x.get("etiket"), "draws": x.get("R"), "seed": x.get("tohum"), "n": x.get("n"),
            "analytic": x.get("analitik"), "value": x.get("deger"), "lo": x.get("alt"), "hi": x.get("ust"),
            "cells": x.get("n_hucre")}


CARD_ITEM_ROWS = 30     # pending card: per-item prediction rows printed (constant items first); the rest stay in the report


def _top_labels(p: dict, k: int | None = None) -> str:
    return ", ".join(f"{_md(a)} {100 * b:.0f}%" for a, b in list((p.get("en_sik") or {}).items())[:k])


def _item_cell(p: dict | None) -> str:
    if not p:
        return "—"
    if p.get("tur") == "etiket":
        return f"{_top_labels(p, 1)} ({p.get('n_farkli')} distinct)"
    if p.get("tur") == "sayi":
        return f"{_num(p.get('ortalama'))} ± {_num(p.get('sd'))}"
    return "vector" if p.get("tur") == "vektor" else "—"


def _predictions_md(preds: dict, en: bool) -> list[str]:
    """Awaiting-truth prediction summary (cekirdek.bekleyen tahmin_ozeti): per column distribution or label counts
    (distinct labels), item counts with constant items, and a per-item table (mean ± sd, or the top label)."""
    col = {c: f"`{column_en(c) if en else c}`" for c in preds}
    out = ["", "## Predictions (awaiting truth)", "",
           "| Column | n | missing | mean | sd | min | median | max | top labels | distinct | items (constant) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    sabit = []
    for c, p in preds.items():
        items = "—" if p.get("n_madde") is None else f"{p['n_madde']} ({p.get('n_sabit_madde', 0)})"
        out.append(f"| {_md(p.get('etiket'))} ({col[c]}) | {p.get('n')} | {p.get('eksik')} | {_num(p.get('ortalama'))} | "
                   f"{_num(p.get('sd'))} | {_num(p.get('en_kucuk'))} | {_num(p.get('medyan'))} | "
                   f"{_num(p.get('en_buyuk'))} | {_top_labels(p) or ('vector' if p.get('tur') == 'vektor' else '—')} | "
                   f"{p.get('n_farkli', '—')} | {items} |")
        if p.get("n_sabit_madde"):
            ms = p.get("sabit_maddeler") or []
            sabit.append(f"- {col[c]}: {p['n_sabit_madde']}/{p['n_madde']} items constant ("
                         + ", ".join(_md(m) for m in ms) + (", …" if p["n_sabit_madde"] > len(ms) else "") + ")")
        elif p.get("sabit"):
            sabit.append(f"- {col[c]}: the whole column is constant")
    if sabit:
        out += ["", "Constant predictions (sd 0; rank / correlation metrics will be undefined there):", ""] + sabit
    per = {c: p["maddeler"] for c, p in preds.items() if p.get("maddeler")}
    if not per:
        return out
    sabit_m = {m for p in preds.values() for m in p.get("sabit_maddeler") or []}
    maddeler = list(dict.fromkeys(m for d in per.values() for m in d))
    maddeler = [m for m in maddeler if m in sabit_m] + [m for m in maddeler if m not in sabit_m]
    n_madde = max((p.get("n_madde") or 0) for p in preds.values())
    out += ["", "## Predictions per item (awaiting truth)", "",
            "| Item | " + " | ".join(f"{_md(preds[c].get('etiket'))} ({col[c]})" for c in per) + " |",
            "|---|" + "---|" * len(per)]
    for m in maddeler[:CARD_ITEM_ROWS]:
        out.append(f"| {_md(m)} | " + " | ".join(_item_cell(per[c].get(m)) for c in per) + " |")
    kirpik = any(p.get("maddeler_kirpildi") for p in preds.values())
    if len(maddeler) > CARD_ITEM_ROWS or kirpik:
        out += ["", f"{min(len(maddeler), CARD_ITEM_ROWS)} of {n_madde} items shown (constant items first)"
                + ("; over 100 items the report keeps per-item summaries for constant items only" if kirpik else "")
                + "; the rest are in `report[\"tahmin_ozeti\"]`."]
    out += ["", "Cell: mean ± sd of the predictions in the item; labels: the most frequent label and its share."]
    return out


def card_markdown(report: dict, lang: str = "mixed") -> str:
    """Evidence card as Markdown: badge, the six card rows, the ladder, the audits and the seals.

    lang="mixed" (default) keeps the research card's Turkish labels next to the English ones;
    lang="en" is English throughout: the badge reason (`summary()["badge_reason_en"]`), flags, row and ladder labels,
    rung kinds (method / control / baseline / calibrated / ceiling), audit names and the common audit details and
    limits are translated, internal columns print as the user's names (pred_x, "baseline random", "control shuffled
    pred_x"), decimals use dots. Text without a translation keeps its Turkish wording."""
    k = report.get("kanit_kart") or _kart.kart(report)
    s = summary(report)
    en = lang == "en"
    out = [f"# Evidence card: {_md(report.get('baslik') or report.get('veri_adi') or '')}", ""]
    reason = _md(k.get("rozet_gerekce"))
    if en:
        reason = re.sub(r" ?Caveat: gain may come from recognition/memorization \([^()]*\)\.", "",
                        _md(s.get("badge_reason_en")))
    badge = _badge_english(k)
    if s.get("below_baseline"):
        badge += " (ours − strongest baseline CI is entirely below 0; research rule: inconclusive)"
    if s.get("only_trivial_baselines"):
        badge += (" (only trivial baselines: random/marginal; declare real ones with unlabeled_baselines= or "
                  "baselines_oof=)")
    cav = s.get("badge_caveat")
    if en and k.get("rozet") == "kararsiz":
        reason = _small_gain_reason(k.get("fark"), reason, report.get("min_effect", k.get("min_effect")))
    out.append(f"**Badge: {badge}.** " + reason + (f" Caveat: {cav}." if cav else ""))
    if k.get("etiketler"):
        out.append("")
        out.append("Flags: " + ", ".join(_md(_en_text(e)) if en else _dot(e, lang) for e in k["etiketler"]))
    meta = (f"Task: {s['task'] or '—'} · primary metric: {s['primary_metric'] or '—'}"
            + (f" (`{s['primary_metric_internal']}`)" if s['primary_metric_internal'] else "")
            + f" · clusters: {k.get('n')}" + (f" (cluster=`{s['cluster']}`)" if s.get("cluster_override") else ""))
    if k.get("kaynak"):
        meta += f" · source: {_md(k.get('kaynak'))}"
    if s.get("aggregate"):
        a = s["aggregate"]
        meta += ((f" · level: rows, unit-clustered (aggregate_by={a.get('by')}: {a.get('n_groups')} units; "
                  f"{a['pooled_primary']} is pooled, NOT a unit mean)" if a.get("pooled_primary") else
                  f" · level: unit mean (aggregate_by={a.get('by')}: metric within each of {a.get('n_groups')} units, "
                  "then their mean)") if a.get("aggregate") == UNIT_MEAN else
                 f" · level: group (aggregate_by={a.get('by')}: {a.get('n_rows')} rows -> {a.get('n_groups')} groups)")
        if a.get("n_dropped"):
            meta += f" · {a['n_dropped']} of {a['n_units']} units dropped (no scored row)"
    if s.get("block_bootstrap"):
        b = s["block_bootstrap"]
        meta += (f" · bootstrap: two-way, blocks of `{b['column']}` ({b['n_blocks']} blocks) × clusters"
                 + (" — CI not reliable across blocks (< 5 blocks)" if (b.get("n_blocks") or 0) < 5 else
                    " — few blocks (< 10): coarse CIs" if (b.get("n_blocks") or 0) < 10 else ""))
    if s.get("min_effect") is not None:
        meta += f" · min_effect={s['min_effect']:g} (smallest difference for 'high')"
    if s.get("ceiling_kind"):
        meta += f" · ceiling: {s['ceiling_kind']}"
    if s.get("declared_baselines"):
        meta += " · declared baselines: " + ", ".join(f"`{column_en(c) if en else c}`" for c in s["declared_baselines"])
    out += ["", meta, ""]
    if s.get("single_origin"):
        so = s["single_origin"]
        out += [f"Note: {_en_text(so['note']) if en else so['note']} (`{so['column']}` = {_md(so['value'])}).", ""]
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
        elif r["tur"] == "tavan" and s.get("ceiling_kind") == SE_CEILING_LABEL:
            row = "Sampling-noise ceiling (SE column)"
        val = f"{_num(r.get('deger'), u)}{_ci(r.get('alt'), r.get('ust'), u)}"
        if r["tur"] == "bosluk" and (s.get("gap_closed") or {}).get("note") == GAP_NA_NOTE:
            val = GAP_NA_NOTE
        if en:
            out.append(f"| {row} | {_md(_label_en(r.get('etiket')))} | {val} | {_md(_en_text(r.get('not')))} |")
        else:
            out.append(f"| {row} | {_md(r.get('etiket'))} | {val} | {_dot(r.get('not'), lang)} |")
    f = k.get("fark")
    if f and f.get("fark") is not None:
        out += ["", f"Ours − strongest baseline (paired, metric direction, positive = better): {f['fark']:+.3f}"
                    + (f" [{f['alt']:+.3f}, {f['ust']:+.3f}]" if f.get("alt") is not None and f.get("ust") is not None else "")
                    + _raw_diff_note(f, report.get("birincil"))]
    se_c = s.get("se_ceiling")
    if se_c:
        out += ["", f"Ceiling type: {SE_CEILING_LABEL}: oracle column truth + N(0, `{_md(se_c['column'])}`) over "
                    f"{se_c['n']} scored rows" + (f" ({se_c['cells']} aggregated cells, cell sd sqrt(sum se²)/n)"
                                                  if se_c.get("cells") is not None else "")
                    + f", {se_c['draws']} seeded draws averaged in {se_c['metric']} -> "
                    f"{_num(se_c.get('value'))}{_ci(se_c.get('lo'), se_c.get('hi'))}; analytic "
                    + (f"{_num(se_c['analytic'])}" if se_c.get("analytic") is not None else "not given")
                    + (" (sum(w·sqrt(2/pi)·se)/sum(w))" if se_c["metric"] == "mae" and se_c.get("cells") is None
                       else "")
                    + ". Unlike a model/oracle reference, clearly beating this floor points to leakage or an SE column "
                      "that overstates the noise."]
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
    cil = [(c, v) for c, v in (report.get("sonuclar") or {}).items()
           if v.get("tur") in ("baseline", "yontem", "kalibre") and "mean_bias" in (v.get("metrikler") or {})]
    if cil:   # calibration-in-the-large: binary = predicted rate − base rate, every arm, cluster-bootstrap CI
        out += ["", "| Calibration-in-the-large | Kind | Mean prediction − mean truth [95% CI] |", "|---|---|---|"]
        for c, v in cil:
            m = v["metrikler"]["mean_bias"]
            lab, col, kind = ((_label_en(v.get("etiket")), column_en(c), _kind_en(v.get("tur"))) if en
                              else (v.get("etiket"), c, v.get("tur")))
            out.append(f"| {_md(lab)} (`{col}`) | {kind} | {_signed(m.get('deger'))}"
                       + (f" [{m['alt']:+.3f}, {m['ust']:+.3f}]" if m.get("alt") is not None and m.get("ust") is not None
                          else "") + " |")
        out += ["", "Signed (0 = the mean is right; binary: + over-predicts the rate). Diagnostic, not a primary metric; "
                    "`paired_ci(..., metric=\"mean_bias\")` compares two arms."]
    if s.get("rate_error"):
        out += _rate_error_md(s["rate_error"], en)
    if s.get("comparisons"):
        out += ["", "## Registered comparisons / drop conditions", "",
                "| Comparison | Kind | Metric | A − B [95% CI] | p | n clusters | Verdict |", "|---|---|---|---|---|---|---|"]
        for c in s["comparisons"]:
            ci_ = (f" [{c['lo']:+.3f}, {c['hi']:+.3f}]" if c.get("lo") is not None and c.get("hi") is not None else "")
            dv = "—" if c.get("diff") is None else f"{c['diff']:+.3f}{ci_}"
            p = "—" if c.get("p") is None else f"{c['p']:.3f}"
            out.append(f"| {_md(_en_text(c.get('label')) if en else c.get('label'))} | {c['kind']} | {c.get('metric')} | {dv} | {p} | {c.get('n')} | {c['verdict']} |")
        out += ["", "A − B on the same cluster-bootstrap resamples as the ladder (b empty: A's value against 0). "
                    "For lower-is-better metrics a negative difference favours A."]
    if s["ladder"]:
        out += ["", "## Baseline ladder", "", "| Rung | Kind | Value [95% CI] | Δ vs previous [95% CI] | p |",
                "|---|---|---|---|---|"]
        for r in report.get("merdiven") or []:
            d = "—" if r.get("fark_onceki") is None else f"{r['fark_onceki']:+.3f}{_ci(r.get('fark_alt'), r.get('fark_ust'))}"
            p = "—" if r.get("p") is None else f"{r['p']:.3f}"
            lab, col, kind = ((_label_en(r.get("etiket")), column_en(r.get("sutun")), _kind_en(r.get("tur"))) if en
                              else (r.get("etiket"), r.get("sutun"), r.get("tur")))
            out.append(f"| {_md(lab)} (`{col}`) | {kind} | "
                       f"{_num(r.get('deger'))}{_ci(r.get('alt'), r.get('ust'))} | {d} | {p} |")
    if s.get("predictions"):
        out += _predictions_md(s["predictions"], en)
    if s["audits"]:
        out += ["", "## Audits", "", "| Check | Status | Detail |", "|---|---|---|"]
        for a in s["audits"]:
            if en:
                out.append(f"| {_md(a['name'])} | {a['status']} | {_md(a['detail'])} |")
            else:
                out.append(f"| {_md(a['check'])} | {a['status']} | {_dot(a['detail_tr'], lang)} |")
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
    note = _rank_note(report, lang)
    if limits or note:
        out += ["", "## Limits", ""] + [f"- {_md(_en_text(x)) if en else _dot(x, lang)}" for x in limits] \
            + ([f"- {note}"] if note else [])
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
                task: str | None = None, aggregate: str = "mean", ci: str = "percentile",
                block_col: str | None = None) -> dict:
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
        together. aggregate="sum": sum per aggregate_by group instead of the mean; aggregate="unit_mean": rows weighted
        1/n_unit and units as clusters, as `evaluate(aggregate="unit_mean")` (an unknown value raises; "unit_mean"
        used to fall through to the group mean silently). group= is the counterpart of `evaluate(cluster=)`.
        ci="basic": 2*diff - quantiles.
    block_col: the two-way block bootstrap of `evaluate(block_col=)` (blocks × clusters resampled independently,
        same default: off), so a pre-registered paired decision does not flip with the bootstrap type (D7 #9:
        evaluate(block_col="origin") gave [-0.059, +0.015], paired_diff without blocks [-0.041, -0.003]). With
        aggregate_by= the column must be constant within each group. On a report: the report's own bootstrap is
        reused (its block_col is returned); a different block_col raises ValueError.
    Returns {"metric", "a", "b", "value_a", "value_b", "diff", "lo", "hi", "p", "n" (clusters), "cluster",
    "block_col", "n_blocks", "a_better", "B", "source"}. diff is raw a − b: for lower-is-better metrics a negative
    diff favours a."""
    if isinstance(frame_or_report, dict):
        return _paired_from_report(frame_or_report, a, b, metric, block_col)
    if aggregate not in AGGREGATES + (UNIT_MEAN,):
        raise ValueError(f"aggregate must be one of {AGGREGATES + (UNIT_MEAN,)}, not {aggregate!r}")
    if aggregate == UNIT_MEAN and aggregate_by is None:
        raise ValueError("aggregate='unit_mean' needs aggregate_by= (the unit column, e.g. the person)")
    d = to_internal(frame_or_report)
    blok = None
    if block_col is not None:
        blok = block_col if block_col in d.columns else COLUMN_TO_INTERNAL.get(block_col, block_col)
        if blok not in d.columns:
            raise ValueError(f"block_col {block_col!r} is not a column of the frame")
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
        if aggregate == UNIT_MEAN:
            d, _ = _unit_mean(d, aggregate_by, task_i, None)     # rows stay rows: the task is kept
        else:
            d, _ = _aggregate(d, aggregate_by, task_i, how=aggregate, keep={blok: "block_col"} if blok else ())
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
    if blok is not None:
        from .cekirdek import _blok_kur
        _blok_kur(v, blok, False)        # as evaluate: NaN raises, one distinct value -> default clusters
    orn = _boot.Orneklem(v.kume, B=B, B_toplu=B, tohum=seed, madde_kume=v.kume_adi == "madde" and v.blok is None,
                         iki_yon=v.iki_yon)
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
            "a_better": _a_better(M, va, vb) if cb else None, "n": v.n_kume, "cluster": v.kume_adi,
            "block_col": block_col if v.blok is not None else None,
            "n_blocks": int(dd[v.blok].nunique()) if v.blok is not None else None, "B": int(B), "source": "frame"}


def _paired_from_report(report: dict, a: str, b: str | None, metric: str | None, block_col: str | None = None) -> dict:
    met = _metric(metric) or report.get("birincil")
    rb = (report.get("blok_bootstrap") or {}).get("sutun")
    if block_col is not None and rb not in (block_col, COLUMN_TO_INTERNAL.get(block_col, block_col)):
        raise ValueError(f"block_col {block_col!r} differs from the report's ({rb!r}): the report's CIs come from its "
                         "own bootstrap; pass the frame to resample other blocks")
    rb_en = INTERNAL_TO_COLUMN.get(rb, rb)     # "zaman" -> "time"; user columns stay
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
                "cluster": report.get("tutma_birimi"), "block_col": rb_en,
                "n_blocks": (report.get("blok_bootstrap") or {}).get("n_blok"), "B": None, "source": "report"}
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
