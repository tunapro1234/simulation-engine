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

import json
import math
import os
import re
import tempfile
import warnings
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
from .sema import ON_EK, gercek_yok, hazirla

VERSION = SURUM
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
CEILING_KIND = {"insan_tekrar": "human retest", "model": "oracle/model"}
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


def _task(task: str | None) -> str | None:
    return None if task is None else TASK_TO_INTERNAL.get(task, task)


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
             multiplier_max_rows: int | None = MULTIPLIER_MAX_ROWS) -> dict:
    """Long table -> evidence report (baseline ladder, paired cluster-bootstrap CIs, calibration, segments,
    audits, effective-sample multiplier, evidence card). Same numbers as kanit.degerlendir.

    ours / raw: method names without the `pred_` prefix. Name `ours` BEFORE looking at results: if omitted the
    last method is used and a selection-bias warning is added to the audits.
    task: binary | multiclass | distribution | continuous | ab (auto-detected when None).
    primary: metric name (English alias or internal); default per task, macro-F1 when the majority class > 60 %.
    B / B_aggregate: bootstrap replicates for per-unit / aggregate metrics. Cluster = group > item > row.
    memorization: result dict of `memorization_audit` or a path to its JSON.
    prereg: list of seal records, or a registry directory (its chain is attached to the report).
    circular: priors trained on the same population's target labels; excluded from "worth N humans".
    unlabeled_baselines: prediction columns that never saw target labels (e.g. persistence) -> baseline rungs.
    baselines_oof: prediction columns trained on the target labels but cross-fitted (out-of-fold) -> baseline rungs.
        The badge and headline compare ours with the strongest baseline (automatic + declared). When nothing is
        declared and the strongest is the random/marginal rung, the card says "only trivial baselines".
    ceiling_col: a model/oracle ceiling column ("pred_x" or any column name) instead of the human `retest`
        ceiling; the card labels the ceiling type and the gap closed is computed from it. ceiling_label names it.
    aggregate_by: evaluate at group level ("group" or any column): truth and predictions are averaged per group
        (and per item / split when present) and the bootstrap resamples groups. Binary/continuous tasks only.
    multiplier_max_rows: single-item frames larger than this compute the PPI++ multiplier on a fixed-seed
        subsample of this many rows (None: whole table, slow). A warning is raised for large frames.
    Missing truth: rows with no truth are dropped (noted); when the truth column is absent or all missing the
    report is an "awaiting truth" card (seals + prediction summary; no ladder, calibration or bootstrap).
    """
    tmp = None
    ezber_json = None
    if isinstance(memorization, dict):
        fd, tmp = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(memorization, f, ensure_ascii=False)
        ezber_json = tmp
    elif memorization is not None:
        ezber_json = str(memorization)
    if prereg is not None and not isinstance(prereg, list):
        if not chain_paths(prereg):
            warnings.warn(f"prereg registry {str(prereg)!r} has no {_onkayit.ZINCIR_ADI} (evidence.seal / se.seal) "
                          f"or {RESEARCH_CHAIN} (research.seal): no seal is attached", UserWarning, stacklevel=2)
        prereg = chain(prereg)
    d = to_internal(df)
    methods_i = _methods(methods)
    labels_i = {_method(k): v for k, v in (labels or {}).items()}
    tavan_sutun = tavan_etiket = None
    if ceiling_col is not None:
        d, tavan_sutun, tavan_etiket = _ceiling_column(d, ceiling_col, ceiling_label, labels_i, ours, raw)
        if methods_i:
            methods_i = [m for m in methods_i if ON_EK + m != ceiling_col and m != _method(ceiling_col)]
    toplama = None
    if aggregate_by is not None:
        d, toplama = _aggregate(d, aggregate_by, _task(task))
        task = "continuous"
        limits = list(limits or []) + [toplama["note"]]
        level = level or "grup"
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
            carpan_en_fazla=multiplier_max_rows)
    finally:
        if tmp:
            os.unlink(tmp)
    if toplama:
        rep["toplama"] = {k: v for k, v in toplama.items() if k != "note"}
    return rep


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


def _aggregate(d: pd.DataFrame, by: str, task: str | None):
    """Average truth, predictions, retest and model ceiling per group (x item x split when present)."""
    col = COLUMN_TO_INTERNAL.get(by, by)
    if col not in d.columns:
        raise ValueError(f"aggregate_by {by!r} is not a column of the frame")
    if task in ("cok_sinif", "dagilim", "ab"):
        raise ValueError(f"aggregate_by works for binary/continuous tasks, not {TASK_TO_ENGLISH.get(task, task)}")
    keys = [col] + [k for k in ("madde", "bolum") if k in d.columns and k != col]
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
    agg = g[num].mean().reset_index()
    agg.insert(len(keys), "n_satir", g.size().to_numpy())
    # segment columns constant within the group survive (first value)
    segs = [c for c in d.columns if isinstance(c, str) and (c == "segment" or c.startswith("seg_"))]
    for c in segs:
        s = d.groupby(keys, sort=True, dropna=False)[c]
        if (s.nunique(dropna=False) <= 1).all():
            agg[c] = s.first().to_numpy()
    if col != "grup":
        agg["grup"] = agg[col].astype(str)
    n_groups = int(agg[col].nunique())
    note = (f"Grup düzeyi değerlendirme (aggregate_by={by}): {len(d)} satır {n_groups} grubun ortalamasına indirildi "
            f"(eşit satır ağırlığı, {len(agg)} değerlendirme satırı); bootstrap gruplar üzerinden. Grup içi örneklem "
            "gürültüsü ortalamalarda kalır. Otomatik tabanlar (genel ortalama, demografi) grup satırları üzerinde "
            "çapraz uydurmalıdır: kat dışı ortalama tutulan grupların ortalamasıyla ters ilişkili olduğundan sıra "
            "metriklerinde (Spearman) bilgisizken bile sıfırın belirgin altında çıkabilir; anlamlı karşılaştırma için "
            "baselines_oof / unlabeled_baselines ile grup düzeyi bir taban bildirin.")
    return agg, {"by": by, "column": col, "n_rows": int(len(d)), "n_groups": n_groups, "n_eval_rows": int(len(agg)),
                 "auto_baselines_cross_fitted_bias": True, "note": note}


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
             "rows": r.get("satir"), "description": r.get("aciklama")}
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
        "title": report.get("baslik"), "task": TASK_TO_ENGLISH.get(report.get("gorev"), report.get("gorev")),
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
    }
    out["headline_en"] = _headline_en(out)
    return out


# ------------------------------------------------------------------ card (markdown) and dashboard
def _num(x, unit=""):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    if unit == "%":
        return f"{100 * x:.0f}%"
    if unit == "×":
        return f"{x:.2f}×"
    return f"{x:.3f}"


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
    out.append(f"**Badge: {badge}.** " + ("" if en else reason))
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
    if s.get("ceiling_kind"):
        meta += f" · ceiling: {s['ceiling_kind']}"
    if s.get("declared_baselines"):
        meta += " · declared baselines: " + ", ".join(f"`{c}`" for c in s["declared_baselines"])
    out += ["", meta, ""]
    if s.get("headline_en"):
        out += [f"Summary: {s['headline_en']}", ""]
    out += ["| Row | Method | Value [95% CI] | Note |", "|---|---|---|---|"]
    for r in k.get("satirlar") or []:
        u = r.get("birim") or ""
        row = ROW_TO_ENGLISH.get(r['tur'], r['tur'])
        if r["tur"] == "tavan" and s.get("ceiling_kind") == "oracle/model":
            row = "Oracle/model ceiling"
        out.append(f"| {row} | {_md(r.get('etiket'))} | "
                   f"{_num(r.get('deger'), u)}{_ci(r.get('alt'), r.get('ust'), u)} | {_dot(r.get('not'), lang)} |")
    f = k.get("fark")
    if f and f.get("fark") is not None:
        out += ["", f"Ours − strongest baseline (paired, metric direction, positive = better): {f['fark']:+.3f}"
                    + (f" [{f['alt']:+.3f}, {f['ust']:+.3f}]" if f.get("alt") is not None and f.get("ust") is not None else "")]
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
        out += ["", "## Sealed predictions (pre-registration chain)", "", "| # | SHA-256 | Sealed at (UTC) | Rows | Description |",
                "|---|---|---|---|---|"]
        for z in s["seals"]:
            out.append(f"| {z['index']} | `{z['sha256']}` | {_md(z['time_utc'])} | {z['rows']} | {_md(z['description'])} |")
    else:
        out += ["", "Sealed predictions: none attached (pass prereg=<registry dir> to evaluate)."]
    limits = list(k.get("yanilgi") or [])
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
    fam = {m.ad: m for m in _metrik.aile(v.gorev, olas, madde_var and v.gorev in ("ikili", "surekli"), birim)}
    ad = _metric(metric) or ("bir_eksi_mae" if birim else _metrik.BIRINCIL[v.gorev])
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
            "a_better": bool(M.yon * (nokta[ca][ad] - nokta[cb][ad]) > 0), "cluster": v.kume_adi,
            "n_clusters": int(v.kume.max()) + 1}


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
         registry_dir: str | os.PathLike | None = None) -> dict:
    """Seal a prediction table before the truth is revealed: SHA-256 of a canonical CSV + timestamp + hash chain.
    registry (default '.onkayit'; `registry_dir=` is an accepted alias) gets `zincir.jsonl` + a gzip copy."""
    return _onkayit.muhurle(predictions, description, _registry(registry, registry_dir), git_commit)


def verify(predictions, sha256: str | None = None, registry: str | os.PathLike | None = None, *,
           registry_dir: str | os.PathLike | None = None) -> dict:
    """Does the table still match a sealed hash? -> {"eslesti" (matched), "kayit" (record), "sha256"}."""
    return _onkayit.dogrula(predictions, sha256, _registry(registry, registry_dir))


def verify_chain(registry: str | os.PathLike | None = None, *, registry_dir: str | os.PathLike | None = None) -> dict:
    """Recompute every link of the seal chain -> {"gecerli" (valid), "n", "kirik" (broken indices)}."""
    return _onkayit.zincir_dogrula(_registry(registry, registry_dir))


def grade(predictions, truth, keys: list[str], registry: str | os.PathLike | None = None, *,
          registry_dir: str | os.PathLike | None = None, **kw) -> dict:
    """Verify the seal, then join predictions with the truth on `keys` (raises ValueError if not sealed)."""
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
