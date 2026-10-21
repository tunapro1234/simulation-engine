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
import tempfile
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
from .sema import ON_EK, hazirla

VERSION = SURUM

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
STATUS_TO_ENGLISH = {"tamam": "ok", "uyari": "warning", "hata": "error", "atlandi": "skipped"}
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
             unlabeled_baselines: Sequence[str] | None = None) -> dict:
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
        prereg = _onkayit._zincir_oku(prereg)
    try:
        return degerlendir(
            to_internal(df), title, dataset, gorev=_task(task), yontemler=_methods(methods), bizim=_method(ours),
            ham=_method(raw), etiketler={_method(k): v for k, v in (labels or {}).items()}, birincil=_metric(primary),
            B=B, B_toplu=B_aggregate, tohum=seed, kalibre=_methods(calibrate), kesim=cutoff, ezber_json=ezber_json,
            carpan=multiplier, sinirlar=list(limits) if limits else None, onkayit=prereg, siniflar=classes,
            baseline=baselines, carpan_yontemler=_methods(multiplier_methods), dongusel=_methods(circular),
            duzey=level, girdi_kaynagi=source, etiketsiz_tabanlar=_methods(unlabeled_baselines))
    finally:
        if tmp:
            os.unlink(tmp)


def save_json(report: dict, path) -> Path:
    """Write the report as JSON (NaN already mapped to null)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return p


def _box(d):
    return None if not d else {"column": d.get("sutun"), "label": d.get("etiket"), "value": d.get("deger"),
                               "lo": d.get("alt"), "hi": d.get("ust")}


def summary(report: dict) -> dict:
    """English flat view of the headline numbers of a report (no new computation)."""
    m = report.get("manset") or {}
    k = report.get("kanit_kart") or {}
    worth = next((r for r in k.get("satirlar") or [] if r.get("tur") == "bedel"), None)
    return {
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
        "badge": BADGE_TO_ENGLISH.get(k.get("rozet"), k.get("rozet")), "badge_reason": k.get("rozet_gerekce"),
        "flags": k.get("etiketler") or [],
        "audits": [{"check": d.get("kontrol"), "status": STATUS_TO_ENGLISH.get(d.get("durum"), d.get("durum")),
                    "detail": d.get("ayrinti"), "value": d.get("deger")} for d in report.get("denetim") or []],
        "ladder": [{"column": r.get("sutun"), "label": r.get("etiket"), "kind": r.get("tur"), "value": r.get("deger"),
                    "lo": r.get("alt"), "hi": r.get("ust"), "diff_prev": r.get("fark_onceki"), "p": r.get("p")}
                   for r in report.get("merdiven") or []],
        "headline": m.get("cumle"),
    }


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


def card_markdown(report: dict) -> str:
    """Evidence card as Markdown: badge, the six card rows, the ladder and the audits."""
    k = report.get("kanit_kart") or _kart.kart(report)
    s = summary(report)
    out = [f"# Evidence card: {_md(report.get('baslik') or report.get('veri_adi') or '')}", ""]
    out.append(f"**Badge: {BADGE_TO_ENGLISH.get(k.get('rozet'), k.get('rozet'))}.** {_md(k.get('rozet_gerekce'))}")
    if k.get("etiketler"):
        out.append("")
        out.append("Flags: " + ", ".join(_md(e) for e in k["etiketler"]))
    out += ["", f"Task: {s['task']} · primary metric: {s['primary_metric']} (`{s['primary_metric_internal']}`) · "
                f"clusters: {k.get('n')} · source: {_md(k.get('kaynak'))}", ""]
    out += ["| Row | Method | Value [95% CI] | Note |", "|---|---|---|---|"]
    for r in k.get("satirlar") or []:
        u = r.get("birim") or ""
        out.append(f"| {ROW_TO_ENGLISH.get(r['tur'], r['tur'])} | {_md(r.get('etiket'))} | "
                   f"{_num(r.get('deger'), u)}{_ci(r.get('alt'), r.get('ust'), u)} | {_md(r.get('not'))} |")
    f = k.get("fark")
    if f and f.get("fark") is not None:
        out += ["", f"Ours − strongest baseline (paired, metric direction): {f['fark']:+.3f}"
                    + (f" [{f['alt']:+.3f}, {f['ust']:+.3f}]" if f.get("alt") is not None and f.get("ust") is not None else "")]
    if s["ladder"]:
        out += ["", "## Baseline ladder", "", "| Rung | Kind | Value [95% CI] | Δ vs previous [95% CI] | p |",
                "|---|---|---|---|---|"]
        for r in report.get("merdiven") or []:
            d = "—" if r.get("fark_onceki") is None else f"{r['fark_onceki']:+.3f}{_ci(r.get('fark_alt'), r.get('fark_ust'))}"
            p = "—" if r.get("p") is None else f"{r['p']:.3f}"
            out.append(f"| {_md(r.get('etiket'))} (`{r.get('sutun')}`) | {r.get('tur')} | "
                       f"{_num(r.get('deger'))}{_ci(r.get('alt'), r.get('ust'))} | {d} | {p} |")
    if s["audits"]:
        out += ["", "## Audits", "", "| Check | Status | Detail |", "|---|---|---|"]
        for a in s["audits"]:
            out.append(f"| {_md(a['check'])} | {a['status']} | {_md(a['detail'])} |")
    if k.get("yanilgi"):
        out += ["", "## Limits", ""] + [f"- {_md(x)}" for x in k["yanilgi"]]
    if s["headline"]:
        out += ["", "Headline (original wording): " + _md(s["headline"])]
    return "\n".join(out) + "\n"


def card(report: dict, path=None) -> str | dict:
    """Evidence card. path *.json -> the card dict (stage schema, `kanit_kart`); otherwise Markdown.
    Returns the content; writes it when `path` is given."""
    if path is not None and str(path).endswith(".json"):
        k = report.get("kanit_kart") or _kart.kart(report)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(k, ensure_ascii=False, indent=1))
        return k
    md = card_markdown(report)
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
                       seed: int = 0) -> dict:
    """Completion-gap memorization test: hit rate on real items − hit rate on control (scrambled) items, item
    bootstrap 95 % CI; lower bound > 0 -> 'uyari' (memorization suspected).
    items: [{"prompt"|"istem", "expected"|"beklenen", "control"|"kontrol": bool}].
    llm_or_caller: a simengine LLM (e.g. FakeLLM) or a str -> str callable."""
    og = [{"istem": o.get("istem", o.get("prompt")), "beklenen": o.get("beklenen", o.get("expected")),
           "kontrol": bool(o.get("kontrol", o.get("control", False)))} for o in items]
    cagir = _ezber.llm_caller(llm_or_caller) if hasattr(llm_or_caller, "complete") else llm_or_caller
    return _ezber.olc(og, cagir, en_fazla=max_calls, B=B, tohum=seed)


answer_hit = _ezber.isabet

# ------------------------------------------------------------------ pre-registration (kanit.onkayit)
canonical_bytes = _onkayit.kanonik_bayt


def seal(predictions, description: str, registry: str | os.PathLike = ".onkayit", git_commit: bool = False) -> dict:
    """Seal a prediction table before the truth is revealed: SHA-256 of a canonical CSV + timestamp + hash chain."""
    return _onkayit.muhurle(predictions, description, registry, git_commit)


def verify(predictions, sha256: str | None = None, registry: str | os.PathLike = ".onkayit") -> dict:
    """Does the table still match a sealed hash? -> {"eslesti" (matched), "kayit" (record), "sha256"}."""
    return _onkayit.dogrula(predictions, sha256, registry)


def verify_chain(registry: str | os.PathLike = ".onkayit") -> dict:
    """Recompute every link of the seal chain -> {"gecerli" (valid), "n", "kirik" (broken indices)}."""
    return _onkayit.zincir_dogrula(registry)


def grade(predictions, truth, keys: list[str], registry: str | os.PathLike = ".onkayit", **kw) -> dict:
    """Verify the seal, then join predictions with the truth on `keys` (raises ValueError if not sealed)."""
    return _onkayit.notla(predictions, truth, keys, registry, **kw)


def chain(registry: str | os.PathLike = ".onkayit") -> list[dict]:
    """Seal records of a registry (pass to evaluate(prereg=...))."""
    return _onkayit._zincir_oku(registry)


# ------------------------------------------------------------------ card pieces
badge = _kart.rozet
card_dict = _kart.kart
panel_cost = _kart.dolar
