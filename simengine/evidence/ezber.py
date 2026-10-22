"""Ezber testi (memorization audit): LLM açık veriyi (ve sonucunu) eğitimde görmüş olabilir mi?

Yöntem (tamamlama farkı): her öğe bir kayıttan kurulmuş istem + beklenen cevap (ör. A/B testinin kazanan başlığı,
anketin gerçek yüzdesi). Aynı biçimde kurulmuş KONTROL öğeleri (bozulmuş/karıştırılmış kayıtlar: cevabı bilmek
mümkün değil) şans düzeyini verir. Ezber skoru = gerçek öğelerde isabet − kontrol öğelerinde isabet; öğe düzeyinde
bootstrap %95 GA. Alt sınır > 0 ise "uyari" (ezber şüphesi): o veride simülasyon başarısı kısmen hatırlama olabilir.

Port note (simengine): the original called `claude -p` through subprocess. Here the caller is any
`simengine.types.LLM` (simengine.llm.Claude / ClaudeCode / FakeLLM) wrapped by `llm_caller`, or any
str -> str callable. Caching, language settings and budget belong to the LLM backend. At most 500 calls per run.
The result dict is what `denetim.ezber` (and `evaluate(..., memorization_json=...)`) reads.

`recall_probe(llm, source, groups, measures, truth=...)`: one persona-free "recall <source> results" call as JSON,
in-JSON refusal detection ({"remember": false}, empty tables, all nulls) and one "best estimate" retry; with truth,
per-group MAE / Spearman and an `audit` dict for `evaluate(memorization=...)`.
"""
from __future__ import annotations

import json
import re

import numpy as np

EN_FAZLA = 500


def llm_caller(llm, system: str = "Answer briefly and directly.", max_tokens: int = 64):
    """simengine LLM (protocol: .complete(system, user)) -> str -> str caller for `olc`."""
    def cagir(istem: str) -> str:
        return llm.complete(system, istem, max_tokens=max_tokens).text.strip()
    return cagir


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9ğüşıöç%.]+", " ", str(s).lower()).strip()


def isabet(yanit: str, beklenen, tolerans: float = 0.02, relative: bool = False) -> bool:
    """Sayısal beklenende yanıttaki herhangi bir sayı ± tolerans (yüzde ise 0-100 ölçeği de denenir); metinde içerme.

    relative=True: tolerans beklenenin büyüklüğüne göre (|v − b| ≤ tolerans·|b|; b = 0 ise mutlak tolerans), 0-100 ya
    da herhangi bir sayısal ölçek için. Bu kipte "v/100" yüzde denemesi yalnız |b| ≤ 1 (pay) iken yapılır."""
    if isinstance(beklenen, (int, float, np.integer, np.floating)):
        b = float(beklenen)
        tol = tolerans * abs(b) if relative and b != 0 else tolerans
        yuzde = not relative or abs(b) <= 1
        for m in re.findall(r"-?\d+(?:[.,]\d+)?", str(yanit)):
            v = float(m.replace(",", "."))
            if abs(v - b) <= tol or (yuzde and abs(v / 100 - b) <= tol):
                return True
        return False
    b = _norm(beklenen)
    return bool(b) and b in _norm(yanit)


def olc(ogeler: list[dict], cagir, en_fazla: int = EN_FAZLA, B: int = 1000, tohum: int = 0, *,
        tolerance: float = 0.02, relative: bool = False) -> dict:
    """ogeler: [{"istem", "beklenen", "kontrol": bool}]. Döner: denetim.ezber'in okuduğu sözlük.

    tolerance / relative: sayısal beklenende isabet toleransı (bkz. `isabet`). Varsayılan mutlak ±0,02 (0-1 payları);
    0-100 ya da başka ölçekte relative=True (ör. tolerance=0.05 → beklenenin ±%5'i) ya da ölçeğe uygun mutlak
    tolerance verin. Öğe başına "tolerans" alanı genel değeri ezer."""
    if len(ogeler) > en_fazla:
        raise ValueError(f"{len(ogeler)} öğe > çağrı sınırı {en_fazla}; alt örnek al")
    g, k = [], []
    for o in ogeler:
        tol = o.get("tolerans", o.get("tolerance", tolerance))
        (k if o.get("kontrol") else g).append(float(isabet(cagir(o["istem"]), o["beklenen"], tol, relative)))
    g, k = np.array(g), np.array(k)
    if not len(g) or not len(k):
        return {"durum": "atlandi", "ayrinti": "gerçek ve kontrol öğeleri birlikte gerekli", "n_gercek": len(g), "n_kontrol": len(k)}
    rng = np.random.default_rng(tohum)
    fark = g.mean() - k.mean()
    fb = np.array([rng.choice(g, len(g)).mean() - rng.choice(k, len(k)).mean() for _ in range(B)])
    alt, ust = np.percentile(fb, [2.5, 97.5])
    durum = "uyari" if alt > 0 else "tamam"
    ayrinti = (f"gerçek öğede isabet {g.mean():.3f} (n={len(g)}), kontrolde {k.mean():.3f} (n={len(k)}); "
               f"fark {fark:+.3f} [{alt:+.3f}, {ust:+.3f}]" + (" · ezber şüphesi" if durum == "uyari" else " · ezber işareti yok"))
    return {"durum": durum, "ayrinti": ayrinti, "deger": float(fark), "alt": float(alt), "ust": float(ust),
            "isabet_gercek": float(g.mean()), "isabet_kontrol": float(k.mean()), "n_gercek": len(g), "n_kontrol": len(k),
            "tolerans": tolerance, "goreli": relative}


# ------------------------------------------------------------------ korelasyon denetimi (adlı vs adsız istem)
BEST_GUESS = "If unsure, give your best guess. Reply with one number only."
BEST_GUESS_SYSTEM = ("You are helping with a research forecast. Give your best numeric estimate even when uncertain; "
                     "never refuse or say you cannot recall. Reply with one number only.")
NAMED_TEMPLATE = ("In the study {study}, what was the {quantity} for {key}? " + BEST_GUESS)
BLIND_TEMPLATE = ("What would the {quantity} be for {key}? " + BEST_GUESS)


def audit_prompts(study: str, quantity: str) -> tuple[str, str]:
    """(named, blind) prompt templates with `{key}` left open, e.g.
    audit_prompts("'<study title>' (<authors>, <journal> <year>)", "mean policy support (0-100)")."""
    return (NAMED_TEMPLATE.format(study=study, quantity=quantity, key="{key}"),
            BLIND_TEMPLATE.format(study=study, quantity=quantity, key="{key}"))


def _best_guess(t: str) -> str:
    return t if "best guess" in t.lower() else t.rstrip() + " " + BEST_GUESS


def _sayi_cek(x) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        return float(x)
    m = re.findall(r"-?\d+(?:[.,]\d+)?", str(x))
    return float(m[0].replace(",", ".")) if m else np.nan


def _sozluk(x) -> dict:
    if hasattr(x, "to_dict"):
        x = x.to_dict()
    return dict(x)


def _rank(a):
    from scipy.stats import rankdata
    return rankdata(a)


def _spearman(a, b) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def _llm_hata(e: BaseException) -> bool:
    """simengine.llm LLMError ailesi (EmptyResponseError, RefusalError, ...): o anahtar çözülemedi sayılır."""
    return any(c.__name__ == "LLMError" for c in type(e).__mro__) and \
        not any(c.__name__ == "StopError" for c in type(e).__mro__)


def _thinking_kw(llm, thinking, system, max_tokens) -> dict:
    """thinking="auto": backend `thinking="off"` kabul ediyorsa (Claude Haiku / Sonnet) onu geçir, yoksa hiçbir şey."""
    if thinking is None:
        return {}
    if thinking != "auto":
        return {"thinking": thinking}
    br = getattr(llm, "build_request", None)
    if not callable(br):
        return {}
    try:
        br(system, "x", max_tokens, thinking="off")
    except Exception:  # noqa: BLE001 - model cannot turn thinking off: leave the backend default
        return {}
    return {"thinking": "off"}


def _sor(cevaplar, anahtarlar, llm, system, max_tokens, samples, en_fazla, sayac, thinking="auto", hata=None):
    """cevaplar: şablon (str, `{key}` içerir) → LLM ile sorulur; ya da {anahtar: cevap} sözlüğü/Series.
    Boş cevap / LLMError (EmptyResponseError dahil) o anahtar için NaN olur ve `hata` listesine yazılır."""
    if not isinstance(cevaplar, str):
        return {k: _sayi_cek(v) for k, v in _sozluk(cevaplar).items()}
    if llm is None:
        raise ValueError("a prompt template needs llm= (a simengine LLM or a str -> str callable)")
    sablon = _best_guess(cevaplar)
    istemler = [sablon.format(key=k) for k in anahtarlar]
    sayac[0] += len(istemler) * samples
    if sayac[0] > en_fazla:
        raise ValueError(f"{sayac[0]} calls > max_calls {en_fazla}; use fewer keys or samples")
    hata = hata if hata is not None else []
    tkw = _thinking_kw(llm, thinking, system, max_tokens) if hasattr(llm, "complete") else {}

    def metin(r):
        if isinstance(r, BaseException):
            if not _llm_hata(r):
                raise r
            hata.append(f"{type(r).__name__}: {str(r)[:120]}")
            return None
        return getattr(r, "text", None)

    toplam = {k: [] for k in anahtarlar}
    for s in range(samples):
        if hasattr(llm, "complete"):
            kw = {"sample": s} if samples > 1 else {}
            kw.update(tkw)
            if hasattr(llm, "map"):
                yan = [metin(r) for r in llm.map([(system, u) for u in istemler], max_tokens=max_tokens,
                                                  return_exceptions=True, **kw)]
            else:
                yan = []
                for u in istemler:
                    try:
                        yan.append(llm.complete(system, u, max_tokens=max_tokens, **kw).text)
                    except Exception as e:  # noqa: BLE001 - only LLMError slots are absorbed (metin re-raises)
                        yan.append(metin(e))
        else:
            yan = [llm(u) for u in istemler]
        for k, t in zip(anahtarlar, yan):
            toplam[k].append(_sayi_cek(t))
    return {k: (float(np.nanmean(v)) if np.isfinite(v).any() else np.nan) for k, v in toplam.items()}


def correlation_audit(truth, named, blind, *, llm=None, system: str = BEST_GUESS_SYSTEM, max_tokens: int = 1024,
                      samples: int = 1, B: int = 1000, seed: int = 0, max_calls: int = EN_FAZLA,
                      margin: float = 0.0, thinking: str | dict | None = "auto") -> dict:
    """Memorization audit for numeric aggregates (e.g. country means on a 0-100 scale), where an exact-hit test
    cannot work. The LLM guesses each key's value twice: with the study NAMED (title, authors, journal) and BLIND
    (same quantity, no source). If the named answers track the truth much better than the blind ones, the model
    may remember the published table.

    truth: {key: true value} (dict or Series). named / blind: either the answers ({key: number or reply text}) or
    a prompt template with `{key}` (then `llm=` is called; a "best guess" sentence is appended when the template
    lacks one, so models do not refuse with "I can't recall"). `audit_prompts(study, quantity)` builds both.
    samples: repeated calls per key (`sample=` cache key), averaged. Replies without a number count as unparsed.
    max_tokens (default 1024) and thinking: "auto" (default) passes thinking="off" when the backend accepts it
    (Claude Haiku / Sonnet), so thinking cannot eat the budget of a one-number reply; None leaves the backend
    default, any other value is passed as is. An empty reply (EmptyResponseError) or another LLMError slot makes
    that key unparsed (counted in n_failed, messages in `errors`) instead of aborting the audit.

    Returns the dict `evaluate(memorization=...)` reads: durum ("uyari" when the paired bootstrap CI of
    Spearman(named) − Spearman(blind) lies above `margin`, else "tamam"), ayrinti, deger (the difference), alt, ust,
    plus spearman_named / spearman_blind with CIs, MAE of each arm, n, n_unparsed_named / n_unparsed_blind.
    max_calls: checked BEFORE any call against the planned total (prompt arms x keys x samples; cached replies
    are counted, since the LLM backend has no public cache probe); the error names the number needed.
    Typical use: a few dozen country means, named vs blind prompt, Spearman with the truth."""
    t = {k: _sayi_cek(v) for k, v in _sozluk(truth).items()}
    keys = [k for k, v in t.items() if np.isfinite(v)]
    # çağrı sınırı ÇAĞRIDAN ÖNCE: iki kolun toplamı baştan bilinir (önbellek isabetleri dahil; simengine.llm önbelleği
    # yoklamak için açık bir API vermiyor, bu yüzden isabetler düşülmez)
    planned = sum(len(keys) * samples for arm in (named, blind) if isinstance(arm, str))
    if planned > max_calls:
        n_arms = sum(isinstance(arm, str) for arm in (named, blind))
        raise ValueError(f"correlation_audit needs {planned} calls ({n_arms} prompt arm(s) x {len(keys)} keys x "
                         f"{samples} sample(s)) > max_calls {max_calls}; pass max_calls={planned} (cached replies are "
                         "counted too: a re-run from the cache is free) or use fewer keys / samples. No call was made.")
    sayac = [0]
    hata: list = []
    a_n = _sor(named, keys, llm, system, max_tokens, samples, max_calls, sayac, thinking, hata)
    a_b = _sor(blind, keys, llm, system, max_tokens, samples, max_calls, sayac, thinking, hata)
    un_n = sum(1 for k in keys if not np.isfinite(a_n.get(k, np.nan)))
    un_b = sum(1 for k in keys if not np.isfinite(a_b.get(k, np.nan)))
    ok = [k for k in keys if np.isfinite(a_n.get(k, np.nan)) and np.isfinite(a_b.get(k, np.nan))]
    base = {"yontem": "korelasyon (adlı vs adsız istem)", "n": len(ok), "n_truth": len(keys),
            "n_unparsed_named": un_n, "n_unparsed_blind": un_b, "calls": sayac[0], "n_failed": len(hata),
            "errors": hata[:5]}
    if len(ok) < 5:
        return {**base, "durum": "atlandi", "deger": None, "alt": None, "ust": None,
                "ayrinti": f"korelasyon denetimi: yalnız {len(ok)} anahtarda iki istemin de sayısal cevabı var "
                           f"(adlı {un_n}, adsız {un_b} çözülemedi); en az 5 gerekli"}
    y = np.array([t[k] for k in ok])
    n_ = np.array([a_n[k] for k in ok])
    b_ = np.array([a_b[k] for k in ok])
    rn, rb = _spearman(y, n_), _spearman(y, b_)
    rng = np.random.default_rng(seed)
    bn, bb = np.full(B, np.nan), np.full(B, np.nan)
    for i in range(B):
        ix = rng.integers(0, len(ok), len(ok))
        bn[i], bb[i] = _spearman(y[ix], n_[ix]), _spearman(y[ix], b_[ix])
    fark = bn - bb
    fin = np.isfinite(fark)
    alt, ust = (np.percentile(fark[fin], [2.5, 97.5]) if fin.sum() >= 10 else (np.nan, np.nan))
    ga = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])] if np.isfinite(v).sum() >= 10 else [None, None]
    d = rn - rb if np.isfinite(rn) and np.isfinite(rb) else np.nan
    durum = "uyari" if np.isfinite(alt) and alt > margin else "tamam"
    fmt = lambda x: "—" if not np.isfinite(x) else f"{x:+.3f}"
    ayrinti = (f"korelasyon denetimi ({len(ok)} anahtar): adlı istem Spearman {fmt(rn)}, adsız {fmt(rb)}; fark {fmt(d)} "
               f"[{fmt(alt)}, {fmt(ust)}] (eşleşik bootstrap)" + (" · ezber şüphesi" if durum == "uyari" else " · ezber işareti yok")
               + (f"; çözülemeyen cevap adlı {un_n}, adsız {un_b}" if un_n or un_b else ""))
    _f = lambda x: float(x) if np.isfinite(x) else None
    return {**base, "durum": durum, "ayrinti": ayrinti, "deger": _f(d), "alt": _f(alt), "ust": _f(ust),
            "spearman_named": _f(rn), "spearman_named_ci": ga(bn), "spearman_blind": _f(rb), "spearman_blind_ci": ga(bb),
            "mae_named": float(np.mean(np.abs(n_ - y))), "mae_blind": float(np.mean(np.abs(b_ - y))),
            "answers_named": {str(k): a_n[k] for k in ok}, "answers_blind": {str(k): a_b[k] for k in ok}}


# ------------------------------------------------------------------ tek çağrılık hatırlama sondası
RECALL_SYSTEM = "You are checking what you remember about published research results. Reply with JSON only."
RECALL_TEMPLATE = (
    "Recall the published results of {source} for these groups and measures.\n"
    "Groups: {groups}\nMeasures: {measures}\n{scale}"
    'Reply with JSON only, in this shape: {{"remember": true or false, "results": {{"<group>": {{"<measure>": '
    '<number or null>}}}}}}. Use the group and measure names exactly as given. Set "remember" to false and use null '
    "for values you do not remember.")
RECALL_RETRY_TEMPLATE = (
    "Give your best estimate of the results of {source} for these groups and measures.\n"
    "Groups: {groups}\nMeasures: {measures}\n{scale}"
    "You may not remember the exact figures; that is fine. Put a number in every cell (no nulls) and set "
    '"remember" to false if these are estimates rather than recalled values. Reply with JSON only, in this shape: '
    '{{"remember": true or false, "results": {{"<group>": {{"<measure>": <number>}}}}}}. Use the group and measure '
    "names exactly as given.")
REFUSAL_KINDS = ("error", "empty_reply", "text", "unparsed", "remember_false", "empty", "all_null")
_META = {"remember", "remembers", "recall", "recalls", "recalled", "known", "knows", "note", "notes", "source",
         "confidence", "comment", "caveat"}
_BODY = ("results", "values", "data", "table", "tables", "estimates", "recalled_values")
_FLAG = ("remember", "remembers", "recall", "recalls", "recalled", "known", "knows")
# "I can't recall those figures": a recall-specific decline that the generic refusal detector leaves alone
_NO_RECALL = re.compile(r"\b(?:can ?not|can't|cannot|unable to|not able to|do not|don't|doesn't)\s+(?:\w+\s+){0,2}?"
                        r"(?:recall|remember|know|have (?:access|reliable|specific|exact))|"
                        r"hatırla(?:mıyorum|yamıyorum|yamam)|bilmiyorum|erişimim yok", re.I)


def _esle(ad, adlar: list[str]):
    n = _norm(ad)
    return next((a for a in adlar if _norm(a) == n), None)


def _hucreler(p, groups: list[str], measures: list[str]) -> tuple[dict, int]:
    """Parsed JSON -> ({(group, measure): float or nan}, number of matched cells, null ones included).
    Shapes: {"results": {group: {measure: v}}}, measure-major {measure: {group: v}}, {group: v} (one measure),
    rows [{"group": g, "measure": m, "value": v}] or [{"group": g, <measure>: v, ...}]."""
    out: dict = {}
    if isinstance(p, dict):
        low = {str(k).strip().lower(): k for k in p}
        govde = next((p[low[k]] for k in _BODY if k in low), None)
        if govde is None:
            govde = {k: v for k, v in p.items() if str(k).strip().lower() not in _META}
        p = govde

    def koy(g, m, v):
        if g is not None and m is not None and not isinstance(v, (dict, list)):
            out[(g, m)] = _sayi_cek(v) if not isinstance(v, bool) else np.nan

    if isinstance(p, list):
        for r in p:
            if not isinstance(r, dict):
                continue
            rl = {str(k).strip().lower(): v for k, v in r.items()}
            g = _esle(rl.get("group", rl.get("name", "")), groups)
            if g is None:
                continue
            if "measure" in rl:
                koy(g, _esle(rl["measure"], measures), rl.get("value"))
            else:
                for k, v in r.items():
                    koy(g, _esle(k, measures), v)
    elif isinstance(p, dict):
        for k, v in p.items():
            g, m = _esle(k, groups), _esle(k, measures)
            if g is not None and isinstance(v, dict):
                for mk, mv in v.items():
                    koy(g, _esle(mk, measures), mv)
            elif m is not None and isinstance(v, dict):
                for gk, gv in v.items():
                    koy(_esle(gk, groups), m, gv)
            elif g is not None and len(measures) == 1:
                koy(g, measures[0], v)
    return out, len(out)


def recall_refusal(text: str | None, groups, measures) -> tuple[str | None, dict]:
    """Classify one recall-probe reply: (refusal kind or None, {(group, measure): value}).

    Kinds (`REFUSAL_KINDS`): "empty_reply", "text" (prose decline: `llm.looks_like_refusal` or "I can't recall / don't remember"), "unparsed" (no JSON),
    "remember_false" (the JSON says it does not remember: "remember"/"recall"/"known": false), "empty" (no table
    or empty tables), "all_null" (cells present, none numeric). `looks_like_refusal` returns False for every JSON
    reply, so the last four are only caught here. "remember_false" may still carry numbers (then they are labelled
    estimates, not recall)."""
    from ..llm.refusal import looks_like_refusal
    from ..sim.parse import parse_json
    groups, measures = [str(g) for g in groups], [str(m) for m in measures]
    if not text or not str(text).strip():
        return "empty_reply", {}
    p = parse_json(text)
    if p is None:
        return ("text" if looks_like_refusal(text) or _NO_RECALL.search(str(text)[:400]) else "unparsed"), {}
    cells, n = _hucreler(p, groups, measures)
    n_num = sum(np.isfinite(v) for v in cells.values())
    flag = None
    if isinstance(p, dict):
        low = {str(k).strip().lower(): v for k, v in p.items()}
        flag = next((low[k] for k in _FLAG if k in low), None)
        if isinstance(flag, str):
            flag = {"true": True, "yes": True, "false": False, "no": False}.get(flag.strip().lower(), flag)
    if flag is False:
        return "remember_false", cells
    if n == 0:
        return "empty", cells
    if n_num == 0:
        return "all_null", cells
    return None, cells


def _nested(cells: dict, groups, measures) -> dict:
    return {g: {m: (float(cells[(g, m)]) if (g, m) in cells and np.isfinite(cells[(g, m)]) else None)
                for m in measures} for g in groups}


def _truth_cells(truth, groups, measures) -> dict:
    if hasattr(truth, "columns") and hasattr(truth, "iterrows"):   # DataFrame: index = group, columns = measure
        truth = {g: r.to_dict() for g, r in truth.iterrows()}
    t = _sozluk(truth)
    out = {}
    for gk, gv in t.items():
        g = _esle(gk, groups)
        if g is None:
            continue
        if isinstance(gv, dict) or hasattr(gv, "to_dict"):
            for mk, mv in _sozluk(gv).items():
                m = _esle(mk, measures)
                if m is not None:
                    out[(g, m)] = _sayi_cek(mv)
        elif len(measures) == 1:
            out[(g, measures[0])] = _sayi_cek(gv)
    return out


def recall_probe(llm, source: str, groups, measures, truth=None, best_estimate_retry: bool = True, *,
                 max_tokens: int = 2048, tag: str | None = None, scale: str | None = None,
                 system: str = RECALL_SYSTEM, thinking: str | dict | None = "auto",
                 warn_spearman: float = 0.7) -> dict:
    """One persona-free call: "recall <source> results for these groups / measures" as JSON. Does the model
    say it remembers the published table, and how close are its numbers?

        probe = ev.ezber.recall_probe(llm, "the <survey> <year> wave", ["18-29", "30-44", "45+"],
                                      ["support", "oppose"], truth=truth, scale="percent, 0-100", tag="probe")
        probe["recalls"], probe["refusal"], probe["used"], probe["mae"], probe["spearman"]
        ev.evaluate(df, ours="llm", memorization=probe["audit"])

    llm: a simengine LLM (.complete(system, user)) or a str -> str callable (gets the user prompt only).
    Refusals inside JSON ({"remember": false} with empty tables or all nulls; `looks_like_refusal` misses every
    JSON reply) are detected by `recall_refusal`. With best_estimate_retry=True and no numbers in the first reply,
    one more call asks for a "best estimate" (`RECALL_RETRY_TEMPLATE`, tag "<tag>:retry"); its numbers are
    estimates, never recall. An LLMError (empty reply and the like) counts as refusal kind "error".

    Returns: raw / raw_retry (reply texts), recalls (True only when the FIRST reply gives numbers and does not
    say "remember": false), refusal (first reply's kind or None), retry_refusal, used ("first" / "retry" / None),
    values ({group: {measure: number or None}} from the used reply), n_values, calls. With truth ({group:
    {measure: v}}, {group: v} for one measure, or a DataFrame indexed by group): per_group ({group: {"mae", "n"}}),
    mae, pearson and spearman over all cells, spearman_by_measure (across groups, >= 3), n_compared, and audit:
    the dict `evaluate(memorization=...)` reads. audit durum is "uyari" when the used numbers reach
    Spearman >= warn_spearman with the truth (recalled or "best estimate": either way the model knows the table
    better than it says), "tamam" below, "atlandi" with < 3 comparable cells. One probe has no blind control:
    general knowledge can also rank groups well, so for a controlled test use `correlation_audit` (named vs blind)."""
    groups = [str(g) for g in groups]
    measures = [measures] if isinstance(measures, str) else [str(m) for m in measures]
    fill = {"source": source, "groups": json.dumps(groups, ensure_ascii=False),
            "measures": json.dumps(measures, ensure_ascii=False), "scale": f"Values: {scale}.\n" if scale else ""}
    complete = hasattr(llm, "complete")
    tkw = _thinking_kw(llm, thinking, system, max_tokens) if complete else {}
    hata: list = []

    def sor(user: str, t: str | None) -> str | None:
        if not complete:
            return llm(user)
        kw = dict(tkw)
        if t:
            kw["tag"] = t
        try:
            return llm.complete(system, user, max_tokens=max_tokens, **kw).text
        except Exception as e:  # noqa: BLE001 - only LLMError is absorbed
            if not _llm_hata(e):
                raise
            hata.append(f"{type(e).__name__}: {str(e)[:120]}")
            return None

    raw = sor(RECALL_TEMPLATE.format(**fill), tag)
    calls = 1
    kind, cells = ("error", {}) if raw is None and hata else recall_refusal(raw, groups, measures)
    n_num = sum(np.isfinite(v) for v in cells.values())
    recalls = kind is None and n_num > 0
    used = "first" if n_num else None
    raw2, kind2 = None, None
    if not n_num and best_estimate_retry:
        n_err = len(hata)
        raw2 = sor(RECALL_RETRY_TEMPLATE.format(**fill), f"{tag}:retry" if tag else None)
        calls += 1
        kind2, cells2 = ("error", {}) if raw2 is None and len(hata) > n_err else recall_refusal(raw2, groups, measures)
        if sum(np.isfinite(v) for v in cells2.values()):
            cells, used = cells2, "retry"
    values = _nested(cells, groups, measures)
    out = {"source": source, "raw": raw, "raw_retry": raw2, "recalls": bool(recalls), "refusal": kind,
           "retry_refusal": kind2, "used": used, "values": values,
           "n_values": int(sum(v is not None for r in values.values() for v in r.values())),
           "n_cells": len(groups) * len(measures), "calls": calls, "errors": hata[:5]}
    if truth is None:
        return out
    t = _truth_cells(truth, groups, measures)
    ok = [c for c in t if np.isfinite(t[c]) and c in cells and np.isfinite(cells[c])]
    _f = lambda x: float(x) if x is not None and np.isfinite(x) else None
    per_group = {}
    for g in groups:
        d = [abs(cells[c] - t[c]) for c in ok if c[0] == g]
        per_group[g] = {"mae": float(np.mean(d)) if d else None, "n": len(d)}
    y = np.array([t[c] for c in ok])
    x = np.array([cells[c] for c in ok])
    pear = float(np.corrcoef(x, y)[0, 1]) if len(ok) >= 3 and np.std(x) > 0 and np.std(y) > 0 else np.nan
    rho = _spearman(y, x) if len(ok) >= 3 else np.nan
    by_m = {}
    for m in measures:
        cm = [c for c in ok if c[1] == m]
        by_m[m] = _f(_spearman(np.array([t[c] for c in cm]), np.array([cells[c] for c in cm]))) if len(cm) >= 3 else None
    mae = float(np.mean(np.abs(x - y))) if len(ok) else None
    kol = {"first": "ilk cevap (hatırlıyor)" if recalls else "ilk cevap (hatırlamıyor dedi, sayı verdi)",
           "retry": f"ilk cevap ret ({kind}), 'best estimate' tekrarı", None: "sayı yok"}[used]
    fmt = lambda v: "—" if v is None or not np.isfinite(v) else f"{v:+.3f}"
    if len(ok) < 3:
        durum, ayrinti = "atlandi", (f"hatırlama sondası: yalnız {len(ok)} hücrede gerçek ve cevap birlikte var "
                                     f"(en az 3 gerekli); {kol}")
    else:
        durum = "uyari" if np.isfinite(rho) and rho >= warn_spearman else "tamam"
        ayrinti = (f"hatırlama sondası ({len(ok)} hücre, {kol}): Spearman {fmt(rho)}, MAE {mae:.3f}; eşik "
                   f"{warn_spearman:.2f}" + (" · ezber şüphesi" if durum == "uyari" else " · ezber işareti yok")
                   + " (tek sonda, adsız kontrol yok)")
    audit = {"yontem": "tek çağrılık hatırlama sondası", "durum": durum, "ayrinti": ayrinti, "deger": _f(rho),
             "alt": None, "ust": None, "n": len(ok), "recalls": bool(recalls), "refusal": kind, "used": used,
             "mae": mae, "esik": warn_spearman}
    out.update({"per_group": per_group, "mae": mae, "pearson": _f(pear), "spearman": _f(rho),
                "spearman_by_measure": by_m, "n_compared": len(ok), "audit": audit})
    return out
