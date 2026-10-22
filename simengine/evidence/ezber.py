"""Ezber testi (memorization audit): LLM açık veriyi (ve sonucunu) eğitimde görmüş olabilir mi?

Yöntem (tamamlama farkı): her öğe bir kayıttan kurulmuş istem + beklenen cevap (ör. A/B testinin kazanan başlığı,
anketin gerçek yüzdesi). Aynı biçimde kurulmuş KONTROL öğeleri (bozulmuş/karıştırılmış kayıtlar: cevabı bilmek
mümkün değil) şans düzeyini verir. Ezber skoru = gerçek öğelerde isabet − kontrol öğelerinde isabet; öğe düzeyinde
bootstrap %95 GA. Alt sınır > 0 ise "uyari" (ezber şüphesi): o veride simülasyon başarısı kısmen hatırlama olabilir.

Port note (simengine): the original called `claude -p` through subprocess. Here the caller is any
`simengine.types.LLM` (simengine.llm.Claude / ClaudeCode / FakeLLM) wrapped by `llm_caller`, or any
str -> str callable. Caching, language settings and budget belong to the LLM backend. At most 500 calls per run.
The result dict is what `denetim.ezber` (and `evaluate(..., memorization_json=...)`) reads.
"""
from __future__ import annotations

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
