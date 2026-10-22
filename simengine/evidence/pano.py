"""Kanıt panosu: bir ya da birden çok `ozet` sözlüğü → tek dosya HTML.

Saf standart kütüphane. Harici script/CSS/font yok; grafikler Python'da üretilen satır içi SVG.
Şema: ../OZET-SEMA.md. Eksik alanlar sessizce atlanır.

    from simengine.evidence import pano
    pano.yaz([ozet1, ozet2], "pano.html", baslik="Kanıt panosu")
"""
from __future__ import annotations

import html as _html
import json
import math
from pathlib import Path

GOREV_ADI = {"ikili": "ikili sınıflama", "cok_sinif": "çok sınıflı", "dagilim": "dağılım",
             "surekli": "sürekli", "ab": "A/B testi"}
METRIK_ADI = {"bir_eksi_mae": "1 − MAE", "bir_eksi_tvd": "1 − TVD", "dogruluk": "doğruluk", "auc": "AUC",
              "macro_f1": "makro-F1", "brier": "Brier", "logloss": "log kayıp", "ece": "ECE", "wasserstein": "Wasserstein",
              "js": "JS ıraksaması", "ks": "KS", "mod_eslesme": "mod eşleşmesi", "pearson": "Pearson r",
              "spearman": "Spearman ρ", "mae": "MAE", "rmse": "RMSE", "r2": "R²", "kazanan_top1": "kazanan (top-1)",
              "cift_isaret": "çift işaret doğruluğu", "spearman_ici": "test içi Spearman",
              "etki_korelasyonu": "etki korelasyonu"}
TUR_ADI = {"baseline": "baseline", "yontem": "yöntem", "tavan": "tavan", "kalibre": "kalibre", "kontrol": "kontrol"}
DURUM = {"tamam": ("✓", "tamam"), "uyari": ("!", "uyarı"), "hata": ("✗", "hata"), "atlandi": ("–", "atlandı")}
SERI = 4  # en fazla bu kadar seri tek grafikte (kategorik renk sırası sabit)


# ------------------------------------------------------------------ biçim yardımcıları
def _e(x) -> str:
    return _html.escape("" if x is None else str(x), quote=True)


def _bos(x) -> bool:
    if x is None or isinstance(x, bool):
        return x is None
    try:
        return not math.isfinite(float(x))
    except (TypeError, ValueError):
        return True


def sayi(x, basamak: int | None = None) -> str:
    """0.7551 → '0,755'; None/NaN → '—'. Küçük değerlerde bir basamak fazla."""
    if _bos(x):
        return "—"
    x = float(x)
    if basamak is None:
        basamak = 1 if abs(x) >= 100 else (4 if 0 < abs(x) < 0.01 else 3)
    m = f"{x:,.{basamak}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return m.replace("-", "−")


def tam(x) -> str:
    """193464 → '193.464'."""
    if _bos(x):
        return "—"
    return f"{int(round(float(x))):,}".replace(",", ".")


def yuzde(x, basamak: int = 0) -> str:
    """0.94 → '%94'."""
    if _bos(x):
        return "—"
    return "%" + sayi(100 * float(x), basamak)


def ga(alt, ust, basamak: int | None = None, yz: bool = False) -> str:
    if _bos(alt) or _bos(ust):
        return ""
    f = (lambda v: sayi(100 * float(v), 0)) if yz else (lambda v: sayi(v, basamak))
    return f"[{f(alt)}–{f(ust)}]"


def p_bicim(p) -> str:
    if _bos(p):
        return "—"
    return "p<0,001" if p < 0.001 else "p=" + sayi(p, 3)


def _deger_ga(d: dict | None, yz: bool = False) -> str:
    if not d:
        return "—"
    v = yuzde(d.get("deger")) if yz else sayi(d.get("deger"))
    g = ga(d.get("alt"), d.get("ust"), yz=yz)
    return f'{_e(v)} <span class="ga">{_e(g)}</span>' if g else _e(v)


def _metrik_adi(ad: str) -> str:
    return METRIK_ADI.get(ad, ad)


def _rozet(tur: str) -> str:
    return f'<span class="rozet r-{_e(tur)}">{_e(TUR_ADI.get(tur, tur))}</span>'


def _tablo(basliklar: list[str], satirlar: list[list[str]], sinif: str = "") -> str:
    """Hücreler hazır HTML. Geniş tablo kendi kutusunda kayar."""
    th = "".join(f"<th>{h}</th>" for h in basliklar)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in s) + "</tr>" for s in satirlar)
    return f'<div class="tkutu"><table class="{sinif}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


# ------------------------------------------------------------------ SVG yardımcıları
def _adim(aralik: float) -> float:
    if aralik <= 0:
        return 0.1
    k = 10 ** math.floor(math.log10(aralik / 5))
    for c in (1, 2, 2.5, 5, 10):
        if aralik / (c * k) <= 6:
            return c * k
    return 10 * k


def _olcek(degerler, pay: float = 0.05):
    d = [float(v) for v in degerler if not _bos(v)]
    if not d:
        return 0.0, 1.0
    lo, hi = min(d), max(d)
    if hi - lo < 1e-12:
        lo, hi = lo - 0.05, hi + 0.05
    p = (hi - lo) * pay
    lo, hi = lo - p, hi + p
    s = _adim(hi - lo)
    return math.floor(lo / s) * s, math.ceil(hi / s) * s


class _Eksen:
    """Doğrusal x ekseni; ters=True ise küçük değer sağda (yön −1: sağ hep 'daha iyi')."""

    def __init__(self, lo, hi, x0, x1, ters=False):
        self.lo, self.hi, self.x0, self.x1, self.ters = lo, hi, x0, x1, ters

    def __call__(self, v) -> float:
        t = (float(v) - self.lo) / (self.hi - self.lo)
        if self.ters:
            t = 1 - t
        return self.x0 + t * (self.x1 - self.x0)

    def izler(self, y0, y1) -> str:
        out = []
        for v, et in _izler(self.lo, self.hi):
            x = self(v)
            out.append(f'<line class="izgara" x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y1}"/>'
                       f'<text class="eksen" x="{x:.1f}" y="{y1 + 14}" text-anchor="middle">{_e(et)}</text>')
        return "".join(out)


def _izler(lo, hi, bicim=None) -> list[tuple[float, str]]:
    """Adımın katlarına oturan izler + etiketleri (ondalık sayısı adımdan)."""
    s = _adim(hi - lo)
    b = 0
    while b < 6 and abs(s * 10 ** b - round(s * 10 ** b)) > 1e-6:
        b += 1
    out, k = [], math.ceil(lo / s - 1e-9)
    while k * s <= hi + 1e-9:
        v = round(k * s, 10)
        out.append((v, bicim(v) if bicim else sayi(v, b)))
        k += 1
    return out


def _ayir(uclar, ara: float = 13) -> list:
    """Uç etiketlerini dikeyde en az `ara` piksel aralıkla it (sıra korunur)."""
    uclar = sorted(uclar, key=lambda u: u[0])
    for k in range(1, len(uclar)):
        uclar[k][0] = max(uclar[k][0], uclar[k - 1][0] + ara)
    return uclar


def _svg(gen: int, yuk: int, ic: str, etiket: str) -> str:
    return (f'<svg viewBox="0 0 {gen} {yuk}" role="img" aria-label="{_e(etiket)}" '
            f'preserveAspectRatio="xMidYMid meet">{ic}</svg>')


def _kisalt(s: str, n: int = 26) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _nokta_ga(x, y, xa, xu, sinif, baslik, r=4.5) -> str:
    """GA bıyığı + nokta; <title> yerel ipucu."""
    b = ""
    if xa is not None and xu is not None:
        b = (f'<line class="biyik {sinif}" x1="{xa:.1f}" x2="{xu:.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
             f'<line class="biyik {sinif}" x1="{xa:.1f}" x2="{xa:.1f}" y1="{y - 4:.1f}" y2="{y + 4:.1f}"/>'
             f'<line class="biyik {sinif}" x1="{xu:.1f}" x2="{xu:.1f}" y1="{y - 4:.1f}" y2="{y + 4:.1f}"/>')
    return f'<g><title>{_e(baslik)}</title>{b}<circle class="nokta {sinif}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/></g>'


def _lejant(ogeler: list[tuple[str, str]]) -> str:
    return '<div class="lejant">' + "".join(
        f'<span><i class="lj {s}"></i>{_e(a)}</span>' for a, s in ogeler) + "</div>"


# ------------------------------------------------------------------ özet yardımcıları
class _Ozet:
    def __init__(self, oz: dict, i: int):
        self.oz, self.i = oz, i
        self.sonuclar = oz.get("sonuclar") or {}
        self.yon = {m["ad"]: m.get("yon", 1) for m in oz.get("metrikler") or [] if "ad" in m}
        self.acik = {m["ad"]: m.get("aciklama", "") for m in oz.get("metrikler") or [] if "ad" in m}
        self.bizim = ((oz.get("manset") or {}).get("bizim") or {}).get("sutun")
        self.ham = ((oz.get("manset") or {}).get("ham") or {}).get("sutun")
        # renk varlığı izler, sırayı değil: bizim, ham, diğer yöntemler, kalibreler, sonra baseline
        sira = sorted(self.sonuclar, key=self._oncelik)
        self.renk = {s: f"s{k % 8 + 1}" for k, s in enumerate(sira)}

    def tur(self, s):
        return (self.sonuclar.get(s) or {}).get("tur", "")

    def etiket(self, s):
        return (self.sonuclar.get(s) or {}).get("etiket") or s

    def _oncelik(self, s):
        if s == self.bizim:
            return (-2, 0)
        if s == self.ham:
            return (-1, 0)
        return ({"yontem": 0, "kalibre": 1, "baseline": 2}.get(self.tur(s), 3),
                list(self.sonuclar).index(s) if s in self.sonuclar else 99)

    def secili(self, sutunlar, n=SERI):
        return sorted(sutunlar, key=self._oncelik)[:n]


def _bolum(baslik: str, ic: str, alt: str = "") -> str:
    a = f'<p class="alt">{alt}</p>' if alt else ""
    return f'<section class="bolum"><h3>{_e(baslik)}</h3>{a}{ic}</section>'


# ------------------------------------------------------------------ 1. kart
def _kart(o: _Ozet) -> str:
    oz, k = o.oz, o.oz.get("kart") or {}
    alanlar = [("Görev", GOREV_ADI.get(oz.get("gorev"), oz.get("gorev"))),
               ("Birincil metrik", _metrik_adi(oz.get("birincil", "")) if oz.get("birincil") else None),
               ("Satır", tam(k.get("n_satir")) if "n_satir" in k else None),
               (f"Küme ({k.get('kume_adi', 'küme')})", tam(k.get("n_kume")) if "n_kume" in k else None),
               ("Madde", tam(k.get("n_madde")) if "n_madde" in k else None),
               ("Bootstrap B", tam(k.get("B")) if "B" in k else None),
               ("Tohum", k.get("tohum")),
               ("Segmentler", ", ".join(k.get("segmentler") or []) or None),
               ("Düzey", oz.get("duzey")), ("Tutma birimi", oz.get("tutma_birimi")),
               ("Girdi kaynağı", oz.get("girdi_kaynagi")),
               ("Oluşturma", oz.get("olusturma")), ("Sürüm", oz.get("surum"))]
    dl = "".join(f"<div><dt>{_e(a)}</dt><dd>{_e(v)}</dd></div>" for a, v in alanlar if v not in (None, ""))
    notlar = "".join(f"<li>{_e(n)}</li>" for n in k.get("notlar") or [])
    return f'<dl class="kart">{dl}</dl>' + (f'<ul class="notlar">{notlar}</ul>' if notlar else "")


# ------------------------------------------------------------------ 2. manşet
def _manset(o: _Ozet) -> str:
    m = o.oz.get("manset")
    if not m:
        return ""
    kutular = []
    d = m.get("bosluk_kapanan")
    if d:
        kutular.append(f'<div class="kutu k-norm"><div class="kad">Kapanan boşluk</div><div class="kdeger">{_e(yuzde(d.get("deger")))}</div>'
                       f'<div class="ga">{_e(ga(d.get("alt"), d.get("ust"), yz=True))}</div>'
                       f'<div class="kalt">(bizim − en güçlü taban) ÷ (tavan − en güçlü taban)</div></div>')
    elif not m.get("tavan"):
        kutular.append('<div class="kutu k-norm"><div class="kad">Kapanan boşluk</div><div class="kdeger">—</div>'
                       '<div class="kalt">tavan yok: test-retest ölçülmemiş</div></div>')
    for anahtar, ad, sinif in (("en_iyi_baseline", "En güçlü taban", "k-base"), ("en_iyi_rakip", "En güçlü diğer yöntem", "k-ham"),
                               ("bizim", "Bizim yöntem", "k-bizim"),
                               ("tavan", {"model": "Model/oracle tavanı", "orneklem_gurultusu": "Örneklem gürültüsü tavanı"}
                                .get(o.oz.get("tavan_turu"), "İnsan tavanı"), "k-tavan")):
        d = m.get(anahtar)
        if not d:
            continue
        alt = _e(o.etiket(d["sutun"])) if d.get("sutun") else ("test-retest" if anahtar == "tavan" else "")
        if anahtar == "tavan" and o.oz.get("tavan_turu") == "orneklem_gurultusu":
            alt = "tavan türü: örneklem gürültüsü (varyans bileşenleri)"
        kutular.append(f'<div class="kutu {sinif}"><div class="kad">{ad}</div><div class="kdeger">{_e(sayi(d.get("deger")))}</div>'
                       f'<div class="ga">{_e(ga(d.get("alt"), d.get("ust")))}</div><div class="kalt">{alt}</div></div>')
    d = m.get("tavana_oran")
    if d:
        tt = m.get("taban_tavana_oran")
        ek = f" · taban tek başına {yuzde(tt)}" if not _bos(tt) else ""
        kutular.append(f'<div class="kutu k-oran"><div class="kad">Tavana oran</div><div class="kdeger">{_e(yuzde(d.get("deger")))}</div>'
                       f'<div class="ga">{_e(ga(d.get("alt"), d.get("ust"), yz=True))}</div>'
                       f'<div class="kalt">bizim ÷ insan tavanı{_e(ek)}; tek başına okunmaz</div></div>')
    cumle = f'<p class="cumle">{_e(m.get("cumle"))}</p>' if m.get("cumle") else ""
    if m.get("cumle_tabanlar"):
        cumle += f'<p class="alt">{_e(m["cumle_tabanlar"])}</p>'
    birincil = o.oz.get("birincil")
    not_ = (f'<p class="alt">Birincil metrik: <b>{_e(_metrik_adi(birincil))}</b>'
            f'{" (küçük iyi)" if o.yon.get(birincil) == -1 else ""} · köşeli parantez %95 bootstrap güven aralığı.</p>') if birincil else ""
    return f'<div class="manset">{cumle}<div class="kutular">{"".join(kutular)}</div>{not_}</div>'


# ------------------------------------------------------------------ 2b. etkin örneklem çarpanı
KESTIRICI = (("ppi", "PPI", ""), ("bayes", "Bayes", "kesik"))  # çizgi deseni de kestiriciyi ayırır


def _log_izler(lo: float, hi: float) -> list[float]:
    """1-2-5 log izleri; çok geniş aralıkta yalnız 10'un katları."""
    out, k = [], math.floor(math.log10(lo))
    carp = (1, 2, 5) if math.log10(hi / lo) <= 3 else (1,)
    while 10 ** k <= hi * 1.0001:
        out += [c * 10 ** k for c in carp if lo * 0.9999 <= c * 10 ** k <= hi * 1.0001]
        k += 1
    return out


def _orneklem_seriler(o: _Ozet):
    """[(sutun, etiket, kestirici_anahtar, kestirici_adi, desen, renk, noktalar)] — en fazla 3 yöntem çizilir."""
    yt = (o.oz.get("orneklem") or {}).get("yontemler") or {}
    yedek = iter(f"s{k}" for k in range(1, 9) if f"s{k}" not in {o.renk.get(s) for s in yt})
    out = []
    for s in o.secili(list(yt), 3):
        renk = o.renk.get(s) or next(yedek, "s1")
        et = yt[s].get("etiket") or o.etiket(s)
        for k, ad, desen in KESTIRICI:
            ps = sorted((p for p in yt[s].get(k) or [] if not _bos(p.get("n")) and not _bos(p.get("n_esdeger"))
                         and p["n"] > 0 and p["n_esdeger"] > 0), key=lambda p: p["n"])
            if ps:
                out.append((s, et, k, ad, desen, renk, ps))
    return out


def _orneklem(o: _Ozet) -> str:
    orn = o.oz.get("orneklem") or {}
    yt = orn.get("yontemler") or {}
    if not yt:
        return ""
    seriler = _orneklem_seriler(o)
    ic, lej = [], []
    if seriler:
        def esd_sinir(p, a):  # GA: n × çarpan sınırı
            c = p.get(a)
            return None if _bos(c) or c <= 0 else p["n"] * c
        xs = [p["n"] for *_, ps in seriler for p in ps]
        ys = [v for *_, ps in seriler for p in ps for v in (p["n_esdeger"], esd_sinir(p, "alt"), esd_sinir(p, "ust")) if v]
        xlo, xhi = min(xs) / 1.25, max(xs) * 1.25
        ylo, yhi = min(ys + [xlo]) / 1.25, max(ys + [xhi]) * 1.25
        G, Y, sol, sag, ust, alt = 600, 300, 52, 130, 10, 34
        lx = lambda v: sol + (math.log10(v) - math.log10(xlo)) / (math.log10(xhi) - math.log10(xlo)) * (G - sol - sag)
        ly = lambda v: ust + (1 - (math.log10(v) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo))) * (Y - ust - alt)
        for v in _log_izler(xlo, xhi):
            ic.append(f'<line class="izgara" x1="{lx(v):.1f}" x2="{lx(v):.1f}" y1="{ust}" y2="{Y - alt}"/>'
                      f'<text class="eksen" x="{lx(v):.1f}" y="{Y - alt + 14}" text-anchor="middle">{_e(tam(v))}</text>')
        for v in _log_izler(ylo, yhi):
            ic.append(f'<line class="izgara" x1="{sol}" x2="{G - sag}" y1="{ly(v):.1f}" y2="{ly(v):.1f}"/>'
                      f'<text class="eksen" x="{sol - 6}" y="{ly(v) + 4:.1f}" text-anchor="end">{_e(tam(v))}</text>')
        # y = x referansı: simülasyonsuz
        a, b = max(xlo, ylo), min(xhi, yhi)
        if a < b:
            ic.append(f'<line class="tavan" x1="{lx(a):.1f}" y1="{ly(a):.1f}" x2="{lx(b):.1f}" y2="{ly(b):.1f}"/>'
                      f'<text class="eksen" x="{lx(m := math.sqrt(a * b)) + 8:.1f}" y="{ly(m) + 16:.1f}">simülasyonsuz (y = x)</text>')
        uclar = []
        for s, et, k, ad, desen, renk, ps in seriler:
            ic.append(f'<polyline class="cizgi {renk} {desen}" points="{" ".join("%.1f,%.1f" % (lx(p["n"]), ly(p["n_esdeger"])) for p in ps)}"/>')
            for p in ps:
                ea, eu = esd_sinir(p, "alt"), esd_sinir(p, "ust")
                x, y = lx(p["n"]), ly(p["n_esdeger"])
                b = (f'<line class="biyik" x1="{x:.1f}" x2="{x:.1f}" y1="{ly(eu):.1f}" y2="{ly(ea):.1f}"/>') if ea and eu else ""
                ic.append(f'<g><title>{_e(et)} · {ad}: {tam(p["n"])} insan + simülasyon ≈ {tam(p["n_esdeger"])} insan; '
                          f'çarpan {sayi(p.get("carpan"), 1)}× {ga(p.get("alt"), p.get("ust"), 1)}</title>{b}'
                          f'<circle class="nokta {renk}" cx="{x:.1f}" cy="{y:.1f}" r="{4 if k == "ppi" else 3.5}"/></g>')
            son = ps[-1]
            uclar.append([ly(son["n_esdeger"]) + 4, lx(son["n"]) + 8, f"{_kisalt(et, 10)} · {ad}"])
            lej.append((f"{et} · {ad}", f"{renk} {desen}"))
        ic += [f'<text class="etk" x="{x:.1f}" y="{y:.1f}">{_e(t)}</text>' for y, x, t in _ayir(uclar)]
        ic.append(f'<text class="eksen" x="{(sol + G - sag) / 2}" y="{Y - 2}" text-anchor="middle">gerçek insan sayısı n (log)</text>'
                  f'<text class="eksen" x="12" y="{(ust + Y - alt) / 2}" transform="rotate(-90 12 {(ust + Y - alt) / 2})" '
                  f'text-anchor="middle">eşdeğer insan sayısı (log)</text>')
    svg = _svg(600, 300, "".join(ic), "Etkin örneklem: gerçek insan sayısına karşı eşdeğer insan sayısı") if ic else ""
    # tablo: satır = n, sütun = yöntem × kestirici
    sutunlar = [(s, k, ad, yt[s].get("etiket") or o.etiket(s)) for s in yt for k, ad, _ in KESTIRICI if yt[s].get(k)]
    nler = sorted({p["n"] for s, k, *_ in sutunlar for p in yt[s][k] if not _bos(p.get("n"))})
    tab = []
    for n in nler:
        h = [_e(tam(n))]
        for s, k, *_ in sutunlar:
            p = next((q for q in yt[s][k] if q.get("n") == n), None)
            h.append("—" if not p else f'{_e(sayi(p.get("carpan"), 1))}× <span class="ga">{_e(ga(p.get("alt"), p.get("ust"), 1))}</span>'
                                       f' · ≈{_e(tam(p.get("n_esdeger")))}')
        tab.append(h)
    tablo = _tablo(["n"] + [f"{_e(et)} · {ad}" for _, _, ad, et in sutunlar], tab, "sayisal") if tab else ""
    # "hiç insan yokken" eşdeğeri (yalniz_sim) JSON'da kalır, panoda gösterilmez: simülasyonun tek bir ortalamayı
    # tutturmasına bağlı, jüriye yanlış güven verir (KARŞIT 4)
    cumle = f'<p class="cumle">{_e(orn["cumle"])}</p>' if orn.get("cumle") else ""
    notlar = "".join(f"<li>{_e(n)}</li>" for n in orn.get("notlar") or [])
    ic_html = (cumle + (_lejant(lej) if lej else "") + svg + tablo
               + (f'<ul class="sinir">{notlar}</ul>' if notlar else ""))
    return _bolum("Etkin örneklem çarpanı", ic_html,
                  _e(orn.get("aciklama") or "Simülasyon eklenince n gerçek yanıt kaç gerçek yanıta eşdeğer.")
                  + " Hücre: çarpan [%95 GA] · ≈ eşdeğer insan sayısı. Kesikli çapraz çizginin üstü kazanç.")


def _carpan_100(oz: dict) -> str:
    """Üst tablo için çarpan @n=100 (manşetteki 'bizim' sütunu, yoksa ilk yöntem)."""
    yt = (oz.get("orneklem") or {}).get("yontemler") or {}
    if not yt:
        return "—"
    s = ((oz.get("manset") or {}).get("bizim") or {}).get("sutun")
    s = s if s in yt else next(iter(yt))
    par = []
    for k, ad, _ in KESTIRICI:
        p = next((q for q in yt[s].get(k) or [] if q.get("n") == 100), None)
        if p and not _bos(p.get("carpan")):
            par.append(f'{ad} {_e(sayi(p["carpan"], 1))}× <span class="ga">{_e(ga(p.get("alt"), p.get("ust"), 1))}</span>')
    return " · ".join(par) or "—"


# ------------------------------------------------------------------ 0. kaç insana bedel (bütün tarzlar, tek grafik)
TARZ_RENK = {"intent": "s2", "persona": "s1", "deney": "s3", "makro": "s4"}
TARZ_AD = {"intent": "Satın alma niyeti (Open e-commerce)", "persona": "Persona ikizi (Twin-2K-500)",
           "deney": "A/B başlık testi (Upworthy)", "makro": "Pazar (makro)"}


def _bedel_svg(hepsi: list[dict], kuram: bool, etiket: str, xyazi: str, yyazi: str) -> str:
    """x = ρ, y = çarpan (log). kuram=True: 1/(1 − ρ²) eğrisi (yalnız ortalama kestirimi panelinde geçerli)."""
    G, Y, sol, sag, ust, alt = 640, 320, 52, 20, 12, 40
    yu = max([q["carpan"] for q in hepsi] + [ga_[1] for q in hepsi if (ga_ := q.get("carpan_ga")) and not _bos(ga_[1])]) * 1.15
    ylo, yhi = 0.9, max(yu, 2.0)
    lx = lambda r: sol + r * (G - sol - sag)
    ly = lambda v: ust + (1 - (math.log10(v) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo))) * (Y - ust - alt)
    ic = []
    for r in (0, .2, .4, .6, .8, 1.0):
        ic.append(f'<line class="izgara" x1="{lx(r):.1f}" x2="{lx(r):.1f}" y1="{ust}" y2="{Y - alt}"/>'
                  f'<text class="eksen" x="{lx(r):.1f}" y="{Y - alt + 14}" text-anchor="middle">{_e(sayi(r, 1))}</text>')
    for v in [1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20]:
        if ylo <= v <= yhi:
            ic.append(f'<line class="izgara" x1="{sol}" x2="{G - sag}" y1="{ly(v):.1f}" y2="{ly(v):.1f}"/>'
                      f'<text class="eksen" x="{sol - 6}" y="{ly(v) + 4:.1f}" text-anchor="end">{_e(sayi(v, 1 if v % 1 else 0))}×</text>')
    if kuram:
        pts, r = [], 0.0
        while r < 0.999:
            v = 1 / (1 - r * r)
            if v > yhi:
                break
            pts.append(f"{lx(r):.1f},{ly(v):.1f}")
            r += 0.005
        ic.append(f'<polyline class="kuram" points="{" ".join(pts)}"/>')
        rk = 0.75
        ic.append(f'<text class="eksen" x="{lx(rk) - 8:.1f}" y="{ly(1 / (1 - rk * rk)) - 6:.1f}" text-anchor="end">kuram 1/(1 − ρ²)</text>')
    ic.append(f'<line class="tavan" x1="{sol}" x2="{G - sag}" y1="{ly(1):.1f}" y2="{ly(1):.1f}"/>'
              f'<text class="eksen" x="{G - sag - 4}" y="{ly(1) - 5:.1f}" text-anchor="end">yalnız gerçek insan (1×)</text>')
    for q in hepsi:
        renk = TARZ_RENK.get(q.get("tarz"), "s7")
        x, y = lx(q["rho"]), ly(q["carpan"])
        bi = ""
        cg, rg = q.get("carpan_ga"), q.get("rho_ga")
        if cg and not _bos(cg[0]) and not _bos(cg[1]):
            bi += f'<line class="biyik ince" x1="{x:.1f}" x2="{x:.1f}" y1="{ly(max(cg[0], ylo)):.1f}" y2="{ly(min(cg[1], yhi)):.1f}"/>'
        if rg and not _bos(rg[0]) and not _bos(rg[1]):
            bi += f'<line class="biyik ince" x1="{lx(rg[0]):.1f}" x2="{lx(rg[1]):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
        ipucu = (f'{q.get("onsel")} · {q.get("veri", "")}: ρ {sayi(q["rho"])}, çarpan {sayi(q["carpan"], 2)}× '
                 f'{ga(*(cg or (None, None)), 2)}; kuram {sayi(q.get("teorik"), 2)}× · kaynak: {q.get("kaynak", "")}')
        sinif = f"bos {renk}" if q.get("rejim") == "arşivsiz" else f"nokta {renk}"
        etk = ""
        if q.get("kisa"):
            yer = q.get("kisa_yer")
            ex_, ey_, ea_ = ((x, y + 24, "middle") if yer == "alt" else (x - 4, y - 12, "end") if yer == "ust"
                             else (x + 9, y + 4, "start") if yer == "sag" else (x - 9, y + 4, "end"))
            etk = f'<text class="etk halo" x="{ex_:.1f}" y="{ey_:.1f}" text-anchor="{ea_}">{_e(q["kisa"])}</text>'
        ic.append(f'<g><title>{_e(ipucu)}</title>{bi}<circle class="{sinif}" cx="{x:.1f}" cy="{y:.1f}" r="5.5"/>{etk}</g>')
    ic.append(f'<text class="eksen" x="{(sol + G - sag) / 2}" y="{Y - 4}" text-anchor="middle">{_e(xyazi)}</text>'
              f'<text class="eksen" x="12" y="{(ust + Y - alt) / 2}" transform="rotate(-90 12 {(ust + Y - alt) / 2})" '
              f'text-anchor="middle">{_e(yyazi)}</text>')
    return _svg(G, Y, "".join(ic), etiket)


def _bedel(b: dict | None) -> str:
    """İki panel. A: kitle ortalaması kestirimi (PPI++), kuram eğrisiyle. B: A/B kazanan seçimi (farklı hedef, Bayes
    birleştirme, çarpan n'ye bağlı): eğri çizilmez, arşivsiz / arşivli müşteri rejimleri ayrı satır."""
    if not b:
        return ""
    gecerli = lambda L: [q for q in L or [] if not _bos(q.get("rho")) and not _bos(q.get("carpan")) and q["carpan"] > 0]
    A, B_ = gecerli(b.get("noktalar")), gecerli(b.get("ayri_hedef"))
    if not A and not B_:
        return ""
    n = tam(b.get("n", 100))
    parca = ['<section class="bolum bedel"><h3>Kaç insana bedel</h3>']
    if A:
        tarzlar = list(dict.fromkeys(q.get("tarz") for q in A))
        satirlar = []
        for q in A:
            cg, rg = q.get("carpan_ga") or (None, None), q.get("rho_ga") or (None, None)
            n_e = q.get("n_esdeger")
            satirlar.append([_e(TARZ_AD.get(q.get("tarz"), q.get("tarz"))).split(" (")[0], _e(q.get("onsel")),
                             f'{_e(sayi(q.get("rho")))} <span class="ga">{_e(ga(*rg))}</span>',
                             f'<b>{_e(sayi(q.get("carpan"), 2))}×</b> <span class="ga">{_e(ga(*cg, 2)) if not _bos(cg[0]) else "GA ölçülmedi"}</span>',
                             _e(sayi(q.get("teorik"), 2)) + "×",
                             _e(tam(n_e)) if not _bos(n_e) else "—",
                             _e(q["panel_dolar"].split(" [")[0]) if q.get("panel_dolar") else "—",
                             (_e(yuzde(q["kapsama"], 1)) if not _bos(q.get("kapsama")) else "—")
                             + (f'<br><span class="ga">en kötü soru {_e(sayi(q["kapsama_en_dusuk"], 3))}</span>'
                                if not _bos(q.get("kapsama_en_dusuk")) and q["kapsama"] - q["kapsama_en_dusuk"] > 0.05 else ""),
                             f'<code>{_e(q.get("kaynak", ""))}</code>' + (f'<br><span class="ga">{_e(q["not"])}</span>' if q.get("not") else "")])
        kap = [q["kapsama"] for q in A if not _bos(q.get("kapsama")) and q.get("carpan_ga")]
        kap_c = ""
        if kap:
            lo, hi = min(kap), max(kap)
            kap_c = (f"Aralık tutuyor mu: PPI++ %95 güven aralığı, ölçtüğümüz {len(kap)} noktada tekrarların "
                     f"%{sayi(100 * lo, 0)}-%{sayi(100 * hi, 0)}'ünde gerçek ortalamayı kapsadı. Nominal %95'in biraz altında "
                     "(n=100'de normal yaklaşım hafif dar); sahnede \"%95 GA\" değil \"%93-94 kapsayan aralık\" denmeli.")
        alt_a = (f"Önsel + {n} gerçek yanıt, kitle ortalamasını kaç gerçek yanıt kadar kesin kestiriyor (PPI++, Angelopoulos, "
                 "Duchi, Zrnic 2023). Kuram eğrisiyle doğrudan kıyaslanır. ≈ insan sütunu sonlu kitle düzeltmeli. Bıyıklar %95 GA. "
                 "Aynı kitlenin hedef etiketiyle eğitilmiş (döngüsel) önseller çizilmez.")
        parca += ['<h4>A. Ortalama kestirimi (PPI++)</h4>', f'<p class="alt">{_e(alt_a)}</p>',
                  _lejant([(TARZ_AD.get(t, t), TARZ_RENK.get(t, "s7")) for t in tarzlar]),
                  _bedel_svg(A, True, "Ortalama kestirimi: korelasyona karşı etkin örneklem çarpanı, kuram eğrisiyle",
                             "ρ: simülasyon ile gerçek cevabın korelasyonu (kişi düzeyi)", "kaç insana bedel: çarpan (log)"),
                  _tablo(["Tarz", "Önsel (simülasyon)", "ρ [%95 GA]", "Çarpan [%95 GA]", "Kuram", f"≈ insan (n={n})",
                          "Panel karşılığı [hesap]", "Aralığın kapsaması", "Kaynak"], satirlar)]
        ke = b.get("kapsama_ek")
        if kap_c and ke:
            kap_c += (f" Ayrı bir veride de aynı: {ke['veri']}, {ke['dalga']} dalga, {tam(ke['soru'])} soru, önsel "
                      f"{ke['onsel']}: kapsama %{sayi(100 * ke['kapsama_alt'], 1)}-%{sayi(100 * ke['kapsama_ust'], 1)}, "
                      f"en kötü soru {sayi(ke['en_kotu_soru'], 3)}.")
        if kap_c:
            parca.append(f'<p class="alt"><b>{_e(kap_c)}</b></p>')
        fiyat = next((q["panel_dolar"] for q in A if q.get("panel_dolar")), None)
        if fiyat:
            parca.append(f'<p class="alt">Panel karşılığı: (≈ insan − {n}) ek yanıt × yanıt başı fiyat '
                         f'{_e(fiyat.split("yanıt başı ")[-1].rstrip("]"))}. Hesap: Pollfish herkese açık tabanı $0,95, '
                         'Prolific 10 dakikalık anket (önerilen ücret + platform payı) ≈ $2,86. Herkese açık alt sınır; '
                         'hedefli kitlede artar.</p>')
    if B_:
        sira = {"arşivsiz": 0, "arşivli": 1}
        B_ = sorted(B_, key=lambda q: (sira.get(q.get("rejim"), 2), q["carpan"]))
        satirlar = []
        for q in B_:
            cg, rg = q.get("carpan_ga") or (None, None), q.get("rho_ga") or (None, None)
            rj = q.get("rejim") or "—"
            satirlar.append([_e({"arşivsiz": "arşivsiz müşteri", "arşivli": "arşivli müşteri"}.get(rj, rj)), _e(q.get("onsel")),
                             f'{_e(sayi(q.get("rho")))} <span class="ga">{_e(ga(*rg))}</span>',
                             f'<b>{_e(sayi(q.get("carpan"), 2))}×</b> <span class="ga">{_e(ga(*cg, 2))}</span>',
                             _e(q.get("veri", "")), f'<code>{_e(q.get("kaynak", ""))}</code>'])
        alt_b = ("A/B testinde kazanan başlığı seçmek: önsel + gösterim başına 100 gerçek tıklama, doğru kazananı kaç kat "
                 "gösterime eşdeğer bulduruyor (deney tarzının ölçümü, Bayes birleştirme). Hedef A'dan farklı, çarpan n'ye bağlı "
                 "(küçük n'de büyük); bu yüzden 1/(1 − ρ²) eğrisi burada çizilmez ve noktalar A paneliyle kıyaslanmaz. "
                 "İçi boş: arşivsiz müşteri (geçmiş test yok, sıfır gün); dolu: arşivli müşteri (geçmiş testlerden öğrenen çapa).")
        parca += ['<h4>B. Kazanan seçimi (A/B, farklı hedef)</h4>', f'<p class="alt">{_e(alt_b)}</p>',
                  _bedel_svg(B_, False, "Kazanan seçimi: korelasyona karşı kazanç, arşivsiz ve arşivli müşteri",
                             "ρ: önsel skoru ile gerçek tıklama farkının (log-odds) korelasyonu, başlık çifti düzeyi", "kazanç: kaç kat gösterime eşdeğer (log)"),
                  _tablo(["Rejim", "Önsel", "ρ [%95 GA]", "Kazanç [%95 GA]", "Veri", "Kaynak"], satirlar)]
        parca += [f'<p class="alt"><b>{_e(x)}</b></p>' for x in b.get("ayri_hedef_notlar") or []]
    tr = b.get("tr")
    if tr and tr.get("satirlar"):
        satirlar = [[_e(r["yontem"]), f'<b>{_e(sayi(r["deger"], 3))}</b> <span class="ga">{_e(ga(*r["ga"], 3))}</span>',
                     _e(r["bosluk"].replace(", ", "; ").replace(".", ",").replace("-", "−")) if r.get("bosluk") else "—",
                     _e(tam(r["insan"])) if not _bos(r.get("insan")) else "—"] for r in tr["satirlar"]]
        parca += [f'<h4>C. {_e(tr["baslik"])}</h4>', f'<p class="alt">{_e(tr["not"])}</p>',
                  _tablo(["Yöntem", f'{_e(tr["metrik"])} [%95 GA]', "Kapanan boşluk (taban: diğer ülkeler)",
                          "≈ kaç Türk yanıtlayıcı"], satirlar),
                  f'<p class="alt">Kaynak: <code>{_e(tr["kaynak"])}</code></p>']
    return "".join(parca) + "</section>"


# ------------------------------------------------------------------ 3. baseline merdiveni
def _merdiven(o: _Ozet) -> str:
    md = [r for r in o.oz.get("merdiven") or [] if not _bos(r.get("deger"))]
    if not md:
        return ""
    birincil = o.oz.get("birincil", "")
    ters = o.yon.get(birincil, 1) == -1
    tavan = (o.oz.get("manset") or {}).get("tavan") or {}
    lo, hi = _olcek([v for r in md for v in (r.get("deger"), r.get("alt"), r.get("ust"))] + [tavan.get("deger")])
    G, sol, sag, satir = 600, 180, 20, 30
    Y = 16 + satir * len(md)
    ex = _Eksen(lo, hi, sol, G - sag, ters)
    ic = [ex.izler(8, Y)]
    for k, r in enumerate(md):
        y = 16 + satir * k + satir / 2 - 4
        xa = ex(r["alt"]) if not _bos(r.get("alt")) else None
        xu = ex(r["ust"]) if not _bos(r.get("ust")) else None
        tur = r.get("tur", "baseline")
        ic.append(f'<text class="etk" x="{sol - 10}" y="{y + 4:.1f}" text-anchor="end">{_e(_kisalt(r.get("etiket") or r.get("sutun")))}</text>')
        ic.append(_nokta_ga(ex(r["deger"]), y, xa, xu, f"t-{tur}",
                            f'{r.get("etiket")}: {sayi(r["deger"])} {ga(r.get("alt"), r.get("ust"))}', r=5.5 if tur == "yontem" else 4.5))
        ic.append(f'<text class="deg" x="{ex(r["deger"]):.1f}" y="{y - 8:.1f}" text-anchor="middle">{_e(sayi(r["deger"]))}</text>')
    if not _bos(tavan.get("deger")):
        x = ex(tavan["deger"])
        ic.append(f'<line class="tavan" x1="{x:.1f}" x2="{x:.1f}" y1="6" y2="{Y}"/>'
                  f'<text class="eksen" x="{x - 4:.1f}" y="12" text-anchor="end">insan tavanı</text>')
    yon_not = "küçük iyi; eksen ters çevrildi, sağ = daha iyi" if ters else "büyük iyi; sağ = daha iyi"
    svg = _svg(G, Y + 22, "".join(ic), f"Baseline merdiveni, {_metrik_adi(birincil)}")
    satirlar = []
    for r in md:
        fark = "—"
        if not _bos(r.get("fark_onceki")):
            fark = f'{_e(("+" if r["fark_onceki"] > 0 else "") + sayi(r["fark_onceki"]))} <span class="ga">{_e(ga(r.get("fark_alt"), r.get("fark_ust")))}</span>'
        nz = (o.sonuclar.get(r.get("sutun")) or {}).get("normalize")
        nzs = f'{_e(yuzde(nz["deger"]))} <span class="ga">{_e(ga(nz.get("alt"), nz.get("ust"), yz=True))}</span>' if nz else "—"
        satirlar.append([f'{_e(r.get("etiket") or r.get("sutun"))} {_rozet(r.get("tur", ""))}',
                         _deger_ga(r), fark, _e(p_bicim(r.get("p"))), nzs])
    tablo = _tablo(["Basamak", "Değer", "Öncekine fark", "p", "Normalize"], satirlar, "sayisal")
    return _bolum("Baseline merdiveni", f'<div class="ikili">{svg}{tablo}</div>',
                  f"{_e(_metrik_adi(birincil))} ({yon_not}). Nokta = değer, bıyık = %95 GA, kesikli çizgi = insan tavanı (test-retest).")


# ------------------------------------------------------------------ 4. metrik tablosu
def _metrik_tablosu(o: _Ozet) -> str:
    if not o.sonuclar:
        return ""
    adlar = [m["ad"] for m in o.oz.get("metrikler") or [] if "ad" in m]
    for r in o.sonuclar.values():
        adlar += [a for a in (r.get("metrikler") or {}) if a not in adlar]
    en_iyi = {}
    for a in adlar:
        aday = [(float(r["metrikler"][a]["deger"]), s) for s, r in o.sonuclar.items()
                if r.get("tur") not in ("tavan", "kontrol") and a in (r.get("metrikler") or {})
                and not _bos(r["metrikler"][a].get("deger"))]
        if aday:
            en_iyi[a] = (max(aday) if o.yon.get(a, 1) == 1 else min(aday))[1]
    basliklar = ["Sütun"] + [f'<span title="{_e(o.acik.get(a, ""))}">{_e(_metrik_adi(a))}{" ↓" if o.yon.get(a) == -1 else " ↑"}</span>'
                             for a in adlar]
    satirlar = []
    for s, r in o.sonuclar.items():
        hucre = [f'{_e(o.etiket(s))} {_rozet(r.get("tur", ""))}']
        for a in adlar:
            d = (r.get("metrikler") or {}).get(a)
            h = _deger_ga(d)
            hucre.append(f'<span class="eniyi">★ {h}</span>' if en_iyi.get(a) == s else h)
        satirlar.append(hucre)
    return _bolum("Metrik tablosu", _tablo(basliklar, satirlar, "sayisal"),
                  "↑ büyük iyi, ↓ küçük iyi. ★ = sütunun en iyisi (tavan ve kontrol satırları hariç).")


# ------------------------------------------------------------------ 5. eşleşik farklar
def _farklar(o: _Ozet) -> str:
    kl = list(o.oz.get("karsilastirmalar") or [])
    kayitli = o.oz.get("kayitli_karsilastirmalar") or []   # evaluate(compare=, blind_col=): kayıtlı / düşme koşulu
    kl += kayitli
    if not kl:
        return ""
    satirlar = []
    for c in kl:
        alt, ust = c.get("alt"), c.get("ust")
        anlamli = (not _bos(alt) and not _bos(ust) and (alt > 0 or ust < 0)) or (not _bos(c.get("p")) and c["p"] < 0.05)
        f = c.get("fark")
        tek = c.get("tur") == "kayitli" and c.get("b") is None   # A'nın değeri 0'a karşı: yön yorumu yok
        if not anlamli:
            rz = '<span class="drm d-atlandi">○ anlamlı değil</span>'
        elif tek:
            yer = "0'ın üstünde" if (not _bos(f) and f > 0) else "0'ın altında"
            rz = f'<span class="drm d-atlandi">● anlamlı · {_e(yer)}</span>'
        elif c.get("a_iyi"):
            rz = '<span class="drm d-tamam">▲ anlamlı · A iyi</span>'
        else:
            rz = '<span class="drm d-hata">▼ anlamlı · B iyi</span>'
        if c.get("tur") in ("kayitli", "kor"):
            et_ = ("kör kol · " if c["tur"] == "kor" else "kayıtlı · ") + (c.get("etiket") or "")
            b_ = _e(o.etiket(c["b"])) if c.get("b") else "0"
            satirlar.append([_e(et_), b_, _e(_metrik_adi(c.get("metrik", ""))),
                             f'{_e(("+" if not _bos(f) and f > 0 else "") + sayi(f))} <span class="ga">{_e(ga(alt, ust))}</span>',
                             _e(p_bicim(c.get("p"))), rz])
            continue
        satirlar.append([_e(o.etiket(c.get("a"))), _e(o.etiket(c.get("b"))), _e(_metrik_adi(c.get("metrik", ""))),
                         f'{_e(("+" if not _bos(f) and f > 0 else "") + sayi(f))} <span class="ga">{_e(ga(alt, ust))}</span>',
                         _e(p_bicim(c.get("p"))), rz])
    return _bolum("Eşleşik farklar (A − B)", _tablo(["A", "B", "Metrik", "Fark [GA]", "p", "Sonuç"], satirlar, "sayisal"),
                  "Aynı bootstrap örneklemlerinde hesaplanan fark; GA sıfırı dışlıyorsa anlamlı. Küçük-iyi metrikte negatif fark A lehinedir.")


# ------------------------------------------------------------------ 6. kalibrasyon
def _guvenilirlik(egri, renk, ad) -> str:
    G, s, b = 240, 36, 196  # sol/alt kenar, çizim boyu
    xy = lambda g, d: (s + g * b, 8 + (1 - d) * b)
    ic = [f'<rect class="cerceve" x="{s}" y="8" width="{b}" height="{b}"/>',
          f'<line class="tavan" x1="{s}" y1="{8 + b}" x2="{s + b}" y2="8"/>']
    for v in (0, 0.5, 1):
        ic.append(f'<text class="eksen" x="{s + v * b:.1f}" y="{8 + b + 14}" text-anchor="middle">{_e(sayi(v, 1))}</text>'
                  f'<text class="eksen" x="{s - 5}" y="{8 + (1 - v) * b + 4:.1f}" text-anchor="end">{_e(sayi(v, 1))}</text>')
    ic.append(f'<text class="eksen" x="{s + b / 2}" y="{G - 2}" text-anchor="middle">güven</text>'
              f'<text class="eksen" x="10" y="{8 + b / 2}" transform="rotate(-90 10 {8 + b / 2})" text-anchor="middle">doğruluk</text>')
    noktalar = [p for p in egri if not _bos(p.get("guven")) and not _bos(p.get("dogruluk"))]
    nmax = max([p.get("n") or 0 for p in noktalar] + [1])
    if noktalar:
        ic.append(f'<polyline class="cizgi {renk}" points="{" ".join("%.1f,%.1f" % xy(p["guven"], p["dogruluk"]) for p in noktalar)}"/>')
    for p in noktalar:
        x, y = xy(p["guven"], p["dogruluk"])
        r = 3 + 5 * math.sqrt((p.get("n") or 0) / nmax)
        ic.append(f'<g><title>güven {sayi(p["guven"], 2)} · doğruluk {sayi(p["dogruluk"], 2)} · n={tam(p.get("n"))}</title>'
                  f'<circle class="nokta {renk} halka" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}"/></g>')
    return _svg(G, G + 4, "".join(ic), f"Güvenilirlik eğrisi: {ad}")


def _kapsam(o: _Ozet, sutunlar) -> str:
    G, Y, sol, ust, alt = 600, 260, 48, 10, 30
    seriler = [(s, [p for p in o.oz["kalibrasyon"][s].get("kapsam") or []
                    if not _bos(p.get("kapsam")) and not _bos(p.get("dogruluk"))]) for s in sutunlar]
    seriler = [(s, sorted(p, key=lambda q: q["kapsam"])) for s, p in seriler if p]
    if not seriler:
        return ""
    lo, hi = _olcek([p["dogruluk"] for _, ps in seriler for p in ps])
    lo, hi = max(lo, 0.0), min(hi, 1.0) if hi <= 1 else hi
    ex = _Eksen(0, 1, sol, G - 120)
    yy = lambda v: ust + (1 - (v - lo) / (hi - lo)) * (Y - ust - alt)
    ic = [ex.izler(ust, Y - alt)]
    v, st = lo, _adim(hi - lo)
    while v <= hi + 1e-9:
        ic.append(f'<line class="izgara" x1="{sol}" x2="{G - 120}" y1="{yy(v):.1f}" y2="{yy(v):.1f}"/>'
                  f'<text class="eksen" x="{sol - 6}" y="{yy(v) + 4:.1f}" text-anchor="end">{_e(sayi(v, 2))}</text>')
        v += st
    uclar = []
    for s, ps in seriler:
        r = o.renk.get(s, "s1")
        ic.append(f'<polyline class="cizgi {r}" points="{" ".join("%.1f,%.1f" % (ex(p["kapsam"]), yy(p["dogruluk"])) for p in ps)}"/>')
        for p in ps:
            ic.append(f'<g><title>{_e(o.etiket(s))}: kapsam {yuzde(p["kapsam"])} · doğruluk {sayi(p["dogruluk"])}</title>'
                      f'<circle class="nokta {r} halka" cx="{ex(p["kapsam"]):.1f}" cy="{yy(p["dogruluk"]):.1f}" r="3.5"/></g>')
        son = ps[-1]
        uclar.append([yy(son["dogruluk"]) + 4, ex(son["kapsam"]) + 8, _kisalt(o.etiket(s), 16)])
    ic += [f'<text class="etk" x="{x:.1f}" y="{y:.1f}">{_e(t)}</text>' for y, x, t in _ayir(uclar)]
    ic.append(f'<text class="eksen" x="{(sol + G - 120) / 2}" y="{Y + 4}" text-anchor="middle">kapsam (en emin olunan pay)</text>')
    svg = _svg(G, Y + 8, "".join(ic), "Çekimser kalma eğrisi")
    return (f'<h4>Çekimser kalma: kapsam–doğruluk</h4>{_lejant([(o.etiket(s), o.renk.get(s, "s1")) for s, _ in seriler])}{svg}'
            '<p class="alt">Model yalnız en emin olduğu payda cevap verirse doğruluk ne olur. Eğri soldan sağa düşüyorsa güven puanı bilgi taşıyor.</p>')


def _kalibrasyon(o: _Ozet) -> str:
    kal = o.oz.get("kalibrasyon") or {}
    if not kal:
        return ""
    ciz = o.secili(list(kal))
    kartlar = []
    for s in ciz:
        k = kal[s]
        kartlar.append(f'<figure class="kalkart"><figcaption><i class="lj {o.renk.get(s, "s1")}"></i>{_e(o.etiket(s))}</figcaption>'
                       f'{_guvenilirlik(k.get("egri") or [], o.renk.get(s, "s1"), o.etiket(s))}'
                       f'<p class="kalsayi">ECE {_e(sayi(k.get("ece")))} · güven AUROC {_e(sayi(k.get("guven_auroc")))} · AURC {_e(sayi(k.get("aurc")))}</p></figure>')
    satirlar = [[f'{_e(o.etiket(s))} {_rozet(o.tur(s))}', _e(sayi(k.get("ece"))), _e(sayi(k.get("guven_auroc"))),
                 _e(sayi(k.get("aurc"))), _e(", ".join(o.etiket(c) for c in k.get("kalibre_sutunlar") or []) or "—")]
                for s, k in kal.items()]
    ic = (f'<div class="kalizgara">{"".join(kartlar)}</div>'
          + _tablo(["Sütun", "ECE ↓", "Güven AUROC ↑", "AURC ↓", "Kalibre edilmiş"], satirlar, "sayisal")
          + _kapsam(o, ciz))
    return _bolum("Kalibrasyon", ic, "Köşegen = mükemmel kalibrasyon; nokta büyüklüğü kutudaki örnek sayısıyla orantılı. "
                  "Çizilen en fazla 4 sütun; hepsi tabloda.")


# ------------------------------------------------------------------ 7. segment hatası
def _segment_tek(o: _Ozet, seg: str, sd: dict) -> str:
    metrik = sd.get("metrik") or o.oz.get("birincil", "")
    ters = o.yon.get(metrik, 1) == -1
    satirlar = [r for r in sd.get("satirlar") or [] if r.get("yontemler")]
    if not satirlar:
        return ""
    yontemler = o.secili(list(dict.fromkeys(s for r in satirlar for s in r["yontemler"])))
    ana = yontemler[0]
    kesit = ""
    if len(satirlar) > 20:  # en kötü 10 + en iyi 5 (ana yönteme göre)
        def puan(r):
            v = (r["yontemler"].get(ana) or {}).get("deger")
            return float("inf") if _bos(v) else (v if not ters else -v)
        sirali = sorted(satirlar, key=puan)
        satirlar = sirali[:10] + sirali[-5:]
        kesit = f" {len(sirali)} segmentten en kötü 10 ve en iyi 5 gösteriliyor."
    degerler = [d.get(a) for r in satirlar for d in r["yontemler"].values() for a in ("deger", "alt", "ust")]
    genel = ((o.sonuclar.get(ana) or {}).get("metrikler") or {}).get(metrik, {}).get("deger")
    lo, hi = _olcek(degerler + [genel])
    G, sol, sag, bant = 600, 190, 20, 14 + 10 * len(yontemler)
    Y = 10 + bant * len(satirlar)
    ex = _Eksen(lo, hi, sol, G - sag, ters)
    ic = [ex.izler(4, Y)]
    for k, r in enumerate(satirlar):
        y0 = 10 + bant * k
        if k % 2:
            ic.append(f'<rect class="bant" x="0" y="{y0}" width="{G}" height="{bant}"/>')
        ic.append(f'<text class="etk" x="{sol - 10}" y="{y0 + bant / 2 + 4:.1f}" text-anchor="end">'
                  f'{_e(_kisalt(r.get("segment"), 20))} <tspan class="eksen">n={_e(tam(r.get("n")))}</tspan></text>')
        for j, s in enumerate(yontemler):
            d = r["yontemler"].get(s) or {}
            if _bos(d.get("deger")):
                continue
            y = y0 + 12 + 10 * j
            ic.append(_nokta_ga(ex(d["deger"]), y, None if _bos(d.get("alt")) else ex(d["alt"]),
                                None if _bos(d.get("ust")) else ex(d["ust"]), o.renk.get(s, "s1"),
                                f'{r.get("segment")} · {o.etiket(s)}: {sayi(d["deger"])} {ga(d.get("alt"), d.get("ust"))}', r=3.5))
    if not _bos(genel):
        x = ex(genel)
        ic.append(f'<line class="tavan" x1="{x:.1f}" x2="{x:.1f}" y1="4" y2="{Y}"/>'
                  f'<text class="eksen" x="{x + 4:.1f}" y="9">genel ({_e(_kisalt(o.etiket(ana), 14))})</text>')
    svg = _svg(G, Y + 22, "".join(ic), f"Segment hatası: {seg}")
    # tablo: değer [GA] · yanlılık
    bas = ["Segment", "n"] + [f"{_e(o.etiket(s))}" for s in yontemler] + [f"Yanlılık · {_e(o.etiket(s))}" for s in yontemler]
    tab = []
    for r in satirlar:
        h = [_e(r.get("segment")), _e(tam(r.get("n")))]
        h += [_deger_ga(r["yontemler"].get(s)) for s in yontemler]
        for s in yontemler:
            yl = (r["yontemler"].get(s) or {}).get("yanlilik")
            h.append("—" if _bos(yl) else _e(("+" if yl > 0 else "") + sayi(yl)))
        tab.append(h)
    kotu = "".join(f'<li><b>{_e(e.get("segment"))}</b> · {_e(o.etiket(e.get("sutun")))}: {_e(sayi(e.get("deger")))}'
                   f' <span class="ga">genelden fark {_e(sayi(e.get("fark_genel")))}</span></li>' for e in sd.get("en_kotu") or [])
    kor = sd.get("segment_fark_kor") or {}
    kor_t = _tablo(["Sütun", "Segment farklarını yakalama (r)"],
                   [[_e(o.etiket(s)), _e(sayi(v))] for s, v in kor.items()], "sayisal") if kor else ""
    yon_not = "küçük iyi, eksen ters" if ters else "büyük iyi"
    return (f'<h4>{_e(seg)} · {_e(_metrik_adi(metrik))} ({yon_not})</h4><p class="alt">{kesit.strip()}</p>'
            f'{_lejant([(o.etiket(s), o.renk.get(s, "s1")) for s in yontemler])}{svg}'
            + (f'<div class="uyari-kutu"><b>En kötü 3 segment</b><ol>{kotu}</ol></div>' if kotu else "")
            + _tablo(bas, tab, "sayisal") + kor_t)


def _segment(o: _Ozet) -> str:
    sg = o.oz.get("segment") or {}
    ic = "".join(_segment_tek(o, s, d) for s, d in sg.items())
    return _bolum("Segment hatası", ic, "Simülasyon hangi grupta tutmuyor? Yanlılık = tahmin ortalaması − gerçek ortalama "
                  "(pozitif: simülasyon o grubu olduğundan yüksek gösteriyor).") if ic else ""


# ------------------------------------------------------------------ 8. yayılım
def _yayilim(o: _Ozet) -> str:
    ya = {s: d for s, d in (o.oz.get("yayilim") or {}).items() if not _bos(d.get("sd_orani"))}
    if not ya:
        return ""
    hi = max([1.2] + [float(d.get("sd_orani_ust") or d["sd_orani"]) for d in ya.values()]) * 1.05
    G, sol, sag, satir = 600, 180, 140, 28
    Y = 10 + satir * len(ya)
    ex = _Eksen(0, hi, sol, G - sag)
    ic = [ex.izler(4, Y)]
    for k, (s, d) in enumerate(ya.items()):
        y = 10 + satir * k
        v = float(d["sd_orani"])
        ic.append(f'<text class="etk" x="{sol - 10}" y="{y + satir / 2 + 2:.1f}" text-anchor="end">{_e(_kisalt(o.etiket(s)))}</text>'
                  f'<g><title>{_e(o.etiket(s))}: sd oranı {sayi(v)} {ga(d.get("sd_orani_alt"), d.get("sd_orani_ust"))}</title>'
                  f'<rect class="cubuk {o.renk.get(s, "s1")}" x="{sol}" y="{y + 6}" width="{max(ex(v) - sol, 1):.1f}" height="{satir - 14}" rx="3"/></g>')
        if not _bos(d.get("sd_orani_alt")) and not _bos(d.get("sd_orani_ust")):
            ym = y + satir / 2 - 1
            ic.append(f'<line class="biyik" x1="{ex(d["sd_orani_alt"]):.1f}" x2="{ex(d["sd_orani_ust"]):.1f}" y1="{ym}" y2="{ym}"/>')
        etiket = "homojen" if v < 0.9 else ("aşırı yayılım" if v > 1.1 else "insan gibi")
        xs = ex(max(v, float(d["sd_orani_ust"]) if not _bos(d.get("sd_orani_ust")) else v))
        ic.append(f'<text class="deg" x="{xs + 6:.1f}" y="{y + satir / 2 + 2:.1f}">×{_e(sayi(v, 2))} {etiket}</text>')
    x1 = ex(1.0)
    ic.append(f'<line class="tavan" x1="{x1:.1f}" x2="{x1:.1f}" y1="2" y2="{Y}"/>')
    svg = _svg(G, Y + 22, "".join(ic), "Dağılım yayılımı, standart sapma oranı")
    tab = [[_e(o.etiket(s)), f'{_e(sayi(d["sd_orani"]))} <span class="ga">{_e(ga(d.get("sd_orani_alt"), d.get("sd_orani_ust")))}</span>',
            _e(sayi(d.get("entropi_orani"))), _e(sayi(d.get("w1")))] for s, d in ya.items()]
    return _bolum("Dağılım yayılımı (homojenleşme)", svg + _tablo(["Sütun", "SD oranı", "Entropi oranı", "W1 ↓"], tab, "sayisal"),
                  "Simüle edilen yayılım ÷ gerçek yayılım. Kesikli çizgi 1,0 = insanlar kadar çeşitli; 1'in altı "
                  "LLM'lerin bilinen homojenleşme sorunu (herkes ortalama cevabı veriyor).")


# ------------------------------------------------------------------ 9. denetimler
KRITIK_DENETIM = ("bölme bilgisi", "grup-ayrık", "grup çakışması", "zaman sızıntısı", "kontaminasyon", "ezber")


def _denetim(o: _Ozet) -> str:
    dn = o.oz.get("denetim") or []
    if not dn:
        return ""
    sayac = {}
    satirlar = []
    for d in dn:
        durum = d.get("durum", "atlandi")
        sayac[durum] = sayac.get(durum, 0) + 1
        simge, ad = DURUM.get(durum, ("?", durum))
        satirlar.append([f'<span class="drm d-{_e(durum)}">{simge} {_e(ad)}</span>', _e(d.get("kontrol")),
                         _e(d.get("ayrinti")), _e(sayi(d.get("deger")) if not _bos(d.get("deger")) else "")])
    ozet = " · ".join(f'<span class="drm d-{k}">{DURUM[k][0]} {sayac[k]} {DURUM[k][1]}</span>'
                      for k in DURUM if sayac.get(k))
    kritik = [d.get("kontrol", "") for d in dn if d.get("durum", "atlandi") == "atlandi"
              and any(k in (d.get("kontrol") or "") for k in KRITIK_DENETIM)]
    if kritik:  # atlandı nötr değil: jüri "denetlendi" diye okumasın
        ozet = (f'<span class="drm d-uyari" title="{_e(", ".join(kritik))}">! kanıtlanmadı: sızıntı/ezber '
                f'({len(kritik)} kritik denetim atlandı)</span> · ' + ozet)
    return _bolum("Denetimler", f'<p class="denozet">{ozet}</p>' + _tablo(["Durum", "Kontrol", "Ayrıntı", "Değer"], satirlar),
                  "Sızıntı, kontaminasyon ve hile testleri. Uyarı ya da hata varsa sonuçlar ona göre okunmalı.")


# ------------------------------------------------------------------ 10. örnekler
def _tvd(a, b) -> float | None:
    try:
        return 0.5 * sum(abs(float(x) - float(y)) for x, y in zip(a, b))
    except (TypeError, ValueError):
        return None


def _ornek(o: _Ozet, orn: dict) -> str:
    sec = orn.get("secenekler") or [str(k + 1) for k in range(len(orn.get("gercek") or []))]
    gercek = orn.get("gercek") or []
    tah = list((orn.get("tahminler") or {}).items())[:3]
    seriler = [("Gerçek", "t-gercek", gercek)] + [(o.etiket(s), o.renk.get(s, "s1"), v) for s, v in tah]
    hi = max([0.1] + [float(x) for _, _, v in seriler for x in v if not _bos(x)])
    hi = math.ceil(hi / _adim(hi) - 1e-9) * _adim(hi)
    G, Y, sol, alt = 420, 190, 36, 34
    grup = (G - sol - 6) / max(len(sec), 1)
    gen = min(22, (grup - 10) / len(seriler) - 2)
    yy = lambda v: 8 + (1 - v / hi) * (Y - 8 - alt)
    ic = []
    for v, et in _izler(0, hi, yuzde):
        ic.append(f'<line class="izgara" x1="{sol}" x2="{G - 6}" y1="{yy(v):.1f}" y2="{yy(v):.1f}"/>'
                  f'<text class="eksen" x="{sol - 4}" y="{yy(v) + 4:.1f}" text-anchor="end">{_e(et)}</text>')
    for i, ad in enumerate(sec):
        gx = sol + i * grup + (grup - (gen + 2) * len(seriler)) / 2
        for j, (sad, renk, v) in enumerate(seriler):
            if i >= len(v) or _bos(v[i]):
                continue
            x, h = gx + j * (gen + 2), max(yy(0) - yy(float(v[i])), 0.5)
            ic.append(f'<g><title>{_e(ad)} · {_e(sad)}: {yuzde(v[i], 1)}</title>'
                      f'<rect class="cubuk {renk}" x="{x:.1f}" y="{yy(0) - h:.1f}" width="{gen:.1f}" height="{h:.1f}" rx="2"/></g>')
        ic.append(f'<text class="eksen" x="{sol + i * grup + grup / 2:.1f}" y="{Y - alt + 16}" text-anchor="middle">{_e(_kisalt(ad, 14))}</text>')
    tvd = " · ".join(f"{_e(o.etiket(s))}: 1−TVD {_e(sayi(1 - t if (t := _tvd(gercek, v)) is not None else None))}" for s, v in tah)
    bas = " · ".join(_e(x) for x in (orn.get("madde"), orn.get("segment")) if x)
    rz = f' <span class="rozet r-kontrol">{_e(orn.get("etiket"))}</span>' if orn.get("etiket") else ""
    return (f'<figure class="orkart"><figcaption>{bas}{rz}</figcaption>{_svg(G, Y, "".join(ic), f"Örnek {bas}")}'
            f'<p class="kalsayi">{tvd}</p></figure>')


def _ornekler(o: _Ozet) -> str:
    ol = o.oz.get("ornekler") or []
    if not ol:
        return ""
    sut = list(dict.fromkeys(s for orn in ol for s in list((orn.get("tahminler") or {}))[:3]))
    lej = _lejant([("Gerçek", "t-gercek")] + [(o.etiket(s), o.renk.get(s, "s1")) for s in sut])
    return _bolum("Örnek maddeler", lej + f'<div class="orizgara">{"".join(_ornek(o, x) for x in ol)}</div>',
                  "Seçenek başına gerçek insan dağılımı ve tahminler. En iyi ve en kötü örnekler birlikte: seçilmiş vitrin değil.")


# ------------------------------------------------------------------ 11-12. ön kayıt, sınırlar
def _onkayit(o: _Ozet) -> str:
    ok = o.oz.get("onkayit") or []
    if not ok:
        return ""
    tab = [[_e(k.get("dosya")), f'<code title="{_e(k.get("sha256"))}">{_e((k.get("sha256") or "")[:16])}…</code>',
            _e(k.get("zaman")), _e(k.get("aciklama"))] for k in ok]
    return _bolum("Ön kayıt", _tablo(["Dosya", "SHA-256", "Zaman", "Açıklama"], tab),
                  "Tahminler gerçek veri açılmadan önce hash'lendi ve zaman damgalandı: önce tahmin, sonra not. "
                  "Tam hash için imleci üzerine getirin.")


def _sinirlar(o: _Ozet) -> str:
    sn = o.oz.get("sinirlar") or []
    if not sn:
        return ""
    return _bolum("Sınırlar", '<ul class="sinir">' + "".join(f"<li>{_e(x)}</li>" for x in sn) + "</ul>",
                  "Bu panonun söylemediği şeyler.")


# ------------------------------------------------------------------ özet ve sayfa
KART_ROZET = {"yuksek": ("d-tamam", "yüksek güven"), "kararsiz": ("d-uyari", "kararsız"), "bekliyor": ("d-atlandi", "kanıt bekliyor")}


def _kanit_karti(o: _Ozet) -> str:
    """Sahnenin tek ekranlık kartıyla aynı satırlar (özet 'kanit_kart'): taban → ham → bizim → tavan → kapanan boşluk → bedel."""
    k = o.oz.get("kanit_kart")
    if not k:
        return ""
    sat = []
    for r in k.get("satirlar") or []:
        yz = r.get("birim") == "%"
        if r.get("deger") is None:
            deg = "—"
        elif r.get("birim") == "×":
            deg = f'<b>{_e(sayi(r["deger"], 2))}×</b> <span class="ga">{_e(ga(r.get("alt"), r.get("ust"), 2))}</span>'
        else:
            deg = f'<b>{_deger_ga(r, yz=yz)}</b>'
        sat.append([_e(r.get("etiket")), deg, _e(r.get("not") or "")])
    sinif, ad = KART_ROZET.get(k.get("rozet"), ("d-atlandi", k.get("rozet") or ""))
    et = "".join(f' <span class="drm d-uyari">{_e(e)}</span>' for e in k.get("etiketler") or [])
    return (f'<section class="bolum"><h3>Kanıt kartı <span class="drm {sinif}">{_e(ad)}</span>{et}</h3>'
            f'<p class="alt">{_e(k.get("rozet_gerekce") or "")}</p>'
            + _tablo(["Satır", "Değer [%95 GA]", "Not"], sat) + "</section>")


def _tahminler(o: _Ozet) -> str:
    """Gerçek bekleyen rapor: tahmin sütunlarının özeti (metrik yok)."""
    t = o.oz.get("tahmin_ozeti") or {}
    if not t:
        return ""
    sat = [[f'{_e(p.get("etiket"))} <code>{_e(c)}</code>', tam(p.get("n")), tam(p.get("eksik")), _e(sayi(p.get("ortalama"))),
            _e(sayi(p.get("sd"))), _e(sayi(p.get("en_kucuk"))), _e(sayi(p.get("medyan"))), _e(sayi(p.get("en_buyuk"))),
            _e(", ".join(f"{a} {yuzde(b)}" for a, b in (p.get("en_sik") or {}).items()))] for c, p in t.items()]
    return _bolum("Tahmin özeti (gerçek bekleniyor)",
                  _tablo(["Sütun", "n", "Eksik", "Ortalama", "Sd", "En küçük", "Medyan", "En büyük", "En sık"], sat, "sayisal"),
                  "Gerçek sonuç henüz yok: metrik, merdiven ve güven aralığı gerçek açılınca hesaplanır.")


def _ozet_html(oz: dict, i: int) -> str:
    o = _Ozet(oz, i)
    bas = oz.get("baslik") or oz.get("veri_adi") or f"Özet {i + 1}"
    govde = "".join(f(o) for f in (_kanit_karti, _tahminler, _manset, _orneklem, _merdiven, _metrik_tablosu, _farklar, _kalibrasyon, _segment,
                                   _yayilim, _denetim, _ornekler, _onkayit, _sinirlar))
    return f'<article class="ozet" id="ozet-{i + 1}"><h2>{_e(bas)}</h2>{_kart(o)}{govde}</article>'


def _ust_tablo(ozetler: list[dict]) -> str:
    satirlar = []
    for i, oz in enumerate(ozetler):
        m = oz.get("manset") or {}
        d = lambda k: _deger_ga(m.get(k)) if m.get(k) else "—"
        satirlar.append([f'<a href="#ozet-{i + 1}">{_e(oz.get("baslik") or oz.get("veri_adi") or f"Özet {i + 1}")}</a>',
                         _e(GOREV_ADI.get(oz.get("gorev"), oz.get("gorev"))), _e(_metrik_adi(oz.get("birincil", ""))),
                         d("en_iyi_baseline"), d("bizim"), d("tavan"),
                         _deger_ga(m["bosluk_kapanan"], yz=True) if m.get("bosluk_kapanan") else ("—" if m.get("tavan") else "tavan yok"),
                         _deger_ga(m["tavana_oran"], yz=True) if m.get("tavana_oran") else "—",
                         _carpan_100(oz)])
    return ('<section class="bolum"><h3>Veri setleri</h3>'
            + _tablo(["Veri seti", "Görev", "Birincil", "En güçlü taban", "Bizim", "İnsan tavanı", "Kapanan boşluk", "Tavana oran", "Çarpan @n=100"], satirlar, "sayisal")
            + "</section>")


_CSS = """
:root{--bg:#f9f9f7;--yuzey:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6d68;--izgara:#e1e0d9;--eksen:#c3c2b7;
--cizgi:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948;
--base:#8a8882;--tavan:#0b0b0b;--iyi:#006300;--iyi-bg:#e6f3e6;--uy:#8a5a00;--uy-bg:#fdf1d6;--kot:#b02a2a;--kot-bg:#fae6e6;
--not-bg:#f1f0ec;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--yuzey:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#a3a198;--izgara:#2c2c2a;--eksen:#383835;--cizgi:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--base:#8f8d86;--tavan:#f0efec;--iyi:#3cc23c;--iyi-bg:#132613;--uy:#fab219;
--uy-bg:#2b2410;--kot:#f07a7a;--kot-bg:#2e1515;--not-bg:#222220;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0d0d0d;--yuzey:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#a3a198;--izgara:#2c2c2a;--eksen:#383835;
--cizgi:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;
--base:#8f8d86;--tavan:#f0efec;--iyi:#3cc23c;--iyi-bg:#132613;--uy:#fab219;--uy-bg:#2b2410;--kot:#f07a7a;--kot-bg:#2e1515;
--not-bg:#222220;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;overflow-x:hidden}
.sayfa{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
header.ust{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}
h1{font-size:1.6rem;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:1.35rem;margin:0 0 12px;padding-top:8px}
h3{font-size:1.05rem;margin:0 0 4px}h4{font-size:.95rem;margin:18px 0 4px}
.alt{color:var(--ink2);font-size:.88rem;margin:0 0 10px}
#tema{background:var(--yuzey);color:var(--ink);border:1px solid var(--cizgi);border-radius:8px;padding:6px 12px;font:inherit;cursor:pointer;white-space:nowrap}
nav.atla{display:flex;flex-wrap:wrap;gap:6px 14px;margin:8px 0 16px;font-size:.9rem}
a{color:var(--s1)}
.ozet{border-top:2px solid var(--cizgi);margin-top:28px;padding-top:12px}
.bolum{background:var(--yuzey);border:1px solid var(--cizgi);border-radius:12px;padding:16px;margin:14px 0}
.kart{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px 16px;margin:0 0 8px}
.kart dt{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}.kart dd{margin:0;font-variant-numeric:tabular-nums}
.notlar{color:var(--ink2);font-size:.85rem;margin:4px 0 0;padding-left:18px}
.manset{margin:12px 0}.cumle{font-size:1.45rem;line-height:1.35;font-weight:600;margin:6px 0 14px;max-width:60ch}
.kutular{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:8px}
.kutu{background:var(--yuzey);border:1px solid var(--cizgi);border-radius:10px;padding:10px 12px;border-top:3px solid var(--base)}
.k-bizim{border-top-color:var(--s1)}.k-norm{border-top-color:var(--iyi);border-top-width:5px}.k-tavan{border-top-color:var(--tavan);border-top-style:dashed}.k-oran{border-top-color:var(--iyi)}
.kad{font-size:.78rem;color:var(--ink2)}.kdeger{font-size:1.7rem;font-weight:650;line-height:1.2}
.kalt{font-size:.75rem;color:var(--muted)}.ga{color:var(--muted);font-size:.85em;white-space:nowrap}
.ikili{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:16px;align-items:start}
@media (max-width:820px){.ikili{grid-template-columns:minmax(0,1fr)}.cumle{font-size:1.2rem}}
svg{width:100%;height:auto;display:block;font-family:inherit}
svg text{font-size:12px;fill:var(--ink2)}svg text.eksen,svg tspan.eksen{fill:var(--muted);font-size:11px}
svg text.etk{fill:var(--ink)}svg text.deg{fill:var(--ink2);font-size:11px;font-variant-numeric:tabular-nums}
.izgara{stroke:var(--izgara);stroke-width:1}.cerceve{fill:none;stroke:var(--eksen)}.bant{fill:var(--not-bg);opacity:.6}
.tavan{stroke:var(--tavan);stroke-width:1.5;stroke-dasharray:5 4;opacity:.75}
.biyik{stroke:var(--ink2);stroke-width:1.5}.cizgi{fill:none;stroke-width:2}
.nokta{stroke:var(--yuzey);stroke-width:2}.halka{fill-opacity:.85}
.t-baseline{fill:var(--base)}.t-kontrol{fill:var(--base)}.t-yontem{fill:var(--s1)}.t-kalibre{fill:var(--s7)}.t-tavan{fill:var(--tavan)}
.t-gercek{fill:var(--ink2)}
.s1{fill:var(--s1);stroke:var(--s1)}.s2{fill:var(--s2);stroke:var(--s2)}.s3{fill:var(--s3);stroke:var(--s3)}.s4{fill:var(--s4);stroke:var(--s4)}
.s5{fill:var(--s5);stroke:var(--s5)}.s6{fill:var(--s6);stroke:var(--s6)}.s7{fill:var(--s7);stroke:var(--s7)}.s8{fill:var(--s8);stroke:var(--s8)}
.nokta.s1,.nokta.s2,.nokta.s3,.nokta.s4,.nokta.s5,.nokta.s6,.nokta.s7,.nokta.s8{stroke:var(--yuzey)}
polyline.cizgi{fill:none}polyline.kuram{fill:none;stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:2 3}
.bos{fill:var(--yuzey);stroke-width:2.2}.biyik.ince{stroke-width:1;opacity:.45}.halo{paint-order:stroke;stroke:var(--yuzey);stroke-width:4px;stroke-linejoin:round}polyline.kesik{stroke-dasharray:6 4}
.lj.kesik{border-radius:2px;height:4px;width:14px;vertical-align:2px}.cubuk{stroke:none}
.lejant{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:.85rem;color:var(--ink2);margin:4px 0 6px}
.lj{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px;background:currentColor}
.lj.s1{color:var(--s1)}.lj.s2{color:var(--s2)}.lj.s3{color:var(--s3)}.lj.s4{color:var(--s4)}.lj.s5{color:var(--s5)}.lj.s6{color:var(--s6)}
.lj.s7{color:var(--s7)}.lj.s8{color:var(--s8)}.lj.t-gercek{color:var(--ink2);border-radius:2px}
.tkutu{overflow-x:auto;margin:8px 0;max-width:100%}
table{border-collapse:collapse;width:100%;font-size:.87rem}
th{text-align:left;font-weight:600;color:var(--ink2);border-bottom:1px solid var(--eksen);padding:6px 8px;white-space:nowrap}
td{border-bottom:1px solid var(--izgara);padding:6px 8px;vertical-align:top}
table.sayisal td{font-variant-numeric:tabular-nums;white-space:nowrap}
.eniyi{font-weight:700}
.rozet{display:inline-block;font-size:.7rem;border:1px solid var(--cizgi);border-radius:999px;padding:0 7px;color:var(--ink2);
background:var(--not-bg);vertical-align:1px;white-space:nowrap}
.r-yontem{border-color:var(--s1);color:var(--ink)}.r-tavan{border-style:dashed}
.drm{display:inline-block;font-weight:600;font-size:.82rem;border-radius:6px;padding:1px 7px;white-space:nowrap}
.d-tamam{color:var(--iyi);background:var(--iyi-bg)}.d-uyari{color:var(--uy);background:var(--uy-bg)}
.d-hata{color:var(--kot);background:var(--kot-bg)}.d-atlandi{color:var(--muted);background:var(--not-bg)}
.denozet{margin:4px 0 8px;display:flex;flex-wrap:wrap;gap:6px}
.kalizgara,.orizgara{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
.orizgara{grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}
figure{margin:0;background:var(--bg);border:1px solid var(--cizgi);border-radius:10px;padding:10px}
figcaption{font-weight:600;font-size:.88rem;margin-bottom:4px}.kalsayi{font-size:.8rem;color:var(--ink2);margin:4px 0 0}
.uyari-kutu{background:var(--not-bg);border-left:3px solid var(--uy);border-radius:6px;padding:8px 12px;margin:10px 0;font-size:.88rem}
.uyari-kutu ol{margin:4px 0 0;padding-left:20px}
.sinir{margin:0;padding-left:18px;color:var(--ink2)}
code{font-size:.85em}
footer{color:var(--muted);font-size:.8rem;margin-top:32px}
"""

_JS_BAS = ("(function(){try{var t=localStorage.getItem('kanit-tema');"
           "if(t==='dark'||t==='light')document.documentElement.setAttribute('data-theme',t);}catch(e){}})();")
_JS_SON = """(function(){var r=document.documentElement,b=document.getElementById('tema');
function koyu(){var t=r.getAttribute('data-theme');return t?t==='dark':window.matchMedia('(prefers-color-scheme: dark)').matches;}
function yaz(){b.textContent=koyu()?'☀ Açık tema':'☾ Koyu tema';}
b.addEventListener('click',function(){var y=koyu()?'light':'dark';r.setAttribute('data-theme',y);
try{localStorage.setItem('kanit-tema',y);}catch(e){}yaz();});yaz();})();"""


def html(ozetler: list[dict] | dict, baslik: str = "Kanıt panosu", bedel: dict | None = None) -> str:
    """Bir ya da birden çok özet → tek HTML metni (harici kaynak yok)."""
    if isinstance(ozetler, dict):
        ozetler = [ozetler]
    ozetler = [json.loads(json.dumps(o, default=str)) for o in ozetler]  # numpy vb. türleri düzle
    ust = ""
    if len(ozetler) > 1:
        bag = "".join(f'<a href="#ozet-{i + 1}">{_e(o.get("baslik") or o.get("veri_adi") or f"Özet {i + 1}")}</a>'
                      for i, o in enumerate(ozetler))
        ust = f'<nav class="atla" aria-label="Veri setleri">{bag}</nav>' + _ust_tablo(ozetler)
    ust = _bedel(json.loads(json.dumps(bedel, default=str)) if bedel else None) + ust
    govde = "".join(_ozet_html(o, i) for i, o in enumerate(ozetler))
    surum = next((o.get("surum") for o in ozetler if o.get("surum")), "")
    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(baslik)}</title><script>{_JS_BAS}</script><style>{_CSS}</style></head>
<body><div class="sayfa"><header class="ust"><div><h1>{_e(baslik)}</h1>
<p class="alt">Simülasyonun gerçek insan verisine karşı kanıtı: baseline merdiveni, güven aralıkları, kalibrasyon, segment hatası, denetimler.</p></div>
<button id="tema" type="button" aria-label="Temayı değiştir">Tema</button></header>
{ust}{govde}
<footer>kanit {_e(surum)} · bütün güven aralıkları küme (kişi) düzeyinde bootstrap, %95.</footer></div>
<script>{_JS_SON}</script></body></html>"""


def yaz(ozetler, yol, baslik: str = "Kanıt panosu", bedel: dict | None = None) -> Path:
    """Panoyu dosyaya yaz, yolu döndür."""
    p = Path(yol)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html(ozetler, baslik, bedel), encoding="utf-8")
    return p
