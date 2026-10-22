"""Otomatik denetimler: sızıntı, persona karıştırma, madde ortalaması, tavan aşımı, çöküş,
sınıf dengesizliği, kontaminasyon (kesim tarihi), pozisyon takası, seçim yanlılığı.

Her denetim: {"kontrol", "durum": tamam|uyari|hata|atlandi, "ayrinti", "deger"}.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .bootstrap import ga, p_degeri


def _k(kontrol, durum, ayrinti, deger=None):
    return {"kontrol": kontrol, "durum": durum, "ayrinti": ayrinti, "deger": deger}


def _f(x, b=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{b}f}".replace(".", ",")


def bolme(veri) -> list:
    out = []
    tam = veri.tam
    if "kat" in tam and "grup" in tam:  # çapraz doğrulama: aynı grup (kişi) tek katta olmalı
        cok = tam.groupby("grup")["kat"].nunique()
        k = int((cok > 1).sum())
        out.append(_k("grup-ayrık katlar (CV)", "hata" if k else "tamam",
                      f"{tam['kat'].nunique()} kat · {len(cok)} grup · {k} grup birden çok katta", k))
    if "bolum" not in tam:
        if out:
            return out
        out.append(_k("bölme bilgisi (eğitim/test)", "atlandi",
                      "girdide 'bolum' yok; tahminlerin hangi bölmeyle üretildiğini kaynak kod/protokol söylemeli"))
        return out
    test = tam["bolum"].astype(str).str.lower().isin(["test", "deger", "degerlendirme", "holdout"])
    if "grup" in tam:
        ortak = set(tam.loc[test, "grup"]) & set(tam.loc[~test, "grup"])
        out.append(_k("grup çakışması (eğitim ∩ test)", "hata" if ortak else "tamam",
                      f"{len(ortak)} ortak grup" + (f" (ör. {list(ortak)[:3]})" if ortak else ""), len(ortak)))
    else:
        out.append(_k("grup çakışması (eğitim ∩ test)", "atlandi", "'grup' sütunu yok: aynı kişinin satırları iki yana düşebilir"))
    if "zaman" in tam:
        z = pd.to_datetime(tam["zaman"], errors="coerce")
        e_max, t_min = z[~test].max(), z[test].min()
        if pd.notna(e_max) and pd.notna(t_min):
            ust = (z[~test] > t_min).mean()
            out.append(_k("zaman sızıntısı (eğitim testten sonra mı)", "uyari" if e_max > t_min else "tamam",
                          f"eğitim son {e_max.date()} · test ilk {t_min.date()} · testten sonraki eğitim payı %{100 * ust:.1f}", float(ust)))
    if "birim" in tam:
        dup = tam.loc[test, "birim"].duplicated().sum()
        out.append(_k("birim tekrarı (testte)", "uyari" if dup else "tamam", f"{dup} tekrar eden birim", int(dup)))
    return out


def fark_denetimi(kontrol, a, b, tekrar, nokta, metrik, esik_ad, ters_yon=False):
    """a − b farkı (metrik yönüne göre) GA ile; 0'dan anlamlı büyükse tamam."""
    if a not in nokta or b not in nokta or metrik.ad not in nokta[a] or metrik.ad not in nokta[b]:
        return _k(kontrol, "atlandi", "sütun yok")
    f = metrik.yon * (tekrar[a][metrik.ad] - tekrar[b][metrik.ad])
    nf = metrik.yon * (nokta[a][metrik.ad] - nokta[b][metrik.ad])
    alt, ust = ga(f)
    p = p_degeri(f)
    iyi = alt is not None and np.isfinite(alt) and alt > 0
    durum = "tamam" if iyi else "uyari"
    return _k(kontrol, durum, f"{metrik.ad}: {esik_ad} farkı {_f(nf)} [{_f(alt)}, {_f(ust)}], p={_f(p, 3)}", nf)


def tavan_asimi(nokta, tekrar, yontemler, metrik):
    if "tavan" not in nokta or metrik.ad not in nokta["tavan"] or not metrik.tavan:
        return _k("tavanı aşan yöntem (sızıntı işareti)", "atlandi", "tekrar ölçümü yok")
    asan = []
    for s in yontemler:
        if metrik.ad not in tekrar.get(s, {}):
            continue
        f = metrik.yon * (tekrar[s][metrik.ad] - tekrar["tavan"][metrik.ad])
        a, _ = ga(f)
        if a is not None and np.isfinite(a) and a > 0:
            asan.append(s)
    return _k("tavanı aşan yöntem (sızıntı işareti)", "uyari" if asan else "tamam",
              (f"insan tekrar ölçümünü anlamlı aşan: {asan}. Gürültüsüz gerçek tahmini tavanı aşabilir ama önce sızıntıyı ele"
               if asan else "hiçbir yöntem test-retest tavanını anlamlı aşmıyor"), len(asan))


def cokus(yayilim: dict, sutun: str, esik: float = 0.8, ust: float = 1.5):
    y = yayilim.get(sutun)
    if not y or y.get("sd_orani") is None:
        if y and y.get("entropi_orani") is not None:
            e = y["entropi_orani"]
            return _k("dağılım çöküşü (entropi oranı)", "uyari" if e < esik else "tamam",
                      f"entropi oranı {_f(e, 2)} (1 = insan kadar çeşitli)", e)
        return _k("dağılım çöküşü (sd oranı)", "atlandi", "hesaplanamadı")
    o = y["sd_orani"]
    # iki yönlü: <esik homojenleşme (LLM'in bilinen hastalığı), >ust aşırı yayılım (gürültüyü kişiye özgü sinyal sanmak)
    return _k("dağılım çöküşü (sd oranı)", "uyari" if o < esik or o > ust else "tamam",
              f"{sutun}: sd oranı {_f(o, 2)} [{_f(y.get('sd_orani_alt'), 2)}, {_f(y.get('sd_orani_ust'), 2)}]; "
              f"<{esik:.1f} homojenleşme, >{ust:.1f} aşırı yayılım", o)


def dengesizlik(veri):
    if veri.gorev not in ("ikili", "cok_sinif"):
        return None
    pay = veri.df["gercek"].astype(str).value_counts(normalize=True).iloc[0]
    return _k("sınıf dengesizliği", "uyari" if pay > 0.7 else "tamam",
              f"en sık sınıf payı %{100 * pay:.1f}" + ("; doğruluk yanıltıcı olabilir, kappa ve dengeli doğruluğa bak" if pay > 0.7 else ""),
              float(pay))


def kontaminasyon(veri, sutun_dizileri, metrik, kesim: str | None):
    """Model eğitim kesim tarihinden önceki ve sonraki birimlerde başarı farkı (ezber işareti)."""
    if not kesim or "zaman" not in veri.df:
        if not kesim and "zaman" in veri.df:
            ayr = ("'zaman' (time) sütunu var ama LLM eğitim kesim tarihi verilmedi: evaluate(..., cutoff=\"2025-03-01\") "
                   "ile kesim öncesi/sonrası karşılaştırılır")
        elif not kesim:
            ayr = "kesim tarihi (cutoff=) ve 'zaman' (time) sütunu yok"
        else:
            ayr = "'zaman' (time) sütunu yok"
        return _k("kontaminasyon (kesim öncesi/sonrası)", "atlandi", ayr)
    z = pd.to_datetime(veri.df["zaman"], errors="coerce")
    once = (z < pd.Timestamp(kesim)).to_numpy()
    if once.sum() < 30 or (~once).sum() < 30:
        return _k("kontaminasyon (kesim öncesi/sonrası)", "atlandi", f"öncesi {once.sum()}, sonrası {(~once).sum()} birim: az")
    # önce − sonra farkı, birim bootstrap'ı (iki yan ayrı yeniden örneklenir); alt GA > 0 → ezber şüphesi
    madde = veri.df["madde"].to_numpy() if "madde" in veri.df else None
    rng = np.random.default_rng(11)
    io, is_ = np.flatnonzero(once), np.flatnonzero(~once)
    parcalar, supheli = [], False
    for s, (y, p, maske) in sutun_dizileri.items():
        def m(ix):
            return metrik.deger(y[ix], p[ix], None if madde is None else madde[ix])
        f0 = metrik.yon * (m(io) - m(is_))
        fb = [metrik.yon * (m(rng.choice(io, len(io))) - m(rng.choice(is_, len(is_)))) for _ in range(200)]
        a, u = np.nanpercentile(fb, [2.5, 97.5])
        supheli |= bool(a > 0)
        parcalar.append(f"{s}: önce {_f(m(io))} / sonra {_f(m(is_))}, fark {_f(f0)} [{_f(a)}, {_f(u)}]")
    return _k("kontaminasyon (kesim öncesi/sonrası)", "uyari" if supheli else "tamam",
              f"kesim {kesim} · " + " · ".join(parcalar)
              + (" · kesimden önceki birimlerde anlamlı daha iyi: ezber şüphesi" if supheli else ""))


def pozisyon_takasi(veri):
    """ab/ikili çiftli görevlerde 'ters_<yontem>' sütunu (A/B sırası değiştirilmiş istem, aynı A-çerçevesinde)
    varsa tutarlılık ve tutarlı doğruluk (iki sırada da doğru)."""
    out = []
    for y_ in veri.yontemler:
        c = f"ters_{y_}"
        if c not in veri.df:
            continue
        p1 = veri.df["tahmin_" + y_].to_numpy(float)
        p2 = pd.to_numeric(veri.df[c], errors="coerce").to_numpy(float)
        m = np.isfinite(p2)
        y = veri.df["gercek"].to_numpy(float)[m]
        tut = ((p1[m] > 0.5) == (p2[m] > 0.5)).mean()
        dogru = (((p1[m] > 0.5) == (y > 0.5)) & ((p2[m] > 0.5) == (y > 0.5))).mean()
        out.append(_k(f"pozisyon takası ({y_})", "uyari" if tut < 0.8 else "tamam",
                      f"sıra değişince aynı kalan %{100 * tut:.1f}; tutarlı doğruluk %{100 * dogru:.1f} (n={m.sum()})", float(dogru)))
    if not out and veri.gorev in ("ab",):
        out.append(_k("pozisyon takası", "atlandi", "'ters_<yontem>' sütunu yok"))
    return out


def ezber(yol: str | None):
    if not yol:
        return _k("ezber testi (LLM veriyi biliyor mu)", "atlandi", "ezber testi sonucu verilmedi (kanit.ezber)")
    try:
        with open(yol) as f:
            r = json.load(f)
    except OSError as e:
        return _k("ezber testi (LLM veriyi biliyor mu)", "atlandi", f"okunamadı: {e}")
    durum = r.get("durum") if r.get("durum") in ("tamam", "uyari", "hata") else "atlandi"
    return _k("ezber testi (LLM veriyi biliyor mu)", durum, r.get("ayrinti", "sonuç dosyasında 'durum' yok"), r.get("deger"))
