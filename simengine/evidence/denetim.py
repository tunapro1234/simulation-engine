"""Otomatik denetimler: sızıntı, persona karıştırma, madde ortalaması, tavan aşımı, çöküş,
sınıf dengesizliği, kontaminasyon (kesim tarihi), pozisyon takası, seçim yanlılığı, taban belirsizliği,
madde başına sabit tahmin (sıra metrikleri tanımsız).

Her denetim: {"kontrol", "durum": tamam|uyari|hata|atlandi, "ayrinti", "deger"}.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from .bootstrap import ga, p_degeri


def _k(kontrol, durum, ayrinti, deger=None):
    return {"kontrol": kontrol, "durum": durum, "ayrinti": ayrinti, "deger": deger}


def _f(x, b=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{b}f}".replace(".", ",")


def _tarih_parca(s: pd.Series) -> pd.Series:
    """Tek biçim turu: pandas biçimi ilk dolu satırdan çıkarır (hızlı yol). ISO değilse gün-önce de denenir, daha çok
    satır çözen alınır (eşitlikte ay-önce, pandas varsayılanı); böylece "03/07/2023" ile "15/07/2023" aynı biçimle
    okunur. ISO satırına gün-önce uygulanmaz ("2023-07-01" -> 7 Ocak olurdu). Saat dilimli değerler UTC'ye çekilir."""
    import re
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # biçim çıkarım / dayfirst uyarıları
        a = pd.to_datetime(s, errors="coerce", utc=True)
        ilk = str(s.iloc[0]) if len(s) else ""
        if a.notna().all() or re.match(r"\s*\d{4}", ilk):
            return a
        b = pd.to_datetime(s, errors="coerce", utc=True, dayfirst=True)
    return b if b.notna().sum() > a.notna().sum() else a


def zaman_coz(s: pd.Series) -> tuple[pd.Series, int]:
    """'zaman' sütunu -> (saat dilimsiz datetime, çözülemeyen dolu satır sayısı). Karışık biçimli tarihler ("2024-03-05"
    ile "15/09/2024") biçim biçim çözülür: her turda kalan satırların ilk dolu değerinden biçim çıkarılır (hızlı,
    vektörel; format="mixed" satır satır dateutil'di, 200 bin satırda ~15x yavaş ve aynı sütunda gün/ay sırasını satıra
    göre değiştiriyordu). Saat dilimli değerler UTC'ye çekilip dilim düşer (kesim tarihi dilimsiz). Çözülemeyen satır
    NaT kalır ve sayılır; hiçbir tarafa (önce/sonra, eğitim/test) yazılmaz.

        z, n_bos = zaman_coz(veri.df["zaman"])"""
    if pd.api.types.is_datetime64_any_dtype(s):
        z = s if getattr(s.dt, "tz", None) is None else s.dt.tz_convert("UTC").dt.tz_localize(None)
        return z, 0
    z = np.full(len(s), np.datetime64("NaT", "ns"))
    kalan = s.notna().to_numpy()
    for _ in range(6):   # biçim başına bir tur; ilerleme yoksa dur
        if not kalan.any():
            break
        parca = _tarih_parca(s[kalan])
        ok = parca.notna().to_numpy()
        if not ok.any():
            break
        ix = np.flatnonzero(kalan)[ok]
        z[ix] = parca[ok].dt.tz_localize(None).to_numpy("datetime64[ns]")
        kalan[ix] = False
    return pd.Series(z, index=s.index, name=s.name), int(kalan.sum())


def zaman_cozumu(veri):
    """'zaman' sütununda çözülemeyen dolu satır varsa denetim satırı (yoksa None): bu satırlar zaman sızıntısı ve
    kontaminasyon denetimlerinin dışında kalır (önceden NaT 'kesimden sonra' sayılıyordu)."""
    if "zaman" not in veri.tam:
        return None
    _, n = zaman_coz(veri.tam["zaman"])
    if not n:
        return None
    return _k("zaman sütunu çözümü (time column parsing)", "uyari",
              f"{n} satırın tarihi çözülemedi; zaman sızıntısı ve kontaminasyon denetimlerinin dışında "
              f"(unparsed time: {n} rows left out of the time leakage and contamination audits)", n)


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
        z, _ = zaman_coz(tam["zaman"])
        e_max, t_min = z[~test].max(), z[test].min()
        if pd.notna(e_max) and pd.notna(t_min):
            ust = (z[~test & z.notna()] > t_min).mean()
            out.append(_k("zaman sızıntısı (eğitim testten sonra mı)", "uyari" if e_max > t_min else "tamam",
                          f"eğitim son {e_max.date()} · test ilk {t_min.date()} · testten sonraki eğitim payı "
                          f"%{100 * ust:.1f}", float(ust)))
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


# birincil bu metriklerden biriyse en iyi nokta tahmini koşullu medyandır: zayıf sinyalde medyana büzmek doğru hareket
MEDYAN_METRIK = ("mae", "bir_eksi_mae", "medae", "wmape")


def cokus(yayilim: dict, sutun: str, esik: float = 0.8, ust: float = 1.5, bernoulli_esik: float = 0.05,
          birincil: str | None = None):
    """Homojenleşme / aşırı yayılım. İkili olasılık tahmininde sd 0/1 gerçeğinkiyle kıyaslanmaz: sd(p) / √(p̄(1−p̄))
    yalnız neredeyse sabit tahmini (< bernoulli_esik) uyarır. Dağılımda entropi ve ayrışma oranları ayrıntıya eklenir.
    birincil MEDYAN_METRIK'ten biriyse düşük sd oranı uyarı değil bilgi notudur ("bilgi_notu": True); aşırı yayılım
    yine uyarır."""
    y = yayilim.get(sutun)
    if y and y.get("sd_bernoulli") is not None:
        o = y["sd_bernoulli"]
        return _k("dağılım çöküşü (Bernoulli sd oranı)", "uyari" if o < bernoulli_esik else "tamam",
                  f"{sutun}: olasılık tahmini, sd(p) / √(p̄(1−p̄)) = {_f(o, 2)} "
                  f"[{_f(y.get('sd_bernoulli_alt'), 2)}, {_f(y.get('sd_bernoulli_ust'), 2)}]; 0/1 gerçeğin sd'siyle kıyas "
                  f"uygulanamaz (0 = sabit tahmin, 1 = kesin 0/1); <{_f(bernoulli_esik, 2)} neredeyse sabit tahmin", o)
    if not y or y.get("sd_orani") is None:
        if y and y.get("entropi_orani") is not None:
            e = y["entropi_orani"]
            return _k("dağılım çöküşü (entropi oranı)", "uyari" if e < esik else "tamam",
                      f"entropi oranı {_f(e, 2)} (1 = insan kadar çeşitli)", e)
        return _k("dağılım çöküşü (sd oranı)", "atlandi", (y or {}).get("uygulanamaz") or "hesaplanamadı")
    o = y["sd_orani"]
    ek = ""
    if y.get("entropi_olcut"):   # dağılım: entropi (ortalamaların oranı) ve hücreler arası ayrışma
        ek = (f" · entropi oranı (ortalamaların oranı) {_f(y.get('entropi_orani'), 2)} "
              f"[{_f(y.get('entropi_orani_alt'), 2)}, {_f(y.get('entropi_orani_ust'), 2)}]")
        if y.get("ayrisma_orani") is not None:
            ek += (f" · gruplar arası ayrışma oranı {_f(y['ayrisma_orani'], 2)} "
                   f"[{_f(y.get('ayrisma_orani_alt'), 2)}, {_f(y.get('ayrisma_orani_ust'), 2)}]")
    # iki yönlü: <esik homojenleşme (LLM'in bilinen hastalığı), >ust aşırı yayılım (gürültüyü kişiye özgü sinyal sanmak)
    ayr = (f"{sutun}: sd oranı {_f(o, 2)} [{_f(y.get('sd_orani_alt'), 2)}, {_f(y.get('sd_orani_ust'), 2)}]; "
           f"<{esik:.1f} homojenleşme, >{ust:.1f} aşırı yayılım" + ek)
    if o < esik and birincil in MEDYAN_METRIK:
        r = _k("dağılım çöküşü (sd oranı)", "tamam",
               ayr + f" · bilgi: birincil {birincil} koşullu medyanı ödüllendirir, medyana büzmek beklenen davranış; "
                     f"homojenleşme uyarısı değil (info: shrinking toward the median is intended under {birincil})", o)
        r["bilgi_notu"] = True
        return r
    return _k("dağılım çöküşü (sd oranı)", "uyari" if o < esik or o > ust else "tamam", ayr, o)


SABIT_KONTROL = "sabit tahmin: sıra metrikleri tanımsız (constant prediction)"


def sabit_tahmin(veri, sutunlar: list, etiket: dict | None = None) -> tuple[dict | None, dict]:
    """Madde başına sabit tahmin (sd 0) ve tanımsız sıra/korelasyon metriği sayımı (sürekli / ikili görev).

    Sabit tahminli maddede Spearman / Pearson / madde_ici_kor NaN olur ve madde ortalamadan sessizce düşer. Madde
    yoksa bütün tablo tek madde sayılır; tek satırlı maddeler sayılmaz. Döner (denetim satırı | None, {sütun:
    {"sabit", "nan", "n_madde"}}); satır yalnız en az bir maddede sabit tahmin varsa ("deger" = en büyük sayı)."""
    d = veri.df
    if veri.gorev not in ("surekli", "ikili"):
        return None, {}
    y = pd.to_numeric(d["gercek"], errors="coerce").to_numpy(float)
    gruplar = [g for g in (d.groupby("madde").indices.values() if "madde" in d else [np.arange(len(d))]) if len(g) >= 2]
    ozet = {}
    for c in dict.fromkeys(sutunlar):
        if c not in d:
            continue
        p = pd.to_numeric(d[c], errors="coerce").to_numpy(float)
        sabit = nan = 0
        for g in gruplar:
            m = np.isfinite(p[g]) & np.isfinite(y[g])
            pi, yi = p[g][m], y[g][m]
            sab = len(pi) >= 2 and float(np.std(pi)) == 0.0
            sabit += sab
            nan += bool(len(pi) < 3 or sab or float(np.std(yi)) == 0.0)
        ozet[c] = {"sabit": int(sabit), "nan": int(nan), "n_madde": len(gruplar)}
    sab = {c: o for c, o in ozet.items() if o["sabit"]}
    if not sab:
        return None, ozet
    etiket = etiket or {}
    ayr = "; ".join(f"{etiket.get(c, c)}: {o['sabit']} madde sabit tahmin (sd 0), {o['nan']}/{o['n_madde']} maddede sıra/"
                    f"korelasyon metriği tanımsız" for c, o in sab.items())
    en = max(o["sabit"] for o in sab.values())
    return _k(SABIT_KONTROL, "uyari",
              ayr + f" · NaN maddeler madde ortalamasından (madde_ici_kor, madde başına Spearman) sessizce düşer "
                    f"(constant prediction: rank metrics undefined ({en} items); NaN items drop out of per-item means)",
              en), ozet


# eşik 0,5'te etiketten okunan, sınıf payına duyarlı metrikler (kappa / dengeli doğruluk düzeltmeli)
DOGRULUK_TURU = ("dogruluk", "macro_f1")


def _ikili_olasilik(veri) -> bool:
    for y_ in veri.yontemler:
        p = pd.to_numeric(veri.df.get("tahmin_" + y_), errors="coerce").to_numpy(float)
        p = p[np.isfinite(p)]
        if len(p) and ((p > 0) & (p < 1)).any():
            return True
    return False


def dengesizlik(veri, birincil: str | None = None):
    """Sınıf dengesizliği uyarısı yalnız doğruluk türü metrik söz konusuyken. birincil verilirse: doğruluk/F1 değilse
    'tamam' notu. Verilmezse ikili olasılık tahmininde 'tamam' notu (AUC, Brier, logloss, ECE sınıf payından
    yanıltılmaz); etiket tahmininde ve çok sınıflıda en sık sınıf > %70 uyarı."""
    if veri.gorev not in ("ikili", "cok_sinif"):
        return None
    pay = veri.df["gercek"].astype(str).value_counts(normalize=True).iloc[0]
    tablo = "; tablodaki doğruluk ve F1 (eşik 0,5) için kappa ve dengeli doğruluğa bak" if pay > 0.7 else ""
    if birincil is not None and birincil not in DOGRULUK_TURU:
        return _k("sınıf dengesizliği", "tamam",
                  f"en sık sınıf payı %{100 * pay:.1f}; birincil metrik {birincil} doğruluk türü değil" + tablo, float(pay))
    if birincil is None and veri.gorev == "ikili" and _ikili_olasilik(veri):
        return _k("sınıf dengesizliği", "tamam",
                  f"en sık sınıf payı %{100 * pay:.1f}; olasılık tahmini: AUC / Brier / logloss / ECE sınıf payından "
                  "yanıltılmaz" + tablo, float(pay))
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
    z, _ = zaman_coz(veri.df["zaman"])
    var = z.notna().to_numpy()       # tarihi çözülemeyen satır ne öncesi ne sonrası
    kt = pd.Timestamp(kesim)
    if kt.tz is not None:            # "2025-03-01T00:00Z" gibi dilimli kesim: z ile aynı (UTC, dilimsiz) ölçek
        kt = kt.tz_convert("UTC").tz_localize(None)
    once = (z < kt).to_numpy() & var
    sonra = var & ~once
    if once.sum() < 30 or sonra.sum() < 30:
        return _k("kontaminasyon (kesim öncesi/sonrası)", "atlandi",
                  f"öncesi {once.sum()}, sonrası {sonra.sum()} birim: az")
    # önce − sonra farkı, birim bootstrap'ı (iki yan ayrı yeniden örneklenir); alt GA > 0 → ezber şüphesi
    madde = veri.df["madde"].to_numpy() if "madde" in veri.df else None
    rng = np.random.default_rng(11)
    io, is_ = np.flatnonzero(once), np.flatnonzero(sonra)
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


def taban_belirsizligi(tekrar: dict, tabanlar: list, metrik: str, yon: float, etiket: dict | None = None,
                       bizim: str | None = None, nokta: dict | None = None, ga_yontemi: str = "percentile"):
    """En güçlü taban belirsiz mi: tabanlar arası seçim gürültüsü.

    tekrar[c][metrik]: her sütunun ortak (eşli) bootstrap tekrarları; en iyi taban (nokta tahminiyle) ile öbür her
    taban arasındaki eşli farkın GA'sı 0'ı içeriyorsa o taban en iyiyle eşittir. İkinci taban ayrışmıyorsa 'uyari':
    eşit küme, eşitlerin hepsine karşı en kötü durum bizim − taban (tekrar başına eşitlerin en iyisi, eşli GA) ve
    taban sayısıyla seçim gürültüsü notu. Ayrışıyorsa 'tamam'; 2'den az taban varsa None.

        den.append(taban_belirsizligi(tekrar, ["taban_B1", "taban_B2"], "mae", -1, etiket, bizim="tahmin_x"))

    Farklar metrik yönüne göre (yon × fark; + = ilk taraf iyi). Rozet ve manşet değişmez; yalnız denetim satırı."""
    etiket = etiket or {}
    t = [c for c in dict.fromkeys(tabanlar) if c != bizim and metrik in tekrar.get(c, {})]

    def v(c):
        if nokta and metrik in nokta.get(c, {}):
            return float(nokta[c][metrik])
        r = np.asarray(tekrar[c][metrik], float)
        return float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan

    t = [c for c in t if np.isfinite(v(c))]
    if len(t) < 2:
        return None
    t.sort(key=lambda c: -yon * v(c))
    en_iyi, ad = t[0], (lambda c: str(etiket.get(c, c)))
    kontrol = "en güçlü taban belirsiz mi (tabanlar arası seçim gürültüsü)"

    def fark(a, b):
        f = yon * (np.asarray(tekrar[a][metrik], float) - np.asarray(tekrar[b][metrik], float))
        nf = yon * (v(a) - v(b))
        al, us = ga(f, nokta=nf, yontem=ga_yontemi)
        return nf, al, us

    esit, ikinci = [en_iyi], None
    for c in t[1:]:
        nf, al, us = fark(en_iyi, c)
        ikinci = ikinci or (c, nf, al, us)
        if np.isfinite(al) and al <= 0:   # GA 0'ı içeriyor: en iyiden ayrışmıyor
            esit.append(c)
    c2, nf2, al2, us2 = ikinci
    ilk_iki = f"{ad(en_iyi)} − {ad(c2)} = {_f(nf2)} [{_f(al2)}, {_f(us2)}]"
    if len(esit) == 1:
        return _k(kontrol, "tamam", f"{metrik}: en güçlü taban {ad(en_iyi)} ikinciden ayrışıyor ({ilk_iki}; "
                                    f"strongest baseline separated from the runner-up) · {len(t)} taban", 1)
    adlar = ", ".join(ad(c) for c in esit)
    ayr = (f"{metrik}: en güçlü taban belirsiz: {adlar} GA içinde (strongest baseline uncertain: {adlar} within CI); "
           f"{ilk_iki}")
    satir = _k(kontrol, "uyari", ayr, len(esit))
    satir["esitler"] = [ad(c) for c in esit]
    if bizim and metrik in tekrar.get(bizim, {}) and np.isfinite(v(bizim)):
        # tekrar başına eşitlerin en iyisine karşı fark: seçimin kendisi de yeniden örneklenir (en kötü durum)
        b = np.asarray(tekrar[bizim][metrik], float)
        F = np.vstack([yon * (b - np.asarray(tekrar[c][metrik], float)) for c in esit])
        ok = np.isfinite(F).all(axis=0)
        fr = np.where(ok, np.min(np.where(np.isfinite(F), F, np.inf), axis=0), np.nan)
        nf = min(yon * (v(bizim) - v(c)) for c in esit)
        al, us = ga(fr, nokta=nf, yontem=ga_yontemi)
        tek = {c: fark(bizim, c) for c in esit}
        zayif = min(esit, key=lambda c: tek[c][1] if np.isfinite(tek[c][1]) else np.inf)
        satir["en_kotu"] = {"fark": nf, "alt": al, "ust": us, "en_dusuk_alt": ad(zayif)}
        satir["ayrinti"] += (f" · bizim − eşitlerin en kötüsü {_f(nf)} [{_f(al)}, {_f(us)}] (worst case ours − tied "
                             f"baselines, paired); tek tek en düşük alt sınır: {ad(zayif)} {_f(tek[zayif][0])} "
                             f"[{_f(tek[zayif][1])}, {_f(tek[zayif][2])}]")
    satir["ayrinti"] += (f" · {len(t)} taban arasından nokta tahminiyle en iyisini seçmek seçim gürültüsü taşır: rozet ve "
                         f"'en güçlü taban' satırını eşitlerin hepsine karşı oku (best of {len(t)} baselines picked by "
                         "point estimate; read the badge against every tied baseline)")
    return satir
