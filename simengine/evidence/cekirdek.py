"""degerlendir(): uzun tablo → özet sözlüğü (OZET-SEMA.md). Pano ve JSON bunu kullanır."""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import math
import warnings

import numpy as np
import pandas as pd

from . import analiz, denetim, kalibrasyon, merdiven
from . import kart as _kart
from .bootstrap import Orneklem, ga, hesapla, p_degeri
from .metrik import BIRINCIL, aile, diziler
from .sema import ON_EK, _normalle, _sinif_vektoru, _vektor, _vektor_mu, gercek_yok, hazirla

SURUM = "0.1"


def _diz(veri, sutun, kaynak: str | None = None):
    """(y, p, maske). 'tekrar' (ya da kaynak= ile verilen model/oracle tavan sütunu) eksik olabilir → maske."""
    d = veri.df
    if sutun != "tekrar" and kaynak is None:
        y, p = diziler(veri, sutun)
        return y, p, None
    kaynak = kaynak or "tekrar"
    if veri.gorev == "dagilim":
        maske = np.array([v is not None and not (np.isscalar(v) and pd.isna(v)) for v in d[kaynak]])
        y, _ = diziler(veri, "gercek")
        p = d[kaynak].to_numpy(object)
        if kaynak != "tekrar":  # tekrar hazirla'da çözüldü; model tavanı burada çözülür
            p = np.array([_normalle(_vektor(v)) if m else None for v, m in zip(p, maske)], dtype=object)
        return y, p, maske
    if veri.gorev == "cok_sinif":
        kod = {k: i for i, k in enumerate(veri.siniflar)}
        y, _ = diziler(veri, "gercek")
        s = d[kaynak]
        if kaynak != "tekrar" and _vektor_mu(s):
            maske = np.array([v is not None and not (np.isscalar(v) and pd.isna(v)) for v in s])
            K = len(veri.siniflar)
            p = np.vstack([_sinif_vektoru(_vektor(v), veri.siniflar) if m else np.full(K, 1 / K)
                           for v, m in zip(s, maske)])
            return y, p, maske
        p = s.map(lambda v: kod.get(v, kod.get(str(v), -1))).to_numpy()
        return y, p, p >= 0
    y = d["gercek"].to_numpy(float)
    p = pd.to_numeric(d[kaynak], errors="coerce").to_numpy(float)
    if veri.gorev == "ikili" and kaynak != "tekrar":
        p = np.clip(p, 0, 1)
    return y, p, np.isfinite(p)


def _temiz(x):
    if isinstance(x, dict):
        return {str(k): _temiz(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_temiz(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_temiz(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _teorik(veri, orneklem):
    """PPI++ kuramsal çarpanı 1/(1 − (1 − n/N)·ρ²); ρ madde içi (ortalaması çıkarılmış) korelasyon."""
    d = veri.df
    if veri.gorev not in ("ikili", "surekli"):
        return
    madde = pd.factorize(d["madde"])[0] if "madde" in d else np.zeros(len(d), int)
    say = np.bincount(madde)
    y = d["gercek"].to_numpy(float)
    ry = y - (np.bincount(madde, y) / say)[madde]
    N = float(np.mean(say))
    for c, oz in orneklem["yontemler"].items():
        f = d[c].to_numpy(float)
        rf = f - (np.bincount(madde, f) / say)[madde]
        rho = float(np.corrcoef(ry, rf)[0, 1]) if rf.std() > 0 and ry.std() > 0 else 0.0
        oz["rho"] = rho
        oz["teorik"] = [{"n": n, "carpan": 1 / (1 - (1 - n / N) * rho ** 2)} for n in orneklem["n_listesi"] if n < N]


def _carpan_verisi(veri, en_fazla, tohum):
    """Tek maddeli (kitle ortalaması) çarpan büyük tabloda yavaş: en_fazla satırlık sabit tohumlu alt örneklem.
    Döner (veri, not | None). Çok maddeli tabloda madde başına satır gerektiği için alt örneklem alınmaz."""
    d = veri.df
    tek = "madde" not in d or d["madde"].nunique() <= 1
    if not en_fazla or not tek or len(d) <= en_fazla:
        return veri, None
    ix = np.sort(np.random.default_rng([tohum, 2000]).choice(len(d), int(en_fazla), replace=False))
    alt = _dc.replace(veri, df=d.iloc[ix].reset_index(drop=True))
    return alt, (f"PPI++ çarpanı {len(d)} satırın sabit tohumlu (tohum={tohum}) {int(en_fazla)} satırlık rastgele alt "
                 f"örnekleminde hesaplandı (hız; kitle = alt örneklem, sonlu kitle düzeltmesi buna göre). Tam tablo için "
                 f"multiplier_max_rows=None.")


def _model_tavan_asimi(nokta, tekrar, yontemler, M):
    """Model/oracle tavanını anlamlı aşan yöntem: sızıntı değil, tavanın zayıf olduğunu gösterir."""
    asan = []
    for s in yontemler:
        if M.ad in tekrar.get(s, {}) and M.ad in tekrar.get("tavan", {}):
            a, _ = ga(M.yon * (tekrar[s][M.ad] - tekrar["tavan"][M.ad]))
            if np.isfinite(a) and a > 0:
                asan.append(s)
    return {"kontrol": "model/oracle tavanını aşan yöntem", "durum": "uyari" if asan else "tamam", "deger": len(asan),
            "ayrinti": (f"model/oracle tavanını anlamlı aşan: {asan}; tavan zayıf ya da yöntem tavanın görmediği bilgiyi "
                        "kullanıyor (sızıntıya bakın)" if asan else "hiçbir yöntem model/oracle tavanını anlamlı aşmıyor")}


def _f(x, b=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{b}f}".replace(".", ",")


GURULTU_METRIK = ("spearman", "pearson", "r2")


def _kor(a, b, sira: bool) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m], b[m]
    if sira:
        from scipy.stats import rankdata
        a, b = rankdata(a), rankdata(b)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _sb_tavan(r: float, r2: bool) -> float:
    """Yarı korelasyonu → Spearman-Brown güvenilirliği → tavan (√güvenilirlik; r2'de güvenilirliğin kendisi)."""
    if not np.isfinite(r) or r <= 0:
        return np.nan
    sb = 2 * r / (1 + r)
    return float(sb if r2 else np.sqrt(sb))


SPEARMAN_MC = 400   # örneklem gürültüsü tavanı: Spearman dönüşümü için Monte Carlo tekrar sayısı


def _vb_guvenilirlik(m, e) -> float:
    """Varyans bileşenleri: güvenilirlik = 1 − ort(s²_g/n_g) / var(grup ortalamaları), küçük örneklem düzeltmesiyle
    (normal kuramda E[1/var̂] = (G−1)/((G−3)·var), oran (G−3)/(G−1) ile düzeltilir). En az 4 grup gerekir."""
    m, e = np.asarray(m, float), np.asarray(e, float)
    G = len(m)
    if G < 4:
        return np.nan
    v = float(np.var(m, ddof=1))
    if not np.isfinite(v) or v <= 0:
        return np.nan
    return 1.0 - float(np.mean(e)) / v * (G - 3) / (G - 1)


def _vb_tavan(rel: float, r2: bool, spearman_orani: float = 1.0) -> float:
    """Güvenilirlik → tavan: r2'de güvenilirlik, korelasyonda √güvenilirlik × spearman_orani (Pearson'da 1)."""
    if not np.isfinite(rel) or rel <= 0:
        return np.nan
    return float(rel if r2 else min(1.0, np.sqrt(rel) * spearman_orani))


def _spearman_mc(rel: float, v: float, noise_sd, tohum) -> float:
    """Normal modelde gerçek değer t ~ N(0, rel·v), gözlenen ortalama m_g = t_g + N(0, gürültü_g²) iken beklenen
    Spearman(t, m) (Monte Carlo, sabit tohum). Grup boyları farklıysa (ör. tek satırlı gruplar) gürültü
    heteroskedastiktir ve sıra korelasyonu normal kuramın (6/π)·asin(ρ/2) değerinden ayrılır; bu yüzden simüle edilir."""
    from scipy.stats import rankdata
    rng = np.random.default_rng([abs(int(tohum or 0)), 7741])
    G = len(noise_sd)
    t = rng.normal(0.0, np.sqrt(rel * v), (SPEARMAN_MC, G))
    mm = t + rng.normal(0.0, 1.0, (SPEARMAN_MC, G)) * np.asarray(noise_sd, float)[None, :]
    rt, rm = rankdata(t, axis=1), rankdata(mm, axis=1)
    rt -= rt.mean(1, keepdims=True)
    rm -= rm.mean(1, keepdims=True)
    return float(np.mean((rt * rm).sum(1) / np.sqrt((rt ** 2).sum(1) * (rm ** 2).sum(1))))


def _gurultu_tavani(veri, ty: dict, birincil: str, orn) -> dict:
    """Örneklem gürültüsü tavanı (varyans bileşenleri, puanlanan TÜM gruplar üzerinden). Grup ortalaması m_g'nin
    gürültü varyansı s²_g / n_g (s²_g grup içi varyans, ddof=1; tek satırlı gruplarda havuzlanmış iç varyans);
    güvenilirlik = 1 − ort(s²_g/n_g) / var(m_g) · (G−3)/(G−1) ≈ var(gerçek değer) / var(m_g). Tavan: Pearson
    √güvenilirlik, r2 güvenilirlik, Spearman normal modelde beklenen Spearman(gerçek, gözlenen ortalama) (Monte Carlo,
    gerçek grup boylarıyla; _spearman_mc), tekrarlarda √güvenilirlik ile aynı oranda ölçeklenir. Bootstrap tekrarı b aynı
    küme yeniden örneğinde yeniden hesaplanır (bizim yöntemle eşleşik). Split-half (R rastgele grup içi bölme,
    Spearman-Brown) yalnız çapraz denetim olarak raporlanır: küçük/tek sayılı gruplarda (1'e 2 bölme) ve tek satırlı
    grupları dışarıda bıraktığı için aşağı yanlıdır, tavan olarak kullanılmaz."""
    if birincil not in GURULTU_METRIK:
        raise NotImplementedError(
            f"noise_ceiling (sampling noise) is defined for correlation-type primary metrics {GURULTU_METRIK}; "
            f"the primary metric is {birincil!r}. For error metrics (mae, rmse) a sampling-noise floor is not "
            "implemented: pass primary='spearman' / 'pearson' / 'r2', or use retest / ceiling_col.")
    poz = veri.df[ty["sutun"]].to_numpy(int)
    sira, r2 = birincil == "spearman", birincil == "r2"
    # ---- varyans bileşenleri (tavan)
    m = pd.to_numeric(veri.df["gercek"], errors="coerce").to_numpy(float)
    n = np.asarray(ty["n"], float)[poz]
    s2 = np.asarray(ty["s2"], float)[poz]
    cok = n >= 2
    df_ = (n[cok] - 1).sum()
    havuz = float(((n[cok] - 1) * s2[cok]).sum() / df_) if df_ > 0 else np.nan
    s2e = np.where(cok, s2, havuz)
    e = s2e / np.where(n > 0, n, np.nan)
    rel = _vb_guvenilirlik(m, e)
    oran = 1.0
    if sira and np.isfinite(rel) and rel > 0 and np.isfinite(havuz):
        # Spearman tavanı: modelde beklenen Spearman / √güvenilirlik; tekrarlarda aynı oran kullanılır
        oran = _spearman_mc(rel, float(np.var(m, ddof=1)), np.sqrt(havuz / n), ty.get("tohum")) / np.sqrt(rel)
    tav = _vb_tavan(rel, r2, oran)
    # ---- split-half çapraz denetimi (raporlanır, kullanılmaz)
    A, Bm = np.asarray(ty["A"], float)[:, poz], np.asarray(ty["B"], float)[:, poz]
    R = A.shape[0]
    rs = np.array([_kor(A[s], Bm[s], sira) for s in range(R)])
    r = float(np.nanmean(rs)) if np.isfinite(rs).any() else np.nan
    her = np.array([_sb_tavan(x, r2) for x in rs])
    gecerli = np.isfinite(A) & np.isfinite(Bm)
    n_tek = int((n == 1).sum())
    olcek = "guvenilirlik" if r2 else ("spearman_mc" if sira else "karekok_guvenilirlik")
    out = {"yontem": "varyans_bilesenleri", "korelasyon": "spearman" if sira else "pearson",
           "guvenilirlik": rel if np.isfinite(rel) else None,
           "spearman_orani": oran if sira else None,
           "tavan": tav if np.isfinite(tav) else None, "olcek": olcek,
           "n_grup": int(len(poz)), "n_satir": int(len(poz)), "n_tek_satir": n_tek,
           "ic_varyans_havuz": havuz if np.isfinite(havuz) else None,
           "ortalama_grup_boyu": ty.get("ortalama_grup_boyu"), "tohum": ty.get("tohum"),
           # split-half çapraz denetimi
           "R": int(R), "yari_korelasyon": r,
           "spearman_brown": (2 * r / (1 + r)) if np.isfinite(r) and r > -1 else None,
           "yari_tavan": _sb_tavan(r, r2) if np.isfinite(_sb_tavan(r, r2)) else None,
           "n_grup_yari": int(gecerli[0].sum()) if R else 0,
           "bolme_yayilimi": [float(x) for x in np.nanpercentile(her, [2.5, 97.5])] if np.isfinite(her).sum() >= 10 else None}
    notlar = []
    if n_tek:
        notlar.append(f"{n_tek} tek satırlı grup puanlandı: gürültüleri havuzlanmış iç varyansla (s² = {_f(havuz)}) "
                      "alındı; split-half denetimi bu grupları dışarıda bırakır")
    if not np.isfinite(havuz):
        notlar.append("en az 2 satırlı grup yok: grup içi varyans tahmin edilemedi")
    if not np.isfinite(tav):
        notlar.append(f"güvenilirlik {_f(rel)} ≤ 0 ya da hesaplanamadı (en az 4 grup ve 2 satırlı grup gerekir)")
        out["not"] = "; ".join(notlar)
        return out
    if notlar:
        out["not"] = "; ".join(notlar)
    rep_ = np.full(orn.B_toplu, np.nan)
    for b in range(orn.B_toplu):
        idx, _ = orn.satirlar(b)
        rep_[b] = _vb_tavan(_vb_guvenilirlik(m[idx], e[idx]), r2, oran)
    out["_tekrar"] = rep_
    return out


def _gurultu_tavan_asimi(nokta, tekrar, yontemler, M):
    """Örneklem gürültüsü tavanını anlamlı aşan yöntem: gürültülü grup ortalamasına beklenenden yakın, sızıntı işareti."""
    asan = []
    for s in yontemler:
        if M.ad in tekrar.get(s, {}) and M.ad in tekrar.get("tavan", {}):
            a, _ = ga(M.yon * (tekrar[s][M.ad] - tekrar["tavan"][M.ad]))
            if np.isfinite(a) and a > 0:
                asan.append(s)
    return {"kontrol": "örneklem gürültüsü tavanını aşan yöntem (sızıntı işareti)", "durum": "uyari" if asan else "tamam",
            "deger": len(asan),
            "ayrinti": (f"örneklem gürültüsü tavanını anlamlı aşan: {asan}; gerçek grup değerini bilen bir tahmin edici bile "
                        "gürültülü ortalamayla bu kadar uyuşmaz: tahminde aynı satırların gerçeği (sızıntı) olabilir"
                        if asan else "hiçbir yöntem örneklem gürültüsü tavanını anlamlı aşmıyor")}


def degerlendir(df: pd.DataFrame, baslik: str, veri_adi: str = "", gorev: str | None = None,
                yontemler: list[str] | None = None, bizim: str | None = None, ham: str | None = None,
                etiketler: dict | None = None, birincil: str | None = None, B: int = 1000,
                B_toplu: int = 200, tohum: int = 0, kalibre: list[str] | None = None,
                kesim: str | None = None, ezber_json: str | None = None, carpan: bool = True,
                sinirlar: list[str] | None = None, onkayit: list | None = None,
                siniflar: list | None = None, baseline: bool = True,
                carpan_yontemler: list[str] | None = None, dongusel: list[str] | None = None,
                duzey: str | None = None, girdi_kaynagi: str | None = None,
                etiketsiz_tabanlar: list[str] | None = None, oof_tabanlar: list[str] | None = None,
                tavan_sutun: str | None = None, tavan_etiket: str | None = None,
                carpan_en_fazla: int | None = 2000, tavan_yarim: dict | None = None, kor: str | None = None,
                kiyaslar: list | None = None, ezber_llm: bool | None = None) -> dict:
    """bizim/ham: tahmin_ ön eki olmadan yöntem adı. bizim verilmezse SON yöntem (önceden belirlenmeli;
    en iyiyi sonradan seçmek iyimser yanlılık yaratır, denetimde işaretlenir).

    oof_tabanlar: hedef etiketiyle eğitilmiş ama çapraz uydurmalı (kat dışı, OOF) tahmin sütunları; etiketsiz
    tabanlar gibi taban basamağı olur. Rozet ve manşet en güçlü tabana (otomatik + bildirilen) karşı; hiç bildirilmiş
    taban yokken en güçlüsü rastgele/marjinal ise kart "yalnız basit tabanlar" der.
    tavan_sutun: df'deki model/oracle tavan sütunu (tahmin_ ön eksiz); verilirse insan tekrar tavanının yerine geçer.
    carpan_en_fazla: tek maddeli (kitle ortalaması) çarpan bu kadar satırdan büyükse sabit tohumlu alt örneklemde
    hesaplanır (None: alt örneklem yok).
    tavan_yarim: örneklem gürültüsü tavanı girdisi {"n", "s2": grup boyu / iç varyans, "A", "B": R × n yarı
    ortalamaları (split-half denetimi), "sutun": satır no sütunu, "R"}; api.evaluate(noise_ceiling=...) kurar. kor: kör (adsız) kol; bizim − kör eşleşik satırı.
    kiyaslar: [(a, b | None, etiket)] kayıtlı eşleşik karşılaştırmalar (düşme koşulları). ezber_llm: bizim yöntem LLM
    kolu içeriyor mu (None: ezber sonucu verildiyse evet sayılır); ezber uyarısında rozet gerekçesine ek düşer."""
    if gercek_yok(df):
        return bekleyen(df, baslik, veri_adi, gorev=gorev, yontemler=yontemler, bizim=bizim, ham=ham,
                        etiketler=etiketler, birincil=birincil, onkayit=onkayit, sinirlar=sinirlar, duzey=duzey,
                        girdi_kaynagi=girdi_kaynagi, tohum=tohum, B=B)
    veri = hazirla(df, gorev, yontemler, siniflar)
    d = veri.df
    etiketler = dict(etiketler or {})
    madde_var = "madde" in d
    olasilik = veri.gorev != "cok_sinif" or isinstance(d[ON_EK + veri.yontemler[0]].iloc[0], np.ndarray)
    birim_aralik = veri.gorev == "surekli" and d["gercek"].between(0, 1).all()
    metrikler = aile(veri.gorev, olasilik, madde_var and veri.gorev in ("ikili", "surekli"), birim_aralik)
    adlar = [m.ad for m in metrikler]
    dengesiz_not = None
    if not birincil and veri.gorev in ("ikili", "cok_sinif"):
        pay = d["gercek"].astype(str).value_counts(normalize=True).max()
        if pay > 0.6:  # dengesiz sınıfta doğruluk "hep çoğunluk"u ödüllendirir
            birincil = "macro_f1"
            dengesiz_not = (f"En sık sınıf payı %{100 * pay:.0f} (> %60): birincil metrik otomatik olarak macro-F1 seçildi; "
                            "doğruluk tabloda duruyor.")
    birincil = birincil or ("bir_eksi_mae" if birim_aralik else BIRINCIL[veri.gorev])
    if birincil not in adlar:
        raise ValueError(f"birincil metrik {birincil} bu görevde yok: {adlar}")
    M = {m.ad: m for m in metrikler}[birincil]
    bizim_sec_oto = bizim is None
    bizim = bizim or veri.yontemler[-1]
    agirlik = d["agirlik"].to_numpy(float) if "agirlik" in d else None

    # ---- sütunlar: baseline, yöntem, kalibre, kontrol, tavan
    turler, etiket = {}, {}
    if baseline:
        ozet_s = "median" if birincil in ("mae", "bir_eksi_mae") else "mean"
        for ad, (et, deg) in merdiven.baselineler(veri, ozet_s, tohum).items():
            c = f"taban_{ad}"
            d[c] = deg if not isinstance(deg, list) else pd.Series(deg, dtype=object).to_numpy()
            turler[c], etiket[c] = "baseline", et
    taban_turu = {c: "otomatik" for c in turler}
    for y_ in veri.yontemler:
        c = ON_EK + y_
        # etiketsiz_tabanlar: hedef etiketi görmeyen basit kurallar (ör. kalıcılık) taban sayılır;
        # oof_tabanlar: etiketle eğitilmiş ama çapraz uydurmalı (kat dışı) tabanlar
        if y_ in (etiketsiz_tabanlar or []):
            turler[c], taban_turu[c] = "baseline", "etiketsiz"
        elif y_ in (oof_tabanlar or []):
            turler[c], taban_turu[c] = "baseline", "oof"
        else:
            turler[c] = "yontem"
        etiket[c] = etiketler.get(y_, y_)
    bilinmeyen = [y_ for y_ in list(etiketsiz_tabanlar or []) + list(oof_tabanlar or []) if y_ not in veri.yontemler]
    if bilinmeyen:
        warnings.warn(f"declared baseline column(s) not in the table, ignored: {bilinmeyen} (need pred_<name>)",
                      UserWarning, stacklevel=3)
    if not bizim_sec_oto and taban_turu.get(ON_EK + bizim, "otomatik") != "otomatik":
        warnings.warn(f"'ours' ({bizim}) is also declared as a baseline", UserWarning, stacklevel=3)
    kalibre = kalibre if kalibre is not None else [bizim] + ([ham] if ham else [])
    kalib_sutun = {}
    if veri.gorev != "ab" and olasilik:
        for y_ in kalibre:
            for ek, deg in kalibrasyon.yeniden(veri, ON_EK + y_).items():
                c = f"kalib_{y_}+{ek}"
                d[c] = deg if not isinstance(deg, list) else pd.Series(deg, dtype=object).to_numpy()
                turler[c], etiket[c] = "kalibre", f"{etiketler.get(y_, y_)} + {ek}"
                kalib_sutun.setdefault(ON_EK + y_, []).append(c)
    kontrol_sutun = {}
    for y_ in dict.fromkeys([bizim] + ([ham] if ham else [])):
        c = f"kontrol_karistir_{y_}"
        d[c] = merdiven.karistir(veri, ON_EK + y_, tohum + 1)
        turler[c], etiket[c] = "kontrol", f"{etiketler.get(y_, y_)} · karıştırılmış"
        kontrol_sutun[ON_EK + y_] = c
    sutunlar = list(turler)
    girdiler = {c: _diz(veri, c) for c in sutunlar}
    tavan_turu = None
    if tavan_sutun:
        if tavan_sutun not in d:
            raise ValueError(f"tavan sütunu tabloda yok: {tavan_sutun}")
        girdiler["tavan"] = _diz(veri, "tavan", kaynak=tavan_sutun)
        tavan_turu = "model"
        turler["tavan"], etiket["tavan"] = "tavan", f"Model/oracle tavanı ({tavan_etiket or tavan_sutun})"
        tavan_metrik = list(metrikler)  # model tavanı bir tahmindir: her metrik anlamlı
        if "tekrar" in d:
            veri.notlar.append("tavan: model/oracle sütunu kullanıldı; 'tekrar' (insan test-retest) tavanı yok sayıldı")
    elif tavan_yarim is not None:
        tavan_turu = "orneklem_gurultusu"
        turler["tavan"], etiket["tavan"] = "tavan", "Örneklem gürültüsü tavanı (varyans bileşenleri)"
        tavan_metrik = []
        if "tekrar" in d:
            veri.notlar.append("tavan: örneklem gürültüsü (varyans bileşenleri) kullanıldı; 'tekrar' (insan test-retest) tavanı yok sayıldı")
    elif "tekrar" in d:
        girdiler["tavan"] = _diz(veri, "tekrar")
        tavan_turu = "insan_tekrar"
        turler["tavan"], etiket["tavan"] = "tavan", "İnsan tavanı (test-retest)"
        tavan_metrik = [m for m in metrikler if m.tavan]
    else:
        tavan_metrik = []
    tavan_adi = {"model": "model/oracle tavanı", "orneklem_gurultusu": "örneklem gürültüsü tavanı"}.get(tavan_turu, "insan tavanı")

    # ---- bootstrap (eşleşik)
    madde_kod = pd.factorize(d["madde"])[0] if madde_var else None
    orn = Orneklem(veri.kume, B=B, B_toplu=B_toplu, tohum=tohum, madde_kume=veri.kume_adi == "madde")
    tavan_girdi = {"tavan": girdiler.pop("tavan")} if "tavan" in girdiler else {}
    nokta, tekrar = hesapla(orn, girdiler, metrikler, madde_kod, agirlik)
    if tavan_girdi:
        n2, t2 = hesapla(orn, tavan_girdi, tavan_metrik, madde_kod, agirlik)
        nokta.update(n2)
        tekrar.update(t2)
        girdiler.update(tavan_girdi)
    gurultu = None
    if tavan_turu == "orneklem_gurultusu":
        gurultu = _gurultu_tavani(veri, tavan_yarim, birincil, orn)
        if gurultu["tavan"] is not None:
            nokta["tavan"] = {birincil: gurultu["tavan"]}
            tekrar["tavan"] = {birincil: gurultu.pop("_tekrar")}
            girdiler["tavan"] = (None, None, None)
            if gurultu.get("not"):
                veri.notlar.append(f"örneklem gürültüsü tavanı: {gurultu['not']}")
        else:
            gurultu.pop("_tekrar", None)
            turler.pop("tavan")
            etiket.pop("tavan")
            veri.notlar.append(f"örneklem gürültüsü tavanı tanımsız: {gurultu['not']}")

    sonuclar = {}
    for c in turler:
        sonuclar[c] = {"tur": turler[c], "etiket": etiket[c], "metrikler": {}}
        for m in metrikler:
            if m.ad in nokta.get(c, {}):
                a, u = ga(tekrar[c][m.ad])
                n = int(len(d) if girdiler[c][2] is None else girdiler[c][2].sum())
                sonuclar[c]["metrikler"][m.ad] = {"deger": nokta[c][m.ad], "alt": a, "ust": u, "n": n}

    def deg(c):
        return nokta.get(c, {}).get(birincil, np.nan)

    def fark(a, b, met=birincil):
        f = tekrar[a][met] - tekrar[b][met]
        al, us = ga(f)
        return nokta[a][met] - nokta[b][met], al, us, p_degeri(f)

    # ---- merdiven
    yontem_c = [ON_EK + y for y in veri.yontemler if turler[ON_EK + y] != "baseline"]
    # merdivende BÜTÜN tahmin sütunları (eskiden > 6 yöntemde yalnız ham + bizim + bizi geçenler kalıyordu)
    secili = yontem_c
    # persona-swap: bizim yöntemin karıştırılmış (başka kişinin bilgisiyle) sürümü, tabanlarla yöntemler arasında
    adim = [c for c in turler if turler[c] == "baseline"] + \
        ([kontrol_sutun[ON_EK + bizim]] if ON_EK + bizim in kontrol_sutun else []) + secili
    adim = [c for c in adim if np.isfinite(deg(c))]
    adim.sort(key=lambda c: M.yon * deg(c))
    if "tavan" in nokta and birincil in nokta["tavan"]:
        adim.append("tavan")
    merd = []
    for i, c in enumerate(adim):
        a, u = ga(tekrar[c][birincil])
        r = {"sutun": c, "etiket": etiket[c], "tur": turler[c], "deger": deg(c), "alt": a, "ust": u,
             "fark_onceki": None, "fark_alt": None, "fark_ust": None, "p": None}
        if i:
            r["fark_onceki"], r["fark_alt"], r["fark_ust"], r["p"] = fark(c, adim[i - 1])
        merd.append(r)

    # ---- manşet
    bc = ON_EK + bizim
    tabanlar = [c for c in turler if turler[c] == "baseline" and np.isfinite(deg(c))]
    en_iyi_b = max(tabanlar, key=lambda c: M.yon * deg(c)) if tabanlar else None
    bildirilen = [c for c in tabanlar if taban_turu.get(c) in ("etiketsiz", "oof")]
    # hiç taban bildirilmediyse ve en güçlüsü rastgele/marjinal ise rozet yalnız basit bir tabana karşıdır: açıkça yazılır
    yalniz_basit = bool(en_iyi_b) and not bildirilen and en_iyi_b in ("taban_rastgele", "taban_marjinal")
    rakipler = [c for c in yontem_c if c != bc and np.isfinite(deg(c))]
    en_iyi_r = max(rakipler, key=lambda c: M.yon * deg(c)) if rakipler else None

    def kutu(c):
        if c is None or c not in nokta:
            return None
        a, u = ga(tekrar[c][birincil])
        return {"sutun": c, "etiket": etiket[c], "deger": deg(c), "alt": a, "ust": u}
    manset = {"ham": kutu(ON_EK + ham) if ham else None, "bizim": kutu(bc), "en_iyi_baseline": kutu(en_iyi_b),
              "en_iyi_rakip": kutu(en_iyi_r), "taban_tavana_oran": None,
              "tavan": kutu("tavan") if "tavan" in nokta and birincil in nokta["tavan"] else None,
              "tavana_oran": None, "bosluk_kapanan": None}
    if manset["tavan"]:
        tb, tt = tekrar[bc][birincil], tekrar["tavan"][birincil]
        oran_r = tb / tt if M.yon > 0 else tt / tb
        oran_n = deg(bc) / deg("tavan") if M.yon > 0 else deg("tavan") / deg(bc)
        a, u = ga(oran_r)
        manset["tavana_oran"] = {"deger": oran_n, "alt": a, "ust": u}
        if en_iyi_b:
            tbb = tekrar[en_iyi_b][birincil]
            kap_r = (tb - tbb) / (tt - tbb)
            kap_n = (deg(bc) - deg(en_iyi_b)) / (deg("tavan") - deg(en_iyi_b))
            a, u = ga(kap_r)
            manset["bosluk_kapanan"] = {"deger": kap_n, "alt": a, "ust": u}
            manset["taban_tavana_oran"] = deg(en_iyi_b) / deg("tavan") if M.yon > 0 else deg("tavan") / deg(en_iyi_b)
    if manset["tavan"]:
        tt = tekrar["tavan"][birincil]
        manset["tabanlar_normalize"] = []
        for c in sorted(tabanlar, key=lambda c: M.yon * deg(c)):
            if abs(deg("tavan") - deg(c)) < 1e-12:
                continue
            a, u = ga((tekrar[bc][birincil] - tekrar[c][birincil]) / (tt - tekrar[c][birincil]))
            manset["tabanlar_normalize"].append({"taban": c, "etiket": etiket[c], "taban_deger": deg(c),
                                                 "deger": (deg(bc) - deg(c)) / (deg("tavan") - deg(c)),
                                                 "alt": a, "ust": u})
    # manşet: en güçlü taban ve en güçlü diğer yöntemle; normalize skor (m − b)/(tavan − b) önce, tavana oran sonra
    parca = []
    if manset["en_iyi_baseline"]:
        parca.append(f"en güçlü taban ({manset['en_iyi_baseline']['etiket']}) {_f(manset['en_iyi_baseline']['deger'])}")
    if manset["en_iyi_rakip"]:
        parca.append(f"en güçlü diğer yöntem ({manset['en_iyi_rakip']['etiket']}) {_f(manset['en_iyi_rakip']['deger'])}")
    elif manset["ham"]:
        parca.append(f"{manset['ham']['etiket']} {_f(manset['ham']['deger'])}")
    parca.append(f"{etiket[bc]} {_f(deg(bc))} [{_f(manset['bizim']['alt'])}–{_f(manset['bizim']['ust'])}]")
    if manset["tavan"]:
        parca.append(f"{tavan_adi} {_f(manset['tavan']['deger'])}")
    cumle = f"{birincil}: " + " · ".join(parca)
    if manset["bosluk_kapanan"]:
        k_ = manset["bosluk_kapanan"]
        cumle += (f" → normalize skor (tabandan tavana kapanan pay) %{100 * k_['deger']:.0f} "
                  f"[{100 * k_['alt']:.0f}–{100 * k_['ust']:.0f}]")
        o = manset["tavana_oran"]
        cumle += (f"; tavana oran %{100 * o['deger']:.0f} (taban tek başına %{100 * manset['taban_tavana_oran']:.0f})")
    elif not manset["tavan"]:
        cumle += " · tavan yok (test-retest ölçülmemiş): normalize skor hesaplanamaz"
    manset["cumle"] = cumle
    if manset.get("tabanlar_normalize"):
        manset["cumle_tabanlar"] = "normalize skor tabana göre: " + " · ".join(
            f"{t['etiket']} %{100 * t['deger']:.0f} [{100 * t['alt']:.0f}–{100 * t['ust']:.0f}]"
            for t in manset["tabanlar_normalize"])

    # normalize skor her sütun için: (m − b)/(tavan − b), b = en güçlü taban; GA eşleşik tekrarlardan
    if manset["tavan"] and en_iyi_b:
        tbb, tt = tekrar[en_iyi_b][birincil], tekrar["tavan"][birincil]
        for c in sonuclar:
            if c != "tavan" and birincil in tekrar.get(c, {}):
                a, u = ga((tekrar[c][birincil] - tbb) / (tt - tbb))
                sonuclar[c]["normalize"] = {"deger": (deg(c) - deg(en_iyi_b)) / (deg("tavan") - deg(en_iyi_b)),
                                            "alt": a, "ust": u, "taban": en_iyi_b}
        for r in merd:
            if r["sutun"] in sonuclar and "normalize" in sonuclar[r["sutun"]]:
                r["normalize"] = sonuclar[r["sutun"]]["normalize"]["deger"]

    # ---- eşleşik karşılaştırmalar
    karsi = []
    hedefler = [c for c in tabanlar] + [c for c in yontem_c if c != bc] + kalib_sutun.get(bc, []) + \
               ([kontrol_sutun[bc]] if bc in kontrol_sutun else []) + (["tavan"] if manset["tavan"] else [])
    for b in hedefler:
        metler = [birincil] if b not in (en_iyi_b, ON_EK + (ham or "")) else [m.ad for m in metrikler]
        for met in metler:
            if met in nokta.get(b, {}) and met in nokta[bc]:
                f, a, u, p = fark(bc, b, met)
                yon = {m.ad: m.yon for m in metrikler}[met]
                karsi.append({"a": bc, "b": b, "metrik": met, "fark": f, "alt": a, "ust": u, "p": p,
                              "a_iyi": bool(yon * f > 0)})

    # ---- kalibrasyon eğrileri
    kal = {}
    if veri.gorev in ("ikili", "cok_sinif") and olasilik:
        for c in dict.fromkeys([bc] + ([ON_EK + ham] if ham else []) + kalib_sutun.get(bc, [])):
            y, p, _ = girdiler[c]
            e = kalibrasyon.egri(veri, y, p if veri.gorev == "ikili" else np.asarray(p))
            if e:
                e["kalibre_sutunlar"] = kalib_sutun.get(c, [])
                kal[c] = e

    # ---- segment, yayılım, örnekler
    sec = [c for c in dict.fromkeys([bc] + ([ON_EK + ham] if ham else []) + ([en_iyi_b] if en_iyi_b else [])) if c]
    seg_diz = {c: girdiler[c] for c in sec}
    if "tavan" in girdiler and (M.tavan or tavan_turu == "model") and tavan_turu != "orneklem_gurultusu":
        seg_diz["tavan"] = girdiler["tavan"]
    w = np.ones(len(d)) if agirlik is None else agirlik
    segment = analiz.segment_analizi(veri, seg_diz, M, w, B=min(B, 500), tohum=tohum) if veri.segmentler else {}
    for s_ad, s_oz in segment.items():
        s_oz["en_kotu"] = analiz.en_kotu(s_oz, bc, M.yon, deg(bc))
    yay_c = list(dict.fromkeys(sec + (yontem_c if len(yontem_c) <= 6 else []) + kalib_sutun.get(bc, [])))
    yay = analiz.yayilim(veri, {c: girdiler[c] for c in yay_c}, B=min(B, 200), tohum=tohum)
    orn_ler = analiz.ornekler(veri, bc, [c for c in [ON_EK + ham if ham else None, "taban_marjinal"] if c and c in d])

    # ---- denetimler
    den = denetim.bolme(veri)
    madde_den = "taban_marjinal" in nokta and madde_var and veri.gorev != "ab"
    if madde_den:
        den.append(denetim.fark_denetimi("madde ortalamasını geçiyor mu", bc, "taban_marjinal", tekrar, nokta, M, "bizim − marjinal"))
    if en_iyi_b and not (madde_den and en_iyi_b == "taban_marjinal"):
        den.append(denetim.fark_denetimi("en güçlü basit tabanı geçiyor mu", bc, en_iyi_b, tekrar, nokta, M, "bizim − taban"))
    if en_iyi_r:
        f_, a_, u_, p_ = fark(bc, en_iyi_r)
        den.append({"kontrol": "en güçlü diğer yöntem", "durum": "uyari" if M.yon * f_ < 0 else "tamam", "deger": f_,
                    "ayrinti": f"{birincil}: bizim − {etiket[en_iyi_r]} = {_f(f_)} [{_f(a_)}, {_f(u_)}], p={_f(p_, 3)}"
                               + (" · başka bir yöntem bizden iyi" if M.yon * f_ < 0 else "")})
    if bc in kontrol_sutun:
        den.append(denetim.fark_denetimi("persona karıştırma (birey bilgisi kullanılıyor mu)", bc, kontrol_sutun[bc],
                                         tekrar, nokta, M, "bizim − karıştırılmış"))
    if "madde_ici_kor" in nokta.get(bc, {}):
        mm = {m.ad: m for m in metrikler}["madde_ici_kor"]
        den.append(denetim.fark_denetimi("kişiye özgü sinyal (madde_ici_kor)", bc, kontrol_sutun.get(bc, bc), tekrar, nokta, mm,
                                         "bizim − karıştırılmış"))
    if "demeaned_r2" in nokta.get(bc, {}):  # sıfır = madde ortalaması; karıştırma burada sıfır hipotezi olamaz
        v_ = nokta[bc]["demeaned_r2"]
        a_, u_ = ga(tekrar[bc]["demeaned_r2"])
        den.append({"kontrol": "kişiye özgü sinyal (demeaned_r2 > 0)", "durum": "tamam" if a_ > 0 else "uyari", "deger": v_,
                    "ayrinti": f"de-meaned R² {_f(v_)} [{_f(a_)}, {_f(u_)}]"
                               + ("" if a_ > 0 else " · kişiye özgü sapmalar madde ortalamasından kötü ya da ayırt edilemiyor; "
                                  "genlik kalibrasyonu gerekebilir")})
    if tavan_turu == "model":
        den.append(_model_tavan_asimi(nokta, tekrar, yontem_c + kalib_sutun.get(bc, []), M))
    elif tavan_turu == "orneklem_gurultusu":
        den.append(_gurultu_tavan_asimi(nokta, tekrar, yontem_c + kalib_sutun.get(bc, []), M))
    else:
        den.append(denetim.tavan_asimi(nokta, tekrar, yontem_c + kalib_sutun.get(bc, []), M))
    if yalniz_basit:
        den.append({"kontrol": "yalnız basit tabanlar", "durum": "uyari", "deger": None,
                    "ayrinti": f"bildirilmiş taban yok; en güçlü taban {etiket[en_iyi_b]} (otomatik rastgele/marjinal). "
                               "Rozet bu basit tabana karşı: gerçek tabanları unlabeled_baselines= (etiketsiz) ya da "
                               "baselines_oof= (etiketle eğitilmiş, çapraz uydurmalı) ile bildirin"})
    den.append(denetim.cokus(yay, bc))
    dz = denetim.dengesizlik(veri)
    if dz:
        den.append(dz)
    den.append(denetim.kontaminasyon(veri, {c: girdiler[c] for c in sec}, M, kesim))
    den.extend(denetim.pozisyon_takasi(veri))
    den.append(denetim.ezber(ezber_json))
    if bizim_sec_oto:
        den.append({"kontrol": "seçim yanlılığı", "durum": "uyari",
                    "ayrinti": "'bizim' yöntem açıkça belirtilmedi (son sütun alındı); yöntemi önceden belirleyin", "deger": None})
    elif len(veri.yontemler) > 1:
        sira = sorted(yontem_c, key=lambda c: -M.yon * deg(c))
        den.append({"kontrol": "seçim yanlılığı", "durum": "tamam" if sira[0] != bc or len(sira) < 4 else "uyari",
                    "ayrinti": f"{len(veri.yontemler)} yöntem; 'bizim' önceden adlandırıldı"
                               + ("; bizim en iyisi: çok yöntem arasından seçildiyse iyimser, bağımsız veride doğrulayın"
                                  if sira[0] == bc and len(sira) >= 4 else ""), "deger": len(veri.yontemler)})
    for n_ in veri.notlar:
        den.append({"kontrol": "girdi notu", "durum": "uyari" if "düşürüldü" in n_ else "tamam", "ayrinti": n_, "deger": None})

    # ---- kayıtlı karşılaştırmalar (düşme koşulları) ve kör kol
    n_kume = int(veri.kume.max()) + 1

    def _cozum(ad):
        if ad is None:
            return None
        for c in (ad, ON_EK + ad):
            if c in nokta and birincil in nokta[c]:
                return c
        raise ValueError(f"comparison column {ad!r} is not an evaluated column (pred_<name>, or a ladder column such "
                         f"as taban_marjinal)")

    def _kiyas(a, b, et_, tur_):
        if b is None:
            r_ = tekrar[a][birincil]
            f_, (al_, us_), p_ = deg(a), ga(r_), p_degeri(r_)
        else:
            f_, al_, us_, p_ = fark(a, b)
        yf, ya, yu = M.yon * f_, (M.yon * al_ if M.yon > 0 else -us_), (M.yon * us_ if M.yon > 0 else -al_)
        if b is None:  # tek kol: değerin kendisi 0'a karşı (ham ölçek)
            ya, yu = al_, us_
        if not (np.isfinite(ya) and np.isfinite(yu)):
            sonuc = "GA yok"
        elif ya > 0:
            sonuc = "A iyi (GA 0'ın üstünde)" if b is not None else "0'ın üstünde"
        elif yu < 0:
            sonuc = "B iyi (GA 0'ın altında)" if b is not None else "0'ın altında"
        else:
            sonuc = "belirsiz (GA 0'ı içeriyor)"
        return {"a": a, "b": b, "etiket": et_, "tur": tur_, "metrik": birincil, "fark": f_, "alt": al_, "ust": us_,
                "p": p_, "a_iyi": None if b is None else bool(yf > 0), "n": n_kume, "sonuc": sonuc}

    kayitli = []
    kor_ozet = None
    if kor is not None:
        kc = _cozum(kor)
        if kc == bc:
            raise ValueError("blind_col is ours")
        kayitli.append(_kiyas(bc, kc, f"{etiket[bc]} − {etiket.get(kc, kc)} (kör kol: ad/başlık yok)", "kor"))
        kor_ozet = {"sutun": kc, "fark": kayitli[-1]["fark"], "alt": kayitli[-1]["alt"], "ust": kayitli[-1]["ust"],
                    "kazanc": None, "kor_kazanc": None, "korunan_pay": None, "kayip_cogu": False}
        if en_iyi_b:
            kaz, kkaz = M.yon * (deg(bc) - deg(en_iyi_b)), M.yon * (deg(kc) - deg(en_iyi_b))
            kor_ozet.update({"taban": en_iyi_b, "kazanc": kaz, "kor_kazanc": kkaz})
            if np.isfinite(kaz) and kaz > 0 and np.isfinite(kkaz):
                kor_ozet["korunan_pay"] = kkaz / kaz
                kor_ozet["kayip_cogu"] = bool(kkaz / kaz < 0.5)
        if kor_ozet["kayip_cogu"]:
            den.append({"kontrol": "kör kol (tanıma kontrolü)", "durum": "uyari", "deger": kor_ozet["korunan_pay"],
                        "ayrinti": f"kör kol en güçlü tabana göre kazancın yalnız %{100 * kor_ozet['korunan_pay']:.0f}'ini "
                                   f"koruyor (bizim − kör {_f(kor_ozet['fark'])} [{_f(kor_ozet['alt'])}, {_f(kor_ozet['ust'])}]): "
                                   "kazancın çoğu ad/başlık bilgisinden, tanıma ya da ezber olabilir"})
        elif kor_ozet["korunan_pay"] is not None:
            den.append({"kontrol": "kör kol (tanıma kontrolü)", "durum": "tamam", "deger": kor_ozet["korunan_pay"],
                        "ayrinti": f"kör kol kazancın %{100 * kor_ozet['korunan_pay']:.0f}'ini koruyor"})
    for a_, b_, et_ in kiyaslar or []:
        ac, bc_ = _cozum(a_), _cozum(b_)
        kayitli.append(_kiyas(ac, bc_, et_ or (f"{etiket.get(ac, ac)} − {etiket.get(bc_, bc_)}" if bc_ else
                                               f"{etiket.get(ac, ac)} (0'a karşı)"), "kayitli"))
    ezber_d = next((x for x in den if str(x.get("kontrol", "")).startswith("ezber testi")), None)
    ezber_uyari = None
    nedenler = []
    if ezber_d and ezber_d.get("durum") == "uyari" and (ezber_llm if ezber_llm is not None else bool(ezber_json)):
        nedenler.append("ezber_denetimi")
    if kor_ozet and kor_ozet["kayip_cogu"]:
        nedenler.append("kor_kol")
    if nedenler:
        ezber_uyari = {"nedenler": nedenler, "ezber_ayrinti": (ezber_d or {}).get("ayrinti") if "ezber_denetimi" in nedenler else None,
                       "kor_korunan_pay": (kor_ozet or {}).get("korunan_pay")}

    # ---- etkin örneklem çarpanı
    orneklem = None
    if carpan and veri.gorev != "ab":
        try:
            from . import orneklem as _orn
            sec = carpan_yontemler or [bizim] + ([ham] if ham else [])
            sec = [y for y in sec if y and y.removeprefix(ON_EK) not in (dongusel or [])]
            if dongusel:
                den.append({"kontrol": "döngüsel önsel (çarpan dışı)", "durum": "uyari", "deger": len(dongusel),
                            "ayrinti": "aynı kitlenin hedef etiketleriyle eğitilmiş önseller çarpana katılmadı: "
                                       + ", ".join(etiketler.get(y, y) for y in dongusel)
                                       + ". Bireysel tahmin karşılaştırması (kat dışı) geçerli; 'kaç insana bedel' döngüsel olurdu."})
            hed = {c: etiket[c] for c in dict.fromkeys(ON_EK + y if not y.startswith(ON_EK) else y for y in sec if y)
                   if c in etiket}
            veri_c, alt_not = _carpan_verisi(veri, carpan_en_fazla, tohum)
            orneklem = _orn.carpan(veri_c, hed, tohum=tohum)
            if orneklem:
                _teorik(veri_c, orneklem)
                if alt_not:
                    orneklem["alt_orneklem"] = {"n": len(veri_c.df), "N": len(d), "tohum": tohum}
                    orneklem.setdefault("notlar", []).append(alt_not)
                if "madde" not in d or d["madde"].nunique() < 20:
                    # az maddede Bayes n0 ve "yalnız simülasyon" eşdeğeri ortalamayı şans eseri tutturmaya bağlı: gösterilmez
                    for oz_ in orneklem["yontemler"].values():
                        oz_.pop("bayes", None)
                        oz_.pop("yalniz_sim", None)
                    orneklem.setdefault("notlar", []).append(
                        "Madde sayısı < 20: yalnız PPI++ gösteriliyor. Bayes büzülme ve 'hiç insan yokken' eşdeğeri, "
                        "simülasyonun tek bir ortalamayı tutturmasına bağlı olduğu için anlamsız.")
        except ImportError:
            orneklem = None
        except Exception as e:  # çarpan isteğe bağlı; pano yine çıksın
            den.append({"kontrol": "etkin örneklem çarpanı", "durum": "atlandi", "ayrinti": f"hata: {e}", "deger": None})

    sinir = [
        f"Bootstrap birimi '{veri.kume_adi}' ({int(veri.kume.max()) + 1} küme, B={B}; toplu metriklerde {orn.B_toplu}). "
        "Maddeler sabit kabul edilir: GA yeni kişilere genellemeyi ölçer, yeni sorulara değil.",
        "Baseline'lar ve kalibrasyon çapraz uydurmalıdır (aynı grubun satırı kendi eşlemesini görmez).",
    ]
    if manset["tavan"] and tavan_turu == "orneklem_gurultusu":
        sinir.append("Tavan örneklem gürültüsü tavanıdır (varyans bileşenleri): puanlanan tüm gruplarda güvenilirlik = "
                     "1 − ort(grup içi varyans / grup boyu) / var(grup ortalamaları); tavan = √güvenilirlik (Spearman'da "
                     "normal kuramla (6/π)·asin(√güvenilirlik/2), r2'de güvenilirlik). Gürültülü grup ortalamasını gerçek "
                     "grup değerinden ayıran tek şey grup içi örneklemdir; tavanı anlamlı aşmak sızıntıya işaret eder.")
    elif manset["tavan"] and tavan_turu == "model":
        sinir.append("Tavan bir model/oracle tahminidir (insan test-retest değil): ulaşılabilir bir üst sınır değil, "
                     "kıyas noktasıdır; kapanan boşluk bu tavana göredir ve tavanı aşmak sızıntı anlamına gelmez.")
    elif manset["tavan"]:
        sinir.append("Tavan, aynı birimin ikinci ölçümüyle uyumdur. Gürültüsüz bir tahmin edici bunu aşabilir; "
                     "tavana oran %100'ü geçerse önce sızıntıya bakılır.")
    if veri.gorev in ("ikili", "cok_sinif"):
        sinir.append("Doğruluk sınıf payına duyarlıdır; kappa ve dengeli doğruluk birlikte okunmalı.")
    sinir += ([dengesiz_not] if dengesiz_not else []) + list(sinirlar or [])
    if yalniz_basit:  # sona: özgün sınır sırası (kart ilk üçünü alır) değişmesin
        sinir.append("Yalnız basit tabanlar (rastgele/marjinal): bildirilmiş güçlü bir taban olmadan rozet iyimserdir.")

    kart = {"n_satir": len(d), "n_kume": int(veri.kume.max()) + 1, "kume_adi": veri.kume_adi,
            "n_madde": int(d["madde"].nunique()) if madde_var else None, "B": B, "tohum": tohum,
            "segmentler": veri.segmentler, "notlar": veri.notlar,
            "yontemler": veri.yontemler, "bizim": bizim, "ham": ham}
    taban_ozet = {"bildirilen": [c for c in turler if taban_turu.get(c) in ("etiketsiz", "oof")],
                  "etiketsiz": [c for c in turler if taban_turu.get(c) == "etiketsiz"],
                  "oof": [c for c in turler if taban_turu.get(c) == "oof"],
                  "otomatik": [c for c in turler if turler[c] == "baseline" and taban_turu.get(c) == "otomatik"],
                  "en_guclu": en_iyi_b, "yalniz_basit": yalniz_basit}
    ozet = {
        "baslik": baslik, "veri_adi": veri_adi, "gorev": veri.gorev,
        "duzey": duzey or ("makro" if veri.gorev in ("dagilim", "ab") else "mikro"),
        "tutma_birimi": veri.kume_adi, "girdi_kaynagi": girdi_kaynagi,
        "olusturma": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "surum": SURUM,
        "kart": kart, "birincil": birincil,
        "metrikler": [{"ad": m.ad, "yon": m.yon, "aciklama": m.aciklama, "tur": m.tur} for m in metrikler],
        "sonuclar": sonuclar, "merdiven": merd, "manset": manset, "orneklem": orneklem,
        "karsilastirmalar": karsi, "kalibrasyon": kal, "segment": segment, "yayilim": yay,
        "denetim": den, "ornekler": orn_ler, "onkayit": onkayit or [], "sinirlar": sinir,
        "tabanlar": taban_ozet, "tavan_turu": tavan_turu,
    }
    if gurultu is not None:
        ozet["gurultu_tavani"] = gurultu
    if kayitli:
        ozet["kayitli_karsilastirmalar"] = kayitli
    if kor_ozet is not None:
        ozet["kor_kol"] = kor_ozet
    if ezber_uyari is not None:
        ozet["ezber_uyarisi"] = ezber_uyari
    ozet = _temiz(ozet)
    ozet["kanit_kart"] = _temiz(_kart.kart(ozet, muhur=(onkayit or [None])[-1]))  # sahne kart.py ile aynı şema
    return ozet


def _tahmin_ozeti(seri: pd.Series) -> dict:
    """Bir tahmin sütununun gerçeksiz özeti: sayısal → dağılım istatistikleri, etiket → en sık sınıflar."""
    v = seri.to_numpy(object)
    eksik = int(sum(x is None or (np.isscalar(x) and pd.isna(x)) for x in v))
    out = {"n": int(len(v)), "eksik": eksik}
    dolu = seri.dropna()
    if dolu.empty:
        return out
    ilk = dolu.iloc[0]
    if isinstance(ilk, (list, tuple, np.ndarray, dict)) or (isinstance(ilk, str) and ilk.lstrip().startswith(("[", "{"))):
        out["tur"] = "vektor"
        return out
    x = pd.to_numeric(dolu, errors="coerce")
    if x.notna().all():
        q = np.percentile(x.to_numpy(float), [0, 25, 50, 75, 100])
        out.update({"tur": "sayi", "ortalama": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
                    "en_kucuk": float(q[0]), "q25": float(q[1]), "medyan": float(q[2]), "q75": float(q[3]),
                    "en_buyuk": float(q[4])})
    else:
        pay = dolu.astype(str).value_counts(normalize=True).head(5)
        out.update({"tur": "etiket", "en_sik": {str(k): float(p) for k, p in pay.items()}})
    return out


def bekleyen(df: pd.DataFrame, baslik: str, veri_adi: str = "", gorev: str | None = None,
             yontemler: list[str] | None = None, bizim: str | None = None, ham: str | None = None,
             etiketler: dict | None = None, birincil: str | None = None, onkayit: list | None = None,
             sinirlar: list | None = None, duzey: str | None = None, girdi_kaynagi: str | None = None,
             tohum: int = 0, B: int = 1000) -> dict:
    """Gerçek yok (sütun yok ya da tümü eksik): kalibrasyon, merdiven ve bootstrap atlanır. Rapor mühür kayıtlarını ve
    tahmin özetini taşır; kart rozeti 'bekliyor' (awaiting truth). Şema degerlendir() ile aynı anahtarlar."""
    etiketler = dict(etiketler or {})
    tsut = [ON_EK + y for y in yontemler] if yontemler else [c for c in df.columns if isinstance(c, str) and c.startswith(ON_EK)]
    eksik = [c for c in tsut if c not in df.columns]
    if eksik:
        raise ValueError(f"eksik tahmin sütunları: {eksik}")
    if not tsut:
        raise ValueError("en az bir 'tahmin_<yontem>' sütunu gerekli")
    d = df
    if "bolum" in d.columns:
        from .sema import TEST_ADLARI
        d = d[d["bolum"].astype(str).str.lower().isin(TEST_ADLARI).to_numpy()]
    adlar = [c[len(ON_EK):] for c in tsut]
    bizim_oto = bizim is None
    bizim = bizim or adlar[-1]
    ozetler = {c: {"etiket": etiketler.get(c[len(ON_EK):], c[len(ON_EK):]), **_tahmin_ozeti(d[c])} for c in tsut}
    kume_adi = next((s for s in ("grup", "madde") if s in d.columns), "satir")
    n_kume = int(d[kume_adi].nunique()) if kume_adi != "satir" else len(d)
    muhurler = list(onkayit or [])
    den = [{"kontrol": "gerçek", "durum": "atlandi", "deger": None,
            "ayrinti": ("'gercek' (truth) sütunu yok" if "gercek" not in df.columns else "'gercek' (truth) sütununun tamamı eksik")
                       + ": kalibrasyon, taban merdiveni, bootstrap ve çarpan atlandı"},
           {"kontrol": "mühür", "durum": "tamam" if muhurler else "uyari", "deger": len(muhurler),
            "ayrinti": (f"{len(muhurler)} mühür kaydı eklendi; son sha256 {str(muhurler[-1].get('sha256'))[:12]}"
                        if muhurler else "mühür kaydı yok: tahmini gerçek açılmadan önce seal() ile mühürleyin "
                                         "ve evaluate(prereg=<dizin>) verin")}]
    if bizim_oto:
        den.append({"kontrol": "seçim yanlılığı", "durum": "uyari", "deger": None,
                    "ayrinti": "'bizim' yöntem açıkça belirtilmedi (son sütun alındı); yöntemi önceden belirleyin"})
    bc = ON_EK + bizim
    manset = {"ham": None, "bizim": {"sutun": bc, "etiket": etiketler.get(bizim, bizim), "deger": None, "alt": None, "ust": None},
              "en_iyi_baseline": None, "en_iyi_rakip": None, "taban_tavana_oran": None, "tavan": None,
              "tavana_oran": None, "bosluk_kapanan": None,
              "cumle": f"Gerçek henüz yok: {len(d)} satır tahmin" + (" mühürlü" if muhurler else " (mühürsüz)")
                       + "; kart gerçek gelince dolar."}
    if ham:
        manset["ham"] = {"sutun": ON_EK + ham, "etiket": etiketler.get(ham, ham), "deger": None, "alt": None, "ust": None}
    sinir = ["Gerçek sonuç yok: bu kart yalnız mühür ve tahmin özetidir, başarı iddiası taşımaz."] + list(sinirlar or [])
    kart = {"n_satir": len(d), "n_kume": n_kume, "kume_adi": kume_adi,
            "n_madde": int(d["madde"].nunique()) if "madde" in d.columns else None, "B": B, "tohum": tohum,
            "segmentler": [], "notlar": [], "yontemler": adlar, "bizim": bizim, "ham": ham}
    ozet = {
        "baslik": baslik, "veri_adi": veri_adi, "gorev": gorev, "duzey": duzey or "mikro",
        "tutma_birimi": kume_adi, "girdi_kaynagi": girdi_kaynagi,
        "olusturma": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "surum": SURUM,
        "kart": kart, "birincil": birincil, "metrikler": [], "sonuclar": {}, "merdiven": [], "manset": manset,
        "orneklem": None, "karsilastirmalar": [], "kalibrasyon": {}, "segment": {}, "yayilim": {}, "denetim": den,
        "ornekler": [], "onkayit": muhurler, "sinirlar": sinir, "tabanlar": None, "tavan_turu": None,
        "gercek_bekleniyor": True, "tahmin_ozeti": ozetler,
    }
    ozet = _temiz(ozet)
    ozet["kanit_kart"] = _temiz(_kart.kart(ozet, muhur=(muhurler or [None])[-1]))
    return ozet
