"""degerlendir(): uzun tablo → özet sözlüğü (OZET-SEMA.md). Pano ve JSON bunu kullanır."""
from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import pandas as pd

from . import analiz, denetim, kalibrasyon, merdiven
from . import kart as _kart
from .bootstrap import Orneklem, ga, hesapla, p_degeri
from .metrik import BIRINCIL, aile, diziler
from .sema import ON_EK, hazirla

SURUM = "0.1"


def _diz(veri, sutun):
    """(y, p, maske). 'tekrar' eksik olabilir → maske."""
    d = veri.df
    if sutun != "tekrar":
        y, p = diziler(veri, sutun)
        return y, p, None
    if veri.gorev == "dagilim":
        maske = np.array([v is not None and not (np.isscalar(v) and pd.isna(v)) for v in d["tekrar"]])
        y, _ = diziler(veri, "gercek")
        return y, d["tekrar"].to_numpy(object), maske
    if veri.gorev == "cok_sinif":
        kod = {k: i for i, k in enumerate(veri.siniflar)}
        y, _ = diziler(veri, "gercek")
        p = d["tekrar"].map(lambda v: kod.get(v, kod.get(str(v), -1))).to_numpy()
        return y, p, p >= 0
    y = d["gercek"].to_numpy(float)
    p = pd.to_numeric(d["tekrar"], errors="coerce").to_numpy(float)
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


def _f(x, b=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{b}f}".replace(".", ",")


def degerlendir(df: pd.DataFrame, baslik: str, veri_adi: str = "", gorev: str | None = None,
                yontemler: list[str] | None = None, bizim: str | None = None, ham: str | None = None,
                etiketler: dict | None = None, birincil: str | None = None, B: int = 1000,
                B_toplu: int = 200, tohum: int = 0, kalibre: list[str] | None = None,
                kesim: str | None = None, ezber_json: str | None = None, carpan: bool = True,
                sinirlar: list[str] | None = None, onkayit: list | None = None,
                siniflar: list | None = None, baseline: bool = True,
                carpan_yontemler: list[str] | None = None, dongusel: list[str] | None = None,
                duzey: str | None = None, girdi_kaynagi: str | None = None,
                etiketsiz_tabanlar: list[str] | None = None) -> dict:
    """bizim/ham: tahmin_ ön eki olmadan yöntem adı. bizim verilmezse SON yöntem (önceden belirlenmeli;
    en iyiyi sonradan seçmek iyimser yanlılık yaratır, denetimde işaretlenir)."""
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
    for y_ in veri.yontemler:
        c = ON_EK + y_
        # etiketsiz_tabanlar: hedef etiketi görmeyen basit kurallar (ör. kalıcılık) taban sayılır
        turler[c] = "baseline" if y_ in (etiketsiz_tabanlar or []) else "yontem"
        etiket[c] = etiketler.get(y_, y_)
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
    if "tekrar" in d:
        girdiler["tavan"] = _diz(veri, "tekrar")
        turler["tavan"], etiket["tavan"] = "tavan", "İnsan tavanı (test-retest)"
        tavan_metrik = [m for m in metrikler if m.tavan]
    else:
        tavan_metrik = []

    # ---- bootstrap (eşleşik)
    madde_kod = pd.factorize(d["madde"])[0] if madde_var else None
    orn = Orneklem(veri.kume, B=B, B_toplu=B_toplu, tohum=tohum, madde_kume=veri.kume_adi == "madde")
    tavan_girdi = {"tavan": girdiler.pop("tavan")} if "tekrar" in d else {}
    nokta, tekrar = hesapla(orn, girdiler, metrikler, madde_kod, agirlik)
    if tavan_girdi:
        n2, t2 = hesapla(orn, tavan_girdi, tavan_metrik, madde_kod, agirlik)
        nokta.update(n2)
        tekrar.update(t2)
        girdiler.update(tavan_girdi)

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
    if len(yontem_c) <= 6:
        secili = yontem_c
    else:  # kalabalıkta ham + bizim + bizi geçen her yöntem (sessizce düşürülmez)
        bd = M.yon * deg(ON_EK + bizim)
        secili = list(dict.fromkeys([ON_EK + y for y in [ham, bizim] if y] + [c for c in yontem_c if M.yon * deg(c) > bd]))
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
        parca.append(f"insan tavanı {_f(manset['tavan']['deger'])}")
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
    if "tavan" in girdiler and M.tavan:
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
    den.append(denetim.tavan_asimi(nokta, tekrar, yontem_c + kalib_sutun.get(bc, []), M))
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
            orneklem = _orn.carpan(veri, hed, tohum=tohum)
            if orneklem:
                _teorik(veri, orneklem)
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
    if manset["tavan"]:
        sinir.append("Tavan, aynı birimin ikinci ölçümüyle uyumdur. Gürültüsüz bir tahmin edici bunu aşabilir; "
                     "tavana oran %100'ü geçerse önce sızıntıya bakılır.")
    if veri.gorev in ("ikili", "cok_sinif"):
        sinir.append("Doğruluk sınıf payına duyarlıdır; kappa ve dengeli doğruluk birlikte okunmalı.")
    sinir += ([dengesiz_not] if dengesiz_not else []) + list(sinirlar or [])

    kart = {"n_satir": len(d), "n_kume": int(veri.kume.max()) + 1, "kume_adi": veri.kume_adi,
            "n_madde": int(d["madde"].nunique()) if madde_var else None, "B": B, "tohum": tohum,
            "segmentler": veri.segmentler, "notlar": veri.notlar,
            "yontemler": veri.yontemler, "bizim": bizim, "ham": ham}
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
    }
    ozet = _temiz(ozet)
    ozet["kanit_kart"] = _temiz(_kart.kart(ozet, muhur=(onkayit or [None])[-1]))  # sahne kart.py ile aynı şema
    return ozet
