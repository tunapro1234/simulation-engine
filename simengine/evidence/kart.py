"""Kanıt kartı: özetten tek ekranlık kart, sahne `kart.py` ile AYNI şema (sahne bu sözlüğü doğrudan çizer).

    {"baslik", "alt_baslik", "metrik", "n", "kaynak",
     "satirlar": [{"tur": taban|ham|kalibre|tavan|bosluk|bedel, "etiket", "deger", "alt", "ust", "birim", "not", "kaynak"}],
     "fark": {"fark", "alt", "ust"} | None,          # bizim − taban, metrik yönüne çevrilmiş
     "rejimler": [{"ad", "aciklama", "satirlar": [{"etiket", "deger", "alt", "ust", "birim", "kaynak"}]}],
     "yanilgi": [str], "muhur": dict | None, "rozet": yuksek|kararsiz|bekliyor, "rozet_gerekce", "etiketler", "dil"}

Taban = etiketsiz tabanların (tur baseline) birincil metrikte en iyisi; bedel satırı = BİZİM yöntemin n=100 PPI++
çarpanı, bizim yöntem döngüselse (çarpanı yok) ham kolun; ikisi de yoksa "ölçülmedi". Karttaki en yüksek çarpan
başka bir yöntemdeyse (ör. etiketsiz taban) bedel notunda adıyla yazılır ama rozete dayanak olmaz (KARŞIT-2 K2-28:
eskiden en yüksek çarpan seçiliyordu, taban kendini onaylıyordu ve en iyiyi seçmek kazananın lanetine açıktı).
Rozet "yuksek" için: bizim − taban GA'sı sıfırın üstünde ve fark ≥ ANLAMLI_FARK; bedel bizim yöntemden geliyorsa
ayrıca alt sınır ≥ ESIK_BEDEL ve tabanın kendi bedel alt sınırının üstünde. Fark yalnız taban_var=False iken
yoksa (taban yok ya da taban insanın kendisi, ör. Upworthy insan@100) bizim yöntemin bedeliyle; fark GA'sız ya da taban
varken eşleşik fark yoksa "kararsiz". Bizim − taban GA'sı tümüyle sıfırın altındaysa her durumda "kararsiz".
"""
from __future__ import annotations

import math

SIRA = ("taban", "ham", "kalibre", "tavan", "bosluk", "bedel")
ANLAMLI_FARK = 0.02
ESIK_BEDEL = 1.10  # bedelde en küçük etki: alt sınır 1,10×'in altındaysa N büyüdükçe 1,05× de "yüksek" almasın


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def satir(tur, etiket, deger=None, alt=None, ust=None, birim="", not_=None, kaynak=None) -> dict:
    return {"tur": tur, "etiket": etiket, "deger": _f(deger), "alt": _f(alt), "ust": _f(ust), "birim": birim,
            "not": not_, "kaynak": kaynak}


def _u(d) -> tuple:
    return (d.get("deger"), d.get("alt"), d.get("ust")) if d else (None, None, None)


def rozet(satirlar: list[dict], fark: dict | None = None, gercek_var: bool = True, bedel_kaynak: str | None = None,
          taban_bedel_alt: float | None = None, taban_var: bool = False) -> tuple[str, str]:
    """bedel_kaynak: "bizim" | "ham" | None (bedel satırı hangi yöntemin çarpanı); yalnız "bizim" rozete dayanak olur.
    taban_var: kartta kıyaslanacak bir taban var; o zaman eşleşik fark GA'sı olmadan "yüksek" verilmez."""
    s = {r["tur"]: r for r in satirlar}
    if not gercek_var or all((s.get(t) or {}).get("deger") is None for t in ("kalibre", "ham")):
        return "bekliyor", "Gerçek sonuç henüz yok: tahmin mühürlü, kart gerçek gelince dolar."
    tam = bool(fark) and all(fark.get(k) is not None for k in ("fark", "alt", "ust"))
    if tam and fark["ust"] < 0:  # sahne kuralına ek: taban altındaysa bedel kurtarmaz
        return "kararsiz", (f"Bizim yöntem en güçlü basit tabanın altında: fark {fark['fark']:+.3f}, "
                            f"GA [{fark['alt']:+.3f}; {fark['ust']:+.3f}].")
    b = s.get("bedel") or {}
    bedel_sorun = None
    if bedel_kaynak == "bizim" and b.get("alt") is not None:
        if b["alt"] < ESIK_BEDEL:
            bedel_sorun = f"bizim yöntemin bedel alt sınırı {b['alt']:.2f}× < {ESIK_BEDEL:.2f}×"
        elif taban_bedel_alt is not None and b["alt"] <= taban_bedel_alt:
            bedel_sorun = f"bizim yöntemin bedel alt sınırı {b['alt']:.2f}× tabanınkini ({taban_bedel_alt:.2f}×) geçmiyor"
    if tam:
        if fark["alt"] > 0 and fark["fark"] >= ANLAMLI_FARK:
            if bedel_sorun:
                return "kararsiz", f"Tabana göre fark {fark['fark']:+.3f} anlamlı, ama {bedel_sorun}."
            return "yuksek", f"Tabana göre fark {fark['fark']:+.3f}, GA [{fark['alt']:+.3f}; {fark['ust']:+.3f}] sıfırın üstünde."
        if fark["alt"] > 0:
            return "kararsiz", (f"Fark anlamlı ama küçük ({fark['fark']:+.3f}, GA [{fark['alt']:+.3f}; {fark['ust']:+.3f}]); "
                                "en güçlü basit taban neredeyse aynı yerde.")
        return "kararsiz", f"Tabana göre fark {fark['fark']:+.3f}, GA [{fark['alt']:+.3f}; {fark['ust']:+.3f}] sıfırı içeriyor."
    if fark or taban_var:  # fark verildi ama GA'sı yok, ya da taban var ama eşleşik kıyas yok
        return "kararsiz", "En güçlü basit tabana göre eşleşik farkın güven aralığı ölçülmedi."
    if bedel_kaynak == "bizim" and b.get("alt") is not None:
        if bedel_sorun:
            return "kararsiz", f"Tabana göre fark yok; {bedel_sorun}."
        return "yuksek", f"Bizim yöntemin kaç insana bedel GA'sı {b['alt']:.2f}×'ten başlıyor (≥ {ESIK_BEDEL:.2f}×)."
    return "kararsiz", "Tabana göre farkın güven aralığı yok ve bizim yöntemin kendi bedeli ölçülmedi."


def _ppi100(oz: dict, sutun: str | None):
    y = ((oz.get("orneklem") or {}).get("yontemler") or {}).get(sutun) if sutun else None
    r = next((q for q in (y or {}).get("ppi") or [] if q.get("n") == 100), None)
    return ((y.get("etiket") or sutun), r) if r and _f(r.get("carpan")) is not None else None


def _bedel_satiri(oz: dict, bizim: str | None = None, ham: str | None = None) -> tuple[dict, str | None]:
    """(satır, kaynak): bizim yöntemin n=100 çarpanı, yoksa ham kolun. Karttaki en yüksek çarpan notta, dayanak değil."""
    yt = (oz.get("orneklem") or {}).get("yontemler") or {}
    en = None
    for c in yt:
        x = _ppi100(oz, c)
        if x and (en is None or x[1]["carpan"] > en[1]["carpan"]):
            en = x
    for kaynak, sutun in (("bizim", bizim), ("ham", ham)):
        x = _ppi100(oz, sutun)
        if not x:
            continue
        et, r = x
        notlar = [f"≈ {r['n_esdeger']} gerçek yanıt (PPI++, sonlu kitle)"] if r.get("n_esdeger") else []
        if kaynak == "ham":
            notlar.append("bizim yöntemin kendi çarpanı yok (aynı kitlenin etiketiyle eğitildi, döngüsel); satır ham kolun")
        if en and en[0] != et:
            notlar.append(f"en yüksek çarpan {en[0]} {en[1]['carpan']:.2f}× (rozete dayanak değil)".replace(".", ","))
        return satir("bedel", f"Kaç insana bedel: {et} + 100 gerçek yanıt", r["carpan"], r.get("alt"), r.get("ust"),
                     birim="×", not_="; ".join(notlar) or None), kaynak
    return satir("bedel", "Kaç insana bedel", not_="ölçülmedi" + (
        f" (bizim ve ham kolun çarpanı yok; en yüksek çarpan {en[0]} {en[1]['carpan']:.2f}×, dayanak değil)".replace(".", ",")
        if en else "")), None


TAVAN_ETIKET = {"model": "Model/oracle tavanı", "orneklem_gurultusu": "Örneklem gürültüsü tavanı (varyans bileşenleri)"}
TAVAN_NOT = {"model": "tür: model/oracle (insan test-retest değil)",
             "orneklem_gurultusu": "tavan türü: örneklem gürültüsü (varyans bileşenleri, tüm gruplar)"}
TAHMIN_NOT = {"tavan": "uygulanamaz: tahmin görevi, insan test-retest yok (not applicable: forecast task)",
              "bedel": "uygulanamaz: tahmin görevi, insan yanıtı yok, PPI++ atlandı (not applicable: forecast task)"}
BILGI_NOT = ("ayar-A tabanı (hedef sorunun cevabını görmez); ayar-B tabanı {} hedef sorunun öbür hücrelerinin "
             "cevabını okur: merdivende, rozete dayanak değil (setting-B baseline not used for the badge)")
EZBER_EKI = "Uyarı: kazanç ezber/tanımadan gelebilir"
EZBER_NEDEN = {"ezber_denetimi": "ezber denetimi uyarıda", "kor_kol": "kör kol kazancın çoğunu kaybediyor"}


def kazanc_var(fark: dict | None) -> bool:
    """Ezber notu yalnız bir kazanç varken anlamlı: bizim − en güçlü taban > 0 (taban yoksa bilinmez: True)."""
    if not fark or fark.get("fark") is None:
        return True
    return bool(fark["fark"] > 0)


def kart(oz: dict, rejimler=None, yanilgi=None, etiketler=None, muhur=None) -> dict:
    man = oz.get("manset") or {}
    tahmin = bool(oz.get("tahmin_gorevi"))
    birincil = oz.get("birincil")
    yon = next((m.get("yon", 1) for m in oz.get("metrikler") or [] if m.get("ad") == birincil), 1)
    son = oz.get("sonuclar") or {}
    tb = man.get("en_iyi_baseline")
    ham, biz, tav, bos = man.get("ham"), man.get("bizim"), man.get("tavan"), man.get("bosluk_kapanan")
    S = [satir("taban", f"En güçlü basit taban: {tb['etiket']}" if tb else "En güçlü basit taban", *_u(tb),
               not_=None if tb else "taban yok"),
         satir("ham", f"Ham: {ham['etiket']}" if ham else "Ham yöntem", *_u(ham), not_=None if ham else "ham kol yok"),
         satir("kalibre", f"Bizim: {biz['etiket']}" if biz else "Bizim yöntem", *_u(biz)),
         satir("tavan", TAVAN_ETIKET.get(oz.get("tavan_turu"), "İnsan tekrar tavanı"), *_u(tav),
               not_=(TAVAN_NOT.get(oz.get("tavan_turu")) if tav else
                     (TAHMIN_NOT["tavan"] if tahmin else "tavan yok (test-tekrar ölçülmemiş)"))),
         satir("bosluk", "Kapanan boşluk (bizim − taban)/(tavan − taban)", *_u(bos), birim="%",
               not_=None if bos else "tavan yok: hesaplanamaz"),
         None]
    if tahmin:
        S[-1], bedel_kaynak = satir("bedel", "Kaç insana bedel", not_=TAHMIN_NOT["bedel"]), None
    else:
        S[-1], bedel_kaynak = _bedel_satiri(oz, (biz or {}).get("sutun"), (ham or {}).get("sutun"))
    tbb = _ppi100(oz, (tb or {}).get("sutun"))
    fark = None
    if biz and tb:
        for k in oz.get("karsilastirmalar") or []:
            if k.get("a") == biz.get("sutun") and k.get("b") == tb.get("sutun") and k.get("metrik") == birincil:
                fark = {"fark": yon * k["fark"], "alt": yon * (k["alt"] if yon > 0 else k["ust"]),
                        "ust": yon * (k["ust"] if yon > 0 else k["alt"])}
    r, g = rozet(S, fark, gercek_var=bool(son), bedel_kaynak=bedel_kaynak,
                 taban_bedel_alt=_f(tbb[1].get("alt")) if tbb else None, taban_var=bool(biz and tb))
    atl = [d.get("kontrol") for d in oz.get("denetim") or [] if d.get("durum") == "atlandi" and not d.get("uygulanamaz")
           and any(k in (d.get("kontrol") or "") for k in ("bölme", "grup", "zaman", "kontaminasyon", "ezber"))]
    llm_yok = any(d.get("uygulanamaz") and "ezber" in (d.get("kontrol") or "") for d in oz.get("denetim") or [])
    # LLM kolu yoksa ezber denetimi eksikliği bayrak sebebi değildir: kalan atlanan sızıntı denetimleri için "sızıntı"
    et = list(etiketler or []) + ([f"kanıtlanmadı: sızıntı ({', '.join(atl)} atlandı)" if llm_yok else
                                   "kanıtlanmadı: sızıntı/ezber"] if atl else [])
    if llm_yok:
        et.append("LLM kolu yok: ezber denetimi uygulanamaz (no LLM arm: memorization audit not applicable)")
    if oz.get("ga_uyarisi"):
        gu = oz["ga_uyarisi"]
        et.append(f"yüzdelik GA nokta tahminini dışarıda bırakıyor ({gu.get('metrik')}, {gu.get('n')} sütun)"
                  + ("; ci='basic' deneyin" if gu.get("yontem") == "percentile" else "; temel (basic) GA kullanıldı"))
    if (oz.get("tabanlar") or {}).get("yalniz_basit"):
        et.append("yalnız basit tabanlar (rastgele/marjinal)")
    bd = oz.get("bilgi_duzeyi") or {}
    if bd.get("B") and tb:   # dağılım: taban_marjinal hedef sorunun öbür hücrelerini okur (ayar B), rozete dayanak değil
        S[0]["not"] = BILGI_NOT.format(", ".join(bd["B"]))
    if bd.get("B_ustte"):
        et.append(f"ayar-B tabanı ({', '.join(bd['B_ustte'])}) daha güçlü: rozet ayar-A tabanına karşı")
    for sut, x in ((oz.get("dolgu") or {}).get("kollar") or {}).items():
        if (_f(x.get("ortalama")) or 0) > (oz.get("dolgu") or {}).get("esik", 0.05):
            et.append(f"dolgu payı yüksek: {sut} %{100 * x['ortalama']:.0f} (geçersiz → tekdüze)")
    kl = oz.get("kayitli_karsilastirmalar") or []
    ph = [k for k in kl if k.get("tur") == "posthoc"]
    if ph:
        et.append(f"sonradan karşılaştırma: {len(ph)} satır ön kayıtta yok (post hoc)")
    zt = [k for k in kl if k.get("zayif_taban")]
    if zt:
        et.append(f"kayıtlı karşılaştırma zayıf tabana karşı (en güçlü: {zt[0]['zayif_taban'].get('en_guclu_etiket')})")
    ez = oz.get("ezber_uyarisi")
    if ez and r != "bekliyor" and kazanc_var(fark):
        neden = ", ".join(EZBER_NEDEN.get(n, n) for n in ez.get("nedenler") or [])
        g = f"{g} {EZBER_EKI} ({neden})."
        et.append("kazanç ezber/tanımadan olabilir")
    if r == "bekliyor" and not (muhur or oz.get("onkayit")):
        g = "Gerçek sonuç henüz yok ve tahmin mühürlenmemiş: gerçek açılmadan önce seal() ile mühürleyin."
    out = {"baslik": oz.get("baslik"), "alt_baslik": f"metrik: {birincil}", "metrik": birincil,
            "n": (oz.get("kart") or {}).get("n_kume"), "kaynak": oz.get("girdi_kaynagi") or oz.get("veri_adi"),
            "satirlar": S, "fark": fark, "rejimler": rejimler or [],
            "yanilgi": list(yanilgi or []) + list(oz.get("sinirlar") or [])[:3], "muhur": muhur,
            "rozet": r, "rozet_gerekce": g, "etiketler": et, "dil": "tr"}
    if oz.get("kayitli_karsilastirmalar"):   # kayıtlı karşılaştırmalar / düşme koşulları (evaluate(compare=, blind_col=))
        out["kayitli"] = oz["kayitli_karsilastirmalar"]
    return out


# Herkese açık panel fiyatı, yanıt başı USD (araştırma notu, 21 Eki 2026): Pollfish "starting at
# $0.95" (taban), Prolific 10 dk anket önerilen ücret + %42,8 ≈ 2,86 (hesap). Yalnız anket tipi kart (birim = yanıt).
YANIT_FIYATI = (0.95, 2.86)


def dolar(n_esdeger, n=100, fiyat=YANIT_FIYATI) -> str | None:
    """Ek kesinliğin panel karşılığı: (n_eşdeğer − n) yanıt × yanıt başı fiyat, 10 dolara yuvarlı."""
    ek = _f(n_esdeger)
    if ek is None or ek - n <= 0:
        return None
    ek = round(ek - n)
    lo, hi = ek * fiyat[0], ek * fiyat[1]
    lo, hi = (round(lo), round(hi)) if hi < 100 else (int(round(lo / 10.0)) * 10, int(round(hi / 10.0)) * 10)
    return f"~{ek} ek yanıt ≈ ${lo}-{hi} [hesap: yanıt başı ${fiyat[0]:.2f}-{fiyat[1]:.2f}]".replace(".", ",")


def ek_not(k: dict, n_esdeger=None, n=100, anket=True, dipnot=None) -> dict:
    """Bedel satırına panel karşılığı (anket tipi kartta) ve dipnot ekler; kartı yerinde günceller."""
    b = next((r for r in k.get("satirlar") or [] if r.get("tur") == "bedel"), None)
    if b is None or b.get("deger") is None:
        return k
    parca = [b.get("not")] if b.get("not") else []
    if anket and (d := dolar(n_esdeger, n)):
        parca.append(d)
    if dipnot:
        parca.append(dipnot)
    b["not"] = "; ".join(dict.fromkeys(parca)) or None
    return k
