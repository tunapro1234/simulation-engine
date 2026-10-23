"""degerlendir(): uzun tablo → özet sözlüğü (OZET-SEMA.md). Pano ve JSON bunu kullanır."""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import difflib
import math
import warnings

import numpy as np
import pandas as pd

from . import analiz, denetim, kalibrasyon, merdiven
from . import kart as _kart
from .bootstrap import Orneklem, ga, hesapla, p_degeri
from .metrik import BIRINCIL, ISARETLI_METRIKLER, ORAN_METRIKLERI, aile, diziler
from .sema import ON_EK, _normalle, _sinif_vektoru, _vektor, _vektor_mu, gercek_yok, hazirla

SURUM = "0.1"
BILGI_B_ETIKET = "ayar B: hedef sorunun öbür hücrelerinin cevabı (setting B)"
DOLGU_ESIK = 0.05   # kol başına dolgu (geçersiz → tekdüze) payı bunu aşarsa denetim uyarısı
# sürekli görevde pair_sign kendiliğinden yalnız toplam madde içi çift sayısı bu kadarken eklenir (her bootstrap
# tekrarında her kolda çiftler yeniden kurulur: 2M çiftte evaluate ~7 kat yavaşlar); primary="pair_sign" sınırı kaldırır
CIFT_EN_FAZLA = 50_000
AZ_KUME = 10        # bootstrap birimi (küme; ab'de test) bundan azsa GA anlamsız: denetim uyarısı, rozet "yüksek" olmaz
# api.METRIC_TO_INTERNAL'a ek İngilizce yazımlar (sklearn / yaygın adlar) → metrik.aile iç adı
BIRINCIL_EK_ADLAR = {"log_loss": "logloss", "cross_entropy": "logloss", "roc_auc": "auc", "f1": "macro_f1",
                     "f1_macro": "macro_f1", "brier_score": "brier", "balanced_acc": "dengeli_dogruluk",
                     "acc": "dogruluk", "cohen_kappa": "kappa", "mean_absolute_error": "mae",
                     "root_mean_squared_error": "rmse", "r_squared": "r2"}


def _metrik_adlari() -> tuple[dict, dict]:
    """(İngilizce/eş ad → iç ad, iç ad → İngilizce ad). Tablo api'de; geç içe aktarma (api bu modülü içe aktarır)."""
    from .api import METRIC_TO_ENGLISH, METRIC_TO_INTERNAL
    return {**METRIC_TO_INTERNAL, **BIRINCIL_EK_ADLAR}, METRIC_TO_ENGLISH


def birincil_ic(ad: str | None) -> str | None:
    """Birincil metrik adı → metrik.aile iç adı: İngilizce ad ya da eş ad (accuracy, log_loss, logloss, Log-Loss...)
    kabul edilir; iç ad (dogruluk) ve bilinmeyen ad olduğu gibi döner (bilinmeyeni degerlendir anlamlı hatayla reddeder)."""
    if ad is None:
        return None
    tablo, _ = _metrik_adlari()
    if ad in tablo:
        return tablo[ad]
    duz = str(ad).strip().lower().replace("-", "_").replace(" ", "_")
    return tablo.get(duz, ad)


def _birincil_hatasi(ad, adlar: list[str], gorev: str) -> ValueError:
    """Bu görevde olmayan birincil metrik: İngilizce adlar + difflib 'did you mean' önerisi."""
    tablo, en = _metrik_adlari()
    adaylar = {en.get(a, a): a for a in adlar}
    adaylar.update({a: a for a in adlar})
    adaylar.update({k: v for k, v in tablo.items() if v in adlar})
    yakin = difflib.get_close_matches(str(ad).strip().lower(), list(adaylar), n=1, cutoff=0.6)
    oneri = f"; did you mean {en.get(adaylar[yakin[0]], adaylar[yakin[0]])!r}?" if yakin else ""
    from .sema import GOREVLER, GOREVLER_EN
    gorev_en = dict(zip(GOREVLER, GOREVLER_EN)).get(gorev, gorev)
    return ValueError(f"primary metric {ad!r} is not available for task {gorev_en!r}: "
                      f"{[en.get(a, a) for a in adlar]} (internal: {adlar}){oneri}")


def _az_kume(veri) -> dict | None:
    """Bootstrap birimi (küme; ab görevinde ayrıca test = madde) AZ_KUME'den azsa denetim satırı: GA anlamlı değil."""
    d = veri.df
    say = {veri.kume_adi: veri.n_kume}
    if veri.gorev == "ab" and "madde" in d:
        say["madde"] = int(d["madde"].nunique())
    ad, n = min(say.items(), key=lambda x: x[1])
    if n >= AZ_KUME:
        return None
    en = {"madde": "tests" if veri.gorev == "ab" else "items", "grup": "clusters (group)", "satir": "rows"}.get(ad, ad)
    return {"kontrol": "küme sayısı (bootstrap GA)", "durum": "uyari", "deger": n, "esik": AZ_KUME,
            "ayrinti": (f"only {n} {en}; CI is not meaningful (cluster bootstrap needs at least {AZ_KUME}; "
                        f"yalnız {n} '{ad}' kümesi, GA anlamsız): the badge is not 'high confidence'")}


# satırı kişi olmayan düzeyler: persona karıştırma ve PPI++ 'kaç insana bedel' anlamsız (evaluate(level=) olduğu gibi gelir)
TOPLU_DUZEY = {"segment": "segment", "study": "study", "calisma": "study", "çalışma": "study", "effect": "effect",
               "etki": "effect"}


def _toplu_duzey(veri, duzey: str | None, madde_ici: bool) -> tuple[str, str] | None:
    """Satırlar kişi değil segment / çalışma / etki tahmini mi: level= ya da her maddede segment başına tek satır.
    Otomatik algı yalnız segment hücreleri maddeler arasında tekrar eden bir ızgara ise (farklı hücre sayısı ≤ madde
    başına en çok satır): kişi düzeyinde ince segment (ör. tam yaş) her maddede tekil olsa da ızgara değildir.
    level= başka bir değerle (ör. "person") verildiyse otomatik algı yapılmaz. Döner (Türkçe neden, İngilizce neden) | None."""
    ad = TOPLU_DUZEY.get(str(duzey).strip().lower()) if duzey else None
    if ad:
        return (f"level={duzey}: satırlar {ad} düzeyinde tahmin, kişi değil",
                f"level={duzey}: rows are {ad}-level estimates, not persons")
    d = veri.df
    if duzey or not (veri.segmentler and "madde" in d and madde_ici):
        return None
    boy = d.groupby(["madde", *veri.segmentler], dropna=False).size()
    hucre = len(d[veri.segmentler].drop_duplicates())
    if int(boy.max()) == 1 and hucre <= int(d.groupby("madde").size().max()):
        return ("her maddede segment başına tek satır: satırlar segment tahmini, kişi değil",
                "one row per segment in every item: rows are segment estimates, not persons")
    return None


# satırı seçim görevi olan düzeyler (conjoint / seçim deneyi): her satır kendi seçenek kümesi
SECIM_DUZEY = ("task", "choice", "choice_task", "gorev", "görev", "secim", "seçim")


def _secim_gorevi(veri, duzey: str | None, olasilik: bool) -> tuple[str, str] | None:
    """Satırlar seçim görevi mi (her görevin kendi seçenekleri): level="task"/"choice" ya da çok sınıfta konumsal
    seçenekler (sınıflar 0..K−1 / 1..K) üzerinde olasılık vektörü + madde sütunu yok; adlı sınıflar ("car", dict
    anahtarları) sınıflandırmadır. Karıştırma bir görevin vektörünü başka seçenek kümesine taşır; PPI++ tek bir soruya
    insan yanıtı ister. Konum kodlu sınıflandırmada (aynı soruyu yanıtlayan kişiler) level="person" ya da item= verin.
    level= başka bir değerle verildiyse otomatik algı yapılmaz. Döner (Türkçe neden, İngilizce neden) | None."""
    ad = str(duzey).strip().lower() if duzey else None
    if ad in SECIM_DUZEY:
        return (f"level={duzey}: satırlar seçim görevi (her görevin kendi seçenekleri), kişi değil",
                f"level={duzey}: rows are choice tasks with their own choice sets, not persons")
    if ad or veri.gorev != "cok_sinif" or not olasilik or "madde" in veri.df or not _konumsal(veri.siniflar):
        return None
    return ("satırlar seçim görevi: konumsal seçenekler (0..K−1) üzerinde olasılık vektörü, madde sütunu yok; vektör "
            "başka görevin seçenek kümesine taşınırdı (aynı soruyu yanıtlayan kişilerse level=\"person\" ya da item= "
            "verin)",
            "rows are choice tasks: probability vectors over positional alternatives (0..K-1) without an item column, "
            "a shuffle would move a vector onto another task's choice set and PPI++ needs human answers to one question "
            "(if the rows are persons answering the same question, pass level=\"person\" or item=)")


def _konumsal(siniflar) -> bool:
    """Sınıf kümesi konum kodu mu (0..K−1 ya da 1..K tamsayı): seçim görevinde seçenek sırası. Adlı sınıflar ("car",
    "train", dict anahtarları) bütün satırlarda aynı seçenek kümesidir: sınıflandırma, seçim görevi sayılmaz."""
    try:
        k = sorted(float(str(s)) for s in siniflar or [])
    except ValueError:
        return False
    return bool(k) and all(v.is_integer() for v in k) and k in ([float(i) for i in range(len(k))],
                                                                  [float(i) for i in range(1, len(k) + 1)])


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


def _anahtar(x):
    """Tahmin hücresinin karşılaştırılabilir anahtarı (vektör → yuvarlanmış demet)."""
    if isinstance(x, dict):
        return tuple(sorted((str(k), _anahtar(v)) for k, v in x.items()))
    if isinstance(x, (np.ndarray, list, tuple)):
        return tuple(np.round(np.asarray(x, float).ravel(), 12).tolist())
    if isinstance(x, (float, np.floating)):
        return "nan" if not np.isfinite(x) else round(float(x), 12)
    return str(x)


def _madde_ici_sabit(veri, sutun: str) -> bool:
    """Tahmin her maddede birimler arasında sabit mi (madde yoksa bütün tabloda). Sabitse karıştırma kontrolü
    kişi bilgisini sınamaz (kontrol = bizim). En az 2 satırlı madde yoksa False (karıştırma zaten etkisiz)."""
    d = veri.df
    v = d[sutun].to_numpy(object)
    gruplar = d.groupby("madde").indices.values() if "madde" in d else [np.arange(len(d))]
    cok = False
    for idx in gruplar:
        if len(idx) < 2:
            continue
        cok = True
        if len({_anahtar(v[i]) for i in idx}) > 1:
            return False
    return cok


DOLGU_ON_EK = "filled_"   # sim.distribution_frame: filled_<yöntem> = hücrede tekdüze vektörle doldurulan yanıt payı


def _dolgu_denetimi(veri, etiket: dict) -> tuple[list[dict], dict]:
    """Kol başına dolgu (geçersiz yanıt → tekdüze vektör) payı denetimi: filled_<yöntem> sayısal sütunlarından.
    Döner (denetim satırları, {tahmin sütunu: {"ortalama", "en_buyuk", "n_hucre", "n_dolgulu"}})."""
    d = veri.df
    den, ozet = [], {}
    for y_ in veri.yontemler:
        c = DOLGU_ON_EK + y_
        if c not in d:
            continue
        x = pd.to_numeric(d[c], errors="coerce").to_numpy(float)
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        ort = float(x.mean())
        ozet[ON_EK + y_] = {"ortalama": ort, "en_buyuk": float(x.max()), "n_hucre": int(len(x)),
                            "n_dolgulu": int((x > 0).sum())}
        cok = ort > DOLGU_ESIK
        den.append({"kontrol": f"dolgu payı (geçersiz → tekdüze): {etiket.get(ON_EK + y_, y_)}",
                    "durum": "uyari" if cok else "tamam", "deger": ort, "sutun": ON_EK + y_,
                    "ayrinti": f"{c}: hücre başına ortalama dolgu payı %{100 * ort:.1f} (en büyük %{100 * x.max():.0f}; "
                               f"{int((x > 0).sum())}/{len(x)} hücrede dolgu)"
                               + (f" · %{100 * DOLGU_ESIK:.0f} eşiğinin üstünde: farkın bir kısmı tekdüze dolgudan gelebilir "
                                  "(filled share above threshold)" if cok else "")})
    return den, ozet


KIYAS_SONRADAN = "sonradan: ön kayıtta yok (post hoc, not in the pre-registration)"


def _onkayit_metinleri(onkayit: list | None, metinler: list | None) -> tuple[list[str], list[str]]:
    """(geçerli, geç): mühür açıklamaları + ön kayıt metinleri; zamanlı metin son mühürden sonraysa 'geç'."""
    zamanlar = [z for z in (_muhur_zamani(r) for r in onkayit or [] if isinstance(r, dict)) if z is not None]
    sinir = max(zamanlar) if zamanlar else None
    gecerli, gec = [], []
    for r in onkayit or []:
        if isinstance(r, dict):
            gecerli += [str(r[k]) for k in ("aciklama", "description") if r.get(k)][:1]
    for t in metinler or []:
        if isinstance(t, dict):
            z = t.get("zaman")
            (gec if sinir is not None and z is not None and int(z) > int(sinir) else gecerli).append(str(t.get("metin", "")))
        else:
            gecerli.append(str(t))
    return gecerli, gec


def _paragraflar(t: str) -> list[str]:
    """Metni paragraflara böler (boş satır ya da başlık satırı ayırır). Bütün ONKAYIT.md her kolun adını bir yerde
    geçirir; a ile b'nin aynı paragrafta geçmesi istenir (G7: 'llm3' bir paragrafta, 'position' taban listesinde)."""
    out, cur = [], []
    for ln in str(t).splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            if cur:
                out.append("\n".join(cur))
            cur = [ln] if ln.strip() else []
            continue
        cur.append(ln)
    if cur:
        out.append("\n".join(cur))
    return out


def _kiyas_kaydi(adlar: list[str], et_: str | None, metinler: list[str]) -> bool:
    """Karşılaştırma ön kayıtta mı: aynı paragrafta a ve b adları (b yoksa yalnız a) ya da metinde kullanıcı etiketi
    geçiyor. İki ad birlikte arandığı için 2 harfli adlar ('L3', 'T3') ön eksiz de eşleşir; tek harfli adlar ön ek ister."""
    import re as _re
    kaliplar = [_re.compile(r"(?<![A-Za-z0-9_])(?:pred_|tahmin_)?" + _re.escape(a) + r"(?![A-Za-z0-9_])")
                if len(a) == 2 else _ad_kalibi(a) for a in adlar if a]
    et_k = (_re.compile(r"(?<![A-Za-z0-9_])" + _re.escape(et_.strip()) + r"(?![A-Za-z0-9_])")
            if et_ and len(et_.strip()) >= 2 else None)
    return any((et_k is not None and et_k.search(t))
               or (kaliplar and any(all(k.search(p) for k in kaliplar) for p in _paragraflar(t)))
               for t in metinler)


_KOR_AD = ("blind", "kor", "kör", "adsiz", "adsız")
_ADLI_AD = ("named", "adli", "adlı", "isimli")


def _sozcukler(x) -> list[str]:
    import re as _re
    return [t for t in _re.split(r"[^0-9A-Za-zÇĞİÖŞÜçğıöşü]+", str(x).casefold()) if t]


def _kiyas_ogeleri(kiyaslar) -> list[tuple]:
    """kiyaslar -> [(a, b, etiket)] ön eksiz adlarla (tuple ya da dict kayıtları)."""
    out = []
    for k_ in kiyaslar or []:
        if isinstance(k_, dict):
            a_, b_, et_ = k_.get("a"), k_.get("b"), k_.get("label", k_.get("etiket"))
        else:
            a_, b_, et_ = (tuple(k_) + (None, None))[:3]
        out.append(tuple(None if x is None else str(x).removeprefix(ON_EK) for x in (a_, b_)) + (et_,))
    return out


def _kor_cifti(a: str | None, b: str | None, et_: str | None) -> bool:
    """Adlı − kör eşleşik çift mi: b, a'nın kör ikizi (b adında 'blind' / 'kör' / 'adsız' sözcüğü, o sözcük ve
    'named' / 'adlı' atılınca a ile aynı ad: llm − llm_blind, gpt_named − gpt_blind) ya da etiket hem adlı hem kör
    sözcüğü taşıyor ("named − blind", "adlı − kör"). Böyle bir çift ezber (tanıma) denetimidir; a o zaman LLM kolu
    sayılır. Yalnız b'nin kör olması yetmez: ours − llm_blind (bizim, kör LLM tabanına karşı) ezber denetimi değil."""
    if not a or not b or a == b:
        return False
    sa, sb = _sozcukler(a), _sozcukler(b)
    if any(t in _KOR_AD for t in sb):
        kok = [t for t in sb if t not in _KOR_AD + _ADLI_AD]
        if kok and kok == [t for t in sa if t not in _ADLI_AD]:
            return True
    se = _sozcukler(et_) if et_ else []
    return any(t in _KOR_AD for t in se) and any(t in _ADLI_AD for t in se)


def _ezber_ciftten(satirlar: list[dict], llm_yok_beyan: bool = False) -> dict:
    """Adlı − kör kayıtlı karşılaştırma satırlarından ezber denetimi: adlı kol anlamlı iyi (GA 0'ı dışlıyor, metrik
    yönünde) → uyari; GA'lar 0'ı içeriyor → tamam; GA yoksa atlandi. "deger" ilk çiftin farkı. llm_yok_beyan:
    ours_uses_llm=False (denetim bizim için değil, çift için)."""
    def ad(c):
        return str(c).removeprefix(ON_EK)
    parca = "; ".join(f"{ad(r_['a'])} − {ad(r_['b'])} {_f(r_['fark'])} [{_f(r_['alt'])}, {_f(r_['ust'])}]"
                      for r_ in satirlar)
    if not satirlar:
        durum, tr, en = "atlandi", "adlı − kör çift değerlendirilemedi", "named − blind pair not evaluated"
    elif any(r_.get("sonuc", "").startswith("A iyi") for r_ in satirlar):
        durum, tr, en = ("uyari", "adlı kol kördan anlamlı iyi: tanıma/ezber şüphesi",
                         "the named arm is significantly better: recognition/memorization suspected")
    elif all(r_.get("sonuc") == "GA yok" for r_ in satirlar):
        durum, tr, en = "atlandi", "GA yok", "no CI"
    else:
        durum, tr, en = "tamam", "adlı ile kör arasında anlamlı kazanç yok", "no significant gain of the named arm"
    if llm_yok_beyan:
        tr += "; bizim kol LLM kullanmıyor, denetim çift için"
        en += "; ours does not use an LLM, ours_uses_llm=False: audit kept for the pair"
    return {"durum": durum, "deger": satirlar[0]["fark"] if satirlar else None,
            "ayrinti": f"adlı − kör {parca} · {tr} (named − blind {parca}: {en})"}


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


KOKEN_SUTUNLARI = ("koken", "origin", "zaman")   # tek köken denetiminde bakılan zaman/köken sütunları
TEK_KOKEN_NOTU = ("single forecast origin: CIs cover series sampling only, not origin-to-origin variation; "
                  "use methods.forecast.backtest(origins=[...])")


AZ_BLOK = 10        # bu kadardan az blokta blok bootstrap GA'sı kaba: denetimde uyarı
COK_AZ_BLOK = 5     # bundan az blokta blok bileşeni güvenilir değil: kartta "GA güvenilir değil" notu


def _blok_kur(veri, blok: str | None, tahmin_gorevi: bool) -> dict:
    """Blok bootstrap sütununu doğrular ve veri.blok'u kurar; tahmin görevinde tek köken uyarısını hazırlar.

    Döner: {"blok", "sinir", "sinir_ek", "denetimler": [...], "tek_koken"} (yalnız ilgili anahtarlar)."""
    d = veri.df
    out: dict = {"denetimler": []}
    if blok is not None and blok not in d.columns:
        raise ValueError(f"block_col {blok!r} is not a column of the evaluated frame")
    if blok is not None and d[blok].isna().any():
        raise ValueError(f"block_col {blok!r} has {int(d[blok].isna().sum())} missing (NaN) rows in the evaluated "
                         "frame; fill or drop them before block bootstrap")
    n_blok = int(d[blok].nunique()) if blok is not None else None
    if blok is not None and n_blok > 1:
        veri.blok = blok
        out["blok"] = {"sutun": blok, "n_blok": n_blok}
        out["sinir"] = (f"Bootstrap iki yönlü (block_col={blok}): {n_blok} blok ve {veri.n_kume} "
                        f"'{veri.kume_adi}' kümesi her tekrarda bağımsız yerine koymalı örneklenir, satır ağırlığı "
                        "blok sayımı × küme sayımı; GA'lar hem bloklar (ör. kökenler/haftalar) arası hem seriler/kümeler "
                        "arası değişkenliği (blokları aşan seri bağımlılığı dahil) kapsar.")
        if n_blok < AZ_BLOK:
            ek = (f"Yalnız {n_blok} blok (< {AZ_BLOK}): blok bileşeninin GA katkısı kaba"
                  + (f"; {COK_AZ_BLOK} bloktan az olduğu için bloklar arası değişkenlik bu GA'larla tahmin edilemez "
                     "(GA güvenilir değil: CI not reliable across blocks), daha çok köken/hafta ile backtest edin."
                     if n_blok < COK_AZ_BLOK else "; blok sayısını artırmak (daha çok köken/hafta) GA'yı sağlamlaştırır."))
            out["sinir_ek"] = ek
            out["denetimler"].append({"kontrol": "blok bootstrap blok sayısı", "durum": "uyari", "deger": n_blok,
                                      "ayrinti": ek})
    elif blok is not None:
        out["denetimler"].append({"kontrol": "blok bootstrap", "durum": "uyari", "deger": n_blok,
                                  "ayrinti": f"block_col={blok} tek değerli: blok bootstrap uygulanmadı, varsayılan "
                                             "kümeler kullanıldı"})
    if not tahmin_gorevi:
        return out
    # tek köken denetimi block_col'dan bağımsız: köken sütunu (yoksa zaman sütunu)
    sutun = next((c for c in KOKEN_SUTUNLARI if c in d.columns), None)
    if sutun is not None and int(d[sutun].nunique()) == 1:
        out["tek_koken"] = {"sutun": sutun, "deger": str(d[sutun].iloc[0]), "not": TEK_KOKEN_NOTU}
        out["denetimler"].append({"kontrol": "tek tahmin kökeni (single origin)", "durum": "uyari", "deger": 1,
                                  "ayrinti": f"'{sutun}' tek değerli ({d[sutun].iloc[0]}): " + TEK_KOKEN_NOTU
                                             + (f"; block_col={blok} blokları köken değil, kökenler arası değişkenlik "
                                                "yine kapsanmaz" if veri.blok is not None and veri.blok != sutun else "")})
    return out


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
                kiyaslar: list | None = None, ezber_llm: bool | None = None, *, tahmin_gorevi: bool = False,
                ga_yontemi: str = "percentile", n_yapilandirma: int | None = None,
                onkayit_metin: list[str] | None = None, blok: str | None = None) -> dict:
    """bizim/ham: tahmin_ ön eki olmadan yöntem adı. bizim verilmezse SON yöntem (önceden belirlenmeli;
    en iyiyi sonradan seçmek iyimser yanlılık yaratır, denetimde işaretlenir).

    tahmin_gorevi: talep/zaman serisi tahmini (api task="forecast"): sürekli görev + WMAPE/yanlılık metrikleri; persona
    karıştırma kontrolü, insan test-retest tavanı ve PPI++ 'kaç insana bedel' uygulanamaz (kartta gerekçesiyle).
    ga_yontemi: "percentile" (varsayılan) | "basic" (2θ − yüzdelikler); sütun, merdiven, manşet ve fark GA'larında.
    Yüzdelik GA nokta tahminini dışarıda bırakırsa uyarı + denetim satırı + kart etiketi.
    n_yapilandirma: denenen yapılandırma sayısı (seçim yanlılığı satırına yazılır). onkayit_metin: ön kayıt metinleri
    ('bizim' adı önceden yazılmış mı diye aranır; mühür kayıtlarıyla birlikte).

    oof_tabanlar: hedef etiketiyle eğitilmiş ama çapraz uydurmalı (kat dışı, OOF) tahmin sütunları; etiketsiz
    tabanlar gibi taban basamağı olur. Rozet ve manşet en güçlü tabana (otomatik + bildirilen) karşı; hiç bildirilmiş
    taban yokken en güçlüsü rastgele/marjinal ise kart "yalnız basit tabanlar" der.
    tavan_sutun: df'deki model/oracle tavan sütunu (tahmin_ ön eksiz); verilirse insan tekrar tavanının yerine geçer.
    carpan_en_fazla: tek maddeli (kitle ortalaması) çarpan bu kadar satırdan büyükse sabit tohumlu alt örneklemde
    hesaplanır (None: alt örneklem yok).
    tavan_yarim: örneklem gürültüsü tavanı girdisi {"n", "s2": grup boyu / iç varyans, "A", "B": R × n yarı
    ortalamaları (split-half denetimi), "sutun": satır no sütunu, "R"}; api.evaluate(noise_ceiling=...) kurar. kor: kör (adsız) kol; bizim − kör eşleşik satırı.
    kiyaslar: [(a, b | None, etiket[, posthoc])] ya da {"a", "b", "label", "posthoc"} kayıtlı eşleşik karşılaştırmalar
    (düşme koşulları). Ön kayıt metni (mühür açıklaması / onkayit_metin) varsa her satırın a/b adı (ya da etiketi)
    orada aranır: bulunmayan satır tur="posthoc", kayit="sonradan"; b en güçlü tabandan zayıf bir tabansa satırda
    zayif_taban + denetim uyarısı. Dağılım görevinde taban_marjinal ayar-B bilgisidir (hedef sorunun öbür
    hücreleri): merdivende etiketli, rozet ayar-A tabanlarına karşı. filled_<yöntem> sütunları kol başına dolgu
    denetimi olur. Madde içinde sabit tahminde karıştırma kontrolü atlanır. ezber_llm: bizim yöntem LLM
    kolu içeriyor mu; ezber uyarısında rozet gerekçesine ek düşer. None: bizim ham kol, adlı − kör çiftin adlı kolu ya
    da kör kolun sahibiyse evet; değilse ezber sonucu verildiyse evet (ihtiyatlı: ours="stack", raw="llm" çoğu zaman
    LLM'in üstüne kurulu); LLM kolu başkaysa denetimde ours_uses_llm=False'u söyleyen satır düşer. kiyaslar içinde b'nin
    a'nın kör ikizi olduğu (llm − llm_blind) ya da etiketi hem adlı hem kör sözcüğü taşıyan ("named − blind") çift adlı
    − kör ezber denetimidir: ezber sonucu yokken ezber satırı ondan dolar (ours − llm_blind çift sayılmaz).
    'zaman' biçim biçim çözülür (karışık biçim, gün-önce, saat dilimi); çözülemeyen satır zaman denetimlerinin dışında
    kalır ve sayılır.
    blok: blok bootstrap sütunu (ör. tahmin kökeni / hafta): küme yerine bu sütunun her değeri bir blok olarak
    yerine koymalı örneklenir (blok içindeki bütün satırlar/kümeler birlikte); sütun, merdiven, manşet ve eşleşik fark
    GA'ları. Tek değerliyse uygulanmaz (varsayılan kümeye dönülür, denetimde not).
    duzey: "segment" / "study" / "effect" (ya da her maddede segment başına tek satır) iken satır kişi değildir:
    persona karıştırma ve PPI++ 'kaç insana bedel' gerekçesiyle uygulanamaz (ozet["carpan_uygulanamaz"]). Madde
    başına sabit tahmin denetimde uyarılır (ozet["sira_tanimsiz"]); ezber_llm=False iken ham kol LLM kolu sayılır.
    duzey="task" / "choice" (ya da çok sınıfta konumsal sınıflarda olasılık vektörü + madde sütunu yok) iken satırlar
    seçim görevidir: aynı iki satır gerekçesiyle uygulanamaz. Sürekli görevde madde içi çift toplamı ≤ CIFT_EN_FAZLA
    ise (ya da birincil="cift_isaret") madde içi çift sıralama doğruluğu (cift_isaret = pair_sign) ikincil metrik olur.
    bizim/ham oof_tabanlar içindeyse PPI++ çarpanında dongusel gibi sayılır (aynı etiketlerle eğitilmiş)."""
    birincil = birincil_ic(birincil)
    if gercek_yok(df):
        return bekleyen(df, baslik, veri_adi, gorev=gorev, yontemler=yontemler, bizim=bizim, ham=ham,
                        etiketler=etiketler, birincil=birincil, onkayit=onkayit, sinirlar=sinirlar, duzey=duzey,
                        girdi_kaynagi=girdi_kaynagi, tohum=tohum, B=B, tahmin_gorevi=tahmin_gorevi,
                        etiketsiz_tabanlar=etiketsiz_tabanlar, oof_tabanlar=oof_tabanlar)
    if ga_yontemi not in ("percentile", "basic"):
        raise ValueError(f"ci must be 'percentile' or 'basic', not {ga_yontemi!r}")
    if tahmin_gorevi:
        gorev = "surekli"
        birincil = birincil or "wmape"
    if birincil in ISARETLI_METRIKLER:   # işaretli: "küçük iyi" sıralaması büyük negatif sapmayı kazandırırdı
        raise ValueError(f"{'bias (yanlilik)' if birincil == 'yanlilik' else birincil} is a signed diagnostic, not a "
                         "primary metric: use wmape/mae (or auc/brier) and read it from the metric table")
    veri = hazirla(df, gorev, yontemler, siniflar)
    d = veri.df
    blok_bilgi = _blok_kur(veri, blok, tahmin_gorevi)
    etiketler = dict(etiketler or {})
    madde_var = "madde" in d
    # her madde tek satırsa madde içi ölçüt (madde_ici_kor, demeaned_r2) ve karıştırma kontrolü tanımsız
    madde_ici = madde_var and bool((d.groupby("madde").size() > 1).any())
    olasilik = veri.gorev != "cok_sinif" or isinstance(d[ON_EK + veri.yontemler[0]].iloc[0], np.ndarray)
    toplu = None if tahmin_gorevi else (_toplu_duzey(veri, duzey, madde_ici) or _secim_gorevi(veri, duzey, olasilik))
    birim_aralik = veri.gorev == "surekli" and d["gercek"].between(0, 1).all()
    oranli = tahmin_gorevi or birincil in ORAN_METRIKLERI
    # sürekli görev + madde başına birden çok satır: madde içi çift sıralama doğruluğu (pair_sign) ikincil metrik;
    # toplam çift sayısı CIFT_EN_FAZLA'yı aşarsa (büyük madde, kişi × soru) kendiliğinden eklenmez, not düşülür
    cift = veri.gorev == "surekli" and madde_ici and not tahmin_gorevi
    if cift and birincil != "cift_isaret":
        boy = d.groupby("madde").size().to_numpy(np.int64)
        n_cift = int((boy * (boy - 1) // 2).sum())
        if n_cift > CIFT_EN_FAZLA:
            cift = False
            veri.notlar.append(f"pair_sign (cift_isaret) eklenmedi: {n_cift} madde içi çift > {CIFT_EN_FAZLA} "
                               "(her bootstrap tekrarında yeniden kurulur, yavaş); istenirse primary=\"pair_sign\" "
                               f"(pair_sign skipped: {n_cift} within-item pairs > {CIFT_EN_FAZLA}; pass "
                               "primary=\"pair_sign\" to get it)")
    metrikler = aile(veri.gorev, olasilik, madde_ici and veri.gorev in ("ikili", "surekli"), birim_aralik, oran=oranli,
                     cift=cift)
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
        raise _birincil_hatasi(birincil, adlar, veri.gorev)
    M = {m.ad: m for m in metrikler}[birincil]
    bizim_sec_oto = bizim is None
    bizim = bizim or veri.yontemler[-1]
    agirlik = d["agirlik"].to_numpy(float) if "agirlik" in d else None

    # ---- sütunlar: baseline, yöntem, kalibre, kontrol, tavan
    turler, etiket = {}, {}
    if baseline:
        ozet_s = "median" if birincil in ("mae", "bir_eksi_mae", "wmape") else "mean"
        for ad, (et, deg) in merdiven.baselineler(veri, ozet_s, tohum).items():
            c = f"taban_{ad}"
            d[c] = deg if not isinstance(deg, list) else pd.Series(deg, dtype=object).to_numpy()
            if tahmin_gorevi:   # test penceresinin gerçeğinden çapraz uydurulur: başlangıçtan ileriye bakar
                et = f"{et} (ileriye bakan / look-ahead)"
            turler[c], etiket[c] = "baseline", et
    taban_turu = {c: "otomatik" for c in turler}
    # bilgi düzeyi: dağılımda taban_marjinal hedef sorunun ÖBÜR hücrelerinin gerçek cevabını okur (bench ayar B);
    # ours bunu görmez (ayar A). Merdivende etiketle gösterilir, rozet yalnız ayar-A tabanlarına karşıdır.
    bilgi_b = [c for c in turler if veri.gorev == "dagilim" and c == "taban_marjinal"]
    for c in bilgi_b:
        etiket[c] += f" · {BILGI_B_ETIKET}"
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
    sabit_kontrol = []   # madde içinde sabit tahmin: karıştırma kişi bilgisini sınamaz, atlanır
    grup_kontrol = []    # grup düzeyi, madde yok: karıştırma yalnız grupları permüte eder, atlanır
    tek_kontrol = []     # her madde tek satır: madde içi karıştırma tahmini yerinde bırakır (kontrol = bizim), atlanır
    toplu_kontrol = []   # segment / çalışma / etki düzeyi: satır kişi değil, persona karıştırma uygulanamaz
    for y_ in ([] if tahmin_gorevi else dict.fromkeys([bizim] + ([ham] if ham else []))):
        if madde_var and not madde_ici:
            tek_kontrol.append(ON_EK + y_)
            continue
        if toplu:
            toplu_kontrol.append(ON_EK + y_)
            continue
        if _madde_ici_sabit(veri, ON_EK + y_):
            sabit_kontrol.append(ON_EK + y_)
            continue
        if duzey == "grup" and not madde_var:   # satırlar grup ortalaması (aggregate_by), madde yok: karıştırma
            grup_kontrol.append(ON_EK + y_)     # grupların tahminini birbirine taşır, persona takası değil
            continue
        c = f"kontrol_karistir_{y_}"
        d[c] = merdiven.karistir(veri, ON_EK + y_, tohum + 1)
        turler[c], etiket[c] = "kontrol", f"{etiketler.get(y_, y_)} · " + (
            "tahmin karıştırma (persona takası değil)" if veri.gorev == "cok_sinif" else "karıştırılmış")
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
    elif "tekrar" in d and tahmin_gorevi:
        tavan_metrik = []
        veri.notlar.append("tahmin görevi: 'tekrar' (insan test-retest) sütunu tavan olarak kullanılmadı")
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
    orn = Orneklem(veri.kume, B=B, B_toplu=B_toplu, tohum=tohum,
                   madde_kume=veri.kume_adi == "madde" and veri.blok is None, iki_yon=veri.iki_yon)
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

    def gaa(r, theta):
        return ga(r, nokta=theta, yontem=ga_yontemi)

    sonuclar = {}
    ga_disi = []   # yüzdelik GA'nın dışında kalan nokta tahminleri (birincil metrik)
    for c in turler:
        sonuclar[c] = {"tur": turler[c], "etiket": etiket[c], "metrikler": {}}
        for m in metrikler:
            if m.ad in nokta.get(c, {}):
                if m.ad == birincil:
                    pa, pu = ga(tekrar[c][m.ad])
                    th = nokta[c][m.ad]
                    tol = 1e-9 * max(1.0, abs(th)) if np.isfinite(th) else 0.0   # sıfır varyanslı kol: kayan nokta
                    if np.isfinite(pa) and np.isfinite(pu) and np.isfinite(th) and not (pa - tol <= th <= pu + tol):
                        ga_disi.append((c, th, pa, pu))
                a, u = gaa(tekrar[c][m.ad], nokta[c][m.ad])
                n = int(len(d) if girdiler[c][2] is None else girdiler[c][2].sum())
                sonuclar[c]["metrikler"][m.ad] = {"deger": nokta[c][m.ad], "alt": a, "ust": u, "n": n}

    def deg(c):
        return nokta.get(c, {}).get(birincil, np.nan)

    def fark(a, b, met=birincil):
        f = tekrar[a][met] - tekrar[b][met]
        al, us = gaa(f, nokta[a][met] - nokta[b][met])
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
        a, u = gaa(tekrar[c][birincil], deg(c))
        r = {"sutun": c, "etiket": etiket[c], "tur": turler[c], "deger": deg(c), "alt": a, "ust": u,
             "fark_onceki": None, "fark_alt": None, "fark_ust": None, "p": None}
        if c in bilgi_b:
            r["bilgi"] = "B"
        if i:
            r["fark_onceki"], r["fark_alt"], r["fark_ust"], r["p"] = fark(c, adim[i - 1])
        merd.append(r)

    # ---- manşet
    bc = ON_EK + bizim
    tabanlar = [c for c in turler if turler[c] == "baseline" and np.isfinite(deg(c))]
    # rozet / manşet: bizimle aynı bilgi düzeyindeki (ayar A) en güçlü taban; ayar-B tabanı yalnız merdivende ve denetimde
    tabanlar_a = [c for c in tabanlar if c not in bilgi_b] or tabanlar
    en_iyi_b = max(tabanlar_a, key=lambda c: M.yon * deg(c)) if tabanlar_a else None
    b_ustte = [c for c in tabanlar if c in bilgi_b and en_iyi_b and c != en_iyi_b
               and M.yon * deg(c) > M.yon * deg(en_iyi_b)]
    bildirilen = [c for c in tabanlar if taban_turu.get(c) in ("etiketsiz", "oof")]
    # hiç taban bildirilmediyse ve en güçlüsü rastgele/marjinal ise rozet yalnız basit bir tabana karşıdır: açıkça yazılır
    yalniz_basit = bool(en_iyi_b) and not bildirilen and en_iyi_b in ("taban_rastgele", "taban_marjinal")
    rakipler = [c for c in yontem_c if c != bc and np.isfinite(deg(c))]
    en_iyi_r = max(rakipler, key=lambda c: M.yon * deg(c)) if rakipler else None

    def kutu(c):
        if c is None or c not in nokta:
            return None
        a, u = gaa(tekrar[c][birincil], deg(c))
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
    elif not manset["tavan"] and tahmin_gorevi:
        cumle += " · tavan uygulanamaz (tahmin görevi: insan test-retest yok; model/oracle tavanı için ceiling_col=)"
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
                # işaretli metrik (yanlılık): 0'a yakın olan iyidir, ham fark yönü değil
                iyi = (bool(abs(nokta[bc][met]) < abs(nokta[b][met])) if met in ISARETLI_METRIKLER
                       else bool(yon * f > 0))
                karsi.append({"a": bc, "b": b, "metrik": met, "fark": f, "alt": a, "ust": u, "p": p,
                              "a_iyi": iyi})

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
        if "taban_marjinal" in bilgi_b:
            den[-1]["ayrinti"] += (" · ayar-B tabanı: hedef sorunun öbür grup/hücrelerinin cevabını okur (örtüşen segment "
                                   "şemalarında dışarıda bırakılan hücrenin kişileri de içinde olabilir); rozete dayanak "
                                   "değil (setting-B baseline, not used for the badge)")
            den[-1]["bilgi"] = "B"
    if en_iyi_b and not (madde_den and en_iyi_b == "taban_marjinal"):
        den.append(denetim.fark_denetimi("en güçlü basit tabanı geçiyor mu", bc, en_iyi_b, tekrar, nokta, M, "bizim − taban"))
    tb_den = denetim.taban_belirsizligi(tekrar, tabanlar_a, birincil, M.yon, etiket, bizim=bc, nokta=nokta,
                                        ga_yontemi=ga_yontemi)   # ilk iki taban GA içinde mi (seçim gürültüsü)
    if tb_den:
        den.append(tb_den)
    if en_iyi_r:
        f_, a_, u_, p_ = fark(bc, en_iyi_r)
        # yalnız eşleşik GA 0'ı kötü yönde dışlarsa uyarı; GA 0'ı içeriyorsa nokta işareti ne olursa olsun berabere
        ga_var = bool(np.isfinite(a_) and np.isfinite(u_))
        kotu = ga_var and (u_ < 0 if M.yon > 0 else a_ > 0)
        iyi_ = ga_var and (a_ > 0 if M.yon > 0 else u_ < 0)
        # GA yoksa beraberlik de söylenemez: eski davranış (nokta işareti) korunur, "yalnız nokta tahmini" diye
        nokta_kotu = not ga_var and bool(M.yon * f_ < 0)
        den.append({"kontrol": "en güçlü diğer yöntem", "durum": "uyari" if kotu or nokta_kotu else "tamam", "deger": f_,
                    "ayrinti": f"{birincil}: bizim − {etiket[en_iyi_r]} = {_f(f_)} [{_f(a_)}, {_f(u_)}], p={_f(p_, 3)}"
                               + (" · başka bir yöntem bizden iyi (another method is better: paired CI excludes 0)"
                                  if kotu else "" if iyi_ else
                                  " · berabere: eşleşik GA 0'ı içeriyor (tied: paired CI includes 0)" if ga_var else
                                  " · GA yok, nokta tahmininde başka yöntem önde (no CI; another method ahead on the "
                                  "point estimate only)" if nokta_kotu else " · GA yok (no CI)")})
    takas_ipucu = ('persona takası için takas kolunu compare=(ours, swap, "persona swap") ile verin '
                   '(pass the swap arm via compare=(ours, swap, "persona swap"))')
    if bc in kontrol_sutun and veri.gorev == "cok_sinif":
        # N'li tanıma / çok sınıf: karıştırma seçilen etiketi başka birime taşır; persona takası değildir
        den.append(denetim.fark_denetimi("tahmin karıştırma (persona takası değil)", bc, kontrol_sutun[bc],
                                         tekrar, nokta, M, "bizim − karıştırılmış tahmin"))
        den[-1]["ayrinti"] += f" · madde içinde tahmin etiketleri birimler arasında karıştırıldı; {takas_ipucu}"
    elif bc in kontrol_sutun:
        den.append(denetim.fark_denetimi("persona karıştırma (birey bilgisi kullanılıyor mu)", bc, kontrol_sutun[bc],
                                         tekrar, nokta, M, "bizim − karıştırılmış"))
    elif bc in sabit_kontrol:
        den.append({"kontrol": "persona karıştırma (birey bilgisi kullanılıyor mu)", "durum": "atlandi", "deger": None,
                    "uygulanamaz": True,
                    "ayrinti": "atlandı: tahmin her maddede birimler arasında sabit (kişi bilgisi yok); karıştırma "
                               "bizimle aynı sonucu verirdi (skipped: predictions constant within item, no person "
                               f"information); {takas_ipucu}"})
    elif bc in toplu_kontrol:
        den.append({"kontrol": "persona karıştırma (birey bilgisi kullanılıyor mu)", "durum": "atlandi", "deger": None,
                    "uygulanamaz": True,
                    "ayrinti": f"uygulanamaz: {toplu[0]}; karıştırma kişi bilgisini sınamaz (not applicable: "
                               f"{toplu[1]}; a shuffle cannot test person information)"})
    elif bc in tek_kontrol:
        den.append({"kontrol": "persona karıştırma (birey bilgisi kullanılıyor mu)", "durum": "atlandi", "deger": None,
                    "uygulanamaz": True,
                    "ayrinti": "atlandı: her madde tek satır; madde içinde karıştırılacak başka birim yok, kontrol bizimle "
                               "aynı olurdu (skipped: one row per item, a within-item shuffle leaves every prediction in "
                               f"place); {takas_ipucu}"})
    elif bc in grup_kontrol:
        den.append({"kontrol": "persona karıştırma (birey bilgisi kullanılıyor mu)", "durum": "atlandi", "deger": None,
                    "uygulanamaz": True,
                    "ayrinti": "atlandı: satırlar grup ortalaması (aggregate_by), madde yok; karıştırma grupların "
                               "tahminini birbirine taşır, kişi bilgisini sınamaz (skipped: group-level rows without "
                               f"item, a shuffle would permute groups, not persons); {takas_ipucu}"})
    if "madde_ici_kor" in nokta.get(bc, {}) and bc in kontrol_sutun:
        # karıştırma kontrolü yoksa (tahmin görevi) bizim − bizim = 0 anlamsız satırı yazılmaz
        mm = {m.ad: m for m in metrikler}["madde_ici_kor"]
        den.append(denetim.fark_denetimi("kişiye özgü sinyal (madde_ici_kor)", bc, kontrol_sutun[bc], tekrar, nokta, mm,
                                         "bizim − karıştırılmış"))
    if "demeaned_r2" in nokta.get(bc, {}):  # sıfır = madde ortalaması; karıştırma burada sıfır hipotezi olamaz
        v_ = nokta[bc]["demeaned_r2"]
        a_, u_ = ga(tekrar[bc]["demeaned_r2"])
        kim, ort = (("seri içi sinyal", "seri ortalamasından") if tahmin_gorevi
                    else ("kişiye özgü sinyal", "madde ortalamasından"))
        den.append({"kontrol": f"{kim} (demeaned_r2 > 0)", "durum": "tamam" if a_ > 0 else "uyari", "deger": v_,
                    "ayrinti": f"de-meaned R² {_f(v_)} [{_f(a_)}, {_f(u_)}]"
                               + ("" if a_ > 0 else f" · {'seri içi' if tahmin_gorevi else 'kişiye özgü'} sapmalar {ort} "
                                  "kötü ya da ayırt edilemiyor; genlik kalibrasyonu gerekebilir")})
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
    if b_ustte:
        den.append({"kontrol": "bilgi düzeyi (ayar A / ayar B)", "durum": "uyari", "deger": len(b_ustte), "bilgi": "B",
                    "ayrinti": f"ayar-B tabanı {', '.join(etiket[c] for c in b_ustte)} ({_f(deg(b_ustte[0]))}) en güçlü "
                               f"ayar-A tabanından ({etiket[en_iyi_b]} {_f(deg(en_iyi_b))}) iyi; hedef sorunun öbür "
                               "hücrelerinin gerçek cevabını kullandığı için rozet ayar-A tabanına karşıdır "
                               "(setting-B baseline beats the strongest setting-A one; badge is against setting A)"})
    dolgu_den, dolgu = _dolgu_denetimi(veri, etiket)
    den.extend(dolgu_den)
    den.append(denetim.cokus(yay, bc, birincil=birincil))
    sira_tanimsiz = {}
    if not madde_var or madde_ici:   # her madde tek satırsa madde içi sıra metriği zaten yok
        # bildirilen tabanlar da: sabit taban sıra metriğinde NaN olur ve merdivenden sessizce düşerdi (G2 #11)
        sabit_den, sira_tanimsiz = denetim.sabit_tahmin(
            veri, yontem_c + ([ON_EK + ham] if ham else [])
            + [c for c in turler if taban_turu.get(c) in ("etiketsiz", "oof")], etiket)
        if sabit_den:
            den.append(sabit_den)
    dz = denetim.dengesizlik(veri, birincil=birincil)
    if dz:
        den.append(dz)
    # ham kol (raw=) LLM koludur: bizim LLM kullanmasa da ezber / kontaminasyon denetimi ham kol için kalır.
    # compare= içindeki adlı − kör çift (llm − llm_blind) de bir LLM kolu ve kendisi bir ezber (tanıma) denetimidir.
    kor_ciftleri = [(a_, b_, et_) for a_, b_, et_ in _kiyas_ogeleri(kiyaslar) if _kor_cifti(a_, b_, et_)]
    llm_kollari = list(dict.fromkeys(([ham] if ham else []) + [a_ for a_, _, _ in kor_ciftleri]
                                     + ([bizim] if kor is not None else [])))
    # bizim LLM kolu mu: açık beyan (ours_uses_llm) ya da bizim ham kol / adlı − kör çiftin adlı kolu / kör kolun
    # sahibi; değilse eski varsayım: ezber sonucu verildiyse evet. Ham kol varken bizim çoğu zaman onun kalibre /
    # yığın sürümüdür (ours="stack", raw="llm"): ihtiyatlı varsayım korunur, satır ours_uses_llm=False'u söyler.
    bizim_kendi_llm = bizim in llm_kollari
    bizim_llm = ezber_llm if ezber_llm is not None else (bizim_kendi_llm or bool(ezber_json))
    llm_yok = ezber_llm is False and not ham and not kor_ciftleri
    ham_llm_notu = (f" · bizim kol LLM kullanmıyor (ours_uses_llm=False); denetim ham LLM kolu ({ham}) için "
                    f"(audit kept for the raw LLM arm)" if ezber_llm is False and ham else
                    f" · bizim kol LLM kullanmıyor (ours_uses_llm=False); denetim adlı − kör çifti "
                    f"({', '.join(f'{a_} − {b_}' for a_, b_, _ in kor_ciftleri)}) için (audit kept for the named − "
                    f"blind pair)" if ezber_llm is False and kor_ciftleri else "")
    if llm_yok and not kesim:
        den.append({"kontrol": "kontaminasyon (kesim öncesi/sonrası)", "durum": "atlandi", "deger": None, "uygulanamaz": True,
                    "ayrinti": "LLM kolu yok (ours_uses_llm=False): eğitim kesim tarihi denetimi uygulanamaz "
                               "(no LLM arm: contamination audit not applicable)"})
    else:
        den.append(denetim.kontaminasyon(veri, {c: girdiler[c] for c in sec}, M, kesim))
        den[-1]["ayrinti"] += ham_llm_notu
    zc = denetim.zaman_cozumu(veri)
    if zc:
        den.append(zc)
    den.extend(denetim.pozisyon_takasi(veri))
    if llm_yok and not ezber_json:
        den.append({"kontrol": "ezber testi (LLM veriyi biliyor mu)", "durum": "atlandi", "deger": None, "uygulanamaz": True,
                    "ayrinti": "LLM kolu yok (ours_uses_llm=False): ezber denetimi uygulanamaz "
                               "(no LLM arm: memorization audit not applicable)"})
    elif kor_ciftleri and not ezber_json:   # adlı − kör çift ezber denetimi: sonuç kayıtlı karşılaştırmalardan dolar
        ezber_cift = {"kontrol": "ezber testi (LLM veriyi biliyor mu)", "durum": "atlandi", "deger": None,
                      "ayrinti": ""}
        den.append(ezber_cift)
    else:
        den.append(denetim.ezber(ezber_json))
        den[-1]["ayrinti"] += ham_llm_notu
    if tahmin_gorevi:
        den.append({"kontrol": "görev türü: tahmin", "durum": "tamam", "deger": None,
                    "ayrinti": "talep/zaman serisi tahmini (task=forecast): persona karıştırma kontrolü, insan test-retest "
                               "tavanı ve PPI++ 'kaç insana bedel' uygulanamaz (insan yanıtı ya da kişi bilgisi yok); "
                               "atlandı. Metrikler WMAPE ve yanlılık içerir."})
    den.extend(blok_bilgi.get("denetimler", []))
    if ga_disi:
        ayr = "; ".join(f"{etiket.get(c, c)} {_f(th)} ∉ [{_f(a)}, {_f(u)}]" for c, th, a, u in ga_disi[:4])
        warnings.warn(f"percentile bootstrap CI excludes the point estimate of {birincil} for {len(ga_disi)} column(s) "
                      f"({ayr}); consider ci='basic'", UserWarning, stacklevel=3)
        den.append({"kontrol": "bootstrap GA (nokta tahmini yüzdelik GA dışında)",
                    "durum": "uyari" if ga_yontemi == "percentile" else "tamam", "deger": len(ga_disi),
                    "ayrinti": f"{birincil}: {ayr}. Oran/toplu metrikte yüzdelik bootstrap yanlı olabilir; "
                               + ("ci='basic' (2θ − yüzdelikler) deneyin." if ga_yontemi == "percentile"
                                  else "kartta temel (basic) GA kullanıldı.")})
    if bizim_sec_oto:
        den.append({"kontrol": "seçim yanlılığı", "durum": "uyari",
                    "ayrinti": "'bizim' yöntem açıkça belirtilmedi (son sütun alındı); yöntemi önceden belirleyin", "deger": None})
    elif len(veri.yontemler) > 1:
        den.append(_secim_yanliligi(veri, yontem_c, bc, bizim, deg, M, onkayit, onkayit_metin, n_yapilandirma))
    az = _az_kume(veri)
    if az:
        warnings.warn(az["ayrinti"], UserWarning, stacklevel=3)
        den.append(az)
    for n_ in veri.notlar:
        den.append({"kontrol": "girdi notu", "durum": "uyari" if "düşürüldü" in n_ else "tamam", "ayrinti": n_, "deger": None})

    # ---- kayıtlı karşılaştırmalar (düşme koşulları) ve kör kol
    n_kume = veri.n_kume

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
        elif ya > 0:  # yer: basılan ham GA'nın (alt, ust) 0'a göre yeri; küçük-iyi metrikte A iyi = GA 0'ın altında
            sonuc = f"A iyi (GA 0'ın {'üstünde' if M.yon > 0 else 'altında'})" if b is not None else "0'ın üstünde"
        elif yu < 0:
            sonuc = f"B iyi (GA 0'ın {'altında' if M.yon > 0 else 'üstünde'})" if b is not None else "0'ın altında"
        else:
            sonuc = "belirsiz (GA 0'ı içeriyor)"
        return {"a": a, "b": b, "etiket": et_, "tur": tur_, "metrik": birincil, "fark": f_, "alt": al_, "ust": us_,
                "p": p_, "a_iyi": None if b is None else bool(yf > 0), "n": n_kume, "sonuc": sonuc}

    kayitli = []
    cift_sonuc = []   # adlı − kör çiftlerin satırları (ezber denetimi)
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
    # kayıtlı mı sonradan mı: ön kayıt metni (mühür açıklaması / prereg_texts) varsa a/b adları orada aranır;
    # açık posthoc=True (4. öğe ya da dict anahtarı) her zaman "sonradan". Metin yoksa satır eskisi gibi "kayitli".
    on_metin, _gec = _onkayit_metinleri(onkayit, onkayit_metin)
    metin_var = bool(on_metin or _gec)
    zayif = []
    for k_ in kiyaslar or []:
        if isinstance(k_, dict):
            a_, b_, et_, ph_ = k_.get("a"), k_.get("b"), k_.get("label", k_.get("etiket")), k_.get("posthoc")
        else:
            a_, b_, et_, ph_ = (tuple(k_) + (None, None))[:4]
        ac, bc_ = _cozum(a_), _cozum(b_)
        r_ = _kiyas(ac, bc_, et_ or (f"{etiket.get(ac, ac)} − {etiket.get(bc_, bc_)}" if bc_ else
                                     f"{etiket.get(ac, ac)} (0'a karşı)"), "kayitli")
        if ph_ is None and metin_var:
            adlar_ = [str(x).removeprefix(ON_EK) for x in (a_, b_) if x is not None]
            ph_ = not _kiyas_kaydi(adlar_, et_, on_metin)
        if ph_ is not None:
            r_["kayit"] = "sonradan" if ph_ else "kayitli"
        if ph_:
            r_["tur"] = "posthoc"
            r_["etiket"] = f"{r_['etiket']} ({KIYAS_SONRADAN})"
        # b bir taban ama en güçlü (aynı bilgi düzeyindeki) taban değil: kayıtlı fark zayıf tabana karşı
        if (bc_ and en_iyi_b and bc_ != en_iyi_b and turler.get(bc_) == "baseline" and bc_ not in bilgi_b
                and M.yon * deg(bc_) < M.yon * deg(en_iyi_b)):
            r_["zayif_taban"] = {"b": bc_, "b_deger": deg(bc_), "en_guclu": en_iyi_b,
                                 "en_guclu_etiket": etiket[en_iyi_b], "en_guclu_deger": deg(en_iyi_b)}
            zayif.append(r_)
        kayitli.append(r_)
        if _kor_cifti(*(None if x is None else str(x).removeprefix(ON_EK) for x in (a_, b_)), et_):
            cift_sonuc.append(r_)
    if kor_ciftleri and not ezber_json:
        ezber_cift.update(_ezber_ciftten(cift_sonuc, ezber_llm is False and not ham))
    if zayif:
        den.append({"kontrol": "kayıtlı karşılaştırma zayıf tabana karşı", "durum": "uyari", "deger": len(zayif),
                    "ayrinti": "; ".join(f"{r_['etiket']}: b = {etiket.get(r_['b'], r_['b'])} {_f(r_['zayif_taban']['b_deger'])}"
                                         for r_ in zayif)
                               + f" · en güçlü bildirilen taban {etiket[en_iyi_b]} = {_f(deg(en_iyi_b))}; kazanç zayıf "
                                 "tabana karşı ölçülüyor, en güçlü tek tabana karşı kaydedin (b is weaker than the "
                                 "strongest reported baseline)"})
    ezber_d = next((x for x in den if str(x.get("kontrol", "")).startswith("ezber testi")), None)
    ezber_uyari = None
    nedenler = []
    if ezber_d and ezber_d.get("durum") == "uyari":
        if bizim_llm:
            nedenler.append("ezber_denetimi")
        if bizim_llm and ezber_llm is None and not bizim_kendi_llm and llm_kollari:
            # varsayımla eklendi: LLM kolu başka (ham / çift); bizim LLM'siz bir modelse (lojistik, GBDT) tek adım
            kl = ", ".join(llm_kollari)
            den.append({"kontrol": "ezber uyarısı ve bizim kol (memorization warning and ours)", "durum": "tamam",
                        "deger": None, "ayrinti": f"ezber uyarısı LLM kolunun ({kl}); bizim kol varsayımla LLM tabanlı "
                        f"sayıldı, rozete ek düştü; bizim LLM kullanmıyorsa ours_uses_llm=False verin (the warning is "
                        f"for the LLM arm {kl}; ours is counted as LLM-based by default and the badge carries the "
                        "caveat; pass ours_uses_llm=False if ours does not use an LLM)"})
        elif not bizim_llm and ezber_llm is None and llm_kollari:   # uyarı LLM kolunun (çift); rozete ek düşmez
            kl = ", ".join(llm_kollari)
            den.append({"kontrol": "ezber uyarısı ve bizim kol (memorization warning and ours)", "durum": "tamam",
                        "deger": None, "ayrinti": f"ezber uyarısı LLM kolunun ({kl}); bizim kol LLM kolu sayılmadı, rozete "
                        f"ek düşmedi; bizim de LLM kullanıyorsa ours_uses_llm=True verin (the warning is for the LLM "
                        f"arm {kl}; ours is not counted as an LLM arm, no badge caveat; pass ours_uses_llm=True if "
                        "ours uses an LLM)"})
    if kor_ozet and kor_ozet["kayip_cogu"]:
        nedenler.append("kor_kol")
    if nedenler:
        ezber_uyari = {"nedenler": nedenler, "ezber_ayrinti": (ezber_d or {}).get("ayrinti") if "ezber_denetimi" in nedenler else None,
                       "kor_korunan_pay": (kor_ozet or {}).get("korunan_pay")}

    # ---- etkin örneklem çarpanı
    orneklem = None
    carpan_na = None
    if carpan and veri.gorev != "ab" and not tahmin_gorevi and toplu:
        # PPI++ "n gerçek kişi + simülasyon" kişi yanıtı ister; segment / çalışma / etki satırında anlamsız
        carpan_na = (f"uygulanamaz: {toplu[0]}; PPI++ 'kaç insana bedel' atlandı (not applicable: {toplu[1]}; "
                     "PPI++ 'worth N humans' skipped)")
        den.append({"kontrol": "etkin örneklem çarpanı (PPI++ kaç insana bedel)", "durum": "atlandi", "deger": None,
                    "uygulanamaz": True, "ayrinti": carpan_na})
    elif carpan and veri.gorev != "ab" and not tahmin_gorevi:
        try:
            from . import orneklem as _orn
            sec = carpan_yontemler or [bizim] + ([ham] if ham else [])
            # bizim/ham kat dışı (OOF) taban diye bildirildiyse aynı etiketlerle eğitilmiş: circular= gibi çarpan dışı
            oof_ad = set(oof_tabanlar or []) - set(dongusel or [])
            dongusel = list(dongusel or []) + [y.removeprefix(ON_EK) for y in dict.fromkeys(sec)
                                               if y and y.removeprefix(ON_EK) in oof_ad]
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
        f"Bootstrap birimi '{veri.kume_adi}' ({veri.n_kume} küme"
        + (f" × {blok_bilgi['blok']['n_blok']} '{veri.blok}' bloğu" if veri.blok is not None else "")
        + f", B={B}; toplu metriklerde {orn.B_toplu}). "
        + (("Blok × küme iki yönlü örneklenir (aşağıda)." if veri.blok is not None else
            "Seriler (madde) sabit kabul edilir: GA aynı serilerde yeni kümelere (seri/grup) genellemeyi ölçer; tek "
            "tahmin başlangıcı (origin) zamanda genellemeyi ölçmez, birden çok başlangıç için backtest kullanın.")
           if tahmin_gorevi else
         "Maddeler sabit kabul edilir: GA yeni kişilere genellemeyi ölçer, yeni sorulara değil."),
        "Baseline'lar ve kalibrasyon çapraz uydurmalıdır (aynı grubun satırı kendi eşlemesini görmez).",
    ]
    for k_ in ("sinir", "sinir_ek"):
        if blok_bilgi.get(k_):
            sinir.append(blok_bilgi[k_])
    if tahmin_gorevi and baseline:
        sinir.append("Tahmin görevi: otomatik tabanlar (rastgele / madde medyanı-ortalaması) test penceresinin "
                     "gerçeğinden çapraz uydurulur, yani tahmin başlangıcından ileriye bakar; meşru bir merdiven basamağı "
                     "değildir. Başlangıçta bilinen tabanları (persistence, son-k ortalama: methods.forecast) pred_ "
                     "sütunu olarak verin ya da baselines=False kullanın.")
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

    kart = {"n_satir": len(d), "n_kume": veri.n_kume, "kume_adi": veri.kume_adi,
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
        "metrikler": [{"ad": m.ad, "yon": 0 if m.ad in ISARETLI_METRIKLER else m.yon, "aciklama": m.aciklama,
                       "tur": m.tur} for m in metrikler],
        "sonuclar": sonuclar, "merdiven": merd, "manset": manset, "orneklem": orneklem,
        "karsilastirmalar": karsi, "kalibrasyon": kal, "segment": segment, "yayilim": yay,
        "denetim": den, "ornekler": orn_ler, "onkayit": onkayit or [], "sinirlar": sinir,
        "tabanlar": taban_ozet, "tavan_turu": tavan_turu,
    }
    if tahmin_gorevi:
        ozet["tahmin_gorevi"] = True
    if blok_bilgi.get("blok"):
        ozet["blok_bootstrap"] = blok_bilgi["blok"]
    if blok_bilgi.get("tek_koken"):
        ozet["tek_koken"] = blok_bilgi["tek_koken"]
    if ga_yontemi != "percentile":
        ozet["ga_yontemi"] = ga_yontemi
    if ga_disi:
        ozet["ga_uyarisi"] = {"metrik": birincil, "n": len(ga_disi), "yontem": ga_yontemi}
    if gurultu is not None:
        ozet["gurultu_tavani"] = gurultu
    if kayitli:
        ozet["kayitli_karsilastirmalar"] = kayitli
    if bilgi_b:
        ozet["bilgi_duzeyi"] = {"B": bilgi_b, "B_ustte": b_ustte, "rozet_tabani": en_iyi_b, "etiket": BILGI_B_ETIKET}
    if dolgu:
        ozet["dolgu"] = {"esik": DOLGU_ESIK, "kollar": dolgu}
    if kor_ozet is not None:
        ozet["kor_kol"] = kor_ozet
    if ezber_uyari is not None:
        ozet["ezber_uyarisi"] = ezber_uyari
    if sira_tanimsiz:
        ozet["sira_tanimsiz"] = sira_tanimsiz
    if toplu:
        ozet["toplu_duzey"] = {"neden": toplu[0], "reason": toplu[1]}
    if carpan_na:
        ozet["carpan_uygulanamaz"] = carpan_na
    bm_kayit, bm = _bizim_muhuru(bizim, onkayit)
    if bm:
        ozet["bizim_muhur"] = bm
    ozet = _temiz(ozet)
    ozet["kanit_kart"] = _temiz(_kart.kart(ozet, muhur=bm_kayit))  # sahne kart.py ile aynı şema
    if carpan_na:   # kart "ölçülmedi" der; satır kişi değilken doğrusu "uygulanamaz" + neden
        for r in ozet["kanit_kart"].get("satirlar") or []:
            if r.get("tur") == "bedel":
                r.update(deger=None, alt=None, ust=None, birim="", **{"not": carpan_na})
    return ozet


def _ad_kalibi(ad: str):
    """Yöntem adı için tam kelime kalıbı; 1-2 harfli adlar ('a', 'b') düz metinde sözcükle karışır: ön ek şart."""
    import re as _re
    on = r"(?:pred_|tahmin_)" if len(ad) <= 2 else r"(?:pred_|tahmin_)?"
    return _re.compile(r"(?<![A-Za-z0-9_])" + on + _re.escape(ad) + r"(?![A-Za-z0-9_])")


def _muhur_zamani(r: dict) -> float | None:
    from datetime import datetime as _dt
    z = r.get("zaman_utc") or r.get("time_utc") or r.get("zaman") or r.get("time")
    try:
        return _dt.fromisoformat(str(z)).timestamp() if z else None
    except ValueError:
        return None


def _onkayit_kaniti(bizim: str, onkayit: list | None, metinler: list | None, adlar=()) -> tuple[str, str]:
    """'bizim' adının önceden seçildiğine kanıt. Yalnız mühür AÇIKLAMALARI (aciklama/description) ve ön kayıt metin
    dosyaları aranır; mühürlenen sütun adları (sutunlar) ve kaynak yolu aranmaz: bütün aday sütunları içeren bir
    tablo mühürlemek hiçbir adı önceden seçmez. Zamanlı metin (dict {"metin", "zaman"}) en son mühürden SONRA
    yazıldıysa sayılmaz. Eşleşen metinde başka yöntem adları da geçiyor ve 'bizim' açıkça işaretlenmemişse
    (ours:/primary:/bizim:/birincil: <ad>) belirsizdir. Açık işaret 1-2 harfli adı da ön eksiz bulur ("ours: H"):
    düz metinde tek harf sözcükle karışır, işaretin ardında karışmaz (ad büyük/küçük harf duyarlı).
    Ad yalnız mühür açıklamasında geçiyorsa (ön kayıt metni / belgesi yok ya da adı geçirmiyor) kod "aciklama": mühür
    sonuç görüldükten sonra yazılabilir, ön kayıt sayılmaz (D7: ad dev koşusundan sonra yazılan açıklamadaydı).
    Dönüş: (kod, açıklama), kod ∈ {"adlandirildi", "aciklama", "bulunamadi", "sonra", "belirsiz"}."""
    import re as _re
    gecerli, gec = _onkayit_metinleri(onkayit, metinler)
    belge = gecerli[len(_onkayit_metinleri(onkayit, None)[0]):]   # açıklamalar başta, ardından zamanı uygun metinler
    kalip = _ad_kalibi(bizim)
    isaret_k = _re.compile(r"(?i:ours|primary|bizim|birincil)\W{0,3}[:=]\s*[`'\"]?(?:pred_|tahmin_)?"
                           + _re.escape(bizim) + r"(?![A-Za-z0-9_])")
    eslesen = [g for g in gecerli if kalip.search(g) or isaret_k.search(g)]
    if not eslesen:
        if any(kalip.search(g) or isaret_k.search(g) for g in gec):
            return "sonra", "ön kayıt metni son mühürden sonra yazılmış (pre-registration text is newer than the seal)"
        return "bulunamadi", ""
    isaret = _re.compile(r"(?i)(?:ours|primary|bizim|birincil)\W{0,3}[:=]\s*[`'\"]?(?:pred_|tahmin_)?"
                         + _re.escape(bizim) + r"(?![A-Za-z0-9_])")
    kod, neden = "adlandirildi", ""
    digerleri = sorted({a for a in adlar if a != bizim and any(_ad_kalibi(a).search(g) for g in eslesen)})
    if digerleri and not any(isaret.search(g) for g in eslesen):
        kod, neden = "belirsiz", ("ön kayıtta başka yöntem adları da geçiyor (" + ", ".join(digerleri) + "); 'bizim' "
                                  "açıkça işaretlenmemiş (ours: <ad>) (several method names in the pre-registration)")
    if not any(kalip.search(g) or isaret.search(g) for g in belge):
        return "aciklama", (neden + "; " if neden else "") + (
            "yalnız mühür açıklamasında geçiyor, ön kayıt belgesinde değil: önceden seçildiği kanıtlanmaz "
            "(named only in a seal description, not in a pre-registration document (seal ≠ pre-registration))")
    return kod, neden


def _onkayitta_geciyor(bizim: str, onkayit: list | None, metinler: list | None) -> bool:
    """Geri uyum: 'bizim' adı ön kayıt kanıtında (açıklama / zamanı uygun metin) geçiyor mu."""
    return _onkayit_kaniti(bizim, onkayit, metinler)[0] in ("adlandirildi", "belirsiz", "aciklama")


def _secim_yanliligi(veri, yontem_c, bc, bizim, deg, M, onkayit, metinler, n_yapilandirma) -> dict:
    """Seçim yanlılığı satırı: 'önceden adlandırıldı' yalnız ön kayıt belgesinde (mühürden önceki metin) kanıt varsa
    yazılır; yalnız mühür açıklamasındaki ad uyarıdır (kod "aciklama")."""
    sira = sorted(yontem_c, key=lambda c: -M.yon * deg(c))
    n_den = max(len(sira), int(n_yapilandirma or 0))
    en_iyi = bool(sira) and sira[0] == bc and n_den >= 4
    n_sutun = len(veri.yontemler)
    yap = (f"; {int(n_yapilandirma)} yapılandırma denendi (frame has {n_sutun} prediction columns; "
           f"n_configs={int(n_yapilandirma)} given)" if n_yapilandirma else "")
    iyimser = "; bizim en iyisi: çok yapılandırma arasından seçildiyse iyimser, bağımsız veride doğrulayın" if en_iyi else ""
    kayit_var = bool(onkayit) or bool(metinler)
    adlar = [str(a)[len(ON_EK):] if str(a).startswith(ON_EK) else str(a) for a in veri.yontemler]
    kod, neden = _onkayit_kaniti(bizim, onkayit, metinler, adlar) if kayit_var else ("yok", "")
    if kod == "adlandirildi":
        durum, ad = ("uyari" if en_iyi else "tamam"), f"'bizim' ({bizim}) ön kayıtta geçiyor: önceden adlandırıldı"
    elif kod in ("sonra", "belirsiz", "aciklama"):
        durum, ad = "uyari", f"doğrulanmadı: {neden} (not verified)"
    elif kayit_var:
        durum, ad = "uyari", (f"doğrulanmadı: 'bizim' ({bizim}) ön kayıtta bulunamadı "
                              "(not verified: ours not found in the pre-registration; sealed column names do not count)")
        if len(str(bizim)) <= 2:
            ad += (f"; 1-2 harfli ad düz metinde sayılmaz: ön kayda 'ours: {bizim}' ya da 'pred_{bizim}' yazın "
                   f"(short names need 'ours: {bizim}' or 'pred_{bizim}')")
    else:
        durum = "uyari" if en_iyi else "atlandi"
        ad = (f"doğrulanmadı: ön kayıt verilmedi (prereg=); 'bizim' ({bizim}) adının önceden seçildiği kanıtlanamaz "
              "(not verified: no pre-registration)")
    return {"kontrol": "seçim yanlılığı", "durum": durum, "ayrinti": f"{len(veri.yontemler)} yöntem; {ad}{yap}{iyimser}",
            "deger": n_sutun, "n_configs": int(n_yapilandirma) if n_yapilandirma else None, "n_pred_columns": n_sutun}


def _bizim_muhuru(bizim: str, muhurler: list | None, cerceve: list | None = None) -> tuple[dict | None, dict | None]:
    """Zincirde 'bizim' kolun mührü (başlıktaki mühür zincirin sonu değil, bizimki olmalı). Öncelik: açıklamada açık
    işaret (ours:/primary:/bizim:/birincil: <ad>) > çerçeveyle eşleşen ve bizim sütunu içeren mühür > açıklama ya da
    kaynak dosya adında ad (mühürde yalnız başka yöntemlerin pred_ sütunları varsa sayılmaz: "lgbm, ours ile kıyas"
    lgbm mührüdür) >
    sütunlarında bizim tahmin sütunu olan mühür; her sınıfta en son kayıt. Hiçbiri yoksa son kayıt, notla. cerceve:
    çerçeveyle sha'sı eşleşen kayıtlar (bekleyen). Döner (kayıt, özet) | (None, None)."""
    import re as _re
    kayitlar = [r for r in muhurler or [] if isinstance(r, dict)]
    if not kayitlar:
        return None, None
    isaret = _re.compile(r"(?i:ours|primary|bizim|birincil)\W{0,3}[:=]\s*[`'\"]?(?:pred_|tahmin_)?"
                         + _re.escape(bizim) + r"(?![A-Za-z0-9_])")
    kalip = _ad_kalibi(bizim)
    sutun_adlari = {f"pred_{bizim}", ON_EK + bizim}

    def metin(r):
        return " ".join(str(r.get(k) or "") for k in ("aciklama", "description", "kaynak"))

    def sutunlu(r):
        return bool(sutun_adlari & {str(s) for s in r.get("sutunlar") or []})

    def baska_tahmin(r):   # yalnız başka yöntemlerin tahmin sütunları mühürlü: addaki 'ours' içeriği tutmuyor
        t = [str(s) for s in r.get("sutunlar") or [] if str(s).startswith(("pred_", ON_EK))]
        return bool(t) and not sutunlu(r)

    cerceve_sha = {r.get("sha256") for r in cerceve or []}
    siniflar = (("aciklama_isaret", [r for r in kayitlar if isaret.search(metin(r))]),
                ("cerceve", [r for r in kayitlar if r.get("sha256") in cerceve_sha and sutunlu(r)]),
                ("aciklama_ad", [r for r in kayitlar if kalip.search(metin(r)) and not baska_tahmin(r)]),
                ("sutun", [r for r in kayitlar if sutunlu(r)]))
    eslesme, kayit = next(((ad, rs[-1]) for ad, rs in siniflar if rs), ("son", kayitlar[-1]))
    oz = {"sha256": kayit.get("sha256"), "sira": kayit.get("sira", kayitlar.index(kayit)), "eslesme": eslesme,
          "n_kayit": len(kayitlar)}
    if eslesme == "son":
        oz["not"] = (f"bizim ({bizim}) hiçbir mühürle eşleşmedi: zincirin son kaydı gösteriliyor (ours not matched: "
                     f"showing the last record; name ours in the seal description, e.g. 'ours: {bizim}')")
    return kayit, oz


def _cerceve_sutunu(s: str, sutunlar) -> str | None:
    """Mühürlenen sütun adı → değerlendirilen (iç adlı) çerçevedeki karşılığı (pred_x → tahmin_x, item → madde...)."""
    if s in sutunlar:
        return s
    if s.startswith("pred_") and ON_EK + s[len("pred_"):] in sutunlar:
        return ON_EK + s[len("pred_"):]
    from .api import COLUMN_TO_INTERNAL
    c = COLUMN_TO_INTERNAL.get(s)
    return c if c in sutunlar else None


def _cerceve_muhurleri(df: pd.DataFrame, muhurler: list) -> tuple[list, list]:
    """(eşleşen, karşılaştırılabilen) mühür kayıtları: mühürlenen sütunların hepsi çerçevede varsa çerçevenin o
    sütunlarının kanonik sha256'sı (onkayit.kanonik_bayt, sütun adları mühürdeki gibi) kayıttakiyle karşılaştırılır."""
    from .onkayit import _sha, kanonik_bayt
    eslesen, karsilastirilan = [], []
    for r in muhurler:
        if not isinstance(r, dict) or not r.get("sutunlar") or not r.get("sha256"):
            continue
        es = {s: _cerceve_sutunu(str(s), df.columns) for s in r["sutunlar"]}
        if any(v is None for v in es.values()) or len(set(es.values())) != len(es):
            continue
        karsilastirilan.append(r)
        alt = df[list(es.values())].copy()
        alt.columns = list(es)
        try:
            if _sha(kanonik_bayt(alt)) == r["sha256"]:
                eslesen.append(r)
        except (TypeError, ValueError):
            continue
    return eslesen, karsilastirilan


def _cerceve_denetimi(bizim: str, kayit: dict | None, eslesen: list, karsilastirilan: list) -> dict:
    """Bizim mühür ile değerlendirilen çerçeve aynı içerik mi (D11 #7)."""
    k = "çerçeve ↔ mühürlü dosya"
    if kayit is None:
        return {"kontrol": k, "durum": "atlandi", "deger": None, "ayrinti": "mühür yok (no seal)"}
    sha = str(kayit.get("sha256"))[:12]
    if kayit in eslesen:
        return {"kontrol": k, "durum": "tamam", "deger": 1,
                "ayrinti": f"çerçeve bizim ({bizim}) mührüyle aynı içerik: sha256 {sha} (frame matches ours' sealed "
                           "file: same canonical sha256)"}
    if kayit in karsilastirilan:
        return {"kontrol": k, "durum": "uyari", "deger": 0,
                "ayrinti": f"çerçevenin mühürlü sütunları bizim ({bizim}) mührüyle (sha256 {sha}) eşleşmiyor: tahmin "
                           "mühürden sonra değişmiş olabilir (frame does not match ours' sealed file; check with "
                           "verify())"}
    return {"kontrol": k, "durum": "uyari", "deger": None,
            "ayrinti": f"bizim ({bizim}) mührünün sütunları çerçevede yok (ör. geniş tahmin dosyası): çerçeve mühürle "
                       f"karşılaştırılamadı, mühürlü dosyayı verify() ile doğrulayın (sealed columns of sha256 {sha} are "
                       "not in the frame; could not compare)"}


MADDE_OZET_EN_FAZLA = 100   # bekleyen raporda madde başına tahmin özeti bu kadar maddeye kadar (fazlasında sabitler)


def _kategorik_mi(seri: pd.Series, gorev: str | None, tahmin_gorevi: bool) -> bool:
    """Tahmin değerleri sınıf etiketi mi: çok sınıflı görev ya da (görev verilmemişken) 3-20 farklı tam sayı kodu."""
    if gorev == "cok_sinif":
        return True
    if gorev is not None or tahmin_gorevi:
        return False
    x = pd.to_numeric(seri.dropna(), errors="coerce")
    if x.empty or x.isna().any():
        return False
    v = x.to_numpy(float)
    return bool(np.all(v == np.round(v)) and 3 <= len(np.unique(v)) <= 20)


def _tahmin_ozeti(seri: pd.Series, kategorik: bool = False) -> dict:
    """Bir tahmin sütununun gerçeksiz özeti: sayısal → dağılım istatistikleri, etiket (ya da kategorik=True ile tam
    sayı sınıf kodu) → en sık sınıflar ve farklı değer sayısı (ortalama/sd sınıf kodunda anlamsız)."""
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
    if x.notna().all() and not kategorik:
        q = np.percentile(x.to_numpy(float), [0, 25, 50, 75, 100])
        out.update({"tur": "sayi", "ortalama": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
                    "en_kucuk": float(q[0]), "q25": float(q[1]), "medyan": float(q[2]), "q75": float(q[3]),
                    "en_buyuk": float(q[4])})
    else:
        s = dolu.map(lambda a: str(int(a)) if isinstance(a, (float, np.floating)) and float(a).is_integer() else str(a))
        pay = s.value_counts(normalize=True)
        out.update({"tur": "etiket", "en_sik": {str(k): float(p) for k, p in pay.head(5).items()},
                    "n_farkli": int(len(pay))})
    return out


def _sabit_mi(seri: pd.Series) -> bool:
    """En az 2 dolu değer ve hepsi aynı (vektör hücreler dahil)."""
    v = [_anahtar(a) for a in seri.dropna().to_numpy(object)]
    return len(v) >= 2 and len(set(v)) == 1


def bekleyen(df: pd.DataFrame, baslik: str, veri_adi: str = "", gorev: str | None = None,
             yontemler: list[str] | None = None, bizim: str | None = None, ham: str | None = None,
             etiketler: dict | None = None, birincil: str | None = None, onkayit: list | None = None,
             sinirlar: list | None = None, duzey: str | None = None, girdi_kaynagi: str | None = None,
             tohum: int = 0, B: int = 1000, tahmin_gorevi: bool = False,
             etiketsiz_tabanlar: list[str] | None = None, oof_tabanlar: list[str] | None = None) -> dict:
    """Gerçek yok (sütun yok ya da tümü eksik): kalibrasyon, merdiven ve bootstrap atlanır. Rapor mühür kayıtlarını ve
    tahmin özetini taşır; kart rozeti 'bekliyor' (awaiting truth). Şema degerlendir() ile aynı anahtarlar.

    etiketsiz_tabanlar / oof_tabanlar: bildirilen tabanlar; ozet["tabanlar"] ve ozet["bildirilen_tabanlar"] (etiketler)
    olarak saklanır. Başlıktaki mühür bizim kolunki (_bizim_muhuru; ozet["bizim_muhur"]) ve çerçeve o mühürle
    karşılaştırılır. Tahmin özeti 'madde' varsa madde başına da verilir ("maddeler", "n_sabit_madde"); sabit tahmin
    (sd 0) denetimde uyarılır; çok sınıflı görevde ya da tam sayı sınıf kodlarında ortalama/sd yerine en sık sınıflar."""
    etiketler = dict(etiketler or {})
    tsut = [ON_EK + y for y in yontemler] if yontemler else [c for c in df.columns if isinstance(c, str) and c.startswith(ON_EK)]
    eksik = [c for c in tsut if c not in df.columns]
    if eksik:
        raise ValueError(f"missing prediction columns: {['pred_' + c[len(ON_EK):] for c in eksik]} (alias: {eksik})")
    if not tsut:
        raise ValueError("at least one 'pred_<method>' column is required (alias: tahmin_<method>)")
    d = df
    if "bolum" in d.columns:
        from .sema import TEST_ADLARI
        d = d[d["bolum"].astype(str).str.lower().isin(TEST_ADLARI).to_numpy()]
    adlar = [c[len(ON_EK):] for c in tsut]
    bizim_oto = bizim is None
    bizim = bizim or adlar[-1]
    madde_var = "madde" in d.columns
    ozetler = {}
    sabitler = []
    for c in tsut:
        kat = _kategorik_mi(d[c], gorev, tahmin_gorevi)
        oz = {"etiket": etiketler.get(c[len(ON_EK):], c[len(ON_EK):]), **_tahmin_ozeti(d[c], kat)}
        if madde_var and oz.get("tur") != "vektor":
            gruplar = list(d.groupby("madde", sort=True)[c])
            sabit_m = [str(m) for m, s in gruplar if _sabit_mi(s)]
            oz["n_madde"], oz["n_sabit_madde"] = len(gruplar), len(sabit_m)
            if sabit_m:
                oz["sabit_maddeler"] = sabit_m[:20]
            oz["maddeler"] = {str(m): _tahmin_ozeti(s, kat) for m, s in gruplar
                              if len(gruplar) <= MADDE_OZET_EN_FAZLA or str(m) in sabit_m}
            if len(gruplar) > MADDE_OZET_EN_FAZLA:
                oz["maddeler_kirpildi"] = True
            if sabit_m:
                sabitler.append(f"{oz['etiket']}: {len(sabit_m)}/{len(gruplar)} maddede sabit "
                                f"({', '.join(sabit_m[:5])}{', …' if len(sabit_m) > 5 else ''})")
        elif oz.get("tur") != "vektor" and _sabit_mi(d[c]):
            oz["sabit"] = True
            sabitler.append(f"{oz['etiket']}: bütün sütun sabit")
        ozetler[c] = oz
    kume_adi = next((s for s in ("grup", "madde") if s in d.columns), "satir")
    n_kume = int(d[kume_adi].nunique()) if kume_adi != "satir" else len(d)
    muhurler = list(onkayit or [])
    cerceve_es, cerceve_kar = _cerceve_muhurleri(df, muhurler) if muhurler else ([], [])
    bm_kayit, bm = _bizim_muhuru(bizim, muhurler, cerceve_es)
    if bm:
        muhur_ayr = (f"{len(muhurler)} mühür kaydı eklendi; bizim ({bizim}) mührü #{bm['sira']} sha256 "
                     f"{str(bm['sha256'])[:12]} (ours' seal; {bm['eslesme']})" + (f" · {bm['not']}" if bm.get("not") else ""))
    else:
        muhur_ayr = ("mühür kaydı yok: tahmini gerçek açılmadan önce seal() ile mühürleyin ve evaluate(prereg=<dizin>) "
                     "verin")
    den = [{"kontrol": "gerçek", "durum": "atlandi", "deger": None,
            "ayrinti": ("'gercek' (truth) sütunu yok" if "gercek" not in df.columns else "'gercek' (truth) sütununun tamamı eksik")
                       + ": kalibrasyon, taban merdiveni, bootstrap ve çarpan atlandı"},
           {"kontrol": "mühür", "durum": "tamam" if muhurler and not (bm or {}).get("not") else "uyari",
            "deger": len(muhurler), "ayrinti": muhur_ayr}]
    if muhurler:
        den.append(_cerceve_denetimi(bizim, bm_kayit, cerceve_es, cerceve_kar))
    if sabitler:
        den.append({"kontrol": "sabit tahmin (sd 0)", "durum": "uyari", "deger": len(sabitler),
                    "ayrinti": "; ".join(sabitler) + " · gerçek gelince bu maddelerde sıra/korelasyon metrikleri "
                               "(Spearman, madde_ici_kor) tanımsız olur (constant prediction: rank metrics will be "
                               "undefined in these items)"})
    if bizim_oto:
        den.append({"kontrol": "seçim yanlılığı", "durum": "uyari", "deger": None,
                    "ayrinti": "'bizim' yöntem açıkça belirtilmedi (son sütun alındı); yöntemi önceden belirleyin"})
    etsiz = [a for a in etiketsiz_tabanlar or [] if a in adlar]
    oof = [a for a in oof_tabanlar or [] if a in adlar and a not in etsiz]
    bilinmeyen = [a for a in list(etiketsiz_tabanlar or []) + list(oof_tabanlar or []) if a not in adlar]
    if bilinmeyen:
        warnings.warn(f"declared baseline column(s) not in the table, ignored: {bilinmeyen} (need pred_<name>)",
                      UserWarning, stacklevel=3)
    taban_ozet = {"bildirilen": [ON_EK + a for a in etsiz + oof], "etiketsiz": [ON_EK + a for a in etsiz],
                  "oof": [ON_EK + a for a in oof], "otomatik": [], "en_guclu": None, "yalniz_basit": False}
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
            "n_madde": int(d["madde"].nunique()) if madde_var else None, "B": B, "tohum": tohum,
            "segmentler": [], "notlar": [], "yontemler": adlar, "bizim": bizim, "ham": ham}
    ozet = {
        "baslik": baslik, "veri_adi": veri_adi, "gorev": gorev, "duzey": duzey or "mikro",
        "tutma_birimi": kume_adi, "girdi_kaynagi": girdi_kaynagi,
        "olusturma": _dt.datetime.now().astimezone().isoformat(timespec="seconds"), "surum": SURUM,
        "kart": kart, "birincil": birincil, "metrikler": [], "sonuclar": {}, "merdiven": [], "manset": manset,
        "orneklem": None, "karsilastirmalar": [], "kalibrasyon": {}, "segment": {}, "yayilim": {}, "denetim": den,
        "ornekler": [], "onkayit": muhurler, "sinirlar": sinir, "tabanlar": taban_ozet, "tavan_turu": None,
        "gercek_bekleniyor": True, "tahmin_ozeti": ozetler,
        "bildirilen_tabanlar": [etiketler.get(a, a) for a in etsiz + oof],
    }
    if bm:
        ozet["bizim_muhur"] = bm
    if tahmin_gorevi:
        ozet["tahmin_gorevi"] = True
        ozet["gorev"] = ozet["gorev"] or "surekli"
        ozet["birincil"] = ozet["birincil"] or "wmape"
    ozet = _temiz(ozet)
    ozet["kanit_kart"] = _temiz(_kart.kart(ozet, muhur=bm_kayit))
    return ozet
