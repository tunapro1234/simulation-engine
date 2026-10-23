"""Otomatik baseline'lar (çapraz uydurmalı) ve kontrol sütunları.

rastgele    bilgi yok (ikili 0,5 · tekdüze dağılım · madde içi rastgele gerçek değer · ab'de hep beraberlik);
            sürekli görevde tek satırlı madde değerini başka maddelerin satırlarından çeker (kendi gerçeği değil)
marjinal    madde başına ortalama/medyan/sınıf payı (eğitim katlarından); "herkes aynı" tahmini
demografi   madde × segment ortalaması, maddeye doğru büzülmüş (k=10); segment varsa
karistir:Y  Y'nin tahminleri madde içinde birimler arasında karıştırılır: kişiye özgü bilgi silinir,
            marjinal korunur. Y bundan iyi değilse persona/birey bilgisi kullanılmıyor demektir.
Çapraz uydurma: bolum=egitim satırları varsa onlardan; yoksa grup (yoksa satır) üzerinden K katlı; bölüm yoksa etiketli
basamakların adı bunu söyler ("cross-fitted, 5-fold by group"), örneklem içi okunmasın.

Dürüstlük kuralları:
  - sabit gerçek (puanlanan satırlarda sd 0): gerçekten uydurulan otomatik basamaklar sabiti kopyalar (MAE 0, en güçlü
    taban, yanlış "tabanın altında" rozeti); düşürülür, denetimde uyarı satırı (veri.notlar) kalır. Gerçeği okumayan
    basamaklar (ikili 0,5, çok sınıflı tekdüze) kalır.
  - ikili, madde yok (ya da her madde tek satır): kat dışı genel oran kat başına sabittir ve dışarıda bırakılan katın
    pozitifleriyle ters ilişkilidir (AUC ~0,38); tek sabit verilir (kat tahminlerinin ortalaması, AUC 0,5).
  - segment madde içinde satırı tanımlıyorsa (segment = birim) demografi basamağı marjinale eşittir: atlanır.
  - çok sınıflı, sert etiketli tahmin ve ETIKET_ESIK'ten çok sınıf: rastgele/marjinal/demografi n × K olasılık
    vektörü değil etikettir (tekdüze rastgele etiket; eğitim katlarının madde/hücre başına en sık sınıfı).

    out = baselineler(veri, "median")   # {ad: (etiket, değerler)}, Veri.df satırlarıyla hizalı
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

K_BUZ = 10.0
K_KAT = 5            # çapraz uydurma kat sayısı (bölüm sütunu yoksa)
ETIKET_ESIK = 200    # çok sınıflı, sert etiketli tahmin: bundan çok sınıfta basamaklar etiket (n × K vektör yok)


def katlar(df: pd.DataFrame, K: int = K_KAT, tohum: int = 0) -> np.ndarray:
    if "grup" in df:
        # grup adından deterministik kat (sıra ve tohumdan bağımsız, tekrarlanabilir)
        return np.array([zlib.crc32(f"{tohum}:{g}".encode()) % K for g in df["grup"].astype(str)])
    return np.random.default_rng(tohum).permutation(len(df)) % K


def _anahtar(df, sutunlar):
    if not sutunlar:
        return np.zeros(len(df), dtype=np.int64), 1
    kod, cats = pd.factorize(pd.MultiIndex.from_frame(df[sutunlar].astype(str)) if len(sutunlar) > 1
                             else df[sutunlar[0]].astype(str))
    return kod, len(cats)


def _grup_istatistik(y_egitim, k_egitim, k_hedef, n_kat, ozet):
    """k_egitim kodlarına göre y istatistiği, k_hedef kodları için döner (NaN = görülmemiş)."""
    out = np.full(len(k_hedef), np.nan)
    s = pd.Series(y_egitim).groupby(k_egitim).agg(ozet)
    m = pd.Series(k_hedef).map(s)
    out[:] = m.to_numpy(float)
    return out


def _bolum_var(veri) -> bool:
    return veri.egitim is not None and len(veri.egitim) > 0


def _capraz_ek(veri) -> str:
    """Bölüm sütunu yoksa etiketli basamak adına eklenir: kat dışı uydurulduğu görünsün (örneklem içi sanılmasın)."""
    if _bolum_var(veri):
        return ""
    return f" (cross-fitted, {K_KAT}-fold by {'group' if 'grup' in veri.df else 'row'})"


def _sabit_gercek(veri) -> bool:
    """Puanlanan satırlarda gerçek sabit mi (sd 0)."""
    g = veri.df["gercek"]
    if veri.gorev == "cok_sinif":
        return len(g) > 0 and g.astype(str).nunique() <= 1
    y = pd.to_numeric(g, errors="coerce").to_numpy(float)
    y = y[np.isfinite(y)]
    return len(y) > 0 and float(np.ptp(y)) == 0.0


def _segment_birim(veri) -> bool:
    """Madde × segment hücresi uydurmada hiç görülmüyorsa (segment madde içinde satırı tanımlar) True: demografi
    basamağı marjinale eşit olur. Bölüm varsa test hücreleri eğitimde aranır, yoksa tabloda tekrar eden hücre aranır."""
    d = veri.df
    anahtar = (["madde"] if "madde" in d else []) + veri.segmentler[:1]
    hucre = pd.MultiIndex.from_frame(d[anahtar].astype(str))
    if _bolum_var(veri):
        if not all(c in veri.egitim for c in anahtar):
            return True
        return not hucre.isin(pd.MultiIndex.from_frame(veri.egitim[anahtar].astype(str))).any()
    return not hucre.duplicated(keep=False).any()


def _sert_etiket(veri) -> bool:
    """Çok sınıflı tahminler etiket mi (olasılık vektörü değil); ilk yöntemin ilk değerine bakar (cekirdek ile aynı)."""
    ys = getattr(veri, "yontemler", None) or []
    c = f"tahmin_{ys[0]}" if ys else None
    return c is not None and c in veri.df and len(veri.df) > 0 and not isinstance(veri.df[c].iloc[0], np.ndarray)


def _not(veri, metin: str) -> None:
    notlar = getattr(veri, "notlar", None)
    if isinstance(notlar, list) and metin not in notlar:
        notlar.append(metin)


def _uydur_uygula(veri, hedef_fn):
    """hedef_fn(egitim_df, test_df) → test için tahmin dizisi. Çapraz uydurma ya da bolum."""
    d = veri.df
    if veri.egitim is not None and len(veri.egitim):
        return hedef_fn(veri.egitim, d)
    k = katlar(d)
    out = [None] * len(d)
    for f in np.unique(k):
        tr, te = d[k != f], d[k == f]
        sonuc = hedef_fn(tr, te)
        for i, v in zip(np.flatnonzero(k == f), sonuc):
            out[i] = v
    if veri.gorev in ("ikili", "surekli"):
        return np.asarray(out, float)
    return out


def _skaler_baseline(veri, ozet: str, segmentli: bool):
    madde = ["madde"] if "madde" in veri.df else []
    seg = veri.segmentler[:1] if segmentli else []

    def fn(tr, te):
        y = tr["gercek"].to_numpy(float)
        glob = float(getattr(np, "median" if ozet == "median" else "mean")(y))
        if madde:
            birlesik = pd.concat([tr[madde], te[madde]])
            kod, _ = _anahtar(birlesik, madde)
            mt = _grup_istatistik(y, kod[:len(tr)], kod[len(tr):], 0, ozet)
            mt = np.where(np.isnan(mt), glob, mt)
        else:
            mt = np.full(len(te), glob)
        if not seg:
            return mt
        birlesik = pd.concat([tr[madde + seg], te[madde + seg]])
        kod, _ = _anahtar(birlesik, madde + seg)
        st = _grup_istatistik(y, kod[:len(tr)], kod[len(tr):], 0, ozet)
        n = _grup_istatistik(np.ones(len(tr)), kod[:len(tr)], kod[len(tr):], 0, "sum")
        n = np.nan_to_num(n)
        st = np.where(np.isnan(st), mt, st)
        return (n * st + K_BUZ * mt) / (n + K_BUZ)
    return _uydur_uygula(veri, fn)


def _sinif_baseline(veri, segmentli: bool):
    K = len(veri.siniflar)
    kod = {k: i for i, k in enumerate(veri.siniflar)}
    madde = ["madde"] if "madde" in veri.df else []
    seg = veri.segmentler[:1] if segmentli else []

    def frek(tr, te, anahtar):
        oh = np.zeros((len(tr), K))
        oh[np.arange(len(tr)), tr["gercek"].map(lambda v: kod.get(v, kod.get(str(v), 0))).to_numpy()] = 1
        if not anahtar:
            return np.tile(oh.mean(0), (len(te), 1)), np.full(len(te), len(tr))
        birlesik = pd.concat([tr[anahtar], te[anahtar]])
        k, n_k = _anahtar(birlesik, anahtar)
        ktr, kte = k[:len(tr)], k[len(tr):]
        top = np.zeros((n_k, K))
        np.add.at(top, ktr, oh)
        say = top.sum(1)
        return top[kte], say[kte]

    def fn(tr, te):
        g_top, _ = frek(tr, te, [])
        g = g_top[0] / max(g_top[0].sum(), 1e-12) if len(te) else np.full(K, 1 / K)
        m_top, m_n = frek(tr, te, madde)
        mt = (m_top + 1e-3 * g) / (m_n[:, None] + 1e-3) if madde else np.tile(g, (len(te), 1))
        if seg:
            s_top, s_n = frek(tr, te, madde + seg)
            mt = (s_top + K_BUZ * mt) / (s_n[:, None] + K_BUZ)
        mt = mt / mt.sum(1, keepdims=True)
        return [r for r in mt]
    return _uydur_uygula(veri, fn)


def _sinif_etiket_baseline(veri, segmentli: bool) -> np.ndarray:
    """Etiket basamağı (çok sınıf): eğitim katlarında madde başına en sık sınıf (eşitlikte küçük sınıf kodu; madde
    görülmemişse genel en sık). Segmentli: madde × segment hücresinde sayım + K_BUZ · madde payı en büyük sınıf (olasılık
    basamağındaki büzülmenin argmax'ı). Bellek ~satır + (hücre × maddenin sınıfları), n × K değil."""
    sinif = np.asarray(veri.siniflar, dtype=object)
    kod = {k: i for i, k in enumerate(veri.siniflar)}
    madde = ["madde"] if "madde" in veri.df else []
    seg = veri.segmentler[:1] if segmentli else []

    def en_sik(t, anahtar, puan):
        return t.sort_values([anahtar, puan, "y"], ascending=[True, False, True]).drop_duplicates(anahtar) \
            .set_index(anahtar)["y"]

    def fn(tr, te):
        y = tr["gercek"].map(lambda v: kod.get(v, kod.get(str(v), -1))).to_numpy(int)
        ok = y >= 0
        glob = int(np.bincount(y[ok], minlength=len(sinif)).argmax()) if ok.any() else 0
        out = np.full(len(te), glob)
        if not madde:
            return sinif[out]
        km, _ = _anahtar(pd.concat([tr[madde], te[madde]]), madde)
        mtr, mte = km[:len(tr)][ok], km[len(tr):]
        sm = pd.DataFrame({"k": mtr, "y": y[ok]}).value_counts().rename("n").reset_index()
        out = pd.Series(mte).map(en_sik(sm, "k", "n")).fillna(glob).to_numpy(int)
        if not seg:
            return sinif[out]
        ks, _ = _anahtar(pd.concat([tr[madde + seg], te[madde + seg]]), madde + seg)
        str_, ste = ks[:len(tr)][ok], ks[len(tr):]
        ss = pd.DataFrame({"s": str_, "y": y[ok]}).value_counts().rename("n_s").reset_index()
        aday = pd.DataFrame({"s": str_, "k": mtr}).drop_duplicates("s").merge(sm, on="k")   # hücre × maddenin sınıfları
        aday = aday.merge(ss, on=["s", "y"], how="left").fillna({"n_s": 0})
        aday["puan"] = aday["n_s"] + K_BUZ * aday["n"] / aday["k"].map(sm.groupby("k")["n"].sum())
        hucre = pd.Series(ste).map(en_sik(aday, "s", "puan"))
        return sinif[np.where(hucre.isna(), out, hucre.fillna(0)).astype(int)]
    return np.asarray(_uydur_uygula(veri, fn), dtype=object)


def baselineler(veri, ozet_surekli: str = "mean", rastgele_tohum: int = 0) -> dict:
    """{ad: (etiket, değerler)}. Değerler Veri.df satırlarıyla hizalı."""
    d, g = veri.df, veri.gorev
    out = {}
    tek = "madde" in d and not (d.groupby("madde").size() > 1).any()
    sabit = g in ("ikili", "cok_sinif", "surekli") and _sabit_gercek(veri)
    seg = bool(veri.segmentler) and not sabit
    if seg and _segment_birim(veri):
        seg = False
        _not(veri, f"Demografi tabanı atlandı: '{veri.segmentler[0]}' madde içinde satırı tanımlıyor (segment = birim), "
                   "taban marjinale eşit olurdu (demographic rung skipped: the segment identifies the row)")
    ek = _capraz_ek(veri)
    if g == "ikili":
        out["rastgele"] = ("Rastgele (0,5)", np.full(len(d), 0.5))
        if not sabit:
            marj = _skaler_baseline(veri, "mean", False)
            if "madde" not in d or tek:
                # kat başına sabit genel oran dışarıdaki katın pozitifleriyle ters ilişkili (G4 #13, AUC 0,38): tek sabit
                marj = np.full(len(d), float(np.mean(marj)) if len(d) else np.nan)
            out["marjinal"] = (("Madde ortalaması" if "madde" in d and not tek else "Genel oran") + ek, marj)
        if seg:
            out["demografi"] = (f"Demografi ortalaması ({veri.segmentler[0]})" + ek, _skaler_baseline(veri, "mean", True))
    elif g == "cok_sinif":
        K = len(veri.siniflar)
        if K > ETIKET_ESIK and _sert_etiket(veri):
            # çok sınıf + etiket tahmini: n × K vektör yerine etiket (bellek n·K, metrikler zaten etikete bakar)
            sinif = np.asarray(veri.siniflar, dtype=object)
            out["rastgele"] = ("Rastgele (tekdüze)", sinif[np.random.default_rng(rastgele_tohum).integers(0, K, len(d))])
            if not sabit:
                out["marjinal"] = (("Madde sınıf payları" if "madde" in d else "Genel sınıf payları") + " (mode)" + ek,
                                   _sinif_etiket_baseline(veri, False))
            if seg:
                out["demografi"] = (f"Demografi sınıf payları ({veri.segmentler[0]}) (mode)" + ek,
                                    _sinif_etiket_baseline(veri, True))
        else:
            out["rastgele"] = ("Rastgele (tekdüze)", [np.full(K, 1 / K) for _ in range(len(d))])
            if not sabit:
                out["marjinal"] = (("Madde sınıf payları" if "madde" in d else "Genel sınıf payları") + ek,
                                   _sinif_baseline(veri, False))
            if seg:
                out["demografi"] = (f"Demografi sınıf payları ({veri.segmentler[0]})" + ek, _sinif_baseline(veri, True))
    elif g == "surekli":
        if not sabit:
            out["rastgele"] = ("Rastgele (başka maddenin gerçek değeri; madde başına tek satır)" if tek
                               else "Rastgele (madde içi gerçek değer)", _rastgele_surekli(veri, rastgele_tohum))
            ad = "Madde medyanı" if ozet_surekli == "median" else "Madde ortalaması"
            if tek and not _bolum_var(veri):
                # tek satırlı madde dışarıda bırakılan katta görülmez: değer eğitim katlarının genel özetidir
                ad = ("Genel medyan" if ozet_surekli == "median" else "Genel ortalama") + " (madde başına tek satır)"
            out["marjinal"] = ((ad if "madde" in d else ad.replace("Madde", "Genel")) + ek,
                               _skaler_baseline(veri, ozet_surekli, False))
        if seg:
            out["demografi"] = (f"Demografi ({veri.segmentler[0]})" + ek, _skaler_baseline(veri, ozet_surekli, True))
    elif g == "dagilim":
        out["rastgele"] = ("Tekdüze dağılım", [np.full(len(v), 1 / len(v)) for v in d["gercek"]])
        if "madde" in d:
            out["marjinal"] = ("Madde geneli (birini dışarıda bırak)", _loo_dagilim(d))
    elif g == "ab":
        out["rastgele"] = ("Rastgele (hep beraberlik)", np.zeros(len(d)))
        out["marjinal"] = ("Çoğunluk yönü (kol ortalaması)", _ab_kol(d))
    if sabit:
        dus = [a for a in ("rastgele", "marjinal", "demografi") if a not in out]
        _not(veri, f"Sabit gerçek (sd 0, {len(d)} puanlanan satır): gerçekten uydurulan otomatik tabanlar ({', '.join(dus)}) "
                   f"düşürüldü; sabiti kopyalayıp en güçlü taban olurlardı (constant truth: automatic {'/'.join(dus)} "
                   "rungs dropped, not in the badge)")
    return out


def _ab_kol(d, K: int = 5):
    """Bilgisiz 'hep varyant' tabanı: kolun (varsa 'kol' sütunu, yoksa test içi sıra) test içi ortalamadan
    sapmasının diğer testlerdeki ortalaması. Katlar test (madde) düzeyinde, çapraz uydurmalı."""
    kol = (d["kol"] if "kol" in d else d.groupby("madde").cumcount()).astype(str).to_numpy()
    y = d["gercek"].to_numpy(float)
    sap = y - d.groupby("madde")["gercek"].transform("mean").to_numpy(float)
    k = np.array([zlib.crc32(f"ab:{m}".encode()) % K for m in d["madde"].astype(str)])
    out = np.zeros(len(d))
    for f in range(K):
        tr, te = k != f, k == f
        ort = pd.Series(sap[tr]).groupby(kol[tr]).mean()
        out[te] = pd.Series(kol[te]).map(ort).fillna(0).to_numpy(float)
    return out


def _rastgele_surekli(veri, tohum):
    """Madde içinde başka bir birimin gerçek değeri: marjinali korur, bilgiyi siler.

    Tek satırlı maddede madde içi permütasyon gerçeği kopyalar (MAE 0, en güçlü taban olur, oranlar sıfıra
    bölünür); o satırlar başka maddelerin satırlarından (bir-dışarıda havuz) rastgele bir gerçek değer alır."""
    d = veri.df
    y = d["gercek"].to_numpy(float)
    rng = np.random.default_rng(tohum)
    if "madde" not in d:
        return rng.permutation(y)
    out = np.empty(len(y))
    tekler = []
    for _, idx in d.groupby("madde").indices.items():
        if len(idx) > 1:
            out[idx] = rng.permutation(y[idx])
        else:
            tekler.append(idx[0])
    n = len(y)
    for i in tekler:
        if n < 2:  # havuz boş: başka satır yok
            out[i] = y[i]
            continue
        j = int(rng.integers(n - 1))
        out[i] = y[j + (j >= i)]  # i dışındaki satırlardan tekdüze (i tek satırlı, yani kendi maddesi yalnız i)
    return out


def _loo_dagilim(d):
    out = [None] * len(d)
    for _, idx in d.groupby("madde").indices.items():
        A = d["gercek"].iloc[idx].tolist()
        uz = {len(a) for a in A}
        if len(idx) < 2 or len(uz) > 1:
            for i, a in zip(idx, A):
                out[i] = np.full(len(a), 1 / len(a))
            continue
        M = np.vstack(A)
        top = M.sum(0)
        for j, i in enumerate(idx):
            out[i] = (top - M[j]) / (len(idx) - 1)
    return out


def karistir(veri, sutun: str, tohum: int = 1):
    """Tahminleri madde içinde birimler arasında karıştır (persona-karıştırma kontrolü)."""
    d = veri.df
    v = d[sutun].to_numpy(object if veri.gorev in ("dagilim", "cok_sinif") else float)
    rng = np.random.default_rng(tohum)
    out = v.copy()
    if "madde" in d and veri.gorev != "ab":
        for _, idx in d.groupby("madde").indices.items():
            out[idx] = v[rng.permutation(idx)]
    elif veri.gorev == "ab":
        for _, idx in d.groupby("madde").indices.items():
            out[idx] = v[rng.permutation(idx)]
    else:
        out = v[rng.permutation(len(v))]
    return out
