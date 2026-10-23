"""ETKİN ÖRNEKLEM ÇARPANI: simülasyon + n gerçek insan, kaç gerçek insana eşdeğer?

Soru: madde m'nin (soru, ürün, segment×soru) nüfus ortalaması θ_m. Elde n gerçek yanıt + simülasyonun
bütün birimler için tahmini f var. Kestiriciler:
  klasik      ȳ_n
  ppi         PPI++: ȳ_n + λ(f̄_N − f̄_n); λ = Cov(y,f)/Var(f) madde içi, maddeler arası havuzlanmış,
              madde yarılarında çapraz uydurmalı (tek maddede λ etiketli n birimden, her tekrarda)
  bayes       önsel büzülme (n0·f̄_N + n·ȳ_n)/(n0+n); n0 ızgaradan, madde yarılarında çapraz uydurmalı
  yalniz_sim  f̄_N (n = 0): kaç insan değerinde = σ²_m / MSE
Gerçek hedef θ̂_m = etiketlenmeyen KALAN birimlerin gerçek ortalaması. Gözlenen kare hatadan kalanın
örneklem gürültüsü σ²/(N−n) düşülür; f̄_N kalan birimleri de içerdiği için çapraz terim geri eklenir:
MSE = gözlenen − σ²/(N−n) + 2·w·σ_yf/N  (w = f̄_N'nin ağırlığı: klasik 0, ppi λ, bayes n0/(n0+n)).
Çarpan(n) = MSE_klasik / MSE_kestirici; n_esdeger = n × çarpan. GA: maddeler üzerinde küme bootstrap
(tek maddede tekrarlar üzerinden). dagilim görevinde gerçek dağılım nüfus kabul edilir, n yanıt
multinomial çekilir; PPI uygulanmaz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N0_IZGARA = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], dtype=float)
MIN_KALAN = 20            # madde başına en az n + 20 birim
ORAN_TAVAN = 1e6          # payda ~0 olunca oran patlamasın
ACIKLAMA = ("Madde ortalamasını n gerçek yanıtla tahmin ederken simülasyonu ekleyince hata kaç gerçek "
            "yanıta eşdeğer düşüyor")
KESTIRICI_ADI = {"ppi": "PPI++", "bayes": "Bayes büzülme"}


def _tr(x: float, b: int = 1) -> str:
    return f"{x:.{b}f}".replace(".", ",")


# ------------------------------------------------------------------ ortak yardımcılar
def _oran_ga(a, b, kume, B, rng):
    """Σa/Σb ve kümeler üzerinde bootstrap %95 GA."""
    kod, _ = pd.factorize(kume)
    A, Bb = np.bincount(kod, a), np.bincount(kod, b)
    C = len(A)

    def oran(pay, payda):
        return np.minimum(pay / np.maximum(payda, np.abs(pay) / ORAN_TAVAN + 1e-300), ORAN_TAVAN)

    deger = float(oran(A.sum(), Bb.sum()))
    if C < 2:
        return deger, np.nan, np.nan
    ix = rng.integers(0, C, (B, C))
    o = oran(A[ix].sum(1), Bb[ix].sum(1))
    return deger, float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def _n0_sec(E, yari):
    """E: (U, G) birim hataları; her yarı için n0 öbür yarıda seçilir. → (U,) hata, [n0_0, n0_1]."""
    secim, n0lar = np.zeros(len(E), int), []
    for h in (0, 1):
        g = int(np.argmin(E[yari != h].sum(0)))
        secim[yari == h] = g
        n0lar.append(float(N0_IZGARA[g]))
    return E[np.arange(len(E)), secim], n0lar


def _satir(n, deger, alt, ust, **ek):
    r = {"n": int(n), "carpan": round(deger, 4), "alt": None if np.isnan(alt) else round(alt, 4),
         "ust": None if np.isnan(ust) else round(ust, 4), "n_esdeger": int(round(n * deger))}
    r.update(ek)
    return r


# ------------------------------------------------------------------ birey düzeyi görevler
# Çok sınıflı etiketler tamsayı kod olarak kalır (tek-sıcak örtük): satır × sınıf tek-sıcak matris ya da
# tekrar × n × sınıf tensör kurulmaz; sayımlar bincount ile, kovaryanslar yeterli istatistiklerle. Yoğun girdiler
# (sürekli/ikili K = 1, olasılık vektörleri) özgün aritmetikle, en çok PARCA öğelik ara dizilerle.
PARCA = 4_000_000         # yoğun ara dizi üst sınırı (öğe; ~32 MB)


def _adim(birim: int, toplam: int) -> int:
    """Parça başına satır: birim × adım ≤ PARCA."""
    return max(1, min(toplam, PARCA // max(int(birim), 1)))


def _toplam(x, K):
    """Birimler üzerinde sütun toplamı → (K,)."""
    return np.bincount(x, minlength=K).astype(float) if x.ndim == 1 else x.sum(0)


def _s2(y, K):
    """Σ_k Var(y_k), ddof = 1."""
    if y.ndim > 1:
        return y.var(0, ddof=1).sum()
    S, N = _toplam(y, K), len(y)
    return (N - S @ S / N) / (N - 1)


def _etiketli(rng, R, N, n, parca=PARCA):
    """argsort(rng.random((R, N)), 1)[:, :n] ile aynı dizinler ve üreteç durumu; ara dizi ≤ parca öğe."""
    lab = np.empty((R, n), np.intp)
    adim = max(1, min(R, parca // N))
    for a in range(0, R, adim):
        u = rng.random((min(adim, R - a), N))
        p = np.argpartition(u, n - 1, axis=1)[:, :n] if n < N else np.broadcast_to(np.arange(N), u.shape)
        lab[a:a + len(u)] = np.take_along_axis(p, np.argsort(np.take_along_axis(u, p, 1), 1), 1)
    return lab


def _topla(x, lab, K):
    """Σ_j x[lab[r, j]] → (R, K): kodda bincount, yoğunda R parçalı x[lab].sum(1)."""
    R = len(lab)
    if x.ndim == 1:
        k = (x[lab] + (np.arange(R) * K)[:, None]).ravel()
        return np.bincount(k, minlength=R * K).reshape(R, K).astype(float)
    adim = _adim(lab.shape[1] * x.shape[1], R)
    return np.concatenate([x[lab[a:a + adim]].sum(1) for a in range(0, R, adim)])


def _yogun(x, lab, K):
    """x[lab] → (r, n, K); kod tek-sıcak açılır."""
    if x.ndim > 1:
        return x[lab]
    o = np.zeros((*lab.shape, K))
    np.put_along_axis(o, x[lab][..., None], 1.0, axis=-1)
    return o


def _lam_tek(y, f, lab, sy, sf, K):
    """Tek madde: λ = Cov(y, f)/Var(f) her tekrarın etiketli n biriminden (sy, sf: etiketli toplamlar). → (R,)"""
    R, n = lab.shape
    if y.ndim == 1 and f.ndim == 1:                                       # iki kod: Σ y·f = eşleşme sayısı
        cov = (y[lab] == f[lab]).sum(1) - (sy * sf).sum(1) / n
        v = n - (sf * sf).sum(1) / n
    else:
        cov, v = np.empty(R), np.empty(R)
        adim = _adim(n * (f.shape[1] if f.ndim > 1 else K), R)
        for a in range(0, R, adim):
            s = slice(a, a + adim)
            yc = _yogun(y, lab[s], K) - (sy[s] / n)[:, None]
            fc = _yogun(f, lab[s], K) - (sf[s] / n)[:, None]
            v[s], cov[s] = (fc * fc).sum((1, 2)), (yc * fc).sum((1, 2))
    return np.where(v > 0, cov / np.where(v > 0, v, 1), 0.0)


def _madde_ist(y, f, K):
    """Bir madde: σ_yf, σ²_f (iz, ddof = 1) ve yalnız simülasyon MSE'si (f̄−ȳ)² − Var(f−y)/N + Var(f)/N."""
    N = len(y)
    if y.ndim > 1:                                                        # yoğun (K = 1): özgün aritmetik
        syf = ((y - y.mean(0)) * (f - f.mean(0))).sum() / (N - 1)
        sf2 = f.var(0, ddof=1).sum()
        return syf, sf2, ((f.mean(0) - y.mean(0)) ** 2).sum() - (f - y).var(0, ddof=1).sum() / N + sf2 / N
    Sy = _toplam(y, K)
    if f.ndim == 1:
        Sf = _toplam(f, K)
        syf, sf2, f_ort = ((y == f).sum() - Sy @ Sf / N) / (N - 1), (N - Sf @ Sf / N) / (N - 1), Sf / N
    else:                                                                 # olasılık vektörü: ortalamadan sapma
        f_ort = f.mean(0)
        fc = f - f_ort
        syf, sf2 = (fc[np.arange(N), y].sum() - Sy @ fc.sum(0) / N) / (N - 1), f.var(0, ddof=1).sum()
    return syf, sf2, ((f_ort - Sy / N) ** 2).sum() - (_s2(y, K) + sf2 - 2 * syf) / N + sf2 / N


def _birey_dizileri(veri, sutunlar):
    """→ Y, Fs, madde, K. cok_sinif: Y kod (U,), Fs[c] kod (U,) ya da olasılık (U, K); diğerleri (U, 1), K = 1."""
    d = veri.df
    if veri.gorev == "cok_sinif":
        kod = {k: i for i, k in enumerate(veri.siniflar)}
        K = len(kod)

        def kodla(s):
            return s.map(lambda v: kod.get(v, kod.get(str(v), -1))).to_numpy(int)
        Y = kodla(d["gercek"])
        Fs = {c: np.vstack(d[c].to_numpy()).astype(float) if isinstance(d[c].iloc[0], np.ndarray) else kodla(d[c])
              for c in sutunlar}
        gecerli = (Y >= 0) & (Y < K)
        for F in Fs.values():
            gecerli &= ((F >= 0) & (F < K)) if F.ndim == 1 else (np.isfinite(F).all(1) & (F.sum(1) > 0))
    else:
        K = 1
        Y = pd.to_numeric(d["gercek"], errors="coerce").to_numpy(float)[:, None]
        Fs = {c: pd.to_numeric(d[c], errors="coerce").to_numpy(float)[:, None] for c in sutunlar}
        gecerli = np.isfinite(Y[:, 0])
        for F in Fs.values():
            gecerli &= np.isfinite(F[:, 0])
    madde = pd.factorize(d["madde"])[0] if "madde" in d else np.zeros(len(d), int)
    if gecerli.all():                                                     # büyük olasılık matrisini kopyalama
        return Y, Fs, madde, K
    return Y[gecerli], {c: F[gecerli] for c, F in Fs.items()}, madde[gecerli], K


def _birey(veri, sutunlar, n_listesi, R, B, rng, notlar):
    Y, Fs, madde, K = _birey_dizileri(veri, sutunlar)
    var = np.flatnonzero(madde >= 0)                                      # factorize: eksik madde kimliği −1, dışarıda
    sira = var[np.argsort(madde[var], kind="stable")]
    gruplar = [ix for ix in np.split(sira, np.cumsum(np.bincount(madde[var]))[:-1]) if len(ix) >= 2] if len(var) else []
    M = len(gruplar)
    if M == 0:
        return None
    tek = M == 1
    if tek:
        notlar.append("tek madde: λ etiketli n birimden, n0 tekrar yarılarında seçildi, GA tekrarlar üzerinden")
    # madde istatistikleri: σ², σ_yf, σ²_f (iz, sınıflar üzerinde toplam), yalnız simülasyon MSE'si
    s2 = np.array([_s2(Y[ix], K) for ix in gruplar])
    ist = {c: np.array([_madde_ist(Y[ix], F[ix], K) for ix in gruplar]) for c, F in Fs.items()}   # (M, 3)
    syf = {c: ist[c][:, 0] for c in Fs}
    sf2 = {c: ist[c][:, 1] for c in Fs}
    Nm = np.array([len(ix) for ix in gruplar], float)
    # λ çapraz uydurma: h yarısı öbür yarının havuzlanmış λ'sını kullanır
    yari_m = rng.permutation(M) % 2
    lam = {}
    for c in Fs:
        pay = np.bincount(yari_m, syf[c] * (Nm - 1), 2)
        payda = np.bincount(yari_m, sf2[c] * (Nm - 1), 2)
        l = np.where(payda > 0, pay / np.where(payda > 0, payda, 1), 0.0)
        lam[c] = l[::-1]
    sonuc = {c: {"ppi": [], "bayes": []} for c in Fs}

    for n in n_listesi:
        uygun = [i for i in range(M) if Nm[i] >= n + MIN_KALAN]
        if not uygun or n < 2:
            continue
        E_kl, E_ppi, E_bay = [], {c: [] for c in Fs}, {c: [] for c in Fs}
        w = N0_IZGARA / (N0_IZGARA + n)                                   # (G,)
        for i in uygun:
            ix = gruplar[i]
            N = len(ix)
            y = Y[ix]
            lab = _etiketli(rng, R, N, n)                                 # (R, n)
            sy = _topla(y, lab, K)
            yn = sy / n
            teta = (_toplam(y, K) - sy) / (N - n)                         # kalan birimlerin ortalaması
            duz = s2[i] / (N - n)
            E_kl.append(((yn - teta) ** 2).sum(1) - duz)
            for c, F in Fs.items():
                f = F[ix]
                sf = _topla(f, lab, K)
                fn, fN = sf / n, _toplam(f, K) / N
                l = _lam_tek(y, f, lab, sy, sf, K) if tek else np.full(R, lam[c][yari_m[i]])   # tek: etiketliden
                ppi = yn + l[:, None] * (fN - fn)
                E_ppi[c].append(((ppi - teta) ** 2).sum(1) - duz + 2 * l * syf[c][i] / N)
                bay = np.array([(((1 - g) * yn + g * fN - teta) ** 2).sum(1) for g in w])   # (G, R)
                E_bay[c].append((bay - duz + 2 * w[:, None] * syf[c][i] / N).T)
        if tek:   # birim = tekrar
            a = E_kl[0]
            kume = np.arange(R)
            yari = np.arange(R) % 2
            ppi_e = {c: E_ppi[c][0] for c in Fs}
            bay_e = {c: E_bay[c][0] for c in Fs}                          # (R, G)
        else:     # birim = madde (tekrar ortalaması)
            a = np.array([e.mean() for e in E_kl])
            kume = np.arange(len(uygun))
            yari = rng.permutation(len(uygun)) % 2
            ppi_e = {c: np.array([e.mean() for e in E_ppi[c]]) for c in Fs}
            bay_e = {c: np.array([e.mean(0) for e in E_bay[c]]) for c in Fs}
        for c in Fs:
            d, lo, hi = _oran_ga(a, ppi_e[c], kume, B, rng)
            sonuc[c]["ppi"].append(_satir(n, d, lo, hi, n_madde=len(uygun),
                                          **({} if tek else {"lambda": round(float(lam[c].mean()), 4)})))
            e, n0lar = _n0_sec(bay_e[c], yari)
            d, lo, hi = _oran_ga(a, e, kume, B, rng)
            sonuc[c]["bayes"].append(_satir(n, d, lo, hi, n0=float(np.median(n0lar)), n0_katlar=n0lar,
                                            n_madde=len(uygun)))
    # yalnız simülasyon: σ²/MSE, MSE = (f̄−ȳ)² − Var(f−y)/N + Var(f)/N
    for c, F in Fs.items():
        if tek:
            sonuc[c]["yalniz_sim"] = _yalniz_tek(Y[gruplar[0]], F[gruplar[0]], B, rng, K)
            continue
        d, lo, hi = _oran_ga(s2, ist[c][:, 2], np.arange(M), B, rng)
        sonuc[c]["yalniz_sim"] = _esdeger(d, lo, hi)
    return sonuc


def _esdeger(d, lo, hi):
    yuv = lambda x: None if np.isnan(x) else round(float(x), 2)
    return {"esdeger_n": yuv(d), "alt": yuv(lo), "ust": yuv(hi)}


def _yalniz_tek(y, f, B, rng, K=1, parca=20):
    """Tek madde: birim bootstrap'ı (parça parça, bellek sınırlı). Kodlu y'de tekrar ağırlıklı yeterli
    istatistikler; (b, N, K) dizi kurulmaz."""
    N = len(y)
    oran = lambda s2, mse: np.minimum(s2 / np.maximum(mse, s2 / ORAN_TAVAN + 1e-300), ORAN_TAVAN)
    if y.ndim > 1:                                                        # yoğun (K = 1): özgün aritmetik
        def es(ix):
            if len(ix) > 1 and len(ix) * N * y.shape[1] > PARCA:               # tek tekrar bölünemez
                adim = _adim(N * y.shape[1], len(ix))
                return np.concatenate([es(ix[a:a + adim]) for a in range(0, len(ix), adim)])
            yy, ff = y[ix], f[ix]                                         # (b, N, K)
            mse = (((ff.mean(1) - yy.mean(1)) ** 2).sum(-1) - (ff - yy).var(1, ddof=1).sum(-1) / N
                   + ff.var(1, ddof=1).sum(-1) / N)
            return oran(yy.var(1, ddof=1).sum(-1), mse)
    else:
        if f.ndim > 1:                                                    # olasılık vektörü: ortalamadan sapma
            f_ort = f.mean(0)
            fc = f - f_ort
            qf, gf = (fc * fc).sum(1), fc[np.arange(N), y]
        else:
            gf = (y == f).astype(float)

        def es(ix):
            b = len(ix)
            W = np.bincount((ix + (np.arange(b) * N)[:, None]).ravel(), minlength=b * N).reshape(b, N).astype(float)
            satir = (np.arange(b) * K)[:, None]
            Sy = np.bincount((satir + y).ravel(), weights=W.ravel(), minlength=b * K).reshape(b, K)
            if f.ndim > 1:
                Sf, ff, ort_f = W @ fc, W @ qf, (W @ fc) / N + f_ort
            else:
                Sf = np.bincount((satir + f).ravel(), weights=W.ravel(), minlength=b * K).reshape(b, K)
                ff, ort_f = N, Sf / N
            s2 = (N - (Sy * Sy).sum(1) / N) / (N - 1)
            sf2 = (ff - (Sf * Sf).sum(1) / N) / (N - 1)
            syf = (W @ gf - (Sy * Sf).sum(1) / N) / (N - 1)
            return oran(s2, ((ort_f - Sy / N) ** 2).sum(1) - (s2 + sf2 - 2 * syf) / N + sf2 / N)
    d = float(es(np.arange(N)[None])[0])
    o = np.concatenate([es(rng.integers(0, N, (min(parca, B - k), N))) for k in range(0, B, parca)])
    return _esdeger(d, np.percentile(o, 2.5), np.percentile(o, 97.5))

# ------------------------------------------------------------------ dağılım görevi
def _dagilim(veri, sutunlar, n_listesi, R, B, rng, notlar, kontrol):
    d = veri.df
    P = [np.asarray(v, float) for v in d["gercek"]]
    Qs = {c: [np.asarray(v, float) for v in d[c]] for c in sutunlar}
    gecerli = np.array([all(len(Qs[c][i]) == len(P[i]) for c in sutunlar) for i in range(len(P))])
    satir = np.flatnonzero(gecerli)
    if not len(satir):
        return None
    kume_ham = d["madde"].to_numpy()[satir] if "madde" in d else satir
    kume = pd.factorize(kume_ham)[0]
    C = kume.max() + 1
    tek = C == 1
    s2 = np.array([(P[i] * (1 - P[i])).sum() for i in satir])
    sonuc = {c: {"ppi": [], "bayes": []} for c in sutunlar}
    Kler = np.array([len(P[i]) for i in satir])
    for n in n_listesi:
        w = N0_IZGARA / (N0_IZGARA + n)
        E_kl = np.zeros((len(satir), R))
        E_bay = {c: np.zeros((len(satir), R, len(w))) for c in sutunlar}
        for K in np.unique(Kler):                                          # aynı seçenek sayılı satırlar birlikte
            j = np.flatnonzero(Kler == K)
            p = np.vstack([P[satir[t]] for t in j])
            p = np.clip(p, 0, None) / np.clip(p, 0, None).sum(1, keepdims=True)
            fr = rng.multinomial(n, p, size=(R, len(j))).transpose(1, 0, 2) / n   # (U, R, K)
            E_kl[j] = ((fr - p[:, None]) ** 2).sum(2)
            for c in sutunlar:
                q = np.vstack([Qs[c][satir[t]] for t in j])
                bay = (1 - w)[None, None, :, None] * fr[:, :, None] + w[None, None, :, None] * q[:, None, None]
                E_bay[c][j] = ((bay - p[:, None, None]) ** 2).sum(3)
        kontrol.setdefault("klasik_ampirik_analitik_orani", {})[str(n)] = round(
            float(E_kl.mean(1).sum() / (s2 / n).sum()), 4)
        if tek:
            a, kumeler, yari = E_kl.mean(0), np.arange(R), np.arange(R) % 2
            bay_e = {c: E_bay[c].mean(0) for c in sutunlar}               # (R, G)
        else:
            a, kumeler = E_kl.mean(1), kume
            yari = (rng.permutation(C) % 2)[kume]
            bay_e = {c: E_bay[c].mean(1) for c in sutunlar}               # (U, G)
        for c in sutunlar:
            e, n0lar = _n0_sec(bay_e[c], yari)
            dd, lo, hi = _oran_ga(a, e, kumeler, B, rng)
            sonuc[c]["bayes"].append(_satir(n, dd, lo, hi, n0=float(np.median(n0lar)), n0_katlar=n0lar,
                                            n_madde=int(C)))
    for c in sutunlar:
        mse = np.array([((Qs[c][i] - P[i]) ** 2).sum() for i in satir])
        dd, lo, hi = _oran_ga(s2, mse, kume if not tek else satir, B, rng)
        sonuc[c]["yalniz_sim"] = _esdeger(dd, lo, hi)
    notlar.append("dagilim: gerçek dağılım nüfus kabul edildi, yanıtlar multinomial çekildi; PPI uygulanmaz")
    return sonuc


# ------------------------------------------------------------------ uygulanabilirlik
def panel_yok(df: pd.DataFrame, gorev: str | None, *, ham_var: bool, llm_yok: bool) -> tuple[str, str] | None:
    """İkili görevde insan paneli yok mu: PPI++ 'kaç insana bedel' n gerçek insanın yanıtına simülasyon ekler; olay
    günlüğü (oturum → satın alma) gibi bir ikili görevde yerine geçilecek insan yanıtı yoktur (D13 #11c, D13-W8 #7).
    Panel yok sayılır: görev ikili, LLM kolu yok (ham kol yok ve ours_uses_llm=False), madde sütunu yok ya da tek
    madde, insan tekrar (retest) sütunu yok. Döner (Türkçe neden, İngilizce neden) | None."""
    if gorev != "ikili" or ham_var or not llm_yok or "tekrar" in df.columns:
        return None
    if "madde" in df.columns and df["madde"].nunique() > 1:
        return None
    return ("ikili görev, insan paneli yok (madde yok, LLM kolu yok: ours_uses_llm=False); yerine geçilecek insan "
            "yanıtı yok",
            "binary task without a human panel (no item column, no LLM arm: ours_uses_llm=False), no human answers "
            "to stand in for")


# ------------------------------------------------------------------ ana API
def carpan(veri, sutunlar: dict, n_listesi=(10, 25, 50, 100, 200), R: int = 40,
           tohum: int = 0, B: int = 300) -> dict | None:
    """Etkin örneklem çarpanı → OZET-SEMA.md 'orneklem' sözlüğü; uygulanamıyorsa None."""
    if veri.gorev == "ab" or not sutunlar:
        return None
    sutunlar = {c: e for c, e in sutunlar.items() if c in veri.df.columns}
    if not sutunlar or len(veri.df) < min(n_listesi) + MIN_KALAN:
        return None
    rng = np.random.default_rng(tohum)
    notlar, kontrol = [], {}
    if "madde" not in veri.df or veri.df["madde"].nunique() < 20:
        R = max(R, 400)  # az madde: Monte Carlo gürültüsünü tekrar sayısıyla bastır
    if veri.gorev == "dagilim":
        sonuc = _dagilim(veri, list(sutunlar), n_listesi, R, B, rng, notlar, kontrol)
    else:
        sonuc = _birey(veri, list(sutunlar), n_listesi, R, B, rng, notlar)
    if not sonuc or not any(s["ppi"] or s["bayes"] for s in sonuc.values()):
        return None
    tek = "madde" not in veri.df or veri.df["madde"].nunique() < 20
    if ("madde" not in veri.df or veri.df["madde"].nunique() == 1) and veri.gorev in ("surekli", "ikili"):
        _bedel_ppi(veri, sutunlar, sonuc, n_listesi, tohum, notlar)
    yontemler = {c: {"etiket": sutunlar[c], **sonuc[c]} for c in sutunlar}
    if tek:
        notlar.append("madde sayısı < 20: Bayes önselinin yanlılığı maddeler arasında sınanamaz, çarpanı iyimser "
                      "olabilir; manşet PPI++ (yanlılıktan bağımsız geçerli) ile verildi")
    return {"aciklama": ACIKLAMA, "n_listesi": [int(n) for n in n_listesi],
            "cumle": _cumle(yontemler, kestiriciler=("ppi",) if tek else ("ppi", "bayes")),
            "yontemler": yontemler, "gorev": veri.gorev, "R": R, "B": B, "n0_izgara": N0_IZGARA.tolist(),
            "notlar": notlar, "kontrol": kontrol}


def _bedel_ppi(veri, sutunlar, sonuc, n_listesi, tohum, notlar, B_bedel: int = 200):
    """Tek maddede PPI++ satırlarını kanit.bedel ile yeniden ölç: pano ile "kaç insana bedel" grafiği aynı sayıyı
    versin (KARŞIT 4: tek kestirim). Hedef kitle ortalaması (sonlu kitle), GA birim bootstrap'ı."""
    try:
        from . import bedel as bd
    except ImportError:  # testler modülü paketsiz yükler
        import importlib.util
        spec = importlib.util.spec_from_file_location("kanit_bedel", Path(__file__).with_name("bedel.py"))
        bd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bd)
    Y, Fs, _, _ = _birey_dizileri(veri, list(sutunlar))
    for c, F in Fs.items():
        satirlar = []
        for n in n_listesi:
            if len(Y) < 2 * n:
                continue
            q = bd.olc([(Y[:, 0], F[:, 0])], n=int(n), B=B_bedel, tohum=tohum)
            r = _satir(n, q["carpan"], q["carpan_ga"][0], q["carpan_ga"][1], n_madde=1, kapsama=round(q["kapsama"], 3))
            r["n_esdeger"] = int(round(q["n_esdeger"]))
            satirlar.append(r)
        sonuc[c]["ppi"] = satirlar
    notlar.append("Tek madde: PPI++ satırları 'kaç insana bedel' grafiğiyle aynı kestirim (kanit.bedel; hedef kitle "
                  "ortalaması, sonlu kitle düzeltmeli; GA birim bootstrap'ı). ≈ insan sütunu sonlu kitle düzeltmeli olduğu "
                  "için n × çarpandan küçüktür.")


def _cumle(yontemler: dict, hedef: int = 100, kestiriciler=("ppi", "bayes")) -> str:
    """En çok çarpan veren yöntem/kestirici için, n = 100 (yoksa en yakın) değerinde jüri cümlesi."""
    en = None
    for c, y in yontemler.items():
        for k in kestiriciler:
            if not y[k]:
                continue
            r = min(y[k], key=lambda s: (abs(np.log(s["n"] / hedef)), -s["n"]))
            anahtar = (abs(np.log(r["n"] / hedef)), -r["carpan"])
            if en is None or anahtar < en[0]:
                en = (anahtar, y["etiket"], k, r)
    if en is None:
        return ""
    _, etiket, k, r = en
    ga = f" [{_tr(r['alt'])}–{_tr(r['ust'])}]" if r["alt"] is not None else ""
    return (f"{etiket} + {r['n']} gerçek yanıt ≈ {r['n_esdeger']} gerçek yanıt "
            f"(çarpan {_tr(r['carpan'])}×{ga}; {KESTIRICI_ADI[k]})")
