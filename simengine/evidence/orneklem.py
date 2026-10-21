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


def _onehot(kod: np.ndarray, K: int) -> np.ndarray:
    o = np.zeros((len(kod), K))
    ok = (kod >= 0) & (kod < K)
    o[np.flatnonzero(ok), kod[ok]] = 1.0
    return o


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
def _birey_dizileri(veri, sutunlar):
    d = veri.df
    if veri.gorev == "cok_sinif":
        kod = {k: i for i, k in enumerate(veri.siniflar)}
        K = len(kod)
        Y = _onehot(d["gercek"].map(lambda v: kod.get(v, kod.get(str(v), -1))).to_numpy(int), K)
        Fs = {}
        for c in sutunlar:
            s = d[c]
            if isinstance(s.iloc[0], np.ndarray):
                Fs[c] = np.vstack(s.to_numpy()).astype(float)
            else:
                Fs[c] = _onehot(s.map(lambda v: kod.get(v, kod.get(str(v), -1))).to_numpy(int), K)
        gecerli = Y.sum(1) > 0
        for F in Fs.values():
            gecerli &= np.isfinite(F).all(1) & (F.sum(1) > 0)
    else:
        Y = pd.to_numeric(d["gercek"], errors="coerce").to_numpy(float)[:, None]
        Fs = {c: pd.to_numeric(d[c], errors="coerce").to_numpy(float)[:, None] for c in sutunlar}
        gecerli = np.isfinite(Y[:, 0])
        for F in Fs.values():
            gecerli &= np.isfinite(F[:, 0])
    madde = pd.factorize(d["madde"])[0] if "madde" in d else np.zeros(len(d), int)
    return Y[gecerli], {c: F[gecerli] for c, F in Fs.items()}, madde[gecerli]


def _birey(veri, sutunlar, n_listesi, R, B, rng, notlar):
    Y, Fs, madde = _birey_dizileri(veri, sutunlar)
    M = int(madde.max()) + 1 if len(madde) else 0
    gruplar = [ix for ix in (np.flatnonzero(madde == m) for m in range(M)) if len(ix) >= 2]
    M = len(gruplar)
    if M == 0:
        return None
    tek = M == 1
    if tek:
        notlar.append("tek madde: λ etiketli n birimden, n0 tekrar yarılarında seçildi, GA tekrarlar üzerinden")
    # madde istatistikleri: σ², σ_yf (iz, sınıflar üzerinde toplam)
    s2 = np.array([Y[ix].var(0, ddof=1).sum() for ix in gruplar])
    syf = {c: np.array([((Y[ix] - Y[ix].mean(0)) * (F[ix] - F[ix].mean(0))).sum() / (len(ix) - 1)
                        for ix in gruplar]) for c, F in Fs.items()}
    sf2 = {c: np.array([F[ix].var(0, ddof=1).sum() for ix in gruplar]) for c, F in Fs.items()}
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
            lab = np.argsort(rng.random((R, N)), axis=1)[:, :n]           # (R, n)
            yl = y[lab]                                                   # (R, n, K)
            yn = yl.mean(1)
            teta = (y.sum(0) - yl.sum(1)) / (N - n)                       # kalan birimlerin ortalaması
            duz = s2[i] / (N - n)
            E_kl.append(((yn - teta) ** 2).sum(1) - duz)
            for c, F in Fs.items():
                f = F[ix]
                fl = f[lab]
                fn, fN = fl.mean(1), f.mean(0)
                if tek:                                                   # λ etiketli örneklemden
                    yc, fc = yl - yn[:, None], fl - fn[:, None]
                    v = (fc * fc).sum((1, 2))
                    l = np.where(v > 0, (yc * fc).sum((1, 2)) / np.where(v > 0, v, 1), 0.0)
                else:
                    l = np.full(R, lam[c][yari_m[i]])
                ppi = yn + l[:, None] * (fN - fn)
                E_ppi[c].append(((ppi - teta) ** 2).sum(1) - duz + 2 * l * syf[c][i] / N)
                bay = (1 - w)[:, None, None] * yn + w[:, None, None] * fN  # (G, R, K)
                E_bay[c].append((((bay - teta) ** 2).sum(2) - duz + 2 * w[:, None] * syf[c][i] / N).T)
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
            sonuc[c]["yalniz_sim"] = _yalniz_tek(Y[gruplar[0]], F[gruplar[0]], B, rng)
            continue
        mse = np.array([((F[ix].mean(0) - Y[ix].mean(0)) ** 2).sum()
                        - (F[ix] - Y[ix]).var(0, ddof=1).sum() / len(ix) + sf2[c][i] / len(ix)
                        for i, ix in enumerate(gruplar)])
        d, lo, hi = _oran_ga(s2, mse, np.arange(M), B, rng)
        sonuc[c]["yalniz_sim"] = _esdeger(d, lo, hi)
    return sonuc


def _esdeger(d, lo, hi):
    yuv = lambda x: None if np.isnan(x) else round(float(x), 2)
    return {"esdeger_n": yuv(d), "alt": yuv(lo), "ust": yuv(hi)}


def _yalniz_tek(y, f, B, rng, parca=20):
    """Tek madde: birim bootstrap'ı (parça parça, bellek sınırlı)."""
    def es(ix):
        yy, ff = y[ix], f[ix]                                             # (b, N, K)
        N = yy.shape[1]
        s2 = yy.var(1, ddof=1).sum(-1)
        mse = (((ff.mean(1) - yy.mean(1)) ** 2).sum(-1) - (ff - yy).var(1, ddof=1).sum(-1) / N
               + ff.var(1, ddof=1).sum(-1) / N)
        return np.minimum(s2 / np.maximum(mse, s2 / ORAN_TAVAN + 1e-300), ORAN_TAVAN)
    N = len(y)
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
    Y, Fs, _ = _birey_dizileri(veri, list(sutunlar))
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
