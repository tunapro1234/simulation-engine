"""Tarzlar arası "kaç insana bedel" noktaları: ρ (simülasyon ile gerçek arasındaki birim düzeyi korelasyon)
ve ölçülen PPI++ çarpanı, kuramsal eğri 1/(1 − (1 − n/N)·ρ²) ile.

Bir nokta = bir veri × bir önsel (simülasyon/model). Hedef her sütunun (sorunun, kitlenin) ortalamasıdır;
n kişi rastgele etiketlenir, önsel herkes için vardır. Tahmin edici intent ppi.py ile aynı (PPI++):
λ = Cov(y,f) / ((1 + n/N)·Var(f)), ikisi de etiketli örneklemden, [0, 1]'e kırpılır. (Paydada kitle varyansı
kullanmak çarpık veride çarpanı düşük gösteriyor: OE GBDT n=25'te 2,0× yerine 5,1×; Twin'de fark yok.)
  çarpan  = ortalama MSE(klasik) / ortalama MSE(PPI++)   (sütunlar sd ile standartlaştırılır); kuram 1/(1 − ρ²)
  n_eşdeğer = PPI++ ile aynı MSE'yi veren düz ortalamanın kişi sayısı, sonlu kitle düzeltmeli (persona twin_ess.py
            tanımı): 1 / (MSE_ppi/S² + 1/N); kuram n/(1 − (1 − n/N)·ρ²). n ≪ N iken ikisi aynı; N=500, n=100'de
            n_eşdeğer/n çarpandan belirgin küçük.
  ρ_etkin = √(sütunlar üzerinden ortalama ρ²)            (çok sütunda kuramsal çarpanı veren ρ)
GA: ≥20 sütunda sütun bootstrap'ı (yeni sorulara genelleme), daha azında birim bootstrap'ı (yeni kişilere).
"""
from __future__ import annotations

import numpy as np

SUTUN_BOOT_ESIK = 20


def _mse(y, f, n, R, rng):
    """Tek sütun: (klasik MSE, PPI++ MSE), θ = y'nin kitle ortalaması. Klasik MSE kapalı form (iadesiz örneklem
    varyansı S²(1 − n/N)/n); çarpık sütunlarda Monte Carlo oranının gürültüsünü yarıya indirir."""
    N = len(y)
    mu, fm = y.mean(), f.mean()
    idx = np.argpartition(rng.random((R, N)), n, axis=1)[:, :n]
    ys, fs = y[idx], f[idx]
    yb, fb = ys.mean(1), fs.mean(1)
    cov = ((ys - yb[:, None]) * (fs - fb[:, None])).sum(1) / (n - 1)
    vf = ((fs - fb[:, None]) ** 2).sum(1) / (n - 1)
    lam = np.clip(np.where(vf > 0, cov / ((1 + n / N) * np.where(vf > 0, vf, 1)), 0.0), 0, 1)
    tah = yb + lam * (fm - fb)
    # %95 GA kapsaması: se² = Var_s(y − λf)(1 − n/N)/n (f herkes için bilinir, belirsizlik yalnız etiketli örneklemden)
    r = ys - lam[:, None] * fs
    se = np.sqrt(((r - r.mean(1)[:, None]) ** 2).sum(1) / (n - 1) * (1 - n / N) / n)
    kapsama = float((np.abs(tah - mu) <= 1.96 * se).mean())
    return y.var(ddof=1) * (1 - n / N) / n, ((tah - mu) ** 2).mean(), kapsama


def _hazir(sutunlar, n, indeks=False):
    out, ix = [], []
    for i, (y, f) in enumerate(sutunlar):
        y, f = np.asarray(y, float), np.asarray(f, float)
        m = np.isfinite(y) & np.isfinite(f)
        y, f = y[m], f[m]
        if len(y) < 2 * n or y.std() == 0:
            continue
        s = y.std()
        out.append((y / s, f / s))
        ix.append(i)
    return (out, ix) if indeks else out


def _basiklik(y) -> float:
    """Fazla basıklık (normal = 0); tek aykırı değerli sütunu işaretler (KARŞIT-2 K2-31: Twin Q166, 5000)."""
    z = (y - y.mean()) / y.std()
    return float((z ** 4).mean() - 3)


def _rho2(y, f):
    return float(np.corrcoef(y, f)[0, 1] ** 2) if f.std() > 0 else 0.0


def olc(sutunlar, n: int = 100, R: int = 400, B: int = 200, tohum: int = 0) -> dict:
    """sutunlar: [(y, f), ...] — her biri bir sütunun (soru/kitle) gerçek ve önsel değerleri."""
    rng = np.random.default_rng(tohum)
    S, ix = _hazir(sutunlar, n, indeks=True)
    if not S:
        raise ValueError(f"n={n} için yeterli satırı olan sütun yok")
    if len(S) < SUTUN_BOOT_ESIK:  # az sütunda Monte Carlo gürültüsü baskın: tekrarı artır
        R = max(R, 4000)
    mk, mp, kap = np.array([_mse(y, f, n, R, rng) for y, f in S]).T
    r2 = np.array([_rho2(y, f) for y, f in S])
    Ns = np.array([len(y) for y, _ in S])
    nokta = {"n": n, "N": float(Ns.mean()), "sutun": len(S), "R": R,
             "carpan": float(mk.sum() / mp.sum()), "rho": float(np.sqrt(r2.mean())),
             "kapsama": float(kap.mean()), "kapsama_en_dusuk": float(kap.min())}
    if len(S) > 1:  # en kötü sütun tek başına mı? (dipnot soru ortalamasını değil o sütunu anlatmalı)
        j = int(kap.argmin())
        bas = np.array([_basiklik(y) for y, _ in S])
        nokta.update({"kapsama_en_dusuk_sutun": ix[j], "kapsama_en_dusuk_basiklik": float(bas[j]),
                      "kapsama_ikinci_en_dusuk": float(np.sort(kap)[1]), "kapsama_en_kotu_haric": float(np.delete(kap, j).mean()),
                      "kapsama_090_alti": int((kap < 0.90).sum()), "basiklik_ust50": int((bas > 50).sum()),
                      "kapsama_basik_haric": float(kap[bas <= 50].mean()) if (bas <= 50).any() else None})
    if len(S) >= SUTUN_BOOT_ESIK:
        k = rng.integers(0, len(S), (B, len(S)))
        cb = mk[k].sum(1) / mp[k].sum(1)
        rb = np.sqrt(r2[k].mean(1))
        nokta["ga_turu"] = "sütun bootstrap"
    else:
        cb, rb = [], []
        for _ in range(B):
            Sb = [(y[i], f[i]) for y, f in S for i in [rng.integers(0, len(y), len(y))]]
            a, b, _ = np.array([_mse(y, f, n, 1000, rng) for y, f in Sb]).T
            cb.append(a.sum() / b.sum())
            rb.append(np.sqrt(np.mean([_rho2(y, f) for y, f in Sb])))
        cb, rb = np.array(cb), np.array(rb)
        nokta["ga_turu"] = "birim bootstrap"
    nokta["carpan_ga"] = [float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))]
    nokta["rho_ga"] = [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]
    nokta["teorik"] = float(1 / (1 - nokta["rho"] ** 2))
    # klasik MSE kapalı form S²(1/n − 1/N) olduğundan n_eşdeğer = 1 / (MSE_ppi/S² + 1/N), sütun ortalamalı
    nokta["n_esdeger"] = float(1 / ((mp / (mk / (1 / n - 1 / Ns))).mean() + (1 / Ns).mean()))
    nokta["n_esdeger_teorik"] = float(n / (1 - (1 - n / nokta["N"]) * nokta["rho"] ** 2))
    return nokta


def egri(n: int | None = None, N: float | None = None, adim: int = 101) -> list[dict]:
    """Kuramsal eğri 1/(1 − k·ρ²); n/N verilirse k = 1 − n/N (n_eşdeğer/n), verilmezse k = 1 (çarpan)."""
    k = 1 - (n / N if n and N else 0)
    return [{"rho": float(r), "carpan": float(1 / (1 - k * r * r))} for r in np.linspace(0, 0.97, adim)]
