"""Kalibrasyon: güvenilirlik eğrisi, ECE, güven AUROC, çekimser kalma eğrisi ve
çapraz uydurmalı yeniden kalibrasyon (öncesi-sonrası aynı bootstrap'tan geçer).

Yeniden kalibrasyon her zaman katlar dışından uydurulur: bir birimin kalibre tahmini, o birimin
grubunu görmemiş bir eşlemeden gelir. Böylece "kalibrasyon sonrası" sayı iyimser değildir.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .merdiven import katlar

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _capraz(veri, uydur_uygula):
    """uydur_uygula(egitim_idx, test_idx) → test tahminleri. bolum=egitim varsa onu kullanmaz:
    kalibrasyon eğitim satırlarında tahmin gerektirir, çoğu girdide yok; test içinde çapraz uydururuz."""
    k = katlar(veri.df, tohum=7)
    out = [None] * len(veri.df)
    for f in np.unique(k):
        tr, te = np.flatnonzero(k != f), np.flatnonzero(k == f)
        for i, v in zip(te, uydur_uygula(tr, te)):
            out[i] = v
    return out


def yeniden(veri, sutun: str) -> dict:
    """{ek_ad: değerler}. ikili: platt + izotonik; cok_sinif: sıcaklık; surekli: doğrusal + izotonik;
    dagilim: sıcaklık (dağılımı keskinleştir/yay, JS'ye göre)."""
    g = veri.gorev
    y_ham = veri.df["gercek"]
    p_ham = veri.df[sutun]
    out = {}
    if g == "ikili":
        y, p = y_ham.to_numpy(float), p_ham.to_numpy(float)
        z = _logit(p).reshape(-1, 1)

        def platt(tr, te):
            if len(np.unique(y[tr])) < 2:
                return p[te]
            m = LogisticRegression(C=1e4, max_iter=200).fit(z[tr], y[tr])
            return m.predict_proba(z[te])[:, 1]

        def izo(tr, te):
            return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[tr], y[tr]).predict(p[te])
        out["platt"] = np.asarray(_capraz(veri, platt), float)
        out["izotonik"] = np.clip(np.asarray(_capraz(veri, izo), float), EPS, 1 - EPS)
    elif g == "cok_sinif" and isinstance(p_ham.iloc[0], np.ndarray):
        kod = {k: i for i, k in enumerate(veri.siniflar)}
        y = y_ham.map(lambda v: kod.get(v, kod.get(str(v), 0))).to_numpy()
        P = np.vstack(p_ham.to_numpy())
        L = np.log(np.clip(P, EPS, 1))

        def olcekle(T, idx):
            z = L[idx] / T
            z -= z.max(1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(1, keepdims=True)

        def sicaklik(tr, te):
            nll = lambda T: -np.mean(np.log(np.clip(olcekle(T, tr)[np.arange(len(tr)), y[tr]], EPS, 1)))
            T = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x
            return list(olcekle(T, te))
        out["sicaklik"] = _capraz(veri, sicaklik)
    elif g == "surekli":
        y, p = y_ham.to_numpy(float), p_ham.to_numpy(float)

        def dogrusal(tr, te):
            A = np.c_[np.ones(len(tr)), p[tr]]
            b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
            return b[0] + b[1] * p[te]

        def izo(tr, te):
            return IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr]).predict(p[te])
        out["dogrusal"] = np.asarray(_capraz(veri, dogrusal), float)
        out["izotonik"] = np.asarray(_capraz(veri, izo), float)
    elif g == "dagilim":
        Y, P = list(y_ham), list(p_ham)
        from .metrik import js

        def us(T, idx):
            r = []
            for i in idx:
                q = np.clip(P[i], 1e-9, 1) ** (1 / T)
                r.append(q / q.sum())
            return r

        def sicaklik(tr, te):
            kayip = lambda T: np.mean([js(Y[i], q) for i, q in zip(tr, us(T, tr))])
            T = minimize_scalar(kayip, bounds=(0.2, 5), method="bounded", options={"xatol": 1e-2}).x
            return us(T, te)
        out["sicaklik"] = _capraz(veri, sicaklik)
    return out


def egri(veri, y, p, kutu: int = 10) -> dict | None:
    """Güvenilirlik eğrisi + ECE + güven AUROC + kapsam-doğruluk (çekimser kalma) eğrisi."""
    if veri.gorev == "ikili":
        guven_poz = p
        dogru = ((p > 0.5) == (y > 0.5)).astype(float)
        guven = np.maximum(p, 1 - p)
        x_kutu, y_kutu = guven_poz, y  # eğri: tahmin edilen pozitif olasılığı vs gözlenen oran
    elif veri.gorev == "cok_sinif" and p.ndim == 2:
        guven = p.max(1)
        dogru = (p.argmax(1) == y).astype(float)
        x_kutu, y_kutu = guven, dogru
    else:
        return None
    k = np.clip((x_kutu * kutu).astype(int), 0, kutu - 1)
    noktalar, ece = [], 0.0
    for b in range(kutu):
        m = k == b
        if m.any():
            noktalar.append({"guven": float(x_kutu[m].mean()), "dogruluk": float(y_kutu[m].mean()), "n": int(m.sum())})
            ece += m.mean() * abs(y_kutu[m].mean() - x_kutu[m].mean())
    sira = np.argsort(-guven, kind="stable")
    kum = np.cumsum(dogru[sira]) / np.arange(1, len(sira) + 1)
    kapsam = []
    for c in np.linspace(0.1, 1.0, 10):
        j = max(int(round(c * len(sira))) - 1, 0)
        kapsam.append({"kapsam": float(round(c, 2)), "dogruluk": float(kum[j])})
    aurc = float(np.mean(1 - kum))  # risk-kapsam eğrisi altı alan (küçük iyi)
    try:
        g_auroc = float(roc_auc_score(dogru, guven)) if 0 < dogru.sum() < len(dogru) else None
    except ValueError:
        g_auroc = None
    return {"egri": noktalar, "ece": float(ece), "guven_auroc": g_auroc, "aurc": aurc, "kapsam": kapsam}
