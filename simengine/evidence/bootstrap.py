"""Küme bootstrap motoru. Bütün sütunlar ve metrikler AYNI yeniden örneklerle hesaplanır (eşleşik).

birim metrikler: küme toplamları × örnek sayıları (hızlı yol, B=1000).
toplu metrikler: her tekrar için satır indeksi açılır (döngü, B_toplu).
"""
from __future__ import annotations

import numpy as np


class Orneklem:
    def __init__(self, kume: np.ndarray, B: int = 1000, B_toplu: int = 200, tohum: int = 0,
                 madde_kume: bool = False):
        self.kume = np.asarray(kume)
        self.G = int(self.kume.max()) + 1 if len(self.kume) else 0
        self.B, self.B_toplu, self.tohum = B, min(B_toplu, B), tohum
        self.madde_kume = madde_kume  # küme = madde ise çoğaltılan maddeler ayrı grup sayılmalı
        sira = np.argsort(self.kume, kind="stable")
        self.sirali = sira
        self.uzunluk = np.bincount(self.kume, minlength=self.G)
        self.baslangic = np.concatenate([[0], np.cumsum(self.uzunluk)[:-1]])

    def secim(self, b: int) -> np.ndarray:
        return np.random.default_rng([self.tohum, b]).integers(0, self.G, self.G)

    def sayim(self, b: int) -> np.ndarray:
        return np.bincount(self.secim(b), minlength=self.G).astype(float)

    def satirlar(self, b: int):
        """b. tekrarın satır indeksleri ve (gerekirse) yeniden etiketlenmiş madde kodları."""
        sec = self.secim(b)
        L = self.uzunluk[sec]
        top = int(L.sum())
        kayma = np.repeat(self.baslangic[sec] - (np.cumsum(L) - L), L)
        idx = self.sirali[kayma + np.arange(top)]
        yeni = np.repeat(np.arange(len(sec)), L) if self.madde_kume else None
        return idx, yeni


def ga(degerler: np.ndarray, alfa: float = 0.05):
    v = np.asarray(degerler, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.nan, np.nan
    return float(np.quantile(v, alfa / 2)), float(np.quantile(v, 1 - alfa / 2))


def p_degeri(fark: np.ndarray) -> float:
    """İki yanlı bootstrap p: farkın 0'ın öbür yanına düşme payı × 2."""
    f = np.asarray(fark, float)
    f = f[np.isfinite(f)]
    if len(f) < 10:
        return np.nan
    p = 2 * min(np.mean(f <= 0), np.mean(f >= 0))
    return float(min(1.0, max(p, 1 / len(f))))


def hesapla(orn: Orneklem, girdiler: dict, metrikler: list, madde=None, agirlik=None):
    """girdiler: {sutun: (y, p, maske)}. maske None ya da bool dizi (eksik tavan vb.).

    Döner: nokta[sutun][metrik], tekrar[sutun][metrik] (np.ndarray, B ya da B_toplu uzunlukta).
    """
    w = np.ones(len(orn.kume)) if agirlik is None else np.asarray(agirlik, float)
    nokta = {s: {} for s in girdiler}
    tekrar = {s: {} for s in girdiler}

    # --- birim: küme toplamları
    birim = [m for m in metrikler if m.tur == "birim"]
    S, W, anahtar = [], [], []
    for s, (y, p, maske) in girdiler.items():
        ww = w if maske is None else w * maske
        for m in birim:
            if maske is None:
                v = m.fn(y, p)
            else:
                v = np.zeros(len(ww))
                v[maske] = m.fn(_al(y, maske), _al(p, maske))
            nokta[s][m.ad] = _son(m, np.sum(ww * v) / max(ww.sum(), 1e-12))
            S.append(np.bincount(orn.kume, ww * v, minlength=orn.G))
            W.append(np.bincount(orn.kume, ww, minlength=orn.G))
            anahtar.append((s, m))
    if anahtar:
        S, W = np.vstack(S).T, np.vstack(W).T  # G × K
        R = np.empty((orn.B, len(anahtar)))
        for b in range(orn.B):
            c = orn.sayim(b)
            R[b] = (c @ S) / np.maximum(c @ W, 1e-12)
        for k, (s, m) in enumerate(anahtar):
            tekrar[s][m.ad] = np.array([_son(m, v) for v in R[:, k]]) if m.son else R[:, k]

    # --- toplu: döngü
    toplu = [m for m in metrikler if m.tur == "toplu"]
    if toplu:
        for s, (y, p, maske) in girdiler.items():
            ym, pm = (y, p) if maske is None else (_al(y, maske), _al(p, maske))
            mm = None if madde is None else (madde if maske is None else madde[maske])
            for m in toplu:
                nokta[s][m.ad] = _guvenli(m, ym, pm, mm)
                tekrar[s][m.ad] = np.full(orn.B_toplu, np.nan)
        for b in range(orn.B_toplu):
            idx, yeni = orn.satirlar(b)
            for s, (y, p, maske) in girdiler.items():
                ii = idx if maske is None else idx[maske[idx]]
                if yeni is not None:
                    mm = yeni if maske is None else yeni[maske[idx]]
                else:
                    mm = None if madde is None else madde[ii]
                yy, pp = _al(y, ii), _al(p, ii)
                for m in toplu:
                    tekrar[s][m.ad][b] = _guvenli(m, yy, pp, mm)
    return nokta, tekrar


def _al(a, idx):
    return a[idx]


def _son(m, v):
    return float(m.son(v)) if m.son else float(v)


def _guvenli(m, y, p, madde):
    try:
        return m.deger(y, p, madde)
    except (ValueError, ZeroDivisionError, IndexError):
        return np.nan
