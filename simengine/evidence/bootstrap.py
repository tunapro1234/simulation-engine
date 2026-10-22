"""Küme bootstrap motoru. Bütün sütunlar ve metrikler AYNI yeniden örneklerle hesaplanır (eşleşik).

birim metrikler: küme toplamları × örnek sayıları (hızlı yol, B=1000).
toplu metrikler: her tekrar için satır indeksi açılır (döngü, B_toplu). Tek kümede yuvalı maddenin her bootstrap
kopyası ayrı madde kodu alır (Orneklem.satirlar): ab çifti kopyalar arasında kurulmaz.
"""
from __future__ import annotations

import numpy as np

HIZLI_SIRA_ESIK = 5000  # bu kadar satırdan büyük tabloda Spearman ve ikili toplu metrikler çokluk ağırlıklı (satır açmadan) yoldan


class Orneklem:
    def __init__(self, kume: np.ndarray, B: int = 1000, B_toplu: int = 200, tohum: int = 0,
                 madde_kume: bool = False, iki_yon=None):
        """iki_yon: (hücre→blok kodu, hücre→küme kodu); verilirse kume hücre kodudur ve her tekrarda blok ve küme
        sayımları bağımsız çekilip çarpılır (iki yönlü / çapraz bootstrap: blok ve seri bağımlılığı birlikte)."""
        self.kume = np.asarray(kume)
        self.G = int(self.kume.max()) + 1 if len(self.kume) else 0
        self.iki_yon = None
        if iki_yon is not None:
            hb, hc = (np.asarray(a, int) for a in iki_yon)
            self.iki_yon = (hb, hc, int(hb.max()) + 1, int(hc.max()) + 1)
        self.B, self.B_toplu, self.tohum = B, min(B_toplu, B), tohum
        self.madde_kume = madde_kume  # küme = madde ise çoğaltılan maddeler ayrı grup sayılmalı
        sira = np.argsort(self.kume, kind="stable")
        self.sirali = sira
        self.uzunluk = np.bincount(self.kume, minlength=self.G)
        self.baslangic = np.concatenate([[0], np.cumsum(self.uzunluk)[:-1]])

    def secim(self, b: int) -> np.ndarray:
        rng = np.random.default_rng([self.tohum, b])
        if self.iki_yon is None:
            return rng.integers(0, self.G, self.G)
        hb, hc, Gb, Gc = self.iki_yon
        cb = np.bincount(rng.integers(0, Gb, Gb), minlength=Gb)
        cc = np.bincount(rng.integers(0, Gc, Gc), minlength=Gc)
        return np.repeat(np.arange(self.G), cb[hb] * cc[hc])

    def sayim(self, b: int) -> np.ndarray:
        return np.bincount(self.secim(b), minlength=self.G).astype(float)

    def satirlar(self, b: int, madde=None, ic=None):
        """b. tekrarın satır indeksleri ve kopya farkında madde kodları.

        madde verilirse: tek kümede yuvalanmış maddenin (ic[madde] True) her kopyası ayrı madde olur
        (kod = (madde, o kümenin kaçıncı çekimi)); birden çok kümeye yayılan madde (ör. kişi kümesi × soru)
        ortak kalır. Böylece bir kez çekilen kümeler maddeyi paylaşır, k kez çekilen yuvalı küme k ağırlık
        alır (k² değil: A₁-B₂ gibi kopyalar arası çift kurulmaz). madde yoksa madde_kume=True eski davranış:
        her çekim ayrı kod."""
        sec = self.secim(b)
        L = self.uzunluk[sec]
        top = int(L.sum())
        kayma = np.repeat(self.baslangic[sec] - (np.cumsum(L) - L), L)
        idx = self.sirali[kayma + np.arange(top)]
        if madde is None:
            return idx, (np.repeat(np.arange(len(sec)), L) if self.madde_kume else None)
        madde = np.asarray(madde)
        if ic is None:
            ic = madde_yuvali(self.kume, madde)
        sira = np.argsort(sec, kind="stable")                       # çekimin kümesi içindeki sırası (0, 1, ...)
        s = sec[sira]
        bas = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        kac = np.empty(len(sec), int)
        kac[sira] = np.arange(len(sec)) - np.repeat(bas, np.diff(np.r_[bas, len(sec)]))
        m = madde[idx]
        if not kac.any():
            return idx, m
        kopya = np.repeat(kac, L) * ic[m]
        return idx, np.unique(m * (int(kopya.max()) + 1) + kopya, return_inverse=True)[1].reshape(-1)


def madde_yuvali(kume: np.ndarray, madde: np.ndarray) -> np.ndarray:
    """madde kodu → bütün satırları tek kümede mi (bootstrap kopyaları ayrı madde sayılır)."""
    madde, kume = np.asarray(madde, int), np.asarray(kume, int)
    if not len(madde):
        return np.zeros(0, bool)
    n = int(madde.max()) + 1
    ilk = np.full(n, -1)
    ilk[madde[::-1]] = kume[::-1]                                     # her maddenin ilk satırının kümesi
    return np.bincount(madde, kume != ilk[madde], n) == 0


def ga(degerler: np.ndarray, alfa: float = 0.05, nokta: float | None = None, yontem: str = "percentile"):
    """Bootstrap GA. yontem="percentile" (varsayılan) ya da "basic": (2θ − q_üst, 2θ − q_alt), θ = nokta tahmini.
    Toplu/oran metriklerinde yüzdelik GA nokta tahminini dışarıda bırakabilir; temel (basic) GA θ'ya göre yansıtır."""
    v = np.asarray(degerler, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.nan, np.nan
    a, u = float(np.quantile(v, alfa / 2)), float(np.quantile(v, 1 - alfa / 2))
    if yontem == "basic":
        if nokta is None or not np.isfinite(nokta):
            return np.nan, np.nan
        return float(2 * nokta - u), float(2 * nokta - a)
    if yontem != "percentile":
        raise ValueError(f"ci must be 'percentile' or 'basic', not {yontem!r}")
    return a, u


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

    # --- oran: pay ve payda küme toplamları, tekrar başına Σpay / Σpayda (ikisi birlikte yeniden örneklenir)
    oran = [m for m in metrikler if m.tur == "oran"]
    if oran:
        P, Q, anahtar_o = [], [], []
        for s, (y, p, maske) in girdiler.items():
            ww = w if maske is None else w * maske
            for m in oran:
                pay, payda = np.zeros(len(ww)), np.zeros(len(ww))
                if maske is None:
                    pay, payda = m.fn(y, p)
                else:
                    pay[maske], payda[maske] = m.fn(_al(y, maske), _al(p, maske))
                tp, tq = float(np.sum(ww * pay)), float(np.sum(ww * payda))
                nokta[s][m.ad] = _son(m, tp / tq) if abs(tq) > 1e-12 else np.nan
                P.append(np.bincount(orn.kume, ww * pay, minlength=orn.G))
                Q.append(np.bincount(orn.kume, ww * payda, minlength=orn.G))
                anahtar_o.append((s, m))
        P, Q = np.vstack(P).T, np.vstack(Q).T
        R = np.empty((orn.B, len(anahtar_o)))
        with np.errstate(invalid="ignore", divide="ignore"):
            for b in range(orn.B):
                c = orn.sayim(b)
                q = c @ Q
                R[b] = np.where(np.abs(q) > 1e-12, (c @ P) / np.where(np.abs(q) > 1e-12, q, 1.0), np.nan)
        for k, (s, m) in enumerate(anahtar_o):
            tekrar[s][m.ad] = m.son(R[:, k]) if m.son else R[:, k]

    # --- toplu: döngü
    toplu = [m for m in metrikler if m.tur == "toplu"]
    if toplu:
        for s, (y, p, maske) in girdiler.items():
            ym, pm = (y, p) if maske is None else (_al(y, maske), _al(p, maske))
            mm = None if madde is None else (madde if maske is None else madde[maske])
            for m in toplu:
                nokta[s][m.ad] = _guvenli(m, ym, pm, mm)
                tekrar[s][m.ad] = np.full(orn.B_toplu, np.nan)
        # büyük tabloda bazı toplu metrikler (Spearman; ikili AUC, macro-F1, kappa, dengeli doğruluk, ECE) satır
        # açmadan hesaplanır: yeniden örnek = satır çoklukları (küme sayımı), sıralar önceden kodlanmış değerlerden
        # birikimli toplamla (eşitlerde ortalama sıra). Sonuç açılmış örneklemdeki özgün metrikle aynı (kayan nokta).
        hizli = {}
        if len(orn.kume) > HIZLI_SIRA_ESIK:
            for s, (y, p, maske) in girdiler.items():
                kod = _sira_kodu(y, p, maske)
                if kod is None:                                   # eksik/sonsuz/vektör değerli sütun: özgün yol
                    continue
                for m in toplu:
                    f = _hizli_metrik(m, kod)
                    if f is not None:
                        hizli[(s, m.ad)] = f
        # hızlı yoldaki metrikler maddeyi kullanmaz; madde gruplu olanlar (cift_isaret, kazanan_top1, ...) aşağıda
        yavas = {s: [m for m in toplu if (s, m.ad) not in hizli] for s in girdiler}
        mk = None if madde is None else _madde_kod(madde)
        ic = None if mk is None else madde_yuvali(orn.kume, mk)
        for b in range(orn.B_toplu):
            if any(yavas.values()):
                idx, yeni = orn.satirlar(b, mk, ic)
                for s, (y, p, maske) in girdiler.items():
                    if not yavas[s]:
                        continue
                    ii = idx if maske is None else idx[maske[idx]]
                    mm = None if yeni is None else (yeni if maske is None else yeni[maske[idx]])
                    yy, pp = _al(y, ii), _al(p, ii)
                    for m in yavas[s]:
                        tekrar[s][m.ad][b] = _guvenli(m, yy, pp, mm)
            if hizli:
                w_tum = np.bincount(orn.secim(b), minlength=orn.G).astype(float)[orn.kume]
                for (s, ad), f in hizli.items():
                    try:
                        tekrar[s][ad][b] = f(w_tum)
                    except (ValueError, ZeroDivisionError, IndexError, FloatingPointError):
                        tekrar[s][ad][b] = np.nan
    return nokta, tekrar


def _madde_kod(madde) -> np.ndarray:
    """Madde etiketleri → 0..n−1 tamsayı kodları (sıra korunur)."""
    return np.unique(np.asarray(madde), return_inverse=True)[1].reshape(-1)


def _sira_kodu(y, p, maske):
    """Satır açmadan hesap için ön kodlama: satırlar, y/p değerleri ve artan değer sırasındaki kodları."""
    sat = np.arange(len(y)) if maske is None else np.flatnonzero(maske)
    try:
        y, p = np.asarray(y, float)[sat], np.asarray(p, float)[sat]
    except (TypeError, ValueError):
        return None
    if y.ndim != 1 or p.ndim != 1 or not (np.isfinite(y).all() and np.isfinite(p).all()):
        return None
    uy, ky = np.unique(y, return_inverse=True)
    up, kp = np.unique(p, return_inverse=True)
    return {"sat": sat, "y": y, "p": p, "ky": ky, "ny": len(uy), "kp": kp, "np": len(up),
            "ikili": bool(np.isin(uy, (0.0, 1.0)).all())}


def _hizli_metrik(m, k):
    """m için çokluk ağırlıklı hızlı hesap f(w_tum) ya da None (özgün yol)."""
    from . import metrik as M
    if m.fn is M._spearman:
        return lambda w_tum: _spearman_cokluk(k["sat"], k["ky"], k["ny"], k["kp"], k["np"], w_tum)
    ikili = {M._auc: _auc_cokluk, M._f1_ikili: _f1_cokluk, M._kappa_ikili: _kappa_cokluk,
             M._dengeli_ikili: _dengeli_cokluk, M._ece_ikili: _ece_cokluk}
    fn = ikili.get(m.fn)
    if fn is None or not k["ikili"]:
        return None
    if "kutu" not in k:
        k["yb"], k["pb"] = k["y"] > 0.5, k["p"] > 0.5
        k["hucre"] = 2 * k["yb"].astype(int) + k["pb"].astype(int)   # 0 TN, 1 FP, 2 FN, 3 TP
        k["kutu"] = np.clip((k["p"] * 10).astype(int), 0, 9)

    def f(w_tum):
        w = w_tum[k["sat"]]
        if w.sum() <= 0:
            return np.nan
        return fn(k, w, lambda: _guvenli(m, *_ac(k, w), None))
    return f


def _spearman_cokluk(sat, ky, ny, kp, np_, w_tum):
    """Satır çoklukları w ile açılmış örneklemin Spearman'ı (eşitlerde ortalama sıra)."""
    w = w_tum[sat]
    sw = w.sum()
    if sw < 2:
        return np.nan
    Wy, Wp = np.bincount(ky, w, ny), np.bincount(kp, w, np_)
    ry = (np.cumsum(Wy) - Wy + (Wy + 1) / 2)[ky]
    rp = (np.cumsum(Wp) - Wp + (Wp + 1) / 2)[kp]
    dy, dp = ry - (w @ ry) / sw, rp - (w @ rp) / sw
    vy, vp = w @ (dy * dy), w @ (dp * dp)
    if vy <= 1e-9 or vp <= 1e-9:  # sabit dizi (özgün: np.std == 0 → nan)
        return np.nan
    return float((w @ (dy * dp)) / np.sqrt(vy * vp))


def _ac(k, w):
    """Çoklukları satıra aç (kenar durumlarında özgün metrik)."""
    ii = np.repeat(np.arange(len(w)), w.astype(int))
    return k["y"][ii], k["p"][ii]


def _karisiklik(k, w):
    tn, fp, fn_, tp = np.bincount(k["hucre"], w, 4)
    return tn, fp, fn_, tp


def _tek_sinifli(tn, fp, fn_, tp):
    return min(tn + fp, fn_ + tp) <= 0 or min(tn + fn_, fp + tp) <= 0


def _dengeli_cokluk(k, w, ozgun):
    tn, fp, fn_, tp = _karisiklik(k, w)
    if _tek_sinifli(tn, fp, fn_, tp):
        return ozgun()
    return float((tn / (tn + fp) + tp / (tp + fn_)) / 2)


def _f1_cokluk(k, w, ozgun):
    tn, fp, fn_, tp = _karisiklik(k, w)
    if _tek_sinifli(tn, fp, fn_, tp):
        return ozgun()
    return float((2 * tp / (2 * tp + fp + fn_) + 2 * tn / (2 * tn + fp + fn_)) / 2)


def _kappa_cokluk(k, w, ozgun):
    tn, fp, fn_, tp = _karisiklik(k, w)
    if _tek_sinifli(tn, fp, fn_, tp):
        return ozgun()
    n = tn + fp + fn_ + tp
    gozlenen = (fp + fn_) / n
    beklenen = ((tn + fp) * (fp + tp) + (fn_ + tp) * (tn + fn_)) / n ** 2
    return float(1 - gozlenen / beklenen)


def _auc_cokluk(k, w, ozgun):
    w1 = float(w @ k["yb"])
    w0 = float(w.sum() - w1)
    if w1 <= 0 or w0 <= 0:
        return np.nan                                                 # özgün: tek sınıf → nan
    Wp = np.bincount(k["kp"], w, k["np"])
    r = (np.cumsum(Wp) - Wp + (Wp + 1) / 2)[k["kp"]]
    return float((w @ (r * k["yb"]) - w1 * (w1 + 1) / 2) / (w1 * w0))


def _ece_cokluk(k, w, ozgun):
    Wb = np.bincount(k["kutu"], w, 10)
    Yb = np.bincount(k["kutu"], w * k["y"], 10)
    Pb = np.bincount(k["kutu"], w * k["p"], 10)
    v = Wb > 0
    return float(np.sum(Wb[v] / Wb.sum() * np.abs(Yb[v] / Wb[v] - Pb[v] / Wb[v])))


def _al(a, idx):
    return a[idx]


def _son(m, v):
    return float(m.son(v)) if m.son else float(v)


def _guvenli(m, y, p, madde):
    try:
        return m.deger(y, p, madde)
    except (ValueError, ZeroDivisionError, IndexError):
        return np.nan
