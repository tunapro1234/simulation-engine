"""Metrik ailesi. Görev tipine göre otomatik seçilir.

İki tür metrik var:
  birim  satır başına kayıp/puan vektörü döner; metrik = ortalaması (+ isteğe bağlı dönüşüm).
         Bootstrap'ta küme toplamlarıyla hızlı hesaplanır.
  oran   (y, p) → (pay, payda) satır vektörleri; metrik = Σpay / Σpayda (WMAPE, yanlılık). Bootstrap'ta pay ve
         payda AYNI küme yeniden örnekleriyle birlikte toplanır (tekrar başına toplamların oranı).
  toplu  (y, p, madde) → tek sayı (AUC, macro-F1, korelasyon, ECE, ab kazananı). Bootstrap'ta döngü.
yon: +1 büyük iyi, -1 küçük iyi. tavan: test-retest tavanı bu metrikte anlamlı mı.

Dağılım görevinde Wasserstein sayısal şık konumlarını kullanır (konum/pos sütunu, sayısal secenekler/options ya da
sayısal classes=); konum yoksa ya da konumlar eşit aralıklıysa eşit aralık (0..K-1). Ör.:
    df["pos"] = [[0, .25, 1]] * len(df)           # şık başına konum
    evaluate(df, primary="wasserstein")           # = bench `wd` (konumlar 0-1'e ölçeklenir)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

EPS = 1e-6


@dataclass(frozen=True)
class Metrik:
    ad: str
    tur: str                   # 'birim' | 'toplu'
    fn: Callable
    yon: int                   # +1 büyük iyi, -1 küçük iyi
    tavan: bool = False        # tekrar ölçümle tavan hesaplanabilir mi
    son: Callable | None = None  # birim ortalamasına uygulanacak dönüşüm (ör. sqrt)
    aciklama: str = ""

    def deger(self, y, p, madde=None, w=None) -> float:
        if self.tur == "oran":
            pay, payda = self.fn(y, p)
            ww = np.ones(len(pay)) if w is None else np.asarray(w, float)
            alt = float(np.sum(ww * payda))
            v = float(np.sum(ww * pay)) / alt if abs(alt) > 1e-12 else np.nan
            return float(self.son(v)) if self.son else v
        if self.tur == "birim":
            v = float(np.average(self.fn(y, p), weights=w))
            return float(self.son(v)) if self.son else v
        return float(self.fn(y, p, madde))


# ------------------------------------------------------------------ diziler
KONUM_SUTUNLARI = ("konum", "pos", "positions")   # dağılım: satır başına şık konumları


class KonumluVektor(np.ndarray):
    """Dağılım vektörü + şık konumları (`konum`). Yalnız gerçek vektörleri taşır; konuma bakan metrik okur."""

    def __new__(cls, v, konum):
        o = np.asarray(v, float).view(cls)
        o.konum = np.asarray(konum, float)
        return o

    def __array_finalize__(self, obj):
        self.konum = getattr(obj, "konum", None)


def _sayisal(xs):
    try:
        x = np.array([float(v) for v in xs], float)
    except (TypeError, ValueError):
        return None
    return x if len(x) >= 2 and np.isfinite(x).all() else None


def _esit_aralik(x) -> bool:
    d = np.diff(np.sort(x))
    return bool(d[0] > 0 and np.allclose(d, d[0]))


def secenek_konumlari(veri) -> list | None:
    """Dağılım görevinde satır başına sayısal şık konumları (yoksa None). Sıra: konum/pos sütunu > sayısal
    'secenekler' sütunu > sayısal veri.siniflar. Eşit aralıklı konumlar None döner (eşit aralık zaten varsayılan)."""
    if veri.gorev != "dagilim":
        return None
    d = veri.df
    K = [len(v) for v in d["gercek"]]
    kaynak = [[_sayisal(v) if v is not None and not np.isscalar(v) else None for v in d[c]]
              for c in (*KONUM_SUTUNLARI, "secenekler") if c in d]
    if veri.siniflar:
        kaynak.append([_sayisal(veri.siniflar)] * len(d))
    for ham in kaynak:   # ilk sayısal kaynak kazanır
        ham = [x if x is not None and len(x) == k and np.ptp(x) > 0 else None for x, k in zip(ham, K)]
        if any(x is not None for x in ham):
            out = [None if x is None or _esit_aralik(x) else x for x in ham]
            return out if any(x is not None for x in out) else None
    return None


def diziler(veri, sutun: str):
    """Veri.df'deki gercek ve verilen sütunu metrik fonksiyonlarının beklediği biçime çevir. Dağılımda sayısal ve
    eşit aralıklı olmayan şık konumları gerçek vektörlerine `KonumluVektor` olarak iliştirilir."""
    d = veri.df
    g, s = d["gercek"], d[sutun]
    if veri.gorev == "cok_sinif":
        kod = {k: i for i, k in enumerate(veri.siniflar)}
        y = g.map(lambda v: kod.get(v, kod.get(str(v), -1))).to_numpy()
        if isinstance(s.iloc[0], np.ndarray):
            return y, np.vstack(s.to_numpy())
        return y, s.map(lambda v: kod.get(v, kod.get(str(v), -1))).to_numpy()
    if veri.gorev == "dagilim":
        y = g.to_numpy(object)
        kon = secenek_konumlari(veri)
        if kon is not None:
            y = y.copy()
            for i, k in enumerate(kon):
                if k is not None:
                    y[i] = KonumluVektor(y[i], k)
        return y, s.to_numpy(object)
    return g.to_numpy(float), s.to_numpy(float)


# ------------------------------------------------------------------ ikili
def _acc_ikili(y, p):
    return np.where(np.isclose(p, 0.5), 0.5, ((p > 0.5) == (y > 0.5)).astype(float))


def _brier_ikili(y, p):
    return (p - y) ** 2


def _logloss_ikili(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _auc(y, p, madde=None):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def _f1_ikili(y, p, madde=None):
    return f1_score(y > 0.5, p > 0.5, average="macro", zero_division=0)


def _kappa_ikili(y, p, madde=None):
    return cohen_kappa_score(y > 0.5, p > 0.5)


def _dengeli_ikili(y, p, madde=None):
    return balanced_accuracy_score(y > 0.5, p > 0.5)


def ece(guven, dogru, kutu: int = 10) -> float:
    """Eşit genişlikli kutularla beklenen kalibrasyon hatası (üst-etiket)."""
    guven, dogru = np.asarray(guven, float), np.asarray(dogru, float)
    k = np.clip((guven * kutu).astype(int), 0, kutu - 1)
    n = len(guven)
    top = 0.0
    for b in range(kutu):
        m = k == b
        if m.any():
            top += m.sum() / n * abs(dogru[m].mean() - guven[m].mean())
    return top


def _ortalama_sapma(y, p):
    """Satır başına p − y; ortalaması = ort. tahmin − ort. gerçek (calibration-in-the-large, işaretli; 0 iyi)."""
    return np.asarray(p, float) - np.asarray(y, float)


def _ece_ikili(y, p, madde=None):
    # ikili için olasılık kalibrasyonu: p'nin kendisi (pozitif sınıf) kutulanır
    return ece(p, y)


# ------------------------------------------------------------------ çok sınıflı
def _olasilik_mi(p):
    return p.ndim == 2


def _acc_cs(y, p):
    if _olasilik_mi(p):
        # beraberlikte payı böl
        m = p.max(1, keepdims=True)
        ust = np.isclose(p, m)
        return ust[np.arange(len(y)), y] / ust.sum(1)
    return (p == y).astype(float)


def _brier_cs(y, p):
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1
    return ((p - oh) ** 2).sum(1)


def _logloss_cs(y, p):
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))


def _etiket(p):
    return p.argmax(1) if _olasilik_mi(p) else p


def _sayimlar(y, p):
    """Sert etiketlerin sınıf sayımları, yalnız görülen etiketler üzerinde (np.unique + ters kodlar): (gerçek_n, tahmin_n,
    doğru_n), her biri L uzunlukta, L = y ∪ tahmin etiket sayısı. K × K karışıklık matrisi kurulmaz: bootstrap
    tekrarı başına O(n log n), sınıf sayısından bağımsız (D9 #3: 800 sınıfta süre sklearn confusion_matrix'teydi)."""
    y, t = np.asarray(y).reshape(-1), np.asarray(_etiket(p)).reshape(-1)
    n = len(y)
    _, kod = np.unique(np.concatenate([y, t]), return_inverse=True)
    kod = kod.reshape(-1)
    ky, kt = kod[:n], kod[n:]
    L = int(kod.max()) + 1 if len(kod) else 0
    return (np.bincount(ky, minlength=L).astype(float), np.bincount(kt, minlength=L).astype(float),
            np.bincount(ky[ky == kt], minlength=L).astype(float))


def _f1_cs(y, p, madde=None):
    """Macro-F1 (sklearn average="macro", zero_division=0): y ∪ tahmin etiketleri üzerinde 2·TP / (2·TP + FP + FN)."""
    gy, gt, tp = _sayimlar(y, p)
    if not len(gy):
        return 0.0
    payda = gy + gt                                   # 2·TP + FP + FN
    return float(np.mean(np.where(payda > 0, 2 * tp / np.where(payda > 0, payda, 1), 0.0)))


def _kappa_cs(y, p, madde=None):
    """Cohen kappa (sklearn cohen_kappa_score ile aynı): 1 − (1 − p_o) / (1 − p_e); p_e = 1 ise tanımsız (nan)."""
    gy, gt, tp = _sayimlar(y, p)
    n = gy.sum()
    if n <= 0:
        return np.nan
    uyumsuz = 1 - tp.sum() / n
    beklenen = 1 - (gy @ gt) / n ** 2
    return float(1 - uyumsuz / beklenen) if beklenen > 0 else np.nan


def _dengeli_cs(y, p, madde=None):
    """Dengeli doğruluk (sklearn balanced_accuracy_score): gerçekte görülen sınıfların duyarlılık ortalaması."""
    gy, _, tp = _sayimlar(y, p)
    v = gy > 0
    return float(np.mean(tp[v] / gy[v])) if v.any() else np.nan


def _ece_cs(y, p, madde=None):
    return ece(p.max(1), (p.argmax(1) == y).astype(float))


# ------------------------------------------------------------------ dağılım (madde başına)
def _per(fn):
    def sar(y, p):
        return np.array([fn(a, b) for a, b in zip(y, p)], dtype=float)
    return sar


def tvd(a, b):
    return 0.5 * np.abs(a - b).sum()


def _simbench(y, p, madde=None):
    """SimBench skoru: 100 · (1 − ort. TVD(gerçek, tahmin) / ort. TVD(gerçek, tekdüze)). 0 = tekdüze, 100 = kusursuz."""
    t = np.mean([tvd(a, b) for a, b in zip(y, p)])
    u = np.mean([tvd(a, np.full(len(a), 1 / len(a))) for a in y])
    return float(100 * (1 - t / u)) if u > 0 else np.nan


def wasserstein_sirali(a, b):
    """Sıralı seçenekler (0..K-1) arası EMD, K-1'e bölünür (0-1 aralığı)."""
    k = len(a)
    return np.abs(np.cumsum(a - b)).sum() / max(k - 1, 1)


def wasserstein_konumlu(a, b, konum=None) -> float:
    """Şık konumlarında Wasserstein-1, konum aralığıyla 0-1'e ölçekli (konumlar 0-1'i kaplıyorsa bench `wd`).
    konum None ise a'nın ya da b'nin `konum` niteliği; o da yoksa eşit aralık (wasserstein_sirali)."""
    if konum is None:
        konum = getattr(a, "konum", None)
    if konum is None:
        konum = getattr(b, "konum", None)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if konum is None or len(konum) != len(a) or np.ptp(konum) <= 0:
        return float(wasserstein_sirali(a, b))
    x = np.asarray(konum, float)
    o = np.argsort(x, kind="stable")
    x = x[o]
    fark = np.cumsum(a[o] - b[o])[:-1]
    return float(np.abs(fark) @ np.diff(x) / (x[-1] - x[0]))


def js(a, b):
    m = 0.5 * (a + b)

    def kl(x, y):
        x = np.clip(x, 1e-12, 1)
        return np.sum(np.where(a * 0 + x > 1e-12, x * np.log2(x / np.clip(y, 1e-12, 1)), 0))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def ks(a, b):
    return np.abs(np.cumsum(a) - np.cumsum(b)).max()


def _mod(a, b):
    m = np.isclose(b, b.max())
    return float(m[np.argmax(a)]) / m.sum()


# ------------------------------------------------------------------ sürekli
def _pearson(y, p, madde=None):
    if np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return stats.pearsonr(y, p)[0]


def _spearman(y, p, madde=None):
    if np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return stats.spearmanr(y, p)[0]


def _r2(y, p, madde=None):
    return 1 - np.sum((y - p) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)


def _madde_ici_kor(y, p, madde):
    """Madde içinde birimler arası Pearson, Fisher-z ortalaması (kişiyi ayırt etme; Twin 'corr')."""
    if madde is None:
        return np.nan
    z = []
    for g in _gruplar(madde):
        if len(g) >= 3 and np.std(y[g]) > 0 and np.std(p[g]) > 0:
            z.append(np.arctanh(np.clip(np.corrcoef(y[g], p[g])[0, 1], -0.999999, 0.999999)))
    return float(np.tanh(np.mean(z))) if z else np.nan


def _demeaned_r2(y, p, madde):
    """Madde ortalamaları çıkarıldıktan sonra R²: kişiye özgü sinyal (madde ortalaması baseline'ı 0 alır)."""
    if madde is None:
        return np.nan
    kod, _ = pd.factorize(madde)
    say = np.bincount(kod)
    ry = y - (np.bincount(kod, y) / say)[kod]
    rp = p - (np.bincount(kod, p) / say)[kod]
    return 1 - np.sum((ry - rp) ** 2) / max(np.sum(ry ** 2), 1e-12)


# ------------------------------------------------------------------ oran (talep tahmini)
def _wmape(y, p):
    """WMAPE / WAPE = Σ|y − p| / Σ|y| (küçük iyi)."""
    return np.abs(y - p), np.abs(y)


def _yanlilik(y, p):
    """Yanlılık = Σp / Σy − 1 (işaretli; 0 iyi, > 0 fazla tahmin)."""
    return p, y


WMAPE = Metrik("wmape", "oran", _wmape, -1, aciklama="Σ|hata| / Σ|gerçek| (WAPE); küme bootstrap'ında pay ve payda birlikte")
YANLILIK = Metrik("yanlilik", "oran", _yanlilik, -1, son=lambda v: v - 1,
                  aciklama="Σtahmin / Σgerçek − 1, işaretli (0 iyi); tanı satırı, birincil metrik olamaz")
ORAN_METRIKLERI = ("wmape", "yanlilik")
# calibration-in-the-large: kol başına ort. tahmin − ort. gerçek (ikili: tahmini oran − taban oranı), küme GA'sıyla.
# Ad İngilizce: compare/paired_ci(metric="mean_bias") doğrudan bulsun.
ORT_SAPMA = Metrik("mean_bias", "birim", _ortalama_sapma, -1,
                   aciklama="ort. tahmin − ort. gerçek (calibration-in-the-large), işaretli (0 iyi); tanı satırı, "
                            "birincil metrik olamaz")
# işaretli metrikler: yönü yok (0 iyi); "daha iyi" = |değer| daha küçük. Metrik.yon iç hesap için -1 kalır,
# rapor tablosunda yon=0 yazılır.
ISARETLI_METRIKLER = ("yanlilik", "mean_bias")


# ------------------------------------------------------------------ ab (madde = test)
def _gruplar(madde):
    kod, _ = pd.factorize(madde)
    sira = np.argsort(kod, kind="stable")
    sinir = np.flatnonzero(np.diff(kod[sira])) + 1
    return [g for g in np.split(sira, sinir) if len(g) >= 2]


def _top1(y, p, madde):
    puan = []
    for g in _gruplar(madde):
        ust = np.isclose(p[g], p[g].max())
        gercek_ust = np.isclose(y[g], y[g].max())
        puan.append((ust & gercek_ust).sum() / ust.sum())
    return float(np.mean(puan)) if puan else np.nan


def _cift_isaret(y, p, madde):
    dogru, n = 0.0, 0
    for g in _gruplar(madde):
        i, j = np.triu_indices(len(g), 1)
        dy = np.sign(y[g][i] - y[g][j])
        dp = np.sign(p[g][i] - p[g][j])
        m = dy != 0
        dogru += np.where(dp[m] == 0, 0.5, (dp[m] == dy[m]).astype(float)).sum()
        n += m.sum()
    return dogru / n if n else np.nan


def _spearman_ici(y, p, madde):
    r = []
    for g in _gruplar(madde):
        if len(g) >= 3 and np.std(y[g]) > 0 and np.std(p[g]) > 0:
            r.append(stats.spearmanr(y[g], p[g])[0])
    return float(np.mean(r)) if r else np.nan


def _etki_kor(y, p, madde):
    """Test içi ortalamadan farkların korelasyonu (etki büyüklüğü uyumu)."""
    dy, dp = [], []
    for g in _gruplar(madde):
        dy.append(y[g] - y[g].mean())
        dp.append(p[g] - p[g].mean())
    if not dy:
        return np.nan
    dy, dp = np.concatenate(dy), np.concatenate(dp)
    if np.std(dp) == 0 or np.std(dy) == 0:
        return np.nan
    return stats.pearsonr(dy, dp)[0]


def _olcek_sapmasi(y, p, madde):
    """|log2(sd(tahmin farkı) / sd(gerçek fark))|, farklar test ortalamasından: 0 = etki ölçeği aynı, 1 = 2 kat sapma.
    Tahmin gerçekle aynı birimdeyse (ör. CTR) anlamlı; olasılık/puan ölçeğinde yalnız göreli okunur."""
    dy, dp = [], []
    for g in _gruplar(madde):
        dy.append(y[g] - y[g].mean())
        dp.append(p[g] - p[g].mean())
    if not dy:
        return np.nan
    sy, sp = np.std(np.concatenate(dy)), np.std(np.concatenate(dp))
    return float(abs(np.log2(sp / sy))) if sy > 0 and sp > 0 else np.nan


# ------------------------------------------------------------------ kayıt
MADDE_ICI = Metrik("madde_ici_kor", "toplu", _madde_ici_kor, +1, tavan=True,
                   aciklama="madde içinde kişiler arası korelasyon (Fisher-z ort.): kişiyi ayırt ediyor mu")
DEMEANED = Metrik("demeaned_r2", "toplu", _demeaned_r2, +1, tavan=True,
                  aciklama="madde ortalaması çıkarılınca R²: kişiye özgü sinyal (madde ortalaması = 0)")


def aile(gorev: str, olasilik: bool = True, madde: bool = False, birim_aralik: bool = False,
         oran: bool = False) -> list[Metrik]:
    """oran=True: sürekli görevde WMAPE ve yanlılık da eklenir (talep tahmini)."""
    ek = [MADDE_ICI, DEMEANED] if madde else []
    if oran and gorev == "surekli":
        ek = ek + [WMAPE, YANLILIK]
    if gorev == "ikili":
        return [
            Metrik("dogruluk", "birim", _acc_ikili, +1, tavan=True, aciklama="eşik 0,5; tam 0,5 yarım puan"),
            Metrik("dengeli_dogruluk", "toplu", _dengeli_ikili, +1, tavan=True, aciklama="sınıf dengesizliğine dayanıklı"),
            Metrik("kappa", "toplu", _kappa_ikili, +1, tavan=True, aciklama="Cohen kappa: şans düzeltmeli uyum"),
            Metrik("auc", "toplu", _auc, +1, aciklama="sıralama kalitesi"),
            Metrik("macro_f1", "toplu", _f1_ikili, +1, tavan=True),
            Metrik("brier", "birim", _brier_ikili, -1, aciklama="olasılık hatası (uygun skor)"),
            Metrik("logloss", "birim", _logloss_ikili, -1, aciklama="olasılık hatası (uygun skor)"),
            Metrik("ece", "toplu", _ece_ikili, -1, aciklama="10 kutu, kalibrasyon hatası"),
            ORT_SAPMA,
        ] + ek
    if gorev == "cok_sinif":
        temel = [
            Metrik("dogruluk", "birim", _acc_cs, +1, tavan=True, aciklama="argmax; beraberlik paylaştırılır"),
            Metrik("dengeli_dogruluk", "toplu", _dengeli_cs, +1, tavan=True),
            Metrik("kappa", "toplu", _kappa_cs, +1, tavan=True, aciklama="Cohen kappa: şans düzeltmeli uyum"),
            Metrik("macro_f1", "toplu", _f1_cs, +1, tavan=True),
        ]
        if not olasilik:
            return temel
        return temel + [
            Metrik("brier", "birim", _brier_cs, -1),
            Metrik("logloss", "birim", _logloss_cs, -1),
            Metrik("ece", "toplu", _ece_cs, -1, aciklama="üst-etiket güveni, 10 kutu"),
        ]
    if gorev == "dagilim":
        return [
            Metrik("bir_eksi_tvd", "birim", _per(lambda a, b: 1 - tvd(a, b)), +1, tavan=True,
                   aciklama="1 − toplam varyasyon uzaklığı (dağılım benchmark'larında TVD; Meister ve ark.)"),
            Metrik("simbench", "toplu", _simbench, +1, tavan=True,
                   aciklama="SimBench skoru: 100·(1 − ort. TVD / tekdüze ort. TVD); tekdüze tahmin 0 alır"),
            Metrik("wasserstein", "birim", _per(wasserstein_konumlu), -1, tavan=True,
                   aciklama="sıralı seçenekler, 0-1 ölçek: sayısal şık konumları varsa (konum/pos sütunu, sayısal "
                            "options ya da classes) konumlarda (bench wd), yoksa eşit aralık 0..K−1 (OpinionQA)"),
            Metrik("js", "birim", _per(js), -1, tavan=True, aciklama="Jensen-Shannon ıraksaması, log2"),
            Metrik("ks", "birim", _per(ks), -1, tavan=True, aciklama="kümülatif dağılım en büyük farkı"),
            Metrik("mod_eslesme", "birim", _per(_mod), +1, tavan=True, aciklama="en sık seçenek aynı mı"),
        ]
    if gorev == "surekli":
        bir = [Metrik("bir_eksi_mae", "birim", lambda y, p: 1 - np.abs(y - p), +1, tavan=True,
                      aciklama="0-1 ölçekte hücre doğruluğu (Twin-2K-500 resmi ölçüsü)")] if birim_aralik else []
        return bir + [
            Metrik("pearson", "toplu", _pearson, +1, tavan=True),
            Metrik("spearman", "toplu", _spearman, +1, tavan=True),
            Metrik("mae", "birim", lambda y, p: np.abs(y - p), -1, tavan=True),
            Metrik("rmse", "birim", lambda y, p: (y - p) ** 2, -1, tavan=True, son=np.sqrt),
            Metrik("r2", "toplu", _r2, +1, tavan=True),
            ORT_SAPMA,
        ] + ek
    if gorev == "ab":
        return [
            Metrik("kazanan_top1", "toplu", _top1, +1, tavan=True, aciklama="testin gerçek kazananı tahminde de ilk mi"),
            Metrik("cift_isaret", "toplu", _cift_isaret, +1, tavan=True, aciklama="varyant çiftlerinde fark yönü doğruluğu"),
            Metrik("spearman_ici", "toplu", _spearman_ici, +1, tavan=True, aciklama="test içi sıralama (≥3 varyant)"),
            Metrik("etki_korelasyonu", "toplu", _etki_kor, +1, tavan=True, aciklama="test ortalamasından farkların korelasyonu"),
            Metrik("olcek_sapmasi", "toplu", _olcek_sapmasi, -1, tavan=True,
                   aciklama="|log2(sd tahmin farkı / sd gerçek fark)|: 0 = aynı etki ölçeği; tahmin gerçekle aynı birimdeyse"),
        ]
    raise ValueError(gorev)


BIRINCIL = {"ikili": "dogruluk", "cok_sinif": "dogruluk", "dagilim": "bir_eksi_tvd",
            "surekli": "pearson", "ab": "cift_isaret"}
