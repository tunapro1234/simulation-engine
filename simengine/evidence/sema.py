"""Girdi şeması: uzun tablo → doğrulanmış, görev tipi belirlenmiş Veri nesnesi.

Satır = bir birim (kişi, kişi×soru, segment×soru, varyant). Sütunlar:
  gercek            zorunlu. Etiket (0/1, sınıf adı), sayı ya da dağılım (liste/dizi/JSON).
  tahmin_<yontem>   en az bir tane. Olasılık (ikili), etiket ya da olasılık vektörü
                    (çok sınıflı), sayı (sürekli), dağılım (dağılım görevi), skor (ab).
  segment, seg_*    isteğe bağlı. Segment hata analizi.
  tekrar            isteğe bağlı. Aynı birimin ikinci gerçek ölçümü (test-retest tavanı).
  grup              isteğe bağlı. Küme/bölme anahtarı (kişi). Bootstrap ve çapraz uydurma bunu
                    birlikte örnekler; aynı kişinin satırları bölmenin iki yanına düşmez.
  madde             isteğe bağlı. Soru/test kimliği. Madde başına marjinal baseline, ab
                    görevinde varyantların yarıştığı test, yayılım analizi.
  bolum             isteğe bağlı. 'egitim'/'test'. Varsa metrikler yalnız test satırlarında;
                    eğitim satırları sızıntı kontrolüne ve baseline uydurmaya girer.
  zaman             isteğe bağlı. Zaman sızıntısı kontrolü.
  birim             isteğe bağlı. Satır kimliği (tekrar kontrolü).

Bootstrap kümesi varsayılan grup > madde > satır; çerçevenin `attrs["kume_sutun"]` değeri (api.evaluate(cluster=))
bunu ezer: o sütunun her değeri bir küme (ör. aynı madde 18 ülkede tekrarlanıyorsa cluster="item", G3 #13).
Grup çapraz uydurma / bölme anahtarı olarak kalır; yalnız bootstrap kümesi değişir.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

GOREVLER = ("ikili", "cok_sinif", "dagilim", "surekli", "ab")
ON_EK = "tahmin_"
TEST_ADLARI = ["test", "deger", "degerlendirme", "holdout"]
GOREVLER_EN = ("binary", "multiclass", "distribution", "continuous", "ab")   # GOREVLER ile aynı sıra


def _ingilizce(sutunlar: list[str]) -> list[str]:
    """Hata mesajı için: tahmin_<ad> → pred_<ad> (EvalFrame yazımı)."""
    return ["pred_" + c[len(ON_EK):] if c.startswith(ON_EK) else c for c in sutunlar]


def _vektor(x):
    """Liste, dizi, JSON metni ya da {sinif: olasilik} sözlüğü → np.ndarray / dict."""
    if isinstance(x, str):
        x = json.loads(x)
    if isinstance(x, dict):
        return x
    return np.asarray(x, dtype=float)


def _vektor_mu(seri: pd.Series) -> bool:
    ornek = seri.dropna()
    if ornek.empty:
        return False
    v = ornek.iloc[0]
    if isinstance(v, (list, tuple, np.ndarray, dict)):
        return True
    if isinstance(v, str) and v.lstrip().startswith(("[", "{")):
        return True
    return False


@dataclass
class Veri:
    df: pd.DataFrame                      # yalnız değerlendirme satırları (bolum=test ya da hepsi)
    gorev: str
    yontemler: list[str]                  # tahmin_ ön eki atılmış adlar, sütun sırasıyla
    siniflar: list | None = None          # cok_sinif / ikili için sınıf listesi
    segmentler: list[str] = field(default_factory=list)
    egitim: pd.DataFrame | None = None    # bolum=egitim satırları (varsa)
    tam: pd.DataFrame | None = None       # ham girdi (sızıntı kontrolü için)
    notlar: list[str] = field(default_factory=list)
    blok: str | None = None               # blok bootstrap sütunu (köken/hafta): verilirse bloklar × kümeler iki yönlü örneklenir
    kume_sutun: str | None = None         # bootstrap küme sütunu (cluster=): grup > madde > satır sırasını ezer

    @property
    def taban_kume(self) -> np.ndarray:
        """Blok dışı bootstrap kümesi: kume_sutun (cluster=) ya da grup > madde > satır."""
        if self.kume_sutun is not None and self.kume_sutun in self.df:
            return pd.factorize(self.df[self.kume_sutun])[0]
        for s in ("grup", "madde"):
            if s in self.df:
                return pd.factorize(self.df[s])[0]
        return np.arange(len(self.df))

    @property
    def taban_kume_adi(self) -> str:
        if self.kume_sutun is not None and self.kume_sutun in self.df:
            return self.kume_sutun
        for s in ("grup", "madde"):
            if s in self.df:
                return s
        return "satir"

    @property
    def kume(self) -> np.ndarray:
        """Bootstrap kümesi kodları. Blok verildiyse hücre (blok × küme) kodları: iki yönlü yeniden örnekleme
        (bkz. iki_yon); yoksa grup > madde > satır."""
        if self.blok is not None:
            return pd.factorize(pd.MultiIndex.from_arrays([pd.factorize(self.df[self.blok])[0], self.taban_kume]))[0]
        return self.taban_kume

    @property
    def iki_yon(self):
        """Blok verildiyse (hücre başına blok kodu, hücre başına küme kodu); Orneklem bunlarla blok ve küme
        ağırlıklarını bağımsız çekip çarpar (iki yönlü bootstrap). Blok yoksa None."""
        if self.blok is None:
            return None
        bk, ck, hucre = pd.factorize(self.df[self.blok])[0], self.taban_kume, self.kume
        G = int(hucre.max()) + 1
        hb, hc = np.zeros(G, int), np.zeros(G, int)
        hb[hucre], hc[hucre] = bk, ck
        return hb, hc

    @property
    def n_kume(self) -> int:
        """Kartta gösterilen küme sayısı (blokta da blok dışı kümeler; blok sayısı ayrıca)."""
        return int(self.taban_kume.max()) + 1 if len(self.df) else 0

    @property
    def kume_adi(self) -> str:
        return self.taban_kume_adi

    def tahmin(self, yontem: str):
        return self.df[ON_EK + yontem]

    def var(self, sutun: str) -> bool:
        return sutun in self.df.columns


def gorev_bul(df: pd.DataFrame, yontem_sutunlari: list[str]) -> str:
    g = df["gercek"]
    if _vektor_mu(g):
        return "dagilim"
    if g.dtype == object or str(g.dtype) in ("category", "string", "bool"):
        return "ikili" if g.nunique() == 2 and set(map(str, g.unique())) <= {"0", "1", "True", "False"} else "cok_sinif"
    vals = pd.unique(g.dropna())
    if set(np.asarray(vals, dtype=float).tolist()) <= {0.0, 1.0}:
        return "ikili"
    if len(vals) <= 12 and np.allclose(vals, np.round(vals)):
        # az sayıda tam sayı: tahminler vektör ya da aynı kümeden etiketse sınıflandırma
        t = df[yontem_sutunlari[0]]
        if _vektor_mu(t) or set(pd.unique(t.dropna())) <= set(vals):
            return "cok_sinif"
    return "surekli"


def hazirla(df: pd.DataFrame, gorev: str | None = None, yontemler: list[str] | None = None,
            siniflar: list | None = None) -> Veri:
    """Girdiyi doğrula, görev tipini bul, vektör sütunlarını çöz. df.attrs["kume_sutun"]: bootstrap küme sütunu."""
    if "gercek" not in df.columns:
        raise ValueError("'truth' column is required (alias: gercek)")
    kume_sutun = df.attrs.get("kume_sutun")
    if kume_sutun is not None and kume_sutun not in df.columns:
        raise ValueError(f"cluster column {kume_sutun!r} is not a column of the frame")
    tsut = [c for c in df.columns if c.startswith(ON_EK)]
    if yontemler:
        tsut = [ON_EK + y for y in yontemler]
        eksik = [c for c in tsut if c not in df.columns]
        if eksik:
            raise ValueError(f"missing prediction columns: {_ingilizce(eksik)} (alias: {eksik})")
    if not tsut:
        raise ValueError("at least one 'pred_<method>' column is required (alias: tahmin_<method>)")
    notlar = []
    tam = df.reset_index(drop=True).copy()
    if "bolum" in tam:
        test_maske = tam["bolum"].astype(str).str.lower().isin(TEST_ADLARI)
        egitim = tam[~test_maske].reset_index(drop=True)
        d = tam[test_maske].reset_index(drop=True)
        if len(d) == 0:
            raise ValueError(_test_yok_mesaji(tam))
        notlar.append(f"bolum sütunu: {len(d)} test, {len(egitim)} eğitim satırı; metrikler yalnız testte")
    else:
        egitim, d = None, tam
    # gerçeği eksik satırlar (henüz açılmamış ya da yanıtsız) değerlendirilemez: sayısıyla düşürülür
    eks = _eksik(d["gercek"])
    if eks.any():
        if eks.all():
            raise ValueError("every 'truth' (alias: gercek) value is missing: no truth yet (evaluate returns an "
                             "'awaiting truth' report in this case)")
        notlar.append(f"gerçeği eksik satırlar düşürüldü: {int(eks.sum())} / {len(d)}")
        d = d[~eks].reset_index(drop=True)
    if egitim is not None and len(egitim):
        egitim = egitim[~_eksik(egitim["gercek"])].reset_index(drop=True)
    gorev = gorev or gorev_bul(d, tsut)
    if gorev not in GOREVLER:
        raise ValueError(f"unknown task {gorev!r}: use one of {GOREVLER_EN} (internal: {GOREVLER})")
    if gorev == "ab" and "madde" not in d:
        raise ValueError("task 'ab' needs an 'item' column, the test id (alias: madde)")

    d = d.copy()
    if gorev == "dagilim":
        for c in ["gercek", *tsut, *(["tekrar"] if "tekrar" in d else [])]:
            d[c] = [_normalle(_vektor(v)) if v is not None else None for v in d[c]]
    elif gorev == "cok_sinif":
        if not siniflar:  # sınıf kümesi = gercek ∪ etiket tahminleri (açık uçlu eylemde tahmin gerçekte olmayabilir)
            ek = [v for c in tsut if not _vektor_mu(d[c]) for v in pd.unique(d[c].dropna())]
            siniflar = sorted(set(pd.unique(d["gercek"].dropna()).tolist()) | set(ek), key=str)
        for c in tsut:
            if _vektor_mu(d[c]):
                d[c] = [_sinif_vektoru(_vektor(v), siniflar) for v in d[c]]
    elif gorev == "ikili":
        d["gercek"] = d["gercek"].map(lambda v: 1.0 if str(v) in ("1", "1.0", "True") else 0.0)
        if "tekrar" in d:
            d["tekrar"] = d["tekrar"].map(lambda v: np.nan if pd.isna(v) else (1.0 if str(v) in ("1", "1.0", "True") else 0.0))
        siniflar = [0, 1]
        for c in tsut:
            d[c] = pd.to_numeric(d[c], errors="coerce").clip(0, 1)
    else:
        for c in ["gercek", *tsut]:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    nan = {c: int(pd.isna(d[c]).sum()) if gorev not in ("dagilim", "cok_sinif") else
           int(sum(v is None or (np.isscalar(v) and pd.isna(v)) for v in d[c])) for c in tsut}
    eksikli = {c: n for c, n in nan.items() if n}
    if eksikli:
        notlar.append(f"eksik tahminler (satır düşürüldü): {eksikli}")
        keep = np.ones(len(d), bool)
        for c in eksikli:
            keep &= ~np.array([v is None or (np.isscalar(v) and pd.isna(v)) for v in d[c]])
        d = d[keep].reset_index(drop=True)

    segs = [c for c in d.columns if c == "segment" or c.startswith("seg_")]
    return Veri(df=d, gorev=gorev, yontemler=[c[len(ON_EK):] for c in tsut], siniflar=siniflar,
                segmentler=segs, egitim=egitim, tam=tam, notlar=notlar, kume_sutun=kume_sutun)


def _eksik(seri: pd.Series) -> np.ndarray:
    """None / NaN / boş metin hücreleri (vektör hücreleri eksik sayılmaz)."""
    return np.array([v is None or (isinstance(v, str) and not v.strip()) or (np.isscalar(v) and pd.isna(v))
                     for v in seri], dtype=bool)


def gercek_yok(df: pd.DataFrame) -> bool:
    """Gerçek sütunu yok ya da (bolum varsa test satırlarında) tamamen eksik mi: mühürlü, gerçeği bekleyen tahmin."""
    if "gercek" not in df.columns or len(df) == 0:
        return True
    g = df["gercek"]
    if bool(_eksik(g).all()):
        # sütun var ama tamamı boş: çoğu zaman başarısız bir birleştirme (ör. 'unit' üzerinden), sessiz 'bekliyor' değil
        warnings.warn(f"the 'truth' (alias: gercek) column is present but all {len(g)} values are missing (NaN/None/"
                      "empty): reporting 'awaiting truth'. If truth was expected, check the join that filled it (e.g. "
                      "on 'unit'); to evaluate sealed predictions before truth, drop the column instead.",
                      UserWarning, stacklevel=3)
        return True
    if "bolum" in df.columns:
        g = g[df["bolum"].astype(str).str.lower().isin(TEST_ADLARI).to_numpy()]
        if len(g) == 0:
            raise ValueError(_test_yok_mesaji(df))
    return bool(_eksik(g).all())


def _test_yok_mesaji(df: pd.DataFrame) -> str:
    vals = sorted({str(v) for v in df["bolum"].dropna().unique()})[:8]
    return (f"no test rows in split column 'bolum' (values: {vals}; accepted test names: {TEST_ADLARI}); "
            "rename the evaluation rows or drop the column")


def _normalle(v):
    v = np.clip(np.asarray(v, dtype=float), 0, None)
    s = v.sum()
    return v / s if s > 0 else np.full(len(v), 1 / len(v))


def _sinif_vektoru(v, siniflar):
    if isinstance(v, dict):
        v = np.array([float(v.get(s, v.get(str(s), 0.0))) for s in siniflar])
    return _normalle(v)
