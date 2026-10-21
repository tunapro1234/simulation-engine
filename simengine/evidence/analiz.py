"""Segment hatası, yanlılık, segment farkını yakalama, dağılım yayılımı (homojenleşme) ve örnekler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .bootstrap import Orneklem, ga
from .metrik import tvd

MAKS_SEGMENT = 40


def _beklenen(v):
    """Sıralı seçenek dağılımının 0-1 ölçekli beklenen değeri."""
    k = len(v)
    return float(np.dot(v, np.arange(k)) / max(k - 1, 1))


def _sd(v):
    k = len(v)
    x = np.arange(k) / max(k - 1, 1)
    m = np.dot(v, x)
    return float(np.sqrt(max(np.dot(v, (x - m) ** 2), 0)))


def _entropi(v):
    v = np.asarray(v, float)
    v = v[v > 0]
    return float(-(v * np.log(v)).sum())


def yanlilik(veri, y, p, maske) -> float | None:
    g = veri.gorev
    if g in ("ikili", "surekli"):
        return float(np.mean(p[maske]) - np.mean(y[maske]))
    if g == "dagilim":
        return float(np.mean([_beklenen(a) - _beklenen(b) for a, b in zip(p[maske], y[maske])]))
    if g == "cok_sinif":
        K = len(veri.siniflar)
        ty = np.bincount(y[maske], minlength=K) / max(maske.sum(), 1)
        tp = p[maske].mean(0) if p.ndim == 2 else np.bincount(p[maske], minlength=K) / max(maske.sum(), 1)
        return float(tvd(tp, ty))
    return None


def segment_analizi(veri, sutun_dizileri: dict, metrik, w, B: int, tohum: int, en_az: int = 30) -> dict:
    """sutun_dizileri: {sutun: (y, p)} — seçili birkaç sütun (bizim, ham, en iyi baseline, tavan)."""
    out = {}
    d = veri.df
    kume = veri.kume
    for seg in veri.segmentler:
        s = d[seg].astype(str).fillna("?")
        sayim = s.value_counts()
        if len(sayim) > MAKS_SEGMENT:
            sayim = sayim.iloc[:MAKS_SEGMENT]
        degerler = [v for v in sayim.index if sayim[v] >= en_az]
        if len(degerler) < 2:
            continue
        sc = s.map({v: i for i, v in enumerate(degerler)}).fillna(-1).astype(int).to_numpy()
        satirlar = {v: {"segment": v, "n": int(sayim[v]), "yontemler": {}} for v in degerler}
        for sutun, (y, p, maske) in sutun_dizileri.items():
            gecerli = np.ones(len(d), bool) if maske is None else maske
            if metrik.tur == "birim":
                v = np.zeros(len(d))
                v[gecerli] = metrik.fn(y[gecerli], p[gecerli])
                ww = w * gecerli
                # küme × segment toplamları; aynı yeniden örneklerle bütün segmentler
                G, S = int(kume.max()) + 1, len(degerler)
                m = sc >= 0
                idx = kume[m] * S + sc[m]
                top = np.bincount(idx, (ww * v)[m], minlength=G * S).reshape(G, S)
                agi = np.bincount(idx, ww[m], minlength=G * S).reshape(G, S)
                orn = Orneklem(kume, B=B, tohum=tohum)
                R = np.empty((B, S))
                for b in range(B):
                    c = orn.sayim(b)
                    R[b] = (c @ top) / np.maximum(c @ agi, 1e-12)
                nokta = top.sum(0) / np.maximum(agi.sum(0), 1e-12)
                if metrik.son:
                    nokta, R = metrik.son(nokta), metrik.son(R)
                for j, deg in enumerate(degerler):
                    a, u = ga(R[:, j])
                    satirlar[deg]["yontemler"][sutun] = {"deger": float(nokta[j]), "alt": a, "ust": u}
            else:
                for j, deg in enumerate(degerler):
                    mm = (sc == j) & gecerli
                    sub_k = pd.factorize(kume[mm])[0]
                    madde = d["madde"].to_numpy()[mm] if "madde" in d else None
                    nokta = _guvenli(metrik, y[mm], p[mm], madde)
                    orn = Orneklem(sub_k, B=min(B, 100), B_toplu=min(B, 100), tohum=tohum)
                    tek = []
                    for b in range(orn.B_toplu):
                        ii, yeni = orn.satirlar(b)
                        mdd = None if madde is None else madde[ii]
                        tek.append(_guvenli(metrik, y[mm][ii], p[mm][ii], mdd))
                    a, u = ga(np.array(tek))
                    satirlar[deg]["yontemler"][sutun] = {"deger": nokta, "alt": a, "ust": u}
            for j, deg in enumerate(degerler):
                mm = (sc == j) & gecerli
                satirlar[deg]["yontemler"][sutun]["yanlilik"] = yanlilik(veri, y, p, mm) if mm.any() else None
        dpd = {}
        for sutun in sutun_dizileri:
            yy = [r["yontemler"][sutun].get("yanlilik") for r in satirlar.values()]
            yy = [v for v in yy if v is not None and np.isfinite(v)]
            dpd[sutun] = float(max(yy) - min(yy)) if len(yy) >= 2 else None
        out[seg] = {"metrik": metrik.ad, "satirlar": list(satirlar.values()), "dpd": dpd,
                    "segment_fark_kor": {s_: segment_fark_kor(veri, y, p, sc, maske)
                                         for s_, (y, p, maske) in sutun_dizileri.items()}}
    return out


def en_kotu(seg_ozet: dict, sutun: str, yon: int, genel: float, k: int = 3) -> list:
    sat = [(r["segment"], r["yontemler"].get(sutun, {}).get("deger")) for r in seg_ozet["satirlar"]]
    sat = [(s, v) for s, v in sat if v is not None and np.isfinite(v)]
    sat.sort(key=lambda t: yon * t[1])
    return [{"sutun": sutun, "segment": s, "deger": v, "fark_genel": v - genel} for s, v in sat[:k]]


def segment_fark_kor(veri, y, p, sc, maske) -> float | None:
    """Simülasyon segmentler arası farkları yakalıyor mu: (madde içinden arındırılmış) segment
    ortalamalarının korelasyonu."""
    g = veri.gorev
    m = (sc >= 0) if maske is None else (sc >= 0) & maske
    if g in ("ikili", "surekli"):
        yy, pp = y[m].astype(float), p[m].astype(float)
    elif g == "dagilim":
        yy = np.array([_beklenen(v) for v in y[m]])
        pp = np.array([_beklenen(v) for v in p[m]])
    else:
        return None
    t = pd.DataFrame({"y": yy, "p": pp, "s": sc[m]})
    t["madde"] = veri.df["madde"].to_numpy()[m] if "madde" in veri.df else 0
    h = t.groupby(["madde", "s"]).agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    h = h[h.n >= 5]
    if len(h) < 3:
        return None
    h["y"] -= h.groupby(level=0)["y"].transform("mean")
    h["p"] -= h.groupby(level=0)["p"].transform("mean")
    if h.y.std() == 0 or h.p.std() == 0:
        return None
    return float(np.corrcoef(h.y, h.p)[0, 1])


def _guvenli(m, y, p, madde):
    try:
        return float(m.deger(y, p, madde))
    except (ValueError, ZeroDivisionError, IndexError):
        return np.nan


# ------------------------------------------------------------------ yayılım
def _yayilim_bir(veri, y, p, satir=None):
    """Havuzlanmış sd oranı, entropi oranı, w1 (madde içinde)."""
    g, d = veri.gorev, veri.df
    madde = d["madde"].to_numpy() if "madde" in d else np.zeros(len(d), int)
    if satir is not None:
        y, p, madde = y[satir], p[satir], madde[satir]
    kod = pd.factorize(madde)[0]
    if g in ("surekli", "ikili"):
        if g == "ikili":
            p = (p > 0.5).astype(float)
        say = np.bincount(kod)

        def var(x):
            m = np.bincount(kod, x) / say
            return np.bincount(kod, (x - m[kod]) ** 2) / say
        vy, vp = var(y.astype(float)), var(p.astype(float))
        oran = float(np.sqrt(vp.sum() / max(vy.sum(), 1e-12)))
        ent = None
        if g == "ikili":
            def h(q):
                q = np.clip(q, 1e-12, 1 - 1e-12)
                return -(q * np.log(q) + (1 - q) * np.log(1 - q))
            qy, qp = np.bincount(kod, y) / say, np.bincount(kod, p) / say
            ent = float(h(qp).sum() / max(h(qy).sum(), 1e-12))
        return oran, ent
    if g == "cok_sinif":
        lab = p.argmax(1) if p.ndim == 2 else p
        K = len(veri.siniflar)
        hy = hp = 0.0
        for k in np.unique(kod):
            m = kod == k
            hy += _entropi(np.bincount(y[m], minlength=K) / m.sum())
            hp += _entropi(np.bincount(lab[m], minlength=K) / m.sum())
        return None, float(hp / max(hy, 1e-12))
    if g == "dagilim":
        sy = np.array([_sd(v) for v in y]) ** 2
        sp = np.array([_sd(v) for v in p]) ** 2
        hy = np.array([_entropi(v) for v in y])
        hp = np.array([_entropi(v) for v in p])
        return float(np.sqrt(sp.sum() / max(sy.sum(), 1e-12))), float(hp.sum() / max(hy.sum(), 1e-12))
    return None, None


def yayilim(veri, sutun_dizileri: dict, B: int, tohum: int) -> dict:
    if veri.gorev == "ab":
        return {}
    out = {}
    orn = Orneklem(veri.kume, B=B, B_toplu=min(B, 100), tohum=tohum)
    for sutun, (y, p, maske) in sutun_dizileri.items():
        if maske is not None:
            continue
        oran, ent = _yayilim_bir(veri, y, p)
        tek = []
        if oran is not None:
            for b in range(orn.B_toplu):
                ii, _ = orn.satirlar(b)
                tek.append(_yayilim_bir(veri, y, p, ii)[0])
        a, u = ga(np.array(tek)) if tek else (None, None)
        w1 = None
        if veri.gorev == "surekli":
            madde = veri.df["madde"].to_numpy() if "madde" in veri.df else np.zeros(len(y))
            w1 = float(np.mean([wasserstein_distance(y[i], p[i]) for i in pd.Series(madde).groupby(madde).indices.values()]))
        out[sutun] = {"sd_orani": oran, "sd_orani_alt": a, "sd_orani_ust": u, "entropi_orani": ent, "w1": w1}
    return out


# ------------------------------------------------------------------ dağılım örnekleri
def ornekler(veri, bizim: str, diger: list[str], k: int = 3) -> list:
    if veri.gorev != "dagilim":
        return []
    d = veri.df
    Y, P = d["gercek"].tolist(), d[bizim].tolist()
    t = np.array([tvd(a, b) for a, b in zip(Y, P)])
    sira = np.argsort(t)
    sec = [(i, "en iyi") for i in sira[:k]] + [(i, "en kötü") for i in sira[::-1][:k]]
    out = []
    for i, et in sec:
        r = d.iloc[i]
        sec_ad = list(r["secenekler"]) if "secenekler" in d and r["secenekler"] is not None else [str(j) for j in range(len(Y[i]))]
        out.append({"madde": str(r.get("madde", i)), "segment": str(r.get("segment", "")), "secenekler": sec_ad,
                    "gercek": [float(x) for x in Y[i]],
                    "tahminler": {s: [float(x) for x in d[s].iloc[i]] for s in [bizim, *diger] if s in d},
                    "etiket": f"{et} (TVD {t[i]:.2f})"})
    return out
