"""talp73 G1 alt grup karşılaştırması: LLM / latent kolları klasik `gecmis`i nerede geçiyor?

    PY="nice -n 19 /data/talp/venv/bin/python"
    PYTHONPATH=. $PY runs/latent/altgrup.py                      # seed 0, bütün kollar, B 1000 -> ALTGRUP-talp73.md
    PYTHONPATH=. $PY runs/latent/altgrup.py --seed 1 2 --tekrar  # bulunan adayları s1/s2'de (yalnız TALP kolları) yeniden sına

Ölçüler (degerlendir.py tanımları, aynı panel/etiket/eşitlik kuralı; degerlendir.py modül olarak yüklenir):
  kisi_auc  kişi içi (alınan, alınmayan) çiftleri, eşitlik 0,5, hepsini alan / hiç almayan kişi dışarıda; alt küme maskesi
            verilirse çiftler yalnız maskedeki hücrelerden (ör. seyrek kategoriler, kişinin hiç almadığı kategoriler).
  hit@3     kişi başına p sırası (degerlendir.py karmalı eşitlik kırıcısı) ilk 3'ünde ≥1 alım; hiç almayan kişi 0.
  satir_auc bütün hücreler havuzda (kişiler arası + kategoriler arası sıralama), Mann-Whitney, eşitlik 0,5.
  brier     ortalama (p - y)^2 (kalibrasyon dahil).
  kat_auc   tek kategori içinde kişiler arası AUC (kim alır sıralaması).
Belirsizlik: alt gruptaki kişiler yeniden çekilir (B, default_rng(42)); eşli fark aynı çekimlerle; GA %95 yüzdelik;
p = normal yaklaşım 2·Φ(−|d|/sd_boot); Holm iki ailede (tekil kollar; karışımlar) × kişi alt grubu × hücre maskesi, ölçü kisi_auc.
Doğrulama: tam maske kişi AUC = sonuc/<kol>_s0.json 4_kisi_auc (betik eşleşmezse durur).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

TALP73_KOD = Path("/srv/lighthouse/hackathon-coz/projeler/talp-simulation-hackathon-2026/uretim/talp73/kod")
TAHMIN_DIRS = [Path("/data/talp/is/talp73/tahmin"), Path("/data/talp/is/latent/talp73/tahmin"),
               Path("/data/talp/is/latent/altgrup_cpu/talp73/tahmin")]
SONUC_DIRS = [Path("/data/talp/is/talp73/sonuc"), Path("/data/talp/is/latent/talp73/sonuc")]
VERI = Path("/data/talp/is/latent/veri/talp73")
CIKTI = Path("/data/talp/is/latent/ALTGRUP-talp73.md")

TALP_KOL = ["gecmis", "gecmis_anket", "anket", "sonumlu", "populerlik",
            "L0-haiku55", "L0-sonnet55", "L1-haiku55", "L1-sonnet55", "L1-haiku55-kal", "L1-sonnet55-kal", "L1x3-haiku55-kal",
            "L1-gpt41mini", "L2-haiku55", "L2-sonnet55", "L2-haiku55-kal", "L2-sonnet55-kal", "L2-gpt41mini",
            "H1cf-haiku55", "H2cf-haiku55", "H1sab-haiku55", "H2sab-haiku55", "anket-cf", "gecmis-cf", "SSR-haiku55-kal"]
LATENT_KOL = ["enc-an", "enc-gc", "enc-angc", "stk-e5", "enc-sq", "enc-angcsq", "stk-sq",
              "prm-q38-L0", "prm-q38-L1", "prm-q38-L2", "prb-q38-L1", "prb-q38-L2", "stk-q38", "caa-q38",
              "prm-g31-L1", "prm-g31-L2", "prb-g31-L1", "prb-g31-L2", "stk-g31"]
ANKET_ONLY = ["anket", "enc-an", "prb-q38-L1", "prm-q38-L1", "L1-haiku55-kal", "L1-sonnet55-kal", "L1x3-haiku55-kal",
              "L1-haiku55", "L1-sonnet55", "L1-gpt41mini", "L0-haiku55", "L0-sonnet55"]
# öğrenmesiz karışım (etiket yok, ağırlık önceden sabit): sigmoid((1-w)·logit p_gecmis + w·logit p_kol), w ∈ {0,2; 0,5}
MIX_KOL = ["anket", "L1-sonnet55-kal", "L2-sonnet55-kal", "L2-haiku55-kal", "enc-an", "enc-gc", "enc-sq", "stk-e5",
           "prb-q38-L1", "prb-q38-L2", "prm-q38-L2"]
MIX_W = (0.2, 0.5)
CPU_RERUN = {"enc-an", "enc-gc", "enc-angc", "stk-e5"}   # bu makinede yeniden koşuldu (altgrup_cpu), GPU kopyası burada yok


def load_degerlendir():
    spec = importlib.util.spec_from_file_location("degerlendir", TALP73_KOD / "degerlendir.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DG = load_degerlendir()


def bul(kol, s):
    for d in TAHMIN_DIRS:
        f = d / f"{kol}_s{s}.parquet"
        if f.exists():
            return f
    return None


# ------------------------------------------------------------------------------------------------- ölçüler
def kisi_auc(P, Y, C=None):
    """(N,) kişi AUC ve (N,) uygunluk; C (N,K) bool hücre maskesi."""
    C = np.ones_like(Y, dtype=bool) if C is None else C
    pos, neg = (Y == 1) & C, (Y == 0) & C
    fark = P[:, :, None] - P[:, None, :]
    cift = pos[:, :, None] & neg[:, None, :]
    kz = (((fark > 0) + 0.5 * (fark == 0)) * cift).sum((1, 2))
    nc = cift.sum((1, 2)).astype(float)
    ok = nc > 0
    return np.where(ok, kz / np.where(ok, nc, 1), 0.0), ok


def hit3(R, Y):
    sira = np.argsort(-R, axis=1, kind="stable")
    return (np.take_along_axis(Y, sira, axis=1)[:, :3].sum(1) > 0).astype(float)


def agirlikli_auc(p, y, w):
    """p, y (M,), w (B, M) satır ağırlıkları -> (B,) Mann-Whitney AUC, eşitlik 0,5."""
    o = np.argsort(p, kind="stable")
    p, y, w = p[o], y[o], w[:, o]
    wn, wp = w * (y == 0), w * (y == 1)
    cn = np.cumsum(wn, axis=1)
    yeni = np.r_[True, p[1:] != p[:-1]]
    gid = np.cumsum(yeni) - 1
    bas = np.flatnonzero(yeni)
    son = np.r_[bas[1:] - 1, len(p) - 1]
    once = np.where(bas[None, :] > 0, cn[:, np.maximum(bas - 1, 0)], 0.0)       # (B, G)
    ici = cn[:, son] - once
    alt = once[:, gid] + 0.5 * ici[:, gid]                                       # (B, M)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (wp * alt).sum(1) / (wp.sum(1) * wn.sum(1))


class Olcer:
    """Bir alt grup (kişi indeksleri) + hücre maskesi için bütün kolların bootstrap ölçüleri."""

    def __init__(self, idx, B, rng_seed=42):
        self.idx = np.asarray(idx)
        n = len(self.idx)
        rng = np.random.default_rng(rng_seed)
        W = np.zeros((B, n))
        for b in range(B):
            np.add.at(W[b], rng.integers(0, n, n), 1)
        self.W = np.vstack([np.ones((1, n)), W])          # satır 0 = nokta

    def kisi(self, P, Y, C=None):
        a, ok = kisi_auc(P[self.idx], Y[self.idx], None if C is None else C[self.idx])
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.W @ (a * ok)) / (self.W @ ok)

    def ort(self, v):
        return (self.W @ v[self.idx]) / self.W.sum(1)

    def satir(self, P, Y, C=None):
        Pi, Yi = P[self.idx], Y[self.idx]
        K = P.shape[1]
        Ci = np.ones_like(Yi, dtype=bool) if C is None else C[self.idx]
        m = Ci.ravel()
        w = np.repeat(self.W, K, axis=1)[:, m]
        return agirlikli_auc(Pi.ravel()[m], Yi.ravel()[m], w)

    def brier(self, P, Y):
        return self.ort(((P - Y) ** 2).mean(1))

    def kat(self, P, Y, k):
        return agirlikli_auc(P[self.idx, k], Y[self.idx, k], self.W)


def ozet(v):
    b = v[1:][np.isfinite(v[1:])]
    if not np.isfinite(v[0]) or len(b) < 10:
        return {"d": float(v[0]) if np.isfinite(v[0]) else None, "lo": None, "hi": None, "p": None}
    lo, hi = np.percentile(b, [2.5, 97.5])
    from scipy.stats import norm
    sd = b.std()
    p = float(2 * norm.sf(abs(v[0]) / sd)) if sd > 0 else (0.0 if v[0] != 0 else 1.0)   # normal yaklaşım (Holm için)
    return {"d": float(v[0]), "lo": float(lo), "hi": float(hi), "p": float(p)}


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


# ------------------------------------------------------------------------------------------------- veri
class Veri:
    def __init__(self, s):
        PANEL = DG.PANEL
        self.kat = pl.read_parquet(f"{PANEL}/kategoriler.parquet").sort("sira")["kategori"].to_list()
        self.orn = {s: sorted(pl.read_parquet(f"{PANEL}/orneklem_s{s}.parquet")["uid"].to_list())}
        anket = pl.read_parquet(f"{PANEL}/anket.parquet")
        self.pn = DG.Panel(s, self.kat, self.orn[s], anket)
        self.s, self.uids, self.Y = s, self.orn[s], self.pn.Y
        a = pl.DataFrame({"uid": self.uids}).join(anket, on="uid", how="left")
        self.yas = a["yas"].to_list()
        self.gelir = a["gelir"].to_list()
        self.cins = a["cinsiyet"].to_list()
        # geçmiş: test kesimi 2022-07-01 öncesi bütün kalemler (bütün kategoriler, 2018'den)
        import datetime as dt
        kes = dt.date(2022, 7, 1)
        g = (pl.scan_parquet(VERI / "gecmis_olay.parquet").filter(pl.col("kesim") == kes)
             .filter(pl.col("uid").is_in(self.uids)).select("uid", "tarih", "kategori").collect())
        n_all = dict(g.group_by("uid").len().iter_rows())
        n12 = dict(g.filter(pl.col("tarih") >= dt.date(2021, 7, 1)).group_by("uid").len().iter_rows())
        self.n_kalem = np.array([n_all.get(u, 0) for u in self.uids], float)
        self.n_12ay = np.array([n12.get(u, 0) for u in self.uids], float)
        gk = g.filter(pl.col("kategori").is_in(self.kat)).select("uid", "kategori").unique()
        pi = {u: i for i, u in enumerate(self.uids)}
        ci = {c: j for j, c in enumerate(self.kat)}
        self.onceden = np.zeros(self.Y.shape, bool)
        for u, c in gk.iter_rows():
            self.onceden[pi[u], ci[c]] = True
        self.oran = self.pn.oran                       # önceki 6 ay alıcı oranı (örneklem dışı kişiler; test etiketi değil)

    def matris(self, f):
        _, s, t = DG.oku_tahmin(str(f), self.kat, self.orn)
        assert s == self.s
        return DG.matris(t, self.uids, self.kat)


ARALIK: list = []      # çeyrek sınırları (tohuma göre değişir; anahtarda yok, metinde yazılır)


def ceyrek(x):
    q = np.quantile(x, [0.25, 0.5, 0.75])
    lab = np.digitize(x, q, right=True)                # 0..3
    adlar = []
    for i in range(4):
        v = x[lab == i]
        adlar.append(f"Q{i + 1}")
        ARALIK.append(f"Q{i + 1} {int(v.min())}-{int(v.max())}" if len(v) else f"Q{i + 1} boş")
    return lab, adlar


def gruplar(v: Veri):
    """[(aile, ad, kişi indeksleri)] kişi alt grupları."""
    N = len(v.uids)
    out = [("hepsi", "400 kişi", np.arange(N))]
    lab, ad = ceyrek(v.n_kalem)
    out += [("geçmiş kalem (2018→kesim)", ad[i], np.flatnonzero(lab == i)) for i in range(4)]
    lab, ad = ceyrek(v.n_12ay)
    out += [("son 12 ay kalem", ad[i], np.flatnonzero(lab == i)) for i in range(4)]
    yas_g = {"18 - 24 years": "18-24", "25 - 34 years": "25-34", "35 - 44 years": "35-44", "45 - 54 years": "45-54",
             "55 - 64 years": "55+", "65 and older": "55+"}
    yg = np.array([yas_g.get(x, "?") for x in v.yas])
    out += [("yaş", g, np.flatnonzero(yg == g)) for g in ["18-24", "25-34", "35-44", "45-54", "55+"]]
    gel_g = {"Less than $25,000": "<50k", "$25,000 - $49,999": "<50k", "$50,000 - $74,999": "50-100k",
             "$75,000 - $99,999": "50-100k", "$100,000 - $149,999": "≥100k", "$150,000 or more": "≥100k"}
    gg = np.array([gel_g.get(x, "belirtilmemiş") for x in v.gelir])
    out += [("gelir", g, np.flatnonzero(gg == g)) for g in ["<50k", "50-100k", "≥100k"]]
    cg = np.array([x if x in ("Female", "Male") else "diğer" for x in v.cins])
    out += [("cinsiyet", g, np.flatnonzero(cg == g)) for g in ["Female", "Male"]]
    ns = v.Y.sum(1)
    out += [("alım sayısı (test)", "hiç almayan hariç", np.flatnonzero(ns > 0))]
    return out


def maskeler(v: Veri):
    """[(ad, (N,K) bool)] hücre maskeleri (kişi AUC çiftleri yalnız maskeden)."""
    K = len(v.kat)
    sira = np.argsort(-v.oran)
    yay = np.zeros(K, bool)
    yay[sira[:K // 2]] = True
    N = len(v.uids)
    ust = sira[:5]
    alt = sira[-5:]
    m5u, m5a = np.zeros(K, bool), np.zeros(K, bool)
    m5u[ust], m5a[alt] = True, True
    return [("yaygın 10 kategori", np.tile(yay, (N, 1))), ("seyrek 10 kategori", np.tile(~yay, (N, 1))),
            ("en yaygın 5", np.tile(m5u, (N, 1))), ("en seyrek 5", np.tile(m5a, (N, 1))),
            ("yeni kategori (kişi önceden hiç almamış)", ~v.onceden), ("tekrar (önceden almış)", v.onceden)]


# ------------------------------------------------------------------------------------------------- koşu
def f3(x):
    return "—" if x is None else f"{x:.3f}"


def fd(o, holm_p=None):
    if o["d"] is None:
        return "—"
    if o["lo"] is None:
        return f"{o['d']:+.3f}"
    yil = ""
    if o["lo"] > 0:
        yil = " **↑**"
    elif o["hi"] < 0:
        yil = " ↓"
    h = "" if holm_p is None else (" ‡" if holm_p < 0.05 else "")
    return f"{o['d']:+.3f} [{o['lo']:+.3f}, {o['hi']:+.3f}]{yil}{h}"


def dogrula(v, mats):
    """Tam maske kişi AUC = degerlendir.py sonuc json'u (3 ondalık)."""
    satir = []
    for kol, (P, R) in mats.items():
        if kol.startswith("mix"):
            continue
        a, ok = kisi_auc(P, v.Y)
        bizim = round(float(a[ok].mean()), 4)
        js = next((d / f"{kol}_s{v.s}.json" for d in SONUC_DIRS if (d / f"{kol}_s{v.s}.json").exists()), None)
        ref = json.load(open(js))["olcu"]["4_kisi_auc"]["deger"] if js else None
        if ref is not None and abs(ref - bizim) > 1e-4:
            raise SystemExit(f"DOĞRULAMA: {kol} bizim {bizim} degerlendir.py {ref}")
        satir.append((kol, bizim, ref))
    return satir


def havuz_tablosu(havuz, taban, B):
    """Üç tohum birleşik kişi AUC: kişi örnekleri (tohum, uid) birleşir, bootstrap uid kümeleriyle (aynı kişi birden çok
    tohumda olabilir). Yalnız her tohumda bulunan kollar."""
    ortak = set.intersection(*[set(m) for _, m, _, _ in havuz.values()])
    kollar = [k for k in TALP_KOL + LATENT_KOL + [x for x in next(iter(havuz.values()))[1] if x.startswith("mix")]
              if k in ortak and k != taban]
    sec = [("hepsi", "400 kişi", None)] + [("geçmiş kalem (2018→kesim)", f"Q{i}", None) for i in range(1, 5)] + \
          [("son 12 ay kalem", "Q1", None), ("hepsi", "400 kişi", "yeni kategori (kişi önceden hiç almamış)"),
           ("hepsi", "400 kişi", "tekrar (önceden almış)"), ("hepsi", "400 kişi", "seyrek 10 kategori"),
           ("hepsi", "400 kişi", "en seyrek 5")]
    uids_all = sorted({u for v, _, _, _ in havuz.values() for u in v.uids})
    ui = {u: i for i, u in enumerate(uids_all)}
    rng = np.random.default_rng(42)
    U = len(uids_all)
    Wu = np.vstack([np.ones(U)] + [np.bincount(rng.integers(0, U, U), minlength=U) for _ in range(B)])   # (B+1, U)
    out = ["## Üç tohum birleşik kişi AUC (kişi örnekleri birleşik, bootstrap kişi kümeleriyle)", "",
           f"_Hücre: kol − `{taban}` [%95 GA]; n = kişi örneği (tohum × kişi). Çeyrekler her tohumun kendi sınırlarıyla._", "",
           "| kol | " + " | ".join(("hepsi" if af == "hepsi" else ("geçmiş " if af.startswith("geçmiş") else "12 ay ") + g)
                                     + ("" if m is None else " · " + m) for af, g, m in sec) + " |", "|---|" + "---|" * len(sec)]
    satir_t, satirlar = [], {k: [] for k in kollar}
    for af, g, mk in sec:
        parca = {k: [] for k in kollar + [taban]}
        sahip = []
        for s, (v, mats, G, M) in havuz.items():
            idx = next(ix for a_, g_, ix in G if a_ == af and g_ == g)
            C = None if mk is None else dict(M)[mk]
            for k in parca:
                a_, ok = kisi_auc(mats[k][0][idx], v.Y[idx], None if C is None else C[idx])
                parca[k].append((a_, ok))
            sahip += [ui[v.uids[i]] for i in idx]
        sahip = np.array(sahip)
        W = Wu[:, sahip]                                       # (B+1, n) her örnek kişisinin çekilme sayısı
        def ort(k):
            a_ = np.concatenate([x for x, _ in parca[k]])
            ok = np.concatenate([o for _, o in parca[k]])
            with np.errstate(invalid="ignore", divide="ignore"):
                return (W @ (a_ * ok)) / (W @ ok)
        tb = ort(taban)
        satir_t.append(f"{tb[0]:.3f} (n={len(sahip)})")
        for k in kollar:
            x = ort(k)
            satirlar[k].append(f"{x[0]:.3f} ({fd(ozet(x - tb))})")
    out.append(f"| {taban} (mutlak) | " + " | ".join(satir_t) + " |")
    for k in kollar:
        out.append(f"| {k} | " + " | ".join(satirlar[k]) + " |")
    return out + [""]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, nargs="*", default=[0])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--taban", default="gecmis")
    ap.add_argument("--cikti", default=str(CIKTI))
    ap.add_argument("--json", default="/data/talp/is/latent/altgrup/altgrup-talp73")
    a = ap.parse_args(argv)
    md = []
    tum = {}                                           # tohum -> (sonuc, holm_p)
    havuz = {}                                         # tohum -> (Veri, mats, gruplar, maskeler)
    kayit = {}
    for s in a.seed:
        v = Veri(s)
        kollar = TALP_KOL + LATENT_KOL
        mats, yok = {}, []
        for k in kollar:
            f = bul(k, s)
            if f is None:
                yok.append(k)
            else:
                mats[k] = v.matris(f)
        lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
        mix = []
        for k in MIX_KOL:
            if k in mats:
                for w in MIX_W:
                    P = 1 / (1 + np.exp(-((1 - w) * lg(mats[a.taban][0]) + w * lg(mats[k][0]))))
                    Rg = mats[a.taban][1] - mats[a.taban][0]                    # aynı karma eşitlik kırıcısı
                    mats[f"mix{int(w * 100)}:{k}"] = (P, P + Rg)
                    mix.append(f"mix{int(w * 100)}:{k}")
        dg = dogrula(v, mats)
        Pt, Rt = mats[a.taban]
        md += [f"# talp73 G1 alt grup karşılaştırması (tohum {s}, B {a.B}, taban `{a.taban}`)", "",
               "_Üretim: `runs/latent/altgrup.py` (simulation-engine). Tanımlar betiğin başında; kişi AUC degerlendir.py ile "
               "aynı (tam maske doğrulaması aşağıda). Hücre: kol − taban farkı [%95 bootstrap GA, alt gruptaki kişiler yeniden "
               "çekilir]; **↑** = GA sıfırın üstünde, ↓ = altında; ‡ = Holm düzeltmesinden sonra da p<0,05 (iki aile: tekil kollar / "
               "karışımlar; her biri kol × kişi alt grubu × hücre maskesi, yalnız kişi AUC; p normal yaklaşım)._", ""]
        md += [f"Bulunamayan kollar (bu makinede tahmin dosyası yok): {', '.join(yok) if yok else 'yok'}.", ""]
        md += ["Doğrulama (tam maske kişi AUC, bu betik / degerlendir.py json): " +
               ", ".join(f"{k} {b:.4f}/{'—' if r is None else f'{r:.4f}'}" for k, b, r in dg), ""]
        ARALIK.clear()
        G = gruplar(v)
        md += [f"Çeyrek sınırları (kalem sayısı; önce 2018→kesim, sonra son 12 ay): {', '.join(ARALIK)}.", ""]
        M = maskeler(v)
        sonuc = {}       # (aile, grup, maske, olcu, kol) -> ozet
        aile_p = []      # (anahtar, p)
        for aile, gad, idx in G:
            ol = Olcer(idx, a.B)
            for mad, C in [("bütün kategoriler", None)] + (M if aile == "hepsi" else []):
                tb = {"kisi_auc": ol.kisi(Pt, v.Y, C)}
                if C is None:
                    tb.update(hit3=ol.ort(hit3(Rt, v.Y)), satir_auc=ol.satir(Pt, v.Y), brier=ol.brier(Pt, v.Y))
                else:
                    tb["satir_auc"] = ol.satir(Pt, v.Y, C)
                for kol, (P, R) in mats.items():
                    d = {"kisi_auc": ol.kisi(P, v.Y, C)}
                    if C is None:
                        d.update(hit3=ol.ort(hit3(R, v.Y)), satir_auc=ol.satir(P, v.Y), brier=ol.brier(P, v.Y))
                    else:
                        d["satir_auc"] = ol.satir(P, v.Y, C)
                    for o, x in d.items():
                        key = (aile, gad, mad, o, kol)
                        fk = tb[o] - x if o == "brier" else x - tb[o]          # brier: iyileşme (taban − kol), + = iyi
                        sonuc[key] = {"deger": float(x[0]), "fark": ozet(fk), "n": int(len(idx))}
                        if o == "kisi_auc" and kol != a.taban:
                            aile_p.append((key, sonuc[key]["fark"]["p"], kol.startswith("mix")))
        # Holm
        holm_p = {}
        for karisim in (False, True):                  # iki aile: tekil kollar / öğrenmesiz karışımlar
            ks = [(k, p) for k, p, m in aile_p if p is not None and m == karisim]
            holm_p.update(dict(zip([k for k, _ in ks], holm([p for _, p in ks]))))
        kayit[s] = {"|".join(k): {**val, "holm_p": holm_p.get(k)} for k, val in sonuc.items()}
        tum[s] = (sonuc, holm_p)
        havuz[s] = (v, {k: m for k, m in mats.items()}, G, M)

        def tablo(baslik, satirlar, kolon_kollar, olcu="kisi_auc", maske="bütün kategoriler"):
            md.extend([f"## {baslik}", "", f"_Ölçü: {olcu}. İlk satır grubu taban `{a.taban}` mutlak değeri, diğer hücreler fark._",
                       "", "| kol | " + " | ".join(f"{g} ({sonuc[(af, g, mk, olcu, a.taban)]['n']})" for af, g, mk in satirlar) +
                       " |", "|---|" + "---|" * len(satirlar)])
            md.append(f"| {a.taban} (mutlak) | " + " | ".join(f3(sonuc[(af, g, mk, olcu, a.taban)]["deger"])
                                                         for af, g, mk in satirlar) + " |")
            for kol in kolon_kollar:
                if kol == a.taban or kol not in mats:
                    continue
                hs = []
                for af, g, mk in satirlar:
                    k = (af, g, mk, olcu, kol)
                    hs.append(f"{f3(sonuc[k]['deger'])} ({fd(sonuc[k]['fark'], holm_p.get(k))})")
                md.append(f"| {kol}{' ᶜ' if kol in CPU_RERUN else ''} | " + " | ".join(hs) + " |")
            md.append("")

        ana = [k for k in TALP_KOL + LATENT_KOL if k in mats]
        aileler = {}
        for aile, gad, idx in G:
            aileler.setdefault(aile, []).append((aile, gad, "bütün kategoriler"))
        md += ["## 0. Hepsi (400 kişi): dört ölçü", "", "| kol | kişi AUC | hit@3 | satır AUC (havuz) | Brier (değer düşük iyi; fark = taban − kol, + iyi) |",
               "|---|---|---|---|---|"]
        for kol in ana:
            c = []
            for o in ("kisi_auc", "hit3", "satir_auc", "brier"):
                k = ("hepsi", "400 kişi", "bütün kategoriler", o, kol)
                c.append(f3(sonuc[k]["deger"]) if kol == a.taban else f"{f3(sonuc[k]['deger'])} ({fd(sonuc[k]['fark'])})")
            md.append(f"| {kol}{' ᶜ' if kol in CPU_RERUN else ''} | " + " | ".join(c) + " |")
        md.append("")
        n = 1
        for aile, sat in aileler.items():
            if aile == "hepsi":
                continue
            tablo(f"{n}. {aile}", sat, ana)
            n += 1
        if mix:
            tablo(f"{n}. Öğrenmesiz karışım gecmis + kol (logit, kol ağırlığı w=0,2 / 0,5; test etiketi kullanılmaz)",
                  [s_ for s_ in [("hepsi", "400 kişi", "bütün kategoriler")] + aileler["geçmiş kalem (2018→kesim)"]], mix)
            tablo(f"{n}b. Karışım, hücre maskeleri", [("hepsi", "400 kişi", mk) for mk, _ in M], mix)
            n += 1
        tablo(f"{n}. Hücre maskeleri (kategori alt kümesi; kişi AUC çiftleri yalnız maskeden)",
              [("hepsi", "400 kişi", mk) for mk, _ in M], ana)
        n += 1
        tablo(f"{n}. Hücre maskeleri, satır AUC (havuz, kişiler arası dahil)",
              [("hepsi", "400 kişi", mk) for mk, _ in M], ana, olcu="satir_auc")
        n += 1
        for aile in ("alım sayısı (test)",):
            tablo(f"{n}. Hiç almayan hariç: satır AUC / hit@3 / Brier", aileler[aile], ana, olcu="satir_auc")
            tablo(f"{n}b. Hiç almayan hariç: hit@3", aileler[aile], ana, olcu="hit3")
            tablo(f"{n}c. Hiç almayan hariç: Brier (fark = taban − kol, + = kol daha iyi)", aileler[aile], ana, olcu="brier")
        n += 1
        # kategori içi kişiler arası AUC
        olh = Olcer(np.arange(len(v.uids)), a.B)
        kat_kol = [k for k in ["gecmis", "anket", "gecmis_anket", "L2-sonnet55-kal", "L2-haiku55-kal", "H2cf-haiku55",
                               "L1-sonnet55-kal", "enc-gc", "enc-angc", "stk-e5", "enc-sq", "stk-sq"] if k in mats]
        md += [f"## {n}. Kategori içi kişiler arası AUC (kim alır), kategori önceki oranına göre yaygından seyreğe", "",
               "| kategori (önceki oran, test alıcı) | " + " | ".join(kat_kol) + " |", "|---|" + "---|" * len(kat_kol)]
        kat_kayit = {}
        for k in np.argsort(-v.oran):
            tbk = olh.kat(Pt, v.Y, k)
            hs = []
            for kol in kat_kol:
                x = olh.kat(mats[kol][0], v.Y, k)
                o = ozet(x - tbk)
                kat_kayit[f"{v.kat[k]}|{kol}"] = {"deger": float(x[0]), "fark": o}
                hs.append(f3(x[0]) if kol == a.taban else f"{f3(x[0])} ({fd(o)})")
            md.append(f"| {v.kat[k]} ({v.oran[k]:.2f}, {int(v.Y[:, k].sum())}) | " + " | ".join(hs) + " |")
        md.append("")
        kayit[s]["kat_auc"] = kat_kayit
        n += 1
        # yalnız anket koşulu: taban anket
        if "anket" in mats:
            Pa = mats["anket"][0]
            ak = [k for k in ANKET_ONLY if k in mats]
            md += [f"## {n}. Yalnız anket (geçmişsiz) koşulu: taban `anket`", "",
                   "_Geçmiş görmeyen kollar. Hücre: değer (kol − anket [%95 GA])._", "",
                   "| kol | kişi AUC | hit@3 | satır AUC | Brier (fark = anket − kol) | kişi AUC, geçmiş Q1 | kişi AUC, yeni kategori |",
                   "|---|---|---|---|---|---|---|"]
            q1 = next(idx for af, g, idx in G if af.startswith("geçmiş kalem") and g.startswith("Q1"))
            yeni = dict(M)["yeni kategori (kişi önceden hiç almamış)"]
            ol1 = Olcer(q1, a.B)
            ank = {}
            for kol in ak:
                P, R = mats[kol]
                c = []
                for nm, x, base in [("kisi_auc", olh.kisi(P, v.Y), olh.kisi(Pa, v.Y)),
                                    ("hit3", olh.ort(hit3(R, v.Y)), olh.ort(hit3(mats["anket"][1], v.Y))),
                                    ("satir_auc", olh.satir(P, v.Y), olh.satir(Pa, v.Y)),
                                    ("brier", olh.brier(P, v.Y), olh.brier(Pa, v.Y)),
                                    ("kisi_auc_Q1", ol1.kisi(P, v.Y), ol1.kisi(Pa, v.Y)),
                                    ("kisi_auc_yeni", olh.kisi(P, v.Y, yeni), olh.kisi(Pa, v.Y, yeni))]:
                    o = ozet(base - x if nm == "brier" else x - base)
                    ank[f"{kol}|{nm}"] = {"deger": float(x[0]), "fark": o}
                    c.append(f3(x[0]) if kol == "anket" else f"{f3(x[0])} ({fd(o)})")
                md.append(f"| {kol}{' ᶜ' if kol in CPU_RERUN else ''} | " + " | ".join(c) + " |")
            md.append("")
            kayit[s]["anket_kosulu"] = ank
        # özet: GA sıfırı dışlayan pozitif farklar
        pos = [(k, val) for k, val in sonuc.items() if k[4] != a.taban and val["fark"]["lo"] is not None
               and val["fark"]["lo"] > 0]
        pos_b = []
        md += ["## Özet: taban `gecmis`i GA sıfırı dışlayarak geçen (kol, alt grup, ölçü)", ""]
        if not pos and not pos_b:
            md.append("Yok.")
        for k, val in sorted(pos + pos_b, key=lambda t: -abs(t[1]["fark"]["d"])):
            md.append(f"- {k[4]} · {k[0]} / {k[1]} / {k[2]} · {k[3]}: {f3(val['deger'])} "
                      f"({fd(val['fark'], holm_p.get(k))}), n={val['n']}")
        md.append("")
    if len(havuz) > 1:
        md += havuz_tablosu(havuz, a.taban, a.B)
    if len(tum) > 1:
        bas = ["# Tohumlar arası tekrar (kişi AUC ve satır AUC, taban `gecmis`)", "",
               "_En az bir tohumda GA sıfırın üstünde olan (kol, alt grup, ölçü). Tohum başına fark [%95 GA]; latent kollar "
               "yalnız tohum 0'da var. Sıra: kaç tohumda ↑, sonra ortalama fark._", "",
               "| kol | alt grup | ölçü | " + " | ".join(f"s{s}" for s in tum) + " | ↑ tohum |", "|---|---|---|" + "---|" * (len(tum) + 1)]
        adaylar = {k for s, (so, _) in tum.items() for k, val in so.items()
                   if k[3] in ("kisi_auc", "satir_auc") and k[4] != a.taban and val["fark"]["lo"] is not None and val["fark"]["lo"] > 0}
        satirlar = []
        for k in adaylar:
            hs, say, ds = [], 0, []
            for s, (so, hp) in tum.items():
                if k in so:
                    o = so[k]["fark"]
                    hs.append(fd(o, hp.get(k)))
                    say += o["lo"] is not None and o["lo"] > 0
                    ds.append(o["d"])
                else:
                    hs.append("—")
            satirlar.append((-say, -np.mean(ds), f"| {k[4]} | {k[0]} / {k[1]} / {k[2]} | {k[3]} | " + " | ".join(hs) + f" | {say}/{len(ds)} |"))
        bas += [t[2] for t in sorted(satirlar)] + [""]
        md = bas + md
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump(kayit, open(a.json + f"_s{'-'.join(map(str, a.seed))}.json", "w"), ensure_ascii=False, indent=0, default=str)
    Path(a.cikti).write_text("\n".join(md) + "\n")
    print(a.cikti)


if __name__ == "__main__":
    main()
