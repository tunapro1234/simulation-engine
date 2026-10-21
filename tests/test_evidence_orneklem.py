"""Ported from kanit testler/test_orneklem.py: effective sample multiplier. Synthetic data."""
import builtins
import os
import sys
import unittest

import numpy as np
import pandas as pd

from simengine.evidence import orneklem as ok
from simengine.evidence import sema


_VERBOSE = os.environ.get("SIMENGINE_TEST_VERBOSE") == "1" or "-v" in sys.argv or "--verbose" in sys.argv


def print(*a, **k):  # noqa: A001 - diagnostic lines only with -v or SIMENGINE_TEST_VERBOSE=1
    if _VERBOSE:
        builtins.print(*a, **k)


def _surekli(M=10, N=1000, tohum=0, mu_sd=1.0):
    rng = np.random.default_rng(tohum)
    mu = rng.normal(0, mu_sd, M)
    madde = np.repeat(np.arange(M), N)
    y = mu[madde] + rng.normal(0, 1, M * N)
    return rng, madde, mu, y


def _veri(madde, y, gorev, **tahminler):
    df = pd.DataFrame({"madde": [f"m{m}" for m in madde], "gercek": y,
                       **{f"tahmin_{k}": v for k, v in tahminler.items()}})
    return sema.hazirla(df, gorev=gorev)


def _bul(liste, n):
    return next(r for r in liste if r["n"] == n)


class TestOrneklem(unittest.TestCase):
    def test_a_ppi_analitik(self):
        # f = y + gürültü (sd 0,5): madde içi ρ² = 0,8; beklenen 1/(1 − (1−n/N)ρ²)
        rng, madde, _, y = _surekli()
        f = y + rng.normal(0, 0.5, len(y))
        o = ok.carpan(_veri(madde, y, "surekli", iyi=f), {"tahmin_iyi": "iyi"}, n_listesi=(25, 100), B=200)
        rho2 = np.corrcoef(y - pd.Series(y).groupby(madde).transform("mean"),
                           f - pd.Series(f).groupby(madde).transform("mean"))[0, 1] ** 2
        for n in (25, 100):
            r = _bul(o["yontemler"]["tahmin_iyi"]["ppi"], n)
            beklenen = 1 / (1 - (1 - n / 1000) * rho2)
            print(f"\n(a) n={n}: PPI çarpan {r['carpan']:.2f} [{r['alt']:.2f}, {r['ust']:.2f}], "
                  f"analitik {beklenen:.2f} (1/(1-ρ²)={1 / (1 - rho2):.2f}), λ={r['lambda']}")
            self.assertGreater(r["carpan"], 2.5)
            self.assertLess(abs(r["carpan"] / beklenen - 1), 0.2)
            self.assertLessEqual(r["alt"], r["carpan"])
            self.assertGreaterEqual(r["ust"], r["carpan"])
        self.assertIn(" + 100 gerçek yanıt ≈", o["cumle"])
        print("   cümle:", o["cumle"])

    def test_b_bagimsiz_gurultu(self):
        # madde ortalamaları geniş (sd 3): sıfır merkezli gürültü önseli bilgisiz, en iyi n0 = σ²/τ² ≈ 0,1
        rng, madde, _, y = _surekli(M=12, N=600, tohum=1, mu_sd=3.0)
        f = rng.normal(0, 1, len(y))
        o = ok.carpan(_veri(madde, y, "surekli", gurultu=f), {"tahmin_gurultu": "gürültü"},
                      n_listesi=(25, 100), B=200)
        s = o["yontemler"]["tahmin_gurultu"]
        for n in (25, 100):
            p, b = _bul(s["ppi"], n), _bul(s["bayes"], n)
            print(f"\n(b) n={n}: PPI {p['carpan']:.3f}, Bayes {b['carpan']:.3f} (n0 {b['n0_katlar']})")
            self.assertGreaterEqual(p["carpan"], 0.9)
            self.assertLess(p["carpan"], 1.15)
            # bilgisiz önsel: n0 ızgaranın dibinde, çarpan ~1 (sonlu madde sayısında küçük n0 seçilebilir)
            self.assertLessEqual(max(b["n0_katlar"]), 5)
            self.assertLess(abs(b["carpan"] - 1), 0.1)
        self.assertLess(s["yalniz_sim"]["esdeger_n"], 2)

    def test_c_yansiz_onsel(self):
        # f = gerçek madde ortalaması (sabit): Bayes büyük n0 seçer, çarpan çok büyük
        rng, madde, mu, y = _surekli(M=12, N=400, tohum=2)
        o = ok.carpan(_veri(madde, y, "surekli", onsel=mu[madde]), {"tahmin_onsel": "önsel"},
                      n_listesi=(25, 100), B=200)
        s = o["yontemler"]["tahmin_onsel"]
        for n in (25, 100):
            b = _bul(s["bayes"], n)
            print(f"\n(c) n={n}: Bayes çarpan {b['carpan']:.1f} (n0 {b['n0_katlar']}), PPI {_bul(s['ppi'], n)['carpan']:.3f}")
            self.assertGreater(b["carpan"], 10)   # kuramsal (1000+n)²/n²; düzeltme gürültüsü üst ucu sınırlar
            self.assertGreaterEqual(b["n0"], 200)

    def test_d_yanli_sim(self):
        rng, madde, mu, y = _surekli(M=12, N=400, tohum=3)
        f = mu[madde] + 0.5 + rng.normal(0, 0.3, len(y))     # yanlılık 0,5 → σ²/b² ≈ 4
        o = ok.carpan(_veri(madde, y, "surekli", yanli=f), {"tahmin_yanli": "yanlı"}, n_listesi=(25, 100), B=200)
        ys = o["yontemler"]["tahmin_yanli"]["yalniz_sim"]
        print(f"\n(d) yalnız sim eşdeğer n {ys['esdeger_n']} [{ys['alt']}, {ys['ust']}]; "
              f"Bayes n=100 n0 {_bul(o['yontemler']['tahmin_yanli']['bayes'], 100)['n0_katlar']}")
        self.assertLess(ys["esdeger_n"], 6)
        self.assertGreater(ys["esdeger_n"], 2.5)

    def test_e_dagilim(self):
        rng = np.random.default_rng(4)
        satir = []
        for m in range(20):
            for s in range(3):
                p = rng.dirichlet(np.ones(4) * 2)
                q = 0.8 * p + 0.2 * rng.dirichlet(np.ones(4))
                satir.append({"madde": f"m{m}", "segment": f"s{s}", "gercek": p.tolist(),
                              "tahmin_sim": q.tolist(), "tahmin_duz": [0.25] * 4})
        v = sema.hazirla(pd.DataFrame(satir))
        self.assertEqual(v.gorev, "dagilim")
        o = ok.carpan(v, {"tahmin_sim": "sim", "tahmin_duz": "tekdüze"}, n_listesi=(10, 50, 200), B=200)
        sim = o["yontemler"]["tahmin_sim"]
        self.assertEqual(sim["ppi"], [])
        for n, oran in o["kontrol"]["klasik_ampirik_analitik_orani"].items():
            self.assertLess(abs(oran - 1), 0.1, n)
        b, t = _bul(sim["bayes"], 50), _bul(o["yontemler"]["tahmin_duz"]["bayes"], 50)
        print(f"\n(e) dagilim n=50: sim Bayes {b['carpan']:.2f} [{b['alt']:.2f}, {b['ust']:.2f}] n0 {b['n0']}; "
              f"tekdüze {t['carpan']:.2f}; yalnız sim {sim['yalniz_sim']['esdeger_n']}; "
              f"kontrol {o['kontrol']['klasik_ampirik_analitik_orani']}")
        self.assertGreater(b["carpan"], 1.5)
        self.assertGreater(b["carpan"], t["carpan"])

    def test_f_kahin(self):
        # θ = μ bilinirken doğrudan MSE (kahin) ile modülün düzeltilmiş tahmini uyuşmalı (çapraz terim dahil)
        rng, madde, mu, y = _surekli(M=20, N=800, tohum=7)
        f = y + rng.normal(0, 0.5, len(y))
        o = ok.carpan(_veri(madde, y, "surekli", iyi=f), {"tahmin_iyi": "iyi"}, n_listesi=(50,), R=60, B=50)
        s = o["yontemler"]["tahmin_iyi"]
        r2, n = np.random.default_rng(1), 50
        lam, n0 = s["ppi"][0]["lambda"], s["bayes"][0]["n0"]
        e = np.zeros(3)
        for m in range(20):
            yy, ff = y[madde == m], f[madde == m]
            for _ in range(100):
                l = r2.choice(len(yy), n, replace=False)
                yn = yy[l].mean()
                e += [(yn - mu[m]) ** 2, (yn + lam * (ff.mean() - ff[l].mean()) - mu[m]) ** 2,
                      ((n0 * ff.mean() + n * yn) / (n0 + n) - mu[m]) ** 2]
        kp, kb = e[0] / e[1], e[0] / e[2]
        print(f"\n(f) kahin PPI {kp:.2f} / modül {s['ppi'][0]['carpan']:.2f}; "
              f"kahin Bayes {kb:.2f} / modül {s['bayes'][0]['carpan']:.2f} (n0 {n0})")
        self.assertLess(abs(s["ppi"][0]["carpan"] / kp - 1), 0.15)
        self.assertLess(abs(s["bayes"][0]["carpan"] / kb - 1), 0.25)

    def test_cok_sinif_ve_ikili(self):
        rng = np.random.default_rng(5)
        M, N, K = 8, 500, 3
        madde = np.repeat(np.arange(M), N)
        pm = rng.dirichlet(np.ones(K) * 3, M)
        y = np.array([rng.choice(K, p=pm[m]) for m in madde])
        dogru = rng.random(len(y)) < 0.7
        f = np.where(dogru, y, rng.integers(0, K, len(y)))
        o = ok.carpan(_veri(madde, y, "cok_sinif", et=f), {"tahmin_et": "etiket"}, n_listesi=(50,), B=100)
        r = o["yontemler"]["tahmin_et"]["ppi"][0]
        print(f"\n(cok_sinif) PPI {r['carpan']:.2f}; cümle: {o['cumle']}")
        self.assertGreater(r["carpan"], 1.3)
        yb = (rng.random(M * N) < 0.4).astype(float)
        fb = np.clip(0.3 + 0.4 * yb + rng.normal(0, 0.1, len(yb)), 0, 1)
        o2 = ok.carpan(_veri(madde, yb, "ikili", p=fb), {"tahmin_p": "p"}, n_listesi=(50,), B=100)
        self.assertGreater(o2["yontemler"]["tahmin_p"]["ppi"][0]["carpan"], 3)

    def test_tek_madde_ve_ab(self):
        rng = np.random.default_rng(6)
        y = rng.normal(0, 1, 800)
        df = pd.DataFrame({"gercek": y, "tahmin_x": y + rng.normal(0, 0.5, 800)})
        o = ok.carpan(sema.hazirla(df, gorev="surekli"), {"tahmin_x": "x"}, n_listesi=(25, 100), R=60, B=100)
        r = _bul(o["yontemler"]["tahmin_x"]["ppi"], 100)
        print(f"\n(tek madde) PPI n=100 {r['carpan']:.2f} [{r['alt']:.2f}, {r['ust']:.2f}]; "
              f"yalnız sim {o['yontemler']['tahmin_x']['yalniz_sim']}")
        self.assertGreater(r["carpan"], 2)
        self.assertTrue(any("tek madde" in n for n in o["notlar"]))
        dfab = pd.DataFrame({"madde": ["t1"] * 40, "gercek": rng.normal(size=40), "tahmin_s": rng.normal(size=40)})
        self.assertIsNone(ok.carpan(sema.hazirla(dfab, gorev="ab"), {"tahmin_s": "s"}))
        # madde başına birim yetersizse o n atlanır
        o3 = ok.carpan(sema.hazirla(df.iloc[:60], gorev="surekli"), {"tahmin_x": "x"}, n_listesi=(25, 100), B=50)
        self.assertEqual([r["n"] for r in o3["yontemler"]["tahmin_x"]["ppi"]], [25])


if __name__ == "__main__":
    unittest.main()
