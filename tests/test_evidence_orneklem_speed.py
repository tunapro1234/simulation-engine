"""Effective-sample multiplier on large many-class frames (W7 deferred D9 #3): no rows x classes one-hot and no
replicates x n x classes tensors; class labels stay integer codes. Pins the output on small inputs to the values the
dense implementation produced, and guards time and peak memory on a synthetic many-class frame. Offline."""
import hashlib
import json
import time
import tracemalloc
import unittest

import numpy as np
import pandas as pd

from simengine.evidence import orneklem as ok
from simengine.evidence import sema


def _vakalar():
    rng = np.random.default_rng(7)
    out = {}
    # continuous, 25 items (R=40, Bayes kept), large offset
    M, N = 25, 120
    madde = np.repeat(np.arange(M), N)
    mu = rng.normal(1000, 2, M)
    y = mu[madde] + rng.normal(0, 1, M * N)
    out["surekli_cok"] = (pd.DataFrame({"madde": [f"m{m}" for m in madde], "gercek": y,
                                        "tahmin_a": y + rng.normal(0, .7, M * N), "tahmin_b": mu[madde] + .3}),
                          "surekli", (10, 25, 50))
    # binary, 6 items (R=400)
    M, N = 6, 150
    madde = np.repeat(np.arange(M), N)
    p = rng.uniform(.2, .8, M)[madde]
    y = (rng.random(M * N) < p).astype(int)
    out["ikili_az"] = (pd.DataFrame({"madde": [f"m{m}" for m in madde], "gercek": y,
                                     "tahmin_a": np.clip(p + rng.normal(0, .1, M * N), 0, 1)}), "ikili", (10, 50))
    # multiclass labels, 21 items, K=5, one column with an unknown label
    M, N, K = 21, 90, 5
    cls = np.array([f"c{i}" for i in range(K)])
    madde = np.repeat(np.arange(M), N)
    t = rng.integers(0, K, M * N)
    f = np.where(rng.random(M * N) < .5, t, rng.integers(0, K, M * N))
    g = rng.integers(0, K, M * N).astype(object)
    g[:7] = "yok"
    out["cok_etiket"] = (pd.DataFrame({"madde": [f"m{m}" for m in madde], "gercek": cls[t], "tahmin_a": cls[f],
                                       "tahmin_b": [cls[v] if v != "yok" else "yok" for v in g]}),
                         "cok_sinif", (10, 25, 50))
    # multiclass single item, labels (lambda per replicate, unit bootstrap)
    N, K = 260, 6
    cls = np.array([f"c{i}" for i in range(K)])
    t = rng.integers(0, K, N)
    f = np.where(rng.random(N) < .6, t, rng.integers(0, K, N))
    out["cok_tek"] = (pd.DataFrame({"gercek": cls[t], "tahmin_a": cls[f]}), "cok_sinif", (10, 25, 100))
    # multiclass probability vectors + labels, 3 items
    M, N, K = 3, 120, 4
    cls = np.array([f"c{i}" for i in range(K)])
    madde = np.repeat(np.arange(M), N)
    t = rng.integers(0, K, M * N)
    P = rng.dirichlet(np.ones(K), M * N)
    P[np.arange(M * N), t] += .5
    P /= P.sum(1, keepdims=True)
    out["cok_olasilik"] = (pd.DataFrame({"madde": [f"m{m}" for m in madde], "gercek": cls[t], "tahmin_p": list(P),
                                         "tahmin_a": cls[P.argmax(1)]}), "cok_sinif", (10, 50))
    # multiclass probability vectors, single item
    N, K = 200, 3
    cls = np.array([f"c{i}" for i in range(K)])
    t = rng.integers(0, K, N)
    P = rng.dirichlet(np.ones(K) * 2, N)
    out["cok_olasilik_tek"] = (pd.DataFrame({"gercek": cls[t], "tahmin_p": list(P)}), "cok_sinif", (10, 25))
    # continuous single item (kanit.bedel path) and distribution task
    N = 300
    y = rng.normal(5, 1, N)
    out["surekli_tek"] = (pd.DataFrame({"gercek": y, "tahmin_a": y + rng.normal(0, 1, N)}), "surekli", (10, 25))
    M, K = 40, 4
    Pg = rng.dirichlet(np.ones(K) * 3, M)
    out["dagilim"] = (pd.DataFrame({"madde": [f"m{m}" for m in range(M)], "gercek": list(Pg),
                                    "tahmin_a": list(.7 * Pg + .3 * rng.dirichlet(np.ones(K), M))}),
                      "dagilim", (10, 50))
    return out


def _calistir(df, gorev, nl):
    cols = {c: c for c in df.columns if c.startswith("tahmin_")}
    return ok.carpan(sema.hazirla(df, gorev=gorev), cols, n_listesi=nl, R=40, tohum=3, B=150)


def _hash(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:16]


# whole report, as produced by the dense implementation (commit 567c6b2)
_PIN = {"surekli_cok": "18bbd34b1da7b726", "ikili_az": "5cfd792518321bab", "cok_etiket": "e6d16232eae508b7",
        "cok_tek": "66bb53153417188c", "cok_olasilik": "d209eab394a691a1", "cok_olasilik_tek": "84989639a0dea319",
        "surekli_tek": "966d1336f25c01e7", "dagilim": "c999628c4346ce80"}
# readable subset of the same reports: (n, carpan, alt, ust, n_esdeger)
_PIN_SATIR = {
    ("cok_etiket", "tahmin_a", "ppi"): [(10, 1.2546, 1.2201, 1.2985, 13), (25, 1.223, 1.1914, 1.2609, 31),
                                        (50, 1.1027, 1.07, 1.133, 55)],
    ("cok_etiket", "tahmin_b", "bayes"): [(10, 7.1966, 4.7576, 13.3018, 72), (25, 3.8149, 2.7499, 6.0325, 95),
                                          (50, 2.7149, 2.0655, 4.365, 136)],
    ("cok_tek", "tahmin_a", "ppi"): [(10, 1.4481, 1.3569, 1.529, 14), (25, 1.4026, 1.3364, 1.4716, 35),
                                     (100, 1.2779, 1.2165, 1.3343, 128)],
    ("cok_olasilik", "tahmin_p", "ppi"): [(10, 1.9432, 1.8953, 1.9823, 19), (50, 1.4514, 1.4239, 1.4688, 73)],
    ("surekli_cok", "tahmin_a", "ppi"): [(10, 2.5927, 2.3678, 2.7883, 26), (25, 2.2176, 2.0391, 2.4275, 55),
                                         (50, 1.6709, 1.5189, 1.8359, 84)],
}
_PIN_YALNIZ = {("cok_etiket", "tahmin_a"): (78.08, 56.31, 125.47), ("cok_tek", "tahmin_a"): (173.78, 50.99, 401.27),
               ("cok_olasilik_tek", "tahmin_p"): (85.87, 16.52, 1000000.0)}


def _cok_sinif_buyuk(N, K, M, tohum=0):
    rng = np.random.default_rng(tohum)
    cls = np.array([f"c{i}" for i in range(K)])
    t = rng.zipf(1.5, N) % K
    ours = np.where(rng.random(N) < .4, t, rng.integers(0, K, N))
    raw = np.where(rng.random(N) < .2, t, rng.integers(0, K, N))
    d = pd.DataFrame({"gercek": cls[t], "tahmin_ours": cls[ours], "tahmin_raw": cls[raw]})
    if M > 1:
        d.insert(0, "madde", [f"q{i % M}" for i in range(N)])
    return sema.hazirla(d, gorev="cok_sinif")


class PinnedOutputTest(unittest.TestCase):
    def test_small_inputs_unchanged(self):
        for ad, (df, gorev, nl) in _vakalar().items():
            with self.subTest(vaka=ad):
                o = _calistir(df, gorev, nl)
                for (v, c, k), satir in _PIN_SATIR.items():
                    if v == ad:
                        self.assertEqual([(r["n"], r["carpan"], r["alt"], r["ust"], r["n_esdeger"])
                                          for r in o["yontemler"][c][k]], satir)
                for (v, c), ys in _PIN_YALNIZ.items():
                    if v == ad:
                        y = o["yontemler"][c]["yalniz_sim"]
                        self.assertEqual((y["esdeger_n"], y["alt"], y["ust"]), ys)
                self.assertEqual(_hash(o), _PIN[ad])

    def test_labels_match_one_hot_vectors(self):
        # the same predictions as label codes and as one-hot probability vectors take different code paths
        rng = np.random.default_rng(11)
        for M, N, K in ((1, 300, 7), (22, 60, 4)):
            cls = np.array([f"c{i}" for i in range(K)])
            t = rng.integers(0, K, M * N)
            f = np.where(rng.random(M * N) < .5, t, rng.integers(0, K, M * N))
            d = pd.DataFrame({"gercek": cls[t], "tahmin_et": cls[f], "tahmin_vek": list(np.eye(K)[f])})
            if M > 1:
                d.insert(0, "madde", [f"m{i % M}" for i in range(M * N)])
            v = sema.hazirla(d, gorev="cok_sinif")
            a = ok.carpan(v, {"tahmin_et": "e"}, n_listesi=(10, 25), R=40, tohum=1, B=100)["yontemler"]["tahmin_et"]
            b = ok.carpan(v, {"tahmin_vek": "v"}, n_listesi=(10, 25), R=40, tohum=1, B=100)["yontemler"]["tahmin_vek"]
            for k in ("ppi", "bayes"):
                for ra, rb in zip(a[k], b[k]):
                    for alan in ("carpan", "alt", "ust"):
                        self.assertAlmostEqual(ra[alan], rb[alan], places=3, msg=(M, k, alan))
            for alan in ("esdeger_n", "alt", "ust"):
                self.assertAlmostEqual(a["yalniz_sim"][alan], b["yalniz_sim"][alan], places=1, msg=(M, alan))

    def test_labelled_draw_matches_argsort(self):
        # chunked draw + argpartition = argsort(rng.random((R, N)))[:, :n] on the same generator state
        for R, N, n in ((7, 50, 10), (40, 1000, 200), (3, 25, 5)):
            a = np.argsort(np.random.default_rng(5).random((R, N)), axis=1)[:, :n]
            rng = np.random.default_rng(5)
            b = ok._etiketli(rng, R, N, n, parca=2 * N + 1)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(rng.random(), np.random.default_rng(5).random(R * N + 1)[-1])


class LargeFrameGuardTest(unittest.TestCase):
    """Dense version (567c6b2) on these frames: peak 2.48 GB / 14.8 s (single item), 1.74 GB / 4.9 s (three items);
    streaming: 60 MB / 0.3 s and 80 MB / 1.0 s."""

    def _olc(self, veri):
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            o = ok.carpan(veri, {"tahmin_ours": "ours", "tahmin_raw": "raw"}, tohum=0)
            sure = time.perf_counter() - t0
            tepe = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        return o, sure, tepe

    def test_single_item_many_classes(self):
        o, sure, tepe = self._olc(_cok_sinif_buyuk(8000, 300, 1))
        self.assertEqual(o["R"], 400)
        self.assertTrue(o["yontemler"]["tahmin_ours"]["ppi"])
        self.assertLess(tepe, 200e6)
        self.assertLess(sure, 10)

    def test_few_items_many_classes(self):
        o, sure, tepe = self._olc(_cok_sinif_buyuk(30000, 600, 3))
        self.assertEqual(len(o["yontemler"]["tahmin_raw"]["ppi"]), 5)
        self.assertLess(tepe, 200e6)
        self.assertLess(sure, 10)


if __name__ == "__main__":
    unittest.main()
