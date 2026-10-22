"""Rehearsal gaps in evidence metrics and audits: calibration-in-the-large (mean_bias), positional Wasserstein,
binary-aware spread (Bernoulli sd) and imbalance audits, distribution entropy ratio as a ratio of means."""
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine as se
from simengine.bench import stats as bstats
from simengine.evidence import analiz, api, denetim, metrik, sema


def binary_frame(n=600, base=0.2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p_true = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + 1.2 * x)))
    y = (rng.random(n) < p_true).astype(int)
    return pd.DataFrame({"group": [f"u{i}" for i in range(n)], "truth": y,
                         "pred_ours": np.clip(p_true + 0.05, 0, 1),          # over-predicts the rate by ~0.05
                         "pred_raw": np.clip(0.5 + 0.1 * x, 0, 1)})


def distribution_frame(pos=None, opts=None, seed=1, n_items=15, zero_entropy=False):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_items):
        for s in range(3):
            t = rng.dirichlet(np.ones(4) * 2)
            if zero_entropy and s == 0:
                t = np.array([1.0, 0, 0, 0])                  # small group, truth entropy 0
            r = {"item": f"q{m}", "segment": f"s{s}", "truth": t.tolist(),
                 "pred_ours": (0.7 * t + 0.3 * rng.dirichlet(np.ones(4))).tolist(),
                 "pred_raw": rng.dirichlet(np.ones(4)).tolist()}
            if pos is not None:
                r["pos"] = list(pos)
            if opts is not None:
                r["options"] = list(opts)
            rows.append(r)
    return pd.DataFrame(rows)


def _ev(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return se.evaluate(df, "t", ours="ours", raw="raw", B=200, B_aggregate=40, **kw)


def _audit(rep, prefix):
    return next(d for d in rep["denetim"] if d["kontrol"].startswith(prefix))


class TestMeanBias(unittest.TestCase):
    def test_binary_card_row_with_ci(self):
        df = binary_frame()
        rep = _ev(df)
        m = rep["sonuclar"]["tahmin_ours"]["metrikler"]["mean_bias"]
        self.assertAlmostEqual(m["deger"], df.pred_ours.mean() - df.truth.mean(), places=12)
        self.assertLess(m["alt"], m["deger"])
        self.assertGreater(m["ust"], m["deger"])
        row = next(r for r in rep["metrikler"] if r["ad"] == "mean_bias")
        self.assertEqual(row["yon"], 0)                     # signed: closer to 0 is better

    def test_paired_ci_metric_mean_bias(self):
        df = binary_frame()
        r = api.paired_ci(df, "ours", "raw", metric="mean_bias", B=200)
        self.assertEqual(r["metric"], "mean_bias")
        self.assertAlmostEqual(r["a"], df.pred_ours.mean() - df.truth.mean(), places=12)
        self.assertAlmostEqual(r["diff"], df.pred_ours.mean() - df.pred_raw.mean(), places=12)
        self.assertEqual(r["a_better"], abs(r["a"]) < abs(r["b"]))

    def test_continuous_family(self):
        fam = {m.ad: m for m in metrik.aile("surekli")}
        y, p = np.array([1.0, 2, 3]), np.array([1.5, 2.5, 2.0])
        self.assertAlmostEqual(fam["mean_bias"].deger(y, p), p.mean() - y.mean())
        self.assertIn("mean_bias", metrik.ISARETLI_METRIKLER)


class TestPositionalWasserstein(unittest.TestCase):
    def test_matches_bench_wd_and_equal_spacing(self):
        rng = np.random.default_rng(3)
        pos = np.array([0, 0.2, 0.3, 1.0])
        for _ in range(20):
            a, b = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4))
            self.assertAlmostEqual(metrik.wasserstein_konumlu(a, b, pos), bstats.wasserstein_ordinal(a, b, pos))
            self.assertAlmostEqual(metrik.wasserstein_konumlu(a, b), metrik.wasserstein_sirali(a, b))
            # positions out of order and on another scale: sorted and rescaled to 0-1
            self.assertAlmostEqual(metrik.wasserstein_konumlu(a[::-1], b[::-1], 10 * pos[::-1]),
                                   bstats.wasserstein_ordinal(a, b, pos))

    def test_evaluate_uses_pos_column(self):
        pos = [0, 0.1, 0.2, 1.0]
        rep = _ev(distribution_frame(pos=pos), primary="wasserstein")
        rep_eq = _ev(distribution_frame(), primary="wasserstein")
        d = distribution_frame()
        want = np.mean([bstats.wasserstein_ordinal(np.array(t), np.array(p), np.array(pos))
                        for t, p in zip(d.truth, d.pred_ours)])
        eq = np.mean([metrik.wasserstein_sirali(np.array(t), np.array(p)) for t, p in zip(d.truth, d.pred_ours)])
        self.assertAlmostEqual(rep["sonuclar"]["tahmin_ours"]["metrikler"]["wasserstein"]["deger"], want, places=12)
        self.assertAlmostEqual(rep_eq["sonuclar"]["tahmin_ours"]["metrikler"]["wasserstein"]["deger"], eq, places=12)
        self.assertNotAlmostEqual(want, eq, places=3)

    def test_option_sources(self):
        def kon(df, **kw):
            return metrik.secenek_konumlari(sema.hazirla(api.to_internal(df), "dagilim", **kw))
        self.assertIsNone(kon(distribution_frame(opts=["a", "b", "c", "d"])))     # text options: equal spacing
        self.assertIsNone(kon(distribution_frame(opts=[1, 2, 3, 4])))             # equally spaced: no change
        k = kon(distribution_frame(opts=["1", "2", "4", "8"]))
        np.testing.assert_allclose(k[0], [1, 2, 4, 8])
        k = kon(distribution_frame(), siniflar=[0, 1, 2, 10])
        np.testing.assert_allclose(k[0], [0, 1, 2, 10])
        # the pos column wins over text options
        k = kon(distribution_frame(pos=[0, 0.5, 0.6, 1], opts=["a", "b", "c", "d"]))
        np.testing.assert_allclose(k[0], [0, 0.5, 0.6, 1])


class TestBinarySpread(unittest.TestCase):
    def test_bernoulli_sd(self):
        df = binary_frame()
        rep = _ev(df)
        y = rep["yayilim"]["tahmin_ours"]
        self.assertIsNone(y["sd_orani"])
        self.assertEqual(y["sd_olcut"], "bernoulli")
        p = df.pred_ours.to_numpy(float)
        self.assertAlmostEqual(y["sd_bernoulli"], p.std() / np.sqrt(p.mean() * (1 - p.mean())), places=12)
        self.assertLessEqual(y["sd_bernoulli_alt"], y["sd_bernoulli"])
        a = _audit(rep, "dağılım çöküşü")
        self.assertIn("Bernoulli", a["kontrol"])
        self.assertEqual(a["durum"], "tamam")

    def test_constant_probability_warns(self):
        y = {"x": {"sd_bernoulli": 0.01, "sd_bernoulli_alt": 0.0, "sd_bernoulli_ust": 0.02}}
        self.assertEqual(denetim.cokus(y, "x")["durum"], "uyari")

    def test_hard_labels_keep_sd_ratio(self):
        df = binary_frame()
        df["pred_ours"] = (df.pred_ours > 0.5).astype(float)
        df["pred_raw"] = (df.pred_raw > 0.5).astype(float)
        y = _ev(df)["yayilim"]["tahmin_ours"]
        self.assertIsNotNone(y["sd_orani"])
        self.assertNotIn("sd_bernoulli", y)


class TestImbalance(unittest.TestCase):
    def test_binary_probability_is_note(self):
        rep = _ev(binary_frame(base=0.1), primary="auc")
        a = _audit(rep, "sınıf dengesizliği")
        self.assertEqual(a["durum"], "tamam")
        self.assertGreater(a["deger"], 0.7)

    def test_hard_labels_and_explicit_primary(self):
        df = binary_frame(base=0.1)
        df["pred_ours"] = (df.pred_ours > 0.5).astype(float)
        df["pred_raw"] = (df.pred_raw > 0.5).astype(float)
        v = sema.hazirla(api.to_internal(df))
        self.assertEqual(denetim.dengesizlik(v)["durum"], "uyari")
        self.assertEqual(denetim.dengesizlik(v, birincil="kappa")["durum"], "tamam")
        vp = sema.hazirla(api.to_internal(binary_frame(base=0.1)))
        self.assertEqual(denetim.dengesizlik(vp, birincil="dogruluk")["durum"], "uyari")
        self.assertEqual(denetim.dengesizlik(vp)["durum"], "tamam")


class TestDistributionEntropy(unittest.TestCase):
    def test_ratio_of_means_with_ci(self):
        df = distribution_frame(zero_entropy=True)
        rep = _ev(df)
        y = rep["yayilim"]["tahmin_ours"]
        h = lambda v: analiz._entropi(np.asarray(v, float))
        want = np.mean([h(v) for v in df.pred_ours]) / np.mean([h(v) for v in df.truth])
        self.assertAlmostEqual(y["entropi_orani"], want, places=12)
        self.assertEqual(y["entropi_olcut"], "ortalamalarin_orani")
        self.assertLess(y["entropi_orani"], 5)                   # per-cell ratios would explode on H(truth) = 0
        self.assertTrue(y["entropi_orani_alt"] <= y["entropi_orani"] <= y["entropi_orani_ust"])
        self.assertIsNotNone(y["ayrisma_orani"])
        a = _audit(rep, "dağılım çöküşü")
        self.assertIn("ortalamaların oranı", a["ayrinti"])
        self.assertIn("ayrışma", a["ayrinti"])


if __name__ == "__main__":
    unittest.main()
