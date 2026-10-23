"""simengine.persona.latent tasks / heads / encoders / selection / scoring on synthetic data (no model, no download)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests._latent_util import CATS, fake_root


class TestTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = fake_root(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_g1_split_and_guard(self):
        from simengine.persona.latent import LeakError, load_g1
        g = load_g1(self.root, 0)
        fit, val, test = g.fit_rows(), g.val_rows(), g.test_rows()
        self.assertFalse(set(fit.uid) & set(test.uid))
        self.assertFalse(set(fit.uid) & set(val.uid))
        self.assertEqual(len(test), 10 * len(CATS))
        guard = g.guard()
        guard.check(fit, "fit")                                       # passes
        with self.assertRaises(LeakError):
            guard.check(g.target, "all rows")                         # test rows
        with self.assertRaises(LeakError):
            guard.check(fit.assign(bolum="train").head(3).assign(uid=test.uid.iloc[0]), "relabelled")  # test uid
        with self.assertRaises(LeakError):
            guard.check_uids(list(fit.uid[:2]) + [test.uid.iloc[0]], "ids")
        # history is keyed by each person's own cut
        k = g.persons().set_index("uid")["kesim"]
        h = g.history.drop_duplicates("uid").set_index("uid")["kesim"]
        self.assertTrue((h == k.reindex(h.index)).all())

    def test_caps_deterministic(self):
        from simengine.persona.latent import load_g1
        a, b = load_g1(self.root, 0, cap_fit=10, cap_val=3), load_g1(self.root, 0, cap_fit=10, cap_val=3)
        self.assertEqual(a.fit_rows().uid.nunique(), 10)
        self.assertEqual(a.val_rows().uid.nunique(), 3)
        self.assertEqual(sorted(a.fit_rows().uid.unique()), sorted(b.fit_rows().uid.unique()))
        self.assertEqual(a.test_rows().uid.nunique(), 10)

    def test_write_pred_checks(self):
        from simengine.persona.latent import write_pred
        g = __import__("simengine.persona.latent", fromlist=["x"]).load_g1(self.root, 0)
        t = g.test_rows().rename(columns={"oge": "kategori"}).assign(p=0.3)
        with tempfile.TemporaryDirectory() as d:
            path = write_pred(d, "enc-an", 0, t, {"model": None}, n_expected=len(t))
            self.assertTrue(path.exists() and path.with_suffix(".meta.json").exists())
            with self.assertRaises(ValueError):
                write_pred(d, "x", 0, t.iloc[1:], {}, n_expected=len(t))      # a missing row
            with self.assertRaises(ValueError):
                write_pred(d, "x", 0, t.assign(p=np.nan), {}, n_expected=len(t))
            with self.assertRaises(ValueError):
                write_pred(d, "x", 0, t.assign(p=1.5), {}, n_expected=len(t))

    def test_g2_g3_loaders(self):
        from simengine.persona.latent import load_g2, load_g3
        g2 = load_g2(self.root)
        self.assertEqual(len(g2.option_items()), 1)
        g3 = load_g3(self.root)
        self.assertTrue((g3.fit_rows().carpan <= 1.0).all())
        self.assertTrue((g3.val_rows().carpan <= 1.0).all())
        self.assertGreater((g3.test_rows().carpan > 1.0).sum(), 0)


class TestEncodersHeads(unittest.TestCase):
    def test_onehot_unseen_and_names(self):
        from simengine.persona.latent import SurveyOneHot
        a = pd.DataFrame({"uid": [1, 2], "x": ["a", "b"], "f": [0, 1]})
        enc = SurveyOneHot().fit(a)
        X = enc.transform(pd.DataFrame({"uid": [3], "x": ["zzz"], "f": [1]}))
        self.assertEqual(X.shape, (1, 4))
        self.assertEqual(X.sum(), 1.0)                              # the unseen 'zzz' is all zeros

    def test_history_features_shape_and_order(self):
        from simengine.persona.latent import history_features
        h = pd.DataFrame({"uid": ["a", "a", "b"], "kesim": "2022-01-01", "oge": ["C1", "C2", "C1"],
                          "k_365": [1, 2, 3], "u_k_365": [5, 5, 7]})
        X = history_features(h, ["b", "a", "zz"], ["C1", "C2"])
        self.assertEqual(X.shape, (3, 3))
        np.testing.assert_allclose(X[0], np.log1p([3, 0, 7]), rtol=1e-6)
        np.testing.assert_allclose(X[2], 0)

    def test_als_and_bag_deterministic_fold_in(self):
        import scipy.sparse as sp
        from simengine.persona.latent import BagSVD, ImplicitALS
        rng = np.random.default_rng(0)
        M = sp.csr_matrix(rng.poisson(0.5, (30, 12)).astype(float))
        a = ImplicitALS(dim=4, iters=3, seed=1).fit(M[:20]).transform(M)
        b = ImplicitALS(dim=4, iters=3, seed=1).fit(M[:20]).transform(M)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (30, 4))
        s = BagSVD(dim=3, seed=0).fit(M[:20]).transform(M)
        self.assertEqual(s.shape, (30, 3))

    def test_heads_seed_determinism_and_signal(self):
        from simengine.persona.latent import EncoderHead, Probe, kisi_auc
        rng = np.random.default_rng(0)
        X = rng.normal(size=(600, 6)).astype(np.float32)
        rows = np.repeat(np.arange(600), 3)
        cat = np.tile(np.arange(3), 600)
        y = (rng.random(1800) < 1 / (1 + np.exp(-(X[rows, 0] * (cat - 1) * 4)))).astype(int)
        fit, val = rows < 450, rows >= 450
        for make in (lambda: EncoderHead("logistic", C=1.0, seed=0), lambda: EncoderHead("mlp", seed=3, epochs=40, lr=1e-2),
                     lambda: Probe("logistic", pca=4, seed=0), lambda: Probe("massmean", pca=4, seed=0)):
            h1 = make().fit(X, rows[fit], cat[fit], y[fit], val=(X, rows[val], cat[val], y[val]))
            h2 = make().fit(X, rows[fit], cat[fit], y[fit], val=(X, rows[val], cat[val], y[val]))
            p1, p2 = h1.predict(X, rows[val], cat[val]), h2.predict(X, rows[val], cat[val])
            np.testing.assert_array_equal(p1, p2)
            self.assertGreater(kisi_auc(p1, y[val], rows[val]), 0.6)

    def test_head_statistics_ignore_unused_persons(self):
        from simengine.persona.latent import EncoderHead, Probe
        rng = np.random.default_rng(2)
        X = rng.normal(size=(60, 5)).astype(np.float32)
        rows, cat = np.repeat(np.arange(40), 2), np.tile([0, 1], 40)
        y = rng.integers(0, 2, 80)
        X2 = X.copy()
        X2[40:] += 1000.0                                           # persons outside the fit rows (test) change
        for make in (lambda: EncoderHead("logistic", seed=0), lambda: Probe("logistic", pca=3, seed=0)):
            a = make().fit(X, rows, cat, y).predict(X, rows, cat)
            b = make().fit(X2, rows, cat, y).predict(X2, rows, cat)
            np.testing.assert_allclose(a, b, rtol=1e-6)

    def test_platt_and_stack(self):
        from simengine.persona.latent import platt, stack_lgbm
        rng = np.random.default_rng(1)
        p = rng.random(400)
        y = (rng.random(400) < p ** 2).astype(int)
        f = platt(p, y, np.arange(400) % 2)
        q = f(p, np.arange(400) % 2)
        self.assertTrue(((q > 0) & (q < 1)).all())
        F = rng.normal(size=(600, 4))
        yy = (F[:, 0] + rng.normal(size=600) * 0.5 > 0).astype(int)
        b, info = stack_lgbm(F[:400], yy[:400], F[400:], yy[400:], seed=0, params={"min_data_in_leaf": 5})
        self.assertGreater(info["rounds"], 0)
        self.assertEqual(len(b.predict(F[:5])), 5)

    def test_trait_vector(self):
        from simengine.persona.latent import TRAIT_EDITS_G1, edit_profile, trait_person_vector, trait_values
        s = pd.DataFrame({t: [e["levels"][0], e["levels"][-1]] for t, e in TRAIT_EDITS_G1.items()})
        s["hd_cocuk"] = [0, 1]
        v = trait_values(s)
        self.assertEqual(v.shape, (2, 8))
        self.assertTrue((v[0] == 0).all())
        dirs = {t: np.eye(8, dtype=np.float32)[i] for i, t in enumerate(TRAIT_EDITS_G1)}
        P = trait_person_vector(v, dirs)
        np.testing.assert_allclose(P[0], -P[1], atol=1e-6)
        self.assertEqual(edit_profile("- Gender: Female\n- Age group: x", "Gender", "Male"),
                         "- Gender: Male\n- Age group: x")
        with self.assertRaises(ValueError):
            edit_profile("- Gender: Female", "Income", "x")


class TestSelectEval(unittest.TestCase):
    def test_kisi_auc_matches_pairs(self):
        from simengine.persona.latent import kisi_auc
        rng = np.random.default_rng(0)
        u = np.repeat(np.arange(30), 6)
        y = rng.integers(0, 2, 180)
        p = np.round(rng.random(180), 1)                            # ties
        ref = []
        for k in range(30):
            m = u == k
            pos, neg = p[m & (y == 1)], p[m & (y == 0)]
            if len(pos) and len(neg):
                ref.append(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))
        self.assertAlmostEqual(kisi_auc(p, y, u), float(np.mean(ref)), places=10)

    def test_choose_validation_tiebreak_and_cost(self):
        from simengine.persona.latent import choose, grid
        cfgs = grid({"a": [1, 2, 3], "cost": [2, 1]})
        self.assertEqual(len(cfgs), 6)
        best, table = choose(cfgs, lambda c: {"kisi_auc": 0.7 if c["a"] >= 2 else 0.6, "1a_dogruluk_a": 0.8},
                             cost_key="cost")
        self.assertEqual(best, {"a": 2, "cost": 1})
        self.assertEqual(int(table.chosen.sum()), 1)

    def test_score_g3_linear_demand(self):
        from simengine.persona.latent import score_g3
        rng = np.random.default_rng(0)
        lv = np.round(np.arange(0, 2.01, 0.2), 1)
        sweep = pd.DataFrame([(u, "P", l, 0.9 - 0.3 * l) for u in range(50) for l in lv],
                             columns=["uid", "oge", "carpan_duzey", "p"])
        truth = pd.DataFrame([(u, "P", l, int(rng.random() < 0.9 - 0.3 * l)) for u in range(50) for l in lv
                              for _ in range(4)], columns=["uid", "oge", "carpan_duzey", "y"])
        r = score_g3(sweep, truth, B=200)
        self.assertAlmostEqual(r["s_hi"], -0.3, places=6)
        self.assertLess(r["s_hi_ci"][1], 0)
        self.assertTrue(0.5 < r["ratio"] < 1.5)
        self.assertTrue(r["D5"])
        flat = sweep.assign(p=0.5)
        self.assertFalse(score_g3(flat, truth, B=50)["D5"])

    def test_score_g2_and_decide(self):
        from simengine.persona.latent import decide, score_g2
        truth = pd.DataFrame({"uid": [1, 1, 2, 2], "oge": ["a", "b", "a", "b"], "y01": [0, 1, 1, 1.0]})
        pred = truth.rename(columns={"y01": "p"})
        r = score_g2(pred, truth, base=pred.assign(p=0.5), B=100)
        self.assertEqual(r["mad_acc"], 1.0)
        self.assertGreater(r["diff"], 0)
        self.assertEqual(decide([(0.01, 0.02)] * 3, "D1")["verdict"], "pass")
        self.assertEqual(decide([(0.01, 0.02), (-0.01, 0.02)], "D2")["verdict"], "fail")
        d1 = decide([(0.01, 0.02)] * 3, "D1", acc_cis=[(-0.01, 0.01)] * 3)
        self.assertEqual((d1["verdict"], d1["label"]), ("pass", "adds ranking signal"))
        self.assertEqual(decide([(-0.005, 0.005)] * 3, "D4")["verdict"], "equivalent")


if __name__ == "__main__":
    unittest.main()
