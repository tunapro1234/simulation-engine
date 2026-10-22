"""risky_choice: gamble distributions, value models, BEAST simulator sanity, metrics and both loaders on tiny synthetic
files in the documented layout. The real-data check (BEAST.sd on the CPC18 competition set) skips without
$SIMENGINE_DATA."""
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from simengine.bench import base
from simengine.bench.tasks import risky_choice as R


def _ev(d):
    return float(np.sum(d[0] * d[1]))


class TestDistributions(unittest.TestCase):
    def test_merge_dist_sorts_merges_and_drops_zero(self):
        x, p = R.merge_dist([5, -1, 5, 3], [0.2, 0.3, 0.5, 0.0])
        np.testing.assert_allclose(x, [-1, 5])
        np.testing.assert_allclose(p, [0.3, 0.7])

    def test_cpc_shapes_keep_expected_value_and_mass(self):
        for shape, num in [("-", 1), ("Symm", 3), ("Symm", 7), ("R-skew", 4), ("L-skew", 5), (2, 6), (3, 2)]:
            d = R.cpc_distribution(20.0, 0.4, -3.0, shape, num)
            self.assertAlmostEqual(d[1].sum(), 1.0, places=12)
            self.assertAlmostEqual(_ev(d), 0.4 * 20 + 0.6 * -3, places=9, msg=(shape, num))
            self.assertTrue(np.all(np.diff(d[0]) > 0))

    def test_cpc_symm_is_binomial(self):
        x, p = R.cpc_distribution(10, 1.0, 0, "Symm", 3)
        np.testing.assert_allclose(x, [9, 10, 11])
        np.testing.assert_allclose(p, [0.25, 0.5, 0.25])

    def test_cpc_rskew_outcomes(self):
        # num 3, c = -4: H - 4 + 2, 4, 8 with weights 1/2, 1/4, 1/4 (last doubled)
        x, p = R.cpc_distribution(0, 1.0, 0, "R-skew", 3)
        np.testing.assert_allclose(x, [-2, 0, 4])
        np.testing.assert_allclose(p, [0.5, 0.25, 0.25])
        with self.assertRaises(ValueError):
            R.cpc_distribution(0, 1.0, 0, "weird", 3)

    def test_pad_keeps_order_and_cdf(self):
        X, P, n = R.pad_dists([R.merge_dist([1, 2], [0.5, 0.5]), R.merge_dist([0, 1, 9], [0.2, 0.3, 0.5])])
        self.assertEqual(X.shape, (2, 3))
        np.testing.assert_array_equal(n, [2, 3])
        self.assertTrue(np.all(np.diff(X, axis=1) >= 0))
        np.testing.assert_allclose(P.sum(1), 1.0)

    def test_p_better_comonotone_matches_monte_carlo(self):
        da = [R.merge_dist([0, 10], [0.5, 0.5]), R.cpc_distribution(5, 0.7, -2, "Symm", 5)]
        db = [R.merge_dist([3, 4, 8], [0.2, 0.3, 0.5]), R.merge_dist([1], [1.0])]
        XA, PA, nA, XB, PB, nB = R._pad_pair(da, db)
        pa, pb = R.p_better_comonotone(XA, PA, XB, PB)
        u = np.random.default_rng(0).random(200_000)
        for r in range(2):
            a = XA[r, R._sample_idx(np.cumsum(PA[r:r + 1], 1), u[None, :], nA[r:r + 1])[0]]
            b = XB[r, R._sample_idx(np.cumsum(PB[r:r + 1], 1), u[None, :], nB[r:r + 1])[0]]
            self.assertAlmostEqual(pa[r], np.mean(a > b), delta=0.005)
            self.assertAlmostEqual(pb[r], np.mean(b > a), delta=0.005)

    def test_fosd(self):
        da = [R.merge_dist([0, 10], [0.5, 0.5]), R.merge_dist([0, 10], [0.5, 0.5]), R.merge_dist([5], [1.0])]
        db = [R.merge_dist([0, 10], [0.6, 0.4]), R.merge_dist([0, 10], [0.5, 0.5]), R.merge_dist([0, 10], [.5, .5])]
        XA, PA, _, XB, PB, _ = R._pad_pair(da, db)
        np.testing.assert_array_equal(R.fosd(XA, PA, XB, PB), [True, False, False])


class TestValueModels(unittest.TestCase):
    def setUp(self):
        self.X, self.P, _ = R.pad_dists([R.merge_dist([-5, 0, 10], [0.2, 0.3, 0.5]), R.merge_dist([2], [1.0]),
                                         R.cpc_distribution(30, 0.1, -1, "L-skew", 4)])

    def test_linear_prospect_value_is_expected_value(self):
        ev = R.expected_value(self.X, self.P)
        for cum in (True, False):
            np.testing.assert_allclose(R.prospect_value(self.X, self.P, 1, 1, 1, 1, cum), ev, atol=1e-12)

    def test_cpt_two_outcome_gain_by_hand(self):
        X, P, _ = R.pad_dists([R.merge_dist([0, 100], [0.7, 0.3])])
        w = 0.8 * 0.3 ** 0.6 / (0.8 * 0.3 ** 0.6 + 0.7 ** 0.6)
        self.assertAlmostEqual(float(R.prospect_value(X, P, 0.9, 2.0, 0.6, 0.8)[0]), w * 100 ** 0.9, places=9)

    def test_cpt_mixed_loss_side_uses_cumulative_from_below(self):
        X, P, _ = R.pad_dists([R.merge_dist([-10, -2], [0.25, 0.75])])
        g, d, a, lam = 0.7, 0.9, 0.8, 2.0
        w = lambda q: d * q ** g / (d * q ** g + (1 - q) ** g)  # noqa: E731
        want = -lam * (w(0.25) * 10 ** a + (1 - w(0.25)) * 2 ** a)
        self.assertAlmostEqual(float(R.prospect_value(X, P, a, lam, g, d)[0]), want, places=9)

    def test_gw_weight_endpoints(self):
        np.testing.assert_allclose(R.gw_weight(np.array([0.0, 1.0, 0.5]), 0.5, 1.0), [0.0, 1.0, 0.5])

    def test_fit_prospect_recovers_generating_model(self):
        rng = np.random.default_rng(0)
        n = 300
        da = [R.merge_dist(np.round(rng.uniform(-20, 40, 2)), [p, 1 - p]) for p in rng.uniform(0.05, 0.95, n)]
        db = [R.merge_dist([v], [1.0]) for v in np.round(rng.uniform(-10, 25, n))]
        XA, PA, _, XB, PB, _ = R._pad_pair(da, db)
        true = {"alpha": 0.85, "lam": 1.8, "gamma": 0.7, "delta": 0.9, "mu": 0.3, "cumulative": True}
        y = R.predict_prospect(true, XA, PA, XB, PB)
        fit = R.fit_prospect(XA, PA, XB, PB, y, cumulative=True)
        self.assertLess(fit["train_mse"], 1e-4)
        self.assertAlmostEqual(fit["lam"], 1.8, delta=0.3)

    def test_ev_logit_fit(self):
        ev_a, ev_b = np.zeros(200), np.linspace(-10, 10, 200)
        y = R.logit_b(ev_a, ev_b, 0.4)
        self.assertAlmostEqual(R.fit_ev_logit(ev_a, ev_b, y), 0.4, places=3)


class TestBeast(unittest.TestCase):
    def test_shape_seed_and_support_dominance(self):
        da = [R.merge_dist([10], [1.0]), R.merge_dist([0, 10], [0.5, 0.5])]
        db = [R.merge_dist([0, 5], [0.5, 0.5]), R.merge_dist([0, 10], [0.5, 0.5])]
        for v in ("sd", "original"):
            a = R.beast_predict(da, db, version=v, n_sims=400, seed=1)
            self.assertEqual(a.shape, (2, 5))
            np.testing.assert_array_equal(a, R.beast_predict(da, db, version=v, n_sims=400, seed=1))
            self.assertEqual(a[0].max(), 0.0)  # min A > max B: nobody chooses B
            self.assertAlmostEqual(a[1].mean(), 0.5, delta=0.05)  # identical options

    def test_symmetric_problems_mirror(self):
        a, b = R.merge_dist([0, 20], [0.6, 0.4]), R.merge_dist([7], [1.0])
        ab = R.beast_predict([a], [b], n_sims=4000, seed=0)
        ba = R.beast_predict([b], [a], n_sims=4000, seed=1)
        np.testing.assert_allclose(ab + ba, 1.0, atol=0.04)

    def test_ambiguity_and_correlation_run(self):
        da = [R.merge_dist([3], [1.0])] * 2
        db = [R.merge_dist([0, 10], [0.5, 0.5])] * 2
        out = R.beast_predict(da, db, amb=[1, 0], corr=[0, -1], n_sims=300, chunk=1)
        self.assertTrue(np.all((out >= 0) & (out <= 1)))


def _c13k_files(root, n=60, seed=0):
    rng = np.random.default_rng(seed)
    d = os.path.join(root, "choices13k", "ham")
    os.makedirs(d)
    rows, probs = [], {}
    for i in range(n):
        ha, la, hb, lb = rng.integers(0, 40), rng.integers(-20, 0), rng.integers(0, 50), rng.integers(-30, 5)
        pha, phb = rng.choice([0.1, 0.5, 0.9]), rng.choice([0.2, 0.6, 1.0])
        shape, num = (0, 1) if i % 3 else (1, 3)
        b = R.cpc_distribution(hb, phb, lb, shape, num)
        evd = (pha * ha + (1 - pha) * la) - (phb * hb + (1 - phb) * lb)
        rate = float(np.clip(1 / (1 + np.exp(0.2 * evd)) + rng.normal(0, 0.05), 0, 1))
        rows.append({"Problem": i // 2, "Feedback": bool(i % 4), "n": 15, "Block": 1 + i % 5, "Ha": ha, "pHa": pha,
                     "La": la, "Hb": hb, "pHb": phb, "Lb": lb, "LotShapeB": shape, "LotNumB": num,
                     "Amb": i % 7 == 0, "Corr": [0, 1, -1][i % 3], "bRate": rate, "bRate_std": 0.3})
        probs[str(i)] = {"A": [[float(pha), float(ha)], [float(1 - pha), float(la)]],
                         "B": [[float(p), float(x)] for x, p in zip(*b)]}
    pd.DataFrame(rows).to_csv(os.path.join(d, "c13k_selections.csv"), index=False)
    with open(os.path.join(d, "c13k_problems.json"), "w") as f:
        json.dump(probs, f)
    return pd.DataFrame(rows)


def _cpc_files(root, seed=0):
    rng = np.random.default_rng(seed)
    d = os.path.join(root, "cpc18", "ham")
    os.makedirs(d)
    rows = []
    for g in range(1, 28):
        st = 1 + (g - 1) // 3  # sets 1..9, three games each
        ha, hb = int(rng.integers(5, 40)), int(rng.integers(0, 50))
        prm = dict(Set=st, GameID=g, Ha=ha, pHa=0.5, La=0, LotShapeA="-", LotNumA=1, Hb=hb, pHb=1.0 if g % 2 else 0.3,
                   Lb=-5, LotShapeB="Symm" if g % 4 == 0 else "-", LotNumB=3 if g % 4 == 0 else 1, Amb=int(g % 5 == 0),
                   Corr=0)
        for s in range(4):
            for t in range(1, 26):
                rows.append({**prm, "SubjID": 100 * g + s, "Trial": t, "block": 1 + (t - 1) // 5,
                             "B": int(rng.random() < (0.7 if hb > ha else 0.3))})
    pd.DataFrame(rows).to_csv(os.path.join(d, "cpc18_all_raw.csv"), index=False)
    return pd.DataFrame(rows)


class TestTasks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_registered_and_options_checked(self):
        self.assertIs(base.get("choices13k").__class__, R.Choices13k)
        self.assertIs(base.get("cpc18").__class__, R.CPC18)
        with self.assertRaises(ValueError):
            R.Choices13k(bogus=1)

    def test_choices13k_load_run_score(self):
        src = _c13k_files(self.root)
        task = R.Choices13k(seed=3, test_size=0.25, beast_sims=200)
        data = task.load(self.root)
        keep = src[src.Feedback & ~src.Amb]
        self.assertEqual(len(data.train) + len(data.test), len(keep))
        self.assertEqual(len(data.test), round(0.25 * len(keep)))
        self.assertTrue(set(data.train.row).isdisjoint(data.test.row))
        np.testing.assert_allclose(data.test.se2, 0.3 ** 2 / 15)
        frame = task.run(data=data)
        for m in ("half", "train_mean", "ev_max", "ev_logit", "pt", "cpt", "gbdt_objective", "beast_sd"):
            v = frame["pred_" + m]
            self.assertTrue(v.between(0, 1).all(), m)
        tab = task.score_all(frame, B=50)
        self.assertEqual(set(tab.metric), {"mse", "completeness", "pref_acc"})
        half = tab[(tab.method == "half") & (tab.metric == "completeness")].value.iloc[0]
        self.assertAlmostEqual(half, 0.0)
        mse = tab[tab.metric == "mse"].set_index("method").value
        self.assertLess(mse["ev_logit"], mse["half"])

    def test_choices13k_all_subset_five_protocol_and_env(self):
        _c13k_files(self.root)
        old = os.environ.get(base.ENV_DATA)
        os.environ[base.ENV_DATA] = self.root
        try:
            task = R.Choices13k(subset="all", beast="both", beast_protocol="five", beast_sims=100, train_frac=0.5)
            data = task.load()
        finally:
            if old is None:
                os.environ.pop(base.ENV_DATA)
            else:
                os.environ[base.ENV_DATA] = old
        self.assertEqual(len(data.test), 6)
        self.assertEqual(len(data.train), 27)
        preds = task.baselines(data)
        self.assertIn("beast", preds)
        self.assertEqual(len(preds["beast_sd"]), 6)

    def test_cpc18_aggregate_and_splits(self):
        raw = _cpc_files(self.root)
        agg = R.CPC18.aggregate(raw)
        g1 = raw[(raw.GameID == 1) & (raw.block == 2)]
        per = g1.groupby("SubjID").B.mean()
        r = agg[(agg.GameID == 1) & (agg.block == 2)].iloc[0]
        self.assertAlmostEqual(r.y, per.mean())
        self.assertAlmostEqual(r.se2, per.var(ddof=1) / 4)
        self.assertEqual(r.n, 4)
        task = R.CPC18(beast_sims=200)
        data = task.load(self.root)
        self.assertEqual(data.meta["n_test_tasks"], 6)
        self.assertEqual(len(data.test), 30)
        frame = task.run(data=data)
        self.assertIn("pred_beast", frame)
        self.assertIn("pred_beast_sd", frame)
        self.assertEqual(frame.groupby("group").size().max(), 5)
        d15 = R.CPC18(split="cpc15", subset="risk_nofb", beast="none").load(self.root)
        self.assertTrue((d15.test.block == 1).all() and (d15.test.Amb == 0).all())
        self.assertTrue(d15.train.Set.le(3).all() and d15.test.Set.isin([4, 5]).all())
        with self.assertRaises(ValueError):
            R.CPC18(split="x").load(self.root)

    def test_missing_files(self):
        with self.assertRaises(base.DataNotFound):
            R.CPC18().load(self.root)


class TestMetrics(unittest.TestCase):
    def test_completeness_endpoints(self):
        y = np.array([0.1, 0.8, 0.6, 0.3])
        se2 = np.full(4, 0.001)
        self.assertAlmostEqual(R.completeness(y, np.full(4, 0.5), se2), 0.0)
        m_r = np.mean((0.5 - y) ** 2)
        self.assertAlmostEqual(R.completeness(y, y, se2), m_r / (m_r - 0.001))

    def test_pref_acc_ties_count_half(self):
        task = R.CPC18()
        f = pd.DataFrame({"truth": [0.7, 0.2, 0.5, 0.9], "group": [1, 2, 3, 4], "pred_x": [0.6, 0.6, 0.1, 0.5],
                          "se2": 0.0})
        self.assertAlmostEqual(task.score(f, "x", "pref_acc", B=20)["value"], (1 + 0 + 0.5 + 0.5) / 4)
        self.assertAlmostEqual(task.score(f, "x", "mse", B=20)["value"], np.mean((f.pred_x - f.truth) ** 2))


@unittest.skipUnless(os.environ.get(base.ENV_DATA), "needs $SIMENGINE_DATA with cpc18/")
class TestRealCPC18(unittest.TestCase):
    def test_beast_sd_close_to_published(self):
        task = R.CPC18(beast="sd", beast_sims=2000)
        try:
            data = task.load()
        except base.DataNotFound as e:
            self.skipTest(str(e))
        frame = task.frame(data, {"beast_sd": task._beast(data, "sd")})
        mse = task.score(frame, "beast_sd", B=50)["value"]
        self.assertLess(abs(mse - 0.00708), 0.001)  # Plonsky et al. Table S1: 100 x MSE = 0.708


if __name__ == "__main__":
    unittest.main()
