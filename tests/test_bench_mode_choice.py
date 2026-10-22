"""Swissmetro mode choice: data preparation, MNL / nested logit estimators on synthetic choices with known parameters,
metrics against hand computation, LLM prompt / parse helpers with FakeLLM, and the task on a synthetic file in the
documented layout. The Biogeme replication on the real file skips without $SIMENGINE_DATA."""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from simengine import bench
from simengine.bench.tasks import swissmetro as S
from simengine.llm.fake import FakeLLM
from simengine.types import Task, TaskData

COLS = ["GROUP", "SURVEY", "SP", "ID", "PURPOSE", "FIRST", "TICKET", "WHO", "LUGGAGE", "AGE", "MALE", "INCOME", "GA",
        "ORIGIN", "DEST", "TRAIN_AV", "CAR_AV", "SM_AV", "TRAIN_TT", "TRAIN_CO", "TRAIN_HE", "SM_TT", "SM_CO", "SM_HE",
        "SM_SEATS", "CAR_TT", "CAR_CO", "CHOICE"]
TRUE = {"asc_train": -0.7, "asc_car": -0.15, "b_time": -1.3, "b_cost": -1.1}


def synthetic_raw(n_resp=400, seed=0, theta=TRUE, nests=None, mu=2.0):
    """Swissmetro-layout table (9 situations per respondent) with choices drawn from a b01 logit (or nested logit)."""
    rng = np.random.default_rng(seed)
    n = n_resp * 9
    per = lambda a: np.repeat(a, 9)
    car_user = rng.random(n_resp) < 0.6
    d = pd.DataFrame({
        "GROUP": per(np.where(car_user, 3, 2)), "SURVEY": per(car_user.astype(int)), "SP": 1,
        "ID": per(np.arange(1, n_resp + 1)), "PURPOSE": per(rng.choice([1, 2, 3, 4], n_resp)),
        "FIRST": per(rng.integers(0, 2, n_resp)), "TICKET": per(rng.integers(0, 11, n_resp)),
        "WHO": per(rng.integers(0, 4, n_resp)), "LUGGAGE": per(rng.choice([0, 1, 3], n_resp)),
        "AGE": per(rng.integers(1, 7, n_resp)), "MALE": per(rng.integers(0, 2, n_resp)),
        "INCOME": per(rng.integers(0, 5, n_resp)), "GA": per((rng.random(n_resp) < 0.15).astype(int)),
        "ORIGIN": per(rng.integers(1, 27, n_resp)), "DEST": per(rng.integers(1, 27, n_resp)),
        "TRAIN_AV": 1, "CAR_AV": per((car_user | (rng.random(n_resp) < 0.3)).astype(int)), "SM_AV": 1,
        "TRAIN_TT": rng.integers(30, 250, n), "TRAIN_CO": rng.integers(10, 200, n),
        "TRAIN_HE": rng.choice([30, 60, 120], n), "SM_TT": rng.integers(10, 150, n),
        "SM_CO": rng.integers(10, 250, n), "SM_HE": rng.choice([10, 20, 30], n), "SM_SEATS": rng.integers(0, 2, n),
        "CAR_TT": rng.integers(30, 250, n), "CAR_CO": rng.integers(10, 200, n), "CHOICE": 1})
    p = S.prepare(d, "all")
    X, names = S.design(p, S.B01)
    th = np.array([theta[k] for k in names] + ([mu] if nests else []))
    lp = S._log_probs(X, S.availability(p), th, nests or {}, len(names))
    pr = np.exp(lp)
    u = rng.random(n)[:, None]
    d["CHOICE"] = np.minimum((pr.cumsum(1) < u).sum(1), 2) + 1
    return d


class TestPrepare(unittest.TestCase):
    def setUp(self):
        self.raw = synthetic_raw(60, seed=1)

    def test_filter_and_derived(self):
        raw = self.raw.copy()
        raw.loc[0, "CHOICE"] = 0
        b, a = S.prepare(raw, "biogeme"), S.prepare(raw, "all")
        self.assertEqual(len(a), len(raw) - 1)
        self.assertEqual(len(b), int((raw.PURPOSE.isin([1, 3]) & (raw.CHOICE != 0)).sum()))
        ga = a[a.GA == 1]
        self.assertTrue(len(ga) > 0 and (ga.TRAIN_COST == 0).all() and (ga.SM_COST == 0).all())
        no = a[a.GA == 0]
        np.testing.assert_allclose(no.TRAIN_COST_SCALED, no.TRAIN_CO / 100)
        np.testing.assert_allclose(a.CAR_CO_SCALED, a.CAR_CO / 100)
        np.testing.assert_array_equal(a.y, a.CHOICE - 1)
        np.testing.assert_array_equal(a.av_car, a.CAR_AV)
        self.assertEqual(a.groupby("respondent").situation.max().max(), 8)
        with self.assertRaises(ValueError):
            S.prepare(raw, "nope")

    def test_splits(self):
        d = S.prepare(self.raw, "all")
        te = S.respondent_split(d, 0.25, seed=3)
        self.assertFalse(set(d.respondent[te]) & set(d.respondent[~te]))
        self.assertEqual(d.respondent[te].nunique(), round(0.25 * d.respondent.nunique()))
        np.testing.assert_array_equal(te, S.respondent_split(d, 0.25, seed=3))
        ts = S.situation_split(d, 2)
        self.assertTrue((d[ts].groupby("respondent").size() == 2).all())
        self.assertTrue((d[ts].situation >= 7).all())


class TestEstimators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = S.prepare(synthetic_raw(2500, seed=0), "all")

    def test_mnl_recovers_parameters(self):
        f = S.fit_mnl(self.d)
        self.assertTrue(f.converged)
        for k, v in TRUE.items():
            self.assertLess(abs(f.coef[k] - v), 4 * f.se[k], k)
        # correctly specified model: sandwich and classic standard errors agree
        np.testing.assert_allclose(f.robust_se, f.se, rtol=0.15)
        self.assertGreater(f.rho2, 0)

    def test_mnl_gradient_and_loglik(self):
        f = S.fit_mnl(self.d)
        p = f.predict_proba(self.d)
        np.testing.assert_allclose(p.sum(1), 1)
        av = S.availability(self.d)
        self.assertTrue((p[~av] == 0).all())
        y = self.d.y.to_numpy()
        self.assertAlmostEqual(np.log(p[np.arange(len(y)), y]).sum(), f.loglik, places=6)
        self.assertAlmostEqual(f.null_loglik, -np.log(av.sum(1)).sum(), places=6)
        # optimum: numerical gradient ~ 0
        X, _ = S.design(self.d, S.B01)
        ll = lambda b: S._log_probs(X, av, b, {}, 4)[np.arange(len(y)), y].sum()
        g = S._num_jac(lambda b: np.array([ll(b)]), f.coef.to_numpy())
        self.assertLess(np.abs(g).max(), 1e-2)

    def test_nested_with_mu_one_is_mnl(self):
        X, _ = S.design(self.d, S.B01)
        av = S.availability(self.d)
        b = np.array([TRUE[k] for k in ("asc_train", "asc_car", "b_time", "b_cost")])
        np.testing.assert_allclose(S._log_probs(X, av, np.r_[b, 1.0], S.NESTS_B09, 4),
                                   S._log_probs(X, av, b, {}, 4))
        lp = S._log_probs(X, av, np.r_[b, 2.5], S.NESTS_B09, 4)
        np.testing.assert_allclose(np.exp(lp).sum(1), 1)

    def test_nested_recovers_mu(self):
        d = S.prepare(synthetic_raw(2500, seed=4, nests=S.NESTS_B09, mu=2.0), "all")
        f = S.fit_nested(d)
        self.assertLess(abs(f.coef["mu_existing"] - 2.0), 4 * f.robust_se["mu_existing"])
        self.assertGreater(f.loglik, S.fit_mnl(d).loglik)
        with self.assertRaises(ValueError):
            S.fit_nested(d, nests={"a": ("train", "car")})

    def test_custom_terms(self):
        f = S.fit_mnl(self.d, S.RICH)
        self.assertEqual(list(f.coef.index[:3]), ["asc_train", "asc_car", "b_time_train"])
        self.assertGreaterEqual(f.loglik, S.fit_mnl(self.d).loglik - 1e-6)


class TestTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(cls.tmp.name, "swissmetro"))
        synthetic_raw(300, seed=2)[COLS].to_csv(os.path.join(cls.tmp.name, "swissmetro", "swissmetro.dat"),
                                                sep="\t", index=False)
        cls.task = S.SwissmetroModeChoice(sample="all")
        cls.data = cls.task.load(cls.tmp.name)
        cls.frame = cls.task.run(data=cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_protocol_and_registry(self):
        self.assertIsInstance(self.task, Task)
        self.assertIn("swissmetro", bench.list_tasks())
        self.assertIsInstance(bench.get("swissmetro"), S.SwissmetroModeChoice)
        with self.assertRaises(ValueError):
            S.SwissmetroModeChoice(bogus=1)
        with self.assertRaises(ValueError):
            S.SwissmetroModeChoice(split="x").load(self.tmp.name)

    def test_load_and_frame(self):
        d, fr = self.data, self.frame
        self.assertEqual(d.meta["n_test_respondents"], 60)
        self.assertFalse(set(d.train.respondent) & set(d.test.respondent))
        for m in ("uniform", "shares", "mnl", "mnl_rich", "nested", "gbdt"):
            P = np.vstack(fr[f"pred_{m}"].to_numpy())
            np.testing.assert_allclose(P.sum(1), 1, atol=1e-9)
            self.assertTrue((P[~S.availability(fr)] == 0).all(), m)
        np.testing.assert_array_equal(fr.truth, d.test.y)
        np.testing.assert_array_equal(fr.group, d.test.respondent)
        with self.assertRaises(ValueError):
            self.task.frame(d, {"bad": np.ones((len(d.test), 2))})
        with self.assertRaises(ValueError):
            self.task.frame(d, {"bad": -np.ones((len(d.test), 3))})

    def test_metrics_by_hand(self):
        fr = self.frame
        av = S.availability(fr)
        y = fr.truth.to_numpy()
        s = self.task.score(fr, "uniform", B=50)
        self.assertAlmostEqual(s["value"], float(-np.log(av.sum(1)).mean()), places=9)
        self.assertAlmostEqual(self.task.score(fr, "uniform", "rho2", B=50)["value"], 0.0, places=9)
        self.assertAlmostEqual(self.task.score(fr, "uniform", "accuracy", B=50)["value"],
                               float((1 / av.sum(1)).mean()), places=9)
        P = np.vstack(fr.pred_mnl.to_numpy())
        tvd = 0.5 * np.abs(P.mean(0) - np.bincount(y, minlength=3) / len(y)).sum()
        self.assertAlmostEqual(self.task.score(fr, "mnl", "share_tvd", B=50)["value"], tvd, places=9)
        r = self.task.score(fr, "mnl", B=200)
        self.assertLess(r["lo"], r["value"])
        self.assertLess(r["value"], r["hi"])
        self.assertEqual(r["n_respondents"], 60)
        # unavailable mass is dropped and rows renormalised; truth one-hot scores ~0
        f2 = self.task.frame(self.data, {"oracle": np.eye(3)[y] + 5 * (~av)})
        self.assertGreater(self.task.score(f2, "oracle", B=20)["value"], -1e-9)
        self.assertEqual(self.task.score(f2, "oracle", "accuracy", B=20)["value"], 1.0)

    def test_mnl_beats_uniform_on_logit_data(self):
        sc = self.task.score_all(self.frame, B=50)
        ll = sc[sc.metric == "loglik"].set_index("method").value
        self.assertGreater(ll["mnl"], ll["shares"])
        self.assertGreater(ll["shares"], ll["uniform"])

    def test_situation_split(self):
        t = S.SwissmetroModeChoice(sample="all", split="situation", lgbm=False, nested=False)
        d = t.load(self.tmp.name)
        self.assertEqual(set(d.test.respondent), set(d.train.respondent))
        preds = t.baselines(d)
        self.assertIn("own_shares", preds)
        P = preds["own_shares"]
        r0 = d.test.respondent.iloc[0]
        c = np.bincount(d.train.y[d.train.respondent == r0], minlength=3) + 1.0
        c = c * S.availability(d.test.iloc[[0]])[0]
        np.testing.assert_allclose(P[0], c / c.sum())


class TestText(unittest.TestCase):
    def setUp(self):
        self.d = S.prepare(synthetic_raw(20, seed=5), "all")

    def test_persona_and_situation(self):
        r = self.d.iloc[0].copy()
        r["GA"], r["av_car"], r["MALE"], r["AGE"], r["SURVEY"] = 1, 0, 1, 3, 0
        p, s = S.persona_text(r), S.situation_text(r)
        self.assertIn("man aged 40-54", p)
        self.assertIn("own a GA", p)
        self.assertIn("do not have a car", p)
        self.assertNotIn("Car:", s)
        self.assertIn("covered by your GA", s)
        self.assertIn(f"{r['SM_TT']:.0f} minutes", s)
        q = S.prompts(self.d.iloc[:3])
        self.assertEqual(len(q), 3)
        self.assertIn("Which would you choose?", q[0][1])

    def test_parse(self):
        np.testing.assert_allclose(S.parse_choice('Sure: {"train": 0.2, "sm": 0.5, "car": 0.3}'), [0.2, 0.5, 0.3])
        np.testing.assert_allclose(S.parse_choice('{"Swissmetro": 1}'), [0, 1, 0])
        np.testing.assert_allclose(S.parse_choice("I would take the car."), [0, 0, 1])
        self.assertIsNone(S.parse_choice("no idea"))

    def test_llm_method_with_fake(self):
        data = TaskData(train=self.d.iloc[:90], test=self.d.iloc[90:].reset_index(drop=True))
        replies = iter(['{"train": 1, "sm": 0, "car": 0}', "hmm"] * 100)
        llm = FakeLLM(lambda s, u: next(replies))
        m = S.llm_method(llm, smooth=0.1)
        P = m(data)
        av = S.availability(data.test)
        self.assertEqual(P.shape, (len(data.test), 3))
        np.testing.assert_allclose(P.sum(1), 1)
        u0 = av[0] / av[0].sum()
        np.testing.assert_allclose(P[0], 0.9 * np.array([1, 0, 0]) + 0.1 * u0)
        np.testing.assert_allclose(P[1], av[1] / av[1].sum())
        self.assertEqual(m.last_unparsed, len(data.test) // 2)
        self.assertEqual(len(llm.calls), len(data.test))
        self.assertTrue(all("survey" in s for s, _ in llm.calls))


@unittest.skipUnless(os.environ.get("SIMENGINE_DATA"), "needs $SIMENGINE_DATA with swissmetro/")
class TestRealData(unittest.TestCase):
    def test_biogeme_replication(self):
        root = bench.resolve_root(None, "swissmetro")
        d = S.prepare(S.read_raw(root))
        self.assertEqual(len(d), 6768)
        f = S.fit_mnl(d)
        # Biogeme b01a_logit (biogeme.epfl.ch examples): LL -5331.252, null -6964.663
        self.assertAlmostEqual(f.loglik, -5331.252, places=2)
        self.assertAlmostEqual(f.null_loglik, -6964.663, places=2)
        for k, v in {"asc_train": -0.701187, "asc_car": -0.154633, "b_time": -1.277859, "b_cost": -1.083790}.items():
            self.assertAlmostEqual(f.coef[k], v, places=4)
        self.assertAlmostEqual(f.robust_se["b_time"], 0.104254, places=4)
        g = S.fit_nested(d)
        # Biogeme b09_nested: LL -5236.9, nest parameter 2.053862
        self.assertAlmostEqual(g.loglik, -5236.9, places=1)
        self.assertAlmostEqual(g.coef["mu_existing"], 2.053862, delta=2e-3)


if __name__ == "__main__":
    unittest.main()
