"""Uplift tasks: metric definitions against a brute-force reference, learners on synthetic heterogeneous effects,
and the three loaders on tiny synthetic files in the documented layout. Real-data checks skip without $SIMENGINE_DATA."""
import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine import bench
from simengine.bench.tasks import uplift as U
from simengine.methods import uplift as MU
from simengine.types import Task


def _parquet_engine() -> bool:
    for m in ("pyarrow", "fastparquet"):
        try:
            __import__(m)
            return True
        except ImportError:
            pass
    return False


PARQUET = _parquet_engine()


def _naive_curve(kind, y, t, s):
    """Literal definition: for every distinct score threshold, the rows with score >= threshold."""
    xs, vs = [0.0], [0.0]
    for thr in np.unique(s)[::-1]:
        m = s >= thr
        nt, nc = (m & (t == 1)).sum(), (m & (t == 0)).sum()
        yt, yc = y[m & (t == 1)].sum(), y[m & (t == 0)].sum()
        if kind == "qini":
            v = yt - (yc * nt / nc if nc else 0.0)
        else:
            v = ((yt / nt if nt else 0.0) - (yc / nc if nc else 0.0)) * m.sum()
        xs.append(float(m.sum()))
        vs.append(float(v))
    return np.array(xs), np.array(vs)


def _synthetic(n=6000, share=0.5, seed=0):
    """x1 drives the effect (+ for x1 > 0, - for x1 < 0); x2 only the base rate."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    t = (rng.random(n) < share).astype(int)
    p = 1 / (1 + np.exp(-(-1 + 0.8 * x2 + t * 1.2 * x1)))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": np.where(x2 > 0, "a", "b"), "t": t, "y": y})


class TestMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.n = 400
        self.y = (rng.random(self.n) < 0.3).astype(float)
        self.t = (rng.random(self.n) < 0.6).astype(float)
        self.s = np.round(rng.normal(size=self.n), 1)  # ties on purpose

    def test_curves_match_brute_force(self):
        for kind, fn in [("qini", U.qini_curve), ("uplift", U.uplift_curve)]:
            x, v = fn(self.y, self.t, self.s)
            xn, vn = _naive_curve(kind, self.y, self.t, self.s)
            np.testing.assert_allclose(x, xn)
            np.testing.assert_allclose(v, vn, atol=1e-9)

    def test_perfect_one_random_near_zero(self):
        y, t = self.y, self.t
        self.assertAlmostEqual(U.qini_coefficient(y, t, y * t - y * (1 - t)), 1.0)
        py, pt = U._perfect_uplift_scores(y, t)
        cr, tn = np.sum(y * (1 - t)), np.sum((1 - y) * t)
        self.assertAlmostEqual(U.auuc(y, t, py if cr > tn else pt), 1.0)
        r = [U.qini_coefficient(y, t, np.random.default_rng(i).random(self.n)) for i in range(200)]
        self.assertLess(abs(np.mean(r)), 0.05)
        self.assertLess(U.qini_coefficient(y, t, -(y * t - y * (1 - t))), 0)

    def test_constant_score_is_zero(self):
        self.assertAlmostEqual(U.qini_coefficient(self.y, self.t, np.zeros(self.n)), 0.0)
        self.assertAlmostEqual(U.auuc(self.y, self.t, np.zeros(self.n)), 0.0)

    def test_weights_equal_row_resampling(self):
        idx = np.random.default_rng(9).integers(0, self.n, self.n)
        w = np.bincount(idx, minlength=self.n).astype(float)
        y, t, s = self.y, self.t, self.s
        self.assertAlmostEqual(U.qini_coefficient(y, t, s, weights=w), U.qini_coefficient(y[idx], t[idx], s[idx]))
        self.assertAlmostEqual(U.auuc(y, t, s, weights=w), U.auuc(y[idx], t[idx], s[idx]))

    def test_uplift_at_k(self):
        s = np.arange(self.n, dtype=float)
        top = s >= self.n - int(0.3 * self.n)
        want = self.y[top & (self.t == 1)].mean() - self.y[top & (self.t == 0)].mean()
        self.assertAlmostEqual(U.uplift_at_k(self.y, self.t, s, 0.3), want)
        # k n not an integer: the first int(k n) rows (scikit-uplift `uplift_at_k`, strategy "overall")
        o = np.argsort(s, kind="mergesort")[::-1][: int(self.n * 0.333)]
        yo, to = self.y[o], self.t[o]
        self.assertAlmostEqual(U.uplift_at_k(self.y, self.t, s, 0.333), yo[to == 1].mean() - yo[to == 0].mean())

    def test_metrics_moved_to_methods_numbers_unchanged(self):
        """The curves, metrics and learners now live in simengine.methods.uplift; the bench imports the same objects,
        and the values pinned before the move (same inputs) still come out."""
        for name in ("qini_coefficient", "auuc", "uplift_at_k", "qini_curve", "uplift_curve", "two_model",
                     "s_learner", "class_transformation", "_norm_area", "_uak", "_thresholds"):
            self.assertIs(getattr(U, name), getattr(MU, name), name)
        y, t, s = self.y, self.t, self.s
        self.assertAlmostEqual(U.qini_coefficient(y, t, s), -0.006825000267624077, places=12)
        self.assertAlmostEqual(U.auuc(y, t, s), -0.0031079388160304214, places=12)
        self.assertAlmostEqual(U.uplift_at_k(y, t, s, 0.3), -0.003396109910466183, places=12)

    def test_task_score_has_ci(self):
        fr = pd.DataFrame({"truth": self.y.astype(int), "treatment": self.t.astype(int), "pred_m": self.s})
        task = U.HillstromUplift()
        for m in ("qini", "auuc", "uplift@30"):
            r = task.score(fr, "m", m, B=200)
            self.assertEqual(r["n"], self.n)
            self.assertLessEqual(r["lo"], r["value"])
            self.assertLessEqual(r["value"], r["hi"])
            self.assertTrue(r["higher_is_better"])


class TestLearners(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self.kw = dict(treatment="t", outcome="y", categorical=["cat"], numeric=["x1", "x2"])

    def _q(self, d, te, s):
        return U.qini_coefficient(te.y.to_numpy(), te.t.to_numpy(), s)

    def test_learners_beat_random(self):
        d = _synthetic()
        tr, te = d.iloc[:4000], d.iloc[4000:]
        for name, s in {"t_log": U.two_model(tr, te, model="logistic", **self.kw),
                        "t_gbdt": U.two_model(tr, te, model="gbdt", **self.kw),
                        "ct": U.class_transformation(tr, te, **self.kw),
                        "s_gbdt": U.s_learner(tr, te, **self.kw)}.items():
            self.assertEqual(len(s), len(te), name)
            # the true-uplift oracle scores ~0.20 here (normalised against a perfect that sees outcomes)
            self.assertGreater(self._q(d, te, s), 0.12, name)
            self.assertGreater(np.corrcoef(s, te.x1)[0, 1], 0.5, name)

    def test_class_transformation_reweights_unbalanced(self):
        d = _synthetic(n=20000, share=0.85, seed=1)
        tr, te = d.iloc[:15000], d.iloc[15000:]
        s = U.class_transformation(tr, te, **self.kw)
        # without reweighting 2P(z=1)-1 is dominated by the 85% treated rows (mostly z = y); with it the score tracks x1
        self.assertGreater(np.corrcoef(s, te.x1)[0, 1], 0.8)
        self.assertLess(abs(np.corrcoef(s, te.x2)[0, 1]), 0.3)

    def test_stratified_split(self):
        d = _synthetic(n=1000)
        m = U.stratified_split(d, ["t", "y"], 0.2, 0)
        for _, g in d.assign(m=m).groupby(["t", "y"]):
            self.assertAlmostEqual(g.m.mean(), 0.2, delta=0.01)
        np.testing.assert_array_equal(m, U.stratified_split(d, ["t", "y"], 0.2, 0))


def _hillstrom(root, n=3000):
    d = os.path.join(root, "hillstrom", "ham")
    os.makedirs(d)
    rng = np.random.default_rng(0)
    seg = rng.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], n)
    womens = rng.integers(0, 2, n)
    p = 0.1 + 0.15 * ((seg == "Womens E-Mail") & (womens == 1)) + 0.05 * (seg == "Mens E-Mail")
    pd.DataFrame({"recency": rng.integers(1, 13, n), "history_segment": rng.choice(["1) $0 - $100", "2) $100 - $200"], n),
                  "history": rng.gamma(2, 100, n).round(2), "mens": rng.integers(0, 2, n), "womens": womens,
                  "zip_code": rng.choice(["Urban", "Rural"], n), "newbie": rng.integers(0, 2, n),
                  "channel": rng.choice(["Web", "Phone"], n), "segment": seg,
                  "visit": (rng.random(n) < p).astype(int), "conversion": (rng.random(n) < p / 5).astype(int),
                  "spend": 0.0}).to_csv(os.path.join(d, U.HillstromUplift.FILE), index=False)


def _criteo(root, n=4000):
    d = os.path.join(root, "criteo-uplift", "islenmis")
    os.makedirs(d)
    rng = np.random.default_rng(1)
    f = rng.normal(size=(n, 12)).astype("float32")
    t = (rng.random(n) < 0.85).astype("int8")
    p = 1 / (1 + np.exp(-(-2 + t * f[:, 0])))
    v = (rng.random(n) < p).astype("int8")
    df = pd.DataFrame(f, columns=[f"f{i}" for i in range(12)])
    df["treatment"], df["visit"] = t, v
    df["conversion"] = (v * (rng.random(n) < 0.3)).astype("int8")
    df["exposure"] = (t * (rng.random(n) < 0.5)).astype("int8")
    if PARQUET:
        df.to_parquet(os.path.join(d, "criteo.parquet"))
    else:  # no parquet engine: the loader's csv.gz path
        os.makedirs(os.path.join(root, "criteo-uplift", "ham"))
        df.to_csv(os.path.join(root, "criteo-uplift", "ham", "criteo-research-uplift-v2.1.csv.gz"), index=False)


def _x5(root, n=3000):
    d = os.path.join(root, "x5-uplift", "ham")
    os.makedirs(d)
    rng = np.random.default_rng(2)
    ids = [f"c{i:06d}" for i in range(n)]
    age = rng.integers(18, 80, n).astype(float)
    age[:5] = [-5, 1500, 30, 40, 50]
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.5 + 0.2 * t * (age < 40)).astype(int)
    issue = pd.Timestamp("2017-03-01") + pd.to_timedelta(rng.integers(0, 600, n), unit="D")
    redeem = (issue + pd.to_timedelta(rng.integers(0, 90, n), unit="D")).astype(str).to_numpy(object)
    redeem[rng.random(n) < 0.1] = None
    redeem[5:15] = "2019-04-02 10:00:00"  # after the SMS: post-treatment, must be masked
    pd.DataFrame({"client_id": ids, "treatment_flg": t, "target": y}).to_csv(
        os.path.join(d, "uplift_train.csv.gz"), index=False)
    pd.DataFrame({"client_id": ids[::-1], "first_issue_date": issue.astype(str)[::-1], "first_redeem_date": redeem[::-1],
                  "age": age[::-1], "gender": rng.choice(["U", "F", "M"], n)}).to_csv(
        os.path.join(d, "clients.csv.gz"), index=False)


class TestTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        _hillstrom(cls.root)
        _criteo(cls.root)
        _x5(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_registered_and_protocol(self):
        for name in ("hillstrom_uplift", "criteo_uplift", "x5_uplift"):
            self.assertIn(name, bench.list_tasks())
            task = bench.get(name)
            self.assertIsInstance(task, Task)
            self.assertEqual(task.primary.name, "qini")
            self.assertIn(task.data_subdir, task.layout)

    def test_unknown_option_and_values(self):
        with self.assertRaises(ValueError):
            U.HillstromUplift(spilt=0.2)
        with self.assertRaises(ValueError):
            U.HillstromUplift(arm="kids").load(self.root)
        with self.assertRaises(ValueError):
            U.X5Uplift(baselines="random+magic").run(data_root=self.root)

    def test_all_tasks_run_and_score(self):
        for task in (U.HillstromUplift(), U.CriteoUplift(outcome="conversion"), U.X5Uplift()):
            data = task.load(self.root)
            n = len(data.train) + len(data.test)
            self.assertAlmostEqual(len(data.test) / n, 0.2, delta=0.01)
            self.assertTrue({"t", "y"} <= set(data.test.columns))
            fr = task.run(methods={"mine": lambda d: d.test["y"].to_numpy() * 0.0}, data=data)
            for b in U.UpliftTask.BASELINES:
                self.assertIn(f"pred_{b}", fr.columns, (task.name, b))
            self.assertEqual(list(fr.columns[:4]), ["unit", "truth", "treatment", "segment"])
            tab = task.score_all(fr, B=50)
            self.assertEqual(len(tab), 6 * 3)
            self.assertTrue(np.isfinite(tab.value).all(), task.name)

    def test_hillstrom_arm_and_signal(self):
        task = U.HillstromUplift(arm="womens", baselines="random+t_logistic")
        data = task.load(self.root)
        self.assertNotIn("Mens E-Mail", set(data.train.segment) | set(data.test.segment))
        fr = task.run(data=data)
        self.assertEqual(sorted(c for c in fr.columns if c.startswith("pred_")), ["pred_random", "pred_t_logistic"])
        # the fixture's effect lives in womens == 1 only: the T-learner should rank it first
        self.assertGreater(np.corrcoef(fr.pred_t_logistic, data.test.womens)[0, 1], 0.5)

    def test_criteo_subsample_and_x5_cleaning(self):
        d = U.CriteoUplift(n_rows=1000).load(self.root)
        self.assertEqual(len(d.train) + len(d.test), 1000)
        x = U.X5Uplift(test_frac=0.5, seed=3).load(self.root)
        allx = pd.concat([x.train, x.test])
        self.assertEqual(len(allx), 3000)
        self.assertEqual(int(allx.age.isna().sum()), 2)        # -5 and 1500 masked
        self.assertTrue((allx.redeem_delay.dropna() >= 0).all())
        self.assertGreater(allx.redeem_day.isna().mean(), 0.05)
        cut = (U.X5Uplift.CUTOFF - pd.Timestamp("2017-01-01")).days
        self.assertTrue((allx.redeem_day.dropna() < cut).all())
        self.assertEqual(int(allx.redeem_day.isna().sum()) - int((allx.redeem_delay.isna()).sum()), 0)


DATA = os.environ.get("SIMENGINE_DATA")


@unittest.skipUnless(DATA and os.path.isdir(os.path.join(DATA, "hillstrom")), "$SIMENGINE_DATA / hillstrom not set")
class TestHillstromReal(unittest.TestCase):
    def test_counts_and_t_logistic(self):
        warnings.simplefilter("ignore")
        task = U.HillstromUplift(baselines="random+t_logistic")
        data = task.load(DATA)
        self.assertEqual(len(data.train) + len(data.test), 64000)
        fr = task.run(data=data)
        self.assertGreater(task.score(fr, "t_logistic", B=100)["value"], task.score(fr, "random", B=100)["value"])


if __name__ == "__main__":
    unittest.main()
