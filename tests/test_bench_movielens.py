"""movielens bench task: synthetic ml-1m files (no real rows); the real-data check runs only when $SIMENGINE_DATA is set."""
import os
import re
import tempfile
import unittest

import numpy as np
import pandas as pd

from simengine.bench.base import DataNotFound
from simengine.bench.tasks import movielens as ml
from simengine.llm.fake import FakeLLM
from simengine.types import Task

N_USERS, N_MOVIES = 60, 80
GENRES = ["Drama", "Comedy", "Action", "Horror"]


def write_fixture(d: str, sub: str = "ml-1m", seed: int = 0) -> None:
    """Users with a taste vector, movies with a quality and a genre; ratings = quality + taste match + noise.
    Popular movies are rated more often; timestamps increase per user with repeated seconds."""
    rng = np.random.default_rng(seed)
    p = os.path.join(d, sub) if sub else d
    os.makedirs(p, exist_ok=True)
    q = rng.normal(0, .8, N_MOVIES)
    g = rng.integers(0, len(GENRES), N_MOVIES)
    taste = rng.normal(0, 1, (N_USERS, len(GENRES)))
    pop = np.exp(np.linspace(1.5, -1.5, N_MOVIES))
    pop /= pop.sum()
    rows = []
    for u in range(1, N_USERS + 1):
        n = int(rng.integers(20, 35))
        mv = rng.choice(N_MOVIES, n, replace=False, p=pop)
        ts = 970000000 + np.sort(rng.integers(0, 40, n)) * 60   # repeated seconds
        r = np.clip(np.round(3.4 + q[mv] + .6 * taste[u - 1, g[mv]] + rng.normal(0, .5, n)), 1, 5).astype(int)
        rows += [f"{u}::{m + 1}::{x}::{t}" for m, x, t in zip(mv, r, ts)]
    rng.shuffle(rows)
    with open(os.path.join(p, "ratings.dat"), "w", encoding="latin-1") as f:
        f.write("\n".join(rows) + "\n")
    ages = list(ml.AGE)
    with open(os.path.join(p, "users.dat"), "w", encoding="latin-1") as f:
        for u in range(1, N_USERS + 1):
            f.write(f"{u}::{'MF'[u % 2]}::{ages[u % len(ages)]}::{u % 21}::0{u:04d}\n")
    with open(os.path.join(p, "movies.dat"), "w", encoding="latin-1") as f:
        for m in range(1, N_MOVIES + 1):
            f.write(f"{m}::Synthetic Film {m:03d}: Part {m % 3} ({1950 + m})::{GENRES[g[m - 1]]}|Drama\n")


class _Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        write_fixture(os.path.join(cls.root, "movielens-1m"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestFiles(_Fixture):
    def test_read_and_layouts(self):
        r, u, m = ml.read_ml1m(os.path.join(self.root, "movielens-1m"))
        self.assertEqual(list(r.columns), ["user_id", "movie_id", "rating", "timestamp"])
        self.assertEqual(len(u), N_USERS)
        self.assertIn(":", m.loc[1, "title"])          # titles may contain ':'
        with tempfile.TemporaryDirectory() as d:     # .dat files directly in movielens-1m/
            write_fixture(os.path.join(d, "movielens-1m"), sub="")
            self.assertEqual(len(ml.MovieLens1M().load(d).meta["users"]), N_USERS)

    def test_missing(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(DataNotFound):
                ml.MovieLens1M().load(d)

    def test_options(self):
        with self.assertRaises(ValueError):
            ml.MovieLens1M(targt="rating")
        with self.assertRaises(ValueError):
            ml.MovieLens1M(target="ranking")
        with self.assertRaises(ValueError):
            ml.MovieLens1M(split="holdout")
        self.assertIsInstance(ml.MovieLens1M(), Task)
        self.assertEqual(ml.MovieLens1M(target="next_item").kind, "binary")
        self.assertEqual(ml.MovieLens1M().primary.name, "rmse")


class TestRating(_Fixture):
    def setUp(self):
        self.task = ml.MovieLens1M()
        self.data = self.task.load(self.root)

    def test_split_is_temporal_per_user(self):
        tr, te = self.data.train, self.data.test
        n = pd.concat([tr, te]).groupby("user_id").size()
        self.assertTrue((te.groupby("user_id").size() == np.ceil(0.2 * n)).all())
        last_tr = tr.groupby("user_id")["timestamp"].max()
        first_te = te.groupby("user_id")["timestamp"].min()
        self.assertTrue((first_te >= last_tr.reindex(first_te.index)).all())
        self.assertFalse(set(zip(tr.user_id, tr.movie_id)) & set(zip(te.user_id, te.movie_id)))

    def test_dev_split_never_sees_test(self):
        dv = ml.MovieLens1M(split="dev").load(self.root)
        test_pairs = set(zip(self.data.test.user_id, self.data.test.movie_id))
        dev_pairs = set(zip(dv.train.user_id, dv.train.movie_id)) | set(zip(dv.test.user_id, dv.test.movie_id))
        self.assertFalse(test_pairs & dev_pairs)
        self.assertEqual(len(dv.train) + len(dv.test), len(self.data.train))

    def test_baselines_and_metrics(self):
        fr = self.task.run(data=self.data)
        self.assertEqual(len(fr), len(self.data.test))
        for c in ("truth", "group", "item", "time", "segment", "split"):
            self.assertIn(c, fr.columns)
        sc = {m: self.task.score(fr, m) for m in ("global_mean", "user_mean", "item_mean", "bias", "item_knn")}
        y, p = fr["truth"].to_numpy(), fr["pred_global_mean"].to_numpy()
        self.assertAlmostEqual(sc["global_mean"]["value"], float(np.sqrt(np.mean((y - p) ** 2))), places=10)
        self.assertEqual(sc["bias"]["n"], N_USERS)
        self.assertLess(sc["bias"]["value"], sc["global_mean"]["value"])
        for r in sc.values():
            self.assertLessEqual(r["lo"], r["value"])
            self.assertGreaterEqual(r["hi"], r["value"])
        mae = self.task.score(fr, "bias", "mae")
        self.assertAlmostEqual(mae["value"], float(np.mean(np.abs(y - fr["pred_bias"]))), places=10)
        self.assertTrue(((fr["pred_item_knn"] >= 1) & (fr["pred_item_knn"] <= 5)).all())

    def test_fit_biases_solves_ridge(self):
        tr = self.data.train
        mu, bu, bi = ml.fit_biases(tr, lam=5.0, iters=200)
        e = tr["rating"] - mu - bu.reindex(tr.user_id).to_numpy() - bi.reindex(tr.movie_id).to_numpy()
        # stationarity: sum of residuals per user = lam * b_u
        g = e.groupby(tr["user_id"].to_numpy()).sum()
        np.testing.assert_allclose(g.to_numpy(), 5.0 * bu.reindex(g.index).to_numpy(), atol=1e-6)

    def test_llm_method_with_persona(self):
        tr = self.data.train
        def responder(system, user):
            m = re.search(r"mean (\d\.\d\d) out of 5", system)
            return m.group(1) if m else "3"
        llm = FakeLLM(responder)

        def llm_method(data):
            ps = ml.personas(data)
            prompts = [(ps[u], f"Rate: {ml.item_text(data, m)}") for u, m in zip(data.test.user_id, data.test.movie_id)]
            return [float(r.text) for r in llm.map(prompts)]
        fr = self.task.run(methods={"llm": llm_method}, data=self.data, baselines=["user_mean"])
        np.testing.assert_allclose(fr["pred_llm"], fr["pred_user_mean"], atol=0.006)
        self.assertEqual(len(llm.calls), len(self.data.test))
        self.assertTrue(np.isfinite(self.task.score(fr, "llm")["value"]))
        self.assertEqual(sorted(ml.personas(self.data)), sorted(tr.user_id.unique()))


class TestPersona(_Fixture):
    def test_train_only_and_content(self):
        data = ml.MovieLens1M().load(self.root)
        uid = 3
        txt = ml.persona_text(data, uid, n_liked=50, n_disliked=50)
        self.assertIn("Age: " + ml.AGE[list(ml.AGE)[uid % len(ml.AGE)]], txt)
        self.assertIn("Occupation: " + ml.OCCUPATION[uid % 21], txt)
        self.assertNotIn("0003", txt)                      # zip code never shown
        test_titles = data.meta["movies"].loc[data.test[data.test.user_id == uid].movie_id, "title"]
        for t in test_titles:
            self.assertNotIn(t, txt)
        self.assertIn(f"Has rated {int((data.train.user_id == uid).sum())} movies", txt)
        self.assertNotIn("Gender", ml.persona_text(data, uid, demographics=False))
        self.assertTrue(ml.item_text(data, 1).endswith(", Drama"))


class TestNextItem(_Fixture):
    def setUp(self):
        self.task = ml.MovieLens1M(target="next_item", n_candidates=20, next_k=0)
        self.data = self.task.load(self.root)

    def test_candidates(self):
        te, tr = self.data.test, self.data.train
        r, _, _ = ml.read_ml1m(os.path.join(self.root, "movielens-1m"))
        rated = r.groupby("user_id")["movie_id"].agg(set)
        self.assertTrue((te.groupby("user_id")["y"].sum() == 1).all())
        self.assertTrue((te.groupby("user_id").size() == 20).all())
        last = ml.order_by_user_time(r).groupby("user_id").tail(1).set_index("user_id")["movie_id"]
        pos = te[te.y == 1].set_index("user_id")["movie_id"]
        self.assertTrue((pos == last.reindex(pos.index)).all())
        neg = te[te.y == 0]
        self.assertFalse(any(m in rated[u] for u, m in zip(neg.user_id, neg.movie_id)))
        self.assertTrue(set(neg.movie_id) <= set(tr.movie_id))
        self.assertFalse((te.groupby("user_id")["y"].first() == 1).all())   # target position is shuffled
        again = ml.MovieLens1M(target="next_item", n_candidates=20).load(self.root).test
        pd.testing.assert_frame_equal(te, again)

    def test_metrics_oracle_constant_and_ties(self):
        fr = self.task.frame(self.data, {"oracle": self.data.test["y"].to_numpy(float),
                                         "const": np.zeros(len(self.data.test)),
                                         "anti": 1 - self.data.test["y"].to_numpy(float)})
        self.assertEqual(self.task.score(fr, "oracle")["value"], 1.0)
        self.assertAlmostEqual(self.task.score(fr, "oracle", "ndcg10")["value"], 1.0)
        self.assertAlmostEqual(self.task.score(fr, "const")["value"], 10 / 20)
        self.assertEqual(self.task.score(fr, "anti")["value"], 0.0)
        hit, ndcg = ml.expected_topk(np.array([0, 9, 10, 8]), np.array([0, 0, 0, 3]))
        np.testing.assert_allclose(hit, [1, 1, 0, 0.5])
        self.assertAlmostEqual(ndcg[1], 1 / np.log2(11))
        self.assertAlmostEqual(ndcg[3], (1 / np.log2(10) + 1 / np.log2(11)) / 4)

    def test_baselines(self):
        fr = self.task.run(data=self.data)
        s = {m: self.task.score(fr, m)["value"] for m in ("random", "popularity", "item_knn")}
        self.assertGreater(s["popularity"], s["random"])
        self.assertGreater(s["item_knn"], 0.3)
        self.assertEqual(self.task.score(fr, "popularity")["n"], N_USERS)

    def test_full_ranking(self):
        d = self.data
        tgt = d.test[d.test.y == 1].set_index("user_id")["movie_id"]

        def oracle(users, movies):
            return (np.asarray(movies)[None, :] == tgt.reindex(users).to_numpy()[:, None]).astype(float)
        r = ml.full_ranking(d, oracle, B=50)
        cold = ~tgt.isin(d.train.movie_id)
        self.assertAlmostEqual(r["hit10"]["value"], 1 - cold.mean())
        pop = ml.full_ranking(d, self.task.scorer(d, "popularity"), B=50)
        self.assertLessEqual(pop["hit10"]["value"], 1.0)
        self.assertEqual(pop["n_items"], d.train.movie_id.nunique())
        # popularity against the sampled pool scores at least as well as against all unrated movies
        fr = self.task.run(data=d, baselines=["popularity"])
        self.assertGreaterEqual(self.task.score(fr, "popularity")["value"] + 1e-12, pop["hit10"]["value"])
        with self.assertRaises(KeyError):
            self.task.scorer(d, "mf")


@unittest.skipUnless(os.environ.get("SIMENGINE_DATA"), "$SIMENGINE_DATA not set")
class TestReal(unittest.TestCase):
    def test_counts_and_ladder(self):
        try:
            task = ml.MovieLens1M()
            data = task.load(None)
        except DataNotFound:
            self.skipTest("MovieLens-1M not under $SIMENGINE_DATA")
        self.assertEqual(len(data.train) + len(data.test), 1000209)
        self.assertEqual(data.meta["n_users"], 6040)
        fr = task.run(data=data, baselines=["global_mean", "bias"])
        self.assertLess(task.score(fr, "bias", B=50)["value"], task.score(fr, "global_mean", B=50)["value"])


if __name__ == "__main__":
    unittest.main()
