"""lt.SequenceEncoder (E2c) on small synthetic purchase sequences, and the kos.py sq arms on the fake data folder."""
import contextlib
import io
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tests._latent_util import HAVE_TORCH, fake_root

CATS = ["A", "B", "C", "D"]
SMALL = dict(dim=8, emb=8, hidden=16, max_len=32, n_labels=6, epochs=3, patience=2, batch_size=16)


def events(n=60, seed=0, cut=date(2022, 1, 1)):
    """n persons; each person has 1-2 favourite categories bought every ~40 days since 2020, plus noise items."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        u = f"p{i:03d}"
        fav = rng.choice(CATS, 1 + i % 2, replace=False)
        d = date(2020, 1, 1) + timedelta(days=int(rng.integers(0, 30)))
        while d < cut:
            c = fav[rng.integers(len(fav))] if rng.random() < 0.8 else f"X{rng.integers(5)}"
            rows.append((u, cut, d, c, float(rng.uniform(2, 40)), 1.0))
            d += timedelta(days=int(rng.integers(10, 70)))
    return pd.DataFrame(rows, columns=["uid", "kesim", "tarih", "kategori", "fiyat", "adet"])


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestSequenceEncoder(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2)
        from simengine.persona import latent as lt
        self.lt = lt
        self.ev = events()
        uids = sorted(self.ev.uid.unique())
        self.fit_u, self.val_u, self.test_u = uids[:36], uids[36:48], uids[48:]
        self.guard = lt.TestGuard(self.test_u)

    def enc(self, seed=0):
        return self.lt.SequenceEncoder(seed=seed, targets=CATS, **SMALL).fit(self.ev, self.fit_u, self.val_u,
                                                                              guard=self.guard)

    def test_shape_and_unknown_person(self):
        e = self.enc()
        uids = self.fit_u + self.test_u + ["nobody"]
        U = e.transform(self.ev, uids)
        self.assertEqual(U.shape, (len(uids), 8))
        self.assertTrue(np.isfinite(U).all())
        self.assertEqual(e.labels_[:4], CATS)                         # targets come first among the labels
        self.assertGreater(e.info_["n_fit_windows"], 36)              # several windows per person
        self.assertEqual(e.info_["n_val_persons"], 12)

    def test_seed_determinism(self):
        uids = self.fit_u + self.test_u
        a, b = self.enc(0).transform(self.ev, uids), self.enc(0).transform(self.ev, uids)
        np.testing.assert_array_equal(a, b)
        c = self.enc(1).transform(self.ev, uids)
        self.assertGreater(float(np.abs(a - c).max()), 0)

    def test_guard_blocks_test_persons(self):
        with self.assertRaises(self.lt.LeakError):
            self.lt.SequenceEncoder(**SMALL).fit(self.ev, self.fit_u + self.test_u[:1], self.val_u, guard=self.guard)

    def test_fit_reads_only_fit_and_val_persons(self):
        """Changing (or dropping) a test person's items leaves the fitted encoder identical."""
        e0 = self.enc()
        ev2 = self.ev[~self.ev.uid.isin(self.test_u[:3])].copy()
        m = ev2.uid.isin(self.test_u)
        ev2.loc[m, "kategori"] = "Z"
        e1 = self.lt.SequenceEncoder(seed=0, targets=CATS, **SMALL).fit(ev2, self.fit_u, self.val_u, guard=self.guard)
        uids = self.fit_u[:5]
        np.testing.assert_array_equal(e0.transform(self.ev, uids), e1.transform(self.ev, uids))
        self.assertEqual(e0.vocab_, e1.vocab_)

    def test_windows_stay_before_the_cut(self):
        e = self.enc()
        tab = e._table(self.ev, self.fit_u[:1])
        rec = tab[self.fit_u[0]]
        ids, num = e._tokens(rec, rec[4] - 181)
        n_before = int((rec[0] < rec[4] - 181).sum())
        self.assertEqual(len(ids), 1 + min(n_before, SMALL["max_len"] - 1))
        self.assertEqual(ids[0], e.BOS)
        self.assertTrue((num[1:, 0] > 0).all())                       # every input item is before the anchor

    def test_one_cut_per_person(self):
        ev = self.ev.copy()
        ev.loc[ev.index[:3], "kesim"] = date(2022, 7, 1)
        with self.assertRaisesRegex(ValueError, "one kesim per uid"):
            self.lt.SequenceEncoder(**SMALL).fit(ev, self.fit_u, self.val_u)

    def test_vector_carries_the_favourites(self):
        """A logistic head on u predicts which category a test person buys next (synthetic signal)."""
        from sklearn.linear_model import LogisticRegression
        e = self.lt.SequenceEncoder(seed=0, targets=CATS, **dict(SMALL, epochs=8, patience=8)).fit(
            self.ev, self.fit_u, self.val_u, guard=self.guard)
        fav = self.ev[self.ev.kategori == "A"].groupby("uid").size()
        y = lambda us: np.array([fav.get(u, 0) > 3 for u in us], int)       # noqa: E731
        tr = self.fit_u + self.val_u
        clf = LogisticRegression(max_iter=2000).fit(e.transform(self.ev, tr), y(tr))
        acc = (clf.predict(e.transform(self.ev, self.test_u)) == y(self.test_u)).mean()
        self.assertGreaterEqual(acc, 0.8)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestKosSqArms(unittest.TestCase):
    def test_parse_and_run(self):
        from tests.test_runs_latent_kos import _kos
        k = _kos()
        self.assertEqual((k.parse_arm("enc-angcsq").u, k.parse_arm("stk-sq").kind), ("angcsq", "stk"))
        self.assertEqual(k.parse_arm("rsd-sq", "q38").name, "rsd-q38-sq")
        self.assertEqual(k.parse_arm("sft-q38-sq").task, "g1")
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(Path(tmp) / "veri")
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                rc = k.main(["--base", tmp, "--root", str(root), "--device", "cpu", "--kucuk", "12:4:4",
                             "--threads", "2", "--arm", "enc-sq", "enc-angcsq", "stk-sq"])
            res = [json.loads(l) for l in out.getvalue().splitlines() if l.startswith("{")]
            self.assertEqual(rc, 0, res)
            d = Path(tmp) / "kucuk" / "talp73" / "tahmin"
            for a in ("enc-sq", "enc-angcsq", "stk-sq"):
                self.assertEqual(len(pd.read_parquet(d / f"{a}_s0.parquet")), 4 * 4)
                meta = json.loads((d / f"{a}_s0.meta.json").read_text())
                self.assertEqual(meta["u_info"]["n_fit_persons"], 12)    # fit persons only, val apart
                self.assertEqual(meta["u_info"]["n_val_persons"], 4)
            self.assertEqual(json.loads((d / "enc-sq_s0.meta.json").read_text())["dim"], 64)


if __name__ == "__main__":
    unittest.main()
