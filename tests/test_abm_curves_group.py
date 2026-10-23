"""BassAnalogy(group_col=): the (log p, log q) prior comes only from history series of the target's own group."""
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine.abm.curves as C


def two_groups(n=12, seed=0):
    """Group 'fast': p 0.03, q 0.6. Group 'slow': p 0.005, q 0.15. Ceiling 80, noise 0.3 pp."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        fast = i % 2 == 0
        p, q = (0.03, 0.6) if fast else (0.005, 0.15)
        t = np.arange(1990, 2021, dtype=float)
        y = C.bass_curve(t, 80.0, p, q, 1990 + i % 3) + rng.normal(0, 0.3, len(t))
        rows += [dict(series=f"g{i}", kind="fast" if fast else "slow", t=tt, y=float(np.clip(yy, 0, 100)))
                 for tt, yy in zip(t, y)]
    return pd.DataFrame(rows)


class TestBassAnalogyGroup(unittest.TestCase):
    def setUp(self):
        self.df = two_groups()
        self.c = 2008
        self.hist = self.df[self.df.t <= self.c]

    def test_groups_get_their_own_prior(self):
        m = C.BassAnalogy(group_col="kind")
        fast = m.fit(self.hist, "g0", self.c)["prior"]
        slow = m.fit(self.hist, "g1", self.c)["prior"]
        self.assertIsNotNone(fast)
        self.assertIsNotNone(slow)
        self.assertGreater(fast[1] - slow[1], 0.5)              # log q: fast group imitates much faster
        np.testing.assert_allclose(np.exp(fast[1]), 0.6, rtol=0.25)
        np.testing.assert_allclose(np.exp(slow[1]), 0.15, rtol=0.25)
        pooled = C.BassAnalogy().fit(self.hist, "g0", self.c)["prior"]
        self.assertFalse(np.allclose(pooled, fast))

    def test_none_is_the_old_pooled_prior(self):
        a, b = C.BassAnalogy(), C.BassAnalogy(group_col=None)
        pa, pb = a.fit(self.hist, "g0", self.c)["prior"], b.fit(self.hist, "g0", self.c)["prior"]
        others = []
        for s, g in self.hist.groupby("series"):
            if s == "g0":
                continue
            t = g["t"].to_numpy(float)
            _, p = C.bass(t, g["y"].to_numpy(float), t[-1:], return_params=True)
            others.append((np.log(p["p"]), np.log(p["q"])))
        np.testing.assert_array_equal(pa, pb)
        np.testing.assert_allclose(pa, np.median(np.array(others), axis=0))
        pd.testing.assert_frame_equal(a.predict(a.fit(self.hist, "g0", self.c), [1, 4]),
                                      b.predict(b.fit(self.hist, "g0", self.c), [1, 4]))

    def test_small_group_falls_back_with_warning(self):
        hist = self.hist.copy()
        hist.loc[hist.series.isin(["g1", "g3"]), "kind"] = "rare"   # target g1 has one same-kind donor
        m = C.BassAnalogy(group_col="kind")
        with self.assertWarns(UserWarning):
            mod = m.fit(hist, "g1", self.c)
        self.assertIsNone(mod["prior"])
        plain = C.CurveBaseline("bass").predict(C.CurveBaseline("bass").fit(hist, "g1", self.c), [1, 4])
        np.testing.assert_allclose(m.predict(mod, [1, 4])["y_pred"], plain["y_pred"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            C.BassAnalogy().fit(hist, "g1", self.c)            # default: no group, no warning

    def test_reused_instance_does_not_reuse_another_frames_prior(self):
        """Same cutoff, same row count, other data: the cached prior of the first frame must not leak in."""
        other = self.hist.copy()
        for _, g in other.groupby("series"):                      # every series slow now
            other.loc[g.index, "y"] = C.bass_curve(g["t"].to_numpy(float), 80.0, 0.005, 0.15, 1990)
        m = C.BassAnalogy()
        m.fit(self.hist, "g0", self.c)
        np.testing.assert_allclose(m.fit(other, "g0", self.c)["prior"],
                                   C.BassAnalogy().fit(other, "g0", self.c)["prior"])

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            C.BassAnalogy(group_col="nope").fit(self.hist, "g0", self.c)


if __name__ == "__main__":
    unittest.main()
