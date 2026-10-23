"""ConditionalMode (multiclass person x slot x day-type mode with smoothing / backoff / decay) and the MAE-ladder
rungs MedianConstant / CellMedian, on synthetic long tables."""
import unittest

import numpy as np
import pandas as pd

from simengine.methods.baselines import CellMedian, ConditionalMode, MajorityBaseline, MedianConstant, stat_for_metric


def visits(n_person=30, n_day=14, seed=0):
    """person x day x slot (48 half hours) -> place label; home at night, work on weekday daytime for most people,
    a personal "third place" in the evening, 15 % noise."""
    rs = np.random.default_rng(seed)
    rows = []
    for p in range(n_person):
        third = ["gym", "cafe", "park"][p % 3]
        for d in range(n_day):
            dt = "we" if d % 7 in (5, 6) else "wd"
            for t in range(0, 48, 2):
                if t < 14 or t >= 44:
                    lab = "home"
                elif dt == "wd" and 18 <= t < 36:
                    lab = "work" if p % 5 else "school"
                elif 36 <= t < 44:
                    lab = third
                else:
                    lab = "home" if dt == "we" else "shop"
                if rs.random() < 0.15:
                    lab = str(rs.choice(["home", "work", "shop", "gym", "cafe", "park", "school"]))
                rows.append((f"p{p}", d, t, dt, lab))
    return pd.DataFrame(rows, columns=["person", "day", "slot", "daytype", "place"])


class TestConditionalMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = visits()
        cls.tr, cls.te = df[df.day < 10], df[df.day >= 10]

    def test_exact_mode_matches_groupby(self):
        keys = ["person", "slot", "daytype"]
        m = ConditionalMode(keys, target="place").fit(self.tr)
        ref = self.tr.groupby(keys)["place"].agg(lambda s: s.value_counts().index[0])
        q = ref.index.to_frame(index=False)
        got, lvl = m.predict(q, return_level=True)
        self.assertTrue((lvl == 0).all())
        # ties may differ from value_counts order; compare counts of the chosen label instead
        cnt = self.tr.groupby(keys + ["place"]).size()
        best = cnt.groupby(level=[0, 1, 2]).max()
        chosen = cnt.reindex(pd.MultiIndex.from_frame(q.assign(place=got))).to_numpy()
        np.testing.assert_array_equal(chosen, best.reindex(pd.MultiIndex.from_frame(q)).to_numpy())

    def test_beats_majority_on_holdout(self):
        m = ConditionalMode(["person", "slot", "daytype"], target="place",
                            backoff=[["slot", "daytype"], ["person"]]).fit(self.tr)
        acc = float((m.predict(self.te) == self.te.place.to_numpy()).mean())
        maj = float((MajorityBaseline().fit(self.tr.place).predict(self.te) == self.te.place.to_numpy()).mean())
        self.assertGreater(acc, 0.75)
        self.assertGreater(acc, maj + 0.3)

    def test_backoff_chain_and_levels(self):
        m = ConditionalMode(["person", "slot", "daytype"], target="place",
                            backoff=[["slot", "daytype"], ["person"]]).fit(self.tr)
        q = pd.DataFrame({"person": ["p1", "nobody", "p1", "nobody"], "slot": [20, 20, 21, 21],
                          "daytype": ["wd", "wd", "wd", "xx"]})
        lab, lvl = m.predict(q, return_level=True)
        self.assertEqual(lvl.tolist(), [0, 1, 2, 3])          # full, slot x daytype, person, global
        self.assertEqual(lab[0], "work")
        self.assertEqual(lab[1], "work")
        self.assertEqual(lab[3], self.tr.place.value_counts().index[0])
        self.assertEqual(m.levels_, [("person", "slot", "daytype"), ("slot", "daytype"), ("person",), ()])

    def test_neighbour_smoothing(self):
        tr = pd.DataFrame({"person": "a", "slot": [10] + [30] * 5, "place": ["x"] + ["y"] * 5})
        q = pd.DataFrame({"person": ["a"], "slot": [11]})
        plain = ConditionalMode(["person", "slot"], target="place", backoff=[["person"]]).fit(tr)
        self.assertEqual(plain.predict(q).tolist(), ["y"])     # slot 11 unseen -> person mode
        sm = ConditionalMode(["person", "slot"], target="place", backoff=[["person"]],
                             smooth={"slot": [(-1, 0.5), (1, 0.5)]}).fit(tr)
        lab, lvl = sm.predict(q, return_level=True)
        self.assertEqual((lab.tolist(), lvl.tolist()), (["x"], [0]))
        # the centre outweighs neighbours: slot 30 itself stays "y" even with a heavy "x" next door
        tr2 = pd.concat([tr, pd.DataFrame({"person": "a", "slot": [31] * 4, "place": ["x"] * 4})])
        self.assertEqual(sm.fit(tr2).predict(pd.DataFrame({"person": ["a"], "slot": [30]})).tolist(), ["y"])

    def test_wrap(self):
        tr = pd.DataFrame({"slot": [47, 47, 5], "place": ["night", "night", "day"]})
        q = pd.DataFrame({"slot": [0]})
        m = ConditionalMode(["slot"], target="place", smooth={"slot": [(-1, 1.0), (1, 1.0)]}).fit(tr)
        self.assertEqual(m.predict(q, return_level=True)[1].tolist(), [1])   # no wrap: slot 0 unseen
        m = ConditionalMode(["slot"], target="place", smooth={"slot": [(-1, 1.0), (1, 1.0)]},
                            wrap={"slot": 48}).fit(tr)
        lab, lvl = m.predict(q, return_level=True)
        self.assertEqual((lab.tolist(), lvl.tolist()), (["night"], [0]))

    def test_decay(self):
        tr = pd.DataFrame({"person": "a", "t": [0, 0, 0, 10, 10], "place": ["old"] * 3 + ["new"] * 2})
        q = pd.DataFrame({"person": ["a"]})
        self.assertEqual(ConditionalMode(["person"], target="place").fit(tr).predict(q).tolist(), ["old"])
        m = ConditionalMode(["person"], target="place", decay=0.5, time_col="t").fit(tr)
        self.assertEqual(m.predict(q).tolist(), ["new"])
        dt = tr.assign(t=pd.to_datetime("2026-01-01") + pd.to_timedelta(tr.t, unit="D"))
        self.assertEqual(ConditionalMode(["person"], target="place", decay=0.5, time_col="t").fit(dt)
                         .predict(q).tolist(), ["new"])
        with self.assertRaises(ValueError):
            ConditionalMode(["person"], target="place", decay=0.5).fit(tr)    # decay needs time_col

    def test_top_k_padded_from_backoff(self):
        tr = pd.DataFrame({"person": ["a"] * 5 + ["b"] * 6,
                           "place": ["x", "x", "x", "y", "y", "z", "z", "z", "w", "w", "x"]})
        m = ConditionalMode(["person"], target="place").fit(tr)
        top = m.predict(pd.DataFrame({"person": ["a", "c"]}), k=4)
        self.assertEqual(top.shape, (2, 4))
        self.assertEqual(top[0].tolist(), ["x", "y", "z", "w"])      # own 2, then global order without repeats
        self.assertEqual(top[1].tolist(), ["x", "z", "w", "y"])      # global: x 4, z 3, then w/y tie 2-2
        self.assertEqual(m.predict(pd.DataFrame({"person": ["a"]}), k=6)[0].tolist()[-2:], [None, None])

    def test_ties_go_to_globally_frequent_label(self):
        tr = pd.DataFrame({"k": ["a", "a", "b", "b", "b"], "place": ["u", "v", "v", "v", "v"]})
        m = ConditionalMode(["k"], target="place").fit(tr)
        self.assertEqual(m.predict(pd.DataFrame({"k": ["a"]})).tolist(), ["v"])

    def test_weights_and_min_weight(self):
        tr = pd.DataFrame({"k": ["a", "a", "a", "b"], "w": [1, 1, 5, 1], "place": ["u", "u", "v", "u"]})
        self.assertEqual(ConditionalMode(["k"], target="place", weight_col="w").fit(tr)
                         .predict(pd.DataFrame({"k": ["a"]})).tolist(), ["v"])
        m = ConditionalMode(["k"], target="place", min_weight=1).fit(tr)
        lab, lvl = m.predict(pd.DataFrame({"k": ["a", "b"]}), return_level=True)
        self.assertEqual(lvl.tolist(), [0, 1])                        # "b" has weight 1, not > 1: global

    def test_numeric_labels_keep_dtype(self):
        tr = pd.DataFrame({"k": [0, 0, 1], "cell": [7, 7, 3]})
        out = ConditionalMode(["k"], target="cell").fit(tr).predict(pd.DataFrame({"k": [0, 1, 2]}))
        self.assertTrue(np.issubdtype(out.dtype, np.integer))
        self.assertEqual(out.tolist(), [7, 3, 7])

    def test_missing_column_raises(self):
        with self.assertRaises(KeyError):
            ConditionalMode(["person", "slot"], target="place").fit(pd.DataFrame({"person": [1], "place": [2]}))


class TestMedianRungs(unittest.TestCase):
    def test_stat_for_metric(self):
        self.assertEqual(stat_for_metric("MAE"), "median")
        self.assertEqual(stat_for_metric("zmae"), "median")
        self.assertEqual(stat_for_metric("rmse"), "mean")
        with self.assertRaises(ValueError):
            stat_for_metric("auc")

    def test_median_constant_beats_mean_on_mae(self):
        rs = np.random.default_rng(0)
        y = rs.lognormal(0, 1, 2000)
        yt = rs.lognormal(0, 1, 2000)
        med = MedianConstant().fit(y)
        mean = MedianConstant(metric="mse").fit(y)
        self.assertAlmostEqual(med.value_, float(np.median(y)))
        self.assertAlmostEqual(mean.value_, float(np.mean(y)))
        self.assertLess(np.abs(yt - med.predict(len(yt))).mean(), np.abs(yt - mean.predict(yt)).mean())
        self.assertAlmostEqual(MedianConstant(stat=0.9).fit(y).value_, float(np.quantile(y, 0.9)))

    def test_weighted_median(self):
        m = MedianConstant().fit([1.0, 2.0, 10.0], weight=[1, 1, 5])
        self.assertEqual(m.value_, 10.0)
        self.assertEqual(MedianConstant().fit([1.0, 2.0, 10.0], weight=[1, 1, 1]).value_, 2.0)
        self.assertAlmostEqual(MedianConstant(stat="mean").fit([1.0, 3.0], weight=[3, 1]).value_, 1.5)

    def test_cell_median_shrink(self):
        df = pd.DataFrame({"g": ["a"] * 3 + ["b"] * 5, "h": [0, 0, 1, 0, 0, 0, 1, 1],
                           "y": [1.0, 2.0, 30.0, 5.0, 6.0, 7.0, 8.0, 100.0]})
        g = float(np.median(df.y))
        cm = CellMedian(by=["g"], y_col="y").fit(df)
        np.testing.assert_allclose(cm.predict(pd.DataFrame({"g": ["a", "b", "zz"]})), [2.0, 7.0, g])
        cs = CellMedian(by=["g"], y_col="y", shrink=3).fit(df)
        np.testing.assert_allclose(cs.predict(pd.DataFrame({"g": ["a", "b"]})),
                                   [(3 * 2.0 + 3 * g) / 6, (5 * 7.0 + 3 * g) / 8])
        big = CellMedian(by=["g"], y_col="y", shrink=1e9).fit(df)
        np.testing.assert_allclose(big.predict(pd.DataFrame({"g": ["a", "b"]})), [g, g], rtol=1e-6)
        two = CellMedian(by=["g", "h"], y_col="y").fit(df)
        np.testing.assert_allclose(two.predict(pd.DataFrame({"g": ["a", "b", "a"], "h": [0, 1, 9]})),
                                   [1.5, 54.0, g])
        mean = CellMedian(by="g", y_col="y", metric="mse").fit(df)
        np.testing.assert_allclose(mean.predict(pd.DataFrame({"g": ["a"]})), [11.0])

    def test_cell_median_weighted(self):
        df = pd.DataFrame({"g": ["a"] * 3, "y": [1.0, 2.0, 10.0], "w": [1.0, 1.0, 5.0]})
        cm = CellMedian(by=["g"], y_col="y", weight="w").fit(df)
        self.assertEqual(cm.predict(pd.DataFrame({"g": ["a"]})).tolist(), [10.0])

    def test_cell_median_picks_shrink_on_valid(self):
        rs = np.random.default_rng(1)
        n_cell = 200

        def draw(n_per):
            cell = np.repeat(np.arange(n_cell), n_per)
            return pd.DataFrame({"c": cell, "y": rs.normal(0, 3, len(cell))})   # no real cell effect
        tr, va = draw(3), draw(3)
        cm = CellMedian(by=["c"], y_col="y", shrink=None).fit(tr, valid=va)
        self.assertGreaterEqual(cm.shrink_, 20)                # pure noise cells -> heavy shrinkage
        self.assertEqual(set(cm.scores_), set(CellMedian.GRID))
        with self.assertRaises(ValueError):
            CellMedian(by=["c"], y_col="y", shrink=None).fit(tr)   # shrink=None needs valid


if __name__ == "__main__":
    unittest.main()
