"""RCT condition effects (within-block difference vs control, averaged over blocks) and the cross-block transfer
baseline, on synthetic many-labs data with known effects."""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.methods import rct_effects, rct_transfer

EFFECTS = {"A": 0.6, "B": -0.3, "C": 0.15, "D": 0.0}


def synth(n_blocks=12, n_per=120, seed=0, seg_shift=0.0, imbalance=False):
    """Block level offsets are large (sd 2); conditions add EFFECTS (+ seg_shift for segment 'hi').
    ``imbalance``: high-offset blocks put more people in the treatment arms, so the naive pooled difference is
    biased and only the within-block estimator recovers EFFECTS."""
    rs = np.random.default_rng(seed)
    rows = []
    offs = np.sort(rs.normal(0, 2, n_blocks))
    for b, off in enumerate(offs):
        for c in ["Control"] + list(EFFECTS):
            n = n_per
            if imbalance:
                n = n_per // 4 if (c == "Control") == (b >= n_blocks // 2) else n_per
            seg = rs.choice(["lo", "hi"], n)
            y = off + EFFECTS.get(c, 0.0) + (seg == "hi") * (c != "Control") * seg_shift + rs.normal(0, 1, n)
            rows += [(f"b{b:02d}", c, s, v) for s, v in zip(seg, y)]
    return pd.DataFrame(rows, columns=["country", "condition", "seg", "y"])


class RctEffectsTest(unittest.TestCase):
    def test_single_block_is_welch_difference(self):
        d = pd.DataFrame({"cond": ["c", "c", "c", "t", "t", "t", "t", "u"],
                          "y": [1.0, 2.0, 3.0, 2.0, 4.0, 4.0, 6.0, np.nan]})
        e = rct_effects(d, "cond", "c", "y")
        self.assertEqual(list(e["cond"]), ["t"])                    # u has only a missing y -> dropped
        r = e.iloc[0]
        self.assertAlmostEqual(r.effect, 4.0 - 2.0)
        self.assertAlmostEqual(r.se, np.sqrt(np.var([2, 4, 4, 6], ddof=1) / 4 + 1.0 / 3))
        self.assertAlmostEqual(r.ci_high - r.ci_low, 2 * 1.959963984540054 * r.se)
        self.assertEqual((r.n_blocks, r.n_cond, r.n_control), (1, 4, 3))
        self.assertEqual(e.attrs["blocks"], [])

    def test_within_block_removes_imbalance_bias(self):
        d = synth(imbalance=True)
        within = rct_effects(d, "condition", "Control", "y", block="country").set_index("condition")
        naive = rct_effects(d, "condition", "Control", "y").set_index("condition")
        for c, true in EFFECTS.items():
            self.assertLess(abs(within.effect[c] - true), 4 * within.se[c])
            self.assertGreater(abs(naive.effect[c] - true), 0.5)    # pooled difference is confounded by block
        self.assertEqual(int(within.n_blocks.iloc[0]), 12)
        self.assertEqual(len(within.attrs["blocks"]), 12)

    def test_block_weights(self):
        d = pd.DataFrame({"b": ["x"] * 4 + ["y"] * 6,
                          "cond": ["c", "c", "t", "t", "c", "c", "c", "t", "t", "t"],
                          "y": [0, 2, 3, 5, 0, 1, 2, 1, 1, 4.0]})
        dx, dy = 4.0 - 1.0, 2.0 - 1.0
        size = rct_effects(d, "cond", "c", "y", block="b").iloc[0]
        equal = rct_effects(d, "cond", "c", "y", block="b", weights="equal").iloc[0]
        custom = rct_effects(d, "cond", "c", "y", block="b", weights={"x": 3.0, "y": 1.0}).iloc[0]
        self.assertAlmostEqual(size.effect, (4 * dx + 6 * dy) / 10)
        self.assertAlmostEqual(equal.effect, (dx + dy) / 2)
        self.assertAlmostEqual(custom.effect, (3 * dx + dy) / 4)
        vx, vy = 2 / 2 + 2 / 2, 3 / 3 + 1 / 3
        self.assertAlmostEqual(equal.se, np.sqrt(vx + vy) / 2)
        self.assertAlmostEqual(equal.mean_control, (1.0 + 1.0) / 2)
        with self.assertRaises(KeyError):
            rct_effects(d, "cond", "c", "y", block="b", weights={"x": 1.0})
        with self.assertRaises(ValueError):
            rct_effects(d, "cond", "c", "y", block="b", weights="inverse")

    def test_segments_and_blocks_without_control(self):
        d = synth(seg_shift=0.5)
        d = d[~((d.country == "b00") & (d.condition == "Control"))]     # b00 cannot contribute
        e = rct_effects(d, "condition", "Control", "y", block="country", by="seg")
        self.assertEqual(len(e), 2 * len(EFFECTS))
        self.assertTrue((e.n_blocks == 11).all())
        self.assertNotIn("b00", e.attrs["blocks"])
        e = e.set_index(["seg", "condition"])
        for c, true in EFFECTS.items():
            self.assertLess(abs(e.effect["lo", c] - true), 4 * e.se["lo", c])
            self.assertLess(abs(e.effect["hi", c] - true - 0.5), 4 * e.se["hi", c])

    def test_singleton_cell_borrows_pooled_variance(self):
        d = pd.DataFrame({"b": ["x", "x", "x", "y", "y", "y"], "cond": ["c", "c", "t", "c", "c", "t"],
                          "y": [0.0, 1.0, 2.0, 0.0, 2.0, 3.0]})
        e = rct_effects(d, "cond", "c", "y", block="b")
        self.assertTrue(np.isnan(e.se.iloc[0]))                     # no arm-t cell with two people anywhere
        d2 = pd.concat([d, pd.DataFrame({"b": ["y"], "cond": ["t"], "y": [5.0]})], ignore_index=True)
        self.assertTrue(np.isfinite(rct_effects(d2, "cond", "c", "y", block="b").se.iloc[0]))

    def test_errors(self):
        d = synth(n_blocks=2, n_per=5)
        with self.assertRaises(ValueError):
            rct_effects(d, "condition", "Kontrol", "y")
        with self.assertRaises(KeyError):
            rct_effects(d, "condition", "Control", "nope")
        with self.assertRaises(ValueError):
            rct_effects(d, "condition", "Control", "y", block="country", by="country")


class RctTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = synth(n_blocks=12, n_per=150)
        cls.test = [f"b{b:02d}" for b in range(0, 12, 3)]

    def test_pred_is_train_effect_truth_is_test_effect(self):
        t = rct_transfer(self.d, "condition", "Control", "y", block="country", test_blocks=self.test)
        train = sorted(set(self.d.country) - set(self.test))
        self.assertEqual(sorted(t.attrs["train_blocks"]), train)
        tr = rct_effects(self.d[self.d.country.isin(train)], "condition", "Control", "y", block="country")
        te = rct_effects(self.d[self.d.country.isin(self.test)], "condition", "Control", "y", block="country")
        t = t.set_index("condition")
        np.testing.assert_allclose(t.pred[tr.condition], tr.effect)
        np.testing.assert_allclose(t.truth[te.condition], te.effect)
        self.assertTrue((t.n_blocks_train == 8).all() and (t.n_blocks_test == 4).all())
        self.assertGreater(t.attrs["spearman"], 0.7)                # effects are well separated (4 conditions)
        self.assertAlmostEqual(t.attrs["mae"], float((t.pred - t.truth).abs().mean()))

    def test_explicit_train_blocks_and_overlap(self):
        t = rct_transfer(self.d, "condition", "Control", "y", block="country", test_blocks=self.test,
                         train_blocks=["b01", "b02"])
        self.assertTrue((t.n_blocks_train == 2).all())
        with self.assertRaises(ValueError):
            rct_transfer(self.d, "condition", "Control", "y", block="country", test_blocks=["b01"],
                         train_blocks=["b01", "b02"])
        with self.assertRaises(ValueError):
            rct_transfer(self.d, "condition", "Control", "y", block="country", test_blocks=[])
        with self.assertRaisesRegex(KeyError, "label type"):          # a typo / str vs int label, not "no control"
            rct_transfer(self.d, "condition", "Control", "y", block="country", test_blocks=["b01", "zz"])
        with self.assertRaisesRegex(KeyError, "train_blocks"):
            rct_transfer(self.d, "condition", "Control", "y", block="country", test_blocks=["b01"],
                         train_blocks=[2])

    def test_categorical_columns(self):
        d = self.d.astype({"condition": "category", "seg": "category", "country": "category"})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            e = rct_effects(d, "condition", "Control", "y", block="country", by="seg")
            t = rct_transfer(d, "condition", "Control", "y", block="country", test_blocks=self.test)
        ref = rct_effects(self.d, "condition", "Control", "y", block="country", by="seg")
        np.testing.assert_allclose(e.effect.to_numpy(), ref.effect.to_numpy())
        np.testing.assert_allclose(e.se.to_numpy(), ref.se.to_numpy())
        self.assertTrue(np.isfinite(t.attrs["spearman"]))

    def test_per_block_and_segments(self):
        t = rct_transfer(self.d, "condition", "Control", "y", block="country", test_blocks=self.test,
                         per_block=True)
        self.assertEqual(len(t), len(self.test) * len(EFFECTS))
        self.assertEqual(list(t.columns[:2]), ["country", "condition"])
        self.assertEqual(set(t.attrs["spearman_by_block"]), set(self.test))
        self.assertAlmostEqual(t.attrs["spearman"], float(np.mean(list(t.attrs["spearman_by_block"].values()))))
        self.assertEqual(t.groupby("condition").pred.nunique().max(), 1)   # same training effect for every block
        repr(t)                                                     # attrs must not break pandas printing
        s = rct_transfer(self.d, "condition", "Control", "y", block="country", test_blocks=self.test, by="seg")
        self.assertEqual(len(s), 2 * len(EFFECTS))
        self.assertEqual(list(s.columns[:2]), ["seg", "condition"])

    def test_condition_unseen_in_training_gets_nan(self):
        d = self.d[~((self.d.condition == "D") & ~self.d.country.isin(self.test))]
        t = rct_transfer(d, "condition", "Control", "y", block="country", test_blocks=self.test).set_index(
            "condition")
        self.assertTrue(np.isnan(t.pred["D"]))
        self.assertEqual(t.n_blocks_train["D"], 0)
        self.assertTrue(np.isfinite(t.truth["D"]))


if __name__ == "__main__":
    unittest.main()
