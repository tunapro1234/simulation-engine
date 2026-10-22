"""Single-text binary baseline (TfidfBinary) and smoothed OOF target encoding, on synthetic query -> outcome rows."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from simengine.methods import TfidfBinary, oof_predict, target_encode

GOOD = ["fast delivery", "great price today", "love this one", "works really well"]
BAD = ["broken item", "refund please", "far too slow", "bad fit size"]


def synth(n=900, seed=0, signal=0.75):
    """y ~ Bernoulli(0.4); with probability ``signal`` the text comes from the y-matching phrase list, else from
    either list. ``k`` is a numeric column carrying extra signal, ``shop`` a categorical one with none."""
    rs = np.random.default_rng(seed)
    y = (rs.random(n) < 0.4).astype(int)
    txt = [str(rs.choice(GOOD if yi else BAD)) if rs.random() < signal else str(rs.choice(GOOD + BAD)) for yi in y]
    k = y * 1.0 + rs.normal(0, 1, n)
    shop = rs.choice(["a", "b", "c"], n)
    return pd.DataFrame({"q": txt, "k": k, "shop": shop, "y": y}, index=[f"r{i}" for i in range(n)])


class TargetEncodeTest(unittest.TestCase):
    def setUp(self):
        self.tr = pd.DataFrame({"c": ["a", "a", "a", "b", "b", "z"], "y": [1, 1, 0, 0, 0, 1]},
                               index=list("pqrstu"))
        self.te = pd.DataFrame({"c": ["a", "b", "new"]}, index=[10, 11, 12])

    def test_in_sample_and_test_values(self):
        tr, te = target_encode(self.tr, self.te, "c", "y", m=2, oof=False)
        prior = 3 / 6
        self.assertAlmostEqual(tr["p"], (2 + 2 * prior) / (3 + 2))
        self.assertAlmostEqual(tr["s"], (0 + 2 * prior) / (2 + 2))
        self.assertAlmostEqual(te[10], (2 + 2 * prior) / 5)
        self.assertAlmostEqual(te[12], prior)                      # unseen category -> prior
        self.assertEqual(list(te.index), [10, 11, 12])
        self.assertEqual(tr.name, "te_c")

    def test_oof_row_never_sees_own_label(self):
        # leave-one-out folds: the singleton category 'z' gets the prior of the other five rows
        ids = np.arange(6)
        tr, te = target_encode(self.tr, self.te, "c", "y", m=2, folds=ids)
        self.assertAlmostEqual(tr["u"], 2 / 5)
        # row p (a, y=1): others in a are 1, 0; prior of the other rows = 2/5
        self.assertAlmostEqual(tr["p"], (1 + 2 * 0.4) / (2 + 2))
        full, _ = target_encode(self.tr, None, "c", "y", m=2, oof=False)
        self.assertNotAlmostEqual(tr["u"], full["u"])
        # test rows always use all training rows
        _, te_full = target_encode(self.tr, self.te, "c", "y", m=2, oof=False)
        pd.testing.assert_series_equal(te, te_full)

    def test_oof_label_shuffle_has_no_signal(self):
        # high-cardinality noise key: in-sample encoding 'predicts' y, OOF does not
        rs = np.random.default_rng(1)
        d = pd.DataFrame({"c": rs.integers(0, 300, 3000).astype(str), "y": rs.integers(0, 2, 3000)})
        ins, _ = target_encode(d, None, "c", "y", m=1, oof=False)
        oof, _ = target_encode(d, None, "c", "y", m=1, folds=5, seed=0)
        self.assertGreater(roc_auc_score(d.y, ins), 0.65)
        self.assertLess(abs(roc_auc_score(d.y, oof) - 0.5), 0.05)

    def test_tuple_key_fold_column_and_missing_y(self):
        d = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": [1, 2, 1, 1], "y": [1.0, np.nan, 0.0, 1.0],
                          "f": [0, 1, 0, 1]})
        tr, _ = target_encode(d, None, ("a", "b"), "y", m=0, oof=False)
        self.assertEqual(tr.name, "te_a_b")
        self.assertAlmostEqual(tr.iloc[0], 1.0)
        self.assertAlmostEqual(tr.iloc[1], 2 / 3)                   # (x, 2) only has a missing y -> prior
        self.assertAlmostEqual(tr.iloc[2], 0.5)
        oof, _ = target_encode(d, None, "a", "y", m=0, folds="f")
        self.assertAlmostEqual(oof.iloc[2], 1.0)                    # fold 0 row (y, 0) sees only fold 1's y=1
        self.assertTrue(np.isfinite(oof).all())

    def test_errors(self):
        with self.assertRaises(KeyError):
            target_encode(self.tr, None, "nope", "y")
        with self.assertRaises(KeyError):
            target_encode(self.tr, None, "c", "nope")
        with self.assertRaises(ValueError):
            target_encode(self.tr, None, "c", "y", m=-1)
        with self.assertRaises(ValueError):
            target_encode(self.tr, None, "c", "y", folds=1)
        with self.assertRaises(ValueError):
            target_encode(self.tr, None, "c", "y", folds=np.arange(3))


class TfidfBinaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synth()
        cls.tr, cls.te = cls.df.iloc[:600], cls.df.iloc[600:]

    def test_learns_text_signal_and_picks_c(self):
        tb = TfidfBinary("q").fit(self.tr, self.tr.y)
        p = tb.predict(self.te)
        self.assertEqual(p.shape, (len(self.te),))
        self.assertTrue(((p > 0) & (p < 1)).all())
        self.assertGreater(roc_auc_score(self.te.y, p), 0.75)
        self.assertIn(tb.C_, tb.C_grid)
        self.assertEqual(set(tb.cv_scores_), set(tb.C_grid))
        self.assertEqual(tb.C_, max(tb.cv_scores_, key=lambda c: (tb.cv_scores_[c], -c)))
        pp = tb.predict_proba(self.te)
        np.testing.assert_allclose(pp[:, 1], p)
        np.testing.assert_allclose(pp.sum(1), 1.0)

    def test_fixed_c_skips_search_and_accepts_texts(self):
        tb = TfidfBinary("q", C=0.5).fit(list(self.tr.q), self.tr.y.to_numpy())
        self.assertEqual(tb.C_, 0.5)
        self.assertEqual(tb.cv_scores_, {})
        np.testing.assert_allclose(tb.predict(list(self.te.q)), tb.predict(self.te[["q"]]))

    def test_extra_and_encoded_columns(self):
        base = TfidfBinary("q", C=1.0).fit(self.tr, self.tr.y)
        ext = TfidfBinary("q", C=1.0, extra_cols=("k", "shop"), encode_cols=("q",)).fit(self.tr, self.tr.y)
        self.assertEqual(ext.num_cols_, ["k", "__te0__"])
        self.assertEqual(ext.cat_cols_, ["shop"])
        a0 = roc_auc_score(self.te.y, base.predict(self.te))
        a1 = roc_auc_score(self.te.y, ext.predict(self.te))
        self.assertGreater(a1, a0)                                  # k carries extra signal
        unseen = self.te.assign(shop="never", q="unseen words")
        self.assertTrue(np.isfinite(ext.predict(unseen)).all())
        with self.assertRaises(ValueError):
            TfidfBinary("q", extra_cols=("k",)).fit(list(self.tr.q), self.tr.y)
        with self.assertRaises(KeyError):
            TfidfBinary("q", extra_cols=("nope",)).fit(self.tr, self.tr.y)

    def test_word_or_char_only(self):
        for kw in ({"word_ngrams": None}, {"char_ngrams": None}):
            tb = TfidfBinary("q", C=1.0, **kw).fit(self.tr, self.tr.y)
            self.assertGreater(roc_auc_score(self.te.y, tb.predict(self.te)), 0.7)
        with self.assertRaises(ValueError):
            TfidfBinary("q", C=1.0, word_ngrams=None, char_ngrams=None).fit(self.tr, self.tr.y)

    def test_oof_predict_compatible(self):
        m = TfidfBinary("q", C=1.0, extra_cols=("k",), encode_cols=("q",))
        for method in ("predict", "predict_proba"):
            o = oof_predict(m, self.df, features=["q", "k"], target="y", method=method)
            self.assertEqual(list(o.index), list(self.df.index))
            self.assertTrue(o.notna().all())
            self.assertGreater(roc_auc_score(self.df.y, o), 0.75)
        self.assertFalse(hasattr(m, "model_"))                      # the template is cloned, never fitted

    def test_one_class_and_bad_target(self):
        tb = TfidfBinary("q").fit(self.tr, np.zeros(len(self.tr)))
        self.assertIsNone(tb.C_)
        np.testing.assert_allclose(tb.predict(self.te), 0.0)
        with self.assertRaises(ValueError):
            TfidfBinary("q").fit(self.tr, np.full(len(self.tr), 2))
        with self.assertRaises(ValueError):
            TfidfBinary("q").fit(self.tr, np.r_[np.nan, self.tr.y.to_numpy()[1:]])


if __name__ == "__main__":
    unittest.main()
