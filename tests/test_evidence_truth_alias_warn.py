"""evaluate with no truth column but a column that looks like the outcome ('y', 'label', ...): the card is
'awaiting truth' as before, now with a warning naming the column (D13-W8 #1: cutoff_windows wrote 'y' and the card
came out empty without a word)."""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine import evidence as ev


def _frame(n=200, seed=0):
    rs = np.random.default_rng(seed)
    y = (rs.random(n) < 0.3).astype(int)
    return pd.DataFrame({"unit": [f"u{i}" for i in range(n)], "y": y,
                         "pred_ours": np.clip(0.3 + 0.2 * (y - 0.3) + rs.normal(0, 0.1, n), 0, 1)})


class TestTruthAliasWarning(unittest.TestCase):
    def test_y_column_warns_and_stays_awaiting(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = ev.evaluate(_frame(), "t", ours="ours", task="binary")
        self.assertEqual(rep["kanit_kart"]["rozet"], "bekliyor")
        msg = [str(x.message) for x in w if "awaiting truth" in str(x.message) and "'y'" in str(x.message)]
        self.assertTrue(msg, [str(x.message) for x in w])

    def test_no_warning_without_a_candidate(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ev.evaluate(_frame().rename(columns={"y": "clicks"}), "t", ours="ours", task="binary")
        self.assertFalse([x for x in w if "rename it to 'truth'" in str(x.message)])

    def test_renamed_truth_scores(self):
        rep = ev.evaluate(_frame().rename(columns={"y": "truth"}), "t", ours="ours", task="binary", B=50)
        self.assertNotEqual(rep["kanit_kart"]["rozet"], "bekliyor")


if __name__ == "__main__":
    unittest.main()
