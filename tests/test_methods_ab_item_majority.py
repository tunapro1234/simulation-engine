"""methods.ab.ItemMajority: the LLM-free "same item, same direction" rung for repeated A/B items (G3 #13)."""
import unittest

import numpy as np
import pandas as pd

from simengine.methods import ItemMajority
from simengine.methods.ab import ItemMajority as ItemMajorityAB


def repeated(n_items=30, per=6, noise=0.15, seed=0):
    """Each (question, amount) item has a fixed winner; it repeats across countries with label noise."""
    rs = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        win = rs.random() < 0.5
        for c in range(per):
            rows.append({"question": f"q{i % 10}", "amount": 10 * (i // 10), "country": c,
                         "y": int(win ^ (rs.random() < noise))})
    return pd.DataFrame(rows)


class TestItemMajority(unittest.TestCase):
    def test_exported(self):
        self.assertIs(ItemMajority, ItemMajorityAB)

    def test_direction_and_shrinkage(self):
        tr = pd.DataFrame({"item": ["a"] * 4 + ["b"] * 1 + ["c"] * 2, "y": [1, 1, 1, 0, 0, 1, 0]})
        m = ItemMajority(item="item", shrink=2.0).fit(tr)
        p = m.predict(pd.DataFrame({"item": ["a", "b", "c", "z"]}))
        np.testing.assert_allclose(p, [(3 + 1) / (4 + 2), (0 + 1) / (1 + 2), 0.5, 0.5])
        self.assertEqual(m.coverage_, 0.75)
        self.assertEqual(int(m.table_.loc["a", "n"]), 4)

    def test_unseen_global(self):
        tr = pd.DataFrame({"item": ["a", "a", "a", "b"], "y": [1, 1, 1, 1]})
        p = ItemMajority(item="item", unseen="global").fit(tr).predict(pd.DataFrame({"item": ["new"]}))
        self.assertEqual(p.tolist(), [1.0])
        with self.assertRaises(ValueError):
            ItemMajority(unseen="mean")

    def test_composite_item_beats_chance_on_held_out_country(self):
        df = repeated()
        tr, te = df[df.country < 5], df[df.country == 5]
        p = ItemMajority(item=["question", "amount"]).fit(tr).predict(te)
        acc = float(((p > 0.5) == (te.y == 1)).mean())
        self.assertGreater(acc, 0.75)
        self.assertEqual(ItemMajority(item=["question"]).fit(tr).table_.shape[0], 10)   # coarser key pools items

    def test_text_pair_item_is_oriented(self):
        tr = pd.DataFrame({"text_a": ["x", "y", "x"], "text_b": ["y", "x", "y"], "y": [1, 0, 1]})
        m = ItemMajority(shrink=0.0).fit(tr)                     # item=None: the unordered text pair
        p = m.predict(pd.DataFrame({"text_a": ["x", "y"], "text_b": ["y", "x"]}))
        np.testing.assert_allclose(p, [1.0, 0.0])                # B side wins when the pair is swapped
        self.assertEqual(len(m.table_), 1)

    def test_turkish_aliases_and_missing_labels(self):
        tr = pd.DataFrame({"metin_a": ["x", "x", "x"], "metin_b": ["y", "y", "y"], "gercek": [1, np.nan, 1]})
        m = ItemMajority(shrink=1.0).fit(tr)
        self.assertEqual(int(m.table_.n.iloc[0]), 2)
        self.assertAlmostEqual(float(m.predict(tr.drop(columns="gercek"))[0]), 2.5 / 3)

    def test_missing_item_column(self):
        with self.assertRaises(KeyError):
            ItemMajority(item="question").fit(pd.DataFrame({"y": [1]}))

    def test_string_or_out_of_range_labels_raise(self):
        # 'A'/'B' winners coerced to NaN gave an empty table and 0.5 everywhere, with no warning
        with self.assertRaises(ValueError):
            ItemMajority(item="q").fit(pd.DataFrame({"q": ["x", "x"], "y": ["A", "B"]}))
        with self.assertRaises(ValueError):
            ItemMajority(item="q").fit(pd.DataFrame({"q": ["x", "x"], "y": [2, 0]}))      # p would exceed 1
        m = ItemMajority(item="q", shrink=0.0).fit(pd.DataFrame({"q": ["x", "x"], "y": [True, None]}))
        np.testing.assert_allclose(m.predict(pd.DataFrame({"q": ["x"]})), [1.0])


if __name__ == "__main__":
    unittest.main()
