"""from_table / from_history id checks (d2 rehearsal gap #5): empty or duplicated ids, id_col=None."""
import unittest

import numpy as np
import pandas as pd

from simengine.persona import Mapping, from_history, from_row, from_table
from simengine.persona.builders import check_ids


def table(ids):
    return pd.DataFrame({"rid": ids, "age": [30 + i for i in range(len(ids))],
                         "city": [f"c{i % 3}" for i in range(len(ids))], "target": [i % 2 for i in range(len(ids))]})


class TestIds(unittest.TestCase):
    def test_empty_ids_raise_with_counts_and_hint(self):
        df = table(["a", None, np.nan, "d", ""])
        with self.assertRaises(ValueError) as cm:
            from_table(df, id_col="rid", forbidden=["target"])
        msg = str(cm.exception)
        self.assertIn("3 empty/NaN ids", msg)
        self.assertIn("5 rows", msg)
        self.assertIn("id_col=None", msg)

    def test_duplicate_ids_raise_with_counts(self):
        df = table(["a", "a", "b", "b", "b", "c"])
        with self.assertRaises(ValueError) as cm:
            from_table(df, id_col="rid", forbidden=["target"])
        msg = str(cm.exception)
        self.assertIn("5 rows share 2 duplicated ids", msg)
        self.assertNotIn("empty", msg)

    def test_both_problems_and_missing_column(self):
        with self.assertRaises(ValueError) as cm:
            check_ids(table(["a", "a", None]), "rid")
        self.assertIn("1 empty/NaN ids", str(cm.exception))
        self.assertIn("2 rows share 1 duplicated ids", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            from_table(table(["a", "b"]), id_col="nope")
        self.assertIn("not in the table", str(cm.exception))

    def test_explicit_mapping_is_checked_too(self):
        m = Mapping(id_col="rid", facets={"city": "city"})
        with self.assertRaises(ValueError):
            from_table(table(["a", np.nan]), m)

    def test_id_col_none_uses_row_numbers(self):
        df = table([None] * 6)
        ps = from_table(df, id_col=None, forbidden=["target", "rid"])
        self.assertEqual([p.pid for p in ps], [f"row{i}" for i in range(6)])
        self.assertNotIn("None", {p.pid for p in ps})
        self.assertEqual(ps[2].facets.get("city"), "c2")
        self.assertNotIn("target", ps[0].facets)

    def test_row_numbers_survive_sampling_and_custom_index(self):
        df = table(list("abcdefgh"))
        df.index = [100 + i for i in range(len(df))]
        ps = from_table(df, id_col=None, forbidden=["target"], n=3, seed=1)
        self.assertEqual(len(ps), 3)
        for p in ps:
            i = int(p.pid[3:])
            self.assertEqual(p.facets["rid"], df.iloc[i]["rid"])     # id = position in the full table

    def test_valid_ids_unchanged(self):
        ps = from_table(table(["u1", "u2", "u3"]), id_col="rid", forbidden=["target"])
        self.assertEqual([p.pid for p in ps], ["u1", "u2", "u3"])

    def test_from_row_never_makes_none_id(self):
        m = Mapping(id_col="rid")
        for bad in (None, np.nan, "  "):
            with self.assertRaises(ValueError):
                from_row({"rid": bad, "age": 3}, m)
        with self.assertRaises(ValueError):
            from_row({"age": 3}, Mapping(id_col=None))
        self.assertEqual(from_row({"age": 3}, Mapping(id_col=None), pid="row7").pid, "row7")

    def test_from_history_checks_ids(self):
        with self.assertRaises(ValueError) as cm:
            from_history(table(["a", "a"]), id_col="rid", level="L0")
        self.assertIn("duplicated", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
