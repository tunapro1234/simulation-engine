"""Session table and funnel helpers of methods.next_action on small synthetic step tables."""
import unittest

import numpy as np
import pandas as pd

from simengine.methods import next_action as na


def toy():
    # s1: converts at position 3; s2: abandons after 2 steps; s3: converts at 0; s4: 5 steps, no purchase.
    # Rows are shuffled so the helpers must sort by step themselves.
    rows = [
        ("s1", 0, "search", "red shoes"), ("s1", 1, "view", None), ("s1", 2, "cart", ""), ("s1", 3, "purchase", None),
        ("s1", 4, "view", "after"),
        ("s2", 0, "search", "lamp"), ("s2", 1, "search", "desk lamp"),
        ("s3", 0, "purchase", None),
        ("s4", 0, "view", None), ("s4", 1, "search", "mug"), ("s4", 2, "search", "mug"), ("s4", 3, "view", None),
        ("s4", 4, "back", None),
    ]
    df = pd.DataFrame(rows, columns=["session", "step", "action", "query"])
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)


class TestSessions(unittest.TestCase):
    def test_table(self):
        t = na.sessions(toy(), outcome_key="purchase", prefix_k=3, text_col="query").set_index("session")
        self.assertEqual(t.loc["s1", "prefix"], ["search", "view", "cart"])
        self.assertEqual(t.loc["s1", "prefix_text"], "search > view > cart")
        self.assertEqual(t.loc["s1", "outcome"], "purchase")
        self.assertEqual(t.loc["s1", "converted"], 1)
        self.assertEqual(t.loc["s1", "outcome_step"], 3)
        self.assertEqual(t.loc["s1", "n_steps"], 5)
        self.assertTrue(t.loc["s1", "reached_k"])
        self.assertEqual(t.loc["s1", "text"], "red shoes")          # "after" is past the prefix
        self.assertEqual(t.loc["s2", "outcome"], "abandon")
        self.assertTrue(pd.isna(t.loc["s2", "outcome_step"]))
        self.assertEqual(t.loc["s2", "prefix_len"], 2)
        self.assertFalse(t.loc["s2", "reached_k"])
        self.assertEqual(t.loc["s2", "text"], "lamp | desk lamp")
        self.assertEqual(t.loc["s3", "prefix"], [])                 # outcome at step 0: empty prefix
        self.assertEqual(t.loc["s3", "text"], "")
        self.assertEqual(t.loc["s4", "text"], "mug")                # repeated query kept once

    def test_prefix_never_contains_outcome(self):
        for k in (1, 2, 3, 4, 6):
            t = na.sessions(toy(), outcome_key="purchase", prefix_k=k).set_index("session")
            for sid, r in t.iterrows():
                self.assertNotIn("purchase", r["prefix"], (k, sid))
                self.assertLessEqual(r["prefix_len"], k)
            self.assertEqual(t.loc["s1", "prefix_len"], min(k, 3))

    def test_flag_column_and_aliases(self):
        df = toy().rename(columns={"session": "session_id", "step": "adim", "action": "hk"})
        df["bought"] = (df["hk"] == "purchase").astype(int)
        t = na.sessions(df, outcome_key="bought", prefix_k=2)          # Turkish aliases accepted
        self.assertIn("session", t.columns)
        self.assertEqual(int(t["converted"].sum()), 2)

    def test_custom_columns(self):
        df = toy().rename(columns={"session": "sid", "step": "t", "action": "ev"})
        t = na.sessions(df, outcome_key="purchase", session_col="sid", order_col="t", action_col="ev",
                        labels=("buy", "leave"))
        self.assertEqual(sorted(t["outcome"].unique()), ["buy", "leave"])
        self.assertIn("sid", t.columns)

    def test_callable_outcome(self):
        def lab(g):
            acts = list(g["action"])
            if "cart" in acts:
                return "carted", acts.index("cart")
            return "browse"
        t = na.sessions(toy(), outcome_key=lab, prefix_k=5, labels=("carted", "browse")).set_index("session")
        self.assertEqual(t.loc["s1", "prefix"], ["search", "view"])
        self.assertEqual(t.loc["s1", "converted"], 1)
        self.assertEqual(t.loc["s3", "outcome"], "browse")
        self.assertEqual(t.loc["s3", "prefix"], ["purchase"])

    def test_callable_without_position_cuts_last_step(self):
        t = na.sessions(toy(), outcome_key=lambda g: "purchase" if len(g) == 2 else "abandon",
                        prefix_k=5).set_index("session")
        self.assertEqual(t.loc["s2", "prefix"], ["search"])
        self.assertEqual(t.loc["s2", "outcome_step"], 1)

    def test_errors(self):
        with self.assertRaises(KeyError):
            na.sessions(toy().drop(columns="step"), outcome_key="purchase")
        with self.assertRaises(ValueError):
            na.sessions(toy(), outcome_key="purchase", prefix_k=0)
        with self.assertRaises(ValueError):
            na.sessions(toy(), outcome_key=lambda g: ("purchase", 99))


class TestFunnel(unittest.TestCase):
    def test_counts(self):
        f = na.funnel(toy(), outcome_key="purchase")
        # depths in funnel: s1 4 (stops at purchase), s2 2, s3 1, s4 5
        self.assertEqual(f["reached"].tolist(), [4, 3, 2, 2, 1])
        self.assertEqual(f["converted"].tolist(), [1, 0, 0, 1, 0])
        self.assertEqual(f["abandoned"].tolist(), [0, 1, 0, 0, 1])
        self.assertEqual(f["continued"].tolist(), [3, 2, 2, 1, 0])
        self.assertAlmostEqual(f.loc[1, "abandon_rate"], 1 / 3)
        self.assertAlmostEqual(f.loc[0, "reach_rate"], 1.0)
        self.assertAlmostEqual(f.loc[4, "reach_rate"], 0.25)
        self.assertEqual(len(na.funnel(toy(), outcome_key="purchase", max_steps=2)), 2)

    def test_invariants_random(self):
        rs = np.random.default_rng(1)
        rows = []
        for s in range(300):
            for t in range(int(rs.integers(1, 9))):
                a = str(rs.choice(["view", "search", "cart", "purchase"], p=[.4, .3, .2, .1]))
                rows.append((f"s{s}", t, a))
        df = pd.DataFrame(rows, columns=["session", "step", "action"])
        f = na.funnel(df, outcome_key="purchase")
        r = f[["continue_rate", "abandon_rate", "convert_rate"]].sum(axis=1)
        self.assertTrue(np.allclose(r, 1.0))
        self.assertTrue((f["continued"].to_numpy()[:-1] == f["reached"].to_numpy()[1:]).all())
        t = na.sessions(df, outcome_key="purchase")
        self.assertEqual(int(f["converted"].sum()), int(t["converted"].sum()))
        self.assertEqual(int(f["abandoned"].sum()), int((t["converted"] == 0).sum()))
        self.assertEqual(int(f.loc[0, "reached"]), len(t))


if __name__ == "__main__":
    unittest.main()
