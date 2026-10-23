"""PPI++ 'worth N humans' default on binary tasks without a human panel (D13-W8 #7, D13-W10 #3; D13 #11c).

An event-log style binary frame (sparse positives, no item column, no LLM arm, ours_uses_llm=False) has no human
answers to stand in for: the default skips the multiplier with a 'not applicable' reason instead of printing
"Worth N humans 1.01x" and spending most of the card time on the PPI++ subsample. multiplier=True still runs it;
frames with a human panel signal (an LLM arm, several items, ours_uses_llm not False) and other tasks are unchanged.
Synthetic data, offline, no LLM."""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine import evidence as ev

CARPAN = "etkin örneklem çarpanı (PPI++ kaç insana bedel)"


def _sparse_binary(n=3000, seed=0, rate=-3.9):
    """Binary event-log frame: ~3 % positives, one row per session, a user group, segments, no item column."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(rate + 0.9 * x)))
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({"truth": y, "group": rng.integers(0, n // 5, n), "segment": rng.integers(0, 4, n).astype(str)})
    df["pred_ours"] = np.clip(p + rng.normal(0, 0.01, n), 1e-4, 1 - 1e-4)
    df["pred_T1"] = np.clip(0.5 * p + 0.015 + rng.normal(0, 0.01, n), 1e-4, 1 - 1e-4)
    df["pred_llm"] = np.clip(0.7 * p + 0.01 + rng.normal(0, 0.02, n), 1e-4, 1 - 1e-4)
    return df


def _ev(df, **kw):
    kw.setdefault("B", 100)
    kw.setdefault("B_aggregate", 40)
    kw.setdefault("task", "binary")
    kw.setdefault("ours", "ours")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rep = ev.evaluate(df, "ppi default", **kw)
    return rep, [str(x.message) for x in w]


def _bedel(rep):
    return next(r for r in rep["kanit_kart"]["satirlar"] if r["tur"] == "bedel")


def _carpan_satiri(rep):
    return next((d for d in rep["denetim"] if d["kontrol"] == CARPAN), None)


class NoHumanPanelDefaultTest(unittest.TestCase):
    def test_default_skips_multiplier_with_reason(self):
        rep, msgs = _ev(_sparse_binary(), primary="ap", baselines_oof=["T1"], ours_uses_llm=False)
        self.assertIsNone(rep["orneklem"])
        self.assertIn("not applicable", rep["carpan_uygulanamaz"])
        self.assertIn("multiplier=True", rep["carpan_uygulanamaz"])
        row = _carpan_satiri(rep)
        self.assertEqual(row["durum"], "atlandi")
        self.assertTrue(row["uygulanamaz"])
        b = _bedel(rep)
        self.assertIsNone(b["deger"])
        self.assertIn("not applicable", b["not"])
        md = ev.card(rep)
        line = next(x for x in md.splitlines() if "Worth N humans" in x)
        self.assertNotIn("×", line)
        self.assertIn("not applicable", line)
        en = next(x for x in ev.card(rep, lang="en").splitlines() if "Worth N humans" in x)
        self.assertIn("not applicable: binary task without a human panel", en)
        self.assertNotIn("ikili", en)
        self.assertFalse(any("multiplier=True" in m and "rows with" in m for m in msgs))

    def test_large_frame_no_subsample_warning(self):
        rep, msgs = _ev(_sparse_binary(n=6000, seed=1), ours_uses_llm=False)
        self.assertIsNone(rep["orneklem"])
        self.assertFalse(any("rows with multiplier=True" in m for m in msgs))

    def test_explicit_true_still_runs(self):
        rep, _ = _ev(_sparse_binary(), multiplier=True, ours_uses_llm=False)
        self.assertIsNotNone(rep["orneklem"])
        self.assertNotIn("carpan_uygulanamaz", rep)
        self.assertIsNotNone(_bedel(rep)["deger"])

    def test_explicit_false_no_not_applicable_row(self):
        rep, _ = _ev(_sparse_binary(), multiplier=False, ours_uses_llm=False)
        self.assertIsNone(rep["orneklem"])
        self.assertNotIn("carpan_uygulanamaz", rep)
        self.assertIsNone(_carpan_satiri(rep))


class HumanPanelSignalsUnchangedTest(unittest.TestCase):
    def test_raw_llm_arm_keeps_multiplier(self):
        rep, _ = _ev(_sparse_binary(), raw="llm", ours_uses_llm=False)
        self.assertIsNotNone(rep["orneklem"])
        self.assertNotIn("carpan_uygulanamaz", rep)

    def test_ours_uses_llm_unset_keeps_multiplier(self):
        rep, _ = _ev(_sparse_binary())
        self.assertIsNotNone(rep["orneklem"])

    def test_items_keep_multiplier(self):
        df = _sparse_binary(n=2400, rate=-1.0)
        df["item"] = [f"q{i % 24}" for i in range(len(df))]
        rep, _ = _ev(df, ours_uses_llm=False)
        self.assertIsNotNone(rep["orneklem"])
        self.assertNotIn("carpan_uygulanamaz", rep)

    def test_continuous_task_keeps_multiplier(self):
        rng = np.random.default_rng(3)
        y = rng.normal(size=800)
        df = pd.DataFrame({"truth": y, "pred_ours": y + rng.normal(0, .5, 800), "pred_T1": rng.normal(size=800)})
        rep, _ = _ev(df, task="continuous", ours_uses_llm=False)
        self.assertIsNotNone(rep["orneklem"])
        self.assertNotIn("carpan_uygulanamaz", rep)


if __name__ == "__main__":
    unittest.main()
