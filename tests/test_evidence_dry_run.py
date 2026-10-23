"""simengine.evidence.dry_run: evaluate on masked truth, stamped, never the real numbers (TALP G2 #1).
Synthetic frames only, no LLM."""
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine.evidence as ev


def _binary(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (x + rng.normal(0, 0.7, n) > 0).astype(int)
    return pd.DataFrame({"group": np.arange(n), "item": rng.choice(["q1", "q2", "q3"], n), "truth": y,
                         "pred_t2": 1 / (1 + np.exp(-2 * x)), "pred_raw": 1 / (1 + np.exp(-0.3 * x))})


def _value(rep, col="pred_t2"):
    r = rep["sonuclar"]
    key = next(k for k in r if k in (col, col[len("pred_"):], "tahmin_" + col[len("pred_"):]))
    return r[key]["metrikler"][rep["birincil"]]["deger"]


def _quiet(f, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **kw)


class DryRunTest(unittest.TestCase):
    def test_numbers_differ_title_stamped_input_untouched(self):
        f = _binary()
        before = f.copy(deep=True)
        real = _quiet(ev.evaluate, f, "T2 ladder", ours="t2", raw="raw", B=50, multiplier=False)
        dry = _quiet(ev.dry_run, f, title="T2 ladder", ours="t2", raw="raw", B=50, multiplier=False)
        pd.testing.assert_frame_equal(f, before)
        self.assertEqual(dry["birincil"], real["birincil"])
        self.assertNotAlmostEqual(_value(dry), _value(real), places=3)
        self.assertLess(abs(_value(dry) - 0.5), abs(_value(real) - 0.5))        # masked truth: near chance
        self.assertTrue(dry["baslik"].startswith(ev.DRY_RUN_STAMP))
        self.assertIn("T2 ladder", dry["baslik"])
        self.assertTrue(dry["sinirlar"][0].startswith(ev.DRY_RUN_STAMP))
        self.assertEqual(dry["dry_run"]["level"], "item")
        md = ev.card_markdown(dry)
        self.assertIn(ev.DRY_RUN_STAMP, md.splitlines()[0])
        self.assertEqual(ev.summary(dry)["title"], dry["baslik"])

    def test_permute_keeps_item_marginals_marginal_mode_redraws(self):
        f = _binary()
        m, info = ev.mask_truth(f, seed=3)
        self.assertEqual(info["mode"], "permute")
        self.assertEqual(m["truth"].dtype, f["truth"].dtype)
        pd.testing.assert_series_equal(m.groupby("item").truth.sum(), f.groupby("item").truth.sum())
        self.assertGreater(info["unchanged_share"], 0.3)                        # binary: half stays by chance
        self.assertLess(info["unchanged_share"], 0.75)
        m2, info2 = ev.mask_truth(f, seed=3, mode="marginal")
        self.assertEqual(info2["mode"], "marginal")
        self.assertTrue(set(m2.truth) <= {0, 1})
        self.assertFalse(m2.truth.equals(f.truth))
        dry = _quiet(ev.dry_run, f, mode="marginal", ours="t2", B=30, multiplier=False)
        self.assertEqual(dry["baslik"], ev.DRY_RUN_STAMP)
        with self.assertRaises(ValueError):
            ev.mask_truth(f, mode="shuffle")

    def test_one_row_per_item_moves_up_a_level(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=80)
        f = pd.DataFrame({"item": [f"q{i}" for i in range(80)], "truth": y, "pred_a": y + rng.normal(0, .3, 80)})
        m, info = ev.mask_truth(f, seed=0)
        self.assertEqual(info["level"], "all")                                   # within-item would be a no-op
        self.assertLess(info["unchanged_share"], 0.1)
        real = _quiet(ev.evaluate, f, ours="a", B=30, multiplier=False)
        dry = _quiet(ev.dry_run, f, ours="a", B=30, multiplier=False)
        self.assertNotAlmostEqual(_value(dry, "pred_a"), _value(real, "pred_a"), places=3)

    def test_aggregate_by_item_is_not_masked_within_item(self):
        f = _binary()
        _, info = ev.mask_truth(f, seed=0, aggregate_by="item")
        self.assertEqual(info["level"], "all")
        _, info = ev.mask_truth(f, seed=0, aggregate_by="group")
        self.assertEqual(info["level"], "item")

    def test_distribution_one_row_per_item_shuffles_components(self):
        rng = np.random.default_rng(2)
        rows = []
        for i in range(30):
            k = 3 + i % 3                                                         # lengths differ: no cross-item swap
            t = rng.dirichlet(np.ones(k) * 2)
            rows.append({"item": f"q{i}", "truth": t.tolist(), "pred_llm": (0.7 * t + 0.3 / k).tolist()})
        f = pd.DataFrame(rows)
        m, info = ev.mask_truth(f, seed=0)
        self.assertEqual(info["level"], "components")
        self.assertTrue(all(isinstance(v, list) and len(v) == len(w) for v, w in zip(m.truth, f.truth)))
        self.assertTrue(all(np.isclose(sum(v), 1) for v in m.truth))
        real = _quiet(ev.evaluate, f, ours="llm", task="distribution", B=30, multiplier=False)
        dry = _quiet(ev.dry_run, f, ours="llm", task="distribution", B=30, multiplier=False)
        self.assertLess(_value(dry, "pred_llm"), _value(real, "pred_llm"))

    def test_turkish_aliases_missing_truth_and_constant(self):
        f = _binary().rename(columns={"truth": "gercek", "item": "madde"})
        f.loc[:9, "gercek"] = np.nan
        m, info = ev.mask_truth(f, seed=0)
        self.assertEqual(info["level"], "item")
        self.assertEqual(info["masked_rows"], len(f) - 10)
        self.assertTrue(m.gercek.iloc[:10].isna().all())
        c = pd.DataFrame({"truth": [1] * 20, "pred_a": np.linspace(0, 1, 20)})
        with self.assertRaises(ValueError):
            ev.dry_run(c, ours="a")
        with self.assertRaises(ValueError):
            ev.mask_truth(pd.DataFrame({"pred_a": [0.1]}))

    def test_exported(self):
        for name in ("dry_run", "mask_truth", "DRY_RUN_STAMP"):
            self.assertIn(name, ev.__all__)


if __name__ == "__main__":
    unittest.main()
