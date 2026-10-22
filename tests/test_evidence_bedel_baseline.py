"""bedel.worth_vs_baseline: 'worth N humans' for ours next to the strongest baseline, paired. Synthetic data."""
import unittest

import numpy as np
import pandas as pd

from simengine.evidence import bedel, extras


def _veri(N=1000, tohum=0, ours_gurultu=0.05, base_gurultu=0.8):
    rng = np.random.default_rng(tohum)
    y = rng.normal(0, 1, N)
    base = y + rng.normal(0, base_gurultu, N)
    return rng, y, base, base + rng.normal(0, ours_gurultu, N)


class TestWorthVsBaseline(unittest.TestCase):
    def test_ours_is_baseline_plus_noise_same_as_baseline(self):
        _, y, base, ours = _veri()
        r = bedel.worth_vs_baseline([(y, ours, base)], "ours", "T2", B=100)
        self.assertTrue(r["same_as_baseline"])
        self.assertGreater(r["multiplier_baseline"], 1.5)          # the baseline carries real information
        self.assertLess(abs(r["difference"]), 0.1)
        lo, hi = r["difference_ci"]
        self.assertLessEqual(lo, 0)
        self.assertGreaterEqual(hi, 0)
        # paired CI is much narrower than either arm's own CI
        self.assertLess(hi - lo, 0.5 * np.diff(r["multiplier_ci_ours"])[0])

    def test_identical_prior_difference_exactly_zero(self):
        _, y, base, _ = _veri()
        r = bedel.worth_vs_baseline([(y, base, base)], B=50)
        self.assertEqual(r["difference"], 0.0)
        self.assertEqual(r["difference_ci"], [0.0, 0.0])
        self.assertTrue(r["same_as_baseline"])

    def test_ours_adds_information_not_same(self):
        rng, y, _, ours = _veri()
        noise = rng.normal(0, 1, len(y))
        r = bedel.worth_vs_baseline([(y, ours, noise)], B=100)
        self.assertFalse(r["same_as_baseline"])
        self.assertGreater(r["difference_ci"][0], 0)
        self.assertGreater(r["humans_ours"], r["humans_baseline"])

    def test_frame_with_items_and_missing_rows(self):
        rng = np.random.default_rng(3)
        rows = []
        for m in range(3):
            y = rng.normal(m, 1, 400)
            base = y + rng.normal(0, 0.8, 400)
            rows.append(pd.DataFrame({"item": f"q{m}", "truth": y, "pred_T2": base,
                                      "pred_ours": base + rng.normal(0, 0.05, 400)}))
        df = pd.concat(rows, ignore_index=True)
        df.loc[5, "pred_ours"] = np.nan            # dropped for both arms
        r = bedel.worth_vs_baseline(df, "pred_ours", "T2", B=60)
        self.assertEqual(r["sutun"], 3)
        self.assertEqual((r["ours"], r["baseline"]), ("ours", "T2"))
        self.assertEqual(r["ga_turu"], "birim bootstrap")
        self.assertTrue(r["same_as_baseline"])
        with self.assertRaises(KeyError):
            bedel.worth_vs_baseline(df, "ours", "missing")
        with self.assertRaises(ValueError):
            bedel.worth_vs_baseline(df, "ours")

    def test_item_bootstrap_from_20_columns(self):
        rng = np.random.default_rng(4)
        S = []
        for _ in range(bedel.SUTUN_BOOT_ESIK):
            y = rng.normal(0, 1, 300)
            base = y + rng.normal(0, 0.8, 300)
            S.append((y, base + rng.normal(0, 0.05, 300), base))
        r = bedel.worth_vs_baseline(S, R=200, B=200)
        self.assertEqual(r["ga_turu"], "sütun bootstrap")
        self.assertTrue(r["same_as_baseline"])

    def test_section_renders(self):
        _, y, base, ours = _veri()
        r = bedel.worth_vs_baseline([(y, ours, base)], "ours", "T2", B=50)
        s = bedel.section(r)
        self.assertEqual(s["kind"], "worth_vs_baseline")
        self.assertEqual(len(s["rows"]), 3)
        md = extras.section_markdown(s)
        self.assertIn("strongest baseline alone ~", md)
        self.assertIn("Same as baseline", md)
        self.assertIn("unit-bootstrap", md)
        self.assertIn("<table>", extras.section_html(s))

    def test_ours_point_equals_olc_same_seed(self):
        # the card's single-item bedel row is olc at the same seed: the ours figure must reproduce it
        _, y, base, ours = _veri(N=400)
        r = bedel.worth_vs_baseline([(y, ours, base)], B=20, seed=3)
        self.assertAlmostEqual(r["multiplier_ours"], bedel.olc([(y, ours)], B=20, tohum=3)["carpan"], places=12)

    def test_single_triple_and_too_few_rows(self):
        _, y, base, ours = _veri(N=400)
        a = bedel.worth_vs_baseline((y, ours, base), B=20, seed=1)
        b = bedel.worth_vs_baseline([(y, ours, base)], B=20, seed=1)
        self.assertEqual(a, b)
        with self.assertRaisesRegex(ValueError, "2n = 200"):
            bedel.worth_vs_baseline((y[:150], ours[:150], base[:150]), B=20)

    def test_olc_unchanged(self):
        _, y, base, _ = _veri(N=400)
        a = bedel.olc([(y, base)], B=20)
        b = bedel.olc([(y, base)], B=20)
        self.assertEqual(a, b)
        self.assertNotIn("same_as_baseline", a)


if __name__ == "__main__":
    unittest.main()
