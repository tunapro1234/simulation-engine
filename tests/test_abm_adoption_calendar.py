"""Adoption curves upper bound + constant growth baseline; adoption_backtest calendar origins, alignment threshold
and ineligible-series accounting. Synthetic series only."""
import unittest

import numpy as np
import pandas as pd

import simengine.abm as A
from simengine.abm import adoption as AD
from simengine.abm import curves as C


class TestUpperBound(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(2000, 2010.0)
        self.y = 40 + 9 * (self.t - 2000)          # 40 .. 121, passes 100
        self.h = np.arange(2010, 2014.0)

    def test_default_clip_unchanged(self):
        self.assertLessEqual(C.linear_trend(self.t, self.y, self.h).max(), 100.0)
        self.assertLessEqual(C.damped_trend(self.t, self.y, self.h).max(), 100.0)

    def test_upper_raises_clip(self):
        np.testing.assert_allclose(C.linear_trend(self.t, self.y, self.h, upper=200), 40 + 9 * (self.h - 2000))
        np.testing.assert_allclose(C.linear_trend(self.t, self.y, self.h, upper=None), 40 + 9 * (self.h - 2000))
        self.assertGreater(C.damped_trend(self.t, self.y, self.h, upper=200).max(), 121)

    def test_s_curves_respect_upper(self):
        t = np.arange(1990, 2005.0)
        y = C.logistic_curve(t, 130, 0.5, 1998)
        h = np.arange(2005, 2012.0)
        for fn in (C.gompertz, C.logistic, C.bass):
            self.assertLessEqual(fn(t, y, h).max(), 100.0, fn.__name__)
            self.assertGreater(fn(t, y, h, upper=200).max(), 110, fn.__name__)
        np.testing.assert_allclose(C.logistic(t, y, h, upper=200), C.logistic_curve(h, 130, 0.5, 1998), atol=1.0)

    def test_curve_baseline_passes_kwargs(self):
        hist = pd.DataFrame(dict(series="a", t=self.t, y=self.y))
        b = C.CurveBaseline("linear", upper=200)
        p = b.predict(b.fit(hist, "a", 2009), [1, 2])
        np.testing.assert_allclose(p["y_pred"], [130, 139])
        self.assertLess(C.BassAnalogy(upper=100).upper, C.BassAnalogy(upper=200).upper)


class TestConstantGrowth(unittest.TestCase):
    def test_exponential_series_exact(self):
        t = np.arange(1990, 1998.0)
        y = 0.5 * 1.4 ** (t - 1990)
        h = np.arange(1998, 2003.0)
        pred = C.constant_growth(t, y, h, phi=1.0, offset=0.0, upper=None)
        np.testing.assert_allclose(pred, 0.5 * 1.4 ** (h - 1990), rtol=1e-9)

    def test_last_window_and_damping(self):
        t = np.arange(1990, 1998.0)
        y = np.r_[np.full(4, 1.0), 2.0, 4.0, 8.0, 8.0]          # last 3 years: 2 -> 8, g = log(4)/3
        g = np.log(4.0) / 3
        pred = C.constant_growth(t, y, [1998, 1999], phi=1.0, offset=0.0)
        np.testing.assert_allclose(pred, 8 * np.exp(g * np.array([1, 2])))
        np.testing.assert_allclose(C.constant_growth(t, y, [1998, 1999], offset=0.0),      # default phi 0.8
                                   8 * np.exp(g * np.array([0.8, 0.8 + 0.64])))
        damped = C.constant_growth(t, y, [1998, 1999], phi=0.5, offset=0.0)
        np.testing.assert_allclose(damped, 8 * np.exp(g * np.array([0.5, 0.75])))
        far = C.constant_growth(t, y, [2050], phi=0.5, offset=0.0)
        self.assertLess(far[0], 8 * np.exp(g) + 1e-9)            # damped sum -> phi / (1 - phi) = 1

    def test_clip_zero_and_short_series(self):
        t = np.arange(2000, 2006.0)
        y = np.array([0, 0, 0.5, 5, 20, 60.0])
        self.assertEqual(C.constant_growth(t, y, [2010]).max(), 100.0)
        self.assertGreater(C.constant_growth(t, y, [2010], phi=1.0, upper=None)[0], 100.0)
        np.testing.assert_allclose(C.constant_growth(t, np.zeros(6), [2007, 2008]), [0, 0])
        np.testing.assert_allclose(C.constant_growth([2000.0], [3.0], [2001, 2002]), [3, 3])
        self.assertTrue(np.isfinite(C.constant_growth(t, y, [2007])).all())

    def test_registered_as_growth(self):
        self.assertIs(A.CURVES["growth"], C.constant_growth)
        hist = pd.DataFrame(dict(series="a", t=np.arange(1990, 1996.0), y=2.0 ** np.arange(6)))
        b = A.CurveBaseline("growth", phi=1.0, offset=0.0, upper=None)
        np.testing.assert_allclose(b.predict(b.fit(hist, "a", 1995), [1])["y_pred"], [64.0])


SMALL = dict(n_sim=300, n_agents=300)
ORIGINS = (1994, 1995, 1996)


def panel():
    """Six early series (cross 1% in the 1980s) plus "late" (Bass start 1993: ~0.2% in 1994, 1% only in 1996)."""
    rng = np.random.default_rng(0)
    t = np.arange(1980, 2001, dtype=float)
    rows = []
    for i in range(6):
        y = A.bass_curve(t, 60 + 5 * i, 0.01, 0.3 + 0.05 * i, 1980 + i) + rng.normal(0, 0.3, len(t))
        rows += [dict(series=f"s{i}", t=a, y=float(np.clip(b, 0, 100))) for a, b in zip(t, y)]
    y = A.bass_curve(t, 60, 0.003, 0.4, 1993)
    rows += [dict(series="late", t=a, y=float(b)) for a, b in zip(t, y)]
    return pd.DataFrame(rows)


class TestThreshold(unittest.TestCase):
    def test_bank_aligned_at_level(self):
        i0 = int(np.flatnonzero(np.isclose(AD.GRID, 0.0))[0])
        for level in (1.0, 0.1):
            B = A.simulation_bank(200, seed=2, n_agents=200, align=level)
            self.assertGreater(len(B["C"]), 0)
            np.testing.assert_allclose(B["C"][:, i0], level, atol=1e-3)
        self.assertIsNot(A.simulation_bank(200, seed=2, n_agents=200),
                         A.simulation_bank(200, seed=2, n_agents=200, align=0.1))

    def test_threshold_validated_and_default(self):
        self.assertEqual(A.AdoptionABC().threshold, 0.01)
        self.assertEqual(A.AdoptionABC().level, 1.0)
        for bad in (0, 1.0, 5):
            with self.assertRaises(ValueError):
                A.AdoptionABC(threshold=bad)
        with self.assertWarns(UserWarning):                                  # 0.1 = 10%, not 0.1%
            A.AdoptionABC(threshold=0.1)

    def test_low_threshold_aligns_early_series(self):
        df = panel()
        late = df[df.series == "late"].set_index("t")["y"]
        self.assertTrue(0.1 <= late[1994] < 1.0 and late[1995] < 1.0)      # precondition of the fixture
        hist = df[df.t <= 1994]
        m1, m01 = A.AdoptionABC(**SMALL), A.AdoptionABC(threshold=0.001, **SMALL)
        self.assertFalse(m1.eligible(hist, "late", 1994))
        self.assertTrue(m01.eligible(hist, "late", 1994))
        with self.assertRaises(AD.IneligibleSeries) as cm:
            m1.fit(hist, "late", 1994)
        self.assertIsInstance(cm.exception, ValueError)                      # old except ValueError still works
        self.assertEqual(cm.exception.series, "late")
        p = m01.predict(m01.fit(hist, "late", 1994), [1, 2, 3])
        self.assertTrue(np.isfinite(p["y_pred"]).all())
        a = A.AdoptionABCAnalogy(threshold=0.001, **SMALL)
        self.assertTrue(np.isfinite(a.predict(a.fit(hist, "late", 1994), [1, 2])["y_pred"]).all())


class TestCalendarBacktest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = panel()[lambda d: d.t <= 1999]
        cls.methods = {"abm": A.AdoptionABC(**SMALL), "growth": A.CurveBaseline("growth")}

    def test_calendar_origins_and_relative_default(self):
        ev = A.adoption_backtest(self.df, {"growth": A.CurveBaseline("growth")}, calendar_origins=ORIGINS, horizon=3)
        self.assertEqual(sorted(ev.cutoff.unique()), [1994.0, 1995.0, 1996.0])
        self.assertIn("late", set(ev.series))                         # calendar mode: every series by default
        self.assertTrue((ev.t <= ev.cutoff + 3).all() and (ev.t > ev.cutoff).all())
        t0 = A.adoption_t0(self.df)
        r = ev[ev.series == "s0"]
        np.testing.assert_allclose(r.cutoff_rel, r.cutoff - t0["s0"])
        late = ev[ev.series == "late"]
        np.testing.assert_allclose(late.cutoff_rel, late.cutoff - 1996)      # label only; negative = before t0
        never = A.adoption_backtest(self.df.assign(y=lambda d: np.where(d.series == "late", 0.2, d.y)),
                                    {"naive": A.CurveBaseline("naive")}, calendar_origins=ORIGINS, horizon=3)
        self.assertTrue(never[never.series == "late"].cutoff_rel.isna().all())
        self.assertEqual(ev.attrs["n_ineligible"], 0)
        rel = A.adoption_backtest(self.df, {"growth": A.CurveBaseline("growth")}, cutoffs=(3, 5))
        self.assertEqual(set(rel.cutoff_rel), {3, 5})

    def test_on_ineligible_raise_is_default(self):
        with self.assertRaises(AD.IneligibleSeries):
            A.adoption_backtest(self.df, self.methods, calendar_origins=ORIGINS, horizon=3)
        with self.assertRaises(ValueError):
            A.adoption_backtest(self.df, self.methods, calendar_origins=ORIGINS, on_ineligible="drop")

    def test_on_ineligible_skip_counts_and_drops_blocks(self):
        ev = A.adoption_backtest(self.df, self.methods, calendar_origins=ORIGINS, horizon=3, on_ineligible="skip")
        self.assertEqual(ev.attrs["n_ineligible"], 1)
        self.assertEqual(ev.attrs["ineligible_series"], ["late"])
        self.assertEqual(sorted(b["cutoff"] for b in ev.attrs["ineligible"]), [1994.0, 1995.0])
        self.assertEqual({b["method"] for b in ev.attrs["ineligible"]}, {"abm"})
        late = ev[ev.series == "late"]
        self.assertEqual(sorted(late.cutoff.unique()), [1996.0])             # 1% reached in 1996
        self.assertFalse(ev[["pred_abm", "pred_growth", "lo80_abm"]].isna().any().any())

    def test_skip_lists_every_method_that_cannot_align(self):
        methods = {"abm": A.AdoptionABC(**SMALL), "bass_ctrl": A.BassABC(**SMALL), "growth": A.CurveBaseline("growth")}
        ev = A.adoption_backtest(self.df, methods, calendar_origins=ORIGINS, horizon=3, on_ineligible="skip")
        got = sorted((b["method"], b["cutoff"]) for b in ev.attrs["ineligible"])
        self.assertEqual(got, [("abm", 1994.0), ("abm", 1995.0), ("bass_ctrl", 1994.0), ("bass_ctrl", 1995.0)])
        self.assertEqual(ev.attrs["n_ineligible"], 1)

    def test_on_ineligible_nan_keeps_rows(self):
        ev = A.adoption_backtest(self.df, self.methods, calendar_origins=ORIGINS, horizon=3, on_ineligible="nan")
        late = ev[(ev.series == "late") & (ev.cutoff < 1996)]
        self.assertGreater(len(late), 0)
        self.assertTrue(late.pred_abm.isna().all() and late.lo80_abm.isna().all())
        self.assertTrue(late.pred_growth.notna().all())
        self.assertEqual(ev.attrs["n_ineligible"], 1)


class TestBankWindow(unittest.TestCase):
    """Calendar origins reach cutoffs decades after the crossing; past R_MAX the grid index used to clip silently
    (flat forecast at the grid-end value, below the last observation)."""

    def test_old_series_past_window_is_ineligible(self):
        t = np.arange(1940, 2002, dtype=float)
        rows = []
        for i, start in enumerate((1945, 1984, 1985, 1986)):
            y = A.bass_curve(t, 90, 0.005, 0.15 if i == 0 else 0.4, start)
            rows += [dict(series=f"s{i}", t=a, y=float(b)) for a, b in zip(t, y)]
        df = pd.DataFrame(rows)
        hist = df[df.t <= 1996]
        m = A.AdoptionABC(**SMALL)
        self.assertFalse(m.eligible(hist, "s0", 1996))
        with self.assertRaises(AD.IneligibleSeries) as cm:
            m.fit(hist, "s0", 1996)
        self.assertIn("bank covers", str(cm.exception))
        self.assertTrue(m.eligible(hist, "s1", 1996))
        fit = m.fit(df[df.t <= 1990], "s1", 1990)                        # fit fine, forecast past the window
        with self.assertRaises(AD.IneligibleSeries):
            m.predict(fit, [40])
        a = A.AdoptionABCAnalogy(**SMALL)                                # old donor truncated, not clipped
        self.assertTrue(np.isfinite(a.predict(a.fit(hist, "s1", 1996), [1, 5])["y_pred"]).all())
        ev = A.adoption_backtest(df, {"abm": m, "naive": A.CurveBaseline("naive")}, calendar_origins=(1996,),
                                 horizon=5, on_ineligible="skip")
        self.assertEqual(ev.attrs["ineligible_series"], ["s0"])
        self.assertNotIn("s0", set(ev.series))


class TestFitMany(unittest.TestCase):
    def test_skip_and_raise(self):
        df = panel()
        fm = AD.fit_many(A.AdoptionABC(**SMALL), df, cutoff=1994)
        self.assertEqual(fm.ineligible, ["late"])
        self.assertEqual(fm.n_ineligible, 1)
        self.assertEqual(sorted(fm.models), [f"s{i}" for i in range(6)])
        fm01 = AD.fit_many(A.AdoptionABC(threshold=0.001, **SMALL), df, cutoff=1994)
        self.assertEqual(fm01.n_ineligible, 0)
        with self.assertRaises(AD.IneligibleSeries):
            AD.fit_many(A.AdoptionABC(**SMALL), df, cutoff=1994, on_ineligible="raise")
        with self.assertRaises(ValueError):                                  # "nan" is a backtest option only
            AD.fit_many(A.AdoptionABC(**SMALL), df, cutoff=1994, on_ineligible="nan")


if __name__ == "__main__":
    unittest.main()
