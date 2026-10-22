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
        pred = C.constant_growth(t, y, h, offset=0.0, upper=None)
        np.testing.assert_allclose(pred, 0.5 * 1.4 ** (h - 1990), rtol=1e-9)

    def test_last_window_and_damping(self):
        t = np.arange(1990, 1998.0)
        y = np.r_[np.full(4, 1.0), 2.0, 4.0, 8.0, 8.0]          # last 3 years: 2 -> 8, g = log(4)/3
        g = np.log(4.0) / 3
        pred = C.constant_growth(t, y, [1998, 1999], offset=0.0)
        np.testing.assert_allclose(pred, 8 * np.exp(g * np.array([1, 2])))
        damped = C.constant_growth(t, y, [1998, 1999], phi=0.5, offset=0.0)
        np.testing.assert_allclose(damped, 8 * np.exp(g * np.array([0.5, 0.75])))
        far = C.constant_growth(t, y, [2050], phi=0.5, offset=0.0)
        self.assertLess(far[0], 8 * np.exp(g) + 1e-9)            # damped sum -> phi / (1 - phi) = 1

    def test_clip_zero_and_short_series(self):
        t = np.arange(2000, 2006.0)
        y = np.array([0, 0, 0.5, 5, 20, 60.0])
        self.assertEqual(C.constant_growth(t, y, [2010]).max(), 100.0)
        self.assertGreater(C.constant_growth(t, y, [2010], upper=None)[0], 100.0)
        np.testing.assert_allclose(C.constant_growth(t, np.zeros(6), [2007, 2008]), [0, 0])
        np.testing.assert_allclose(C.constant_growth([2000.0], [3.0], [2001, 2002]), [3, 3])
        self.assertTrue(np.isfinite(C.constant_growth(t, y, [2007])).all())

    def test_registered_as_growth(self):
        self.assertIs(A.CURVES["growth"], C.constant_growth)
        hist = pd.DataFrame(dict(series="a", t=np.arange(1990, 1996.0), y=2.0 ** np.arange(6)))
        b = A.CurveBaseline("growth", offset=0.0, upper=None)
        np.testing.assert_allclose(b.predict(b.fit(hist, "a", 1995), [1])["y_pred"], [64.0])


if __name__ == "__main__":
    unittest.main()
