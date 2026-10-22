"""CLI `evaluate` forecast mode after the D4 rehearsal (demand forecasting): --task forecast, --primary wmape,
bias refused as primary, --aggregate sum, --ci basic, --n-configs; plus the new top-level forecast names."""
import contextlib
import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import simengine as se
from simengine import cli


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def frame(n_items=30, weeks=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        lam = rng.gamma(2.0, 3.0)
        for w in range(weeks):
            rows.append({"item": f"i{i}", "store": f"s{i % 3}", "time": f"2016-0{w + 1}-01",
                         "truth": float(rng.poisson(lam)), "pred_good": lam,
                         "pred_naive": max(lam + rng.normal(0, 3), 0.0)})
    df = pd.DataFrame(rows)
    df["store_week"] = df["store"] + "_" + df["time"]
    return df


class TestCliForecast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.csv = str(cls.dir / "bt.csv")
        frame().to_csv(cls.csv, index=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _report(self, out):
        return json.loads((Path(out) / "report.json").read_text(encoding="utf-8"))

    def test_forecast_wmape_basic_ci_n_configs(self):
        out = self.dir / "fc"
        code, so, se_ = run_cli("evaluate", self.csv, "--ours", "good", "--task", "forecast", "--primary", "wmape",
                                "--unlabeled-baselines", "naive", "--ci", "basic", "--n-configs", "6",
                                "--cutoff", "2015-01-01", "--B", "200", "--B-aggregate", "50", "--out", str(out))
        self.assertEqual(code, 0, se_)
        rep = self._report(out)
        self.assertEqual(rep["birincil"], "wmape")
        self.assertTrue(rep.get("tahmin_gorevi"))
        self.assertIn("yanlilik", rep["sonuclar"]["tahmin_good"]["metrikler"])
        self.assertIn("wmape", so)
        self.assertEqual(se.evidence.summary(rep)["task"], "forecast")
        self.assertIn("6", json.dumps(rep, ensure_ascii=False, default=str))

    def test_aggregate_sum_matches_hand_total(self):
        out = self.dir / "sum"
        code, _, se_ = run_cli("evaluate", self.csv, "--ours", "good", "--task", "forecast", "--primary", "wmape",
                               "--aggregate-by", "store_week", "--aggregate", "sum", "--B", "200",
                               "--B-aggregate", "50", "--out", str(out))
        self.assertEqual(code, 0, se_)
        rep = self._report(out)
        df = frame()
        # aggregate="sum" is the store x week total even though the frame has an item column
        g = df.groupby(["store_week"]).sum(numeric_only=True)
        want = np.abs(g.truth - g.pred_good).sum() / np.abs(g.truth).sum()
        self.assertEqual(rep["toplama"]["aggregate"], "sum")
        self.assertAlmostEqual(rep["sonuclar"]["tahmin_good"]["metrikler"]["wmape"]["deger"], want, places=8)

    def test_bias_primary_and_sum_without_group_are_refused(self):
        code, _, err = run_cli("evaluate", self.csv, "--ours", "good", "--primary", "bias",
                               "--out", str(self.dir / "b"))
        self.assertEqual(code, 2)
        self.assertIn("bias", err)
        code, _, err = run_cli("evaluate", self.csv, "--ours", "good", "--aggregate", "sum",
                               "--out", str(self.dir / "s"))
        self.assertEqual(code, 2)
        self.assertIn("--aggregate-by", err)


class TestTopLevelForecastNames(unittest.TestCase):
    def test_reexports(self):
        from simengine.methods import forecast as fc
        for name in ("persistence_forecast", "seasonal_naive", "moving_mean", "ses", "make_features",
                     "DirectForecaster", "backtest", "summarize_backtest", "point_for_metric", "llm_prior_feature"):
            self.assertIs(getattr(se, name), getattr(fc, name), name)
            self.assertIn(name, se.__all__)

    def test_backtest_moving_mean_top_level(self):
        rng = np.random.default_rng(1)
        panel = pd.DataFrame([{"id": f"s{i}", "time": t, "y": float(rng.poisson(5 + i))}
                              for i in range(4) for t in range(30)])
        bt = se.backtest(se.moving_mean, panel, origins=[20, 24], horizon=2)
        self.assertEqual(len(bt), 4 * 2 * 2)
        s = se.summarize_backtest(bt, by="origin")
        self.assertEqual(len(s), 2)


if __name__ == "__main__":
    unittest.main()
