"""CLI `evaluate` --block-col / --point-pairs / --split (D4 #8): the values reach evaluate() unchanged
(mocked evaluate) and a tiny forecast frame runs end to end with them."""
import contextlib
import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from simengine import cli
from simengine.evidence import api
from simengine.evidence.sema import TEST_ADLARI


class _Stop(Exception):
    pass


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def frame(n_items=20, weeks=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        lam = rng.gamma(2.0, 3.0)
        for w in range(weeks):
            rows.append({"item": f"i{i}", "time": f"2016-0{w + 1}-01", "truth": float(rng.poisson(lam)),
                         "pred_good": lam, "pred_good_median": np.floor(lam),
                         "pred_naive": max(lam + rng.normal(0, 3), 0)})
    return pd.DataFrame(rows)


class TestCliW9(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.csv = str(cls.dir / "fc.csv")
        frame().to_csv(cls.csv, index=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _kwargs(self, *extra):
        """evaluate() kwargs the CLI builds (evaluate mocked, stopped before any output is written)."""
        with mock.patch("simengine.evidence.evaluate", side_effect=_Stop) as ev:
            args = cli.build_parser().parse_args(["evaluate", self.csv, "--ours", "good", *extra])
            with self.assertRaises(_Stop):
                cli.cmd_evaluate(args)
        return ev.call_args.kwargs

    def test_defaults_pass_nothing(self):
        kw = self._kwargs()
        self.assertIsNone(kw.get("block_col"))
        self.assertIsNone(kw.get("point_pairs"))
        self.assertIsNone(kw.get("split"))

    def test_block_col_reaches_evaluate(self):
        self.assertEqual(self._kwargs("--block-col", "time")["block_col"], "time")

    def test_point_pairs_auto_and_list(self):
        self.assertEqual(self._kwargs("--point-pairs", "auto")["point_pairs"], "auto")
        kw = self._kwargs("--point-pairs", "good:good_median, naive:pred_naive")
        self.assertEqual(kw["point_pairs"], [("good", "good_median"), ("naive", "pred_naive")])

    def test_point_pairs_bad_format_is_one_line_error(self):
        code, _, err = run_cli("evaluate", self.csv, "--ours", "good", "--point-pairs", "good")
        self.assertEqual(code, 2)
        self.assertIn("--point-pairs", err)
        self.assertEqual(len(err.strip().splitlines()), 1)

    def test_split_reaches_evaluate(self):
        for v in ("test", "oof", "cv", "time", "holdout"):
            self.assertEqual(self._kwargs("--split", v)["split"], v)

    def test_split_choices_match_api(self):
        # the parser spells the accepted labels out (no pandas import for --help); keep them in sync with evaluate
        self.assertEqual(set(cli.SPLIT_NAMES), set(TEST_ADLARI) | set(api.SPLIT_DECLARED))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["evaluate", self.csv, "--split", "train"])

    def test_help_mentions_flags(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["evaluate", "--help"])
        h = out.getvalue()
        for flag in ("--block-col", "--point-pairs", "--split"):
            self.assertIn(flag, h)

    def test_end_to_end_tiny_frame(self):
        out = self.dir / "e2e"
        code, so, err = run_cli("evaluate", self.csv, "--ours", "good", "--task", "forecast", "--primary", "mae",
                                "--unlabeled-baselines", "naive", "--block-col", "time", "--point-pairs", "auto",
                                "--split", "time", "--no-multiplier", "--B", "50", "--B-aggregate", "20",
                                "--out", str(out), "--lang", "en")
        self.assertEqual(code, 0, err)
        rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
        s = api.summary(rep)
        self.assertIn((s.get("block_bootstrap") or {}).get("column"), ("time", "zaman"))   # internal name
        audits = json.dumps(rep.get("denetim"), ensure_ascii=False)
        self.assertIn("split=", audits)
        card = (out / "card.md").read_text(encoding="utf-8")
        self.assertIn("median", card.lower())

    def test_point_pairs_unknown_column_fails_before_evaluate(self):
        # a misspelt median column used to surface only after the whole bootstrap run
        with mock.patch("simengine.evidence.evaluate") as ev:
            code, _, err = run_cli("evaluate", self.csv, "--ours", "good", "--point-pairs", "good:good_medain")
        self.assertEqual(code, 2)
        self.assertIn("pred_good_medain", err)
        self.assertIn("--point-pairs", err)
        ev.assert_not_called()

    def test_point_pairs_auto_without_median_column_says_so(self):
        csv = str(self.dir / "nomed.csv")
        frame().drop(columns="pred_good_median").to_csv(csv, index=False)
        code, _, err = run_cli("evaluate", csv, "--ours", "good", "--task", "forecast", "--primary", "mae",
                               "--unlabeled-baselines", "naive", "--point-pairs", "auto", "--no-multiplier",
                               "--B", "20", "--B-aggregate", "10", "--out", str(self.dir / "nomed"))
        self.assertEqual(code, 0, err)
        self.assertIn("--point-pairs auto added no mean/median row", err)
        self.assertIn("pred_good_median", err)


if __name__ == "__main__":
    unittest.main()
