"""CLI `evaluate` after the rehearsal gaps: awaiting truth (#16), one seal (#3), declared OOF baselines (#13),
model ceiling (#14), group-level evaluation (#15), multiplier subsample (#12); plus the new top-level names."""
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


def frame(n=80, groups=10, truth="full", seed=0):
    r = np.random.default_rng(seed)
    g = np.array([f"g{i % groups}" for i in range(n)])
    shift = {f"g{i}": r.normal() for i in range(groups)}
    t = np.array([shift[x] for x in g]) + r.normal(0, 1, n)
    df = pd.DataFrame({"group": g, "pred_ours": t + r.normal(0, 0.8, n), "pred_raw": r.normal(size=n),
                       "pred_oofbase": t + r.normal(0, 1.2, n), "pred_oracle": t + r.normal(0, 0.3, n)})
    if truth == "full":
        df["truth"] = t
    elif truth == "nan":
        df["truth"] = np.nan
    return df


class TestCliEvaluateGaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _csv(self, name, df):
        p = self.dir / f"{name}.csv"
        df.to_csv(p, index=False)
        return str(p)

    def _report(self, out):
        return json.loads((Path(out) / "report.json").read_text(encoding="utf-8"))

    def test_awaiting_truth_no_column_and_all_nan(self):
        for kind in ("none", "nan"):
            out = self.dir / f"await-{kind}"
            code, txt, err = run_cli("evaluate", self._csv(f"await-{kind}", frame(truth=kind)), "--ours", "ours",
                                     "--task", "continuous", "--primary", "spearman", "--B", "40", "--out", str(out),
                                     "--lang", "en")
            self.assertEqual(code, 0, kind)
            self.assertIn("badge: awaiting truth", txt)
            self.assertIn("no --prereg", err)
            self.assertTrue((out / "card.md").exists())
            self.assertTrue((out / "dashboard.html").exists())

    def test_awaiting_truth_with_seal_any_api(self):
        df = frame(truth="none")
        reg = self.dir / "seal-ev"
        se.seal(df, "test predictions before truth", registry=reg)
        out = self.dir / "await-sealed"
        code, txt, err = run_cli("evaluate", self._csv("await-sealed", df), "--ours", "ours", "--task", "continuous",
                                 "--primary", "spearman", "--prereg", str(reg), "--out", str(out), "--B", "40")
        self.assertEqual(code, 0)
        self.assertIn("awaiting truth", txt)
        self.assertNotIn("no --prereg", err)
        self.assertTrue(self._report(out)["onkayit"])
        # a research.seal registry (.prereg/chain.jsonl) is accepted too
        reg2 = self.dir / "seal-research"
        se.research.seal(df, "research seal", reg2)
        code, _, err = run_cli("evaluate", self._csv("await-sealed2", df), "--ours", "ours", "--task", "continuous",
                               "--prereg", str(reg2), "--out", str(self.dir / "await-sealed2"), "--B", "40")
        self.assertEqual(code, 0, err)

    def test_prereg_missing_names_both_chains(self):
        code, _, err = run_cli("evaluate", self._csv("x", frame()), "--ours", "ours", "--prereg",
                               str(self.dir / "empty-reg"))
        self.assertEqual(code, 2)
        self.assertIn("zincir.jsonl", err)
        self.assertIn("chain.jsonl", err)

    def test_baselines_oof_and_ceiling(self):
        out = self.dir / "oof"
        code, txt, err = run_cli("evaluate", self._csv("oof", frame()), "--ours", "ours", "--raw", "raw",
                                 "--task", "continuous", "--primary", "spearman", "--baselines-oof", "oofbase",
                                 "--ceiling-col", "pred_oracle", "--ceiling-label", "oracle", "--no-multiplier",
                                 "--B", "60", "--out", str(out))
        self.assertEqual(code, 0, err)
        rep = self._report(out)
        s = se.summary(rep)
        self.assertEqual([b.split("_", 1)[-1] for b in s["declared_baselines"]], ["oofbase"])   # internal column
        self.assertFalse(s["only_trivial_baselines"])
        self.assertEqual(rep["tavan_turu"], "model")
        self.assertEqual(s["ceiling_kind"], "oracle/model")

    def test_aggregate_by_group(self):
        out = self.dir / "agg"
        code, _, err = run_cli("evaluate", self._csv("agg", frame(n=120, groups=12)), "--ours", "ours",
                               "--task", "continuous", "--primary", "spearman", "--aggregate-by", "group",
                               "--no-multiplier", "--B", "60", "--out", str(out))
        self.assertEqual(code, 0, err)
        rep = self._report(out)
        self.assertIsNotNone(rep.get("toplama"))
        self.assertIsNotNone(se.summary(rep)["aggregate"])

    def test_multiplier_max_rows_flag(self):
        args = cli.build_parser().parse_args(["evaluate", "t.csv", "--multiplier-max-rows", "0"])
        self.assertEqual(args.multiplier_max_rows, 0)
        self.assertIsNone(cli.build_parser().parse_args(["evaluate", "t.csv"]).multiplier_max_rows)
        out = self.dir / "mmr"
        code, _, err = run_cli("evaluate", self._csv("mmr", frame(n=60)), "--ours", "ours", "--task",
                               "continuous", "--multiplier-max-rows", "30", "--B", "30", "--B-aggregate", "20",
                               "--out", str(out))
        self.assertEqual(code, 0, err)
        self.assertIn("alt_orneklem", json.dumps(self._report(out).get("orneklem") or {}))


class TestTopLevelNames(unittest.TestCase):
    def test_new_entry_points(self):
        self.assertIs(se.seal, se.evidence.seal)
        self.assertIs(se.chain, se.evidence.chain)
        self.assertIs(se.EmptyResponseError, se.llm.EmptyResponseError)
        self.assertIs(se.ThinkingBudgetWarning, se.llm.ThinkingBudgetWarning)
        self.assertIs(se.looks_like_refusal, se.llm.looks_like_refusal)
        self.assertIs(se.CostLedger, se.llm.CostLedger)
        self.assertIs(se.GroupTransfer, se.methods.GroupTransfer)
        self.assertIs(se.oof_predict, se.methods.oof_predict)
        self.assertIs(se.correlation_audit, se.evidence.correlation_audit)
        self.assertIs(se.evaluate_aggregate, se.evidence.evaluate_aggregate)
        self.assertIs(se.survey_frame, se.sim.survey_frame)
        for name in se.__all__:
            getattr(se, name)


if __name__ == "__main__":
    unittest.main()
