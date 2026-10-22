"""CLI after blind rehearsal D3: evaluate --memorization (#8), --offset-col (#2), --noise-ceiling (#7), --blind-col
(#11), --compare (#12), the `seal` command with --no-copy / --archive-dir (#10), and the new top-level names.
Offline: small synthetic tables only."""
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
import simengine.evidence as ev
from simengine import cli

WARN = {"durum": "uyari", "ayrinti": "correlation audit: named 0.78, blind 0.17", "deger": 0.61, "alt": 0.49,
        "ust": 0.72}


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def items(G=40, n=10, seed=0):
    """Synthetic new-item table: member offset um, item effect, rating = um + effect + noise."""
    r = np.random.default_rng(seed)
    item = np.repeat(np.arange(G), n)
    um = r.normal(3.5, 0.4, G * n)
    eff = r.normal(0, 0.3, G)
    df = pd.DataFrame({"item_id": item, "um": um, "truth": um + eff[item] + r.normal(0, 0.8, G * n)})
    df["pred_ours"] = um + eff[item] + r.normal(0, 0.15, G)[item]
    df["pred_base"] = um + 0.5 * eff[item] + r.normal(0, 0.3, G)[item]
    df["pred_blind"] = um + r.normal(0, 0.3, G)[item]
    return df


class TestCliEvaluateD3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.df = items()
        cls.csv = cls.dir / "pairs.csv"
        cls.df.to_csv(cls.csv, index=False)
        cls.audit = cls.dir / "audit.json"
        cls.audit.write_text(json.dumps(WARN), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _report(self, out):
        return json.loads((Path(out) / "report.json").read_text(encoding="utf-8"))

    def test_all_new_flags_match_python(self):
        out = self.dir / "full"
        code, stdout, _ = run_cli("evaluate", str(self.csv), "--ours", "ours", "--primary", "spearman",
                                  "--baselines-oof", "base", "--aggregate-by", "item_id", "--offset-col", "um",
                                  "--noise-ceiling", "split_half", "--noise-ceiling-repeats", "30",
                                  "--memorization", str(self.audit), "--blind-col", "blind",
                                  "--compare", "ours:base", "--compare", "blind::c: blind ~ 0",
                                  "--B", "60", "--B-aggregate", "40", "--no-multiplier", "--out", str(out))
        self.assertEqual(code, 0, stdout)
        rep = self._report(out)
        s = ev.summary(rep)
        self.assertEqual(rep["ofset"]["column"], "um")
        self.assertEqual(s["ceiling_kind"], "sampling noise (variance components)")
        self.assertIn("recognition/memorization", s["badge_caveat"] or "")
        kinds = [c.get("kind") for c in s["comparisons"]]
        self.assertIn("blind", kinds)
        self.assertGreaterEqual(len(s["comparisons"]), 3)
        # same numbers as the Python call with the same arguments
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            py = ev.evaluate(self.df, title="pairs", ours="ours", primary="spearman", baselines_oof=["base"],
                             aggregate_by="item_id", offset_col="um", noise_ceiling="split_half",
                             noise_ceiling_repeats=30, memorization=WARN, blind_col="blind",
                             compare=[("ours", "base"), ("blind", None, "c: blind ~ 0")], B=60, B_aggregate=40,
                             multiplier=False)
        self.assertAlmostEqual(s["ours"]["value"], ev.summary(py)["ours"]["value"], places=10)
        self.assertEqual([round(c["diff"], 10) for c in s["comparisons"]],
                         [round(c["diff"], 10) for c in ev.summary(py)["comparisons"]])
        self.assertIn("c: blind ~ 0", (out / "card.md").read_text(encoding="utf-8"))

    def test_memorization_flag_removes_skipped_audit(self):
        base = ["evaluate", str(self.csv), "--ours", "ours", "--primary", "spearman", "--aggregate-by", "item_id",
                "--offset-col", "um", "--B", "40", "--B-aggregate", "30", "--no-multiplier"]
        _, _, _ = run_cli(*base, "--out", str(self.dir / "nomem"))
        _, _, _ = run_cli(*base, "--memorization", str(self.audit), "--out", str(self.dir / "mem"))
        self.assertIsNone(ev.summary(self._report(self.dir / "nomem"))["badge_caveat"])
        self.assertIsNotNone(ev.summary(self._report(self.dir / "mem"))["badge_caveat"])
        _, _, _ = run_cli(*base, "--memorization", str(self.audit), "--ours-no-llm", "--out", str(self.dir / "nollm"))
        self.assertIsNone(ev.summary(self._report(self.dir / "nollm"))["badge_caveat"])

    def test_bad_inputs_exit_2(self):
        code, _, err = run_cli("evaluate", str(self.csv), "--ours", "ours", "--memorization",
                               str(self.dir / "missing.json"))
        self.assertEqual(code, 2)
        self.assertIn("save_json", err)
        code, _, err = run_cli("evaluate", str(self.csv), "--ours", "ours", "--compare", ":base")
        self.assertEqual(code, 2)

    def test_compare_arg_parsing(self):
        self.assertEqual(cli._compare_arg("P1:L1"), ("P1", "L1"))
        self.assertEqual(cli._compare_arg("P2"), ("P2", None))
        self.assertEqual(cli._compare_arg("P2:"), ("P2", None))
        self.assertEqual(cli._compare_arg("P2::c: P2 = 0"), ("P2", None, "c: P2 = 0"))
        self.assertEqual(cli._compare_arg("P1:L1:b"), ("P1", "L1", "b"))


class TestCliSeal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.pred = self.dir / "pred.csv"
        pd.DataFrame({"id": [1, 2, 3], "pred": [0.1, 0.2, 0.3]}).to_csv(self.pred, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_keeps_copy_and_warns(self):
        reg = self.dir / "reg"
        code, out, err = run_cli("seal", str(self.pred), "test preds", "--registry", str(reg))
        self.assertEqual(code, 0)
        self.assertIn("sha256", out)
        self.assertIn("git-ignored", err)
        self.assertEqual(len(list(reg.glob("*.csv.gz"))), 1)
        self.assertEqual(len(ev.chain(reg)), 1)

    def test_no_copy_writes_chain_only_and_evaluate_reads_it(self):
        reg = self.dir / "reg"
        code, out, err = run_cli("seal", str(self.pred), "test preds", "--registry", str(reg), "--no-copy")
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in reg.iterdir()), ["zincir.jsonl"])
        self.assertNotIn("git-ignored", err)
        self.assertIsNone(ev.chain(reg)[0]["arsiv"])
        self.assertTrue(ev.verify_chain(reg)["gecerli"])
        # an awaiting-truth card attaches the copy-less seal
        frame = pd.DataFrame({"group": ["a", "b", "c"], "pred_ours": [0.1, 0.2, 0.3]})
        fcsv = self.dir / "test.csv"
        frame.to_csv(fcsv, index=False)
        code, _, _ = run_cli("evaluate", str(fcsv), "--ours", "ours", "--primary", "spearman", "--prereg", str(reg),
                             "--out", str(self.dir / "ev"))
        self.assertEqual(code, 0)
        rep = json.loads((self.dir / "ev" / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(rep["onkayit"])

    def test_archive_dir_outside_registry(self):
        reg, arc = self.dir / "reg", self.dir / "archive"
        code, _, _ = run_cli("seal", str(self.pred), "p", "--registry", str(reg), "--archive-dir", str(arc))
        self.assertEqual(code, 0)
        self.assertEqual(len(list(arc.glob("*.csv.gz"))), 1)
        self.assertEqual(list(reg.glob("*.csv.gz")), [])
        self.assertNotIn(str(arc), (reg / "zincir.jsonl").read_text(encoding="utf-8"))

    def test_errors(self):
        code, _, err = run_cli("seal", str(self.pred), "p", "--registry", str(self.dir / "r"), "--no-copy",
                               "--archive-dir", str(self.dir / "a"))
        self.assertEqual(code, 2)
        code, _, _ = run_cli("seal", str(self.dir / "nope.csv"), "p")
        self.assertEqual(code, 2)


class TestTopLevelD3Names(unittest.TestCase):
    def test_reexports(self):
        self.assertIs(se.paired_diff, ev.paired_diff)
        self.assertIs(se.item_holdout_features, se.methods.item_holdout_features)
        self.assertIs(se.MFBaseline, se.methods.MFBaseline)
        self.assertIs(se.ItemHoldout, se.methods.ItemHoldout)
        self.assertIs(se.procrustes_align, se.methods.procrustes_align)
        self.assertIs(se.compose_tag, se.llm.compose_tag)
        for n in ("paired_diff", "item_holdout_features", "MFBaseline", "compose_tag"):
            self.assertIn(n, se.__all__)


if __name__ == "__main__":
    unittest.main()
