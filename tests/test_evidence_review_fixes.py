"""Regression tests for the review of the evidence gap fixes: split without test rows (review #2), correlation_audit
defaults and empty replies (#4), fast multiplier bootstrap only from 2000 rows (#6), aggregate-mode baseline note
(#8), seal registry_dir alias (#10). Offline: stub Claude client / FakeLLM, small synthetic data."""
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import simengine as se
import simengine.evidence as ev
from simengine.evidence import bedel
from simengine.evidence.sema import gercek_yok
from simengine.llm import Claude, EmptyResponseError, FakeLLM
from tests.test_evidence_gaps import FAST, _cont


# ------------------------------------------------------------------ review #2
class TestSplitWithoutTestRows(unittest.TestCase):
    def _frame(self, split, truth=True):
        df = _cont(200)
        df["split"] = split
        if not truth:
            df["truth"] = np.nan
        return df

    def test_truth_but_no_test_rows_raises(self):
        for split in ("cv", "dev", "oof"):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, "no test rows"):
                    ev.evaluate(self._frame(split), ours="good", raw="raw", **FAST)
        with self.assertRaisesRegex(ValueError, "accepted test names"):
            gercek_yok(ev.to_internal(self._frame("cv")))

    def test_awaiting_truth_still_works(self):
        self.assertTrue(gercek_yok(ev.to_internal(self._frame("cv", truth=False))))
        self.assertTrue(gercek_yok(ev.to_internal(_cont(50).drop(columns="truth"))))
        df = self._frame("cv")
        df.loc[:99, "split"] = "test"
        self.assertFalse(gercek_yok(ev.to_internal(df)))


# ------------------------------------------------------------------ review #4
class _Messages:
    """Anthropic-like messages.create: an empty reply when thinking is on and max_tokens < 200 (gap #4 shape)."""

    def __init__(self, value="42"):
        self.value, self.params = value, []

    def create(self, **p):
        self.params.append(p)
        th = p.get("thinking")
        thinking_on = not (isinstance(th, dict) and th.get("type") == "disabled")
        if thinking_on and p["max_tokens"] < 200:
            return {"content": [{"type": "thinking", "thinking": "..."}], "stop_reason": "max_tokens",
                    "model": "claude-haiku-5-5", "usage": {"input_tokens": 10, "output_tokens": p["max_tokens"]}}
        return {"content": [{"type": "text", "text": self.value}], "stop_reason": "end_turn",
                "model": "claude-haiku-5-5", "usage": {"input_tokens": 10, "output_tokens": 2}}


class _Client:
    def __init__(self):
        self.messages = _Messages()


class TestCorrelationAuditDefaults(unittest.TestCase):
    TRUTH = {f"k{i}": float(i) for i in range(8)}

    def test_haiku_defaults_do_not_abort(self):
        c = _Client()
        llm = Claude(model="claude-haiku-5-5", effort="low", client=c, api_key="x", sleep=lambda s: None)
        named, blind = ev.audit_prompts("'X' (A, J 2020)", "mean support (0-100)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ev.correlation_audit(self.TRUTH, named, blind, llm=llm, B=50)
        self.assertEqual((r["n_failed"], r["n_unparsed_named"]), (0, 0))
        self.assertTrue(all(p["thinking"] == {"type": "disabled"} for p in c.messages.params))
        self.assertTrue(all(p["max_tokens"] == 1024 for p in c.messages.params))

    def test_thinking_none_and_empty_reply_counted(self):
        c = _Client()
        llm = Claude(model="claude-haiku-5-5", effort="low", client=c, api_key="x", sleep=lambda s: None)
        named, blind = ev.audit_prompts("'X' (A, J 2020)", "mean support (0-100)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ev.correlation_audit(self.TRUTH, named, blind, llm=llm, B=50, thinking=None, max_tokens=40)
        self.assertEqual(r["durum"], "atlandi")
        self.assertEqual(r["n_failed"], 16)
        self.assertIn("EmptyResponseError", r["errors"][0])
        self.assertNotIn("thinking", c.messages.params[0])

    def test_error_slot_in_map_backend(self):
        class Flaky(FakeLLM):
            def map(self, prompts, *, max_tokens=256, parallel=4, return_exceptions=False, **kw):
                self.kw = dict(kw, return_exceptions=return_exceptions)
                return [EmptyResponseError("empty") if "k3" in u else self.complete(s, u) for s, u in prompts]
        llm = Flaky(lambda s, u: "50")
        r = ev.correlation_audit(self.TRUTH, "Guess {key} (source: S).", "Guess {key}.", llm=llm, B=50)
        self.assertEqual((r["n_failed"], r["n_unparsed_named"], r["n_unparsed_blind"]), (2, 1, 1))
        self.assertTrue(llm.kw["return_exceptions"])
        self.assertNotIn("thinking", llm.kw)            # FakeLLM has no build_request: nothing passed


# ------------------------------------------------------------------ review #6
class TestFastBootstrapThreshold(unittest.TestCase):
    def _col(self, n):
        rng = np.random.default_rng(1)
        y = rng.normal(size=n)
        return [(y, y + rng.normal(size=n))]

    def test_threshold_is_2000(self):
        self.assertEqual(bedel.HIZLI_ESIK, ev.MULTIPLIER_MAX_ROWS)
        self.assertNotIn("hizli_bootstrap", bedel.olc(self._col(1500), n=100, B=5))
        self.assertTrue(bedel.olc(self._col(2000), n=100, B=5).get("hizli_bootstrap"))

    def test_below_threshold_matches_explicit_slow_path(self):
        a = bedel.olc(self._col(1200), n=100, B=5, tohum=3)
        b = bedel.olc(self._col(1200), n=100, B=5, tohum=3, hizli=False)
        self.assertEqual(a["carpan_ga"], b["carpan_ga"])


# ------------------------------------------------------------------ review #8
class TestAggregateBaselineNote(unittest.TestCase):
    def test_note_and_flag(self):
        r = ev.evaluate(_cont(800, groups=20), ours="good", raw="raw", aggregate_by="group", primary="spearman",
                        **FAST)
        self.assertTrue(r["toplama"]["auto_baselines_cross_fitted_bias"])
        self.assertTrue(any("bilgisizken bile" in x and "baselines_oof" in x for x in r["sinirlar"]))
        self.assertTrue(any("bilgisizken bile" in x for x in r["kanit_kart"]["yanilgi"]))


# ------------------------------------------------------------------ review #10
class TestSealAlias(unittest.TestCase):
    def test_registry_dir_alias_and_research_reads_both_chains(self):
        df = pd.DataFrame({"group": ["a", "b", "c"], "pred_x": [0.1, 0.2, 0.3]})
        truth = pd.DataFrame({"group": ["a", "b", "c"], "truth": [0, 1, 1]})
        with tempfile.TemporaryDirectory() as t:
            reg = Path(t) / ".prereg"
            rec = se.seal(df, "x", registry_dir=reg)
            self.assertTrue((reg / "zincir.jsonl").exists())
            self.assertTrue(ev.verify(df, registry_dir=reg)["eslesti"])
            self.assertTrue(ev.verify_chain(registry_dir=reg)["gecerli"])
            v = se.research.verify_seal(df, registry_dir=reg)
            self.assertTrue(v["matched"])
            self.assertEqual((v["record"]["sha256"], v["record"]["seq"], v["record"]["chain"]),
                             (rec["sha256"], 0, "zincir.jsonl"))
            self.assertEqual(se.research.grade_sealed(df, truth, ["group"], registry_dir=reg)["n_joined"], 3)
            self.assertTrue(se.research.verify_chain(reg)["valid"])
            with self.assertRaises(TypeError):
                se.seal(df, "x", registry=Path(t) / "a", registry_dir=Path(t) / "b")

    def test_research_default_dir_falls_back_to_sibling(self):
        df = pd.DataFrame({"group": ["a", "b"], "pred_x": [0.1, 0.2]})
        with tempfile.TemporaryDirectory() as t:
            se.seal(df, "x", registry=Path(t) / ".onkayit")
            self.assertTrue(se.research.verify_seal(df, registry_dir=Path(t) / ".prereg")["matched"])
            self.assertFalse(se.research.verify_seal(df.assign(pred_x=0.5), registry_dir=Path(t) / ".prereg")["matched"])


if __name__ == "__main__":
    unittest.main()
