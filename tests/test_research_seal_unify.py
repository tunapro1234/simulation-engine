"""One seal (#3): `se.seal` is `simengine.evidence.seal`; `simengine.research.seal` keeps its own chain
(`.prereg/chain.jsonl`) and evidence.evaluate(prereg=...) reads it."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import simengine as se
import simengine.evidence as ev
from simengine import research


def _pred():
    return pd.DataFrame({"unit": range(30), "pred_a": [i / 30 for i in range(30)]})


class TestResearchSealStillWorks(unittest.TestCase):
    def test_top_level_mapping(self):
        self.assertIs(se.seal, ev.seal)
        self.assertEqual(se.seal.__module__, "simengine.evidence.api")

    def test_research_seal_chain_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            reg = Path(t) / ".prereg"
            rec = research.seal(_pred(), "r", registry_dir=reg)
            self.assertTrue((reg / "chain.jsonl").exists())
            self.assertFalse((reg / "zincir.jsonl").exists())
            self.assertTrue(research.verify_chain(reg)["valid"])
            self.assertTrue(research.verify_seal(_pred(), rec["sha256"], registry_dir=reg)["matched"])

    def test_evidence_reads_research_chain(self):
        with tempfile.TemporaryDirectory() as t:
            reg = Path(t) / ".prereg"
            rec = research.seal(_pred(), "r", registry_dir=reg)
            recs = ev.chain(reg)
            self.assertEqual(recs[0]["sha256"], rec["sha256"])
            self.assertEqual(recs[0]["aciklama"], "r")
            self.assertEqual(recs[0]["satir"], 30)

    def test_evidence_seal_goes_to_onkayit(self):
        with tempfile.TemporaryDirectory() as t:
            reg = Path(t) / ".onkayit"
            se.seal(_pred(), "e", registry=reg)
            self.assertTrue((reg / "zincir.jsonl").exists())
            self.assertTrue(ev.verify_chain(reg)["gecerli"])


if __name__ == "__main__":
    unittest.main()
