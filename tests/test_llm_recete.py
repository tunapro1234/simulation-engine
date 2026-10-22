"""Offline tests for the recipe-rehearsal llm gaps: G1 #7 (`Claude(spend_log=..., ledger=CostLedger(...))` raised
ValueError), G4 #5 (`llm.stats()` TypeError: stats was a plain property), G6 #4 (no `spent_usd`).
No network, no real API or CLI calls (FakeClient / FakeRun from test_llm)."""
from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from simengine.llm import AGENT_ENV, SPEND_LOG_ENV, TAG_PREFIX_ENV, ClaudeCode, CostLedger
from tests.test_llm import FakeClient, FakeRun, envelope, ok_message, tmpdir
from tests.test_llm_tags import make

CLEAN_ENV = {SPEND_LOG_ENV: "", AGENT_ENV: "", TAG_PREFIX_ENV: ""}


def tsv_rows(path) -> list[list[str]]:
    return [ln.split("\t") for ln in path.read_text(encoding="utf-8").splitlines()]


@mock.patch.dict(os.environ, CLEAN_ENV)
class TestLedgerPlusSpendLog(unittest.TestCase):
    def test_instance_without_tsv_takes_spend_log_agent_prefix(self):
        d = tmpdir(self)
        led = CostLedger(d / "l.jsonl")
        c = make(FakeClient(), spend_log=d / "s.log", ledger=led, agent="team-g1", tag_prefix="recete-G1/")
        self.assertIs(c.ledger, led)
        c.complete("s", "u", tag="L1")
        row = tsv_rows(d / "s.log")[0]
        self.assertEqual((row[1], row[6]), ("team-g1", "recete-G1/L1"))
        self.assertEqual(json.loads((d / "l.jsonl").read_text().splitlines()[0])["tag_prefix"], "recete-G1/")

    def test_ledger_path_with_spend_log(self):
        d = tmpdir(self)
        c = make(FakeClient(), ledger=d / "l.jsonl", spend_log=d / "s.log", agent="a", tag_prefix="p/")
        c.complete("s", "u", tag="t")
        self.assertEqual(tsv_rows(d / "s.log")[0][6], "p/t")
        self.assertEqual(len((d / "l.jsonl").read_text().splitlines()), 1)

    def test_same_values_are_not_a_conflict(self):
        d = tmpdir(self)
        led = CostLedger(None, tsv_path=d / "s.log", agent="a", tag_prefix="p/")
        make(FakeClient(), ledger=led, spend_log=d / "s.log", agent="a", tag_prefix="p/")

    def test_env_values_are_replaced(self):
        d = tmpdir(self)
        with mock.patch.dict(os.environ, {SPEND_LOG_ENV: str(d / "env.log"), AGENT_ENV: "envag",
                                          TAG_PREFIX_ENV: "env/"}):
            led = CostLedger(None)
        c = make(FakeClient(), ledger=led, spend_log=d / "s.log", agent="a", tag_prefix="p/")
        c.complete("s", "u", tag="t")
        self.assertFalse((d / "env.log").exists())
        self.assertEqual(tsv_rows(d / "s.log")[0][1::5], ["a", "p/t"])

    def test_real_conflict_names_the_fix_and_changes_nothing(self):
        d = tmpdir(self)
        led = CostLedger(None, tsv_path=d / "a.log", agent="a")
        with self.assertRaises(ValueError) as cm:
            make(FakeClient(), ledger=led, spend_log=d / "b.log", agent="b", tag_prefix="p/")
        msg = str(cm.exception)
        self.assertIn("spend_log=", msg)
        self.assertIn("agent=b", msg)
        self.assertIn("ledger=path, spend_log=", msg)
        self.assertEqual((led.tsv_path, led.agent, led.tag_prefix), (d / "a.log", "a", ""))

    def test_shared_ledger_second_backend_must_agree(self):
        d = tmpdir(self)
        led = CostLedger(None)
        make(FakeClient(), ledger=led, spend_log=d / "s.log", agent="a")
        make(FakeClient(), ledger=led)                       # nothing passed: keeps the attached values
        with self.assertRaises(ValueError):
            make(FakeClient(), ledger=led, agent="b")

    def test_tsv_disabled_explicitly_conflicts_with_spend_log(self):
        d = tmpdir(self)
        with self.assertRaises(ValueError):
            make(FakeClient(), ledger=CostLedger(None, tsv_path=False), spend_log=d / "s.log")

    def test_claude_code_attaches_too(self):
        d = tmpdir(self)
        cc = ClaudeCode(runner=FakeRun([envelope("ok")]), ledger=CostLedger(None), spend_log=d / "s.log",
                        agent="obs", sleep=lambda s: None)
        cc.complete("s", "u", tag="observer")
        self.assertEqual(tsv_rows(d / "s.log")[0][1], "obs")


@mock.patch.dict(os.environ, CLEAN_ENV)
class TestStatsAndSpent(unittest.TestCase):
    def test_stats_attribute_and_call(self):
        c = make(FakeClient(default=ok_message(inp=1_000_000, out=0)))
        c.complete("s", "u")
        self.assertEqual(c.stats["calls"], 1)
        d = c.stats()
        self.assertIs(type(d), dict)
        self.assertEqual(d, c.stats)
        d["calls"] = 99                                       # a copy: the backend's counters do not change
        self.assertEqual(c.stats()["calls"], 1)
        self.assertEqual(json.loads(json.dumps(c.stats))["calls"], 1)

    def test_spent_usd_is_instance_cost(self):
        led = CostLedger(None)
        a = make(FakeClient(default=ok_message(inp=1_000_000, out=0)), ledger=led)
        b = make(FakeClient(default=ok_message(inp=1_000_000, out=0)), ledger=led)
        self.assertEqual(a.spent_usd, 0.0)
        a.complete("s", "u1")
        b.complete("s", "u2")
        one = led.rows()[0]["cost_usd"]
        self.assertGreater(one, 0)
        self.assertAlmostEqual(a.spent_usd, one)             # this instance only
        self.assertEqual(a.spent_usd, a.stats["cost_usd"])
        self.assertAlmostEqual(a.stats["ledger_total_usd"], 2 * one)   # both backends

    def test_cache_hit_costs_nothing(self):
        d = tmpdir(self)
        c = make(FakeClient(default=ok_message(inp=1_000_000, out=0)), cache_dir=d / "c")
        c.complete("s", "u")
        c.complete("s", "u")
        self.assertEqual(c.stats()["cache_hits"], 1)
        self.assertAlmostEqual(c.spent_usd, c.ledger.rows()[0]["cost_usd"])
        self.assertEqual(len(c.ledger.rows()), 1)


if __name__ == "__main__":
    unittest.main()
