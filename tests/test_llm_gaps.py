"""Offline tests for the D2-rehearsal llm gaps (SIMENGINE-EKSIK #2 spend TSV, #4 empty replies,
#6 soft refusals + cache_if, #9 tag through map). No network, no real API or CLI calls."""
from __future__ import annotations

import json
import os
import re
import time
import unittest
import warnings
from pathlib import Path
from unittest import mock

from simengine.llm import (AGENT_ENV, SPEND_LOG_ENV, TSV_COLUMNS, Claude, ClaudeCode, CostLedger, EmptyResponseError,
                           FakeLLM, LLMError, ThinkingBudgetWarning, looks_like_refusal)
from simengine.llm.ledger import tsv_line, tsv_model
from tests.test_llm import FakeClient, FakeRun, envelope, ok_message, tmpdir

CLEAN_ENV = {SPEND_LOG_ENV: "", AGENT_ENV: ""}


def rehearsal_line(model, inp, out, usd, tag, agent, prefix=""):
    """The D2 rehearsal's HarcamaLedger layout, rebuilt independently (time column excluded)."""
    return "\t".join([agent, model, str(int(inp)), str(int(out)), f"{float(usd):.6f}", f"{prefix}{tag}"])


def tsv_rows(path: Path) -> list[list[str]]:
    return [ln.split("\t") for ln in path.read_text(encoding="utf-8").splitlines()]


def make(client, **kw):
    kw.setdefault("sleep", lambda s: None)
    return Claude("claude-haiku-5-5", effort="low", client=client, **kw)


# ---------------------------------------------------------------- #2 TSV spend log
@mock.patch.dict(os.environ, CLEAN_ENV)
class TestSpendTSV(unittest.TestCase):
    def test_columns_constant(self):
        self.assertEqual(TSV_COLUMNS, ("time", "agent", "model", "input_tokens", "output_tokens", "usd", "tag"))

    def test_line_matches_rehearsal_format(self):
        d = tmpdir(self)
        led = CostLedger(d / "ledger.jsonl", tsv_path=d / "spend.log", agent="team-a", tag_prefix="d2/")
        led.record("anthropic/claude-haiku-5-5", 1234, 56, 0.0001234567, "country")
        rows = tsv_rows(d / "spend.log")
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), len(TSV_COLUMNS))
        self.assertRegex(rows[0][0], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d{4}$")
        self.assertEqual("\t".join(rows[0][1:]),
                         rehearsal_line("anthropic/claude-haiku-5-5", 1234, 56, 0.0001234567, "country", "team-a", "d2/"))
        self.assertEqual(rows[0][5], "0.000123")
        # JSONL ledger is unchanged and still the budget source
        self.assertEqual(len(led.rows()), 1)
        self.assertAlmostEqual(led.total(), 0.0001234567, places=8)

    def test_model_prefix_and_field_sanitizing(self):
        self.assertEqual(tsv_model("claude-haiku-5-5"), "anthropic/claude-haiku-5-5")
        self.assertEqual(tsv_model("anthropic/claude-haiku-5-5"), "anthropic/claude-haiku-5-5")
        self.assertEqual(tsv_model("claude-p/claude-haiku-5-5"), "claude-p/claude-haiku-5-5")
        line = tsv_line({"time": "t", "model": "claude-x", "input_tokens": 1, "output_tokens": 2, "cost_usd": 0.5,
                         "tag": "a\tb\nc"}, "ag")
        self.assertEqual(line, "t\tag\tanthropic/claude-x\t1\t2\t0.500000\ta b c\n")

    def test_env_vars_and_disable(self):
        d = tmpdir(self)
        with mock.patch.dict(os.environ, {SPEND_LOG_ENV: str(d / "env.log"), AGENT_ENV: "env-agent"}):
            CostLedger(None).record("anthropic/claude-haiku-5-5", 1, 2, 0.001, "t")
            CostLedger(None, tsv_path=False).record("anthropic/claude-haiku-5-5", 1, 2, 0.001, "t")
            CostLedger(None, agent="explicit").record("anthropic/claude-haiku-5-5", 1, 2, 0.001, "t")
        rows = tsv_rows(d / "env.log")
        self.assertEqual([r[1] for r in rows], ["env-agent", "explicit"])
        # no env, no tsv_path -> no TSV at all; default agent name
        led = CostLedger(None)
        self.assertIsNone(led.tsv_path)
        self.assertEqual(led.agent, "simengine")

    def test_unwritable_tsv_warns_but_call_survives(self):
        d = tmpdir(self)
        (d / "adir").mkdir()
        led = CostLedger(d / "ledger.jsonl", tsv_path=d / "adir")       # a directory: open() fails
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            led.record("anthropic/claude-haiku-5-5", 1, 1, 0.01, "t")
            led.record("anthropic/claude-haiku-5-5", 1, 1, 0.01, "t")
        self.assertEqual(len([x for x in w if issubclass(x.category, RuntimeWarning)]), 1)
        self.assertEqual(len(led.rows()), 2)

    def test_claude_real_calls_only_and_tags(self):
        d = tmpdir(self)
        fc = FakeClient(default=ok_message("x", inp=1000, out=500))
        c = make(fc, cache_dir=d / "c", spend_log=d / "spend.log", agent="a1", max_tokens=2000)
        r = c.complete("s", "u", tag="persona")
        c.complete("s", "u", tag="persona")                       # cache hit: no TSV line
        c.map([("s", "v1"), ("s", "v2")], tag="no_persona", parallel=2)
        rows = tsv_rows(d / "spend.log")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][1:5], ["a1", "anthropic/claude-haiku-5-5", "1000", "500"])
        self.assertEqual(rows[0][5], f"{r.cost_usd:.6f}")
        self.assertEqual([x[6] for x in rows], ["persona", "no_persona", "no_persona"])
        self.assertEqual(sorted(x["tag"] for x in c.ledger.rows()), ["no_persona", "no_persona", "persona"])
        # constructor tag is the default
        c2 = make(FakeClient(), spend_log=d / "spend2.log", tag="arm-a", max_tokens=2000)
        c2.complete("s", "u")
        self.assertEqual(tsv_rows(d / "spend2.log")[0][6], "arm-a")

    def test_env_reaches_default_backend_ledger(self):
        d = tmpdir(self)
        with mock.patch.dict(os.environ, {SPEND_LOG_ENV: str(d / "s.log"), AGENT_ENV: "ag"}):
            c = make(FakeClient(), cache_dir=d / "c", max_tokens=2000)
            c.complete("s", "u", tag="t")
        self.assertEqual(tsv_rows(d / "s.log")[0][1], "ag")

    def test_ledger_instance_plus_spend_log_is_an_error(self):
        d = tmpdir(self)
        with self.assertRaises(ValueError):
            make(FakeClient(), ledger=CostLedger(None), spend_log=d / "x.log")

    def test_claude_code_rows(self):
        d = tmpdir(self)
        cc = ClaudeCode(runner=FakeRun([envelope("ok")]), spend_log=d / "s.log", agent="obs", sleep=lambda s: None)
        cc.complete("s", "u", tag="observer")
        row = tsv_rows(d / "s.log")[0]
        self.assertEqual(row[1:], ["obs", "claude-p/claude-haiku-5-5", "10", "5", "0.000000", "observer"])


# ---------------------------------------------------------------- #4 empty replies, thinking budget
@mock.patch.dict(os.environ, CLEAN_ENV)
class TestEmptyReply(unittest.TestCase):
    def empty(self):
        return ok_message("", inp=50, out=300, stop="max_tokens")

    def test_empty_max_tokens_raises_and_is_not_cached(self):
        d = tmpdir(self)
        fc = FakeClient(default=self.empty())
        c = make(fc, cache_dir=d, thinking="off", max_tokens=300)
        with self.assertRaises(EmptyResponseError) as cm:
            c.complete("s", "u")
        self.assertIsInstance(cm.exception, LLMError)
        msg = str(cm.exception)
        self.assertIn("thinking='off'", msg)
        self.assertIn("max_tokens", msg)
        with self.assertRaises(EmptyResponseError):
            c.complete("s", "u")                                   # not served from cache
        self.assertEqual(len(fc.calls), 2)                         # not retried within a call either
        self.assertEqual(len(c.ledger.rows()), 2)                  # but billed
        self.assertEqual(c.stats["errors"], 2)
        self.assertFalse(any(d.glob("*/*.json")))

    def test_empty_end_turn_unchanged(self):
        c = make(FakeClient(default=ok_message("", stop="end_turn")), thinking="off")
        self.assertEqual(c.complete("s", "u").text, "")

    def test_map_handles_empty_like_other_llm_errors(self):
        def resp(p):
            return self.empty() if p["messages"][0]["content"] == "bad" else ok_message("fine")
        c = make(FakeClient(default=resp), thinking="off")
        out = c.map([("s", "a"), ("s", "bad"), ("s", "b")], parallel=2, return_exceptions=True)
        self.assertEqual(out[0].text, "fine")
        self.assertIsInstance(out[1], EmptyResponseError)
        self.assertEqual(out[2].text, "fine")
        with self.assertRaises(EmptyResponseError):
            c.map([("s", "bad")], parallel=1)

    def test_thinking_budget_warning_once(self):
        fc = FakeClient()
        c = make(fc, max_tokens=300)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c.complete("s", "u1")
            c.complete("s", "u2")
        hits = [x for x in w if issubclass(x.category, ThinkingBudgetWarning)]
        self.assertEqual(len(hits), 1)
        self.assertIn("thinking tokens count toward max_tokens", str(hits[0].message))
        self.assertNotIn("thinking", fc.calls[0])                  # default thinking unchanged (omitted)

    def test_no_warning_when_off_or_large_or_cached(self):
        d = tmpdir(self)
        cases = [dict(thinking="off", max_tokens=300), dict(thinking={"type": "disabled"}, max_tokens=300),
                 dict(max_tokens=1024), dict(max_tokens=2000)]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for kw in cases:
                make(FakeClient(), **kw).complete("s", "u")
            make(FakeClient(), max_tokens=300).complete("s", "u", thinking="off")      # per-call override
            make(FakeClient(), max_tokens=300).complete("s", "u", max_tokens=4000)     # per-call max_tokens
            make(FakeClient(), cache_dir=d, thinking="off", max_tokens=300).complete("s", "u")
            make(FakeClient(), cache_dir=d, thinking="off", max_tokens=300).complete("s", "u")  # cache hit
        self.assertEqual([x for x in w if issubclass(x.category, ThinkingBudgetWarning)], [])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make(FakeClient(), max_tokens=2000).complete("s", "u", max_tokens=200)
        self.assertEqual(len([x for x in w if issubclass(x.category, ThinkingBudgetWarning)]), 1)


# ---------------------------------------------------------------- #6 soft refusals, cache_if
class TestLooksLikeRefusal(unittest.TestCase):
    POSITIVE = [
        "I can't produce this as a survey dataset. Inventing answers for people who don't exist would be misleading.",
        "I cannot create responses for these 20 people.",
        "I'm sorry, but I can't help with that.",
        "I'm not able to role-play as these respondents.",
        "I won't fabricate survey responses, but here is how you could collect them.",
        "I am unable to generate fabricated personal opinions.",
        "I really can't make up answers on behalf of real demographic groups.",
        "I must respectfully decline this request.",
        "I don't feel comfortable pretending to be a specific person.",
        "**I can't simulate individual respondents.** Instead, here is aggregate information...",
        "As an AI, I don't have personal political opinions.",
        "Thanks for the detailed setup. I can't invent answers for people who don't exist.",
        "Maalesef bunu yapamam.",
        "Bu yanıtları üretemem; gerçek kişiler adına konuşamam.",
        "Var olmayan kişiler adına cevap uyduramam.",
        "Üzgünüm, bu konuda yardımcı olamam.",
    ]
    NEGATIVE = [
        "",
        None,
        "4",
        "Answer: 3. I can't afford a new car right now, so price matters most.",
        'I would choose option B, "I can\'t produce enough to live on", because my farm is small.',
        "I'd pick 'I cannot create a budget' since I never plan my spending.",
        '{"answer": 2, "reason": "I can\'t produce this as a survey dataset"}',
        '[{"id": 1, "answer": "I cannot create responses"}]',
        "```json\n{\"answer\": \"I won't fabricate\"}\n```",
        "I can't make up my mind, but I lean towards 4.",
        "I decline the offer; 2 out of 10 is unfair.",
        "I can't participate in elections because I am not a citizen. Answer: 1.",
        "I can't provide for my family on this salary, so I strongly agree (5).",
        "As an AI researcher, I would rate this 4.",
        ("My answer is 4. " + "I think the policy helps families with children and small business owners. " * 5
         + "I can't produce more detail than that."),
        "Bu fiyata alışveriş yapamam, o yüzden 2 derim.",
        "Bunu yapamam, çok pahalı. Cevabım: 1",
        "Sahte haberlere güvenmem, paylaşım yapamam.",
    ]

    def test_positives(self):
        for t in self.POSITIVE:
            with self.subTest(t=t):
                self.assertTrue(looks_like_refusal(t))

    def test_negatives(self):
        for t in self.NEGATIVE:
            with self.subTest(t=t):
                self.assertFalse(looks_like_refusal(t))

    def test_fast(self):
        long = "word " * 20000
        t0 = time.perf_counter()
        for _ in range(200):
            looks_like_refusal(long)
        self.assertLess(time.perf_counter() - t0, 2.0)


@mock.patch.dict(os.environ, CLEAN_ENV)
class TestCacheIf(unittest.TestCase):
    def test_refusal_returned_but_not_cached(self):
        d = tmpdir(self)
        refusal = "I can't produce this as a survey dataset."
        fc = FakeClient([ok_message(refusal), ok_message("3"), ok_message("unused")])
        c = make(fc, cache_dir=d, thinking="off")
        keep = lambda t: not looks_like_refusal(t)  # noqa: E731
        r1 = c.complete("s", "u", cache_if=keep)
        self.assertEqual(r1.text, refusal)
        self.assertTrue(r1.raw["not_cached"])
        self.assertEqual(len(fc.calls), 1)                         # no retry on cache_if=False
        r2 = c.complete("s", "u", cache_if=keep)                   # asked again, now answered
        self.assertEqual(r2.text, "3")
        self.assertNotIn("not_cached", r2.raw)
        r3 = c.complete("s", "u", cache_if=keep)
        self.assertTrue(r3.cached)
        self.assertEqual(r3.text, "3")
        self.assertEqual(len(fc.calls), 2)
        self.assertEqual(len(c.ledger.rows()), 2)                  # refusal still billed

    def test_cache_if_through_map_and_errors_mean_no_cache(self):
        d = tmpdir(self)
        fc = FakeClient(default=lambda p: ok_message(p["messages"][0]["content"]))
        c = make(fc, cache_dir=d, thinking="off")
        c.map([("s", "keep"), ("s", "drop")], cache_if=lambda t: t == "keep", parallel=2)
        c.map([("s", "keep"), ("s", "drop")], cache_if=lambda t: t == "keep", parallel=2)
        self.assertEqual(len(fc.calls), 3)
        def boom(t):
            raise RuntimeError("x")
        r = c.complete("s", "other", cache_if=boom)
        self.assertTrue(r.raw["not_cached"])

    def test_validate_failures_never_cache(self):
        d = tmpdir(self)
        fc = FakeClient(default=ok_message("not a number"))
        c = make(fc, cache_dir=d, thinking="off", max_attempts=2)
        with self.assertRaises(LLMError):
            c.complete("s", "u", validate=int)
        self.assertEqual(len(fc.calls), 2)
        self.assertFalse(any(d.glob("*/*.json")))

    def test_claude_code_cache_if(self):
        d = tmpdir(self)
        run = FakeRun([envelope("I cannot create responses for these people."), envelope("2")])
        cc = ClaudeCode(runner=run, cache_dir=d, sleep=lambda s: None)
        keep = lambda t: not looks_like_refusal(t)  # noqa: E731
        self.assertTrue(cc.complete("s", "u", cache_if=keep).raw["not_cached"])
        self.assertEqual(cc.complete("s", "u", cache_if=keep).text, "2")
        self.assertTrue(cc.complete("s", "u", cache_if=keep).cached)

    def test_fake_accepts_new_keywords(self):
        f = FakeLLM()
        self.assertEqual(len(f.map([("s", "1")], tag="arm", cache_if=lambda t: True)), 1)
        f.complete("s", "u", tag="arm")


if __name__ == "__main__":
    unittest.main()
