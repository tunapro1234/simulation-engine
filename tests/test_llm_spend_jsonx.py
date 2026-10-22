"""llm.spend_summary (shared TSV spend log per tag prefix, G2#7) and complete_json truncation retry (G9#6)."""
import tempfile
import threading
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

from simengine.llm import (Claude, CostLedger, EmptyResponseError, FakeLLM, JsonBudgetWarning, SpendSummary,
                           complete_json, is_truncated, spend_summary)
from simengine.llm.jsonx import json_unclosed
from simengine.types import LLMResponse


def tmpdir(tc: unittest.TestCase) -> Path:
    d = tempfile.TemporaryDirectory()
    tc.addCleanup(d.cleanup)
    return Path(d.name)


class SizedFake(FakeLLM):
    """FakeLLM that also records the max_tokens of each call and can attach a stop_reason."""

    def __init__(self, script):
        super().__init__(lambda s, u: "")
        self.script = list(script)
        self.budgets: list = []

    def complete(self, system, user, *, max_tokens=None, **kw):
        self.calls.append((system, user))
        self.budgets.append(max_tokens)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        text, stop = item if isinstance(item, tuple) else (item, None)
        return LLMResponse(text=text, model=self.model, raw={"stop_reason": stop} if stop else None)


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.lock = threading.Lock()
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **params):
        with self.lock:
            self.calls.append(params)
            return self.script.pop(0)


def msg(text, stop="end_turn", out=20):
    return {"id": "m", "model": "claude-haiku-5-5", "stop_reason": stop, "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 100, "output_tokens": out}}


class TestSpendSummary(unittest.TestCase):
    def test_prefix_and_groups(self):
        d = tmpdir(self)
        log = d / "spend.log"
        led = CostLedger(None, tsv_path=log, agent="team-a", tag_prefix="recete-G2/")
        led.record("anthropic/claude-haiku-5-5", 100, 10, 0.25, tag="L1:survey")
        led.record("anthropic/claude-haiku-5-5", 50, 5, 0.125, tag="L1:survey")
        led.record("anthropic/claude-sonnet-5-5", 10, 1, 1.0, tag="L2")
        CostLedger(None, tsv_path=log, agent="team-b", tag_prefix="other/").record("m", 1, 1, 9.0, tag="x")
        with log.open("a") as f:
            f.write("broken line\n\n")
            f.write("t\tteam-a\tm\tNaN-tokens\t1\t0.1\trecete-G2/bad\n")
        s = spend_summary(log, prefix="recete-G2/")
        self.assertIsInstance(s, SpendSummary)
        self.assertEqual(s.total, {"calls": 3, "input_tokens": 160, "output_tokens": 16, "usd": 1.375})
        self.assertEqual(s.usd, 1.375)
        self.assertEqual(list(s.groups), ["recete-G2/L2", "recete-G2/L1:survey"])    # most expensive first
        self.assertEqual(s.groups["recete-G2/L1:survey"]["calls"], 2)
        self.assertAlmostEqual(s.groups["recete-G2/L1:survey"]["usd"], 0.375)
        self.assertEqual(s.skipped, 2)
        self.assertIn("TOTAL\t3\t160\t16\t1.375000", str(s))
        self.assertEqual(spend_summary(log).total["calls"], 4)                    # no prefix: every valid line
        by_agent = spend_summary(log, by="agent")
        self.assertEqual(by_agent.groups["team-b"]["usd"], 9.0)
        self.assertEqual(spend_summary(log, prefix="recete-G2/", by=None).groups, {})
        root = spend_summary(log, prefix="recete-G2/", by=lambda r: r["tag"].split(":")[0])
        self.assertEqual(root.groups["recete-G2/L1"]["calls"], 2)
        self.assertEqual(root.to_dict()["total"]["calls"], 3)

    def test_env_missing_and_bad_by(self):
        d = tmpdir(self)
        import os
        old = os.environ.pop("SIMENGINE_SPEND_LOG", None)
        try:
            with self.assertRaises(ValueError):
                spend_summary()
            os.environ["SIMENGINE_SPEND_LOG"] = str(d / "none.log")
            s = spend_summary()
            self.assertEqual(s.total["calls"], 0)
            self.assertEqual(s.usd, 0.0)
        finally:
            os.environ.pop("SIMENGINE_SPEND_LOG", None)
            if old is not None:
                os.environ["SIMENGINE_SPEND_LOG"] = old
        with self.assertRaises(ValueError):
            spend_summary(d / "none.log", by="colour")

    def test_backend_rows_round_trip(self):
        d = tmpdir(self)
        log = d / "spend.log"
        c = Claude("claude-haiku-5-5", thinking="off", client=FakeClient([msg('{"a": 1}'), msg('{"b": 2}')]),
                   spend_log=log, agent="team-a", tag_prefix="recete-G2/", tag="L1")
        c.complete("s", "u1", max_tokens=600, tag="survey")
        c.complete("s", "u2", max_tokens=600, tag="survey")
        s = spend_summary(log, prefix="recete-G2/")
        self.assertEqual(list(s.groups), ["recete-G2/L1:survey"])
        self.assertEqual(s.total["calls"], 2)
        self.assertAlmostEqual(s.usd, c.spent_usd, places=6)


class TestTruncation(unittest.TestCase):
    def test_unclosed_detector(self):
        self.assertTrue(json_unclosed('Sure! {"a": 1, "b": "lon'))
        self.assertTrue(json_unclosed('[{"a": 1}, {"a": 2'))
        self.assertFalse(json_unclosed('{"a": "}{"} trailing ['))
        self.assertFalse(json_unclosed("no json here"))
        self.assertFalse(json_unclosed(""))
        self.assertTrue(is_truncated(LLMResponse(text='{"a": 1}', model="m", raw={"stop_reason": "max_tokens"})))
        self.assertFalse(is_truncated(LLMResponse(text='{"a": ', model="m", raw={"stop_reason": "end_turn"})))
        self.assertTrue(is_truncated(LLMResponse(text='{"a": ', model="m")))

    def test_fake_truncated_first_then_full(self):
        first = [True]

        def resp(system, user):
            if first[0]:
                first[0] = False
                return 'Let me explain the reasoning first. {"p": 0.4, "why": "because the'
            return '{"p": 0.4}'
        f = FakeLLM(resp)
        st = {}
        self.assertEqual(complete_json(f, "s", "q", stats=st), {"p": 0.4})
        self.assertEqual(len(f.calls), 2)
        self.assertEqual(f.calls[0], f.calls[1])            # same prompt, only a larger budget
        self.assertEqual(st, {"calls": 2, "truncated": 1, "truncation_retries": 1})

    def test_budget_doubles_and_caps(self):
        f = SizedFake([('{"a": 1, "b":', "max_tokens"), ('{"a": 1, "b": 2}', "end_turn")])
        with self.assertWarns(JsonBudgetWarning):
            self.assertEqual(complete_json(f, "s", "q", max_tokens=300), {"a": 1, "b": 2})
        self.assertEqual(f.budgets, [300, 600])
        f = SizedFake([('{"a":', "max_tokens"), ('{"a": 1}', None)])
        self.assertEqual(complete_json(f, "s", "q", max_tokens=3000, max_tokens_cap=4096), {"a": 1})
        self.assertEqual(f.budgets, [3000, 4096])

    def test_stop_reason_beats_salvage(self):
        # a cut array parses (salvaged objects), but stop_reason says it was cut: ask again first
        f = SizedFake([('[{"id": 1}, {"id": 2}, {"id"', "max_tokens"), ('[{"id": 1}, {"id": 2}, {"id": 3}]', None)])
        self.assertEqual(len(complete_json(f, "s", "q", max_tokens=800)), 3)

    def test_still_cut_returns_salvage_or_fails(self):
        f = SizedFake([('[{"id": 1}, {"id"', "max_tokens"), ('[{"id": 1}, {"id": 2}, {"id"', "max_tokens")])
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(complete_json(f, "s", "q", max_tokens=800), [{"id": 1}, {"id": 2}])
        # nothing salvageable: the cut reply counts as a failed attempt, then the 'JSON only' re-ask
        f = SizedFake([('{"a": "x', "max_tokens"), ('{"a": "xy', "max_tokens"), ('{"a": "z"}', None)])
        st = {}
        self.assertEqual(complete_json(f, "s", "q", max_tokens=800, stats=st), {"a": "z"})
        self.assertEqual(f.budgets, [800, 1600, 1600])
        self.assertIn("Return only valid JSON", f.calls[2][1])
        self.assertEqual(st["truncated"], 2)
        self.assertEqual(st["parse_failures"], 1)
        f = SizedFake([('{"a": "x', "max_tokens")] * 3)
        with self.assertRaises(ValueError):
            complete_json(f, "s", "q", max_tokens=800)

    def test_empty_response_retried_then_reraised(self):
        f = SizedFake([EmptyResponseError("thinking ate it"), '{"ok": true}'])
        self.assertEqual(complete_json(f, "s", "q", max_tokens=1000), {"ok": True})
        self.assertEqual(f.budgets, [1000, 2000])
        f = SizedFake([EmptyResponseError("a"), EmptyResponseError("b")])
        with self.assertRaises(EmptyResponseError):
            complete_json(f, "s", "q", max_tokens=1000)
        f = SizedFake([EmptyResponseError("a")])
        with self.assertRaises(EmptyResponseError):
            complete_json(f, "s", "q", max_tokens=1000, truncation_retries=0)

    def test_old_behaviour_kept(self):
        def resp(system, user):
            return '{"ok": true}' if "only valid JSON" in user else "I think the answer is yes."
        f = FakeLLM(resp)
        with warnings.catch_warnings():
            warnings.simplefilter("error")                  # FakeLLM without max_tokens: no budget warning
            self.assertEqual(complete_json(f, "s", "q"), {"ok": True})
        self.assertEqual(len(f.calls), 2)
        with self.assertRaises(ValueError):
            complete_json(FakeLLM(lambda s, u: "nope"), "s", "q")

    def test_claude_end_to_end(self):
        d = tmpdir(self)
        fc = FakeClient([msg('Reasoning: the persona is cautious, so {"p": 0.3, "note": "becau', stop="max_tokens",
                             out=300),
                         msg('{"p": 0.3}')])
        c = Claude("claude-haiku-5-5", thinking="off", client=fc, cache_dir=d, sleep=lambda s: None)
        with self.assertWarns(JsonBudgetWarning):
            self.assertEqual(c.complete_json("Return JSON only.", "q", max_tokens=300), {"p": 0.3})
        self.assertEqual([p["max_tokens"] for p in fc.calls], [300, 600])
        self.assertEqual(c.stats["json_truncated"], 1)
        self.assertEqual(c.stats["json_truncation_retries"], 1)
        # the cut reply was cached under max_tokens=300; a rerun hits the cache for both and still parses
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", JsonBudgetWarning)
            self.assertEqual(c.complete_json("Return JSON only.", "q", max_tokens=300), {"p": 0.3})
        self.assertEqual(len(fc.calls), 2)
        # the instance default counts too (Claude max_tokens=256)
        c2 = Claude("claude-haiku-5-5", thinking="off", client=FakeClient([msg('{"x": 1}')]))
        with self.assertWarns(JsonBudgetWarning):
            c2.complete_json("s", "q")
        c3 = Claude("claude-haiku-5-5", thinking="off", client=FakeClient([msg('{"x": 1}')]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            c3.complete_json("s", "q", max_tokens=600)


if __name__ == "__main__":
    unittest.main()
