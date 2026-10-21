"""Offline tests for simengine.llm (no network, no real API or CLI calls)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from simengine.llm import (PRICES, AuthError, BudgetExceeded, Claude, ClaudeCode, CostLedger, DiskCache, FakeLLM,
                           LLMError, RefusalError, SpendLimitError, cache_key, classify_error, complete_json, cost_usd,
                           json_extract, load_api_key, parallel_map, price_for, usage_cost)
from simengine.types import LLM


# ---------------------------------------------------------------- fakes
class FakeAPIError(Exception):
    def __init__(self, status, message, headers=None):
        super().__init__(message)
        self.status_code = status
        self.message = message
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


def ok_message(text="hello", inp=100, out=20, cread=0, cwrite=0, stop="end_turn", model="claude-haiku-5-5"):
    return {"id": "msg_1", "model": model, "stop_reason": stop,
            "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": text}],
            "usage": {"input_tokens": inp, "output_tokens": out, "cache_read_input_tokens": cread,
                      "cache_creation_input_tokens": cwrite}}


class FakeClient:
    """Mimics anthropic.Anthropic().messages.create; `script` items are dicts or exceptions."""

    def __init__(self, script=None, default=None):
        self.script = list(script or [])
        self.default = default or ok_message()
        self.calls = []
        self.lock = threading.Lock()
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **params):
        with self.lock:
            self.calls.append(params)
            item = self.script.pop(0) if self.script else self.default
        if callable(item) and not isinstance(item, dict):
            item = item(params)
        if isinstance(item, BaseException):
            raise item
        return item


def tmpdir(tc: unittest.TestCase) -> Path:
    d = tempfile.TemporaryDirectory()
    tc.addCleanup(d.cleanup)
    return Path(d.name)


# ---------------------------------------------------------------- pricing / ledger
class TestPricing(unittest.TestCase):
    def test_table(self):
        self.assertEqual(price_for("claude-haiku-5-5").input_per_mtok, 0.10)
        self.assertEqual(price_for("claude-sonnet-5-5").output_per_mtok, 10.0)
        self.assertEqual(price_for("claude-opus-5-5").input_per_mtok, 4.0)
        self.assertEqual(price_for("claude-fable-5-1").output_per_mtok, 50.0)
        self.assertEqual(price_for("anthropic/claude-opus-5-5"), PRICES["claude-opus-5-5"])
        with self.assertRaises(KeyError):
            price_for("gpt-x")

    def test_cost_math(self):
        self.assertAlmostEqual(cost_usd("claude-haiku-5-5", 100_000, 1_000_000), 0.01 + 0.50)
        self.assertAlmostEqual(cost_usd("claude-sonnet-5-5", 1000, 500), 0.002 + 0.005)
        self.assertAlmostEqual(cost_usd("claude-opus-5-5", 0, 0, cache_read_tokens=1_000_000), 0.4)   # 0.1x of $4
        self.assertAlmostEqual(cost_usd("claude-opus-5-5", 0, 0, cache_write_tokens=1_000_000), 5.0)  # 1.25x
        self.assertAlmostEqual(cost_usd("claude-fable-5-1", 2000, 1000), 0.02 + 0.05)
        # Haiku long-context tier above 100K prompt tokens
        self.assertAlmostEqual(cost_usd("claude-haiku-5-5", 200_000, 1000), 0.10 + 0.0025)
        u = {"input_tokens": 577, "output_tokens": 322}
        self.assertAlmostEqual(usage_cost("claude-haiku-5-5", u), (577 * 0.1 + 322 * 0.5) / 1e6)

    def test_ledger_file_and_incremental_total(self):
        p = tmpdir(self) / "led" / "ledger.jsonl"
        a, b = CostLedger(p), CostLedger(p)       # two writers on one file (like two processes)
        a.record("anthropic/claude-haiku-5-5", 100, 20, 0.25, "t1")
        self.assertAlmostEqual(b.total(), 0.25)
        b.record("anthropic/claude-sonnet-5-5", 100, 20, 1.5, "t2")
        self.assertAlmostEqual(a.total(), 1.75)
        self.assertAlmostEqual(a.total(lambda m: "sonnet" in m), 1.5)
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        self.assertEqual(rows[0]["tag"], "t1")
        self.assertEqual(set(a.summary()), {"anthropic/claude-haiku-5-5", "anthropic/claude-sonnet-5-5"})
        p.write_text("")                           # truncation resets the running sum
        self.assertEqual(a.total(), 0.0)

    def test_ledger_memory(self):
        l = CostLedger(None)
        l.record("m", cost_usd=0.1)
        l.record("m", cost_usd=0.2)
        self.assertAlmostEqual(l.total(), 0.3)
        self.assertEqual(len(l.rows()), 2)


# ---------------------------------------------------------------- cache / json / parallel
class TestCacheJsonParallel(unittest.TestCase):
    def test_cache_key(self):
        k = cache_key("m", "s", "u", {"max_tokens": 10, "sample": 0})
        self.assertEqual(k, cache_key("m", "s", "u", {"sample": 0, "max_tokens": 10}))
        self.assertNotEqual(k, cache_key("m", "s", "u", {"max_tokens": 10, "sample": 1}))
        self.assertNotEqual(k, cache_key("m2", "s", "u", {"max_tokens": 10, "sample": 0}))
        self.assertEqual(len(k), 64)

    def test_disk_cache(self):
        c = DiskCache(tmpdir(self))
        self.assertIsNone(c.get("k", "m"))
        c.put("k", {"text": "x"}, "anthropic/m")
        self.assertEqual(c.get("k", "anthropic/m")["text"], "x")
        c.path("bad", "m").parent.mkdir(parents=True, exist_ok=True)
        c.path("bad", "m").write_text("{trunc")
        self.assertIsNone(c.get("bad", "m"))

    def test_json_extract(self):
        self.assertEqual(json_extract('{"a": 1}'), {"a": 1})
        self.assertEqual(json_extract('Sure!\n```json\n{"a": [1, 2],}\n```\nbye'), {"a": [1, 2]})
        self.assertEqual(json_extract("answer: [1, 2, 3] ok"), [1, 2, 3])
        self.assertEqual(json_extract('{not json} but {"x": 2}'), {"x": 2})
        self.assertEqual(json_extract('[{"id": "p1", "v": 1}, {"id": "p2", "v": {"k": 2}}, {"id": "p3", "v'),
                         [{"id": "p1", "v": 1}, {"id": "p2", "v": {"k": 2}}])
        self.assertEqual(json_extract('{"a": {"b": [1, 2,]},}'), {"a": {"b": [1, 2]}})
        self.assertIsNone(json_extract("no json here", default=None))
        with self.assertRaises(ValueError):
            json_extract("")
        with self.assertRaises(ValueError):
            json_extract(None)

    def test_complete_json_reasks(self):
        def resp(system, user):
            return '{"ok": true}' if "only valid JSON" in user else "I think the answer is yes."
        f = FakeLLM(resp)
        self.assertEqual(complete_json(f, "s", "q"), {"ok": True})
        self.assertEqual(len(f.calls), 2)
        with self.assertRaises(ValueError):
            complete_json(FakeLLM(lambda s, u: "nope"), "s", "q")

    def test_parallel_map_order_and_stop(self):
        self.assertEqual(parallel_map(lambda x: x * x, range(20), parallel=4), [x * x for x in range(20)])
        self.assertEqual(parallel_map(lambda x: x, [], parallel=4), [])

        def f(x):
            if x == 3:
                raise ValueError("boom")
            return x
        out = parallel_map(f, range(6), parallel=3, return_exceptions=True)
        self.assertIsInstance(out[3], ValueError)
        self.assertEqual(out[5], 5)
        with self.assertRaises(ValueError):
            parallel_map(f, range(6), parallel=3)

        def g(x):
            if x == 0:
                raise BudgetExceeded("stop")
            return x
        with self.assertRaises(BudgetExceeded):
            parallel_map(g, range(50), parallel=2, return_exceptions=True)


# ---------------------------------------------------------------- Claude (mocked client)
class TestClaude(unittest.TestCase):
    def make(self, client, **kw):
        kw.setdefault("cache_dir", tmpdir(self))
        return Claude("claude-haiku-5-5", effort="low", client=client, sleep=lambda s: None, **kw)

    def test_protocol(self):
        c = self.make(FakeClient())
        self.assertIsInstance(c, LLM)
        self.assertIsInstance(FakeLLM(), LLM)
        self.assertIsInstance(ClaudeCode(), LLM)

    def test_request_building(self):
        fc = FakeClient()
        c = self.make(fc, max_tokens=300)
        r = c.complete("SYS", "USER", stop_sequences=["\n\n"])
        p = fc.calls[0]
        self.assertEqual(p["model"], "claude-haiku-5-5")
        self.assertEqual(p["max_tokens"], 300)
        self.assertEqual(p["output_config"], {"effort": "low"})
        self.assertEqual(p["messages"], [{"role": "user", "content": "USER"}])
        self.assertEqual(p["system"], [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}])
        self.assertEqual(p["stop_sequences"], ["\n\n"])
        self.assertNotIn("temperature", p)
        self.assertNotIn("thinking", p)
        self.assertEqual(r.text, "hello")          # thinking block skipped, text joined
        c.complete("SYS", "U2", max_tokens=50, effort="medium", system_cache=False,
                   output_config={"format": {"type": "json_schema", "schema": {"type": "object"}}})
        p = fc.calls[1]
        self.assertEqual(p["max_tokens"], 50)
        self.assertEqual(p["output_config"]["effort"], "medium")
        self.assertIn("format", p["output_config"])
        self.assertEqual(p["system"], "SYS")
        c.complete("", "no system")
        self.assertNotIn("system", fc.calls[2])

    def test_thinking_modes(self):
        fc = FakeClient()
        c = self.make(fc, thinking="off")
        c.complete("s", "u")
        self.assertEqual(fc.calls[0]["thinking"], {"type": "disabled"})
        s = Claude("claude-sonnet-5-5", thinking="off", client=fc, cache_dir=None)
        s.complete("s", "u")
        self.assertEqual(fc.calls[1]["thinking"], {"type": "between_tools"})
        o = Claude("claude-opus-5-5", thinking="off", client=fc)
        with self.assertRaises(ValueError):
            o.complete("s", "u")
        with self.assertRaises(KeyError):
            Claude("unpriced-model", client=fc)

    def test_cache_hit(self):
        fc = FakeClient(default=ok_message("cached answer", 1000, 100))
        d = tmpdir(self)
        c = self.make(fc, cache_dir=d)
        r1 = c.complete("s", "u")
        r2 = c.complete("s", "u")
        self.assertEqual(len(fc.calls), 1)
        self.assertFalse(r1.cached)
        self.assertTrue(r2.cached)
        self.assertEqual(r2.text, "cached answer")
        self.assertEqual(r2.cost_usd, 0.0)
        self.assertAlmostEqual(r2.raw["original_cost_usd"], r1.cost_usd)
        # a fresh instance on the same directory also hits (persistent across processes)
        fc2 = FakeClient()
        c2 = self.make(fc2, cache_dir=d)
        self.assertTrue(c2.complete("s", "u").cached)
        self.assertEqual(fc2.calls, [])
        # different sample / max_tokens / effort / model -> new call
        c.complete("s", "u", sample=1)
        c.complete("s", "u", max_tokens=999)
        c.complete("s", "u", effort="high")
        self.assertEqual(len(fc.calls), 4)
        # cache_control does not enter the key
        c.complete("s", "u", system_cache=False)
        self.assertEqual(len(fc.calls), 4)
        self.assertEqual(c.stats["cache_hits"], 2)

    def test_ledger_cost(self):
        fc = FakeClient(default=ok_message("x", inp=1000, out=500, cread=2000, cwrite=400))
        d = tmpdir(self)
        c = self.make(fc, cache_dir=d)
        r = c.complete("s", "u", tag="exp1")
        want = ((1000 + 1.25 * 400 + 0.1 * 2000) * 0.10 + 500 * 0.50) / 1e6
        self.assertAlmostEqual(r.cost_usd, want)
        self.assertEqual(r.input_tokens, 1400)
        self.assertEqual(r.cache_read_tokens, 2000)
        rows = [json.loads(l) for l in (d / "ledger.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "anthropic/claude-haiku-5-5")
        self.assertEqual(rows[0]["tag"], "exp1")
        self.assertAlmostEqual(rows[0]["cost_usd"], round(want, 8))
        c.complete("s", "u")                        # cache hit: no ledger line
        self.assertEqual(len((d / "ledger.jsonl").read_text().splitlines()), 1)

    def test_budget_guard(self):
        d = tmpdir(self)
        CostLedger(d / "ledger.jsonl").record("anthropic/claude-haiku-5-5", cost_usd=5.0)
        fc = FakeClient()
        c = self.make(fc, cache_dir=d, budget_usd=5.0)
        with self.assertRaises(BudgetExceeded):
            c.complete("s", "u")
        self.assertEqual(fc.calls, [])              # raised BEFORE the call
        # budget reached by our own spending: first call ok, second refused
        fc = FakeClient(default=ok_message(inp=100_000, out=0))   # $0.01 per call
        c = self.make(fc, budget_usd=0.01)
        c.complete("s", "a")
        with self.assertRaises(BudgetExceeded):
            c.complete("s", "b")
        self.assertEqual(len(fc.calls), 1)
        # cached answers are still served after the budget is gone
        self.assertTrue(c.complete("s", "a").cached)

    def test_map_stops_on_budget(self):
        fc = FakeClient(default=ok_message(inp=100_000, out=0))    # $0.01 per call
        c = self.make(fc, budget_usd=0.025)
        with self.assertRaises(BudgetExceeded):
            c.map([("s", f"u{i}") for i in range(30)], parallel=1)
        self.assertEqual(len(fc.calls), 3)

    def test_map_order_parallel(self):
        fc = FakeClient(default=lambda p: ok_message(text=p["messages"][0]["content"].upper()))
        c = self.make(fc)
        rs = c.map([("s", f"u{i}") for i in range(12)], parallel=4)
        self.assertEqual([r.text for r in rs], [f"U{i}" for i in range(12)])

    def test_401_stops_immediately(self):
        fc = FakeClient([FakeAPIError(401, "invalid x-api-key")])
        c = self.make(fc)
        with self.assertRaises(AuthError):
            c.complete("s", "u")
        self.assertEqual(len(fc.calls), 1)

    def test_spend_limit_stops_immediately(self):
        for st, msg in [(429, "You have reached your specified API usage limits."),
                        (400, "Your credit balance is too low to access the Anthropic API."),
                        (403, "spend limit exceeded")]:
            fc = FakeClient([FakeAPIError(st, msg)])
            c = self.make(fc)
            with self.assertRaises(SpendLimitError):
                c.complete("s", "u")
            self.assertEqual(len(fc.calls), 1, msg)

    def test_rate_limit_backoff(self):
        waits = []
        fc = FakeClient([FakeAPIError(429, "rate_limit_error: too many requests"),
                         FakeAPIError(529, "overloaded_error: Overloaded"),
                         FakeAPIError(429, "rate limited", headers={"retry-after": "7"}),
                         ok_message("finally")])
        c = Claude("claude-haiku-5-5", client=fc, sleep=waits.append, rate_wait=15.0)
        r = c.complete("s", "u")
        self.assertEqual(r.text, "finally")
        self.assertEqual(waits, [15.0, 30.0, 7.0])
        self.assertEqual(c.stats["retries"], 3)
        # exhausted attempts -> LLMError
        fc = FakeClient([FakeAPIError(529, "Overloaded")] * 10)
        c = Claude("claude-haiku-5-5", client=fc, sleep=lambda s: None, max_attempts=3)
        with self.assertRaises(LLMError):
            c.complete("s", "u")
        self.assertEqual(len(fc.calls), 3)

    def test_fatal_400_not_retried(self):
        fc = FakeClient([FakeAPIError(400, "temperature: unsupported value")])
        c = self.make(fc)
        with self.assertRaises(LLMError) as cm:
            c.complete("s", "u")
        self.assertNotIsInstance(cm.exception, SpendLimitError)
        self.assertEqual(len(fc.calls), 1)

    def test_classify(self):
        self.assertEqual(classify_error(FakeAPIError(401, "x")), "auth")
        self.assertEqual(classify_error(FakeAPIError(429, "rate limit")), "retry")
        self.assertEqual(classify_error(FakeAPIError(429, "usage limit reached")), "limit")
        self.assertEqual(classify_error(FakeAPIError(500, "x")), "retry")
        self.assertEqual(classify_error(FakeAPIError(404, "model not found")), "fatal")

        class APIConnectionError(Exception):
            pass
        self.assertEqual(classify_error(APIConnectionError("down")), "retry")

    def test_refusal_not_cached(self):
        fc = FakeClient([ok_message("", stop="refusal"), ok_message("fine")])
        c = self.make(fc)
        with self.assertRaises(RefusalError):
            c.complete("s", "u")
        self.assertEqual(c.complete("s", "u").text, "fine")
        self.assertEqual(len(fc.calls), 2)

    def test_validate_retries(self):
        fc = FakeClient([ok_message("not json"), ok_message('{"p": 0.3}')])
        c = self.make(fc)
        r = c.complete("s", "u", validate=json_extract)
        self.assertEqual(r.text, '{"p": 0.3}')
        self.assertEqual(len(fc.calls), 2)
        self.assertEqual(len(c.ledger.rows()), 2)   # both calls cost money and are logged

    def test_api_key_sources_and_secrecy(self):
        secret = "sk-test-NOT-A-REAL-KEY-123"
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": secret}, clear=False):
            self.assertEqual(load_api_key(), secret)
        kf = tmpdir(self) / "k.txt"
        kf.write_text(secret + "\n")
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "SIMENGINE_KEY_FILE")}
        with mock.patch.dict(os.environ, {**env, "SIMENGINE_KEY_FILE": str(kf)}, clear=True):
            self.assertEqual(load_api_key(), secret)
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(AuthError) as cm:
                load_api_key()
            self.assertNotIn(secret, str(cm.exception))
            # no key: cached answers still work, a new call raises AuthError
            d = tmpdir(self)
            self.make(FakeClient(default=ok_message("x")), cache_dir=d).complete("s", "u")
            c = Claude("claude-haiku-5-5", cache_dir=d)
            self.assertTrue(c.complete("s", "u").cached)
            with self.assertRaises(AuthError):
                c.complete("s", "other")
        c = Claude("claude-haiku-5-5", api_key=secret, client=FakeClient(default=ok_message()))
        self.assertNotIn(secret, repr(c))
        d = tmpdir(self)
        c = self.make(FakeClient(default=ok_message()), cache_dir=d, api_key=secret)
        c.complete("s", "u")
        for f in d.rglob("*"):
            if f.is_file():
                self.assertNotIn(secret, f.read_text())


# ---------------------------------------------------------------- ClaudeCode (mocked subprocess)
class FakeRun:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        o = self.outputs.pop(0)
        if isinstance(o, BaseException):
            raise o
        return o


def envelope(result="ok", is_error=False, cost=0.0123):
    return SimpleNamespace(returncode=1 if is_error else 0, stderr="",
                           stdout=json.dumps({"type": "result", "is_error": is_error, "result": result,
                                              "total_cost_usd": cost, "session_id": "abc",
                                              "usage": {"input_tokens": 10, "output_tokens": 5,
                                                        "cache_read_input_tokens": 2400,
                                                        "cache_creation_input_tokens": 0}}))


class TestClaudeCode(unittest.TestCase):
    def test_command_construction(self):
        cc = ClaudeCode("claude-haiku-5-5", claude_bin="/opt/claude")
        self.assertEqual(cc.build_command("SYS"),
                         ["/opt/claude", "-p", "--model", "claude-haiku-5-5", "--output-format", "json", "--tools", "",
                          "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "",
                          "--settings", '{"language": "English"}', "--system-prompt", "SYS"])
        self.assertNotIn("--system-prompt", cc.build_command(""))
        cc2 = ClaudeCode("claude-sonnet-5-5", language="Turkish", extra_args=["--safe-mode"], claude_bin="claude")
        cmd = cc2.build_command("")
        self.assertEqual(json.loads(cmd[cmd.index("--settings") + 1]), {"language": "Turkish"})
        self.assertEqual(cmd[-1], "--safe-mode")

    def test_call_cache_and_ledger(self):
        d = tmpdir(self)
        run = FakeRun([envelope('Here: {"change": "k", "value": 3}')])
        cc = ClaudeCode("claude-haiku-5-5", claude_bin="claude", cache_dir=d, runner=run, timeout=42, sleep=lambda s: None)
        r = cc.complete("SYS", "USER")
        cmd, kw = run.calls[0]
        self.assertEqual(kw["input"], "USER")
        self.assertEqual(kw["timeout"], 42)
        self.assertTrue(kw["capture_output"])
        self.assertTrue(Path(kw["cwd"]).is_dir())
        self.assertEqual(list(Path(kw["cwd"]).iterdir()), [])
        self.assertEqual(r.cost_usd, 0.0)
        self.assertEqual(r.raw["list_price_usd"], 0.0123)
        self.assertEqual(json_extract(r.text), {"change": "k", "value": 3})
        rows = cc.ledger.rows()
        self.assertEqual(rows[0]["cost_usd"], 0.0)
        self.assertEqual(rows[0]["list_price_usd"], 0.0123)
        self.assertTrue(cc.complete("SYS", "USER").cached)     # no second subprocess
        self.assertEqual(len(run.calls), 1)
        # language enters the key
        run2 = FakeRun([envelope("x")])
        tr = ClaudeCode("claude-haiku-5-5", language="Turkish", cache_dir=d, runner=run2)
        self.assertFalse(tr.complete("SYS", "USER").cached)
        # complete_json helper
        run3 = FakeRun([envelope("```json\n[1, 2]\n```")])
        self.assertEqual(ClaudeCode(runner=run3).complete_json("s", "u"), [1, 2])

    def test_inline_system(self):
        run = FakeRun([envelope("x")])
        cc = ClaudeCode(runner=run, use_system_prompt=False)
        cc.complete("SYS", "USER")
        cmd, kw = run.calls[0]
        self.assertNotIn("--system-prompt", cmd)
        self.assertEqual(kw["input"], "SYS\n\nUSER")

    def test_errors(self):
        run = FakeRun([subprocess.TimeoutExpired("claude", 1), envelope("transient", is_error=True), envelope("good")])
        cc = ClaudeCode(runner=run, sleep=lambda s: None)
        self.assertEqual(cc.complete("s", "u").text, "good")
        self.assertEqual(len(run.calls), 3)
        run = FakeRun([envelope("Claude AI usage limit reached|1760000000", is_error=True)])
        with self.assertRaises(SpendLimitError):
            ClaudeCode(runner=run, sleep=lambda s: None).complete("s", "u")
        self.assertEqual(len(run.calls), 1)
        run = FakeRun([envelope("Invalid API key · Please run /login", is_error=True)])
        with self.assertRaises(AuthError):
            ClaudeCode(runner=run, sleep=lambda s: None).complete("s", "u")
        run = FakeRun([SimpleNamespace(returncode=1, stdout="garbage", stderr="boom")] * 2)
        with self.assertRaises(LLMError):
            ClaudeCode(runner=run, sleep=lambda s: None, max_attempts=2).complete("s", "u")

    def test_call_budget(self):
        run = FakeRun([envelope("a"), envelope("b"), envelope("c")])
        cc = ClaudeCode(runner=run, max_calls=2)
        rs = cc.map([("s", "1"), ("s", "2")], parallel=2)
        self.assertEqual(sorted(r.text for r in rs), ["a", "b"])
        with self.assertRaises(BudgetExceeded):
            cc.complete("s", "3")

    def test_unsupported_kwargs(self):
        with self.assertRaises(TypeError):
            ClaudeCode(runner=FakeRun([])).complete("s", "u", temperature=0.5)


class TestFakeUnchanged(unittest.TestCase):
    def test_fake(self):
        f = FakeLLM()
        a = f.complete("s", "u")
        self.assertEqual(a.text, FakeLLM().complete("s", "u").text)
        self.assertEqual(len(f.map([("s", "1"), ("s", "2")])), 2)
        self.assertEqual(len(f.calls), 3)


# ---------------------------------------------------------------- real SDK, mocked HTTP transport (offline)
def _sdk():
    try:
        import anthropic
        try:
            import httpx2 as hx
        except ImportError:
            import httpx as hx
        return anthropic, hx
    except ImportError:
        return None, None


@unittest.skipIf(_sdk()[0] is None, "anthropic SDK not installed")
class TestRealSDKOffline(unittest.TestCase):
    """The request dict is accepted and serialised by the real SDK; HTTP errors map to our stops."""

    def client(self, handler):
        anthropic, hx = _sdk()
        return anthropic.Anthropic(api_key="sk-fake", max_retries=0, base_url="http://test.invalid",
                                   http_client=hx.Client(transport=hx.MockTransport(handler)))

    def test_wire_format_and_errors(self):
        _, hx = _sdk()
        seen = []

        def ok(req):
            seen.append(json.loads(req.content))
            return hx.Response(200, json={"id": "msg_x", "type": "message", "role": "assistant",
                                          "model": "claude-haiku-5-5", "content": [{"type": "text", "text": "hi"}],
                                          "stop_reason": "end_turn", "stop_sequence": None,
                                          "usage": {"input_tokens": 12, "output_tokens": 3}})

        def err(status, etype, msg):
            return lambda req: hx.Response(status, json={"type": "error", "error": {"type": etype, "message": msg}})

        r = Claude("claude-haiku-5-5", client=self.client(ok)).complete("SYS", "hello", max_tokens=64)
        self.assertEqual(r.text, "hi")
        self.assertAlmostEqual(r.cost_usd, (12 * 0.1 + 3 * 0.5) / 1e6)
        self.assertEqual(seen[0]["output_config"], {"effort": "low"})
        self.assertEqual(seen[0]["system"][0]["cache_control"], {"type": "ephemeral"})
        cases = [(401, "authentication_error", "invalid x-api-key", AuthError),
                 (429, "rate_limit_error", "You have reached your specified API usage limits.", SpendLimitError),
                 (400, "invalid_request_error", "Your credit balance is too low.", SpendLimitError)]
        for st, et, msg, exc in cases:
            c = Claude("claude-haiku-5-5", client=self.client(err(st, et, msg)), sleep=lambda s: None)
            with self.assertRaises(exc):
                c.complete("s", "u")
        waits = []
        calls = {"n": 0}

        def flaky(req):
            calls["n"] += 1
            if calls["n"] == 1:
                return hx.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
            return ok(req)
        c = Claude("claude-haiku-5-5", client=self.client(flaky), sleep=waits.append)
        self.assertEqual(c.complete("s", "u").text, "hi")
        self.assertEqual(waits, [15.0])


if __name__ == "__main__":
    unittest.main()
