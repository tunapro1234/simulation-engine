"""Offline tests for D3-rehearsal gap #3 (llm side): tag_prefix on the backends, env SIMENGINE_SPEND_TAG_PREFIX,
per-call tag combined with the instance tag, ask_panel's "prefixed with the backend's own tag" claim.
No network, no real API or CLI calls."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from simengine.llm import (AGENT_ENV, SPEND_LOG_ENV, TAG_PREFIX_ENV, Claude, ClaudeCode, CostLedger, FakeLLM,
                           compose_tag)
from tests.test_llm import FakeClient, FakeRun, envelope, ok_message, tmpdir

CLEAN_ENV = {SPEND_LOG_ENV: "", AGENT_ENV: "", TAG_PREFIX_ENV: ""}


def tsv_tags(path: Path) -> list[str]:
    return [ln.split("\t")[6] for ln in path.read_text(encoding="utf-8").splitlines()]


def jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]


def make(client, **kw):
    kw.setdefault("sleep", lambda s: None)
    kw.setdefault("max_tokens", 2000)
    kw.setdefault("thinking", "off")
    return Claude("claude-haiku-5-5", effort="low", client=client, **kw)


class TestComposeTag(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(compose_tag("", "survey:persona"), "survey:persona")
        self.assertEqual(compose_tag("", None), "")
        self.assertEqual(compose_tag(None, "x"), "x")
        self.assertEqual(compose_tag("L1", None), "L1")
        self.assertEqual(compose_tag("L1", ""), "L1")
        self.assertEqual(compose_tag("L1", "survey:persona"), "L1:survey:persona")
        # the call tag is never inspected: no "already combined" guess (regression: Claude(tag="survey") + a
        # default survey() logged "survey:persona", dropping the backend tag)
        self.assertEqual(compose_tag("L1", "L1"), "L1:L1")
        self.assertEqual(compose_tag("survey", "survey:persona"), "survey:survey:persona")
        self.assertEqual(compose_tag("dev", "dev:x"), "dev:dev:x")
        self.assertEqual(compose_tag("L1", "L10"), "L1:L10")


@mock.patch.dict(os.environ, CLEAN_ENV)
class TestBackendTags(unittest.TestCase):
    def test_old_behaviour_without_prefix_or_instance_tag(self):
        d = tmpdir(self)
        c = make(FakeClient(), cache_dir=d / "c", spend_log=d / "s.log")
        c.complete("s", "u1", tag="persona")
        c.complete("s", "u2")
        self.assertEqual(tsv_tags(d / "s.log"), ["persona", ""])
        rows = jsonl_rows(d / "c" / "ledger.jsonl")
        self.assertEqual([r["tag"] for r in rows], ["persona", ""])
        self.assertTrue(all("tag_prefix" not in r for r in rows))

    def test_instance_tag_combines_with_call_tag(self):
        d = tmpdir(self)
        c = make(FakeClient(), cache_dir=d / "c", spend_log=d / "s.log", tag="L1_dev")
        c.complete("s", "u1", tag="prior")
        c.complete("s", "u2")
        c.map([("s", "v1"), ("s", "v2")], tag="survey:persona", parallel=2)
        self.assertEqual(sorted(tsv_tags(d / "s.log")),
                         sorted(["L1_dev:prior", "L1_dev", "L1_dev:survey:persona", "L1_dev:survey:persona"]))
        self.assertEqual(sorted(r["tag"] for r in jsonl_rows(d / "c" / "ledger.jsonl")),
                         sorted(["L1_dev:prior", "L1_dev", "L1_dev:survey:persona", "L1_dev:survey:persona"]))

    def test_tag_prefix_jsonl_and_tsv(self):
        d = tmpdir(self)
        c = make(FakeClient(), cache_dir=d / "c", spend_log=d / "s.log", agent="a1", tag_prefix="prova-d3/",
                 tag="L1")
        self.assertEqual(c.tag_prefix, "prova-d3/")
        self.assertEqual(c.full_tag("survey:persona"), "prova-d3/L1:survey:persona")
        c.complete("s", "u", tag="survey:persona")
        c.complete("s", "u", tag="survey:persona")            # cache hit: no rows
        line = (d / "s.log").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(line), 1)
        self.assertEqual(line[0].split("\t")[1:3], ["a1", "anthropic/claude-haiku-5-5"])
        self.assertEqual(tsv_tags(d / "s.log"), ["prova-d3/L1:survey:persona"])
        rows = jsonl_rows(d / "c" / "ledger.jsonl")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tag"], "L1:survey:persona")
        self.assertEqual(rows[0]["tag_prefix"], "prova-d3/")
        self.assertEqual(rows[0]["tag_prefix"] + rows[0]["tag"], c.full_tag("survey:persona"))

    def test_prefix_without_instance_tag(self):
        d = tmpdir(self)
        c = make(FakeClient(), spend_log=d / "s.log", tag_prefix="prova-d3/")
        c.complete("s", "u", tag="survey:no_persona")
        self.assertEqual(tsv_tags(d / "s.log"), ["prova-d3/survey:no_persona"])

    def test_env_prefix_and_explicit_override(self):
        d = tmpdir(self)
        with mock.patch.dict(os.environ, {TAG_PREFIX_ENV: "envrun/", SPEND_LOG_ENV: str(d / "s.log")}):
            c = make(FakeClient(), cache_dir=d / "c")
            c.complete("s", "u1", tag="a")
            c2 = make(FakeClient(), tag_prefix="")             # explicit "" ignores the env var
            c2.complete("s", "u2", tag="b")
            c3 = make(FakeClient(), tag_prefix="arg/")         # explicit value wins
            c3.complete("s", "u3", tag="c")
        self.assertEqual(tsv_tags(d / "s.log"), ["envrun/a", "b", "arg/c"])
        self.assertEqual(jsonl_rows(d / "c" / "ledger.jsonl")[0]["tag_prefix"], "envrun/")

    def test_tag_prefix_with_ledger_instance_is_an_error(self):
        with self.assertRaises(ValueError):
            make(FakeClient(), ledger=CostLedger(None), tag_prefix="x/")
        # the instance's own prefix is used
        d = tmpdir(self)
        c = make(FakeClient(), ledger=CostLedger(None, tsv_path=d / "s.log", tag_prefix="own/"), tag="t")
        c.complete("s", "u", tag="k")
        self.assertEqual(tsv_tags(d / "s.log"), ["own/t:k"])

    def test_ledger_path_gets_prefix(self):
        d = tmpdir(self)
        c = make(FakeClient(), ledger=d / "l.jsonl", spend_log=d / "s.log", tag_prefix="p/")
        c.complete("s", "u", tag="k")
        self.assertEqual(tsv_tags(d / "s.log"), ["p/k"])
        self.assertEqual(jsonl_rows(d / "l.jsonl")[0]["tag_prefix"], "p/")

    def test_claude_code_rows(self):
        d = tmpdir(self)
        cc = ClaudeCode(runner=FakeRun([envelope("ok"), envelope("ok2")]), cache_dir=d / "c", spend_log=d / "s.log",
                        agent="obs", tag="loop", tag_prefix="prova-d3/", sleep=lambda s: None)
        cc.complete("s", "u1", tag="observer")
        cc.complete("s", "u2")
        self.assertEqual(tsv_tags(d / "s.log"), ["prova-d3/loop:observer", "prova-d3/loop"])
        rows = jsonl_rows(d / "c" / "ledger.jsonl")
        self.assertEqual([r["tag"] for r in rows], ["loop:observer", "loop"])
        self.assertEqual({r["tag_prefix"] for r in rows}, {"prova-d3/"})
        self.assertEqual({r["model"] for r in rows}, {"claude-p/claude-haiku-5-5"})

    def test_fake_llm_still_accepts_tag(self):
        self.assertEqual(len(FakeLLM().map([("s", "u")], tag="x")), 1)


@mock.patch.dict(os.environ, CLEAN_ENV)
class TestAskPanelTagClaim(unittest.TestCase):
    """simengine.sim.ask_panel documents: the ledger tag is `llm_tag` (default `tag`), prefixed with the
    backend's own tag. ask_many already builds "<backend tag>:<llm_tag>"; the backend must not double it."""

    def personas(self):
        from simengine.sim import as_personas
        return as_personas([{"id": "p1", "age": 30}, {"id": "p2", "age": 40}])

    def reply(self, params):
        user = params["messages"][0]["content"]
        ids = [i for i in ("p1", "p2") if i in user]
        return ok_message(json.dumps([{"id": i, "score": 5} for i in ids]))

    def test_backend_tag_and_prefix(self):
        from simengine.sim import ask_panel
        d = tmpdir(self)
        c = make(FakeClient(default=self.reply), cache_dir=d / "c", spend_log=d / "s.log", tag="L1",
                 tag_prefix="prova-d3/")
        out = ask_panel(c, self.personas(), "Rate it.", {"score": "int 1-10"}, batch=1, llm_tag="survey:persona")
        self.assertEqual(sorted(out), ["p1", "p2"])
        self.assertEqual(tsv_tags(d / "s.log"), ["prova-d3/L1:survey:persona"] * 2)
        self.assertEqual([r["tag"] for r in jsonl_rows(d / "c" / "ledger.jsonl")], ["L1:survey:persona"] * 2)

    def test_backend_tag_equal_to_call_prefix_is_kept(self):
        """Regression: a backend tag that is also the first part of the call tag was silently dropped."""
        from simengine.sim import ask_panel
        d = tmpdir(self)
        c = make(FakeClient(default=self.reply), cache_dir=d / "c", spend_log=d / "s.log", tag="survey")
        ask_panel(c, self.personas(), "Rate it.", {"score": "int 1-10"}, batch=2, llm_tag="survey:persona")
        self.assertEqual(tsv_tags(d / "s.log"), ["survey:survey:persona"])

    def test_foreign_backend_gets_composed_tag(self):
        from simengine.sim.panel import _ledger_tag

        class Other:
            tag = "survey"
        self.assertEqual(_ledger_tag(Other(), "survey:persona"), "survey:survey:persona")
        self.assertEqual(_ledger_tag(Other(), ""), "survey")
        self.assertEqual(_ledger_tag(object(), "x"), "x")
        c = make(FakeClient(default=self.reply), tag="L1")
        self.assertEqual(_ledger_tag(c, "survey:persona"), "survey:persona")   # the backend composes it

    def test_no_backend_tag_unchanged(self):
        from simengine.sim import ask_panel
        d = tmpdir(self)
        c = make(FakeClient(default=self.reply), spend_log=d / "s.log")
        ask_panel(c, self.personas(), "Rate it.", {"score": "int 1-10"}, batch=2, tag="ask")
        self.assertEqual(tsv_tags(d / "s.log"), ["ask"])


if __name__ == "__main__":
    unittest.main()
