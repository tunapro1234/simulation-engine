import collections
import json
import os
import re
import tempfile
import unittest

from simengine.llm import FakeLLM
from simengine.persona import (Audience, AudienceSpec, ClaudeSource, FakeSource, LLMError, Persona, TalpSource,
                               cell_plan, json_extract, open_source)
from simengine.types import LLMResponse


def spec(**aud):
    base = dict(identity_text="urban renters", behavioral_anchor="orders groceries weekly")
    base.update(aud)
    return AudienceSpec(target_description="Urban renters", locale="en-US", vertical="grocery",
                        audience=Audience(**base))


def persona_responder(system, user):
    """FakeLLM responder: valid persona JSON for generate prompts, answers for ask prompts."""
    if "Write one persona for each id" in user:
        ids = re.findall(r"- id (p\d+):", user)
        return "```json\n" + json.dumps([{"id": i, "name": f"Name {i}", "tagline": "t", "persona": "story",
                                          "behavior_voice": "I shop", "buying_context": "app",
                                          "motivations": ["m"], "frustrations": ["f"], "media_habits": [],
                                          "price_sensitivity": 0.25, "novelty_seeking": 1}
                                         for i in ids]) + "\n```"
    ids = re.findall(r"^### (\S+)$", user, re.M)
    return json.dumps([{"id": i, "score": len(i)} for i in ids])


class TestCellPlan(unittest.TestCase):
    def test_balanced_and_deterministic(self):
        s = spec(income_tier=["low", "middle"], usage_intensity=["light", "heavy", "bogus"])
        plan = cell_plan(s, 10, seed=3)
        self.assertEqual(len(plan), 10)
        self.assertEqual(collections.Counter(c["income_tier"] for c in plan), {"low": 5, "middle": 5})
        self.assertEqual(set(c["usage_intensity"] for c in plan), {"light", "heavy"})   # out-of-enum dropped
        self.assertEqual(set(c["gender"] for c in plan), {"female", "male"})             # default gender mix
        self.assertEqual(collections.Counter(c["age_band"] for c in plan).most_common(1)[0][1], 2)
        self.assertEqual(plan, cell_plan(s, 10, seed=3))
        self.assertNotEqual(plan, cell_plan(s, 10, seed=4))
        self.assertNotIn("household_composition", plan[0])


class TestFakeSource(unittest.TestCase):
    def test_generate_and_ask(self):
        src = FakeSource(budget=10)
        ps = src.generate(spec(income_tier=["low", "affluent"]), n=10, batch=4)
        self.assertEqual([p.pid for p in ps], [f"p{i:03d}" for i in range(1, 11)])
        self.assertEqual(src.counter.calls, 3)
        self.assertTrue(all(p.trait("income_tier") in ("low", "affluent") for p in ps))
        self.assertTrue(ps[0].authored.persona.startswith("(fake)"))
        self.assertEqual(ps[0].segment, ps[0].trait("income_tier"))
        ans = src.ask(ps, "Rate it", {"score": "int 1-10", "pick": "one of A|B", "p": "float 0.0-1.0",
                                      "why": "str"}, batch=5)
        self.assertEqual(set(ans), {p.pid for p in ps})
        for a in ans.values():
            self.assertTrue(1 <= a["score"] <= 10)
            self.assertIn(a["pick"], ("A", "B"))
            self.assertTrue(0 <= a["p"] <= 1)
        again = FakeSource().ask(ps, "Rate it", {"score": "int 1-10"}, batch=5)
        self.assertEqual(again, FakeSource().ask(ps, "Rate it", {"score": "int 1-10"}, batch=5))

    def test_budget_gives_marked_stubs(self):
        src = FakeSource(budget=2)
        ps = src.generate(spec(), n=24, batch=8)
        self.assertEqual(len(ps), 24)
        self.assertEqual(src.counter.calls, 2)
        self.assertTrue(ps[-1].extra.get("generation_error"))
        self.assertTrue(ps[-1].to_dict()["generation_error"])
        self.assertFalse(ps[0].extra.get("generation_error"))

    def test_log_and_events(self):
        with tempfile.TemporaryDirectory() as d:
            ev = []
            src = FakeSource(log_path=os.path.join(d, "log.jsonl"), on_event=ev.append)
            src.ask([Persona.from_dict({"identity": {"persona_id": "a"}})], "q", {"x": "int 1-3"})
            with open(os.path.join(d, "log.jsonl")) as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["driver"], "fake")
            self.assertEqual(ev[-1]["kind"], "reply")

    def test_open_source(self):
        self.assertIsInstance(open_source("fake"), FakeSource)
        with self.assertRaises(ValueError):
            open_source("nope")


class TestClaudeSource(unittest.TestCase):
    def test_generate_with_fake_llm(self):
        llm = FakeLLM(persona_responder)
        src = ClaudeSource(llm, budget=50)
        ps = src.generate(spec(income_tier=["middle"]), n=5, batch=2)
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(ps[4].identity.name, "Name p005")
        self.assertEqual(ps[0].quantitative_dispositions[0].dimension, "price_sensitivity")
        self.assertEqual(len(ps[0].quantitative_dispositions), 2)
        self.assertIn("income_tier=middle", llm.calls[0][1])
        self.assertIn("Behavioral anchor: orders groceries weekly", llm.calls[0][1])
        ans = src.ask(ps, "How much?", {"score": "int"}, batch=3, context={"p001": "bought last week"})
        self.assertEqual(ans["p001"]["score"], 4)
        self.assertIn("Context: bought last week", llm.calls[3][1])
        self.assertEqual(src.summary()["calls"], 5)

    def test_cached_replies_do_not_use_budget(self):
        class CachedLLM(FakeLLM):
            def complete(self, system, user, *, max_tokens=256, **kw):
                r = super().complete(system, user, max_tokens=max_tokens, **kw)
                r.cached = True
                return r
        src = ClaudeSource(CachedLLM(persona_responder), budget=1)
        ps = src.generate(spec(), n=6, batch=2)
        self.assertEqual(src.counter.cached, 3)
        self.assertEqual(src.counter.calls, 0)
        self.assertFalse(any(p.extra.get("generation_error") for p in ps))

    def test_auth_error_stops_run(self):
        class BrokenLLM:
            model = "broken"
            n = 0

            def complete(self, system, user, *, max_tokens=256, **kw):
                BrokenLLM.n += 1
                raise RuntimeError("HTTP 401 unauthorized")

            def map(self, prompts, *, max_tokens=256, parallel=4, **kw):
                raise RuntimeError("HTTP 401 unauthorized")
        src = ClaudeSource(BrokenLLM())
        ps = [Persona.from_dict({"identity": {"persona_id": f"x{i}"}}) for i in range(6)]
        self.assertEqual(src.ask(ps, "q", {"a": "str"}, batch=2), {})
        self.assertEqual(BrokenLLM.n, 1)                 # stopped after the first failure
        self.assertEqual(src.counter.errors, 3)
        with self.assertRaises(LLMError):
            src.check_errors()

    def test_partial_failure_isolated(self):
        class FlakyLLM(FakeLLM):
            def map(self, prompts, **kw):
                raise RuntimeError("one bad request")

            def complete(self, system, user, *, max_tokens=256, **kw):
                if "### b" in user:
                    raise ValueError("bad")
                return super().complete(system, user, max_tokens=max_tokens, **kw)
        src = ClaudeSource(FlakyLLM(persona_responder))
        ps = [Persona.from_dict({"identity": {"persona_id": i}}) for i in "ab"]
        ans = src.ask(ps, "q", {"score": "int"}, batch=1)
        self.assertEqual(set(ans), {"a"})
        self.assertEqual(src.counter.errors, 1)


class TestMisc(unittest.TestCase):
    def test_json_extract(self):
        self.assertEqual(json_extract('noise ```json\n[{"id": 1}]\n``` tail'), [{"id": 1}])
        self.assertEqual(json_extract('Sure: {"a": {"b": 2}} done'), {"a": {"b": 2}})
        self.assertEqual(json_extract('[{"id": "a", "v": 1}, {"id": "b", "v": 2}, {"id": "c", "v'),
                         [{"id": "a", "v": 1}, {"id": "b", "v": 2}])
        self.assertIsNone(json_extract("no json here"))
        self.assertIsNone(json_extract(None))

    def test_talp_slot_is_empty(self):
        old = {k: os.environ.pop(k, None) for k in ("TALP_API_URL", "TALP_API_KEY")}
        try:
            t = TalpSource()
            self.assertFalse(t.configured)
            with self.assertRaises(NotImplementedError):
                t.generate(spec(), 3)
            with self.assertRaises(NotImplementedError):
                t.ask([], "q", {})
            with self.assertRaises(NotImplementedError):
                TalpSource(base_url="http://127.0.0.1:9", api_key="dummy").generate(spec(), 3)
        finally:
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v

    def test_llm_response_type(self):
        r = FakeLLM().complete("s", "u")
        self.assertIsInstance(r, LLMResponse)


if __name__ == "__main__":
    unittest.main()
