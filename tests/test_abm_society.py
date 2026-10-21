import json
import os
import re
import unittest

import simengine.abm as A
from simengine.llm import FakeLLM

AGENTS = [A.SocietyAgent("P01", "19, college student, retweets jokes freely."),
          A.SocietyAgent("P02", "60, retiree, rarely shares anything."),
          A.SocietyAgent("P03", "24, student journalist, shares breaking news."),
          A.SocietyAgent("P04", "45, small business owner, shares local alerts.")]


def per_agent_responder(system, user):
    """Students share at the first checkpoint, everyone else once two people have shared."""
    k = int(re.search(r"count (\d+)", user).group(1))
    share = "student" in system or k >= 2
    return json.dumps({"share": share})


def batch_responder(system, user):
    ids = re.findall(r"\[(P\d\d)\]", user)
    return json.dumps({"share": [i for i in ids if i in ("P01", "P03")]})


class TestSociety(unittest.TestCase):
    def test_per_agent_cascade(self):
        llm = FakeLLM(per_agent_responder)
        run = A.AgentSociety(AGENTS, A.ShareCascadeWorld("Big news!"), llm, mode="per_agent").run()
        self.assertEqual(run.states[1]["shared"], ["P01", "P03"])
        self.assertEqual(run.states[-1]["curve"], [2, 4, 4])
        self.assertEqual(run.stopped, "completed")               # turn 3 advances without LLM calls
        self.assertEqual(run.calls, 4 + 2)                       # only non-sharers are asked
        self.assertEqual(run.invalid, 0)
        self.assertEqual(set(run.log.columns) >= {"turn", "agent", "action", "valid", "raw", "prompt_id"}, True)

    def test_batch_research_format(self):
        llm = FakeLLM(batch_responder)
        run = A.AgentSociety(AGENTS, A.ShareCascadeWorld("Meh."), llm, mode="batch").run()
        self.assertEqual(run.calls, 3)                           # one call per turn
        self.assertEqual(run.states[-1]["shared"], ["P01", "P03"])
        self.assertTrue(run.log.valid.all())

    def test_invalid_replies_fall_back_and_are_flagged(self):
        llm = FakeLLM()                                          # hash strings, never JSON
        run = A.AgentSociety(AGENTS, A.ShareCascadeWorld("x"), llm, retries=1).run()
        self.assertEqual(run.invalid, len(run.log))
        self.assertEqual(run.states[-1]["shared"], [])
        self.assertEqual(run.calls, 3 * 4 * 2)                   # each agent retried once per turn

    def test_budget_stop(self):
        run = A.AgentSociety(AGENTS, A.ShareCascadeWorld("x"), FakeLLM(per_agent_responder), max_calls=5).run()
        self.assertTrue(run.stopped.startswith("budget"))
        self.assertLessEqual(run.calls, 5)

    def test_deterministic_replay(self):
        def go():
            w = A.ChoiceWorld("Which phone?", ["A", "B"], n_turns=3, contexts=["", "A is 10% cheaper.", ""])
            llm = FakeLLM(lambda s, u: json.dumps({"choice": "A" if ("cheaper" in u or "student" in s) else "B"}))
            return A.AgentSociety(AGENTS, w, llm).run()
        a, b = go(), go()
        self.assertTrue(a.log.equals(b.log))
        self.assertEqual(a.states, b.states)
        self.assertEqual(a.states[1]["history"][0], {"A": 2, "B": 2})
        self.assertEqual(a.states[2]["history"][1], {"A": 4, "B": 0})
        self.assertEqual(a.events[1]["changed"], ["P02", "P04"])

    def test_world_protocol(self):
        self.assertIsInstance(A.ShareCascadeWorld("x"), A.World)
        self.assertIsInstance(A.ChoiceWorld("q", ["a"]), A.World)

    def test_claude_code_backend_with_fake_runner(self):
        import tempfile
        from types import SimpleNamespace
        from simengine.llm import ClaudeCode
        seen = []

        def runner(cmd, input=None, **kw):
            seen.append(cmd)
            system = cmd[cmd.index("--system-prompt") + 1] if "--system-prompt" in cmd else ""
            text = per_agent_responder(system, input)
            return SimpleNamespace(stdout=json.dumps({"result": text, "is_error": False, "usage": {}}), stderr="",
                                   returncode=0)

        with tempfile.TemporaryDirectory() as d:
            cc = ClaudeCode("claude-haiku-5-5", cache_dir=d, runner=runner, sleep=lambda s: None, max_calls=4)
            run = A.AgentSociety(AGENTS, A.ShareCascadeWorld("Big news!"), cc, parallel=2).run()
            self.assertEqual(run.states[1]["shared"], ["P01", "P03"])
            self.assertTrue(run.stopped.startswith("budget"))       # the backend's own call cap stops the loop
            self.assertTrue(all(c[1] == "-p" for c in seen))

    @unittest.skipUnless(os.environ.get("SIMENGINE_LIVE") == "1", "live smoke test (claude -p) only with SIMENGINE_LIVE=1")
    def test_live_claude_code(self):  # pragma: no cover
        from simengine.llm import ClaudeCode
        run = A.AgentSociety(AGENTS[:2], A.ShareCascadeWorld("Free pizza on campus today!", times=["1 hour"]),
                             ClaudeCode(model="claude-haiku-5-5"), max_calls=4).run()
        self.assertLessEqual(run.calls, 4)


if __name__ == "__main__":
    unittest.main()
