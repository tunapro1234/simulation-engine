"""simengine.games.play: schema, invalid answers, retries, belief layer, resume, cost gate (FakeLLM only)."""
import json
import os
import re
import tempfile
import unittest

import pandas as pd

import simengine.games as gm
from simengine.llm import FakeLLM

# scripted answers per game: (game -> function(condition_id) -> answer)
SCRIPT = {
    "dictator": lambda c: 30,
    "ultimatum_proposer": lambda c: 40,
    "ultimatum_responder": lambda c: "Reject" if c == "low" else "Accept",
    "trust_sender": lambda c: 50,
    "trust_returner": lambda c: {"low": 10, "mid": 30, "high": 50}[c],
    "beauty_contest": lambda c: 33,
}


def scripted(belief_answer=40):
    """Responder that identifies the game and condition from the rendered text (the persona is in system)."""
    texts = {}
    for name in SCRIPT:
        for c in gm.GAMES[name]["conditions"]:
            texts[gm.render(name, c["id"])] = (name, c["id"])

    def respond(system, user):
        if "predict" in user and "Before this decision" not in user:
            return json.dumps({"answer": belief_answer})
        for t, (name, cid) in texts.items():
            if t in user:
                return json.dumps({"answer": SCRIPT[name](cid)})
        return "no idea"
    return respond


PERSONAS = {"baseline": {"baseline-000": ""},
            "big5": {"big5-000": {"text": "You are warm and outgoing.", "params": {"A": 4.5, "E": 4.0}},
                     "big5-001": {"text": "You are reserved.", "params": {"A": 2.0, "E": 1.5}}}}


class TestPlay(unittest.TestCase):
    def test_schema_and_values(self):
        llm = FakeLLM(scripted())
        res = gm.play(llm, PERSONAS, list(SCRIPT), runs=2)
        self.assertEqual(list(res.columns), gm.COLUMNS + gm.EXTRA_COLUMNS)
        n_cond = sum(len(gm.GAMES[g]["conditions"]) for g in SCRIPT)
        self.assertEqual(len(res), 3 * n_cond * 2)
        self.assertTrue(res["valid"].all())
        d = res[res.game == "dictator"]
        self.assertEqual(set(d["decision"]), {30})
        ur = res[(res.game == "ultimatum_responder")].set_index(["agent_id", "run", "condition"])["decision"]
        self.assertEqual(ur[("big5-000", 0, "low")], "Reject")
        self.assertEqual(set(res["wording_id"]), {0})
        self.assertEqual(set(res["run"]), {0, 1})
        p = res[res.agent_id == "big5-000"]["persona_params"].iloc[0]
        self.assertEqual(json.loads(p), {"A": 4.5, "E": 4.0})
        self.assertEqual(set(res["max_tokens"]), {1024})
        self.assertEqual(set(res["seed"]), {0})
        self.assertTrue(res[res.game == "beauty_contest"]["holdout"].all())

    def test_persona_goes_in_system_and_belief_first(self):
        llm = FakeLLM(scripted(belief_answer=45))
        res = gm.play(llm, {"big5": {"big5-000": "You are warm."}}, ["trust_sender"], runs=1)
        self.assertEqual(len(llm.calls), 2)
        (s1, u1), (s2, u2) = llm.calls
        self.assertEqual(s1, "You are warm.")
        self.assertEqual(s2, "You are warm.")
        self.assertIn("predict how many dollars", u1)
        self.assertIn("Your answer was: 45", u2)
        self.assertNotIn("You are warm", u2)
        row = res.iloc[0]
        self.assertEqual(row["belief"], 45)
        self.assertTrue(row["belief_valid"])
        res2 = gm.play(FakeLLM(scripted()), {"big5": {"big5-000": "x"}}, ["trust_sender"], runs=1, belief=False)
        self.assertTrue(pd.isna(res2.iloc[0]["belief"]))

    def test_unusable_answers_are_invalid_never_zero(self):
        llm = FakeLLM(lambda s, u: "I'd rather not say.")
        res = gm.play(llm, PERSONAS, ["dictator", "ultimatum_responder"], runs=1, retries=2)
        self.assertFalse(res["valid"].any())
        self.assertTrue(res["decision"].isna().all())
        self.assertFalse((res["decision"] == 0).any())
        self.assertTrue((res["attempts"] == 3).all())
        self.assertTrue(res["error"].notna().all())
        m = gm.moments(res)
        self.assertTrue(m["value"].isna().all())
        self.assertTrue((m["n_valid"] == 0).all())

    def test_out_of_range_kept_and_flagged(self):
        llm = FakeLLM(lambda s, u: '{"answer": 40}')
        res = gm.play(llm, {"m": {"a": ""}}, ["trust_returner"], runs=1, retries=0)
        r = res.set_index("condition")
        self.assertFalse(r.loc["low", "valid"])          # 40 > 30 received
        self.assertEqual(r.loc["low", "decision"], 40)
        self.assertTrue(r.loc["mid", "valid"])
        flags = gm.coherence_audit(res)
        self.assertIn("tr_overreturn", set(flags["flag"]))

    def test_retry_feeds_error_back(self):
        replies = iter(['{"answer": 500}', '{"answer": 20}'])
        llm = FakeLLM(lambda s, u: next(replies))
        res = gm.play(llm, {"m": {"a": ""}}, ["dictator"], runs=1)
        self.assertTrue(res.iloc[0]["valid"])
        self.assertEqual(res.iloc[0]["decision"], 20)
        self.assertEqual(res.iloc[0]["attempts"], 2)
        self.assertIn("outside 0-100", llm.calls[1][1])
        self.assertNotIn("outside", llm.calls[0][1])

    def test_resume_without_duplicates(self):
        llm = FakeLLM(scripted())
        first = gm.play(llm, PERSONAS, ["dictator", "ultimatum_responder"], runs=2)
        n = len(llm.calls)
        again = gm.play(llm, PERSONAS, ["dictator", "ultimatum_responder"], runs=2, resume=first)
        self.assertEqual(len(llm.calls), n)              # nothing re-asked
        self.assertEqual(len(again), len(first))
        more = gm.play(llm, PERSONAS, ["dictator", "ultimatum_responder"], runs=3, resume=first)
        self.assertEqual(len(more), len(first) * 3 // 2)
        self.assertFalse(more.duplicated(subset=gm.KEY).any())

    def test_resume_from_csv_out(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "results.csv")
            llm = FakeLLM(scripted())
            a = gm.play(llm, PERSONAS, ["dictator"], runs=1, out=out)
            b = gm.play(llm, PERSONAS, ["dictator", "beauty_contest"], runs=1, out=out)
            self.assertEqual(len(b), 6)
            disk = gm.read_results(out)
            self.assertEqual(len(disk), 6)
            self.assertFalse(disk.duplicated(subset=gm.KEY).any())
            self.assertEqual(disk["valid"].dtype, bool)
            self.assertEqual(len(a), 3)

    def test_cost_gate(self):
        llm = FakeLLM(scripted())
        with self.assertRaises(gm.CostGateError) as cm:
            gm.play(llm, PERSONAS, ["dictator"], runs=3, est_cost_per_call=1.0)
        self.assertEqual(cm.exception.estimate["calls"], 9)
        self.assertEqual(len(llm.calls), 0)
        res = gm.play(llm, PERSONAS, ["dictator"], runs=3, est_cost_per_call=1.0, confirm=True)
        self.assertEqual(len(res), 9)
        # a higher gate lets it through; FakeLLM without an estimate is free
        self.assertEqual(gm.estimate_cost(llm, PERSONAS, ["dictator"])["usd"], 0.0)

    def test_cost_gate_prices_real_models_and_refuses_unknown(self):
        class Wrapper:
            def __init__(self, model):
                self.model, self.inner = model, FakeLLM(scripted())

            def complete(self, s, u, **kw):
                return self.inner.complete(s, u, **kw)

            def map(self, prompts, **kw):
                return self.inner.map(prompts, **kw)

        est = gm.estimate_cost(Wrapper("claude-haiku-5-5"), PERSONAS, "all", runs=3)
        self.assertGreater(est["usd"], 0)
        with self.assertRaises(ValueError):
            gm.estimate_cost(Wrapper("mystery-model"), PERSONAS, ["dictator"])
        big = gm.estimate_cost(Wrapper("claude-opus-5-5"), {"m": {f"a{i}": "" for i in range(50)}}, "all", runs=3)
        self.assertGreater(big["usd"], 5.0)
        with self.assertRaises(gm.CostGateError):
            gm.play(Wrapper("claude-opus-5-5"), {"m": {f"a{i}": "" for i in range(50)}}, "all", runs=3)

    def test_runs_are_independent_samples(self):
        seen = []

        class Rec(FakeLLM):
            def complete(self, system, user, **kw):
                seen.append(kw.get("sample"))
                return super().complete(system, user, **kw)

        gm.play(Rec(scripted()), {"m": {"a": ""}}, ["dictator"], runs=3, seed=2)
        self.assertEqual(seen, [2000, 2001, 2002])

    def test_as_agents_forms(self):
        self.assertEqual(len(gm.as_agents(["a", "b"])), 2)
        ag = gm.as_agents([{"id": "p1", "method": "gps", "persona": "x", "traits": {"z": 1}}])
        self.assertEqual(ag[0], {"agent_id": "p1", "persona_method": "gps", "text": "x", "params": {"z": 1}})
        ag = gm.as_agents({"p1": "hello"}, method="demo")
        self.assertEqual(ag[0]["persona_method"], "demo")
        with self.assertRaises(ValueError):
            gm.as_agents([{"id": "p1"}, {"id": "p1"}])


if __name__ == "__main__":
    unittest.main()
