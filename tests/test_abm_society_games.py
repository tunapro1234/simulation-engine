"""Economic-game societies: who-to-punish (Ertan, Page & Putterman 2009 design), repeated PD, trust game.
All offline with scripted FakeLLM responders."""
import json
import re
import unittest

import simengine.abm as A
from simengine.llm import FakeLLM

COOP, FREE = "A steady cooperator.", "A free rider."


def vote_no(user):
    return 2 if "Vote 1 outcome" in user else 1


def item_of(user):
    m = re.search(r"who contributed (LESS than|exactly|MORE than) the group average\?", user)
    return {"LESS than": "low", "exactly": "avg", "MORE than": "high"}[m.group(1)]


def player(user):
    return re.search(r"You are Player ([A-D])", user).group(1)


def punisher_responder(system, user):
    """Vote 1 allows punishing low contributors, vote 2 removes it (dismantling); free rider gives 0, others 10;
    every allowed punishment is $2."""
    if "Ballot item" in user:
        yes = vote_no(user) == 1 and item_of(user) == "low"
        return json.dumps({"vote": "Yes" if yes else "No"})
    if "How much of your $10" in user:
        return json.dumps({"contribution": 0 if system == FREE else 10})
    if "reduce their earnings" in user:
        return 'Reasoning... {"punish": 2}'
    raise AssertionError(user)


def zero_responder(system, user):
    if "Ballot item" in user:
        return '{"vote": "No"}'
    if "How much of your $10" in user:
        return '{"contribution": 0}'
    raise AssertionError("no punishment stage expected")


class TestWhoToPunish(unittest.TestCase):
    def test_zero_contributors_no_punish_hand_payoffs(self):
        llm = FakeLLM(zero_responder)
        run = A.run_who_to_punish(llm, seed=7)
        g = run["groups"][0]
        self.assertEqual(g["votes"], [False, False])
        self.assertEqual(g["rules"], [{"low": False, "avg": False, "high": False}] * 2)
        self.assertEqual(len(g["periods"]), 6)
        for p in g["periods"]:
            self.assertEqual(p["contrib"], {L: 0 for L in "ABCD"})
            self.assertEqual(p["earnings"], {L: 10.0 for L in "ABCD"})    # (10 - 0) + 0.4 * 0
            self.assertEqual(p["received"], {L: 0 for L in "ABCD"})
            self.assertEqual(p["n_punish_pairs"], 0)
        self.assertEqual(run["calls"], 2 * 12 + 6 * 4)                    # 3 items x 4 players per vote
        self.assertEqual(run["punishments"], [])
        self.assertEqual(run["invalid"]["total"], 0)
        self.assertFalse(g["dismantled"])
        self.assertTrue(run["complete"])
        self.assertEqual(run["rules"][0]["tally"]["low"], dict(yes=0, no=4, no_preference=0, invalid=0))
        row = run["contributions"][0]
        for k in ("persona_method", "agent_id", "seed", "traits", "max_tokens", "wording_id", "valid"):
            self.assertIn(k, row)
        self.assertEqual(row["seed"], 7)
        json.dumps(run)                                                   # JSON-serialisable

    def test_punishment_hand_calculation_and_dismantling(self):
        llm = FakeLLM(punisher_responder)
        run = A.run_who_to_punish(llm, [COOP, COOP, COOP, FREE])
        g = run["groups"][0]
        self.assertEqual(g["votes"], [True, False])
        self.assertTrue(g["dismantled"])
        self.assertFalse(g["ever_punish_high"])
        p1 = g["periods"][0]
        self.assertTrue(p1["punish_low"])
        self.assertEqual(p1["total"], 30)
        self.assertEqual(p1["avg"], 7.5)
        self.assertEqual(p1["given"], dict(A=2, B=2, C=2, D=0))
        self.assertEqual(p1["received"], dict(A=0, B=0, C=0, D=6))
        # A: (10-10) + 0.4*30 - 0.25*2 = 11.5 ; D: (10-0) + 12 - 6 = 16
        self.assertEqual(p1["earnings"], dict(A=11.5, B=11.5, C=11.5, D=16.0))
        p4 = g["periods"][3]
        self.assertFalse(p4["punish_low"])
        self.assertEqual(p4["earnings"], dict(A=12.0, B=12.0, C=12.0, D=22.0))
        self.assertEqual(len(run["punishments"]), 3 * 3)                  # 3 punishers -> D, periods 1-3
        pr = run["punishments"][0]
        self.assertEqual((pr["punisher"], pr["target"], pr["target_contribution"], pr["avg"], pr["amount"]),
                         ("A", "D", 0, 7.5, 2))
        self.assertEqual(pr["target_category"], "low")
        self.assertEqual(run["calls"], 24 + 24 + 9)
        # the persona is the system prompt, re-sent on every call; the game text is the user prompt
        self.assertTrue(all(s in (COOP, FREE) for s, _ in llm.calls))
        self.assertTrue(all(COOP not in u and FREE not in u for _, u in llm.calls))
        later = [u for s, u in llm.calls if "This is period 2." in u]
        self.assertTrue(all("Period 1: you contributed" in u for u in later))

    def test_majority_ties_fail_and_no_preference_is_excluded(self):
        def resp(system, user):
            if "reduce their earnings" in user:
                return '{"punish": 0}'
            if "Ballot item" not in user:
                return '{"contribution": 5}'
            L, k = player(user), item_of(user)
            if k == "low":
                return '{"vote": "%s"}' % ("Yes" if L in "AB" else "No")           # 2-2 tie
            if k == "avg":
                return '{"vote": "%s"}' % ("Yes" if L == "A" else "No preference")  # 1-0
            return '{"vote": "no preference"}'                                  # 0-0
        run = A.run_who_to_punish(FakeLLM(resp), votes=1, periods=1)
        self.assertEqual(run["groups"][0]["rules"][0], {"low": False, "avg": True, "high": False})
        self.assertEqual(run["rules"][0]["tally"]["avg"], dict(yes=1, no=0, no_preference=3, invalid=0))
        # everybody gave 5 = the average, so every player is punishable; nobody is asked via the low rule
        self.assertEqual(run["groups"][0]["periods"][0]["n_punish_pairs"], 12)

    def _missing_d_period2(self):
        def resp(system, user):
            if "How much of your $10" in user and "This is period 2." in user and player(user) == "D":
                return "I would rather not say."
            if "reduce their earnings" in user:
                return '{"punish": 0}'
            return punisher_responder(system, user) if "Ballot item" in user else '{"contribution": 10}'
        return FakeLLM(resp)

    def test_none_answer_stays_invalid_and_stops_group(self):
        run = A.run_who_to_punish(self._missing_d_period2(), retries=2)
        g = run["groups"][0]
        self.assertEqual(g["n_periods_completed"], 1)
        self.assertFalse(g["complete"])
        self.assertFalse(run["complete"])
        self.assertEqual(g["stopped"], "invalid: contribution period 2 player D")
        rows = [r for r in run["contributions"] if r["period"] == 2 and r["label"] == "D"]
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["contribution"])                        # never $0
        self.assertFalse(rows[0]["valid"])
        self.assertEqual(rows[0]["attempts"], 3)
        self.assertEqual(run["invalid"], dict(ballots=0, contributions=1, punishments=0, total=1))
        self.assertEqual([p["period"] for p in run["periods"]], [1])

    def test_none_answer_abstain_sits_out_not_zero(self):
        llm = self._missing_d_period2()
        run = A.run_who_to_punish(llm, retries=1, on_invalid="abstain")
        p2 = run["groups"][0]["periods"][1]
        self.assertEqual(p2["sat_out"], ["D"])
        self.assertEqual(p2["contrib"], dict(A=10, B=10, C=10))
        self.assertEqual(p2["avg"], 10)                                   # D is not averaged in as 0
        self.assertIsNone(p2["earnings"]["D"])
        self.assertEqual(p2["earnings"]["A"], 12.0)                       # 0 + 0.4 * 30
        self.assertEqual(len(run["groups"][0]["periods"]), 6)
        self.assertEqual(run["invalid"]["contributions"], 1)
        nxt = [u for s, u in llm.calls if "This is period 3." in u and player(u) == "D"]
        self.assertTrue(nxt and "sat out" in nxt[0])

    def test_none_answer_raise(self):
        with self.assertRaises(A.InvalidAnswer):
            A.run_who_to_punish(self._missing_d_period2(), on_invalid="raise")

    def test_out_of_range_is_invalid_not_clamped(self):
        def resp(system, user):
            return '{"vote": "No"}' if "Ballot item" in user else '{"contribution": 15}'
        run = A.run_who_to_punish(FakeLLM(resp), retries=0)
        self.assertEqual(run["invalid"]["contributions"], 4)
        self.assertTrue(all(r["contribution"] is None for r in run["contributions"]))
        self.assertEqual(run["periods"], [])

    def test_retry_with_feedback_recovers(self):
        def resp(system, user):
            if "Ballot item" in user:
                return '{"vote": "No"}'
            return '{"contribution": 3}' if "previous reply was not valid" in user else "three dollars maybe"
        run = A.run_who_to_punish(FakeLLM(resp), votes=1, periods=1)
        self.assertTrue(all(r["valid"] and r["attempts"] == 2 for r in run["contributions"]))
        self.assertEqual(run["invalid"]["total"], 0)
        self.assertEqual(run["calls"], 12 + 8)

    def test_groups_samples_ids_and_replay(self):
        seen = []

        class Rec(FakeLLM):
            def map(self, prompts, *, max_tokens=256, parallel=4, **kw):
                seen.append(kw.get("sample"))
                return super().map(prompts, max_tokens=max_tokens, parallel=parallel, **kw)

        personas = [A.SocietyAgent(f"P{i}", COOP if i % 4 else FREE, traits={"i": i}) for i in range(8)]
        a = A.run_who_to_punish(Rec(punisher_responder), personas, groups=2, seed=3)
        self.assertEqual(set(seen), {3 * 1_000_003, 3 * 1_000_003 + 1})
        self.assertEqual(a["groups"][1]["agent_ids"], dict(A="P4", B="P5", C="P6", D="P7"))
        self.assertEqual(a["agents"][5]["traits"], {"i": 5})
        b = A.run_who_to_punish(FakeLLM(punisher_responder), personas, groups=2, seed=3)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        seen.clear()
        A.run_who_to_punish(Rec(zero_responder), sample_key=False, votes=1, periods=1)
        self.assertEqual(set(seen), {None})

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            A.run_who_to_punish(FakeLLM(zero_responder), ["x"] * 3)
        with self.assertRaises(ValueError):
            A.run_who_to_punish(FakeLLM(zero_responder), on_invalid="impute")
        with self.assertRaises(ValueError):
            A.run_who_to_punish(FakeLLM(zero_responder), [A.SocietyAgent("same", "")] * 4)

    def test_call_cap_stops_cleanly(self):
        run = A.run_who_to_punish(FakeLLM(zero_responder), max_calls=20)
        self.assertEqual(run["calls"], 12 + 4 + 4)                        # period 3 would exceed the cap
        self.assertEqual(run["groups"][0]["n_periods_completed"], 2)
        self.assertTrue(run["stopped"].startswith("budget"))
        self.assertFalse(run["complete"])
        self.assertEqual(run["groups"][0]["votes"], [False])


class TestRepeatedPD(unittest.TestCase):
    def test_tit_for_tat_vs_defector(self):
        def resp(system, user):
            if system == "always defect":
                return '{"move": "Defect"}'
            last = re.findall(r"your opponent chose (\w+)\.", user)
            return json.dumps({"move": last[-1] if last else "Cooperate"})
        run = A.run_repeated_pd(FakeLLM(resp), ["tit for tat", "always defect"])
        p = run["pairs"][0]
        self.assertEqual(p["moves_a"], ["Cooperate"] + ["Defect"] * 4)
        self.assertEqual(p["moves_b"], ["Defect"] * 5)
        self.assertEqual((p["score_a"], p["score_b"]), (0 + 4 * 1, 5 + 4 * 1))
        self.assertEqual(run["cooperation_by_round"][1], 0.5)
        self.assertEqual(run["cooperation_by_round"][2], 0.0)
        self.assertEqual(run["calls"], 10)
        self.assertTrue(run["complete"])

    def test_invalid_move_stops_pair_unscored(self):
        def resp(system, user):
            return "hmm" if "round 3 of" in user and "Player B" in user else '{"move": "cooperate"}'
        run = A.run_repeated_pd(FakeLLM(resp), retries=1)
        p = run["pairs"][0]
        self.assertEqual(p["rounds_completed"], 2)
        self.assertEqual(p["score_a"], 6)
        self.assertIn("invalid", p["stopped"])
        r3 = [m for m in run["moves"] if m["round"] == 3]
        self.assertTrue(all(m["payoff"] is None for m in r3))
        self.assertEqual(run["invalid"]["moves"], 1)


class TestTrust(unittest.TestCase):
    def test_hand_payoffs(self):
        def resp(system, user):
            return '{"sent": 50}' if "How much do you send?" in user else '{"returned": 60}'
        run = A.run_trust_interaction(FakeLLM(resp), pairs=2)
        g = run["games"][0]
        self.assertEqual((g["sent"], g["received"], g["returned"]), (50, 150, 60))
        self.assertEqual((g["investor_payoff"], g["trustee_payoff"]), (110, 90))
        self.assertEqual(run["summary"]["n_valid"], 2)
        self.assertAlmostEqual(run["summary"]["mean_return_share"], 0.4)
        self.assertTrue(run["complete"])

    def test_return_above_received_is_invalid(self):
        def resp(system, user):
            return '{"sent": 10}' if "How much do you send?" in user else '{"returned": 999}'
        run = A.run_trust_interaction(FakeLLM(resp), retries=1)
        g = run["games"][0]
        self.assertFalse(g["valid_returned"])
        self.assertIsNone(g["returned"])
        self.assertIsNone(g["investor_payoff"])
        self.assertEqual(run["invalid"]["returned"], 1)
        self.assertFalse(run["complete"])


if __name__ == "__main__":
    unittest.main()
