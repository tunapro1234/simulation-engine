"""simengine.sim.constrained: schema-validated answers with retry feedback (port of replicant FormController)."""
import json
import math
import re
import unittest

from simengine.llm import FakeLLM
from simengine.sim import ConstrainedResult, FieldSpec, ask_constrained, completion_summary, validate_answers
from simengine.sim.constrained import JSON_ONLY, build_user_prompt


def scripted(*replies):
    """FakeLLM responder that returns the given replies in order (the last one repeats)."""
    it = list(replies)
    state = {"i": 0}

    def respond(system, user):
        r = it[min(state["i"], len(it) - 1)]
        state["i"] += 1
        return r(system, user) if callable(r) else r

    return respond


CONTRIB = FieldSpec("contribution", "int", min=0, max=20, label="Tokens to the group account")
VOTE = FieldSpec("vote", "choice", choices=[("1", "Yes"), ("0", "No")])


class TestFieldSpec(unittest.TestCase):
    def test_choice_forms(self):
        self.assertEqual(FieldSpec("a", "choice", choices=["x", "y"]).choices, (("x", "x"), ("y", "y")))
        self.assertEqual(FieldSpec("a", "choice", choices={1: "Yes"}).choices, (("1", "Yes"),))
        self.assertEqual(VOTE.displays, ["Yes", "No"])

    def test_bad_specs(self):
        with self.assertRaises(ValueError):
            FieldSpec("a", "bool")
        with self.assertRaises(ValueError):
            FieldSpec("a", "choice")
        with self.assertRaises(ValueError):
            FieldSpec("a", "int", min=5, max=1)
        with self.assertRaises(ValueError):
            FieldSpec("a", "int", choices=["x"])


class TestValidate(unittest.TestCase):
    def test_missing_values(self):
        for v in (None, "", "  ", "nan", float("nan")):
            clean, errs = validate_answers({"contribution": v}, [CONTRIB])
            self.assertEqual(clean, {})
            self.assertEqual(errs, ["Missing required field: contribution"])
        clean, errs = validate_answers({}, [CONTRIB, VOTE])
        self.assertEqual(len(errs), 2)

    def test_choice_value_display_case(self):
        for v, want in (("1", "1"), (1, "1"), (1.0, "1"), ("Yes", "1"), ("no", "0"), (" YES ", "1")):
            clean, errs = validate_answers({"vote": v}, [VOTE])
            self.assertEqual((clean, errs), ({"vote": want}, []), v)
        clean, errs = validate_answers({"vote": "maybe"}, [VOTE])
        self.assertEqual(clean, {})
        self.assertIn("is not a valid choice", errs[0])
        self.assertIn("['Yes', 'No']", errs[0])

    def test_no_substring_salvage(self):
        # replicant _try_direct_value would read "yes" inside this text; here it is an error
        clean, errs = validate_answers({"vote": "no, yes I think"}, [VOTE])
        self.assertEqual(clean, {})
        clean, errs = validate_answers({"contribution": "about 12 tokens"}, [CONTRIB])
        self.assertEqual(clean, {})
        self.assertIn("is not a number", errs[0])

    def test_numeric_bounds(self):
        self.assertEqual(validate_answers({"contribution": "12"}, [CONTRIB]), ({"contribution": 12}, []))
        self.assertEqual(validate_answers({"contribution": 20.0}, [CONTRIB]), ({"contribution": 20}, []))
        self.assertEqual(validate_answers({"contribution": 0}, [CONTRIB]), ({"contribution": 0}, []))
        _, errs = validate_answers({"contribution": 25}, [CONTRIB])
        self.assertEqual(errs, ["contribution: 25 is above maximum 20"])
        _, errs = validate_answers({"contribution": -1}, [CONTRIB])
        self.assertIn("below minimum 0", errs[0])
        _, errs = validate_answers({"contribution": 3.5}, [CONTRIB])
        self.assertIn("not a whole number", errs[0])
        _, errs = validate_answers({"contribution": True}, [CONTRIB])
        self.assertIn("not a number", errs[0])
        _, errs = validate_answers({"contribution": "inf"}, [CONTRIB])
        self.assertIn("finite", errs[0])
        f = FieldSpec("p", "float", min=0, max=1)
        self.assertEqual(validate_answers({"p": "0.25"}, [f]), ({"p": 0.25}, []))
        self.assertIsInstance(validate_answers({"p": 1}, [f])[0]["p"], float)

    def test_text_and_dict_fields(self):
        t = FieldSpec("why", "text")
        self.assertEqual(validate_answers({"why": "  fair share "}, [t]), ({"why": "fair share"}, []))
        self.assertEqual(validate_answers({"why": {"a": 1}}, [t])[0], {})
        clean, errs = validate_answers({"why": "ok"}, [{"name": "why"}])
        self.assertEqual(clean, {"why": "ok"})
        self.assertEqual(validate_answers("not a dict", [t])[1], ["Missing required field: why"])


class TestAskConstrained(unittest.TestCase):
    def test_valid_first_answer_one_attempt(self):
        llm = FakeLLM(scripted('{"contribution": 12, "vote": "Yes"}'))
        res = ask_constrained(llm, "You are a careful saver.", "Public goods game, round 1.", [CONTRIB, VOTE])
        self.assertIsInstance(res, ConstrainedResult)
        self.assertTrue(res.completed)
        self.assertEqual(res.attempts, 1)
        self.assertEqual(res.values, {"contribution": 12, "vote": "1"})
        self.assertEqual(res.failures, [])
        self.assertEqual(res.mode, "batch")
        self.assertEqual(res.usage.calls, 1)
        system, user = llm.calls[0]
        self.assertTrue(system.startswith("You are a careful saver."))
        self.assertTrue(system.endswith(JSON_ONLY))
        self.assertIn("Public goods game, round 1.", user)
        self.assertIn('(integer from 0 to 20)', user)
        self.assertIn('(choose exactly one of: "Yes", "No")', user)
        self.assertNotIn("Your previous response", user)

    def test_out_of_range_then_valid_feeds_error_back(self):
        llm = FakeLLM(scripted('{"contribution": 25}', '```json\n{"contribution": 7}\n```'))
        res = ask_constrained(llm, "persona", "game", [CONTRIB])
        self.assertTrue(res.completed)
        self.assertEqual(res.attempts, 2)
        self.assertEqual(res.values, {"contribution": 7})
        second = llm.calls[1][1]
        self.assertIn("Your previous response had errors:", second)
        self.assertIn("contribution: 25 is above maximum 20", second)
        self.assertIn("Attempt 2 of 8", second)
        self.assertNotIn("above maximum", llm.calls[0][1])
        self.assertEqual(res.trace[0]["errors"], ["contribution: 25 is above maximum 20"])

    def test_eight_empty_answers_fail_cleanly(self):
        llm = FakeLLM(scripted(""))
        res = ask_constrained(llm, "persona", "game", [CONTRIB])
        self.assertFalse(res.completed)
        self.assertFalse(res.valid)
        self.assertEqual(res.attempts, 8)
        self.assertEqual(len(llm.calls), 8)
        self.assertEqual(res.values, {})
        self.assertNotIn("contribution", res.row())
        self.assertEqual(res.failures, [{"field": "contribution", "attempt": 8,
                                         "error": "Missing required field: contribution", "last_response_len": 0}])
        self.assertEqual(res.usage.unparsed_calls, 8)
        self.assertIn("could not be parsed as JSON", llm.calls[1][1])
        # every retry prompt differs (attempt counter), so a cached backend does not replay the same answer
        self.assertEqual(len({u for _, u in llm.calls}), 8)
        json.dumps(res.as_dict())

    def test_sequential_passes_prior_answers(self):
        fields = [FieldSpec(f"q{i}", "int", min=0, max=10) for i in range(4)]

        def respond(system, user):
            name = re.search(r'^- "(q\d)"', user, re.M).group(1)
            return json.dumps({name: int(name[1]) + 1})

        llm = FakeLLM(respond)
        res = ask_constrained(llm, "persona", "page", fields)
        self.assertEqual(res.mode, "sequential")
        self.assertTrue(res.completed)
        self.assertEqual(res.values, {"q0": 1, "q1": 2, "q2": 3, "q3": 4})
        self.assertEqual(res.attempts, 4)
        users = [u for _, u in llm.calls]
        self.assertNotIn("Your previous answers on this page", users[0])
        self.assertIn('"q0": 1', users[1])
        self.assertIn('"q2": 3', users[3])
        self.assertEqual(users[3].count('- "q'), 1)     # one field per call
        # 3 fields stay in one call; the threshold is configurable
        self.assertEqual(ask_constrained(FakeLLM(scripted('{"q0":1,"q1":1,"q2":1}')), "", "p", fields[:3]).mode,
                         "batch")
        self.assertEqual(ask_constrained(FakeLLM(respond), "", "p", fields, sequential_threshold=None).mode, "batch")

    def test_sequential_stops_at_failed_field(self):
        fields = [FieldSpec(f"q{i}", "int", min=0, max=10) for i in range(4)]

        def respond(system, user):
            name = re.search(r'^- "(q\d)"', user, re.M).group(1)
            return "" if name == "q1" else json.dumps({name: 5})

        llm = FakeLLM(respond)
        res = ask_constrained(llm, "persona", "page", fields, max_retries=3)
        self.assertFalse(res.completed)
        self.assertEqual(res.values, {"q0": 5})
        self.assertEqual(res.attempts, 1 + 3)
        self.assertEqual([(f["field"], f["attempt"]) for f in res.failures], [("q1", 3), ("q2", 0), ("q3", 0)])
        res2 = ask_constrained(FakeLLM(respond), "persona", "page", fields, max_retries=3, stop_on_failure=False)
        self.assertEqual(res2.values, {"q0": 5, "q2": 5, "q3": 5})
        self.assertEqual([f["field"] for f in res2.failures], ["q1"])

    def test_single_key_salvage_but_no_text_salvage(self):
        llm = FakeLLM(scripted('{"answer": 4}'))
        res = ask_constrained(llm, "", "page", [CONTRIB])
        self.assertEqual(res.values, {"contribution": 4})
        llm = FakeLLM(scripted("I would contribute 4 tokens."))
        res = ask_constrained(llm, "", "page", [CONTRIB], max_retries=2)
        self.assertFalse(res.completed)
        self.assertEqual(res.values, {})

    def test_choice_display_text_resolves(self):
        llm = FakeLLM(scripted('{"vote": "yes"}'))
        res = ask_constrained(llm, "persona", "Vote on the rule.", [VOTE])
        self.assertEqual(res.values, {"vote": "1"})
        self.assertEqual(res.attempts, 1)

    def test_batch_partial_keeps_only_valid_fields(self):
        llm = FakeLLM(scripted('{"contribution": 5, "vote": "maybe"}'))
        res = ask_constrained(llm, "", "page", [CONTRIB, VOTE], max_retries=2)
        self.assertFalse(res.completed)
        self.assertEqual(res.values, {"contribution": 5})
        self.assertEqual([f["field"] for f in res.failures], ["vote"])
        self.assertEqual(res.failures[0]["attempt"], 2)

    def test_kwargs_meta_and_suffix(self):
        seen = {}

        class Spy(FakeLLM):
            def complete(self, system, user, *, max_tokens=256, **kw):
                seen.update(kw, max_tokens=max_tokens)
                return super().complete(system, user, max_tokens=max_tokens)

        llm = Spy(scripted('{"contribution": 3}'))
        res = ask_constrained(llm, "", "page", [CONTRIB], max_tokens=4096, effort="low", system_suffix="",
                              meta={"condition": "c1", "agent_id": "a1", "seed": 0})
        self.assertEqual(seen, {"effort": "low", "max_tokens": 4096})
        self.assertEqual(llm.calls[0][0], "")
        self.assertEqual(res.max_tokens, 4096)
        row = res.row()
        self.assertEqual((row["condition"], row["agent_id"], row["contribution"], row["valid"]), ("c1", "a1", 3, True))
        with self.assertRaises(ValueError):
            ask_constrained(llm, "", "page", [])

    def test_prompt_shape(self):
        u = build_user_prompt("ctx", [CONTRIB, FieldSpec("x", "float", min=0)], "fix it", {"a": 1})
        self.assertTrue(u.startswith("Page context:\nctx\n\nYour previous answers on this page:"))
        self.assertIn("(number, min: 0)", u)
        self.assertTrue(u.endswith('Respond: {"contribution": ..., "x": ...}'))


class TestCompletionSummary(unittest.TestCase):
    def test_counts(self):
        fields = [FieldSpec("a", "int", min=0, max=5), FieldSpec("b", "int", min=0, max=5),
                  FieldSpec("c", "int", min=0, max=5), FieldSpec("d", "int", min=0, max=5)]

        def responder(bad):
            def respond(system, user):
                name = re.search(r'^- "(\w)"', user, re.M).group(1)
                return "" if name == bad else json.dumps({name: 1})
            return respond

        results = []
        for i in range(3):
            results.append(ask_constrained(FakeLLM(responder(None)), "", "p", fields, meta={"condition": "default"}))
        results.append(ask_constrained(FakeLLM(responder("b")), "", "p", fields, max_retries=2,
                                       meta={"condition": "coop"}))
        results.append(ask_constrained(FakeLLM(responder("c")), "", "p", fields, max_retries=2,
                                       meta={"condition": "coop"}).as_dict())
        results.append(ask_constrained(FakeLLM(responder(None)), "", "p", fields, meta={"condition": "coop"}))
        df = completion_summary(results, by="condition", planned={"coop": 5})
        self.assertEqual(list(df.columns[:6]),
                         ["condition", "n_planned", "n_results", "n_completed", "completion_rate", "attempts"])
        d = df.set_index("condition")
        self.assertEqual(d.loc["default", "n_planned"], 3)
        self.assertEqual(d.loc["default", "n_completed"], 3)
        self.assertEqual(d.loc["default", "completion_rate"], 1.0)
        self.assertEqual(d.loc["default", "attempts"], 12)
        self.assertEqual(d.loc["coop", "n_planned"], 5)
        self.assertEqual(d.loc["coop", "n_results"], 3)
        self.assertEqual(d.loc["coop", "n_completed"], 1)
        self.assertAlmostEqual(d.loc["coop", "completion_rate"], 0.2)
        self.assertEqual(d.loc["coop", "attempts"], (1 + 2) + (2 + 2) + 4)
        self.assertEqual(d.loc["coop", "failed_b"], 1)
        self.assertEqual(d.loc["coop", "missing_b"], 1)
        self.assertEqual(d.loc["coop", "failed_c"], 1)
        self.assertEqual(d.loc["coop", "missing_c"], 2)       # failed once, not reached once
        self.assertEqual(d.loc["coop", "missing_d"], 2)
        self.assertEqual(d.loc["coop", "failed_d"], 0)
        self.assertEqual(d.loc["default", "missing_a"], 0)

    def test_grouping_forms(self):
        ok = ask_constrained(FakeLLM(scripted('{"vote": "No"}')), "", "p", [VOTE])
        bad = ask_constrained(FakeLLM(scripted("")), "", "p", [VOTE], max_retries=1)
        df = completion_summary([ok, bad], by=None)
        self.assertEqual(df.iloc[0]["group"], "all")
        self.assertEqual((df.iloc[0]["n_completed"], df.iloc[0]["failed_vote"]), (1, 1))
        df = completion_summary({"x": [ok], "y": [bad, bad]}, planned=4)
        self.assertEqual(df.set_index("condition").loc["y", "missing_vote"], 2)
        self.assertEqual(df.set_index("condition").loc["y", "n_planned"], 4)
        self.assertTrue(math.isnan(completion_summary([ok], planned=0).iloc[0]["completion_rate"]))


if __name__ == "__main__":
    unittest.main()
