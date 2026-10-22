"""sim.ab_test lift vs a reference arm, pairwise win matrix, choice_disagrees, dict variants; ab_many vote
accounting and schema_responder coverage. Offline with FakeLLM."""
import json
import re
import unittest
import warnings

from simengine.llm import FakeLLM
from simengine.sim import ab_many, ab_test, confidence_check, schema_responder
from simengine.sim.ab import ab_lift_frame
from simengine.sim.panel import Usage

BASE = {"control": 4, "C1": 7, "C2": 5}


def personas(n=12):
    return [{"id": f"u{i}", "name": f"N{i}", "text": f"age {20 + i}, city {i}", "segment": "A" if i % 2 else "B"}
            for i in range(n)]


def gap_responder(base=BASE, choice=None):
    """Panel answers with a known score per variant (+-1 by persona parity); `choice` forces the pick."""
    def respond(system, user):
        pids = re.findall(r"^### (\S+)\s*$", user, re.M)
        labels = re.findall(r'^\s+"score_([^"]+)":', user, re.M)
        out = []
        for k, pid in enumerate(pids):
            jit = 1 if int(pid[1:]) % 2 else 0
            sc = {lab: base.get(lab, 5) + jit for lab in labels}
            out.append({"id": pid, **{f"score_{lab}": v for lab, v in sc.items()},
                        "choice": choice or max(sc, key=sc.get), "reason": "fits my budget"})
        return json.dumps(out)
    return respond


class TestLift(unittest.TestCase):
    def test_lift_vs_reference(self):
        variants = [{"id": k, "text": f"creative {k}"} for k in BASE]
        r = ab_test(personas(), variants, llm=FakeLLM(gap_responder()), B=200, controls=[], reference="control")
        json.dumps(r)
        self.assertEqual(set(r["lift"]), {"C1", "C2"})
        c1 = r["lift"]["C1"]
        self.assertAlmostEqual(c1["abs"], 3.0)
        self.assertGreater(c1["abs_ci"][0], 0)
        self.assertGreater(c1["rel"], 0)
        self.assertGreater(c1["rel_ci"][0], 0)
        self.assertLess(c1["p"], 0.05)
        self.assertEqual(c1["prob_better"], 1.0)
        f = ab_lift_frame(r)
        self.assertEqual(list(f.variant), ["C1", "C2"])
        self.assertTrue((f.abs_low > 0).all())
        # pairwise matrix is defined for the control, unlike win_prob
        pw = r["pairwise_win_prob"]
        self.assertAlmostEqual(pw["C1"]["control"], 1.0)
        self.assertAlmostEqual(pw["control"]["C1"], 0.0)
        self.assertAlmostEqual(pw["C2"]["C1"] + pw["C1"]["C2"], 1.0)
        self.assertEqual(r["tendency"], r["winner"])

    def test_no_reference_keeps_result(self):
        r = ab_test(personas(4), ["red", "blue"], llm=FakeLLM(schema_responder(1)), B=20, controls=[])
        self.assertNotIn("lift", r)
        self.assertTrue(ab_lift_frame(r).empty)
        with self.assertRaises(ValueError):
            ab_test(personas(4), ["red", "blue"], llm=FakeLLM(schema_responder(1)), B=20, reference="Z")


class TestVariants(unittest.TestCase):
    def test_dict_variants_reach_prompt(self):
        llm = FakeLLM(schema_responder(2))
        r = ab_test(personas(4), {"A": "free delivery tonight", "B": "15% off"}, llm=llm, B=20, controls=[])
        self.assertEqual(r["variants"], [{"id": "A", "text": "free delivery tonight"}, {"id": "B", "text": "15% off"}])
        self.assertTrue(all("free delivery tonight" in u and "15% off" in u for _, u in llm.calls))

    def test_unsupported_type(self):
        for bad in ("AB", {"A", "B"}, 3):
            with self.assertRaises(TypeError):
                ab_test(personas(4), bad, llm=FakeLLM(schema_responder()), B=10)


class TestChoiceDisagrees(unittest.TestCase):
    def test_reason_from_summaries(self):
        s = {"persona": {"winner": "A", "choice_winner": "B", "sign_test": {"p": 0.01},
                         "variants": {"A": {"share": 0.1}, "B": {"share": 0.9}}}}
        c = confidence_check(s)
        self.assertTrue(c["low"])
        self.assertEqual([x["code"] for x in c["reasons"]], ["choice_disagrees"])

    def test_fires_in_ab_test(self):
        # C1 scores highest but every persona picks control
        r = ab_test(personas(), [{"id": k, "text": k} for k in BASE], llm=FakeLLM(gap_responder(choice="control")),
                    B=50, controls=[])
        self.assertEqual(r["overall"]["choice_winner"], "control")
        self.assertIn("choice_disagrees", [x["code"] for x in r["confidence"]["reasons"]])


class TestABManyVotes(unittest.TestCase):
    TESTS = [{"test_id": f"t{i}", "a": f"head A{i}", "b": f"head B{i}"} for i in range(3)]

    def test_schema_responder_votes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            df = ab_many(personas(4), self.TESTS, llm=FakeLLM(schema_responder()), tests_per_call=2)
        self.assertTrue((df.n_votes == 4).all())
        self.assertEqual(df.attrs["warnings"], [])

    def test_no_votes_warns_and_counts(self):
        u = Usage()
        with self.assertWarns(RuntimeWarning):
            df = ab_many(personas(4), self.TESTS, llm=FakeLLM(lambda s, x: "[]"), tests_per_call=2, usage=u)
        self.assertTrue((df.n_votes == 0).all())
        self.assertEqual(u.missing, 12)
        self.assertTrue(df.attrs["warnings"])
        u = Usage()
        with self.assertWarns(RuntimeWarning):
            ab_many(personas(4), self.TESTS, llm=FakeLLM(lambda s, x: "no idea"), tests_per_call=2, usage=u)
        self.assertEqual(u.unparsed_calls, 2)


if __name__ == "__main__":
    unittest.main()
