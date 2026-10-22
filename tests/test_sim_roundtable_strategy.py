"""roundtable_strategy (one roundtable per condition, effect vs control, paired round shift) and print_events."""
import io
import json
import re
import unittest

from simengine.llm import FakeLLM
from simengine.sim import roundtable, schema_responder
from simengine.sim._stats import Emitter, print_events
from simengine.sim.roundtable import (hide_persona_fields, roundtable_conditions, roundtable_strategy,
                                      roundtable_strategy_frame)

BASE = {"Control": 40, "Discount": 70, "Story": 45}


def personas(n=6):
    return [{"id": f"u{i}", "name": f"N{i}", "age": 20 + i, "condition": "Discount" if i % 3 else "Control",
             "segment": "A" if i % 2 else "B"} for i in range(n)]


def responder(system, user):
    """Intent = condition base (+10 for segment A personas u1/u3/u5) + 5 per round after the first."""
    ids = re.findall(r"^### (\S+)\s*$", user, re.M)
    cond = next(c for c in BASE if f"offer {c}" in user)
    t = int(re.search(r"round (\d+) of", user).group(1)) - 1
    return json.dumps([{"id": p, "intent": BASE[cond] + (10 if int(p[1:]) % 2 else 0) + 5 * t * (cond == "Discount"),
                        "remark": f"{cond} ok"} for p in ids])


SCEN = {c: f"offer {c}" for c in BASE}


class TestRoundtableStrategy(unittest.TestCase):
    def test_effects_shift_ranking(self):
        llm = FakeLLM(responder)
        r = roundtable_strategy(personas(), SCEN, llm=llm, rounds=3, B=50)
        json.dumps(r)
        self.assertEqual(r["control"], "Control")
        self.assertEqual(set(r["runs"]), set(BASE))
        self.assertEqual(len(r["table"]), 3 * 3 * 3)          # condition x round x {all, A, B}
        m = {(x["condition"], x["round"], x["segment"]): x["mean"] for x in r["table"]}
        self.assertEqual(m[("Control", 0, "B")], 40)
        self.assertEqual(m[("Control", 0, "A")], 50)
        self.assertEqual(m[("Discount", 2, "all")], 85)
        e = {(x["condition"], x["round"], x["segment"]): x for x in r["effects"]}
        self.assertNotIn(("Control", 0, "all"), e)
        self.assertEqual(e[("Discount", 0, "all")]["effect"], 30)
        self.assertEqual(e[("Discount", 2, "A")]["effect"], 40)
        self.assertEqual(e[("Story", 1, "all")]["n"], 6)
        sh = {(x["condition"], x["segment"]): x for x in r["shift"]}
        self.assertEqual(sh[("Discount", "all")]["value"], 10)
        self.assertEqual(sh[("Discount", "all")]["effect_shift_value"], 10)
        self.assertEqual(sh[("Story", "B")]["value"], 0)
        self.assertNotIn("effect_shift_value", sh[("Control", "all")])
        self.assertEqual(r["ranking"], {"round0": ["Discount", "Story", "Control"],
                                        "last": ["Discount", "Story", "Control"]})
        self.assertEqual(r["usage"]["calls"], len(llm.calls))

    def test_condition_hidden_and_list_input(self):
        llm = FakeLLM(schema_responder(1))
        r = roundtable_conditions(personas(4), ["offer Control", "offer Story"], llm=llm, rounds=2, B=20,
                                  control="S1", scope="all")
        self.assertEqual([c["id"] for c in r["conditions"]], ["S1", "S2"])
        self.assertEqual(r["scope"], "all")
        self.assertTrue(r["effects"])
        self.assertTrue(all("condition" not in u and "Discount" not in u for _, u in llm.calls))
        self.assertTrue(any("age: 21" in u for _, u in llm.calls))
        r2 = roundtable_strategy(personas(4), SCEN, llm=FakeLLM(responder), rounds=1, hide_fields=(), B=20)
        self.assertEqual(r2["rounds"], 1)

    def test_no_control_and_errors(self):
        r = roundtable_strategy(personas(4), SCEN, llm=FakeLLM(responder), rounds=2, control="Nope", B=20)
        self.assertEqual(r["effects"], [])
        self.assertIsNone(r["control"])
        self.assertTrue(any("Nope" in w for w in r["warnings"]))
        with self.assertRaises(ValueError):
            roundtable_strategy(personas(), ["only one"], llm=FakeLLM(responder))
        with self.assertRaises(ValueError):
            roundtable_strategy(personas(), [{"id": "X", "text": "a"}, {"id": "X", "text": "b"}],
                                llm=FakeLLM(responder))

    def test_frame(self):
        r = roundtable_strategy(personas(), SCEN, llm=FakeLLM(responder), rounds=3, B=20)
        f = roundtable_strategy_frame(r, truth={"Discount": 72, "Story": 44})
        self.assertEqual(len(f), 3 * 3)                         # condition x {A, B, all}
        row = f[f.unit == "Discount|all"].iloc[0]
        self.assertEqual((row.truth, row.pred_round0, row.pred_round2, row["item"]), (72, 75, 85, "Discount"))
        fe = roundtable_strategy_frame(r, truth={"Discount": 25, "Story": -1}, effect=True)
        self.assertEqual(set(fe.condition), {"Discount", "Story"})
        row = fe[fe.unit == "Discount|all"].iloc[0]
        self.assertEqual((row.truth, row.pred_round0, row.pred_round2), (25, 30, 40))
        fp = roundtable_strategy_frame(r)
        self.assertEqual(len(fp), 3 * 6)                        # person rows by default
        self.assertTrue(fp.unit.str.startswith(("Control|u", "Discount|u", "Story|u")).all())
        with self.assertRaises(ValueError):
            roundtable_strategy_frame({**r, "control": None}, effect=True)

    def test_hide_persona_fields_talp_and_objects(self):
        talp = {"identity": {"persona_id": "t1", "name": "T", "condition": "X"}, "segment": "S",
                "traits": [{"label": "condition", "value": "X"}, {"label": "city", "value": "Y"}]}

        class P:
            segment, pid = "Z", "o1"

            def to_dict(self):
                return {"name": "O", "condition": "X", "hobby": "tea"}
        v = hide_persona_fields([talp, P()])
        self.assertEqual([(x.pid, x.seg) for x in v], [("t1", "S"), ("o1", "Z")])
        self.assertTrue(all("X" not in x.text() for x in v))
        self.assertIn("city: Y", v[0].text())
        self.assertIn("hobby: tea", v[1].text())


class TestPrintEvents(unittest.TestCase):
    def test_lines(self):
        buf = io.StringIO()
        em = Emitter(lambda e: print_events(e, file=buf), "survey")
        em.stage("round 0", round=0)
        em.row("u1", "A", "fine\\nreally", 55.0, round=0)
        em.done(n=3)
        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertRegex(lines[0], r"^\d\d:\d\d:\d\d \[survey\] stage round 0 round=0$")
        self.assertIn("row u1 A 55: fine\\nreally round=0", lines[1])
        self.assertTrue(lines[2].endswith("[survey] done n=3"))

    def test_as_on_event(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            roundtable([{"id": "u0", "text": "x"}], "topic", llm=FakeLLM(schema_responder(0)), rounds=1, B=10,
                       on_event=print_events)
        self.assertIn("[roundtable] done", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
