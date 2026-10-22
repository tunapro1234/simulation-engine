"""evidence.ezber.recall_probe: one-call recall probe, in-JSON refusal detection, best-estimate retry (D6 #6)."""
import json
import unittest

from simengine.evidence import ezber
from simengine.llm import FakeLLM

GROUPS = ["18-29", "30-44", "45-64", "65+"]
MEASURES = ["support", "oppose"]
TRUTH = {"18-29": {"support": 61, "oppose": 30}, "30-44": {"support": 55, "oppose": 37},
         "45-64": {"support": 48, "oppose": 45}, "65+": {"support": 40, "oppose": 52}}


def _table(shift=0.0, scale=1.0):
    return {g: {m: v * scale + shift for m, v in r.items()} for g, r in TRUTH.items()}


def _llm(first: str, retry: str | None = None):
    def resp(system, user):
        return retry if retry is not None and "best estimate" in user else first
    return FakeLLM(resp)


class RecallRefusalTest(unittest.TestCase):
    def test_kinds(self):
        k = lambda t: ezber.recall_refusal(t, GROUPS, MEASURES)[0]
        self.assertEqual(k('{"remember": false, "results": {}}'), "remember_false")
        self.assertEqual(k('{"results": {"18-29": {}, "30-44": {}}}'), "empty")
        self.assertEqual(k('{"remember": true, "results": {"18-29": {"support": null, "oppose": null}}}'), "all_null")
        self.assertEqual(k("I'm sorry, but I can't recall those figures."), "text")
        self.assertEqual(k("The numbers were roughly as follows"), "unparsed")
        self.assertEqual(k(""), "empty_reply")
        self.assertIsNone(k(json.dumps({"remember": True, "results": _table()})))

    def test_shapes(self):
        rows = [{"group": g, "measure": m, "value": v} for g, r in TRUTH.items() for m, v in r.items()]
        kind, cells = ezber.recall_refusal("```json\n" + json.dumps(rows) + "\n```", GROUPS, MEASURES)
        self.assertIsNone(kind)
        self.assertEqual(cells[("65+", "oppose")], 52)
        measure_major = {m: {g: TRUTH[g][m] for g in GROUPS} for m in MEASURES}
        kind, cells = ezber.recall_refusal(json.dumps(measure_major), GROUPS, MEASURES)
        self.assertEqual(len(cells), 8)
        # remember false with numbers: classified, numbers kept (estimates, not recall)
        kind, cells = ezber.recall_refusal(json.dumps({"remember": "false", "results": _table()}), GROUPS, MEASURES)
        self.assertEqual(kind, "remember_false")
        self.assertEqual(len(cells), 8)


class RecallProbeTest(unittest.TestCase):
    def test_in_json_refusal_then_retry(self):
        first = json.dumps({"remember": False, "results": {g: {} for g in GROUPS}})
        llm = _llm(first, json.dumps({"remember": False, "results": _table(shift=2)}))
        r = ezber.recall_probe(llm, "Survey X 2024", GROUPS, MEASURES, truth=TRUTH, tag="probe")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(r["calls"], 2)
        self.assertFalse(r["recalls"])
        self.assertEqual(r["refusal"], "remember_false")
        self.assertEqual(r["used"], "retry")
        self.assertEqual(r["raw"], first)
        self.assertEqual(r["values"]["18-29"]["support"], 63)
        self.assertAlmostEqual(r["mae"], 2.0)
        self.assertEqual(r["per_group"]["65+"], {"mae": 2.0, "n": 2})
        self.assertAlmostEqual(r["spearman"], 1.0)
        self.assertEqual(r["audit"]["durum"], "uyari")
        self.assertIn("best estimate", r["audit"]["ayrinti"])
        self.assertIn("Survey X 2024", llm.calls[0][1])
        self.assertIn("best estimate", llm.calls[1][1].lower())

    def test_no_retry(self):
        llm = _llm(json.dumps({"remember": True, "results": {g: {m: None for m in MEASURES} for g in GROUPS}}))
        r = ezber.recall_probe(llm, "Survey X", GROUPS, MEASURES, truth=TRUTH, best_estimate_retry=False)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(r["refusal"], "all_null")
        self.assertIsNone(r["used"])
        self.assertEqual(r["n_values"], 0)
        self.assertEqual(r["audit"]["durum"], "atlandi")

    def test_recalls_first_call(self):
        llm = _llm(json.dumps({"remember": True, "results": _table()}))
        r = ezber.recall_probe(llm, "Survey X", GROUPS, MEASURES, truth=TRUTH)
        self.assertEqual(len(llm.calls), 1)
        self.assertTrue(r["recalls"])
        self.assertIsNone(r["refusal"])
        self.assertEqual(r["used"], "first")
        self.assertEqual(r["mae"], 0.0)
        self.assertEqual(r["spearman_by_measure"]["support"], 1.0)

    def test_unrelated_numbers_tamam_and_evaluate_reads_audit(self):
        noise = {"18-29": {"support": 40, "oppose": 50}, "30-44": {"support": 61, "oppose": 31},
                 "45-64": {"support": 44, "oppose": 52}, "65+": {"support": 55, "oppose": 30}}
        r = ezber.recall_probe(_llm(json.dumps({"results": noise})), "Survey X", GROUPS, MEASURES, truth=TRUTH)
        self.assertEqual(r["audit"]["durum"], "tamam")
        json.dumps(r["audit"])   # JSON-serialisable: evaluate(memorization=dict) writes it to a file
        from simengine.evidence import denetim
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.json")
            with open(p, "w") as f:
                json.dump(r["audit"], f)
            self.assertEqual(denetim.ezber(p)["durum"], "tamam")

    def test_single_measure_and_callable(self):
        truth = {g: TRUTH[g]["support"] for g in GROUPS}
        r = ezber.recall_probe(lambda u: json.dumps({g: v + 1 for g, v in truth.items()}), "Survey X", GROUPS,
                               "support", truth=truth)
        self.assertEqual(r["used"], "first")
        self.assertAlmostEqual(r["mae"], 1.0)

    def test_no_truth(self):
        r = ezber.recall_probe(_llm("Sorry, I cannot recall that.", "not json either"), "Survey X", GROUPS, MEASURES)
        self.assertEqual(r["refusal"], "text")
        self.assertEqual(r["retry_refusal"], "unparsed")
        self.assertNotIn("audit", r)


if __name__ == "__main__":
    unittest.main()
