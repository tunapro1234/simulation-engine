"""sim.robustness: option-order mapping, retest independence (sample keyword), framing axis, ratio and card row."""
import json
import re
import unittest

import numpy as np

from simengine.llm.fake import FakeLLM
from simengine.sim import FRAMINGS_HIDDEN, schema_responder, w1_ordinal
from simengine.types import LLMResponse
from simengine.sim.robustness import mean_position, robustness, robustness_frame, tvd

OPTS = ["Strongly agree", "Agree", "Disagree", "Strongly disagree"]
QS = [{"id": "q1", "text": "Is the city safe?", "options": OPTS},
      {"id": "q2", "text": "Is transit good?", "options": OPTS}]
SEGS = {"A": "young renters", "B": "older owners"}
TRUTH = {  # group description -> question text -> option label -> weight
    "young renters": {"Is the city safe?": [1, 2, 3, 4], "Is transit good?": [4, 3, 2, 1]},
    "older owners": {"Is the city safe?": [4, 3, 2, 1], "Is transit good?": [2, 2, 3, 3]},
    "adults": {"Is the city safe?": [2, 3, 3, 2], "Is transit good?": [3, 3, 2, 2]},
}


class SimFake(FakeLLM):
    """Reads group, question and option order from the distribution prompt and answers from TRUTH (by option
    label, so the true answer does not depend on order). primacy adds weight to whichever option is listed first;
    noise perturbs the answer by the `sample` keyword (sample 0 / absent = no perturbation)."""

    def __init__(self, primacy=0.0, noise=0.0):
        super().__init__()
        self.primacy, self.noise, self.samples = primacy, noise, []

    def complete(self, system, user, *, max_tokens=256, **kw):
        self.calls.append((system, user))
        self.samples.append(kw.get("sample"))
        text = self._answer(user, kw.get("sample") or 0)
        return LLMResponse(text=text, model=self.model, input_tokens=1, output_tokens=1)

    def _answer(self, user, sample):
        group = re.search(r"(?:who are|Consider) (.+?)\.", user).group(1)
        stem = re.search(r"^Question: (.+)$", user, re.M).group(1)
        labels = re.findall(r"^\d+\. (.+)$", user.split("Answer options:\n", 1)[1], re.M)
        w = np.array([TRUTH[group][stem][OPTS.index(x)] for x in labels], float)
        w[0] += self.primacy
        if sample and self.noise:
            w = w * (1 + self.noise * np.random.default_rng(sample * 1000 + len(user)).random(len(w)))
        return json.dumps({str(j + 1): round(100 * x / w.sum(), 6) for j, x in enumerate(w)})


def truth_vec(group, qi):
    w = np.array(TRUTH[group][QS[qi]["text"]], float)
    return w / w.sum()


class TestRobustness(unittest.TestCase):
    def test_order_maps_back_without_bias(self):
        rb = robustness(SimFake(), SEGS, QS, axes=("order",), n_boot=0)
        o = rb["runs"]["order"]
        for s, g in SEGS.items():
            for qi, q in enumerate(QS):
                np.testing.assert_allclose(o["dist"][s][q["id"]], truth_vec(g, qi), atol=1e-6)
                np.testing.assert_allclose(rb["runs"]["base"]["dist"][s][q["id"]], truth_vec(g, qi), atol=1e-6)
        row = rb["conditions"][0]
        self.assertEqual(row["condition"], "order")
        self.assertAlmostEqual(row["w1"], 0.0, places=6)
        self.assertAlmostEqual(row["shift"], 0.0, places=6)
        self.assertAlmostEqual(rb["ratio"], 0.0, places=6)
        self.assertTrue(o["reversed"])

    def test_primacy_gives_negative_shift_and_ratio(self):
        llm = SimFake(primacy=3.0)
        rb = robustness(llm, SEGS, QS, axes=("order",), n_boot=200)
        self.assertTrue(any("1. Strongly disagree" in u for _, u in llm.calls))
        row = rb["conditions"][0]
        self.assertLess(row["shift"], 0)                 # pulled toward the option listed first
        # hand computation: base and reversed vectors from the fake's rule
        cond, sig = [], []
        for qi, q in enumerate(QS):
            c_q, s_q = [], []
            for g in SEGS.values():
                w = np.array(TRUTH[g][q["text"]], float)
                a = w.copy(); a[0] += 3.0
                b = w.copy(); b[-1] += 3.0                # reversed: original last option listed first
                c_q.append(w1_ordinal(a / a.sum(), b / b.sum()))
                p = np.array(TRUTH["adults"][q["text"]], float); p[0] += 3.0
                s_q.append(w1_ordinal(a / a.sum(), p / p.sum()))
            cond.append(np.mean(c_q)); sig.append(np.mean(s_q))
        self.assertAlmostEqual(row["w1"], np.mean(cond), places=5)
        self.assertAlmostEqual(rb["signal"]["w1"], np.mean(sig), places=5)
        self.assertAlmostEqual(rb["ratio"], np.mean(cond) / np.mean(sig), places=5)
        self.assertEqual(rb["worst"], "order")
        card = rb["card"]
        self.assertEqual(card["tur"], "robustness")
        self.assertAlmostEqual(card["deger"], rb["ratio"], places=3)
        self.assertIn("sıra kayması", card["not"])

    def test_retest_uses_new_sample(self):
        llm = SimFake()
        rb = robustness(llm, SEGS, QS, axes=("retest",), retests=2, signal="mean", n_boot=0)
        n = len(SEGS) * len(QS)
        self.assertEqual(llm.samples, [None] * n + [1] * n + [2] * n)
        self.assertEqual([r["condition"] for r in rb["conditions"]], ["retest:1", "retest:2"])
        self.assertAlmostEqual(rb["conditions"][0]["w1"], 0.0, places=6)   # deterministic model: no retest noise
        llm = SimFake(noise=1.0)
        rb = robustness(llm, SEGS, QS, axes=("retest",), ask_kw={"sample": 5}, signal="mean", n_boot=0)
        self.assertEqual(llm.samples, [5] * n + [6] * n)
        self.assertGreater(rb["conditions"][0]["w1"], 0.0)
        self.assertEqual(rb["calls"], 2 * n)                                 # signal="mean": no population call

    def test_framing_axis_and_given_signal(self):
        fr = {"h1": FRAMINGS_HIDDEN["h1"], "h2": FRAMINGS_HIDDEN["h2"]}
        pop = {"q1": [0.25] * 4, "q2": [0.25] * 4}
        rb = robustness(SimFake(), SEGS, QS, axes=("framing",), framings=fr, signal={"population": pop}, n_boot=0)
        self.assertEqual(rb["conditions"][0]["condition"], "framing:h2")
        self.assertAlmostEqual(rb["conditions"][0]["w1"], 0.0, places=6)  # the fake ignores wording
        self.assertEqual(rb["signal"]["by"], "given")
        want = np.mean([w1_ordinal(truth_vec(g, qi), pop[q["id"]]) for qi, q in enumerate(QS) for g in SEGS.values()])
        self.assertAlmostEqual(rb["signal"]["w1"], want, places=5)
        with self.assertRaises(ValueError):
            robustness(SimFake(), SEGS, QS, axes=("framing",), framings={"h1": FRAMINGS_HIDDEN["h1"]})
        with self.assertRaises(ValueError):
            robustness(SimFake(), SEGS, QS, axes=("wording",))

    def test_all_axes_schema_responder_frame_and_nominal(self):
        rb = robustness(FakeLLM(schema_responder()), SEGS, QS, n_boot=50)
        self.assertEqual([r["axis"] for r in rb["conditions"]], ["retest", "framing", "framing", "order"])
        self.assertEqual(rb["calls"], (1 + 1 + 2 + 1) * len(SEGS) * len(QS) + len(QS))
        df = robustness_frame(rb)
        self.assertEqual(len(df), 4 * len(SEGS) * len(QS))   # 4 conditions
        self.assertIn("signal", df.columns)
        self.assertFalse(df["signal"].isna().any())
        json.dumps(rb)
        rn = robustness(SimFake(primacy=2.0), SEGS, QS, axes=("order",), nominal=True, n_boot=0)
        self.assertEqual(rn["metric"], "TVD")
        self.assertNotIn("shift", rn["conditions"][0])
        self.assertGreater(rn["conditions"][0]["w1"], 0)

    def test_helpers(self):
        self.assertAlmostEqual(tvd([1, 0], [0, 1]), 1.0)
        self.assertAlmostEqual(mean_position([0, 0, 1]), 1.0)
        self.assertAlmostEqual(mean_position([1, 1]), 0.5)


if __name__ == "__main__":
    unittest.main()
