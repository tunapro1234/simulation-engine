"""simengine.sim frames are accepted by simengine.evidence.evaluate (EvalFrame contract)."""
import unittest
import warnings

import numpy as np

from simengine.evidence import evaluate
from simengine.llm import FakeLLM
from simengine.sim import (distribution_frame, predict_people, prediction_frame, schema_responder,
                           segment_distribution)


class TestFramesEvaluate(unittest.TestCase):
    def test_binary_and_distribution(self):
        rng = np.random.default_rng(0)
        llm = FakeLLM(schema_responder(0))
        people = {f"u{i}": {"age": str(20 + i)} for i in range(30)}
        items = ["A", "B", "C"]
        r = predict_people(llm, people, items, level="L1")
        truth = {u: {k: int(rng.random() < 0.3) for k in items} for u in people}
        f = prediction_frame(r, truth=truth)
        qs = [{"id": "q1", "text": "x", "options": ["a", "b", "c"]}, {"id": "q2", "text": "y", "options": ["a", "b"]}]
        d = segment_distribution(llm, {"s1": "g1", "s2": "g2", "s3": "g3"}, qs)
        fd = distribution_frame(d, truth={s: {"q1": [0.2, 0.3, 0.5], "q2": [0.5, 0.5]} for s in ("s1", "s2", "s3")})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rb = evaluate(f, title="binary", ours="L1", B=50, B_aggregate=20, multiplier=False)
            rd = evaluate(fd, title="dist", ours="llm_dist", B=50, B_aggregate=20, multiplier=False)
        self.assertIsInstance(rb, dict)
        self.assertEqual(rd["gorev"], "dagilim")


if __name__ == "__main__":
    unittest.main()
