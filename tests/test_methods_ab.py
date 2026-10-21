"""A/B winner methods and LLM prior + early test Bayes, offline with FakeLLM."""
import json
import re
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.llm import FakeLLM
from simengine.methods import (ABStack, EarlyTestBayes, LengthAB, LLMDirectAB, LogitAverageAB, TfidfAB,
                               TfidfWordAB, calibrate_prior, early_lor, equivalent_multiplier, posterior_lor,
                               simulate_saving, simulate_saving_calibrated)

WORDS = ["river", "garden", "window", "letter", "bridge", "market", "winter", "silver", "forest", "engine"]
HOOK = ["shocking", "secret", "unbelievable"]


def pairs(n, seed=0):
    """A wins when it has a hook word and B does not (with 10% label noise)."""
    rs = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        ha, hb = rs.random() < 0.5, rs.random() < 0.5
        mk = lambda h: " ".join(list(rs.choice(WORDS, 4)) + ([str(rs.choice(HOOK))] if h else []))
        a, b = mk(ha), mk(hb)
        s = (ha - hb) + rs.normal(0, 0.3)
        rows.append({"text_a": a, "text_b": b, "y": int(s > 0), "group": i, "significant": 1})
    return pd.DataFrame(rows)


LINE = re.compile(r'^(\d+)\. A: "(.*?)" \| B: "(.*?)"$', re.M)


def responder(system, user):
    out = []
    for i, a, b in LINE.findall(user):
        ha, hb = any(h in a for h in HOOK), any(h in b for h in HOOK)
        out.append({"i": int(i), "p_a": 0.5 + 0.3 * (ha - hb)})
    return "```json\n" + json.dumps(out) + "\n```"


def acc(y, p):
    return float(np.mean((np.asarray(p) > 0.5) == (np.asarray(y) == 1)))


class TestAB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = pairs(500, 0), pairs(150, 1)

    def test_length_and_aliases(self):
        df = pd.DataFrame({"metin_a": ["long text here", "a"], "metin_b": ["short", "bb"]})
        np.testing.assert_allclose(LengthAB().fit().predict(df), [0.6, 0.4])

    def test_tfidf_symmetry_and_accuracy(self):
        m = TfidfAB().fit(self.train)
        p = m.predict(self.test)
        sw = self.test.rename(columns={"text_a": "text_b", "text_b": "text_a"})
        np.testing.assert_allclose(p, 1 - m.predict(sw), atol=1e-9)
        self.assertGreater(acc(self.test.y, p), 0.7)
        self.assertGreater(acc(self.test.y, TfidfWordAB().fit(self.train).predict(self.test)), 0.7)

    def test_llm_direct_with_fake(self):
        llm = FakeLLM(responder)
        m = LLMDirectAB(llm, batch=15)
        p = m.predict(self.test)
        self.assertEqual(m.coverage_, 1.0)
        self.assertEqual(m.calls_, 2 * int(np.ceil(len(self.test) / 15)))
        self.assertGreater(acc(self.test.y, p), 0.6)
        bad = LLMDirectAB(FakeLLM(lambda s, u: "no idea"), reverse=False)
        np.testing.assert_allclose(bad.predict(self.test.head(5)), 0.5)
        self.assertEqual(bad.coverage_, 0.0)

    def test_stack_learns_and_falls_back(self):
        llm = LLMDirectAB(FakeLLM(responder))
        st = ABStack(llm_method=llm, dev_n=100).fit(self.train)
        self.assertIn("held out", st.note_)
        p = st.predict(self.test)
        self.assertGreater(acc(self.test.y, p), 0.7)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            st2 = ABStack(llm_method=llm, dev_n=0).fit(self.train.head(60))
            self.assertTrue(any("stored" in str(x.message) for x in w))
        np.testing.assert_allclose(st2.w_, ABStack.W0)
        st3 = ABStack(llm_method=llm).fit(self.train, dev=self.test.head(60))
        self.assertIn("dev frame", st3.note_)
        c = st3.combine(np.array([0.7]), np.array([0.5]), np.array([False]))
        self.assertGreater(c[0], 0.5)
        la = LogitAverageAB(TfidfAB(), llm).fit(self.train)
        self.assertEqual(len(la.predict(self.test.head(10))), 10)


class TestBayesPrior(unittest.TestCase):
    def test_update_math(self):
        m, v = posterior_lor(1.0, 1.0, 0.0, 1.0)
        self.assertAlmostEqual(m, 0.5)
        self.assertAlmostEqual(v, 0.5)
        lor, se2 = early_lor(0, 100, 0, 100)
        self.assertAlmostEqual(float(lor), 0.0)
        self.assertGreater(float(se2), 0)
        k, t2 = calibrate_prior([0.5, 0.7, 0.3], [0.0, 0.4, -0.4])
        self.assertGreater(k, 0)
        self.assertGreaterEqual(t2, 1e-3)

    def test_equivalent_multiplier(self):
        self.assertAlmostEqual(equivalent_multiplier(0.7, [0.6, 0.7, 0.8], ns=(50, 100, 200), n=50), 2.0)
        self.assertIsNone(equivalent_multiplier(0.95, [0.6, 0.7, 0.8], ns=(50, 100, 200)))

    def test_saving_with_informative_prior(self):
        rs = np.random.default_rng(3)
        n = 300
        base = rs.uniform(0.01, 0.03, n)
        lor = rs.normal(0, 0.25, n)
        ctr_a = base
        ctr_b = 1 / (1 + np.exp(-(np.log(base / (1 - base)) - lor)))
        prior = 1 / (1 + np.exp(-(lor + rs.normal(0, 0.15, n))))
        r = simulate_saving(ctr_a, ctr_b, prior, ns=(50, 100, 200, 400), reps=40)
        self.assertGreater(r["hybrid"]["50"], r["human"]["50"])
        self.assertGreater(r["prior_only"], 0.7)
        b = EarlyTestBayes().fit(prior[:100], lor[:100])
        p = b.predict(prior[100:], np.full(200, 1), 100, np.full(200, 1), 100)
        self.assertGreater(np.mean((p > 0.5) == (lor[100:] > 0)), 0.7)
        self.assertEqual(b.decide(0.9, 1, 100, 1, 100), True)

    @staticmethod
    def _synthetic(n=120, seed=3):
        rs = np.random.default_rng(seed)
        base = rs.uniform(0.01, 0.03, n)
        lor = rs.normal(0, 0.25, n)
        ctr_b = 1 / (1 + np.exp(-(np.log(base / (1 - base)) - lor)))
        prior = 1 / (1 + np.exp(-(lor + rs.normal(0, 0.15, n))))
        return base, ctr_b, prior

    def test_saving_rng_order_matches_research_script(self):
        # reference: the human curve of ornek_tasarrufu_llm.py (all replays first, then the tie-break draws)
        ctr_a, ctr_b, prior = self._synthetic()
        ns, R = (50, 100, 200), 30
        y = ctr_a > ctr_b
        rs = np.random.default_rng(11)
        sims = [(rs.binomial(n, ctr_a, size=(R, len(y))), rs.binomial(n, ctr_b, size=(R, len(y)))) for n in ns]
        ref = []
        for ca, cb in sims:
            d = ca - cb
            ref.append(float(np.mean(np.where(d == 0, rs.random(d.shape) < 0.5, d > 0) == y)))
        r = simulate_saving(ctr_a, ctr_b, prior, ns=ns, reps=R)
        self.assertEqual([r["human"][str(n)] for n in ns], ref)

    def test_saving_calibrated_block_protocol(self):
        ctr_a, ctr_b, prior = self._synthetic(n=200)
        r = simulate_saving_calibrated(ctr_a, ctr_b, prior, n_calib=20, ns=(25, 50, 100, 200, 400), reps=20,
                                       draws=2, n_boot=50)
        self.assertEqual(r["n_calib"], 20)
        self.assertGreater(r["hybrid"]["100"], r["human"]["100"])
        self.assertGreaterEqual(r["human_decides_prior_breaks_ties"], r["human"]["100"] - 0.02)
        self.assertTrue(r["multiplier"] is None or r["multiplier"] > 1)
        if r["multiplier"] is not None and r["ci"][0] is not None:
            self.assertLessEqual(r["ci"][0], r["ci"][1])
        with self.assertRaises(ValueError):
            simulate_saving_calibrated(ctr_a, ctr_b, prior, ns=(50, 200))


if __name__ == "__main__":
    unittest.main()
