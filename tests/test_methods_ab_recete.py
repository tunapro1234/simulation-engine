"""methods.ab rehearsal gaps: offline A/B responder, cache-aware call counts, ABStack.predict(p_llm=), ab_archive."""
import json
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.llm import FakeLLM
from simengine.methods.ab import ABStack, LLMDirectAB, PAIR_LINE, ab_archive, ab_responder
from simengine.methods.bayes_prior import simulate_saving_calibrated
from simengine.types import LLMResponse

WORDS = ["river", "garden", "window", "letter", "bridge", "market", "winter", "silver", "forest", "engine"]


def pairs(n, seed=0):
    """y = 1 when A is the longer text."""
    rs = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        a, b = " ".join(rs.choice(WORDS, rs.integers(2, 7))), " ".join(rs.choice(WORDS, rs.integers(2, 7)))
        if len(a) == len(b):
            a += " x"
        rows.append({"text_a": a, "text_b": b, "y": int(len(a) > len(b)), "group": i})
    return pd.DataFrame(rows)


class CachingFake(FakeLLM):
    """FakeLLM with an in-memory cache: a repeated prompt comes back with cached=True and is not 'sent'."""

    def __init__(self, responder):
        super().__init__(responder)
        self.store = {}

    def complete(self, system, user, *, max_tokens=256, **kw):
        if (system, user) in self.store:
            return LLMResponse(text=self.store[(system, user)], model=self.model, cached=True)
        r = super().complete(system, user, max_tokens=max_tokens, **kw)
        self.store[(system, user)] = r.text
        return r


class TestABResponder(unittest.TestCase):
    def test_full_coverage_and_direction(self):
        df = pairs(40)
        for reverse in (False, True):
            m = LLMDirectAB(FakeLLM(ab_responder(len)), reverse=reverse)
            p = m.predict(df)
            self.assertEqual(m.coverage_, 1.0)
            self.assertTrue(np.array_equal(p > 0.5, df.y.to_numpy() == 1), reverse)

    def test_contract_parsed_from_prompt(self):
        df = pd.DataFrame({"text_a": ['say "hi" | B: x', "b"], "text_b": ["c", 'quote "end"']})
        user = LLMDirectAB(FakeLLM()).prompt(df)
        got = [(int(i), a, b) for i, a, b in PAIR_LINE.findall(user)]
        self.assertEqual(got, [(0, 'say "hi" | B: x', "c"), (1, "b", 'quote "end"')])
        out = json.loads(ab_responder(len)("", user))
        self.assertEqual([d["i"] for d in out], [0, 1])

    def test_default_hash_and_ties(self):
        df = pd.DataFrame({"text_a": ["same", "alpha"], "text_b": ["same", "beta"]})
        m = LLMDirectAB(FakeLLM(ab_responder()), reverse=False)
        p = m.predict(df)
        self.assertEqual(p[0], 0.5)
        self.assertNotEqual(p[1], 0.5)
        self.assertTrue(np.allclose(p, LLMDirectAB(FakeLLM(ab_responder()), reverse=False).predict(df)))

    def test_noise_is_seeded_and_order_free(self):
        df = pairs(30)
        r = ab_responder(len, noise=3.0, seed=1)
        p1 = LLMDirectAB(FakeLLM(r), reverse=False, batch=7).predict(df)
        p2 = LLMDirectAB(FakeLLM(r), reverse=False, batch=30).predict(df.iloc[::-1])[::-1]
        self.assertTrue(np.allclose(p1, p2))
        p3 = LLMDirectAB(FakeLLM(ab_responder(len, noise=3.0, seed=2)), reverse=False).predict(df)
        self.assertFalse(np.allclose(p1, p3))
        acc = np.mean((p1 > 0.5) == (df.y.to_numpy() == 1))
        self.assertTrue(0.5 < acc < 1.0, acc)


class TestCallCounts(unittest.TestCase):
    def test_cached_rerun_sends_nothing(self):
        df = pairs(40)
        llm = CachingFake(ab_responder(len))
        m = LLMDirectAB(llm, batch=15)                 # 3 chunks x 2 orders
        p1 = m.predict(df)
        self.assertEqual((m.calls_, m.cache_hits_), (6, 0))
        m2 = LLMDirectAB(llm, batch=15)
        p2 = m2.predict(df)
        self.assertEqual((m2.calls_, m2.cache_hits_), (0, 6))
        self.assertTrue(np.allclose(p1, p2))
        m.predict(df)                                  # counters accumulate on one instance
        self.assertEqual((m.calls_, m.cache_hits_), (6, 6))


class TestStackReuse(unittest.TestCase):
    def test_predict_with_p_llm_skips_llm(self):
        train, dev, test = pairs(120, 1), pairs(60, 2), pairs(40, 3)
        llm = FakeLLM(ab_responder(len, noise=1.0))
        direct = LLMDirectAB(llm)
        stack = ABStack(llm_method=direct).fit(train, dev=dev)
        ref = stack.predict(test)
        p_direct = direct.predict(test)
        n = len(llm.calls)
        got = stack.predict(test, p_llm=p_direct, valid=direct.valid_)
        self.assertEqual(len(llm.calls), n)
        self.assertTrue(np.allclose(got, ref))
        self.assertTrue(np.allclose(stack.predict(test, p_llm=p_direct), ref))   # valid defaults to all
        with self.assertRaises(ValueError):
            stack.predict(test, p_llm=p_direct[:-1])

    def test_no_llm_method_needs_p_llm(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stack = ABStack(llm_method=None, dev_n=0).fit(pairs(80))
        test = pairs(20, 4)
        self.assertEqual(len(stack.predict(test, p_llm=np.full(20, 0.5))), 20)
        with self.assertRaises(ValueError):
            stack.predict(test)


class TestArchive(unittest.TestCase):
    def wide(self):
        return pd.DataFrame({
            "tid": [1, 1, 2, 3, 4, 5],
            "ha": ["a1", "a1b", "a2", "a3", "a4", None],
            "hb": ["b1", "b1b", "b2", "b3", "b4", "b5"],
            "ca": [10, 30, 5, 7, 3, 1], "na": [100, 100, 100, 100, 0, 10],
            "cb": [20, 10, 5, 3, 1, 1], "nb": [100, 100, 100, 100, 10, 10]})

    def test_wide(self):
        kw = dict(test="tid", text_a="ha", text_b="hb", clicks_a="ca", n_a="na", clicks_b="cb", n_b="nb")
        f = ab_archive(self.wide(), **kw)
        # test 2 is an exact tie, test 4 has n=0, test 5 has no text; test 1 keeps one of its two pairs
        self.assertEqual(sorted(f.group), [1, 3])
        self.assertEqual(list(f.columns), ["group", "text_a", "text_b", "clicks_a", "n_a", "clicks_b", "n_b",
                                           "ctr_a", "ctr_b", "tie", "y"])
        r3 = f[f.group == 3].iloc[0]
        self.assertEqual((r3.ctr_a, r3.ctr_b, r3.y), (0.07, 0.03, 1))
        g = ab_archive(self.wide(), drop_ties=False, one_pair_per_test=False, **kw)
        self.assertEqual(sorted(g.group), [1, 1, 2, 3])
        self.assertEqual(int(g.tie.sum()), 1)
        self.assertEqual(int(g[g.tie].y.iloc[0]), 0)
        picks = {ab_archive(self.wide(), seed=s, **kw).query("group == 1").text_a.iloc[0] for s in range(20)}
        self.assertEqual(picks, {"a1", "a1b"})

    def test_long_and_downstream(self):
        rs = np.random.default_rng(0)
        rows = []
        for t in range(60):
            for v in range(rs.integers(1, 4)):        # tests with one variant give no pair
                n = 1000
                rows.append({"test_id": t, "headline": " ".join(rs.choice(WORDS, v + 2)),
                             "clicks": int(rs.binomial(n, 0.02 + 0.01 * v)), "impressions": n})
        raw = pd.DataFrame(rows)
        f = ab_archive(raw, test="test_id", text_a="headline", clicks_a="clicks", n_a="impressions")
        multi = raw.groupby("test_id").size().loc[lambda s: s >= 2].index
        self.assertLessEqual(set(f.group), set(multi))
        self.assertTrue(f.group.is_unique)
        self.assertFalse(f.tie.any())
        allp = ab_archive(raw, test="test_id", text_a="headline", clicks_a="clicks", n_a="impressions",
                          drop_ties=False, one_pair_per_test=False)
        k = raw.groupby("test_id").size()
        self.assertEqual(len(allp), int((k * (k - 1) // 2).sum()))
        p = LLMDirectAB(FakeLLM(ab_responder(len))).predict(f)
        self.assertEqual(len(p), len(f))
        res = simulate_saving_calibrated(f.ctr_a, f.ctr_b, p, f.clicks_a, f.clicks_b, reps=10, draws=1, n_boot=20)
        self.assertEqual(res["n_pairs"], len(f))

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            ab_archive(self.wide(), test="tid", text_a="ha", text_b="hb", clicks_a="ca", n_a="na")
        with self.assertRaises(KeyError):
            ab_archive(self.wide(), test="nope", text_a="ha", clicks_a="ca", n_a="na")


if __name__ == "__main__":
    unittest.main()
