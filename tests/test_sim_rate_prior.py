"""sim.segment_rate_prior (segment x event-rate table from sampled answers), the NumericField export from
simengine.sim, and ab_many reader truncation (reader_chars, one warning with the count). Offline with FakeLLM."""
import hashlib
import inspect
import json
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine.sim as sim
from simengine.llm import FakeLLM
from simengine.llm.errors import LLMError
from simengine.llm.parallel import parallel_map
from simengine.methods import GroupTransfer
from simengine.sim import ab_many, parse_rate, schema_responder, segment_rate_prior

SEGS = {"new|fast": "new visitors whose first two page views are under 60 seconds apart",
        "new|slow": "new visitors whose first two page views are 300 seconds or more apart",
        "back|fast": "returning visitors whose first two page views are under 60 seconds apart"}
EVENT = "the session ends with a purchase"
TRUE_PCT = {"new|fast": 1.0, "new|slow": 2.0, "back|fast": 6.0}


class SampleLLM(FakeLLM):
    """Answers by the segment named in the prompt, + sample index (so samples differ); records the sample kw."""

    def __init__(self, reply=None):
        super().__init__(reply or self._reply)
        self.samples = []

    @staticmethod
    def _reply(system, user):
        for k, desc in SEGS.items():
            if desc in user:
                return json.dumps({"pct": TRUE_PCT[k]})
        return '{"pct": 3}'                     # blind arm: no segment description in the prompt

    def complete(self, system, user, *, max_tokens=256, **kw):
        self.samples.append(kw.get("sample"))
        r = super().complete(system, user, max_tokens=max_tokens, **kw)
        s = kw.get("sample") or 0
        try:
            v = json.loads(r.text)["pct"] + 0.5 * s
            r = type(r)(text=json.dumps({"pct": v}), model=r.model, input_tokens=r.input_tokens,
                        output_tokens=r.output_tokens)
        except (ValueError, KeyError, TypeError):
            pass
        return r


class TestSegmentRatePrior(unittest.TestCase):
    def test_table_mean_sd_counts(self):
        llm = SampleLLM()
        df = segment_rate_prior(llm, SEGS, EVENT, samples=3)
        self.assertEqual(list(df.index), list(SEGS))
        self.assertEqual(df.index.name, "segment")
        for k, pct in TRUE_PCT.items():                 # samples answer pct, pct + .5, pct + 1
            self.assertAlmostEqual(df.loc[k, "mean"], (pct + 0.5) / 100)
            self.assertAlmostEqual(df.loc[k, "sd"], 0.005)
            self.assertEqual(df.loc[k, "n_ok"], 3)
            self.assertEqual(df.loc[k, "n_failed"], 0)
        self.assertTrue(((df["mean"] >= 0) & (df["mean"] <= 1)).all())
        self.assertEqual(len(llm.calls), 9)
        self.assertEqual(sorted(set(llm.samples)), [0, 1, 2])
        json.dumps(df.attrs["usage"])

    def test_one_sample_is_plain_call(self):
        llm = SampleLLM()
        df = segment_rate_prior(llm, SEGS, EVENT, samples=1)
        self.assertEqual(set(llm.samples), {None})       # no sample= kw: the plain call's cache entry
        self.assertTrue(df["sd"].isna().all())
        self.assertAlmostEqual(df.loc["back|fast", "mean"], 0.06)

    def test_keyword_llm_and_list_segments(self):
        df = segment_rate_prior(list(SEGS.values()), EVENT, llm=SampleLLM(), samples=2)
        self.assertEqual(list(df.index), list(SEGS.values()))
        with self.assertRaises(TypeError):
            segment_rate_prior(SampleLLM(), SEGS, EVENT, llm=SampleLLM())

    def test_failed_answers_counted_and_warned(self):
        def reply(system, user):
            if SEGS["new|slow"] in user:
                return "I cannot know that."
            if SEGS["back|fast"] in user:
                return '{"pct": 140}'                    # out of range -> failed, never clipped
            return "Sure: ```json\n{\"pct\": \"12%\"}\n```"
        with self.assertWarns(RuntimeWarning) as cm:
            df = segment_rate_prior(FakeLLM(reply), SEGS, EVENT, samples=2)
        self.assertIn("4 of 6", str(cm.warning))
        self.assertEqual(df.loc["new|fast", "n_ok"], 2)
        self.assertAlmostEqual(df.loc["new|fast", "mean"], 0.12)
        for k in ("new|slow", "back|fast"):
            self.assertEqual(df.loc[k, "n_ok"], 0)
            self.assertEqual(df.loc[k, "n_failed"], 2)
            self.assertTrue(np.isnan(df.loc[k, "mean"]))
        self.assertEqual(df.attrs["warnings"][0], str(cm.warning))

    def test_proportion_mode(self):
        llm = FakeLLM(lambda s, u: '{"rate": 0.04}')
        df = segment_rate_prior(llm, SEGS, EVENT, samples=1, parse_pct=False)
        self.assertTrue(np.allclose(df["mean"], 0.04))
        self.assertIn("between 0 and 1", llm.calls[0][1])
        with self.assertWarns(RuntimeWarning):
            bad = segment_rate_prior(FakeLLM(lambda s, u: '{"rate": 4}'), SEGS, EVENT, samples=1, parse_pct=False)
        self.assertEqual(int(bad["n_failed"].sum()), 3)
        self.assertAlmostEqual(parse_rate("4%", parse_pct=False), 0.04)
        self.assertAlmostEqual(parse_rate('{"Percent": "7.5 %"}'), 0.075)
        self.assertIsNone(parse_rate('{"pct": true}'))
        self.assertIsNone(parse_rate("between 5 and 10"))

    def test_call_errors_are_failed_samples(self):
        def reply(system, user):
            if SEGS["new|slow"] in user:
                raise LLMError("upstream 500")
            return '{"pct": 5}'
        class SlotLLM(FakeLLM):                          # map with error slots, like the real backends
            def map(self, prompts, *, max_tokens=256, return_exceptions=False, parallel=None, **kw):
                return parallel_map(lambda su: self.complete(su[0], su[1], max_tokens=max_tokens, **kw),
                                    list(prompts), parallel=1, return_exceptions=return_exceptions)
        with self.assertWarns(RuntimeWarning):
            df = segment_rate_prior(SlotLLM(reply), SEGS, EVENT, samples=2)
        self.assertEqual(df.loc["new|slow", "n_failed"], 2)
        self.assertEqual(df.loc["new|fast", "n_ok"], 2)
        self.assertEqual(df.attrs["usage"]["error_calls"], 2)

    def test_blind_arm(self):
        llm = SampleLLM()
        df = segment_rate_prior(llm, SEGS, EVENT, samples=1, blind=True)
        self.assertTrue(np.allclose(df["mean_blind"], 0.03))
        self.assertTrue({"sd_blind", "n_ok_blind", "n_failed_blind"} <= set(df.columns))
        blind_prompts = [u for _, u in llm.calls if not any(d in u for d in SEGS.values())]
        self.assertEqual(len(blind_prompts), 3)
        self.assertTrue(all(EVENT in u for u in blind_prompts))
        self.assertIn("group A", blind_prompts[0])
        custom = segment_rate_prior(SampleLLM(), SEGS, EVENT, samples=1,
                                    blind={k: f"visitors of kind {i}" for i, k in enumerate(SEGS)})
        self.assertTrue(np.allclose(custom["mean_blind"], 0.03))

    def test_prompt_has_no_extra_names(self):
        llm = SampleLLM()
        segment_rate_prior(llm, SEGS, EVENT, samples=1, context="An online shop logs page views.")
        system, user = llm.calls[0]
        self.assertTrue(user.startswith("An online shop logs page views."))
        self.assertIn(EVENT, user)
        self.assertIn('"pct"', user)

    def test_group_transfer_feature(self):
        rng = np.random.default_rng(0)
        groups = np.repeat(list(SEGS), 40)
        X = pd.DataFrame({"x": rng.normal(size=len(groups))})
        y = np.array([TRUE_PCT[g] / 100 for g in groups]) + 0.01 * rng.normal(size=len(groups))
        prior = segment_rate_prior(SampleLLM(), SEGS, EVENT, samples=2)
        m = GroupTransfer(group_features=prior[["mean"]], person_model=None).fit(X, y, groups=groups)
        self.assertIn("g_mean", m.group_feature_names_)
        self.assertEqual(len(m.predict(X, groups=groups)), len(X))

    def test_exported(self):
        self.assertIn("segment_rate_prior", sim.__all__)
        params = inspect.signature(segment_rate_prior).parameters
        self.assertEqual(params["samples"].default, 5)
        self.assertIs(params["blind"].default, False)
        self.assertIs(params["parse_pct"].default, True)


class TestNumericFieldExport(unittest.TestCase):
    def test_import_from_sim(self):
        from simengine.sim import NumericField, numeric_fields
        self.assertIn("NumericField", sim.__all__)
        self.assertIn("numeric_fields", sim.__all__)
        f = numeric_fields([NumericField("spend", 0, 200, unit="USD"), ("visits", 0, 30)])
        self.assertEqual([x.name for x in f], ["spend", "visits"])


def long_personas(n=3, size=600):
    return [{"id": f"r{i}", "text": f"reader {i} " + "x" * size} for i in range(n)]


TESTS = [{"test_id": "t0", "a": "head A", "b": "head B"}]


class TestABManyReaderChars(unittest.TestCase):
    def test_default_prompt_bytes_unchanged(self):
        ps = long_personas(2) + [{"id": "short", "text": "short reader"}]
        llm = FakeLLM(schema_responder())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ab_many(ps, TESTS, llm=llm)
        system, user = llm.calls[0]
        # sha256 of system + NUL + user as built before reader_chars existed (cut at 400, recorded on 567c6b2)
        self.assertEqual(hashlib.sha256((system + "\x00" + user).encode()).hexdigest(),
                         "623c6515ba9082a7a33d26ddd5b4bc7b18799052a221fff63bad3dc26b9e59b6")

    def test_cut_warns_once_with_count(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = ab_many(long_personas(3) + [{"id": "short", "text": "short"}], TESTS * 1,
                         llm=FakeLLM(schema_responder()))
        cut = [x for x in w if "reader_chars" in str(x.message)]
        self.assertEqual(len(cut), 1)
        self.assertIn("3 of 4", str(cut[0].message))
        self.assertIn(str(cut[0].message), df.attrs["warnings"])

    def test_reader_chars_param(self):
        llm = FakeLLM(schema_responder())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ab_many(long_personas(2), TESTS, llm=llm, reader_chars=None)
        self.assertIn("x" * 600, llm.calls[0][1])
        llm = FakeLLM(schema_responder())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ab_many(long_personas(2), TESTS, llm=llm, reader_chars=50)
        self.assertNotIn("x" * 50, llm.calls[0][1])
        self.assertTrue(any("2 of 2" in str(x.message) and "50" in str(x.message) for x in w))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ab_many([{"id": "s", "text": "short"}], TESTS, llm=FakeLLM(schema_responder()))   # no cut, no warning


if __name__ == "__main__":
    unittest.main()
