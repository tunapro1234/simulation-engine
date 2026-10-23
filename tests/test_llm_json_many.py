"""llm.complete_json_many: complete_json over many requests through the backend's parallel setting (G10 #6)."""
import threading
import time
import unittest

from simengine.llm import AuthError, FakeLLM, complete_json_many
from simengine.types import LLMResponse


class SleepyFake(FakeLLM):
    """FakeLLM whose replies take `delay` seconds; records the peak number of calls in flight."""

    def __init__(self, responder, delay=0.05, parallel=None):
        super().__init__(responder)
        if parallel is not None:
            self.parallel = parallel
        self.delay = delay
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def complete(self, system, user, *, max_tokens=256, **kw):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            return super().complete(system, user, max_tokens=max_tokens, **kw)
        finally:
            with self._lock:
                self.active -= 1


def echo(system, user):
    """'{"i": n}' for a user prompt 'n'; 'bad' never parses; 'cut' opens an object and never closes it."""
    if user.startswith("bad"):
        return "no json here"
    if user.startswith("cut"):
        return '{"i": 1, "rest": "'
    return '{"i": %d}' % int(user.split()[0])


class CompleteJsonManyTest(unittest.TestCase):
    def test_order_kept_with_uneven_latency(self):
        def slow_first(system, user):
            time.sleep(0.02 * (10 - int(user)))         # early requests finish last
            return echo(system, user)
        llm = FakeLLM(slow_first)
        out = complete_json_many(llm, [("Return JSON only.", str(i)) for i in range(10)], parallel=5)
        self.assertEqual(out, [{"i": i} for i in range(10)])

    def test_dict_requests_and_per_request_kwargs(self):
        seen = []

        class Rec(FakeLLM):
            def complete(self, system, user, *, max_tokens=256, **kw):
                seen.append((user, max_tokens, kw.get("tag")))
                return super().complete(system, user, max_tokens=max_tokens)
        llm = Rec(echo)
        out = complete_json_many(llm, [{"system": "s", "user": "1"}, {"system": "s", "user": "2", "max_tokens": 900},
                                       ("s", "3")], parallel=1, max_tokens=600, tag="L2")
        self.assertEqual(out, [{"i": 1}, {"i": 2}, {"i": 3}])
        self.assertEqual(seen, [("1", 600, "L2"), ("2", 900, "L2"), ("3", 600, "L2")])

    def test_one_failure_does_not_sink_the_batch(self):
        llm = FakeLLM(echo)
        reqs = [("s", "0"), ("s", "bad"), ("s", "2")]
        out = complete_json_many(llm, reqs, parallel=3)
        self.assertEqual(out[0], {"i": 0})
        self.assertIsInstance(out[1], ValueError)
        self.assertEqual(out[2], {"i": 2})
        out = complete_json_many(FakeLLM(echo), reqs, parallel=3, return_exceptions=False)
        self.assertEqual(out, [{"i": 0}, None, {"i": 2}])

    def test_stop_error_still_raises(self):
        def stop(system, user):
            if user == "3":
                raise AuthError("401")
            return echo(system, user)
        with self.assertRaises(AuthError):
            complete_json_many(FakeLLM(stop), [("s", str(i)) for i in range(6)], parallel=2)

    def test_stats_summed(self):
        llm = FakeLLM(echo)
        st = {}
        reqs = [("s", "0"), ("s", "bad"), ("s", "cut"), ("s", "3")]
        out = complete_json_many(llm, reqs, parallel=4, attempts=2, max_tokens=600, stats=st)
        self.assertEqual(out[0], {"i": 0})
        self.assertEqual(out[3], {"i": 3})
        self.assertIsInstance(out[1], ValueError)
        self.assertIsInstance(out[2], ValueError)
        # 0, 3: one call each. bad: 2 calls, 2 parse failures. cut: call + one bigger-budget retry + the
        # "JSON only" re-ask = 3 calls, all 3 truncated, 1 truncation retry, 2 parse failures.
        self.assertEqual(st, {"requests": 4, "failed": 2, "calls": 7, "parse_failures": 4, "truncated": 3,
                              "truncation_retries": 1})
        self.assertEqual(len(llm.calls), 7)
        st2 = {"calls": 10}
        complete_json_many(FakeLLM(echo), [("s", "1")], stats=st2)
        self.assertEqual(st2, {"calls": 11, "requests": 1, "failed": 0})

    def test_parallel_overlaps(self):
        reqs = [("s", str(i)) for i in range(8)]
        seq = SleepyFake(echo, delay=0.05)
        t0 = time.perf_counter()
        complete_json_many(seq, reqs, parallel=1)
        t_seq = time.perf_counter() - t0
        par = SleepyFake(echo, delay=0.05)
        t0 = time.perf_counter()
        out = complete_json_many(par, reqs, parallel=8)
        t_par = time.perf_counter() - t0
        self.assertEqual(out, [{"i": i} for i in range(8)])
        self.assertEqual(seq.peak, 1)
        self.assertGreater(par.peak, 1)
        self.assertLess(t_par, t_seq / 2)

    def test_parallel_defaults_to_backend_setting(self):
        llm = SleepyFake(echo, delay=0.03, parallel=3)
        complete_json_many(llm, [("s", str(i)) for i in range(9)])
        self.assertEqual(llm.peak, 3)

    def test_empty_and_bad_request(self):
        self.assertEqual(complete_json_many(FakeLLM(echo), []), [])
        with self.assertRaises(TypeError):
            complete_json_many(FakeLLM(echo), ["just a user prompt"])


if __name__ == "__main__":
    unittest.main()
