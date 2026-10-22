"""SSR embedding speed-ups and the missing-model message: max_chars cut, disk cache, progress, loader error.

    nice -n 19 python -m unittest tests.test_sim_ssr_embed
"""
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from simengine.sim import ssr as ssr_mod
from simengine.sim.ssr import SSR


class CountingEncoder:
    """texts -> (n, 3) deterministic vectors; records every batch it is asked for."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return np.array([[len(t), sum(map(ord, t)) % 97 + 1, 1.0] for t in texts], dtype=np.float32)

    @property
    def seen(self):
        return [t for c in self.calls for t in c]


class TestTruncation(unittest.TestCase):
    def test_constructor_default_cuts(self):
        enc = CountingEncoder()
        s = SSR(encoder=enc, max_chars=10)
        s.embed(["x" * 50, "short"])
        self.assertEqual(enc.seen, ["x" * 10, "short"])

    def test_per_call_override_and_none(self):
        enc = CountingEncoder()
        s = SSR(encoder=enc, max_chars=10)
        s.embed(["y" * 50], max_chars=4)
        s.embed(["z" * 50], max_chars=None)
        self.assertEqual(enc.seen, ["y" * 4, "z" * 50])

    def test_default_is_2000(self):
        enc = CountingEncoder()
        SSR(encoder=enc).embed(["a" * 5000])
        self.assertEqual(len(enc.seen[0]), 2000)

    def test_cut_texts_share_one_embedding(self):
        enc = CountingEncoder()
        s = SSR(encoder=enc, max_chars=5)
        E = s.embed(["abcdeX", "abcdeY", "abcde"])
        self.assertEqual(enc.seen, ["abcde"])
        np.testing.assert_allclose(E[0], E[2])


class TestDiskCache(unittest.TestCase):
    def test_second_object_reads_disk(self):
        with tempfile.TemporaryDirectory() as d:
            enc = CountingEncoder()
            E1 = SSR(encoder=enc, cache_dir=d).embed(["one", "two", "one"])
            self.assertEqual(enc.seen, ["one", "two"])
            enc2 = CountingEncoder()
            E2 = SSR(encoder=enc2, cache_dir=d).embed(["two", "one", "three"])
            self.assertEqual(enc2.seen, ["three"])  # only the new text is encoded
            np.testing.assert_allclose(E2[0], E1[1])
            np.testing.assert_allclose(E2[1], E1[0])
            files = [f for _, _, fs in os.walk(d) for f in fs]
            self.assertEqual(len(files), 3)
            self.assertTrue(all(f.endswith(".npy") for f in files))  # no .tmp left behind

    def test_full_hit_never_calls_encoder(self):
        with tempfile.TemporaryDirectory() as d:
            SSR(encoder=CountingEncoder(), cache_dir=d).embed(["a", "b"])
            enc = mock.Mock(side_effect=AssertionError("encoder called"))
            enc.__qualname__ = CountingEncoder.__qualname__
            enc.__module__ = CountingEncoder.__module__
            E = SSR(encoder=enc, cache_dir=d).embed(["a", "b"])
            self.assertEqual(E.shape, (2, 3))
            enc.assert_not_called()

    def test_key_depends_on_model(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = SSR(cache_dir=d, model="m1"), SSR(cache_dir=d, model="m2")
            self.assertNotEqual(a._disk_path("t"), b._disk_path("t"))
            self.assertEqual(a._disk_path("t"), SSR(cache_dir=d, model="m1")._disk_path("t"))

    def test_cache_key_separates_encoders(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "cache_key"):  # two lambdas would share one key
                SSR(encoder=lambda t: np.ones((len(t), 2)), cache_dir=d)
            a = SSR(encoder=lambda t: np.ones((len(t), 2)), cache_dir=d, cache_key="enc-v1")
            c = SSR(encoder=lambda t: np.ones((len(t), 2)), cache_dir=d, cache_key="enc-v2")
            self.assertNotEqual(a._disk_path("t"), c._disk_path("t"))
            SSR(encoder=lambda t: np.ones((len(t), 2)))  # no disk cache: no key needed

    def test_corrupt_file_is_a_miss(self):
        with tempfile.TemporaryDirectory() as d:
            enc = CountingEncoder()
            s = SSR(encoder=enc, cache_dir=d)
            p = s._disk_path("bad")
            os.makedirs(os.path.dirname(p))
            with open(p, "wb") as f:
                f.write(b"not npy")
            s.embed(["bad"])
            self.assertEqual(enc.seen, ["bad"])

    def test_empty_file_is_a_miss(self):
        with tempfile.TemporaryDirectory() as d:
            enc = CountingEncoder()
            s = SSR(encoder=enc, cache_dir=d)
            p = s._disk_path("empty")
            os.makedirs(os.path.dirname(p))
            open(p, "wb").close()
            s.embed(["empty"])
            self.assertEqual(enc.seen, ["empty"])


class TestProgress(unittest.TestCase):
    def test_callable_gets_one_call_per_chunk(self):
        got = []
        enc = CountingEncoder()
        s = SSR(encoder=enc, progress=lambda done, total, rate: got.append((done, total, rate)), progress_every=2)
        s.embed([f"t{i}" for i in range(5)])
        self.assertEqual([(d, t) for d, t, _ in got], [(2, 5), (4, 5), (5, 5)])
        self.assertTrue(all(r > 0 for _, _, r in got))
        self.assertEqual([len(c) for c in enc.calls], [2, 2, 1])

    def test_true_prints_rate_to_stderr(self):
        s = SSR(encoder=CountingEncoder(), progress=True, progress_every=10)
        with mock.patch("sys.stderr") as err:
            s.embed(["a", "b", "c"])
        out = "".join(c.args[0] for c in err.write.call_args_list)
        self.assertIn("3/3 texts", out)
        self.assertIn("texts/s", out)

    def test_off_by_default_and_cached_texts_not_reported(self):
        got = []
        s = SSR(encoder=CountingEncoder(), progress=lambda *a: got.append(a))
        s.embed(["a"])
        s.embed(["a"])
        self.assertEqual(len(got), 1)


class TestMissingModel(unittest.TestCase):
    def test_offline_miss_names_the_knobs(self):
        def boom(name, **kw):
            raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")

        with mock.patch.object(ssr_mod, "_sentence_transformer", boom), \
                mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=False):
            os.environ.pop("SIMENGINE_HF_CACHE", None)
            with self.assertRaises(OSError) as cm:
                SSR(model="org/some-model").embed(["hello"])
        msg = str(cm.exception)
        for needle in ("SIMENGINE_HF_CACHE", "cache_folder", "HF_HOME", "offline=False", "org/some-model",
                       "couldn't connect"):
            self.assertIn(needle, msg)
        self.assertIsInstance(cm.exception.__cause__, OSError)

    def test_cache_folder_is_passed_and_named(self):
        seen = {}

        def boom(name, **kw):
            seen.update(kw)
            raise ValueError("not found locally")

        with mock.patch.object(ssr_mod, "_sentence_transformer", boom):
            with self.assertRaises(OSError) as cm:
                SSR(model="m", cache_folder="hf-cache-dir").embed(["x"])
        self.assertEqual(seen["cache_folder"], "hf-cache-dir")
        self.assertIn("cache_folder='hf-cache-dir'", str(cm.exception))

    def test_hf_home_folder_resolves_to_hub(self):
        seen = {}

        def boom(name, **kw):
            seen.update(kw)
            raise OSError("miss")

        with tempfile.TemporaryDirectory() as home:
            os.makedirs(os.path.join(home, "hub", "models--org--m"))
            with mock.patch.object(ssr_mod, "_sentence_transformer", boom):
                with self.assertRaises(OSError) as cm:
                    SSR(model="org/m", cache_folder=home).embed(["x"])
                self.assertEqual(seen["cache_folder"], os.path.join(home, "hub"))
                self.assertIn(os.path.join(home, "hub"), str(cm.exception))
                hub = os.path.join(home, "hub")
                with self.assertRaises(OSError):
                    SSR(model="org/m", cache_folder=hub).embed(["x"])
                self.assertEqual(seen["cache_folder"], hub)  # a hub folder is used as given

    def test_loader_used_once(self):
        class Model:
            def encode(self, texts, **kw):
                return np.ones((len(texts), 4), dtype=np.float32) / 2

        calls = []
        with mock.patch.object(ssr_mod, "_sentence_transformer", lambda n, **kw: calls.append(n) or Model()):
            s = SSR(model="m")
            s.embed(["a"])
            s.embed(["b"])
        self.assertEqual(calls, ["m"])


if __name__ == "__main__":
    unittest.main()
