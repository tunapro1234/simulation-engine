"""Optional live smoke test. Runs only with SIMENGINE_LIVE=1 and a key in ANTHROPIC_API_KEY or
SIMENGINE_KEY_FILE. One Haiku call, max_tokens 64, budget $0.01."""
import os
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("SIMENGINE_LIVE") == "1", "live test: set SIMENGINE_LIVE=1")
class TestLiveClaude(unittest.TestCase):
    def test_one_call(self):
        from simengine.llm import Claude
        with tempfile.TemporaryDirectory() as d:
            llm = Claude("claude-haiku-5-5", effort="low", cache_dir=d, budget_usd=0.01)
            r = llm.complete("Answer with one word.", "What colour is the sky on a clear day?", max_tokens=64)
            self.assertTrue(r.text.strip())
            self.assertTrue(llm.complete("Answer with one word.", "What colour is the sky on a clear day?",
                                         max_tokens=64).cached)


if __name__ == "__main__":
    unittest.main()
