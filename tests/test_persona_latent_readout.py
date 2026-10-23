"""simengine.persona.latent.readout: prefix-cached P(yes) equals the uncached forward on every family."""
import unittest

import numpy as np

from tests._latent_util import needs_torch, tiny

FAMILIES = ("qwen3_5", "qwen3_5_causal", "gemma4", "gemma4_moe", "qwen2", "qwen3", "qwen3_moe")
PRE = ["<|im_start|>user\nSurvey answers: age 25-34, income high, buys books often. Question: in the category",
       "<|im_start|>user\nShort. Question: in the category",
       "<|im_start|>user\nA middle length persona text. Question: in the category"]
SUF = [" Books? Answer Yes or No.<|im_end|>\n<|im_start|>assistant\n",
       " Pet food? Answer Yes or No.<|im_end|>\n<|im_start|>assistant\n"]


def _uncached(ld, text, hook=None):
    import torch
    from simengine.persona.latent import prompts, readout
    ans = prompts.answer_token_ids(ld.tok)
    ids = torch.tensor([ld.tok(text, add_special_tokens=False)["input_ids"]])
    with torch.no_grad():
        lg = ld.model(input_ids=ids).logits[:, -1]
    return float(torch.sigmoid(readout.yes_logit(lg, ans["Yes"], ans["No"]))[0])


@needs_torch
class TestCachedReadout(unittest.TestCase):
    def test_cached_equals_uncached(self):
        from simengine.persona.latent import yes_prob
        for fam in FAMILIES:
            with self.subTest(fam):
                ld = tiny(fam)
                p, mass = yes_prob(ld, PRE, SUF, batch_size=2, cell_batch=3)
                ref = np.array([[_uncached(ld, a + b) for b in SUF] for a in PRE])
                self.assertEqual(p.shape, (3, 2))
                self.assertLess(float(np.abs(p - ref).max()), 1e-5)
                self.assertTrue(((mass > 0) & (mass <= 1)).all())

    def test_cells_subset_and_order(self):
        from simengine.persona.latent import yes_prob
        ld = tiny("qwen3_5")
        full, _ = yes_prob(ld, PRE, SUF)
        cells = np.array([[2, 1], [0, 0], [2, 0]])
        p, _ = yes_prob(ld, PRE, SUF, cells=cells)
        np.testing.assert_allclose(p, full[cells[:, 0], cells[:, 1]], atol=1e-6)

    def test_select_cache_leaves_source_intact(self):
        import torch
        from simengine.persona.latent import readout
        ld = tiny("qwen3_5")
        seqs = readout._ids(ld.tok, PRE)
        cache, mask, _ = readout.prefix_forward(ld, seqs)
        suf = torch.tensor([readout._ids(ld.tok, SUF[:1])[0]] * 2)
        with torch.no_grad():
            a = readout.suffix_logits(ld, cache, mask, [0, 2], suf)
            b = readout.suffix_logits(ld, cache, mask, [0, 2], suf)       # same cache reused
        self.assertEqual(float((a - b).abs().max()), 0.0)

    def test_expand_cache(self):
        from simengine.persona.latent import readout
        ld = tiny("gemma4")
        cache, mask, _ = readout.prefix_forward(ld, readout._ids(ld.tok, PRE[:1]))
        big = readout.expand_cache(cache, 3)
        self.assertEqual(big.layers[0].keys.shape[0], 3)
        self.assertEqual(cache.layers[0].keys.shape[0], 1)

    def test_residual_injection_matches_manual_hook(self):
        import torch
        from simengine.persona.latent import Injection, add_to_residual, yes_prob
        ld = tiny("qwen3_5")
        vecs = torch.randn(3, ld.hidden_size) * 4
        cells = np.array([[0, 0, 2], [1, 1, 0]])
        p, _ = yes_prob(ld, PRE, SUF, cells=cells, inject=Injection("residual", vecs, layer=2, positions="last4"))
        for k, (i, j, v) in enumerate(cells):
            text = PRE[i] + SUF[j]
            n = len(ld.tok(text, add_special_tokens=False)["input_ids"])
            with add_to_residual(ld.model, 2, vecs[v], positions=slice(n - 4, n)):
                ref = _uncached(ld, text)
            self.assertAlmostEqual(p[k], ref, places=5)

    def test_answer_ids_drop_shared_space(self):
        from simengine.persona.latent import answer_token_ids
        ld = tiny("qwen2")
        ids = answer_token_ids(ld.tok)
        self.assertFalse(set(ids["Yes"]) & set(ids["No"]))
        space = ld.tok.encode(" ", add_special_tokens=False)[0]
        self.assertNotIn(space, ids["Yes"])
        with self.assertRaises(ValueError):
            answer_token_ids(ld.tok, ("Xa", "Xb"))                    # both start with byte 'X'

    def test_option_probs_normalised(self):
        from simengine.persona.latent import option_probs
        ld = tiny("qwen2")
        out, mass = option_probs(ld, PRE[:2], [" Pick 1, 2 or 3.\n"], [["1", "2", "3"]])
        self.assertEqual(len(out), 2)
        for q in out:
            self.assertAlmostEqual(float(q.sum()), 1.0, places=6)
        self.assertTrue((mass > 0).all())

    def test_prefix_states_order_and_span_mean(self):
        from simengine.persona.latent import prefix_states, readout
        ld = tiny("qwen3_5")
        a = prefix_states(ld, PRE, layers=[1, 7], batch_size=2).hidden
        b = prefix_states(ld, PRE[::-1], layers=[1, 7], batch_size=3).hidden
        np.testing.assert_allclose(a[7], b[7][::-1], atol=1e-5)
        spans = [readout.token_span(ld.tok, t, "Question") for t in PRE]
        m = prefix_states(ld, PRE, layers=[1], pool="mean", spans=spans).hidden[1]
        self.assertEqual(m.shape, (3, ld.hidden_size))


@needs_torch
class TestPrompts(unittest.TestCase):
    def test_split_round_trip(self):
        from simengine.persona.latent import prompts as P
        ld = tiny("qwen2")
        msgs = P.g1_messages("L2", "- Age group: 25 - 34 years", "BOOK: 3 items", "January 2018 - June 2022")
        pre, suf = P.split(ld.tok, msgs, cell="Books")
        full = P.render(ld.tok, P.fill(msgs, "Books"))
        self.assertEqual(pre + suf, full)
        self.assertTrue(suf.startswith(" Books between July-December 2022"))
        self.assertNotIn("Books", pre)
        spre, ssuf = P.short_split(ld.tok, P.fill(P.g1_messages("short", window="train"), "Books"), soft_k=2,
                                   soft_token="<|soft|>")
        self.assertIn("Books between January-June 2022", spre)
        self.assertEqual(ssuf.count("<|soft|>"), 2)


if __name__ == "__main__":
    unittest.main()
