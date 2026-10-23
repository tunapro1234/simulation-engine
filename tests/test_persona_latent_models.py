"""simengine.persona.latent.models: tiny random models of every target family (no download)."""
import unittest

from tests._latent_util import needs_torch, tiny


@needs_torch
class TestTinyModels(unittest.TestCase):
    def test_layer_lookup_every_family(self):
        from simengine.persona.latent import models as M
        expect = {"qwen3_5": ("Qwen3_5ForConditionalGeneration", 8, [3, 7]),
                  "qwen3_5_causal": ("Qwen3_5ForCausalLM", 8, [3, 7]),
                  "gemma4": ("Gemma4ForConditionalGeneration", 12, [5, 11]),
                  "gemma4_moe": ("Gemma4ForConditionalGeneration", 12, [5, 11]),
                  "qwen3": ("Qwen3ForCausalLM", 4, [0, 1, 2, 3]), "qwen2": ("Qwen2ForCausalLM", 4, [0, 1, 2, 3]),
                  "llama": ("LlamaForCausalLM", 4, [0, 1, 2, 3]), "qwen3_moe": ("Qwen3MoeForCausalLM", 4, [0, 1, 2, 3])}
        for fam, (cls, n, full) in expect.items():
            with self.subTest(fam):
                ld = tiny(fam)
                self.assertEqual(type(ld.model).__name__, cls)
                self.assertEqual(len(M.layers(ld.model)), n)
                self.assertEqual(ld.n_layers, n)
                self.assertEqual(ld.full_attention, full)
                self.assertEqual(ld.hidden_size, 64)
                self.assertTrue(all(not p.requires_grad for p in ld.model.parameters()))

    def test_multimodal_wrappers_use_language_model_layers(self):
        from simengine.persona.latent import models as M
        for fam in ("qwen3_5", "gemma4"):
            ld = tiny(fam)
            self.assertIs(M.layers(ld.model), ld.model.model.language_model.layers)

    def test_gemma_embed_scale(self):
        self.assertAlmostEqual(tiny("gemma4").embed_scale, 8.0, places=4)      # sqrt(64)
        self.assertEqual(tiny("qwen2").embed_scale, 1.0)

    def test_layer_index_clipped(self):
        ld = tiny("qwen2")
        self.assertEqual(ld.layer_index(0.5), 2)
        self.assertEqual(ld.layer_index(1.0), 3)
        self.assertEqual(ld.layer_index(-1), 0)

    def test_layers_error_on_unknown_module(self):
        import torch
        from simengine.persona.latent import models as M
        with self.assertRaises(ValueError):
            M.layers(torch.nn.Linear(2, 2))

    def test_precision_guards_without_download(self):
        import torch
        from simengine.persona.latent import models as M
        if not torch.cuda.is_available():
            with self.assertRaises(ValueError):
                M.quant_config("fp8")
        with self.assertRaises(ValueError):
            M.quant_config("nf4", moe=True)
        with self.assertRaises(ValueError):
            M.quant_config("int3")
        with self.assertRaises(ValueError):
            M.load("q38f", "bf16")                      # the FP8 checkpoint only loads as fp8
        self.assertIsNone(M.quant_config("bf16"))

    def test_check_backward_tiny(self):
        self.assertTrue(__import__("simengine.persona.latent", fromlist=["x"]).check_backward(tiny("qwen3_5")))

    def test_byte_tokenizer_chat(self):
        from simengine.persona.latent import models as M, prompts as P
        tok = M.byte_tokenizer()
        s = P.render(tok, [{"role": "user", "content": "hi"}])
        self.assertTrue(s.endswith("<|im_start|>assistant\n"))
        self.assertEqual(tok.decode(tok("héllo Yes", add_special_tokens=False)["input_ids"]), "héllo Yes")


class TestRevision(unittest.TestCase):
    def test_snapshot_hash_from_cache(self):
        import tempfile
        from pathlib import Path
        from simengine.persona.latent import models
        with tempfile.TemporaryDirectory() as d:
            rep = Path(d) / "models--org--m"
            (rep / "snapshots" / "abc123").mkdir(parents=True)
            (rep / "snapshots" / "abc123" / "config.json").write_text("{}")
            (rep / "refs").mkdir()
            (rep / "refs" / "main").write_text("abc123")
            self.assertEqual(models._snapshot("org/m", None, d), "abc123")
            self.assertIsNone(models._snapshot("org/other", None, d))


if __name__ == "__main__":
    unittest.main()
