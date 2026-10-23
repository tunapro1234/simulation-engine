"""simengine.persona.latent.hooks: residual add, capture, soft-token replacement, clean removal."""
import unittest

from tests._latent_util import needs_torch, tiny

FAMILIES = ("qwen3_5", "gemma4", "gemma4_moe", "qwen2", "qwen3")


def _ids(ld, text="Will this customer buy books? Answer Yes or No."):
    import torch
    return torch.tensor([ld.tok(text, add_special_tokens=False)["input_ids"]])


@needs_torch
class TestResidualHook(unittest.TestCase):
    def test_add_changes_output_and_is_removed(self):
        import torch
        from simengine.persona.latent import add_to_residual
        for fam in FAMILIES:
            with self.subTest(fam):
                ld = tiny(fam)
                ids = _ids(ld)
                with torch.no_grad():
                    base = ld.model(input_ids=ids).logits
                    v = torch.randn(ld.hidden_size) * 5
                    with add_to_residual(ld.model, ld.full_attention[0], v, positions="all"):
                        steered = ld.model(input_ids=ids).logits
                    after = ld.model(input_ids=ids).logits
                self.assertGreater(float((steered - base).abs().max()), 1e-3)
                self.assertEqual(float((after - base).abs().max()), 0.0)

    def test_only_later_positions_change(self):
        import torch
        from simengine.persona.latent import add_to_residual
        for fam in FAMILIES:
            with self.subTest(fam):
                ld = tiny(fam)
                ids = _ids(ld)
                T = ids.shape[1]
                with torch.no_grad():
                    base = ld.model(input_ids=ids).logits
                    with add_to_residual(ld.model, 1, torch.randn(ld.hidden_size) * 5, positions=slice(T - 3, T - 2)):
                        out = ld.model(input_ids=ids).logits
                self.assertLess(float((out[:, :T - 3] - base[:, :T - 3]).abs().max()), 1e-5)
                self.assertGreater(float((out[:, T - 3:] - base[:, T - 3:]).abs().max()), 1e-4)

    def test_hook_removed_on_error(self):
        import torch
        from simengine.persona.latent import add_to_residual, layers
        ld = tiny("qwen2")
        lay = layers(ld.model)[1]
        n0 = len(lay._forward_hooks)
        with self.assertRaises(RuntimeError):
            with add_to_residual(ld.model, 1, torch.zeros(ld.hidden_size)):
                raise RuntimeError("boom")
        self.assertEqual(len(lay._forward_hooks), n0)

    def test_per_row_vectors_last_position(self):
        import torch
        from simengine.persona.latent import add_to_residual
        ld = tiny("qwen3_5")
        ids = _ids(ld).repeat(2, 1)
        v = torch.stack([torch.zeros(ld.hidden_size), torch.randn(ld.hidden_size) * 5])
        with torch.no_grad():
            base = ld.model(input_ids=ids).logits[:, -1]
            with add_to_residual(ld.model, 2, v, positions="last"):
                out = ld.model(input_ids=ids).logits[:, -1]
        self.assertEqual(float((out[0] - base[0]).abs().max()), 0.0)
        self.assertGreater(float((out[1] - base[1]).abs().max()), 1e-4)


@needs_torch
class TestCaptureAndSoft(unittest.TestCase):
    def test_capture_matches_hidden_states(self):
        import torch
        from simengine.persona.latent import capture
        for fam in FAMILIES:
            with self.subTest(fam):
                ld = tiny(fam)
                ids = _ids(ld)
                with torch.no_grad(), capture(ld.model, [0, 2], pool="none") as st:
                    hs = ld.model(input_ids=ids, output_hidden_states=True).hidden_states
                for l in (0, 2):
                    self.assertLess(float((st[l] - hs[l + 1].float()).abs().max()), 1e-5)

    def test_replace_embeddings_equals_inputs_embeds(self):
        import torch
        from simengine.persona.latent import input_embeddings, replace_embeddings
        for fam in FAMILIES:
            with self.subTest(fam):
                ld = tiny(fam)
                sid = ld.tok.convert_tokens_to_ids("<|soft|>")
                ids = _ids(ld)
                ids = torch.cat([ids[:, :3], torch.full((1, 2), sid), ids[:, 3:]], 1)
                vec = torch.randn(1, 2, ld.hidden_size)
                with torch.no_grad():
                    emb = input_embeddings(ld.model)(ids).clone()
                    emb[0, 3:5] = vec[0]
                    ref = ld.model(inputs_embeds=emb).logits
                    with replace_embeddings(ld.model, sid, vec):
                        out = ld.model(input_ids=ids).logits
                self.assertLess(float((out - ref).abs().max()), 1e-5)

    def test_replace_embeddings_count_checked(self):
        import torch
        from simengine.persona.latent import replace_embeddings
        ld = tiny("qwen2")
        sid = ld.tok.convert_tokens_to_ids("<|soft|>")
        ids = torch.cat([_ids(ld), torch.full((1, 1), sid)], 1)
        with self.assertRaises(ValueError), torch.no_grad():
            with replace_embeddings(ld.model, sid, torch.zeros(1, 2, ld.hidden_size)):
                ld.model(input_ids=ids)


if __name__ == "__main__":
    unittest.main()
