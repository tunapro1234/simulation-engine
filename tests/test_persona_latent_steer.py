"""simengine.persona.latent.steer: gradients reach only the projector, training is seed-deterministic, controls."""
import unittest

import numpy as np

from tests._latent_util import needs_torch, tiny


def _setup(ld, soft_k=0):
    from simengine.persona.latent import prompts as P, steer
    sid, stext = steer.soft_placeholder(ld.tok)
    cats = ["Books", "Pet food", "Shoes"]
    prefixes, suffix = [], None
    for c in cats:
        pre, suf = P.short_split(ld.tok, P.fill(P.g1_messages("short", window="train"), c), soft_k=soft_k,
                                 soft_token=stext)
        prefixes.append(pre)
        suffix = suf
    rng = np.random.default_rng(0)
    U = rng.normal(size=(12, 5)).astype(np.float32)
    y = (U[:, 0] > 0).astype(int)
    cells = np.array([[c, u, y[u] if c != 1 else 1 - y[u]] for c in range(3) for u in range(12)])
    return prefixes, suffix, U, cells, sid


@needs_torch
class TestGradients(unittest.TestCase):
    def _grad_check(self, fam, kind):
        import torch
        from simengine.persona.latent import readout, steer
        ld = tiny(fam)
        prefixes, suffix, U, cells, sid = _setup(ld, soft_k=2 if kind == "soft" else 0)
        if kind == "soft":
            proj = steer.soft_token_projector(5, ld.hidden_size, k=2, target_norm=1.0)
        else:
            proj = steer.residual_projector(5, ld.hidden_size)
            torch.nn.init.normal_(proj[2].weight, std=0.1)               # non-zero output for the check
        inj = steer._injection(kind, proj, U, 2, "last", sid)
        cache, mask, _ = readout.prefix_forward(ld, readout._ids(ld.tok, prefixes))
        suf = torch.tensor([readout._ids(ld.tok, [suffix])[0]] * 4)
        with torch.enable_grad():
            lg = readout.suffix_logits(ld, cache, mask, [0, 1, 2, 0], suf, inject=inj, vec_idx=[0, 1, 2, 3])
            lg.logsumexp(-1).sum().backward()
        grads = [p.grad for p in proj.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)
        self.assertTrue(all(p.grad is None for p in ld.model.parameters()))

    def test_soft_tokens_grad_only_to_projector(self):
        for fam in ("qwen3_5", "gemma4", "qwen2", "gemma4_moe"):
            with self.subTest(fam):
                self._grad_check(fam, "soft")

    def test_residual_projector_grad_only_to_projector(self):
        for fam in ("qwen3_5", "gemma4", "qwen3"):
            with self.subTest(fam):
                self._grad_check(fam, "residual")


@needs_torch
class TestTraining(unittest.TestCase):
    def _train(self, kind, seed):
        from simengine.persona.latent import steer
        ld = tiny("qwen3_5")
        prefixes, suffix, U, cells, sid = _setup(ld, soft_k=1 if kind == "soft" else 0)
        proj = (steer.soft_token_projector(5, ld.hidden_size, k=1, seed=seed) if kind == "soft"
                else steer.residual_projector(5, ld.hidden_size, seed=seed))
        proj, hist = steer.train_injector(ld, proj, kind, prefixes, suffix, U, cells, U, cells, layer=2,
                                          positions="last", lr=1e-2, epochs=1.0, batch_size=12, seed=seed,
                                          soft_id=sid)
        p = steer.predict_injector(ld, proj, kind, prefixes, suffix, U, cells[:, :2], layer=2, soft_id=sid)
        return p, hist

    def test_seed_determinism_and_learning(self):
        for kind in ("residual", "soft"):
            with self.subTest(kind):
                p1, h1 = self._train(kind, 0)
                p2, h2 = self._train(kind, 0)
                np.testing.assert_array_equal(p1, p2)
                self.assertEqual([h["val"] for h in h1], [h["val"] for h in h2])
                self.assertTrue(np.isfinite(p1).all())

    def test_zero_init_residual_is_noop(self):
        from simengine.persona.latent import steer, yes_prob
        ld = tiny("qwen3_5")
        prefixes, suffix, U, cells, _ = _setup(ld)
        proj = steer.residual_projector(5, ld.hidden_size)
        p = steer.predict_injector(ld, proj, "residual", prefixes, suffix, U, cells[:, :2], layer=2)
        base, _ = yes_prob(ld, prefixes, [suffix])
        np.testing.assert_allclose(p, base[cells[:, 0], 0], atol=1e-6)

    def test_leak_guard_in_training(self):
        from simengine.persona.latent import LeakError, TestGuard, steer
        ld = tiny("qwen2")
        prefixes, suffix, U, cells, _ = _setup(ld)
        guard = TestGuard(["u3"])
        with self.assertRaises(LeakError):
            steer.train_injector(ld, steer.residual_projector(5, ld.hidden_size), "residual", prefixes, suffix, U,
                                 cells, U, cells, layer=1, guard=guard, uids_fit=[f"u{i}" for i in range(12)])


class TestDirections(unittest.TestCase):
    def test_derange(self):
        from simengine.persona.latent import derange
        for n in (2, 3, 10, 400):
            p = derange(n, seed=0)
            self.assertEqual(sorted(p), list(range(n)))
            self.assertFalse((p == np.arange(n)).any())
        np.testing.assert_array_equal(derange(50, seed=1), derange(50, seed=1))
        with self.assertRaises(ValueError):
            derange(1)

    def test_directions_recover_signal(self):
        from simengine.persona.latent import steer
        rng = np.random.default_rng(0)
        d = rng.normal(size=20)
        d /= np.linalg.norm(d)
        y = rng.integers(0, 2, 4000)
        X = rng.normal(size=(4000, 20)) + np.outer(y * 2 - 1, d)
        for v in (steer.mean_diff_direction(X[y == 1], X[y == 0]), steer.pca_direction(X, y),
                  steer.probe_direction(X, y)):
            self.assertGreater(float(steer.unit(v) @ d), 0.9)

    def test_caa_vector_scale(self):
        from simengine.persona.latent import CAA
        rng = np.random.default_rng(0)
        dirs = {"a": rng.normal(size=8).astype(np.float32), "b": rng.normal(size=8).astype(np.float32)}
        vals = rng.integers(0, 4, size=(100, 2)).astype(float)
        caa = CAA(dirs, layer=3, c=0.5, unit_norm=10.0)
        v = caa.vectors(vals)
        self.assertAlmostEqual(float(np.percentile(np.linalg.norm(v, axis=1), 95)), 5.0, places=3)
        self.assertEqual(caa.injection(vals).layer, 3)


if __name__ == "__main__":
    unittest.main()
