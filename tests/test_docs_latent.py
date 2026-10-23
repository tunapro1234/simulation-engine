"""docs/modules/latent.md: the 'own data' recipe runs as written (tiny random model, no download)."""
import contextlib
import io
import re
import unittest
from pathlib import Path

from tests._latent_util import needs_torch

DOC = (Path(__file__).resolve().parents[1] / "docs" / "modules" / "latent.md").read_text(encoding="utf-8")


def recipe() -> str:
    m = re.search(r"<!-- test: latent-own-data -->\n```python\n(.*?)\n```", DOC, re.S)
    return m.group(1) if m else ""


@needs_torch
class TestOwnDataRecipe(unittest.TestCase):
    def test_runs(self):
        import torch
        torch.set_num_threads(4)
        code = recipe()
        self.assertIn("lt.tiny(", code)
        ns: dict = {}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(code, "latent.md", "exec"), ns)
        for arm in ("enc", "prompt", "probe", "rsd", "rsd-swap", "sft", "caa"):
            p = ns["preds"][arm]
            self.assertEqual(len(p), int(ns["m"]["test"].sum()), arm)
            self.assertTrue(((p >= 0) & (p <= 1)).all(), arm)


class TestErrors(unittest.TestCase):
    def test_cat_labels(self):
        import numpy as np
        from simengine.persona.latent import EncoderHead
        X = np.random.default_rng(0).normal(size=(6, 3))
        with self.assertRaisesRegex(ValueError, "integer category codes"):
            EncoderHead().fit(X, np.arange(6), np.array(["a", "b"] * 3), np.array([0, 1] * 3))

    @needs_torch
    def test_layer_out_of_range(self):
        from simengine.persona import latent as lt
        from tests._latent_util import tiny
        ld = tiny("qwen3")
        with self.assertRaisesRegex(ValueError, "decoder layers"):
            with lt.add_to_residual(ld.model, 30, None):
                pass

    @needs_torch
    def test_injector_soft_without_placeholder(self):
        import numpy as np
        from simengine.persona import latent as lt
        from tests._latent_util import tiny
        ld = tiny("qwen3")
        proj = lt.soft_token_projector(4, ld.hidden_size, k=1)
        U = np.zeros((2, 4), np.float32)
        with self.assertRaisesRegex(ValueError, "soft placeholder"):
            lt.predict_injector(ld, proj, "soft", ["Q?"], " A:", U, np.array([[0, 0], [0, 1]]))
        with self.assertRaisesRegex(ValueError, "needs layer"):
            lt.predict_injector(ld, proj, "residual", ["Q?"], " A:", U, np.array([[0, 0], [0, 1]]))


if __name__ == "__main__":
    unittest.main()
