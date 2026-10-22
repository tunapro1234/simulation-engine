"""The snippets in examples/hackathon_day.md (steps 0, 2, 4, 5) run as written, offline.

    python3 -m unittest tests.test_playbook_snippets
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import simengine as se
from simengine.llm._base import CachedBackend
from simengine.llm.ledger import CostLedger
from simengine.methods.hybrid import BinaryCalibrator, DevStack, ShrunkDeviation
from simengine.research import Prereg, lock_prereg


def _audit(rep, name):
    return next(k for k in rep["denetim"] if k["kontrol"].startswith(name))


def _frame(n=200, seed=0, split=True):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    d = pd.DataFrame({"truth": y, "pred_stack": np.clip(0.3 * y + 0.7 * rng.random(n), 0, 1),
                      "pred_logistic": rng.random(n)})
    if split:
        d["split"] = ["train"] * (n // 2) + ["test"] * (n - n // 2)
    return d


class Step0Setup(unittest.TestCase):
    def test_spend_log_with_own_ledger_raises(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                CachedBackend(ledger=CostLedger(Path(t) / "l.jsonl"), spend_log=Path(t) / "s.tsv")
            b = CachedBackend(ledger=CostLedger(Path(t) / "l.jsonl", tsv_path=Path(t) / "s.tsv", agent="a"))
            self.assertIsInstance(b.stats, dict)              # a property, not a method
            for k in ("calls", "cache_hits", "retries", "errors", "cost_usd", "ledger_total_usd"):
                self.assertIn(k, b.stats)


class Step2Prereg(unittest.TestCase):
    def _run(self, notes):
        with tempfile.TemporaryDirectory() as t:
            reg = Path(t) / ".onkayit"
            Prereg(title="t", question="q", metrics="AUC (higher better)", baselines=["constant", "logistic"],
                   expected="stack beats logistic", notes=notes).write(reg / "prereg.md")
            lock_prereg(reg / "prereg.md")
            fr = _frame()
            se.seal(fr.drop(columns="truth"), "test predictions", registry=str(reg))
            rep = se.evaluate(fr, ours="stack", task="binary", primary="auc", prereg=str(reg),
                              unlabeled_baselines=["logistic"])
            return _audit(rep, "seçim yanlılığı")

    def test_ours_marker_in_registry_is_found(self):
        a = self._run("ours: stack")
        self.assertEqual(a["durum"], "tamam")
        self.assertIn("önceden adlandırıldı", a["ayrinti"])

    def test_without_marker_other_arm_names_make_it_ambiguous(self):
        a = self._run("")
        self.assertEqual(a["durum"], "uyari")
        self.assertIn("ours: <ad>", a["ayrinti"])


class Step4Learner(unittest.TestCase):
    def test_devstack_and_calibrator(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 300)
        learner, llm = np.clip(0.4 * y + 0.6 * rng.random(300), 0.01, 0.99), rng.random(300)
        st = DevStack("binary").fit([learner[:200], llm[:200]], y[:200])
        p = st.predict([learner[200:], llm[200:]])
        self.assertEqual(p.shape, (100,))
        self.assertEqual(st.dev_pred_.shape, (200,))
        q = BinaryCalibrator().fit(learner[:200], y[:200]).predict(learner[200:])
        self.assertTrue(np.all((q >= 0) & (q <= 1)))

    def test_shrunk_deviation_distribution(self):
        rng = np.random.default_rng(2)
        pop = np.array([0.5, 0.3, 0.2])
        llm_seg = rng.dirichlet([2, 2, 2], 8)
        true_seg = rng.dirichlet([5, 3, 2], 8)
        sd = ShrunkDeviation(distribution=True).fit(base=pop, llm=llm_seg, y=true_seg, ref=llm_seg.mean(0))
        P = sd.predict(base=pop, llm=llm_seg, ref=llm_seg.mean(0))
        self.assertEqual(P.shape, (8, 3))
        np.testing.assert_allclose(P.sum(1), 1.0)


class Step4FreeTextHistory(unittest.TestCase):
    def test_log_raw_renders_texts_before_cutoff(self):
        import simengine.persona.inject as pi
        person = {"id": "u1", "history": [{"t": "2024-01-02", "text": "Battery died after a week."},
                                          {"t": "2024-05-01", "text": "after the cutoff"}]}
        q = {"id": "q1", "text": "Would you buy again?", "options": ["yes", "no"], "cutoff": "2024-04-01"}
        system = pi.get_injector("log_raw", n=5)(person, q).system
        self.assertIn("Battery died after a week.", system)
        self.assertNotIn("after the cutoff", system)


class Step5Split(unittest.TestCase):
    def test_split_column_runs_the_split_audit(self):
        rep = se.evaluate(_frame(), ours="stack", task="binary", primary="auc")
        kontroller = [k["kontrol"] for k in rep["denetim"]]
        self.assertNotIn("bölme bilgisi (eğitim/test)", kontroller)
        self.assertIn("grup çakışması (eğitim ∩ test)", kontroller)

    def test_without_split_column_the_audit_is_skipped(self):
        rep = se.evaluate(_frame(split=False), ours="stack", task="binary", primary="auc")
        self.assertEqual(_audit(rep, "bölme bilgisi")["durum"], "atlandi")


if __name__ == "__main__":
    unittest.main()
