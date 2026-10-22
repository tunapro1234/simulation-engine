"""evidence.evaluate: pre-registration documents linked to the seal chain (prereg list / prereg_doc=, lock time,
seal(prereg_doc=)), a declared constant split (split=) and the dominant-block check (balance_col=). Offline, small
synthetic data only."""
import json
import os
import tempfile
import time
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import simengine.evidence as ev
from simengine import research

FAST = dict(B=100, B_aggregate=20, multiplier=False)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _cont(n=240, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, n)
    return pd.DataFrame({"truth": y, "pred_good": y + rng.normal(0, 0.5, n), "pred_naive": y + rng.normal(0, 1.5, n),
                         "pred_noise": rng.normal(0, 1, n), "pred_other": y + rng.normal(0, 1.0, n)})


def _sel(rep):
    return next(d for d in rep["denetim"] if d["kontrol"] == "seçim yanlılığı")


def _audit(rep, name):
    return next((d for d in rep["denetim"] if name in d["kontrol"]), None)


def _future(f: Path):
    t = time.time() + 3600                       # e.g. a git checkout after the seal
    os.utime(f, (t, t))


class TestPreregDoc(unittest.TestCase):
    def setUp(self):
        self.df = _cont()
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)
        self.reg = self.root / ".onkayit"
        self.doc = self.root / "ONKAYIT.md"
        self.doc.write_text("# Plan\n\nours: good (pred_good), compared with naive and other.\n", encoding="utf-8")

    def tearDown(self):
        self.t.cleanup()

    def _seal(self, **kw):
        return ev.seal(self.df.drop(columns=["truth"]), "test predictions", registry=self.reg, store_copy=False, **kw)

    def test_registry_alone_does_not_see_the_doc(self):
        self._seal()
        s = _sel(ev.evaluate(self.df, ours="good", prereg=self.reg, **FAST))
        self.assertIn("ours not found", s["ayrinti"])

    def test_list_mixing_registry_and_doc(self):
        old = time.time() - 3600
        os.utime(self.doc, (old, old))
        self._seal()
        rep = ev.evaluate(self.df, ours="good", prereg=[self.reg, self.doc], **FAST)
        self.assertIn("önceden adlandırıldı", _sel(rep)["ayrinti"])
        self.assertEqual(len(ev.summary(rep)["seals"]), 1)
        docs = ev.summary(rep)["prereg_docs"]
        self.assertEqual([d["file"] for d in docs], ["ONKAYIT.md"])
        self.assertEqual(docs[0]["time_source"], "file mtime")
        self.assertIn("Pre-registration documents searched", ev.card_markdown(rep))
        # same through prereg_doc= and through string paths in a tuple
        rep2 = ev.evaluate(self.df, ours="good", prereg=str(self.reg), prereg_doc=str(self.doc), **FAST)
        self.assertIn("önceden adlandırıldı", _sel(rep2)["ayrinti"])
        rep3 = ev.evaluate(self.df, ours="good", prereg=(str(self.reg / "zincir.jsonl"), str(self.doc)), **FAST)
        self.assertIn("önceden adlandırıldı", _sel(rep3)["ayrinti"])

    def test_checkout_mtime_vs_lock_time(self):
        lock = research.lock_prereg(self.doc)
        self._seal()
        _future(self.doc)
        # without the lock the new mtime makes the doc look newer than the seal
        os.rename(str(self.doc) + ".lock.json", self.root / "moved.json")
        s = _sel(ev.evaluate(self.df, ours="good", prereg=self.reg, prereg_doc=self.doc, **FAST))
        self.assertIn("newer than the seal", s["ayrinti"])
        os.rename(self.root / "moved.json", str(self.doc) + ".lock.json")
        rep = ev.evaluate(self.df, ours="good", prereg=self.reg, prereg_doc=self.doc, **FAST)
        self.assertIn("önceden adlandırıldı", _sel(rep)["ayrinti"])
        d = ev.summary(rep)["prereg_docs"][0]
        self.assertEqual(d["time_source"], "lock")
        self.assertEqual(d["sha256"], lock["sha256"])
        self.assertFalse(d["lock_mismatch"])

    def test_changed_after_lock_warns_and_uses_mtime(self):
        research.lock_prereg(self.doc)
        self._seal()
        self.doc.write_text("ours: good, edited after the lock\n", encoding="utf-8")
        _future(self.doc)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = ev.evaluate(self.df, ours="good", prereg=[self.reg, self.doc], **FAST)
        self.assertTrue(any("differs from its lock" in str(x.message) for x in w))
        self.assertIn("newer than the seal", _sel(rep)["ayrinti"])
        d = ev.summary(rep)["prereg_docs"][0]
        self.assertTrue(d["lock_mismatch"])
        self.assertEqual(d["time_source"], "file mtime")
        self.assertTrue(any("kilidinden sonra değişmiş" in x for x in rep["sinirlar"]))
        self.assertIn("CHANGED AFTER LOCK", ev.card_markdown(rep))

    def test_seal_records_the_doc_and_evaluate_finds_it(self):
        rec = self._seal(prereg_doc=self.doc)
        sha = research.file_sha256(self.doc)
        self.assertEqual(rec["onkayit_belgesi"], {"dosya": "ONKAYIT.md", "sha256": sha})
        self.assertIn(f"ONKAYIT.md sha256:{sha}", rec["aciklama"])
        self.assertNotIn(str(self.root), json.dumps(rec["onkayit_belgesi"]))      # file name only
        self.assertTrue(ev.verify_chain(self.reg)["gecerli"])
        _future(self.doc)
        rep = ev.evaluate(self.df, ours="good", prereg=self.reg, **FAST)        # doc found next to the registry
        self.assertIn("önceden adlandırıldı", _sel(rep)["ayrinti"])
        self.assertEqual(ev.summary(rep)["prereg_docs"][0]["time_source"], "seal")
        # edited afterwards: the sha no longer matches the seal, the (future) mtime decides
        self.doc.write_text("ours: good, edited\n", encoding="utf-8")
        _future(self.doc)
        self.assertIn("newer than the seal", _sel(ev.evaluate(self.df, ours="good", prereg=self.reg, **FAST))["ayrinti"])
        with self.assertRaises(FileNotFoundError):
            self._seal(prereg_doc=self.root / "missing.md")

    def test_cli_belge(self):
        from simengine.evidence import onkayit
        f = self.root / "pred.csv"
        self.df.drop(columns=["truth"]).to_csv(f, index=False)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(onkayit.main(["--dizin", str(self.reg), "--json", "muhurle", str(f), "--aciklama", "x",
                                           "--belge", str(self.doc)]), 0)
        self.assertIn("onkayit_belgesi", ev.chain(self.reg)[-1])

    def test_missing_doc_and_backward_compatible_record_list(self):
        with self.assertRaises(FileNotFoundError):
            ev.evaluate(self.df, ours="good", prereg_doc=self.root / "nope.md", **FAST)
        self._seal()
        recs = ev.chain(self.reg)
        rep = ev.evaluate(self.df, ours="good", prereg=recs, **FAST)
        self.assertEqual(len(ev.summary(rep)["seals"]), 1)
        self.assertNotIn("prereg_docs", ev.summary(rep))
        # doc only (no seal): the text counts
        rep = ev.evaluate(self.df, ours="good", prereg_doc=self.doc, **FAST)
        self.assertIn("önceden adlandırıldı", _sel(rep)["ayrinti"])


# ------------------------------------------------------------------ split=
def _dist(n_items=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        for s in ("a", "b"):
            t = rng.dirichlet(np.ones(4))
            p = 0.7 * t + 0.3 * rng.dirichlet(np.ones(4))
            rows.append({"unit": f"{s}|q{i}", "truth": list(t), "pred_llm": list(p), "pred_flat": [0.25] * 4,
                         "segment": s, "item": f"q{i}", "group": f"q{i}"})
    return pd.DataFrame(rows)


class TestDeclaredSplit(unittest.TestCase):
    def test_distribution_frame(self):
        df = _dist()
        rep = _quiet(ev.evaluate, df, ours="llm", task="distribution", **FAST)
        self.assertEqual(_audit(rep, "bölme bilgisi")["durum"], "atlandi")
        rep = _quiet(ev.evaluate, df, ours="llm", task="distribution", split="test", **FAST)
        self.assertIsNone(_audit(rep, "bölme bilgisi"))
        self.assertEqual(_audit(rep, "grup çakışması")["durum"], "tamam")
        self.assertEqual(_audit(rep, "bölme beyanı")["durum"], "tamam")
        self.assertIn("bölme beyanı", ev.card_markdown(rep))
        self.assertNotIn("bolum", df.columns)                     # caller's frame untouched

    def test_holdout_and_errors(self):
        df = _cont()
        rep = ev.evaluate(df, ours="good", split="holdout", **FAST)
        self.assertEqual(_audit(rep, "bölme beyanı")["deger"], "holdout")
        with self.assertRaises(ValueError):
            ev.evaluate(df, ours="good", split="train", **FAST)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rep = ev.evaluate(df.assign(split="test"), ours="good", split="test", **FAST)
        self.assertTrue(any("already has a split column" in str(x.message) for x in w))
        self.assertIsNone(_audit(rep, "bölme beyanı"))


# ------------------------------------------------------------------ balance_col=
def _countries(dominant=True, seed=0):
    """Persons x condition, one dominant country; the dominant country's condition effects are shifted so the
    person-weighted and the country-equal-weighted condition means differ."""
    rng = np.random.default_rng(seed)
    sizes = {"US": 120, "DE": 30, "FR": 30, "TR": 30, "JP": 30, "BR": 30} if dominant else \
        {c: 45 for c in ("US", "DE", "FR", "TR", "JP", "BR")}
    rows = []
    for c, n in sizes.items():
        for k in range(10):
            eff = k / 10 + (0.8 * np.sin(k) if c == "US" else 0.0)
            for _ in range(n // 10):
                rows.append({"country": c, "cond": f"c{k}", "truth": eff + rng.normal(0, 0.5),
                             "pred_good": k / 10 + rng.normal(0, 0.05), "pred_naive": rng.normal(0, 1)})
    return pd.DataFrame(rows)


def _two_step(df, col):
    a = df.groupby("cond")[["truth", col]].mean()
    b = df.groupby(["cond", "country"])[["truth", col]].mean().groupby(level=0).mean()
    return float(np.corrcoef(a["truth"], a[col])[0, 1]), float(np.corrcoef(b["truth"], b[col])[0, 1])


class TestDominantBlock(unittest.TestCase):
    def test_aggregate_by_with_balance_col(self):
        df = _countries()
        rep = _quiet(ev.evaluate, df, ours="good", aggregate_by="cond", balance_col="country", **FAST)
        a = _audit(rep, "baskın blok")
        self.assertEqual(a["durum"], "uyari")
        self.assertAlmostEqual(a["deger"], 120 / 270)
        b = rep["baskin_blok"]
        self.assertEqual((b["blok"], b["n_blok"], b["agirlik"]), ("US", 6, "iki_adim"))
        pw, be = _two_step(df, "pred_good")
        self.assertAlmostEqual(b["degerler"]["good"]["kisi_agirlikli"], pw, places=9)
        self.assertAlmostEqual(b["degerler"]["good"]["blok_esit"], be, places=9)
        self.assertAlmostEqual(rep["manset"]["bizim"]["deger"], pw, places=9)   # = the reported value
        s = ev.summary(rep)["dominant_block"]
        self.assertTrue(s["dominant"])
        self.assertEqual(s["metric"], "pearson")
        md = ev.card_markdown(rep)
        self.assertIn("dominant block `country` = US holds 44.4%", md)
        self.assertIn(f"{be:.3f}", md)
        self.assertIn("| baskın blok (dominant block) | warning |", md)
        self.assertNotIn("Baskın blok:", md)                  # note shown once, not repeated in Limits
        self.assertTrue(any(x.startswith("Baskın blok:") for x in rep["kanit_kart"]["yanilgi"]))

    def test_balanced_explicit_ok_row_and_implicit_silence(self):
        df = _countries(dominant=False)
        rep = _quiet(ev.evaluate, df, ours="good", aggregate_by="cond", balance_col="country", **FAST)
        self.assertEqual(_audit(rep, "baskın blok")["durum"], "tamam")
        self.assertFalse(ev.summary(rep)["dominant_block"]["dominant"])
        self.assertNotIn("dominant block `", ev.card_markdown(rep))
        rep = _quiet(ev.evaluate, df, ours="good", aggregate_by="cond", **FAST)
        self.assertIsNone(_audit(rep, "baskın blok"))

    def test_row_level_weights(self):
        df = _countries()
        rep = _quiet(ev.evaluate, df, ours="good", balance_col="country", primary="mae", **FAST)
        b = rep["baskin_blok"]
        self.assertEqual(b["agirlik"], "agirlik_1_n_blok")
        err = (df["truth"] - df["pred_good"]).abs()
        w = 1 / df.groupby("country")["country"].transform("size")
        self.assertAlmostEqual(b["degerler"]["good"]["kisi_agirlikli"], float(err.mean()), places=9)
        self.assertAlmostEqual(b["degerler"]["good"]["blok_esit"], float((err * w).sum() / w.sum()), places=9)
        rep = _quiet(ev.evaluate, df, ours="good", balance_col="country", **FAST)          # pearson: per-block mean
        b = rep["baskin_blok"]
        self.assertEqual(b["agirlik"], "blok_ortalamasi")
        per = [np.corrcoef(g["truth"], g["pred_good"])[0, 1] for _, g in df.groupby("country")]
        self.assertAlmostEqual(b["degerler"]["good"]["blok_esit"], float(np.mean(per)), places=9)

    def test_block_col_triggers_check_and_threshold_scales(self):
        df = _countries().assign(wave=lambda x: np.where(x["country"] == "US", "w1", "w2"))
        # two blocks: 44 % is below 1.5 / 2 = 75 %, so no dominance; implicit (block_col) check stays silent
        rep = _quiet(ev.evaluate, df, ours="good", block_col="wave", **FAST)
        self.assertIsNone(_audit(rep, "baskın blok"))
        rep = _quiet(ev.evaluate, df, ours="good", block_col="country", **FAST)
        self.assertEqual(_audit(rep, "baskın blok")["durum"], "uyari")
        with self.assertRaises(ValueError):
            ev.evaluate(df, ours="good", balance_col="nope", **FAST)


if __name__ == "__main__":
    unittest.main()
