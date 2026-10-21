"""English API of simengine.evidence on small synthetic data (offline, FakeLLM only)."""
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import simengine.evidence as ev
from simengine.llm import FakeLLM


def _frame(n=400, seed=0):
    """Binary EvalFrame in the English convention, with a train/test split and a person cluster."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (x + rng.normal(0, 0.8, n) > 0).astype(int)
    return pd.DataFrame({
        "unit": np.arange(n), "group": [f"p{i}" for i in range(n)],
        "split": np.where(np.arange(n) < 100, "train", "test"), "segment": rng.choice(["A", "B"], n),
        "truth": y, "pred_llm": 1 / (1 + np.exp(-0.4 * x)), "pred_hybrid": 1 / (1 + np.exp(-2 * x)),
    })


def _items(P=50, M=20, seed=1):
    rng = np.random.default_rng(seed)
    kisi, madde = np.repeat(np.arange(P), M), np.tile(np.arange(M), P)
    mu = rng.normal(0, 1, M)
    y = mu[madde] + rng.normal(0, 1, P * M)
    return pd.DataFrame({"group": kisi, "item": madde, "truth": y,
                         "pred_good": y + rng.normal(0, 0.5, P * M), "pred_flat": mu[madde] + 0.0,
                         "pred_noise": rng.normal(0, 1, P * M)})


class TestNames(unittest.TestCase):
    def test_english_and_turkish_frames_match(self):
        en = _frame()
        tr = en.rename(columns={"unit": "birim", "group": "grup", "split": "bolum", "truth": "gercek",
                                "pred_llm": "tahmin_llm", "pred_hybrid": "tahmin_hybrid"})
        tr["bolum"] = tr["bolum"].map({"train": "egitim", "test": "test"})
        pd.testing.assert_frame_equal(ev.to_internal(en), ev.to_internal(tr))
        self.assertIn("tahmin_llm", ev.to_internal(en).columns)
        self.assertEqual(set(ev.to_internal(en)["bolum"]), {"egitim", "test"})

    def test_extra_columns(self):
        d = pd.DataFrame({"truth": [1, 0], "pred_a": [0.6, 0.4], "reverse_a": [0.4, 0.6], "weight": [1, 2],
                          "fold": [0, 1], "arm": ["x", "y"]})
        self.assertEqual(list(ev.to_internal(d).columns), ["gercek", "tahmin_a", "ters_a", "agirlik", "kat", "kol"])


class TestEvaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = ev.evaluate(_frame(), title="synthetic binary", ours="hybrid", raw="pred_llm", B=200,
                              B_aggregate=40, multiplier=False)

    def test_report_schema(self):
        r = self.rep
        for k in ("baslik", "gorev", "birincil", "sonuclar", "merdiven", "manset", "karsilastirmalar", "kalibrasyon",
                  "segment", "denetim", "kanit_kart", "sinirlar"):
            self.assertIn(k, r)
        self.assertEqual(r["gorev"], "ikili")
        self.assertEqual(r["kart"]["n_satir"], 300)          # metrics on test rows only
        self.assertIn("taban_marjinal", r["sonuclar"])
        kinds = [x["tur"] for x in r["merdiven"]]
        self.assertEqual(kinds[0], "baseline")
        self.assertTrue(r["manset"]["bizim"]["alt"] <= r["manset"]["bizim"]["deger"] <= r["manset"]["bizim"]["ust"])
        json.dumps(r)                                        # JSON-serialisable (NaN -> null)

    def test_summary_english(self):
        s = ev.summary(self.rep)
        self.assertEqual(s["task"], "binary")
        self.assertIn(s["primary_metric"], ("accuracy", "macro_f1"))
        self.assertEqual(s["ours"]["column"], "tahmin_hybrid")
        self.assertIn(s["badge"], ("high confidence", "inconclusive", "below baseline", "awaiting truth"))
        statuses = {a["check"]: a["status"] for a in s["audits"]}
        self.assertEqual(statuses["grup çakışması (eğitim ∩ test)"], "ok")
        self.assertIsNone(s["ceiling"])                      # no retest column -> no ceiling

    def test_card_and_dashboard(self):
        with tempfile.TemporaryDirectory() as d:
            md = ev.card(self.rep, Path(d) / "card.md")
            self.assertTrue(md.startswith("# Evidence card: synthetic binary"))
            self.assertIn("| Ours |", md)
            self.assertIn("## Baseline ladder", md)
            self.assertEqual((Path(d) / "card.md").read_text(), md)
            k = ev.card(self.rep, Path(d) / "card.json")
            self.assertEqual(k, json.loads((Path(d) / "card.json").read_text()))
            self.assertEqual([r["tur"] for r in k["satirlar"]], ["taban", "ham", "kalibre", "tavan", "bosluk", "bedel"])
            p = ev.dashboard([self.rep, self.rep], Path(d) / "pano.html", title="Evidence")
            h = p.read_text()
            self.assertTrue(h.startswith("<!doctype html>"))
            self.assertNotRegex(h, r"<script[^>]+src=")      # single file, no external scripts
            self.assertIn('id="ozet-2"', h)

    def test_memorization_and_prereg_flow_into_report(self):
        d = _frame(300, seed=3)
        with tempfile.TemporaryDirectory() as t:
            ev.seal(d.drop(columns="truth"), "before truth", Path(t) / "reg")
            rep = ev.evaluate(d, "x", ours="hybrid", B=100, B_aggregate=20, multiplier=False,
                              memorization={"durum": "tamam", "ayrinti": "no memorization signal", "deger": 0.0},
                              prereg=Path(t) / "reg")
        self.assertEqual(len(rep["onkayit"]), 1)
        mem = next(a for a in rep["denetim"] if a["kontrol"].startswith("ezber"))
        self.assertEqual(mem["durum"], "tamam")

    def test_selection_bias_warning_when_ours_missing(self):
        rep = ev.evaluate(_frame(200), "x", B=50, B_aggregate=10, multiplier=False)
        sb = next(a for a in rep["denetim"] if a["kontrol"] == "seçim yanlılığı")
        self.assertEqual(sb["durum"], "uyari")


class TestPieces(unittest.TestCase):
    def test_worth_humans_pairs_and_frame(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=1500)
        f = y + rng.normal(0, 0.5, 1500)                     # rho ~ 0.89 -> theory ~ 4.9
        q = ev.worth_humans((y, f), n=100, B=40)
        self.assertGreater(q["carpan"], 3)
        self.assertLess(abs(q["carpan"] / q["teorik"] - 1), 0.3)
        self.assertLessEqual(q["carpan_ga"][0], q["carpan"])
        pts = ev.worth_humans(_items(), n=10, B=40, circular=["flat"])
        self.assertEqual([p["onsel"] for p in pts], ["good", "noise"])
        good, noise = pts
        self.assertGreater(good["carpan"], 2)
        self.assertLess(noise["carpan"], 1.2)
        self.assertEqual(good["ga_turu"], "sütun bootstrap")  # 20 items -> item bootstrap
        curve = ev.worth_theory_curve()
        self.assertAlmostEqual(curve[0]["carpan"], 1.0)

    def test_sample_multiplier(self):
        o = ev.sample_multiplier(_items(), ["good"], n_list=(10, 25), B=50)
        r = next(x for x in o["yontemler"]["tahmin_good"]["ppi"] if x["n"] == 10)
        self.assertGreater(r["carpan"], 2)

    def test_paired_ci_and_metrics(self):
        d = _items()
        r = ev.paired_ci(d, "good", "noise", metric="pearson", B=200, B_aggregate=50)
        self.assertTrue(r["a_better"])
        self.assertGreater(r["lo"], 0)
        self.assertEqual(r["cluster"], "grup")
        m = ev.compute_metrics(d)
        self.assertGreater(m["good"]["pearson"], m["noise"]["pearson"])
        self.assertIn("within_item_corr", m["good"])
        self.assertLess(m["flat"]["demeaned_r2"], 0.01)       # item mean carries no person signal

    def test_hit_at_k(self):
        truth = [0, 1, 0, 0, 0, 1, 1, 0, 0]
        score = [0.9, 0.8, 0.1, 0.5, 0.4, 0.3, 0.2, 0.9, 0.95]
        item = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        self.assertEqual(ev.hit_at_k(truth, score, item, k=(1, 2, 3)), {"hit@1": 0.0, "hit@2": 1 / 3, "hit@3": 1.0})

    def test_baselines_calibration_audits(self):
        d = _frame(300)
        b = ev.baseline_ladder(d)
        self.assertEqual(set(b), {"rastgele", "marjinal", "demografi"})
        self.assertEqual(len(b["marjinal"][1]), 200)
        cal = ev.recalibrate(d, "llm")
        self.assertEqual(set(cal), {"platt", "izotonik"})
        rel = ev.reliability(d, "llm")
        self.assertEqual(len(rel["kapsam"]), 10)
        bad = d.copy()
        bad.loc[0, "group"] = bad.loc[150, "group"]          # a person in train and test
        a = {x["kontrol"]: x for x in ev.split_audit(bad)}
        self.assertEqual(a["grup çakışması (eğitim ∩ test)"]["durum"], "hata")

    def test_memorization_audit_with_fake_llm(self):
        items = [{"prompt": f"real {i}", "expected": "yes", "control": False} for i in range(40)] + \
                [{"prompt": f"ctrl {i}", "expected": "yes", "control": True} for i in range(40)]
        llm = FakeLLM(lambda s, u: "yes" if u.startswith("real") or int(u.split()[1]) % 2 else "no")
        r = ev.memorization_audit(items, llm, B=200)
        self.assertEqual(r["durum"], "uyari")
        self.assertAlmostEqual(r["deger"], 0.5)
        self.assertEqual(len(llm.calls), 80)
        with self.assertRaises(ValueError):
            ev.memorization_audit(items, llm, max_calls=10)

    def test_seal_verify_grade(self):
        pred = pd.DataFrame({"id": [1, 2, 3], "pred_x": [0.1, 0.5, 0.9]})
        with tempfile.TemporaryDirectory() as t:
            k = ev.seal(pred, "first", t)
            self.assertTrue(ev.verify(pred.iloc[::-1], registry=t)["eslesti"])
            self.assertTrue(ev.verify_chain(t)["gecerli"])
            g = ev.grade(pred, pd.DataFrame({"id": [1, 2, 3], "truth": [0, 1, 1]}), ["id"], t)
            self.assertEqual(len(g["tablo"]), 3)
            self.assertEqual(ev.chain(t)[0]["sha256"], k["sha256"])
            changed = pred.assign(pred_x=[0.2, 0.5, 0.9])
            self.assertFalse(ev.verify(changed, registry=t)["eslesti"])

    def test_no_machine_paths_or_subprocess_llm(self):
        root = Path(ev.__file__).parent
        for p in root.glob("*.py"):
            s = p.read_text()
            self.assertIsNone(re.search(r"['\"]/(srv|data)/", s), p.name)
            self.assertNotIn('"claude", "-p"', s, p.name)


if __name__ == "__main__":
    unittest.main()
