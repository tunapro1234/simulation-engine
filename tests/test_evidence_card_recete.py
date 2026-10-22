"""Evidence card fixes from the recipe rehearsals (G1/G5/G6/G7): setting-B marginal baseline kept out of the badge,
registered vs post-hoc compare rows, weak-baseline warning on a registered row, shuffle-control scope (constant
within item / multiclass), per-arm filled-share audit. Offline: small synthetic tables only."""
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import simengine.evidence as ev
from simengine.evidence import cekirdek
from simengine.evidence.api import summary, to_internal

FAST = dict(B=200, B_aggregate=100, multiplier=False)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _dist(n_item=30, n_cell=8, K=4, seed=0, filled=None):
    """Question x cell distributions: truth = item profile + small cell shift. ours sees a noisy item profile
    shrunk toward uniform (better than uniform, worse than the leave-one-out item marginal)."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_item):
        base = rng.dirichlet(np.ones(K) * 2)
        for c in range(n_cell):
            t = np.clip(base + rng.normal(0, 0.02, K), 1e-3, None)
            t /= t.sum()
            p = 0.5 * np.clip(base + rng.normal(0, 0.08, K), 1e-3, None) + 0.5 / K
            row = {"unit": f"c{c}|q{q}", "truth": list(t / t.sum()), "pred_ours": list(p / p.sum()),
                   "item": f"q{q}", "group": f"q{q}"}
            if filled is not None:
                row["filled_ours"] = filled
            rows.append(row)
    return pd.DataFrame(rows)


def _binary(n_item=40, n_unit=15, seed=1):
    rng = np.random.default_rng(seed)
    item = np.repeat(np.arange(n_item), n_unit)
    x = rng.normal(0, 1, len(item))
    a = rng.normal(0, 1, n_item)[item]
    y = (rng.random(len(item)) < 1 / (1 + np.exp(-(a + 1.5 * x)))).astype(int)
    sig = lambda z: 1 / (1 + np.exp(-z))
    return pd.DataFrame({"item": item, "unit": np.arange(len(item)), "truth": y,
                         "pred_ours": sig(a + 1.4 * x), "pred_base": sig(a + 0.3 * x),
                         "pred_t1": sig(0.2 * x), "pred_t2": sig(a + 0.8 * x), "pred_t3": sig(0.5 * a),
                         "pred_flat": sig(a)})


class SettingBMarginal(unittest.TestCase):
    """G5#10, G7#6: taban_marjinal reads other cells' answers to the target question (setting B)."""

    @classmethod
    def setUpClass(cls):
        cls.r = _quiet(ev.evaluate, _dist(), ours="ours", task="distribution", **FAST)

    def test_ladder_marks_information_level(self):
        md = {r["sutun"]: r for r in self.r["merdiven"]}
        self.assertEqual(md["taban_marjinal"].get("bilgi"), "B")
        self.assertIn("setting B", md["taban_marjinal"]["etiket"])
        self.assertNotIn("bilgi", md["taban_rastgele"])
        lab = {x["column"]: x["label"] for x in summary(self.r)["ladder"]}
        self.assertIn("setting B", lab["taban_marjinal"])

    def test_badge_against_setting_a_baseline(self):
        r = self.r
        self.assertGreater(r["sonuclar"]["taban_marjinal"]["metrikler"][r["birincil"]]["deger"],
                           r["sonuclar"]["tahmin_ours"]["metrikler"][r["birincil"]]["deger"])
        self.assertEqual(r["manset"]["en_iyi_baseline"]["sutun"], "taban_rastgele")
        self.assertEqual(r["bilgi_duzeyi"]["B_ustte"], ["taban_marjinal"])
        s = summary(r)
        self.assertFalse(s["below_baseline"])
        k = r["kanit_kart"]
        self.assertIn("setting-B", k["satirlar"][0]["not"])
        self.assertTrue(any("ayar-B" in e for e in k["etiketler"]))
        d = next(x for x in r["denetim"] if x["kontrol"].startswith("bilgi düzeyi"))
        self.assertEqual(d["durum"], "uyari")
        m = next(x for x in r["denetim"] if x["kontrol"] == "madde ortalamasını geçiyor mu")
        self.assertEqual(m.get("bilgi"), "B")
        self.assertIn("Ayar-B basamağı", ev.dashboard_html([r]))

    def test_non_distribution_unchanged(self):
        r = _quiet(ev.evaluate, _binary(), ours="ours", methods=["base", "ours"], **FAST)
        self.assertNotIn("bilgi_duzeyi", r)
        self.assertFalse(any(x.get("bilgi") for x in r["merdiven"]))


class FilledShare(unittest.TestCase):
    """G7#7: share of answers filled with a uniform vector, per arm."""

    def test_audit_row_and_card_flag(self):
        r = _quiet(ev.evaluate, _dist(filled=0.2), ours="ours", task="distribution", **FAST)
        d = next(x for x in r["denetim"] if x["kontrol"].startswith("dolgu payı"))
        self.assertEqual(d["durum"], "uyari")
        self.assertAlmostEqual(d["deger"], 0.2)
        self.assertAlmostEqual(r["dolgu"]["kollar"]["tahmin_ours"]["ortalama"], 0.2)
        self.assertTrue(any(e.startswith("dolgu payı yüksek") for e in r["kanit_kart"]["etiketler"]))

    def test_low_share_ok_and_absent_column(self):
        r = _quiet(ev.evaluate, _dist(filled=0.0), ours="ours", task="distribution", **FAST)
        d = next(x for x in r["denetim"] if x["kontrol"].startswith("dolgu payı"))
        self.assertEqual(d["durum"], "tamam")
        self.assertFalse(any(e.startswith("dolgu payı") for e in r["kanit_kart"]["etiketler"]))
        r2 = _quiet(ev.evaluate, _dist(), ours="ours", task="distribution", **FAST)
        self.assertNotIn("dolgu", r2)
        self.assertFalse(any(x["kontrol"].startswith("dolgu payı") for x in r2["denetim"]))


class ShuffleScope(unittest.TestCase):
    """G1#10 (arm without person information), G6#7 (multiclass: prediction shuffle, not persona swap)."""

    def test_constant_within_item_skipped(self):
        r = _quiet(ev.evaluate, _binary(), ours="flat", methods=["base", "flat"], **FAST)
        self.assertNotIn("kontrol_karistir_flat", r["sonuclar"])
        d = next(x for x in r["denetim"] if x["kontrol"].startswith("persona karıştırma"))
        self.assertEqual(d["durum"], "atlandi")
        self.assertIn("constant within item", d["ayrinti"])
        self.assertFalse(any(x["sutun"].startswith("kontrol_") for x in r["merdiven"]))

    def test_varying_arm_keeps_control(self):
        r = _quiet(ev.evaluate, _binary(), ours="ours", methods=["base", "ours"], **FAST)
        self.assertIn("kontrol_karistir_ours", r["sonuclar"])
        d = next(x for x in r["denetim"] if x["kontrol"].startswith("persona karıştırma"))
        self.assertNotEqual(d["durum"], "atlandi")

    def test_multiclass_labelled_prediction_shuffle(self):
        rng = np.random.default_rng(3)
        n_item, n = 20, 10
        y = rng.integers(0, 4, n_item * n)
        pred = np.where(rng.random(len(y)) < 0.7, y, rng.integers(0, 4, len(y)))
        df = pd.DataFrame({"item": np.repeat(np.arange(n_item), n), "unit": np.arange(len(y)),
                           "truth": [f"k{v}" for v in y], "pred_ours": [f"k{v}" for v in pred]})
        r = _quiet(ev.evaluate, df, ours="ours", task="multiclass", **FAST)
        self.assertIn("persona takası değil", r["sonuclar"]["kontrol_karistir_ours"]["etiket"])
        d = next(x for x in r["denetim"] if x["kontrol"].startswith("tahmin karıştırma"))
        self.assertIn('compare=(ours, swap, "persona swap")', d["ayrinti"])
        self.assertFalse(any(x["kontrol"].startswith("persona karıştırma") for x in r["denetim"]))


class RegisteredCompare(unittest.TestCase):
    """G7#9 (post-hoc rows labelled registered), G6#8 (registered b weaker than the strongest baseline)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pre = Path(self.tmp.name) / "prereg.md"
        self.pre.write_text("ours: ours\nC1 = ours − base (paired)\nC2: ours − t3 (combined baseline)\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _eval(self, **kw):
        return _quiet(ev.evaluate, _binary(), ours="ours", methods=["base", "t1", "t2", "t3", "ours"],
                      unlabeled_baselines=["t1", "t2", "t3"], **FAST, **kw)

    def test_posthoc_rows_detected_from_prereg_text(self):
        r = self._eval(prereg=str(self.pre), compare=[("ours", "base"), ("ours", "t1", "late add")])
        a, b = r["kayitli_karsilastirmalar"]
        self.assertEqual((a["tur"], a["kayit"]), ("kayitli", "kayitli"))
        self.assertEqual((b["tur"], b["kayit"]), ("posthoc", "sonradan"))
        self.assertIn("post hoc", b["etiket"])
        kinds = [c["kind"] for c in summary(r)["comparisons"]]
        self.assertEqual(kinds, ["registered", "posthoc"])
        self.assertIn("sonradan · late add", ev.dashboard_html([r]))
        self.assertTrue(any(e.startswith("sonradan karşılaştırma") for e in r["kanit_kart"]["etiketler"]))

    def test_label_match_counts_as_registered(self):
        r = self._eval(prereg=str(self.pre), compare=[("ours", "t1", "C1")])
        self.assertEqual(r["kayitli_karsilastirmalar"][0]["kayit"], "kayitli")

    def test_no_prereg_text_keeps_old_rows(self):
        r = self._eval(compare=[("ours", "base")])
        row = r["kayitli_karsilastirmalar"][0]
        self.assertEqual(row["tur"], "kayitli")
        self.assertNotIn("kayit", row)

    def test_weak_baseline_warning(self):
        r = self._eval(prereg=str(self.pre), compare=[("ours", "t3")])
        strongest = r["manset"]["en_iyi_baseline"]["sutun"]
        self.assertEqual(strongest, "tahmin_t2")
        row = r["kayitli_karsilastirmalar"][0]
        self.assertEqual(row["kayit"], "kayitli")
        self.assertEqual(row["zayif_taban"]["en_guclu"], "tahmin_t2")
        d = next(x for x in r["denetim"] if x["kontrol"] == "kayıtlı karşılaştırma zayıf tabana karşı")
        self.assertEqual(d["durum"], "uyari")
        self.assertIn("t2", d["ayrinti"])
        self.assertTrue(any("zayıf tabana karşı" in e for e in r["kanit_kart"]["etiketler"]))
        r2 = self._eval(compare=[("ours", "t2")])
        self.assertNotIn("zayif_taban", r2["kayitli_karsilastirmalar"][0])

    def test_explicit_posthoc_in_degerlendir(self):
        d = to_internal(_binary())
        r = _quiet(cekirdek.degerlendir, d, "t", gorev="ikili", yontemler=["base", "ours"], bizim="ours", B=100,
                   B_toplu=50, carpan=False,
                   kiyaslar=[("ours", "base", "x", True), {"a": "ours", "b": None, "label": "y", "posthoc": False}])
        a, b = r["kayitli_karsilastirmalar"]
        self.assertEqual(a["tur"], "posthoc")
        self.assertEqual(b["kayit"], "kayitli")


if __name__ == "__main__":
    unittest.main()
