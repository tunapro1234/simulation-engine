"""Evidence core (D5 #5, D8 #8a/#8b/#8c, D7 #11, D7 #15, D12 #13): within-item pair accuracy (pair_sign) on
continuous tasks with an item column, persona shuffle and PPI++ marked not applicable on multiclass choice rows,
no CI warning on a zero-variance arm, a seal description is not a pre-registration document, the selection row
prints the frame's prediction-column count next to n_configs, ours declared as an OOF baseline is circular for
PPI++. Small synthetic frames, offline."""
import itertools
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from simengine import evidence as ev
from simengine.evidence import cekirdek

FAST = dict(B=200, B_aggregate=60, multiplier=False)
PERSONA = "persona karıştırma (birey bilgisi kullanılıyor mu)"
CARPAN = "etkin örneklem çarpanı (PPI++ kaç insana bedel)"
SECIM = "seçim yanlılığı"


def _ev(df, **kw):
    for k, v in FAST.items():
        kw.setdefault(k, v)
    kw.setdefault("ours", "ours")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ev.evaluate(df, "t", **kw)


def _den(rep):
    return {d["kontrol"]: d for d in rep["denetim"]}


def _effects(n_exp=30, per=4, seed=0):
    """D5 shape: one row per experiment x outcome effect; item = experiment."""
    rng = np.random.default_rng(seed)
    n = n_exp * per
    y = rng.normal(0, 1, n).round(2)
    return pd.DataFrame({"truth": y, "pred_ours": (y + rng.normal(0, 1, n)).round(1),
                         "pred_base": rng.normal(0, 1, n), "item": [f"e{i // per}" for i in range(n)]})


def _hand_pairs(df, col):
    ok = n = 0.0
    for _, g in df.groupby("item"):
        for (y1, p1), (y2, p2) in itertools.combinations(zip(g["truth"], g[col]), 2):
            if y1 == y2:
                continue
            n += 1
            ok += 0.5 if p1 == p2 else float((p1 > p2) == (y1 > y2))
    return ok / n


def _choice(n=300, K=3, seed=0, person=True):
    """D8 shape: one row per choice task, truth = chosen alternative, one probability vector per task."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(K) * 2, n)
    t = np.array([rng.choice(K, p=p) for p in P])
    d = pd.DataFrame({"truth": t, "pred_ours": list(P), "pred_flat": [np.ones(K) / K] * n})
    if person:
        d["group"] = [f"p{i // 5}" for i in range(n)]
    return d


class PairSignContinuousTest(unittest.TestCase):
    """D5 #5: the brief asked MAE + pair accuracy + Pearson; pair accuracy had to be written by hand."""

    def test_pair_sign_matches_hand_count(self):
        df = _effects()
        rep = _ev(df, task="continuous", primary="mae")
        m = rep["sonuclar"]["tahmin_ours"]["metrikler"]
        self.assertIn("cift_isaret", m)
        self.assertAlmostEqual(m["cift_isaret"]["deger"], _hand_pairs(df, "pred_ours"), places=12)
        self.assertTrue(np.isfinite(m["cift_isaret"]["alt"]) and np.isfinite(m["cift_isaret"]["ust"]))
        self.assertEqual(rep["birincil"], "mae")

    def test_secondary_unless_asked(self):
        df = _effects(seed=1)
        self.assertNotEqual(_ev(df, task="continuous")["birincil"], "cift_isaret")
        self.assertEqual(_ev(df, task="continuous", primary="pair_sign")["birincil"], "cift_isaret")

    def test_absent_without_item_or_on_one_row_items(self):
        df = _effects(seed=2)
        rep = _ev(df.drop(columns="item"), task="continuous")
        self.assertNotIn("cift_isaret", rep["sonuclar"]["tahmin_ours"]["metrikler"])
        one = _effects(n_exp=40, per=1, seed=3)
        self.assertNotIn("cift_isaret", _ev(one, task="continuous")["sonuclar"]["tahmin_ours"]["metrikler"])


class ChoiceRowsNotApplicableTest(unittest.TestCase):
    """D8 #8a/#8b: vectors over alternatives, one row per task: the shuffle moved vectors across tasks with
    different choice sets ('ok, fark 0.649') and PPI++ printed 'worth 139 humans' with no human answers."""

    def test_shuffle_and_multiplier_not_applicable(self):
        for person in (True, False):
            rep = _ev(_choice(person=person), task="multiclass", multiplier=True)
            den = _den(rep)
            self.assertTrue(den[PERSONA].get("uygulanamaz"), den.get(PERSONA))
            self.assertIn("choice", den[PERSONA]["ayrinti"])
            self.assertNotIn("tahmin karıştırma (persona takası değil)", den)
            self.assertTrue(den[CARPAN].get("uygulanamaz"))
            self.assertIn("choice", rep["carpan_uygulanamaz"])
            self.assertIsNone(rep.get("orneklem"))

    def test_explicit_level_task(self):
        d = _choice()
        d["item"] = "q"   # an item column alone would run the shuffle; level="task" says the rows are tasks
        rep = _ev(d, task="multiclass", level="task", multiplier=True)
        self.assertTrue(_den(rep)[PERSONA].get("uygulanamaz"))
        self.assertIn("carpan_uygulanamaz", rep)

    def test_one_question_with_item_keeps_shuffle(self):
        d = _choice()
        d["item"] = "q"   # persons answering the same question: the within-item shuffle is a persona test
        den = _den(_ev(d, task="multiclass"))
        self.assertIn("tahmin karıştırma (persona takası değil)", den)
        self.assertNotIn(PERSONA, den)

    def test_zero_variance_arm_no_ci_warning(self):
        # D8 #8c: a flat arm has the same log loss in every resample; the point must not read as outside its CI
        for n in (300, 2076):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                rep = ev.evaluate(_choice(n=n, seed=n), "t", ours="ours", task="multiclass", primary="logloss",
                                  B=200, B_aggregate=60, multiplier=False)
            self.assertFalse([x for x in w if "excludes the point" in str(x.message)])
            self.assertNotIn("ga_uyarisi", rep)


class SealDescriptionNotPreregTest(unittest.TestCase):
    """D7 #11: ours was named only in a seal description written after the dev run; the card said 'named in
    advance'."""

    def test_unit(self):
        kod, neden = cekirdek._onkayit_kaniti("good", [{"aciklama": "ours: pred_good"}], None)
        self.assertEqual(kod, "aciklama")
        self.assertIn("seal ≠ pre-registration", neden)
        self.assertEqual(cekirdek._onkayit_kaniti("good", [{"aciklama": "ours: good"}], ["ours: good"])[0],
                         "adlandirildi")
        self.assertTrue(cekirdek._onkayitta_geciyor("good", [{"aciklama": "ours: good"}], None))

    def test_selection_row(self):
        df = _effects().drop(columns="item")
        with tempfile.TemporaryDirectory() as t:
            reg = Path(t) / "reg"
            ev.seal(df.drop(columns="truth"), "ours: pred_ours (pool_lgbm)", registry=reg, store_copy=False)
            s = _den(_ev(df, task="continuous", prereg=reg))[SECIM]
            self.assertEqual(s["durum"], "uyari")
            self.assertIn("named only in a seal description, not in a pre-registration document", s["ayrinti"])
            self.assertIn("seal ≠ pre-registration", s["ayrinti"])


class NConfigsCountTest(unittest.TestCase):
    """D7 #15: the row printed the user's n_configs only."""

    def test_frame_count_next_to_n_configs(self):
        df = _effects().drop(columns="item")
        df["pred_third"] = df["truth"] + 0.3
        s = _den(_ev(df, task="continuous", n_configs=12))[SECIM]
        self.assertIn("frame has 3 prediction columns; n_configs=12 given", s["ayrinti"])
        self.assertEqual(s["n_configs"], 12)
        self.assertEqual(s["n_pred_columns"], 3)
        s = _den(_ev(df, task="continuous"))[SECIM]
        self.assertEqual(s["n_pred_columns"], 3)
        self.assertNotIn("n_configs=", s["ayrinti"])


class OursOofCircularTest(unittest.TestCase):
    """D12 #13: ours is a supervised model fitted on the same labels; declared among the OOF baselines it is
    circular for PPI++ the way circular= is."""

    def test_ours_in_oof_baselines(self):
        rng = np.random.default_rng(4)
        n = 300
        y = rng.normal(0, 1, n)
        df = pd.DataFrame({"truth": y, "pred_ours": y + rng.normal(0, .5, n), "pred_cell": y + rng.normal(0, 1, n),
                           "group": [f"p{i // 3}" for i in range(n)]})
        rep = _ev(df, task="continuous", baselines_oof=["ours", "cell"], multiplier=True)
        den = _den(rep)
        self.assertIn("döngüsel önsel (çarpan dışı)", den)
        self.assertNotIn("tahmin_ours", (rep.get("orneklem") or {}).get("yontemler", {}))


if __name__ == "__main__":
    unittest.main()
