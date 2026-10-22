"""Evidence guardrails: English error messages (internal alias in parentheses), English primary-metric aliases
with a 'did you mean' suggestion, the few-cluster warning (audit row + no 'high confidence' badge on a one-test
A/B frame with a zero-width CI) and the all-NaN truth warning. Small synthetic data, offline."""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine import evidence as ev
from simengine.evidence import kart
from simengine.evidence.cekirdek import AZ_KUME, birincil_ic

KUME_KONTROL = "küme sayısı (bootstrap GA)"


def _binary(n=80, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    return pd.DataFrame({"group": [f"g{i}" for i in range(n)], "truth": t,
                         "pred_ours": np.clip(0.7 * t + 0.15 + rng.normal(0, .1, n), 0, 1),
                         "pred_raw": rng.uniform(0, 1, n)})


def _one_ab_test():
    # one test, five variants: every bootstrap draw resamples the same test -> CI [x, x]
    t = np.array([0.10, 0.12, 0.08, 0.15, 0.11])
    return pd.DataFrame({"item": ["t1"] * 5, "truth": t, "pred_llm": t + np.array([.001, -.002, .001, .003, 0]),
                         "pred_x": np.array([.11, .09, .12, .10, .13])})


def _evaluate(df, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rep = ev.evaluate(df, B=200, **kw)
    return rep, [str(x.message) for x in w]


class EnglishMessagesTest(unittest.TestCase):
    def test_missing_truth_column(self):
        with self.assertRaisesRegex(ValueError, r"'truth' column is required \(alias: gercek\)"):
            ev.paired_diff(_binary().drop(columns="truth"), "ours", "raw")
        with self.assertRaisesRegex(ValueError, r"'truth' column is required"):
            ev.prepare(_binary().drop(columns="truth"))

    def test_other_schema_messages(self):
        with self.assertRaisesRegex(ValueError, r"at least one 'pred_<method>' column"):
            ev.prepare(_binary()[["group", "truth"]])
        with self.assertRaisesRegex(ValueError, r"missing prediction columns: \['pred_nope'\]"):
            ev.prepare(_binary(), methods=["nope"])
        with self.assertRaisesRegex(ValueError, r"unknown task 'kedi'"):
            ev.prepare(_binary(), task="kedi")
        with self.assertRaisesRegex(ValueError, r"task 'ab' needs an 'item' column"):
            ev.prepare(_binary(), task="ab")


class MetricAliasTest(unittest.TestCase):
    def test_alias_table(self):
        for a in ("log_loss", "logloss", "Log-Loss", "logloss"):
            self.assertEqual(birincil_ic(a), "logloss")
        self.assertEqual(birincil_ic("accuracy"), "dogruluk")
        self.assertEqual(birincil_ic("balanced_accuracy"), "dengeli_dogruluk")
        self.assertEqual(birincil_ic("dogruluk"), "dogruluk")      # internal names still work
        self.assertIsNone(birincil_ic(None))

    def test_log_loss_accepted(self):
        rep, _ = _evaluate(_binary(), task="binary", primary="log_loss", ours="ours", raw="raw")
        self.assertEqual(rep["birincil"], "logloss")
        self.assertEqual(ev.summary(rep)["primary_metric"], "logloss")

    def test_unknown_metric_english_with_suggestion(self):
        with self.assertRaises(ValueError) as cm:
            ev.evaluate(_binary(), task="binary", primary="acuracy", ours="ours", B=50)
        msg = str(cm.exception)
        self.assertIn("primary metric 'acuracy' is not available for task 'binary'", msg)
        self.assertIn("'accuracy'", msg)                  # English names listed
        self.assertIn("did you mean 'accuracy'?", msg)
        self.assertNotIn("bu görevde yok", msg)


class FewClustersTest(unittest.TestCase):
    def test_one_test_ab_warns_and_no_high_badge(self):
        rep, w = _evaluate(_one_ab_test(), task="ab", ours="llm")
        self.assertTrue(any("only 1 tests; CI is not meaningful" in m for m in w), w)
        den = {d["kontrol"]: d for d in rep["denetim"]}
        self.assertEqual(den[KUME_KONTROL]["durum"], "uyari")
        self.assertEqual(den[KUME_KONTROL]["deger"], 1)
        k = rep["kanit_kart"]
        self.assertNotEqual(k["rozet"], "yuksek")
        self.assertEqual(k["rozet"], "kararsiz")
        self.assertIn("only 1 clusters/tests", k["rozet_gerekce"])
        self.assertNotEqual(ev.summary(rep)["badge"], "high confidence")

    def test_enough_clusters_no_row(self):
        rep, w = _evaluate(_binary(), task="binary", ours="ours", raw="raw")
        self.assertNotIn(KUME_KONTROL, {d["kontrol"] for d in rep["denetim"]})
        self.assertFalse(any("CI is not meaningful" in m for m in w))

    def test_badge_zero_width_and_few_clusters(self):
        S = [kart.satir("taban", "t", .5, .4, .6), kart.satir("kalibre", "b", .9, .9, .9)]
        sifir = {"fark": .4, "alt": .4, "ust": .4}
        genis = {"fark": .4, "alt": .3, "ust": .5}
        self.assertEqual(kart.rozet(S, genis, taban_var=True)[0], "yuksek")               # old behaviour kept
        self.assertEqual(kart.rozet(S, genis, taban_var=True, n_kume=AZ_KUME)[0], "yuksek")
        r, g = kart.rozet(S, genis, taban_var=True, n_kume=AZ_KUME - 1)
        self.assertEqual(r, "kararsiz")
        self.assertIn("< 10", g)
        r, g = kart.rozet(S, sifir, taban_var=True)
        self.assertEqual(r, "kararsiz")
        self.assertIn("zero-width CI", g)


class TruthAllNaNTest(unittest.TestCase):
    def test_all_nan_truth_warns(self):
        df = _binary().assign(truth=np.nan)        # e.g. a failed join on 'unit'
        rep, w = _evaluate(df, ours="ours")
        self.assertEqual(rep["kanit_kart"]["rozet"], "bekliyor")
        self.assertTrue(any("all 80 values are missing" in m and "check the join" in m for m in w), w)

    def test_no_truth_column_does_not_warn(self):
        rep, w = _evaluate(_binary().drop(columns="truth"), ours="ours")
        self.assertEqual(rep["kanit_kart"]["rozet"], "bekliyor")
        self.assertFalse(any("values are missing" in m for m in w), w)


if __name__ == "__main__":
    unittest.main()
