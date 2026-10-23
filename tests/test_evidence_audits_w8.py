"""Evidence audits (D13 #11e, G10 #7, G3 #10, G3 #11, G2 #11, G0 #4a): the memorization caveat only on LLM arms,
a named − blind compare pair read as the memorization check, no doubled card-row prefix, mixed-format dates, a
constant declared baseline printed as undefined, no shuffle row on one-row items. Small synthetic frames, offline."""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine import evidence as ev
from simengine.evidence import kart

EZBER = "ezber testi (LLM veriyi biliyor mu)"
KONT = "kontaminasyon (kesim öncesi/sonrası)"
WARN = {"durum": "uyari", "ayrinti": "korelasyon denetimi: adlı 0,78, adsız 0,17; fark +0,61 [+0,49, +0,72]",
        "deger": 0.61, "alt": 0.49, "ust": 0.72}
FAST = dict(B=200, B_aggregate=60, multiplier=False)


def _ev(df, **kw):
    for k, v in FAST.items():
        kw.setdefault(k, v)
    kw.setdefault("task", "continuous")
    kw.setdefault("ours", "ours")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ev.evaluate(df, "t", **kw)


def _den(rep):
    return {d["kontrol"]: d for d in rep["denetim"]}


def _arms(n=400, seed=0):
    """ours (tabular model) clearly beats the declared baseline; llm is a named LLM arm, llm_blind its blind copy
    (much worse: the named gain is recognition)."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, n)
    return pd.DataFrame({"truth": y, "pred_base": y + rng.normal(0, 1.5, n), "pred_ours": y + rng.normal(0, .4, n),
                         "pred_llm": y + rng.normal(0, .5, n), "pred_llm_blind": rng.normal(0, 1, n),
                         "pred_ours_blind": y + rng.normal(0, .45, n)})


class MemorizationFlagLlmArmTest(unittest.TestCase):
    """D13 #11e: ours was logistic + LightGBM, the LLM was only the raw= arm; the caveat stuck to ours."""

    def test_raw_llm_arm_alone_does_not_caveat_ours(self):
        r = _ev(_arms(), raw="llm", unlabeled_baselines=["base"], memorization=WARN)
        self.assertIsNone(r.get("ezber_uyarisi"))
        self.assertNotIn(kart.EZBER_EKI, r["kanit_kart"]["rozet_gerekce"])
        self.assertIsNone(ev.summary(r)["badge_caveat"])
        self.assertEqual(_den(r)[EZBER]["durum"], "uyari")      # the audit itself still reports the warning
        d = _den(r)["ezber uyarısı ve bizim kol (memorization warning and ours)"]
        self.assertIn("ours_uses_llm=True", d["ayrinti"])
        self.assertIn("ours_uses_llm=True", ev.card(r, lang="en"))

    def test_declared_or_raw_ours_keeps_caveat(self):
        r = _ev(_arms(), raw="llm", unlabeled_baselines=["base"], memorization=WARN, ours_uses_llm=True)
        self.assertIn(kart.EZBER_EKI, r["kanit_kart"]["rozet_gerekce"])
        r = _ev(_arms(), raw="ours", unlabeled_baselines=["base"], memorization=WARN)
        self.assertIn(kart.EZBER_EKI, r["kanit_kart"]["rozet_gerekce"])

    def test_ours_named_blind_pair_marks_ours_as_llm(self):
        r = _ev(_arms(), raw="llm", unlabeled_baselines=["base"], memorization=WARN,
                compare=[("ours", "ours_blind")])
        self.assertIn(kart.EZBER_EKI, r["kanit_kart"]["rozet_gerekce"])

    def test_no_llm_arm_named_keeps_legacy_default(self):
        r = _ev(_arms(), unlabeled_baselines=["base"], memorization=WARN)
        self.assertIn(kart.EZBER_EKI, r["kanit_kart"]["rozet_gerekce"])


class NamedBlindPairTest(unittest.TestCase):
    """G10 #7: ours_uses_llm=False with a named − blind compare pair said 'no LLM arm: not applicable'."""

    def test_pair_is_the_memorization_check(self):
        r = _ev(_arms(), unlabeled_baselines=["base"], ours_uses_llm=False, compare=[("llm", "llm_blind")])
        d = _den(r)[EZBER]
        self.assertFalse(d.get("uygulanamaz"))
        self.assertEqual(d["durum"], "uyari")                  # named arm far better than blind: recognition
        self.assertIn("named − blind", d["ayrinti"])
        self.assertIn("recognition/memorization suspected", ev.card(r, lang="en"))
        self.assertFalse(_den(r)[KONT].get("uygulanamaz"))
        for lang in ("tr", "mixed", "en"):
            self.assertNotIn("no LLM arm", ev.card(r, lang=lang))
        self.assertIsNone(r.get("ezber_uyarisi"))              # ours declared LLM-free: no caveat on ours

    def test_pair_without_gap_is_ok(self):
        df = _arms()
        df["pred_llm_blind"] = df["pred_llm"] + np.random.default_rng(3).normal(0, .05, len(df))
        d = _den(_ev(df, unlabeled_baselines=["base"], compare=[("llm", "llm_blind")]))[EZBER]
        self.assertEqual(d["durum"], "tamam")

    def test_memorization_result_wins_over_pair(self):
        r = _ev(_arms(), unlabeled_baselines=["base"], compare=[("llm", "llm_blind")], memorization=WARN)
        self.assertEqual(_den(r)[EZBER]["ayrinti"].split(" · ")[0], WARN["ayrinti"])

    def test_raw_llm_arm_with_ours_false_not_applicable_absent(self):
        r = _ev(_arms(), raw="llm", unlabeled_baselines=["base"], ours_uses_llm=False)
        for lang in ("tr", "mixed", "en"):
            self.assertNotIn("no LLM arm", ev.card(r, lang=lang))

    def test_plain_pair_is_not_a_blind_pair(self):
        r = _ev(_arms(), unlabeled_baselines=["base"], ours_uses_llm=False, compare=[("ours", "base")])
        self.assertTrue(_den(r)[EZBER].get("uygulanamaz"))


class CardPrefixTest(unittest.TestCase):
    """G3 #10: a label already carrying the row prefix printed 'Bizim: Bizim: ...'."""

    def test_no_doubled_prefix(self):
        r = _ev(_arms(), raw="llm", unlabeled_baselines=["base"],
                labels={"ours": "Bizim: yığın", "llm": "Ham: LLM", "base": "En güçlü basit taban: kalıcılık"})
        et = [s.get("etiket") or "" for s in r["kanit_kart"]["satirlar"]]
        self.assertIn("Bizim: yığın", et)
        self.assertIn("Ham: LLM", et)
        self.assertIn("En güçlü basit taban: kalıcılık", et)
        for x in et:
            for p in ("Bizim: ", "Ham: ", "En güçlü basit taban: "):
                self.assertFalse(x.startswith(p + p), x)
        self.assertNotIn("Bizim: Bizim:", ev.card(r, lang="tr"))

    def test_plain_label_gets_prefix(self):
        r = _ev(_arms(), raw="llm", unlabeled_baselines=["base"], labels={"ours": "yığın"})
        self.assertIn("Bizim: yığın", [s.get("etiket") for s in r["kanit_kart"]["satirlar"]])


def _mixed_dates(n=60, seed=0):
    """Four blocks of n rows: ISO / day-first strings, before / after the 2024-01-01 cutoff. ISO rows are accurate,
    slash rows noisy, so before and after are alike; reading the slash rows as 'after' fakes a memorization gap."""
    rng = np.random.default_rng(seed)
    parts = []
    for fmt, yil, sd in (("iso", 2023, .2), ("slash", 2023, 1.5), ("iso", 2024, .2), ("slash", 2024, 1.5)):
        days = pd.Timestamp(f"{yil}-02-13") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
        t = rng.normal(0, 1, n)
        parts.append(pd.DataFrame({"time": days.strftime("%Y-%m-%d" if fmt == "iso" else "%d/%m/%Y"), "truth": t,
                                   "pred_ours": t + rng.normal(0, sd, n), "pred_base": t + rng.normal(0, 2, n)}))
    return pd.concat(parts, ignore_index=True)


class MixedDatesTest(unittest.TestCase):
    """G3 #11: mixed-format dates were coerced to NaT and counted as 'after the cutoff'."""

    def test_no_false_memorization_warning(self):
        r = _ev(_mixed_dates(), unlabeled_baselines=["base"], cutoff="2024-01-01", primary="mae")
        d = _den(r)[KONT]
        self.assertEqual(d["durum"], "tamam", d["ayrinti"])
        self.assertNotIn("zaman sütunu çözümü (time column parsing)", _den(r))

    def test_unparsed_rows_counted_not_guessed(self):
        df = _mixed_dates()
        df.loc[:9, "time"] = "not a date"
        r = _ev(df, unlabeled_baselines=["base"], cutoff="2024-01-01", primary="mae")
        d = _den(r)["zaman sütunu çözümü (time column parsing)"]
        self.assertEqual(d["deger"], 10)
        self.assertIn("unparsed time: 10 rows left out", ev.card(r, lang="en"))
        self.assertEqual(_den(r)[KONT]["durum"], "tamam")

    def test_time_leak_seen_through_mixed_formats(self):
        rng = np.random.default_rng(1)
        n = 80
        y = rng.normal(0, 1, 2 * n)
        tr = ["2023-03-%02d" % (1 + i % 28) for i in range(n - 1)] + ["20/09/2023"]    # last train row after test
        te = ["2023-07-%02d" % (1 + i % 28) for i in range(n)]
        df = pd.DataFrame({"time": tr + te, "split": ["train"] * n + ["test"] * n, "truth": y,
                           "pred_ours": y + rng.normal(0, .5, 2 * n), "pred_base": y + rng.normal(0, 1.5, 2 * n)})
        d = _den(_ev(df, unlabeled_baselines=["base"]))["zaman sızıntısı (eğitim testten sonra mı)"]
        self.assertEqual(d["durum"], "uyari")
        self.assertIn("2023-09-20", d["ayrinti"])


class ConstantBaselineTest(unittest.TestCase):
    """G2 #11: a constant declared baseline has no Spearman; it vanished from the card without a word."""

    def test_constant_baseline_rank_metric_undefined(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=300)
        df = pd.DataFrame({"truth": y, "pred_ours": y + rng.normal(0, 1, 300), "pred_flat": np.full(300, .3)})
        r = _ev(df, primary="spearman", unlabeled_baselines=["flat"])
        self.assertIn("tahmin_flat", r["sira_tanimsiz"])
        d = _den(r)["sabit tahmin: sıra metrikleri tanımsız (constant prediction)"]
        self.assertIn("flat", d["ayrinti"])
        self.assertIn("undefined", ev.card(r, lang="en"))


class OneRowItemsShuffleTest(unittest.TestCase):
    """G0 #4a: the within-item shuffle control has no meaning when every item is one row."""

    def test_shuffle_row_skipped(self):
        rng = np.random.default_rng(0)
        t = rng.normal(0, 1, 60)
        df = pd.DataFrame({"item": [f"e{i}" for i in range(60)], "truth": t, "pred_ours": t + rng.normal(0, .5, 60)})
        r = _ev(df, primary="mae")
        self.assertFalse(any(c.startswith("kontrol_karistir_") for c in r["sonuclar"]))
        self.assertFalse(any(m["tur"] == "kontrol" for m in r["merdiven"]))
        self.assertNotIn("shuffled", ev.card(r, lang="en").split("## Audits")[0])


if __name__ == "__main__":
    unittest.main()
