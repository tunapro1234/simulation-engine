"""Evidence card rows: every limit note is kept (was cut to 3), declared baselines on the pending card, persona field
rows; `persona.field_report` (how many persona fields reached the prompt, which source columns were dropped).
Small synthetic tables, offline."""
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine.evidence as ev
import simengine.persona as sp
from simengine.evidence import cekirdek, kart


def _binary(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return pd.DataFrame({"unit": np.arange(n), "truth": y,
                         "pred_ours": np.clip(0.5 * y + rng.uniform(0, 0.5, n), 0, 1)})


def _pending(**kw):
    df = pd.DataFrame({"unit": np.arange(20), "pred_ours": np.linspace(0, 1, 20)})
    return cekirdek.bekleyen(df.rename(columns={"pred_ours": "tahmin_ours"}), "bekleyen", bizim="ours", **kw)


def _taban(k):
    return next(r for r in k["satirlar"] if r["tur"] == "taban")


class AllLimitNotes(unittest.TestCase):
    """intent-llm EKSIK 9: yanilgi kept only sinirlar[:3]; prereg / dominance notes after the third vanished."""

    def test_card_with_five_limits_prints_five(self):
        notes = [f"limit note {i}" for i in range(5)]
        k = kart.kart({"sinirlar": notes, "birincil": "accuracy"})
        self.assertEqual(k["yanilgi"], notes)

    def test_evaluate_limits_all_reach_card_and_markdown(self):
        notes = [f"user limit {i}" for i in range(5)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = ev.evaluate(_binary(), task="binary", ours="ours", B=100, B_aggregate=50, multiplier=False,
                              limits=notes)
        for x in notes:
            self.assertIn(x, rep["kanit_kart"]["yanilgi"])
        md = ev.card(rep)
        self.assertEqual(sum(f"- {x}" in md for x in notes), 5)

    def test_duplicates_dropped_order_kept(self):
        k = kart.kart({"sinirlar": ["b", "a", "b", "c"], "birincil": "accuracy"}, yanilgi=["a", "z"])
        self.assertEqual(k["yanilgi"], ["a", "z", "b", "c"])


class DeclaredBaselinesPending(unittest.TestCase):
    """D5 #8e family: the pending card said 'taban yok' although baselines were declared."""

    def test_declared_labels_printed(self):
        oz = _pending()
        oz["bildirilen_tabanlar"] = ["T0", "T1"]
        r = _taban(kart.kart(oz))
        self.assertEqual(r["not"], "declared, scored when truth arrives: T0, T1")
        self.assertIsNone(r["deger"])

    def test_dict_labels(self):
        oz = _pending()
        oz["bildirilen_tabanlar"] = [{"etiket": "T0", "sutun": "tahmin_t0"}, {"sutun": "tahmin_t1"}]
        self.assertEqual(_taban(kart.kart(oz))["not"], "declared, scored when truth arrives: T0, tahmin_t1")

    def test_none_label_skipped(self):
        oz = _pending()
        oz["bildirilen_tabanlar"] = [{"etiket": None}, {}, "T1"]
        self.assertEqual(_taban(kart.kart(oz))["not"], "declared, scored when truth arrives: T1")

    def test_evaluate_pending_prints_declared(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"unit": np.arange(40), "pred_ours": rng.uniform(0, 1, 40), "pred_t0": rng.uniform(0, 1, 40)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = ev.evaluate(df, task="binary", ours="ours", unlabeled_baselines=["pred_t0"], multiplier=False)
        self.assertIn("declared, scored when truth arrives: t0", ev.card(rep, lang="en"))

    def test_absent_key_unchanged(self):
        self.assertEqual(_taban(_pending()["kanit_kart"])["not"], "taban yok")
        self.assertEqual(kart.kart(_pending())["rozet"], "bekliyor")


class PersonaFieldRows(unittest.TestCase):
    """G5#12: 46/801 households lost their age field silently; the card did not say how many fields reached it."""

    def test_rows_printed(self):
        oz = _pending()
        oz["persona_alanlari"] = {"n_fields": 6, "dropped": ["Age", "region"],
                                  "per_field_missing": {"age_band": 46, "gender": 0}, "n_people": 801}
        k = kart.kart(oz)
        turs = [r["tur"] for r in k["satirlar"]]
        self.assertEqual(turs[:len(kart.SIRA)], list(kart.SIRA))
        self.assertEqual(turs[len(kart.SIRA):], ["persona", "dusen"])
        p, d = k["satirlar"][-2:]
        self.assertEqual((p["sayi"], p["deger"]), (6, None))   # a count, not a metric value printed as 6.000
        self.assertIn("6 persona fields reached the prompt, 801 people", p["etiket"])
        self.assertIn("age_band 46/801", p["not"])
        self.assertEqual((d["sayi"], d["deger"]), (2, None))
        self.assertIn("Age, region", d["not"])

    def test_no_drop(self):
        oz = _pending()
        oz["persona_alanlari"] = {"n_fields": 3, "dropped": []}
        d = kart.kart(oz)["satirlar"][-1]
        self.assertEqual((d["tur"], d["sayi"]), ("dusen", 0))

    def test_absent_key_six_rows(self):
        self.assertEqual([r["tur"] for r in kart.kart(_pending())["satirlar"]], list(kart.SIRA))


def _table():
    return pd.DataFrame({"rid": ["a", "b", "c", "d"], "age": [25, 40, None, 70], "gender": ["female", "male", "female", "male"],
                         "visits": [1, 2, 3, 4], "region": ["n", "s", "e", "w"], "target": [0, 1, 0, 1]})


class FieldReport(unittest.TestCase):
    def test_unknown_column_in_mapping_is_dropped(self):
        df = _table()
        m = sp.Mapping(id_col="rid", fields={"age_band": "Age", "gender": "gender"}, facets={"visits": "visits"},
                       forbidden=["target"])
        people = sp.from_table(df, m)
        rep = sp.field_report(people, df, m)
        self.assertEqual(rep["used"], ["gender", "visits"])
        self.assertEqual(rep["n_fields"], 2)
        self.assertIn("age_band", rep["dropped"])          # mapped to 'Age', which is not a column
        self.assertIn("region", rep["dropped"])            # column not in the mapping
        self.assertIn("age", rep["dropped"])
        self.assertNotIn("target", rep["dropped"])         # forbidden on purpose
        self.assertNotIn("rid", rep["dropped"])
        self.assertIn("not a column", rep["why"]["age_band"])
        self.assertEqual(rep["per_field_missing"], {"gender": 0, "visits": 0})

    def test_auto_mapping_partial_missing(self):
        df = _table()
        people = sp.from_table(df, id_col="rid", forbidden=["target"])
        rep = sp.field_report(people, df)
        self.assertEqual(rep["n_people"], 4)
        self.assertEqual(rep["per_field_missing"]["age_band"], 1)  # the NaN age
        self.assertIn("age_band", rep["used"])
        self.assertEqual(rep["dropped"], ["target"])               # no mapping given: forbidden is unknown here
        line = sp.as_limit(rep)
        self.assertIsInstance(line, str)
        self.assertNotIn("\n", line)
        self.assertIn("age_band 1/4", line)
        self.assertIn("target", line)

    def test_people_only(self):
        people = sp.from_table(_table(), id_col="rid", forbidden=["target"])
        rep = sp.field_report(people)
        self.assertEqual(rep["dropped"], [])
        self.assertEqual(rep["used"], ["age_band", "gender", "visits", "region"])
        self.assertEqual(rep["n_fields"], 4)

    def test_report_feeds_card(self):
        df = _table()
        people = sp.from_table(df, id_col="rid", forbidden=["target"])
        oz = _pending(sinirlar=[sp.as_limit(sp.field_report(people, df))])
        oz["persona_alanlari"] = sp.field_report(people, df)
        k = kart.kart(oz)
        self.assertEqual(k["satirlar"][-2]["sayi"], 4)
        self.assertTrue(any(x.startswith("Persona fields:") for x in k["yanilgi"]))


class AttachToEvaluatedReport(unittest.TestCase):
    """The documented path (report["persona_alanlari"] = rep) never reached se.card(report): the card is the stored
    report["kanit_kart"]; rebuilding it with kart.kart(report) dropped the seal record (badge reason said 'seal()')."""

    def _rep(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ev.evaluate(_binary(), task="binary", ours="ours", B=100, B_aggregate=50, multiplier=False)

    def test_rows_and_limit_reach_markdown(self):
        df = _table()
        people = sp.from_table(df, id_col="rid", forbidden=["target"])
        rep = self._rep()
        before = dict(rep["kanit_kart"], satirlar=list(rep["kanit_kart"]["satirlar"]))
        out = sp.attach(rep, sp.field_report(people, df))
        self.assertIs(out, rep)
        md = ev.card(rep)
        self.assertIn("| Persona fields |", md)        # ROW_TO_ENGLISH names the persona / dusen rows (W8)
        self.assertIn("| Dropped fields |", md)
        self.assertNotIn("4.000", md)                       # the count is not printed as a metric value
        self.assertIn("- Persona fields: 4 reached the prompt", md)
        k = rep["kanit_kart"]
        self.assertEqual(k["satirlar"][:len(before["satirlar"])], before["satirlar"])
        self.assertEqual((k["rozet"], k["rozet_gerekce"]), (before["rozet"], before["rozet_gerekce"]))

    def test_idempotent(self):
        df = _table()
        people = sp.from_table(df, id_col="rid", forbidden=["target"])
        rep = self._rep()
        sp.attach(rep, sp.field_report(people, df))
        sp.attach(rep, sp.field_report(people[:2], df))
        k = rep["kanit_kart"]
        self.assertEqual([r["tur"] for r in k["satirlar"]].count("persona"), 1)
        self.assertEqual(sum(str(x).startswith("Persona fields:") for x in k["yanilgi"]), 1)
        self.assertEqual(sum(str(x).startswith("Persona fields:") for x in rep["sinirlar"]), 1)
        self.assertIn("for 2 people", k["yanilgi"][-1])

    def test_unhashable_limit_does_not_crash(self):
        k = kart.kart({"sinirlar": ["a", {"x": 1}, {"x": 1}, "a"], "birincil": "accuracy"})
        self.assertEqual(k["yanilgi"], ["a", {"x": 1}])


if __name__ == "__main__":
    unittest.main()
