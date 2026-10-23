"""runs/centaur/korpus.py (Consumer-101): transcript format and the leakage assert, on synthetic mini data (no download,
no TALP repo, no tokenizer)."""
import importlib.util
import unittest
from pathlib import Path

YOL = Path(__file__).resolve().parents[1] / "runs" / "centaur" / "korpus.py"
_spec = importlib.util.spec_from_file_location("centaur_korpus", YOL)
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)

ADLAR = {"BOOK": "Books", "SOCKS": "Socks", "BATTERY": "Batteries"}
PROFIL = "- Age group: 35 - 44 years\n- Gender: Female"
H1_2021, H2_2021, H1_2022 = K.donem_indeksi(K.date(2021, 3, 1)), K.donem_indeksi(K.date(2021, 9, 1)), K.SON_EGITIM_DONEMI
DONEM = {H2_2021: {"BOOK": 1}, H1_2022: {"SOCKS": 1, "BATTERY": 1}}
DIGER = {H2_2021: {"kalem": 3, "kategori": 2, "en_cok": [("pet food", 2), ("tea", 1)]}}
TWIN_OGE = [
    {"oge": "Q1_1", "qid": "Q1", "tip": "Matrix", "soru": "Would you support or oppose...", "alt": "A carbon tax?",
     "secenekler": '["Oppose", "Neutral", "Support"]', "alt_sinir": 1, "ust_sinir": 3},
    {"oge": "Q2", "qid": "Q2", "tip": "Slider", "soru": "How likely is rain tomorrow?", "alt": "",
     "secenekler": "[]", "alt_sinir": 0, "ust_sinir": 100},
]


def egitim(uid="u1"):
    return K.egitim_kaydi("talp73", uid, K.talp73_metni(uid, PROFIL, DONEM, ADLAR, DIGER), "consumer101/talp73")


def test_kaydi(uid="t1", sorulan=None):
    return K.talp73_test_kaydi(uid, 0, PROFIL, DONEM, ADLAR, DIGER, "kisi", sorulan=sorulan)


def etiketler(kayitlar):
    return [{"kaynak": r["kaynak"], "ayrim": r["ayrim"], "uid": r["uid"], "anahtar": h["anahtar"], "y": 0}
            for r in kayitlar for h in r["hedefler"]]


YASAK = (K.donem_etiketi(K.HEDEF_DONEMI),)


class TestTalp73Format(unittest.TestCase):
    def test_periods_and_choices(self):
        t = egitim()["text"]
        # starts at the first period with an order (2021-H2), ends at 2022-H1, 20 -> 3 categories per period
        self.assertNotIn(K.donem_etiketi(H1_2021), t)
        self.assertIn("July - December 2021:", t)
        self.assertIn("January - June 2022:", t)
        self.assertNotIn("July - December 2022", t)
        self.assertEqual(len(K.secim_araliklari(t)), 2 * len(ADLAR))
        self.assertIn("Books: <<yes>>", t.split("January - June 2022:")[0])
        self.assertIn("Socks: <<yes>>", t.split("January - June 2022:")[1])
        self.assertIn("Books: <<no>>", t.split("January - June 2022:")[1])
        self.assertIn("pet food (2)", t)
        self.assertIn("Other purchases in this period: none.", t)
        self.assertEqual(t, K.talp73_metni("u1", PROFIL, DONEM, ADLAR, DIGER))     # deterministic order

    def test_choice_contents_only(self):
        t = egitim()["text"]
        self.assertEqual({t[a:b] for a, b in K.secim_araliklari(t)}, {"yes", "no"})

    def test_period_after_cut_refused(self):
        with self.assertRaises(AssertionError):
            K.talp73_metni("u1", PROFIL, {**DONEM, K.HEDEF_DONEMI: {"BOOK": 1}}, ADLAR, DIGER)

    def test_test_record(self):
        r = test_kaydi()
        b = r["baglam"]
        self.assertTrue(b.endswith("July - December 2022:\n"))
        self.assertNotIn("<<", b[b.rfind("July - December 2022"):])
        self.assertEqual([h["devam"] for h in r["hedefler"]], ["Books: <<", "Socks: <<", "Batteries: <<"])
        self.assertEqual([h["oge"] for h in test_kaydi(sorulan=["SOCKS"])["hedefler"]], ["SOCKS"])
        self.assertTrue(b.startswith(egitim("t1")["text"]))      # same transcript as a train record, then the header

    def test_cover_story(self):
        r = K.talp73_test_kaydi("t1", 0, PROFIL, DONEM, {k: K.KAPAK[k] for k in ("SOCKS",)}, DIGER, "kapak", "kapak")
        self.assertIn("Hosiery: <<", r["baglam"])
        self.assertNotIn("Amazon", r["baglam"])
        self.assertNotIn("Socks", r["baglam"])

    def test_kapak_names_differ(self):
        for k, v in K.KAPAK.items():
            self.assertNotIn(v.lower(), {"books", "socks", "shoes", "pants", "headphones"}, k)


class TestLeakage(unittest.TestCase):
    def test_clean_passes(self):
        tr, te = [egitim("u1"), egitim("u2")], [test_kaydi("t1")]
        n = K.sizinti_kontrol(tr, te, etiketler(te), YASAK)
        self.assertEqual((n["egitim_kisi"], n["test_kisi"], n["etiket"]), (2, 1, 3))

    def test_test_person_in_train(self):
        tr, te = [egitim("u1"), egitim("t1")], [test_kaydi("t1")]
        with self.assertRaises(AssertionError):
            K.sizinti_kontrol(tr, te, etiketler(te), YASAK)

    def test_target_period_in_train(self):
        bad = egitim("u1")
        bad["text"] += "\nJuly - December 2022:\nBooks: <<yes>>\n"
        te = [test_kaydi("t1")]
        with self.assertRaises(AssertionError):
            K.sizinti_kontrol([bad], te, etiketler(te), YASAK)

    def test_answer_in_test_context(self):
        r = test_kaydi("t1")
        r["baglam"] += "Books: <<yes>>\n"
        with self.assertRaises(AssertionError):
            K.sizinti_kontrol([egitim("u1")], [r], etiketler([r]), YASAK)

    def test_labels_must_match_targets(self):
        te = [test_kaydi("t1")]
        with self.assertRaises(AssertionError):
            K.sizinti_kontrol([egitim("u1")], te, etiketler(te)[:-1], YASAK)
        extra = etiketler(te) + [{"kaynak": "talp73", "ayrim": "kisi", "uid": "u1", "anahtar": "BOOK", "y": 1}]
        with self.assertRaises(AssertionError):
            K.sizinti_kontrol([egitim("u1")], te, extra, YASAK)

    def test_same_uid_other_source_allowed(self):
        tw = K.egitim_kaydi("twin", "t1", "x <<1>>", "consumer101/twin2k500")
        te = [test_kaydi("t1")]
        K.sizinti_kontrol([tw], te, etiketler(te), YASAK)


class TestTwin(unittest.TestCase):
    def test_train_and_test(self):
        d13, d4 = {"Q1_1": 3.0, "Q2": 40.0}, {"Q1_1": 2.0, "Q2": 55.0}
        t, no = K.twin_metni(7, "- Age: 30-49", TWIN_OGE, d13, d4)
        self.assertEqual(no, {"Q1_1": "Q1", "Q2": "Q2"})
        b1, b2 = t.split("Part 2 (wave 4")
        self.assertIn("[Q1] A carbon tax?: <<3>>", b1)
        self.assertIn("options: 1 = Oppose; 2 = Neutral; 3 = Support", b1)
        self.assertIn("(slider from 0 to 100): <<40>>", b1)
        self.assertIn("[Q1] A carbon tax?: <<2>>", b2)
        self.assertIn("[Q2] How likely is rain tomorrow?: <<55>>", b2)
        r = K.twin_test_kaydi(8, "- Age: 30-49", TWIN_OGE, {"Q1_1": 1.0})
        self.assertIn("[Q2] How likely is rain tomorrow? (slider from 0 to 100): (no answer)", r["baglam"])
        self.assertTrue(r["baglam"].rstrip().endswith("in a new order."))
        self.assertEqual([h["devam"] for h in r["hedefler"]], ["[Q1] A carbon tax?: <<", "[Q2] How likely is rain tomorrow?: <<"])
        r["uid"] = "8"
        tr = [K.egitim_kaydi("twin", 7, t, "consumer101/twin2k500")]
        K.sizinti_kontrol(tr, [r], etiketler([r]))
        r["baglam"] += "[Q1] A carbon tax?: <<2>>\n"
        with self.assertRaises(AssertionError):
            K.sizinti_kontrol(tr, [r], etiketler([r]))


class TestHelpers(unittest.TestCase):
    def test_api_prompt_has_no_label(self):
        r = test_kaydi("t1")
        p = K.api_istemi(r, K.ornek_sec([egitim("u1"), egitim("u2")], 1))
        self.assertEqual(p["anahtarlar"], ["BOOK", "SOCKS", "BATTERY"])
        self.assertIn("Example record 1", p["user"])
        hedef = p["user"].split("=== Record to predict ===")[1]
        self.assertNotIn("July - December 2022:\nBooks: <<", hedef)
        self.assertEqual(hedef.count("<<"), egitim("t1")["text"].count("<<"))    # the person's history only

    def test_chat_roundtrip(self):
        t = egitim()["text"]
        m = K.chat_mesajlari(t)
        self.assertEqual("".join(x["content"] for x in m), t)
        self.assertEqual(sum(x["role"] == "assistant" for x in m), len(K.secim_araliklari(t)))
        self.assertTrue(all(x["content"].startswith("<<") for x in m if x["role"] == "assistant"))

    def test_counts_without_tokenizer(self):
        s = K.say([egitim()["text"]])
        self.assertEqual(s["secim"], 6)
        self.assertIsNone(s["secim_token"])
        self.assertIn("yaklasik", s["token_kaynak"])

    def test_time_estimate(self):
        e = K.sure_tahmini(3_600_000)
        self.assertEqual(e["80GB"]["saat"], "0.7-1.4")


if __name__ == "__main__":
    unittest.main()
