"""simengine.evidence fixes from blind rehearsal D3 (new-item cold start, group residual metric): offset group metric
(#2), correlation_audit call budget checked before any call (#6), sampling-noise ceiling (#7; variance components over all groups), seal
without a stored copy / external archive (#10), memorization caveat on the badge + blind arm row (#11), paired_diff
and registered comparisons (#12). Offline: FakeLLM and small synthetic data only."""
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import simengine as se
import simengine.evidence as ev
from simengine.evidence import kart
from simengine.llm import FakeLLM

FAST = dict(B=200, B_aggregate=100, multiplier=False)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _films(G=60, n=20, seed=0, sd_eff=0.3, sd_noise=0.8):
    """Synthetic 'new item' table: member offset um, item effect eff, rating = um + eff + noise."""
    rng = np.random.default_rng(seed)
    film = np.repeat(np.arange(G), n)
    um = rng.normal(3.5, 0.4, G * n)
    eff = rng.normal(0, sd_eff, G)
    df = pd.DataFrame({"movie_id": film, "um": um, "truth": um + eff[film] + rng.normal(0, sd_noise, G * n)})
    df["pred_ours"] = um + eff[film] + rng.normal(0, 0.15, G)[film]
    df["pred_base"] = um + 0.5 * eff[film] + rng.normal(0, 0.3, G)[film]
    df["pred_blind"] = um + rng.normal(0, 0.3, G)[film]
    return df, eff


def _group_resid(df, col):
    return (df[col] - df.um).groupby(df.movie_id).mean()


# ------------------------------------------------------------------ #2 offset group metric
class TestOffset(unittest.TestCase):
    def test_offset_group_metric_is_film_residual_spearman(self):
        df, _ = _films()
        r = _quiet(ev.evaluate, df, ours="ours", raw="blind", offset_col="um", aggregate_by="movie_id",
                   primary="spearman", **FAST)
        want = spearmanr(_group_resid(df, "pred_ours"), _group_resid(df, "truth"))[0]
        self.assertAlmostEqual(ev.summary(r)["ours"]["value"], want, places=10)
        self.assertEqual(r["ofset"]["column"], "um")
        self.assertEqual(ev.summary(r)["offset"]["n_missing"], 0)
        self.assertTrue(any("offset_col=um" in x for x in r["sinirlar"]))

    def test_offset_equals_manual_residual_frame(self):
        df, _ = _films(G=40, n=10, seed=1)
        man = df.copy()
        for c in ("truth", "pred_ours", "pred_base", "pred_blind"):
            man[c] = df[c] - df.um
        a = _quiet(ev.evaluate, df, ours="ours", offset_col="um", aggregate_by="movie_id", primary="spearman", **FAST)
        b = _quiet(ev.evaluate, man.drop(columns="um"), ours="ours", aggregate_by="movie_id", primary="spearman", **FAST)
        self.assertEqual([x["value"] for x in ev.summary(a)["ladder"]], [x["value"] for x in ev.summary(b)["ladder"]])

    def test_offset_row_level_and_errors(self):
        df, _ = _films(G=20, n=10)
        r = _quiet(ev.evaluate, df, ours="ours", offset_col="um", primary="pearson", **FAST)
        want = np.corrcoef(df.pred_ours - df.um, df.truth - df.um)[0, 1]
        self.assertAlmostEqual(ev.summary(r)["ours"]["value"], want, places=10)
        with self.assertRaises(ValueError):
            ev.evaluate(df, ours="ours", offset_col="nope", **FAST)
        with self.assertRaises(ValueError):
            ev.evaluate(df, ours="ours", offset_col="pred_base", **FAST)
        with self.assertRaises(ValueError):
            ev.evaluate(df, ours="ours", offset_col="um", task="multiclass", **FAST)

    def test_missing_offset_warns_and_drops(self):
        df, _ = _films(G=20, n=10)
        df.loc[:4, "um"] = np.nan
        with self.assertWarns(UserWarning):
            r = ev.evaluate(df, ours="ours", offset_col="um", aggregate_by="movie_id", primary="spearman", **FAST)
        self.assertEqual(r["ofset"]["n_missing"], 5)


# ------------------------------------------------------------------ #7 split-half noise ceiling
class TestNoiseCeiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.eff = _films(G=150, n=30, seed=2)
        cls.rep = _quiet(ev.evaluate, cls.df, ours="ours", raw="blind", offset_col="um", aggregate_by="movie_id",
                         primary="spearman", noise_ceiling="split_half", baselines_oof=["base"], **FAST)

    def test_ceiling_near_theory_and_on_card(self):
        s = ev.summary(self.rep)
        nc = self.rep["gurultu_tavani"]
        rel = self.eff.var() / (self.eff.var() + 0.64 / 30)       # reliability of a 30-row group mean
        self.assertLess(abs(nc["tavan"] - np.sqrt(rel)), 0.06)
        self.assertAlmostEqual(nc["spearman_brown"], 2 * nc["yari_korelasyon"] / (1 + nc["yari_korelasyon"]))
        self.assertAlmostEqual(nc["yari_tavan"], np.sqrt(nc["spearman_brown"]))   # split-half cross-check
        self.assertEqual(nc["yontem"], "varyans_bilesenleri")
        self.assertEqual((nc["n_grup"], nc["n_tek_satir"]), (150, 0))
        # equal 30-row groups: variance components and split-half agree closely
        self.assertLess(abs(nc["tavan"] - nc["yari_tavan"]), 0.02)
        self.assertEqual(nc["R"], 200)
        self.assertEqual(s["ceiling_kind"], "sampling noise (variance components)")
        self.assertAlmostEqual(s["ceiling"]["value"], nc["tavan"])
        self.assertIsNotNone(s["gap_closed"])
        lo, hi = s["ceiling"]["lo"], s["ceiling"]["hi"]
        self.assertTrue(lo < nc["tavan"] < hi)
        md = ev.card(self.rep, lang="en")
        self.assertIn("Sampling-noise ceiling (variance components)", md)
        self.assertIn("Ceiling type: sampling noise (variance components)", md)
        k = self.rep["kanit_kart"]
        row = next(r for r in k["satirlar"] if r["tur"] == "tavan")
        self.assertIn("örneklem gürültüsü", row["not"])
        bos = next(r for r in k["satirlar"] if r["tur"] == "bosluk")
        self.assertIsNotNone(bos["deger"])
        self.assertIn("Örneklem gürültüsü tavanı", ev.dashboard_html([self.rep]))

    def test_deterministic_and_seeded(self):
        r2 = _quiet(ev.evaluate, self.df, ours="ours", offset_col="um", aggregate_by="movie_id", primary="spearman",
                    noise_ceiling="split_half", noise_ceiling_repeats=50, **FAST)
        r3 = _quiet(ev.evaluate, self.df, ours="ours", offset_col="um", aggregate_by="movie_id", primary="spearman",
                    noise_ceiling="split_half", noise_ceiling_repeats=50, **FAST)
        self.assertEqual(r2["gurultu_tavani"]["tavan"], r3["gurultu_tavani"]["tavan"])
        self.assertEqual(r2["gurultu_tavani"]["R"], 50)

    def test_r2_ceiling_is_reliability(self):
        kw = dict(ours="ours", offset_col="um", aggregate_by="movie_id", noise_ceiling="split_half",
                  noise_ceiling_repeats=40, **FAST)
        rp = _quiet(ev.evaluate, self.df, primary="pearson", **kw)["gurultu_tavani"]
        rr = _quiet(ev.evaluate, self.df, primary="r2", **kw)["gurultu_tavani"]
        self.assertEqual(rp["korelasyon"], "pearson")
        self.assertAlmostEqual(rr["tavan"], rp["tavan"] ** 2, places=10)

    def test_errors(self):
        with self.assertRaises(ValueError):
            ev.evaluate(self.df, ours="ours", noise_ceiling="split_half", **FAST)       # no aggregate_by
        with self.assertRaises(ValueError):
            ev.evaluate(self.df, ours="ours", aggregate_by="movie_id", noise_ceiling="bootstrap", **FAST)
        with self.assertRaises(ValueError):
            ev.evaluate(self.df, ours="ours", aggregate_by="movie_id", noise_ceiling="split_half",
                        ceiling_col="pred_base", **FAST)
        with self.assertRaises(NotImplementedError):
            _quiet(ev.evaluate, self.df, ours="ours", aggregate_by="movie_id", noise_ceiling="split_half",
                   primary="mae", **FAST)

    def test_leak_exceeds_ceiling_and_retest_ignored(self):
        df, _ = _films(G=60, n=20, seed=3)
        df["pred_leak"] = df.truth                    # the truth itself: beats the noise ceiling
        df["retest"] = df.truth
        r = _quiet(ev.evaluate, df, ours="ours", aggregate_by="movie_id", primary="spearman",
                   noise_ceiling="split_half", noise_ceiling_repeats=50, **FAST)
        a = next(x for x in r["denetim"] if x["kontrol"].startswith("örneklem gürültüsü tavanını"))
        self.assertEqual(a["durum"], "uyari")
        self.assertIn("pred_leak".replace("pred_", "tahmin_"), a["ayrinti"])
        self.assertEqual(r["tavan_turu"], "orneklem_gurultusu")
        self.assertTrue(any("varyans bileşenleri" in n for n in r["kart"]["notlar"]))

    def test_variance_components_matches_hand_formula(self):
        """Pearson ceiling = sqrt(1 - mean(s2_g/n_g) / var(means) * (G-3)/(G-1)) over every scored group."""
        df, _ = _films(G=50, n=7, seed=5)
        df = df.iloc[np.r_[0:3, 7:len(df)]]                       # film 0 keeps 3 rows: unequal sizes
        r = _quiet(ev.evaluate, df, ours="ours", offset_col="um", aggregate_by="movie_id", primary="pearson",
                   noise_ceiling="variance_components", noise_ceiling_repeats=20, **FAST)
        res = (df.truth - df.um).groupby(df.movie_id)
        m, v, n = res.mean(), res.var(ddof=1), res.size()
        G = len(m)
        rel = 1 - (v / n).mean() / m.var(ddof=1) * (G - 3) / (G - 1)
        nc = r["gurultu_tavani"]
        self.assertAlmostEqual(nc["guvenilirlik"], rel, places=10)
        self.assertAlmostEqual(nc["tavan"], np.sqrt(rel), places=10)
        same = _quiet(ev.evaluate, df, ours="ours", offset_col="um", aggregate_by="movie_id", primary="pearson",
                      noise_ceiling="split_half", noise_ceiling_repeats=20, **FAST)["gurultu_tavani"]
        self.assertEqual(same["tavan"], nc["tavan"])               # older name, same estimator

    @staticmethod
    def _mc(sizes, prim, reps, G=80, seed0=100):
        est, orc, sh = [], [], []
        for rep in range(reps):
            rng = np.random.default_rng(seed0 + rep)
            tg = rng.normal(0, .5, G)
            n = sizes(rng, G)
            g = np.repeat(np.arange(G), n)
            df = pd.DataFrame({"item_id": g, "truth": tg[g] + rng.normal(0, 1, len(g)), "pred_oracle": tg[g]})
            r = _quiet(ev.evaluate, df, ours="oracle", primary=prim, aggregate_by="item_id", noise_ceiling="split_half",
                       noise_ceiling_repeats=10, B=50, B_aggregate=20, multiplier=False)
            est.append(r["gurultu_tavani"]["tavan"])
            sh.append(r["gurultu_tavani"]["yari_tavan"])
            orc.append(ev.summary(r)["ours"]["value"])
        return np.mean(est), np.mean(orc), np.mean(sh), r["gurultu_tavani"]

    def test_one_row_groups_not_overstated(self):
        """Regression: one-row groups were scored but left out of the split-half ceiling (0.85 vs 0.70 achievable)."""
        sizes = lambda rng, G: np.where(np.arange(G) % 10 < 3, 1, rng.integers(4, 21, G))
        for prim in ("pearson", "spearman"):
            est, orc, sh, nc = self._mc(sizes, prim, reps=12)
            self.assertLess(abs(est - orc), 0.06, prim)            # an oracle that knows the true group value
            self.assertGreater(sh - orc, 0.1, prim)                # the old split-half value was far too high
            self.assertEqual(nc["n_tek_satir"], 24)
            self.assertEqual(nc["n_grup"], 80)

    def test_small_groups_not_understated(self):
        """Regression: 3-row groups split 1 vs 2 biased the split-half ceiling low, so a perfect predictor exceeded it."""
        est, orc, sh, nc = self._mc(lambda rng, G: np.full(G, 3), "pearson", reps=12)
        self.assertLess(abs(est - orc), 0.04)
        self.assertIsNone(nc.get("not"))

    def test_one_row_note_on_card(self):
        df, _ = _films(G=40, n=6, seed=8)
        df = df[~((df.movie_id < 5) & (df.groupby("movie_id").cumcount() > 0))]   # films 0-4 keep one row
        r = _quiet(ev.evaluate, df, ours="ours", offset_col="um", aggregate_by="movie_id", primary="spearman",
                   noise_ceiling="split_half", noise_ceiling_repeats=20, **FAST)
        nc = r["gurultu_tavani"]
        self.assertEqual((nc["n_grup"], nc["n_tek_satir"], nc["n_grup_yari"]), (40, 5, 35))
        self.assertTrue(any("tek satırlı" in x for x in r["kart"]["notlar"]))
        self.assertIn("5 with one row", ev.card(r, lang="en"))

    def test_default_report_unchanged(self):
        r = _quiet(ev.evaluate, self.df, ours="ours", aggregate_by="movie_id", primary="spearman", **FAST)
        for key in ("gurultu_tavani", "kayitli_karsilastirmalar", "kor_kol", "ezber_uyarisi", "ofset"):
            self.assertNotIn(key, r)
        self.assertNotIn("kayitli", r["kanit_kart"])
        self.assertIsNone(ev.summary(r)["badge_caveat"])
        self.assertIsNone(r["manset"]["tavan"])


# ------------------------------------------------------------------ #6 call budget before any call
class TestAuditBudget(unittest.TestCase):
    def test_budget_checked_before_calls(self):
        truth = {f"k{i}": float(i) for i in range(300)}
        llm = FakeLLM(lambda s, u: "3")
        with self.assertRaises(ValueError) as cm:
            ev.correlation_audit(truth, "In the study S, value for {key}?", "Value for {key}?", llm=llm, B=20)
        self.assertIn("600 calls", str(cm.exception))
        self.assertIn("max_calls=600", str(cm.exception))
        self.assertEqual(len(llm.calls), 0)
        with self.assertRaises(ValueError) as cm:
            ev.correlation_audit(truth, "S {key}?", "{key}?", llm=llm, samples=2, max_calls=1000, B=20)
        self.assertIn("1200", str(cm.exception))
        self.assertEqual(len(llm.calls), 0)

    def test_answer_dict_arm_needs_no_calls(self):
        truth = {f"k{i}": float(i) for i in range(300)}
        llm = FakeLLM(lambda s, u: "3")
        r = ev.correlation_audit(truth, {k: v for k, v in truth.items()}, "Value for {key}?", llm=llm, B=20)
        self.assertEqual(len(llm.calls), 300)
        self.assertEqual(r["calls"], 300)


# ------------------------------------------------------------------ #10 seal without copy / external archive
class TestSealCopy(unittest.TestCase):
    def setUp(self):
        self.pred = pd.DataFrame({"id": [1, 2, 3, 4], "p": [0.1, 0.4, 0.35, 0.8]})
        self.truth = pd.DataFrame({"id": [1, 2, 3, 4], "y": [0, 1, 0, 1]})

    def test_default_keeps_copy_and_record_shape(self):
        with tempfile.TemporaryDirectory() as t:
            rec = ev.seal(self.pred, "x", registry=t)
            self.assertTrue((Path(t) / rec["arsiv"]).exists())
            self.assertNotIn("arsiv_konum", rec)
            g = ev.grade(None, self.truth, ["id"], registry=t)            # reopen from the archive
            self.assertEqual(g["n_birlesik"], 4)
            self.assertEqual(g["dogrulama"]["kaynak"], "arsiv")

    def test_store_copy_false_chain_only(self):
        with tempfile.TemporaryDirectory() as t:
            rec = ev.seal(self.pred, "no copy", registry=t, store_copy=False)
            self.assertIsNone(rec["arsiv"])
            self.assertEqual(sorted(p.name for p in Path(t).iterdir()), ["zincir.jsonl"])
            self.assertTrue(ev.verify(self.pred, registry=t)["eslesti"])
            self.assertTrue(ev.verify_chain(t)["gecerli"])
            self.assertEqual(ev.grade(self.pred, self.truth, ["id"], registry=t)["n_birlesik"], 4)
            with self.assertRaises(FileNotFoundError):
                ev.grade(None, self.truth, ["id"], registry=t)
            from simengine import research
            self.assertTrue(research.verify_seal(self.pred, registry_dir=t)["matched"])
            self.assertEqual(research.grade_sealed(self.pred, self.truth, ["id"], registry_dir=t)["n_joined"], 4)
            rep = _quiet(ev.evaluate, self.pred.rename(columns={"p": "pred_m"}), ours="m", prereg=t)
            self.assertEqual(ev.summary(rep)["seals"][0]["sha256"], rec["sha256"])

    def test_archive_dir_elsewhere(self):
        with tempfile.TemporaryDirectory() as t, tempfile.TemporaryDirectory() as arch:
            reg = Path(t) / ".onkayit"
            rec = se.seal(self.pred, "archived elsewhere", registry=reg, archive_dir=arch)
            self.assertFalse((reg / rec["arsiv"]).exists())
            self.assertTrue((Path(arch) / rec["arsiv"]).exists())
            self.assertEqual(rec["arsiv_konum"], "dis")
            self.assertNotIn(str(arch), (reg / "zincir.jsonl").read_text())
            self.assertTrue(ev.verify_chain(reg)["gecerli"])
            g = ev.grade(None, self.truth, ["id"], registry=reg, archive_dir=arch)
            self.assertEqual(g["n_birlesik"], 4)
            with self.assertRaises(FileNotFoundError):
                ev.grade(None, self.truth, ["id"], registry=reg)
            with self.assertRaises(ValueError):
                ev.seal(self.pred, "x", registry=reg, store_copy=False, archive_dir=arch)

    def test_tampered_archive_rejected(self):
        import gzip
        with tempfile.TemporaryDirectory() as t:
            rec = ev.seal(self.pred, "x", registry=t)
            (Path(t) / rec["arsiv"]).write_bytes(gzip.compress(b"id,p\n1,0.9\n"))
            with self.assertRaises(ValueError):
                ev.grade(None, self.truth, ["id"], registry=t)

    def test_docstring_warns_about_working_directory(self):
        self.assertIn("git-ignored", ev.seal.__doc__)
        self.assertIn("store_copy=False", ev.seal.__doc__)

    def test_cli_flags(self):
        from simengine.evidence import onkayit
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "p.csv"
            self.pred.to_csv(f, index=False)
            reg = Path(t) / "reg"
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(onkayit.main(["--dizin", str(reg), "muhurle", str(f), "--aciklama", "a", "--kopyasiz"]), 0)
            self.assertEqual(sorted(p.name for p in reg.iterdir()), ["zincir.jsonl"])


# ------------------------------------------------------------------ #11 memorization caveat + blind arm
WARN = {"durum": "uyari", "ayrinti": "korelasyon denetimi: adlı 0,78, adsız 0,17; fark +0,61 [+0,49, +0,72]",
        "deger": 0.61, "alt": 0.49, "ust": 0.72}
OK = {"durum": "tamam", "ayrinti": "ezber işareti yok", "deger": 0.01, "alt": -0.1, "ust": 0.1}


class TestMemorizationCaveat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, _ = _films(G=50, n=10, seed=4)
        cls.kw = dict(ours="ours", offset_col="um", aggregate_by="movie_id", primary="spearman", baselines_oof=["base"],
                      **FAST)

    def test_warning_adds_caveat_tr_and_en(self):
        r = _quiet(ev.evaluate, self.df, memorization=WARN, **self.kw)
        k = r["kanit_kart"]
        self.assertIn(kart.EZBER_EKI, k["rozet_gerekce"])
        self.assertIn("kazanç ezber/tanımadan olabilir", k["etiketler"])
        s = ev.summary(r)
        self.assertIn("recognition/memorization", s["badge_caveat"])
        self.assertIn("memorization audit warning", s["badge_caveat"])
        for lang in ("mixed", "en"):
            self.assertIn("Caveat: gain may come from recognition/memorization", ev.card(r, lang=lang))
        self.assertIn(kart.EZBER_EKI, ev.dashboard_html([r]))
        self.assertEqual(r["ezber_uyarisi"]["nedenler"], ["ezber_denetimi"])
        # card dict recomputed from the report keeps the caveat
        self.assertIn(kart.EZBER_EKI, ev.card_dict(r)["rozet_gerekce"])

    def test_no_caveat_when_ok_or_no_llm(self):
        r = _quiet(ev.evaluate, self.df, memorization=OK, **self.kw)
        self.assertIsNone(ev.summary(r)["badge_caveat"])
        r = _quiet(ev.evaluate, self.df, memorization=WARN, ours_uses_llm=False, **self.kw)
        self.assertIsNone(ev.summary(r)["badge_caveat"])
        self.assertNotIn(kart.EZBER_EKI, r["kanit_kart"]["rozet_gerekce"])

    def test_no_caveat_without_gain(self):
        """Regression: with ours below the baseline the badge still said 'gain may come from memorization'."""
        rng = np.random.default_rng(2)
        y = rng.normal(size=400)
        df = pd.DataFrame({"truth": y, "pred_ours": rng.normal(size=400), "pred_base": y + rng.normal(0, .5, 400)})
        r = _quiet(ev.evaluate, df, ours="ours", primary="spearman", unlabeled_baselines=["base"], memorization=WARN,
                   multiplier=False, B=200)
        s = ev.summary(r)
        self.assertEqual(s["badge"], "below baseline")
        self.assertIsNone(s["badge_caveat"])
        self.assertNotIn(kart.EZBER_EKI, r["kanit_kart"]["rozet_gerekce"])
        self.assertNotIn("kazanç ezber/tanımadan olabilir", r["kanit_kart"]["etiketler"])
        for lang in ("mixed", "en"):
            self.assertNotIn("recognition/memorization", ev.card(r, lang=lang))
        self.assertIsNotNone(r.get("ezber_uyarisi"))       # the audit result itself is kept
        self.assertTrue(kart.kazanc_var(None))
        self.assertFalse(kart.kazanc_var({"fark": 0.0}))

    def test_blind_arm_row_and_flag(self):
        r = _quiet(ev.evaluate, self.df, blind_col="pred_blind", **self.kw)
        s = ev.summary(r)
        row = next(c for c in s["comparisons"] if c["kind"] == "blind")
        pc = ev.paired_diff(self.df, "ours", "blind", aggregate_by="movie_id", offset_col="um", metric="spearman", B=100)
        self.assertAlmostEqual(row["diff"], pc["diff"], places=10)
        self.assertTrue(r["kor_kol"]["kayip_cogu"])
        self.assertLess(r["kor_kol"]["korunan_pay"], 0.5)
        self.assertIn("the blind arm loses most of the gain", s["badge_caveat"])
        self.assertTrue(any(a["check"] == "kör kol (tanıma kontrolü)" and a["status"] == "warning" for a in s["audits"]))

    def test_blind_arm_keeping_gain_is_not_flagged(self):
        df = self.df.copy()
        df["pred_blind"] = df.pred_ours + np.random.default_rng(9).normal(0, 0.01, len(df))
        r = _quiet(ev.evaluate, df, blind_col="blind", **self.kw)
        self.assertFalse(r["kor_kol"]["kayip_cogu"])
        self.assertIsNone(ev.summary(r)["badge_caveat"])
        with self.assertRaises(ValueError):
            _quiet(ev.evaluate, df, blind_col="ours", **self.kw)


# ------------------------------------------------------------------ #12 paired_diff + registered comparisons
def _esli(ft, a, b, B=2000, seed=0):
    """Reference: the rehearsal's group-level paired bootstrap (rows = groups), point difference only."""
    return spearmanr(ft[a], ft.truth)[0] - spearmanr(ft[b], ft.truth)[0]


class TestPairedDiff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, _ = _films(G=50, n=12, seed=5)
        cls.ft = pd.DataFrame({c: _group_resid(cls.df, c) for c in ("truth", "pred_ours", "pred_base", "pred_blind")})

    def test_frame_matches_group_level_reference(self):
        d = ev.paired_diff(self.df, "ours", "base", aggregate_by="movie_id", offset_col="um", metric="spearman", B=500)
        self.assertAlmostEqual(d["diff"], _esli(self.ft, "pred_ours", "pred_base"), places=10)
        self.assertEqual(d["n"], 50)
        self.assertEqual(d["B"], 500)
        self.assertTrue(d["lo"] < d["diff"] < d["hi"])
        # a film-level frame gives the same point estimate with no aggregation
        f2 = self.ft.reset_index().rename(columns={"movie_id": "group"})
        d2 = ev.paired_diff(f2, "pred_ours", "pred_base", metric="spearman", B=200)
        self.assertAlmostEqual(d2["diff"], d["diff"], places=10)
        one = ev.paired_diff(f2, "blind", None, metric="spearman", B=200)
        self.assertAlmostEqual(one["diff"], spearmanr(self.ft.pred_blind, self.ft.truth)[0], places=10)
        self.assertIsNone(one["a_better"])

    def test_group_cluster_column(self):
        df = self.df.copy()
        df["block"] = df.movie_id // 5                     # coarser cluster, constant within each film
        d = ev.paired_diff(df, "ours", "base", aggregate_by="movie_id", group="block", offset_col="um",
                           metric="spearman", B=200)
        self.assertEqual(d["n"], 10)
        df.loc[0, "block"] = 99
        with self.assertRaises(ValueError):
            ev.paired_diff(df, "ours", "base", aggregate_by="movie_id", group="block", B=50)

    def test_evaluate_compare_table_and_report_lookup(self):
        r = _quiet(ev.evaluate, self.df, ours="ours", offset_col="um", aggregate_by="movie_id", primary="spearman",
                   compare=[("ours", "base"), ("blind", "base", "c: blind − base"), ("pred_blind", None, "blind ≈ 0")],
                   **FAST)
        rows = r["kayitli_karsilastirmalar"]
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(rows[0]["fark"], _esli(self.ft, "pred_ours", "pred_base"), places=10)
        self.assertEqual(rows[1]["etiket"], "c: blind − base")
        self.assertIsNone(rows[2]["b"])
        self.assertEqual(r["kanit_kart"]["kayitli"], rows)
        md = ev.card(r)
        self.assertIn("## Registered comparisons / drop conditions", md)
        self.assertIn("blind ≈ 0", md)
        self.assertIn("kayıtlı · c: blind − base", ev.dashboard_html([r]))
        fwd = ev.paired_diff(r, "ours", "base")
        rev = ev.paired_diff(r, "base", "ours")
        self.assertEqual(fwd["source"], "report")
        self.assertAlmostEqual(fwd["diff"], -rev["diff"])
        self.assertAlmostEqual(fwd["lo"], -rev["hi"])
        self.assertTrue(fwd["a_better"])
        self.assertEqual(fwd["n"], 50)
        # ours − X comparisons of the report are found too
        self.assertAlmostEqual(ev.paired_diff(r, "ours", "blind")["diff"],
                               _esli(self.ft, "pred_ours", "pred_blind"), places=10)
        with self.assertRaises(ValueError):
            ev.paired_diff(r, "blind", "base_missing")
        json.dumps(r)                                      # report stays JSON-friendly

    def test_single_arm_against_zero_has_no_direction(self):
        """Regression: on a lower-is-better metric an 'A against 0' row showed '▼ anlamlı · B iyi' with B = 0."""
        r = _quiet(ev.evaluate, self.df, ours="ours", offset_col="um", aggregate_by="movie_id", primary="mae",
                   compare=[("base", None, "base vs 0")], **FAST)
        row = r["kayitli_karsilastirmalar"][0]
        self.assertIsNone(row["a_iyi"])
        self.assertEqual(row["sonuc"], "0'ın üstünde")
        c = next(x for x in ev.summary(r)["comparisons"] if x["label"] == "base vs 0")
        self.assertIsNone(c["a_better"])
        self.assertEqual(c["verdict"], "above 0")
        html = ev.dashboard_html([r])
        self.assertIn("anlamlı · 0&#x27;ın üstünde" if "&#x27;" in html else "anlamlı · 0'ın üstünde", html)
        self.assertNotIn("B iyi", html.split("kayıtlı · base vs 0")[1].split("</tr>")[0])

    def test_compare_errors(self):
        with self.assertRaises(ValueError):
            _quiet(ev.evaluate, self.df, ours="ours", aggregate_by="movie_id", compare=[("ours", "nope")], **FAST)
        with self.assertRaises(ValueError):
            _quiet(ev.evaluate, self.df, ours="ours", aggregate_by="movie_id", compare=["ours"], **FAST)

    def test_exported(self):
        self.assertIn("paired_diff", ev.__all__)


if __name__ == "__main__":
    unittest.main()
