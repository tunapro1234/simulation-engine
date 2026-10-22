"""simengine.evidence gap fixes: one seal (#3), tolerant/correlation memorization audit (#11), fast multiplier and
bootstrap (#12), declared baselines + full ladder (#13), model ceiling (#14), group-level evaluation (#15) and the
"awaiting truth" report (#16). Offline: FakeLLM and small synthetic data only."""
import json
import tempfile
import time
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import simengine as se
import simengine.evidence as ev
from simengine.evidence import bedel, bootstrap, metrik
from simengine.llm import FakeLLM


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _cont(n=400, seed=0, groups=40):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, groups, n)
    y = rng.normal(0, 1, groups)[g] + rng.normal(0, 1, n)
    return pd.DataFrame({"group": g, "truth": y, "pred_good": y + rng.normal(0, 0.6, n),
                         "pred_raw": y + rng.normal(0, 1.5, n), "pred_noise": rng.normal(0, 1, n)})


def _bin(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (x + rng.normal(0, 0.8, n) > 0).astype(int)
    return pd.DataFrame({"truth": y, "pred_llm": 1 / (1 + np.exp(-0.4 * x)), "pred_ours": 1 / (1 + np.exp(-2 * x))})


FAST = dict(B=200, B_aggregate=50, multiplier=False)


# ------------------------------------------------------------------ #3 one seal
class TestOneSeal(unittest.TestCase):
    def test_top_level_seal_is_evidence_seal(self):
        self.assertIs(se.seal, ev.seal)
        from simengine import research
        self.assertTrue(callable(research.seal))
        self.assertIsNot(research.seal, ev.seal)

    def test_evaluate_finds_evidence_chain(self):
        df = _cont(200)
        with tempfile.TemporaryDirectory() as t:
            reg = Path(t) / ".onkayit"
            rec = se.seal(df.drop(columns=["truth"]), "before truth", registry=reg)
            self.assertTrue((reg / "zincir.jsonl").exists())
            self.assertEqual(len(ev.chain_paths(reg)), 1)
            r = ev.evaluate(df, ours="good", raw="raw", prereg=reg, **FAST)
            self.assertEqual([s["sha256"] for s in ev.summary(r)["seals"]], [rec["sha256"]])

    def test_evaluate_finds_research_chain(self):
        from simengine import research
        df = _cont(200)
        with tempfile.TemporaryDirectory() as t:
            reg = Path(t) / ".prereg"
            rec = research.seal(df.drop(columns=["truth"]), "research seal", registry_dir=reg)
            self.assertTrue((reg / "chain.jsonl").exists())
            recs = ev.chain(reg)
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["zincir"], "chain.jsonl")
            self.assertEqual(recs[0]["sha256"], rec["sha256"])
            with warnings.catch_warnings():
                warnings.simplefilter("error")      # a found chain must not warn
                r = ev.evaluate(df, ours="good", raw="raw", prereg=reg, **FAST)
            s = ev.summary(r)["seals"]
            self.assertEqual(s[0]["sha256"], rec["sha256"])
            self.assertEqual(s[0]["description"], "research seal")
            self.assertEqual(r["onkayit"][0]["zincir"], "chain.jsonl")

    def test_default_dir_fallback_and_both_files(self):
        from simengine import research
        df = _cont(120)
        with tempfile.TemporaryDirectory() as t:
            research.seal(df.drop(columns=["truth"]), "r", registry_dir=Path(t) / ".prereg")
            # asking for the evidence default dir finds the research chain next to it
            self.assertEqual([p.name for p in ev.chain_paths(Path(t) / ".onkayit")], ["chain.jsonl"])
            one = Path(t) / "both"
            ev.seal(df.drop(columns=["truth"]), "e", registry=one)
            research.seal(df.drop(columns=["truth", "pred_noise"]), "r2", registry_dir=one)
            self.assertEqual(len(ev.chain(one)), 2)

    def test_missing_chain_warns_naming_both_files(self):
        with tempfile.TemporaryDirectory() as t:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ev.evaluate(_cont(120), ours="good", raw="raw", prereg=Path(t) / "nothing", **FAST)
            msg = " ".join(str(x.message) for x in w)
            self.assertIn("zincir.jsonl", msg)
            self.assertIn("chain.jsonl", msg)


# ------------------------------------------------------------------ #11 memorization audits
class TestMemorizationTolerance(unittest.TestCase):
    def test_absolute_default_unchanged(self):
        self.assertTrue(ev.answer_hit("0.51", 0.5))
        self.assertFalse(ev.answer_hit("53", 50))           # ±0.02 absolute on a 0-100 value
        self.assertTrue(ev.answer_hit("51%", 0.5))           # percent fallback kept

    def test_relative_tolerance(self):
        self.assertTrue(ev.answer_hit("51", 50, 0.05, relative=True))
        self.assertFalse(ev.answer_hit("56", 50, 0.05, relative=True))
        self.assertTrue(ev.answer_hit("0.51", 0.5, 0.05, relative=True))

    def test_memorization_audit_relative(self):
        items = [{"prompt": f"q{i}", "expected": 40 + i, "control": i % 2 == 1} for i in range(40)]
        llm = FakeLLM(lambda s, u: str(40 + int(u[1:]) + 1))        # always off by one point
        a = ev.memorization_audit(items, llm, B=200)
        b = ev.memorization_audit(items, llm, B=200, tolerance=0.05, relative=True)
        self.assertEqual(a["tolerans"], 0.02)
        self.assertFalse(a["goreli"])
        self.assertTrue(b["goreli"])
        self.assertGreater(b["isabet_gercek"] if "isabet_gercek" in b else 1, -1)
        self.assertNotEqual(json.dumps(a, default=str), json.dumps(b, default=str))


class TestCorrelationAudit(unittest.TestCase):
    truth = {f"c{i}": float(v) for i, v in enumerate(np.linspace(20, 80, 30))}

    def _llm(self, memorized: bool):
        rng = np.random.default_rng(0)
        noise = {k: float(rng.normal(0, 30)) for k in self.truth}
        seen = []

        def resp(system, user):
            seen.append(user)
            key = user.split(" for ")[1].split("?")[0]
            named = "In the study" in user
            v = self.truth[key] + (1 if (named and memorized) else noise[key])
            return f"About {v:.1f}"
        return FakeLLM(resp), seen

    def test_prompts_ask_for_best_guess(self):
        n, b = ev.audit_prompts("'Some study' (A. B., Journal 2020)", "mean support (0-100)")
        self.assertIn("best guess", n.lower())
        self.assertIn("best guess", b.lower())
        self.assertIn("{key}", n)
        self.assertNotIn("Some study", b)

    def test_memorized_named_arm_flags(self):
        llm, seen = self._llm(True)
        n, b = ev.audit_prompts("'Some study'", "mean support")
        r = ev.correlation_audit(self.truth, n, b, llm=llm, B=300)
        self.assertEqual(r["durum"], "uyari")
        self.assertGreater(r["spearman_named"], r["spearman_blind"])
        self.assertGreater(r["alt"], 0)
        self.assertEqual(r["n"], 30)
        self.assertTrue(all("best guess" in u.lower() for u in seen))

    def test_not_memorized_ok_and_template_gets_best_guess(self):
        llm, seen = self._llm(False)
        r = ev.correlation_audit(self.truth, "In the study X, what was the value for {key}?",
                                 "What would the value be for {key}?", llm=llm, B=300)
        self.assertEqual(r["durum"], "tamam")
        self.assertTrue(all("best guess" in u.lower() for u in seen))

    def test_given_answers_and_unparsed(self):
        named = {k: (v if i % 3 else "I can't recall") for i, (k, v) in enumerate(self.truth.items())}
        blind = {k: 50.0 + (i % 5) for i, k in enumerate(self.truth)}
        r = ev.correlation_audit(self.truth, named, blind, B=200)
        self.assertEqual(r["n_unparsed_named"], 10)
        self.assertEqual(r["n"], 20)
        few = ev.correlation_audit({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "c": 3}, {"a": 3, "b": 2, "c": 1})
        self.assertEqual(few["durum"], "atlandi")

    def test_result_feeds_evaluate(self):
        r = ev.correlation_audit(self.truth, dict(self.truth), {k: 50.0 + (i % 7) for i, k in enumerate(self.truth)},
                                 B=200)
        rep = ev.evaluate(_cont(150), ours="good", raw="raw", memorization=r, **FAST)
        self.assertTrue(any("ezber" in str(d.get("kontrol")).lower() for d in rep["denetim"]))


# ------------------------------------------------------------------ #12 speed
class TestSpeed(unittest.TestCase):
    def test_fast_unit_bootstrap_is_exact_below_threshold(self):
        rng = np.random.default_rng(3)
        y = rng.normal(size=300)
        cols = [(y, y + rng.normal(0, 0.5, 300))]
        slow = bedel.olc(cols, n=50, R=30, B=50, hizli=False)
        auto = bedel.olc(cols, n=50, R=30, B=50)
        self.assertEqual(json.dumps(slow, default=str), json.dumps(auto, default=str))
        fast = bedel.olc(cols, n=50, R=30, B=50, hizli=True)
        self.assertTrue(fast.get("hizli_bootstrap"))
        self.assertFalse(slow.get("hizli_bootstrap", False))

    def test_fast_toplu_metrics_match_expanded_sample(self):
        rng = np.random.default_rng(1)
        fns = [metrik._auc, metrik._f1_ikili, metrik._kappa_ikili, metrik._dengeli_ikili, metrik._ece_ikili,
               metrik._spearman]
        for t in range(60):
            n = int(rng.integers(4, 50))
            y = (rng.random(n) < rng.random()).astype(float)
            p = np.round(rng.random(n), 1) if t % 5 else np.full(n, 0.7)
            k = bootstrap._sira_kodu(y, p, None)
            w = rng.integers(0, 3, n).astype(float)
            w[0] += 1
            ii = np.repeat(np.arange(n), w.astype(int))
            for fn in fns:
                m = metrik.Metrik(fn.__name__, "toplu", fn, 1)
                a = _quiet(bootstrap._hizli_metrik(m, k), w)
                b = _quiet(bootstrap._guvenli, m, y[ii], p[ii], None)
                if np.isnan(b):
                    self.assertTrue(np.isnan(a), fn.__name__)
                else:
                    self.assertAlmostEqual(a, b, places=12, msg=fn.__name__)

    def test_20k_rows_8_methods_fast(self):
        rng = np.random.default_rng(0)
        N = 20000
        g = rng.integers(0, 43, N)
        y = rng.normal(50, 10, 43)[g] + rng.normal(0, 15, N)
        df = pd.DataFrame({"truth": y, "group": g})
        for k in range(8):
            df[f"pred_m{k}"] = y * k / 8 + rng.normal(0, 10, N)
        t0 = time.time()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = ev.evaluate(df, ours="m7", raw="m3", task="continuous", primary="spearman", B=300, B_aggregate=100)
        dt = time.time() - t0
        self.assertLess(dt, 15, f"20k-row evaluate took {dt:.1f}s")
        self.assertTrue(any("multiplier" in str(x.message) for x in w))
        self.assertEqual(r["orneklem"]["alt_orneklem"]["n"], 2000)
        self.assertTrue(any("alt örnekleminde" in s for s in r["orneklem"]["notlar"]))
        self.assertEqual(len([x for x in ev.summary(r)["ladder"] if x["column"].startswith("tahmin_")]), 8)

    def test_small_frame_no_subsample_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = ev.evaluate(_cont(300), ours="good", raw="raw", B=100, B_aggregate=30)
        self.assertFalse(any("multiplier=True" in str(x.message) for x in w))
        self.assertNotIn("alt_orneklem", r.get("orneklem") or {})


# ------------------------------------------------------------------ #13 declared baselines, full ladder
class TestBaselines(unittest.TestCase):
    def test_oof_baseline_is_compared(self):
        df = _cont(400)
        rng = np.random.default_rng(9)
        df["pred_ridge"] = df["truth"] + rng.normal(0, 0.8, len(df))
        r = ev.evaluate(df, ours="good", raw="raw", baselines_oof=["ridge"], **FAST)
        s = ev.summary(r)
        self.assertEqual(s["declared_baselines"], ["tahmin_ridge"])
        self.assertEqual(s["best_baseline"]["column"], "tahmin_ridge")
        self.assertFalse(s["only_trivial_baselines"])
        self.assertEqual(r["tabanlar"]["oof"], ["tahmin_ridge"])
        rung = next(x for x in s["ladder"] if x["column"] == "tahmin_ridge")
        self.assertEqual(rung["kind"], "baseline")

    def test_only_trivial_flag(self):
        r = ev.evaluate(_cont(300), ours="good", raw="raw", **FAST)
        s = ev.summary(r)
        self.assertTrue(s["only_trivial_baselines"])
        self.assertIn("only trivial baselines", s["headline_en"])
        self.assertIn("trivial", ev.card_markdown(r).lower())

    def test_unknown_declared_baseline_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ev.evaluate(_cont(200), ours="good", raw="raw", baselines_oof=["nope"], **FAST)
        self.assertTrue(any("nope" in str(x.message) for x in w))

    def test_ladder_lists_all_methods(self):
        df = _cont(300)
        rng = np.random.default_rng(2)
        for k in range(6):
            df[f"pred_x{k}"] = df["truth"] + rng.normal(0, 1 + k, len(df))
        r = ev.evaluate(df, ours="good", raw="raw", **FAST)
        cols = {x["column"] for x in ev.summary(r)["ladder"]}
        self.assertTrue({"tahmin_" + c[5:] for c in df.columns if c.startswith("pred_")} <= cols)


# ------------------------------------------------------------------ #14 model ceiling
class TestModelCeiling(unittest.TestCase):
    def test_ceiling_col_pred(self):
        df = _cont(400)
        df["pred_oracle"] = df["truth"] + np.random.default_rng(4).normal(0, 0.2, len(df))
        r = ev.evaluate(df, ours="good", raw="raw", ceiling_col="pred_oracle", ceiling_label="Oracle", **FAST)
        s = ev.summary(r)
        self.assertEqual(r["tavan_turu"], "model")
        self.assertEqual(s["ceiling_kind"], "oracle/model")
        self.assertIsNotNone(s["ceiling"]["value"])
        self.assertIsNotNone(s["gap_closed"])
        self.assertNotIn("tahmin_oracle", {x["column"] for x in s["ladder"] if x["kind"] != "ceiling"} - {"tavan_model"})
        self.assertIn("Oracle/model ceiling", ev.card_markdown(r))
        self.assertIn("oracle/model ceiling", s["headline_en"])

    def test_ceiling_col_plain_and_binary(self):
        df = _bin(400)
        df["upper"] = np.clip(df["truth"] * 0.9 + 0.05, 0, 1)
        r = ev.evaluate(df, ours="ours", raw="llm", ceiling_col="upper", **FAST)
        self.assertEqual(r["tavan_turu"], "model")
        self.assertIsNotNone(ev.summary(r)["ceiling"]["value"])

    def test_human_retest_default(self):
        df = _cont(300)
        df["retest"] = df["truth"] + np.random.default_rng(5).normal(0, 0.5, len(df))
        r = ev.evaluate(df, ours="good", raw="raw", **FAST)
        self.assertEqual(ev.summary(r)["ceiling_kind"], "human retest")

    def test_bad_ceiling_col(self):
        with self.assertRaises(ValueError):
            ev.evaluate(_cont(100), ours="good", raw="raw", ceiling_col="missing", **FAST)
        with self.assertRaises(ValueError):
            ev.evaluate(_cont(100), ours="good", raw="raw", ceiling_col="pred_good", **FAST)


# ------------------------------------------------------------------ #15 group level
class TestAggregate(unittest.TestCase):
    def test_group_level(self):
        df = _cont(2000, groups=25)
        r = ev.evaluate(df, ours="good", raw="raw", aggregate_by="group", primary="spearman", **FAST)
        self.assertEqual(r["kart"]["n_kume"], 25)
        self.assertEqual(r["toplama"]["n_groups"], 25)
        self.assertEqual(r["toplama"]["n_rows"], 2000)
        self.assertEqual(r["duzey"], "grup")
        self.assertEqual(ev.summary(r)["aggregate"]["n_eval_rows"], 25)
        r2 = ev.evaluate_aggregate(df, "group", ours="good", raw="raw", primary="spearman", **FAST)
        self.assertEqual(r2["manset"]["bizim"]["deger"], r["manset"]["bizim"]["deger"])
        means = df.groupby("group")[["truth", "pred_good"]].mean()
        from scipy.stats import spearmanr
        self.assertAlmostEqual(r["manset"]["bizim"]["deger"], spearmanr(means.truth, means.pred_good)[0], places=10)

    def test_other_key_and_errors(self):
        df = _cont(600, groups=20)
        df["country"] = "k" + (df["group"] % 10).astype(str)
        r = ev.evaluate(df, ours="good", raw="raw", aggregate_by="country", **FAST)
        self.assertEqual(r["toplama"]["n_groups"], 10)
        with self.assertRaises(ValueError):
            ev.evaluate(df, ours="good", aggregate_by="nope", **FAST)


# ------------------------------------------------------------------ #16 awaiting truth
class TestAwaitingTruth(unittest.TestCase):
    def test_no_truth_column(self):
        df = _cont(200).drop(columns=["truth"])
        with tempfile.TemporaryDirectory() as t:
            ev.seal(df, "sealed early", registry=Path(t) / ".onkayit")
            r = ev.evaluate(df, ours="good", raw="raw", prereg=Path(t) / ".onkayit", **FAST)
            s = ev.summary(r)
            self.assertTrue(s["awaiting_truth"])
            self.assertEqual(s["badge"], "awaiting truth")
            self.assertEqual(len(s["seals"]), 1)
            self.assertEqual(set(s["predictions"]), {"tahmin_good", "tahmin_raw", "tahmin_noise"} & set(s["predictions"]))
            self.assertGreaterEqual(len(s["predictions"]), 3)
            self.assertIn("sealed", s["headline_en"])
            md = ev.card_markdown(r)
            self.assertIn("awaiting truth", md.lower())
            self.assertIn("Predictions (awaiting truth)", md)
            html = ev.dashboard_html([r])
            self.assertIn("<html", html.lower())
            json.dumps(r, default=str)

    def test_all_nan_truth_unsealed(self):
        df = _bin(150)
        df["truth"] = np.nan
        r = ev.evaluate(df, ours="ours", raw="llm", **FAST)
        s = ev.summary(r)
        self.assertTrue(s["awaiting_truth"])
        self.assertIn("NOT sealed", s["headline_en"])
        self.assertEqual(r["merdiven"], [])

    def test_partial_truth_rows_dropped(self):
        df = _cont(300)
        df.loc[:49, "truth"] = np.nan
        r = ev.evaluate(df, ours="good", raw="raw", **FAST)
        self.assertFalse(ev.summary(r)["awaiting_truth"])
        self.assertEqual(r["kart"]["n_satir"], 250)
        self.assertTrue(any("gerçeği eksik" in str(x) for x in json.dumps(r, ensure_ascii=False).split("\"")))


if __name__ == "__main__":
    unittest.main()
