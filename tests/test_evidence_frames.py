"""simengine.evidence.frames: repeated-split summary, counts frame, wide-to-long, segment bias (D9 #5 #6,
D11 #5d, D12 #3 #7 #20 #21). Synthetic frames only."""
import unittest
import warnings

import numpy as np
import pandas as pd

import simengine.evidence as ev


def _split_frame(seed: int, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    return pd.DataFrame({"truth": y, "pred_t1": y + rng.normal(scale=0.6, size=n),
                         "pred_o1": y + rng.normal(scale=0.8, size=n), "group": np.arange(n)})


class SplitSummaryTest(unittest.TestCase):
    def test_frames_seed_loop_and_paired_share(self):
        ss = ev.split_summary(_split_frame, seeds=range(40), metric="mae", pairs=[("t1", "o1")])
        self.assertEqual(ss.n_splits, 40)
        self.assertEqual(ss.metric, "mae")
        self.assertEqual(ss.direction, -1)
        d = ss.table.set_index("name").loc["t1 - o1"]
        self.assertLess(d["median"], 0)                       # t1 has the smaller error
        self.assertLessEqual(d["q025"], d["median"])
        self.assertLessEqual(d["median"], d["q975"])
        per = ss.per_split[ss.per_split.kind == "diff"]
        self.assertEqual(len(per), 40)
        self.assertAlmostEqual(d["share_a_better"], float((per["value"] < 0).mean()))
        self.assertGreater(d["share_a_better"], 0.9)
        # the per-split value equals the plain MAE of that split
        f = _split_frame(3)
        v = ss.per_split[(ss.per_split.split == 3) & (ss.per_split.name == "t1")]["value"].item()
        self.assertAlmostEqual(v, float(np.mean(np.abs(f.truth - f.pred_t1))))
        self.assertIn("t1 - o1", ss.markdown)
        self.assertIn("not a CI", ss.markdown)
        self.assertEqual(len(ss.limits), 1)
        self.assertIn("repeated splits (n=40)", ss.limits[0])

    def test_reports_default_pair_is_ours_vs_best_baseline(self):
        reps = {f"s{s}": ev.evaluate(_split_frame(s), "T", ours="t1", raw="o1", B=50, multiplier=False)
                for s in range(3)}
        ss = ev.split_summary(reps)
        self.assertEqual(ss.metric, "pearson")
        self.assertEqual(list(ss.table.query("kind == 'diff'").name), ["ours - best_baseline"])
        self.assertEqual(set(ss.per_split.split), {"s0", "s1", "s2"})
        self.assertEqual(ss.table.query("kind == 'diff'").share_a_better.item(), 1.0)
        # explicit pairs read the report's columns; limits go to evaluate(limits=)
        ss2 = ev.split_summary(list(reps.values()), metric="mae", pairs=[("pred_t1", "o1")])
        self.assertEqual(ss2.metric, "mae")
        rep = ev.evaluate(_split_frame(9), "T", ours="t1", B=50, multiplier=False, limits=ss2.limits)
        self.assertTrue(any("repeated splits" in s for s in rep["sinirlar"]))

    def test_best_baseline_follows_the_summarised_metric(self):
        # the headline's best baseline is picked on the primary (pearson: random); on mae the strongest baseline is
        # the marginal one, and ours - best_baseline must not compare against the mae-weakest baseline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reps = [ev.evaluate(_split_frame(s), "T", ours="t1", B=20, multiplier=False) for s in range(3)]
        ss = ev.split_summary(reps, metric="mae")
        diff = ss.per_split.query("kind == 'diff'")["value"].to_numpy()
        for r, dv in zip(reps, diff):
            res = r["sonuclar"]
            best = min(v["metrikler"]["mae"]["deger"] for v in res.values() if v.get("tur") == "baseline")
            ours = res[r["manset"]["bizim"]["sutun"]]["metrikler"]["mae"]["deger"]
            self.assertAlmostEqual(dv, ours - best)
        # on the primary it stays the headline's best baseline
        ss = ev.split_summary(reps)
        for r, dv in zip(reps, ss.per_split.query("kind == 'diff'")["value"]):
            res, m = r["sonuclar"], r["manset"]
            self.assertAlmostEqual(dv, res[m["bizim"]["sutun"]]["metrikler"]["pearson"]["deger"]
                                   - res[m["en_iyi_baseline"]["sutun"]]["metrikler"]["pearson"]["deger"])

    def test_errors(self):
        with self.assertRaises(ValueError):
            ev.split_summary(_split_frame)                                     # callable without seeds
        with self.assertRaises(ValueError):
            ev.split_summary([_split_frame(0)])                                # one split
        with self.assertRaises(ValueError):
            ev.split_summary([_split_frame(0), _split_frame(1)], pairs=[("t1", "nope")])


class CountsFrameTest(unittest.TestCase):
    def _people(self):
        # two areas; class "a" is the majority everywhere, the calibrated probabilities match the truth shares
        rng = np.random.default_rng(1)
        rows = []
        for area, share in (("x", [0.6, 0.3, 0.1]), ("y", [0.5, 0.2, 0.3])):
            for _ in range(400):
                rows.append({"area": area, "region": "r1" if area == "x" else "r2",
                             "truth": rng.choice(["a", "b", "c"], p=share), "p_vec": np.array(share)})
        return pd.DataFrame(rows)

    def test_argmax_trap_label_vs_expected_counts(self):
        p = self._people()
        cf = ev.counts_frame(p, {"probs": "p_vec"}, by="area", probs=True, classes=["a", "b", "c"], argmax_too=True)
        self.assertEqual(len(cf), 6)                                           # 2 areas x 3 classes, zeros kept
        self.assertEqual(set(cf.columns) >= {"area", "class", "item", "group", "truth", "pred_probs",
                                             "pred_probs_argmax", "n"}, True)
        self.assertEqual(cf.truth.sum(), 800)
        self.assertAlmostEqual(cf.pred_probs.sum(), 800)
        self.assertEqual(cf.pred_probs_argmax.sum(), 800)
        # argmax puts everyone in "a": zero predicted count for b / c
        self.assertEqual(cf[cf["class"] != "a"].pred_probs_argmax.sum(), 0)
        wm = {a: (cf.truth - cf["pred_" + a]).abs().sum() / cf.truth.sum() for a in ("probs", "probs_argmax")}
        self.assertLess(wm["probs"], wm["probs_argmax"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # 2 area clusters: few-cluster warning is expected here
            rep = ev.evaluate(cf, "density", ours="probs", task="forecast", primary="wmape", B=50)
        self.assertEqual(rep["birincil"], "wmape")
        self.assertAlmostEqual(rep["sonuclar"]["tahmin_probs"]["metrikler"]["wmape"]["deger"], wm["probs"])

    def test_label_arm_cluster_dict_probs_and_missing(self):
        p = self._people()
        p["lab"] = "a"
        p.loc[0, "lab"] = None
        p["pa"], p["pb"], p["pc"] = 0.5, 0.3, 0.2
        cf = ev.counts_frame(p, {"lab": "lab", "pr": {"a": "pa", "b": "pb", "c": "pc"}}, by=["area"],
                             probs=["pr"], cluster="region")
        self.assertEqual(sorted(cf.group.unique()), ["r1", "r2"])
        x = cf[cf.area == "x"]
        self.assertTrue(x.pred_lab.isna().all())                               # missing prediction: not a zero count
        self.assertEqual(cf[cf.area == "y"].pred_lab.sum(), 400)
        self.assertAlmostEqual(x.set_index("class").pred_pr["b"], 120.0)
        with self.assertRaises(ValueError):
            ev.counts_frame(p, {"lab": "lab"}, by=["area"], cluster="truth")   # not constant within cells

    def test_weights_and_dict_rows(self):
        p = pd.DataFrame({"cell": ["u", "u", "v"], "truth": ["a", "b", "a"],
                          "pd": [{"a": 0.9, "b": 0.1}, {"a": 0.2, "b": 0.8}, {"a": 1.0}], "w": [2.0, 1.0, 1.0]})
        cf = ev.counts_frame(p, {"m": "pd"}, by="cell", probs=True, weight="w").set_index(["cell", "class"])
        self.assertAlmostEqual(cf.loc[("u", "a"), "truth"], 2.0)
        self.assertAlmostEqual(cf.loc[("u", "a"), "pred_m"], 2 * 0.9 + 0.2)
        self.assertAlmostEqual(cf.loc[("v", "b"), "pred_m"], 0.0)

    def test_probabilities_not_summing_to_one_warn(self):
        p = pd.DataFrame({"cell": ["u", "u"], "truth": ["a", "b"], "pv": [np.array([60.0, 40.0]), np.array([30.0, 70.0])]})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ev.counts_frame(p, {"m": "pv"}, by="cell", probs=True, classes=["a", "b"])
        self.assertTrue(any("do not sum to 1" in str(x.message) for x in w))


class LongFrameTest(unittest.TestCase):
    def test_sd_train_scaling_gives_zmae(self):
        rng = np.random.default_rng(2)
        ids = np.arange(50)
        train = pd.DataFrame({"id": np.arange(100), "t_small": rng.normal(0, 1, 100), "t_big": rng.normal(0, 100, 100)})
        truth = pd.DataFrame({"id": ids, "t_small": rng.normal(0, 1, 50), "t_big": rng.normal(0, 100, 50)})
        pred = truth.copy()
        pred["t_small"] += 0.5
        pred["t_big"] += 50.0
        lf = ev.long_frame(pred, truth, "id", scale="sd_train", train=train)
        self.assertEqual(list(lf.columns), ["group", "item", "truth", "pred_ours"])
        self.assertEqual(len(lf), 100)
        sd = train[["t_small", "t_big"]].std(ddof=1)
        self.assertAlmostEqual(lf.attrs["scale"]["t_big"], sd["t_big"])
        z = (lf.pred_ours - lf.truth).abs().groupby(lf["item"]).mean()
        self.assertAlmostEqual(z["t_small"], 0.5 / sd["t_small"])
        self.assertAlmostEqual(z["t_big"], 50.0 / sd["t_big"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # constant error: degenerate CI warning is expected here
            rep = ev.evaluate(lf, "zMAE", ours="ours", task="continuous", primary="mae", B=50, multiplier=False)
        self.assertAlmostEqual(rep["sonuclar"]["tahmin_ours"]["metrikler"]["mae"]["deger"], z.mean())

    def test_csv_arms_missing_units_and_errors(self):
        import tempfile, os
        truth = pd.DataFrame({"id": [1, 2, 3], "a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        p1 = pd.DataFrame({"id": [1, 2, 3], "a": [1.0, 2.0, 4.0], "b": [4.0, 5.0, 6.0]})
        p2 = pd.DataFrame({"id": [1, 2], "a": [0.0, 0.0], "b": [0.0, 0.0]})
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sealed.csv")
            p1.to_csv(path, index=False)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                lf = ev.long_frame({"m1": path, "m2": p2}, truth, "id", scale={"a": 2.0, "b": 1.0})
            self.assertTrue(any("no prediction" in str(x.message) for x in w))
        self.assertEqual(sorted(lf.group.unique()), [1, 2])
        self.assertAlmostEqual(lf[(lf.group == 2) & (lf["item"] == "a")].truth.item(), 1.0)
        with self.assertRaises(ValueError):
            ev.long_frame(p1, truth, "id", scale="sd_train")                  # no train=
        with self.assertRaises(ValueError):
            ev.long_frame(p1, truth, "id", scale={"a": 0.0, "b": 1.0})
        with self.assertRaises(ValueError):
            ev.long_frame(p1, truth, "id", targets=["a", "zz"])

    def test_csv_text_ids_and_label_columns(self):
        import tempfile, os
        # text ids with leading zeros survive the CSV round trip; numeric vs text ids match as text and keep the
        # truth's ids; a non-numeric column is not a default target; no shared unit raises instead of an empty frame
        truth = pd.DataFrame({"id": ["001", "002"], "name": ["u", "v"], "a": [1.0, 2.0]})
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sealed.csv")
            truth.assign(a=[1.5, 2.5]).to_csv(path, index=False)
            lf = ev.long_frame(path, truth, "id")
            self.assertEqual(sorted(lf.group), ["001", "002"])
            self.assertEqual(list(lf["item"].unique()), ["a"])
            num = pd.DataFrame({"id": [1, 2], "a": [1.0, 2.0]})
            num.assign(a=[0.0, 0.0]).to_csv(path, index=False)
            lf = ev.long_frame(path, num, "id")
            self.assertEqual(sorted(lf.group), [1, 2])
            self.assertAlmostEqual((lf.truth - lf.pred_ours).abs().mean(), 1.5)
        with self.assertRaises(ValueError), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev.long_frame(pd.DataFrame({"id": [7, 8], "a": [1.0, 2.0]}), num, "id")


class SegmentBiasTest(unittest.TestCase):
    def _frame(self, seed=0, n_seg=20, per=30, bias_b=0.0):
        rng = np.random.default_rng(seed)
        seg = np.repeat(np.arange(n_seg), per)
        y = rng.normal(size=len(seg)) + seg * 0.1
        return pd.DataFrame({"truth": y, "seg": seg, "group": np.arange(len(seg)) // 2,
                             "pred_ours": y + rng.normal(scale=1.0, size=len(seg)),
                             "pred_base": y + bias_b + rng.normal(scale=1.0, size=len(seg))})

    def test_point_matches_brief_formula(self):
        f = self._frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ev.segment_bias(f, "seg", ["ours", "base"], B=200)
        g = f.groupby("seg")
        hand = float((g.size() * (g.pred_ours.mean() - g.truth.mean()).abs()).sum() / len(f))
        self.assertAlmostEqual(r["arms"]["ours"]["value"], hand)
        eq = ev.segment_bias(f, "seg", ["ours"], weight="equal", B=50)
        self.assertAlmostEqual(eq["arms"]["ours"]["value"], float((g.pred_ours.mean() - g.truth.mean()).abs().mean()))
        self.assertEqual(r["cluster"], "group")
        self.assertEqual(len(r["segments"]), 20)

    def test_single_arm_ci_excludes_point_paired_diff_does_not(self):
        # defect D12 #21: with no true segment error the bootstrap of sum |mean error| sits above the point
        # (60 segments x 10 rows: outside in 29 of 30 seeds; the paired CI covered the true 0 in 30 of 30)
        f = self._frame(seed=0, n_seg=60, per=10)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = ev.segment_bias(f, "seg", ["ours", "base"], B=1000)
        a = r["arms"]["ours"]
        self.assertTrue(a["point_outside_ci"])
        self.assertLess(a["value"], a["lo"])
        self.assertTrue(any("outside its bootstrap CI" in str(x.message) for x in w))
        d = r["diff"][0]
        self.assertLessEqual(d["lo"], d["diff"])
        self.assertLessEqual(d["diff"], d["hi"])
        self.assertIn("paired difference", r["note"])

    def test_paired_diff_detects_biased_arm(self):
        f = self._frame(seed=1, bias_b=0.8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ev.segment_bias(f, "seg", ["ours", "pred_base"], B=500)
        d = r["diff"][0]
        self.assertEqual((d["a"], d["b"]), ("ours", "base"))
        self.assertTrue(d["a_better"])
        self.assertLess(d["hi"], 0)

    def test_one_cluster_has_no_ci_and_few_clusters_warn(self):
        # one cluster: every resample is the sample, so the CI collapsed on the point and p read 1/B = 0.005
        f = self._frame(n_seg=4, per=5).assign(group=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = ev.segment_bias(f, "seg", ["ours", "base"], B=200)
        self.assertTrue(any("bootstrap clusters" in str(x.message) for x in w))
        self.assertTrue(np.isnan(r["diff"][0]["p"]))
        self.assertTrue(np.isnan(r["arms"]["ours"]["lo"]))
        self.assertFalse(r["arms"]["ours"]["point_outside_ci"])
        f = self._frame(n_seg=4, per=5).assign(group=lambda x: x.index % 5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = ev.segment_bias(f, "seg", ["ours", "base"], B=200)
        self.assertTrue(any("only 5 bootstrap clusters" in str(x.message) for x in w))
        self.assertTrue(np.isfinite(r["diff"][0]["p"]))

    def test_errors(self):
        f = self._frame()
        with self.assertRaises(ValueError):
            ev.segment_bias(f, "nope", ["ours"])
        with self.assertRaises(ValueError):
            ev.segment_bias(f, "seg", ["missing"])


if __name__ == "__main__":
    unittest.main()
