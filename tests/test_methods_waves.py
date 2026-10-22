"""methods.waves: repeated-survey ladder (D6#1, D6#7a). Offline: synthetic waves with a regime switch every
4 waves, an in-party approval item and a stable item."""
import unittest
import warnings

import numpy as np
import pandas as pd

from simengine.methods.waves import (BacktestResult, merge_frames, mixture_reweight, one_minus_tvd,
                                     pooled_shrunk, role_swap, same_regime, subgroup_mixture,
                                     transition_backtest, wave_persistence)

CATS = {"approve": ["yes", "no"], "q2": ["a", "b", "c"]}
REGIME = {w: ("D" if ((w - 2000) // 4) % 2 == 0 else "R") for w in range(2000, 2013)}
SEGS = ["s0", "s1", "s2", "s3"]
DEM_SHARE = {"s0": 0.8, "s1": 0.6, "s2": 0.35, "s3": 0.15}
Q2 = {"s0": [0.6, 0.3, 0.1], "s1": [0.4, 0.4, 0.2], "s2": [0.2, 0.5, 0.3], "s3": [0.1, 0.3, 0.6]}


def _panel(waves=range(2000, 2013), n=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in waves:
        for s in SEGS:
            party = np.where(rng.random(n) < DEM_SHARE[s], "Dem", "Rep")
            inparty = (party == "Dem") == (REGIME[w] == "D")
            yes = rng.random(n) < np.where(inparty, 0.85, 0.15)
            q2 = rng.choice(CATS["q2"], n, p=Q2[s])
            wt = rng.uniform(0.5, 1.5, n)
            for i in range(n):
                rows.append({"year": w, "seg": s, "q": "approve", "ans": "yes" if yes[i] else "no",
                             "pid": party[i], "w": wt[i], "regime": REGIME[w], "rid": f"{w}-{s}-{i}"})
                rows.append({"year": w, "seg": s, "q": "q2", "ans": q2[i], "pid": party[i], "w": wt[i],
                             "regime": REGIME[w], "rid": f"{w}-{s}-{i}"})
    return pd.DataFrame(rows)


DF = _panel()
COLS = dict(wave="year", segment="seg", item="q", value="ans", categories=CATS)


def _emp(df, s, it, cats, weight=None):
    d = df[(df.seg == s) & (df.q == it) & df.ans.isin(cats)]
    w = d[weight] if weight else pd.Series(1.0, index=d.index)
    c = w.groupby(d.ans).sum().reindex(cats).fillna(0).to_numpy(float)
    return c / c.sum()


def _cell(fr, s, it, col):
    return np.asarray(fr.loc[(fr.segment == s) & (fr.item == it), col].iloc[0], float)


class Persistence(unittest.TestCase):
    def test_k1_copies_last_wave_and_frame_layout(self):
        fr = wave_persistence(DF, **COLS, target=2010)
        self.assertEqual(len(fr), 8)
        for c in ["unit", "truth", "pred_persistence", "segment", "item", "group", "wave", "n",
                  "n_train_persistence"]:
            self.assertIn(c, fr.columns)
        self.assertEqual(fr.unit.iloc[0].count("|"), 1)
        self.assertTrue((fr.group == fr.item).all())
        last = DF[DF.year == 2009]
        np.testing.assert_allclose(_cell(fr, "s2", "q2", "pred_persistence"), _emp(last, "s2", "q2", CATS["q2"]))
        np.testing.assert_allclose(_cell(fr, "s2", "q2", "truth"), _emp(DF[DF.year == 2010], "s2", "q2",
                                                                         CATS["q2"]))
        self.assertEqual(int(fr.n.iloc[0]), 300)

    def test_k_pools_counts_weighted(self):
        fr = wave_persistence(DF, **COLS, target=2010, k=3, weight="w")
        win = DF[DF.year.isin([2007, 2008, 2009])]
        np.testing.assert_allclose(_cell(fr, "s0", "approve", "pred_persistence"),
                                   _emp(win, "s0", "approve", CATS["approve"], "w"))
        self.assertLess(float(fr.n_train_persistence.iloc[0]), 900)   # Kish effective size < rows

    def test_future_target_and_no_leakage(self):
        fr = wave_persistence(DF, **COLS)            # target = 2013, not in df
        self.assertTrue((fr.wave == 2013).all())
        self.assertTrue(fr.truth.isna().all())
        a = wave_persistence(DF, **COLS, target=2006)
        b = wave_persistence(DF[DF.year <= 2006], **COLS, target=2006)
        pd.testing.assert_frame_equal(a, b)

    def test_missing_segment_gets_population(self):
        fr = wave_persistence(DF, **COLS, target=2010, segments=SEGS + ["new"])
        np.testing.assert_allclose(_cell(fr, "new", "q2", "pred_persistence"),
                                   _emp(DF[DF.year == 2009].assign(seg="x"), "x", "q2", CATS["q2"]))
        self.assertEqual(float(fr.loc[fr.segment == "new", "n_train_persistence"].iloc[0]), 0.0)

    def test_wide_frame_equals_long(self):
        wide = DF.pivot_table(index=["rid", "year", "seg"], columns="q", values="ans", aggfunc="first")
        wide = wide.reset_index()
        a = wave_persistence(DF, **COLS, target=2011).sort_values("unit").reset_index(drop=True)
        cols = dict(COLS, item=None, value=["approve", "q2"])
        b = wave_persistence(wide, **cols, target=2011).sort_values("unit").reset_index(drop=True)
        for x, y in zip(a.pred_persistence, b.pred_persistence):
            np.testing.assert_allclose(x, y)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            wave_persistence(DF, **COLS, target=2000)
        with self.assertRaises(ValueError):
            wave_persistence(DF, **COLS, k=0)


class PooledShrunk(unittest.TestCase):
    def test_m0_is_pooled_and_large_m_is_population(self):
        a = pooled_shrunk(DF, **COLS, target=2010, k=4, m=0)
        b = wave_persistence(DF, **COLS, target=2010, k=4)
        for x, y in zip(a.pred_pooled_shrunk, b.pred_persistence):
            np.testing.assert_allclose(x, y)
        big = pooled_shrunk(DF, **COLS, target=2010, k=4, m=1e9)
        pop = _emp(DF[DF.year.isin(range(2006, 2010))].assign(seg="x"), "x", "q2", CATS["q2"])
        np.testing.assert_allclose(_cell(big, "s3", "q2", "pred_pooled_shrunk"), pop, atol=1e-5)

    def test_formula_and_prior_dict(self):
        prior = {"approve": [0.5, 0.5], "q2": [1 / 3, 1 / 3, 1 / 3]}
        fr = pooled_shrunk(DF, **COLS, target=2010, k=2, m=100, prior=prior)
        own = _emp(DF[DF.year.isin([2008, 2009])], "s1", "q2", CATS["q2"])
        np.testing.assert_allclose(_cell(fr, "s1", "q2", "pred_pooled_shrunk"),
                                   (600 * own + 100 * np.full(3, 1 / 3)) / 700)
        with self.assertRaises(ValueError):
            pooled_shrunk(DF, **COLS, target=2010, prior="bogus")


class SameRegime(unittest.TestCase):
    def test_picks_same_regime_waves_and_beats_persistence_at_switch(self):
        # 2000-2003 D, 2004-2007 R, 2008 D: the same-regime rung pools 2000-2003, persistence copies 2007 (R)
        fr = same_regime(DF, **COLS, regime="regime", target=2008, k=4)
        self.assertEqual(fr.attrs["waves"], [2000, 2001, 2002, 2003])
        self.assertEqual(fr.attrs["target_regime"], "D")
        per = wave_persistence(DF, **COLS, target=2008)
        ap = fr.item == "approve"
        s_reg = np.mean([one_minus_tvd(p, t) for p, t in zip(fr.pred_same_regime[ap], fr.truth[ap])])
        s_per = np.mean([one_minus_tvd(p, t) for p, t in zip(per.pred_persistence[ap.to_numpy()],
                                                              per.truth[ap.to_numpy()])])
        self.assertGreater(s_reg, s_per + 0.2)

    def test_mapping_target_regime_and_fallback(self):
        fr = same_regime(DF, **COLS, regime=REGIME, target=2013, target_regime="R", k=2)
        self.assertEqual(fr.attrs["waves"], [2007, 2012])
        with self.assertWarns(UserWarning):
            fr = same_regime(DF, **COLS, regime="regime", target=2002, target_regime="R", k=2)
        self.assertEqual(fr.attrs["waves"], [2000, 2001])
        with self.assertRaises(ValueError):
            same_regime(DF, **COLS, regime="regime", target=2013)      # regime unknown
        bad = DF.copy()
        bad.loc[bad.index[:5], "regime"] = "R"
        with self.assertRaises(ValueError):
            same_regime(bad, **COLS, regime="regime", target=2008)


class Mixture(unittest.TestCase):
    def test_mixture_reweight_and_role_swap(self):
        cond = {"Dem": [0.9, 0.1], "Rep": [0.2, 0.8]}
        np.testing.assert_allclose(mixture_reweight(cond, {"Dem": 3, "Rep": 1}), [0.725, 0.275])
        with self.assertWarns(UserWarning):
            np.testing.assert_allclose(mixture_reweight(cond, {"Dem": 1, "Ind": 1}), [0.9, 0.1])
        with self.assertRaises(ValueError), self.assertWarns(UserWarning):
            mixture_reweight(cond, {"Ind": 1})
        sw = role_swap(dict(cond, Ind=[0.5, 0.5]), {"Dem": "Rep", "Rep": "Dem"})
        self.assertEqual(sw["Dem"], [0.2, 0.8])
        self.assertEqual(sw["Ind"], [0.5, 0.5])
        with self.assertRaises(KeyError):
            role_swap(cond, {"Dem": "Ind"})

    def test_swap_recovers_switch(self):
        # 2012 is R after D 2008-2011: last-wave conditionals swapped by party fit the new in-party pattern
        kw = dict(**COLS, subgroup="pid", target=2012, k=1)
        plain = subgroup_mixture(DF, **kw)
        swp = subgroup_mixture(DF, **kw, swap={"Dem": "Rep", "Rep": "Dem"}, swap_items=["approve"],
                               method="mix_swap")
        ap = plain.item == "approve"
        sc = lambda f, c: np.mean([one_minus_tvd(p, t) for p, t in zip(f[c][ap], f.truth[ap])])
        self.assertGreater(sc(swp, "pred_mix_swap"), sc(plain, "pred_mixture") + 0.2)
        self.assertGreater(sc(swp, "pred_mix_swap"), 0.93)
        q = plain.item == "q2"   # not swapped: identical
        for x, y in zip(plain.pred_mixture[q], swp.pred_mix_swap[q]):
            np.testing.assert_allclose(x, y)

    def test_weight_variants_and_segment_level(self):
        kw = dict(**COLS, subgroup="pid", target=2012)
        flat = subgroup_mixture(DF, **kw, weights={"Dem": 1, "Rep": 1})
        self.assertEqual(len({tuple(np.round(p, 9)) for p in flat.pred_mixture[flat.item == "q2"]}), 1)
        per = subgroup_mixture(DF, **kw, weights={s: {"Dem": 1.0, "Rep": 0.0} for s in SEGS})
        dem = _emp(DF[(DF.year == 2011) & (DF.pid == "Dem")].assign(seg="x"), "x", "q2", CATS["q2"])
        np.testing.assert_allclose(_cell(per, "s3", "q2", "pred_mixture"), dem)
        tgt = subgroup_mixture(DF, **kw, weights="target")
        self.assertEqual(len(tgt), 8)
        with self.assertRaises(ValueError):
            subgroup_mixture(DF, **kw, weights="bogus")
        with self.assertRaises(ValueError):    # swapping every item would swap stable items too
            subgroup_mixture(DF, **kw, swap={"Dem": "Rep", "Rep": "Dem"})
        seg = subgroup_mixture(DF, **kw, m=10, k=4)
        # segment x subgroup cells shrunk toward the subgroup: q2 now depends on the segment again
        self.assertGreater(one_minus_tvd(_cell(seg, "s0", "q2", "pred_mixture"), Q2["s0"]),
                           one_minus_tvd(_cell(subgroup_mixture(DF, **kw, k=4), "s0", "q2", "pred_mixture"),
                                         Q2["s0"]))


    def test_target_weights_from_labels_without_answers(self):
        # future wave: subgroup labels known, answers missing; s0 is all Rep there
        fut = DF[DF.year == 2012].assign(year=2013, ans=np.nan)
        fut.loc[fut.seg == "s0", "pid"] = "Rep"
        d = pd.concat([DF, fut], ignore_index=True)
        fr = subgroup_mixture(d, **COLS, subgroup="pid", target=2013, weights="target")
        rep = _emp(DF[(DF.year == 2012) & (DF.pid == "Rep")].assign(seg="x"), "x", "approve", CATS["approve"])
        np.testing.assert_allclose(_cell(fr, "s0", "approve", "pred_mixture"), rep)
        self.assertTrue(fr.truth.isna().all())


class Backtest(unittest.TestCase):
    def test_by_wave_summary_and_evaluate(self):
        methods = {"t0": (wave_persistence, {}), "t2": (pooled_shrunk, {"k": 4, "m": 20}),
                   "t4b": (same_regime, {"regime": "regime", "k": 4}),
                   "flat": lambda d, t: wave_persistence(d, **COLS, target=t, k=2, method="x")}
        bt = transition_backtest(DF, waves=[2008, 2012], methods=methods, **COLS)
        self.assertIsInstance(bt, BacktestResult)
        self.assertEqual(list(bt.by_wave.index), [2008, 2012])
        self.assertEqual(set(bt.by_wave.columns), set(methods))
        self.assertEqual(bt.summary.index[0], "t4b")
        self.assertEqual(int(bt.summary.loc["t4b", "waves_led"]), 2)
        self.assertEqual(bt.waves_used[(2012, "t4b")], [2004, 2005, 2006, 2007])
        self.assertEqual(len(bt.frame), 16)
        self.assertIn("pred_flat", bt.frame.columns)
        import simengine.evidence as ev
        fr = bt.frame.assign(unit=bt.frame.wave.astype(str) + "|" + bt.frame.unit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = ev.evaluate(fr, ours="t4b", task="distribution", B=100, B_aggregate=50, multiplier=False,
                              unlabeled_baselines=["t0", "t2", "flat"], ours_uses_llm=False)
        self.assertIsInstance(rep, dict)

    def test_merge_frames_rejects_duplicate_methods(self):
        a = wave_persistence(DF, **COLS, target=2010)
        with self.assertRaises(ValueError):
            merge_frames([a, a])
        with self.assertRaises(ValueError):
            transition_backtest(DF, waves=[2020], methods={"t0": (wave_persistence, {})}, **COLS)


if __name__ == "__main__":
    unittest.main()
