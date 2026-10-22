"""subpop_opinionqa / global_opinionqa bench tasks: synthetic SubPOP and GlobalOpinionQA files (no real rows); the
real-data check runs only when $SIMENGINE_DATA is set."""
import os
import re
import tempfile
import unittest

import numpy as np
import pandas as pd

from simengine.bench.base import DataNotFound
from simengine.bench.tasks import subpop as sp
from simengine.llm.fake import FakeLLM
from simengine.types import Task

N_Q = 60
LEAN = {("SEX", "Male"): 0.0, ("SEX", "Female"): 0.05, ("POLPARTY", "Democrat"): -1.2,
        ("POLPARTY", "Republican"): 1.2, ("AGE", "18-29"): -0.4}


def _dist(z: float, m: int = 4) -> np.ndarray:
    p = np.exp(-0.5 * (np.arange(m) - (m - 1) / 2 - z) ** 2)
    return p / p.sum()


def write_subpop(d: str, seed: int = 0) -> None:
    """Groups lean on a 4-point scale; Female is Male's near twin, the parties are opposite. One qkey is asked with
    two texts, one question has constant ordinals (dropped) and 'Refused' is listed but has no share."""
    rng = np.random.default_rng(seed)
    meta = os.path.join(d, "data", "subpopulation_metadata")
    proc = os.path.join(d, "data", "opinionqa", "processed")
    os.makedirs(meta)
    os.makedirs(proc)
    pd.DataFrame({"attribute": ["SEX", "POLPARTY"], "group": ["['Male', 'Female']", "['Democrat', 'Republican']"]}
                 ).to_csv(os.path.join(meta, "demographics_22.csv"), index=False)
    pd.DataFrame({"attribute": ["SEX", "POLPARTY", "AGE"],
                  "group": ["['Male', 'Female']", "['Democrat', 'Republican']", "['18-29']"]}
                 ).to_csv(os.path.join(meta, "demographics_61.csv"), index=False)
    rows = []
    qs = [(f"Q{i:02d}_W{26 + i % 3}", f"Synthetic question {i}?") for i in range(N_Q)]
    qs += [("DUP_W30", "Synthetic variant one?"), ("DUP_W30", "Synthetic variant two?"), ("FLAT_W30", "Flat?")]
    for qk, text in qs:
        shift = rng.normal(0, .6)
        ordn = [1.0, 1.0, 1.0, 1.0] if qk.startswith("FLAT") else [1.0, 2.0, 3.0, 4.0]
        opts = str(["Strongly agree", "Agree", "Disagree", "Strongly disagree", "Refused"])
        ps = {g: _dist(shift + z + rng.normal(0, .05)) for g, z in LEAN.items()}
        ps[("Overall", "Overall")] = np.mean(list(ps.values()), 0)
        for (a, g), p in ps.items():
            rows.append({"qkey": qk, "attribute": a, "group": g, "responses": str([float(x) for x in p]),
                         "refusal_rate": 0.01, "ordinal": str(ordn), "question": text, "options": opts})
    pd.DataFrame(rows).to_csv(os.path.join(proc, "opinionqa.csv"), index=False)


COUNTRIES = ["Atlantis", "Borduria", "Carpania", "Dorne", "Elbonia", "Freedonia"]


def write_goqa(d: str, seed: int = 1) -> None:
    """Countries in two blocs (Atlantis~Borduria, Carpania~Dorne, ...); one single-country question, one all-zero row,
    selections written as SubPOP's defaultdict repr."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(d, "data"))
    rows = []
    for i in range(N_Q):
        m = 3 + i % 3
        base = rng.dirichlet(np.ones(m))
        sel = {}
        for k, c in enumerate(COUNTRIES):
            bloc = rng.dirichlet(np.ones(m) * 2) if k % 2 == 0 else sel[COUNTRIES[k - 1]]
            p = 0.5 * base + 0.5 * bloc if k % 2 == 0 else np.clip(np.asarray(bloc) + rng.normal(0, .01, m), 0, 1)
            sel[c] = [float(v) for v in p]
        if i == 0:
            sel["Elbonia"] = [0.0] * m
        if i % 2:                                           # a second sample of Atlantis on half the questions
            sel["Atlantis (Non-national sample)"] = sel["Atlantis"]
        opts = [f"Option {j}" for j in range(m - 1)] + ["DK/Refused"]
        if i == 1:
            opts[0] = 0.0                                   # numeric option label, as in some WVS rows
        rows.append({"question": f"Global question {i}?", "selections": f"defaultdict(<class 'list'>, {sel})",
                     "options": str(opts), "source": "GAS" if i % 4 else "WVS"})
    rows.append({"question": "Lonely question?", "selections": "defaultdict(<class 'list'>, {'Dorne': [0.5, 0.5]})",
                 "options": str(["Yes", "No"]), "source": "GAS"})
    pd.DataFrame(rows).to_csv(os.path.join(d, "data", "global_opinions.csv"), index=False)


class _Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        write_subpop(os.path.join(cls.root, "subpop-opinionqa"))
        write_goqa(os.path.join(cls.root, "globalopinionqa"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestDistances(unittest.TestCase):
    def test_ordinal_wd_matches_subpop(self):
        # SubPOP ordinal_emd docstring example 1: positions 1, 2, 1.5 -> 0.15
        self.assertAlmostEqual(sp.ordinal_wd([.1, .5, .4], [.2, .3, .5], [1, 2, 1.5]), 0.15)
        # example 2: negative ordinal dropped and the rest renormalised (the code, not the docstring's 0.1)
        self.assertAlmostEqual(sp.ordinal_wd([.2, .5, .3], [.3, .4, .3], [1, 2, -1]), 0.1 / 0.7)
        self.assertTrue(np.isnan(sp.ordinal_wd([.5, .5], [.2, .8], [1, 1])))
        # scale invariant in the ordinal values; 1 between the two ends
        self.assertAlmostEqual(sp.ordinal_wd([1, 0, 0], [0, 0, 1], [10, 20, 30]), 1.0)

    def test_tvd_and_js(self):
        from scipy.spatial.distance import jensenshannon
        rng = np.random.default_rng(0)
        for _ in range(20):
            p, q = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
            self.assertAlmostEqual(sp.js_distance(p, q), float(jensenshannon(p, q, base=2)), places=10)
            self.assertAlmostEqual(sp.tvd(p, q), 0.5 * np.abs(p - q).sum())
        self.assertAlmostEqual(sp.js_distance([1, 0], [0, 1]), 1.0)
        self.assertAlmostEqual(sp.tvd([1, 0], [0, 1]), 1.0)
        self.assertEqual(sp.js_distance([.3, .7], [.3, .7]), 0.0)

    def test_ordinal_wd_matches_reference_loop(self):
        """Line-by-line port of SubPOP's ordinal_emd (sort, drop negatives, renormalise, cumulative loop)."""
        def ref(l1, l2, ov):
            if max(ov) == min(ov):
                return float("nan")
            ov, l1, l2 = zip(*sorted(zip(ov, l1, l2)))
            k = next((i for i, v in enumerate(ov) if v >= 0), 0)
            ov, l1, l2 = ov[k:], np.array(l1[k:]) / sum(l1[k:]), np.array(l2[k:]) / sum(l2[k:])
            c1, c2 = np.cumsum(l1), np.cumsum(l2)
            return sum(abs(c1[i] - c2[i]) * (ov[i + 1] - ov[i]) for i in range(len(l1) - 1)) / (max(ov) - min(ov))
        rng = np.random.default_rng(3)
        for _ in range(200):
            m = int(rng.integers(3, 7))
            ov = list(rng.choice([1.0, 1.5, 2.0, 3.0, 4.0, -1.0], m))
            if len({v for v in ov if v >= 0}) < 2:
                continue
            p, q = rng.dirichlet(np.ones(m)), rng.dirichlet(np.ones(m))
            self.assertAlmostEqual(sp.ordinal_wd(p, q, ov), ref(p, q, ov), places=12)

    def test_degenerate_predictions(self):
        # no mass on the ordered options, or NaN: scored as uniform (SubPOP's code would return NaN and drop the row)
        u = sp.ordinal_wd([.2, .3, .5], [1 / 3] * 3, [1, 2, 3])
        self.assertAlmostEqual(sp.ordinal_wd([.2, .3, .5], [0, 0, 0], [1, 2, 3]), u)
        self.assertAlmostEqual(sp.ordinal_wd([.2, .3, .5], [np.nan, .5, .5], [1, 2, 3]), u)
        self.assertAlmostEqual(sp.ordinal_wd([.2, .3, .5, 0], [0, 0, 0, 1], [1, 2, 3, -1]), u)

    def test_pairwise_matches_scalar(self):
        rng = np.random.default_rng(1)
        P = rng.dirichlet(np.ones(4), 6)
        o = [1, 3, 2, 4]
        D, T = sp.pairwise(P, o), sp.pairwise(P)
        for i in range(6):
            for j in range(6):
                self.assertAlmostEqual(D[i, j], sp.ordinal_wd(P[i], P[j], o))
                self.assertAlmostEqual(T[i, j], sp.tvd(P[i], P[j]))


class TestSubPOP(_Fixture):
    def setUp(self):
        self.task = sp.SubPOPOpinionQA()
        self.data = self.task.load(self.root)

    def test_loader(self):
        df = sp.load_subpop_opinionqa(os.path.join(self.root, "subpop-opinionqa"))
        self.assertEqual(sorted(df["item"][df.qkey == "DUP_W30"].unique()), ["DUP_W30#0", "DUP_W30#1"])
        self.assertNotIn("FLAT_W30", set(df.qkey))
        self.assertEqual(set(df.group), {"Male", "Female", "Democrat", "Republican"})
        self.assertEqual(df["options"].iloc[0][-1], "Strongly disagree")         # 'Refused' dropped
        self.assertTrue(all(abs(p.sum() - 1) < 1e-12 for p in df["dist"]))
        df61 = sp.load_subpop_opinionqa(os.path.join(self.root, "subpop-opinionqa"), "61")
        self.assertIn("18-29", set(df61.group))
        self.assertIn("18-29", {g for _, g in sp.SubPOPOpinionQA(groups="61").load(self.root).meta["groups"]})

    def test_split(self):
        tr, te = self.data.train, self.data.test
        self.assertFalse(set(tr["item"]) & set(te["item"]))
        self.assertEqual(tr["item"].nunique() + te["item"].nunique(), N_Q + 2)
        self.assertTrue(0.25 < te["item"].nunique() / (N_Q + 2) < 0.75)
        again = self.task.load(self.root)
        self.assertEqual(list(again.test["item"]), list(te["item"]))
        dv = sp.SubPOPOpinionQA(split="dev").load(self.root)
        self.assertFalse((set(dv.train["item"]) | set(dv.test["item"])) & set(te["item"]))
        self.assertEqual(set(dv.train["item"]) | set(dv.test["item"]), set(tr["item"]))

    def test_options_and_missing(self):
        for bad in ({"setting": "C"}, {"split": "holdout"}, {"test_share": 1.0}, {"groups": "7"}):
            with self.assertRaises(ValueError):
                sp.SubPOPOpinionQA(**bad).load(self.root)
        with self.assertRaises(ValueError):
            sp.SubPOPOpinionQA(setings="B")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(DataNotFound):
                sp.SubPOPOpinionQA().load(d)
        self.assertIsInstance(self.task, Task)
        self.assertEqual(self.task.primary.name, "wd")
        self.assertEqual(sp.GlobalOpinionQA().primary.name, "tvd")

    def test_setting_a(self):
        t = sp.SubPOPOpinionQA(setting="A")
        d = t.load(self.root)
        self.assertTrue(d.test["overall"].isna().all())
        self.assertEqual(set(t.baselines(d)), {"uniform"})

    def test_setting_b_baselines(self):
        te = self.data.test
        preds = self.task.baselines(self.data)
        self.assertEqual(set(preds), {"uniform", "overall", "nearest_group"})
        # default peers='blocks': Female's twin Male is in her block (SEX) and never her prediction
        for i, r in te.iterrows():
            rows = te[(te.item == r["item"]) & (te.block != r.block)]
            self.assertTrue(any(np.allclose(preds["nearest_group"][i], p) for p in rows["dist"]))
        # peers='all': nearest group of Female is Male (and vice versa); its prediction is Male's own row
        ta = sp.SubPOPOpinionQA(peers="all")
        da = ta.load(self.root)
        pa = ta.baselines(da)
        for i, r in da.test.iterrows():
            twin = {"Female": "Male", "Male": "Female"}.get(r.group)
            if twin:
                want = da.test[(da.test.item == r["item"]) & (da.test.group == twin)]["dist"].iloc[0]
                np.testing.assert_allclose(pa["nearest_group"][i], want)
        fa = ta.run(data=da)
        self.assertLess(ta.score(fa, "nearest_group")["value"], ta.score(fa, "overall")["value"])
        # overall = the file's Overall row (mean of the five synthetic groups incl. AGE)
        x = te.iloc[0]
        self.assertEqual(len(x.overall), 4)
        fr = self.task.run(data=self.data)
        sc = {m: self.task.score(fr, m)["value"] for m in preds}
        self.assertLess(sc["overall"], sc["uniform"])         # SEX's only other blocks lean +-1.2: no near twin
        self.assertEqual(self.data.meta["peers"], "blocks")
        manual = np.mean([sp.ordinal_wd(t, p, o) for t, p, o in zip(fr.truth, fr.pred_overall, fr.ordinal)])
        short = fr.assign(pred_short=pd.Series([p[:1] for p in fr.pred_uniform], dtype=object))
        for m in ("wd", "tvd", "js_sim"):                    # a 1-vector would broadcast in tvd / js
            with self.assertRaises(ValueError):
                self.task.score(short, "short", m)
        self.assertAlmostEqual(sc["overall"], manual, places=12)
        r = self.task.score(fr, "uniform")
        self.assertEqual(r["n"], te["item"].nunique())
        self.assertLessEqual(r["lo"], r["value"])
        self.assertGreaterEqual(r["hi"], r["value"])
        self.assertEqual(len(self.task.score_all(fr)), 3 * 3)

    def test_others_and_no_self_read(self):
        te = self.data.test
        x = te[te.group == "Male"].iloc[0]
        on_q = te[te.item == x["item"]]
        o = self.task.others(self.data, x["item"], x.group, "SEX")
        self.assertEqual(set(o.group), {"Democrat", "Republican"})       # Female (same block) hidden too
        oa = sp.SubPOPOpinionQA(peers="all").load(self.root)
        self.assertEqual(len(sp.OpinionDistTask.others(oa, x["item"], "Male")), len(on_q) - 1)
        da = sp.SubPOPOpinionQA(setting="A").load(self.root)
        self.assertEqual(len(sp.OpinionDistTask.others(da, x["item"], "Male")), 0)
        with self.assertRaises(ValueError):
            self.task.others(self.data, x["item"], "Nobody")
        # the leak the block rule closes: Overall row = weighted sum of a block's groups, so with peers='all' a
        # complete partition hands over the target by subtraction (here Overall is the mean of the five groups)
        o61 = sp.SubPOPOpinionQA(peers="all", groups="61").load(self.root)
        r = o61.test[(o61.test.item == x["item"])]
        self.assertEqual(len(r), 5)
        back = 5 * np.asarray(r.overall.iloc[0]) - np.sum([p for g, p in zip(r.group, r.dist) if g != "Male"], 0)
        np.testing.assert_allclose(back, x.dist, atol=1e-6)
        self.assertNotIn("SEX", set(o.block))
        # nearest_group never returns the target's own distribution
        D, N = sp.group_distances(self.data.train)
        self.assertTrue(np.allclose(np.diag(D.to_numpy()), 0))
        self.assertTrue((N.to_numpy() == N.to_numpy().T).all())

    def test_llm_method(self):
        """A method that asks an LLM per (question, group) and turns the letter into a peaked distribution."""
        def responder(system, user):
            return "D" if "Republican" in system else ("A" if "Democrat" in system else "B")
        llm = FakeLLM(responder)

        def ask(data):
            out = []
            for r in data.test.itertuples():
                letters = "ABCD"[:len(r.options)]
                txt = llm.complete(f"You answer as a {r.group} respondent.",
                                   r.question + " " + " ".join(f"({c}) {o}" for c, o in zip(letters, r.options))).text
                k = letters.index(re.search(r"[A-D]", txt).group(0))
                p = np.full(len(r.options), 0.1 / (len(r.options) - 1))
                p[k] = 0.9
                out.append(p)
            return out
        fr = self.task.run(methods={"llm": ask}, data=self.data, baselines=["uniform"])
        s = self.task.score(fr, "llm")
        self.assertTrue(np.isfinite(s["value"]))
        seg = fr.assign(v=[sp.ordinal_wd(t, p, o) for t, p, o in zip(fr.truth, fr.pred_llm, fr.ordinal)])
        self.assertEqual(set(seg.segment), {"SEX", "POLPARTY"})


class TestGlobal(_Fixture):
    def setUp(self):
        self.task = sp.GlobalOpinionQA()
        self.data = self.task.load(self.root)

    def test_loader(self):
        df = sp.load_global_opinionqa(os.path.join(self.root, "globalopinionqa"))
        q0 = df[df.question == "Global question 0?"]
        self.assertNotIn("Elbonia", set(q0.group))                  # all-zero row dropped
        self.assertEqual(df[df.question == "Global question 1?"]["options"].iloc[0][0], "0.0")
        self.assertTrue(df["item"].str.match(r"^(GAS|WVS):[0-9a-f]{12}$").all())
        self.assertEqual(df[df.question == "Global question 4?"]["source"].iloc[0], "WVS")
        items = set(self.data.train["item"]) | set(self.data.test["item"])
        self.assertNotIn(df[df.question == "Lonely question?"]["item"].iloc[0], items)
        self.assertEqual(len(items), N_Q)

    def test_overall_is_leave_block_out(self):
        te = self.data.test
        self.assertEqual(set(te.block[te.group.str.startswith("Atlantis")]), {"Atlantis"})
        for _, g in te.groupby("item"):
            P, b = np.stack(g["dist"].to_numpy()), g["block"].to_numpy()
            for i, ov in enumerate(g["overall"]):
                np.testing.assert_allclose(ov, P[b != b[i]].mean(0))
        # neither Atlantis sample is read for the other (it is a copy here), unless peers='all'
        q = te[te.group == "Atlantis (Non-national sample)"]["item"].iloc[0]
        self.assertNotIn("Atlantis", set(self.task.others(self.data, q, "Atlantis (Non-national sample)").block))
        preds = self.task.baselines(self.data)
        for i, r in te.iterrows():
            if r.block == "Atlantis":
                self.assertGreater(sp.tvd(preds["nearest_group"][i], r.dist), 1e-6)
        ta = sp.GlobalOpinionQA(peers="all")
        da = ta.load(self.root)
        pa = ta.baselines(da)
        i = int(np.flatnonzero((da.test.group == "Atlantis (Non-national sample)").to_numpy())[0])
        self.assertLess(sp.tvd(pa["nearest_group"][i], da.test.dist[i]), 1e-9)

    def test_baselines(self):
        fr = self.task.run(data=self.data)
        sc = {m: self.task.score(fr, m)["value"] for m in ("uniform", "overall", "nearest_group")}
        self.assertLess(sc["nearest_group"], sc["overall"])
        self.assertLess(sc["overall"], sc["uniform"])
        js = self.task.score(fr, "nearest_group", "js_sim")
        self.assertTrue(0 < js["value"] < 1)
        D, _ = sp.group_distances(self.data.train, ordinal=False)
        self.assertEqual(D.loc["country:Borduria"].drop(["country:Borduria", "country:Atlantis (Non-national sample)"])
                         .idxmin(), "country:Atlantis")


@unittest.skipUnless(os.environ.get("SIMENGINE_DATA"), "$SIMENGINE_DATA not set")
class TestRealData(unittest.TestCase):
    def test_uniform_close_to_subpop_upper_bound(self):
        root = os.environ["SIMENGINE_DATA"]
        try:
            df = sp.load_subpop_opinionqa(os.path.join(root, "subpop-opinionqa"))
        except DataNotFound:
            self.skipTest("subpop-opinionqa not under $SIMENGINE_DATA")
        v = [sp.ordinal_wd(p, np.ones(len(p)) / len(p), o) for p, o in zip(df.dist, df.ordinal)]
        self.assertAlmostEqual(float(np.mean(v)), 0.178, delta=0.005)   # SubPOP Table 1, uniform, OpinionQA


if __name__ == "__main__":
    unittest.main()
