"""Next-action ranking: Markov-2 baseline and LightGBM lambdarank on synthetic sessions."""
import unittest

import numpy as np
import pandas as pd

from simengine.methods import (Markov2Baseline, NextActionRanker, add_history, build_features, hit_at_k,
                               lightgbm_available, negative_sample, top1)

KEYS = ["search", "view", "cart", "pay", "back", "end"]
NEXT = {("START", "START"): "search", "search": "view", "view": "cart", "cart": "pay", "pay": "end",
        "back": "search", "end": "end"}


def sessions(n_sess=120, seed=0, offset=0):
    rs = np.random.default_rng(seed)
    steps, cands, ex = [], [], offset
    for s in range(n_sess):
        prev2, prev = "START", "START"
        for t in range(6):
            if (prev2, prev) == ("START", "START"):
                a = "search"
            elif prev == "view" and prev2 == "back":
                a = "back"                                  # second-order rule
            else:
                a = NEXT[prev] if rs.random() < 0.8 else str(rs.choice(KEYS))
            steps.append({"ex": ex, "session": f"s{offset}_{s}", "step": t, "action": a, "user": f"u{s % 30}",
                          "page": "checkout" if prev in ("cart", "pay") else "shop"})
            for k in KEYS:
                cands.append({"ex": ex, "key": k, "is_end": int(k == "end")})
            prev2, prev, ex = prev, a, ex + 1
    return pd.DataFrame(steps), pd.DataFrame(cands)


class TestNextAction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr_s, cls.tr_c = sessions(150, 0, 0)
        cls.te_s, cls.te_c = sessions(40, 1, 100000)

    def test_history_and_oof(self):
        h = add_history(self.tr_s.rename(columns={"session": "session_id", "step": "adim"}))
        self.assertEqual(h.groupby("session").head(1)["prev_key"].unique().tolist(), ["START"])
        tr, te = build_features(self.tr_s, self.tr_c, self.te_s, self.te_c)
        self.assertEqual(len(tr), len(self.tr_c))
        self.assertEqual(len(te), len(self.te_c))
        self.assertEqual(int(tr.groupby("ex")["y"].sum().max()), 1)

    def test_markov2(self):
        te, s = Markov2Baseline().fit(self.tr_s, self.tr_c).score(self.te_s, self.te_c)
        h = hit_at_k(te, s, ks=(1, 3))
        self.assertGreater(h["hit@1"], 0.7)
        self.assertGreaterEqual(h["hit@3"], h["hit@1"])
        t = top1(te, s)
        self.assertEqual(len(t), self.te_s.ex.nunique())

    @unittest.skipUnless(lightgbm_available(), "lightgbm not installed")
    def test_ranker(self):
        r = NextActionRanker(n_estimators=60, top=3, random=2).fit(self.tr_s, self.tr_c, numeric=["is_end"],
                                                                    categorical=["page"])
        te, s = r.score(self.te_s, self.te_c)
        te_m, s_m = Markov2Baseline().fit(self.tr_s, self.tr_c).score(self.te_s, self.te_c)
        self.assertGreater(hit_at_k(te, s)["hit@1"], hit_at_k(te_m, s_m)["hit@1"] - 0.05)
        r3 = NextActionRanker(n_estimators=20, top=3, random=2, n_seeds=2).fit(self.tr_s, self.tr_c, numeric=["is_end"])
        self.assertEqual(len(r3.models_), 2)
        self.assertEqual(len(r3.score(self.te_s, self.te_c)[1]), len(te))

    def test_negative_sample_keeps_non_click(self):
        tr = pd.DataFrame({"ex": [0] * 6, "y": [1, 0, 0, 0, 0, 0], "f_page": [.9, .8, .1, .1, .1, .1],
                           "f_markov": 0.0, "f_user": 0.0,
                           "kind": ["click", "click", "click", "click", "terminate", "input"]})
        out = negative_sample(tr, top=1, random=0)
        self.assertEqual(sorted(out.index.tolist()), [0, 4, 5])
        self.assertEqual(sorted(negative_sample(tr, top=1, random=0, keep_non_click=False).index.tolist()), [0])
        self.assertEqual(len(negative_sample(tr.drop(columns="kind"), top=1, random=0)), 1)


if __name__ == "__main__":
    unittest.main()
