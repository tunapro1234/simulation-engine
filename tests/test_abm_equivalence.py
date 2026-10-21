"""Port equivalence against the original research code (optional).

Set SIMENGINE_ABM_SOURCE to the research branch directory that holds `abm/kod/motor.py` and
`kod/kit/yontemler/abm_benimseme.py` (the TALP prep `makro` branch). Skipped when unset. The research code is only
imported, never modified: bytecode writing is disabled and its disk cache is redirected to a temporary directory.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import simengine.abm as A

SRC = os.environ.get("SIMENGINE_ABM_SOURCE")


@unittest.skipUnless(SRC and Path(SRC, "abm", "kod", "motor.py").exists(), "set SIMENGINE_ABM_SOURCE to compare")
class TestEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True
        cls.tmp = tempfile.TemporaryDirectory()
        os.environ["ABM_ONBELLEK"] = cls.tmp.name
        for p in (Path(SRC, "abm", "kod"), Path(SRC, "kod", "kit"), Path(SRC, "kod", "kit", "yontemler")):
            sys.path.insert(0, str(p))
        import abm_benimseme
        import baseline
        import motor
        cls.motor, cls.baseline, cls.O = motor, baseline, abm_benimseme

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_engines(self):
        rng = np.random.default_rng(3)
        th = A.sample_adoption_prior(10, rng)
        np.testing.assert_array_equal(A.adoption_sim(th, 15, n_agents=150, seed=5),
                                      self.motor.benimseme_sim(th, 15, n_ajan=150, tohum=5))
        tm = A.sample_market_prior(6, rng)
        lp = np.r_[np.zeros(26), -0.2 * np.ones(8)]
        pr = np.r_[np.zeros(26), np.ones(8)]
        for cohort in (False, True):
            np.testing.assert_array_equal(A.market_sim(tm, lp, pr, seed=2, cohort=cohort),
                                          self.motor.pazar_sim(tm, lp, pr, tohum=2, kohort=cohort))

    def test_curves(self):
        t = np.arange(1995, 2005.0)
        y = np.array([0.5, 1, 2, 4, 7, 11, 16, 22, 28, 33.0])
        h = np.arange(2005, 2015.0)
        for a, b in [("naive", "naive"), ("linear", "dogrusal"), ("damped", "sonumlu"), ("logistic", "lojistik"),
                     ("gompertz", "gompertz"), ("bass", "bass")]:
            np.testing.assert_allclose(A.CURVES[a](t, y, h), self.baseline.YONTEMLER[b](t, y, h))

    def test_abc_methods(self):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(8):
            t = np.arange(1990, 2021.0)
            yy = A.bass_curve(t, 60 + 5 * i, 0.01 * (1 + i % 3), 0.3 + 0.05 * i, 1990 + i) + rng.normal(0, 0.5, len(t))
            rows += [dict(seri=f"s{i}", t=a, y=max(b, 0)) for a, b in zip(t, yy)]
        d = pd.DataFrame(rows)
        meta = {s: dict(t0=int(g[g.y >= 1].t.iloc[0])) for s, g in d.groupby("seri")}
        s, c = "s3", meta["s3"]["t0"] + 5
        eg = d[d.t <= c]
        pairs = [(self.O.AbmAbc, A.AdoptionABC), (self.O.AbmAbcAnaloji, A.AdoptionABCAnalogy),
                 (self.O.BassAbc, A.BassABC), (self.O.BassAbcAnaloji, A.BassABCAnalogy)]
        for ocls, ncls in pairs:
            o = ocls(n_sim=300, n_ajan=300)
            a = o.tahmin(o.uydur(eg, dict(hedef_seri=s, kesim=c, meta=meta)),
                         pd.DataFrame(dict(seri=s, kesim=c, ufuk=np.arange(1, 11))))
            n = ncls(n_sim=300, n_agents=300)
            b = n.predict(n.fit(eg.rename(columns={"seri": "series"}), s, c), np.arange(1, 11))
            np.testing.assert_allclose(a.y_tahmin.to_numpy(), b.y_pred.to_numpy(), err_msg=ncls.__name__)
            np.testing.assert_allclose(a.ust80.to_numpy(), b.hi80.to_numpy(), err_msg=ncls.__name__)


if __name__ == "__main__":
    unittest.main()
