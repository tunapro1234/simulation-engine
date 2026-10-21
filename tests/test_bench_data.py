"""Data-present checks: reproduce published development numbers on the real catalogue when it is on this machine.

Skipped when the catalogue is absent. Root: $SIMENGINE_TEST_DATA, else $SIMENGINE_DATA,
else the research machine's catalogue (the ROOT default below).
Each expected number comes from the research branch file named in the assertion message. Budget < 30 s.
"""
import os
import unittest
import warnings

from simengine import bench

ROOT = os.environ.get("SIMENGINE_TEST_DATA") or os.environ.get("SIMENGINE_DATA") or "/data/talp/veri"


def _has(*parts):
    return os.path.exists(os.path.join(ROOT, *parts))


@unittest.skipUnless(_has("gss", "panel2016_2020.parquet"), "GSS panels not present")
class TestGSSReal(unittest.TestCase):
    def test_persistence_dev_matches_dongu(self):
        warnings.simplefilter("ignore")
        r = bench.tasks.gss.cross_panel_dev(ROOT, {"yontem": "kendi"}, items="climb")
        # uretim/dongu/turlar/gss_dalga/tabanlar.json, T-kendi, climb set: z 0.4754, n 101
        self.assertEqual(r["mean_z"]["n"], 101)
        self.assertAlmostEqual(r["mean_z"]["value"], 0.4754, places=4)
        self.assertAlmostEqual(r["corr"]["value"], 0.4425, places=4)


@unittest.skipUnless(_has("opinionqa", "islenmis", "dagilim.parquet")
                     and _has("subpop-opinionqa", "data", "opinionqa", "processed", "opinionqa.csv"),
                     "OpinionQA / SubPOP files not present")
class TestOpinionQAReal(unittest.TestCase):
    def test_uniform_dev_matches_dongu(self):
        task = bench.get("opinionqa_groups", split="dev", items="climb")
        data = task.load(ROOT)
        fr = task.frame(data, {"uniform": task.predict_with(data, "uniform")})
        r = task.score(fr, "uniform")
        # uretim/dongu/turlar/oqa_grup_A/tabanlar.json, T-tekduze, climb set: WD 0.1976 [0.1908, 0.2044], n 717
        self.assertEqual(r["n"], 717)
        self.assertAlmostEqual(r["value"], 0.19756, places=4)
        self.assertAlmostEqual(r["lo"], 0.19085, places=4)


@unittest.skipUnless(_has("bank-marketing", "bank-additional", "bank-additional-full.csv"), "bank-marketing absent")
class TestBankReal(unittest.TestCase):
    def test_logistic_matches_league(self):
        warnings.simplefilter("ignore")
        task = bench.get("bank_marketing", lgbm=False)
        data = task.load(ROOT)
        self.assertEqual(len(data.test), 8238)
        fr = task.run(data=data)
        # uretim/intent/gbdt/lig/bank_lig.json (bank.py kos): lojistik AUC 0.7483, kalicilik 0.6408
        self.assertAlmostEqual(task.score(fr, "logistic", B=100)["value"], 0.7483, places=4)
        self.assertAlmostEqual(task.score(fr, "persistence", B=100)["value"], 0.6408, places=4)


if __name__ == "__main__":
    unittest.main()
