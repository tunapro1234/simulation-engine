"""D13-W8 #4 / W10 #4: the badge row printed "+0.002, CI [-0.003; +0.004]" while the summary and the recorded
comparison said "+0.002 [-0.002; +0.004]". With tied strongest baselines the badge is against the worst of them
(min over the tied set), the recorded comparison is ours − the single strongest baseline. Every CI the badge row
prints must either be the recorded one or be labelled with what it measures. Synthetic, offline."""
import re
import unittest

import warnings

from simengine import evidence as ev
from simengine.evidence import api, kart
from tests.test_evidence_baseline_uncertain import _audit, _eval, _frame

CI_RE = re.compile(r"([+-]\d+\.\d{3})(?:,? (?:GA|CI))? \[([+-]\d+\.\d{3})[;,] ([+-]\d+\.\d{3})\]")


def _cis(text):
    return {m.groups() for m in CI_RE.finditer(text)}


def _eval_up(df):
    """Same frame, higher-is-better primary metric (R²)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ev.evaluate(df, "t", ours="ours", task="continuous", primary="r2", B=300, multiplier=False,
                           unlabeled_baselines=[c[5:] for c in df if c.startswith("pred_B")])


def _tied_rep(alt=-0.01, ev_=_eval):
    rep = ev_(_frame(sds=(1.0, 1.0, 1.0, 1.0), ours_sd=0.6))
    a = _audit(rep)
    assert a and a["durum"] == "uyari", a
    a["en_kotu"]["alt"] = alt          # worst tied lower bound differs from the single strongest one
    return rep


class TestBadgeCi(unittest.TestCase):
    def test_badge_ci_is_recorded_or_labelled(self):
        rep = _tied_rep()
        k = kart.kart(rep)
        f, g = k["fark"], k["rozet_gerekce"]
        recorded = (f"{f['fark']:+.3f}", f"{f['alt']:+.3f}", f"{f['ust']:+.3f}")
        cis = _cis(g)
        self.assertGreaterEqual(len(cis), 2, g)            # the worst-tied CI and the recorded one
        self.assertIn(recorded, cis, g)
        biz, tb = rep["manset"]["bizim"]["etiket"], rep["manset"]["en_iyi_baseline"]["etiket"]
        # the recorded CI is named as the strongest baseline − ours: MAE is lower-is-better, so the card's
        # direction-aligned difference (+ = ours better) is baseline − ours, not ours − baseline
        self.assertIn(f"{tb} − {biz} = {recorded[0]} [{recorded[1]}; {recorded[2]}]", g)
        self.assertNotIn(f"{biz} − {tb} =", g)
        # every other CI is named as the minimum over the tied baselines
        for c in cis - {recorded}:
            self.assertRegex(g, r"min\([^)]*\) = " + re.escape(f"{c[0]} [{c[1]}; {c[2]}]"))

    def test_english_badge_keeps_both_labels(self):
        rep = _tied_rep()
        k = kart.kart(rep)
        en = api._en_text(k["rozet_gerekce"])
        self.assertIsNone(api._TR_LETTERS.search(en), en)
        self.assertIn("worst tied baseline", en)
        self.assertIn("min(", en)
        f = k["fark"]
        self.assertIn(f"[{f['alt']:+.3f}; {f['ust']:+.3f}]", en)

    def test_label_matches_paired_diff_both_directions(self):
        # the labelled "X − Y = d [lo; hi]" must be what paired_diff(report, X, Y) returns for the recorded comparison,
        # for a lower-is-better (mae) and a higher-is-better (r2) primary metric alike
        for ev_ in (_eval, _eval_up):
            rep = _tied_rep(ev_=ev_)
            g = kart.kart(rep)["rozet_gerekce"]
            biz, tb = rep["manset"]["bizim"], rep["manset"]["en_iyi_baseline"]
            m = re.search(r"; (\S+) − (\S+) = ([+-]\d+\.\d{3}) \[([+-]\d+\.\d{3}); ([+-]\d+\.\d{3})\]", g)
            self.assertIsNotNone(m, g)
            cols = {biz["etiket"]: biz["sutun"], tb["etiket"]: tb["sutun"]}
            self.assertEqual(set(m.group(1, 2)), set(cols), g)
            x, y = (cols[m.group(i)].removeprefix("tahmin_") for i in (1, 2))
            pd_ = api.paired_diff(rep, x, y)
            self.assertEqual(m.group(3, 4, 5), (f"{pd_['diff']:+.3f}", f"{pd_['lo']:+.3f}", f"{pd_['hi']:+.3f}"),
                             (ev_.__name__, g, pd_))

    def test_single_strongest_baseline_unchanged(self):
        rep = _eval(_frame(sds=(0.3, 2.0, 2.0, 2.0), ours_sd=0.2))
        k = kart.kart(rep)
        self.assertNotIn("min(", k["rozet_gerekce"])
        f = k["fark"]
        for c in _cis(k["rozet_gerekce"]):
            self.assertEqual(c, (f"{f['fark']:+.3f}", f"{f['alt']:+.3f}", f"{f['ust']:+.3f}"))


if __name__ == "__main__":
    unittest.main()
