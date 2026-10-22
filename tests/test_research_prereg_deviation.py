"""Dated addenda chained to a locked prereg, sealed deviation arms and side-by-side arm grading (offline)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from simengine.research import (Prereg, add_addendum, grade_arms, lock_prereg, read_addendum, seal, seal_deviation,
                                verify_addenda, verify_chain, verify_prereg, verify_seal)


def _parent(d) -> Path:
    p = Prereg(title="wave forecast", question="Does C5 beat the strongest base?", falsification="C5 - T4 CI has 0",
               stop_rule="<= 6 configs").write(Path(d) / "ONKAYIT.md")
    lock_prereg(p)
    return p


class TestAddendum(unittest.TestCase):
    def test_addendum_chains_and_keeps_parent_lock(self):
        with tempfile.TemporaryDirectory() as d:
            p = _parent(d)
            lock_before = (Path(d) / "ONKAYIT.md.lock.json").read_text()
            r1 = add_addendum(p, reason="referee found a stronger base", deviation="ours = H_joint",
                              falsification="H_joint - T4b CI contains 0", changes=["seal C5 separately", "add T4b"])
            self.assertTrue(r1["path"].endswith("ONKAYIT-EK1.md"))
            self.assertEqual((Path(d) / "ONKAYIT.md.lock.json").read_text(), lock_before)   # parent lock untouched
            self.assertTrue(verify_prereg(p)["ok"])
            self.assertEqual(r1["parent_sha256"], verify_prereg(p)["sha256"])
            doc = read_addendum(r1["path"])
            self.assertEqual(doc["parent_sha256"], r1["parent_sha256"])
            self.assertEqual(doc["changes"], ["seal C5 separately", "add T4b"])
            text = Path(r1["path"]).read_text()
            self.assertIn("## Deviation", text)
            self.assertIn(r1["parent_sha256"], text)
            r2 = add_addendum(p, reason="related version", deviation="HB counts as the same arm")
            self.assertTrue(r2["path"].endswith("ONKAYIT-EK2.md"))
            self.assertEqual(r2["prev_addendum"], "ONKAYIT-EK1.md")
            self.assertEqual(r2["prev_addendum_sha256"], r1["sha256"])
            self.assertIn("Unchanged from the parent", Path(r2["path"]).read_text())
            v = verify_addenda(p)
            self.assertTrue(v["ok"], v)
            self.assertEqual([a["index"] for a in v["addenda"]], [1, 2])
            # the parent's old lock is overwritten -> every addendum notices
            p.write_text(p.read_text() + "\nedited\n")
            lock_prereg(p, overwrite=True)
            v = verify_addenda(p)
            self.assertFalse(v["ok"])
            self.assertFalse(any(a["parent_ok"] for a in v["addenda"]))

    def test_edited_addendum_and_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ONKAYIT.md"
            Prereg(title="t", question="q").write(p)
            with self.assertRaises(ValueError):                      # parent not locked
                add_addendum(p, reason="r", deviation="x")
            lock_prereg(p)
            with self.assertRaises(ValueError):                      # reason and deviation are required
                add_addendum(p, reason="", deviation="x")
            r1 = add_addendum(p, reason="r", deviation="x")
            with self.assertRaises(ValueError):                      # never rewritten
                add_addendum(p, reason="r", deviation="y", path=r1["path"])
            Path(r1["path"]).write_text(Path(r1["path"]).read_text() + "late edit\n")
            self.assertFalse(verify_addenda(p)["ok"])
            with self.assertRaises(ValueError):                      # previous addendum no longer verifies
                add_addendum(p, reason="r2", deviation="z")
            # an edited parent refuses new addenda
            q = Path(d) / "OTHER.md"
            Prereg(title="t2", question="q2").write(q)
            lock_prereg(q)
            q.write_text(q.read_text() + "x")
            with self.assertRaises(ValueError):
                add_addendum(q, reason="r", deviation="x")

    def test_hand_written_addendum_is_not_chained(self):
        with tempfile.TemporaryDirectory() as d:
            p = _parent(d)
            ek = Path(d) / "ONKAYIT-EK1.md"
            ek.write_text("# EK-1\nmanual\n")
            lock_prereg(ek)
            v = verify_addenda(p)
            self.assertFalse(v["ok"])
            self.assertIn("not chained", v["addenda"][0]["reason"])
            # lock_prereg(parent=) gives the same link add_addendum writes
            ek2 = Path(d) / "ONKAYIT-EK2.md"
            ek2.write_text("# EK-2\nmanual\n")
            rec = lock_prereg(ek2, parent=p)
            self.assertEqual(rec["addendum_index"], 2)
            self.assertEqual(rec["prev_addendum"], "ONKAYIT-EK1.md")
            self.assertTrue(verify_addenda(p)["addenda"][1]["link_ok"])


class TestDeviationArms(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.truth = pd.DataFrame({"seg": np.repeat(np.arange(30), 2), "item": ["a", "b"] * 30,
                                   "y": rng.normal(size=60)})
        noise = rng.normal(size=60)
        self.good = self.truth.assign(y=self.truth["y"] + 0.1 * noise)
        self.bad = self.truth.assign(y=self.truth["y"] + 1.0 * noise)

    def test_seal_and_grade_both_directions(self):
        for pre, post, sign in ((self.good, self.bad, 1), (self.bad, self.good, -1)):   # D6 lost, D7 won
            with tempfile.TemporaryDirectory() as d:
                p = _parent(d)
                add_addendum(p, reason="post-hoc choice", deviation="submit the logit arm")
                reg = Path(d) / ".prereg"
                out = seal_deviation(pre, post, "test forecast", reg, reason="plausibility check",
                                     addendum=Path(d) / "ONKAYIT-EK1.md", pre_name="pool", post_name="pool_logit")
                self.assertTrue(out["chain"]["valid"])
                self.assertLess(out["pre"]["seq"], out["post"]["seq"])
                self.assertEqual(out["pre"]["role"], "preregistered")
                self.assertEqual(out["post"]["deviation_of"], "pool")
                self.assertEqual(out["post"]["deviation_of_record"], out["pre"]["record_hash"])
                self.assertEqual(out["post"]["addendum"], "ONKAYIT-EK1.md")
                again = seal_deviation(pre, post, "test forecast", reg, reason="plausibility check",
                                       addendum=Path(d) / "ONKAYIT-EK1.md", pre_name="pool", post_name="pool_logit")
                self.assertEqual(again["post"]["record_hash"], out["post"]["record_hash"])   # idempotent
                self.assertEqual(verify_chain(reg)["n"], 2)
                # truth keys as strings: the archived arms still join
                truth = self.truth.assign(seg=self.truth["seg"].astype(str))
                tab = grade_arms(reg, truth, ["seg", "item"])
                self.assertEqual(list(tab["arm"]), ["pool", "pool_logit"])
                self.assertEqual(list(tab["n_joined"]), [60, 60])
                self.assertTrue(np.isnan(tab.loc[0, "diff_vs_pre"]))
                row = tab.loc[1]
                self.assertAlmostEqual(row["diff_vs_pre"], row["mae"] - tab.loc[0, "mae"])
                self.assertEqual(np.sign(row["diff_vs_pre"]), sign)
                self.assertTrue(row["diff_lo"] > 0 if sign > 0 else row["diff_hi"] < 0)
                # same numbers from in-memory tables
                tab2 = grade_arms(reg, self.truth, ["seg", "item"], column="y",
                                  predictions={"pool": pre, "pool_logit": post})
                self.assertAlmostEqual(tab2.loc[1, "mae"], row["mae"])
                with self.assertRaises(ValueError):
                    grade_arms(reg, self.truth, ["seg", "item"], predictions={"pool": pre.assign(y=0.0)})

    def test_record_fields_count_in_hash_and_old_seal_compatible(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Path(d) / "reg"
            r0 = seal(self.good, "plain", reg)
            self.assertNotIn("arm", r0)                                    # old record shape
            self.assertEqual(seal(self.good, "again", reg)["seq"], 0)
            r1 = seal(self.good, "plain", reg, arm="C5", role="preregistered")
            self.assertEqual(r1["seq"], 1)                                 # same table, now named
            self.assertTrue(verify_seal(self.good, registry_dir=reg)["matched"])
            self.assertTrue(verify_chain(reg)["valid"])
            lines = (reg / "chain.jsonl").read_text().splitlines()
            rec = json.loads(lines[1])
            rec["role"] = "post_deviation"
            lines[1] = json.dumps(rec)
            (reg / "chain.jsonl").write_text("\n".join(lines) + "\n")
            self.assertEqual(verify_chain(reg)["broken"], [1])
            with self.assertRaises(ValueError):
                seal(self.good, "x", reg, meta={"sha256": "0"})

    def test_refusals_and_custom_scorer(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Path(d) / "reg"
            with self.assertRaises(ValueError):
                seal_deviation(self.good, self.good.copy(), "same", reg, reason="r")
            with self.assertRaises(ValueError):
                seal_deviation(self.good, self.bad, "x", reg, reason=" ")
            with self.assertRaises(ValueError):
                grade_arms(reg, self.truth, ["seg", "item"])               # no arms yet
            f = Path(d) / "kol_C5.csv"
            self.good.to_csv(f, index=False)
            out = seal_deviation(f, self.bad, "x", reg, reason="r")
            self.assertEqual(out["pre"]["arm"], "kol_C5")
            self.assertEqual(out["post"]["arm"], "post_deviation")
            tab = grade_arms(reg, self.truth, ["seg", "item"],
                             scorer=lambda t: {"r": float(np.corrcoef(t["y"], t["y_truth"])[0, 1])})
            self.assertIn("r", tab.columns)
            self.assertNotIn("diff_vs_pre", tab.columns)                  # no per-unit loss -> no paired diff
            sq = grade_arms(reg, self.truth, ["seg", "item"], unit_loss=lambda p, t: (p - t) ** 2, arms=["kol_C5"])
            self.assertEqual(list(sq["arm"]), ["kol_C5"])
            self.assertIn("loss", sq.columns)
            with self.assertRaises(KeyError):
                grade_arms(reg, self.truth, ["seg", "item"], arms=["nope"])


if __name__ == "__main__":
    unittest.main()
