"""research additions (PLAN 2.8): declared targets hashed into the prereg lock, harness one-change variants and sweep.

Offline: no LLM, no network. The target values below are the two PAPER_FINDINGS of the replicant Ertan, Page &
Putterman (2009) config (`papers/ertan2009/config.py`), used only as example input.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from simengine.research import (HarnessSpace, Loop, Prereg, Space, compare_targets, declare_targets, harness_sweep,
                                harness_variants, lock_prereg, one_change_variants, targets_sha256, variant_label,
                                verify_prereg, write_prereg)
from simengine.research.proposers import GridProposer, ProposalContext, validate_proposal

ERTAN_FINDINGS = {
    "vote_punish_low_pct": (0.85, "% voting to allow punishing low contributors (late rounds)"),
    "vote_punish_high_pct": (0.00, "% voting to allow punishing high contributors"),
}


def _prereg(**kw):
    return Prereg(title="ertan vignette", question="Do agents vote like the paper's groups?", data="one-shot vignette",
                  split="dev = 2 conditions, test = held-out condition", metrics=["abs error (lower is better)"],
                  baselines={"no persona": "persona_method=baseline"}, expected="punish-low near 0.85",
                  falsification="punish-low below 0.5", **kw)


class TestDeclareTargets(unittest.TestCase):
    def test_paper_findings_form_is_normalised_and_sorted(self):
        t = declare_targets(ERTAN_FINDINGS)
        self.assertEqual(list(t), ["vote_punish_high_pct", "vote_punish_low_pct"])
        self.assertEqual(t["vote_punish_low_pct"], {"value": 0.85, "description": ERTAN_FINDINGS[
            "vote_punish_low_pct"][1]})

    def test_other_forms(self):
        t = declare_targets({"a": 1, "b": {"value": 2.0, "tolerance": 0.5, "source": "Table 2"},
                             "c": (0.3, "share", {"interval": [0.2, 0.4]}), "d": {"interval": (1, 3)},
                             "e": "cooperate", "f": np.float64(0.25)})
        self.assertEqual(t["a"], {"value": 1})
        self.assertEqual(t["b"], {"value": 2.0, "tolerance": 0.5, "source": "Table 2"})
        self.assertEqual(t["c"], {"value": 0.3, "description": "share", "interval": [0.2, 0.4]})
        self.assertEqual(t["d"], {"interval": [1, 3]})
        self.assertEqual(t["e"], {"value": "cooperate"})
        self.assertIs(type(t["f"]["value"]), float)
        lst = declare_targets([{"metric": "x", "value": 1}, {"name": "y", "value": 2, "unit": "%"}])
        self.assertEqual(lst, {"x": {"value": 1}, "y": {"value": 2, "unit": "%"}})
        self.assertEqual(declare_targets(t), t)          # idempotent

    def test_invalid_targets_are_refused(self):
        bad = [{"a": float("nan")}, {"a": {"value": 1, "tolerence": 0.1}}, {"a": {"interval": [3, 1]}},
               {"a": {"value": "x", "tolerance": 0.1}}, {"a": {"value": 1, "tolerance": -1}}, {"a": {}},
               {"a": (1, "d", "not a dict")}, {"": 1}, {"a": [1, 2, 3, 4]}]
        for b in bad:
            with self.assertRaises(ValueError, msg=str(b)):
                declare_targets(b)
        with self.assertRaises(ValueError):
            declare_targets([{"metric": "x", "value": 1}, {"metric": "x", "value": 2}])
        with self.assertRaises(TypeError):
            declare_targets(42)

    def test_targets_hash_ignores_order_but_not_values(self):
        a = targets_sha256(ERTAN_FINDINGS)
        self.assertEqual(a, targets_sha256(dict(reversed(list(ERTAN_FINDINGS.items())))))
        self.assertEqual(a, targets_sha256({**ERTAN_FINDINGS, "vote_punish_low_pct": (0.85 + 1e-13, ERTAN_FINDINGS[
            "vote_punish_low_pct"][1])}))
        self.assertNotEqual(a, targets_sha256({**ERTAN_FINDINGS, "vote_punish_low_pct": (0.80, "x")}))


class TestTargetsInLock(unittest.TestCase):
    def test_declared_targets_change_the_lock_hash(self):
        with tempfile.TemporaryDirectory() as d:
            plain = write_prereg(_prereg(), Path(d) / "plain.md")
            p = _prereg()
            declare_targets(ERTAN_FINDINGS, p)
            witht = p.write(Path(d) / "with.md")
            other = _prereg()
            declare_targets({**ERTAN_FINDINGS, "vote_punish_low_pct": (0.80, "changed")}, other)
            other_p = other.write(Path(d) / "other.md")
            r0, r1, r2 = lock_prereg(plain), lock_prereg(witht), lock_prereg(other_p)
            self.assertEqual(len({r0["sha256"], r1["sha256"], r2["sha256"]}), 3)
            self.assertNotIn("targets_sha256", r0)
            self.assertEqual(r1["targets_sha256"], targets_sha256(ERTAN_FINDINGS))
            self.assertEqual(r1["n_targets"], 2)
            self.assertNotEqual(r1["targets_sha256"], r2["targets_sha256"])
            v = verify_prereg(witht)
            self.assertTrue(v["ok"])
            self.assertEqual(v["targets_sha256"], r1["targets_sha256"])
            md = witht.read_text()
            self.assertIn("## Declared targets", md)
            self.assertIn("| vote_punish_low_pct | 0.85 |", md)

    def test_json_roundtrip_and_no_targets_keeps_old_layout(self):
        with tempfile.TemporaryDirectory() as d:
            p = _prereg(targets=declare_targets(ERTAN_FINDINGS))
            for name in ("p.json", "p.md"):
                self.assertEqual(Prereg.read(p.write(Path(d) / name)).targets, p.targets)
            self.assertNotIn("targets", _prereg().to_dict())
            self.assertNotIn("targets", json.loads((write_prereg(_prereg(), Path(d) / "q.json")).read_text()))
            self.assertEqual(Prereg.read(Path(d) / "q.json").targets, {})

    def test_declare_on_path_before_lock_and_refuse_after(self):
        with tempfile.TemporaryDirectory() as d:
            f = write_prereg(_prereg(), Path(d) / "p.md")
            declare_targets(ERTAN_FINDINGS, f)
            self.assertEqual(Prereg.read(f).targets, declare_targets(ERTAN_FINDINGS))
            rec = lock_prereg(f)
            with self.assertRaises(ValueError):
                declare_targets({"vote_punish_low_pct": 0.5}, f)
            self.assertTrue(verify_prereg(f)["ok"])          # refused call did not touch the file
            declare_targets({"vote_punish_low_pct": 0.5}, f, overwrite=True)
            self.assertFalse(verify_prereg(f)["ok"])
            rec2 = lock_prereg(f, overwrite=True)
            self.assertNotEqual(rec2["targets_sha256"], rec["targets_sha256"])

    def test_lock_of_a_free_form_file_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "notes.md"
            f.write_text("# my own prereg\nno json block here\n")
            rec = lock_prereg(f)
            self.assertNotIn("targets_sha256", rec)
            self.assertTrue(verify_prereg(f)["ok"])


class TestCompareTargets(unittest.TestCase):
    def test_statuses(self):
        t = declare_targets({"low": {"value": 0.85, "tolerance": 0.1}, "high": {"value": 0.0, "interval": [0, 0.05]},
                             "plain": 2.0, "label": "cooperate", "gone": 1.0, "nan": 1.0})
        obs = {"low": 0.80, "high": 0.2, "plain": 2.5, "label": "cooperate", "nan": float("nan"), "posthoc": 3}
        df = compare_targets(t, obs).set_index("metric")
        self.assertEqual(df.loc["low", "status"], "pass")
        self.assertAlmostEqual(df.loc["low", "diff"], -0.05)
        self.assertEqual(df.loc["high", "status"], "fail")
        self.assertEqual(df.loc["plain", "status"], "reported")
        self.assertEqual(df.loc["label", "status"], "pass")
        self.assertEqual(df.loc["gone", "status"], "missing")
        self.assertEqual(df.loc["nan", "status"], "missing")
        self.assertIsNone(df.loc["nan", "observed"])          # never filled in
        self.assertEqual(df.loc["posthoc", "status"], "undeclared")

    def test_reads_targets_from_a_prereg_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = _prereg(targets=declare_targets(ERTAN_FINDINGS)).write(Path(d) / "p.md")
            df = compare_targets(f, {"vote_punish_low_pct": 0.94, "vote_punish_high_pct": 0.0})
            self.assertEqual(sorted(df["status"]), ["reported", "reported"])
            self.assertAlmostEqual(df.set_index("metric").loc["vote_punish_low_pct", "diff"], 0.09)


# --------------------------------------------------------------------------- harness variants

def _ctx(space, cfg):
    return ProposalContext(turn=1, incumbent={"n": 0, "cfg": cfg, "metrics": {}}, start=cfg, history=[],
                           space=space, seen=set(), primary="err", higher_is_better=False)


class TestHarnessVariants(unittest.TestCase):
    def setUp(self):
        self.space = harness_variants({"w1": "Would you allow punishing low contributors?",
                                       "w2": "Should low contributors be punishable?"},
                                      [None, 8192], [0.0, 0.7, 1.0])

    def test_space_shape_and_defaults(self):
        s = self.space
        self.assertIsInstance(s, HarnessSpace)
        self.assertIsInstance(s, Space)
        self.assertEqual(list(s.params), ["wording_id", "max_tokens", "temperature"])
        self.assertEqual(s.defaults, {"wording_id": "w1", "max_tokens": None, "temperature": 0.0})
        self.assertEqual(s.wording({"wording_id": "w2"}), "Should low contributors be punishable?")
        self.assertEqual(s.validate({"max_tokens": 8192})[1], [])
        self.assertTrue(s.validate({"max_tokens": 100})[1])

    def test_variants_are_one_change_per_turn(self):
        vs = self.space.variants()
        self.assertEqual(len(vs), 1 + 1 + 1 + 2)
        base = vs[0]
        self.assertEqual((base["label"], base["change"]), ("default", {}))
        self.assertEqual([v["label"] for v in vs[1:]],
                         ["wording_id=w2", "max_tokens=8192", "temperature=0.7", "temperature=1.0"])
        configs = set()
        for v in vs[1:]:
            diff = {k for k in v["config"] if v["config"][k] != base["config"][k]}
            self.assertEqual(diff, {v["key"]})
            self.assertEqual(v["change"], {v["key"]: v["value"]})
            full, err = validate_proposal({"change": dict(v["change"])}, _ctx(self.space, base["config"]),
                                          require_fields=False)
            self.assertIsNone(err)
            self.assertEqual(full, v["config"])
            self.assertEqual(self.space.label(v["config"]), v["label"])
            configs.add(json.dumps(v["config"], sort_keys=True))
        self.assertEqual(len(configs), 4)
        two = {**base["config"], "max_tokens": 8192, "temperature": 1.0}
        _, err = validate_proposal({"change": {"max_tokens": 8192, "temperature": 1.0}},
                                   _ctx(self.space, base["config"]), require_fields=False)
        self.assertIn("exactly one key", err)
        self.assertEqual(variant_label(two, base["config"]), "max_tokens=8192,temperature=1.0")

    def test_variants_from_another_base_and_extra_axes(self):
        s = harness_variants(["a", "b"], 256, [0.0, 1.0], extra={"harness": ["edsl", "otree"]},
                             defaults={"harness": "otree"})
        self.assertEqual(s.defaults["harness"], "otree")
        self.assertEqual(s.params["max_tokens"], [256])
        vs = s.variants({"wording_id": "b"})
        self.assertEqual([v["label"] for v in vs], ["default", "wording_id=a", "temperature=1.0", "harness=edsl"])
        self.assertIsNone(s.wordings)
        with self.assertRaises(KeyError):
            s.wording("a")

    def test_fixed_forbidden_and_check_are_respected(self):
        s = harness_variants(["a", "b"], [256, 1024], [0.0, 1.0], fixed=["wording_id"],
                             check=lambda c: ["no hot long"] if c["max_tokens"] == 1024 and c["temperature"] else [])
        labels = [v["label"] for v in one_change_variants(s, {"temperature": 1.0})]
        self.assertEqual(labels, ["default", "temperature=0.0"])

    def test_invalid_axes_are_refused(self):
        for kw in ({"wordings": []}, {"wordings": ["a", "a"]}, {"wordings": [""]}, {"wordings": [1]},
                   {"wordings": ["a"], "max_tokens_list": [0]}, {"wordings": ["a"], "max_tokens_list": [True]},
                   {"wordings": ["a"], "max_tokens_list": [1.5]}, {"wordings": ["a"], "temperatures": [-0.1]},
                   {"wordings": ["a"], "temperatures": [float("inf")]}, {"wordings": ["a"], "extra": {"max_tokens": [1]}},
                   {"wordings": ["a"], "defaults": {"wording_id": "zz"}}):
            with self.assertRaises(ValueError, msg=str(kw)):
                harness_variants(**kw)

    def test_fingerprint_covers_wording_text(self):
        a = harness_variants({"w1": "text one"}, [None], [0.0])
        b = harness_variants({"w1": "text one, edited"}, [None], [0.0])
        self.assertNotEqual(a.fingerprint(), b.fingerprint())
        self.assertEqual(a.fingerprint(), harness_variants({"w1": "text one"}, [None], [0.0]).fingerprint())


def _harness_runner(calls=None, nan_units=(), crash_on=None):
    """Per-unit 0/1 'punish-low vote'. Only max_tokens=8192 changes the behaviour (a token cap acting as a
    treatment); wording and temperature have no effect. Per-unit propensity is shared across configs."""
    units = [f"u{i}" for i in range(200)]
    p = np.random.default_rng(7).uniform(0.2, 0.6, len(units))

    def run(cfg):
        if calls is not None:
            calls.append(dict(cfg))
        if crash_on and cfg.get(crash_on[0]) == crash_on[1]:
            raise RuntimeError("harness down")
        shift = 0.35 if cfg["max_tokens"] == 8192 else 0.0
        rng = np.random.default_rng(int(hashlib.sha256(f"{cfg['wording_id']}|{cfg['temperature']}".encode())
                                    .hexdigest()[:8], 16))
        noise = rng.normal(0, 0.02, len(units))
        y = np.clip(p + shift + noise, 0, 1)
        y = [None if u in nan_units else float(v) for u, v in zip(units, y)]
        return {"per_unit": pd.DataFrame({"unit": units, "punish_low": y})}
    return run


class TestHarnessSweep(unittest.TestCase):
    def test_sweep_finds_the_one_knob_that_matters(self):
        s = harness_variants(["w1", "w2"], [None, 8192], [0.0, 1.0])
        calls = []
        df = harness_sweep(_harness_runner(calls, nan_units={"u3", "u4"}), s, "punish_low")
        self.assertEqual(list(df["label"]), ["default", "wording_id=w2", "max_tokens=8192", "temperature=1.0"])
        self.assertTrue(all(c["split"] == "dev" for c in calls))
        self.assertEqual(len(calls), 4)
        d = df.set_index("label")
        self.assertEqual(d.loc["max_tokens=8192", "direction"], "higher")
        self.assertGreater(d.loc["max_tokens=8192", "lo_bonf"], 0.25)
        self.assertEqual(d.loc["wording_id=w2", "direction"], "same")
        self.assertEqual(d.loc["temperature=1.0", "direction"], "same")
        self.assertTrue((df["n_missing"] == 2).all())
        self.assertTrue((df["n_units"] == 198).all())
        self.assertTrue(pd.isna(d.loc["default", "diff"]))
        self.assertLessEqual(d.loc["max_tokens=8192", "lo_bonf"], d.loc["max_tokens=8192", "lo"])
        self.assertEqual(df.attrs["k"], 3)
        self.assertGreater(df.attrs["mean_range"][1] - df.attrs["mean_range"][0], 0.25)

    def test_crashing_variant_is_recorded_not_fatal(self):
        s = harness_variants(["w1", "w2"], [None, 8192], [0.0])
        df = harness_sweep(_harness_runner(crash_on=("wording_id", "w2")), s, "punish_low", split="select")
        d = df.set_index("label")
        self.assertIn("harness down", d.loc["wording_id=w2", "error"])
        self.assertEqual(d.loc["max_tokens=8192", "direction"], "higher")

    def test_generic_space_with_ranges(self):
        s = Space({"k": (1, 9), "m": ["a", "b"]}, defaults={"k": 5, "m": "a"})
        vs = one_change_variants(s, n_grid=3)
        self.assertEqual([v["label"] for v in vs], ["default", "k=1", "k=9", "m=b"])
        with self.assertRaises(ValueError):
            one_change_variants(s, {"k": 99})


class TestHarnessInLoop(unittest.TestCase):
    def test_loop_with_targets_prereg_and_harness_space(self):
        s = harness_variants(["w1", "w2"], [None, 8192], [0.0])
        with tempfile.TemporaryDirectory() as d:
            p = _prereg()
            declare_targets(ERTAN_FINDINGS, p)
            f = p.write(Path(d) / "prereg.md")
            lock_prereg(f)
            loop = Loop(_harness_runner(), s, proposer=GridProposer(), primary="punish_low", higher_is_better=True,
                        stop_after=3, max_turns=4, prereg=f, workdir=Path(d) / "run")
            out = loop.run(finalize=False)
            st = json.loads((Path(d) / "run" / "state.json").read_text())
            self.assertEqual(st["prereg"]["sha256"], lock_prereg(f)["sha256"])
            self.assertEqual(out["best_cfg"]["max_tokens"], 8192)


if __name__ == "__main__":
    unittest.main()
