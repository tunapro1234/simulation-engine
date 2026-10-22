"""opinionqa_groups LLM arm: group descriptions, segments() and predict_with(data, 'llm') on a tiny synthetic task
with FakeLLM (no real calls, no real rows); the real-data check runs only when $SIMENGINE_DATA is set."""
import json
import os
import re
import tempfile
import unittest

import numpy as np

from simengine import bench
from simengine.bench.base import DataNotFound
from simengine.bench.tasks.opinionqa import GROUP_DESCRIPTIONS, OpinionQAGroups, _long, describe_group, segment_key
from simengine.llm.fake import FakeLLM
from simengine.types import TaskData

from .test_bench_adapters import _oqa

GROUPS = [("RACE", "Hispanic"), ("SEX", "Female"), ("AGE", "18-29")]
# description -> favourite option text: the fake puts 70% there, so option order mapping is checkable
FAV = {"Hispanic adults in the US": "Agree", "women in the US": "Disagree", "US adults aged 18-29": "Neutral"}


def _question(wave, text, options, groups):
    m = len(options)
    pos = np.linspace(0, 1, m)
    return dict(wave=wave, text=text, options=options, pos=pos, overall=np.ones(m) / m,
                dist={g: np.ones(m) / m for g in groups})


def tiny_data() -> TaskData:
    """Two waves; the second test question lists its options reversed and lacks one group."""
    pool = {"P1_W1": _question("W1", "Pool question one", ["Agree", "Neutral", "Disagree"], GROUPS),
            "P2_W2": _question("W2", "Pool question two", ["Agree", "Neutral", "Disagree"], GROUPS)}
    test = {"T1_W1": _question("W1", "Taxes should rise", ["Agree", "Neutral", "Disagree"], GROUPS),
            "T2_W2": _question("W2", "Schools are safe", ["Disagree", "Neutral", "Agree"], GROUPS[:2])}
    return TaskData(train=_long(pool), test=_long(test).assign(overall=None),
                    meta={"groups": GROUPS, "setting": "A"})


def responder(system, user):
    desc = re.search(r"(?:who are|Consider) (.+?)\.", user).group(1)
    opts = re.findall(r"^(\d+)\. (.+)$", user, re.M)
    fav = FAV.get(desc)
    rest = 30 / (len(opts) - 1)
    return json.dumps({n: 70 if o == fav else rest for n, o in opts})


class TestDescriptions(unittest.TestCase):
    def test_known_and_fallback(self):
        self.assertEqual(describe_group("RACE", "Hispanic"), "Hispanic adults in the US")
        self.assertEqual(OpinionQAGroups.describe_group("POLPARTY", "Democrat"), "US adults who identify as Democrats")
        self.assertEqual(describe_group("AGE", "65+"), "US adults aged 65+")
        self.assertIn("zodiac", describe_group("ZODIAC", "Leo"))
        self.assertEqual(len(GROUP_DESCRIPTIONS), 22)

    def test_segments(self):
        task = bench.get("opinionqa_groups")
        self.assertEqual(len(task.segments()), 22)
        segs = task.segments(tiny_data(), descriptions={("SEX", "Female"): "female respondents"})
        self.assertEqual(segs, {"RACE:Hispanic": "Hispanic adults in the US", "SEX:Female": "female respondents",
                                "AGE:18-29": "US adults aged 18-29"})
        self.assertEqual(segment_key("RACE", "Hispanic"), "RACE:Hispanic")


class TestLLMArm(unittest.TestCase):
    def test_row_and_option_order(self):
        task, data = bench.get("opinionqa_groups"), tiny_data()
        llm = FakeLLM(responder)
        preds = task.predict_with(data, "llm", llm=llm, framings={"h2": "Consider {group}."})
        self.assertEqual(len(preds), len(data.test))
        for r, p in zip(data.test.itertuples(), preds):
            self.assertAlmostEqual(float(np.sum(p)), 1.0)
            fav = FAV[describe_group(r.attribute, r.group)]
            self.assertEqual(r.options[int(np.argmax(p))], fav, (r.qkey, r.group))
        # only the 5 needed cells were asked (T2 has no AGE row), one framing, one sample
        self.assertEqual(len(llm.calls), 5)
        self.assertEqual(task.last_llm["calls"], 5)
        self.assertEqual(task.last_llm["invalid"], 0)
        self.assertEqual(task.last_llm["filled"][("T1_W1", "RACE", "Hispanic")], 0.0)
        frame = task.frame(data, {"llm": preds, **task.baselines(data)})
        self.assertEqual(task.score(frame, "llm")["metric"], "wd")

    def test_framings_samples_and_invalid(self):
        task, data = bench.get("opinionqa_groups"), tiny_data()
        llm = FakeLLM()   # hash replies: every answer invalid -> uniform fill
        preds = task.predict_with(data, "llm", llm=llm, samples=2)
        self.assertEqual(len(llm.calls), 5 * 3 * 2)   # default hidden framings x 2 samples
        self.assertEqual(task.last_llm["invalid"], 30)
        self.assertAlmostEqual(task.last_llm["filled_share"], 1.0)
        for p in preds:
            np.testing.assert_allclose(p, np.ones(3) / 3)

    def test_skip_fallback_counts_uniform_stand_in(self):
        task, data = bench.get("opinionqa_groups"), tiny_data()
        preds = task.predict_with(data, "llm", llm=FakeLLM(), fallback="skip", framings={"h2": "Consider {group}."})
        self.assertEqual(task.last_llm["missing"], 5)
        self.assertAlmostEqual(task.last_llm["filled_share"], 1.0)
        self.assertEqual(task.last_llm["filled"][("T2_W2", "SEX", "Female")], 1.0)
        for p in preds:
            np.testing.assert_allclose(p, np.ones(3) / 3)

    def test_needs_llm(self):
        with self.assertRaises(TypeError):
            bench.get("opinionqa_groups").predict_with(tiny_data(), "llm")

    def test_end_to_end_from_files(self):
        with tempfile.TemporaryDirectory() as root:
            _oqa(root)
            task = bench.get("opinionqa_groups")
            data = task.load(root)
            preds = task.predict_with(data, "llm", llm=FakeLLM(responder), framings={"h2": "Consider {group}."})
            self.assertEqual([len(p) for p in preds], [len(x) for x in data.test["pos"]])
            frame = task.run(methods={"llm": lambda d: preds}, data=data)
            self.assertTrue(np.isfinite(task.score(frame, "llm")["value"]))


@unittest.skipUnless(os.environ.get("SIMENGINE_DATA"), "$SIMENGINE_DATA not set")
class TestRealData(unittest.TestCase):
    def test_all_groups_described_and_rows_mapped(self):
        task = bench.get("opinionqa_groups")
        try:
            data = task.load(os.environ["SIMENGINE_DATA"])
        except DataNotFound:
            self.skipTest("opinionqa not under $SIMENGINE_DATA")
        groups = [tuple(g) for g in data.meta["groups"]]
        self.assertTrue(all(g in GROUP_DESCRIPTIONS for g in groups))
        keep = data.test["qkey"].isin(sorted(data.test["qkey"].unique())[:3])
        small = TaskData(train=data.train.iloc[:0], test=data.test[keep].reset_index(drop=True), meta=data.meta)
        llm = FakeLLM()
        preds = task.predict_with(small, "llm", llm=llm, framings={"h2": "Consider {group}."})
        self.assertEqual(len(preds), len(small.test))
        self.assertEqual(len(llm.calls), len(small.test))
        self.assertEqual([len(p) for p in preds], [len(x) for x in small.test["pos"]])


if __name__ == "__main__":
    unittest.main()
