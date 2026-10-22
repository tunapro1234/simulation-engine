"""Equivalence with the original research package `kanit` (same input -> same report).

Imports the original `kanit` package from the directory named in SIMENGINE_KANIT_SRC (the research tree's
`kanit/kod`); skipped when the variable is unset or the package is not there.
"""
import importlib
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import simengine.evidence as ev
from simengine.evidence import bedel, kart, onkayit, orneklem, pano, sema

_SRC_ENV = os.environ.get("SIMENGINE_KANIT_SRC")
_SRC = Path(_SRC_ENV) if _SRC_ENV else None


def _orig():
    if _SRC is None or not (_SRC / "kanit" / "__init__.py").exists():
        return None
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True       # never write __pycache__ into the read-only research tree
    sys.path.insert(0, str(_SRC))
    try:
        k = importlib.import_module("kanit")
        mods = {m: importlib.import_module(f"kanit.{m}") for m in ("bedel", "orneklem", "onkayit", "pano", "kart", "sema")}
        return k, mods
    finally:
        sys.path.remove(str(_SRC))
        sys.dont_write_bytecode = old


ORIG = _orig()
EN = {"gercek": "truth", "bolum": "split", "grup": "group", "madde": "item", "tekrar": "retest", "zaman": "time"}


def _english(df):
    out = df.rename(columns={**EN, **{c: "pred_" + c[len("tahmin_"):] for c in df.columns if c.startswith("tahmin_")}})
    if "split" in out:
        out["split"] = out["split"].map({"egitim": "train"}).fillna(out["split"])
    return out


def _strip(oz):
    oz = json.loads(json.dumps(oz))
    oz.pop("olusturma", None)
    return oz


TRIVIAL = "yalnız basit tabanlar"


def _base(oz):
    """Drop the additive simengine extensions (declared-baseline block, ceiling kind, the "only trivial baselines"
    audit, limit line and card flag) so the rest can be compared with the original kanit report."""
    oz = json.loads(json.dumps(oz))
    oz.pop("tabanlar", None)
    oz.pop("tavan_turu", None)
    oz["denetim"] = [d for d in oz.get("denetim") or [] if d.get("kontrol") != TRIVIAL]
    oz["sinirlar"] = [x for x in oz.get("sinirlar") or [] if not x.lower().startswith(TRIVIAL)]
    k = oz.get("kanit_kart")
    if k:
        k["etiketler"] = [x for x in k.get("etiketler") or [] if not x.startswith(TRIVIAL)]
        k["yanilgi"] = [x for x in k.get("yanilgi") or [] if not x.lower().startswith(TRIVIAL)]
    return oz


def _continuous(seed=0, P=60, M=24):
    rng = np.random.default_rng(seed)
    kisi = np.repeat(np.arange(P), M)
    madde = np.tile(np.arange(M), P)
    mu = rng.uniform(0.2, 0.8, M)
    a = rng.normal(0, 0.08, P)
    y = np.clip(mu[madde] + a[kisi] + rng.normal(0, 0.1, P * M), 0, 1)
    return pd.DataFrame({
        "grup": [f"p{i}" for i in kisi], "madde": [f"q{j}" for j in madde],
        "segment": np.where(kisi % 2, "A", "B"),
        "gercek": y, "tekrar": np.clip(y + rng.normal(0, 0.08, P * M), 0, 1),
        "tahmin_llm": np.clip(mu[madde] + rng.normal(0, 0.15, P * M), 0, 1),
        "tahmin_hibrit": np.clip(mu[madde] + 0.6 * a[kisi] + rng.normal(0, 0.08, P * M), 0, 1),
    })


def _binary(seed=1, n=500):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (x + rng.normal(0, 1, n) > 0).astype(int)
    return pd.DataFrame({
        "grup": [f"u{i}" for i in range(n)], "bolum": np.where(np.arange(n) < 150, "egitim", "test"),
        "seg_yas": rng.choice(["genc", "yasli"], n),
        "gercek": y, "tahmin_llm": 1 / (1 + np.exp(-0.5 * x)), "tahmin_bizim": 1 / (1 + np.exp(-1.5 * x)),
    })


def _distribution(seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(20):
        for s in range(3):
            p = rng.dirichlet(np.ones(4) * 2)
            rows.append({"madde": f"m{m}", "segment": f"s{s}", "gercek": p.tolist(),
                         "tahmin_sim": (0.8 * p + 0.2 * rng.dirichlet(np.ones(4))).tolist(),
                         "tahmin_llm": rng.dirichlet(np.ones(4)).tolist()})
    return pd.DataFrame(rows)


def _multiclass(seed=3, n=400):
    rng = np.random.default_rng(seed)
    g = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    return pd.DataFrame({"grup": np.arange(n), "gercek": g,
                         "tahmin_x": np.where(rng.random(n) < 0.6, g, rng.choice(["a", "b", "c"], n)),
                         "tahmin_markov": rng.choice(["a", "b"], n)})


@unittest.skipIf(ORIG is None, "original kanit package not present")
class TestEquivalence(unittest.TestCase):
    def _cmp(self, df, **kw_en):
        kanit, _ = ORIG
        tr_kw = {"bizim": kw_en.get("ours"), "ham": kw_en.get("raw"), "B": 200, "B_toplu": 40,
                 "etiketsiz_tabanlar": kw_en.get("unlabeled_baselines")}
        if "task" in kw_en:
            tr_kw["gorev"] = ev.TASK_TO_INTERNAL[kw_en["task"]]
        if "multiplier" in kw_en:
            tr_kw["carpan"] = kw_en["multiplier"]
        o = kanit.degerlendir(df.copy(), "t", "v", **tr_kw)
        e = _base(ev.evaluate(_english(df), "t", dataset="v", B=200, B_aggregate=40, **kw_en))
        self.assertEqual(_strip(o), _strip(e))
        # Turkish aliases and English columns reach the same internal frame
        pd.testing.assert_frame_equal(ev.to_internal(df), ev.to_internal(_english(df)))
        # dashboard HTML identical for the same report
        _, mods = ORIG
        e["olusturma"] = o["olusturma"]
        self.assertEqual(mods["pano"].html([o]), pano.html([e]))
        self.assertEqual(mods["kart"].kart(o), kart.kart(e))
        return e

    def test_continuous_with_ceiling(self):
        e = self._cmp(_continuous(), ours="hibrit", raw="llm")
        self.assertIsNotNone(e["manset"]["bosluk_kapanan"])

    def test_binary_with_split(self):
        # single-item multiplier path (slow PPI++ bootstrap) is covered by test_worth_humans_and_multiplier
        e = self._cmp(_binary(), ours="bizim", raw="llm", multiplier=False)
        self.assertEqual(e["gorev"], "ikili")

    def test_distribution(self):
        self._cmp(_distribution(), ours="sim", raw="llm")

    def test_multiclass_label_baseline(self):
        self._cmp(_multiclass(), ours="x", task="multiclass", unlabeled_baselines=["markov"])

    def test_worth_humans_and_multiplier(self):
        _, mods = ORIG
        rng = np.random.default_rng(4)
        y = rng.normal(size=800)
        f = y + rng.normal(0, 0.5, 800)
        self.assertEqual(mods["bedel"].olc([(y, f)], n=100, B=50), bedel.olc([(y, f)], n=100, B=50))
        self.assertEqual(mods["bedel"].olc([(y, f)], n=100, B=50), ev.worth_humans((y, f), n=100, B=50))
        d = _continuous()
        vo, vn = mods["sema"].hazirla(d), sema.hazirla(d)
        a = mods["orneklem"].carpan(vo, {"tahmin_hibrit": "h"}, n_listesi=(10, 25), B=50)
        b = orneklem.carpan(vn, {"tahmin_hibrit": "h"}, n_listesi=(10, 25), B=50)
        self.assertEqual(json.dumps(a), json.dumps(b))

    def test_canonical_bytes(self):
        _, mods = ORIG
        d = _distribution()
        self.assertEqual(mods["onkayit"].kanonik_bayt(d), onkayit.kanonik_bayt(d))


if __name__ == "__main__":
    unittest.main()
