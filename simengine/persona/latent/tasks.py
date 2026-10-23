"""Task tables of the latent-persona experiments (prepared by runs/latent/veri_hazirla.py) and the leak guard.

G1 talp73 (person x category purchase, three seeds), G2 Twin (wave-4 items), G3 Twin pricing (counterfactual).
`root` is the prepared data directory (with talp73/, twin/, twin_fiyat/ under it); no machine path is built in.

    g = load_g1(root, seed=0)
    fit, val = g.fit_rows(), g.val_rows()        # train persons outside / inside ic_dogrulama
    guard = g.guard()                            # TestGuard holding this seed's test uids
    guard.check(fit, "probe fit")                # RuntimeError if a test row or a test uid is in there
    write_pred(out_dir, "prb-q38-L2", 0, pred, meta, n_expected=8000)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class LeakError(RuntimeError):
    """A test row or a test person reached a fit / select / calibrate step."""


class TestGuard:
    """Leak guard. `check(df, where)` raises LeakError if df has a bolum == "test" row or, on an instance, any uid
    of the held-out persons; `check_uids(uids, where)` checks a bare id list."""
    __test__ = False                                   # not a unittest class

    def __init__(self, test_uids: Iterable = ()):
        self.test_uids = frozenset(test_uids)

    @staticmethod
    def check_rows(df, where: str) -> None:
        if df is not None and "bolum" in getattr(df, "columns", ()) and bool((df["bolum"] == "test").any()):
            raise LeakError(f"{where}: test rows passed to a fit / select / calibrate step")

    def check_uids(self, uids, where: str) -> None:
        bad = self.test_uids.intersection(pd.unique(pd.Series(list(uids))).tolist()) if self.test_uids else set()
        if bad:
            raise LeakError(f"{where}: {len(bad)} test person(s) in a training input (e.g. {sorted(map(str, bad))[:3]})")

    def check(self, df, where: str) -> None:
        self.check_rows(df, where)
        if self.test_uids and df is not None and "uid" in getattr(df, "columns", ()):
            self.check_uids(df["uid"], where)


def _read(path: Path, columns=None, filters=None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns, filters=filters)


@dataclass
class G1:
    """talp73 for one seed. target: uid, oge, y, bolum, kesim, ic_dogrulama. Person tables are keyed by uid and,
    for history, by (uid, kesim): a train person's inputs are at the 2022-01-01 cut, a test person's at 2022-07-01."""
    seed: int
    root: Path
    target: pd.DataFrame
    survey: pd.DataFrame
    profile: pd.DataFrame
    history: pd.DataFrame
    history_text: pd.DataFrame
    items: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def persons(self) -> pd.DataFrame:
        """One row per person of this seed: uid, kesim, bolum, ic_dogrulama (sorted by bolum, uid)."""
        return (self.target.drop_duplicates("uid")[["uid", "kesim", "bolum", "ic_dogrulama"]]
                .sort_values(["bolum", "uid"]).reset_index(drop=True))

    def fit_rows(self) -> pd.DataFrame:
        t = self.target
        return t[(t.bolum == "train") & ~t.ic_dogrulama].reset_index(drop=True)

    def val_rows(self) -> pd.DataFrame:
        t = self.target
        return t[(t.bolum == "train") & t.ic_dogrulama].reset_index(drop=True)

    def train_rows(self) -> pd.DataFrame:
        t = self.target
        return t[t.bolum == "train"].reset_index(drop=True)

    def test_rows(self) -> pd.DataFrame:
        t = self.target
        return t[t.bolum == "test"].reset_index(drop=True)

    def categories(self) -> list[str]:
        return self.items.sort_values("sira")["oge"].tolist()

    def guard(self) -> TestGuard:
        return TestGuard(self.test_rows()["uid"].unique())

    def events(self, columns=None) -> pd.DataFrame:
        """All line items before each person's cut (1.7M rows; read on demand)."""
        return _read(self.root / "gecmis_olay.parquet", columns=columns)


def load_g1(root, seed: int, *, cap_fit: int | None = None, cap_val: int | None = None) -> G1:
    """Seed `seed` of talp73. cap_fit / cap_val keep at most that many fit / validation persons (24 GB plan:
    default_rng(7) / default_rng(8) over sorted uids); test persons are never capped."""
    d = Path(root) / "talp73"
    t = _read(d / "hedef.parquet")
    t = t[t["tohum"] == seed].drop(columns="tohum").reset_index(drop=True)
    t["kesim"] = pd.to_datetime(t["kesim"]).dt.date
    for cap, val, rng in ((cap_fit, False, 7), (cap_val, True, 8)):
        if cap is None:
            continue
        pool = np.array(sorted(t[(t.bolum == "train") & (t.ic_dogrulama == val)].uid.unique()))
        if len(pool) > cap:
            drop = set(pool) - set(np.random.default_rng(rng).choice(pool, cap, replace=False).tolist())
            t = t[~t.uid.isin(drop)].reset_index(drop=True)
    uids = set(t.uid)
    hist = _read(d / "gecmis.parquet")
    hist["kesim"] = pd.to_datetime(hist["kesim"]).dt.date
    keys = t.drop_duplicates("uid")[["uid", "kesim"]]
    hist = hist.merge(keys, on=["uid", "kesim"], how="inner")
    ht = _read(d / "gecmis_metin.parquet")
    ht["kesim"] = pd.to_datetime(ht["kesim"]).dt.date
    ht = ht.merge(keys, on=["uid", "kesim"], how="inner")
    survey = _read(d / "kisi_ozellik.parquet")
    profile = _read(d / "kisi_metin.parquet")
    items = _read(d / "oge.parquet")
    return G1(seed, d, t, survey[survey.uid.isin(uids)].reset_index(drop=True),
              profile[profile.uid.isin(uids)].reset_index(drop=True), hist.reset_index(drop=True),
              ht.reset_index(drop=True), items, {"cap_fit": cap_fit, "cap_val": cap_val})


@dataclass
class G2:
    """Twin-2K-500 wave-4 items, one split (seed 0). target: uid, oge, y, y01, bolum, ic_dogrulama."""
    root: Path
    target: pd.DataFrame
    survey: pd.DataFrame
    profile: pd.DataFrame
    items: pd.DataFrame
    seed: int = 0

    def option_items(self) -> pd.DataFrame:
        return self.items[self.items.tip.isin(["MC", "Matrix"])].reset_index(drop=True)

    def fit_rows(self):
        t = self.target
        return t[(t.bolum == "train") & ~t.ic_dogrulama].reset_index(drop=True)

    def val_rows(self):
        t = self.target
        return t[(t.bolum == "train") & t.ic_dogrulama].reset_index(drop=True)

    def train_rows(self):
        return self.target[self.target.bolum == "train"].reset_index(drop=True)

    def test_rows(self):
        return self.target[self.target.bolum == "test"].reset_index(drop=True)

    def guard(self) -> TestGuard:
        return TestGuard(self.test_rows()["uid"].unique())


def load_g2(root) -> G2:
    d = Path(root) / "twin"
    return G2(d, _read(d / "hedef.parquet"), _read(d / "kisi_ozellik.parquet"), _read(d / "kisi_metin.parquet"),
              _read(d / "oge.parquet"))


@dataclass
class G3:
    """Twin pricing. target: uid, oge, y, fiyat, carpan, bolum, ic_dogrulama; history: the person's 30 history
    products (al, fiyat, carpan). Fit / val rows keep carpan <= max_carpan (the extrapolation split)."""
    root: Path
    target: pd.DataFrame
    history: pd.DataFrame
    items: pd.DataFrame
    price_grid: pd.DataFrame
    survey: pd.DataFrame
    profile: pd.DataFrame
    max_carpan: float = 1.0
    seed: int = 0

    def fit_rows(self):
        t = self.target
        return t[(t.bolum == "train") & ~t.ic_dogrulama & (t.carpan <= self.max_carpan + 1e-9)].reset_index(drop=True)

    def val_rows(self):
        t = self.target
        return t[(t.bolum == "train") & t.ic_dogrulama & (t.carpan <= self.max_carpan + 1e-9)].reset_index(drop=True)

    def test_rows(self):
        return self.target[self.target.bolum == "test"].reset_index(drop=True)

    def grid(self) -> pd.DataFrame:
        return self.price_grid

    def guard(self) -> TestGuard:
        return TestGuard(self.test_rows()["uid"].unique())


def load_g3(root, *, max_carpan: float = 1.0) -> G3:
    d = Path(root) / "twin_fiyat"
    tw = Path(root) / "twin"
    return G3(d, _read(d / "hedef.parquet"), _read(d / "gecmis.parquet"), _read(d / "oge.parquet"),
              _read(d / "fiyat_izgara.parquet"), _read(tw / "kisi_ozellik.parquet"), _read(tw / "kisi_metin.parquet"),
              max_carpan)


# ------------------------------------------------------------------------------------------------- predictions
def check_g1_pred(df: pd.DataFrame, n_expected: int | None = 8000) -> None:
    """talp73 prediction schema: uid, kategori, p in [0, 1], no NaN, no duplicate (uid, kategori)."""
    if not {"uid", "kategori", "p"} <= set(df.columns):
        raise ValueError(f"prediction needs uid, kategori, p (got {list(df.columns)})")
    p = df["p"].to_numpy(float)
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p must be in [0, 1] with no NaN")
    if df.duplicated(["uid", "kategori"]).any():
        raise ValueError("duplicate (uid, kategori) rows")
    if n_expected is not None and len(df) != n_expected:
        raise ValueError(f"expected {n_expected} rows, got {len(df)}")


def write_pred(out_dir, arm: str, seed: int, df: pd.DataFrame, meta: dict, *, n_expected: int | None = 8000,
               seal_registry=None, g1: bool = True) -> Path:
    """Write <out_dir>/<arm>_s<seed>.parquet + .meta.json. With seal_registry, the table is sealed first
    (simengine.research.seal, arm=f"{arm}_s{seed}"). g1=True checks the talp73 schema."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if g1:
        check_g1_pred(df, n_expected)
        df = df[["uid", "kategori", "p"]].sort_values(["uid", "kategori"]).reset_index(drop=True)
    path = out / f"{arm}_s{seed}.parquet"
    m = dict(meta)
    if seal_registry is not None:
        from simengine.research import seal
        rec = seal(df, f"latent {arm} seed {seed}", registry_dir=str(seal_registry), arm=f"{arm}_s{seed}")
        m["seal"] = {k: rec.get(k) for k in ("seq", "sha256", "time", "record_hash") if k in rec}
    df.to_parquet(path, index=False)
    path.with_suffix(".meta.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    return path


def link_baselines(src_dir, dst_dir, names=("gecmis", "gecmis_anket_ey", "anket_ey", "populerlik", "sabit"),
                   seeds=(0, 1, 2)) -> list[Path]:
    """Symlink existing talp73 baseline predictions next to the new ones (degerlendir.py --taban reads the same
    folder). Existing links are kept."""
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    made = []
    for n in names:
        for s in seeds:
            src = Path(src_dir) / f"{n}_s{s}.parquet"
            d = dst / src.name
            if src.exists() and not d.exists():
                os.symlink(src, d)
                made.append(d)
    return made


__all__ = ["LeakError", "TestGuard", "G1", "G2", "G3", "load_g1", "load_g2", "load_g3", "check_g1_pred",
           "write_pred", "link_baselines"]
