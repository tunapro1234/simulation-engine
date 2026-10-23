"""Shared helpers of the simengine.persona.latent tests (tiny random models, no download)."""
import importlib.util
import unittest

HAVE_TORCH = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "tokenizers"))
needs_torch = unittest.skipUnless(HAVE_TORCH, "torch / transformers not installed")

_CACHE = {}


def tiny(family, seed=0):
    """A cached tiny model per (family, seed); tests must not change its weights."""
    import torch
    torch.set_num_threads(4)
    from simengine.persona.latent import models
    key = (family, seed)
    if key not in _CACHE:
        _CACHE[key] = models.tiny(family, seed=seed)
    return _CACHE[key]


CATS = ["ABIS_BOOK", "PET_FOOD", "SHOES", "BATTERY"]


def fake_root(tmp, n_train=40, n_test=10, seed=0):
    """A miniature prepared-data directory (talp73 / twin / twin_fiyat layout of runs/latent/veri_hazirla.py) with
    a learnable signal: survey 'gelir' and past purchases drive y. Returns the root Path."""
    from datetime import date
    from pathlib import Path

    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    root = Path(tmp)
    d = root / "talp73"
    d.mkdir(parents=True, exist_ok=True)
    uids = [f"u{i:03d}" for i in range(n_train + n_test)]
    gelir = rng.choice(["Less than $25,000", "$50,000 - $74,999", "$150,000 or more"], len(uids))
    yas = rng.choice(["18 - 24 years", "35 - 44 years", "65 and older"], len(uids))
    pd.DataFrame({"uid": uids, "yas": yas, "gelir": gelir, "cinsiyet": rng.choice(["Female", "Male"], len(uids)),
                  "hd_cocuk": rng.integers(0, 2, len(uids))}).to_parquet(d / "kisi_ozellik.parquet")
    prof = [f"- Age group: {a}\n- Household income before taxes, past 12 months: {g}\n- Gender: x"
            for a, g in zip(yas, gelir)]
    pd.DataFrame({"uid": uids, "metin": prof}).to_parquet(d / "kisi_metin.parquet")
    pd.DataFrame({"sira": range(1, len(CATS) + 1), "oge": CATS, "ad": CATS, "gecmis_alici": 1, "gecmis_oran": 0.5,
                  "istem_adi": [c.title() for c in CATS]}).to_parquet(d / "oge.parquet")
    rows, hist, ht, ev = [], [], [], []
    rich = np.array([g.startswith("$150") for g in gelir])
    for s in (0, 1, 2):
        test = set(uids[n_train:]) if s == 0 else set(rng.choice(uids, n_test, replace=False).tolist())
        train = [u for u in uids if u not in test]
        val = set(rng.choice(train, max(len(train) // 5, 2), replace=False).tolist())
        for i, u in enumerate(uids):
            b = "test" if u in test else "train"
            k = date(2022, 7, 1) if b == "test" else date(2022, 1, 1)
            for j, c in enumerate(CATS):
                past = int(rng.integers(0, 4))
                y = int(rng.random() < 0.15 + 0.5 * (past > 1) + 0.25 * rich[i] * (j % 2))
                rows.append((s, u, c, y, b, k, u in val))
                if s == 0 or b == "test":
                    hist.append((u, k, c, past, past * 2, float(past) * 0.5, past * 3))
    t = pd.DataFrame(rows, columns=["tohum", "uid", "oge", "y", "bolum", "kesim", "ic_dogrulama"])
    t["y"] = t["y"].astype("int8")
    t.to_parquet(d / "hedef.parquet")
    h = pd.DataFrame(hist, columns=["uid", "kesim", "oge", "k_365", "k_730", "son_365", "u_k_365"]).drop_duplicates(
        ["uid", "kesim", "oge"])
    h.to_parquet(d / "gecmis.parquet")
    for (u, k), g in h.groupby(["uid", "kesim"]):
        ht.append((u, k, "January 2018 - June 2022", "\n".join(f"{c}: {n} items" for c, n in zip(g.oge, g.k_365))))
        for c, n in zip(g.oge, g.k_365):
            ev += [(u, k, date(2021, 5, 1), c, 9.99, 1.0, "item")] * int(n)
        ev.append((u, k, date(2021, 6, 1), "OTHER_" + str(rng.integers(0, 3)), 5.0, 1.0, "x"))
    pd.DataFrame(ht, columns=["uid", "kesim", "aralik", "metin"]).to_parquet(d / "gecmis_metin.parquet")
    pd.DataFrame(ev, columns=["uid", "kesim", "tarih", "kategori", "fiyat", "adet", "baslik"]).to_parquet(
        d / "gecmis_olay.parquet")
    # twin / twin_fiyat (G2 / G3)
    tw, tf = root / "twin", root / "twin_fiyat"
    tw.mkdir(exist_ok=True)
    tf.mkdir(exist_ok=True)
    pid = list(range(1, n_train + n_test + 1))
    sk = rng.normal(size=len(pid))
    pd.DataFrame({"uid": pid, "QID1": rng.integers(1, 4, len(pid)), "skor_x_d1": sk}).to_parquet(
        tw / "kisi_ozellik.parquet")
    pd.DataFrame({"uid": pid, "metin": "long", "metin_demografi": [f"- Age: {a}" for a in yas]}).to_parquet(
        tw / "kisi_metin.parquet")
    bol = ["train"] * n_train + ["test"] * n_test
    icd = [i % 5 == 0 and i < n_train for i in range(len(pid))]
    pd.DataFrame({"oge": ["Q1_1"], "qid": ["Q1"], "blok": ["b"], "tip": ["Matrix"], "soru": ["Support?"],
                  "alt": ["A tax"], "n_secenek": [3.0], "alt_sinir": [1.0], "ust_sinir": [3.0],
                  "secenekler": ['["No", "Maybe", "Yes"]'], "kosul_ailesi": ["b"], "olcek_alt": [1.0],
                  "olcek_ust": [3.0]}).to_parquet(tw / "oge.parquet")
    yv = np.clip(np.round(2 + sk), 1, 3)
    pd.DataFrame({"uid": pid, "oge": "Q1_1", "y": yv, "y01": (yv - 1) / 2, "bolum": bol, "ic_dogrulama": icd}
                 ).to_parquet(tw / "hedef.parquet")
    prods = ["P01", "P02", "P03"]
    pd.DataFrame({"oge": prods + ["P04", "P05"], "urun": ["Soap", "Tea", "Rice", "Milk", "Salt"],
                  "kategori": "c", "medyan_fiyat": [2.0, 3.0, 4.0, 1.0, 1.0], "fiyat_duzey": 11,
                  "rol": ["hedef"] * 3 + ["gecmis"] * 2}).to_parquet(tf / "oge.parquet")
    levels = np.round(np.arange(0, 2.01, 0.2), 1)
    grid = pd.DataFrame([(p, m, l, round(m * l, 2)) for p, m in zip(prods, [2.0, 3.0, 4.0]) for l in levels],
                        columns=["oge", "medyan_fiyat", "carpan_duzey", "fiyat"])
    grid.to_parquet(tf / "fiyat_izgara.parquet")
    tr, gh = [], []
    for i, u in enumerate(pid):
        for p, m in zip(prods, [2.0, 3.0, 4.0]):
            l = float(rng.choice(levels))
            tr.append((u, p, int(rng.random() < 0.9 - 0.35 * l), bol[i], icd[i], round(m * l, 2), l, l, "s"))
        for p in ("P04", "P05"):
            l = float(rng.choice(levels))
            gh.append((u, p, int(rng.random() < 0.9 - 0.35 * l), l, l, l, "s"))
    pd.DataFrame(tr, columns=["uid", "oge", "y", "bolum", "ic_dogrulama", "fiyat", "carpan", "carpan_duzey", "slot"]
                 ).to_parquet(tf / "hedef.parquet")
    pd.DataFrame(gh, columns=["uid", "oge", "al", "fiyat", "carpan", "carpan_duzey", "slot"]).to_parquet(
        tf / "gecmis.parquet")
    return root
