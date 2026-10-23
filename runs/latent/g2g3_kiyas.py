"""G2 / G3 side-by-side: LLM-free arms (kos.py), the dataset's own LLM simulations (Twin-2K-500
LLM_simulation_results, 13 specifications: GPT-4.1 / 4.1-mini / Gemini Flash 2.5, persona formats, fine-tune) and our
Claude prompt arms (claude_g2g3.py), all on the same 400 test persons and the same measures.

    PY="nice -n 19 /data/talp/venv/bin/python"
    PYTHONPATH=. $PY runs/latent/g2g3_kiyas.py            # -> /data/talp/is/latent/g2g3/kiyas.json + tables on stdout

G2: mad_acc = 1 - mean |y01_hat - y01| per person, then mean over persons (Twin paper's accuracy, lt.mad_accuracy);
    item_r = mean over items of the across-person Pearson r within the item (individual differentiation; a constant
    prediction has none). Paired person bootstrap (B 1000) of the difference to the base arm on the rows both have.
G3: test rows (observed price): auc, kisi_auc (within person), acc@0.5, brier; demand curve over the 11 price levels
    (pearson, slope); sweep arms also lt.score_g3 (slope over carpan >= 1 vs the true slope). Paired person bootstrap
    of auc / kisi_auc / brier differences to the base arm.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simengine.persona import latent as lt  # noqa: E402

BASE = Path("/data/talp/is/latent")
VERI = BASE / "veri"
TWIN = Path("/data/talp/veri/twin2k500/LLM_simulation_results")
SPEC = TWIN / "llm_simulations_all_specifications"
B = 1000
RNG = np.random.default_rng(42)


def spec_short(name: str) -> str:
    return "ds:" + name.replace(" - ", "|").replace(" ", "")


def dataset_specs(cols: list[str]) -> dict[str, pd.DataFrame]:
    """Every specification's imputed wave 4 answers, long (uid, col, v) for the catalog columns `cols`."""
    m = json.load(open(TWIN / "wave4_formatted_to_catalog_mapping.json"))
    es = {e["catalog_csv_column"]: e["formatted_column"] for e in m}
    out = {}
    files = sorted(glob.glob(str(SPEC / "*/csv_comparison/csv_formatted/responses_llm_imputed_formatted.csv")))
    for f in files:
        name = spec_short(Path(f).parts[-4])
        l = pd.read_csv(f, low_memory=False)
        l = l[pd.to_numeric(l["TWIN_ID"], errors="coerce").notna()].copy()
        l["uid"] = l["TWIN_ID"].astype(int)
        t = pd.DataFrame({c: pd.to_numeric(l[es[c]], errors="coerce").values for c in cols if c in es}, index=l["uid"])
        out[name] = t.stack().rename("v").reset_index().rename(columns={"level_1": "col"})
    return out


def boot_idx(n):
    return RNG.integers(0, n, size=(B, n))


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)] if len(x) else None


# ------------------------------------------------------------------------------------------------- G2
def item_r(d: pd.DataFrame, col: str) -> float:
    rs = []
    for _, g in d.groupby("oge"):
        if g["y01"].std() > 1e-9 and g[col].std() > 1e-9:
            rs.append(np.corrcoef(g["y01"], g[col])[0, 1])
    return float(np.mean(rs)) if rs else float("nan")


def g2():
    it = pd.read_parquet(VERI / "twin/oge.parquet").set_index("oge")
    h = pd.read_parquet(VERI / "twin/hedef.parquet")
    tr, te = h[h.bolum == "train"], h[h.bolum == "test"][["uid", "oge", "y01"]]
    arms = {"madde_ort": te[["uid", "oge"]].assign(p=te.oge.map(tr.groupby("oge").y01.mean()))}
    g = pd.read_parquet(VERI / "twin/gecmis.parquet")[["uid", "oge", "y_d1301"]].rename(columns={"y_d1301": "p"})
    arms["retest_d13 (gecmis, tavan)"] = te[["uid", "oge"]].merge(g, on=["uid", "oge"]).dropna()
    for f in sorted((BASE / "twin/tahmin").glob("*_s0.parquet")):
        arms[f.stem[:-3]] = pd.read_parquet(f)[["uid", "oge", "p"]]
    for name, t in dataset_specs(list(it.index)).items():
        t = t.rename(columns={"col": "oge"})
        lo, hi = it.loc[t.oge, "olcek_alt"].values, it.loc[t.oge, "olcek_ust"].values
        arms[name] = t.assign(p=np.clip((t.v - lo) / (hi - lo), 0, 1))[["uid", "oge", "p"]]
    test_uids = set(te.uid)
    arms = {k: v[v.uid.isin(test_uids)].dropna() for k, v in arms.items()}
    base = "ridge-tam"

    def pair(ka, kb):
        """Paired difference ka - kb on the (uid, oge) rows both arms have; person bootstrap."""
        db = te.merge(arms[ka], on=["uid", "oge"]).merge(arms[kb].rename(columns={"p": "pb"}), on=["uid", "oge"])
        pa = 1 - (db.p - db.y01).abs().groupby(db.uid).mean()
        pb = 1 - (db.pb - db.y01).abs().groupby(db.uid).mean()
        diff = (pa - pb).to_numpy()
        uids = np.array(sorted(db.uid.unique()))
        dbi = db.set_index("uid")
        rb = [item_r(dbi.loc[uids[r]].reset_index(), "p") - item_r(dbi.loc[uids[r]].reset_index(), "pb")
              for r in RNG.integers(0, len(uids), size=(200, len(uids)))]
        return {"a": ka, "b": kb, "n_persons": len(uids), "mad_a": round(float(pa.mean()), 4),
                "mad_b": round(float(pb.mean()), 4), "d_mad": round(float(diff.mean()), 4),
                "d_mad_ci": ci(diff[boot_idx(len(diff))].mean(1)), "item_r_a": round(item_r(db, "p"), 4),
                "item_r_b": round(item_r(db, "pb"), 4), "d_item_r": round(item_r(db, "p") - item_r(db, "pb"), 4),
                "d_item_r_ci": ci(rb)}

    # pre-specified blends (no fitted weight): mean of the classic and the LLM prediction
    for k in [k for k in arms if k.endswith("-persona") and k.startswith("cl-")] + ["ds:JSONPersona(PredictedOutput)|GPT4.1"]:
        if k in arms:
            m = arms[base].merge(arms[k], on=["uid", "oge"], suffixes=("_c", "_l"))
            arms[f"blend:{base}+{k}"] = m.assign(p=(m.p_c + m.p_l) / 2)[["uid", "oge", "p"]]
    rows = []
    for k, a in arms.items():
        d = te.merge(a, on=["uid", "oge"])
        if d.empty:
            continue
        r = pair(k, base)
        rows.append({"arm": k, "n_persons": int(d.uid.nunique()), "n_rows": len(d),
                     "mad_acc": round(float((1 - (d.p - d.y01).abs().groupby(d.uid).mean()).mean()), 4),
                     "item_r": round(item_r(d, "p"), 4), f"d_mad_vs_{base}": r["d_mad"], "d_mad_ci": r["d_mad_ci"],
                     f"d_item_r_vs_{base}": r["d_item_r"], "d_item_r_ci": r["d_item_r_ci"],
                     f"{base}_mad_same_rows": r["mad_b"]})
    extra = [(f"cl-{m}-persona", f"cl-{m}-yok") for m in ("h55", "s55")] + [
        ("cl-s55-persona", "cl-h55-persona"), ("ds:TextPersona|GPT4.1-mini", "ds:DemographicsOnly|GPT4.1-mini")]
    return rows, [pair(x, y) for x, y in extra if x in arms and y in arms]


# ------------------------------------------------------------------------------------------------- G3
def g3():
    from sklearn.metrics import roc_auc_score
    h = pd.read_parquet(VERI / "twin_fiyat/hedef.parquet")
    tr, te = h[h.bolum == "train"], h[h.bolum == "test"]
    truth = te[["uid", "oge", "carpan_duzey", "y", "slot"]]
    arms, sweeps = {}, {}
    cell = tr.groupby(["oge", "carpan_duzey"]).y.mean()
    arms["urun_fiyat_ort"] = truth.assign(p=[cell.get((o, c), np.nan) for o, c in zip(truth.oge, truth.carpan_duzey)])
    for f in sorted((BASE / "twin_fiyat/tahmin").glob("*_s0.parquet")):
        d = pd.read_parquet(f)
        k = f.stem[:-3]
        t = d[d.tur == "test"][["uid", "oge", "carpan_duzey", "p"]]
        arms[k] = truth.merge(t, on=["uid", "oge", "carpan_duzey"])
        sw = d[d.tur == "sweep"]
        if len(sw):
            sweeps[k] = sw
    for name, t in dataset_specs([f"QID9_{i}" for i in range(1, 41)]).items():
        t = t.rename(columns={"col": "slot"})
        arms[name] = truth.merge(t.assign(p=(t.v == 1).astype(float))[["uid", "slot", "p"]], on=["uid", "slot"])
    arms = {k: v.dropna(subset=["p"]) for k, v in arms.items()}
    base = "lgbm-tam"
    true_curve = truth.groupby("carpan_duzey").y.mean()

    def met(d):
        y, p = d.y.to_numpy(), d.p.to_numpy(float)
        cur = d.groupby("carpan_duzey").p.mean()
        return {"auc": roc_auc_score(y, p), "kisi_auc": lt.kisi_auc(p, y, d.uid.to_numpy()),
                "acc": float(np.mean((p >= 0.5) == (y == 1))), "brier": float(np.mean((p - y) ** 2)),
                "curve_r": float(np.corrcoef(cur, true_curve.loc[cur.index])[0, 1]),
                "slope": float(np.polyfit(cur.index.values.astype(float), cur.values, 1)[0]), "p_mean": float(p.mean())}

    def pair(ka, kb, hi=False):
        """Paired ka - kb; hi=True: only rows priced above the median (carpan > 1), the range the kos.py
        extrapolation arms (enc-tf, lgbm-tf: trained on carpan <= 1) never saw."""
        b = arms[ka].merge(arms[kb][["uid", "oge", "p"]].rename(columns={"p": "pb"}), on=["uid", "oge"])
        if hi:
            b = b[b.carpan_duzey > 1 + 1e-9]
        uids = np.array(sorted(b.uid.unique()))
        bi = b.set_index("uid")
        ka_ = pd.Series({u: lt.kisi_auc(g.p, g.y, g.uid) for u, g in b.groupby("uid")}).reindex(uids)
        kb_ = pd.Series({u: lt.kisi_auc(g.pb, g.y, g.uid) for u, g in b.groupby("uid")}).reindex(uids)
        kd = (ka_ - kb_).to_numpy()
        br = ((b.p - b.y) ** 2 - (b.pb - b.y) ** 2).groupby(b.uid).mean().reindex(uids).to_numpy()
        idx = boot_idx(len(uids))
        dd = {"kisi_auc": np.nanmean(kd[idx], 1), "brier": br[idx].mean(1), "auc": []}
        for r in idx[:300]:
            s_ = bi.loc[uids[r]]
            if s_.y.nunique() > 1:
                dd["auc"].append(roc_auc_score(s_.y, s_.p) - roc_auc_score(s_.y, s_.pb))
        ma, mb = met(b), met(b.assign(p=b.pb))
        out = {"a": ka, "b": kb, "rows": "carpan>1" if hi else "all", "n_persons": len(uids)}
        for x in ("auc", "kisi_auc", "brier"):
            out.update({f"{x}_a": round(ma[x], 4), f"{x}_b": round(mb[x], 4), f"d_{x}": round(ma[x] - mb[x], 4),
                        f"d_{x}_ci": ci(dd[x])})
        return out

    for k in [k for k in arms if k.endswith("-persona") and k.startswith("cl-")] + ["ds:JSONPersona|GPT4.1"]:
        if k in arms:
            m = arms[base].merge(arms[k][["uid", "oge", "p"]], on=["uid", "oge"], suffixes=("_c", "_l"))
            arms[f"blend:{base}+{k}"] = m.assign(p=(m.p_c + m.p_l) / 2).drop(columns=["p_c", "p_l"])
    rows = []
    for k, a in arms.items():
        if a.empty:
            continue
        m = met(a)
        r = pair(k, base)
        row = {"arm": k, "n_persons": int(a.uid.nunique()), "n_rows": len(a), **{x: round(v, 4) for x, v in m.items()}}
        ah = a[a.carpan_duzey > 1 + 1e-9]
        row.update(auc_hi=round(roc_auc_score(ah.y, ah.p), 4), kisi_auc_hi=round(lt.kisi_auc(ah.p, ah.y, ah.uid), 4),
                   brier_hi=round(float(np.mean((ah.p - ah.y) ** 2)), 4))
        for x in ("auc", "kisi_auc", "brier"):
            row[f"d_{x}_vs_{base}"] = r[f"d_{x}"]
            row[f"d_{x}_ci"] = r[f"d_{x}_ci"]
        if k in sweeps:
            s = lt.score_g3(sweeps[k], truth[["uid", "oge", "carpan_duzey", "y"]], B=B)
            row.update({"sweep_s_hi": round(s["s_hi"], 4), "sweep_s_true_hi": round(s["s_true_hi"], 4),
                        "sweep_ratio": round(s["ratio"], 3), "sweep_ratio_ci": [round(x, 3) for x in s["ratio_ci"]],
                        "sweep_curve_r": round(s["pearson_curve"], 4)})
        rows.append(row)
    extra = [(f"cl-{m}-{a}", f"cl-{m}-{b}") for m in ("h55", "s55") for a, b in
             (("persona", "yok"), ("gecmis", "yok"), ("persona", "gecmis"))] + [
        ("cl-s55-persona", "cl-h55-persona"), ("ds:TextPersona|GPT4.1-mini", "ds:DemographicsOnly|GPT4.1-mini")]
    hi = [(x, y) for x in ("cl-h55-persona", "cl-h55-gecmis", "cl-h55-yok", "cl-s55-persona", "cl-s55-yok",
                           "ds:JSONPersona|GPT4.1") for y in ("enc-tf", "lgbm-tf")]
    return rows, [pair(x, y) for x, y in extra if x in arms and y in arms] + [
        pair(x, y, hi=True) for x, y in hi if x in arms and y in arms]


# ------------------------------------------------------------------------------------------------- full-strength classics
def classic_full():
    """The strongest classic baselines of veri_hazirla.py (ozet.json), re-fit and written as prediction files so they
    are scored like every arm: ridge-tam (G2: RidgeCV per item on kisi_ozellik, alphas 10..10000) and lgbm-tam (G3:
    LightGBM product + carpan + history buy rates + skor_* + 5 demographics, trained on ALL price levels, test rows +
    the 11-level sweep). Unlike kos.py lgbm-tf / enc-tf they are not limited to carpan <= 1 in training."""
    import lightgbm as lgb
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    out2 = BASE / "twin/tahmin/ridge-tam_s0.parquet"
    if not out2.exists():
        h = pd.read_parquet(VERI / "twin/hedef.parquet")
        x = pd.read_parquet(VERI / "twin/kisi_ozellik.parquet")
        tr_idx = sorted(h[h.bolum == "train"].uid.unique())
        X = x.set_index("uid").astype(float)
        X = X.loc[:, X.loc[tr_idx].notna().mean() > 0.5]
        X = X.fillna(X.loc[tr_idx].median())
        X = X.loc[:, X.loc[tr_idx].std() > 0]
        Z = pd.DataFrame(StandardScaler().fit(X.loc[tr_idx]).transform(X), index=X.index)
        trh, te = h[h.bolum == "train"], h[h.bolum == "test"]
        parts = []
        for c in sorted(h.oge.unique()):
            a = trh[trh.oge == c]
            m = RidgeCV(alphas=[10, 100, 1000, 10000]).fit(Z.loc[a.uid], a.y01)
            b = te[te.oge == c]
            parts.append(b[["uid", "oge"]].assign(p=np.clip(m.predict(Z.loc[b.uid]), 0, 1)))
        pd.concat(parts).to_parquet(out2, index=False)
    out3 = BASE / "twin_fiyat/tahmin/lgbm-tam_s0.parquet"
    if not out3.exists():
        h = pd.read_parquet(VERI / "twin_fiyat/hedef.parquet")
        g = pd.read_parquet(VERI / "twin_fiyat/gecmis.parquet")
        grid = pd.read_parquet(VERI / "twin_fiyat/fiyat_izgara.parquet")
        x = pd.read_parquet(VERI / "twin/kisi_ozellik.parquet").set_index("uid")
        skor = x[[c for c in x.columns if c.startswith("skor_")] + ["QID11", "QID12", "QID13", "QID14", "QID15"]]
        gs = g.assign(ucuz=g.carpan < 1).groupby("uid").apply(
            lambda b: pd.Series({"g_al": b.al.mean(), "g_al_ucuz": b.al[b.ucuz].mean(),
                                 "g_al_pahali": b.al[~b.ucuz].mean()}), include_groups=False)
        prods = sorted(grid.oge.unique())
        cols = ["oge", "carpan"] + list(gs.columns) + list(skor.columns)

        def X(df):
            z = df[["uid", "oge", "carpan"]].merge(gs, left_on="uid", right_index=True, how="left") \
                .merge(skor, left_on="uid", right_index=True, how="left")
            z["oge"] = pd.Categorical(z["oge"], categories=prods)
            return z[cols]
        par = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1, seed=0)
        tr, te = h[h.bolum == "train"], h[h.bolum == "test"]
        m = tr.ic_dogrulama.values
        b = lgb.train(par, lgb.Dataset(X(tr[~m]), tr.y[~m]), 2000, valid_sets=[lgb.Dataset(X(tr[m]), tr.y[m])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        son = lgb.train(par, lgb.Dataset(X(tr), tr.y), b.best_iteration)
        t = te[["uid", "oge", "carpan_duzey", "carpan"]]
        sw = te[["uid", "oge"]].drop_duplicates().merge(grid[["oge", "carpan_duzey"]], on="oge")
        sw["carpan"] = sw.carpan_duzey
        pd.concat([t.assign(p=son.predict(X(t)), tur="test"), sw.assign(p=son.predict(X(sw)), tur="sweep")]
                  )[["uid", "oge", "carpan_duzey", "p", "tur"]].to_parquet(out3, index=False)


def main():
    classic_full()
    (r2, p2), (r3, p3) = g2(), g3()
    out = {"g2": r2, "g2_pairs": p2, "g3": r3, "g3_pairs": p3}
    d = BASE / "g2g3"
    d.mkdir(exist_ok=True)
    (d / "kiyas.json").write_text(json.dumps(out, indent=1, default=float))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for k, v in out.items():
        print(f"== {k}")
        print(pd.DataFrame(v).to_string(index=False))


if __name__ == "__main__":
    main()
