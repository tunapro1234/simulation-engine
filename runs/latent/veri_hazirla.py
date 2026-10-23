"""Leak-free task data for the latent-persona experiments (soft tokens / residual steering vs. LLM-free baselines).

    nice -n 19 /data/talp/venv/bin/python runs/latent/veri_hazirla.py              # all three tasks
    nice -n 19 /data/talp/venv/bin/python runs/latent/veri_hazirla.py --gorev twin  # one task

Tasks (output /data/talp/is/latent/veri/<gorev>/, parquet + README.md + ozet.json):
  talp73      person x Amazon category purchase, forward in time (TALP repo uretim/talp73/PROTOKOL.md V1-V8). Train people
              are outside sample s, features cut 2022-01-01, label 2022-H1 (as klasik.py); test = the 400 people of sample s,
              cut 2022-07-01, label 2022-H2 (panel etiket_s<s>). Seeds 0, 1, 2 in one hedef table (column tohum).
  twin        Twin-2K-500: persona from waves 1-3 (171 QIDs that are NOT asked again in wave 4), target = wave 4 answers
              to the 86 non-pricing items. People split 1658 train / 400 test (seed 0). gecmis = the same item answered
              in waves 1-3 (test-retest; the "history" arm, not the persona arm).
  twin_fiyat  Twin-2K-500 pricing study as a counterfactual check: 40 products, each person saw each product at a
              randomised price (11 levels, 0-2x the median). Target = wave 4 buy/no-buy for 10 held-out products;
              gecmis = the person's waves 1-3 answers on the other 30 products. Same people split as twin.
Rules: test people never appear in a train row; nothing after the cut enters gecmis; Twin persona text never contains a
wave 4 question. Reads the TALP repo (read-only) and /data/talp/veri; writes only under --cikti. Seeds fixed.
"""
import argparse
import glob
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "12")  # shared machine
import re
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

TALP = "/srv/lighthouse/hackathon-coz/projeler/talp-simulation-hackathon-2026/uretim"
for _d in ("intent/kod/eval", "talp73/kod/klasik", "talp73/kod/llm"):
    sys.path.insert(0, f"{TALP}/{_d}")
IS73 = "/data/talp/is/talp73"
TWIN = "/data/talp/veri/twin2k500"
CIKTI = "/data/talp/is/latent/veri"
TOHUMLAR = (0, 1, 2)
N_TEST = 400
AYLAR = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
         "December"]


def yaz(df, yol):
    (df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)).write_parquet(yol)
    return yol


def ic_dogrulama(uids, s):
    """klasik.lgbm's internal 20 % validation people: default_rng(100 + s) over sorted train uids."""
    ku = np.array(sorted(set(uids)))
    return set(np.random.default_rng(100 + s).choice(ku, int(0.2 * len(ku)), replace=False).tolist())


def ikili_olcu(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"auc": round(float(roc_auc_score(y, p)), 4), "brier": round(float(np.mean((p - y) ** 2)), 4),
            "logloss": round(float(log_loss(y, p)), 4), "dogruluk_0.5": round(float(np.mean((p >= 0.5) == y)), 4),
            "y_ort": round(float(y.mean()), 4), "p_ort": round(float(p.mean()), 4)}


# ----------------------------------------------------------------------------------------------------------- talp73
def aralik(bas, son):
    """'July 2021 - June 2022' for [bas, son)."""
    s = son - timedelta(days=1)
    return f"{AYLAR[bas.month - 1]} {bas.year} - {AYLAR[s.month - 1]} {s.year}"


def gecmis_metni(oz, kat, kesim):
    """L2 history block, same wording as talp73 kod/llm/kos.py gecmis_metinleri, for any cut."""
    a12 = aralik(date(kesim.year - 1, kesim.month, 1), kesim)
    out = []
    for (u,), d in oz.sort("uid").group_by("uid", maintain_order=True):
        r = {k: v for k, v in zip(d["kategori"], d.iter_rows(named=True))}
        sat = []
        for k in kat:
            x = r[k]
            sat.append(f"{k}: never purchased" if x["kalem"] == 0 else
                       f"{k}: {x['kalem']} items on {x['gun']} different days, last purchase {x['son_alim']}, "
                       f"{x['kalem_12ay']} items in {a12}")
        out.append((u, "\n".join(sat)))
    return pl.DataFrame(out, schema={"uid": pl.Utf8, "metin": pl.Utf8}, orient="row")


def ozet_l2(a, kat, uids, kesim):
    """panel.py gecmis_ozet columns (kalem, adet, gun, son_alim, kalem_12ay) at an arbitrary cut."""
    g = a.filter((pl.col("tarih") < kesim) & pl.col("kategori").is_in(kat) & pl.col("uid").is_in(uids))
    oz = g.group_by("uid", "kategori").agg(
        pl.len().alias("kalem"), pl.col("adet").fill_null(1).sum().alias("adet"), pl.col("tarih").n_unique().alias("gun"),
        pl.col("tarih").max().alias("son_alim"),
        (pl.col("tarih") >= date(kesim.year - 1, kesim.month, 1)).sum().alias("kalem_12ay"))
    iz = pl.DataFrame({"uid": sorted(uids)}).join(pl.DataFrame({"kategori": kat}), how="cross")
    return iz.join(oz, on=["uid", "kategori"], how="left").with_columns(
        *[pl.col(c).fill_null(0).cast(pl.Int32) for c in ("kalem", "gun", "kalem_12ay")], pl.col("adet").fill_null(0.0))


def talp73(cikti):
    import hazirla_oe as H
    import klasik as K
    import istem as I

    panel = f"{IS73}/panel"
    kat = pl.read_parquet(f"{panel}/kategoriler.parquet").sort("sira")["kategori"].to_list()
    a = H.oku_alimlar().select("uid", "tarih", "kategori", "fiyat", "adet", "baslik")
    k_eg, k_te = K.KESIM_EGITIM, K.KESIM_TEST
    uygun = sorted(H.son_alimlar(a).filter(pl.col("son_alim") >= k_te)["uid"].to_list())  # klasik kul_egitim_tum
    orn = {s: sorted(pl.read_parquet(f"{panel}/orneklem_s{s}.parquet")["uid"].to_list()) for s in TOHUMLAR}
    birlesik = sorted(set().union(*orn.values()))
    assert set(birlesik) <= set(uygun)

    # hedef: per seed, train = eligible minus sample (label 2022-H1, cut 2022-01-01), test = sample (panel labels)
    y_eg = K.etiket(a, kat, uygun, k_eg).rename({"kategori": "oge"})
    log = [json.loads(x) for x in open(f"{IS73}/tahmin/klasik_log.jsonl")]
    parca, kontrol = [], {}
    for s in TOHUMLAR:
        tr = sorted(set(uygun) - set(orn[s]))
        ic = ic_dogrulama(tr, s)
        e = (y_eg.filter(pl.col("uid").is_in(tr))
             .with_columns(pl.lit("train").alias("bolum"), pl.lit(k_eg).alias("kesim"), pl.col("uid").is_in(list(ic)).alias("ic_dogrulama")))
        t = (pl.read_parquet(f"{panel}/etiket_s{s}.parquet").rename({"kategori": "oge"})
             .with_columns(pl.col("y").cast(pl.Int8), pl.lit("test").alias("bolum"), pl.lit(k_te).alias("kesim"), pl.lit(False).alias("ic_dogrulama")))
        assert t.height == N_TEST * len(kat) and not set(e["uid"]) & set(t["uid"])
        parca.append(pl.concat([e.select(t.columns), t]).with_columns(pl.lit(s, pl.Int8).alias("tohum")))
        ref = [r for r in log if r.get("yontem") == "anket" and r.get("tohum") == s and "egitim_y_ort" in r]
        kontrol[f"s{s}"] = {"train_kisi": len(tr), "train_y_ort": round(float(e["y"].mean()), 4),
                            "klasik_log_egitim_y_ort": ref[-1]["egitim_y_ort"] if ref else None,
                            "klasik_log_egitim_kisi": ref[-1].get("egitim_kisi") if ref else None,
                            "test_y_ort": round(float(t["y"].mean()), 4)}
    hedef = pl.concat(parca).select("tohum", "uid", "oge", "y", "bolum", "kesim", "ic_dogrulama").sort("tohum", "bolum", "uid", "oge")
    yaz(hedef, f"{cikti}/hedef.parquet")

    # kisi_ozellik / kisi_metin: survey answers (filled 2023-Q1, V7) for every person in hedef
    an = pl.read_parquet(f"{panel}/anket.parquet").select(["uid"] + K.ANKET_SUTUN).filter(pl.col("uid").is_in(uygun))
    yaz(an.sort("uid"), f"{cikti}/kisi_ozellik.parquet")
    S = I.anket()
    km = pl.DataFrame({"uid": uygun, "metin": [I.profil(S.loc[u]) for u in uygun]})
    yaz(km, f"{cikti}/kisi_metin.parquet")

    # gecmis: (uid, kesim) pairs = eligible at the train cut, sample union at the test cut
    roller = [(uygun, k_eg), (birlesik, k_te)]
    oz_l, gm_l, ol_l, ozl2 = [], [], [], {}
    for uids, c in roller:
        f = K.gecmis_ozellik(a, kat, uids, c).with_columns(pl.lit(c).alias("kesim"))
        oz_l.append(f)
        o2 = ozet_l2(a, kat, uids, c)
        ozl2[c] = o2
        oz_l[-1] = f.join(o2.select("uid", "kategori", "kalem", "adet", "gun", "son_alim", "kalem_12ay"), on=["uid", "kategori"])
        gm_l.append(gecmis_metni(o2, kat, c).with_columns(pl.lit(c).alias("kesim"), pl.lit(aralik(date(2018, 1, 1), c)).alias("aralik")))
        ol_l.append(a.filter(pl.col("uid").is_in(uids) & (pl.col("tarih") < c)).with_columns(pl.lit(c).alias("kesim")))
    # the test-cut summary must equal the panel's L2 source (gecmis_ozet.parquet) for the 996 sample people
    p = pl.read_parquet(f"{panel}/gecmis_ozet.parquet").filter(pl.col("uid").is_in(birlesik)).sort("uid", "kategori")
    q = ozl2[k_te].sort("uid", "kategori")
    for c in ("kalem", "gun", "kalem_12ay", "son_alim"):
        assert (p[c] == q[c]).fill_null(True).all() and (p[c].is_null() == q[c].is_null()).all(), c
    gecmis = pl.concat(oz_l, how="diagonal_relaxed").rename({"kategori": "oge"})
    yaz(gecmis.select(["uid", "kesim", "oge"] + [c for c in gecmis.columns if c not in ("uid", "kesim", "oge")]),
        f"{cikti}/gecmis.parquet")
    yaz(pl.concat(gm_l).select("uid", "kesim", "aralik", "metin"), f"{cikti}/gecmis_metin.parquet")
    olay = pl.concat(ol_l).select("uid", "kesim", "tarih", "kategori", "fiyat", "adet", "baslik").sort("kesim", "uid", "tarih")
    yaz(olay, f"{cikti}/gecmis_olay.parquet")
    pk = pl.read_parquet(f"{panel}/kategoriler.parquet").sort("sira")
    yaz(pk.rename({"kategori": "oge"}).with_columns(pl.col("oge").map_elements(I.etiket, return_dtype=pl.Utf8).alias("istem_adi")),
        f"{cikti}/oge.parquet")

    # baselines from the existing talp73 prediction / result files (same 400 x 20 rows per seed)
    yontem = ["sabit", "populerlik", "anket", "anket_ey", "kalicilik", "sonumlu", "gecmis", "gecmis_anket", "gecmis_anket_ey",
              "L0-haiku55", "L1-haiku55", "L1-sonnet55", "L2-haiku55", "L2-sonnet55", "H2cf-haiku55"]
    olcu = ["1a_dogruluk_a", "1b_dogruluk_c", "4_kisi_auc", "2_precision@3"]
    taban = {}
    for m in yontem:
        sat = {}
        for s in TOHUMLAR:
            yol = f"{IS73}/tahmin/{m}_s{s}.parquet"
            if not os.path.exists(yol):
                continue
            t = pl.read_parquet(yol).join(pl.read_parquet(f"{panel}/etiket_s{s}.parquet"), on=["uid", "kategori"])
            assert t.height == N_TEST * len(kat)
            r = {} if m == "sabit" else ikili_olcu(t["y"], t["p"])
            js = f"{IS73}/sonuc/{m}_s{s}.json"
            if os.path.exists(js):
                o = json.load(open(js))["olcu"]
                r.update({k: round(o[k]["deger"] if isinstance(o[k], dict) else o[k], 4) for k in olcu if k in o})
            sat[f"s{s}"] = r
        if sat:
            anahtar = sorted(set().union(*[set(v) for v in sat.values()]))
            sat["ort"] = {k: round(float(np.mean([v[k] for v in sat.values() if k in v])), 4) for k in anahtar}
            taban[m] = sat
    ozet = {"gorev": "talp73", "kategori": len(kat), "uygun_kisi": len(uygun), "test_birlesik_kisi": len(birlesik),
            "hedef_satir": {f"s{s}": {b: hedef.filter((pl.col("tohum") == s) & (pl.col("bolum") == b)).height for b in ("train", "test")}
                            for s in TOHUMLAR},
            "kontrol": kontrol, "gecmis_satir": gecmis.height, "gecmis_olay_satir": olay.height, "taban": taban}
    return ozet


# ------------------------------------------------------------------------------------------------------------ twin
def twin_bolme():
    w = pd.read_csv(f"{TWIN}/question_catalog_and_human_response_csv/wave4_response.csv")
    pids = sorted(w["pid"].astype(int).tolist())
    test = set(np.random.default_rng(0).choice(pids, N_TEST, replace=False).tolist())
    tr = [p for p in pids if p not in test]
    ic = ic_dogrulama(tr, 0)
    return pids, test, ic


def twin_split_json():
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{TWIN}/wave_split/chunks/*.parquet"))])
    d["pid"] = d["pid"].astype(int)
    return d.set_index("pid").sort_index()


def sorular(blok_json):
    return [q for b in json.loads(blok_json) for q in b.get("Questions", [])]


def demografi_metni(js):
    out = []
    for b in json.loads(js):
        if b["BlockName"].strip() != "Demographics":
            continue
        for q in b["Questions"]:
            a = (q.get("Answers") or {}).get("SelectedText")
            if a is not None:
                out.append(f"- {q['QuestionText'].strip()}: {a}")
    return "\n".join(out)


def twin_kisi(cikti, d, pids):
    """kisi_ozellik (waves 1-3 numeric answers minus every wave 4 column, plus construct scores) and kisi_metin."""
    w13 = pd.read_csv(f"{TWIN}/question_catalog_and_human_response_csv/wave1_3_response.csv", low_memory=False)
    w4 = pd.read_csv(f"{TWIN}/question_catalog_and_human_response_csv/wave4_response.csv", nrows=1)
    sut = [c for c in w13.columns if c != "pid" and c not in w4.columns and pd.api.types.is_numeric_dtype(w13[c])]
    x = w13[["pid"] + sut].copy()
    for w in (1, 2, 3):
        s = pd.read_csv(f"{TWIN}/raw_data/wave {w} scores.csv").rename(columns={"TWIN_ID": "pid"})
        s = s.set_index("pid").apply(pd.to_numeric, errors="coerce")
        s.columns = [f"skor_{c.replace('score_', '')}_d{w}" for c in s.columns]
        x = x.merge(s.reset_index(), on="pid", how="left")
    x = x.rename(columns={"pid": "uid"}).sort_values("uid")
    yaz(x, f"{cikti}/kisi_ozellik.parquet")
    for js in d["wave1_3_persona_json"]:
        assert not {q["QuestionID"] for q in sorular(js)} & set(w4.columns), "wave 4 item in the persona"
    km = pd.DataFrame({"uid": d.index, "metin": d["wave1_3_persona_text"].values,
                       "metin_demografi": [demografi_metni(js) for js in d["wave1_3_persona_json"]]})
    yaz(km, f"{cikti}/kisi_metin.parquet")
    return x, km, sut


def katalog_satirlari():
    """csv column -> (QID, block, type, text, n_options or slider range)."""
    out = {}
    for q in json.load(open(f"{TWIN}/question_catalog_and_human_response_csv/question_catalog.json")):
        for j, c in enumerate(q.get("csv_columns") or []):
            tip = q["QuestionType"]
            alt = ""
            if tip == "Matrix" and q.get("Rows"):
                alt = q["Rows"][j] if j < len(q["Rows"]) else ""
                n, lo, hi = len(q["Columns"]), 1, len(q["Columns"])
            elif tip == "Slider":
                alt = (q.get("Statements") or [""] * (j + 1))[j] if q.get("Statements") else ""
                n, lo, hi = None, q["Range"]["Min"], q["Range"]["Max"]
            elif tip == "MC":
                n, lo, hi = len(q["Options"]), 1, len(q["Options"])
            else:
                n, lo, hi = None, None, None
            out[c] = {"qid": q["QuestionID"], "blok": q.get("BlockName", "").strip(), "tip": tip,
                      "soru": (q.get("QuestionText") or "").strip(), "alt": alt, "n_secenek": n, "alt_sinir": lo, "ust_sinir": hi,
                      "secenekler": json.dumps(q.get("Options") or q.get("Columns") or [], ensure_ascii=False)}
    return out


def twin(cikti):
    pids, test, ic = twin_bolme()
    d = twin_split_json()
    assert sorted(d.index) == pids
    x, km, sut = twin_kisi(cikti, d, pids)
    w4 = pd.read_csv(f"{TWIN}/question_catalog_and_human_response_csv/wave4_response.csv").set_index("pid")
    w13 = pd.read_csv(f"{TWIN}/question_catalog_and_human_response_csv/wave1_3_response.csv", low_memory=False).set_index("pid")
    kat = katalog_satirlari()
    oge = [c for c in w4.columns if not c.startswith("QID9_")]
    tr_idx = [p for p in pids if p not in test]
    aile = lambda c: re.sub(r"\s*-?\s*(high|low|no|yes)$", "", kat[c]["blok"])  # noqa: E731  condition pair of a block
    rows = []
    for c in oge:
        k = kat[c]
        lo, hi = k["alt_sinir"], k["ust_sinir"]
        if lo is None:  # TE number: train people's waves 1-3 + wave 4 1st/99th pct, pooled over the condition pair
            es = [o for o in oge if kat[o]["tip"] == "TE" and aile(o) == aile(c)]
            v = pd.concat([w4.loc[tr_idx, es].stack(), w13.loc[tr_idx, es].stack()]).astype(float).dropna()
            lo, hi = float(v.quantile(0.01)), float(v.quantile(0.99))
        rows.append({"oge": c, **k, "kosul_ailesi": aile(c), "olcek_alt": float(lo), "olcek_ust": float(hi)})
    og = pd.DataFrame(rows)
    yaz(og, f"{cikti}/oge.parquet")
    sc = og.set_index("oge")[["olcek_alt", "olcek_ust"]]

    def uzun(tab, ad):
        t = tab[oge].stack().rename(ad).reset_index().rename(columns={"level_1": "oge", "pid": "uid"})
        t[ad] = t[ad].astype(float)
        lo, hi = sc.loc[t["oge"], "olcek_alt"].values, sc.loc[t["oge"], "olcek_ust"].values
        t[ad + "01"] = np.clip((t[ad] - lo) / (hi - lo), 0, 1)
        return t

    h = uzun(w4, "y")
    h["bolum"] = np.where(h["uid"].isin(test), "test", "train")
    h["ic_dogrulama"] = h["uid"].isin(ic)
    yaz(h.sort_values(["bolum", "uid", "oge"]), f"{cikti}/hedef.parquet")
    g = uzun(w13.loc[:, oge], "y_d13")
    yaz(g.sort_values(["uid", "oge"]), f"{cikti}/gecmis.parquet")

    # baselines on the 400 test people: item mean, waves 1-3 retest, ridge on kisi_ozellik, GPT-4.1-mini (paper's run)
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    te = h[h.bolum == "test"].copy()
    trh = h[h.bolum == "train"]
    te["madde_ort"] = te["oge"].map(trh.groupby("oge")["y01"].mean())
    te = te.merge(g[["uid", "oge", "y_d1301"]], on=["uid", "oge"], how="left")
    X = x.set_index("uid").astype(float)
    X = X.loc[:, X.loc[tr_idx].notna().mean() > 0.5]
    X = X.fillna(X.loc[tr_idx].median())
    X = X.loc[:, X.loc[tr_idx].std() > 0]
    Z = pd.DataFrame(StandardScaler().fit(X.loc[tr_idx]).transform(X), index=X.index)
    tahmin = {}
    for c in oge:
        a = trh[trh.oge == c]
        m = RidgeCV(alphas=[10, 100, 1000, 10000]).fit(Z.loc[a.uid], a.y01)
        b = te[te.oge == c]
        tahmin.update(dict(zip(zip(b.uid, b.oge), np.clip(m.predict(Z.loc[b.uid]), 0, 1))))
    te["ridge_persona"] = [tahmin[(u, o)] for u, o in zip(te.uid, te.oge)]
    llm = gpt41mini_twin(oge, sc)
    te = te.merge(llm, on=["uid", "oge"], how="left")
    taban = {}
    for m in ("madde_ort", "ridge_persona", "gpt41mini_persona", "y_d1301"):
        ok = te[m].notna()
        e = te[ok]
        kor = [np.corrcoef(b["y01"], b[m])[0, 1] for _, b in e.groupby("oge") if b["y01"].std() > 1e-9 and b[m].std() > 1e-9]
        taban[m] = {"satir": int(ok.sum()), "mad_dogruluk": round(float(1 - np.mean(np.abs(e[m] - e["y01"]))), 4),
                    "madde_ici_pearson_ort": round(float(np.mean(kor)), 4) if kor else None, "madde_sayisi_kor": len(kor)}
    ozet = {"gorev": "twin", "kisi": len(pids), "test_kisi": len(test), "train_kisi": len(tr_idx), "ic_dogrulama_kisi": len(ic),
            "oge": len(oge), "ozellik_sutun": X.shape[1], "hedef_satir": {b: int((h.bolum == b).sum()) for b in ("train", "test")},
            "metin_karakter_ort": int(km["metin"].str.len().mean()), "taban": taban}
    return ozet


def gpt41mini_twin(sutunlar, sc=None):
    """The dataset's own GPT-4.1-mini wave 4 simulation (persona = waves 1-3), long format (uid, oge, value)."""
    m = json.load(open(f"{TWIN}/LLM_simulation_results/wave4_formatted_to_catalog_mapping.json"))
    l = pd.read_csv(f"{TWIN}/LLM_simulation_results/GPT4.1-mini-simulation-llm-vs-human/responses_llm_imputed_formatted.csv",
                    low_memory=False)
    l = l[pd.to_numeric(l["TWIN_ID"], errors="coerce").notna()].copy()
    l["uid"] = l["TWIN_ID"].astype(int)
    es = {e["catalog_csv_column"]: e["formatted_column"] for e in m}
    t = pd.DataFrame({c: pd.to_numeric(l[es[c]], errors="coerce").values for c in sutunlar if c in es}, index=l["uid"])
    t = t.stack().rename("v").reset_index().rename(columns={"level_1": "oge"})
    if sc is not None:
        lo, hi = sc.loc[t["oge"], "olcek_alt"].values, sc.loc[t["oge"], "olcek_ust"].values
        t["gpt41mini_persona"] = np.clip((t["v"] - lo) / (hi - lo), 0, 1)
        return t[["uid", "oge", "gpt41mini_persona"]]
    return t


# ----------------------------------------------------------------------------------------------------- twin_fiyat
FIYAT_RX = re.compile(r"product category:\s*(.*?)\.\s*Suppose.*?following product in that category:\s*(.*?)\s*"
                      r"The product is priced at:?\s*\$?([\d.,]+?)\.?\s*Would", re.S)


def fiyat_satirlari(d):
    rows = []
    for pid, r in d.iterrows():
        for col, dalga in (("wave4_Q_wave1_3_A", "d13"), ("wave4_Q_wave4_A", "d4")):
            for q in sorular(r[col]):
                if q["QuestionID"].startswith("QID9_"):
                    m = FIYAT_RX.search(q["QuestionText"])
                    a = (q.get("Answers") or {}).get("SelectedByPosition")
                    rows.append((pid, dalga, q["QuestionID"], m.group(1).strip(), m.group(2).strip().rstrip("."),
                                 float(m.group(3).replace(",", "")), None if a is None else int(a == 1)))
    return pd.DataFrame(rows, columns=["uid", "dalga", "slot", "kategori", "urun", "fiyat", "al"])


def twin_fiyat(cikti):
    import lightgbm as lgb
    pids, test, ic = twin_bolme()
    d = twin_split_json()
    t = fiyat_satirlari(d)
    assert t["al"].notna().all() and t["urun"].nunique() == 40
    p = t.pivot_table(index=["uid", "slot"], columns="dalga", values=["urun", "fiyat"], aggfunc="first")
    assert (p[("urun", "d13")] == p[("urun", "d4")]).all() and (p[("fiyat", "d13")] == p[("fiyat", "d4")]).all()
    urun = sorted(t["urun"].unique())
    og = pd.DataFrame({"oge": [f"P{i + 1:02d}" for i in range(40)], "urun": urun})
    og = og.merge(t[t.dalga == "d13"].groupby("urun").agg(kategori=("kategori", "first"), medyan_fiyat=("fiyat", "median"),
                                                          fiyat_duzey=("fiyat", "nunique")).reset_index(), on="urun")
    hedef_oge = set(np.random.default_rng(0).choice(og["oge"], 10, replace=False).tolist())
    og["rol"] = np.where(og["oge"].isin(hedef_oge), "hedef", "gecmis")
    yaz(og, f"{cikti}/oge.parquet")
    t = t.merge(og[["oge", "urun", "medyan_fiyat", "rol"]], on="urun")
    t["carpan"] = (t["fiyat"] / t["medyan_fiyat"]).round(2)
    t["carpan_duzey"] = (t["carpan"] * 5).round() / 5
    h = t[(t.dalga == "d4") & (t.rol == "hedef")].copy()
    h["y"] = h["al"]
    h["bolum"] = np.where(h["uid"].isin(test), "test", "train")
    h["ic_dogrulama"] = h["uid"].isin(ic)
    hc = ["uid", "oge", "y", "bolum", "ic_dogrulama", "fiyat", "carpan", "carpan_duzey", "slot"]
    yaz(h[hc].sort_values(["bolum", "uid", "oge"]), f"{cikti}/hedef.parquet")
    g = t[(t.dalga == "d13") & (t.rol == "gecmis")][["uid", "oge", "al", "fiyat", "carpan", "carpan_duzey", "slot"]]
    yaz(g.sort_values(["uid", "oge"]), f"{cikti}/gecmis.parquet")
    iz = og[og.rol == "hedef"][["oge", "medyan_fiyat"]].merge(pd.DataFrame({"carpan_duzey": sorted(t["carpan_duzey"].unique())}), how="cross")
    iz["fiyat"] = (iz["medyan_fiyat"] * iz["carpan_duzey"]).round(2)
    yaz(iz, f"{cikti}/fiyat_izgara.parquet")
    # person features shared with twin (same files); history summaries for the baselines
    x = pl.read_parquet(f"{os.path.dirname(cikti)}/twin/kisi_ozellik.parquet").to_pandas().set_index("uid")  # run twin first
    skor = x[[c for c in x.columns if c.startswith("skor_")] + ["QID11", "QID12", "QID13", "QID14", "QID15"]]
    gs = g.assign(ucuz=g.carpan < 1).groupby("uid").apply(
        lambda b: pd.Series({"g_al": b.al.mean(), "g_al_ucuz": b.al[b.ucuz].mean(), "g_al_pahali": b.al[~b.ucuz].mean()}),
        include_groups=False)
    tr = h[h.bolum == "train"]
    te = h[h.bolum == "test"].copy()
    te["urun_ort"] = te.oge.map(tr.groupby("oge").y.mean())
    cell = tr.groupby(["oge", "carpan_duzey"]).y.mean()
    te["urun_fiyat_ort"] = [cell.get((o, c), np.nan) for o, c in zip(te.oge, te.carpan_duzey)]

    def lgbm(cols, ad):
        def X(df):
            z = df[["uid", "oge", "carpan"]].merge(gs, left_on="uid", right_index=True, how="left") \
                .merge(skor, left_on="uid", right_index=True, how="left")
            z["oge"] = pd.Categorical(z["oge"], categories=sorted(hedef_oge))
            return z[cols]
        par = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1, seed=0)
        m = tr.ic_dogrulama.values
        b = lgb.train(par, lgb.Dataset(X(tr[~m]), tr.y[~m]), 2000, valid_sets=[lgb.Dataset(X(tr[m]), tr.y[m])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        son = lgb.train(par, lgb.Dataset(X(tr), tr.y), b.best_iteration)
        te[ad] = son.predict(X(te))
        # counterfactual sweep: every test person x target product at every price level
        cf = te[["uid", "oge"]].drop_duplicates().merge(iz[["oge", "carpan_duzey"]], on="oge").rename(columns={"carpan_duzey": "carpan"})
        cf["p"] = son.predict(X(cf))
        return cf.groupby("carpan").p.mean()

    temel = ["oge", "carpan"]
    cf = {"lgbm_fiyat": lgbm(temel, "lgbm_fiyat"),
          "lgbm_persona": lgbm(temel + list(skor.columns), "lgbm_persona"),
          "lgbm_gecmis": lgbm(temel + list(gs.columns), "lgbm_gecmis"),
          "lgbm_gecmis_persona": lgbm(temel + list(gs.columns) + list(skor.columns), "lgbm_gecmis_persona")}
    llm = gpt41mini_twin([f"QID9_{i}" for i in range(1, 41)])
    llm = llm.rename(columns={"oge": "slot"})
    llm["gpt41mini_persona"] = (llm["v"] == 1).astype(float)
    te = te.merge(llm[["uid", "slot", "gpt41mini_persona"]], on=["uid", "slot"], how="left")
    gercek = te.groupby("carpan_duzey").y.mean()
    egim = lambda s: float(np.polyfit(s.index.values.astype(float), s.values, 1)[0])  # noqa: E731
    taban = {}
    for m in ("urun_ort", "urun_fiyat_ort", "lgbm_fiyat", "lgbm_persona", "lgbm_gecmis", "lgbm_gecmis_persona", "gpt41mini_persona"):
        e = te[te[m].notna()]
        r = ikili_olcu(e.y, e[m])
        r["kisi_ici_auc"] = round(float(np.mean([roc_auc_score(b.y, b[m]) for _, b in e.groupby("uid") if 0 < b.y.sum() < len(b)])), 4)
        tah = e.groupby("carpan_duzey")[m].mean()
        r["talep_egrisi_pearson"] = round(float(np.corrcoef(tah, gercek.loc[tah.index])[0, 1]), 4)
        r["talep_egimi_gozlenen"] = round(egim(tah), 4)
        if m in cf:
            r["talep_egimi_cf_tarama"] = round(egim(cf[m]), 4)
        taban[m] = r
    tekrar = t.pivot_table(index=["uid", "oge"], columns="dalga", values="al")
    tekrar = tekrar.loc[[(u, o) for u, o in tekrar.index if u in test and o in hedef_oge]]
    ozet = {"gorev": "twin_fiyat", "kisi": len(pids), "test_kisi": len(test), "hedef_urun": sorted(hedef_oge), "gecmis_urun": 30,
            "fiyat_duzey": int(og.fiyat_duzey.max()), "carpan_duzeyleri": sorted(map(float, t.carpan_duzey.unique())),
            "hedef_satir": {b: int((h.bolum == b).sum()) for b in ("train", "test")}, "gecmis_satir": len(g),
            "test_y_ort": round(float(te.y.mean()), 4), "gercek_talep_egrisi": {str(k): round(v, 4) for k, v in gercek.items()},
            "gercek_talep_egimi": round(egim(gercek), 4),
            "insan_tekrar_uyumu_ust_sinir": round(float((tekrar["d13"] == tekrar["d4"]).mean()), 4), "taban": taban}
    return ozet


GOREVLER = {"talp73": talp73, "twin": twin, "twin_fiyat": twin_fiyat}

METIN = {
    "talp73": """# G1 talp73: kişi × kategori alımı, zamanda ileri

Kaynak: Open e-commerce 1.0 (TALP repo `uretim/talp73/PROTOKOL.md` V1-V8, panel `/data/talp/is/talp73/panel`). Birim kişi × 20
Amazon kategorisi; y = pencerede o kategoriden ≥1 kalem. Tohum 0/1/2 (`hedef.tohum`), her tohumda test = örneklemin 400 kişisi.

## Dosyalar
- `hedef.parquet` tohum, uid, oge (= kategori), y, bolum, kesim, ic_dogrulama. train: kişiler 2022-07-01'de tam gözlenen
  (son alım ≥ 2022-07-01) ve örneklem s DIŞINDA; kesim 2022-01-01, y = 2022-01-01 → 2022-06-30 (181 gün, klasik.py `etiket`).
  test: örneklem s; kesim 2022-07-01, y = panel `etiket_s<s>` (2022-07-01 → 2022-12-31, değerlendiricinin etiketi).
  `ic_dogrulama` = klasik.py'nin erken durdurma kişileri (default_rng(100+s), train kişilerin %20'si).
- `kisi_ozellik.parquet` uid + 32 anket sütunu (klasik.py ANKET_SUTUN: demografi + beyanlar + hayat olayı bayrakları), kategorik metin.
- `kisi_metin.parquet` uid, metin = L1 persona profili (`kod/llm/istem.py profil()`, LLM kollarının gördüğü blok, birebir).
- `gecmis.parquet` (uid, kesim) × kategori: klasik.py `gecmis_ozellik` sütunları (k_30..k_730, son_90/180/365 sönümlü sıklık,
  pay_365, u_* kullanıcı toplamları, harcama) + L2 özeti (kalem, adet, gun, son_alim, kalem_12ay). Kesimden SONRASI yok.
- `gecmis_metin.parquet` (uid, kesim, aralik, metin): L2 istemindeki geçmiş bloğu (kos.py `gecmis_metinleri` biçimi); test
  kesiminde panel `gecmis_ozet` ile birebir aynı olduğu betikte assert ediliyor.
- `gecmis_olay.parquet` (uid, kesim, tarih, kategori, fiyat, adet, baslik): kesimden önceki BÜTÜN kalemler (bütün kategoriler,
  2018'den), dizi/metin kodlayıcıları için.
- `oge.parquet` 20 kategori, okunur ad, istem adı.

## Sızıntı kuralları
- Persona vektörü/kodlayıcı/steering yalnız `bolum == "train"` satırlarıyla ve aynı tohumun train kişileriyle eğitilir; test
  kişisi ve 2022-H2 penceresi eğitimde YOK (betik kesişimi assert ediyor). Hiperparametre seçimi `ic_dogrulama` kişileriyle.
- Bir kişi bir tohumda test, başka tohumda train olabilir; bu yüzden `gecmis*` tabloları (uid, kesim) anahtarlı: train satırı
  kesim 2022-01-01 geçmişiyle, test satırı kesim 2022-07-01 geçmişiyle eşlenir. Tohumlar arası karıştırma yok.
- V7: anket 2023-Q1'de doldurulmuş (test penceresinden SONRA). `siklik_beyan` gibi beyanlar pencere sonrası öz-beyan; persona/anket
  kollarının lehine, geçmiş kolunu etkilemez. V8: train etiketi 2022-H1 (mevsim farkı, klasik tabanların aleyhine).
- Tahmin dosyası şeması talp73 ile aynı tutulursa (`tahmin/<yontem>_s<s>.parquet`: uid, kategori, p; 8.000 satır)
  `uretim/talp73/kod/degerlendir.py` doğrudan değerlendirir.

## Karşı-olgusal (G3) notu
Bu veride fiyat yalnız ödenen birim fiyat olarak var (`gecmis_olay.fiyat`); kampanya/indirim alanı, gösterilen-ama-alınmayan ürün
ya da fiyat değişimi deneyi YOK. Fiyat içsel (kişi neyi hangi fiyata aldıysa o), karşı-olgusal fiyat testi kurulamaz. G3 bu yüzden
Twin-2K-500 fiyat çalışmasında (`../twin_fiyat`), fiyatı rastgele atanmış.
""",
    "twin": """# G2 twin: anketin bir kısmından persona, kalan sorular

Kaynak: Twin-2K-500 (`/data/talp/veri/twin2k500`, CC-BY-4.0), 2.058 ABD'li. Dalga 1-3: demografi, kişilik, bilişsel testler,
ekonomik tercihler + H&B deneyleri; dalga 4 (2 hafta sonra) H&B deneylerini ve fiyat çalışmasını TEKRARLAR.

## Kurgu
- Persona girdisi = dalga 1-3'te sorulup dalga 4'te SORULMAYAN 171 QID (veri setinin `wave1_3_persona_*`, `wave_split` kurgusu).
- Hedef = dalga 4'teki 86 fiyat-dışı madde (`QID9_*` fiyat maddeleri `../twin_fiyat`ta). y ham kod, y01 = [0,1] ölçek:
  MC/Matrix (konum-1)/(k-1), Slider (v-min)/(max-min), TE sayı: train kişilerin 1-99. yüzdeliği (koşul çifti birlikte, ör. anchoring
  high+low aynı ölçek) ile kırpılıp ölçeklenir.
- Kişi ayrımı: default_rng(0) ile 400 test, 1.658 train; `ic_dogrulama` = train'in %20'si (default_rng(100)).
- Kişi başına ~56 hedef satırı: gruplar-arası deneylerde kişi tek koşul görür (dalga 1-3 ile dalga 4'te aynı koşul).

## Dosyalar
- `kisi_ozellik.parquet` uid + dalga 1-3 sayısal cevaplar (dalga 4 sütunları HARİÇ) + 3 dalganın yapı skorları (`skor_*_d<dalga>`;
  hiçbirinin dalga 4 maddelerinden türemediği ridge R² ile kontrol edildi, en yüksek 0,27).
- `kisi_metin.parquet` uid, metin (`wave1_3_persona_text`, ort. ~95 bin karakter ≈ 25 bin token), metin_demografi (14 demografi sorusu, kısa).
  `full_persona` klasörünün özeti/metni dalga 4 cevaplarını İÇERİR, kullanılmadı.
- `gecmis.parquet` (uid, oge, y_d13, y_d1301): AYNI maddenin dalga 1-3 cevabı (test-tekrar). talp73'teki "geçmiş"in karşılığı;
  persona-yalnız kollar KULLANMAZ. İnsan test-tekrar tavanı olarak da okunur.
- `oge.parquet` madde metni, alt satır (matris satırı/slider ifadesi), seçenekler, blok (= deney koşulu), kosul_ailesi, ölçek.

## Sızıntı kuralları
- Persona/kodlayıcı/steering eğitimi yalnız train kişileri; test kişisinin hiçbir dalga 4 cevabı eğitimde yok.
- Persona kolunda `gecmis` YOK; geçmişli kol ayrı satır olarak raporlanır.
- Karşı-olgusal yan kontrol: `blok`ta rastgele atanmış koşul çiftleri var (anchoring high/low, disease gain/loss, Linda conjunction,
  outcome bias success/failure, sunk cost yes/no, base-rate 30/70 ...). Grup farkı nedensel; model koşul etkisini koruyor mu ölçülebilir.
""",
    "twin_fiyat": """# G3 twin_fiyat: karşı-olgusal fiyat kontrolü

Kaynak: Twin-2K-500 fiyat çalışması. 40 market ürünü; her kişi her ürünü kişiye özel RASTGELE bir fiyatta görür (11 düzey: medyan
fiyatın 0; 0,2; ...; 2,0 katı) ve "alırım / almam" der. Ürün sırası kişiye göre karışık (`slot`), fiyat ve ürün dalga 1-3 ile dalga
4'te aynı (assert). Fiyat rastgele atandığı için gözlenen talep eğrisi nedensel: modelin fiyat duyarlılığı buna karşı ölçülür.

## Kurgu
- Ürünler P01-P40 (ada göre sıralı); default_rng(0) ile 10 ürün `rol = hedef`, 30 ürün `rol = gecmis`.
- `hedef.parquet` uid, oge, y (dalga 4 alır = 1), fiyat, carpan (= fiyat / ürün medyanı), carpan_duzey, slot, bolum, ic_dogrulama.
  Kişi ayrımı twin ile aynı (400 test).
- `gecmis.parquet` kişinin dalga 1-3'te 30 geçmiş ürününe verdiği cevaplar + gördüğü fiyat (kişisel fiyat duyarlılığı buradan).
  Hedef ürünlerin dalga 1-3 cevabı (aynı fiyatta test-tekrar, uyum ~%83) GİRDİ DEĞİL; yalnız insan tavanı olarak `ozet.json`da.
- `fiyat_izgara.parquet` 10 hedef ürün × 11 fiyat düzeyi: karşı-olgusal tarama için (her test kişisi × ürün × fiyat sorulur,
  ortalama talep eğrisi gerçek eğriyle karşılaştırılır).
- Persona girdileri `../twin/kisi_ozellik.parquet` ve `../twin/kisi_metin.parquet` (aynı kişiler; fiyat maddeleri orada yok).

## Ölçüler (ozet.json taban)
auc, brier, logloss, dogruluk_0.5, kisi_ici_auc; talep_egrisi_pearson (11 düzeyde ortalama tahmin vs gerçek alma oranı),
talep_egimi_gozlenen (tahmin edilen alma oranının carpan'a doğrusal eğimi; gerçek eğim `gercek_talep_egimi`),
talep_egimi_cf_tarama (aynı kişi × ürün bütün fiyatlarda sorulunca eğim). LLM içine persona sokulunca bu eğim korunmalı.
""",
}


def readme(g, oz, d):
    sat = [METIN[g], "## Sayılar\n", "| dosya | satır |", "|---|---|"]
    sat += [f"| {k} | {v:,} |" for k, v in oz["dosyalar"].items()]
    sat += ["", "Ayrıntı `ozet.json`. Üretim: `python runs/latent/veri_hazirla.py --gorev " + g + "` (simulation-engine repo).", "",
            "## Tabanlar (test kişileri)", ""]
    tb = oz["taban"]
    if g == "talp73":
        sut = ["auc", "brier", "1a_dogruluk_a", "1b_dogruluk_c", "4_kisi_auc", "2_precision@3"]
        sat += ["Mevcut `tahmin/*.parquet` + `sonuc/*.json` (degerlendir.py) üç tohum ortalaması. auc/brier satır düzeyi (8.000), "
                "diğerleri PROTOKOL tanımları. L* = API LLM kolları (yalnız karşılaştırma).", "",
                "| yöntem | " + " | ".join(sut) + " |", "|---" * (len(sut) + 1) + "|"]
        sat += [f"| {m} | " + " | ".join("" if v["ort"].get(c) is None else f"{v['ort'][c]:.4f}" for c in sut) + " |" for m, v in tb.items()]
    else:
        sut = sorted({k for v in tb.values() for k in v if k not in ("y_ort",)})
        sat += ["| yöntem | " + " | ".join(sut) + " |", "|---" * (len(sut) + 1) + "|"]
        sat += [f"| {m} | " + " | ".join("" if v.get(c) is None else str(v[c]) for c in sut) + " |" for m, v in tb.items()]
        sat += ["", "madde_ort / urun_ort / urun_fiyat_ort: train kişilerin madde (ürün, ürün × fiyat düzeyi) ortalaması. ridge_persona: "
                "kisi_ozellik üzerinde madde başına RidgeCV. y_d1301: dalga 1-3 test-tekrarı (geçmiş kolu, insan tavanı). lgbm_*: "
                "LightGBM ürün + carpan, _persona + skor_* ve 5 demografi, _gecmis + geçmiş alma oranları (hepsi / ucuz / pahalı). "
                "mad_dogruluk = 1 − ort |ŷ01 − y01| (Twin makalesinin doğruluğu); madde_ici_pearson_ort = madde içinde kişiler arası "
                "korelasyonun ortalaması (sabit tahminde tanımsız)."]
        sat += ["", "gpt41mini_persona = veri setinin kendi GPT-4.1-mini simülasyonu (dalga 1-3 persona metni; bizim koşumuz değil, "
                "API çağrısı yapılmadı)." + (" Kesin 0/1 cevap verdiği için auc/brier/logloss onun aleyhine." if g == "twin_fiyat" else "")]
    with open(f"{d}/README.md", "w") as f:
        f.write("\n".join(sat) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gorev", nargs="*", default=list(GOREVLER))
    ap.add_argument("--cikti", default=CIKTI)
    arg = ap.parse_args()
    for g in arg.gorev:
        d = f"{arg.cikti}/{g}"
        os.makedirs(d, exist_ok=True)
        oz = GOREVLER[g](d)
        oz["dosyalar"] = {os.path.basename(f): pl.scan_parquet(f).select(pl.len()).collect().item()
                          for f in sorted(glob.glob(f"{d}/*.parquet"))}
        with open(f"{d}/ozet.json", "w") as f:
            json.dump(oz, f, ensure_ascii=False, indent=1, default=str)
        readme(g, oz, d)
        print(json.dumps(oz, ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
