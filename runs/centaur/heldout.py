"""Consumer-101 held-out evaluation: classical baselines and a Claude in-context arm on the corpus test files.

    PY="nice -n 19 /data/talp/venv/bin/python"; cd simulation-engine
    $PY runs/centaur/heldout.py klasik                          # every classical arm, talp73 seeds 0 1 2 + Twin (CPU, minutes)
    $PY runs/centaur/heldout.py llm --model haiku --n 400       # Claude in-context arm, every held-out set, seed 0
    $PY runs/centaur/heldout.py llm --model sonnet --n 150 --ayrim kisi gorev
    $PY runs/centaur/heldout.py skor                            # metrics + paired person bootstrap -> sonuc/*.json, tablo.md
    $PY runs/centaur/heldout.py skor --qlora DIR                # also score QLoRA predictions written by runs/centaur/qlora.py

Held-out sets (README of /data/talp/is/centaur/korpus):
  kisi      unseen people, same task. Classical arms fit on talp73_s{s}_train (rolling cut-offs inside the transcripts);
            the in-context arm sees K talp73 train transcripts (with their 2022-H1 answers) before the test person.
  gorev     unseen task: nothing from talp73 may be used for fitting. Classical arms are parameter-free (the person's
            own history only); the in-context arm is zero-shot (no example transcripts). Same test file as kisi.
  kapak     cover story: categories renamed, "Amazon record" -> "household diary". Classical arms are name-free, so they
            give the kisi numbers (checked); the in-context examples keep the original wording.
  kategori  5 categories never seen in training (ABIS_BOOK, CELLULAR_PHONE_CASE, DRINKING_CUP, HEALTH_PERSONAL_CARE,
            SOCKS). Classical arms fit on talp73_s{s}_train_kategori_disi without a category identity feature; the
            in-context examples come from the same 15-category transcripts. The test context still shows the person's
            own history of all 20 categories.
  twin      Twin-2K-500 unseen people (wave 4 given waves 1-3). Classical: Part-1 repeat, item mean, per-item linear.

Every number comes from the files written here: tahmin/<ayrim>__<kol>__s<s>.parquet, sonuc/*.json. Labels are read only
by `skor` (and by `klasik` for nothing: classical arms fit on train transcripts only). The Claude client is the TALP
one (`claude_api.call`: disk cache, spend log, budget gate; the key is read inside it and never printed here).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

KORPUS = "/data/talp/is/centaur/korpus"
CIKTI = "/data/talp/is/centaur/heldout"
LATENT = "/data/talp/is/latent/veri"
TALP_LLM = "/srv/lighthouse/hackathon-coz/projeler/talp-simulation-hackathon-2026/uretim/talp73/kod/llm"
TALP73_TAHMIN = "/data/talp/is/talp73/tahmin"
SABIT = 20261024
AYRIMLAR = ("kisi", "gorev", "kapak", "kategori", "twin")
MODELLER = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
K_ORNEK = {"kisi": 4, "kapak": 4, "kategori": 4, "gorev": 0, "twin": 2}


def _korpus_modulu():
    spec = importlib.util.spec_from_file_location("c101_korpus", os.path.join(os.path.dirname(__file__), "korpus.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


K = _korpus_modulu()


def test_dosyasi(ayrim: str, s: int = 0) -> str:
    if ayrim == "twin":
        return f"{KORPUS}/twin_test_kisi.jsonl"
    return f"{KORPUS}/talp73_s{s}_test_{'kisi' if ayrim == 'gorev' else ayrim}.jsonl"


def etiket_dosyasi(ayrim: str, s: int = 0) -> str:
    return test_dosyasi(ayrim, s).replace(".jsonl", "_etiket.jsonl")


# ------------------------------------------------------------------------------------------------ transcript parsing
BASLIK = re.compile(r"^(January|July) - (June|December) (\d{4}):$")
SATIR = re.compile(r"^(.*): <<(yes|no)>>$")
DIGER = re.compile(r"^Other (?:purchases in this period|orders in this block): (?:(\d+) items|none)")


def ad_haritasi() -> dict:
    """shown name -> category key, for both the original names and the cover-story names."""
    oge = pd.read_parquet(f"{LATENT}/talp73/oge.parquet")
    m = dict(zip(oge["ad"], oge["oge"]))
    m.update({v: k for k, v in K.KAPAK.items()})
    return m


def ayristir(metin: str, adlar: dict) -> list[dict]:
    """[{donem, secim: {key: 0/1}, diger: n items}] for every period with at least one << >> line, in order."""
    out, cur = [], None
    for ln in metin.splitlines():
        m = BASLIK.match(ln.strip())
        if m:
            yil, h = int(m.group(3)), m.group(1) == "July"
            cur = {"donem": (yil - K.ILK_YIL) * 2 + h, "secim": {}, "diger": 0}
            out.append(cur)
            continue
        if cur is None:
            continue
        m = SATIR.match(ln)
        if m:
            cur["secim"][adlar[m.group(1)]] = int(m.group(2) == "yes")
            continue
        m = DIGER.match(ln)
        if m:
            cur["diger"] = int(m.group(1) or 0)
    out = [d for d in out if d["secim"]]
    for d in out:
        d["kisi_oran"] = float(np.mean(list(d["secim"].values())))
    return out


OZ = ["n", "oran", "son1", "son2", "gecen_yil", "son4", "yakin", "kisi_son", "kisi_ort", "diger_son", "diger_ort",
      "hedef_yari"]


def ozellik(gec: list[dict], k: str, hedef: int) -> dict:
    """Features of (person, category) from the periods before `hedef`. Name-free and category-free by construction."""
    v = [d["secim"][k] for d in gec]
    n = len(v)
    w = np.array([0.5 ** (n - 1 - i) for i in range(n)])
    kisi = [d["kisi_oran"] for d in gec]
    gy = [d["secim"][k] for d in gec if d["donem"] == hedef - 2]
    return {"n": n, "oran": float(np.mean(v)), "son1": v[-1], "son2": v[-2] if n >= 2 else np.nan,
            "gecen_yil": gy[0] if gy else np.nan, "son4": float(np.mean(v[-4:])),
            "yakin": float((w @ np.array(v) + 0.5) / (w.sum() + 1.0)), "kisi_son": kisi[-1],
            "kisi_ort": float(np.mean(kisi)), "diger_son": math.log1p(gec[-1]["diger"]),
            "diger_ort": float(np.mean([math.log1p(d["diger"]) for d in gec])), "hedef_yari": hedef % 2}


def egitim_satirlari(yol: str, adlar: dict, en_az: int = 1) -> pd.DataFrame:
    """Rolling cut-offs inside each train transcript: target = every period with >= en_az earlier periods."""
    rows = []
    for r in K.oku_jsonl(yol):
        ds = ayristir(r["text"], adlar)
        for j in range(en_az, len(ds)):
            gec, hd = ds[:j], ds[j]
            for k, y in hd["secim"].items():
                rows.append({"uid": r["uid"], "anahtar": k, "y": y, **ozellik(gec, k, hd["donem"])})
    return pd.DataFrame(rows)


def test_satirlari(kayitlar: list[dict], adlar: dict) -> pd.DataFrame:
    rows = []
    for r in kayitlar:
        ds = ayristir(r["baglam"], adlar)
        hedef = ds[-1]["donem"] + 1
        assert K.donem_etiketi(hedef) == r["hedef_donem"], r["uid"]
        for h in r["hedefler"]:
            rows.append({"uid": r["uid"], "anahtar": h["anahtar"], **ozellik(ds, h["anahtar"], hedef)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ classical arms
def _lojistik(tr: pd.DataFrame, te: pd.DataFrame, kat: bool):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def X(d):
        x = d[OZ].copy()
        for c in ("son2", "gecen_yil"):
            x[c + "_yok"] = x[c].isna().astype(float)
            x[c] = x[c].fillna(x["oran"])
        if kat:
            x = pd.concat([x, pd.get_dummies(d["anahtar"], prefix="k", dtype=float)], axis=1)
        return x
    xtr, xte = X(tr), X(te)
    xte = xte.reindex(columns=xtr.columns, fill_value=0.0)
    sc = StandardScaler().fit(xtr)
    m = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(xtr), tr["y"])
    return m.predict_proba(sc.transform(xte))[:, 1]


def _gbm(tr: pd.DataFrame, te: pd.DataFrame, kat: bool):
    import lightgbm as lgb
    cols = OZ + (["kat"] if kat else [])

    def X(d):
        x = d[OZ].copy()
        if kat:
            x["kat"] = pd.Categorical(d["anahtar"], categories=sorted(tr["anahtar"].unique()))
        return x
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=0, verbose=-1)
    m.fit(X(tr)[cols], tr["y"])
    return m.predict_proba(X(te)[cols])[:, 1]


def _yaz(df: pd.DataFrame, ayrim: str, kol: str, s: int):
    os.makedirs(f"{CIKTI}/tahmin", exist_ok=True)
    df.to_parquet(f"{CIKTI}/tahmin/{ayrim}__{kol}__s{s}.parquet", index=False)


def klasik_talp(s: int, adlar: dict):
    t0 = time.time()
    tr = egitim_satirlari(f"{KORPUS}/talp73_s{s}_train.jsonl", adlar)
    trk = egitim_satirlari(f"{KORPUS}/talp73_s{s}_train_kategori_disi.jsonl", adlar)
    print(f"s{s}: train rows {len(tr)} (all 20), {len(trk)} (15 categories), {time.time() - t0:.0f}s", flush=True)
    son_oran = float(tr.loc[tr["hedef_yari"] == 0, "y"].mean())   # base rate of the train targets in first halves
    for ayrim in ("kisi", "gorev", "kapak", "kategori"):
        te = test_satirlari(K.oku_jsonl(test_dosyasi(ayrim, s)), adlar)
        base = te[["uid", "anahtar"]].copy()
        kollar = {"frekans": (te["oran"] * te["n"] + 1) / (te["n"] + 2), "yakin": te["yakin"],
                  "rastgele": np.random.default_rng(SABIT + s).random(len(te))}
        if ayrim == "kisi" or ayrim == "kapak":
            kollar.update({"sabit": np.full(len(te), float(tr["y"].mean())),
                           "lojistik": _lojistik(tr, te, True), "gbm": _gbm(tr, te, True)})
        elif ayrim == "kategori":
            kollar.update({"sabit": np.full(len(te), float(trk["y"].mean())),
                           "lojistik": _lojistik(trk, te, False), "gbm": _gbm(trk, te, False),
                           # reference only, NOT held out: the same models fit on all 20 categories
                           "gbm-gordu": _gbm(tr, te, True), "lojistik-gordu": _lojistik(tr, te, True)})
        else:  # gorev: no talp73 fitting allowed; a constant 0.5 is the only prior-free constant
            kollar["sabit"] = np.full(len(te), 0.5)
        for kol, p in kollar.items():
            _yaz(base.assign(p=np.asarray(p, float)), ayrim, kol, s)
        print(f"s{s} {ayrim}: {len(te)} rows, arms {sorted(kollar)} (train 2022-H1-type base rate {son_oran:.3f})",
              flush=True)


def twin_veri():
    og = pd.read_parquet(f"{LATENT}/twin/oge.parquet").set_index("oge")
    h = pd.read_parquet(f"{LATENT}/twin/hedef.parquet")
    g = pd.read_parquet(f"{LATENT}/twin/gecmis.parquet")
    return og, h, g


def klasik_twin():
    og, h, g = twin_veri()
    tr_uid = {r["uid"] for r in K.oku_jsonl(f"{KORPUS}/twin_train.jsonl")}
    te = pd.DataFrame(K.oku_jsonl(etiket_dosyasi("twin")))[["uid", "anahtar", "oge"]]
    h = h.assign(uid=h["uid"].astype(str))
    g = g.assign(uid=g["uid"].astype(str))
    htr = h[h["uid"].isin(tr_uid)].merge(g, on=["uid", "oge"], how="left")
    ort = htr.groupby("oge")["y01"].mean()
    # per-item linear y4 = a + b * y13 on train people (y01 scale); items without enough pairs fall back to the mean
    ab = {}
    for o, d in htr.dropna(subset=["y_d1301"]).groupby("oge"):
        if len(d) >= 30 and d["y_d1301"].std() > 0:
            b, a = np.polyfit(d["y_d1301"], d["y01"], 1)
            ab[o] = (a, b)
    te = te.merge(g, on=["uid", "oge"], how="left")
    tekrar = te["y_d1301"].fillna(te["oge"].map(ort))
    lin = [ab[o][0] + ab[o][1] * x if (o in ab and not np.isnan(x)) else ort[o] for o, x in zip(te["oge"], te["y_d1301"])]
    for kol, p in {"tekrar": tekrar, "ortalama": te["oge"].map(ort), "dogrusal": lin}.items():
        _yaz(te[["uid", "anahtar", "oge"]].assign(y01=np.clip(np.asarray(p, float), 0, 1)), "twin", kol, 0)
    print(f"twin: {len(te)} rows; {len(ab)} items with a linear fit; train people {len(tr_uid)}", flush=True)


def cmd_klasik(a):
    adlar = ad_haritasi()
    for s in a.tohum:
        klasik_talp(s, adlar)
    klasik_twin()


# ------------------------------------------------------------------------------------------------ Claude in-context arm
def _claude():
    sys.dont_write_bytecode = True     # TALP repo is read-only for us: no __pycache__ there
    sys.path.insert(0, TALP_LLM)
    import claude_api as C
    import openrouter as O
    O.AJAN = "lh-hack-c101-heldout"
    return C


def orneklem(uids: list, n: int) -> list:
    return random.Random(SABIT).sample(sorted(uids), min(n, len(uids)))


def ornek_blok(ayrim: str, s: int = 0) -> list[str]:
    k = K_ORNEK[ayrim]
    if not k:
        return []
    yol = {"kisi": f"talp73_s{s}_train", "kapak": f"talp73_s{s}_train", "twin": "twin_train",
           "kategori": f"talp73_s{s}_train_kategori_disi"}[ayrim]
    return K.ornek_sec(K.oku_jsonl(f"{KORPUS}/{yol}.jsonl"), k, tohum=s)


def istem(kayit: dict, ornekler: list[str]) -> list[dict]:
    """System = fixed instruction + the example transcripts (identical for every call -> prompt cache).
    User = korpus.api_istemi(kayit) (no labels: built from the test record only)."""
    p = K.api_istemi(kayit)
    sistem = p["system"]
    user = p["user"]
    if ornekler:
        sistem += (f"\n\nHere are {len(ornekler)} complete records of other people, for reference:\n\n"
                   + "\n".join(f"=== Example record {i} ===\n{t.strip()}\n" for i, t in enumerate(ornekler, 1)))
        user = "=== Record to predict ===\n" + user
    return [{"role": "system", "content": sistem}, {"role": "user", "content": user}]


def dogrulayici(anahtarlar: list[str], twin: bool):
    def f(metin: str):
        t = metin.strip()
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
        i, j = t.find("{"), t.rfind("}")
        d = json.loads(t[i:j + 1])
        out = {}
        for k in anahtarlar:
            v = float(d[k])
            if not twin and not 0 <= v <= 100:
                raise ValueError(f"{k}={v} outside 0-100")
            out[k] = v
        return out
    return f


def twin_olcek_haritasi() -> dict:
    og = pd.read_parquet(f"{LATENT}/twin/oge.parquet")
    return {r.oge: (float(r.olcek_alt), float(r.olcek_ust)) for r in og.itertuples()}


def cmd_llm(a):
    C = _claude()
    model = MODELLER[a.model]
    os.environ.setdefault("CLAUDE_DUR_USD", str(C.DUR_USD))
    print(f"spend so far (Claude, whole log) {C.harcama_toplam():.2f} $, stop at {C.DUR_USD} $", flush=True)
    olcek = twin_olcek_haritasi()
    for ayrim in a.ayrim:
        kayitlar = K.oku_jsonl(test_dosyasi(ayrim, 0))
        secilen = set(orneklem([r["uid"] for r in kayitlar], a.n))
        kayitlar = [r for r in kayitlar if r["uid"] in secilen]
        ornekler = ornek_blok(ayrim)
        twin = ayrim == "twin"
        h0 = C.harcama_toplam()
        sonuc, hata = {}, {}

        def is_(r):
            m = istem(r, ornekler)
            keys = [h["anahtar"] for h in r["hedefler"]]
            kw = {"dusunme": "kapali", "effort": "low", "max_tokens": 1500 if twin else 700}
            return r["uid"], C.call(m, model, etiket=f"c101:{ayrim}:{a.model}", dogrula=dogrulayici(keys, twin), **kw)
        with ThreadPoolExecutor(a.paralel) as ex:
            fut = {ex.submit(is_, r): r["uid"] for r in kayitlar}
            for f in as_completed(fut):
                try:
                    u, r = f.result()
                    sonuc[u] = r["veri"]
                except Exception as e:                               # noqa: BLE001 - recorded, counted, reported
                    hata[fut[f]] = f"{type(e).__name__}: {str(e)[:160]}"
                    if type(e).__name__ in ("ButceAsildi", "Yetkisiz"):
                        raise
        rows = []
        for r in kayitlar:
            if r["uid"] not in sonuc:
                continue
            for h in r["hedefler"]:
                v = sonuc[r["uid"]][h["anahtar"]]
                if twin:
                    lo, hi = olcek[h["oge"]]
                    rows.append({"uid": r["uid"], "anahtar": h["anahtar"], "oge": h["oge"], "ham": v,
                                 "y01": float(np.clip((v - lo) / (hi - lo), 0, 1)) if hi > lo else np.nan})
                else:
                    rows.append({"uid": r["uid"], "anahtar": h["anahtar"], "p": v / 100.0})
        _yaz(pd.DataFrame(rows), ayrim, f"claude-{a.model}", 0)
        json.dump({"ayrim": ayrim, "model": model, "n_istenen": len(kayitlar), "n_tamam": len(sonuc), "hata": hata,
                   "k_ornek": len(ornekler), "usd_bu_ayrim": round(C.harcama_toplam() - h0, 4)},
                  open(f"{CIKTI}/tahmin/{ayrim}__claude-{a.model}__s0.meta.json", "w"), ensure_ascii=False, indent=1)
        print(f"{ayrim} {a.model}: {len(sonuc)}/{len(kayitlar)} people, {len(hata)} errors, "
              f"{C.harcama_toplam() - h0:.3f} $ (total {C.harcama_toplam():.2f} $)", flush=True)


# ------------------------------------------------------------------------------------------------ Qwen3.8 zero-shot logprob
OR_MODEL = "qwen/qwen3.8-27b"
OR_SAGLAYICI = {"order": ["Modal"], "allow_fallbacks": False, "require_parameters": True}
_BUTCE = {"t": 0.0, "usd": 0.0, "eklenen": 0.0}


def _butce_asildi(O) -> bool:
    """The spend log is 30+ MB: re-read it at most every 60 s, add this process's own spend in between."""
    if time.time() - _BUTCE["t"] > 60:
        _BUTCE.update(t=time.time(), usd=O.harcama_toplam(), eklenen=0.0)
    return _BUTCE["usd"] + _BUTCE["eklenen"] >= O.DUR_USD


def or_evet_olasiligi(prompt: str, O, etiket: str, onbellek_yalniz: bool = False) -> dict:
    """One completion token with top-20 logprobs; P(yes) = renormalised over the yes/no tokens. OpenRouter wraps the
    prompt in the chat template (the first answer token is often 'Based'), so this is a zero-shot approximation of
    the raw-text readout that runs/centaur/qlora.py does on the GPU. Disk cache keyed on the prompt."""
    import hashlib
    import requests
    h = hashlib.sha256(json.dumps([OR_MODEL, OR_SAGLAYICI, prompt]).encode()).hexdigest()[:32]
    fp = f"{CIKTI}/or_cache/{h}.json"
    if os.path.exists(fp):
        return json.load(open(fp))
    if onbellek_yalniz:
        raise KeyError("not in cache")
    hata = ""
    for i in range(4):
        if _butce_asildi(O):
            raise O.ButceAsildi(f"OpenRouter log total >= {O.DUR_USD} $")
        body = {"model": OR_MODEL, "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": True,
                "top_logprobs": 20, "reasoning": {"enabled": False}, "provider": OR_SAGLAYICI,
                "usage": {"include": True}}
        try:
            resp = requests.post("https://openrouter.ai/api/v1/completions", json=body, timeout=120,
                                 headers={"Authorization": f"Bearer {O._anahtar()}", "X-Title": "c101"})
        except requests.RequestException as e:
            hata = type(e).__name__
            time.sleep(3 * (i + 1))
            continue
        if resp.status_code == 402:                       # out of OpenRouter credits: no retry, stop
            raise O.Durdur("OpenRouter HTTP 402: insufficient credits")
        if resp.status_code != 200:
            hata = f"HTTP {resp.status_code}"
            time.sleep((10 if resp.status_code == 429 else 3) * (i + 1))
            continue
        out = resp.json()
        u = out.get("usage") or {}
        O._log_yaz(OR_MODEL, int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0),
                   float(u.get("cost") or 0.0), etiket)
        _BUTCE["eklenen"] += float(u.get("cost") or 0.0)
        try:
            top = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            hata = "no logprobs"
            continue
        lp = {}
        for t in top:
            k = t["token"].strip().lower()
            if k in ("yes", "no"):
                lp[k] = max(lp.get(k, -1e9), t["logprob"])
        taban = min(t["logprob"] for t in top)            # a token outside the top-20 is below the last one
        ly, ln_ = lp.get("yes", taban), lp.get("no", taban)
        r = {"p": float(1 / (1 + math.exp(ln_ - ly))), "yes_top": "yes" in lp, "no_top": "no" in lp,
             "ilk": top[0]["token"], "saglayici": out.get("provider")}
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        json.dump(r, open(fp, "w"))
        return r
    raise RuntimeError(f"OpenRouter call failed: {hata}")


def cmd_qwen(a):
    _claude()
    import openrouter as O
    print(f"OpenRouter log total {O.harcama_toplam():.2f} $ (stop {O.DUR_USD} $)", flush=True)
    for dosya in a.dosya:                     # kisi file -> stored as gorev (the base model never saw talp73)
        kayitlar = K.oku_jsonl(test_dosyasi(dosya, 0))
        secilen = set(orneklem([r["uid"] for r in kayitlar], a.n))
        isler = [(r["uid"], h["anahtar"], r["baglam"] + h["devam"]) for r in kayitlar if r["uid"] in secilen
                 for h in r["hedefler"]]
        ayrim = "gorev" if dosya == "kisi" else dosya
        h0 = O.harcama_toplam()
        rows, hata = [], 0
        with ThreadPoolExecutor(a.paralel) as ex:
            fut = {ex.submit(or_evet_olasiligi, p, O, f"c101:{ayrim}:qwen38-sifir", a.onbellek_yalniz): (u, k)
                   for u, k, p in isler}
            for f in as_completed(fut):
                u, k = fut[f]
                try:
                    rows.append({"uid": u, "anahtar": k, **f.result()})
                except O.Durdur:
                    raise
                except Exception:                                       # noqa: BLE001 - counted
                    hata += 1
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"{dosya}: no target answered", flush=True)
            continue
        tam = df.groupby("uid")["anahtar"].transform("size") == len(kayitlar[0]["hedefler"])
        df = df[tam]                               # keep people with every target answered (paired comparisons)
        _yaz(df[["uid", "anahtar", "p"]], ayrim, "qwen38-sifir", 0)
        json.dump({"dosya": dosya, "ayrim": ayrim, "model": OR_MODEL, "saglayici": OR_SAGLAYICI, "n_hedef": len(isler),
                   "n_tamam": len(df), "hata": hata, "yes_top_orani": float(df["yes_top"].mean()),
                   "no_top_orani": float(df["no_top"].mean()),
                   "ilk_token": df["ilk"].value_counts().head(5).to_dict(), "usd": round(O.harcama_toplam() - h0, 4)},
                  open(f"{CIKTI}/tahmin/{ayrim}__qwen38-sifir__s0.meta.json", "w"), indent=1)
        print(f"{dosya}->{ayrim}: {len(df)}/{len(isler)} targets, {hata} errors, {O.harcama_toplam() - h0:.3f} $",
              flush=True)


# ------------------------------------------------------------------------------------------------ scoring
def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def kisi_auc(df: pd.DataFrame) -> float:
    """Mean within-person AUC over people with both classes (talp73 metric 4)."""
    v = [auc(d["y"], d["p"]) for _, d in df.groupby("uid") if 0 < d["y"].sum() < len(d)]
    return float(np.mean(v)) if v else float("nan")


def ikili_metrik(df: pd.DataFrame) -> dict:
    p = np.clip(df["p"].to_numpy(float), 0.01, 0.99)
    y = df["y"].to_numpy(float)
    return {"n_kisi": int(df["uid"].nunique()), "n_karar": int(len(df)), "y_oran": float(y.mean()),
            "auc": auc(y, p), "kisi_auc": kisi_auc(df), "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2))}


def twin_metrik(df: pd.DataFrame) -> dict:
    """MAE on the 0-1 scale; mean per-item Pearson r across people (individual differences, base rates removed)."""
    d = df.dropna(subset=["y01", "p01"])
    rs = []
    for _, g in d.groupby("oge"):
        if len(g) >= 20 and g["y01"].std() > 0 and g["p01"].std() > 0:
            rs.append(np.corrcoef(g["y01"], g["p01"])[0, 1])
    return {"n_kisi": int(d["uid"].nunique()), "n_karar": int(len(d)), "mae": float(np.mean(np.abs(d["y01"] - d["p01"]))),
            "madde_r": float(np.mean(rs)) if rs else float("nan"), "n_madde": len(rs)}


def bootstrap(a: pd.DataFrame, b: pd.DataFrame, fn, anahtarlar=("kisi_auc", "logloss"), n: int = 1000) -> dict:
    """Paired person-level bootstrap of metric(a) - metric(b) on the same people and keys. Binary metrics use
    per-person sufficient statistics (fast); Twin metrics are recomputed on the resampled rows."""
    uids = sorted(set(a["uid"]) & set(b["uid"]))
    a, b = a[a["uid"].isin(uids)], b[b["uid"].isin(uids)]
    rng = np.random.default_rng(SABIT)
    fark = {k: [] for k in anahtarlar}
    if fn is ikili_metrik:
        def kisi_tablo(d):
            g = {u: x for u, x in d.groupby("uid")}
            y = [g[u]["y"].to_numpy(float) for u in uids]
            p = [np.clip(g[u]["p"].to_numpy(float), 0.01, 0.99) for u in uids]
            ka = np.array([auc(yy, pp) if 0 < yy.sum() < len(yy) else np.nan for yy, pp in zip(y, p)])
            ll = np.array([-(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)).sum() for yy, pp in zip(y, p)])
            br = np.array([((pp - yy) ** 2).sum() for yy, pp in zip(y, p)])
            return y, p, ka, ll, br, np.array([len(yy) for yy in y])
        ta, tb = kisi_tablo(a), kisi_tablo(b)
        for _ in range(n):
            s = rng.integers(0, len(uids), len(uids))
            for t, out in ((ta, "a"), (tb, "b")):
                y, p, ka, ll, br, nn = t
                m = {"kisi_auc": np.nanmean(ka[s]), "logloss": ll[s].sum() / nn[s].sum(), "brier": br[s].sum() / nn[s].sum()}
                if "auc" in anahtarlar:
                    m["auc"] = auc(np.concatenate([y[i] for i in s]), np.concatenate([p[i] for i in s]))
                if out == "a":
                    ma = m
                else:
                    mb = m
            for k in anahtarlar:
                fark[k].append(ma[k] - mb[k])
    else:
        ga, gb = dict(list(a.groupby("uid"))), dict(list(b.groupby("uid")))
        for _ in range(n):
            s = rng.choice(uids, len(uids), replace=True)
            da = pd.concat([ga[u].assign(uid=f"{u}#{i}") for i, u in enumerate(s)])
            db = pd.concat([gb[u].assign(uid=f"{u}#{i}") for i, u in enumerate(s)])
            ma, mb = fn(da), fn(db)
            for k in anahtarlar:
                fark[k].append(ma[k] - mb[k])
    full_a, full_b = fn(a), fn(b)
    return {k: {"fark": full_a[k] - full_b[k], "ci95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]}
            for k, v in fark.items()}


def baska_kisi(df: pd.DataFrame) -> pd.DataFrame:
    """Control: each person's predictions scored against the NEXT person's labels (same keys). What survives is base
    rate / category knowledge, not the individual."""
    uids = sorted(df["uid"].unique())
    nxt = dict(zip(uids, uids[1:] + uids[:1]))
    p = df[["uid", "anahtar", "p"]].assign(uid=lambda d: d["uid"].map(nxt))
    return p.merge(df[["uid", "anahtar", "y"]], on=["uid", "anahtar"])


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def cmd_skor(a):
    os.makedirs(f"{CIKTI}/sonuc", exist_ok=True)
    tah = sorted(f for f in os.listdir(f"{CIKTI}/tahmin") if f.endswith(".parquet"))
    for d in a.qlora or []:
        tah += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".parquet")]
    oku = lambda f: pd.read_parquet(f if os.path.isabs(f) else f"{CIKTI}/tahmin/{f}")  # noqa: E731
    sonuc = {"satirlar": [], "karsilastirma": [], "kontrol": []}
    etk = {}
    for f in tah:
        ayrim, kol, s = os.path.basename(f)[:-8].split("__")
        s = int(s[1:])
        if (ayrim, s) not in etk:
            e = pd.DataFrame(K.oku_jsonl(etiket_dosyasi(ayrim, s)))
            e["uid"] = e["uid"].astype(str)
            etk[(ayrim, s)] = e
        e, p = etk[(ayrim, s)], oku(f)
        p["uid"] = p["uid"].astype(str)
        if ayrim == "twin":
            d = p.rename(columns={"y01": "p01"}).merge(e[["uid", "anahtar", "y01"]], on=["uid", "anahtar"])
            m = twin_metrik(d)
        else:
            d = p.merge(e[["uid", "anahtar", "y"]], on=["uid", "anahtar"])
            m = ikili_metrik(d)
        assert len(d) == len(p), f"{f}: {len(p) - len(d)} predictions without a label"
        sonuc["satirlar"].append({"ayrim": ayrim, "kol": kol, "tohum": s, **m})
    # paired comparisons on the LLM people (seed 0): each LLM arm vs the best classical arm of that split
    best = {"kisi": "gbm", "kapak": "gbm", "kategori": "gbm", "gorev": "yakin", "twin": "dogrusal"}
    ref = {"kisi": ["gbm", "lojistik", "frekans", "yakin"], "kapak": ["gbm"], "kategori": ["gbm", "lojistik", "frekans", "gbm-gordu"],
           "gorev": ["yakin", "frekans"], "twin": ["dogrusal", "tekrar"]}
    for ayrim in AYRIMLAR:
        llm = [f for f in tah if os.path.basename(f).startswith(f"{ayrim}__")
               and any(t in os.path.basename(f) for t in ("claude-", "qlora", "qwen38"))]
        for f in llm:
            kol = os.path.basename(f).split("__")[1]
            e = etk[(ayrim, 0)]
            p = oku(f).assign(uid=lambda d: d["uid"].astype(str))
            uids = set(p["uid"])
            for rk in ref[ayrim]:
                q = oku(f"{ayrim}__{rk}__s0.parquet").assign(uid=lambda d: d["uid"].astype(str))
                q = q[q["uid"].isin(uids)]
                if ayrim == "twin":
                    da = p.rename(columns={"y01": "p01"}).merge(e[["uid", "anahtar", "y01"]], on=["uid", "anahtar"])
                    db = q.rename(columns={"y01": "p01"}).merge(e[["uid", "anahtar", "y01"]], on=["uid", "anahtar"])
                    fn, ks = twin_metrik, ("mae", "madde_r")
                else:
                    da = p.merge(e[["uid", "anahtar", "y"]], on=["uid", "anahtar"])
                    db = q.merge(e[["uid", "anahtar", "y"]], on=["uid", "anahtar"])
                    fn, ks = ikili_metrik, ("auc", "kisi_auc", "logloss", "brier")
                row = {"ayrim": ayrim, "llm": kol, "klasik": rk, "llm_m": fn(da), "klasik_m": fn(db),
                       "boot": bootstrap(da, db, fn, ks, n=a.boot if ayrim != "twin" else min(a.boot, 200))}
                if ayrim != "twin" and rk == best[ayrim]:      # equal-weight logit average (no fitting)
                    st = da[["uid", "anahtar", "y"]].assign(p=1 / (1 + np.exp(-(logit(da["p"].to_numpy()) + logit(
                        db.set_index(["uid", "anahtar"]).loc[list(zip(da["uid"], da["anahtar"])), "p"].to_numpy())) / 2)))
                    row["yigin_m"] = fn(st)
                    row["yigin_boot"] = bootstrap(st, db, fn, ks, n=a.boot)
                sonuc["karsilastirma"].append(row)
                print(f"{ayrim} {kol} vs {rk}: " + ", ".join(
                    f"{k} {row['llm_m'][k]:.4f}/{row['klasik_m'][k]:.4f}" for k in ks), flush=True)
            if ayrim != "twin":
                d = p.merge(e[["uid", "anahtar", "y"]], on=["uid", "anahtar"])
                q = oku(f"{ayrim}__{best[ayrim]}__s0.parquet").assign(uid=lambda x: x["uid"].astype(str))
                q = q[q["uid"].isin(uids)].merge(e[["uid", "anahtar", "y"]], on=["uid", "anahtar"])
                sonuc["kontrol"].append({"ayrim": ayrim, "kol": kol, "baska_kisi": ikili_metrik(baska_kisi(d)),
                                         "klasik": best[ayrim], "klasik_baska_kisi": ikili_metrik(baska_kisi(q))})
    # LLM vs LLM on the same people and labels: in-context examples (kisi, K=4) vs none (gorev); renamed (kapak) vs
    # original wording (kisi). Labels of kisi / gorev / kapak are the same panel labels for the same (uid, key).
    sonuc["llm_llm"] = []
    e = etk.get(("kisi", 0))
    for (a1, k1), (a2, k2) in [(("kisi", "claude-haiku"), ("gorev", "claude-haiku")),
                               (("kapak", "claude-haiku"), ("kisi", "claude-haiku")),
                               (("kategori", "claude-haiku"), ("gorev", "claude-haiku"))]:
        f1, f2 = f"{a1}__{k1}__s0.parquet", f"{a2}__{k2}__s0.parquet"
        if e is None or f1 not in tah or f2 not in tah:
            continue
        d1 = oku(f1).assign(uid=lambda x: x["uid"].astype(str)).merge(e[["uid", "anahtar", "y"]], on=["uid", "anahtar"])
        d2 = oku(f2).assign(uid=lambda x: x["uid"].astype(str)).merge(e[["uid", "anahtar", "y"]], on=["uid", "anahtar"])
        d2 = d2.merge(d1[["uid", "anahtar"]], on=["uid", "anahtar"])          # same (uid, key) set
        m = d1.merge(d2, on=["uid", "anahtar"], suffixes=("_1", "_2"))
        sonuc["llm_llm"].append({"a": f"{a1}:{k1}", "b": f"{a2}:{k2}", "a_m": ikili_metrik(d1), "b_m": ikili_metrik(d2),
                                 "r_tahmin": float(np.corrcoef(m["p_1"], m["p_2"])[0, 1]),
                                 "boot": bootstrap(d1, d2, ikili_metrik, ("auc", "kisi_auc", "logloss", "brier"),
                                                   n=a.boot)})
    # talp73 G1 reference rows (same 400 people and labels, seed 0): gecmis (all events) and L2-sonnet55
    e = etk.get(("kisi", 0))
    if e is not None:
        for kol in ("gecmis", "L2-sonnet55"):
            yol = f"{TALP73_TAHMIN}/{kol}_s0.parquet"
            if os.path.exists(yol):
                p = pd.read_parquet(yol).rename(columns={"kategori": "anahtar"})
                d = p.merge(e[["uid", "anahtar", "y"]], on=["uid", "anahtar"])
                sonuc["satirlar"].append({"ayrim": "kisi", "kol": f"talp73:{kol}", "tohum": 0, **ikili_metrik(d)})
                for f in tah:
                    if os.path.basename(f).startswith("kisi__claude-"):
                        u = set(oku(f)["uid"].astype(str))
                        sonuc["satirlar"].append({"ayrim": "kisi", "kol": f"talp73:{kol}", "tohum": 0,
                                                  "alt": os.path.basename(f).split("__")[1],
                                                  **ikili_metrik(d[d["uid"].isin(u)])})
    json.dump(sonuc, open(f"{CIKTI}/sonuc/skor.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(f"-> {CIKTI}/sonuc/skor.json", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    k = sp.add_parser("klasik")
    k.add_argument("--tohum", type=int, nargs="+", default=[0, 1, 2])
    l_ = sp.add_parser("llm")
    l_.add_argument("--model", choices=sorted(MODELLER), required=True)
    l_.add_argument("--n", type=int, default=150)
    l_.add_argument("--ayrim", nargs="+", default=list(AYRIMLAR), choices=AYRIMLAR)
    l_.add_argument("--paralel", type=int, default=8)
    q_ = sp.add_parser("qwen")
    q_.add_argument("--n", type=int, default=100)
    q_.add_argument("--dosya", nargs="+", default=["kisi", "kategori", "kapak"], choices=["kisi", "kategori", "kapak"])
    q_.add_argument("--paralel", type=int, default=8)
    q_.add_argument("--onbellek-yalniz", action="store_true", help="no API call: use the disk cache only")
    s_ = sp.add_parser("skor")
    s_.add_argument("--boot", type=int, default=500)
    s_.add_argument("--qlora", nargs="*", default=[], help="directories with <ayrim>__qlora-...__s0.parquet files")
    a = ap.parse_args(argv)
    {"klasik": cmd_klasik, "llm": cmd_llm, "qwen": cmd_qwen, "skor": cmd_skor}[a.cmd](a)


if __name__ == "__main__":
    main()
