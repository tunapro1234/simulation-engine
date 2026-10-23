"""Consumer-101: a Psych-101-style corpus of consumer choices for a Centaur-style QLoRA (and an in-context API arm).

    nice -n 19 /data/talp/venv/bin/python runs/centaur/korpus.py                  # everything, seeds 0 1 2
    nice -n 19 /data/talp/venv/bin/python runs/centaur/korpus.py --tohum 0        # one talp73 seed
    nice -n 19 /data/talp/venv/bin/python runs/centaur/korpus.py --token-yok      # skip the tokenizer statistics

Every record is one person, written as one natural-language transcript; the person's own choices are wrapped in
<< >> and only those tokens go into the loss (Binz et al., Nature 2025). Output: /data/talp/is/centaur/korpus/.

Sources
  talp73     Open e-commerce 1.0 (CC0), the TALP %73 reproduction (TALP repo uretim/talp73/PROTOKOL.md V1-V8). The survey
             profile (L1 persona block), then six-month periods from the account's first order to 2022-H1: for each of the 20
             categories "<<yes>>" / "<<no>>" (>= 1 item in the period), then a summary line of the other categories
             bought in that period. Train people = eligible people outside sample s (latent/veri/talp73 hedef, bolum
             train); their transcript ends 2022-06-30. Test people = the 400 of sample s; context up to 2022-06-30, the
             2022-H2 choices are asked one category per prompt and their labels are in a separate file.
  twin       Twin-2K-500 (CC-BY-4.0, latent/veri/twin): demographics, Part 1 = the person's waves 1-3 answers to the 86
             repeated items, Part 2 = the same items in wave 4. Train people: both parts marked; test people: Part 1 is
             context, Part 2 is asked per item (labels separate).
  psych101   a small stratified subset of marcelbinz/Psych-101 (Apache-2.0), unchanged, to keep the general skill.
  Not used:  OpinionQA / SimBench / llm_global_opinions hold group answer distributions (SimBench, GlobalOpinionQA) or
             Pew ATP person rows whose terms forbid redistribution (OpinionQA human_resp); see README.

Splits (separate test files): held-out people (same task), held-out task (train on twin + psych101 only, test on talp73),
cover story (talp73 test with different category wording and framing), held-out categories (5 categories removed from the
talp73 train transcripts, asked at test).

Leakage: no event on or after 2022-07-01 is ever read into a transcript; no test person is in a train file; the train
2022-H1 choices equal latent hedef (bolum train) labels; test labels equal the panel's etiket_s<s>. `sizinti_kontrol`
asserts these on every run. The functions above `main` are pure Python (unit-tested with synthetic data); TALP repo,
polars and the tokenizer are imported only inside the loaders. No model is trained, no LLM API is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from datetime import date

KESIM = date(2022, 7, 1)          # test cut: nothing on/after this date enters any transcript
ILK_YIL = 2018
CIKTI = "/data/talp/is/centaur/korpus"
LATENT = "/data/talp/is/latent/veri"
PANEL = "/data/talp/is/talp73/panel"
TALP = "/srv/lighthouse/hackathon-coz/projeler/talp-simulation-hackathon-2026/uretim"
HF_HOME = "/data/talp/hf"
TOKENIZER_REPO = "Qwen/Qwen3.8-27B"
PSYCH_REPO = "marcelbinz/Psych-101"
TOHUMLAR = (0, 1, 2)
PSYCH_KISI = 8                    # participants per Psych-101 experiment
PSYCH_MAKS_KARAKTER = 60_000      # skip very long Psych-101 transcripts (~15k tokens)
N_KATEGORI_DISI = 5
SABIT = 20261024                  # fixed seed for every random choice in this file
AYLAR = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
         "November", "December"]
SECIM = re.compile(r"<<(.*?)>>", re.S)

# cover story: same 20 categories, different wording (none of these strings is the original name)
KAPAK = {
    "ABIS_BOOK": "Bound reading volumes",
    "ELECTRONIC_CABLE": "Wires and cords for connecting devices",
    "CELLULAR_PHONE_CASE": "Protective shells for mobile handsets",
    "SHIRT": "Tops (tees, blouses, button-downs)",
    "HEADPHONES": "Earphones and headsets",
    "HEALTH_PERSONAL_CARE": "Hygiene and wellness goods",
    "SHOES": "Footwear",
    "NUTRITIONAL_SUPPLEMENT": "Dietary pills and powders",
    "PANTS": "Trousers and leggings",
    "CHARGING_ADAPTER": "Power bricks for devices",
    "SKIN_MOISTURIZER": "Hydrating creams",
    "SOCKS": "Hosiery",
    "BATTERY": "Disposable and rechargeable cells",
    "SCREEN_PROTECTOR": "Display shields for phones and tablets",
    "GIFT_CARD": "Prepaid store vouchers",
    "PORTABLE_ELECTRONIC_DEVICE_COVER": "Sleeves for tablets and e-readers",
    "DRINKING_CUP": "Drinkware",
    "CLEANING_AGENT": "Detergents and sprays for the home",
    "MEDICATION": "Pharmacy remedies without prescription",
    "WRITING_INSTRUMENT": "Handheld writing tools",
}

CERCEVE = {
    "amazon": dict(
        giris="This is the shopping record of one Amazon customer, reconstructed from the account's real order history.\n"
              "The account holder first answered a survey:\n",
        yontem="\nThe record then goes through the account's orders in six-month periods. For each period and each of "
               "20 product categories it shows whether the account bought at least one item in that category (yes or "
               "no, the answer in double angle brackets). After each period, purchases in other categories are "
               "summarised.\n",
        diger="Other purchases in this period"),
    "kapak": dict(
        giris="Below is a household's online shopping diary, kept by a research panel from the household's own orders.\n"
              "The panel member described the household like this:\n",
        yontem="\nThe diary is kept in half-year blocks. In each block the member marks, for 20 kinds of goods, whether "
               "the household ordered any of them (yes or no, in double angle brackets). Each block ends with a note on the "
               "other orders.\n",
        diger="Other orders in this block"),
}


# ------------------------------------------------------------------------------------------------ shared helpers
def donem_indeksi(d: date) -> int:
    """Half-year index: 0 = January-June 2018, 1 = July-December 2018, ..."""
    return (d.year - ILK_YIL) * 2 + (d.month > 6)


def donem_etiketi(i: int) -> str:
    y, h = ILK_YIL + i // 2, i % 2
    return f"{AYLAR[6 * h]} - {AYLAR[6 * h + 5]} {y}"


SON_EGITIM_DONEMI = donem_indeksi(date(2022, 6, 30))   # 2022-H1
HEDEF_DONEMI = donem_indeksi(KESIM)                    # 2022-H2


def _tohum(*parca) -> int:
    return int(hashlib.md5("|".join(map(str, (SABIT,) + parca)).encode()).hexdigest()[:8], 16)


def sirali(oge: list, *anahtar) -> list:
    """Deterministic per-(person, period) shuffle: the same person gets the same order in every seed and file."""
    o = list(oge)
    random.Random(_tohum(*anahtar)).shuffle(o)
    return o


def secim_araliklari(metin: str) -> list[tuple[int, int]]:
    """Character spans of the choice contents (inside << >>): the only tokens that go into the loss."""
    return [m.span(1) for m in SECIM.finditer(metin)]


def chat_mesajlari(metin: str) -> list[dict]:
    """The transcript as user/assistant turns (assistant = '<<x>>'), for finetune.chat_rows-style trainers that put the
    loss on assistant turns. Concatenating the contents gives the transcript back."""
    out, son = [], 0
    for m in SECIM.finditer(metin):
        out.append({"role": "user", "content": metin[son:m.start()]})
        out.append({"role": "assistant", "content": m.group(0)})
        son = m.end()
    if son < len(metin):
        out.append({"role": "user", "content": metin[son:]})
    return out


# ------------------------------------------------------------------------------------------------ talp73 text
def diger_satiri(diger: dict, baslik: str) -> str:
    """diger = {"kalem": n items outside the 20, "kategori": n categories, "en_cok": [(name, count), ...]}"""
    if not diger or not diger.get("kalem"):
        return f"{baslik}: none."
    en = ", ".join(f"{a} ({n})" for a, n in diger.get("en_cok", [])[:5])
    return (f"{baslik}: {diger['kalem']} items in {diger['kategori']} other categories"
            + (f"; most items in: {en}." if en else "."))


def talp73_blok(uid: str, donem: int, secim: dict, adlar: dict, diger: dict | None, cerceve: str) -> str:
    """One period: header, one line per category in a per-(uid, period) shuffled order, the other-purchases line."""
    sat = [f"{donem_etiketi(donem)}:"]
    for k in sirali(list(adlar), uid, donem):
        sat.append(f"{adlar[k]}: <<{'yes' if secim.get(k, 0) else 'no'}>>")
    sat.append(diger_satiri(diger, CERCEVE[cerceve]["diger"]))
    return "\n".join(sat) + "\n"


def talp73_metni(uid: str, profil: str, donemler: dict, adlar: dict, diger: dict, cerceve: str = "amazon",
                 son: int = SON_EGITIM_DONEMI) -> str:
    """Whole transcript up to and including period `son`. donemler = {period: {oge: 0/1}} for periods with any order
    (missing categories = 0); diger = {period: diger dict}; adlar = {oge: shown name} (drop a key to hold a category
    out). The record starts at the first period with an order."""
    c = CERCEVE[cerceve]
    ilk = min(donemler) if donemler else son
    assert max(donemler, default=son) <= son, "a period after the cut reached the transcript"
    bloklar = [talp73_blok(uid, d, donemler.get(d, {}), adlar, diger.get(d), cerceve) for d in range(ilk, son + 1)]
    return c["giris"] + profil.strip() + "\n" + c["yontem"] + "\n" + "\n".join(bloklar)


def talp73_test_kaydi(uid: str, tohum: int, profil: str, donemler: dict, adlar: dict, diger: dict, ayrim: str,
                      cerceve: str = "amazon", sorulan: list | None = None) -> dict:
    """Context = transcript to 2022-H1 + the 2022-H2 header; each target category is its own continuation
    'Name: <<' (scored as P(yes) vs P(no) at the next token), so no target answer conditions another."""
    baglam = talp73_metni(uid, profil, donemler, adlar, diger, cerceve) + "\n" + donem_etiketi(HEDEF_DONEMI) + ":\n"
    hed = [{"anahtar": k, "oge": k, "devam": f"{adlar[k]}: <<", "aciklama": adlar[k], "secenekler": ["yes", "no"]}
           for k in adlar if sorulan is None or k in sorulan]
    return {"kaynak": "talp73", "ayrim": ayrim, "tohum": tohum, "uid": uid, "cerceve": cerceve,
            "hedef_donem": donem_etiketi(HEDEF_DONEMI), "baglam": baglam, "hedefler": hed}


# ------------------------------------------------------------------------------------------------ twin text
def twin_olcek(o: dict) -> str:
    if o["tip"] in ("MC", "Matrix") and o.get("secenekler"):
        s = json.loads(o["secenekler"]) if isinstance(o["secenekler"], str) else list(o["secenekler"])
        if s:
            return "options: " + "; ".join(f"{i + 1} = {str(x).strip()}" for i, x in enumerate(s))
    if o["tip"] == "Slider":
        return f"slider from {fmt(o['alt_sinir'])} to {fmt(o['ust_sinir'])}"
    return "answer with a number"


def fmt(v) -> str:
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")


def _kisa(s: str, n: int = 90) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " ..."


def twin_hatirlatma(o: dict) -> str:
    return _kisa(o.get("alt") or o.get("soru") or "").rstrip(": ")


def twin_bolum1(ogeler: list[dict], cevap: dict) -> tuple[str, dict]:
    """Part 1: items in questionnaire order, a Matrix question printed once with its rows; returns (text, oge -> Qn)."""
    sat, no, son_qid = [], {}, None
    for i, o in enumerate(ogeler, 1):
        no[o["oge"]] = f"Q{i}"
        v = cevap.get(o["oge"])
        a = "(no answer)" if v is None else f"<<{fmt(v)}>>"
        if o["tip"] == "Matrix":
            if o["qid"] != son_qid:
                sat.append(f"{' '.join(str(o['soru']).split())} ({twin_olcek(o)})")
            sat.append(f"[Q{i}] {' '.join(str(o['alt']).split())}: {a}")
        else:
            alt = f" {' '.join(str(o['alt']).split())}" if o.get("alt") else ""
            sat.append(f"[Q{i}] {' '.join(str(o['soru']).split())}{alt} ({twin_olcek(o)}): {a}")
        son_qid = o["qid"]
    return "\n".join(sat) + "\n", no


TWIN_GIRIS = ("This is the record of one participant in a four-wave online survey study of U.S. adults. "
              "The participant's demographic answers:\n")
TWIN_B1 = "\nPart 1 (waves 1-3). The participant's answers are in double angle brackets.\n"
TWIN_B2 = "\nPart 2 (wave 4, about two weeks later): the same questions were asked again, in a new order.\n"


def twin_metni(uid, demografi: str, ogeler: list[dict], d13: dict, d4: dict | None) -> tuple[str, dict]:
    """d4 None -> stop after the Part 2 header (test context). Part 2 lines '[Qn] reminder: <<answer>>' in a fixed
    per-person shuffled order."""
    b1, no = twin_bolum1(ogeler, d13)
    metin = TWIN_GIRIS + demografi.strip() + "\n" + TWIN_B1 + b1 + TWIN_B2
    if d4 is not None:
        sat = [f"[{no[o['oge']]}] {twin_hatirlatma(o)}: <<{fmt(d4[o['oge']])}>>"
               for o in sirali(ogeler, uid, "d4") if o["oge"] in d4]
        metin += "\n".join(sat) + "\n"
    return metin, no


def twin_test_kaydi(uid, demografi, ogeler, d13) -> dict:
    baglam, no = twin_metni(uid, demografi, ogeler, d13, None)
    hed = [{"anahtar": no[o["oge"]], "oge": o["oge"], "devam": f"[{no[o['oge']]}] {twin_hatirlatma(o)}: <<",
            "aciklama": f"{_kisa(o.get('soru'), 300)} {o.get('alt') or ''} ({twin_olcek(o)})".strip()} for o in ogeler]
    return {"kaynak": "twin", "ayrim": "kisi", "uid": uid, "hedef_donem": "wave 4", "baglam": baglam, "hedefler": hed}


# ------------------------------------------------------------------------------------------------ in-context (API) arm
API_SISTEM = ("You predict what a specific real person did, from their own record. Answer with calibrated probabilities "
              "or values, not with what a typical person would do. Return only the JSON object.")


def api_istemi(kayit: dict, ornekler: list[str] = ()) -> dict:
    """Prompt for the untrained in-context arm (Claude/OpenRouter, no token logprobs): the same context the fine-tuned
    model sees, optionally after complete example transcripts of TRAIN people (a cacheable fixed block), then a verbal
    request. Returns {"system", "user", "anahtarlar"}. Never contains a label: it is built from the test record only."""
    parca = []
    if ornekler:
        parca.append(f"Here are {len(ornekler)} complete records of other people, for reference:\n")
        parca += [f"=== Example record {i} ===\n{t.strip()}\n" for i, t in enumerate(ornekler, 1)]
        parca.append("=== Record to predict ===")
    parca.append(kayit["baglam"].rstrip() + " ...\n")
    if kayit["kaynak"] == "talp73":
        sat = "\n".join(f"{h['anahtar']}: {h['aciklama']}" for h in kayit["hedefler"])
        soru = (f"For {kayit['hedef_donem']}, give for each category below the probability (0-100) that this record "
                f"shows 'yes' (at least one item bought in that period).\n{sat}\n"
                f"Return a JSON object with exactly these {len(kayit['hedefler'])} keys and integer values 0-100.")
    else:
        sat = "\n".join(f"{h['anahtar']}: {h['aciklama']}" for h in kayit["hedefler"])
        soru = (f"Predict this participant's Part 2 answer to each question below (same scale as in Part 1).\n{sat}\n"
                f"Return a JSON object with exactly these {len(kayit['hedefler'])} keys and numeric values.")
    parca.append(soru)
    return {"system": API_SISTEM, "user": "\n".join(parca), "anahtarlar": [h["anahtar"] for h in kayit["hedefler"]]}


def ornek_sec(egitim: list[dict], k: int, tohum: int = 0, kaynak: str | None = None) -> list[str]:
    """k train transcripts (fixed seed) for the in-context block."""
    havuz = sorted((r for r in egitim if kaynak is None or r["kaynak"] == kaynak), key=lambda r: str(r["uid"]))
    return [r["text"] for r in random.Random(SABIT + tohum).sample(havuz, min(k, len(havuz)))]


# ------------------------------------------------------------------------------------------------ leakage check
def sizinti_kontrol(egitim: list[dict], testler: list[dict], etiketler: list[dict], yasak_metin: tuple = ()) -> dict:
    """Asserts: (1) no test person (per source) is in a train record; (2) no train text contains a forbidden period
    string (e.g. 'July - December 2022'); (3) every test context ends at its target header, with no << >> after it;
    (4) every label belongs to a test record and no label key is missing. Returns counts."""
    tr = {(r["kaynak"], str(r["uid"])) for r in egitim}
    te = {(r["kaynak"], str(r["uid"])) for r in testler}
    ortak = tr & te
    assert not ortak, f"test people in the train corpus: {sorted(ortak)[:5]}"
    for r in egitim:
        for y in yasak_metin:
            assert y not in r["text"], f"forbidden period '{y}' in train text of {r['uid']}"
    for r in testler:
        b = r["baglam"]
        bas = b.rfind("Part 2 (wave 4") if r["kaynak"] == "twin" else b.rfind(r["hedef_donem"] + ":")
        assert bas >= 0, f"target header missing for {r['uid']}"
        assert "<<" not in b[bas:], f"a target answer is in the test context of {r['uid']}"
        for y in yasak_metin:
            assert b.count(y) <= 1, f"forbidden period '{y}' repeated in the test context of {r['uid']}"
        for h in r["hedefler"]:
            assert h["devam"].endswith("<<") and "<<" not in h["devam"][:-2], h["devam"]
    hk = {(r["kaynak"], r["ayrim"], str(r["uid"]), h["anahtar"]) for r in testler for h in r["hedefler"]}
    ek = {(e["kaynak"], e["ayrim"], str(e["uid"]), e["anahtar"]) for e in etiketler}
    assert ek == hk, f"labels and targets differ: {len(ek - hk)} extra, {len(hk - ek)} missing"
    return {"egitim_kisi": len(tr), "test_kisi": len(te), "etiket": len(ek)}


# ------------------------------------------------------------------------------------------------ io + stats
def yaz_jsonl(satirlar, yol: str) -> int:
    n = 0
    with open(yol, "w", encoding="utf-8") as f:
        for r in satirlar:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def oku_jsonl(yol: str) -> list[dict]:
    with open(yol, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def egitim_kaydi(kaynak: str, uid, metin: str, deney: str) -> dict:
    """Psych-101 field names (text, experiment, participant) plus ours."""
    return {"text": metin, "experiment": deney, "participant": str(uid), "kaynak": kaynak, "uid": str(uid),
            "n_secim": len(secim_araliklari(metin))}


def say(metinler: list[str], tok=None) -> dict:
    """Records, characters, choices; with a tokenizers.Tokenizer also tokens and choice tokens (tokens whose span
    overlaps a choice content). Without one: tokens ~ characters / 4 (marked yaklasik)."""
    n_sec = sum(len(secim_araliklari(t)) for t in metinler)
    out = {"kayit": len(metinler), "karakter": sum(map(len, metinler)), "secim": n_sec}
    if tok is None:
        out.update(token=round(out["karakter"] / 4), secim_token=None, token_kaynak="yaklasik (karakter/4)")
        return out
    top = sec = 0
    uz = []
    for i in range(0, len(metinler), 512):
        enc = tok.encode_batch(metinler[i:i + 512], add_special_tokens=False)
        for t, e in zip(metinler[i:i + 512], enc):
            ar = secim_araliklari(t)
            top += len(e.ids)
            uz.append(len(e.ids))
            j = 0
            for (a, b) in e.offsets:
                while j < len(ar) and ar[j][1] <= a:
                    j += 1
                if j < len(ar) and a < ar[j][1] and b > ar[j][0]:
                    sec += 1
    uz.sort()
    out.update(token=top, secim_token=sec, token_kaynak=TOKENIZER_REPO,
               token_kayit_medyan=uz[len(uz) // 2] if uz else 0, token_kayit_maks=uz[-1] if uz else 0)
    return out


# throughput assumptions (tok/s, every transcript token through forward+backward), extrapolated, not measured:
# Centaur 70B QLoRA ~590 tok/s on one A100 80GB (253.6M tokens / ~5 days); a 27B dense model ~2.6x fewer FLOPs.
HIZ = {"24GB": (150, 500), "48GB": (400, 900), "80GB": (700, 1500)}


def sure_tahmini(token: int) -> dict:
    return {g: {"tok_s": f"{a}-{b}", "saat": f"{token / b / 3600:.1f}-{token / a / 3600:.1f}"} for g, (a, b) in HIZ.items()}


# ------------------------------------------------------------------------------------------------ loaders (real data)
def talp73_veri():
    """Raw orders (< KESIM only), category names, profiles, people per seed, panel test labels, latent train labels."""
    import pandas as pd
    import polars as pl
    sys.dont_write_bytecode = True        # the TALP repo is read-only: no __pycache__ there
    sys.path.insert(0, f"{TALP}/intent/kod/eval")
    import hazirla_oe as H

    oge = pd.read_parquet(f"{LATENT}/talp73/oge.parquet").sort_values("sira")
    adlar = dict(zip(oge["oge"], oge["ad"]))
    hedef = pd.read_parquet(f"{LATENT}/talp73/hedef.parquet")
    uygun = sorted(hedef["uid"].unique())
    a = (H.oku_alimlar().select("uid", "tarih", "kategori", "adet")
         .filter(pl.col("uid").is_in(uygun) & (pl.col("tarih") < KESIM)))
    assert a["tarih"].max() < KESIM
    a = a.with_columns(((pl.col("tarih").dt.year() - ILK_YIL) * 2 + (pl.col("tarih").dt.month() > 6).cast(pl.Int32))
                       .alias("donem"))
    a20 = a.filter(pl.col("kategori").is_in(list(adlar))).select("uid", "donem", "kategori").unique()
    donemler: dict = {}
    for u, d, k in a20.iter_rows():
        donemler.setdefault(u, {}).setdefault(d, {})[k] = 1
    for u, d in a.select("uid", "donem").unique().iter_rows():        # periods with any order, no 20-category item
        donemler.setdefault(u, {}).setdefault(d, {})
    ad = lambda k: "uncategorized items" if k is None else k.replace("_", " ").lower()  # noqa: E731
    dg = (a.filter(~pl.col("kategori").is_in(list(adlar)) | pl.col("kategori").is_null())
          .group_by("uid", "donem", "kategori").agg(pl.len().alias("n")).sort("uid", "donem", "n", "kategori",
                                                                             descending=[False, False, True, False]))
    diger: dict = {}
    for u, d, k, n in dg.iter_rows():
        x = diger.setdefault(u, {}).setdefault(d, {"kalem": 0, "kategori": 0, "en_cok": []})
        x["kalem"] += n
        x["kategori"] += 1
        if len(x["en_cok"]) < 5:
            x["en_cok"].append((ad(k), n))
    km = pd.read_parquet(f"{LATENT}/talp73/kisi_metin.parquet")
    profil = dict(zip(km["uid"], km["metin"]))
    etiket = {s: pd.read_parquet(f"{PANEL}/etiket_s{s}.parquet") for s in TOHUMLAR}
    return dict(adlar=adlar, hedef=hedef, donemler=donemler, diger=diger, profil=profil, etiket=etiket)


def talp73_uret(V: dict, s: int, kat_disi: list) -> dict:
    adlar, hedef = V["adlar"], V["hedef"]
    h = hedef[hedef["tohum"] == s]
    tr = sorted(h.loc[h["bolum"] == "train", "uid"].unique())
    te = sorted(h.loc[h["bolum"] == "test", "uid"].unique())
    assert not set(tr) & set(te) and len(te) == 400
    # the 2022-H1 choices written for train people must equal the latent train labels (cut 2022-01-01, y = 2022-H1)
    y = h[h["bolum"] == "train"]
    yaz_y = {(u, k): int(V["donemler"].get(u, {}).get(SON_EGITIM_DONEMI, {}).get(k, 0)) for u, k in zip(y["uid"], y["oge"])}
    fark = sum(yaz_y[(u, k)] != int(v) for u, k, v in zip(y["uid"], y["oge"], y["y"]))
    assert fark == 0, f"{fark} train 2022-H1 choices differ from latent hedef"
    kalan = {k: v for k, v in adlar.items() if k not in kat_disi}
    egitim, egitim_kd = [], []
    for u in tr:
        d, g = V["donemler"].get(u, {}), V["diger"].get(u, {})
        egitim.append(egitim_kaydi("talp73", u, talp73_metni(u, V["profil"][u], d, adlar, g), "consumer101/talp73"))
        egitim_kd.append(egitim_kaydi("talp73", u, talp73_metni(u, V["profil"][u], d, kalan, g),
                                      "consumer101/talp73-kategori-disi"))
    for r, n in [(r, len(adlar)) for r in egitim] + [(r, len(kalan)) for r in egitim_kd]:
        assert r["n_secim"] % n == 0 and r["text"].count("<<") == r["n_secim"], r["uid"]   # no stray << >> in prose
    test = {"kisi": [], "kapak": [], "kategori": []}
    for u in te:
        d, g = V["donemler"].get(u, {}), V["diger"].get(u, {})
        test["kisi"].append(talp73_test_kaydi(u, s, V["profil"][u], d, adlar, g, "kisi"))
        test["kapak"].append(talp73_test_kaydi(u, s, V["profil"][u], d, KAPAK, g, "kapak", "kapak"))
        test["kategori"].append(talp73_test_kaydi(u, s, V["profil"][u], d, adlar, g, "kategori", sorulan=kat_disi))
    e = V["etiket"][s]
    assert sorted(e["uid"].unique()) == te
    ey = {(u, k): int(v) for u, k, v in zip(e["uid"], e["kategori"], e["y"])}
    etiket = {a: [{"kaynak": "talp73", "ayrim": a, "tohum": s, "uid": r["uid"], "anahtar": hh["anahtar"],
                   "oge": hh["oge"], "y": ey[(r["uid"], hh["oge"])]} for r in test[a] for hh in r["hedefler"]]
              for a in test}
    return dict(egitim=egitim, egitim_kd=egitim_kd, test=test, etiket=etiket, tr=tr, te=te)


def twin_uret() -> dict:
    import pandas as pd
    og = pd.read_parquet(f"{LATENT}/twin/oge.parquet")
    sira = {o: i for i, o in enumerate(og["oge"])}
    ogd = {r["oge"]: r for r in og.to_dict("records")}
    h = pd.read_parquet(f"{LATENT}/twin/hedef.parquet")
    g = pd.read_parquet(f"{LATENT}/twin/gecmis.parquet")
    km = pd.read_parquet(f"{LATENT}/twin/kisi_metin.parquet")
    demo = dict(zip(km["uid"], km["metin_demografi"]))
    d13 = {}
    for u, o, v in zip(g["uid"], g["oge"], g["y_d13"]):
        if pd.notna(v):
            d13.setdefault(u, {})[o] = float(v)
    egitim, test, etiket = [], [], []
    tr = sorted(h.loc[h["bolum"] == "train", "uid"].unique())
    te = sorted(h.loc[h["bolum"] == "test", "uid"].unique())
    assert not set(tr) & set(te)
    for (u, b), d in h.sort_values(["uid", "oge"]).groupby(["uid", "bolum"]):
        ogeler = [ogd[o] for o in sorted(d["oge"], key=sira.get)]
        d4 = dict(zip(d["oge"], d["y"].astype(float)))
        if b == "train":
            egitim.append(egitim_kaydi("twin", u, twin_metni(u, demo[u], ogeler, d13.get(u, {}), d4)[0],
                                       "consumer101/twin2k500"))
        else:
            r = twin_test_kaydi(u, demo[u], ogeler, d13.get(u, {}))
            test.append(r)
            y01 = dict(zip(d["oge"], d["y01"]))
            etiket += [{"kaynak": "twin", "ayrim": "kisi", "uid": str(u), "anahtar": hh["anahtar"], "oge": hh["oge"],
                        "y": d4[hh["oge"]], "y01": float(y01[hh["oge"]])} for hh in r["hedefler"]]
    for r in test:
        r["uid"] = str(r["uid"])
    return dict(egitim=egitim, test=test, etiket=etiket, tr=tr, te=te)


def psych_uret() -> tuple[list[dict], dict]:
    from huggingface_hub import hf_hub_download
    yol = hf_hub_download(PSYCH_REPO, "prompts_training.jsonl", repo_type="dataset")
    gruplar: dict = {}
    toplam = 0
    with open(yol, encoding="utf-8") as f:
        for i, x in enumerate(f):
            d = json.loads(x)
            toplam += 1
            if len(d["text"]) <= PSYCH_MAKS_KARAKTER:
                gruplar.setdefault(d["experiment"], []).append((i, d))
    out = []
    for e in sorted(gruplar):
        sec = random.Random(_tohum("psych", e)).sample(gruplar[e], min(PSYCH_KISI, len(gruplar[e])))
        for i, d in sorted(sec, key=lambda t: t[0]):
            r = egitim_kaydi("psych101", f"{e}#{d['participant']}", d["text"], f"psych101/{e}")
            r["participant"] = str(d["participant"])
            out.append(r)
    bilgi = {"kaynak_satir": toplam, "deney": len(gruplar), "secilen": len(out), "kisi_deney_basi": PSYCH_KISI,
             "maks_karakter": PSYCH_MAKS_KARAKTER, "lisans": "Apache-2.0 (dataset card)"}
    return out, bilgi


def tokenizer_yukle():
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        return Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, "tokenizer.json"))
    except Exception as e:  # noqa: BLE001  network / package problems: fall back to the estimate, say so
        print(f"tokenizer unavailable ({e}); token counts are estimates", file=sys.stderr)
        return None


def sinir_kontrol(tok) -> dict:
    """Does a prompt ending in '<<' tokenize as a prefix of the full answer line, and are yes/no single tokens?"""
    if tok is None:
        return {}
    out = {}
    for on in ("Books: <<", "[Q3] How likely: <<"):
        p = tok.encode("x\n" + on, add_special_tokens=False).ids
        for a in ("yes", "no", "4", "55"):
            t = tok.encode("x\n" + on + a + ">>", add_special_tokens=False).ids
            out[f"{on}{a}"] = {"onek": t[:len(p)] == p, "cevap_token": [tok.id_to_token(i) for i in t[len(p):]]}
    return out


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cikti", default=CIKTI)
    ap.add_argument("--tohum", type=int, nargs="*", default=list(TOHUMLAR))
    ap.add_argument("--token-yok", action="store_true", help="skip the tokenizer (estimate tokens as chars/4)")
    arg = ap.parse_args(argv)
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.makedirs(arg.cikti, exist_ok=True)
    yol = lambda ad: os.path.join(arg.cikti, ad)  # noqa: E731
    tok = None if arg.token_yok else tokenizer_yukle()
    ozet: dict = {"sabit_tohum": SABIT, "kesim": str(KESIM), "dosyalar": {}, "istatistik": {}, "sizinti": {}}

    def kaydet(ad, satirlar, egitim=False):
        n = yaz_jsonl(satirlar, yol(ad))
        ozet["dosyalar"][ad] = n
        if egitim:
            ozet["istatistik"][ad] = say([r["text"] for r in satirlar], tok)
        return n

    psych, ozet["psych101"] = psych_uret()
    kaydet("psych101_alt.jsonl", psych, True)
    tw = twin_uret()
    kaydet("twin_train.jsonl", tw["egitim"], True)
    kaydet("twin_test_kisi.jsonl", tw["test"])
    kaydet("twin_test_kisi_etiket.jsonl", tw["etiket"])
    ozet["sizinti"]["twin"] = sizinti_kontrol(tw["egitim"], tw["test"], tw["etiket"])
    ozet["istatistik"]["twin_test_kisi.jsonl"] = say([r["baglam"] for r in tw["test"]], tok)

    V = talp73_veri()
    kat_disi = sorted(random.Random(SABIT).sample(sorted(V["adlar"]), N_KATEGORI_DISI))
    ozet["kategori_disi"] = kat_disi
    yasak = (donem_etiketi(HEDEF_DONEMI), "2022-07", "2022-08", "2022-09", "2022-10", "2022-11", "2022-12")
    gorev_disi = tw["egitim"] + psych
    karisim = sorted(gorev_disi, key=lambda r: (r["kaynak"], r["uid"]))
    random.Random(SABIT).shuffle(karisim)
    kaydet("karisim_gorev_disi.jsonl", karisim, True)
    for s in arg.tohum:
        T = talp73_uret(V, s, kat_disi)
        kaydet(f"talp73_s{s}_train.jsonl", T["egitim"], True)
        kaydet(f"talp73_s{s}_train_kategori_disi.jsonl", T["egitim_kd"], True)
        for a in ("kisi", "kapak", "kategori"):
            kaydet(f"talp73_s{s}_test_{a}.jsonl", T["test"][a])
            kaydet(f"talp73_s{s}_test_{a}_etiket.jsonl", T["etiket"][a])
            ozet["istatistik"][f"talp73_s{s}_test_{a}.jsonl"] = say([r["baglam"] for r in T["test"][a]], tok)
        hepsi_test = [r for a in T["test"] for r in T["test"][a]]
        hepsi_et = [e for a in T["etiket"] for e in T["etiket"][a]]
        ozet["sizinti"][f"talp73_s{s}"] = sizinti_kontrol(T["egitim"] + T["egitim_kd"] + gorev_disi, hepsi_test,
                                                          hepsi_et, yasak)
        # held-out kapak names never appear in a train file; held-out categories never in the kategori_disi file
        for r in T["egitim"] + T["egitim_kd"]:
            assert not any(f"{v}: <<" in r["text"] for v in KAPAK.values())
        for r in T["egitim_kd"]:
            assert not any(f"{V['adlar'][k]}: <<" in r["text"] for k in kat_disi)
        for ad, parca in (("tum", T["egitim"] + gorev_disi), ("kategori_disi", T["egitim_kd"] + gorev_disi)):
            m = sorted(parca, key=lambda r: (r["kaynak"], r["uid"]))
            random.Random(SABIT + s).shuffle(m)
            kaydet(f"karisim_s{s}_{ad}.jsonl", m, True)
    ozet["sinir"] = sinir_kontrol(tok)
    ozet["sure_tahmini"] = {ad: sure_tahmini(v["token"]) for ad, v in ozet["istatistik"].items()
                            if ad.startswith(("karisim", "talp73_s0_train.jsonl"))}
    ozet["hiz_varsayimi"] = {"not": "extrapolated from Centaur (~590 tok/s, 70B QLoRA, A100 80GB), not measured",
                             **{g: f"{a}-{b} tok/s" for g, (a, b) in HIZ.items()}}
    with open(yol("ozet.json"), "w") as f:
        json.dump(ozet, f, indent=1, ensure_ascii=False)
    return ozet


if __name__ == "__main__":
    o = main()
    print(json.dumps({k: o[k] for k in ("dosyalar", "sizinti")}, indent=1))
