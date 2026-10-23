"""Centaur-style (Q)LoRA on Consumer-101 and its held-out evaluation, for the GPU box. NOT run on the CPU server.

Recipe (Binz et al., Nature 2025; docs/centaur/NOTES.md 1.2): frozen base (4-bit nf4 by default), LoRA r 8 on every
linear layer of the language model, 1 epoch, lr 5e-5, weight decay 0.01, 8-bit AdamW, effective batch 32 records,
linear warm-up. The loss is cross-entropy on the tokens INSIDE << >> only (korpus.secim_araliklari); logits are computed
only at those positions (logits_to_keep), so long transcripts do not materialise a full vocabulary x length tensor.

    export PYTHONPATH=$PWD HF_HOME=/data/talp/hf; PY="nice -n 19 /data/talp/venv/bin/python"
    $PY runs/centaur/qlora.py olc   --veri talp73_s0_train_kategori_disi                 # 12 steps: tok/s, memory, ETA
    $PY runs/centaur/qlora.py egit  --veri talp73_s0_train_kategori_disi --cikti DIR --bitis 03:30
    $PY runs/centaur/qlora.py degerlendir --adapter DIR/adapter --etiket kd --ayrim kisi kapak kategori
    $PY runs/centaur/qlora.py degerlendir --adapter yok --etiket taban --ayrim gorev          # base model, zero-shot
    bash runs/centaur/gece_qlora.sh                                                         # the whole night, in order
    # CPU smoke (Qwen3.5-2B fp32, a few records, short contexts): --kucuk

Evaluation (korpus README): every target is its own prompt baglam + devam ("Books: <<"); P(yes) = softmax over the
next-token logits of "yes" and "no". The context is run once per person and its cache is copied for each target.
Twin targets: MC/Matrix items score every option "k>>" and take the expected code; other items are generated
greedily up to ">>". Output: <cikti>/<ayrim>__qlora-<etiket>__s0.parquet in the format of runs/centaur/heldout.py,
scored there with `heldout.py skor --qlora <cikti>`.

Seed discipline (korpus README): a model trained on a seed-s file is evaluated only on talp73_s{s}_test_* files.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("HF_HOME", "/data/talp/hf")

import heldout as H                                         # noqa: E402  (same folder; korpus loader + paths)

K = H.K
KORPUS = H.KORPUS
CIKTI = "/data/talp/is/centaur/qlora"


# ------------------------------------------------------------------------------------------------ model
def yukle(kesinlik: str, kucuk: bool):
    from simengine.persona.latent import models as M
    if kucuk:
        return M.load("q35s", "fp32", device="cpu")
    return M.load("q38", kesinlik, device="cuda")


def lora_ekle(model, r: int, alpha: int, kesinlik: str):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if kesinlik in ("nf4", "int8"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules="all-linear", exclude_modules=r".*(visual|vision|mtp|lm_head).*")
    model = get_peft_model(model, cfg)
    model.config.use_cache = False
    return model


# ------------------------------------------------------------------------------------------------ data
def kodla(tok, metin: str, max_len: int) -> dict | None:
    """input_ids and the positions of choice tokens (any overlap with a << >> content span). Records longer than
    max_len are cut at max_len (instructions and early trials kept); a record with no choice left is dropped."""
    enc = tok(metin, return_offsets_mapping=True, add_special_tokens=False)
    ids, off = enc["input_ids"], enc["offset_mapping"]
    araliklar = K.secim_araliklari(metin)
    hedef, j = [], 0
    for i, (a, b) in enumerate(off):
        while j < len(araliklar) and araliklar[j][1] <= a:
            j += 1
        if j < len(araliklar) and a < araliklar[j][1] and b > araliklar[j][0]:
            hedef.append(i)
    ids, hedef = ids[:max_len], [i for i in hedef if 0 < i < max_len]
    return {"ids": ids, "hedef": hedef} if hedef else None


def veri_oku(ad: str, tok, max_len: int, limit: int | None, tohum: int = 0) -> list[dict]:
    yol = ad if os.path.isabs(ad) else f"{KORPUS}/{ad}.jsonl"
    kay = K.oku_jsonl(yol)
    random.Random(K.SABIT + tohum).shuffle(kay)
    if limit:
        kay = kay[:limit]
    out = []
    for r in kay:
        x = kodla(tok, r["text"], max_len)
        if x:
            out.append(x)
    return out


# ------------------------------------------------------------------------------------------------ training
def kayip(model, ornek: dict, device):
    import torch
    ids = torch.tensor([ornek["ids"]], device=device)
    hedef = torch.tensor(ornek["hedef"], device=device)
    poz = hedef - 1                                            # logits at t-1 predict token t
    out = model(input_ids=ids, logits_to_keep=poz, use_cache=False)
    logits = out.logits[0].float()
    assert logits.shape[0] == len(ornek["hedef"]), "logits_to_keep did not select the choice positions"
    return torch.nn.functional.cross_entropy(logits, ids[0, hedef], reduction="mean")


def zaman_siniri(bitis: str | None) -> datetime | None:
    if not bitis:
        return None
    h, m = map(int, bitis.split(":"))
    now = datetime.now()
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > now else t + timedelta(days=1)


def egit(a, olc: bool = False):
    import torch
    from transformers import get_linear_schedule_with_warmup
    ld = yukle(a.kesinlik, a.kucuk)
    tok, device = ld.tok, ld.device
    veri = veri_oku(a.veri, tok, a.max_len, a.limit)
    n_tok = sum(len(x["ids"]) for x in veri)
    n_adim = max(1, math.ceil(len(veri) / a.etkin_batch))
    model = lora_ekle(ld.model, a.r, a.alpha, a.kesinlik)
    model.train()
    eg = [p for p in model.parameters() if p.requires_grad]
    print(f"records {len(veri)}, tokens {n_tok:,}, choice tokens {sum(len(x['hedef']) for x in veri):,}, "
          f"optimizer steps {n_adim}, trainable params {sum(p.numel() for p in eg):,}", flush=True)
    if a.kucuk or not torch.cuda.is_available():
        opt = torch.optim.AdamW(eg, lr=a.lr, weight_decay=a.wd)
    else:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(eg, lr=a.lr, weight_decay=a.wd)
    warm = a.warmup if a.warmup >= 1 else max(1, int(a.warmup * n_adim))
    sch = get_linear_schedule_with_warmup(opt, warm, n_adim)
    sinir = zaman_siniri(a.bitis)
    os.makedirs(a.cikti, exist_ok=True)
    log = open(os.path.join(a.cikti, "egitim_log.jsonl"), "a")
    t0, tok_say, adim, toplam = time.time(), 0, 0, 0.0
    durdu = None
    for i, x in enumerate(veri):
        loss = kayip(model, x, device) / a.etkin_batch
        loss.backward()
        toplam += loss.item() * a.etkin_batch
        tok_say += len(x["ids"])
        if (i + 1) % a.etkin_batch == 0 or i + 1 == len(veri):
            torch.nn.utils.clip_grad_norm_(eg, 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            adim += 1
            dt = time.time() - t0
            hiz = tok_say / dt
            kalan = (n_tok - tok_say) / hiz
            kayit = {"adim": adim, "kayit": i + 1, "loss": toplam / ((i % a.etkin_batch) + 1), "tok_s": round(hiz, 1),
                     "gecen_sa": round(dt / 3600, 3), "kalan_sa": round(kalan / 3600, 2), "lr": sch.get_last_lr()[0],
                     "bellek_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None}
            log.write(json.dumps(kayit) + "\n")
            log.flush()
            print(kayit, flush=True)
            toplam = 0.0
            if adim % a.kaydet_her == 0:
                model.save_pretrained(os.path.join(a.cikti, "adapter"))
            if olc and adim >= a.olc_adim:
                break
            if sinir and datetime.now() + timedelta(minutes=2) >= sinir:
                durdu = f"time limit {a.bitis} reached at step {adim}/{n_adim}"
                print(durdu, flush=True)
                break
    if not olc:
        model.save_pretrained(os.path.join(a.cikti, "adapter"))
    ozet = {"veri": a.veri, "kesinlik": a.kesinlik, "kayit": len(veri), "token": n_tok, "adim": adim, "n_adim": n_adim,
            "tok_s": round(tok_say / (time.time() - t0), 1), "sure_sa": round((time.time() - t0) / 3600, 3),
            "durdu": durdu, "tamam_oran": round(tok_say / n_tok, 4), "r": a.r, "alpha": a.alpha, "lr": a.lr,
            "wd": a.wd, "etkin_batch": a.etkin_batch, "warmup_adim": warm, "max_len": a.max_len,
            "taban": "Qwen/Qwen3.8-27B" if not a.kucuk else "Qwen/Qwen3.5-2B (kucuk)", "olc": olc}
    if olc:                                  # whole file at the measured speed (first steps include warm-up noise)
        ozet["tahmini_tam_sure_sa"] = round(n_tok / max(ozet["tok_s"], 1e-9) / 3600, 2)
    json.dump(ozet, open(os.path.join(a.cikti, "olc.json" if olc else "egitim_ozet.json"), "w"), indent=1)
    print(ozet, flush=True)
    return model, tok


# ------------------------------------------------------------------------------------------------ evaluation
def tek_token(tok, s: str) -> int:
    ids = tok(s, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, f"{s!r} is not one token: {ids}"
    return ids[0]


def ortak_onek(a: list, b: list) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def kisi_hesapla(model, tok, kayit: dict, device, olcek: dict | None = None) -> list[dict]:
    """Prefix forward once; each target continues from a deep copy of the cache (targets never see each other)."""
    import torch
    tam = [tok(kayit["baglam"] + h["devam"], add_special_tokens=False)["input_ids"] for h in kayit["hedefler"]]
    taban = tok(kayit["baglam"], add_special_tokens=False)["input_ids"]
    L = min(ortak_onek(taban, t) for t in tam)
    L = min(L, min(len(t) for t in tam) - 1)                   # at least one continuation token per target
    with torch.no_grad():
        out = model(input_ids=torch.tensor([tam[0][:L]], device=device), use_cache=True, logits_to_keep=1)
        onbellek = out.past_key_values
        sonuc = []
        for h, t in zip(kayit["hedefler"], tam):
            devam = torch.tensor([t[L:]], device=device)
            o = model(input_ids=devam, past_key_values=copy.deepcopy(onbellek), use_cache=True, logits_to_keep=1)
            lp = torch.log_softmax(o.logits[0, -1].float(), -1)
            if kayit["kaynak"] == "talp73":
                y, n = lp[tek_token(tok, "yes")].item(), lp[tek_token(tok, "no")].item()
                sonuc.append({"uid": str(kayit["uid"]), "anahtar": h["anahtar"], "p": 1 / (1 + math.exp(n - y)),
                              "kutle_yes_no": math.exp(y) + math.exp(n)})
            else:
                sonuc.append({"uid": str(kayit["uid"]), "anahtar": h["anahtar"], "oge": h["oge"],
                              **twin_oku(model, tok, o, h, olcek[h["oge"]], device)})
    return sonuc


def twin_oku(model, tok, o, h: dict, olcek: tuple, device) -> dict:
    """MC/Matrix ('options: 1 = ...'): score every 'k>>' continuation, expected code. Otherwise greedy to '>>'."""
    import re
    import torch
    lo, hi = olcek
    m = re.findall(r"(\d+) = ", h["aciklama"].split("options:", 1)[1]) if "options:" in h["aciklama"] else []
    if m:
        lps = []
        for k in m:
            ids = tok(f"{k}>>", add_special_tokens=False)["input_ids"]
            c = copy.deepcopy(o.past_key_values)
            lp = torch.log_softmax(o.logits[0, -1].float(), -1)[ids[0]].item()
            if len(ids) > 1:
                oo = model(input_ids=torch.tensor([ids[:-1]], device=device), past_key_values=c, use_cache=True)
                lsm = torch.log_softmax(oo.logits[0].float(), -1)
                lp += sum(lsm[i, ids[i + 1]].item() for i in range(len(ids) - 1))
            lps.append(lp)
        w = [math.exp(x - max(lps)) for x in lps]
        ev = sum(int(k) * x for k, x in zip(m, w)) / sum(w)
        return {"ham": ev, "y01": (ev - lo) / (hi - lo) if hi > lo else float("nan")}
    c, nxt, metin = copy.deepcopy(o.past_key_values), int(o.logits[0, -1].argmax()), ""
    for _ in range(8):
        metin += tok.decode([nxt])
        if ">>" in metin:
            break
        oo = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=c, use_cache=True)
        nxt = int(oo.logits[0, -1].argmax())
    sayi = re.findall(r"-?\d+(?:\.\d+)?", metin.split(">>")[0])
    v = float(sayi[0]) if sayi else float("nan")
    return {"ham": v, "y01": min(max((v - lo) / (hi - lo), 0.0), 1.0) if (hi > lo and sayi) else float("nan")}


def degerlendir(a, model=None, tok=None):
    import pandas as pd
    import torch
    if model is None:
        ld = yukle(a.kesinlik, a.kucuk)
        model, tok = ld.model, ld.tok
        if a.adapter and a.adapter != "yok":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    olcek = H.twin_olcek_haritasi()
    os.makedirs(a.cikti, exist_ok=True)
    for ayrim in a.ayrim:
        kay = K.oku_jsonl(H.test_dosyasi(ayrim, 0))
        if a.n:
            sec = set(H.orneklem([r["uid"] for r in kay], a.n))
            kay = [r for r in kay if r["uid"] in sec]
        if a.kucuk:                                  # CPU smoke: 2 people, 3 targets each
            kay = [dict(r, hedefler=r["hedefler"][:3]) for r in kay[:2]]
        t0, rows = time.time(), []
        for i, r in enumerate(kay):
            rows += kisi_hesapla(model, tok, r, device, olcek)
            if (i + 1) % 25 == 0:
                print(f"{ayrim}: {i + 1}/{len(kay)} people, {time.time() - t0:.0f}s", flush=True)
        yol = os.path.join(a.cikti, f"{ayrim}__qlora-{a.etiket}__s0.parquet")
        pd.DataFrame(rows).to_parquet(yol, index=False)
        print(f"{ayrim}: {len(rows)} targets -> {yol} ({time.time() - t0:.0f}s)", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for ad in ("olc", "egit", "degerlendir"):
        p = sp.add_parser(ad)
        p.add_argument("--kesinlik", default="nf4", choices=["nf4", "int8", "bf16"])
        p.add_argument("--kucuk", action="store_true", help="CPU smoke with Qwen3.5-2B fp32")
        p.add_argument("--cikti", default=CIKTI)
        if ad in ("olc", "egit"):
            p.add_argument("--veri", required=True, help="korpus file name without .jsonl, or an absolute path")
            p.add_argument("--max-len", type=int, default=16384)
            p.add_argument("--limit", type=int, default=None)
            p.add_argument("--r", type=int, default=8)
            p.add_argument("--alpha", type=int, default=16)
            p.add_argument("--lr", type=float, default=5e-5)
            p.add_argument("--wd", type=float, default=0.01)
            p.add_argument("--etkin-batch", type=int, default=32)
            p.add_argument("--warmup", type=float, default=0.1, help=">=1: steps; <1: fraction of the steps")
            p.add_argument("--kaydet-her", type=int, default=25)
            p.add_argument("--bitis", default=None, help="HH:MM wall clock: save and stop training before it")
            p.add_argument("--olc-adim", type=int, default=12)
        if ad in ("egit", "degerlendir"):
            p.add_argument("--ayrim", nargs="*", default=[], choices=list(H.AYRIMLAR))
            p.add_argument("--n", type=int, default=None, help="people per held-out set (default all 400)")
            p.add_argument("--etiket", default="run")
        if ad == "degerlendir":
            p.add_argument("--adapter", required=True, help="adapter folder, or 'yok' for the base model")
    a = ap.parse_args(argv)
    if a.cmd == "olc":
        egit(a, olc=True)
    elif a.cmd == "egit":
        model, tok = egit(a)
        if a.ayrim:
            degerlendir(a, model, tok)
    else:
        degerlendir(a)


if __name__ == "__main__":
    main()
