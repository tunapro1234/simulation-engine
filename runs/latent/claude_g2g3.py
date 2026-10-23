"""Claude prompt arms for G2 (Twin wave 4 items) and G3 (Twin price counterfactual), same 400 test persons as kos.py.

    PY="nice -n 19 /data/talp/venv/bin/python"
    $PY runs/latent/claude_g2g3.py --model claude-haiku-5-5 --kol persona yok gecmis --n 400
    $PY runs/latent/claude_g2g3.py --model claude-sonnet-5-5 --kol persona yok --n 80 --usd 11

Arms (kol):
  persona  system = the full waves 1-3 persona text (kisi_metin.metin, ~25k tokens, the same text the dataset's GPT-4.1
           "Text Persona" runs used; no wave 4 item inside). G3 adds the person's 30 history-product answers (the same
           input the classic lgbm-tf / enc-tf get).
  yok      no persona, no history ("a randomly chosen US adult"): the persona-free control.
  gecmis   G3 only: history-product answers, no persona text.
Per person, sequential calls (the persona system block is prompt-cached across them): G2 = every target item of the
person in one call (option number / slider / number), G3 obs = P(buy) 0-100 for the 10 target products at the
person's own prices, G3 sweep = P(buy) for each product at the 11 price levels.
Output: <base>/twin/tahmin/cl-<m>-<kol>_s0.parquet (uid, oge, p in [0,1]) and
<base>/twin_fiyat/tahmin/cl-<m>-<kol>_s0.parquet (uid, oge, carpan_duzey, p, tur test|sweep), + .meta.json with cost.
Calls go through the TALP claude_api module (key file, response cache, spend log, global stop); no temperature sent.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/srv/lighthouse/hackathon-coz/projeler/talp-simulation-hackathon-2026/uretim/talp73/kod/llm")
import claude_api as ca  # noqa: E402

BASE = Path("/data/talp/is/latent")
VERI = BASE / "veri"
TAG = {"claude-haiku-5-5": "h55", "claude-sonnet-5-5": "s55"}

SYS_PERSONA = ("You simulate how one specific real US adult answers survey questions. Below are this person's own answers "
               "to an earlier survey. Answer as this person would actually answer, with their own views, habits, "
               "knowledge, biases and mistakes, not as an ideal or average respondent.\n\n=== THIS PERSON'S EARLIER "
               "SURVEY ANSWERS ===\n{metin}")
SYS_HIST = ("You simulate how one specific real US adult answers shopping survey questions. Answer as this person "
            "would actually answer, not as an average shopper.")
SYS_NONE = ("You simulate how a randomly chosen US adult answers survey questions. Answer as a typical real respondent "
            "would, not as an ideal reasoner.")


def jparse(t: str) -> dict:
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        raise ValueError("no JSON object")
    return json.loads(m.group(0))


# ------------------------------------------------------------------------------------------------- G2
def g2_text(items: pd.DataFrame) -> str:
    out = ["Answer each question below as this person. Questions in the same block were shown together, in order.",
           "Return only one JSON object mapping each question id to the answer: for questions with numbered options "
           "the option number; for a slider a number in its range; for a number question a number.", ""]
    for blok, g in items.groupby("blok", sort=False):
        out.append(f"--- block: {blok} ---")
        for r in g.itertuples():
            q = r.soru + (f"\n  Statement: {r.alt}" if r.alt else "")
            if r.tip in ("MC", "Matrix"):
                opts = json.loads(r.secenekler)
                q += "\n  Options: " + "; ".join(f"{i + 1} = {o}" for i, o in enumerate(opts))
            elif r.tip == "Slider":
                q += f"\n  Slider from {r.alt_sinir:g} to {r.ust_sinir:g}."
            else:
                q += "\n  Enter a number."
            out.append(f"[{r.oge}] {q}")
        out.append("")
    return "\n".join(out)


def g2_check(ids):
    def f(t):
        d = jparse(t)
        v = {k: float(d[k]) for k in ids if k in d and d[k] is not None and str(d[k]).strip() != ""}
        if len(v) < 0.9 * len(ids):
            raise ValueError(f"only {len(v)}/{len(ids)} answers")
        return v
    return f


def g2_scale(items: pd.DataFrame, oge: str, v: float) -> float:
    r = items.loc[oge]
    if r.tip in ("MC", "Matrix"):
        v = min(max(round(v), 1), int(r.n_secenek))
    return float(np.clip((v - r.olcek_alt) / (r.olcek_ust - r.olcek_alt), 0, 1))


# ------------------------------------------------------------------------------------------------- G3
def tf_line(cat, prod, price):
    return (f"Product category: {cat}. Suppose you are in a grocery store, and you see the following product in that "
            f"category: {prod}. The product is priced at: ${price:.2f}. Would you or would you not purchase it?")


def hist_text(h: pd.DataFrame, it: pd.DataFrame) -> str:
    return "\n".join(f"- {tf_line(it.at[o, 'kategori'], it.at[o, 'urun'], f)} -> "
                     f"{'Yes, would purchase' if a else 'No, would not purchase'}"
                     for o, f, a in zip(h.oge, h.fiyat, h.al))


def g3_obs_text(rows: pd.DataFrame, it, hist: str | None) -> str:
    head = (f"Earlier, this person answered these shopping questions (same kind of question, other products):\n{hist}\n\n"
            if hist else "")
    qs = "\n".join(f"[{o}] {tf_line(it.at[o, 'kategori'], it.at[o, 'urun'], f)}" for o, f in zip(rows.oge, rows.fiyat))
    return (head + "For each product below, give the probability (0-100) that this person answers \"Yes, I would "
            "purchase the product\". Each question was asked on its own, at the stated price.\n\n" + qs +
            "\n\nReturn only one JSON object mapping each product id to the probability, e.g. {\"P01\": 40, ...}.")


def g3_sweep_text(prods, levels, price, it, hist: str | None) -> str:
    head = (f"Earlier, this person answered these shopping questions (same kind of question, other products):\n{hist}\n\n"
            if hist else "")
    qs = "\n".join(f"[{o}] category {it.at[o, 'kategori']}: {it.at[o, 'urun']}. Prices: " +
                   ", ".join(f"${price[(o, l)]:.2f}" for l in levels) for o in prods)
    return (head + "Suppose this person is in a grocery store and sees each product below in its category. For each "
            "product and EACH listed price separately, give the probability (0-100) that they would answer \"Yes, I "
            "would purchase the product\" if it were priced at that price.\n\n" + qs +
            f"\n\nReturn only one JSON object mapping each product id to a list of {len(levels)} probabilities in the "
            "order of its prices, e.g. {\"P01\": [90, 85, ...], ...}.")


def obs_check(ids):
    def f(t):
        d = jparse(t)
        v = {k: float(d[k]) for k in ids if k in d}
        if len(v) < len(ids):
            raise ValueError(f"only {len(v)}/{len(ids)}")
        return v
    return f


def sweep_check(ids, n):
    def f(t):
        d = jparse(t)
        v = {k: [float(x) for x in d[k]] for k in ids if k in d and len(d[k]) == n}
        if len(v) < len(ids):
            raise ValueError(f"only {len(v)}/{len(ids)} full lists")
        return v
    return f


# ------------------------------------------------------------------------------------------------- run
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-5-5", choices=sorted(TAG))
    ap.add_argument("--kol", nargs="+", default=["persona", "yok"], choices=("persona", "yok", "gecmis"))
    ap.add_argument("--gorev", nargs="+", default=["g2", "g3"], choices=("g2", "g3"))
    ap.add_argument("--n", type=int, default=400, help="test persons (first n of a fixed shuffle, rng 1)")
    ap.add_argument("--paralel", type=int, default=8)
    ap.add_argument("--usd", type=float, default=15.0, help="per arm: stop opening persons once the arm's spend reaches it")
    a = ap.parse_args(argv)

    it2 = pd.read_parquet(VERI / "twin/oge.parquet").set_index("oge")
    h2 = pd.read_parquet(VERI / "twin/hedef.parquet")
    h2 = h2[h2.bolum == "test"]
    metin = pd.read_parquet(VERI / "twin/kisi_metin.parquet").set_index("uid").metin
    it3 = pd.read_parquet(VERI / "twin_fiyat/oge.parquet").set_index("oge")
    h3 = pd.read_parquet(VERI / "twin_fiyat/hedef.parquet")
    h3 = h3[h3.bolum == "test"]
    g3h = pd.read_parquet(VERI / "twin_fiyat/gecmis.parquet")
    grid = pd.read_parquet(VERI / "twin_fiyat/fiyat_izgara.parquet")
    levels = sorted(grid.carpan_duzey.unique().tolist())
    price = {(o, l): float(f) for o, l, f in zip(grid.oge, grid.carpan_duzey, grid.fiyat)}
    prods = sorted(grid.oge.unique())
    order2 = list(it2.index)

    test = sorted(h2.uid.unique())
    persons = [test[i] for i in np.random.default_rng(1).permutation(len(test))[: a.n]]
    tag = TAG[a.model]
    spent = {"usd": 0.0, "calls": 0, "cached": 0, "fail": 0}

    def call(system, user, etiket, uid, dogrula, mt):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        r = ca.call(msgs, a.model, etiket=etiket, ornek=int(uid), dogrula=dogrula, max_tokens=mt, effort="low",
                    dusunme="kapali")
        spent["calls"] += 1
        spent["cached"] += bool(r.get("onbellek"))
        if not r.get("onbellek"):
            spent["usd"] += float(r.get("usd") or 0)
        return r["veri"]

    for kol in a.kol:
        gor = [g for g in a.gorev if not (kol == "gecmis" and g == "g2")]
        spent.update(usd=0.0, calls=0, cached=0, fail=0)          # --usd is per arm
        rows2, rows3 = [], []

        def one(uid):
            if spent["usd"] >= a.usd:
                return "budget"
            if kol == "persona":
                system = SYS_PERSONA.format(metin=metin[uid])
            else:
                system = SYS_HIST if kol == "gecmis" else SYS_NONE
            hist = None
            if kol in ("persona", "gecmis"):
                hist = hist_text(g3h[g3h.uid == uid], it3)
            out2, out3 = [], []
            if "g2" in gor:
                mine = h2[h2.uid == uid].oge.tolist()
                items = it2.loc[[o for o in order2 if o in set(mine)]].reset_index()
                v = call(system, g2_text(items), f"latent-g2g3:{tag}-{kol}:g2", uid, g2_check(items.oge.tolist()), 3000)
                out2 = [(uid, o, g2_scale(it2, o, x)) for o, x in v.items()]
            if "g3" in gor:
                rows = h3[h3.uid == uid].sort_values("oge")
                v = call(system, g3_obs_text(rows, it3, hist), f"latent-g2g3:{tag}-{kol}:g3obs", uid,
                         obs_check(rows.oge.tolist()), 800)
                out3 += [(uid, o, float(l), min(max(v[o], 0), 100) / 100, "test")
                         for o, l in zip(rows.oge, rows.carpan_duzey)]
                v = call(system, g3_sweep_text(prods, levels, price, it3, hist), f"latent-g2g3:{tag}-{kol}:g3sw", uid,
                         sweep_check(prods, len(levels)), 2500)
                out3 += [(uid, o, float(l), min(max(x, 0), 100) / 100, "sweep")
                         for o in prods for l, x in zip(levels, v[o])]
            return out2, out3

        t0 = time.time()
        with ThreadPoolExecutor(a.paralel) as ex:
            fs = {ex.submit(one, u): u for u in persons}
            for k, f in enumerate(as_completed(fs)):
                try:
                    r = f.result()
                except (ca.ButceAsildi, ca.Yetkisiz):
                    raise
                except Exception as e:  # noqa: BLE001  one person's format failure does not stop the arm
                    spent["fail"] += 1
                    print(f"{kol} uid {fs[f]}: {type(e).__name__} {str(e)[:120]}", flush=True)
                    continue
                if r == "budget":
                    continue
                rows2 += r[0]
                rows3 += r[1]
                if k % 25 == 0:
                    print(f"{kol} {k + 1}/{len(persons)} usd {spent['usd']:.2f} calls {spent['calls']} "
                          f"{time.time() - t0:.0f}s", flush=True)
        meta = {"model": a.model, "kol": kol, "n_persons_asked": len(persons), "usd": round(spent["usd"], 4),
                **{k: spent[k] for k in ("calls", "cached", "fail")}, "zaman": time.strftime("%F %T"),
                "system": SYS_PERSONA[:200] if kol == "persona" else (SYS_HIST if kol == "gecmis" else SYS_NONE)}
        name = f"cl-{tag}-{kol}_s0"
        if rows2:
            d = BASE / "twin/tahmin"
            d.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(rows2, columns=["uid", "oge", "p"])
            df.to_parquet(d / f"{name}.parquet", index=False)
            (d / f"{name}.meta.json").write_text(json.dumps({**meta, "n_persons": int(df.uid.nunique())}, indent=1))
        if rows3:
            d = BASE / "twin_fiyat/tahmin"
            d.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(rows3, columns=["uid", "oge", "carpan_duzey", "p", "tur"])
            df.to_parquet(d / f"{name}.parquet", index=False)
            (d / f"{name}.meta.json").write_text(json.dumps({**meta, "n_persons": int(df.uid.nunique())}, indent=1))
        print(json.dumps({"arm": name, **meta}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
