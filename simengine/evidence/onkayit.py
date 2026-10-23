"""Ön kayıt: "önce tahmin et, sonra notla".

Tahmin tablosu gerçek sonuç açıklanmadan önce mühürlenir (sha256 + zaman damgası +
hash zinciri). Gerçek gelince tahminin değişmediği kanıtlanır, sonra notlanır.

    python -m simengine.evidence.onkayit muhurle tahmin.csv --aciklama "ilk tahmin"
    python -m simengine.evidence.onkayit muhurle tahmin.csv --aciklama "ours: H" --belge ONKAYIT.md
    python -m simengine.evidence.onkayit dogrula tahmin.csv
    python -m simengine.evidence.onkayit zincir
    python -m simengine.evidence.onkayit notla tahmin.csv gercek.csv --anahtar kisi,soru
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ZINCIR_ADI = "zincir.jsonl"


# ---------- kanonik biçim ----------

def _oku(df_ya_da_yol) -> pd.DataFrame:
    if isinstance(df_ya_da_yol, pd.DataFrame):
        return df_ya_da_yol
    yol = Path(df_ya_da_yol)
    ad = yol.name.lower()
    if ad.endswith((".parquet", ".pq")):
        return pd.read_parquet(yol)
    if ad.endswith((".jsonl", ".jsonl.gz")):
        return pd.read_json(yol, lines=True)
    if ad.endswith((".json", ".json.gz")):
        return pd.read_json(yol)
    if ad.endswith((".csv", ".csv.gz", ".tsv")):
        return pd.read_csv(yol, sep="\t" if ad.endswith(".tsv") else ",")
    raise ValueError(f"desteklenmeyen dosya türü: {yol}")


def _sayi(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    s = format(x, ".10g")
    return "0" if s == "-0" else s


def _yapi(x):
    # liste/dizi/sözlük içini JSON'a uygun, float'ları sabitlenmiş hale getirir
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, dict):
        return {str(k): _yapi(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_yapi(v) for v in x]
    if isinstance(x, float):
        return float(_sayi(x)) if math.isfinite(x) else _sayi(x)
    return x


def _hucre(x) -> str:
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)) and getattr(x, "ndim", 0) > 0:
        return json.dumps(_yapi(x), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if isinstance(x, (list, tuple, dict)):
        return json.dumps(_yapi(x), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if x is None or x is pd.NA or x is pd.NaT:
        return "nan"
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()  # numpy skaler → python
    if isinstance(x, bool):
        return "True" if x else "False"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return _sayi(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    return str(x)


def kanonik_bayt(df_ya_da_yol) -> bytes:
    """Aynı içerik → aynı bayt (sütun ve satır sırasından bağımsız)."""
    df = _oku(df_ya_da_yol)
    sutunlar = sorted(df.columns, key=str)
    satirlar = [[_hucre(v) for v in satir]
                for satir in df[sutunlar].itertuples(index=False, name=None)]
    satirlar.sort()  # kararlı; tüm sütunlara göre
    tampon = io.StringIO()
    yazici = csv.writer(tampon, lineterminator="\n")
    yazici.writerow([str(s) for s in sutunlar])
    yazici.writerows(satirlar)
    return tampon.getvalue().encode("utf-8")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ---------- zincir ----------

def _zincir_oku(kayit_dizini) -> list:
    yol = Path(kayit_dizini) / ZINCIR_ADI
    if not yol.exists():
        return []
    with open(yol, encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def _kayit_hash(onceki_kayit_hash, kayit: dict) -> str:
    govde = {k: v for k, v in kayit.items() if k != "kayit_hash"}
    return _sha(((onceki_kayit_hash or "") + _json(govde)).encode("utf-8"))


def _git_commit(dizin: Path, mesaj: str) -> None:
    try:
        calis = lambda *a: subprocess.run(["git", *a], cwd=dizin, check=True,
                                          capture_output=True, text=True)
        calis("rev-parse", "--is-inside-work-tree")
        calis("add", "--", ".")
        calis("commit", "-m", mesaj, "--", ".")
    except (OSError, subprocess.CalledProcessError) as h:
        ayrinti = getattr(h, "stderr", "") or str(h)
        warnings.warn(f"onkayit: git commit atlandı ({ayrinti.strip()})")


def _dosya_sha(tahmin) -> str | None:
    """Dosya yolu verildiyse dosyanın ham baytlarının sha256'sı (kanonik CSV hash'inden farklıdır)."""
    if isinstance(tahmin, pd.DataFrame):
        return None
    try:
        p = Path(tahmin)
        return _sha(p.read_bytes()) if p.is_file() else None
    except (TypeError, OSError, ValueError):
        return None


def belge_kaydi(belge) -> dict:
    """Ön kayıt belgesinin (ONKAYIT.md gibi) ham bayt sha256'sı: {"dosya": ad, "sha256"} (yol yazılmaz).
    research.lock_prereg kilidindeki sha256 ile aynı hesap."""
    p = Path(belge)
    if not p.is_file():
        raise FileNotFoundError(f"ön kayıt belgesi yok: {p.name} (prereg_doc)")
    return {"dosya": p.name, "sha256": _sha(p.read_bytes())}


KAYNAK_YOLU = {"ad": "ad", "name": "ad", "tam": "tam", "full": "tam"}


def _kaynak(tahmin, kaynak_yolu: str) -> str | None:
    """Kayıttaki "kaynak": varsayılan yalnız dosya adı (zincir commit'lenebilir, makine yolu girmesin); "tam"/"full"
    verilen yolu olduğu gibi yazar. DataFrame → None."""
    if kaynak_yolu not in KAYNAK_YOLU:
        raise ValueError(f"kaynak_yolu {kaynak_yolu!r}: 'ad' (name, default) or 'tam' (full)")
    if isinstance(tahmin, pd.DataFrame):
        return None
    return str(tahmin) if KAYNAK_YOLU[kaynak_yolu] == "tam" else Path(tahmin).name


def muhurle(tahmin, aciklama: str, kayit_dizini=".onkayit", git_commit: bool = False, *, kopya: bool = True,
            arsiv_dizini=None, belge=None, kaynak_yolu: str = "ad") -> dict:
    """Tahmini mühürler: hash + zaman + zincir bağı; kanonik kopyayı saklar.

    kopya=False: yalnız zincir kaydı yazılır (kanonik kopya yok; kayıtta "arsiv": None). Doğrulama zincirle yapılır,
    notlama tahminin kendisini ister. arsiv_dizini: kopya kayıt dizini yerine buraya yazılır (kayıtta yalnız dosya adı
    ve "arsiv_konum": "dis"; yol zincire girmez). Varsayılan kayıt dizini çalışma dizinindedir: lisanslı veride
    git-ignored ya da repo dışında tutun.
    belge: ön kayıt belgesi (ONKAYIT.md): ham bayt sha256'sı kayda "onkayit_belgesi" olarak girer ve açıklamanın
    sonuna "[ön kayıt belgesi <ad> sha256:<hash>]" eklenir; evaluate() bu belgeyi mühürle bağlar (belge o anki
    içeriğiyle mühürden önce vardı).
    kaynak_yolu: kayıttaki "kaynak" alanı: "ad" (varsayılan, name) yalnız dosya adı; "tam" (full) verilen yolun
    tamamı (mutlak bir makine yolu commit'lenen zincire girer). Eski, tam yollu kayıtlar aynen doğrulanır."""
    kaynak = _kaynak(tahmin, kaynak_yolu)
    if not kopya and arsiv_dizini is not None:
        raise ValueError("kopya=False ile arsiv_dizini birlikte verilemez (store_copy=False, archive_dir)")
    dizin = Path(kayit_dizini)
    dizin.mkdir(parents=True, exist_ok=True)
    bk = belge_kaydi(belge) if belge is not None else None
    if bk:
        aciklama = f"{aciklama} [ön kayıt belgesi {bk['dosya']} sha256:{bk['sha256']}]"
    df = _oku(tahmin)
    bayt = kanonik_bayt(df)
    sha = _sha(bayt)
    zincir = _zincir_oku(dizin)
    son = zincir[-1] if zincir else None
    simdi = datetime.now().astimezone()
    kayit = {
        "sira": len(zincir),
        "sha256": sha,
        "zaman": simdi.isoformat(timespec="seconds"),
        "zaman_utc": simdi.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "satir": int(df.shape[0]),
        "sutun": int(df.shape[1]),
        "sutunlar": sorted(str(s) for s in df.columns),
        "aciklama": aciklama,
        "kaynak": kaynak,
        "dosya_sha256": _dosya_sha(tahmin),
        "arsiv": f"{sha[:12]}.csv.gz" if kopya else None,
        "onceki": son["sha256"] if son else None,
        "onceki_kayit": son["kayit_hash"] if son else None,
    }
    if bk:
        kayit["onkayit_belgesi"] = bk
    if arsiv_dizini is not None:
        kayit["arsiv_konum"] = "dis"   # kopya kayıt dizininin dışında (yol bilerek yazılmaz)
    kayit["kayit_hash"] = _kayit_hash(kayit["onceki_kayit"], kayit)
    # kanonik kopya: açılmış içeriğin sha256'sı doğrudan kayıttaki sha256'dır
    if kopya:
        hedef = Path(arsiv_dizini) if arsiv_dizini is not None else dizin
        hedef.mkdir(parents=True, exist_ok=True)
        with open(hedef / kayit["arsiv"], "wb") as f:
            f.write(gzip.compress(bayt, mtime=0))
    with open(dizin / ZINCIR_ADI, "a", encoding="utf-8") as f:
        f.write(_json(kayit) + "\n")
    if git_commit:
        _git_commit(dizin, f"onkayit: {sha[:12]} {aciklama}")
    return kayit


def dogrula(tahmin, sha256: str | None = None, kayit_dizini=".onkayit") -> dict:
    """Tahminin hash'i verilen sha256'ya ya da zincirdeki bir kayda uyuyor mu?"""
    sha = _sha(kanonik_bayt(tahmin))
    eslesenler = [k for k in _zincir_oku(kayit_dizini) if k["sha256"] == sha]
    kayit = eslesenler[0] if eslesenler else None  # en eski mühür = en güçlü kanıt
    eslesti = (sha == sha256.lower()) if sha256 else kayit is not None
    return {"eslesti": eslesti, "kayit": kayit, "sha256": sha}


def zincir_dogrula(kayit_dizini=".onkayit") -> dict:
    """Bütün zincirin bağlarını ve kayıt hash'lerini yeniden hesaplar."""
    zincir = _zincir_oku(kayit_dizini)
    kirik, onceki = [], None
    for i, k in enumerate(zincir):
        beklenen_onceki = onceki["sha256"] if onceki else None
        beklenen_bag = onceki["kayit_hash"] if onceki else None
        if (k.get("onceki") != beklenen_onceki or k.get("onceki_kayit") != beklenen_bag
                or k.get("sira") != i or _kayit_hash(beklenen_bag, k) != k.get("kayit_hash")):
            kirik.append(i)
        onceki = k
    return {"gecerli": not kirik, "n": len(zincir), "kirik": kirik}


def arsivden(kayit_dizini=".onkayit", sha256: str | None = None, arsiv_dizini=None) -> pd.DataFrame:
    """Mühürlü kanonik kopyayı açar (sha256 verilmezse zincirin son kaydı). Açılmış baytların sha256'sı kayıtla
    eşleşmezse ValueError. Kopya yoksa (kopya=False ile mühür) FileNotFoundError: tahmini kendiniz verin."""
    k = _zincir_kaydi(kayit_dizini, sha256)
    if k is None:
        raise ValueError("arşivden okuma: zincirde eşleşen mühür kaydı yok")
    if not k.get("arsiv"):
        raise FileNotFoundError(f"mühür {k['sha256'][:12]} kopyasız (store_copy=False): tahmin tablosunu verin")
    yol = Path(arsiv_dizini if arsiv_dizini is not None else kayit_dizini) / k["arsiv"]
    if not yol.exists():
        raise FileNotFoundError(f"arşiv kopyası yok: {k['arsiv']}" + (" (kopya kayıt dizininin dışında: archive_dir= verin)"
                                                                   if k.get("arsiv_konum") == "dis" and arsiv_dizini is None else ""))
    bayt = gzip.decompress(yol.read_bytes())
    if _sha(bayt) != k["sha256"]:
        raise ValueError(f"arşiv kopyası mühürle eşleşmiyor: {k['arsiv']}")
    return pd.read_csv(io.BytesIO(bayt))


def notla(tahmin, gercek, anahtar: list[str], kayit_dizini=".onkayit", **degerlendir_kw) -> dict:
    """Mühür doğrulanırsa tahmin ile gerçeği anahtar sütunlarından birleştirir.

    tahmin=None: mühürlü kanonik kopya açılır (`arsivden`; sha256= ve arsiv_dizini= ile); baytları mühürle eşleşir."""
    arsiv_dizini = degerlendir_kw.pop("arsiv_dizini", None)
    if tahmin is None:
        sha = degerlendir_kw.pop("sha256", None)
        tahmin = arsivden(kayit_dizini, sha, arsiv_dizini)      # baytların sha256'sı kayıtla eşleşti
        k = _zincir_kaydi(kayit_dizini, sha)
        return _birlestir(tahmin, gercek, anahtar, {"eslesti": True, "kayit": k, "sha256": k["sha256"], "kaynak": "arsiv"},
                          degerlendir_kw)
    dog = dogrula(tahmin, degerlendir_kw.pop("sha256", None), kayit_dizini)
    if not dog["eslesti"]:
        raise ValueError(f"tahmin mühürlü kayıtla eşleşmiyor (sha256={dog['sha256'][:12]})")
    return _birlestir(tahmin, gercek, anahtar, dog, degerlendir_kw)


def _zincir_kaydi(kayit_dizini, sha256=None) -> dict | None:
    zincir = _zincir_oku(kayit_dizini)
    if sha256:
        return next((r for r in zincir if r.get("sha256") == sha256.lower()), None)
    return zincir[-1] if zincir else None


def _birlestir(tahmin, gercek, anahtar, dog, degerlendir_kw) -> dict:
    t, g = _oku(tahmin), _oku(gercek)
    eksik = [a for a in anahtar if a not in t.columns or a not in g.columns]
    if eksik:
        raise KeyError(f"anahtar sütun(lar) eksik: {eksik}")
    tablo = t.merge(g, on=anahtar, how=degerlendir_kw.pop("nasil", "inner"),
                    suffixes=("", "_gercek"))
    return {"dogrulama": dog, "tablo": tablo,
            "n_tahmin": len(t), "n_gercek": len(g), "n_birlesik": len(tablo),
            "notlama_zamani": datetime.now().astimezone().isoformat(timespec="seconds"),
            "ek": degerlendir_kw}


# ---------- CLI ----------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="simengine.evidence.onkayit", description="Ön kayıt: önce tahmin et, sonra notla")
    ap.add_argument("--dizin", default=".onkayit", help="kayıt dizini (varsayılan .onkayit)")
    ap.add_argument("--json", action="store_true", help="çıktıyı JSON ver")
    alt = ap.add_subparsers(dest="komut", required=True)
    m = alt.add_parser("muhurle", help="tahmini mühürle")
    m.add_argument("dosya")
    m.add_argument("--aciklama", required=True)
    m.add_argument("--git", action="store_true", help="kayıt dizinini git'e commit et")
    m.add_argument("--kopyasiz", action="store_true", help="kanonik kopya yazma, yalnız zincir kaydı (store_copy=False)")
    m.add_argument("--arsiv", help="kanonik kopyayı bu dizine yaz (archive_dir)")
    m.add_argument("--belge", help="ön kayıt belgesi (ONKAYIT.md): sha256'sı mühür kaydına girer (prereg_doc)")
    m.add_argument("--tam-yol", action="store_true", help="kayda dosya adı yerine verilen yolun tamamını yaz")
    d = alt.add_parser("dogrula", help="tahmin mühürlü mü")
    d.add_argument("dosya")
    d.add_argument("--sha256")
    alt.add_parser("zincir", help="zincir bütünlüğünü kontrol et")
    n = alt.add_parser("notla", help="doğrula ve gerçekle birleştir")
    n.add_argument("tahmin")
    n.add_argument("gercek")
    n.add_argument("--anahtar", required=True, help="virgülle ayrılmış sütunlar")
    n.add_argument("--cikti", help="birleşik tabloyu CSV olarak yaz")
    a = ap.parse_args(argv)

    def yaz(veri, metin):
        print(json.dumps(veri, ensure_ascii=False, indent=2) if a.json else metin)

    if a.komut == "muhurle":
        k = muhurle(a.dosya, a.aciklama, a.dizin, a.git, kopya=not a.kopyasiz, arsiv_dizini=a.arsiv, belge=a.belge,
                    kaynak_yolu="tam" if a.tam_yol else "ad")
        yaz(k, f"Mühürlendi #{k['sira']}: {k['sha256']}\n  zaman: {k['zaman']}  "
               f"({k['satir']} satır × {k['sutun']} sütun)\n  açıklama: {k['aciklama']}\n"
               f"  zincir: {k['kayit_hash'][:16]}…  önceki: {(k['onceki'] or 'yok')[:12]}")
        return 0
    if a.komut == "dogrula":
        s = dogrula(a.dosya, a.sha256, a.dizin)
        if s["eslesti"]:
            ek = f"\n  mühür #{s['kayit']['sira']} {s['kayit']['zaman']}: {s['kayit']['aciklama']}" if s["kayit"] else ""
            metin = f"EŞLEŞTİ: {s['sha256']}{ek}"
        else:
            metin = f"EŞLEŞMEDİ: {s['sha256']} (tahmin mühürden sonra değişmiş ya da hiç mühürlenmemiş)"
        yaz(s, metin)
        return 0 if s["eslesti"] else 1
    if a.komut == "zincir":
        z = zincir_dogrula(a.dizin)
        yaz(z, f"Zincir {'GEÇERLİ' if z['gecerli'] else 'KIRIK'}: {z['n']} kayıt"
               + (f", kırık indeksler: {z['kirik']}" if z["kirik"] else ""))
        return 0 if z["gecerli"] else 1
    try:
        s = notla(a.tahmin, a.gercek, [x.strip() for x in a.anahtar.split(",")], a.dizin)
    except ValueError as h:
        yaz({"hata": str(h)}, f"HATA: {h}")
        return 1
    if a.cikti:
        s["tablo"].to_csv(a.cikti, index=False)
    ozet = {k: v for k, v in s.items() if k != "tablo"}
    yaz(ozet, f"Doğrulandı ({s['dogrulama']['sha256'][:12]}); "
              f"{s['n_birlesik']} birleşik satır (tahmin {s['n_tahmin']}, gerçek {s['n_gercek']})"
              + (f" → {a.cikti}" if a.cikti else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
