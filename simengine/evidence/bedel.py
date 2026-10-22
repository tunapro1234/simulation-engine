"""Tarzlar arası "kaç insana bedel" noktaları: ρ (simülasyon ile gerçek arasındaki birim düzeyi korelasyon)
ve ölçülen PPI++ çarpanı, kuramsal eğri 1/(1 − (1 − n/N)·ρ²) ile.

Bir nokta = bir veri × bir önsel (simülasyon/model). Hedef her sütunun (sorunun, kitlenin) ortalamasıdır;
n kişi rastgele etiketlenir, önsel herkes için vardır. Tahmin edici intent ppi.py ile aynı (PPI++):
λ = Cov(y,f) / ((1 + n/N)·Var(f)), ikisi de etiketli örneklemden, [0, 1]'e kırpılır. (Paydada kitle varyansı
kullanmak çarpık veride çarpanı düşük gösteriyor: OE GBDT n=25'te 2,0× yerine 5,1×; Twin'de fark yok.)
  çarpan  = ortalama MSE(klasik) / ortalama MSE(PPI++)   (sütunlar sd ile standartlaştırılır); kuram 1/(1 − ρ²)
  n_eşdeğer = PPI++ ile aynı MSE'yi veren düz ortalamanın kişi sayısı, sonlu kitle düzeltmeli (persona twin_ess.py
            tanımı): 1 / (MSE_ppi/S² + 1/N); kuram n/(1 − (1 − n/N)·ρ²). n ≪ N iken ikisi aynı; N=500, n=100'de
            n_eşdeğer/n çarpandan belirgin küçük.
  ρ_etkin = √(sütunlar üzerinden ortalama ρ²)            (çok sütunda kuramsal çarpanı veren ρ)
GA: ≥20 sütunda sütun bootstrap'ı (yeni sorulara genelleme), daha azında birim bootstrap'ı (yeni kişilere).
"""
from __future__ import annotations

import numpy as np

SUTUN_BOOT_ESIK = 20
# birim bootstrap'ında en az bu kadar birimli sütunda hızlı yol (ortak etiketli örneklem indeksleri). evaluate'in
# alt örneklem boyuyla (api.MULTIPLIER_MAX_ROWS = 2000) aynı: 2000 satırdan küçük veri özgün yoldan, birebir aynı GA ile.
HIZLI_ESIK = 2000


def _indeks(N, n, R, rng):
    """R tekrar × n etiketli birim, iadesiz (her satırda farklı n birim)."""
    return np.argpartition(rng.random((R, N)), n, axis=1)[:, :n]


def _mse(y, f, n, R, rng, idx=None):
    """Tek sütun: (klasik MSE, PPI++ MSE), θ = y'nin kitle ortalaması. Klasik MSE kapalı form (iadesiz örneklem
    varyansı S²(1 − n/N)/n); çarpık sütunlarda Monte Carlo oranının gürültüsünü yarıya indirir.
    idx: önceden çekilmiş (R, n) etiketli örneklem indeksleri (hızlı bootstrap'ta tekrarlar arasında ortak)."""
    N = len(y)
    mu, fm = y.mean(), f.mean()
    if idx is None:
        idx = _indeks(N, n, R, rng)
    ys, fs = y[idx], f[idx]
    yb, fb = ys.mean(1), fs.mean(1)
    cov = ((ys - yb[:, None]) * (fs - fb[:, None])).sum(1) / (n - 1)
    vf = ((fs - fb[:, None]) ** 2).sum(1) / (n - 1)
    lam = np.clip(np.where(vf > 0, cov / ((1 + n / N) * np.where(vf > 0, vf, 1)), 0.0), 0, 1)
    tah = yb + lam * (fm - fb)
    # %95 GA kapsaması: se² = Var_s(y − λf)(1 − n/N)/n (f herkes için bilinir, belirsizlik yalnız etiketli örneklemden)
    r = ys - lam[:, None] * fs
    se = np.sqrt(((r - r.mean(1)[:, None]) ** 2).sum(1) / (n - 1) * (1 - n / N) / n)
    kapsama = float((np.abs(tah - mu) <= 1.96 * se).mean())
    return y.var(ddof=1) * (1 - n / N) / n, ((tah - mu) ** 2).mean(), kapsama


def _hazir(sutunlar, n, indeks=False):
    out, ix = [], []
    for i, (y, f) in enumerate(sutunlar):
        y, f = np.asarray(y, float), np.asarray(f, float)
        m = np.isfinite(y) & np.isfinite(f)
        y, f = y[m], f[m]
        if len(y) < 2 * n or y.std() == 0:
            continue
        s = y.std()
        out.append((y / s, f / s))
        ix.append(i)
    return (out, ix) if indeks else out


def _basiklik(y) -> float:
    """Fazla basıklık (normal = 0); tek aykırı değerli sütunu işaretler (KARŞIT-2 K2-31: Twin Q166, 5000)."""
    z = (y - y.mean()) / y.std()
    return float((z ** 4).mean() - 3)


def _rho2(y, f):
    return float(np.corrcoef(y, f)[0, 1] ** 2) if f.std() > 0 else 0.0


def olc(sutunlar, n: int = 100, R: int = 400, B: int = 200, tohum: int = 0, hizli: bool | None = None) -> dict:
    """sutunlar: [(y, f), ...] — her biri bir sütunun (soru/kitle) gerçek ve önsel değerleri.

    hizli: birim bootstrap'ında (< SUTUN_BOOT_ESIK sütun) her tekrar için yeni (1000, N) rastgele matris ve argpartition
    yerine sütun başına bir kez çekilen (1000, n) etiketli örneklem indeksleri bütün bootstrap tekrarlarında ortak
    kullanılır (ortak rastgele sayılar). Nokta kestirimi değişmez; GA'nın Monte Carlo gürültüsü azalır, süre ~N/n kat
    kısalır. None (varsayılan): en büyük sütun en az HIZLI_ESIK (2000) birimliyse açık, yani yalnız evaluate'in 2000
    satırlık alt örnekleminde ya da daha büyük veride; 2000'den küçük veride nokta ve GA özgün kanit ile aynı."""
    rng = np.random.default_rng(tohum)
    S, ix = _hazir(sutunlar, n, indeks=True)
    if not S:
        raise ValueError(f"n={n} için yeterli satırı olan sütun yok")
    if len(S) < SUTUN_BOOT_ESIK:  # az sütunda Monte Carlo gürültüsü baskın: tekrarı artır
        R = max(R, 4000)
    mk, mp, kap = np.array([_mse(y, f, n, R, rng) for y, f in S]).T
    r2 = np.array([_rho2(y, f) for y, f in S])
    Ns = np.array([len(y) for y, _ in S])
    nokta = {"n": n, "N": float(Ns.mean()), "sutun": len(S), "R": R,
             "carpan": float(mk.sum() / mp.sum()), "rho": float(np.sqrt(r2.mean())),
             "kapsama": float(kap.mean()), "kapsama_en_dusuk": float(kap.min())}
    if len(S) > 1:  # en kötü sütun tek başına mı? (dipnot soru ortalamasını değil o sütunu anlatmalı)
        j = int(kap.argmin())
        bas = np.array([_basiklik(y) for y, _ in S])
        nokta.update({"kapsama_en_dusuk_sutun": ix[j], "kapsama_en_dusuk_basiklik": float(bas[j]),
                      "kapsama_ikinci_en_dusuk": float(np.sort(kap)[1]), "kapsama_en_kotu_haric": float(np.delete(kap, j).mean()),
                      "kapsama_090_alti": int((kap < 0.90).sum()), "basiklik_ust50": int((bas > 50).sum()),
                      "kapsama_basik_haric": float(kap[bas <= 50].mean()) if (bas <= 50).any() else None})
    if len(S) >= SUTUN_BOOT_ESIK:
        k = rng.integers(0, len(S), (B, len(S)))
        cb = mk[k].sum(1) / mp[k].sum(1)
        rb = np.sqrt(r2[k].mean(1))
        nokta["ga_turu"] = "sütun bootstrap"
    else:
        if hizli is None:
            hizli = int(Ns.max()) >= HIZLI_ESIK
        ortak = [_indeks(len(y), n, 1000, rng) for y, _ in S] if hizli else [None] * len(S)
        cb, rb = [], []
        for _ in range(B):
            Sb = [(y[i], f[i]) for y, f in S for i in [rng.integers(0, len(y), len(y))]]
            a, b, _ = np.array([_mse(y, f, n, 1000, rng, idx=ix_) for (y, f), ix_ in zip(Sb, ortak)]).T
            cb.append(a.sum() / b.sum())
            rb.append(np.sqrt(np.mean([_rho2(y, f) for y, f in Sb])))
        cb, rb = np.array(cb), np.array(rb)
        nokta["ga_turu"] = "birim bootstrap"
        if hizli:
            nokta["hizli_bootstrap"] = True   # ortak etiketli örneklem indeksleri (bkz. olc docstring)
    nokta["carpan_ga"] = [float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))]
    nokta["rho_ga"] = [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]
    nokta["teorik"] = float(1 / (1 - nokta["rho"] ** 2))
    # klasik MSE kapalı form S²(1/n − 1/N) olduğundan n_eşdeğer = 1 / (MSE_ppi/S² + 1/N), sütun ortalamalı
    nokta["n_esdeger"] = float(1 / ((mp / (mk / (1 / n - 1 / Ns))).mean() + (1 / Ns).mean()))
    nokta["n_esdeger_teorik"] = float(n / (1 - (1 - n / nokta["N"]) * nokta["rho"] ** 2))
    return nokta


def egri(n: int | None = None, N: float | None = None, adim: int = 101) -> list[dict]:
    """Kuramsal eğri 1/(1 − k·ρ²); n/N verilirse k = 1 − n/N (n_eşdeğer/n), verilmezse k = 1 (çarpan)."""
    k = 1 - (n / N if n and N else 0)
    return [{"rho": float(r), "carpan": float(1 / (1 - k * r * r))} for r in np.linspace(0, 0.97, adim)]


# ------------------------------------------------------------------ ours next to the strongest baseline
# "Worth N humans" for ours alone reads as if the LLM earned the multiplier even when the information comes from an
# LLM-free baseline (G2: the stated-intent arm T2 gives the same 1.5×). worth_vs_baseline measures both priors on the
# same labelled draws and the same bootstrap resamples and reports the paired difference.
#
#   r = bedel.worth_vs_baseline(frame, "ours", "T2", n=100)      # frame: truth + pred_<name> (+ optional item)
#   r["difference"], r["difference_ci"], r["same_as_baseline"]
#   extras.section_markdown(bedel.section(r))
def _hazir_cift(ucluler, n):
    """[(y, f_ours, f_base), ...] -> standardised triples on the rows where all three are finite (olc's filter)."""
    out = []
    for y, a, b in ucluler:
        y, a, b = (np.asarray(v, float) for v in (y, a, b))
        m = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
        y, a, b = y[m], a[m], b[m]
        if len(y) < 2 * n or y.std() == 0:
            continue
        s = y.std()
        out.append((y / s, a / s, b / s))
    return out


def _cerceve_ucluler(df, ours: str, baseline: str):
    from .api import _method, to_internal   # lazy: api imports this module
    from .sema import ON_EK
    d = to_internal(df)
    ca, cb = ON_EK + _method(ours), ON_EK + _method(baseline)
    for c in (ca, cb):
        if c not in d.columns:
            raise KeyError(f"prediction column for {c[len(ON_EK):]!r} not found (expected pred_{c[len(ON_EK):]})")
    gruplar = [x for _, x in d.groupby("madde", observed=True)] if "madde" in d.columns else [d]
    return [(x["gercek"].to_numpy(float), x[ca].to_numpy(float), x[cb].to_numpy(float)) for x in gruplar]


def _iki_mse(y, a, b, n, R, rng, idx=None):
    """(classical MSE, PPI++ MSE ours, PPI++ MSE baseline) on one shared set of labelled draws."""
    if idx is None:
        idx = _indeks(len(y), n, R, rng)
    mk, ma, _ = _mse(y, a, n, R, rng, idx=idx)
    _, mb, _ = _mse(y, b, n, R, rng, idx=idx)
    return mk, ma, mb


def _n_esdeger(mk, mp, n, Ns):
    return float(1 / ((mp / (mk / (1 / n - 1 / Ns))).mean() + (1 / Ns).mean()))


def worth_vs_baseline(data, ours: str | None = None, baseline: str | None = None, n: int = 100, R: int = 400,
                      B: int = 200, seed: int = 0, hizli: bool | None = None) -> dict:
    """PPI++ multiplier of ours and of the strongest baseline on the same resamples, with the paired difference.

    data: a DataFrame (truth + pred_<ours> + pred_<baseline>, optional item: one target per item), a list of
    (y, f_ours, f_baseline) triples or a single triple. Rows where any of the three is missing are dropped, so both
    arms use the same units; items with fewer than 2n complete rows are skipped (as in olc). Point: each column
    draws one (R, n) labelled index set shared by both priors (closed-form classical MSE, as in olc). CI: column bootstrap with >= SUTUN_BOOT_ESIK columns, otherwise unit bootstrap; each resample is shared
    by both priors, so the difference CI is paired. same_as_baseline is True when the difference CI includes 0: the
    "worth N humans" figure is then not evidence for what ours adds. olc itself is unchanged.
    baseline: an LLM-free prior that never saw the target labels (e.g. stated intent). An OOF model fitted on the
    same population's labels is circular: its multiplier is inflated and ours wrongly reads "below baseline".
    Same seed and rows as the card (single item, the rows evaluate scored, e.g. split="test") reproduce the card's
    bedel row for ours; multi-item cards use orneklem's pooled-λ estimator and can differ.
    hizli: as in olc (shared labelled draws across the unit bootstrap; default on from HIZLI_ESIK units).
    Coverage of the difference CI (simulation, 300 populations each, N = 400-600, n = 100): unit bootstrap
    (1-3 items, incl. skewed and 0/1 truth, weak priors) 0.95-0.98; item bootstrap at 25 heterogeneous items
    0.92-0.93 (percentile bootstrap over few items runs narrow; olc's own item-bootstrap CI does the same), so
    there a CI edge just past 0 is weak evidence either way.
    """
    if hasattr(data, "columns"):
        if ours is None or baseline is None:
            raise ValueError("worth_vs_baseline(frame, ours, baseline): name both prediction columns")
        data = _cerceve_ucluler(data, ours, baseline)
    elif isinstance(data, tuple) and len(data) == 3 and np.ndim(data[0]) == 1:
        data = [data]                     # one (y, f_ours, f_baseline) triple, as worth_humans takes one pair
    rng = np.random.default_rng(seed)
    S = _hazir_cift(list(data), n)
    if not S:
        raise ValueError(f"worth_vs_baseline: no item has >= 2n = {2 * n} rows with truth, ours and baseline all "
                         f"present (and truth not constant); pass a smaller n")
    if len(S) < SUTUN_BOOT_ESIK:
        R = max(R, 4000)
    mk, ma, mb = np.array([_iki_mse(y, a, b, n, R, rng) for y, a, b in S]).T
    Ns = np.array([len(y) for y, _, _ in S])
    ra = np.array([_rho2(y, a) for y, a, _ in S])
    rb_ = np.array([_rho2(y, b) for y, _, b in S])
    if len(S) >= SUTUN_BOOT_ESIK:
        k = rng.integers(0, len(S), (B, len(S)))
        ca, cb = mk[k].sum(1) / ma[k].sum(1), mk[k].sum(1) / mb[k].sum(1)
        tur = "sütun bootstrap"
    else:
        if hizli is None:
            hizli = int(Ns.max()) >= HIZLI_ESIK
        ortak = [_indeks(len(y), n, 1000, rng) if hizli else None for y, _, _ in S]
        ca, cb = np.empty(B), np.empty(B)
        for t in range(B):
            top = np.zeros(3)
            for (y, a, b), ix_ in zip(S, ortak):
                i = rng.integers(0, len(y), len(y))
                top += _iki_mse(y[i], a[i], b[i], n, 1000, rng, idx=ix_)
            ca[t], cb[t] = top[0] / top[1], top[0] / top[2]
        tur = "birim bootstrap"
    d = ca - cb

    def ga(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    fark_ga = ga(d)
    out = {"n": n, "N": float(Ns.mean()), "sutun": len(S), "R": R, "B": B, "ga_turu": tur,
           "ours": _ad(ours, "ours"), "baseline": _ad(baseline, "baseline"),
           "multiplier_ours": float(mk.sum() / ma.sum()), "multiplier_baseline": float(mk.sum() / mb.sum()),
           "multiplier_ci_ours": ga(ca), "multiplier_ci_baseline": ga(cb),
           "humans_ours": _n_esdeger(mk, ma, n, Ns), "humans_baseline": _n_esdeger(mk, mb, n, Ns),
           "rho_ours": float(np.sqrt(ra.mean())), "rho_baseline": float(np.sqrt(rb_.mean())),
           "difference_ci": fark_ga, "same_as_baseline": bool(fark_ga[0] <= 0 <= fark_ga[1])}
    out["difference"] = out["multiplier_ours"] - out["multiplier_baseline"]
    if tur == "birim bootstrap" and hizli:
        out["hizli_bootstrap"] = True
    return out


def _ad(x, varsayilan):
    if x is None:
        return varsayilan
    for p in ("pred_", "tahmin_"):
        if x.startswith(p):
            return x[len(p):]
    return x


_GA_EN = {"sütun bootstrap": "item-bootstrap", "birim bootstrap": "unit-bootstrap"}


def section(result: dict, *, title: str | None = None) -> dict:
    """worth_vs_baseline result -> an evidence-extras section dict (render with extras.section_markdown/_html)."""
    r = result
    o, b = r["ours"], r["baseline"]

    def aralik(ci):
        return f"[{ci[0]:.2f}, {ci[1]:.2f}]"

    lo, hi = r["difference_ci"]
    rows = [[o, r["multiplier_ours"], aralik(r["multiplier_ci_ours"]),
             round(r["humans_ours"]), r["rho_ours"]],
            [f"{b} (strongest baseline alone)", r["multiplier_baseline"],
             aralik(r["multiplier_ci_baseline"]),
             round(r["humans_baseline"]), r["rho_baseline"]],
            [f"difference {o} - {b}", r["difference"], aralik(r["difference_ci"]), None, None]]
    notes = [f"{o} + {r['n']} real answers is worth ~{r['humans_ours']:.0f} humans "
             f"({r['multiplier_ours']:.2f}×); strongest baseline alone ~{r['humans_baseline']:.0f} "
             f"({r['multiplier_baseline']:.2f}×); difference {r['difference']:+.2f}× [{lo:+.2f}, {hi:+.2f}]."]
    if r["same_as_baseline"]:
        notes.append(f"Same as baseline: the difference CI includes 0, so {o} is not shown to be worth more humans "
                     f"than {b} alone; the multiplier is no evidence for what {o} adds.")
    elif r["difference"] < 0:
        notes.append(f"Below baseline: {b} alone is worth more humans than {o}.")
    notes.append(f"Paired on the same labelled draws and {r['B']} {_GA_EN.get(r['ga_turu'], r['ga_turu'])} resamples "
                 f"({r['sutun']} column(s), mean N = {r['N']:.0f}).")
    return {"kind": "worth_vs_baseline", "title": title or f"Worth N humans: {o} vs strongest baseline {b}",
            "columns": ["arm", "multiplier", "95% CI", "≈ humans", "rho"], "rows": rows, "notes": notes,
            "digits": 2, "summary": {k: r[k] for k in ("multiplier_ours", "multiplier_baseline", "difference",
                                                        "difference_ci", "same_as_baseline")}}
