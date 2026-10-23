"""Latent-persona runner: one run = arm x model x task x seed (prereg docs/prereg/latent-persona-2026-10-23.md,
package simengine.persona.latent, usage docs/modules/latent.md).

    PY="nice -n 19 /data/talp/venv/bin/python"
    $PY runs/latent/kos.py --arm enc-an enc-gc enc-angc --seed 0 1 2                     # LLM-free arms (CPU)
    $PY runs/latent/kos.py --model q38 --precision bf16 --arm prm-L2 prb-L2 stk --seed 0    # G1 (task from the arm)
    $PY runs/latent/kos.py --model q38 --arm rsd-angc rsd-angc-swap rsd-angc-zero sft-angc --seed 0
    $PY runs/latent/kos.py --model q38 --arm prm-tf prm-tf0 rsd-tf enc-tf lgbm-tf          # G3 (one split)
    $PY runs/latent/kos.py --model q35s --precision fp32 --device cpu --arm prb-L2 --kucuk # smoke: 40/10/20 persons
    $PY runs/latent/kos.py --calibrate --model q38 --precision fp8                         # tok/s, one rsd/sft step
    $PY runs/latent/kos.py --score g1 --arm prb-q38-L2 --taban enc-angc gecmis --seed 0     # degerlendir.py, B 1000
    $PY runs/latent/kos.py --list                                                          # arm ids + run log
    $PY runs/latent/kos.py --plan 80gb --until 07:30                                       # the whole priority list

Arms: an arm without a model id (prb-L2) takes --model; the file name always carries it (prb-q38-L2_s0).
Resume: a run whose prediction and .meta.json exist is skipped (--force reruns it); the learned-injector sweeps keep
each config's validation score in <task>/secim/<arm>_s<s>.sweep.jsonl and continue from there.
Run log: <base>/kosular.jsonl, one line per run (model, precision, layer, coefficient, config, val score, seconds,
card, status). --kucuk writes everything under <base>/kucuk/ and seals nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("HF_HOME", "/data/talp/hf")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simengine.persona import latent as lt  # noqa: E402
from simengine.persona.latent import prompts as P  # noqa: E402

BASE = Path(os.environ.get("LATENT_BASE", "/data/talp/is/latent"))
TALP73 = Path("/srv/lighthouse/hackathon-coz/projeler/talp-simulation-hackathon-2026/uretim/talp73")
DEGERLENDIR = TALP73 / "kod" / "degerlendir.py"
TABAN_KAYNAK = Path("/data/talp/is/talp73/tahmin")
PANEL = Path(os.environ.get("TALP73_PANEL", "/data/talp/is/talp73/panel"))
TASK_DIR = {"g1": "talp73", "g2": "twin", "g3": "twin_fiyat"}
N_TEST_G1 = 8000

LLM_FREE = {"g1": ("enc-an", "enc-gc", "enc-angc", "enc-e5L1", "enc-e5L2", "stk-e5"),
            "g2": ("enc-tw", "enc-tw-e5"), "g3": ("enc-tf", "lgbm-tf")}
LLM_U = {"prm": ("L0", "L1", "L2", "tw", "tf", "tf0"), "prb": ("L1", "L2", "tw"), "stk": ("",), "caa": ("",),
         "rsd": ("angc", "e5L2", "tf"), "sft": ("angc", "tf")}
VARIANTS = {"swap": ("caa", "rsd", "sft"), "zero": ("caa", "rsd", "sft")}

# grids of the prereg (Extra); --kucuk keeps the first two configs of each
PRB_FRACS = (0.25, 0.4, 0.5, 0.6, 0.75, 0.9)
CAA_FRACS, CAA_C = (0.3, 0.45, 0.55, 0.65), (0.02, 0.05, 0.1, 0.2, 0.4)
RSD = {"frac": (0.4, 0.55, 0.7), "positions": ("last", "last4"), "lr": (1e-4, 3e-4)}
SFT = {"k": (1, 4), "lr": (3e-4, 1e-3)}
EPOCHS = {"rsd": 3.0, "sft": 2.0}


# ------------------------------------------------------------------------------------------------- arms
@dataclass
class Arm:
    name: str
    kind: str
    model: str | None
    u: str
    variant: str | None
    task: str

    @property
    def base(self) -> str:
        """The arm without its control suffix (swap / zero share its selection and projector)."""
        return self.name[: -len(self.variant) - 1] if self.variant else self.name


def model_tag(code: str) -> str:
    return code.replace("-", "_")


def tag_code(tag: str) -> str:
    return "tiny-" + tag[5:] if tag.startswith("tiny_") else tag


def parse_arm(text: str, model: str | None = None) -> Arm:
    """'enc-an' | 'prb-L2' (+ --model) | 'prb-q38-L2' | 'rsd-q38-angc-swap' -> Arm. The task follows from u."""
    for task, names in LLM_FREE.items():
        if text in names:
            kind, _, u = text.partition("-")
            return Arm(text, kind, None, u, None, task)
    parts = text.split("-")
    kind, rest = parts[0], parts[1:]
    if kind not in LLM_U:
        raise ValueError(f"unknown arm {text!r} (see --list)")
    variant = rest.pop() if rest and rest[-1] in VARIANTS else None
    if rest and (tag_code(rest[0]) in lt.MODELS or rest[0].startswith("tiny_")):
        m = tag_code(rest.pop(0))
    elif model:
        m = model
    else:
        raise ValueError(f"{text}: no model; pass --model or write {kind}-<model>-...")
    u = "-".join(rest)
    if u not in LLM_U[kind]:
        raise ValueError(f"{text}: {kind} takes {[x for x in LLM_U[kind] if x] or 'no suffix'}")
    if variant and kind not in VARIANTS[variant]:
        raise ValueError(f"{text}: -{variant} exists only for {VARIANTS[variant]}")
    task = "g2" if u == "tw" else "g3" if u in ("tf", "tf0") else "g1"
    name = "-".join([kind, model_tag(m)] + ([u] if u else []) + ([variant] if variant else []))
    return Arm(name, kind, m, u, variant, task)


# ------------------------------------------------------------------------------------------------- context
def _card() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}"
    except ImportError:
        pass
    return "cpu"


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


class Ctx:
    def __init__(self, a):
        self.a = a
        self.base = Path(a.base)
        self.root = Path(a.root) if a.root else self.base / "veri"
        self.kucuk = tuple(int(x) for x in a.kucuk.split(":")) if a.kucuk else None
        self.work = self.base / "kucuk" if self.kucuk else self.base
        self.seal = None if (self.kucuk or a.no_seal) else self.base / ".prereg"
        self.log_path = self.work / "kosular.jsonl"
        self._ld, self._ld_key, self.load_seconds = None, None, 0.0
        self._views: dict = {}
        self.commit = _commit()

    def ld(self, code: str):
        """The loaded model for an arm's model code in --precision (q38 + fp8 loads the official FP8 checkpoint
        q38f; the arm keeps the name q38). One model in memory at a time."""
        key = (code, self.a.precision)
        if self._ld is None or self._ld_key != key:
            self.release()
            load_code = "q38f" if (code == "q38" and self.a.precision == "fp8") else code
            t = time.time()
            self._ld = lt.load(load_code, self.a.precision, device=self.a.device)
            self._ld_key = key
            self.load_seconds = round(time.time() - t, 1)
        return self._ld

    def release(self):
        if self._ld is not None:
            self._ld = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def tdir(self, task: str, sub: str = "tahmin") -> Path:
        d = self.work / TASK_DIR[task] / sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    def states_dir(self) -> Path:
        return self.work / "states"

    def small(self, configs: list) -> list:
        return configs[:2] if self.kucuk else configs

    def log(self, rec: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def _cap(uids, n, seed):
    uids = np.array(sorted(set(uids)))
    return set(uids.tolist()) if n is None or len(uids) <= n else set(
        np.random.default_rng(seed).choice(uids, n, replace=False).tolist())


def _cap_target(t: pd.DataFrame, kucuk) -> pd.DataFrame:
    """--kucuk: keep fit / val / test persons up to the given counts (rng 7 / 8 / 9)."""
    if not kucuk:
        return t
    nf, nv, nt = kucuk
    keep = (_cap(t[(t.bolum == "train") & ~t.ic_dogrulama.astype(bool)].uid, nf, 7)
            | _cap(t[(t.bolum == "train") & t.ic_dogrulama.astype(bool)].uid, nv, 8)
            | _cap(t[t.bolum == "test"].uid, nt, 9))
    return t[t.uid.isin(keep)].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------- G1 view
class V1:
    """One talp73 seed: person order, index maps, rows, texts, leak guard."""

    def __init__(self, ctx: Ctx, seed: int):
        a = ctx.a
        cf, cv = (ctx.kucuk[0], ctx.kucuk[1]) if ctx.kucuk else (a.cap_fit, a.cap_val)
        g = lt.load_g1(ctx.root, seed, cap_fit=cf, cap_val=cv)
        if ctx.kucuk:
            g.target = _cap_target(g.target, ctx.kucuk)
        self.g, self.seed = g, seed
        self.persons = g.persons()
        self.uids = self.persons.uid.tolist()
        self.pi = {u: i for i, u in enumerate(self.uids)}
        self.cats = g.categories()
        self.ci = {c: i for i, c in enumerate(self.cats)}
        self.names = dict(zip(g.items.oge, g.items.istem_adi))
        self.is_train = self.persons.bolum.eq("train").to_numpy()
        self.is_fit = self.is_train & ~self.persons.ic_dogrulama.astype(bool).to_numpy()
        self.fit, self.val, self.train, self.test = g.fit_rows(), g.val_rows(), g.train_rows(), g.test_rows()
        self.guard = g.guard()
        self.prof = dict(zip(g.profile.uid, g.profile.metin))
        self.hist = {u: (a_, m) for u, a_, m in zip(g.history_text.uid, g.history_text.aralik, g.history_text.metin)}
        self.n_expected = None if ctx.kucuk else N_TEST_G1
        self._vec: dict = {}
        self._events = None

    def idx(self, rows: pd.DataFrame):
        return (rows.uid.map(self.pi).to_numpy(np.int64), rows.oge.map(self.ci).to_numpy(np.int64),
                rows.y.to_numpy(int))

    def window_index(self, rows: pd.DataFrame) -> np.ndarray:
        """0 = train window (January-June 2022), 1 = test window (July-December 2022)."""
        return rows.bolum.eq("test").to_numpy().astype(np.int64)

    def messages(self, level: str, uid, *, window: str = "test", cf: str | None = None):
        aralik, hist = self.hist.get(uid, ("", ""))
        return P.g1_messages(level, self.prof.get(uid, ""), hist, aralik, window=window, cf=cf)

    def events(self) -> pd.DataFrame:
        if self._events is None:
            ev = self.g.events(columns=["uid", "kesim", "kategori"])
            ev["kesim"] = pd.to_datetime(ev["kesim"]).dt.date
            self._events = ev.merge(self.persons[["uid", "kesim"]], on=["uid", "kesim"], how="inner")
        return self._events


def view1(ctx: Ctx, seed: int) -> V1:
    key = ("g1", seed, ctx.a.cap_fit, ctx.a.cap_val)
    if key not in ctx._views:
        ctx._views[key] = V1(ctx, seed)
    return ctx._views[key]


def g1_vectors(ctx: Ctx, v: V1, u: str) -> np.ndarray:
    """Person vectors u (rows = v.uids): an, gc, angc, e5L1, e5L2. Encoders are fitted on train persons only."""
    if u in v._vec:
        return v._vec[u]
    tr_uids = [x for x, t in zip(v.uids, v.is_train) if t]
    v.guard.check_uids(tr_uids, f"{u} encoder fit")
    if u == "angc":
        X = np.hstack([g1_vectors(ctx, v, "an"), g1_vectors(ctx, v, "gc")])
    elif u == "an":
        S = v.g.survey.set_index("uid").reindex(v.uids).reset_index()
        X = lt.SurveyOneHot().fit(S[v.is_train]).transform(S)
    elif u == "gc":
        H = lt.history_features(v.g.history, v.uids, v.cats)
        ev = v.events()
        Mtr, vocab = lt.count_matrix(ev, tr_uids)
        M, _ = lt.count_matrix(ev, v.uids, vocab=vocab)
        X = np.hstack([H, lt.ImplicitALS(dim=32, seed=v.seed).fit(Mtr).transform(M)])
    elif u in ("e5L1", "e5L2"):
        texts = []
        for x in v.uids:
            t = v.prof.get(x, "")
            if u == "e5L2":
                aralik, hist = v.hist.get(x, ("", ""))
                t += f"\n\nTheir Amazon purchase history, {aralik}:\n{hist}"
            texts.append(t)
        X = lt.text_embedding(texts, model=ctx.a.e5, cache_dir=ctx.states_dir() / "e5")
    else:
        raise ValueError(u)
    v._vec[u] = np.asarray(X, np.float32)
    return v._vec[u]


def pred_g1(rows: pd.DataFrame, p) -> pd.DataFrame:
    return pd.DataFrame({"uid": rows.uid.to_numpy(), "kategori": rows.oge.to_numpy(), "p": np.asarray(p, float)})


def _choose(ctx, arm: Arm, seed: int, configs, score_fn, **kw):
    best, table = lt.choose(configs, score_fn, out_path=ctx.tdir(arm.task, "secim") / f"{arm.name}_s{seed}.parquet",
                            **kw)
    row = table[table.chosen].iloc[0].to_dict()
    val = {k: row[k] for k in ("kisi_auc", "1a_dogruluk_a", "logloss", "auc", "mad_acc") if k in row}
    return best, val, len(configs)


def _chosen_path(ctx, arm_name, task, seed) -> Path:
    return ctx.tdir(task, "secim") / f"{arm_name}_s{seed}.chosen.json"


def _save_chosen(ctx, arm: Arm, seed, cfg, val, precision: str | None = None):
    _chosen_path(ctx, arm.base, arm.task, seed).write_text(json.dumps({"config": cfg, "val": val,
                                                                      "precision": precision}, default=str))


def _load_chosen(ctx, arm: Arm, seed, precision: str | None = None):
    """The saved selection of arm.base, or None when there is none or it was made in another precision (a
    selection is only reused in the precision it was made in)."""
    p = _chosen_path(ctx, arm.base, arm.task, seed)
    if not p.exists():
        return None
    ch = json.loads(p.read_text())
    return ch if precision is None or ch.get("precision") == precision else None


# ------------------------------------------------------------------------------------------------- G1 LLM-free
def run_enc_g1(ctx: Ctx, arm: Arm, seed: int):
    v = view1(ctx, seed)
    X = g1_vectors(ctx, v, arm.u)
    f, va, te = v.idx(v.fit), v.idx(v.val), v.idx(v.test)
    v.guard.check(v.fit, "enc fit")
    v.guard.check(v.val, "enc select")
    configs = ctx.small([{"head": "logistic", "C": c} for c in (0.01, 0.1, 1.0)] + [{"head": "mlp"}])

    def make(c):
        return lt.EncoderHead(c["head"], C=c.get("C", 0.1), seed=seed)

    def score(c):
        h = make(c).fit(X, f[0], f[1], f[2], val=(X, va[0], va[1], va[2]))
        return lt.val_metrics_g1(h.predict(X, va[0], va[1]), v.val)

    best, val, n = _choose(ctx, arm, seed, configs, score)
    if best["head"] == "logistic":                     # refit on all train persons (prereg)
        tr = v.idx(v.train)
        v.guard.check(v.train, "enc refit")
        h = make(best).fit(X, tr[0], tr[1], tr[2])
    else:                                              # early-stopped MLP: kept as trained on fit persons
        h = make(best).fit(X, f[0], f[1], f[2], val=(X, va[0], va[1], va[2]))
    return pred_g1(v.test, h.predict(X, te[0], te[1])), {"config": best, "val": val, "n_configs": n,
                                                         "dim": int(X.shape[1])}


def _hist_long(v: V1) -> pd.DataFrame:
    """klasik history features per (uid, oge), log1p, recency missing -> 5000 days."""
    h = v.g.history.copy()
    if "son_alim" in h:
        h["son_alim"] = (pd.to_datetime(h["kesim"]) - pd.to_datetime(h["son_alim"])).dt.days
    num = [c for c in h.columns if c not in ("uid", "kesim", "oge") and pd.api.types.is_numeric_dtype(h[c])]
    for c in num:
        x = h[c].astype(float)
        if c.endswith("yas") or c == "son_alim":
            x = x.fillna(5000.0)
        h[c] = np.log1p(x.fillna(0.0).clip(lower=0))
    return h.set_index(["uid", "oge"])[num]


def _stack(ctx: Ctx, arm: Arm, seed: int, v: V1, E: np.ndarray, extra_meta: dict):
    """stk: LightGBM on the cell category's history features + PCA(k) of the person embedding E + category code."""
    from sklearn.decomposition import PCA
    hl = _hist_long(v)
    v.guard.check(v.fit, "stk fit")
    v.guard.check(v.val, "stk select")

    def feats(rows, Z):
        H = hl.reindex(pd.MultiIndex.from_arrays([rows.uid, rows.oge])).fillna(0.0).to_numpy(np.float32)
        pi, ci, _ = v.idx(rows)
        return np.hstack([H, Z[pi], ci[:, None].astype(np.float32)])

    params = {"min_data_in_leaf": 5} if ctx.kucuk else None
    fitted = {}

    def score(c):
        pca = PCA(min(c["pca"], int(v.is_train.sum()) - 1, E.shape[1]), random_state=seed).fit(E[v.is_train])
        Z = pca.transform(E).astype(np.float32)
        b, info = lt.stack_lgbm(feats(v.fit, Z), v.fit.y.to_numpy(), feats(v.val, Z), v.val.y.to_numpy(), seed=seed,
                                params=params)
        fitted[c["pca"]] = (b, Z, info["rounds"])
        return lt.val_metrics_g1(info["val_pred"], v.val)

    best, val, n = _choose(ctx, arm, seed, ctx.small([{"pca": 16}, {"pca": 32}]), score)
    b, Z, rounds = fitted[best["pca"]]
    return pred_g1(v.test, b.predict(feats(v.test, Z))), {"config": best, "val": val, "n_configs": n,
                                                          "lgbm_rounds": rounds, **extra_meta}


def run_stk_e5(ctx: Ctx, arm: Arm, seed: int):
    v = view1(ctx, seed)
    return _stack(ctx, arm, seed, v, g1_vectors(ctx, v, "e5L2"), {"embedding": "e5L2"})


# ------------------------------------------------------------------------------------------------- G1 prompt arms
def _suffixes_g1(tok, v: V1, level: str):
    """Suffix per (window, category): index w * n_cat + c (w 0 train window, 1 test window)."""
    out = []
    for w in ("train", "test"):
        msgs = P.g1_messages(level, "x", "x", "x", window=w)
        out += [P.split(tok, msgs, v.names[c])[1] for c in v.cats]
    return out


def _person_prefixes(tok, v: V1, level: str, *, cf=None):
    """Person-part prefix of each person (window-independent) and the persona block's token span."""
    pre, spans = [], []
    for u in v.uids:
        msgs = v.messages(level, u, cf=cf)
        a, _ = P.split(tok, msgs, "")
        pre.append(a)
        if level in ("L1", "L2"):
            block = msgs[-1]["content"].split("Question:")[0].strip()
            spans.append(lt.readout.token_span(tok, a, block))
    return pre, (spans or None)


def _yes(ctx, ld, prefixes, suffixes, cells, inject=None):
    uniq, inv = np.unique(np.asarray(prefixes, dtype=object).astype(str), return_inverse=True)
    cells = np.asarray(cells, np.int64).copy()
    cells[:, 0] = inv[cells[:, 0]]
    return lt.yes_prob(ld, list(uniq), suffixes, cells=cells, inject=inject, batch_size=ctx.a.batch,
                       cell_batch=ctx.a.cell_batch)


def _g1_cells(v: V1, rows):
    pi, ci, y = v.idx(rows)
    return np.stack([pi, v.window_index(rows) * len(v.cats) + ci], 1), y, ci


def run_prm_g1(ctx: Ctx, arm: Arm, seed: int):
    v = view1(ctx, seed)
    ld = ctx.ld(arm.model)
    pre, _ = _person_prefixes(ld.tok, v, arm.u)
    suf = _suffixes_g1(ld.tok, v, arm.u)
    v.guard.check(v.val, "prm calibrate")
    cv, yv, catv = _g1_cells(v, v.val)
    ct, _, catt = _g1_cells(v, v.test)
    pv, mv = _yes(ctx, ld, pre, suf, cv)
    pt, mt = _yes(ctx, ld, pre, suf, ct)
    cal = lt.platt(pv, yv, catv)
    meta = {"val": lt.val_metrics_g1(pv, v.val), "val_note": "raw P(yes), before calibration", "mass_test": float(np.mean(mt)), "mass_val": float(np.mean(mv)),
            "format_failure": bool(np.mean(mt) < 0.5), "calibration": "per-category Platt on validation persons"}
    extra = {f"{arm.name}-raw": pred_g1(v.test, pt)}
    if ctx.a.cf:
        meta["cf"] = _cf_g1(ctx, ld, v, arm, suf, ct, pt)
    return pred_g1(v.test, cal(pt, catt)), meta, extra


def _cf_g1(ctx, ld, v: V1, arm: Arm, suf, ct, p0):
    """D5b: mean change of raw P(yes) per category under the price-up / discount lines (test persons)."""
    out = {}
    for kind in ("price_up_50", "discount_30"):
        pre, _ = _person_prefixes(ld.tok, v, arm.u, cf=kind)
        p, m = _yes(ctx, ld, pre, suf, ct)
        d = pd.Series(p - p0).groupby(v.test.oge.to_numpy()).mean()
        out[kind] = {"delta": {k: round(float(x), 5) for k, x in d.items()}, "mass": float(np.mean(m))}
    up = pd.Series(out["price_up_50"]["delta"])
    dn = pd.Series(out["discount_30"]["delta"])
    out["n_categories_right_direction"] = int(((up < 0) & (dn.reindex(up.index) > 0)).sum())
    out["D5b"] = bool(out["n_categories_right_direction"] >= int(np.ceil(0.9 * len(up))))
    return out


def _probe_configs(ld, levels_pool=("last", "mean")):
    layers = sorted({ld.layer_index(f) for f in PRB_FRACS} | {ld.n_layers - 1})
    cfg = []
    for l in layers:
        for pool in levels_pool:
            cfg += [{"layer": l, "pool": pool, "est": "logistic", "pca": k, "C": C} for k in (64, 256)
                    for C in (0.01, 0.1, 1.0)]
            cfg.append({"layer": l, "pool": pool, "est": "massmean", "pca": 256, "C": 0.1})
    return layers, cfg


def _states_g1(ctx, ld, v: V1, level: str, layers, pools):
    pre, spans = _person_prefixes(ld.tok, v, level)
    return lt.llm_states(ld, pre, layers, pool=tuple(pools), spans=spans, batch_size=ctx.a.batch,
                         cache_dir=ctx.states_dir(), tag=f"g1s{v.seed}_{level}")      # one pass for all pools


def run_prb_g1(ctx: Ctx, arm: Arm, seed: int):
    v = view1(ctx, seed)
    ld = ctx.ld(arm.model)
    layers, configs = _probe_configs(ld)
    configs = ctx.small(configs)
    pools = sorted({c["pool"] for c in configs})
    H = _states_g1(ctx, ld, v, arm.u, sorted({c["layer"] for c in configs}), pools)
    f, va, te, tr = v.idx(v.fit), v.idx(v.val), v.idx(v.test), v.idx(v.train)
    v.guard.check(v.fit, "probe fit")
    v.guard.check(v.val, "probe select")

    def make(c):
        return lt.Probe(c["est"], pca=c["pca"], C=c["C"], seed=seed)

    def score(c):
        X = H[c["pool"]][c["layer"]]
        return lt.val_metrics_g1(make(c).fit(X, f[0], f[1], f[2]).predict(X, va[0], va[1]), v.val)

    best, val, n = _choose(ctx, arm, seed, configs, score)
    _save_chosen(ctx, arm, seed, best, val, ld.precision)
    X = H[best["pool"]][best["layer"]]
    v.guard.check(v.train, "probe refit")
    h = make(best).fit(X, tr[0], tr[1], tr[2])
    return pred_g1(v.test, h.predict(X, te[0], te[1])), {"config": best, "val": val, "n_configs": n,
                                                         "layer": best["layer"], "layers_swept": layers}


def run_stk_m(ctx: Ctx, arm: Arm, seed: int):
    v = view1(ctx, seed)
    ld = ctx.ld(arm.model)
    ch = _load_chosen(ctx, parse_arm(f"prb-{model_tag(arm.model)}-L2"), seed, ld.precision)
    cfg = ch["config"] if ch else {"layer": ld.layer_index(0.5), "pool": "last"}
    H = _states_g1(ctx, ld, v, "L2", [cfg["layer"]], [cfg["pool"]])[cfg["pool"]][cfg["layer"]]
    return _stack(ctx, arm, seed, v, H, {"layer": cfg["layer"], "pool": cfg["pool"],
                                         "layer_from": "prb-L2 selection" if ch else
                                         f"default 0.5 (no prb-L2 run in {ld.precision})"})


# ------------------------------------------------------------------------------------------------- G1 injection
def _short_g1(tok, v: V1, *, soft_k=0, soft_text=""):
    """Short persona-free prompts: prefix per (window, category) = index w * n_cat + c; one shared suffix."""
    pre, suf = [], None
    for w in ("train", "test"):
        for c in v.cats:
            msgs = P.fill(P.g1_messages("short", window=w), v.names[c])
            a, b = P.short_split(tok, msgs, soft_k=soft_k, soft_token=soft_text)
            pre.append(a)
            suf = b
    return pre, suf


def _trait_dirs(ctx, ld, v: V1, layers, seed):
    """E5 directions {layer: {trait: d}} from paired L1 profile edits of up to 200 fit persons."""
    S = v.g.survey
    edits = {t: e for t, e in lt.TRAIT_EDITS_G1.items() if t in S.columns}
    fit_uids = [u for u, f in zip(v.uids, v.is_fit) if f]
    v.guard.check_uids(fit_uids, "trait directions")
    profs = [v.prof[u] for u in fit_uids if u in v.prof]
    rng = np.random.default_rng(seed)
    n = 20 if ctx.kucuk else 200
    dirs = {l: {} for l in layers}
    used = {}
    for t, e in edits.items():
        ok = [p for p in profs if f"\n- {e['label']}:" in "\n" + p]
        if len(ok) < 2:
            continue
        pick = rng.choice(len(ok), min(n, len(ok)), replace=False)

        def prefix(prof):
            return P.split(ld.tok, P.g1_messages("L1", prof), "")[0]
        lo = [prefix(lt.edit_profile(ok[i], e["label"], e["levels"][0])) for i in pick]
        hi = [prefix(lt.edit_profile(ok[i], e["label"], e["levels"][-1])) for i in pick]
        st = lt.llm_states(ld, lo + hi, layers, batch_size=ctx.a.batch, cache_dir=ctx.states_dir(),
                           tag=f"caa_s{seed}_{t}")
        for l in layers:
            dirs[l][t] = (st[l][len(lo):] - st[l][:len(lo)]).mean(0).astype(np.float32)
        used[t] = e
    return dirs, used


def _derange_test(V: np.ndarray, test_rows_idx: np.ndarray, seed: int = 0) -> np.ndarray:
    """Swap control: the test persons' vectors permuted among themselves without a fixed point (default_rng(0))."""
    W = V.copy()
    t = np.asarray(sorted(set(test_rows_idx.tolist())))
    W[t] = V[t[lt.derange(len(t), seed=seed)]]
    return W


def run_caa_g1(ctx: Ctx, arm: Arm, seed: int):
    v = view1(ctx, seed)
    ld = ctx.ld(arm.model)
    pre, suf = _short_g1(ld.tok, v)
    layers = sorted({ld.layer_index(f) for f in CAA_FRACS})
    configs = ctx.small([{"layer": l, "c": c} for l in layers for c in CAA_C])
    layers = sorted({c["layer"] for c in configs})
    dirs, used = _trait_dirs(ctx, ld, v, layers, seed)
    S = v.g.survey.set_index("uid").reindex(v.uids).reset_index()
    vals = lt.trait_values(S, used)
    texts = [p + suf for p in pre[: len(v.cats)]]
    norm = {l: lt.residual_norm(ld.model, ld.tok, texts, l) for l in layers}
    v.guard.check(v.val, "caa select")
    cv, yv, catv = _g1_cells(v, v.val)
    ct, _, catt = _g1_cells(v, v.test)

    def vecs(c):
        caa = lt.CAA(dirs[c["layer"]], c["layer"], c["c"], unit_norm=norm[c["layer"]])
        return caa.vectors(vals, ref=vals[v.is_train])

    def inj(c, V):
        return lt.Injection("residual", V, layer=c["layer"], positions="last")

    def run(c, cells, V):
        cc = np.stack([cells[:, 1], np.zeros(len(cells), np.int64), cells[:, 0]], 1)
        return lt.yes_prob(ld, pre, [suf], cells=cc, inject=inj(c, V), batch_size=len(pre),
                           cell_batch=ctx.a.cell_batch)

    ch = _load_chosen(ctx, arm, seed, ld.precision)
    if ch is None:
        best, val, n = _choose(ctx, arm, seed, configs, lambda c: lt.val_metrics_g1(run(c, cv, vecs(c))[0], v.val))
        _save_chosen(ctx, arm, seed, best, val, ld.precision)
    else:
        best, val, n = ch["config"], ch["val"], len(configs)
    V = vecs(best)
    if arm.variant == "swap":
        V = _derange_test(V, ct[:, 0])
    elif arm.variant == "zero":
        V = np.zeros_like(V)
    pv, _ = run(best, cv, vecs(best))
    pt, mt = run(best, ct, V)
    cal = lt.platt(pv, yv, catv)
    meta = {"config": best, "val": val, "n_configs": n, "layer": best["layer"], "coef": best["c"],
            "unit_norm": norm[best["layer"]], "traits": sorted(used), "mass_test": float(np.mean(mt)),
            "format_failure": bool(np.mean(mt) < 0.5), "positions": "last"}
    return pred_g1(v.test, cal(pt, catt)), meta, {f"{arm.name}-raw": pred_g1(v.test, pt)}


def _std_vectors(X, train_mask):
    st = lt.Standardizer().fit(X[train_mask])
    return st.transform(X).astype(np.float32)


def _sweep(ctx, arm: Arm, seed: int, configs, fn, precision: str | None = None):
    """Validation sweep with resume: each config's score is appended to <arm>_s<seed>.sweep.jsonl. Only lines of
    the same precision are reused."""
    path = ctx.tdir(arm.task, "secim") / f"{arm.base}_s{seed}.sweep.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("precision") == precision:
                done[json.dumps(r["config"], sort_keys=True)] = r
    for c in configs:
        k = json.dumps(c, sort_keys=True)
        if k in done:
            continue
        t = time.time()
        score, hist = fn(c)
        r = {"config": c, "val": score, "seconds": round(time.time() - t, 1), "hist": hist, "precision": precision}
        done[k] = r
        with path.open("a") as f:
            f.write(json.dumps(r, default=str) + "\n")
    rows = [done[json.dumps(c, sort_keys=True)] for c in configs]
    i = int(np.nanargmax([r["val"] if r["val"] == r["val"] else -np.inf for r in rows]))
    return configs[i], rows[i]["val"], rows


def _injector(ctx: Ctx, arm: Arm, seed: int, ld, U, pre, suf_fn, cells_fit, cells_val, val_score, uids_fit, guard,
              kind: str, configs):
    """Shared rsd / sft path: sweep (1 epoch, seed 0 config reused for other seeds), final training (early stopped),
    projector saved to <task>/proj/<arm>_s<seed>.pt and reused by the -swap / -zero controls."""
    import torch
    proj_path = ctx.tdir(arm.task, "proj") / f"{arm.base}_s{seed}.pt"
    soft_id, soft_text = lt.soft_placeholder(ld.tok) if kind == "soft" else (None, "")
    target_norm = lt.embedding_norm(ld.model) if kind == "soft" else None

    def build(c):
        if kind == "soft":
            return lt.soft_token_projector(U.shape[1], ld.hidden_size, k=c["k"], target_norm=target_norm, seed=seed)
        return lt.residual_projector(U.shape[1], ld.hidden_size, seed=seed)

    def layer_of(c):
        return ld.layer_index(c["frac"]) if kind == "residual" else None

    def train(c, epochs):
        return lt.train_injector(ld, build(c), kind, pre, suf_fn(c), U, cells_fit, U, cells_val, layer=layer_of(c),
                                 positions=c.get("positions", "last"), lr=c["lr"], epochs=epochs, seed=seed,
                                 batch_size=ctx.a.train_batch, soft_id=soft_id, score_fn=val_score, guard=guard,
                                 uids_fit=uids_fit, max_seconds=ctx.a.max_train_seconds)

    if proj_path.exists() and not (ctx.a.force and arm.variant is None):
        saved = torch.load(proj_path, map_location="cpu", weights_only=False)
        if saved.get("precision") != ld.precision:     # a projector is applied in the precision it was trained in
            raise ValueError(f"{proj_path.name} was trained in {saved.get('precision')}, this run is "
                             f"{ld.precision}: rerun {arm.base} with --force in {ld.precision}, or use another --base")
        best, val, n, hist = saved["config"], saved["val"], saved["n_configs"], saved["hist"]
        proj = build(best)
        proj.load_state_dict(saved["state"])
        proj = proj.to(ld.device).eval()
    else:
        ch0 = _load_chosen(ctx, arm, 0, ld.precision) if seed != 0 else None
        if ch0 is not None:
            best, n, src = ch0["config"], len(configs), "seed 0 selection"
        else:
            best, _, rows = _sweep(ctx, arm, seed, configs, lambda c: (lambda r: (
                max(h["val"] for h in r[1]), r[1]))(train(c, 1.0)), ld.precision)
            n, src = len(configs), "validation sweep"
            _save_chosen(ctx, arm, seed, best, {"sweep_val": max(r["val"] for r in rows)}, ld.precision)
        proj, hist = train(best, 1.0 if ctx.kucuk else EPOCHS[arm.kind])
        val = max(h["val"] for h in hist)
        torch.save({"config": best, "val": val, "n_configs": n, "hist": hist, "state": proj.state_dict(),
                    "selected_by": src, "precision": ld.precision, "model": ld.code}, proj_path)
    return proj, best, val, n, hist, soft_id, layer_of(best)


def _injector_g1(ctx: Ctx, arm: Arm, seed: int, kind: str):
    v = view1(ctx, seed)
    ld = ctx.ld(arm.model)
    U = _std_vectors(g1_vectors(ctx, v, arm.u), v.is_train)
    soft_text = lt.soft_placeholder(ld.tok)[1] if kind == "soft" else ""
    pre, _ = _short_g1(ld.tok, v)
    sufs = {}

    def suf_fn(c):
        k = c.get("k", 0)
        if k not in sufs:
            sufs[k] = _short_g1(ld.tok, v, soft_k=k, soft_text=soft_text)[1]
        return sufs[k]

    def cells3(rows):
        c2, y, _ = _g1_cells(v, rows)
        return np.stack([c2[:, 1], c2[:, 0], y], 1)

    v.guard.check(v.fit, f"{kind} fit")
    v.guard.check(v.val, f"{kind} select")
    cf_, cv_ = cells3(v.fit), cells3(v.val)

    def val_score(p, cells):
        return lt.kisi_auc(p, np.asarray(cells)[:, 2], np.asarray(cells)[:, 1])

    grid = RSD if kind == "residual" else SFT
    configs = ctx.small(lt.grid(grid))
    proj, best, val, n, hist, soft_id, layer = _injector(ctx, arm, seed, ld, U, pre, suf_fn, cf_, cv_, val_score,
                                                         v.fit.uid.unique(), v.guard, kind, configs)
    ct = cells3(v.test)
    Ut = U
    if arm.variant == "swap":
        Ut = _derange_test(U, ct[:, 1])
    elif arm.variant == "zero":
        Ut = np.zeros_like(U)
    pt = lt.predict_injector(ld, proj, kind, pre, suf_fn(best), Ut, ct[:, :2], layer=layer,
                             positions=best.get("positions", "last"), soft_id=soft_id, batch_size=ctx.a.cell_batch)
    meta = {"config": best, "val": {"kisi_auc": val}, "n_configs": n, "layer": layer, "u": arm.u,
            "train_hist": hist}
    return pred_g1(v.test, pt), meta


def run_rsd_g1(ctx, arm, seed):
    return _injector_g1(ctx, arm, seed, "residual")


def run_sft_g1(ctx, arm, seed):
    return _injector_g1(ctx, arm, seed, "soft")


# ------------------------------------------------------------------------------------------------- G3 (Twin pricing)
class V3:
    def __init__(self, ctx: Ctx):
        g = lt.load_g3(ctx.root)
        g.target = _cap_target(g.target, ctx.kucuk)
        self.g = g
        t = g.target
        self.persons = t.drop_duplicates("uid")[["uid", "bolum", "ic_dogrulama"]].sort_values(["bolum", "uid"]
                                                                                             ).reset_index(drop=True)
        self.uids = self.persons.uid.tolist()
        self.pi = {u: i for i, u in enumerate(self.uids)}
        self.is_train = self.persons.bolum.eq("train").to_numpy()
        it = g.items
        self.products = it[it.rol == "hedef"].oge.tolist()
        self.prod_i = {p: i for i, p in enumerate(self.products)}
        self.names = dict(zip(it.oge, it.urun))
        self.kat = dict(zip(it.oge, it.kategori))
        self.grid = g.grid()
        self.levels = sorted(self.grid.carpan_duzey.unique().tolist())
        self.price = {(o, round(float(l), 1)): float(f) for o, l, f in
                      zip(self.grid.oge, self.grid.carpan_duzey, self.grid.fiyat)}
        self.fit, self.val, self.test = g.fit_rows(), g.val_rows(), g.test_rows()
        self.guard = g.guard()
        tp = self.persons[self.persons.bolum == "test"].uid
        self.sweep = pd.DataFrame([(u, o, float(l)) for u in tp for o in self.products for l in self.levels],
                                  columns=["uid", "oge", "carpan_duzey"])
        self._u = None

    def cell_key(self, rows) -> np.ndarray:
        """Index of (product, level) = product * n_levels + level index."""
        li = {round(float(l), 1): i for i, l in enumerate(self.levels)}
        return np.array([self.prod_i[o] * len(self.levels) + li[round(float(l), 1)]
                         for o, l in zip(rows.oge, rows.carpan_duzey)], np.int64)

    def u_tf(self) -> np.ndarray:
        """u_tf: numeric survey columns (train means for NaN) + history-product buy rates (overall, carpan <= 0.6,
        >= 1.4, per product category), standardised on train persons."""
        if self._u is not None:
            return self._u
        S = self.g.survey.set_index("uid").reindex(self.uids)
        S = S[[c for c in S.columns if pd.api.types.is_numeric_dtype(S[c])]].astype(float)
        S = S.fillna(S[self.is_train].mean()).fillna(0.0)
        h = self.g.history.copy()
        h["kat"] = h.oge.map(self.kat)
        parts = [h.groupby("uid").al.mean().rename("buy_all"),
                 h[h.carpan <= 0.6 + 1e-9].groupby("uid").al.mean().rename("buy_lo"),
                 h[h.carpan >= 1.4 - 1e-9].groupby("uid").al.mean().rename("buy_hi"),
                 h.pivot_table(index="uid", columns="kat", values="al", aggfunc="mean").add_prefix("buy_kat_")]
        Hh = pd.concat(parts, axis=1).reindex(self.uids)
        Hh = Hh.fillna(Hh[self.is_train].mean()).fillna(0.0)
        X = np.hstack([S.to_numpy(np.float32), Hh.to_numpy(np.float32)])
        self._u = _std_vectors(X, self.is_train)
        return self._u

    def profile_text(self, uid) -> str:
        if not hasattr(self, "_prof"):
            m = self.g.profile.set_index("uid")
            s = self.g.survey.set_index("uid")
            sk = [c for c in s.columns if str(c).startswith("skor_")]
            self._prof = {}
            for u in self.uids:
                row = {"metin_demografi": m.metin_demografi.get(u, "") if "metin_demografi" in m else ""}
                if u in s.index:
                    row.update({c: s.at[u, c] for c in sk})
                self._prof[u] = P.twin_profile(row)
        return self._prof[uid]

    def history_text(self, uid) -> str:
        if not hasattr(self, "_hist"):
            self._hist = {u: P.tf_history_lines(g.to_dict("records"), self.names)
                          for u, g in self.g.history.groupby("uid")}
        return self._hist.get(uid, "")


def view3(ctx) -> V3:
    if "g3" not in ctx._views:
        ctx._views["g3"] = V3(ctx)
    return ctx._views["g3"]


def _auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _g3_out(v: V3, p_test, p_sweep) -> pd.DataFrame:
    a = v.test[["uid", "oge", "carpan_duzey"]].assign(p=np.asarray(p_test, float), tur="test")
    b = v.sweep.assign(p=np.asarray(p_sweep, float), tur="sweep")
    return pd.concat([a, b], ignore_index=True)


def _tf_features(v: V3, rows) -> np.ndarray:
    U = v.u_tf()
    oh = np.zeros((len(rows), len(v.products)), np.float32)
    oh[np.arange(len(rows)), rows.oge.map(v.prod_i).to_numpy()] = 1
    return np.hstack([oh, rows.carpan_duzey.to_numpy(np.float32)[:, None], U[rows.uid.map(v.pi).to_numpy()]])


def run_enc_tf(ctx: Ctx, arm: Arm, seed: int):
    from sklearn.linear_model import LogisticRegression
    v = view3(ctx)
    v.guard.check(v.fit, "enc-tf fit")
    v.guard.check(v.val, "enc-tf select")
    Ff, Fv = _tf_features(v, v.fit), _tf_features(v, v.val)

    def make(c):
        return LogisticRegression(C=c["C"], max_iter=3000, random_state=seed)

    best, val, n = _choose(ctx, arm, seed, ctx.small([{"C": c} for c in (0.01, 0.1, 1.0)]),
                           lambda c: {"auc": _auc(v.val.y, make(c).fit(Ff, v.fit.y).predict_proba(Fv)[:, 1])},
                           primary="auc", tiebreak=())
    m = make(best).fit(np.vstack([Ff, Fv]), np.concatenate([v.fit.y, v.val.y]))
    return _g3_out(v, m.predict_proba(_tf_features(v, v.test))[:, 1],
                   m.predict_proba(_tf_features(v, v.sweep))[:, 1]), {"config": best, "val": val, "n_configs": n}


def run_lgbm_tf(ctx: Ctx, arm: Arm, seed: int):
    v = view3(ctx)
    v.guard.check(v.fit, "lgbm-tf fit")
    v.guard.check(v.val, "lgbm-tf select")
    params = {"min_data_in_leaf": 5} if ctx.kucuk else None
    b, info = lt.stack_lgbm(_tf_features(v, v.fit), v.fit.y.to_numpy(), _tf_features(v, v.val), v.val.y.to_numpy(),
                            seed=seed, params=params)
    return _g3_out(v, b.predict(_tf_features(v, v.test)), b.predict(_tf_features(v, v.sweep))), {
        "val": {"auc": _auc(v.val.y, info["val_pred"])}, "lgbm_rounds": info["rounds"]}


def _tf_suffixes(tok, v: V3, msgs):
    return [P.split(tok, msgs, P.tf_cell(v.names[o], v.price[(o, round(float(l), 1))]))[1]
            for o in v.products for l in v.levels]


def run_prm_tf(ctx: Ctx, arm: Arm, seed: int):
    v = view3(ctx)
    ld = ctx.ld(arm.model)
    persona = arm.u == "tf"

    def msgs(u):
        return P.tf_messages(v.profile_text(u), v.history_text(u)) if persona else P.tf_messages(None, None)

    pre = [P.split(ld.tok, msgs(u), "")[0] for u in v.uids]
    suf = _tf_suffixes(ld.tok, v, msgs(v.uids[0]))
    v.guard.check(v.val, "prm-tf calibrate")

    def cells(rows):
        return np.stack([rows.uid.map(v.pi).to_numpy(np.int64), v.cell_key(rows)], 1)

    pv, _ = _yes(ctx, ld, pre, suf, cells(v.val))
    pt, mt = _yes(ctx, ld, pre, suf, cells(v.test))
    ps, _ = _yes(ctx, ld, pre, suf, cells(v.sweep))
    cal = lt.platt(pv, v.val.y.to_numpy())              # one global map: the price slope stays the model's
    meta = {"val": {"auc": _auc(v.val.y, pv)}, "mass_test": float(np.mean(mt)),
            "format_failure": bool(np.mean(mt) < 0.5), "calibration": "global Platt on validation rows (carpan<=1)"}
    return _g3_out(v, cal(pt), cal(ps)), meta, {f"{arm.name}-raw": _g3_out(v, pt, ps)}


def _injector_tf(ctx: Ctx, arm: Arm, seed: int, kind: str):
    v = view3(ctx)
    ld = ctx.ld(arm.model)
    U = v.u_tf()
    soft_text = lt.soft_placeholder(ld.tok)[1] if kind == "soft" else ""

    def short(k=0):
        pre, suf = [], None
        for o in v.products:
            for l in v.levels:
                q = P.tf_question(v.names[o], v.price[(o, round(float(l), 1))])
                a, suf = P.short_split(ld.tok, [{"role": "system", "content": P.SYSTEM_SHOP},
                                                {"role": "user", "content": q}], soft_k=k, soft_token=soft_text)
                pre.append(a)
        return pre, suf

    pre, _ = short()
    sufs = {}

    def suf_fn(c):
        k = c.get("k", 0)
        if k not in sufs:
            sufs[k] = short(k)[1]
        return sufs[k]

    def cells3(rows, y=True):
        return np.stack([v.cell_key(rows), rows.uid.map(v.pi).to_numpy(np.int64),
                         rows.y.to_numpy(np.int64) if y else np.zeros(len(rows), np.int64)], 1)

    v.guard.check(v.fit, f"{kind}-tf fit")
    v.guard.check(v.val, f"{kind}-tf select")
    configs = ctx.small(lt.grid(RSD if kind == "residual" else SFT))
    proj, best, val, n, hist, soft_id, layer = _injector(
        ctx, arm, seed, ld, U, pre, suf_fn, cells3(v.fit), cells3(v.val),
        lambda p, c: _auc(np.asarray(c)[:, 2], p), v.fit.uid.unique(), v.guard, kind, configs)
    ct, cs = cells3(v.test), cells3(v.sweep, y=False)
    Ut = _derange_test(U, ct[:, 1]) if arm.variant == "swap" else np.zeros_like(U) if arm.variant == "zero" else U

    def pred(c):
        return lt.predict_injector(ld, proj, kind, pre, suf_fn(best), Ut, c[:, :2], layer=layer,
                                   positions=best.get("positions", "last"), soft_id=soft_id,
                                   batch_size=ctx.a.cell_batch)
    return _g3_out(v, pred(ct), pred(cs)), {"config": best, "val": {"auc": val}, "n_configs": n, "layer": layer,
                                            "train_hist": hist}


# ------------------------------------------------------------------------------------------------- G2 (Twin items)
class V2:
    def __init__(self, ctx: Ctx):
        g = lt.load_g2(ctx.root)
        g.target = _cap_target(g.target, ctx.kucuk)
        self.g = g
        t = g.target
        self.persons = t.drop_duplicates("uid")[["uid", "bolum", "ic_dogrulama"]].sort_values(["bolum", "uid"]
                                                                                             ).reset_index(drop=True)
        self.uids = self.persons.uid.tolist()
        self.pi = {u: i for i, u in enumerate(self.uids)}
        self.is_train = self.persons.bolum.eq("train").to_numpy()
        self.fit, self.val, self.test = g.fit_rows(), g.val_rows(), g.test_rows()
        self.guard = g.guard()
        self.items = g.items.set_index("oge")
        self.option = set(g.option_items().oge)

    def survey_x(self) -> np.ndarray:
        S = self.g.survey.set_index("uid").reindex(self.uids)
        S = S[[c for c in S.columns if pd.api.types.is_numeric_dtype(S[c])]].astype(float)
        S = S.fillna(S[self.is_train].mean()).fillna(0.0)
        return _std_vectors(S.to_numpy(np.float32), self.is_train)

    def profile_text(self, uid) -> str:
        m = self.g.profile.set_index("uid")
        s = self.g.survey.set_index("uid")
        sk = [c for c in s.columns if str(c).startswith("skor_")]
        row = {"metin_demografi": m.metin_demografi.get(uid, "") if "metin_demografi" in m else ""}
        if uid in s.index:
            row.update({c: s.at[uid, c] for c in sk})
        return P.twin_profile(row)


def view2(ctx) -> V2:
    if "g2" not in ctx._views:
        ctx._views["g2"] = V2(ctx)
    return ctx._views["g2"]


def _ridge_items(v: V2, X, alpha, fit_rows, pred_rows) -> np.ndarray:
    """Ridge per item on person vectors X (rows of fit_rows -> pred_rows), clipped to [0, 1]."""
    from sklearn.linear_model import Ridge
    out = np.full(len(pred_rows), np.nan)
    for o, g in pred_rows.groupby("oge"):
        f = fit_rows[fit_rows.oge == o]
        Xp = X[g.uid.map(v.pi).to_numpy()]
        if len(f) < 2:
            out[g.index.to_numpy()] = f.y01.mean() if len(f) else 0.5
            continue
        m = Ridge(alpha=alpha).fit(X[f.uid.map(v.pi).to_numpy()], f.y01.to_numpy(float))
        out[g.index.to_numpy()] = m.predict(Xp)
    return np.clip(out, 0, 1)


def _mad(rows, p) -> float:
    return float(lt.mad_accuracy(rows[["uid", "oge"]].assign(p=p), rows[["uid", "oge", "y01"]]).mean())


def _ridge_arm(ctx, arm, seed, v: V2, X, extra):
    v.guard.check(v.fit, "ridge fit")
    v.guard.check(v.val, "ridge select")
    best, val, n = _choose(ctx, arm, seed, ctx.small([{"alpha": a} for a in (1.0, 10.0, 100.0)]),
                           lambda c: {"mad_acc": _mad(v.val, _ridge_items(v, X, c["alpha"], v.fit, v.val))},
                           primary="mad_acc", tiebreak=())
    tr = pd.concat([v.fit, v.val], ignore_index=True)
    p = _ridge_items(v, X, best["alpha"], tr, v.test)
    return v.test[["uid", "oge"]].assign(p=p), {"config": best, "val": val, "n_configs": n, **extra}


def run_enc_tw(ctx: Ctx, arm: Arm, seed: int):
    v = view2(ctx)
    if arm.u == "tw-e5":
        X = lt.text_embedding([v.profile_text(u) for u in v.uids], model=ctx.a.e5,
                              cache_dir=ctx.states_dir() / "e5")
        return _ridge_arm(ctx, arm, seed, v, _std_vectors(X, v.is_train), {"u": "e5 of the compact profile"})
    return _ridge_arm(ctx, arm, seed, v, v.survey_x(), {"u": "kisi_ozellik numeric"})


def _tw_prompt(tok, v: V2, uid, oge):
    item = {k: (None if isinstance(x, float) and x != x else x) for k, x in v.items.loc[oge].to_dict().items()}
    msgs0, labels = P.twin_item(item, None)
    body = msgs0[-1]["content"].split("\n\n", 1)[1]
    head = f"About this person:\n{v.profile_text(uid)}\n\n"
    msgs = [msgs0[0], {"role": "user", "content": head + P.SENTINEL}]
    return P.split(tok, msgs, body), labels


def run_prm_tw(ctx: Ctx, arm: Arm, seed: int):
    v = view2(ctx)
    ld = ctx.ld(arm.model)
    rows = v.test[v.test.oge.isin(v.option)].reset_index(drop=True)
    items = sorted(v.option)
    pre = [_tw_prompt(ld.tok, v, u, items[0])[0][0] for u in v.uids]
    suf, labs = [], []
    for o in items:
        (_, s), lab = _tw_prompt(ld.tok, v, v.uids[0], o)
        suf.append(s)
        labs.append(lab)
    ii = {o: i for i, o in enumerate(items)}
    uniq, inv = np.unique(np.asarray(pre, dtype=object).astype(str), return_inverse=True)
    cells = np.stack([inv[rows.uid.map(v.pi).to_numpy()], rows.oge.map(ii).to_numpy()], 1)
    dist, mass = lt.option_probs(ld, list(uniq), suf, labs, cells=cells, batch_size=ctx.a.batch,
                                 cell_batch=ctx.a.cell_batch)
    p = np.array([float(np.dot(q, np.arange(len(q))) / max(len(q) - 1, 1)) for q in dist])
    return rows[["uid", "oge"]].assign(p=p), {"mass_test": float(np.mean(mass)), "items": "option (MC + Matrix)",
                                              "format_failure": bool(np.mean(mass) < 0.5)}


def run_prb_tw(ctx: Ctx, arm: Arm, seed: int):
    from sklearn.decomposition import PCA
    v = view2(ctx)
    ld = ctx.ld(arm.model)
    items = sorted(v.items.index)
    pre = [_tw_prompt(ld.tok, v, u, items[0])[0][0] for u in v.uids]
    layers = sorted({ld.layer_index(f) for f in PRB_FRACS} | {ld.n_layers - 1})
    configs = ctx.small([{"layer": l, "pca": k, "alpha": a} for l in layers for k in (64, 256)
                         for a in (1.0, 10.0, 100.0)])
    H = lt.llm_states(ld, pre, sorted({c["layer"] for c in configs}), batch_size=ctx.a.batch,
                      cache_dir=ctx.states_dir(), tag="g2")

    def X_of(c):
        X = H[c["layer"]]
        pca = PCA(min(c["pca"], int(v.is_train.sum()) - 1, X.shape[1]), random_state=seed).fit(X[v.is_train])
        return _std_vectors(pca.transform(X), v.is_train)

    v.guard.check(v.fit, "prb-tw fit")
    v.guard.check(v.val, "prb-tw select")
    best, val, n = _choose(ctx, arm, seed, configs,
                           lambda c: {"mad_acc": _mad(v.val, _ridge_items(v, X_of(c), c["alpha"], v.fit, v.val))},
                           primary="mad_acc", tiebreak=())
    tr = pd.concat([v.fit, v.val], ignore_index=True)
    return v.test[["uid", "oge"]].assign(p=_ridge_items(v, X_of(best), best["alpha"], tr, v.test)), {
        "config": best, "val": val, "n_configs": n, "layer": best["layer"]}


# ------------------------------------------------------------------------------------------------- dispatch
def runner(arm: Arm):
    k, u, t = arm.kind, arm.u, arm.task
    table = {("enc", "g1"): run_enc_g1, ("stk", "g1"): run_stk_e5 if arm.model is None else run_stk_m,
             ("prm", "g1"): run_prm_g1, ("prb", "g1"): run_prb_g1, ("caa", "g1"): run_caa_g1,
             ("rsd", "g1"): run_rsd_g1, ("sft", "g1"): run_sft_g1,
             ("enc", "g3"): run_enc_tf, ("lgbm", "g3"): run_lgbm_tf, ("prm", "g3"): run_prm_tf,
             ("rsd", "g3"): lambda c, a, s: _injector_tf(c, a, s, "residual"),
             ("sft", "g3"): lambda c, a, s: _injector_tf(c, a, s, "soft"),
             ("enc", "g2"): run_enc_tw, ("prm", "g2"): run_prm_tw, ("prb", "g2"): run_prb_tw}
    if (k, t) not in table:
        raise ValueError(f"{arm.name}: not implemented for {t} ({u})")
    return table[(k, t)]


def _write(ctx: Ctx, arm_name: str, task: str, seed: int, df: pd.DataFrame, meta: dict, v=None) -> Path:
    out = ctx.tdir(task)
    if task == "g1":
        return lt.write_pred(out, arm_name, seed, df, meta, n_expected=v.n_expected if v else None,
                             seal_registry=ctx.seal)
    return lt.write_pred(out, arm_name, seed, df, meta, seal_registry=ctx.seal, g1=False)


def run_one(ctx: Ctx, arm: Arm, seed: int) -> dict:
    a = ctx.a
    if arm.task != "g1" and seed != 0:
        return {"arm": arm.name, "seed": seed, "status": "skip", "why": "G2 / G3 have one split (seed 0)"}
    out = ctx.tdir(arm.task) / f"{arm.name}_s{seed}.parquet"
    if out.exists() and out.with_suffix(".meta.json").exists() and not a.force:
        done_in = json.loads(out.with_suffix(".meta.json").read_text()).get("precision")
        if arm.model is None or done_in == a.precision:
            return {"arm": arm.name, "seed": seed, "status": "skip", "why": "done", "path": str(out)}
        rec = {"zaman": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": arm.name, "task": arm.task, "seed": seed,
               "status": "error", "precision": a.precision, "path": str(out),
               "error": f"done in {done_in}, asked for {a.precision}: the file name has no precision; use another "
                        f"--base for the second precision (prereg X5) or --force to replace it"}
        ctx.log(rec)
        return rec
    try:
        import torch
        torch.set_num_threads(a.threads)
        torch.manual_seed(seed)
    except ImportError:
        pass
    t0 = time.time()
    rec = {"zaman": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": arm.name, "task": arm.task, "seed": seed,
           "kucuk": ctx.kucuk, "card": _card(), "commit": ctx.commit, "device": a.device}
    try:
        res = runner(arm)(ctx, arm, seed)
        df, meta = res[0], res[1]
        extra = res[2] if len(res) > 2 else {}
        if arm.model is not None and ctx._ld is not None:
            meta.update(ctx._ld.meta())
            meta["load_seconds"] = ctx.load_seconds
        meta.update({"arm": arm.name, "seed": seed, "task": arm.task, "seconds": round(time.time() - t0, 1),
                     "card": rec["card"], "commit": ctx.commit, "kucuk": ctx.kucuk, "cap_fit": a.cap_fit,
                     "cap_val": a.cap_val,
                     "prereg": "docs/prereg/latent-persona-2026-10-23.md"})
        v = view1(ctx, seed) if arm.task == "g1" else None
        for name, d in extra.items():
            _write(ctx, name, arm.task, seed, d, dict(meta, arm=name, secondary_of=arm.name), v)
        path = _write(ctx, arm.name, arm.task, seed, df, meta, v)
        rec.update(status="ok", path=str(path), model=meta.get("model"), hf_id=meta.get("hf_id"),
                   revision=meta.get("revision"), precision=meta.get("precision"), layer=meta.get("layer"),
                   coef=meta.get("coef"), config=meta.get("config"), val=meta.get("val"),
                   seconds=meta["seconds"], mass_test=meta.get("mass_test"))
    except Exception as e:                                           # noqa: BLE001 - logged, next run goes on
        rec.update(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:],
                   seconds=round(time.time() - t0, 1))
    ctx.log(rec)
    return rec


# ------------------------------------------------------------------------------------------------- plans (one command)
K12 = ["prm-L0", "prm-L1", "prm-L2", "prb-L1", "prb-L2", "stk"]
K12_L2 = ["prm-L2", "prb-L2", "stk"]
K3 = ["caa", "caa-swap", "caa-zero"]
K4 = ["rsd-angc", "rsd-e5L2", "rsd-angc-swap", "rsd-angc-zero"]
K5 = ["sft-angc", "sft-angc-swap"]
KG3 = ["prm-tf", "prm-tf0", "rsd-tf", "rsd-tf-swap"]
FREE = [{"arms": list(sum(LLM_FREE.values(), ())), "seeds": [0, 1, 2]}]
# prereg Extra 'priority'; G1 LLM arms run seed 0 only (stop rule: seeds 1-2 after the seed-0 score, by hand)
PLANS = {
    "80gb": FREE + [
        {"model": "q35s", "precision": "bf16", "arms": K12 + K3 + K4 + K5},
        {"model": "q38", "precision": "bf16", "arms": K12},
        {"model": "q38", "precision": "bf16", "arms": K4},
        {"model": "q38", "precision": "bf16", "arms": KG3},
        {"model": "q38", "precision": "bf16", "arms": K5},
        {"model": "q3", "precision": "bf16", "arms": K12 + K4},
        {"model": "q38", "precision": "bf16", "arms": K3},
        {"model": "g31", "precision": "bf16", "arms": K12 + K4},
        {"model": "q35", "precision": "bf16", "arms": K12},
        {"model": "g26", "precision": "bf16", "arms": K12},
        {"model": "q38", "precision": "bf16", "arms": ["prm-tw", "prb-tw"]},
        {"model": "q25", "precision": "bf16", "arms": K12}],
    "48gb": FREE + [
        {"model": "q35s", "precision": "bf16", "arms": K12 + K3 + K4 + K5},
        {"model": "q38", "precision": "fp8|nf4", "arms": K12},
        {"model": "q38", "precision": "fp8|nf4", "arms": K4},
        {"model": "q38", "precision": "fp8|nf4", "arms": KG3},
        {"model": "q38", "precision": "fp8|nf4", "arms": K3},
        {"model": "q3", "precision": "nf4", "arms": K12_L2 + K4},
        {"model": "g31", "precision": "nf4", "arms": K12_L2},
        {"model": "q35", "precision": "nf4", "arms": ["prb-L2"]},
        {"model": "q38", "precision": "fp8|nf4", "arms": K5}],
    "24gb": FREE + [
        {"model": "q35s", "precision": "bf16", "arms": K12 + K3 + K4 + K5},
        {"model": "q38", "precision": "nf4", "arms": K12_L2, "cap_fit": 1000, "cap_val": 400},
        {"model": "q38", "precision": "nf4", "arms": K4, "cap_fit": 1000, "cap_val": 400},
        {"model": "q38", "precision": "nf4", "arms": KG3},
        {"model": "q38", "precision": "nf4", "arms": K3, "cap_fit": 1000, "cap_val": 400},
        {"model": "q3", "precision": "nf4", "arms": ["prb-L2"], "cap_fit": 1000, "cap_val": 400},
        {"model": "g31", "precision": "nf4", "arms": ["prb-L2"], "cap_fit": 1000, "cap_val": 400}],
}


def _resolve_precision(text: str) -> str:
    """'fp8|nf4': fp8 when the card has compute capability >= 8.9, else nf4 (no silent bf16)."""
    if "|" not in text:
        return text
    import torch
    first, second = text.split("|")
    ok = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
    return first if ok else second


def _deadline(until: str | None):
    """Next local occurrence of HH:MM after now (a plan started at 20:00 with --until 07:30 ends next morning)."""
    if not until:
        return None
    import datetime as dt
    now = dt.datetime.now()
    h, m = map(int, until.split(":"))
    d = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return d if d > now else d + dt.timedelta(days=1)


def run_plan(ctx: Ctx, name: str) -> int:
    """Run a card plan top to bottom; no new run starts after --until (HH:MM, local time). A failed run is logged
    and the plan goes on. G1 LLM arms run seed 0 only (the prereg stop rule decides seeds 1-2 after scoring)."""
    import datetime as dt
    a = ctx.a
    end = _deadline(a.until)
    bad = 0
    todo = [(item, s, x) for item in PLANS[name] for s in item.get("seeds", [0]) for x in item["arms"]]
    for item, s, x in todo:
        if end is not None and dt.datetime.now() >= end:
            ctx.log({"kind": "plan", "plan": name, "status": "stopped", "why": f"--until {a.until}",
                     "zaman": time.strftime("%Y-%m-%d %H:%M:%S"), "left": len(todo)})
            print(json.dumps({"plan": name, "status": "stopped", "until": a.until}), flush=True)
            break
        if item.get("model"):
            a.precision = _resolve_precision(item.get("precision", "bf16"))
        a.cap_fit, a.cap_val = item.get("cap_fit"), item.get("cap_val")
        r = run_one(ctx, parse_arm(x, item.get("model")), s)
        bad += r["status"] == "error"
        print(json.dumps({k: r.get(k) for k in ("arm", "seed", "status", "seconds", "why", "error")}, default=str),
              flush=True)
    ctx.release()
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------------- calibrate / score
def calibrate(ctx: Ctx) -> dict:
    """Throughput on the card: prefill tok/s on up to 200 L2 prompts, one rsd and one sft training step, backward
    through the loaded precision, peak memory."""
    import torch
    torch.set_num_threads(ctx.a.threads)
    ld = ctx.ld(ctx.a.model)
    v = view1(ctx, 0)
    n = 5 if ctx.kucuk else 200
    pre, _ = _person_prefixes(ld.tok, v, "L2")
    pre = pre[:n]
    ntok = sum(len(x) for x in lt.readout._ids(ld.tok, pre))
    t = time.time()
    lt.prefix_states(ld, pre, layers=[ld.layer_index(0.5)], batch_size=ctx.a.batch)
    sec = time.time() - t
    out = {"kind": "calibrate", "card": _card(), **ld.meta(), "load_seconds": ctx.load_seconds, "n_prompts": len(pre),
           "tokens": ntok, "prefill_seconds": round(sec, 2), "tok_per_s": round(ntok / max(sec, 1e-9), 1),
           "check_backward": lt.check_backward(ld)}
    spre, ssuf = _short_g1(ld.tok, v)
    U = np.random.default_rng(0).normal(size=(64, 16)).astype(np.float32)
    cells = np.stack([np.arange(64) % len(spre), np.arange(64), np.arange(64) % 2], 1)
    sid, stext = lt.soft_placeholder(ld.tok)
    for kind in ("residual", "soft"):
        suf = ssuf if kind == "residual" else _short_g1(ld.tok, v, soft_k=1, soft_text=stext)[1]
        proj = (lt.residual_projector(16, ld.hidden_size) if kind == "residual" else
                lt.soft_token_projector(16, ld.hidden_size, k=1, target_norm=lt.embedding_norm(ld.model)))
        t = time.time()
        try:
            lt.train_injector(ld, proj, kind, spre, suf, U, cells, U, cells[:8], layer=ld.layer_index(0.55),
                              epochs=64 / len(cells), batch_size=64, soft_id=sid if kind == "soft" else None)
            out[f"{kind}_step_seconds"] = round(time.time() - t, 2)
        except Exception as e:                                       # noqa: BLE001
            out[f"{kind}_step_error"] = f"{type(e).__name__}: {e}"
    if torch.cuda.is_available():
        out["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    ctx.log(out)
    return out


def kucuk_panel(ctx: Ctx, seeds, tdir: Path, src_panel: Path = PANEL, src_taban: Path = TABAN_KAYNAK) -> Path:
    """--kucuk G1 scoring: degerlendir.py accepts only its full sample (400 persons), so smoke predictions are scored
    against a panel copy whose sample is the smoke test persons (every other file a symlink) and baselines cut to
    those persons. Returns the panel folder (TALP73_PANEL)."""
    dst = ctx.work / "talp73" / "panel"
    dst.mkdir(parents=True, exist_ok=True)
    for f in Path(src_panel).iterdir():
        d = dst / f.name
        if not f.name.startswith("orneklem_s") and not d.exists():
            os.symlink(f, d)
    for s in seeds:
        uids = sorted(view1(ctx, s).test.uid.unique().tolist())
        pd.DataFrame({"uid": uids}).to_parquet(dst / f"orneklem_s{s}.parquet", index=False)
        for b in ctx.a.taban:
            src, d = Path(src_taban) / f"{b}_s{s}.parquet", tdir / f"{b}_s{s}.parquet"
            if src.exists():
                if d.is_symlink():
                    d.unlink()
                t = pd.read_parquet(src)
                t[t.uid.isin(set(uids))].reset_index(drop=True).to_parquet(d, index=False)
    return dst


def score(ctx: Ctx, task: str, arms, seeds) -> list:
    res = []
    if task == "g1":
        tdir = ctx.tdir("g1")
        panel = kucuk_panel(ctx, seeds, tdir) if ctx.kucuk else None
        if not ctx.kucuk:
            lt.link_baselines(TABAN_KAYNAK, tdir, seeds=seeds)
        preds = [tdir / f"{a.name}_s{s}.parquet" for a in arms for s in seeds
                 if (tdir / f"{a.name}_s{s}.parquet").exists()]
        r = lt.score_g1(DEGERLENDIR, preds, bases=ctx.a.taban, sonuc_dir=ctx.tdir("g1", "sonuc"), B=ctx.a.B,
                        panel_dir=panel)
        return [{"pred": str(p), **(x if isinstance(x, dict) else {})} for p, x in zip(preds, r)]
    if task == "g2":
        v = view2(ctx)
        base = ctx.tdir("g2") / "enc-tw_s0.parquet"
        bdf = pd.read_parquet(base) if base.exists() else None
        for a in arms:
            pr = pd.read_parquet(ctx.tdir("g2") / f"{a.name}_s0.parquet")
            r = lt.score_g2(pr, v.test, base=bdf, B=ctx.a.B,
                            items=sorted(v.option) if a.kind == "prm" else None)
            res.append({"arm": a.name, **r})
    elif task == "g3":
        v = view3(ctx)
        truth = v.test[["uid", "oge", "carpan_duzey", "y"]]
        for a in arms:
            d = pd.read_parquet(ctx.tdir("g3") / f"{a.name}_s0.parquet")
            r = lt.score_g3(d[d.tur == "sweep"], truth, B=ctx.a.B)
            t = d[d.tur == "test"].reset_index(drop=True)
            y = v.test.y.to_numpy()
            lo = v.test.carpan_duzey.to_numpy() <= 1.0 + 1e-9
            r.update(auc_lo=_auc(y[lo], t.p[lo]), auc_hi=_auc(y[~lo], t.p[~lo]),
                     kisi_auc_lo=lt.kisi_auc(t.p[lo], y[lo], t.uid[lo]),
                     kisi_auc_hi=lt.kisi_auc(t.p[~lo], y[~lo], t.uid[~lo]))
            res.append({"arm": a.name, **r})
    for r in res:
        (ctx.tdir(task, "sonuc") / f"{r['arm']}_s0.json").write_text(json.dumps(r, indent=1, default=str))
    return res


def list_arms(ctx: Ctx) -> str:
    lines = ["LLM-free: " + " ".join(sum(LLM_FREE.values(), ())),
             "per model (<kind>-<model>-<u>[-swap|-zero]): " + "; ".join(
                 f"{k}: {', '.join(x or '(none)' for x in v)}" for k, v in LLM_U.items()),
             "models: " + " ".join(lt.MODELS) + " | tiny-<family> (no download)"]
    if ctx.log_path.exists():
        last = {}
        for line in ctx.log_path.read_text().splitlines():
            r = json.loads(line)
            if "arm" in r:
                last[(r["arm"], r["seed"])] = r
        lines.append(f"run log {ctx.log_path}:")
        lines += [f"  {a}_s{s}: {r['status']} {r.get('seconds', '')}s val={r.get('val')}" for (a, s), r in
                  sorted(last.items())]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="*", default=[], help="arm ids (see --list)")
    ap.add_argument("--model", help="model code (lt.MODELS or tiny-<family>) for arms written without one")
    ap.add_argument("--precision", default="bf16", choices=lt.PRECISIONS + ("fp8|nf4",),
                    help="fp8|nf4 = fp8 on compute capability >= 8.9, else nf4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", nargs="*", type=int, default=[0])
    ap.add_argument("--task", choices=("g1", "g2", "g3"), help="only check that every arm belongs to this task")
    ap.add_argument("--kucuk", "--smoke", nargs="?", const="40:10:20", default=None,
                    help="smoke run with FIT:VAL:TEST persons (default 40:10:20), 2 configs per grid, nothing sealed")
    ap.add_argument("--cap-fit", type=int)
    ap.add_argument("--cap-val", type=int)
    ap.add_argument("--cf", action="store_true", help="prm G1: also the price-up / discount lines (D5b)")
    ap.add_argument("--batch", type=int, default=8, help="prefixes per forward")
    ap.add_argument("--cell-batch", type=int, default=64, help="suffix cells per forward")
    ap.add_argument("--train-batch", type=int, default=64)
    ap.add_argument("--max-train-seconds", type=float)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--e5", default="intfloat/multilingual-e5-base")
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--root", help="prepared data (default <base>/veri)")
    ap.add_argument("--no-seal", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--score", choices=("g1", "g2", "g3"))
    ap.add_argument("--taban", nargs="*", default=["gecmis"], help="G1 paired baselines for --score g1")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--plan", choices=sorted(PLANS), help="the prereg priority list of a card class, top to bottom")
    ap.add_argument("--until", help="--plan: start no run after this local time (HH:MM, e.g. 07:30)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    ctx = Ctx(a)
    if a.list:
        print(list_arms(ctx))
        return 0
    a.precision = _resolve_precision(a.precision)          # 'fp8|nf4' -> one precision before any load
    if a.calibrate:
        print(json.dumps(calibrate(ctx), default=str))
        return 0
    if a.plan:
        return run_plan(ctx, a.plan)
    arms = [parse_arm(x, a.model) for x in a.arm]
    if a.task and any(x.task != a.task for x in arms):
        raise SystemExit(f"--task {a.task}: {[x.name for x in arms if x.task != a.task]} belong to another task")
    if a.score:
        for r in score(ctx, a.score, arms, a.seed):
            print(json.dumps(r, default=str)[:1000])
        return 0
    bad = 0
    for s in a.seed:
        for arm in arms:
            r = run_one(ctx, arm, s)
            bad += r["status"] == "error"
            print(json.dumps({k: r.get(k) for k in ("arm", "seed", "status", "seconds", "val", "why", "error")},
                             default=str), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
