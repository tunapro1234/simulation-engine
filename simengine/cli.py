"""`simengine` command line: evaluate | seal | card | bench | loop | demo.

    simengine evaluate frame.csv --ours hybrid --raw llm --primary auc        # -> runs/frame-evidence/{report.json,card.md,...}
    simengine evaluate test.csv --ours hybrid --prereg .onkayit                # no truth yet -> sealed "awaiting truth" card
    simengine evaluate pairs.csv --ours ours --primary spearman --aggregate-by item_id --offset-col member_mean \
        --noise-ceiling variance_components --memorization audit.json --blind-col ours_blind --compare P1:L1 --compare P2
    simengine evaluate bt.csv --ours lgbm --task forecast --primary wmape --aggregate-by store_dept_week \
        --aggregate sum --ci basic --n-configs 6 --unlabeled-baselines mean4 --cutoff 2016-01-01    # forecast mode
    simengine evaluate bt.csv --ours lgbm --task forecast --primary mae --block-col origin --point-pairs auto \
        --split time                                   # two-way block bootstrap, mean/median rows, declared split
    simengine seal preds.csv "test predictions" --registry runs/.onkayit --no-copy   # hash-chain record only
    simengine card runs/card1/report.json -o card.md                           # Markdown (or *.json card dict)
    simengine bench list
    simengine bench run gss opinionqa_groups:split=dev --data-root /path/to/catalogue
    simengine loop loop.json                                                   # research loop from a JSON config
    simengine demo --out runs/demo                                             # offline end-to-end, FakeLLM
    simengine demo --lang en                                                   # English headline (dot decimals)

Everything here is offline except `loop` with an LLM proposer ("claude-p" / "claude") and bench data you
point it at. No command makes a network call by default.

A missing file, an unknown task / column or a bad value ends with one line `simengine: error: <msg>` on stderr
and exit code 2; set SIMENGINE_DEBUG=1 to get the traceback instead.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence


# ------------------------------------------------------------------ helpers

def _csv_list(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _read_table(path: str):
    import pandas as pd
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in (".jsonl", ".ndjson"):
        return pd.read_json(p, lines=True)
    if p.suffix == ".json":
        return pd.read_json(p)
    return pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")


def _write_outputs(report: dict, out: Path, *, dashboard_title: str | None = None,
                   lang: str = "mixed") -> dict[str, str]:
    from . import evidence as ev
    out.mkdir(parents=True, exist_ok=True)
    paths = {"report": out / "report.json", "card": out / "card.md", "card_json": out / "card.json",
             "dashboard": out / "dashboard.html"}
    ev.save_json(report, paths["report"])
    ev.card(report, paths["card"], lang=lang)
    ev.card(report, paths["card_json"])
    ev.dashboard([report], paths["dashboard"], title=dashboard_title or report.get("baslik") or "Evidence")
    return {k: str(v) for k, v in paths.items()}


def _pred_columns(df) -> list[str]:
    """pred_* columns of a table, Turkish tahmin_* aliases shown under their English name."""
    from .types import PRED_ALIASES, PRED_PREFIX
    out = []
    for c in df.columns:
        if not isinstance(c, str):
            continue
        for p in (PRED_PREFIX, *PRED_ALIASES):
            if c.startswith(p):
                out.append(PRED_PREFIX + c[len(p):])
                break
    return out


def _check_methods(df, named: Sequence[tuple[str, str | None]]) -> None:
    """Every (flag, method) must have a pred_<method> column; else ValueError naming the available ones."""
    from .types import PRED_ALIASES, PRED_PREFIX
    avail = _pred_columns(df)
    for flag, m in named:
        if not m:
            continue
        for p in (PRED_PREFIX, *PRED_ALIASES):
            if m.startswith(p):
                m = m[len(p):]
                break
        if PRED_PREFIX + m not in avail:
            raise ValueError(f"no column {PRED_PREFIX}{m}; available: {', '.join(avail) or 'none'} ({flag})")


def _layout(s: str) -> str:
    """Re-indent a stripped BenchTask.layout: entries flush left, continuation lines keep their offset."""
    lines = (s or "").splitlines()
    rest = [ln for ln in lines[1:] if ln.strip()]
    if not rest:
        return s
    # the first line lost its class-body indent to .strip(); sibling '<root>/...' entries carry it
    ind = min((len(ln) - len(ln.lstrip()) for ln in rest if ln.lstrip().startswith("<")), default=4)
    return "\n".join(lines[:1] + [ln[min(ind, len(ln) - len(ln.lstrip())):] for ln in lines[1:]])


_EN_COLUMNS = (("tahmin_", "pred_"), ("taban_rastgele", "baseline_random"), ("taban_marjinal", "baseline_marginal"),
               ("taban_demografi", "baseline_demographic"), ("taban_", "baseline_"))


def _headline(s: dict, lang: str) -> str:
    """Printed headline: Turkish summary()['headline'], or for en headline_en with internal column names in English."""
    if lang != "en":
        return s.get("headline") or ""
    line = s.get("headline_en") or ""
    for tr, en in _EN_COLUMNS:
        line = re.sub(r"\b" + tr, en, line)
    return line


def _parse_value(v: str) -> Any:
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return v


def _task_spec(spec: str):
    """'name' or 'name:key=val,key=val' -> name or (name, options)."""
    if ":" not in spec:
        return spec
    name, rest = spec.split(":", 1)
    opts = {}
    for kv in rest.split(","):
        if not kv.strip():
            continue
        k, _, v = kv.partition("=")
        opts[k.strip()] = _parse_value(v.strip())
    return (name, opts)


def load_callable(ref: str, base_dir: Path | None = None):
    """'package.module:function' or 'path/to/file.py:function' -> the callable."""
    if ":" not in ref:
        raise ValueError(f"runner must look like 'module:function' or 'file.py:function', got {ref!r}")
    mod_ref, attr = ref.rsplit(":", 1)
    if mod_ref.endswith(".py"):
        p = Path(mod_ref)
        if not p.is_absolute() and base_dir is not None and not p.exists():
            p = base_dir / p
        spec = importlib.util.spec_from_file_location(p.stem, p)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {p}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(p.stem, mod)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_ref)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"{ref} is not callable")
    return fn


def _space_from_json(cfg: dict):
    """JSON space: params values are a list of choices, or {"range": [lo, hi]} for a numeric range."""
    from .research import Space
    params = {}
    for k, v in (cfg.get("params") or {}).items():
        if isinstance(v, dict) and "range" in v:
            lo, hi = v["range"]
            params[k] = (lo, hi)
        else:
            params[k] = list(v)
    return Space(params, defaults=cfg.get("defaults"), fixed=cfg.get("fixed", ()),
                 forbidden=cfg.get("forbidden"))


# ------------------------------------------------------------------ commands

def _compare_arg(spec: str) -> tuple:
    """--compare 'A:B' | 'A:B:label' | 'A' | 'A::label' -> (a, b | None, label | None); b empty = A against 0."""
    parts = spec.split(":", 2)
    a = parts[0].strip()
    if not a:
        raise ValueError(f"--compare {spec!r}: give A:B (paired A − B) or A (A against 0)")
    b = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    label = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
    return (a, b) if label is None else (a, b, label)


def _point_pairs_arg(spec: str | None):
    """--point-pairs 'auto' | 'MEAN:MEDIAN[,MEAN:MEDIAN...]' -> evaluate(point_pairs=) value (None when not given)."""
    if spec is None:
        return None
    if spec.strip() == "auto":
        return "auto"
    pairs = []
    for part in (x.strip() for x in spec.split(",")):
        mean, sep, median = (y.strip() for y in part.partition(":"))
        if not (sep and mean and median and ":" not in median):
            raise ValueError(f"--point-pairs {spec!r}: give auto or MEAN:MEDIAN pairs, comma separated "
                             "(e.g. ours:ours_median)")
        pairs.append((mean, median))
    return pairs


def cmd_evaluate(a) -> int:
    from . import evidence as ev
    from .evidence.onkayit import ZINCIR_ADI
    from .evidence.api import RESEARCH_CHAIN
    if a.prereg and not ev.chain_paths(a.prereg):
        print(f"error: --prereg {a.prereg!r} has no {ZINCIR_ADI} (se.seal / se.evidence.seal, default '.onkayit') "
              f"or {RESEARCH_CHAIN} (se.research.seal, default '.prereg'); pass the directory the seal was written "
              "to", file=sys.stderr)
        return 2
    if a.memorization and not Path(a.memorization).is_file():
        print(f"error: --memorization {a.memorization!r} is not a file (save the correlation_audit / "
              "memorization_audit result with se.evidence.save_json(result, 'audit.json'))", file=sys.stderr)
        return 2
    try:
        compare = [_compare_arg(c) for c in (a.compare or [])] or None
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    point_pairs = _point_pairs_arg(a.point_pairs)
    df = _read_table(a.table)
    _check_methods(df, [("--ours", a.ours), ("--raw", a.raw)]
                   + [("--methods", m) for m in _csv_list(a.methods) or []]
                   # a misspelt pair column would otherwise only fail after the whole bootstrap run
                   + [("--point-pairs", m) for pair in (point_pairs if isinstance(point_pairs, list) else [])
                      for m in pair])
    mmr = a.multiplier_max_rows
    extra = {} if mmr is None else {"multiplier_max_rows": (mmr or None)}
    if a.noise_ceiling_repeats is not None:
        extra["noise_ceiling_repeats"] = a.noise_ceiling_repeats
    agg_extra = {}
    if a.aggregate is not None:
        if a.aggregate == "sum" and not a.aggregate_by:
            print("error: --aggregate sum needs --aggregate-by (the column whose groups are summed, e.g. "
                  "store_dept_week)", file=sys.stderr)
            return 2
        agg_extra["aggregate"] = a.aggregate
    if a.ci is not None:
        agg_extra["ci"] = a.ci
    if a.n_configs is not None:
        agg_extra["n_configs"] = a.n_configs
    if a.primary and a.primary.lower() == "bias":
        print("error: --primary bias: bias is signed (sum p / sum y - 1) and cannot be the primary metric; use "
              "--primary wmape (bias is reported next to it)", file=sys.stderr)
        return 2
    extra.update(agg_extra)
    report = ev.evaluate(df, title=a.title or Path(a.table).stem, ours=a.ours, raw=a.raw, task=a.task,
                         primary=a.primary, B=a.B, B_aggregate=a.B_aggregate, seed=a.seed, dataset=a.dataset or "",
                         cutoff=a.cutoff, methods=_csv_list(a.methods), circular=_csv_list(a.circular),
                         unlabeled_baselines=_csv_list(a.unlabeled_baselines), calibrate=_csv_list(a.calibrate),
                         baselines_oof=_csv_list(a.baselines_oof), ceiling_col=a.ceiling_col,
                         ceiling_label=a.ceiling_label, aggregate_by=a.aggregate_by,
                         prereg=a.prereg, multiplier=False if a.no_multiplier else None, memorization=a.memorization,
                         offset_col=a.offset_col, noise_ceiling=a.noise_ceiling, blind_col=a.blind_col,
                         compare=compare, ours_uses_llm=False if a.ours_no_llm else None, block_col=a.block_col,
                         point_pairs=point_pairs, split=a.split, **extra)
    out = Path(a.out or Path("runs") / (Path(a.table).with_suffix("").name + "-evidence"))
    paths = _write_outputs(report, out, lang=a.lang)
    s = ev.summary(report)
    print(_headline(s, a.lang))
    print(f"badge: {s.get('badge')} (primary metric: {s.get('primary_metric')})")
    if s.get("badge") == "awaiting truth" and not a.prereg:
        print("note: awaiting truth and no --prereg: seal the predictions first (se.seal(...)) and pass the "
              "registry directory, so the card carries the seal hashes.", file=sys.stderr)
    if point_pairs == "auto" and not report.get("ortalama_medyan"):
        o = (a.ours or "<ours>").removeprefix("pred_")
        print(f"note: --point-pairs auto added no mean/median row: it needs a pred_{o}_median "
              "column, --task forecast and an L1 primary (mae / wmape); or pass MEAN:MEDIAN pairs", file=sys.stderr)
    if not a.primary:
        print(f"warning: --primary not given; graded on the task default '{s.get('primary_metric')}'. Pass the "
              "metric you pre-registered, e.g. --primary auc.", file=sys.stderr)
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


def cmd_seal(a) -> int:
    from . import evidence as ev
    if not Path(a.predictions).is_file():
        print(f"error: {a.predictions!r} is not a file", file=sys.stderr)
        return 2
    try:
        rec = ev.seal(a.predictions, a.description, registry=a.registry, git_commit=a.git,
                      store_copy=not a.no_copy, archive_dir=a.archive_dir)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"sealed: sha256 {rec.get('sha256')}  {rec.get('zaman_utc') or rec.get('zaman')}")
    print(f"registry: {a.registry}  copy: {rec.get('arsiv') or 'none (chain record only)'}")
    if not a.no_copy and a.archive_dir is None:
        print(f"note: a full canonical copy of the predictions is in {a.registry}; keep that directory git-ignored "
              "or outside the repo for licensed data, or pass --no-copy / --archive-dir.", file=sys.stderr)
    return 0


def cmd_card(a) -> int:
    from . import evidence as ev
    report = json.loads(Path(a.report).read_text(encoding="utf-8"))
    res = ev.card(report, a.output, lang=a.lang)
    if a.dashboard:
        ev.dashboard([report], a.dashboard)
        print(f"dashboard: {a.dashboard}")
    if a.output:
        print(f"card: {a.output}")
    else:
        print(res if isinstance(res, str) else json.dumps(res, ensure_ascii=False, indent=1))
    return 0


def cmd_bench(a) -> int:
    from . import bench
    if a.bench_cmd == "list":
        for name in bench.list():
            t = bench.get(name)
            if a.verbose:
                d = t.describe()
                acc = "\n".join(f"  {k}: {v}" for k, v in (d.get("accepted_options") or {}).items()) or "  (none)"
                print(f"## {name} ({d['kind']})\nmetrics: {', '.join(d['metrics'])}\n"
                      f"options (pass as {name}:key=value,...):\n{acc}\nlayout:\n{_layout(d['layout'])}\n")
            else:
                print(f"{name:20s} {t.kind:13s} metrics: {', '.join(m.name for m in t.metrics)}")
        return 0
    if a.bench_cmd == "describe":
        d = bench.get(a.task).describe()
        d["layout"] = _layout(d.get("layout") or "")
        print(json.dumps(d, ensure_ascii=False, indent=1))
        return 0
    tasks = [_task_spec(s) for s in (a.tasks or bench.list())]
    for t in tasks:                       # fail early and readably on an unknown task or option
        try:
            bench.get(t[0], **t[1]) if isinstance(t, tuple) else bench.get(t)
        except (KeyError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    table = bench.run(tasks, data_root=a.data_root, B=a.B, seed=a.seed,
                      metrics="primary" if a.primary_only else "all")
    print(bench.to_markdown(table, primary_only=a.primary_only))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(a.out, index=False)
        print(f"table: {a.out}")
    missing = table[table.get("status", "ok") == "missing_data"] if "status" in table else table.iloc[0:0]
    if len(missing):
        print(f"\n{len(missing)} task(s) skipped, data not found (pass --data-root or set SIMENGINE_DATA): "
              + ", ".join(missing["task"].astype(str)), file=sys.stderr)
    return 0


def cmd_loop(a) -> int:
    from .research import Loop
    cfg_path = Path(a.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    runner = load_callable(cfg["runner"], base_dir=cfg_path.parent)
    space = _space_from_json(cfg.get("space") or {})
    kw = dict(cfg.get("loop") or {})
    if a.workdir:
        kw["workdir"] = a.workdir
    if a.proposer:
        kw["proposer"] = a.proposer
    if a.max_turns is not None:
        kw["max_turns"] = a.max_turns
    proposer = kw.pop("proposer", "random")
    loop = Loop(runner, space, proposer, **kw)
    out = loop.run(finalize=not a.no_finalize)
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
        return 0
    fin = out.get("final") or {}
    short = {k: out.get(k) for k in ("stopped", "turns", "hit_rate") if k in out}
    short.update({k: fin.get(k) for k in ("winner_cfg", "test_diff", "verdict") if k in fin})
    print(json.dumps(short, ensure_ascii=False, indent=1, default=str))
    imp = (fin.get("verdict") or {}).get("dev_improvement")
    if imp:
        hib = (fin.get("verdict") or {}).get("higher_is_better")
        print(f"dev improvement vs start (positive = better; metric is {'higher' if hib else 'lower'}-is-better, "
              f"so dev_gain has the {'same' if hib else 'opposite'} sign): {imp['diff']:+.4f} "
              f"[{imp['lo']:+.4f}, {imp['hi']:+.4f}]")
    if kw.get("workdir"):
        print(f"workdir: {kw['workdir']} (state.json, JOURNAL.md, results.tsv)")
    return 0


def run_demo(out: str | Path = "runs/demo", *, n: int = 120, seed: int = 0, B: int = 300,
             verbose: bool = True, lang: str = "mixed") -> dict:
    """Offline end-to-end run: synthetic survey table -> personas -> simulated Likert survey (FakeLLM, with
    no-persona and persona-swap controls) -> EvalFrame with a label-free persistence baseline -> evidence
    report, Markdown card and HTML dashboard. FakeLLM answers are hashed noise: the card is expected to come out
    inconclusive, which is the point of the demo (the ladder catches a method with no signal).
    lang="en": English card and headline (summary()['headline_en'], dot decimals); "mixed" keeps the Turkish one."""
    import numpy as np
    import pandas as pd

    from . import evidence as ev
    from . import persona as sp
    from . import sim
    from .llm import FakeLLM

    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "uid": [f"u{i:03d}" for i in range(n)],
        "age": rng.integers(18, 75, n),
        "gender": rng.choice(["female", "male"], n),
        "income_tier": rng.choice(["low", "middle", "high"], n),
        "orders_last_year": rng.poisson(6, n),
    })
    # synthetic "truth": agreement with "I enjoy shopping online" (1-5), driven by last year's orders
    df["enjoys_online"] = np.clip(np.round(1.5 + df["orders_last_year"] / 3 + rng.normal(0, 0.8, n)), 1, 5)

    panel = sp.from_table(df, id_col="uid", segment="income_tier", forbidden=["enjoys_online"])
    llm = FakeLLM(sim.schema_responder(seed))
    res = sim.survey(panel, [{"id": "q1", "text": "I enjoy shopping online.", "kind": "likert5"}], llm=llm,
                     B=B)
    truth = {"q1": dict(zip(df["uid"], df["enjoys_online"].astype(float)))}
    frame = sim.survey_frame(res, truth)
    frame = frame.drop(columns=[c for c in frame.columns if c.startswith("pred_") and frame[c].isna().all()])
    # label-free baseline: last year's orders, scaled to 1-5 (never sees the target)
    last = dict(zip(df["uid"], df["orders_last_year"]))
    frame["pred_persistence"] = [float(np.clip(1.5 + last[g] / 3, 1, 5)) for g in frame["group"]]
    report = ev.evaluate(frame, title="simengine demo (FakeLLM, synthetic)", ours="score", task="continuous",
                         unlabeled_baselines=["persistence"], B=B, B_aggregate=min(B, 200), seed=seed,
                         dataset="synthetic", limits=["FakeLLM answers are hashed noise, not a measurement"])
    paths = _write_outputs(report, Path(out), dashboard_title="simengine demo", lang=lang)
    frame.to_csv(Path(out) / "frame.csv", index=False)
    paths["frame"] = str(Path(out) / "frame.csv")
    s = ev.summary(report)
    if verbose:
        print(f"personas: {len(panel)}  LLM calls: {len(llm.calls)}  rows: {len(frame)}")
        print(_headline(s, lang))
        print(f"badge: {s.get('badge')}")
        for k, v in paths.items():
            print(f"{k}: {v}")
    return {"summary": s, "paths": paths, "report": report}


def cmd_demo(a) -> int:
    run_demo(a.out, n=a.n, seed=a.seed, B=a.B, lang=a.lang)
    return 0


# ------------------------------------------------------------------ parser

EPILOG = """\
examples:
  simengine bench list
  simengine bench describe <task>
  simengine bench run <task> --out x.csv
  simengine evaluate frame.csv --ours x --lang en
  simengine demo --lang en

errors: a missing file, unknown task or column, or bad value prints one line `simengine: error: <msg>` and
exits 2 (SIMENGINE_DEBUG=1 shows the traceback)."""

LANG_HELP = ("mixed (default) = English card with the research Turkish labels and the Turkish headline "
             "(comma decimals); en = English card and English headline (dot decimals)")


# evidence.api.METRIC_TO_INTERNAL keys minus "bias", spelled out so building the parser (--help, --version, loop)
# does not import numpy/pandas; tests/test_cli_newuser.py keeps the two in sync
PRIMARY_NAMES = ("1-mae", "1-tvd", "accuracy", "ap", "auc", "average_precision", "balanced_accuracy", "brier",
                 "demeaned_r2", "ece", "effect_corr", "js", "kappa", "ks", "logloss", "macro_f1", "mae", "mode_match",
                 "one_minus_mae", "one_minus_tvd", "pair_sign", "pearson", "pr_auc", "r2", "rmse", "scale_error",
                 "simbench", "spearman", "wape", "wasserstein", "winner_top1", "within_item_corr", "within_spearman",
                 "wmape")


# evaluate(split=) labels: evidence.sema.TEST_ADLARI + evidence.api.SPLIT_DECLARED, spelled out for the same reason
# (tests/test_cli_w9.py keeps them in sync)
SPLIT_NAMES = ("test", "holdout", "deger", "degerlendirme", "oof", "cv", "time")


def _primary_names() -> str:
    return ", ".join(PRIMARY_NAMES)


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    p = argparse.ArgumentParser(prog="simengine", description="Population simulation + evidence pipeline.",
                                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"simengine {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("evaluate", help="EvalFrame table (csv/tsv/parquet/jsonl) -> report.json, card.md, "
                                        "dashboard.html")
    e.add_argument("table", help="EvalFrame file: truth + pred_<method> columns (csv, tsv, parquet, json, jsonl)")
    e.add_argument("--ours", help="our method (column pred_<ours>); name it before looking at results")
    e.add_argument("--raw", help="raw / unhybridised method, e.g. the plain LLM arm")
    e.add_argument("--task", choices=["binary", "multiclass", "distribution", "continuous", "ab", "forecast"],
                   help="forecast = demand/time-series forecast: continuous metrics + wmape and bias, default "
                        "primary wmape, no persona/human-ceiling/worth-N-humans blocks")
    e.add_argument("--primary", help="primary metric, e.g. auc, accuracy, pearson, wasserstein, wmape (default per "
                   "task: binary accuracy, or macro_f1 when the majority class > 60%%; forecast wmape; see "
                   "docs/modules/evidence.md). bias is reported but refused as primary (signed). Accepted: "
                   + _primary_names().replace("%", "%%"))
    e.add_argument("--title", help="report title on the card and dashboard (default: the table file name)")
    e.add_argument("--dataset", help="dataset name recorded in the report (free text)")
    e.add_argument("--methods", help="comma list of methods to include (default: every pred_* column)")
    e.add_argument("--unlabeled-baselines", help="comma list of label-free prediction columns (baseline rungs)")
    e.add_argument("--baselines-oof", dest="baselines_oof",
                   help="comma list of baselines trained on the target labels but cross-fitted (out-of-fold); "
                        "the badge compares ours with the strongest declared or automatic baseline")
    e.add_argument("--circular", help="comma list of priors trained on the target labels (no worth-N-humans)")
    e.add_argument("--ceiling-col", dest="ceiling_col",
                   help="model/oracle ceiling column (pred_x or any column) instead of the human retest ceiling")
    e.add_argument("--ceiling-label", dest="ceiling_label", help="name of the ceiling on the card")
    e.add_argument("--aggregate-by", dest="aggregate_by",
                   help="evaluate at group level: average truth and predictions per this column (e.g. group), "
                        "bootstrap over groups; binary/continuous only")
    e.add_argument("--aggregate", choices=["mean", "sum"],
                   help="with --aggregate-by: mean (default) or sum per group (hierarchical totals, e.g. "
                        "store x dept x week WMAPE)")
    e.add_argument("--ci", choices=["percentile", "basic"],
                   help="bootstrap CI method (default percentile; basic = 2*theta - quantiles)")
    e.add_argument("--n-configs", dest="n_configs", type=int,
                   help="configurations tried before picking ours (selection-bias audit)")
    e.add_argument("--calibrate", help="comma list of methods to recalibrate")
    e.add_argument("--cutoff", help="model training cutoff for the contamination audit (needs a time column)")
    e.add_argument("--prereg", help="seal registry directory to attach")
    e.add_argument("--memorization", help="memorization audit JSON (correlation_audit / memorization_audit result "
                   "saved with se.evidence.save_json); without it the card says the audit was skipped")
    e.add_argument("--ours-no-llm", dest="ours_no_llm", action="store_true",
                   help="ours uses no LLM arm: no 'gain may come from memorization' caveat on the badge")
    e.add_argument("--offset-col", dest="offset_col",
                   help="per-row offset (e.g. the member's training mean) subtracted from truth and every prediction "
                        "before aggregation; with --aggregate-by this scores the group residual")
    e.add_argument("--noise-ceiling", dest="noise_ceiling", choices=["variance_components", "split_half"],
                   help="sampling-noise ceiling of group means from variance components over all scored groups "
                        "(split_half = older name, same estimator; needs --aggregate-by; spearman/pearson/r2 primary)")
    e.add_argument("--noise-ceiling-repeats", dest="noise_ceiling_repeats", type=int,
                   help="random half splits for the reported split-half cross-check (default 200)")
    e.add_argument("--blind-col", dest="blind_col",
                   help="blind twin of ours (no names/titles), e.g. ours_blind: paired ours − blind row")
    e.add_argument("--compare", action="append", metavar="A:B",
                   help="registered comparison, repeatable: A:B (paired A − B), A (A against 0), A:B:label")
    e.add_argument("--block-col", dest="block_col",
                   help="block bootstrap column (e.g. the forecast origin or week column, time accepted): blocks and "
                        "clusters are resampled independently, so the CIs cover block-to-block variation too")
    e.add_argument("--point-pairs", dest="point_pairs", metavar="auto|MEAN:MEDIAN[,...]",
                   help="mean / median point forecasts of the same model: auto (ours with ours_median) or "
                        "MEAN:MEDIAN pairs, comma separated; a card row per pair in forecast mode with an L1 primary "
                        "(mae / wmape)")
    e.add_argument("--split", choices=SPLIT_NAMES,
                   help="frame without a split column: test / holdout / deger / degerlendirme fill a constant test "
                        "split; oof / cv / time declare cross-fitted predictions (every row is scored); the audit "
                        "says the split was declared, not derived")
    e.add_argument("--B", type=int, default=1000, help="bootstrap resamples for the metric CIs (default 1000)")
    e.add_argument("--B-aggregate", dest="B_aggregate", type=int, default=200,
                   help="bootstrap resamples for the slower aggregate blocks, e.g. the multiplier (default 200)")
    e.add_argument("--seed", type=int, default=0, help="random seed for bootstrap and subsampling (default 0)")
    e.add_argument("--no-multiplier", action="store_true",
                   help="skip the PPI++ multiplier block (faster; default: auto, off on binary tasks without a human "
                        "panel)")
    e.add_argument("--multiplier-max-rows", dest="multiplier_max_rows", type=int,
                   help="single-item frames above this many rows compute the multiplier on a fixed-seed subsample "
                        "(default 2000; 0 = whole table, slow)")
    e.add_argument("--out", help="output directory (default runs/<table>-evidence)")
    e.add_argument("--lang", choices=["mixed", "en"], default="mixed",
                   help="card and printed headline language: " + LANG_HELP)
    e.set_defaults(func=cmd_evaluate)

    sl = sub.add_parser("seal", help="seal a prediction file before the truth is opened (hash chain)")
    sl.add_argument("predictions", help="prediction table (csv/tsv/parquet/jsonl)")
    sl.add_argument("description", help="what is sealed, in words (stored in the chain record), e.g. "
                                        "'test predictions, hybrid v2'")
    sl.add_argument("--registry", default=".onkayit",
                    help="registry directory (default .onkayit; git-ignored, keep it out of the repo for licensed "
                         "data); pass the same directory to evaluate --prereg")
    sl.add_argument("--no-copy", dest="no_copy", action="store_true",
                    help="write only the hash-chain record, no canonical copy of the predictions (store_copy=False)")
    sl.add_argument("--archive-dir", dest="archive_dir",
                    help="write the canonical copy here instead (the chain keeps only the file name)")
    sl.add_argument("--git", action="store_true", help="git commit the chain record")
    sl.set_defaults(func=cmd_seal)

    c = sub.add_parser("card", help="report.json -> Markdown card (or card dict for *.json), optional dashboard")
    c.add_argument("report", help="report.json written by evaluate (or se.evidence.save_json)")
    c.add_argument("-o", "--output", help="write the card here: *.md = Markdown, *.json = card dict "
                                          "(default: print Markdown to stdout)")
    c.add_argument("--dashboard", help="also write an HTML dashboard here")
    c.add_argument("--lang", choices=["mixed", "en"], default="mixed", help="card language: " + LANG_HELP)
    c.set_defaults(func=cmd_card)

    b = sub.add_parser("bench", help="benchmark tasks with LLM-free baselines")
    bs = b.add_subparsers(dest="bench_cmd", required=True, metavar="{list,describe,run}")
    bl = bs.add_parser("list", help="registered tasks with kind and metrics",
                       description="List the registered benchmark tasks: name, kind and metrics (-v: options and "
                                   "the expected data layout).")
    bl.add_argument("-v", "--verbose", action="store_true", help="also print options and file layouts")
    bd = bs.add_parser("describe", help="one task as JSON: layout, source, metrics, accepted options",
                       description="Print one task's description as JSON: data layout under the catalogue root, "
                                   "source, metrics and the options accepted as name:key=value.")
    bd.add_argument("task", help="task name, see `simengine bench list`")
    br = bs.add_parser("run", help="run tasks with their LLM-free baselines -> Markdown table (+ CSV)",
                       description="Run benchmark tasks with their LLM-free baselines and print a Markdown table; "
                                   "tasks whose data is missing are skipped and listed on stderr.")
    br.add_argument("tasks", nargs="*", help="task names, optionally name:key=val,key=val (default: all)")
    br.add_argument("--data-root", help="dataset catalogue root (else $SIMENGINE_DATA)")
    br.add_argument("--B", type=int, default=1000, help="bootstrap resamples for the CIs (default 1000)")
    br.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    br.add_argument("--primary-only", action="store_true", help="score and print only each task's primary metric")
    br.add_argument("--out", help="write the long table as CSV")
    b.set_defaults(func=cmd_bench)

    lp = sub.add_parser("loop", help="run a research loop from a JSON config")
    lp.add_argument("config", help='{"runner": "mod:fn" | "file.py:fn", "space": {"params": {...}, "defaults": '
                                   '{...}}, "loop": {Loop kwargs}}')
    lp.add_argument("--workdir", help="state directory (state.json, JOURNAL.md, results.tsv); an existing one "
                                      "is resumed. Overrides loop.workdir in the config")
    lp.add_argument("--proposer", help="random | grid | greedy | halving | claude-p | claude")
    lp.add_argument("--max-turns", type=int)
    lp.add_argument("--no-finalize", action="store_true", help="do not run the test split")
    lp.add_argument("--json", action="store_true", help="print the full result dict")
    lp.set_defaults(func=cmd_loop)

    d = sub.add_parser("demo", help="offline end-to-end demo on synthetic data (FakeLLM)")
    d.add_argument("--out", default="runs/demo", help="output directory (default runs/demo)")
    d.add_argument("--n", type=int, default=120, help="synthetic respondents / personas (default 120)")
    d.add_argument("--seed", type=int, default=0, help="random seed for the data and FakeLLM (default 0)")
    d.add_argument("--B", type=int, default=300, help="FakeLLM samples per question and bootstrap resamples "
                                                       "(default 300)")
    d.add_argument("--lang", choices=["mixed", "en"], default="mixed", help="card and printed headline language: "
                                                                            + LANG_HELP)
    d.set_defaults(func=cmd_demo)
    return p


def _error_text(e: BaseException) -> str:
    if isinstance(e, FileNotFoundError) and e.filename:
        msg = f"no such file: {e.filename}"
    elif isinstance(e, KeyError) and len(e.args) == 1 and isinstance(e.args[0], str):
        k = e.args[0]
        msg = k if " " in k else f"missing key {k!r}"
    else:
        msg = str(e) or type(e).__name__
    return " ".join(msg.split())          # one line


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (FileNotFoundError, KeyError, ValueError) as e:
        if os.environ.get("SIMENGINE_DEBUG") == "1":
            raise
        print(f"simengine: error: {_error_text(e)}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
