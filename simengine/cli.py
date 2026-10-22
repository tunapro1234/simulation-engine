"""`simengine` command line: evaluate | card | bench | loop | demo.

    simengine evaluate frame.csv --ours hybrid --raw llm --primary auc        # -> runs/frame-evidence/{report.json,card.md,...}
    simengine evaluate test.csv --ours hybrid --prereg .onkayit                # no truth yet -> sealed "awaiting truth" card
    simengine card runs/card1/report.json -o card.md                           # Markdown (or *.json card dict)
    simengine bench list
    simengine bench run gss opinionqa_groups:split=dev --data-root /path/to/catalogue
    simengine loop loop.json                                                   # research loop from a JSON config
    simengine demo --out runs/demo                                             # offline end-to-end, FakeLLM

Everything here is offline except `loop` with an LLM proposer ("claude-p" / "claude") and bench data you
point it at. No command makes a network call by default.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
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

def cmd_evaluate(a) -> int:
    from . import evidence as ev
    from .evidence.onkayit import ZINCIR_ADI
    from .evidence.api import RESEARCH_CHAIN
    if a.prereg and not ev.chain_paths(a.prereg):
        print(f"error: --prereg {a.prereg!r} has no {ZINCIR_ADI} (se.seal / se.evidence.seal, default '.onkayit') "
              f"or {RESEARCH_CHAIN} (se.research.seal, default '.prereg'); pass the directory the seal was written "
              "to", file=sys.stderr)
        return 2
    df = _read_table(a.table)
    mmr = a.multiplier_max_rows
    extra = {} if mmr is None else {"multiplier_max_rows": (mmr or None)}
    report = ev.evaluate(df, title=a.title or Path(a.table).stem, ours=a.ours, raw=a.raw, task=a.task,
                         primary=a.primary, B=a.B, B_aggregate=a.B_aggregate, seed=a.seed, dataset=a.dataset or "",
                         cutoff=a.cutoff, methods=_csv_list(a.methods), circular=_csv_list(a.circular),
                         unlabeled_baselines=_csv_list(a.unlabeled_baselines), calibrate=_csv_list(a.calibrate),
                         baselines_oof=_csv_list(a.baselines_oof), ceiling_col=a.ceiling_col,
                         ceiling_label=a.ceiling_label, aggregate_by=a.aggregate_by,
                         prereg=a.prereg, multiplier=not a.no_multiplier, **extra)
    out = Path(a.out or Path("runs") / (Path(a.table).with_suffix("").name + "-evidence"))
    paths = _write_outputs(report, out, lang=a.lang)
    s = ev.summary(report)
    print(s.get("headline_en") if a.lang == "en" else s.get("headline", ""))
    print(f"badge: {s.get('badge')} (primary metric: {s.get('primary_metric')})")
    if s.get("badge") == "awaiting truth" and not a.prereg:
        print("note: awaiting truth and no --prereg: seal the predictions first (se.seal(...)) and pass the "
              "registry directory, so the card carries the seal hashes.", file=sys.stderr)
    if not a.primary:
        print(f"warning: --primary not given; graded on the task default '{s.get('primary_metric')}'. Pass the "
              "metric you pre-registered, e.g. --primary auc.", file=sys.stderr)
    for k, v in paths.items():
        print(f"{k}: {v}")
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
                      f"options (pass as {name}:key=value,...):\n{acc}\nlayout:\n{d['layout']}\n")
            else:
                print(f"{name:20s} {t.kind:13s} metrics: {', '.join(m.name for m in t.metrics)}")
        return 0
    if a.bench_cmd == "describe":
        print(json.dumps(bench.get(a.task).describe(), ensure_ascii=False, indent=1))
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
             verbose: bool = True) -> dict:
    """Offline end-to-end run: synthetic survey table -> personas -> simulated Likert survey (FakeLLM, with
    no-persona and persona-swap controls) -> EvalFrame with a label-free persistence baseline -> evidence
    report, Markdown card and HTML dashboard. FakeLLM answers are hashed noise: the card is expected to come out
    inconclusive, which is the point of the demo (the ladder catches a method with no signal)."""
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
    paths = _write_outputs(report, Path(out), dashboard_title="simengine demo")
    frame.to_csv(Path(out) / "frame.csv", index=False)
    paths["frame"] = str(Path(out) / "frame.csv")
    s = ev.summary(report)
    if verbose:
        print(f"personas: {len(panel)}  LLM calls: {len(llm.calls)}  rows: {len(frame)}")
        print(s.get("headline", ""))
        print(f"badge: {s.get('badge')}")
        for k, v in paths.items():
            print(f"{k}: {v}")
    return {"summary": s, "paths": paths, "report": report}


def cmd_demo(a) -> int:
    run_demo(a.out, n=a.n, seed=a.seed, B=a.B)
    return 0


# ------------------------------------------------------------------ parser

def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    p = argparse.ArgumentParser(prog="simengine", description="Population simulation + evidence pipeline.")
    p.add_argument("--version", action="version", version=f"simengine {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("evaluate", help="EvalFrame table (csv/tsv/parquet/jsonl) -> report.json, card.md, "
                                        "dashboard.html")
    e.add_argument("table")
    e.add_argument("--ours", help="our method (column pred_<ours>); name it before looking at results")
    e.add_argument("--raw", help="raw / unhybridised method, e.g. the plain LLM arm")
    e.add_argument("--task", choices=["binary", "multiclass", "distribution", "continuous", "ab"])
    e.add_argument("--primary", help="primary metric, e.g. auc, accuracy, pearson, wasserstein (default per task: "
                   "binary accuracy, or macro_f1 when the majority class > 60%%; see docs/modules/evidence.md)")
    e.add_argument("--title")
    e.add_argument("--dataset")
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
    e.add_argument("--calibrate", help="comma list of methods to recalibrate")
    e.add_argument("--cutoff", help="model training cutoff for the contamination audit (needs a time column)")
    e.add_argument("--prereg", help="seal registry directory to attach")
    e.add_argument("--B", type=int, default=1000)
    e.add_argument("--B-aggregate", dest="B_aggregate", type=int, default=200)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--no-multiplier", action="store_true", help="skip the PPI++ multiplier block (faster)")
    e.add_argument("--multiplier-max-rows", dest="multiplier_max_rows", type=int,
                   help="single-item frames above this many rows compute the multiplier on a fixed-seed subsample "
                        "(default 2000; 0 = whole table, slow)")
    e.add_argument("--out", help="output directory (default runs/<table>-evidence)")
    e.add_argument("--lang", choices=["mixed", "en"], default="mixed",
                   help="card language: mixed (English + research Turkish labels) or en")
    e.set_defaults(func=cmd_evaluate)

    c = sub.add_parser("card", help="report.json -> Markdown card (or card dict for *.json), optional dashboard")
    c.add_argument("report")
    c.add_argument("-o", "--output")
    c.add_argument("--dashboard", help="also write an HTML dashboard here")
    c.add_argument("--lang", choices=["mixed", "en"], default="mixed")
    c.set_defaults(func=cmd_card)

    b = sub.add_parser("bench", help="benchmark tasks with LLM-free baselines")
    bs = b.add_subparsers(dest="bench_cmd", required=True)
    bl = bs.add_parser("list")
    bl.add_argument("-v", "--verbose", action="store_true", help="also print file layouts")
    bd = bs.add_parser("describe")
    bd.add_argument("task")
    br = bs.add_parser("run")
    br.add_argument("tasks", nargs="*", help="task names, optionally name:key=val,key=val (default: all)")
    br.add_argument("--data-root", help="dataset catalogue root (else $SIMENGINE_DATA)")
    br.add_argument("--B", type=int, default=1000)
    br.add_argument("--seed", type=int, default=42)
    br.add_argument("--primary-only", action="store_true")
    br.add_argument("--out", help="write the long table as CSV")
    b.set_defaults(func=cmd_bench)

    lp = sub.add_parser("loop", help="run a research loop from a JSON config")
    lp.add_argument("config", help='{"runner": "mod:fn" | "file.py:fn", "space": {"params": {...}, "defaults": '
                                   '{...}}, "loop": {Loop kwargs}}')
    lp.add_argument("--workdir")
    lp.add_argument("--proposer", help="random | grid | greedy | halving | claude-p | claude")
    lp.add_argument("--max-turns", type=int)
    lp.add_argument("--no-finalize", action="store_true", help="do not run the test split")
    lp.add_argument("--json", action="store_true", help="print the full result dict")
    lp.set_defaults(func=cmd_loop)

    d = sub.add_parser("demo", help="offline end-to-end demo on synthetic data (FakeLLM)")
    d.add_argument("--out", default="runs/demo")
    d.add_argument("--n", type=int, default=120)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--B", type=int, default=300)
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
