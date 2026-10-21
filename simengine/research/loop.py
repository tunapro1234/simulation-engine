"""Turn-based research loop (port of uretim/dongu/kod/dongu.py, rules of onkayit/dongu.md v1.0-v1.2).

    loop = Loop(runner, space, proposer="random", primary="wd", higher_is_better=False, workdir="runs/x")
    result = loop.run()          # turns on the dev split, stop rule, then ONE evaluation on the test split

Contract with the runner (simengine.types.Runner): `runner(cfg)` gets the configuration plus `cfg[split_key]`
("dev" / "select" / "test" by default) and returns
    {"per_unit": DataFrame with a unit column and the primary metric column (or Series / {unit: score}),
     "metrics": {...optional extra numbers...}}
The runner is fixed: a turn changes the configuration, never the code. The loop only reads per-unit scores and
computes every number itself (no self-reported metrics).

Turn n: config = current best + one proposed change -> run on dev -> paired bootstrap diff vs the current best on the
common units -> status keep (CI excludes 0 on the good side) / discard / crash / repeat (identical per-unit scores
to an earlier turn). Stop after `stop_after` consecutive non-kept turns, `max_turns`, the proposer budget, an
exhausted proposer, or two invalid proposals in one turn. Then (finalize): optional selection split (top-3 dev
configs), optional ablation, and the winner + start + baselines run ONCE on the test split; test numbers are never fed
back. State is JSON in `workdir` (resume: run() again with the same workdir); the journal is markdown.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .prereg import verify_prereg
from .proposers import ProposalContext, make_proposer, validate_proposal
from .space import Space, config_diff, config_hash
from .stats import bootstrap_mean, direction, is_better, is_tie, paired_bootstrap_diff

STATE_VERSION = 1


def _fingerprint(fn) -> str | None:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


class Loop:
    def __init__(self, runner: Callable[[dict], dict], space: Space, proposer: Any = "random", *,
                 start: dict | None = None, primary: str = "score", higher_is_better: bool = True,
                 unit_col: str = "unit", group_col: str | None = None, stop_after: int | None = 3,
                 max_turns: int | None = 12, budget_usd: float | None = None, alpha: float = 0.05,
                 baselines: dict[str, dict] | None = None, task: str = "", workdir: str | Path | None = None,
                 split_key: str = "split", dev_split: str = "dev", select_split: str | None = None,
                 test_split: str = "test", fidelity_key: str = "fidelity", simpler_ties: bool = True,
                 ablation: bool = True, prereg: str | Path | None = None, seed: int = 42, n_boot: int = 1000,
                 check_runner: bool = True, verbose: bool = False, proposer_kw: dict | None = None):
        self.runner, self.space = runner, space
        self.proposer = make_proposer(proposer, **(proposer_kw or {}))
        start_full, err = space.validate(start or {})
        if err:
            raise ValueError(f"invalid start config: {err}")
        self.start = start_full
        self.primary, self.hib = primary, higher_is_better
        self.unit_col, self.group_col = unit_col, group_col
        self.stop_after, self.max_turns, self.budget_usd, self.alpha = stop_after, max_turns, budget_usd, alpha
        self.baselines = {}
        for a, c in (baselines or {}).items():
            full, err = space.validate(c)
            if err:
                raise ValueError(f"invalid baseline {a!r}: {err}")
            self.baselines[a] = full
        self.task = task
        self.workdir = Path(workdir) if workdir else None
        self.split_key, self.dev, self.select, self.test = split_key, dev_split, select_split, test_split
        self.fidelity_key = fidelity_key
        self.simpler_ties, self.ablation = simpler_ties, ablation
        self.prereg = Path(prereg) if prereg else None
        self.seed, self.n_boot, self.verbose = seed, n_boot, verbose
        self.check_runner = check_runner
        self._units: dict[str, pd.Series] = {}     # eval key -> per-unit primary (index = unit as str)
        self._frames: dict[str, pd.DataFrame] = {}
        self.state = self._load_state() or self._new_state()

    # ================================================================ state
    def _new_state(self) -> dict:
        return {"version": STATE_VERSION, "runner_fingerprint": _fingerprint(self.runner),
                "space_fingerprint": self.space.fingerprint(), "start": self.start, "primary": self.primary,
                "higher_is_better": self.hib, "proposer": getattr(self.proposer, "name", type(self.proposer).__name__),
                "baselines": {}, "turns": [], "pending": None, "stopped": None,
                "counters": {"proposals": 0, "invalid": 0, "repeat_rejected": 0, "test_refs": 0, "repairs_ok": 0},
                "cost_usd": 0.0, "evals": {}, "final": None, "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    def _state_path(self):
        return self.workdir / "state.json" if self.workdir else None

    def _load_state(self) -> dict | None:
        p = self._state_path()
        if p is None or not p.exists():
            return None
        st = json.loads(p.read_text(encoding="utf-8"))
        fp = _fingerprint(self.runner)
        if self.check_runner and st.get("runner_fingerprint") and fp and st["runner_fingerprint"] != fp:
            raise RuntimeError("runner source changed since the loop started (fingerprint "
                               f"{st['runner_fingerprint']} != {fp}); a turn may change the config, never the runner. "
                               "Start a new workdir or pass check_runner=False.")
        if st.get("space_fingerprint") != self.space.fingerprint():
            raise RuntimeError("config space changed since the loop started; start a new workdir")
        if st.get("start") != _jsonable(self.start):
            raise RuntimeError("start config differs from the saved state; start a new workdir")
        return st

    def _save(self):
        if self.workdir is None:
            return
        self.workdir.mkdir(parents=True, exist_ok=True)
        tmp = self.workdir / "state.json.tmp"
        tmp.write_text(json.dumps(_jsonable(self.state), indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(self.workdir / "state.json")
        self._write_journal()
        self._write_table()

    # ================================================================ evaluation
    def _key(self, cfg: dict, split: str, fidelity: float | None = None) -> str:
        k = f"{split}_{config_hash(cfg)}"
        return k if fidelity is None else f"{k}_f{fidelity:.4g}"

    def _per_unit(self, res: dict) -> pd.DataFrame:
        pu = res.get("per_unit")
        if pu is None:
            raise ValueError("runner must return 'per_unit' (one score per unit) for paired comparisons")
        if isinstance(pu, dict):
            pu = pd.DataFrame({self.unit_col: list(pu), self.primary: list(pu.values())})
        elif isinstance(pu, pd.Series):
            pu = pd.DataFrame({self.unit_col: pu.index, self.primary: pu.values})
        elif not isinstance(pu, pd.DataFrame):
            pu = pd.DataFrame(pu)
        if self.unit_col not in pu.columns:
            pu = pu.reset_index().rename(columns={"index": self.unit_col})
        if self.primary not in pu.columns:
            raise ValueError(f"per_unit has no primary column {self.primary!r}")
        pu = pu.copy()
        pu[self.unit_col] = pu[self.unit_col].astype(str)
        return pu

    def _smoke(self, pu: pd.DataFrame, split: str) -> str | None:
        """Smoke gate (dongu v1.1 §2, generic part): units present, unique, finite primary; same n as turn 0 on dev."""
        if len(pu) == 0:
            return "no units"
        if pu[self.unit_col].duplicated().any():
            return "duplicate unit ids"
        v = pd.to_numeric(pu[self.primary], errors="coerce").to_numpy(float)
        if not np.isfinite(v).all():
            return f"non-finite {self.primary} in {int((~np.isfinite(v)).sum())} units"
        n0 = self.state.get("dev_n")
        if split == self.dev and n0 is not None and len(pu) != n0:
            return f"unit count {len(pu)} != {n0} of turn 0"
        return None

    def _units_path(self, key):
        return self.workdir / "units" / f"{key}.csv" if self.workdir else None

    def evaluate(self, cfg: dict, split: str | None = None, *, fidelity: float | None = None,
                 _allow_test: bool = False) -> dict:
        """Run the fixed runner once per (config, split[, fidelity]) and cache the summary. The test split is only
        reachable from finalize()."""
        split = split or self.dev
        if split == self.test and not _allow_test:
            raise PermissionError("the test split is evaluated only once, by finalize()")
        key = self._key(cfg, split, fidelity)
        if key in self.state["evals"]:
            return self.state["evals"][key]
        call = {**cfg, self.split_key: split}
        if fidelity is not None:
            call[self.fidelity_key] = fidelity
        t0 = time.time()
        try:
            res = self.runner(call) or {}
            if res.get("status") == "crash":
                raise RuntimeError(res.get("error", "runner reported crash"))
            pu = self._per_unit(res)
            err = self._smoke(pu, split) if fidelity is None else None
        except Exception as e:  # a crashing turn is recorded, not fatal
            res, pu, err = {}, None, f"{type(e).__name__}: {str(e)[:500]}"
        summ = {"key": key, "split": split, "cfg_hash": config_hash(cfg), "seconds": round(time.time() - t0, 3)}
        if err:
            summ.update(status="crash", error=err)
        else:
            metrics = {}
            for c in pu.columns:
                if c in (self.unit_col, self.group_col):
                    continue
                x = pd.to_numeric(pu[c], errors="coerce")
                if x.notna().all():
                    metrics[c] = bootstrap_mean(x.to_numpy(float), seed=self.seed, B=self.n_boot, alpha=self.alpha)
            summ.update(status="ok", n=len(pu), metrics=metrics, runner_metrics=_jsonable(res.get("metrics", {})))
            if self.group_col and self.group_col in pu.columns:
                summ["groups"] = {str(g): round(float(v), 4) for g, v in pu.groupby(self.group_col)[self.primary].mean().items()}
            self._units[key] = pu.set_index(self.unit_col)[self.primary].astype(float)
            self._frames[key] = pu
            p = self._units_path(key)
            if p is not None:
                p.parent.mkdir(parents=True, exist_ok=True)
                pu.to_csv(p, index=False)
        self.state["evals"][key] = summ
        return summ

    def units(self, key: str) -> pd.Series:
        if key not in self._units:
            p = self._units_path(key)
            if p is None or not p.exists():
                raise KeyError(f"per-unit scores for {key} not available")
            pu = pd.read_csv(p, dtype={self.unit_col: str}, float_precision="round_trip")
            self._units[key] = pu.set_index(self.unit_col)[self.primary].astype(float)
        return self._units[key]

    def paired(self, key_a: str, key_b: str, alpha: float | None = None) -> dict:
        """mean(a - b) over common units with a bootstrap CI (negative = a lower)."""
        a, b = self.units(key_a), self.units(key_b)
        common = a.index.intersection(b.index)
        return paired_bootstrap_diff(a.loc[common].values, b.loc[common].values, seed=self.seed, B=self.n_boot,
                                     alpha=alpha or self.alpha)

    def _better(self, f):
        return is_better(f, self.hib)

    def _rank(self, m):
        return -m["value"] if self.hib else m["value"]

    # ================================================================ turns
    def _incumbent(self) -> dict | None:
        inc = None
        for t in self.state["turns"]:
            if t["status"] == "keep":
                inc = t
        return inc

    def _consecutive_not_kept(self) -> int:
        c = 0
        for t in reversed(self.state["turns"][1:]):
            if t["status"] == "keep":
                break
            c += 1
        return c

    def _run_turn(self, cfg: dict, proposal: dict | None):
        st = self.state
        n = len(st["turns"])
        inc = self._incumbent()
        ev = self.evaluate(cfg, self.dev)
        t = {"n": n, "cfg": cfg, "cfg_hash": config_hash(cfg), "proposal": proposal, "eval_key": ev["key"],
             "diff_start": config_diff(cfg, self.start), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        if ev["status"] == "crash":
            t.update(status="crash", error=ev["error"], inc_before=inc["n"] if inc else None)
            if inc is None:
                raise RuntimeError(f"the start configuration crashed: {ev['error']}")
        else:
            t["metrics"] = ev["metrics"]
            t["n_units"] = ev["n"]
            if ev.get("groups"):
                t["groups"] = ev["groups"]
            if n == 0:
                st["dev_n"] = ev["n"]
            cur = self.units(ev["key"]).sort_index()
            for o in st["turns"]:
                if o["status"] in ("crash",) or o["eval_key"] == ev["key"]:
                    continue
                prev = self.units(o["eval_key"]).sort_index()
                if prev.index.equals(cur.index) and np.allclose(prev.values, cur.values, atol=1e-12, rtol=0):
                    t["repeat_of"] = o["n"]
                    break
            if inc is None:
                t["status"] = "keep"
            else:
                t["inc_before"] = inc["n"]
                f = t["diff_inc"] = self.paired(ev["key"], inc["eval_key"])
                t["observed"] = direction(f)
                if t.get("repeat_of") is not None:
                    t["status"] = "repeat"
                elif self._better(f):
                    t["status"] = "keep"
                elif (self.simpler_ties and is_tie(f) and self.space.complexity(cfg) is not None
                      and self.space.complexity(cfg) < self.space.complexity(inc["cfg"])):
                    t["status"] = "keep"
                    t["kept_by"] = "simpler tie"
                else:
                    t["status"] = "discard"
                p = proposal or {}
                if p.get("expected") in ("lower", "same", "higher"):
                    t["hit"] = p["expected"] == t["observed"]
        st["turns"].append(t)
        st["pending"] = None
        if self.verbose:
            m = (t.get("metrics") or {}).get(self.primary, {})
            print(f"turn {n} {t['status']} {self.primary}={m.get('value')} change={proposal and proposal.get('change')}",
                  flush=True)
        return t

    def _stop_reason(self) -> str | None:
        st = self.state
        n = len(st["turns"]) - 1
        if self.stop_after and self._consecutive_not_kept() >= self.stop_after:
            return f"{self.stop_after} consecutive turns not kept (turn {n})"
        if self.max_turns is not None and n >= self.max_turns:
            return f"turn limit {self.max_turns}"
        if self.budget_usd is not None and st["cost_usd"] >= self.budget_usd:
            return f"proposer budget ${self.budget_usd:.2f} used (${st['cost_usd']:.4f})"
        return None

    def _context(self, error=None) -> ProposalContext:
        inc = self._incumbent()
        hist = [{k: v for k, v in t.items()} for t in self.state["turns"]]
        bl = {a: self.state["evals"][b["eval_key"]].get("metrics", {}) for a, b in self.state["baselines"].items()
              if b.get("eval_key") in self.state["evals"]}
        return ProposalContext(turn=len(self.state["turns"]), incumbent={"n": inc["n"], "cfg": inc["cfg"],
                                                                        "metrics": inc.get("metrics")},
                               start=self.start, history=hist, space=self.space,
                               seen={t["cfg_hash"] for t in self.state["turns"]}, primary=self.primary,
                               higher_is_better=self.hib, task=self.task, baselines=bl, error=error,
                               stop_after=self.stop_after, max_turns=self.max_turns, screen=self._screen)

    def _screen(self, cfgs: list, fidelity: float) -> list:
        out = []
        for c in cfgs:
            ev = self.evaluate(c, self.dev, fidelity=fidelity)
            self.state["counters"]["screen_runs"] = self.state["counters"].get("screen_runs", 0) + 1
            out.append(ev["metrics"][self.primary]["value"] if ev["status"] == "ok" else float("nan"))
        return out

    def _propose(self):
        """One proposal with one repair attempt. Sets state['pending'] or state['stopped']."""
        st, c = self.state, self.state["counters"]
        req = getattr(self.proposer, "requires_fields", False)
        notes = []
        error, prop = None, None
        for attempt in range(2):
            ctx = self._context(error)
            prop = self.proposer.propose(ctx)
            if prop is None:
                if attempt == 0 and error is None:
                    st["stopped"] = "proposer has nothing left to propose"
                    return
            c["proposals"] += 1
            st["cost_usd"] += float((prop or {}).get("cost_usd", 0.0) or 0.0)
            cfg, err = validate_proposal(prop, ctx, require_fields=req)
            if err is None:
                if attempt == 1:
                    c["repairs_ok"] += 1
                    notes.append(f"first proposal invalid ({error}); repair valid")
                rec = {k: v for k, v in prop.items() if k not in ("prompt",)}
                rec["attempt_notes"] = notes
                st["pending"] = {"cfg": cfg, "proposal": rec, "from_best": ctx.incumbent["n"]}
                self._log_raw(ctx.turn, prop, attempt)
                return
            c["invalid"] += 1
            c["repeat_rejected"] += err.startswith("repeat")
            c["test_refs"] += err.startswith("reply refers to the test")
            self._log_raw(ctx.turn, prop, attempt, err)
            if attempt == 0:
                error = err
                notes.append(f"invalid: {err}")
            else:
                st["stopped"] = f"proposer invalid twice (turn {ctx.turn}): {error}; {err}"
                if st["turns"]:
                    st["turns"][-1]["proposal_failed"] = notes + [f"repair invalid: {err}"]

    def _log_raw(self, turn, prop, attempt, err=None):
        if self.workdir is None or not prop or "raw" not in prop:
            return
        d = self.workdir / "observer"
        d.mkdir(parents=True, exist_ok=True)
        suffix = "" if attempt == 0 else "_repair"
        (d / f"turn{turn}{suffix}_prompt.txt").write_text(prop.get("prompt", ""), encoding="utf-8")
        (d / f"turn{turn}{suffix}_reply.txt").write_text((prop.get("raw") or "") + (f"\n\n[rejected: {err}]" if err else ""),
                                                         encoding="utf-8")

    # ================================================================ main
    def _check_prereg(self, when):
        if self.prereg is None:
            return
        v = verify_prereg(self.prereg)
        if not v["ok"]:
            raise RuntimeError(f"pre-registration check failed {when}: {v['reason']} ({self.prereg})")
        self.state.setdefault("prereg", {"file": self.prereg.name, "sha256": v["sha256"], "locked_at": v["locked_at"]})

    def run(self, finalize: bool = True, extra_test: dict[str, dict] | None = None) -> dict:
        """Run (or resume) turns until the stop rule fires, then finalize() unless finalize=False."""
        self._check_prereg("at start")
        st = self.state
        for a, c in self.baselines.items():
            if a not in st["baselines"]:
                ev = self.evaluate(c, self.dev)
                st["baselines"][a] = {"cfg": c, "eval_key": ev["key"]}
        self._save()
        while not st["stopped"]:
            if not st["turns"]:
                self._run_turn(self.start, None)
            elif st["pending"]:
                self._run_turn(st["pending"]["cfg"], st["pending"]["proposal"])
            else:
                self._propose()
                self._save()
                continue
            reason = self._stop_reason()
            if reason:
                st["stopped"] = reason
                self._save()
                break
            self._propose()
            self._save()
        if finalize:
            return self.finalize(extra_test=extra_test)
        return self.result()

    def finalize(self, extra_test: dict[str, dict] | None = None) -> dict:
        """Selection (optional), ablation (optional) and the single test evaluation. Idempotent: the test split runs
        once; calling again returns the saved result."""
        st = self.state
        if st.get("final"):
            return self.result()
        if not st["stopped"]:
            raise RuntimeError("finalize() before the loop stopped")
        self._check_prereg("before the test evaluation")
        fin = {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
        ok = [t for t in st["turns"] if t["status"] != "crash"]
        inc = self._incumbent()
        winner = inc
        if self.select:
            cands, seen = [], set()
            for t in sorted(ok, key=lambda t: self._rank(t["metrics"][self.primary])):
                if t["cfg_hash"] not in seen:
                    seen.add(t["cfg_hash"])
                    cands.append(t)
                if len(cands) == 3:
                    break
            sel = []
            for t in cands:
                ev = self.evaluate(t["cfg"], self.select)
                if ev["status"] == "ok":
                    sel.append({"turn": t["n"], "cfg_hash": t["cfg_hash"], "dev": t["metrics"][self.primary],
                                "select": ev["metrics"][self.primary], "eval_key": ev["key"]})
            fin["selection"] = sel
            if sel:
                best = min(sel, key=lambda s: self._rank(s["select"]))
                winner = st["turns"][best["turn"]]
        fin["winner_turn"] = winner["n"]
        fin["winner_cfg"] = winner["cfg"]
        # ablation: revert each changed key of the winner to its start value, one at a time
        abl_split = self.select or self.dev
        diffs = config_diff(winner["cfg"], self.start)
        k_abl = max(len(diffs), 1)
        abl = []
        if self.ablation and diffs:
            w_ev = self.evaluate(winner["cfg"], abl_split)
            for k in diffs:
                c, err = self.space.validate({**winner["cfg"], k: self.start.get(k)})
                if err or config_hash(c) == config_hash(winner["cfg"]):
                    continue
                ev = self.evaluate(c, abl_split)
                if ev["status"] != "ok" or w_ev["status"] != "ok":
                    abl.append({"reverted": k, "status": "crash"})
                    continue
                abl.append({"reverted": k, "to": self.start.get(k), "split": abl_split,
                            "metric": ev["metrics"][self.primary],
                            "diff_reverted_minus_winner": self.paired(ev["key"], w_ev["key"]),
                            "diff_bonferroni": self.paired(ev["key"], w_ev["key"], alpha=self.alpha / k_abl)})
        fin["ablation"] = abl
        # the single test evaluation
        test_cfgs = {"winner": winner["cfg"], "start": self.start, **self.baselines, **(extra_test or {})}
        tests = {}
        for a, c in test_cfgs.items():
            ev = self.evaluate(c, self.test, _allow_test=True)
            tests[a] = {"cfg_hash": config_hash(c), "status": ev["status"], "eval_key": ev["key"],
                        "metrics": ev.get("metrics"), "n": ev.get("n"), "error": ev.get("error")}
        fin["test"] = tests
        k = max(len(st["turns"]) - 1, 1)  # number of proposal turns = comparisons
        fin["bonferroni_k"] = k
        cmp = {}
        for a in test_cfgs:
            if a == "winner" or tests[a]["status"] != "ok" or tests["winner"]["status"] != "ok":
                continue
            cmp[f"winner-{a}"] = {"ci95": self.paired(tests["winner"]["eval_key"], tests[a]["eval_key"]),
                                  "bonferroni": self.paired(tests["winner"]["eval_key"], tests[a]["eval_key"],
                                                            alpha=self.alpha / k)}
        fin["test_diff"] = cmp
        # falsification lines of dongu.md: (a) winner does not beat the start/baselines on test with the corrected CI;
        # (b) more than half of the proposals invalid or wasted
        wasted = sum(t["status"] == "repeat" for t in st["turns"])
        rate = (st["counters"]["invalid"] + wasted) / max(st["counters"]["proposals"], 1)
        refs = [a for a in cmp]
        fin["verdict"] = {
            "beats_all_on_test": bool(refs) and all(self._better(cmp[a]["bonferroni"]) for a in refs),
            "beats_start_on_test": self._better(cmp["winner-start"]["bonferroni"]) if "winner-start" in cmp else None,
            "invalid_or_wasted_rate": rate, "proposer_failed": rate > 0.5,
            "dev_gain": (self.paired(winner["eval_key"], st["turns"][0]["eval_key"])
                         if winner["n"] != 0 else None),
            "higher_is_better": bool(self.hib),
        }
        g = fin["verdict"]["dev_gain"]
        # dev_gain is winner - start in metric units; dev_improvement is the same in "positive = better" units
        fin["verdict"]["dev_improvement"] = None if not g or g.get("diff") is None else (
            {"diff": g["diff"], "lo": g["lo"], "hi": g["hi"]} if self.hib else
            {"diff": -g["diff"], "lo": None if g.get("hi") is None else -g["hi"],
             "hi": None if g.get("lo") is None else -g["lo"]})
        if winner["n"] == 0:
            fin["verdict"]["note"] = "no turn beat the start configuration on dev"
        st["final"] = fin
        self._save()
        return self.result()

    # ================================================================ outputs
    def result(self) -> dict:
        st = self.state
        inc = self._incumbent()
        hits = [t["hit"] for t in st["turns"] if "hit" in t]
        return {"stopped": st["stopped"], "turns": len(st["turns"]),
                "kept": sum(t["status"] == "keep" for t in st["turns"][1:]),
                "best_turn": inc["n"] if inc else None, "best_cfg": inc["cfg"] if inc else None,
                "best_dev": (inc.get("metrics") or {}).get(self.primary) if inc else None,
                "counters": dict(st["counters"]), "cost_usd": st["cost_usd"],
                "hit_rate": (sum(hits) / len(hits)) if hits else None, "final": st.get("final"),
                "history": st["turns"]}

    def _write_table(self):
        rows = []
        for a, b in self.state["baselines"].items():
            m = self.state["evals"].get(b["eval_key"], {}).get("metrics", {}).get(self.primary, {})
            rows.append({"turn": f"baseline:{a}", self.primary: m.get("value"), "ci_lo": m.get("lo"), "ci_hi": m.get("hi"),
                         "n": m.get("n")})
        for t in self.state["turns"]:
            m = (t.get("metrics") or {}).get(self.primary, {})
            f = t.get("diff_inc") or {}
            p = t.get("proposal") or {}
            rows.append({"turn": t["n"], "cfg_hash": t["cfg_hash"], self.primary: m.get("value"), "ci_lo": m.get("lo"),
                         "ci_hi": m.get("hi"), "n": m.get("n"), "best_before": t.get("inc_before"),
                         "diff": f.get("diff"), "diff_lo": f.get("lo"), "diff_hi": f.get("hi"), "status": t["status"],
                         "kind": p.get("kind"), "expected": p.get("expected"), "observed": t.get("observed"),
                         "hit": t.get("hit"), "change": json.dumps(p.get("change"), default=str) if p else ""})
        pd.DataFrame(rows).to_csv(self.workdir / "results.tsv", sep="\t", index=False, float_format="%.5f")

    def journal_markdown(self) -> str:
        st = self.state
        P = self.primary
        L = [f"# Research loop journal", "",
             f"Primary metric `{P}` ({'higher' if self.hib else 'lower'} is better) on split `{self.dev}`; keep rule: "
             f"paired bootstrap {100 * (1 - self.alpha):.0f}% CI of the difference vs the current best excludes 0"
             + (" (or a statistically tied, simpler config)" if self.simpler_ties and self.space._complexity is not None
                else "") + f". Stop: "
             f"{self.stop_after} consecutive non-kept turns, {self.max_turns} turns"
             + (f", ${self.budget_usd} proposer budget" if self.budget_usd is not None else "")
             + f". Proposer `{st['proposer']}`. Numbers are computed by the loop from per-unit scores; observer text is "
               "quoted as written.", ""]
        if st.get("prereg"):
            L += [f"Pre-registration `{st['prereg']['file']}` sha256 {st['prereg']['sha256'][:12]} "
                  f"(locked {st['prereg']['locked_at']}), verified at start.", ""]
        if st["baselines"]:
            L.append("Baselines (dev):")
            for a, b in st["baselines"].items():
                m = st["evals"].get(b["eval_key"], {}).get("metrics", {}).get(P)
                L.append(f"- {a}: {P} {_m3(m)}")
            L.append("")
        for t in st["turns"]:
            L.append(f"## Turn {t['n']} · {t['time']}")
            L.append(f"- config vs start: `{json.dumps(t['diff_start'], default=str)}` · status **{t['status']}**"
                     + (f" ({t['kept_by']})" if t.get("kept_by") else ""))
            p = t.get("proposal")
            if p:
                L.append(f"- change: `{json.dumps(p.get('change'), default=str)}` from best turn {t.get('inc_before')}"
                         f" · kind {p.get('kind')}")
                if p.get("hypothesis") not in (None, "-"):
                    L.append(f"- hypothesis: {p.get('hypothesis')} · expected {p.get('expected')} · "
                             f"falsifier: {p.get('falsifier')}")
                if p.get("rationale"):
                    L.append(f"- rationale: {p.get('rationale')}")
                for nn in p.get("attempt_notes") or []:
                    L.append(f"- note: {nn}")
            if t["status"] == "crash":
                L.append(f"- crash: {t.get('error')}")
            else:
                L.append("- metrics (dev): " + "  ".join(f"{k}={_m3(v)}" for k, v in t["metrics"].items())
                         + f" · n={t.get('n_units')}")
                if t.get("diff_inc"):
                    L.append(f"- paired {P} diff vs best (turn {t['inc_before']}): {_f3(t['diff_inc'])} · observed "
                             f"{t.get('observed')}" + (f" · expected {p.get('expected')}" if p and
                                                        p.get("expected") in ("lower", "same", "higher") else ""))
                if t.get("repeat_of") is not None:
                    L.append(f"- wasted: per-unit scores identical to turn {t['repeat_of']}")
            nxt = st["turns"][t["n"] + 1]["proposal"] if t["n"] + 1 < len(st["turns"]) else \
                (st["pending"] or {}).get("proposal") if t is st["turns"][-1] else None
            notes = (nxt or {}).get("notes") or {}
            for k, lab in (("what_changed", "observer: what changed"), ("surprising", "observer: surprising"),
                           ("leakage_suspicion", "observer: leakage suspicion")):
                if notes.get(k):
                    L.append(f"- {lab}: {notes[k]}")
            if t.get("proposal_failed"):
                L.append(f"- proposal failed: {'; '.join(t['proposal_failed'])}")
            if t is st["turns"][-1] and st["stopped"]:
                L.append(f"- STOPPED: {st['stopped']}")
            L.append("")
        fin = st.get("final")
        if fin:
            L += ["## Final (test split evaluated once)", "",
                  f"Winner: turn {fin['winner_turn']} `{json.dumps(config_diff(fin['winner_cfg'], self.start), default=str)}`", ""]
            if fin.get("selection"):
                L += [f"| selection candidate | dev {P} | {self.select} {P} |", "|---|---|---|"]
                L += [f"| turn {s['turn']} | {_m3(s['dev'])} | {_m3(s['select'])} |" for s in fin["selection"]]
                L.append("")
            L += [f"| config | test {P} [CI] | n |", "|---|---|---|"]
            for a, r in fin["test"].items():
                m = (r.get("metrics") or {}).get(P)
                L.append(f"| {a} | {_m3(m) if r['status'] == 'ok' else 'crash'} | {r.get('n')} |")
            L += ["", f"| test paired diff | 95% CI | Bonferroni (k={fin['bonferroni_k']}) |", "|---|---|---|"]
            for a, f in fin["test_diff"].items():
                L.append(f"| {a} | {_f3(f['ci95'])} | {_f3(f['bonferroni'])} |")
            if fin.get("ablation"):
                L += ["", f"| ablation ({fin['ablation'][0].get('split', '')}): reverted | reverted - winner | Bonferroni |",
                      "|---|---|---|"]
                for a in fin["ablation"]:
                    L.append(f"| {a['reverted']} -> {a.get('to')} | {_f3(a.get('diff_reverted_minus_winner'))} | "
                             f"{_f3(a.get('diff_bonferroni'))} |" if a.get("status") != "crash" else
                             f"| {a['reverted']} | crash | |")
            v = fin["verdict"]
            L += ["", f"Verdict: beats every reference on test (Bonferroni) = {v['beats_all_on_test']} · beats start = "
                      f"{v['beats_start_on_test']} · invalid+wasted proposal rate {v['invalid_or_wasted_rate']:.2f} "
                      f"(> 0.5 = proposer adds nothing)" + (f" · {v['note']}" if v.get("note") else ""), ""]
        return "\n".join(L) + "\n"

    def _write_journal(self):
        (self.workdir / "JOURNAL.md").write_text(self.journal_markdown(), encoding="utf-8")


def _m3(m):
    if not m or m.get("value") is None:
        return "-"
    return f"{m['value']:.4f} [{m['lo']:.4f}, {m['hi']:.4f}]"


def _f3(f):
    return f"{f['diff']:+.4f} [{f['lo']:+.4f}, {f['hi']:+.4f}]" if f else "-"


def control_arm(loop: Loop, proposer: Any = "random", workdir: str | Path | None = None, **kw) -> Loop:
    """A control loop with the same runner, space, start and baselines that runs exactly as many proposal turns as
    `loop` did, without the stop rule (dongu v1.1 §5: 'was random search enough?')."""
    n = max(len(loop.state["turns"]) - 1, 1)
    return Loop(loop.runner, loop.space, proposer, start=loop.start, primary=loop.primary,
                higher_is_better=loop.hib, unit_col=loop.unit_col, group_col=loop.group_col, stop_after=None,
                max_turns=n, alpha=loop.alpha, baselines=loop.baselines, task=loop.task, workdir=workdir,
                split_key=loop.split_key, dev_split=loop.dev, select_split=loop.select, test_split=loop.test,
                fidelity_key=loop.fidelity_key, simpler_ties=loop.simpler_ties, ablation=False, seed=loop.seed,
                n_boot=loop.n_boot, check_runner=loop.check_runner, **kw)
