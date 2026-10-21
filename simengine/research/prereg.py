"""Pre-registration and sealed predictions ("predict first, grade later").

Two tools:

1. **Pre-registration lock.** Write the question, data/split, metrics, baselines, expected result and the
   falsification line *before* running anything (`Prereg.write`), then `lock_prereg` stores the file's SHA-256 and a
   timestamp next to it (and optionally commits both to git). `verify_prereg` later proves the file was not edited.

2. **Sealed predictions** (ported from uretim/kanit/kod/kanit/onkayit.py: muhurle / dogrula / zincir_dogrula / notla).
   A prediction table is hashed in a canonical form (independent of row and column order) and appended to a
   hash chain *before* the ground truth is opened. `grade_sealed` refuses to join predictions with the truth unless the
   hash matches a sealed record.

3. **Declared targets.** `declare_targets` normalises the paper findings a replication will be compared against
   (replicant's `PAPER_FINDINGS`: `{"metric": (value, "description")}`) and writes them into the pre-registration
   before it is locked, so the lock hash covers them. `compare_targets` later sets observed metrics against them;
   a metric that was not declared is reported as "undeclared", a missing one stays missing.

A git commit time and a local clock are the author's own claim, not third-party evidence. For stronger proof, publish
the SHA-256 somewhere external (an email, an issue comment) at lock time.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CHAIN_NAME = "chain.jsonl"


# ===================================================================== pre-registration document

@dataclass
class Prereg:
    """A pre-registration. All fields are free text (or lists/dicts of text) written before any result is seen."""
    title: str
    question: str
    data: str = ""
    split: str = ""
    metrics: Any = ""                 # primary first; say the direction (lower/higher is better)
    baselines: Any = ""               # the baselines every result is compared against
    expected: str = ""                # the expected result, with numbers if possible
    falsification: str = ""           # "the claim fails if ..."
    stop_rule: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)   # declare_targets(...) output; hashed with the file at lock time

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d["targets"]:          # keeps the bytes of preregs without targets identical to earlier versions
            d.pop("targets")
        return d

    def to_markdown(self) -> str:
        def blok(x):
            if isinstance(x, (list, tuple)):
                return "\n".join(f"- {v}" for v in x)
            if isinstance(x, dict):
                return "\n".join(f"- **{k}**: {v}" for k, v in x.items())
            return str(x)
        L = [f"# Pre-registration: {self.title}", "",
             "_Written and locked before any result was computed. Changes only as dated addenda below._", ""]
        for baslik, deger in (("Question", self.question), ("Data", self.data), ("Split", self.split),
                              ("Metrics", self.metrics), ("Baselines", self.baselines),
                              ("Expected result", self.expected), ("Falsification", self.falsification),
                              ("Stop rule", self.stop_rule), ("Notes", self.notes)):
            if deger not in ("", None, [], {}):
                L += [f"## {baslik}", blok(deger), ""]
        if self.extra:
            L += ["## Extra", blok(self.extra), ""]
        if self.targets:
            L += ["## Declared targets", "", "| metric | target | description | source |", "|---|---|---|---|"]
            for m, t in self.targets.items():
                cell = [str(x).replace("|", "\\|").replace("\n", " ")
                        for x in (m, _target_text(t), t.get("description", ""), t.get("source", ""))]
                L.append("| " + " | ".join(cell) + " |")
            L += ["", f"Targets sha256: `{targets_sha256(self.targets)}`", ""]
        L += ["## Machine-readable copy", "```json", json.dumps(self.to_dict(), indent=1, ensure_ascii=False,
                                                                 sort_keys=True), "```", ""]
        return "\n".join(L)

    def write(self, path) -> Path:
        """Write as JSON if the path ends with .json, otherwise as markdown (with an embedded JSON block)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            p.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                         encoding="utf-8")
        else:
            p.write_text(self.to_markdown(), encoding="utf-8")
        return p

    @classmethod
    def read(cls, path) -> "Prereg":
        """Read a prereg written by `write` (JSON file or the markdown's embedded JSON block)."""
        t = Path(path).read_text(encoding="utf-8")
        if Path(path).suffix.lower() == ".json":
            d = json.loads(t)
        else:
            bas = t.index("```json") + len("```json")
            d = json.loads(t[bas:t.index("```", bas)])
        return cls(**d)


def write_prereg(prereg: Prereg | dict, path) -> Path:
    if isinstance(prereg, dict):
        prereg = Prereg(**prereg)
    return prereg.write(path)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _lock_path(path, lock_path=None) -> Path:
    return Path(lock_path) if lock_path else Path(str(path) + ".lock.json")


def _git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)


def _git_commit(paths: list[Path], message: str) -> str | None:
    """Commit the given files in their repo; return the commit hash or None (warns on failure)."""
    cwd = paths[0].resolve().parent
    try:
        if _git(cwd, "rev-parse", "--is-inside-work-tree").returncode != 0:
            raise OSError("not inside a git work tree")
        names = [str(p.resolve()) for p in paths]
        r = _git(cwd, "add", "--", *names)
        if r.returncode:
            raise OSError(r.stderr)
        r = _git(cwd, "commit", "-q", "-m", message, "--", *names)
        if r.returncode and "nothing" not in (r.stdout + r.stderr):
            raise OSError(r.stderr)
        return _git(cwd, "rev-parse", "HEAD").stdout.strip() or None
    except OSError as e:
        warnings.warn(f"research.prereg: git commit skipped ({str(e).strip()[:200]})")
        return None


def lock_prereg(path, *, git_commit: bool = False, lock_path=None, overwrite: bool = False) -> dict:
    """Lock a pre-registration file: SHA-256 + timestamp into `<path>.lock.json` (optionally git commit both).

    Refuses to re-lock an already locked file with different content unless overwrite=True (a changed prereg needs a
    dated addendum and a new lock, not a silent replacement)."""
    p = Path(path)
    lp = _lock_path(p, lock_path)
    sha = file_sha256(p)
    if lp.exists() and not overwrite:
        old = json.loads(lp.read_text(encoding="utf-8"))
        if old.get("sha256") == sha:
            return old
        raise ValueError(f"{p} is already locked with a different hash ({old.get('sha256', '')[:12]}); "
                         "write an addendum and lock with overwrite=True")
    now = datetime.now().astimezone()
    rec = {"file": p.name, "sha256": sha, "size": p.stat().st_size,
           "locked_at": now.isoformat(timespec="seconds"),
           "locked_at_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds")}
    tg = _prereg_targets(p)
    if tg:
        rec["targets_sha256"], rec["n_targets"] = targets_sha256(tg), len(tg)
    lp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if git_commit:
        rec = dict(rec, git_commit=_git_commit([p, lp], f"prereg: lock {p.name} {sha[:12]}"))
    return rec


def verify_prereg(path, *, lock_path=None, check_git: bool = False) -> dict:
    """Check that a locked pre-registration is unchanged.

    Returns {ok, sha256, expected, locked_at, reason[, git]}. With check_git, also reports whether the current content
    equals the committed version and the time of the first commit that added the file."""
    p = Path(path)
    lp = _lock_path(p, lock_path)
    if not p.exists():
        return {"ok": False, "reason": "prereg file missing", "sha256": None, "expected": None}
    sha = file_sha256(p)
    if not lp.exists():
        return {"ok": False, "reason": "not locked (no lock file)", "sha256": sha, "expected": None}
    rec = json.loads(lp.read_text(encoding="utf-8"))
    ok = rec.get("sha256") == sha
    out = {"ok": ok, "sha256": sha, "expected": rec.get("sha256"), "locked_at": rec.get("locked_at"),
           "reason": "match" if ok else "content changed after lock"}
    if rec.get("targets_sha256"):
        out["targets_sha256"] = rec["targets_sha256"]
    if check_git:
        cwd = p.resolve().parent
        name = str(p.resolve())
        ilk = _git(cwd, "log", "--diff-filter=A", "--format=%H %cI", "--", name).stdout.strip().splitlines()
        temiz = _git(cwd, "diff", "--quiet", "HEAD", "--", name).returncode == 0
        out["git"] = {"committed": bool(ilk), "first_commit": ilk[-1].split()[0] if ilk else None,
                      "first_commit_time": ilk[-1].split()[1] if ilk else None, "clean": bool(ilk) and temiz}
        out["ok"] = out["ok"] and out["git"]["clean"]
        if ok and not out["git"]["clean"]:
            out["reason"] = "lock matches but git copy is missing or differs"
    return out


# ===================================================================== declared targets (replicant PAPER_FINDINGS)

TARGET_FIELDS = ("value", "description", "tolerance", "interval", "source", "unit", "notes")


def _target_value(metric, v):
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        v = v.item()
    if isinstance(v, (bool, str)) or v is None:
        return v
    if isinstance(v, (int, float)):
        if not math.isfinite(v):
            raise ValueError(f"target {metric!r}: value must be finite, got {v!r}")
        return v
    raise ValueError(f"target {metric!r}: value must be a number, a string or a bool, got {type(v).__name__}")


def _one_target(metric, spec) -> dict:
    if isinstance(spec, dict):
        t = dict(spec)
    elif isinstance(spec, (tuple, list)):
        if not 1 <= len(spec) <= 3 or (len(spec) == 3 and not isinstance(spec[2], dict)):
            raise ValueError(f"target {metric!r}: use (value, description) or (value, description, {{fields}})")
        t = {"value": spec[0]}
        if len(spec) > 1:
            t["description"] = spec[1]
        if len(spec) == 3:
            t.update(spec[2])
    else:
        t = {"value": spec}
    unknown = sorted(set(t) - set(TARGET_FIELDS))
    if unknown:
        raise ValueError(f"target {metric!r}: unknown field(s) {unknown}; allowed {list(TARGET_FIELDS)}")
    out = {}
    if t.get("value") is not None:
        out["value"] = _target_value(metric, t["value"])
    if t.get("interval") is not None:
        iv = t["interval"]
        if not isinstance(iv, (tuple, list)) or len(iv) != 2:
            raise ValueError(f"target {metric!r}: interval must be [lo, hi]")
        lo, hi = (_target_value(metric, x) for x in iv)
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (lo, hi)) or lo > hi:
            raise ValueError(f"target {metric!r}: interval must be two numbers with lo <= hi, got {iv!r}")
        out["interval"] = [lo, hi]
    if t.get("tolerance") is not None:
        tol = _target_value(metric, t["tolerance"])
        if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol < 0:
            raise ValueError(f"target {metric!r}: tolerance must be a number >= 0")
        if not isinstance(out.get("value"), (int, float)) or isinstance(out.get("value"), bool):
            raise ValueError(f"target {metric!r}: tolerance needs a numeric value")
        out["tolerance"] = tol
    if "value" not in out and "interval" not in out:
        raise ValueError(f"target {metric!r}: give a value or an interval")
    for k in ("description", "source", "unit", "notes"):
        if t.get(k) not in (None, ""):
            out[k] = str(t[k])
    return out


def declare_targets(findings, prereg=None, *, overwrite: bool = False) -> dict:
    """Normalise the findings a run will be compared against and (optionally) write them into a pre-registration.

    `findings` (port of replicant `papers/TEMPLATE/config.py` PAPER_FINDINGS, declared before the run):
      - `{"metric": (value, "description")}` as in PAPER_FINDINGS, or `(value, "description", {fields})`,
      - `{"metric": value}`,
      - `{"metric": {"value": .., "description": .., "tolerance": .., "interval": [lo, hi], "source": .., ...}}`,
      - or a list of such dicts with a `"metric"` key.
    Values must be finite (no NaN targets). Returns `{metric: {...}}` sorted by metric.

    `prereg`: a `Prereg` (its `.targets` is set in place) or the path of a prereg file written by `Prereg.write` (the
    file is rewritten with the targets). A path that is already locked is refused unless `overwrite=True`, and then
    the old lock no longer verifies until `lock_prereg(..., overwrite=True)`: targets belong *before* the lock."""
    if isinstance(findings, Prereg):
        findings = findings.targets
    if isinstance(findings, (list, tuple)):
        d = {}
        for row in findings:
            if not isinstance(row, dict) or not (row.get("metric") or row.get("name")):
                raise ValueError("a list of targets needs dicts with a 'metric' key")
            row = dict(row)
            m = str(row.pop("metric", None) or row.pop("name"))
            row.pop("name", None)
            if m in d:
                raise ValueError(f"target {m!r} declared twice")
            d[m] = row
        findings = d
    if not isinstance(findings, dict):
        raise TypeError("findings must be a dict {metric: spec} or a list of {'metric': ..} dicts")
    out = {}
    for m, spec in findings.items():
        if not isinstance(m, str) or not m.strip():
            raise ValueError(f"target metric names must be non-empty strings, got {m!r}")
        out[m.strip()] = _one_target(m.strip(), spec)
    if len(out) != len(findings):
        raise ValueError("target metric names collide after stripping whitespace")
    out = dict(sorted(out.items()))
    if isinstance(prereg, Prereg):
        prereg.targets = out
    elif prereg is not None:
        p = Path(prereg)
        if _lock_path(p).exists() and not overwrite:
            raise ValueError(f"{p} is already locked; targets must be declared before the lock "
                             "(write an addendum and pass overwrite=True, then re-lock)")
        doc = Prereg.read(p)
        doc.targets = out
        doc.write(p)
    return out


def targets_sha256(targets) -> str:
    """SHA-256 of the canonical form of the declared targets (key order and float noise do not matter)."""
    return _sha(canonical_bytes({"targets": declare_targets(targets)}))


def _target_text(t: dict) -> str:
    s = "" if "value" not in t else str(t["value"])
    if "tolerance" in t:
        s += f" ± {t['tolerance']}"
    if "interval" in t:
        s += (" " if s else "") + f"[{t['interval'][0]}, {t['interval'][1]}]"
    return s + (f" {t['unit']}" if t.get("unit") else "")


def _prereg_targets(path) -> dict:
    """Targets of a prereg file written by `Prereg.write`, or {} for any other file."""
    try:
        return Prereg.read(path).targets
    except (ValueError, TypeError, KeyError, OSError):
        return {}


def compare_targets(targets, observed: dict) -> pd.DataFrame:
    """Set observed metrics against declared targets. One row per declared metric, plus one per observed metric
    that was not declared (status "undeclared": a post-hoc metric, report it as such).

    status: "pass"/"fail" (inside/outside `interval`, or |observed - value| <= `tolerance`; for a non-numeric value,
    equality), "reported" (a numeric target with neither tolerance nor interval: the difference is shown, no verdict),
    "missing" (no observed value, or None/NaN: never filled in), "undeclared".

    `targets`: the `declare_targets` output, raw findings, a `Prereg` or a prereg file path."""
    if isinstance(targets, (str, Path)):
        targets = Prereg.read(targets).targets
    tg = declare_targets(targets)
    rows = []
    for m, t in tg.items():
        v = t.get("value")
        o = observed.get(m)
        if hasattr(o, "item") and not isinstance(o, (str, bytes)):
            o = o.item()
        num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
        row = {"metric": m, "target": v, "lo": t.get("interval", [None, None])[0],
               "hi": t.get("interval", [None, None])[1], "tolerance": t.get("tolerance"), "observed": o,
               "diff": None, "status": None, "description": t.get("description", "")}
        if o is None or (isinstance(o, float) and math.isnan(o)):
            row["observed"], row["status"] = None, "missing"
        else:
            if num(o) and num(v):
                row["diff"] = float(o) - float(v)
            if "interval" in t:
                row["status"] = "pass" if num(o) and t["interval"][0] <= o <= t["interval"][1] else "fail"
            elif "tolerance" in t:
                row["status"] = "pass" if num(o) and abs(float(o) - float(v)) <= t["tolerance"] else "fail"
            elif num(v):
                row["status"] = "reported"
            else:
                row["status"] = "pass" if o == v else "fail"
        rows.append(row)
    for m in sorted(set(observed) - set(tg), key=str):
        rows.append({"metric": m, "target": None, "lo": None, "hi": None, "tolerance": None, "observed": observed[m],
                     "diff": None, "status": "undeclared", "description": ""})
    cols = ["metric", "target", "lo", "hi", "tolerance", "observed", "diff", "status", "description"]
    return pd.DataFrame(rows, columns=cols)


# ===================================================================== sealed predictions (port of kanit.onkayit)

def _read(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    p = Path(obj)
    ad = p.name.lower()
    if ad.endswith((".parquet", ".pq")):
        return pd.read_parquet(p)
    if ad.endswith((".jsonl", ".jsonl.gz")):
        return pd.read_json(p, lines=True)
    if ad.endswith((".json", ".json.gz")):
        return pd.read_json(p)
    if ad.endswith((".csv", ".csv.gz", ".tsv")):
        return pd.read_csv(p, sep="\t" if ad.endswith(".tsv") else ",")
    raise ValueError(f"unsupported file type: {p}")


def _num(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    s = format(x, ".10g")
    return "0" if s == "-0" else s


def _struct(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, dict):
        return {str(k): _struct(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_struct(v) for v in x]
    if isinstance(x, float):
        return float(_num(x)) if math.isfinite(x) else _num(x)
    return x


def _cell(x) -> str:
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)) and getattr(x, "ndim", 0) > 0:
        return json.dumps(_struct(x), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if isinstance(x, (list, tuple, dict)):
        return json.dumps(_struct(x), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if x is None or x is pd.NA or x is pd.NaT:
        return "nan"
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, bool):
        return "True" if x else "False"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return _num(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    return str(x)


def canonical_bytes(obj) -> bytes:
    """Same content -> same bytes, independent of column and row order. Accepts a DataFrame, a table file
    (csv/tsv/parquet/json/jsonl), a dict (canonical JSON) or any other file (raw bytes)."""
    if isinstance(obj, dict):
        return json.dumps(_struct(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if not isinstance(obj, pd.DataFrame):
        p = Path(obj)
        try:
            df = _read(p)
        except ValueError:
            return p.read_bytes()
    else:
        df = obj
    cols = sorted(df.columns, key=str)
    rows = [[_cell(v) for v in r] for r in df[cols].itertuples(index=False, name=None)]
    rows.sort()
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([str(c) for c in cols])
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _chain_read(registry_dir) -> list:
    p = Path(registry_dir) / CHAIN_NAME
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def _record_hash(prev_record_hash, rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "record_hash"}
    return _sha(((prev_record_hash or "") + _json(body)).encode("utf-8"))


def _shape(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.shape[0], obj.shape[1], sorted(str(c) for c in obj.columns)
    if isinstance(obj, dict):
        return None, None, sorted(str(k) for k in obj)
    try:
        df = _read(obj)
        return df.shape[0], df.shape[1], sorted(str(c) for c in df.columns)
    except ValueError:
        return None, None, None


def seal(predictions, description: str, registry_dir=".prereg", *, git_commit: bool = False) -> dict:
    """Seal a prediction (DataFrame, table file, dict or any file): SHA-256 of the canonical form + time + hash-chain
    link; a gzip copy of the canonical bytes is archived. Sealing the same content twice returns the oldest record."""
    d = Path(registry_dir)
    d.mkdir(parents=True, exist_ok=True)
    b = canonical_bytes(predictions)
    sha = _sha(b)
    chain = _chain_read(d)
    for k in chain:
        if k["sha256"] == sha:
            return k
    last = chain[-1] if chain else None
    now = datetime.now().astimezone()
    rows, ncols, cols = _shape(predictions)
    rec = {"seq": len(chain), "sha256": sha, "time": now.isoformat(timespec="seconds"),
           "time_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
           "rows": rows, "cols": ncols, "columns": cols, "description": description,
           "source": None if isinstance(predictions, (pd.DataFrame, dict)) else Path(predictions).name,
           "archive": f"{sha[:12]}.gz", "prev": last["sha256"] if last else None,
           "prev_record": last["record_hash"] if last else None}
    rec["record_hash"] = _record_hash(rec["prev_record"], rec)
    with open(d / rec["archive"], "wb") as f:
        f.write(gzip.compress(b, mtime=0))
    with open(d / CHAIN_NAME, "a", encoding="utf-8") as f:
        f.write(_json(rec) + "\n")
    if git_commit:
        _git_commit([d / CHAIN_NAME, d / rec["archive"]], f"prereg: seal {sha[:12]} {description}")
    return rec


def verify_seal(predictions, sha256: str | None = None, registry_dir=".prereg") -> dict:
    """Does the prediction's canonical hash match `sha256`, or any record in the chain? Returns {matched, record, sha256}."""
    sha = _sha(canonical_bytes(predictions))
    hits = [k for k in _chain_read(registry_dir) if k["sha256"] == sha]
    rec = hits[0] if hits else None
    matched = (sha == sha256.lower()) if sha256 else rec is not None
    return {"matched": matched, "record": rec, "sha256": sha}


def verify_chain(registry_dir=".prereg") -> dict:
    """Recompute every link and record hash of the chain. Returns {valid, n, broken}."""
    chain = _chain_read(registry_dir)
    broken, prev = [], None
    for i, k in enumerate(chain):
        exp_prev = prev["sha256"] if prev else None
        exp_link = prev["record_hash"] if prev else None
        if (k.get("prev") != exp_prev or k.get("prev_record") != exp_link or k.get("seq") != i
                or _record_hash(exp_link, k) != k.get("record_hash")):
            broken.append(i)
        prev = k
    return {"valid": not broken, "n": len(chain), "broken": broken}


def grade_sealed(predictions, truth, keys: list[str], registry_dir=".prereg", *, sha256: str | None = None,
                 how: str = "inner") -> dict:
    """Join predictions with the truth on `keys`, but only if the predictions match a sealed record."""
    v = verify_seal(predictions, sha256, registry_dir)
    if not v["matched"]:
        raise ValueError(f"predictions do not match a sealed record (sha256={v['sha256'][:12]})")
    t, g = _read(predictions), _read(truth)
    missing = [a for a in keys if a not in t.columns or a not in g.columns]
    if missing:
        raise KeyError(f"key column(s) missing: {missing}")
    table = t.merge(g, on=keys, how=how, suffixes=("", "_truth"))
    return {"verification": v, "table": table, "n_pred": len(t), "n_truth": len(g), "n_joined": len(table),
            "graded_at": datetime.now().astimezone().isoformat(timespec="seconds")}
