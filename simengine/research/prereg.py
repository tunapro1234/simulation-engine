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

4. **Dated addenda and deviations.** A locked prereg is never edited or re-locked. `add_addendum` writes
   `ONKAYIT-EK<k>.md` next to it (reason, deviation, new falsification line), records the parent's SHA-256 and lock
   time, and locks the addendum with its own lock chained to the parent lock (and to the previous addendum's lock);
   `verify_addenda` checks the whole chain. `seal_deviation` seals the pre-deviation arm under its own name before
   the post-deviation arm, in the same chain, and `grade_arms` grades every sealed arm with one scorer, side by side.

        add_addendum("ONKAYIT.md", reason="referee found a stronger base", deviation="ours = H_joint",
                     falsification="claim fails if H_joint - T4b CI contains 0")      # -> ONKAYIT-EK1.md + lock
        seal_deviation(c5_df, h_df, "test forecast", ".prereg", reason="post-referee choice",
                       addendum="ONKAYIT-EK1.md", pre_name="C5", post_name="H_joint")
        grade_arms(".prereg", truth_df, keys=["segment", "item"], column="p", cluster="segment")  # row per arm

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
import re
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
             "_Written and locked before any result was computed. Never edited after the lock: changes go into "
             "dated, separately locked addenda next to this file (`<name>-EK<k>`, research.add_addendum)._", ""]
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


def lock_prereg(path, *, git_commit: bool = False, lock_path=None, overwrite: bool = False, parent=None) -> dict:
    """Lock a pre-registration file: SHA-256 + timestamp into `<path>.lock.json` (optionally git commit both).

    Refuses to re-lock an already locked file with different content unless overwrite=True (a changed prereg needs a
    dated addendum, `add_addendum`, not a replacement: overwrite loses the old lock). `parent`: the locked prereg
    this file is an addendum of; the lock then records the parent's sha256, lock time and lock hash, and the lock
    hash of the parent's latest other addendum (`add_addendum` does this for you)."""
    p = Path(path)
    lp = _lock_path(p, lock_path)
    sha = file_sha256(p)
    if lp.exists() and not overwrite:
        old = json.loads(lp.read_text(encoding="utf-8"))
        if old.get("sha256") == sha:
            return old
        raise ValueError(f"{p} is already locked with a different hash ({old.get('sha256', '')[:12]}); "
                         "restore it and write a dated addendum with add_addendum() (overwrite=True loses the "
                         "old lock)")
    return _write_lock(p, lp, sha, _addendum_link(parent, p) if parent is not None else None, git_commit)


def _write_lock(p: Path, lp: Path, sha: str, link: dict | None, git_commit: bool) -> dict:
    now = datetime.now().astimezone()
    rec = {"file": p.name, "sha256": sha, "size": p.stat().st_size,
           "locked_at": now.isoformat(timespec="seconds"),
           "locked_at_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds")}
    tg = _prereg_targets(p)
    if tg:
        rec["targets_sha256"], rec["n_targets"] = targets_sha256(tg), len(tg)
    if link:
        rec.update(link)
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


# ===================================================================== dated addenda (D6 #8, D8 #7)

def _lock_sha(lock_path) -> str:
    """SHA-256 of a lock record (canonical JSON, so re-indenting the lock file does not change it)."""
    return _sha(_json(json.loads(Path(lock_path).read_text(encoding="utf-8"))).encode("utf-8"))


def _addendum_files(parent) -> list[tuple[int, Path]]:
    """[(k, path)] of `<stem>-EK<k><suffix>` files next to `parent`, by k."""
    p = Path(parent)
    pat = re.compile(rf"^{re.escape(p.stem)}-EK(\d+){re.escape(p.suffix)}$")
    out = [(int(m.group(1)), f) for f in p.parent.glob(f"{p.stem}-EK*{p.suffix}") if (m := pat.match(f.name))]
    return sorted(out)


def _addendum_link(parent, child) -> dict:
    """Lock fields that chain an addendum to its locked parent and to the parent's latest earlier addendum."""
    pp, cp = Path(parent), Path(child)
    v = verify_prereg(pp)
    if not v["ok"]:
        raise ValueError(f"parent {pp.name} does not verify ({v['reason']}); an addendum needs an intact, locked "
                         "parent")
    plr = json.loads(_lock_path(pp).read_text(encoding="utf-8"))
    files = _addendum_files(pp)
    k_child = next((k for k, f in files if f.resolve() == cp.resolve()), None)
    prev = [(k, f) for k, f in files if f.resolve() != cp.resolve() and (k_child is None or k < k_child)]
    link = {"parent": pp.name, "parent_sha256": plr["sha256"], "parent_locked_at": plr.get("locked_at"),
            "parent_lock_sha256": _lock_sha(_lock_path(pp)), "addendum_index": k_child,
            "prev_addendum": None, "prev_addendum_sha256": None, "prev_lock_sha256": _lock_sha(_lock_path(pp))}
    if prev:
        k, f = prev[-1]
        vp = verify_prereg(f)
        if not vp["ok"]:
            raise ValueError(f"previous addendum {f.name} does not verify ({vp['reason']}); lock or restore it first")
        link.update(prev_addendum=f.name, prev_addendum_sha256=vp["sha256"], prev_lock_sha256=_lock_sha(_lock_path(f)))
    if k_child is None:
        link["addendum_index"] = (prev[-1][0] + 1) if prev else 1
    return link


def _md_block(x) -> str:
    if isinstance(x, (list, tuple)):
        return "\n".join(f"- {v}" for v in x)
    if isinstance(x, dict):
        return "\n".join(f"- **{k}**: {v}" for k, v in x.items())
    return str(x)


def add_addendum(parent="ONKAYIT.md", reason: str = "", deviation: str = "", falsification: str = "", path=None, *,
                 changes=None, notes: str = "", git_commit: bool = False) -> dict:
    """Write and lock a dated addendum to a locked pre-registration, without touching the parent or its lock.

    The addendum goes to `path` (default `<parent stem>-EK<k><suffix>` next to the parent, k = next free number,
    e.g. ONKAYIT-EK1.md). Its text records the parent's sha256, lock time and lock hash, the previous addendum (if
    any), the date, `reason`, `deviation` (what changes, said plainly), the new `falsification` line ("" = the
    parent's line stands), optional `changes` (list or text) and `notes`, plus a machine-readable JSON copy. It is
    then locked with `lock_prereg(..., parent=parent)`: its own `<file>.lock.json`, chained to the parent lock hash
    and to the previous addendum's lock hash. Refuses when the parent is not locked or no longer matches its lock,
    when a previous addendum does not verify, or when `path` exists (addenda are never rewritten).

    Returns the lock record plus `path`. `verify_addenda(parent)` checks the whole chain later."""
    if not str(reason).strip() or not str(deviation).strip():
        raise ValueError("an addendum needs a reason and the deviation it makes")
    pp = Path(parent)
    if not pp.exists():
        raise FileNotFoundError(f"parent prereg missing: {pp}")
    if not _lock_path(pp).exists():
        raise ValueError(f"{pp.name} is not locked; lock_prereg it first (an addendum amends a locked prereg)")
    if path is None:
        files = _addendum_files(pp)
        k = (files[-1][0] + 1) if files else 1
        path = pp.with_name(f"{pp.stem}-EK{k}{pp.suffix or '.md'}")
    ap = Path(path)
    if ap.exists():
        raise ValueError(f"{ap} exists; addenda are never rewritten, write the next one")
    link = _addendum_link(pp, ap)
    now = datetime.now().astimezone()
    k = link["addendum_index"]
    doc = {"kind": "prereg_addendum", "index": k, "written_at": now.isoformat(timespec="seconds"),
           "written_at_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
           "reason": str(reason), "deviation": str(deviation), "falsification": str(falsification),
           "changes": changes if changes not in (None, "", []) else None, "notes": str(notes),
           **{f: link[f] for f in ("parent", "parent_sha256", "parent_locked_at", "parent_lock_sha256",
                                   "prev_addendum", "prev_addendum_sha256", "prev_lock_sha256")}}
    L = [f"# Pre-registration addendum EK-{k} ({now.strftime('%Y-%m-%d %H:%M %z')})", "",
         f"_Amends `{pp.name}` after its lock. The parent file and its lock are not edited; this addendum has its "
         "own lock, chained to the parent's._", "",
         "## Parent", f"- file: `{pp.name}`", f"- sha256: `{link['parent_sha256']}`",
         f"- locked at: {link['parent_locked_at']}", f"- lock sha256: `{link['parent_lock_sha256']}`"]
    if link["prev_addendum"]:
        L += [f"- previous addendum: `{link['prev_addendum']}` (sha256 `{link['prev_addendum_sha256']}`)"]
    L += ["", "## Reason", str(reason), "", "## Deviation", str(deviation), "",
          "## Falsification", str(falsification) if str(falsification).strip() else "Unchanged from the parent.", ""]
    if doc["changes"] is not None:
        L += ["## Changes", _md_block(changes), ""]
    if str(notes).strip():
        L += ["## Notes", str(notes), ""]
    L += ["## Machine-readable copy", "```json", json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True,
                                                             default=str), "```", ""]
    ap.parent.mkdir(parents=True, exist_ok=True)
    ap.write_text("\n".join(L), encoding="utf-8")
    rec = _write_lock(ap, _lock_path(ap), file_sha256(ap), link, git_commit)
    return dict(rec, path=str(ap))


def read_addendum(path) -> dict:
    """The machine-readable copy of an addendum written by `add_addendum`."""
    t = Path(path).read_text(encoding="utf-8")
    bas = t.index("```json") + len("```json")
    return json.loads(t[bas:t.index("```", bas)])


def verify_addenda(parent="ONKAYIT.md") -> dict:
    """Check a prereg and every `<stem>-EK<k>` addendum next to it.

    Per addendum: its own lock matches (`ok`), its lock names the parent's current sha256 and lock hash
    (`parent_ok`: fails if the parent lock was overwritten or the parent re-locked), and its `prev_lock_sha256` is
    the lock hash of the addendum before it, or of the parent for the first (`link_ok`). A hand-written addendum with
    a plain lock has no link fields and fails `link_ok` ("not chained"). Returns {ok, parent, addenda: [...]}."""
    pp = Path(parent)
    vp = verify_prereg(pp)
    parent_lock = _lock_sha(_lock_path(pp)) if _lock_path(pp).exists() else None
    prev_lock, rows = parent_lock, []
    for k, f in _addendum_files(pp):
        v = verify_prereg(f)
        row = {"index": k, "file": f.name, "ok": v["ok"], "sha256": v["sha256"], "locked_at": v.get("locked_at"),
               "parent_ok": False, "link_ok": False, "reason": v["reason"]}
        lp = _lock_path(f)
        if lp.exists():
            lr = json.loads(lp.read_text(encoding="utf-8"))
            if "parent_lock_sha256" not in lr:
                row["reason"] = "not chained (plain lock, no parent link)"
            else:
                row["parent_ok"] = (lr.get("parent_sha256") == vp.get("expected") == vp.get("sha256")
                                    and lr.get("parent_lock_sha256") == parent_lock)
                row["link_ok"] = lr.get("prev_lock_sha256") == prev_lock
                if not row["parent_ok"]:
                    row["reason"] = "parent or parent lock changed after this addendum"
                elif not row["link_ok"]:
                    row["reason"] = "chain link to the previous addendum broken"
            prev_lock = _lock_sha(lp)
        try:
            row["addendum"] = read_addendum(f)
        except (ValueError, OSError):
            row["addendum"] = None
        rows.append(row)
    ok = vp["ok"] and all(r["ok"] and r["parent_ok"] and r["link_ok"] for r in rows)
    return {"ok": ok, "parent": vp, "addenda": rows, "n": len(rows)}


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


EVIDENCE_CHAIN_NAME = "zincir.jsonl"     # chain written by simengine.evidence.seal / se.seal (Turkish keys)
_DEFAULT_REGISTRIES = (".prereg", ".onkayit")
_EVIDENCE_TO_EN = {"sira": "seq", "zaman": "time", "zaman_utc": "time_utc", "satir": "rows", "sutun": "cols",
                   "sutunlar": "columns", "aciklama": "description", "kaynak": "source", "arsiv": "archive",
                   "onceki": "prev", "onceki_kayit": "prev_record", "kayit_hash": "record_hash"}


def _seal_records(registry_dir) -> list:
    """Seal records for verify_seal / grade_sealed: this module's chain.jsonl plus an evidence-format zincir.jsonl
    (se.seal / evidence.seal) in the same directory, its records given English key aliases and
    "chain": "zincir.jsonl". When the directory holds neither and is a default name ('.prereg' / '.onkayit'), the
    sibling default directory is read, so a seal made with either API is found. Oldest first within each file."""
    def read(d: Path) -> list:
        out = list(_chain_read(d))
        z = d / EVIDENCE_CHAIN_NAME
        if z.exists():
            with open(z, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        r2 = dict(r)
                        for tr, en in _EVIDENCE_TO_EN.items():
                            if tr in r and en not in r2:
                                r2[en] = r[tr]
                        r2["chain"] = EVIDENCE_CHAIN_NAME
                        out.append(r2)
        return out
    d = Path(registry_dir)
    recs = read(d)
    if not recs and d.name in _DEFAULT_REGISTRIES:
        for other in _DEFAULT_REGISTRIES:
            if other != d.name:
                recs = read(d.parent / other)
    return recs


_RECORD_FIELDS = ("seq", "sha256", "time", "time_utc", "rows", "cols", "columns", "description", "source", "archive",
                  "prev", "prev_record", "record_hash")


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


ARM_ROLES = ("preregistered", "post_deviation")      # `role` values written by seal_deviation
_ARM_FIELDS = ("arm", "role", "deviation_of")


def seal(predictions, description: str, registry_dir=".prereg", *, git_commit: bool = False, arm: str | None = None,
         role: str | None = None, deviation_of: str | None = None, meta: dict | None = None) -> dict:
    """Seal a prediction (DataFrame, table file, dict or any file): SHA-256 of the canonical form + time + hash-chain
    link; a gzip copy of the canonical bytes is archived. Sealing the same content twice returns the oldest record.

    `arm`, `role` ("preregistered" / "post_deviation" / free text), `deviation_of` (the arm this one deviates from)
    and `meta` (extra JSON fields) are stored in the record and covered by its record hash; records without them keep
    the old shape. With any of them set, an earlier record of the same content counts as "the same seal" only if it
    carries the same values (so an arm can be sealed under its name after a plain seal of the same table)."""
    d = Path(registry_dir)
    tags = {f: v for f, v in zip(_ARM_FIELDS, (arm, role, deviation_of)) if v is not None}
    if meta:
        clash = sorted(set(meta) & (set(_RECORD_FIELDS) | set(_ARM_FIELDS)))
        if clash:
            raise ValueError(f"seal meta must not set record fields {clash}")
        tags.update(_struct(dict(meta)))
    d.mkdir(parents=True, exist_ok=True)
    b = canonical_bytes(predictions)
    sha = _sha(b)
    chain = _chain_read(d)
    for k in chain:
        if k["sha256"] == sha and all(k.get(f) == v for f, v in tags.items()):
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
    rec.update(tags)
    rec["record_hash"] = _record_hash(rec["prev_record"], rec)
    with open(d / rec["archive"], "wb") as f:
        f.write(gzip.compress(b, mtime=0))
    with open(d / CHAIN_NAME, "a", encoding="utf-8") as f:
        f.write(_json(rec) + "\n")
    if git_commit:
        _git_commit([d / CHAIN_NAME, d / rec["archive"]], f"prereg: seal {sha[:12]} {description}")
    return rec


def verify_seal(predictions, sha256: str | None = None, registry_dir=".prereg") -> dict:
    """Does the prediction's canonical hash match `sha256`, or any record in the chain? Returns {matched, record, sha256}.
    Reads seals of both APIs (this module's chain.jsonl and se.seal's zincir.jsonl, see `_seal_records`)."""
    sha = _sha(canonical_bytes(predictions))
    hits = [k for k in _seal_records(registry_dir) if k.get("sha256") == sha]
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


# ===================================================================== deviation arms (D6 / D7: both arms gradable)

def _arm_name(obj, name, default) -> str:
    if name:
        return str(name)
    if isinstance(obj, (str, Path)):
        return Path(obj).name.split(".")[0] or default
    return default


def seal_deviation(pre_arm, post_arm, description: str, registry_dir=".prereg", *, reason: str, addendum=None,
                   pre_name: str | None = None, post_name: str | None = None, git_commit: bool = False) -> dict:
    """Seal both arms of a deviation from the pre-registration, pre-deviation arm first, in one chain.

    `pre_arm` is what the locked prereg said to submit; `post_arm` is what is submitted after the deviation (both:
    DataFrame, table file, dict or file, as `seal`). The pre arm gets `arm=pre_name`, `role="preregistered"` and
    `superseded_by=post_name`; the post arm gets `arm=post_name`, `role="post_deviation"`, `deviation_of=pre_name`,
    `deviation_of_record` (the pre record's hash), `deviation_reason` and, when given, the `addendum` file name and
    sha256 (the addendum must exist and verify against its lock: write it with `add_addendum` first). Names default
    to the file stem, else "preregistered" / "post_deviation". Identical arms are refused (no deviation to record).
    If the pre arm's table was already sealed without an arm name (`seal`, `se.seal`) in this registry, the pre
    record points at the oldest such seal (`first_sealed_at`, `first_sealed_seq`, `first_sealed_record`,
    `first_sealed_chain`): that earlier time, not this call's, is when the pre arm is known to have been fixed.
    Calling it again with the same inputs returns the same records. Returns {pre, post, chain}.

    The registry keeps a full gzip copy of both arms; for licensed data put `registry_dir` outside the repo."""
    if not str(reason).strip():
        raise ValueError("seal_deviation needs the reason for the deviation")
    pre_name = _arm_name(pre_arm, pre_name, "preregistered")
    post_name = _arm_name(post_arm, post_name, "post_deviation")
    if pre_name == post_name:
        raise ValueError(f"both arms are named {pre_name!r}; pass pre_name= and post_name=")
    if _sha(canonical_bytes(pre_arm)) == _sha(canonical_bytes(post_arm)):
        raise ValueError("pre- and post-deviation arms are identical; seal() it once instead")
    meta = {"deviation_reason": str(reason)}
    if addendum is not None:
        ap = Path(addendum)
        if not ap.exists():
            raise FileNotFoundError(f"addendum missing: {ap}")
        v = verify_prereg(ap)
        if not v["ok"]:
            raise ValueError(f"addendum {ap.name} does not verify ({v['reason']}); lock it (add_addendum does)")
        meta.update(addendum=ap.name, addendum_sha256=v["sha256"])
    pre_meta = {"superseded_by": post_name}
    pre_sha = _sha(canonical_bytes(pre_arm))
    plain = [r for r in _seal_records(registry_dir) if r.get("sha256") == pre_sha and not r.get("arm")]
    first = min(plain, key=lambda r: str(r.get("time_utc") or r.get("time") or "")) if plain else None
    if first is not None:                      # plain seals only, so a rerun finds the same one (idempotent)
        pre_meta.update(first_sealed_at=first.get("time"), first_sealed_seq=first.get("seq"),
                        first_sealed_record=first.get("record_hash"),
                        first_sealed_chain=first.get("chain", CHAIN_NAME))
    pre = seal(pre_arm, description, registry_dir, git_commit=git_commit, arm=pre_name, role="preregistered",
               meta=pre_meta)
    post = seal(post_arm, description, registry_dir, git_commit=git_commit, arm=post_name, role="post_deviation",
                deviation_of=pre_name, meta=dict(meta, deviation_of_record=pre["record_hash"]))
    return {"pre": pre, "post": post, "chain": verify_chain(registry_dir)}


def _keyed(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Key columns in canonical text form, so an archived arm (read back from CSV) joins a truth table whose key
    dtypes differ (int vs str vs float)."""
    df = df.copy()
    for k in keys:
        df[k] = df[k].map(_cell)
    return df


def _arm_table(rec: dict, d: Path, predictions, keys: list[str]) -> pd.DataFrame:
    arm = rec["arm"]
    if predictions is not None and arm in predictions:
        df = _read(predictions[arm])
        if _sha(canonical_bytes(df)) != rec["sha256"]:
            raise ValueError(f"predictions for arm {arm!r} do not match its seal ({rec['sha256'][:12]})")
        return df
    if rec.get("rows") is None:
        raise ValueError(f"arm {arm!r} was not sealed as a table; pass it in predictions=")
    f = d / rec["archive"]
    if not f.exists():
        raise FileNotFoundError(f"archive of arm {arm!r} missing: {f.name}; pass it in predictions=")
    b = gzip.decompress(f.read_bytes())
    if _sha(b) != rec["sha256"]:
        raise ValueError(f"archive of arm {arm!r} does not match its seal")
    return pd.read_csv(io.BytesIO(b), dtype={k: str for k in keys})    # keys as sealed: "01" stays "01"


def grade_arms(registry_dir, truth, keys: list[str], *, column: str | None = None, scorer=None, unit_loss=None,
               predictions: dict | None = None, arms=None, how: str = "inner", cluster=None,
               seed: int = 42) -> pd.DataFrame:
    """Grade every sealed arm (records with an `arm` field, see `seal_deviation` / `seal(arm=)`) with the same
    scorer and return them side by side, one row per arm record in chain order.

    Arm tables come from the chain's gzip archives (verified against the seal) or from `predictions={arm: data}`
    (verified too). Each is joined with `truth` on `keys` (truth columns get the suffix `_truth`).
    Scoring: `scorer(table) -> dict | float` if given; otherwise per-unit `unit_loss(pred, truth)` (default absolute
    error) on `column` (default: the only non-key column the arm and truth share) gives `mae`, `rmse`, `bias`
    (or `loss` for a custom unit_loss). With per-unit losses, an arm with `deviation_of` also gets
    `diff_vs_pre` = mean loss(arm) - mean loss(pre arm) on shared units with a paired bootstrap 95% CI
    (`diff_lo`, `diff_hi`, `diff_n`; negative = the deviation helped). Units with a missing loss are left out and
    counted in `n_missing`; arm rows whose key has no truth row are counted in `n_unmatched` (with a warning: a key
    format mismatch such as "01" vs 1 would otherwise silently shrink the graded set). When the arms end up scored
    on different units, a warning says their mean columns are not comparable; `diff_vs_pre` uses shared units only.

    `cluster`: key column(s) whose rows are resampled together for the diff CI (`diff_cluster`, `diff_n_clusters`),
    e.g. "segment" when every segment has several items. Rows of one segment share its errors; the row bootstrap
    then covers ~0.80 (7 items per segment, segment-varying advantage) instead of 0.95. With several keys and no
    `cluster` the CI treats rows as independent and a warning says so; `cluster=False` means rows are independent."""
    d = Path(registry_dir)
    recs = [r for r in _chain_read(d) if r.get("arm")]
    if arms is not None:
        want = [arms] if isinstance(arms, str) else list(arms)
        missing = sorted(set(want) - {r["arm"] for r in recs})
        if missing:
            raise KeyError(f"no sealed arm(s) {missing} in {d}")
        recs = [r for r in recs if r["arm"] in want]
    if not recs:
        raise ValueError(f"no arm records in {d / CHAIN_NAME}; seal with seal_deviation() or seal(..., arm=)")
    g = _read(truth)
    if [k for k in keys if k not in g.columns]:
        raise KeyError(f"key column(s) missing from truth: {[k for k in keys if k not in g.columns]}")
    gk = _keyed(g, keys)
    if cluster is None or cluster is False:
        cl = []
    else:
        cl = [cluster] if isinstance(cluster, str) else list(cluster)
        if not cl or [c for c in cl if c not in keys]:
            raise KeyError(f"cluster= must name key column(s) out of {keys}, got {cluster!r}")
    gidx = pd.MultiIndex.from_frame(gk[keys])
    from .stats import paired_bootstrap_diff, paired_cluster_bootstrap_diff
    rows, losses = [], {}
    for r in recs:
        t = _arm_table(r, d, predictions, keys)
        miss = [k for k in keys if k not in t.columns]
        if miss:
            raise KeyError(f"key column(s) missing from arm {r['arm']!r}: {miss}")
        tk = _keyed(t, keys)
        table = tk.merge(gk, on=keys, how=how, suffixes=("", "_truth"))
        n_unmatched = int((~pd.MultiIndex.from_frame(tk[keys]).isin(gidx)).sum())
        if n_unmatched:
            warnings.warn(f"grade_arms: {n_unmatched} of {len(t)} rows of arm {r['arm']!r} have no truth row on "
                          f"{keys} (key format, e.g. '01' vs 1?)", UserWarning, stacklevel=2)
        row = {"seq": r.get("seq"), "arm": r["arm"], "role": r.get("role"), "deviation_of": r.get("deviation_of"),
               "sealed_at": r.get("time"), "first_sealed_at": r.get("first_sealed_at"),
               "sha256": r["sha256"][:12], "n_pred": len(t), "n_joined": len(table), "n_unmatched": n_unmatched}
        loss = None
        if scorer is None or unit_loss is not None:
            col = column
            if col is None:
                shared = [c for c in t.columns if c in g.columns and c not in keys]
                if len(shared) != 1:
                    raise ValueError(f"pass column=: arm {r['arm']!r} and truth share {shared or 'no'} non-key "
                                     "columns")
                col = shared[0]
            if col not in t.columns or col not in g.columns:
                raise KeyError(f"column {col!r} missing from arm {r['arm']!r} or truth")
            pred = pd.to_numeric(table[col], errors="coerce")
            tru = pd.to_numeric(table[f"{col}_truth"], errors="coerce")
            raw = (pred - tru).abs() if unit_loss is None else pd.Series(unit_loss(pred, tru), index=table.index)
            loss = pd.Series(pd.to_numeric(raw, errors="coerce").to_numpy(),
                             index=pd.MultiIndex.from_frame(table[keys])).dropna()
            row["n_missing"] = len(table) - len(loss)
            if scorer is None:
                if unit_loss is None:
                    e = (pred - tru).dropna()
                    row.update(mae=float(loss.mean()) if len(loss) else math.nan,
                               rmse=float(math.sqrt((e ** 2).mean())) if len(e) else math.nan,
                               bias=float(e.mean()) if len(e) else math.nan)
                else:
                    row["loss"] = float(loss.mean()) if len(loss) else math.nan
        if scorer is not None:
            m = scorer(table)
            row.update(m if isinstance(m, dict) else {"score": m})
        losses[r["record_hash"]] = loss
        rows.append(row)
    units = {r["arm"]: frozenset(losses[r["record_hash"]].index) for r in recs
             if losses[r["record_hash"]] is not None}
    if len(set(units.values())) > 1:          # side-by-side means over different units are not comparable
        warnings.warn("grade_arms: arms are scored on different units (" +
                      ", ".join(f"{a}: {len(u)}" for a, u in units.items()) + "); their mae/rmse/loss columns "
                      "are not comparable, read diff_vs_pre (shared units only)", UserWarning, stacklevel=2)
    by_hash = {r["record_hash"]: r for r in recs}
    for r, row in zip(recs, rows):
        ref = r.get("deviation_of_record")
        if ref not in by_hash:
            ref = next((q["record_hash"] for q in recs if q["arm"] == r.get("deviation_of")), None)
        a, b = losses.get(r["record_hash"]), losses.get(ref) if ref else None
        if a is None or b is None or not (a.index.is_unique and b.index.is_unique):   # duplicate keys: no pairing
            continue
        both = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
        if not len(both):
            continue
        if cl:
            grp = pd.MultiIndex.from_frame(both.index.to_frame(index=False)[cl]).factorize()[0] if len(cl) > 1 \
                else both.index.get_level_values(cl[0]).factorize()[0]
            diff = paired_cluster_bootstrap_diff(both["a"].to_numpy(), both["b"].to_numpy(), grp, seed=seed)
        else:
            if cluster is None and len(keys) > 1:
                warnings.warn(f"grade_arms: the diff CI of arm {r['arm']!r} resamples rows as independent; with "
                              f"keys {keys} pass cluster=<key> (e.g. {keys[0]!r}) if rows share a segment or item, "
                              "or cluster=False", UserWarning, stacklevel=2)
            diff = paired_bootstrap_diff(both["a"].to_numpy(), both["b"].to_numpy(), seed=seed)
        row.update(diff_vs_pre=diff["diff"], diff_lo=diff["lo"], diff_hi=diff["hi"], diff_n=diff["n"],
                   diff_cluster="+".join(cl) if cl else "row", diff_n_clusters=diff.get("n_clusters", diff["n"]))
    out = pd.DataFrame(rows)
    out.attrs["graded_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return out

