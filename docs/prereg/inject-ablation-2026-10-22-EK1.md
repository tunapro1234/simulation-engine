# Pre-registration addendum EK-1 (2026-10-23 03:16 +0300)

_Amends `inject-ablation-2026-10-22.md` after its lock. The parent file and its lock are not edited; this addendum has its own lock, chained to the parent's._

## Parent
- file: `inject-ablation-2026-10-22.md`
- sha256: `c4d436a9bb437323645813cc5a9a81a9f7af5a4a2eb0c412708e53687bccee88`
- locked at: 2026-10-23T01:54:28+03:00
- lock sha256: `501f77af523ba5f22eb88d2a271ecaad577de10ac4163cfea43d970ea2d0b6ca`

## Reason
Before any target call, the runner's plan check refused to start: the log_retrieved(k=10) prompts no longer hashed to the value listed above (c7a640d719a7). Cause: persona.inject.from_logs.bm25_rank summed term scores over a Python set, whose order follows PYTHONHASHSEED, so near-equal BM25 scores ranked differently in every process (4 hash seeds gave 4 rankings on a synthetic panel). The listed hash was one random draw and cannot be reproduced. Fixed in simengine b7b25e9 (sorted term order, regression test across hash seeds). Only a 3-person narrative smoke call had been made (0.0008 USD); no target answer was seen.

## Deviation
log_retrieved(k=10) in part A uses the deterministic ranking of b7b25e9; its prompt sha256 becomes 2d8aac374406 (full dry-run plan regenerated 23 Oct 03:1x, plan/plan.json; the 01:53 plan is kept as plan-onkayit-0153.json). Every other arm's prompt hash, both parts' cell hashes, the arm lists and the cost bound (7.0389 USD) are unchanged. Runner: ablation.py with run/smoke/score modes added after the lock and reviewed (swap-arm cache key per arm; failed calls retried on resume), sha256 8d908e69f69d48e60181899a48e352df4c03cecf74eb27089a89e01799934a43. Data, cells, arms, metrics, comparisons, Holm families, bootstrap and decision rules are unchanged.

## Falsification
Unchanged from the parent.

## Changes
- log_retrieved(k=10) prompt sha256 c7a640d719a7 -> 2d8aac374406 (deterministic BM25 order, b7b25e9)
- runner sha256 c48fe31c00a9 -> 8d908e69f69d (run/smoke/score modes, reviewed)

## Machine-readable copy
```json
{
 "changes": [
  "log_retrieved(k=10) prompt sha256 c7a640d719a7 -> 2d8aac374406 (deterministic BM25 order, b7b25e9)",
  "runner sha256 c48fe31c00a9 -> 8d908e69f69d (run/smoke/score modes, reviewed)"
 ],
 "deviation": "log_retrieved(k=10) in part A uses the deterministic ranking of b7b25e9; its prompt sha256 becomes 2d8aac374406 (full dry-run plan regenerated 23 Oct 03:1x, plan/plan.json; the 01:53 plan is kept as plan-onkayit-0153.json). Every other arm's prompt hash, both parts' cell hashes, the arm lists and the cost bound (7.0389 USD) are unchanged. Runner: ablation.py with run/smoke/score modes added after the lock and reviewed (swap-arm cache key per arm; failed calls retried on resume), sha256 8d908e69f69d48e60181899a48e352df4c03cecf74eb27089a89e01799934a43. Data, cells, arms, metrics, comparisons, Holm families, bootstrap and decision rules are unchanged.",
 "falsification": "",
 "index": 1,
 "kind": "prereg_addendum",
 "notes": "",
 "parent": "inject-ablation-2026-10-22.md",
 "parent_lock_sha256": "501f77af523ba5f22eb88d2a271ecaad577de10ac4163cfea43d970ea2d0b6ca",
 "parent_locked_at": "2026-10-23T01:54:28+03:00",
 "parent_sha256": "c4d436a9bb437323645813cc5a9a81a9f7af5a4a2eb0c412708e53687bccee88",
 "prev_addendum": null,
 "prev_addendum_sha256": null,
 "prev_lock_sha256": "501f77af523ba5f22eb88d2a271ecaad577de10ac4163cfea43d970ea2d0b6ca",
 "reason": "Before any target call, the runner's plan check refused to start: the log_retrieved(k=10) prompts no longer hashed to the value listed above (c7a640d719a7). Cause: persona.inject.from_logs.bm25_rank summed term scores over a Python set, whose order follows PYTHONHASHSEED, so near-equal BM25 scores ranked differently in every process (4 hash seeds gave 4 rankings on a synthetic panel). The listed hash was one random draw and cannot be reproduced. Fixed in simengine b7b25e9 (sorted term order, regression test across hash seeds). Only a 3-person narrative smoke call had been made (0.0008 USD); no target answer was seen.",
 "written_at": "2026-10-23T03:16:07+03:00",
 "written_at_utc": "2026-10-23T00:16:07+00:00"
}
```
