# Pre-registration addendum EK-2 (2026-10-23 03:58 +0300)

_Amends `inject-ablation-2026-10-22.md` after its lock. The parent file and its lock are not edited; this addendum has its own lock, chained to the parent's._

## Parent
- file: `inject-ablation-2026-10-22.md`
- sha256: `c4d436a9bb437323645813cc5a9a81a9f7af5a4a2eb0c412708e53687bccee88`
- locked at: 2026-10-23T01:54:28+03:00
- lock sha256: `501f77af523ba5f22eb88d2a271ecaad577de10ac4163cfea43d970ea2d0b6ca`
- previous addendum: `inject-ablation-2026-10-22-EK1.md` (sha256 `da85d33bd059a5b956860a205314183c4b5b91b5d3e91c837a77ab0f19a29996`)

## Reason
Before any target call, the second smoke run passed part A (narrative, llm_narrative, own_answers_all 18/18 parsed; 0.0047 USD) and then the plan check refused part B: own_answers_rag(k=8) and swap_own_answers_rag(k=8) no longer hashed to the EK-1 plan (caf6131711e7, 6c4109d9bcc4). Cause: persona.inject.own_answers.score_bm25 summed term scores over a Python set of query terms, the same PYTHONHASHSEED dependence fixed for bm25_rank in EK-1, so the retrieved answers differed per process. Fixed in simengine cef20ba (sorted term order; regression test across hash seeds fails on the old code). No score was computed and no target answer was compared.

## Deviation
own_answers_rag and swap_own_answers_rag in part B use the deterministic BM25 of cef20ba; their prompt sha256 become 10464e056a19 and ea1478b41f8b. The full dry-run plan was regenerated twice in separate processes (23 Oct 03:49 and 03:5x) and the two plan.json files are byte-identical; the EK-1 plan is kept as plan-ek1.json. Every other arm's prompt hash, both parts' cell hashes, the arm lists, the cost bound (7.0389 USD) and the runner (8d908e69f69d) are unchanged. Data, cells, arms, metrics, comparisons, Holm families, bootstrap and decision rules are unchanged.

## Falsification
Unchanged from the parent.

## Changes
- own_answers_rag(k=8) prompt sha256 caf6131711e7 -> 10464e056a19 (deterministic score_bm25, cef20ba)
- swap_own_answers_rag(k=8) prompt sha256 6c4109d9bcc4 -> ea1478b41f8b (deterministic score_bm25, cef20ba)

## Machine-readable copy
```json
{
 "changes": [
  "own_answers_rag(k=8) prompt sha256 caf6131711e7 -> 10464e056a19 (deterministic score_bm25, cef20ba)",
  "swap_own_answers_rag(k=8) prompt sha256 6c4109d9bcc4 -> ea1478b41f8b (deterministic score_bm25, cef20ba)"
 ],
 "deviation": "own_answers_rag and swap_own_answers_rag in part B use the deterministic BM25 of cef20ba; their prompt sha256 become 10464e056a19 and ea1478b41f8b. The full dry-run plan was regenerated twice in separate processes (23 Oct 03:49 and 03:5x) and the two plan.json files are byte-identical; the EK-1 plan is kept as plan-ek1.json. Every other arm's prompt hash, both parts' cell hashes, the arm lists, the cost bound (7.0389 USD) and the runner (8d908e69f69d) are unchanged. Data, cells, arms, metrics, comparisons, Holm families, bootstrap and decision rules are unchanged.",
 "falsification": "",
 "index": 2,
 "kind": "prereg_addendum",
 "notes": "",
 "parent": "inject-ablation-2026-10-22.md",
 "parent_lock_sha256": "501f77af523ba5f22eb88d2a271ecaad577de10ac4163cfea43d970ea2d0b6ca",
 "parent_locked_at": "2026-10-23T01:54:28+03:00",
 "parent_sha256": "c4d436a9bb437323645813cc5a9a81a9f7af5a4a2eb0c412708e53687bccee88",
 "prev_addendum": "inject-ablation-2026-10-22-EK1.md",
 "prev_addendum_sha256": "da85d33bd059a5b956860a205314183c4b5b91b5d3e91c837a77ab0f19a29996",
 "prev_lock_sha256": "4fa159ae7c92b9995eb3098ac0ad808b9a617543c17f7e3d5cede2fd4948c4c4",
 "reason": "Before any target call, the second smoke run passed part A (narrative, llm_narrative, own_answers_all 18/18 parsed; 0.0047 USD) and then the plan check refused part B: own_answers_rag(k=8) and swap_own_answers_rag(k=8) no longer hashed to the EK-1 plan (caf6131711e7, 6c4109d9bcc4). Cause: persona.inject.own_answers.score_bm25 summed term scores over a Python set of query terms, the same PYTHONHASHSEED dependence fixed for bm25_rank in EK-1, so the retrieved answers differed per process. Fixed in simengine cef20ba (sorted term order; regression test across hash seeds fails on the old code). No score was computed and no target answer was compared.",
 "written_at": "2026-10-23T03:58:08+03:00",
 "written_at_utc": "2026-10-23T00:58:08+00:00"
}
```
