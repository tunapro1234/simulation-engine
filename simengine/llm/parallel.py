"""Order-preserving thread-pool map with stop-the-run semantics.

Ported from talp73/kod/llm/openrouter.py (`paralel`): a `StopError` (budget, 401, spend limit)
cancels everything not yet started and is re-raised; finished calls are already in the disk cache.
"""
from __future__ import annotations

from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Any, Callable, Sequence

from .errors import StopError


def parallel_map(fn: Callable[[Any], Any], items: Sequence[Any], parallel: int = 4,
                 return_exceptions: bool = False) -> list[Any]:
    """Apply `fn` to every item with up to `parallel` threads; results in input order.

    return_exceptions=False: the first exception is re-raised (pending items are cancelled).
    return_exceptions=True: ordinary exceptions are put into their result slot; a StopError is
    still raised (after cancelling pending items), because continuing cannot succeed.
    """
    items = list(items)
    if not items:
        return []
    parallel = max(1, int(parallel))
    if parallel == 1 or len(items) == 1:
        out = []
        for x in items:
            try:
                out.append(fn(x))
            except StopError:
                raise
            except Exception as e:  # noqa: BLE001
                if not return_exceptions:
                    raise
                out.append(e)
        return out
    results: list[Any] = [None] * len(items)
    with ThreadPoolExecutor(min(parallel, len(items))) as ex:
        futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
        pending = set(futs)
        while pending:
            done, pending = wait(pending, return_when=FIRST_EXCEPTION)
            for f in done:
                i = futs[f]
                if f.cancelled():
                    continue
                e = f.exception()
                if e is None:
                    results[i] = f.result()
                elif isinstance(e, StopError) or not return_exceptions:
                    for g in pending:
                        g.cancel()
                    raise e
                else:
                    results[i] = e
    return results


__all__ = ["parallel_map"]
