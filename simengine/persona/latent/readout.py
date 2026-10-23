"""Prefix-cached readout: P(yes) and option distributions over many (prefix, suffix, vector) cells.

Every prefix runs once (left-padded batches); its KV / recurrent cache is copied row-wise to the cells that use
it (`select_cache`, works for the qwen3_5 hybrid cache and Gemma 4 sliding layers), and only the suffix tokens
run per cell. Cells are grouped by suffix length, so suffixes are never padded and position -1 is each row's last
token. An `Injection` adds a per-cell residual vector or replaces soft placeholder embeddings in the suffix.

    pre = [prefix_of(person) for person in people]                 # person part, cached once each
    suf = [suffix_of(cat) for cat in categories]                    # " Books between ...? Answer Yes or No.<tail>"
    p, mass = yes_prob(ld, pre, suf)                                # (N, M) arrays: P(Yes)/(P(Yes)+P(No)), Yes+No mass
    cells = np.array([[c, 0, u] for c in range(20) for u in range(n)])   # short prompts: prefix = category
    p, mass = yes_prob(ld, short_pre, [tail], cells=cells, inject=Injection("residual", vecs, layer=30))
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .hooks import add_to_residual, capture, replace_embeddings
from .prompts import YES_NO, answer_token_ids


@dataclass
class Injection:
    """Per-cell person vector put inside the model during the suffix pass.

    kind "residual": vectors (V, d) added to decoder layer `layer`'s output at `positions` ("last", "last4", "all").
    kind "soft": vectors (V, k, d) replace the k `soft_id` placeholders of the suffix (embedding output scale).
    `fn(idx)` instead of `vectors`: a callable giving the rows for vector indices idx (LongTensor) - a learned
    projector, so gradients flow during training. `scale` multiplies the vectors (CAA coefficient)."""
    kind: str
    vectors: Any = None
    layer: int | None = None
    positions: str = "last"
    soft_id: int | None = None
    fn: Callable | None = None
    scale: float = 1.0

    def rows(self, idx, device, dtype):
        import torch
        if self.fn is not None:
            v = self.fn(idx.to(device))
        else:
            v = self.vectors if torch.is_tensor(self.vectors) else torch.as_tensor(np.asarray(self.vectors))
            v = v.to(device)[idx.to(device)]
        return (v * self.scale).to(dtype) if self.scale != 1.0 else v.to(dtype)


@dataclass
class PrefixOut:
    cache: Any
    mask: Any
    hidden: dict
    n_tokens: np.ndarray


def _ids(tok, texts: Sequence[str]) -> list[list[int]]:
    return [tok(t, add_special_tokens=False)["input_ids"] for t in texts]


def _left_pad(seqs: Sequence[Sequence[int]], pad_id: int, device):
    import torch
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s):
            ids[i, T - len(s):] = torch.as_tensor(s, dtype=torch.long)
            mask[i, T - len(s):] = 1
    return ids.to(device), mask.to(device)


def _pad_id(tok) -> int:
    return tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)


def _assign_recurrent(layer):
    """Make a linear-attention cache layer assign its new recurrent state instead of copying it in place (an in-place
    copy into a tensor autograd still needs breaks the backward pass of the injectors)."""
    def update_recurrent_state(recurrent_states, state_idx: int = 0, **kwargs):
        layer.recurrent_states[state_idx] = recurrent_states
        layer.is_recurrent_states_initialized[state_idx] = True
        return recurrent_states
    layer.update_recurrent_state = update_recurrent_state
    layer.record_past = True                       # conv states are then assigned too, not copied in place


def select_cache(cache, idx):
    """A new cache holding rows `idx` (LongTensor) of `cache`; the original is left untouched (KV layers, linear
    attention conv / recurrent states, sliding-window layers). The copy updates out of place (autograd-safe)."""
    new = copy.copy(cache)
    new.layers = []
    for lay in cache.layers:
        c = copy.copy(lay)
        for k, v in vars(lay).items():
            if isinstance(v, dict):
                setattr(c, k, dict(v))
        c.reorder_cache(idx)
        if hasattr(c, "update_recurrent_state"):
            _assign_recurrent(c)
        new.layers.append(c)
    return new


def expand_cache(cache, n: int):
    """Copy a one-row cache to n rows."""
    import torch
    dev = None
    for lay in cache.layers:
        t = getattr(lay, "keys", None)
        if t is not None and hasattr(t, "device"):
            dev = t.device
            break
    return select_cache(cache, torch.zeros(n, dtype=torch.long, device=dev))


def prefix_forward(ld, seqs: Sequence[Sequence[int]], *, layers: Sequence[int] = (), pool: str = "last",
                   spans: Sequence[tuple[int, int]] | None = None, keep_cache: bool = True):
    """One left-padded forward over token sequences. Returns (cache | None, mask, {layer: (B, d) float32 tensor}).
    spans[i] = (start, end) token range of row i's persona block for pool="mean"."""
    import torch
    from transformers import DynamicCache
    ids, mask = _left_pad(seqs, _pad_id(ld.tok), ld.device)
    span = None
    if pool == "mean" and spans is not None:
        span = torch.zeros_like(mask, dtype=torch.bool)
        T = ids.shape[1]
        for i, (a, b) in enumerate(spans):
            off = T - len(seqs[i])
            span[i, off + a:off + b] = True
    cache = DynamicCache(config=ld.model.config) if keep_cache else None
    with torch.no_grad(), capture(ld.model, list(layers), pool=pool, span=span) as st:
        ld.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=keep_cache,
                 logits_to_keep=1)                     # logits unused: no (B, T, vocab) tensor
    return cache, mask, dict(st)


def prefix_states(ld, prefixes: Sequence[str], *, layers: Sequence[int] = (), pool: str = "last",
                  spans: Sequence[tuple[int, int]] | None = None, batch_size: int = 8) -> PrefixOut:
    """Residual states of each prefix at `layers` (no cache kept): hidden {layer: (N, d) float32 numpy}."""
    seqs = _ids(ld.tok, prefixes)
    order = np.argsort([len(s) for s in seqs], kind="stable")       # similar lengths per batch: less padding
    parts: dict[int, list] = {int(l): [] for l in layers}
    for b in range(0, len(order), batch_size):
        ix = order[b:b + batch_size]
        _, _, st = prefix_forward(ld, [seqs[i] for i in ix], layers=layers, pool=pool, keep_cache=False,
                                  spans=None if spans is None else [spans[i] for i in ix])
        for l in parts:
            parts[l].append(st[l].cpu().numpy())
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    hidden = {l: np.concatenate(v, 0)[inv] for l, v in parts.items() if v}
    return PrefixOut(None, None, hidden, np.array([len(s) for s in seqs]))


def suffix_logits(ld, cache, mask, rows, suffix, *, inject: Injection | None = None, vec_idx=None):
    """Last-position logits (C, V) of `suffix` (C, Ls) LongTensor run on cache rows `rows`. Gradients flow when the
    injection's fn has trainable parameters (call under torch.enable_grad)."""
    import contextlib
    import torch
    rows = torch.as_tensor(rows, dtype=torch.long, device=ld.device)
    c = select_cache(cache, rows)
    am = torch.cat([mask[rows], torch.ones_like(suffix)], 1)
    ctx = contextlib.nullcontext()
    if inject is not None:
        vecs = inject.rows(torch.as_tensor(vec_idx, dtype=torch.long), ld.device, torch.float32)
        if inject.kind == "residual":
            ctx = add_to_residual(ld.model, int(inject.layer), vecs, positions=inject.positions)
        elif inject.kind == "soft":
            ctx = replace_embeddings(ld.model, int(inject.soft_id), vecs)
        else:
            raise ValueError("Injection.kind must be 'residual' or 'soft'")
    with ctx:
        out = ld.model(input_ids=suffix, attention_mask=am, past_key_values=c, use_cache=True, logits_to_keep=1)
    return out.logits[:, -1].float()


def _two_way(logits, yes_ids, no_ids):
    import torch
    lp = torch.log_softmax(logits.float(), -1)
    ly = torch.logsumexp(lp[:, yes_ids], -1)
    ln = torch.logsumexp(lp[:, no_ids], -1)
    return ly, ln


def yes_logit(logits, yes_ids, no_ids):
    """log P(yes) - log P(no) over the answer-token sets (the BCE logit of the injectors)."""
    ly, ln = _two_way(logits, yes_ids, no_ids)
    return ly - ln


def _plan(cells: np.ndarray, suffix_len: np.ndarray, batch_size: int, cell_batch: int):
    """Yield (prefix ids of the batch, [(cell positions, suffix length)])."""
    pref = np.unique(cells[:, 0])
    for b in range(0, len(pref), batch_size):
        pb = pref[b:b + batch_size]
        sel = np.flatnonzero(np.isin(cells[:, 0], pb))
        groups = []
        lens = suffix_len[cells[sel, 1]]
        for L in np.unique(lens):
            g = sel[lens == L]
            for k in range(0, len(g), cell_batch):
                groups.append(g[k:k + cell_batch])
        yield pb, groups


def yes_prob(ld, prefixes: Sequence[str], suffixes: Sequence[str], *, cells=None, inject: Injection | None = None,
             words: Sequence[str] = YES_NO, batch_size: int = 8, cell_batch: int = 64):
    """P(words[0]) / (P(words[0]) + P(words[1])) at the end of prefix + suffix, and the two words' probability mass.
    cells: (K, 2|3) int array of (prefix, suffix[, vector]) indices -> (K,) arrays; None = every prefix x suffix ->
    (N, M) arrays (vector index = prefix index)."""
    import torch
    full = cells is None
    if full:
        N, M = len(prefixes), len(suffixes)
        cells = np.stack(np.meshgrid(np.arange(N), np.arange(M), indexing="ij"), -1).reshape(-1, 2)
    cells = np.asarray(cells, dtype=np.int64)
    if cells.shape[1] == 2:
        cells = np.concatenate([cells, cells[:, :1]], 1)
    ans = answer_token_ids(ld.tok, words)
    yes_ids, no_ids = ans[words[0]], ans[words[1]]
    pre_ids, suf_ids = _ids(ld.tok, prefixes), _ids(ld.tok, suffixes)
    suf_len = np.array([len(s) for s in suf_ids])
    p = np.full(len(cells), np.nan)
    mass = np.full(len(cells), np.nan)
    with torch.no_grad():
        for pb, groups in _plan(cells, suf_len, batch_size, cell_batch):
            cache, mask, _ = prefix_forward(ld, [pre_ids[i] for i in pb])
            where = {int(v): i for i, v in enumerate(pb)}
            for g in groups:
                rows = [where[int(i)] for i in cells[g, 0]]
                suffix = torch.as_tensor([suf_ids[int(j)] for j in cells[g, 1]], dtype=torch.long, device=ld.device)
                lg = suffix_logits(ld, cache, mask, rows, suffix, inject=inject, vec_idx=cells[g, 2])
                ly, ln = _two_way(lg, yes_ids, no_ids)
                p[g] = torch.sigmoid(ly - ln).cpu().numpy()
                mass[g] = (ly.exp() + ln.exp()).cpu().numpy()
            del cache
    if full:
        return p.reshape(N, M), mass.reshape(N, M)
    return p, mass


def option_probs(ld, prefixes: Sequence[str], suffixes: Sequence[str], labels: Sequence[Sequence[str]], *,
                 cells=None, batch_size: int = 8, cell_batch: int = 64) -> tuple[list[np.ndarray], np.ndarray]:
    """Distribution over each cell's option labels (renormalised over the labels' first tokens) and its mass.
    labels[j] belongs to suffix j. cells as in yes_prob ((K, 2)); returns ([K arrays], mass (K,))."""
    import torch
    if cells is None:
        cells = np.stack(np.meshgrid(np.arange(len(prefixes)), np.arange(len(suffixes)), indexing="ij"),
                         -1).reshape(-1, 2)
    cells = np.asarray(cells, dtype=np.int64)
    lab_ids = []
    for labs in labels:
        ids = [ld.tok.encode(l, add_special_tokens=False)[0] for l in labs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"option labels {list(labs)} share first tokens")
        lab_ids.append(ids)
    pre_ids, suf_ids = _ids(ld.tok, prefixes), _ids(ld.tok, suffixes)
    suf_len = np.array([len(s) for s in suf_ids])
    out: list = [None] * len(cells)
    mass = np.full(len(cells), np.nan)
    with torch.no_grad():
        for pb, groups in _plan(cells, suf_len, batch_size, cell_batch):
            cache, mask, _ = prefix_forward(ld, [pre_ids[i] for i in pb])
            where = {int(v): i for i, v in enumerate(pb)}
            for g in groups:
                rows = [where[int(i)] for i in cells[g, 0]]
                suffix = torch.as_tensor([suf_ids[int(j)] for j in cells[g, 1]], dtype=torch.long, device=ld.device)
                pr = torch.softmax(suffix_logits(ld, cache, mask, rows, suffix), -1).cpu().numpy()
                for r, k in enumerate(g):
                    q = pr[r, lab_ids[int(cells[k, 1])]]
                    mass[k] = q.sum()
                    out[k] = q / max(q.sum(), 1e-12)
            del cache
    return out, mass


def token_span(tok, text: str, sub: str) -> tuple[int, int]:
    """Token index range [a, b) of the first occurrence of `sub` inside `text` (tokenised alone, no specials)."""
    i = text.find(sub)
    if i < 0:
        raise ValueError("sub not in text")
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    a = next(k for k, (s, e) in enumerate(offs) if e > i)
    b = next((k for k, (s, e) in enumerate(offs) if s >= i + len(sub)), len(offs))
    return a, b


__all__ = ["Injection", "PrefixOut", "select_cache", "expand_cache", "prefix_forward", "prefix_states",
           "suffix_logits", "yes_logit", "yes_prob", "option_probs", "token_span"]
