"""Forward hooks on decoder layers and the token embedding: residual add, state capture, soft-token replacement.

All hooks are removed when their `with` block exits (also on error). Prompts are left-padded everywhere in this
package, so position -1 is every row's last real token.

    with add_to_residual(model, 20, vecs, positions="last"):    # vecs (B, d): one vector per row
        out = model(input_ids=ids)
    with capture(model, [10, 20], pool="last") as st:
        model(input_ids=ids)
    st[10]                                                      # (B, d) float32, pre final norm
    with replace_embeddings(model, soft_id, vectors):           # vectors (B, k, d), post embed_scale
        model(input_ids=ids_with_k_soft_ids)
"""
from __future__ import annotations

import contextlib
from typing import Sequence

from .models import input_embeddings, layers


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _rebuild(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


def _layer(model, layer: int):
    """Decoder layer `layer` with a readable error when it is out of range (pick it with ld.layer_index(frac))."""
    lay = layers(model)
    if not -len(lay) <= int(layer) < len(lay):
        raise ValueError(f"layer {layer} is out of range: the model has {len(lay)} decoder layers "
                         f"(use ld.layer_index(0.5) for half depth)")
    return lay[int(layer)]


def _pos_slice(positions, T: int):
    """positions: "last" | "last<k>" | "all" | slice | int -> slice over the T positions of this forward."""
    if isinstance(positions, slice):
        return positions
    if isinstance(positions, int):
        return slice(positions, positions + 1 if positions != -1 else None)
    if positions == "all":
        return slice(0, T)
    if positions == "last":
        return slice(T - 1, T)
    if isinstance(positions, str) and positions.startswith("last"):
        k = int(positions[4:])
        return slice(max(T - k, 0), T)
    raise ValueError(f"positions must be 'last', 'last<k>', 'all', an int or a slice, not {positions!r}")


@contextlib.contextmanager
def add_to_residual(model, layer: int, delta, *, positions="last", scale: float = 1.0):
    """Add `delta` to decoder layer `layer`'s output at `positions` while inside the block.

    delta: tensor (d,), (B, d) or (B, T_sel, d), or a callable h -> such a tensor (h = the layer output, (B, T, d);
    a learned projector closes over its inputs here). Cast to the layer output's dtype / device. In a KV-cached step
    the forward has only the new tokens, so "last" is that step's last token. No in-place edit (autograd-safe)."""
    lay = _layer(model, layer)

    def hook(_m, _inp, out):
        h = _hidden(out)
        sl = _pos_slice(positions, h.shape[1])
        d = delta(h) if callable(delta) else delta
        d = d.to(device=h.device, dtype=h.dtype) * scale
        if d.dim() == 1:
            d = d.view(1, 1, -1)
        elif d.dim() == 2:
            d = d[:, None, :]
        n_sel = len(range(*sl.indices(h.shape[1])))
        if n_sel == 0:
            return out
        add = h.new_zeros(h.shape)
        add[:, sl] = d.expand(h.shape[0], n_sel, h.shape[2]) if d.shape[1] == 1 else d
        return _rebuild(out, h + add)

    handle = lay.register_forward_hook(hook)
    try:
        yield model
    finally:
        handle.remove()


@contextlib.contextmanager
def capture(model, layer_ids: Sequence[int], *, pool: str = "last", span=None):
    """Collect decoder layer outputs (the residual stream after layer l; the last layer is taken BEFORE the final
    norm, unlike hidden_states[-1]). pool: "last" -> (B, d), "mean" -> mean over `span` (B, T) bool mask or over all
    positions, "none" -> (B, T, d). Values are float32 tensors on the model device, detached. Multiple forwards in
    one block are concatenated along the batch."""
    import torch
    if pool not in ("last", "mean", "none"):
        raise ValueError("pool must be 'last', 'mean' or 'none'")
    store: dict[int, list] = {int(l): [] for l in layer_ids}
    lay = {l: _layer(model, l) for l in store}
    handles = []

    def make(l):
        def hook(_m, _inp, out):
            h = _hidden(out).detach().float()
            if pool == "last":
                v = h[:, -1]
            elif pool == "mean":
                if span is None:
                    v = h.mean(1)
                else:
                    m = span.to(h.device).float()[:, -h.shape[1]:, None]
                    v = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
            else:
                v = h
            store[l].append(v)
        return hook

    result: dict[int, object] = {}
    try:
        for l in store:
            handles.append(lay[l].register_forward_hook(make(l)))
        yield result
    finally:
        for hd in handles:
            hd.remove()
        for l, parts in store.items():
            if parts:
                result[l] = torch.cat(parts, 0) if pool != "none" or len({p.shape[1] for p in parts}) == 1 \
                    else parts


@contextlib.contextmanager
def replace_embeddings(model, placeholder_id: int, vectors):
    """Soft tokens: replace the embedding output at every `placeholder_id` position with `vectors` (B, k, d), row b
    taking vectors[b] in order. Works on every family the same way (Gemma 4 cannot generate from inputs_embeds).
    vectors are in the embedding OUTPUT scale (after Gemma's sqrt(d) embed_scale). Each row must hold exactly k
    placeholders, else ValueError. Gradients flow into `vectors`."""
    emb = input_embeddings(model)

    def hook(_m, inp, out):
        ids = inp[0] if inp else None
        if ids is None or not hasattr(ids, "eq"):
            return out
        mask = ids.eq(placeholder_id)
        n = mask.sum(1)
        if int(n.sum()) == 0:                       # a later cached step without placeholders
            return out
        k = vectors.shape[1]
        if vectors.shape[0] != ids.shape[0] or bool((n != k).any()):
            raise ValueError(f"each row needs exactly {k} placeholder tokens (got {n.tolist()}) and "
                             f"{ids.shape[0]} vector rows (got {vectors.shape[0]})")
        v = vectors.to(device=out.device, dtype=out.dtype).reshape(-1, out.shape[-1])
        new = out.clone()
        new[mask] = v
        return new

    handle = emb.register_forward_hook(hook)
    try:
        yield model
    finally:
        handle.remove()


def residual_norm(model, tok, texts: Sequence[str], layer: int, *, batch_size: int = 8) -> float:
    """Median L2 norm of layer `layer`'s output over the last token of `texts` (CAA coefficient unit)."""
    import torch
    norms = []
    dev = next(model.parameters()).device
    for i in range(0, len(texts), batch_size):
        enc = tok(list(texts[i:i + batch_size]), return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.no_grad(), capture(model, [layer], pool="last") as st:
            model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev))
        norms.append(st[layer].norm(dim=-1).cpu())
    return float(torch.cat(norms).median())


def embedding_norm(model, ids=None) -> float:
    """Median L2 norm of the embedding output (post embed_scale) over `ids` or the whole vocabulary."""
    import torch
    emb = input_embeddings(model)
    with torch.no_grad():
        w = emb.weight if ids is None else emb.weight[ids]
        scale = getattr(emb, "embed_scale", 1.0)
        scale = float(scale.item() if hasattr(scale, "item") else scale)
        return float((w.float().norm(dim=-1) * scale).median())


__all__ = ["add_to_residual", "capture", "replace_embeddings", "residual_norm", "embedding_norm"]
