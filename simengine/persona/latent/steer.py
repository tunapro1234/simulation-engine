"""Steering: fixed directions (mean difference / CAA, PCA, probe weights), CAA person vectors, and learned
injectors (vector -> residual projector, vector -> k soft tokens) trained against a frozen model with BCE on P(yes).

    d = mean_diff_direction(states[y == 1], states[y == 0])          # or pca_direction / probe_direction
    caa = CAA(dirs, layer=30, c=0.1, unit_norm=med_norm)             # E5 trait directions -> per-person vector
    inj = caa.injection(trait_values_array, ref=train_values)        # Injection for readout.yes_prob
    proj = residual_projector(d_in=U.shape[1], d_model=ld.hidden_size)
    proj, hist = train_injector(ld, proj, "residual", prefixes, suffix, U_fit, cells_fit, U_val, cells_val,
                                layer=30, positions="last", lr=3e-4, epochs=2, seed=0)
    p = predict_injector(ld, proj, "residual", prefixes, suffix, U_test, cells_test, layer=30, positions="last")
"""
from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np

from .readout import Injection, _ids, prefix_forward, suffix_logits, yes_logit, yes_prob
from .prompts import YES_NO, answer_token_ids


# ------------------------------------------------------------------------------------------------- directions
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def mean_diff_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """CAA / mass-mean direction: mean(pos) - mean(neg) (not normalised)."""
    return (np.asarray(pos, np.float64).mean(0) - np.asarray(neg, np.float64).mean(0)).astype(np.float32)


def pca_direction(states: np.ndarray, y: np.ndarray | None = None, *, k: int = 0) -> np.ndarray:
    """Unit k-th principal direction of the centred states; with y, the sign points to higher y."""
    X = np.asarray(states, np.float64)
    X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    d = vt[k]
    if y is not None and np.corrcoef(X @ d, y)[0, 1] < 0:
        d = -d
    return d.astype(np.float32)


def probe_direction(states: np.ndarray, y: np.ndarray, *, C: float = 0.1, seed: int = 0) -> np.ndarray:
    """Unit weight vector of an L2 logistic probe on standardised states, mapped back to state space."""
    from sklearn.linear_model import LogisticRegression
    X = np.asarray(states, np.float64)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit((X - X.mean(0)) / sd, y)
    return unit(lr.coef_[0] / sd).astype(np.float32)


class CAA:
    """R4: person vector = sum_t z_t d_t (E5), divided by the training 95th-percentile norm, times c x `unit_norm`
    (the median residual norm at the layer), added at `layer` / `positions`."""

    def __init__(self, directions: Mapping[str, np.ndarray], layer: int, c: float, *, unit_norm: float = 1.0,
                 positions: str = "last"):
        self.directions, self.layer, self.c, self.unit_norm, self.positions = dict(directions), layer, c, \
            unit_norm, positions

    def vectors(self, values: np.ndarray, *, ref: np.ndarray | None = None) -> np.ndarray:
        from .encoders import trait_person_vector
        ref = values if ref is None else ref
        V = trait_person_vector(values, self.directions, ref=ref)
        R = trait_person_vector(ref, self.directions, ref=ref)
        q = np.percentile(np.linalg.norm(R, axis=1), 95) or 1.0
        return (V / q * self.c * self.unit_norm).astype(np.float32)

    def injection(self, values: np.ndarray, *, ref: np.ndarray | None = None) -> Injection:
        return Injection("residual", self.vectors(values, ref=ref), layer=self.layer, positions=self.positions)


# ------------------------------------------------------------------------------------------------- learned
def residual_projector(d_in: int, d_model: int, hidden: int = 256, *, seed: int = 0):
    """R5: u -> Linear(hidden) -> GELU -> Linear(d_model), output layer zero-initialised (starts as no-op)."""
    import torch
    torch.manual_seed(seed)
    nn = torch.nn
    net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_model))
    nn.init.zeros_(net[2].weight)
    nn.init.zeros_(net[2].bias)
    return net


def soft_token_projector(d_in: int, d_model: int, k: int = 1, hidden: int = 256, *, target_norm: float = 1.0,
                         seed: int = 0):
    """R6: u -> Linear(hidden) -> GELU -> Linear(k * d_model) -> (B, k, d), each token rescaled to `target_norm`
    (the median input-embedding norm, after Gemma's embed_scale)."""
    import torch
    torch.manual_seed(seed)
    nn = torch.nn

    class SoftTokens(nn.Module):
        def __init__(self):
            super().__init__()
            self.k, self.d, self.target_norm = k, d_model, float(target_norm)
            self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, k * d_model))

        def forward(self, u):
            v = self.net(u).view(u.shape[0], self.k, self.d)
            return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.target_norm

    return SoftTokens()


def derange(n: int, seed: int = 0) -> np.ndarray:
    """Permutation of range(n) with no fixed point (swap control). n >= 2."""
    if n < 2:
        raise ValueError("a derangement needs n >= 2")
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def _injection(kind, proj, U, layer, positions, soft_id):
    import torch
    Ut = torch.as_tensor(np.asarray(U, np.float32))

    def fn(idx):
        dev = next(proj.parameters()).device
        return proj(Ut.to(dev)[idx.to(dev)])
    return Injection(kind, fn=fn, layer=layer, positions=positions, soft_id=soft_id)


def soft_placeholder(tok) -> tuple[int, str]:
    """(id, text) of the soft-token placeholder: '<|soft|>' when the tokenizer has it, else its pad / eos token."""
    for t in ("<|soft|>",):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id and i is not None and i >= 0:
            return i, t
    t = tok.pad_token or tok.eos_token
    return tok.convert_tokens_to_ids(t), t


def train_injector(ld, proj, kind: str, prefixes: Sequence[str], suffix: str, U_fit, cells_fit, U_val, cells_val, *,
                   layer: int | None, positions: str = "last", lr: float = 3e-4, epochs: float = 1.0,
                   batch_size: int = 64, seed: int = 0, eval_every: float = 0.5, weight_decay: float = 0.01,
                   soft_id: int | None = None, score_fn=None, words: Sequence[str] = YES_NO, guard=None,
                   uids_fit=None, max_seconds: float | None = None):
    """Train `proj` with the model frozen. cells_*: int arrays (K, 3) of (prefix index, person row in U_*, y).
    Loss: BCE on log P(yes) - log P(no) at the end of prefix + suffix with proj(u) injected. Prefix caches are built
    once. Validation every `eval_every` epochs: score_fn(p_val, cells_val) -> float (higher is better; default
    -log loss); the best state is kept. guard.check_uids(uids_fit) runs first when given. Returns (proj, history)."""
    import torch
    if guard is not None and uids_fit is not None:
        guard.check_uids(uids_fit, "train_injector")
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    dev = ld.device
    proj = proj.to(dev).float().train()
    ans = answer_token_ids(ld.tok, words)
    yes_ids, no_ids = ans[words[0]], ans[words[1]]
    pre_ids = _ids(ld.tok, prefixes)
    suf = torch.as_tensor(_ids(ld.tok, [suffix])[0], dtype=torch.long, device=dev)
    cache, mask, _ = prefix_forward(ld, pre_ids)
    cells_fit = np.asarray(cells_fit, np.int64)
    inj = _injection(kind, proj, U_fit, layer, positions, soft_id)
    opt = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(cells_fit)
    steps = int(np.ceil(epochs * n / batch_size))
    every = max(int(np.ceil(eval_every * n / batch_size)), 1)
    hist, best, best_state = [], -np.inf, None
    t0 = time.time()
    y_all = torch.as_tensor(cells_fit[:, 2], dtype=torch.float32, device=dev)
    perm = torch.randperm(n, generator=g)
    pos = 0
    for step in range(1, steps + 1):
        if pos >= n:
            perm, pos = torch.randperm(n, generator=g), 0
        b = perm[pos:pos + batch_size].numpy()
        pos += batch_size
        with torch.enable_grad():
            lg = suffix_logits(ld, cache, mask, cells_fit[b, 0], suf.expand(len(b), -1), inject=inj,
                               vec_idx=cells_fit[b, 1])
            z = yes_logit(lg, yes_ids, no_ids)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y_all[torch.as_tensor(b, device=dev)])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step % every == 0 or step == steps:
            proj.eval()
            p = predict_injector(ld, proj, kind, prefixes, suffix, U_val, np.asarray(cells_val)[:, :2], layer=layer,
                                 positions=positions, soft_id=soft_id, batch_size=batch_size)
            yv = np.asarray(cells_val)[:, 2]
            sc = score_fn(p, cells_val) if score_fn else float(np.mean(yv * np.log(np.clip(p, 1e-6, 1)) +
                                                                          (1 - yv) * np.log(np.clip(1 - p, 1e-6, 1))))
            hist.append({"step": step, "epoch": round(step * batch_size / n, 3), "loss": float(loss.detach()), "val": sc,
                         "seconds": round(time.time() - t0, 1)})
            if sc > best:
                best = sc
                best_state = {k: v.detach().clone() for k, v in proj.state_dict().items()}
            proj.train()
            if max_seconds and time.time() - t0 > max_seconds:
                break
    if best_state is not None:
        proj.load_state_dict(best_state)
    return proj.eval(), hist


def predict_injector(ld, proj, kind: str, prefixes: Sequence[str], suffix: str, U, cells, *, layer: int | None,
                     positions: str = "last", soft_id: int | None = None, batch_size: int = 64) -> np.ndarray:
    """P(yes) per cell (K, 2) = (prefix index, person row in U) with proj(U[row]) injected."""
    import torch
    cells = np.asarray(cells, np.int64)[:, :2]
    inj = _injection(kind, proj, U, layer, positions, soft_id)
    c3 = np.stack([cells[:, 0], np.zeros(len(cells), np.int64), cells[:, 1]], 1)
    with torch.no_grad():
        p, _ = yes_prob(ld, prefixes, [suffix], cells=c3, inject=inj, batch_size=max(len(prefixes), 1),
                        cell_batch=batch_size)
    return p


__all__ = ["unit", "mean_diff_direction", "pca_direction", "probe_direction", "CAA", "residual_projector",
           "soft_token_projector", "soft_placeholder", "derange", "train_injector", "predict_injector"]
