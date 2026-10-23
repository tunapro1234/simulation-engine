"""Person -> vector extractors (E1-E7 of the latent-persona prereg). Every fitted extractor is fitted on training
persons only (pass the TestGuard of the task); transform works for anyone.

    an = SurveyOneHot().fit(survey[fit_mask]).transform(survey)                      # E1, sparse-free dense array
    ae = SurveyAutoencoder(dim=16, seed=0).fit(an_fit).transform(an)                 # E1b, MLP autoencoder codes
    gc = history_features(g.history, uids, cats)                                     # E2, klasik features, wide
    als = ImplicitALS(dim=32, seed=0).fit(counts_fit).transform(counts)              # E2, user factors (fold-in)
    bag = BagSVD(dim=32, seed=0).fit(counts_fit).transform(counts)                   # E2b, tf-idf bag + SVD
    sq = SequenceEncoder(dim=64, seed=0).fit(ev, fit_uids, val_uids, guard=g).transform(ev, uids)  # E2c, GRU
    e5 = text_embedding(texts)                                                       # E3, multilingual-e5-base
    hs = llm_states(ld, prefixes, [20, 40], pool="last")                             # E4, residual states
    dirs = trait_directions(ld, profiles, TRAIT_EDITS_G1, prefix_fn, layer=30)       # E5, paired profile edits
    bd = behaviour_directions(states, y, cats)                                       # E6, buyers - non-buyers
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

# E5: profile line label, ordered levels (low ... high) of the 8 talp73 traits (kisi_ozellik column -> L1 line)
TRAIT_EDITS_G1: dict[str, dict] = {
    "yas": {"label": "Age group", "levels": ["18 - 24 years", "25 - 34 years", "35 - 44 years", "45 - 54 years",
                                             "55 - 64 years", "65 and older"]},
    "gelir": {"label": "Household income before taxes, past 12 months",
              "levels": ["Less than $25,000", "$25,000 - $49,999", "$50,000 - $74,999", "$75,000 - $99,999",
                         "$100,000 - $149,999", "$150,000 or more"]},
    "egitim": {"label": "Highest education completed",
               "levels": ["Some high school or less", "High school diploma or GED", "Bachelor's degree",
                          "Graduate or professional degree (MA, MS, MBA, PhD, JD, MD, DDS, etc)"]},
    "cinsiyet": {"label": "Gender", "levels": ["Female", "Male"]},
    "hane_buyuklugu": {"label": "Household size", "levels": ["1 (just me!)", "2", "3", "4+"]},
    "hesap_kisi": {"label": "Number of people who log in and order with this Amazon account",
                   "levels": ["1 (just me!)", "2", "3", "4+"]},
    "siklik_beyan": {"label": "How often the account orders Amazon deliveries",
                     "levels": ["Less than 5 times per month", "5 - 10 times per month",
                                "More than 10 times per month"]},
    "hd_cocuk": {"label": "Life changes in 2021 (self or account sharer)", "levels": ["None", "Had a child"],
                 "value_map": {0: 0, 1: 1}},
}
TRAITS_G1 = tuple(TRAIT_EDITS_G1)


# ------------------------------------------------------------------------------------------------- E1 survey
class SurveyOneHot:
    """One-hot of every survey column (strings and small ints alike); categories from the fit rows only, unseen
    values -> all zeros."""

    def __init__(self, columns: Sequence[str] | None = None):
        self.columns = list(columns) if columns else None

    def fit(self, survey: pd.DataFrame) -> "SurveyOneHot":
        from sklearn.preprocessing import OneHotEncoder
        cols = self.columns or [c for c in survey.columns if c != "uid"]
        self.columns = cols
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.enc.fit(survey[cols].astype(str).to_numpy())
        return self

    def transform(self, survey: pd.DataFrame) -> np.ndarray:
        return self.enc.transform(survey[self.columns].astype(str).to_numpy())

    def feature_names(self) -> list[str]:
        return list(self.enc.get_feature_names_out(self.columns))


class SurveyAutoencoder:
    """MLP autoencoder on the one-hot survey: x -> 64 -> dim -> 64 -> x (BCE). Codes are the persona vector."""

    def __init__(self, dim: int = 16, hidden: int = 64, epochs: int = 200, lr: float = 3e-3, seed: int = 0):
        self.dim, self.hidden, self.epochs, self.lr, self.seed = dim, hidden, epochs, lr, seed

    def fit(self, X: np.ndarray) -> "SurveyAutoencoder":
        import torch
        g = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        d = X.shape[1]
        nn = torch.nn
        self.enc = nn.Sequential(nn.Linear(d, self.hidden), nn.GELU(), nn.Linear(self.hidden, self.dim))
        self.dec = nn.Sequential(nn.Linear(self.dim, self.hidden), nn.GELU(), nn.Linear(self.hidden, d))
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=self.lr)
        x = torch.as_tensor(X, dtype=torch.float32)
        for _ in range(self.epochs):
            perm = torch.randperm(len(x), generator=g)
            for i in range(0, len(x), 256):
                b = x[perm[i:i + 256]]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(self.dec(self.enc(b)), b)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.loss_ = float(loss)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            return self.enc(torch.as_tensor(X, dtype=torch.float32)).numpy()


# ------------------------------------------------------------------------------------------------- E2 history
def history_features(history: pd.DataFrame, uids: Sequence, categories: Sequence[str], *,
                     columns: Sequence[str] | None = None) -> np.ndarray:
    """klasik.py history features of every category, one row per uid (history keyed uid x oge at that person's
    cut): per-category counts / decays / recency (log1p; missing recency -> 5000 days) + the user totals once.
    Shape (len(uids), 20 * n_cat_features + n_user_features)."""
    h = history.copy()
    for c in ("son_alim",):
        if c in h:
            h[c] = (pd.to_datetime(h["kesim"]) - pd.to_datetime(h[c])).dt.days
    skip = {"uid", "kesim", "oge"}
    num = [c for c in (columns or h.columns) if c not in skip and pd.api.types.is_numeric_dtype(h[c])]
    user = [c for c in num if c.startswith("u_")]
    cat = [c for c in num if c not in user]
    for c in cat + user:
        v = h[c].astype(float)
        if c.endswith("yas") or c == "son_alim":
            v = v.fillna(5000.0)
        h[c] = np.log1p(v.fillna(0.0).clip(lower=0))
    w = h.pivot_table(index="uid", columns="oge", values=cat, aggfunc="first")
    w = w.reindex(columns=pd.MultiIndex.from_product([cat, list(categories)]))
    u = h.groupby("uid")[user].first()
    X = pd.concat([w.set_axis([f"{a}|{b}" for a, b in w.columns], axis=1), u], axis=1)
    return X.reindex(pd.Index(list(uids))).fillna(0.0).to_numpy(np.float32)


def count_matrix(events: pd.DataFrame, uids: Sequence, *, vocab: Sequence[str] | None = None,
                 col: str = "kategori"):
    """User x category line-item counts (scipy csr) from gecmis_olay rows (already cut per person). Returns
    (matrix, vocab). Pass the training vocab when transforming other persons."""
    import scipy.sparse as sp
    ev = events[events["uid"].isin(set(uids))]
    vocab = list(vocab) if vocab is not None else sorted(ev[col].dropna().unique().tolist())
    vi = {v: i for i, v in enumerate(vocab)}
    ui = {u: i for i, u in enumerate(uids)}
    c = ev.groupby(["uid", col]).size().reset_index(name="n")
    c = c[c[col].isin(vi)]
    M = sp.csr_matrix((c["n"].to_numpy(float), (c["uid"].map(ui).to_numpy(), c[col].map(vi).to_numpy())),
                      shape=(len(uids), len(vocab)))
    return M, vocab


class ImplicitALS:
    """Implicit-feedback ALS (Hu, Koren, Volinsky 2008): confidence 1 + alpha * log1p(count). Item factors are
    fitted on the training users; transform folds any user in with the item factors fixed (no test person in fit)."""

    def __init__(self, dim: int = 32, reg: float = 0.1, alpha: float = 10.0, iters: int = 10, seed: int = 0):
        self.dim, self.reg, self.alpha, self.iters, self.seed = dim, reg, alpha, iters, seed

    def _solve(self, M, Y):
        YtY = Y.T @ Y
        k = Y.shape[1]
        out = np.zeros((M.shape[0], k))
        M = M.tocsr()
        for u in range(M.shape[0]):
            a, b = M.indptr[u], M.indptr[u + 1]
            idx, cnt = M.indices[a:b], M.data[a:b]
            if len(idx) == 0:
                continue
            c = self.alpha * np.log1p(cnt)
            Yi = Y[idx]
            A = YtY + (Yi.T * c) @ Yi + self.reg * np.eye(k)
            out[u] = np.linalg.solve(A, (Yi.T * (1 + c)).sum(1))
        return out

    def fit(self, M) -> "ImplicitALS":
        rng = np.random.default_rng(self.seed)
        n_u, n_i = M.shape
        X = rng.normal(0, 0.01, (n_u, self.dim))
        Y = rng.normal(0, 0.01, (n_i, self.dim))
        Mt = M.T.tocsr()
        for _ in range(self.iters):
            X = self._solve(M, Y)
            Y = self._solve(Mt, X)
        self.item_factors_ = Y
        return self

    def transform(self, M) -> np.ndarray:
        return self._solve(M, self.item_factors_).astype(np.float32)


class BagSVD:
    """Bag of categories: tf-idf of log1p counts, then TruncatedSVD (fitted on training users)."""

    def __init__(self, dim: int = 32, seed: int = 0):
        self.dim, self.seed = dim, seed

    def fit(self, M) -> "BagSVD":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfTransformer
        self.tfidf = TfidfTransformer().fit(M.log1p())
        self.svd = TruncatedSVD(min(self.dim, M.shape[1] - 1), random_state=self.seed)
        self.svd.fit(self.tfidf.transform(M.log1p()))
        return self

    def transform(self, M) -> np.ndarray:
        return self.svd.transform(self.tfidf.transform(M.log1p())).astype(np.float32)


class Standardizer:
    """z-score with training mean / std (std 0 -> 1)."""

    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)


# ------------------------------------------------------------------------------------------------- E2c sequence
def _day_numbers(x) -> np.ndarray:
    return pd.to_datetime(pd.Series(x)).to_numpy().astype("datetime64[D]").astype(np.int64)


class SequenceEncoder:
    """E2c: learned person vector from the time-ordered purchase line items (one token per item: category embedding
    + days before the cut, gap to the previous item, log price, log quantity, day-of-year sin/cos) -> GRU -> mean /
    max / last pooling -> `dim`-d vector u.

    Training is self-supervised on the persons passed to fit (the training persons only; guard.check_uids): each
    person's own history is cut again at kesim - k * horizon (k = 1..n_anchors), the items before that point are
    the input and the label is which of the label categories (the `targets` first, then the most frequent others
    among the fit persons) the person bought in the next `horizon` days, still before kesim. No target-window label
    of anyone is read. Early stopping on the within-person AUC of the `targets` labels over the validation persons'
    windows. transform cuts at each person's kesim (all items in `events` are already before it).

    events: uid, kesim, tarih, kategori (+ fiyat, adet when present; missing -> 0), one kesim per uid.

        enc = SequenceEncoder(dim=64, seed=0, targets=cats).fit(ev, fit_uids, val_uids, guard=guard)
        U = enc.transform(ev, uids)                                                   # (len(uids), 64)
    """
    PAD, UNK, BOS = 0, 1, 2
    N_NUM = 6

    def __init__(self, dim: int = 64, *, emb: int = 48, hidden: int = 128, max_len: int = 256, horizon: int = 181,
                 n_anchors: int = 4, targets: Sequence[str] | None = None, n_labels: int = 256,
                 target_weight: float = 1.0, min_count: int = 3, epochs: int = 30, patience: int = 4,
                 lr: float = 2e-3, weight_decay: float = 1e-4, batch_size: int = 128, dropout: float = 0.1,
                 seed: int = 0, device: str = "cpu", max_seconds: float | None = None):
        self.dim, self.emb, self.hidden, self.max_len, self.horizon = dim, emb, hidden, max_len, horizon
        self.n_anchors, self.targets, self.n_labels = n_anchors, list(targets or []), n_labels
        self.target_weight, self.min_count, self.epochs, self.patience = target_weight, min_count, epochs, patience
        self.lr, self.weight_decay, self.batch_size, self.dropout = lr, weight_decay, batch_size, dropout
        self.seed, self.device, self.max_seconds = seed, device, max_seconds

    # ---- data
    @staticmethod
    def _table(events: pd.DataFrame, uids) -> dict:
        """Per uid: (days, category, price, qty) sorted by date (stable) and the cut day."""
        ev = events[events["uid"].isin(set(uids))]
        n = len(ev)
        d = pd.DataFrame({"uid": ev["uid"].to_numpy(), "day": _day_numbers(ev["tarih"]),
                          "cut": _day_numbers(ev["kesim"]), "cat": ev["kategori"].astype(str).to_numpy(),
                          "price": ev["fiyat"].to_numpy(float) if "fiyat" in ev else np.zeros(n),
                          "qty": ev["adet"].to_numpy(float) if "adet" in ev else np.zeros(n)})
        d = d.sort_values(["uid", "day"], kind="mergesort")
        if (d.groupby("uid")["cut"].nunique() > 1).any():
            raise ValueError("SequenceEncoder: one kesim per uid expected (filter events to each person's cut)")
        out = {}
        for u, g in d.groupby("uid", sort=False):
            out[u] = (g["day"].to_numpy(), g["cat"].to_numpy(), np.nan_to_num(g["price"].to_numpy()),
                      np.nan_to_num(g["qty"].to_numpy()), int(g["cut"].iloc[0]))
        return out

    def _tokens(self, rec, anchor: int):
        """Token ids (BOS first) and numeric features of the last max_len - 1 items strictly before anchor."""
        day, cat, price, qty, _ = rec
        end = int(np.searchsorted(day, anchor, side="left"))
        a = max(0, end - (self.max_len - 1))
        dd, cc = day[a:end], cat[a:end]
        ids = np.array([self.BOS] + [self.vocab_.get(c, self.UNK) for c in cc], np.int64)
        num = np.zeros((len(ids), self.N_NUM), np.float32)
        if len(dd):
            gap = np.diff(dd, prepend=day[a - 1] if a > 0 else dd[0])
            doy = 2 * np.pi * (dd % 365.25) / 365.25                   # day 0 = 1970-01-01
            num[1:, 0] = np.log1p(np.maximum(anchor - dd, 0)) / 7
            num[1:, 1] = np.log1p(np.maximum(gap, 0)) / 7
            num[1:, 2] = np.log1p(np.maximum(price[a:end], 0)) / 5
            num[1:, 3] = np.log1p(np.maximum(qty[a:end], 0)) / 2
            num[1:, 4], num[1:, 5] = np.sin(doy), np.cos(doy)
        return ids, num

    def _batch(self, items):
        """Right-padded (cat (n, L), num (n, L, 6), lengths (n,)) as torch tensors."""
        import torch
        L = max(len(i) for i, _ in items)
        cat = np.zeros((len(items), L), np.int64)
        num = np.zeros((len(items), L, self.N_NUM), np.float32)
        for k, (i, x) in enumerate(items):
            cat[k, :len(i)] = i
            num[k, :len(i)] = x
        return (torch.as_tensor(cat), torch.as_tensor(num),
                torch.as_tensor(np.array([len(i) for i, _ in items], np.int64)))

    def _samples(self, tab: dict, uids):
        """Self-supervised windows: (tokens, label vector) per (uid, k) with at least one item before the anchor."""
        X, Y, who = [], [], []
        for u in uids:
            rec = tab.get(u)
            if rec is None:
                continue
            day, cat, cut = rec[0], rec[1], rec[4]
            for k in range(1, self.n_anchors + 1):
                anchor = cut - k * self.horizon
                if np.searchsorted(day, anchor, side="left") == 0:
                    continue
                m = (day >= anchor) & (day < anchor + self.horizon)
                y = np.zeros(len(self.labels_), np.float32)
                hit = [self.label_ix_[c] for c in set(cat[m].tolist()) if c in self.label_ix_]
                y[hit] = 1.0
                X.append(self._tokens(rec, anchor))
                Y.append(y)
                who.append(f"{u}|{k}")
        return X, (np.stack(Y) if Y else np.zeros((0, len(self.labels_)), np.float32)), who

    # ---- model
    def _net(self):
        import torch
        nn = torch.nn
        enc = self

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(len(enc.vocab_) + 3, enc.emb, padding_idx=0)
                self.inp = nn.Linear(enc.emb + enc.N_NUM, enc.hidden)
                self.gru = nn.GRU(enc.hidden, enc.hidden, batch_first=True)
                self.proj = nn.Linear(3 * enc.hidden, enc.dim)
                self.drop = nn.Dropout(enc.dropout)
                self.head = nn.Linear(enc.dim, len(enc.labels_))

            def encode(self, cat, num, lengths):
                from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
                x = torch.nn.functional.gelu(self.inp(torch.cat([self.emb(cat), num], -1)))
                out, h = self.gru(pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False))
                out, _ = pad_packed_sequence(out, batch_first=True, total_length=cat.shape[1])
                mask = (torch.arange(cat.shape[1])[None, :] < lengths[:, None]).to(out.device).unsqueeze(-1)
                mean = (out * mask).sum(1) / lengths.to(out.device)[:, None]
                mx = out.masked_fill(~mask, -1e4).max(1).values
                return self.proj(torch.cat([mean, mx, h[-1]], -1))

            def forward(self, cat, num, lengths):
                return self.head(self.drop(self.encode(cat, num, lengths)))

        return Net()

    def _logits(self, net, X, bs=512):
        import torch
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                c, n, l = self._batch(X[i:i + bs])
                out.append(net(c.to(self.device), n.to(self.device), l).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels_)), np.float32)

    def fit(self, events: pd.DataFrame, fit_uids, val_uids=(), *, guard=None) -> "SequenceEncoder":
        import time

        import torch
        from .select import kisi_auc
        t0 = time.time()
        fit_uids, val_uids = list(fit_uids), list(val_uids)
        if guard is not None:
            guard.check_uids(fit_uids + val_uids, "sq encoder fit")
        if set(fit_uids) & set(val_uids):
            raise ValueError("fit and validation persons overlap")
        tab = self._table(events, fit_uids + val_uids)
        fit_set = set(fit_uids)
        ev_fit = events[events["uid"].isin(fit_set)]
        cnt = ev_fit["kategori"].astype(str).value_counts()
        self.vocab_ = {c: i + 3 for i, c in enumerate(sorted(cnt[cnt >= self.min_count].index))}
        npers = ev_fit.drop_duplicates(["uid", "kategori"])["kategori"].astype(str).value_counts()
        others = [c for c in npers.index if c not in set(self.targets)]
        self.labels_ = (self.targets + others)[:max(self.n_labels, len(self.targets))]
        self.label_ix_ = {c: i for i, c in enumerate(self.labels_)}
        tcols = np.arange(len(self.targets))
        Xf, Yf, _ = self._samples(tab, fit_uids)
        Xv, Yv, wv = self._samples(tab, val_uids)
        if not Xf:
            raise ValueError("SequenceEncoder: no training window (no fit person has items before kesim - horizon)")
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        net = self._net().to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        Yf_t = torch.as_tensor(Yf)
        best, best_state, bad, hist = -np.inf, None, 0, []
        for ep in range(self.epochs):
            net.train()
            perm = torch.randperm(len(Xf), generator=g).numpy()
            tot = 0.0
            for i in range(0, len(perm), self.batch_size):
                b = perm[i:i + self.batch_size]
                c, n, l = self._batch([Xf[j] for j in b])
                y = Yf_t[b].to(self.device)
                z = net(c.to(self.device), n.to(self.device), l)
                loss = bce(z, y)
                if len(tcols) and self.target_weight:
                    loss = loss + self.target_weight * bce(z[:, tcols], y[:, tcols])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(b)
            rec = {"epoch": ep + 1, "train_loss": round(tot / len(perm), 5), "seconds": round(time.time() - t0, 1)}
            if Xv:
                zv = self._logits(net, Xv)
                cols = tcols if len(tcols) else np.arange(zv.shape[1])
                rec["val_loss"] = round(float(bce(torch.as_tensor(zv), torch.as_tensor(Yv))), 5)
                rec["val_kisi_auc"] = kisi_auc(zv[:, cols].ravel(), Yv[:, cols].ravel(),
                                               np.repeat(np.arange(len(Xv)), len(cols)))
                score = rec["val_kisi_auc"] if rec["val_kisi_auc"] == rec["val_kisi_auc"] else -rec["val_loss"]
            else:
                score = -rec["train_loss"]
            hist.append(rec)
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
            if bad >= self.patience or (self.max_seconds and time.time() - t0 > self.max_seconds):
                break
        net.load_state_dict(best_state)
        self.net_ = net.eval()
        best_ep = max(hist, key=lambda r: r.get("val_kisi_auc", -r.get("val_loss", r["train_loss"])))["epoch"] \
            if Xv else hist[-1]["epoch"]
        self.info_ = {"seconds": round(time.time() - t0, 1), "epochs_run": len(hist), "best_epoch": best_ep,
                      "best_val": float(best), "n_fit_windows": len(Xf), "n_val_windows": len(Xv),
                      "n_fit_persons": len(fit_uids), "n_val_persons": len(val_uids), "vocab": len(self.vocab_),
                      "labels": len(self.labels_), "hist": hist,
                      "config": {k: getattr(self, k) for k in ("dim", "emb", "hidden", "max_len", "horizon",
                                                               "n_anchors", "n_labels", "target_weight", "min_count",
                                                               "epochs", "patience", "lr", "weight_decay",
                                                               "batch_size", "dropout", "seed")}}
        return self

    def transform(self, events: pd.DataFrame, uids, bs: int = 512) -> np.ndarray:
        """(len(uids), dim): each person's items before their own kesim; a person without items gets the BOS-only
        sequence."""
        import torch
        tab = self._table(events, uids)
        empty = (np.zeros(0, np.int64), np.zeros(0, object), np.zeros(0), np.zeros(0), 0)
        X = [self._tokens(tab.get(u, empty), tab[u][4] if u in tab else 0) for u in uids]
        out = []
        self.net_.eval()
        with torch.no_grad():
            for i in range(0, len(X), bs):
                c, n, l = self._batch(X[i:i + bs])
                out.append(self.net_.encode(c.to(self.device), n.to(self.device), l).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)


# ------------------------------------------------------------------------------------------------- E3 / E4
def _cache_key(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def text_embedding(texts: Sequence[str], *, model: str = "intfloat/multilingual-e5-base", prefix: str = "query: ",
                   batch_size: int = 32, device: str | None = None, cache_dir=None) -> np.ndarray:
    """Normalised sentence embeddings (E3). e5 expects a 'query: ' / 'passage: ' prefix. Long texts are truncated
    at the model's max length (512 tokens for e5-base). Cached as <cache_dir>/e5_<hash>.npy when cache_dir is set."""
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"e5_{_cache_key(model, prefix, list(texts))}.npy"
        if path.exists():
            return np.load(path)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model, device=device)
    X = m.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                 show_progress_bar=False).astype(np.float32)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, X)
    return X


def llm_states(ld, prefixes: Sequence[str], layers: Sequence[int], *, pool="last",
               spans: Sequence[tuple[int, int]] | None = None, batch_size: int = 8, cache_dir=None,
               tag: str = "") -> dict:
    """E4: the model's residual state after each layer in `layers` at the prefix's last token (pool "last") or
    averaged over spans[i] (pool "mean"; without spans over all non-pad tokens): {layer: (N, d)}. A tuple of pools
    runs one forward pass for all of them and returns {pool: {layer: (N, d)}}. Cached per (model, precision, tag,
    layer, pool, texts hash) as .npy; spans enter the key of pool "mean" only."""
    from .readout import prefix_states
    pools = (pool,) if isinstance(pool, str) else tuple(pool)
    d = Path(cache_dir) / ld.code if cache_dir is not None else None
    path = {}
    for p in pools:
        key = _cache_key(ld.code, ld.precision, list(prefixes), p, spans if p == "mean" else None)
        for l in layers:
            path[(p, int(l))] = d / f"{tag}_{int(l)}_{p}_{ld.precision}_{key}.npy" if d is not None else None
    out = {p: {} for p in pools}
    todo = []
    for (p, l), f in path.items():
        if f is not None and f.exists():
            out[p][l] = np.load(f)
        else:
            todo.append((p, l))
    if todo:
        tp = tuple(sorted({p for p, _ in todo}))
        tl = sorted({l for _, l in todo})
        res = prefix_states(ld, prefixes, layers=tl, pool=tp, spans=spans, batch_size=batch_size).hidden
        for p, l in todo:
            out[p][l] = res[p][l]
            if path[(p, l)] is not None:
                d.mkdir(parents=True, exist_ok=True)
                np.save(path[(p, l)], res[p][l])
    return out[pool] if isinstance(pool, str) else out


# ------------------------------------------------------------------------------------------------- E5 / E6 / E7
def edit_profile(profile: str, label: str, value: str) -> str:
    """Set the '- <label>: ...' line of a profile text to `value` (ValueError if the line is missing)."""
    lines = profile.split("\n")
    hit = [i for i, l in enumerate(lines) if l.startswith(f"- {label}:")]
    if not hit:
        raise ValueError(f"no '- {label}:' line in the profile")
    lines[hit[0]] = f"- {label}: {value}"
    return "\n".join(lines)


def trait_directions(ld, profiles: Sequence[str], edits: Mapping[str, Mapping], prefix_fn: Callable[[str], str],
                     layer: int, *, n_people: int = 200, seed: int = 0, batch_size: int = 8) -> dict[str, np.ndarray]:
    """E5 contrastive directions: for n_people profiles, the trait line set to its highest vs lowest level, all else
    equal; direction = mean(state(high) - state(low)) at `layer`, last token of prefix_fn(profile)."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(profiles), min(n_people, len(profiles)), replace=False)
    out = {}
    for t, e in edits.items():
        lo = [prefix_fn(edit_profile(profiles[i], e["label"], e["levels"][0])) for i in pick]
        hi = [prefix_fn(edit_profile(profiles[i], e["label"], e["levels"][-1])) for i in pick]
        st = llm_states(ld, lo + hi, [layer], batch_size=batch_size)[layer]
        out[t] = (st[len(lo):] - st[:len(lo)]).mean(0).astype(np.float32)
    return out


def trait_values(survey: pd.DataFrame, edits: Mapping[str, Mapping] = TRAIT_EDITS_G1) -> np.ndarray:
    """Ordinal level index of each trait (NaN for 'Prefer not to say' / unknown), (n, n_traits)."""
    cols = []
    for t, e in edits.items():
        m = e.get("value_map") or {v: i for i, v in enumerate(e["levels"])}
        cols.append(survey[t].map(m).astype(float).to_numpy())
    return np.stack(cols, 1)


def trait_person_vector(values: np.ndarray, directions: Mapping[str, np.ndarray], *, ref: np.ndarray | None = None
                        ) -> np.ndarray:
    """sum_t z_t * d_t with z from `ref` (training values; default `values`), NaN -> 0. (n, d)."""
    ref = values if ref is None else ref
    mu, sd = np.nanmean(ref, 0), np.nanstd(ref, 0)
    sd[~(sd > 0)] = 1.0
    z = np.nan_to_num((values - mu) / sd)
    D = np.stack([directions[t] for t in directions], 0)
    if z.shape[1] != D.shape[0]:
        raise ValueError(f"{z.shape[1]} trait columns but {D.shape[0]} directions")
    return (z @ D).astype(np.float32)


def behaviour_directions(states: np.ndarray, y: np.ndarray, cats: np.ndarray, categories: Sequence[str] | None = None
                         ) -> tuple[list, np.ndarray]:
    """E6 mass-mean direction per category: mean state of buyers - mean of non-buyers. states (n_rows, d) aligned
    with y and cats (one row per person x category). Returns (categories, (n_cat, d))."""
    cats = np.asarray(cats)
    cs = list(categories) if categories is not None else sorted(pd.unique(cats).tolist())
    D = []
    for c in cs:
        m = cats == c
        pos, neg = states[m & (y == 1)], states[m & (y == 0)]
        D.append((pos.mean(0) - neg.mean(0)) if len(pos) and len(neg) else np.zeros(states.shape[1]))
    return cs, np.stack(D).astype(np.float32)


def load_sae(path) -> dict:
    """Load SAE encoder weights from a .safetensors / .pt / .npz file into numpy: W_enc (d, F), b_enc (F,), and
    optional b_dec (d,), threshold (F,) (JumpReLU), k (TopK). Key names are matched loosely (W_enc / encoder.weight).
    The Qwen-Scope file layout was not inspected; check the keys before use."""
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.numpy import load_file
        raw = load_file(str(p))
    elif p.suffix == ".npz":
        raw = dict(np.load(p))
    else:
        import torch
        raw = {k: v.float().numpy() for k, v in torch.load(p, map_location="cpu").items()}
    keys = {k.lower(): k for k in raw}

    def pick(*names):
        for n in names:
            if n.lower() in keys:
                return np.asarray(raw[keys[n.lower()]], dtype=np.float32)
        return None
    W = pick("W_enc", "encoder.weight", "w_enc")
    if W is None:
        raise KeyError(f"no encoder weight among {sorted(raw)[:10]}")
    sae = {"W_enc": W, "b_enc": pick("b_enc", "encoder.bias"), "b_dec": pick("b_dec", "decoder.bias"),
           "threshold": pick("threshold", "jump_threshold")}
    return sae


def sae_codes(states: np.ndarray, sae: Mapping, *, top_k: int | None = None, d_model: int | None = None):
    """E7: SAE feature activations of states (n, d) as scipy csr. ReLU (or JumpReLU with threshold); top_k keeps the
    k largest per row. W_enc may be stored (d, F) or (F, d); it is oriented by d_model / states.shape[1]."""
    import scipy.sparse as sp
    W = np.asarray(sae["W_enc"])
    d = d_model or states.shape[1]
    if W.shape[0] != d:
        W = W.T
    x = states - (sae["b_dec"] if sae.get("b_dec") is not None else 0.0)
    a = x @ W + (sae["b_enc"] if sae.get("b_enc") is not None else 0.0)
    if sae.get("threshold") is not None:
        a = np.where(a > sae["threshold"], a, 0.0)
    a = np.maximum(a, 0.0)
    if top_k:
        kth = np.partition(a, -top_k, axis=1)[:, -top_k][:, None]
        a = np.where(a >= kth, a, 0.0)
    return sp.csr_matrix(a.astype(np.float32))


__all__ = ["TRAIT_EDITS_G1", "TRAITS_G1", "SurveyOneHot", "SurveyAutoencoder", "history_features", "count_matrix",
           "ImplicitALS", "BagSVD", "SequenceEncoder", "Standardizer", "text_embedding", "llm_states", "edit_profile",
           "trait_directions", "trait_values", "trait_person_vector", "behaviour_directions", "load_sae", "sae_codes"]
