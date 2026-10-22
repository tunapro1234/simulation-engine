"""SSR, semantic similarity rating (Maier et al. 2025, "LLMs reproduce human purchase intent via semantic
similarity elicitation of Likert ratings"). Port of sahne/kod/sahne/ssr.py (+ the pmf->probability mapping of
talp73/kod/llm/ssr.py).

The persona answers a Likert question in free text; the text embedding is compared with the 5 anchor sentences
of the scale (1 = most negative ... 5 = most positive) by cosine similarity:
    s_i = cos(e(answer), e(anchor_i));  p_i ∝ (s_i - min_j s_j + eps) ** (1 / T)
With several anchor sets the per-set distributions are averaged (as in the paper). Anchors are English; for
Turkish answers the runners also ask for a one-sentence English paraphrase and SSR scores that.

    s = SSR()                                         # needs sentence-transformers + a local model, lazily
    P = s.distribution(["I'd probably buy it if it went on sale."], "intent")   # (1, 5)
    SSR.expected(P)                                   # [~3.6]
    s = SSR(encoder=my_fn)                            # any texts -> (n, d) array; no model download
    s = SSR(max_chars=2000, cache_dir="emb-cache", progress=True)   # long texts: cut, cache on disk, report rate
    E = s.embed(texts, max_chars=800)                 # per-call override of the cut

The embedding model loads offline by default (HF_HUB_OFFLINE=1 unless already set); point SIMENGINE_HF_CACHE at
a local Hugging Face cache, or pass `cache_folder`. Either the HF_HOME folder or its `hub/` subfolder works (a
folder whose `hub/` holds the models is read through `hub/`). A model missing from the cache raises an OSError
that names SIMENGINE_HF_CACHE, `cache_folder`, HF_HOME and `offline=False`.

Long texts: each text is cut to `max_chars` characters (default 2000; None = no cut, the old behaviour) before
encoding. bge-base reads at most 512 tokens (all-MiniLM-L6-v2: 256) and English runs at roughly 4 characters per
token, so the default cut sits near what bge-base reads anyway: it loses little and gains little. Speed comes from
a lower cut, since encoding cost grows with tokens up to the model's limit. One run (22 Oct, bge-base, CPU, 4
threads, nice 19 on a shared, loaded machine, 96 synthetic English texts of ~10k characters): no cut 2.3 texts/s,
2000 chars 3.0, 1000 chars 6.4, 500 chars 13.7. `cache_dir=` keeps one .npy per text,
keyed by sha256(model name + text) and written atomically, so a rerun or a second process re-uses embeddings
(with `encoder=` the key is the encoder's qualified name, or `cache_key=`; a lambda or nested function needs
`cache_key=`, since two of them would share a name). `progress=True` prints one stderr line
per `progress_every` new texts with texts/s; a callable gets (done, total, texts_per_s) instead. Each text is
still embedded once per SSR object (in-memory cache).

Default model bge-base: on 10 hand-labelled intent sentences, expected score vs label Spearman 0.97
(all-MiniLM-L6-v2: 0.62; sahne branch, ssr.py comment and RAPOR.md "Mimari", 21 Oct).
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import threading
import time
from typing import Callable, Sequence, Union

import numpy as np

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE = "BAAI/bge-base-en-v1.5"
DEFAULT_MODEL = os.environ.get("SIMENGINE_SSR_MODEL", BGE)

# Each scale: one or more 5-anchor sets (1 -> 5). Sentences written for the sahne branch after the paper's idea.
SCALES: dict[str, list[list[str]]] = {
    "intent": [
        ["I would definitely not buy it.", "I probably would not buy it.", "I might or might not buy it.",
         "I would probably buy it.", "I would definitely buy it."],
        ["There is no chance I am purchasing this.", "It is unlikely that I would purchase this.",
         "I am unsure whether I would purchase this.", "It is likely that I would purchase this.",
         "I am certain I would purchase this."],
        ["Not interested at all, I'll pass.", "Not really for me, I doubt I'd get it.",
         "Maybe, I'd have to think about it.", "I'm interested and would likely get it.",
         "I want this, I'd get it right away."],
    ],
    "agreement": [
        ["I strongly disagree.", "I disagree.", "I neither agree nor disagree.", "I agree.", "I strongly agree."],
        ["That is completely wrong for me.", "That is mostly not true for me.", "That is partly true for me.",
         "That is mostly true for me.", "That is completely true for me."],
    ],
    "satisfaction": [
        ["I am very dissatisfied.", "I am somewhat dissatisfied.", "I feel neutral about it.",
         "I am somewhat satisfied.", "I am very satisfied."],
        ["It's terrible.", "It's disappointing.", "It's okay.", "It's good.", "It's excellent."],
    ],
    "likelihood": [
        ["Very unlikely.", "Unlikely.", "Neither likely nor unlikely.", "Likely.", "Very likely."],
        ["I definitely won't.", "I probably won't.", "Fifty-fifty.", "I probably will.", "I definitely will."],
    ],
    "liking": [
        ["I really dislike it.", "I don't like it much.", "It's neither good nor bad to me.",
         "I like it.", "I really love it."],
    ],
    # talp73 v1.7d: past purchase in a category (1 = surely did not buy ... 5 = surely bought)
    "purchased": [
        ["I definitely did not buy anything like this.", "I probably did not buy anything like this.",
         "I might or might not have bought something like this.", "I probably bought something like this.",
         "I definitely bought something like this."],
    ],
}
SCALE_ALIASES = {"niyet": "intent", "katilim": "agreement", "memnuniyet": "satisfaction",
                 "olasilik": "likelihood", "begeni": "liking"}


Progress = Union[bool, Callable[[int, int, float], None], None]


def _sentence_transformer(name: str, **kw):
    """Load a SentenceTransformer (optional dependency). A module function so tests can replace it."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, **kw)


def _hub_folder(folder: str) -> str:
    """HF_HOME-style folder (models under `hub/`) -> its `hub/`; a hub folder itself is returned unchanged."""
    hub = os.path.join(folder, "hub")
    try:
        if os.path.isdir(hub) and not any(n.startswith("models--") for n in os.listdir(folder)):
            return hub
    except OSError:
        pass
    return folder


def _missing_model_error(model: str, cache_folder: str | None, offline: bool, err: Exception) -> OSError:
    where = f"cache_folder={cache_folder!r}" if cache_folder else "no cache_folder (SIMENGINE_HF_CACHE unset)"
    return OSError(
        f"SSR: could not load embedding model {model!r} ({type(err).__name__}: {err}). Looked in {where}, "
        f"HF_HOME={os.environ.get('HF_HOME')!r}, HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')!r} "
        f"(offline={offline}). Point SIMENGINE_HF_CACHE (or SSR(cache_folder=...), or HF_HOME) at a local Hugging "
        f"Face cache that holds the model, or pass offline=False to allow a download, or encoder= to skip it.")


def anchor_sets(anchors) -> list[list[str]]:
    """Scale name, one 5-sentence list, or a list of such lists -> list of anchor sets."""
    if anchors is None:
        anchors = "agreement"
    if isinstance(anchors, str):
        anchors = SCALES[SCALE_ALIASES.get(anchors, anchors)]
    if anchors and isinstance(anchors[0], (list, tuple)):
        return [list(a) for a in anchors]
    return [list(anchors)]


class SSR:
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 1.0, eps: float = 0.0,
                 cache_folder: str | None = None, device: str = "cpu",
                 encoder: Callable[[list[str]], np.ndarray] | None = None, offline: bool = True,
                 max_chars: int | None = 2000, cache_dir: str | None = None, progress: Progress = False,
                 progress_every: int = 256, batch_size: int = 64, cache_key: str | None = None):
        self.model_name = model
        self.temperature = temperature
        self.eps = eps
        self.cache_folder = cache_folder or os.environ.get("SIMENGINE_HF_CACHE") or None
        self.device = device
        self.encoder = encoder
        self.offline = offline
        self.max_chars = max_chars
        self.cache_dir = cache_dir
        self.progress = progress
        self.progress_every = max(1, int(progress_every))
        self.batch_size = batch_size
        self.cache_key = cache_key
        if cache_dir and encoder is not None and not cache_key and "<" in self._model_key():
            raise ValueError("SSR: cache_dir with a lambda or nested encoder needs cache_key= (its qualified name "
                             f"{self._model_key()!r} is shared by every such function, so their embeddings would mix)")
        self._m = None
        self._lock = threading.Lock()
        self.cache: dict[str, np.ndarray] = {}

    def _load(self):
        if self.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        kw = {"device": self.device}
        cf = _hub_folder(self.cache_folder) if self.cache_folder else None
        if cf:
            kw["cache_folder"] = cf
        try:
            return _sentence_transformer(self.model_name, **kw)
        except ImportError:
            raise
        except (OSError, ValueError) as e:  # HF raises OSError / LocalEntryNotFoundError on an offline miss
            raise _missing_model_error(self.model_name, cf, self.offline, e) from e

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.encoder is not None:
            E = np.asarray(self.encoder(texts), dtype=np.float32)
            n = np.linalg.norm(E, axis=1, keepdims=True)
            return E / np.where(n > 0, n, 1)
        if self._m is None:
            self._m = self._load()
        return np.asarray(self._m.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                         show_progress_bar=False), dtype=np.float32)

    # --- disk cache: <cache_dir>/<sha256[:2]>/<sha256>.npy, key = sha256(model key + NUL + text) ---
    def _model_key(self) -> str:
        if self.cache_key:
            return self.cache_key
        if self.encoder is None:
            return self.model_name
        f = self.encoder
        return "encoder:" + getattr(f, "__module__", "") + "." + getattr(f, "__qualname__", type(f).__qualname__)

    def _disk_path(self, text: str) -> str:
        h = hashlib.sha256((self._model_key() + "\0" + text).encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, h[:2], h + ".npy")

    def _disk_get(self, text: str) -> np.ndarray | None:
        try:
            return np.load(self._disk_path(text))
        except (OSError, ValueError, EOFError):  # EOFError: an empty file
            return None

    def _disk_put(self, text: str, v: np.ndarray) -> None:
        path = self._disk_path(text)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.save(f, np.asarray(v, dtype=np.float32))
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _report(self, done: int, total: int, t0: float) -> None:
        rate = done / max(time.perf_counter() - t0, 1e-9)
        if callable(self.progress):
            self.progress(done, total, rate)
        elif self.progress:
            print(f"SSR embed: {done}/{total} texts, {rate:.1f} texts/s", file=sys.stderr, flush=True)

    def _cut(self, t, max_chars: int | None) -> str:
        t = str(t or "").strip() or "-"
        return t[:max_chars] if max_chars else t

    def embed(self, texts: Sequence[str], max_chars: int | None = -1) -> np.ndarray:
        """Normalised embeddings (n, d); a text is embedded once. Texts are cut to `max_chars` characters first
        (default: the constructor's; None = no cut). Misses are looked up in `cache_dir`, then encoded in chunks of
        `progress_every`, each chunk written to disk before the next one starts."""
        mc = self.max_chars if max_chars == -1 else max_chars
        texts = [self._cut(t, mc) for t in texts]
        with self._lock:
            new = [t for t in dict.fromkeys(texts) if t not in self.cache]
            if new and self.cache_dir:
                hit = {t: v for t in new if (v := self._disk_get(t)) is not None}
                self.cache.update(hit)
                new = [t for t in new if t not in hit]
            if new:
                t0, step = time.perf_counter(), self.progress_every
                for i in range(0, len(new), step):
                    chunk = new[i:i + step]
                    E = self._encode(chunk)
                    self.cache.update(zip(chunk, E))
                    if self.cache_dir:
                        for t, v in zip(chunk, E):
                            self._disk_put(t, v)
                    if self.progress:
                        self._report(i + len(chunk), len(new), t0)
            return np.stack([self.cache[t] for t in texts])

    def _one_set(self, E: np.ndarray, anchors: list[str], T: float) -> np.ndarray:
        S = E @ self.embed(anchors).T
        W = np.clip(S - S.min(axis=1, keepdims=True) + self.eps, 0, None) ** (1.0 / T)
        tot = W.sum(axis=1, keepdims=True)
        k = len(anchors)
        return np.where(tot > 0, W / np.where(tot > 0, tot, 1), 1.0 / k)

    def distribution(self, texts: Sequence[str], anchors="agreement", temperature: float | None = None
                     ) -> np.ndarray:
        """texts (n) x anchors (k) -> probability matrix (n, k). `anchors`: scale name, 5 sentences, or sets."""
        sets = anchor_sets(anchors)
        if not len(texts):
            return np.zeros((0, len(sets[0])))
        E = self.embed(list(texts))
        T = temperature or self.temperature
        return np.mean([self._one_set(E, s, T) for s in sets], axis=0)

    @staticmethod
    def expected(P: np.ndarray, base: int = 1) -> np.ndarray:
        """Distribution -> expected score (base .. base+k-1)."""
        P = np.atleast_2d(P)
        return P @ np.arange(base, base + P.shape[1])

    @staticmethod
    def to_probability(P: np.ndarray) -> np.ndarray:
        """Distribution over k ordered anchors -> value in [0, 1] (talp73: p = sum pmf_i (i-1)/(k-1))."""
        P = np.atleast_2d(P)
        return P @ np.linspace(0, 1, P.shape[1])


__all__ = ["SSR", "SCALES", "SCALE_ALIASES", "anchor_sets", "DEFAULT_MODEL", "BGE", "MINILM"]
