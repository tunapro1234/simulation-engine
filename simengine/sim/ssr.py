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

The embedding model loads offline by default (HF_HUB_OFFLINE=1 unless already set); point SIMENGINE_HF_CACHE at
a local Hugging Face cache, or pass `cache_folder`. Default model bge-base: on 10 hand-labelled intent
sentences, expected score vs label Spearman 0.97 (all-MiniLM-L6-v2: 0.62; sahne branch, ssr.py comment and
RAPOR.md "Mimari", 21 Oct).
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Sequence

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
                 encoder: Callable[[list[str]], np.ndarray] | None = None, offline: bool = True):
        self.model_name = model
        self.temperature = temperature
        self.eps = eps
        self.cache_folder = cache_folder or os.environ.get("SIMENGINE_HF_CACHE") or None
        self.device = device
        self.encoder = encoder
        self.offline = offline
        self._m = None
        self._lock = threading.Lock()
        self.cache: dict[str, np.ndarray] = {}

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.encoder is not None:
            E = np.asarray(self.encoder(texts), dtype=np.float32)
            n = np.linalg.norm(E, axis=1, keepdims=True)
            return E / np.where(n > 0, n, 1)
        if self._m is None:
            if self.offline:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer  # optional dependency
            kw = {"device": self.device}
            if self.cache_folder:
                kw["cache_folder"] = self.cache_folder
            self._m = SentenceTransformer(self.model_name, **kw)
        return np.asarray(self._m.encode(texts, batch_size=64, normalize_embeddings=True,
                                         show_progress_bar=False), dtype=np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Normalised embeddings (n, d); a text is embedded once."""
        texts = [str(t or "").strip() or "-" for t in texts]
        with self._lock:
            new = [t for t in dict.fromkeys(texts) if t not in self.cache]
            if new:
                self.cache.update(zip(new, self._encode(new)))
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
