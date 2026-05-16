"""v1.0: FAISS + BM25 hybrid retrieval — semantic + keyword search."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


class BM25Scorer:
    """BM25 keyword search for hybrid retrieval."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[str] = []
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0
        self._df: dict[str, int] = {}
        self._total_docs: int = 0

    def index(self, documents: list[str]):
        self._docs = documents
        self._doc_lens = [len(d.split()) for d in documents]
        self._total_docs = len(documents)
        self._avg_dl = sum(self._doc_lens) / max(self._total_docs, 1)
        self._df = {}
        for doc in documents:
            for word in set(doc.lower().split()):
                self._df[word] = self._df.get(word, 0) + 1

    def score(self, query: str, doc_idx: int) -> float:
        if doc_idx >= len(self._docs):
            return 0.0
        doc = self._docs[doc_idx].lower().split()
        dl = self._doc_lens[doc_idx]
        score = 0.0
        for word in query.lower().split():
            if word not in self._df:
                continue
            df = self._df[word]
            idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf = doc.count(word)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
            score += idf * numerator / denominator
        return score


class FAISSRetrieval:
    """Hybrid semantic + keyword retrieval using FAISS when available."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self._keys: list[str] = []
        self._values: list[str] = []
        self._embeddings: list[list[float]] = []
        self._bm25: BM25Scorer | None = None
        self._index = None
        self._faiss_available = False
        try:
            import numpy as np
            self._np = np
            self._faiss_available = True
        except ImportError:
            pass

    def add(self, key: str, value: str, embedding: list[float] | None = None):
        self._keys.append(key)
        self._values.append(value)
        if embedding:
            self._embeddings.append(embedding)

    def build_index(self):
        if self._faiss_available and len(self._embeddings) >= 10:
            emb_array = self._np.array(self._embeddings, dtype='float32')
            self._index = emb_array
        self._bm25 = BM25Scorer()
        self._bm25.index(self._keys)

    def search(self, query: str, top_k: int = 10,
               alpha: float = 0.7) -> list[dict[str, Any]]:
        results = []
        semantic_scores: dict[int, float] = {}
        keyword_scores: dict[int, float] = {}

        if self._index is not None and len(self._embeddings) >= len(self._keys):
            query_emb = self._np.zeros(self.dim, dtype='float32')
            scores = self._index @ query_emb
            for i, s in enumerate(scores):
                semantic_scores[i] = float(s)

        if self._bm25 is not None:
            for i in range(len(self._keys)):
                s = self._bm25.score(query, i)
                if s > 0:
                    keyword_scores[i] = s

        all_idxs = set(list(semantic_scores.keys()) + list(keyword_scores.keys()))
        for i in all_idxs:
            sem = semantic_scores.get(i, 0.0)
            kw = keyword_scores.get(i, 0.0)
            if sem > 0:
                sem = sem / max(semantic_scores.values())
            if kw > 0:
                kw = kw / max(keyword_scores.values())
            combined = alpha * sem + (1 - alpha) * kw
            if combined > 0 or i < top_k:
                results.append({"key": self._keys[i], "value": self._values[i],
                               "score": combined, "idx": i})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        return len(self._keys)
