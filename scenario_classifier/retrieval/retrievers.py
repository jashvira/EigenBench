"""Small retrieval helpers shared by scenario classifier audits."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?")
STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "criterion",
    "criteria",
    "even",
    "prefer",
    "prefers",
    "response",
    "responses",
}


def tokenize(text: str) -> list[str]:
    """Normalize text into BM25 terms."""
    return [
        token
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 1 and not token.isdigit() and token not in STOPWORDS
    ]


def criterion_text(doc: dict[str, Any]) -> str:
    return str(doc.get("claim") or doc.get("criterion_text") or doc.get("embedding_text") or "")


def normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


class BM25Retriever:
    """Lexical retriever over criterion documents."""

    def __init__(self, docs: list[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(str(doc.get("embedding_text", ""))) for doc in docs]
        self.doc_counts = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        doc_freq: Counter[str] = Counter()
        for counts in self.doc_counts:
            doc_freq.update(counts.keys())

        n_docs = len(self.docs)
        return {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def score(self, query: str) -> list[float]:
        query_terms = set(tokenize(query))
        scores: list[float] = []
        for counts, doc_len in zip(self.doc_counts, self.doc_lengths):
            score = 0.0
            length_norm = 1 - self.b + self.b * (doc_len / self.avgdl) if self.avgdl else 1.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * length_norm
                score += self.idf.get(term, 0.0) * (numerator / denominator)
            scores.append(score)
        return scores

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        scores = self.score(query)
        query_terms = set(tokenize(query))
        ranked = [
            item
            for item in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
            if item[1] > 0
        ][:top_k]

        rows = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            doc = self.docs[idx]
            rows.append(
                {
                    "rank": rank,
                    "score": round(score, 6),
                    "criterion_id": doc.get("criterion_id", ""),
                    "constitution": doc.get("constitution", ""),
                    "text": criterion_text(doc),
                    "matched_terms": sorted(query_terms & set(self.doc_counts[idx].keys())),
                }
            )
        return rows

class EmbeddingRetriever:
    """Cosine retriever over precomputed criterion vectors."""

    def __init__(self, docs: list[dict[str, Any]], vectors: np.ndarray) -> None:
        self.docs = docs
        self.vectors = normalize(vectors)

    def scores(self, query_vectors: np.ndarray) -> np.ndarray:
        return normalize(query_vectors) @ self.vectors.T

    def retrieve_many(self, query_vectors: np.ndarray, top_k: int) -> list[list[dict[str, Any]]]:
        return [self._rank(row, top_k) for row in self.scores(query_vectors)]

    def _rank(self, scores: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        rows = []
        for rank, idx in enumerate(np.argsort(-scores)[:top_k], start=1):
            doc = self.docs[int(idx)]
            rows.append(
                {
                    "rank": rank,
                    "score": round(float(scores[int(idx)]), 6),
                    "criterion_id": doc.get("criterion_id", ""),
                    "constitution": doc.get("constitution", ""),
                }
            )
        return rows
