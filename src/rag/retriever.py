"""Retriever abstractions. MVP: KeywordRetriever. Stub: VectorRetriever for future embeddings+FAISS."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.rag.store import Chunk


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric."""
    return [t.lower() for t in re.split(r"[^a-zA-Z0-9]+", text) if t]


def _tf_like_score(query_tokens: List[str], chunk_tokens: List[str]) -> float:
    """Simple overlap score: sum of (count in chunk) per query token, normalized by chunk length."""
    if not chunk_tokens:
        return 0.0
    chunk_counts: Dict[str, int] = {}
    for t in chunk_tokens:
        chunk_counts[t] = chunk_counts.get(t, 0) + 1
    score = 0.0
    for q in query_tokens:
        if q in chunk_counts:
            score += chunk_counts[q]
    return score / max(1, len(chunk_tokens))


class Retriever(ABC):
    @abstractmethod
    def build_index(self, chunks: List[Chunk]) -> None:
        pass

    @abstractmethod
    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Returns list of context dicts (chunk_id, doc_id, source_uri, title, snippet, text, retrieval_score)."""
        pass


class KeywordRetriever(Retriever):
    """MVP: token overlap / TF-like scoring. No embeddings."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._chunk_tokens: List[List[str]] = []

    def build_index(self, chunks: List[Chunk]) -> None:
        self._chunks = list(chunks)
        self._chunk_tokens = [_tokenize(c.text) for c in self._chunks]

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._chunks:
            return []
        query_tokens = _tokenize(text)
        if not query_tokens:
            return []
        scored: List[tuple[float, Chunk]] = []
        for i, ch in enumerate(self._chunks):
            sc = _tf_like_score(query_tokens, self._chunk_tokens[i])
            if sc > 0:
                scored.append((sc, ch))
        scored.sort(key=lambda x: -x[0])
        top = scored[:top_k]
        out: List[Dict[str, Any]] = []
        for sc, ch in top:
            snippet = ch.text[:200] + ("..." if len(ch.text) > 200 else "")
            out.append({
                "chunk_id": ch.chunk_id,
                "doc_id": ch.doc_id,
                "source_uri": ch.source_uri,
                "title": ch.title,
                "snippet": snippet,
                "text": ch.text,
                "retrieval_score": round(sc, 6),
            })
        return out


class VectorRetriever(Retriever):
    """
    Stub for future: embeddings + FAISS (or similar).
    Methods raise NotImplementedError.
    """

    def build_index(self, chunks: List[Chunk]) -> None:
        raise NotImplementedError(
            "VectorRetriever is not implemented. Future: use embeddings + FAISS (or similar) for semantic search."
        )

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "VectorRetriever is not implemented. Future: embed query and nearest-neighbor search."
        )
