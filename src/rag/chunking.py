"""Simple character-based chunking (no tokenizer dependency)."""
from __future__ import annotations

from typing import List, Tuple

from src.rag.store import Chunk, Document


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Tuple[int, int, str]]:
    """
    Split text into overlapping segments. Returns list of (start_char, end_char, chunk_text).
    """
    if not text or chunk_size <= 0:
        return []
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)
    step = chunk_size - chunk_overlap
    out: List[Tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        segment = text[start:end]
        out.append((start, end, segment))
        if end >= len(text):
            break
        start = start + step
    return out


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Chunk]:
    """Turn documents into chunks with stable chunk_id = doc_id + '_' + index."""
    chunks: List[Chunk] = []
    for doc in docs:
        segments = chunk_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, (start_char, end_char, segment) in enumerate(segments):
            chunk_id = f"{doc.doc_id}_{i}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source_uri=doc.source_uri,
                    title=doc.title,
                    text=segment,
                    start_char=start_char,
                    end_char=end_char,
                    meta=dict(doc.meta),
                )
            )
    return chunks
