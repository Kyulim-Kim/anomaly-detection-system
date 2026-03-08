"""RAG pipeline: build/load index from store, retrieve contexts. Cache = JSONL with meta line for invalidation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.chunking import chunk_documents
from src.rag.retriever import Retriever
from src.rag.store import Chunk, Document, DocumentStore


def compute_docs_fingerprint(docs: List[Document]) -> str:
    """Single sha1 from (source_uri, len(text), sha1(text[:5000])) per doc, sorted by source_uri. Used for cache invalidation."""
    parts = []
    for d in sorted(docs, key=lambda x: x.source_uri):
        head = d.text[:5000] if len(d.text) > 5000 else d.text
        part = f"{d.source_uri}\t{len(d.text)}\t{hashlib.sha1(head.encode('utf-8')).hexdigest()}"
        parts.append(part)
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


class RAGEngine:
    """
    Build or load chunk index from a document store, then retrieve with a retriever.
    Cache format: one JSON object per line (chunk_id, doc_id, source_uri, title, text, start_char, end_char, meta).
    """

    def __init__(
        self,
        store: DocumentStore,
        retriever: Retriever,
        cache_path: Path = Path("artifacts/knowledge_index.jsonl"),
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._cache_path = Path(cache_path)

    def build_or_load_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Load docs from store, check cache fingerprint; rebuild if missing/mismatch.
        Returns dict with cache_invalidated (bool), docs_count (int), chunks_count (int).
        """
        docs = self._store.list_documents()
        current_fp = compute_docs_fingerprint(docs)
        chunks = chunk_documents(docs)
        out: Dict[str, Any] = {"cache_invalidated": False, "docs_count": len(docs), "chunks_count": len(chunks)}

        if force_rebuild or not self._cache_path.exists():
            self._build_and_save_index(docs, chunks, current_fp)
            if not force_rebuild:
                out["cache_invalidated"] = False
            else:
                out["cache_invalidated"] = True
            return out

        # Cache exists: read first line (meta)
        try:
            with self._cache_path.open("r", encoding="utf-8") as f:
                first_line = f.readline().strip()
        except Exception:
            self._build_and_save_index(docs, chunks, current_fp)
            out["cache_invalidated"] = True
            return out

        if not first_line:
            self._build_and_save_index(docs, chunks, current_fp)
            out["cache_invalidated"] = True
            return out

        try:
            meta = json.loads(first_line)
            if meta.get("_type") != "meta" or meta.get("docs_fingerprint") != current_fp:
                self._build_and_save_index(docs, chunks, current_fp)
                out["cache_invalidated"] = True
                return out
            out["docs_count"] = meta.get("docs_count", len(docs))
            out["chunks_count"] = meta.get("chunks_count", len(chunks))
        except (json.JSONDecodeError, TypeError):
            self._build_and_save_index(docs, chunks, current_fp)
            out["cache_invalidated"] = True
            return out

        self._load_index_from_cache()
        return out

    def _load_index_from_cache(self) -> None:
        chunks_list: List[Chunk] = []
        with self._cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("_type") == "meta":
                    continue
                chunks_list.append(
                    Chunk(
                        chunk_id=obj["chunk_id"],
                        doc_id=obj["doc_id"],
                        source_uri=obj["source_uri"],
                        title=obj["title"],
                        text=obj["text"],
                        start_char=obj["start_char"],
                        end_char=obj["end_char"],
                        meta=obj.get("meta", {}),
                    )
                )
        self._retriever.build_index(chunks_list)

    def _build_and_save_index(
        self,
        docs: List[Document],
        chunks: List[Chunk],
        docs_fingerprint: str,
    ) -> None:
        from src.utils.time import utc_now_iso

        self._retriever.build_index(chunks)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "_type": "meta",
            "docs_fingerprint": docs_fingerprint,
            "created_at": utc_now_iso(),
            "docs_count": len(docs),
            "chunks_count": len(chunks),
        }
        with self._cache_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for c in chunks:
                obj = {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "source_uri": c.source_uri,
                    "title": c.title,
                    "text": c.text,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "meta": c.meta,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_chars_result: int = 1500,
        max_chars_llm: int = 3500,
    ) -> Dict[str, Any]:
        """
        Returns dict with:
          context_used: bool
          contexts_for_llm: list for LLM input (text up to max_chars_llm)
          contexts_for_result: list for result.json (text up to max_chars_result)
          notes: str
        """
        raw = self._retriever.query(query, top_k=top_k)
        if not raw:
            return {
                "context_used": False,
                "contexts_for_llm": [],
                "contexts_for_result": [],
                "notes": "No contexts retrieved.",
            }

        def trim(ctx: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
            d = dict(ctx)
            text = d.get("text", "")
            if len(text) > max_chars:
                d["text"] = text[:max_chars] + "..."
            snippet = d.get("snippet", text)
            if len(snippet) > 200:
                d["snippet"] = snippet[:200] + "..."
            return d

        contexts_for_llm = [trim(c, max_chars_llm) for c in raw]
        contexts_for_result = [trim(c, max_chars_result) for c in raw]
        return {
            "context_used": True,
            "contexts_for_llm": contexts_for_llm,
            "contexts_for_result": contexts_for_result,
            "notes": f"Retrieved {len(raw)} chunks for query.",
        }
