"""Document store abstractions. Local folder implementation + S3/DB stubs for future extension."""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class Document:
    doc_id: str  # stable, e.g. sha1 of source_uri
    source_uri: str  # e.g. file://assets/knowledge/foo.md
    title: str
    text: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # doc_id + index
    doc_id: str
    source_uri: str
    title: str
    text: str
    start_char: int
    end_char: int
    meta: Dict[str, Any]


class DocumentStore(ABC):
    @abstractmethod
    def list_documents(self) -> List[Document]:
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        pass

    def iter_documents(self) -> Iterator[Document]:
        for doc in self.list_documents():
            yield doc


def _sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _first_heading(text: str) -> Optional[str]:
    """First line that looks like # Heading."""
    for line in text.splitlines():
        m = re.match(r"^#+\s+(.+)$", line.strip())
        if m:
            return m.group(1).strip()
    return None


# Priority order: first found wins. README.md is last fallback only (avoid false positives).
_REPO_MARKERS = (".git", "pyproject.toml", "poetry.lock", "requirements.txt", "README.md")


def find_repo_root(start_path: Path) -> Tuple[Optional[Path], str]:
    """
    Search upward from start_path for repo root. Returns (root_dir, marker_used).
    Marker is one of .git, pyproject.toml, poetry.lock, requirements.txt, README.md.
    README.md is used only as last fallback. Returns (None, "") if no marker found.
    """
    current = Path(start_path).resolve()
    if current.is_file():
        current = current.parent
    while True:
        for name in _REPO_MARKERS:
            p = current / name
            if p.exists():
                return current, name
        parent = current.parent
        if parent == current:
            return None, ""
        current = parent


class LocalFolderStore(DocumentStore):
    """
    Load .md files from a root directory. source_uri is always repo-root relative:
    file://assets/knowledge/foo.md (no OS absolute paths).
    """

    def __init__(self, root_dir: Path, glob: str = "**/*.md") -> None:
        self._root = Path(root_dir).resolve()
        self._glob = glob
        self._repo_root, self._repo_root_marker = find_repo_root(self._root)
        if self._repo_root is None:
            self._repo_root = self._root
            self._repo_root_marker = ""

    def get_repo_root_marker(self) -> str:
        """Marker used to determine repo root (e.g. .git, README.md). Empty if repo not found."""
        return getattr(self, "_repo_root_marker", "")

    def list_documents(self) -> List[Document]:
        return list(self.iter_documents())

    def get_document(self, doc_id: str) -> Optional[Document]:
        for doc in self.iter_documents():
            if doc.doc_id == doc_id:
                return doc
        return None

    def iter_documents(self) -> Iterator[Document]:
        if not self._root.exists():
            return
        for path in sorted(self._root.glob(self._glob)):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            try:
                rel = path.relative_to(self._repo_root)
            except ValueError:
                rel = path.relative_to(self._root)
            source_uri = f"file://{rel.as_posix()}"
            doc_id = _sha1_str(source_uri)
            title = _first_heading(text) or path.stem
            yield Document(
                doc_id=doc_id,
                source_uri=source_uri,
                title=title,
                text=text,
                meta={"path": rel.as_posix()},
            )


class S3Store(DocumentStore):
    """Stub for future S3-backed document store. Methods raise NotImplementedError."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def list_documents(self) -> List[Document]:
        raise NotImplementedError(
            "S3Store is not implemented yet. Use LocalFolderStore or extend this class for S3."
        )

    def get_document(self, doc_id: str) -> Optional[Document]:
        raise NotImplementedError(
            "S3Store is not implemented yet. Use LocalFolderStore or extend this class for S3."
        )


class DBStore(DocumentStore):
    """Stub for future DB-backed document store. Methods raise NotImplementedError."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def list_documents(self) -> List[Document]:
        raise NotImplementedError(
            "DBStore is not implemented yet. Use LocalFolderStore or extend this class for DB."
        )

    def get_document(self, doc_id: str) -> Optional[Document]:
        raise NotImplementedError(
            "DBStore is not implemented yet. Use LocalFolderStore or extend this class for DB."
        )
