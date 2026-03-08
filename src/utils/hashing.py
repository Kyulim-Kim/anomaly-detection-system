from __future__ import annotations

import hashlib
from pathlib import Path


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_sample_id(input_path: Path, sha1_hex: str) -> str:
    """
    Generate a filesystem-friendly sample_id stable across runs.
    """
    stem = input_path.stem.replace(" ", "_")
    return f"{stem}__{sha1_hex[:12]}"

