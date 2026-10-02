from __future__ import annotations
import hashlib
import os


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str | None:
    """sha256 hex digest of a file, or None if it does not exist."""
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
