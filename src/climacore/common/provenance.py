"""
Provenance utilities for computing SHA-256 checksums, verifying file integrity,
and capturing runtime/dataset metadata.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, Any


def compute_sha256(file_path: Path | str) -> str:
    """Compute the SHA-256 checksum of a file in streaming chunks."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_file_metadata(file_path: Path | str) -> Dict[str, Any]:
    """Return file size and SHA-256 checksum."""
    path = Path(file_path)
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": compute_sha256(path),
    }
