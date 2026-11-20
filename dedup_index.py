"""
SHA-256 content-hash deduplication manifest for incremental document ingestion.

Prevents redundant embedding and vector-store re-indexing when documents have not changed.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IngestionManifest:
    """Tracks indexed document paths and their latest SHA-256 digests on disk."""

    def __init__(self, manifest_path: Optional[Path] = None) -> None:
        self.manifest_path = manifest_path
        self._records: Dict[str, Dict[str, object]] = {}
        if self.manifest_path and self.manifest_path.exists():
            self._load()

    def needs_indexing(self, doc_id: str, content_hash: str) -> bool:
        rec = self._records.get(doc_id)
        if not rec:
            return True
        return rec.get("sha256") != content_hash

    def mark_indexed(self, doc_id: str, content_hash: str, chunk_count: int = 0) -> None:
        self._records[doc_id] = {
            "sha256": content_hash,
            "chunks": chunk_count,
            "indexed_at": time.time(),
        }
        self._save()

    def remove(self, doc_id: str) -> bool:
        existed = doc_id in self._records
        self._records.pop(doc_id, None)
        if existed:
            self._save()
        return existed

    def count(self) -> int:
        return len(self._records)

    def _save(self) -> None:
        if not self.manifest_path:
            return
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self._records, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.manifest_path:
            return
        try:
            self._records = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._records = {}
