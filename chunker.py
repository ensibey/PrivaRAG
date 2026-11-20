"""
Structure-aware recursive document chunker for PrivaRAG.

Splits long documents along natural boundaries (headings, paragraphs, sentences, words)
while preserving configurable token/character overlap and source metadata for citations.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentChunk:
    chunk_id: str
    source: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecursiveChunker:
    """Split text into overlapping chunks respecting paragraph and sentence boundaries."""

    SEPARATORS = ("\n\n", "\n", ". ", " ")

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_recursive(self, text: str, sep_idx: int = 0) -> List[str]:
        if len(text) <= self.chunk_size or sep_idx >= len(self.SEPARATORS):
            if len(text) <= self.chunk_size:
                return [text]
            # Hard character slice fallback
            step = self.chunk_size - self.chunk_overlap
            return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

        sep = self.SEPARATORS[sep_idx]
        parts = text.split(sep)
        if len(parts) == 1:
            return self._split_recursive(text, sep_idx + 1)

        merged: List[str] = []
        current = ""
        for part in parts:
            piece = part if not current else f"{current}{sep}{part}"
            if len(piece) <= self.chunk_size:
                current = piece
            else:
                if current:
                    merged.append(current)
                if len(part) > self.chunk_size:
                    merged.extend(self._split_recursive(part, sep_idx + 1))
                    current = ""
                else:
                    current = part
        if current:
            merged.append(current)
        return merged

    def chunk_text(
        self,
        text: str,
        source: str = "document.txt",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        cleaned = re.sub(r"\r\n?", "\n", text or "").strip()
        if not cleaned:
            return []

        base_segments = self._split_recursive(cleaned)
        chunks: List[DocumentChunk] = []
        cursor = 0

        for idx, seg in enumerate(base_segments):
            content = seg.strip()
            if not content:
                continue
            if idx > 0 and self.chunk_overlap > 0 and chunks:
                tail = chunks[-1].text[-self.chunk_overlap :].strip()
                if tail and not content.startswith(tail):
                    combined = f"{tail} {content}"
                    if len(combined) <= self.chunk_size + self.chunk_overlap:
                        content = combined

            pos = cleaned.find(seg.strip(), max(0, cursor - self.chunk_overlap))
            start_char = pos if pos >= 0 else cursor
            end_char = start_char + len(seg.strip())
            cursor = end_char

            digest = hashlib.sha256(f"{source}:{idx}:{content}".encode("utf-8")).hexdigest()[:12]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{source}#{idx}-{digest}",
                    source=source,
                    text=content,
                    chunk_index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=dict(metadata or {}),
                )
            )

        return chunks
