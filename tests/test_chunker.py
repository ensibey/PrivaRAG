"""Tests for structure-aware recursive document chunking."""

import pytest

from chunker import RecursiveChunker


def test_chunker_splits_paragraphs_with_overlap():
    text = (
        "Paragraph one explains air-gapped enterprise RAG security.\n\n"
        "Paragraph two describes BM25 sparse lexical matching and dense embeddings.\n\n"
        "Paragraph three details SQLite read-only safety validation."
    )
    chunker = RecursiveChunker(chunk_size=90, chunk_overlap=20)
    chunks = chunker.chunk_text(text, source="architecture.md", metadata={"dept": "security"})

    assert len(chunks) >= 3
    assert all(c.source == "architecture.md" for c in chunks)
    assert all(c.metadata["dept"] == "security" for c in chunks)
    # Overlap carries context from previous chunk
    assert chunks[0].text[-15:] in chunks[1].text


def test_chunker_handles_empty_and_invalid_params():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.chunk_text("   \n\n  ") == []

    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=50, chunk_overlap=50)
