"""Tests for SHA-256 incremental ingestion deduplication manifest."""

from pathlib import Path

from dedup_index import IngestionManifest, sha256_file, sha256_text


def test_manifest_detects_unchanged_and_modified_docs(tmp_path: Path):
    manifest_file = tmp_path / "manifest.json"
    manifest = IngestionManifest(manifest_file)

    doc_file = tmp_path / "policy.txt"
    doc_file.write_text("Initial policy v1", encoding="utf-8")
    h1 = sha256_file(doc_file)

    assert manifest.needs_indexing("policy.txt", h1) is True
    manifest.mark_indexed("policy.txt", h1, chunk_count=3)
    assert manifest.needs_indexing("policy.txt", h1) is False

    # Reload from disk
    reloaded = IngestionManifest(manifest_file)
    assert reloaded.count() == 1
    assert reloaded.needs_indexing("policy.txt", h1) is False

    # Modify file -> must re-index
    doc_file.write_text("Updated policy v2", encoding="utf-8")
    h2 = sha256_file(doc_file)
    assert h1 != h2
    assert reloaded.needs_indexing("policy.txt", h2) is True
    assert reloaded.remove("policy.txt") is True
    assert reloaded.count() == 0


def test_sha256_text_is_deterministic():
    assert sha256_text("hello") == sha256_text("hello")
    assert sha256_text("hello") != sha256_text("world")
