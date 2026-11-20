"""End-to-end integration tests for the complete offline PrivaRAG pipeline."""

from pathlib import Path

from chunker import RecursiveChunker
from dedup_index import IngestionManifest, sha256_text
from faithfulness import evaluate_faithfulness
from hybrid_retriever import HybridRRFRetriever
from pii_guard import redact_pii
from query_router import RouteTarget, route_query
from semantic_cache import VectorSemanticCache


def test_full_air_gapped_rag_pipeline_with_pii_dedup_cache_and_faithfulness(tmp_path: Path):
    raw_doc = (
        "Confidential Q4 Financial Brief. Contact CFO at cfo@acme.io or +90 532 111 22 33.\n\n"
        "Total Q4 revenue reached 18.6M EUR with 22% operating margin across EMEA.\n\n"
        "All SQLite audit snapshots must remain strictly read-only for 7 years."
    )

    # 1. Redact PII before indexing
    sanitized = redact_pii(raw_doc)
    assert "[REDACTED_EMAIL]" in sanitized.redacted_text
    assert "[REDACTED_PHONE]" in sanitized.redacted_text

    # 2. Check SHA-256 dedup manifest
    manifest = IngestionManifest(tmp_path / "manifest.json")
    digest = sha256_text(sanitized.redacted_text)
    assert manifest.needs_indexing("q4_brief.md", digest) is True

    # 3. Chunk & index into Hybrid RRF
    chunker = RecursiveChunker(chunk_size=140, chunk_overlap=25)
    chunks = chunker.chunk_text(sanitized.redacted_text, source="q4_brief.md")
    manifest.mark_indexed("q4_brief.md", digest, chunk_count=len(chunks))
    assert manifest.needs_indexing("q4_brief.md", digest) is False

    retriever = HybridRRFRetriever()
    retriever.add_documents(
        [c.chunk_id for c in chunks],
        [c.text for c in chunks],
        [{"source": c.source} for c in chunks],
    )

    # 4. Route query & retrieve
    question = "What was the Q4 revenue and operating margin?"
    assert route_query(question).target == RouteTarget.DOC_RAG
    hits = retriever.search(question, top_k=2)
    assert len(hits) >= 1
    assert "18.6M" in hits[0].text

    # 5. Verify answer faithfulness against retrieved chunks
    answer = "According to the brief [1], Q4 revenue reached 18.6M EUR with 22% operating margin."
    score = evaluate_faithfulness(answer, [h.text for h in hits])
    assert score.is_grounded is True

    # 6. Cache answer and verify invalidation when doc updates
    cache = VectorSemanticCache(similarity_threshold=0.88, persist_path=str(tmp_path / "cache.json"))
    cache.store(question, answer, [{"source": "q4_brief.md"}])
    hit = cache.lookup(question)
    assert hit is not None and hit["cache_hit"] is True

    assert cache.invalidate_source("q4_brief.md") == 1
    assert cache.lookup(question) is None
