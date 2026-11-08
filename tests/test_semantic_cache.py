"""Unit tests for PrivaRAG Vector Semantic Cache & Latency Telemetry."""

import os
import tempfile
from semantic_cache import VectorSemanticCache, deterministic_ngram_embedding, _cosine_similarity


def test_deterministic_embedding_normalization():
    vec = deterministic_ngram_embedding("What is our Q3 net revenue?", dim=128)
    assert len(vec) == 128
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_exact_and_near_duplicate_cache_hit():
    cache = VectorSemanticCache(similarity_threshold=0.85)
    question = "What was the total revenue in Q3 2025?"
    sources = [{"source": "Finance_Report.pdf", "page": 12}]

    assert cache.lookup(question) is None
    cache.store(question, "Total revenue in Q3 2025 was $14.2M.", sources)

    # Exact hit
    hit = cache.lookup(question)
    assert hit is not None
    assert hit["cache_hit"] is True
    assert hit["similarity"] >= 0.99
    assert hit["answer"] == "Total revenue in Q3 2025 was $14.2M."
    assert hit["latency_ms"] < 50.0

    # Near-identical query (punctuation / casing variation)
    near_hit = cache.lookup("what was the total revenue in q3 2025")
    assert near_hit is not None
    assert near_hit["cache_hit"] is True

    # Unrelated query should miss
    miss = cache.lookup("How do I configure the PostgreSQL connection pool?")
    assert miss is None

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 2
    assert stats["entries"] == 1
    assert stats["estimated_saved_ms"] > 0


def test_cache_persistence_and_clear():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cache.json")
        cache1 = VectorSemanticCache(similarity_threshold=0.88, persist_path=path)
        cache1.store("Explain the air-gapped RAG security model", "Zero external API calls.", [])

        cache2 = VectorSemanticCache(similarity_threshold=0.88, persist_path=path)
        res = cache2.lookup("Explain the air-gapped RAG security model")
        assert res is not None
        assert res["answer"] == "Zero external API calls."

        cache2.clear()
        assert cache2.stats()["entries"] == 0
