"""Tests for Hybrid Sparse + Dense Reciprocal Rank Fusion retriever."""

from hybrid_retriever import HybridRRFRetriever


def test_hybrid_rrf_retrieves_relevant_document():
    retriever = HybridRRFRetriever()
    retriever.add_documents(
        doc_ids=["sec-1", "hr-1", "fin-1"],
        texts=[
            "Zero-egress air-gapped deployment keeps all LLM weights and SQLite databases on-premise.",
            "Annual vacation allowance starts at 20 business days for full-time engineers.",
            "FY2025 gross margin improved to 64.2% following infrastructure optimization.",
        ],
    )

    hits = retriever.search("on-premise air-gapped SQLite deployment", top_k=2)
    assert len(hits) == 2
    assert hits[0].doc_id == "sec-1"
    assert hits[0].bm25_rank == 1
    assert hits[0].rrf_score > hits[1].rrf_score
