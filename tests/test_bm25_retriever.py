"""Tests for Okapi BM25 sparse lexical retriever."""

from bm25_retriever import OkapiBM25Retriever, tokenize_lexical


def test_bm25_ranks_exact_lexical_match_first():
    bm25 = OkapiBM25Retriever()
    bm25.add_documents(
        doc_ids=["d1", "d2", "d3"],
        texts=[
            "Q4 revenue reached 12.4M EUR with 18% EBITDA margin.",
            "Employee remote work policy allows 3 days per week from home.",
            "Incident response SLA requires P1 triage within 15 minutes.",
        ],
    )

    hits = bm25.search("What was the Q4 EBITDA margin?", top_k=2)
    assert len(hits) >= 1
    assert hits[0].doc_id == "d1"
    assert hits[0].score > 0


def test_bm25_supports_turkish_unicode_tokens():
    tokens = tokenize_lexical("İstanbul Şube Güvenlik Politikası")
    assert "istanbul" in tokens
    assert "şube" in tokens
    assert "güvenlik" in tokens
