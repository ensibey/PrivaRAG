"""Tests for RAG answer citation grounding and faithfulness verifier."""

from faithfulness import evaluate_faithfulness


def test_grounded_answer_passes_with_valid_numbers_and_citation():
    chunks = [
        "FY2025 Q4 revenue reached 12.4M EUR with an EBITDA margin of 18%.",
        "Incident response SLA requires P1 triage within 15 minutes.",
    ]
    answer = "According to the financial report [1], Q4 revenue reached 12.4M EUR with 18% EBITDA margin."
    score = evaluate_faithfulness(answer, chunks)

    assert score.is_grounded is True
    assert score.numeric_grounding == 1.0
    assert score.citations_valid is True
    assert score.unsupported_numbers == []


def test_detects_hallucinated_number_and_out_of_range_citation():
    chunks = ["Q4 revenue reached 12.4M EUR with 18% EBITDA margin."]
    answer = "Q4 revenue reached 99.9M EUR according to source [3]."
    score = evaluate_faithfulness(answer, chunks)

    assert score.is_grounded is False
    assert "99.9" in score.unsupported_numbers
    assert score.citations_valid is False
