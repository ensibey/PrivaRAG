"""
Zero-LLM citation grounding and faithfulness verifier for PrivaRAG answers.

Catches hallucinated numbers, unsupported claims, and invalid citation indices
by comparing generated answers directly against retrieved context chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "can", "will", "into", "over", "under",
    "bir", "ve", "ile", "için", "olan", "olarak", "daha", "gibi", "kadar", "bu",
}

NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
TOKEN_RE = re.compile(r"[0-9a-zA-ZçğıöşüÇĞİÖŞÜ]{3,}")
CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class FaithfulnessScore:
    token_grounding: float
    numeric_grounding: float
    citations_valid: bool
    unsupported_numbers: List[str]
    is_grounded: bool


def evaluate_faithfulness(
    answer: str,
    context_chunks: Sequence[str],
    min_token_grounding: float = 0.45,
) -> FaithfulnessScore:
    """Evaluate whether a RAG answer is lexically and numerically grounded in retrieved context."""
    joined_context = " ".join(context_chunks).lower()
    context_tokens = {t for t in TOKEN_RE.findall(joined_context) if t not in STOPWORDS}
    context_numbers = set(NUMBER_RE.findall(joined_context))

    ans_lower = (answer or "").lower()
    ans_tokens = [t for t in TOKEN_RE.findall(ans_lower) if t not in STOPWORDS]
    ans_numbers = NUMBER_RE.findall(ans_lower)

    if not ans_tokens:
        return FaithfulnessScore(0.0, 1.0, True, [], False)

    supported_tokens = sum(1 for t in ans_tokens if t in context_tokens)
    token_ratio = supported_tokens / len(ans_tokens)

    unsupported_nums = [n for n in ans_numbers if n not in context_numbers]
    num_ratio = (
        (len(ans_numbers) - len(unsupported_nums)) / len(ans_numbers)
        if ans_numbers
        else 1.0
    )

    cited_indices = [int(m.group(1)) for m in CITATION_RE.finditer(answer or "")]
    citations_valid = all(1 <= idx <= len(context_chunks) for idx in cited_indices)

    is_grounded = (
        token_ratio >= min_token_grounding
        and len(unsupported_nums) == 0
        and citations_valid
    )

    return FaithfulnessScore(
        token_grounding=round(token_ratio, 4),
        numeric_grounding=round(num_ratio, 4),
        citations_valid=citations_valid,
        unsupported_numbers=unsupported_nums,
        is_grounded=is_grounded,
    )
