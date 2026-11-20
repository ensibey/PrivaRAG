"""
Air-gapped PII redaction guard for PrivaRAG.

Masks personally identifiable information (emails, phone numbers, IBANs,
checksum-verified Turkish National IDs, and credit card numbers) before
prompt assembly or telemetry logging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\w)")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[\s-]?[0-9A-Z]{4}){3,6}(?:[\s-]?[0-9A-Z]{1,4})?\b")
TCKN_RE = re.compile(r"\b[1-9]\d{10}\b")
CARD_RE = re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b")


def is_valid_tckn(candidate: str) -> bool:
    """Validate 11-digit Turkish National ID checksum to avoid false positives on order IDs."""
    if len(candidate) != 11 or not candidate.isdigit() or candidate[0] == "0":
        return False
    digits = [int(ch) for ch in candidate]
    odd_sum = sum(digits[0:9:2])
    even_sum = sum(digits[1:8:2])
    d10 = ((odd_sum * 7) - even_sum) % 10
    d11 = sum(digits[:10]) % 10
    return digits[9] == d10 and digits[10] == d11


@dataclass
class RedactionReport:
    redacted_text: str
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def total_redacted(self) -> int:
        return sum(self.counts.values())


def redact_pii(text: str) -> RedactionReport:
    """Detect and mask PII entities in text, returning sanitized text and entity counts."""
    counts: Dict[str, int] = {}
    out = text

    def _replace_with(pattern: re.Pattern, tag: str) -> None:
        nonlocal out
        matches = list(pattern.finditer(out))
        if matches:
            counts[tag] = counts.get(tag, 0) + len(matches)
            out = pattern.sub(f"[REDACTED_{tag}]", out)

    _replace_with(EMAIL_RE, "EMAIL")
    _replace_with(IBAN_RE, "IBAN")
    _replace_with(CARD_RE, "CARD")
    _replace_with(PHONE_RE, "PHONE")

    def _tckn_sub(match: re.Match) -> str:
        val = match.group(0)
        if is_valid_tckn(val):
            counts["TCKN"] = counts.get("TCKN", 0) + 1
            return "[REDACTED_TCKN]"
        return val

    out = TCKN_RE.sub(_tckn_sub, out)
    return RedactionReport(redacted_text=out, counts=counts)
