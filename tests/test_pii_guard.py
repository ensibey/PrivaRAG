"""Tests for air-gapped PII redaction guard."""

from pii_guard import is_valid_tckn, redact_pii


def test_redacts_email_iban_phone_and_tckn():
    # 10000000146 is a mathematically valid TCKN checksum test fixture
    assert is_valid_tckn("10000000146") is True
    assert is_valid_tckn("12345678901") is False

    sample = (
        "Contact ahmet.yilmaz@corp.com or +90 532 555 12 34. "
        "IBAN: TR330006100519786457841326, ID: 10000000146, Order: 99999999999."
    )
    report = redact_pii(sample)

    assert "[REDACTED_EMAIL]" in report.redacted_text
    assert "[REDACTED_PHONE]" in report.redacted_text
    assert "[REDACTED_IBAN]" in report.redacted_text
    assert "[REDACTED_TCKN]" in report.redacted_text
    # Invalid checksum 11-digit order number must NOT be redacted
    assert "99999999999" in report.redacted_text
    assert report.total_redacted == 4
