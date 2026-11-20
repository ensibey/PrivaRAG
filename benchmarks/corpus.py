"""Labeled enterprise RAG corpus and query set for reproducible retrieval benchmarking."""

from __future__ import annotations

from typing import Dict, List, Tuple

ENTERPRISE_DOCS: List[Tuple[str, str, Dict[str, str]]] = [
    ("fin-q4-revenue", "FY2025 Q4 revenue reached 12.4M EUR with 18.2% EBITDA margin driven by enterprise renewals.", {"dept": "finance"}),
    ("fin-cloud-cost", "Annual GPU inference infrastructure spend dropped 31% after migrating to quantized local Llama 3 8B.", {"dept": "finance"}),
    ("fin-audit-policy", "External financial audits require read-only SQLite ledger snapshots retained for 7 years.", {"dept": "finance"}),
    ("sec-airgap", "Air-gapped PrivaRAG clusters prohibit outbound HTTP traffic and run Ollama models on internal VLANs.", {"dept": "security"}),
    ("sec-p1-sla", "Critical P1 security incidents require on-call triage within 15 minutes and CISO notification within 1 hour.", {"dept": "security"}),
    ("sec-pii-masking", "Before prompt assembly, customer emails, IBANs, phone numbers, and TCKN identifiers must be redacted.", {"dept": "security"}),
    ("sec-key-rotation", "Production API signing keys and TLS certificates rotate automatically every 90 days via Vault.", {"dept": "security"}),
    ("hr-remote-work", "Engineering teams follow a hybrid schedule with up to 3 remote workdays per week and core hours 10:00-15:00.", {"dept": "hr"}),
    ("hr-parental-leave", "Paid parental leave provides 16 weeks for primary caregivers and 6 weeks for secondary caregivers.", {"dept": "hr"}),
    ("hr-learning-budget", "Every full-time employee receives a 2,000 EUR annual stipend for conferences, books, and certifications.", {"dept": "hr"}),
    ("eng-hybrid-search", "Hybrid retrieval fuses Okapi BM25 lexical scores with dense cosine embeddings via Reciprocal Rank Fusion (k=60).", {"dept": "engineering"}),
    ("eng-semantic-cache", "VectorSemanticCache matches repeat questions above 0.90 cosine similarity in under 2 milliseconds.", {"dept": "engineering"}),
    ("eng-sql-guard", "Natural-language SQL queries are restricted to single SELECT or WITH statements in SQLite read-only URI mode.", {"dept": "engineering"}),
    ("legal-gdpr-dpa", "EU customer data processing agreements enforce strict data residency inside Frankfurt and Zurich data centers.", {"dept": "legal"}),
    ("legal-nda-term", "Standard mutual non-disclosure agreements remain binding for 5 years from the effective signing date.", {"dept": "legal"}),
]

EVAL_QUERIES: List[Tuple[str, str]] = [
    ("What was the Q4 EBITDA margin and revenue?", "fin-q4-revenue"),
    ("How much did GPU inference spend drop after Llama 3 quantization?", "fin-cloud-cost"),
    ("How many years are SQLite ledger snapshots retained for audits?", "fin-audit-policy"),
    ("How does air-gapped PrivaRAG handle outbound HTTP traffic?", "sec-airgap"),
    ("What is the triage SLA for critical P1 security incidents?", "sec-p1-sla"),
    ("Which PII fields must be redacted before prompt assembly?", "sec-pii-masking"),
    ("How often do production TLS certificates and signing keys rotate?", "sec-key-rotation"),
    ("How many remote workdays per week are allowed for engineering?", "hr-remote-work"),
    ("How many weeks of paid parental leave are provided?", "hr-parental-leave"),
    ("What is the annual conference and certification learning stipend?", "hr-learning-budget"),
    ("How does hybrid search fuse Okapi BM25 and dense embeddings?", "eng-hybrid-search"),
    ("What cosine similarity threshold does VectorSemanticCache use?", "eng-semantic-cache"),
    ("How are natural-language SQL queries restricted for safety?", "eng-sql-guard"),
    ("Where is EU customer data residency enforced under GDPR DPA?", "legal-gdpr-dpa"),
    ("How long do standard mutual NDA agreements remain binding?", "legal-nda-term"),
]
