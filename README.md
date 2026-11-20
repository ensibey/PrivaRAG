# PrivaRAG

[![PrivaRAG CI](https://github.com/ensibey/PrivaRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/ensibey/PrivaRAG/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-10B981)
![Tests: 31 passing](https://img.shields.io/badge/tests-31%20passing-10B981)

**Air-gapped, privacy-first enterprise RAG engine** built for zero external API egress. Combines **Okapi BM25 + dense cosine Reciprocal Rank Fusion (RRF)** retrieval, **PII redaction**, **SHA-256 incremental deduplication**, **source-scoped semantic caching**, **citation/numeric faithfulness verification**, and **read-only natural-language SQLite/Excel routing** via local Ollama.

---

## Architecture

```text
Documents (PDF / MD / TXT / CSV / XLSX)
   │
   ▼
[1] Air-Gapped PII Redaction (`pii_guard.py`) — Emails, Phones, IBAN, Checksum-Verified TCKN
   │
   ▼
[2] Recursive Chunking (`chunker.py`) + SHA-256 Incremental Manifest (`dedup_index.py`)
   │
   ▼
[3] Deterministic Query Router (`query_router.py`)
   ├──► DOC_RAG     ──► VectorSemanticCache (`semantic_cache.py`, sub-1ms lookup)
   │                    └── miss ──► Hybrid RRF (`bm25_retriever.py` + `hybrid_retriever.py`)
   │                                 └──► Local Ollama + Faithfulness Verifier (`faithfulness.py`)
   ├──► SQL         ──► Read-Only SQL Safety Guard (`sql_guard.py`, `?mode=ro`) ──► SQLite
   └──► SPREADSHEET ──► Pandas Excel/CSV Agent (`excel_agent.py`)
```

---

## Key Capabilities

- **Hybrid Sparse + Dense Retrieval (`bm25_retriever.py`, `hybrid_retriever.py`):** Zero-dependency Okapi BM25 (with English + Turkish Unicode tokenization) fused with dense cosine embeddings via Reciprocal Rank Fusion (`k=60`).
- **Source-Scoped Vector Semantic Cache (`semantic_cache.py`):** Matches repeat queries above `0.90` cosine similarity in `<1 ms` and supports `invalidate_source(doc)` when underlying policies change.
- **Read-Only SQL Safety Gate (`sql_guard.py`):** Blocks stacked statements and destructive DDL/DML (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ATTACH`, `PRAGMA`) before execution, and opens SQLite in native `?mode=ro` URI mode.
- **Air-Gapped PII Redaction (`pii_guard.py`):** Masks emails, phones, IBANs, credit cards, and mathematically checksum-verified Turkish National IDs (TCKN) without flagging random order numbers.
- **Zero-LLM Faithfulness & Citation Verifier (`faithfulness.py`):** Flags hallucinated financial numbers and out-of-range `[n]` citations against retrieved chunks.

---

## Reproducible Retrieval & Cache Benchmark

Measured on the 15-document, 15-query enterprise corpus (`benchmarks/corpus.py` — finance, security, HR, engineering, legal):

```bash
python benchmarks/run_benchmark.py
```

| Method | Recall@1 | Recall@3 | MRR | NDCG@3 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Okapi BM25 (Sparse)** | `100.0%` | `100.0%` | `1.000` | `1.000` | `0.025 ms` |
| **Dense Cosine** | `100.0%` | `100.0%` | `1.000` | `1.000` | `0.220 ms` |
| **Hybrid RRF (BM25 + Dense)** | `100.0%` | `100.0%` | `1.000` | `1.000` | `0.295 ms` |
| **VectorSemanticCache Hit** | `100.0% hit` | — | — | — | `0.226 ms` |

---

## Quick Start

### 1. Offline CLI (Zero API Keys / No Ollama Required)

```bash
# Hybrid BM25 + dense RRF search over local docs
python cli.py search docs/ "air-gapped SQLite deployment"

# Redact PII from sensitive text
python cli.py redact "Contact ahmet@corp.com, IBAN TR330006100519786457841326"

# Audit an LLM-generated SQL query for read-only safety
python cli.py audit-sql "SELECT region, SUM(revenue) FROM sales GROUP BY region"
python cli.py audit-sql "SELECT * FROM sales; DROP TABLE sales;"

# Classify query intent (DOC_RAG vs SQL vs SPREADSHEET)
python cli.py route "Compare Q3 budget column in the Excel sheet"
```

### 2. Full Local UI & Ollama Mode

```bash
pip install -r requirements.txt
ollama pull llama3
ollama pull nomic-embed-text
streamlit run app.py
```

### 3. Run Tests

```bash
pytest -v tests/
```

---

## License

MIT
