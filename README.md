# PrivaRAG

[![CI](https://github.com/ensibey/PrivaRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/ensibey/PrivaRAG/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Local, air-gapped Retrieval-Augmented Generation (RAG) and structured data query engine running on **Ollama** (`llama3`, `nomic-embed-text`), **ChromaDB**, and **SQLite**. Designed for environments where documents (financial tables, internal PDFs, legal contracts) cannot be sent to external LLM APIs.

---

## Architecture & Design Decisions

```text
User Query
   │
   ├─► [1. VectorSemanticCache] ──► Cosine similarity >= 0.88? ──► Return cached response (<3 ms)
   │          │ (miss)
   ├─► [2. Conversation Memory] ──► > 5 turns? ──► Compress older turns into 2-sentence summary
   │
   ├─► [3. EnsembleRetriever]
   │          ├── BM25Retriever (weight=0.4)   -> Exact lexical match (skus, IDs, dates, clauses)
   │          └── ChromaDB Dense (weight=0.6)  -> Semantic vector match (nomic-embed-text, 768d)
   │
   └─► [4. Local Ollama LLM]    ──► Grounded synthesis with page/sheet citations
```

### Why 40% BM25 + 60% Dense Retrieval?
Pure dense embeddings (`nomic-embed-text`) often miss exact alphanumeric identifiers—such as invoice numbers, part codes, or Turkish statutory article numbers (`Madde 14/B`)—while pure lexical search fails on paraphrased questions. Combining `BM25Retriever` (0.4) and `Chroma` (0.6) via Reciprocal Rank Fusion in `EnsembleRetriever` resolves both without requiring a cross-encoder reranker on CPU.

### Vector Semantic Cache (`semantic_cache.py`)
Local LLM generation on consumer hardware takes 2.5–4.0 seconds per query. `VectorSemanticCache` stores normalized query vectors alongside their grounded citations on disk (`chroma_db/semantic_cache.json`, configurable TTL). Paraphrased repeat queries above the `0.88` cosine threshold bypass LLM inference and return in `< 3 ms`. When an embedding server is unreachable during unit testing or cold start, it falls back to a deterministic signed character 3-gram feature hash (`128d`).

### Structured Data Routing (`sql_agent.py`, `excel_agent.py`)
Chunking spreadsheets into vector stores destroys column-wise aggregations (`SUM`, `AVG`, `GROUP BY`). PrivaRAG routes `.xlsx` / `.csv` / `.db` analytical queries either through an auto-converted SQLite schema (`create_sql_query_chain`) or a Pandas dataframe agent (`PandasDataFrameAgent`).

---

## Quickstart

### 1. Prerequisites
Install [Ollama](https://ollama.com) and pull the default inference and embedding models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Installation

```bash
git clone https://github.com/ensibey/PrivaRAG.git
cd PrivaRAG
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit Interface

```bash
streamlit run app.py
```

### 4. CLI Usage (Headless Ingestion, SQL & Evaluation)

```bash
# Index a PDF or Excel workbook from the command line
python ingest.py ./samples/quarterly_report.pdf

# Query a SQLite database in plain language
python sql_agent.py ./sales.db "What were the top 3 regions by total revenue in Q3?"

# Run faithfulness and relevance evaluation across a benchmark JSON
python eval.py --output eval_results.json
```

---

## Local Model Comparison

Evaluated on 50 internal PDF/Excel QA pairs (16 GB RAM, RTX 3060 Laptop GPU, `TOP_K_RESULTS=5`, `CHUNK_SIZE=800`):

| Model (via Ollama) | Mean Faithfulness | Mean Answer Relevance | Mean Latency (Cold) | Mean Latency (Cache Hit) |
| :--- | :---: | :---: | :---: | :---: |
| `llama3:8b-instruct-q4_K_M` | **0.91** | **0.89** | 3.18 s | **0.002 s** |
| `mistral:7b-instruct-v0.3-q4_K_M` | 0.87 | 0.86 | **2.45 s** | **0.002 s** |
| `qwen2.5:7b-instruct` | 0.90 | 0.88 | 2.72 s | **0.002 s** |

---

## Project Layout

```text
PrivaRAG/
├── app.py                     # Streamlit UI (Document Q&A, Excel Agent, SQL Agent)
├── rag_engine.py              # Hybrid BM25+Chroma retrieval chain & memory compression
├── semantic_cache.py          # Cosine similarity query cache with TTL and disk persistence
├── ingest.py                  # PyMuPDF & Pandas chunking pipeline
├── sql_agent.py               # Excel-to-SQLite converter and text-to-SQL executor
├── excel_agent.py             # Pandas DataFrame agent for tabular calculations
├── eval.py                    # LLM-as-a-Judge faithfulness & relevance benchmark CLI
├── config.py                  # Model names, chunk sizes, and cache thresholds
└── tests/
    └── test_semantic_cache.py # Unit tests for vector cache hits, misses, and eviction
```

## Testing

```bash
pytest -v tests/
```

## License

MIT License.
