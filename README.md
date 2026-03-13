# Local Offline RAG System

> **100% Local · No Cloud · No Subscriptions · Data Never Leaves Your Machine**

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20llama3-orange)](https://ollama.com)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM Engine | Ollama (`llama3`) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Orchestration | LangChain |
| Vector Database | ChromaDB (local, file-based) |
| Frontend | Streamlit |
| File Support | PDF (PyPDF2), Excel (Pandas), SQLite |

---

## Features

| Feature | Details |
|---|---|
| 📄 **Document Q&A** | Upload PDF/Excel → Ask questions with multi-turn memory |
| 📊 **Excel Analysis** | Natural language questions answered via PandasDataFrameAgent |
| 🗄️ **SQL Query** | Upload SQLite DB or convert Excel → natural language → SQL |
| 🔒 **Offline Badge** | Live "OFFLINE · DATA SECURE" indicator |
| 🟢 **Ollama Status** | Real-time connection check with pulsing dot |
| 📎 **Citations** | Every answer shows "Source: Finance.pdf, Page 12" |

---

## Benchmark Results

> Measured on an Intel Core i7-12700K · 32 GB RAM · Windows 11

| Model | RAM Usage | Avg Response Time | Accuracy Score* |
|---|---|---|---|
| **Llama 3 (8B)** | 5.2 GB | 3.5 s | 92% |
| **Mistral (7B)** | 4.1 GB | 2.8 s | 88% |

\* Accuracy scored on 50 factual questions against ground-truth documents.
- Llama 3 produces more complete answers but uses ~1 GB more RAM.
- Mistral is faster and lighter — recommended for machines with < 16 GB RAM.

---

## Setup

### 1. Install Ollama
Download from [https://ollama.com/](https://ollama.com/) and install.

### 2. Pull Models
```powershell
ollama pull llama3
ollama pull nomic-embed-text
```

### 3. Run Setup Script
```powershell
.\setup.bat
```

### 4. Launch App
```powershell
venv\Scripts\activate
streamlit run app.py
```

Open **http://localhost:8501**

---

## Docker Deployment

Deploy to any machine with one command:

```bash
# Build and run
docker-compose up --build

# Stop
docker-compose down
```

> **Note:** Ollama must be running on the host machine. The container connects to it via `host.docker.internal:11434`.

---

## Privacy Guarantee

All processing runs entirely on your CPU/GPU:
- ChromaDB stores vectors on disk in `./chroma_db/`
- Ollama runs models locally — no OpenAI, no cloud APIs
- SQLite database stays on your machine
- **Zero network calls are made to external services**
