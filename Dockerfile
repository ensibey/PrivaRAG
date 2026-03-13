# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Local RAG System (Python app only)
# Ollama runs on the host machine, not inside this container.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps (needed for some pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py ingest.py rag_engine.py excel_agent.py sql_agent.py app.py ./

# Persistent directories
RUN mkdir -p chroma_db uploaded_docs

# Streamlit config — disable telemetry and set server options
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

# Ollama runs on the HOST — Docker app calls it via OLLAMA_BASE_URL env var
# Default points to host machine. Override with:
#   docker run -e OLLAMA_BASE_URL=http://host.docker.internal:11434 ...
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
