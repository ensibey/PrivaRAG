# config.py — Central configuration for the Local RAG System

# --- Ollama Models ---
LLM_MODEL = "llama3"
EMBEDDING_MODEL = "nomic-embed-text"

# --- ChromaDB ---
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "local_rag_docs"

# --- Text Splitting ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Retrieval ---
TOP_K_RESULTS = 5          # Number of doc chunks to retrieve per query

# --- Conversation Memory ---
MAX_HISTORY_TURNS = 5      # Turns to keep verbatim; older turns get summarised

# --- Document Storage (for PDF preview) ---
DOCS_STORE_DIR = "./uploaded_docs"

# --- Ollama Server ---
OLLAMA_BASE_URL = "http://localhost:11434"
