# config.py — Central configuration for PrivaRAG (Privacy-Centric Enterprise RAG)

# --- Ollama Models ---
LLM_MODEL = "llama3"
EMBEDDING_MODEL = "nomic-embed-text"

# --- Vector Stores (ChromaDB & Qdrant / Semantic Cache) ---
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "local_rag_docs"
SEMANTIC_CACHE_PATH = "./chroma_db/semantic_cache.json"
SEMANTIC_CACHE_THRESHOLD = 0.88
SEMANTIC_CACHE_TTL = 86400

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
