"""Central runtime configuration for PrivaRAG."""

LLM_MODEL = "llama3"
EMBEDDING_MODEL = "nomic-embed-text"

CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "local_rag_docs"
SEMANTIC_CACHE_PATH = "./chroma_db/semantic_cache.json"
SEMANTIC_CACHE_THRESHOLD = 0.88
SEMANTIC_CACHE_TTL = 86400

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

TOP_K_RESULTS = 5
MAX_HISTORY_TURNS = 5

DOCS_STORE_DIR = "./uploaded_docs"
OLLAMA_BASE_URL = "http://localhost:11434"
