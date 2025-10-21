# Configuration pour le chatbot

# LLM Settings
LLM_MODEL = "llama2"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 500

# Embedding Settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG Settings
RAG_TOP_K = 3  # Documents à retriever
RAG_MIN_RELEVANCE = 0.5

# Hybrid Search Weights
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3

# Reranking
USE_RERANKING = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Database
CHROMA_DB_PATH = "./chroma_data"
WIKI_DATA_PATH = "./processed_wiki"

# Server
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000