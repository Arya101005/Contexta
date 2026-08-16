import os  # access environment variables
from pathlib import Path  # build cross-platform paths
from dotenv import load_dotenv  # load variables from .env file

# load .env from project root (parents[1] = backend/app/config.py -> project root)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class Settings:
    APP_NAME = os.getenv("APP_NAME", "Contexta")  # app name, defaults to Contexta
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"  # boolean debug flag

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # Qdrant cloud/self-hosted URL
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")  # Qdrant API key for auth
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "document")  # single collection for all docs

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")  # HuggingFace model id
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))  # vector size for this model

    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")  # CrossEncoder model id

    LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # LLM provider API key
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")  # custom base URL if needed
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")  # Groq model name
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # provider identifier

    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/contexta")  # PostgreSQL conn

    TOP_K = int(os.getenv("TOP_K", "15"))  # legacy generic top-k (kept for compat)
    DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "10"))  # dense vector search result count
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))  # sparse BM25 search result count
    RRF_TOP_K = int(os.getenv("RRF_TOP_K", "10"))  # final fused candidate count after RRF
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))  # chunks kept after CrossEncoder reranking
    RRF_K = int(os.getenv("RRF_K", "60"))  # RRF constant — lower = more weight to top ranks

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # target token count per text chunk
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # token overlap between adjacent chunks

    EVIDENCE_THRESHOLD = float(os.getenv("EVIDENCE_THRESHOLD", "-1.0"))  # min rerank score to trust answer
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))  # max tokens in LLM response


settings = Settings()  # singleton config instance imported by other modules
