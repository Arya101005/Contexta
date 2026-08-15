import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class Settings:
    APP_NAME = os.getenv("APP_NAME", "Contexta")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"

    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION = os.getenv(
        "QDRANT_COLLECTION",
        "contexta_documents"
    )

    # Embedding
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-base-en-v1.5"
    )

    # Reranker
    RERANKER_MODEL = os.getenv(
        "RERANKER_MODEL",
        "BAAI/bge-reranker-v2-m3"
    )

    # LLM
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
    LLM_MODEL = os.getenv(
        "LLM_MODEL",
        "Qwen/Qwen2.5-7B-Instruct"
    )

    # PostgreSQL
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/contexta"
    )

    # Retrieval
    TOP_K = int(os.getenv("TOP_K", "10"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


settings = Settings()