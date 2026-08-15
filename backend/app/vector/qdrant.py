import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from backend.app.vector.embeddings import embed


# Load QDRANT_URL and QDRANT_API_KEY from backend/.env
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


db = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION = "document"


# BGE-base-en-v1.5 creates 768-dimensional vectors.
if not db.collection_exists(COLLECTION):
    db.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
        ),
    )


def store(chunk_id, text: str, payload: dict):
    """Convert chunk text to a vector and store it in Qdrant."""
    db.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=chunk_id,
                vector=embed(text),
                payload=payload,
            )
        ],
    )


def search(question: str, limit: int = 5):
    """Find the chunks whose meaning is closest to the question."""
    results = db.query_points(
        collection_name=COLLECTION,
        query=embed(question),
        limit=limit,
        with_payload=True,
    ).points
    return results
