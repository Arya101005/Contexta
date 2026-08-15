import os
<<<<<<< HEAD

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from embeddings import embed


# Load QDRANT_URL and QDRANT_API_KEY from .env
load_dotenv()
=======
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from backend.app.vector.embeddings import embed


# Load QDRANT_URL and QDRANT_API_KEY from backend/.env
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

>>>>>>> 4669588 (add retrieval cache and citations)

db = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION = "document"


# BGE-base-en-v1.5 creates 768-dimensional vectors.
<<<<<<< HEAD
# COSINE measures how similar two vectors are.
db.create_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
    ),
)


def store(id: int, text: str):
    """Convert text to a vector and store it in Qdrant."""

=======
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
>>>>>>> 4669588 (add retrieval cache and citations)
    db.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
<<<<<<< HEAD
                id=id,
                vector=embed(text),
                payload={"content": text},
=======
                id=chunk_id,
                vector=embed(text),
                payload=payload,
>>>>>>> 4669588 (add retrieval cache and citations)
            )
        ],
    )


def search(question: str, limit: int = 5):
<<<<<<< HEAD
    """Find the texts whose meaning is closest to the question."""

=======
    """Find the chunks whose meaning is closest to the question."""
>>>>>>> 4669588 (add retrieval cache and citations)
    results = db.query_points(
        collection_name=COLLECTION,
        query=embed(question),
        limit=limit,
        with_payload=True,
    ).points
<<<<<<< HEAD

    return results


# ------------------------------------------------------------
# Sample data
# ------------------------------------------------------------

texts = [
    "I love eating pizza.",
    "Pizza is my favorite food.",
    "The weather is very hot today.",
    "Dogs are friendly animals.",
    "I enjoy playing football.",
]

for id, text in enumerate(texts, start=1):
    store(id, text)


# ------------------------------------------------------------
# Test semantic search
# ------------------------------------------------------------

question = "I really like pizza."
print(f"\nQUERY: {question}\n")

for result in search(question):

    print(f"Score:   {result.score:.4f}")
    print(f"Text:    {result.payload['content']}")
    print("-" * 60)
=======
    return results
>>>>>>> 4669588 (add retrieval cache and citations)
