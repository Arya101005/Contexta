import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from embeddings import embed


# Load QDRANT_URL and QDRANT_API_KEY from .env
load_dotenv()

db = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION = "document"


# BGE-base-en-v1.5 creates 768-dimensional vectors.
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

    db.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=id,
                vector=embed(text),
                payload={"content": text},
            )
        ],
    )


def search(question: str, limit: int = 5):
    """Find the texts whose meaning is closest to the question."""

    results = db.query_points(
        collection_name=COLLECTION,
        query=embed(question),
        limit=limit,
        with_payload=True,
    ).points

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