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


def store_batch(points: list[tuple[str, str, dict]]):
    """Convert multiple chunk texts to vectors and store them in Qdrant in one upsert."""
    from backend.app.vector.embeddings import embed_batch

    chunk_ids = [p[0] for p in points]
    texts = [p[1] for p in points]
    payloads = [p[2] for p in points]
    vectors = embed_batch(texts)

    db.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=chunk_id,
                vector=vector,
                payload=payload,
            )
            for chunk_id, vector, payload in zip(chunk_ids, vectors, payloads)
        ],
    )


def delete_by_doc_id(doc_id: str):
    """Delete all chunks associated with a document from Qdrant."""
    db.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )
        ),
    )


def search(question: str, limit: int = 5, document_ids: list[str] | None = None):
    """Find the chunks whose meaning is closest to the question."""
    query_filter = None
    if document_ids:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchAny(any=document_ids),
                )
            ]
        )

    results = db.query_points(
        collection_name=COLLECTION,
        query=embed(question),
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
    ).points
    return results
