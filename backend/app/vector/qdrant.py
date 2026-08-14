from qdrant_client import QdrantClient, models
from .embeddings import embed

db = QdrantClient(":memory:")

db.create_collection(
    "documents",
    vectors_config=models.VectorParams(
        size=1024,
        distance=models.Distance.COSINE
    )
)

def store(chunk_id, text, document_id, page):
    db.upsert(
        "documents",
        [models.PointStruct(
            id=chunk_id,
            vector=embed(text),
            payload={
                "document_id": document_id,
                "page": page,
                "content": text
            }
        )]
    )

def search(question):
    return db.query_points(
        "documents",
        query=embed(question),
        limit=5
    ).points