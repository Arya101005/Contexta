import os  # read environment variables
from pathlib import Path  # locate the .env file

from dotenv import load_dotenv  # load variables from .env into os.environ
from qdrant_client import QdrantClient, models  # Qdrant vector DB client + types

from backend.app.config import settings  # centralized config
from backend.app.vector.embeddings import embed, embed_batch  # text -> vector conversion

# load QDRANT_URL and QDRANT_API_KEY from backend/.env
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# create the Qdrant client — connects to cloud or self-hosted instance
db = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

COLLECTION = settings.QDRANT_COLLECTION  # one collection for all documents

# create the collection on first run if it doesn't exist
if not db.collection_exists(COLLECTION):
    db.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=settings.EMBEDDING_DIMENSION,  # 768 for BGE-base
            distance=models.Distance.COSINE,  # cosine similarity for semantic search
        ),
    )

# ensure payload index exists for doc_id filtering (required by Qdrant)
try:
    db.create_payload_index(
        collection_name=COLLECTION,
        field_name="doc_id",
        field_type=models.PayloadSchemaType.KEYWORD,
    )
except Exception:
    pass  # index already exists


def store(chunk_id, text: str, payload: dict):
    """Convert a single text into a vector and upsert it into Qdrant."""
    db.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=chunk_id,  # stable UUID so we can update/delete later
                vector=embed(text),  # 768-dim embedding
                payload=payload,  # metadata stored alongside the vector
            )
        ],
    )


def store_batch(points: list[tuple[str, str, dict]]):
    """Convert multiple texts into vectors and upsert them all in one network call."""
    chunk_ids = [p[0] for p in points]
    texts = [p[1] for p in points]
    payloads = [p[2] for p in points]
    vectors = embed_batch(texts)  # batch embed for speed

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
    """Delete all vectors whose payload contains the given doc_id."""
    db.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",  # payload field to filter on
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )
        ),
    )


def search(question: str, limit: int = 15, document_ids: list[str] | None = None):
    """Find chunks whose meaning is closest to the question using dense vector search."""
    query_filter = None
    if document_ids:
        # scope search to only the selected documents
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchAny(any=document_ids),
                )
            ]
        )

    response = db.query_points(
        collection_name=COLLECTION,
        query=embed(question),  # embed the user's question into the same vector space
        limit=limit,  # return top N most similar chunks
        with_payload=True,  # include metadata so we can show citations
        query_filter=query_filter,
    )

    # normalize results into plain dicts for consistent handling downstream
    results = []
    for point in response.points:
        payload = point.payload or {}
        results.append({
            "id": str(point.id),
            "chunk_id": payload.get("chunk_id", str(point.id)),
            "score": point.score,  # cosine similarity score
            "payload": payload,
            "text": payload.get("text", ""),
            "doc_id": payload.get("doc_id"),
            "filename": payload.get("filename"),
            "page_number": payload.get("page_number"),
            "section_title": payload.get("section_title"),
            "section_path": payload.get("section_path"),
            "content_type": payload.get("content_type"),
        })

    return results
