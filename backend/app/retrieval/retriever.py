import logging  # structured logging
from typing import List, Dict, Any, Optional  # type hints
from rank_bm25 import BM25Okapi  # lightweight BM25 implementation for sparse retrieval

from backend.app.models.models import Chunk  # ORM model for chunk records
from backend.app.config import settings  # retrieval config (top-k values, etc.)

logger = logging.getLogger(__name__)

# global in-memory BM25 index — rebuilt on startup and after every upload/delete
_bm25_index: Optional["BM25Index"] = None


def get_bm25_index() -> "BM25Index":
    """Return the singleton BM25 index, creating it if needed."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index


def rebuild_bm25(db) -> None:
    """Rebuild the entire BM25 index from all chunks currently in PostgreSQL."""
    bm25 = get_bm25_index()
    bm25.build(db.query(Chunk).all())  # load every chunk from DB and index it
    logger.info("BM25 index rebuilt with %d chunks", len(db.query(Chunk).all()))


class BM25Index:
    """In-memory BM25 sparse retrieval index over chunk text."""

    def __init__(self):
        self.ids: List[str] = []  # ordered list of chunk IDs
        self.corpus: List[List[str]] = []  # tokenized text for each chunk
        self.bm25: Optional[BM25Okapi] = None  # the actual BM25 model
        self.meta: Dict[str, Dict[str, Any]] = {}  # chunk_id -> metadata mapping
        self.texts: Dict[str, str] = {}  # chunk_id -> raw text mapping

    def build(self, chunks: List[Chunk]) -> None:
        """Rebuild the index from a list of Chunk ORM objects."""
        self.ids, self.corpus, self.meta, self.texts = [], [], {}, {}
        for c in chunks:
            self.ids.append(c.chunk_id)
            self.corpus.append(c.text.lower().split())  # lowercase + split for BM25
            self.meta[c.chunk_id] = {
                "document_id": c.document_id,
                "content_type": c.content_type,
                "doc_id": c.document.doc_id if c.document else None,  # Qdrant-scoped doc_id
                "filename": c.filename,
                "page_number": c.page_number,
                "section_title": c.section_title,
                "section_path": c.section_path,
            }
            self.texts[c.chunk_id] = c.text
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None  # build model or set None if empty

    def search(self, query: str, top_k: int = 15, document_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Search the BM25 index and return top_k results, optionally filtered by document IDs."""
        if not self.bm25:
            return []

        scores = self.bm25.get_scores(query.lower().split())  # score every chunk against the query
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)  # sort by score descending

        results = []
        for idx in ranked:
            if len(results) >= top_k:
                break
            cid = self.ids[idx]
            m = self.meta.get(cid, {})
            if document_ids is not None and m.get("document_id") not in document_ids:
                continue  # skip chunks from documents the user didn't select
            results.append({
                "chunk_id": cid,
                "score": float(scores[idx]),  # raw BM25 score
                "document_id": m.get("document_id"),
                "content_type": m.get("content_type"),
                "text": self.texts.get(cid, ""),  # include text for reranker
                "payload": {  # normalize to same shape as Qdrant results
                    "chunk_id": cid,
                    "doc_id": m.get("doc_id"),
                    "filename": m.get("filename"),
                    "page_number": m.get("page_number"),
                    "section_title": m.get("section_title"),
                    "section_path": m.get("section_path"),
                    "content_type": m.get("content_type", ""),
                },
            })
        return results


def reciprocal_rank_fusion(dense_results, bm25_results, k: int = 60, top_k: int = 15):
    """
    Merge dense and sparse results using Reciprocal Rank Fusion.
    RRF(chunk) = sum over all lists of 1 / (k + rank)
    k=60 is a standard constant that smooths the contribution of lower ranks.
    """
    scores, items = {}, {}
    for rank, result in enumerate(dense_results, 1):  # ranks start at 1
        cid = result.get("chunk_id") or result.get("id")
        if cid:
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            items[cid] = result  # keep the dense result (has vector score)
    for rank, result in enumerate(bm25_results, 1):
        cid = result.get("chunk_id") or result.get("id")
        if cid:
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in items:
                items[cid] = result  # only add BM25 result if not already in dense

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)  # sort by RRF score
    return [items[cid] for cid, _ in ranked[:top_k]]  # return top_k fused results


class HybridRetriever:
    """Combines dense vector search + BM25 sparse search with RRF fusion."""

    def __init__(self, vector_db, bm25: BM25Index):
        self.vector_db = vector_db  # Qdrant client
        self.bm25 = bm25  # in-memory BM25 index

    def retrieve(self, query, document_ids: Optional[List[str]] = None):
        """Run both retrievers, fuse with RRF, return top_k results."""
        from backend.app.vector.qdrant import search as qdrant_search

        dense = qdrant_search(query, limit=settings.DENSE_TOP_K, document_ids=document_ids)

        # convert Qdrant doc_ids (SHA-256 strings) to PostgreSQL integer IDs for BM25 filtering
        doc_int_ids = None
        if document_ids:
            from backend.app.database import SessionLocal
            from backend.app.models.models import Document
            db = SessionLocal()
            try:
                docs = db.query(Document).filter(Document.doc_id.in_(document_ids)).all()
                doc_int_ids = [d.id for d in docs]
            finally:
                db.close()

        bm25_results = self.bm25.search(query, top_k=settings.BM25_TOP_K, document_ids=doc_int_ids)

        return reciprocal_rank_fusion(dense, bm25_results, k=settings.RRF_K, top_k=settings.RRF_TOP_K)
