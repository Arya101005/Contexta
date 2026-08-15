from functools import lru_cache
from backend.app.vector.qdrant import search as dense_search


def bm25_search(query):
    # Stub: returns empty list since BM25 not implemented
    return []


def remove_duplicates(results):
    # Remove duplicates based on chunk_id
    seen = set()
    unique = []
    for r in results:
        chunk_id = r.get("chunk_id") if isinstance(r, dict) else r.payload.get("chunk_id") if r.payload else None
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(r)
    return unique


@lru_cache(maxsize=100)
def retrieve(query, document_ids=None):
    """
    Retrieve relevant chunks using
    dense search + BM25.
    """

    # 1. Dense retrieval from Qdrant
    dense_results = dense_search(query)

    # 2. Keyword retrieval using BM25
    bm25_results = bm25_search(query)

    # 3. Combine both result sets
    results = list(dense_results) + list(bm25_results)

    # 4. Remove duplicate chunks
    results = remove_duplicates(results)

    # 5. Return candidates for reranking
    # Convert ScoredPoint to dict format for reranker
    for i, r in enumerate(results):
        if hasattr(r, 'payload') and r.payload:
            # Convert from ScoredPoint
            payload = r.payload
            # Convert to dict format for reranker
            results[i] = {
                "text": payload.get("text", ""),
                "metadata": payload
            }
    
    return results[:20]


class HybridRetriever:
    """Adapter so the RAG pipeline can call retrieve(query=, document_ids=)."""

    def retrieve(self, query, document_ids=None):
        return retrieve(query, document_ids)