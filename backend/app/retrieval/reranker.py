from sentence_transformers import CrossEncoder  # reranker model — scores query-document pairs
from functools import lru_cache  # cache model loading so it only happens once


@lru_cache(maxsize=1)  # load CrossEncoder model only once across all instances
def _load_model():
    return CrossEncoder("BAAI/bge-reranker-v2-m3")


class Reranker:
    """Reranks retrieved chunks using a CrossEncoder for better relevance scoring."""

    def __init__(self):
        self._model = None  # lazy-load on first use

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model()
        return self._model

    def rerank(self, query, chunks, top_k=5):
        """Score each chunk against the query and return the top_k most relevant."""
        if not chunks:
            return []
        pairs = [[query, chunk.get("text", "")] for chunk in chunks]
        scores = self.model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return chunks[:top_k]

    def rank(self, query, documents, top_k=5):
        return self.rerank(query, documents, top_k)
