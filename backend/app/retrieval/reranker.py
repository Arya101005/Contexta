from sentence_transformers import CrossEncoder  # reranker model — scores query-document pairs


class Reranker:
    """Reranks retrieved chunks using a CrossEncoder for better relevance scoring."""

    def __init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-v2-m3")  # multilingual reranker

    def rerank(self, query, chunks, top_k=5):
        """Score each chunk against the query and return the top_k most relevant."""
        if not chunks:
            return []

        # CrossEncoder expects pairs of [query, document] strings
        pairs = [[query, chunk.get("text", "")] for chunk in chunks]
        scores = self.model.predict(pairs)  # run the reranker model on all pairs at once

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)  # attach reranker score to each chunk

        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)  # highest score first
        return chunks[:top_k]  # keep only the top_k most relevant chunks

    def rank(self, query, documents, top_k=5):
        """Pipeline-compatible alias for rerank()."""
        return self.rerank(query, documents, top_k)
