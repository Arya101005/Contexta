# reranker.py

from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):
        self.model = CrossEncoder(
            "BAAI/bge-reranker-large"
        )

    def rerank(self, query, chunks, top_k=5):

        # Create query-document pairs
        pairs = [
            [query, chunk["text"]]
            for chunk in chunks
        ]

        # Calculate relevance scores
        scores = self.model.predict(pairs)

        # Attach score to every chunk
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        # Highest score first
        chunks.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return chunks[:top_k]

    # Pipeline-compatible alias
    def rank(self, query, documents, top_k=5):
        return self.rerank(query, documents, top_k)