<<<<<<< feature/vector-embeddings
from functools import lru_cache

=======
>>>>>>> dev
from backend.app.rag.pipeline import RAGPipeline
from backend.app.retrieval.retriever import HybridRetriever
from backend.app.retrieval.reranker import Reranker


class QueryProcessor:
    def process(self, query: str) -> str:
        return query.strip()


class ContextBuilder:
    def build(self, chunks: list) -> str:
        parts = []
        for c in chunks:
            text = c.get("text", "")
            metadata = c.get("metadata", {})
            prefix = metadata.get("context_prefix", "")
            parts.append(f"{prefix}{text}" if prefix else text)
        return "\n\n".join(parts)
        return "\n\n".join(c.get("text", "") for c in chunks)


class LLM:
    def generate(self, query: str, context: str) -> str:
        from backend.app.llm.prompts import build_prompt
        from backend.app.llm.client import generate_answer
        from backend.app.llm.guardrails import validate_answer

        prompt = build_prompt(query, context)
        answer = generate_answer(prompt)
        return validate_answer(answer, context)


@lru_cache(maxsize=1)
def build_pipeline() -> RAGPipeline:
    """Wire the real retrieval + LLM components into the RAG pipeline."""
    return RAGPipeline(
        query_processor=QueryProcessor(),
        retriever=HybridRetriever(),
        reranker=Reranker(),
        context_builder=ContextBuilder(),
        llm=LLM(),
    )


def ingest_pdf(file_path: str):
    """Loader -> Parser -> Chunker -> Embedding -> Qdrant."""
    from backend.app.ingestion.loader import DocumentLoader
    from backend.app.ingestion.parser import DocumentParser
    from backend.app.chunking.chunker import SmartChunker
    from backend.app.vector.qdrant import store_batch

    document = DocumentLoader.load_from_path(file_path)
    doc_id = document["doc_id"]

    parsed_pages = DocumentParser().parse(
        file_bytes=document["file_bytes"],
        doc_id=doc_id,
    from backend.app.vector.qdrant import store

    document = DocumentLoader.load_from_path(file_path)

    parsed_pages = DocumentParser().parse(
        file_bytes=document["file_bytes"],
        doc_id=document["doc_id"],
        filename=document["metadata"]["filename"],
    )

    chunks = SmartChunker().chunk_document(parsed_pages)

    points = []
    for chunk in chunks:
        points.append((
            chunk.chunk_id,
            chunk.context_prefix + chunk.text,
            chunk.to_qdrant_payload(),
        ))

    store_batch(points)

    return chunks, doc_id
    for chunk in chunks:
        store(
            chunk_id=chunk.chunk_id,
            text=chunk.context_prefix + chunk.text,
            payload=chunk.to_qdrant_payload(),
        )

    return chunks
