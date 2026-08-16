import logging  # structured logging
from functools import lru_cache  # cache the pipeline so models load only once

from backend.app.rag.pipeline import RAGPipeline, clear_query_cache  # pipeline + cache invalidation
from backend.app.retrieval.retriever import HybridRetriever, get_bm25_index  # retrieval components
from backend.app.retrieval.reranker import Reranker  # CrossEncoder reranker

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Lightweight query cleaner — strips whitespace."""
    def process(self, query: str) -> str:
        return query.strip()


class ContextBuilder:
    """Concatenates retrieved chunks into a single context string for the LLM."""
    def build(self, chunks: list) -> str:
        parts = []
        for c in chunks:
            text = c.get("text", "")
            prefix = c.get("context_prefix") or ""  # use stored prefix if available
            if not prefix:
                # fallback: build prefix from payload metadata
                payload = c.get("payload", {})
                prefix = (
                    f"Document: {payload.get('filename', '')} | "
                    f"Section: {payload.get('section_title', '')} | "
                    f"Page: {payload.get('page_number', '')}\n"
                )
            parts.append(f"{prefix}{text}" if prefix else text)
        return "\n\n".join(parts)  # join chunks with double newlines for readability


class LLM:
    """Wrapper that builds a prompt, calls the LLM, and returns the raw answer."""
    def generate(self, query: str, context: str) -> str:
        from backend.app.llm.prompts import build_prompt  # build the full prompt
        from backend.app.llm.client import generate_answer  # call Groq API

        prompt = build_prompt(query, context)
        answer = generate_answer(prompt)
        return answer  # guardrails are applied in the pipeline, not here


@lru_cache(maxsize=1)  # cache the pipeline instance so models are loaded only once
def build_pipeline() -> RAGPipeline:
    """Wire all RAG components together and return a ready-to-use pipeline."""
    from backend.app.vector.qdrant import db as qdrant_db  # import here to avoid circular deps

    return RAGPipeline(
        query_processor=QueryProcessor(),
        retriever=HybridRetriever(
            vector_db=qdrant_db,  # Qdrant client for dense retrieval
            bm25=get_bm25_index(),  # in-memory BM25 for sparse retrieval
        ),
        reranker=Reranker(),  # CrossEncoder for reranking
        context_builder=ContextBuilder(),  # builds the LLM context string
        llm=LLM(),  # Groq LLM wrapper
    )


def ingest_pdf(file_path: str, db_session=None, document_id: int | None = None):
    """
    Full ingestion pipeline:
    PDF -> loader -> parser -> chunker -> PostgreSQL -> embed -> Qdrant -> BM25
    """
    import time
    from backend.app.ingestion.loader import DocumentLoader
    from backend.app.ingestion.parser import DocumentParser
    from backend.app.chunking.chunker import SmartChunker
    from backend.app.vector.qdrant import store_batch
    from backend.app.retrieval.retriever import get_bm25_index
    from backend.app.models.models import Chunk

    timings = {}
    t0 = time.time()

    # STEP 1: Load PDF and compute SHA-256 doc_id
    document = DocumentLoader.load_from_path(file_path)
    doc_id = document["doc_id"]
    timings["load"] = time.time() - t0

    # STEP 2: Parse every page into layout elements (text, tables, images)
    t1 = time.time()
    parsed_pages = DocumentParser().parse(
        file_bytes=document["file_bytes"],
        doc_id=doc_id,
        filename=document["metadata"]["filename"],
    )
    timings["parse"] = time.time() - t1

    # STEP 3: Chunk the parsed pages into semantic chunks
    t2 = time.time()
    chunks = SmartChunker().chunk_document(parsed_pages)
    timings["chunk"] = time.time() - t2

    logger.info("Ingestion timings: %s", {k: f"{v:.3f}s" for k, v in timings.items()})

    # STEP 4: Store chunks in PostgreSQL (if a DB session is provided)
    if db_session is not None and document_id is not None:
        for chunk in chunks:
            db_chunk = Chunk(
                chunk_id=chunk.chunk_id,
                document_id=document_id,  # link to the uploaded document
                filename=chunk.filename,
                page_number=chunk.page_number,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                section_title=chunk.section_title,
                section_path=chunk.section_path,
                parent_section=chunk.parent_section,
                content_type=chunk.content_type,
                chunk_index=chunk.chunk_index,
                previous_chunk_id=chunk.previous_chunk_id,
                next_chunk_id=chunk.next_chunk_id,
                token_count=chunk.token_count,
                char_count=chunk.char_count,
                text=chunk.text,
                context_prefix=chunk.context_prefix,
            )
            db_session.add(db_chunk)
        db_session.flush()  # write to DB so we can query it for BM25

        # STEP 5: Rebuild BM25 index with all chunks (including new ones)
        bm25 = get_bm25_index()
        bm25.build(db_session.query(Chunk).all())

    # STEP 6: Batch embed and upsert into Qdrant
    t3 = time.time()
    points = []
    for chunk in chunks:
        points.append((
            chunk.chunk_id,
            chunk.context_prefix + chunk.text,  # what gets embedded
            chunk.to_qdrant_payload(),  # metadata stored alongside the vector
        ))
    store_batch(points)
    timings["embed_and_store"] = time.time() - t3

    return chunks, doc_id  # return for API layer to update document status
