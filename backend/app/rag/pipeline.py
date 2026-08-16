import logging  # structured logging
import time  # timing instrumentation for performance monitoring
from dataclasses import dataclass  # simple data container for RAG response

from backend.app.config import settings  # retrieval/reranker config
from backend.app.llm.guardrails import validate_answer  # post-process LLM output

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Standardized response from the RAG pipeline."""
    answer: str  # the final answer text (or "not found" message)
    sources: list  # list of citation dicts for the frontend
    retrieved_count: int  # how many candidates were retrieved (after fusion)
    reranked_count: int  # how many chunks survived reranking


class RAGPipeline:
    """
    Orchestrates the full RAG flow:
    query -> retrieve (dense + BM25 + RRF) -> rerank -> expand context -> LLM -> answer
    """

    def __init__(self, query_processor, retriever, reranker, context_builder, llm):
        self.query_processor = query_processor  # cleans/normalizes the query
        self.retriever = retriever  # HybridRetriever (dense + BM25 + RRF)
        self.reranker = reranker  # CrossEncoder reranker
        self.context_builder = context_builder  # concatenates chunks into LLM context
        self.llm = llm  # generates the final answer

    def _expand_context(self, chunks: list, db_session) -> list:
        """
        After reranking, fetch neighboring chunks (previous + next) from PostgreSQL
        to give the LLM more surrounding context without expensive semantic search.
        """
        from backend.app.models.models import Chunk

        expanded = {c.get("chunk_id") or c.get("id"): c for c in chunks}  # index by chunk_id
        chunk_ids = list(expanded.keys())

        if not chunk_ids:
            return chunks

        # fetch the actual Chunk ORM objects for our top results
        existing_chunks = db_session.query(Chunk).filter(Chunk.chunk_id.in_(chunk_ids)).all()
        existing_map = {c.chunk_id: c for c in existing_chunks}

        # find neighboring chunk IDs (previous + next in document order)
        neighbor_ids = set()
        for cid in chunk_ids:
            chunk = existing_map.get(cid)
            if chunk:
                if chunk.previous_chunk_id:
                    neighbor_ids.add(chunk.previous_chunk_id)
                if chunk.next_chunk_id:
                    neighbor_ids.add(chunk.next_chunk_id)

        if not neighbor_ids:
            return list(expanded.values())  # no neighbors to add

        # fetch neighbor chunks from PostgreSQL
        neighbors = db_session.query(Chunk).filter(Chunk.chunk_id.in_(list(neighbor_ids))).all()

        for neighbor in neighbors:
            nid = neighbor.chunk_id
            if nid not in expanded:  # don't duplicate chunks we already have
                expanded[nid] = {
                    "chunk_id": nid,
                    "text": neighbor.text,
                    "context_prefix": neighbor.context_prefix or "",
                    "payload": {
                        "chunk_id": nid,
                        "doc_id": None,
                        "filename": neighbor.filename,
                        "page_number": neighbor.page_number,
                        "section_title": neighbor.section_title,
                        "section_path": neighbor.section_path,
                        "content_type": neighbor.content_type,
                    },
                }

        # return reranked chunks first, then their neighbors
        ordered = [expanded[cid] for cid in chunk_ids if cid in expanded]
        for cid, chunk in expanded.items():
            if cid not in chunk_ids:
                ordered.append(chunk)

        return ordered

    def answer_query(self, query: str, document_ids: list[str] | None = None) -> RAGResponse:
        """Execute the full RAG pipeline for a single user query."""
        timings = {}
        t0 = time.time()

        if not query or not query.strip():
            return RAGResponse(answer="Please provide a question.", sources=[], retrieved_count=0, reranked_count=0)

        processed_query = self.query_processor.process(query)  # clean/normalize query
        timings["query_process"] = time.time() - t0

        # RETRIEVAL: dense (Qdrant) + sparse (BM25) -> RRF fusion
        t1 = time.time()
        candidates = self.retriever.retrieve(query=processed_query, document_ids=document_ids)
        timings["retrieval"] = time.time() - t1

        if not candidates:
            logger.info("Query timings: %s", {k: f"{v:.3f}s" for k, v in timings.items()})
            return RAGResponse(
                answer="I could not find relevant information in the provided documents.",
                sources=[], retrieved_count=0, reranked_count=0
            )

        # RERANKING: CrossEncoder scores the fused candidates
        t2 = time.time()
        reranked = self.reranker.rank(query=processed_query, documents=candidates, top_k=settings.RERANK_TOP_K)
        timings["rerank"] = time.time() - t2

        if not reranked:
            logger.info("Query timings: %s", {k: f"{v:.3f}s" for k, v in timings.items()})
            return RAGResponse(
                answer="I could not find relevant information in the provided documents.",
                sources=[], retrieved_count=len(candidates), reranked_count=0
            )

        # EVIDENCE CHECK: if top rerank score is below threshold, don't bother calling the LLM
        best_score = reranked[0].get("rerank_score", 0.0)
        if best_score < settings.EVIDENCE_THRESHOLD:
            logger.info("Query timings: %s", {k: f"{v:.3f}s" for k, v in timings.items()})
            return RAGResponse(
                answer="I could not find sufficient information in the provided documents to answer this question.",
                sources=[], retrieved_count=len(candidates), reranked_count=len(reranked)
            )

        # CONTEXT EXPANSION: add previous/next chunks for richer context
        t3 = time.time()
        from backend.app.database import SessionLocal
        db = SessionLocal()
        try:
            expanded = self._expand_context(reranked, db)
        finally:
            db.close()
        timings["context_expansion"] = time.time() - t3

        # CONTEXT BUILDING: concatenate expanded chunks into a single context string
        t4 = time.time()
        context = self.context_builder.build(expanded)
        timings["context_build"] = time.time() - t4

        # LLM GENERATION: send context + question to the language model
        t5 = time.time()
        answer = self.llm.generate(query=query, context=context)
        timings["llm"] = time.time() - t5

        # validate the answer (check for empty, no context, etc.)
        answer = validate_answer(answer, context, sources=reranked)

        # log timing breakdown for performance monitoring
        timings["total"] = time.time() - t0
        logger.info("Query timings: %s", {k: f"{v:.3f}s" for k, v in timings.items()})

        sources = self._extract_sources(reranked)

        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_count=len(candidates),
            reranked_count=len(reranked)
        )

    @staticmethod
    def _extract_sources(chunks):
        """Extract citation metadata from the top reranked chunks."""
        sources = []
        for chunk in chunks:
            metadata = chunk.get("payload", {})
            sources.append({
                "document_id": metadata.get("doc_id"),
                "page": metadata.get("page_number"),
                "section": metadata.get("section_title"),
                "chunk_id": metadata.get("chunk_id"),
                "retrieval_score": chunk.get("score"),  # dense or BM25 score
                "rerank_score": chunk.get("rerank_score"),  # CrossEncoder score
            })
        return sources
